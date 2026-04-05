import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ml_utils import (
    FEATURE_NAMES,
    apply_standardizer,
    events_to_grouped_features,
    fit_standardizer,
    load_events,
    save_standardizer,
)


def jitter_augmentation_sample(X_sample, jitter_sigma=0.01):
    """
    Add Gaussian noise to a single window to increase training variance.
    Used within CV training folds only (never on test data).
    
    Args:
        X_sample: (window_size, feature_dim) array
        jitter_sigma: std dev of noise
    
    Returns:
        Augmented (window_size, feature_dim) array
    """
    if X_sample.shape[1] < 4:
        return X_sample
    
    X_aug = X_sample.copy()
    noise = np.random.normal(0, jitter_sigma, size=(X_sample.shape[0], 4))
    X_aug[:, :4] = np.clip(X_aug[:, :4] + noise, -5, 5)
    return X_aug


def augment_training_fold(X_train, y_train, augmentation_type="jitter", multiplier=1):
    """
    Apply augmentation ONLY to training fold inside CV loop.
    CRITICAL: Never augment before CV split (causes test leakage).
    
    Args:
        X_train: Training sequences (N, window_size, feature_dim)
        y_train: Training labels (N,)
        augmentation_type: "jitter", "none", or others
        multiplier: How many augmented copies per original (1=no augmentation)
    
    Returns:
        (X_augmented, y_augmented)
    """
    if multiplier <= 1 or augmentation_type == "none":
        return X_train, y_train
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    for m in range(1, multiplier):
        X_aug = np.zeros_like(X_train)
        for i in range(len(X_train)):
            if augmentation_type == "jitter":
                X_aug[i] = jitter_augmentation_sample(X_train[i])
            else:
                X_aug[i] = X_train[i]
        X_aug_list.append(X_aug)
        y_aug_list.append(y_train)
    
    return np.vstack(X_aug_list), np.hstack(y_aug_list)


def build_sequences(grouped_data, window_size: int, min_events: int):
    seq_X, seq_y = [], []
    seq_end_ts, seq_session = [], []

    for session_id, data in grouped_data.items():
        X = data["X"]
        y = data["y"]
        ts = data["ts"]

        if len(X) < max(window_size, min_events):
            continue

        session_seq_X = []
        session_seq_y = []
        session_seq_end_ts = []
        session_seq_session = []
        window_attack_counts = []
        window_attack_ratios = []

        for end in range(window_size, len(X) + 1):
            start = end - window_size
            window_X = X[start:end]
            window_y = y[start:end]

            # Sequence is anomalous when suspicious events cluster in a window.
            attack_ratio = float(window_y.mean())
            recent_hits = int(window_y[-5:].sum()) if len(window_y) >= 5 else int(window_y.sum())
            label = 1 if attack_ratio >= 0.35 or recent_hits >= 3 else 0

            session_seq_X.append(window_X)
            session_seq_y.append(label)
            session_seq_end_ts.append(int(ts[end - 1]))
            session_seq_session.append(session_id)
            window_attack_counts.append(int(window_y.sum()))
            window_attack_ratios.append(attack_ratio)

        # If a session has attack events but no positive windows, tag a few highest-attack windows.
        if int(y.sum()) > 0 and sum(session_seq_y) == 0:
            candidates = [i for i, count in enumerate(window_attack_counts) if count > 0]
            if candidates:
                ranked = sorted(
                    candidates,
                    key=lambda i: (window_attack_ratios[i], window_attack_counts[i]),
                    reverse=True,
                )
                for idx in ranked[:1]:
                    session_seq_y[idx] = 1

        seq_X.extend(session_seq_X)
        seq_y.extend(session_seq_y)
        seq_end_ts.extend(session_seq_end_ts)
        seq_session.extend(session_seq_session)

    if not seq_X:
        return (
            np.empty((0, window_size, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="<U1"),
        )

    return (
        np.array(seq_X, dtype=np.float32),
        np.array(seq_y, dtype=np.int64),
        np.array(seq_end_ts, dtype=np.int64),
        np.array(seq_session),
    )


def main():
    parser = argparse.ArgumentParser(description="Build LSTM-ready sequences from honeypot logs")
    parser.add_argument("--input", default="logs/raw_events.jsonl", help="Path to raw JSONL events")
    parser.add_argument("--window-size", type=int, default=20, help="Number of events per sequence")
    parser.add_argument("--min-events", type=int, default=25, help="Minimum session events to keep")
    parser.add_argument("--output", default="data/sequences_large.npz", help="Output NPZ dataset path")
    parser.add_argument("--scaler", default="models/scaler_large.json", help="Output scaler metadata path")

    args = parser.parse_args()

    events = load_events(Path(args.input))
    grouped = events_to_grouped_features(events)
    total_sessions = len(grouped)
    session_lengths = {sid: len(data["X"]) for sid, data in grouped.items()}
    events_total = int(sum(session_lengths.values()))
    min_required = max(args.window_size, args.min_events)
    eligible_sessions = int(sum(1 for count in session_lengths.values() if count >= min_required))
    dropped_sessions = int(total_sessions - eligible_sessions)
    drop_reasons = Counter()
    for count in session_lengths.values():
        if count < min_required:
            drop_reasons["too_few_events"] += 1

    X, y, end_ts, session_ids = build_sequences(grouped, window_size=args.window_size, min_events=args.min_events)

    if len(X) == 0:
        raise SystemExit("No sequences generated. Collect more logs or lower --min-events/--window-size.")

    mean, std = fit_standardizer(X)
    X_scaled = apply_standardizer(X, mean, std)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X_scaled,
        y=y,
        end_ts=end_ts,
        session_ids=session_ids,
        feature_names=np.array(FEATURE_NAMES),
        window_size=np.array([args.window_size], dtype=np.int64),
    )

    save_standardizer(Path(args.scaler), mean, std)

    pos_count = int(y.sum())
    pos_ratio = pos_count / len(y) if len(y) > 0 else 0.0
    print(f"Saved dataset: {output_path}")
    print(f"  sequences={len(X_scaled)}")
    print(f"  positives={pos_count}")
    print(f"  positive_ratio={pos_ratio:.3f}")
    print(f"  events_total={events_total}")
    print(f"  sessions_total={total_sessions}")
    print(f"  sessions_used={eligible_sessions}")
    print(f"  sessions_dropped={dropped_sessions}")
    if drop_reasons:
        print(f"  drop_reasons={dict(drop_reasons)}")
    print("  [Note: Augmentation and class balancing applied during CV training folds, not here]")


if __name__ == "__main__":
    main()
