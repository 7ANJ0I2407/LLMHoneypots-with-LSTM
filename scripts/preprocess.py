import argparse
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


def build_sequences(grouped_data, window_size: int, min_events: int):
    seq_X, seq_y = [], []
    seq_end_ts, seq_session = [], []

    for session_id, data in grouped_data.items():
        X = data["X"]
        y = data["y"]
        ts = data["ts"]

        if len(X) < max(window_size, min_events):
            continue

        for end in range(window_size, len(X) + 1):
            start = end - window_size
            window_X = X[start:end]
            window_y = y[start:end]

            # Sequence is anomalous when suspicious events cluster in a window.
            attack_ratio = float(window_y.mean())
            recent_hits = int(window_y[-5:].sum())
            label = 1 if attack_ratio >= 0.35 or recent_hits >= 3 else 0

            seq_X.append(window_X)
            seq_y.append(label)
            seq_end_ts.append(int(ts[end - 1]))
            seq_session.append(session_id)

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
    parser.add_argument("--output", default="data/sequences.npz", help="Output NPZ dataset path")
    parser.add_argument("--scaler", default="models/scaler.json", help="Output scaler metadata path")

    args = parser.parse_args()

    events = load_events(Path(args.input))
    grouped = events_to_grouped_features(events)

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

    print(f"Saved dataset: {output_path} | sequences={len(X_scaled)} | positives={int(y.sum())}")


if __name__ == "__main__":
    main()
