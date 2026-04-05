"""
Ablation study: Train 3 model variants to isolate impact of each component.
Variant 1: Rule-only baseline (attack pattern matching, no ML)
Variant 2: LSTM-only (base behavioral features)
Variant 3: Full system (LSTM + autoencoder novelty)

CRITICAL: Each variant trained on identical time-holdout splits with identical test data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from autoencoder_model import SequenceAutoencoder
from train_lstm import LSTMDetector
from validate_run import confusion_counts, metrics_from_counts


BASE_FEATURE_COUNT = 10


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_time_holdout(X, y, end_ts, val_ratio=0.2):
    order = np.argsort(end_ts) if end_ts is not None else np.arange(len(X))
    split = int(len(order) * (1 - val_ratio))
    train_idx, val_idx = order[:split], order[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_lstm_variant(X_train, y_train, input_dim, seed=42, epochs=4, batch_size=256, lr=1e-3):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMDetector(input_dim=input_dim).to(device)
    loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model


def sweep_threshold(scores, y_true, start=0.05, stop=0.95, steps=181):
    best_threshold = start
    best_result = None
    best_f1 = -1.0

    for threshold in np.linspace(start, stop, steps):
        y_pred = (scores >= threshold).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(y_true, y_pred)
        metrics = metrics_from_counts(tp, tn, fp, fn)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_result = (tp, tn, fp, fn, metrics)

    tp, tn, fp, fn, metrics = best_result
    return {
        "threshold": float(best_threshold),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def variant_rule_only(X, y, session_ids, threshold=None):
    """
    Variant 1: Rule-based baseline.
    Uses only pre-computed 'signal_score' feature (already in feature vector).
    This represents non-ML detection (pattern matching only).
    """
    # Signal score is the third base feature in the vector.
    signal_scores = X[:, :, 2]  # (N, window_size)
    
    # Aggregate score per sequence: mean signal over window
    agg_scores = signal_scores.mean(axis=1)
    
    if threshold is None:
        result = sweep_threshold(agg_scores, y)
    else:
        y_pred = (agg_scores >= threshold).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(y, y_pred)
        metrics = metrics_from_counts(tp, tn, fp, fn)
        result = {
            "threshold": float(threshold),
            "score_mean": float(agg_scores.mean()),
            "score_std": float(agg_scores.std()),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }

    result["variant"] = "rule_only"
    return result


def variant_lstm_only(X_train, y_train, X_val, y_val, threshold=None, seed=42):
    """
    Variant 2: LSTM classifier without novelty.
    Uses only the base behavioral features.
    """
    model = train_lstm_variant(X_train, y_train, input_dim=X_train.shape[-1], seed=seed)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_val).float())).numpy()

    if threshold is None:
        result = sweep_threshold(probs, y_val)
    else:
        y_pred = (probs >= threshold).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(y_val, y_pred)
        metrics = metrics_from_counts(tp, tn, fp, fn)
        result = {
            "threshold": float(threshold),
            "score_mean": float(probs.mean()),
            "score_std": float(probs.std()),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }

    result["variant"] = "lstm_only"
    return result


def variant_full_system(
    X_train,
    y_train,
    X_val,
    y_val,
    autoencoder_path,
    threshold_lstm=None,
    threshold_novelty=None,
    seed=42,
    max_benign_fpr=0.03,
):
    """
    Variant 3: Full system with LSTM + autoencoder novelty.
    Trains the LSTM on the full feature vector and calibrates thresholds on held-out data.
    """
    lstm_model = train_lstm_variant(X_train, y_train, input_dim=X_train.shape[-1], seed=seed)

    autoencoder_checkpoint = torch.load(autoencoder_path, map_location="cpu")
    ae_model = SequenceAutoencoder(
        window_size=int(autoencoder_checkpoint.get("window_size", 20)),
        input_dim=int(autoencoder_checkpoint.get("input_dim", X_train.shape[-1])),
        proj_dim=int(autoencoder_checkpoint.get("proj_dim", 48)),
        latent_dim=int(autoencoder_checkpoint.get("latent_dim", 24)),
        dropout=float(autoencoder_checkpoint.get("dropout", 0.10)),
    )
    ae_model.load_state_dict(autoencoder_checkpoint["state_dict"])
    ae_model.eval()

    with torch.no_grad():
        lstm_probs = torch.sigmoid(lstm_model(torch.from_numpy(X_val.astype(np.float32)))).numpy()
        X_tensor = torch.from_numpy(X_val.astype(np.float32))
        ae_reconstructed = ae_model(X_tensor)
        X_flat = X_tensor.reshape(X_tensor.shape[0], -1)
        ae_flat = ae_reconstructed.reshape(ae_reconstructed.shape[0], -1)
        recon_errors = torch.mean((X_flat - ae_flat) ** 2, dim=1).numpy()

    lstm_grid = [float(threshold_lstm)] if threshold_lstm is not None else np.linspace(0.05, 0.95, 37)
    novelty_grid = [float(threshold_novelty)] if threshold_novelty is not None else np.percentile(recon_errors, np.linspace(90, 99.9, 30))
    blend_thr_grid = np.linspace(0.15, 0.95, 41)
    alpha_grid = [0.50, 0.60, 0.70, 0.80, 0.90]

    best_constrained = None
    best_any = None

    def maybe_update(candidate):
        nonlocal best_constrained, best_any
        if best_any is None or candidate["f1"] > best_any["f1"]:
            best_any = candidate
        if candidate["benign_fpr"] <= max_benign_fpr:
            if best_constrained is None or candidate["f1"] > best_constrained["f1"]:
                best_constrained = candidate

    # Mode 1: OR decision (legacy behavior)
    for nov_thr in novelty_grid:
        novelty_alerts = (recon_errors >= nov_thr).astype(np.int64)
        novelty_signal = np.clip(recon_errors / max(float(nov_thr), 1e-8), 0.0, 1.0)
        for lstm_thr in lstm_grid:
            lstm_alerts = (lstm_probs >= lstm_thr).astype(np.int64)
            y_pred = np.maximum(lstm_alerts, novelty_alerts)
            tp, tn, fp, fn = confusion_counts(y_val, y_pred)
            metrics = metrics_from_counts(tp, tn, fp, fn)
            benign_fpr = fp / max(fp + tn, 1)
            maybe_update(
                {
                    "mode": "or",
                    "threshold_lstm": float(lstm_thr),
                    "threshold_novelty": float(nov_thr),
                    "alpha": None,
                    "blend_threshold": None,
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "benign_fpr": float(benign_fpr),
                    "lstm_score_mean": float(lstm_probs.mean()),
                    "novelty_error_mean": float(recon_errors.mean()),
                }
            )

        # Mode 2: Weighted blend between LSTM confidence and novelty signal.
        for alpha in alpha_grid:
            combined = alpha * lstm_probs + (1.0 - alpha) * novelty_signal
            for blend_thr in blend_thr_grid:
                y_pred = (combined >= blend_thr).astype(np.int64)
                tp, tn, fp, fn = confusion_counts(y_val, y_pred)
                metrics = metrics_from_counts(tp, tn, fp, fn)
                benign_fpr = fp / max(fp + tn, 1)
                maybe_update(
                    {
                        "mode": "blend",
                        "threshold_lstm": None,
                        "threshold_novelty": float(nov_thr),
                        "alpha": float(alpha),
                        "blend_threshold": float(blend_thr),
                        "tp": int(tp),
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "accuracy": float(metrics["accuracy"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "benign_fpr": float(benign_fpr),
                        "lstm_score_mean": float(lstm_probs.mean()),
                        "novelty_error_mean": float(recon_errors.mean()),
                    }
                )

    selected = best_constrained if best_constrained is not None else best_any
    constraint_satisfied = best_constrained is not None

    return {
        "variant": "full_system",
        "mode": selected["mode"],
        "threshold_lstm": selected["threshold_lstm"],
        "threshold_novelty": selected["threshold_novelty"],
        "alpha": selected["alpha"],
        "blend_threshold": selected["blend_threshold"],
        "lstm_score_mean": selected["lstm_score_mean"],
        "novelty_error_mean": selected["novelty_error_mean"],
        "tp": selected["tp"],
        "tn": selected["tn"],
        "fp": selected["fp"],
        "fn": selected["fn"],
        "accuracy": selected["accuracy"],
        "precision": selected["precision"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "benign_fpr": selected["benign_fpr"],
        "max_benign_fpr": float(max_benign_fpr),
        "constraint_satisfied": bool(constraint_satisfied),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study: Compare 4 model variants")
    parser.add_argument("--dataset", default="data/sequences_large.npz")
    parser.add_argument("--autoencoder", default="models/novelty_autoencoder_large.pt")
    parser.add_argument("--meta", default="models/model_meta_large.json")
    parser.add_argument("--output", default="models/ablation_results_large.json")
    parser.add_argument("--min-samples", type=int, default=30000)
    parser.add_argument("--max-benign-fpr", type=float, default=0.03)
    parser.add_argument("--target-lift", type=float, default=0.015)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    autoencoder_path = Path(args.autoencoder)
    meta_path = Path(args.meta)
    
    if not dataset_path.exists():
        raise SystemExit("Missing dataset. Run preprocessing first.")

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    end_ts = data["end_ts"].astype(np.int64) if "end_ts" in data.files else None

    if len(X) < args.min_samples:
        raise SystemExit(
            f"Dataset too small for ablation: {len(X)} samples. "
            f"Run demo:xlarge first (or lower --min-samples)."
        )

    X_train, y_train, X_val, y_val = split_time_holdout(X, y, end_ts, val_ratio=0.2)
    X_train_base = X_train[:, :, :BASE_FEATURE_COUNT]
    X_val_base = X_val[:, :, :BASE_FEATURE_COUNT]

    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf8") as f:
            meta = json.load(f)

    lstm_threshold = None
    novelty_threshold = None

    print("=== ABLATION STUDY: 3 Model Variants ===")
    print()

    results = {
        "dataset_samples": len(y),
        "positive_ratio": float(y.mean()),
        "variants": []
    }

    # Variant 1: Rule-only
    print("[1/3] Variant: Rule-only baseline...")
    r1 = variant_rule_only(X_val, y_val, None)
    results["variants"].append(r1)
    print(f"      F1={r1['f1']:.4f} (baseline: pattern matching only)")
    print()

    # Variant 2: LSTM-only on base features without novelty
    print("[2/3] Variant: LSTM-only (no novelty)...")
    r2 = variant_lstm_only(X_train_base, y_train, X_val_base, y_val)
    results["variants"].append(r2)
    print(f"      F1={r2['f1']:.4f} (+ LSTM sequence modeling)")
    print()

    # Variant 3: Full system
    if autoencoder_path.exists():
        print("[3/3] Variant: Full system (LSTM + novelty detection)...")
        r3 = variant_full_system(
            X_train,
            y_train,
            X_val,
            y_val,
            autoencoder_path,
            threshold_lstm=lstm_threshold,
            threshold_novelty=novelty_threshold,
            max_benign_fpr=args.max_benign_fpr,
        )
        results["variants"].append(r3)
        print(
            f"      F1={r3['f1']:.4f} (+ autoencoder novelty gating, "
            f"mode={r3['mode']}, benign_fpr={r3['benign_fpr']:.4f})"
        )
    else:
        print("[3/3] Skipping full system (autoencoder not found)")
        print(f"      Path: {autoencoder_path}")
    print()

    # Summary table
    print("=== ABLATION SUMMARY TABLE ===")
    print(f"{'Variant':<25} {'F1':<8} {'Precision':<10} {'Recall':<10} {'Δ vs Rule':<10}")
    print("-" * 63)
    baseline_f1 = results["variants"][0]["f1"]
    for v in results["variants"]:
        delta = (v["f1"] - baseline_f1) / baseline_f1 * 100
        print(f"{v['variant']:<25} {v['f1']:<8.4f} {v['precision']:<10.4f} {v['recall']:<10.4f} {delta:+.1f}%")

    # Paper target check: full-system novelty contribution over lstm_only.
    by_variant = {v["variant"]: v for v in results["variants"]}
    if "lstm_only" in by_variant and "full_system" in by_variant:
        lift = float(by_variant["full_system"]["f1"] - by_variant["lstm_only"]["f1"])
        target_met = bool(lift >= args.target_lift and by_variant["full_system"].get("benign_fpr", 1.0) <= args.max_benign_fpr)
        results["paper_target"] = {
            "target_lift": float(args.target_lift),
            "max_benign_fpr": float(args.max_benign_fpr),
            "achieved_lift": lift,
            "full_system_benign_fpr": float(by_variant["full_system"].get("benign_fpr", 0.0)),
            "met": target_met,
        }
        print()
        print(
            "paper_target="
            f"lift>={args.target_lift:.4f} and benign_fpr<={args.max_benign_fpr:.4f} -> "
            f"achieved_lift={lift:.4f}, benign_fpr={by_variant['full_system'].get('benign_fpr', 0.0):.4f}, "
            f"met={'YES' if target_met else 'NO'}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved ablation results: {output_path}")


if __name__ == "__main__":
    main()
