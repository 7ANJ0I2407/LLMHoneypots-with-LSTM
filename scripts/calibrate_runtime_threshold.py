import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ml_utils import apply_standardizer, events_to_grouped_features, load_events, load_standardizer
from train_lstm import LSTMDetector


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMDetector(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, int(checkpoint["window_size"])


def weak_sequence_label(window_y: np.ndarray):
    attack_ratio = float(window_y.mean())
    recent_hits = int(window_y[-5:].sum()) if len(window_y) >= 5 else int(window_y.sum())
    return 1 if attack_ratio >= 0.35 or recent_hits >= 3 else 0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1, precision, recall


def main():
    parser = argparse.ArgumentParser(description="Calibrate runtime LSTM threshold from recent logs")
    parser.add_argument("--input", default="logs/raw_events.jsonl")
    parser.add_argument("--meta", default="models/model_meta.json")
    parser.add_argument("--model", default="models/lstm_detector.pt")
    parser.add_argument("--scaler", default="models/scaler.json")
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--min-threshold", type=float, default=0.10)

    args = parser.parse_args()

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    meta_path = Path(args.meta)

    if not model_path.exists() or not scaler_path.exists() or not meta_path.exists():
        raise SystemExit("Missing model/scaler/meta files. Run training first.")

    events = load_events(Path(args.input))
    grouped = events_to_grouped_features(events)

    mean, std, _ = load_standardizer(scaler_path)
    model, window_size = load_model(model_path)

    windows = []
    labels = []
    end_ts = []

    for _sid, data in grouped.items():
        X = data["X"]
        y = data["y"]
        ts = data.get("ts")

        if len(X) < window_size:
            continue

        for end in range(window_size, len(X) + 1):
            win_X = X[end - window_size : end]
            win_y = y[end - window_size : end]
            label = weak_sequence_label(win_y)
            windows.append(win_X)
            labels.append(label)
            end_ts.append(int(ts[end - 1]) if ts is not None else end)

    if not windows:
        raise SystemExit("No windows available for calibration.")

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    t = np.array(end_ts, dtype=np.int64)

    idx = np.argsort(t)
    X = X[idx]
    y = y[idx]

    if len(X) > args.max_windows:
        X = X[-args.max_windows :]
        y = y[-args.max_windows :]

    Xs = apply_standardizer(X, mean, std)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(Xs))).numpy().astype(np.float32)

    best = {
        "threshold": 0.1,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for thr in np.linspace(0.02, 0.5, 97):
        pred = (probs >= thr).astype(np.int64)
        f1, precision, recall = f1_score(y, pred)
        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    calibrated_threshold = max(float(best["threshold"]), float(args.min_threshold))

    meta = json.loads(meta_path.read_text(encoding="utf8"))
    meta["runtime_threshold"] = calibrated_threshold
    meta["runtime_calibration"] = {
        "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "windows": int(len(X)),
        "positive_windows": int(y.sum()),
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "raw_best_threshold": best["threshold"],
        "min_threshold": float(args.min_threshold),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")

    print(
        "runtime_threshold_calibrated="
        f"{calibrated_threshold:.4f} windows={len(X)} positives={int(y.sum())} "
        f"f1={best['f1']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
