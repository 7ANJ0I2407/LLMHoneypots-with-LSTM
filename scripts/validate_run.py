import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_lstm import LSTMDetector


def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def metrics_from_counts(tp, tn, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMDetector(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_block(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(np.int64)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, tn, fp, fn)
    return y_pred, (tp, tn, fp, fn), metrics


def main():
    parser = argparse.ArgumentParser(description="Validate trained model outputs on current dataset")
    parser.add_argument("--dataset", default="data/sequences.npz")
    parser.add_argument("--model", default="models/lstm_detector.pt")
    parser.add_argument("--meta", default="models/model_meta.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    meta_path = Path(args.meta)

    if not dataset_path.exists() or not model_path.exists() or not meta_path.exists():
        raise SystemExit("Missing required files. Run demo:e2e or pipeline first.")

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    end_ts = data["end_ts"].astype(np.int64) if "end_ts" in data.files else None

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    threshold = float(meta.get("threshold", 0.5))

    model = load_model(model_path)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()

    y_pred, (tp, tn, fp, fn), metrics = evaluate_block(y, probs, threshold)

    positive_ratio = float(y.mean()) if len(y) else 0.0
    score_mean = float(probs.mean()) if len(probs) else 0.0

    checks = [
        ("dataset_nontrivial", len(y) >= 100, f"samples={len(y)} expected>=100"),
        ("contains_both_classes", int(y.min()) == 0 and int(y.max()) == 1, "expected both 0 and 1 labels"),
        ("prediction_not_constant", int(y_pred.min()) == 0 and int(y_pred.max()) == 1, "expected both 0 and 1 predictions"),
        ("reasonable_precision", metrics["precision"] >= 0.75, f"precision={metrics['precision']:.3f} expected>=0.75"),
        ("reasonable_recall", metrics["recall"] >= 0.75, f"recall={metrics['recall']:.3f} expected>=0.75"),
    ]

    time_holdout_line = None
    if end_ts is not None and len(end_ts) == len(y):
        idx = np.argsort(end_ts)
        holdout_start = int(len(idx) * 0.8)
        holdout_idx = idx[holdout_start:]

        y_h = y[holdout_idx]
        p_h = probs[holdout_idx]
        _pred_h, (_tp_h, _tn_h, _fp_h, _fn_h), m_h = evaluate_block(y_h, p_h, threshold)

        time_holdout_line = {
            "samples": int(len(y_h)),
            "accuracy": m_h["accuracy"],
            "precision": m_h["precision"],
            "recall": m_h["recall"],
            "f1": m_h["f1"],
        }
        checks.append(
            (
                "time_holdout_f1",
                m_h["f1"] >= 0.7,
                f"time_holdout_f1={m_h['f1']:.3f} expected>=0.70",
            )
        )

    all_pass = all(c[1] for c in checks)

    print("=== VALIDATION SUMMARY ===")
    print(f"samples={len(y)}")
    print(f"positive_ratio={positive_ratio:.4f}")
    print(f"threshold={threshold}")
    print(f"score_mean={score_mean:.4f}")
    print(f"confusion_tp={tp} tn={tn} fp={fp} fn={fn}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")

    if time_holdout_line is not None:
        print(
            "time_holdout="
            f"samples={time_holdout_line['samples']} "
            f"acc={time_holdout_line['accuracy']:.4f} "
            f"precision={time_holdout_line['precision']:.4f} "
            f"recall={time_holdout_line['recall']:.4f} "
            f"f1={time_holdout_line['f1']:.4f}"
        )

    for name, passed, detail in checks:
        print(f"check_{name}={'PASS' if passed else 'FAIL'} ({detail})")

    print(f"validation_gate={'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
