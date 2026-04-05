import argparse
import json
from pathlib import Path

import numpy as np
import torch

from autoencoder_model import SequenceAutoencoder, reconstruction_error
from train_lstm import LSTMDetector


def load_lstm(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMDetector(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_novelty(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = SequenceAutoencoder(
        window_size=int(checkpoint["window_size"]),
        input_dim=int(checkpoint["input_dim"]),
        proj_dim=int(checkpoint.get("proj_dim", 48)),
        latent_dim=int(checkpoint.get("latent_dim", 24)),
        dropout=float(checkpoint.get("dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(y_true), 1)
    benign_fpr = fp / max(fp + tn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "benign_fpr": float(benign_fpr),
    }


def predict_scores(lstm, novelty, X: np.ndarray, batch_size: int, temperature_scale: float = 1.0):
    probs = np.empty((len(X),), dtype=np.float32)
    errs = np.empty((len(X),), dtype=np.float32)
    temp = float(temperature_scale)
    if temp <= 1e-6:
        temp = 1.0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = torch.from_numpy(X[start:end])
            logits = lstm(batch)
            probs[start:end] = torch.sigmoid(logits / temp).numpy()
            recon = novelty(batch)
            errs[start:end] = reconstruction_error(batch, recon).numpy()
    return probs, errs


def split_time_indices(end_ts: np.ndarray, tune_frac: float, test_frac: float):
    order = np.argsort(end_ts)
    n = len(order)
    test_n = max(1, int(n * test_frac))
    tune_n = max(1, int(n * tune_frac))
    if tune_n + test_n >= n:
        tune_n = max(1, int(n * 0.1))
        test_n = max(1, int(n * 0.2))

    train_end = n - (tune_n + test_n)
    tune_end = n - test_n
    train_idx = order[:train_end]
    tune_idx = order[train_end:tune_end]
    test_idx = order[tune_end:]
    return train_idx, tune_idx, test_idx


def tune_blend_threshold(scores: np.ndarray, y: np.ndarray, max_benign_fpr: float | None):
    best = None
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (scores >= thr).astype(np.int64)
        m = metrics(y, pred)
        if max_benign_fpr is not None and m["benign_fpr"] > max_benign_fpr:
            continue
        if best is None or m["f1"] > best["f1"]:
            best = {"threshold": float(thr), **m}
    return best


def main():
    parser = argparse.ArgumentParser(description="Compare OR vs Blend on time-holdout only")
    parser.add_argument("--dataset", default="data/sequences_large.npz")
    parser.add_argument("--model", default="models/lstm_detector_large.pt")
    parser.add_argument("--novelty-model", default="models/novelty_autoencoder_large.pt")
    parser.add_argument("--meta", default="models/model_meta_large.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--tune-frac", type=float, default=0.2)
    parser.add_argument("--tune-blend-threshold", action="store_true")
    parser.add_argument("--max-benign-fpr", type=float, default=None)
    parser.add_argument("--output", default="models/fusion_holdout_report_large.json")
    parser.add_argument("--include-or", action="store_true", help="Include OR-mode metrics in the report")
    parser.add_argument("--drop-or-if-benign-fpr", type=float, default=0.2)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    novelty_model_path = Path(args.novelty_model)
    meta_path = Path(args.meta)

    if not dataset_path.exists() or not model_path.exists() or not novelty_model_path.exists() or not meta_path.exists():
        raise SystemExit("Missing dataset/model/meta for holdout comparison.")

    data = np.load(dataset_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if "end_ts" not in data.files:
        raise SystemExit("Dataset has no end_ts. Time-holdout comparison requires end_ts.")
    end_ts = np.asarray(data["end_ts"], dtype=np.int64)

    meta = json.loads(meta_path.read_text(encoding="utf8"))
    runtime_threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.5)))
    runtime_novelty_threshold = float(meta.get("runtime_novelty_threshold", meta.get("novelty_threshold", 0.0)))
    temperature_scale = float(meta.get("temperature_scale", 1.0) or 1.0)
    if temperature_scale <= 1e-6:
        temperature_scale = 1.0
    fusion = meta.get("runtime_fusion") or {}
    alpha = float(fusion.get("alpha", 0.5))
    blend_threshold = fusion.get("blend_threshold", None)
    if blend_threshold is None:
        blend_threshold = 0.5
    blend_threshold = float(blend_threshold)

    _, tune_idx, test_idx = split_time_indices(end_ts, tune_frac=args.tune_frac, test_frac=args.test_frac)

    lstm = load_lstm(model_path)
    novelty = load_novelty(novelty_model_path)

    probs, errs = predict_scores(lstm, novelty, X, batch_size=args.batch_size, temperature_scale=temperature_scale)
    novelty_signal = np.clip(errs / max(runtime_novelty_threshold, 1e-8), 0.0, 1.0)
    blend_scores = (alpha * probs) + ((1.0 - alpha) * novelty_signal)

    if args.tune_blend_threshold:
        tuned = tune_blend_threshold(blend_scores[tune_idx], y[tune_idx], args.max_benign_fpr)
        if tuned is None:
            raise SystemExit("No feasible blend threshold under supplied benign FPR constraint.")
        blend_threshold = float(tuned["threshold"])
    else:
        tuned = None

    pred_blend = (blend_scores[test_idx] >= blend_threshold).astype(np.int64)

    m_or = None
    or_deprecated = None
    if args.include_or:
        pred_or = ((probs[test_idx] >= runtime_threshold) | (errs[test_idx] >= runtime_novelty_threshold)).astype(np.int64)
        m_or = metrics(y[test_idx], pred_or)
        if args.drop_or_if_benign_fpr is not None and m_or["benign_fpr"] > float(args.drop_or_if_benign_fpr):
            or_deprecated = {
                "reason": "benign_fpr_exceeds_limit",
                "limit": float(args.drop_or_if_benign_fpr),
                "benign_fpr": float(m_or["benign_fpr"]),
            }
            m_or = None
    m_blend = metrics(y[test_idx], pred_blend)

    report = {
        "dataset": str(dataset_path),
        "samples_total": int(len(y)),
        "samples_test": int(len(test_idx)),
        "thresholds": {
            "runtime_threshold": runtime_threshold,
            "runtime_novelty_threshold": runtime_novelty_threshold,
            "blend_alpha": alpha,
            "blend_threshold_used": blend_threshold,
        },
        "tuning": {
            "enabled": bool(args.tune_blend_threshold),
            "tune_samples": int(len(tune_idx)),
            "max_benign_fpr": args.max_benign_fpr,
            "best_tune_metrics": tuned,
        },
        "or_mode": m_or,
        "or_mode_deprecated": or_deprecated,
        "blend_mode": m_blend,
    }

    if m_or is not None:
        report["delta_blend_minus_or"] = {
            k: float(m_blend[k] - m_or[k]) for k in ["accuracy", "precision", "recall", "f1", "benign_fpr"]
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf8")

    print("=== HOLDOUT FUSION COMPARISON ===")
    print(f"dataset={dataset_path} total_samples={len(y)} test_samples={len(test_idx)}")
    print(
        "thresholds="
        f"lstm:{runtime_threshold:.6f} novelty:{runtime_novelty_threshold:.6f} alpha:{alpha:.3f} blend:{blend_threshold:.6f}"
    )
    if tuned is not None:
        print(
            "blend_tuned_on_validation="
            f"threshold:{tuned['threshold']:.6f} f1:{tuned['f1']:.6f} benign_fpr:{tuned['benign_fpr']:.6f}"
        )

    if m_or is not None:
        print(
            "or_mode="
            f"f1:{m_or['f1']:.6f} recall:{m_or['recall']:.6f} precision:{m_or['precision']:.6f} benign_fpr:{m_or['benign_fpr']:.6f}"
        )
    elif or_deprecated is not None:
        print(
            "or_mode=DEPRECATED "
            f"benign_fpr:{or_deprecated['benign_fpr']:.6f} limit:{or_deprecated['limit']:.3f}"
        )
    print(
        "blend_mode="
        f"f1:{m_blend['f1']:.6f} recall:{m_blend['recall']:.6f} precision:{m_blend['precision']:.6f} benign_fpr:{m_blend['benign_fpr']:.6f}"
    )
    print(f"report_saved={out_path}")


if __name__ == "__main__":
    main()
