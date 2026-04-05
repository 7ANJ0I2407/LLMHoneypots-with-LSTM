import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.model_selection import GroupKFold  # type: ignore
except Exception:  # pragma: no cover
    GroupKFold = None

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


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_block(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(np.int64)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, tn, fp, fn)
    return y_pred, (tp, tn, fp, fn), metrics


def predict_probs_batched(model, X, batch_size=256, temperature_scale=1.0):
    probs = np.empty((len(X),), dtype=np.float32)
    temp = float(temperature_scale)
    if temp <= 1e-6:
        temp = 1.0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = torch.from_numpy(X[start:end])
            logits = model(batch)
            probs[start:end] = torch.sigmoid(logits / temp).numpy()
    return probs


class SequenceDataset(Dataset):
    def __init__(self, X, y, indices=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        if indices is None:
            self.indices = np.arange(len(self.X), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32)


def train_fold_model(
    X_train,
    y_train,
    epochs=4,
    batch_size=256,
    lr=1e-3,
    seed=42,
    dropout=0.35,
    label_smoothing=0.0,
    weight_decay=0.0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMDetector(input_dim=X_train.shape[-1], dropout=dropout).to(device)
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            if label_smoothing > 0:
                batch_y = batch_y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def best_threshold_from_probs(y_true, probs, steps=201):
    curve = compute_curve(y_true, probs, steps=steps)
    best = best_f1_point(curve)
    if best is None:
        return 0.5
    return float(best["threshold"])


def compute_curve(y_true, probs, steps=201):
    points = []
    thresholds = np.linspace(0.0, 1.0, steps)
    for thr in thresholds:
        y_pred = (probs >= thr).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(y_true, y_pred)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        points.append({
            "threshold": float(thr),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    return points


def best_f1_point(points):
    if not points:
        return None
    return max(points, key=lambda p: p.get("f1", 0.0))


def area_under_curve(points, x_key, y_key):
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda p: p[x_key])
    xs = np.array([p[x_key] for p in ordered], dtype=np.float64)
    ys = np.array([p[y_key] for p in ordered], dtype=np.float64)
    if hasattr(np, "trapz"):
        area = np.trapz(ys, xs)
    else:
        area = np.trapezoid(ys, xs)
    return float(area)


def stratified_time_holdout_idx(y: np.ndarray, end_ts: np.ndarray, holdout_ratio: float = 0.2):
    y = np.asarray(y, dtype=np.int64)
    end_ts = np.asarray(end_ts, dtype=np.int64)

    holdout_parts = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) == 0:
            continue
        cls_sorted = cls_idx[np.argsort(end_ts[cls_idx])]
        holdout_n = max(1, int(round(len(cls_sorted) * holdout_ratio)))
        holdout_n = min(holdout_n, len(cls_sorted))
        holdout_parts.append(cls_sorted[-holdout_n:])

    if not holdout_parts:
        return np.empty((0,), dtype=np.int64)

    holdout_idx = np.concatenate(holdout_parts).astype(np.int64)
    holdout_idx = holdout_idx[np.argsort(end_ts[holdout_idx])]
    return holdout_idx


def _fallback_group_splits(session_ids, n_splits, seed=42):
    session_ids = np.asarray(session_ids).astype(str)
    unique_sessions = np.unique(session_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_sessions.copy()
    rng.shuffle(shuffled)
    session_chunks = np.array_split(shuffled, n_splits)

    for chunk in session_chunks:
        test_sessions = set(chunk.tolist())
        test_idx = np.array([i for i, sid in enumerate(session_ids.tolist()) if sid in test_sessions], dtype=np.int64)
        train_idx = np.array([i for i, sid in enumerate(session_ids.tolist()) if sid not in test_sessions], dtype=np.int64)
        yield train_idx, test_idx


def validate_with_group_kfold(
    X,
    y,
    session_ids,
    model_path: Path,
    threshold: float,
    n_splits: int = 10,
    batch_size: int = 256,
    temperature_scale: float = 1.0,
    train_per_fold: bool = False,
    train_epochs: int = 4,
    train_batch_size: int = 256,
    train_lr: float = 1e-3,
    train_seed: int = 42,
    train_dropout: float = 0.35,
    train_label_smoothing: float = 0.0,
    train_weight_decay: float = 0.0,
):
    """
    Group K-Fold evaluation with session IDs as groups.
    Each fold contains many sessions, producing statistically meaningful test blocks.
    """
    session_ids = np.asarray(session_ids).astype(str)
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < n_splits:
        raise SystemExit(f"Not enough unique sessions for GroupKFold: sessions={len(unique_sessions)} splits={n_splits}")

    print(f"[validate] Running GroupKFold with n_splits={n_splits}, sessions={len(unique_sessions)}")

    model = None
    probs_all = None
    if not train_per_fold:
        model = load_model(model_path)
        probs_all = predict_probs_batched(model, X, batch_size=batch_size, temperature_scale=temperature_scale)

    if GroupKFold is not None:
        splitter = GroupKFold(n_splits=n_splits).split(np.zeros(len(y), dtype=np.int8), y, groups=session_ids)
    else:
        splitter = _fallback_group_splits(session_ids, n_splits=n_splits, seed=42)

    fold_results = []
    f1_values = []
    zero_f1_folds = 0
    class_empty_folds = 0
    pooled_y_true = []
    pooled_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter, 1):
        y_test = y[test_idx]
        if train_per_fold:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            fold_model = train_fold_model(
                X_train,
                y_train,
                epochs=train_epochs,
                batch_size=train_batch_size,
                lr=train_lr,
                seed=train_seed,
                dropout=train_dropout,
                label_smoothing=train_label_smoothing,
                weight_decay=train_weight_decay,
            )
            probs_train = predict_probs_batched(fold_model, X_train, batch_size=batch_size, temperature_scale=1.0)
            fold_threshold = best_threshold_from_probs(y_train, probs_train)
            probs_test = predict_probs_batched(fold_model, X_test, batch_size=batch_size, temperature_scale=1.0)
        else:
            fold_threshold = threshold
            probs_test = probs_all[test_idx]
        y_pred, (tp, tn, fp, fn), metrics = evaluate_block(y_test, probs_test, fold_threshold)
        pooled_y_true.append(y_test.astype(np.int64))
        pooled_y_pred.append(y_pred.astype(np.int64))

        positives = int(y_test.sum())
        negatives = int(len(y_test) - positives)
        test_sessions = np.unique(session_ids[test_idx])
        has_both_classes = positives > 0 and negatives > 0

        if not has_both_classes:
            class_empty_folds += 1
        if metrics["f1"] == 0.0:
            zero_f1_folds += 1

        fold_results.append({
            "fold": int(fold_idx),
            "samples": int(len(y_test)),
            "sessions": int(len(test_sessions)),
            "positives": positives,
            "negatives": negatives,
            "has_both_classes": bool(has_both_classes),
            "threshold": float(fold_threshold),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        })
        f1_values.append(float(metrics["f1"]))

        print(
            f"[validate] Fold {fold_idx}/{n_splits}: sessions={len(test_sessions)} "
            f"samples={len(y_test)} f1={metrics['f1']:.4f}"
        )

    mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    std_f1 = float(np.std(f1_values)) if f1_values else 0.0
    ci_half = 1.96 * std_f1 / np.sqrt(max(1, len(f1_values)))

    mean_precision = float(np.mean([f["precision"] for f in fold_results])) if fold_results else 0.0
    mean_recall = float(np.mean([f["recall"] for f in fold_results])) if fold_results else 0.0
    mean_accuracy = float(np.mean([f["accuracy"] for f in fold_results])) if fold_results else 0.0

    if pooled_y_true:
        pooled_true = np.concatenate(pooled_y_true, axis=0)
        pooled_pred = np.concatenate(pooled_y_pred, axis=0)
        pooled_tp, pooled_tn, pooled_fp, pooled_fn = confusion_counts(pooled_true, pooled_pred)
    else:
        pooled_tp = pooled_tn = pooled_fp = pooled_fn = 0
    pooled_metrics = metrics_from_counts(pooled_tp, pooled_tn, pooled_fp, pooled_fn)

    return {
        "fold_count": int(len(fold_results)),
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "ci_low": float(mean_f1 - ci_half),
        "ci_high": float(mean_f1 + ci_half),
        "zero_f1_folds": int(zero_f1_folds),
        "zero_f1_rate": float(zero_f1_folds / max(1, len(fold_results))),
        "class_empty_folds": int(class_empty_folds),
        "class_empty_rate": float(class_empty_folds / max(1, len(fold_results))),
        "f1_values": f1_values,
        "mean_accuracy": mean_accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "pooled": {
            "tp": pooled_tp,
            "tn": pooled_tn,
            "fp": pooled_fp,
            "fn": pooled_fn,
            "accuracy": float(pooled_metrics["accuracy"]),
            "precision": float(pooled_metrics["precision"]),
            "recall": float(pooled_metrics["recall"]),
            "f1": float(pooled_metrics["f1"]),
        },
        "folds": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate trained model outputs on current dataset")
    parser.add_argument("--dataset", default="data/sequences_large.npz")
    parser.add_argument("--model", default="models/lstm_detector_large.pt")
    parser.add_argument("--meta", default="models/model_meta_large.json")
    parser.add_argument("--group-kfold", action="store_true", help="Use Group 10-Fold session CV")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds for Group K-Fold")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size (lower = less RAM)")
    parser.add_argument("--group-kfold-train", action="store_true", help="Train a fresh model for each GroupKFold split")
    parser.add_argument("--train-epochs", type=int, default=4, help="Epochs for per-fold training")
    parser.add_argument("--train-batch-size", type=int, default=256, help="Batch size for per-fold training")
    parser.add_argument("--train-lr", type=float, default=1e-3, help="Learning rate for per-fold training")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed for per-fold training")
    parser.add_argument("--train-dropout", type=float, default=0.35, help="Dropout for per-fold training")
    parser.add_argument("--train-label-smoothing", type=float, default=0.02, help="Label smoothing for per-fold training")
    parser.add_argument("--train-weight-decay", type=float, default=1e-4, help="Weight decay for per-fold training")
    parser.add_argument("--roc-output", default=None, help="Write ROC/PR curve JSON for time-holdout evaluation")
    parser.add_argument("--roc-steps", type=int, default=201, help="Number of threshold steps for ROC/PR curve")
    parser.add_argument("--use-best-threshold", action="store_true", help="Report time-holdout metrics at best-F1 threshold")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    meta_path = Path(args.meta)

    if not dataset_path.exists() or not model_path.exists() or not meta_path.exists():
        raise SystemExit("Missing required files. Run demo:e2e or pipeline first.")

    data = np.load(dataset_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    end_ts = np.asarray(data["end_ts"], dtype=np.int64) if "end_ts" in data.files else None
    session_ids = data["session_ids"] if "session_ids" in data.files else None

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    threshold = float(meta.get("threshold", 0.5))
    temperature_scale = float(meta.get("temperature_scale", 1.0) or 1.0)
    if temperature_scale <= 1e-6:
        temperature_scale = 1.0

    print("=== VALIDATION SUMMARY ===")
    print(f"dataset_path={dataset_path}")
    print(f"dataset_sha256={sha256_file(dataset_path)}")
    print(f"samples={len(y)}")
    if args.group_kfold:
        print(f"eval_strategy=GROUP_KFOLD_{args.n_splits}")
    else:
        print("eval_strategy=time_holdout")
    print(f"temperature_scale={temperature_scale:.4f}")

    if args.group_kfold and session_ids is not None:
        group_results = validate_with_group_kfold(
            X,
            y,
            session_ids,
            model_path,
            threshold,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            temperature_scale=temperature_scale,
            train_per_fold=bool(args.group_kfold_train),
            train_epochs=args.train_epochs,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_seed=args.train_seed,
            train_dropout=args.train_dropout,
            train_label_smoothing=args.train_label_smoothing,
            train_weight_decay=args.train_weight_decay,
        )

        print(f"group_folds={group_results['fold_count']}")
        print(f"group_mean_f1={group_results['mean_f1']:.4f}")
        print(f"group_std_f1={group_results['std_f1']:.4f}")
        print(f"group_ci_95 ≈ [{group_results['ci_low']:.4f}, {group_results['ci_high']:.4f}]")
        print(f"group_mean_accuracy={group_results['mean_accuracy']:.4f}")
        print(f"group_mean_precision={group_results['mean_precision']:.4f}")
        print(f"group_mean_recall={group_results['mean_recall']:.4f}")
        print(f"group_zero_f1_folds={group_results['zero_f1_folds']}")
        print(f"group_zero_f1_rate={group_results['zero_f1_rate']:.4f}")
        print(f"group_class_empty_folds={group_results['class_empty_folds']}")
        print(f"group_class_empty_rate={group_results['class_empty_rate']:.4f}")

        pooled = group_results.get("pooled") or {}
        if pooled:
            print("\n[group_pooled_metrics]")
            print(f"pooled_tp={pooled['tp']} tn={pooled['tn']} fp={pooled['fp']} fn={pooled['fn']}")
            print(f"pooled_accuracy={pooled['accuracy']:.4f}")
            print(f"pooled_precision={pooled['precision']:.4f}")
            print(f"pooled_recall={pooled['recall']:.4f}")
            print(f"pooled_f1={pooled['f1']:.4f}")

        print("\n=== GROUP-KFOLD DETAILS ===")
        for fold in group_results["folds"][:5]:
            print(
                f"Fold {fold['fold']}: sessions={fold['sessions']} samples={fold['samples']} "
                f"positives={fold['positives']} f1={fold['f1']:.4f}"
            )
        if len(group_results["folds"]) > 5:
            print(f"... and {len(group_results['folds']) - 5} more folds")

        all_pass = group_results["mean_f1"] >= 0.80

    else:
        # Standard time-split evaluation
        model = load_model(model_path)
        probs = predict_probs_batched(model, X, batch_size=args.batch_size, temperature_scale=temperature_scale)
        
        display_pred, (display_tp, display_tn, display_fp, display_fn), display_metrics = evaluate_block(
            y,
            probs,
            threshold,
        )
        display_threshold = threshold
        time_holdout_line = None
        display_y = y
        display_probs = probs
        if end_ts is not None and len(end_ts) == len(y):
            holdout_idx = stratified_time_holdout_idx(y, end_ts, holdout_ratio=0.2)

            y_h = y[holdout_idx]
            p_h = probs[holdout_idx]
            display_pred, (display_tp, display_tn, display_fp, display_fn), display_metrics = evaluate_block(
                y_h,
                p_h,
                threshold,
            )
            # Report holdout metrics as primary in time mode.
            display_y = y_h
            display_probs = p_h
            time_holdout_line = {
                "samples": int(len(y_h)),
                "accuracy": display_metrics["accuracy"],
                "precision": display_metrics["precision"],
                "recall": display_metrics["recall"],
                "f1": display_metrics["f1"],
            }

        if args.roc_output or args.use_best_threshold:
            curve = compute_curve(display_y, display_probs, steps=args.roc_steps)
            best_point = best_f1_point(curve)
            if best_point is not None:
                print(f"best_f1_threshold={best_point['threshold']:.4f}")
                print(f"best_f1={best_point['f1']:.4f}")
                if args.use_best_threshold:
                    display_threshold = float(best_point["threshold"])
                    display_pred, (display_tp, display_tn, display_fp, display_fn), display_metrics = evaluate_block(
                        display_y,
                        display_probs,
                        display_threshold,
                    )
                    time_holdout_line = {
                        "samples": int(len(display_y)),
                        "accuracy": display_metrics["accuracy"],
                        "precision": display_metrics["precision"],
                        "recall": display_metrics["recall"],
                        "f1": display_metrics["f1"],
                    }
        checks = [
            ("dataset_nontrivial", len(y) >= 100, f"samples={len(y)} expected>=100"),
            ("contains_both_classes", int(y.min()) == 0 and int(y.max()) == 1, "expected both 0 and 1 labels"),
            (
                "prediction_not_constant",
                int(display_pred.min()) == 0 and int(display_pred.max()) == 1,
                "expected both 0 and 1 predictions",
            ),
        ]
        if time_holdout_line is not None:
            checks.append(
                (
                    "time_holdout_f1",
                    display_metrics["f1"] >= 0.7,
                    f"time_holdout_f1={display_metrics['f1']:.3f} expected>=0.70",
                )
            )
        checks.append(
            (
                "reasonable_precision",
                display_metrics["precision"] >= 0.75,
                f"precision={display_metrics['precision']:.3f} expected>=0.75",
            )
        )
        checks.append(
            (
                "reasonable_recall",
                display_metrics["recall"] >= 0.75,
                f"recall={display_metrics['recall']:.3f} expected>=0.75",
            )
        )

        all_pass = all(c[1] for c in checks)

        if args.roc_output:
            if "curve" not in locals():
                curve = compute_curve(display_y, display_probs, steps=args.roc_steps)
            roc_auc = area_under_curve(curve, "fpr", "tpr")
            pr_auc = area_under_curve(curve, "recall", "precision")
            roc_payload = {
                "dataset_path": str(dataset_path),
                "samples": int(len(display_y)),
                "positive_ratio": float(display_y.mean()) if len(display_y) else 0.0,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "curve": curve,
            }
            Path(args.roc_output).write_text(json.dumps(roc_payload, indent=2), encoding="utf8")
            print(f"roc_output={args.roc_output}")
            print(f"roc_auc={roc_auc:.6f}")
            print(f"pr_auc={pr_auc:.6f}")

    if not args.group_kfold:
        positive_ratio = float(display_y.mean()) if len(display_y) else 0.0
        score_mean = float(display_probs.mean()) if len(display_probs) else 0.0

        print(f"positive_ratio={positive_ratio:.4f}")
        print(f"threshold={display_threshold}")
        print(f"score_mean={score_mean:.4f}")
        print(f"confusion_tp={display_tp} tn={display_tn} fp={display_fp} fn={display_fn}")
        print(f"accuracy={display_metrics['accuracy']:.4f}")
        print(f"precision={display_metrics['precision']:.4f}")
        print(f"recall={display_metrics['recall']:.4f}")
        print(f"f1={display_metrics['f1']:.4f}")

    if not args.group_kfold and time_holdout_line is not None:
        print(
            "time_holdout="
            f"samples={time_holdout_line['samples']} "
            f"acc={time_holdout_line['accuracy']:.4f} "
            f"precision={time_holdout_line['precision']:.4f} "
            f"recall={time_holdout_line['recall']:.4f} "
            f"f1={time_holdout_line['f1']:.4f}"
        )

    if not args.group_kfold:
        for name, passed, detail in checks:
            print(f"check_{name}={'PASS' if passed else 'FAIL'} ({detail})")

    print(f"validation_gate={'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
