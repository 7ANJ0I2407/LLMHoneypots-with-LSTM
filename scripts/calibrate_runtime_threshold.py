import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from autoencoder_model import SequenceAutoencoder, reconstruction_error
from ml_utils import apply_standardizer, events_to_grouped_features, load_events, load_standardizer
from train_lstm import LSTMDetector


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMDetector(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, int(checkpoint["window_size"])


def load_novelty_model(model_path: Path):
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
    parser.add_argument("--dataset", default="data/sequences_large.npz")
    parser.add_argument("--meta", default="models/model_meta_large.json")
    parser.add_argument("--model", default="models/lstm_detector_large.pt")
    parser.add_argument("--novelty-model", default="models/novelty_autoencoder_large.pt")
    parser.add_argument("--scaler", default="models/scaler_large.json")
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--min-threshold", type=float, default=0.50)
    parser.add_argument("--max-threshold", type=float, default=0.995)
    parser.add_argument(
        "--runtime-max-benign-fpr",
        type=float,
        default=0.05,
        help="Prefer runtime threshold with benign FPR below this bound when labels are available.",
    )
    parser.add_argument(
        "--calibration-source",
        default="dataset",
        choices=["dataset", "live", "auto"],
        help="Source for runtime threshold calibration. 'dataset' is reproducible default.",
    )
    parser.add_argument("--min-live-windows", type=int, default=1000)
    parser.add_argument("--min-live-positives", type=int, default=30)
    parser.add_argument("--runtime-novelty-quantile", type=float, default=0.995)
    parser.add_argument(
        "--runtime-novelty-target-recall",
        type=float,
        default=0.75,
        help="Target recall for novelty calibration when labeled live windows are available.",
    )
    parser.add_argument(
        "--runtime-novelty-benign-percentile",
        type=float,
        default=99.5,
        help="Benign percentile anchor for runtime novelty calibration.",
    )
    parser.add_argument(
        "--runtime-novelty-midpoint-weight",
        type=float,
        default=0.3,
        help="Weight for midpoint between benign anchor and attack p01.",
    )
    parser.add_argument(
        "--runtime-novelty-min-separation",
        type=float,
        default=0.15,
        help="Minimum absolute separation above benign anchor.",
    )
    parser.add_argument(
        "--runtime-novelty-margin",
        type=float,
        default=1.0,
        help="Multiply calibrated novelty threshold by this margin (>1.0 to reduce benign novelty alerts).",
    )
    parser.add_argument("--min-novelty-negatives", type=int, default=200)
    parser.add_argument(
        "--runtime-novelty-mode",
        default="keep-meta",
        choices=["keep-meta", "recompute"],
        help="How to set runtime novelty threshold. keep-meta is reproducible default.",
    )
    parser.add_argument("--z-clip", type=float, default=3.0)
    parser.add_argument("--fusion-enabled", action="store_true", help="Enable runtime fusion decision mode")
    parser.add_argument("--fusion-mode", default="or", choices=["or", "blend"], help="Fusion mode for runtime decisions")
    parser.add_argument("--fusion-alpha", type=float, default=0.5, help="Blend alpha when --fusion-mode=blend")
    parser.add_argument("--fusion-blend-threshold", type=float, default=None, help="Decision threshold for blend fusion score")
    parser.add_argument("--fusion-calibrate-blend", action="store_true", help="Auto-calibrate blend threshold from calibration windows")
    parser.add_argument("--fusion-max-benign-fpr", type=float, default=0.03, help="Max benign FPR when auto-calibrating blend threshold")

    args = parser.parse_args()

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    meta_path = Path(args.meta)

    if not model_path.exists() or not scaler_path.exists() or not meta_path.exists():
        raise SystemExit("Missing model/scaler/meta files. Run training first.")

    meta = json.loads(meta_path.read_text(encoding="utf8"))
    temperature_scale = float(meta.get("temperature_scale", 1.0) or 1.0)
    if temperature_scale <= 1e-6:
        temperature_scale = 1.0

    mean, std, _ = load_standardizer(scaler_path)
    model, window_size = load_model(model_path)

    calibration_source = "dataset"
    live_windows = []
    live_labels = []
    live_end_ts = []

    events = load_events(Path(args.input))
    grouped = events_to_grouped_features(events)

    runtime_errs = []
    runtime_labels = []
    explicit_labels = []
    for event in events:
        rd = event.get("runtimeDetection") or {}
        novelty = rd.get("novelty") or {}
        err = novelty.get("error")
        label = str(event.get("label", "")).strip().lower()
        if label in {"attack", "benign"}:
            explicit_labels.append(1 if label == "attack" else 0)
        if err is None or label not in {"attack", "benign"}:
            continue
        runtime_errs.append(float(err))
        runtime_labels.append(1 if label == "attack" else 0)

    attack_session_ids = set()
    benign_session_ids = set()
    trigger_path = Path("logs/latest_trigger_session.json")
    benign_run_path = Path("benign_runs/calibration_run.json")
    if not benign_run_path.exists():
        benign_run_path = Path("benign_runs/run_latest.json")
    if trigger_path.exists():
        try:
            payload = json.loads(trigger_path.read_text(encoding="utf8"))
            sid = payload.get("sessionId")
            if sid:
                attack_session_ids.add(sid)
        except Exception:
            pass
    if benign_run_path.exists():
        try:
            payload = json.loads(benign_run_path.read_text(encoding="utf8"))
            run_ids = payload.get("runtime_session_ids") or []
            if payload.get("session_id"):
                run_ids = list(run_ids) + [payload.get("session_id")]
            benign_session_ids.update([sid for sid in run_ids if sid])
        except Exception:
            pass

    labeled_attack_errors = []
    labeled_benign_errors = []
    if attack_session_ids or benign_session_ids:
        for event in events:
            sid = event.get("sessionId")
            if sid not in attack_session_ids and sid not in benign_session_ids:
                continue
            rd = event.get("runtimeDetection") or {}
            novelty = rd.get("novelty") or {}
            err = novelty.get("error")
            if err is None:
                continue
            if sid in attack_session_ids:
                labeled_attack_errors.append(float(err))
            elif sid in benign_session_ids:
                labeled_benign_errors.append(float(err))

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
            live_windows.append(win_X)
            live_labels.append(label)
            live_end_ts.append(int(ts[end - 1]) if ts is not None else end)

    live_ready = (
        len(live_windows) >= int(args.min_live_windows)
        and int(np.sum(live_labels)) >= int(args.min_live_positives)
    )

    # Always calibrate the LSTM threshold on the dataset for stability.
    use_live = False

    if use_live:
        X = np.array(live_windows, dtype=np.float32)
        y = np.array(live_labels, dtype=np.int64)
        t = np.array(live_end_ts, dtype=np.int64)
        idx = np.argsort(t)
        X = X[idx]
        y = y[idx]
        if len(X) > args.max_windows:
            X = X[-args.max_windows :]
            y = y[-args.max_windows :]
        Xs = apply_standardizer(X, mean, std)
        if args.z_clip > 0:
            Xs = np.clip(Xs, -args.z_clip, args.z_clip)
        positive_windows = int(y.sum())
        window_count = int(len(X))
    else:
        calibration_source = "dataset"
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise SystemExit(
                "Insufficient live calibration windows and dataset file is missing. "
                "Run preprocess.py or pass --dataset."
            )
        data = np.load(dataset_path, allow_pickle=True)
        Xs = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        Xs_full = Xs
        y_full = y
        end_ts = np.asarray(data["end_ts"], dtype=np.int64) if "end_ts" in data.files else np.arange(len(y), dtype=np.int64)
        idx = np.argsort(end_ts)
        Xs = Xs[idx]
        y = y[idx]
        Xs_full = Xs_full[idx]
        y_full = y_full[idx]

        dataset_sampling = "all"
        if len(Xs) > args.max_windows:
            tail_Xs = Xs[-args.max_windows :]
            tail_y = y[-args.max_windows :]
            if int(tail_y.sum()) == 0:
                rng = np.random.default_rng(42)
                pos_idx = np.flatnonzero(y == 1)
                neg_idx = np.flatnonzero(y == 0)
                pos_ratio = float(y.mean()) if len(y) else 0.0
                desired_pos = int(round(args.max_windows * pos_ratio))
                desired_pos = max(1, min(args.max_windows - 1, desired_pos))
                desired_pos = min(desired_pos, len(pos_idx))
                neg_need = args.max_windows - desired_pos
                if neg_need > len(neg_idx):
                    neg_need = len(neg_idx)
                    desired_pos = max(1, args.max_windows - neg_need)
                pos_pick = rng.choice(pos_idx, size=desired_pos, replace=False) if len(pos_idx) else np.empty((0,), dtype=np.int64)
                neg_pick = rng.choice(neg_idx, size=neg_need, replace=False) if neg_need > 0 else np.empty((0,), dtype=np.int64)
                pick = np.concatenate([pos_pick, neg_pick])
                pick = np.sort(pick)
                Xs = Xs[pick]
                y = y[pick]
                dataset_sampling = "stratified"
            else:
                Xs = tail_Xs
                y = tail_y
                dataset_sampling = "tail"

        if args.z_clip > 0:
            Xs = np.clip(Xs, -args.z_clip, args.z_clip)
        positive_windows = int(y.sum())
        window_count = int(len(Xs))

    with torch.no_grad():
        logits = model(torch.from_numpy(Xs))
        probs = torch.sigmoid(logits / temperature_scale).numpy().astype(np.float32)

    best = {
        "threshold": 0.1,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    thr_min = float(min(args.min_threshold, args.max_threshold))
    thr_max = float(max(args.min_threshold, args.max_threshold))
    if thr_max <= 0:
        raise SystemExit("--max-threshold must be > 0")

    best_any = None
    best_constrained = None
    best_low_fpr = None

    for thr in np.linspace(thr_min, thr_max, 191):
        pred = (probs >= thr).astype(np.int64)
        f1, precision, recall = f1_score(y, pred)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        benign_fpr = fp / max(fp + tn, 1)
        row = {
            "threshold": float(thr),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "benign_fpr": float(benign_fpr),
        }

        if best_any is None or row["f1"] > best_any["f1"]:
            best_any = row

        if (
            best_low_fpr is None
            or row["benign_fpr"] < best_low_fpr["benign_fpr"]
            or (row["benign_fpr"] == best_low_fpr["benign_fpr"] and row["f1"] > best_low_fpr["f1"])
        ):
            best_low_fpr = row

        if benign_fpr <= float(args.runtime_max_benign_fpr):
            if best_constrained is None or row["f1"] > best_constrained["f1"]:
                best_constrained = row

    best = best_constrained if best_constrained is not None else best_low_fpr
    if best is None:
        raise SystemExit("Unable to calibrate threshold from provided windows.")

    calibrated_threshold = float(best["threshold"])

    runtime_novelty_threshold = None
    runtime_novelty_source = None
    aligned_novelty_threshold = None
    runtime_novelty_benign_anchor = None
    runtime_novelty_attack_p01 = None
    runtime_novelty_benign_percentile = float(args.runtime_novelty_benign_percentile)
    novelty_model = None
    novelty_model_path = Path(args.novelty_model)
    if novelty_model_path.exists():
        novelty_model = load_novelty_model(novelty_model_path)

    if args.runtime_novelty_mode == "keep-meta":
        runtime_novelty_threshold = meta.get("runtime_novelty_threshold", meta.get("novelty_threshold"))
        runtime_novelty_source = "meta"
        if runtime_novelty_threshold is None:
            raise SystemExit(
                "No novelty threshold found in meta for --runtime-novelty-mode keep-meta. "
                "Use --runtime-novelty-mode recompute once, then rerun."
            )
        runtime_novelty_threshold = float(runtime_novelty_threshold)
    elif novelty_model is not None:
        if labeled_attack_errors and labeled_benign_errors:
            benign_arr = np.asarray(labeled_benign_errors, dtype=np.float32)
            attack_arr = np.asarray(labeled_attack_errors, dtype=np.float32)
            benign_anchor = float(np.percentile(benign_arr, runtime_novelty_benign_percentile))
            attack_p01 = float(np.percentile(attack_arr, 1))
            runtime_novelty_benign_anchor = benign_anchor
            runtime_novelty_attack_p01 = attack_p01
            midpoint_weight = float(args.runtime_novelty_midpoint_weight)
            min_separation = float(args.runtime_novelty_min_separation)
            if attack_p01 > benign_anchor:
                candidate = benign_anchor + midpoint_weight * (attack_p01 - benign_anchor)
                runtime_novelty_threshold = max(candidate, benign_anchor + min_separation)
            else:
                fallback = benign_anchor * (1.0 + float(args.runtime_novelty_margin))
                runtime_novelty_threshold = max(fallback, benign_anchor + min_separation)
            runtime_novelty_source = "runtime_errors_labeled"
        elif runtime_errs and runtime_labels:
            labels_arr = np.asarray(runtime_labels, dtype=np.int64)
            errs_arr = np.asarray(runtime_errs, dtype=np.float32)
            attack_errs = errs_arr[labels_arr == 1]
            benign_errs = errs_arr[labels_arr == 0]
            if len(attack_errs) > 0 and len(benign_errs) > 0:
                target_recall = float(args.runtime_novelty_target_recall)
                target_recall = max(0.5, min(0.99, target_recall))
                candidate = float(np.quantile(attack_errs, 1.0 - target_recall))
                candidate = max(candidate, float(benign_errs.max()))
                runtime_novelty_threshold = candidate
                runtime_novelty_source = "runtime_errors_labeled"

        # Prefer live negatives for runtime novelty calibration when enough windows exist.
        live_neg_idx = np.flatnonzero(np.asarray(live_labels, dtype=np.int64) == 0)
        explicit_mixed = bool(explicit_labels) and (0 in explicit_labels) and (1 in explicit_labels)
        allow_live_novelty = args.calibration_source in {"live", "auto"} and explicit_mixed
        if runtime_novelty_threshold is None:
            if allow_live_novelty and len(live_neg_idx) >= int(args.min_novelty_negatives):
                X_live = np.asarray(live_windows, dtype=np.float32)
                Xn = apply_standardizer(X_live[live_neg_idx], mean, std)
                if args.z_clip > 0:
                    Xn = np.clip(Xn, -args.z_clip, args.z_clip)
                runtime_novelty_source = "live_log"
            else:
                runtime_novelty_threshold = meta.get("runtime_novelty_threshold", meta.get("novelty_threshold"))
                if runtime_novelty_threshold is not None:
                    runtime_novelty_threshold = float(runtime_novelty_threshold)
                    runtime_novelty_source = "meta"
                else:
                    negative_idx = np.flatnonzero(y_full == 0)
                    if len(negative_idx) < int(args.min_novelty_negatives):
                        Xn = None
                    else:
                        Xn = Xs_full[negative_idx]
                        if args.z_clip > 0:
                            Xn = np.clip(Xn, -args.z_clip, args.z_clip)
                        runtime_novelty_source = "dataset"

        if runtime_novelty_threshold is None and Xn is not None:
            err_chunks = []
            batch_size = 128
            with torch.no_grad():
                for start in range(0, len(Xn), batch_size):
                    end = min(start + batch_size, len(Xn))
                    batch = torch.from_numpy(Xn[start:end])
                    recon = novelty_model(batch)
                    err_chunks.append(reconstruction_error(batch, recon).numpy())
            errs = np.concatenate(err_chunks, axis=0) if err_chunks else np.empty((0,), dtype=np.float32)
            if len(errs) > 0:
                runtime_novelty_threshold = float(np.quantile(errs, args.runtime_novelty_quantile))
                runtime_novelty_threshold *= float(max(args.runtime_novelty_margin, 0.1))

    if runtime_novelty_threshold is not None:
        train_novelty_threshold = meta.get("novelty_threshold")
        if train_novelty_threshold is not None:
            train_val = float(train_novelty_threshold)
            ratio = runtime_novelty_threshold / max(train_val, 1e-12)
            if ratio > 10.0 or ratio < 0.1:
                meta["novelty_threshold"] = float(runtime_novelty_threshold)
                meta["novelty_threshold_source"] = "runtime_alignment"
                aligned_novelty_threshold = float(runtime_novelty_threshold)

    fusion_blend_threshold = args.fusion_blend_threshold
    fusion_calibration = None
    if (
        args.fusion_enabled
        and args.fusion_mode == "blend"
        and args.fusion_calibrate_blend
        and runtime_novelty_threshold is not None
        and novelty_model is not None
    ):
        # Calibrate blend threshold on the same window set used for runtime threshold fitting.
        with torch.no_grad():
            X_tensor = torch.from_numpy(Xs)
            recon = novelty_model(X_tensor)
            errs_all = reconstruction_error(X_tensor, recon).numpy()
        novelty_signal = np.clip(errs_all / max(runtime_novelty_threshold, 1e-8), 0.0, 1.0)
        combined = (float(args.fusion_alpha) * probs) + ((1.0 - float(args.fusion_alpha)) * novelty_signal)

        best_fusion = None
        for thr in np.linspace(0.05, 0.95, 181):
            pred = (combined >= thr).astype(np.int64)
            f1, precision, recall = f1_score(y, pred)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            benign_fpr = fp / max(fp + tn, 1)
            if benign_fpr <= float(args.fusion_max_benign_fpr):
                row = {
                    "threshold": float(thr),
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "benign_fpr": float(benign_fpr),
                }
                if best_fusion is None or row["f1"] > best_fusion["f1"]:
                    best_fusion = row

        if best_fusion is not None:
            fusion_blend_threshold = best_fusion["threshold"]
            fusion_calibration = {
                "alpha": float(args.fusion_alpha),
                "threshold": float(fusion_blend_threshold),
                "f1": float(best_fusion["f1"]),
                "precision": float(best_fusion["precision"]),
                "recall": float(best_fusion["recall"]),
                "benign_fpr": float(best_fusion["benign_fpr"]),
                "max_benign_fpr": float(args.fusion_max_benign_fpr),
            }

    meta["runtime_threshold"] = calibrated_threshold
    if runtime_novelty_threshold is not None:
        meta["runtime_novelty_threshold"] = runtime_novelty_threshold
        train_threshold = meta.get("novelty_threshold")
        if train_threshold is not None:
            try:
                train_threshold = float(train_threshold)
            except (TypeError, ValueError):
                train_threshold = None
        if train_threshold is not None and train_threshold > 0:
            ratio = float(runtime_novelty_threshold) / train_threshold
            if ratio > 10.0:
                meta.setdefault("novelty_threshold_train", train_threshold)
                meta["novelty_threshold"] = float(runtime_novelty_threshold)
                meta["novelty_threshold_source"] = "runtime_override"
                meta["novelty_threshold_ratio"] = float(ratio)
    meta["runtime_fusion"] = {
        "enabled": bool(args.fusion_enabled),
        "mode": str(args.fusion_mode),
        "alpha": float(args.fusion_alpha),
        "blend_threshold": None if fusion_blend_threshold is None else float(fusion_blend_threshold),
    }
    if fusion_calibration is not None:
        meta["runtime_fusion"]["calibration"] = fusion_calibration
    meta["runtime_calibration"] = {
        "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": calibration_source,
        "windows": window_count,
        "positive_windows": positive_windows,
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "raw_best_threshold": best["threshold"],
        "min_threshold": float(args.min_threshold),
        "max_threshold": float(args.max_threshold),
        "runtime_max_benign_fpr": float(args.runtime_max_benign_fpr),
        "selected_benign_fpr": float(best.get("benign_fpr", 0.0)),
        "z_clip": float(args.z_clip),
        "runtime_novelty_threshold": runtime_novelty_threshold,
        "runtime_novelty_quantile": float(args.runtime_novelty_quantile),
        "runtime_novelty_margin": float(args.runtime_novelty_margin),
        "runtime_novelty_source": runtime_novelty_source,
        "runtime_novelty_mode": str(args.runtime_novelty_mode),
        "runtime_novelty_target_recall": float(args.runtime_novelty_target_recall),
        "calibration_source_mode": str(args.calibration_source),
        "runtime_novelty_benign_anchor": runtime_novelty_benign_anchor,
        "runtime_novelty_benign_percentile": runtime_novelty_benign_percentile,
        "runtime_novelty_attack_p01": runtime_novelty_attack_p01,
        "runtime_novelty_midpoint_weight": float(args.runtime_novelty_midpoint_weight),
        "runtime_novelty_min_separation": float(args.runtime_novelty_min_separation),
    }
    if calibration_source == "dataset":
        meta["runtime_calibration"]["dataset_sampling"] = locals().get("dataset_sampling")
    meta["runtime_calibration"]["aligned_novelty_threshold"] = aligned_novelty_threshold

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")

    print(
        "runtime_threshold_calibrated="
        f"{calibrated_threshold:.4f} source={calibration_source} windows={window_count} positives={positive_windows} "
        f"f1={best['f1']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f}"
    )
    if runtime_novelty_threshold is not None:
        print(
            "runtime_novelty_threshold_calibrated="
            f"{runtime_novelty_threshold:.6f} quantile={args.runtime_novelty_quantile:.4f} source={runtime_novelty_source}"
        )
        if runtime_novelty_benign_anchor is not None and runtime_novelty_attack_p01 is not None:
            print(
                "runtime_novelty_split="
                f"benign_p{runtime_novelty_benign_percentile:.1f}={runtime_novelty_benign_anchor:.4f} "
                f"attack_p01={runtime_novelty_attack_p01:.4f} separable={runtime_novelty_attack_p01 > runtime_novelty_benign_anchor}"
            )
    if fusion_calibration is not None:
        print(
            "runtime_fusion_blend_calibrated="
            f"alpha={fusion_calibration['alpha']:.3f} threshold={fusion_calibration['threshold']:.4f} "
            f"f1={fusion_calibration['f1']:.4f} benign_fpr={fusion_calibration['benign_fpr']:.4f}"
        )


if __name__ == "__main__":
    main()
