import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch

from autoencoder_model import SequenceAutoencoder, reconstruction_error
from ml_utils import apply_standardizer, build_feature_vector, compute_session_features, load_standardizer, parse_iso
from train_lstm import LSTMDetector


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMDetector(input_dim=int(checkpoint["input_dim"]))
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


def clamp01(value: float):
    return max(0.0, min(1.0, float(value)))


def window_events_to_matrix(window):
    if not window:
        return np.empty((0, 0), dtype=np.float32)

    if not isinstance(window[0], dict):
        return np.array(window, dtype=np.float32)

    ordered = sorted(window, key=lambda rec: rec.get("timestamp", ""))
    session_features = compute_session_features(ordered)

    rows = []
    prev_ts = None
    for rec, extra in zip(ordered, session_features):
        current_ts = parse_iso(rec["timestamp"])
        inter_arrival_sec = 0.0 if prev_ts is None else max(0.0, (current_ts - prev_ts).total_seconds())
        prev_ts = current_ts
        hour_float = current_ts.hour + (current_ts.minute / 60.0)
        angle = 2.0 * np.pi * hour_float / 24.0
        rows.append(
            build_feature_vector(
                {**rec, **extra},
                inter_arrival_sec,
                hour_sin=float(np.sin(angle)),
                hour_cos=float(np.cos(angle)),
            )
        )

    return np.asarray(rows, dtype=np.float32)


def load_runtime_context(args):
    model_path = Path(args.model)
    meta_path = Path(args.meta)
    scaler_path = Path(args.scaler)

    if not (model_path.exists() and meta_path.exists() and scaler_path.exists()):
        raise FileNotFoundError("model_files_missing")

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.5)))
    temperature_scale = float(meta.get("temperature_scale", 1.0) or 1.0)
    if temperature_scale <= 1e-6:
        temperature_scale = 1.0
    train_novelty = meta.get("novelty_threshold_train", meta.get("novelty_threshold"))
    runtime_novelty = meta.get("runtime_novelty_threshold", meta.get("novelty_threshold"))
    if train_novelty is not None and runtime_novelty is not None:
        train_val = float(train_novelty)
        runtime_val = float(runtime_novelty)
        ratio = runtime_val / max(train_val, 1e-12)
        if ratio > 10.0:
            raise ValueError(
                "novelty_threshold_mismatch "
                f"train={train_val:.6f} runtime={runtime_val:.6f} ratio={ratio:.2f}"
            )
    novelty_threshold = float(runtime_novelty) if runtime_novelty is not None else 0.0
    runtime_fusion = meta.get("runtime_fusion") or {}
    fusion_enabled = bool(runtime_fusion.get("enabled", False))
    fusion_mode = str(runtime_fusion.get("mode", "or"))
    fusion_alpha = float(runtime_fusion.get("alpha", 0.5))
    fusion_blend_threshold = runtime_fusion.get("blend_threshold", None)
    if fusion_blend_threshold is not None:
        fusion_blend_threshold = float(fusion_blend_threshold)
    mean, std, _ = load_standardizer(scaler_path)
    model, window_size = load_model(model_path)

    novelty_model = None
    novelty_model_path = Path(args.novelty_model)
    if novelty_model_path.exists() and novelty_threshold > 0:
        novelty_model = load_novelty_model(novelty_model_path)

    return {
        "threshold": threshold,
        "temperature_scale": temperature_scale,
        "novelty_threshold": novelty_threshold,
        "fusion_enabled": fusion_enabled,
        "fusion_mode": fusion_mode,
        "fusion_alpha": fusion_alpha,
        "fusion_blend_threshold": fusion_blend_threshold,
        "mean": mean,
        "std": std,
        "model": model,
        "window_size": window_size,
        "novelty_model": novelty_model,
        "z_clip": float(args.z_clip),
    }


def score_payload(payload, ctx):
    window = payload.get("window")
    if not isinstance(window, list) or not window:
        return {"ok": False, "error": "window_missing"}

    X = window_events_to_matrix(window)
    if X.ndim != 2 or X.size == 0:
        return {"ok": False, "error": "window_shape_invalid"}

    window_size = int(ctx["window_size"])
    if X.shape[0] < window_size:
        pad_rows = np.repeat(X[:1], window_size - X.shape[0], axis=0)
        X = np.vstack([pad_rows, X])
    elif X.shape[0] > window_size:
        X = X[-window_size:]

    X = X[None, ...]
    X_scaled = apply_standardizer(X, ctx["mean"], ctx["std"])
    if float(ctx.get("z_clip", 0.0)) > 0:
        X_scaled = np.clip(X_scaled, -float(ctx["z_clip"]), float(ctx["z_clip"]))

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled)
        logits = ctx["model"](X_tensor)
        temp = float(ctx.get("temperature_scale", 1.0) or 1.0)
        if temp <= 1e-6:
            temp = 1.0
        score = float(torch.sigmoid(logits / temp).numpy()[0])

    novelty_threshold = float(ctx["novelty_threshold"])
    lstm_threshold = float(ctx["threshold"])
    novelty_payload = {
        "enabled": False,
        "error": 0.0,
        "threshold": novelty_threshold,
        "decision": False,
        "signal": 0.0,
    }

    if ctx["novelty_model"] is not None and novelty_threshold > 0:
        with torch.no_grad():
            recon = ctx["novelty_model"](X_tensor)
            err = float(reconstruction_error(X_tensor, recon).numpy()[0])
        if os.environ.get("DEBUG_NOVELTY") == "1":
            print(f"DEBUG novelty_error={err:.8f} threshold={novelty_threshold:.8f}", flush=True)
        novelty_payload = {
            "enabled": True,
            "error": err,
            "threshold": novelty_threshold,
            "decision": bool(err >= novelty_threshold),
            "signal": clamp01(err / max(novelty_threshold, 1e-8)),
        }


    lstm_decision = bool(score >= lstm_threshold)
    fusion_enabled = bool(ctx.get("fusion_enabled", False))
    fusion_mode = str(ctx.get("fusion_mode", "or"))
    fusion_alpha = float(ctx.get("fusion_alpha", 0.5))

    fusion_score = score
    fusion_threshold = lstm_threshold
    fusion_decision = bool(lstm_decision or novelty_payload["decision"])
    decision_source = "or"

    if fusion_enabled and fusion_mode == "blend" and novelty_payload["enabled"]:
        fusion_score = float((fusion_alpha * score) + ((1.0 - fusion_alpha) * novelty_payload["signal"]))
        if ctx.get("fusion_blend_threshold") is not None:
            fusion_threshold = float(ctx["fusion_blend_threshold"])
        fusion_decision = bool(fusion_score >= fusion_threshold)
        decision_source = "fusion_blend"

    final_decision = fusion_decision if fusion_enabled else bool(lstm_decision or novelty_payload["decision"])

    return {
        "ok": True,
        "score": score,
        "threshold": lstm_threshold,
        "windowSize": window_size,
        "decision": final_decision,
        "decisionSource": decision_source,
        "lstm": {
            "score": score,
            "threshold": lstm_threshold,
            "decision": lstm_decision,
        },
        "fusion": {
            "enabled": fusion_enabled,
            "mode": fusion_mode,
            "alpha": fusion_alpha if fusion_mode == "blend" else None,
            "score": fusion_score,
            "threshold": fusion_threshold,
            "decision": fusion_decision,
        },
        "novelty": novelty_payload,
    }


def run_service_mode(ctx):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            result = score_payload(payload, ctx)
            print(json.dumps({"id": req_id, "result": result}), flush=True)
        except Exception as exc:  # pragma: no cover
            print(json.dumps({"id": req_id, "error": str(exc)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Score a single session window for online detection")
    parser.add_argument("--model", default="models/lstm_detector.pt")
    parser.add_argument("--novelty-model", default="models/novelty_autoencoder.pt")
    parser.add_argument("--meta", default="models/model_meta.json")
    parser.add_argument("--scaler", default="models/scaler.json")
    parser.add_argument("--serve", action="store_true", help="Run as persistent JSONL scorer service")
    parser.add_argument("--z-clip", type=float, default=3.0, help="Clip standardized features to +/-z_clip during inference")
    args = parser.parse_args()

    try:
        ctx = load_runtime_context(args)
    except FileNotFoundError:
        if args.serve:
            print(json.dumps({"id": None, "error": "model_files_missing"}), flush=True)
        else:
            print(json.dumps({"ok": False, "error": "model_files_missing"}))
        return

    if args.serve:
        run_service_mode(ctx)
        return

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps({"ok": False, "error": "invalid_json_input"}))
        return

    print(json.dumps(score_payload(payload, ctx)))


if __name__ == "__main__":
    main()
