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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_ms(ts):
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def explain_window(window: np.ndarray, lstm_score: float, threshold: float, novelty_signal: float = 0.0):
    latest = window[-1]
    mean_signal = float(np.mean(window[:, 2]))
    mean_interarrival = float(np.mean(window[:, 6]))
    max_prompt = float(np.max(window[:, 0]))

    rule_proxy = clamp01(latest[2] / 4.0)
    velocity_proxy = clamp01(1.0 - min(mean_interarrival, 10.0) / 10.0)
    hybrid_risk = clamp01(0.50 * lstm_score + 0.30 * rule_proxy + 0.10 * velocity_proxy + 0.10 * novelty_signal)

    factors = [
        {
            "name": "lstm_sequence_score",
            "impact": round(lstm_score, 4),
            "detail": "Sequence-level behavior anomaly confidence",
        },
        {
            "name": "rule_signal_proxy",
            "impact": round(rule_proxy, 4),
            "detail": "Latest event keyword/pattern suspiciousness",
        },
        {
            "name": "rapid_interarrival_proxy",
            "impact": round(velocity_proxy, 4),
            "detail": "Short time gaps indicate burst-like probing",
        },
        {
            "name": "novelty_signal",
            "impact": round(novelty_signal, 4),
            "detail": "Reconstruction-error novelty confidence",
        },
        {
            "name": "window_mean_signal",
            "impact": round(clamp01(mean_signal / 4.0), 4),
            "detail": "Average suspicious signal in current window",
        },
        {
            "name": "max_prompt_length_proxy",
            "impact": round(clamp01(max_prompt / 3000.0), 4),
            "detail": "Large prompts can indicate extraction attempts",
        },
    ]
    factors = sorted(factors, key=lambda f: f["impact"], reverse=True)[:3]

    if hybrid_risk >= 0.85:
        severity = "critical"
    elif hybrid_risk >= 0.65:
        severity = "high"
    elif hybrid_risk >= 0.40:
        severity = "medium"
    else:
        severity = "low"

    uncertainty = abs(lstm_score - threshold)
    confidence_band = "uncertain" if uncertainty <= 0.07 else "confident"

    return hybrid_risk, severity, confidence_band, factors


def main():
    parser = argparse.ArgumentParser(description="Run trained LSTM detector on latest honeypot logs")
    parser.add_argument("--input", default="logs/raw_events.jsonl")
    parser.add_argument("--model", default="models/lstm_detector.pt")
    parser.add_argument("--novelty-model", default="models/novelty_autoencoder.pt")
    parser.add_argument("--meta", default="models/model_meta.json")
    parser.add_argument("--scaler", default="models/scaler.json")
    parser.add_argument("--alerts-out", default="logs/alerts.jsonl")
    parser.add_argument("--z-clip", type=float, default=3.0)

    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    scaler_path = Path(args.scaler)

    if not model_path.exists() or not meta_path.exists() or not scaler_path.exists():
        raise SystemExit("Missing model files. Run preprocess.py and train_lstm.py first.")

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.5)))
    novelty_threshold = float(meta.get("runtime_novelty_threshold", meta.get("novelty_threshold", 0.0)))
    temperature_scale = float(meta.get("temperature_scale", 1.0) or 1.0)
    if temperature_scale <= 1e-6:
        temperature_scale = 1.0
    runtime_fusion = meta.get("runtime_fusion") or {}
    fusion_enabled = bool(runtime_fusion.get("enabled", False))
    fusion_mode = str(runtime_fusion.get("mode", "or"))
    fusion_alpha = float(runtime_fusion.get("alpha", 0.5))
    fusion_blend_threshold = runtime_fusion.get("blend_threshold", None)
    if fusion_blend_threshold is not None:
        fusion_blend_threshold = float(fusion_blend_threshold)

    events = load_events(Path(args.input))
    trigger_path = Path("logs/latest_trigger_session.json")
    session_filter = None
    start_ms = None
    if trigger_path.exists():
        try:
            trig = json.loads(trigger_path.read_text(encoding="utf8"))
            session_filter = trig.get("sessionId")
            start_ms = parse_ms(trig.get("startedAt") or trig.get("createdAt"))
        except Exception:
            session_filter = None
            start_ms = None

    if session_filter:
        filtered = []
        for ev in events:
            if ev.get("sessionId") != session_filter:
                continue
            if start_ms is not None:
                ev_ms = parse_ms(ev.get("timestamp"))
                if ev_ms is None or ev_ms < start_ms:
                    continue
            filtered.append(ev)
        events = filtered
    grouped = events_to_grouped_features(events)

    mean, std, _feature_names = load_standardizer(scaler_path)
    model, window_size = load_model(model_path)
    novelty_model = None
    novelty_model_path = Path(args.novelty_model)
    if novelty_model_path.exists() and novelty_threshold > 0:
        novelty_model = load_novelty_model(novelty_model_path)

    alerts = []

    for session_id, data in grouped.items():
        X = data["X"]
        if len(X) < window_size:
            continue

        window = X[-window_size:][None, ...].astype(np.float32)
        window_scaled = apply_standardizer(window, mean, std)
        if args.z_clip > 0:
            window_scaled = np.clip(window_scaled, -args.z_clip, args.z_clip)
        window_tensor = torch.from_numpy(window_scaled)

        with torch.no_grad():
            logits = model(window_tensor)
            prob = float(torch.sigmoid(logits / temperature_scale).numpy()[0])

        novelty_error = 0.0
        novelty_signal = 0.0
        novelty_decision = False
        fusion_score = prob
        fusion_threshold = threshold
        fusion_decision = bool(prob >= threshold)
        decision_source = "or"
        if novelty_model is not None:
            with torch.no_grad():
                recon = novelty_model(window_tensor)
                novelty_error = float(reconstruction_error(window_tensor, recon).numpy()[0])
            novelty_signal = clamp01(novelty_error / max(novelty_threshold, 1e-8))
            novelty_decision = novelty_error >= novelty_threshold

        if fusion_enabled and fusion_mode == "blend" and novelty_model is not None:
            fusion_score = float((fusion_alpha * prob) + ((1.0 - fusion_alpha) * novelty_signal))
            if fusion_blend_threshold is not None:
                fusion_threshold = fusion_blend_threshold
            fusion_decision = bool(fusion_score >= fusion_threshold)
            decision_source = "fusion_blend"
        else:
            fusion_decision = bool(prob >= threshold or novelty_decision)

        if fusion_decision:
            hybrid_risk, severity, confidence_band, top_factors = explain_window(
                window[0],
                fusion_score,
                fusion_threshold,
                novelty_signal,
            )
            alert_type = "lstm_behavior_alert"
            if decision_source == "fusion_blend" and not (prob >= threshold) and not novelty_decision:
                alert_type = "fusion_behavior_alert"
            elif not (prob >= threshold):
                alert_type = "novelty_behavior_alert"
            alert = {
                "type": alert_type,
                "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "sessionId": session_id,
                "score": prob,
                "fusionMode": fusion_mode,
                "fusionEnabled": fusion_enabled,
                "fusionAlpha": fusion_alpha if fusion_mode == "blend" else None,
                "fusionScore": fusion_score,
                "fusionThreshold": fusion_threshold,
                "fusionDecision": fusion_decision,
                "decisionSource": decision_source,
                "noveltyError": novelty_error,
                "noveltySignal": novelty_signal,
                "noveltyDecision": novelty_decision,
                "hybridRisk": hybrid_risk,
                "severity": severity,
                "confidenceBand": confidence_band,
                "threshold": threshold,
                "noveltyThreshold": novelty_threshold,
                "windowSize": window_size,
                "topFactors": top_factors,
            }
            alerts.append(alert)

    if not alerts:
        print("No LSTM alerts right now.")
        return

    alerts_path = Path(args.alerts_out)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    with alerts_path.open("a", encoding="utf8") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")

    print(f"Alerts generated: {len(alerts)}")


if __name__ == "__main__":
    main()
