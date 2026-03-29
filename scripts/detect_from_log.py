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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def explain_window(window: np.ndarray, lstm_score: float, threshold: float):
    latest = window[-1]
    mean_signal = float(np.mean(window[:, 2]))
    mean_interarrival = float(np.mean(window[:, 6]))
    max_prompt = float(np.max(window[:, 0]))

    rule_proxy = clamp01(latest[2] / 4.0)
    velocity_proxy = clamp01(1.0 - min(mean_interarrival, 10.0) / 10.0)
    hybrid_risk = clamp01(0.55 * lstm_score + 0.35 * rule_proxy + 0.10 * velocity_proxy)

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
    parser.add_argument("--meta", default="models/model_meta.json")
    parser.add_argument("--scaler", default="models/scaler.json")
    parser.add_argument("--alerts-out", default="logs/alerts.jsonl")

    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    scaler_path = Path(args.scaler)

    if not model_path.exists() or not meta_path.exists() or not scaler_path.exists():
        raise SystemExit("Missing model files. Run preprocess.py and train_lstm.py first.")

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.5)))

    events = load_events(Path(args.input))
    grouped = events_to_grouped_features(events)

    mean, std, _feature_names = load_standardizer(scaler_path)
    model, window_size = load_model(model_path)

    alerts = []

    for session_id, data in grouped.items():
        X = data["X"]
        if len(X) < window_size:
            continue

        window = X[-window_size:][None, ...].astype(np.float32)
        window_scaled = apply_standardizer(window, mean, std)

        with torch.no_grad():
            logits = model(torch.from_numpy(window_scaled))
            prob = float(torch.sigmoid(logits).numpy()[0])

        if prob >= threshold:
            hybrid_risk, severity, confidence_band, top_factors = explain_window(
                window[0],
                prob,
                threshold,
            )
            alert = {
                "type": "lstm_behavior_alert",
                "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "sessionId": session_id,
                "score": prob,
                "hybridRisk": hybrid_risk,
                "severity": severity,
                "confidenceBand": confidence_band,
                "threshold": threshold,
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
