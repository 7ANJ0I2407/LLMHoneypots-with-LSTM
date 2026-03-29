import json
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "prompt_length",
    "message_count",
    "signal_score",
    "latency_ms",
    "temperature",
    "max_tokens",
    "inter_arrival_sec",
    "signal_hit_count",
    "hour_sin",
    "hour_cos",
]


def _should_flip_label(session_id: str, timestamp: str, base_prob: float) -> bool:
    if base_prob <= 0.0:
        return False
    key = f"{session_id}|{timestamp}".encode("utf8")
    digest = hashlib.sha256(key).hexdigest()
    val = int(digest[:8], 16) / 0xFFFFFFFF
    return val < base_prob


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_events(jsonl_path: Path):
    events = []
    if not jsonl_path.exists():
        return events

    with jsonl_path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def events_to_grouped_features(events):
    grouped = defaultdict(list)

    for event in events:
        timestamp = event.get("timestamp")
        session_id = event.get("sessionId") or event.get("ip", "unknown")
        if not timestamp:
            continue

        grouped[session_id].append(event)

    output = {}
    for session_id, records in grouped.items():
        records.sort(key=lambda e: e.get("timestamp", ""))

        prev_ts = None
        rows = []
        labels = []
        ts_ms = []

        for rec in records:
            current_ts = parse_iso(rec["timestamp"])
            inter_arrival_sec = 0.0 if prev_ts is None else max(0.0, (current_ts - prev_ts).total_seconds())
            prev_ts = current_ts

            hour_float = current_ts.hour + (current_ts.minute / 60.0)
            angle = 2.0 * np.pi * hour_float / 24.0

            signal_hits = rec.get("signalHits") or []
            signal_score = float(rec.get("signalScore", 0.0))

            vector = [
                float(rec.get("promptLength", 0.0)),
                float(rec.get("messageCount", 0.0)),
                signal_score,
                float(rec.get("latencyMs", 0.0)),
                float(rec.get("temperature", 1.0)),
                float(rec.get("maxTokens", 0.0)),
                float(inter_arrival_sec),
                float(len(signal_hits)),
                float(np.sin(angle)),
                float(np.cos(angle)),
            ]

            # Weak labels with deterministic noise to avoid unrealistically clean classes.
            label = 1 if signal_score >= 2.0 or len(signal_hits) >= 3 else 0

            if label == 1:
                # Strong positives are usually correct; borderline positives are noisier.
                if signal_score < 3.0 and len(signal_hits) < 4 and _should_flip_label(session_id, rec["timestamp"], 0.10):
                    label = 0
            else:
                # A small false-negative/false-positive floor keeps the dataset realistic.
                if signal_score >= 1.0 and _should_flip_label(session_id, rec["timestamp"], 0.08):
                    label = 1
                elif _should_flip_label(session_id, rec["timestamp"], 0.03):
                    label = 1

            rows.append(vector)
            labels.append(label)
            ts_ms.append(int(current_ts.timestamp() * 1000))

        output[session_id] = {
            "X": np.array(rows, dtype=np.float32),
            "y": np.array(labels, dtype=np.int64),
            "ts": np.array(ts_ms, dtype=np.int64),
        }

    return output


def fit_standardizer(X: np.ndarray):
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def save_standardizer(path: Path, mean: np.ndarray, std: np.ndarray):
    payload = {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "feature_names": FEATURE_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)


def load_standardizer(path: Path):
    with path.open("r", encoding="utf8") as f:
        payload = json.load(f)

    mean = np.array(payload["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.array(payload["std"], dtype=np.float32).reshape(1, 1, -1)
    feature_names = payload["feature_names"]
    return mean, std, feature_names
