import json
import hashlib
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from embedding_utils import encode_texts

BASE_FEATURE_NAMES = [
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

SESSION_FEATURE_WINDOW = 5

EXTRA_FEATURE_NAMES = [
    "prompt_similarity_to_session_mean",
    "distinct_models_in_window",
    "prompt_char_entropy",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES

EMBEDDING_MODE = os.environ.get("HONEYPOT_EMBEDDING_MODE", "sbert").strip().lower()
if os.environ.get("DISABLE_SBERT") == "1":
    EMBEDDING_MODE = "off"


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


def extract_prompt_text(record):
    text = record.get("promptText") or record.get("promptPreview") or ""
    return str(text)


def prompt_char_entropy(text: str) -> float:
    text = str(text or "")
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def compute_session_features(records, window_size: int = SESSION_FEATURE_WINDOW):
    if not records:
        return []

    texts = [extract_prompt_text(rec) for rec in records]
    entropies = [prompt_char_entropy(text) for text in texts]

    if EMBEDDING_MODE == "sbert":
        try:
            embeddings = encode_texts(texts)
        except Exception:
            embeddings = np.zeros((len(texts), 0), dtype=np.float32)
    else:
        embeddings = np.zeros((len(texts), 0), dtype=np.float32)

    similarities = []
    distinct_models = []

    for idx, rec in enumerate(records):
        start = max(0, idx - window_size)
        prev = embeddings[start:idx] if embeddings.size else np.empty((0, 0), dtype=np.float32)

        if prev.size == 0:
            sim = 0.0
        else:
            mean_vec = prev.mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm <= 1e-8:
                sim = 0.0
            else:
                sim = float(np.dot(embeddings[idx], mean_vec) / norm)
        sim = float(np.clip(sim, -1.0, 1.0))
        similarities.append(sim)

        window_start = max(0, idx - window_size + 1)
        model_names = {
            str(r.get("model") or "unknown")
            for r in records[window_start: idx + 1]
        }
        distinct_models.append(float(len(model_names)))

    return [
        {
            "promptSimilarityToSessionMean": similarities[i],
            "distinctModelsInWindow": distinct_models[i],
            "promptCharEntropy": entropies[i],
        }
        for i in range(len(records))
    ]


def build_base_feature_vector(record, inter_arrival_sec, hour_sin=0.0, hour_cos=0.0):
    signal_hits = record.get("signalHits") or []
    signal_score = float(record.get("signalScore", 0.0))

    return [
        float(record.get("promptLength", 0.0)),
        float(record.get("messageCount", 0.0)),
        signal_score,
        float(record.get("latencyMs", 0.0)),
        float(record.get("temperature", 1.0)),
        float(record.get("maxTokens", 0.0)),
        float(inter_arrival_sec),
        float(len(signal_hits)),
        float(hour_sin),
        float(hour_cos),
    ]


def build_feature_vector(record, inter_arrival_sec, hour_sin=0.0, hour_cos=0.0):
    base = build_base_feature_vector(
        record,
        inter_arrival_sec,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
    )
    prompt_similarity = float(record.get("promptSimilarityToSessionMean", 0.0) or 0.0)
    distinct_models = float(record.get("distinctModelsInWindow", 0.0) or 0.0)
    prompt_entropy = float(record.get("promptCharEntropy", 0.0) or 0.0)
    return base + [prompt_similarity, distinct_models, prompt_entropy]


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
        inter_arrivals = []
        angles = []

        for rec in records:
            current_ts = parse_iso(rec["timestamp"])
            inter_arrival_sec = 0.0 if prev_ts is None else max(0.0, (current_ts - prev_ts).total_seconds())
            prev_ts = current_ts

            hour_float = current_ts.hour + (current_ts.minute / 60.0)
            angle = 2.0 * np.pi * hour_float / 24.0
            inter_arrivals.append(inter_arrival_sec)
            angles.append(angle)

            explicit_label = str(rec.get("label", "")).strip().lower()
            if explicit_label in {"attack", "benign"}:
                label = 1 if explicit_label == "attack" else 0
            else:
                # Weak labels with deterministic noise to avoid unrealistically clean classes
                # only when explicit labels are unavailable.
                signal_hits = rec.get("signalHits") or []
                signal_score = float(rec.get("signalScore", 0.0))
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

            labels.append(label)
            ts_ms.append(int(current_ts.timestamp() * 1000))

        session_features = compute_session_features(records)
        rows = [
            build_feature_vector(
                {**rec, **extra},
                inter_arrival_sec,
                hour_sin=np.sin(angle),
                hour_cos=np.cos(angle),
            )
            for rec, extra, inter_arrival_sec, angle in zip(records, session_features, inter_arrivals, angles)
        ]

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
    flat_mean = mean.reshape(-1).tolist()
    flat_std = std.reshape(-1).tolist()
    payload = {
        "mean": flat_mean,
        "std": flat_std,
        # Keep legacy key for backward compatibility with older artifacts/tools.
        "scale": flat_std,
        "feature_names": FEATURE_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)


def load_standardizer(path: Path):
    with path.open("r", encoding="utf8") as f:
        payload = json.load(f)

    if "mean" not in payload:
        raise KeyError("scaler payload missing 'mean'")

    std_values = payload.get("std")
    if std_values is None:
        std_values = payload.get("scale")
    if std_values is None:
        raise KeyError("scaler payload missing both 'std' and legacy 'scale'")

    mean = np.array(payload["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std_values, dtype=np.float32).reshape(1, 1, -1)
    std[np.abs(std) < 1e-6] = 1.0
    feature_names = payload.get("feature_names") or [f"f_{i}" for i in range(mean.shape[-1])]
    return mean, std, feature_names
