import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml_utils import apply_standardizer, load_standardizer
from train_lstm import LSTMDetector


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = LSTMDetector(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, int(checkpoint["window_size"])


def main():
    parser = argparse.ArgumentParser(description="Score a single session window for online detection")
    parser.add_argument("--model", default="models/lstm_detector.pt")
    parser.add_argument("--meta", default="models/model_meta.json")
    parser.add_argument("--scaler", default="models/scaler.json")
    args = parser.parse_args()

    try:
        payload = json.load(__import__("sys").stdin)
    except json.JSONDecodeError:
        print(json.dumps({"ok": False, "error": "invalid_json_input"}))
        return

    window = payload.get("window")
    if not isinstance(window, list) or not window:
        print(json.dumps({"ok": False, "error": "window_missing"}))
        return

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    scaler_path = Path(args.scaler)

    if not (model_path.exists() and meta_path.exists() and scaler_path.exists()):
        print(json.dumps({"ok": False, "error": "model_files_missing"}))
        return

    with meta_path.open("r", encoding="utf8") as f:
        meta = json.load(f)

    threshold = float(meta.get("runtime_threshold", meta.get("threshold", 0.5)))
    mean, std, _ = load_standardizer(scaler_path)
    model, window_size = load_model(model_path)

    X = np.array(window, dtype=np.float32)
    if X.ndim != 2:
        print(json.dumps({"ok": False, "error": "window_shape_invalid"}))
        return

    X = X[None, ...]
    X_scaled = apply_standardizer(X, mean, std)

    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled))
        score = float(torch.sigmoid(logits).numpy()[0])

    print(
        json.dumps(
            {
                "ok": True,
                "score": score,
                "threshold": threshold,
                "windowSize": window_size,
                "decision": bool(score >= threshold),
            }
        )
    )


if __name__ == "__main__":
    main()
