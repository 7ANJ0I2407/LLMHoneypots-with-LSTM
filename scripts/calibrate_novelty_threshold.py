import argparse
import json
from pathlib import Path

import numpy as np
import torch

from autoencoder_model import SequenceAutoencoder, reconstruction_error


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


def main():
    parser = argparse.ArgumentParser(description="Calibrate novelty threshold from existing model and dataset")
    parser.add_argument("--dataset", default="data/sequences.npz")
    parser.add_argument("--novelty-model", default="models/novelty_autoencoder.pt")
    parser.add_argument("--meta", default="models/model_meta.json")
    parser.add_argument("--quantile", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--z-clip", type=float, default=3.0)
    args = parser.parse_args()

    data_path = Path(args.dataset)
    model_path = Path(args.novelty_model)
    meta_path = Path(args.meta)
    if not data_path.exists() or not model_path.exists() or not meta_path.exists():
        raise SystemExit("Missing dataset/model/meta for novelty calibration.")

    data = np.load(data_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)

    neg_idx = np.flatnonzero(y == 0)
    if len(neg_idx) < 100:
        raise SystemExit("Not enough negative windows for novelty calibration.")

    Xn = X[neg_idx]
    if args.z_clip > 0:
        Xn = np.clip(Xn, -args.z_clip, args.z_clip)

    model = load_novelty_model(model_path)

    err_chunks = []
    with torch.no_grad():
        for start in range(0, len(Xn), args.batch_size):
            end = min(start + args.batch_size, len(Xn))
            batch = torch.from_numpy(Xn[start:end])
            recon = model(batch)
            err_chunks.append(reconstruction_error(batch, recon).numpy())

    errs = np.concatenate(err_chunks, axis=0) if err_chunks else np.empty((0,), dtype=np.float32)
    if len(errs) == 0:
        raise SystemExit("No novelty errors were computed.")

    threshold = float(np.quantile(errs, args.quantile))

    meta = json.loads(meta_path.read_text(encoding="utf8"))
    meta["novelty_threshold"] = threshold
    novelty_training = meta.get("novelty_training") or {}
    novelty_training["threshold_quantile"] = float(args.quantile)
    novelty_training["calibrated_on_negatives"] = int(len(Xn))
    novelty_training["z_clip"] = float(args.z_clip)
    novelty_training["error_q95"] = float(np.quantile(errs, 0.95))
    novelty_training["error_q99"] = float(np.quantile(errs, 0.99))
    meta["novelty_training"] = novelty_training
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")

    print(
        "novelty_threshold_calibrated="
        f"{threshold:.6f} quantile={args.quantile:.4f} negatives={len(Xn)} "
        f"error_q95={np.quantile(errs,0.95):.6f} error_q99={np.quantile(errs,0.99):.6f}"
    )


if __name__ == "__main__":
    main()
