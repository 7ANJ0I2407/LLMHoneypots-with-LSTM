import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autoencoder_model import SequenceAutoencoder, reconstruction_error


class WindowDataset(Dataset):
    def __init__(self, X, indices=None):
        self.X = np.asarray(X, dtype=np.float32)
        if indices is None:
            self.indices = np.arange(len(self.X), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        return torch.from_numpy(self.X[i])


def split_train_val(indices, val_ratio=0.2, seed=42):
    idx = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    return tr_idx, va_idx


def main():
    parser = argparse.ArgumentParser(description="Train novelty autoencoder on normal windows")
    parser.add_argument("--dataset", default="data/sequences_large.npz")
    parser.add_argument("--meta", default="models/model_meta_large.json")
    parser.add_argument("--model-out", default="models/novelty_autoencoder_large.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile", type=float, default=0.97)
    parser.add_argument("--proj-dim", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    window_size = int(data["window_size"][0])

    normal_idx = np.flatnonzero(y == 0)
    if len(normal_idx) < 80:
        raise SystemExit("Not enough normal windows to train novelty model.")

    train_idx, val_idx = split_train_val(normal_idx, val_ratio=0.2, seed=args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceAutoencoder(
        window_size=window_size,
        input_dim=X.shape[-1],
        proj_dim=args.proj_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(WindowDataset(X, train_idx), batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        print(f"epoch={epoch} recon_loss={loss_sum / max(len(train_loader), 1):.6f}")

    model.eval()
    val_loader = DataLoader(WindowDataset(X, val_idx), batch_size=args.batch_size, shuffle=False)
    err_chunks = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            val_recon = model(batch)
            err_chunks.append(reconstruction_error(batch, val_recon).cpu().numpy())
    val_err = np.concatenate(err_chunks, axis=0) if err_chunks else np.empty((0,), dtype=np.float32)

    novelty_threshold = float(np.quantile(val_err, args.quantile))

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "window_size": window_size,
            "input_dim": int(X.shape[-1]),
            "quantile": float(args.quantile),
            "proj_dim": int(args.proj_dim),
            "latent_dim": int(args.latent_dim),
            "dropout": float(args.dropout),
        },
        model_out,
    )

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text(encoding="utf8")) if meta_path.exists() else {}
    meta["novelty_threshold"] = novelty_threshold
    meta["novelty_threshold_train"] = novelty_threshold
    meta["novelty_model"] = str(model_out)
    meta["novelty_training"] = {
        "normal_samples": int(len(normal_idx)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "threshold_quantile": float(args.quantile),
        "proj_dim": int(args.proj_dim),
        "latent_dim": int(args.latent_dim),
        "dropout": float(args.dropout),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")

    print(
        "novelty_model_saved="
        f"{model_out} novelty_threshold={novelty_threshold:.6f} "
        f"normal_samples={len(normal_idx)}"
    )


if __name__ == "__main__":
    main()
