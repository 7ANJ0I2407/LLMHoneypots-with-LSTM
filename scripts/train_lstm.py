import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, num_layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.classifier(last).squeeze(1)
        return logits


def split_data_random(X, y, val_ratio=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    split = int(len(idx) * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def split_data_time(X, y, end_ts, val_ratio=0.2):
    idx = np.argsort(end_ts)
    split = int(len(idx) * (1 - val_ratio))

    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(batch_y.numpy().tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true, dtype=np.float32)
    y_prob = np.array(y_prob, dtype=np.float32)

    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= thr).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    y_pred = (y_prob >= best_thr).astype(np.int32)
    acc = float((y_pred == y_true).mean())

    return {"accuracy": acc, "f1": float(best_f1), "threshold": best_thr}


def main():
    parser = argparse.ArgumentParser(description="Train LSTM attack behavior detector")
    parser.add_argument("--dataset", default="data/sequences.npz", help="Path to preprocessed NPZ")
    parser.add_argument("--model-out", default="models/lstm_detector.pt", help="Path to save model")
    parser.add_argument("--meta-out", default="models/model_meta.json", help="Path to save metadata")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    end_ts = data["end_ts"].astype(np.int64) if "end_ts" in data.files else None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(X) < 20:
        raise SystemExit("Not enough training sequences. Collect more data first.")

    if len(X) < 50:
        print("Warning: low sample count. Model quality may be unstable.")

    if args.split_mode == "time" and end_ts is not None:
        X_train, y_train, X_val, y_val = split_data_time(X, y, end_ts, val_ratio=0.2)
    else:
        X_train, y_train, X_val, y_val = split_data_random(X, y, val_ratio=0.2)

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMDetector(input_dim=X.shape[-1]).to(device)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        print(f"epoch={epoch} loss={epoch_loss / max(len(train_loader), 1):.4f}")

    metrics = evaluate(model, val_loader, device)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X.shape[-1],
            "window_size": int(data["window_size"][0]),
        },
        model_out,
    )

    meta = {
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "samples": int(len(X)),
        "positive_samples": int(y.sum()),
        "split_mode": args.split_mode if end_ts is not None else "random",
    }

    with Path(args.meta_out).open("w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)

    print("Validation metrics:", meta)
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()
