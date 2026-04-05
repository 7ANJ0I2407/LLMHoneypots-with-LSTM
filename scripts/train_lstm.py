import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X, y, indices=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        if indices is None:
            self.indices = np.arange(len(self.X), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32)


class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, num_layers=2, dropout=0.35):
        super().__init__()
        conv_channels = 64
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bigru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Conv1D extracts local temporal patterns before recurrent modeling.
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)

        out, _ = self.bilstm(x)
        out, _ = self.bigru(out)
        last = out[:, -1, :]
        logits = self.classifier(last).squeeze(1)
        return logits


def split_data_random(X, y, val_ratio=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    split = int(len(idx) * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    return train_idx, val_idx


def split_data_time(X, y, end_ts, val_ratio=0.2):
    y = np.asarray(y, dtype=np.int64)
    end_ts = np.asarray(end_ts, dtype=np.int64)

    def split_class(label):
        cls_idx = np.flatnonzero(y == label)
        if len(cls_idx) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        cls_sorted = cls_idx[np.argsort(end_ts[cls_idx])]
        val_n = max(1, int(round(len(cls_sorted) * val_ratio)))
        val_n = min(val_n, len(cls_sorted))
        return cls_sorted[:-val_n], cls_sorted[-val_n:]

    tr0, va0 = split_class(0)
    tr1, va1 = split_class(1)

    train_idx = np.concatenate([tr0, tr1]).astype(np.int64)
    val_idx = np.concatenate([va0, va1]).astype(np.int64)

    train_idx = train_idx[np.argsort(end_ts[train_idx])]
    val_idx = val_idx[np.argsort(end_ts[val_idx])]
    return train_idx, val_idx


def collect_logits(model, loader, device):
    model.eval()
    logits_out, y_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X).detach().cpu().numpy()
            logits_out.append(logits)
            y_true.append(batch_y.numpy())

    if not logits_out:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.concatenate(logits_out, axis=0), np.concatenate(y_true, axis=0)


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = float(temperature)
    if temp <= 1e-6:
        temp = 1.0
    logits_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32)) / temp
    return torch.sigmoid(logits_tensor).numpy()


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    if len(probs) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lower = edges[i]
        upper = edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (float(mask.mean()) * abs(acc - conf))
    return float(ece)


def temperature_nll(logits: np.ndarray, y_true: np.ndarray, temperature: float) -> float:
    temp = float(temperature)
    if temp <= 1e-6:
        temp = 1.0
    logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32)) / temp
    labels = torch.from_numpy(np.asarray(y_true, dtype=np.float32))
    loss = F.binary_cross_entropy_with_logits(logits_t, labels)
    return float(loss.item())


def find_temperature(logits: np.ndarray, y_true: np.ndarray, min_temp: float = 0.05, max_temp: float = 6.0, steps: int = 120):
    temps = np.linspace(min_temp, max_temp, steps)
    base_probs = probs_from_logits(logits, 1.0)
    best = {
        "temperature": 1.0,
        "nll": temperature_nll(logits, y_true, 1.0),
        "ece": expected_calibration_error(base_probs, y_true),
    }
    for temp in temps:
        nll = temperature_nll(logits, y_true, temp)
        probs = probs_from_logits(logits, temp)
        ece = expected_calibration_error(probs, y_true)
        if nll < best["nll"]:
            best = {"temperature": float(temp), "nll": float(nll), "ece": float(ece)}
    return best


def best_f1_threshold(probs: np.ndarray, y_true: np.ndarray, steps: int = 181):
    y_true = np.asarray(y_true, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    best = {"f1": -1.0, "threshold": 0.5, "accuracy": 0.0}
    for thr in np.linspace(0.05, 0.95, steps):
        y_pred = (probs >= thr).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        acc = float((y_pred == y_true).mean())

        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(thr), "accuracy": acc}
    return best


def metrics_from_threshold(probs: np.ndarray, y_true: np.ndarray, threshold: float):
    y_true = np.asarray(y_true, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    y_pred = (probs >= float(threshold)).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((y_pred == y_true).mean())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"accuracy": acc, "f1": float(f1)}


def main():
    parser = argparse.ArgumentParser(description="Train LSTM attack behavior detector")
    parser.add_argument("--dataset", default="data/sequences_large.npz", help="Path to preprocessed NPZ")
    parser.add_argument("--model-out", default="models/lstm_detector_large.pt", help="Path to save model")
    parser.add_argument("--meta-out", default="models/model_meta_large.json", help="Path to save metadata")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    end_ts = np.asarray(data["end_ts"], dtype=np.int64) if "end_ts" in data.files else None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(X) < 20:
        raise SystemExit("Not enough training sequences. Collect more data first.")

    if len(X) < 50:
        print("Warning: low sample count. Model quality may be unstable.")

    if args.split_mode == "time" and end_ts is not None:
        train_idx, val_idx = split_data_time(X, y, end_ts, val_ratio=0.2)
    else:
        train_idx, val_idx = split_data_random(X, y, val_ratio=0.2)

    train_loader = DataLoader(SequenceDataset(X, y, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X, y, val_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMDetector(input_dim=X.shape[-1], dropout=args.dropout).to(device)

    y_train = y[train_idx]
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            if args.label_smoothing > 0:
                smooth = float(args.label_smoothing)
                batch_y = batch_y * (1.0 - smooth) + 0.5 * smooth
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        print(f"epoch={epoch} loss={epoch_loss / max(len(train_loader), 1):.4f}")

    logits, y_val = collect_logits(model, val_loader, device)
    temp_result = find_temperature(logits, y_val)
    temperature_scale = float(temp_result["temperature"])
    calibration_ece = float(temp_result["ece"])
    calibration_nll = float(temp_result["nll"])
    probs = probs_from_logits(logits, temperature_scale)
    metrics = best_f1_threshold(probs, y_val)

    train_logits, y_train_eval = collect_logits(model, train_loader, device)
    train_probs = probs_from_logits(train_logits, temperature_scale)
    train_metrics = metrics_from_threshold(train_probs, y_train_eval, metrics["threshold"])

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
        "temperature_scale": temperature_scale,
        "calibration_ece": calibration_ece,
        "calibration_nll": calibration_nll,
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_f1": float(train_metrics["f1"]),
        "train_epochs": int(args.epochs),
        "train_dropout": float(args.dropout),
        "train_label_smoothing": float(args.label_smoothing),
        "train_weight_decay": float(args.weight_decay),
    }

    with Path(args.meta_out).open("w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)

    print("Validation metrics:", meta)
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()
