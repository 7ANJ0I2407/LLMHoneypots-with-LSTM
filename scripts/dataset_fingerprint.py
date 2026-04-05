import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Print dataset reproducibility fingerprint")
    parser.add_argument("--dataset", default="data/sequences.npz")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    X = np.asarray(data["X"]) if "X" in data.files else np.empty((0,))
    y = np.asarray(data["y"]) if "y" in data.files else np.empty((0,))
    end_ts = np.asarray(data["end_ts"]) if "end_ts" in data.files else np.empty((0,))

    print("=== DATASET FINGERPRINT ===")
    print(f"dataset={dataset_path}")
    print(f"sha256={sha256_file(dataset_path)}")
    print(f"samples={len(y)}")
    print(f"positives={int(y.sum()) if len(y) else 0}")
    print(f"shape_X={tuple(X.shape)}")
    print(f"has_end_ts={bool(len(end_ts))}")


if __name__ == "__main__":
    main()
