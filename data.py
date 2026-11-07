import os
from pathlib import Path

import numpy as np
import torch


def read_idx(path):
    raw = path.read_bytes()
    dimensions = int.from_bytes(raw[:4], "big") & 0xFF
    if dimensions == 3:
        count, rows, cols = (int.from_bytes(raw[4 + i * 4 : 8 + i * 4], "big") for i in range(3))
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows * cols).copy()
    if dimensions == 1:
        count = int.from_bytes(raw[4:8], "big")
        return np.frombuffer(raw, dtype=np.uint8, offset=8, count=count).copy()
    raise ValueError(f"Unsupported IDX file: {path}")


def stratified_split(images, labels, validation_split=0.2, seed=23):
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for class_idx in range(10):
        indices = np.flatnonzero(labels == class_idx)
        rng.shuffle(indices)
        count = int(round(len(indices) * validation_split))
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return images[train_indices], labels[train_indices], images[validation_indices], labels[validation_indices]


def full_mnist_data():
    root = Path(os.environ.get("MNIST_DATA_DIR", Path(__file__).resolve().parent / "data" / "MNIST" / "raw"))
    train_x = read_idx(root / "train-images-idx3-ubyte").astype(np.float32) / 255.0
    train_y = read_idx(root / "train-labels-idx1-ubyte")
    test_x = read_idx(root / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0
    test_y = read_idx(root / "t10k-labels-idx1-ubyte")
    train_x, train_y, val_x, val_y = stratified_split(train_x, train_y, seed=42)
    train_x = np.concatenate([train_x, val_x])
    train_y = np.concatenate([train_y, val_y])
    return (
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).long(),
        torch.from_numpy(test_x).float(),
        torch.from_numpy(test_y).long(),
        torch.from_numpy(test_x).float(),
        torch.from_numpy(test_y).long(),
    )
