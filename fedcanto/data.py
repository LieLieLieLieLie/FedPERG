from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class FederatedData:
    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    client_train: List[np.ndarray]
    client_test: List[np.ndarray]
    calibration_indices: np.ndarray
    classes: int
    input_dim: int
    dataset_name: str
    regime: str


def _dirichlet_partition(
    y: np.ndarray, clients: int, alpha: float, rng: np.random.Generator, min_size: int = 16
) -> List[np.ndarray]:
    classes = int(y.max()) + 1
    for _ in range(80):
        buckets: List[List[int]] = [[] for _ in range(clients)]
        for label in range(classes):
            idx = np.flatnonzero(y == label)
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            for cid, part in enumerate(np.split(idx, cuts)):
                buckets[cid].extend(part.tolist())
        if min(map(len, buckets)) >= min_size:
            return [np.asarray(rng.permutation(v), dtype=np.int64) for v in buckets]
    order = rng.permutation(len(y))
    return [np.asarray(v, dtype=np.int64) for v in np.array_split(order, clients)]


def _matched_test_partition(
    y_test: np.ndarray, train_parts: Sequence[np.ndarray], y_train: np.ndarray, rng: np.random.Generator
) -> List[np.ndarray]:
    clients = len(train_parts)
    classes = int(max(y_train.max(), y_test.max())) + 1
    buckets: List[List[int]] = [[] for _ in range(clients)]
    hist = np.stack(
        [np.bincount(y_train[p], minlength=classes).astype(np.float64) + 0.2 for p in train_parts]
    )
    for label in range(classes):
        idx = np.flatnonzero(y_test == label)
        rng.shuffle(idx)
        probs = hist[:, label] / hist[:, label].sum()
        assignment = rng.choice(clients, size=len(idx), p=probs)
        for cid in range(clients):
            buckets[cid].extend(idx[assignment == cid].tolist())
    missing = [i for i, b in enumerate(buckets) if not b]
    for cid in missing:
        donor = int(np.argmax([len(b) for b in buckets]))
        buckets[cid].append(buckets[donor].pop())
    return [np.asarray(v, dtype=np.int64) for v in buckets]


def _specialized_partition(
    y: np.ndarray, clients: int, specialization: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Balanced label-specialization partition with complete cohort coverage.

    Client c is the anchor for class c modulo the number of clients. At zero
    specialization every label is allocated uniformly; at one, each label is
    allocated only to its anchor. The rule is fixed before Round-8 execution.
    """
    specialization = float(np.clip(specialization, 0.0, 1.0))
    buckets: List[List[int]] = [[] for _ in range(clients)]
    for label in range(int(y.max()) + 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        probs = np.full(clients, (1.0 - specialization) / clients)
        probs[label % clients] += specialization
        assignment = rng.choice(clients, size=len(idx), p=probs)
        for cid in range(clients):
            buckets[cid].extend(idx[assignment == cid].tolist())
    if min(map(len, buckets)) == 0:
        raise RuntimeError("Specialization partition produced an empty client")
    return [np.asarray(rng.permutation(v), dtype=np.int64) for v in buckets]


def _load_arrays(data_dir: Path, dataset: str, seed: int):
    rng = np.random.default_rng(seed + 1907)
    if dataset == "synthetic_complementarity":
        classes, dimension = 6, 24
        directions = rng.normal(size=(classes, dimension))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        # A fixed signal ladder avoids a perfectly symmetric toy in which all
        # leave-one residuals are identical, while every class remains required
        # by the balanced population objective.
        centers = directions * np.linspace(2.0, 3.5, classes)[:, None]

        def sample(per_class: int) -> Tuple[np.ndarray, np.ndarray]:
            labels = np.repeat(np.arange(classes, dtype=np.int64), per_class)
            features = centers[labels] + rng.normal(0.0, 1.0,
                                                    size=(len(labels), dimension))
            order = rng.permutation(len(labels))
            return features[order].astype(np.float32), labels[order]

        x_train, y_train = sample(600)
        x_test, y_test = sample(200)
        return x_train, y_train, x_test, y_test
    if dataset == "mnist":
        raw = data_dir / "mnist_raw"

        if not raw.is_dir():
            from torchvision.datasets import MNIST

            root = data_dir / "mnist_torchvision"
            train = MNIST(root=str(root), train=True, download=False)
            test = MNIST(root=str(root), train=False, download=False)
            return (
                train.data.numpy()[:, None].astype(np.float32) / 255.0,
                train.targets.numpy().astype(np.int64),
                test.data.numpy()[:, None].astype(np.float32) / 255.0,
                test.targets.numpy().astype(np.int64),
            )

        def images(name: str) -> np.ndarray:
            with (raw / name).open("rb") as stream:
                magic, count, height, width = struct.unpack(">IIII", stream.read(16))
                if magic != 2051 or (height, width) != (28, 28):
                    raise ValueError(f"Invalid MNIST image file: {name}")
                pixels = np.frombuffer(stream.read(), dtype=np.uint8)
            if pixels.size != count * height * width:
                raise ValueError(f"Incomplete MNIST image file: {name}")
            return pixels.reshape(count, 1, height, width).astype(np.float32) / 255.0

        def labels(name: str) -> np.ndarray:
            with (raw / name).open("rb") as stream:
                magic, count = struct.unpack(">II", stream.read(8))
                if magic != 2049:
                    raise ValueError(f"Invalid MNIST label file: {name}")
                values = np.frombuffer(stream.read(), dtype=np.uint8).astype(np.int64)
            if values.size != count:
                raise ValueError(f"Incomplete MNIST label file: {name}")
            return values

        return (images("train-images-idx3-ubyte"), labels("train-labels-idx1-ubyte"),
                images("t10k-images-idx3-ubyte"), labels("t10k-labels-idx1-ubyte"))
    if dataset == "officehome_raw":
        z = np.load(data_dir / "officehome_raw_48.npz", allow_pickle=False)
        x, y = z["images"], z["labels"].astype(np.int64)
        domains = z["domains"].astype(np.int64)
        train_idx, test_idx = [], []
        for domain in np.unique(domains):
            for label in np.unique(y):
                idx = np.flatnonzero((domains == domain) & (y == label))
                if not len(idx):
                    continue
                rng.shuffle(idx)
                cut = max(1, int(0.72 * len(idx)))
                train_idx.extend(idx[:cut].tolist())
                test_idx.extend(idx[cut:].tolist())
        return (x[train_idx].astype(np.float32) / 255.0, y[train_idx],
                x[test_idx].astype(np.float32) / 255.0, y[test_idx])
    if dataset == "cifar10_raw":
        from torchvision.datasets import CIFAR10

        root = data_dir / "cifar10_torchvision"
        train = CIFAR10(root=str(root), train=True, download=False)
        test = CIFAR10(root=str(root), train=False, download=False)
        x_train = np.asarray(train.data).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        x_test = np.asarray(test.data).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return x_train, np.asarray(train.targets, dtype=np.int64), x_test, np.asarray(test.targets, dtype=np.int64)
    if dataset in {"mnist_mobilenet", "mnist_mobilenet_specialization"}:
        z = np.load(data_dir / "mnist_mobilenet_v3_64.npz", allow_pickle=False)
        x_train = z["x_train"].astype(np.float32)
        y_train = z["y_train"].astype(np.int64)
        x_test = z["x_test"].astype(np.float32)
        y_test = z["y_test"].astype(np.int64)
    elif dataset == "eurosat_resnet18":
        z = np.load(data_dir / "eurosat_resnet18.npz", allow_pickle=False)
        x_train = z["x_train"].astype(np.float32)
        y_train = z["y_train"].astype(np.int64)
        x_test = z["x_test"].astype(np.float32)
        y_test = z["y_test"].astype(np.int64)
    elif dataset in {"cifar10", "cifar100"}:
        path = data_dir / f"{dataset}_resnet18_ftta_v1.npz"
        z = np.load(path, allow_pickle=True)
        x_train = np.concatenate([z["x_source"], z["x_cal"]]).astype(np.float32)
        y_train = np.concatenate([z["y_source"], z["y_cal"]]).astype(np.int64)
        # Episode zero is the clean target split in the cached extraction.
        x_test = z["x_target"][0].astype(np.float32)
        y_test = z["y_target"].astype(np.int64)
    elif dataset == "officehome":
        z = np.load(data_dir / "officehome65_resnet18.npz", allow_pickle=True)
        x, y = z["features"].astype(np.float32), z["labels"].astype(np.int64)
        domains = z["domains"].astype(np.int64)
        train_idx, test_idx = [], []
        for domain in np.unique(domains):
            for label in np.unique(y):
                idx = np.flatnonzero((domains == domain) & (y == label))
                if not len(idx):
                    continue
                rng.shuffle(idx)
                cut = max(1, int(0.72 * len(idx)))
                train_idx.extend(idx[:cut].tolist())
                test_idx.extend(idx[cut:].tolist())
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-5
    x_train = np.clip((x_train - mean) / std, -8.0, 8.0)
    x_test = np.clip((x_test - mean) / std, -8.0, 8.0)
    return x_train, y_train, x_test, y_test


def load_federated_data(
    data_dir: str, dataset: str, regime: str, clients: int, alpha: float, seed: int,
    calibration_fraction: float, specialization: float = 0.0,
    calibration_shift: str = "none",
) -> FederatedData:
    rng = np.random.default_rng(seed + 31)
    x_train, y_train, x_test, y_test = _load_arrays(Path(data_dir), dataset, seed)
    if dataset in {"synthetic_complementarity", "mnist_mobilenet_specialization"}:
        train_parts = _specialized_partition(y_train, clients, specialization, rng)
        test_parts = _specialized_partition(y_test, clients, specialization,
                                            np.random.default_rng(seed + 7331))
    else:
        effective_alpha = {"label_skew": 0.08, "quantity_skew": 0.6,
                           "compound": alpha}[regime]
        train_parts = _dirichlet_partition(y_train, clients, effective_alpha, rng)
    if regime in {"quantity_skew", "compound"}:
        factors = np.exp(rng.normal(0.0, 0.85, size=clients))
        factors /= factors.max()
        clipped = []
        for cid, idx in enumerate(train_parts):
            keep = max(20, int(len(idx) * (0.28 + 0.72 * factors[cid])))
            clipped.append(idx[: min(keep, len(idx))])
        train_parts = clipped
    if dataset != "synthetic_complementarity":
        test_parts = _matched_test_partition(y_test, train_parts, y_train, rng)

    # This held-out server calibration split is removed from every client.
    calibration_size = max(32, int(len(y_train) * calibration_fraction))
    if calibration_shift == "none":
        calibration_indices = rng.choice(len(y_train), size=calibration_size, replace=False)
    elif calibration_shift == "low_label_bias":
        # Same pool size but an intentionally mismatched label distribution:
        # 80% of calibration examples come from the lower half of classes.
        split = (int(y_train.max()) + 2) // 2
        low = np.flatnonzero(y_train < split)
        high = np.flatnonzero(y_train >= split)
        low_count = min(len(low), int(round(0.8 * calibration_size)))
        high_count = min(len(high), calibration_size - low_count)
        if low_count + high_count < calibration_size:
            low_count = min(len(low), calibration_size - high_count)
        calibration_indices = np.concatenate([
            rng.choice(low, size=low_count, replace=False),
            rng.choice(high, size=high_count, replace=False),
        ])
        rng.shuffle(calibration_indices)
    else:
        raise ValueError(f"Unknown server calibration shift: {calibration_shift}")

    if regime == "compound":
        # Deterministic client-specific sensor distortions and sparse label faults.
        x_mod = x_train.copy()
        y_mod = y_train.copy()
        for cid, idx in enumerate(train_parts):
            scale = 0.88 + 0.24 * ((cid * 37) % 11) / 10.0
            if x_mod.ndim == 2:
                bias = 0.12 * np.sin(np.arange(x_mod.shape[1]) * (cid + 1) * 0.017)
                x_mod[idx] = x_mod[idx] * scale + bias.astype(np.float32)
            else:
                x_mod[idx] = np.clip(x_mod[idx] * scale, 0.0, 1.0)
            if cid % 7 == 0:
                corrupt = idx[: max(1, int(0.18 * len(idx)))]
                y_mod[corrupt] = (y_mod[corrupt] + 1 + cid) % (int(y_train.max()) + 1)
        x_train, y_train = x_mod, y_mod

    return FederatedData(
        x_train=torch.from_numpy(x_train),
        y_train=torch.from_numpy(y_train),
        x_test=torch.from_numpy(x_test),
        y_test=torch.from_numpy(y_test),
        client_train=train_parts,
        client_test=test_parts,
        calibration_indices=calibration_indices.astype(np.int64),
        classes=int(max(y_train.max(), y_test.max())) + 1,
        input_dim=int(x_train.shape[1]),
        dataset_name=dataset,
        regime=regime,
    )
