"""Download public datasets and build the frozen feature caches used by FedCANTO.

CIFAR-10, CIFAR-100, and MNIST are downloaded through torchvision. Office-Home
must be downloaded from its official site and extracted locally because its
license does not permit this repository to redistribute the images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    resnet18,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OFFICE_DOMAINS = ("Art", "Clipart", "Product", "Real World")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CIFAR_EPISODES = (
    ("gaussian_noise", "gaussian_noise", 1),
    ("motion_blur", "motion_blur", 2),
    ("contrast", "contrast", 3),
    ("brightness", "brightness", 4),
    ("pixelate", "pixelate", 5),
    ("jpeg", "jpeg_compression", 3),
)


class ArrayImages(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


class OfficeHomeImages(Dataset):
    def __init__(self, root: Path, transform) -> None:
        domain_dirs = [root / name for name in OFFICE_DOMAINS]
        missing = [str(path) for path in domain_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Office-Home must contain Art, Clipart, Product, and Real World; "
                f"missing: {', '.join(missing)}"
            )
        class_names = sorted({p.name for domain in domain_dirs for p in domain.iterdir() if p.is_dir()})
        class_to_id = {name: index for index, name in enumerate(class_names)}
        samples = []
        for domain_id, domain in enumerate(domain_dirs):
            for class_name in class_names:
                class_dir = domain / class_name
                if not class_dir.is_dir():
                    continue
                for path in sorted(class_dir.rglob("*")):
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((path, class_to_id[class_name], domain_id))
        if not samples:
            raise RuntimeError(f"No Office-Home images found below {root}")
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, domain = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, domain


@torch.inference_mode()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    features, labels, domains = [], [], []
    model.eval()
    for batch in loader:
        images, target = batch[:2]
        features.append(model(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float16))
        labels.append(np.asarray(target, dtype=np.int64))
        if len(batch) == 3:
            domains.append(np.asarray(batch[2], dtype=np.int64))
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(domains) if domains else None,
    )


def make_loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def select_indices(length: int, count: int, seed: int) -> np.ndarray:
    if count > length:
        raise ValueError(f"Requested {count} examples from a dataset of length {length}")
    return np.random.default_rng(seed).permutation(length)[:count]


def balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels)):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has {len(candidates)} examples, need {per_class}")
        selected.extend(rng.permutation(candidates)[:per_class].tolist())
    return np.asarray(selected, dtype=np.int64)


def resolve_cifar_c(root: Path, name: str) -> Path:
    folder = "CIFAR-10-C" if name == "cifar10" else "CIFAR-100-C"
    candidate = root / folder
    path = candidate if candidate.is_dir() else root
    required = [path / f"{source}.npy" for _, source, _ in CIFAR_EPISODES]
    required.append(path / "labels.npy")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Extract {folder} below --cifar-c-root; missing: {', '.join(missing)}"
        )
    return path


def save_cifar_cache(
    name: str,
    cifar_c_root: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    overwrite: bool,
) -> None:
    output = data_dir / f"{name}_resnet18_ftta_v1.npz"
    if output.exists() and not overwrite:
        print(f"SKIP {output} (use --overwrite to replace)")
        return
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    dataset_type = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    raw_root = data_dir / f"{name}_torchvision"
    train = dataset_type(root=raw_root, train=True, download=True, transform=weights.transforms())
    clean_test = dataset_type(root=raw_root, train=False, download=True)
    train_targets = np.asarray(train.targets, dtype=np.int64)
    train_idx = balanced_indices(train_targets, 500, 20260930)
    x_train, y_train, _ = extract_features(
        model, make_loader(Subset(train, train_idx), batch_size, workers), device
    )
    cifar_c = resolve_cifar_c(cifar_c_root, name)
    all_corrupt_labels = np.load(cifar_c / "labels.npy").astype(np.int64)
    base_labels = all_corrupt_labels[:10_000]
    test_idx = balanced_indices(base_labels, 200 if name == "cifar10" else 100, 20260931)
    x_target = []
    for _, source, severity in CIFAR_EPISODES:
        images = np.load(cifar_c / f"{source}.npy", mmap_mode="r")
        offset = (severity - 1) * 10_000
        episode = np.asarray(images[offset : offset + 10_000][test_idx])
        labels = base_labels[test_idx]
        features, _, _ = extract_features(
            model,
            make_loader(ArrayImages(episode, labels, weights.transforms()), batch_size, workers),
            device,
        )
        x_target.append(features)
    y_test = base_labels[test_idx]
    source_idx, cal_idx = [], []
    for label in sorted(np.unique(y_train)):
        members = np.flatnonzero(y_train == label)
        source_idx.extend(members[:425].tolist())
        cal_idx.extend(members[425:].tolist())
    source_idx = np.asarray(source_idx, dtype=np.int64)
    cal_idx = np.asarray(cal_idx, dtype=np.int64)
    np.savez_compressed(
        output,
        x_source=x_train[source_idx],
        y_source=y_train[source_idx],
        source_client=np.zeros(len(source_idx), dtype=np.int64),
        x_cal=x_train[cal_idx],
        y_cal=y_train[cal_idx],
        cal_client=np.zeros(len(cal_idx), dtype=np.int64),
        x_target=np.stack(x_target),
        y_target=y_test,
        target_client=np.zeros(len(y_test), dtype=np.int64),
        episode_names=np.asarray([item[0] for item in CIFAR_EPISODES]),
        severities=np.asarray([item[2] for item in CIFAR_EPISODES], dtype=np.int64),
        class_names=np.asarray(clean_test.classes),
    )
    print(f"WROTE {output} ({len(x_train)} train, {len(y_test)} test per episode)")


def save_mnist_cache(
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    overwrite: bool,
) -> None:
    output = data_dir / "mnist_mobilenet_v3_64.npz"
    if output.exists() and not overwrite:
        print(f"SKIP {output} (use --overwrite to replace)")
        return
    weights = MobileNet_V3_Large_Weights.DEFAULT
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    raw_root = data_dir / "mnist_torchvision"
    train = datasets.MNIST(root=raw_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=raw_root, train=False, download=True, transform=transform)
    model = mobilenet_v3_large(weights=weights)
    model.classifier = nn.Identity()
    model.to(device)
    x_train, y_train, _ = extract_features(model, make_loader(train, batch_size, workers), device)
    x_test, y_test, _ = extract_features(model, make_loader(test, batch_size, workers), device)
    np.savez_compressed(output, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"WROTE {output} ({len(x_train)} train, {len(x_test)} test)")


def save_officehome_cache(
    raw_root: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    overwrite: bool,
) -> None:
    output = data_dir / "officehome65_resnet18.npz"
    if output.exists() and not overwrite:
        print(f"SKIP {output} (use --overwrite to replace)")
        return
    weights = ResNet18_Weights.DEFAULT
    dataset = OfficeHomeImages(raw_root, weights.transforms())
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    features, labels, domains = extract_features(
        model, make_loader(dataset, batch_size, workers), device
    )
    np.savez_compressed(
        output,
        features=features,
        labels=labels,
        domains=domains,
        class_names=np.asarray(dataset.class_names),
    )
    raw_output = data_dir / "officehome_raw_48.npz"
    raw_transform = transforms.Compose([transforms.Resize((48, 48)), transforms.PILToTensor()])
    raw_dataset = OfficeHomeImages(raw_root, raw_transform)
    raw_images, raw_labels, raw_domains = [], [], []
    for image, label, domain in make_loader(raw_dataset, batch_size, workers):
        raw_images.append(image.numpy().astype(np.uint8))
        raw_labels.append(np.asarray(label, dtype=np.int64))
        raw_domains.append(np.asarray(domain, dtype=np.int64))
    np.savez_compressed(
        raw_output,
        images=np.concatenate(raw_images),
        labels=np.concatenate(raw_labels),
        domains=np.concatenate(raw_domains),
    )
    print(f"WROTE {output} ({len(features)} images)")
    print(f"WROTE {raw_output} ({len(features)} images)")


def requested_datasets(value: str) -> Iterable[str]:
    return ("cifar10", "cifar100", "mnist", "officehome") if value == "all" else (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100", "mnist", "officehome", "all"),
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--cifar-c-root",
        type=Path,
        help="Directory containing extracted CIFAR-10-C and CIFAR-100-C folders.",
    )
    parser.add_argument(
        "--officehome-raw",
        type=Path,
        help="Extracted Office-Home directory containing the four domain folders.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    for name in requested_datasets(args.dataset):
        if name in {"cifar10", "cifar100"}:
            if args.cifar_c_root is None:
                raise SystemExit("--cifar-c-root is required for CIFAR feature-cache construction")
            save_cifar_cache(
                name,
                args.cifar_c_root,
                args.data_dir,
                device,
                args.batch_size,
                args.workers,
                args.overwrite,
            )
        elif name == "mnist":
            save_mnist_cache(args.data_dir, device, args.batch_size, args.workers, args.overwrite)
        else:
            if args.officehome_raw is None:
                raise SystemExit("--officehome-raw is required for --dataset officehome or all")
            save_officehome_cache(
                args.officehome_raw,
                args.data_dir,
                device,
                args.batch_size,
                args.workers,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
