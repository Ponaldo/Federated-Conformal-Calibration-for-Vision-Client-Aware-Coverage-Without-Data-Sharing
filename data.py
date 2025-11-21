import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    image_size: int


DATASETS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec("cifar10", 10, 32),
    "cifar100": DatasetSpec("cifar100", 100, 32),
    "tiny-imagenet": DatasetSpec("tiny-imagenet", 200, 64),
    "domainnet": DatasetSpec("domainnet", 345, 224),  # subclasses vary; set high-level count
}


def _default_transforms(spec: DatasetSpec, train: bool, use_randaugment: bool = True) -> T.Compose:
    if spec.image_size >= 96:
        resize = T.Resize((spec.image_size, spec.image_size))
    else:
        resize = T.Resize(spec.image_size) if spec.image_size != 32 else T.Identity()

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        aug: List[torch.nn.Module] = [T.RandomHorizontalFlip(), T.RandomCrop(spec.image_size, padding=4)]
        if use_randaugment:
            aug.append(T.RandAugment(num_ops=2, magnitude=9))
        aug.extend([T.ToTensor(), normalize])
        return T.Compose([resize, *aug])
    return T.Compose([resize, T.ToTensor(), normalize])


def default_transforms(spec: DatasetSpec, train: bool, use_randaugment: bool = True) -> T.Compose:
    return _default_transforms(spec, train=train, use_randaugment=use_randaugment)


def load_base_dataset(
    name: str,
    root: str,
    train: bool,
    transform: T.Compose,
    domains: Sequence[str] = ("clipart", "painting", "real", "sketch"),
) -> Dataset:
    lname = name.lower()
    if lname == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if lname == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    if lname == "tiny-imagenet":
        # Expect standard Tiny-ImageNet directory with train/val folders.
        split = "train" if train else "val"
        return datasets.ImageFolder(os.path.join(root, split), transform=transform)
    if lname == "domainnet":
        # Expect DomainNet structure: root/domain/class/*.jpg
        subset: List[Dataset] = []
        for dom in domains:
            dom_path = os.path.join(root, dom, "train" if train else "test")
            if os.path.isdir(dom_path):
                subset.append(datasets.ImageFolder(dom_path, transform=transform))
        if not subset:
            raise ValueError(f"No DomainNet domains found under {root}; expected folders: {domains}.")
        return torch.utils.data.ConcatDataset(subset)
    raise ValueError(f"Unsupported dataset: {name}")


def _labels_from_dataset(ds: Dataset) -> List[int]:
    if hasattr(ds, "targets"):
        return list(getattr(ds, "targets"))
    if hasattr(ds, "labels"):
        return list(getattr(ds, "labels"))
    if hasattr(ds, "imgs"):  # ImageFolder stores (path, class)
        return [y for _, y in ds.imgs]
    if hasattr(ds, "samples"):
        return [y for _, y in ds.samples]
    raise ValueError("Cannot extract labels from dataset")


def labels_from_dataset(ds: Dataset) -> List[int]:
    return _labels_from_dataset(ds)


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    min_per_client: int = 1,
    seed: int = 0,
    fraction: float = 1.0,
) -> List[List[int]]:
    """
    Returns a list of index lists, one per client, sampled via Dirichlet(alpha).
    fraction allows sub-sampling the global dataset before partitioning.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    all_idx = np.arange(n)
    if fraction < 1.0:
        keep = rng.choice(n, size=int(n * fraction), replace=False)
        all_idx = keep
    labels = np.array(labels)[all_idx]
    classes = np.unique(labels)
    per_client: List[List[int]] = [[] for _ in range(num_clients)]

    while True:
        per_client = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = all_idx[labels == cls]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            parts = np.split(cls_idx, splits)
            for i, part in enumerate(parts):
                per_client[i].extend(part.tolist())
        sizes = [len(p) for p in per_client]
        if min(sizes) >= min_per_client:
            break
        alpha *= 1.1  # relax slightly if a client is empty
    return per_client


def split_client_data(
    dataset: Dataset,
    client_indices: List[List[int]],
    calib_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Subset], List[Subset]]:
    rng = random.Random(seed)
    train_subsets: List[Subset] = []
    calib_subsets: List[Subset] = []
    for indices in client_indices:
        rng.shuffle(indices)
        calib_size = max(1, int(len(indices) * calib_fraction))
        calib_idx = indices[:calib_size]
        train_idx = indices[calib_size:]
        train_subsets.append(Subset(dataset, train_idx))
        calib_subsets.append(Subset(dataset, calib_idx))
    return train_subsets, calib_subsets


def build_client_loaders(
    train_subsets: Sequence[Dataset],
    calib_subsets: Sequence[Dataset],
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        for subset in train_subsets
    ]
    calib_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        for subset in calib_subsets
    ]
    return train_loaders, calib_loaders


def build_test_loader(dataset: Dataset, batch_size: int, num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
