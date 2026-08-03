from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CIFAR100WithCoarse(CIFAR100):
    """Torchvision CIFAR-100 with both fine and coarse targets.

    Torchvision exposes the fine label.  The official CIFAR-100 pickle also
    contains ``coarse_labels``; this class loads them from the same file.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool,
        transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, download=download)
        downloaded_list = self.train_list if train else self.test_list
        coarse_targets: list[int] = []
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as handle:
                entry = pickle.load(handle, encoding="latin1")
            if "coarse_labels" not in entry:
                raise RuntimeError(f"CIFAR-100 file lacks coarse_labels: {file_path}")
            coarse_targets.extend(int(x) for x in entry["coarse_labels"])
        if len(coarse_targets) != len(self.targets):
            raise RuntimeError("fine and coarse target lengths do not match")
        self.coarse_targets = coarse_targets
        self._load_coarse_meta()

    def _load_coarse_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, "meta")
        with open(path, "rb") as handle:
            entry = pickle.load(handle, encoding="latin1")
        self.coarse_classes = list(entry.get("coarse_label_names", []))
        if len(self.coarse_classes) != 20:
            raise RuntimeError("expected 20 CIFAR-100 coarse classes")

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index])
        fine = int(self.targets[index])
        coarse = int(self.coarse_targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, fine, coarse, int(index)


class TwoViewTransform:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image), self.transform(image)


def simclr_transform(image_size: int = 32) -> TwoViewTransform:
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    base = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return TwoViewTransform(base)


def supervised_train_transform(image_size: int = 32) -> Callable:
    return transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def network_eval_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def raw_eval_transform() -> Callable:
    return transforms.ToTensor()


def balanced_fine_subset_indices(
    fine_targets: Sequence[int],
    *,
    per_class: int,
    seed: int,
    num_classes: int = 100,
) -> np.ndarray:
    targets = np.asarray(fine_targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in range(num_classes):
        candidates = np.flatnonzero(targets == label)
        if candidates.size < per_class:
            raise ValueError(
                f"fine class {label} has only {candidates.size} examples; "
                f"cannot select {per_class}"
            )
        chosen = rng.choice(candidates, size=per_class, replace=False)
        selected.extend(int(x) for x in chosen)
    selected_array = np.asarray(selected, dtype=np.int64)
    # Shuffle after balancing so batches are not class-blocked.
    rng.shuffle(selected_array)
    return selected_array


def make_eval_subset(
    root: str | Path,
    *,
    per_fine_class: int,
    subset_seed: int,
    transform: Callable,
    download: bool = False,
) -> tuple[Subset, np.ndarray]:
    dataset = CIFAR100WithCoarse(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    indices = balanced_fine_subset_indices(
        dataset.targets,
        per_class=per_fine_class,
        seed=subset_seed,
    )
    return Subset(dataset, indices.tolist()), indices


class SyntheticCIFARLike(Dataset):
    """Small offline dataset used only by unit and smoke tests."""

    def __init__(self, n: int = 64, num_fine: int = 8, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, 3, 32, 32, generator=generator)
        self.fine = torch.arange(n) % num_fine
        self.coarse = self.fine // max(1, num_fine // 2)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], int(self.fine[index]), int(self.coarse[index]), index
