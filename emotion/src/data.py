"""Datasets, transforms, and DataLoaders for FER-2013.

FER-2013 is grayscale 48x48. MobileNetV2 (ImageNet-pretrained) expects
RGB 224x224 with ImageNet normalisation, so we:
    1. Open the image, convert "L" -> "RGB" (triplicates the single channel).
    2. Resize up to 224.
    3. Augment (train only) and normalise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from . import config as C


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _train_tf() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 48x48 L -> 48x48 RGB
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        # RandomErasing acts as a mild dropout in image space; helps generalise.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])


def _eval_tf() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def inference_tf() -> transforms.Compose:
    """Public: used by predict.py at inference time."""
    return _eval_tf()


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------
def _stratified_train_val_split(
    full_train: datasets.ImageFolder, val_frac: float, seed: int
) -> Tuple[Subset, Subset]:
    """Split per-class so every class is represented in val
    (important - "disgust" is rare; a random split could miss it)."""
    rng = np.random.RandomState(seed)
    targets = np.array(full_train.targets)
    train_idx, val_idx = [], []
    for cls_idx in np.unique(targets):
        cls_positions = np.where(targets == cls_idx)[0]
        rng.shuffle(cls_positions)
        n_val = max(1, int(len(cls_positions) * val_frac))
        val_idx.extend(cls_positions[:n_val].tolist())
        train_idx.extend(cls_positions[n_val:].tolist())
    return Subset(full_train, train_idx), Subset(full_train, val_idx)


def _class_counts(image_folder: datasets.ImageFolder, indices) -> np.ndarray:
    targets = np.array(image_folder.targets)[indices]
    return np.bincount(targets, minlength=C.NUM_CLASSES)


def build_dataloaders(
    train_dir: Path | None = None,
    test_dir: Path | None = None,
    batch_size: int = C.BATCH_SIZE,
    num_workers: int = C.NUM_WORKERS,
    use_weighted_sampler: bool = False,
):
    """Return (train_loader, val_loader, test_loader, class_weights_tensor).

    `class_weights_tensor` is for use in CrossEntropyLoss(weight=...).
    `use_weighted_sampler=True` switches to a WeightedRandomSampler instead -
    pick one strategy, not both. Default is class-weighted loss (simpler to
    explain in the interview, and works well with FER-2013's imbalance).
    """
    train_dir = train_dir or C.TRAIN_DIR
    test_dir = test_dir or C.TEST_DIR

    # ImageFolder reads the directory structure; classes are sorted alpha,
    # which matches config.CLASSES.
    full_train = datasets.ImageFolder(str(train_dir), transform=_train_tf())
    test_set = datasets.ImageFolder(str(test_dir), transform=_eval_tf())

    # Sanity-check class order. If this fires, FER-2013 folder names differ.
    assert full_train.classes == C.CLASSES, (
        f"Class order mismatch.\n  on disk: {full_train.classes}\n"
        f"  config : {C.CLASSES}"
    )

    train_subset, val_subset = _stratified_train_val_split(
        full_train, C.VAL_SPLIT, C.SEED
    )

    # The validation subset references the same ImageFolder, which means it
    # would currently receive training augmentations. Wrap it so it uses the
    # eval transform instead.
    val_image_folder = datasets.ImageFolder(str(train_dir), transform=_eval_tf())
    val_subset = Subset(val_image_folder, val_subset.indices)

    # ---- Class weights (inverse frequency, normalised) --------------------
    train_counts = _class_counts(full_train, train_subset.indices)
    # Guard against divide-by-zero if a class is missing.
    inv = 1.0 / np.maximum(train_counts, 1)
    class_weights = inv / inv.sum() * C.NUM_CLASSES   # mean ~ 1.0
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    # ---- Loaders ---------------------------------------------------------
    if use_weighted_sampler:
        sample_weights = class_weights[np.array(full_train.targets)[train_subset.indices]]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights_t, train_counts
