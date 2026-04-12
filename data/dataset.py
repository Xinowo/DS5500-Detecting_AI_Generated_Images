"""
Dataset utilities for AIGI-Detection.

Provides:
  - AIDataset       : PyTorch Dataset for loading labelled images from a DataFrame.
  - get_transforms  : Returns train / val-test torchvision transform pipelines.
  - prepare_splits  : Stratified train / val / test split with optional JSON persistence.
  - get_dataloaders : Convenience wrapper that returns three DataLoader objects.
"""

from __future__ import annotations

import logging
import os
import json
import random

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AIDataset(Dataset):
    """Binary image dataset: label 1 = AI-generated, label 0 = human/real.

    Args:
        dataframe: DataFrame with columns ``file_name`` and ``label``.
        data_root: Directory containing the image files.
        transform:  Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_root: str | Path,
        transform=None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        img_name: str = row["file_name"]
        label: int    = int(row["label"])

        # Strip prefix so the path works whether it contains 'train_data/' or not
        if img_name.startswith("train_data/") or img_name.startswith("test_data_v2/"):
            img_name = os.path.basename(img_name)

        img_path = self.data_root / img_name
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning(
                "Could not load image '%s' (%s); substituting blank placeholder.",
                img_path, exc,
            )
            image = Image.new("RGB", (256, 256))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, eval_transform)`` for 224-px models.

    Train pipeline applies random augmentations; eval pipeline crops deterministically.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Data-split helpers
# ---------------------------------------------------------------------------
def prepare_splits(
    df: pd.DataFrame,
    train_size: int | None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_json_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split with optional JSON persistence.

    If ``split_json_path`` points to an existing file the split is loaded from
    disk (ensuring exact reproducibility across runs). Otherwise the split is
    computed, and – when ``split_json_path`` is provided – saved to that file.

    Args:
        df:              Full CSV DataFrame with columns ``file_name`` and ``label``.
        train_size:      Number of samples to draw from ``df`` before splitting.
                         Pass ``None`` to use the entire DataFrame.
        val_ratio:       Fraction of the final pool to use as validation.
        test_ratio:      Fraction of the final pool to use as test.
        seed:            Random seed for reproducibility.
        split_json_path: Optional path to a JSON file for split persistence.

    Returns:
        ``(df_train, df_val, df_test)`` DataFrames.
    """
    # --- Input validation ---
    required_cols = {"file_name", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    valid_labels = {0, 1}
    actual_labels = set(df["label"].dropna().unique().tolist())
    bad_labels = actual_labels - valid_labels
    if bad_labels:
        raise ValueError(
            f"Column 'label' must contain only 0 or 1; "
            f"found unexpected values: {bad_labels}"
        )

    # Optional sub-sampling
    if train_size is not None:
        df_pool, _ = train_test_split(
            df,
            train_size=train_size,
            stratify=df["label"],
            random_state=seed,
        )
    else:
        df_pool = df.copy()

    # Load from JSON if it exists
    split_json_path = Path(split_json_path) if split_json_path else None
    if split_json_path is not None and split_json_path.exists():
        with open(split_json_path, "r") as fh:
            split = json.load(fh)
        df_train = df_pool[df_pool["file_name"].isin(split["train"])].copy().reset_index(drop=True)
        df_val   = df_pool[df_pool["file_name"].isin(split["val"])  ].copy().reset_index(drop=True)
        df_test  = df_pool[df_pool["file_name"].isin(split["test"]) ].copy().reset_index(drop=True)
        logger.info("[Split] Loaded split from %s", split_json_path)
        return df_train, df_val, df_test

    # Compute split
    val_test_ratio = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df_pool,
        test_size=val_test_ratio,
        stratify=df_pool["label"],
        random_state=seed,
    )
    test_size_from_temp = test_ratio / val_test_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_size_from_temp,
        stratify=df_temp["label"],
        random_state=seed,
    )
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    # Persist split
    if split_json_path is not None:
        split_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed":       seed,
            "val_ratio":  val_ratio,
            "test_ratio": test_ratio,
            "train":      df_train["file_name"].tolist(),
            "val":        df_val["file_name"].tolist(),
            "test":       df_test["file_name"].tolist(),
        }
        with open(split_json_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("[Split] Saved split to %s", split_json_path)

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from split DataFrames.

    Automatically detects whether ``data_root`` contains per-split
    subdirectories (``train/``, ``validation/``, ``test/``).  If they exist
    each split uses its own sub-directory; otherwise all three splits share
    the same root (legacy / Colab single-directory layout).

    Args:
        df_train:    Training split DataFrame.
        df_val:      Validation split DataFrame.
        df_test:     Test split DataFrame.
        data_root:   Base directory for image files, e.g. ``data/sampled_data_5k``.
        batch_size:  Samples per batch.
        num_workers: Worker processes for data loading.

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    root = Path(data_root)

    # Auto-detect per-split sub-directory layout (supports both "validation" and "val")
    train_root = root / "train"      if (root / "train").is_dir()      else root
    val_root   = (root / "validation" if (root / "validation").is_dir()
                  else root / "val"   if (root / "val").is_dir()
                  else root)
    test_root  = root / "test"       if (root / "test").is_dir()       else root

    logger.info("[DataLoader] train images : %s", train_root)
    logger.info("[DataLoader] val images   : %s", val_root)
    logger.info("[DataLoader] test images  : %s", test_root)

    train_tf, eval_tf = get_transforms()

    train_ds = AIDataset(df_train, train_root, transform=train_tf)
    val_ds   = AIDataset(df_val,   val_root,   transform=eval_tf)
    test_ds  = AIDataset(df_test,  test_root,  transform=eval_tf)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
