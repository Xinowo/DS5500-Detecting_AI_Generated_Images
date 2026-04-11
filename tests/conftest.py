"""Shared pytest fixtures for the AIGI-Detection test suite."""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # force non-interactive backend (no display needed)

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture()
def tmp_image_dir(tmp_path: Path):
    """Create 20 synthetic JPEG images + a matching DataFrame in a temp dir.

    Returns:
        (tmp_path, df) where df has columns 'file_name' and 'label'.
        Labels alternate 0/1 so the mini-set is balanced.
    """
    filenames, labels = [], []
    for i in range(20):
        fname = f"img_{i:04d}.jpg"
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / fname)
        filenames.append(fname)
        labels.append(i % 2)
    df = pd.DataFrame({"file_name": filenames, "label": labels})
    return tmp_path, df


@dataclass
class MinimalConfig:
    """Minimal config dataclass matching Trainer's expected interface."""
    # paths (will be overridden in tests using tmp_path)
    save_dir: str = "checkpoints"
    outputs_dir: str = "outputs"
    run_name: str = "test_run"
    # training
    epochs: int = 1
    batch_size: int = 4
    lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.0
    patience: int = 3
    grad_clip: float = 0.0
    use_amp: bool = False   # always False in tests — no GPU required
    eta_min: float = 1e-5
    warmup_epochs: int = 0
