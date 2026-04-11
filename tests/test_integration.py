"""End-to-end smoke test: dataset → model → trainer → checkpoint.

This test intentionally uses a tiny synthetic dataset and a minimal
model so it finishes quickly without GPU or pretrained-weight downloads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from data.dataset import get_dataloaders, prepare_splits
from training.trainer import Trainer
from tests.conftest import MinimalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_model() -> nn.Module:
    """Tiny image classifier that accepts (N, 3, H, W) input without
    needing pretrained weights or large parameter counts."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),  # (N, 3, H, W) -> (N, 3, 1, 1)
        nn.Flatten(),                  # -> (N, 3)
        nn.Linear(3, 2),               # -> (N, 2)
    )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline from DataLoaders to a saved checkpoint."""

    def test_fit_saves_checkpoint(self, tmp_image_dir, tmp_path):
        """Training for one epoch should create a .pth checkpoint file."""
        data_root, df = tmp_image_dir

        # Split the 20-row DataFrame into train/val/test
        df_train, df_val, df_test = prepare_splits(
            df,
            train_size=None,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        # Build DataLoaders (single root, no sub-dir layout)
        train_loader, val_loader, _ = get_dataloaders(
            df_train, df_val, df_test,
            data_root=data_root,
            batch_size=4,
            num_workers=0,  # 0 workers is safest on Windows CI
        )

        # Configure a CPU-safe Trainer
        cfg = MinimalConfig(
            save_dir=str(tmp_path / "ckpts"),
            outputs_dir=str(tmp_path / "outputs"),
            run_name="smoke",
            epochs=1,
            patience=5,
        )
        model = _mini_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        trainer = Trainer(model, criterion, cfg, device)

        trainer.fit(train_loader, val_loader)

        assert trainer.best_ckpt_path is not None, "Trainer.best_ckpt_path should be set after fit"
        assert Path(trainer.best_ckpt_path).exists(), "Checkpoint file must exist on disk"

    def test_evaluate_saves_metrics_and_preds(self, tmp_image_dir, tmp_path):
        """evaluate() should produce a JSON metrics file and a .npz predictions file."""
        data_root, df = tmp_image_dir

        df_train, df_val, df_test = prepare_splits(
            df,
            train_size=None,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=0,
        )

        train_loader, val_loader, test_loader = get_dataloaders(
            df_train, df_val, df_test,
            data_root=data_root,
            batch_size=4,
            num_workers=0,
        )

        save_dir = str(tmp_path / "ckpts2")
        cfg = MinimalConfig(
            save_dir=save_dir,
            outputs_dir=str(tmp_path / "outputs2"),
            run_name="smoke_eval",
            epochs=1,
            patience=5,
        )
        model = _mini_model()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        trainer = Trainer(model, criterion, cfg, device)

        trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader)

        # Verify returned metrics have expected keys
        for key in ("roc_auc", "accuracy", "f1"):
            assert key in metrics, f"Missing key '{key}' in evaluate() result"

        # Verify files were written to save_dir
        ckpt_dir = Path(save_dir)
        json_files = list(ckpt_dir.glob("test_metrics_*.json"))
        npz_files  = list(ckpt_dir.glob("test_preds_*.npz"))
        assert len(json_files) == 1, "Expected exactly one test_metrics JSON file"
        assert len(npz_files)  == 1, "Expected exactly one test_preds NPZ file"
