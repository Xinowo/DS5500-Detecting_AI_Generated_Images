"""Unit tests for training/trainer.py and training/train.py utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import Trainer
from training.train import seed_everything
from tests.conftest import MinimalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    """Return a DataLoader with random float images and binary labels."""
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _tiny_model() -> nn.Module:
    """Tiny 2-class linear model that accepts (B, 3, 224, 224) input."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 2),
    )


def _make_trainer(tmp_path: Path) -> Trainer:
    cfg = MinimalConfig(
        save_dir=str(tmp_path / "ckpts"),
        outputs_dir=str(tmp_path / "outputs"),
    )
    model = _tiny_model()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    return Trainer(model, criterion, cfg, device)


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_same_seed_same_output(self):
        seed_everything(42)
        a = torch.rand(5).tolist()
        seed_everything(42)
        b = torch.rand(5).tolist()
        assert a == b

    def test_different_seeds_different_output(self):
        seed_everything(0)
        a = torch.rand(5).tolist()
        seed_everything(1)
        b = torch.rand(5).tolist()
        assert a != b


# ---------------------------------------------------------------------------
# Trainer._compute_metrics  (static — no model needed)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 0, 1, 1])
        # probs close to 0 for class-0 samples, close to 1 for class-1 samples
        probs  = np.array([0.1, 0.1, 0.9, 0.9])
        m = Trainer._compute_metrics(probs, labels)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["roc_auc"]  == pytest.approx(1.0)
        assert m["f1"]       == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        labels = np.array([0, 0, 1, 1])
        probs  = np.array([0.9, 0.9, 0.1, 0.1])
        m = Trainer._compute_metrics(probs, labels)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        labels = np.array([0, 1, 0, 1])
        probs  = np.array([0.2, 0.8, 0.3, 0.7])
        m = Trainer._compute_metrics(probs, labels)
        for key in ("roc_auc", "accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in m

    def test_confusion_matrix_shape(self):
        labels = np.array([0, 1, 0, 1])
        probs  = np.array([0.2, 0.8, 0.3, 0.7])
        m = Trainer._compute_metrics(probs, labels)
        assert m["confusion_matrix"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Trainer initialization
# ---------------------------------------------------------------------------

class TestTrainerInit:
    def test_optimizer_created(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.optimizer is not None

    def test_scheduler_created(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.scheduler is not None

    def test_scaler_is_none_when_amp_disabled(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.scaler is None  # MinimalConfig sets use_amp=False


# ---------------------------------------------------------------------------
# Trainer._train_one_epoch  (smoke — 1 epoch on tiny data)
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_loss_is_finite(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        loader = _tiny_loader()
        loss = trainer._train_one_epoch(loader)
        assert np.isfinite(loss)

    def test_loss_is_positive(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        loader = _tiny_loader()
        loss = trainer._train_one_epoch(loader)
        assert loss > 0


# ---------------------------------------------------------------------------
# Trainer.fit (1 epoch — checks checkpoint is saved)
# ---------------------------------------------------------------------------

class TestTrainerFit:
    def test_checkpoint_saved_after_fit(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        train_loader = _tiny_loader()
        val_loader   = _tiny_loader(n=8, batch_size=4)
        trainer.fit(train_loader, val_loader)
        assert trainer.best_ckpt_path is not None
        assert Path(trainer.best_ckpt_path).exists()
