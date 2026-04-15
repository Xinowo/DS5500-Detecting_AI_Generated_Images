"""Unit tests for visualization helpers."""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest

from visualization.visualize import plot_confusion_matrix, plot_roc_curve, plot_training_curves

_gradcam_available = importlib.util.find_spec("pytorch_grad_cam") is not None
_skip_no_gradcam = pytest.mark.skipif(
    not _gradcam_available,
    reason="pytorch_grad_cam not installed - gradcam checkpoint discovery test skipped",
)


class TestPlotConfusionMatrix:
    def test_runs_without_error(self, tmp_path):
        cm = np.array([[45, 5], [3, 47]])
        plot_confusion_matrix(cm, save_path=tmp_path / "cm.png")

    def test_output_file_created(self, tmp_path):
        cm = np.array([[45, 5], [3, 47]])
        out = tmp_path / "cm.png"
        plot_confusion_matrix(cm, save_path=out)
        assert out.exists()

    def test_custom_class_names(self, tmp_path):
        cm = np.array([[10, 2], [1, 9]])
        plot_confusion_matrix(cm, class_names=["Real", "AI"], save_path=tmp_path / "cm2.png")
        assert (tmp_path / "cm2.png").exists()


class TestPlotRocCurve:
    def test_runs_without_error(self, tmp_path):
        labels = np.array([0, 0, 1, 1, 0, 1])
        probs  = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        plot_roc_curve(labels, probs, save_path=tmp_path / "roc.png")

    def test_output_file_created(self, tmp_path):
        labels = np.array([0, 0, 1, 1])
        probs  = np.array([0.1, 0.4, 0.6, 0.9])
        out = tmp_path / "roc.png"
        plot_roc_curve(labels, probs, save_path=out)
        assert out.exists()


class TestPlotTrainingCurves:
    def _make_history(self, n: int = 5) -> dict:
        return {
            "train_loss":   [0.9 - i * 0.1 for i in range(n)],
            "val_loss":     [1.0 - i * 0.09 for i in range(n)],
            "val_accuracy": [0.5 + i * 0.05 for i in range(n)],
        }

    def test_runs_without_error(self, tmp_path):
        plot_training_curves(self._make_history(), save_path=tmp_path / "curves.png")

    def test_output_file_created(self, tmp_path):
        out = tmp_path / "curves.png"
        plot_training_curves(self._make_history(), save_path=out)
        assert out.exists()

    def test_works_without_accuracy_key(self, tmp_path):
        history = {
            "train_loss": [0.9, 0.8, 0.7],
            "val_loss":   [1.0, 0.9, 0.8],
        }
        out = tmp_path / "curves_no_acc.png"
        plot_training_curves(history, save_path=out)
        assert out.exists()


class TestGradcamCheckpointDiscovery:
    @_skip_no_gradcam
    def test_prefers_newest_best_model_file(self, tmp_path):
        from visualization.gradcam import _find_best_checkpoint  # noqa: PLC0415

        older = tmp_path / "best_model_20260317_220741.pth"
        newer = tmp_path / "best_model_20260415_101500.pth"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        older.touch()
        newer.touch()

        resolved = _find_best_checkpoint(tmp_path, "ResNet-50")
        assert resolved == newer
