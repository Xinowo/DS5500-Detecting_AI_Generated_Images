"""
Visualisation utilities for AIGI-Detection.

Functions:
  - plot_confusion_matrix  : Heatmap of the 2x2 confusion matrix.
  - plot_roc_curve         : ROC curve with AUC annotation.
  - plot_training_curves   : Train/val loss and accuracy curves from a history CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        cm:          2-D integer array of shape ``[n_classes, n_classes]``.
        class_names: List of class label strings. Defaults to ``["Human", "AI"]``.
        save_path:   If provided, the figure is saved to this path.
        title:       Plot title.
    """
    if class_names is None:
        class_names = ["Human (0)", "AI (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------
def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "ROC Curve",
) -> None:
    """Plot and optionally save the ROC curve.

    Args:
        labels:    Ground-truth binary labels.
        probs:     Model probability scores for the positive class (AI).
        save_path: If provided, the figure is saved to this path.
        title:     Plot title.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    title: str = "Training Curves",
) -> None:
    """Plot train / val loss and accuracy over epochs.

    Args:
        history:   Dictionary with keys ``"train_loss"``, ``"val_loss"``,
                   and optionally ``"val_accuracy"``.
                   Each value is a list of per-epoch floats.
        save_path: If provided, the figure is saved to this path.
        title:     Figure suptitle.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    n_plots = 2 if "val_accuracy" in history else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Loss subplot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()

    # Accuracy subplot (optional)
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="green")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy")
        axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate all plots for a given training run.

    Usage::

        python -m visualization.visualize \\
            --history_csv    outputs/metrics/resnet50-5k_20260307_143022_history.csv \\
            --checkpoint_dir checkpoints/resnet50 \\
            --timestamp      20260307_143022

    Saves three figures to ``outputs/figures/``:
      ``<timestamp>_<run_name>_training_curves.png``
      ``<timestamp>_<run_name>_confusion_matrix.png``
      ``<timestamp>_<run_name>_roc_curve.png``
    """
    import argparse
    import json
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate AIGI-Detection training visualizations.")
    parser.add_argument("--history_csv",    required=True,
                        help="Path to <run_name>_<timestamp>_history.csv")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Checkpoint directory (contains test_metrics_<ts>.json and test_preds_<ts>.npz)")
    parser.add_argument("--timestamp",      required=True,
                        help="Timestamp string matching the training run, e.g. 20260307_143022")
    args = parser.parse_args()

    ts       = args.timestamp
    ckpt_dir = Path(args.checkpoint_dir)
    fig_dir  = Path("outputs/figures")

    # Derive run_name from CSV stem: <run_name>_<timestamp>_history
    csv_stem = Path(args.history_csv).stem              # e.g. "resnet50-5k_20260307_143022_history"
    run_name = csv_stem.replace(f"_{ts}_history", "")  # e.g. "resnet50-5k"

    # 1. Training curves
    df = pd.read_csv(args.history_csv)
    plot_training_curves(
        history   = df.to_dict(orient="list"),
        save_path = fig_dir / f"{ts}_{run_name}_training_curves.png",
        title     = f"{run_name} Training Curves",
    )

    # 2. Confusion matrix
    metrics_path = ckpt_dir / f"test_metrics_{ts}.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    plot_confusion_matrix(
        cm        = np.array(metrics["confusion_matrix"]),
        save_path = fig_dir / f"{ts}_{run_name}_confusion_matrix.png",
        title     = f"{run_name} Confusion Matrix",
    )

    # 3. ROC curve
    preds_path = ckpt_dir / f"test_preds_{ts}.npz"
    data = np.load(preds_path)
    plot_roc_curve(
        labels    = data["labels"],
        probs     = data["probs"],
        save_path = fig_dir / f"{ts}_{run_name}_roc_curve.png",
        title     = f"{run_name} ROC Curve",
    )

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
    plt.close(fig)
else:
    plt.show()
