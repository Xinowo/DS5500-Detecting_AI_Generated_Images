"""
Trainer class for AIGI-Detection.

Encapsulates one epoch of training, one epoch of validation, and
the full training loop with early stopping, model checkpointing,
and local CSV metric logging.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)


class Trainer:
    """Manages the full training life-cycle for a binary image classifier.

    Args:
        model:           ``nn.Module`` to train.
        criterion:       Loss callable ``(logits, labels) -> scalar``.
        cfg:             Configuration dataclass (must expose all fields listed
                         in the YAML configs).
        device:          PyTorch device.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion,
        cfg,
        device: torch.device,
    ) -> None:
        self.model     = model.to(device)
        self.criterion = criterion
        self.cfg       = cfg
        self.device    = device

        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.scaler    = GradScaler() if cfg.use_amp else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Run the training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
        """
        cfg = self.cfg
        os.makedirs(cfg.save_dir, exist_ok=True)

        # Local CSV log: outputs/metrics/<run_name>_<timestamp>_history.csv
        import datetime
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("outputs/metrics")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{cfg.run_name}_{self.timestamp}_history.csv"
        csv_fields = ["epoch", "train_loss", "val_loss", "val_roc_auc",
                      "val_accuracy", "val_precision", "val_recall", "val_f1"]
        log_file = open(log_path, "w", newline="")
        writer = csv.DictWriter(log_file, fieldnames=csv_fields)
        writer.writeheader()

        best_val_loss    = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{cfg.epochs}")

            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_preds, val_labels = self._eval_one_epoch(val_loader)

            metrics = self._compute_metrics(val_preds, val_labels)

            print(
                f"  Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {metrics['roc_auc']:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f}"
            )

            writer.writerow({
                "epoch":        epoch,
                "train_loss":   round(train_loss, 6),
                "val_loss":     round(val_loss, 6),
                "val_roc_auc":  round(metrics["roc_auc"], 6),
                "val_accuracy": round(metrics["accuracy"], 6),
                "val_precision":round(metrics["precision"], 6),
                "val_recall":   round(metrics["recall"], 6),
                "val_f1":       round(metrics["f1"], 6),
            })
            log_file.flush()

            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss       = val_loss
                patience_counter    = 0
                self.best_ckpt_path = Path(cfg.save_dir) / f"best_model_{self.timestamp}.pth"
                torch.save(self.model.state_dict(), self.best_ckpt_path)
                print(f"  Checkpoint saved -> {self.best_ckpt_path}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience {patience_counter}/{cfg.patience}")

            if patience_counter >= cfg.patience:
                print(f"  Early stopping triggered.")
                break

        log_file.close()
        print(f"\nTraining complete. Epoch log saved -> {log_path}")

    def evaluate(self, test_loader: DataLoader, checkpoint_path: str | Path | None = None) -> dict:
        """Evaluate the model on a held-out test set.

        Args:
            test_loader:      Test DataLoader.
            checkpoint_path:  Optional path to a ``.pth`` checkpoint to load
                              before evaluation. If ``None``, uses current weights.

        Returns:
            Dictionary of test metrics.
        """
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Loaded checkpoint: {checkpoint_path}")

        test_loss, test_preds, test_labels = self._eval_one_epoch(test_loader)
        metrics = self._compute_metrics(test_preds, test_labels)

        print("\n--- Test Set Results ---")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

        result   = {"loss": test_loss, **metrics}
        out      = {k: float(v) if not hasattr(v, "tolist") else v.tolist() for k, v in result.items()}
        ts = getattr(self, "timestamp", "unknown")
        json_path = Path(self.cfg.save_dir) / f"test_metrics_{ts}.json"
        with open(json_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  Test metrics saved -> {json_path}")

        # Save per-sample probs and labels for ROC curve plotting
        npz_path = Path(self.cfg.save_dir) / f"test_preds_{ts}.npz"
        np.savez(npz_path, probs=test_preds, labels=test_labels)
        print(f"  Test predictions saved -> {npz_path}")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.cfg.use_amp):
                logits = self.model(inputs)
                loss   = self.criterion(logits, labels)

            if self.cfg.use_amp:
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_one_epoch(
        self, loader: DataLoader
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_probs:  list[float] = []
        all_labels: list[int]   = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.cfg.use_amp):
                    logits = self.model(inputs)
                    loss   = self.criterion(logits, labels)

                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)[:, 1]  # P(AI)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        return avg_loss, np.array(all_probs), np.array(all_labels)

    @staticmethod
    def _compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
        preds  = (probs > 0.5).astype(int)
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        return {
            "roc_auc":          roc_auc_score(labels, probs),
            "accuracy":         report["accuracy"],
            "precision":        report["macro avg"]["precision"],
            "recall":           report["macro avg"]["recall"],
            "f1":               report["macro avg"]["f1-score"],
            "confusion_matrix": confusion_matrix(labels, preds),
        }

    def _build_optimizer(self) -> AdamW:
        cfg    = self.cfg
        head_params     = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = any(k in name for k in ("fc", "heads", "classifier", "head"))
            if is_head:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [{"params": head_params, "lr": cfg.lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})

        return AdamW(param_groups, weight_decay=cfg.weight_decay)
