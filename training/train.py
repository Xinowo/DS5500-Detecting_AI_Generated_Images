"""
Main entry-point for training an AIGI-Detection model.

Usage (Google Colab / terminal):
    python -m training.train --config configs/resnet50.yaml

The script:
  1. Loads the YAML config.
  2. Seeds all RNGs.
  3. Loads and splits the CSV data.
  4. Builds DataLoaders.
  5. Instantiates the model via model_factory.
  6. Runs the Trainer.fit() loop.
  7. Runs Trainer.evaluate() on the test set and saves metrics to outputs/.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Ensure project root is on sys.path.
# Needed when this file is executed directly (e.g. `python training/train.py`)
# or inside Colab/notebooks.  Has no effect when invoked correctly via
# `python -m training.train` from the project root (CWD is already on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset    import prepare_splits, get_dataloaders
from models          import build_model
from training.trainer import Trainer
from visualization.visualize import plot_training_curves, plot_confusion_matrix, plot_roc_curve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Data
    data_root:   str   = "data/sampled_data_5k"  # base dir; sub-dirs train/validation/test auto-detected
    splits_dir:  str   = "data/splits"            # dir with df_train/val/test.csv; if CSVs exist they are loaded directly
    csv_path:    str   = ""                        # full dataset CSV; only used when splits_dir CSVs are absent
    train_size:  int | None = None                 # None = use all rows in csv_path (ignored when loading from splits_dir)
    val_ratio:   float = 0.20
    test_ratio:  float = 0.20
    num_workers: int   = 2

    # Model
    model_name:             str  = "resnet50"
    freeze_backbone:        bool = True
    unfreeze_last_n_blocks: int  = 0

    # Training
    epochs:          int   = 20
    batch_size:      int   = 64
    lr:              float = 3e-4
    backbone_lr:     float = 1e-5
    weight_decay:    float = 1e-2
    label_smoothing: float = 0.1
    patience:        int   = 5
    grad_clip:       float = 1.0
    use_amp:         bool  = True
    eta_min:         float = 1e-5
    warmup_epochs:   int   = 0

    # Logging
    run_name: str        = "run"

    # Misc
    seed:        int = 42
    save_dir:    str = "checkpoints"
    outputs_dir: str = "outputs"  # base dir for metrics/figures/logs


def load_config(yaml_path: str) -> Config:
    """Merge YAML file into a Config dataclass."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        overrides = yaml.safe_load(fh)
    cfg = Config()
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logger.warning("Unknown config key ignored: '%s'", key)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Config) -> None:
    """Raise ValueError if any hyperparameter value is obviously wrong."""
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {cfg.epochs}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {cfg.batch_size}")
    if cfg.lr <= 0:
        raise ValueError(f"lr must be > 0, got {cfg.lr}")
    if cfg.backbone_lr <= 0:
        raise ValueError(f"backbone_lr must be > 0, got {cfg.backbone_lr}")
    if cfg.patience <= 0:
        raise ValueError(f"patience must be > 0, got {cfg.patience}")
    if not (0.0 <= cfg.val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {cfg.val_ratio}")
    if not (0.0 <= cfg.test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0, 1), got {cfg.test_ratio}")
    if cfg.val_ratio + cfg.test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0, "
            f"got {cfg.val_ratio} + {cfg.test_ratio} = {cfg.val_ratio + cfg.test_ratio}"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AIGI-Detection model.")
    parser.add_argument("--config",      required=True, help="Path to YAML config file.")
    parser.add_argument("--data_root",   default=None,  help="Override data_root from config.")
    parser.add_argument("--splits_dir",  default=None,  help="Override splits_dir from config.")
    parser.add_argument("--csv_path",    default=None,  help="Override csv_path from config.")
    parser.add_argument("--save_dir",    default=None,  help="Override save_dir from config.")
    parser.add_argument("--outputs_dir", default=None,  help="Override outputs_dir from config.")
    parser.add_argument("--num_workers", default=None,  type=int,   help="Override num_workers from config.")
    # Hyperparameter overrides
    parser.add_argument("--epochs",      default=None,  type=int,   help="Override epochs.")
    parser.add_argument("--batch_size",  default=None,  type=int,   help="Override batch_size.")
    parser.add_argument("--lr",          default=None,  type=float, help="Override lr.")
    parser.add_argument("--backbone_lr", default=None,  type=float, help="Override backbone_lr.")
    parser.add_argument("--weight_decay",default=None,  type=float, help="Override weight_decay.")
    parser.add_argument("--patience",    default=None,  type=int,   help="Override patience.")
    parser.add_argument("--run_name",    default=None,              help="Override run_name.")
    parser.add_argument("--unfreeze_last_n_blocks", default=None, type=int, help="Override unfreeze_last_n_blocks.")
    parser.add_argument("--eta_min",       default=None, type=float, help="Override eta_min (lr floor for CosineAnnealingLR).")
    parser.add_argument("--warmup_epochs", default=None, type=int,   help="Override warmup_epochs (epochs before cosine decay starts).")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pth checkpoint to warm-start from (e.g. Stage 1 best model).")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)

    # Command-line overrides
    if args.data_root:              cfg.data_root   = args.data_root
    if args.splits_dir:             cfg.splits_dir  = args.splits_dir
    if args.csv_path:               cfg.csv_path    = args.csv_path
    if args.save_dir:               cfg.save_dir    = args.save_dir
    if args.outputs_dir:            cfg.outputs_dir = args.outputs_dir
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.epochs      is not None: cfg.epochs      = args.epochs
    if args.batch_size  is not None: cfg.batch_size  = args.batch_size
    if args.lr          is not None: cfg.lr          = args.lr
    if args.backbone_lr is not None: cfg.backbone_lr = args.backbone_lr
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.patience    is not None: cfg.patience    = args.patience
    if args.run_name:                cfg.run_name    = args.run_name
    if args.unfreeze_last_n_blocks is not None: cfg.unfreeze_last_n_blocks = args.unfreeze_last_n_blocks
    if args.eta_min        is not None: cfg.eta_min        = args.eta_min
    if args.warmup_epochs  is not None: cfg.warmup_epochs  = args.warmup_epochs

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Config: %s", cfg)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    splits_dir = Path(cfg.splits_dir)
    train_csv  = splits_dir / "df_train.csv"
    val_csv    = splits_dir / "df_val.csv"
    test_csv   = splits_dir / "df_test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        # Pre-split CSVs exist — load directly (local / sampled-data workflow)
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)
        df_test  = pd.read_csv(test_csv)
        logger.info("[Data] Loaded pre-existing splits from %s/", splits_dir)
    else:
        # Compute splits from the full CSV (Colab / full-dataset workflow)
        if not cfg.csv_path:
            raise ValueError("csv_path must be set when split CSVs are absent in splits_dir.")
        df = pd.read_csv(cfg.csv_path)
        logger.info("[Data] CSV loaded: %d rows", len(df))
        df_train, df_val, df_test = prepare_splits(
            df         = df,
            train_size = cfg.train_size,
            val_ratio  = cfg.val_ratio,
            test_ratio = cfg.test_ratio,
            seed       = cfg.seed,
        )
        splits_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(  val_csv,   index=False)
        df_test.to_csv( test_csv,  index=False)
        logger.info("[Data] Splits saved to %s/", splits_dir)

    logger.info("[Data] train=%d  val=%d  test=%d", len(df_train), len(df_val), len(df_test))

    train_loader, val_loader, test_loader = get_dataloaders(
        df_train    = df_train,
        df_val      = df_val,
        df_test     = df_test,
        data_root   = cfg.data_root,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(
        model_name             = cfg.model_name,
        freeze_backbone        = cfg.freeze_backbone,
        unfreeze_last_n_blocks = cfg.unfreeze_last_n_blocks,
    )

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        logger.info("[Checkpoint] Warm-started from %s", args.checkpoint)

    criterion = lambda logits, targets: F.cross_entropy(
        logits, targets.long(), label_smoothing=cfg.label_smoothing
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    config_out = Path(cfg.save_dir) / "config.yaml"
    with open(config_out, "w") as _f:
        yaml.dump(vars(cfg), _f, default_flow_style=False, sort_keys=False)
    logger.info("[Config] Saved to %s", config_out)

    trainer = Trainer(model=model, criterion=criterion, cfg=cfg, device=device)
    trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    trainer.evaluate(test_loader, checkpoint_path=trainer.best_ckpt_path)

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    import json
    ts       = trainer.timestamp
    fig_dir  = Path(cfg.outputs_dir) / "figures"
    ckpt_dir = Path(cfg.save_dir)

    history_csv = Path(cfg.outputs_dir) / "metrics" / f"{cfg.run_name}_{ts}_history.csv"
    df = pd.read_csv(history_csv)
    plot_training_curves(
        history   = df.to_dict(orient="list"),
        save_path = fig_dir / f"{ts}_{cfg.run_name}_training_curves.png",
        title     = f"{cfg.run_name} Training Curves",
    )

    with open(ckpt_dir / f"test_metrics_{ts}.json") as f:
        metrics = json.load(f)
    plot_confusion_matrix(
        cm        = np.array(metrics["confusion_matrix"]),
        save_path = fig_dir / f"{ts}_{cfg.run_name}_confusion_matrix.png",
        title     = f"{cfg.run_name} Confusion Matrix",
    )

    preds = np.load(ckpt_dir / f"test_preds_{ts}.npz")
    plot_roc_curve(
        labels    = preds["labels"],
        probs     = preds["probs"],
        save_path = fig_dir / f"{ts}_{cfg.run_name}_roc_curve.png",
        title     = f"{cfg.run_name} ROC Curve",
    )

    logger.info("\nAll figures saved to %s/", fig_dir)

    # Explicitly release DataLoader worker processes and GPU memory so the
    # SLURM job exits promptly instead of lingering with an idle GPU.
    del train_loader, val_loader, test_loader
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[Cleanup] GPU memory released. Job exiting.")


if __name__ == "__main__":
    main()
