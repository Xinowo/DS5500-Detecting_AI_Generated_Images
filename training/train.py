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
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Add project root to sys.path so local modules resolve on Colab
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset    import prepare_splits, get_dataloaders
from models          import build_model
from training.trainer import Trainer


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

    # W&B
    run_name: str        = "run"

    # Misc
    seed:     int = 42
    save_dir: str = "checkpoints"


def load_config(yaml_path: str) -> Config:
    """Merge YAML file into a Config dataclass."""
    with open(yaml_path, "r") as fh:
        overrides = yaml.safe_load(fh)
    cfg = Config()
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"[Warning] Unknown config key ignored: '{key}'")
    return cfg


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
    parser.add_argument("--config",     required=True, help="Path to YAML config file.")
    parser.add_argument("--data_root",  default=None,  help="Override data_root from config.")
    parser.add_argument("--splits_dir", default=None,  help="Override splits_dir from config.")
    parser.add_argument("--csv_path",   default=None,  help="Override csv_path from config.")
    parser.add_argument("--save_dir",   default=None,  help="Override save_dir from config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Command-line overrides
    if args.data_root:  cfg.data_root  = args.data_root
    if args.splits_dir: cfg.splits_dir = args.splits_dir
    if args.csv_path:   cfg.csv_path   = args.csv_path
    if args.save_dir:   cfg.save_dir   = args.save_dir

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}")

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
        print(f"[Data] Loaded pre-existing splits from {splits_dir}/")
    else:
        # Compute splits from the full CSV (Colab / full-dataset workflow)
        if not cfg.csv_path:
            raise ValueError("csv_path must be set when split CSVs are absent in splits_dir.")
        df = pd.read_csv(cfg.csv_path)
        print(f"[Data] CSV loaded: {len(df)} rows")
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
        print(f"[Data] Splits saved to {splits_dir}/")

    print(f"[Data] train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

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

    criterion = lambda logits, targets: F.cross_entropy(
        logits, targets.long(), label_smoothing=cfg.label_smoothing
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, Path(cfg.save_dir) / "config.yaml")
    print(f"[Config] Saved to {Path(cfg.save_dir) / 'config.yaml'}")

    trainer = Trainer(model=model, criterion=criterion, cfg=cfg, device=device)
    trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    ckpt_path = Path(cfg.save_dir) / "best_model.pth"
    trainer.evaluate(test_loader, checkpoint_path=ckpt_path)


if __name__ == "__main__":
    main()
