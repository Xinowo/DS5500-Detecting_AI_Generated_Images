# Training

---

## Quick start

```bash
# Smoke test — CPU, 2 epochs, verifies the full pipeline works
python -m training.train --config configs/smoke_test.yaml

# ResNet-50 (GPU recommended)
python -m training.train --config configs/resnet50.yaml

# ViT-B/16 (GPU recommended)
python -m training.train --config configs/vit_b16.yaml
```

The script automatically saves the best checkpoint to `save_dir` and writes
training-curve / evaluation plots to `outputs_dir`.

---

## Config fields

All hyperparameters live in a YAML file under `configs/`.
Pass a different file with `--config path/to/config.yaml`.
Individual fields can be overridden on the command line (e.g. `--epochs 5`).

| Field | Description |
|-------|-------------|
| `data_root` | Path to the image folder (`data/sampled_data_5k`) |
| `splits_dir` | Path to the CSV split files (`data/splits`) |
| `model_name` | `"resnet50"` or `"vit_b_16"` |
| `freeze_backbone` | `true` = linear probe (head-only training) |
| `unfreeze_last_n_blocks` | Backbone blocks to unfreeze from the end; `0` = head only |
| `epochs` | Maximum number of training epochs |
| `batch_size` | Mini-batch size |
| `lr` | Learning rate for the classification head |
| `backbone_lr` | Learning rate for unfrozen backbone layers |
| `weight_decay` | L2 regularization coefficient for AdamW |
| `label_smoothing` | Cross-entropy label smoothing (0.1 = 10 %) |
| `patience` | Early-stopping patience in epochs (monitors **val loss**) |
| `grad_clip` | Max gradient norm for gradient clipping |
| `use_amp` | `true` = automatic mixed precision (faster on GPUs with Tensor Cores) |
| `eta_min` | Minimum LR floor for `CosineAnnealingLR` |
| `warmup_epochs` | Epochs to hold LR constant before cosine decay starts |
| `seed` | Global random seed — set once at startup |
| `save_dir` | Directory to write the best checkpoint `.pth` file |
| `outputs_dir` | Directory to write training curves and evaluation plots |

---

## Metrics

Three metrics are tracked during training and reported at final test evaluation:

| Metric | Why we use it |
|--------|--------------|
| **ROC-AUC** | Threshold-independent; measures the model's ability to rank AI-generated images above real ones regardless of the decision threshold. Logged every epoch and reported at final evaluation. |
| **F1** | Harmonic mean of precision and recall. Captures both false positives and false negatives equally — important if the cost of each type of mistake is similar. |
| **Accuracy** | Overall correctness rate. Straightforward to interpret and meaningful here because the test set is 50/50 balanced. |

Metrics are computed on the validation set at the end of every epoch and on the
test set once after training completes. The training-history CSV is saved to
`outputs_dir/metrics/`. The final test-metrics JSON and per-sample prediction
NPZ are saved to `save_dir` alongside the best checkpoint.

> **Early stopping** monitors **validation loss** (not ROC-AUC). Training halts
> when val loss has not improved for `patience` consecutive epochs.

---

## Reproducibility

Seed `42` is set globally at the start of every run by `seed_everything()`,
which seeds Python's `random`, `numpy`, `torch`, and `torch.cuda`, and sets
`torch.backends.cudnn.deterministic = True` / `benchmark = False` to enable
cuDNN deterministic mode. The pre-computed split CSVs in `data/splits/` are
committed, so re-running the same config on the same machine should produce
results very close to those reported above. Bit-identical reproduction is not
guaranteed because PyTorch's multi-process DataLoader workers and certain GPU
operations retain non-deterministic behaviour even with a fixed seed.

---

## Files

| File | Contents |
|---|---|
| `train.py` | CLI entry-point: config loading, config validation, `seed_everything`, calls `Trainer` |
| `trainer.py` | `Trainer` class: `fit()` loop, `evaluate()`, early stopping, metric logging |
