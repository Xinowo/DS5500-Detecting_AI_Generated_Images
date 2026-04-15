# Visualization

This folder contains two CLI tools for inspecting model behavior.

| File | Purpose |
|---|---|
| `gradcam.py` | Grad-CAM heatmaps for ResNet-50 and ViT-B/16 |
| `visualize.py` | Training curves, confusion matrix, ROC curve |

---

## Grad-CAM (`gradcam.py`)

Requires both model checkpoint directories to contain at least one `best_model*.pth` file (or pass custom paths via flags).  
All commands are run from the **project root**.

### Quick-start

```bash
# Single image — both models side-by-side (interactive window)
python visualization/gradcam.py --image data/sampled_data_5k/test/some_image.jpg --model both

# Single image — ResNet-50 only, save PNG
python visualization/gradcam.py --image data/sampled_data_5k/test/some_image.jpg \
    --model resnet50 --save-dir <outputs_dir>/gradcam/

# Whole folder — ViT-B/16 only, save all figures
python visualization/gradcam.py --folder data/sampled_data_5k/test/ \
    --model vit --save-dir <outputs_dir>/gradcam/

# Custom checkpoint paths
python visualization/gradcam.py --image path/to/image.jpg --model both \
    --resnet-ckpt path/to/best_model_<timestamp>.pth \
    --vit-ckpt    path/to/best_model_<timestamp>.pth
```

### Flag reference

| Flag | Default | Description |
|---|---|---|
| `--image` / `--folder` | — | Single image or directory (mutually exclusive, one required) |
| `--model` | `both` | `resnet50`, `vit`, or `both` |
| `--resnet-ckpt` | *(auto)* | ResNet-50 checkpoint; if omitted, auto-discovers the newest `best_model*.pth` in `checkpoints/resnet50/` |
| `--vit-ckpt` | *(auto)* | ViT-B/16 checkpoint; if omitted, auto-discovers the newest `best_model*.pth` in `checkpoints/vit_b16/` |
| `--save-dir` | *(interactive)* | Directory to save PNG figures; omit to display interactively |
| `--device` | *(auto)* | `cuda` or `cpu` |
| `--image-size` | `224` | Resize target in pixels |

---

## Training plots (`visualize.py`)

Called automatically at the end of each training run. Figures are saved under
the configured `outputs_dir`, typically in `<outputs_dir>/figures/`.

| File | Contents |
|---|---|
| `<timestamp>_<run_name>_training_curves.png` | Train/val loss and val accuracy per epoch |
| `<timestamp>_<run_name>_confusion_matrix.png` | Test-set confusion matrix |
| `<timestamp>_<run_name>_roc_curve.png` | Test-set ROC curve with AUC annotation |

---

## Training output files reference

### `checkpoints/<model>/`

| File | Contents |
|---|---|
| `best_model_<timestamp>.pth` | Best checkpoint (lowest val loss) |
| `config.yaml` | Config used for this run |
| `test_metrics_<timestamp>.json` | Test accuracy, AUC, F1, confusion matrix |
| `test_preds_<timestamp>.npz` | Per-sample predicted probabilities and true labels |

### `<outputs_dir>/metrics/`

| File | Contents |
|---|---|
| `<run_name>_<timestamp>_history.csv` | Per-epoch train loss, val loss, val metrics |

All files from the same run share the same `<timestamp>` (`YYYYMMDD_HHMMSS`).
