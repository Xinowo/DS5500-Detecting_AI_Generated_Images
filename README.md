# DS5500 - Detecting AI-Generated Images

**Team 2 | Northeastern University DS5500 Data Capstone | Spring 2026**

**Team Members:** Xin Wang, Jiajun Fang

---

## Problem Statement and Objectives

Generative AI models can now produce photorealistic images that are difficult to
distinguish from photographs taken by humans.  This project builds a binary image
classifier to detect whether a given image is **AI-generated (label 1)** or
**human/real (label 0)**.

Objectives:
- Benchmark transfer-learning approaches (frozen backbone → linear probe → fine-tuning)
- Compare CNN-based (ResNet-50) and transformer-based (ViT-B/16) architectures
- Establish a reproducible training and evaluation pipeline

---

## Results (5 k-sample baseline)

| Model      | Test Accuracy | Test ROC AUC | Epochs |
|------------|--------------|--------------|--------|
| ResNet-50  | 90.20 %      | 0.9662       | 20     |
| ViT-B/16   | 79.20 %      | 0.8692       | 5      |

Both runs use a frozen backbone (linear probe).  Fine-tuning is supported via
`unfreeze_last_n_blocks` in the config.

---

## Project Structure

```
DS5500-Detecting_AI_Generated_Images/
├── configs/                  # YAML hyperparameter configs (one per model)
│   ├── resnet50.yaml
│   ├── vit_b16.yaml
│   └── smoke_test.yaml       # CPU / quick sanity-check config
│
├── data/
│   ├── dataset.py            # AIDataset, transforms, split logic, DataLoader factory
│   ├── sampled_data_5k/      # 5 k-image local subset
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── splits/               # Pre-computed train/val/test split CSVs (gitignored)
│
├── docs/
│   ├── workflow.md           # End-to-end pipeline walkthrough
│   └── code_walk_requirements.md
│
├── models/
│   ├── resnet.py             # ResNet-50 builder
│   ├── vit.py                # ViT-B/16 builder
│   └── model_factory.py      # build_model() dispatcher
│
├── training/
│   ├── train.py              # CLI entry-point
│   └── trainer.py            # Trainer class (fit / evaluate)
│
├── visualization/
│   └── visualize.py          # Confusion matrix, ROC curve, training curve plots
│
├── notebooks/                # Google Colab demo notebooks (full dataset)
│   ├── AIGI-Detection-ResNet50.ipynb
│   └── AIGI-Detection_ViT.ipynb
│
├── checkpoints/              # Saved model weights (gitignored)
├── outputs/                  # Figures and metric JSON files (gitignored)
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
└── .gitignore
```

---

## Dataset

- **Source:** Kaggle - [AI vs. Human-Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)
- **Size:** 79,950 labelled images (`train_data/`)
- **Balance:** 50 % AI-generated / 50 % human
- **Schema:** `file_name`, `label` (1 = AI, 0 = human)
- **Working subset:** 5,000 images sampled (3,000 train / 1,000 val / 1,000 test), stored in `data/sampled_data_5k/`

The original test set (`test_data_v2`) has no labels and is not used.

---

## Methods Overview

| Stage | Approach |
|-------|----------|
| Pre-processing | Resize → RandomCrop/CenterCrop → ColorJitter (train only) → ImageNet normalisation |
| Models | ResNet-50 and ViT-B/16, both pre-trained on ImageNet |
| Training strategy | Frozen backbone (linear probe); selective backbone unfreezing available |
| Loss | Cross-entropy with label smoothing (0.1) |
| Optimiser | AdamW with cosine annealing LR |
| Regularisation | Gradient clipping, early stopping (patience = 5), AMP |

---

## How to Run

> **Environment note:**  
> The Jupyter notebooks in `notebooks/` were developed and run on **Google Colab** (full 79,950-image dataset, GPU).  
> The `training/` scripts can be run **locally** using the pre-sampled 5,000-image subset in `data/sampled_data_5k/`.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke test (CPU, no GPU required)

To verify the full pipeline runs without errors on a CPU machine:

```bash
python -m training.train --config configs/smoke_test.yaml
```

This runs 2 epochs with `batch_size=8`, `use_amp=false`, and `num_workers=0` — finishes in a few minutes on CPU.

### 3. Local training (pre-sampled data in `data/sampled_data_5k/`)

Split CSVs are already present in `data/splits/`.  Run directly from the repo root:

```bash
python -m training.train --config configs/resnet50.yaml
```

### 4. Google Colab — notebooks or full-dataset training

Open a notebook from `notebooks/` directly in Colab, or run the CLI with the full dataset by mounting Google Drive and passing paths as CLI flags:

```bash
python -m training.train \
    --config    configs/resnet50.yaml \
    --data_root /content/train_data \
    --csv_path  /content/drive/MyDrive/.../train.csv \
    --save_dir  /content/drive/MyDrive/checkpoints/resnet50
```

### 5. Outputs

After training completes, the following files are saved automatically:

**`checkpoints/<model>/`**

| File | Contents |
|---|---|
| `best_model_<timestamp>.pth` | Best checkpoint (lowest val loss) |
| `config.yaml` | Config used for this run |
| `test_metrics_<timestamp>.json` | Test set accuracy, AUC, F1, confusion matrix |
| `test_preds_<timestamp>.npz` | Per-sample predicted probs and true labels |

**`outputs/metrics/`**

| File | Contents |
|---|---|
| `<run_name>_<timestamp>_history.csv` | Per-epoch train loss, val loss, val metrics |

**`outputs/figures/`** (auto-generated after training)

| File | Contents |
|---|---|
| `<timestamp>_<run_name>_training_curves.png` | Train/val loss and val accuracy curves |
| `<timestamp>_<run_name>_confusion_matrix.png` | Test set confusion matrix |
| `<timestamp>_<run_name>_roc_curve.png` | Test set ROC curve with AUC |

All files from the same run share the same `<timestamp>` (`YYYYMMDD_HHMMSS`).

---

## Supported Models

| Config key        | Architecture    | Source         |
|-------------------|-----------------|----------------|
| `resnet50`        | ResNet-50       | torchvision    |
| `vit_b_16`        | ViT-B/16        | torchvision    |

Add new models by registering them in `models/model_factory.py`.

---


## Assumptions and Limitations

- **Dataset size:** Experiments use a 5,000-image sample (6 % of the full dataset).  Results may not generalise to the full distribution.
- **Class balance:** The 50/50 split is preserved in all sub-samples; results assume balanced evaluation.
- **ViT training duration:** ViT-B/16 was trained for 5 epochs vs. 20 for ResNet-50; a direct comparison is premature.
- **Backbone frozen:** Both models are evaluated in linear-probe mode only; full fine-tuning has not yet been tested.
- **Hardware:** Training was performed on a Google Colab Tesla T4 GPU.  Results may differ on other hardware.
- **No augmentation at inference:** Validation and test splits use deterministic center-crop transforms only.

---

## Current Progress and Next Steps

**Completed:**
- Data pipeline: sampling, stratified splits, DataLoaders with augmentation, split-CSV persistence
- Linear-probe baseline: ResNet-50 (~90 % accuracy, 0.97 AUC) and ViT-B/16 (~79 % accuracy, 0.87 AUC)
- Modular codebase with YAML configs and CLI training scripts

**Next Steps:**
- Scale training to the full 79,950-image dataset
- Fine-tune backbone layers (`unfreeze_last_n_blocks > 0`) for both architectures
- Apply stronger augmentations (`albumentations`) to improve generalisation
- Perform error analysis on misclassified test images
- Ensemble predictions from multiple models
