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

| Model      | Test Accuracy | Test ROC AUC | Epochs (early stop) | Training hardware |
|------------|--------------|--------------|---------------------|-------------------|
| ResNet-50  | 90.20 %      | 0.9662       | 20                  | Google Colab T4   |
| ViT-B/16   | 85.90 %      | 0.9294       | 11                  | HPC cluster (V100-SXM2) |

Both runs use a frozen backbone (linear probe).  Fine-tuning is supported via
`unfreeze_last_n_blocks` in the config.

---

## Project Structure

```
DS5500-Detecting_AI_Generated_Images/
├── configs/                        # YAML hyperparameter configs
│   ├── resnet50.yaml
│   ├── vit_b16.yaml
│   └── smoke_test.yaml             # CPU / quick sanity-check config
│
├── data/
│   ├── dataset.py                  # AIDataset, transforms, split logic, DataLoader factory
│   └── splits/                     # Pre-computed train/val/test split CSVs
│       ├── df_train.csv
│       ├── df_val.csv
│       └── df_test.csv
│
├── models/
│   ├── resnet.py                   # ResNet-50 builder
│   ├── vit.py                      # ViT-B/16 builder
│   └── model_factory.py            # build_model() dispatcher
│
├── training/
│   ├── train.py                    # CLI entry-point
│   └── trainer.py                  # Trainer class (fit / evaluate)
│
├── visualization/
│   └── visualize.py                # Confusion matrix, ROC curve, training curve plots
│
├── notebooks/                      # Google Colab demo notebooks (full dataset)
│   ├── AIGI-Detection-ResNet50.ipynb
│   └── AIGI-Detection_ViT.ipynb
│
├── slurm/                          # HPC job scripts (see slurm/README.md)
│   ├── train_resnet50.slurm
│   ├── train_vit_b16.slurm
│   ├── submit_all.sh
│   └── README.md
│
├── Final_Milestone_Report_Team_2.pdf
├── requirements.txt
└── .gitignore
```

> **Note:** The folders below are gitignored and not pushed to the repository.
> They must be created locally or on the cluster before training.

### Local-only folders (gitignored)

```
data/sampled_data_5k/           # 5 k-image working subset (download from Kaggle or shared drive)
│   ├── train/                  # 3,000 images
│   ├── validation/             # 1,000 images
│   └── test/                   # 1,000 images
│
checkpoints/                    # Local training artifacts (created automatically)
│   ├── resnet50/
│   │   ├── best_model_<timestamp>.pth
│   │   ├── config.yaml
│   │   ├── test_metrics_<timestamp>.json
│   │   └── test_preds_<timestamp>.npz
│   └── smoke_test/
│       ├── best_model.pth
│       ├── config.yaml
│       └── test_metrics.json
│
outputs/                        # Training metrics and figures (created automatically)
│   ├── metrics/
│   │   └── <run_name>_<timestamp>_history.csv
│   └── figures/
│       ├── <timestamp>_<run_name>_training_curves.png
│       ├── <timestamp>_<run_name>_confusion_matrix.png
│       └── <timestamp>_<run_name>_roc_curve.png
│
aigi_runs/                      # Downloaded HPC cluster run artifacts
│   └── aigi_runs/
│       ├── <RUN_ID>/           # e.g. 20260317_220724/ (ViT-B/16 linear-probe run)
│       │   ├── checkpoints/
│       │   │   ├── best_model_<timestamp>.pth
│       │   │   ├── config.yaml
│       │   │   ├── test_metrics_<timestamp>.json
│       │   │   └── test_preds_<timestamp>.npz
│       │   └── outputs/
│       │       ├── metrics/<run_name>_<timestamp>_history.csv
│       │       ├── figures/
│       │       └── train_console.log
│       └── latest_vit -> <RUN_ID>/  # symlink to most recent ViT run
```

On the **HPC cluster**, artifacts are written to `/scratch/$USER/DS5500_Data_Capstone/aigi_runs/<RUN_ID>/` and downloaded locally into `aigi_runs/` (see [slurm/README.md](slurm/README.md)).

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
> - **Local**: `training/` scripts run against the pre-sampled 5,000-image subset in `data/sampled_data_5k/`.  
> - **HPC cluster**: SLURM scripts in `slurm/` handle environment setup, scratch-storage paths, and logging automatically.  
> - **Google Colab**: notebooks in `notebooks/` use the full 79,950-image dataset with a GPU runtime.

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
python -m training.train --config configs/vit_b16.yaml
```

### 4. HPC / Cluster training

**Interactive (online) — `srun`:**
```bash
srun --partition=gpu --gres=gpu:v100-sxm2:1 --pty bash
cd /home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m training.train --config configs/vit_b16.yaml \
    --data_root /scratch/$USER/DS5500_Data_Capstone/data/sampled_data_5k \
    --num_workers 1
```

**Background (offline) — `sbatch`:**
```bash
cd /home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images
sbatch slurm/train_vit_b16.slurm
```

For hyperparameter overrides, grid search, and artifact paths see [slurm/README.md](slurm/README.md).

### 5. Google Colab — notebooks or full-dataset training

Open a notebook from `notebooks/` directly in Colab, or run the CLI with the full dataset by mounting Google Drive and passing paths as CLI flags:

```bash
python -m training.train \
    --config    configs/resnet50.yaml \
    --data_root /content/train_data \
    --csv_path  /content/drive/MyDrive/.../train.csv \
    --save_dir  /content/drive/MyDrive/checkpoints/resnet50
```

### 6. Outputs

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

- **Dataset size:** Experiments use a 5,000-image sample (~6 % of the full dataset).  Results may not generalise to the full distribution.
- **Class balance:** The 50/50 split is preserved in all sub-samples; results assume balanced evaluation.
- **Hardware difference:** ResNet-50 was trained on Google Colab (Tesla T4); ViT-B/16 was trained on the Northeastern Discovery HPC cluster (V100-SXM2).  Direct timing comparisons are not meaningful.
- **Backbone frozen:** Both models are evaluated in linear-probe mode only; full fine-tuning has not yet been tested.
- **No augmentation at inference:** Validation and test splits use deterministic center-crop transforms only.

---

## Current Progress and Next Steps

**Completed:**
- Data pipeline: sampling, stratified splits, DataLoaders with augmentation, split-CSV persistence
- Linear-probe baseline: ResNet-50 (90.20 % accuracy, 0.9662 AUC, 20 epochs) on Google Colab
- Linear-probe baseline: ViT-B/16 (85.90 % accuracy, 0.9294 AUC, 11 epochs) on HPC cluster
- HPC SLURM job scripts with automatic artifact management and `latest_vit` symlink
- Modular codebase with YAML configs, CLI training scripts, and full visualization pipeline
- Initial milestone report draft (`docs/draft_final_milestone_report_natural_language.md`)

**Next Steps:**
- Web demo for users to upload images and get AI-generated vs. real predictions
