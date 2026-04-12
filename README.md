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
├── data/                           # Dataset, transforms, splits (see data/README.md)
│   ├── dataset.py                  # AIDataset, transforms, split logic, DataLoader factory
│   └── splits/                     # Pre-computed train/val/test split CSVs
│       ├── df_train.csv
│       ├── df_val.csv
│       └── df_test.csv
│
├── demo/
│   └── app.py                      # Gradio web demo (ResNet-50 + ViT + Grad-CAM)
│
├── models/                         # Model builders (see models/README.md)
│   ├── resnet.py                   # ResNet-50 builder
│   ├── vit.py                      # ViT-B/16 builder
│   └── model_factory.py            # build_model() dispatcher
│
├── training/                       # Training loop and config (see training/README.md)
│   ├── train.py                    # CLI entry-point
│   └── trainer.py                  # Trainer class (fit / evaluate)
│
├── visualization/                  # Grad-CAM and training-curve tools (see visualization/README.md)
│   ├── visualize.py                # Confusion matrix, ROC curve, training curve plots
│   └── gradcam.py                  # Grad-CAM overlays for ResNet-50 and ViT-B/16
│
├── notebooks/                      # Exploratory notebooks
│   ├── AIGI-Detection-ResNet50.ipynb
│   ├── AIGI-Detection_ViT.ipynb
│   └── GradCAM_workflow.ipynb
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

> **Note:** The following folders are gitignored and must be created locally before training.

```
data/sampled_data_5k/   # 5k-image working subset (download from Kaggle)
│   ├── train/          # 3,000 images
│   ├── validation/     # 1,000 images
│   └── test/           # 1,000 images
│
checkpoints/            # Created automatically during training
outputs/                # Created automatically during training
```

For HPC cluster artifact paths and the `aigi_runs/` layout, see [slurm/README.md](slurm/README.md).

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

```bash
cd /home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images
sbatch slurm/train_vit_b16.slurm
```

See [slurm/README.md](slurm/README.md) for interactive `srun` sessions, hyperparameter overrides, grid search, and artifact paths.

### 5. Google Colab — notebooks or full-dataset training

Open a notebook from `notebooks/` directly in Colab, or run the CLI with the full dataset by mounting Google Drive and passing paths as CLI flags:

```bash
python -m training.train \
    --config    configs/resnet50.yaml \
    --data_root /content/train_data \
    --csv_path  /content/drive/MyDrive/.../train.csv \
    --save_dir  /content/drive/MyDrive/checkpoints/resnet50
```

### 6. Grad-CAM visualisation

Checkpoints are loaded from their default paths in `checkpoints/`.

```bash
# Single image — both models (interactive)
python visualization/gradcam.py --image data/sampled_data_5k/test/some_image.jpg --model both

# Save figures to disk
python visualization/gradcam.py --image data/sampled_data_5k/test/some_image.jpg \
    --model resnet50 --save-dir outputs/gradcam/
```

See [visualization/README.md](visualization/README.md) for all flags and usage patterns.

---

### 7. Gradio Demo

A one-click local web app that runs both models and shows Grad-CAM heatmaps alongside the verdict.

**Prerequisites**

1. Install dependencies (includes `gradio` and `grad-cam`):
   ```bash
   pip install -r requirements.txt
   ```
2. Both model checkpoints must be present at their default paths:
   ```
   checkpoints/resnet50/best_model_resnet50.pth
   checkpoints/vit_b16/best_model_20260317_220741.pth
   ```
   If you trained locally, they are created automatically.  
   To use downloaded HPC artifacts, copy the `.pth` files into the paths above.

**Launch**

```bash
# from the project root
python demo/app.py
```

Gradio prints a local URL, e.g.:

```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in any browser, upload an image (JPG / PNG), and the app will display:

- **Binary verdict + confidence score** from both ResNet-50 and ViT-B/16
- **Grad-CAM heatmaps** showing which regions each model used to make its decision

Example images from `data/sampled_data_5k/test/` are loaded automatically as quick-start examples (if the folder exists).

> **CPU-only machines:** Inference is slower but fully supported — no GPU required.

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



