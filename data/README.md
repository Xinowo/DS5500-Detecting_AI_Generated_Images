# Data

---

## Source

Kaggle: [AI vs. Human-Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)

- 79,950 labelled images in `train_data/`
- Binary labels: `1` = AI-generated, `0` = human/real
- 50/50 class balance
- Schema: `file_name`, `label`

The original `test_data_v2/` folder has no labels and is not used.

---

## Working subset

We sample 5,000 images to keep training times manageable on small GPU allocations.
The full dataset is available in the Colab notebooks.

```
data/sampled_data_5k/
├── train/        # 3,000 images
├── validation/   # 1,000 images
└── test/         # 1,000 images
```

This folder is gitignored. Download from Kaggle and place images here before training.

---

## Train / val / test split

Splits are produced by `prepare_splits()` in `dataset.py` using
`sklearn.model_selection.train_test_split` with `stratify=labels`.

| Split | Size  | Ratio |
|-------|-------|-------|
| Train | 3,000 | 60 %  |
| Val   | 1,000 | 20 %  |
| Test  | 1,000 | 20 %  |

- **Stratified:** the 50/50 class balance is preserved in every split.
- **Seed:** `random_state=42` — rerunning produces the same split.
- **Saved to:** `data/splits/df_train.csv`, `df_val.csv`, `df_test.csv`

The CSVs are committed so training runs are reproducible without re-splitting.
No image appears in more than one split (verified by the test suite).

---

## Image transforms

Two pipelines, returned by `get_transforms()` in `dataset.py`:

### Training

| Step | Reason |
|------|--------|
| `RandomResizedCrop(224)` | Random scale and position — teaches the model to be location-invariant |
| `RandomHorizontalFlip()` | Doubles effective dataset size for symmetric subjects |
| `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` | Mild color variation to reduce sensitivity to lighting and post-processing |
| `ToTensor()` | Convert PIL image to `[0, 1]` float tensor |
| `Normalize(ImageNet mean/std)` | Required: both backbones were pretrained on ImageNet with these statistics |

### Validation / test

| Step | Reason |
|------|--------|
| `Resize(256)` | Slightly oversized before cropping |
| `CenterCrop(224)` | Deterministic crop — no random variation at eval time |
| `ToTensor()` | Convert PIL image to `[0, 1]` float tensor |
| `Normalize(ImageNet mean/std)` | Same as training |

ImageNet normalization constants:
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## Files

| File | Contents |
|---|---|
| `dataset.py` | `AIDataset`, `get_transforms`, `prepare_splits`, `get_dataloaders` |
| `splits/df_train.csv` | Pre-computed training split |
| `splits/df_val.csv` | Pre-computed validation split |
| `splits/df_test.csv` | Pre-computed test split |
