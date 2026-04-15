# Demo

A local Gradio web app that runs both models on a user-uploaded image and shows
Grad-CAM heatmaps alongside the classification verdict.

---

## Prerequisites

1. Install all dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   ```

2. Each model checkpoint directory must contain at least one file matching `best_model*.pth`:
   ```
   checkpoints/resnet50/best_model*.pth
   checkpoints/vit_b16/best_model*.pth
   ```
   The demo automatically loads the newest matching file in each directory when no explicit checkpoint path is provided.
   If you trained locally, they are created automatically by the training script.
   If you want to use the provided pretrained artifacts, download `checkpoints.zip`
   from the repo's **GitHub Releases** section and extract it at the repo root.

---

## Launch

Run from the **project root** (not from inside `demo/`):

```bash
python demo/app.py

# optional: explicitly choose checkpoint files
python demo/app.py \
    --resnet-ckpt path/to/best_model_<timestamp>.pth \
    --vit-ckpt path/to/best_model_<timestamp>.pth
```

Gradio will print a local URL:

```
Running on local URL:  http://127.0.0.1:7862
```

Open that URL in any browser.

---

## What you see

| Output | Description |
|--------|-------------|
| **Verdict** | AI-generated or Real, with a confidence score (0–1) |
| **ResNet-50 Grad-CAM** | Heatmap showing which spatial regions influenced ResNet-50's decision |
| **ViT-B/16 Grad-CAM** | Heatmap showing which patch regions influenced ViT-B/16's decision |

Example images are loaded from `data/sampled_data_5k/test/` if that optional
subset exists locally; otherwise the app falls back to `data/train_data/` plus
`data/splits/`.

> CPU-only machines are fully supported — inference will be slower but functional.
