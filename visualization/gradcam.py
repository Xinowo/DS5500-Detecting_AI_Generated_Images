"""
Grad-CAM visualization for AIGI-Detection.

Supported models
----------------
- ResNet-50  : target layer ``model.layer4[-1]``
- ViT-B/16   : target layer ``model.encoder.layers[-1].ln_1``
               (uses a reshape transform to convert patch tokens → spatial map)

Default checkpoint discovery (relative to the project root)
-----------------------------------------------------------
- ResNet-50 : newest file matching ``checkpoints/resnet50/best_model*.pth``
- ViT-B/16  : newest file matching ``checkpoints/vit_b16/best_model*.pth``

Public API
----------
  load_resnet50(path, device)   -> nn.Module
  load_vit_b16(path, device)    -> nn.Module
  run_gradcam(img_path, model, cam, device, image_size, targets)
                                -> (PIL.Image, np.ndarray, np.ndarray, float)
  visualize(img_path, model_type, *, resnet_ckpt, vit_ckpt,
            save_dir, device, image_size)  -> None

CLI
---
  # Single image, both models side-by-side
  python visualization/gradcam.py --image path/to/image.jpg --model both

  # Whole folder, ResNet-50 only, save figures to <outputs_dir>/gradcam/
  python visualization/gradcam.py --folder data/my_imgs/ --model resnet50 \\
      --save-dir <outputs_dir>/gradcam/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project-level model builders (works when run from the project root)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.resnet import build_resnet50
from models.vit import build_vit_b16

# ---------------------------------------------------------------------------
# Default paths (relative to project root, i.e. the directory above this file)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]

RESNET_CHECKPOINT_DIR: Path = _ROOT / "checkpoints" / "resnet50"
VIT_CHECKPOINT_DIR:    Path = _ROOT / "checkpoints" / "vit_b16"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# ImageNet normalization (same as training)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _find_best_checkpoint(ckpt_dir: str | Path, model_label: str) -> Path:
    """Return the newest checkpoint matching ``best_model*.pth`` in ``ckpt_dir``."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"{model_label} checkpoint directory not found: {ckpt_dir}. "
            "Please download the checkpoints or train the model first."
        )

    candidates = sorted(ckpt_dir.glob("best_model*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching 'best_model*.pth' was found in: {ckpt_dir}. "
            "Please download the checkpoints or train the model first."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)

def load_resnet50(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load the trained ResNet-50 checkpoint.

    The model is rebuilt via ``build_resnet50`` (freeze_backbone=False so
    *all* weights are loaded), then the saved state-dict is applied.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ResNet-50 checkpoint not found: {path}. "
            "Please download the checkpoint or update the path."
        )
    model = build_resnet50(freeze_backbone=False, num_classes=2)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_vit_b16(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load the trained ViT-B/16 checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ViT-B/16 checkpoint not found: {path}. "
            "Please download the checkpoint or update the path."
        )
    model = build_vit_b16(freeze_backbone=False, num_classes=2)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# ViT reshape transform
# ---------------------------------------------------------------------------

def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Convert ViT patch-token sequence  →  spatial feature map.

    ViT-B/16 on 224×224 images produces 196 patch tokens (14×14) plus
    one CLS token.  We drop the CLS token and reshape the remainder into
    (B, C, H, W) so pytorch-grad-cam can generate a spatial heatmap.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _load_image(path: str | Path, image_size: int = 224):
    """Return ``(pil_image_resized, input_tensor)``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        img = Image.open(path).convert("RGB")
    except (Image.UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Cannot open image '{path}': {exc}. "
            "Please provide a valid JPEG, PNG, or WebP file."
        ) from exc
    img_resized = img.resize((image_size, image_size))
    tensor = _make_transform(image_size)(img).unsqueeze(0)
    return img_resized, tensor


# ---------------------------------------------------------------------------
# Core Grad-CAM runner
# ---------------------------------------------------------------------------

def run_gradcam(
    img_path: str | Path,
    model: nn.Module,
    cam: GradCAM,
    device: str = "cpu",
    image_size: int = 224,
    target_class: int = 1,
) -> tuple[Image.Image, np.ndarray, np.ndarray, float]:
    """Run Grad-CAM for one image.

    Args:
        img_path:     Path to the input image.
        model:        Loaded, eval-mode model.
        cam:          Initialised ``GradCAM`` instance.
        device:       Torch device string.
        image_size:   Spatial size used during preprocessing.
        target_class: Class index to explain (default 1 = "AI-generated").

    Returns:
        img_resized  : PIL image resized to ``image_size × image_size``.
        rgb_img      : Float32 array in [0, 1] for overlay rendering.
        visualization: Grad-CAM overlay as uint8 RGB array.
        prob         : Softmax probability for ``target_class``.
    """
    img_resized, input_tensor = _load_image(img_path, image_size)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)[0, target_class].item()

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(img_resized, dtype=np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return img_resized, rgb_img, visualization, prob


# ---------------------------------------------------------------------------
# High-level visualise function
# ---------------------------------------------------------------------------

def visualize(
    img_path: str | Path,
    model_type: str = "both",
    *,
    resnet_ckpt: str | Path | None = None,
    vit_ckpt:    str | Path | None = None,
    save_dir:    Optional[str | Path] = None,
    device:      Optional[str] = None,
    image_size:  int = 224,
) -> None:
    """Generate and display (or save) Grad-CAM overlays for one image.

    Args:
        img_path:    Path to the image file.
        model_type:  One of ``"resnet50"``, ``"vit"``, or ``"both"``.
        resnet_ckpt: Path to the ResNet-50 checkpoint.
        vit_ckpt:    Path to the ViT-B/16 checkpoint.
        save_dir:    Directory to save the figure.  If ``None``, the figure
                     is displayed interactively.
        device:      Torch device string.  Auto-detected when ``None``.
        image_size:  Resize target (default 224).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_type = model_type.lower()
    if resnet_ckpt is None:
        resnet_ckpt = _find_best_checkpoint(RESNET_CHECKPOINT_DIR, "ResNet-50")
    if vit_ckpt is None:
        vit_ckpt = _find_best_checkpoint(VIT_CHECKPOINT_DIR, "ViT-B/16")

    panels: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    if model_type in ("resnet50", "both"):
        resnet = load_resnet50(resnet_ckpt, device)
        cam_r  = GradCAM(model=resnet, target_layers=[resnet.layer4[-1]])
        _, rgb_img, vis, prob = run_gradcam(img_path, resnet, cam_r, device, image_size)
        panels.append(("ResNet-50 Grad-CAM", rgb_img, vis, prob))

    if model_type in ("vit", "both"):
        vit   = load_vit_b16(vit_ckpt, device)
        t_layer = vit.encoder.layers[-1].ln_1
        cam_v = GradCAM(
            model=vit,
            target_layers=[t_layer],
            reshape_transform=_vit_reshape_transform,
        )
        _, rgb_img, vis, prob = run_gradcam(img_path, vit, cam_v, device, image_size)
        panels.append(("ViT-B/16 Grad-CAM", rgb_img, vis, prob))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    n_models = len(panels)
    fig, axes = plt.subplots(1, n_models * 2, figsize=(6 * n_models * 2, 5))
    if n_models * 2 == 2:
        axes = [axes[0], axes[1]]

    img_name = Path(img_path).name
    fig.suptitle(img_name, fontsize=13)

    for i, (title, rgb_img, vis, prob) in enumerate(panels):
        ax_orig = axes[i * 2]
        ax_cam  = axes[i * 2 + 1]

        ax_orig.imshow(rgb_img)
        ax_orig.set_title("Original")
        ax_orig.axis("off")

        ax_cam.imshow(vis)
        ax_cam.set_title(f"{title}\nAI prob: {prob:.4f}")
        ax_cam.axis("off")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem
        out_path = save_dir / f"gradcam_{model_type}_{stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_folder(
    folder: str | Path,
    model_type: str = "both",
    *,
    resnet_ckpt: str | Path | None = None,
    vit_ckpt:    str | Path | None = None,
    save_dir:    Optional[str | Path] = None,
    device:      Optional[str] = None,
    image_size:  int = 224,
) -> None:
    """Run Grad-CAM on every image in *folder*.

    Args:
        folder:     Directory containing images.
        model_type: ``"resnet50"``, ``"vit"``, or ``"both"``.
        (remaining args identical to :func:`visualize`)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_type = model_type.lower()
    if resnet_ckpt is None:
        resnet_ckpt = _find_best_checkpoint(RESNET_CHECKPOINT_DIR, "ResNet-50")
    if vit_ckpt is None:
        vit_ckpt = _find_best_checkpoint(VIT_CHECKPOINT_DIR, "ViT-B/16")

    # Build models once, reuse across images
    resnet = cam_r = vit = cam_v = None

    if model_type in ("resnet50", "both"):
        resnet = load_resnet50(resnet_ckpt, device)
        cam_r  = GradCAM(model=resnet, target_layers=[resnet.layer4[-1]])

    if model_type in ("vit", "both"):
        vit   = load_vit_b16(vit_ckpt, device)
        cam_v = GradCAM(
            model=vit,
            target_layers=[vit.encoder.layers[-1].ln_1],
            reshape_transform=_vit_reshape_transform,
        )

    image_files = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in: {folder}")
        return

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        panels: list[tuple[str, np.ndarray, np.ndarray, float]] = []

        if cam_r is not None:
            _, rgb, vis, prob = run_gradcam(img_path, resnet, cam_r, device, image_size)
            panels.append(("ResNet-50 Grad-CAM", rgb, vis, prob))
            print(f"  [ResNet-50] AI prob = {prob:.4f}")

        if cam_v is not None:
            _, rgb, vis, prob = run_gradcam(img_path, vit, cam_v, device, image_size)
            panels.append(("ViT-B/16 Grad-CAM", rgb, vis, prob))
            print(f"  [ViT-B/16] AI prob = {prob:.4f}")

        n_models = len(panels)
        fig, axes = plt.subplots(1, n_models * 2, figsize=(6 * n_models * 2, 5))
        if n_models * 2 == 2:
            axes = [axes[0], axes[1]]

        fig.suptitle(img_path.name, fontsize=13)

        for i, (title, rgb, vis, prob) in enumerate(panels):
            axes[i * 2].imshow(rgb)
            axes[i * 2].set_title("Original")
            axes[i * 2].axis("off")
            axes[i * 2 + 1].imshow(vis)
            axes[i * 2 + 1].set_title(f"{title}\nAI prob: {prob:.4f}")
            axes[i * 2 + 1].axis("off")

        plt.tight_layout()

        if save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            out_path = out / f"gradcam_{model_type}_{img_path.stem}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {out_path}")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for AIGI-Detection (ResNet-50 / ViT-B/16)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file.")
    group.add_argument("--folder", type=str, help="Path to a folder of images.")

    parser.add_argument(
        "--model", type=str, default="both",
        choices=["resnet50", "vit", "both"],
        help="Which model(s) to use (default: both).",
    )
    parser.add_argument(
        "--resnet-ckpt", type=str, default=None,
        help="Path to the ResNet-50 checkpoint. If omitted, auto-discovers the newest best_model*.pth under checkpoints/resnet50/.",
    )
    parser.add_argument(
        "--vit-ckpt", type=str, default=None,
        help="Path to the ViT-B/16 checkpoint. If omitted, auto-discovers the newest best_model*.pth under checkpoints/vit_b16/.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save output figures. If omitted, figures are shown interactively.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help='Torch device, e.g. "cuda" or "cpu". Auto-detected when omitted.',
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Resize images to this spatial size (default: 224).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    kwargs = dict(
        model_type  = args.model,
        resnet_ckpt = args.resnet_ckpt,
        vit_ckpt    = args.vit_ckpt,
        save_dir    = args.save_dir,
        device      = args.device,
        image_size  = args.image_size,
    )

    if args.image:
        visualize(args.image, **kwargs)
    else:
        visualize_folder(args.folder, **kwargs)
