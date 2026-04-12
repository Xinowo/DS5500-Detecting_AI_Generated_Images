"""
demo/app.py  —  Gradio Web Interface for AI-Generated Image Detection
======================================================================

Launches a local web app where a user can upload any image and instantly see:
  • Binary verdict + confidence probability from ResNet-50 and ViT-B/16
  • Grad-CAM attention heatmaps showing where each model "looks"

Run from the project root:
    python demo/app.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from models.resnet import build_resnet50
from models.vit import build_vit_b16

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RESNET_CKPT = ROOT / "checkpoints" / "resnet50" / "best_model_resnet50.pth"
VIT_CKPT    = ROOT / "checkpoints" / "vit_b16"  / "best_model_20260317_220741.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Pick a handful of example images from the test split (shown at the bottom)
_EXAMPLES_DIR = ROOT / "data" / "sampled_data_5k" / "test"
EXAMPLE_IMAGES: list[str] = []
if _EXAMPLES_DIR.exists():
    _all = sorted(_EXAMPLES_DIR.glob("*.jpg"))
    # Spread picks across the folder so examples look visually varied
    _step = max(1, len(_all) // 6)
    EXAMPLE_IMAGES = [str(p) for p in _all[::_step][:6]]

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model cache — loaded once on first inference
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _vit_reshape_transform(
    tensor: torch.Tensor, height: int = 14, width: int = 14
) -> torch.Tensor:
    """Convert ViT patch tokens → spatial feature map for Grad-CAM."""
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    return result.permute(0, 3, 1, 2)


def _load_models() -> None:
    """Build both models + GradCAM wrappers and cache them."""
    global _cache
    if _cache:
        return

    # Suppress the "Trainable / Total" print from model builders
    with redirect_stdout(io.StringIO()):
        resnet = build_resnet50(freeze_backbone=False, num_classes=2)
    state = torch.load(RESNET_CKPT, map_location=DEVICE, weights_only=True)
    resnet.load_state_dict(state)
    resnet.to(DEVICE).eval()
    cam_resnet = GradCAM(model=resnet, target_layers=[resnet.layer4[-1]])

    with redirect_stdout(io.StringIO()):
        vit = build_vit_b16(freeze_backbone=False, num_classes=2)
    state = torch.load(VIT_CKPT, map_location=DEVICE, weights_only=True)
    vit.load_state_dict(state)
    vit.to(DEVICE).eval()
    cam_vit = GradCAM(
        model=vit,
        target_layers=[vit.encoder.layers[-1].ln_1],
        reshape_transform=_vit_reshape_transform,
    )

    _cache.update(
        resnet=resnet, cam_resnet=cam_resnet,
        vit=vit, cam_vit=cam_vit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess(pil_img: Image.Image, size: int = 224) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)


def _run_model(
    pil_img: Image.Image,
    model: torch.nn.Module,
    cam: GradCAM,
    target_class: int = 1,
) -> tuple[dict[str, float], Image.Image]:
    """Run inference + Grad-CAM for one model.

    Returns:
        label_dict : ``{"AI-Generated": p_ai, "Real": p_real}``
        cam_image  : PIL RGB image with heatmap overlay
    """
    img_224 = pil_img.convert("RGB").resize((224, 224))
    tensor  = _preprocess(pil_img)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    prob_ai   = float(probs[1])
    prob_real = float(probs[0])

    # Grad-CAM on the AI-Generated class (index 1)
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )[0]

    rgb_arr = np.array(img_224, dtype=np.float32) / 255.0
    vis_arr = show_cam_on_image(rgb_arr, grayscale_cam, use_rgb=True)

    return {"AI-Generated": prob_ai, "Real": prob_real}, Image.fromarray(vis_arr)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict badge HTML
# ─────────────────────────────────────────────────────────────────────────────
def _verdict_html(label_dict: dict[str, float]) -> str:
    top  = max(label_dict, key=label_dict.get)
    conf = label_dict[top]
    if top == "AI-Generated":
        icon, color, bg = "🤖", "#d97706", "#fffbeb"
    else:
        icon, color, bg = "📷", "#059669", "#ecfdf5"
    return (
        f'<div style="text-align:center;padding:16px 10px 12px;'
        f'border-radius:12px;background:{bg};border:2px solid {color};'
        f'margin:6px 0 2px;">'
        f'<div style="font-size:2.4rem;line-height:1.2;">{icon}</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:{color};'
        f'margin:4px 0 2px;">{top}</div>'
        f'<div style="font-size:0.85rem;color:#6b7280;">'
        f'Confidence &nbsp;<strong>{conf:.1%}</strong></div>'
        f'</div>'
    )


_BLANK_BADGE = (
    '<div style="height:100px;border-radius:12px;background:#f3f4f6;'
    'border:2px dashed #e5e7eb;"></div>'
)


# ─────────────────────────────────────────────────────────────────────────────
# Main predict function
# ─────────────────────────────────────────────────────────────────────────────
def predict(pil_img: Image.Image | None):
    """Gradio callback — returns 6 outputs."""
    if pil_img is None:
        return _BLANK_BADGE, None, None, _BLANK_BADGE, None, None

    _load_models()

    lbl_r, cam_r = _run_model(pil_img, _cache["resnet"],  _cache["cam_resnet"])
    lbl_v, cam_v = _run_model(pil_img, _cache["vit"],     _cache["cam_vit"])

    return _verdict_html(lbl_r), lbl_r, cam_r, _verdict_html(lbl_v), lbl_v, cam_v


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Global ────────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* ── Hero header ───────────────────────────────────────────────────────── */
#hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
#hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
#hero p {
    color: #6b7280;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 640px;
    margin: 0 auto;
}

/* ── Section dividers ──────────────────────────────────────────────────── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.6rem 0 0.8rem;
}
.section-divider::before,
.section-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #e5e7eb;
}
.section-divider span {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9ca3af;
    white-space: nowrap;
}

/* ── Model cards ───────────────────────────────────────────────────────── */
.model-card {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 2px 14px rgba(0,0,0,.055) !important;
    padding: 1.1rem 1rem !important;
    background: #ffffff !important;
}
.card-title {
    text-align: center;
    font-size: 0.88rem;
    font-weight: 700;
    color: #4b5563;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding-bottom: 8px;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 10px;
}

/* ── Upload zone ───────────────────────────────────────────────────────── */
#upload-panel {
    border-radius: 16px !important;
    border: 2px dashed #c7d2fe !important;
    background: #fafbff !important;
}

/* ── Analyze button ────────────────────────────────────────────────────── */
#analyze-btn {
    width: 100%;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border-radius: 10px !important;
    margin-top: 0.65rem !important;
    padding: 0.65rem !important;
}

/* ── Cam caption ───────────────────────────────────────────────────────── */
.cam-note {
    text-align: center;
    font-size: 0.8rem;
    color: #9ca3af;
    margin: 0 0 0.9rem;
    line-height: 1.5;
}

/* ── Hide default gr.Image label bar ──────────────────────────────────── */
.cam-img .image-frame { border-radius: 10px; overflow: hidden; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="AI Image Detector") as demo:

    # ── Hero ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="hero">
        <h1>🔍 AI-Generated Image Detector</h1>
        <p>Upload any image and get instant predictions from two independent deep learning models —
        <strong>ResNet-50</strong> and <strong>ViT-B/16</strong> — along with
        Grad-CAM attention maps that reveal <em>where</em> each model looks.</p>
    </div>
    """)

    # ── Row 1: Upload + Predictions ──────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── (1,1) Upload Image ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=240):
            img_input = gr.Image(
                label="Upload Image",
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                height=340,
                elem_id="upload-panel",
            )
            analyze_btn = gr.Button(
                "⚡  Analyze Image",
                variant="primary",
                elem_id="analyze-btn",
            )

        # ── (1,2) ResNet Prediction ──────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes="model-card"):
                gr.HTML('<div class="card-title">ResNet-50</div>')
                verdict_resnet = gr.HTML(value=_BLANK_BADGE)
                label_resnet   = gr.Label(num_top_classes=2, label="Class Probabilities")

        # ── (1,3) ViT Prediction ─────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes="model-card"):
                gr.HTML('<div class="card-title">ViT-B / 16</div>')
                verdict_vit = gr.HTML(value=_BLANK_BADGE)
                label_vit   = gr.Label(num_top_classes=2, label="Class Probabilities")

    # ── Row 2: Examples + Grad-CAM ───────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── (2,1) Example Images ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=240):
            if EXAMPLE_IMAGES:
                gr.HTML('<div class="section-divider"><span>Try an Example</span></div>')
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_IMAGES],
                    inputs=[img_input],
                    label="",
                    examples_per_page=6,
                )

        # ── (2,2) ResNet-50 Grad-CAM ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML('<div class="section-divider"><span>Grad-CAM Attention Maps</span></div>')
            gr.HTML('<p class="cam-note"><strong>Warmer colors</strong> (red/yellow) indicate higher model attention.</p>')
            with gr.Column(elem_classes="model-card"):
                gr.HTML('<div class="card-title">ResNet-50 · Attention Heatmap</div>')
                cam_resnet_out = gr.Image(
                    type="pil",
                    label="",
                    show_label=False,
                    height=240,
                    interactive=False,
                    elem_classes="cam-img",
                )

        # ── (2,3) ViT Grad-CAM ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML('<div class="section-divider"><span>Grad-CAM Attention Maps</span></div>')
            gr.HTML('<p class="cam-note"><strong>Warmer colors</strong> (red/yellow) indicate higher model attention.</p>')
            with gr.Column(elem_classes="model-card"):
                gr.HTML('<div class="card-title">ViT-B/16 · Attention Heatmap</div>')
                cam_vit_out = gr.Image(
                    type="pil",
                    label="",
                    show_label=False,
                    height=240,
                    interactive=False,
                    elem_classes="cam-img",
                )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;margin-top:2rem;padding-top:1rem;
                border-top:1px solid #f3f4f6;color:#d1d5db;font-size:0.78rem;">
        DS 5500 · Detecting AI-Generated Images · Northeastern University 2026
    </div>
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────
    _outputs = [
        verdict_resnet, label_resnet, cam_resnet_out,
        verdict_vit,    label_vit,    cam_vit_out,
    ]

    analyze_btn.click(fn=predict, inputs=[img_input], outputs=_outputs)
    img_input.upload(fn=predict,  inputs=[img_input], outputs=_outputs)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # demo.launch(
    #     server_name="127.0.0.1",
    #     share=False,
    #     show_error=True,
    #     favicon_path=None,
    # )
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        allowed_paths=["../data"],
    )
