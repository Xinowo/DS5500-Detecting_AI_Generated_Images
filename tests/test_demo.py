"""Smoke tests for the Gradio demo entry-point (demo/app.py).

These tests deliberately avoid launching a server or loading checkpoints.
They only verify module-level structure so they run fast on any machine.

Tests that require gradio to be installed are skipped automatically when
it is not present (e.g. in base conda environments without demo extras).
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Detect whether gradio is installed in the current environment.
_gradio_available = importlib.util.find_spec("gradio") is not None
_skip_no_gradio = pytest.mark.skipif(
    not _gradio_available,
    reason="gradio not installed — run `pip install gradio` to enable these 6 tests",
)

# Resolve the app source file relative to this test file's location.
_APP_SRC = Path(__file__).resolve().parents[1] / "demo" / "app.py"


def _make_synthetic_image(size: int = 64) -> Image.Image:
    """Return a random RGB PIL image for testing."""
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@_skip_no_gradio
def test_demo_module_imports():
    """demo.app must be importable without errors or side-effects."""
    mod = importlib.import_module("demo.app")
    # The top-level Gradio Blocks object must exist
    assert hasattr(mod, "demo"), "Expected 'demo' Blocks object in demo.app"


@_skip_no_gradio
def test_checkpoint_dir_constants_are_paths():
    """Checkpoint directory constants must be Path objects inside checkpoints/."""
    from demo.app import RESNET_CKPT_DIR, VIT_CKPT_DIR  # noqa: PLC0415

    assert isinstance(RESNET_CKPT_DIR, Path), "RESNET_CKPT_DIR should be a Path"
    assert isinstance(VIT_CKPT_DIR, Path), "VIT_CKPT_DIR should be a Path"
    assert "checkpoints" in RESNET_CKPT_DIR.parts, "RESNET_CKPT_DIR should be under checkpoints/"
    assert "checkpoints" in VIT_CKPT_DIR.parts, "VIT_CKPT_DIR should be under checkpoints/"


@_skip_no_gradio
def test_find_best_checkpoint_prefers_matching_pattern(tmp_path):
    """Checkpoint auto-discovery should resolve a best_model*.pth file."""
    import demo.app as app  # noqa: PLC0415

    older = tmp_path / "best_model_20260317_220741.pth"
    newer = tmp_path / "best_model_20260415_101500.pth"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.touch()
    newer.touch()

    resolved = app._find_best_checkpoint(tmp_path, "ResNet-50")
    assert resolved == newer


def test_server_port_matches_docs():
    """Confirm demo/app.py hard-codes port 7862, consistent with demo/README.md.

    Reads the source file as plain text — no gradio import required.
    """
    assert _APP_SRC.exists(), f"demo/app.py not found at {_APP_SRC}"
    src = _APP_SRC.read_text(encoding="utf-8")
    assert "server_port=7862" in src, (
        "Expected server_port=7862 in demo/app.py — "
        "update demo/README.md if the port changes"
    )


@_skip_no_gradio
def test_predict_none_input_returns_blank_badges():
    """predict(None) must return 6 outputs with blank badge HTML and None images."""
    from demo.app import predict, _BLANK_BADGE  # noqa: PLC0415

    result = predict(None)
    assert len(result) == 6, "predict() must return exactly 6 outputs"
    badge_r, lbl_r, cam_r, badge_v, lbl_v, cam_v = result
    assert badge_r == _BLANK_BADGE
    assert badge_v == _BLANK_BADGE
    assert lbl_r is None
    assert lbl_v is None
    assert cam_r is None
    assert cam_v is None


@_skip_no_gradio
def test_predict_valid_image_returns_correct_structure():
    """predict() with a synthetic image should return verdict HTML, label dicts,
    and PIL heatmap images — without loading real checkpoints."""
    import demo.app as app  # noqa: PLC0415

    fake_heatmap = _make_synthetic_image(224)
    fake_labels = {"AI-Generated": 0.7, "Real": 0.3}
    fake_cache = {
        "resnet": MagicMock(), "cam_resnet": MagicMock(),
        "vit": MagicMock(), "cam_vit": MagicMock(),
    }

    with (
        patch.object(app, "_load_models"),
        patch.object(app, "_cache", fake_cache),
        patch.object(app, "_run_model", return_value=(fake_labels, fake_heatmap)),
    ):
        result = app.predict(_make_synthetic_image())

    assert len(result) == 6, "predict() must return exactly 6 outputs"
    badge_r, lbl_r, cam_r, badge_v, lbl_v, cam_v = result

    # Verdict badges should be non-empty HTML strings
    assert isinstance(badge_r, str) and len(badge_r) > 0
    assert isinstance(badge_v, str) and len(badge_v) > 0

    # Label dicts must contain the two expected keys
    assert set(lbl_r.keys()) == {"AI-Generated", "Real"}
    assert set(lbl_v.keys()) == {"AI-Generated", "Real"}

    # Heatmaps should be PIL images
    assert isinstance(cam_r, Image.Image)
    assert isinstance(cam_v, Image.Image)


@_skip_no_gradio
def test_predict_inference_error_returns_error_html():
    """When _run_model raises an exception, predict() must return error HTML
    instead of propagating the exception (graceful degradation from A1)."""
    import demo.app as app  # noqa: PLC0415

    fake_cache = {
        "resnet": MagicMock(), "cam_resnet": MagicMock(),
        "vit": MagicMock(), "cam_vit": MagicMock(),
    }

    with (
        patch.object(app, "_load_models"),
        patch.object(app, "_cache", fake_cache),
        patch.object(app, "_run_model", side_effect=RuntimeError("GPU OOM")),
    ):
        result = app.predict(_make_synthetic_image())

    assert len(result) == 6
    badge_r, lbl_r, cam_r, badge_v, lbl_v, cam_v = result

    # Both badges should contain error information
    assert "Error" in badge_r or "error" in badge_r.lower()
    assert lbl_r is None
    assert cam_r is None


@_skip_no_gradio
def test_preprocess_rejects_corrupt_data():
    """_preprocess() must raise ValueError for a non-image PIL object
    constructed from invalid bytes."""
    import demo.app as app  # noqa: PLC0415

    # Build a mock PIL image whose .convert() raises UnidentifiedImageError
    bad_img = MagicMock(spec=Image.Image)
    bad_img.convert.side_effect = Image.UnidentifiedImageError("not an image")

    with pytest.raises((ValueError, Exception)):
        app._preprocess(bad_img)
