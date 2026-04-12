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

import pytest

# Detect whether gradio is installed in the current environment.
_gradio_available = importlib.util.find_spec("gradio") is not None
_skip_no_gradio = pytest.mark.skipif(
    not _gradio_available, reason="gradio not installed in this environment"
)

# Resolve the app source file relative to this test file's location.
_APP_SRC = Path(__file__).resolve().parents[1] / "demo" / "app.py"


@_skip_no_gradio
def test_demo_module_imports():
    """demo.app must be importable without errors or side-effects."""
    mod = importlib.import_module("demo.app")
    # The top-level Gradio Blocks object must exist
    assert hasattr(mod, "demo"), "Expected 'demo' Blocks object in demo.app"


@_skip_no_gradio
def test_checkpoint_path_constants_are_paths():
    """RESNET_CKPT and VIT_CKPT must be Path objects pointing inside checkpoints/."""
    from demo.app import RESNET_CKPT, VIT_CKPT  # noqa: PLC0415

    assert isinstance(RESNET_CKPT, Path), "RESNET_CKPT should be a Path"
    assert isinstance(VIT_CKPT, Path), "VIT_CKPT should be a Path"
    assert "checkpoints" in RESNET_CKPT.parts, "RESNET_CKPT should be under checkpoints/"
    assert "checkpoints" in VIT_CKPT.parts, "VIT_CKPT should be under checkpoints/"


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
