"""Unit tests for models/model_factory.py, models/resnet.py, models/vit.py.

NOTE: The first run downloads pretrained ImageNet weights (~100 MB for ResNet-50,
~330 MB for ViT-B/16). Subsequent runs use the local cache.
"""

from __future__ import annotations

import pytest
import torch

from models import build_model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_resnet50_returns_module(self):
        model = build_model("resnet50")
        assert isinstance(model, torch.nn.Module)

    def test_vit_b16_returns_module(self):
        model = build_model("vit_b_16")
        assert isinstance(model, torch.nn.Module)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            build_model("does_not_exist")


# ---------------------------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------------------------

class TestResNet50:
    @pytest.fixture(scope="class")
    def model(self):
        return build_model("resnet50", freeze_backbone=True, unfreeze_last_n_blocks=0)

    def test_output_shape(self, model):
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_backbone_frozen(self, model):
        for name, param in model.named_parameters():
            if "fc" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_head_trainable(self, model):
        for name, param in model.named_parameters():
            if "fc" in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_unfreeze_last_1_block_unfreezes_layer4(self):
        m = build_model("resnet50", freeze_backbone=True, unfreeze_last_n_blocks=1)
        assert any(p.requires_grad for n, p in m.named_parameters() if "layer4" in n)

    def test_layer3_stays_frozen_when_1_block_unfrozen(self):
        m = build_model("resnet50", freeze_backbone=True, unfreeze_last_n_blocks=1)
        assert all(not p.requires_grad for n, p in m.named_parameters() if "layer3" in n)

    def test_num_classes_3(self):
        m = build_model("resnet50", num_classes=3)
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 224, 224))
        assert out.shape == (1, 3)


# ---------------------------------------------------------------------------
# ViT-B/16
# ---------------------------------------------------------------------------

class TestViTB16:
    @pytest.fixture(scope="class")
    def model(self):
        return build_model("vit_b_16", freeze_backbone=True, unfreeze_last_n_blocks=0)

    def test_output_shape(self, model):
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_backbone_frozen(self, model):
        for name, param in model.named_parameters():
            if "heads" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_head_trainable(self, model):
        for name, param in model.named_parameters():
            if "heads" in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_unfreeze_last_1_block(self):
        m = build_model("vit_b_16", freeze_backbone=True, unfreeze_last_n_blocks=1)
        # torchvision names blocks as encoder.layers.encoder_layer_0 … encoder_layer_11
        assert any(p.requires_grad for n, p in m.named_parameters() if "encoder_layer_11" in n)

    def test_num_classes_3(self):
        m = build_model("vit_b_16", num_classes=3)
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 224, 224))
        assert out.shape == (1, 3)
