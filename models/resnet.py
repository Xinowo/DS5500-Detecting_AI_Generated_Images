"""
ResNet-50 builder for binary AI-generated image classification.

Loads pre-trained ImageNet weights from torchvision, replaces the
fully-connected head with a 2-class linear layer, and optionally
freezes the backbone for linear-probe training.
"""

from __future__ import annotations

import logging
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


def build_resnet50(
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    num_classes: int = 2,
) -> nn.Module:
    """Return a ResNet-50 adapted for binary classification.

    Args:
        freeze_backbone:        If ``True``, all backbone parameters are frozen.
        unfreeze_last_n_blocks: Number of residual stages to unfreeze from the
                                end (layer4=1, layer3+4=2, …).  Only relevant
                                when ``freeze_backbone`` is ``True``.
        num_classes:            Number of output classes (default 2).

    Returns:
        Configured ``nn.Module``.
    """
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        # Freeze everything, then selectively unfreeze
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        if unfreeze_last_n_blocks > 0:
            stages = [model.layer4, model.layer3, model.layer2, model.layer1]
            for stage in stages[:unfreeze_last_n_blocks]:
                for param in stage.parameters():
                    param.requires_grad = True

    _print_param_summary(model)
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _print_param_summary(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("[ResNet-50] Trainable: %s / Total: %s (%.4f%%)", f"{trainable:,}", f"{total:,}", trainable / total * 100)
