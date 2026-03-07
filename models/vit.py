"""
ViT-B/16 builder for binary AI-generated image classification.

Loads pre-trained ImageNet weights from torchvision, replaces the
classification head with a 2-class linear layer, and optionally
freezes the transformer backbone for linear-probe training.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


def build_vit_b16(
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    num_classes: int = 2,
) -> nn.Module:
    """Return a ViT-B/16 adapted for binary classification.

    Args:
        freeze_backbone:        If ``True``, all backbone parameters are frozen.
        unfreeze_last_n_blocks: Number of transformer encoder blocks to unfreeze
                                from the end of the sequence.
        num_classes:            Number of output classes (default 2).

    Returns:
        Configured ``nn.Module``.
    """
    weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
    model = tv_models.vit_b_16(weights=weights)

    # Replace classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the new head
        for param in model.heads.parameters():
            param.requires_grad = True

        if unfreeze_last_n_blocks > 0:
            for blk in model.encoder.layers[-unfreeze_last_n_blocks:]:
                for param in blk.parameters():
                    param.requires_grad = True

    _print_param_summary(model)
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _print_param_summary(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[ViT-B/16]  Trainable: {trainable:,} / Total: {total:,} ({trainable / total:.4%})")
