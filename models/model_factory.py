"""
Model factory for AIGI-Detection.

Maps a ``model_name`` string (matching the YAML config) to the
corresponding builder function.  Add new architectures here.
"""

from __future__ import annotations

import torch.nn as nn

from models.resnet import build_resnet50
from models.vit    import build_vit_b16


# Map config model_name -> builder callable
_REGISTRY: dict[str, callable] = {
    "resnet50":  build_resnet50,
    "vit_b_16":  build_vit_b16,
}


def build_model(
    model_name: str,
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    num_classes: int = 2,
) -> nn.Module:
    """Instantiate and return a model by name.

    Args:
        model_name:             Key matching a YAML config ``model_name`` field.
                                Supported: ``"resnet50"``, ``"vit_b_16"``.
        freeze_backbone:        Freeze backbone for linear-probe training.
        unfreeze_last_n_blocks: Number of backbone blocks to unfreeze.
        num_classes:            Output dimensionality (default 2).

    Returns:
        Configured ``nn.Module``.

    Raises:
        ValueError: When ``model_name`` is not in the registry.
    """
    if model_name in _REGISTRY:
        return _REGISTRY[model_name](
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            num_classes=num_classes,
        )

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Registered models: {list(_REGISTRY.keys())}"
    )
