# Models

Two architectures are compared: ResNet-50 (CNN) and ViT-B/16 (Vision Transformer).

---

## Why these two?

**ResNet-50** is a classic CNN. Convolutions give it a spatial inductive bias out of the box,
and its ImageNet pretraining is strong. It is computationally cheap, trains quickly, and serves
as our CNN baseline.

**ViT-B/16** takes a different approach: the image is split into 16×16 non-overlapping patches
and processed as a sequence by transformer encoder blocks. It has no convolutional inductive
bias — spatial relationships are learned from data via self-attention. Comparing it against
ResNet-50 tells us whether global context is useful for distinguishing AI-generated texture
patterns, which are often globally consistent but locally indistinguishable from real photos.

---

## Training strategy: linear probe

Both models are loaded with ImageNet-pretrained weights. Their backbones are **fully frozen**
(`freeze_backbone: true` in the config), and only the final classification head
(a single `nn.Linear` layer) is trained. This is called a *linear probe*.

Why this approach:

- **Small dataset:** our working set is 5,000 images. Training all ~25 M parameters would
  overfit quickly.
- **Efficiency:** training just the head converges in ≤ 20 epochs (minutes on a single GPU).
- **Representation quality:** a linear probe directly measures how well the frozen pretrained
  backbone separates the two classes in feature space.

Linear-probe results (5k sample, see main README for metrics):

| Model      | Test Accuracy | Test ROC-AUC |
|------------|--------------|--------------|
| ResNet-50  | 90.20 %      | 0.9662       |
| ViT-B/16   | 85.90 %      | 0.9294       |

ResNet-50 edges out ViT-B/16 on this small sample, likely because its spatial inductive bias
helps when data is limited.

---

## Selective unfreezing (future work)

The configs expose `unfreeze_last_n_blocks` for partial backbone fine-tuning:

- **ResNet-50:** `1` = unfreeze `layer4`; `2` = `layer3 + layer4`; up to `4` for the full backbone.
- **ViT-B/16:** `N` = unfreeze the last N transformer encoder blocks.

This would require more data or stronger regularization to avoid overfitting and has not been
evaluated yet.

---

## Files

| File | Contents |
|---|---|
| `resnet.py` | `build_resnet50()` — loads pretrained weights, swaps FC head, optional freeze |
| `vit.py` | `build_vit_b16()` — loads pretrained weights, swaps classifier head, optional freeze |
| `model_factory.py` | `build_model(name, ...)` dispatcher — accepts `"resnet50"` or `"vit_b_16"` |

To add a new architecture, register it in `model_factory.py` and create a corresponding builder
module following the same pattern.
