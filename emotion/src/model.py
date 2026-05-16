"""MobileNetV2 with a 7-class classification head for FER-2013."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from . import config as C


def build_mobilenetv2(
    num_classes: int = C.NUM_CLASSES,
    unfreeze_last_n: int = C.UNFREEZE_LAST_N_BLOCKS,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """Return a MobileNetV2 ready for fine-tuning on FER-2013.

    Strategy:
      - Load ImageNet pretrained weights (allowed: project rules permit
        transfer learning + partial fine-tuning).
      - Freeze all backbone parameters.
      - Unfreeze the last `unfreeze_last_n` inverted-residual blocks so the
        higher-level features adapt to facial expressions.
      - Replace the 1000-class classifier with a new 7-class head.
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # `features` is an nn.Sequential of 19 modules. Unfreeze the tail.
    n_blocks = len(model.features)
    for i in range(n_blocks - unfreeze_last_n, n_blocks):
        for p in model.features[i].parameters():
            p.requires_grad = True

    # Replace the classifier head. Original last layer is Linear(1280, 1000).
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    # New head must train.
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def parameter_groups(model: nn.Module, lr_head: float, lr_backbone: float):
    """Two-LR optimiser groups: head trains faster than the unfrozen tail."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": head_params, "lr": lr_head},
        {"params": backbone_params, "lr": lr_backbone},
    ]


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts - useful for the report."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_checkpoint(model: nn.Module, path, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def load_checkpoint(path, map_location="cpu") -> nn.Module:
    """Rebuild architecture and load weights. Mirrors build_mobilenetv2 defaults."""
    model = build_mobilenetv2(pretrained=False)
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()
    return model
