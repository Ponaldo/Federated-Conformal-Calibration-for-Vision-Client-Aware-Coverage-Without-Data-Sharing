from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tvm


ModelName = Literal["resnet50", "vit-b16"]


def build_model(name: ModelName, num_classes: int, pretrained: bool = False) -> nn.Module:
    lname = name.lower()
    if lname == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if lname in ("vit-b16", "vit_b16", "vitb16"):
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {name}")


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
