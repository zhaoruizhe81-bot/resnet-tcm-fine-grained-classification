from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ResNet101_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet101,
    resnet18,
    resnet34,
    resnet50,
)


MODEL_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
}


def build_model(name: str, num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    if name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"不支持的模型 {name}，可选值：{supported}")

    builder, weights = MODEL_REGISTRY[name]
    model = builder(weights=weights if pretrained else None)
    in_features = model.fc.in_features

    head_layers: list[nn.Module] = []
    if dropout > 0:
        head_layers.append(nn.Dropout(p=dropout))
    head_layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*head_layers)
    return model
