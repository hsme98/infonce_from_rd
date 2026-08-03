from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    """ResNet-18 with the standard CIFAR stem."""

    output_dim: int = 512

    def __init__(self) -> None:
        super().__init__()
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Identity()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, projection_hidden_dim: int = 512, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = CIFARResNet18()
        self.projector = ProjectionHead(
            CIFARResNet18.output_dim, projection_hidden_dim, projection_dim
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        return {"h": h, "z": z}


class SupervisedModel(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.encoder = CIFARResNet18()
        self.classifier = nn.Linear(CIFARResNet18.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        logits = self.classifier(h)
        return {"h": h, "logits": logits}


@dataclass(frozen=True)
class ModelSpec:
    projection_hidden_dim: int = 512
    projection_dim: int = 128


def build_model(role: str, spec: ModelSpec) -> nn.Module:
    if role == "simclr":
        return SimCLRModel(
            projection_hidden_dim=spec.projection_hidden_dim,
            projection_dim=spec.projection_dim,
        )
    if role == "supervised":
        return SupervisedModel(num_classes=100)
    raise ValueError(f"unknown model role: {role}")
