"""Model construction and loading"""

from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18

def build_model() -> nn.Module:
    """Build the ResNet18 architecture used for gate-leakage detection."""

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def load_model(
        weights_path: str | Path,
        device: str = "cpu",
) -> nn.Module:
    """Load the trained gate-leakage model."""

    device = torch.device(device)
    model = build_model()

    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model