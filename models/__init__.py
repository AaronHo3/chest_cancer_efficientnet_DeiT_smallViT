"""
Model factory and device selection for chest cancer CT classification.
Supports: efficientnet_b2, deit_tiny (small ViT), vit_small.
"""
import torch

from .efficient_net import build_efficientnet_b2
from .vit_small import build_deit_tiny, build_vit_small


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Return best available device: cuda > mps (Apple Silicon) > cpu.
    On MacBook, MPS often gives a nice speedup; set prefer_mps=False to force CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
) -> torch.nn.Module:
    """
    name: "efficientnet_b2" | "deit_tiny" | "vit_small"
    num_classes: 4 (adenocarcinoma, large_cell, squamous, normal).
    """
    name = name.strip().lower()
    if name == "efficientnet_b2":
        return build_efficientnet_b2(num_classes=num_classes, pretrained=pretrained)
    if name == "deit_tiny":
        return build_deit_tiny(num_classes=num_classes, pretrained=pretrained)
    if name == "vit_small":
        return build_vit_small(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}. Use one of: efficientnet_b2, deit_tiny, vit_small")
