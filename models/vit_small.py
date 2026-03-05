"""
Small Vision Transformer (DeiT-Tiny) for 4-class chest cancer CT classification (3 NSCLC + normal).
MacBook-friendly: ~5.7M params, runs on CPU or MPS.
"""
import torch.nn as nn
import timm


def build_deit_tiny(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build DeiT-Tiny with a 4-class head. Uses 224x224 input; same ImageNet norm as CNNs.
    """
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=0,  # remove head; we add our own
    )
    feat_dim = model.num_features  # 192 for deit_tiny
    model.head = nn.Linear(feat_dim, num_classes)
    return model


def build_vit_small(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Slightly larger ViT-Small (~22M params). Use if you have more RAM/time. 4-class head.
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=0,
    )
    feat_dim = model.num_features  # 384
    model.head = nn.Linear(feat_dim, num_classes)
    return model
