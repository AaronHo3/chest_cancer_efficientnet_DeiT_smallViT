"""
EfficientNet-B2 for 4-class chest cancer CT classification (3 NSCLC types + normal).
"""
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


def build_efficientnet_b2(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B2 with a 4-class head. Input 224x224, ImageNet normalization.
    """
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features  # 1408 for B2
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model
