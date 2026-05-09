"""
Model Architectures for Retinal Disease Classification
=======================================================
Implements ViT, ResNet-50, and EfficientNet-B4 models.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, List, Dict, Any


class RetinalClassifier(nn.Module):
    """
    Base class for retinal disease classification models.
    Wraps pretrained models with a custom classification head.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the backbone model (timm model name)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for classification head
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Features of shape (B, feature_dim)
        """
        return self.backbone(x)


class ViTClassifier(RetinalClassifier):
    """
    Vision Transformer (ViT) based classifier.
    """

    def __init__(
        self,
        variant: str = "base",
        patch_size: int = 16,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize ViT classifier.

        Args:
            variant: ViT variant ('tiny', 'small', 'base', 'large')
            patch_size: Patch size (8, 16, or 32)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        model_name = f"vit_{variant}_patch{patch_size}_224"
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )


class ResNetClassifier(RetinalClassifier):
    """
    ResNet-based classifier.
    """

    def __init__(
        self,
        variant: int = 50,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize ResNet classifier.

        Args:
            variant: ResNet variant (18, 34, 50, 101, 152)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        model_name = f"resnet{variant}"
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )


class EfficientNetClassifier(RetinalClassifier):
    """
    EfficientNet-based classifier.
    """

    def __init__(
        self,
        variant: str = "b4",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize EfficientNet classifier.

        Args:
            variant: EfficientNet variant ('b0' through 'b7')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        model_name = f"efficientnet_{variant}"
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )


def create_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_name: Model name or shorthand
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate

    Returns:
        Model instance
    """
    # Handle shortcuts
    model_mapping = {
        "vit": "vit_base_patch16_224",
        "vit-b16": "vit_base_patch16_224",
        "vit-b32": "vit_base_patch32_224",
        "vit-l16": "vit_large_patch16_224",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "efficientnet-b4": "efficientnet_b4",
        "efficientnet-b0": "efficientnet_b0",
        "efficientnet-b7": "efficientnet_b7",
    }

    if model_name.lower() in model_mapping:
        model_name = model_mapping[model_name.lower()]

    return RetinalClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


def get_layer_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """
    Get parameter groups for layer-wise learning rate decay.

    Args:
        model: The model

    Returns:
        List of parameter groups
    """
    if hasattr(model, 'backbone'):
        backbone_params = list(model.backbone.parameters())
        classifier_params = list(model.classifier.parameters())
        return [backbone_params, classifier_params]
    return [list(model.parameters())]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: The model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
    }


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters."""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters."""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")

    # Test ViT
    vit = create_model("vit_base_patch16_224", num_classes=5)
    params = count_parameters(vit)
    print(f"ViT-B/16 parameters: {params['total']:,}")

    # Test ResNet
    resnet = create_model("resnet50", num_classes=5)
    params = count_parameters(resnet)
    print(f"ResNet-50 parameters: {params['total']:,}")

    # Test EfficientNet
    effnet = create_model("efficientnet_b4", num_classes=5)
    params = count_parameters(effnet)
    print(f"EfficientNet-B4 parameters: {params['total']:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = vit(dummy_input)
    print(f"Output shape: {output.shape}")
