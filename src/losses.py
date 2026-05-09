"""
Loss Functions for Retinal Disease Classification
==================================================
Implements Focal Loss and other loss functions for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Focal Loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights tensor of shape (num_classes,)
            reduction: Reduction method ('mean', 'sum', 'none')
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Computed loss
        """
        num_classes = inputs.shape[1]

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Compute focal weight
        focal_weight = (1 - p) ** self.gamma

        # Compute cross entropy
        ce = -targets_smooth * torch.log(p.clamp(min=1e-8))

        # Apply focal weight
        focal_loss = focal_weight * ce

        # Apply class weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha.unsqueeze(0).expand_as(focal_loss)
            focal_loss = alpha_weight * focal_loss

        # Sum over classes
        focal_loss = focal_loss.sum(dim=1)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """
        Initialize Label Smoothing Cross Entropy.

        Args:
            smoothing: Label smoothing factor
            weight: Class weights
            reduction: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.

        Args:
            inputs: Model predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Computed loss
        """
        num_classes = inputs.shape[1]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + \
                        self.smoothing / num_classes

        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)

        # Compute loss
        loss = -targets_smooth * log_probs

        # Apply class weights
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            weight = self.weight.unsqueeze(0).expand_as(loss)
            loss = weight * loss

        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MixupCriterion:
    """
    Criterion wrapper for Mixup training.
    """

    def __init__(self, criterion: nn.Module):
        """
        Initialize Mixup criterion.

        Args:
            criterion: Base loss function
        """
        self.criterion = criterion

    def __call__(
        self,
        inputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute mixed loss.

        Args:
            inputs: Model predictions
            targets_a: Original labels
            targets_b: Shuffled labels
            lam: Mixing coefficient

        Returns:
            Mixed loss
        """
        return lam * self.criterion(inputs, targets_a) + \
               (1 - lam) * self.criterion(inputs, targets_b)


def create_loss_function(
    loss_name: str = "focal",
    num_classes: int = 5,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """
    Create a loss function.

    Args:
        loss_name: Name of the loss function ('focal', 'cross_entropy', 'label_smoothing')
        num_classes: Number of classes
        class_weights: Optional class weights
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor

    Returns:
        Loss function module
    """
    if loss_name.lower() == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )
    elif loss_name.lower() == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights,
        )
    elif loss_name.lower() == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 8
    num_classes = 5
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")

    # Test with class weights
    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5])
    focal_loss_weighted = FocalLoss(gamma=2.0, alpha=class_weights)
    loss_weighted = focal_loss_weighted(inputs, targets)
    print(f"Focal Loss (weighted): {loss_weighted.item():.4f}")

    # Test Label Smoothing CE
    lsce = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_smooth = lsce(inputs, targets)
    print(f"Label Smoothing CE: {loss_smooth.item():.4f}")

    # Test factory function
    loss_fn = create_loss_function("focal", class_weights=class_weights)
    loss_factory = loss_fn(inputs, targets)
    print(f"Factory Focal Loss: {loss_factory.item():.4f}")
