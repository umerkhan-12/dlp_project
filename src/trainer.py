"""
Training Module for Retinal Disease Classification
===================================================
Complete training pipeline with mixed precision and gradient accumulation.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, LambdaLR
from typing import Dict, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .utils import (
    AverageMeter, EarlyStopping, ModelCheckpoint,
    get_lr, format_time, set_seed, get_device
)
from .metrics import compute_metrics
from .augmentations import Mixup


class Trainer:
    """
    Complete training pipeline for retinal disease classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device('cuda'),
        num_classes: int = 5,
        class_names: Optional[list] = None,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        mixup_alpha: float = 0.0,
        experiment_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_classes: Number of classes
            class_names: List of class names
            mixed_precision: Use mixed precision training
            gradient_clip: Gradient clipping value
            mixup_alpha: Mixup alpha value (0 to disable)
            experiment_dir: Directory to save results
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip

        # Mixup
        self.mixup = Mixup(alpha=mixup_alpha) if mixup_alpha > 0 else None

        # Experiment directory
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        if self.experiment_dir:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision and device.type == 'cuda' else None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_balanced_acc': [],
            'val_f1': [],
            'val_kappa': [],
            'lr': [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            batch_size = images.size(0)

            # Apply Mixup
            use_mixup = self.mixup is not None and np.random.random() > 0.5
            if use_mixup:
                images_np = images.cpu().numpy()
                labels_np = labels.cpu().numpy()
                images_mixed, labels_a, labels_b, lam = self.mixup(images_np, labels_np)
                images = torch.from_numpy(images_mixed).to(self.device)
                labels_a = torch.from_numpy(labels_a).to(self.device)
                labels_b = torch.from_numpy(labels_b).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.mixed_precision and self.scaler:
                with autocast():
                    outputs = self.model(images)
                    if use_mixup:
                        loss = lam * self.criterion(outputs, labels_a) + \
                               (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        loss = self.criterion(outputs, labels)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mixup:
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

                loss.backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.optimizer.step()

            # Compute accuracy (without mixup labels)
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                if not use_mixup:
                    correct = (preds == labels).sum().item()
                    acc = correct / batch_size
                else:
                    acc = lam * (preds == labels_a).float().mean().item() + \
                          (1 - lam) * (preds == labels_b).float().mean().item()

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
            })

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter()
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc='Validation', leave=False)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            if self.mixed_precision and self.device.type == 'cuda':
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            loss_meter.update(loss.item(), batch_size)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, all_probs, self.class_names)
        metrics['loss'] = loss_meter.avg

        return metrics

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Complete training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            verbose: Print training progress

        Returns:
            Training history
        """
        # Initialize components
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max',
            verbose=verbose,
        )

        checkpoint_dir = checkpoint_dir or (
            str(self.experiment_dir / 'checkpoints') if self.experiment_dir else 'checkpoints'
        )
        checkpoint = ModelCheckpoint(
            save_dir=checkpoint_dir,
            monitor='val_balanced_accuracy',
            mode='max',
            save_best_only=True,
            verbose=verbose,
        )

        best_metrics = None
        start_time = time.time()

        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Total Epochs: {num_epochs}")
        print("=" * 60 + "\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step()
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = get_lr(self.optimizer)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['val_kappa'].append(val_metrics['quadratic_kappa'])
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print progress
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} [{format_time(epoch_time)}]")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
                print(f"  Balanced Acc: {val_metrics['balanced_accuracy']:.4f} | "
                      f"F1: {val_metrics['f1_macro']:.4f} | Kappa: {val_metrics['quadratic_kappa']:.4f}")
                print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            checkpoint.save(
                self.model, self.optimizer, self.scheduler,
                epoch, val_metrics['balanced_accuracy'], val_metrics
            )

            # Update best metrics
            if best_metrics is None or val_metrics['balanced_accuracy'] > best_metrics['balanced_accuracy']:
                best_metrics = val_metrics.copy()

            # Early stopping
            if early_stopping(val_metrics['balanced_accuracy'], epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            print()

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total Time: {format_time(total_time)}")
        print(f"Best Epoch: {early_stopping.best_epoch + 1}")
        print(f"Best Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
        print(f"Best F1 Score (Macro): {best_metrics['f1_macro']:.4f}")
        print(f"Best Quadratic Kappa: {best_metrics['quadratic_kappa']:.4f}")
        print("=" * 60 + "\n")

        return self.history


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional layer-wise learning rate decay.

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adamw', 'sgd')
        learning_rate: Base learning rate
        weight_decay: Weight decay value
        layer_decay: Layer-wise decay factor (0 to disable)

    Returns:
        Optimizer instance
    """
    if layer_decay > 0 and hasattr(model, 'backbone'):
        # Layer-wise LR decay for transformer-based models
        param_groups = []

        # Classifier head (full learning rate)
        classifier_params = list(model.classifier.parameters())
        param_groups.append({
            'params': classifier_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
        })

        # Backbone (decayed learning rate)
        backbone_params = list(model.backbone.parameters())
        param_groups.append({
            'params': backbone_params,
            'lr': learning_rate * layer_decay,
            'weight_decay': weight_decay,
        })
    else:
        param_groups = [
            {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}
        ]

    if optimizer_name.lower() == 'adamw':
        return AdamW(param_groups)
    elif optimizer_name.lower() == 'sgd':
        return SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    steps_per_epoch: int = 100,
) -> Optional[Any]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        steps_per_epoch: Steps per epoch

    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == 'cosine':
        # Cosine annealing with warm restarts
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs,
            T_mult=1,
            eta_min=1e-7,
        )
    elif scheduler_name.lower() == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / num_epochs,
        )
    elif scheduler_name.lower() == 'warmup_cosine':
        # Linear warmup then cosine decay
        def lr_lambda(current_step):
            if current_step < warmup_epochs:
                return float(current_step) / float(max(1, warmup_epochs))
            progress = float(current_step - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)
    else:
        return None


if __name__ == "__main__":
    print("Training module loaded successfully!")
