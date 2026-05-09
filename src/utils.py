"""
Utility Functions for Retinal Disease Classification
=====================================================
Common utilities for training and evaluation.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "cuda") -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device: Preferred device ("cuda", "cpu", "mps")

    Returns:
        torch.device object
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


class EarlyStopping:
    """
    Early stopping handler for training.
    """

    def __init__(
        self,
        patience: int = 15,
        mode: str = 'max',
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            epoch: Current epoch

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """
    Model checkpoint handler.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize checkpoint handler.

        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            verbose: Print messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_score = None
        self.best_path = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        score: float,
        metrics: Dict[str, Any],
    ) -> Optional[str]:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            score: Current metric value
            metrics: All current metrics

        Returns:
            Path to saved checkpoint or None
        """
        should_save = True

        if self.save_best_only:
            if self.best_score is None:
                should_save = True
            elif self.mode == 'max':
                should_save = score > self.best_score
            else:
                should_save = score < self.best_score

        if not should_save:
            return None

        self.best_score = score

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'score': score,
            'metrics': metrics,
        }

        # Save checkpoint
        filename = f"best_model.pth" if self.save_best_only else f"checkpoint_epoch_{epoch}.pth"
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

        if self.best_path and self.best_path != save_path and self.save_best_only:
            # Remove old best model
            if self.best_path.exists():
                self.best_path.unlink()

        self.best_path = save_path

        if self.verbose:
            print(f"Checkpoint saved: {save_path} (score: {score:.4f})")

        return str(save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_experiment_dir(
    base_dir: str,
    model_name: str,
    experiment_name: Optional[str] = None,
) -> Path:
    """
    Create experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        model_name: Name of the model
        experiment_name: Optional experiment name

    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name:
        dir_name = f"{model_name}_{experiment_name}_{timestamp}"
    else:
        dir_name = f"{model_name}_{timestamp}"

    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)

    return exp_dir


def save_training_config(
    config: Any,
    save_path: str,
) -> None:
    """
    Save training configuration to JSON.

    Args:
        config: Configuration object or dictionary
        save_path: Path to save file
    """
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)

    # Convert to serializable format
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return str(obj)

    serializable_config = make_serializable(config_dict)

    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> None:
    """
    Print model summary.

    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    params = count_parameters(model)
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters:     {params['total']:,}")
    print(f"Trainable Parameters: {params['trainable']:,}")
    print(f"Non-trainable Parameters: {params['non_trainable']:,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test seed
    set_seed(42)
    print(f"Random seed set to 42")

    # Test device
    device = get_device("cuda")
    print(f"Device: {device}")

    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")

    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    for epoch, score in enumerate([0.5, 0.6, 0.55, 0.54, 0.53]):
        if early_stopping(score, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    # Test time formatting
    print(f"Time format: {format_time(3725)}")
