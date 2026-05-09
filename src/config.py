"""
Configuration Module for Retinal Disease Classification
========================================================
Handles loading and validation of configuration settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    image_size: int = 224
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 5
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0


@dataclass
class LossConfig:
    name: str = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    class_weighted: bool = True


@dataclass
class AugmentationConfig:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    hue: float = 0.05
    rotation_limit: int = 15
    scale_range: Tuple[float, float] = (0.85, 1.0)
    horizontal_flip: bool = True
    mixup_alpha: float = 0.2


@dataclass
class NormalizeConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class HardwareConfig:
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class Config:
    """Main configuration class containing all settings."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    classes: List[str] = field(default_factory=lambda: [
        "No_DR", "Mild_DR", "Moderate_DR", "Severe_DR", "Proliferative_DR"
    ])
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Load configuration from YAML file."""
    config = Config()

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            # Update data config
            if 'data' in yaml_config:
                for key, value in yaml_config['data'].items():
                    if hasattr(config.data, key):
                        setattr(config.data, key, value)

            # Update model config
            if 'model' in yaml_config:
                for key, value in yaml_config['model'].items():
                    if hasattr(config.model, key):
                        setattr(config.model, key, value)

            # Update training config
            if 'training' in yaml_config:
                for key, value in yaml_config['training'].items():
                    if hasattr(config.training, key):
                        setattr(config.training, key, value)

            # Update loss config
            if 'loss' in yaml_config:
                for key, value in yaml_config['loss'].items():
                    if hasattr(config.loss, key):
                        setattr(config.loss, key, value)

            # Update augmentation config
            if 'augmentation' in yaml_config:
                aug = yaml_config['augmentation']
                if 'color_jitter' in aug:
                    config.augmentation.brightness = aug['color_jitter'].get('brightness', 0.2)
                    config.augmentation.contrast = aug['color_jitter'].get('contrast', 0.2)
                    config.augmentation.saturation = aug['color_jitter'].get('saturation', 0.1)
                    config.augmentation.hue = aug['color_jitter'].get('hue', 0.05)
                if 'rotation_limit' in aug:
                    config.augmentation.rotation_limit = aug['rotation_limit']
                if 'scale_range' in aug:
                    config.augmentation.scale_range = tuple(aug['scale_range'])
                if 'horizontal_flip' in aug:
                    config.augmentation.horizontal_flip = aug['horizontal_flip']
                if 'mixup_alpha' in aug:
                    config.augmentation.mixup_alpha = aug['mixup_alpha']

            # Update normalize config
            if 'normalize' in yaml_config:
                config.normalize.mean = yaml_config['normalize'].get('mean', config.normalize.mean)
                config.normalize.std = yaml_config['normalize'].get('std', config.normalize.std)

            # Update hardware config
            if 'hardware' in yaml_config:
                for key, value in yaml_config['hardware'].items():
                    if hasattr(config.hardware, key):
                        setattr(config.hardware, key, value)

            # Update classes
            if 'classes' in yaml_config:
                config.classes = yaml_config['classes']

            # Update directories
            if 'checkpoint' in yaml_config:
                config.checkpoint_dir = yaml_config['checkpoint'].get('save_dir', 'checkpoints')
            if 'logging' in yaml_config:
                config.log_dir = yaml_config['logging'].get('log_dir', 'logs')
            if 'evaluation' in yaml_config:
                config.results_dir = yaml_config['evaluation'].get('results_dir', 'results')

    return config


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = {
        'data': {
            'raw_dir': config.data.raw_dir,
            'processed_dir': config.data.processed_dir,
            'splits_dir': config.data.splits_dir,
            'image_size': config.data.image_size,
            'train_ratio': config.data.train_ratio,
            'val_ratio': config.data.val_ratio,
            'test_ratio': config.data.test_ratio,
            'num_workers': config.data.num_workers,
        },
        'model': {
            'name': config.model.name,
            'pretrained': config.model.pretrained,
            'num_classes': config.model.num_classes,
            'dropout': config.model.dropout,
        },
        'training': {
            'batch_size': config.training.batch_size,
            'max_epochs': config.training.max_epochs,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'optimizer': config.training.optimizer,
            'scheduler': config.training.scheduler,
            'warmup_epochs': config.training.warmup_epochs,
            'early_stopping_patience': config.training.early_stopping_patience,
            'gradient_clip': config.training.gradient_clip,
        },
        'loss': {
            'name': config.loss.name,
            'focal_gamma': config.loss.focal_gamma,
            'label_smoothing': config.loss.label_smoothing,
            'class_weighted': config.loss.class_weighted,
        },
        'classes': config.classes,
        'hardware': {
            'device': config.hardware.device,
            'mixed_precision': config.hardware.mixed_precision,
            'seed': config.hardware.seed,
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config("configs/config.yaml")
    print(f"Model: {config.model.name}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Classes: {config.classes}")
