"""
Data Augmentation Module for Retinal Disease Classification
============================================================
Implements augmentation pipelines using Albumentations library.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional


def get_train_transforms(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    rotation_limit: int = 15,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.1,
    hue: float = 0.05,
    scale_range: Tuple[float, float] = (0.85, 1.0),
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        image_size: Target image size
        mean: Normalization mean values
        std: Normalization std values
        rotation_limit: Max rotation angle in degrees
        brightness: Brightness adjustment range
        contrast: Contrast adjustment range
        saturation: Saturation adjustment range
        hue: Hue adjustment range
        scale_range: Scale range for random resized crop

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        # Resize and crop
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=scale_range,
            ratio=(0.9, 1.1),
            p=1.0
        ),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=rotation_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),

        # Color augmentations
        A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=0.5
        ),

        # Additional augmentations for medical imaging
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        # CLAHE for contrast enhancement (common in medical imaging)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Normalize and convert to tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    """
    Get validation/test augmentation pipeline (no augmentation, just preprocessing).

    Args:
        image_size: Target image size
        mean: Normalization mean values
        std: Normalization std values

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_tta_transforms(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> List[A.Compose]:
    """
    Get Test Time Augmentation (TTA) transforms.
    Returns multiple transform pipelines for TTA.

    Args:
        image_size: Target image size
        mean: Normalization mean values
        std: Normalization std values

    Returns:
        List of transform pipelines for TTA
    """
    base_transforms = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    tta_transforms = [
        # Original
        A.Compose(base_transforms),

        # Horizontal flip
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),

        # Vertical flip
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),

        # Rotate 90
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),

        # Rotate -90
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=(-90, -90), p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]),
    ]

    return tta_transforms


class Mixup:
    """Mixup augmentation for training."""

    def __init__(self, alpha: float = 0.2):
        """
        Initialize Mixup.

        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha

    def __call__(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Apply Mixup to a batch.

        Args:
            images: Batch of images (N, C, H, W)
            labels: Batch of labels (N,)

        Returns:
            Mixed images, original labels, shuffled labels, lambda value
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.shape[0]
        index = np.random.permutation(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """CutMix augmentation for training."""

    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.

        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha

    def _rand_bbox(
        self,
        size: Tuple[int, ...],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Apply CutMix to a batch.

        Args:
            images: Batch of images (N, C, H, W)
            labels: Batch of labels (N,)

        Returns:
            CutMix images, original labels, shuffled labels, adjusted lambda
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.shape[0]
        index = np.random.permutation(batch_size)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape, lam)

        images_cutmix = images.copy()
        images_cutmix[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[2] * images.shape[3]))

        labels_a, labels_b = labels, labels[index]

        return images_cutmix, labels_a, labels_b, lam


def remove_black_borders(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove black borders from fundus images.

    Args:
        image: Input image (H, W, C) or (H, W)
        threshold: Pixel value threshold for detecting black regions

    Returns:
        Cropped image with black borders removed
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Find non-black region
    mask = gray > threshold
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    if len(image.shape) == 3:
        return image[y0:y1, x0:x1, :]
    else:
        return image[y0:y1, x0:x1]


def apply_ben_graham_preprocessing(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Apply Ben Graham's preprocessing technique for fundus images.
    This technique enhances local contrast and removes lighting variations.

    Args:
        image: Input image (H, W, C)
        sigmaX: Gaussian blur sigma

    Returns:
        Preprocessed image
    """
    image = cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (0, 0), sigmaX), -4,
        128
    )
    return image


def preprocess_fundus_image(
    image: np.ndarray,
    target_size: int = 224,
    remove_borders: bool = True,
    apply_ben_graham: bool = False,
) -> np.ndarray:
    """
    Complete preprocessing pipeline for fundus images.

    Args:
        image: Input image
        target_size: Target size for output
        remove_borders: Whether to remove black borders
        apply_ben_graham: Whether to apply Ben Graham preprocessing

    Returns:
        Preprocessed image
    """
    if remove_borders:
        image = remove_black_borders(image)

    if apply_ben_graham:
        image = apply_ben_graham_preprocessing(image)

    # Resize to target size
    image = cv2.resize(image, (target_size, target_size))

    return image


if __name__ == "__main__":
    # Test augmentation pipeline
    import torch

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    # Apply transforms
    train_result = train_transform(image=dummy_image)
    val_result = val_transform(image=dummy_image)

    print(f"Train output shape: {train_result['image'].shape}")
    print(f"Val output shape: {val_result['image'].shape}")

    # Test Mixup
    batch_images = np.random.randn(8, 3, 224, 224).astype(np.float32)
    batch_labels = np.array([0, 1, 2, 3, 4, 0, 1, 2])

    mixup = Mixup(alpha=0.2)
    mixed_images, labels_a, labels_b, lam = mixup(batch_images, batch_labels)
    print(f"Mixup lambda: {lam:.4f}")
