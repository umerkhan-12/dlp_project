"""
Dataset Module for Retinal Disease Classification
==================================================
Handles data loading, preprocessing, and dataset management.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict, Any, Callable
from pathlib import Path
import albumentations as A


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for retinal fundus images.

    Supports both APTOS 2019 and ODIR-5K datasets.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        preprocess_fn: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            image_paths: List of paths to images
            labels: List of integer labels
            transform: Albumentations transform pipeline
            preprocess_fn: Optional preprocessing function
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.

        Returns:
            Tensor of class weights
        """
        labels_array = np.array(self.labels)
        class_counts = np.bincount(labels_array)
        total = len(labels_array)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


class APTOSDataset:
    """
    Handler for APTOS 2019 Blindness Detection dataset.
    """

    def __init__(
        self,
        data_dir: str,
        csv_file: str = "train.csv",
        image_folder: str = "train_images",
    ):
        """
        Initialize APTOS dataset handler.

        Args:
            data_dir: Path to the dataset directory
            csv_file: Name of the CSV file with labels
            image_folder: Name of the folder containing images
        """
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / csv_file
        self.image_dir = self.data_dir / image_folder

        self.class_names = [
            "No_DR",
            "Mild_DR",
            "Moderate_DR",
            "Severe_DR",
            "Proliferative_DR"
        ]

        self.df = None
        if self.csv_path.exists():
            self.df = pd.read_csv(self.csv_path)

    def load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels.

        Returns:
            Tuple of (image_paths, labels)
        """
        if self.df is None:
            raise ValueError(f"CSV file not found: {self.csv_path}")

        image_paths = []
        labels = []

        for _, row in self.df.iterrows():
            image_id = row['id_code']
            label = row['diagnosis']

            # Try different image extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                image_path = self.image_dir / f"{image_id}{ext}"
                if image_path.exists():
                    image_paths.append(str(image_path))
                    labels.append(int(label))
                    break

        return image_paths, labels

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to counts
        """
        if self.df is None:
            return {}

        counts = self.df['diagnosis'].value_counts().sort_index()
        return {self.class_names[i]: int(counts.get(i, 0)) for i in range(len(self.class_names))}


class ODIRDataset:
    """
    Handler for ODIR-5K dataset (multi-label).
    """

    def __init__(
        self,
        data_dir: str,
        csv_file: str = "full_df.csv",
    ):
        """
        Initialize ODIR dataset handler.

        Args:
            data_dir: Path to the dataset directory
            csv_file: Name of the CSV file with labels
        """
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / csv_file

        self.class_names = [
            "Normal",
            "Diabetes",
            "Glaucoma",
            "Cataract",
            "AMD",
            "Hypertension",
            "Myopia",
            "Other"
        ]

        self.df = None
        if self.csv_path.exists():
            self.df = pd.read_csv(self.csv_path)

    def load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels (using primary diagnosis).

        Returns:
            Tuple of (image_paths, labels)
        """
        if self.df is None:
            raise ValueError(f"CSV file not found: {self.csv_path}")

        image_paths = []
        labels = []

        # Simplified loading - adapt based on actual CSV structure
        for _, row in self.df.iterrows():
            # Adjust column names based on actual CSV structure
            if 'Left-Fundus' in self.df.columns:
                image_path = self.data_dir / row['Left-Fundus']
                if image_path.exists():
                    image_paths.append(str(image_path))
                    # Get primary label
                    label = self._get_primary_label(row)
                    labels.append(label)

        return image_paths, labels

    def _get_primary_label(self, row: pd.Series) -> int:
        """Extract primary label from row."""
        # Simplified - adapt based on actual data structure
        return 0


def create_data_splits(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Create stratified train/val/test splits.

    Args:
        image_paths: List of image paths
        labels: List of labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

    # First split: train vs (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=temp_labels
    )

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels),
    }


def create_dataloaders(
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Optional[Tuple[List[str], List[int]]],
    train_transform: A.Compose,
    val_transform: A.Compose,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create DataLoader objects for training, validation, and testing.

    Args:
        train_data: Tuple of (paths, labels) for training
        val_data: Tuple of (paths, labels) for validation
        test_data: Optional tuple of (paths, labels) for testing
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Dictionary of DataLoaders
    """
    train_dataset = RetinalDataset(
        train_data[0], train_data[1],
        transform=train_transform
    )

    val_dataset = RetinalDataset(
        val_data[0], val_data[1],
        transform=val_transform
    )

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    if test_data is not None:
        test_dataset = RetinalDataset(
            test_data[0], test_data[1],
            transform=val_transform
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return dataloaders


def save_splits(
    splits: Dict[str, Tuple[List[str], List[int]]],
    save_dir: str,
) -> None:
    """
    Save data splits to CSV files for reproducibility.

    Args:
        splits: Dictionary of splits
        save_dir: Directory to save files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (paths, labels) in splits.items():
        df = pd.DataFrame({
            'image_path': paths,
            'label': labels
        })
        df.to_csv(save_dir / f"{split_name}.csv", index=False)


def load_splits(
    splits_dir: str,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Load data splits from CSV files.

    Args:
        splits_dir: Directory containing split files

    Returns:
        Dictionary of splits
    """
    splits_dir = Path(splits_dir)
    splits = {}

    for split_name in ['train', 'val', 'test']:
        csv_path = splits_dir / f"{split_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            splits[split_name] = (
                df['image_path'].tolist(),
                df['label'].tolist()
            )

    return splits


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset module...")

    # Test APTOS dataset handler
    aptos = APTOSDataset("data/raw/aptos2019-blindness-detection")
    print(f"APTOS class names: {aptos.class_names}")

    # Test split creation
    dummy_paths = [f"img_{i}.png" for i in range(100)]
    dummy_labels = [i % 5 for i in range(100)]

    splits = create_data_splits(dummy_paths, dummy_labels)
    print(f"Train size: {len(splits['train'][0])}")
    print(f"Val size: {len(splits['val'][0])}")
    print(f"Test size: {len(splits['test'][0])}")
