"""
Dataset Download Script
=======================
Download datasets from Kaggle for retinal disease classification.
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def download_aptos(data_dir: str = "data/raw") -> None:
    """
    Download APTOS 2019 Blindness Detection dataset from Kaggle.

    Args:
        data_dir: Directory to save the dataset
    """
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not installed. Installing...")
        os.system("pip install kaggle")
        import kaggle

    data_dir = Path(data_dir)
    aptos_dir = data_dir / "aptos2019-blindness-detection"

    if aptos_dir.exists():
        print(f"Dataset already exists at {aptos_dir}")
        return

    print("Downloading APTOS 2019 Blindness Detection dataset...")
    print("Note: Make sure you have accepted the competition rules on Kaggle!")
    print("Visit: https://www.kaggle.com/competitions/aptos2019-blindness-detection")

    try:
        # Create directory
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download using Kaggle API
        os.system(f'kaggle competitions download -c aptos2019-blindness-detection -p "{data_dir}"')

        # Extract zip file
        zip_path = data_dir / "aptos2019-blindness-detection.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(aptos_dir)
            zip_path.unlink()  # Remove zip file
            print(f"Dataset extracted to {aptos_dir}")
        else:
            print("Download may have failed. Please check your Kaggle credentials.")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://www.kaggle.com/competitions/aptos2019-blindness-detection")
        print("2. Accept the competition rules")
        print("3. Download the dataset manually")
        print(f"4. Extract to {aptos_dir}")


def download_odir(data_dir: str = "data/raw") -> None:
    """
    Download ODIR-5K dataset from Kaggle.

    Args:
        data_dir: Directory to save the dataset
    """
    try:
        import kaggle
    except ImportError:
        print("Kaggle package not installed. Installing...")
        os.system("pip install kaggle")
        import kaggle

    data_dir = Path(data_dir)
    odir_dir = data_dir / "odir-5k"

    if odir_dir.exists():
        print(f"Dataset already exists at {odir_dir}")
        return

    print("Downloading ODIR-5K dataset...")

    try:
        # Create directory
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download using Kaggle API
        os.system(f'kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k -p "{data_dir}"')

        # Extract zip file
        zip_path = data_dir / "ocular-disease-recognition-odir5k.zip"
        if zip_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(odir_dir)
            zip_path.unlink()  # Remove zip file
            print(f"Dataset extracted to {odir_dir}")
        else:
            print("Download may have failed. Please check your Kaggle credentials.")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k")
        print("2. Download the dataset manually")
        print(f"3. Extract to {odir_dir}")


def verify_dataset(data_dir: str, dataset_name: str = "aptos") -> bool:
    """
    Verify that the dataset was downloaded correctly.

    Args:
        data_dir: Directory containing the dataset
        dataset_name: Name of the dataset ('aptos' or 'odir')

    Returns:
        True if dataset is valid, False otherwise
    """
    data_dir = Path(data_dir)

    if dataset_name == "aptos":
        required_files = [
            "train.csv",
            "train_images",
        ]
        dataset_dir = data_dir / "aptos2019-blindness-detection"
    else:
        required_files = [
            "full_df.csv",
        ]
        dataset_dir = data_dir / "odir-5k"

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return False

    missing_files = []
    for f in required_files:
        if not (dataset_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"Missing files in {dataset_dir}:")
        for f in missing_files:
            print(f"  - {f}")
        return False

    print(f"Dataset verified successfully: {dataset_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download datasets for retinal disease classification')
    parser.add_argument('--dataset', type=str, default='aptos',
                       choices=['aptos', 'odir', 'all'],
                       help='Dataset to download')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory to save datasets')

    args = parser.parse_args()

    print("=" * 60)
    print("DATASET DOWNLOAD SCRIPT")
    print("=" * 60)

    if args.dataset in ['aptos', 'all']:
        download_aptos(args.data_dir)
        verify_dataset(args.data_dir, 'aptos')
        print()

    if args.dataset in ['odir', 'all']:
        download_odir(args.data_dir)
        verify_dataset(args.data_dir, 'odir')
        print()

    print("Done!")


if __name__ == '__main__':
    main()
