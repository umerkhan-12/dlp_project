# Retinal Disease Classification using Vision Transformers

A comprehensive deep learning pipeline for automated retinal disease classification using fine-tuned Vision Transformers (ViT).

## Overview

This project implements state-of-the-art Vision Transformer architectures for classifying diabetic retinopathy severity from retinal fundus images. The pipeline includes:

- **Vision Transformer (ViT-B/16)** - Main model for classification
- **ResNet-50** - CNN baseline for comparison
- **EfficientNet-B4** - Efficient CNN baseline

## Features

- Focal Loss for handling class imbalance
- Mixup and CutMix augmentation
- Mixed precision training (FP16)
- Layer-wise learning rate decay
- Cosine annealing with warm restarts
- Comprehensive evaluation metrics
- TensorBoard logging

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### APTOS 2019 Blindness Detection

1. **Accept competition rules** at: https://www.kaggle.com/competitions/aptos2019-blindness-detection

2. **Configure Kaggle API**:
   - Get your API token from https://www.kaggle.com/settings
   - Place `kaggle.json` in `~/.kaggle/`

3. **Download dataset**:
```bash
python scripts/download_data.py --dataset aptos
```

## Training

### Quick Start

```bash
# Train ViT-B/16 (default)
python main.py --data-dir data/raw/aptos2019-blindness-detection

# Train ResNet-50
python main.py --model resnet50

# Train EfficientNet-B4
python main.py --model efficientnet_b4
```

### Google Colab Full Training (Recommended for complete project training)

Use the notebook below to train all models on full APTOS data (ViT, ResNet-50, EfficientNet-B4) with GPU:

- **`Retinal_Disease_Classification_Full_Training_Colab_v2.ipynb`** ← Use this one (improved with Kaggle verification)

Older version (issues with dataset download):
- `Retinal_Disease_Classification_Full_Training_Colab.ipynb`

### Full Configuration

```bash
python main.py \
    --data-dir data/raw/aptos2019-blindness-detection \
    --model vit_base_patch16_224 \
    --epochs 100 \
    --batch-size 32 \
    --lr 3e-4 \
    --loss focal \
    --focal-gamma 2.0 \
    --mixup-alpha 0.2 \
    --early-stopping 15 \
    --output-dir experiments
```

## Evaluation

```bash
python main.py \
    --evaluate-only \
    --checkpoint experiments/vit_base_patch16_224_*/checkpoints/best_model.pth
```

## Project Structure

```
retinal_disease_classification/
├── configs/
│   └── config.yaml          # Configuration file
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed data
│   └── splits/               # Train/val/test splits
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration handling
│   ├── dataset.py            # Data loading
│   ├── augmentations.py      # Data augmentation
│   ├── models.py             # Model architectures
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   ├── trainer.py            # Training loop
│   └── utils.py              # Utilities
├── scripts/
│   └── download_data.py      # Dataset download
├── experiments/              # Experiment outputs
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md
```

## Expected Results

| Model | Accuracy | Balanced Acc | F1 (Macro) | Quad. Kappa |
|-------|----------|--------------|------------|-------------|
| ViT-B/16 | 85%+ | 80%+ | 0.80+ | 0.80+ |
| ResNet-50 | 82%+ | 77%+ | 0.76+ | 0.78+ |
| EfficientNet-B4 | 83%+ | 78%+ | 0.77+ | 0.79+ |

## Class Distribution (APTOS 2019)

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild DR | 370 | 10.1% |
| 2 | Moderate DR | 999 | 27.3% |
| 3 | Severe DR | 193 | 5.3% |
| 4 | Proliferative DR | 295 | 8.0% |

## References

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
2. Gulshan et al. (2016). "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy"
3. Lin et al. (2017). "Focal Loss for Dense Object Detection"

## License

This project is for educational and research purposes only.
