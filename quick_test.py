"""
Quick Test Script - CPU Training
================================
Fast validation of the training pipeline with reduced settings.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import timm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration for QUICK CPU test
class Config:
    data_dir = 'data/raw/aptos2019-blindness-detection'
    image_size = 128  # Smaller for speed
    num_classes = 5
    batch_size = 16
    num_epochs = 2  # Quick test
    learning_rate = 1e-3
    num_samples = 500  # Use subset for speed
    class_names = ['No_DR', 'Mild_DR', 'Moderate_DR', 'Severe_DR', 'Proliferative_DR']

config = Config()

print("=" * 60)
print("QUICK CPU TEST - Retinal Disease Classification")
print("=" * 60)
print(f"Image size: {config.image_size}x{config.image_size}")
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.num_epochs}")
print(f"Samples: {config.num_samples} (subset for speed)")
print("=" * 60)

# Simple transforms
train_transform = A.Compose([
    A.Resize(height=config.image_size, width=config.image_size),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=config.image_size, width=config.image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Dataset
class RetinalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row['diagnosis']

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

# Load data
print("\nLoading data...")
train_df = pd.read_csv(f'{config.data_dir}/train.csv')
train_df['image_path'] = train_df['id_code'].apply(
    lambda x: f"{config.data_dir}/train_images/{x}.png"
)

# Use subset for speed
if config.num_samples < len(train_df):
    train_df = train_df.sample(n=config.num_samples, random_state=42)

print(f"Using {len(train_df)} samples")

# Split
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['diagnosis']
)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Dataloaders
train_dataset = RetinalDataset(train_data, train_transform)
val_dataset = RetinalDataset(val_data, val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

# Model - using smaller efficientnet for speed
print("\nCreating model...")
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=config.num_classes)
device = torch.device('cpu')
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: EfficientNet-B0 ({total_params:,} parameters)")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(config.num_epochs):
    # Train
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_acc = accuracy_score(train_labels, train_preds)

    # Validate
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Val]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')

    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
    print(f"  Val F1 (Macro): {val_f1:.4f} | Val Kappa: {val_kappa:.4f}")

# Save model
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'model_name': 'efficientnet_b0',
        'num_classes': config.num_classes,
        'image_size': config.image_size,
    },
    'metrics': {
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_kappa': val_kappa,
    }
}, 'quick_test_model.pth')
print("Model saved to quick_test_model.pth")

print("\n" + "=" * 60)
print("QUICK TEST COMPLETE!")
print("=" * 60)
print(f"Final Val Accuracy: {val_acc:.4f}")
print(f"Final Val F1: {val_f1:.4f}")
print(f"Final Val Kappa: {val_kappa:.4f}")
print("\nThe pipeline is working! For full training, use Google Colab with GPU.")
