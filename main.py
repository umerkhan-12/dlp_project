"""
Main Entry Point for Retinal Disease Classification
====================================================
Train and evaluate Vision Transformer models for retinal disease classification.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, Config
from src.dataset import APTOSDataset, create_data_splits, create_dataloaders, save_splits, load_splits
from src.augmentations import get_train_transforms, get_val_transforms
from src.models import create_model, count_parameters
from src.losses import create_loss_function
from src.trainer import Trainer, create_optimizer, create_scheduler
from src.metrics import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_training_history, save_evaluation_results, generate_classification_report
)
from src.utils import set_seed, get_device, create_experiment_dir, save_training_config, print_model_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Retinal Disease Classification using Vision Transformers'
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/raw/aptos2019-blindness-detection',
                       help='Path to dataset directory')
    parser.add_argument('--splits-dir', type=str, default='data/splits',
                       help='Path to save/load data splits')

    # Model arguments
    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                       choices=['vit_base_patch16_224', 'resnet50', 'efficientnet_b4'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--early-stopping', type=int, default=15,
                       help='Early stopping patience')

    # Loss arguments
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['focal', 'cross_entropy', 'label_smoothing'],
                       help='Loss function')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')

    # Augmentation arguments
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='Mixup alpha (0 to disable)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')

    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory for experiments')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')

    # Mode arguments
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Class names for APTOS dataset
    class_names = ["No_DR", "Mild_DR", "Moderate_DR", "Severe_DR", "Proliferative_DR"]
    num_classes = len(class_names)

    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.model, args.experiment_name)
    print(f"Experiment directory: {exp_dir}")

    # Save configuration
    save_training_config(vars(args), exp_dir / 'config.json')

    # Load or create data splits
    splits_dir = Path(args.splits_dir)
    if (splits_dir / 'train.csv').exists():
        print("Loading existing data splits...")
        splits = load_splits(str(splits_dir))
    else:
        print("Creating new data splits...")
        # Load APTOS dataset
        aptos = APTOSDataset(args.data_dir)
        image_paths, labels = aptos.load_data()
        print(f"Loaded {len(image_paths)} images")

        # Show class distribution
        dist = aptos.get_class_distribution()
        print("Class distribution:")
        for name, count in dist.items():
            print(f"  {name}: {count}")

        # Create splits
        splits = create_data_splits(image_paths, labels)
        save_splits(splits, str(splits_dir))
        print(f"Splits saved to {splits_dir}")

    print(f"Train: {len(splits['train'][0])} samples")
    print(f"Val: {len(splits['val'][0])} samples")
    print(f"Test: {len(splits['test'][0])} samples")

    # Create transforms
    train_transform = get_train_transforms(
        image_size=args.image_size,
        rotation_limit=15,
        brightness=0.2,
        contrast=0.2,
    )
    val_transform = get_val_transforms(image_size=args.image_size)

    # Create data loaders
    dataloaders = create_dataloaders(
        train_data=splits['train'],
        val_data=splits['val'],
        test_data=splits['test'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    print_model_summary(model)

    # Move model to device
    model = model.to(device)

    # Calculate class weights from training data
    train_labels = np.array(splits['train'][1])
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = len(train_labels) / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Create loss function
    criterion = create_loss_function(
        loss_name=args.loss,
        num_classes=num_classes,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name='adamw',
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        layer_decay=0.65,  # Layer-wise LR decay
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name='cosine',
        num_epochs=args.epochs,
        warmup_epochs=5,
        steps_per_epoch=len(dataloaders['train']),
    )

    if args.evaluate_only:
        # Evaluation mode
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for evaluation")

        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on test set
        print("\nEvaluating on test set...")
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['test'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            mixed_precision=args.mixed_precision,
        )

        test_metrics = trainer.validate()

        # Print results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Accuracy:         {test_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:{test_metrics['balanced_accuracy']:.4f}")
        print(f"F1 (Macro):       {test_metrics['f1_macro']:.4f}")
        print(f"Quadratic Kappa:  {test_metrics['quadratic_kappa']:.4f}")
        if 'roc_auc_macro' in test_metrics:
            print(f"ROC-AUC (Macro):  {test_metrics['roc_auc_macro']:.4f}")
        print("=" * 60)

        # Save results
        save_evaluation_results(test_metrics, class_names, str(exp_dir / 'results'), prefix='test_')

    else:
        # Training mode
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            mixed_precision=args.mixed_precision,
            gradient_clip=1.0,
            mixup_alpha=args.mixup_alpha,
            experiment_dir=str(exp_dir),
        )

        # Train
        history = trainer.train(
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping,
            checkpoint_dir=str(exp_dir / 'checkpoints'),
        )

        # Plot training history
        plot_training_history(history, save_path=str(exp_dir / 'results' / 'training_history.png'))

        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best checkpoint
        best_checkpoint = exp_dir / 'checkpoints' / 'best_model.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint (epoch {checkpoint['epoch'] + 1})")

        # Evaluate on test set
        trainer.val_loader = dataloaders['test']
        test_metrics = trainer.validate()

        print(f"Test Accuracy:         {test_metrics['accuracy']:.4f}")
        print(f"Test Balanced Accuracy:{test_metrics['balanced_accuracy']:.4f}")
        print(f"Test F1 (Macro):       {test_metrics['f1_macro']:.4f}")
        print(f"Test Quadratic Kappa:  {test_metrics['quadratic_kappa']:.4f}")
        if 'roc_auc_macro' in test_metrics:
            print(f"Test ROC-AUC (Macro):  {test_metrics['roc_auc_macro']:.4f}")

        # Save results
        save_evaluation_results(test_metrics, class_names, str(exp_dir / 'results'), prefix='test_')

        # Generate and save classification report
        print("\nClassification Report:")
        # Need to run inference again to get predictions
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloaders['test']:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        report = generate_classification_report(
            np.array(all_labels), np.array(all_preds), class_names
        )
        print(report)

        # Save report
        with open(exp_dir / 'results' / 'classification_report.txt', 'w') as f:
            f.write(report)

        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(all_labels), np.array(all_preds), class_names,
            save_path=str(exp_dir / 'results' / 'confusion_matrix.png')
        )

        print(f"\nAll results saved to: {exp_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()
