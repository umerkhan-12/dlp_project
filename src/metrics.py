"""
Evaluation Metrics for Retinal Disease Classification
======================================================
Comprehensive evaluation metrics and visualization utilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    cohen_kappa_score,
)
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    num_classes = len(np.unique(y_true))

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'quadratic_kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'linear_kappa': cohen_kappa_score(y_true, y_pred, weights='linear'),
    }

    # Per-class metrics
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        try:
            # One-hot encode for multi-class ROC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

            # Compute ROC-AUC for each class
            roc_auc_per_class = []
            for i in range(num_classes):
                if len(np.unique(y_true_bin[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc_per_class.append(auc(fpr, tpr))
                else:
                    roc_auc_per_class.append(0.0)

            metrics['roc_auc_per_class'] = roc_auc_per_class
            metrics['roc_auc_macro'] = np.mean(roc_auc_per_class)
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")

    return metrics


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Tuple[List[float], List[float]]:
    """
    Compute sensitivity (recall) and specificity for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Tuple of (sensitivities, specificities)
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    sensitivities = []
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
        normalize: Whether to normalize the matrix
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for each class.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.preprocessing import label_binarize

    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})'
            )

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - One vs Rest')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot training history.

    Args:
        history: Dictionary containing training metrics over epochs
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Train')
        axes[1].plot(history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Learning Rate
    if 'lr' in history:
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Generate a text classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Classification report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def save_evaluation_results(
    metrics: Dict[str, Any],
    class_names: List[str],
    save_dir: str,
    prefix: str = "",
) -> None:
    """
    Save evaluation results to files.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        save_dir: Directory to save results
        prefix: Prefix for file names
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as text
    with open(save_dir / f"{prefix}metrics.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"F1 (Macro):         {metrics['f1_macro']:.4f}\n")
        f.write(f"F1 (Weighted):      {metrics['f1_weighted']:.4f}\n")
        f.write(f"Quadratic Kappa:    {metrics['quadratic_kappa']:.4f}\n")

        if 'roc_auc_macro' in metrics:
            f.write(f"ROC-AUC (Macro):    {metrics['roc_auc_macro']:.4f}\n")

        f.write("\n\nPer-Class Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 40 + "\n")

        for i, name in enumerate(class_names):
            f.write(f"{name:<20} ")
            f.write(f"{metrics['precision_per_class'][i]:.4f}       ")
            f.write(f"{metrics['recall_per_class'][i]:.4f}       ")
            f.write(f"{metrics['f1_per_class'][i]:.4f}\n")

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing evaluation metrics...")

    num_samples = 100
    num_classes = 5
    class_names = ["No_DR", "Mild_DR", "Moderate_DR", "Severe_DR", "Proliferative_DR"]

    # Generate dummy predictions
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    y_prob = np.random.rand(num_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"Quadratic Kappa: {metrics['quadratic_kappa']:.4f}")

    # Test plotting
    fig = plot_confusion_matrix(y_true, y_pred, class_names)
    plt.close(fig)
    print("Confusion matrix plotted successfully")

    # Test classification report
    report = generate_classification_report(y_true, y_pred, class_names)
    print("\nClassification Report:")
    print(report)
