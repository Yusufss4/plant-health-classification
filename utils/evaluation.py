"""
Evaluation utilities for model performance assessment.

Model-agnostic metrics and visualization helpers for PyTorch classifiers.
"""

import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


def evaluate_model(
    model,
    dataloader,
    device,
    class_names=('healthy', 'diseased', 'background'),
    measure_inference_time=True,
):
    """
    Evaluate model on a dataset and return comprehensive metrics.

    Works for both binary (2-class) and multi-class (3+) setups. Per-class
    confusion-matrix counts (``per_class_tp``/``fp``/``fn``/``tn``) are always
    provided. Binary-only fields (``specificity``, scalar ROC-AUC) are
    populated only when ``len(class_names) == 2`` so legacy reports keep working.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on (cuda/cpu)
        class_names: Sequence of class names in label-index order. Default
            matches the 3-class model: 0=healthy, 1=diseased, 2=background.
        measure_inference_time: If True, wall-clock time for the full eval loop
            (transfer + forward + softmax).

    Returns:
        dict: Dictionary containing evaluation metrics including:
            - accuracy, balanced_accuracy, mcc
            - precision/recall/f1: scalar (binary) or macro-averaged (multi-class)
            - per_class_precision/recall/f1: arrays of length len(class_names)
            - per_class_tp/fp/fn/tn: one-vs-rest counts per class
            - confusion_matrix (NxN)
            - inference_timing
            - probabilities, predictions, labels (raw arrays)
            - classification_report
    """
    class_names = list(class_names)
    n_classes = len(class_names)
    is_binary = n_classes == 2

    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    if measure_inference_time and str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t_start = time.perf_counter() if measure_inference_time else None

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    if measure_inference_time:
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        total_sec = t_end - t_start
    else:
        total_sec = None

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(all_labels, all_predictions)

    avg = 'binary' if is_binary else 'macro'
    precision = precision_score(
        all_labels, all_predictions, average=avg, zero_division=0,
        labels=list(range(n_classes)),
    )
    recall = recall_score(
        all_labels, all_predictions, average=avg, zero_division=0,
        labels=list(range(n_classes)),
    )
    f1 = f1_score(
        all_labels, all_predictions, average=avg, zero_division=0,
        labels=list(range(n_classes)),
    )

    per_class_precision = precision_score(
        all_labels, all_predictions, average=None, zero_division=0,
        labels=list(range(n_classes)),
    )
    per_class_recall = recall_score(
        all_labels, all_predictions, average=None, zero_division=0,
        labels=list(range(n_classes)),
    )
    per_class_f1 = f1_score(
        all_labels, all_predictions, average=None, zero_division=0,
        labels=list(range(n_classes)),
    )

    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(n_classes)))

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)

    # Per-class confusion-matrix counts (one-vs-rest), valid for any class
    # count. For class i: TP is the diagonal, FP the column sum minus the
    # diagonal, FN the row sum minus the diagonal, TN the remainder.
    diag = np.diag(cm)
    per_class_tp = diag.astype(int)
    per_class_fp = (cm.sum(axis=0) - diag).astype(int)
    per_class_fn = (cm.sum(axis=1) - diag).astype(int)
    per_class_tn = (cm.sum() - per_class_tp - per_class_fp - per_class_fn).astype(int)

    # Binary-only convenience fields, kept for the legacy 2-class setup.
    specificity = None
    roc_auc = None
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except ValueError:
            roc_auc = None

    n_samples = len(all_labels)
    if measure_inference_time and total_sec is not None and n_samples > 0:
        inference_timing = {
            'total_sec': total_sec,
            'num_samples': n_samples,
            'avg_ms_per_image': (total_sec / n_samples) * 1000.0,
            'throughput_imgs_per_sec': n_samples / total_sec if total_sec > 0 else 0.0,
        }
    else:
        inference_timing = None

    results = {
        'class_names': class_names,
        'n_classes': n_classes,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'balanced_accuracy': balanced_acc,
        'specificity': specificity,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'per_class_tp': per_class_tp.tolist(),
        'per_class_fp': per_class_fp.tolist(),
        'per_class_fn': per_class_fn.tolist(),
        'per_class_tn': per_class_tn.tolist(),
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'inference_timing': inference_timing,
    }

    return results


def print_evaluation_results(results, class_names=None):
    """
    Print evaluation results in a formatted way (binary or multi-class).

    Args:
        results: Dictionary returned by evaluate_model
        class_names: Optional override; otherwise uses results['class_names'].
    """
    if class_names is None:
        class_names = results.get('class_names', ['healthy', 'diseased', 'background'])
    n_classes = len(class_names)
    is_binary = n_classes == 2
    metric_avg_label = 'binary' if is_binary else 'macro'

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    if results.get('balanced_accuracy') is not None:
        print(f"  Balanced accuracy:  {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    print(f"  Precision ({metric_avg_label}): {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall    ({metric_avg_label}): {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1-Score  ({metric_avg_label}): {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    if is_binary and results.get('specificity') is not None:
        print(f"  Specificity ({class_names[0]}): {results['specificity']:.4f}")
    if results.get('mcc') is not None:
        print(f"  MCC:                {results['mcc']:.4f}")
    if is_binary and results.get('roc_auc') is not None:
        print(f"  ROC-AUC:            {results['roc_auc']:.4f}")
    elif is_binary and 'roc_auc' in results:
        print(f"  ROC-AUC:            n/a (need both classes in labels)")

    timing = results.get('inference_timing')
    if timing:
        print(f"\nInference timing (full eval loop, {timing['num_samples']} samples):")
        print(f"  Total:     {timing['total_sec']:.4f} s")
        print(f"  Avg/image: {timing['avg_ms_per_image']:.3f} ms")
        print(f"  Throughput: {timing['throughput_imgs_per_sec']:.2f} img/s")

    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    cm = results['confusion_matrix']
    col_w = max(10, max(len(c) for c in class_names) + 2)
    header = " " * (col_w + 2) + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = f"{row_name:>{col_w}}  " + "".join(f"{int(cm[i, j]):>{col_w}}" for j in range(n_classes))
        print(row)

    per_class_tp = results.get('per_class_tp')
    per_class_fp = results.get('per_class_fp')
    per_class_fn = results.get('per_class_fn')
    per_class_tn = results.get('per_class_tn')
    if None not in (per_class_tp, per_class_fp, per_class_fn, per_class_tn):
        print(f"\nPer-Class Counts (one-vs-rest):")
        name_w = max(10, max(len(c) for c in class_names) + 2)
        print(f"{'Class':>{name_w}}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}")
        for i, class_name in enumerate(class_names):
            print(
                f"{class_name:>{name_w}}{int(per_class_tp[i]):>8}"
                f"{int(per_class_fp[i]):>8}{int(per_class_fn[i]):>8}"
                f"{int(per_class_tn[i]):>8}"
            )

    print(f"\nPer-Class Metrics:")
    report = results['classification_report']
    for class_name in class_names:
        if class_name not in report:
            continue
        metrics = report[class_name]
        print(f"\n  {class_name.capitalize()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1-score']:.4f}")
        print(f"    Support:   {metrics['support']}")

    print("=" * 60)


def plot_confusion_matrix(cm, class_names=('healthy', 'diseased', 'background'), save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix (2D array)
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    total = cm.sum()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()  # Close the figure to free memory


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.close()  # Close the figure to free memory


def calculate_metrics_per_epoch(model, dataloader, device):
    """
    Calculate metrics for a single epoch (used during training).
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on
    
    Returns:
        tuple: (loss, accuracy)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def plot_comprehensive_evaluation(results, model_name='Model', save_dir=None):
    """
    Create comprehensive evaluation plots for a model.
    
    Args:
        results: Evaluation results dictionary
        model_name: Name of the model for titles
        save_dir: Directory to save plots (optional)
    
    Returns:
        dict: Dictionary with plot paths if save_dir is provided
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    plot_paths = {}
    
    safe_model_name = model_name.replace('/', '-').replace('\\', '-')

    print(f"\n1. Plotting Confusion Matrix for {model_name}...")
    cm_path = os.path.join(save_dir, f'{safe_model_name}_confusion_matrix.png') if save_dir else None
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=results['class_names'],
        save_path=cm_path,
    )
    if cm_path:
        plot_paths['confusion_matrix'] = cm_path

    print(f"\nAll plots generated for {model_name}")
    
    return plot_paths


