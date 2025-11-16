"""
Evaluation utilities for model performance assessment.

This module provides comprehensive evaluation metrics and visualization tools
for comparing EfficientNet-B0 and MobileViT-v2 models.
"""

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
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)


def evaluate_model(model, dataloader, device, class_names=['healthy', 'diseased']):
    """
    Evaluate model on a dataset and return comprehensive metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on (cuda/cpu)
        class_names: List of class names
    
    Returns:
        dict: Dictionary containing all evaluation metrics including:
            - accuracy, precision, recall, f1_score
            - confusion_matrix with TP, TN, FP, FN
            - probabilities for threshold analysis
            - classification_report
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Classification report
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        output_dict=True
    )
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    return results


def print_evaluation_results(results, class_names=['healthy', 'diseased']):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary returned by evaluate_model
        class_names: List of class names
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"                  Predicted")
    print(f"              {class_names[0]:>10} {class_names[1]:>10}")
    print(f"Actual {class_names[0]:>10}  {cm[0,0]:>10}  {cm[0,1]:>10}")
    print(f"       {class_names[1]:>10}  {cm[1,0]:>10}  {cm[1,1]:>10}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp} ⚠️ (healthy classified as diseased)")
    print(f"  False Negatives (FN): {fn} ⚠️ (diseased classified as healthy - CRITICAL!)")
    print(f"  True Positives (TP):  {tp}")
    
    print(f"\nPer-Class Metrics:")
    report = results['classification_report']
    for class_name in class_names:
        metrics = report[class_name]
        print(f"\n  {class_name.capitalize()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1-score']:.4f}")
        print(f"    Support:   {metrics['support']}")
    
    print("=" * 60)


def plot_confusion_matrix(cm, class_names=['healthy', 'diseased'], save_path=None):
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
    
    # Add percentage annotations
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
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
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


def compare_models(results1, results2, model1_name='Model 1', model2_name='Model 2'):
    """
    Compare two models' performance metrics.
    
    Args:
        results1: Evaluation results for first model
        results2: Evaluation results for second model
        model1_name: Name of first model
        model2_name: Name of second model
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    print(f"\n{'Metric':<15} {model1_name:<20} {model2_name:<20} {'Improvement':<15}")
    print("-" * 80)
    
    for metric in metrics:
        val1 = results1[metric]
        val2 = results2[metric]
        improvement = ((val2 - val1) / val1) * 100 if val1 > 0 else 0
        
        print(f"{metric.capitalize():<15} {val1:.4f} ({val1*100:.2f}%){'':<6} "
              f"{val2:.4f} ({val2*100:.2f}%){'':<6} "
              f"{improvement:+.2f}%")
    
    # Compare confusion matrices
    print(f"\n{'Error Type':<30} {model1_name:<15} {model2_name:<15} {'Reduction':<15}")
    print("-" * 80)
    
    cm1 = results1['confusion_matrix']
    cm2 = results2['confusion_matrix']
    
    fp1 = cm1[0, 1]
    fp2 = cm2[0, 1]
    fp_reduction = ((fp1 - fp2) / fp1) * 100 if fp1 > 0 else 0
    
    fn1 = cm1[1, 0]
    fn2 = cm2[1, 0]
    fn_reduction = ((fn1 - fn2) / fn1) * 100 if fn1 > 0 else 0
    
    print(f"{'False Positives':<30} {fp1:<15} {fp2:<15} {fp_reduction:.1f}%")
    print(f"{'False Negatives (CRITICAL)':<30} {fn1:<15} {fn2:<15} {fn_reduction:.1f}%")
    
    print("=" * 80)


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


if __name__ == "__main__":
    # Example usage
    print("Evaluation Utilities")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - evaluate_model(): Comprehensive model evaluation")
    print("  - print_evaluation_results(): Print formatted results")
    print("  - plot_confusion_matrix(): Visualize confusion matrix")
    print("  - plot_training_history(): Plot training curves")
    print("  - compare_models(): Compare two models")
    print("  - calculate_metrics_per_epoch(): Quick metrics during training")
    print("\nExample:")
    print("  results = evaluate_model(model, test_loader, device)")
    print("  print_evaluation_results(results)")
    print("  plot_confusion_matrix(results['confusion_matrix'])")


def plot_metrics_vs_threshold(results, save_path=None):
    """
    Plot Precision, Recall, and F1-Score vs Classification Threshold.
    
    Args:
        results: Evaluation results dictionary with probabilities
        save_path: Optional path to save the figure
    """
    labels = results['labels']
    # Use probability of positive class (diseased = class 1)
    probs = results['probabilities'][:, 1]
    
    # Test different thresholds
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        
        # Handle edge cases
        if len(np.unique(preds)) < 2:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
        else:
            p = precision_score(labels, preds, zero_division=0)
            r = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Precision vs Threshold
    axes[0].plot(thresholds, precisions, 'b-', linewidth=2)
    axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    axes[0].set_xlabel('Classification Threshold', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    
    # Recall vs Threshold
    axes[1].plot(thresholds, recalls, 'g-', linewidth=2)
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    axes[1].set_xlabel('Classification Threshold', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    
    # F1-Score vs Threshold
    axes[2].plot(thresholds, f1_scores, 'm-', linewidth=2)
    axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    
    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    axes[2].plot(optimal_threshold, optimal_f1, 'r*', markersize=15, 
                label=f'Optimal: {optimal_threshold:.2f} (F1={optimal_f1:.3f})')
    
    axes[2].set_xlabel('Classification Threshold', fontsize=12)
    axes[2].set_ylabel('F1-Score', fontsize=12)
    axes[2].set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics vs threshold plot saved to {save_path}")
    
    plt.close()  # Close the figure to free memory
    
    return optimal_threshold, optimal_f1


def plot_precision_recall_curve(results, save_path=None):
    """
    Plot Precision-Recall (PR) Curve.
    
    Args:
        results: Evaluation results dictionary with probabilities
        save_path: Optional path to save the figure
    """
    labels = results['labels']
    # Use probability of positive class (diseased = class 1)
    probs = results['probabilities'][:, 1]
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.fill_between(recall, precision, alpha=0.2)
    
    # Add iso-F1 curves
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], 'gray', alpha=0.3, linestyle='--', linewidth=0.8)
        plt.annotate(f'F1={f_score:.1f}', xy=(0.9, f_score * 0.9 / (2 * 0.9 - f_score)), 
                    fontsize=8, color='gray')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    plt.close()  # Close the figure to free memory
    
    return avg_precision


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
    
    # Sanitize model name for use in filenames (replace forward slashes)
    safe_model_name = model_name.replace('/', '-').replace('\\', '-')
    
    # 1. Confusion Matrix
    print(f"\n1. Plotting Confusion Matrix for {model_name}...")
    cm_path = os.path.join(save_dir, f'{safe_model_name}_confusion_matrix.png') if save_dir else None
    plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
    if cm_path:
        plot_paths['confusion_matrix'] = cm_path
    
    # 2. Metrics vs Threshold
    print(f"\n2. Plotting Metrics vs Threshold for {model_name}...")
    threshold_path = os.path.join(save_dir, f'{safe_model_name}_metrics_vs_threshold.png') if save_dir else None
    optimal_threshold, optimal_f1 = plot_metrics_vs_threshold(results, save_path=threshold_path)
    if threshold_path:
        plot_paths['metrics_vs_threshold'] = threshold_path
    
    # 3. Precision-Recall Curve
    print(f"\n3. Plotting Precision-Recall Curve for {model_name}...")
    pr_path = os.path.join(save_dir, f'{safe_model_name}_pr_curve.png') if save_dir else None
    avg_precision = plot_precision_recall_curve(results, save_path=pr_path)
    if pr_path:
        plot_paths['pr_curve'] = pr_path
    
    print(f"\n✓ All plots generated for {model_name}")
    print(f"  - Average Precision: {avg_precision:.3f}")
    print(f"  - Optimal Threshold: {optimal_threshold:.3f} (F1={optimal_f1:.3f})")
    
    return plot_paths


def compare_models_comprehensive(results1, results2, model1_name='EfficientNet-B0', 
                                 model2_name='MobileViT-v2', save_dir=None):
    """
    Comprehensive comparison of two models with all metrics and plots.
    
    Args:
        results1: Evaluation results for first model
        results2: Evaluation results for second model
        model1_name: Name of first model
        model2_name: Name of second model
        save_dir: Directory to save comparison plots
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE MODEL COMPARISON: {model1_name} vs {model2_name}")
    print("="*80)
    
    # Print detailed metrics
    print(f"\n{model1_name} Results:")
    print("-"*80)
    print(f"Accuracy:  {results1['accuracy']:.4f} ({results1['accuracy']*100:.2f}%)")
    print(f"Precision: {results1['precision']:.4f} ({results1['precision']*100:.2f}%)")
    print(f"Recall:    {results1['recall']:.4f} ({results1['recall']*100:.2f}%)")
    print(f"F1-Score:  {results1['f1_score']:.4f} ({results1['f1_score']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {results1['true_positive']}, TN: {results1['true_negative']}")
    print(f"  FP: {results1['false_positive']}, FN: {results1['false_negative']}")
    
    print(f"\n{model2_name} Results:")
    print("-"*80)
    print(f"Accuracy:  {results2['accuracy']:.4f} ({results2['accuracy']*100:.2f}%)")
    print(f"Precision: {results2['precision']:.4f} ({results2['precision']*100:.2f}%)")
    print(f"Recall:    {results2['recall']:.4f} ({results2['recall']*100:.2f}%)")
    print(f"F1-Score:  {results2['f1_score']:.4f} ({results2['f1_score']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {results2['true_positive']}, TN: {results2['true_negative']}")
    print(f"  FP: {results2['false_positive']}, FN: {results2['false_negative']}")
    
    # Comparison table
    print(f"\n{'Metric':<20} {model1_name:<20} {model2_name:<20} {'Difference':<15}")
    print("-"*80)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        v1 = results1[metric]
        v2 = results2[metric]
        diff = v2 - v1
        symbol = "+" if diff > 0 else ""
        print(f"{metric.capitalize():<20} {v1*100:>6.2f}%{'':<13} {v2*100:>6.2f}%{'':<13} {symbol}{diff*100:>6.2f}%")
    
    print(f"\n{'Error Type':<20} {model1_name:<20} {model2_name:<20} {'Reduction':<15}")
    print("-"*80)
    fp_reduction = ((results1['false_positive'] - results2['false_positive']) / 
                    results1['false_positive'] * 100) if results1['false_positive'] > 0 else 0
    fn_reduction = ((results1['false_negative'] - results2['false_negative']) / 
                    results1['false_negative'] * 100) if results1['false_negative'] > 0 else 0
    
    print(f"{'False Positives':<20} {results1['false_positive']:<20} {results2['false_positive']:<20} {fp_reduction:.1f}%")
    print(f"{'False Negatives':<20} {results1['false_negative']:<20} {results2['false_negative']:<20} {fn_reduction:.1f}%")
    
    # Create side-by-side comparison plots
    if save_dir:
        # Side-by-side confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Model 1 confusion matrix
        sns.heatmap(results1['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['healthy', 'diseased'], yticklabels=['healthy', 'diseased'],
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_ylabel('Actual', fontsize=12)
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_title(f'{model1_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
        
        # Model 2 confusion matrix
        sns.heatmap(results2['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                   xticklabels=['healthy', 'diseased'], yticklabels=['healthy', 'diseased'],
                   ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_ylabel('Actual', fontsize=12)
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_title(f'{model2_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        comparison_path = os.path.join(save_dir, 'model_comparison_confusion_matrices.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {comparison_path}")
        plt.close()  # Close the figure to free memory
    
    print("\n" + "="*80)


