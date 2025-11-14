"""
Evaluation utilities for model performance assessment.
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
    classification_report
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
        dict: Dictionary containing all evaluation metrics
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
    
    plt.show()


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
    
    plt.show()


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
