"""
Evaluation script for plant health classification models.

Usage:
    python evaluate.py --model fcnn --weights checkpoints/fcnn_best.pth
    python evaluate.py --model vit --weights checkpoints/vit_best.pth
"""

import argparse
import os
import torch

from models import create_fcnn_model, create_vit_model
from utils import (
    create_data_loaders,
    evaluate_model,
    print_evaluation_results,
    plot_confusion_matrix
)


def load_model(model_type, weights_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: Type of model ('fcnn' or 'vit')
        weights_path: Path to model weights
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f'Loading {model_type.upper()} model from {weights_path}...')
    
    # Create model
    if model_type == 'fcnn':
        model = create_fcnn_model(num_classes=2)
    elif model_type == 'vit':
        model = create_vit_model(num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint info:')
        if 'epoch' in checkpoint:
            print(f'  Epoch: {checkpoint["epoch"]}')
        if 'val_acc' in checkpoint:
            print(f'  Val Accuracy: {checkpoint["val_acc"]:.4f} ({checkpoint["val_acc"]*100:.2f}%)')
        if 'val_loss' in checkpoint:
            print(f'  Val Loss: {checkpoint["val_loss"]:.4f}')
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded successfully!')
    print(f'Total parameters: {model.get_num_parameters():,}')
    
    return model


def evaluate_on_dataset(model, dataloader, device, dataset_name='Test'):
    """
    Evaluate model on a dataset and print results.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        dataset_name: Name of the dataset (for display)
    """
    print(f'\n{"="*80}')
    print(f'Evaluating on {dataset_name} Set ({len(dataloader.dataset)} samples)')
    print(f'{"="*80}')
    
    # Evaluate
    results = evaluate_model(model, dataloader, device)
    
    # Print results
    print_evaluation_results(results)
    
    return results


def compare_two_models(args):
    """Compare two models side by side."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('\nLoading test data...')
    _, _, test_loader = create_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load both models
    model1 = load_model(args.model1, args.weights1, device)
    model2 = load_model(args.model2, args.weights2, device)
    
    # Evaluate both models
    print(f'\n{"#"*80}')
    print(f'MODEL 1: {args.model1.upper()}')
    print(f'{"#"*80}')
    results1 = evaluate_on_dataset(model1, test_loader, device, 'Test')
    
    print(f'\n{"#"*80}')
    print(f'MODEL 2: {args.model2.upper()}')
    print(f'{"#"*80}')
    results2 = evaluate_on_dataset(model2, test_loader, device, 'Test')
    
    # Compare models
    from utils import compare_models
    print('\n')
    compare_models(
        results1, 
        results2, 
        model1_name=args.model1.upper(),
        model2_name=args.model2.upper()
    )
    
    # Plot confusion matrices side by side
    if args.save_plots:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Model 1
        plt.sca(axes[0])
        plot_confusion_matrix(
            results1['confusion_matrix'],
            save_path=None
        )
        axes[0].set_title(f'{args.model1.upper()} - Accuracy: {results1["accuracy"]:.2%}')
        
        # Model 2
        plt.sca(axes[1])
        plot_confusion_matrix(
            results2['confusion_matrix'],
            save_path=None
        )
        axes[1].set_title(f'{args.model2.upper()} - Accuracy: {results2["accuracy"]:.2%}')
        
        plt.tight_layout()
        save_path = 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\nComparison plot saved to {save_path}')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate plant health classification model')
    
    # Model arguments
    parser.add_argument('--model', type=str, choices=['fcnn', 'vit'],
                        help='Model architecture (fcnn or vit)')
    parser.add_argument('--weights', type=str,
                        help='Path to model weights')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Compare two models')
    parser.add_argument('--model1', type=str, choices=['fcnn', 'vit'],
                        help='First model for comparison')
    parser.add_argument('--weights1', type=str,
                        help='Weights for first model')
    parser.add_argument('--model2', type=str, choices=['fcnn', 'vit'],
                        help='Second model for comparison')
    parser.add_argument('--weights2', type=str,
                        help='Weights for second model')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, default='data/train',
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default='data/val',
                        help='Path to validation data directory')
    parser.add_argument('--test-dir', type=str, default='data/test',
                        help='Path to test data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Visualization
    parser.add_argument('--save-plots', action='store_true',
                        help='Save visualization plots')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare:
        if not all([args.model1, args.weights1, args.model2, args.weights2]):
            parser.error('--compare requires --model1, --weights1, --model2, and --weights2')
        compare_two_models(args)
        return
    
    if not args.model or not args.weights:
        parser.error('Single model evaluation requires --model and --weights')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model, args.weights, device)
    
    # Load data
    print('\nLoading test data...')
    _, _, test_loader = create_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate
    results = evaluate_on_dataset(model, test_loader, device, 'Test')
    
    # Save plots
    if args.save_plots:
        cm_path = os.path.join(args.output_dir, f'{args.model}_confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
        
        print(f'\nResults saved to {args.output_dir}/')
    
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()
