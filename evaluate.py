"""
Evaluation script for plant health classification models.

Usage:
    python evaluate.py [--model MODEL_TYPE]
    
Arguments:
    --model: Model type to evaluate ('cnn' or 'vit'). Default: 'cnn'
"""

import argparse
import os
import torch

from models import create_cnn_model, create_vit_model
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
        model_type: Type of model ('cnn' or 'vit')
        weights_path: Path to model weights
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f'Loading {model_type.upper()} model from {weights_path}...')
    
    # Create model
    if model_type == 'cnn':
        model = create_cnn_model(num_classes=2)
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


def evaluate_single_model(model_type='cnn'):
    """
    Evaluate a single model on the test set.
    
    Args:
        model_type: Type of model to evaluate ('cnn' or 'vit')
    """
    # Hardcoded configuration
    test_dir = 'data/test'
    batch_size = 32
    model_path = f'checkpoints/{model_type}_best.pth'
    num_workers = 4
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(model_type, model_path, device)
    
    # Load data
    print('\nLoading test data...')
    _, _, test_loader = create_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        test_dir=test_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Evaluate
    print(f'\n{"="*80}')
    print(f'Evaluating on Test Set ({len(test_loader.dataset)} samples)')
    print(f'{"="*80}')
    
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print_evaluation_results(results)
    
    # Show confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], save_path=None)
    
    print('\nEvaluation completed!')
    
    return results


def main():
    """Evaluate model based on command line argument."""
    parser = argparse.ArgumentParser(description='Evaluate plant health classification model')
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=['cnn', 'vit'],
        help='Model type to evaluate (cnn or vit). Default: cnn'
    )
    
    args = parser.parse_args()
    
    model_names = {
        'cnn': 'EfficientNet-B0',
        'vit': 'DINOv2 Vision Transformer'
    }
    
    print("="*80)
    print(f"Evaluating {model_names[args.model]} Model")
    print("="*80)
    evaluate_single_model(model_type=args.model)


if __name__ == '__main__':
    main()
