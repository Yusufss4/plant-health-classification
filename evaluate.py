"""
Evaluation script for plant health classification models.

Usage:
    python evaluate.py [--model MODEL_TYPE]

Arguments:
    --model: Registered model key (default: mobilenet_v3)
"""

import argparse
import torch

from models import build_model, get_model_spec, list_model_types
from utils import (
    DEFAULT_CLASSES,
    create_data_loaders,
    evaluate_model,
    print_evaluation_results,
    plot_confusion_matrix,
)


def _checkpoint_class_info(checkpoint, fallback_num_classes=3):
    """Read num_classes and class_names from checkpoint, with defaults."""
    num_classes = int(checkpoint.get('num_classes', fallback_num_classes))
    class_names = list(checkpoint.get('class_names', DEFAULT_CLASSES))
    if len(class_names) != num_classes:
        raise ValueError(
            f'Checkpoint class_names length ({len(class_names)}) != '
            f'num_classes ({num_classes})'
        )
    return num_classes, class_names


def load_model(model_type, weights_path, device):
    """
    Load a trained model from checkpoint.

    Args:
        model_type: Registered model key
        weights_path: Path to model weights
        device: Device to load model on

    Returns:
        tuple: (loaded model, class_names list)
    """
    spec = get_model_spec(model_type)
    print(f'Loading {spec.display_name} ({model_type}) from {weights_path}...')

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = (
        checkpoint['model_state_dict']
        if 'model_state_dict' in checkpoint
        else checkpoint
    )
    num_classes, class_names = _checkpoint_class_info(
        checkpoint if isinstance(checkpoint, dict) else {}
    )

    if isinstance(checkpoint, dict) and 'model_type' in checkpoint:
        ckpt_type = checkpoint['model_type']
        if ckpt_type != model_type:
            print(
                f'  Warning: checkpoint model_type={ckpt_type!r} differs from '
                f'CLI --model={model_type!r}; using CLI.'
            )

    model = build_model(model_type, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print('Checkpoint info:')
        if 'model_type' in checkpoint:
            print(f'  Model type: {checkpoint["model_type"]}')
        if 'num_classes' in checkpoint:
            print(f'  Num classes: {checkpoint["num_classes"]}')
        if 'class_names' in checkpoint:
            print(f'  Class names: {checkpoint["class_names"]}')
        if 'epoch' in checkpoint:
            print(f'  Epoch: {checkpoint["epoch"]}')
        if 'val_acc' in checkpoint:
            print(f'  Val Accuracy: {checkpoint["val_acc"]:.4f} ({checkpoint["val_acc"]*100:.2f}%)')
        if 'val_loss' in checkpoint:
            print(f'  Val Loss: {checkpoint["val_loss"]:.4f}')

    model = model.to(device)
    model.eval()

    print('Model loaded successfully!')
    print(f'Total parameters: {model.get_num_parameters():,}')

    return model, class_names


def evaluate_single_model(model_type='mobilenet_v3'):
    """
    Evaluate a single model on the test set.

    Args:
        model_type: Registered model key
    """
    test_dir = 'data/test'
    batch_size = 32
    model_path = f'checkpoints/{model_type}_3cls_best.pth'
    num_workers = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model, class_names = load_model(model_type, model_path, device)

    print('\nLoading test data...')
    _, _, test_loader = create_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        test_dir=test_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        classes=class_names,
    )

    print(f'\n{"="*80}')
    print(f'Evaluating on Test Set ({len(test_loader.dataset)} samples)')
    print(f'{"="*80}')

    results = evaluate_model(
        model, test_loader, device, class_names=class_names
    )

    print_evaluation_results(results, class_names=class_names)

    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=class_names,
        save_path=None,
    )

    print('\nEvaluation completed!')

    return results


def main():
    """Evaluate model based on command line argument."""
    parser = argparse.ArgumentParser(description='Evaluate plant health classification model')
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenet_v3',
        choices=list_model_types(),
        help='Registered model type (default: mobilenet_v3)',
    )

    args = parser.parse_args()
    spec = get_model_spec(args.model)

    print("=" * 80)
    print(f"Evaluating {spec.display_name} Model")
    print("=" * 80)
    evaluate_single_model(model_type=args.model)


if __name__ == '__main__':
    main()
