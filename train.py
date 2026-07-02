"""
Training script for plant health classification models.

Usage:
    python train.py [--model MODEL_TYPE]

Arguments:
    --model: Registered model key (default: mobilenet_v3)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import build_model, get_model_spec, list_model_types
from utils import DEFAULT_CLASSES, create_data_loaders, calculate_metrics_per_epoch


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    val_loss, val_acc = calculate_metrics_per_epoch(model, val_loader, device)

    print(f'Epoch {epoch} [Val] - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)')

    return val_loss, val_acc


def train_model(model_type='mobilenet_v3'):
    """Main training function."""
    spec = get_model_spec(model_type)

    data_dir = 'data/'
    num_classes = len(DEFAULT_CLASSES)
    epochs = spec.epochs
    batch_size = spec.batch_size
    lr = spec.lr
    dropout = spec.dropout
    weight_decay = 1e-4
    checkpoint_dir = 'checkpoints'
    num_workers = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\nLoading data...')
    train_loader, val_loader, _ = create_data_loaders(
        train_dir=data_dir + 'train',
        val_dir=data_dir + 'val',
        test_dir=data_dir + 'test',
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')

    if len(val_loader.dataset) == 0:
        raise ValueError(
            'Validation set is empty. Run prepare_data.py (and '
            'prepare_background_data.py for the background class) first.'
        )
    if len(train_loader.dataset) == 0:
        raise ValueError(
            'Training set is empty. Run prepare_data.py (and '
            'prepare_background_data.py for the background class) first.'
        )

    checkpoint_path = os.path.join(
        checkpoint_dir, f'{model_type}_3cls_best.pth'
    )
    saved_checkpoint = False

    print(f'\nCreating {spec.display_name} ({model_type}, num_classes={num_classes})...')
    model = build_model(model_type, num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    print(f'Total parameters: {model.get_num_parameters():,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f'\nTraining for {epochs} epochs...\n')

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch}/{epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)')
        print(f'  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)')
        print(f'  Train-Val Gap: {abs(train_acc - val_acc)*100:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

            torch.save({
                'epoch': epoch,
                'model_type': model_type,
                'num_classes': num_classes,
                'class_names': list(DEFAULT_CLASSES),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history,
            }, checkpoint_path)
            saved_checkpoint = True
            print(f'  ✅ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})')

        print('-' * 80)

    print(f'\n{"="*80}')
    print('Training completed!')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
    if saved_checkpoint:
        print(f'Best model saved to: {checkpoint_path}')
    else:
        print('No checkpoint was saved (validation loss never improved).')
    print(f'{"="*80}')

    return history


def main():
    parser = argparse.ArgumentParser(description='Train plant health classification model')
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
    print(f"Training {spec.display_name} Model")
    print("=" * 80)
    train_model(model_type=args.model)


if __name__ == '__main__':
    main()
