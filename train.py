"""
Training script for plant health classification models.

Usage:
    python train.py --model fcnn --epochs 50 --batch-size 32
    python train.py --model vit --epochs 100 --batch-size 16
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm

from models import create_fcnn_model, create_vit_model
from utils import create_data_loaders, calculate_metrics_per_epoch


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (for ViT stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    val_loss, val_acc = calculate_metrics_per_epoch(model, val_loader, device)
    
    print(f'Epoch {epoch} [Val] - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)')
    
    return val_loss, val_acc


def train_model(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('\nLoading data...')
    train_loader, val_loader, _ = create_data_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Create model
    print(f'\nCreating {args.model.upper()} model...')
    if args.model == 'fcnn':
        model = create_fcnn_model(num_classes=2, dropout_rate=args.dropout)
    elif args.model == 'vit':
        model = create_vit_model(num_classes=2, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f'Total parameters: {model.get_num_parameters():,}')
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.model == 'fcnn':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
    else:  # vit
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f'\nTraining for {args.epochs} epochs...\n')
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        if args.model == 'fcnn':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{args.epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)')
        print(f'  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)')
        print(f'  Train-Val Gap: {abs(train_acc - val_acc)*100:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'{args.model}_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
            print(f'  ✅ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epochs')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            break
        
        print('-' * 80)
    
    # Save final model
    final_checkpoint_path = os.path.join(
        args.checkpoint_dir, 
        f'{args.model}_final.pth'
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_checkpoint_path)
    
    print(f'\n{"="*80}')
    print('Training completed!')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
    print(f'Best model saved to: {checkpoint_path}')
    print(f'Final model saved to: {final_checkpoint_path}')
    print(f'{"="*80}')
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train plant health classification model')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['fcnn', 'vit'],
                        help='Model architecture (fcnn or vit)')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, default='data/train',
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default='data/val',
                        help='Path to validation data directory')
    parser.add_argument('--test-dir', type=str, default='data/test',
                        help='Path to test data directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.001 for FCNN, 0.0001 for ViT)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (default: 1e-4 for FCNN, 0.05 for ViT)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (default: 0.3 for FCNN, 0.1 for ViT)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set model-specific defaults
    if args.lr is None:
        args.lr = 0.001 if args.model == 'fcnn' else 0.0001
    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.model == 'fcnn' else 0.05
    if args.dropout is None:
        args.dropout = 0.3 if args.model == 'fcnn' else 0.1
    
    print('Configuration:')
    print(f'  Model: {args.model.upper()}')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch Size: {args.batch_size}')
    print(f'  Learning Rate: {args.lr}')
    print(f'  Weight Decay: {args.weight_decay}')
    print(f'  Dropout: {args.dropout}')
    print(f'  Early Stopping Patience: {args.patience}')
    
    # Train model
    history = train_model(args)
    
    # Optionally plot training history
    try:
        from utils import plot_training_history
        plot_training_history(
            history, 
            save_path=f'{args.checkpoint_dir}/{args.model}_training_history.png'
        )
    except Exception as e:
        print(f'Could not plot training history: {e}')


if __name__ == '__main__':
    main()
