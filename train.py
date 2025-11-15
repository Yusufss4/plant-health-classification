"""
Training script for plant health classification models.

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import create_cnn_model, create_vit_model
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


def train_model(model_type='cnn'):
    """Main training function."""
    
    # Hardcoded configuration
    data_dir = 'data/'
    if model_type == 'cnn':
        epochs = 10
        batch_size = 32
        lr = 0.001
        dropout = 0.3
    elif model_type == 'vit':
        epochs = 25
        batch_size = 16
        lr = 0.0001
        dropout = 0.1
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cnn' or 'vit'")
    
    weight_decay = 1e-4
    checkpoint_dir = 'checkpoints'
    num_workers = 4
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
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
    
    # Create model
    print(f'\nCreating {model_type.upper()} model...')
    if model_type == 'cnn':
        model = create_cnn_model(num_classes=2, dropout=dropout)
    else:  # vit
        model = create_vit_model(num_classes=2, dropout=dropout)
    
    model = model.to(device)
    print(f'Total parameters: {model.get_num_parameters():,}')
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - use Adam for both models
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
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
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f'\nTraining for {epochs} epochs...\n')
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)')
        print(f'  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f} ({val_acc*100:.2f}%)')
        print(f'  Train-Val Gap: {abs(train_acc - val_acc)*100:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'{model_type}_best.pth'
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
        
        print('-' * 80)
    
    print(f'\n{"="*80}')
    print('Training completed!')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    print(f'Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
    print(f'Best model saved to: {checkpoint_path}')
    print(f'{"="*80}')
    
    return history


def main():
    """Train both models."""
    print("="*80)
    print("Training EfficientNet-B0 Model")
    print("="*80)
    train_model(model_type='cnn')
    
    # print("\n\n")
    # print("="*80)
    # print("Training DINOv2 ViT-S/14 Model")
    # print("="*80)
    # train_model(model_type='vit')


if __name__ == '__main__':
    main()
