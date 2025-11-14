# Training Pipeline

## Overview

This document describes the complete training pipeline for both FCNN and Vision Transformer models, including preprocessing, training configuration, optimization, and evaluation.

---

## 1. Data Preprocessing Pipeline

### Stage 1: Data Loading

```python
# Directory structure
data/
├── train/
│   ├── healthy/
│   └── diseased/
├── val/
│   ├── healthy/
│   └── diseased/
└── test/
    ├── healthy/
    └── diseased/
```

**Process:**
1. Load images from directory structure
2. Maintain class labels based on folder names
3. Create dataset objects for train/val/test splits

### Stage 2: Image Transformations

#### Training Set Transformations

```python
train_transforms = Compose([
    Resize((224, 224)),           # Resize to standard input size
    RandomHorizontalFlip(p=0.5),  # 50% chance horizontal flip
    RandomRotation(degrees=15),   # ±15 degree rotation
    ColorJitter(
        brightness=0.2,           # ±20% brightness
        contrast=0.2,             # ±20% contrast
        saturation=0.1,           # ±10% saturation
        hue=0.1                   # ±10% hue
    ),
    ToTensor(),                   # Convert to tensor [C, H, W]
    Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

**Why These Augmentations?**
- **Horizontal Flip**: Leaves can appear from any angle
- **Rotation**: Simulates different image capture angles
- **Color Jitter**: Accounts for lighting variations
- **Normalization**: Standardizes input distribution

#### Validation/Test Set Transformations

```python
val_test_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**No Augmentation**: Validation and test sets use only resizing and normalization for consistent evaluation.

### Stage 3: DataLoader Configuration

```python
# FCNN DataLoader
train_loader_fcnn = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)

# ViT DataLoader
train_loader_vit = DataLoader(
    train_dataset,
    batch_size=16,  # Smaller batch due to memory
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

**Key Parameters:**
- **Batch Size**: FCNN uses 32, ViT uses 16 (memory constraints)
- **Shuffle**: True for training, False for val/test
- **Num Workers**: Parallel data loading (4-8 workers)
- **Pin Memory**: Speeds up CPU-to-GPU data transfer

---

## 2. Model Initialization

### FCNN Initialization

```python
model_fcnn = FCNN(
    input_size=224 * 224 * 3,  # 150,528
    hidden_sizes=[2048, 1024, 512, 256],
    num_classes=2,
    dropout_rate=0.3
)

# Weight Initialization
for layer in model_fcnn.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)  # He initialization
        nn.init.constant_(layer.bias, 0.0)
```

**Initialization Strategy:**
- **Kaiming/He Initialization**: Optimal for ReLU activations
- **Bias Initialization**: Set to zero

### ViT Initialization

```python
model_vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=2,
    dim=768,              # Embedding dimension
    depth=12,             # Number of transformer layers
    heads=12,             # Number of attention heads
    mlp_dim=3072,         # MLP hidden dimension
    dropout=0.1,
    emb_dropout=0.1
)

# Option 1: Random initialization (training from scratch)
# Option 2: Load pre-trained weights from ImageNet
model_vit.load_pretrained('vit_base_patch16_224')
```

**Initialization Options:**
1. **From Scratch**: Random initialization
2. **Pre-trained**: Transfer learning from ImageNet (recommended)

---

## 3. Loss Function

### Binary Cross-Entropy Loss

Both models use **Cross-Entropy Loss** for binary classification:

```python
criterion = nn.CrossEntropyLoss()
```

**Mathematical Formula:**

```
Loss = -Σ [y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]

Where:
- y_true: Ground truth label (0 or 1)
- y_pred: Model prediction (probability)
```

**Why Cross-Entropy?**
- Standard for classification tasks
- Penalizes confident wrong predictions heavily
- Smooth gradients for optimization
- Works well with softmax output

### Weighted Loss (Optional)

For class imbalance:

```python
# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=[0, 1],
    y=train_labels
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights)
)
```

---

## 4. Optimization Strategy

### FCNN Optimizer

```python
optimizer_fcnn = torch.optim.Adam(
    model_fcnn.parameters(),
    lr=0.001,           # Learning rate
    betas=(0.9, 0.999), # Momentum parameters
    eps=1e-08,
    weight_decay=1e-4   # L2 regularization
)
```

**Optimizer Choice**: Adam
- Adaptive learning rates
- Momentum-based
- Efficient and widely used

### ViT Optimizer

```python
optimizer_vit = torch.optim.AdamW(
    model_vit.parameters(),
    lr=0.0001,          # Lower learning rate
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.05   # Stronger regularization
)
```

**Optimizer Choice**: AdamW
- Adam with decoupled weight decay
- Better regularization for transformers
- Recommended for ViT models

### Learning Rate Scheduling

#### FCNN: ReduceLROnPlateau

```python
scheduler_fcnn = ReduceLROnPlateau(
    optimizer_fcnn,
    mode='min',
    factor=0.5,      # Reduce LR by 50%
    patience=5,      # Wait 5 epochs
    min_lr=1e-6
)
```

**Strategy**: Reduce learning rate when validation loss plateaus

#### ViT: Cosine Annealing with Warmup

```python
scheduler_vit = CosineAnnealingWarmRestarts(
    optimizer_vit,
    T_0=10,          # Initial restart interval
    T_mult=2,        # Multiply interval after restart
    eta_min=1e-6     # Minimum LR
)

# Warmup for first 5 epochs
warmup_scheduler = LinearLR(
    optimizer_vit,
    start_factor=0.1,
    total_iters=5
)
```

**Strategy**: Warmup followed by cosine annealing
- Gradual learning rate increase (warmup)
- Smooth learning rate decay (cosine)
- Periodic restarts for better convergence

---

## 5. Training Loop

### FCNN Training Loop

```python
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training Phase
    model_fcnn.train()
    train_loss = 0.0
    train_correct = 0
    
    for images, labels in train_loader:
        # Move to GPU
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model_fcnn(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer_fcnn.zero_grad()
        loss.backward()
        optimizer_fcnn.step()
        
        # Track metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
    
    # Calculate epoch metrics
    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_dataset)
    
    # Validation Phase
    model_fcnn.eval()
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_fcnn(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset)
    
    # Learning rate scheduling
    scheduler_fcnn.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_fcnn.state_dict(), 'checkpoints/fcnn_best.pth')
    
    # Logging
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
```

### ViT Training Loop

Similar structure with additional components:

```python
num_epochs = 100  # ViT may need more epochs

for epoch in range(num_epochs):
    # Warmup learning rate (first 5 epochs)
    if epoch < 5:
        warmup_scheduler.step()
    else:
        scheduler_vit.step()
    
    # Training phase (same structure as FCNN)
    # ...
    
    # Gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_norm_(model_vit.parameters(), max_norm=1.0)
```

**ViT-Specific Considerations:**
- More epochs (50-100)
- Gradient clipping
- Warmup learning rate
- May benefit from mixed precision training

---

## 6. Evaluation Metrics

### Metrics Computed

#### 1. Accuracy

```python
accuracy = (correct_predictions / total_predictions) * 100
```

**Interpretation**: Overall percentage of correct classifications

#### 2. Precision

```python
precision = true_positives / (true_positives + false_positives)
```

**Interpretation**: Of all predicted diseased, how many were actually diseased?

#### 3. Recall (Sensitivity)

```python
recall = true_positives / (true_positives + false_negatives)
```

**Interpretation**: Of all actual diseased, how many did we detect?  
**Critical for disease detection** - missing diseased plants is costly!

#### 4. F1-Score

```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

**Interpretation**: Harmonic mean of precision and recall

#### 5. Confusion Matrix

```
                  Predicted
              Healthy  Diseased
Actual Healthy    TN      FP
       Diseased   FN      TP

Where:
- TN: True Negatives (correctly identified healthy)
- FP: False Positives (healthy classified as diseased)
- FN: False Negatives (diseased classified as healthy) ⚠️ CRITICAL
- TP: True Positives (correctly identified diseased)
```

### Evaluation Code

```python
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Collect predictions
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
cm = confusion_matrix(all_labels, all_predictions)

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'\nConfusion Matrix:\n{cm}')
```

### Visualization

#### Confusion Matrix Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Healthy', 'Diseased'],
    yticklabels=['Healthy', 'Diseased']
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

#### Training Curves

```python
plt.figure(figsize=(12, 5))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()
```

---

## 7. Model Checkpointing

### Saving Best Model

```python
# Save during training
if val_loss < best_val_loss:
    best_val_loss = val_loss
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, 'checkpoints/best_model.pth')
```

### Loading Model

```python
# Load for inference
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 8. Hardware and Training Time

### Computational Requirements

#### FCNN
- **GPU Memory**: ~4-6 GB
- **Training Time**: ~30-60 minutes (50 epochs, single GPU)
- **Inference Time**: ~5-10 ms per image

#### ViT
- **GPU Memory**: ~8-12 GB
- **Training Time**: ~2-4 hours (100 epochs, single GPU)
- **Inference Time**: ~10-20 ms per image

### Recommended Hardware

- **GPU**: NVIDIA RTX 3060+ or equivalent
- **RAM**: 16GB+
- **Storage**: 10GB+ (for dataset and checkpoints)

---

## Training Configuration Summary

| Parameter | FCNN | ViT |
|-----------|------|-----|
| **Epochs** | 50 | 100 |
| **Batch Size** | 32 | 16 |
| **Initial LR** | 0.001 | 0.0001 |
| **Optimizer** | Adam | AdamW |
| **LR Schedule** | ReduceLROnPlateau | Cosine + Warmup |
| **Weight Decay** | 1e-4 | 0.05 |
| **Dropout** | 0.3 | 0.1 |
| **Augmentation** | Standard | Standard |
| **Early Stopping** | Patience 10 | Patience 15 |

---

**Previous**: [← Solution Methods](03_solution_methods.md) | **Next**: [Results & Comparison →](05_results_comparison.md)
