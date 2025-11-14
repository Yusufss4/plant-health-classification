# Plant Health Classification

A streamlined deep learning project for classifying plant leaves as healthy or diseased.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in the following structure:
```
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

## Usage

### Train Models

Train both EfficientNet-B0 and MobileViT-v2 models:
```bash
python train.py
```

This will:
- Train EfficientNet-B0 for 50 epochs with batch_size=32, lr=0.001
- Train MobileViT-v2 for 100 epochs with batch_size=16, lr=0.0001
- Save best models to `checkpoints/fcnn_best.pth` and `checkpoints/vit_best.pth`

Note: The model type parameter uses 'fcnn' for EfficientNet-B0 and 'vit' for MobileViT-v2 for backward compatibility.

### Evaluate Models

Evaluate a single model (EfficientNet-B0 by default):
```bash
python evaluate.py
```

This will evaluate the EfficientNet-B0 model on the test set and display metrics with confusion matrix.

### Compare Models

Compare both models with comprehensive metrics and plots:
```bash
python compare_models.py
```

This generates:
- Confusion matrices for both models
- Accuracy, precision, recall, F1-score metrics
- Comprehensive evaluation plots in `results/` directory
- Side-by-side model comparison
