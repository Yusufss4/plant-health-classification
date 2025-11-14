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

Train EfficientNet-B0:
```bash
python train.py --model fcnn --epochs 50 --batch-size 32
```

Train MobileViT-v2:
```bash
python train.py --model vit --epochs 50 --batch-size 16
```

### Evaluate Models

Evaluate a single model:
```bash
python evaluate.py --model fcnn --weights checkpoints/fcnn_best.pth
python evaluate.py --model vit --weights checkpoints/vit_best.pth
```

### Compare Models

Compare both models with comprehensive metrics and plots:
```bash
python compare_models.py \
    --efficientnet-weights checkpoints/efficientnet_best.pth \
    --mobilevit-weights checkpoints/mobilevit_best.pth \
    --data-dir data/ \
    --output-dir results/
```

This generates:
- Confusion matrices
- Accuracy, precision, recall, F1-score metrics
- Precision/recall/F1-score vs threshold plots
- Precision-Recall curves
- Side-by-side model comparison
