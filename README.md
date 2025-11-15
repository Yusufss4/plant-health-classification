# Plant Health Classification

A streamlined deep learning project for classifying plant leaves as healthy or diseased.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the dataset:

**Option A: Automatic download (Recommended)**
```bash
python prepare_data.py
```

This will automatically:
- Download the PlantVillage dataset from TensorFlow Datasets
- Extract tomato leaf images (healthy vs diseased)
- Split into train (70%), validation (15%), and test (15%) sets
- Organize into the required directory structure

**Option B: Manual preparation**

If you have your own dataset, organize it in the following structure:
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

Train both EfficientNet-B0 and DINOv2 ViT-S/14 models:
```bash
python train.py
```

This will:
- Train EfficientNet-B0 for 10 epochs with batch_size=32, lr=0.001
- Train DINOv2 ViT-S/14 for 25 epochs with batch_size=16, lr=0.0001
- Save best models to `checkpoints/cnn_best.pth` and `checkpoints/vit_best.pth`

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
