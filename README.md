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
- Extract all plant leaf images (healthy vs diseased) from 14 crop species
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

Train models with command line arguments:

```bash
# Train both models (default)
python train.py
python train.py --model both

# Train only EfficientNet-B0
python train.py --model cnn

# Train only DINOv3 ViT
python train.py --model vit
```

This will:
- Train EfficientNet-B0 for 10 epochs with batch_size=32, lr=0.001
- Train DINOv3 ViT-S/14 for 25 epochs with batch_size=16, lr=0.0001
- Save best models to `checkpoints/cnn_best.pth` and `checkpoints/vit_best.pth`

### Evaluate Models

Evaluate a specific model:

```bash
# Evaluate EfficientNet-B0 (default)
python evaluate.py
python evaluate.py --model cnn

# Evaluate DINOv3 ViT
python evaluate.py --model vit
```

This will evaluate the specified model on the test set and display metrics with confusion matrix.
Results are saved to `results/` directory.

- Side-by-side model comparison
