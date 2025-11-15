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

Train both EfficientNet-B0 and DINOv3 ViT-S/16 models:
```bash
python train.py
```

This will:
- Train EfficientNet-B0 for 10 epochs with batch_size=32, lr=0.001
- Train DINOv3 ViT-S/16 for 25 epochs with batch_size=16, lr=0.0001
- Save best models to `checkpoints/cnn_best.pth` and `checkpoints/vit_best.pth`
- DINOv3 model uses 256x256 input images with ImageNet normalization
- Supports both feature extraction (frozen backbone) and fine-tuning modes

### DINOv3 ViT-S/16 Model

The project now uses **DINOv3 ViT-S/16** (`facebook/dinov2-small`) instead of MobileViT-v2:

**Key Features:**
- Self-supervised learning on large-scale diverse datasets (LVD-142M)
- State-of-the-art transfer learning performance
- ~22M parameters (ViT-S variant)
- 256x256 input resolution
- Better generalization across different domains

**Training Modes:**

1. **Fine-tuning Mode** (default - recommended):
   ```python
   from models import create_vit_model
   model = create_vit_model(num_classes=2, freeze_backbone=False)
   # Trains entire model end-to-end
   ```

2. **Feature Extraction Mode** (faster training):
   ```python
   from models import create_vit_model
   model = create_vit_model(num_classes=2, freeze_backbone=True)
   # Only trains classification head, backbone frozen
   ```

3. **Switch modes dynamically**:
   ```python
   model.freeze_backbone_layers()  # Switch to feature extraction
   model.unfreeze_backbone()       # Switch to fine-tuning
   ```

**Note:** When internet is unavailable, the model automatically falls back to a ResNet18 architecture for compatibility.

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
