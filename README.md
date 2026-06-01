# Plant Health Classification

Deep learning project for classifying plant leaves and scene context as **healthy**, **diseased**, or **background** (non-leaf / empty frame). Training uses PyTorch; edge deployment uses **MobileNet-v3** exported to ONNX and C++ ONNX Runtime.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare the dataset:

**Step 1 — Leaf images (PlantVillage)**

```bash
python prepare_data.py
# Reproducible split:
python prepare_data.py --seed 42
```

This downloads PlantVillage via TensorFlow Datasets, maps 38 fine-grained labels to **healthy** vs **diseased**, and splits into train (70%), validation (15%), and test (15%).

**Step 2 — Background class (required for 3-class training)**

```bash
python prepare_background_data.py
```

Adds `data/{train,val,test}/background/` from filtered COCO images and synthetic patches. Without this step, the model trains with only two populated classes while the head still has three outputs.

**Verify layout**

```bash
python prepare_data.py --verify-only
python prepare_data.py --verify-only --require-background
```

**Manual layout** (if not using the download scripts):

```
data/
├── train/
│   ├── healthy/
│   ├── diseased/
│   └── background/
├── val/
│   ├── healthy/
│   ├── diseased/
│   └── background/
└── test/
    ├── healthy/
    ├── diseased/
    └── background/
```

Class index order (fixed everywhere — Python, checkpoints, ONNX metadata, C++):

| Index | Name |
|-------|------|
| 0 | `healthy` |
| 1 | `diseased` |
| 2 | `background` |

## Usage

### Train models

```bash
# MobileNet-v3 (edge / ONNX path)
python train.py --model mobilenet_v3

# EfficientNet-B0
python train.py --model cnn

# DINOv3 ViT (timm)
python train.py --model vit

# Train EfficientNet + ViT (default)
python train.py
python train.py --model both
```

Training hyperparameters (see [`train.py`](train.py)):

| Model | Epochs | Batch | LR | Dropout | Checkpoint |
|-------|--------|-------|-----|---------|------------|
| `cnn` (EfficientNet-B0) | 10 | 32 | 1e-4 | 0.3 | `checkpoints/cnn_3cls_best.pth` |
| `mobilenet_v3` | 15 | 32 | 1e-4 | 0.2 | `checkpoints/mobilenet_v3_3cls_best.pth` |
| `vit` | 25 | 16 | 1e-4 | 0.1 | `checkpoints/vit_3cls_best.pth` |

Checkpoints include `num_classes`, `class_names`, and `model_type` metadata for evaluation and export.

### Evaluate models

```bash
python evaluate.py --model mobilenet_v3
python evaluate.py --model cnn
python evaluate.py --model vit
```

Loads `checkpoints/{model}_3cls_best.pth` and reports test metrics plus a confusion matrix.

### Export ONNX (MobileNet → C++ / Pi)

```bash
python export_mobilenet_onnx.py
# → checkpoints/mobilenet_v3_3cls.onnx (with class metadata in ONNX metadata_props)
```

### C++ inference

See [`cpp/README.md`](cpp/README.md) for building `phc_infer_mobilenet`, `phc_evaluate_mobilenet`, and optional `live_infer_web`.

Parity check (shared preprocessed tensor):

```bash
bash scripts/validate_cpp_inference.sh /path/to/leaf.jpg ./cpp/build/local-release/phc_infer_mobilenet
```

## End-to-end workflow (deployment)

1. `python prepare_data.py [--seed 42]`
2. `python prepare_background_data.py`
3. `python train.py --model mobilenet_v3`
4. `python export_mobilenet_onnx.py`
5. `python evaluate.py --model mobilenet_v3`
6. Build and run C++ tools with `checkpoints/mobilenet_v3_3cls.onnx`
