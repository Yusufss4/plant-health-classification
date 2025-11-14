# Plant Health Classification Project

## Overview

This project implements a binary classification system to identify healthy vs. diseased plant leaves using deep learning approaches. We compare two distinct methods:

1. **EfficientNet-B0 CNN (EfficientNet-B0)** - Traditional dense layer approach
2. **MobileViT-v2** - Modern attention-based architecture

## Project Structure

```
plant-health-classification/
├── docs/                           # Detailed documentation
│   ├── 01_problem_definition.md   # Problem statement and motivation
│   ├── 02_dataset_explanation.md  # PlantVillage dataset details
│   ├── 03_solution_methods.md     # EfficientNet-B0 and MobileViT-v2 architectures
│   ├── 04_training_pipeline.md    # Training and evaluation process
│   ├── 05_results_comparison.md   # Results and comparative analysis
│   └── presentation_slides.md     # Slide-ready content
├── models/                         # Model architectures
│   ├── fcnn.py                    # EfficientNet-B0 implementation
│   └── vit.py                     # MobileViT-v2 implementation
├── utils/                          # Utility functions
│   ├── data_loader.py             # Data loading and preprocessing
│   └── evaluation.py              # Evaluation metrics
├── train.py                        # Training script
├── evaluate.py                     # Evaluation script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Download the PlantVillage dataset and organize it as follows:

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

### Training

Train the EfficientNet-B0 model:
```bash
python train.py --model fcnn --epochs 50 --batch-size 32
```

Train the MobileViT-v2 model:
```bash
python train.py --model vit --epochs 50 --batch-size 16
```

### Evaluation

```bash
python evaluate.py --model fcnn --weights checkpoints/fcnn_best.pth
python evaluate.py --model vit --weights checkpoints/vit_best.pth
```

## Key Features

- **Binary Classification**: Healthy vs. Diseased plant leaves
- **Dual Approach**: Compare traditional EfficientNet-B0 with modern ViT
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Reproducible**: Detailed documentation and configurable parameters

## Documentation

For detailed explanations, please refer to the documentation in the `docs/` directory:

1. **[Problem Definition](docs/01_problem_definition.md)** - Why plant health classification matters
2. **[Dataset Explanation](docs/02_dataset_explanation.md)** - PlantVillage dataset details
3. **[Solution Methods](docs/03_solution_methods.md)** - EfficientNet-B0 and MobileViT-v2 architectures
4. **[Training Pipeline](docs/04_training_pipeline.md)** - Complete training workflow
5. **[Results & Comparison](docs/05_results_comparison.md)** - Performance analysis

## Presentation

For slide-ready content suitable for presentations, see:
- **[Presentation Slides](docs/presentation_slides.md)** - Complete slide deck in Markdown format

## Technologies Used

- **PyTorch**: Deep learning framework
- **torchvision**: Image preprocessing and augmentation
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Evaluation metrics

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| EfficientNet-B0  | ~85-90%  | ~87%      | ~86%   | ~86%     |
| MobileViT-v2   | ~93-97%  | ~95%      | ~94%   | ~94%     |

**Key Findings:**
- ✅ MobileViT-v2 significantly outperforms EfficientNet-B0
- ✅ MobileViT-v2 better preserves spatial information through patch-based processing
- ✅ MobileViT-v2 shows less overfitting due to attention mechanisms
- ⚠️ EfficientNet-B0 struggles with spatial relationships due to convolutional layersing
- ⚠️ MobileViT-v2 requires more computational resources

## License

This project is for educational purposes.

## Acknowledgments

- PlantVillage Dataset
- PyTorch Community
- MobileViT-v2 Paper: "An Image is Worth 16x16 Words"

## Contact

For questions or feedback, please open an issue in the repository.

## Comprehensive Model Evaluation

This project includes comprehensive evaluation tools with all metrics and visualizations.

### Quick Comparison

Compare both models with all metrics and plots:

```bash
python compare_models.py \
    --efficientnet-weights checkpoints/efficientnet_best.pth \
    --mobilevit-weights checkpoints/mobilevit_best.pth \
    --data-dir data/ \
    --output-dir results/
```

### Generated Metrics

- **Confusion Matrix** with TP, TN, FP, FN
- **Accuracy, Precision, Recall, F1-Score**
- **Precision vs Threshold** plot
- **Recall vs Threshold** plot
- **F1-Score vs Threshold** plot (with optimal threshold)
- **Precision-Recall (PR) Curve** with average precision

### Output Structure

```
results/
├── efficientnet/          # EfficientNet-B0 plots
├── mobilevit/             # MobileViT-v2 plots
├── comparison/            # Side-by-side comparison
└── comparison_results.txt # Detailed metrics
```

See `docs/EVALUATION_GUIDE.md` for complete documentation.

