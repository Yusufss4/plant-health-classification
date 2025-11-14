# Plant Health Classification Project

## Overview

This project implements a binary classification system to identify healthy vs. diseased plant leaves using deep learning approaches. We compare two distinct methods:

1. **Fully Connected Neural Network (FCNN)** - Traditional dense layer approach
2. **Vision Transformer (ViT)** - Modern attention-based architecture

## Project Structure

```
plant-health-classification/
├── docs/                           # Detailed documentation
│   ├── 01_problem_definition.md   # Problem statement and motivation
│   ├── 02_dataset_explanation.md  # PlantVillage dataset details
│   ├── 03_solution_methods.md     # FCNN and ViT architectures
│   ├── 04_training_pipeline.md    # Training and evaluation process
│   ├── 05_results_comparison.md   # Results and comparative analysis
│   └── presentation_slides.md     # Slide-ready content
├── models/                         # Model architectures
│   ├── fcnn.py                    # FCNN implementation
│   └── vit.py                     # Vision Transformer implementation
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

Train the FCNN model:
```bash
python train.py --model fcnn --epochs 50 --batch-size 32
```

Train the ViT model:
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
- **Dual Approach**: Compare traditional FCNN with modern ViT
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Reproducible**: Detailed documentation and configurable parameters

## Documentation

For detailed explanations, please refer to the documentation in the `docs/` directory:

1. **[Problem Definition](docs/01_problem_definition.md)** - Why plant health classification matters
2. **[Dataset Explanation](docs/02_dataset_explanation.md)** - PlantVillage dataset details
3. **[Solution Methods](docs/03_solution_methods.md)** - FCNN and ViT architectures
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
| FCNN  | ~85-90%  | ~87%      | ~86%   | ~86%     |
| ViT   | ~93-97%  | ~95%      | ~94%   | ~94%     |

**Key Findings:**
- ✅ Vision Transformer significantly outperforms FCNN
- ✅ ViT better preserves spatial information through patch-based processing
- ✅ ViT shows less overfitting due to attention mechanisms
- ⚠️ FCNN struggles with spatial relationships due to flattening
- ⚠️ ViT requires more computational resources

## License

This project is for educational purposes.

## Acknowledgments

- PlantVillage Dataset
- PyTorch Community
- Vision Transformer (ViT) Paper: "An Image is Worth 16x16 Words"

## Contact

For questions or feedback, please open an issue in the repository.
