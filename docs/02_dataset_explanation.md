# Dataset Explanation: PlantVillage Dataset

## Dataset Overview

### What is PlantVillage?

The **PlantVillage dataset** is a comprehensive, publicly available collection of plant leaf images widely used for plant disease recognition research.

- **Source**: PlantVillage Project (Penn State University)
- **Purpose**: Support development of automated plant disease detection systems
- **Availability**: Open-source, freely available for research and educational use
- **Community**: Widely used benchmark in agricultural AI research

### Dataset Statistics

**Full Dataset:**
- **Total Images**: ~54,000+ images
- **Plant Species**: 14 different crop species
- **Disease Classes**: 38 different disease categories + healthy
- **Image Format**: Color (RGB) JPEG images
- **Image Quality**: High-resolution, controlled environment photos

**For This Binary Classification Project:**
- **Classes**: 2 (Healthy, Diseased)
- **Images per Class**: Balanced distribution (~10,000+ per class after aggregation)
- **Image Size**: Variable (typically 256x256 to 512x512 pixels)
- **Color Space**: RGB (3 channels)

## Dataset Characteristics

### Image Properties

1. **Controlled Environment**: 
   - Images captured in standardized conditions
   - Consistent lighting and background
   - Clear leaf visibility

2. **Variety**:
   - Multiple plant species (tomato, potato, apple, corn, grape, etc.)
   - Different disease types aggregated into "diseased" class
   - Various growth stages

3. **Quality**:
   - High-resolution images
   - Minimal noise or artifacts
   - Clear disease symptoms visible

### Class Distribution

**Healthy Leaves:**
- Clear, uniform green coloration
- No spots, lesions, or discoloration
- Normal leaf structure and shape

**Diseased Leaves:**
- Various disease symptoms:
  - Spots and lesions
  - Discoloration (yellowing, browning)
  - Wilting or deformation
  - Mold or fungal growth
- Aggregated from multiple disease types

## Why PlantVillage is Suitable for This Task

### Advantages

1. **Large Scale**: Sufficient data for deep learning training
2. **High Quality**: Clear images with visible features
3. **Diverse**: Multiple plant species and disease types
4. **Labeled**: Accurate ground-truth labels
5. **Standardized**: Consistent image capture methodology
6. **Benchmarked**: Enables comparison with existing research
7. **Accessible**: Free and open-source

### Relevance to Real-World

- **Representative Symptoms**: Covers common agricultural diseases
- **Visual Patterns**: Clear visual differences between healthy and diseased
- **Generalization**: Variety helps models generalize across species
- **Practical Scale**: Size enables robust model training

## Data Preprocessing Pipeline

### 1. Image Loading

```python
# Load images from directory structure
data/
  train/
    healthy/
    diseased/
  val/
    healthy/
    diseased/
  test/
    healthy/
    diseased/
```

### 2. Resizing

**Purpose**: Standardize input dimensions for neural networks

- **Target Size**: 224x224 pixels (standard for vision models)
- **Method**: Bilinear interpolation
- **Aspect Ratio**: Maintained or padded as needed

**Why 224x224?**
- Compatible with pre-trained models (ImageNet standard)
- Balance between detail preservation and computational efficiency
- Sufficient resolution to capture disease symptoms

### 3. Normalization

**Purpose**: Standardize pixel intensity distributions

**Method**: Z-score normalization using ImageNet statistics
```python
mean = [0.485, 0.456, 0.406]  # RGB channels
std = [0.229, 0.224, 0.225]   # RGB channels
normalized_image = (image - mean) / std
```

**Benefits:**
- Faster convergence during training
- Improved numerical stability
- Better gradient flow
- Consistent with pre-trained model expectations

### 4. Data Augmentation (Training Only)

**Purpose**: Increase dataset diversity and improve generalization

**Techniques Applied:**
- **Random Horizontal Flip**: Mirror images (50% probability)
- **Random Rotation**: ±15 degrees
- **Random Brightness**: ±20%
- **Random Contrast**: ±20%
- **Color Jitter**: Slight color variations

**Why Augmentation?**
- Simulates real-world variations (lighting, angle)
- Prevents overfitting
- Increases effective dataset size
- Improves model robustness

### 5. Tensor Conversion

Convert preprocessed images to PyTorch tensors:
- **Format**: [C, H, W] (Channels, Height, Width)
- **Data Type**: Float32
- **Range**: Normalized values

## Train/Validation/Test Split

### Split Ratios

- **Training Set**: 70% (~7,000+ images per class)
- **Validation Set**: 15% (~1,500+ images per class)
- **Test Set**: 15% (~1,500+ images per class)

### Split Strategy

**Stratified Split**: Maintains class distribution across splits

```
Total per class: 10,000 images
├── Train: 7,000 images (70%)
├── Validation: 1,500 images (15%)
└── Test: 1,500 images (15%)
```

### Purpose of Each Split

1. **Training Set**:
   - Used to train model parameters
   - Model learns feature representations
   - Gradient updates based on this data

2. **Validation Set**:
   - Monitor training progress
   - Hyperparameter tuning
   - Early stopping decisions
   - Model selection

3. **Test Set**:
   - Final performance evaluation
   - Never seen during training
   - Estimates real-world performance
   - Reports final metrics

## Data Loading Configuration

### Batch Processing

- **Batch Size**: 
  - FCNN: 32-64 images
  - ViT: 16-32 images (due to higher memory requirements)
  
- **Shuffling**: 
  - Training: Shuffled each epoch
  - Validation/Test: No shuffling

- **Workers**: Multi-threaded data loading (4-8 workers)

### Memory Optimization

- **On-the-fly Loading**: Images loaded and preprocessed during training
- **Caching**: Frequently accessed data cached in memory
- **GPU Transfer**: Efficient batching for GPU memory

## Dataset Statistics Summary

| Attribute | Value |
|-----------|-------|
| Total Images | ~20,000+ (binary split) |
| Image Channels | 3 (RGB) |
| Input Size | 224x224 pixels |
| Classes | 2 (Healthy, Diseased) |
| Training Samples | ~14,000 |
| Validation Samples | ~3,000 |
| Test Samples | ~3,000 |
| Augmentation | Yes (training only) |
| Normalization | ImageNet statistics |

## Ethical Considerations

- **Data Source**: Publicly available, ethically collected
- **Consent**: Images collected with appropriate permissions
- **Bias**: Dataset includes multiple plant species and disease types
- **Limitations**: Controlled environment; real-world conditions may vary

## Expected Challenges

1. **Class Imbalance**: May require balancing techniques
2. **Overfitting Risk**: Despite augmentation, model may memorize patterns
3. **Transfer to Real-World**: Controlled dataset vs. field conditions
4. **Species Generalization**: Performance may vary across plant species

---

**Previous**: [← Problem Definition](01_problem_definition.md) | **Next**: [Solution Methods →](03_solution_methods.md)
