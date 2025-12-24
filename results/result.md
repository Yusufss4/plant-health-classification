# Plant Health Classification: Evaluation Results

## Executive Summary

Both models exhibit exceptional performance on the test set (8,170 samples), achieving accuracies above 99%. However, the **EfficientNet-B0 (CNN)** outperformed the ViT across all key metrics while maintaining a significantly smaller parameter footprint.

| Metric | DINOv3 ViT | EfficientNet-B0 | Winner |
| --- | --- | --- | --- |
| **Accuracy** | 99.20% | **99.66%** | CNN |
| **Precision** | 99.54% | **99.88%** | CNN |
| **Recall** | 99.35% | **99.64%** | CNN |
| **F1-Score** | 99.44% | **99.76%** | CNN |
| **False Negatives** | 38 | **21** | CNN |
| **Parameters** | ~28.68M | **~4.01M** | CNN |

---

## Model 1: DINOv3 Vision Transformer

**Architecture:** `vit_small_plus_patch16_dinov3.lvd1689m`

The ViT model leverages the DINOv3 self-supervised pre-training. It reached its best validation state at Epoch 14.

### Performance Breakdown

* **Precision:** 0.9954
* **Recall:** 0.9935
* **F1-Score:** 0.9944

### Confusion Matrix

|  | Predicted: Healthy | Predicted: Diseased |
| --- | --- | --- |
| **Actual: Healthy** | 2306 (TN) | 27 (FP) |
| **Actual: Diseased** | 38 (FN) | 5799 (TP) |

---

## Model 2: EfficientNet-B0 (CNN)

**Architecture:** EfficientNet-B0

The CNN model achieved superior convergence faster (Epoch 8) and demonstrated higher precision and recall, effectively minimizing the "Critical" false negatives where a diseased plant is missed.

### Performance Breakdown

* **Precision:** 0.9988
* **Recall:** 0.9964
* **F1-Score:** 0.9976

### Confusion Matrix

|  | Predicted: Healthy | Predicted: Diseased |
| --- | --- | --- |
| **Actual: Healthy** | 2326 (TN) | 7 (FP) |
| **Actual: Diseased** | 21 (FN) | 5816 (TP) |

---

## Key Observations

1. **Efficiency:** The EfficientNet-B0 model is roughly **7x smaller** in terms of parameters (4M vs 28.6M) while delivering higher accuracy. This makes it the ideal candidate for deployment on edge devices or mobile platforms.
2. **Reliability:** In a plant pathology context, **False Negatives (FN)** are the most costly error (missing a disease). The CNN reduced these errors by **~45%** compared to the ViT (21 vs 38).
3. **Class Imbalance:** Both models handled the higher support for the "Diseased" class (5,837 samples) vs "Healthy" (2,333 samples) well, though the CNN showed better robustness in its "Healthy" class recall.