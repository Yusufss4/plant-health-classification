# Plant Health Classification: Evaluation Results (Updated)

## Summary

| Metric | DINOv3 ViT | EfficientNet-B0 (Optimized) | Winner |
| --- | --- | --- | --- |
| **Accuracy** | 99.20% | **99.87%** | **CNN** |
| **Precision** | 99.54% | **99.90%** | **CNN** |
| **Recall** | 99.35% | **99.91%** | **CNN** |
| **F1-Score** | 99.44% | **99.91%** | **CNN** |
| **False Negatives** | 38 | **5** | **CNN** |
| **Parameters** | ~28.68M | **~4.01M** | **CNN** |

---

## Model 1: DINOv3 Vision Transformer

**Architecture:** `vit_small_plus_patch16_dinov3.lvd1689m`

The ViT model leverages DINOv3 self-supervised pre-training. While highly accurate, it struggles slightly more than the CNN with fine-grained feature extraction in this specific dataset.

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

### Performance Breakdown

* **Precision:** 0.9990
* **Recall:** 0.9991
* **F1-Score:** 0.9991

### Confusion Matrix

|  | Predicted: Healthy | Predicted: Diseased |
| --- | --- | --- |
| **Actual: Healthy** | 2327 (TN) | 6 (FP) |
| **Actual: Diseased** | 5 (FN) | 5832 (TP) |

---

## Key Observations

- **EfficientNet-B0 (CNN) outperforms DINOv3 ViT** on all major metrics (accuracy, precision, recall, F1-score) for this plant health classification task.
- **CNN model has significantly fewer parameters** (~4.01M) compared to the ViT (~28.68M), making it more efficient and lightweight for deployment.
- **False negatives are much lower** for EfficientNet-B0 (5 vs. 38), indicating better sensitivity in detecting diseased plants.
- **Both models achieve very high performance**, but EfficientNet-B0 is more robust and generalizes better on this dataset.
- **ViT model, despite its larger size and advanced architecture, does not provide a significant advantage** for this specific binary classification problem.
- **EfficientNet-B0 is preferable for real-world applications** where computational resources and inference speed are important.


