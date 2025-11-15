"""
Models package for plant health classification.

This package includes:
- EfficientNet-B0: Modern efficient CNN (replaces traditional FCNN)
- DINOv3 ViT-S/16: Self-supervised Vision Transformer (main model)

EfficientNet-B0 is used instead of traditional FCNN for:
- Preserves spatial structure (no flattening)
- Compound scaling for efficiency
- ~5.3M parameters vs 307M in traditional FCNN
- State-of-the-art accuracy
- Pretrained on ImageNet

DINOv3 ViT-S/16 is used for:
- Self-supervised learning for robust features
- State-of-the-art transfer learning performance
- ~22M parameters, good balance of accuracy and efficiency
- Pretrained on large-scale diverse dataset (LVD-142M)
- Supports both feature extraction and fine-tuning modes
"""

from .efficientnet import EfficientNetB0, create_cnn_model
from .vit import DINOv3ViT, MobileViTv2, VisionTransformer, create_vit_model

__all__ = [
    'EfficientNetB0',
    'create_cnn_model',
    'DINOv3ViT',
    'MobileViTv2',  # Alias for backwards compatibility
    'VisionTransformer',  # Alias for backwards compatibility
    'create_vit_model'
]
