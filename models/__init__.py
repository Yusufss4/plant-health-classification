"""
Models package for plant health classification.

This package includes:
- EfficientNet-B0: Modern efficient CNN (replaces traditional FCNN)
- DINOv2 ViT-S/14: Self-supervised Vision Transformer (main model)

EfficientNet-B0 is used instead of traditional FCNN for:
- Preserves spatial structure (no flattening)
- Compound scaling for efficiency
- ~5.3M parameters vs 307M in traditional FCNN
- State-of-the-art accuracy
- Pretrained on ImageNet

DINOv2 ViT-S/14 is used for:
- Self-supervised learning on diverse data
- High-quality visual features
- ~21M parameters in backbone
- Robust to domain shift
- Supports frozen features and fine-tuning
- Register tokens for enhanced features
"""

from .efficientnet import EfficientNetB0, create_cnn_model
from .vit import DINOv2ViT, VisionTransformer, create_vit_model

__all__ = [
    'EfficientNetB0',
    'create_cnn_model',
    'DINOv2ViT',
    'VisionTransformer',  # Alias for backwards compatibility
    'create_vit_model'
]
