"""
Models package for plant health classification.

This package includes:
- EfficientNet-B0: Modern efficient CNN (replaces traditional FCNN)
- DINOv3 ViT-S/14: Self-supervised Vision Transformer (main model)

EfficientNet-B0 is used instead of traditional FCNN for:
- Preserves spatial structure (no flattening)
- Compound scaling for efficiency
- ~5.3M parameters vs 307M in traditional FCNN
- State-of-the-art accuracy
- Pretrained on ImageNet

DINOv3 ViT-S/14 is used for:
- Enhanced self-supervised learning on massive datasets
- Superior visual features for downstream tasks
- ~21M parameters in backbone
- Even better robustness to domain shift
- Supports frozen features and fine-tuning
- Improved register tokens for enhanced features
"""

from .efficientnet import EfficientNetB0, create_cnn_model
from .vit import DINOv3ViT, VisionTransformer, create_vit_model

__all__ = [
    'EfficientNetB0',
    'create_cnn_model',
    'DINOv3ViT',
    'VisionTransformer',  # Alias for backwards compatibility
    'create_vit_model'
]
