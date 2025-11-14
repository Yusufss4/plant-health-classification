"""
Models package for plant health classification.

This package includes:
- EfficientNet-B0: Modern efficient CNN (replaces traditional FCNN)
- MobileViT-v2: Mobile Vision Transformer v2 (main model)

EfficientNet-B0 is used instead of traditional FCNN for:
- Preserves spatial structure (no flattening)
- Compound scaling for efficiency
- ~5.3M parameters vs 307M in traditional FCNN
- State-of-the-art accuracy
- Pretrained on ImageNet

MobileViT-v2 is used for:
- Hybrid CNN+Transformer design
- Separable self-attention (linear complexity)
- ~5M parameters, mobile/edge optimized
- Pretrained on ImageNet-1k
"""

from .efficientnet import EfficientNetB0, FCNN, create_fcnn_model
from .vit import MobileViTv2, VisionTransformer, create_vit_model

__all__ = [
    'EfficientNetB0',
    'FCNN',  # Alias for backwards compatibility
    'create_fcnn_model',
    'MobileViTv2',
    'VisionTransformer',  # Alias for backwards compatibility
    'create_vit_model'
]
