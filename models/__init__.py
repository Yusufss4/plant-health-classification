"""
Models package for plant health classification.

This package includes:
- FCNN: Fully Connected Neural Network (baseline)
- MobileViT-v2: Mobile Vision Transformer v2 (main model)

MobileViT-v2 is used instead of standard ViT for its efficiency:
- Hybrid CNN+Transformer design
- Separable self-attention (linear complexity)
- ~5M parameters vs ~86M in standard ViT
- Optimized for mobile/edge deployment
- Pretrained on ImageNet-1k
"""

from .fcnn import FCNN, create_fcnn_model
from .vit import MobileViTv2, VisionTransformer, create_vit_model

__all__ = [
    'FCNN',
    'create_fcnn_model',
    'MobileViTv2',
    'VisionTransformer',  # Alias for backwards compatibility
    'create_vit_model'
]
