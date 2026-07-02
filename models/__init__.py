"""
Models package for plant health classification.

MobileNet-v3-Small is the production backbone (PyTorch training + ONNX export).
Additional architectures can be registered in models.registry.
"""

from .mobilenet_v3 import MobileNetV3Small, create_mobilenet_v3_model
from .registry import (
    MODEL_REGISTRY,
    ModelSpec,
    build_model,
    get_model_spec,
    list_model_types,
)

__all__ = [
    "MobileNetV3Small",
    "create_mobilenet_v3_model",
    "MODEL_REGISTRY",
    "ModelSpec",
    "build_model",
    "get_model_spec",
    "list_model_types",
]
