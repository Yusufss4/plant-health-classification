"""
Model registry for plant health classification.

Register new backbones here (factory + training hyperparameters) so train.py
and evaluate.py stay model-agnostic.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .mobilenet_v3 import create_mobilenet_v3_model


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    factory: Callable[..., Any]
    epochs: int
    batch_size: int
    lr: float
    dropout: float


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mobilenet_v3": ModelSpec(
        display_name="MobileNet-v3-Small",
        factory=create_mobilenet_v3_model,
        epochs=15,
        batch_size=32,
        lr=1e-4,
        dropout=0.2,
    ),
}


def list_model_types() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_spec(model_type: str) -> ModelSpec:
    if model_type not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model type: {model_type!r}. Registered: {known}"
        )
    return MODEL_REGISTRY[model_type]


def build_model(model_type: str, num_classes: int, **factory_kwargs: Any):
    spec = get_model_spec(model_type)
    return spec.factory(num_classes=num_classes, **factory_kwargs)
