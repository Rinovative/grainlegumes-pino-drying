"""
Neural-operator device, inference, loss, metric, model, and training services.

Provides:
- device: concrete runtime device resolution and runtime inspection
- device_policy: dependency-free device-policy vocabulary and validation
- inference: saved-run inference reconstruction
- losses: supervised and physics-informed loss construction
- metrics: task-resolved training and evaluation metrics
- models: semantic neural-operator model construction
- temporal: explicit transient time-conditioning validation and transforms
- training: optimization, checkpoint, and completed-epoch execution
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import inference, losses, metrics, models, training
    from . import learning_device as device
    from . import learning_device_policy as device_policy
    from . import learning_temporal as temporal

_MODULES = {
    "device": "learning_device",
    "device_policy": "learning_device_policy",
    "inference": "inference",
    "losses": "losses",
    "metrics": "metrics",
    "models": "models",
    "temporal": "learning_temporal",
    "training": "training",
}
__all__ = ["device", "device_policy", "inference", "losses", "metrics", "models", "temporal", "training"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
