"""
Saved-run steady and transient inference reconstruction.

Provides:
- context: steady model, normalizer, split, and dataloader reconstruction
- transient: explicit transient-drying model and physical rollout service
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import learning_inference as context
    from . import learning_inference_transient as transient

_MODULES = {"context": "learning_inference", "transient": "learning_inference_transient"}
__all__ = ["context", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
