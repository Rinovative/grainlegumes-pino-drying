"""
Semantic neural-operator model construction.

Provides:
- naming: metadata-only architecture descriptors
- factory: configuration-driven FNO and UNO construction
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import learning_models_factory as factory
    from . import learning_models_naming as naming

_MODULES = {"factory": "learning_models_factory", "naming": "learning_models_naming"}
__all__ = ["factory", "naming"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
