"""
Explicit production experiment validation.

Provides:
- data_pipeline: typed read-only full-data pipeline validation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import experiments_validation_data_pipeline as data_pipeline

_MODULES = {"data_pipeline": "experiments_validation_data_pipeline"}
__all__ = ["data_pipeline"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
