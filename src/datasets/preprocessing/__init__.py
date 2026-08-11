"""
Identity-bound Dataset splitting and train-only normalization services.

Provides:
- normalization: normalizer fitting, reconstruction, and artifact admission
- splits: deterministic split and membership contracts
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_preprocessing_normalization as normalization
    from . import dataset_preprocessing_splits as splits

_MODULES = {
    "normalization": "dataset_preprocessing_normalization",
    "splits": "dataset_preprocessing_splits",
}
__all__ = ["normalization", "splits"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
