"""
Identity-bound Dataset splitting and train-only scaling services.

Provides:
- normalization: steady normalizer fitting, reconstruction, and preprocessing specification
- scaling: shared scaling identity, fit-or-load cache, and persisted scaling artifact admission
- splits: the one persisted split schema of both tasks and its admission
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_preprocessing_normalization as normalization
    from . import dataset_preprocessing_scaling as scaling
    from . import dataset_preprocessing_splits as splits

_MODULES = {
    "normalization": "dataset_preprocessing_normalization",
    "scaling": "dataset_preprocessing_scaling",
    "splits": "dataset_preprocessing_splits",
}
__all__ = ["normalization", "scaling", "splits"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
