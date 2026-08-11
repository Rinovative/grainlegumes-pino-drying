"""
Unified steady and transient Dataset loading and DataLoader orchestration.

Provides:
- factory: package request, Dataset, and DataLoader construction
- package_validation: published-package inspection and loader smoke validation
- steady: materialized steady-flow Dataset runtime
- training: split and training DataLoader orchestration
- transient: lazy physical-unit transient Dataset runtime
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_runtime_factory as factory
    from . import dataset_runtime_package_validation as package_validation
    from . import dataset_runtime_steady as steady
    from . import dataset_runtime_training as training
    from . import dataset_runtime_transient as transient

_MODULES = {
    "factory": "dataset_runtime_factory",
    "package_validation": "dataset_runtime_package_validation",
    "steady": "dataset_runtime_steady",
    "training": "dataset_runtime_training",
    "transient": "dataset_runtime_transient",
}
__all__ = ["factory", "package_validation", "steady", "training", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
