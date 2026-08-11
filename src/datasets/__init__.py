"""
Dataset responsibility packages and the canonical package-builder facade.

Provides:
- contracts: Dataset view, identity, metadata, and temporal contracts
- dataset_packages: canonical persisted package builder and supported CLI
- packages: source admission, package planning, and immutable publication
- preprocessing: identity-bound splitting and train-only normalization
- runtime: unified Dataset loading and DataLoader orchestration
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import contracts, dataset_packages, packages, preprocessing, runtime

_MODULES = {
    "contracts": "contracts",
    "dataset_packages": "dataset_packages",
    "packages": "packages",
    "preprocessing": "preprocessing",
    "runtime": "runtime",
}
__all__ = ["contracts", "dataset_packages", "packages", "preprocessing", "runtime"]


def __getattr__(name: str) -> object:
    """Resolve one declared public package or facade on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
