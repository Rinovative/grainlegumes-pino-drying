"""
Dataset contracts, package builders, and the unified dual-view runtime factory.

Provides:
- base: steady experiment splitting and train-only preprocessing
- factory: authoritative dataset, selector, and DataLoader construction
- generated_batch: strict generated-simulation batch admission
- identity: deterministic steady dataset and membership identities
- metadata: dataset metadata admission and summaries
- packages: campaign-owned dual-view package assembly and inspection
- transient: lazy physical-unit transient transition indexes
- transient_contract: canonical transient step-data channels
- views: typed buildable dataset-view registry
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_base as base
    from . import dataset_factory as factory
    from . import dataset_generated_batch as generated_batch
    from . import dataset_identity as identity
    from . import dataset_metadata as metadata
    from . import dataset_packages as packages
    from . import dataset_transient as transient
    from . import dataset_transient_contract as transient_contract
    from . import dataset_views as views

_MODULES = {
    "base": "dataset_base",
    "factory": "dataset_factory",
    "generated_batch": "dataset_generated_batch",
    "identity": "dataset_identity",
    "metadata": "dataset_metadata",
    "packages": "dataset_packages",
    "transient": "dataset_transient",
    "transient_contract": "dataset_transient_contract",
    "views": "dataset_views",
}
__all__ = [
    "base",
    "factory",
    "generated_batch",
    "identity",
    "metadata",
    "packages",
    "transient",
    "transient_contract",
    "views",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
