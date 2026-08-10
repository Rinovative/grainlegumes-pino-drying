"""
Dataset contracts, package builders, and the unified dual-view runtime factory.

Provides:
- factory: authoritative dataset, selector, and DataLoader construction
- generated_batch: strict generated-simulation batch admission
- identity: deterministic steady dataset and membership identities
- metadata: dataset metadata admission and summaries
- normalization: fitted preprocessing reconstruction and identity binding
- packages: campaign-owned dual-view package assembly and inspection
- splits: identity-bound split admission and role evidence
- steady: verified steady-flow tensor loading and selection
- training: deterministic split, preprocessing, and DataLoader orchestration
- transient: lazy physical-unit transient transition indexes
- transient_contract: canonical transient step-data channels
- views: typed buildable dataset-view registry
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_factory as factory
    from . import dataset_generated_batch as generated_batch
    from . import dataset_identity as identity
    from . import dataset_metadata as metadata
    from . import dataset_normalization as normalization
    from . import dataset_packages as packages
    from . import dataset_splits as splits
    from . import dataset_steady as steady
    from . import dataset_training as training
    from . import dataset_transient as transient
    from . import dataset_transient_contract as transient_contract
    from . import dataset_views as views

_MODULES = {
    "factory": "dataset_factory",
    "generated_batch": "dataset_generated_batch",
    "identity": "dataset_identity",
    "metadata": "dataset_metadata",
    "normalization": "dataset_normalization",
    "packages": "dataset_packages",
    "splits": "dataset_splits",
    "steady": "dataset_steady",
    "training": "dataset_training",
    "transient": "dataset_transient",
    "transient_contract": "dataset_transient_contract",
    "views": "dataset_views",
}
__all__ = [
    "factory",
    "generated_batch",
    "identity",
    "metadata",
    "normalization",
    "packages",
    "splits",
    "steady",
    "training",
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
