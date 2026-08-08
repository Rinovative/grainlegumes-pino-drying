"""
Dataset contracts, metadata, loaders, and simulation readers.

Provides:
- base: shared dataset interfaces
- build: final dataset and metadata publication
- generated_batch: strict generated-simulation batch admission
- identity: deterministic dataset identity and fingerprint contracts
- metadata: dataset metadata admission and summary services
- modules: model-ready task dataset modules
- packages: campaign-owned multi-batch dataset assembly
- simulation: persisted simulation-dataset access
- transient: physical-unit one-hour transition builder and loader
- transient_contract: unregistered transient step-data schema
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset_base as base
    from . import dataset_build as build
    from . import dataset_generated_batch as generated_batch
    from . import dataset_identity as identity
    from . import dataset_metadata as metadata
    from . import dataset_modules as modules
    from . import dataset_packages as packages
    from . import dataset_simulation as simulation
    from . import dataset_transient as transient
    from . import dataset_transient_contract as transient_contract

_MODULES = {
    "base": "dataset_base",
    "build": "dataset_build",
    "generated_batch": "dataset_generated_batch",
    "identity": "dataset_identity",
    "metadata": "dataset_metadata",
    "modules": "dataset_modules",
    "packages": "dataset_packages",
    "simulation": "dataset_simulation",
    "transient": "dataset_transient",
    "transient_contract": "dataset_transient_contract",
}
__all__ = ["base", "build", "generated_batch", "identity", "metadata", "modules", "packages", "simulation", "transient", "transient_contract"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
