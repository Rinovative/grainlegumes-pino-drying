"""
Reference-simulation configuration, execution, validation, and publication API.

Provides:
- campaign_runtime: campaign planning, execution evidence, and finalization
- case: deterministic scientific case-input generation
- config: resolved campaign and batch configuration
- contracts: immutable cross-package profile and vocabulary descriptors
- pilot: transient pilot preparation, analysis, and terminal evidence
- readiness: fail-closed production-readiness reporting
- runtime: single-case execution and atomic terminal publication
- sentinels: deterministic no-COMSOL scientific integration checks
- smoke: paired technical-runtime smoke evidence
- storage: canonical HDF5 conversion and admission
- workflow: transfer, package, retention, and cleanup lifecycle
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_campaign_runtime as campaign_runtime
    from . import generation_case as case
    from . import generation_config as config
    from . import generation_contracts as contracts
    from . import generation_pilot as pilot
    from . import generation_readiness as readiness
    from . import generation_runtime as runtime
    from . import generation_sentinels as sentinels
    from . import generation_smoke as smoke
    from . import generation_storage as storage
    from . import generation_workflow as workflow

_MODULES = {
    "campaign_runtime": "generation_campaign_runtime",
    "case": "generation_case",
    "config": "generation_config",
    "contracts": "generation_contracts",
    "pilot": "generation_pilot",
    "readiness": "generation_readiness",
    "runtime": "generation_runtime",
    "sentinels": "generation_sentinels",
    "smoke": "generation_smoke",
    "storage": "generation_storage",
    "workflow": "generation_workflow",
}
__all__ = [
    "campaign_runtime",
    "case",
    "config",
    "contracts",
    "pilot",
    "readiness",
    "runtime",
    "sentinels",
    "smoke",
    "storage",
    "workflow",
]


def __getattr__(name: str) -> object:
    """Resolve one declared public service on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
