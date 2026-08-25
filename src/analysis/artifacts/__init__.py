"""
Artifact generation, validation, and runtime evidence.

Provides:
- contracts: artifact schemas, identities, and provenance validation
- generation: deterministic Parquet and NPZ artifact production
- service: artifact lifecycle, cache, rebuilding, and publication
- timing: COMSOL and neural runtime evidence
- transient: sequence-aware transient artifact planning and generation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis_artifact_contracts as contracts
    from . import analysis_artifact_generation as generation
    from . import analysis_artifact_service as service
    from . import analysis_artifact_timing as timing
    from . import analysis_artifact_transient as transient

_MODULES = {
    "contracts": "analysis_artifact_contracts",
    "generation": "analysis_artifact_generation",
    "service": "analysis_artifact_service",
    "timing": "analysis_artifact_timing",
    "transient": "analysis_artifact_transient",
}
__all__ = ["contracts", "generation", "service", "timing", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
