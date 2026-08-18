"""
Generation pilot analysis and deterministic scientific validation services.

Provides:
- policy: shared blocking-integrity and advisory-diagnostic severity contract
- pilot: transient pilot planning and terminal validation
- pilot_analysis: convergence, stability, and stopping diagnostics
- sentinels: deterministic no-COMSOL scientific integration checks
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_validation_pilot as pilot
    from . import generation_validation_pilot_analysis as pilot_analysis
    from . import generation_validation_policy as policy
    from . import generation_validation_sentinels as sentinels

_MODULES = {
    "policy": "generation_validation_policy",
    "pilot": "generation_validation_pilot",
    "pilot_analysis": "generation_validation_pilot_analysis",
    "sentinels": "generation_validation_sentinels",
}
__all__ = ["pilot", "pilot_analysis", "policy", "sentinels"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
