"""
Physics diagnostics of published reference fields.

Provides:
- plots: static figures of residual, derivative, sensitivity, and balance tables
- sources: read-only steady and transient reference-case loading
- steady: Darcy-Brinkman residual variants, backend comparison, and sensitivity
- transient: water-mass and sorption balance closures and persistence gaps
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import physics_diagnostics_plots as plots
    from . import physics_diagnostics_sources as sources
    from . import physics_diagnostics_steady as steady
    from . import physics_diagnostics_transient as transient

_MODULES = {
    "plots": "physics_diagnostics_plots",
    "sources": "physics_diagnostics_sources",
    "steady": "physics_diagnostics_steady",
    "transient": "physics_diagnostics_transient",
}
__all__ = ["plots", "sources", "steady", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
