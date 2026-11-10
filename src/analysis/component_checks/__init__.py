"""
Training-free component checks on published reference fields.

Provides:
- increments: Train-fitted scaled-increment statistics, Huber regime, and stored time steps
- plots: static figures of residual, derivative, sensitivity, balance, and mode tables
- sources: read-only membership-filtered steady and transient reference-case loading
- spectral: Fourier-mode truncation floors with domain padding and spatial-stride floors
- steady: Darcy-Brinkman residual variants, backend comparison, crops, and sensitivity
- summary: proposed component criteria applied to the measured tables
- timing: physics-loss cost per derivative backend and device
- transient: water-mass and sorption balance closures and persistence gaps
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import component_checks_increments as increments
    from . import component_checks_plots as plots
    from . import component_checks_sources as sources
    from . import component_checks_spectral as spectral
    from . import component_checks_steady as steady
    from . import component_checks_summary as summary
    from . import component_checks_timing as timing
    from . import component_checks_transient as transient

_MODULES = {
    "increments": "component_checks_increments",
    "plots": "component_checks_plots",
    "sources": "component_checks_sources",
    "spectral": "component_checks_spectral",
    "steady": "component_checks_steady",
    "summary": "component_checks_summary",
    "timing": "component_checks_timing",
    "transient": "component_checks_transient",
}
__all__ = ["increments", "plots", "sources", "spectral", "steady", "summary", "timing", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
