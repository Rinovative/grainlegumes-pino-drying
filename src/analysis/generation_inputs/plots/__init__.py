"""
Scientific tables and figures for generation-input exploratory analysis.

Provides:
- boundaries: boundary tables and exact transient schedule comparisons
- layout: canonical map, normalization, and colorbar geometry
- moisture: transient moisture, sorption, and inlet-bed diagnostics
- overview: A/B comparison and raw dataset overview tables
- permeability: tensor, principal-value, and anisotropy diagnostics
- spatial: A/B/B-minus-A map blocks and retained distributions
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_input_plot_boundaries as boundaries
    from . import generation_input_plot_layout as layout
    from . import generation_input_plot_moisture as moisture
    from . import generation_input_plot_overview as overview
    from . import generation_input_plot_permeability as permeability
    from . import generation_input_plot_spatial as spatial

_MODULES = {
    "boundaries": "generation_input_plot_boundaries",
    "layout": "generation_input_plot_layout",
    "moisture": "generation_input_plot_moisture",
    "overview": "generation_input_plot_overview",
    "permeability": "generation_input_plot_permeability",
    "spatial": "generation_input_plot_spatial",
}
__all__ = [
    "boundaries",
    "layout",
    "moisture",
    "overview",
    "permeability",
    "spatial",
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
