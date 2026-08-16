"""
Generation-input discovery, diagnostics, presentation, and notebook views.

Provides:
- diagnostics: immutable scientific diagnostics over admitted generation inputs
- labels: canonical-metadata-driven compact dataset presentation labels
- panel: lazy generation-input EDA panel with view-local controls
- plots: generation-input scientific tables and figures
- presentation: ordered generation-input section and view registry
- selection: shared canonical Dataset A/B and Case A/B session state
- sources: manifest-driven input-batch grouping and lazy admission
- workspace: typed plain-summary and separate-panel notebook preparation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_input_diagnostics as diagnostics
    from . import generation_input_labels as labels
    from . import generation_input_panel as panel
    from . import generation_input_presentation as presentation
    from . import generation_input_selection as selection
    from . import generation_input_sources as sources
    from . import generation_input_workspace as workspace
    from . import plots

_MODULES = {
    "diagnostics": "generation_input_diagnostics",
    "labels": "generation_input_labels",
    "panel": "generation_input_panel",
    "plots": "plots",
    "presentation": "generation_input_presentation",
    "selection": "generation_input_selection",
    "sources": "generation_input_sources",
    "workspace": "generation_input_workspace",
}
__all__ = [
    "diagnostics",
    "labels",
    "panel",
    "plots",
    "presentation",
    "selection",
    "sources",
    "workspace",
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
