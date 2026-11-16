"""
Plot catalog of the plot-standard notebook: every plot sub-type drawn by its real builder on synthetic inputs.

Provides:
- checks: component-check, training-inspection and training-plot entries
- entries: the entry contract, the base types and the scratch folder of synthetic inputs
- evaluation: steady evaluation entries
- inputs: generation-input and Dataset EDA entries
- panel: entry collection, lookup and the catalog panel
- transient: transient evaluation and transient Dataset EDA entries

Imported as ``from src.analysis import catalog``; it is deliberately not listed in ``src.analysis``'s surface, because
the analysis code projections hash that initializer and a development catalog must not change the recorded identity of
evaluation artifacts.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis_catalog_checks as checks
    from . import analysis_catalog_entries as entries
    from . import analysis_catalog_evaluation as evaluation
    from . import analysis_catalog_inputs as inputs
    from . import analysis_catalog_panel as panel
    from . import analysis_catalog_transient as transient

_MODULES = {
    "checks": "analysis_catalog_checks",
    "entries": "analysis_catalog_entries",
    "evaluation": "analysis_catalog_evaluation",
    "inputs": "analysis_catalog_inputs",
    "panel": "analysis_catalog_panel",
    "transient": "analysis_catalog_transient",
}
__all__ = ["checks", "entries", "evaluation", "inputs", "panel", "transient"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
