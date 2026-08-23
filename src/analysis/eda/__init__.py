"""
Exploratory analysis of admitted generated batches.

Provides:
- capabilities: authoritative field and view capability resolution
- controls: shared dataset, scope, case, channel, and scale widget bindings
- dataframe: generated-batch materialization for analysis
- panel: adaptive exploratory-analysis notebook panel
- plots: scientific generated-data visualizations
- selection: lazy profile-native views and shared selection state
- transient: completed-output transient scientific summaries and selectors
- sources: read-only generated-output campaign and case discovery
- viewers: shared-state live EDA viewers
- workspace: thin generated-output EDA notebook orchestration
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import eda_capabilities as capabilities
    from . import eda_controls as controls
    from . import eda_dataframe as dataframe
    from . import eda_panel as panel
    from . import eda_selection as selection
    from . import eda_sources as sources
    from . import eda_transient as transient
    from . import eda_viewers as viewers
    from . import eda_workspace as workspace
    from . import plots

_MODULES = {
    "capabilities": "eda_capabilities",
    "controls": "eda_controls",
    "dataframe": "eda_dataframe",
    "panel": "eda_panel",
    "plots": "plots",
    "selection": "eda_selection",
    "sources": "eda_sources",
    "transient": "eda_transient",
    "viewers": "eda_viewers",
    "workspace": "eda_workspace",
}
__all__ = [
    "capabilities",
    "controls",
    "dataframe",
    "panel",
    "plots",
    "selection",
    "sources",
    "transient",
    "viewers",
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
