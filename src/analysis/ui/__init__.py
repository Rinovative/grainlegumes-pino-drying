"""
Reusable notebook controls, viewers, and panel composition.

Provides:
- components: shared widget controls and display components
- notebook: notebook-panel construction utilities
- tables: styled analysis-table presentation
- viewers: interactive field and case viewers
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis_ui_components as components
    from . import analysis_ui_notebook as notebook
    from . import analysis_ui_tables as tables
    from . import analysis_ui_viewers as viewers

_MODULES = {
    "components": "analysis_ui_components",
    "notebook": "analysis_ui_notebook",
    "tables": "analysis_ui_tables",
    "viewers": "analysis_ui_viewers",
}
__all__ = ["components", "notebook", "tables", "viewers"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
