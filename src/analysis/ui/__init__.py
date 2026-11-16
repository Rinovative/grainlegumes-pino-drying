"""
Reusable notebook controls, viewers, and panel composition.

Provides:
- catalog: synthetic catalog of the control standard for the plot-standard notebook
- controls: primitive controls of the control standard, sized on the column grid
- figure_layout: panel grids, text sizes, legends, row labels and titles of line, bar, and distribution figures
- navigation: case, time, and count navigators
- notebook: notebook-panel construction utilities
- panel: control rows, view panels, popovers, on-demand views, and tab sets
- plot_layout: shared physical-map and colorbar presentation
- selectors: run and dataset selectors with grouped, collapsible choices
- style: control-standard geometry, column grid, and scoped stylesheet
- tables: styled analysis-table presentation
- time: validated stored-time coordinates, master-time resolution, and physical-time axis formatting
- viewers: case-count viewers and the render, clear, and failure publication of view panels
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis_ui_catalog as catalog
    from . import analysis_ui_controls as controls
    from . import analysis_ui_figure_layout as figure_layout
    from . import analysis_ui_navigation as navigation
    from . import analysis_ui_notebook as notebook
    from . import analysis_ui_panel as panel
    from . import analysis_ui_plot_layout as plot_layout
    from . import analysis_ui_selectors as selectors
    from . import analysis_ui_style as style
    from . import analysis_ui_tables as tables
    from . import analysis_ui_time as time
    from . import analysis_ui_viewers as viewers

_MODULES = {
    "catalog": "analysis_ui_catalog",
    "controls": "analysis_ui_controls",
    "figure_layout": "analysis_ui_figure_layout",
    "navigation": "analysis_ui_navigation",
    "notebook": "analysis_ui_notebook",
    "panel": "analysis_ui_panel",
    "plot_layout": "analysis_ui_plot_layout",
    "selectors": "analysis_ui_selectors",
    "style": "analysis_ui_style",
    "tables": "analysis_ui_tables",
    "time": "analysis_ui_time",
    "viewers": "analysis_ui_viewers",
}
__all__ = [
    "catalog",
    "controls",
    "figure_layout",
    "navigation",
    "notebook",
    "panel",
    "plot_layout",
    "selectors",
    "style",
    "tables",
    "time",
    "viewers",
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
