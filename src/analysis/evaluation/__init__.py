"""
Evaluation of persisted model predictions and scientific evidence.

Provides:
- artifact_loader: strict path-based evaluable-run artifact access
- case: case-level prediction artifact access
- dataframe: aggregate evaluation-table construction
- panel: interactive evaluation notebook panel
- presentation: current-native field, grid, and metadata presentation data
- plots: scientific evaluation visualizations
- run_discovery: read-only persisted-run and artifact discovery
- selection: synchronized task-aware notebook selection state
- session: bounded steady evaluation-case and numerical-summary reuse
- transient_artifact: strict sequence-artifact persistence and admission
- transient_comparison: matched-compute and Airflow-to-Drying comparison contracts
- transient_metrics: sequence-aware Drying metrics and diagnostics
- transient_rollout: teacher-forced, autonomous, and rolling-origin inference
- transient_session: bounded sequence reductions and tracking summaries
- transient_timing: admitted runtime components and speedup formulas
- workflow: task-aware run preparation, reporting, and panel composition
- workspace: automatic EDA-aligned notebook entry points
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import evaluation_artifact_loader as artifact_loader
    from . import evaluation_case as case
    from . import evaluation_dataframe as dataframe
    from . import evaluation_panel as panel
    from . import evaluation_plot as plots
    from . import evaluation_presentation as presentation
    from . import evaluation_run_discovery as run_discovery
    from . import evaluation_selection as selection
    from . import evaluation_session as session
    from . import evaluation_transient_artifact as transient_artifact
    from . import evaluation_transient_comparison as transient_comparison
    from . import evaluation_transient_metrics as transient_metrics
    from . import evaluation_transient_rollout as transient_rollout
    from . import evaluation_transient_session as transient_session
    from . import evaluation_transient_timing as transient_timing
    from . import evaluation_workflow as workflow
    from . import evaluation_workspace as workspace

_MODULES = {
    "artifact_loader": "evaluation_artifact_loader",
    "case": "evaluation_case",
    "dataframe": "evaluation_dataframe",
    "panel": "evaluation_panel",
    "plots": "evaluation_plot",
    "presentation": "evaluation_presentation",
    "run_discovery": "evaluation_run_discovery",
    "selection": "evaluation_selection",
    "session": "evaluation_session",
    "transient_artifact": "evaluation_transient_artifact",
    "transient_comparison": "evaluation_transient_comparison",
    "transient_metrics": "evaluation_transient_metrics",
    "transient_rollout": "evaluation_transient_rollout",
    "transient_session": "evaluation_transient_session",
    "transient_timing": "evaluation_transient_timing",
    "workflow": "evaluation_workflow",
    "workspace": "evaluation_workspace",
}
__all__ = [
    "artifact_loader",
    "case",
    "dataframe",
    "panel",
    "plots",
    "presentation",
    "run_discovery",
    "selection",
    "session",
    "transient_artifact",
    "transient_comparison",
    "transient_metrics",
    "transient_rollout",
    "transient_session",
    "transient_timing",
    "workflow",
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
