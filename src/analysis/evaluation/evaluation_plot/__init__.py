"""
Scientific plots for model-evaluation sections.

Provides:
- error_behavior: predictive-error distributions, maps, and decompositions
- layout: title, case-count and unavailable-panel helpers of evaluation figures and the text of their map grids
- physical_consistency: residual and boundary-consistency plots
- run_summary: completed-run overview plots
- samples_outliers: sample, outlier, and extreme-case plots
- sensitivity_capacity: capacity and metadata-sensitivity plots
- series: fixed series styles of entities and the grouping of steady frames into panels
- spectral_fidelity: prediction-spectrum fidelity plots
- training: training curves, study progress, learning curves, and the break-even case count
- transient: sequence, process, matched-compute, and timing plots
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import evaluation_plot_error_behavior as error_behavior
    from . import evaluation_plot_layout as layout
    from . import evaluation_plot_physical_consistency as physical_consistency
    from . import evaluation_plot_run_summary as run_summary
    from . import evaluation_plot_samples_outliers as samples_outliers
    from . import evaluation_plot_sensitivity_capacity as sensitivity_capacity
    from . import evaluation_plot_series as series
    from . import evaluation_plot_spectral_fidelity as spectral_fidelity
    from . import evaluation_plot_training as training
    from . import evaluation_plot_transient as transient

_MODULES = {
    "error_behavior": "evaluation_plot_error_behavior",
    "layout": "evaluation_plot_layout",
    "physical_consistency": "evaluation_plot_physical_consistency",
    "run_summary": "evaluation_plot_run_summary",
    "samples_outliers": "evaluation_plot_samples_outliers",
    "sensitivity_capacity": "evaluation_plot_sensitivity_capacity",
    "series": "evaluation_plot_series",
    "spectral_fidelity": "evaluation_plot_spectral_fidelity",
    "training": "evaluation_plot_training",
    "transient": "evaluation_plot_transient",
}
__all__ = [
    "error_behavior",
    "layout",
    "physical_consistency",
    "run_summary",
    "samples_outliers",
    "sensitivity_capacity",
    "series",
    "spectral_fidelity",
    "training",
    "transient",
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
