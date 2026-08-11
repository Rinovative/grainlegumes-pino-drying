"""
Scientific plots for exploratory generated-data analysis.

Provides:
- case_statistics: case-level parameter and field-statistic plots
- spectral: isotropic, directional, and evolution spectrum plots
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import eda_plot_case_statistics as case_statistics
    from . import eda_plot_spectral_analysis as spectral

_MODULES = {
    "case_statistics": "eda_plot_case_statistics",
    "spectral": "eda_plot_spectral_analysis",
}
__all__ = ["case_statistics", "spectral"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
