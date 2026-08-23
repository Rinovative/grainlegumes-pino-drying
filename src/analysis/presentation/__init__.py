"""
Scientific presentation services.

Provides:
- channel_semantics: shared scientific display-channel ordering and compatibility
- curated: fixed local scientific media rendering
- display_labels: concise task and dataset display labels
- field_labels: canonical user-facing scientific field labels
- histograms: shared exact constant and non-constant histogram rendering
- registry: numbered EDA and evaluation presentation definitions
- visual_semantics: semantic colormaps and deterministic dataset colors
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis_channel_semantics as channel_semantics
    from . import analysis_display_labels as display_labels
    from . import analysis_field_labels as field_labels
    from . import analysis_histograms as histograms
    from . import analysis_presentation_curated as curated
    from . import analysis_presentation_registry as registry
    from . import analysis_visual_semantics as visual_semantics

_MODULES = {
    "channel_semantics": "analysis_channel_semantics",
    "curated": "analysis_presentation_curated",
    "display_labels": "analysis_display_labels",
    "field_labels": "analysis_field_labels",
    "histograms": "analysis_histograms",
    "registry": "analysis_presentation_registry",
    "visual_semantics": "analysis_visual_semantics",
}
__all__ = ["channel_semantics", "curated", "display_labels", "field_labels", "histograms", "registry", "visual_semantics"]


def __getattr__(name: str) -> object:
    """Resolve one declared public name on first access."""
    module_name = _MODULES.get(name)
    if module_name is None:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
