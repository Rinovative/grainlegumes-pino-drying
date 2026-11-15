"""
Scientific presentation services.

Provides:
- channel_semantics: shared scientific display-channel ordering and compatibility
- dataset_selection: dataset items, inventories, and separate or pooled draw choices
- display_labels: concise task and dataset display labels
- field_labels: canonical user-facing scientific field labels
- field_maps: exact shared discrete scientific field-map scales
- histograms: shared exact constant and non-constant histogram rendering
- moisture_display: wet-basis percent display of water densities with masked unphysical cells
- registry: numbered EDA and evaluation presentation definitions
- visual_semantics: semantic colormaps, dataset colors, and fixed material, architecture, and role series styles
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import analysis_channel_semantics as channel_semantics
    from . import analysis_dataset_selection as dataset_selection
    from . import analysis_display_labels as display_labels
    from . import analysis_field_labels as field_labels
    from . import analysis_field_maps as field_maps
    from . import analysis_histograms as histograms
    from . import analysis_moisture_display as moisture_display
    from . import analysis_presentation_registry as registry
    from . import analysis_visual_semantics as visual_semantics

_MODULES = {
    "channel_semantics": "analysis_channel_semantics",
    "dataset_selection": "analysis_dataset_selection",
    "display_labels": "analysis_display_labels",
    "field_labels": "analysis_field_labels",
    "field_maps": "analysis_field_maps",
    "histograms": "analysis_histograms",
    "moisture_display": "analysis_moisture_display",
    "registry": "analysis_presentation_registry",
    "visual_semantics": "analysis_visual_semantics",
}
__all__ = [
    "channel_semantics",
    "dataset_selection",
    "display_labels",
    "field_labels",
    "field_maps",
    "histograms",
    "moisture_display",
    "registry",
    "visual_semantics",
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
