"""
===============================================================================
generation_input_plot_permeability.py
===============================================================================
Compose tensor and derived-permeability A/B spatial comparisons.
Responsibilities:
  - Group Kxx, Kxy, and Kyy as tensor-component rows
  - Group per-case principal permeabilities and anisotropy as derived rows
  - Reuse the common independent-or-locked A/B/B-minus-A map contract
Design principles:
  - Kxy remains signed and zero-centred in raw and difference maps
  - Nonlinear eigenvalue diagnostics are derived per case before aggregation
  - Every component retains its physical unit and distribution
This module does NOT:
  - Render single-case alternatives or recompute permeability diagnostics
  - Define tensor validity, resample grids, or load input files
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from . import generation_input_plot_spatial as spatial

if TYPE_CHECKING:
    import ipywidgets as widgets
    from matplotlib.figure import Figure

    from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics

TENSOR_FIELDS: Final = ("Kxx", "Kxy", "Kyy")
DERIVED_FIELDS: Final = ("K_min", "K_max", "K_anisotropy")


def tensor_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    lock_scale: bool,
) -> Figure | widgets.HTML:
    """Render all permeability tensor components as A/B/B-minus-A rows."""
    return spatial.comparison_block(
        first,
        mean_a,
        second,
        mean_b,
        TENSOR_FIELDS,
        title="Permeability tensor",
        lock_scale=lock_scale,
        include_distributions=True,
    )


def derived_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    lock_scale: bool,
) -> Figure | widgets.HTML:
    """Render principal permeability and anisotropy as A/B/B-minus-A rows."""
    return spatial.comparison_block(
        first,
        mean_a,
        second,
        mean_b,
        DERIVED_FIELDS,
        title="Derived permeability",
        lock_scale=lock_scale,
        include_distributions=True,
    )
