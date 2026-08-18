"""
generation_input_presentation.py

Declare the compact generation-input EDA sections and views.
Responsibilities:
  - Keep view keys, labels, and order explicit in one immutable registry
  - Omit transient schedules and moisture views when unavailable
  - Delegate display numbering to the shared presentation contract
Design principles:
  - Every case-level view uses the common A/B selector model
  - Presentation order never participates in scientific or persisted identity
  - Dataset overview remains separate from case-level comparison
This module does NOT:
  - Construct widgets, discover datasets, or implement calculations
  - Modify completed-output EDA or evaluation registries
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.analysis.presentation import registry as shared_registry
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Iterable

_TRANSIENT_ONLY_VIEW_KEYS: Final = frozenset(
    {
        "transient_schedules",
        "moisture_sorption",
    }
)

GENERATION_INPUT_SECTIONS: Final = (
    shared_registry.SectionPresentation(
        key="case_comparison",
        name="Case comparison",
        plots=(
            shared_registry.PlotPresentation(
                "case_context",
                "Case context",
            ),
            shared_registry.PlotPresentation(
                "parameters",
                "Parameters",
            ),
            shared_registry.PlotPresentation(
                "field_summaries",
                "Field summaries",
            ),
        ),
    ),
    shared_registry.SectionPresentation(
        key="boundary_schedule_comparison",
        name="Boundary and schedule comparison",
        plots=(
            shared_registry.PlotPresentation(
                "boundary_conditions",
                "Boundary conditions",
            ),
            shared_registry.PlotPresentation(
                "transient_schedules",
                "Transient schedules",
            ),
        ),
    ),
    shared_registry.SectionPresentation(
        key="spatial_comparison",
        name="Spatial comparison",
        plots=(
            shared_registry.PlotPresentation(
                "basic_spatial_inputs",
                "Basic spatial inputs",
            ),
            shared_registry.PlotPresentation(
                "permeability_tensor",
                "Permeability tensor",
            ),
            shared_registry.PlotPresentation(
                "derived_permeability",
                "Derived permeability",
            ),
            shared_registry.PlotPresentation(
                "moisture_sorption",
                "Moisture and sorption",
            ),
        ),
    ),
    shared_registry.SectionPresentation(
        key="dataset_overview",
        name="Dataset overview",
        plots=(
            shared_registry.PlotPresentation(
                "dataset_cases_parameters",
                "Dataset cases and parameters",
            ),
            shared_registry.PlotPresentation(
                "dataset_field_summaries",
                "Dataset field summaries",
            ),
        ),
    ),
)


def sections_for_profiles(
    profile_ids: Iterable[str],
) -> tuple[shared_registry.SectionPresentation, ...]:
    """
    Return the ordered views supported by available maintained profiles.

    Transient schedules and moisture/sorption are the only capability-gated
    views. All other sections preserve the same scientific responsibilities.
    """
    available = frozenset(profile_ids)
    if not available:
        msg = "Generation-input presentation requires at least one profile."
        raise ValueError(msg)
    for profile_id in available:
        profiles.resolve_profile(profile_id)
    has_transient = profiles.TRANSIENT_DRYING_PROFILE in available
    sections = tuple(
        shared_registry.SectionPresentation(
            key=section.key,
            name=section.name,
            plots=tuple(plot for plot in section.plots if has_transient or plot.key not in _TRANSIENT_ONLY_VIEW_KEYS),
        )
        for section in GENERATION_INPUT_SECTIONS
    )
    shared_registry.validate_registry(sections)
    return sections


shared_registry.validate_registry(GENERATION_INPUT_SECTIONS)
