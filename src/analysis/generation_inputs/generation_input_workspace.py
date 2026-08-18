"""
generation_input_workspace.py

Prepare the read-only generation-input notebook workspace.
Responsibilities:
  - Discover admitted canonical input datasets for notebook presentation
  - Summarize datasets, discovery issues, and canonical naming conventions
  - Establish deterministic session selection and construct the EDA panel
  - Provide actionable plain guidance when no canonical dataset is admitted
Design principles:
  - Existing discovery, label, selection, and panel owners remain authoritative
  - Workspace preparation reads canonical storage without mutating it
  - Plain summary output remains separate from the interactive panel
This module does NOT:
  - Generate, plan, reuse, or publish canonical input cases
  - Implement scientific diagnostics, plots, or panel internals
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from . import generation_input_labels as labels
from . import generation_input_panel as panel
from . import generation_input_selection as selection
from . import generation_input_sources as sources

if TYPE_CHECKING:
    import ipywidgets as widgets

_GENERATION_DOCUMENTATION_LINK: Final = "../docs/simulation_generation.md#canonical-input-case-generation-and-eda"
_GENERIC_GENERATION_COMMAND: Final = (
    "./scripts/docker_python.sh -m src.generation.cli.cli_generation "
    'generate-input-cases "$CAMPAIGN_CONFIG" '
    '--only-batch "$BATCH_NAME" --case-start 1 --case-count "$CASE_COUNT" '
    '--git-commit "$(git rev-parse HEAD)" --storage-root "$STORAGE_ROOT"'
)
_MAXIMUM_DISPLAYED_ISSUES: Final = 5


@dataclass(frozen=True, slots=True)
class GenerationInputEDAWorkspace:
    """Hold plain notebook guidance and the separately displayed EDA panel."""

    summary_text: str
    panel: widgets.Output | None


def _dataset_summary(
    catalog: sources.GenerationInputDatasetCatalog,
    *,
    storage_root: Path,
) -> str:
    """Build the plain admitted-catalog summary through canonical label owners."""
    display_labels = {key: label for label, key in catalog.dataset_options()}
    dataset_rows = "\n".join(f"- {display_labels[sources.dataset_key(dataset)]}: {len(dataset.cases)} cases" for dataset in catalog.datasets)
    return (
        "Canonical input summary\n\n"
        f"Storage root: {storage_root}\n"
        f"Canonical input datasets: {len(catalog.datasets)}\n"
        f"Manifested input cases: {sum(len(dataset.cases) for dataset in catalog.datasets)}\n"
        f"Skipped invalid input batches: {len(catalog.issues)}\n\n"
        f"Datasets:\n{dataset_rows}"
    )


def _naming_summary(
    catalog: sources.GenerationInputDatasetCatalog,
) -> str:
    """Build plain purpose and profile explanations through authoritative helpers."""
    purpose_rows = labels.campaign_purpose_legend_rows(dataset.campaign_purpose for dataset in catalog.datasets)
    purpose_body = "\n".join(f"- {row.abbreviation} = {row.campaign_purpose}" for row in purpose_rows)
    profile_body = "\n".join(f"- {label} = {profile_id}" for label, profile_id in labels.profile_label_rows(catalog.profiles))
    return f"Campaign-purpose abbreviations:\n{purpose_body}\n\nProfile labels:\n{profile_body}"


def _issue_details(
    catalog: sources.GenerationInputDatasetCatalog,
) -> str:
    """Return bounded plain-text discovery issue details."""
    if not catalog.issues:
        return ""
    details = "\n".join(f"- {issue.source_id}: {issue.message}" for issue in catalog.issues[:_MAXIMUM_DISPLAYED_ISSUES])
    omitted = len(catalog.issues) - _MAXIMUM_DISPLAYED_ISSUES
    if omitted > 0:
        details += f"\n- {omitted} additional issues omitted."
    return f"Skipped batch details:\n{details}"


def _empty_catalog_workspace(
    catalog: sources.GenerationInputDatasetCatalog,
    *,
    storage_root: Path,
) -> GenerationInputEDAWorkspace:
    """Build actionable, read-only guidance for an empty admitted catalog."""
    details = _issue_details(catalog)
    summary = (
        "Canonical input summary\n\n"
        f"Storage root: {storage_root}\n"
        "Canonical input datasets: 0\n"
        "Manifested input cases: 0\n"
        f"Skipped invalid input batches: {len(catalog.issues)}\n\n"
        "No manifested canonical input cases were admitted.\n"
        "Generate canonical input cases through generate-input-cases, then rerun this notebook.\n"
        f"See {_GENERATION_DOCUMENTATION_LINK}.\n\n"
        f"{_GENERIC_GENERATION_COMMAND}"
    )
    if details:
        summary = f"{summary}\n\n{details}"
    return GenerationInputEDAWorkspace(summary_text=summary, panel=None)


def prepare_generation_input_eda_workspace(
    *,
    storage_root: Path | str,
    title: str = "Generation-input EDA",
) -> GenerationInputEDAWorkspace:
    """
    Prepare one read-only workspace over admitted canonical generation inputs.

    Parameters
    ----------
    storage_root : Path | str
        Canonical storage root containing already generated input cases.
    title : str, optional
        Visible title for the collapsible interactive panel.

    Returns
    -------
    GenerationInputEDAWorkspace
        Plain summary text and a separately displayable interactive panel. The
        panel is absent when the admitted catalog is empty.

    Notes
    -----
    Preparation performs discovery and admission reads only. It never plans,
    generates, reuses through mutation, or publishes canonical input cases.

    """
    root = Path(storage_root).expanduser()
    if not isinstance(title, str) or not title.strip():
        message = "title must be a non-empty string."
        raise ValueError(message)
    catalog = sources.discover_generation_input_datasets(storage_root=root)
    if not catalog.datasets:
        return _empty_catalog_workspace(catalog, storage_root=root)
    selection_state = selection.GenerationInputSelectionState(catalog)
    interactive_panel = panel.build_generation_input_eda_panel(
        datasets=catalog,
        selection_state=selection_state,
        title=title,
    )
    summary_parts = (
        _dataset_summary(catalog, storage_root=root),
        _naming_summary(catalog),
        _issue_details(catalog),
    )
    return GenerationInputEDAWorkspace(
        summary_text="\n\n".join(part for part in summary_parts if part),
        panel=interactive_panel,
    )
