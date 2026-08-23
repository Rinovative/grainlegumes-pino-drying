"""
eda_workspace.py

Prepare the read-only unified generated-output EDA notebook workspace.

Responsibilities:
  - Discover independently admitted complete and partial Generation outputs once
  - Project each admitted batch into one lightweight profile-native view
  - Bind lazy dataframe loaders, shared selection, summary, and one outer panel
  - Report terminal, partial, failed, incomplete, and invalid source accounting

Design principles:
  - Source discovery and case admission remain independent of Dataset publication
  - Scientific payloads load only when an active view requests selected data
  - Simulation-profile metadata drives scientific dispatch without a user task filter

This module does NOT:
  - Finalize Generation batches or publish Dataset packages
  - Define EDA scientific calculations, plots, or widget callbacks
  - Reconstruct campaign evidence or relax case validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src import domain
from src.analysis.presentation import analysis_display_labels as display_labels

from . import eda_dataframe as dataframe
from . import eda_panel as panel
from . import eda_selection as selection
from . import eda_sources as sources

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import ipywidgets as widgets
    import pandas as pd

_MAXIMUM_DISPLAYED_ISSUES: Final = 5


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDAWorkspace:
    """Hold guidance, lazy view state, and the separately displayable panel."""

    summary_text: str
    panel: widgets.Widget | None
    catalog: selection.GeneratedOutputEDACatalog
    selection_state: selection.GeneratedOutputSelectionState | None


def _batch_label_metadata(batch: sources.GeneratedOutputEDABatch) -> display_labels.DatasetDisplayMetadata:
    """Project one admitted batch into shared presentation metadata."""
    return display_labels.DatasetDisplayMetadata(
        task_id=batch.simulation_profile,
        material_family=batch.material_family,
        sampling_regime=batch.sampling_regime,
        campaign_purpose=batch.campaign_purpose,
        source_role=batch.material_role,
        evaluation_regime=batch.evaluation_regime,
        canonical_identity=batch.batch_id,
    )


def _batch_labels(batches: Sequence[sources.GeneratedOutputEDABatch]) -> dict[str, str]:
    """Return collision-safe concise labels keyed by canonical batch identity."""
    selected = tuple(batches)
    labels = display_labels.dataset_display_labels(_batch_label_metadata(batch) for batch in selected)
    return {batch.batch_id: label for batch, label in zip(selected, labels, strict=True)}


def _issues_text(catalog: sources.GeneratedOutputEDACatalog) -> str:
    """Return bounded source-admission issue details."""
    if not catalog.issues:
        return ""
    rows = [f"- {issue.source_id}: {issue.message}" for issue in catalog.issues[:_MAXIMUM_DISPLAYED_ISSUES]]
    omitted = catalog.total_issue_count - len(rows)
    if omitted:
        rows.append(f"- {omitted} additional source issues omitted.")
    return "Skipped source details:\n" + "\n".join(rows)


def _summary(catalog: selection.GeneratedOutputEDACatalog, *, storage_root: Path) -> str:
    """Build honest source and lazy-view accounting without loading case arrays."""
    source_catalog = catalog.source_catalog
    batches = source_catalog.batches
    admitted = sum(batch.available_case_count for batch in batches)
    failed = sum(batch.failed_case_count for batch in batches)
    incomplete = sum(batch.incomplete_case_count for batch in batches)
    invalid = sum(batch.invalid_case_count for batch in batches)
    labels = _batch_labels(batches)
    rows = [
        (
            f"- {labels[batch.batch_id]} "
            f"[batch_id={batch.batch_id}; storage={batch.batch_storage_name}; profile={batch.simulation_profile}] "
            f"({batch.source_kind}): {batch.available_case_count} admitted, "
            f"{batch.failed_case_count} failed, {batch.incomplete_case_count} incomplete, "
            f"{batch.invalid_case_count} invalid"
        )
        for batch in batches
    ]
    body = "\n".join(rows) if rows else "- none"
    return (
        "Generated-output EDA summary\n\n"
        f"Storage root: {storage_root}\n"
        f"Discovered candidate batches: {source_catalog.discovered_batch_count}\n"
        f"Admitted datasets: {len(catalog.views)}\n"
        f"Complete terminal batches: {sum(batch.source_kind == 'terminal' for batch in batches)}\n"
        f"Partial batches: {sum(batch.source_kind == 'partial' for batch in batches)}\n"
        f"Admitted completed source cases: {admitted}\n"
        "Scientific payloads materialized during preparation: 0\n"
        f"Failed cases: {failed}\n"
        f"Incomplete or running cases: {incomplete}\n"
        f"Invalid or corrupt cases: {invalid}\n"
        f"Discovery issues: {source_catalog.total_issue_count}\n\n"
        f"Batches:\n{body}"
    )


def _view_loader(
    batch: sources.GeneratedOutputEDABatch,
    *,
    storage_root: Path,
    max_cases: int | None,
) -> Callable[[], pd.DataFrame]:
    """Return one zero-argument profile-native authoritative lazy dataframe loader."""
    task = domain.tasks.registry.get_task(batch.simulation_profile)

    def load() -> pd.DataFrame:
        """Materialize exactly one admitted batch projection on demand."""
        if batch.source_kind == "terminal":
            frame, _logs = dataframe.generate_eda_dataframe(
                batch.batch_storage_name,
                task=task,
                storage_root=storage_root,
                show_progress=False,
                max_cases=max_cases,
            )
        else:
            frame, _logs = dataframe.generate_eda_dataframe_from_completed_cases(
                batch,
                task=task,
                show_progress=False,
                max_cases=max_cases,
            )
        return frame

    return load


def prepare_generated_output_eda_workspace(
    *,
    storage_root: Path | str,
    max_cases: int | None = None,
    title: str = "Generated-output EDA",
) -> GeneratedOutputEDAWorkspace:
    """
    Prepare one adaptive read-only workspace over all admitted Generation outputs.

    Parameters
    ----------
    storage_root : Path | str
        Canonical storage root containing persisted campaign evidence.
    max_cases : int | None, optional
        Positive per-view case-reference and materialization bound.
    title : str, optional
        Visible title for the lazy interactive panel.

    Returns
    -------
    GeneratedOutputEDAWorkspace
        Plain source accounting, lazy catalog/state, and optional outer panel.

    Notes
    -----
    Discovery and full case-level admission occur at preparation. Scientific
    arrays remain unloaded until an active view requests selected profile-native
    data. Partial campaigns never become Dataset publications.

    """
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        message = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise ValueError(message)
    if not isinstance(title, str) or not title.strip():
        message = "title must be non-empty text."
        raise ValueError(message)
    root = Path(storage_root).expanduser().resolve()
    source_catalog = sources.discover_generated_output_eda_catalog(storage_root=root)
    batches = tuple(batch for batch in source_catalog.batches if batch.cases)
    labels_by_batch = _batch_labels(batches)
    views = tuple(
        selection.GeneratedOutputEDAView(
            label=labels_by_batch[batch.batch_id],
            batch=batch,
            case_limit=max_cases,
            loader=_view_loader(batch, storage_root=root, max_cases=max_cases),
        )
        for batch in batches
    )
    catalog = selection.GeneratedOutputEDACatalog(views, source_catalog=source_catalog)
    summary = _summary(catalog, storage_root=root)
    issues = _issues_text(source_catalog)
    if issues:
        summary = f"{summary}\n\n{issues}"
    if not views:
        return GeneratedOutputEDAWorkspace(
            summary_text=(
                f"{summary}\n\n"
                "No individually admitted completed cases are available. Generate or "
                "resume the configured campaign, then rerun this notebook."
            ),
            panel=None,
            catalog=catalog,
            selection_state=None,
        )
    selection_state = selection.GeneratedOutputSelectionState(catalog)
    interactive_panel = panel.build_eda_panel(catalog=catalog, selection_state=selection_state, title=title)
    return GeneratedOutputEDAWorkspace(
        summary_text=summary,
        panel=interactive_panel,
        catalog=catalog,
        selection_state=selection_state,
    )
