"""Prepare the read-only unified generated-output EDA notebook workspace.

Responsibilities:
  - Discover current Generation campaign decisions once
  - Project successful candidates into lightweight profile-native views
  - Order the material views of a published package as the evaluation does (role, then material), make equal «material, role»
    labels unique by package name (by the exact package key where no token tells them apart), name each package alias once
    in the summary, and open the first view in manifest order
  - Bind lazy dataframe loaders, which also record each dataset's label without its package alias for the figures, shared
    selection, summary, and one outer panel
  - Report selected, failed, pending, and invalid source accounting

Design principles:
  - Source discovery and exact case admission remain independent of Dataset publication
  - Scientific payloads load only when an active view requests selected data
  - Simulation-profile metadata drives scientific dispatch without a user task filter

This module does NOT:
  - Finalize Generation batches or publish Dataset packages
  - Define EDA scientific calculations, plots, or widget callbacks
  - Reconstruct campaign evidence or relax case validation
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import domain
from src.analysis.presentation import analysis_dataset_selection as dataset_selection
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels

from . import eda_capabilities as capabilities
from . import eda_dataframe as dataframe
from . import eda_panel as panel
from . import eda_selection as selection
from . import eda_sources as sources

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import ipywidgets as widgets
    import pandas as pd

_MAXIMUM_DISPLAYED_ISSUES: Final = 5
_ROLE_RANK: Final = {role: rank for rank, role in enumerate(dataset_selection.EVALUATION_GROUP_ORDER)}  # a view without one role last


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
        domain_profile=batch.domain_profile,
        canonical_identity=batch.batch_id,
    )


def _batch_labels(batches: Sequence[sources.GeneratedOutputEDABatch]) -> dict[str, str]:
    """Return collision-safe concise labels keyed by campaign view identity."""
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
    if not isinstance(source_catalog, sources.GeneratedOutputEDACatalog):
        message = "Generation summary requires an admitted Generation catalog."
        raise TypeError(message)
    batches = source_catalog.batches
    admitted = sum(batch.available_case_count for batch in batches)
    failed = sum(batch.failed_case_count for batch in batches)
    incomplete = sum(batch.incomplete_case_count for batch in batches)
    invalid = sum(batch.invalid_case_count for batch in batches)
    labels = _batch_labels(batches)
    rows = [
        (
            f"- {labels[batch.batch_id]} "
            f"[campaign={batch.campaign_name}; profile={batch.simulation_profile}; domain={batch.domain_profile}] "
            f"{batch.available_case_count} successful, {batch.selected_case_count} selected, "
            f"{batch.failed_case_count} failed, {batch.incomplete_case_count} incomplete, "
            f"{batch.invalid_case_count} invalid"
        )
        for batch in batches
    ]
    body = "\n".join(rows) if rows else "- none"
    return (
        "Generated-output EDA summary\n\n"
        f"Storage root: {storage_root}\n"
        f"Discovered campaigns: {source_catalog.discovered_campaign_count}\n"
        f"Discovered candidates: {source_catalog.discovered_candidate_count}\n"
        f"Admitted datasets: {len(catalog.views)}\n"
        f"Admitted completed source cases: {admitted}\n"
        "Scientific payloads materialized during preparation: 0\n"
        f"Failed cases: {failed}\n"
        f"Incomplete or running cases: {incomplete}\n"
        f"Invalid or corrupt cases: {invalid}\n"
        f"Discovery issues: {source_catalog.total_issue_count}\n\n"
        f"Campaign views:\n{body}"
    )


def _stride_note(display_spatial_stride: display_grid.DisplayStride) -> str:
    """Return the summary line that names how maps are sampled."""
    return f"Map display stride: {display_spatial_stride}; statistics and scale limits use complete source grids."


def _recording_bare_label(load: Callable[[], pd.DataFrame], bare: str) -> Callable[[], pd.DataFrame]:
    """Return ``load`` whose frame also records the dataset's ``bare`` label, which a figure shows where no other plotted label equals it."""

    def load_labelled() -> pd.DataFrame:
        """Load the frame and name its dataset without the alias that tells it from equal labels elsewhere in the catalog."""
        frame = load()
        frame.attrs[capabilities.DISPLAY_LABEL_ATTRIBUTE] = bare
        return frame

    return load_labelled


def _view_loader(
    batch: sources.GeneratedOutputEDABatch,
    *,
    max_cases: int | None,
) -> Callable[[], pd.DataFrame]:
    """Return one zero-argument profile-native authoritative lazy dataframe loader."""
    task = domain.tasks.registry.get_task(batch.simulation_profile)

    def load() -> pd.DataFrame:
        """Materialize exactly one admitted batch projection on demand."""
        frame, _logs = dataframe.generate_eda_dataframe_from_completed_cases(
            batch,
            task=task,
            show_progress=False,
            max_cases=max_cases,
        )
        return frame

    return _recording_bare_label(load, display_labels.dataset_display_label(_batch_label_metadata(batch)))


def _published_view(
    manifest: dict[str, Any], material: str | None, *, storage_root: Path, max_cases: int | None, label: str | None = None, bare: str | None = None
) -> selection.PublishedDatasetEDAView:
    """Bind one package, or one material of a mixed package, to its bounded authoritative loader; ``bare`` is its label without alias."""
    loader: Callable[[], pd.DataFrame] = partial(
        dataframe.load_published_dataset_frame, manifest, storage_root=storage_root, max_cases=max_cases, material=material
    )
    if bare is not None:
        loader = _recording_bare_label(loader, bare)
    return selection.PublishedDatasetEDAView(manifest=manifest, case_limit=max_cases, loader=loader, material=material, label=label)


def prepare_generated_output_eda_workspace(
    *,
    storage_root: Path | str,
    max_cases: int | None = None,
    title: str = "Generated-output EDA",
    display_spatial_stride: display_grid.DisplayStride = "auto",
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

    display_spatial_stride : {1, 2, 4, "auto"}, optional
        Endpoint-preserving map samples; statistics use complete grids.

    Returns
    -------
    GeneratedOutputEDAWorkspace
        Plain source accounting, lazy catalog/state, and optional outer panel.

    Notes
    -----
    Discovery reads campaign decisions at preparation. Exact case admission and
    scientific arrays remain deferred until an active view requests the selected
    profile-native data. Active campaigns can expose completed candidates.

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
            loader=_view_loader(batch, max_cases=max_cases),
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
    interactive_panel = panel.build_eda_panel(
        catalog=catalog, selection_state=selection_state, title=title, display_spatial_stride=display_spatial_stride
    )
    return GeneratedOutputEDAWorkspace(
        summary_text=f"{summary}\n\n{_stride_note(display_spatial_stride)}",
        panel=interactive_panel,
        catalog=catalog,
        selection_state=selection_state,
    )


def prepare_published_dataset_eda_workspace(
    *,
    storage_root: Path | str,
    max_cases: int | None = 8,
    title: str = "Published Dataset EDA",
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> GeneratedOutputEDAWorkspace:
    """
    Prepare package-backed inputs, targets and aggregate EDA lazily.

    Package discovery validates metadata only. Runtime activation verifies the
    full payload hash, preserves the authoritative spatial grid, and applies
    the explicit case bound before transient scientific-array materialization.
    """
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        message = "max_cases must be a positive integer or None."
        raise ValueError(message)
    root = Path(storage_root).expanduser().resolve()
    source_catalog = sources.discover_published_dataset_eda_catalog(storage_root=root)
    # A mixed-material package shows as one view per material over its own cases, so no series pools materials; the views of a
    # package follow the evaluation order (role, then material), and equal «material, role» labels take the package name.
    specs = []
    opening = ""  # the key of the first view in manifest order, which opens whatever the display order
    for manifest in source_catalog.packages:
        materials = (None,) if len(manifest["materials"]) == 1 else manifest["materials"]
        package = [(manifest, material, _published_view(manifest, material, storage_root=root, max_cases=max_cases)) for material in materials]
        opening = opening or package[0][2].key
        specs += sorted(package, key=lambda spec: (_ROLE_RANK.get(spec[2].role or "", len(_ROLE_RANK)), spec[2].material_family or ""))
    bases = [view.label for _manifest, _material, view in specs]
    facets = [(str(manifest["dataset_name"]), view.package_key) for manifest, _material, view in specs]
    try:
        labels = display_labels.unique_display_labels(bases, facets)
    except ValueError:  # no token of the names or keys tells the packages apart (a__b, b__a): their exact keys do
        labels = tuple(f"{base} · {key}" if bases.count(base) > 1 else base for base, (_name, key) in zip(bases, facets, strict=True))
    views = tuple(
        _published_view(manifest, material, storage_root=root, max_cases=max_cases, label=label, bare=base)
        for (manifest, material, _view), label, base in zip(specs, labels, bases, strict=True)
    )
    catalog = selection.GeneratedOutputEDACatalog(views, source_catalog=source_catalog)
    by_package: dict[str, list[selection.PublishedDatasetEDAView]] = {}
    for view in views:
        by_package.setdefault(view.package_key, []).append(view)
    packages = "\n".join(f"- {'; '.join(view.label for view in group)}: {group[0].package_label}" for group in by_package.values())
    summary = (
        f"Published Dataset EDA: {len(source_catalog.packages)} admitted package manifests as {len(views)} views "
        f"(a mixed-material package as one view per material); {len(source_catalog.issues)} discovery issues.\n"
        f"{packages}\n"
        f"Case bound per view: {max_cases if max_cases is not None else 'all'}; scientific payloads opened during preparation: 0.\n"
        "Transient: complete selected cases from the canonical HDF5 backend; full index admission is required.\n"
        "Boundary inputs retain exact interval endpoints/startup support; "
        "full authored schedules and completion evidence are not part of the runtime view.\n"
        "Steady: the immutable PT payload must load in full once; displayed cases use the explicit bound.\n"
        "Values precede train-fitted normalization. Package membership is authoritative; training fitted scalers are separate."
    )
    if source_catalog.issues:
        summary += "\n" + "\n".join(f"- {issue.source_id}: {issue.message}" for issue in source_catalog.issues[:_MAXIMUM_DISPLAYED_ISSUES])
    if not views:
        return GeneratedOutputEDAWorkspace(summary, None, catalog, None)
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((opening,))  # one view, so the tensor inspector opens on it
    interactive_panel = panel.build_eda_panel(catalog=catalog, selection_state=state, title=title, display_spatial_stride=display_spatial_stride)
    return GeneratedOutputEDAWorkspace(f"{summary}\n{_stride_note(display_spatial_stride)}", interactive_panel, catalog, state)
