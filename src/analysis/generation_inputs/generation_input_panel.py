"""Compose the compact input-generated EDA panel.

Responsibilities:
  - Bind the four-section registry to A/B comparison and dataset views, and the paired template comparison when two campaigns share a design,
    opened on the campaigns of Dataset A's task when that task has a pair
  - Place each view's local Dataset A/Case A and Dataset B/Case B controls in its panel rows
  - Open the dataset overviews on the panel's Dataset A and the campaign outcomes on the campaign its batch locator names,
    which says when it has no candidates; a local choice stays
  - Show campaign candidates, their rates per material, role and origin, and failure reasons in display names and headers,
    the rates and reasons in their campaign's material order under every filter, sampled parameters headed by their table
    row labels, and pass the atlas the Dataset its case belongs to
  - Write the case context's input directories from the storage root the catalog was discovered from
  - Preserve automatic first rendering, lazy tabs, and current-figure PDF export
  - Keep transient capability filtering and notebook output ownership explicit: a transient Dataset or campaign offers the
    transient-only views, which say when no transient Dataset can be selected

Design principles:
  - Every case-level view uses one common comparison selection model
  - Discovery occurs once when the notebook executes
  - Scientific data loads only through the active view's local controls
  - A view that is not shown renders nothing and draws once when it is shown again

This module does NOT:
  - Refresh or generate inputs, implement plot mathematics, or inspect outputs
  - Modify completed-output EDA or evaluation behavior
"""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import ipywidgets as widgets

from src.analysis import ui
from src.analysis.generation_inputs import generation_input_controls as controls
from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.generation_inputs import generation_input_presentation as presentation
from src.analysis.generation_inputs import generation_input_selection as selection_service
from src.analysis.generation_inputs import generation_input_sources as source_service
from src.analysis.generation_inputs import generation_input_table_schema as table_schema
from src.analysis.generation_inputs import generation_input_template_comparison as template_service
from src.analysis.generation_inputs import plots
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import registry as shared_presentation
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    from matplotlib.figure import Figure

_CANDIDATE_PREVIEW_ROWS: Final = 200  # the candidate table is a bounded preview; the rates count every candidate


class _LocalControls(Protocol):
    """Define the controls contract required by one live view."""

    @property
    def rows(self) -> Sequence[widgets.Widget]:
        """Return the view's control rows."""
        ...

    @property
    def popovers(self) -> Sequence[tuple[widgets.ToggleButton, widgets.DOMWidget]]:
        """Return the headers and bodies of the view's collapsed selections."""
        ...

    @property
    def status(self) -> ui.controls.StatusLine:
        """Return the status line the view's navigators report on."""
        ...

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the automatic render callback."""
        ...


def _live_view(
    local: _LocalControls,
    render: Callable[[], Any],
    *,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ui.panel.ViewPanel:
    """
    Build one automatically updating view over its local controls; a hidden view idles until it is shown again.

    A view result is one item or a tuple of items, such as a figure with its notes; the plot slot publishes all items of
    the newest request together and exports the first figure.
    """
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}

    def update() -> None:
        """Render the current selection; only the newest request publishes."""
        ui.viewers.render_view(view, render, **export)

    def refresh() -> None:
        """Follow a control change while the view is shown."""
        if view.active:
            update()

    view = ui.panel.ViewPanel(local.rows, status=local.status, slot=ui.panel.PlotSlot(), popovers=local.popovers, on_activate=update)
    local.set_callback(refresh)
    update()
    return view


def _factory(build: Callable[[], tuple[_LocalControls, Callable[[], Any]]]) -> Callable[..., Any]:
    """Return a view factory that builds its live view on first use and returns the same view afterwards."""
    cached: ui.panel.ViewPanel | None = None

    def invoke(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> ui.panel.ViewPanel:
        """Construct local controls once and preserve view-local state."""
        nonlocal cached
        if cached is None:
            local, render = build()
            cached = _live_view(local, render, export_state=export_state, export_plot_name=export_plot_name, export_title=export_title)
        return cached

    return invoke


def _pair_factory(
    catalog: source_service.GenerationInputDatasetCatalog,
    selection_state: selection_service.GenerationInputSelectionState,
    render: Callable[[controls.SelectedComparison], Any],
    *,
    include_scale_lock: bool = False,
) -> Callable[..., Any]:
    """Return one cached view bound to the common A/B selection state."""

    def build() -> tuple[_LocalControls, Callable[[], Any]]:
        local = controls.PairCaseControls(catalog, selection_state=selection_state, include_scale_lock=include_scale_lock)
        return local, lambda: render(local.selected_comparison())

    return _factory(build)


def _dataset_factory(
    catalog: source_service.GenerationInputDatasetCatalog,
    selection_state: selection_service.GenerationInputSelectionState,
    render: Callable[[Any], Any],
) -> Callable[..., Any]:
    """Return one cached dataset-overview view factory that opens on the panel's Dataset A; its own choice stays afterwards."""

    def build() -> tuple[_LocalControls, Callable[[], Any]]:
        local = controls.DatasetControls(catalog, selected=selection_state.selection.dataset_a_key)
        return local, lambda: render(local.selected_diagnostics())

    return _factory(build)


def _view_factories(
    catalog: source_service.GenerationInputDatasetCatalog,
    selection_state: selection_service.GenerationInputSelectionState,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Mapping[str, Callable[..., Any]]:
    """Bind every compact presentation key to one cached local factory."""

    def comparison(plot: Callable[..., Any], **fixed: Any) -> Callable[[controls.SelectedComparison], Any]:
        """Render both selected cases and both empirical dataset means with ``plot``."""
        return lambda selection: plot(selection.case_a, selection.mean_a, selection.case_b, selection.mean_b, **fixed)

    def maps(plot: Callable[..., Any]) -> Callable[[controls.SelectedComparison], Any]:
        """Render map comparisons that follow the local scale lock and the display stride."""
        return lambda selection: plot(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
            lock_scale=selection.lock_scale,
            display_spatial_stride=display_spatial_stride,
        )

    def pair(render: Callable[[controls.SelectedComparison], Any], *, lock: bool = False) -> Callable[..., Any]:
        """Bind a comparison renderer to the common A/B selection."""
        return _pair_factory(catalog, selection_state, render, include_scale_lock=lock)

    def case_context(selection: controls.SelectedComparison) -> Any:
        """Render the case context with its input directories written from the root the catalog was discovered from."""
        return plots.overview.case_context(selection.case_a, selection.mean_a, selection.case_b, selection.mean_b, storage_root=catalog.storage_root)

    def transient_schedules(selection: controls.SelectedComparison) -> Figure:
        """Render all schedule channels, means, and exact supports."""
        return plots.boundaries.schedule_comparison(
            selection.case_a, selection.mean_a, selection.case_b, selection.mean_b, same_dataset=selection.same_dataset
        )

    def transient_only(name: str, render: Callable[[controls.SelectedComparison], Any]) -> Callable[[controls.SelectedComparison], Any]:
        """Explain steady inputs in the plot's place: the catalog offers the view when any Dataset or campaign is transient, not per selection."""

        def guarded(selection: controls.SelectedComparison) -> Any:
            # Datasets A and B always share one simulation profile (generation_input_selection.py:244), so both are steady here.
            if diagnostics.is_transient(selection.case_a) and diagnostics.is_transient(selection.case_b):
                return render(selection)
            if profiles.TRANSIENT_DRYING_PROFILE in catalog.profiles:
                return f"{name} needs transient-drying inputs, but Datasets A and B are steady flow; select a transient campaign."
            return f"{name} needs transient-drying inputs, but no transient Dataset can be selected yet: no transient campaign has admitted inputs."

        return guarded

    factories: dict[str, Callable[..., Any]] = {
        "campaign_outcomes": _outcome_factory(
            catalog.outcomes, preferred=lambda: source_service.dataset_campaign(catalog.dataset(selection_state.selection.dataset_a_key))
        ),
        "field_atlas": _atlas_factory(catalog, selection_state, display_spatial_stride=display_spatial_stride),
        "case_context": pair(case_context),
        "parameters": pair(comparison(plots.overview.parameters)),
        "field_summaries": pair(comparison(plots.overview.field_summaries)),
        "boundary_conditions": pair(comparison(plots.boundaries.boundary_comparison_table)),
        "transient_schedules": pair(transient_only("Transient schedules", transient_schedules)),
        "basic_spatial_inputs": pair(maps(plots.spatial.basic_comparison), lock=True),
        "permeability_tensor": pair(maps(plots.permeability.tensor_comparison), lock=True),
        "derived_permeability": pair(maps(plots.permeability.derived_comparison), lock=True),
        "moisture_sorption": pair(transient_only("Moisture and sorption", maps(plots.moisture.moisture_comparison)), lock=True),
        "dataset_cases_parameters": _dataset_factory(catalog, selection_state, plots.overview.dataset_cases_parameters),
        "dataset_field_summaries": _dataset_factory(catalog, selection_state, plots.overview.dataset_field_summaries),
    }
    partners = template_service.comparable_campaigns(catalog.outcomes.rows)
    if partners:
        factories["template_comparison"] = _template_factory(catalog.outcomes, partners, selection_state)
    return factories


def _template_factory(
    catalog: source_service.CampaignOutcomeCatalog,
    partners: Mapping[template_service.Campaign, tuple[template_service.Campaign, ...]],
    selection_state: selection_service.GenerationInputSelectionState,
) -> Callable[..., Any]:
    """
    Bind one cached paired template comparison of the campaigns in ``partners``.

    Campaign A starts on a campaign of the task (simulation profile) of the common Dataset A when that task has partners,
    so the view opens on the campaigns of the panel's own task; otherwise on the first campaign with a partner.
    """

    def build() -> tuple[_LocalControls, Callable[[], Any]]:
        task = selection_state.selection.dataset_a_key[0]
        own_task_first = dict(sorted(partners.items(), key=lambda item: item[0][0] != task))  # a stable sort keeps the order within each group
        local = controls.TemplateComparisonControls(catalog, own_task_first)
        return local, lambda: plots.template.template_comparison(local.comparison(), local.captions)

    return _factory(build)


def _atlas_factory(
    catalog: source_service.GenerationInputDatasetCatalog,
    selection_state: selection_service.GenerationInputSelectionState,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Callable[..., Any]:
    """Keep one single-case atlas and its field choices through lazy activation."""

    def build() -> tuple[_LocalControls, Callable[[], Any]]:
        local = controls.FieldAtlasControls(catalog, selection_state)

        def render() -> tuple[Figure | None, widgets.HTML]:
            """Reuse admitted arrays and scales while toggling visible fields."""
            atlas = local.selected_atlas()
            notes = " ".join(atlas.notes)
            if not local.fields.selected:
                notes = "Select a spatial field. " + notes
            dataset = catalog.dataset(selection_state.selection.dataset_a_key)
            figure = plots.atlas.field_atlas(atlas, local.fields.selected, dataset=dataset, display_spatial_stride=display_spatial_stride)
            return figure, widgets.HTML(f"<p>{escape(notes)}</p>")

        return local, render

    return _factory(build)


def build_generation_input_eda_panel(
    *,
    datasets: source_service.GenerationInputDatasetCatalog,
    initial_selection: selection_service.GenerationInputSelection | None = None,
    selection_state: selection_service.GenerationInputSelectionState | None = None,
    title: str = "Generation-input EDA",
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> widgets.VBox:
    """Build one collapsible panel with session-scoped shared A/B state."""
    if not isinstance(datasets, source_service.GenerationInputDatasetCatalog):
        message = "build_generation_input_eda_panel requires a GenerationInputDatasetCatalog as datasets."
        raise TypeError(message)
    if selection_state is not None and initial_selection is not None:
        message = "Supply initial_selection or selection_state, not both."
        raise ValueError(message)
    if selection_state is not None and selection_state.catalog is not datasets:
        message = "selection_state must use the supplied dataset catalog."
        raise ValueError(message)
    if not datasets.datasets:
        message = "Generation-input panels require at least one admitted dataset."
        raise ValueError(message)
    resolved_state = selection_state or selection_service.GenerationInputSelectionState(
        datasets,
        initial_selection=initial_selection,
    )
    factories = _view_factories(datasets, resolved_state, display_spatial_stride=display_spatial_stride)
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": None, "title": None}
    sections: list[ui.notebook.ViewSection] = []
    titles: list[str] = []
    # A transient campaign offers the transient-only views before it has an admitted Dataset; they then say why none can be selected.
    tasks = {*datasets.profiles, *(str(row["task"]) for row in datasets.outcomes.rows)}
    sections_for_view = presentation.sections_for_profiles(tasks, template_comparison="template_comparison" in factories)
    for _section, section_label, numbered_views in shared_presentation.numbered_registry(sections_for_view):
        entries = []
        for view, view_label in numbered_views:
            try:
                factory = factories[view.key]
            except KeyError as error:
                msg = f"Generation-input presentation view {view.key!r} has no factory."
                raise ValueError(msg) from error
            entries.append((view_label, factory, view.key))
        sections.append(ui.notebook.view_section(entries, export_state=export_state))
        titles.append(section_label)
    return ui.notebook.LazyPanel(sections, titles, open_label=f"{title} - Open", export_state=export_state).widget


def _parameter_header(name: str) -> str:
    """Return the display header of a sampled parameter or boundary column from the shared row specs: its category and label."""
    spec = table_schema.boundary_row_spec(name) if name in table_schema.BOUNDARY_ORDER else table_schema.parameter_row_spec(name)
    return spec.label if spec.section == table_schema.UNCATEGORIZED_SECTION else f"{spec.category} · {spec.label}"


def _display_candidates(table: pd.DataFrame) -> pd.DataFrame:
    """
    Return a candidate, rate or failure-reason table with display values and headers; sampled parameters read as their table row labels.

    Failure stages, classes and reasons read as display text here; the evidence details of a candidate keep the exact codes and
    parameter names.
    """
    shown = table.copy()
    for column, display in (
        ("material_family", display_labels.material_display_label),
        ("sampling_regime", display_labels.regime_display_label),
        ("domain_profile", display_labels.humanize_identifier),
        ("origin", display_labels.humanize_identifier),
        ("outcome", display_labels.token_display_label),  # record tokens as the EDA figures write them
        ("failure_stage", display_labels.token_display_label),
        ("failure_class", display_labels.token_display_label),
        ("reason_code", display_labels.token_display_label),
    ):
        if column in shown:
            shown[column] = shown[column].map(display, na_action="ignore")  # a candidate that did not fail has no failure codes
    names = [str(column) for column in shown]
    return shown.rename(columns={name: table_schema.CANDIDATE_COLUMN_LABELS.get(name) or _parameter_header(name) for name in names})


def _outcome_factory(
    catalog: source_service.CampaignOutcomeCatalog,
    *,
    preferred: Callable[[], template_service.Campaign] | None = None,
) -> Callable[..., Any]:
    """Bind one cached candidate outcome table, evidence record, and optional plot, opened on the ``preferred`` campaign if one is given."""

    def build() -> tuple[_LocalControls, Callable[[], Any]]:
        local = controls.CampaignOutcomeControls(catalog, preferred=None if preferred is None else preferred())

        def render() -> object:
            if all((row["task"], row["source_name"]) != local.campaign.value for row in catalog.rows):  # only a preferred campaign can be empty
                return f"Campaign {local.campaign.label} has no candidate records at this discovery checkpoint: its evidence is unavailable."
            rows = local.candidates()
            scope = local.candidates(all_outcomes=True)
            # The frozen plan's material order: no filter and no partial checkpoint that leaves an ID material test designs only reorders it.
            campaign = cast("tuple[str, str]", local.campaign.value)  # the guard above matched it to a row's (task, source name)
            materials = source_service.campaign_material_order(catalog.designs.get(campaign, {}))
            table = source_service.campaign_outcome_table(rows)
            rates = source_service.campaign_outcome_rates(scope, materials=materials)
            failure_rates = source_service.campaign_failure_rates(scope, materials=materials)
            shown = min(len(rows), _CANDIDATE_PREVIEW_ROWS)
            message = widgets.HTML(
                f"<p>{len(rows)} matching candidates; the candidate table shows the first {shown}. "
                "Rates count every candidate in the selected campaign/material/origin scope, per material, role and origin; "
                "their denominator is the group's terminal candidates. "
                "Pending and technical attempts remain separate. Failed candidates never satisfy supervised success counts. "
                "Parameter plots draw, for every chosen parameter, the candidates of the selected campaign/material/origin scope with "
                "an admitted finite scalar value in the chosen unit; the legend counts are those of the whole scope, "
                "and a panel that draws fewer names its own n.</p>"
            )
            result: list[object] = [
                message,
                ui.tables.styled_dataframe(
                    _display_candidates(table.head(_CANDIDATE_PREVIEW_ROWS)),
                    title=f"Candidate outcomes — first {shown} of {len(rows)}",
                    shade_constant=False,
                ),
                ui.tables.styled_dataframe(_display_candidates(rates), title="Outcome rates by material, role and origin", shade_constant=False),
            ]
            if not failure_rates.empty:
                result.append(
                    ui.tables.styled_dataframe(
                        _display_candidates(failure_rates), title="Terminal failure reasons by material, role and origin", shade_constant=False
                    )
                )
            selected = next((row for row in rows if (row["material_family"], row["case_id"]) == local.case.value), None)
            if selected is not None:
                evidence = escape(json.dumps(selected, indent=2, default=str, ensure_ascii=False))
                result.append(
                    widgets.HTML(
                        "<details><summary>Exact selected candidate evidence, input locators, attempts and diagnostics</summary>"
                        f"<pre>{evidence}</pre></details>"
                    )
                )
            chosen = local.chosen_parameters()
            if chosen:
                values = {parameter: source_service.outcome_parameter_values(scope, *parameter) for parameter in chosen}
                counts = Counter(str(row["outcome"]) for row in scope)
                result.append(plots.overview.outcome_parameter_distributions(values, counts))
                result.append(plots.overview.failures_by_parameter(scope, chosen))
            return tuple(result)

        return local, render

    built = _factory(build)

    def invoke(
        *, export_state: dict[str, Any] | None = None, export_plot_name: str | None = None, export_title: str | None = None
    ) -> ui.panel.ViewPanel | widgets.HTML:
        if not catalog.rows:
            issues = " ".join(catalog.issues)
            return widgets.HTML(f"<p>No canonical campaign candidate records at this discovery checkpoint. {escape(issues)}</p>")
        return built(export_state=export_state, export_plot_name=export_plot_name, export_title=export_title)

    return invoke


def build_campaign_outcome_panel(catalog: source_service.CampaignOutcomeCatalog) -> widgets.VBox:
    """Keep pending/failed candidate evidence visible even before input admission."""
    shown = _outcome_factory(catalog)()
    shell = widgets.VBox((ui.style.stylesheet_widget(), shown.widget if isinstance(shown, ui.panel.ViewPanel) else shown))
    shell.add_class(ui.style.ROOT_CLASS)
    shell.add_class("glp-shell")
    return shell
