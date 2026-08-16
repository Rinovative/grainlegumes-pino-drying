"""
===============================================================================
generation_input_panel.py
===============================================================================
Compose the compact input-generated EDA panel.
Responsibilities:
  - Bind the four-section registry to A/B comparison and dataset views
  - Construct local Dataset A/Case A and Dataset B/Case B controls
  - Preserve automatic first rendering, lazy tabs, and current-figure PDF export
  - Keep transient capability filtering and notebook output ownership explicit
Design principles:
  - Every case-level view uses one common comparison selection model
  - Discovery occurs once when the notebook executes
  - Scientific data loads only through the active view's local controls
This module does NOT:
  - Refresh or generate inputs, implement plot mathematics, or inspect outputs
  - Modify completed-output EDA or evaluation behavior
===============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from matplotlib.figure import Figure

from src.analysis import ui
from src.analysis.generation_inputs import generation_input_controls as controls
from src.analysis.generation_inputs import generation_input_presentation as presentation
from src.analysis.generation_inputs import generation_input_selection as selection_service
from src.analysis.generation_inputs import generation_input_sources as source_service
from src.analysis.generation_inputs import plots
from src.analysis.presentation import registry as shared_presentation

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_HANDLED_VIEW_ERRORS = (
    OSError,
    TypeError,
    ValueError,
    RuntimeError,
    KeyError,
)


class _LocalControls(Protocol):
    """Define the controls contract required by one live view."""

    @property
    def widget(self) -> widgets.Widget:
        """Return the visible view-local control container."""
        ...

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the automatic render callback."""
        ...


def _result_items(result: Any) -> tuple[Any, ...]:
    """Normalize one view result without treating strings as sequences."""
    return result if isinstance(result, tuple) else (result,)


def _live_view(
    local_controls: _LocalControls,
    render: Callable[[], Any],
    *,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> tuple[widgets.VBox, Callable[[], None]]:
    """Build one active-aware automatically updating local-controls view."""
    output = widgets.Output(layout=widgets.Layout(width="100%", overflow="visible"))

    def update(*, force: bool = False) -> None:
        """Render active local state and synchronize its export target."""
        if not force and export_state is not None and export_state.get("plot_name") != export_plot_name:
            return
        if export_state is not None:
            previous = export_state.get("fig")
            if isinstance(previous, Figure):
                plt.close(previous)
            export_state.update(
                {
                    "fig": None,
                    "plot_name": export_plot_name,
                    "title": export_title,
                }
            )
        with output:
            output.clear_output(wait=True)
            try:
                items = _result_items(render())
            except _HANDLED_VIEW_ERRORS as error:
                display(widgets.HTML(f"<p><b>View unavailable for the current selection.</b> {escape(str(error))}</p>"))
                return
            figure: Figure | None = None
            for item in items:
                if isinstance(item, Figure):
                    if figure is None:
                        figure = item
                    display(item)
                    plt.close(item)
                elif item is not None:
                    display(item)
            if export_state is not None:
                export_state["fig"] = figure

    local_controls.set_callback(update)
    update(force=True)
    widget = widgets.VBox(
        (local_controls.widget, output),
        layout=widgets.Layout(
            width="100%",
            align_items="stretch",
        ),
    )
    return widget, lambda: update(force=True)


def _pair_factory(
    catalog: source_service.GenerationInputDatasetCatalog,
    selection_state: selection_service.GenerationInputSelectionState,
    render: Callable[[controls.SelectedComparison], Any],
    *,
    include_scale_lock: bool = False,
) -> Callable[..., widgets.VBox]:
    """Return one cached view bound to the common A/B selection state."""
    cached: widgets.VBox | None = None
    activate: Callable[[], None] | None = None

    def invoke(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> widgets.VBox:
        """Construct local controls once and preserve view-local state."""
        nonlocal activate, cached
        if export_state is not None:
            export_state.update(
                {
                    "plot_name": export_plot_name,
                    "title": export_title,
                }
            )
        if cached is None:
            local = controls.PairCaseControls(
                catalog,
                selection_state=selection_state,
                include_scale_lock=include_scale_lock,
            )
            cached, activate = _live_view(
                local,
                lambda: render(local.selected_comparison()),
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
        elif activate is not None:
            activate()
        return cached

    return invoke


def _dataset_factory(
    catalog: source_service.GenerationInputDatasetCatalog,
    render: Callable[[Any], Any],
) -> Callable[..., widgets.VBox]:
    """Return one cached dataset-overview view factory."""
    cached: widgets.VBox | None = None
    activate: Callable[[], None] | None = None

    def invoke(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> widgets.VBox:
        """Construct one local dataset selector once and preserve its value."""
        nonlocal activate, cached
        if export_state is not None:
            export_state.update(
                {
                    "plot_name": export_plot_name,
                    "title": export_title,
                }
            )
        if cached is None:
            local = controls.DatasetControls(catalog)
            cached, activate = _live_view(
                local,
                lambda: render(local.selected_diagnostics()),
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
        elif activate is not None:
            activate()
        return cached

    return invoke


def _view_factories(
    catalog: source_service.GenerationInputDatasetCatalog,
    selection_state: selection_service.GenerationInputSelectionState,
) -> Mapping[str, Callable[..., widgets.Widget]]:
    """Bind every compact presentation key to one cached local factory."""

    def case_context(selection: controls.SelectedComparison) -> widgets.Widget:
        """Render A/B context and technical provenance."""
        return plots.overview.case_context(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
        )

    def parameters(selection: controls.SelectedComparison) -> widgets.Widget:
        """Render exact scalar values and empirical dataset means."""
        return plots.overview.parameters(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
        )

    def field_summaries(
        selection: controls.SelectedComparison,
    ) -> widgets.Widget:
        """Render maintained field statistics and empirical means."""
        return plots.overview.field_summaries(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
        )

    def boundary_conditions(
        selection: controls.SelectedComparison,
    ) -> widgets.Widget:
        """Render boundary values under the common table contract."""
        return plots.boundaries.boundary_comparison_table(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
        )

    def transient_schedules(
        selection: controls.SelectedComparison,
    ) -> Figure:
        """Render all schedule channels, means, and exact supports."""
        return plots.boundaries.schedule_comparison(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
            same_dataset=selection.same_dataset,
        )

    def basic_spatial(
        selection: controls.SelectedComparison,
    ) -> Any:
        """Render porosity maps, pressure lines, distributions, and summaries."""
        return plots.spatial.basic_comparison(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
            lock_scale=selection.lock_scale,
        )

    def permeability_tensor(
        selection: controls.SelectedComparison,
    ) -> Any:
        """Render tensor maps, distributions, and embedded summaries."""
        return plots.permeability.tensor_comparison(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
            lock_scale=selection.lock_scale,
        )

    def derived_permeability(
        selection: controls.SelectedComparison,
    ) -> Any:
        """Render derived permeability maps, distributions, and summaries."""
        return plots.permeability.derived_comparison(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
            lock_scale=selection.lock_scale,
        )

    def moisture_sorption(
        selection: controls.SelectedComparison,
    ) -> Any:
        """Render moisture maps, distributions, summaries, and RH relation."""
        return plots.moisture.moisture_comparison(
            selection.case_a,
            selection.mean_a,
            selection.case_b,
            selection.mean_b,
            lock_scale=selection.lock_scale,
        )

    return {
        "case_context": _pair_factory(catalog, selection_state, case_context),
        "parameters": _pair_factory(catalog, selection_state, parameters),
        "field_summaries": _pair_factory(catalog, selection_state, field_summaries),
        "boundary_conditions": _pair_factory(
            catalog,
            selection_state,
            boundary_conditions,
        ),
        "transient_schedules": _pair_factory(
            catalog,
            selection_state,
            transient_schedules,
        ),
        "basic_spatial_inputs": _pair_factory(
            catalog,
            selection_state,
            basic_spatial,
            include_scale_lock=True,
        ),
        "permeability_tensor": _pair_factory(
            catalog,
            selection_state,
            permeability_tensor,
            include_scale_lock=True,
        ),
        "derived_permeability": _pair_factory(
            catalog,
            selection_state,
            derived_permeability,
            include_scale_lock=True,
        ),
        "moisture_sorption": _pair_factory(
            catalog,
            selection_state,
            moisture_sorption,
            include_scale_lock=True,
        ),
        "dataset_cases_parameters": _dataset_factory(
            catalog,
            plots.overview.dataset_cases_parameters,
        ),
        "dataset_field_summaries": _dataset_factory(
            catalog,
            plots.overview.dataset_field_summaries,
        ),
    }


class _GenerationInputPanelShell:
    """Own the sole outer panel, lazy tabs, and current export state."""

    def __init__(
        self,
        catalog: source_service.GenerationInputDatasetCatalog,
        *,
        title: str,
        export_dir: Path | str,
        selection_state: selection_service.GenerationInputSelectionState | None,
    ) -> None:
        """Initialize one collapsed panel without eager figures."""
        self._catalog = catalog
        self._selection_state = selection_state
        self._export_dir = Path(export_dir)
        self._export_state: dict[str, Any] = {
            "fig": None,
            "plot_name": None,
            "title": None,
        }
        self._main_output = widgets.Output()
        self._status_output = widgets.Output()
        self._open_button = widgets.Button(
            description=f"{title} - Open",
            button_style="primary",
            layout=widgets.Layout(width="auto"),
        )
        self._close_button = widgets.Button(
            description="Close",
            button_style="danger",
            layout=widgets.Layout(width="145px"),
        )
        self._export_button = widgets.Button(
            description="Export PDF",
            button_style="success",
            layout=widgets.Layout(width="145px"),
        )
        self._tabs = widgets.Tab()
        header = widgets.HBox((self._close_button, self._export_button))
        self._panel = widgets.VBox(
            (header, self._status_output, self._tabs),
            layout=widgets.Layout(width="100%"),
        )
        self._initialized_tabs: set[int] = set()
        self._tab_export_states: list[dict[str, Any]] = []
        self._tab_entries: list[tuple[tuple[str, Callable[..., widgets.Widget], str], ...]] = []
        self._section_keys: tuple[str, ...] = ()
        self._panel_open = False
        self._tabs.observe(
            self._on_tab_change,
            names="selected_index",
        )
        self._open_button.on_click(self._show_panel)
        self._close_button.on_click(self._show_open_button)
        self._export_button.on_click(self._export_current)
        self._build_tabs()
        self._show_open_button()

    @property
    def output(self) -> widgets.Output:
        """Return the single collapsed or expanded notebook output owner."""
        return self._main_output

    def _snapshot_tab(self, index: int | None) -> None:
        """Preserve one initialized tab's current export target."""
        if index is None or not 0 <= index < len(self._tab_export_states):
            return
        self._tab_export_states[index] = {
            "fig": self._export_state.get("fig"),
            "plot_name": self._export_state.get("plot_name"),
            "title": self._export_state.get("title"),
        }

    def _activate_selected_view(self) -> None:
        """Initialize the selected tab's first view or restore export state."""
        selected = self._tabs.selected_index
        if selected is None or not 0 <= selected < len(self._tabs.children):
            return
        children = getattr(
            self._tabs.children[selected],
            "children",
            (),
        )
        if not children or not isinstance(children[0], widgets.Dropdown):
            self._export_state.update({"fig": None, "plot_name": None, "title": None})
            return
        dropdown = children[0]
        if selected not in self._initialized_tabs:
            if dropdown.index is None and dropdown.options:
                dropdown.index = 0
            self._initialized_tabs.add(selected)
            self._snapshot_tab(selected)
        elif selected < len(self._tab_entries):
            view_index = dropdown.value
            entries = self._tab_entries[selected]
            if isinstance(view_index, int) and not isinstance(view_index, bool) and 0 <= view_index < len(entries):
                title, factory, plot_name = entries[view_index]
                factory(
                    export_state=self._export_state,
                    export_plot_name=plot_name,
                    export_title=title,
                )
                self._snapshot_tab(selected)
            elif selected < len(self._tab_export_states):
                self._export_state.update(self._tab_export_states[selected])

    def _build_tabs(self) -> None:
        """Build capability-filtered sections from immutable discovery."""
        if not self._catalog.datasets:
            message = "Generation-input panels require at least one admitted dataset."
            raise RuntimeError(message)
        if self._selection_state is None:
            message = "A non-empty generation-input panel requires shared selection state."
            raise RuntimeError(message)
        factories = _view_factories(self._catalog, self._selection_state)
        sections: list[widgets.Widget] = []
        titles: list[str] = []
        keys: list[str] = []
        tab_entries: list[tuple[tuple[str, Callable[..., widgets.Widget], str], ...]] = []
        visible = presentation.sections_for_profiles(self._catalog.profiles)
        for (
            section,
            section_label,
            numbered_views,
        ) in shared_presentation.numbered_registry(visible):
            entries = []
            for view, view_label in numbered_views:
                try:
                    factory = factories[view.key]
                except KeyError as error:
                    msg = f"Generation-input presentation view {view.key!r} has no factory."
                    raise ValueError(msg) from error
                entries.append((view_label, factory, view.key))
            sections.append(
                ui.notebook.make_dropdown_section(
                    entries,
                    export_state=self._export_state,
                    select_first=True,
                )
            )
            tab_entries.append(tuple(entries))
            titles.append(section_label)
            keys.append(section.key)
        self._tabs.children = tuple(sections)
        for index, title in enumerate(titles):
            self._tabs.set_title(index, title)
        self._section_keys = tuple(keys)
        self._tab_entries = tab_entries
        self._tab_export_states = [{"fig": None, "plot_name": None, "title": None} for _section in sections]
        self._tabs.selected_index = 0

    def _export_current(self, _button: object = None) -> None:
        """Write the current view's primary figure to a timestamped PDF."""
        with self._status_output:
            self._status_output.clear_output(wait=True)
            figure = self._export_state.get("fig")
            if not isinstance(figure, Figure):
                print("[Export] The current view has no rendered figure.")
                return
            self._export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stem = str(self._export_state.get("plot_name") or "generation_input_view")
            destination = self._export_dir / f"{stem}_{timestamp}.pdf"
            figure.savefig(destination, bbox_inches="tight")
            print(f"[Export] Saved: {destination}")

    def _show_panel(self, _button: object = None) -> None:
        """Open the panel with the first current view already rendered."""
        self._panel_open = True
        self._snapshot_tab(self._tabs.selected_index)
        self._activate_selected_view()
        with self._main_output:
            clear_output(wait=True)
            display(self._panel)

    def _show_open_button(self, _button: object = None) -> None:
        """Collapse to the sole open button without discarding state."""
        self._panel_open = False
        self._snapshot_tab(self._tabs.selected_index)
        with self._main_output:
            clear_output(wait=True)
            display(self._open_button)

    def _on_tab_change(self, change: dict[str, object]) -> None:
        """Snapshot the hidden tab and initialize the selected view."""
        old = change.get("old")
        self._snapshot_tab(old if isinstance(old, int) else None)
        self._activate_selected_view()


def build_generation_input_eda_panel(
    *,
    datasets: source_service.GenerationInputDatasetCatalog,
    initial_selection: selection_service.GenerationInputSelection | None = None,
    selection_state: selection_service.GenerationInputSelectionState | None = None,
    title: str = "Generation-input EDA",
    export_dir: Path | str = "",
) -> widgets.Output:
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
    return _GenerationInputPanelShell(
        datasets,
        title=title,
        export_dir=export_dir,
        selection_state=resolved_state,
    ).output
