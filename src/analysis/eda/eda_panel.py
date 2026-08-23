"""
eda_panel.py

Compose one capability-adaptive lazy EDA panel from generated-output views.

Responsibilities:
  - Bind one panel-level dataset selector to the shared selection state
  - Filter one presentation registry by selected-dataset capability unions
  - Construct explicit cached view factories without task-selected panels
  - Preserve lazy activation and single- or multi-page PDF export

Design principles:
  - One outer panel and one catalog own steady and transient datasets together
  - Semantic registry keys, rather than visible numbers, own view identity
  - Capability changes replace only visible sections, never the outer panel

This module does NOT:
  - Discover or materialize generated data directly
  - Define scientific plots, channel widgets, or stored-time navigation
  - Catch programming errors from incorrectly registered factories
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import ipywidgets as widgets

from src.analysis import ui
from src.analysis.presentation import registry as presentation

from . import eda_controls as controls
from . import eda_selection as selection
from . import eda_viewers as viewers
from . import plots

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.figure import Figure

_TransientCaseKind = Literal["snapshot", "trajectory"]
_ORIENTATION_SELECTOR_WIDTH_PX = int(ui.notebook.STANDARD_VIEW_SELECTOR_WIDTH_PX * 1.25)


def _cached_factory(
    builder: Callable[..., viewers.ActivatableView],
) -> Callable[..., widgets.Widget]:
    """Wrap one explicit builder in a state-preserving lazy factory."""
    cached: viewers.ActivatableView | None = None

    def invoke(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> widgets.Widget:
        """Construct once or reactivate the cached current-selection view."""
        nonlocal cached
        if cached is None:
            cached = builder(
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
        else:
            cached.activate()
        return cached

    return invoke


def _view_factories(
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
) -> Mapping[str, Callable[..., widgets.Widget]]:
    """Bind every maintained semantic registry key to one cached factory."""

    def statistics(
        plot_function: Callable[..., widgets.Widget],
        *,
        required_capabilities: Sequence[selection.GeneratedOutputCapability],
    ) -> Callable[..., widgets.Widget]:
        """Bind one established capability-adaptive case-statistics viewer."""

        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> viewers.ActivatableView:
            """Build one shared-selection statistics view."""
            return viewers.make_statistics_view(
                catalog=catalog,
                selection_state=selection_state,
                plot_function=plot_function,
                required_capabilities=required_capabilities,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )

        return _cached_factory(build)

    def spectral(
        *,
        single: Callable[..., Figure],
        aggregate: Callable[..., Figure] | None,
        semantic_controls: Mapping[str, widgets.ValueWidget] | None = None,
    ) -> Callable[..., widgets.Widget]:
        """Bind one capability-adaptive spectral view."""

        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> viewers.ActivatableView:
            """Build one shared-selection spectral view."""
            return viewers.make_spectral_view(
                catalog=catalog,
                selection_state=selection_state,
                single_plot_function=single,
                aggregate_plot_function=aggregate,
                semantic_controls=semantic_controls,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )

        return _cached_factory(build)

    def spatial_maps(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> viewers.ActivatableView:
        """Build shared steady/transient retained spatial-field maps."""
        return viewers.make_spatial_case_view(
            catalog=catalog,
            selection_state=selection_state,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def transient_case(kind: _TransientCaseKind) -> Callable[..., widgets.Widget]:
        """Bind one transient case view through shared case/time controls."""

        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> viewers.ActivatableView:
            """Build one task-native transient case view."""
            return viewers.make_transient_case_view(
                catalog=catalog,
                selection_state=selection_state,
                kind=kind,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )

        return _cached_factory(build)

    def completion_target(
        *,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> viewers.ActivatableView:
        """Build the consolidated target diagnostic and companion table."""
        return viewers.make_completion_target_view(
            catalog=catalog,
            selection_state=selection_state,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    orientation = widgets.Dropdown(
        options=(
            (
                "Cross-stream spectrum k_x along flow direction y",
                "cross_stream_along_flow",
            ),
            (
                "Flow-direction spectrum k_y across cross-stream direction x",
                "flow_across_cross_stream",
            ),
        ),
        value="cross_stream_along_flow",
        description="Orientation:",
        style={"description_width": "initial"},
        layout=widgets.Layout(
            width=f"{_ORIENTATION_SELECTOR_WIDTH_PX}px",
            max_width=f"{_ORIENTATION_SELECTOR_WIDTH_PX}px",
            flex=f"0 0 {_ORIENTATION_SELECTOR_WIDTH_PX}px",
        ),
    )
    return {
        "metadata_statistics": statistics(
            plots.case_statistics.plot_meta_statistics,
            required_capabilities=("generated_output",),
        ),
        "parameter_distributions": statistics(
            plots.case_statistics.plot_meta_parameters,
            required_capabilities=("generated_output",),
        ),
        "field_value_distributions": statistics(
            plots.case_statistics.plot_field_value_distributions,
            required_capabilities=("spatial_fields",),
        ),
        "spatial_field_maps": _cached_factory(spatial_maps),
        "isotropic_spectra": spectral(
            single=plots.spectral.plot_isotropic_spectral_case,
            aggregate=plots.spectral.plot_isotropic_spectral_summary,
        ),
        "directional_spectra": spectral(
            single=plots.spectral.plot_directional_spectral_case,
            aggregate=plots.spectral.plot_directional_spectral_summary,
        ),
        "spectral_evolution": spectral(
            single=plots.spectral.plot_vertical_spectral_case,
            aggregate=plots.spectral.plot_vertical_spectral_evolution,
            semantic_controls={"orientation": orientation},
        ),
        "transient_state_snapshots": transient_case("snapshot"),
        "transient_state_trajectories": transient_case("trajectory"),
        "transient_completion_target": _cached_factory(completion_target),
    }


def _sections_for_capabilities(
    *,
    capabilities: Sequence[selection.GeneratedOutputCapability],
    factories: Mapping[str, Callable[..., widgets.Widget]],
    export_state: dict[str, Any],
) -> tuple[tuple[widgets.Widget, ...], tuple[str, ...]]:
    """Build lazy numbered sections for one selected-dataset capability union."""
    sections: list[widgets.Widget] = []
    titles: list[str] = []
    visible = presentation.eda_sections_for_capabilities(capabilities)
    for _section, section_label, numbered_plots in presentation.numbered_registry(visible):
        entries = []
        for plot, plot_label in numbered_plots:
            try:
                factory = factories[plot.key]
            except KeyError as error:
                message = f"EDA presentation view {plot.key!r} has no registered factory."
                raise ValueError(message) from error
            entries.append((plot_label, factory, plot.key))
        sections.append(
            ui.notebook.make_dropdown_section(
                entries,
                export_state=export_state,
                select_first=True,
            )
        )
        titles.append(section_label)
    if not sections:
        message = "Selected generated-output datasets expose no maintained EDA views."
        raise ValueError(message)
    return tuple(sections), tuple(titles)


def _selected_capabilities(
    selection_state: selection.GeneratedOutputSelectionState,
) -> tuple[selection.GeneratedOutputCapability, ...]:
    """Return the deterministic union declared by currently selected datasets."""
    selected = selection_state.selected_views()
    union = frozenset().union(*(view.capabilities for view in selected))
    return tuple(sorted(union))


def build_eda_panel(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    title: str = "Generated-output EDA",
) -> widgets.Widget:
    """Build the sole capability-adaptive generated-output EDA outer panel."""
    if not isinstance(catalog, selection.GeneratedOutputEDACatalog):
        message = "EDA panel construction requires a GeneratedOutputEDACatalog."
        raise TypeError(message)
    if not catalog.views:
        message = "EDA panel construction requires at least one admitted dataset."
        raise ValueError(message)
    if not isinstance(selection_state, selection.GeneratedOutputSelectionState) or selection_state.catalog is not catalog:
        message = "EDA panel selection state must use the supplied catalog."
        raise ValueError(message)
    if not isinstance(title, str) or not title.strip():
        message = "EDA panel title must be non-empty text."
        raise ValueError(message)
    factories = _view_factories(catalog, selection_state)
    export_state: dict[str, Any] = {
        "fig": None,
        "figures": (),
        "plot_name": None,
        "title": None,
    }
    capabilities = _selected_capabilities(selection_state)
    sections, tab_titles = _sections_for_capabilities(
        capabilities=capabilities,
        factories=factories,
        export_state=export_state,
    )
    dataset_control = controls.GeneratedOutputDatasetControl(
        selection_state=selection_state,
    )
    outer = ui.notebook.make_lazy_panel_with_tabs(
        sections,
        tab_titles=tab_titles,
        open_btn_text=f"{title.strip()} - Open",
        close_btn_text="Close",
        panel_controls=(dataset_control.widget,),
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )
    displayed_capabilities = {"value": capabilities}

    def selection_changed(_current: selection.GeneratedOutputSelection) -> None:
        """Replace only registry sections when the capability union changes."""
        current_capabilities = _selected_capabilities(selection_state)
        if not current_capabilities or current_capabilities == displayed_capabilities["value"]:
            return
        replacement_sections, replacement_titles = _sections_for_capabilities(
            capabilities=current_capabilities,
            factories=factories,
            export_state=export_state,
        )
        displayed_capabilities["value"] = current_capabilities
        outer.replace_tabs(replacement_sections, replacement_titles)

    selection_state.observe(selection_changed)
    return outer
