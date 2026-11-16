"""Protect the single capability-adaptive generated-output panel and registry.

Responsibilities:
  - Check capability registry coverage and lazy adaptive view reuse
  - Check the panel-owned dataset selector through tab replacement and an empty selection
  - Check view failure display and deactivated-view isolation from shared state

Design principles:
  - One panel follows admitted selection capabilities
  - A hidden view never loads or renders; a failing view never keeps a stale export

This module does NOT:
  - Load scientific case arrays or implement plot numerics
"""

from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import pandas as pd
from matplotlib.figure import Figure

from src.analysis.eda import eda_panel as panel
from src.analysis.eda import eda_selection as selection
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_viewers as viewers
from src.analysis.presentation import analysis_presentation_registry as presentation
from src.analysis.ui import analysis_ui_controls as ui_controls
from src.analysis.ui import analysis_ui_panel as ui_panel
from src.analysis.ui import analysis_ui_viewers as ui_viewers

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import pytest


_DEFAULT_VIEWS = (("steady_flow", "Airflow · Reference range"), ("transient_drying", "Drying · Lentil · Parameter extension"))


def _catalog(unreadable: str | None = None, specs: tuple[tuple[str, str], ...] = _DEFAULT_VIEWS) -> selection.GeneratedOutputEDACatalog:
    """Return one profile-native view per ``(task, label)`` spec, by default one steady and one transient; the ``unreadable`` task fails to load."""

    def loader(frame: pd.DataFrame, task_id: str) -> Callable[[], pd.DataFrame]:
        def load() -> pd.DataFrame:
            if task_id == unreadable:
                message = f"{task_id} dataset unreadable"
                raise RuntimeError(message)
            return frame

        return load

    views = []
    batches = []
    for task_id, label in specs:
        frame = pd.DataFrame(
            {"value": [1.0]},
            index=pd.Index(["case_0001"], name="sample_id"),
        )
        frame.attrs["task_id"] = task_id
        row = {
            "campaign_name": f"{task_id}-campaign",
            "source_name": f"{task_id}-campaign__{'a' * 16}",
            "campaign_digest": "a" * 64,
            "material_family": "lentil",
            "domain_profile": "parameter_extension" if task_id == "transient_drying" else "reference_range",
            "sampling_regime": "natural",
            "case_id": "case_0001",
            "case_input_id": "1" * 64,
            "simulation_case_id": "2" * 64,
            "case_hdf5_sha256": "3" * 64,
            "outcome": "successful",
            "selected": True,
        }
        case = sources.GeneratedOutputEDACase(case_id="case_0001", case_index=1, row=row)
        batch = sources.GeneratedOutputEDABatch(
            generation_root=Path(),
            batch_id=f"{label}-batch",
            campaign_name=f"{task_id}-campaign",
            source_name=row["source_name"],
            campaign_digest="a" * 64,
            campaign_purpose="source_generation",
            simulation_profile=task_id,
            available_learning_views=(task_id,),
            material_family="lentil",
            domain_profile=row["domain_profile"],
            sampling_regime="natural",
            cases=(case,),
            candidate_rows=(row,),
            storage_root=Path(),
        )
        batches.append(batch)
        views.append(
            selection.GeneratedOutputEDAView(
                label=label,
                batch=batch,
                case_limit=None,
                loader=loader(frame, task_id),
            )
        )
    source_catalog = sources.GeneratedOutputEDACatalog(
        batches=tuple(batches),
        issues=(),
        discovered_campaign_count=2,
        discovered_candidate_count=2,
        total_issue_count=0,
    )
    return selection.GeneratedOutputEDACatalog(views, source_catalog=source_catalog)


def _empty_view() -> ui_panel.ViewPanel:
    """Return one inert view with the production lifecycle surface."""
    return ui_panel.ViewPanel(())


def _case_count_viewer(
    *, datasets: dict[str, pd.DataFrame], styles: object = None, plot: Callable[..., Figure] | None = None, start_cases: int = 100, **options: Any
) -> ui_panel.ViewPanel:
    """Return a real nested case-count viewer over datasets of 5 and 100 cases of its own, as the statistics plot functions return one."""
    del datasets, styles
    frames: dict[Hashable, pd.DataFrame] = {"short": pd.DataFrame({"value": range(5)}), "long": pd.DataFrame({"value": range(100)})}
    return ui_viewers.case_count_view(plot or (lambda **_kwargs: Figure()), datasets=frames, start_cases=start_cases, step_size=1, **options)


def test_factories_cover_capability_registry_with_one_consolidated_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke every visible semantic factory with one consolidated diagnostic."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    calls: list[tuple[str, object]] = []

    def statistics(**kwargs: Any) -> ui_panel.ViewPanel:
        calls.append(("statistics", kwargs["plot_function"]))
        return _empty_view()

    def spectral(**kwargs: Any) -> ui_panel.ViewPanel:
        calls.append(("spectral", kwargs["single_plot_function"]))
        return _empty_view()

    def spatial(**_kwargs: Any) -> ui_panel.ViewPanel:
        calls.append(("spatial", "shared"))
        return _empty_view()

    def transient_case(**kwargs: Any) -> ui_panel.ViewPanel:
        calls.append(("transient_case", kwargs["kind"]))
        return _empty_view()

    def completion(**_kwargs: Any) -> ui_panel.ViewPanel:
        calls.append(("completion", "consolidated"))
        return _empty_view()

    monkeypatch.setattr(panel.viewers, "make_statistics_view", statistics)
    monkeypatch.setattr(panel.viewers, "make_spectral_view", spectral)
    monkeypatch.setattr(panel.viewers, "make_spatial_case_view", spatial)
    monkeypatch.setattr(panel.viewers, "make_transient_case_view", transient_case)
    monkeypatch.setattr(panel.viewers, "make_completion_target_view", completion)

    factories = panel._view_factories(catalog, state)
    visible = presentation.eda_sections_for_capabilities(tuple(catalog.capabilities))
    expected_keys = tuple(plot.key for section in visible for plot in section.plots)
    assert set(factories) == set(expected_keys)
    assert expected_keys[-1] == "transient_completion_target"
    assert "transient_schedule_boundaries" not in expected_keys
    for key in expected_keys:
        parameters = signature(factories[key]).parameters
        assert tuple(parameters) == (
            "export_state",
            "export_plot_name",
            "export_title",
        )
        assert all(parameter.kind is not Parameter.VAR_KEYWORD for parameter in parameters.values())
        result = factories[key](
            export_state={
                "fig": None,
                "figures": (),
                "plot_name": key,
                "title": key,
            },
            export_plot_name=key,
            export_title=key,
        )
        assert isinstance(result, ui_panel.ViewPanel)
    assert len(calls) == len(expected_keys)
    assert calls.count(("completion", "consolidated")) == 1


def _walk(widget: widgets.Widget) -> tuple[widgets.Widget, ...]:
    """Return one widget tree in deterministic depth-first order."""
    descendants = [widget]
    for child in getattr(widget, "children", ()):
        descendants.extend(_walk(child))
    return tuple(descendants)


def test_lazy_panel_reuses_views_and_tracks_selection_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct views lazily, preserve them on reopen, and rebuild tabs for capabilities below one persistent selector."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    factory_calls: list[str] = []

    def factory(key: str) -> Callable[..., ui_panel.ViewPanel]:
        def invoke(**_kwargs: Any) -> ui_panel.ViewPanel:
            factory_calls.append(key)
            return _empty_view()

        return invoke

    monkeypatch.setattr(
        panel,
        "_view_factories",
        lambda *_args, **_kwargs: {plot.key: factory(plot.key) for section in presentation.EDA_SECTIONS for plot in section.plots},
    )
    outer = panel.build_eda_panel(catalog=catalog, selection_state=state)
    (collapsed,) = outer.children
    assert isinstance(collapsed, widgets.VBox)
    assert factory_calls == []

    next(item for item in _walk(collapsed) if isinstance(item, widgets.Button)).click()
    (opened_panel,) = outer.children
    assert isinstance(opened_panel, widgets.VBox)
    assert opened_panel is not collapsed
    tree = _walk(opened_panel)
    tabs = next(item for item in tree if isinstance(item, widgets.Tab))
    header = _dataset_header(tree)
    all_sections = tabs.children
    calls_after_first_open = tuple(factory_calls)
    assert "2 von 2 gewählt" in header.description

    close = next(item for item in tree if isinstance(item, widgets.Button) and item.description == "Close")
    close.click()
    assert outer.children == (collapsed,)
    next(item for item in _walk(collapsed) if isinstance(item, widgets.Button)).click()
    assert outer.children == (opened_panel,)
    assert tuple(factory_calls) == calls_after_first_open

    steady_key = catalog.views[0].key
    state.select_datasets((steady_key,))
    assert tabs.children != all_sections
    steady_sections = tabs.children
    assert _dataset_header(_walk(opened_panel)) is header
    assert "1 von 2 gewählt" in header.description
    state.select_datasets(())
    assert tabs.children == steady_sections
    assert "0 von 2 gewählt" in header.description
    state.select_datasets(tuple(view.key for view in catalog.views))
    assert tabs.children != steady_sections
    assert _dataset_header(_walk(opened_panel)) is header
    assert "2 von 2 gewählt" in header.description


def _dataset_header(tree: tuple[widgets.Widget, ...]) -> widgets.ToggleButton:
    """Return the disclosure header of the panel's dataset selector."""
    return next(item for item in tree if isinstance(item, widgets.ToggleButton) and "Datensätze" in item.description)


def test_published_tensor_view_shows_load_failure_and_clears_export() -> None:
    """A failing package load is shown in the view and never leaves the previous figure exportable."""
    manifest = {
        "dataset_id": "pkg",
        "dataset_name": "Package",
        "dataset_view": "steady_flow",
        "source_case_identities": [{"package_case_id": "c1", "source": {"simulation_case_id": "s1"}}],
    }

    def failing_loader() -> pd.DataFrame:
        message = "package unreadable"
        raise RuntimeError(message)

    package = selection.PublishedDatasetEDAView(manifest=manifest, case_limit=None, loader=failing_loader)
    catalog = selection.GeneratedOutputEDACatalog([package], source_catalog=sources.PublishedDatasetEDACatalog((manifest,), ()))
    stale = Figure()
    export_state: dict[str, Any] = {"fig": stale, "figures": (stale,), "plot_name": "old", "title": "old"}

    view = viewers.make_published_tensor_view(
        catalog=catalog,
        selection_state=selection.GeneratedOutputSelectionState(catalog),
        export_state=export_state,
        export_plot_name="tensor",
        export_title="Tensor",
    )

    assert "View failed — RuntimeError: package unreadable" in view.failure.widget.value
    assert export_state["fig"] is None
    assert export_state["figures"] == ()
    assert (export_state["plot_name"], export_state["title"]) == ("tensor", "Tensor")


def test_channels_check_only_their_default_at_the_first_offer_then_keep_the_retained_choice() -> None:
    """The first offer checks the default channels that exist; once a choice is retained, a rebind keeps it and checks new channels."""
    state = selection.GeneratedOutputSelectionState(_catalog())
    channels = viewers._Channels("Fields", state, "transient_snapshot_fields", ui_controls.StatusLine(), default=("T", "phi", "w_gr"))
    channels.rebind(("T", "phi", "w_surf", "w_gr", "rho"), {})
    assert channels.selected == ("T", "phi", "w_gr")
    channels.observe(lambda: None)
    channels.checks.boxes["phi"].value = False
    channels.rebind(("T", "phi", "w_gr", "p"), {})
    assert channels.selected == ("T", "w_gr", "p")


def test_deactivated_view_ignores_shared_selection_until_reactivated() -> None:
    """A hidden view neither loads nor renders on shared changes and renders once on reactivation."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    renders: list[tuple[str, ...]] = []

    def plot(*, datasets: dict[str, pd.DataFrame], **kwargs: Any) -> ui_panel.ViewPanel:
        renders.append(tuple(datasets))
        return _case_count_viewer(datasets=datasets, **kwargs)

    stale = Figure()
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "stats", "title": "Stats"}
    view = viewers.make_statistics_view(
        catalog=catalog,
        selection_state=state,
        plot_function=plot,
        required_capabilities=(),
        export_state=export_state,
        export_plot_name="stats",
        export_title="Stats",
    )
    assert len(renders) == 1

    view.deactivate()
    state.select_datasets((catalog.views[0].key,))
    assert len(renders) == 1

    view.activate()
    assert renders[1:] == [(catalog.views[0].label,)]

    export_state["fig"] = stale
    state.select_datasets(())
    assert len(renders) == 2
    assert view.status.text == "Select at least one dataset."
    assert export_state["fig"] is None


def test_failing_statistics_plot_shows_its_own_error() -> None:
    """A statistics plot that raises before building its viewer shows that error in place; nothing else escapes."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((catalog.views[0].key,))

    def plot(**_kwargs: Any) -> ui_panel.ViewPanel:
        message = "statistics reduction failed"
        raise ValueError(message)

    view = viewers.make_statistics_view(
        catalog=catalog,
        selection_state=state,
        plot_function=plot,
        required_capabilities=(),
        export_state={"fig": None, "figures": (), "plot_name": "stats", "title": "Stats"},
        export_plot_name="stats",
        export_title="Stats",
    )

    assert view.failure.shown
    assert "statistics reduction failed" in view.failure.widget.value


def test_failing_dataset_load_is_shown_in_the_view_and_clears_the_export() -> None:
    """A newly selected dataset that cannot load fails inside the view, not in the selection callback, and its stale figure goes."""
    catalog = _catalog(unreadable="transient_drying")
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((catalog.views[0].key,))
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "stats", "title": "Stats"}
    view = viewers.make_statistics_view(
        catalog=catalog,
        selection_state=state,
        plot_function=_case_count_viewer,
        required_capabilities=(),
        export_state=export_state,
        export_plot_name="stats",
        export_title="Stats",
    )
    assert not view.failure.shown
    export_state["fig"] = Figure()

    state.select_datasets(tuple(item.key for item in catalog.views))

    assert view.failure.shown
    assert "transient_drying dataset unreadable" in view.failure.widget.value
    assert export_state["fig"] is None
    assert (export_state["plot_name"], export_state["title"]) == ("stats", "Stats")


def test_capability_change_replaces_the_tabs_after_the_selector_callback_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selection notification finishes with the old tabs in place; the selector then replaces them once."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    monkeypatch.setattr(
        panel,
        "_view_factories",
        lambda *_args, **_kwargs: {plot.key: (lambda **_kwargs: _empty_view()) for section in presentation.EDA_SECTIONS for plot in section.plots},
    )
    outer = panel.build_eda_panel(catalog=catalog, selection_state=state)
    next(item for item in _walk(outer) if isinstance(item, widgets.Button)).click()
    tree = _walk(outer)
    tabs = next(item for item in tree if isinstance(item, widgets.Tab))
    before = tabs.children
    seen: list[tuple[widgets.Widget, ...]] = []
    state.observe(lambda _current: seen.append(tabs.children))

    next(item for item in tree if isinstance(item, widgets.Checkbox) and item.description == catalog.views[1].label).value = False

    assert seen == [before]
    assert tabs.children != before


def test_deactivation_supersedes_a_render_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """A render that is still running when its view is hidden cannot publish into the shared export state."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((catalog.views[1].key,))
    holder: list[ui_panel.ViewPanel] = []
    calls = 0

    def plot(**_kwargs: Any) -> Figure:
        nonlocal calls
        calls += 1
        if calls == 2:
            holder[0].deactivate()
        return Figure()

    monkeypatch.setattr(viewers.plots.transient, "plot_completion_target_analysis", plot)
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "completion", "title": "Completion"}
    view = viewers.make_completion_target_view(
        catalog=catalog, selection_state=state, export_state=export_state, export_plot_name="completion", export_title="Completion"
    )
    holder.append(view)
    first = export_state["fig"]
    assert isinstance(first, Figure)

    state.select_datasets(tuple(item.key for item in catalog.views))

    assert calls == 2
    assert export_state["fig"] is first


def test_a_dataset_keeps_its_line_style_and_hatch_when_another_is_deselected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The viewer resolves styles over the whole catalog: with A deselected, batch B of the same material stays dashed and hatched."""
    catalog = _catalog(specs=(("transient_drying", "A"), ("transient_drying", "B")))
    state = selection.GeneratedOutputSelectionState(catalog)
    drawn: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def plot(*, datasets: dict[str, pd.DataFrame], styles: dict[str, Any]) -> Figure:
        drawn.append((tuple(datasets), styles))
        return Figure()

    monkeypatch.setattr(viewers.plots.transient, "plot_completion_target_analysis", plot)
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "completion", "title": "Completion"}
    viewers.make_completion_target_view(
        catalog=catalog, selection_state=state, export_state=export_state, export_plot_name="completion", export_title="Completion"
    )
    state.select_datasets((catalog.views[1].key,))

    assert [names for names, _styles in drawn] == [("A", "B"), ("B",)]
    both, alone = drawn[0][1]["B"], drawn[1][1]["B"]
    assert alone == both
    assert alone.linestyle != "-"
    assert alone.hatch


def test_a_published_package_of_one_material_takes_that_materials_color_without_loading() -> None:
    """The material comes from the manifest: one material colors the package like a generated batch of it; several do not."""

    def package(name: str, materials: list[str]) -> selection.PublishedDatasetEDAView:
        manifest = {
            "dataset_id": name,
            "dataset_name": name,
            "dataset_view": "transient_drying",
            "materials": materials,
            "source_case_identities": [{"source": {"simulation_case_id": "1" * 64}}],
        }
        return selection.PublishedDatasetEDAView(manifest=manifest, case_limit=None, loader=pd.DataFrame)

    views = (package("single", ["lentil"]), package("mixed", ["lentil", "chickpea"]))
    styles = viewers._dataset_styles(cast("Any", SimpleNamespace(views=views)))
    assert styles[views[0].label].color == viewers.plots.transient.visual_semantics.material_color("lentil")
    assert styles[views[1].label].color != styles[views[0].label].color
    assert not views[0].is_loaded


def test_a_published_package_label_shows_its_identity_once_when_the_name_equals_it() -> None:
    def label(name: str) -> str:
        identities = [{"package_case_id": "c1", "source": {"simulation_case_id": "s1"}}]
        manifest = {"dataset_id": "pkg", "dataset_name": name, "dataset_view": "steady_flow", "source_case_identities": identities}
        return selection.PublishedDatasetEDAView(manifest=manifest, case_limit=None, loader=pd.DataFrame).label

    assert label("pkg") == "pkg"
    assert label("Package") == "Package [pkg]"


def test_statistics_view_keeps_its_case_count_across_activations() -> None:
    """The nested case-count viewer is rebuilt for the selected datasets but starts at the count the user had chosen."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    nested: list[ui_panel.ViewPanel] = []

    def plot(**kwargs: Any) -> ui_panel.ViewPanel:
        nested.append(_case_count_viewer(**kwargs))
        return nested[-1]

    def count() -> widgets.IntSlider:
        return next(item for item in _walk(nested[-1].widget) if isinstance(item, widgets.IntSlider))

    view = viewers.make_statistics_view(
        catalog=catalog,
        selection_state=state,
        plot_function=plot,
        required_capabilities=(),
        export_state=None,
        export_plot_name="stats",
        export_title="Stats",
    )
    assert count().value == 5
    count().value = 2

    view.deactivate()
    view.activate()
    assert count().value == 2

    state.select_datasets((catalog.views[0].key,))
    assert count().value == 2

    count().value = 20
    view.deactivate()
    view.activate()
    assert count().value == 20


type _HookedStatistics = tuple[
    ui_panel.ViewPanel, selection.GeneratedOutputSelectionState, selection.GeneratedOutputEDACatalog, dict[str, Any], list[ui_panel.ViewPanel]
]


def _statistics_view_with_nested_render_hook(on_render: Callable[[int], None]) -> _HookedStatistics:
    """Build a statistics view over a real nested viewer whose figure callback reports its call number to ``on_render``."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((catalog.views[0].key,))
    nested: list[ui_panel.ViewPanel] = []
    calls = 0

    def figure(**_kwargs: Any) -> Figure:
        nonlocal calls
        calls += 1
        on_render(calls)
        return Figure()

    def plot(**kwargs: Any) -> ui_panel.ViewPanel:
        nested.append(_case_count_viewer(plot=figure, **kwargs))
        return nested[-1]

    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "stats", "title": "Stats"}
    view = viewers.make_statistics_view(
        catalog=catalog,
        selection_state=state,
        plot_function=plot,
        required_capabilities=(),
        export_state=export_state,
        export_plot_name="stats",
        export_title="Stats",
    )
    return view, state, catalog, export_state, nested


def test_hiding_the_statistics_view_supersedes_a_nested_render_in_flight() -> None:
    """A figure that the nested viewer is still drawing when the statistics view is hidden is never exported."""
    holder: list[ui_panel.ViewPanel] = []

    def on_render(number: int) -> None:
        if number == 2:
            holder[0].deactivate()

    view, _state, _catalog, export_state, nested = _statistics_view_with_nested_render_hook(on_render)
    holder.append(view)
    first = export_state["fig"]
    assert isinstance(first, Figure)

    next(item for item in _walk(nested[-1].widget) if isinstance(item, widgets.IntSlider)).value = 2

    assert export_state["fig"] is first


def test_replacing_the_nested_viewer_supersedes_the_one_still_being_built() -> None:
    """A nested viewer whose first render runs while its statistics view is hidden publishes nothing."""
    holder: list[ui_panel.ViewPanel] = []

    def on_render(number: int) -> None:
        if number == 2:
            holder[0].deactivate()

    view, state, catalog, export_state, _nested = _statistics_view_with_nested_render_hook(on_render)
    holder.append(view)
    first = export_state["fig"]
    assert isinstance(first, Figure)

    state.select_datasets(tuple(item.key for item in catalog.views))

    assert export_state["fig"] is first
