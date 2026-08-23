# ruff: noqa: S101, SLF001, PLR2004
"""Protect the single capability-adaptive generated-output panel and registry."""

from __future__ import annotations

from inspect import Parameter, signature
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import pandas as pd

from src.analysis.eda import eda_panel as panel
from src.analysis.eda import eda_selection as selection
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_viewers as viewers
from src.analysis.presentation import analysis_presentation_registry as presentation
from src.analysis.ui import analysis_ui_notebook as notebook_ui

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def _catalog() -> selection.GeneratedOutputEDACatalog:
    """Return one steady and one transient profile-native view in one catalog."""
    case = SimpleNamespace(case_index=1, case_id="case_0001")
    views = []
    for task_id, label in (
        ("steady_flow", "Airflow · Reference · fg · ID"),
        (
            "transient_drying",
            "Drying · Lentil · fg · F OOD",
        ),
    ):
        frame = pd.DataFrame(
            {"value": [1.0]},
            index=pd.Index(["case_0001"], name="sample_id"),
        )
        frame.attrs["task_id"] = task_id
        batch = SimpleNamespace(
            simulation_profile=task_id,
            available_learning_views=(task_id,),
            batch_id=f"{task_id}-batch",
            batch_storage_name=f"{task_id}-storage",
            campaign_purpose="family_generalization",
            material_role="id_source",
            evaluation_regime=None,
            cases=(case,),
        )
        views.append(
            selection.GeneratedOutputEDAView(
                label=label,
                batch=cast("Any", batch),
                case_limit=None,
                loader=lambda current=frame: current,
            )
        )
    source_catalog = sources.GeneratedOutputEDACatalog(
        batches=(),
        issues=(),
        discovered_batch_count=0,
        complete_batch_count=0,
        partial_batch_count=0,
        total_issue_count=0,
    )
    return selection.GeneratedOutputEDACatalog(views, source_catalog=source_catalog)


def _empty_view() -> viewers.ActivatableView:
    """Return one inert view with the production lifecycle surface."""
    return viewers.ActivatableView((widgets.Label("fixture"),), activate=lambda: None)


def _walk(widget: widgets.Widget) -> tuple[widgets.Widget, ...]:
    """Return one widget tree in deterministic depth-first order."""
    descendants = [widget]
    for child in getattr(widget, "children", ()):
        descendants.extend(_walk(child))
    return tuple(descendants)


def test_factories_cover_capability_registry_with_one_consolidated_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke every visible semantic factory with one consolidated diagnostic."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    calls: list[tuple[str, object]] = []

    def statistics(**kwargs: Any) -> viewers.ActivatableView:
        calls.append(("statistics", kwargs["plot_function"]))
        return _empty_view()

    def spectral(**kwargs: Any) -> viewers.ActivatableView:
        calls.append(("spectral", kwargs["single_plot_function"]))
        return _empty_view()

    def spatial(**_kwargs: Any) -> viewers.ActivatableView:
        calls.append(("spatial", "shared"))
        return _empty_view()

    def transient_case(**kwargs: Any) -> viewers.ActivatableView:
        calls.append(("transient_case", kwargs["kind"]))
        return _empty_view()

    def completion(**_kwargs: Any) -> viewers.ActivatableView:
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
        assert isinstance(result, viewers.ActivatableView)
    assert len(calls) == len(expected_keys)
    assert calls.count(("completion", "consolidated")) == 1


def test_one_original_launcher_opens_one_capability_adaptive_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one lazy launcher, one dataset selector, and no task control."""
    catalog = _catalog()
    state = selection.GeneratedOutputSelectionState(catalog)
    factory_calls: list[str] = []

    def factory(key: str) -> Callable[..., viewers.ActivatableView]:
        def invoke(**_kwargs: Any) -> viewers.ActivatableView:
            factory_calls.append(key)
            return _empty_view()

        return invoke

    monkeypatch.setattr(
        panel,
        "_view_factories",
        lambda *_args: {plot.key: factory(plot.key) for section in presentation.EDA_SECTIONS for plot in section.plots},
    )
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)

    outer = panel.build_eda_panel(catalog=catalog, selection_state=state)
    assert isinstance(outer, notebook_ui.LazyTabbedPanelOutput)
    assert len(displayed) == 1
    launcher = displayed[-1]
    assert isinstance(launcher, widgets.Button)
    assert launcher.description == "Generated-output EDA - Open"
    assert launcher.button_style == "primary"

    launcher.click()
    opened_panel = displayed[-1]
    assert isinstance(opened_panel, widgets.VBox)
    tree = _walk(opened_panel)
    assert sum(isinstance(item, widgets.HTML) and item.value == "<b>Datasets</b>" for item in tree) == 1
    dataset_boxes = tuple(item for item in tree if isinstance(item, widgets.Checkbox))
    assert len(dataset_boxes) == 2
    assert tuple(box.description for box in dataset_boxes) == tuple(view.label for view in catalog.views)
    assert not any(getattr(item, "description", "") == "Task:" or (isinstance(item, widgets.HTML) and "Task:" in item.value) for item in tree)
    assert outer.tabs is not None
    assert len(outer.tabs.children) == 3
    last_dropdown = outer.tabs.children[-1].children[0]
    assert last_dropdown.options[-1][0].startswith("3-3.")
    assert "Completion and target attainment" in last_dropdown.options[-1][0]
    calls_after_first_open = tuple(factory_calls)

    close = next(item for item in tree if isinstance(item, widgets.Button) and item.description == "Close")
    close.click()
    assert displayed[-1] is launcher
    launcher.click()
    assert displayed[-1] is opened_panel
    assert tuple(factory_calls) == calls_after_first_open

    steady_key = catalog.views[0].key
    state.select_datasets((steady_key,))
    assert len(outer.tabs.children) == 2
    state.select_datasets(tuple(view.key for view in catalog.views))
    assert len(outer.tabs.children) == 3


def test_completion_viewer_declares_every_dependency_explicitly() -> None:
    """Keep the consolidated callback free of hidden keyword dependencies."""
    parameters = signature(viewers.make_completion_target_view).parameters
    assert tuple(parameters) == (
        "catalog",
        "selection_state",
        "export_state",
        "export_plot_name",
        "export_title",
    )
    assert all(parameter.kind is not Parameter.VAR_KEYWORD for parameter in parameters.values())
