"""Protect transient completed-output EDA semantics and steady compatibility.

Responsibilities:
  - Check completed-source dispatch, exact-time trajectories, target outcomes, and spatial maps
  - Check the aggregate state rows over the drying phase: reached cases only at their nearest stored states, censored ones
    named, schedules kept in hours, and the phase offered only where every case carries completion evidence

Design principles:
  - Physical support and steady compatibility survive display transformations

This module does NOT:
  - Forward-fill completed cases or change source arrays
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure

from src import common, datasets, domain, generation
from src.analysis.catalog import analysis_catalog_transient as catalog_transient
from src.analysis.eda import dataframe as eda_dataframe
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda import eda_selection as selection
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_transient as transient
from src.analysis.eda import eda_viewers as viewers
from src.analysis.eda import eda_workspace as workspace
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.analysis.ui import analysis_ui_controls as ui_controls
from src.datasets.packages import dataset_packages_trajectory as trajectory

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.transforms import Bbox


@pytest.fixture
def transient_frame(transient_package: dict[str, Any]) -> pd.DataFrame:
    """Return one fully interpreted current Source case EDA frame."""
    storage_root = transient_package["root"]
    source_selector = transient_package["source_selectors"][0]
    rows = generation.publication.catalog.list_campaign_cases("transient_drying", transient_package["source_campaign"], storage_root=storage_root)
    row = next(row for row in rows if all(row[key] == value for key, value in source_selector.items()))
    source_case = sources.GeneratedOutputEDACase(case_id=str(row["case_id"]), case_index=1, row=dict(row))
    batch = sources.GeneratedOutputEDABatch(
        generation_root=common.paths.get_generation_root(storage_root=storage_root),
        batch_id=f"transient_drying__{source_selector['campaign_name']}__{row['material_family']}",
        campaign_name=str(source_selector["campaign_name"]),
        source_name=transient_package["source_campaign"],
        campaign_digest=str(source_selector["campaign_digest"]),
        campaign_purpose="source_generation",
        simulation_profile="transient_drying",
        available_learning_views=("transient_drying",),
        material_family=str(row["material_family"]),
        domain_profile=str(row["domain_profile"]),
        sampling_regime=str(row["sampling_regime"]),
        cases=(source_case,),
        candidate_rows=(dict(row),),
        storage_root=storage_root,
    )
    task = domain.tasks.registry.get_task("transient_drying")
    frame, logs = eda_dataframe.generate_eda_dataframe_from_completed_cases(batch, task=task, show_progress=False, max_cases=1)
    assert any("completed trajectories" in line for line in logs)
    assert "case_family" not in frame.iloc[0]["meta"]
    assert frame.iloc[0]["meta"]["parameter_units"]["T_init"] == "K"
    return frame


def _tight_bbox(axis: Axes, renderer: RendererBase) -> Bbox:
    """Return the axis' tight bounding box, which exists once the axis is drawn."""
    bbox = axis.get_tightbbox(renderer)
    assert bbox is not None
    return bbox


def _row_start(axis: Axes) -> int:
    """Return the first grid row of an axis that sits in a grid."""
    spec = axis.get_subplotspec()
    assert spec is not None
    return spec.rowspan.start


def _widget_descriptions(widget: Any) -> set[str]:
    """Collect descriptions recursively from one widget tree."""
    descriptions = {str(widget.description)} if hasattr(widget, "description") else set()
    for child in getattr(widget, "children", ()):
        descriptions.update(_widget_descriptions(child))
    return descriptions


def _widgets_by_description(widget: Any, description: str) -> list[Any]:
    """Collect widgets with one description from a nested test panel."""
    matches = [widget] if getattr(widget, "description", None) == description else []
    for child in getattr(widget, "children", ()):
        matches.extend(_widgets_by_description(child, description))
    return matches


def _widget_by_description(widget: Any, description: str) -> Any:
    """Return one uniquely described widget from a nested test panel."""
    matches = _widgets_by_description(widget, description)
    if len(matches) != 1:
        raise LookupError(description)
    return matches[0]


def _is_shown(widget: Any, description: str, hidden: bool = False) -> bool:
    """Return whether the described widget exists and no box on its way from ``widget`` is hidden."""
    hidden = hidden or widget.layout.display == "none"
    if getattr(widget, "description", None) == description:
        return not hidden
    return any(_is_shown(child, description, hidden) for child in getattr(widget, "children", ()))


def _adaptive_catalog(
    frames: tuple[tuple[str, pd.DataFrame], ...],
) -> selection.GeneratedOutputEDACatalog:
    """Build one test-owned lazy transient catalog without storage discovery."""
    case = SimpleNamespace(case_index=1, case_id="case_0001")
    views = []
    for position, (label, frame) in enumerate(frames):
        frame.attrs["task_id"] = "transient_drying"
        batch = SimpleNamespace(
            simulation_profile="transient_drying",
            available_learning_views=("transient_drying",),
            batch_id=f"batch-{position}",
            campaign_name=f"campaign-{position}",
            campaign_digest="a" * 64,
            campaign_purpose="family_generalization",
            domain_profile="reference_range",
            material_family="lentil",
            selected_case_count=1,
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
        discovered_campaign_count=0,
        discovered_candidate_count=0,
        total_issue_count=0,
    )
    return selection.GeneratedOutputEDACatalog(
        views,
        source_catalog=source_catalog,
    )


def test_physical_time_channel_and_view_control_contracts(
    transient_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve exact time, channel state, and aggregate/single-case behavior."""
    assert transient.resolve_dynamic_channels(transient_frame) == (
        "T",
        "phi",
        "w_surf",
        "w_int",
    )

    render_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        viewers.ui.viewers,
        "render_view",
        lambda *_args, **kwargs: render_calls.append(kwargs),
    )
    catalog = _adaptive_catalog((("Drying · Lentil · fg · ID", transient_frame),))
    state = selection.GeneratedOutputSelectionState(catalog)
    snapshot_viewer = viewers.make_transient_case_view(
        catalog=catalog,
        selection_state=state,
        kind="snapshot",
        export_state={"fig": None, "plot_name": "snapshot", "title": "snapshot"},
        export_plot_name="snapshot",
        export_title="snapshot",
    ).widget
    assert not _is_shown(snapshot_viewer, "Lock color scale")
    assert _is_shown(snapshot_viewer, "Time [h]")
    time_control = _widget_by_description(snapshot_viewer, "Time [h]")
    assert not any(isinstance(widget, widgets.Play) for widget in _walk_widgets(snapshot_viewer))
    retained_channel = _widget_by_description(snapshot_viewer, "φ [% RH]")
    retained_channel.value = False
    time_control.value = 1.1
    assert time_control.value == 1.0
    assert retained_channel.value is False
    assert state.physical_time_selection("transient_snapshot") == 1.0
    assert render_calls

    trajectory_viewer = viewers.make_transient_case_view(
        catalog=catalog,
        selection_state=state,
        kind="trajectory",
        export_state=None,
        export_plot_name=None,
        export_title=None,
    ).widget
    scope = next(widget for widget in _walk_widgets(trajectory_viewer) if isinstance(widget, widgets.ToggleButtons))
    assert scope.value == "aggregate"
    assert _is_shown(trajectory_viewer, "Cases")
    assert not _is_shown(trajectory_viewer, "Case")
    scope.value = "single"
    assert _is_shown(trajectory_viewer, "Case")
    assert not _is_shown(trajectory_viewer, "Cases")
    assert not any(getattr(widget, "description", "") == "Time [h]" for widget in _walk_widgets(trajectory_viewer))


def _walk_widgets(widget: Any) -> tuple[Any, ...]:
    """Return one nested widget tree for structural control assertions."""
    children = tuple(getattr(widget, "children", ()))
    return (widget, *(descendant for child in children for descendant in _walk_widgets(child)))


def test_spectral_scope_and_orientation_keep_functional_state(
    transient_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switch spectral scope and pass the special orientation explicitly."""
    catalog = _adaptive_catalog((("Drying · Lentil · fg · ID", transient_frame),))
    state = selection.GeneratedOutputSelectionState(catalog)
    render_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        viewers.ui.viewers,
        "render_view",
        lambda *_args, **kwargs: render_calls.append(kwargs),
    )
    orientation = ui_controls.dropdown(("cross_stream_along_flow", "flow_across_cross_stream"), description="Orientation")

    viewer = viewers.make_spectral_view(
        catalog=catalog,
        selection_state=state,
        single_plot_function=cast("Any", lambda **_kwargs: None),
        aggregate_plot_function=cast("Any", lambda **_kwargs: None),
        semantic_controls={"orientation": orientation},
        export_state=None,
        export_plot_name=None,
        export_title=None,
    ).widget
    scope = next(widget for widget in _walk_widgets(viewer) if isinstance(widget, widgets.ToggleButtons))
    assert scope.value == "aggregate"
    assert _is_shown(viewer, "Cases")
    assert not _is_shown(viewer, "Case")
    orientation.value = "flow_across_cross_stream"
    assert render_calls[-1]["kwargs"]["orientation"] == "flow_across_cross_stream"
    scope.value = "single"
    assert _is_shown(viewer, "Case")
    assert not _is_shown(viewer, "Cases")


def test_mixed_spatial_selection_keeps_union_channels_without_fabrication(
    transient_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep transient union fields selected only for datasets that provide them."""
    reduced_row = copy.deepcopy(transient_frame.iloc[0].to_dict())
    reduced_row["static_fields"].pop("p")
    reduced_frame = pd.DataFrame([reduced_row], index=transient_frame.index.copy())
    reduced_frame.attrs = copy.deepcopy(transient_frame.attrs)
    categories = copy.deepcopy(reduced_frame.attrs["field_categories"])
    categories["static_spatial"] = tuple(field for field in categories["static_spatial"] if field != "p")
    reduced_frame.attrs["field_categories"] = categories
    catalog = _adaptive_catalog(
        (("Reference", transient_frame), ("Reduced", reduced_frame)),
    )
    state = selection.GeneratedOutputSelectionState(catalog)
    render_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        viewers.ui.viewers,
        "render_view",
        lambda *_args, **kwargs: render_calls.append(kwargs),
    )
    spatial_viewer = viewers.make_spatial_case_view(
        catalog=catalog,
        selection_state=state,
        export_state={"fig": None, "plot_name": "spatial", "title": "spatial"},
        export_plot_name="spatial",
        export_title="spatial",
    ).widget
    assert isinstance(_widget_by_description(spatial_viewer, "p [Pa]"), widgets.Checkbox)
    resolution = transient_plots.capabilities.resolve_fields(
        {"Reference": transient_frame, "Reduced": reduced_frame},
        view="spatial_map",
        requested=("p",),
    )
    assert resolution.datasets_by_field["p"] == ("Reference",)
    assert resolution.omitted_by_field["p"] == ("Reduced",)

    reduced_key = catalog.views[1].key
    state.select_datasets((reduced_key,))
    assert _widgets_by_description(spatial_viewer, "p [Pa]") == []
    state.select_datasets(tuple(view.key for view in catalog.views))
    assert isinstance(_widget_by_description(spatial_viewer, "p [Pa]"), widgets.Checkbox)
    assert render_calls


def test_failing_field_resolution_is_shown_in_the_view_and_clears_the_export(transient_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Channel resolution after a selection change fails inside the view, not in the selection callback."""
    monkeypatch.setattr(viewers.ui.viewers, "render_view", lambda *_args, **_kwargs: None)
    catalog = _adaptive_catalog((("Reference", transient_frame), ("Second", transient_frame)))
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((catalog.views[0].key,))
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "spatial", "title": "spatial"}
    view = viewers.make_spatial_case_view(
        catalog=catalog, selection_state=state, export_state=export_state, export_plot_name="spatial", export_title="spatial"
    )
    export_state["fig"] = Figure()

    def unreadable(*_args: Any, **_kwargs: Any) -> None:
        message = "fields unreadable"
        raise RuntimeError(message)

    monkeypatch.setattr(viewers.capabilities, "resolve_fields", unreadable)
    state.select_datasets(tuple(item.key for item in catalog.views))

    assert view.failure.shown
    assert "fields unreadable" in view.failure.widget.value
    assert export_state["fig"] is None


def test_failure_supersedes_a_render_in_flight(transient_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """A render that is still running when a newer selection fails cannot clear the failure or export the old selection."""
    catalog = _adaptive_catalog((("Reference", transient_frame), ("Second", transient_frame)))
    state = selection.GeneratedOutputSelectionState(catalog)
    state.select_datasets((catalog.views[0].key,))
    calls = 0

    def unreadable(*_args: Any, **_kwargs: Any) -> None:
        message = "fields unreadable"
        raise RuntimeError(message)

    def plot(**_kwargs: Any) -> Figure:
        nonlocal calls
        calls += 1
        if calls == 2:
            monkeypatch.setattr(viewers.capabilities, "resolve_fields", unreadable)
            state.select_datasets(tuple(item.key for item in catalog.views))
        return Figure()

    monkeypatch.setattr(viewers.plots.transient, "plot_spatial_field_comparison", plot)
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "spatial", "title": "spatial"}
    view = viewers.make_spatial_case_view(
        catalog=catalog, selection_state=state, export_state=export_state, export_plot_name="spatial", export_title="spatial"
    )
    assert calls == 1
    assert isinstance(export_state["fig"], Figure)

    _widget_by_description(view.widget, "Lock color scale").value = True

    assert calls == 2
    assert view.failure.shown
    assert export_state["fig"] is None


def test_published_tensor_view_steps_stored_transitions_with_the_time_navigator(
    transient_package: dict[str, Any],
) -> None:
    """A transient package shows the time navigator over its transition start times and names the transition it plots."""
    prepared = workspace.prepare_published_dataset_eda_workspace(storage_root=transient_package["root"], max_cases=1)
    assert prepared.selection_state is not None
    package = prepared.catalog.views[0]
    export_state: dict[str, Any] = {"fig": None, "figures": (), "plot_name": "tensor", "title": "Tensor"}

    view = viewers.make_published_tensor_view(
        catalog=prepared.catalog,
        selection_state=prepared.selection_state,
        export_state=export_state,
        export_plot_name="tensor",
        export_title="Tensor",
    )
    try:
        assert not view.failure.shown
        assert _is_shown(view.widget, "Time [h]")
        assert isinstance(export_state["fig"], Figure)
        time = package.load().iloc[0]["time"]
        starts = tuple(float(value) for value in time["transition_t_n_hours"])
        assert len(starts) > 1
        _widget_by_description(view.widget, "Time [h]").value = starts[-1]
        assert f"Time {starts[-1]:g} h" in view.status.text
        assert "→" not in view.status.text
        title = export_state["fig"].get_suptitle()
        assert f"{starts[-1]:g} h → {float(time['transition_t_n_plus_1_hours'][-1]):g} h" in title
    finally:
        plt.close("all")


def test_target_attainment_censoring_grouping_exclusions_and_runtime_separation(
    transient_frame: pd.DataFrame,
) -> None:
    """Keep canonical target, censoring, physical duration, and runtime distinct."""
    reached = copy.deepcopy(transient_frame.iloc[0].to_dict())
    reached["runtime"]["comsol_process_seconds"] = 12.0
    unreached = copy.deepcopy(reached)
    unreached["meta"]["material_family"] = "chickpea"
    unreached["completion"].update(
        {
            "target_reached": False,
            "right_censored": True,
            "time_to_target_hours": None,
            "final_wet_fraction": 0.07,
            "target_wet_fraction_limit": 0.05,
        }
    )
    frame = pd.DataFrame(
        [reached, unreached],
        index=pd.Index(["case_0001", "case_0002"], name="sample_id"),
    )
    frame.attrs = copy.deepcopy(transient_frame.attrs)
    frame.attrs.update(
        {
            "loaded_case_count": 2,
            "available_case_count": 3,
            "total_discovered_case_count": 3,
            "exclusion_reasons": {"bounded_prefix": 1},
        }
    )
    diagnostic = transient.target_attainment_diagnostic(frame)
    assert diagnostic.summary["reached_count"] == 1
    assert diagnostic.summary["unreached_count"] == 1
    assert diagnostic.summary["excluded_count"] == 1
    assert diagnostic.summary["reached_percentage"] == 50.0
    assert diagnostic.summary["reached_percentage_denominator"] == ("eligible_target_diagnostic_cases")
    assert diagnostic.reached_distribution["count"] == 1
    assert diagnostic.cases.loc[diagnostic.cases["case_id"] == "case_0002", "time_to_target_hours"].isna().all()
    gaps = dict(zip(diagnostic.cases["case_id"], diagnostic.cases["final_target_gap"], strict=True))
    assert gaps["case_0001"] < 0.0
    assert gaps["case_0002"] > 0.0
    assert diagnostic.groups["case_count"].sum() == 2
    assert diagnostic.exclusion_reasons == {"bounded_prefix": 1}
    assert diagnostic.cases["comsol_process_seconds"].eq(12.0).all()
    assert diagnostic.cases["queue_wait_seconds"].isna().all()

    del frame.attrs["failed_case_count"]
    with pytest.raises(ValueError, match=r"canonical_generation_hdf5.*failed_case_count"):
        transient.target_attainment_diagnostic(frame)


def test_steady_dataframe_behavior_remains_unchanged() -> None:
    """Retain steady field order, values, speed derivation, and metadata columns."""
    task = domain.tasks.registry.get_task("steady_flow")
    shape = (2, 3)
    row = {
        **{name: np.full(shape, index + 1.0, dtype=np.float32) for index, name in enumerate(task.input_names)},
        **{name: np.full(shape, index + 10.0, dtype=np.float32) for index, name in enumerate(task.output_names)},
        "meta": {"case_id": "case_0001"},
    }
    loaded = {
        "task": task,
        "sample_ids": ["case_0001"],
        "rows": [row],
        "available_case_count": 1,
        "total_discovered_case_count": 1,
        "failed_case_count": 0,
        "incomplete_case_count": 0,
        "invalid_case_count": 0,
        "exclusion_reasons": {},
        "case_accounting_scope": "test_owned_discovery",
        "analysis_representation": "steady_field_rows",
        "dataset_backend": "test_owned_backend",
        "generated_batch_identity": {"batch_manifest_identity_sha256": "f" * 64},
        "manifest_sha256": "d" * 64,
        "generation_root": Path("/synthetic"),
    }
    frame, _logs = eda_dataframe._steady_dataframe(loaded, batch_name="synthetic_steady", task=task)
    assert tuple(frame.columns) == (*task.input_names, *task.output_names, "meta", "U")
    np.testing.assert_allclose(frame.iloc[0]["U"], np.hypot(frame.iloc[0]["u"], frame.iloc[0]["v"]))
    assert frame.attrs["field_names"] == (*task.input_names, *task.output_names, "U")
    assert frame.attrs["spatial_shape"] == shape
    assert frame.attrs["failed_case_count"] == 0

    del loaded["failed_case_count"]
    with pytest.raises(ValueError, match=r"'synthetic_steady'.*failed_case_count"):
        eda_dataframe._steady_dataframe(loaded, batch_name="synthetic_steady", task=task)


def test_hdf5_and_pt_items_match_after_selected_source_cleanup(
    transient_package: dict[str, Any],
) -> None:
    """Produce equal semantic EDA from real HDF5/PT items, then use PT alone."""
    storage_root = transient_package["root"]
    dataset_id = transient_package["train_dataset"]
    index_path = common.paths.resolve_dataset_path("transient_drying", dataset_id, dataset_root=storage_root / "02_datasets")
    index = trajectory.load_transient_index(index_path)
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    positions = (0, 1, 2)
    physical = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=sampling,
        source_root=storage_root,
        sample_indices=positions,
    )
    try:
        hdf5_items = [physical[index] for index in range(len(physical))]
    finally:
        physical.close()
    hdf5_frame = transient.frame_from_transient_items(
        hdf5_items,
        backend="canonical_hdf5",
    )
    selected_source = storage_root / str(index["cases"][0]["source_relative_path"])
    selected_source.unlink()
    sharded = datasets.runtime.transient.TransientPTShardDataset(
        index_path,
        sampling=sampling,
        source_root=storage_root,
        sample_indices=positions,
    )
    try:
        pt_items = [sharded[index] for index in range(len(sharded))]
    finally:
        sharded.close()
    pt_frame = transient.frame_from_transient_items(
        pt_items,
        backend="pt_shards",
    )
    assert not selected_source.exists()
    for category in (
        "state_trajectories",
        "static_fields",
        "boundary_intervals",
        "scalar_conditioning",
    ):
        left = hdf5_frame.iloc[0][category]
        right = pt_frame.iloc[0][category]
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_allclose(left[name], right[name], rtol=0.0, atol=0.0)


def test_supported_schedule_stops_at_exact_final_time_without_mutating_source(
    transient_frame: pd.DataFrame,
) -> None:
    """Clip one schedule at exact simulated support and reject extrapolation."""
    row = transient_frame.loc["case_0001"]
    source_time = np.array(row["schedule"]["t"], copy=True)
    source_values = np.array(row["schedule"]["T_in_bc"], copy=True)
    exact_final = float(row["completion"]["physical_duration_hours"])

    series = transient.supported_schedule_series(
        transient_frame,
        "case_0001",
        "T_in_bc",
    )

    assert series.final_time_hours == pytest.approx(exact_final)
    assert series.physical_time_hours[-1] == pytest.approx(exact_final)
    assert np.all(series.physical_time_hours <= exact_final)
    assert series.values[-1] == pytest.approx(np.interp(exact_final, source_time, source_values))  # noqa: PD011 -- dataclass values is a NumPy array
    np.testing.assert_array_equal(row["schedule"]["t"], source_time)
    np.testing.assert_array_equal(row["schedule"]["T_in_bc"], source_values)
    with pytest.raises(ValueError, match="outside simulated support"):
        series.value_at(exact_final + 1.0e-6)


def test_aggregate_schedule_drops_completed_cases_without_forward_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate only schedules whose exact simulated support includes the time."""
    supported = {
        "short": transient.SupportedScheduleSeries(
            quantity="T_in_bc",
            unit="K",
            physical_time_hours=np.asarray((0.0, 1.0, 1.25)),
            values=np.asarray((290.0, 291.0, 291.25)),
            final_time_hours=1.25,
        ),
        "long": transient.SupportedScheduleSeries(
            quantity="T_in_bc",
            unit="K",
            physical_time_hours=np.asarray((0.0, 1.0, 2.0, 2.5)),
            values=np.asarray((300.0, 301.0, 302.0, 302.5)),
            final_time_hours=2.5,
        ),
    }
    monkeypatch.setattr(
        transient_plots.transient,
        "supported_schedule_series",
        lambda _frame, case_id, _quantity: supported[case_id],
    )
    frame = pd.DataFrame(index=pd.Index(("short", "long"), name="sample_id"))

    physical_time, q10, median, q90, contributors = transient_plots._aggregate_supported_schedule(
        frame,
        ("short", "long"),
        "T_in_bc",
    )

    np.testing.assert_allclose(physical_time, (0.0, 1.0, 1.25, 2.0, 2.5))
    np.testing.assert_array_equal(contributors, (2, 2, 2, 1, 1))
    assert physical_time[-1] == pytest.approx(2.5)
    assert median[-1] == pytest.approx(302.5)
    assert q10[-1] == pytest.approx(302.5)
    assert q90[-1] == pytest.approx(302.5)


def test_inlet_pressure_profile_uses_the_authoritative_spatial_coordinate(
    transient_frame: pd.DataFrame,
) -> None:
    """Extract the inlet-pressure line on its stored spatial support."""
    row = cast("pd.Series[Any]", transient_frame.loc["case_0001"])
    inlet_coordinate, pressure = capabilities.inlet_pressure_boundary(
        transient_frame,
        row,
    )
    static = row["static_fields"]

    np.testing.assert_array_equal(inlet_coordinate, static["x"][0, :])
    np.testing.assert_array_equal(pressure, static["p_in_bc"][0, :])


def test_trajectory_time_views_split_exact_support_at_one_hour() -> None:
    """Split at one hour without adding or changing stored coordinates."""
    physical_time = np.asarray((0.0, 0.25, 1.0, 2.5, 8.0))
    values = np.asarray((10.0, 11.0, 12.0, 13.0, 14.0))
    original_time = physical_time.copy()
    original_values = values.copy()

    main, startup = transient_plots._split_physical_time_support(
        physical_time,
        values,
    )

    np.testing.assert_array_equal(main[0], (1.0, 2.5, 8.0))
    np.testing.assert_array_equal(main[1][0], (12.0, 13.0, 14.0))
    np.testing.assert_array_equal(startup[0], (0.0, 0.25, 1.0))
    np.testing.assert_array_equal(startup[1][0], (10.0, 11.0, 12.0))
    assert set(main[0]).union(startup[0]).issubset(set(physical_time))
    np.testing.assert_array_equal(physical_time, original_time)
    np.testing.assert_array_equal(values, original_values)


def test_trajectory_view_draws_its_main_and_startup_windows_through_the_shared_time_helper(transient_frame: pd.DataFrame) -> None:
    """Every row has the helper's two windows: titles on the top row, main from hour 1, startup 0-1 h on a shared y, time label at the bottom."""
    label = "Drying · Lentil · ID"
    figure = transient_plots.plot_state_trajectory_comparison(
        datasets={label: transient_frame}, case_ids={label: "case_0001"}, styles=transient_plots.batch_styles({label: None}), channels=("T",)
    )
    try:
        windows = transient_plots.time_axis.STARTUP_WINDOWS
        grid = [
            axis for axis in figure.axes if axis.axison and (spec := axis.get_subplotspec()) is not None and spec.get_gridspec().ncols == len(windows)
        ]
        main, startup = grid[0::2], grid[1::2]  # row-major: the main and the startup axis of each row
        assert len(main) == len(startup) > 1
        assert [main[0].get_title(), startup[0].get_title()] == [window.title for window in windows]
        assert all(axis.get_xlim()[0] == 1.0 for axis in main)
        assert all(axis.get_xlim() == (0.0, 1.0) for axis in startup)
        for main_axis, startup_axis in zip(main, startup, strict=True):
            assert main_axis.get_shared_y_axes().joined(main_axis, startup_axis)
            assert not startup_axis.yaxis.get_tick_params()["labelleft"]
        assert all(axis.get_xlabel() == "" and not axis.xaxis.get_tick_params()["labelbottom"] for axis in (*main[:-1], *startup[:-1]))
        assert main[-1].get_xlabel().startswith("Time")
        assert startup[-1].get_xlabel() == "Time [h]"
    finally:
        plt.close(figure)


def test_changed_transient_plot_builders_preserve_authoritative_time_support(
    transient_frame: pd.DataFrame,
) -> None:
    """Build the changed plot paths without drawing or resampling source time."""
    row = transient_frame.loc["case_0001"]
    stored_time = np.array(row["time"]["regular_state_hours"], copy=True)
    schedule_time = np.array(row["schedule"]["t"], copy=True)
    figures = (
        transient_plots.plot_spatial_field_comparison(
            datasets={"Drying · Lentil · ID": transient_frame},
            case_ids={"Drying · Lentil · ID": "case_0001"},
            fields=("T",),
            physical_time_hours=1.0,
        ),
        transient_plots.plot_state_snapshot_comparison(
            datasets={"Drying · Lentil · ID": transient_frame},
            case_ids={"Drying · Lentil · ID": "case_0001"},
            physical_time_hours=1.0,
            styles=transient_plots.batch_styles({"Drying · Lentil · ID": None}),
            channels=("T",),
        ),
        transient_plots.plot_state_trajectory_comparison(
            datasets={"Drying · Lentil · ID": transient_frame},
            case_ids={"Drying · Lentil · ID": "case_0001"},
            styles=transient_plots.batch_styles({"Drying · Lentil · ID": None}),
            channels=("T",),
        ),
    )
    try:
        np.testing.assert_array_equal(
            row["time"]["regular_state_hours"],
            stored_time,
        )
        np.testing.assert_array_equal(row["schedule"]["t"], schedule_time)
    finally:
        for figure in figures:
            plt.close(figure)


def test_spatial_map_grid_has_a_unit_bar_per_map_an_empty_unavailable_cell_and_grows_with_its_datasets(transient_frame: pd.DataFrame) -> None:
    """Rows are fields and columns datasets on the common geometry: every map has its own unit bar, an unavailable cell stays empty."""
    reduced_row = copy.deepcopy(transient_frame.iloc[0].to_dict())
    reduced_row["static_fields"].pop("p")
    reduced = pd.DataFrame([reduced_row], index=transient_frame.index.copy())
    reduced.attrs = copy.deepcopy(transient_frame.attrs)
    categories = copy.deepcopy(reduced.attrs["field_categories"])
    categories["static_spatial"] = tuple(field for field in categories["static_spatial"] if field != "p")
    reduced.attrs["field_categories"] = categories
    ids = {"A": "case_0001", "B": "case_0001", "C": "case_0001"}

    def draw(frames: dict[str, pd.DataFrame]) -> Figure:
        case_ids = {label: ids[label] for label in frames}
        return transient_plots.plot_spatial_field_comparison(datasets=frames, case_ids=case_ids, fields=("T", "p"), physical_time_hours=1.0)

    figure = draw({"A": transient_frame, "B": reduced})
    wider = draw({"A": transient_frame, "B": reduced, "C": transient_frame})
    try:
        maps = [axis for axis in figure.axes if axis.get_aspect() == 1.0 and axis.collections]
        bars = [axis for axis in figure.axes if axis.get_aspect() != 1.0 and axis.get_visible() and axis.get_ylabel()]
        assert len(maps) == len(bars) == 3  # T for both datasets, p only for the one that provides it
        assert all(axis.get_ylabel().endswith("]") for axis in bars)
        assert any("unavailable" in text.get_text() for axis in figure.axes for text in axis.texts)
        assert wider.get_figwidth() > figure.get_figwidth()
    finally:
        plt.close(figure)
        plt.close(wider)


def test_dataset_styles_keep_material_colors_and_tell_batches_of_one_material_apart() -> None:
    """Batches of one material share its color and differ in line style and hatch; a lone material stays solid; order colors avoid every material."""
    semantics = transient_plots.visual_semantics
    materials = set(semantics.MATERIAL_COLORS.values())
    assert not materials & set(transient_plots._ORDER_COLORS)
    styles = transient_plots.batch_styles({"B": "chickpea", "A": "chickpea", "C": "lentil", "D": "unregistered", "E": None})
    assert styles["A"].color == styles["B"].color == semantics.material_color("chickpea")
    assert styles["A"].linestyle == "-" != styles["B"].linestyle  # the sorted first label keeps the solid line
    assert styles["A"].hatch == "" != styles["B"].hatch
    assert (styles["C"].color, styles["C"].linestyle) == (semantics.material_color("lentil"), "-")
    assert styles["D"].linestyle == styles["E"].linestyle == "-"
    assert styles["D"].color != styles["E"].color
    assert {styles["D"].color, styles["E"].color} <= set(transient_plots._ORDER_COLORS)


def test_trajectory_summary_over_the_phase_pools_only_reached_cases_and_keeps_the_schedules_in_hours() -> None:
    """
    Each batch's reached 24 h case enters all 20 phase points of the state row; its 48 h censored case is counted in the note, not pooled.

    Stored at 0, 1 and 24 h, the reached case gives its 1 h state up to τ = 0.5 and its 24 h state beyond. The schedule rows stay
    on the clock with their own time label, as bands of the cases still running.
    """
    frames = catalog_transient._eda_datasets()
    styles = transient_plots.batch_styles(dict.fromkeys(frames, "lentil"))
    figure = transient_plots.plot_state_trajectory_summary(datasets=frames, max_cases=2, styles=styles, channels=("T",), time_axis="phase")
    try:
        (phase_axis,) = [axis for axis in figure.axes if axis.get_xlabel() == "τ = t / t_end [-]"]
        grid = transient_plots.time_axis.PHASE_GRID
        means = [line for line in phase_axis.get_lines() if np.array_equal(line.get_xdata(), grid)]
        assert len(means) == len(frames)
        for line, frame in zip(means, frames.values(), strict=True):
            _times, case_means, *_extrema = capabilities.field_display_trajectory(frame, capabilities.case_row(frame, "case_0001"), "T")
            np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=np.float64), case_means[np.repeat([1, 2], 10)])
        counts = [line for axis in figure.axes if axis.get_ylabel() == "n(τ)" for line in axis.get_lines()]
        assert counts
        assert all(np.array_equal(line.get_ydata(), np.ones(grid.size)) for line in counts)
        assert any(axis.get_xlabel().startswith("Time") for axis in figure.axes)  # the schedule rows keep hours
        note = " ".join(" ".join(str(cast("Any", artist).get_text()) for artist in figure.artists if hasattr(artist, "get_text")).split())
        assert "t_end 24-24 h, median 24 h; 1 reached, 1 censored not pooled" in note
    finally:
        plt.close(figure)


@pytest.mark.parametrize("evidence", [True, False])
def test_the_trajectory_viewer_offers_the_drying_phase_only_where_every_case_carries_completion_evidence(
    evidence: bool, transient_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The trajectory viewer offers the drying phase only where every selected frame carries completion evidence.

    With it the clock choice shows and the phase is drawn by default; beside a frame without it (a published Dataset) the
    choice is hidden and the view draws hours, and the plot asked directly for the phase names that frame.
    """
    render_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(viewers.ui.viewers, "render_view", lambda *_args, **kwargs: render_calls.append(kwargs))
    published = transient_frame.copy()
    published["completion"] = None
    frames = {"Generated": transient_frame, **({} if evidence else {"Published": published})}
    catalog = _adaptive_catalog(tuple(frames.items()))
    viewer = viewers.make_transient_case_view(
        catalog=catalog,
        selection_state=selection.GeneratedOutputSelectionState(catalog),
        kind="trajectory",
        export_state=None,
        export_plot_name=None,
        export_title=None,
    ).widget
    (clock,) = [widget for widget in _walk_widgets(viewer) if isinstance(widget, widgets.ToggleButtons) and widget.value == "phase"]
    assert (clock.layout.display != "none") is evidence
    assert render_calls[-1]["kwargs"]["time_axis"] == ("phase" if evidence else "hours")
    if not evidence:
        styles = transient_plots.batch_styles(dict.fromkeys(frames, "lentil"))
        with pytest.raises(ValueError, match=r"\['Published'\] lack completion evidence"):
            transient_plots.plot_state_trajectory_summary(datasets=frames, max_cases=1, styles=styles, channels=("T",), time_axis="phase")


def test_trajectory_lines_keep_batch_colors_and_line_styles_with_n_t_on_every_row_and_one_outside_legend(transient_frame: pd.DataFrame) -> None:
    """Two batches of one material draw one color in two line styles; n(t) is on every row; the legend sits right of all axes."""
    frames = {label: transient_frame.copy() for label in ("A", "B")}
    styles = transient_plots.batch_styles({"A": "chickpea", "B": "chickpea"})
    figure = transient_plots.plot_state_trajectory_summary(datasets=frames, max_cases=1, styles=styles, channels=("T",))
    try:
        lines = [line for axis in figure.axes for line in axis.get_lines() if line.get_linestyle() != "None"]
        assert {line.get_color() for line in lines} == {transient_plots.visual_semantics.material_color("chickpea")}
        assert {line.get_linestyle() for line in lines} == {"-", "--"}
        twins = [axis for axis in figure.axes if axis.get_ylabel() == "n(t)"]
        assert len(twins) == 2 * (len(transient_plots._schedule_fields()) + 1)
        counts = [line for axis in twins for line in axis.get_lines()]
        assert {line.get_linestyle() for line in counts} == {"-", "--"}  # n(t) carries the batch line style
        assert {line.get_drawstyle() for line in counts} == {"steps-post"}
        assert figure.legends
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert isinstance(figure.canvas, FigureCanvasAgg)
        renderer = figure.canvas.get_renderer()
        right = max(_tight_bbox(axis, renderer).x1 for axis in figure.axes)  # the n(t) tick labels count as panel
        assert all(legend.get_window_extent(renderer).x0 >= right for legend in figure.legends)
    finally:
        plt.close(figure)


def test_map_row_without_a_first_column_map_keeps_its_y_labels_on_its_first_visible_map(transient_frame: pd.DataFrame) -> None:
    """When column 0 of a row is an empty cell the row's y tick labels and ``y [m]`` move to its first map."""
    reduced_row = copy.deepcopy(transient_frame.iloc[0].to_dict())
    reduced_row["static_fields"].pop("p")
    reduced = pd.DataFrame([reduced_row], index=transient_frame.index.copy())
    reduced.attrs = copy.deepcopy(transient_frame.attrs)
    categories = copy.deepcopy(reduced.attrs["field_categories"])
    categories["static_spatial"] = tuple(field for field in categories["static_spatial"] if field != "p")
    reduced.attrs["field_categories"] = categories
    figure = transient_plots.plot_spatial_field_comparison(
        datasets={"A": reduced, "B": transient_frame},
        case_ids={"A": "case_0001", "B": "case_0001"},
        fields=("T", "p"),
        physical_time_hours=1.0,
    )
    try:
        maps = [axis for axis in figure.axes if axis.get_aspect() == 1.0 and axis.collections]
        assert len(maps) == 3
        labelled = [axis for axis in maps if axis.get_ylabel() == "y [m]"]
        assert len(labelled) == 2  # one per map row, whichever column holds its first map
        assert all(axis.yaxis.get_tick_params()["labelleft"] for axis in labelled)
    finally:
        plt.close(figure)


def test_two_line_column_header_clears_the_schedule_band_above_the_maps(transient_frame: pd.DataFrame) -> None:
    """A long two-line dataset label stays below the x label and tick labels of the bottom schedule row."""
    label = "Lentil · parameter OOD · test · extended campaign with a second long descriptive tail"
    figure = transient_plots.plot_state_snapshot_comparison(
        datasets={label: transient_frame},
        case_ids={label: "case_0001"},
        physical_time_hours=1.0,
        styles=transient_plots.batch_styles({label: None}),
        channels=("T",),
    )
    try:
        figure.canvas.draw()
        assert isinstance(figure.canvas, FigureCanvasAgg)
        renderer = figure.canvas.get_renderer()
        header = next(axis for axis in figure.axes if axis.get_aspect() == 1.0 and axis.title.get_text()).title.get_window_extent(renderer)
        schedules = [axis for axis in figure.axes if (spec := axis.get_subplotspec()) is not None and spec.get_gridspec().ncols == 2]
        bottom_row = max(schedules, key=_row_start)
        texts = [bottom_row.xaxis.label, *bottom_row.get_xticklabels()]
        lowest = min(text.get_window_extent(renderer).y0 for text in texts if text.get_text())
        assert header.y1 <= lowest
    finally:
        plt.close(figure)


def test_single_case_map_figure_names_the_case_in_its_title(transient_frame: pd.DataFrame) -> None:
    """A figure of one dataset and one case carries the case in its figure title."""
    figure = transient_plots.plot_spatial_field_comparison(
        datasets={"A": transient_frame}, case_ids={"A": "case_0001"}, fields=("T",), physical_time_hours=1.0
    )
    try:
        assert "case_0001" in figure.get_suptitle()
    finally:
        plt.close(figure)


def test_schedule_band_has_no_left_labels_on_startup_axes_and_a_whole_word_legend_at_the_top_frame(transient_frame: pd.DataFrame) -> None:
    """Startup axes share the row's y axis without tick labels; the legend keeps words whole and starts at the pressure frame."""
    label = "Lentil · parameter-OOD-extra-long-hyphenated-token · test"
    figure = transient_plots.plot_state_snapshot_comparison(
        datasets={label: transient_frame},
        case_ids={label: "case_0001"},
        physical_time_hours=1.0,
        styles=transient_plots.batch_styles({label: None}),
        channels=("T",),
    )
    try:
        startups = [
            axis
            for axis in figure.axes
            if (spec := axis.get_subplotspec()) is not None and spec.get_gridspec().ncols == 2 and spec.colspan.start == 1
        ]
        assert startups
        for axis in startups:
            assert axis.get_ylabel() == ""
            assert not axis.yaxis.get_tick_params()["labelleft"]
        legend = figure.legends[0]
        words = {word for text in legend.get_texts() for word in text.get_text().split()}
        assert set(label.split()) <= words
        figure.canvas.draw()
        pressure = max((axis for axis in figure.axes if axis.get_visible()), key=lambda axis: axis.get_position().y1)
        top = legend.get_window_extent().y1
        assert abs(top - pressure.get_window_extent().y1) < 1.0
    finally:
        plt.close(figure)


def test_spatial_maps_lock_full_grid_discrete_permeability_across_display_strides(transient_frame: pd.DataFrame) -> None:
    """Physical permeability uses one full-grid logarithmic scale across maps."""
    row = transient_frame.loc["case_0001"]
    original = np.asarray(row["static_fields"]["Kxx"])
    values = np.geomspace(1e-10, 1e-8, original.size).reshape(original.shape)
    row["static_fields"]["Kxx"] = values
    first = transient_plots.plot_spatial_field_comparison(
        datasets={"A": transient_frame, "B": transient_frame},
        case_ids={"A": "case_0001", "B": "case_0001"},
        fields=("Kxx",),
        lock_scale=True,
        display_spatial_stride=1,
    )
    second = transient_plots.plot_spatial_field_comparison(
        datasets={"A": transient_frame, "B": transient_frame},
        case_ids={"A": "case_0001", "B": "case_0001"},
        fields=("Kxx",),
        lock_scale=True,
        display_spatial_stride=4,
    )
    try:
        maps = [axis.collections[0] for axis in first.axes if axis.get_aspect() == 1.0]
        sampled = [axis.collections[0] for axis in second.axes if axis.get_aspect() == 1.0]
        assert len(maps) == len(sampled) == 2
        assert maps[0].norm is maps[1].norm
        for item in (*maps, *sampled):
            assert item.cmap.N == 11
            assert isinstance(item.norm, BoundaryNorm)
            np.testing.assert_allclose(item.norm.boundaries, np.geomspace(1e-10, 1e-8, 12))
        sampled_array = sampled[0].get_array()
        full_array = maps[0].get_array()
        assert sampled_array is not None
        assert full_array is not None
        assert sampled_array.size < full_array.size
        np.testing.assert_array_equal(row["static_fields"]["Kxx"], values)
    finally:
        row["static_fields"]["Kxx"] = original
        plt.close(first)
        plt.close(second)
