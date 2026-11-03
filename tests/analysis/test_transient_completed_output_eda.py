"""Protect transient completed-output EDA semantics and steady compatibility.

Responsibilities:
  - Check completed-source dispatch, exact-time trajectories, target outcomes, and spatial maps

Design principles:
  - Physical support and steady compatibility survive display transformations

This module does NOT:
  - Forward-fill completed cases or change source arrays
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import BoundaryNorm

from src import common, datasets, domain, generation
from src.analysis.eda import dataframe as eda_dataframe
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda import eda_selection as selection
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_transient as transient
from src.analysis.eda import eda_viewers as viewers
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.datasets.packages import dataset_packages_trajectory as trajectory


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
        "render_figure",
        lambda **kwargs: render_calls.append(kwargs),
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
    )
    assert _widgets_by_description(snapshot_viewer, "Lock color scale") == []
    time_control = _widget_by_description(snapshot_viewer, "Time [h]:")
    assert not any(isinstance(widget, widgets.Play) for widget in _walk_widgets(snapshot_viewer))
    retained_channel = _widget_by_description(snapshot_viewer, "φ [-]")
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
    )
    scope = next(widget for widget in _walk_widgets(trajectory_viewer) if isinstance(widget, widgets.ToggleButtons))
    assert scope.value == "aggregate"
    assert _widgets_by_description(trajectory_viewer, "Cases:")
    assert _widgets_by_description(trajectory_viewer, "Case:") == []
    scope.value = "single"
    assert _widgets_by_description(trajectory_viewer, "Case:")
    assert _widgets_by_description(trajectory_viewer, "Cases:") == []
    assert not any(getattr(widget, "description", "") == "Time [h]:" for widget in _walk_widgets(trajectory_viewer))


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
        "render_figure",
        lambda **kwargs: render_calls.append(kwargs),
    )
    orientation = widgets.Dropdown(
        options=("cross_stream_along_flow", "flow_across_cross_stream"),
        value="cross_stream_along_flow",
        description="Orientation:",
    )

    viewer = viewers.make_spectral_view(
        catalog=catalog,
        selection_state=state,
        single_plot_function=cast("Any", lambda **_kwargs: None),
        aggregate_plot_function=cast("Any", lambda **_kwargs: None),
        semantic_controls={"orientation": orientation},
        export_state=None,
        export_plot_name=None,
        export_title=None,
    )
    scope = next(widget for widget in _walk_widgets(viewer) if isinstance(widget, widgets.ToggleButtons))
    assert scope.value == "aggregate"
    assert _widgets_by_description(viewer, "Cases:")
    assert _widgets_by_description(viewer, "Case:") == []
    orientation.value = "flow_across_cross_stream"
    assert render_calls[-1]["kwargs"]["orientation"] == "flow_across_cross_stream"
    scope.value = "single"
    assert _widgets_by_description(viewer, "Case:")
    assert _widgets_by_description(viewer, "Cases:") == []


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
        "render_figure",
        lambda **kwargs: render_calls.append(kwargs),
    )
    spatial_viewer = viewers.make_spatial_case_view(
        catalog=catalog,
        selection_state=state,
        export_state={"fig": None, "plot_name": "spatial", "title": "spatial"},
        export_plot_name="spatial",
        export_title="spatial",
    )
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
        "generated_batch_identity": {"batch_manifest_identity_sha256": "f" * 64},
        "manifest_sha256": "d" * 64,
        "generation_root": Path("/synthetic"),
    }
    frame, _logs = eda_dataframe._steady_dataframe(loaded, batch_name="synthetic_steady", task=task)
    assert tuple(frame.columns) == (*task.input_names, *task.output_names, "meta", "U")
    np.testing.assert_allclose(frame.iloc[0]["U"], np.hypot(frame.iloc[0]["u"], frame.iloc[0]["v"]))
    assert frame.attrs["field_names"] == (*task.input_names, *task.output_names, "U")
    assert frame.attrs["spatial_shape"] == shape


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
            channels=("T",),
        ),
        transient_plots.plot_state_trajectory_comparison(
            datasets={"Drying · Lentil · ID": transient_frame},
            case_ids={"Drying · Lentil · ID": "case_0001"},
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
