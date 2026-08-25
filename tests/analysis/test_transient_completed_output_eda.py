# ruff: noqa: S101, PLR2004, PD011, SLF001
"""Protect transient completed-output EDA semantics and steady compatibility."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import datasets, domain
from src.analysis.eda import dataframe as eda_dataframe
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda import eda_selection as selection
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_transient as transient
from src.analysis.eda import eda_viewers as viewers
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.datasets.packages import dataset_packages_generated_batch as generated
from tests.generation.test_generation_transient import (
    _synthetic_scientific_contract,
    _write_transient_case,
)
from tests.generation.test_generation_transient_shards import (
    _fixture_context,
    _publication_identity,
)

if TYPE_CHECKING:
    from src.generation.runtime.generation_runtime_batch import (
        TerminalBatchEvidence,
        TerminalCaseEvidence,
    )


class _Artifact:
    """Expose the terminal artifact API over one test-owned file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.size_bytes = path.stat().st_size

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


class _Case:
    """Provide the admitted-case surface consumed by the direct interpreter."""

    case_id = "case_0001"
    case_index = 1
    case_input_id = "1" * 64
    simulation_case_id = "2" * 64
    material_family = "lentil"
    hdf5_identity = SimpleNamespace(
        simulation_profile="transient_drying",
        git_commit="a" * 40,
    )

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.artifact_requests: list[tuple[str, str]] = []

    def artifact(self, stage: str, relative_path: str) -> _Artifact:
        self.artifact_requests.append((stage, relative_path))
        return _Artifact(self.directory / relative_path)

    artifact_evidence = artifact

    @staticmethod
    def metadata_payload() -> dict[str, Any]:
        return {
            "seed_evidence": {"case_seed": 7},
            "sampled_values": {"T_init": 295.0},
            "sampled_units": {"T_init": "K"},
            "material_role": "seen",
            "evaluation_regime": "id",
            "natural_support_state": "natural",
            "spatial_diagnostics": {},
            "schedule_diagnostics": {"schedule_class": "synthetic"},
            "ood": {},
        }


class _Batch:
    """Provide one exact terminal-batch binding for a synthetic case."""

    simulation_profile = "transient_drying"
    available_learning_views = ("steady_flow", "transient_drying")
    airflow_source = "comsol_coupled_reference"
    batch_id = "batch-1"
    sampling_regime = "natural"
    template_sha256 = "e" * 64
    git_commit = "a" * 40

    def __init__(self, case: _Case) -> None:
        self.cases = (case,)
        self._case = case

    def case(self, case_id: str) -> _Case:
        if case_id != self._case.case_id:
            raise ValueError(case_id)
        return self._case

    @staticmethod
    def scientific_config_payload() -> dict[str, Any]:
        return _synthetic_scientific_contract()


def _write_sidecars(directory: Path, *, target_reached: bool = True) -> None:
    """Write current Generation timing and status evidence beside one HDF5 case."""
    timing = {
        "schema_kind": "simulation_case_timing",
        "schema_version": 1,
        "batch_id": "batch-1",
        "case_id": "case_0001",
        "case_input_id": "1" * 64,
        "simulation_case_id": "2" * 64,
        "simulation_profile": "transient_drying",
        "git_commit": "a" * 40,
        "exit_code": 0,
        "timed_out": False,
        "runtime_s": 12.0,
        "comsol_process_seconds": 12.0,
        "export_conversion_seconds": 3.0,
        "complete_execution_s": 16.0,
        "license_wait_seconds": 2.0,
    }
    status = {
        "schema_kind": "simulation_case_status",
        "schema_version": 1,
        "solver_success": True,
        "target_reached": target_reached,
        "t_stop_exact": 2.5,
        "f_wet_dm_final": 0.04 if target_reached else 0.07,
        "runtime_s": 12.0,
        "units": {
            "runtime_s": "s",
            "t_stop_exact": "h",
            "f_wet_dm_final": "1",
        },
        "contains_nan_or_inf": False,
        "field_shape_valid": True,
        "case_state": "successful",
        "stages": dict.fromkeys(("solver", "exports", "conversion", "publication"), "succeeded"),
    }
    (directory / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")


@pytest.fixture
def transient_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Return one fully interpreted canonical completed-case EDA frame."""
    _write_transient_case(tmp_path / "case.h5")
    _write_sidecars(tmp_path)
    case = _Case(tmp_path)
    task = domain.tasks.registry.get_task("transient_drying")
    row = generated.interpret_generated_transient_case(
        cast("TerminalBatchEvidence", _Batch(case)),
        cast("TerminalCaseEvidence", case),
        task=task,
    )
    assert case.artifact_requests == [
        ("processed", "case.h5"),
        ("processed", "timing.json"),
        ("processed", "status.json"),
    ]
    loaded = {
        "sample_ids": [case.case_id],
        "rows": [row],
        "available_case_count": 1,
        "task": task,
        "generated_batch_identity": {"batch_manifest_identity_sha256": "f" * 64},
        "manifest_sha256": "d" * 64,
        "generation_root": tmp_path,
        "analysis_representation": "transient_complete_case_rows",
        "dataset_backend": "canonical_generation_hdf5",
    }
    monkeypatch.setattr(
        datasets.packages.generated_batch,
        "load_generated_batch",
        lambda *_args, **_kwargs: loaded,
    )
    frame, logs = eda_dataframe.generate_eda_dataframe(
        "synthetic_transient",
        task=task,
        show_progress=False,
    )
    assert any("completed trajectories" in line for line in logs)
    assert frame.iloc[0]["meta"]["case_family"] == "id"
    assert frame.iloc[0]["meta"]["parameter_units"] == {"T_init": "K"}
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
            batch_storage_name=f"batch-{position}-storage",
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
    return selection.GeneratedOutputEDACatalog(
        views,
        source_catalog=source_catalog,
    )


def test_transient_dispatch_discovers_complete_canonical_evidence(
    transient_frame: pd.DataFrame,
) -> None:
    """Discover every canonical field category and preserve startup endpoints."""
    discovery = transient.discover_fields(transient_frame)
    assert tuple(field.name for field in discovery["dynamic_state"]) == (
        "T",
        "phi",
        "w_surf",
        "w_int",
    )
    assert {field.name for field in discovery["static_spatial"]}.issuperset({"x", "y", "u", "v", "p", "eps_bed", "rho_bu_dry"})
    assert tuple(field.name for field in discovery["boundary_interval"]) == (
        "T_in_bc_t_n",
        "T_in_bc_t_n_plus_1",
        "omega_in_bc_t_n",
        "omega_in_bc_t_n_plus_1",
        "T_amb",
        "startup_support_time_offset",
        "T_in_bc_startup_support",
        "omega_in_bc_startup_support",
        "startup_support_present",
    )
    assert tuple(field.name for field in discovery["scalar_material"]) == (
        "r_surf_0",
        "r_int_surf",
        "f_surf",
        "A_osw",
        "B_osw",
        "C_osw",
        "k_gr",
        "cp_gr_dry",
    )
    assert tuple(field.name for field in discovery["complete_schedule"]) == (
        "t",
        "T_in_bc",
        "omega_in_bc",
    )
    intervals = transient.boundary_interval_table(transient_frame)
    assert intervals.loc[0, "startup_support_present"] == 1.0
    support_offset = float(intervals.loc[0, "startup_support_time_offset"])
    assert support_offset > 0.0
    support_time = float(intervals.loc[0, "t_n_hours"]) + support_offset
    schedule_time = np.asarray(transient_frame.iloc[0]["schedule"]["t"], dtype=float)
    assert np.count_nonzero(np.isclose(schedule_time, support_time, rtol=0.0, atol=1.0e-7)) == 1
    schedule = transient_frame.iloc[0]["schedule"]
    for schedule_name, current_name, following_name in (
        ("T_in_bc", "T_in_bc_t_n", "T_in_bc_t_n_plus_1"),
        ("omega_in_bc", "omega_in_bc_t_n", "omega_in_bc_t_n_plus_1"),
    ):
        current_index = int(np.flatnonzero(np.isclose(schedule_time, intervals.loc[0, "t_n_hours"]))[0])
        following_index = int(np.flatnonzero(np.isclose(schedule_time, intervals.loc[0, "t_n_plus_1_hours"]))[0])
        assert intervals.loc[0, current_name] == pytest.approx(schedule[schedule_name][current_index])
        assert intervals.loc[0, following_name] == pytest.approx(schedule[schedule_name][following_index])
    assert not transient.schedule_summary(transient_frame).empty
    assert transient.scalar_parameter_table(transient_frame)["parameter"].nunique() == 8
    time = transient_frame.iloc[0]["time"]
    assert time["trajectory_length"] == len(time["regular_state_hours"])
    assert np.asarray(time["valid_state_mask"]).all()


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
    snapshot = transient.select_state_snapshot(transient_frame, "case_0001", 1.0)
    assert snapshot.channels == ("T", "phi", "w_surf", "w_int")
    assert snapshot.diagnostic_exact_stop is False
    exact = transient.select_state_snapshot(transient_frame, "case_0001", 2.5)
    assert exact.diagnostic_exact_stop is True
    with pytest.raises(transient.PhysicalTimeUnavailableError):
        transient.select_state_snapshot(transient_frame, "case_0001", 1.5)

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
    fixed_time = transient.fixed_time_summary(frame, 1.0)
    assert fixed_time.attrs["right_censored_case_count"] == 1
    assert fixed_time.attrs["right_censored_contributor_count"] == 1
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

    runtime = transient.runtime_table(frame)
    assert runtime["physical_drying_duration_hours"].eq(2.5).all()
    assert runtime["comsol_process_seconds"].eq(12.0).all()
    assert runtime["queue_wait_seconds"].isna().all()
    assert runtime.iloc[0]["component_timing_availability"]["queue_wait_seconds"] == "unavailable_not_persisted"


def test_steady_dataframe_behavior_remains_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        datasets.packages.generated_batch,
        "load_generated_batch",
        lambda *_args, **_kwargs: loaded,
    )
    frame, _logs = eda_dataframe.generate_eda_dataframe(
        "synthetic_steady",
        task=task,
        show_progress=False,
    )
    assert tuple(frame.columns) == (*task.input_names, *task.output_names, "meta", "U")
    np.testing.assert_allclose(frame.iloc[0]["U"], np.hypot(frame.iloc[0]["u"], frame.iloc[0]["v"]))
    assert frame.attrs["field_names"] == (*task.input_names, *task.output_names, "U")
    assert frame.attrs["spatial_shape"] == shape


def test_hdf5_and_pt_items_match_after_selected_source_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce equal semantic EDA from real HDF5/PT items, then use PT alone."""
    dataset_id = "transient_drying__lentil__id__eda_backend_fixture"
    manifest, index = _fixture_context(tmp_path, dataset_id=dataset_id)
    shard_service = datasets.packages.transient_shards

    def package_context(
        _dataset_id: str,
        *,
        storage_root: Path,
        validation_depth: str,
    ) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
        assert validation_depth in {"full", "evidence"}
        return Path(storage_root), manifest, index, "4" * 64

    monkeypatch.setattr(shard_service, "_package_context", package_context)
    shard_service.build_transient_shards(
        dataset_id,
        storage_root=tmp_path,
        publication_identity=_publication_identity(),
        target_shard_bytes=1_000_000,
    )
    index_path = tmp_path / "02_datasets/packages" / dataset_id / f"{dataset_id}.json"
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    positions = (0, 1, 2)
    physical = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=sampling,
        source_root=tmp_path,
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
    with pytest.raises(ValueError, match="contiguous transition chain"):
        transient.frame_from_transient_items(
            [hdf5_items[0], hdf5_items[2]],
            backend="canonical_hdf5",
        )

    selected_source = tmp_path / str(index["cases"][0]["source_relative_path"])
    selected_source.unlink()
    sharded = datasets.runtime.transient.TransientPTShardDataset(
        index_path,
        sampling=sampling,
        source_root=tmp_path,
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
    left_snapshot = transient.select_state_snapshot(hdf5_frame, "2" * 64, 2.0)
    right_snapshot = transient.select_state_snapshot(pt_frame, "2" * 64, 2.0)
    for name in left_snapshot.channels:
        np.testing.assert_allclose(left_snapshot.fields[name], right_snapshot.fields[name], rtol=0.0, atol=0.0)


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
    assert series.values[-1] == pytest.approx(np.interp(exact_final, source_time, source_values))
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
    row = transient_frame.loc["case_0001"]
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


def test_split_time_plot_converts_only_main_coordinates_to_days() -> None:
    """Convert only displayed main-time coordinates and preserve exact support."""
    physical_time = np.asarray((0.0, 0.5, 1.0, 24.0, 48.0))
    values = np.asarray((10.0, 11.0, 12.0, 13.0, 14.0))
    original_time = physical_time.copy()
    original_values = values.copy()
    figure, (main_axis, startup_axis) = plt.subplots(1, 2)
    try:
        main_support, startup_support = transient_plots._plot_split_time_series(
            main_axis,
            startup_axis,
            physical_time,
            values,
            lower=None,
            upper=None,
            color="tab:blue",
        )
        transient_plots._configure_split_time_axes(
            main_axis,
            startup_axis,
            (main_support,),
            (startup_support,),
        )

        np.testing.assert_allclose(
            main_axis.lines[0].get_xdata(),
            np.asarray((1.0, 24.0, 48.0)) / 24.0,
        )
        np.testing.assert_array_equal(
            startup_axis.lines[0].get_xdata(),
            (0.0, 0.5, 1.0),
        )
        assert main_axis.get_xlabel() == "Time [d]"
        assert startup_axis.get_xlabel() == "Time [h]"
        np.testing.assert_array_equal(physical_time, original_time)
        np.testing.assert_array_equal(values, original_values)
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
