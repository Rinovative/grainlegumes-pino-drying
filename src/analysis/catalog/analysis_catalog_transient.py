"""Catalog the transient evaluation and transient Dataset EDA plots on synthetic records and frames.

Responsibilities:
  - Declare one catalog entry per transient sub-type, each drawn through its real builder
  - Build the synthetic transient Evaluation records, frames and scaling state that the catalog and the tests share
  - Build one small transient Dataset EDA frame set with schedules, trajectories and completion evidence, and one saved
    two-run Evaluation session over synthetic records

Design principles:
  - Entries call the real builders on small synthetic inputs; nothing is drawn through a copy of a builder
  - Record states are constant over the smallest grid with an H1 interior, so per-case errors equal the offsets; the
    catalog records add a spatial ramp on top, so maps and trajectories show structure
  - Inputs live in memory; the session and the EDA frames are built once per process

This module does NOT:
  - Read storage, run inference or write artifacts
  - Represent production results or check the scientific validity of the figures it draws
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import cache
from types import SimpleNamespace
from typing import Any, Final

import numpy as np
import pandas as pd
import torch

from src import datasets, domain
from src.analysis.eda.plots import eda_plot_transient as eda_plots
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.evaluation import evaluation_transient_artifact as artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_session as session_module
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.analysis.evaluation.evaluation_plot import evaluation_plot_transient as plots
from src.datasets.contracts import dataset_contracts_transient as step_contract
from src.learning.transient import learning_transient_contracts as transient_contracts

from .analysis_catalog_entries import CatalogPlot

ANALYSIS_SHA256 = "9" * 64
GRID_SHAPE = (5, 5)
_STATIC_CONSTANTS = (1.0, 1.0, 101325.0, 0.4, 100.0)
_STEP_HOURS: Final = 12.0
_RUNS: Final = (("FNO run A", "z-run", "a" * 64, 1.0), ("FNO run B", "a-run", "b" * 64, 2.0))
_TARGET_HOURS: Final = 26.0
# The first case carries the case ID of the one timing case that ``production_frame`` records, so runtime evidence pairs with it.
_MATERIALS: Final = (("case", "lentil", 1.0), ("b", "chickpea", 2.0))
_FIELDS: Final = ("T", "phi")
_EDA_SHAPE: Final = (4, 6)
_EDA_LABELS: Final = ("Lentil · batch A", "Lentil · batch B")


def scaling_state(*, membership_digest: str = "a" * 64) -> dict[str, object]:
    """Return one valid synthetic transient scaling state with a two-unit state scale and unit scales elsewhere."""
    normalization = datasets.preprocessing.normalization
    task = domain.tasks.registry.get_task(normalization.TRANSIENT_TASK)
    view = {
        "spatial_stride": 1,
        "canonical_ny": GRID_SHAPE[0],
        "canonical_nx": GRID_SHAPE[1],
        "effective_ny": GRID_SHAPE[0],
        "effective_nx": GRID_SHAPE[1],
    }
    tensorizer = transient_contracts.TransientTensorizerSpec("canonical_physics_complete", transient_contracts.TemporalConditioningSpec("none"))
    groups = {}
    for name, names in normalization.expected_groups(task.id).items():
        std = 2.0 if name in {"state", "physical_state"} else 1.0
        groups[name] = normalization.ChannelStatistics(names=names, mean=torch.zeros(len(names)), std=torch.full((len(names),), std))
    return normalization.TrainScaling(
        task=task.id,
        task_contract_digest=task.contract_digest,
        data_contract_digest=task.data_contract_digest,
        preprocessing=normalization.preprocessing_spec(task.id, spatial_view=view, tensorizer=tensorizer.as_dict()),
        dataset_identity={"synthetic": "session"},
        train_membership_digest=membership_digest,
        train_case_count=1,
        spatial_shape=GRID_SHAPE,
        groups=groups,
        horizon=8.0,
    ).state_dict()


def static_conditioning() -> np.ndarray:
    """Return static fields on a 4 cm by 8 cm grid with constant u, v, p, porosity and dry bulk density."""
    y, x = np.meshgrid(np.linspace(0.0, 0.04, GRID_SHAPE[0]), np.linspace(0.0, 0.08, GRID_SHAPE[1]), indexing="ij")
    return np.stack((x, y, *(np.full(GRID_SHAPE, value) for value in _STATIC_CONSTANTS))).astype(np.float32)


def sequence_record(
    mode: str,
    horizon: int | str,
    *,
    case_id: str = "case",
    offset: float = 2.0,
    material_family: str = "lentil",
    length: int | None = None,
    target_moisture_wb: float = 0.12,
) -> SimpleNamespace:
    """Return one synthetic four-channel sequence record with a constant prediction offset after the stored origin and its wet-basis target."""
    if length is None:
        length = 2 if horizon == 1 else 3
    reference = np.zeros((length, 4, *GRID_SHAPE), dtype=np.float32)
    prediction = reference.copy()
    prediction[1:] = offset
    full_reference = mode == "autonomous_full" and horizon == "full"
    target = {
        "target_moisture_wb": target_moisture_wb,
        "predicted_evidence_scope": "regular_sequence_grid",
        "predicted_available": True,
        "predicted_unavailable_reason": None,
        "predicted_reached": True,
        "predicted_time_to_target": 1.0,
        "predicted_final_gap": -0.1,
        "predicted_final_time": float(length - 1),
        "reference_evidence_scope": ("canonical_completed_case" if full_reference else "unavailable_partial_interval"),
        "reference_available": full_reference,
        "reference_unavailable_reason": (None if full_reference else "canonical_completed_case_target_unavailable_for_partial_interval"),
        "reference_reached": False,
        "reference_time_to_target": None,
        "reference_final_gap": (0.2 if full_reference else None),
        "reference_final_time": (float(length - 1) if full_reference else None),
    }
    return SimpleNamespace(
        mode=mode,
        case_id=case_id,
        requested_horizon=horizon,
        origin_index=0,
        physical_times=np.arange(length, dtype=np.float64),
        reference_states=reference,
        predicted_states=prediction,
        static_conditioning=static_conditioning(),
        scalar_conditioning=np.asarray([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
        identity={
            "material_family": material_family,
            "input_profile": "canonical_physics_complete",
            "model_kind": "fno",
            "pt_payload_identity": "a" * 64,
            "timing_evidence_identity": "b" * 64,
            "scaling_identity": {"semantic_digest": "c" * 64},
            "evaluation_assignment": None,
            "spatial_representation": {"grid_identity_sha256": "f" * 64},
        },
        target=target,
    )


def sequence_frame(records: tuple[SimpleNamespace, ...]) -> pd.DataFrame:
    """Return an artifact frame carrying the given in-memory sequence records and their recorded analysis producer."""
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "artifact_provenance": {"producer": {"analysis_sha256": ANALYSIS_SHA256}},
            "transient_sequence_records": records,
            "transient_unavailable_horizons": ({"case_id": "case", "requested_horizon": 4, "reason": "short"},),
            "transient_scaling_state": scaling_state(),
        }
    )
    return frame


def production_frame(
    records: tuple[SimpleNamespace, ...],
    *,
    artifact_root: str,
    dataset_name: str = "transient_dataset",
    dataset_role: str = "id",
) -> pd.DataFrame:
    """Return a sequence frame with complete run, Dataset, timing, and lineage provenance and the completed run's summary."""
    frame = sequence_frame(records)
    report = timing.build_transient_timing_report(
        (
            timing.TransientTimingCase(
                case_id="case",
                repetitions={
                    "drying_no_rollout_model_seconds": (0.01, 0.02, 0.03),
                    "drying_no_prepared_rollout_wall_seconds": (0.02, 0.03, 0.04),
                    "comsol_transient_drying_seconds": (0.3,),
                },
                device="cpu",
                precision="float32",
                pt_payload_identity="a" * 64,
                warmup_passes=1,
                solver_hostname="solver-node",
                solver_requested_cores=8,
                cpu="test-cpu",
                software_versions={"python": "test", "pytorch": "test", "numpy": "test"},
                cold_timings={"drying_no_rollout_model_seconds": 0.05},
            ),
        )
    )
    frame.attrs.update(
        {
            "artifact_root": artifact_root,
            "artifact_provenance": {
                "producer": {"analysis_sha256": ANALYSIS_SHA256},
                "run": {
                    "best_checkpoint_sha256": "d" * 64,
                    "best_checkpoint_epoch": 3,
                },
                "dataset": {
                    "name": dataset_name,
                    "role": dataset_role,
                    "source_dataset_ids": [dataset_name],
                    "source_identities": [{"dataset_id": dataset_name}],
                    "case_ids_by_source": [[record.case_id for record in records]],
                    "membership_digests": ["e" * 64],
                    "evaluation_assignments": [],
                },
                "evaluation": {
                    "pipeline_analysis": {
                        "conditions": [
                            "A_comsol_reference",
                            "B_drying_on_comsol_airflow",
                            "C_drying_on_airflow_no",
                        ],
                        "c_available": False,
                        "c_unavailable_reasons": ["compatible_airflow_checkpoint_not_selected"],
                        "fabricated_prediction_count": 0,
                    },
                },
                "timing": {
                    "report": json.loads(json.dumps(asdict(report))),
                    "component_availability": {
                        "case": {"dataset_materialization_seconds": 0.005},
                    },
                },
                "lineage": {
                    "stage_identity": {"stage": "b", "comparison_arm": "b"},
                    "training_strategy": "rollout",
                    "matched_compute_manifest": {
                        "planned": None,
                        "actual": {"arm": "B", "stage": "stage_b_self_fed"},
                        "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
                        "budget": {
                            "unit": "supervised_transitions",
                            "total": 8,
                            "units": 4,
                            "supervised_transitions": 8,
                            "train_seconds": 1.0,
                            "validated_point": 20,
                            "skipped_points": [],
                        },
                    },
                },
            },
            # The loader attaches the completed run's summary; this one records only the run and its arm.
            dataframe.COMPLETED_RUN_SUMMARY_ATTR: {
                "task": "transient_drying",
                "run_name": "transient-fixture",
                "ended_at": "2026-11-11T00:00:00+00:00",
                "terminal_controller": {"arm": "B"},
            },
        }
    )
    return frame


def _record(case_id: str, material: str, mode: str, horizon: int | str, *, level: float, offset: float, checkpoint: str) -> SimpleNamespace:
    """Return a record on a 12-hour grid whose states vary in space and time, scaled by ``level``, and whose error grows from the origin."""
    length = {1: 2, 2: 3}.get(horizon, 4) if isinstance(horizon, int) else 4
    record = sequence_record(mode, horizon, case_id=case_id, material_family=material, length=length)
    record.identity.update(dataset_role="id", checkpoint_identity={"best_checkpoint_sha256": checkpoint})
    record.physical_times = np.arange(length, dtype=np.float64) * _STEP_HOURS
    final_time = float(record.physical_times[-1])
    record.target.update(
        predicted_final_time=final_time, predicted_time_to_target=_TARGET_HOURS + 3.0 * offset, predicted_final_gap=-0.04 + 0.01 * offset
    )
    if mode == "autonomous_full":
        record.target.update(
            reference_final_time=final_time, reference_reached=True, reference_time_to_target=_TARGET_HOURS, reference_final_gap=-0.04
        )
    ramp = np.linspace(0.0, 1.0, GRID_SHAPE[0] * GRID_SHAPE[1]).reshape(GRID_SHAPE)
    levels = (np.array([300.0, 0.3, 0.5, 0.6]), np.array([10.0, 0.1, 0.1, 0.1]), np.array([10.0, 0.1, 0.2, 0.2]))
    base, step, spatial = (level.reshape(1, 4, 1, 1) for level in levels)
    reference = base + step * np.arange(length).reshape(-1, 1, 1, 1) + (0.6 + 0.4 * level) * spatial * ramp
    growth = np.arange(length).reshape(-1, 1, 1, 1) / max(length - 1, 1)
    record.reference_states = reference.astype(np.float32)
    record.predicted_states = (reference + 0.3 * offset * growth * spatial * (ramp - 0.2)).astype(np.float32)
    record.static_conditioning[artifact.STATIC_ORDER.index("rho_bu_dry")] = 700.0 + 100.0 * ramp
    return record


@cache
def _session() -> session_module.TransientEvaluationSession:
    """Return a saved two-run Evaluation session: full and rolling-origin records of two materials per run."""
    frames = {}
    for name, run_name, checkpoint, scale in _RUNS:
        records = tuple(
            _record(case, material, mode, horizon, level=level, offset=scale * level, checkpoint=checkpoint)
            for case, material, level in _MATERIALS
            for mode, horizon in (("autonomous_full", "full"), ("rolling_origin", 1), ("rolling_origin", 2))
        )
        frames[name] = production_frame(records, artifact_root=f"synthetic-catalog/{run_name}")
        frames[name].attrs["artifact_provenance"]["run"].update(name=run_name, best_checkpoint_sha256=checkpoint)
    return session_module.TransientEvaluationSession(frames)


def _records() -> dict[str, tuple[Any, ...]]:
    """Return the full-autonomous records of every run of the session."""
    evaluation = _session()
    return {name: evaluation.full_autonomous_records(name) for name in evaluation.frame_names}


def _eda_case(position: int, shift: float) -> dict[str, object]:
    """Return one Source case of the EDA frames: a reached case stops at 24 h, a right-censored one at its 48 h horizon."""
    reached = position == 0
    hours = np.asarray((0.0, 1.0, 24.0 if reached else 48.0))
    y, x = np.meshgrid(np.linspace(0.0, 0.3, _EDA_SHAPE[0]), np.linspace(0.0, 0.9, _EDA_SHAPE[1]), indexing="ij")
    ramp = x / 0.9
    levels = (300.0 + shift + position, 0.5, 20.0, 25.0)
    states = {
        field.name: np.stack([level * (1.0 + 0.04 * step + 0.08 * ramp) for step in range(len(hours))])
        for field, level in zip(step_contract.TRANSIENT_STEP_CONTRACT.dynamic_state, levels, strict=True)
    }
    schedule_end = float(hours[-1])
    return {
        "state_trajectories": states,
        "static_fields": {
            "x": x,
            "y": y,
            "u": x + 1.0,
            "v": y + 2.0,
            "p": 100.0 * x - 50.0 * y,
            "eps_bed": np.full(_EDA_SHAPE, 0.4),
            "rho_bu_dry": np.full(_EDA_SHAPE, 650.0),
            "p_in_bc": np.full(_EDA_SHAPE, 50.0 + 10.0 * position),
        },
        "boundary_intervals": {},
        "scalar_conditioning": {},
        "time": {
            "regular_state_hours": hours,
            "valid_state_mask": np.ones(len(hours), dtype=bool),
            "trajectory_length": len(hours),
            "configured_horizon_hours": 48.0,
            "classification_tolerance_hours": 1.0e-9,
        },
        "schedule": {
            "t": np.asarray((0.0, schedule_end / 2.0, schedule_end)),
            "T_in_bc": np.asarray((310.0, 315.0, 320.0)) + shift + position,
            "omega_in_bc": np.asarray((0.005, 0.006, 0.007)),
        },
        "exact_stop": None,
        "meta": {"material_family": "lentil"},
        "completion": {
            "target_reached": reached,
            "right_censored": not reached,
            "physical_duration_hours": float(hours[-1]),
            "time_to_target_hours": 24.0 if reached else None,
            "final_wet_fraction": 0.04 if reached else 0.7,
            "target_wet_fraction_limit": 0.05,
            "final_bulk_moisture_wb": 0.10 if reached else 0.28,
            "target_moisture_wb": 0.12,
        },
        "runtime": {
            "stationary_airflow_solver_seconds": 2.0,
            "transient_drying_solver_seconds": 7.0,
            "scientific_solver_seconds": 9.0,
            "comsol_process_seconds": 12.0,
            "queue_wait_seconds": 1.0,
            "licence_wait_seconds": 0.5,
            "generation_compute_end_to_end_seconds": 13.0,
            "complete_execution_seconds": 14.0,
        },
    }


@cache
def _eda_datasets() -> dict[str, pd.DataFrame]:
    """Return two Source batches of two cases each, with schedules, trajectories and completion evidence."""
    task = domain.tasks.registry.get_task("transient_drying")
    names = tuple(field.name for field in step_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    statics = ("x", "y", "u", "v", "p", "eps_bed", "rho_bu_dry", "p_in_bc")
    units = {**{field.name: field.unit for field in step_contract.TRANSIENT_STEP_CONTRACT.dynamic_state}}
    units.update(x="m", y="m", u="m/s", v="m/s", p="Pa", eps_bed="1", rho_bu_dry="kg/m^3", p_in_bc="Pa")
    frames = {}
    for label, shift in zip(_EDA_LABELS, (0.0, 5.0), strict=True):
        frame = pd.DataFrame([_eda_case(position, shift) for position in range(2)], index=pd.Index(["case_0001", "case_0002"], name="sample_id"))
        frame.attrs.update(
            {
                "task_id": task.id,
                "task_contract_digest": task.contract_digest,
                "field_names": names,
                "field_categories": {"dynamic_state": names, "static_spatial": statics},
                "field_units": units,
                "field_representations": dict.fromkeys(names, "absolute_physical_state"),
                "source_manifest_sha256": "a" * 64,
                "total_discovered_case_count": 2,
                "failed_case_count": 0,
                "incomplete_case_count": 0,
                "exclusion_reasons": {},
                "case_accounting_scope": "synthetic_catalog",
            }
        )
        frames[label] = frame
    return frames


def _eda_styles() -> dict[str, Any]:
    """Return the two batches' styles: one material, so they share its color and tell themselves apart by the line style of their roles."""
    return eda_plots.batch_styles(dict.fromkeys(_EDA_LABELS, "lentil"), dict(zip(_EDA_LABELS, ("id", "parameter_ood"), strict=True)))


def _draw_map_grid_02() -> object:
    """Draw the retained spatial fields of two Source batches."""
    sets = _eda_datasets()
    return eda_plots.plot_spatial_field_comparison(
        datasets=sets, case_ids=dict.fromkeys(sets, "case_0001"), fields=("T", "p"), physical_time_hours=1.0
    )


def _draw_map_grid_08() -> object:
    """Draw reference, prediction and error maps of one full rollout."""
    return plots.plot_state_maps(_records()["FNO run A"][0], state_fields=("T", "phi", "w_surf"), physical_time=_STEP_HOURS)


def _draw_map_grid_09() -> object:
    """Draw the mean error maps of both runs at one exact time."""
    return plots.plot_aggregate_error_maps(_records(), state_fields=("T", "phi"), physical_time=_STEP_HOURS, statistic="mean")


def _draw_map_plus_lines_03() -> object:
    """Draw transient states with the inlet schedules of two Source batches."""
    sets = _eda_datasets()
    return eda_plots.plot_state_snapshot_comparison(
        datasets=sets,
        case_ids=dict.fromkeys(sets, "case_0001"),
        physical_time_hours=1.0,
        styles=_eda_styles(),
        channels=("T", "phi"),
    )


def _draw_time_series_02() -> object:
    """Draw the aggregated schedules in hours and the reached cases' trajectories over the drying phase of two Source batches."""
    return eda_plots.plot_state_trajectory_summary(
        datasets=_eda_datasets(), max_cases=2, styles=_eda_styles(), channels=("T", "phi"), time_axis="phase"
    )


def _draw_time_series_04() -> object:
    """Draw the mean reference and predicted trajectories of both runs over the drying phase, the pooled views' default."""
    return plots.plot_state_trajectory_summary(_records(), state_fields=_FIELDS, styles=_session().series_styles(), time_axis="phase")


def _draw_time_series_05() -> object:
    """Draw the error of both runs over the drying phase, the pooled views' default."""
    evaluation = _session()
    scales = {name: evaluation.scaling_state(name) for name in evaluation.frame_names}
    return plots.plot_error_over_physical_time(
        _records(), scaling_states=scales, state_fields=_FIELDS, styles=evaluation.series_styles(), time_axis="phase"
    )


def _draw_time_series_06() -> object:
    """Draw the error over the rollout horizon."""
    evaluation = _session()
    return plots.plot_horizon_error(evaluation.dataset_dataframe(), styles=evaluation.series_styles())


def _draw_time_series_07() -> object:
    """Draw the matched-compute training performance of the A0, A1 and B arms of two materials, whose lineage admits all three arms."""
    arms = {
        name: comparison.TrainingArmEvidence(
            arm=name,
            parent_id="parent",
            architecture_id="fno",
            dataset_id="dataset",
            split_id="split",
            input_profile_id="profile",
            scaling_id="scaling",
            boundary_representation_id="boundary",
            strategy="rollout",
            planned_post_handoff_compute=compute,
            actual_post_handoff_compute=compute,
            safe_boundary_overrun=0.0,
        )
        for name, compute in (("A0", 0.0), ("A1", 8.0), ("B", 8.0))
    }
    rows = [
        {
            "comparison_arm": name,
            "material_family": material,
            "optimizer_device_compute": arms[name].actual_post_handoff_compute,
            "training_compute": arms[name].actual_post_handoff_compute,
            "compute_clock": "supervised_transitions",
            "normalized_drying_group_macro_rmse": error * scale,
            "checkpoint_role": "best",
            "mode": "autonomous_full",
            "requested_horizon": "full",
            "scope": "cumulative",
        }
        for material, scale in (("lentil", 1.0), ("chickpea", 1.4))
        for name, error in (("A0", 0.30), ("A1", 0.22), ("B", 0.16))
    ]
    lineage = comparison.LineageComparison(
        a0=arms["A0"], a1=arms["A1"], b=arms["B"], primary_comparison="B versus A1 at matched compute", separate_comparison="A0 as the shared handoff"
    )
    return plots.plot_training_performance_vs_compute(pd.DataFrame(rows), lineage=lineage)


def _draw_scatter_parity_06() -> object:
    """Draw predicted against reference channel values of both runs at one exact time."""
    return plots.plot_predicted_vs_reference_channels(_records(), state_fields=_FIELDS, physical_time=_STEP_HOURS, styles=_session().series_styles())


def _draw_scatter_parity_07() -> object:
    """Draw the endpoint against the cumulative error of both runs."""
    evaluation = _session()
    return plots.plot_endpoint_vs_cumulative(evaluation.dataset_dataframe(), styles=evaluation.series_styles())


def _draw_scatter_parity_08() -> object:
    """Draw the target-time comparison and the accuracy-against-runtime comparison of both runs."""
    evaluation = _session()
    styles = evaluation.series_styles()
    targets = {name: evaluation.full_autonomous_summaries(name) for name in evaluation.frame_names}
    reports = {name: evaluation.timing_report(name) for name in evaluation.frame_names}
    cases = evaluation.case_dataframe(modes=("autonomous_full",))
    cumulative = cases.loc[(cases["scope"] == "cumulative") & (cases["requested_horizon"] == "full")]
    return plots.plot_target_time(targets, styles=styles), plots.plot_accuracy_runtime_comparison(cumulative, reports, styles=styles)


def _draw_bar_compare_05() -> object:
    """Draw the transient channel error of both runs."""
    evaluation = _session()
    return plots.plot_channel_error(evaluation.dataset_dataframe(), state_fields=_FIELDS, styles=evaluation.series_styles())


def _draw_bar_compare_06() -> object:
    """Draw the pipeline degradation bars."""
    pipeline = comparison.AirflowDegradationMetrics(
        metric_id="synthetic",
        drying_surrogate_error=0.10,
        complete_pipeline_error=0.15,
        airflow_substitution_discrepancy=0.05,
        signed_airflow_degradation=0.05,
        airflow_degradation_ratio=1.5,
        upstream_airflow_error=0.20,
        c_available=True,
        c_unavailable_reason=None,
    )
    return plots.plot_pipeline_degradation((pipeline,))


def _draw_scatter_parity_12() -> object:
    """Draw the duration ratios of one run as points with whiskers on a log scale."""
    return plots.plot_timing_speedups(_session().timing_report("FNO run A"))


def _draw_table_06() -> object:
    """Draw the transient evaluation summary table of both runs."""
    return panel._transient_summary_table(_session())  # noqa: SLF001 -- the panel's own private table builder is the catalogued builder


def _draw_own_01() -> object:
    """Draw the completion-target analysis of two Source batches."""
    return eda_plots.plot_completion_target_analysis(datasets=_eda_datasets(), styles=_eda_styles())


_EDA: Final = "src.analysis.eda.plots.eda_plot_transient"
_EVALUATION: Final = "src.analysis.evaluation.evaluation_plot.evaluation_plot_transient"
ENTRIES: Final[tuple[CatalogPlot, ...]] = (
    CatalogPlot("map-grid_02", "Fields x datasets", (f"{_EDA}.plot_spatial_field_comparison",), _draw_map_grid_02),
    CatalogPlot("map-grid_08", "Transient sample states", (f"{_EVALUATION}.plot_state_maps",), _draw_map_grid_08),
    CatalogPlot("map-grid_09", "Aggregate transient error maps", (f"{_EVALUATION}.plot_aggregate_error_maps",), _draw_map_grid_09),
    CatalogPlot("map-plus-lines_03", "Transient snapshots with context", (f"{_EDA}.plot_state_snapshot_comparison",), _draw_map_plus_lines_03),
    CatalogPlot("time-series_02", "Dataset trajectories", (f"{_EDA}.plot_state_trajectory_summary",), _draw_time_series_02),
    CatalogPlot("time-series_04", "Reference and predicted trajectories", (f"{_EVALUATION}.plot_state_trajectory_summary",), _draw_time_series_04),
    CatalogPlot("time-series_05", "Error over drying phase", (f"{_EVALUATION}.plot_error_over_physical_time",), _draw_time_series_05),
    CatalogPlot("time-series_06", "Horizon error", (f"{_EVALUATION}.plot_horizon_error",), _draw_time_series_06),
    CatalogPlot(
        "time-series_07", "Training performance over compute", (f"{_EVALUATION}.plot_training_performance_vs_compute",), _draw_time_series_07
    ),
    CatalogPlot(
        "scatter-parity_06", "Predicted versus reference channels", (f"{_EVALUATION}.plot_predicted_vs_reference_channels",), _draw_scatter_parity_06
    ),
    CatalogPlot("scatter-parity_07", "Endpoint versus cumulative error", (f"{_EVALUATION}.plot_endpoint_vs_cumulative",), _draw_scatter_parity_07),
    CatalogPlot(
        "scatter-parity_08",
        "Target time and runtime comparisons",
        (f"{_EVALUATION}.plot_target_time", f"{_EVALUATION}.plot_accuracy_runtime_comparison"),
        _draw_scatter_parity_08,
    ),
    CatalogPlot(
        "scatter-parity_12",
        "Duration ratios, points with whiskers on a log scale",
        (f"{_EVALUATION}.plot_timing_speedups",),
        _draw_scatter_parity_12,
    ),
    CatalogPlot("bar-compare_05", "Transient channel error", (f"{_EVALUATION}.plot_channel_error",), _draw_bar_compare_05),
    CatalogPlot("bar-compare_06", "Pipeline degradation bars", (f"{_EVALUATION}.plot_pipeline_degradation",), _draw_bar_compare_06),
    CatalogPlot("table_06", "Transient evaluation summary", ("src.analysis.evaluation.evaluation_panel._transient_summary_table",), _draw_table_06),
    CatalogPlot("own_01", "Completion target", (f"{_EDA}.plot_completion_target_analysis",), _draw_own_01),
)
