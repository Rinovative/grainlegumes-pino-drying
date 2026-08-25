# ruff: noqa: S101
"""Protect transient Evaluation aggregation, presentation, and local reports."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from inspect import Parameter, signature
from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure

from src import analysis, datasets, domain, experiments
from src.analysis.evaluation import evaluation_transient_session as session
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.learning.transient.learning_transient_contracts import TemporalConditioningSpec, TransientTensorizerSpec
from src.learning.transient.learning_transient_scaling import SCALE_FLOOR, TransientScalingArtifact

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_TRANSIENT_TRACKING_SUMMARY_LIMIT = 470
_TRANSIENT_PUBLICATION_SUMMARY_LIMIT = 512
_SCOPE_DETAIL_CONTROL_COUNT = 3


def _scaling_state() -> dict[str, object]:
    spec = TransientTensorizerSpec("canonical_physics_complete_v1", TemporalConditioningSpec("none"))
    contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
    names = tuple(
        tuple(field.name for field in group)
        for group in (contract.dynamic_state, contract.static_spatial_conditioning, contract.step_boundary_conditioning, contract.scalar_conditioning)
    )
    artifact = TransientScalingArtifact(
        task_contract_digest=domain.tasks.registry.get_task("transient_drying").contract_digest,
        data_contract_digest=domain.tasks.registry.get_task("transient_drying").data_contract_digest,
        tensorizer=spec,
        dataset_identity={"synthetic": "session"},
        train_membership_digest="a" * 64,
        scale_mode="state_std",
        numerical_floor=SCALE_FLOOR,
        unique_train_state_count=1,
        unique_transition_count=1,
        transition_count=1,
        spatial_shape=(1, 1),
        state_names=names[0],
        static_names=names[1],
        boundary_names=names[2],
        scalar_names=names[3],
        state_mean=torch.zeros(4),
        state_std=torch.full((4,), 2.0),
        delta_rms=torch.ones(4),
        increment_scale=torch.full((4,), 2.0),
        static_mean=torch.zeros(7),
        static_std=torch.ones(7),
        scalar_mean=torch.zeros(8),
        scalar_std=torch.ones(8),
        omega_boundary_mean=torch.tensor(0.0),
        omega_boundary_std=torch.tensor(1.0),
        horizon=8.0,
    )
    return artifact.state_dict()


def _record(
    mode: str,
    horizon: int | str,
    *,
    case_id: str = "case",
    offset: float = 2.0,
    material_family: str = "lentil",
) -> SimpleNamespace:
    length = 2 if horizon == 1 else 3
    reference = np.zeros((length, 4, 1, 1), dtype=np.float32)
    prediction = reference.copy()
    prediction[1:] = offset
    full_reference = mode == "autonomous_full" and horizon == "full"
    target = {
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
        spatial_mask=np.ones((1, 1), dtype=bool),
        static_conditioning=np.asarray(
            [
                [[0.0]],
                [[0.0]],
                [[1.0]],
                [[1.0]],
                [[101325.0]],
                [[0.4]],
                [[100.0]],
            ],
            dtype=np.float32,
        ),
        scalar_conditioning=np.asarray([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
        identity={
            "material_family": material_family,
            "input_profile": "canonical_physics_complete_v1",
            "model_kind": "fno",
            "dataset_backend": "canonical_hdf5",
            "timing_evidence_identity": "b" * 64,
            "scaling_identity": {"semantic_digest": "c" * 64},
        },
        target=target,
    )


def _frame(records: tuple[SimpleNamespace, ...]) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "transient_sequence_records": records,
            "transient_unavailable_horizons": ({"case_id": "case", "requested_horizon": 4, "reason": "short"},),
            "transient_scaling_state": _scaling_state(),
        }
    )
    return frame


def _production_frame(
    records: tuple[SimpleNamespace, ...],
    *,
    artifact_root: str,
    dataset_name: str = "transient_dataset",
    dataset_role: str = "id",
) -> pd.DataFrame:
    frame = _frame(records)
    report = timing.build_transient_timing_report(
        (
            timing.TransientTimingCase(
                case_id="case",
                repetitions={
                    "drying_no_rollout_model_seconds": (0.01, 0.02, 0.03),
                    "drying_no_end_to_end_seconds": (0.02, 0.03, 0.04),
                    "comsol_transient_drying_seconds": (0.3,),
                },
                device="cpu",
                precision="float32",
                dataset_backend="canonical_hdf5",
                warmup_passes=1,
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
                "run": {
                    "best_checkpoint_sha256": "d" * 64,
                    "best_checkpoint_epoch": 3,
                },
                "dataset": {
                    "name": dataset_name,
                    "role": dataset_role,
                    "source_dataset_ids": [dataset_name],
                    "membership_digests": ["e" * 64],
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
                    "component_availability": {
                        "case": {"dataset_materialization_seconds": 0.005},
                    },
                    "timing_report": json.loads(json.dumps(asdict(report))),
                },
                "lineage": {
                    "stage_identity": {"stage": "b", "comparison_arm": "b"},
                    "training_strategy": "rollout",
                    "matched_compute_manifest": {
                        "planned": {"clock": "cuda_device_seconds"},
                        "actual": {"budget_complete": True},
                    },
                },
            },
        }
    )
    return frame


def _channel_capability_frame(
    fields: tuple[str, ...],
    *,
    bulk_moisture: bool = True,
    metadata: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Return exact test-owned transient presentation provenance."""
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "artifact_provenance": {
                "evaluation": {
                    "objective": {"fields": list(fields)},
                    "field_metadata": {} if metadata is None else metadata,
                    "process_diagnostic_policy": {"bulk_moisture": {"available": bulk_moisture}},
                }
            },
        }
    )
    return frame


def test_transient_channel_resolution_uses_metadata_derivation_and_intersection() -> None:
    """Resolve current, derived, future, omitted, and compared channels exactly."""
    current = _channel_capability_frame(analysis.evaluation.transient_artifact.STATE_ORDER)
    resolution = analysis.evaluation.presentation.transient_channel_resolution((current,))
    assert resolution.keys == (
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    )
    grain = next(field for field in resolution.fields if field.key == "w_gr")
    assert grain.stored is False
    assert grain.dependencies == ("w_surf", "w_int", "f_surf")
    assert grain.unit == "kg/m^3"

    future = _channel_capability_frame(
        (*analysis.evaluation.transient_artifact.STATE_ORDER, "future_state"),
        metadata={
            "future_state": {
                "label": "Future state",
                "unit": "mol/m^3",
            }
        },
    )
    future_resolution = analysis.evaluation.presentation.transient_channel_resolution((future,))
    assert future_resolution.keys == (
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
        "future_state",
    )
    assert future_resolution.fields[-1].label == "Future state"
    assert future_resolution.fields[-1].unit == "mol/m^3"

    missing = _channel_capability_frame(
        (*analysis.evaluation.transient_artifact.STATE_ORDER, "undeclared"),
    )
    missing_resolution = analysis.evaluation.presentation.transient_channel_resolution((missing,))
    assert "undeclared" not in missing_resolution.keys
    assert "authoritative label and unit" in missing_resolution.omitted["undeclared"]

    no_derived = _channel_capability_frame(
        analysis.evaluation.transient_artifact.STATE_ORDER,
        bulk_moisture=False,
    )
    assert "w_gr" not in analysis.evaluation.presentation.transient_channel_resolution((no_derived,)).keys

    shared = analysis.evaluation.presentation.transient_channel_resolution((future, current))
    assert shared.keys == (
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    )
    assert shared.omitted["future_state"] == "not supplied by every compared artifact"

    conflicting_metadata = _channel_capability_frame(
        analysis.evaluation.transient_artifact.STATE_ORDER,
        metadata={"w_surf": {"unit": "g/m^3"}},
    )
    conflicting_attribute = _channel_capability_frame(analysis.evaluation.transient_artifact.STATE_ORDER)
    conflicting_attribute.attrs["field_units"] = {"w_int": "g/m^3"}
    for conflicting, field in (
        (conflicting_metadata, "w_surf"),
        (conflicting_attribute, "w_int"),
    ):
        conflict_resolution = analysis.evaluation.presentation.transient_channel_resolution((conflicting,))
        assert field not in conflict_resolution.keys
        assert "canonical unit" in conflict_resolution.omitted[field]
        assert "w_gr" not in conflict_resolution.keys


def test_indexed_session_pools_unequal_statistics_without_loading_payloads(
    tmp_path: Path,
) -> None:
    """Pool float64 sums rather than case RMSE and keep aggregate paths payload-free."""
    short = _record(
        "autonomous_full",
        "full",
        case_id="short",
        offset=1.0,
    )
    short.reference_states = short.reference_states[:2]
    short.predicted_states = short.predicted_states[:2]
    short.physical_times = short.physical_times[:2]
    records = (
        short,
        _record("autonomous_full", "full", case_id="long", offset=3.0),
    )
    scaling = TransientScalingArtifact.from_state_dict(_scaling_state())

    def statistics(record: SimpleNamespace) -> dict[str, dict[str, object]]:
        """Build test-owned persisted states through the production accumulator."""
        result: dict[str, dict[str, object]] = {}
        for scope in ("cumulative", "endpoint"):
            prediction = record.predicted_states[1:] if scope == "cumulative" else record.predicted_states[-1:]
            reference = record.reference_states[1:] if scope == "cumulative" else record.reference_states[-1:]
            accumulator = analysis.evaluation.transient_metrics.TransientMetricAccumulator(scope=scope)
            mask = np.broadcast_to(
                record.spatial_mask[None, None],
                prediction.shape,
            ).copy()
            accumulator.update(
                normalized_prediction=(scaling.encode_state(torch.from_numpy(prediction)).detach().cpu().numpy()),
                normalized_reference=(scaling.encode_state(torch.from_numpy(reference)).detach().cpu().numpy()),
                physical_prediction=prediction,
                physical_reference=reference,
                f_surf=np.asarray(record.scalar_conditioning[2]),
                rho_bu_dry=record.static_conditioning[analysis.evaluation.transient_artifact.STATIC_ORDER.index("rho_bu_dry")],
                cell_weights=analysis.evaluation.transient_metrics.trapezoidal_cell_weights(record.spatial_mask),
                valid_mask=mask,
            )
            result[scope] = accumulator.state_dict()
        diagnostic_prediction = record.predicted_states.copy()
        if record.case_id == "long":
            diagnostic_prediction[1, 0, 0, 0] = 3_000.0
        result["diagnostics"] = {
            "plausibility": asdict(
                analysis.evaluation.transient_metrics.derive_plausibility_diagnostics(
                    diagnostic_prediction[1:],
                    temperature_range=analysis.evaluation.transient_metrics.TEMPERATURE_PLAUSIBILITY_RANGE_K,
                )
            ),
            "stability": asdict(analysis.evaluation.transient_metrics.derive_stability_diagnostics(diagnostic_prediction)),
        }
        return result

    summaries = tuple(
        analysis.evaluation.transient_artifact.TransientSequenceRecordSummary(
            record_id=f"record-{index}",
            mode="autonomous_full",
            case_id=record.case_id,
            dataset_role="id",
            origin_index=0,
            requested_horizon="full",
            available_horizon=len(record.physical_times) - 1,
            trajectory_length=len(record.physical_times),
            origin_time=0.0,
            elapsed_physical_time=float(record.physical_times[-1]),
            payload_path=f"npz/{record.case_id}.npz",
            payload_sha256=str(index) * 64,
            chain_id=f"chain-{index}",
            metric_statistics=statistics(record),
            identity=record.identity,
            target=record.target,
            timing={},
            exclusion={},
        )
        for index, record in enumerate(records, start=1)
    )
    frame = pd.DataFrame()
    index = analysis.evaluation.transient_artifact.TransientSequenceArtifactIndex(
        root=tmp_path,
        dataset_name="transient_dataset",
        dataset_role="id",
        summaries=summaries,
        unavailable_horizons=(),
        frame=frame,
        provenance={},
        identity_sha256="f" * 64,
        _rows={summary.record_id: {} for summary in summaries},
    )
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "transient_sequence_index": index,
            "transient_unavailable_horizons": (),
            "transient_scaling_state": _scaling_state(),
        }
    )
    evaluation = session.TransientEvaluationSession({"id": frame})

    cases = evaluation.case_dataframe(modes=("autonomous_full",))
    datasets_frame = evaluation.dataset_dataframe(modes=("autonomous_full",))
    cumulative = datasets_frame.loc[datasets_frame["scope"] == "cumulative"].iloc[0]
    expected_normalized = np.sqrt((0.5**2 + 2 * 1.5**2) / 3)
    expected_physical = np.sqrt((1.0**2 + 2 * 3.0**2) / 3)

    assert set(cases["case_id"]) == {"short", "long"}
    assert set(cases["scope"]) == {"cumulative", "endpoint"}
    assert cumulative["normalized_drying_group_macro_rmse"] == pytest.approx(expected_normalized)
    assert cumulative["physical_w_gr_rmse"] == pytest.approx(expected_physical)
    assert cumulative["normalized_drying_group_macro_rmse"] != pytest.approx(1.0)
    assert index.cache_size == 0
    assert evaluation.cache_sizes["payload_records"] == 0
    assert cumulative["temperature_range_violations"] > 0
    compact = evaluation.full_autonomous_summaries()
    assert tuple(summary.case_id for summary in compact) == ("short", "long")
    assert index.cache_size == 0
    evaluation.close()


def test_session_uses_saved_scale_excludes_origin_and_retains_default_modes() -> None:
    """Use persisted scaling, omit origins, and retain all three required modes."""
    values = (_record("teacher_forced_one_step", 1), _record("autonomous_full", "full"), _record("rolling_origin", 2))
    evaluation = session.TransientEvaluationSession({"id": _frame(values)})
    summaries = evaluation.summaries()
    teacher_cumulative = next(item for item in summaries if item.mode == "teacher_forced_one_step" and item.scope == "cumulative")
    assert teacher_cumulative.metrics.normalized_drying_group_macro_rmse == pytest.approx(1.0)
    assert teacher_cumulative.metrics.physical_w_gr_rmse == pytest.approx(2.0)
    assert teacher_cumulative.metrics.bulk_dry_basis_rmse == pytest.approx(0.02)
    assert teacher_cumulative.metrics.bulk_wet_basis_rmse == pytest.approx(2.0 / 102.0)
    assert teacher_cumulative.metrics.bulk_moisture_valid_count == 1
    assert teacher_cumulative.contributing_record_count == 1
    assert set(evaluation.case_dataframe()["mode"]) == {"teacher_forced_one_step", "autonomous_full", "rolling_origin"}


def test_target_records_select_one_full_autonomous_outcome_per_case() -> None:
    """Avoid multiplying target outcomes across rollout modes, origins, and horizons."""
    records = (
        _record("teacher_forced_one_step", 1, case_id="a"),
        _record("autonomous_full", "full", case_id="a"),
        _record("rolling_origin", 2, case_id="a"),
        _record("autonomous_full", "full", case_id="b"),
        _record("rolling_origin", 4, case_id="b"),
    )
    evaluation = session.TransientEvaluationSession({"id": _frame(records)})
    selected = evaluation.full_autonomous_records()
    assert tuple(record.case_id for record in selected) == ("a", "b")
    assert all(record.mode == "autonomous_full" and record.requested_horizon == "full" for record in selected)


def test_session_keeps_material_summaries_and_unavailability_separate() -> None:
    """Group sufficient statistics by identity-owned material without parsing case IDs."""
    records = (
        _record(
            "rolling_origin",
            4,
            case_id="chickpea__case_0001",
            offset=2.0,
            material_family="chickpea",
        ),
        _record(
            "rolling_origin",
            4,
            case_id="lentil__case_0001",
            offset=4.0,
            material_family="lentil",
        ),
    )
    frame = _frame(records)
    frame.attrs["transient_unavailable_horizons"] = (
        {
            "case_id": "chickpea__case_0001",
            "requested_horizon": 4,
            "reason": "short",
        },
        {
            "case_id": "lentil__case_0001",
            "requested_horizon": 4,
            "reason": "short",
        },
    )
    evaluation = session.TransientEvaluationSession({"id": frame})

    summaries = evaluation.summaries(modes=("rolling_origin",))
    cumulative = {item.material_family: item for item in summaries if item.scope == "cumulative"}
    dataset_frame = evaluation.dataset_dataframe(modes=("rolling_origin",))
    case_frame = evaluation.case_dataframe(modes=("rolling_origin",))

    assert set(cumulative) == {"chickpea", "lentil"}
    assert cumulative["chickpea"].contributing_case_count == 1
    assert cumulative["lentil"].contributing_case_count == 1
    assert cumulative["chickpea"].unavailable_case_count == 1
    assert cumulative["lentil"].unavailable_case_count == 1
    assert set(dataset_frame["material_family"]) == {"chickpea", "lentil"}
    assert set(case_frame["material_family"]) == {"chickpea", "lentil"}
    assert evaluation.material_families("id") == ("chickpea", "lentil")
    assert evaluation.case_ids("id", material_family="chickpea") == ("chickpea__case_0001",)


def test_session_aggregates_cases_scopes_unavailability_and_target_evidence() -> None:
    """Aggregate sufficient statistics while preserving scope and censoring evidence."""
    records = (_record("rolling_origin", 4, case_id="a", offset=2.0), _record("rolling_origin", 4, case_id="b", offset=4.0))
    frame = _frame(records)
    frame.attrs["transient_unavailable_horizons"] = ({"case_id": "a", "requested_horizon": 4, "reason": "short"},)
    evaluation = session.TransientEvaluationSession({"id": frame})
    summaries = evaluation.summaries(modes=("rolling_origin",))
    cumulative = next(item for item in summaries if item.scope == "cumulative")
    endpoint = next(item for item in summaries if item.scope == "endpoint")
    assert cumulative.metrics.normalized_drying_group_macro_rmse == pytest.approx(np.sqrt((1.0**2 + 2.0**2) / 2.0))
    assert endpoint.metrics.normalized_drying_group_macro_rmse == pytest.approx(cumulative.metrics.normalized_drying_group_macro_rmse)
    assert (cumulative.contributing_case_count, cumulative.unavailable_case_count) == (2, 1)
    assert (cumulative.target.available_count, cumulative.target.reference_right_censored_count) == (0, 0)
    assert cumulative.target_time_error_count == 0
    assert cumulative.target_gap_count == 0
    assert cumulative.predicted_final_target_gap_mean == pytest.approx(-0.1)
    assert cumulative.reference_final_target_gap_mean is None
    assert cumulative.target_final_gap_error_mean is None
    assert cumulative.target_final_gap_error_mae is None
    assert all(key.startswith("evaluation/") for key in evaluation.wandb_summary(modes=("rolling_origin",)))


def test_exact_stop_gap_is_not_paired_with_a_regular_prediction_endpoint() -> None:
    """Keep final-gap errors unavailable when canonical and predicted endpoints differ."""
    record = _record("autonomous_full", "full")
    target = dict(record.target)
    target.update(
        {
            "reference_reached": True,
            "reference_time_to_target": 2.5,
            "reference_final_gap": -0.2,
            "reference_final_time": 2.5,
        }
    )
    record.target = target
    evaluation = session.TransientEvaluationSession({"id": _frame((record,))})
    cumulative = next(item for item in evaluation.summaries() if item.mode == "autonomous_full" and item.scope == "cumulative")
    assert cumulative.target_time_error_count == 1
    assert cumulative.target_gap_count == 0
    assert cumulative.predicted_final_target_gap_mean == pytest.approx(-0.1)
    assert cumulative.reference_final_target_gap_mean == pytest.approx(-0.2)
    assert cumulative.target_final_gap_error_mean is None


def test_tracking_summary_stays_bounded_for_complete_two_role_horizon_inventory(tmp_path: Path) -> None:
    """Keep the complete two-role aggregate inventory within the W&B scalar cap."""
    records = (
        _record("teacher_forced_one_step", 1),
        _record("autonomous_full", "full"),
        *(_record("rolling_origin", horizon) for horizon in analysis.evaluation.transient_artifact.FIXED_HORIZONS),
        _record("rolling_origin", "full"),
    )
    frames = {
        role: _production_frame(
            records,
            artifact_root=str(tmp_path / role),
            dataset_name=f"{role}_dataset",
            dataset_role=role,
        )
        for role in ("id", "ood")
    }
    metric_id = "normalized_drying_group_macro_rmse"
    for frame in frames.values():
        pipeline = frame.attrs["artifact_provenance"]["evaluation"]["pipeline_analysis"]
        pipeline.update(
            {
                "c_available": True,
                "c_unavailable_reasons": [],
                "upstream_airflow_error": 0.1,
                "metrics": {
                    metric_id: {
                        "complete_pipeline_error": 1.5,
                        "airflow_substitution_discrepancy": 0.5,
                    }
                },
            }
        )
    evaluation = session.TransientEvaluationSession(frames)
    values = evaluation.wandb_summary()
    assert values
    assert len(values) <= _TRANSIENT_TRACKING_SUMMARY_LIMIT

    publication = evaluation.wandb_publication_summary()
    assert set(values).issubset(publication)
    assert len(publication) <= _TRANSIENT_PUBLICATION_SUMMARY_LIMIT
    assert all(key.startswith("evaluation/") for key in publication)
    assert all(isinstance(value, (bool, float, int, str)) for value in publication.values())
    validator = experiments.tracking._validate_transient_evaluation_summary_entry  # noqa: SLF001
    assert all(validator(key, value) == key for key, value in publication.items())
    assert publication["evaluation/identity/checkpoint_sha256"] == "d" * 64
    assert publication["evaluation/identity/input_profile"] == "canonical_physics_complete_v1"
    assert publication["evaluation/identity/backend"] == "canonical_hdf5"
    assert {
        "evaluation/identity/scaling_semantic_digest",
        "evaluation/identity/comparison_arm",
        "evaluation/identity/training_strategy",
        "evaluation/identity/matched_compute_manifest_sha256",
    }.isdisjoint(publication)
    assert publication["evaluation/timing/precision"] == "float32"
    assert publication["evaluation/timing/component_composed"] is True
    assert publication["evaluation/timing/component/dataset_materialization_seconds/available_count"] == len(frames)
    assert publication["evaluation/timing/component/dataset_materialization_seconds/median_seconds"] == pytest.approx(0.005)
    assert not any(fragment in key for key in publication for fragment in ("reference_states", "predicted_states", "repetitions", "case_id"))


def test_transient_registry_invokes_every_visible_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invoke every registered transient entry through explicit export context."""
    records = (
        _record("teacher_forced_one_step", 1),
        _record("autonomous_full", "full"),
        _record("rolling_origin", 2),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": _production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    captured: list[tuple[str, Callable[..., object], str]] = []

    def capture(
        plots: list[tuple[str, Callable[..., object], str]],
        **_kwargs: object,
    ) -> widgets.VBox:
        captured.extend(plots)
        return widgets.VBox()

    monkeypatch.setattr(
        analysis.ui.notebook,
        "make_dropdown_section",
        capture,
    )
    captured_tabs: list[str] = []

    def capture_tabs(
        _sections: object,
        *,
        tab_titles: tuple[str, ...] | list[str],
        **_kwargs: object,
    ) -> widgets.Label:
        captured_tabs.extend(tab_titles)
        return widgets.Label("panel")

    monkeypatch.setattr(
        analysis.ui.notebook,
        "make_lazy_panel_with_tabs",
        capture_tabs,
    )

    outer = analysis.evaluation.panel.build_transient_panel(
        session=evaluation,
    )

    assert isinstance(outer, widgets.Label)
    assert captured
    assert len({plot_name for _label, _factory, plot_name in captured}) == len(captured)
    registered_names = {plot_name for _label, _factory, plot_name in captured}
    assert captured_tabs == [
        "Overview",
        "Global Error Analysis",
        "Architecture Sensitivity",
        "Error Decomposition",
        "Sample Viewer",
        "Outlier & Extreme Case Analysis",
        "Temporal & Rollout Analysis",
    ]
    assert tuple(label for label, _factory, _plot_name in captured) == (
        "Overview: Summary table",
        "1-1. Global error metrics",
        "1-2. Global error distribution",
        "1-3. GT vs Prediction (mean)",
        "1-4. Mean error maps",
        "1-5. Std error maps",
        "2-1. Model-family configuration",
        "3-1. Error vs |GT| magnitude",
        "7-1. Sample GT vs Prediction",
        "8-1. Worst per-channel cases (tables)",
        "8-2. Worst per-channel cases (field plots)",
        "9-1. Reference vs prediction trajectories",
        "9-2. Error over physical time",
        "9-3. Error vs rollout horizon",
        "9-4. Final-state drying summary",
    )
    assert {
        "1_3_gt_vs_prediction_mean",
        "1_4_mean_error_maps",
        "7_1_sample_gt_vs_prediction",
        "9_1_reference_prediction_trajectories",
    }.issubset(registered_names)
    assert not any(
        fragment in plot_name for plot_name in registered_names for fragment in ("pressure", "velocity", "divergence", "permeability", "timing")
    )
    results: dict[str, object] = {}
    for _label, factory, plot_name in captured:
        parameters = signature(factory).parameters
        assert all(parameter.kind is not Parameter.VAR_KEYWORD for parameter in parameters.values())
        result = analysis.ui.notebook._invoke_dropdown_entry(  # noqa: SLF001
            factory,
            export_state={
                "fig": None,
                "plot_name": plot_name,
                "title": plot_name,
            },
            export_plot_name=plot_name,
            export_title=plot_name,
        )
        results[plot_name] = result
        if isinstance(result, Figure):
            plt.close(result)

    sample = results["7_1_sample_gt_vs_prediction"]
    error_time = results["9_2_error_over_physical_time"]
    assert isinstance(sample, widgets.VBox)
    assert isinstance(error_time, widgets.VBox)
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(sample))
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(error_time))
    evaluation.close()
    plt.close("all")


def test_transient_field_views_reuse_exact_eda_widget_structure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep case, status, channels, scale, time, output, and scope ordering exact."""
    frame = _production_frame(
        (_record("autonomous_full", "full"),),
        artifact_root=str(tmp_path / "artifact"),
    )
    frame.attrs["artifact_provenance"]["evaluation"].update(
        {
            "objective": {"fields": list(analysis.evaluation.transient_artifact.STATE_ORDER)},
            "process_diagnostic_policy": {"bulk_moisture": {"available": True}},
        }
    )
    evaluation = session.TransientEvaluationSession({"id": frame})
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)  # noqa: SLF001
    monkeypatch.setattr(analysis.ui.viewers, "render_figure", lambda **_kwargs: None)

    sample = analysis.evaluation.panel._make_transient_sample_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        comparison=False,
    )
    assert tuple(type(child) for child in sample.children) == (
        widgets.HBox,
        widgets.HTML,
        widgets.VBox,
        widgets.HBox,
        widgets.VBox,
        widgets.Output,
    )
    case_row = sample.children[0]
    assert isinstance(case_row, widgets.HBox)
    assert case_row.layout.width == "230px"
    assert case_row.children[0].layout.width == "150px"
    assert tuple(child.layout.width for child in case_row.children[1:]) == (
        "40px",
        "40px",
    )
    channel_container = sample.children[2]
    assert isinstance(channel_container, widgets.VBox)
    channel_widget = channel_container.children[1]
    assert isinstance(channel_widget, widgets.VBox)
    channel_group = cast(
        "analysis.ui.components.CheckboxGroup",
        channel_widget,
    )
    channel_grid = channel_widget.children[0]
    assert isinstance(channel_grid, widgets.GridBox)
    assert channel_grid.layout.grid_template_columns.startswith("repeat(3,")
    assert tuple(channel_group.boxes) == (
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    )
    scale_container = sample.children[3]
    time_container = sample.children[4]
    assert isinstance(scale_container, widgets.HBox)
    assert isinstance(time_container, widgets.VBox)
    assert isinstance(scale_container.children[0], widgets.Checkbox)
    time_row = time_container.children[0]
    assert isinstance(time_row, widgets.HBox)
    assert tuple(control.description for control in time_row.children) == (
        "Time [h]:",
        "≪",
        "←",
        "→",
        "≫",
    )

    scoped = analysis.evaluation.panel._make_transient_scope_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        plot_kind="trajectory",
        comparison=False,
    )
    scope_row = scoped.children[0]
    assert isinstance(scope_row, widgets.HBox)
    scope, detail = scope_row.children
    assert isinstance(detail, widgets.HBox)
    assert isinstance(scope, widgets.ToggleButtons)
    assert tuple(scope.options) == (
        ("Aggregate", "aggregate"),
        ("Single case", "single"),
    )
    assert len(detail.children) == _SCOPE_DETAIL_CONTROL_COUNT
    scope.value = "single"
    assert len(detail.children) == _SCOPE_DETAIL_CONTROL_COUNT
    assert detail.children[0].layout.width == "150px"
    assert tuple(child.layout.width for child in detail.children[1:]) == (
        "40px",
        "40px",
    )
    evaluation.close()


def _widget_tree(widget: object) -> tuple[object, ...]:
    """Return one widget and every nested child in deterministic preorder."""
    descendants = [widget]
    for child in getattr(widget, "children", ()):
        descendants.extend(_widget_tree(child))
    return tuple(descendants)


def test_aggregate_spatial_plots_use_exact_values_and_reuse_loaded_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pool exact-time case evidence and keep slider/channel changes payload-free."""
    records = (
        _record("autonomous_full", "full", case_id="low", offset=1.0),
        _record("autonomous_full", "full", case_id="high", offset=-3.0),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": _production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    spatial = analysis.evaluation.plots.transient.plot_aggregate_spatial_error(
        records,
        state_field="w_surf",
        physical_time=1.0,
    )
    comparison = analysis.evaluation.plots.transient.plot_predicted_vs_reference(
        records,
        state_field="w_surf",
        physical_time=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(spatial.axes[0].collections[0].get_array()),
        np.asarray([[2.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(comparison.axes[0].collections[0].get_offsets(), dtype=np.float64),
        np.asarray(((0.0, 1.0), (0.0, -3.0))),
    )
    assert "kg/m^3" in comparison.axes[0].get_xlabel()
    plt.close(spatial)
    plt.close(comparison)

    faceted = analysis.evaluation.plots.transient.plot_aggregate_error_maps(
        {
            "Chickpea · ID": (records[0],),
            "Kidney bean · Near-family OOD": (records[1],),
        },
        state_fields=("w_surf",),
        physical_time=1.0,
        statistic="mean",
    )
    assert tuple(axis.get_title() for axis in faceted.axes[:2]) == (
        "Chickpea · ID",
        "Kidney bean · Near-family OOD",
    )
    np.testing.assert_allclose(
        np.asarray(faceted.axes[0].collections[0].get_array()),
        np.asarray([[1.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(faceted.axes[1].collections[0].get_array()),
        np.asarray([[3.0]]),
    )
    plt.close(faceted)

    calls: list[tuple[str, str]] = []
    original = session.TransientEvaluationSession.record_for_coordinates

    def counted(
        current: session.TransientEvaluationSession,
        frame_name: str,
        case_id: str,
        **coordinates: Any,
    ) -> object:
        calls.append((frame_name, case_id))
        return original(current, frame_name, case_id, **coordinates)

    monkeypatch.setattr(
        session.TransientEvaluationSession,
        "record_for_coordinates",
        counted,
    )
    export_state: dict[str, object] = {}
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)  # noqa: SLF001
    viewer = analysis.evaluation.panel._make_transient_aggregate_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        plot_kind="mean_error",
        comparison=False,
        export_state=export_state,
        export_plot_name="spatial_error",
    )
    time_control = next(widget for widget in _widget_tree(viewer) if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]:")
    channel_group = cast(
        "analysis.ui.components.CheckboxGroup",
        next(widget for widget in _widget_tree(viewer) if hasattr(widget, "boxes")),
    )
    assert calls == [("id", "low"), ("id", "high")]
    time_control.value = 1.0
    channel_group.boxes["T"].value = False
    assert calls == [("id", "low"), ("id", "high")]
    assert "channel_phi-w_surf-w_int" in str(export_state["filename_stem"])
    assert "t_1h" in str(export_state["filename_stem"])
    evaluation.close()


def test_partitioned_case_selector_routes_materials_to_owning_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose one material-labelled union while retaining exact artifact ownership."""

    def owned_record(case_id: str, material: str, compact_case: str) -> SimpleNamespace:
        record = _record(
            "autonomous_full",
            "full",
            case_id=case_id,
            material_family=material,
        )
        record.identity["dataset_identity"] = {
            "artifact_dataset_name": f"dataset_{material}",
            "source_dataset_id": f"dataset_{material}",
        }
        record.identity["simulation_identity"] = {
            "generation_case_id": compact_case,
            "package_case_id": case_id,
        }
        return record

    id_records = (
        owned_record(
            "transient_drying__chickpea__natural__case_0051",
            "chickpea",
            "case_0051",
        ),
        owned_record(
            "transient_drying__lentil__natural__case_0010",
            "lentil",
            "case_0010",
        ),
    )
    ood_records = (
        owned_record(
            "transient_drying__kidney_bean__natural__case_0004",
            "kidney_bean",
            "case_0004",
        ),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "Model ID": _production_frame(
                id_records,
                artifact_root=str(tmp_path / "id"),
                dataset_name="transient_drying__lentil+chickpea__id",
                dataset_role="id",
            ),
            "Model OOD": _production_frame(
                ood_records,
                artifact_root=str(tmp_path / "ood"),
                dataset_name="transient_drying__kidney_bean__near_family_ood",
                dataset_role="ood",
            ),
        }
    )
    routed: list[tuple[str, str]] = []
    original = session.TransientEvaluationSession.record_for_coordinates

    def counted(
        current: session.TransientEvaluationSession,
        frame_name: str,
        case_id: str,
        **coordinates: Any,
    ) -> object:
        routed.append((frame_name, case_id))
        return original(
            current,
            frame_name,
            case_id,
            **coordinates,
        )

    monkeypatch.setattr(
        session.TransientEvaluationSession,
        "record_for_coordinates",
        counted,
    )
    monkeypatch.setattr(
        analysis.ui.viewers,
        "render_figure",
        lambda **_kwargs: None,
    )
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)  # noqa: SLF001
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        comparison=False,
    )
    case_selector = next(widget for widget in _widget_tree(viewer) if isinstance(widget, widgets.Dropdown) and widget.description == "Case:")
    assert tuple(label for label, _value in case_selector.options) == (
        "Chickpea · case_0051",
        "Lentil · case_0010",
        "Kidney bean · case_0004",
    )
    assert not any(isinstance(widget, widgets.Dropdown) and widget.description == "Model / artifact:" for widget in _widget_tree(viewer))
    case_selector.value = 2
    assert routed[-1] == (
        "Model OOD",
        "transient_drying__kidney_bean__natural__case_0004",
    )

    inventory = evaluation.partitioned_case_inventory()
    assert tuple(entry.dataset_role for entry in inventory) == ("id", "id", "ood")
    assert tuple(entry.artifact_root for entry in inventory) == (
        (tmp_path / "id").resolve(),
        (tmp_path / "id").resolve(),
        (tmp_path / "ood").resolve(),
    )
    aggregate = evaluation.dataset_dataframe(modes=("autonomous_full",))
    assert set(aggregate["material_family"]) == {
        "chickpea",
        "lentil",
        "kidney_bean",
    }
    assert set(aggregate["dataset_role"]) == {"id", "ood"}
    assert set(aggregate["artifact_root"]) == {
        str((tmp_path / "id").resolve()),
        str((tmp_path / "ood").resolve()),
    }

    routed.clear()
    aggregate_view = analysis.evaluation.panel._make_transient_aggregate_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        plot_kind="mean_error",
        comparison=False,
    )
    assert routed == [
        (
            "Model ID",
            "transient_drying__chickpea__natural__case_0051",
        ),
        (
            "Model ID",
            "transient_drying__lentil__natural__case_0010",
        ),
        (
            "Model OOD",
            "transient_drying__kidney_bean__natural__case_0004",
        ),
    ]
    assert not any(isinstance(widget, widgets.Dropdown) and widget.description == "Material:" for widget in _widget_tree(aggregate_view))
    coverage_status = next(
        widget for widget in _widget_tree(aggregate_view) if isinstance(widget, widgets.HTML) and "All material-role partitions" in widget.value
    )
    assert all(
        label in coverage_status.value
        for label in (
            "Chickpea · ID",
            "Lentil · ID",
            "Kidney bean · Near-family OOD",
        )
    )
    evaluation.close()


def test_sample_view_keeps_physical_time_local_and_rollout_coordinates_outside(
    tmp_path: Path,
) -> None:
    """Use full-rollout sample time without adding global protocol or horizon controls."""
    records = (
        _record("teacher_forced_one_step", 1),
        _record("autonomous_full", "full"),
        _record("rolling_origin", 2),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": _production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)  # noqa: SLF001
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        comparison=False,
    )
    assert not any(
        isinstance(widget, widgets.Dropdown) and widget.description in {"Protocol:", "Origin:", "Horizon:"} for widget in _widget_tree(viewer)
    )
    assert any(isinstance(widget, widgets.FloatText) and widget.description == "Time [h]:" for widget in _widget_tree(viewer))

    trajectory = analysis.evaluation.plots.transient.plot_state_trajectory(
        records[1],
        state_field="w_gr",
    )
    np.testing.assert_allclose(
        np.asarray(trajectory.axes[0].lines[0].get_ydata(), dtype=np.float64),
        (0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        np.asarray(trajectory.axes[0].lines[1].get_ydata(), dtype=np.float64),
        (0.0, 2.0, 2.0),
    )
    assert "kg/m^3" in trajectory.axes[0].get_ylabel()
    plt.close(trajectory)
    evaluation.close()


def test_outlier_field_view_starts_from_persisted_worst_channel_case(
    tmp_path: Path,
) -> None:
    """Route the per-channel outlier field view through one exact ranked case."""
    records = (
        _record("autonomous_full", "full", case_id="best", offset=1.0),
        _record("autonomous_full", "full", case_id="worst", offset=3.0),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": _production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)  # noqa: SLF001
    export_state: dict[str, object] = {}
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        comparison=False,
        export_state=export_state,
        export_plot_name="worst_cases",
        outlier=True,
    )
    case_row = viewer.children[0]
    status = viewer.children[1]
    assert isinstance(case_row, widgets.HBox)
    assert isinstance(status, widgets.HTML)
    selector = case_row.children[0]
    assert isinstance(selector, widgets.Dropdown)
    assert "worst" in str(selector.label)
    assert "selected-channel error-max time" in status.value
    assert "case_worst" in str(export_state["filename_stem"])
    evaluation.close()


def test_horizon_error_uses_one_global_axis_for_sparse_frame_inventories() -> None:
    """Align nonidentical frame horizons to shared ticks instead of local positions."""
    summary = pd.DataFrame.from_records(
        (
            {
                "frame": "A",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": 1,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.1,
                "elapsed_physical_time_median": 1.0,
            },
            {
                "frame": "A",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": "full",
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.3,
                "elapsed_physical_time_median": 3.0,
            },
            {
                "frame": "B",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": 2,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.2,
                "elapsed_physical_time_median": 2.0,
            },
            {
                "frame": "B",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": "full",
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.4,
                "elapsed_physical_time_median": 4.0,
            },
        )
    )

    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary)
    horizon_axis = figure.axes[0]
    lines = {line.get_label(): line for line in horizon_axis.lines}

    assert [label.get_text() for label in horizon_axis.get_xticklabels()] == ["1", "2", "full"]
    np.testing.assert_array_equal(lines["Lentil · A — cumulative"].get_xdata(), np.asarray([0, 2]))
    np.testing.assert_array_equal(lines["Lentil · B — cumulative"].get_xdata(), np.asarray([1, 2]))
    plt.close(figure)


def test_evaluation_physical_time_axes_reuse_shared_units_and_export() -> None:
    """Format simulated time centrally while leaving rollout indices distinct."""
    record = _record("autonomous_full", "full")
    record.physical_times = np.asarray((0.0, 84.0, 168.0))
    central = analysis.evaluation.plots.transient.plot_central_error_vs_time(
        record,
        scaling_state=_scaling_state(),
    )
    summary = pd.DataFrame.from_records(
        (
            {
                "frame": "A",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": 1,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.1,
                "elapsed_physical_time_median": 24.0,
            },
            {
                "frame": "A",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": "full",
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.3,
                "elapsed_physical_time_median": 168.0,
            },
        )
    )
    horizon = analysis.evaluation.plots.transient.plot_horizon_error(summary)
    targets = analysis.evaluation.plots.transient.plot_target_time(
        (
            SimpleNamespace(
                target={
                    "reference_available": True,
                    "predicted_available": True,
                    "reference_time_to_target": 24.0,
                    "predicted_time_to_target": 48.0,
                }
            ),
            SimpleNamespace(
                target={
                    "reference_available": True,
                    "predicted_available": True,
                    "reference_time_to_target": 168.0,
                    "predicted_time_to_target": 144.0,
                }
            ),
        )
    )
    figures = (central, horizon, targets)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for figure in figures:
                figure.canvas.draw()
                buffer = BytesIO()
                figure.savefig(buffer, format="pdf", bbox_inches="tight")
                assert buffer.getbuffer().nbytes > 0
        assert central.axes[0].get_xlabel() == "Time [d]"
        assert horizon.axes[0].get_xlabel() == "Requested horizon [transitions]"
        assert horizon.axes[1].get_xlabel() == "Median elapsed physical time [d]"
        assert targets.axes[0].get_xlabel() == "Reference time to target [d]"
        assert targets.axes[0].get_ylabel() == "Predicted time to target [d]"
        assert not any("constrained_layout not applied" in str(item.message) for item in caught)
    finally:
        for figure in figures:
            plt.close(figure)


def test_transient_panel_plots_pipeline_and_local_report_remain_sequence_aware(tmp_path: Path) -> None:
    """Keep exact-time controls, all states, unavailable C, and local report outputs."""
    records = (
        _record("teacher_forced_one_step", 1),
        _record("autonomous_full", "full"),
        _record("rolling_origin", 2),
    )
    frame = _production_frame(records, artifact_root=str(tmp_path / "artifact"))
    frame.attrs["artifact_provenance"]["evaluation"].update(
        {
            "objective": {"fields": list(analysis.evaluation.transient_artifact.STATE_ORDER)},
            "process_diagnostic_policy": {"bulk_moisture": {"available": True}},
        }
    )
    evaluation = session.TransientEvaluationSession({"id": frame})
    export_state: dict[str, object] = {
        "fig": None,
        "plot_name": "maps",
        "title": "maps",
    }
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)  # noqa: SLF001
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        comparison=False,
        export_state=export_state,
        export_plot_name="maps",
        export_title="maps",
    )
    assert isinstance(viewer, widgets.VBox)
    case_row = viewer.children[0]
    assert isinstance(case_row, widgets.HBox)
    assert tuple(control.description for control in case_row.children) == (
        "Case:",
        "←",
        "→",
    )
    channel_row = viewer.children[2].children[1]
    assert hasattr(channel_row, "boxes")
    assert all(isinstance(checkbox, widgets.Checkbox) and checkbox.value is True for checkbox in channel_row.boxes.values())
    time_container = viewer.children[4]
    navigator_controls = time_container.children[0]
    assert tuple(control.description for control in navigator_controls.children) == (
        "Time [h]:",
        "≪",
        "←",
        "→",
        "≫",
    )
    assert isinstance(navigator_controls.children[0], widgets.FloatText)
    assert not any(isinstance(control, widgets.Play) for control in navigator_controls.children)
    physical_time = navigator_controls.children[0]
    physical_time.value = 0.0
    channel_row.boxes["T"].value = False
    assert physical_time.value == 0.0
    assert isinstance(export_state["fig"], Figure)
    for checkbox in channel_row.boxes.values():
        checkbox.value = False
    assert export_state["fig"] is None
    channel_row.boxes["T"].value = True

    trajectory_viewer = analysis.evaluation.panel._make_transient_scope_viewer(  # noqa: SLF001
        session=evaluation,
        fields=fields,
        plot_kind="trajectory",
        comparison=False,
    )
    trajectory_fields = cast(
        "analysis.ui.components.CheckboxGroup",
        next(widget for widget in _widget_tree(trajectory_viewer) if hasattr(widget, "boxes")),
    )
    assert tuple(trajectory_fields.boxes) == (
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    )
    trajectory_fields.boxes["w_gr"].value = True
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(trajectory_viewer))

    with pytest.raises(ValueError, match="unavailable"):
        analysis.evaluation.plots.transient.plot_state_maps(records[0], physical_time=0.5)
    figure = analysis.evaluation.plots.transient.plot_state_maps(records[0])
    figure.canvas.draw()
    expected_map_count = len(analysis.evaluation.transient_artifact.STATE_ORDER) * 3
    map_axes = tuple(figure.axes[:expected_map_count])
    colorbar_axes = tuple(figure.axes[expected_map_count:])
    assert tuple(axis.get_title() for axis in map_axes[:3]) == (
        "Reference",
        "Prediction",
        "Signed error",
    )
    row_labels = tuple(text.get_text() for axis in map_axes for text in axis.texts if text.get_gid() == "channel-row-label")
    assert len(row_labels) == len(analysis.evaluation.transient_artifact.STATE_ORDER)
    assert "°C" in row_labels[0]
    assert len(map_axes) == expected_map_count
    assert len(colorbar_axes) == expected_map_count
    np.testing.assert_allclose(
        figure.get_size_inches(),
        analysis.ui.plot_layout.MAP_LAYOUT.figure_size(
            rows=len(analysis.evaluation.transient_artifact.STATE_ORDER),
            columns=3,
        ),
    )
    assert set(figure.axes) == {*map_axes, *colorbar_axes}
    assert all(
        colorbar_axis.get_position().height == pytest.approx(map_axis.get_position().height)
        for map_axis, colorbar_axis in zip(map_axes, colorbar_axes, strict=True)
    )
    plt.close(figure)

    summary_frame = evaluation.dataset_dataframe()
    channel_figure = analysis.evaluation.plots.transient.plot_channel_error(summary_frame)
    assert channel_figure.axes[0].get_ylabel() == "Normalized RMSE [1]"
    assert len(channel_figure.axes[0].patches) == len(analysis.evaluation.transient_artifact.STATE_ORDER)
    plt.close(channel_figure)
    worst_figure = analysis.evaluation.plots.transient.plot_worst_case_errors(evaluation.case_dataframe(modes=("autonomous_full",)))
    assert "Worst" in worst_figure.axes[0].get_title()
    plt.close(worst_figure)

    paired = session.TransientEvaluationSession(
        {
            "Model ID": _production_frame(
                records,
                artifact_root=str(tmp_path / "id-artifact"),
            ),
            "Model OOD": _production_frame(
                records,
                artifact_root=str(tmp_path / "ood-artifact"),
            ),
        }
    )
    generalization = analysis.evaluation.plots.transient.plot_id_ood_generalization(paired.dataset_dataframe())
    assert generalization.axes[0].get_title() == "ID-to-OOD generalization"
    assert "Near-family OOD" in generalization.axes[0].get_ylabel()
    plt.close(generalization)
    paired.close()

    degradation = evaluation.pipeline_degradation()
    assert len(degradation) == 1
    assert degradation[0].c_available is False
    assert degradation[0].complete_pipeline_error is None
    accuracy = evaluation.case_dataframe(modes=("autonomous_full",))
    accuracy = accuracy.loc[(accuracy["scope"] == "cumulative") & (accuracy["requested_horizon"] == "full")]
    accuracy_figure = analysis.evaluation.plots.transient.plot_accuracy_vs_inference_time(
        accuracy,
        evaluation.timing_report("id"),
    )
    np.testing.assert_allclose(
        np.asarray(accuracy_figure.axes[0].collections[0].get_offsets(), dtype=np.float64),
        np.asarray([[0.03, 1.0]], dtype=np.float64),
    )
    plt.close(accuracy_figure)
    speedup_figure = analysis.evaluation.plots.transient.plot_accuracy_vs_speedup(
        accuracy,
        evaluation.timing_report("id"),
    )
    np.testing.assert_allclose(
        np.asarray(speedup_figure.axes[0].collections[0].get_offsets(), dtype=np.float64),
        np.asarray([[15.0, 1.0]], dtype=np.float64),
    )
    plt.close(speedup_figure)
    runtime_figure = analysis.evaluation.plots.transient.plot_timing_distributions(evaluation.timing_report("id"))
    assert all(axis.get_ylabel() == "Warmed runtime [s]" for axis in runtime_figure.axes)
    plt.close(runtime_figure)
    report = analysis.presentation.curated.render_curated_transient_analysis(
        session=evaluation,
        output_dir=tmp_path / "report",
    )
    assert set(report.media_files).union(report.tables) == analysis.presentation.curated.TRANSIENT_CURATED_ANALYSIS_KEYS
    assert all(report_path.is_file() for report_path in report.media_files.values())
    evaluation.close()
