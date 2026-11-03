"""Protect transient Evaluation metrics, timing, lineage, and pipeline semantics.

Responsibilities:
  - Check masked macro metrics, target mass fractions, timing formulas, and matched lineage

Design principles:
  - Matched evidence and unavailable pipeline terms remain explicit

This module does NOT:
  - Publish sequence payloads or run solver comparisons
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.analysis.artifacts import analysis_artifact_transient as artifact_generator
from src.analysis.evaluation import evaluation_transient_artifact as transient_artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_metrics as transient_metrics
from src.analysis.evaluation import evaluation_transient_rollout as rollout
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.learning.inference import learning_inference_transient as inference


def _states(*, offset: float = 0.0) -> np.ndarray:
    values = np.zeros((2, 4, 1, 1), dtype=np.float64)
    values[:, 0] = 300.0 + offset
    values[:, 1] = 0.5 + offset
    values[:, 2] = 2.0 + offset
    values[:, 3] = 4.0 + offset
    return values


def test_metrics_use_central_macro_float64_masks_and_granular_water() -> None:
    """Use central Drying weights, sufficient statistics, masks, and physical water."""
    reference = _states()
    prediction = reference + np.asarray([1.0, 2.0, 3.0, 6.0], dtype=np.float64)[None, :, None, None]
    mask = np.ones_like(reference, dtype=bool)
    mask[1, :, :, :] = False
    accumulator = transient_metrics.TransientMetricAccumulator(scope="cumulative")
    accumulator.update(
        normalized_prediction=prediction,
        normalized_reference=reference,
        physical_prediction=prediction,
        physical_reference=reference,
        f_surf=np.asarray([[[0.25]]]),
        rho_bu_dry=np.asarray([[10.0]]),
        cell_weights=np.asarray([[1.0]]),
        valid_mask=mask,
    )
    summary = accumulator.finalize()
    assert summary.scope == "cumulative"
    assert summary.normalized_drying_group_macro_rmse == pytest.approx(2.5)
    assert summary.physical_w_gr_rmse == pytest.approx(5.25)
    assert summary.bulk_dry_basis_rmse == pytest.approx(0.525)
    assert summary.bulk_wet_basis_rmse == pytest.approx((8.75 / 18.75) - (3.5 / 13.5))
    assert (summary.bulk_moisture_valid_count, summary.bulk_moisture_unavailable_count) == (1, 1)
    assert summary.valid_counts == {"T": 1, "phi": 1, "w_surf": 1, "w_int": 1}

    partitioned = transient_metrics.TransientMetricAccumulator(scope="cumulative")
    for index in range(2):
        partitioned.update(
            normalized_prediction=prediction[index : index + 1],
            normalized_reference=reference[index : index + 1],
            physical_prediction=prediction[index : index + 1],
            physical_reference=reference[index : index + 1],
            f_surf=np.asarray([[[0.25]]]),
            rho_bu_dry=np.asarray([[10.0]]),
            cell_weights=np.asarray([[1.0]]),
            valid_mask=np.ones_like(reference[index : index + 1], dtype=bool),
        )
    partitioned_summary = partitioned.finalize()
    assert partitioned_summary.normalized_drying_group_macro_rmse == pytest.approx(summary.normalized_drying_group_macro_rmse)
    assert partitioned_summary.bulk_dry_basis_rmse == pytest.approx(summary.bulk_dry_basis_rmse)
    assert transient_metrics.trapezoidal_cell_weights(np.ones((2, 2), dtype=bool)).tolist() == [
        [0.25, 0.25],
        [0.25, 0.25],
    ]


def test_target_fraction_uses_canonical_masked_trapezoidal_mass_weighting() -> None:
    """Weight the canonical wet dry-solid fraction as a structured-grid integral."""
    states = np.zeros((1, len(transient_artifact.STATE_ORDER), 3, 3), dtype=np.float64)
    moisture_indices = tuple(transient_artifact.STATE_ORDER.index(field) for field in ("w_surf", "w_int"))
    for index in moisture_indices:
        states[0, index, 1, 1] = 1.0
    static = np.zeros((len(transient_artifact.STATIC_ORDER), 3, 3), dtype=np.float64)
    static[transient_artifact.STATIC_ORDER.index("rho_bu_dry")] = 1.0
    scalars = np.zeros(len(transient_artifact.SCALAR_ORDER), dtype=np.float64)
    scalars[transient_artifact.SCALAR_ORDER.index("f_surf")] = 0.5
    fraction = rollout._wet_fraction_series(
        states,
        static=static,
        scalars=scalars,
        mask=np.ones((3, 3), dtype=bool),
        target_wet_basis=0.25,
    )
    assert fraction.tolist() == pytest.approx([0.25])


def test_target_plausibility_and_stability_do_not_require_monotonic_moisture() -> None:
    """Keep censoring and stability diagnostics free of false moisture monotonicity."""
    targets = transient_metrics.derive_target_censoring_diagnostics(
        predicted_reached=np.asarray([True, False, None], dtype=object),
        reference_reached=np.asarray([True, True, None], dtype=object),
    )
    assert (targets.available_count, targets.agreement_count, targets.predicted_right_censored_count) == (2, 1, 1)
    states = np.asarray([[[[300.0]], [[0.5]], [[2.0]], [[4.0]]], [[[305.0]], [[1.2]], [[3.0]], [[3.0]]]], dtype=np.float64)
    plausibility = transient_metrics.derive_plausibility_diagnostics(states, temperature_range=(290.0, 310.0))
    assert plausibility.relative_humidity_bound_violations == 1
    stability = transient_metrics.derive_stability_diagnostics(np.concatenate((states, states[:1]), axis=0))
    assert stability.increment_count > 0
    nonfinite_states = np.concatenate((states, states[:1]), axis=0)
    nonfinite_states[1, 0, 0, 0] = np.nan
    nonfinite_stability = transient_metrics.derive_stability_diagnostics(nonfinite_states)
    assert nonfinite_stability.nonfinite_increment_count == len(nonfinite_states) - 1
    assert nonfinite_stability.abnormal_growth_count == 0


def _case(case_id: str, *, omit: str | None = None) -> timing.TransientTimingCase:
    components = {
        "comsol_transient_drying_seconds": (100.0,),
        "drying_no_rollout_model_seconds": (10.0,),
        "drying_no_prepared_rollout_wall_seconds": (12.0,),
        "comsol_scientific_solver_seconds": (200.0,),
        "airflow_no_model_seconds": (5.0,),
        "comsol_stationary_airflow_seconds": (20.0,),
        "comsol_process_seconds": (300.0,),
        "surrogate_pipeline_end_to_end_seconds": (30.0,),
        "generation_compute_end_to_end_seconds": (600.0,),
    }
    if omit is not None:
        components.pop(omit)
    return timing.TransientTimingCase(
        case_id,
        components,
        "cpu",
        "float32",
        "canonical_hdf5",
        1,
        cold_timings={"drying_no_rollout_model_seconds": 1_000.0},
        unavailable_reasons=({"comsol_process_seconds": "persisted process timing unavailable"} if omit == "comsol_process_seconds" else {}),
    )


def test_timing_reports_all_formulas_ratio_of_sums_and_unavailability() -> None:
    """Derive every speedup from warmed paired components and ratio of sums."""
    report = timing.build_transient_timing_report((_case("a"), _case("b", omit="comsol_process_seconds")))
    assert report.component_composed is True
    assert report.speedups["drying_only_solver_speedup"].ratio_of_sums == pytest.approx(10.0)
    assert report.speedups["full_pipeline_solver_speedup"].ratio_of_sums == pytest.approx(200.0 / 15.0)
    assert report.speedups["hybrid_component_speedup"].ratio_of_sums == pytest.approx(200.0 / 30.0)
    process = report.speedups["comsol_process_speedup"]
    assert process.available_count == 1
    assert process.cases[1].unavailable_reason == "persisted process timing unavailable"
    assert report.speedups["generation_compute_end_to_end_speedup"].ratio_of_sums == pytest.approx(20.0)
    assert timing.component_case_medians(report, "drying_no_prepared_rollout_wall_seconds") == {"a": 12.0, "b": 12.0}
    for summary in report.speedups.values():
        assert summary.reference_components
        assert summary.surrogate_components
        assert summary.reference_boundary
        assert summary.surrogate_boundary
        assert summary.comparison_scope
    assert process.reference_components == ("comsol_process_seconds",)
    assert process.surrogate_components == ("surrogate_pipeline_end_to_end_seconds",)


def test_timing_aggregate_pairs_warmed_medians_before_ratio_of_sums() -> None:
    """Unequal denominators distinguish pooled duration ratios from mean ratios."""
    second = _case("b")
    second = replace(
        second,
        repetitions={**second.repetitions, "drying_no_rollout_model_seconds": (25.0, 30.0, 80.0)},
    )
    report = timing.build_transient_timing_report((_case("a"), second))
    summary = report.speedups["drying_only_solver_speedup"]
    assert summary.ratio_of_sums == pytest.approx(200.0 / 40.0)
    assert summary.mean == pytest.approx((10.0 + 100.0 / 30.0) / 2.0)
    assert report.cases[1].repetitions["drying_no_rollout_model_seconds"] == (25.0, 30.0, 80.0)
    assert report.cases[1].cold_timings["drying_no_rollout_model_seconds"] == 1_000.0


def test_timing_adapter_preserves_measured_boundaries_and_unavailable_pipeline() -> None:
    """Admit independent clocks without promoting a prepared rollout to a pipeline."""
    benchmark = rollout.TransientRolloutBenchmark(
        case_id="case",
        warmup_passes=1,
        repetitions=2,
        model_clock="time.perf_counter",
        wall_clock="time.perf_counter",
        model_calls_per_repetition=2,
        cold_model_seconds=9.0,
        cold_prepared_rollout_wall_seconds=10.0,
        warmed_model_seconds=(1.0, 2.0),
        warmed_prepared_rollout_wall_seconds=(3.0, 4.0),
    )
    adapted = artifact_generator._timing_case(
        case_id="case",
        benchmark=benchmark,
        expected_model_calls=2,
        generation_timing={
            "stationary_airflow_solver_seconds": 15.0,
            "transient_drying_solver_seconds": 300.0,
            "scientific_solver_seconds": 315.0,
            "comsol_process_seconds": 338.74,
            "complete_execution_seconds": 370.12,
            "component_timing_availability": {"generation_compute_end_to_end_seconds": "queue_wait_not_measured"},
        },
        dataset_backend="canonical_hdf5",
        pt_identity=None,
        runtime_metadata={
            "resolved_device": "cpu",
            "processor": "test-cpu",
            "cuda_device_name": None,
            "python_version": "test",
            "pytorch_version": "test",
        },
    )
    assert adapted.repetitions["drying_no_prepared_rollout_wall_seconds"] == (3.0, 4.0)
    assert adapted.cold_timings["drying_no_prepared_rollout_wall_seconds"] == 10.0
    assert adapted.repetitions["drying_no_rollout_model_seconds"] == (1.0, 2.0)
    assert adapted.repetitions["comsol_transient_drying_seconds"] == (300.0,)
    assert adapted.repetitions["comsol_scientific_solver_seconds"] == (315.0,)
    assert adapted.repetitions["comsol_stationary_airflow_seconds"] == (15.0,)
    assert adapted.repetitions["comsol_process_seconds"] == (338.74,)
    report = timing.build_transient_timing_report((adapted,))
    assert report.speedups["drying_only_solver_speedup"].ratio_of_sums == pytest.approx(300.0 / 1.5)
    process = report.speedups["comsol_process_speedup"]
    assert process.ratio_of_sums is None
    assert process.cases[0].reference_seconds == 338.74
    assert process.cases[0].unavailable_reason == "complete_airflow_to_drying_pipeline_not_selected"
    generation = report.speedups["generation_compute_end_to_end_speedup"]
    assert generation.cases[0].reference_seconds is None
    assert generation.cases[0].unavailable_reason is not None
    assert "queue_wait_not_measured" in generation.cases[0].unavailable_reason


def test_prepared_rollout_wall_clock_excludes_case_and_request_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large preparation costs never enter the measured diagnostic rollout wall."""
    clock = [0.0]
    calls = [0]
    prepared = object()
    request = object()
    context = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    case = rollout.TransientEvaluationCase(
        case_id="case",
        dataset_role="id",
        physical_times=np.arange(3, dtype=np.float64),
        reference_states=np.zeros((3, 4, 1, 1)),
        static_conditioning=np.zeros((1, 1, 1)),
        boundary_conditioning=np.zeros((2, 1)),
        scalar_conditioning=np.zeros((2, 1)),
        spatial_mask=np.ones((1, 1)),
        metadata={},
    )

    def prepare(*_args: Any, **_kwargs: Any) -> object:
        clock[0] += 100.0
        return prepared

    def window(value: object, **_kwargs: Any) -> object:
        assert value is prepared
        clock[0] += 200.0
        return request

    def diagnostic(_context: object, value: object) -> SimpleNamespace:
        assert value is request
        calls[0] += 1
        clock[0] += float(calls[0] + 2)
        return SimpleNamespace(timing=SimpleNamespace(model_seconds=float(calls[0]), model_calls=2))

    monkeypatch.setattr(inference, "TransientInferenceContext", SimpleNamespace)
    monkeypatch.setattr(rollout, "_prepared_evaluation_case", prepare)
    monkeypatch.setattr(rollout, "_prepared_request", window)
    monkeypatch.setattr(rollout, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(inference, "rollout_prepared_transient_autonomous_diagnostic", diagnostic)
    # The class is monkeypatched to SimpleNamespace above, so the fake passes the runtime isinstance check.
    benchmark = rollout.benchmark_transient_full_rollout(cast("inference.TransientInferenceContext", context), case, warmup_passes=1, repetitions=2)
    assert calls[0] == 4
    assert benchmark.cold_model_seconds == 1.0
    assert benchmark.cold_prepared_rollout_wall_seconds == 3.0
    assert benchmark.warmed_model_seconds == (3.0, 4.0)
    assert benchmark.warmed_prepared_rollout_wall_seconds == (5.0, 6.0)


def _arm(
    name: str,
    strategy: str,
    actual: float | None = None,
    *,
    planned: float | None = None,
) -> comparison.TrainingArmEvidence:
    default_compute = 0.0 if name == "A0" else 10.0
    admitted_actual = default_compute if actual is None else actual
    admitted_planned = default_compute if planned is None else planned
    return comparison.TrainingArmEvidence(
        name,
        "parent",
        "arch",
        "data",
        "split",
        "profile",
        "scale",
        "boundary",
        strategy,
        admitted_planned,
        admitted_actual,
        admitted_actual - admitted_planned,
        parent_experiment_sha256=None if name == "A0" else "p" * 64,
        receipt_sha256=None if name == "A0" else "r" * 64,
        a0_handoff_sha256=None if name == "A0" else "h" * 64,
    )


def test_matched_a1_b_lineage_uses_common_origin_and_receipt_backed_time() -> None:
    """Admit committed B/A1 seconds while retaining a distinct A1 identity."""
    shared = {"parent_experiment_sha256": "p" * 64}
    a0 = _arm("A0", "stage_a")
    b = replace(
        _arm("B", "rollout", actual=12.5, planned=10.0),
        **shared,
        clock_kind="train_seconds",
        receipt_sha256="r" * 64,
        a0_handoff_sha256="h" * 64,
    )
    a1 = replace(
        _arm("A1", "teacher_forced_matched_train_seconds", actual=13.0, planned=12.5),
        **shared,
        clock_kind="train_seconds",
        receipt_sha256="r" * 64,
        a0_handoff_sha256="h" * 64,
    )
    admitted = comparison.validate_matched_a1_b_lineage_comparison(a0=a0, b=b, a1=a1)
    assert admitted.a1 == a1
    assert admitted.primary_comparison == "B_vs_A1"
    with pytest.raises(ValueError, match="branch parent identity"):
        comparison.validate_matched_a1_b_lineage_comparison(a0=a0, b=b, a1=replace(a1, parent_experiment_sha256="different"))
    with pytest.raises(ValueError, match="immutable B receipt"):
        comparison.validate_matched_a1_b_lineage_comparison(a0=a0, b=b, a1=replace(a1, receipt_sha256="x" * 64))
    with pytest.raises(ValueError, match="receipt-backed committed"):
        comparison.validate_matched_a1_b_lineage_comparison(a0=a0, b=b, a1=replace(a1, planned_post_handoff_compute=12.0, safe_boundary_overrun=1.0))
    with pytest.raises(ValueError, match="immutable lineage"):
        comparison.validate_matched_a1_b_lineage_comparison(a0=a0, b=b, a1=replace(a1, parent_id="other"))


def test_pipeline_degradation_retains_truthful_absence() -> None:
    """Retain unavailable pipeline evidence without zeros."""
    degradation = comparison.build_airflow_degradation_metrics(
        metric_id="normalized_drying_group_macro_rmse",
        drying_surrogate_error=2.0,
        complete_pipeline_error=3.0,
        airflow_substitution_discrepancy=1.5,
        upstream_airflow_error=0.25,
    )
    assert degradation.signed_airflow_degradation == pytest.approx(1.0)
    assert degradation.airflow_degradation_ratio == pytest.approx(1.5)
    missing = comparison.build_airflow_degradation_metrics(
        metric_id="normalized_drying_group_macro_rmse",
        drying_surrogate_error=2.0,
        complete_pipeline_error=None,
        airflow_substitution_discrepancy=None,
        upstream_airflow_error=None,
        unavailable_reason="compatible airflow checkpoint missing",
    )
    assert missing.c_available is False
    assert missing.complete_pipeline_error is None


def test_serialized_timing_admission_recomputes_aggregates_and_rejects_coercion() -> None:
    """Recompute timing aggregates from raw repetitions and reject typed-field drift."""
    report = timing.build_transient_timing_report((_case("a"),))
    serialized = json.loads(json.dumps(asdict(report)))
    admitted = timing.admit_transient_timing_report(serialized)
    assert admitted.speedups["drying_only_solver_speedup"].ratio_of_sums == pytest.approx(10.0)

    tampered = json.loads(json.dumps(serialized))
    tampered["speedups"]["drying_only_solver_speedup"]["ratio_of_sums"] = 11.0
    with pytest.raises(ValueError, match="aggregates"):
        timing.admit_transient_timing_report(tampered)

    mislabeled = json.loads(json.dumps(serialized))
    mislabeled["speedups"]["drying_only_solver_speedup"]["surrogate_boundary"] = "complete pipeline wall time"
    with pytest.raises(ValueError, match="aggregates"):
        timing.admit_transient_timing_report(mislabeled)

    coerced = json.loads(json.dumps(serialized))
    coerced["cases"][0]["warmup_passes"] = "1"
    with pytest.raises(TypeError, match="scalar fields"):
        timing.admit_transient_timing_report(coerced)
