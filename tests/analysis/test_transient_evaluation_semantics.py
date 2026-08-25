# ruff: noqa: S101, SLF001
"""Protect transient Evaluation metrics, timing, lineage, and pipeline semantics."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from src.analysis.evaluation import evaluation_transient_artifact as transient_artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_metrics as transient_metrics
from src.analysis.evaluation import evaluation_transient_rollout as rollout
from src.analysis.evaluation import evaluation_transient_timing as timing


def _states(*, offset: float = 0.0) -> np.ndarray:
    values = np.zeros((2, 4, 1, 1), dtype=np.float64)
    values[:, 0] = 300.0 + offset
    values[:, 1] = 0.5 + offset
    values[:, 2] = 2.0 + offset
    values[:, 3] = 4.0 + offset
    return values


def test_airflow_metric_explicit_id_and_historical_alias_preserve_values() -> None:
    """Preserve explicit and historical Airflow metric values without ambiguity."""
    explicit = transient_metrics.resolve_airflow_metric_value({"normalized_airflow_group_macro_rmse": 0.25})
    historical = transient_metrics.resolve_airflow_metric_value({"normalized_group_macro_rmse": 0.25})
    assert explicit == historical == pytest.approx(0.25)
    with pytest.raises(ValueError, match="disagree"):
        transient_metrics.resolve_airflow_metric_value(
            {
                "normalized_airflow_group_macro_rmse": 0.25,
                "normalized_group_macro_rmse": 0.5,
            }
        )


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
        "drying_no_end_to_end_seconds": (12.0,),
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
    assert timing.component_case_medians(report, "drying_no_end_to_end_seconds") == {"a": 12.0, "b": 12.0}


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
    )


def _condition() -> comparison.AirflowDryingCondition:
    return comparison.AirflowDryingCondition(
        "case", "checkpoint", "scale", "profile", "initial", "boundary", "startup", "material", "grid", "horizon", "float32", "post", ("u", "v", "p")
    )


def test_lineage_and_pipeline_require_matched_evidence_and_truthful_absence() -> None:
    """Require matched lineage and retain unavailable pipeline evidence without zeros."""
    lineage = comparison.validate_lineage_comparison(a0=_arm("A0", "stage_a"), a_plus=_arm("A+", "teacher"), b=_arm("B", "rollout"))
    assert (lineage.primary_comparison, lineage.separate_comparison) == ("B_vs_A_plus", "B_vs_A0")
    with pytest.raises(ValueError, match="safe-boundary"):
        comparison.validate_lineage_comparison(a0=_arm("A0", "stage_a"), a_plus=_arm("A+", "teacher"), b=_arm("B", "rollout", actual=9.0))
    with pytest.raises(ValueError, match="zero post-handoff"):
        comparison.validate_lineage_comparison(
            a0=_arm("A0", "stage_a", actual=1.0),
            a_plus=_arm("A+", "teacher"),
            b=_arm("B", "rollout"),
        )
    safe_overrun = comparison.validate_lineage_comparison(
        a0=_arm("A0", "stage_a"),
        a_plus=_arm("A+", "teacher", actual=10.75, planned=10.5),
        b=_arm("B", "rollout", actual=10.5),
    )
    assert safe_overrun.primary_comparison == "B_vs_A_plus"
    unavailable = comparison.validate_airflow_drying_comparison(
        reference_case_id="case", drying_on_comsol=_condition(), drying_on_airflow_no=None, unavailable_reason="missing compatible airflow checkpoint"
    )
    assert unavailable.c_available is False
    assert unavailable.c_unavailable_reason is not None
    available = comparison.validate_airflow_drying_comparison(
        reference_case_id="case", drying_on_comsol=_condition(), drying_on_airflow_no=_condition()
    )
    assert available.c_available is True
    altered = comparison.AirflowDryingCondition(
        "case", "other", "scale", "profile", "initial", "boundary", "startup", "material", "grid", "horizon", "float32", "post", ("u", "v", "p")
    )
    with pytest.raises(ValueError, match="drying_checkpoint_id"):
        comparison.validate_airflow_drying_comparison(reference_case_id="case", drying_on_comsol=_condition(), drying_on_airflow_no=altered)

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

    coerced = json.loads(json.dumps(serialized))
    coerced["cases"][0]["warmup_passes"] = "1"
    with pytest.raises(TypeError, match="scalar fields"):
        timing.admit_transient_timing_report(coerced)
