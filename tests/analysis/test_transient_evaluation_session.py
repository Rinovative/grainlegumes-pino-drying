"""Protect transient Evaluation aggregation, presentation, and local reports.

Responsibilities:
  - Check pooled statistics, target records, lazy case access, panels, and local reports
  - Check that a session admits only frames recording one valid analysis producer
  - Check that the comparison ranking and the B/A pipeline error reduce cases exactly as checkpoint selection does
  - Check that water densities show as wet-basis percent with masked unphysical cells, errors in percentage points,
    bulk means on shared valid cells and unchanged normalized errors

Design principles:
  - Unequal support, unavailable metrics, and material summaries retain distinct meanings
  - The producer check reads recorded provenance only, never the running analysis code

This module does NOT:
  - Generate model predictions or refit saved scales
"""

from __future__ import annotations

import copy
import math
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
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from support.transient_evaluation import GRID_SHAPE, production_frame, scaling_state, sequence_frame, sequence_record

from src import analysis, domain, learning
from src.analysis.evaluation import evaluation_panel
from src.analysis.evaluation import evaluation_transient_session as session
from src.analysis.evaluation import evaluation_transient_validity as transient_validity
from src.analysis.evaluation import evaluation_workspace as workspace
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_RHO_CHANNEL = analysis.evaluation.transient_artifact.STATIC_ORDER.index("rho_bu_dry")


def _wet_basis_percent(water: np.ndarray, rho_bu_dry: np.ndarray) -> np.ndarray:
    """Return the domain wet-basis moisture of physical water in percent."""
    water = np.asarray(water, dtype=np.float64)
    return 100.0 * domain.moisture.wet_basis_moisture(water, np.broadcast_to(np.asarray(rho_bu_dry, dtype=np.float64), water.shape).copy())


def _children(widget: widgets.Widget) -> tuple[widgets.Widget, ...]:
    """Return the children of a box found while walking a generic widget tree."""
    return cast("widgets.Box", widget).children


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


def test_indexed_session_weighs_unequal_cases_equally_without_loading_payloads(
    tmp_path: Path,
) -> None:
    """Weigh the short and the long case equally in the objective and the grain-water error, and stay payload-free."""
    short = sequence_record(
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
        sequence_record("autonomous_full", "full", case_id="long", offset=3.0),
    )

    def statistics(record: SimpleNamespace) -> dict[str, dict[str, object]]:
        """Build test-owned persisted states through the production accumulator."""
        result: dict[str, dict[str, object]] = {}
        for scope in ("cumulative", "endpoint"):
            prediction = record.predicted_states[1:] if scope == "cumulative" else record.predicted_states[-1:]
            reference = record.reference_states[1:] if scope == "cumulative" else record.reference_states[-1:]
            accumulator = analysis.evaluation.transient_metrics.TransientMetricAccumulator(scope=scope)
            accumulator.update(
                physical_prediction=prediction,
                physical_reference=reference,
                f_surf=np.asarray(record.scalar_conditioning[2]),
                rho_bu_dry=record.static_conditioning[analysis.evaluation.transient_artifact.STATIC_ORDER.index("rho_bu_dry")],
                spacing=analysis.evaluation.transient_metrics.grid_spacing(record.static_conditioning),
            )
            state = accumulator.state_dict()
            value_count = int(prediction.size)
            result[scope] = {
                "classification": "REQUIRES_ALL_FINITE_VALUES",
                "available": True,
                "unavailable_reason": None,
                "required_value_count": value_count,
                "computed_value_count": value_count,
                "finite_value_count": value_count,
                "nonfinite_value_count": 0,
                "statistics": state,
            }
        diagnostic_prediction = record.predicted_states.copy()
        if record.case_id == "long":
            diagnostic_prediction[1, 0, 0, 0] = 3_000.0
        result["diagnostics"] = {
            "classification": "PHYSICAL_VALIDITY_METRIC",
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
            prediction_validity=transient_validity.build_prediction_validity(
                scaled_model_outputs=np.diff(record.predicted_states, axis=0),
                decoded_physical_increments=np.diff(
                    record.predicted_states,
                    axis=0,
                ),
                reconstructed_states=record.predicted_states[1:],
                prediction_available=np.ones(
                    len(record.physical_times) - 1,
                    dtype=bool,
                ),
                physical_times=record.physical_times,
                mode="autonomous_full",
                origin_index=0,
            ),
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
            "artifact_provenance": {"producer": {"analysis_sha256": "9" * 64}},
            "transient_sequence_index": index,
            "transient_unavailable_horizons": (),
            "transient_scaling_state": scaling_state(),
        }
    )
    evaluation = session.TransientEvaluationSession({"id": frame})

    cases = evaluation.case_dataframe(modes=("autonomous_full",))
    datasets_frame = evaluation.dataset_dataframe(modes=("autonomous_full",))
    cumulative = datasets_frame.loc[datasets_frame["scope"] == "cumulative"].iloc[0]
    # One predicted step of the short case and two of the long one: the cases weigh equally, whatever their length.
    expected_normalized = np.sqrt((0.5**2 + 1.5**2) / 2)
    expected_physical = np.sqrt((1.0**2 + 3.0**2) / 2)

    assert set(cases["case_id"]) == {"short", "long"}
    assert set(cases["scope"]) == {"cumulative", "endpoint"}
    assert cumulative["normalized_drying_group_macro_rmse"] == pytest.approx(expected_normalized)
    assert cumulative["physical_w_gr_rmse"] == pytest.approx(expected_physical)
    assert cumulative["normalized_drying_group_macro_rmse"] != pytest.approx(1.0)
    assert index.cache_size == 0
    assert cumulative["temperature_range_violations"] > 0
    compact = evaluation.full_autonomous_summaries()
    assert tuple(summary.case_id for summary in compact) == ("short", "long")
    assert index.cache_size == 0
    evaluation.close()


def test_grain_water_and_bulk_moisture_weigh_cases_equally_like_every_field_metric() -> None:
    """
    Grain-water and bulk-moisture errors weigh each case equally, whatever its rollout length; RMSE takes its root last.

    Lentil case a has two predicted states with offset 2, case b five with offset 4: grain-water MSE 4 and 16 give
    sqrt(10), not the length-pooled sqrt(88 / 7); bulk dry-basis errors 0.02 and 0.04 likewise give sqrt(0.001).
    """
    records = (
        sequence_record("autonomous_full", "full", case_id="a", offset=2.0),
        sequence_record("autonomous_full", "full", case_id="b", offset=4.0, length=6),
    )
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(records)})
    row = evaluation.dataset_dataframe(modes=("autonomous_full",)).query("scope == 'cumulative'").iloc[0]
    assert row["physical_w_gr_rmse"] == pytest.approx(math.sqrt(10.0))
    assert row["physical_w_gr_mae"] == pytest.approx(3.0)
    assert row["bulk_dry_basis_rmse"] == pytest.approx(math.sqrt(0.001))
    assert row["bulk_dry_basis_mae"] == pytest.approx(0.03)
    assert row["predicted_bulk_dry_basis_mean"] == pytest.approx(0.03)
    assert row["bulk_moisture_valid_count"] == 7


def test_session_uses_saved_scale_excludes_origin_and_retains_default_modes() -> None:
    """Use persisted scaling, omit origins, and retain all three required modes."""
    values = (sequence_record("teacher_forced_one_step", 1), sequence_record("autonomous_full", "full"), sequence_record("rolling_origin", 2))
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(values)})
    summaries = evaluation.summaries()
    teacher_cumulative = next(item for item in summaries if item.mode == "teacher_forced_one_step" and item.scope == "cumulative")
    teacher_metrics = teacher_cumulative.metrics
    assert teacher_metrics is not None
    assert teacher_cumulative.field_metrics is not None
    assert teacher_cumulative.field_metrics["normalized_drying_group_macro_rmse"] == pytest.approx(1.0)
    assert teacher_metrics.physical_w_gr_rmse == pytest.approx(2.0)
    assert teacher_metrics.bulk_dry_basis_rmse == pytest.approx(0.02)
    assert teacher_metrics.bulk_wet_basis_rmse == pytest.approx(2.0 / 102.0)
    assert teacher_metrics.bulk_moisture_valid_count == 1
    assert teacher_cumulative.contributing_record_count == 1
    assert set(evaluation.case_dataframe()["mode"]) == {"teacher_forced_one_step", "autonomous_full", "rolling_origin"}


@pytest.mark.parametrize(
    ("producer", "message"),
    [
        ({"analysis_sha256": "0" * 64}, "different analysis implementations"),
        (None, "'B' cannot be compared: Artifact producer must be exactly"),
        ({"analysis_sha256": "9" * 63}, "'B' cannot be compared: Artifact producer must be exactly"),
    ],
    ids=["different", "missing", "malformed"],
)
def test_session_requires_one_recorded_analysis_producer(producer: dict[str, str] | None, message: str) -> None:
    """Every session view compares its frames, so all of them must record the same valid analysis producer."""
    records = (sequence_record("autonomous_full", "full"),)
    peer = sequence_frame(records)
    session.TransientEvaluationSession({"A": sequence_frame(records), "B": peer})

    if producer is None:
        del peer.attrs["artifact_provenance"]["producer"]
    else:
        peer.attrs["artifact_provenance"]["producer"] = producer
    with pytest.raises(ValueError, match=message):
        session.TransientEvaluationSession({"A": sequence_frame(records), "B": peer})


def test_target_records_select_one_full_autonomous_outcome_per_case() -> None:
    """Avoid multiplying target outcomes across rollout modes, origins, and horizons."""
    records = (
        sequence_record("teacher_forced_one_step", 1, case_id="a"),
        sequence_record("autonomous_full", "full", case_id="a"),
        sequence_record("rolling_origin", 2, case_id="a"),
        sequence_record("autonomous_full", "full", case_id="b"),
        sequence_record("rolling_origin", 4, case_id="b"),
    )
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(records)})
    selected = evaluation.full_autonomous_records()
    assert tuple(record.case_id for record in selected) == ("a", "b")
    assert all(record.mode == "autonomous_full" and record.requested_horizon == "full" for record in selected)


def test_session_keeps_material_summaries_and_unavailability_separate() -> None:
    """Group sufficient statistics by identity-owned material without parsing case IDs."""
    records = (
        sequence_record(
            "rolling_origin",
            4,
            case_id="chickpea__case_0001",
            offset=2.0,
            material_family="chickpea",
        ),
        sequence_record(
            "rolling_origin",
            4,
            case_id="lentil__case_0001",
            offset=4.0,
            material_family="lentil",
        ),
    )
    frame = sequence_frame(records)
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
    records = (sequence_record("rolling_origin", 4, case_id="a", offset=2.0), sequence_record("rolling_origin", 4, case_id="b", offset=4.0))
    frame = sequence_frame(records)
    frame.attrs["transient_unavailable_horizons"] = ({"case_id": "a", "requested_horizon": 4, "reason": "short"},)
    evaluation = session.TransientEvaluationSession({"id": frame})
    summaries = evaluation.summaries(modes=("rolling_origin",))
    cumulative = next(item for item in summaries if item.scope == "cumulative")
    endpoint = next(item for item in summaries if item.scope == "endpoint")
    assert cumulative.field_metrics is not None
    assert endpoint.field_metrics is not None
    objective = "normalized_drying_group_macro_rmse"
    assert cumulative.field_metrics[objective] == pytest.approx(np.sqrt((1.0**2 + 2.0**2) / 2.0))
    assert endpoint.field_metrics[objective] == pytest.approx(cumulative.field_metrics[objective])
    assert (cumulative.contributing_case_count, cumulative.unavailable_case_count) == (2, 1)
    assert (cumulative.target.available_count, cumulative.target.reference_right_censored_count) == (0, 0)
    assert cumulative.target_time_error_count == 0
    assert cumulative.target_gap_count == 0
    assert cumulative.predicted_final_target_gap_mean == pytest.approx(-0.1)
    assert cumulative.reference_final_target_gap_mean is None
    assert cumulative.target_final_gap_error_mean is None
    assert cumulative.target_final_gap_error_mae is None


def test_session_reports_support_counts_instead_of_metrics_over_a_nonfinite_prediction() -> None:
    """A non-finite case makes the group's error metrics unavailable in rows; its finite peer is not reduced alone."""
    records = (sequence_record("rolling_origin", 2, case_id="a", offset=np.nan), sequence_record("rolling_origin", 2, case_id="b"))
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(records)})
    cumulative = next(item for item in evaluation.summaries(modes=("rolling_origin",)) if item.scope == "cumulative")
    assert cumulative.metrics is None
    per_case = 2 * 4 * GRID_SHAPE[0] * GRID_SHAPE[1]
    assert (cumulative.metric_support["finite_value_count"], cumulative.metric_support["required_value_count"]) == (per_case, 2 * per_case)

    row = evaluation.dataset_dataframe(modes=("rolling_origin",)).query("scope == 'cumulative'").iloc[0]
    assert (row["nonfinite_value_count"], pd.isna(row["normalized_drying_group_macro_rmse"])) == (per_case, True)


def test_relative_norms_below_the_reference_floor_stay_unavailable_and_counted_per_time_and_in_the_overview() -> None:
    """
    A case whose reference lies below the floor is unavailable, never zero, at every time and in the overview rows.

    Lentil case a keeps a reference of 300 in every field, so its relative L2 is 3 / 300; chickpea case b has a zero
    reference, so its material has no supported case and the two-material value is unavailable at each time.
    """
    supported = sequence_record("autonomous_full", "full", case_id="a", offset=303.0, material_family="lentil")
    supported.reference_states = supported.reference_states + 300.0
    unsupported = sequence_record("autonomous_full", "full", case_id="b", offset=1.0, material_family="chickpea")
    evaluation = session.TransientEvaluationSession({"id": sequence_frame((supported, unsupported))})

    curve = evaluation.relative_norm_curve("id").query("metric_id == 'physical_relative_l2_T'")
    assert list(curve["physical_time"]) == [1.0, 2.0]
    assert (list(curve["cases"]), list(curve["supported_cases"])) == ([2, 2], [1, 1])
    assert curve["value"].isna().all()
    rows = evaluation.dataset_dataframe(modes=("autonomous_full",)).query("scope == 'cumulative'").set_index("material_family")
    assert rows.loc["lentil", "physical_relative_l2_T"] == pytest.approx(0.01)
    assert pd.isna(rows.loc["chickpea", "physical_relative_l2_T"])
    counts = rows["physical_relative_l2_T_unavailable_cases"]
    assert (counts["lentil"], counts["chickpea"]) == (0, 1)


def test_relative_norm_curve_keeps_the_finite_states_of_a_rollout_that_diverges_later() -> None:
    """
    A non-finite predicted state makes only its own time unavailable: earlier states count and the case stays present.

    Lentil cases a and b keep a reference of 300 with an offset of 3, so each relative L2 is 0.01; b's prediction is
    non-finite at time 2 only, so time 1 has two supported cases and time 2 one, both with the value 0.01.
    """
    finite = sequence_record("autonomous_full", "full", case_id="a", offset=303.0)
    diverging = sequence_record("autonomous_full", "full", case_id="b", offset=303.0)
    for record in (finite, diverging):
        record.reference_states = record.reference_states + 300.0
    diverging.predicted_states[2] = np.nan
    evaluation = session.TransientEvaluationSession({"id": sequence_frame((finite, diverging))})

    curve = evaluation.relative_norm_curve("id").query("metric_id == 'physical_relative_l2_T'")
    assert (list(curve["cases"]), list(curve["supported_cases"])) == ([2, 2], [2, 1])
    np.testing.assert_allclose(curve["value"], [0.01, 0.01])


def test_relative_norm_curve_of_an_indexed_artifact_reads_the_full_records_not_the_array_free_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded artifact serves compact summaries; the time view opens each case's record once and reduces its states."""
    loads: list[str] = []
    evaluation = _indexed_session(tmp_path, monkeypatch, loads, compact_summaries=True)

    curve = evaluation.relative_norm_curve("id")

    assert sorted(loads) == ["record-0", "record-1"]
    assert list(curve.query("metric_id == 'physical_relative_l2_T'")["physical_time"]) == [1.0, 2.0]
    assert set(curve["cases"]) == {2}
    evaluation.close()


def test_a_short_case_without_a_supported_origin_counts_as_missing_at_the_horizon_it_cannot_reach() -> None:
    """Case membership of a material comes from the artifact, not from the records that exist at that horizon."""
    reachable = sequence_record("rolling_origin", 4, case_id="long", offset=2.0)
    short = sequence_record("autonomous_full", "full", case_id="short", offset=2.0)
    frame = sequence_frame((reachable, short))
    frame.attrs["transient_unavailable_horizons"] = ({"case_id": "short", "requested_horizon": 4, "reason": "short"},)
    evaluation = session.TransientEvaluationSession({"id": frame})

    cumulative = next(item for item in evaluation.summaries(modes=("rolling_origin",)) if item.scope == "cumulative")

    assert (cumulative.contributing_case_count, cumulative.unavailable_case_count) == (1, 1)


def test_a_material_without_any_record_at_a_reached_horizon_shows_that_row_with_every_case_missing() -> None:
    """Every material gets a row at each rolling horizon another material reaches; its metrics are unavailable, never zero."""
    reached = sequence_record("rolling_origin", 4, case_id="lentil_long", offset=2.0, material_family="lentil")
    short = sequence_record("rolling_origin", 2, case_id="chickpea_short", offset=2.0, material_family="chickpea")
    frame = sequence_frame((reached, short))
    frame.attrs["transient_unavailable_horizons"] = ({"case_id": "chickpea_short", "requested_horizon": 4, "reason": "short"},)
    evaluation = session.TransientEvaluationSession({"id": frame})

    rows = evaluation.dataset_dataframe(modes=("rolling_origin",)).query("requested_horizon == 4 and scope == 'cumulative'")
    row = rows.set_index("material_family").loc["chickpea"]

    assert set(rows["material_family"]) == {"lentil", "chickpea"}
    assert (row["contributing_case_count"], row["unavailable_case_count"], row["contributing_record_count"]) == (0, 1, 0)
    values = row.to_dict()
    assert math.isnan(values["normalized_drying_group_macro_rmse"])
    assert math.isnan(values["physical_w_gr_rmse"])
    short_rows = evaluation.dataset_dataframe(modes=("rolling_origin",)).query("requested_horizon == 2 and scope == 'cumulative'")
    assert short_rows.set_index("material_family")["contributing_case_count"].to_dict() == {"chickpea": 1, "lentil": 0}
    assert set(evaluation.case_dataframe(modes=("rolling_origin",)).query("requested_horizon == 4")["material_family"]) == {"lentil"}


def test_a_requested_horizon_no_case_reaches_still_yields_zero_case_rows_with_the_full_missing_count() -> None:
    """The horizons of the unavailable entries count as rolling horizons, so their missing cases reach tables and exports."""
    records = (
        sequence_record("rolling_origin", 2, case_id="lentil_a", material_family="lentil"),
        sequence_record("rolling_origin", 2, case_id="chickpea_b", material_family="chickpea"),
    )
    frame = sequence_frame(records)
    frame.attrs["transient_unavailable_horizons"] = tuple(
        {"case_id": case_id, "requested_horizon": 4, "reason": "short"} for case_id in ("lentil_a", "chickpea_b")
    )
    evaluation = session.TransientEvaluationSession({"id": frame})

    rows = evaluation.dataset_dataframe(modes=("rolling_origin",)).query("requested_horizon == 4 and scope == 'cumulative'")
    cells = analysis.evaluation.panel._transient_summary_table(evaluation).query("requested_horizon == 4 and scope == 'cumulative'")

    assert rows.set_index("material_family")[["contributing_case_count", "unavailable_case_count"]].to_dict("index") == {
        "lentil": {"contributing_case_count": 0, "unavailable_case_count": 1},
        "chickpea": {"contributing_case_count": 0, "unavailable_case_count": 1},
    }
    assert set(cells["normalized_drying_group_macro_rmse"]) == {"unavailable (no cases at this horizon; 1 missing)"}


def test_zero_case_rows_are_left_out_of_the_horizon_plot_and_named_in_the_summary_table() -> None:
    """A material without records at a horizon draws no point there and its table cell names the missing cases."""
    reached = sequence_record("rolling_origin", 4, case_id="lentil_long", offset=2.0, material_family="lentil")
    short = sequence_record("rolling_origin", 2, case_id="chickpea_short", offset=2.0, material_family="chickpea")
    frame = sequence_frame((reached, short))
    frame.attrs["transient_unavailable_horizons"] = ({"case_id": "chickpea_short", "requested_horizon": 4, "reason": "short"},)
    evaluation = session.TransientEvaluationSession({"id": frame})

    figure = analysis.evaluation.plots.transient.plot_horizon_error(evaluation.dataset_dataframe(modes=("rolling_origin",)))
    table = analysis.evaluation.panel._transient_summary_table(evaluation)
    cells = table.query("material_family == 'chickpea' and requested_horizon == 4 and scope == 'cumulative'")
    endpoint = analysis.evaluation.plots.transient.plot_endpoint_vs_cumulative(evaluation.dataset_dataframe(modes=("rolling_origin",)))
    plt.close(figure)
    plt.close(endpoint)

    assert cells["normalized_drying_group_macro_rmse"].tolist() == ["unavailable (no cases at this horizon; 1 missing)"]
    lentil = table.query("material_family == 'lentil' and requested_horizon == 2 and scope == 'cumulative'")
    assert lentil["normalized_drying_group_macro_rmse"].tolist() == ["unavailable (no cases at this horizon)"]


def test_exact_stop_gap_is_not_paired_with_a_regular_prediction_endpoint() -> None:
    """Keep final-gap errors unavailable when canonical and predicted endpoints differ."""
    record = sequence_record("autonomous_full", "full")
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
    evaluation = session.TransientEvaluationSession({"id": sequence_frame((record,))})
    cumulative = next(item for item in evaluation.summaries() if item.mode == "autonomous_full" and item.scope == "cumulative")
    assert cumulative.target_time_error_count == 1
    assert cumulative.target_gap_count == 0
    assert cumulative.predicted_final_target_gap_mean == pytest.approx(-0.1)
    assert cumulative.reference_final_target_gap_mean == pytest.approx(-0.2)
    assert cumulative.target_final_gap_error_mean is None


def test_transient_registry_invokes_every_visible_entry(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invoke every registered transient entry through explicit export context, none of them failing."""
    records = (
        sequence_record("teacher_forced_one_step", 1),
        sequence_record("autonomous_full", "full"),
        sequence_record("rolling_origin", 2),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )

    outer = analysis.evaluation.panel.build_transient_panel(
        session=evaluation,
    )

    lazy = analysis.evaluation.panel._LAZY_PANELS[outer]
    captured: list[tuple[str, Callable[..., object], str]] = [entry for section in lazy.sections for entry in section.entries]
    assert captured
    assert all(label == plot_name for label, _factory, plot_name in captured)
    assert len({plot_name for _label, _factory, plot_name in captured}) == len(captured)
    registered_names = {plot_name for _label, _factory, plot_name in captured}
    assert {
        "2-3. GT vs Prediction (mean)",
        "2-4. Mean error maps",
        "5-1. Sample GT vs Prediction",
        "4-1. Reference vs prediction trajectories",
    }.issubset(registered_names)
    assert not any(fragment in plot_name for plot_name in registered_names for fragment in ("pressure", "velocity", "divergence", "permeability"))
    results: dict[str, object] = {}
    for _label, factory, plot_name in captured:
        parameters = signature(factory).parameters
        assert all(parameter.kind is not Parameter.VAR_KEYWORD for parameter in parameters.values())
        result = analysis.ui.notebook._invoke_dropdown_entry(
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

    sample = results["5-1. Sample GT vs Prediction"]
    error_time = results["4-2. Error over physical time"]
    assert isinstance(sample, analysis.ui.panel.ViewPanel)
    assert isinstance(error_time, analysis.ui.panel.ViewPanel)
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(sample.widget))
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(error_time.widget))
    assert not [record.getMessage() for record in caplog.records if "View failed" in record.getMessage()]
    evaluation.close()
    plt.close("all")


def test_transient_controls_expose_admitted_state_and_derived_channels(
    tmp_path: Path,
) -> None:
    """Expose source-backed state and derived moisture without a widget-layout snapshot."""
    frame = production_frame(
        (sequence_record("autonomous_full", "full"),),
        artifact_root=str(tmp_path / "artifact"),
    )
    frame.attrs["artifact_provenance"]["evaluation"].update(
        {
            "objective": {"fields": list(analysis.evaluation.transient_artifact.STATE_ORDER)},
            "process_diagnostic_policy": {"bulk_moisture": {"available": True}},
        }
    )
    evaluation = session.TransientEvaluationSession({"id": frame})
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    assert tuple(field.key for field in fields) == (*analysis.evaluation.transient_artifact.STATE_ORDER, "w_gr")
    evaluation.close()


def _tab_layout(panel: widgets.Widget) -> tuple[tuple[str, ...], list[list[str]]]:
    """Return the displayed tab titles and each tab's view titles of one lazy panel."""
    lazy = analysis.evaluation.panel._LAZY_PANELS[panel]
    return tuple(cast("Any", lazy.tabs).titles), [list(section.titles) for section in lazy.sections]


def _transient_evaluation(tmp_path: Path, *records: SimpleNamespace) -> session.TransientEvaluationSession:
    """Open one session over a single production-shaped frame of the given records."""
    return session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path / "artifact"))})


def test_transient_tabs_follow_the_decided_order_and_numbers_follow_the_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order the tabs per mode, keep the configuration table in the right tab, and number tabs and views by position."""
    monkeypatch.setattr(session.TransientEvaluationSession, "require_common_comparison_grids", lambda _self: None)
    evaluation = _transient_evaluation(
        tmp_path,
        sequence_record("teacher_forced_one_step", 1),
        sequence_record("autonomous_full", "full"),
        sequence_record("rolling_origin", 2),
    )
    titles = analysis.evaluation.panel.EVALUATION_SECTION_TITLES
    single = ("overview", "global_error", "error_decomposition", "temporal_rollout", "sample_viewer", "outlier_analysis", "runtime", "matched_stages")
    comparison = (
        "overview",
        "global_error",
        "architecture",
        "error_decomposition",
        "temporal_rollout",
        "sample_viewer",
        "outlier_analysis",
        "runtime",
        "matched_stages",
    )
    layouts = {}
    for keys, is_comparison in ((single, False), (comparison, True)):
        tabs, views = _tab_layout(analysis.evaluation.panel.build_transient_panel(session=evaluation, comparison=is_comparison))
        assert tabs == tuple(f"{number}. {titles[key]}" for number, key in enumerate(keys, start=1))
        for number, labels in enumerate(views, start=1):
            assert all(label.startswith(f"{number}-{position}. ") for position, label in enumerate(labels, start=1))
        layouts[is_comparison] = views
    assert any("Model-family configuration" in label for label in layouts[False][0])
    assert not any("Model-family configuration" in label for label in layouts[True][0])
    assert len(layouts[True][2]) == 1
    assert "Model-family configuration" in layouts[True][2][0]
    evaluation.close()


def test_transient_numbers_leave_no_gap_for_filtered_or_selected_sections(tmp_path: Path) -> None:
    """Number the tabs that are displayed: no temporal evidence closes its gap and a sample-only panel starts at one."""
    evaluation = _transient_evaluation(tmp_path, sequence_record("teacher_forced_one_step", 1))
    titles = analysis.evaluation.panel.EVALUATION_SECTION_TITLES
    tabs, _views = _tab_layout(analysis.evaluation.panel.build_transient_panel(session=evaluation))
    assert not any(titles["temporal_rollout"] in title for title in tabs)
    assert [title.partition(".")[0] for title in tabs] == [str(number) for number in range(1, len(tabs) + 1)]
    sample_tabs, sample_views = _tab_layout(analysis.evaluation.panel.build_transient_panel(session=evaluation, sections=("sample_viewer",)))
    assert sample_tabs == (f"1. {titles['sample_viewer']}",)
    assert [label.partition(".")[0] for label in sample_views[0]] == ["1-1"]
    evaluation.close()


def test_closing_a_panel_closes_its_rendered_views_and_silences_compute_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered views sit in output areas: closing the panel must still close them and stop their compute callbacks."""
    evaluation = _transient_evaluation(tmp_path, sequence_record("autonomous_full", "full"))
    rendered: list[object] = []
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **kwargs: rendered.append(kwargs) or True)
    built: list[analysis.ui.panel.ViewPanel] = []
    original = analysis.evaluation.panel._make_transient_aggregate_viewer

    def capture(**kwargs: Any) -> analysis.ui.panel.ViewPanel:
        built.append(original(**kwargs))
        return built[-1]

    monkeypatch.setattr(analysis.evaluation.panel, "_make_transient_aggregate_viewer", capture)
    preference = compute.AnalysisComputePreference("cpu")
    panel = analysis.evaluation.panel.build_transient_panel(session=evaluation, evaluation_compute_device=preference)
    lazy = analysis.evaluation.panel._LAZY_PANELS[panel]
    lazy.open()
    lazy.tabs.selected_index = 1
    section = lazy.sections[1]
    section.selector.index = next(index for index, (label, _value) in enumerate(section.selector.options) if "Mean error maps" in label)
    assert len(built) == 1
    assert built[0].widget.comm is not None
    before = len(rendered)
    preference.set_requested("auto")
    assert len(rendered) > before
    # A view that a re-entrant selection displaced is no longer its section's current view, but it was rendered and is active.
    displaced = analysis.ui.panel.ViewPanel(())
    analysis.evaluation.panel._RENDERED_VIEWS[panel].append(displaced)

    analysis.evaluation.panel.close_panel(panel)
    after = len(rendered)
    preference.set_requested("cpu")
    assert len(rendered) == after
    assert not built[0].active
    assert not displaced.active
    assert built[0].widget.comm is None
    assert lazy.tabs.comm is None
    assert section.selector.comm is None
    evaluation.close()


def test_replacing_a_workspace_workflow_closes_the_previous_panel_before_its_session() -> None:
    """Close the replaced panel's widgets first, then its session, and keep the new panel installed."""
    events: list[str] = []

    class Prepared:
        def __init__(self, name: str) -> None:
            self.name = name
            self.button = widgets.Button()
            self.panel = widgets.VBox((self.button,))

        def close(self) -> None:
            events.append(f"{self.name}: session closed, panel closed {self.panel.comm is None}, child closed {self.button.comm is None}")

    previous, replacement = Prepared("old"), Prepared("new")
    said: list[str] = []
    controller = SimpleNamespace(
        view=analysis.ui.panel.ViewPanel((), slot=analysis.ui.panel.PlotSlot()),
        _workflow=previous,
        _say=lambda summary, _level, _details: said.append(summary),
    )
    publish = cast("Any", workspace._EvaluationWorkspaceController)._publish
    publish(controller, lambda: workspace._Outcome(cast("Any", replacement), "new loaded", "normal", ""))

    assert events == ["old: session closed, panel closed True, child closed True"]
    assert said == ["new loaded"]
    assert controller._workflow is replacement
    assert replacement.panel.comm is not None

    # A panel its publication could not make is closed at once, and neither its outcome nor the panel on show changes.
    stale = Prepared("stale")
    controller.view.deactivate()
    publish(controller, lambda: workspace._Outcome(cast("Any", stale), "stale loaded", "normal", ""))
    assert events[-1] == "stale: session closed, panel closed True, child closed True"
    assert said == ["new loaded"]
    assert controller._workflow is replacement


def test_a_publication_that_starts_another_leaves_the_newer_one_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load started while the older panel is displayed is not replaced by the older outcome, which closes unshown."""
    events: list[str] = []
    said: list[str] = []

    class Prepared:
        def __init__(self, name: str) -> None:
            self.name = name
            self.panel = widgets.Label(name)

        def close(self) -> None:
            events.append(f"{self.name} closed")

    first, second = Prepared("first"), Prepared("second")
    controller = SimpleNamespace(
        view=analysis.ui.panel.ViewPanel((), slot=analysis.ui.panel.PlotSlot()),
        _workflow=None,
        _say=lambda summary, _level, _details: said.append(summary),
    )
    publish = cast("Any", workspace._EvaluationWorkspaceController)._publish

    element = analysis.ui.panel.item_widget

    def display(item: object) -> widgets.Widget:
        if item is first.panel:  # the user chooses again while the first panel is being displayed
            publish(controller, lambda: workspace._Outcome(cast("Any", second), "second loaded", "normal", ""))
        return element(item)

    monkeypatch.setattr(analysis.ui.panel, "item_widget", display)
    publish(controller, lambda: workspace._Outcome(cast("Any", first), "first loaded", "normal", ""))

    assert controller._workflow is second
    assert said == ["second loaded"]
    assert events == ["first closed"]


def _shown_text(view: analysis.ui.panel.ViewPanel) -> str:
    """Return the HTML text a view's plot slot shows, nested views included."""
    assert view.slot is not None
    return "\n".join(item.value for item in _widget_tree(view.slot.widget) if isinstance(item, widgets.HTML))


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
        sequence_record("autonomous_full", "full", case_id="low", offset=1.0),
        sequence_record("autonomous_full", "full", case_id="high", offset=3.0),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    records[1].static_conditioning[_RHO_CHANNEL] = 300.0  # each case keeps its own dry bulk density
    faceted = analysis.evaluation.plots.transient.plot_aggregate_error_maps(
        {
            "Chickpea · ID": (records[0],),
            "Kidney bean · OOD": (records[1],),
        },
        state_fields=("w_surf",),
        physical_time=1.0,
        statistic="mean",
    )
    assert tuple(axis.get_title() for axis in faceted.axes[:2]) == (
        "Chickpea · ID",
        "Kidney bean · OOD",
    )
    # Each case converts its water with its own dry bulk density before pooling: wet-basis differences in pp.
    for axis, record in zip(faceted.axes[:2], records, strict=True):
        water = record.predicted_states[1, 2]
        expected = _wet_basis_percent(water, record.static_conditioning[_RHO_CHANNEL]) - _wet_basis_percent(
            record.reference_states[1, 2], record.static_conditioning[_RHO_CHANNEL]
        )
        np.testing.assert_allclose(np.asarray(axis.collections[0].get_array()), expected)
    plt.close(faceted)
    # Pooled in one panel, the mean is over the per-case pp differences, not a difference of pooled water.
    pooled = analysis.evaluation.plots.transient.plot_aggregate_error_maps(
        {"Pooled": records}, state_fields=("w_surf",), physical_time=1.0, statistic="mean"
    )
    per_case = [
        _wet_basis_percent(record.predicted_states[1, 2], record.static_conditioning[_RHO_CHANNEL])
        - _wet_basis_percent(record.reference_states[1, 2], record.static_conditioning[_RHO_CHANNEL])
        for record in records
    ]
    np.testing.assert_allclose(np.asarray(pooled.axes[0].collections[0].get_array()), np.mean(per_case, axis=0))
    plt.close(pooled)

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
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    viewer = analysis.evaluation.panel._make_transient_aggregate_viewer(
        session=evaluation,
        fields=fields,
        plot_kind="mean_error",
        comparison=False,
        export_state=export_state,
        export_plot_name="spatial_error",
    )
    time_control = next(
        widget for widget in _widget_tree(viewer.widget) if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]"
    )
    temperature = next(widget for widget in _widget_tree(viewer.widget) if isinstance(widget, widgets.Checkbox) and widget.tooltip == "T")
    assert calls == [("id", "low"), ("id", "high")]
    time_control.value = 1.0
    temperature.value = False
    assert calls == [("id", "low"), ("id", "high")]
    assert "channel_phi-w_surf-w_int" in str(export_state["filename_stem"])
    assert "t_1h" in str(export_state["filename_stem"])
    evaluation.close()


def _indexed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loads: list[str],
    *,
    compact_summaries: bool = False,
) -> session.TransientEvaluationSession:
    """Build a session whose index counts record loads instead of opening bundles; compact summaries carry no arrays, as loaded ones."""
    records = (
        sequence_record("autonomous_full", "full", case_id="low", offset=1.0),
        sequence_record("autonomous_full", "full", case_id="high", offset=-3.0),
    )
    for number, record in enumerate(records):
        record.record_id = f"record-{number}"
        record.chain_id = f"chain-{number}"
        record.available_horizon = len(record.physical_times) - 1
    frame = production_frame(records, artifact_root=str(tmp_path / "artifact"))
    frame.attrs["transient_sequence_index"] = analysis.evaluation.transient_artifact.TransientSequenceArtifactIndex(
        root=tmp_path / "artifact",
        dataset_name="transient_dataset",
        dataset_role="id",
        summaries=cast(
            "Any",
            tuple(
                SimpleNamespace(**{key: value for key, value in vars(record).items() if key not in ("predicted_states", "reference_states")})
                for record in records
            )
            if compact_summaries
            else records,
        ),
        unavailable_horizons=(),
        frame=frame,
        provenance=frame.attrs["artifact_provenance"],
        identity_sha256="f" * 64,
        _rows={record.record_id: {} for record in records},
    )
    if compact_summaries:
        del frame.attrs["transient_sequence_records"]
    by_id = {record.record_id: record for record in records}

    def load(_index: object, record_id: str) -> SimpleNamespace:
        loads.append(record_id)
        return by_id[record_id]

    monkeypatch.setattr(analysis.evaluation.transient_artifact.TransientSequenceArtifactIndex, "record", load)
    return session.TransientEvaluationSession({"id": frame})


def test_views_of_one_session_share_one_record_load_and_a_new_session_loads_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each case record is opened once per session, however many views ask for it."""
    loads: list[str] = []
    first = _indexed_session(tmp_path, monkeypatch, loads)
    fields = analysis.evaluation.panel._transient_channel_fields(first)
    for plot_kind in ("mean_error", "std_error"):
        analysis.evaluation.panel._make_transient_aggregate_viewer(
            session=first,
            fields=fields,
            plot_kind=plot_kind,
            comparison=False,
        )
    analysis.evaluation.panel._make_transient_scope_viewer(
        session=first,
        fields=fields,
        plot_kind="trajectory",
        comparison=False,
        selection_state=None,
    )
    assert sorted(loads) == ["record-0", "record-1"]
    first.close()

    second = _indexed_session(tmp_path, monkeypatch, loads)
    second.record_for_coordinates("id", "low", mode="autonomous_full", origin_index=0, requested_horizon="full")
    assert loads.count("record-0") == 2
    second.full_autonomous_records("id")
    assert loads.count("record-0") == 2
    frame = second.canonical_frames["id"]
    frame.attrs["transient_sequence_index"] = copy.copy(frame.attrs["transient_sequence_index"])
    second.record_for_coordinates("id", "low", mode="autonomous_full", origin_index=0, requested_horizon="full")
    assert loads.count("record-0") == 3
    second.close()


def test_transient_temporal_union_preserves_short_tails_counts_and_nonfinite_unavailability() -> None:
    """Exact-time cohorts retain late support and never average away invalid predictions."""
    long = sequence_record("autonomous_full", "full", case_id="long", offset=1.0)
    short = sequence_record("autonomous_full", "full", case_id="short", offset=3.0)
    short.physical_times = short.physical_times[:2]
    short.reference_states = short.reference_states[:2]
    short.predicted_states = short.predicted_states[:2]
    assert analysis.evaluation.panel._shared_transient_physical_times((long, short)) == (0.0, 1.0, 2.0)
    rows = (
        (np.array((1.0, 2.0)), np.array((1.0, 2.0))),
        (np.array((1.0,)), np.array((np.nan,))),
    )
    exact = analysis.evaluation.plots.transient._aggregate_exact_time_values(rows)
    np.testing.assert_array_equal(exact.times, [1.0, 2.0])
    np.testing.assert_array_equal(exact.count, [2, 1])
    np.testing.assert_array_equal(exact.invalid_count, [1, 0])
    assert np.isnan(exact.mean[0])
    assert exact.mean[1] == 2.0
    assert np.isnan(exact.lower).all()

    trajectory = analysis.evaluation.plots.transient.plot_state_trajectory_summary({"ID": (long, short)}, state_fields=("T",))
    trajectory_times = np.asarray(trajectory.axes[0].lines[0].get_xdata())
    assert trajectory_times.size == 3
    np.testing.assert_array_equal(trajectory.axes[1].lines[0].get_ydata(), [2, 2, 1])
    plt.close(trajectory)
    error = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": (long, short)}, scaling_states={"ID": scaling_state()}, state_fields=("T",)
    )
    error_times = np.asarray(error.axes[0].lines[0].get_xdata())
    assert error_times.size == 2
    np.testing.assert_array_equal(error.axes[1].lines[0].get_ydata(), [2, 1])
    plt.close(error)

    empty_context = analysis.evaluation.plots.transient.plot_aggregate_error_maps(
        {"ID": (long,), "OOD": ()}, state_fields=("T",), physical_time=2.0, statistic="mean"
    )
    assert any("No case at this exact time" in text.get_text() for text in empty_context.axes[1].texts)
    plt.close(empty_context)


def test_error_over_time_line_is_the_root_of_the_material_case_mean_squared_error() -> None:
    """
    The central error line reduces per-case mean squared errors equally over cases and materials, root after the mean.

    With a state scale of 2, lentil cases a and b have normalized MSE 0 and 4 and chickpea case c has 1, so the line
    is sqrt((2 + 1) / 2) at both times, not the mean of the per-case RMSEs (1).
    """
    records = (
        sequence_record("autonomous_full", "full", case_id="a", offset=0.0, material_family="lentil"),
        sequence_record("autonomous_full", "full", case_id="b", offset=4.0, material_family="lentil"),
        sequence_record("autonomous_full", "full", case_id="c", offset=2.0, material_family="chickpea"),
    )
    figure = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": records}, scaling_states={"ID": scaling_state()}, state_fields=("T",)
    )
    np.testing.assert_allclose(np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=np.float64), [math.sqrt(1.5)] * 2)
    plt.close(figure)


def test_partitioned_case_selector_routes_materials_to_owning_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose one material-labelled union while retaining exact artifact ownership."""

    def owned_record(case_id: str, material: str, compact_case: str) -> SimpleNamespace:
        record = sequence_record(
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
        if material == "kidney_bean":
            record.identity["evaluation_assignment"] = {
                "dataset_id": f"dataset_{material}",
                "role": "material_ood",
                "relation": "held-out material family",
                "assumptions": ["same schedule"],
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
            "Model ID": production_frame(
                id_records,
                artifact_root=str(tmp_path / "id"),
                dataset_name="transient_drying__lentil+chickpea__id",
                dataset_role="id",
            ),
            "Model Material OOD": production_frame(
                ood_records,
                artifact_root=str(tmp_path / "material_ood"),
                dataset_name="kidney_bean_holdout",
                dataset_role="material_ood",
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
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    with monkeypatch.context() as stubbed:
        stubbed.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
        viewer = analysis.evaluation.panel._make_transient_sample_viewer(
            session=evaluation,
            fields=fields,
            comparison=False,
        )
        assert routed[-1] == ("Model ID", "transient_drying__chickpea__natural__case_0051")
        # Each dataset (material and role) lists its own cases; choosing the kidney-bean dataset routes to the owning artifact.
        kidney = next(
            widget
            for widget in _widget_tree(viewer.widget)
            if isinstance(widget, widgets.ToggleButton) and widget.description == "kidney bean, Material OOD"
        )
        kidney.value = True
        assert routed[-1] == (
            "Model Material OOD",
            "transient_drying__kidney_bean__natural__case_0004",
        )

    inventory = evaluation.partitioned_case_inventory()
    assert tuple(entry.dataset_role for entry in inventory) == ("id", "id", "material_ood")
    assert inventory[-1].evaluation_assignment is not None
    assert inventory[-1].evaluation_assignment["role"] == "material_ood"
    assert tuple(entry.artifact_root for entry in inventory) == (
        (tmp_path / "id").resolve(),
        (tmp_path / "id").resolve(),
        (tmp_path / "material_ood").resolve(),
    )
    aggregate = evaluation.dataset_dataframe(modes=("autonomous_full",))
    assert set(aggregate["material_family"]) == {
        "chickpea",
        "lentil",
        "kidney_bean",
    }
    assert set(aggregate["dataset_role"]) == {"id", "material_ood"}
    assert set(aggregate["artifact_root"]) == {
        str((tmp_path / "id").resolve()),
        str((tmp_path / "material_ood").resolve()),
    }

    routed.clear()
    aggregate_view = analysis.evaluation.panel._make_transient_aggregate_viewer(
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
            "Model Material OOD",
            "transient_drying__kidney_bean__natural__case_0004",
        ),
    ]
    printed = _shown_text(aggregate_view)
    assert all(f"{label}: n(t)=1" in printed for label in ("chickpea, ID", "lentil, ID", "kidney bean, Material OOD"))
    evaluation.close()
    plt.close("all")


def test_sample_view_keeps_physical_time_local_and_rollout_coordinates_outside(
    tmp_path: Path,
) -> None:
    """Use full-rollout sample time without adding global protocol or horizon controls."""
    records = (
        sequence_record("teacher_forced_one_step", 1),
        sequence_record("autonomous_full", "full"),
        sequence_record("rolling_origin", 2),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(
        session=evaluation,
        fields=fields,
        comparison=False,
    )
    assert not any(
        isinstance(widget, widgets.Dropdown) and widget.description in {"Protocol", "Origin", "Horizon"} for widget in _widget_tree(viewer.widget)
    )
    assert any(isinstance(widget, widgets.FloatText) and widget.description == "Time [h]" for widget in _widget_tree(viewer.widget))
    evaluation.close()
    plt.close("all")


def test_outlier_field_view_starts_from_persisted_worst_channel_case(
    tmp_path: Path,
) -> None:
    """Route the per-channel outlier field view through one exact ranked case."""
    records = (
        sequence_record("autonomous_full", "full", case_id="best", offset=1.0),
        sequence_record("autonomous_full", "full", case_id="worst", offset=3.0),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    export_state: dict[str, object] = {}
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(
        session=evaluation,
        fields=fields,
        comparison=False,
        export_state=export_state,
        export_plot_name="worst_cases",
        outlier=True,
    )
    assert "worst" in viewer.status.text
    assert "selected-channel error-max time" in _shown_text(viewer)
    assert "case_worst" in str(export_state["filename_stem"])
    assert not any(isinstance(widget, widgets.FloatText) and widget.description == "Time [h]" for widget in _widget_tree(viewer.widget))
    evaluation.close()
    plt.close("all")


def test_a_comparison_draws_one_dataset_in_the_aggregate_and_scope_views_where_one_model_draws_several(
    tmp_path: Path,
) -> None:
    """Every comparison view is single-choice (O1): the dataset selector holds one-of buttons, a single model keeps checkboxes."""
    records = (
        sequence_record("autonomous_full", "full", case_id="first", material_family="chickpea"),
        sequence_record("autonomous_full", "full", case_id="second", material_family="lentil"),
    )
    evaluation = session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path / "artifact"))})
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    datasets = {"chickpea, ID", "lentil, ID"}

    def dataset_widgets(view: analysis.ui.panel.ViewPanel) -> list[Any]:
        return [widget for widget in _widget_tree(view.widget) if getattr(widget, "description", None) in datasets]

    for comparison in (True, False):
        views = (
            analysis.evaluation.panel._make_transient_aggregate_viewer(
                session=evaluation, fields=fields, plot_kind="mean_error", comparison=comparison
            ),
            analysis.evaluation.panel._make_transient_scope_viewer(session=evaluation, fields=fields, plot_kind="trajectory", comparison=comparison),
        )
        for view in views:
            offered = dataset_widgets(view)
            assert len(offered) == len(datasets)
            if comparison:
                assert all(isinstance(widget, widgets.ToggleButton) for widget in offered)
                assert [widget.value for widget in offered].count(True) == 1
            else:
                assert all(isinstance(widget, widgets.Checkbox) for widget in offered)
    evaluation.close()
    plt.close("all")


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

    summary["contributing_record_count"] = 1
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary)
    horizon_axis = figure.axes[0]
    lines = {line.get_label(): line for line in horizon_axis.lines}

    assert [label.get_text() for label in horizon_axis.get_xticklabels()] == ["1", "2", "full"]
    np.testing.assert_array_equal(lines["Lentil · A — cumulative"].get_xdata(), np.asarray([0, 2]))
    np.testing.assert_array_equal(lines["Lentil · B — cumulative"].get_xdata(), np.asarray([1, 2]))
    plt.close(figure)


def test_evaluation_physical_time_axes_reuse_shared_units_and_export() -> None:
    """Format simulated time centrally while leaving rollout indices distinct."""
    record = sequence_record("autonomous_full", "full")
    record.physical_times = np.asarray((0.0, 84.0, 168.0))
    central = analysis.evaluation.plots.transient.plot_central_error_vs_time(
        record,
        scaling_state=scaling_state(),
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
    summary["contributing_record_count"] = 1
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


def test_transient_panel_plots_and_pipeline_remain_sequence_aware(tmp_path: Path) -> None:
    """Keep exact-time controls, all states, and unavailable C."""
    records = (
        sequence_record("teacher_forced_one_step", 1),
        sequence_record("autonomous_full", "full"),
        sequence_record("rolling_origin", 2),
    )
    frame = production_frame(records, artifact_root=str(tmp_path / "artifact"))
    scale = error_diagnostics.transient_reference_scale(
        frame.attrs["transient_scaling_state"],
        field="T",
        unit="K",
        dataset_identity="a" * 64,
    )
    assert isinstance(scale, error_diagnostics.ReferenceScale)
    for record in records:
        record.identity["scaling_identity"]["semantic_digest"] = scale.scaler_identity
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
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(
        session=evaluation,
        fields=fields,
        comparison=False,
        export_state=export_state,
        export_plot_name="maps",
        export_title="maps",
    )
    assert isinstance(viewer, analysis.ui.panel.ViewPanel)
    case_row, channel_row, _display_row, time_row = (cast("widgets.Box", row) for row in viewer.rows)
    assert tuple(getattr(control, "description", None) for control in cast("widgets.Box", case_row.children[1]).children) == ("Case", "◀", "▶")
    boxes = {box.tooltip: box for box in cast("widgets.Box", channel_row.children[0]).children if isinstance(box, widgets.Checkbox)}
    assert all(box.value is True for box in boxes.values())
    gif_fps, gif_button = time_row.children[1], time_row.children[2]
    assert isinstance(gif_fps, widgets.BoundedIntText)
    assert (gif_fps.value, gif_fps.min, gif_fps.max) == (3, 1, 10)
    assert isinstance(gif_button, widgets.Button)
    assert gif_button.description == "Export GIF"
    navigator_controls = cast("widgets.Box", time_row.children[0])
    assert tuple(cast("Any", control).description for control in navigator_controls.children) == (
        "Time [h]",
        "◀◀",
        "◀",
        "▶",
        "▶▶",
    )
    assert isinstance(navigator_controls.children[0], widgets.FloatText)
    assert not any(isinstance(control, widgets.Play) for control in navigator_controls.children)
    physical_time = navigator_controls.children[0]
    physical_time.value = 0.0
    boxes["T"].value = False
    assert physical_time.value == 0.0
    assert isinstance(export_state["fig"], Figure)
    for checkbox in boxes.values():
        checkbox.value = False
    assert export_state["fig"] is None
    boxes["T"].value = True

    trajectory_viewer = analysis.evaluation.panel._make_transient_scope_viewer(
        session=evaluation,
        fields=fields,
        plot_kind="trajectory",
        comparison=False,
    )
    trajectory_fields = {box.tooltip: box for box in _children(_children(trajectory_viewer.rows[1])[0]) if isinstance(box, widgets.Checkbox)}
    assert tuple(trajectory_fields) == (
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    )
    trajectory_fields["w_gr"].value = True
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(trajectory_viewer.widget))

    with pytest.raises(ValueError, match="unavailable"):
        analysis.evaluation.plots.transient.plot_state_maps(records[0], physical_time=0.5)
    figure = analysis.evaluation.plots.transient.plot_state_maps(records[0], error_mode="Signed error")
    figure.canvas.draw()
    expected_map_count = len(analysis.evaluation.transient_artifact.STATE_ORDER) * 3
    map_axes = tuple(figure.axes[:expected_map_count])
    colorbar_axes = tuple(figure.axes[expected_map_count:])
    assert tuple(axis.get_title() for axis in map_axes[:3]) == ("Reference", "Prediction", "Signed error")
    row_labels = tuple(text.get_text() for axis in map_axes for text in axis.texts if text.get_gid() == "channel-row-label")
    assert len(row_labels) == len(analysis.evaluation.transient_artifact.STATE_ORDER)
    assert "°C" not in row_labels[0]
    assert colorbar_axes[0].get_ylabel() == "[°C]"
    assert len(map_axes) == expected_map_count
    assert len(colorbar_axes) == expected_map_count
    geometry = analysis.evaluation.plots.transient.state_map_geometry(records[0])
    np.testing.assert_allclose(figure.get_size_inches(), (geometry.figure_width, geometry.figure_height))
    assert set(figure.axes) == {*map_axes, *colorbar_axes}
    figure.canvas.draw()
    assert all(
        map_axis.get_position().y0 - 1e-6 <= colorbar_axis.get_position().y0 and colorbar_axis.get_position().y1 <= map_axis.get_position().y1 + 1e-6
        for map_axis, colorbar_axis in zip(map_axes, colorbar_axes, strict=True)
    )
    plt.close(figure)

    summary_frame = evaluation.dataset_dataframe()
    channel_figure = analysis.evaluation.plots.transient.plot_channel_error(summary_frame)
    assert channel_figure.axes[0].get_ylabel() == "Normalized RMSE [-]"
    assert len(channel_figure.axes[0].patches) == len(analysis.evaluation.transient_artifact.STATE_ORDER)
    plt.close(channel_figure)

    degradation = evaluation.pipeline_degradation()
    assert len(degradation) == 1
    assert degradation[0].c_available is False
    assert degradation[0].complete_pipeline_error is None
    runtime_figure = analysis.evaluation.plots.transient.plot_timing_distributions(evaluation.timing_report("id"))
    assert all(axis.get_ylabel() == "Recorded duration [s]" for axis in runtime_figure.axes)
    plt.close(runtime_figure)
    evaluation.close()


def _map_record() -> SimpleNamespace:
    """Return a 6x6 autonomous record with three times whose states vary in space and time."""
    record = sequence_record("autonomous_full", "full")
    ramp = np.linspace(0.0, 1.0, 36).reshape(6, 6)
    base = np.array([300.0, 0.3, 0.5, 0.6]).reshape(1, 4, 1, 1)
    step = np.array([10.0, 0.1, 0.1, 0.1]).reshape(1, 4, 1, 1)
    spatial = np.array([10.0, 0.1, 0.2, 0.2]).reshape(1, 4, 1, 1)
    reference = base + step * np.arange(3).reshape(3, 1, 1, 1) + spatial * ramp
    record.reference_states = reference.astype(np.float32)
    record.predicted_states = (reference + 0.3 * spatial * (ramp - 0.5)).astype(np.float32)
    record.static_conditioning = np.zeros((7, 6, 6), dtype=np.float32)
    record.static_conditioning[0], record.static_conditioning[1] = np.meshgrid(np.arange(6.0), np.arange(6.0))
    record.static_conditioning[_RHO_CHANNEL] = 700.0 + 100.0 * ramp  # varying, so a bulk value differs from a cell mean
    return record


def test_error_scales_cover_every_given_time_while_state_scales_follow_the_drawn_time() -> None:
    """Fixed error scales start at 0 and span all given times; reference and prediction scale to the drawn time alone."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    common = {"state_fields": ("T", "w_surf"), "error_mode": "Absolute error"}
    over_time = cast("Any", transient.state_map_norms(record, time_indices=(0, 2), lock_scale=False, **common))
    for field, channel in (("T", 0), ("w_surf", 2)):
        reference, prediction = (
            np.asarray(states[[0, 2], channel], dtype=np.float64) for states in (record.reference_states, record.predicted_states)
        )
        if field == "w_surf":  # differences of wet-basis moisture in percentage points
            reference, prediction = (_wet_basis_percent(values, record.static_conditioning[_RHO_CHANNEL]) for values in (reference, prediction))
        shown = np.abs(prediction - reference)
        assert over_time[field].error.boundaries[0] == 0.0
        assert over_time[field].error.boundaries[-1] >= shown.max()
    figure = transient.plot_state_maps(record, physical_time=1.0, norms=over_time, lock_scale=False, **common)
    for column, states in ((0, record.reference_states), (1, record.predicted_states)):
        # Without the lock each panel gets round discrete bands over its own 5th to 95th percentile and marks the cells beyond them with arrows.
        drawn = transient._display_state_values("T", states[1, 0], error=False)
        image = cast("Any", figure.axes[column].collections[0])
        low, high = np.percentile(drawn, (5.0, 95.0))
        assert isinstance(image.norm, BoundaryNorm)
        assert image.norm.vmin <= low <= high <= image.norm.vmax
        assert float(image.norm.boundaries[-1] - image.norm.boundaries[0]) <= 2.0 * (high - low) + 1.0e-9  # tight, not half empty
    # A bar given norms of several times draws the union of the norms' arrows, whichever time is drawn.
    for row, field in enumerate(common["state_fields"]):
        for column in (0, 1):
            assert cast("Any", figure.axes[3 * row + column].collections[0]).colorbar.extend == over_time[field].arrows[column]
    plt.close(figure)


def test_water_shows_as_wet_basis_percent_with_negative_predicted_water_masked_and_errors_in_percentage_points() -> None:
    """Every water view converts with the domain formula per cell; negative predicted water is masked, never raised or clipped."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[1, 2:4, 2:4, 2:4] = -0.1  # negative surface and internal water in four cells at t = 1
    rho = np.asarray(record.static_conditioning[_RHO_CHANNEL], dtype=np.float64)
    grain_reference, grain_prediction = (
        transient._grain_moisture(record, np.asarray(states, dtype=np.float64))[1] for states in (record.reference_states, record.predicted_states)
    )
    valid = grain_prediction >= 0.0
    reference_percent = _wet_basis_percent(grain_reference, rho)
    prediction_percent = _wet_basis_percent(grain_prediction[valid], rho[valid])
    figure = transient.plot_state_maps(record, state_fields=("w_gr",), physical_time=1.0, error_mode="Signed error")
    reference_panel, prediction_panel, error_panel = (cast("Any", figure.axes[column].collections[0].get_array()) for column in range(3))
    np.testing.assert_allclose(reference_panel, reference_percent)
    assert np.ma.count_masked(prediction_panel) == np.ma.count_masked(error_panel) == 4
    np.testing.assert_allclose(prediction_panel.compressed(), prediction_percent)
    np.testing.assert_allclose(error_panel.compressed(), prediction_percent - reference_percent[valid])
    row_labels = [text.get_text() for text in figure.axes[0].texts if text.get_gid() == "channel-row-label"]
    assert row_labels == [r"$\mathbf{X_{\mathbf{wb}}}$"]  # the default symbol style
    assert [analysis.presentation.moisture_display.display_name(field) for field in ("w_surf", "w_int")] == ["X_wb,surf", "X_wb,int"]
    bar_labels = [slot.get_ylabel() for slot in figure.axes[3:]]
    assert bar_labels[:2] == ["[%]", "[%]"]
    assert bar_labels[2] == "[pp]"  # no power-of-ten factor: the tick decimals carry small steps
    plt.close(figure)
    # The normalized error stays on the stored water and the saved train scale.
    scale = error_diagnostics.transient_reference_scale(scaling_state(), field="w_gr", unit="kg/m^3", dataset_identity="a" * 64)
    assert isinstance(scale, error_diagnostics.ReferenceScale)
    normalized = transient.plot_state_maps(
        record, state_fields=("w_gr",), physical_time=1.0, error_mode=error_diagnostics.NORMALIZED_ABSOLUTE_ERROR, reference_scales={"w_gr": scale}
    )
    normalized_error = np.abs(grain_prediction - grain_reference) / scale.value
    np.testing.assert_allclose(cast("Any", normalized.axes[2].collections[0].get_array()), normalized_error)
    plt.close(normalized)
    trajectory = transient.plot_state_trajectory_summary({"case": [record]}, state_fields=("w_gr",))
    lines = {line.get_label(): cast("Any", line.get_ydata()) for line in trajectory.axes[0].get_lines()}
    for label, water in (("case · Reference", grain_reference), ("case · Prediction", grain_prediction)):
        assert lines[label][1] == pytest.approx(100.0 * water[valid].sum() / (rho[valid].sum() + water[valid].sum()))
    assert trajectory.axes[0].get_ylabel() == r"$X_{\mathrm{wb}}$ [%]"
    plt.close(trajectory)
    aggregate = transient.plot_aggregate_error_maps([record], state_fields=("w_gr",), physical_time=1.0, statistic="mean")
    pooled = np.asarray(cast("Any", aggregate.axes[0].collections[0].get_array()))
    np.testing.assert_allclose(pooled[valid], np.abs(prediction_percent - reference_percent[valid]))
    assert np.isnan(pooled[~valid]).all()
    plt.close(aggregate)
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("w_gr",))
    assert trend.axes[0].get_lines()[0].get_label() == "case · 4 of 72 values non-finite, excluded"
    plt.close(trend)
    assert transient.maximum_error_physical_time(record, state_fields=("w_gr",)) in {1.0, 2.0}


def test_relative_humidity_shows_in_percent_rh_with_errors_in_percentage_points_and_normalized_errors_on_the_fraction() -> None:
    """Phi maps and trajectories read % RH, its errors pp without a power of ten; |e|/sigma_train stays on the stored fraction."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    reference, prediction = (np.asarray(states[1, 1], dtype=np.float64) for states in (record.reference_states, record.predicted_states))
    figure = transient.plot_state_maps(record, state_fields=("phi",), physical_time=1.0, error_mode="Signed error")
    reference_panel, _prediction_panel, error_panel = (cast("Any", figure.axes[column].collections[0].get_array()) for column in range(3))
    np.testing.assert_allclose(reference_panel, 100.0 * reference, rtol=1e-6)
    np.testing.assert_allclose(error_panel, 100.0 * (prediction - reference), rtol=1e-5, atol=1e-9)
    assert [slot.get_ylabel() for slot in figure.axes[3:]] == ["[%]", "[%]", "[pp]"]
    plt.close(figure)
    scale = error_diagnostics.transient_reference_scale(scaling_state(), field="phi", unit="1", dataset_identity="a" * 64)
    assert isinstance(scale, error_diagnostics.ReferenceScale)
    normalized = transient.plot_state_maps(
        record, state_fields=("phi",), physical_time=1.0, error_mode=error_diagnostics.NORMALIZED_ABSOLUTE_ERROR, reference_scales={"phi": scale}
    )
    np.testing.assert_allclose(cast("Any", normalized.axes[2].collections[0].get_array()), np.abs(prediction - reference) / scale.value, rtol=1e-6)
    plt.close(normalized)
    trajectory = transient.plot_state_trajectory_summary({"case": [record]}, state_fields=("phi",))
    lines = {line.get_label(): cast("Any", line.get_ydata()) for line in trajectory.axes[0].get_lines()}
    assert lines["case · Reference"][1] == pytest.approx(100.0 * reference.mean())
    assert trajectory.axes[0].get_ylabel() == r"$\varphi$ [%]"
    plt.close(trajectory)


def test_maximum_error_time_lands_on_a_diverging_state_prediction() -> None:
    """A non-finite predicted state cell makes its time the worst instead of being averaged away."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[2, 0] += 5.0
    record.predicted_states[1, 0, 3, 3] = np.inf
    assert transient.maximum_error_physical_time(record, state_fields=("T", "w_gr")) == pytest.approx(float(record.physical_times[1]))


def test_a_partly_diverging_prediction_masks_only_its_cells_and_is_named_in_every_pooled_view() -> None:
    """Diverging cells are masked in error maps and pooled errors and counted in the parity and error-trend plots."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[1:, 0, 3, 3] = np.nan
    maps = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0, error_mode="Absolute error")
    assert np.ma.count_masked(maps.axes[2].collections[0].get_array()) == 1
    plt.close(maps)
    _x, _y, mean_error, std_error, _reference_means, prediction_means = transient._aggregate_snapshot_evidence(
        [record], state_field="T", physical_time=1.0
    )
    assert np.isnan(mean_error[3, 3])
    assert np.isnan(std_error[3, 3])
    assert np.count_nonzero(np.isfinite(mean_error)) == mean_error.size - 1
    assert not np.isfinite(prediction_means).any()
    plt.close(transient.plot_aggregate_error_maps([record], state_fields=("T",), physical_time=1.0, statistic="mean"))
    parity = transient.plot_predicted_vs_reference_channels([record], state_fields=("T",), physical_time=1.0)
    assert any(text.get_text().startswith("1 non-finite") for text in parity.axes[0].texts)
    plt.close(parity)
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T",))
    assert trend.axes[0].get_lines()[0].get_label() == "case · 2 of 72 values non-finite, excluded"
    plt.close(trend)
    record.predicted_states[1:, 0] = np.nan
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T",))
    (line,) = trend.axes[0].get_lines()
    assert line.get_label() == "case · 72 of 72 values non-finite, excluded"
    assert len(cast("Any", line.get_ydata())) == 0
    plt.close(trend)


def test_thinned_error_trends_name_the_exact_non_finite_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Divergences between the thinning positions are counted exactly and named, while the finite medians stay as they were."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    monkeypatch.setattr(transient, "_TARGET_MAGNITUDE_POINT_LIMIT", 36)  # 72 post-origin elements: stride 2
    finite_centers, finite_medians, *finite_counts = transient._finite_error_trend([record], "T")
    odd_cells = (np.arange(36).reshape(6, 6) % 2).astype(bool)
    record.predicted_states[1:, 0][:, odd_cells] = np.nan  # every odd raveled element, which stride 2 never samples
    centers, medians, *counts = transient._finite_error_trend([record], "T")
    assert (finite_counts, counts) == ([0, 72], [36, 72])
    np.testing.assert_array_equal(centers, finite_centers)
    np.testing.assert_array_equal(medians, finite_medians)
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T",))
    assert trend.axes[0].get_lines()[0].get_label() == "case · 36 of 72 values non-finite, excluded"
    plt.close(trend)


def test_the_error_trend_xlabel_is_the_plot_label_of_the_field_in_the_active_style_between_absolute_bars() -> None:
    """The |reference| xlabel follows the label style and its unit, for a water density and for a plain state alike."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    labels = {}
    try:
        for style in ("symbol", "name"):
            analysis.presentation.field_labels.set_label_style(style)
            trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T", "w_gr"))
            labels[style] = [axis.get_xlabel() for axis in trend.axes]
            plt.close(trend)
    finally:
        analysis.presentation.field_labels.set_label_style("symbol")
    assert labels["symbol"] == ["|$T$| [°C]", r"|$X_{\mathrm{wb}}$| [%]"]
    assert labels["name"] == ["|Temperature| [°C]", "|Grain moisture| [% w.b.]"]


def test_panels_without_finite_values_state_so_and_a_locked_scale_takes_the_reference_percentiles_alone() -> None:
    """An all-non-finite prediction panel is labelled; a locked scale spans the reference's 5th to 95th percentile, arrows mark the rest."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[1, 0] = np.nan
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    assert "unavailable: no finite values" in [text.get_text() for text in figure.axes[1].texts]
    plt.close(figure)
    record = _map_record()
    record.predicted_states[1, 0] = record.reference_states[1, 0].max() + 50.0  # every predicted cell above the reference
    reference = transient._display_state_values("T", record.reference_states[1, 0], error=False)
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    for column, extend in ((0, None), (1, "max")):
        image = cast("Any", figure.axes[column].collections[0])
        low, high = np.percentile(reference, (5.0, 95.0))
        assert isinstance(image.norm, BoundaryNorm)
        assert image.norm.vmin <= low <= high <= image.norm.vmax  # the reference's round bands; the prediction takes them unchanged
        bands = np.array([image.norm.vmin, image.norm.vmax])
        assert image.colorbar.extend == (extend if column else transient.field_maps.locked_extend(reference, bands))
    plt.close(figure)


def test_a_constant_reference_falls_back_to_a_padded_reference_scale_and_the_prediction_gets_an_arrow() -> None:
    """A constant reference gets a padded reference-only scale instead of raising; the prediction never widens it."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.reference_states[1, 0] = 300.0
    record.predicted_states[1, 0] = 301.0
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    for column, extend in ((0, "neither"), (1, "max")):
        image = cast("Any", figure.axes[column].collections[0])
        assert (image.norm.vmin, image.norm.vmax) == pytest.approx((300.0 - 273.15, 300.0 - 273.15), abs=1e-6)
        assert image.colorbar.extend == extend
    plt.close(figure)


def test_each_error_map_carries_one_mean_line_in_its_bar_unit_and_state_maps_none() -> None:
    """The line states the mean |e| of the drawn error (signed: in the bar's unit; normalized: over sigma_train); it never touches the headers."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    scale = error_diagnostics.transient_reference_scale(scaling_state(), field="w_gr", unit="kg/m^3", dataset_identity="a" * 64)
    assert isinstance(scale, error_diagnostics.ReferenceScale)
    for mode, fields, scales in (
        ("Signed error", ("T", "phi"), None),
        (error_diagnostics.NORMALIZED_ABSOLUTE_ERROR, ("w_gr",), {"w_gr": scale}),
    ):
        figure = transient.plot_state_maps(record, state_fields=fields, physical_time=1.0, error_mode=mode, reference_scales=scales)
        figure.canvas.draw()
        rows = len(fields)
        maps, slots = figure.axes[: 3 * rows], figure.axes[3 * rows :]
        for row in range(rows):
            assert not [text for axis in maps[3 * row : 3 * row + 2] for text in axis.texts if text.get_gid() != "channel-row-label"]
            (line,) = (text for text in maps[3 * row + 2].texts if text.get_gid() != "channel-row-label")
            mean = np.nanmean(np.abs(np.ma.filled(cast("Any", maps[3 * row + 2].collections[0].get_array()).astype(float), np.nan)))
            # The line takes its bar's unit and power of ten, such as [10⁻³] of a fine normalized bar.
            bracket = slots[3 * row + 2].get_ylabel().rpartition("[")[2].rstrip("]")
            power, _space, unit = bracket.partition(" ") if bracket.startswith("10") else ("", "", bracket)
            exponent = int(power[2:].translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"))) if power else 0
            value = f"{mean / 10.0**exponent:.3g}" + (f" \N{MULTIPLICATION SIGN} {power}" if power else "")
            expected = f"mean |e| = {value} {unit}" if mode == "Signed error" else f"mean {error_diagnostics.NORMALIZED_ERROR_SYMBOL} = {value}"
            assert line.get_text() == expected
        # The top row's line stands between the header and its map, and the header stays below the scale note.
        renderer = cast("Any", figure.canvas).get_renderer()
        (top_line,) = (text for text in maps[2].texts if text.get_gid() != "channel-row-label")
        (note,) = (text for text in figure.texts if text.get_text().startswith("Colour scales:"))
        header = maps[2].title.get_window_extent(renderer)
        assert note.get_window_extent(renderer).y0 >= header.y1 >= header.y0 >= top_line.get_window_extent(renderer).y1
        plt.close(figure)


def test_the_error_lines_reserve_their_room_in_every_frame_so_the_maps_never_move() -> None:
    """A frame whose error is unavailable writes "mean |e| n/a" and every map and bar keeps the position of a finite frame."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[2, 0] = np.nan
    fields = ("T", "phi")
    norms = transient.state_map_norms(record, state_fields=fields, time_indices=range(3), error_mode="Signed error")
    figure, _grid = analysis.ui.plot_layout.map_figure(transient.state_map_geometry(record, state_fields=fields))
    positions, lines = [], []
    for time in (1.0, 2.0):
        transient.plot_state_maps(record, state_fields=fields, physical_time=time, norms=norms, figure=figure)
        figure.canvas.draw()
        positions.append([axis.get_position().bounds for axis in figure.axes])
        lines.append([text.get_text() for axis in figure.axes[:6] for text in axis.texts if text.get_text().startswith("mean")])
    assert positions[0] == positions[1]
    assert [len(frame) for frame in lines] == [2, 2]
    assert lines[1][0] == "mean |e| n/a"  # the temperature error of the diverged frame
    assert lines[0][0].startswith("mean |e| = ")
    plt.close(figure)


def test_one_scale_note_stands_under_the_title_of_the_signed_default() -> None:
    """The error column is signed by default; the one note line names the scale rule, and the title leads."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    figure = transient.plot_state_maps(record, state_fields=("T", "phi"), physical_time=1.0, title_context="Lentil · case 7")
    assert figure.axes[2].get_title() == "Signed error"
    (note,) = (text for text in figure.texts if text.get_text().startswith("Colour scales:"))
    assert "5th\N{EN DASH}95th percentile" in note.get_text()
    assert "prediction uses the reference scale" in note.get_text()
    (title,) = (text for text in figure.texts if text.get_text() == figure.get_suptitle())
    assert note.get_position()[1] < title.get_position()[1]
    plt.close(figure)
    unlocked = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0, lock_scale=False)
    assert "prediction uses the reference scale" not in "".join(text.get_text() for text in unlocked.texts)
    plt.close(unlocked)


def _gif_viewer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[session.TransientEvaluationSession, widgets.Button, Any]:
    """Return a sample viewer over one case with three varying stored times and an export folder in ``tmp_path``, its GIF button, and the viewer."""
    record = sequence_record("autonomous_full", "full")
    values = (0.3 * np.arange(3, dtype=np.float32)).reshape(3, 1, 1, 1) * np.ones((3, 4, *GRID_SHAPE), dtype=np.float32)
    record.reference_states = values
    record.predicted_states = values + 0.1 + 0.05 * np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1)
    evaluation = session.TransientEvaluationSession({"id": production_frame((record,), artifact_root=str(tmp_path / "artifact"))})
    monkeypatch.setattr(analysis.ui.notebook, "_export_destination", lambda: tmp_path)
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(
        session=evaluation,
        fields=analysis.evaluation.panel._transient_channel_fields(evaluation),
        comparison=False,
    )
    boxes = {box.tooltip: box for box in _children(_children(viewer.rows[1])[0]) if isinstance(box, widgets.Checkbox)}
    boxes["phi"].value = False
    boxes["w_int"].value = False
    return evaluation, cast("widgets.Button", _children(viewer.rows[3])[2]), viewer


def test_gif_export_writes_all_stored_times_of_the_active_channels_with_one_frame_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two active channels and three stored times give one GIF with three equally sized, looping frames."""
    from PIL import Image  # noqa: PLC0415 -- Pillow reads the written file back; it is a matplotlib dependency

    evaluation, button, viewer = _gif_viewer(tmp_path, monkeypatch)
    button.click()
    (path,) = tmp_path.glob("*.gif")
    assert "channel_t-w_surf" in path.name
    assert "t-all_fixed-t" in path.name  # temperature is fixed over time by default; the export names it, lower-cased
    assert button.disabled is False
    geometry = analysis.evaluation.plots.transient.state_map_geometry(sequence_record("autonomous_full", "full"), state_fields=("T", "w_surf"))
    with Image.open(path) as image:
        assert cast("Any", image).n_frames == 3
        assert image.info["loop"] == 0
        assert image.info["duration"] == pytest.approx(1000 / evaluation_panel._GIF_FPS, abs=10)  # the field's default
        assert image.size[0] == pytest.approx(geometry.figure_width * evaluation_panel._GIF_DPI, abs=1)
        assert image.size[1] == pytest.approx(geometry.figure_height * evaluation_panel._GIF_DPI, abs=1)
    # The fps field next to the button sets the speed: its value goes to the writer.
    path.unlink()
    cast("widgets.BoundedIntText", _children(viewer.rows[3])[1]).value = 10
    button.click()
    (path,) = tmp_path.glob("*.gif")
    with Image.open(path) as image:
        assert image.info["duration"] == pytest.approx(100, abs=10)
    evaluation.close()


def test_gif_export_follows_the_viewers_lock_toggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every GIF frame is drawn with the viewer's lock toggle: on shares the reference's scale, off gives the prediction its own."""
    evaluation, button, viewer = _gif_viewer(tmp_path, monkeypatch)
    locks = []
    render = analysis.evaluation.plots.transient.plot_state_maps

    def record_lock(*args: object, **kwargs: object) -> Figure:
        locks.append(kwargs["lock_scale"])
        return render(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_state_maps", record_lock)
    for locked in (True, False):
        cast("widgets.Checkbox", _children(viewer.rows[2])[1]).value = locked
        locks.clear()
        button.click()
        assert locks == [locked] * 3
        for path in tmp_path.glob("*.gif"):
            path.unlink()
    evaluation.close()


def _fixed_time_boxes(viewer: Any) -> dict[str, widgets.Checkbox]:
    """Return the "Fixed over time" checkboxes of a sample viewer by channel key."""
    return {box.tooltip: box for box in _children(_children(viewer.rows[2])[2]) if isinstance(box, widgets.Checkbox)}


def test_the_viewer_fixes_all_channels_but_the_moisture_ones_over_time_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The "Fixed over time" control offers the channels of the channel control and starts with temperature and humidity checked, moisture free."""
    evaluation, _button, viewer = _gif_viewer(tmp_path, monkeypatch)
    channels = [box.tooltip for box in _children(_children(viewer.rows[1])[0]) if isinstance(box, widgets.Checkbox)]
    boxes = _fixed_time_boxes(viewer)
    assert cast("widgets.Label", _children(_children(viewer.rows[2])[2])[0]).value == "Fixed over time"
    assert list(boxes) == channels
    assert {key for key, box in boxes.items() if box.value} == {"T", "phi"}
    evaluation.close()


def test_gif_export_follows_the_viewers_fixed_over_time_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every GIF frame is drawn with the checked fixed fields among the active channels, its norms pool exactly those, and the name lists them."""
    evaluation, button, viewer = _gif_viewer(tmp_path, monkeypatch)
    boxes = _fixed_time_boxes(viewer)
    seen = []
    render = analysis.evaluation.plots.transient.plot_state_maps

    def record_fields(*args: object, **kwargs: object) -> Figure:
        norms = cast("Any", kwargs["norms"])
        if norms is not None:  # the viewer draws a setting without a fixed field from its own time
            seen.append((kwargs["fixed_time_fields"], tuple(field for field, norm in norms.items() if norm.fixed is not None)))
        return render(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_state_maps", record_fields)
    # The active channels are T and w_surf; phi is checked fixed by default but not drawn.
    for changes, expected in (({}, ("T",)), ({"T": False}, ()), ({"T": True, "w_surf": True}, ("T", "w_surf"))):
        for key, value in changes.items():
            boxes[key].value = value
        seen.clear()
        button.click()
        assert seen == [(expected, expected)] * 3
        (path,) = tmp_path.glob("*.gif")
        assert ("t-all_fixed-" + "-".join(expected).lower() if expected else "t-all") in path.name  # the export lower-cases the stem
        assert ("fixed" in path.name) == bool(expected)
        path.unlink()
    evaluation.close()


def test_the_viewer_computes_the_norms_over_all_stored_times_once_per_setting_while_a_field_is_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time steps reuse the norms over all stored times, a changed norm setting recomputes them, and with no fixed field a step uses its own time."""
    evaluation, _button, viewer = _gif_viewer(tmp_path, monkeypatch)
    transient = analysis.evaluation.plots.transient
    spans = []
    build = transient.state_map_norms

    def counted(*args: Any, **kwargs: Any) -> Any:
        spans.append(len(tuple(kwargs["time_indices"])))
        return build(*args, **kwargs)

    monkeypatch.setattr(transient, "state_map_norms", counted)
    clock = next(widget for widget in _widget_tree(viewer.widget) if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]")
    cast("widgets.Checkbox", _children(viewer.rows[2])[1]).value = False  # the lock enters the norms
    assert spans == [3]
    for time in (0.0, 1.0, 2.0):
        clock.value = time
    assert spans == [3]
    _fixed_time_boxes(viewer)["T"].value = False  # nothing fixed: every time step scales to itself
    clock.value = 1.0
    assert spans == [3, 1, 1]
    evaluation.close()


def test_with_a_fixed_field_the_viewer_draws_a_time_with_the_scales_arrows_and_ticks_of_its_gif_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The viewer at a stored time and the GIF frame of that time carry the same colour bands, arrows and tick labels."""
    evaluation, button, viewer = _gif_viewer(tmp_path, monkeypatch)
    transient = analysis.evaluation.plots.transient
    render = transient.plot_state_maps
    drawn: dict[str, list[tuple[float, Any]]] = {"viewer": [], "gif": []}

    def scales(figure: Figure) -> list[tuple[Any, ...]]:
        figure.canvas.draw()
        images = [cast("Any", axis.collections[0]) for axis in figure.axes if axis.collections]
        return [
            (
                tuple(image.norm.boundaries),
                getattr(image.colorbar, "extend", None),
                tuple(label.get_text() for label in image.colorbar.ax.get_yticklabels()) if getattr(image, "colorbar", None) else (),
            )
            for image in images
            if isinstance(image.norm, BoundaryNorm)
        ]

    def record_scales(*args: Any, **kwargs: Any) -> Figure:
        figure = render(*args, **kwargs)
        drawn["gif" if kwargs.get("figure") is not None else "viewer"].append((float(kwargs["physical_time"]), scales(figure)))
        return figure

    monkeypatch.setattr(transient, "plot_state_maps", record_scales)
    clock = next(widget for widget in _widget_tree(viewer.widget) if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]")
    clock.value = 1.0
    button.click()
    (time, viewer_scales) = drawn["viewer"][-1]
    assert time == 1.0
    assert dict(drawn["gif"])[time] == viewer_scales
    assert viewer_scales  # the comparison covers drawn bands
    evaluation.close()


def test_the_viewer_norms_are_not_kept_across_cases_or_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """One cached entry: the same case and setting return the same norms, another case or setting computes anew, and no fixed field computes none."""
    computed: list[Any] = []

    def fake_norms(record: Any, _plot: Any) -> dict[str, Any]:
        computed.append(record)
        return {"record": record}

    monkeypatch.setattr(evaluation_panel, "_state_norms_over_all_times", fake_norms)
    plot: dict[str, Any] = {
        "state_fields": ("T",),
        "lock_scale": True,
        "error_mode": "Signed error",
        "fixed_time_fields": ("T",),
        "reference_scales": {"T": "scale 1"},
    }
    cache: dict[str, Any] = {}
    first, second = (SimpleNamespace(frame_name="id", case_id=case) for case in ("a", "b"))
    norms = evaluation_panel._viewer_state_norms(cache, first, "record a", plot)
    assert evaluation_panel._viewer_state_norms(cache, first, "record a", plot) is norms
    assert evaluation_panel._viewer_state_norms(cache, second, "record b", plot) == {"record": "record b"}
    assert evaluation_panel._viewer_state_norms(cache, first, "record a", {**plot, "error_mode": "Absolute error"}) == {"record": "record a"}
    assert evaluation_panel._viewer_state_norms(cache, first, "record a", {**plot, "reference_scales": {"T": "scale 2"}}) == {"record": "record a"}
    assert evaluation_panel._viewer_state_norms(cache, first, "record a", {**plot, "fixed_time_fields": ()}) is None
    assert computed == ["record a", "record b", "record a", "record a"]


def test_gif_export_refuses_above_the_memory_limit_and_reenables_the_button_after_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An oversized GIF creates no file; a failing frame removes the partial file and frees the button, and the status says why."""
    evaluation, button, viewer = _gif_viewer(tmp_path, monkeypatch)
    monkeypatch.setattr(analysis.evaluation.panel, "_GIF_MEMORY_LIMIT_BYTES", 1)
    button.click()
    assert "[GIF] Refused" in viewer.status.text
    assert viewer.status.level == "warning"
    assert not list(tmp_path.glob("*.gif"))
    assert button.disabled is False
    monkeypatch.undo()
    monkeypatch.setattr(analysis.ui.notebook, "_export_destination", lambda: tmp_path)

    def failing(*_args: object, **kwargs: object) -> Figure:
        if kwargs.get("figure") is None:
            plt.figure()  # a renderer that allocates its own figure before failing
        message = "frame failed"
        raise RuntimeError(message)

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_state_maps", failing)
    open_figures = plt.get_fignums()
    button.click()
    assert not list(tmp_path.glob("*.gif"))
    assert button.disabled is False
    assert plt.get_fignums() == open_figures
    assert viewer.status.level == "error"
    assert viewer.status.text.startswith("[Export] Failed")
    evaluation.close()


def test_gif_export_never_writes_through_a_path_replaced_after_its_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A symlink swapped in at the reserved name is neither followed nor reported as the saved GIF."""
    evaluation, button, viewer = _gif_viewer(tmp_path, monkeypatch)
    decoy = tmp_path / "decoy.bin"
    decoy.write_bytes(b"keep")
    render = analysis.evaluation.plots.transient.plot_state_maps

    def swap_then_render(*args: object, **kwargs: object) -> Figure:
        for reserved in tmp_path.glob("*.gif"):
            if not reserved.is_symlink():
                reserved.unlink()
                reserved.symlink_to(decoy)
        return render(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_state_maps", swap_then_render)
    button.click()
    status = viewer.status.text
    assert decoy.read_bytes() == b"keep"
    assert "was replaced during the export" in status
    assert "Saved" not in status
    evaluation.close()


def test_comparison_ranking_orders_by_the_selection_objective_and_never_ranks_partial_evidence() -> None:
    """
    Rank frames by their selection objective; a frame with an unavailable case ranks last as unavailable.

    The incomplete frame's only available case is better than every complete case, so it would lead on partial evidence.
    """
    complete = (
        sequence_record("autonomous_full", "full", case_id="a", offset=2.0, material_family="lentil"),
        sequence_record("autonomous_full", "full", case_id="b", offset=6.0, material_family="chickpea"),
    )
    incomplete = (
        sequence_record("autonomous_full", "full", case_id="a", offset=0.1, material_family="lentil"),
        sequence_record("autonomous_full", "full", case_id="b", offset=np.nan, material_family="chickpea"),
    )
    evaluation = session.TransientEvaluationSession({"Incomplete": sequence_frame(incomplete), "Complete": sequence_frame(complete)})
    ranked = analysis.evaluation.panel._transient_comparison_summary(evaluation)
    assert list(ranked["frame"]) == ["Complete", "Incomplete"]
    # Lentil 4 and chickpea 36 average to 20 per field; every field has Train scale 2.
    assert ranked.iloc[0]["normalized_drying_group_macro_rmse"] == pytest.approx(math.sqrt(20.0) / 2.0)
    assert math.isnan(evaluation.selection_objective("Incomplete"))
    assert pd.isna(ranked.iloc[1]["normalized_drying_group_macro_rmse"])
    assert not ranked.iloc[1]["Metric available"]
    assert list(ranked["Material count"]) == [2, 2]


def test_ranking_objective_equals_the_checkpoint_selection_objective_of_the_same_cases() -> None:
    """
    Reduce three full rollouts of two materials once through the session ranking and once through the selection metric.

    Both must weigh the two lentil cases as one material against the single chickpea case and agree exactly.
    """
    offsets = {("lentil", "a"): 2.0, ("lentil", "b"): 4.0, ("chickpea", "c"): 6.0}
    records = tuple(
        sequence_record("autonomous_full", "full", case_id=case_id, offset=offset, material_family=material)
        for (material, case_id), offset in offsets.items()
    )
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(records)})
    ranking = evaluation.selection_objective("id")

    task = domain.tasks.registry.get_task("transient_drying")
    objective = next(metric for metric in task.default_metrics if metric.kind == "group_macro_rmse")
    config = {"task": task.id, "evaluation": {"metrics": [objective.as_dict(all_fields=task.metric_names)]}}
    metric = learning.metrics.metrics.build_evaluation_metrics(config, device=torch.device("cpu"), output_scales=torch.full((4,), 2.0))[objective.id]
    assert isinstance(metric, learning.metrics.metrics.GroupMacroRMSEMetric)
    for index, ((material, case_id), offset) in enumerate(offsets.items()):
        # Two predicted steps after the stored origin, as in each record.
        prediction = torch.full((2, 4, 1, 1), offset, dtype=torch.float64)
        cases = learning.metrics.metrics.CaseMembership((case_id,) * 2, (material,) * 2)
        metric.update(prediction, torch.zeros_like(prediction), space="physical", batch_index=index, cases=cases)

    # Lentil 0.5 * (4 + 16) = 10 and chickpea 36 average to 23 per field; every field has Train scale 2.
    assert ranking == pytest.approx(math.sqrt(23.0) / 2.0)
    assert metric.compute() == pytest.approx(ranking, rel=1e-12)


def test_scoreboard_fields_and_material_rows_reduce_cases_like_the_selection_objective() -> None:
    """
    Weigh the cases of a material and the materials equally in every normalized column, whatever the rollout lengths.

    Lentil holds a short and a long case and chickpea one; pooling would weigh the long lentil case 5 : 2.
    """
    records = (
        sequence_record("autonomous_full", "full", case_id="a", offset=2.0, material_family="lentil"),
        sequence_record("autonomous_full", "full", case_id="b", offset=4.0, material_family="lentil", length=6),
        sequence_record("autonomous_full", "full", case_id="c", offset=6.0, material_family="chickpea"),
    )
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(records)})
    fields = analysis.evaluation.transient_artifact.STATE_ORDER
    (row,) = analysis.evaluation.panel._transient_comparison_summary(evaluation).to_dict("records")
    # Lentil 0.5 * (4 + 16) = 10 and chickpea 36 average to 23 per field; every field has Train scale 2.
    assert [row[f"normalized_rmse_{field}"] for field in fields] == pytest.approx([math.sqrt(23.0) / 2.0] * len(fields))
    materials = evaluation.dataset_dataframe(modes=("autonomous_full",)).query("scope == 'cumulative'").set_index("material_family")
    columns = ["normalized_drying_group_macro_rmse", *(f"normalized_rmse_{field}" for field in fields)]
    for material, mse in (("lentil", 10.0), ("chickpea", 36.0)):
        material_row = materials.loc[material]
        assert [material_row[column] for column in columns] == pytest.approx([math.sqrt(mse) / 2.0] * len(columns))


def test_pipeline_surrogate_error_is_the_selection_objective_for_unequal_case_counts_and_lengths(tmp_path: Path) -> None:
    """The B/A surrogate error weighs the cases of a material and the materials equally, whatever their counts and lengths."""
    records = (
        sequence_record("autonomous_full", "full", case_id="a", offset=2.0, material_family="lentil"),
        sequence_record("autonomous_full", "full", case_id="b", offset=4.0, material_family="lentil", length=6),
        sequence_record("autonomous_full", "full", case_id="c", offset=6.0, material_family="chickpea"),
    )
    evaluation = session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path / "artifact"))})
    (degradation,) = evaluation.pipeline_degradation()
    # Lentil 0.5 * (4 + 16) = 10 and chickpea 36 average to 23 per field; pooling the 2 + 5 + 2 predicted steps gives 160 / 9.
    assert degradation.drying_surrogate_error == pytest.approx(math.sqrt(23.0) / 2.0)
    assert degradation.drying_surrogate_error == evaluation.selection_objective("id")
