"""Protect transient Evaluation aggregation, presentation, and local reports.

Responsibilities:
  - Check pooled statistics, target records, lazy case access, panels, and local reports
  - Check that a session admits only frames recording one valid analysis producer, and that a mismatch of scoped roots names
    deleting them and generating again, never --rebuild
  - Check that the comparison ranking and the B/A pipeline error reduce cases exactly as checkpoint selection does, and that
    the scoreboard and the summary table shade their error columns, the configuration table never, φ errors in percentage points
  - Check that water densities show as wet-basis percent with masked unphysical cells, errors in percentage points,
    bulk means on shared valid cells and unchanged normalized errors
  - Check the time windows of the mean and error over time: the windows, the last 24 h before each case's end, the
    difference and case-count rows, the target lines and the one figure legend
  - Check the drying-phase views: every reached case at its nearest stored state (the origin included) at every phase point
    of the error over time, the relative norms, the maps and the parity, with equal material weight; censored cases counted
    and named with the effective time; target lines of the pooled cases only; a map note that stays off the maps; the phase
    default of the pooled time views over several cases, hours for one case per curve, a picked clock kept; and the refusal of
    a case without a time after its origin
  - Check the chosen time (the coverage default, a choice kept per session across views and cases, a cleared or emptied
    selection without a time) and the run labels of titles: only explicit labels, of every run a figure compares, after a case
    title's time; a compared run without drawable evidence stays named and the figure's note names its gap
  - Check series styles resolved once per comparison and looked up by record identity and summary provenance, one shade
    per run in every view also where only some runs have evidence, the matched-compute markers in their runs' colours, and the
    duration-ratio error bars with their refusal of ratios that are not positive and finite

Design principles:
  - Unequal support, unavailable metrics, and material summaries retain distinct meanings
  - The producer check reads recorded provenance only, never the running analysis code

This module does NOT:
  - Generate model predictions or refit saved scales
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import warnings
from dataclasses import asdict, replace
from inspect import Parameter, signature
from io import BytesIO
from itertools import pairwise
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import BoundaryNorm, to_hex
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
from matplotlib.quiver import Quiver
from matplotlib.text import Annotation, Text
from support.transient_evaluation import GRID_SHAPE, production_frame, scaling_state, sequence_frame
from support.transient_evaluation import sequence_record as support_sequence_record

from src import analysis, domain, learning
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_panel
from src.analysis.evaluation import evaluation_transient_session as session
from src.analysis.evaluation import evaluation_transient_validity as transient_validity
from src.analysis.evaluation import evaluation_workspace as workspace
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    from matplotlib.typing import ColorType

_RHO_CHANNEL = analysis.evaluation.transient_artifact.STATIC_ORDER.index("rho_bu_dry")


def sequence_record(*args: Any, dataset_role: str = "id", **kwargs: Any) -> SimpleNamespace:
    """
    Return the shared synthetic record with the entity identity of a produced one: its Dataset role and selected checkpoint.

    The checkpoint is the one ``production_frame`` names as the run's, so a record view and a summary view share the run.
    """
    record = support_sequence_record(*args, **kwargs)
    record.identity.update(dataset_role=dataset_role, checkpoint_identity={"best_checkpoint_sha256": "d" * 64})
    return record


# The comparison styles of the one FNO run that every record of ``sequence_record`` belongs to, in every role and material.
_STYLES = analysis.evaluation.plots.series.comparison_styles(
    ("fno", role, material, "d" * 64) for role in ("id", "parameter_ood", "material_ood") for material in ("lentil", "chickpea", "kidney_bean")
)


def _comparison_of(*records: SimpleNamespace) -> Any:
    """Return the comparison styles of the runs, roles and materials of ``records``, as a session resolves its frames'."""
    return analysis.evaluation.plots.series.comparison_styles(
        (
            record.identity["model_kind"],
            record.identity["dataset_role"],
            record.identity["material_family"],
            record.identity["checkpoint_identity"]["best_checkpoint_sha256"],
        )
        for record in records
    )


def _table_styles(rows: pd.DataFrame) -> Any:
    """Return the comparison styles of every run, role and material of hand-made summary rows, as a session resolves its frames'."""
    return analysis.evaluation.plots.series.comparison_styles(
        rows[["model_kind", "dataset_role", "material_family", "checkpoint_sha256"]].itertuples(index=False, name=None)
    )


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


def test_a_producer_mismatch_of_scoped_roots_names_only_deleting_them_and_canonical_roots_keep_the_rebuild() -> None:
    """A scoped (bounded) root takes no --rebuild, so two of different analysis code are to be deleted and generated again."""
    records = (sequence_record("autonomous_full", "full"),)
    for scope in ("scoped_partial", "canonical"):
        frames = {name: sequence_frame(records) for name in ("A ID", "B ID")}
        frames["B ID"].attrs["artifact_provenance"]["producer"] = {"analysis_sha256": "0" * 64}
        for frame in frames.values():
            frame.attrs["artifact_scope"] = scope  # as the loader sets it on every admitted role
        with pytest.raises(ValueError, match="different analysis implementations") as refused:
            session.TransientEvaluationSession(frames)
        assert ("--rebuild" in str(refused.value)) is (scope == "canonical"), scope
        assert ("delete" in str(refused.value)) is (scope == "scoped_partial"), scope


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


def test_relative_norms_below_the_reference_floor_stay_unavailable_and_counted_per_time_and_in_the_overview(tmp_path: Path) -> None:
    """
    A case whose reference lies below the floor is unavailable, never zero, at every time and in the overview rows.

    Lentil case a keeps a reference of 300 in every field, so its relative L2 is 3 / 300; chickpea case b has a zero
    reference, so its material has no supported case and the two-material value is unavailable at each time. That gap holds
    no non-finite prediction, so the view marks no x.
    """
    supported = sequence_record("autonomous_full", "full", case_id="a", offset=303.0, material_family="lentil")
    supported.reference_states = supported.reference_states + 300.0
    unsupported = sequence_record("autonomous_full", "full", case_id="b", offset=1.0, material_family="chickpea")
    evaluation = session.TransientEvaluationSession({"id": production_frame((supported, unsupported), artifact_root=str(tmp_path / "artifact"))})

    curves = evaluation.relative_norm_curve("id")
    curve = curves.query("metric_id == 'physical_relative_l2_T'")
    assert list(curve["physical_time"]) == [1.0, 2.0]
    assert (list(curve["cases"]), list(curve["supported_cases"])) == ([2, 2], [1, 1])
    assert curve["value"].isna().all()
    assert (curves["nonfinite_cases"] == 0).all()
    figure = analysis.evaluation.plots.transient.plot_relative_norms_over_time({"id": curves}, styles=evaluation.series_styles())
    assert not [axis for axis in figure.axes if _crosses(axis)]
    plt.close(figure)
    rows = evaluation.dataset_dataframe(modes=("autonomous_full",)).query("scope == 'cumulative'").set_index("material_family")
    assert rows.loc["lentil", "physical_relative_l2_T"] == pytest.approx(0.01)
    assert pd.isna(rows.loc["chickpea", "physical_relative_l2_T"])
    counts = rows["physical_relative_l2_T_unavailable_cases"]
    assert (counts["lentil"], counts["chickpea"]) == (0, 1)


def test_relative_norm_curve_voids_only_the_field_and_time_a_case_predicts_non_finite(tmp_path: Path) -> None:
    """
    A non-finite predicted field voids that field's value at that time, counted; earlier times and the other fields still count.

    Lentil cases a and b keep a reference of 300 with an offset of 3, so each relative L2 is 0.01; b predicts T non-finite at
    time 2 only, so T at time 2 is unavailable in L2 and H1 with one non-finite case, while phi there keeps both cases. The
    view marks that time with an x in both norm panels of T, and nowhere else.
    """
    finite = sequence_record("autonomous_full", "full", case_id="a", offset=303.0)
    diverging = sequence_record("autonomous_full", "full", case_id="b", offset=303.0)
    for record in (finite, diverging):
        record.reference_states = record.reference_states + 300.0
    diverging.predicted_states[2, 0] = np.nan
    evaluation = session.TransientEvaluationSession({"id": production_frame((finite, diverging), artifact_root=str(tmp_path / "artifact"))})

    curve = evaluation.relative_norm_curve("id")
    temperature, humidity = (curve.query(f"metric_id == 'physical_relative_l2_{field}'") for field in ("T", "phi"))
    assert (list(temperature["cases"]), list(temperature["supported_cases"]), list(temperature["nonfinite_cases"])) == ([2, 2], [2, 1], [0, 1])
    np.testing.assert_allclose(temperature["value"], [0.01, np.nan])
    np.testing.assert_allclose(curve.query("metric_id == 'physical_relative_h1_T'")["value"], [0.01, np.nan])
    assert (list(humidity["supported_cases"]), list(humidity["nonfinite_cases"])) == ([2, 2], [0, 0])
    np.testing.assert_allclose(humidity["value"], [0.01, 0.01])
    figure = analysis.evaluation.plots.transient.plot_relative_norms_over_time({"id": curve}, styles=evaluation.series_styles())
    fields = len(analysis.evaluation.transient_artifact.STATE_ORDER)
    marked = {index: [np.asarray(line.get_xdata(), dtype=float).tolist() for line in _crosses(axis)] for index, axis in enumerate(figure.axes)}
    assert {index: marks for index, marks in marked.items() if marks} == {0: [[2.0]], 2 * fields: [[2.0]]}  # T leads the L2 and the H1 row
    plt.close(figure)


def test_relative_norm_curves_of_one_dataset_choice_hold_its_cases_and_their_members() -> None:
    """``case_ids`` restricts a curve to one material's cases; its members name that material for the style lookup."""
    lentil = sequence_record("autonomous_full", "full", case_id="a", offset=303.0, material_family="lentil")
    chickpea = sequence_record("autonomous_full", "full", case_id="b", offset=306.0, material_family="chickpea")
    for record in (lentil, chickpea):
        record.reference_states = record.reference_states + 300.0
    evaluation = session.TransientEvaluationSession({"id": sequence_frame((lentil, chickpea))})

    pooled, alone = evaluation.relative_norm_curve("id"), evaluation.relative_norm_curve("id", case_ids=("b",))
    assert set(alone["cases"]) == {1}
    assert [member[2] for member in alone.attrs["members"]] == ["chickpea"]
    assert [member[2] for member in pooled.attrs["members"]] == ["chickpea", "lentil"]
    np.testing.assert_allclose(alone.query("metric_id == 'physical_relative_l2_T'")["value"], [0.02, 0.02])


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
    # The row says once why it has no values; its metric cells stay empty instead of repeating the reason.
    assert set(cells["availability"]) == {"unavailable (no cases at this horizon; 1 missing)"}
    assert cells["normalized_drying_group_macro_rmse"].isna().all()


def test_zero_case_rows_are_left_out_of_the_horizon_plot_and_named_in_the_summary_table(tmp_path: Path) -> None:
    """A material without records at a horizon draws no point there and its table cell names the missing cases."""
    reached = sequence_record("rolling_origin", 4, case_id="lentil_long", offset=2.0, material_family="lentil")
    short = sequence_record("rolling_origin", 2, case_id="chickpea_short", offset=2.0, material_family="chickpea")
    frame = production_frame((reached, short), artifact_root=str(tmp_path / "artifact"))
    frame.attrs["transient_unavailable_horizons"] = ({"case_id": "chickpea_short", "requested_horizon": 4, "reason": "short"},)
    evaluation = session.TransientEvaluationSession({"id": frame})

    summary, styles = evaluation.dataset_dataframe(modes=("rolling_origin",)), evaluation.series_styles()
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=styles)
    table = analysis.evaluation.panel._transient_summary_table(evaluation)
    cells = table.query("material_family == 'chickpea' and requested_horizon == 4 and scope == 'cumulative'")
    endpoint = analysis.evaluation.plots.transient.plot_endpoint_vs_cumulative(summary, styles=styles)
    plt.close(figure)
    plt.close(endpoint)

    assert cells["availability"].tolist() == ["unavailable (no cases at this horizon; 1 missing)"]
    lentil = table.query("material_family == 'lentil' and requested_horizon == 2 and scope == 'cumulative'")
    assert lentil["availability"].tolist() == ["unavailable (no cases at this horizon)"]
    assert set(table.loc[table["contributing_record_count"] > 0, "availability"]) == {""}  # a row with values names no gap


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

    outer = analysis.evaluation.panel.build_transient_panel(session=evaluation, training_histories=_stage_histories)

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
    error_time = results["4-2. Error over drying phase and hours"]
    assert isinstance(sample, analysis.ui.panel.ViewPanel)
    assert isinstance(error_time, analysis.ui.panel.ViewPanel)
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(sample.widget))
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(error_time.widget))
    assert not [record.getMessage() for record in caplog.records if "View failed" in record.getMessage()]
    evaluation.close()
    plt.close("all")


def _stage_history() -> analysis.evaluation.run_discovery.StageHistory:
    """Return a two-epoch A0 history validated at the end of each epoch."""
    return analysis.evaluation.run_discovery.StageHistory(
        run_name="a0",
        stage="A0",
        model_family="fno",
        pipeline="a0",
        start_epoch=0,
        objective_id="rmse",
        objective_direction="minimize",
        selected_point=2,
        epochs=({"epoch": 1.0, "global_step": 10.0, "train/loss_total": 1.0}, {"epoch": 2.0, "global_step": 20.0, "train/loss_total": 0.5}),
        validations=(
            {"epoch": 1.0, "global_step": 10.0, "validation/point": 1.0, "validation/eligible": 1.0, "id/rmse": 0.4},
            {"epoch": 2.0, "global_step": 20.0, "validation/point": 2.0, "validation/eligible": 1.0, "id/rmse": 0.2},
        ),
    )


def _stage_histories() -> dict[str, Any]:
    """Return the histories of one loaded A0 stage."""
    return {"A0": _stage_history()}


def test_the_runtime_and_matched_stage_sections_offer_break_even_and_lazy_training_curves(tmp_path: Path) -> None:
    """The runtime section costs the run against COMSOL, training curves load their histories only when opened, and tables are styled."""
    frame = production_frame((sequence_record("autonomous_full", "full"),), artifact_root=str(tmp_path / "artifact"))
    summary = frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR]
    summary.update(model_kind="fno", elapsed_seconds=7200.0, training_units={"train_cases": 4}, terminal_controller={"arm": "A0"})
    evaluation = session.TransientEvaluationSession({"id": frame})
    loads: list[str] = []

    def histories() -> dict[str, Any]:
        loads.append("read")
        return {"A0": _stage_history()}

    registry = evaluation_panel._build_transient_sections(
        session=evaluation,
        comparison=False,
        selection_state=None,
        evaluation_compute_device="cpu",
        training_histories=histories,
    )
    entries = {section: {title: factory for title, factory, _stem in section_entries} for section, (section_entries, _name) in registry.items()}

    assert "Break-even" in entries["runtime"]
    assert "Training curves" in entries["matched_stages"]
    assert loads == []
    try:
        curves = entries["matched_stages"]["Training curves"]()
        assert loads == ["read"]
        assert len([axis for axis in curves.axes if axis.lines]) == 2
        assert len([axis for axis in entries["runtime"]["Break-even"]()[0].axes if axis.lines]) == 1
        assert isinstance(entries["overview"]["Summary table"](), widgets.VBox)
    finally:
        plt.close("all")
        evaluation.close()


def test_training_curves_take_each_runs_session_colour_also_of_a_single_run(tmp_path: Path) -> None:
    """Every run takes its run's session colour, as its break-even does, also alone (not the pooled role's); the line style stays the stage's."""
    evaluation = _staged_session(tmp_path, (("e" * 64, 3600.0),))  # frames «A0 0» and «B», two checkpoints of one architecture
    histories = {"A0": _stage_history(), "B": replace(_stage_history(), stage="B", start_epoch=2)}
    runs = {"A0 0": evaluation_panel.FrameRun("A0", named=True), "B": evaluation_panel.FrameRun("B", named=True)}
    record = sequence_record("autonomous_full", "full")
    alone = session.TransientEvaluationSession({"FNO · ID": production_frame((record,), artifact_root=str(tmp_path))})
    try:
        styled = evaluation_panel._transient_training_curves(evaluation, lambda: histories, runs)
        colours = [line.get_color() for line in styled.axes[0].get_lines()]
        session_styles = evaluation.series_styles()
        assert colours == [session_styles["A0 0"].color, session_styles["B"].color]
        assert colours[0] != colours[1]
        assert [line.get_linestyle() for line in styled.axes[0].get_lines()] == ["-", "--"]
        single_run = {"FNO · ID": evaluation_panel.FrameRun("FNO", named=True)}
        single = evaluation_panel._transient_training_curves(alone, lambda: {"FNO": histories["A0"]}, single_run)
        # The single run takes its material's colour, as in its break-even.
        assert to_hex(single.axes[0].get_lines()[0].get_color()) == to_hex(alone.series_styles()["FNO · ID"].color)
    finally:
        plt.close("all")
        evaluation.close()
        alone.close()


def test_the_duration_ratios_and_training_curves_end_their_titles_with_the_explicit_run_labels(tmp_path: Path) -> None:
    """The ratios name the run of the selected frame, the training curves the explicit labels of the histories they draw («A0 vs B»)."""
    evaluation = _staged_session(tmp_path, (("e" * 64, 3600.0),))  # frames «A0 0» and «B», two runs
    histories = {"A0": _stage_history(), "B": replace(_stage_history(), stage="B", start_epoch=2)}
    registry = evaluation_panel._build_transient_sections(
        session=evaluation,
        comparison=True,
        selection_state=None,
        evaluation_compute_device="cpu",
        training_histories=lambda: histories,
        frame_runs={"A0 0": evaluation_panel.FrameRun("A0", named=True), "B": evaluation_panel.FrameRun("B", named=True)},
    )
    entries = {title: factory for section_entries, _name in registry.values() for title, factory, _stem in section_entries}
    ratios: dict[str, Any] = {}
    entries["Boundary-specific duration ratios"](export_state=ratios)  # its model selector starts on the first frame, «A0 0»
    try:
        assert _figure_title(ratios["fig"]).endswith(" (A0)")
        assert _figure_title(entries["Training curves"]()).endswith(" (A0 vs B)")
    finally:
        plt.close("all")
        evaluation.close()


def test_matched_compute_markers_take_each_runs_session_colour_only_with_several_runs(tmp_path: Path) -> None:
    """With three runs each arm's marker takes its run's session colour over the material line; A0 alone keeps its material markers."""
    comparison = analysis.evaluation.transient_comparison
    evaluation = _staged_session(tmp_path, (("e" * 64, 3600.0), ("f" * 64, 3600.0)))  # frames «A0 0», «A0 1» and «B»: three runs
    arms = {
        name: comparison.TrainingArmEvidence(name, "p", "a", "d", "s", "i", "c", "b", "rollout", spent, spent, 0.0)
        for name, spent in (("A0", 0.0), ("A1", 8.0), ("B", 8.0))
    }
    lineage = comparison.LineageComparison(a0=arms["A0"], a1=arms["A1"], b=arms["B"], primary_comparison="B_vs_A1", separate_comparison="B_vs_A0")
    rows = pd.DataFrame(
        {
            "frame": ["A0 0", "A0 1", "B"],
            "comparison_arm": ["A0", "A1", "B"],
            "optimizer_device_compute": [0.0, 8.0, 8.0],
            "training_compute": [0.0, 8.0, 8.0],
            "normalized_drying_group_macro_rmse": [0.3, 0.2, 0.1],
            "material_family": "lentil",
            "compute_clock": "supervised_transitions",
            "checkpoint_role": "best",
            "mode": "autonomous_full",
            "requested_horizon": "full",
            "scope": "cumulative",
        }
    )
    registry = evaluation_panel._build_transient_sections(
        session=evaluation,
        comparison=True,
        selection_state=None,
        evaluation_compute_device="cpu",
        training_histories=dict,
        training_performance=lambda: rows,
        training_lineage=lineage,
    )
    matched = {title: factory for title, factory, _stem in registry["matched_stages"][0]}["Shared-A0 performance vs committed budget"]
    styles = evaluation.series_styles()
    unavailable = {arm.lower(): comparison.UnavailableArm(arm, "no_eligible_checkpoint", 1.0, 0, "r" * 64, "h" * 64) for arm in ("A1", "B")}
    try:
        markers = [to_hex(colour) for collection in matched().axes[0].collections for colour in collection.get_facecolors()]
        assert markers == [to_hex(styles[frame].color) for frame in ("A0 0", "A0 1", "B")]
        assert len(set(markers)) == len(markers)
        transient = analysis.evaluation.plots.transient
        alone = transient.plot_training_performance_vs_compute(rows.iloc[:1], lineage=replace(lineage, **unavailable), styles=styles)
        lentil = analysis.presentation.visual_semantics.material_color("lentil")
        # one run: its point keeps the material colour, and the note carries the arms without a result
        faces = [to_hex(cast("ColorType", colour)) for collection in alone.axes[0].collections for colour in collection.get_facecolor()]
        assert faces == [to_hex(lentil)]
        assert "A1: No eligible checkpoint at 100 % of its budget, 0 eligible points, no result" in _figure_note(alone)
    finally:
        plt.close("all")
        evaluation.close()


_HANDOFF_CHECKPOINT = "d" * 64  # the selected checkpoint of the A0 a B stage starts from: the support frame's own


def _staged_session(
    tmp_path: Path, a0_stages: tuple[tuple[str, float | None], ...], *, b_seconds: float | None = 3600.0
) -> session.TransientEvaluationSession:
    """Open a session of a B stage (1 h) from the A0 of checkpoint ``_HANDOFF_CHECKPOINT`` and the given A0 stages (checkpoint, seconds)."""
    frames = {}
    for name, arm, checkpoint, seconds in (
        *((f"A0 {index}", "A0", sha, elapsed) for index, (sha, elapsed) in enumerate(a0_stages)),
        ("B", "B", _HANDOFF_CHECKPOINT, b_seconds),
    ):
        frame = production_frame((sequence_record("autonomous_full", "full"),), artifact_root=str(tmp_path / name))
        frame.attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
        summary = frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR]
        summary.update(
            run_name="pipeline",
            ended_at=f"2026-11-11T0{len(frames)}:00:00+00:00",
            model_kind="fno",
            elapsed_seconds=seconds,
            training_units={"train_cases": 4},
            terminal_controller={"arm": arm},
        )
        if arm == "B":
            summary["teacher_handoff"] = {"source_run_name": "pipeline", "best_epoch": 3, "files": {"checkpoint": {"sha256": _HANDOFF_CHECKPOINT}}}
        frames[name] = frame
    return session.TransientEvaluationSession(frames)


def _break_even_entry(evaluation: session.TransientEvaluationSession) -> Any:
    """Return the factory of the transient runtime section's Break-even entry."""
    registry = evaluation_panel._build_transient_sections(
        session=evaluation, comparison=True, selection_state=None, evaluation_compute_device="cpu", training_histories=_stage_histories
    )
    return {title: factory for title, factory, _stem in registry["runtime"][0]}["Break-even"]


def test_a_continuation_stage_costs_its_a0_seconds_and_its_own(tmp_path: Path) -> None:
    """
    B's training cost is A0's 2 h plus its own 1 h; the A0 is the one whose selected checkpoint the handoff records, not another of its name.

    The clocks and hardware stand as plain text below the figure, never inside it, and the seconds keep three significant digits.
    """
    evaluation = _staged_session(tmp_path, (("e" * 64, 36000.0), (_HANDOFF_CHECKPOINT, 7200.0)))
    figure, details = _break_even_entry(evaluation)()
    try:
        texts = "\n".join(text.get_text() for text in figure.artists if hasattr(text, "get_text"))
        assert "B · ID: " in texts  # the note names the costed role
        assert "$C_\\mathrm{train}$ = 3.00\N{NO-BREAK SPACE}h" in texts
        assert "$C_\\mathrm{train}$ = 11" not in texts  # the same-named A0 of another checkpoint
        assert "B: COMSOL transient drying study time on solver-node (8 cores)" in details
        assert "A0 + stage" in details
        assert "solver-node" not in "\n".join(text.get_text() for text in figure.findobj(Text))
        assert not figure.get_supxlabel()
    finally:
        plt.close("all")
        evaluation.close()


@pytest.mark.parametrize(
    ("a0_stages", "b_seconds", "named", "unnamed"),
    [
        ((("e" * 64, 7200.0),), 3600.0, "B (no selected A0 has its handoff checkpoint)", "no training seconds"),
        (
            ((_HANDOFF_CHECKPOINT, 7200.0), (_HANDOFF_CHECKPOINT, 9000.0)),
            3600.0,
            "B (2 selected A0 runs have its handoff checkpoint, not one)",
            "no training seconds",
        ),
        (((_HANDOFF_CHECKPOINT, 7200.0),), None, "B (no training seconds)", "A0"),
        (((_HANDOFF_CHECKPOINT, None),), 3600.0, "B (no training seconds of its A0)", "no training seconds;"),
    ],
    ids=["only another A0", "two A0 of the handoff checkpoint", "no seconds of its own", "no seconds of its matched A0"],
)
def test_a_continuation_stage_without_its_evidence_names_what_is_missing(
    tmp_path: Path, a0_stages: tuple[tuple[str, float | None], ...], b_seconds: float | None, named: str, unnamed: str
) -> None:
    """B is never costed from its own seconds or another A0's; a missing A0 and missing seconds of its own are named apart."""
    evaluation = _staged_session(tmp_path, a0_stages, b_seconds=b_seconds)

    shown = _break_even_entry(evaluation)()
    try:
        text = shown.value if isinstance(shown, widgets.HTML) else shown[0].get_supxlabel()  # the matched A0 without seconds draws nothing
        (without,) = re.findall(r"\bB \([^)]*\)", text)
        assert without == named
        assert unnamed not in without
    finally:
        plt.close("all")
        evaluation.close()


def test_a_continuation_stage_without_any_selected_a0_says_so(tmp_path: Path) -> None:
    """B alone: no A0 is selected at all, which differs from A0 runs that are selected but none of its handoff checkpoint."""
    evaluation = _staged_session(tmp_path, ())

    shown = _break_even_entry(evaluation)()
    try:
        assert isinstance(shown, widgets.HTML)  # the only run lacks its evidence, so a message stands in place of the figure
        assert "B (no A0 is selected: select the A0 of its handoff to cost the stage)" in shown.value
    finally:
        evaluation.close()


def test_the_break_even_colours_each_run_as_the_sessions_other_views_do(tmp_path: Path) -> None:
    """Three runs of one architecture take run colours: each drawn cost takes its frame's session colour, never a palette of its own."""
    evaluation = _staged_session(tmp_path, (("e" * 64, 3600.0), ("f" * 64, 3600.0)))  # B's A0 is not loaded: two runs draw
    figure, _details = _break_even_entry(evaluation)()
    try:
        styles = evaluation.series_styles()
        drawn = {axis.get_title(): to_hex(axis.lines[1].get_color()) for axis in figure.axes if axis.lines}
        assert drawn == {frame: to_hex(styles[frame].color) for frame in ("A0 0", "A0 1")}
    finally:
        plt.close("all")
        evaluation.close()


def test_a_break_even_passes_each_available_clock_so_the_unpaired_cases_are_counted(tmp_path: Path) -> None:
    """A run with one paired case and two cases that only have the surrogate clock reports two unpaired cases."""
    frame = production_frame((sequence_record("autonomous_full", "full"),), artifact_root=str(tmp_path / "artifact"))
    timing = analysis.evaluation.transient_timing

    def timed(case_id: str, repetitions: dict[str, tuple[float, ...]]) -> Any:
        return timing.TransientTimingCase(
            case_id=case_id,
            repetitions=repetitions,
            device="cpu",
            precision="float32",
            pt_payload_identity="a" * 64,
            warmup_passes=1,
            solver_hostname="solver-node",
            solver_requested_cores=8,
            cpu="test-cpu",
            software_versions={"python": "test", "pytorch": "test", "numpy": "test"},
        )

    surrogate_only = {"drying_no_rollout_model_seconds": (0.01, 0.02)}
    paired = timed("first", {**surrogate_only, "comsol_transient_drying_seconds": (0.3,)})
    report = timing.build_transient_timing_report((paired, timed("second", surrogate_only), timed("third", surrogate_only)))
    frame.attrs["artifact_provenance"]["timing"]["report"] = json.loads(json.dumps(asdict(report)))
    frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR].update(
        model_kind="fno", elapsed_seconds=7200.0, training_units={"train_cases": 4}, terminal_controller={"arm": "A0"}
    )
    evaluation = session.TransientEvaluationSession({"id": frame})
    figure, _details = _break_even_entry(evaluation)()
    try:
        note = " ".join(" ".join(text.get_text() for text in figure.artists if hasattr(text, "get_text")).split())  # a long note wraps
        assert "over 1 paired cases (2 unpaired)" in note
    finally:
        plt.close("all")
        evaluation.close()


def test_the_pooled_time_views_default_to_the_drying_phase_and_pass_the_clock_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over two cases the trajectory, error and relative-norm views default to the phase and pass the switch on; the magnitude view has none."""
    records = tuple(sequence_record("autonomous_full", "full", case_id=name, offset=offset) for name, offset in (("low", 1.0), ("high", 3.0)))
    evaluation = session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path / "artifact"))})
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    seen: list[dict[str, Any]] = []

    def record_call(series: dict[str, Any], **kwargs: Any) -> Figure:
        frame = next(iter(series.values()))  # the relative norms pass curves, whose clock column names their time axis
        seen.append({"time_axis": "phase" if "phase" in frame else "hours"} if isinstance(frame, pd.DataFrame) else kwargs)
        return plt.figure()

    for name in ("plot_error_over_physical_time", "plot_state_trajectory_summary", "plot_relative_norms_over_time"):
        monkeypatch.setattr(analysis.evaluation.plots.transient, name, record_call)

    def switch(plot_kind: str) -> widgets.ToggleButtons | None:
        if plot_kind == "relative_norms":
            viewer = analysis.evaluation.panel._transient_relative_norm_entry(evaluation, comparison=False, selection_state=None)()
        else:
            viewer = analysis.evaluation.panel._make_transient_scope_viewer(session=evaluation, fields=fields, plot_kind=plot_kind, comparison=False)
        found = [item for item in _widget_tree(viewer.widget) if isinstance(item, widgets.ToggleButtons) and item.value == "phase"]
        return found[0] if found else None

    try:
        assert switch("target_magnitude") is None
        for plot_kind in ("trajectory", "error_time", "relative_norms"):
            control = switch(plot_kind)
            assert control is not None
            assert seen[-1]["time_axis"] == "phase"
            control.value = "hours"
            assert seen[-1]["time_axis"] == "hours"
    finally:
        plt.close("all")
        evaluation.close()


def test_a_pooled_time_view_with_one_case_per_curve_defaults_to_hours_and_keeps_a_picked_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trajectory and error views show hours while every curve holds one case, else the phase, until the user picks a clock."""
    evaluation = _transient_evaluation(tmp_path, *(sequence_record("autonomous_full", "full", case_id=name) for name in ("low", "high")))
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    seen: list[str] = []
    for name in ("plot_error_over_physical_time", "plot_state_trajectory_summary"):
        monkeypatch.setattr(analysis.evaluation.plots.transient, name, lambda _series, **kwargs: seen.append(kwargs["time_axis"]) or plt.figure())
    for plot_kind in ("trajectory", "error_time"):
        view = analysis.evaluation.panel._make_transient_scope_viewer(session=evaluation, fields=fields, plot_kind=plot_kind, comparison=False)
        tree = _widget_tree(view.widget)
        cases = next(widget for widget in tree if isinstance(widget, widgets.IntSlider) and widget.description == "Cases")
        clock = next(widget for widget in tree if isinstance(widget, widgets.ToggleButtons) and widget.value in {"phase", "hours"})
        assert seen[-1] == clock.value == "phase"
        cases.value = 1
        assert seen[-1] == clock.value == "hours"
        clock.value = "phase"  # the user's pick outlasts every later cohort
        cases.value = 2
        cases.value = 1
        assert seen[-1] == clock.value == "phase"
    plt.close("all")
    evaluation.close()


def test_the_relative_norms_of_one_case_frames_open_on_hours_and_draw_a_picked_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each curve pools one case, so the view opens on hours like the trajectory and error views; the phase the user picks is drawn."""
    evaluation = _transient_evaluation(tmp_path, sequence_record("autonomous_full", "full", case_id="one"))
    seen: list[str] = []
    monkeypatch.setattr(
        analysis.evaluation.plots.transient,
        "plot_relative_norms_over_time",
        lambda curves, **_kwargs: seen.append("phase" if "phase" in next(iter(curves.values())) else "hours") or plt.figure(),
    )
    view = analysis.evaluation.panel._transient_relative_norm_entry(evaluation, comparison=False, selection_state=None)()
    clock = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.ToggleButtons) and widget.value in {"phase", "hours"})
    assert seen == ["hours"]
    assert clock.value == "hours"
    clock.value = "phase"
    assert seen[-1] == clock.value == "phase"
    plt.close("all")
    evaluation.close()


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
        tabs, views = _tab_layout(
            analysis.evaluation.panel.build_transient_panel(session=evaluation, comparison=is_comparison, training_histories=_stage_histories)
        )
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
    tabs, _views = _tab_layout(analysis.evaluation.panel.build_transient_panel(session=evaluation, training_histories=_stage_histories))
    assert not any(titles["temporal_rollout"] in title for title in tabs)
    assert [title.partition(".")[0] for title in tabs] == [str(number) for number in range(1, len(tabs) + 1)]
    sample_tabs, sample_views = _tab_layout(
        analysis.evaluation.panel.build_transient_panel(session=evaluation, sections=("sample_viewer",), training_histories=_stage_histories)
    )
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
    panel = analysis.evaluation.panel.build_transient_panel(
        session=evaluation, evaluation_compute_device=preference, training_histories=_stage_histories
    )
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


def _figure_title(figure: Figure) -> str:
    """Return the title ``figure_layout.set_figure_title`` wrote: the first text artist of the figure, before its note."""
    return next(artist.get_text() for artist in figure.artists if isinstance(artist, Text))


def _figure_note(figure: Figure) -> str:
    """Return the note ``figure_layout.set_figure_title`` wrote under the title, whitespace collapsed (a long note wraps)."""
    texts = [artist.get_text() for artist in figure.artists if isinstance(artist, Text)]
    return " ".join(texts[1].split()) if len(texts) > 1 else ""


def _trend_count(field: str, count: int, kind: str = "non-finite") -> str:
    """Return the note part of the error trend that names the ``count`` of 72 values of the series «case» left out as ``kind``."""
    return f"case: {analysis.evaluation.plots.transient._display_symbol(field)} {count} {kind} of 72"


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
    assert tuple(axis.get_title() for axis in faceted.axes[:2]) == (  # each header counts its own cases
        "Chickpea · ID · n(t) = 1",
        "Kidney bean · OOD · n(t) = 1",
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
    assert "__channel_phi__" in str(export_state["filename_stem"])
    assert "t_1h" in str(export_state["filename_stem"])
    evaluation.close()


def test_only_the_mean_error_maps_offer_the_error_mode_and_pass_it_with_the_scales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mean maps carry the transient error-mode control, its value reaches the builder, and the normalized mode adds each context's scales."""
    records = (sequence_record("autonomous_full", "full", case_id="low", offset=1.0),)
    evaluation = session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path / "artifact"))})
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    seen: list[dict[str, Any]] = []

    def record_call(series: dict[str, Any], **kwargs: Any) -> Figure:
        seen.append({**kwargs, "shown": set(series)})
        return plt.figure()

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_aggregate_error_maps", record_call)
    monkeypatch.setattr(
        analysis.evaluation.panel, "_transient_reference_scales", lambda _session, _frame, names, _record: dict.fromkeys(names, "scale")
    )

    def mode_control(plot_kind: str) -> tuple[Any, widgets.ToggleButtons | None]:
        viewer = analysis.evaluation.panel._make_transient_aggregate_viewer(session=evaluation, fields=fields, plot_kind=plot_kind, comparison=False)
        found = [item for item in _widget_tree(viewer.widget) if isinstance(item, widgets.ToggleButtons) and item.description == "Error mode"]
        return viewer, found[0] if found else None

    assert mode_control("std_error")[1] is None
    viewer, control = mode_control("mean_error")
    assert control is not None
    assert seen[-1]["error_mode"] == "Signed error"
    assert "reference_scales" not in seen[-1]
    control.value = error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    assert seen[-1]["error_mode"] == control.value
    assert seen[-1]["shown"]
    assert set(seen[-1]["reference_scales"]) == seen[-1]["shown"]
    assert viewer is not None
    plt.close("all")
    evaluation.close()


def test_normalized_mean_maps_take_each_series_scale_from_its_own_frame_and_refuse_a_foreign_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each «frame · dataset» series takes its scales from its own frame, once per render; a scaler other than the records' refuses."""
    frames = {
        name: production_frame((sequence_record("autonomous_full", "full", case_id="low", offset=1.0),), artifact_root=str(tmp_path / name))
        for name in ("first", "second")
    }
    evaluation = session.TransientEvaluationSession(frames)
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    seen: list[dict[str, Any]] = []
    reads: list[str] = []

    def record_call(_series: dict[str, Any], **kwargs: Any) -> Figure:
        seen.append(kwargs)
        return plt.figure()

    def scales_of(_session: object, frame: str, names: tuple[str, ...]) -> dict[str, Any]:
        reads.append(frame)
        value, scaler = (1.0, "c" * 64) if frame == "first" else (2.0, "e" * 64)  # the records carry the scaler identity "c" * 64
        return {name: error_diagnostics.ReferenceScale(name, "K", value, "s", "train", "d", scaler) for name in names}

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_aggregate_error_maps", record_call)
    monkeypatch.setattr(analysis.evaluation.panel, "_frame_reference_scales", scales_of)
    viewer = analysis.evaluation.panel._make_transient_aggregate_viewer(session=evaluation, fields=fields, plot_kind="mean_error", comparison=True)
    (control,) = [item for item in _widget_tree(viewer.widget) if isinstance(item, widgets.ToggleButtons) and item.description == "Error mode"]
    reads.clear()
    control.value = error_diagnostics.NORMALIZED_ABSOLUTE_ERROR

    by_frame = {label.split(" · ", 1)[0]: scales for label, scales in seen[-1]["reference_scales"].items()}
    assert set(by_frame) == {"first", "second"}
    assert {scale.value for scale in by_frame["first"].values()} == {1.0}
    assert all(isinstance(scale, str) for scale in by_frame["second"].values())
    assert sorted(reads) == ["first", "second"]  # once per frame and render
    plt.close("all")
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

    trajectory = analysis.evaluation.plots.transient.plot_state_trajectory_summary({"ID": (long, short)}, state_fields=("T",), styles=_STYLES)
    trajectory_times = np.asarray(trajectory.axes[0].lines[0].get_xdata())
    assert trajectory_times.size == 3
    np.testing.assert_array_equal(trajectory.axes[-1].lines[0].get_ydata(), [2, 2, 1])  # the case-count row is the last
    plt.close(trajectory)
    error = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": (long, short)}, scaling_states={"ID": scaling_state()}, state_fields=("T",), styles=_STYLES
    )
    error_times = np.asarray(error.axes[0].lines[0].get_xdata())
    assert error_times.size == 2
    np.testing.assert_array_equal(error.axes[-1].lines[0].get_ydata(), [2, 1])
    plt.close(error)

    empty_context = analysis.evaluation.plots.transient.plot_aggregate_error_maps(
        {"ID": (long,), "OOD": ()}, state_fields=("T",), physical_time=2.0, statistic="mean"
    )
    assert any("No case at this exact time" in text.get_text() for text in empty_context.axes[1].texts)
    plt.close(empty_context)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every figure a test of this module opened, passed or failed."""
    yield
    plt.close("all")


def _timed_record(case_id: str, steps: int, *, target: float = 0.12, stop_after_grid: float = 0.0) -> SimpleNamespace:
    """
    Return one full-autonomous record on a regular two-hour grid (``2 * (steps - 1)`` hours long) with its wet-basis target moisture.

    Its recorded final reference time (the exact stop) lies ``stop_after_grid`` hours after the last grid time.
    """
    record = sequence_record("autonomous_full", "full", case_id=case_id, length=steps, target_moisture_wb=target)
    record.physical_times = np.arange(steps, dtype=np.float64) * 2.0
    record.target["reference_final_time"] = float(record.physical_times[-1]) + stop_after_grid
    return record


def _water_nan(record: SimpleNamespace, index: int) -> None:
    """Make the predicted water of one stored time non-finite everywhere, so the grain moisture (not T) is unavailable there."""
    states = analysis.evaluation.transient_artifact.STATE_ORDER
    record.predicted_states[index, [states.index("w_surf"), states.index("w_int")]] = np.nan


def _crosses(axis: Any) -> list[Any]:
    """Return the lines of one axis that mark non-finite times with an x."""
    return [line for line in axis.get_lines() if line.get_marker() == "x"]


def test_mean_and_error_over_time_show_one_window_for_a_short_run_and_three_for_a_long_one() -> None:
    """Below 72 h the whole run is one window; a long run shows its first 24 h, the middle in days and its last 24 h."""
    transient = analysis.evaluation.plots.transient
    for steps, titles in ((20, ["whole run, 38 h"]), (60, ["first 24 h", "day 1 to 3.9", "last 24 h"])):
        series = {"ID": (_timed_record("a", steps), _timed_record("b", steps))}
        mean = transient.plot_state_trajectory_summary(series, state_fields=("T",), styles=_STYLES)
        error = transient.plot_error_over_physical_time(series, scaling_states={"ID": scaling_state()}, state_fields=("T",), styles=_STYLES)
        assert [axis.get_title() for axis in mean.axes if axis.get_title()] == titles
        assert [axis.get_title() for axis in error.axes if axis.get_title()] == titles
        assert (len(mean.axes), len(error.axes)) == (3 * len(titles), 2 * len(titles))  # state, difference, count rows; state, count rows


def test_the_last_24_hours_are_hours_before_each_end_only_for_cases_of_different_duration() -> None:
    """Cases ending at 118 h and 98 h share the last window as -24 to 0 h before their end; cases ending together keep the clock."""
    transient = analysis.evaluation.plots.transient

    def last_window(records: tuple[SimpleNamespace, ...]) -> tuple[np.ndarray, str, np.ndarray]:
        figure = transient.plot_state_trajectory_summary({"ID": records}, state_fields=("T",), styles=_STYLES)
        last, count = figure.axes[2], figure.axes[-1]
        reference = next(line for line in last.get_lines() if line.get_label() == "ID · Reference")
        return np.asarray(reference.get_xdata()), count.get_xlabel(), np.asarray(count.lines[0].get_ydata())

    time, label, count = last_window((_timed_record("long", 60), _timed_record("short", 50)))
    assert (time.min(), time.max(), label) == (-22.0, 0.0, "Time before end [h]")  # (-24, 0] h on a two-hour grid
    np.testing.assert_array_equal(count, 2.0)  # both cases have all of their last 24 h
    time, label, _count = last_window((_timed_record("a", 60), _timed_record("b", 60)))
    assert (time.min(), time.max(), label) == (96.0, 118.0, "Time [h]")  # (94, 118] h


def test_the_middle_window_of_cases_of_different_duration_stops_at_each_cases_end_minus_24_hours() -> None:
    """A case that ends at 98 h is in the middle window up to 74 h and in the last window from there, never in both beyond that hour."""
    transient = analysis.evaluation.plots.transient
    records = (_timed_record("long", 60), _timed_record("short", 50))
    figure = transient.plot_state_trajectory_summary({"ID": records}, state_fields=("T",), styles=_STYLES)
    clock = np.asarray(next(line for line in figure.axes[1].get_lines() if line.get_label() == "ID · Reference").get_xdata())
    assert clock.max() == 94.0  # the long case, 118 h minus 24 h
    middle_count = figure.axes[-2].lines[0]  # the case-count row: first, middle and last window
    time, count = np.asarray(middle_count.get_xdata()), np.asarray(middle_count.get_ydata())
    np.testing.assert_array_equal(count[time <= 74.0], 2.0)
    np.testing.assert_array_equal(count[time > 74.0], 1.0)  # the short case is in the last window from there on
    np.testing.assert_array_equal(figure.axes[-1].lines[0].get_ydata(), 2.0)  # both cases have all of their last 24 h


def test_cases_under_48_hours_are_in_one_window_per_stored_hour_and_the_count_is_the_true_coverage() -> None:
    """A 40 h case is in the first window to 24 h and in the last after it, a 20 h case only in the first; the long case alone fills the middle."""
    transient = analysis.evaluation.plots.transient
    records = (_timed_record("long", 60), _timed_record("mid", 21), _timed_record("short", 11))  # 118 h, 40 h and 20 h
    figure = transient.plot_state_trajectory_summary({"ID": records}, state_fields=("T",), styles=_STYLES)
    first, middle, last = ((np.asarray(axis.lines[0].get_xdata()), np.asarray(axis.lines[0].get_ydata())) for axis in figure.axes[-3:])
    assert (first[1][first[0] <= 20.0] == 3.0).all()
    assert (first[1][first[0] > 20.0] == 2.0).all()  # the 20 h case has ended; the 40 h case is still in its first window
    assert middle[0].max() == 94.0
    assert (middle[1] == 1.0).all()  # the 40 h case's middle (24 h to its end minus 24 h) is empty
    assert (last[1][last[0] < -14.0] == 1.0).all()
    assert (last[1][last[0] >= -14.0] == 2.0).all()  # the 40 h case joins the last window with its first hour after 24 h, 14 h before its end


def test_records_of_one_case_with_different_target_moistures_are_refused() -> None:
    """The target moisture is a property of the case: two records of one case that disagree are refused, two cases may differ."""
    transient = analysis.evaluation.plots.transient
    plt.close(
        transient.plot_state_trajectory_summary(
            {"ID": (_timed_record("a", 20, target=0.12), _timed_record("b", 20, target=0.135))}, state_fields=("w_gr",), styles=_STYLES
        )
    )
    disagreeing = {"ID": (_timed_record("a", 20, target=0.12),), "OOD": (_timed_record("a", 20, target=0.135),)}
    open_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match=r"case 'a' carry different target_moisture_wb"):
        transient.plot_state_trajectory_summary(disagreeing, state_fields=("w_gr",), styles=_STYLES)
    assert set(plt.get_fignums()) == open_figures


def test_the_last_window_aligns_on_each_cases_last_stored_time_so_cases_with_off_grid_stops_still_aggregate() -> None:
    """Exact stops 0.4, 0.8 and 1.2 h after the last grid times do not move the offsets: the three cases share every point of the last window."""
    transient = analysis.evaluation.plots.transient
    records = (_timed_record("a", 60, stop_after_grid=0.4), _timed_record("b", 50, stop_after_grid=0.8), _timed_record("c", 55, stop_after_grid=1.2))
    figure = transient.plot_state_trajectory_summary({"ID": records}, state_fields=("T",), styles=_STYLES)
    reference = next(line for line in figure.axes[2].get_lines() if line.get_label() == "ID · Reference")
    time = np.asarray(reference.get_xdata())
    assert (time.min(), time.max()) == (-22.0, 0.0)  # offsets before each case's last stored time
    np.testing.assert_array_equal(figure.axes[-1].lines[0].get_ydata(), 3.0)  # n(t) = 3 in the last window


def test_time_numbers_and_the_time_label_stand_under_the_bottom_row_of_every_window_only() -> None:
    """The rows above the case-count row carry neither tick numbers nor a time label, in the mean and the error plot alike."""
    transient = analysis.evaluation.plots.transient
    series = {"ID": (_timed_record("a", 60), _timed_record("b", 50))}
    mean = transient.plot_state_trajectory_summary(series, state_fields=("T", "phi"), styles=_STYLES)
    error = transient.plot_error_over_physical_time(series, scaling_states={"ID": scaling_state()}, state_fields=("T", "phi"), styles=_STYLES)
    for figure, rows in ((mean, 5), (error, 3)):  # T and phi with a difference row each and the count row; T, phi and the count row
        upper, bottom = figure.axes[: 3 * (rows - 1)], figure.axes[3 * (rows - 1) :]
        assert all(axis.get_xlabel() == "" and not axis.xaxis.get_tick_params()["labelbottom"] for axis in upper)
        assert all(axis.get_xlabel().startswith("Time") and axis.xaxis.get_tick_params()["labelbottom"] for axis in bottom)


def test_the_difference_row_is_prediction_minus_reference_in_the_difference_unit() -> None:
    """Each state has a row of prediction minus reference in its difference unit: percent becomes percentage points."""
    transient = analysis.evaluation.plots.transient
    fields = ("T", "phi", "w_gr")
    figure = transient.plot_state_trajectory_summary({"ID": (_timed_record("a", 20),)}, state_fields=fields, styles=_STYLES)
    quantity_rows, difference_rows = figure.axes[0:6:2], figure.axes[1:6:2]  # one window: state, difference, state, difference, ...
    labels = [axis.get_ylabel() for axis in difference_rows]
    assert labels == [f"Prediction \N{MINUS SIGN}\nreference [{transient._display_unit(field, difference=True)}]" for field in fields]
    assert [label.endswith("[pp]") for label in labels] == [False, True, True]
    assert [axis.get_ylabel() for axis in quantity_rows] == [transient._display_label(field) for field in fields]
    difference = np.asarray(difference_rows[0].lines[0].get_ydata(), dtype=np.float64)
    np.testing.assert_allclose(difference, [0.0] + [2.0] * 19)  # the offset of the record after its origin


def test_the_case_count_is_the_stored_times_and_a_non_finite_value_marks_only_its_own_fields_row() -> None:
    """n(t) counts the cases with a stored time and is the same for every state; an x stands on the difference row of the state it voids."""
    transient = analysis.evaluation.plots.transient
    records = (_timed_record("a", 20), _timed_record("b", 20))
    _water_nan(records[0], -1)
    figure = transient.plot_state_trajectory_summary({"ID": records}, state_fields=("T", "w_gr"), styles=_STYLES)
    temperature_difference, moisture_difference, count = figure.axes[1], figure.axes[3], figure.axes[4]  # one window: T, d, w_gr, d, n(t)
    assert _crosses(temperature_difference) == _crosses(count) == []
    (cross,) = _crosses(moisture_difference)
    assert list(cross.get_xdata()) == [38.0]  # the last stored time, where one case has no finite water
    np.testing.assert_array_equal(count.lines[0].get_ydata(), 2.0)  # both cases still have that time


def test_the_difference_row_pairs_each_case_so_a_case_without_a_finite_prediction_never_enters_it_while_the_reference_keeps_it() -> None:
    """
    Case a, 10 K above case b, predicts T non-finite at 10 h: the difference is void and marked there, never b's alone.

    Elsewhere the difference is the mean of each case's own difference (1 K and 3 K), and the flat reference curve keeps a.
    """
    transient = analysis.evaluation.plots.transient
    a, b = _timed_record("a", 20), _timed_record("b", 20)
    a.reference_states = a.reference_states + 10.0
    for record, offset in ((a, 1.0), (b, 3.0)):
        record.predicted_states = record.reference_states + offset * (record.physical_times > 0.0)[:, None, None, None]
    a.predicted_states[5, 0] = np.nan  # T of case a at 10 h
    figure = transient.plot_state_trajectory_summary({"ID": (a, b)}, state_fields=("T",), styles=_STYLES)
    quantity, difference_axis = figure.axes[0], figure.axes[1]
    difference = np.asarray(difference_axis.lines[0].get_ydata(), dtype=np.float64)
    assert np.isnan(difference[5])
    np.testing.assert_allclose(np.delete(difference, 5), [0.0] + [2.0] * 18)
    (cross,) = _crosses(difference_axis)
    assert list(cross.get_xdata()) == [10.0]
    reference = np.asarray(next(line for line in quantity.get_lines() if line.get_label() == "ID · Reference").get_ydata(), dtype=np.float64)
    np.testing.assert_allclose(reference, reference[0])  # a's reference stays in the curve at 10 h too


def test_the_one_figure_legend_stands_outside_the_right_edge_without_growing_a_row() -> None:
    """Many entries do not change any row height; the legend starts at the top frame, right of the last window, and gathers every axis."""
    transient = analysis.evaluation.plots.transient
    few = transient.plot_state_trajectory_summary({"ID": (_timed_record("a", 60),)}, state_fields=("T",), styles=_STYLES)
    many_records = {f"Context {index}": (_timed_record(f"c{index}", 60),) for index in range(15)}
    many_records["Context 0"][0].predicted_states[-1] = np.nan  # a non-finite last time, drawn in the last window only
    many = transient.plot_state_trajectory_summary(many_records, state_fields=("T",), styles=_STYLES)
    for figure in (few, many):
        figure.canvas.draw()
    few_heights = [axis.get_window_extent().height for axis in few.axes]
    assert [axis.get_window_extent().height for axis in many.axes] == pytest.approx(few_heights, abs=0.5)  # the legend grows the figure, not a row
    assert len(many.legends) == 1
    legend = many.legends[0]
    assert legend.get_window_extent().y1 == pytest.approx(many.axes[0].get_window_extent().y1, abs=1.0)
    assert legend.get_window_extent().x0 > many.axes[2].get_window_extent().x1
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == len(set(labels)) == 15 * 2 + 1  # reference and prediction of 15 contexts (n(t) is named by its strip), one non-finite entry
    assert "non-finite, unavailable" in labels  # drawn in the last window only


def test_the_grain_moisture_mean_draws_each_distinct_target_moisture_once() -> None:
    """One line per distinct wet-basis target of all shown cases, one legend entry, and each value at the right end of the first window."""
    transient = analysis.evaluation.plots.transient
    series = {
        "ID": (_timed_record("a", 60, target=0.12), _timed_record("b", 60, target=0.12), _timed_record("c", 60, target=0.135)),
        "OOD": (_timed_record("d", 60, target=0.12),),
    }
    figure = transient.plot_state_trajectory_summary(series, state_fields=("T", "w_gr"), styles=_STYLES)
    temperature, moisture = figure.axes[0:3], figure.axes[6:9]  # rows: T, difference, w_gr, difference, count; three windows

    def targets(axis: Any) -> list[float]:  # dotted in the target grey; the curves take their series colours
        return [float(np.asarray(line.get_ydata())[0]) for line in axis.get_lines() if line.get_linestyle() == ":" and line.get_color() == "0.35"]

    for axis in moisture:
        assert targets(axis) == pytest.approx([12.0, 13.5])
    assert not any(targets(axis) for axis in temperature)
    assert [text.get_text() for text in figure.legends[0].get_texts() if text.get_text().startswith("Target")] == [
        f"Target {transient._display_symbol('w_gr')}"
    ]
    assert [[text.get_text() for text in axis.texts] for axis in moisture] == [[], [], ["12 %", "13.5 %"]]
    # in a strip right of the last window, clear of every curve and band
    assert all(cast("Annotation", text).xy[0] == 1.0 and text.get_horizontalalignment() == "left" for text in moisture[2].texts)
    without_target = _timed_record("e", 20)
    without_target.target = {name: value for name, value in without_target.target.items() if name != "target_moisture_wb"}
    transient.plot_state_trajectory_summary({"ID": (without_target,)}, state_fields=("T",), styles=_STYLES)  # no moisture row needs it
    open_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="target_moisture_wb"):
        transient.plot_state_trajectory_summary({"ID": (without_target,)}, state_fields=("w_gr",), styles=_STYLES)
    assert set(plt.get_fignums()) == open_figures  # the refused plot closed its figure


def test_close_target_values_are_pushed_apart_and_stay_one_per_line() -> None:
    """Six targets half a percentage point apart get six values that do not overlap, each joined to its own line."""
    transient = analysis.evaluation.plots.transient
    records = tuple(_timed_record(f"t{index}", 20, target=0.12 + 0.005 * index) for index in range(6))
    figure = transient.plot_state_trajectory_summary({"ID": records}, state_fields=("w_gr",), styles=_STYLES)
    figure.canvas.draw()
    values = figure.axes[0].texts
    assert [text.get_text() for text in values] == ["12 %", "12.5 %", "13 %", "13.5 %", "14 %", "14.5 %"]
    boxes = [Text.get_window_extent(text) for text in values]
    assert all(upper.y0 >= lower.y1 for lower, upper in pairwise(boxes))
    assert all(isinstance(text, Annotation) and text.arrow_patch is not None for text in values)


def test_aggregate_mean_maps_are_signed_by_default_with_a_locked_scale_per_row_and_a_unit_on_every_bar() -> None:
    """The mean is the signed pooled error (absolute on request), one scale per field row serves all contexts, and the figure grows by column."""
    transient = analysis.evaluation.plots.transient
    low, high = (sequence_record("autonomous_full", "full", case_id=name, offset=offset) for name, offset in (("low", 1.0), ("high", -3.0)))
    series = {"Low": (low,), "High": (high,)}
    fields = ("T", "phi")
    figure = transient.plot_aggregate_error_maps(series, state_fields=fields, physical_time=1.0, statistic="mean")
    maps, bars = figure.axes[:4], figure.axes[4:]
    for axis, expected in zip(maps, (1.0, -3.0, 100.0, -300.0), strict=True):  # row-major: T, then phi in pp
        np.testing.assert_allclose(cast("Any", axis.collections[0].get_array()), expected)
    for row in (0, 1):
        left, right = (cast("BoundaryNorm", maps[2 * row + column].collections[0].norm).boundaries for column in (0, 1))
        np.testing.assert_array_equal(left, right)
        assert left[0] == pytest.approx(-left[-1])
    assert [bar.get_ylabel() for bar in bars] == [f"[{transient._display_unit(field, difference=True)}]" for field in fields for _ in series]
    single = transient.plot_aggregate_error_maps({"Low": (low,)}, state_fields=fields, physical_time=1.0, statistic="mean")
    assert single.get_figwidth() < figure.get_figwidth()
    plt.close(single)
    plt.close(figure)
    pooled = {"Pooled": (low, high)}
    signed = transient.plot_aggregate_error_maps(pooled, state_fields=("T",), physical_time=1.0, statistic="mean")
    absolute = transient.plot_aggregate_error_maps(pooled, state_fields=("T",), physical_time=1.0, statistic="mean", error_mode="Absolute error")
    np.testing.assert_allclose(cast("Any", signed.axes[0].collections[0].get_array()), -1.0)
    np.testing.assert_allclose(cast("Any", absolute.axes[0].collections[0].get_array()), 2.0)
    unit = transient._display_unit("T", difference=True)
    # One mean line above each mean map: the mean of |shown values| (the signed map shows -1, the absolute one 2).
    for figure, mean in ((signed, "1.00"), (absolute, "2.00")):
        assert [text.get_text() for text in figure.axes[0].texts if text.get_text().startswith("mean")] == [f"mean |e| = {mean} {unit}"]
    plt.close(signed)
    plt.close(absolute)
    std = transient.plot_aggregate_error_maps(pooled, state_fields=("T",), physical_time=1.0, statistic="std")
    assert not any(text.get_text().startswith("mean") for text in std.axes[0].texts)
    plt.close(std)
    with pytest.raises(ValueError, match="at least one"):  # no context has a case at all: refused before any map is built
        transient.plot_aggregate_error_maps({"ID": ()}, state_fields=("T",), physical_time=1.0, statistic="mean")


def test_aggregate_normalized_mean_maps_need_one_train_sigma_over_the_shown_contexts() -> None:
    """The absolute mean over the train sigma is refused for scales of different provenance, for water and without scales."""
    transient = analysis.evaluation.plots.transient
    record = sequence_record("autonomous_full", "full", offset=3.0)
    scales = {
        name: {
            "T": error_diagnostics.transient_reference_scale(scaling_state(membership_digest=digest), field="T", unit="K", dataset_identity=digest)
        }
        for name, digest in (("A", "a" * 64), ("B", "b" * 64))
    }
    arguments: dict[str, Any] = {
        "state_fields": ("T",),
        "physical_time": 1.0,
        "statistic": "mean",
        "error_mode": error_diagnostics.NORMALIZED_ABSOLUTE_ERROR,
    }
    shown = {"A": (record,), "B": (record,)}
    with pytest.raises(ValueError, match="incompatible"):
        transient.plot_aggregate_error_maps(shown, reference_scales=scales, **arguments)
    with pytest.raises(ValueError, match="No valid train reference scale"):
        transient.plot_aggregate_error_maps(shown, **arguments)
    with pytest.raises(ValueError, match="stored states"):
        transient.plot_aggregate_error_maps(shown, reference_scales=scales, **{**arguments, "state_fields": ("w_gr",)})
    for field, unit in (("phi", "K"), ("T", "degC")):  # a scale of another field, or of T in another unit, is no sigma for T in K
        foreign = error_diagnostics.transient_reference_scale(scaling_state(), field=field, unit=unit, dataset_identity="a" * 64)
        with pytest.raises(ValueError, match="different physical field or unit"):
            transient.plot_aggregate_error_maps({"A": (record,)}, reference_scales={"A": {"T": foreign}}, **arguments)
    with pytest.raises(
        ValueError, match="No valid train reference scale"
    ):  # an unavailable scale of any shown context refuses, even without a reason
        transient.plot_aggregate_error_maps(shown, reference_scales={"A": scales["A"], "B": {"T": ""}}, **arguments)
    sigma = cast("Any", scales["A"]["T"]).value
    figure = transient.plot_aggregate_error_maps({"A": (record,)}, reference_scales=scales, **arguments)
    np.testing.assert_allclose(cast("Any", figure.axes[0].collections[0].get_array()), 3.0 / sigma)
    plt.close(figure)


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
        {"ID": records}, scaling_states={"ID": scaling_state()}, state_fields=("T",), styles=_STYLES
    )
    np.testing.assert_allclose(np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=np.float64), [math.sqrt(1.5)] * 2)
    plt.close(figure)


def _reached(record: SimpleNamespace, *, reached: bool = True) -> SimpleNamespace:
    """Mark a record's reference as having reached its target (or, with ``reached`` False, as stopped without it)."""
    record.target["reference_reached"] = reached
    return record


def _sloped(record: SimpleNamespace, slope: float, *, material: str = "lentil", reached: bool = True) -> SimpleNamespace:
    """Give a record an error growing from 0 at its origin by ``slope`` per stored hour in every cell, its material and completion."""
    record.predicted_states = (record.reference_states + slope * record.physical_times[:, None, None, None]).astype(np.float32)
    record.identity["material_family"] = material
    return _reached(record, reached=reached)


# The stored state each case of 3, 5, 9 and 11 states takes at the 20 phase points: the nearest, the origin included, ties earlier.
_PHASE_STATES = {
    3: np.repeat([0, 1, 2], [5, 10, 5]),
    5: np.asarray([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]),
    9: np.asarray([0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8]),
    11: np.arange(1, 21) // 2,
}


def test_error_over_phase_pools_each_reached_case_at_its_nearest_state_with_equal_material_weight() -> None:
    """
    Lentil cases of 4 h, 8 h and a chickpea case of 20 h enter all 20 points; censored cases, origin-only too, are counted.

    With a state scale of 2 the lentil errors are 2 k and the chickpea error k at stored state k, so the line is
    sqrt(k_a² + k_b² + k_c² / 2): equal over the lentil cases, then over the materials (a plain case mean or another state
    differs), 0 where every case resolves to its origin. The note states t_end in hours; an unknown clock is refused.
    """
    transient = analysis.evaluation.plots.transient
    records = (
        _sloped(_timed_record("a", 3), 2.0),
        _sloped(_timed_record("b", 5), 2.0),
        _sloped(_timed_record("c", 11), 1.0, material="chickpea"),
        _sloped(_timed_record("cut", 6), 50.0, reached=False),
        _sloped(_timed_record("stub", 1), 50.0, reached=False),
    )
    arguments: dict[str, Any] = {"scaling_states": {"ID": scaling_state()}, "state_fields": ("T",), "styles": _STYLES}
    figure = transient.plot_error_over_physical_time({"ID": records}, **arguments, time_axis="phase")
    line, count = figure.axes[0].lines[0], figure.axes[-1]
    np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=np.float64), analysis.ui.time.PHASE_GRID)
    np.testing.assert_allclose(
        np.asarray(line.get_ydata(), dtype=np.float64),
        np.sqrt(_PHASE_STATES[3] ** 2 + _PHASE_STATES[5] ** 2 + _PHASE_STATES[11] ** 2 / 2),
        atol=1e-12,
    )
    np.testing.assert_array_equal(count.lines[0].get_ydata(), 3)
    assert (count.get_xlabel(), count.get_xlim(), count.get_ylabel()) == ("τ = t / t_end [-]", (0.0, 1.0), "n(τ)")
    note = " ".join(" ".join(cast("Any", text).get_text() for text in figure.artists if hasattr(text, "get_text")).split())
    assert "t_end 4-20 h, median 8 h; 3 reached, 2 censored not pooled" in note
    with pytest.raises(ValueError, match="time_axis"):
        transient.plot_error_over_physical_time({"ID": records}, **arguments, time_axis=cast("Any", "tau"))


def test_relative_norms_over_phase_pool_each_reached_case_at_its_nearest_state_with_equal_material_weight(tmp_path: Path) -> None:
    """
    Lentil cases of 2 h and 4 h and a chickpea case of 8 h enter all 20 phase points; a censored lentil case is only counted.

    The relative L2 grows from 0 at the origin by 0.01 (lentil) and 0.005 (chickpea) per hour, so each point is the mean of the
    lentil mean and the chickpea value at the stored states each case resolves. In hours the 8 h case stands alone after 4 h.
    """
    records = []
    cases = (("a", "lentil", 3, 0.01), ("b", "lentil", 5, 0.01), ("c", "chickpea", 9, 0.005), ("cut", "lentil", 3, 1.0))
    for case_id, material, length, rate in cases:
        record = sequence_record("autonomous_full", "full", case_id=case_id, length=length)
        record.reference_states = np.full_like(record.reference_states, 300.0)
        records.append(_sloped(record, 300.0 * rate, material=material, reached=case_id != "cut"))
    evaluation = session.TransientEvaluationSession({"id": production_frame(tuple(records), artifact_root=str(tmp_path / "artifact"))})
    hours = evaluation.relative_norm_curve("id").query("metric_id == 'physical_relative_l2_T'")
    assert list(hours.query("physical_time > 4")["cases"]) == [1] * 4
    curve = evaluation.relative_norm_curve("id", time_axis="phase")
    phase = curve.query("metric_id == 'physical_relative_l2_T'")
    np.testing.assert_allclose(phase["phase"], analysis.ui.time.PHASE_GRID)
    lentil = 0.01 * (_PHASE_STATES[3] + _PHASE_STATES[5]) / 2
    np.testing.assert_allclose(phase["value"], (lentil + 0.005 * _PHASE_STATES[9]) / 2, atol=1e-12)
    assert set(phase["cases"]) == set(phase["supported_cases"]) == {3}
    assert curve.attrs["phase_label"] == "t_end 2-8 h, median 4 h; 3 reached, 1 censored not pooled"
    figure = analysis.evaluation.plots.transient.plot_relative_norms_over_time({"id": curve}, styles=evaluation.series_styles())
    fields = len(analysis.evaluation.transient_artifact.STATE_ORDER)
    norm_axis, strip = figure.axes[0], figure.axes[fields]  # the supported cases stand in their own strip under the norm row
    assert (norm_axis.get_xlabel(), strip.get_xlabel(), strip.get_ylabel()) == ("", "τ = t / t_end [-]", "n(τ)")
    np.testing.assert_array_equal(strip.get_lines()[0].get_ydata(), [3] * len(analysis.ui.time.PHASE_GRID))
    assert [line.get_label() for line in norm_axis.get_lines()] == ["id"]  # no support step overlays the norm
    twice = analysis.evaluation.plots.transient.plot_relative_norms_over_time({"one": curve, "two": curve}, styles=evaluation.series_styles())
    steps = twice.axes[fields].get_lines()
    assert [line.get_markevery() for line in steps] == [(0, 2), (1, 2)]  # equal counts stay apart by their staggered markers
    assert all(np.array_equal(np.asarray(line.get_ydata()), [3] * len(analysis.ui.time.PHASE_GRID)) for line in steps)  # unmoved


def test_the_norm_lines_of_a_run_over_two_materials_carry_their_markers_in_the_panel_and_the_key(tmp_path: Path) -> None:
    """
    Run F spans lentil and chickpea beside run U: F's two norm lines share colour and line style, so each carries its marker.

    The key shows the same markers. Three curves over two stored times: the third strip's stagger stays inside its two points,
    so every strip still shows a marker.
    """
    frames = {}
    for name, checkpoint, materials in (("F ID", "f" * 64, ("lentil", "chickpea")), ("U ID", "e" * 64, ("lentil",))):
        records = tuple(
            sequence_record("autonomous_full", "full", case_id=material, offset=303.0, material_family=material) for material in materials
        )
        for record in records:
            record.reference_states = record.reference_states + 300.0
            record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        frames[name] = production_frame(records, artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
    evaluation = session.TransientEvaluationSession(frames)
    curves = {
        "F · Lentil, ID": evaluation.relative_norm_curve("F ID", case_ids=("lentil",)),
        "F · Chickpea, ID": evaluation.relative_norm_curve("F ID", case_ids=("chickpea",)),
        "U · Lentil, ID": evaluation.relative_norm_curve("U ID"),
    }
    figure = analysis.evaluation.plots.transient.plot_relative_norms_over_time(curves, styles=evaluation.series_styles())
    shapes = analysis.presentation.visual_semantics.MATERIAL_MARKERS
    lines = {line.get_label(): line.get_marker() for line in figure.axes[0].get_lines()}
    keys = {  # one titled legend per run, its entries without the run
        (legend.get_title().get_text(), text.get_text()): cast("Any", handle).get_marker()
        for legend in figure.legends
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    }
    try:
        assert (lines["F · Lentil, ID"], lines["F · Chickpea, ID"]) == (shapes["lentil"], shapes["chickpea"])
        assert (keys["F", "Lentil, ID"], keys["F", "Chickpea, ID"]) == (shapes["lentil"], shapes["chickpea"])
        strips = figure.axes[len(analysis.evaluation.transient_artifact.STATE_ORDER)].get_lines()  # under the first norm panel
        assert all(cast("tuple[int, int]", line.get_markevery())[0] < np.asarray(line.get_xdata()).size for line in strips)
        # A voided first point, where F · Lentil's only staggered marker falls, moves that marker to the finite point.
        gapped = curves["F · Lentil, ID"].copy()
        gapped.loc[gapped["physical_time"] == gapped["physical_time"].min(), "value"] = np.nan
        plt.close(figure)
        figure = analysis.evaluation.plots.transient.plot_relative_norms_over_time(
            {**curves, "F · Lentil, ID": gapped}, styles=evaluation.series_styles()
        )
        shown = {line.get_label(): line.get_markevery() for line in figure.axes[0].get_lines()}
        assert shown["F · Lentil, ID"] == [1]
    finally:
        evaluation.close()
        plt.close(figure)


def test_aggregate_maps_and_parity_at_a_phase_take_each_reached_case_at_its_nearest_state_and_keep_the_note_off_the_maps() -> None:
    """
    Maps and parity at a phase take each reached case at its nearest stored state, and the phase note stays below the maps.

    At τ = 0.5 a 4 h case gives its 2 h state (error 2 K) and a 20 h case its 10 h state (10 K): the mean map is 6 K and the
    parity points lie 2 K and 10 K above the reference. At τ = 0.25 the ties take the 4 h case's origin (0 K) and the 20 h
    case's 4 h state (4 K). A censored case of 100 K per hour is only counted; the note of three contexts stays below the maps.
    """
    transient = analysis.evaluation.plots.transient
    records = (_sloped(_timed_record("short", 3), 1.0), _sloped(_timed_record("long", 11), 1.0))
    records += (_sloped(_timed_record("cut", 6), 100.0, reached=False),)
    for phase, mean, hours in ((0.5, 6.0, "2-10"), (0.25, 2.0, "0-4")):
        figure = transient.plot_aggregate_error_maps({"ID": records}, state_fields=("T",), phase=phase, statistic="mean")
        np.testing.assert_allclose(cast("Any", figure.axes[0].collections[0].get_array()), mean)
        footer = figure.texts[-1].get_text()
        assert f"τ = {phase:.2f}: stored t {hours} h" in footer
        assert "2 reached, 1 censored not pooled" in footer
    parity = transient.plot_predicted_vs_reference_channels({"ID": records}, state_fields=("T",), phase=0.5, styles=_STYLES)
    offsets = np.asarray(parity.axes[0].collections[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 1] - offsets[:, 0], [2.0, 10.0])
    wide = transient.plot_aggregate_error_maps(dict.fromkeys(("A", "B", "C"), records), state_fields=("T",), phase=0.5, statistic="mean")
    wide.canvas.draw()
    renderer = cast("Any", wide.canvas).get_renderer()
    assert wide.texts[-1].get_window_extent(renderer).y1 < min(cast("Any", axis.get_tightbbox(renderer)).y0 for axis in wide.axes[:3])
    with pytest.raises(ValueError, match="exactly one"):
        transient.plot_aggregate_error_maps({"ID": records}, state_fields=("T",), physical_time=2.0, phase=0.5, statistic="mean")


def test_the_trajectory_over_phase_draws_only_the_targets_of_the_pooled_cases() -> None:
    """A censored case with its own target adds a target line in hours, none over the phase; with no pooled case, no target at all."""
    transient = analysis.evaluation.plots.transient
    censored = _reached(_timed_record("cut", 6, target=0.2), reached=False)
    series = {"ID": (_reached(_timed_record("a", 3)), _reached(_timed_record("b", 11)), censored)}
    for shown, time_axis, expected in (
        (series, "hours", [12.0, 20.0]),
        (series, "phase", [12.0]),
        ({"ID": (censored,)}, "phase", []),
    ):
        figure = transient.plot_state_trajectory_summary(shown, state_fields=("w_gr",), styles=_STYLES, time_axis=cast("Any", time_axis))
        levels = {float(np.asarray(line.get_ydata())[0]) for axis in figure.axes for line in axis.get_lines() if line.get_linestyle() == ":"}
        assert sorted(levels) == pytest.approx(expected)
        plt.close(figure)


def test_error_over_time_refuses_a_record_without_a_time_after_its_origin_instead_of_dropping_it() -> None:
    """A reached one-point record beside longer ones would silently leave the cohort: refused in hours, and over the phase too."""
    transient = analysis.evaluation.plots.transient
    single = _reached(_timed_record("single", 1))
    for time_axis, refusal in (("hours", "after its origin"), ("phase", "no drying phase")):
        with pytest.raises(ValueError, match=refusal):
            transient.plot_error_over_physical_time(
                {"ID": (single, _timed_record("long", 20))},
                scaling_states={"ID": scaling_state()},
                state_fields=("T",),
                styles=_STYLES,
                time_axis=cast("Any", time_axis),
            )


def test_a_run_keeps_its_shade_in_every_view_of_a_comparison_of_two_runs_also_where_only_one_has_evidence(tmp_path: Path) -> None:
    """
    Two FNO runs of one comparison (06) get the two FNO shades by checkpoint in the record, summary and relative-norm views.

    Only A has rolling-origin records, so the horizon view shows A alone, still in its FNO shade. The session frames carry each
    run's model kind; the run names sort the other way round than the checkpoints and must not matter.
    """
    transient = analysis.evaluation.plots.transient
    semantics = analysis.presentation.visual_semantics
    frames = {}
    for name, run_name, checkpoint in (("A ID", "z-run", "a" * 64), ("B ID", "a-run", "b" * 64)):
        # Chickpea: its material colour differs from both FNO shades, so a single-run style would show (lentil's is the FNO hue).
        modes = (("autonomous_full", "full"), ("rolling_origin", 2))[: 2 if name == "A ID" else 1]
        records = tuple(sequence_record(mode, horizon, material_family="chickpea") for mode, horizon in modes)
        records[0].target.update(reference_time_to_target=2.0)
        for record in records:
            record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        frames[name] = production_frame(records, artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"].update(name=run_name, best_checkpoint_sha256=checkpoint)
    evaluation = session.TransientEvaluationSession(frames)
    summary, cases, styles = evaluation.dataset_dataframe(), evaluation.case_dataframe(modes=("autonomous_full",)), evaluation.series_styles()
    assert set(summary["model_kind"]) == set(cases["model_kind"]) == {"fno"}
    accuracy = cases.loc[cases["scope"] == "cumulative"]
    reports = {name: evaluation.timing_report(name) for name in frames}
    targets = {name: evaluation.full_autonomous_summaries(name) for name in frames}
    curves = {name: evaluation.relative_norm_curve(name) for name in frames}
    shades = [to_hex(semantics.architecture_shade("fno", index, 2)) for index in range(2)]  # A's checkpoint sorts first
    views = {
        "target": (transient.plot_target_time(targets, styles=styles), "collections", shades),
        "channel": (transient.plot_channel_error(summary, state_fields=("T",), styles=styles), "patches", shades),
        "distribution": (transient.plot_error_distributions(cases, state_fields=("T",), styles=styles), "collections", shades),  # one case: a point
        "horizon": (transient.plot_horizon_error(summary, styles=styles), "lines", shades[:1]),
        "endpoint": (transient.plot_endpoint_vs_cumulative(summary, styles=styles), "collections", shades),
        "runtime": (transient.plot_accuracy_runtime_comparison(accuracy, reports, styles=styles), "collections", shades),
        "relative norms": (transient.plot_relative_norms_over_time(curves, styles=styles), "lines", shades),
    }

    def colour(artist: Any) -> str:
        face = artist.get_facecolor() if hasattr(artist, "get_facecolor") else artist.get_color()  # a line has no face
        return to_hex(face[0] if np.ndim(face) == 2 else face)  # a collection holds one face colour per point

    for view, (figure, artists, expected) in views.items():
        assert list(dict.fromkeys(colour(artist) for artist in getattr(figure.axes[0], artists))) == expected, view


def test_record_series_take_the_material_colour_and_role_line_style_of_one_run_and_the_architecture_colour_of_several() -> None:
    """One checkpoint: lentil and chickpea in their colours, the parameter-OOD series dashed; FNO and U-NO checkpoints: their colours."""
    transient = analysis.evaluation.plots.transient
    semantics = analysis.presentation.visual_semantics
    lentil, chickpea = _timed_record("a", 20), _timed_record("b", 20)
    chickpea.identity = {**chickpea.identity, "material_family": "chickpea", "dataset_role": "parameter_ood"}
    figure = transient.plot_state_trajectory_summary({"Lentil": (lentil,), "Chickpea": (chickpea,)}, state_fields=("T",), styles=_STYLES)
    looks = {line.get_label(): (line.get_color(), line.get_linestyle()) for line in figure.axes[0].get_lines()}
    assert looks["Lentil · Prediction"] == (semantics.material_color("lentil"), "-")
    assert looks["Chickpea · Prediction"] == looks["Chickpea · Reference"] == (semantics.material_color("chickpea"), "--")
    fno, uno = _timed_record("a", 20), _timed_record("a", 20)
    fno.identity = {**fno.identity, "checkpoint_identity": {"best_checkpoint_sha256": "f"}}
    uno.identity = {**uno.identity, "model_kind": "uno", "checkpoint_identity": {"best_checkpoint_sha256": "u"}}
    scales = {"FNO": scaling_state(), "U-NO": scaling_state()}
    error = transient.plot_error_over_physical_time(
        {"FNO": (fno,), "U-NO": (uno,)}, scaling_states=scales, state_fields=("T",), styles=_comparison_of(fno, uno)
    )
    colours = {line.get_label(): line.get_color() for line in error.axes[0].get_lines()}
    assert colours["FNO · material-case RMSE"] == semantics.architecture_color("fno")
    assert colours["U-NO · material-case RMSE"] == semantics.architecture_color("uno")


def test_four_runs_of_one_architecture_take_one_run_colour_each_and_say_so_in_the_context_line() -> None:
    """Four FNO checkpoints exceed the FNO shades: each run takes its run colour in checkpoint order, noted «colour = run»."""
    transient = analysis.evaluation.plots.transient
    semantics = analysis.presentation.visual_semantics
    records = {}
    for checkpoint in "dcba":  # the series order differs from the checkpoint order
        record = _timed_record(checkpoint, 20)
        record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        records[f"Run {checkpoint}"] = (record,)
    styles, scales = _comparison_of(*(record for (record,) in records.values())), dict.fromkeys(records, scaling_state())
    figure = transient.plot_error_over_physical_time(records, scaling_states=scales, state_fields=("T",), styles=styles)
    colours = {line.get_label(): to_hex(line.get_color()) for line in figure.axes[0].get_lines()}
    assert [colours[f"Run {checkpoint} · material-case RMSE"] for checkpoint in "abcd"] == [to_hex(colour) for colour in semantics.RUN_COLORS[:4]]
    assert any(isinstance(text, Text) and text.get_text().splitlines()[-1] == "colour = run" for text in figure.artists)


def test_summary_bars_take_the_material_colour_and_role_hatch_and_rows_need_their_model_kind() -> None:
    """One run: a bar per frame and material in its colour, the parameter-OOD one hatched; several runs: by model_kind; rows without it: refused."""
    transient = analysis.evaluation.plots.transient
    semantics = analysis.presentation.visual_semantics

    def rows(frame: str, material: str, role: str, run: str, kind: str = "fno") -> pd.DataFrame:
        row = {"frame": frame, "material_family": material, "mode": "autonomous_full", "requested_horizon": "full", "scope": "cumulative"}
        values = pd.DataFrame([{**row, "normalized_drying_group_macro_rmse": 0.1, "normalized_rmse_T": 0.2}])
        return _with_provenance(values, role=role, run=run, model_kind=kind)

    one_run = pd.concat((rows("M ID", "lentil", "id", "m"), rows("M OOD", "chickpea", "parameter_ood", "m")), ignore_index=True)
    bars = transient.plot_channel_error(one_run, state_fields=("T",), styles=_table_styles(one_run)).axes[0].patches
    assert [(to_hex(bar.get_facecolor()), bar.get_hatch() or "") for bar in bars] == [  # the run's series in the selector order
        (to_hex(semantics.material_color("lentil")), ""),
        (to_hex(semantics.material_color("chickpea")), "//"),
    ]
    with pytest.raises(ValueError, match="model_kind"):
        transient.plot_channel_error(one_run.drop(columns="model_kind"), state_fields=("T",), styles=_table_styles(one_run))
    two_runs = pd.concat((rows("F ID", "lentil", "id", "f", "fno"), rows("U ID", "lentil", "id", "u", "uno")), ignore_index=True)
    bars = transient.plot_channel_error(two_runs, state_fields=("T",), styles=_table_styles(two_runs)).axes[0].patches
    assert [to_hex(bar.get_facecolor()) for bar in bars] == [to_hex(semantics.architecture_color(kind)) for kind in ("fno", "uno")]


def _paired_timing_report(seconds: Sequence[float]) -> Any:
    """Return a timing report of one case per reference duration, each against a 0.01 s model call."""
    transient_timing = analysis.evaluation.transient_timing
    cases = tuple(
        transient_timing.TransientTimingCase(
            case_id=f"c{index}",
            repetitions={"comsol_transient_drying_seconds": (value,), "drying_no_rollout_model_seconds": (0.01,)},
            device="cpu",
            precision="float32",
            pt_payload_identity="a" * 64,
            warmup_passes=1,
            solver_hostname="solver-node",
            solver_requested_cores=8,
        )
        for index, value in enumerate(seconds)
    )
    return transient_timing.build_transient_timing_report(cases)


def test_duration_ratios_are_points_at_the_median_with_the_10th_to_90th_percentile_of_the_paired_case_ratios() -> None:
    """Five paired cases give one neutral point, no bar, at the median of their ratios with a P10-P90 whisker; formulas without pairs say so."""
    seconds = (0.3, 0.5, 1.1, 2.0, 7.0)
    figure = analysis.evaluation.plots.transient.plot_timing_speedups(_paired_timing_report(seconds), role="id")
    axis = figure.axes[0]
    ratios = np.asarray(seconds) / 0.01
    assert not axis.patches
    (whisker,) = axis.containers
    assert isinstance(whisker, ErrorbarContainer)
    data_line, _caps, (bar_lines,) = whisker.lines
    assert data_line is not None
    np.testing.assert_allclose(np.asarray(data_line.get_xdata(), dtype=np.float64), [np.median(ratios)])
    (segment,) = bar_lines.get_segments()
    np.testing.assert_allclose(sorted(segment[:, 0]), np.quantile(ratios, (0.1, 0.9)))
    assert axis.get_xscale() == "log"
    assert [text.get_text() for text in axis.texts] == ["no paired case"] * 4
    assert " · ID cases" in _figure_title(figure)  # the report's role is named in the title


def test_a_duration_ratio_that_is_not_positive_and_finite_is_named_unavailable_and_keeps_the_scale_linear() -> None:
    """A zero median, a negative 10th or an infinite 90th percentile has no place on a log scale: no point, a note, a linear axis, the run named."""
    report = _paired_timing_report((0.3, 0.5, 1.1))
    name = next(name for name, summary in report.speedups.items() if summary.median is not None)
    for invalid in ({"median": 0.0}, {"q10": -1.0}, {"q90": math.inf}):
        broken = replace(report, speedups={**report.speedups, name: replace(report.speedups[name], **invalid)})
        figure = analysis.evaluation.plots.transient.plot_timing_speedups(broken, models=("A",))
        axis = figure.axes[0]
        assert not axis.containers
        assert axis.get_xscale() == "linear"
        assert sorted(text.get_text() for text in axis.texts) == ["no paired case"] * 4 + ["no positive finite ratio"], invalid
        assert _figure_title(figure).endswith(" (A)"), invalid  # the compared run stays named; its row says why it has no bar
    plt.close("all")


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
            dataset_role="material_ood" if material == "kidney_bean" else "id",
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
    # Before any view chose channels, a transient view checks T, phi and w_gr where offered (the owner's default, as in EDA).
    channels = analysis.evaluation.panel._Channels("Channels", fields, None)
    assert set(channels.checks.options) == {"T", "phi", "w_surf", "w_int"}
    assert channels.checks.selected == ("T", "phi")
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
            if isinstance(widget, widgets.ToggleButton) and widget.description == "Kidney bean, Material OOD"
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
    assert all(f"{label}: n(t)=1" in printed for label in ("Chickpea, ID", "Lentil, ID", "Kidney bean, Material OOD"))
    evaluation.close()
    plt.close("all")


def _shared_state(tmp_path: Path) -> Any:
    """Return a real selection state over runs ``a`` and ``b`` of one transient experiment, shared by their panels as in the workspace."""
    runs = tuple(SimpleNamespace(run_dir=(tmp_path / name).resolve()) for name in ("a", "b"))
    group = analysis.evaluation.run_discovery.EvaluationRunGroup("a" * 64, "runs", "transient_drying", cast("Any", runs))
    catalog = analysis.evaluation.run_discovery.EvaluationRunCatalog(cast("Any", runs), (group,), (), {}, {}, {}, {})
    return analysis.evaluation.selection.EvaluationSelectionState(catalog, comparison=False)


def _snapshot(evaluation: session.TransientEvaluationSession, state: Any) -> tuple[analysis.ui.panel.ViewPanel, widgets.FloatText]:
    """Open the mean error maps, a snapshot view, on the shared ``state`` and return the view and its time field."""
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    view = analysis.evaluation.panel._make_transient_aggregate_viewer(
        session=evaluation, fields=fields, plot_kind="mean_error", comparison=False, selection_state=state
    )
    time = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]")
    return view, time


def test_the_snapshot_defaults_to_the_latest_time_every_shown_case_still_stores() -> None:
    """A 4 h and a 1 h lentil case beside a 2 h chickpea case open on 1 h, where every case has a state, not where most materials do."""
    records = (
        SimpleNamespace(identity={"material_family": "lentil"}, physical_times=(0.0, 1.0, 2.0, 3.0, 4.0)),
        SimpleNamespace(identity={"material_family": "lentil"}, physical_times=(0.0, 1.0)),
        SimpleNamespace(identity={"material_family": "chickpea"}, physical_times=(0.0, 1.0, 2.0)),
    )
    assert evaluation_panel._coverage_time({"lentil": records[:2], "chickpea": records[2:]}) == 1.0
    with pytest.raises(ValueError, match="share no stored time"):
        evaluation_panel._coverage_time({"disjoint": (SimpleNamespace(physical_times=(1.0,)), SimpleNamespace(physical_times=(2.0,)))})


def test_a_chosen_snapshot_time_opens_the_other_views_of_its_run_but_never_another_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run change opens a new session, whose views open on their own default even where they store the time chosen before."""
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    state = _shared_state(tmp_path)
    first = _transient_evaluation(tmp_path / "a", sequence_record("autonomous_full", "full", case_id="one", length=5))
    second = _transient_evaluation(
        tmp_path / "b",
        sequence_record("autonomous_full", "full", case_id="long", length=5),
        sequence_record("autonomous_full", "full", case_id="short", material_family="chickpea"),
    )
    _view, chosen = _snapshot(first, state)
    assert chosen.value == 4.0  # one lentil case opens on its last stored time
    chosen.value = 1.0
    assert _snapshot(first, state)[1].value == 1.0  # another view of the same run
    state.select_experiment("a" * 64, run_dirs=((tmp_path / "b").resolve(),))  # the workspace's run change, which opens a new session
    assert _snapshot(second, state)[1].value == 2.0  # the latest time both cases store, though the long case stores later ones
    first.close()
    second.close()


def test_a_time_chosen_without_a_shared_selection_stays_when_the_case_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The session alone shares a chosen time: a sample view without a selection state shows the next case at 2 h, not at its last time."""
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    evaluation = _transient_evaluation(tmp_path, *(sequence_record("autonomous_full", "full", case_id=name, length=5) for name in ("a", "b")))
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    tree = _widget_tree(analysis.evaluation.panel._make_transient_sample_viewer(session=evaluation, fields=fields, comparison=False).widget)
    time = next(widget for widget in tree if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]")
    case = next(widget for widget in tree if isinstance(widget, widgets.IntText) and widget.description == "Case")
    time.value = 2.0
    case.value = 2
    assert time.value == 2.0
    evaluation.close()


def test_a_snapshot_view_opened_on_a_cleared_selection_says_so_and_binds_its_time_once_records_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After another view cleared every dataset, the view opens on the empty-selection message without a time; choosing a dataset binds it."""
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    state = _shared_state(tmp_path)
    evaluation = _transient_evaluation(tmp_path, sequence_record("autonomous_full", "full", case_id="one", length=5))
    state.select_datasets((), pooled=False)
    view, time = _snapshot(evaluation, state)
    navigator = next(box for box in _widget_tree(view.widget) if isinstance(box, widgets.HBox) and time in box.children)
    assert analysis.ui.viewers.EMPTY_SELECTION_MESSAGE in _shown_text(view)
    assert navigator.layout.display == "none"
    assert view.status.text == ""
    lentil = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.Checkbox) and widget.description == "Lentil, ID")
    lentil.value = True
    assert navigator.layout.display is None
    assert time.value == 4.0
    evaluation.close()


def test_emptying_the_selection_of_a_snapshot_view_clears_its_time_from_the_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The navigator hides with the records, and its time goes from the status instead of standing under the empty-selection message."""
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    evaluation = _transient_evaluation(tmp_path, sequence_record("autonomous_full", "full", case_id="one", length=5))
    view, time = _snapshot(evaluation, _shared_state(tmp_path))
    assert f"{time.value:g} h" in view.status.text
    lentil = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.Checkbox) and widget.description == "Lentil, ID")
    lentil.value = False
    assert analysis.ui.viewers.EMPTY_SELECTION_MESSAGE in _shown_text(view)
    assert view.status.text == ""
    evaluation.close()


def test_a_snapshot_view_on_the_drying_phase_shows_no_time_and_reports_it_again_back_on_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phase clock hides the time navigator and its time note; switched back to hours, the navigator reports its time again."""
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    evaluation = _transient_evaluation(tmp_path, sequence_record("autonomous_full", "full", case_id="one", length=5))
    view, time = _snapshot(evaluation, _shared_state(tmp_path))
    clock = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.ToggleButtons) and widget.value == "hours")
    note = f"{time.value:g} h"
    assert note in view.status.text
    clock.value = "phase"
    assert note not in view.status.text
    clock.value = "hours"
    assert note in view.status.text
    evaluation.close()


def test_a_snapshot_view_switched_to_the_drying_phase_keeps_another_controls_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hidden navigator rebinds without reporting its time, so a message another control wrote stays on the shared status line."""
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    evaluation = _transient_evaluation(tmp_path, sequence_record("autonomous_full", "full", case_id="one", length=5))
    view, _time = _snapshot(evaluation, _shared_state(tmp_path))
    clock = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.ToggleButtons) and widget.value == "hours")
    view.status.set("Unavailable channels removed: T.", "warning")
    clock.value = "phase"
    assert view.status.text == "Unavailable channels removed: T."
    evaluation.close()


def test_titles_name_only_an_explicit_run_label_given_per_frame_and_a_mapping_must_cover_the_frames(tmp_path: Path) -> None:
    """
    Case and aggregate titles end with the explicit label of the frame's run; an alias or no mapping names none; a bad mapping is refused.

    The rendered case title ends with the label, after its time: «… · t = 2 h (FNO · B · seed 9)».
    """
    panel = analysis.evaluation.panel
    frame = production_frame((sequence_record("autonomous_full", "full"),), artifact_root=str(tmp_path / "artifact"))
    evaluation = session.TransientEvaluationSession({"FNO ID": frame})
    fields = panel._transient_channel_fields(evaluation)
    cases = (
        (None, False),
        ({"FNO ID": panel.FrameRun("FNO", named=False)}, False),  # a storage alias is never named
        ({"FNO ID": panel.FrameRun("FNO · B · seed 9", named=True)}, True),
    )
    for frame_runs, named in cases:
        export_state: dict[str, Any] = {}
        panel._make_transient_sample_viewer(session=evaluation, fields=fields, comparison=False, export_state=export_state, frame_runs=frame_runs)
        registry = panel._build_transient_sections(
            session=evaluation,
            comparison=False,
            selection_state=None,
            evaluation_compute_device="cpu",
            training_histories=dict,
            frame_runs=frame_runs,
        )
        figure = {title: factory for title, factory, _stem in registry["global_error"][0]}["Global error metrics"]()
        texts = [text.get_text() for text in figure.findobj(Text)]
        assert any(text.endswith("(FNO · B · seed 9)") for text in texts) is named
        assert export_state["fig"].get_suptitle().endswith(" h (FNO · B · seed 9)") is named
        plt.close(figure)
    for frame_runs in ({"FNO ID": panel.FrameRun(" ", named=True)}, {"FNO MATERIAL_OOD": panel.FrameRun("FNO", named=True)}):
        with pytest.raises(ValueError, match="Frame runs"):
            panel.build_transient_panel(session=evaluation, training_histories=dict, frame_runs=frame_runs)
    evaluation.close()
    plt.close("all")


def test_a_comparison_title_names_every_compared_run_and_a_note_names_the_run_without_evidence(tmp_path: Path) -> None:
    """
    B pairs no timing case, has no rolling origin and no predicted target: those titles still end with «(A vs B)».

    B has no point and no legend entry in these figures, so each note names B's gap.
    """
    frames = {}
    for name, checkpoint, case_id in (("A ID", "a" * 64, "case"), ("B ID", "b" * 64, "other")):  # only «case» has timing evidence
        modes = (("autonomous_full", "full"), ("rolling_origin", 2))[: 2 if name == "A ID" else 1]
        records = tuple(sequence_record(mode, horizon, case_id=case_id) for mode, horizon in modes)
        records[0].target.update(predicted_available=name == "A ID")
        for record in records:
            record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        frames[name] = production_frame(records, artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
    evaluation = session.TransientEvaluationSession(frames)
    registry = evaluation_panel._build_transient_sections(
        session=evaluation,
        comparison=True,
        selection_state=None,
        evaluation_compute_device="cpu",
        training_histories=dict,
        frame_runs={name: evaluation_panel.FrameRun(name.removesuffix(" ID"), named=True) for name in frames},
    )
    entries = {title: factory for section_entries, _name in registry.values() for title, factory, _stem in section_entries}
    runtime: dict[str, Any] = {}
    entries["Accuracy vs measured runtime"](export_state=runtime)
    ratios: dict[str, Any] = {}
    ratio_view = entries["Boundary-specific duration ratios"](export_state=ratios)
    # the ratio view draws one run and names its role; its figure and its status list every other selected run by its label
    # and role, never by its frame, with its paired ratios
    others = "Other selected runs and roles: B · ID (1 of 5 ratios paired)"
    assert _figure_title(ratios["fig"]) == "Boundary-specific duration ratios · ID cases (A)"
    assert others in _figure_note(ratios["fig"])
    assert ratio_view.status.text.endswith(f" {others}.")
    figures = {  # the runtime and target notes name the run and its role, never the frame token
        "B · ID: no paired accuracy and runtime evidence": runtime["fig"],
        "B ID: no finite rolling-origin error": entries["Error vs rollout horizon"](),
        "B · ID: paired reached 0, censored 0, unavailable 1": entries["Final-state drying summary"](),
    }
    try:
        for note, figure in figures.items():
            assert _figure_title(figure).endswith(" (A vs B)"), note
            assert any(note in " ".join(text.get_text().split()) for text in figure.findobj(Text)), note
    finally:
        plt.close("all")
        evaluation.close()


@pytest.mark.parametrize(
    ("plot", "column", "note"),
    [
        ("plot_channel_error", "normalized_rmse_T", "no finite full-rollout error"),
        ("plot_horizon_error", "normalized_drying_group_macro_rmse", "no finite rolling-origin error"),
    ],
)
def test_a_compared_run_with_rows_but_no_finite_error_stays_named_and_the_note_says_so(tmp_path: Path, plot: str, column: str, note: str) -> None:
    """B's rows hold no finite error, as a non-finite rollout leaves them: it shows no bar or line, the title names it, the note says why."""
    frames = {}
    for name, checkpoint in (("A ID", "a" * 64), ("B ID", "b" * 64)):
        records = tuple(sequence_record(mode, horizon) for mode, horizon in (("autonomous_full", "full"), ("rolling_origin", 2)))
        for record in records:
            record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        frames[name] = production_frame(records, artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
    evaluation = session.TransientEvaluationSession(frames)
    summary = evaluation.dataset_dataframe()
    summary.loc[summary["frame"] == "B ID", column] = np.nan  # the session leaves the errors of a non-finite rollout unset
    fields = {"state_fields": ("T",)} if plot == "plot_channel_error" else {}
    figure = getattr(analysis.evaluation.plots.transient, plot)(summary, styles=evaluation.series_styles(), models=("A", "B"), **fields)
    try:
        assert _figure_title(figure).endswith(" (A vs B)")
        assert any(f"B ID: {note}" in " ".join(text.get_text().split()) for text in figure.findobj(Text))
    finally:
        plt.close("all")
        evaluation.close()


def test_a_snapshot_title_names_both_compared_runs_where_one_has_no_case_at_the_chosen_time(tmp_path: Path) -> None:
    """A holds a 10 h case, B a 5 h case: at 8 h B's maps say «No case at this exact time», and the title still names «A vs B»."""
    frames = {}
    for name, checkpoint, case_id, length in (("A ID", "a" * 64, "case", 11), ("B ID", "b" * 64, "other", 6)):
        record = sequence_record("autonomous_full", "full", case_id=case_id, length=length)
        record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        frames[name] = production_frame((record,), artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
    evaluation = session.TransientEvaluationSession(frames)
    state: dict[str, Any] = {}
    view = evaluation_panel._make_transient_aggregate_viewer(
        session=evaluation,
        fields=evaluation_panel._transient_channel_fields(evaluation),
        plot_kind="mean_error",
        comparison=True,
        export_state=state,
        frame_runs={name: evaluation_panel.FrameRun(name.removesuffix(" ID"), named=True) for name in frames},
    )
    time = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.FloatText) and widget.description == "Time [h]")
    time.value = 8.0
    try:
        assert state["fig"].get_suptitle().endswith(" (A vs B)")
        assert any(text.get_text() == "No case at this exact time" for text in state["fig"].findobj(Text))
        # B's empty column still puts its run on a line of its own, from its context label.
        assert [axis.get_title() for axis in state["fig"].axes[:2]] == ["A\nLentil, ID · n(t) = 1", "B\nLentil, ID · n(t) = 0"]
    finally:
        plt.close("all")
        evaluation.close()


def _two_run_session(tmp_path: Path, **identity: Any) -> session.TransientEvaluationSession:
    """Open the comparison of two FNO runs «A ID» and «B ID», each with the one case «case», channel counts and ``identity`` in their identity."""
    frames = {}
    for index, (name, checkpoint) in enumerate((("A ID", "a" * 64), ("B ID", "b" * 64))):
        record = sequence_record("autonomous_full", "full")
        parameters = {"hidden_channels": 8, "in_channels": 12, "out_channels": 4}
        record.identity.update(checkpoint_identity={"best_checkpoint_sha256": checkpoint}, model_parameters=parameters, **identity)
        frames[name] = production_frame((record,), artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
        frames[name].attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR]["ended_at"] = f"2026-11-11T0{index}:00:00+00:00"
    return session.TransientEvaluationSession(frames)


def test_a_compared_error_over_time_with_run_labels_draws_in_hours_and_over_the_phase(tmp_path: Path) -> None:
    """With the runs' labels given, the aggregate error over time of two runs draws in both clocks: each series takes its frame's scaler."""
    evaluation = _two_run_session(tmp_path)
    state: dict[str, Any] = {}
    view = evaluation_panel._make_transient_scope_viewer(
        session=evaluation,
        fields=evaluation_panel._transient_channel_fields(evaluation),
        plot_kind="error_time",
        comparison=True,
        export_state=state,
        frame_runs={name: evaluation_panel.FrameRun(name.removesuffix(" ID"), named=True) for name in evaluation.frame_names},
    )
    clock = next(widget for widget in _widget_tree(view.widget) if isinstance(widget, widgets.ToggleButtons) and widget.value in {"phase", "hours"})
    try:
        for mode, title in (("phase", "Error over drying phase"), ("hours", "Error over physical time")):
            clock.value = mode
            assert isinstance(state.get("fig"), Figure), mode
            assert _figure_title(state["fig"]) == f"{title} (A vs B)"
    finally:
        plt.close("all")
        evaluation.close()


def test_two_runs_given_one_label_keep_their_own_series_and_configuration_rows(tmp_path: Path) -> None:
    """Two runs given the same label stay two runs: the compared dataset draws two series and the configuration has two rows, labels unique."""
    evaluation = _two_run_session(tmp_path)
    runs = {name: evaluation_panel.FrameRun("FNO", named=True) for name in evaluation.frame_names}
    inventory = dataset_cuts.transient_inventory(evaluation, refused={})
    choices = analysis.presentation.dataset_selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=False)
    try:
        series = evaluation_panel._transient_series(
            evaluation, dataset_cuts.transient_contexts(evaluation, choices), limit=1, comparison=True, frame_runs=runs
        )
        table = evaluation_panel._transient_architecture_table(evaluation, runs)
        assert len(series) == 2
        assert len(table) == 2
        assert len(set(table.index)) == 2
        assert all(label.startswith("FNO") for label in table.index)  # the shared label, each made unique by its run
    finally:
        evaluation.close()


def test_the_transient_configuration_keeps_the_channel_counts_of_each_run(tmp_path: Path) -> None:
    """The input and output channel counts of the record identity stay rows of the configuration, beside the architecture values."""
    evaluation = _two_run_session(tmp_path)
    try:
        table = evaluation_panel._transient_architecture_table(evaluation)
        assert set(table["Data · Input channels"]) == {12}
        assert set(table["Data · Output channels"]) == {4}
        assert set(table["Architecture · Hidden channels"]) == {8}
    finally:
        evaluation.close()


def test_the_transient_configuration_reads_the_stage_from_the_artifact_stage_identity_without_a_saved_run_config(tmp_path: Path) -> None:
    """Without a saved run config the stage comes from the record's stage identity, a mapping, and reads upper case."""
    evaluation = _two_run_session(tmp_path, stage_identity={"stage": "b", "comparison_arm": "b"})
    try:
        assert set(evaluation_panel._transient_architecture_table(evaluation)["Stage"]) == {"B"}
    finally:
        evaluation.close()


def test_the_airflow_substitution_view_takes_the_evidence_of_its_own_frame_when_an_earlier_frame_has_none() -> None:
    """The session skips a frame without a finite B/A error, so the evidence is matched by frame, never by position."""
    fake = SimpleNamespace(
        frame_names=("A0 run ID", "B run ID"),
        selection_objective=lambda name: math.nan if name == "A0 run ID" else 0.5,
        pipeline_degradation=lambda: ("evidence of B",),
    )
    assert evaluation_panel._pipeline_evidence(cast("Any", fake)) == {"B run ID": "evidence of B"}


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


def test_outlier_field_view_ranks_the_cases_of_the_chosen_dataset(
    tmp_path: Path,
) -> None:
    """The worst cases of another dataset never leave the chosen one empty: the outlier view ranks within the chosen dataset."""
    lentil = (
        sequence_record("autonomous_full", "full", case_id=f"lentil-{index}", material_family="lentil", offset=2.0 + index)
        for index in range(analysis.evaluation.panel._OUTLIER_COUNT)
    )
    records = (sequence_record("autonomous_full", "full", case_id="chickpea-mild", material_family="chickpea", offset=1.0), *lentil)
    evaluation = session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path / "artifact"))})
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(session=evaluation, fields=fields, comparison=False, outlier=True)
    # One-of datasets start on the first inventory item (Chickpea, ID), while every case of a ranking over both is a lentil case.
    assert "chickpea-mild" in viewer.status.text
    # Choosing lentil ranks again: its worst case comes first, not its first case in inventory order.
    lentil_choice = next(
        widget for widget in _widget_tree(viewer.widget) if isinstance(widget, widgets.ToggleButton) and widget.description == "Lentil, ID"
    )
    lentil_choice.value = True
    assert f"lentil-{analysis.evaluation.panel._OUTLIER_COUNT - 1}" in viewer.status.text
    evaluation.close()
    plt.close("all")


def test_a_comparison_opens_the_aggregate_scope_and_relative_norm_views_on_every_dataset_like_one_model(
    tmp_path: Path,
) -> None:
    """
    Comparison or not, the aggregate, scope and relative-norm views open on every dataset: checkboxes, all checked, all drawn.

    Runs A and B each hold a chickpea and a lentil case. Compared, the map headers, the predicted trajectories and the norm
    curves name each run on each dataset; run A alone names each dataset.
    """
    datasets = {"Chickpea, ID", "Lentil, ID"}

    def dataset_widgets(view: analysis.ui.panel.ViewPanel) -> list[Any]:
        return [widget for widget in _widget_tree(view.widget) if getattr(widget, "description", None) in datasets]

    for comparison, runs in ((True, ("A", "B")), (False, ("A",))):
        frames = {}
        for run in runs:
            records = tuple(
                sequence_record("autonomous_full", "full", case_id=material, material_family=material) for material in ("chickpea", "lentil")
            )
            for record in records:
                record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": run.lower() * 64}
            frames[f"{run} ID"] = production_frame(records, artifact_root=str(tmp_path / f"{run}-{comparison}"))
            frames[f"{run} ID"].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = run.lower() * 64
        evaluation = session.TransientEvaluationSession(frames)
        fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
        frame_runs = {name: evaluation_panel.FrameRun(name.removesuffix(" ID"), named=True) for name in frames}
        states: tuple[dict[str, Any], ...] = ({}, {}, {})
        views = (
            evaluation_panel._make_transient_aggregate_viewer(
                session=evaluation, fields=fields, plot_kind="mean_error", comparison=comparison, export_state=states[0], frame_runs=frame_runs
            ),
            evaluation_panel._make_transient_scope_viewer(
                session=evaluation, fields=fields, plot_kind="trajectory", comparison=comparison, export_state=states[1], frame_runs=frame_runs
            ),
            evaluation_panel._transient_relative_norm_entry(evaluation, comparison=comparison, selection_state=None, frame_runs=frame_runs)(
                export_state=states[2]
            ),
        )
        shown = {f"{run} · {dataset}" for run in runs for dataset in datasets} if comparison else datasets
        try:
            for view in views:
                offered = dataset_widgets(view)
                assert len(offered) == len(datasets)
                assert all(isinstance(widget, widgets.Checkbox) and widget.value for widget in offered), comparison
            maps, trajectories, norms = (state["fig"] for state in states)
            headers = {axis.get_title().removesuffix(" · n(t) = 1") for axis in maps.axes if axis.get_title().endswith(" · n(t) = 1")}
            assert headers == {label.replace(" · ", "\n", 1) for label in shown}  # a map header puts its run on its own line
            predictions = {line.get_label() for axis in trajectories.axes for line in axis.get_lines()}
            assert {label.removesuffix(" · Prediction") for label in predictions if label.endswith(" · Prediction")} == shown
            assert {line.get_label() for line in norms.axes[0].get_lines() if not line.get_label().startswith("_")} == shown
        finally:
            plt.close("all")
            evaluation.close()


def test_horizon_error_uses_one_global_axis_for_sparse_frame_inventories() -> None:
    """Align nonidentical frame horizons to shared ticks instead of local positions; a line breaks at a horizon inside its span it lacks."""
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

    summary = _with_provenance(summary)
    summary["contributing_record_count"] = 1
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    horizon_axis = figure.axes[0]
    lines = {line.get_label(): line for line in horizon_axis.lines}

    assert [label.get_text() for label in horizon_axis.get_xticklabels()] == ["1", "2", "full"]
    # Each series names its run (here its frame key, no run label given), then its material and role from the row, never from the key.
    np.testing.assert_array_equal(lines["A · Lentil, ID — cumulative"].get_xdata(), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(lines["A · Lentil, ID — cumulative"].get_ydata(), np.asarray([0.1, np.nan, 0.3]))
    np.testing.assert_array_equal(lines["B · Lentil, ID — cumulative"].get_xdata(), np.asarray([1, 2]))
    # A horizon a frame has no row for is a gap as well: the note lists it, as it lists a horizon without a case.
    note = _figure_note(figure)
    assert "no case at horizon 2: A · Lentil, ID" in note
    assert "no case at horizon 1: B · Lentil, ID" in note
    plt.close(figure)


def _with_provenance(summary: pd.DataFrame, *, role: str = "id", run: str = "run", model_kind: str = "fno") -> pd.DataFrame:
    """Return hand-made summary rows with the provenance columns a session frame carries: role, run checkpoint and model kind."""
    return summary.assign(dataset_role=role, run_name=run, checkpoint_sha256=f"{run}-checkpoint", model_kind=model_kind)


def test_evaluation_physical_time_axes_reuse_shared_units_and_export() -> None:
    """Format simulated time centrally while leaving rollout indices distinct."""
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
    summary = _with_provenance(summary)
    summary["contributing_record_count"] = 1
    horizon = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    reached = []
    for reference_time, predicted_time in ((24.0, 48.0), (168.0, 144.0)):
        record = sequence_record("autonomous_full", "full")
        record.target.update(reference_available=True, reference_time_to_target=reference_time, predicted_time_to_target=predicted_time)
        reached.append(record)
    targets = analysis.evaluation.plots.transient.plot_target_time(tuple(reached), styles=_STYLES)
    figures = (horizon, targets)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for figure in figures:
                figure.canvas.draw()
                buffer = BytesIO()
                figure.savefig(buffer, format="pdf", bbox_inches="tight")
                assert buffer.getbuffer().nbytes > 0
        assert horizon.axes[0].get_xlabel() == "Requested horizon [transitions]"
        assert horizon.axes[1].get_xlabel() == "Median elapsed physical time [d]"
        assert targets.axes[0].get_xlabel() == "Reference time to target [h]"  # target times stay in hours over a week
        assert targets.axes[0].get_ylabel() == "Predicted time to target [h]"
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
    assert {key for key, box in boxes.items() if box.value} == {"T", "phi", "w_gr"} & set(boxes)
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
    channel_figure = analysis.evaluation.plots.transient.plot_channel_error(summary_frame, styles=evaluation.series_styles())
    assert channel_figure.axes[0].get_ylabel() == "Normalized RMSE [-]"
    assert len(channel_figure.axes[0].patches) == len(analysis.evaluation.transient_artifact.STATE_ORDER)
    plt.close(channel_figure)

    degradation = evaluation.pipeline_degradation()
    assert len(degradation) == 1
    assert degradation[0].c_available is False
    assert degradation[0].complete_pipeline_error is None
    pipeline = analysis.evaluation.plots.transient.plot_pipeline_degradation(degradation, role="id", models=("A",))
    stages = pipeline.axes[0]
    # An unavailable C has neither bar nor key: B against A is the one narrow bar, named by its tick; no legend repeats it.
    assert not pipeline.legends
    assert stages.get_legend() is None
    assert [text.get_text() for text in stages.get_xticklabels()] == ["B vs A\ndrying surrogate"]
    (bar,) = stages.patches
    assert cast("Any", bar).get_width() < 0.5
    # The metric is named by its display label; the title names the run, the note the conditions, scope, upstream error and C's reason.
    assert stages.get_ylabel() == "Drying group macro RMSE [-]"
    assert _figure_title(pipeline) == "Airflow substitution evidence (A)"
    note = _figure_note(pipeline)
    reason = analysis.evaluation.plots.transient.unavailable_reason_text(degradation[0].c_unavailable_reason)  # the reason in words
    for part in ("not training stages", "ID cases", "Upstream airflow error unavailable", f"C unavailable: {reason}"):
        assert part in " ".join(note.split()), part
    assert not stages.texts
    stacked = analysis.evaluation.plots.transient.plot_pipeline_degradation(degradation * 2)
    assert not stacked.legends
    assert _figure_note(stacked).count(f"C unavailable: {reason}") == 1  # a line every panel shares stands once, unnamed
    assert "Panel" not in _figure_note(stacked)
    evaluation.close()


def test_the_airflow_substitution_panels_share_one_value_scale_and_state_each_distinct_line_once() -> None:
    """Panels of 0.2, 1.5 and 0.4 share one y scale, so heights compare; the shared C reason stands once, each upstream error by its panels."""
    build = analysis.evaluation.transient_comparison.build_airflow_degradation_metrics
    evidence = [
        build(
            metric_id="normalized_drying_group_macro_rmse",
            drying_surrogate_error=value,
            complete_pipeline_error=None,
            airflow_substitution_discrepancy=None,
            upstream_airflow_error=upstream,
            unavailable_reason="compatible_airflow_checkpoint_not_selected",
        )
        for value, upstream in ((0.2, None), (1.5, None), (0.4, 0.25))
    ]
    semantics = analysis.presentation.visual_semantics
    styles = [semantics.series_style(material="lentil", role=role) for role in ("id", "material_ood", "id")]
    figure = analysis.evaluation.plots.transient.plot_pipeline_degradation(evidence, labels=("A · ID", "A · Material OOD", "B · ID"), styles=styles)
    assert [axis.patches[0].get_hatch() for axis in figure.axes] == [style.hatch for style in styles]  # the role hatch, as the global error bars
    assert len({axis.get_ylim() for axis in figure.axes}) == 1
    assert figure.axes[0].get_ylim()[1] > 1.5
    note = _figure_note(figure)
    assert note.count("C unavailable: Compatible airflow checkpoint not selected") == 1  # every panel's, so unnamed
    assert "A · ID, A · Material OOD: upstream airflow error unavailable" in note
    assert "B · ID: upstream airflow error 0.250" in note


def test_an_unavailability_reason_reads_its_identifier_tokens_in_words_and_keeps_recorded_prose() -> None:
    """A recorded token reads in words; recorded prose stays as written, an identifier inside it included."""
    reason = "compatible_airflow_checkpoint_not_selected; No compatible checkpoint for run_b"
    text = analysis.evaluation.plots.transient.unavailable_reason_text(reason)
    assert text == "Compatible airflow checkpoint not selected; No compatible checkpoint for run_b"


def test_the_airflow_substitution_view_draws_each_run_with_evidence_in_its_colour_and_names_every_runs_gap(tmp_path: Path) -> None:
    """Runs B and C have no finite B/A error: beside A, A's bars take A's colour and the note names B; alone, the view names both."""
    frames = {}
    for name, checkpoint in (("A ID", "a" * 64), ("B ID", "b" * 64), ("C ID", "c" * 64)):
        record = sequence_record("autonomous_full", "full", offset=2.0 if name == "A ID" else np.nan)
        record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
        frames[name] = production_frame((record,), artifact_root=str(tmp_path / name))
        frames[name].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = checkpoint
    gap = "no finite B/A drying-surrogate error, so no evidence"
    for names in (("A ID", "B ID"), ("B ID", "C ID")):
        evaluation = session.TransientEvaluationSession({name: frames[name] for name in names})
        state: dict[str, Any] = {}
        runs = {name: evaluation_panel.FrameRun(name.removesuffix(" ID"), named=True) for name in names}
        view = evaluation_panel._make_transient_runtime_viewer(session=evaluation, kind="pipeline", export_state=state, frame_runs=runs)
        try:
            if "A ID" in names:
                (axis,) = state["fig"].axes
                assert axis.get_title() == "A · ID"
                assert {to_hex(patch.get_facecolor()) for patch in axis.patches} == {to_hex(evaluation.series_styles()["A ID"].color)}
                assert f"B · ID: {gap}" in _figure_note(state["fig"])
            else:  # no run to draw: an empty view that names each run's gap instead of a failure
                assert state.get("fig") is None
                assert all(f"{run} · ID: {gap}" in _shown_text(view) for run in ("B", "C"))
        finally:
            plt.close("all")
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
    trajectory = transient.plot_state_trajectory_summary({"case": [record]}, state_fields=("w_gr",), styles=_STYLES)
    lines = {line.get_label(): cast("Any", line.get_ydata()) for line in trajectory.axes[0].get_lines()}

    def bulk(water: np.ndarray, cells: np.ndarray) -> float:
        return float(100.0 * water[cells].sum() / (rho[cells].sum() + water[cells].sum()))

    # The reference averages its own valid cells (here all), the prediction and the difference row the cells valid in both.
    assert lines["case · Reference"][1] == pytest.approx(bulk(grain_reference, np.ones(valid.shape, dtype=bool)))
    assert lines["case · Prediction"][1] == pytest.approx(bulk(grain_prediction, valid))
    difference = cast("Any", trajectory.axes[1].get_lines()[0].get_ydata())
    assert difference[1] == pytest.approx(bulk(grain_prediction, valid) - bulk(grain_reference, valid))
    assert "cells valid in both" in _figure_note(trajectory)
    assert trajectory.axes[0].get_ylabel() == r"$X_{\mathrm{wb}}$ [%]"
    plt.close(trajectory)
    aggregate = transient.plot_aggregate_error_maps([record], state_fields=("w_gr",), physical_time=1.0, statistic="mean")
    pooled = np.asarray(cast("Any", aggregate.axes[0].collections[0].get_array()))
    np.testing.assert_allclose(pooled[valid], prediction_percent - reference_percent[valid])  # the default mean map is the signed one
    assert np.isnan(pooled[~valid]).all()
    plt.close(aggregate)
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("w_gr",), styles=_STYLES)
    assert not trend.axes[0].texts  # the count stands in the note, off the trend data
    assert _trend_count("w_gr", 4, "negative water masked") in _figure_note(trend)  # display-masked water, not a raw non-finite value
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
    trajectory = transient.plot_state_trajectory_summary({"case": [record]}, state_fields=("phi",), styles=_STYLES)
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
    _x, _y, mean_error, signed_mean, std_error, _reference_means, prediction_means = transient._aggregate_snapshot_evidence(
        [record], state_field="T", physical_times=(1.0,)
    )
    assert np.isnan(mean_error[3, 3])
    assert np.isnan(signed_mean[3, 3])
    assert np.isnan(std_error[3, 3])
    assert np.count_nonzero(np.isfinite(mean_error)) == mean_error.size - 1
    assert not np.isfinite(prediction_means).any()
    plt.close(transient.plot_aggregate_error_maps([record], state_fields=("T",), physical_time=1.0, statistic="mean"))
    parity = transient.plot_predicted_vs_reference_channels([record], state_fields=("T",), physical_time=1.0, styles=_STYLES)
    assert any(text.get_text().startswith("1 non-finite") for text in parity.axes[0].texts)
    plt.close(parity)
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T",), styles=_STYLES)
    assert _trend_count("T", 2) in _figure_note(trend)
    plt.close(trend)
    record.predicted_states[1:, 0] = np.nan
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T",), styles=_STYLES)
    (line,) = trend.axes[0].get_lines()
    assert _trend_count("T", 72) in _figure_note(trend)
    assert len(cast("Any", line.get_ydata())) == 0
    plt.close(trend)


def test_a_raw_non_finite_water_cell_voids_its_trajectory_time_while_the_parity_keeps_the_case_on_its_paired_cells() -> None:
    """One predicted water cell of t = 1 h is non-finite: the trajectory marks that time void, the parity keeps the case's bulk means."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[1, analysis.evaluation.transient_artifact.STATE_ORDER.index("w_surf"), 0, 0] = np.nan
    rho = np.asarray(record.static_conditioning[_RHO_CHANNEL], dtype=np.float64)
    sides = (record.reference_states, record.predicted_states)
    grain = [transient._grain_moisture(record, np.asarray(states, dtype=np.float64))[1] for states in sides]
    paired = np.isfinite(grain[1])  # every cell but the non-finite one
    parity = transient.plot_predicted_vs_reference_channels([record], state_fields=("w_gr",), physical_time=1.0, styles=_STYLES)
    (points,) = parity.axes[0].collections
    bulk = [100.0 * water[paired].sum() / (rho[paired].sum() + water[paired].sum()) for water in grain]
    np.testing.assert_allclose(np.asarray(points.get_offsets()), [bulk])  # the case keeps its point on the cells valid in both
    assert not parity.axes[0].texts  # no non-finite case mean to name
    trajectory = transient.plot_state_trajectory_summary({"case": [record]}, state_fields=("w_gr",), styles=_STYLES)
    (cross,) = _crosses(trajectory.axes[1])
    assert list(cross.get_xdata()) == [1.0]
    difference = np.asarray(trajectory.axes[1].get_lines()[0].get_ydata(), dtype=np.float64)
    assert np.isnan(difference[1])
    assert np.isfinite(difference[[0, 2]]).all()


def test_thinned_error_trends_name_the_exact_non_finite_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Divergences between the thinning positions are counted exactly and named, while the finite medians stay as they were."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    monkeypatch.setattr(transient, "_TARGET_MAGNITUDE_POINT_LIMIT", 36)  # 72 post-origin elements: stride 2
    finite_centers, finite_medians, *finite_counts = transient._finite_error_trend([record], "T")
    odd_cells = (np.arange(36).reshape(6, 6) % 2).astype(bool)
    record.predicted_states[1:, 0][:, odd_cells] = np.nan  # every odd raveled element, which stride 2 never samples
    centers, medians, *counts = transient._finite_error_trend([record], "T")
    assert (finite_counts, counts) == ([0, 0, 72], [36, 0, 72])  # raw non-finite, display-masked, total
    np.testing.assert_array_equal(centers, finite_centers)
    np.testing.assert_array_equal(medians, finite_medians)
    trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T",), styles=_STYLES)
    assert _trend_count("T", 36) in _figure_note(trend)
    plt.close(trend)


def test_the_error_trend_note_names_each_run_once_with_each_materials_raw_and_masked_counts_of_its_total() -> None:
    """Run A spans lentil and chickpea, run B lentil: one note line per run, each material with its own counts and denominator."""
    transient = analysis.evaluation.plots.transient
    states = analysis.evaluation.transient_artifact.STATE_ORDER
    lentil_a, chickpea_a, lentil_b = _map_record(), _map_record(), _map_record()
    chickpea_a.identity["material_family"] = "chickpea"
    lentil_b.identity["checkpoint_identity"] = {"best_checkpoint_sha256": "e" * 64}
    lentil_a.predicted_states[1, states.index("w_surf"), 0, 0] = np.nan  # one raw non-finite element
    lentil_a.predicted_states[2, [states.index("w_surf"), states.index("w_int")], 1, 1] = -0.1  # one display-masked element
    chickpea_a.predicted_states[1:, states.index("w_int"), 0, 0] = np.nan  # two
    lentil_b.predicted_states[1, states.index("w_surf"), :, 0] = np.nan  # six
    series = {"A · Lentil, ID": [lentil_a], "A · Chickpea, ID": [chickpea_a], "B · Lentil, ID": [lentil_b]}
    trend = transient.plot_error_vs_target_magnitude(series, state_fields=("w_gr",), styles=_comparison_of(lentil_a, chickpea_a, lentil_b))
    note, symbol = _figure_note(trend), transient._display_symbol("w_gr")
    assert f"A: Lentil, ID ({symbol} 1 non-finite, 1 negative water masked of 72); Chickpea, ID ({symbol} 2 non-finite of 72)" in note
    assert f"B: Lentil, ID ({symbol} 6 non-finite of 72)" in note
    assert "A · " not in note  # the run is named once, never again before each material


def test_the_error_trend_note_lists_its_runs_in_legend_order_wrapped_to_the_panel_width() -> None:
    """
    Run B masks surface water, run A internal water: A's line comes first, as in the legend, though B's first field comes earlier.

    A long line breaks between its words, never inside a symbol, so the note stays within the panel's width instead of widening the figure.
    """
    transient = analysis.evaluation.plots.transient
    states = analysis.evaluation.transient_artifact.STATE_ORDER
    first, second = _map_record(), _map_record()
    second.identity["checkpoint_identity"] = {"best_checkpoint_sha256": "e" * 64}
    first.predicted_states[1, states.index("w_int"), 0, 0] = -0.1
    second.predicted_states[1, states.index("w_surf"), 0, 0] = -0.1
    series = {f"FNO · {run} · seed 9 · rollout-32 · Lentil, ID": [record] for run, record in (("A", first), ("B", second))}
    trend = transient.plot_error_vs_target_magnitude(series, state_fields=("w_surf", "w_int", "w_gr"), styles=_comparison_of(first, second))
    trend.canvas.draw()
    note = [artist for artist in trend.artists if isinstance(artist, Text)][1]
    assert note.get_text().index("FNO · A") < note.get_text().index("FNO · B")
    assert note.get_window_extent().width <= trend.axes[0].get_window_extent().width
    assert f"{transient._display_symbol('w_int')} 1 negative water masked of 72" in _figure_note(trend)


def test_the_error_trend_note_keeps_the_legends_run_order_where_a_runs_first_series_excludes_nothing() -> None:
    """A's lentil series comes first and excludes nothing, then B's lentil and A's chickpea exclude a value each: A's line still comes first."""
    transient = analysis.evaluation.plots.transient
    states = analysis.evaluation.transient_artifact.STATE_ORDER
    lentil_a, lentil_b, chickpea_a = _map_record(), _map_record(), _map_record()
    lentil_b.identity["checkpoint_identity"] = {"best_checkpoint_sha256": "e" * 64}
    chickpea_a.identity["material_family"] = "chickpea"
    for record in (lentil_b, chickpea_a):
        record.predicted_states[1, states.index("w_surf"), 0, 0] = np.nan
    series = {"A · Lentil, ID": [lentil_a], "B · Lentil, ID": [lentil_b], "A · Chickpea, ID": [chickpea_a]}
    trend = transient.plot_error_vs_target_magnitude(series, state_fields=("w_gr",), styles=_comparison_of(lentil_a, lentil_b, chickpea_a))
    legends = [*trend.legends, *(legend for axis in trend.axes if (legend := axis.get_legend()) is not None)]
    keys = [text.get_text() for legend in legends for text in legend.get_texts()]
    assert keys.index("A") < keys.index("B")  # the legend's run headings
    note = _figure_note(trend)
    assert note.index("A: Chickpea, ID") < note.index("B: Lentil, ID")


def test_the_error_trend_runs_over_the_signed_reference_value_named_in_the_active_style() -> None:
    """
    The x data are the displayed reference values with their sign (°C may be negative), labelled «Reference» and the field.

    The label follows the label style and its unit, for a water density and for a plain state alike.
    """
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.reference_states[1:, 0] -= 300.0  # about 0 to 10 K: about -273 °C, so an absolute value would flip the axis
    labels = {}
    try:
        for style in ("symbol", "name"):
            analysis.presentation.field_labels.set_label_style(style)
            trend = transient.plot_error_vs_target_magnitude({"case": [record]}, state_fields=("T", "w_gr"), styles=_STYLES)
            labels[style] = [axis.get_xlabel() for axis in trend.axes]
            if style == "symbol":
                assert (np.asarray(trend.axes[0].get_lines()[0].get_xdata(), dtype=float) < 0.0).all()
            plt.close(trend)
    finally:
        analysis.presentation.field_labels.set_label_style("symbol")
    assert labels["symbol"] == ["Reference value — $T$ [°C]", r"Reference value — $X_{\mathrm{wb}}$ [%]"]
    assert labels["name"] == ["Reference value — Temperature [°C]", "Reference value — Grain moisture [% w.b.]"]


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
    boxes["w_surf"].value = True
    boxes["phi"].value = False
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


def test_comparison_ranking_orders_by_the_selection_objective_and_never_ranks_partial_evidence(tmp_path: Path) -> None:
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
    evaluation = session.TransientEvaluationSession(
        {
            name: production_frame(records, artifact_root=str(tmp_path / name))
            for name, records in (("Incomplete", incomplete), ("Complete", complete))
        }
    )
    ranked = analysis.evaluation.panel._transient_comparison_summary(evaluation)
    assert list(ranked["frame"]) == ["Complete", "Incomplete"]
    # Lentil 4 and chickpea 36 average to 20 per field; every field has Train scale 2.
    assert ranked.iloc[0]["normalized_drying_group_macro_rmse"] == pytest.approx(math.sqrt(20.0) / 2.0)
    assert math.isnan(evaluation.selection_objective("Incomplete"))
    assert pd.isna(ranked.iloc[1]["normalized_drying_group_macro_rmse"])
    assert not ranked.iloc[1]["Metric available"]
    assert list(ranked["Material count"]) == [2, 2]
    # The scoreboard numbers the ranked run only; the unavailable one stands without a rank, as the docs say.
    html = "".join(str(child.value) for child in analysis.evaluation.panel._comparison_summary_view(evaluation).children if hasattr(child, "value"))
    assert "1. Complete" in html
    assert "Incomplete" in html
    assert "2. Incomplete" not in html


def test_the_scoreboard_heads_one_runs_frames_by_run_label_and_role_without_repeating_or_cutting_the_label(tmp_path: Path) -> None:
    """The ID and material-OOD frames of one run read «rank. run · role», as the airflow and runtime views name them."""
    frames = {
        name: production_frame((sequence_record("autonomous_full", "full", offset=offset),), artifact_root=str(tmp_path / role), dataset_role=role)
        for name, role, offset in (("FNO · B · seed 9 · rollout-32 · ID", "id", 1.0), ("FNO · B · seed 9 · rollout-32 · OOD", "material_ood", 2.0))
    }
    evaluation = session.TransientEvaluationSession(frames)
    runs = {name: evaluation_panel.FrameRun("FNO · B · seed 9 · rollout-32", named=True) for name in frames}
    try:
        view = evaluation_panel._comparison_summary_view(evaluation, runs)
        html = "".join(str(child.value) for child in view.children if hasattr(child, "value"))
        assert ">1. FNO · B · seed 9 · rollout-32 · ID<" in html
        assert ">2. FNO · B · seed 9 · rollout-32 · Material OOD<" in html
        assert "…" not in html
    finally:
        evaluation.close()


def test_a_scoreboard_of_one_compared_run_shades_no_row(tmp_path: Path) -> None:
    """With a single ranked run every row is constant, so no fill would compare anything."""
    evaluation = session.TransientEvaluationSession(
        {"Only": production_frame((sequence_record("autonomous_full", "full"),), artifact_root=str(tmp_path))}
    )
    view = analysis.evaluation.panel._comparison_summary_view(evaluation)
    assert "background-color" not in "".join(str(child.value) for child in view.children if hasattr(child, "value"))


def test_the_ranked_scoreboard_shows_the_humidity_errors_in_percentage_points_and_ranks_in_stored_units() -> None:
    """A φ MAE of 0.02 reads 2 under a [pp] column, T keeps its value; the order and the objective stay those of the stored metrics."""
    offsets = {"Worse": 0.04, "Better": 0.02}
    evaluation = session.TransientEvaluationSession(
        {name: sequence_frame((sequence_record("autonomous_full", "full", case_id="a", offset=offset),)) for name, offset in offsets.items()}
    )
    ranked = analysis.evaluation.panel._transient_comparison_summary(evaluation)
    assert list(ranked["frame"]) == ["Better", "Worse"]
    assert list(ranked["normalized_drying_group_macro_rmse"]) == [evaluation.selection_objective(name) for name in ("Better", "Worse")]
    assert list(ranked["physical_mae_phi [pp]"]) == pytest.approx([2.0, 4.0])
    assert list(ranked["physical_rmse_phi [pp]"]) == pytest.approx([2.0, 4.0])
    assert list(ranked["physical_mae_T [K]"]) == pytest.approx([0.02, 0.04])


def test_of_the_transient_tables_the_error_columns_are_shaded_and_the_configuration_table_carries_no_fill(tmp_path: Path) -> None:
    """Two compared runs: the scoreboard and the summary fill their error cells (under their units), the configuration no cell."""
    evaluation = _two_run_session(tmp_path)
    registry = evaluation_panel._build_transient_sections(
        session=evaluation,
        comparison=True,
        selection_state=None,
        evaluation_compute_device="cpu",
        training_histories=dict,
        frame_runs={name: evaluation_panel.FrameRun(name.removesuffix(" ID"), named=True) for name in evaluation.frame_names},
    )
    entries = {title: factory for section_entries, _name in registry.values() for title, factory, _stem in section_entries}
    try:
        titles = ("Global comparison summary", "Summary table", "Model-family configuration")
        html = {title: str(entries[title]().children[1].value) for title in titles}
        assert "background-color" in html["Global comparison summary"]
        assert "background-color" in html["Summary table"]
        assert "[pp]" in html["Global comparison summary"]
        assert "background-color" not in html["Model-family configuration"]
    finally:
        evaluation.close()


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
        metric.update(prediction, torch.zeros_like(prediction), batch_index=index, cases=cases)

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


def _summary_rows(*rows: tuple[str, str, str, str], **columns: Any) -> pd.DataFrame:
    """Return complete-rollout summary rows of (frame, material, run, model kind) with the given metric columns and their provenance."""
    frames = []
    for frame, material, run, kind in rows:
        row = {"frame": frame, "material_family": material, "mode": "autonomous_full", "requested_horizon": "full", "scope": "cumulative"}
        frames.append(_with_provenance(pd.DataFrame([{**row, **columns}]), run=run, model_kind=kind))
    return pd.concat(frames, ignore_index=True)


def _inside(axis: Any, points: np.ndarray) -> bool:
    """Return whether every point lies strictly inside the axis limits, off the frame."""
    (left, right), (bottom, top) = sorted(axis.get_xlim()), sorted(axis.get_ylim())
    return bool(np.all((left < points[:, 0]) & (points[:, 0] < right) & (bottom < points[:, 1]) & (points[:, 1] < top)))


def test_grouped_views_list_a_runs_series_together_in_the_selector_material_order() -> None:
    """Interleaved rows of two runs come out run by run (first appearance), each run's materials alphabetical as in the selector."""
    rows = _summary_rows(
        ("B ID", "lentil", "b", "fno"), ("A ID", "lentil", "a", "fno"), ("B ID", "chickpea", "b", "fno"), ("A ID", "chickpea", "a", "fno")
    )
    groups = analysis.evaluation.plots.transient._summary_series(rows)
    order = [(group["frame"].iloc[0], group["material_family"].iloc[0]) for group in groups.values()]
    assert order == [("B ID", "chickpea"), ("B ID", "lentil"), ("A ID", "chickpea"), ("A ID", "lentil")]


def test_a_spread_map_of_one_case_says_insufficient_support_instead_of_drawing_zero() -> None:
    """One case has no spread: its standard-deviation map is an explicit message, while two cases draw their spread."""
    transient = analysis.evaluation.plots.transient
    low, high = (sequence_record("autonomous_full", "full", case_id=name, offset=offset) for name, offset in (("low", 1.0), ("high", -3.0)))
    single = transient.plot_aggregate_error_maps({"One": (low,)}, state_fields=("T",), physical_time=1.0, statistic="std")
    assert not single.axes[0].collections
    assert any("Insufficient support" in text.get_text() for text in single.axes[0].texts)
    pooled = transient.plot_aggregate_error_maps({"Two": (low, high)}, state_fields=("T",), physical_time=1.0, statistic="std")
    np.testing.assert_allclose(cast("Any", pooled.axes[0].collections[0].get_array()), 2.0)  # the spread of -1 and 3 about their mean


def test_a_single_unequal_parity_pair_stays_inside_the_padded_axes() -> None:
    """A singleton reference/prediction pair no longer sits on both bounds: every nonempty parity range is padded."""
    record = sequence_record("autonomous_full", "full", offset=1.0)
    figure = analysis.evaluation.plots.transient.plot_predicted_vs_reference_channels(
        [record], state_fields=("T",), physical_time=1.0, styles=_STYLES
    )
    axis = figure.axes[0]
    assert _inside(axis, np.asarray(axis.collections[0].get_offsets()))


def test_a_one_case_error_distribution_is_a_point_and_boxes_draw_their_median_in_the_series_colour() -> None:
    """A series of one case is one point in its colour; a series of several keeps its box, whose median line takes its colour."""
    semantics = analysis.presentation.visual_semantics
    cases = _summary_rows(
        ("M ID", "lentil", "m", "fno"),
        ("M ID", "chickpea", "m", "fno"),
        ("M ID", "chickpea", "m", "fno"),
        normalized_rmse_T=0.2,
        normalized_drying_group_macro_rmse=0.1,
    )
    cases.loc[2, "normalized_rmse_T"] = 0.4
    figure = analysis.evaluation.plots.transient.plot_error_distributions(cases, state_fields=("T",), styles=_table_styles(cases))
    axis = figure.axes[0]
    (point,) = axis.collections
    assert to_hex(cast("ColorType", point.get_facecolor()[0])) == to_hex(semantics.material_color("lentil"))
    assert len(axis.patches) == 1  # the chickpea box only
    medians = [line for line in axis.get_lines() if to_hex(line.get_color()) == to_hex(semantics.material_color("chickpea"))]
    assert medians


def test_a_runs_materials_carry_their_marker_on_bars_and_horizon_lines_and_the_scope_is_the_marker_fill() -> None:
    """Of two runs, the one spanning two materials marks each bar with its material marker; horizon lines take the material marker."""
    semantics = analysis.presentation.visual_semantics
    transient = analysis.evaluation.plots.transient
    rows = _summary_rows(
        ("F ID", "lentil", "f", "fno"),
        ("F ID", "chickpea", "f", "fno"),
        ("U ID", "lentil", "u", "uno"),
        normalized_rmse_T=0.2,
        normalized_drying_group_macro_rmse=0.1,
    )
    bars = transient.plot_channel_error(rows, state_fields=("T",), styles=_table_styles(rows)).axes[0]
    assert len(bars.collections) == 2  # one marker per bar of the two-material run, none for the one-material run
    horizon = rows.assign(mode="rolling_origin", requested_horizon=1, elapsed_physical_time_median=1.0, contributing_record_count=1)
    horizon = pd.concat((horizon, horizon.assign(scope="endpoint")), ignore_index=True)
    lines = {line.get_label(): line for line in transient.plot_horizon_error(horizon, styles=_table_styles(rows)).axes[0].get_lines()}
    assert lines["F ID · Lentil, ID — cumulative"].get_marker() == semantics.MATERIAL_MARKERS["lentil"]
    assert lines["F ID · Chickpea, ID — endpoint"].get_marker() == semantics.MATERIAL_MARKERS["chickpea"]
    assert lines["F ID · Chickpea, ID — endpoint"].get_markerfacecolor() == "white"


def test_the_elapsed_horizon_panel_breaks_at_a_missing_horizon_on_padded_limits() -> None:
    """A series without horizon 2 gets a gap there instead of a line joining 1 and full; no point sits on the frame."""
    summary = pd.DataFrame.from_records(
        {
            "frame": frame,
            "material_family": "lentil",
            "mode": "rolling_origin",
            "requested_horizon": horizon,
            "scope": "cumulative",
            "normalized_drying_group_macro_rmse": value,
            "elapsed_physical_time_median": hours,
            "contributing_record_count": 1,
        }
        for frame, horizon, value, hours in (("A", 1, 0.1, 1.0), ("A", "full", 0.3, 3.0), ("B", 2, 0.2, 2.0), ("B", "full", 0.4, 4.0))
    )
    summary = _with_provenance(summary)
    elapsed = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary)).axes[1]
    first = elapsed.get_lines()[0]
    np.testing.assert_array_equal(np.asarray(first.get_xdata(), dtype=float), [1.0, np.nan, 3.0])
    left, right = elapsed.get_xlim()
    assert left < 1.0
    assert right > 4.0


def test_target_time_marks_materials_and_censored_bounds_on_padded_axes_and_says_when_no_pair_reached() -> None:
    """Reached pairs take their material marker filled; a censored pair stands open at its final-time bound; none reached is said."""
    transient = analysis.evaluation.plots.transient
    lentil, chickpea, censored = (
        sequence_record("autonomous_full", "full", case_id=case, material_family=material)
        for case, material in (("l1", "lentil"), ("c1", "chickpea"), ("l2", "lentil"))
    )
    for record, times in ((lentil, (2.0, 1.0)), (chickpea, (1.0, 2.0))):
        record.target.update(reference_reached=True, reference_time_to_target=times[0], predicted_time_to_target=times[1])
    axis = transient.plot_target_time((lentil, chickpea, censored), styles=_STYLES).axes[0]
    collections = {collection.get_label(): collection for collection in axis.collections if not isinstance(collection, Quiver)}
    censored_label = "Lentil, ID · censored, final time as bound"
    assert set(collections) == {"Lentil, ID", "Chickpea, ID", censored_label}
    np.testing.assert_allclose(np.asarray(collections[censored_label].get_offsets()), [[2.0, 1.0]])
    assert _bounds(axis) == {"Reference": [[2.0, 1.0]]}  # the reference side only: the prediction reached
    assert "Reference not reached: final time, a lower bound" in [line.get_label() for line in axis.get_lines()]  # the arrow's key
    assert _inside(axis, np.concatenate([np.asarray(collection.get_offsets()) for collection in axis.collections]))
    assert not axis.texts
    none_reached = transient.plot_target_time((censored,), styles=_STYLES).axes[0]
    assert [text.get_text() for text in none_reached.texts] == ["No case pair reached the target"]


def test_a_single_case_draws_no_band_and_no_band_legend_entry() -> None:
    """Below two cases the 10th-90th bounds are NaN: the error over time draws no band and its legend names none."""
    figure = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": (_timed_record("a", 20),)}, scaling_states={"ID": scaling_state()}, state_fields=("T",), styles=_STYLES
    )
    assert not figure.axes[0].collections
    assert not [text.get_text() for legend in figure.legends for text in legend.get_texts() if "10th-90th" in text.get_text()]


def test_series_with_one_reference_cohort_draw_it_once_in_a_neutral_colour() -> None:
    """Two runs on the same cases share their reference: one neutral curve under one short key, each prediction in its run colour."""
    first, second = _timed_record("a", 20), _timed_record("a", 20)
    first.identity["checkpoint_identity"] = {"best_checkpoint_sha256": "b"}
    second.identity["checkpoint_identity"] = {"best_checkpoint_sha256": "c"}
    figure = analysis.evaluation.plots.transient.plot_state_trajectory_summary(
        {"B": (first,), "A1": (second,)}, state_fields=("T",), styles=_comparison_of(first, second)
    )
    references = [line for line in figure.axes[0].get_lines() if str(line.get_label()).endswith("Reference")]
    assert [(line.get_label(), line.get_color()) for line in references] == [("Reference", "0.55")]  # never the run labels joined


def test_accuracy_runtime_points_keep_their_case_material_marker() -> None:
    """Each paired case keeps its material through the pairing: one marked series per material, named by material and role."""
    semantics = analysis.presentation.visual_semantics
    accuracy = _summary_rows(("M ID", "lentil", "m", "fno"), ("M ID", "chickpea", "m", "fno"), normalized_drying_group_macro_rmse=0.1)
    accuracy["case_id"] = ["c0", "c1"]
    report = _paired_timing_report((0.3, 0.5))
    figure = analysis.evaluation.plots.transient.plot_accuracy_runtime_comparison(
        accuracy.assign(frame="M"), {"M": report}, styles=_table_styles(accuracy.assign(frame="M")), measure="model"
    )
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Chickpea, ID", "Lentil, ID"]
    markers = [cast("Any", handle).get_marker() for handle in legend.legend_handles]
    assert markers == [semantics.MATERIAL_MARKERS[material] for material in ("chickpea", "lentil")]
    assert "ID: 2/2 cases paired" in _figure_note(figure)


def test_a_runtime_legend_names_each_run_once_beside_a_material_key_when_its_points_differ_only_in_marker() -> None:
    """Run A on lentil ID and chickpea material OOD, B on both ID: each run is one swatch over all its roles, the materials a grey key."""
    semantics = analysis.presentation.visual_semantics
    rows = (("A ID", "lentil", "a", "fno"), ("A OOD", "chickpea", "a", "fno"), ("B", "lentil", "b", "fno"), ("B", "chickpea", "b", "fno"))
    accuracy = _summary_rows(*rows, normalized_drying_group_macro_rmse=0.1)
    accuracy.loc[accuracy["frame"] == "A OOD", "dataset_role"] = "material_ood"
    accuracy["case_id"] = ["c0", "c1", "c0", "c1"]
    reports = {frame: _paired_timing_report((0.3, 0.5)) for frame in ("A ID", "A OOD", "B")}
    figure = analysis.evaluation.plots.transient.plot_accuracy_runtime_comparison(
        accuracy, reports, styles=_table_styles(accuracy), measure="model", run_labels={"A ID": "FNO a", "A OOD": "FNO a", "B": "FNO b"}
    )
    legend = figure.axes[0].get_legend()
    assert legend is not None
    legend_rows = zip(legend.get_texts(), legend.legend_handles, strict=True)
    entries = {text.get_text(): cast("Any", handle) for text, handle in legend_rows if text.get_text().strip()}  # the headings, not the gap row
    assert set(entries) == {"Run", "FNO a", "FNO b", "Material", "Chickpea", "Lentil"}
    assert all(isinstance(entries[run], Patch) for run in ("FNO a", "FNO b"))  # a swatch reads as no material marker
    assert to_hex(entries["FNO a"].get_facecolor()) != to_hex(entries["FNO b"].get_facecolor())
    assert "FNO a · Material OOD: 1/1 cases paired" in _figure_note(figure)  # the role stays in the pairing note
    assert entries["Chickpea"].get_marker() == semantics.MATERIAL_MARKERS["chickpea"]
    assert to_hex(entries["Lentil"].get_color()) == to_hex("0.55")


def test_the_grey_material_key_follows_the_selector_order_of_role_group_then_material() -> None:
    """Runs A and B over chickpea and lentil ID and kidney bean material OOD: the key reads Chickpea, Lentil, Kidney bean, never alphabetically."""
    datasets = (("ID", "chickpea", "c0"), ("ID", "lentil", "c1"), ("OOD", "kidney_bean", "c0"))
    rows = ((f"{run} {role}", material, run.lower(), "fno") for run in "AB" for role, material, _case in datasets)
    accuracy = _summary_rows(*rows, normalized_drying_group_macro_rmse=0.1)
    accuracy["case_id"] = [case for _run in "AB" for _role, _material, case in datasets]
    accuracy.loc[accuracy["frame"].str.endswith("OOD"), "dataset_role"] = "material_ood"
    reports = {frame: _paired_timing_report((0.3, 0.5)) for frame in dict.fromkeys(accuracy["frame"])}
    figure = analysis.evaluation.plots.transient.plot_accuracy_runtime_comparison(
        accuracy, reports, styles=_table_styles(accuracy), measure="model", run_labels={frame: f"FNO {frame[0]}" for frame in reports}
    )
    legend = figure.axes[0].get_legend()
    assert legend is not None
    texts = [text.get_text() for text in legend.get_texts() if text.get_text().strip()]  # the headings and entries, not the gap rows
    assert texts[texts.index("Material") + 1 :] == ["Chickpea", "Lentil", "Kidney bean"]


def test_duration_ratio_rows_have_short_labels_half_row_margins_and_their_summed_ratios_in_the_note() -> None:
    """Tick labels name the ratio and paired count only; the note gives each ratio of summed durations; rows keep half a row of margin."""
    report = _paired_timing_report((0.3, 0.5, 1.1))
    figure = analysis.evaluation.plots.transient.plot_timing_speedups(report)
    axis = figure.axes[0]
    assert axis.get_ylim() == (len(report.speedups) - 0.5, -0.5)
    summary = next(iter(report.speedups.values()))
    assert all(summary.reference_boundary not in label.get_text() for label in axis.get_yticklabels())
    note = _figure_note(figure)  # the panel's status carries the full boundaries
    assert note.startswith("Ratio of summed durations: ")
    summed = [value for value in report.speedups.values() if value.ratio_of_sums is not None]
    assert note.count(";") == len(summed) - 1
    # The names read in sentence case, the solver in its registered spelling, in the tick and in the note.
    assert any(label.get_text().startswith("COMSOL process\n") for label in axis.get_yticklabels())
    assert any(label.get_text().startswith("Drying only solver\n") for label in axis.get_yticklabels())
    assert not any("Comsol" in text.get_text() for text in figure.findobj(Text))


def test_target_time_marks_each_bound_by_its_direction_so_an_observed_time_never_reads_as_a_bound() -> None:
    """Only a side that missed the target carries a bound mark, right for the reference and up for the prediction."""
    prediction_only, both = (sequence_record("autonomous_full", "full", case_id=case) for case in ("prediction_only", "both"))
    prediction_only.target.update(reference_time_to_target=24.0, predicted_time_to_target=None, predicted_final_time=30.0)
    both.target.update(reference_final_time=20.0, predicted_time_to_target=None, predicted_final_time=36.0)
    axis = analysis.evaluation.plots.transient.plot_target_time((prediction_only, both), styles=_STYLES).axes[0]
    assert _bounds(axis) == {"Reference": [[20.0, 36.0]], "Prediction": [[24.0, 30.0], [20.0, 36.0]]}
    assert axis.get_xlabel() == "Reference time to target [h]"  # in hours like every time axis, also beyond three days


def _bounds(axis: Any) -> dict[str, list[list[float]]]:
    """Return the points of each lower-bound arrow set of a target plot by its direction: right for the reference, up for the prediction."""
    bounds = {}
    for arrows in (collection for collection in axis.collections if isinstance(collection, Quiver)):
        side = "Reference" if (float(np.asarray(arrows.U).flat[0]), float(np.asarray(arrows.V).flat[0])) == (1.0, 0.0) else "Prediction"
        bounds[side] = np.asarray(arrows.get_offsets()).tolist()
    return bounds


def test_the_horizon_lines_break_at_a_requested_horizon_no_series_supports() -> None:
    """Horizon 4 has only zero-support rows: it keeps its place, so both panels' lines break there instead of joining 2 and full."""
    summary = pd.DataFrame.from_records(
        {
            "frame": "A",
            "material_family": "lentil",
            "mode": "rolling_origin",
            "requested_horizon": horizon,
            "scope": "cumulative",
            "normalized_drying_group_macro_rmse": value,
            "elapsed_physical_time_median": hours,
            "contributing_record_count": count,
        }
        for horizon, value, hours, count in ((2, 0.1, 1.0, 1), (4, math.nan, math.nan, 0), ("full", 0.3, 3.0, 1))
    )
    summary = _with_provenance(summary)
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    index, elapsed = figure.axes[:2]
    assert [label.get_text() for label in index.get_xticklabels()] == ["2", "4\nno case", "full"]  # the tick says why it breaks
    np.testing.assert_array_equal(np.asarray(index.get_lines()[0].get_ydata(), dtype=float), [0.1, np.nan, 0.3])
    np.testing.assert_array_equal(np.asarray(elapsed.get_lines()[0].get_xdata(), dtype=float), [1.0, np.nan, 3.0])


@pytest.mark.parametrize(("largest", "log"), [(150.0, False), (200.0, False), (200.5, True)])
def test_an_axis_turns_log_only_where_a_runs_largest_value_exceeds_20_times_every_other_runs_largest_group_median(largest: float, log: bool) -> None:
    """B's plotted groups have the medians 1, 1 and 10 (pooled median 1): A's largest value turns the axis log only beyond 200."""
    figure, axis = plt.subplots()
    axis.set_ylabel("Error")
    groups = {"A": [np.asarray([1.0, largest])], "B": [np.asarray([1.0]), np.asarray([1.0]), np.asarray([0.0, 10.0, 12.0])]}
    note = analysis.evaluation.plots.transient._log_if_wide(axis, groups)
    assert (axis.get_yscale(), note is not None, axis.get_ylabel().endswith("(log scale)")) == ("log" if log else "linear", log, log)
    assert not log or "1 non-positive values cannot be shown" in str(note)  # B's zero, named instead of clipped
    plt.close(figure)


def _endpoint_rows(pairs: Mapping[str, Sequence[tuple[float, float]]]) -> pd.DataFrame:
    """Return rolling-origin summary rows of lentil, one run per frame, with each (cumulative, endpoint) pair at horizons 1, 2, ..."""
    rows = [
        _with_provenance(
            pd.DataFrame.from_records(
                {
                    "frame": frame,
                    "material_family": "lentil",
                    "mode": "rolling_origin",
                    "requested_horizon": horizon,
                    "scope": scope,
                    "normalized_drying_group_macro_rmse": value,
                }
                for horizon, pair in enumerate(values, start=1)
                for scope, value in zip(("cumulative", "endpoint"), pair, strict=True)
            ),
            run=frame,
        )
        for frame, values in pairs.items()
    ]
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize(
    ("pairs", "scale", "limits"),
    [
        ({"A": ((0.1, 0.2), (0.3, math.inf)), "B": ((0.2, 0.1), (0.4, 0.5))}, "linear", (0.0, 0.525)),
        ({"A": ((0.01, 0.02), (0.3, math.inf)), "B": ((5.0, 8.0), (6.0, 4.0))}, "log", (0.008, 10.0)),
    ],
    ids=["linear", "log"],
)
def test_the_endpoint_plot_leaves_out_and_counts_a_non_finite_pair_and_keeps_one_padded_range_on_both_axes_after_its_key(
    pairs: Mapping[str, Sequence[tuple[float, float]]], scale: str, limits: tuple[float, float]
) -> None:
    """A's second pair is infinite: never drawn, it is counted; both axes share one range from zero, or on log padded round the values."""
    rows = _endpoint_rows(pairs)
    figure = analysis.evaluation.plots.transient.plot_endpoint_vs_cumulative(rows, styles=_table_styles(rows))
    axis = figure.axes[0]
    assert len(figure.axes) == 1  # the key stands inside the one panel, so the legend fitter has run on its limits
    assert (axis.get_xscale(), axis.get_yscale()) == (scale, scale)
    assert axis.get_xlim() == axis.get_ylim()
    assert axis.get_xlim() == pytest.approx(limits)
    points = np.concatenate([np.asarray(collection.get_offsets(), dtype=np.float64) for collection in axis.collections])
    assert np.isfinite(points).all()
    assert len(points) == 3
    assert "1 pair with an unavailable or non-finite error not shown" in _figure_note(figure)
    plt.close(figure)


def test_the_endpoint_plot_counts_a_pair_without_a_finite_side_and_a_series_without_any_endpoint_instead_of_dropping_them() -> None:
    """A's second pair is unavailable on both sides and every endpoint of B is: none is drawn, all three are counted and B is named."""
    rows = _endpoint_rows({"A": ((0.1, 0.2), (math.nan, math.nan)), "B": ((0.2, math.nan), (0.4, math.nan))})
    figure = analysis.evaluation.plots.transient.plot_endpoint_vs_cumulative(rows, styles=_table_styles(rows))
    points = np.concatenate([np.asarray(collection.get_offsets(), dtype=np.float64) for collection in figure.axes[0].collections])
    np.testing.assert_allclose(points, [[0.1, 0.2]])
    note = _figure_note(figure)
    assert "3 pairs with an unavailable or non-finite error not shown" in note
    assert "B: no paired endpoint and cumulative evidence" in note
    plt.close(figure)


@pytest.mark.parametrize(
    ("voided", "side", "expected"),
    [
        ((("B", 0), ("A1", 0), ("A1", 1)), "predicted_states", ["Reference"]),
        ((("B", 0), ("A1", 1)), "predicted_states", ["Reference"]),
        ((("B", 0), ("A1", 0), ("A1", 1)), "reference_states", ["B · Reference", "A1 · Reference"]),
        ((("B", 0), ("A1", 1)), "reference_states", ["B · Reference", "A1 · Reference"]),
    ],
    ids=["prediction-counts", "prediction-cases", "reference-counts", "reference-cases"],
)
def test_series_share_their_reference_unless_the_reference_itself_differs(
    voided: tuple[tuple[str, int], ...], side: str, expected: list[str]
) -> None:
    """
    Same cases and values, but at one time one run voids the grain moisture of case a and the other of both cases, or of case b only.

    Voided predictions leave the reference alone, so the runs share one reference key; voiding the reference water of b where
    the other run voids a leaves the window's cases, summaries and counts equal bit for bit, and still not one reference.
    """
    records = {label: (_timed_record("a", 20), _timed_record("b", 20)) for label in ("B", "A1")}
    for label, checkpoint in (("B", "b"), ("A1", "c")):
        for record in records[label]:
            record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": checkpoint}
    states = analysis.evaluation.transient_artifact.STATE_ORDER
    for label, case in voided:
        getattr(records[label][case], side)[5, [states.index("w_surf"), states.index("w_int")]] = np.nan
    figure = analysis.evaluation.plots.transient.plot_state_trajectory_summary(
        records, state_fields=("w_gr",), styles=_comparison_of(*records["B"], *records["A1"])
    )
    references = [line.get_label() for line in figure.axes[0].get_lines() if str(line.get_label()).endswith("Reference")]
    assert references == expected


def test_a_second_run_never_changes_the_reference_curve_and_two_shared_cohorts_of_one_dataset_key_their_runs() -> None:
    """
    Run A alone and A beside B draw one reference curve; runs C and D share another reference of the same dataset, keyed apart.

    The predicted water of t = 1 h is negative in one cell, which the paired cells leave out but the reference curve keeps.
    """
    transient = analysis.evaluation.plots.transient
    states = analysis.evaluation.transient_artifact.STATE_ORDER
    records = {label: _map_record() for label in "ABCD"}
    for label, record in records.items():
        record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": label.lower() * 64}
        record.predicted_states[1, [states.index("w_surf"), states.index("w_int")], 0, 0] = -0.1
    for label in "CD":  # another cohort: its reference water of t = 2 h is non-finite in one cell
        records[label].reference_states[2, states.index("w_surf"), 5, 5] = np.nan
    styles = _comparison_of(*records.values())

    def references(labels: str) -> dict[str, np.ndarray]:
        figure = transient.plot_state_trajectory_summary({label: (records[label],) for label in labels}, state_fields=("w_gr",), styles=styles)
        lines = figure.axes[0].get_lines()
        return {str(line.get_label()): np.asarray(line.get_ydata(), dtype=np.float64) for line in lines if "Reference" in str(line.get_label())}

    alone, beside, cohorts = references("A"), references("AB"), references("ABCD")
    np.testing.assert_array_equal(alone["A · Reference"], beside["Reference"])
    assert list(cohorts) == ["Reference · A, B", "Reference · C, D"]
    np.testing.assert_array_equal(cohorts["Reference · A, B"], alone["A · Reference"])


def test_an_isolated_finite_band_interval_is_drawn_as_a_vertical_whisker_under_the_band_key() -> None:
    """Two cases share one post-origin hour only: their 10th-90th interval there is a visible vertical whisker, keyed once."""
    early, late = (sequence_record("autonomous_full", "full", case_id=case, offset=offset) for case, offset in (("early", 1.0), ("late", 3.0)))
    early.physical_times, late.physical_times = np.asarray([0.0, 2.0, 4.0]), np.asarray([0.0, 3.0, 4.0])
    figure = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": (early, late)}, scaling_states={"ID": scaling_state()}, state_fields=("T",), styles=_STYLES
    )
    (band,) = [collection for collection in figure.axes[0].collections if isinstance(collection, LineCollection)]
    ((start, end),) = band.get_segments()
    assert start[0] == end[0]
    assert end[1] > start[1]
    assert [text.get_text() for legend in figure.legends for text in legend.get_texts()].count("Per-case RMSE 10th-90th") == 1


def test_a_runs_frames_of_several_roles_list_their_series_in_the_selector_order_of_role_group_then_material() -> None:
    """A material-OOD frame of chickpea before an ID frame of lentil of the same run: the run's ID group comes first, as in the selector."""
    rows = _summary_rows(("M OOD", "chickpea", "m", "fno"), ("U ID", "lentil", "u", "uno"), ("M ID", "lentil", "m", "fno"))
    rows.loc[rows["frame"] == "M OOD", "dataset_role"] = "material_ood"
    groups = analysis.evaluation.plots.transient._summary_series(rows)
    order = [(group["frame"].iloc[0], group["material_family"].iloc[0]) for group in groups.values()]
    assert order == [("M ID", "lentil"), ("M OOD", "chickpea"), ("U ID", "lentil")]


def test_a_single_runs_target_and_runtime_points_take_their_material_colours() -> None:
    """One run with lentil and chickpea cases: its target and runtime points take the two material colours, not the pooled role colour."""
    semantics = analysis.presentation.visual_semantics
    transient = analysis.evaluation.plots.transient
    lentil, chickpea = (to_hex(semantics.material_color(material)) for material in ("lentil", "chickpea"))
    records = tuple(sequence_record("autonomous_full", "full", case_id=material, material_family=material) for material in ("lentil", "chickpea"))
    for record in records:
        record.target.update(reference_time_to_target=2.0)
    target = transient.plot_target_time(records, styles=_STYLES).axes[0]
    accuracy = _summary_rows(("M", "lentil", "m", "fno"), ("M", "chickpea", "m", "fno"), normalized_drying_group_macro_rmse=0.1)
    accuracy["case_id"] = ["c0", "c1"]
    runtime = transient.plot_accuracy_runtime_comparison(
        accuracy, {"M": _paired_timing_report((0.3, 0.5))}, styles=_table_styles(accuracy), measure="model"
    ).axes[0]
    targets = {collection.get_label(): to_hex(cast("ColorType", collection.get_facecolor()[0])) for collection in target.collections}
    assert targets == {"Lentil, ID": lentil, "Chickpea, ID": chickpea}
    legend = runtime.get_legend()
    assert legend is not None
    handles = zip(legend.get_texts(), legend.legend_handles, strict=True)
    entries = {text.get_text(): to_hex(cast("Any", handle).get_color()) for text, handle in handles}
    assert entries == {"Chickpea, ID": chickpea, "Lentil, ID": lentil}
    assert {to_hex(cast("ColorType", collection.get_facecolor()[0])) for collection in runtime.collections} == {lentil, chickpea}


def test_snapshot_headers_count_each_dataset_also_at_zero_and_the_titles_count_nothing() -> None:
    """At 1 h one dataset holds a case and the other none: each map and parity header ends with its own n(t), 0 included, never a sum."""
    transient = analysis.evaluation.plots.transient
    series = {"Lentil, ID": (sequence_record("autonomous_full", "full", case_id="a"),), "Chickpea, ID": ()}
    maps = transient.plot_aggregate_error_maps(series, state_fields=("T",), physical_time=1.0, statistic="mean")
    parity = transient.plot_predicted_vs_reference_channels(series, state_fields=("T",), styles=_STYLES, physical_time=1.0)
    try:
        for figure, title in ((maps, maps.get_suptitle()), (parity, _figure_title(parity))):
            assert [axis.get_title() for axis in figure.axes[:2]] == ["Lentil, ID · n(t) = 1", "Chickpea, ID · n(t) = 0"]
            assert not re.search(r"n\(t\)|\bcases?\b", title), title
    finally:
        plt.close("all")


def test_a_compared_column_without_a_case_still_puts_its_run_on_a_line_of_its_own() -> None:
    """At 1 h run B holds no case: its map and parity headers still break after the run, from the context label, and keep n(t) = 0."""
    transient = analysis.evaluation.plots.transient
    series = {"A · Lentil, ID": (sequence_record("autonomous_full", "full", case_id="a"),), "B · Lentil, ID": ()}
    contexts = dict.fromkeys(series, "Lentil, ID")
    maps = transient.plot_aggregate_error_maps(series, state_fields=("T",), physical_time=1.0, statistic="mean", context_labels=contexts)
    parity = transient.plot_predicted_vs_reference_channels(series, state_fields=("T",), styles=_STYLES, physical_time=1.0, context_labels=contexts)
    for figure in (maps, parity):
        assert [axis.get_title() for axis in figure.axes[:2]] == ["A\nLentil, ID · n(t) = 1", "B\nLentil, ID · n(t) = 0"]


def test_a_compared_column_whose_cases_are_all_censored_puts_its_run_on_a_line_of_its_own_also_without_context_labels() -> None:
    """At τ = 0.50 run B's one case is censored: without context labels its headers still break after the run, from its records."""
    transient = analysis.evaluation.plots.transient
    series = {
        "A · Lentil, ID": (_sloped(_timed_record("a", 3), 1.0),),
        "B · Lentil, ID": (_sloped(_timed_record("b", 3), 1.0, reached=False),),
    }
    maps = transient.plot_aggregate_error_maps(series, state_fields=("T",), phase=0.5, statistic="mean")
    parity = transient.plot_predicted_vs_reference_channels(series, state_fields=("T",), styles=_STYLES, phase=0.5)
    for figure in (maps, parity):
        assert [axis.get_title() for axis in figure.axes[:2]] == ["A\nLentil, ID · n(τ) = 1", "B\nLentil, ID · n(τ) = 0"]


def test_snapshots_at_a_phase_that_no_context_reached_still_draw_every_context_at_zero_with_its_phase_note() -> None:
    """Every case is censored: the maps and the parity at τ = 0.50 still draw each context, headed n(τ) = 0, and keep the phase note."""
    transient = analysis.evaluation.plots.transient
    series = {
        "Lentil, ID": (_sloped(_timed_record("a", 3), 1.0, reached=False),),
        "Chickpea, ID": (_sloped(_timed_record("b", 3), 1.0, material="chickpea", reached=False),),
    }
    maps = transient.plot_aggregate_error_maps(series, state_fields=("T",), phase=0.5, statistic="mean")
    parity = transient.plot_predicted_vs_reference_channels(series, state_fields=("T",), styles=_STYLES, phase=0.5)
    try:
        for figure in (maps, parity):
            assert [axis.get_title() for axis in figure.axes[:2]] == ["Lentil, ID · n(τ) = 0", "Chickpea, ID · n(τ) = 0"]
            texts = [text.get_text() for text in figure.findobj(Text)]
            assert texts.count("No reached case to pool") == len(series)
            assert sum("1 censored not pooled" in text for text in texts) == 1  # the one phase note, a line per context
    finally:
        plt.close("all")


def test_each_dataset_keeps_one_error_scale_and_one_parity_range_that_its_compared_runs_share() -> None:
    """Runs A and B on lentil ID, A on chickpea OOD with larger errors: the ID columns share a scale and range the OOD one never sets."""
    transient = analysis.evaluation.plots.transient
    channel = analysis.evaluation.transient_artifact.STATE_ORDER.index("T")
    records = {
        "A · Lentil, ID": sequence_record("autonomous_full", "full", case_id="l", material_family="lentil"),
        "B · Lentil, ID": sequence_record("autonomous_full", "full", case_id="l", material_family="lentil"),
        "A · Chickpea, OOD": sequence_record("autonomous_full", "full", case_id="c", material_family="chickpea", dataset_role="material_ood"),
    }
    records["B · Lentil, ID"].identity["checkpoint_identity"] = {"best_checkpoint_sha256": "e" * 64}
    for record, reach in zip(records.values(), (1.0, 2.0, 10.0), strict=True):  # errors ramp from 0 to the reach over the map at 1 h
        record.predicted_states = record.reference_states.copy()
        shape = record.predicted_states[1, channel].shape
        record.predicted_states[1, channel] += np.linspace(0.0, reach, math.prod(shape)).reshape(shape)
    series = {label: (record,) for label, record in records.items()}
    maps = transient.plot_aggregate_error_maps(series, state_fields=("T",), physical_time=1.0, statistic="mean")
    styles = _comparison_of(*records.values())
    parity = transient.plot_predicted_vs_reference_channels(series, state_fields=("T",), styles=styles, physical_time=1.0)
    try:
        bands = [tuple(np.asarray(cast("Any", axis.collections[0].norm).boundaries).tolist()) for axis in maps.axes[:3]]
        assert bands[0] == bands[1] != bands[2]
        ranges = [axis.get_xlim() for axis in parity.axes[:3]]
        assert ranges[0] == ranges[1] != ranges[2]
    finally:
        plt.close("all")


def test_the_last_window_counts_hours_before_each_end_as_soon_as_two_case_ends_differ_at_all() -> None:
    """Last stored times 1e-7 h apart already differ, compared exactly as in the Dataset EDA; equal ends keep the clock."""
    labels = {}
    for shift in (0.0, 1e-7):
        records = (_timed_record("a", 60), _timed_record("b", 60))
        records[1].physical_times[-1] += shift
        records[1].target["reference_final_time"] += shift
        figure = analysis.evaluation.plots.transient.plot_state_trajectory_summary({"ID": records}, state_fields=("T",), styles=_STYLES)
        labels[shift] = {axis.get_xlabel() for axis in figure.axes}
        plt.close(figure)
    assert "Time before end [h]" not in labels[0.0]
    assert "Time before end [h]" in labels[1e-7]


def test_the_10th_to_90th_percentiles_are_whiskers_at_times_of_their_own_in_no_line_style_keyed_once_per_kind() -> None:
    """
    With one series or two, every band is vertical whiskers, never a line in a role's style, and no two bands share a time.

    A whisker spans the 10th to 90th percentile of the case means (cases a and b predict 2 K and 4 K above their reference); one
    key per kind names them with a vertical stroke, not a line.
    """
    transient = analysis.evaluation.plots.transient
    one = {"ID": (_timed_record("a", 20), _timed_record("b", 20))}
    one["ID"][1].predicted_states[1:] = 4.0
    for series in (one, {**one, "OOD": (_timed_record("c", 20), _timed_record("d", 20))}):
        figure = transient.plot_state_trajectory_summary(series, state_fields=("T",), styles=_STYLES)
        axis = figure.axes[0]
        assert all(str(line.get_label()) and not str(line.get_label()).startswith("_") for line in axis.get_lines())  # keyed curves only
        whiskers = [collection for collection in axis.collections if isinstance(collection, LineCollection)]
        assert len(whiskers) == 2 * len(series)  # each series' reference and prediction
        segments = [collection.get_segments() for collection in whiskers]
        assert all(start[0] == end[0] for band in segments for start, end in band)  # vertical
        times = [float(start[0]) for band in segments for start, _end in band]
        assert len(times) == len(set(times))  # staggered: no two bands share a time
        keys = {text.get_text(): handle for legend in figure.legends for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)}
        assert [cast("Any", keys[kind]).get_marker() for kind in ("Reference 10th-90th", "Prediction 10th-90th")] == ["|", "|"]
    shown = transient._display_state_values("T", np.asarray([2.0, 4.0]), error=False)  # a's and b's prediction
    prediction = segments[1]  # the ID prediction, after the ID reference
    bounds = np.asarray([(start[1], end[1]) for start, end in prediction], dtype=np.float64)
    np.testing.assert_allclose(bounds, np.tile(np.quantile(shown, (0.1, 0.9)), (len(prediction), 1)))
    plt.close("all")


def test_the_whiskers_of_two_bands_never_share_a_time_where_one_band_lacks_a_finite_bound() -> None:
    """
    Case d lacks the first post-origin time, 2 h, where OOD's band then has one case and no finite bound, while ID's has one.

    The bands take their slots on the panel's common time grid, never on their own finite times, so OOD's whiskers still stand
    at times of their own.
    """
    series = {"ID": (_timed_record("a", 6), _timed_record("b", 6)), "OOD": (_timed_record("c", 6), _timed_record("d", 5))}
    for records in series.values():
        records[1].predicted_states[1:] = 4.0  # a band of non-zero width
    series["OOD"][1].physical_times = np.asarray([0.0, 4.0, 6.0, 8.0, 10.0])
    figure = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        series, scaling_states=dict.fromkeys(series, scaling_state()), state_fields=("T",), styles=_STYLES
    )
    bands = [collection.get_segments() for collection in figure.axes[0].collections if isinstance(collection, LineCollection)]
    times = [float(start[0]) for band in bands for start, _end in band]
    assert len(bands) == 2
    assert len(times) == len(set(times))  # no two whiskers at one time


def test_a_band_without_a_finite_bound_at_its_slot_takes_a_free_time_nearby_and_one_finite_at_one_time_keeps_it() -> None:
    """
    Of two bands on six common times, the first takes the slots 0, 2 and 4, the second 1, 3 and 5.

    The first has no finite bound at 2, so it takes 3, the nearest time no whisker holds, since the second has none there either;
    the second keeps 1 and 5. A band finite at one time only keeps that whisker; a band holding it there and at another time
    gives it up, so no two whiskers share a time.
    """
    transient = analysis.evaluation.plots.transient

    def band(size: int, finite: Sequence[int]) -> Any:
        bounds = np.full(size, np.nan)
        bounds[list(finite)] = 1.0
        return transient._ExactTimeSummary(
            np.arange(size, dtype=np.float64), bounds, bounds, bounds + 1.0, np.full(size, 2), np.zeros(size, dtype=np.int64)
        )

    assert [place.tolist() for place in transient._whisker_positions([band(6, (0, 1, 3, 4, 5)), band(6, (1, 5))])] == [[0, 3, 4], [1, 5]]
    assert [place.tolist() for place in transient._whisker_positions([band(4, range(4)), band(4, (2,))])] == [[0], [2]]


def test_a_zero_width_whisker_is_a_short_stroke_across_its_curve_so_the_key_names_a_drawn_mark() -> None:
    """Cases a and b agree, so their 10th and 90th percentiles coincide: each whisker shows as a short vertical stroke at that bound."""
    figure = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": (_timed_record("a", 6), _timed_record("b", 6))}, scaling_states={"ID": scaling_state()}, state_fields=("T",), styles=_STYLES
    )
    axis = figure.axes[0]
    (band,) = [collection for collection in axis.collections if isinstance(collection, LineCollection)]
    (strokes,) = [collection for collection in axis.collections if isinstance(collection, PathCollection)]
    bounds = [start.tolist() for start, end in band.get_segments() if np.array_equal(start, end)]
    assert bounds
    assert np.asarray(strokes.get_offsets()).tolist() == bounds  # every zero-width whisker, at its common bound
    vertices = np.asarray(strokes.get_paths()[0].vertices)
    assert np.ptp(vertices[:, 0]) == 0 < np.ptp(vertices[:, 1])  # vertical, like the key
    curve = next(line for line in axis.get_lines() if str(line.get_label()).endswith("material-case RMSE"))
    assert math.sqrt(strokes.get_sizes()[0]) > curve.get_linewidth()  # taller than its curve is wide, so it shows across it
    assert to_hex(cast("ColorType", strokes.get_edgecolor()[0])) == to_hex(cast("ColorType", band.get_color()[0]))
    assert [text.get_text() for legend in figure.legends for text in legend.get_texts()].count("Per-case RMSE 10th-90th") == 1


def test_the_rollout_error_note_names_its_stored_state_channel_scope() -> None:
    """The saved-scale nRMSE exists for the stored states only, so the note says the derived grain moisture has none."""
    figure = analysis.evaluation.plots.transient.plot_error_over_physical_time(
        {"ID": (_timed_record("a", 20),)}, scaling_states={"ID": scaling_state()}, state_fields=("T", "w_gr"), styles=_STYLES
    )
    assert "saved-scale nRMSE of the stored states only: the derived grain moisture has none" in _figure_note(figure)
    plt.close(figure)


def test_full_horizon_points_are_named_and_a_horizon_without_a_case_is_named_by_its_tick_alone() -> None:
    """In the elapsed-time panel the full-horizon point carries one keyed ring; a horizon no series has a case at says so under its tick, no x."""
    summary = _with_provenance(
        pd.DataFrame.from_records(
            {
                "frame": "A",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": horizon,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": value,
                "elapsed_physical_time_median": hours,
                "contributing_record_count": count,
            }
            for horizon, value, hours, count in ((1, 0.1, 1.0, 1), (2, math.nan, math.nan, 0), ("full", 0.3, 3.0, 1))
        )
    )
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    horizon_axis, elapsed = figure.axes[:2]
    assert not elapsed.texts  # a keyed ring instead of a text per point
    (ring,) = (collection for collection in elapsed.collections if collection.get_label() == analysis.evaluation.plots.transient._FULL_HORIZON)
    assert np.asarray(ring.get_offsets()).tolist() == [[3.0, 0.3]]
    assert [label.get_text() for label in horizon_axis.get_xticklabels()] == ["1", "2\nno case", "full"]  # horizon 2 keeps its place
    assert not _crosses(horizon_axis)  # the tick says it; an x would stand at an invented height
    plt.close(figure)


def test_a_horizon_some_series_lack_gets_one_neutral_x_on_the_axis_line_whose_key_matches_it() -> None:
    """B has no case at horizon 2, no series at 4: one grey x on the axis line at 2, never at a value, keyed alike; 4 says so in its tick."""
    rows = (("A", 1, 0.1, 1), ("A", 2, 0.2, 1), ("A", 4, math.nan, 0), ("B", 1, 0.15, 1), ("B", 2, math.nan, 0), ("B", 4, math.nan, 0))
    summary = _with_provenance(
        pd.DataFrame.from_records(
            {
                "frame": frame,
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": horizon,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": value,
                "elapsed_physical_time_median": float(horizon) if count else math.nan,
                "contributing_record_count": count,
            }
            for frame, horizon, value, count in rows
        )
    )
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    horizon_axis = figure.axes[0]
    assert [label.get_text() for label in horizon_axis.get_xticklabels()] == ["1", "2", "4\nno case"]
    (cross,) = _crosses(horizon_axis)
    figure.canvas.draw()
    ((x, y),) = cross.get_transform().transform(np.column_stack([cross.get_xdata(), cross.get_ydata()]))
    assert x == pytest.approx(horizon_axis.transData.transform((1.0, 0.0))[0])  # horizon 2
    assert y == pytest.approx(horizon_axis.get_window_extent().y0)  # on the axis line: no error value
    keys = {text.get_text(): handle for legend in figure.legends for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)}
    assert to_hex(cast("Any", keys[str(cross.get_label())]).get_color()) == to_hex(cross.get_color()) == to_hex("0.55")
    assert "no case at horizon 2: B · Lentil, ID" in _figure_note(figure)
    plt.close(figure)


def test_rolling_origin_rows_without_a_case_at_any_horizon_keep_every_requested_horizon_tick_on_empty_panels() -> None:
    """Every rolling-origin row counts zero cases: each requested horizon's tick says «no case», both panels stay empty, the note names the series."""
    summary = _with_provenance(
        pd.DataFrame.from_records(
            {
                "frame": "A",
                "material_family": "lentil",
                "mode": "rolling_origin",
                "requested_horizon": horizon,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": math.nan,
                "elapsed_physical_time_median": math.nan,
                "contributing_record_count": 0,
            }
            for horizon in (1, 2, "full")
        )
    )
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    horizon_axis, elapsed = figure.axes[:2]
    assert [label.get_text() for label in horizon_axis.get_xticklabels()] == ["1\nno case", "2\nno case", "full\nno case"]
    assert not [*horizon_axis.lines, *horizon_axis.collections, *elapsed.lines, *elapsed.collections]
    assert not [*elapsed.get_xticks(), *elapsed.get_yticks(), *horizon_axis.get_yticks()]  # no time span or error value claimed
    assert "Lentil, ID: no case at any horizon" in _figure_note(figure)


def test_a_material_without_a_case_at_any_horizon_is_named_and_a_drawn_one_lists_its_horizons_without_a_case() -> None:
    """Chickpea of the run has no case at either horizon beside lentil: the note names it instead of dropping it, and lentil's gap at 2."""
    summary = _with_provenance(
        pd.DataFrame.from_records(
            {
                "frame": "A",
                "material_family": material,
                "mode": "rolling_origin",
                "requested_horizon": horizon,
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": value,
                "elapsed_physical_time_median": hours,
                "contributing_record_count": count,
            }
            for material, horizon, value, hours, count in (
                ("lentil", 1, 0.1, 1.0, 1),
                ("lentil", 2, math.nan, math.nan, 0),
                ("chickpea", 1, math.nan, math.nan, 0),
                ("chickpea", 2, math.nan, math.nan, 0),
            )
        )
    )
    figure = analysis.evaluation.plots.transient.plot_horizon_error(summary, styles=_table_styles(summary))
    note = _figure_note(figure)
    plt.close(figure)
    assert "Chickpea, ID: no case at any horizon" in note
    assert "no case at horizon 2: Lentil, ID" in note


def test_the_compute_route_of_the_aggregate_maps_goes_to_the_log_never_onto_the_figure(caplog: pytest.LogCaptureFixture) -> None:
    """The device that reduced the snapshot is kernel-log evidence; the figure keeps only the scientific support."""
    transient = analysis.evaluation.plots.transient
    with caplog.at_level(logging.INFO, logger=transient.__name__):
        figure = transient.plot_aggregate_error_maps(
            {"ID": (sequence_record("autonomous_full", "full", case_id="a"),)}, state_fields=("T",), physical_time=1.0, statistic="mean"
        )
    (logged,) = [record for record in caplog.records if record.getMessage().startswith("Aggregate error maps")]
    route = str(cast("tuple[Any, ...]", logged.args)[1])
    assert route
    assert route != "no route"
    assert not [text.get_text() for text in figure.findobj(Text) if route in text.get_text()]
    plt.close(figure)


def test_boxes_of_a_run_over_several_materials_mark_each_median_with_its_material_marker_in_the_panel_and_the_key() -> None:
    """Run F spans lentil and chickpea, so its boxes share colour and hatch: each median carries its material marker, and so does its key."""
    semantics = analysis.presentation.visual_semantics
    rows = _summary_rows(
        *(("F ID", material, "f", "fno") for material in ("lentil", "lentil", "chickpea", "chickpea")),
        ("U ID", "lentil", "u", "uno"),
        ("U ID", "lentil", "u", "uno"),
        normalized_rmse_T=0.2,
        normalized_drying_group_macro_rmse=0.1,
    )
    rows.loc[[1, 3, 5], "normalized_rmse_T"] = 0.4
    axis = analysis.evaluation.plots.transient.plot_error_distributions(rows, state_fields=("T",), styles=_table_styles(rows)).axes[0]
    marks = [collection for collection in axis.collections if isinstance(collection, PathCollection)]
    assert len(marks) == 2  # F's two medians; U's single material needs no marker
    for collection, material in zip(marks, ("chickpea", "lentil"), strict=True):  # the selector's material order
        shape = MarkerStyle(semantics.MATERIAL_MARKERS[material])
        np.testing.assert_allclose(np.asarray(collection.get_offsets())[:, 1], [0.3])
        drawn, expected = collection.get_paths()[0].vertices, shape.get_path().transformed(shape.get_transform()).vertices
        np.testing.assert_allclose(np.asarray(drawn, dtype=np.float64), np.asarray(expected, dtype=np.float64))
    legend = axis.get_legend()
    assert legend is not None
    assert len(legend.findobj(PathCollection)) == 2  # the keys of F's two series carry the marker over the box swatch
    plt.close("all")


def test_matched_budget_panels_follow_mode_and_horizon_and_the_key_names_each_run_once_beside_the_material_markers() -> None:
    """Two protocols give two titled panels joined only from A0; three runs over two materials key each run once, in its colour."""
    comparison = analysis.evaluation.transient_comparison
    semantics = analysis.presentation.visual_semantics
    spent = {"A0": 0.0, "A1": 8.0, "B": 8.0}
    arms = {name: comparison.TrainingArmEvidence(name, "p", "a", "d", "s", "i", "c", "b", "rollout", spent[name], spent[name], 0.0) for name in spent}
    lineage = comparison.LineageComparison(a0=arms["A0"], a1=arms["A1"], b=arms["B"], primary_comparison="B_vs_A1", separate_comparison="B_vs_A0")
    colours = {"A0 · ID": "#000000", "A1 · ID": "#0072b2", "B · ID": "#d55e00"}
    rows = pd.DataFrame.from_records(
        {
            "frame": frame,
            "comparison_arm": frame.split(" ")[0],
            "training_compute": spent[frame.split(" ")[0]],
            "optimizer_device_compute": spent[frame.split(" ")[0]],
            "normalized_drying_group_macro_rmse": 0.1,
            "material_family": material,
            "compute_clock": "supervised_transitions",
            "checkpoint_role": "best",
            "mode": mode,
            "requested_horizon": horizon,
            "scope": "cumulative",
        }
        for frame in colours
        for material in ("lentil", "chickpea")
        for mode, horizon in (("autonomous_full", "full"), ("rolling_origin", 2))
    )
    styles = analysis.evaluation.plots.series.SeriesStyles({frame: semantics.SeriesStyle(colour, "-", None) for frame, colour in colours.items()})
    transient = analysis.evaluation.plots.transient
    figure = transient.plot_training_performance_vs_compute(rows, lineage=lineage, styles=styles, run_labels={"A0 · ID": "A0"})
    try:
        panels = [axis for axis in figure.axes if axis.get_visible()]
        mode = analysis.presentation.display_labels.evaluation_mode_display_label
        assert [axis.get_title() for axis in panels] == [mode("autonomous_full"), f"{mode('rolling_origin')}, horizon 2"]
        for axis in panels:  # A0 to A1 and to B for each material, never A1 to B at the same budget
            assert [np.asarray(line.get_xdata(), dtype=np.float64).tolist() for line in axis.get_lines()] == [[0.0, 8.0]] * 4
        assert _figure_title(figure) == "Shared-A0 performance vs committed budget"
        legends = {legend.get_title().get_text(): legend for legend in figure.legends}
        assert [text.get_text() for text in legends["Run"].get_texts()] == ["A0", "A1 · ID", "B · ID"]
        run_keys = [(isinstance(handle, Patch), to_hex(cast("Any", handle).get_facecolor())) for handle in legends["Run"].legend_handles]
        assert run_keys == [(True, colour) for colour in colours.values()]  # a swatch, no marker: the run's points carry two shapes
        assert [text.get_text() for text in legends["Material"].get_texts()] == ["Chickpea", "Lentil"]
        shapes = [cast("Any", handle).get_marker() for handle in legends["Material"].legend_handles]
        assert shapes == [semantics.MATERIAL_MARKERS[material] for material in ("chickpea", "lentil")]
    finally:
        plt.close(figure)


def test_matched_budget_panels_order_the_horizons_ascending_with_full_last() -> None:
    """Conditions first seen at horizon 16, full and 8: the panels read 8, 16 and full, as every view orders horizons."""
    comparison = analysis.evaluation.transient_comparison
    spent = {"A0": 0.0, "A1": 8.0, "B": 8.0}
    arms = {name: comparison.TrainingArmEvidence(name, "p", "a", "d", "s", "i", "c", "b", "rollout", spent[name], spent[name], 0.0) for name in spent}
    lineage = comparison.LineageComparison(a0=arms["A0"], a1=arms["A1"], b=arms["B"], primary_comparison="B_vs_A1", separate_comparison="B_vs_A0")
    rows = pd.DataFrame.from_records(
        {
            "comparison_arm": arm,
            "training_compute": spent[arm],
            "optimizer_device_compute": spent[arm],
            "normalized_drying_group_macro_rmse": 0.1,
            "material_family": "lentil",
            "compute_clock": "supervised_transitions",
            "checkpoint_role": "best",
            "mode": "rolling_origin",
            "requested_horizon": horizon,
            "scope": "cumulative",
        }
        for horizon in (16, "full", 8)
        for arm in spent
    )
    figure = analysis.evaluation.plots.transient.plot_training_performance_vs_compute(rows, lineage=lineage)
    mode = analysis.presentation.display_labels.evaluation_mode_display_label("rolling_origin")
    assert [axis.get_title() for axis in figure.axes if axis.get_visible()] == [f"{mode}, horizon {horizon}" for horizon in (8, 16, "full")]
    plt.close(figure)
