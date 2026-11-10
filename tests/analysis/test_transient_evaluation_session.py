"""Protect transient Evaluation aggregation, presentation, and local reports.

Responsibilities:
  - Check pooled statistics, target records, lazy case access, panels, and local reports
  - Check that a session admits only frames recording one valid analysis producer

Design principles:
  - Unequal support, unavailable metrics, and material summaries retain distinct meanings
  - The producer check reads recorded provenance only, never the running analysis code

This module does NOT:
  - Generate model predictions or refit saved scales
"""

from __future__ import annotations

import copy
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
from support.transient_evaluation import production_frame, scaling_state, sequence_frame, sequence_record

from src import analysis, experiments
from src.analysis.evaluation import evaluation_transient_session as session
from src.analysis.evaluation import evaluation_transient_validity as transient_validity
from src.analysis.evaluation import evaluation_workspace as workspace
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_TRANSIENT_TRACKING_SUMMARY_LIMIT = 470
_TRANSIENT_PUBLICATION_SUMMARY_LIMIT = 512


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
    scaling = TransientScalingArtifact.from_state_dict(scaling_state())

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
            state = accumulator.state_dict()
            counts = cast("list[int]", state["counts"])
            value_count = int(sum(counts))
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
    expected_normalized = np.sqrt((0.5**2 + 2 * 1.5**2) / 3)
    expected_physical = np.sqrt((1.0**2 + 2 * 3.0**2) / 3)

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


def test_session_uses_saved_scale_excludes_origin_and_retains_default_modes() -> None:
    """Use persisted scaling, omit origins, and retain all three required modes."""
    values = (sequence_record("teacher_forced_one_step", 1), sequence_record("autonomous_full", "full"), sequence_record("rolling_origin", 2))
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(values)})
    summaries = evaluation.summaries()
    teacher_cumulative = next(item for item in summaries if item.mode == "teacher_forced_one_step" and item.scope == "cumulative")
    teacher_metrics = teacher_cumulative.metrics
    assert teacher_metrics is not None
    assert teacher_metrics.normalized_drying_group_macro_rmse == pytest.approx(1.0)
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
    assert cumulative.metrics is not None
    assert endpoint.metrics is not None
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


def test_session_reports_support_counts_instead_of_metrics_over_a_nonfinite_prediction() -> None:
    """A non-finite case makes the group's error metrics unavailable in rows and W&B; its finite peer is not reduced alone."""
    records = (sequence_record("rolling_origin", 2, case_id="a", offset=np.nan), sequence_record("rolling_origin", 2, case_id="b"))
    evaluation = session.TransientEvaluationSession({"id": sequence_frame(records)})
    cumulative = next(item for item in evaluation.summaries(modes=("rolling_origin",)) if item.scope == "cumulative")
    assert cumulative.metrics is None
    assert (cumulative.metric_support["finite_value_count"], cumulative.metric_support["required_value_count"]) == (8, 16)

    row = evaluation.dataset_dataframe(modes=("rolling_origin",)).query("scope == 'cumulative'").iloc[0]
    assert (row["nonfinite_value_count"], pd.isna(row["normalized_drying_group_macro_rmse"])) == (8, True)
    values = evaluation.wandb_summary(modes=("rolling_origin",))
    support = {key: value for key, value in values.items() if key.endswith("/rolling_origin/2/cumulative/nonfinite_value_count")}
    assert list(support.values()) == [8]
    assert not any("rmse" in key for key in values)


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


def test_tracking_summary_stays_bounded_for_complete_two_role_horizon_inventory(tmp_path: Path) -> None:
    """Keep the complete two-role aggregate inventory within the W&B scalar cap."""
    records = (
        sequence_record("teacher_forced_one_step", 1),
        sequence_record("autonomous_full", "full"),
        *(sequence_record("rolling_origin", horizon) for horizon in analysis.evaluation.transient_artifact.FIXED_HORIZONS),
        sequence_record("rolling_origin", "full"),
    )
    frames = {
        role: production_frame(
            records,
            artifact_root=str(tmp_path / role),
            dataset_name=f"{role}_dataset",
            dataset_role=role,
        )
        for role in ("id", "material_ood")
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
    validator = experiments.tracking._validate_evaluation_summary_entry
    assert all(validator(key, value) == key for key, value in publication.items())
    assert publication["evaluation/identity/checkpoint_sha256"] == "d" * 64
    assert publication["evaluation/identity/input_profile"] == "canonical_physics_complete"
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
    assert isinstance(sample, widgets.VBox)
    assert isinstance(error_time, widgets.VBox)
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(sample))
    assert not any(isinstance(widget, widgets.Play) for widget in _widget_tree(error_time))
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
    """Return the displayed tab titles and each tab's dropdown labels of one lazy panel."""
    tabs = cast("Any", panel).tabs
    return tuple(tabs.titles), [[label for label, _index in section.children[0].options] for section in tabs.children]


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
    monkeypatch.setattr(analysis.ui.viewers, "render_figure", lambda **kwargs: rendered.append(kwargs))
    built: list[widgets.Widget] = []
    original = analysis.evaluation.panel._make_transient_aggregate_viewer

    def capture(**kwargs: Any) -> widgets.Widget:
        built.append(original(**kwargs))
        return built[-1]

    monkeypatch.setattr(analysis.evaluation.panel, "_make_transient_aggregate_viewer", capture)
    preference = compute.AnalysisComputePreference("cpu")
    panel = analysis.evaluation.panel.build_transient_panel(session=evaluation, evaluation_compute_device=preference)
    tabs = cast("Any", panel).tabs
    dropdown = tabs.children[1].children[0]
    dropdown.index = next(index for index, (label, _value) in enumerate(dropdown.options) if "Mean error maps" in label)
    assert len(built) == 1
    assert built[0].comm is not None
    before = len(rendered)
    preference.set_requested("auto")
    assert len(rendered) > before

    analysis.evaluation.panel.close_panel(panel)
    after = len(rendered)
    preference.set_requested("cpu")
    assert len(rendered) == after
    assert built[0].comm is None
    assert tabs.comm is None
    assert dropdown.comm is None
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
    controller = SimpleNamespace(analysis=widgets.VBox(), _workflow=previous)
    cast("Any", workspace._EvaluationWorkspaceController)._replace_workflow(controller, replacement)

    assert events == ["old: session closed, panel closed True, child closed True"]
    assert controller.analysis.children == (replacement.panel,)
    assert controller._workflow is replacement
    assert replacement.panel.comm is not None


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
        sequence_record("autonomous_full", "full", case_id="high", offset=-3.0),
    )
    evaluation = session.TransientEvaluationSession(
        {
            "id": production_frame(
                records,
                artifact_root=str(tmp_path / "artifact"),
            )
        }
    )
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
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    viewer = analysis.evaluation.panel._make_transient_aggregate_viewer(
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


def _indexed_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loads: list[str]) -> session.TransientEvaluationSession:
    """Build a session whose index counts record loads instead of opening bundles."""
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
        summaries=cast("Any", records),
        unavailable_horizons=(),
        frame=frame,
        provenance=frame.attrs["artifact_provenance"],
        identity_sha256="f" * 64,
        _rows={record.record_id: {} for record in records},
    )
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
    monkeypatch.setattr(
        analysis.ui.viewers,
        "render_figure",
        lambda **_kwargs: None,
    )
    fields = analysis.evaluation.panel._transient_channel_fields(evaluation)
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(
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
    assert not any(isinstance(widget, widgets.Dropdown) and widget.description == "Material:" for widget in _widget_tree(aggregate_view))
    support_status = next(widget for widget in _widget_tree(aggregate_view) if isinstance(widget, widgets.HTML) and "n(t)=" in widget.value)
    assert all(
        f"{label}: n(t)=1" in support_status.value
        for label in (
            "Chickpea · ID",
            "Lentil · ID",
            "Kidney bean · Material OOD",
        )
    )
    evaluation.close()


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
        isinstance(widget, widgets.Dropdown) and widget.description in {"Protocol:", "Origin:", "Horizon:"} for widget in _widget_tree(viewer)
    )
    assert any(isinstance(widget, widgets.FloatText) and widget.description == "Time [h]:" for widget in _widget_tree(viewer))
    evaluation.close()


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
    case_row = viewer.children[0]
    status = viewer.children[4]
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


def test_transient_panel_plots_pipeline_and_local_report_remain_sequence_aware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep exact-time controls, all states, unavailable C, and local report outputs."""
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
        state_order=analysis.evaluation.transient_artifact.STATE_ORDER,
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
    assert isinstance(viewer, widgets.VBox)
    case_row = viewer.children[0]
    assert isinstance(case_row, widgets.HBox)
    assert tuple(control.description for control in case_row.children) == (
        "Case:",
        "←",
        "→",
    )
    channel_row = viewer.children[1].children[1]
    assert hasattr(channel_row, "boxes")
    assert all(isinstance(checkbox, widgets.Checkbox) and checkbox.value is True for checkbox in channel_row.boxes.values())
    time_row = viewer.children[3].children[0]
    assert isinstance(time_row.children[1], widgets.Button)
    assert time_row.children[1].description == "Export GIF"
    navigator_controls = time_row.children[0]
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

    trajectory_viewer = analysis.evaluation.panel._make_transient_scope_viewer(
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
    figure = analysis.evaluation.plots.transient.plot_state_maps(records[0], error_mode="Signed error")
    figure.canvas.draw()
    expected_map_count = len(analysis.evaluation.transient_artifact.STATE_ORDER) * 3
    map_axes = tuple(figure.axes[:expected_map_count])
    colorbar_axes = tuple(figure.axes[expected_map_count:])
    assert tuple(axis.get_title() for axis in map_axes[:3]) == (
        "Reference",
        "Prediction",
        "Signed error · mean |e|/s n/a",
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
    figure.canvas.draw()
    assert all(
        map_axis.get_position().y0 - 1e-6 <= colorbar_axis.get_position().y0 and colorbar_axis.get_position().y1 <= map_axis.get_position().y1 + 1e-6
        for map_axis, colorbar_axis in zip(map_axes, colorbar_axes, strict=True)
    )
    plt.close(figure)

    summary_frame = evaluation.dataset_dataframe()
    channel_figure = analysis.evaluation.plots.transient.plot_channel_error(summary_frame)
    assert channel_figure.axes[0].get_ylabel() == "Normalized RMSE [1]"
    assert len(channel_figure.axes[0].patches) == len(analysis.evaluation.transient_artifact.STATE_ORDER)
    plt.close(channel_figure)

    degradation = evaluation.pipeline_degradation()
    assert len(degradation) == 1
    assert degradation[0].c_available is False
    assert degradation[0].complete_pipeline_error is None
    runtime_figure = analysis.evaluation.plots.transient.plot_timing_distributions(evaluation.timing_report("id"))
    assert all(axis.get_ylabel() == "Recorded duration [s]" for axis in runtime_figure.axes)
    plt.close(runtime_figure)
    state_maps = analysis.evaluation.plots.transient.plot_state_maps
    curated_map_checks: list[bool] = []

    def check_curated_normalized_maps(record: Any, **kwargs: Any) -> Figure:
        figure = state_maps(record, **kwargs)
        for channel in range(len(analysis.evaluation.transient_artifact.STATE_ORDER)):
            error_axis = figure.axes[channel * 3 + 2]
            assert error_axis.collections, "Curated normalized error requires the persisted train-only scale."
            np.testing.assert_allclose(
                np.asarray(error_axis.collections[0].get_array()),
                np.abs(record.predicted_states[-1, channel] - record.reference_states[-1, channel]) / 2.0,
            )
        curated_map_checks.append(True)
        return figure

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_state_maps", check_curated_normalized_maps)
    report = analysis.presentation.curated.render_curated_transient_analysis(
        session=evaluation,
        output_dir=tmp_path / "report",
    )
    assert set(report.media_files).union(report.tables) == analysis.presentation.curated.TRANSIENT_CURATED_ANALYSIS_KEYS
    assert all(report_path.is_file() for report_path in report.media_files.values())
    assert curated_map_checks == [True]
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
    record.spatial_mask = np.ones((6, 6), dtype=bool)
    record.static_conditioning = np.zeros((7, 6, 6), dtype=np.float32)
    record.static_conditioning[0], record.static_conditioning[1] = np.meshgrid(np.arange(6.0), np.arange(6.0))
    return record


def test_state_map_norms_cover_every_given_time_and_the_single_frame_default_uses_its_own_time() -> None:
    """The fixed scales span all given times; a plot without norms scales to its own time."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    common = {"state_fields": ("T", "w_surf"), "lock_scale": False, "error_mode": "Absolute error"}
    over_time = transient.state_map_norms(record, time_indices=(0, 2), **common)
    single = transient.state_map_norms(record, time_indices=(1,), **common)
    for field, channel in (("T", 0), ("w_surf", 2)):
        expected = transient._display_state_values(field, record.reference_states[[0, 2], channel], error=False)
        assert over_time[field].reference.vmin == pytest.approx(expected.min())
        assert over_time[field].reference.vmax == pytest.approx(expected.max())
        absolute = np.abs(record.predicted_states[[0, 2], channel] - record.reference_states[[0, 2], channel])
        shown = transient._display_state_values(field, absolute, error=True)
        assert over_time[field].error.boundaries[0] <= shown.min()
        assert over_time[field].error.boundaries[-1] >= shown.max()
        assert single[field].reference.vmin > over_time[field].reference.vmin
    figure = transient.plot_state_maps(record, physical_time=1.0, **common)
    assert figure.axes[0].collections[0].norm.vmin == pytest.approx(single["T"].reference.vmin)
    assert figure.axes[0].collections[0].norm.vmax == pytest.approx(single["T"].reference.vmax)
    plt.close(figure)


def test_negative_predicted_water_enters_every_grain_moisture_path_as_predicted() -> None:
    """Negative predicted water is shown and averaged over all cells like the reference; nothing is excluded."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[1, 2:4, 2:4, 2] = -0.1
    figure = transient.plot_state_maps(record, state_fields=("w_gr",), physical_time=1.0, error_mode="Absolute error")
    shown = figure.axes[1].collections[0].get_array()
    assert np.ma.count_masked(shown) == 0
    assert shown.min() < 0.0
    assert not any("Unavailable" in text.get_text() for text in figure.axes[2].texts)
    plt.close(figure)
    trajectory = transient.plot_state_trajectory_summary({"case": [record]}, state_fields=("w_gr",))
    (line,) = (line for line in trajectory.axes[0].get_lines() if line.get_label() == "case · Prediction")
    grain = transient._grain_moisture(record, np.asarray(record.predicted_states, dtype=np.float64))
    assert line.get_ydata()[1] == pytest.approx(transient._display_state_values("w_gr", grain, error=False)[1].mean())
    plt.close(trajectory)
    aggregate = transient.plot_aggregate_error_maps([record], state_fields=("w_gr",), physical_time=1.0, statistic="mean")
    plt.close(aggregate)
    assert transient.maximum_error_physical_time(record, state_fields=("w_gr",)) in {1.0, 2.0}


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
    text = transient._mean_normalized_error_text(record, "T", record.reference_states[1, 0], record.predicted_states[1, 0], None)
    assert text == "mean |e|/s non-finite"
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
    assert len(line.get_ydata()) == 0
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


def test_panels_without_finite_values_state_so_and_an_out_of_range_prediction_gets_its_own_scale() -> None:
    """An all-non-finite prediction panel is labelled; a locked scale yields to an unphysical prediction."""
    transient = analysis.evaluation.plots.transient
    record = _map_record()
    record.predicted_states[1, 0] = np.nan
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    assert [text.get_text() for text in figure.axes[1].texts] == ["unavailable: no finite values"]
    plt.close(figure)
    record = _map_record()
    locked = {"state_fields": ("T",), "time_indices": (1,), "error_mode": "Absolute error"}
    shared = transient.state_map_norms(record, lock_scale=True, **locked)["T"]
    assert shared.note is None
    assert (shared.reference.vmin, shared.reference.vmax) == (shared.prediction.vmin, shared.prediction.vmax)
    record.predicted_states[1, 0, 3, 3] = 5000.0
    own = transient.state_map_norms(record, lock_scale=True, **locked)["T"]
    assert own.note == "prediction outside physical range: own colour scale"
    assert own.prediction.vmax > own.reference.vmax
    assert transient.state_map_norms(record, lock_scale=False, **locked)["T"].note is None
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    assert any(text.get_text() == own.note for text in figure.axes[1].texts)
    plt.close(figure)


def test_error_panel_title_carries_the_mean_normalized_error_with_a_scale_and_na_without() -> None:
    """The mean |e|/s of the bed is printed with a train-only scale and marked n/a otherwise."""
    transient = analysis.evaluation.plots.transient
    record = sequence_record("autonomous_full", "full")
    scale = error_diagnostics.transient_reference_scale(
        scaling_state(),
        field="T",
        unit="K",
        state_order=analysis.evaluation.transient_artifact.STATE_ORDER,
        dataset_identity="a" * 64,
    )
    assert isinstance(scale, error_diagnostics.ReferenceScale)
    with_scale = transient.plot_state_maps(record, state_fields=("T",), error_mode="Signed error", reference_scales={"T": scale})
    without_scale = transient.plot_state_maps(record, state_fields=("T",), error_mode="Signed error")
    assert with_scale.axes[2].get_title() == "Signed error · mean |e|/s = 1"
    assert without_scale.axes[2].get_title() == "Signed error · mean |e|/s n/a"
    plt.close(with_scale)
    plt.close(without_scale)


def _gif_viewer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[session.TransientEvaluationSession, widgets.Button, Any]:
    """Return a sample viewer over one case with three varying stored times and an export folder in ``tmp_path``."""
    record = sequence_record("autonomous_full", "full")
    values = (0.3 * np.arange(3, dtype=np.float32)).reshape(3, 1, 1, 1) * np.ones((3, 4, 1, 1), dtype=np.float32)
    record.reference_states = values
    record.predicted_states = values + 0.1 + 0.05 * np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1)
    evaluation = session.TransientEvaluationSession({"id": production_frame((record,), artifact_root=str(tmp_path / "artifact"))})
    monkeypatch.setattr(analysis.ui.notebook, "_export_destination", lambda: tmp_path)
    viewer = analysis.evaluation.panel._make_transient_sample_viewer(
        session=evaluation,
        fields=analysis.evaluation.panel._transient_channel_fields(evaluation),
        comparison=False,
    )
    channel_row = viewer.children[1].children[1]
    channel_row.boxes["phi"].value = False
    channel_row.boxes["w_int"].value = False
    return evaluation, viewer.children[3].children[0].children[1], viewer


def test_gif_export_writes_all_stored_times_of_the_active_channels_with_one_frame_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two active channels and three stored times give one GIF with three equally sized, looping frames."""
    from PIL import Image  # noqa: PLC0415 -- Pillow reads the written file back; it is a matplotlib dependency

    evaluation, button, _viewer = _gif_viewer(tmp_path, monkeypatch)
    button.click()
    (path,) = tmp_path.glob("*.gif")
    assert "channel_t-w_surf" in path.name
    assert "t-all" in path.name
    assert button.disabled is False
    width, height = analysis.ui.plot_layout.MAP_LAYOUT.figure_size(rows=2, columns=3)
    with Image.open(path) as image:
        assert image.n_frames == 3
        assert image.info["loop"] == 0
        assert image.info["duration"] == 200
        assert image.size[0] == pytest.approx(width * 60, abs=1)
        assert image.size[1] == pytest.approx(height * 60, abs=1)
    evaluation.close()


def test_gif_export_refuses_above_the_memory_limit_and_reenables_the_button_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An oversized GIF creates no file; a failing frame removes the partial file and frees the button."""
    evaluation, button, _viewer = _gif_viewer(tmp_path, monkeypatch)
    monkeypatch.setattr(analysis.evaluation.panel, "_GIF_MEMORY_LIMIT_BYTES", 1)
    capsys.readouterr()
    button.click()
    assert "Refused" in capsys.readouterr().out
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
    evaluation.close()


def test_gif_export_never_writes_through_a_path_replaced_after_its_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A symlink swapped in at the reserved name is neither followed nor reported as the saved GIF."""
    evaluation, button, _viewer = _gif_viewer(tmp_path, monkeypatch)
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
    capsys.readouterr()
    button.click()
    status = capsys.readouterr().out
    assert decoy.read_bytes() == b"keep"
    assert "was replaced during the export" in status
    assert "Saved" not in status
    evaluation.close()


def test_comparison_ranking_requires_every_material_metric() -> None:
    """Do not rank a partially unavailable model using only its easier material."""
    rows = []
    for frame_name, values in (("Complete", (1.0, 3.0)), ("Incomplete", (0.1, float("nan")))):
        for material, value in zip(("lentil", "chickpea"), values, strict=True):
            rows.append(
                {
                    "frame": frame_name,
                    "material_family": material,
                    "requested_horizon": "full",
                    "scope": "cumulative",
                    "normalized_drying_group_macro_rmse": value,
                    **{f"normalized_rmse_{field}": value for field in analysis.evaluation.transient_artifact.STATE_ORDER},
                }
            )
    fake = SimpleNamespace(dataset_dataframe=lambda **_kwargs: pd.DataFrame(rows))
    ranked = analysis.evaluation.panel._transient_comparison_summary(cast("session.TransientEvaluationSession", fake))
    assert list(ranked["frame"]) == ["Complete", "Incomplete"]
    assert ranked.iloc[0]["normalized_drying_group_macro_rmse"] == pytest.approx(2.0)
    assert pd.isna(ranked.iloc[1]["normalized_drying_group_macro_rmse"])
    assert not ranked.iloc[1]["Metric available"]
    assert list(ranked["Material count"]) == [2, 2]
