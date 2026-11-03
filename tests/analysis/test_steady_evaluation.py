"""Protect steady Evaluation maps, metrics, timing, and shared widget state.

Responsibilities:
  - Check steady maps, metrics, timing bindings, exact grids, and shared controls

Design principles:
  - Matched-case grid identity is checked independently of lazy view admission

This module does NOT:
  - Train steady models or provide transient evaluation behavior
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import analysis, common, domain
from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader
from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_panel as panel

if TYPE_CHECKING:
    from matplotlib.patches import Rectangle

_CASE_COUNT = 2
_ORIGIN_MAIN_STEADY_BASELINE_COMMIT = "bf8058aefaa960dc34ffac569a6f67b59de1bf5c"


def _numeric_array(values: Any) -> np.ndarray:
    """Narrow third-party plotting array protocols for numeric assertions."""
    return np.asarray(values, dtype=float)


def _runtime_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Build exact two-case steady runtime evidence for the fixture."""
    provenance = frame.attrs["artifact_provenance"]
    dataset = provenance["dataset"]
    selection = provenance["selection"]
    run = provenance["run"]
    manifest_digest = "d" * 64
    simulation = {
        "schema_kind": analysis.artifacts.timing.SIMULATION_TIMING_SCHEMA_KIND,
        "schema_version": 1,
        "simulation_profile": "steady_flow",
        "batch_id": "fixture-batch",
        "batch_manifest_sha256": manifest_digest,
        "cases": [
            {"case_id": "case-a", "comsol_process_seconds": 10.0},
            {"case_id": "case-b", "comsol_process_seconds": 40.0},
        ],
        "aggregates": {
            "measured_case_count": 2,
            "mean_s": 25.0,
            "median_s": 25.0,
            "p10_s": 13.0,
            "p90_s": 37.0,
        },
    }
    return analysis.artifacts.timing.build_runtime_comparison(
        split_role="eval",
        dataset_identity={
            "name": dataset["name"],
            "fingerprint": dataset["fingerprint"],
            "data_contract_digest": dataset["data_contract_digest"],
            "saved_membership_digest": dataset["saved_membership_digest"],
            "effective_ordered_source_indices_sha256": selection["effective_ordered_source_indices_sha256"],
        },
        model_identity={
            "run_name": run["name"],
            "effective_config_digest": run["effective_config_digest"],
            "best_checkpoint_sha256": run["best_checkpoint_sha256"],
        },
        neural_runtime={
            "requested_policy": "cpu",
            "resolved_device": "cpu",
            "device_type": "cpu",
            "pytorch_version": "fixture",
            "hostname": "fixture-host",
            "platform": "fixture-platform",
            "processor": "fixture-cpu",
            "python_version": "3.11",
            "inference_dtype": "float32",
            "torch_num_threads": 1,
        },
        batch_size=2,
        cases=[
            {"case_id": "case-a", "source_index": 0, "batch_amortized_neural_operator_forward_s": 0.1},
            {"case_id": "case-b", "source_index": 1, "batch_amortized_neural_operator_forward_s": 0.2},
        ],
        comsol_timing=simulation,
        batch_manifest_sha256=manifest_digest,
        unavailable_reason=None,
    )


def _steady_frame(tmp_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Write two small exact steady cases with analytically simple fields."""
    task = domain.tasks.steady_flow.STEADY_FLOW
    height, width = 5, 6
    y_values = np.linspace(0.0, 1.0, height)
    x_values = np.linspace(0.0, 2.0, width)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    coordinates = np.stack((x_grid, y_grid))
    zero = np.zeros_like(x_grid)
    kappa = np.stack((np.full_like(x_grid, 1.0e-4), zero, np.full_like(x_grid, 2.0e-4)))
    kappa_encoded = np.stack((np.full_like(x_grid, -4.0), zero, np.full_like(x_grid, np.log10(2.0e-4))))
    input_by_field = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": kappa_encoded[0],
        "Kxy": kappa_encoded[1],
        "Kyy": kappa_encoded[2],
        "eps_bed": np.full_like(x_grid, 0.4),
        "p_in_bc": zero,
    }
    inputs = np.stack([input_by_field[field] for field in task.input_names])
    learned_units = tuple(field.unit for field in task.outputs)
    artifact_fields = (*task.output_names, "U")
    artifact_units = (*learned_units, "m/s")
    rows: list[dict[str, Any]] = []
    for position, scale in enumerate((1.0, 2.0)):
        reference = np.stack(
            (
                0.5 * scale * x_grid,
                0.5 * scale * x_grid,
                0.25 * scale * y_grid,
            )
        )
        prediction = np.stack(
            (
                scale * x_grid,
                scale * x_grid,
                0.5 * scale * y_grid,
            )
        )
        reference_speed = np.sqrt(reference[1] ** 2 + reference[2] ** 2)
        prediction_speed = np.sqrt(prediction[1] ** 2 + prediction[2] ** 2)
        prediction_all = np.concatenate((prediction, prediction_speed[None]), axis=0)
        reference_all = np.concatenate((reference, reference_speed[None]), axis=0)
        momentum_x = scale * np.ones_like(x_grid)
        momentum_y = 2.0 * scale * np.ones_like(x_grid)
        flux_balance = 0.1 * scale * np.ones(x_grid.shape[0])
        path = tmp_path / f"steady_case_{position + 1}.npz"
        np.savez_compressed(
            path,
            case_index=np.int64(position + 1),
            source_index=np.int64(position),
            split_local_index=np.int64(position),
            pred=prediction_all,
            gt=reference_all,
            err=prediction_all - reference_all,
            artifact_fields=np.asarray(artifact_fields),
            artifact_units=np.asarray(artifact_units),
            input_fields=np.asarray(task.input_names),
            output_fields=np.asarray(task.output_names),
            output_units=np.asarray(learned_units),
            x_raw=inputs,
            y_raw=reference,
            meta=np.asarray(json.dumps({"sample_id": ("case-a", "case-b")[position]})),
            kappa_encoded=kappa_encoded,
            kappa=kappa,
            kappa_names=np.asarray(("Kxx", "Kxy", "Kyy")),
            p_in_bc=zero[None],
            coordinates=coordinates,
            Rx=momentum_x,
            Ry=momentum_y,
            flux_balance=flux_balance,
        )
        row: dict[str, Any] = {
            "case_index": position + 1,
            "source_index": position,
            "split_local_index": position,
            "npz_path": str(path),
            "rel_l2": 0.1 * (position + 1),
            "rel_h1": 0.2 * (position + 1),
            "normalized_velocity_vector_rmse": float(np.sqrt(np.mean((prediction_speed - reference_speed) ** 2))),
            "physical_rmse_speed_magnitude": float(np.sqrt(np.mean((prediction_speed - reference_speed) ** 2))),
            "momentum_residual_mse": 5.0 * scale**2,
            "flux_balance_mse": (0.1 * scale) ** 2,
            "pressure_boundary_mse": scale**2,
            "parameters_bed.structure.coarse_len_rel": 0.1 * (position + 1),
        }
        for field_index, field in enumerate(task.output_names):
            values = prediction[field_index] - reference[field_index]
            squared_error = float(np.sum(values**2, dtype=np.float64))
            count = int(values.size)
            rmse = float(np.sqrt(squared_error / count))
            for columns in (
                analysis.artifacts.contracts.physical_statistic_columns(field),
                analysis.artifacts.contracts.normalized_statistic_columns(field),
            ):
                row.update(dict(zip(columns, (squared_error, count, rmse), strict=True)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    groups = analysis.artifacts.contracts.output_group_payload(task.output_groups)
    scales = dict.fromkeys(task.output_names, 1.0)
    aggregate = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
        frame,
        output_groups=task.output_groups,
        train_standard_deviations=scales,
        normalization_denominator_floor=0.0,
    )
    membership_digest = analysis.artifacts.contracts.ordered_indices_sha256((0, 1))
    objective = next(metric for metric in task.default_metrics if metric.kind == "group_macro_rmse").as_dict(all_fields=task.output_names)
    provenance = {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "run": {
            "name": "steady-fixture",
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "effective_config_digest": "a" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "normalizer_sha256": "c" * 64,
        },
        "dataset": {
            "name": "steady-fixture-dataset",
            "fingerprint": "e" * 64,
            "data_contract_digest": task.data_contract_digest,
            "saved_membership_digest": membership_digest,
        },
        "selection": {
            "effective_case_count": 2,
            "effective_ordered_source_indices_sha256": membership_digest,
        },
        "split_role": "eval",
        "runtime": {"batch_size": 2},
        "model": {
            "kind": "fno",
            "physics_enabled": False,
            "parameter_counts": {"total": 8, "trainable": 8},
            "architecture": {},
        },
        "normalizer": {
            "denominator_floor": 0.0,
            "output_standard_deviations": scales,
        },
        "evaluator": {
            "objective": objective,
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
            "input_fields": list(task.input_names),
            "input_units": {field.name: field.unit for field in task.inputs},
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
            "output_groups": groups,
            "predictive_metrics": {"rel_l2": "fixture", "rel_h1": "fixture"},
        },
        "physics": {
            "residual_schema_version": analysis.artifacts.contracts.RESIDUAL_SCHEMA_VERSION,
            "equation_kind": task.physics.kind,
            "continuity_kind": task.physics.continuity,
            "boundary_condition_kind": task.physics.boundary,
            "scales": {},
            "derivatives": {"kind": "fixture"},
            "interior_crop": analysis.artifacts.contracts.EVAL_PAD,
            "scalar_definitions": {},
            "array_definitions": {},
            "residual_evaluation_region": {},
        },
        "aggregate": aggregate,
        "outputs": {},
    }
    frame.attrs.update(
        {
            "provenance_complete": True,
            "artifact_provenance": provenance,
            "artifact_root": str(tmp_path),
            "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
            "task_id": task.id,
            "output_fields": task.output_names,
            "output_units": learned_units,
            "input_fields": task.input_names,
            "input_units": tuple(field.unit for field in task.inputs),
            "output_groups": tuple((group.id, group.fields) for group in task.output_groups),
            "train_standard_deviations": scales,
            "normalization_denominator_floor": 0.0,
            "dataset_role": "eval",
            "residual_schema_version": analysis.artifacts.contracts.RESIDUAL_SCHEMA_VERSION,
            dataframe.PRIMARY_OBJECTIVE_ID: aggregate,
        }
    )
    frame.attrs[dataframe.RUNTIME_COMPARISON_ATTR] = _runtime_payload(frame)
    return frame, x_grid


def test_steady_comparison_checks_every_matched_case_grid_after_lazy_panel_admission(tmp_path: Path) -> None:
    """A later case's changed coordinates cannot enter scoreboards or Pareto."""
    frame, _grid = _steady_frame(tmp_path)
    alternate = frame.copy(deep=True)
    original_path = Path(str(alternate.iloc[-1]["npz_path"]))
    with np.load(original_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["coordinates"] = arrays["coordinates"].copy()
    arrays["coordinates"][0] += 0.01
    alternate_path = tmp_path / "alternate_case_2.npz"
    np.savez_compressed(alternate_path, **arrays)
    alternate.loc[alternate.index[-1], "npz_path"] = str(alternate_path)
    datasets = {"A": frame, "B": alternate}
    dataframe.validate_comparison(datasets, verify_grid=False)
    with pytest.raises(dataframe.ComparisonCompatibilityError, match=r"case.*coordinate_digest"):
        dataframe.validate_comparison(datasets)


def test_exact_grid_cache_rejects_changed_payload_witness(tmp_path: Path) -> None:
    """A saved grid digest never excuses a changed source payload."""
    frame, _grid = _steady_frame(tmp_path)
    alternate = frame.copy(deep=True)
    datasets = {"A": frame, "B": alternate}
    dataframe.validate_comparison(datasets)
    path = Path(str(frame.iloc[-1]["npz_path"]))
    witness = path.stat()
    os.utime(path, ns=(witness.st_atime_ns, witness.st_mtime_ns + 1_000_000_000))
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="changed after exact grid admission"):
        dataframe.validate_comparison(datasets)


def test_mixed_case_grids_fail_single_and_identically_mixed_model_comparison(tmp_path: Path) -> None:
    """A scoreboard cannot accept a changing grid within one model's cases."""
    frame, _grid = _steady_frame(tmp_path)
    mixed = frame.copy(deep=True)
    source = Path(str(mixed.iloc[-1]["npz_path"]))
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["coordinates"] = arrays["coordinates"].copy()
    arrays["coordinates"][0] += 0.02
    changed = tmp_path / "mixed_case_2.npz"
    np.savez_compressed(changed, **arrays)
    mixed.loc[mixed.index[-1], "npz_path"] = str(changed)
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="mixes incompatible grids"):
        dataframe.validate_comparison({"A": mixed})
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="mixes incompatible grids"):
        dataframe.validate_comparison({"A": mixed, "B": _steady_peer(mixed)})


def _steady_peer(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one exact-case peer with a distinct persisted model identity."""
    peer = frame.copy(deep=True)
    peer.attrs = copy.deepcopy(frame.attrs)
    provenance = peer.attrs["artifact_provenance"]
    provenance["run"].update(
        {
            "name": "steady-peer",
            "effective_config_digest": "f" * 64,
            "best_checkpoint_sha256": "1" * 64,
        }
    )
    provenance["model"].update(
        {
            "parameter_counts": {"total": 16, "trainable": 16},
            "architecture": {"modes": 8},
        }
    )
    peer.attrs.pop(dataframe.RUNTIME_COMPARISON_ATTR, None)
    return peer


def test_steady_maps_metrics_timing_and_task_gating(tmp_path: Path) -> None:
    """Use exact fixture arrays for steady case, aggregate, timing, and gating checks."""
    frame, x_grid = _steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    case = cases.load_case(frame, 0)
    fields = analysis.evaluation.presentation.display_fields(frame)
    pressure = next(field for field in fields if field.key == "p")
    np.testing.assert_allclose(
        analysis.evaluation.presentation.case_field(case, pressure, source="reference"),
        0.5 * x_grid,
    )
    np.testing.assert_allclose(
        analysis.evaluation.presentation.case_field(case, pressure, source="prediction"),
        x_grid,
    )
    assert pressure.unit == "Pa"

    case_figure = analysis.evaluation.plots.samples_outliers.plot_sample_prediction_overview(
        datasets=datasets,
        row_position=0,
        error_mode="MAE",
        scale_mode="Independent",
    )
    titles = tuple(axis.get_title() for axis in case_figure.axes)
    assert any("pred [Pa]" in title for title in titles)
    assert any("true [Pa]" in title for title in titles)
    assert any("MAE [Pa]" in title for title in titles)
    plt.close(case_figure)

    with analysis.evaluation.session.EvaluationSession(datasets) as session:
        summary = session.full_summary(frame).require_spatial()
        pressure_index = summary.grid.fields.index("p")
        absolute_error_mean = summary.absolute_error_mean
        assert absolute_error_mean is not None
        np.testing.assert_allclose(absolute_error_mean[pressure_index], 0.75 * x_grid)
        aggregate_figure = analysis.evaluation.plots.error_behavior.plot_mean_error_maps(
            datasets=datasets,
            max_cases=2,
            error_mode="MAE",
        )
    assert "Mean absolute error maps" in aggregate_figure.get_suptitle()
    plt.close(aggregate_figure)

    expected_pressure_rmse = float(np.sqrt(0.625 * np.mean(x_grid**2)))
    aggregate = frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]
    assert aggregate["field_statistics"]["p"]["normalized_rmse"] == pytest.approx(expected_pressure_rmse)

    registry = panel._build_sections(datasets, comparison=False)
    assert "timing" not in registry
    assert not {"spatial", "rollout", "process", "outcomes"}.intersection(registry)
    titles = tuple(title.lower() for entries, _section_title in registry.values() for title, _factory, _stem in entries)
    assert not any(token in item for item in titles for token in ("trajectory", "rollout horizon", "physical time"))


def test_every_steady_registry_entry_builds_with_steady_fields_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke each visible steady entry once and expose only steady output fields."""
    frame, _x_grid = _steady_frame(tmp_path)
    monkeypatch.setattr(analysis.ui.notebook, "display", lambda _value: None)
    registry = panel._build_sections({"Steady fixture": frame}, comparison=False)
    entries = [entry for section_entries, _section_title in registry.values() for entry in section_entries]

    assert entries
    assert len({stem for _title, _factory, stem in entries}) == len(entries)
    try:
        for title, factory, stem in entries:
            analysis.ui.notebook._invoke_dropdown_entry(
                factory,
                export_state={"fig": None, "plot_name": stem, "title": title},
                export_plot_name=stem,
                export_title=title,
            )
    finally:
        plt.close("all")
    assert {field.key for field in analysis.evaluation.presentation.display_fields(frame)} == {"p", "u", "v", "velocity_magnitude"}


def test_steady_global_decomposition_capacity_and_map_values(tmp_path: Path) -> None:
    """Protect core global, spatial, architecture, and decomposition values."""
    frame, x_grid = _steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    plots = analysis.evaluation.plots
    with analysis.evaluation.session.EvaluationSession(datasets) as session:
        full = session.full_summary(frame).require_spatial()
        prefix = session.prefix_summary(frame, _CASE_COUNT)
    pressure_index = full.grid.fields.index("p")
    absolute_error_mean = full.absolute_error_mean
    signed_error_std = full.signed_error_std
    assert absolute_error_mean is not None
    assert signed_error_std is not None
    np.testing.assert_allclose(absolute_error_mean[pressure_index], 0.75 * x_grid)
    np.testing.assert_allclose(signed_error_std[pressure_index], 0.25 * x_grid)

    overview = plots.run_summary.build_run_summary_table(datasets)
    assert overview.loc["Steady fixture", "task_id"] == "steady_flow"
    assert overview.loc["Steady fixture", "sample_count"] == _CASE_COUNT
    assert overview.loc["Steady fixture", "normalized_group_macro_rmse"] == pytest.approx(frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"])

    global_figure = plots.error_behavior.plot_global_error_metrics(datasets=datasets)
    cdf_axes = {axis.get_title(): axis for axis in global_figure.axes if axis.get_title().endswith("CDF")}
    np.testing.assert_allclose(_numeric_array(cdf_axes["Relative L2 - CDF"].lines[0].get_xdata()), (0.1, 0.2))
    np.testing.assert_allclose(_numeric_array(cdf_axes["Relative L2 - CDF"].lines[0].get_ydata()), (0.5, 1.0))
    np.testing.assert_allclose(_numeric_array(cdf_axes["Relative H1 - CDF"].lines[0].get_xdata()), (0.2, 0.4))
    plt.close(global_figure)

    distribution = plots.error_behavior.plot_predictive_error_distributions(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_numeric_array(distribution.axes[0].lines[0].get_xdata()), (0.1, 0.2))
    np.testing.assert_allclose(_numeric_array(distribution.axes[0].lines[0].get_ydata()), (0.5, 1.0))
    for line, field in zip(
        distribution.axes[1].lines,
        analysis.evaluation.presentation.display_fields(frame),
        strict=True,
    ):
        source = prefix.magnitudes["velocity"].local_relative_error if field.is_magnitude else prefix.local_relative_error[field.key]
        np.testing.assert_allclose(_numeric_array(line.get_xdata()), source.quantiles)
        np.testing.assert_allclose(_numeric_array(line.get_ydata()), source.probabilities)
    plt.close(distribution)

    bias = plots.error_behavior.plot_mean_field_bias(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(bias.axes[0].collections[0].get_offsets()),
        ((0.5, 1.0), (1.0, 2.0)),
    )
    plt.close(bias)

    for figure, title in (
        (
            plots.error_behavior.plot_mean_error_maps(
                datasets=datasets,
                max_cases=_CASE_COUNT,
                error_mode="MAE",
            ),
            "Mean absolute error maps",
        ),
        (
            plots.error_behavior.plot_std_error_maps(
                datasets=datasets,
                max_cases=_CASE_COUNT,
            ),
            "Error standard-deviation maps",
        ),
    ):
        assert title in figure.get_suptitle()
        assert len(tuple(axis for axis in figure.axes if axis.get_title())) == 4
        plt.close(figure)

    architecture = plots.run_summary.build_architecture_table(datasets)
    assert architecture.loc["Steady fixture", "Architecture · Model family"] == "FNO"
    assert architecture.loc["Steady fixture", "Runtime/evidence · Trainable parameters"] == 8
    capacity = plots.sensitivity_capacity.plot_capacity_accuracy(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(capacity.axes[0].collections[0].get_offsets()),
        ((8.0, frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"]),),
    )
    plt.close(capacity)

    magnitude = plots.error_behavior.plot_error_vs_target_magnitude(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(magnitude.axes[0].lines[0].get_xdata()),
        prefix.target_magnitude_error["p"].centers,
    )
    np.testing.assert_allclose(
        _numeric_array(magnitude.axes[0].lines[0].get_ydata()),
        prefix.target_magnitude_error["p"].medians,
    )
    plt.close(magnitude)

    boundary = plots.error_behavior.plot_boundary_error_decomposition(
        datasets=datasets,
        max_cases=_CASE_COUNT,
        channels=("p",),
    )
    np.testing.assert_allclose(
        tuple(cast("Rectangle", patch).get_height() for patch in boundary.axes[0].patches),
        (1.25, np.nan, np.nan, 2.0, 1.0),
        equal_nan=True,
    )
    plt.close(boundary)


def test_steady_physical_spectral_and_sensitivity_values(tmp_path: Path) -> None:
    """Protect steady-only physics, spectral transfer, and parameter evidence."""
    frame, _x_grid = _steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    plots = analysis.evaluation.plots

    physical = plots.physical_consistency.build_physical_consistency_summary_table(datasets)
    expected_medians = {
        "momentum_residual_mse median": 12.5,
        "flux_balance_mse median": 0.025,
        "pressure_boundary_mse median": 2.5,
    }
    for column, expected in expected_medians.items():
        assert physical.loc["Steady fixture", column] == pytest.approx(expected)

    momentum = plots.physical_consistency.plot_brinkman_residual(datasets=datasets, max_cases=_CASE_COUNT)
    np.testing.assert_allclose(_numeric_array(momentum.axes[0].lines[0].get_xdata()), (5.0, 20.0))
    np.testing.assert_allclose(_numeric_array(momentum.axes[0].lines[0].get_ydata()), (0.0, 1.0))
    np.testing.assert_allclose(_numeric_array(momentum.axes[1].images[0].get_array()), 1.5 * np.sqrt(5.0))
    plt.close(momentum)

    flux = plots.physical_consistency.plot_flux_balance(datasets=datasets, max_cases=_CASE_COUNT)
    np.testing.assert_allclose(_numeric_array(flux.axes[0].lines[0].get_xdata()), (0.01, 0.04))
    np.testing.assert_allclose(_numeric_array(flux.axes[1].lines[0].get_xdata()), 0.15)
    plt.close(flux)

    cdf_grid = plots.physical_consistency.plot_physical_consistency_cdf_grid(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_numeric_array(cdf_grid.axes[0].lines[0].get_xdata()), (0.01, 0.04))
    np.testing.assert_allclose(_numeric_array(cdf_grid.axes[1].lines[0].get_xdata()), (5.0, 20.0))
    np.testing.assert_allclose(_numeric_array(cdf_grid.axes[3].lines[0].get_xdata()), (1.0, 4.0))
    plt.close(cdf_grid)

    pressure = plots.physical_consistency.plot_pressure_consistency(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_numeric_array(pressure.axes[1].lines[0].get_xdata()), (1.0, 4.0))
    assert pressure.axes[1].get_xlabel() == r"$\mathrm{MSE}(p_\Gamma-p_{bc})/\Delta p^2$ [1]"
    plt.close(pressure)

    demand = plots.spectral_fidelity.plot_spectral_demand_prediction_error(
        datasets=datasets,
        max_cases=_CASE_COUNT,
        channels=("p",),
        normalize=False,
    )
    reference_power = _numeric_array(demand.axes[0].lines[0].get_ydata())
    prediction_power = _numeric_array(demand.axes[0].lines[1].get_ydata())
    error_power = _numeric_array(demand.axes[1].lines[0].get_ydata())
    np.testing.assert_allclose(prediction_power, 4.0 * reference_power)
    np.testing.assert_allclose(error_power, reference_power)
    plt.close(demand)

    transfer = plots.spectral_fidelity.plot_spectral_transfer_ratio(
        datasets=datasets,
        max_cases=_CASE_COUNT,
        channels=("p",),
    )
    np.testing.assert_allclose(
        _numeric_array(transfer.axes[0].lines[0].get_ydata()),
        np.log10(4.0),
    )
    np.testing.assert_allclose(_numeric_array(transfer.axes[1].lines[0].get_ydata()), 1.0)
    plt.close(transfer)

    heatmap = plots.sensitivity_capacity.plot_metadata_error_heatmap(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_numeric_array(heatmap.axes[0].images[0].get_array()), 1.0)
    assert heatmap.axes[-1].get_ylabel() == "Spearman correlation"
    plt.close(heatmap)

    trend = plots.sensitivity_capacity.plot_metadata_error_trends(
        datasets=datasets,
        max_cases=_CASE_COUNT,
        channels=("p",),
    )
    assert np.isnan(_numeric_array(trend.axes[0].lines[0].get_xdata())).all()
    assert any(text.get_text() == "No resolved sensitivity for these cases" for text in trend.axes[0].texts)
    plt.close(trend)


def test_steady_sample_kappa_outlier_and_comparison_values(tmp_path: Path) -> None:
    """Protect exact selected cases, Kappa evidence, rankings, and paired fields."""
    frame, x_grid = _steady_frame(tmp_path)
    peer = _steady_peer(frame)
    datasets = {"Steady fixture": frame}
    plots = analysis.evaluation.plots.samples_outliers
    case = cases.load_case(frame, 0)
    assert case.permeability is not None
    np.testing.assert_allclose(case.permeability[0], 1.0e-4)
    np.testing.assert_allclose(case.permeability[1], 0.0)
    np.testing.assert_allclose(case.permeability[2], 2.0e-4)
    np.testing.assert_allclose(case.reference[0], 0.5 * x_grid)
    np.testing.assert_allclose(case.prediction[0], x_grid)

    linked_outlier = plots.plot_linked_outlier_cases(
        datasets=datasets,
        selection_index=0,
        channel="p",
    )
    assert linked_outlier.get_suptitle() == "p outlier field view — Case 2"
    plt.close(linked_outlier)

    linked_extreme = plots.plot_linked_input_extreme_cases(
        datasets=datasets,
        selection_index=0,
        parameter="parameters_bed.structure.coarse_len_rel",
    )
    assert linked_extreme.get_suptitle() == "Input extreme: Coarse correlation length minimum — Case 1"
    plt.close(linked_extreme)

    comparison = plots.plot_pressure_velocity_comparison(
        datasets={"Model A": frame, "Model B": peer},
        row_position=0,
    )
    assert comparison.get_suptitle() == "Pressure and velocity comparison — Case 1"
    assert tuple(axis.get_title() for axis in comparison.axes if axis.get_title()) == (
        "GT p [Pa]",
        r"GT $|\mathbf{u}|$ [m/s]",
        "Model A p [Pa]",
        r"Model A $|\mathbf{u}|$ [m/s]",
        "Model B p [Pa]",
        r"Model B $|\mathbf{u}|$ [m/s]",
    )
    plt.close(comparison)

    scoreboard = analysis.evaluation.plots.run_summary.plot_relative_comparison_scoreboard(
        datasets={"Model A": frame, "Model B": peer},
    )
    assert "Comparison-relative normalized error" in scoreboard.get_suptitle()
    plt.close(scoreboard)
    pareto = analysis.evaluation.plots.run_summary.plot_accuracy_physics_pareto(
        datasets={"Model A": frame, "Model B": peer},
    )
    assert tuple(axis.get_title() for axis in pareto.axes[:3]) == (
        "momentum_residual_mse",
        "flux_balance_mse",
        "pressure_boundary_mse",
    )
    plt.close(pareto)


def test_steady_runtime_binding_rejects_identity_drift_without_invalidating_core(tmp_path: Path) -> None:
    """Reject optional timing drift while retaining the admitted scientific frame."""
    frame, _x_grid = _steady_frame(tmp_path)
    payload = frame.attrs[dataframe.RUNTIME_COMPARISON_ATTR]
    provenance = frame.attrs["artifact_provenance"]
    role = artifact_loader._SavedRole(
        split_role="eval",
        dataset_name=provenance["dataset"]["name"],
        dataset_identity={
            **provenance["dataset"],
            "sample_ids": ("case-a", "case-b"),
        },
        source_indices=(0, 1),
        saved_membership_digest=provenance["dataset"]["saved_membership_digest"],
        root=tmp_path,
    )
    assert artifact_loader._validate_runtime_comparison_binding(payload, frame=frame, role=role) == payload

    for section, key in (("model_identity", "best_checkpoint_sha256"), ("dataset_identity", "fingerprint")):
        mismatched = copy.deepcopy(payload)
        mismatched[section][key] = "mismatch"
        with pytest.raises(ValueError, match="Dataset or checkpoint identity"):
            artifact_loader._validate_runtime_comparison_binding(mismatched, frame=frame, role=role)

    scientific_only = frame.copy()
    scientific_only.attrs.pop(dataframe.RUNTIME_COMPARISON_ATTR, None)
    invalid = copy.deepcopy(payload)
    invalid["dataset_identity"]["fingerprint"] = "mismatch"
    common.serialization.atomic_write_json(
        analysis.artifacts.timing.runtime_comparison_path(tmp_path),
        invalid,
    )
    artifact_loader._attach_optional_runtime_comparison(scientific_only, role=role)
    assert dataframe.RUNTIME_COMPARISON_ATTR not in scientific_only.attrs
    assert "Dataset or checkpoint identity" in scientific_only.attrs[dataframe.RUNTIME_COMPARISON_ERROR_ATTR]
    dataframe.validate_comparison({"Steady fixture": scientific_only})


def test_steady_payload_membership_binds_file_names_to_saved_dataset_case_ids(tmp_path: Path) -> None:
    """Admit NPZ files named by the saved Dataset case IDs and reject any other name in table or manifest."""
    sample_ids = ("lentil__case_0003__fe83e052fbf2", "ood_a::field_pea__case_0001__0123456789ab")
    names = ("lentil__case_0003__fe83e052fbf2.npz", "field_pea__case_0001__0123456789ab.npz")
    (tmp_path / "npz").mkdir()
    for name in names:
        (tmp_path / "npz" / name).write_bytes(b"payload")

    def role(ids: tuple[str, ...]) -> Any:
        return artifact_loader._SavedRole(
            split_role="additional_evaluation",
            dataset_name="ood_a",
            dataset_identity={"sample_ids": ids},
            source_indices=(0, 1),
            saved_membership_digest="digest",
            root=tmp_path,
        )

    def membership(table_names: tuple[str, ...], manifest_names: tuple[str, ...]) -> dict[str, Any]:
        frame = pd.DataFrame(
            {
                "source_index": [0, 1],
                "split_local_index": [0, 1],
                "case_index": [1, 2],
                "npz_path": [str(tmp_path / "npz" / name) for name in table_names],
            }
        )
        outputs = {"parquet": {"path": "ood_a.parquet"}, "npz": [{"path": f"npz/{name}"} for name in sorted(manifest_names)]}
        return {"frame": frame, "provenance": {"outputs": outputs}, "effective_indices": (0, 1)}

    artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names, names))
    with pytest.raises(ValueError, match="do not identify exact run-owned case payloads"):
        artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names[::-1], names))
    with pytest.raises(ValueError, match="manifest NPZ membership"):
        artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names, ("case_0001.npz", "case_0002.npz")))
    with pytest.raises(ValueError, match="package case ID"):
        artifact_loader._validate_payload_membership(role=role(("case-a", "case-b")), **membership(names, names))


class _SharedSelection:
    """Record exact case/channel publications from steady cached viewers."""

    def __init__(self) -> None:
        self.selection = SimpleNamespace(case_id="2", channels=("p",))

    def select_case(self, case_id: str) -> None:
        self.selection.case_id = case_id

    def select_channels(self, channels: tuple[str, ...]) -> None:
        self.selection.channels = channels


def test_steady_case_and_channel_controls_share_one_state_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synchronize exact case/channel state without transient-only controls."""
    frame, _x_grid = _steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    fields = analysis.evaluation.presentation.display_fields(frame)
    labels = tuple(field.label for field in fields)
    shared = _SharedSelection()
    monkeypatch.setattr(analysis.ui.viewers, "render_figure", lambda **_kwargs: None)

    def make_view() -> widgets.VBox:
        channel = analysis.ui.components.ui_dropdown_channel(channels=labels)
        view = panel._make_steady_indexed_viewer(
            lambda **_kwargs: None,
            datasets=datasets,
            controls={"channel": channel},
            selection_state=shared,  # type: ignore[arg-type]
            channel_fields=fields,
            export_plot_name="steady_case",
        )
        assert type(view) is widgets.VBox
        return view

    first = make_view()
    first_header = first.children[0]
    assert isinstance(first_header, widgets.HBox)
    first_case = next(child for child in first_header.children if isinstance(child, widgets.BoundedIntText))
    first_channel = next(child for child in first_header.children if isinstance(child, widgets.Dropdown))
    assert first_case.value == _CASE_COUNT
    assert first_channel.value == "p"

    second = make_view()
    second_header = second.children[0]
    assert isinstance(second_header, widgets.HBox)
    second_case = next(child for child in second_header.children if isinstance(child, widgets.BoundedIntText))
    second_channel = next(child for child in second_header.children if isinstance(child, widgets.Dropdown))

    first_case.value = 1
    first_channel.value = "|u|"
    assert shared.selection.case_id == "1"
    assert shared.selection.channels == ("velocity_magnitude",)
    cast("Any", second).activate()
    assert second_case.value == 1
    assert second_channel.value == "|u|"
    descriptions = {str(child.description) for child in (*first_header.children, *second_header.children) if hasattr(child, "description")}
    assert not {"Time:", "Protocol:", "Horizon:"}.intersection(descriptions)
