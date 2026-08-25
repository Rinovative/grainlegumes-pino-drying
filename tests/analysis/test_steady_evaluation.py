# ruff: noqa: PLR2004, S101, SLF001
"""Protect steady Evaluation maps, metrics, timing, and shared widget state."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from src import analysis, common, domain
from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader
from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_panel as panel

if TYPE_CHECKING:
    from pathlib import Path


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
            {"case_id": "case-a", "elapsed_seconds": 10.0},
            {"case_id": "case-b", "elapsed_seconds": 40.0},
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
            {"case_id": "case-a", "source_index": 0, "neural_operator_forward_s": 0.1},
            {"case_id": "case-b", "source_index": 1, "neural_operator_forward_s": 0.2},
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
        div_velocity = 0.1 * scale * np.ones_like(x_grid)
        div_eps_velocity = 0.04 * scale * np.ones_like(x_grid)
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
            div_u=div_velocity,
            div_eps_u=div_eps_velocity,
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
            "div_velocity_mse": (0.1 * scale) ** 2,
            "div_eps_velocity_mse": (0.04 * scale) ** 2,
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
            "selected_training_continuity": "div_velocity",
            "residual_schema_version": analysis.artifacts.contracts.RESIDUAL_SCHEMA_VERSION,
            "equation_kind": task.physics.kind,
            "boundary_condition_kind": task.physics.boundary,
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
    np.testing.assert_allclose(analysis.evaluation.presentation.case_error(case, pressure), 0.5 * x_grid)
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

    timing_table = analysis.evaluation.plots.run_summary.build_runtime_summary_table(datasets)
    assert timing_table.loc["Steady fixture", "neural_operator_forward_median_s"] == pytest.approx(0.15)
    assert timing_table.loc["Steady fixture", "comsol_solve_median_s"] == pytest.approx(25.0)
    assert timing_table.loc["Steady fixture", "speedup_median"] == pytest.approx(150.0)
    timing_figure = analysis.evaluation.plots.run_summary.plot_runtime_comparison(datasets=datasets)
    runtime_bars = [patch.get_height() for patch in timing_figure.axes[0].patches if isinstance(patch, Rectangle)]
    speedup_bars = [patch.get_height() for patch in timing_figure.axes[1].patches if isinstance(patch, Rectangle)]
    assert runtime_bars == pytest.approx([0.15, 25.0])
    assert speedup_bars == pytest.approx([150.0])
    plt.close(timing_figure)

    registry = panel._build_sections(datasets, comparison=False)
    assert "timing" not in registry
    assert not {"spatial", "rollout", "process", "outcomes"}.intersection(registry)
    titles = tuple(title.lower() for entries, _section_title in registry.values() for title, _factory, _stem in entries)
    assert not any(token in item for item in titles for token in ("trajectory", "rollout horizon", "physical time"))


def test_steady_registry_preserves_origin_main_section_and_plot_contract(tmp_path: Path) -> None:
    """Keep established section identity, titles, and plot order unchanged."""
    frame, _x_grid = _steady_frame(tmp_path)
    registry = panel._build_sections({"Steady fixture": frame}, comparison=False)
    assert tuple(registry) == panel.SINGLE_MODEL_EVALUATION_SECTION_KEYS, _ORIGIN_MAIN_STEADY_BASELINE_COMMIT
    assert tuple(title for _entries, title in registry.values()) == (
        "Overview",
        "Global Error Analysis",
        "Architecture Sensitivity",
        "Error Decomposition",
        "Physical Consistency",
        "Spectral & Representation Analysis",
        "Error Sensitivity",
        "Sample Viewer",
        "Outlier & Extreme Case Analysis",
    )
    assert tuple(title for entries, _section in registry.values() for title, _factory, _name in entries) == (
        "Overview: Summary table",
        "1-1. Global error metrics",
        "1-2. Global error distribution",
        "1-3. GT vs Prediction (mean)",
        "1-4. Mean error maps",
        "1-5. Std error maps",
        "2-1. Model-family configuration",
        "2-2. Exact trainable parameters vs performance",
        "3-1. Error vs |GT| magnitude",
        "3-2. Boundary vs interior error",
        "4-1. Physical consistency summary table",
        "4-2. Physical consistency CDF grid (2x2)",
        "4-3. Velocity continuity residual (∇·u)",
        "4-4. Porosity-weighted continuity residual (∇·(εu))",
        "4-5. Darcy–Brinkman operator residual",  # noqa: RUF001
        "4-6. Pressure and boundary-condition consistency",
        "5-1. Demand vs prediction + error",
        "5-2. Log spectral transfer with support",
        "6-1. Parameter-error correlation (heatmap)",
        "6-2. Error vs input parameter (binned trend)",
        "7-1. Sample GT vs Prediction",
        "7-2. Kappa tensor with error overlay",
        "8-1. Worst per-channel cases (tables)",
        "8-2. Worst per-channel cases (field plots)",
        "8-3. Extreme input parameters (table view)",
        "8-4. Extreme input parameter cases (field plots)",
    )


def test_steady_registry_uses_exact_origin_main_plot_owners_and_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind every steady entry to its verified baseline owner and local defaults."""
    frame, _x_grid = _steady_frame(tmp_path)
    captured: list[tuple[str, Any, dict[str, Any]]] = []

    def capture_shortcut(_datasets: dict[str, pd.DataFrame]) -> Any:
        def capture(
            title: str,
            owner: Any,
            *,
            plot_name: str | None = None,
            **kwargs: Any,
        ) -> tuple[str, Any, str]:
            captured.append((title, owner, kwargs))
            return title, lambda: None, plot_name or title

        return capture

    monkeypatch.setattr(analysis.ui.notebook, "make_toggle_shortcut", capture_shortcut)
    panel._build_sections({"Steady fixture": frame}, comparison=False)
    plots = analysis.evaluation.plots
    viewers = analysis.ui.viewers
    expected_owners = (
        (plots.run_summary.plot_run_summary_table, None),
        (plots.error_behavior.plot_global_error_metrics, None),
        (viewers.make_casecount_viewer, plots.error_behavior.plot_predictive_error_distributions),
        (viewers.make_casecount_viewer, plots.error_behavior.plot_mean_field_bias),
        (viewers.make_casecount_viewer, plots.error_behavior.plot_mean_error_maps),
        (viewers.make_casecount_viewer, plots.error_behavior.plot_std_error_maps),
        (plots.run_summary.plot_architecture_table, None),
        (viewers.make_casecount_viewer, plots.sensitivity_capacity.plot_capacity_accuracy),
        (viewers.make_casecount_viewer, plots.error_behavior.plot_error_vs_target_magnitude),
        (viewers.make_casecount_viewer, plots.error_behavior.plot_boundary_error_decomposition),
        (plots.physical_consistency.plot_physical_consistency_summary_table, None),
        (viewers.make_casecount_viewer, plots.physical_consistency.plot_physical_consistency_cdf_grid),
        (viewers.make_casecount_viewer, plots.physical_consistency.plot_velocity_divergence),
        (viewers.make_casecount_viewer, plots.physical_consistency.plot_div_eps_u_consistency),
        (viewers.make_casecount_viewer, plots.physical_consistency.plot_brinkman_residual),
        (viewers.make_casecount_viewer, plots.physical_consistency.plot_pressure_consistency),
        (viewers.make_casecount_viewer, plots.spectral_fidelity.plot_spectral_demand_prediction_error),
        (viewers.make_casecount_viewer, plots.spectral_fidelity.plot_spectral_transfer_ratio),
        (viewers.make_casecount_viewer, plots.sensitivity_capacity.plot_metadata_error_heatmap),
        (viewers.make_casecount_viewer, plots.sensitivity_capacity.plot_metadata_error_trends),
        (panel._make_steady_indexed_viewer, plots.samples_outliers.plot_sample_prediction_overview),
        (panel._make_steady_indexed_viewer, plots.samples_outliers.plot_permeability_error_overlay),
        (panel._make_outlier_table_viewer, plots.samples_outliers.plot_outlier_table),
        (viewers.make_indexed_viewer, plots.samples_outliers.plot_linked_outlier_cases),
        (plots.samples_outliers.plot_input_extremes_table, None),
        (viewers.make_indexed_viewer, plots.samples_outliers.plot_linked_input_extreme_cases),
    )
    assert tuple((owner, kwargs.get("plot_func")) for _title, owner, kwargs in captured) == expected_owners

    casecount_defaults = {
        "1-2. Global error distribution": (50, 50),
        "1-3. GT vs Prediction (mean)": (50, 50),
        "1-4. Mean error maps": (100, 50),
        "1-5. Std error maps": (100, 50),
        "2-2. Exact trainable parameters vs performance": (50, 50),
        "3-1. Error vs |GT| magnitude": (50, 50),
        "3-2. Boundary vs interior error": (100, 50),
        "4-2. Physical consistency CDF grid (2x2)": (100, 50),
        "4-3. Velocity continuity residual (∇·u)": (100, 50),
        "4-4. Porosity-weighted continuity residual (∇·(εu))": (100, 50),
        "4-5. Darcy–Brinkman operator residual": (100, 50),  # noqa: RUF001
        "4-6. Pressure and boundary-condition consistency": (100, 50),
        "5-1. Demand vs prediction + error": (50, 50),
        "5-2. Log spectral transfer with support": (50, 50),
        "6-1. Parameter-error correlation (heatmap)": (100, 50),
        "6-2. Error vs input parameter (binned trend)": (100, 50),
    }
    by_title = {title: kwargs for title, _owner, kwargs in captured}
    assert {
        title: (kwargs["start_cases"], kwargs["step_size"]) for title, kwargs in by_title.items() if "start_cases" in kwargs
    } == casecount_defaults
    assert by_title["3-1. Error vs |GT| magnitude"]["allow_dataset_selection"] is True
    assert by_title["3-1. Error vs |GT| magnitude"]["selector_title"] == "datasets"
    assert by_title["7-1. Sample GT vs Prediction"]["dataset_selection"] == "dropdown"
    assert by_title["7-2. Kappa tensor with error overlay"]["dataset_selection"] == "dropdown"
    assert by_title["8-2. Worst per-channel cases (field plots)"]["max_positions"] == 6
    assert by_title["8-4. Extreme input parameter cases (field plots)"]["max_positions"] == 2
    assert tuple(by_title["1-4. Mean error maps"]["controls"]) == ("error_mode",)
    assert tuple(by_title["5-1. Demand vs prediction + error"]["controls"]) == (
        "channels",
        "normalize",
    )
    assert by_title["5-1. Demand vs prediction + error"]["controls"]["normalize"].value is True
    assert tuple(by_title["7-2. Kappa tensor with error overlay"]["controls"]) == (
        "kappa_scale",
        "channel",
        "error_mode",
    )


def _widget_tree_contract(widget: widgets.Widget) -> tuple[Any, ...]:
    """Return stable visible widget semantics without model identities."""
    raw_options = getattr(widget, "options", ())
    options = tuple((str(option[0]), option[1]) if isinstance(option, tuple) and len(option) == 2 else option for option in raw_options)
    return (
        type(widget),
        getattr(widget, "description", None),
        options,
        getattr(widget, "value", None),
        cast("Any", widget).layout.get_state(),
        tuple(_widget_tree_contract(child) for child in getattr(widget, "children", ())),
    )


def _widget_visible_text(widget: widgets.Widget) -> tuple[str, ...]:
    """Return descriptions, option labels, and HTML text recursively."""
    values: list[str] = []
    for attribute in ("description", "value"):
        value = getattr(widget, attribute, None)
        if isinstance(value, str):
            values.append(value)
    for option in getattr(widget, "options", ()):
        label = option[0] if isinstance(option, tuple) and option else option
        if isinstance(label, str):
            values.append(label)
    for child in getattr(widget, "children", ()):
        values.extend(_widget_visible_text(child))
    return tuple(values)


def test_steady_indexed_viewer_matches_origin_main_widget_tree_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the baseline VBox tree while attaching nonvisual shared state."""
    frame, _x_grid = _steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    labels = tuple(field.label for field in analysis.evaluation.presentation.display_fields(frame))
    monkeypatch.setattr(analysis.ui.viewers, "render_figure", lambda **_kwargs: None)
    baseline = analysis.ui.viewers.make_indexed_viewer(
        lambda **_kwargs: None,
        datasets=datasets,
        controls={"channel": analysis.ui.components.ui_dropdown_channel(channels=labels)},
        dataset_selection="dropdown",
    )
    current = panel._make_steady_indexed_viewer(
        lambda **_kwargs: None,
        datasets=datasets,
        controls={"channel": analysis.ui.components.ui_dropdown_channel(channels=labels)},
        dataset_selection="dropdown",
    )

    assert type(baseline) is widgets.VBox
    assert type(current) is widgets.VBox
    assert _widget_tree_contract(current) == _widget_tree_contract(baseline)
    assert callable(getattr(current, "activate", None))


def test_steady_lazy_panel_and_no_transient_leakage_match_origin_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep lazy tabs, local View controls, export, and steady-only vocabulary."""
    frame, _x_grid = _steady_frame(tmp_path)
    shown: list[widgets.Widget] = []
    monkeypatch.setattr(analysis.ui.notebook, "display", shown.append)
    lazy = panel._build_panel(
        datasets={"Steady fixture": frame},
        comparison=False,
        sections="all",
    )

    assert isinstance(lazy, analysis.ui.notebook.LazyTabbedPanelOutput)
    assert lazy.tabs is not None
    tabs = lazy.tabs
    assert tuple(tabs.get_title(index) for index in range(len(tabs.children))) == (
        "Overview",
        "Global Error Analysis",
        "Architecture Sensitivity",
        "Error Decomposition",
        "Physical Consistency",
        "Spectral & Representation Analysis",
        "Error Sensitivity",
        "Sample Viewer",
        "Outlier & Extreme Case Analysis",
    )
    dropdowns = tuple(section.children[0] for section in tabs.children)
    assert all(type(dropdown) is widgets.Dropdown for dropdown in dropdowns)
    assert all(dropdown.description == "" for dropdown in dropdowns)
    assert all(dropdown.layout.width == "230px" for dropdown in dropdowns)
    assert all(dropdown.value is None for dropdown in dropdowns)
    assert tuple(label for label, _value in dropdowns[0].options) == ("Overview: Summary table",)
    assert tuple(label for label, _value in dropdowns[1].options) == (
        "1-1. Global error metrics",
        "1-2. Global error distribution",
        "1-3. GT vs Prediction (mean)",
        "1-4. Mean error maps",
        "1-5. Std error maps",
    )

    open_button = shown[-1]
    assert type(open_button) is widgets.Button
    assert open_button.description == "Single model - Open Evaluation"
    open_button.click()
    expanded = shown[-1]
    assert type(expanded) is widgets.VBox
    header = expanded.children[0]
    assert type(header) is widgets.HBox
    close_button, export_button = header.children
    assert close_button.description == "Close"
    assert export_button.description == "Export PDF"
    assert dropdowns[0].value == 0
    assert all(dropdown.value is None for dropdown in dropdowns[1:])

    visible_text = _widget_visible_text(expanded)
    assert tuple((field.key, field.label, field.unit) for field in analysis.evaluation.presentation.display_fields(frame)) == (
        ("p", "p", "Pa"),
        ("u", "u", "m/s"),
        ("v", "v", "m/s"),
        ("velocity_magnitude", "|u|", "m/s"),
    )
    assert "T" not in visible_text
    forbidden = (
        "Temporal & Rollout Analysis",
        "phi",
        "w_surf",
        "w_int",
        "Physical time:",
        "Protocol:",
        "Horizon:",
        "Drying",
    )
    assert not any(token in value for token in forbidden for value in visible_text)
    close_button.click()
    assert shown[-1] is open_button


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
        "div_velocity_mse median": 0.025,
        "div_eps_velocity_mse median": 0.004,
        "pressure_boundary_mse median": 2.5,
    }
    for column, expected in expected_medians.items():
        assert physical.loc["Steady fixture", column] == pytest.approx(expected)

    physical_plots = (
        (
            plots.physical_consistency.plot_velocity_divergence,
            (0.01, 0.04),
            0.15,
        ),
        (
            plots.physical_consistency.plot_div_eps_u_consistency,
            (0.0016, 0.0064),
            0.06,
        ),
        (
            plots.physical_consistency.plot_brinkman_residual,
            (5.0, 20.0),
            1.5 * np.sqrt(5.0),
        ),
    )
    for owner, expected_cdf, expected_map in physical_plots:
        figure = owner(datasets=datasets, max_cases=_CASE_COUNT)
        np.testing.assert_allclose(_numeric_array(figure.axes[0].lines[0].get_xdata()), expected_cdf)
        np.testing.assert_allclose(_numeric_array(figure.axes[0].lines[0].get_ydata()), (0.0, 1.0))
        np.testing.assert_allclose(_numeric_array(figure.axes[1].images[0].get_array()), expected_map)
        plt.close(figure)

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
    assert pressure.axes[1].get_xlabel() == r"$\mathrm{MSE}(p_\Gamma-p_{bc})$ [$Pa^2$]"
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

    sample = plots.plot_sample_prediction_overview(
        datasets=datasets,
        row_position=0,
        error_mode="MAE",
        scale_mode="Independent",
    )
    assert sample.get_suptitle() == "Sample GT vs prediction — Case 1"
    assert tuple(axis.get_title() for axis in sample.axes if axis.get_title()) == (
        "p pred [Pa]",
        "p true [Pa]",
        "p MAE [Pa]",
        "kappa [m²]",
        "u pred [m/s]",
        "u true [m/s]",
        "u MAE [m/s]",
        "log10(kappa / 1 m²)",
        "v pred [m/s]",
        "v true [m/s]",
        "v MAE [m/s]",
        r"$|\mathbf{u}|$ pred [m/s]",
        r"$|\mathbf{u}|$ true [m/s]",
        r"$|\mathbf{u}|$ MAE [m/s]",
    )
    plt.close(sample)

    kappa = plots.plot_permeability_error_overlay(
        datasets=datasets,
        row_position=0,
        channel="p",
        kappa_scale="log10(kappa)",
        error_mode="MAE",
    )
    assert kappa.get_suptitle() == "Kappa tensor with error overlay — Case 1"
    assert tuple(axis.get_title() for axis in kappa.axes if axis.get_title()) == (
        "log10(Kxx / 1 m²)",
        "Kxy [m²]",
        "p true [Pa]",
        "Kyx [m²]",
        "log10(Kyy / 1 m²)",
        "p MAE",
    )
    plt.close(kappa)

    outliers = plots.build_outlier_table(datasets, top_k=_CASE_COUNT)
    rel_l2 = outliers[outliers["metric"] == "rel_l2"]
    assert tuple(rel_l2["case_index"]) == (2, 1)
    assert tuple(rel_l2["value"]) == pytest.approx((0.2, 0.1))
    linked_outlier = plots.plot_linked_outlier_cases(
        datasets=datasets,
        selection_index=0,
        channel="p",
    )
    assert linked_outlier.get_suptitle() == "p outlier field view — Case 2"
    plt.close(linked_outlier)

    extremes = plots.build_input_extremes_table(datasets)
    low = extremes[extremes["extreme"] == "low"]
    high = extremes[extremes["extreme"] == "high"]
    assert tuple(low["case_index"]) == (1, 2)
    assert tuple(high["case_index"]) == (2, 1)
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
    assert tuple(axis.get_title() for axis in pareto.axes[:4]) == (
        "momentum_residual_mse",
        "div_velocity_mse",
        "div_eps_velocity_mse",
        "pressure_boundary_mse",
    )
    plt.close(pareto)


def test_steady_comparison_preserves_baseline_sections_and_local_plot_additions(
    tmp_path: Path,
) -> None:
    """Keep comparison-only baseline views within the established sections."""
    frame, _x_grid = _steady_frame(tmp_path)
    peer = _steady_peer(frame)
    single = panel._build_sections({"Model A": frame}, comparison=False)
    comparison = panel._build_sections(
        {
            "Model A": frame,
            "Model B": peer,
        },
        comparison=True,
    )

    assert tuple(comparison) == panel.COMPARISON_EVALUATION_SECTION_KEYS
    assert tuple(title for _entries, title in comparison.values()) == tuple(title for _entries, title in single.values())
    assert tuple(title for title, _factory, _name in comparison["overview"][0]) == (
        "Overview: Summary table",
        "Overview: Global comparison summary",
        "Overview: Pareto (Error vs Physics)",
    )
    assert tuple(title for title, _factory, _name in comparison["sample_viewer"][0]) == (
        "7-1. Sample GT vs Prediction",
        "7-2. Kappa tensor with error overlay",
        "7-3. Pressure & velocity field comparison",
    )
    for key in set(single).difference({"overview", "sample_viewer"}):
        assert tuple(title for title, _factory, _name in comparison[key][0]) == tuple(title for title, _factory, _name in single[key][0])


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
