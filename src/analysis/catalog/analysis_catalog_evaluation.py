"""Catalog the steady evaluation plots on synthetic steady artifact frames.

Responsibilities:
  - Declare one catalog entry per steady evaluation sub-type, each drawn through its real builder
  - Build the compact two-case steady artifact frame (payload files, provenance, runtime evidence) the entries and the tests share
  - Derive the peer model and the models-on-datasets view the comparison plots need

Design principles:
  - Entries call the real builders on synthetic inputs, never on copies of them
  - Fields are analytically simple, so per-case errors and metrics follow from the case scale alone
  - Payload files live in the process scratch folder and are written once

This module does NOT:
  - Read storage, run inference, or represent production results
"""

from __future__ import annotations

import copy
import json
import os
from functools import cache
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
import torch

from src import analysis, common, domain, learning
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract

from .analysis_catalog_entries import CatalogPlot, scratch_directory

if TYPE_CHECKING:
    from pathlib import Path

_PLOTS: Final = "src.analysis.evaluation.evaluation_plot.evaluation_plot_"


def _runtime_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Build exact two-case steady runtime evidence for the fixture."""
    provenance = frame.attrs["artifact_provenance"]
    run = provenance["run"]
    solver_cases = {
        case_id: {"comsol_solver_s": solver, "comsol_process_s": process, "solver_hostname": "fixture-node", "solver_requested_cores": 4}
        for case_id, solver, process in (("case-a", 4.0, 10.0), ("case-b", None, 40.0))
    }
    return analysis.artifacts.timing.build_runtime_comparison(
        split_role="eval",
        dataset_identity=analysis.artifacts.timing.runtime_dataset_identity(provenance),
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
        solver_cases=solver_cases,
    )


def steady_frame(tmp_path: Path, *, pressure_drop: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Write two small exact steady cases with analytically simple fields and a declared inlet-to-outlet pressure drop."""
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
    recorded_payloads: list[dict[str, str]] = []
    scales = dict.fromkeys(task.output_names, 1.0)
    floors = {field: task.reference_floor(field) for field in task.output_names}
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
            p_in_bc=(pressure_drop * (1.0 - y_grid))[None],
            coordinates=coordinates,
            Rx=momentum_x,
            Ry=momentum_y,
            flux_balance=flux_balance,
        )
        recorded_payloads.append({"path": path.name, "sha256": common.serialization.cached_file_sha256(path)})
        row: dict[str, Any] = {
            "case_index": position + 1,
            "source_index": position,
            "split_local_index": position,
            "source_dataset_id": "steady-fixture-dataset",
            "simulation_case_id": ("case-a", "case-b")[position],
            "source_case_id": f"case_{position + 1:04d}",  # the compact Generation case id every steady Dataset package records
            "material_family": "lentil",
            "npz_path": str(path),
            "normalized_velocity_vector_rmse": float(np.sqrt(np.mean((prediction_speed - reference_speed) ** 2))),
            "physical_rmse_speed_magnitude": float(np.sqrt(np.mean((prediction_speed - reference_speed) ** 2))),
            "momentum_residual_mse": 5.0 * scale**2,
            "flux_balance_mse": (0.1 * scale) ** 2,
            "pressure_inlet_mse": 0.25 * scale**2,
            "pressure_outlet_mean_square": 0.75 * scale**2,
            "pressure_boundary_mse": scale**2,
            "parameters_bed.structure.coarse_len_rel": 0.1 * (position + 1),
            "prediction_validity": json.dumps(validity_contract.build_field_prediction_validity(prediction, channels=task.output_names)),
        }
        spacing = (float(y_values[1] - y_values[0]), float(x_values[1] - x_values[0]))
        pair = (torch.from_numpy(prediction)[None], torch.from_numpy(reference)[None])
        sums = learning.metrics.metrics.field_statistics(*pair, grid_spacing=spacing)[0]
        for field_index, field in enumerate(task.output_names):
            row.update(zip(analysis.artifacts.contracts.case_statistic_columns(field), sums[field_index].tolist(), strict=True))
        row.update(
            analysis.artifacts.contracts.case_metric_values(
                sums, fields=task.output_names, output_groups=task.output_groups, train_scales=scales, reference_floors=floors
            )
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    groups = analysis.artifacts.contracts.output_group_payload(task.output_groups)
    aggregate = analysis.artifacts.contracts.aggregate_normalized_group_macro_rmse(
        frame,
        output_groups=task.output_groups,
        train_scales=scales,
        reference_floors=floors,
    )
    table_path = tmp_path / "artifacts.parquet"
    table_path.write_bytes(b"table")
    membership_digest = analysis.artifacts.contracts.ordered_indices_sha256((0, 1))
    objective = next(metric for metric in task.default_metrics if metric.kind == "group_macro_rmse").as_dict(all_fields=task.output_names)
    provenance = {
        "producer": {"analysis_sha256": "d" * 64},
        "run": {
            "name": "steady-fixture",
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "effective_config_digest": "a" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "best_checkpoint_epoch": 7,
            "normalizer_sha256": "c" * 64,
        },
        "dataset": {
            "name": "steady-fixture-dataset",
            "role": "id",
            "source_dataset_ids": ["steady-fixture-dataset"],
            "source_identities": [{"dataset_id": "steady-fixture-dataset", "case_ids": ["case-a", "case-b"]}],
            "case_ids_by_source": [["case-a", "case-b"]],
            "membership_digests": [membership_digest],
            "evaluation_assignments": [],
        },
        "selection": {
            "effective_case_count": 2,
            "effective_ordered_source_indices_sha256": membership_digest,
        },
        "split_role": "eval",
        "runtime": {"inference_batch_size": 2},
        "model": {
            "kind": "fno",
            "physics_enabled": False,
            "parameter_counts": {"total": 8, "trainable": 8},
            "architecture": {},
        },
        "normalizer": {
            "output_scales": scales,
        },
        "evaluator": {
            "objective": objective,
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
            "input_fields": list(task.input_names),
            "input_units": {field.name: field.unit for field in task.inputs},
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
            "output_groups": groups,
            "metric_evidence": analysis.artifacts.contracts.metric_evidence_provenance(task),
        },
        "physics": {
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
        "evaluation": {"prediction_validity": validity_contract.table_role_prediction_validity(frame)},
        "outputs": {"parquet": {"path": table_path.name, "sha256": common.serialization.cached_file_sha256(table_path)}, "npz": recorded_payloads},
    }
    frame.attrs.update(
        {
            "provenance_complete": True,
            "artifact_provenance": provenance,
            "artifact_root": str(tmp_path),
            "task_id": task.id,
            "output_fields": task.output_names,
            "output_units": learned_units,
            "input_fields": task.input_names,
            "input_units": tuple(field.unit for field in task.inputs),
            "output_groups": tuple((group.id, group.fields) for group in task.output_groups),
            "train_scales": scales,
            "reference_floors": floors,
            "dataset_role": "eval",
            dataframe.PRIMARY_OBJECTIVE_ID: aggregate,
            # The loader attaches the completed run's summary; this one records only the run.
            dataframe.COMPLETED_RUN_SUMMARY_ATTR: {"task": task.id, "run_name": "steady-fixture", "ended_at": "2026-11-11T00:00:00+00:00"},
        }
    )
    provenance["timing"] = {"report": _runtime_payload(frame)}
    return frame, x_grid


@cache
def _models() -> dict[str, pd.DataFrame]:
    """Return the fixture frame and a peer with a distinct persisted model identity, keyed ``Model A`` and ``Model B``."""
    folder = scratch_directory() / "evaluation"
    folder.mkdir(exist_ok=True)
    frame, _grid = steady_frame(folder)
    peer = frame.copy(deep=True)
    peer.attrs = copy.deepcopy(frame.attrs)
    provenance = peer.attrs["artifact_provenance"]
    provenance["run"].update({"name": "steady-peer", "effective_config_digest": "f" * 64, "best_checkpoint_sha256": "1" * 64})
    provenance["model"].update({"parameter_counts": {"total": 16, "trainable": 16}, "architecture": {"modes": 8}, "kind": "uno"})
    return {"Model A": frame, "Model B": peer}


# A forked child writes its own frame files: the cached frames point into the parent's scratch folder, which the parent removes.
os.register_at_fork(after_in_child=_models.cache_clear)


def _shown() -> dataset_cuts.DisplayFrames:
    """Return the models on the datasets ID and OOD as display frames, every key ``<model> · <dataset>``."""
    cells = {}
    for dataset, role in (("ID", "eval"), ("OOD", "material_ood")):
        for model, source in zip(("A", "B"), _models().values(), strict=True):
            cut = source.copy(deep=True)
            cut.attrs = copy.deepcopy(source.attrs)
            cut.attrs["dataset_role"] = role
            cut.attrs["artifact_provenance"]["split_role"] = role
            cells[f"{model} · {dataset}"] = (cut, (dataset, model))
    return dataset_cuts.DisplayFrames({key: cut for key, (cut, _label) in cells.items()}, {key: label for key, (_cut, label) in cells.items()})


def _draw_map_grid_04() -> object:
    return analysis.evaluation.plots.error_behavior.plot_mean_error_maps(datasets=_models(), error_mode="MAE")


def _draw_map_grid_05() -> object:
    return analysis.evaluation.plots.samples_outliers.plot_sample_prediction_overview(datasets={"lentil, ID": _models()["Model A"]}, row_position=0)


def _draw_map_grid_06() -> object:
    return analysis.evaluation.plots.samples_outliers.plot_permeability_error_overlay(
        datasets={"lentil, ID": _models()["Model A"]}, row_position=0, channel="p"
    )


def _draw_map_grid_07() -> object:
    return analysis.evaluation.plots.samples_outliers.plot_pressure_velocity_comparison(
        datasets=_models(), row_position=0, models=("Model A", "Model B")
    )


def _draw_map_plus_lines_04() -> object:
    return analysis.evaluation.plots.physical_consistency.plot_brinkman_residual(datasets=_models())


def _draw_map_plus_lines_05() -> object:
    return analysis.evaluation.plots.physical_consistency.plot_spatial_residuals(datasets=_models())


def _draw_line_relationship_03() -> object:
    return analysis.evaluation.plots.error_behavior.plot_error_vs_target_magnitude(datasets=_shown())


def _draw_line_relationship_04() -> object:
    return analysis.evaluation.plots.sensitivity_capacity.plot_metadata_error_trends(datasets=_shown())


def _draw_distribution_04() -> object:
    return analysis.evaluation.plots.error_behavior.plot_global_error_metrics(datasets=_shown())


def _draw_distribution_05() -> object:
    return analysis.evaluation.plots.physical_consistency.plot_physical_consistency_cdf_grid(datasets=_shown())


def _draw_scatter_parity_04() -> object:
    shown = _shown()
    return analysis.evaluation.plots.run_summary.plot_accuracy_physics_pareto(
        datasets=shown
    ), analysis.evaluation.plots.sensitivity_capacity.plot_capacity_accuracy(datasets=shown)


def _draw_scatter_parity_05() -> object:
    return analysis.evaluation.plots.error_behavior.plot_mean_field_bias(datasets=_shown())


def _draw_bar_compare_02() -> object:
    return analysis.evaluation.plots.run_summary.plot_relative_comparison_scoreboard(datasets=_shown())


def _draw_bar_compare_03() -> object:
    return analysis.evaluation.plots.error_behavior.plot_boundary_error_decomposition(datasets=_shown(), channels=("p",))


def _draw_bar_compare_04() -> object:
    return analysis.evaluation.plots.physical_consistency.plot_pressure_boundary_summary(datasets=_shown())


def _draw_matrix_01() -> object:
    return analysis.evaluation.plots.sensitivity_capacity.plot_metadata_error_heatmap(datasets=_shown())


def _draw_spectral_05() -> object:
    return analysis.evaluation.plots.spectral_fidelity.plot_spectral_demand_prediction_error(datasets=_shown())


def _draw_spectral_06() -> object:
    return analysis.evaluation.plots.spectral_fidelity.plot_spectral_transfer_ratio(datasets=_shown())


def _draw_table_05() -> object:
    return analysis.evaluation.plots.run_summary.plot_run_summary_table(datasets=_models())


def _draw_own_03() -> object:
    return analysis.evaluation.plots.physical_consistency.plot_flux_balance(datasets=_shown())


ENTRIES: Final[tuple[CatalogPlot, ...]] = (
    CatalogPlot("map-grid_04", "Mean or spread of errors x models", (f"{_PLOTS}error_behavior.plot_mean_error_maps",), _draw_map_grid_04),
    CatalogPlot("map-grid_05", "Steady sample overview", (f"{_PLOTS}samples_outliers.plot_sample_prediction_overview",), _draw_map_grid_05),
    CatalogPlot("map-grid_06", "Permeability and error overlay", (f"{_PLOTS}samples_outliers.plot_permeability_error_overlay",), _draw_map_grid_06),
    CatalogPlot(
        "map-grid_07", "Pressure and velocity comparison", (f"{_PLOTS}samples_outliers.plot_pressure_velocity_comparison",), _draw_map_grid_07
    ),
    CatalogPlot("map-plus-lines_04", "Residual CDF and maps", (f"{_PLOTS}physical_consistency.plot_brinkman_residual",), _draw_map_plus_lines_04),
    CatalogPlot(
        "map-plus-lines_05", "Residual map and flux profile", (f"{_PLOTS}physical_consistency.plot_spatial_residuals",), _draw_map_plus_lines_05
    ),
    CatalogPlot(
        "line-relationship_03",
        "Error versus target magnitude",
        (f"{_PLOTS}error_behavior.plot_error_vs_target_magnitude",),
        _draw_line_relationship_03,
    ),
    CatalogPlot(
        "line-relationship_04", "Metadata sensitivity", (f"{_PLOTS}sensitivity_capacity.plot_metadata_error_trends",), _draw_line_relationship_04
    ),
    CatalogPlot("distribution_04", "Global error distributions", (f"{_PLOTS}error_behavior.plot_global_error_metrics",), _draw_distribution_04),
    CatalogPlot(
        "distribution_05",
        "Physical-consistency distributions",
        (f"{_PLOTS}physical_consistency.plot_physical_consistency_cdf_grid",),
        _draw_distribution_05,
    ),
    CatalogPlot(
        "scatter-parity_04",
        "Model accuracy/physics or capacity trade-off",
        (f"{_PLOTS}run_summary.plot_accuracy_physics_pareto", f"{_PLOTS}sensitivity_capacity.plot_capacity_accuracy"),
        _draw_scatter_parity_04,
    ),
    CatalogPlot("scatter-parity_05", "Mean field bias", (f"{_PLOTS}error_behavior.plot_mean_field_bias",), _draw_scatter_parity_05),
    CatalogPlot("bar-compare_02", "Relative model scoreboard", (f"{_PLOTS}run_summary.plot_relative_comparison_scoreboard",), _draw_bar_compare_02),
    CatalogPlot(
        "bar-compare_03", "Boundary error decomposition", (f"{_PLOTS}error_behavior.plot_boundary_error_decomposition",), _draw_bar_compare_03
    ),
    CatalogPlot(
        "bar-compare_04", "Pressure boundary summary", (f"{_PLOTS}physical_consistency.plot_pressure_boundary_summary",), _draw_bar_compare_04
    ),
    CatalogPlot("matrix_01", "Metadata correlation", (f"{_PLOTS}sensitivity_capacity.plot_metadata_error_heatmap",), _draw_matrix_01),
    CatalogPlot(
        "spectral_05", "Demand, prediction and error", (f"{_PLOTS}spectral_fidelity.plot_spectral_demand_prediction_error",), _draw_spectral_05
    ),
    CatalogPlot("spectral_06", "Transfer ratio", (f"{_PLOTS}spectral_fidelity.plot_spectral_transfer_ratio",), _draw_spectral_06),
    CatalogPlot("table_05", "Steady evaluation summary", (f"{_PLOTS}run_summary.plot_run_summary_table",), _draw_table_05),
    CatalogPlot("own_03", "Flux balance", (f"{_PLOTS}physical_consistency.plot_flux_balance",), _draw_own_03),
)
