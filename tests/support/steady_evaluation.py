"""Provide a compact synthetic steady Evaluation frame with payload files and complete provenance.

Responsibilities:
  - Write two small exact steady cases as payload files and build their provenance-complete artifact frame

Design principles:
  - Values are small, exact, and test-owned so panel, session, and dataset-cut tests share one fixture shape
  - Fields are analytically simple, so per-case errors and metrics follow from the case scale alone

This module does NOT:
  - Run inference, write a parquet table with real content, or represent production results
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src import analysis, common, domain, learning
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract

if TYPE_CHECKING:
    from pathlib import Path


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
