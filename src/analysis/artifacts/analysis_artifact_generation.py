"""Create persistent evaluation artifacts from trained neural-operator runs.

Responsibilities:
  - Run inference over explicit evaluation loaders in saved membership order
  - Write Parquet scalar metrics and NPZ field artifacts
  - Compute field, physics, boundary and metadata diagnostics
  - Keep artifact columns stable for downstream analysis modules

Design principles:
  - Artifacts preserve split identity and saved membership order
  - Physical units and the caller-provided TaskSpec field order are explicit
  - Provenance publishes last and binds every completed payload digest
  - Heavy inference work stays out of plotting modules

This module does NOT:
  - Reconstruct models or choose the saved split membership to evaluate
  - Reuse, rebuild, lock, or upload an existing artifact cache target
  - Render interactive or curated scientific figures

Notes:
  Artifact contract:
    Artifact contents:
      - Parquet stores case-level scalar metrics, artifact paths and JSON-safe metadata
      - NPZ stores predictions, targets, errors, kappa fields, raw tensors and residual fields
      - every NPZ is named <package case id>.npz; case_index stays the one-based Dataset position
      - residual metrics use the same canonical evaluation crop for model comparability
    Parquet case table:
      - case_index        : stable one-based source case id
      - source_index      : zero-based position in the ordered concatenation of the role's complete source Datasets
      - source_dataset_id : source Dataset of the case (one role may span several additional Evaluation Datasets)
      - split_local_index : zero-based position in the saved split
      - npz_path          : path to the corresponding NPZ artifact
      Relative field errors (dimensionless, channel-balanced):
        - rel_l2            : channel-balanced mean relative L2 over TaskSpec outputs on the full domain
        - rel_h1            : channel-balanced mean relative H1 (NeuralOp H1Loss as in training) over TaskSpec outputs on the EVAL_PAD interior
      Physical diagnostics:
        - physical_rmse_<field> : per-field RMSE over the full domain
        - physical_velocity_vector_rmse : vector RMSE over velocity components
        - physical_rmse_speed_magnitude : RMSE of velocity-group speed magnitude |u|
      Dimensionless physics residual metrics (scales from inputs only):
        - momentum_residual_mse : mean(Rx**2 + Ry**2) over the EVAL_PAD-cropped interior
        - flux_balance_mse      : mean(flux_balance**2) over every grid row
      Dimensionless boundary metrics (full-grid inlet/outlet masks):
        - pressure_inlet_mse            : mean_inlet((p-p_in_bc)**2) / dp**2
        - pressure_outlet_mean_square   : (mean_outlet(p) / dp)**2
        - pressure_boundary_mse         : sum of the preceding two terms
      Global-reduction evidence (one pair of triplets per TaskSpec output):
        - physical_sse_<field>     : float64 per-case physical squared-error sum
        - physical_count_<field>   : exact selected physical element count
        - physical_rmse_<field>    : per-case physical diagnostic
        - normalized_sse_<field>   : float64 per-case normalized squared-error sum
        - normalized_count_<field> : exact selected normalized element count
        - normalized_rmse_<field>  : per-case normalized diagnostic
      Diagnostics:
        - kappa_names       : list of available permeability tensor components
        - meta              : JSON-safe metadata dictionary (stored as JSON string)
        - prediction_validity : VALID or NONFINITE document with per-channel counts;
          a NONFINITE row keeps its element counts and stores NaN for every other metric
    NPZ case fields:
      - case_index   : stable one-based source case id
      - source_index : zero-based original final-dataset index
      - split_local_index : zero-based position in the saved split
      - pred         : (C_artifact, H, W) prediction aligned with artifact_fields
      - gt           : (C_artifact, H, W) ground truth aligned with artifact_fields
      - err          : (C_artifact, H, W) prediction error (pred - gt)
      - artifact_fields : list[str] names aligned with pred/gt/err
      - artifact_units  : list[str] physical units aligned with artifact_fields
      - kappa_encoded: (C_kappa, H, W) task-stored permeability representations
      - kappa        : (C_kappa, H, W) physical permeability components
      - kappa_names  : list[str], same order as kappa channels
      - p_in_bc         : (1, H, W) pressure boundary condition
      - coordinates  : (2, H, W) physical x/y coordinate fields
      Declared inputs and targets retained for downstream analysis:
        - x_raw        : (C_in, H, W) raw input tensor (physical units)
        - y_raw        : (C_out, H, W) raw target tensor (physical units)
        - input_fields : list[str] canonical input channel names
        - output_fields: list[str] canonical learned-output names aligned with y_raw
        - output_units : list[str] physical units aligned with output_fields
      Dimensionless physics diagnostic fields (full grid, not cropped):
        - Rx           : (H, W) x-momentum residual divided by dp / L_y
        - Ry           : (H, W) y-momentum residual divided by dp / L_y
        - flux_balance : (H,) per-row cross-section flux deviation divided by U L_x
        - meta         : JSON string with full metadata
    Completion provenance:
      Provenance publishes last and contains the scientific identity, exact output
      digests, saved train-fitted output scales, task-owned output groups, the
      aggregate normalized_group_macro_rmse finalized once from global physical
      SSE/count evidence (counts instead of a value when any prediction is non-finite),
      and the role-level prediction validity under evaluation.

"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src import common, domain
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.learning import losses

from . import contracts, timing

if TYPE_CHECKING:
    from .analysis_artifact_performance import ArtifactProgressReporter

# ============================================================================
# Global constants
# ============================================================================

INTERNAL_KAPPA_NAMES = set(domain.permeability.INTERNAL_KAPPA_2D_ORDER) | set(domain.permeability.INTERNAL_KAPPA_3D_ORDER)
_MIN_NORMALIZED_TENSOR_RANK = 3
_MIN_BATCH_TENSOR_RANK = 2


def normalized_case_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    output_fields: Iterable[str],
) -> list[dict[str, float | int]]:
    """
    Return exact float64 normalized SSE/count/RMSE evidence per sample and field.

    The subtraction and square are performed after conversion to float64. These
    component-normalized values remain diagnostics and never define the grouped
    selection objective, which is finalized from physical evidence and raw fitted
    training scales.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Matching normalized tensors in batch, channel, and spatial order.
    output_fields : Iterable[str]
        Unique TaskSpec output names aligned with the channel axis.

    Returns
    -------
    list[dict[str, float | int]]
        One row per case containing each field's exact normalized squared-error
        sum, element count, and derived per-case RMSE. Non-finite predictions
        yield non-finite values, which case admission classifies.

    Raises
    ------
    ValueError
        If tensor shapes/ranks or channel-to-field alignment disagree.

    """
    fields = tuple(output_fields)
    if prediction.shape != target.shape or prediction.ndim < _MIN_NORMALIZED_TENSOR_RANK:
        msg = (
            "Normalized prediction and target must have matching "
            f"batch/channel/spatial shapes, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
        raise ValueError(msg)
    if prediction.shape[1] != len(fields) or len(fields) != len(set(fields)):
        msg = "Normalized artifact statistics require unique output fields matching the tensor channels."
        raise ValueError(msg)
    error_squared = (prediction.double() - target.double()).square()
    rows: list[dict[str, float | int]] = []
    for sample_index in range(prediction.shape[0]):
        row: dict[str, float | int] = {}
        for field_index, field_name in enumerate(fields):
            sse_column, count_column, rmse_column = contracts.normalized_statistic_columns(field_name)
            field_error = error_squared[sample_index, field_index]
            squared_error_sum = float(field_error.sum().detach().cpu().item())
            element_count = int(field_error.numel())
            field_rmse = math.sqrt(squared_error_sum / element_count)
            row[sse_column] = squared_error_sum
            row[count_column] = element_count
            row[rmse_column] = field_rmse
        rows.append(row)
    return rows


def physical_case_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    output_fields: Iterable[str],
) -> list[dict[str, float | int]]:
    """
    Return exact float64 physical SSE/count/RMSE evidence per sample and field.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Matching physical tensors in batch, channel, and spatial order.
    output_fields : Iterable[str]
        Unique TaskSpec output names aligned with the channel axis.

    Returns
    -------
    list[dict[str, float | int]]
        One row per case containing physical squared-error sums, element counts,
        and diagnostic RMSE values. Non-finite predictions yield non-finite
        values, which case admission classifies.

    Raises
    ------
    ValueError
        If tensor shapes, ranks, or channel-to-field alignment disagree.

    """
    fields = tuple(output_fields)
    if prediction.shape != target.shape or prediction.ndim < _MIN_NORMALIZED_TENSOR_RANK:
        msg = (
            "Physical prediction and target must have matching "
            f"batch/channel/spatial shapes, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
        raise ValueError(msg)
    if prediction.shape[1] != len(fields) or len(fields) != len(set(fields)):
        msg = "Physical artifact statistics require unique output fields matching the tensor channels."
        raise ValueError(msg)
    error_squared = (prediction.double() - target.double()).square()
    rows: list[dict[str, float | int]] = []
    for sample_index in range(prediction.shape[0]):
        row: dict[str, float | int] = {}
        for field_index, field_name in enumerate(fields):
            sse_column, count_column, rmse_column = contracts.physical_statistic_columns(field_name)
            field_error = error_squared[sample_index, field_index]
            squared_error_sum = float(field_error.sum().detach().cpu().item())
            element_count = int(field_error.numel())
            field_rmse = math.sqrt(squared_error_sum / element_count)
            row[sse_column] = squared_error_sum
            row[count_column] = element_count
            row[rmse_column] = field_rmse
        rows.append(row)
    return rows


def _required_output_group(
    task: domain.tasks.spec.TaskSpec,
    *,
    group_id: str,
) -> domain.tasks.spec.OutputGroupSpec:
    """Return one exact task-owned output group required by an artifact diagnostic."""
    matches = tuple(group for group in task.output_groups if group.id == group_id)
    if len(matches) != 1:
        available = {group.id: list(group.fields) for group in task.output_groups}
        msg = f"Artifact diagnostics require one TaskSpec output group {group_id!r}. Available groups: {available}."
        raise ValueError(msg)
    return matches[0]


def _objective_context(
    *,
    task: domain.tasks.spec.TaskSpec,
    processor: Any,
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """Return task groups, raw saved training scales, and diagnostic floor."""
    groups = contracts.output_group_payload(task.output_groups)
    flattened_fields = tuple(field for group in groups for field in group["fields"])
    if flattened_fields != task.output_names:
        msg = "Artifact output groups do not partition TaskSpec outputs in declared order."
        raise ValueError(msg)
    normalizer = provenance.get("normalizer")
    if not isinstance(normalizer, Mapping):
        msg = "Artifact provenance must contain normalizer evidence."
        raise TypeError(msg)
    raw_scales = normalizer.get("output_standard_deviations")
    denominator_floor = normalizer.get("denominator_floor")
    if isinstance(denominator_floor, bool) or not isinstance(denominator_floor, (int, float)):
        msg = "Artifact normalizer denominator_floor must be a real scalar."
        raise TypeError(msg)
    if not math.isfinite(float(denominator_floor)) or float(denominator_floor) < 0.0:
        msg = "Artifact normalizer denominator_floor must be finite and non-negative."
        raise ValueError(msg)
    if not isinstance(raw_scales, Mapping):
        msg = "Artifact normalizer provenance must contain output_standard_deviations."
        raise TypeError(msg)
    scales = {field: float(raw_scales[field]) for field in task.output_names if field in raw_scales}
    if set(scales) != set(task.output_names):
        msg = "Artifact normalizer provenance scales do not cover TaskSpec outputs exactly."
        raise ValueError(msg)
    out_normalizer = getattr(processor, "out_normalizer", None)
    if out_normalizer is None:
        msg = "Artifact generation requires processor.out_normalizer."
        raise TypeError(msg)
    processor_scales = contracts.output_standard_deviations_from_state(
        {"out_normalizer.std": getattr(out_normalizer, "std", None)},
        output_fields=task.output_names,
    )
    if processor_scales != scales:
        msg = "Artifact processor output scales contradict saved normalizer provenance."
        raise ValueError(msg)
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("output_groups") != groups:
        msg = "Artifact evaluator output groups contradict TaskSpec."
        raise ValueError(msg)
    return groups, scales, float(denominator_floor)


def _group_case_metrics(
    *,
    task: domain.tasks.spec.TaskSpec,
    physical_statistics: Mapping[str, float | int],
    output_groups: Iterable[Mapping[str, Any]],
    train_standard_deviations: Mapping[str, float],
) -> dict[str, float]:
    """Return task-declared physical and normalized group diagnostics for one case."""
    values: dict[str, float] = {}
    metrics_by_signature = {
        (metric.kind, metric.fields): metric.id for metric in task.default_metrics if metric.kind in {"group_rmse", "vector_rmse"}
    }
    for group in output_groups:
        fields = tuple(group["fields"])
        physical_mse_sum = sum(
            float(physical_statistics[contracts.physical_statistic_columns(field)[0]])
            / int(physical_statistics[contracts.physical_statistic_columns(field)[1]])
            for field in fields
        )
        normalized_id = metrics_by_signature.get(("group_rmse", fields))
        if normalized_id is not None:
            variance_sum = sum(train_standard_deviations[field] ** 2 for field in fields)
            values[normalized_id] = math.sqrt(physical_mse_sum / variance_sum)
        physical_id = metrics_by_signature.get(("vector_rmse", fields))
        if physical_id is not None:
            values[physical_id] = math.sqrt(physical_mse_sum)
    return values


def _provenance_with_aggregate(
    provenance: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    output_groups: Iterable[Mapping[str, Any]],
    train_standard_deviations: Mapping[str, float],
    normalization_denominator_floor: float,
) -> dict[str, Any]:
    """
    Add the authoritative group aggregate and role prediction validity to generation provenance.

    Caller-supplied aggregate or evaluation results are rejected so only recomputed
    physical Parquet SSE/count evidence, saved raw training scales and the per-row
    validity documents can determine the published selection objective.
    """
    payload = dict(provenance)
    if "aggregate" in payload or "evaluation" in payload:
        msg = "Caller-provided artifact provenance cannot define aggregate or evaluation results."
        raise ValueError(msg)
    payload["aggregate"] = contracts.aggregate_normalized_group_macro_rmse(
        frame,
        output_groups=output_groups,
        train_standard_deviations=train_standard_deviations,
        normalization_denominator_floor=normalization_denominator_floor,
    )
    payload["evaluation"] = {"prediction_validity": validity_contract.table_role_prediction_validity(frame)}
    return payload


def _admitted_case_metrics(prediction: torch.Tensor, metrics: Mapping[str, float | int], *, task: domain.tasks.spec.TaskSpec) -> dict[str, Any]:
    """
    Classify one physical prediction and admit its scalar metrics with the validity document.

    A valid case must have finite metrics. A non-finite case keeps its element counts
    and reports NaN for every other metric, so no missing value can read as a valid zero.
    """
    document = validity_contract.build_field_prediction_validity(prediction.squeeze(0).detach().cpu().numpy(), channels=task.output_names)
    counts = contracts.element_count_columns(task.output_names)
    if document["status"] == validity_contract.NONFINITE:
        metrics = {name: value if name in counts else math.nan for name, value in metrics.items()}
    elif not np.isfinite(np.asarray(list(metrics.values()), dtype=float)).all():
        msg = "Artifact case produced non-finite scalar metrics from a finite prediction."
        raise FloatingPointError(msg)
    return {**metrics, "prediction_validity": json.dumps(document)}


# =============================================================================
# JSON / type normalisation utilities
# =============================================================================


def meta_to_jsonable(obj: Any) -> Any:
    """
    Recursively convert supported tensor and NumPy values to JSON-native values.

    Parameters
    ----------
    obj : Any
        Tensor, NumPy value, mapping, list/tuple, or already JSON-native leaf.

    Returns
    -------
    Any
        Tensors and arrays become scalars/lists, NumPy scalars become Python
        scalars, and nested dictionaries/sequences are converted recursively.

    Notes
    -----
    Unrecognized leaf objects pass through unchanged. Callers that require JSON
    serialization must still supply JSON-compatible custom leaves.

    """
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return float(arr) if arr.ndim == 0 else arr.tolist()

    if isinstance(obj, np.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


def _value_has_batch_axis(value: Any, batch_size: int) -> bool:
    """Return whether a collated metadata value exposes the leading batch axis."""
    if isinstance(value, torch.Tensor | np.ndarray):
        return value.ndim > 0 and value.shape[0] == batch_size
    if isinstance(value, Mapping):
        return bool(value) and all(_value_has_batch_axis(item, batch_size) for item in value.values())
    return False


def _select_collated_case(value: Any, *, case_offset: int, batch_size: int) -> Any:
    """
    Slice one case from nested collated metadata while preserving batch rank.

    Tensor/array values with the declared leading batch size retain a size-one
    axis. Mappings and nested sequences recurse. Non-batched values pass through
    unchanged so dataset metadata structure is preserved beside ``x`` and ``y``.
    """
    if isinstance(value, torch.Tensor | np.ndarray):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[case_offset : case_offset + 1]
        return value
    if isinstance(value, Mapping):
        return {key: _select_collated_case(item, case_offset=case_offset, batch_size=batch_size) for key, item in value.items()}
    if isinstance(value, list | tuple):
        if value and all(_value_has_batch_axis(item, batch_size) for item in value):
            items = [_select_collated_case(item, case_offset=case_offset, batch_size=batch_size) for item in value]
            return tuple(items) if isinstance(value, tuple) else items
        if len(value) == batch_size:
            return value[case_offset : case_offset + 1]
        items = [_select_collated_case(item, case_offset=case_offset, batch_size=batch_size) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _iter_inferred_artifact_cases(
    loader: Iterable[Mapping[str, Any]],
    *,
    model: Any,
    processor: Any,
    device: torch.device,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """
    Infer each loader batch once and yield deterministic size-one case views.

    The fitted processor follows the online evaluation lifecycle exactly: it is
    switched to evaluation mode and preprocesses a copied raw batch before the
    model receives the complete loader batch. Predictions and prepared tensors
    are then sliced for one-row/one-NPZ storage without repeating model forwards.
    Batch forward duration is divided equally among its cases as an amortized
    throughput measurement.

    Parameters
    ----------
    loader : Iterable[Mapping[str, Any]]
        Ordered batches containing compatible tensor ``x`` and ``y`` entries.
    model : Any
        Evaluation-mode model receiving each complete preprocessed loader batch.
    processor : Any
        Fitted processor exposing ``eval()``, ``preprocess()``, and an output
        normalizer.
    device : torch.device
        Concrete inference device used by online and artifact evaluation.

    Yields
    ------
    tuple[int, dict[str, Any]]
        Contiguous artifact position and one-sample raw/preprocessed prediction
        mapping in saved loader order.

    Raises
    ------
    TypeError, ValueError, RuntimeError, FloatingPointError
        If a raw or preprocessed batch is malformed, empty, non-finite, or has
        incompatible prediction and target shapes.

    """
    processor_eval = getattr(processor, "eval", None)
    preprocess = getattr(processor, "preprocess", None)
    out_normalizer = getattr(processor, "out_normalizer", None)
    if not callable(processor_eval) or not callable(preprocess) or out_normalizer is None:
        msg = "Artifact generation requires an evaluation-capable fitted data processor."
        raise TypeError(msg)

    case_position = 0
    model.eval()
    for batch_index, raw_batch in enumerate(loader):
        if not isinstance(raw_batch, Mapping):
            msg = f"Artifact loader batches must be mappings, got {type(raw_batch).__name__}."
            raise TypeError(msg)
        raw_inputs = raw_batch.get("x")
        raw_targets = raw_batch.get("y")
        if not isinstance(raw_inputs, torch.Tensor) or not isinstance(raw_targets, torch.Tensor):
            msg = "Artifact loader batches must contain tensor keys 'x' and 'y'."
            raise TypeError(msg)
        if raw_inputs.ndim < _MIN_BATCH_TENSOR_RANK or raw_targets.ndim < _MIN_BATCH_TENSOR_RANK or raw_inputs.shape[0] != raw_targets.shape[0]:
            msg = f"Artifact input/target batches must share a non-empty leading axis, got {tuple(raw_inputs.shape)} and {tuple(raw_targets.shape)}."
            raise ValueError(msg)
        batch_size = int(raw_inputs.shape[0])
        if batch_size <= 0:
            msg = "Artifact loader produced an empty batch."
            raise RuntimeError(msg)
        _require_finite_artifact_tensor(raw_inputs, label=f"Artifact batch {batch_index} raw inputs")
        _require_finite_artifact_tensor(raw_targets, label=f"Artifact batch {batch_index} raw targets")

        with torch.inference_mode():
            processor_eval()
            processed = preprocess(dict(raw_batch))
            if not isinstance(processed, Mapping):
                msg = f"Artifact processor must return a mapping, got {type(processed).__name__}."
                raise TypeError(msg)
            normalized_inputs = processed.get("x")
            targets = processed.get("y")
            if not isinstance(normalized_inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                msg = "Artifact processor output must contain tensor keys 'x' and 'y'."
                raise TypeError(msg)
            normalized_inputs = normalized_inputs.to(device)
            targets = targets.to(device)
            inputs = raw_inputs.to(device)
            if (
                normalized_inputs.ndim < _MIN_BATCH_TENSOR_RANK
                or targets.ndim < _MIN_BATCH_TENSOR_RANK
                or normalized_inputs.shape[0] != batch_size
                or targets.shape[0] != batch_size
            ):
                msg = "Artifact preprocessing must preserve the complete loader batch axis."
                raise ValueError(msg)
            normalized_prediction, batch_forward_s = timing.measure_forward(
                model=model,
                normalized_inputs=normalized_inputs,
                device=device,
            )
            if not isinstance(normalized_prediction, torch.Tensor):
                msg = "Artifact model output must be a tensor."
                raise TypeError(msg)
            normalized_target = out_normalizer.transform(targets)
            prediction = out_normalizer.inverse_transform(normalized_prediction)

        # Predictions stay unchecked here: case admission classifies them without changing values.
        tensors = {
            "normalized inputs": normalized_inputs,
            "physical targets": targets,
            "normalized target": normalized_target,
        }
        for label, value in tensors.items():
            _require_finite_artifact_tensor(value, label=f"Artifact batch {batch_index} {label}")
        if prediction.shape != targets.shape or normalized_prediction.shape != normalized_target.shape:
            msg = (
                "Artifact prediction/target shapes disagree after complete-batch inference: "
                f"physical={tuple(prediction.shape)}/{tuple(targets.shape)}, "
                f"normalized={tuple(normalized_prediction.shape)}/{tuple(normalized_target.shape)}."
            )
            raise RuntimeError(msg)
        if prediction.shape[0] != batch_size:
            msg = "Artifact model output must preserve the complete loader batch axis."
            raise RuntimeError(msg)

        amortized_forward_s = batch_forward_s / batch_size
        for case_offset in range(batch_size):
            case = {
                "x": inputs[case_offset : case_offset + 1],
                "y": targets[case_offset : case_offset + 1],
                "y_normalized": normalized_target[case_offset : case_offset + 1],
                "prediction_normalized": normalized_prediction[case_offset : case_offset + 1],
                "prediction": prediction[case_offset : case_offset + 1],
                "amortized_forward_s": amortized_forward_s,
                "forward_calls": int(case_offset == 0),
                "batch_forward_s": batch_forward_s if case_offset == 0 else 0.0,
                "source_index": _select_collated_case(
                    raw_batch.get("source_index"),
                    case_offset=case_offset,
                    batch_size=batch_size,
                ),
                "split_local_index": _select_collated_case(
                    raw_batch.get("split_local_index"),
                    case_offset=case_offset,
                    batch_size=batch_size,
                ),
                "meta": _select_collated_case(
                    raw_batch.get("meta", {}),
                    case_offset=case_offset,
                    batch_size=batch_size,
                ),
            }
            yield case_position, case
            case_position += 1


def write_artifact_provenance(save_root: str | Path, provenance: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """
    Atomically publish provenance and payload digests as the completion marker.

    Parameters
    ----------
    save_root : str | pathlib.Path
        Fully written staged artifact target.
    provenance : Mapping[str, Any]
        Scientific identity to enrich with exact output digests.

    Returns
    -------
    tuple[pathlib.Path, dict[str, Any]]
        Newly published path and the exact JSON-safe persisted payload including
        its output manifest.

    Raises
    ------
    FileExistsError
        If a completion marker already exists. Provenance is never overwritten.
    RuntimeError
        If the artifact payload is incomplete.

    Notes
    -----
    Publishing this sidecar last is the cache-completion transaction boundary.

    """
    provenance_path = contracts.artifact_provenance_path(save_root)
    if provenance_path.exists():
        msg = f"Refusing to overwrite existing artifact provenance: {provenance_path}"
        raise FileExistsError(msg)

    payload = meta_to_jsonable(dict(provenance))
    if not isinstance(payload, dict):
        msg = "Artifact provenance must normalise to a JSON object."
        raise TypeError(msg)
    if "outputs" in payload:
        msg = "Caller-provided artifact provenance cannot define output digests."
        raise ValueError(msg)
    payload["outputs"] = contracts.artifact_output_manifest(save_root)

    published_path = common.serialization.atomic_write_json(provenance_path, payload)
    return published_path, payload


def _require_batch_scalar_int(batch: Mapping[str, Any], key: str) -> int:
    """
    Extract one exact integer identity from a size-one artifact case mapping.

    Torch, NumPy, singleton-sequence, and native integer encodings are accepted.
    Booleans, floating values, missing keys, and multi-value containers fail so
    persisted membership identity is never coerced ambiguously.
    """
    if key not in batch:
        msg = f"Artifact batches must provide top-level {key!r} identity."
        raise KeyError(msg)

    value = batch[key]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            msg = f"Artifact batch {key!r} must contain exactly one integer value, not {value!r}."
            raise TypeError(msg)
        return int(value.detach().cpu().item())

    if isinstance(value, np.ndarray):
        if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
            msg = f"Artifact batch {key!r} must contain exactly one integer value, not {value!r}."
            raise TypeError(msg)
        return int(value.reshape(-1)[0])

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            msg = f"Artifact batch {key!r} must contain exactly one integer value, not {value!r}."
            raise TypeError(msg)
        value = value[0]

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"Artifact batch {key!r} must be an integer, not {type(value).__name__}."
        raise TypeError(msg)
    return int(value)


def _require_finite_artifact_tensor(value: torch.Tensor, *, label: str) -> None:
    """Reject non-floating or non-finite tensors before artifact publication."""
    if not value.is_floating_point() or value.is_complex():
        msg = f"{label} must be a real floating-point tensor."
        raise TypeError(msg)
    if not torch.isfinite(value).all():
        msg = f"{label} contains non-finite values."
        raise FloatingPointError(msg)


def _artifact_complete_case_count(provenance: Mapping[str, Any]) -> int:
    """
    Require complete saved-membership selection evidence before generation.

    Authoritative artifacts cannot represent bounded prefixes. Full and effective
    counts and ordered-membership digests must therefore be identical, while the
    explicit generation limit remains ``None``.
    """
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping):
        msg = "Artifact provenance must contain a selection mapping."
        raise TypeError(msg)

    full_count = selection.get("full_selected_case_count")
    effective_count = selection.get("effective_case_count")
    if (
        isinstance(full_count, bool)
        or not isinstance(full_count, int)
        or full_count <= 0
        or isinstance(effective_count, bool)
        or not isinstance(effective_count, int)
        or effective_count != full_count
    ):
        msg = "Artifact selection must cover the complete positive saved membership."
        raise ValueError(msg)
    if "generation_limit" not in selection or selection["generation_limit"] is not None:
        msg = "Authoritative artifact generation requires generation_limit=None."
        raise ValueError(msg)
    full_digest = selection.get("full_ordered_source_indices_sha256")
    effective_digest = selection.get("effective_ordered_source_indices_sha256")
    if not isinstance(full_digest, str) or not full_digest or effective_digest != full_digest:
        msg = "Artifact full and effective ordered-membership digests must be identical."
        raise ValueError(msg)
    return effective_count


def _validate_generated_source_indices(provenance: Mapping[str, Any], source_indices: list[int]) -> None:
    """
    Prove generated loader order matches the provenance membership digest.

    The complete ordered source-index sequence participates in the digest, so
    reordering, omission, substitution, or duplication fails before provenance
    can publish the target as complete.
    """
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping):
        msg = "Artifact provenance must contain a selection mapping."
        raise TypeError(msg)
    expected_digest = selection.get("effective_ordered_source_indices_sha256")
    if not isinstance(expected_digest, str) or not expected_digest:
        msg = "Artifact provenance must contain an effective ordered source-index digest."
        raise TypeError(msg)
    actual_digest = contracts.ordered_indices_sha256(source_indices)
    if actual_digest != expected_digest:
        msg = f"Generated ordered source_index values do not match artifact provenance: expected digest {expected_digest}, got {actual_digest}."
        raise RuntimeError(msg)


def _ensure_artifact_targets_absent(save_root: Path, dataset_name: str) -> None:
    """
    Enforce create-only semantics for one staged artifact target.

    Existing Parquet, provenance, NPZ, or interrupted temporary files are all
    reported together and rejected. Generation never overwrites partial output.
    """
    npz_dir = save_root / "npz"
    candidates = [
        save_root / f"{dataset_name}.parquet",
        contracts.artifact_provenance_path(save_root),
        *sorted(npz_dir.glob("*.npz")),
        *sorted(save_root.glob(".*.tmp")),
        *sorted(npz_dir.glob(".*.tmp")),
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        msg = f"Refusing to overwrite existing or interrupted artifacts:\n{formatted}"
        raise FileExistsError(msg)


# =============================================================================
# Kappa field utilities (fields only, no scalar statistics)
# =============================================================================


def detect_kappa_channels_from_inputs(include_inputs: list[str]) -> list[str]:
    """
    Detect permeability-related input channels based on their names.

    Parameters
    ----------
    include_inputs : list[str]
        List of canonical input channel names.

    Returns
    -------
    list[str]
        Names of all channels that represent permeability components.

    """
    return [name for name in include_inputs if name in INTERNAL_KAPPA_NAMES]


def extract_kappa(
    x_tensor: torch.Tensor,
    *,
    input_fields: list[str],
    kappa_names: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract stored and physical permeability fields from a BCHW input tensor.

    Parameters
    ----------
    x_tensor : torch.Tensor
        Input tensor with shape ``(batch, input_field, y, x)``.
    input_fields : list[str]
        Canonical input-channel names aligned with the tensor channel axis.
    kappa_names : list[str]
        Ordered permeability components to extract. An empty list returns
        zero-channel tensors with the original batch and spatial dimensions.

    Returns
    -------
    dict[str, torch.Tensor]
        ``kappa_encoded`` retains stored task representations, while ``kappa`` contains
        physical square-metre components in the same channel order.

    Raises
    ------
    KeyError
        If requested permeability names are absent or physical reconstruction
        lacks required ``kxx``/``kyy`` diagonal components.

    Notes
    -----
    Diagonals are reconstructed as ``10**stored``. ``kxy`` is its stored
    dimensionless ratio times ``sqrt(kxx * kyy)``.

    """
    if not kappa_names:
        return {
            "kappa_encoded": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
            "kappa": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
        }

    index_map = {name: i for i, name in enumerate(input_fields)}
    kappa_indices = [index_map[name] for name in kappa_names]

    # Task-encoded permeability representations as stored in the dataset.
    kappa_encoded = x_tensor[:, kappa_indices, :, :]

    # Physical permeability reconstruction in square metres.
    kappa_phys = torch.zeros_like(kappa_encoded)
    name_to_pos = {name: i for i, name in enumerate(kappa_names)}

    # kxx, kyy (always log10-physical)
    kxx = torch.pow(10.0, kappa_encoded[:, name_to_pos["Kxx"]])
    kyy = torch.pow(10.0, kappa_encoded[:, name_to_pos["Kyy"]])
    kappa_phys[:, name_to_pos["Kxx"]] = kxx
    kappa_phys[:, name_to_pos["Kyy"]] = kyy

    # kxy is a dimensionless ratio to sqrt(kxx * kyy).
    if "Kxy" in name_to_pos:
        kxy_ratio = kappa_encoded[:, name_to_pos["Kxy"]]
        kappa_phys[:, name_to_pos["Kxy"]] = kxy_ratio * torch.sqrt(kxx * kyy)

    return {
        "kappa_encoded": kappa_encoded,
        "kappa": kappa_phys,
    }


# =============================================================================
# Run-directory utilities
# =============================================================================


def infer_current_run_dir(save_root: Path) -> Path:
    """
    Find the nearest current-contract run ancestor of an artifact path.

    Parameters
    ----------
    save_root : pathlib.Path
        Artifact or staging path from which to walk toward the filesystem root.

    Returns
    -------
    pathlib.Path
        Nearest current run ancestor, or the original path when none is found.

    Notes
    -----
    The result is used only for diagnostic metadata and logging. Model, loader,
    processor, and device ownership remain explicit caller inputs.

    """
    candidate = Path(save_root)
    while candidate.parent != candidate:
        if common.paths.is_evaluable_run_dir(candidate):
            return candidate
        candidate = candidate.parent
    return Path(save_root)


# =============================================================================
# Main artifact generator
# =============================================================================


def _artifact_case_ids(
    *,
    case_ids: Sequence[str],
    provenance: Mapping[str, Any],
    timing_cases: list[dict[str, Any]] | None,
    expected_case_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the authoritative case IDs against the provenance sources; return them with each case's source Dataset."""
    if timing_cases:
        msg = "Artifact timing collector must be empty at generation start."
        raise ValueError(msg)
    ids = tuple(case_ids)
    if len(ids) != expected_case_count or len(ids) != len(set(ids)):
        msg = "Artifact case IDs must exactly match unique effective membership."
        raise ValueError(msg)
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        msg = "Artifact case IDs must be non-empty strings."
        raise TypeError(msg)
    dataset = provenance["dataset"]
    sources = tuple(zip(dataset["source_dataset_ids"], dataset["case_ids_by_source"], strict=True))
    if tuple(case_id for _, source_case_ids in sources for case_id in source_case_ids) != ids:
        msg = "Artifact case IDs contradict the ordered source membership in their provenance."
        raise ValueError(msg)
    return ids, tuple(str(dataset_id) for dataset_id, source_case_ids in sources for _ in source_case_ids)


def _record_case(
    case: Mapping[str, Any],
    *,
    case_id: str,
    source_index: int,
    timing_cases: list[dict[str, Any]] | None,
    progress_reporter: ArtifactProgressReporter | None,
) -> None:
    """Record one finished case's batch-amortized forward share and its progress evidence."""
    if timing_cases is not None:
        timing_cases.append(
            {
                "case_id": case_id,
                "source_index": source_index,
                "batch_amortized_neural_operator_forward_s": float(case["amortized_forward_s"]),
            }
        )
    if progress_reporter is not None:
        forward_calls = int(case["forward_calls"])
        progress_reporter.case_completed(
            case_id=case_id,
            material=case_id.split("__", 1)[0],
            forward_calls=forward_calls,
            timed_forward_calls=forward_calls,
            model_forward_seconds=float(case["batch_forward_s"]),
        )


def _generate_steady_flow_artifacts(  # noqa: PLR0915
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    case_ids: Sequence[str],
    publication_root: str | Path | None = None,
    timing_cases: list[dict[str, Any]] | None = None,
    progress_reporter: ArtifactProgressReporter | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate steady-Brinkman evaluation artifacts from saved-membership batches.

    Complete loader batches are preprocessed and inferred exactly once before
    their case views are evaluated for channel-balanced relative errors, normalized
    sufficient statistics, and dimensionless momentum, flux-balance, and
    pressure-boundary diagnostics. One NPZ is written per case and one Parquet row per membership.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated steady-flow task owning exact channel names, units, and physics.
    model : Any
        Reconstructed best-checkpoint neural operator.
    loader : Iterable[Mapping[str, Any]]
        Deterministic saved-split batches with explicit source/local identity.
    processor : Any
        Restored training input/output normalizers.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Empty exact target or staging directory.
    dataset_name : str
        Logical dataset name used for the Parquet filename.
    provenance : Mapping[str, Any]
        Versioned request evidence, including the physics contract and exact
        complete membership digest.
    case_ids : Sequence[str]
        Authoritative Dataset case IDs aligned with complete saved membership; they name the NPZ files.
    publication_root : str | pathlib.Path | None, optional
        Final target recorded in Parquet NPZ paths while writing a sibling stage.
    timing_cases : list[dict[str, Any]] | None, optional
        Empty collector populated with amortized shares of complete-batch forwards.
    progress_reporter : ArtifactProgressReporter | None, optional
        Shared progress reporter that receives one completion per written case.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Per-case table and atomically written Parquet payload path.

    Raises
    ------
    KeyError, TypeError, ValueError, RuntimeError, FloatingPointError
        If task/provenance semantics, batch identity, shapes, finite values,
        physics fields, metrics, or complete membership violate the contract.
    FileExistsError
        If any complete or interrupted target payload already exists.

    Notes
    -----
    Full-grid residual arrays coexist with scalar gradient diagnostics cropped by
    ``EVAL_PAD``. Provenance, including exact output digests and the aggregate
    normalized physical-group macro RMSE, publishes only after every payload succeeds.

    """
    save_root = Path(save_root)
    published_root = save_root if publication_root is None else Path(publication_root)
    expected_case_count = _artifact_complete_case_count(provenance)
    output_groups, train_standard_deviations, normalization_denominator_floor = _objective_context(
        task=task,
        processor=processor,
        provenance=provenance,
    )
    velocity_group = _required_output_group(task, group_id="velocity")
    velocity_indices = tuple(task.output_names.index(field) for field in velocity_group.fields)
    velocity_units = {task.field(field).unit for field in velocity_group.fields}
    if len(velocity_units) != 1:
        msg = f"Task velocity-group fields must share one physical unit, got {sorted(velocity_units)}."
        raise ValueError(msg)
    velocity_unit = next(iter(velocity_units))
    authoritative_case_ids, case_source_dataset_ids = _artifact_case_ids(
        case_ids=case_ids,
        provenance=provenance,
        timing_cases=timing_cases,
        expected_case_count=expected_case_count,
    )
    case_files = contracts.dataset_case_payload_filenames(authoritative_case_ids)
    _ensure_artifact_targets_absent(save_root, dataset_name)
    model.eval()

    # Infer run_dir from save_root for logging/metadata only.
    run_dir = infer_current_run_dir(save_root)
    run_name = run_dir.name
    if provenance.get("physics") != contracts.steady_physics_provenance(task):
        msg = "Steady-flow artifact provenance must carry the current steady-Brinkman physics contract."
        raise ValueError(msg)
    physics_variant = f"{task.physics.continuity}-{contracts.ARTIFACT_DERIVATIVE_KIND}"

    print(f"artifacts {run_name} | physics {task.physics.kind} {physics_variant} pad {contracts.EVAL_PAD} | into {save_root.name}/")

    # --------------------------------------------------
    # Build domain-owned residual diagnostics
    # --------------------------------------------------
    physics_evaluator = domain.physics.brinkman.resolve_physics_evaluator(task.physics.kind)
    derivative_operator = domain.physics.derivatives.build_derivative_operator(contracts.ARTIFACT_DERIVATIVE_KIND)

    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    generated_source_indices: list[int] = []

    # Detect available kappa channels from field contracts
    kappa_names = detect_kappa_channels_from_inputs(list(task.input_names))

    for idx, batch in _iter_inferred_artifact_cases(
        loader,
        model=model,
        processor=processor,
        device=device,
    ):
        split_local_index = _require_batch_scalar_int(batch, "split_local_index")
        source_index = _require_batch_scalar_int(batch, "source_index")
        if split_local_index != idx:
            msg = f"Artifact loader order does not match saved split-local identity: iteration={idx}, split_local_index={split_local_index}."
            raise RuntimeError(msg)
        if source_index < 0:
            msg = f"Artifact source_index must be non-negative, got {source_index}."
            raise ValueError(msg)
        case_id = source_index + 1
        generated_source_indices.append(source_index)

        x = batch["x"]
        y = batch["y"]
        y_norm = batch["y_normalized"]
        y_hat_norm = batch["prediction_normalized"]
        y_hat = batch["prediction"]
        inference_time_ms = None

        # Preserve generator metadata in a JSON-safe form.
        source_meta = meta_to_jsonable(batch.get("meta", {}))
        meta_clean = dict(source_meta) if isinstance(source_meta, dict) else {"source_meta": source_meta}
        reserved_meta_keys = {"case_index", "source_index", "split_local_index"}.intersection(meta_clean)
        if reserved_meta_keys:
            msg = f"Source metadata contains reserved artifact identity keys and cannot be preserved unambiguously: {sorted(reserved_meta_keys)}."
            raise KeyError(msg)
        # Pressure boundary condition (stored for diagnostics)
        p_bc_idx = task.input_names.index("p_in_bc")
        p_in_bc = x[:, p_bc_idx : p_bc_idx + 1].detach().cpu()

        # Permeability fields (no scalar stats here)
        kappa_info = extract_kappa(
            x,
            input_fields=list(task.input_names),
            kappa_names=kappa_names,
        )

        # --------------------------------------------------
        # Physics diagnostics (exact training-consistent implementation)
        # --------------------------------------------------
        with torch.no_grad():
            diag = physics_evaluator(
                x,
                y_hat,
                input_fields=task.input_names,
                output_fields=task.output_names,
                derivatives=derivative_operator,
                boundary=task.physics.boundary,
                interior_crop=contracts.EVAL_PAD,
            ).as_dict()
        # Residuals of a non-finite prediction are stored unchanged; the validity document classifies the case.
        for diagnostic_name, diagnostic_value in diag.items() if bool(torch.isfinite(y_hat).all()) else ():
            _require_finite_artifact_tensor(
                diagnostic_value,
                label=f"Artifact case {case_id} physics diagnostic {diagnostic_name}",
            )

        momentum_residual_mse = float(diag["momentum_residual_mse"].detach().cpu().item())
        flux_balance_mse = float(diag["flux_balance_mse"].detach().cpu().item())
        pressure_boundary_mse = float(diag["pressure_boundary_mse"].detach().cpu().item())
        pressure_inlet_mse = float(diag["pressure_inlet_mse"].detach().cpu().item())
        pressure_outlet_mean_square = float(diag["pressure_outlet_mean_square"].detach().cpu().item())
        Rx_np = diag["Rx"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Ry_np = diag["Ry"].detach().cpu().squeeze(0).squeeze(0).numpy()
        flux_balance_np = diag["flux_balance"].detach().cpu().squeeze(0).numpy()

        # --------------------------------------------------
        # Outputs (de-normalised, physical units)
        # --------------------------------------------------
        y_hat_main = y_hat
        y_main = y
        err_main = y_hat_main - y_main
        velocity_index_tensor = torch.tensor(velocity_indices, dtype=torch.long, device=y_hat.device)
        velocity_prediction = y_hat_main.index_select(1, velocity_index_tensor)
        velocity_target = y_main.index_select(1, velocity_index_tensor)
        U = torch.linalg.vector_norm(velocity_prediction, dim=1, keepdim=True)
        U_gt = torch.linalg.vector_norm(velocity_target, dim=1, keepdim=True)

        # Full tensors for NPZ export and plotting include derived speed magnitude.
        y_hat_ext = torch.cat([y_hat_main, U], dim=1)
        y_ext = torch.cat([y_main, U_gt], dim=1)
        err_ext = y_hat_ext - y_ext

        # Physical diagnostics and global-reduction evidence
        physical_statistics = physical_case_statistics(
            y_hat_main,
            y_main,
            output_fields=task.output_names,
        )[0]
        group_metrics = _group_case_metrics(
            task=task,
            physical_statistics=physical_statistics,
            output_groups=output_groups,
            train_standard_deviations=train_standard_deviations,
        )
        physical_rmse_speed_magnitude = torch.sqrt(torch.mean((U - U_gt) ** 2)).item()

        # Grid spacing from coordinate fields (physical)
        idx_x = task.input_names.index("x")
        idx_y = task.input_names.index("y")
        dx = float((x[0, idx_x, 0, 1] - x[0, idx_x, 0, 0]).abs().detach().cpu().item())
        dy = float((x[0, idx_y, 1, 0] - x[0, idx_y, 0, 0]).abs().detach().cpu().item())

        metric_denominator_floor = 1e-12
        relative_h1 = losses.factory.build_relative_h1(2, reduction="mean", grid_spacing=(dy, dx))

        # ------------------------------------------------------------------
        # Dimensionless relative L2/H1, normalized independently per field
        # ------------------------------------------------------------------
        rel_l2_per_channel: list[float] = []
        rel_h1_per_channel: list[float] = []

        for c in range(y_main.shape[1]):
            e_c = err_main[:, c : c + 1]
            r_c = y_main[:, c : c + 1]

            # Relative L2 per channel (global norm ratio)
            l2_e_c = torch.linalg.norm(e_c)
            l2_r_c = torch.linalg.norm(r_c)
            rel_l2_c = (l2_e_c / (l2_r_c + metric_denominator_floor)).item()
            rel_l2_per_channel.append(float(rel_l2_c))

            # Relative H1 per channel: the training/validation implementation on the EVAL_PAD interior
            pad = contracts.EVAL_PAD
            p_i, r_i = (t[..., pad:-pad, pad:-pad] if pad > 0 else t for t in (y_hat_main[:, c : c + 1], r_c))
            rel_h1_c = relative_h1(p_i, r_i).item()
            rel_h1_per_channel.append(float(rel_h1_c))

        rel_l2 = float(np.mean(rel_l2_per_channel))
        rel_h1 = float(np.mean(rel_h1_per_channel))
        normalized_statistics = normalized_case_statistics(
            y_hat_norm,
            y_norm,
            output_fields=task.output_names,
        )[0]
        metric_values = _admitted_case_metrics(
            y_hat_main,
            {
                "rel_l2": rel_l2,
                "rel_h1": rel_h1,
                "physical_rmse_speed_magnitude": physical_rmse_speed_magnitude,
                "momentum_residual_mse": momentum_residual_mse,
                "flux_balance_mse": flux_balance_mse,
                "pressure_boundary_mse": pressure_boundary_mse,
                "pressure_inlet_mse": pressure_inlet_mse,
                "pressure_outlet_mean_square": pressure_outlet_mean_square,
                **physical_statistics,
                **normalized_statistics,
                **group_metrics,
            },
            task=task,
        )

        # --------------------------------------------------
        # Write NPZ artifact
        # --------------------------------------------------
        npz_path = npz_dir / case_files[idx]
        if npz_path.exists():
            msg = f"Refusing to overwrite an existing NPZ artifact: {npz_path}"
            raise FileExistsError(msg)
        x_raw = x.squeeze(0).detach().cpu().numpy()  # (C_in,H,W)
        y_raw = y.squeeze(0).detach().cpu().numpy()  # (C_out,H,W)

        artifact_fields = (*task.output_names, "U")
        artifact_units = (*(field.unit for field in task.outputs), velocity_unit)
        npz_payload = {
            "case_index": np.int64(case_id),
            "source_index": np.int64(source_index),
            "split_local_index": np.int64(split_local_index),
            "pred": y_hat_ext.squeeze(0).cpu().numpy(),
            "gt": y_ext.squeeze(0).cpu().numpy(),
            "err": err_ext.squeeze(0).cpu().numpy(),
            "artifact_fields": np.asarray(artifact_fields),
            "artifact_units": np.asarray(artifact_units),
            "kappa_encoded": kappa_info["kappa_encoded"].squeeze(0).cpu().numpy(),
            "kappa": kappa_info["kappa"].squeeze(0).cpu().numpy(),
            "kappa_names": np.asarray(kappa_names),
            "p_in_bc": p_in_bc.squeeze(0).numpy(),
            "coordinates": x_raw[[task.input_names.index("x"), task.input_names.index("y")]],
            "meta": json.dumps(meta_clean),
            "x_raw": x_raw,
            "y_raw": y_raw,
            "input_fields": np.asarray(task.input_names),
            "output_fields": np.asarray(task.output_names),
            "output_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "Rx": Rx_np,
            "Ry": Ry_np,
            "flux_balance": flux_balance_np,
        }

        def write_npz(temp_path: Path, content: dict[str, Any] = npz_payload) -> None:
            """Serialize one steady-flow case into the atomic writer's temporary path."""
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, **content)

        common.serialization.atomic_path_write(npz_path, write_npz)

        # --------------------------------------------------
        # Parquet row (scalar metrics + metadata only)
        # --------------------------------------------------
        rows.append(
            {
                "artifact_schema_version": contracts.ARTIFACT_SCHEMA_VERSION,
                "task_id": task.id,
                "output_fields": list(task.output_names),
                "output_units": [field.unit for field in task.outputs],
                "inference_time_ms": inference_time_ms,
                "case_index": case_id,
                "source_index": source_index,
                "source_dataset_id": case_source_dataset_ids[idx],
                "split_local_index": split_local_index,
                "npz_path": (Path("npz") / npz_path.name).as_posix(),
                "kappa_names": kappa_names,
                **metric_values,
                "meta": json.dumps(meta_clean),
            }
        )
        _record_case(
            batch,
            case_id=authoritative_case_ids[idx],
            source_index=source_index,
            timing_cases=timing_cases,
            progress_reporter=progress_reporter,
        )

    df = pd.DataFrame(rows)
    if not df.columns.is_unique:
        msg = "Steady-flow artifact table contains duplicate columns."
        raise RuntimeError(msg)
    _validate_generated_source_indices(provenance, generated_source_indices)
    if len(df) != expected_case_count:
        msg = f"Artifact generation produced {len(df)} cases, expected {expected_case_count} from provenance."
        raise RuntimeError(msg)

    parquet_path = save_root / f"{dataset_name}.parquet"
    common.serialization.atomic_path_write(
        parquet_path,
        lambda temp_path: df.to_parquet(temp_path, index=False),
    )
    complete_provenance = _provenance_with_aggregate(
        provenance,
        frame=df,
        output_groups=output_groups,
        train_standard_deviations=train_standard_deviations,
        normalization_denominator_floor=normalization_denominator_floor,
    )
    _provenance_path, published_provenance = write_artifact_provenance(save_root, complete_provenance)
    df.attrs["artifact_provenance"] = published_provenance
    df.attrs["artifact_root"] = str(published_root)

    return df, parquet_path


def generate_artifacts(
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    case_ids: Sequence[str],
    publication_root: str | Path | None = None,
    timing_cases: list[dict[str, Any]] | None = None,
    progress_reporter: ArtifactProgressReporter | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate artifacts through the TaskSpec-driven storage contract.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated task owning ordered input/output fields, units, and physics.
    model : Any
        Reconstructed best-checkpoint model.
    loader : Iterable[Mapping[str, Any]]
        Deterministic saved-split batches with any positive batch size.
    processor : Any
        Restored training normalizer processor.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Exact empty artifact target or staging directory.
    dataset_name : str
        Logical dataset name used for the Parquet filename.
    provenance : Mapping[str, Any]
        Exact cache request identity. Generated aggregate and output digests are
        appended only after all case payloads succeed.
    case_ids : Sequence[str]
        Authoritative Dataset case IDs aligned with complete saved membership; they name the NPZ files.
    publication_root : str | pathlib.Path | None, optional
        Final target recorded in Parquet NPZ paths while payloads are staged.
    timing_cases : list[dict[str, Any]] | None, optional
        Empty collector populated with amortized shares of complete-batch forwards.
    progress_reporter : ArtifactProgressReporter | None, optional
        Shared progress reporter that receives one completion per written case.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Generated table and atomically written Parquet payload path.

    Raises
    ------
    KeyError, FileExistsError, TypeError, ValueError, RuntimeError, FloatingPointError
        If target occupancy, task/request identity, batch membership, tensor
        values, scientific diagnostics, or publication evidence is invalid.

    Notes
    -----
    Only the maintained steady-flow task is supported (Brinkman diagnostic adapter);
    any other task id raises ``ValueError``. This function
    creates payloads only: cache locking and target replacement belong to the
    artifact service. Provenance is always the final completion marker.

    """
    dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    if task.id == domain.tasks.steady_flow.STEADY_FLOW.id:
        return _generate_steady_flow_artifacts(
            task=task,
            model=model,
            loader=loader,
            processor=processor,
            device=device,
            save_root=save_root,
            dataset_name=dataset_name,
            provenance=provenance,
            case_ids=case_ids,
            publication_root=publication_root,
            timing_cases=timing_cases,
            progress_reporter=progress_reporter,
        )
    msg = f"Artifact generation supports only task {domain.tasks.steady_flow.STEADY_FLOW.id!r}, got {task.id!r}."
    raise ValueError(msg)
