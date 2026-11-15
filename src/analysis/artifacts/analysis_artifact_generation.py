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
  - Reuse, rebuild, or lock an existing artifact cache target
  - Render interactive scientific figures

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
      - source_dataset_id : source Dataset of the case (the material-OOD role may span several Datasets)
      - split_local_index : zero-based position in the saved split
      - npz_path          : path to the corresponding NPZ artifact
      Case statistics (float64 sums per TaskSpec output, learning.metrics.metrics.CASE_STATISTICS):
        - physical_<statistic>_<field> : count, squared_error, absolute_error and reference_square on the
          full grid; h1_count, h1_error and h1_reference on the H1 interior (derivatives in metres, l = 1 m)
      Per-case metrics derived from them (contracts.case_metric_values):
        - physical_rmse_<field>, normalized_rmse_<field> : case RMSE, physical and over the Train scale
        - physical_relative_l2, physical_relative_h1 : group-weighted per-field relative norms; NaN when a
          field's reference RMS is below its TaskSpec floor
        - physical_velocity_vector_rmse : vector RMSE over velocity components
        - physical_rmse_speed_magnitude : RMSE of velocity-group speed magnitude |u|
      Dimensionless physics residual metrics (scales from inputs only):
        - momentum_residual_mse : mean(Rx**2 + Ry**2) over the EVAL_PAD-cropped interior
        - flux_balance_mse      : mean(flux_balance**2) over every grid row
      Dimensionless boundary metrics (full-grid inlet/outlet masks):
        - pressure_inlet_mse            : mean_inlet((p-p_in_bc)**2) / dp**2
        - pressure_outlet_mean_square   : (mean_outlet(p) / dp)**2
        - pressure_boundary_mse         : sum of the preceding two terms
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
      digests, saved train-fitted output scales, task-owned output groups and reference
      floors, the aggregate (normalized_group_macro_rmse with the per-field RMSE, MAE and
      relative norms) finalized once from the case statistics (counts instead of values
      when any prediction is non-finite), and the role-level prediction validity under evaluation.

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

from src import common, domain, learning
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract

from . import contracts, timing

if TYPE_CHECKING:
    from src.datasets.preprocessing.dataset_preprocessing_normalization import TrainScaling

    from .analysis_artifact_performance import ArtifactProgressReporter

# ============================================================================
# Global constants
# ============================================================================

INTERNAL_KAPPA_NAMES = set(domain.permeability.INTERNAL_KAPPA_2D_ORDER) | set(domain.permeability.INTERNAL_KAPPA_3D_ORDER)
_MIN_BATCH_TENSOR_RANK = 2


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
    scaling: TrainScaling,
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return task groups and the saved Train state scales after checking them and the metric evidence against the task and scaling."""
    groups = contracts.output_group_payload(task.output_groups)
    flattened_fields = tuple(field for group in groups for field in group["fields"])
    if flattened_fields != task.output_names:
        msg = "Artifact output groups do not partition TaskSpec outputs in declared order."
        raise ValueError(msg)
    normalizer = provenance.get("normalizer")
    if not isinstance(normalizer, Mapping):
        msg = "Artifact provenance must contain normalizer evidence."
        raise TypeError(msg)
    scales = contracts.validated_train_scales(normalizer.get("output_scales"), fields=task.output_names)
    if contracts.output_scales(scaling, output_fields=task.output_names) != scales:
        msg = "Artifact scaling output scales contradict saved normalizer provenance."
        raise ValueError(msg)
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("output_groups") != groups:
        msg = "Artifact evaluator output groups contradict TaskSpec."
        raise ValueError(msg)
    if evaluator.get("metric_evidence") != contracts.metric_evidence_provenance(task):
        msg = "Artifact evaluator metric evidence contradicts TaskSpec."
        raise ValueError(msg)
    return groups, scales


def _group_case_metrics(
    *,
    task: domain.tasks.spec.TaskSpec,
    case_values: Mapping[str, float],
    output_groups: Iterable[Mapping[str, Any]],
    train_scales: Mapping[str, float],
) -> dict[str, float]:
    """Return task-declared physical and normalized group diagnostics for one case."""
    values: dict[str, float] = {}
    metrics_by_signature = {
        (metric.kind, metric.fields): metric.id for metric in task.default_metrics if metric.kind in {"group_rmse", "vector_rmse"}
    }
    for group in output_groups:
        fields = tuple(group["fields"])
        physical_mse_sum = sum(case_values[f"physical_rmse_{field}"] ** 2 for field in fields)
        normalized_id = metrics_by_signature.get(("group_rmse", fields))
        if normalized_id is not None:
            scale_square_sum = sum(train_scales[field] ** 2 for field in fields)
            values[normalized_id] = math.sqrt(physical_mse_sum / scale_square_sum)
        physical_id = metrics_by_signature.get(("vector_rmse", fields))
        if physical_id is not None:
            values[physical_id] = math.sqrt(physical_mse_sum)
    return values


def _provenance_with_aggregate(
    provenance: Mapping[str, Any],
    *,
    frame: pd.DataFrame,
    output_groups: Iterable[Mapping[str, Any]],
    train_scales: Mapping[str, float],
) -> dict[str, Any]:
    """
    Add the authoritative group aggregate and role prediction validity to generation provenance.

    Caller-supplied aggregate or evaluation results are rejected so only recomputed
    physical Parquet SSE/count evidence, the saved Train state scales and the per-row
    validity documents can determine the published selection objective.
    """
    payload = dict(provenance)
    if "aggregate" in payload or "evaluation" in payload:
        msg = "Caller-provided artifact provenance cannot define aggregate or evaluation results."
        raise ValueError(msg)
    payload["aggregate"] = contracts.aggregate_normalized_group_macro_rmse(
        frame,
        output_groups=output_groups,
        train_scales=train_scales,
        reference_floors=payload["evaluator"]["metric_evidence"]["reference_floors"],
    )
    payload["evaluation"] = {"prediction_validity": validity_contract.table_role_prediction_validity(frame)}
    return payload


def _admitted_case_metrics(prediction: torch.Tensor, metrics: Mapping[str, float | int], *, task: domain.tasks.spec.TaskSpec) -> dict[str, Any]:
    """
    Classify one physical prediction and admit its scalar metrics with the validity document.

    A valid case must have finite metrics, except a relative norm whose reference is below
    its floor. A non-finite case keeps its element counts and reports NaN for every other
    metric, so no missing value can read as a valid zero.
    """
    document = validity_contract.build_field_prediction_validity(prediction.squeeze(0).detach().cpu().numpy(), channels=task.output_names)
    counts = contracts.element_count_columns(task.output_names)
    if document["status"] == validity_contract.NONFINITE:
        metrics = {name: value if name in counts else math.nan for name, value in metrics.items()}
    elif not np.isfinite([value for name, value in metrics.items() if name not in contracts.RELATIVE_CASE_COLUMNS]).all():
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
    scaling: TrainScaling,
    device: torch.device,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """
    Infer each loader batch once and yield deterministic size-one case views.

    The restored scaling follows the online evaluation lifecycle exactly: it
    encodes the inputs of the complete loader batch on the device, the model
    predicts encoded states, and the scaling decodes them to physical units.
    Predictions and prepared tensors are then sliced for one-row/one-NPZ storage
    without repeating model forwards. Batch forward duration is divided equally
    among its cases as an amortized throughput measurement.

    Parameters
    ----------
    loader : Iterable[Mapping[str, Any]]
        Ordered batches containing compatible tensor ``x`` and ``y`` entries.
    model : Any
        Evaluation-mode model receiving each complete encoded loader batch.
    scaling : TrainScaling
        The run's restored Train scaling on ``device``.
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
            inputs = raw_inputs.to(device)
            targets = raw_targets.to(device)
            normalized_inputs = scaling.encode_inputs(inputs)
            normalized_prediction, batch_forward_s = timing.measure_forward(
                model=model,
                normalized_inputs=normalized_inputs,
                device=device,
            )
            if not isinstance(normalized_prediction, torch.Tensor):
                msg = "Artifact model output must be a tensor."
                raise TypeError(msg)
            prediction = scaling.decode_state(normalized_prediction)

        # Predictions stay unchecked here: case admission classifies them without changing values.
        _require_finite_artifact_tensor(normalized_inputs, label=f"Artifact batch {batch_index} normalized inputs")
        if prediction.shape != targets.shape:
            msg = f"Artifact prediction/target shapes disagree after complete-batch inference: {tuple(prediction.shape)}/{tuple(targets.shape)}."
            raise RuntimeError(msg)
        if prediction.shape[0] != batch_size:
            msg = "Artifact model output must preserve the complete loader batch axis."
            raise RuntimeError(msg)

        amortized_forward_s = batch_forward_s / batch_size
        for case_offset in range(batch_size):
            case = {
                "x": inputs[case_offset : case_offset + 1],
                "y": targets[case_offset : case_offset + 1],
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


def _ensure_artifact_targets_absent(save_root: Path) -> None:
    """
    Enforce create-only semantics for one staged artifact target.

    Existing Parquet, provenance, NPZ, or interrupted temporary files are all
    reported together and rejected. Generation never overwrites partial output.
    """
    npz_dir = save_root / "npz"
    candidates = [
        save_root / common.paths.ANALYSIS_ARTIFACTS_FILENAME,
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
    scaling, and device ownership remain explicit caller inputs.

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
    scaling: TrainScaling,
    device: torch.device,
    save_root: str | Path,
    provenance: Mapping[str, Any],
    case_ids: Sequence[str],
    publication_root: str | Path | None = None,
    timing_cases: list[dict[str, Any]] | None = None,
    progress_reporter: ArtifactProgressReporter | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate steady-Brinkman evaluation artifacts from saved-membership batches.

    Complete loader batches are preprocessed and inferred exactly once before
    their case views are evaluated for the physical case statistics, the per-case
    RMSE and relative norms, and dimensionless momentum, flux-balance, and
    pressure-boundary diagnostics. One NPZ is written per case and one Parquet row per membership.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated steady-flow task owning exact channel names, units, and physics.
    model : Any
        Reconstructed best-checkpoint neural operator.
    loader : Iterable[Mapping[str, Any]]
        Deterministic saved-split batches with explicit source/local identity.
    scaling : TrainScaling
        The run's restored Train scaling on ``device``.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Empty exact target or staging directory.
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
    output_groups, train_scales = _objective_context(
        task=task,
        scaling=scaling,
        provenance=provenance,
    )
    reference_floors = {field: task.reference_floor(field) for field in task.output_names}
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
    _ensure_artifact_targets_absent(save_root)
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
        scaling=scaling,
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
        velocity_index_tensor = torch.tensor(velocity_indices, dtype=torch.long, device=y_hat.device)
        velocity_prediction = y_hat_main.index_select(1, velocity_index_tensor)
        velocity_target = y_main.index_select(1, velocity_index_tensor)
        U = torch.linalg.vector_norm(velocity_prediction, dim=1, keepdim=True)
        U_gt = torch.linalg.vector_norm(velocity_target, dim=1, keepdim=True)

        # Full tensors for NPZ export and plotting include derived speed magnitude.
        y_hat_ext = torch.cat([y_hat_main, U], dim=1)
        y_ext = torch.cat([y_main, U_gt], dim=1)
        err_ext = y_hat_ext - y_ext

        # Physical case statistics on the full grid and the H1 interior, with derivatives in metres
        idx_x = task.input_names.index("x")
        idx_y = task.input_names.index("y")
        dx = float((x[0, idx_x, 0, 1] - x[0, idx_x, 0, 0]).abs().detach().cpu().item())
        dy = float((x[0, idx_y, 1, 0] - x[0, idx_y, 0, 0]).abs().detach().cpu().item())
        sums = learning.metrics.metrics.field_statistics(y_hat_main, y_main, grid_spacing=(dy, dx))[0]
        case_values = contracts.case_metric_values(
            sums,
            fields=task.output_names,
            output_groups=task.output_groups,
            train_scales=train_scales,
            reference_floors=reference_floors,
        )
        metric_values = _admitted_case_metrics(
            y_hat_main,
            {
                "physical_rmse_speed_magnitude": torch.sqrt(torch.mean((U - U_gt) ** 2)).item(),
                "momentum_residual_mse": momentum_residual_mse,
                "flux_balance_mse": flux_balance_mse,
                "pressure_boundary_mse": pressure_boundary_mse,
                "pressure_inlet_mse": pressure_inlet_mse,
                "pressure_outlet_mean_square": pressure_outlet_mean_square,
                **{
                    column: float(value)
                    for field, row in zip(task.output_names, sums, strict=True)
                    for column, value in zip(contracts.case_statistic_columns(field), row, strict=True)
                },
                **case_values,
                **_group_case_metrics(task=task, case_values=case_values, output_groups=output_groups, train_scales=train_scales),
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

    parquet_path = save_root / common.paths.ANALYSIS_ARTIFACTS_FILENAME
    common.serialization.atomic_path_write(
        parquet_path,
        lambda temp_path: df.to_parquet(temp_path, index=False),
    )
    complete_provenance = _provenance_with_aggregate(
        provenance,
        frame=df,
        output_groups=output_groups,
        train_scales=train_scales,
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
    scaling: TrainScaling,
    device: torch.device,
    save_root: str | Path,
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
    scaling : TrainScaling
        The run's restored Train scaling on ``device``.
    device : torch.device
        Concrete inference device.
    save_root : str | pathlib.Path
        Exact empty artifact target or staging directory.
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
    if task.id == domain.tasks.steady_flow.STEADY_FLOW.id:
        return _generate_steady_flow_artifacts(
            task=task,
            model=model,
            loader=loader,
            scaling=scaling,
            device=device,
            save_root=save_root,
            provenance=provenance,
            case_ids=case_ids,
            publication_root=publication_root,
            timing_cases=timing_cases,
            progress_reporter=progress_reporter,
        )
    msg = f"Artifact generation supports only task {domain.tasks.steady_flow.STEADY_FLOW.id!r}, got {task.id!r}."
    raise ValueError(msg)
