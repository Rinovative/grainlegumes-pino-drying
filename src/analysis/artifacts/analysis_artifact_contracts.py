"""Define lightweight persisted contracts for generated analysis artifacts.

Responsibilities:
  - Declare artifact, provenance, residual, and derivative schema constants
  - Own the steady-Brinkman residual diagnostic provenance contract
  - Name physical and normalized sufficient-statistic columns
  - Finalize the equal physical-output-group selection objective
  - Build ordered membership digests, completion paths, and payload manifests
  - Hash each file identity at most once per process
  - Recognize Optuna trial leaves from the validated layout recorded in a saved run config and refuse them with one wording
  - Build and validate the runtime provenance key set both tasks persist
  - Own the per-case NPZ file name, which is the Dataset package case ID
  - Declare the per-task analysis projection and the producer evidence every artifact of both tasks records
  - Own the freshness guard for requested artifacts and stages and the producer check of every comparison

Design principles:
  - Persisted names and schema values have one authoritative owner
  - Objective aggregation delegates to the online group-metric finalizer
  - Physical SSE, counts, and train-fitted scales remain explicit evidence
  - Reading an artifact validates its recorded analysis digest as evidence; only reuse or continuation of a
    requested artifact compares it with the running projection

This module does NOT:
  - Import Torch directly or run trained-model inference
  - Write artifact payloads or publish completion markers
  - Admit, rebuild, lock, time, render, or upload artifact caches
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src import common, domain, experiments, learning
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.domain.tasks.domain_task_spec import OutputGroupSpec
from src.learning.metrics.learning_metrics import finalize_group_rmse_statistics

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from src.domain.tasks.domain_task_spec import TaskSpec


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 1
RESIDUAL_SCHEMA_VERSION = 1
ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"
STEADY_ARTIFACT_KIND = "steady_field"
INFERENCE_PRECISION = "float32"
_RUNTIME_REQUIRED_KEYS = frozenset(
    {"requested_policy", "resolved_device", "device_type", "pytorch_version", "precision", "inference_batch_size", "source_tree_id"}
)
_RUNTIME_OPTIONAL_KEYS = frozenset({"cuda_index", "cuda_device_name", "cuda_runtime_version"})
NORMALIZED_OBJECTIVE_TOLERANCE = {"rtol": 1e-12, "atol": 1e-12}
NORMALIZED_DIAGNOSTIC_TOLERANCE = {"rtol": 2e-5, "atol": 1e-12}
EVAL_PAD = 2
# Artifact residuals use one fixed, most accurate reference operator for every
# run, independent of the derivative a run was trained with.
ARTIFACT_DERIVATIVE_KIND = "physical"
_ARTIFACT_IDENTITY_CONTRACT = {
    "schema_kind": "evaluation_artifact_semantic_identity",
    "schema_version": 1,
    "checkpoint_semantics": "exact_best_checkpoint_bytes",
    "dataset_semantics": "exact_dataset_and_ordered_split_membership",
    "evaluation_semantics": "resolved_metric_evaluator_and_physics_contracts",
    "output_integrity": "separate_manifest_of_exact_payload_bytes",
}
ARTIFACT_IDENTITY_CONTRACT_DIGEST = hashlib.sha256(
    json.dumps(
        _ARTIFACT_IDENTITY_CONTRACT,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
_PACKAGE_CASE_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*__case_[0-9]{4,}__[0-9a-f]{12}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REBUILD_REMEDY: Final = "rerun only this exact target with --rebuild"
_ANALYSIS_FILES: Final = (
    "analysis/artifacts/analysis_artifact_contracts.py",
    "analysis/artifacts/analysis_artifact_plan.py",
    "analysis/artifacts/analysis_artifact_service.py",
    "analysis/artifacts/analysis_artifact_timing.py",
    "analysis/evaluation/evaluation_prediction_validity.py",
)
_ANALYSIS_EXEMPTIONS: Final = {
    "analysis/artifacts/analysis_artifact_performance.py": "operational progress and performance reporting, outside scientific identity",
    "analysis/evaluation/evaluation_artifact_loader.py": "admission reads persisted artifacts and produces none",
    "analysis/presentation/analysis_presentation_curated.py": "curated figures rendered for tracking upload, never persisted evidence",
    "experiments/config/experiments_config_loader.py": "admission of the saved resolved config, whose recorded identities bind the run",
    "experiments/experiments_run.py": "admission of the completed run and its checkpoints, bound by their recorded identities",
    "learning/losses/learning_losses_factory.py": "training H1 loss constructor of the metric module, never reached by artifact metrics",
    "learning/training/learning_training_checkpoint.py": (
        "checkpoint schema and projection declarations; recorded identities bind the checkpoint and the model projection is included"
    ),
}
ANALYSIS_PROJECTIONS: Final[dict[str, common.code_identity.CodeProjection]] = {
    "steady_flow": common.code_identity.CodeProjection(
        name="analysis_steady_flow",
        directories=("domain/physics", "learning/metrics"),
        files=(
            *_ANALYSIS_FILES,
            "analysis/artifacts/analysis_artifact_generation.py",
            "domain/domain_permeability.py",
            "learning/inference/learning_inference.py",
        ),
        excluded_markers=("transient",),
        exemptions=_ANALYSIS_EXEMPTIONS,
        includes=(learning.training.checkpoint.MODEL_PROJECTIONS["steady_flow"],),
    ),
    "transient_drying": common.code_identity.CodeProjection(
        name="analysis_transient_drying",
        directories=("learning/metrics",),
        files=(
            *_ANALYSIS_FILES,
            "analysis/artifacts/analysis_artifact_transient.py",
            "analysis/evaluation/evaluation_transient_artifact.py",
            "analysis/evaluation/evaluation_transient_metrics.py",
            "analysis/evaluation/evaluation_transient_rollout.py",
            "analysis/evaluation/evaluation_transient_timing.py",
            "analysis/evaluation/evaluation_transient_validity.py",
            "datasets/packages/dataset_packages_generated_batch.py",
            "domain/domain_moisture.py",
            "experiments/experiments_matched_budget.py",
            "learning/inference/learning_inference_transient.py",
        ),
        excluded_markers=("steady", "_pino"),
        exemptions={
            **_ANALYSIS_EXEMPTIONS,
            "analysis/artifacts/analysis_artifact_generation.py": "steady artifact writer, reached through the service's task dispatch",
            "analysis/evaluation/evaluation_transient_session.py": "summary of admitted artifacts for tracking, never persisted evidence",
            "domain/physics/domain_physics_brinkman.py": "steady Brinkman residual contract, reached only by steady artifacts",
            "domain/physics/domain_physics_contracts.py": "steady physics-kind vocabulary, reached only by the steady payload check",
            "learning/inference/learning_inference.py": "steady inference reconstruction, reached through the service's task dispatch",
            "learning/transient/learning_transient_handoff.py": (
                "training handoff publication; artifacts use only its directory name and bind the manifest bytes by digest"
            ),
        },
        includes=(learning.training.checkpoint.MODEL_PROJECTIONS["transient_drying"],),
    ),
}


def _identity_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one mapping used by evaluation-artifact semantic identity."""
    if not isinstance(value, Mapping):
        message = f"Artifact identity {label} must be a mapping."
        raise TypeError(message)
    return value


def _selected_identity_fields(
    value: Any,
    *,
    fields: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    """Select required semantic fields from one persisted provenance section."""
    mapping = _identity_mapping(value, label=label)
    missing = [field for field in fields if field not in mapping]
    if missing:
        message = f"Artifact identity {label} is missing semantic fields {missing}."
        raise ValueError(message)
    return {field: deepcopy(mapping[field]) for field in fields}


def artifact_identity_payload(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return positive model, Dataset, split, evaluator, and physics dependencies."""
    required_top_level = (
        "provenance_schema_version",
        "artifact_schema_version",
        "producer",
        "run",
        "model",
        "split_role",
        "dataset",
        "selection",
        "normalizer",
        "evaluator",
        "generation",
    )
    missing = [field for field in required_top_level if field not in provenance]
    if missing:
        message = f"Artifact identity provenance is missing semantic sections {missing}."
        raise ValueError(message)
    run = _selected_identity_fields(
        provenance["run"],
        fields=(
            "task",
            "task_contract_digest",
            "effective_config_digest",
            "best_checkpoint_sha256",
            "normalizer_sha256",
        ),
        label="run",
    )
    model_mapping = _identity_mapping(provenance["model"], label="model")
    model = _selected_identity_fields(
        model_mapping,
        fields=("kind", "architecture", "physics_enabled"),
        label="model",
    )
    if "parameter_counts" in model_mapping:
        model["parameter_counts"] = deepcopy(model_mapping["parameter_counts"])
    payload = {
        "identity_contract_sha256": ARTIFACT_IDENTITY_CONTRACT_DIGEST,
        "provenance_schema_version": deepcopy(provenance["provenance_schema_version"]),
        "artifact_schema_version": deepcopy(provenance["artifact_schema_version"]),
        "producer": validate_analysis_producer(provenance["producer"]),
        "run": run,
        "model": model,
        "split_role": deepcopy(provenance["split_role"]),
        "dataset": _selected_identity_fields(
            provenance["dataset"],
            fields=("name", "role", "source_dataset_ids", "source_identities", "case_ids_by_source", "membership_digests", "evaluation_assignments"),
            label="dataset",
        ),
        "selection": _selected_identity_fields(
            provenance["selection"],
            fields=(
                "full_selected_case_count",
                "effective_case_count",
                "generation_limit",
                "full_ordered_source_indices_sha256",
                "effective_ordered_source_indices_sha256",
            ),
            label="selection",
        ),
        "normalizer": _selected_identity_fields(
            provenance["normalizer"],
            fields=(
                "sha256",
                "fit_split",
                "output_normalization",
                "denominator_floor",
                "output_standard_deviations",
            ),
            label="normalizer",
        ),
        "evaluator": _selected_identity_fields(
            provenance["evaluator"],
            fields=(
                "metrics",
                "objective",
                "input_fields",
                "input_units",
                "output_fields",
                "output_units",
                "output_groups",
                "physics_kind",
                "group_objective_evidence",
                "predictive_metrics",
            ),
            label="evaluator",
        ),
        "generation": _selected_identity_fields(
            provenance["generation"],
            fields=("effective_case_limit", "inference_batch_size", "compression"),
            label="generation",
        ),
    }
    if "physics" in provenance:
        payload["physics"] = _selected_identity_fields(
            provenance["physics"],
            fields=(
                "residual_schema_version",
                "task_id",
                "task_contract_digest",
                "equation_kind",
                "equation_set",
                "continuity_kind",
                "boundary_condition_kind",
                "constants",
                "permeability_representation",
                "scales",
                "derivatives",
                "interior_crop",
                "residual_evaluation_region",
                "scalar_definitions",
                "array_definitions",
            ),
            label="physics",
        )
    return payload


def artifact_identity_digest(provenance: Mapping[str, Any]) -> str:
    """Hash the current dependency-scoped evaluation-artifact identity."""
    payload = artifact_identity_payload(provenance)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def resolve_case_payload_path(
    artifact_root: Path | str,
    stored_path: object,
    *,
    expected_filename: str,
) -> Path:
    """
    Resolve one exact bundle-relative NPZ reference.

    Rows must store exactly ``npz/<expected_filename>``. Absolute paths, parent
    traversal, and symbolic-link substitutions below the artifact root are
    rejected.
    """
    if not isinstance(stored_path, str) or not stored_path:
        msg = "Artifact Parquet npz_path must be a non-empty string."
        raise TypeError(msg)
    if not isinstance(expected_filename, str) or not expected_filename.endswith(".npz") or Path(expected_filename).name != expected_filename:
        msg = f"Expected NPZ filename is invalid: {expected_filename!r}."
        raise ValueError(msg)
    raw = Path(stored_path)
    expected_relative = Path("npz") / expected_filename
    if raw != expected_relative or raw.is_absolute() or ".." in raw.parts:
        msg = f"Artifact relative NPZ path must be exactly {expected_relative.as_posix()!r}, got {stored_path!r}."
        raise ValueError(msg)

    root = Path(artifact_root).expanduser().resolve()
    candidate = root / expected_relative
    if not candidate.is_relative_to(root) or candidate.parent.is_symlink() or candidate.is_symlink():
        msg = f"Artifact NPZ path uses an unsafe symbolic-link substitution: {candidate}"
        raise ValueError(msg)
    if not candidate.is_file() or candidate.resolve() != candidate:
        msg = f"Artifact NPZ payload is missing or escapes its current root: {candidate}"
        raise FileNotFoundError(msg)
    return candidate


def case_payload_filename(case_id: str) -> str:
    """
    Return ``<package case id>.npz`` for one exact Dataset package case ID.

    Raises
    ------
    ValueError
        If the ID is not ``<material>__case_<NNNN>__<12 hex>``.

    """
    if _PACKAGE_CASE_ID.fullmatch(case_id) is None:
        msg = f"Case payload names require a package case ID '<material>__case_NNNN__<12 hex>', got {case_id!r}."
        raise ValueError(msg)
    return f"{case_id}.npz"


def dataset_case_payload_filenames(sample_ids: Sequence[str]) -> tuple[str, ...]:
    """
    Return unique NPZ names for ordered steady Dataset sample IDs.

    A combined evaluation Dataset qualifies package case IDs as ``<dataset>::<case ID>``.
    The file keeps the package case ID alone, so a case repeated across components is rejected.
    """
    names = tuple(case_payload_filename(sample_id.rpartition("::")[2]) for sample_id in sample_ids)
    if len(names) != len(set(names)):
        msg = "Dataset sample IDs map to duplicate case payload names."
        raise ValueError(msg)
    return names


_FILE_DIGESTS: dict[str, tuple[tuple[int, int, int], str]] = {}


def cached_file_sha256(path: Path | str) -> str:
    """Return a file's SHA-256 digest, hashing each file identity at most once per process.

    The digest is reused while the path, size, modification time and inode are unchanged.
    A same-size in-place edit that restores the modification time would evade this; no writer does that.
    """
    file = Path(path)
    status = file.stat()
    identity = (status.st_size, status.st_mtime_ns, status.st_ino)
    cached = _FILE_DIGESTS.get(str(file))
    if cached is not None and cached[0] == identity:
        return cached[1]
    digest = common.serialization.file_sha256(file)
    _FILE_DIGESTS[str(file)] = (identity, digest)
    return digest


def is_optuna_trial_config(config: object) -> bool:
    """Return whether a saved run config records a valid Optuna trial layout (identity, not directory name).

    A layout that names a trial but is malformed raises ``ValueError`` instead of counting as a trial.
    """
    run = config.get("run") if isinstance(config, Mapping) else None
    layout = run.get("layout") if isinstance(run, Mapping) else None
    if not isinstance(layout, Mapping) or layout.get("trial") is None:
        return False
    common.paths.resolve_config_run_dir({"run": {"layout": layout}, "paths": {"output_root": "."}})
    return True


def runtime_provenance(device_metadata: Mapping[str, Any], *, inference_batch_size: int) -> dict[str, Any]:
    """Return the one runtime provenance key set of both tasks: resolved device, precision, inference batch size and executing source tree."""
    return {
        **dict(device_metadata),
        "precision": INFERENCE_PRECISION,
        "inference_batch_size": inference_batch_size,
        "source_tree_id": experiments.identity.executing_source_tree_id(),
    }


def validate_runtime_provenance(value: object) -> int:
    """Validate runtime provenance of either task without comparing device choices; return its inference batch size."""
    if not isinstance(value, Mapping):
        msg = "Artifact runtime provenance must be a mapping."
        raise TypeError(msg)
    missing = sorted(_RUNTIME_REQUIRED_KEYS.difference(value))
    unexpected = sorted(set(value).difference(_RUNTIME_REQUIRED_KEYS | _RUNTIME_OPTIONAL_KEYS))
    if missing or unexpected:
        msg = f"Artifact runtime provenance schema mismatch: missing={missing}, unexpected={unexpected}."
        raise ValueError(msg)
    texts = ("requested_policy", "resolved_device", "device_type", "pytorch_version", "cuda_device_name", "cuda_runtime_version")
    if any(key in value and (not isinstance(value[key], str) or not value[key]) for key in texts) or value["precision"] != INFERENCE_PRECISION:
        msg = f"Artifact runtime provenance needs non-empty device texts and precision {INFERENCE_PRECISION!r}."
        raise ValueError(msg)
    if value["source_tree_id"] is not None and not common.paths.is_source_tree_id(value["source_tree_id"]):
        msg = f"Artifact runtime source_tree_id must be null or <commit12>-<fingerprint12>, got {value['source_tree_id']!r}."
        raise ValueError(msg)
    for key, minimum in (("inference_batch_size", 1), ("cuda_index", 0)):
        number = value.get(key, minimum)
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            msg = f"Artifact runtime {key} must be an integer of at least {minimum}."
            raise TypeError(msg)
    return int(value["inference_batch_size"])


def analysis_producer(task_id: str) -> dict[str, str]:
    """Return the producer section a new artifact or stage of one task records: the running analysis projection digest."""
    return {"analysis_sha256": ANALYSIS_PROJECTIONS[task_id].sha256()}


def validate_analysis_producer(value: object) -> dict[str, str]:
    """Admit a recorded producer section as evidence: exactly one lowercase SHA-256, never compared with the running code."""
    digest = value.get("analysis_sha256") if isinstance(value, Mapping) and set(value) == {"analysis_sha256"} else None
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        msg = "Artifact producer must be exactly {'analysis_sha256': <lowercase SHA-256>}."
        raise ValueError(msg)
    return {"analysis_sha256": digest}


def require_current_analysis(provenance: object, *, task_id: str, unit: str) -> None:
    """Refuse to reuse or continue an artifact or stage whose recorded analysis implementation is not the running one."""
    producer = provenance.get("producer") if isinstance(provenance, Mapping) else None
    recorded = producer.get("analysis_sha256") if isinstance(producer, Mapping) else None
    common.code_identity.require_current_implementation(ANALYSIS_PROJECTIONS[task_id], recorded, unit=unit, remedy=_REBUILD_REMEDY)


def require_common_analysis_producer(provenances: Mapping[str, object]) -> None:
    """Refuse a comparison of artifacts whose recorded analysis implementations differ, or are missing or malformed."""
    digests: dict[str, str] = {}
    for label, provenance in provenances.items():
        try:
            digests[label] = validate_analysis_producer(provenance.get("producer") if isinstance(provenance, Mapping) else None)["analysis_sha256"]
        except ValueError as error:
            msg = f"Artifact {label!r} cannot be compared: {error}"
            raise ValueError(msg) from error
    if len(set(digests.values())) > 1:
        recorded = ", ".join(f"{label}={digest[:12]}" for label, digest in digests.items())
        msg = f"Artifacts were produced by different analysis implementations ({recorded}); rebuild them with one analysis code using --rebuild."
        raise ValueError(msg)


def refuse_optuna_trial(config: object, *, label: str) -> None:
    """Refuse an Optuna trial for every artifact path of either task: a trial keeps no checkpoints, so no artifact binds to it."""
    if is_optuna_trial_config(config):
        msg = (
            f"{label} is an Optuna trial, which keeps no checkpoints and has no artifacts. "
            "Select its parent study directory (its trials are skipped) or a training run instead."
        )
        raise ValueError(msg)


def normalized_statistic_columns(field_name: str) -> tuple[str, str, str]:
    """
    Return task-derived per-case normalized SSE, count, and RMSE columns.

    Parameters
    ----------
    field_name : str
        Exact non-empty TaskSpec output field name.

    Returns
    -------
    tuple[str, str, str]
        Ordered sufficient-statistic and convenience-RMSE column names.

    Raises
    ------
    ValueError
        If the field name is empty or not text.

    """
    if not isinstance(field_name, str) or not field_name:
        msg = "Artifact output field names must be non-empty strings."
        raise ValueError(msg)
    return (
        f"normalized_sse_{field_name}",
        f"normalized_count_{field_name}",
        f"normalized_rmse_{field_name}",
    )


def physical_statistic_columns(field_name: str) -> tuple[str, str, str]:
    """
    Return task-derived per-case physical SSE, count, and RMSE columns.

    Parameters
    ----------
    field_name : str
        Exact non-empty TaskSpec output field name.

    Returns
    -------
    tuple[str, str, str]
        Ordered physical sufficient-statistic and convenience-RMSE columns.

    Raises
    ------
    ValueError
        If the field name is empty or not text.

    """
    if not isinstance(field_name, str) or not field_name:
        msg = "Artifact output field names must be non-empty strings."
        raise ValueError(msg)
    return (
        f"physical_sse_{field_name}",
        f"physical_count_{field_name}",
        f"physical_rmse_{field_name}",
    )


def element_count_columns(field_names: Iterable[str]) -> frozenset[str]:
    """Return the physical and normalized per-case element-count columns of ``field_names``."""
    return frozenset(columns(field)[1] for field in field_names for columns in (physical_statistic_columns, normalized_statistic_columns))


def output_group_payload(output_groups: Iterable[Any]) -> list[dict[str, Any]]:
    """
    Normalize task-owned output groups to an ordered JSON-safe declaration.

    Parameters
    ----------
    output_groups : Iterable[Any]
        Task ``OutputGroupSpec`` objects or equivalent mappings containing
        ``id`` and ``fields``.

    Returns
    -------
    list[dict[str, Any]]
        Ordered group identifiers and their ordered member fields.

    Raises
    ------
    TypeError, ValueError
        If group declarations are malformed, duplicated, or overlap.

    """
    payload: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_fields: set[str] = set()
    for index, raw_group in enumerate(output_groups):
        if isinstance(raw_group, Mapping):
            group_id = raw_group.get("id")
            raw_fields = raw_group.get("fields")
        else:
            group_id = getattr(raw_group, "id", None)
            raw_fields = getattr(raw_group, "fields", None)
        if not isinstance(group_id, str) or not group_id:
            msg = f"Artifact output group {index} must have a non-empty string id."
            raise TypeError(msg)
        if isinstance(raw_fields, np.ndarray):
            raw_fields = raw_fields.tolist()
        if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
            msg = f"Artifact output group {group_id!r} must contain fields."
            raise TypeError(msg)
        fields = tuple(raw_fields)
        if any(not isinstance(field, str) or not field for field in fields):
            msg = f"Artifact output group {group_id!r} fields must be non-empty strings."
            raise TypeError(msg)
        if group_id in seen_group_ids or len(fields) != len(set(fields)):
            msg = f"Artifact output group {group_id!r} is duplicated or contains duplicate fields."
            raise ValueError(msg)
        overlap = seen_fields.intersection(fields)
        if overlap:
            msg = f"Artifact output groups overlap on fields: {sorted(overlap)}."
            raise ValueError(msg)
        payload.append({"id": group_id, "fields": list(fields)})
        seen_group_ids.add(group_id)
        seen_fields.update(fields)
    if not payload:
        msg = "Artifact group objective requires at least one output group."
        raise ValueError(msg)
    return payload


def output_standard_deviations_from_state(
    normalizer_state: Mapping[str, Any],
    *,
    output_fields: Iterable[str],
) -> dict[str, float]:
    """
    Extract positive train-fitted output scales from saved normalizer state.

    Parameters
    ----------
    normalizer_state : collections.abc.Mapping[str, Any]
        Persisted normalizer mapping containing ``out_normalizer.std`` with the
        authoritative ``[1, channels, 1, 1]`` layout.
    output_fields : Iterable[str]
        Ordered unique TaskSpec output names aligned with the channel axis.

    Returns
    -------
    dict[str, float]
        Output field to exact finite positive training standard deviation.

    Raises
    ------
    KeyError, TypeError, ValueError
        If state, layout, fields, or fitted scales are invalid.

    """
    if not isinstance(normalizer_state, Mapping):
        msg = "Artifact normalizer state must be a mapping."
        raise TypeError(msg)
    fields = tuple(output_fields)
    if not fields or len(fields) != len(set(fields)) or any(not isinstance(field, str) or not field for field in fields):
        msg = "Artifact output fields must be unique non-empty strings."
        raise ValueError(msg)
    if "out_normalizer.std" not in normalizer_state:
        msg = "Artifact normalizer state is missing 'out_normalizer.std'."
        raise KeyError(msg)
    raw_scales = normalizer_state["out_normalizer.std"]
    detach = getattr(raw_scales, "detach", None)
    if callable(detach):
        raw_scales = detach()
    cpu = getattr(raw_scales, "cpu", None)
    if callable(cpu):
        raw_scales = cpu()
    to_numpy = getattr(raw_scales, "numpy", None)
    if callable(to_numpy):
        raw_scales = to_numpy()
    array = np.asarray(raw_scales)
    expected_shape = (1, len(fields), 1, 1)
    if array.shape != expected_shape or not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        msg = f"Artifact output normalizer standard deviations must have real shape {expected_shape}, got {array.shape}."
        raise TypeError(msg)
    values = array.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        msg = "Artifact output normalizer standard deviations must be finite and strictly positive."
        raise ValueError(msg)
    return {field: float(value) for field, value in zip(fields, values, strict=True)}


def _validated_train_scales(
    train_standard_deviations: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
) -> dict[str, float]:
    """Return an exact finite positive train-scale mapping for the fields."""
    if not isinstance(train_standard_deviations, Mapping) or set(train_standard_deviations) != set(fields):
        msg = "Artifact train standard deviations must map exactly the grouped output fields."
        raise ValueError(msg)
    result: dict[str, float] = {}
    for field in fields:
        raw_value = train_standard_deviations[field]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
            msg = f"Artifact train standard deviation for {field!r} must be a real scalar."
            raise TypeError(msg)
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            msg = f"Artifact train standard deviation for {field!r} must be finite and strictly positive."
            raise ValueError(msg)
        result[field] = value
    return result


def _row_statistic(
    *,
    raw_sse: Any,
    raw_count: Any,
    raw_rmse: Any,
    sse_column: str,
    count_column: str,
    rmse_column: str,
    row_index: int,
) -> tuple[float, int, float]:
    """Validate one row-local SSE/count/RMSE triplet and return native values."""
    if isinstance(raw_sse, bool) or not isinstance(raw_sse, (int, float, np.integer, np.floating)):
        msg = f"{sse_column} row {row_index} must be a real scalar."
        raise TypeError(msg)
    if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)) or int(raw_count) <= 0:
        msg = f"{count_column} row {row_index} must be a positive integer."
        raise TypeError(msg)
    if isinstance(raw_rmse, bool) or not isinstance(raw_rmse, (int, float, np.integer, np.floating)):
        msg = f"{rmse_column} row {row_index} must be a real scalar."
        raise TypeError(msg)
    sse_value = float(raw_sse)
    count_value = int(raw_count)
    rmse_value = float(raw_rmse)
    if not math.isfinite(sse_value) or sse_value < 0.0 or not math.isfinite(rmse_value) or rmse_value < 0.0:
        msg = f"Artifact evidence in {sse_column!r} row {row_index} must be finite and non-negative."
        raise ValueError(msg)
    expected_rmse = math.sqrt(sse_value / count_value)
    if not math.isclose(
        rmse_value,
        expected_rmse,
        rel_tol=NORMALIZED_OBJECTIVE_TOLERANCE["rtol"],
        abs_tol=NORMALIZED_OBJECTIVE_TOLERANCE["atol"],
    ):
        msg = f"{rmse_column} row {row_index} does not match its SSE/count evidence."
        raise ValueError(msg)
    return sse_value, count_value, rmse_value


def aggregate_normalized_group_macro_rmse(
    frame: pd.DataFrame,
    *,
    output_groups: Iterable[Any],
    train_standard_deviations: Mapping[str, Any],
    normalization_denominator_floor: float,
) -> dict[str, Any]:
    """
    Finalize the global equal macro mean over physical output groups.

    Physical SSE and element counts are summed before any square root. Each
    group's physical component MSEs are divided by the sum of its train-fitted
    component variances, and finalized group errors receive equal macro weight.
    Per-case or per-batch RMSE values never enter the objective reduction.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema per-case table containing physical and normalized
        sufficient-statistic columns.
    output_groups : Iterable[Any]
        Ordered task-owned output groups covering the complete output contract.
    train_standard_deviations : collections.abc.Mapping[str, Any]
        Exact positive per-field standard deviations fitted on training data.
    normalization_denominator_floor : float
        Persisted non-negative denominator addition used only to verify retained
        component-normalized diagnostics. It does not enter the objective.

    Returns
    -------
    dict[str, Any]
        Objective semantics, global field and group statistics, equal group
        weights, the dimensionless objective value, agreement tolerance, and the
        complete-finite-support counts. When any prediction value is non-finite,
        the objective is unavailable: value and statistics are ``None`` and the
        counts explain why; it is never computed over the valid subset.

    Raises
    ------
    KeyError
        If any required evidence column, including ``prediction_validity``, is absent.
    TypeError, ValueError
        If groups, scales, evidence, counts, or diagnostic consistency fail.
    RuntimeError, FloatingPointError
        If no elements can be finalized or the aggregate is non-finite.

    """
    groups = output_group_payload(output_groups)
    fields = tuple(field for group in groups for field in group["fields"])
    scales = _validated_train_scales(train_standard_deviations, fields=fields)
    if (
        isinstance(normalization_denominator_floor, bool)
        or not isinstance(normalization_denominator_floor, (int, float, np.integer, np.floating))
        or not math.isfinite(float(normalization_denominator_floor))
        or float(normalization_denominator_floor) < 0.0
    ):
        msg = "Artifact normalization denominator floor must be finite and non-negative."
        raise ValueError(msg)
    denominator_floor = float(normalization_denominator_floor)
    if not frame.columns.is_unique:
        msg = "Artifact group macro RMSE cannot consume duplicate DataFrame columns."
        raise ValueError(msg)
    objective = {
        "objective_id": "normalized_group_macro_rmse",
        "kind": "group_macro_rmse",
        "reduction": "group_macro_element_mean",
        "space": "physical",
        "groups": groups,
        "direction": "minimize",
        "unit": "1",
        "agreement_tolerance": dict(NORMALIZED_OBJECTIVE_TOLERANCE),
        **validity_contract.support_counts(
            validity_contract.admit_field_prediction_validity(text, channels=fields) for text in frame["prediction_validity"]
        ),
    }
    if not objective["available"]:
        return {**objective, "value": None, "field_statistics": None, "group_statistics": None}

    field_summary: dict[str, dict[str, float | int]] = {}
    for field_name in fields:
        physical_columns = physical_statistic_columns(field_name)
        normalized_columns = normalized_statistic_columns(field_name)
        missing = [name for name in (*physical_columns, *normalized_columns) if name not in frame.columns]
        if missing:
            msg = f"Artifact group macro RMSE is missing sufficient-statistic columns: {missing}."
            raise KeyError(msg)
        physical_sse = 0.0
        physical_count = 0
        normalized_sse = 0.0
        normalized_count = 0
        for row_index, values in enumerate(
            zip(
                *(frame[column].tolist() for column in (*physical_columns, *normalized_columns)),
                strict=True,
            )
        ):
            row_physical_sse, row_physical_count, _ = _row_statistic(
                raw_sse=values[0],
                raw_count=values[1],
                raw_rmse=values[2],
                sse_column=physical_columns[0],
                count_column=physical_columns[1],
                rmse_column=physical_columns[2],
                row_index=row_index,
            )
            row_normalized_sse, row_normalized_count, _ = _row_statistic(
                raw_sse=values[3],
                raw_count=values[4],
                raw_rmse=values[5],
                sse_column=normalized_columns[0],
                count_column=normalized_columns[1],
                rmse_column=normalized_columns[2],
                row_index=row_index,
            )
            expected_normalized_sse = row_physical_sse / (scales[field_name] + denominator_floor) ** 2
            if row_physical_count != row_normalized_count or not math.isclose(
                row_normalized_sse,
                expected_normalized_sse,
                rel_tol=NORMALIZED_DIAGNOSTIC_TOLERANCE["rtol"],
                abs_tol=NORMALIZED_DIAGNOSTIC_TOLERANCE["atol"],
            ):
                msg = (
                    f"Normalized evidence for field {field_name!r}, row {row_index} contradicts physical evidence "
                    "and the saved normalizer denominator."
                )
                raise ValueError(msg)
            physical_sse += row_physical_sse
            physical_count += row_physical_count
            normalized_sse += row_normalized_sse
            normalized_count += row_normalized_count
        if physical_count <= 0 or normalized_count <= 0:
            msg = f"Artifact group macro RMSE cannot finalize field {field_name!r} without elements."
            raise RuntimeError(msg)
        physical_mse = physical_sse / physical_count
        normalized_rmse = math.sqrt(normalized_sse / normalized_count)
        field_summary[field_name] = {
            "physical_squared_error_sum": physical_sse,
            "physical_element_count": physical_count,
            "physical_rmse": math.sqrt(physical_mse),
            "train_standard_deviation": scales[field_name],
            "normalized_squared_error_sum": normalized_sse,
            "normalized_element_count": normalized_count,
            "normalized_rmse": normalized_rmse,
        }

    resolved_groups = tuple(OutputGroupSpec(id=str(group["id"]), fields=tuple(group["fields"])) for group in groups)
    squared_error_sums = {field: float(field_summary[field]["physical_squared_error_sum"]) for field in fields}
    element_counts = {field: int(field_summary[field]["physical_element_count"]) for field in fields}
    finalized = finalize_group_rmse_statistics(
        resolved_groups,
        squared_error_sums=squared_error_sums,
        element_counts=element_counts,
        train_standard_deviations=scales,
    )

    group_summary: dict[str, dict[str, Any]] = {}
    group_weight = 1.0 / len(groups)
    for group in groups:
        group_id = str(group["id"])
        group_fields = tuple(group["fields"])
        physical_mse_sum = sum(squared_error_sums[field] / element_counts[field] for field in group_fields)
        train_variance_sum = sum(scales[field] ** 2 for field in group_fields)
        normalized_group_rmse = finalized.normalized[group_id]
        physical_group_rmse = finalized.physical[group_id]
        group_summary[group_id] = {
            "fields": list(group_fields),
            "weight": group_weight,
            "physical_component_mse_sum": physical_mse_sum,
            "train_variance_sum": train_variance_sum,
            "normalized_rmse": normalized_group_rmse,
            "physical_rmse": physical_group_rmse,
            "objective_contribution": group_weight * normalized_group_rmse,
        }

    value = finalized.normalized_macro
    if not math.isfinite(value):
        msg = "Artifact normalized_group_macro_rmse finalized to a non-finite value."
        raise FloatingPointError(msg)
    return {**objective, "value": value, "field_statistics": field_summary, "group_statistics": group_summary}


def steady_physics_provenance(task: TaskSpec) -> dict[str, Any]:
    """
    Return the exact steady-Brinkman residual diagnostic contract of one task.

    Parameters
    ----------
    task : TaskSpec
        Validated steady-flow task owning equation, continuity, and boundary.

    Returns
    -------
    dict[str, Any]
        JSON-safe provenance written by artifact generation and required
        verbatim by artifact readers.

    """
    brinkman = domain.physics.brinkman
    return {
        "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
        "task_id": task.id,
        "task_contract_digest": task.contract_digest,
        "equation_kind": task.physics.kind,
        "equation_set": task.physics.equation_set,
        "continuity_kind": task.physics.continuity,
        "boundary_condition_kind": task.physics.boundary,
        "constants": {
            "dynamic_viscosity_pa_s": brinkman.AIR_DYNAMIC_VISCOSITY,
            "porosity_floor": brinkman.POROSITY_FLOOR,
            "permeability_scale_floor_m2": brinkman.PERMEABILITY_SCALE_FLOOR,
            "permeability_determinant_floor": brinkman.PERMEABILITY_DETERMINANT_FLOOR,
            "permeability_cross_ratio_clip": brinkman.PERMEABILITY_CROSS_RATIO_CLIP,
        },
        "permeability_representation": {
            "Kxx": "10**stored_log10_ratio_to_1_m2",
            "Kxy": "stored_dimensionless_ratio_times_sqrt(Kxx*Kyy)",
            "Kyy": "10**stored_log10_ratio_to_1_m2",
            "inverse": "normalized_symmetric_2x2_inverse_with_declared_floors",
        },
        "scales": {
            "pressure": {"formula": "mean_inlet(p_in_bc)", "unit": "Pa"},
            "pressure_gradient": {"formula": "pressure / (max(y) - min(y))", "unit": "Pa/m"},
            "velocity": {"formula": "exp(mean(0.5*(ln(Kxx) + ln(Kyy)))) * pressure_gradient / mu", "unit": "m/s"},
            "flux": {"formula": "velocity * (max(x) - min(x))", "unit": "m^2/s"},
        },
        "derivatives": {
            "kind": ARTIFACT_DERIVATIVE_KIND,
            "operator_axes": list(task.operator_axes),
            "grid_axes": ["y", "x"],
        },
        "interior_crop": EVAL_PAD,
        "residual_evaluation_region": {
            "momentum_residual_mse": "interior grid after symmetric cell crop",
            "flux_balance_mse": "every full-grid row",
            "pressure_boundary_mse": "full-grid inlet and outlet masks",
            "pressure_inlet_mse": "full-grid y-min inlet mask",
            "pressure_outlet_mean_square": "square of the sample mean on the full-grid y-max outlet mask",
            "residual_arrays": "full grid",
        },
        "scalar_definitions": {
            "momentum_residual_mse": {"formula": "mean(Rx**2 + Ry**2)", "unit": "1"},
            "flux_balance_mse": {"formula": "mean(flux_balance**2)", "unit": "1"},
            "pressure_boundary_mse": {"formula": "pressure_inlet_mse + pressure_outlet_mean_square", "unit": "1"},
            "pressure_inlet_mse": {"formula": "mean_inlet((p-p_in_bc)**2) / pressure**2", "unit": "1"},
            "pressure_outlet_mean_square": {"formula": "(mean_outlet(p) / pressure)**2", "unit": "1"},
        },
        "array_definitions": {
            "Rx": {"formula": "(-dp/dx + div(tau)_x - mu*(K^-1*u)_x) / pressure_gradient", "unit": "1"},
            "Ry": {"formula": "(-dp/dy + div(tau)_y - mu*(K^-1*u)_y) / pressure_gradient", "unit": "1"},
            "flux_balance": {"formula": "(q(y) - mean_y(q)) / flux with q(y) = trapezoid_x(v)", "unit": "1"},
        },
    }


def ordered_indices_sha256(indices: Iterable[int]) -> str:
    """
    Return the canonical digest for an ordered integer membership.

    Parameters
    ----------
    indices : Iterable[int]
        Source indices in saved membership order. Order and duplicates, if any,
        participate in the canonical compact-JSON byte representation.

    Returns
    -------
    str
        Lowercase SHA-256 digest of that exact ordered representation.

    """
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_stage_path(save_root: str | Path) -> Path:
    """Return the one deterministic sibling stage of an artifact target for both tasks."""
    target = Path(save_root)
    return target.parent / f".{target.name}.stage"


def artifact_provenance_path(save_root: str | Path) -> Path:
    """Return the versioned provenance sidecar path for an artifact root."""
    return Path(save_root) / ARTIFACT_PROVENANCE_FILENAME


def artifact_output_manifest(save_root: str | Path) -> dict[str, Any]:
    """
    Build the exact digest manifest for one complete artifact payload.

    Parameters
    ----------
    save_root : str | pathlib.Path
        Artifact target containing exactly one Parquet table and a non-empty
        ``npz`` directory.

    Returns
    -------
    dict[str, Any]
        Artifact-relative paths and SHA-256 digests in deterministic name order.

    Raises
    ------
    RuntimeError
        If the target lacks exactly one Parquet file or any NPZ case payload.

    """
    root = Path(save_root)
    parquet_files = sorted(root.glob("*.parquet"))
    npz_files = sorted((root / "npz").glob("*.npz"))
    if len(parquet_files) != 1:
        msg = f"Artifact payload must contain exactly one Parquet file, found {len(parquet_files)} in {root}."
        raise RuntimeError(msg)
    if not npz_files:
        msg = f"Artifact payload contains no NPZ files: {root / 'npz'}"
        raise RuntimeError(msg)

    def entry(payload_path: Path) -> dict[str, Any]:
        """Bind one artifact-relative payload path to its complete-file digest."""
        return {
            "path": payload_path.relative_to(root).as_posix(),
            "sha256": cached_file_sha256(payload_path),
        }

    return {
        "parquet": entry(parquet_files[0]),
        "npz": [entry(payload_path) for payload_path in npz_files],
    }
