"""Define lightweight persisted contracts for generated analysis artifacts.

Responsibilities:
  - Declare artifact, provenance, residual, and derivative contract constants
  - Own the steady-Brinkman residual diagnostic provenance contract
  - Name the per-case physical statistic and metric columns and derive the per-case metrics from the statistics
  - Finalize the material-case selection objective and per-field metrics of one artifact role from its case statistics
  - Read the output Train scales from the run's common scaling and validate saved scale mappings
  - Build ordered membership digests, completion paths, and payload manifests
  - Hash each file identity at most once per process
  - Recognize Optuna trial leaves from the validated layout recorded in a saved run config and refuse them with one wording
  - Build and validate the runtime provenance key set both tasks persist
  - Own the per-case NPZ file name, which is the Dataset package case ID
  - Declare the per-task analysis projection and the producer evidence every artifact of both tasks records
  - Own the freshness guard for requested artifacts and stages and the producer check of every comparison

Design principles:
  - Persisted names and schema values have one authoritative owner
  - Objective aggregation delegates to the one group objective of checkpoint selection
  - Physical case statistics, counts, reference floors and train-fitted scales remain explicit evidence
  - Reading an artifact validates its recorded analysis digest as evidence; only reuse or continuation of a
    requested artifact compares it with the running projection

This module does NOT:
  - Import Torch directly or run trained-model inference
  - Write artifact payloads or publish completion markers
  - Admit, rebuild, lock, time, or render artifact caches
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Final

import numpy as np

from src import common, domain, experiments, learning
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.domain.tasks.domain_task_spec import OutputGroupSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from src.datasets.preprocessing.dataset_preprocessing_normalization import TrainScaling
    from src.domain.tasks.domain_task_spec import TaskSpec


ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"
STEADY_ARTIFACT_KIND = "steady_field"
INFERENCE_PRECISION = "float32"
_RUNTIME_REQUIRED_KEYS = frozenset(
    {"requested_policy", "resolved_device", "device_type", "pytorch_version", "precision", "inference_batch_size", "source_tree_id"}
)
_RUNTIME_OPTIONAL_KEYS = frozenset({"cuda_index", "cuda_device_name", "cuda_build_version"})
NORMALIZED_OBJECTIVE_TOLERANCE = {"rtol": 1e-12, "atol": 1e-12}
EVAL_PAD = 2
# Per-case group-weighted relative norms; NaN marks a reference below its floor, never an invalid prediction alone.
RELATIVE_CASE_COLUMNS: Final = ("physical_relative_l2", "physical_relative_h1")
# Artifact residuals use one fixed, most accurate reference operator for every
# run, independent of the derivative a run was trained with.
ARTIFACT_DERIVATIVE_KIND = "physical"
_ARTIFACT_IDENTITY_CONTRACT = {
    "schema_kind": "evaluation_artifact_semantic_identity",
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
_SHA256_HEX_LENGTH: Final = 64
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
            "domain/physics/domain_physics_brinkman.py": "steady Brinkman residual contract, reached only by steady artifacts",
            "domain/physics/domain_physics_contracts.py": "steady physics-kind vocabulary, reached only by the steady payload check",
            "learning/inference/learning_inference.py": "steady inference reconstruction, reached through the service's task dispatch",
            "learning/training/learning_training_budget.py": (
                "admission of a completed lineage from its recorded budget state, bound by the checkpoint and receipt digests"
            ),
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
                "normalization",
                "output_scales",
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
                "metric_evidence",
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
    texts = ("requested_policy", "resolved_device", "device_type", "pytorch_version", "cuda_device_name", "cuda_build_version")
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


def case_statistic_columns(field_name: str) -> tuple[str, ...]:
    """
    Return the per-case physical statistic columns of one output field.

    Parameters
    ----------
    field_name : str
        Exact non-empty TaskSpec output field name.

    Returns
    -------
    tuple[str, ...]
        ``physical_<statistic>_<field>`` in :data:`learning.metrics.metrics.CASE_STATISTICS` order.

    Raises
    ------
    ValueError
        If the field name is empty or not text.

    """
    if not isinstance(field_name, str) or not field_name:
        msg = "Artifact output field names must be non-empty strings."
        raise ValueError(msg)
    return tuple(f"physical_{statistic}_{field_name}" for statistic in learning.metrics.metrics.CASE_STATISTICS)


def element_count_columns(field_names: Iterable[str]) -> frozenset[str]:
    """Return the per-case full-grid and H1-interior element-count columns of ``field_names``."""
    return frozenset(f"physical_{count}_{field}" for field in field_names for count in ("count", "h1_count"))


def case_metric_columns(field_names: Iterable[str]) -> tuple[str, ...]:
    """Return the per-case metric columns derived from the case statistics, in :func:`case_metric_values` order."""
    return (*(f"{space}_rmse_{field}" for field in field_names for space in ("physical", "normalized")), *RELATIVE_CASE_COLUMNS)


def case_metric_values(
    sums: np.ndarray,
    *,
    fields: Sequence[str],
    output_groups: Sequence[OutputGroupSpec],
    train_scales: Mapping[str, float],
    reference_floors: Mapping[str, float],
) -> dict[str, float]:
    """
    Return the per-case metrics persisted beside one case's statistics.

    ``physical_rmse_<field>`` and ``normalized_rmse_<field>`` are the case's RMSE in physical units and
    divided by the Train scale. ``physical_relative_<norm>`` averages the case's per-field relative norms
    with the output-group weights and is NaN when a field's reference falls below its floor.
    """
    metrics = learning.metrics.metrics
    count, squared = (metrics.CASE_STATISTICS.index(name) for name in ("count", "squared_error"))
    values: dict[str, float] = {}
    for index, field in enumerate(fields):
        rmse = math.sqrt(float(sums[index, squared] / sums[index, count]))
        values.update({f"physical_rmse_{field}": rmse, f"normalized_rmse_{field}": rmse / train_scales[field]})
    for norm in metrics.RELATIVE_NORMS:
        per_field = {field: metrics.case_relative_norm(sums[index], norm=norm, floor=reference_floors[field]) for index, field in enumerate(fields)}
        value = metrics.group_weighted_mean(per_field, output_groups)
        values[f"physical_relative_{norm}"] = math.nan if value is None else value
    return values


def metric_evidence_provenance(task: TaskSpec) -> dict[str, Any]:
    """Return the evaluator's per-case statistic, reduction and reference-floor contract of one steady task."""
    metrics = learning.metrics.metrics
    return {
        "case_statistic_columns": {field: list(case_statistic_columns(field)) for field in task.output_names},
        "accumulation_dtype": "float64",
        "h1": {"length_scale_m": metrics.H1_LENGTH_SCALE_M, "interior_crop": metrics.H1_INTERIOR_CROP, "derivatives": "non-periodic, metres"},
        "reference_floors": {field: task.reference_floor(field) for field in task.output_names},
        "train_scale_source": "saved_run_normalizer.output_scales",
        "reduction": "per case over its grid, equally over the cases of a material, then over the materials; "
        "RMSE roots after the means, a relative norm divides within each case",
    }


def output_group_payload(output_groups: Iterable[Any]) -> list[dict[str, Any]]:
    """
    Normalize task-owned output groups to an ordered JSON-safe declaration.

    Parameters
    ----------
    output_groups : Iterable[Any]
        Task ``OutputGroupSpec`` objects or equivalent mappings containing
        ``id``, ``fields`` and ``vector``.

    Returns
    -------
    list[dict[str, Any]]
        Ordered group identifiers, their ordered member fields and whether
        they form one vector error term.

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
            vector = raw_group.get("vector")
        else:
            group_id = getattr(raw_group, "id", None)
            raw_fields = getattr(raw_group, "fields", None)
            vector = getattr(raw_group, "vector", None)
        if not isinstance(vector, bool):
            msg = f"Artifact output group {index} must declare a boolean vector flag."
            raise TypeError(msg)
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
        payload.append({"id": group_id, "fields": list(fields), "vector": vector})
        seen_group_ids.add(group_id)
        seen_fields.update(fields)
    if not payload:
        msg = "Artifact group objective requires at least one output group."
        raise ValueError(msg)
    return payload


def output_scales(
    scaling: TrainScaling,
    *,
    output_fields: Iterable[str],
) -> dict[str, float]:
    """
    Return the operational physical Train state scale of every output field.

    Parameters
    ----------
    scaling : TrainScaling
        The run's common Train-fitted scaling; its physical state statistics are read, never the dimensionless
        steady encoding.
    output_fields : Iterable[str]
        Ordered unique TaskSpec output names; they must equal the scaling's state channels.

    Returns
    -------
    dict[str, float]
        Output field to ``max(sigma, floor)`` of the physical state, the scale of the selection and this objective.

    Raises
    ------
    ValueError
        If the output fields do not equal the scaling's state channels in order.

    """
    fields = tuple(output_fields)
    state = scaling.physical_state
    if state.names != fields:
        msg = f"Saved scaling state channels {list(state.names)} do not equal the output fields {list(fields)}."
        raise ValueError(msg)
    return dict(zip(fields, (float(value) for value in state.scale.double().tolist()), strict=True))


def validated_train_scales(
    train_scales: object,
    *,
    fields: tuple[str, ...],
    label: str = "Train scale",
) -> dict[str, float]:
    """Return an exact finite positive per-field mapping (Train scales or reference floors) for the fields."""
    if not isinstance(train_scales, Mapping) or set(train_scales) != set(fields):
        msg = f"Artifact {label}s must map exactly the grouped output fields."
        raise ValueError(msg)
    result: dict[str, float] = {}
    for field in fields:
        raw_value = train_scales[field]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
            msg = f"Artifact {label} for {field!r} must be a real scalar."
            raise TypeError(msg)
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            msg = f"Artifact {label} for {field!r} must be finite and strictly positive."
            raise ValueError(msg)
        result[field] = value
    return result


def artifact_case_membership(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """
    Return the ``(simulation_case_id, material_family)`` of every artifact row.

    Raw Parquet tables carry both in their JSON ``meta`` column, as size-one
    lists because each row keeps the collated batch rank; expanded analysis
    frames carry them as flattened metadata columns.
    """
    if "meta" in frame.columns:
        metadata = [json.loads(value) if isinstance(value, str) else value for value in frame["meta"].tolist()]
    elif {"simulation_case_id", "material_family"} <= set(frame.columns):
        metadata = frame[["simulation_case_id", "material_family"]].to_dict("records")
    else:
        msg = "Artifact rows must carry simulation_case_id and material_family metadata."
        raise KeyError(msg)

    def unwrap(value: object) -> object:
        return value[0] if isinstance(value, list | tuple) and len(value) == 1 else value

    memberships: list[tuple[str, str]] = []
    for row_index, meta in enumerate(metadata):
        case_id = unwrap(meta.get("simulation_case_id")) if isinstance(meta, Mapping) else None
        material = unwrap(meta.get("material_family")) if isinstance(meta, Mapping) else None
        if not isinstance(case_id, str) or not case_id or not isinstance(material, str) or not material:
            msg = f"Artifact row {row_index} lacks a simulation_case_id or material_family."
            raise ValueError(msg)
        memberships.append((case_id, material))
    return tuple(memberships)


def aggregate_normalized_group_macro_rmse(
    frame: pd.DataFrame,
    *,
    output_groups: Iterable[Any],
    train_scales: Mapping[str, Any],
    reference_floors: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Finalize the selection objective and the per-field metrics of one artifact role from its case statistics.

    Every row is one case with its physical statistics per field. The reducers of
    :mod:`learning.metrics.metrics` that training validation uses average them
    equally over the cases of a material and then over the materials: the group
    objective and the per-field RMSE take their root after the means, a relative
    norm divides within each case first.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema per-case table containing the case statistic and per-case
        metric columns and each case's identity and material.
    output_groups : Iterable[Any]
        Ordered task-owned output groups covering the complete output contract.
    train_scales : collections.abc.Mapping[str, Any]
        Exact positive per-field Train state scales ``max(sigma, floor)``.
    reference_floors : collections.abc.Mapping[str, Any]
        Positive per-field reference-RMS floors of the relative norms.

    Returns
    -------
    dict[str, Any]
        Objective semantics, ``field_metrics`` keyed by the training metric ids
        (physical and normalized RMSE and MAE, relative L2 and H1 per field and
        group-weighted, ``None`` when unavailable) with ``relative_unavailable_cases``,
        group statistics and weights, the dimensionless objective value, agreement
        tolerance, the population sizes, and the complete-finite-support counts.
        When any prediction value is non-finite,
        everything is unavailable: value and statistics are ``None`` and the counts
        explain why; it is never computed over the valid subset.

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
    scales = validated_train_scales(train_scales, fields=fields)
    floors = validated_train_scales(reference_floors, fields=fields, label="reference floor")
    if not frame.columns.is_unique:
        msg = "Artifact group macro RMSE cannot consume duplicate DataFrame columns."
        raise ValueError(msg)
    objective = {
        "objective_id": "normalized_group_macro_rmse",
        "kind": "group_macro_rmse",
        "reduction": "material_case_mean",
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
        return {**objective, "value": None, "field_metrics": None, "relative_unavailable_cases": None, "group_statistics": None}

    metrics = learning.metrics.metrics
    statistic_columns = [column for field in fields for column in case_statistic_columns(field)]
    metric_columns = list(case_metric_columns(fields))
    missing = [name for name in (*statistic_columns, *metric_columns) if name not in frame.columns]
    if missing:
        msg = f"Artifact role metrics are missing per-case columns: {missing}."
        raise KeyError(msg)
    sums = frame[statistic_columns].to_numpy(dtype=np.float64).reshape(len(frame), len(fields), len(metrics.CASE_STATISTICS))
    memberships = artifact_case_membership(frame)
    cases = tuple(metrics.CaseStatistics(case_id, material, fields, sums[row]) for row, (case_id, material) in enumerate(memberships))
    specs = tuple(OutputGroupSpec(id=str(group["id"]), fields=tuple(group["fields"]), vector=bool(group["vector"])) for group in groups)
    derived = (case_metric_values(case.sums, fields=fields, output_groups=specs, train_scales=scales, reference_floors=floors) for case in cases)
    expected = [[values[column] for column in metric_columns] for values in derived]
    if not np.allclose(frame[metric_columns].to_numpy(dtype=np.float64), expected, equal_nan=True, **NORMALIZED_OBJECTIVE_TOLERANCE):
        msg = "Artifact per-case metrics contradict their case statistics, Train scales or reference floors."
        raise ValueError(msg)
    terms = domain.tasks.spec.error_terms(specs)
    population = metrics.group_objective((case.errors() for case in cases), terms=terms, train_scales=scales)
    field_metrics, unavailable_cases = metrics.field_metric_values(cases, train_scales=scales, reference_floors=floors, output_groups=specs)
    group_summary: dict[str, dict[str, Any]] = {}
    for group in groups:
        members = [term for term in terms if set(term.fields) <= set(group["fields"])]
        weight = sum(term.weight for term in members)
        contribution = sum(term.weight * population.term_rmse[term.id] for term in members)
        group_summary[str(group["id"])] = {
            "fields": list(group["fields"]),
            "weight": weight,
            "normalized_rmse": contribution / weight,
            "physical_rmse": math.sqrt(sum(population.field_mse[field] for field in group["fields"])),
            "objective_contribution": contribution,
        }
    return {
        **objective,
        "value": population.value,
        "case_count": population.case_count,
        "material_count": population.material_count,
        "field_metrics": field_metrics,
        "relative_unavailable_cases": unavailable_cases,
        "group_statistics": group_summary,
    }


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
        "task_id": task.id,
        "task_contract_digest": task.contract_digest,
        "equation_kind": task.physics.kind,
        "equation_set": task.physics.equation_set,
        "continuity_kind": task.physics.continuity,
        "boundary_condition_kind": task.physics.boundary,
        "constants": {
            "dynamic_viscosity_pa_s": domain.physics.contracts.AIR_DYNAMIC_VISCOSITY,
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


def manifest_npz_digests(outputs: Any) -> dict[str, str]:
    """Return the recorded SHA-256 digest of each NPZ payload, keyed by artifact-relative path."""
    if not isinstance(outputs, Mapping) or set(outputs) != {"parquet", "npz"} or not isinstance(outputs["npz"], list):
        msg = "Artifact output manifest has an unsupported schema."
        raise ValueError(msg)
    digests: dict[str, str] = {}
    for entry in outputs["npz"]:
        path, digest = (entry.get("path"), entry.get("sha256")) if isinstance(entry, Mapping) and set(entry) == {"path", "sha256"} else (None, None)
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
            or path in digests
        ):
            msg = "Artifact output manifest NPZ entry is unsafe."
            raise ValueError(msg)
        digests[path] = digest
    return digests


def require_output_structure(save_root: str | Path, outputs: Any) -> None:
    """Bind the on-disk payload set to the recorded manifest; NPZ digests are verified when a case opens."""
    root = Path(save_root)
    recorded_npz = manifest_npz_digests(outputs)
    msg = "Artifact output manifest does not match the payload files."
    npz_directory = root / "npz"
    try:
        parquet_files = sorted(root.glob("*.parquet"))
        if (
            len(parquet_files) != 1
            or parquet_files[0].is_symlink()
            or not parquet_files[0].is_file()
            or npz_directory.is_symlink()
            or not npz_directory.is_dir()
        ):
            raise ValueError(msg)
        on_disk_npz = sorted(path.relative_to(root).as_posix() for path in npz_directory.glob("*.npz"))
        parquet_entry = {"path": parquet_files[0].relative_to(root).as_posix(), "sha256": common.serialization.cached_file_sha256(parquet_files[0])}
        if (
            outputs["parquet"] != parquet_entry
            or on_disk_npz != sorted(recorded_npz)
            or any(not (root / relative).is_file() or (root / relative).is_symlink() for relative in on_disk_npz)
        ):
            raise ValueError(msg)
    except OSError as error:
        raise ValueError(msg) from error


@contextmanager
def open_verified_case_payload(artifact_root: str | Path, provenance: Mapping[str, Any], path: Path) -> Iterator[BinaryIO]:
    """Yield a case payload handle whose complete-file digest equals the one recorded in the artifact provenance; parse only from this handle."""
    root = Path(artifact_root).resolve()
    unreadable = f"Steady case payload has no recorded digest or is unreadable: {path}"
    try:
        recorded = manifest_npz_digests(provenance["outputs"]).get(path.resolve().relative_to(root).as_posix())
    except (KeyError, OSError, ValueError) as error:
        msg = f"{unreadable}: {error}"
        raise ValueError(msg) from error
    changed = ValueError(f"Steady case payload digest changed: {path}")
    if recorded is None:
        raise changed
    with ExitStack() as stack:
        try:
            verified = stack.enter_context(common.serialization.open_verified_file(path, recorded, changed))
        except OSError as error:
            msg = f"{unreadable}: {error}"
            raise ValueError(msg) from error
        yield verified.stream
        # The caller has read everything it uses; an in-place rewrite meanwhile would have fed it unverified bytes.
        try:
            verified.require_unchanged()
        except OSError as error:
            msg = f"Steady case payload changed while it was read: {path}"
            raise ValueError(msg) from error


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
            "sha256": common.serialization.cached_file_sha256(payload_path),
        }

    return {
        "parquet": entry(parquet_files[0]),
        "npz": [entry(payload_path) for payload_path in npz_files],
    }
