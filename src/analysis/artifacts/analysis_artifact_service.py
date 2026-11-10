"""Discover, validate, rebuild, generate, and reuse split-aware run artifacts.

Responsibilities:
  - Admit completed runs by explicit current path, skipping Optuna trial leaves by their recorded layout and refusing one selected as the target
  - Skip non-completed leaves while discovering an unselected root, and refuse them when selected by name
  - Bind caches to exact checkpoint/config/normalizer/dataset/split identity
  - Stamp steady artifacts with the running analysis producer and refuse a complete target or stage of other analysis code
  - Reject partial or semantically mismatched caches without substitution
  - Rebuild only explicit exact targets and propagate every material failure
  - Load valid artifacts or generate only missing roles through one build flow shared by both tasks

Design principles:
  - Saved split membership is always required
  - Provenance is published last as the cache completion marker
  - One deterministic stage per target survives failures; the role validator of evaluation_artifact_loader admits
    the stage, every cache hit, and the published target
  - The required timing block is written into the stage before validation, so a timing failure publishes nothing
  - Rebuild publishes a validated replacement only for the requested target
  - Local notebook orchestration never invokes tracking uploads

This module does NOT:
  - Parse CLI arguments or choose which runs a user intends to process
  - Define scientific metrics or the Parquet/NPZ payload schema
  - Render notebook figures or bypass bounded W&B publication contracts
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src import common, domain, experiments
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.datasets.contracts import dataset_contracts_transient as transient_contract

from . import analysis_artifact_performance as artifact_performance
from . import analysis_artifact_plan as artifact_plan
from . import contracts, generation, timing, transient

if TYPE_CHECKING:
    from src.analysis.evaluation.evaluation_artifact_loader import LoadedRunArtifacts
    from src.learning.learning_device import DeviceResolution

ArtifactSplit = artifact_plan.ArtifactSplit
ArtifactDatasetRole = artifact_plan.ArtifactDatasetRole
_MINIMUM_GRID_VARIANT_STRIDE = 2


class ArtifactCacheError(RuntimeError):
    """
    Signal rejection of an existing or upload-bound artifact cache.

    Artifact admission raises this exception when schema, scientific identity,
    aggregate, manifest, or payload evidence cannot prove that a cache is both
    current and complete. Missing source runs/datasets and invalid caller paths
    retain their ordinary project exception types.
    """


@dataclass(frozen=True)
class ArtifactRequest:
    """
    Describe the complete request identity expected from one artifact target.

    Parameters
    ----------
    provenance : dict[str, Any]
        Canonical run, model, dataset, evaluator, physics, selection,
        generation, and runtime request evidence. Generated aggregate/results
        are deliberately absent until artifact publication.
    source_indices : tuple[int, ...]
        Exact ordered membership as positions in the ordered concatenation of the role's complete source Datasets.
    batch_size : int
        Saved resolved evaluation batch size used by online and artifact inference.
    case_ids : tuple[str, ...]
        Dataset case IDs in saved membership order; they name the per-case NPZ files.

    Notes
    -----
    Field rebinding is frozen. The provenance mapping is internal transport and
    is treated as immutable after request construction.

    """

    provenance: dict[str, Any]
    source_indices: tuple[int, ...]
    batch_size: int
    case_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TargetBuild:
    """Task-specific steps of one locked target build; the shared flow owns staging, timing, admission and publication."""

    case_ids: tuple[str, ...]
    generate: Callable[[Path, artifact_performance.ArtifactProgressReporter], None]
    finish_stage: Callable[[Path], None]


@dataclass(frozen=True)
class _EvaluatorArtifactContract:
    """
    Carry the task-declared payload schema resolved from admitted provenance.

    Parameters
    ----------
    task_id : str
        Exact task identity copied from ``provenance.run.task``.
    input_fields, output_fields, output_units : tuple[str, ...]
        Ordered names and output units required in every Parquet row and NPZ.
    output_groups : tuple[tuple[str, tuple[str, ...]], ...]
        Ordered task-owned physical output groups.
    train_standard_deviations : dict[str, float]
        Exact raw output scales from the saved training normalizer.
    normalization_denominator_floor : float
        Saved transform denominator addition used to validate normalized evidence.
    group_metric_columns : tuple[str, ...]
        Task-declared per-case physical and normalized group diagnostics.
    physics_kind : str
        Physics contract selecting generic versus steady-Brinkman payload rules.

    """

    task_id: str
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    output_units: tuple[str, ...]
    output_groups: tuple[tuple[str, tuple[str, ...]], ...]
    train_standard_deviations: dict[str, float]
    normalization_denominator_floor: float
    group_metric_columns: tuple[str, ...]
    physics_kind: str

    @property
    def is_steady_brinkman(self) -> bool:
        """Return whether the concrete steady-flow diagnostic schema applies."""
        return self.physics_kind == domain.physics.contracts.STEADY_BRINKMAN_KIND

    @property
    def velocity_unit(self) -> str:
        """Return the shared unit of the task-owned velocity output group."""
        matching_groups = tuple(fields for group_id, fields in self.output_groups if group_id == "velocity")
        if len(matching_groups) != 1:
            msg = "Steady artifact validation requires one task-owned velocity output group."
            raise ArtifactCacheError(msg)
        unit_by_field = dict(zip(self.output_fields, self.output_units, strict=True))
        try:
            units = {unit_by_field[field] for field in matching_groups[0]}
        except KeyError as error:
            msg = "Steady artifact velocity-group fields must be declared outputs."
            raise ArtifactCacheError(msg) from error
        if len(units) != 1:
            msg = f"Steady artifact velocity-group fields must share one unit, got {sorted(units)}."
            raise ArtifactCacheError(msg)
        return next(iter(units))


# ======================================================================
# Utilities
# ======================================================================


def cleanup_runtime(device: torch.device) -> None:
    """
    Collect Python state and release CUDA caches only for a CUDA execution path.

    Parameters
    ----------
    device : torch.device
        Concrete CPU or CUDA device resolved by the artifact-service boundary.

    Raises
    ------
    TypeError
        If ``device`` is not a concrete supported ``torch.device``.

    Notes
    -----
    CPU cleanup intentionally performs no CUDA availability query or CUDA API call.

    """
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Artifact cleanup requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _refuse_optuna_trial(run_dir: Path) -> None:
    """Refuse an explicitly selected Optuna trial leaf: trials keep no checkpoints and have no artifacts."""
    if common.paths.resolve_run_config_path(run_dir).is_file():
        contracts.refuse_optuna_trial(_load_run_config(run_dir), label=str(run_dir))


def iter_run_dirs(root: Path, *, run_names: Iterable[str] | None = None) -> Iterable[Path]:
    """Yield validated saved leaves, optionally selected by persisted scientific names.

    Without a name selection, a leaf whose summary reports a status other than
    ``completed`` is skipped with one printed line; a selected name is admitted
    strictly and refused with the resume hint. Optuna trial leaves below the root
    are skipped; a root that is itself a trial leaf is refused.
    """
    root = Path(root).expanduser()
    selected = {common.paths.validate_logical_name(name, label="run_name") for name in (run_names or ())}
    if not root.is_dir():
        msg = f"Run discovery root not found: {root}"
        raise FileNotFoundError(msg)
    markers = set(common.paths.CURRENT_RUN_REQUIRED_FILES)
    candidates: list[Path] = []
    for directory, children, filenames in os.walk(root):
        if markers.intersection(filenames):
            candidates.append(Path(directory))
            children.clear()  # Run artifacts belong to this leaf, not another saved run.
    candidates.sort()
    if root in candidates:
        _refuse_optuna_trial(root)
    found: set[str] = set()
    for candidate in candidates:
        config = _load_run_config(candidate) if common.paths.resolve_run_config_path(candidate).is_file() else None
        if config is not None and contracts.is_optuna_trial_config(config):
            continue  # Optuna trials keep no checkpoints and are never evaluated; a normal run still fails closed below.
        run_section = config.get("run") if config is not None else None
        name = run_section.get("name") if isinstance(run_section, Mapping) else None
        if selected and isinstance(name, str) and name not in selected:
            continue  # Select by name before admission so an unrelated run cannot block the selection.
        if not selected and common.paths.resolve_run_summary_path(candidate).is_file():
            status = experiments.run.read_run_summary(candidate).get("status")
            if isinstance(status, str) and status != "completed":
                print(f"[SKIP] {name or candidate.name} | status {status} (resume it to completion to evaluate it)")
                continue  # A selected name is admitted below instead and refused with the same resume hint.
        admitted = experiments.run.validate_evaluable_run(candidate)
        found.add(str(admitted["scientific_run_name"]))
        yield candidate.resolve()
    if selected - found:
        msg = f"Selected scientific run names were not found below {root}: {sorted(selected - found)}."
        raise FileNotFoundError(msg)


def _load_run_config(run_dir: Path) -> Mapping[str, Any]:
    """Load and validate the top-level mapping in current config.yaml."""
    config_path = common.paths.resolve_run_config_path(run_dir)
    config = experiments.config.loader.load_yaml(config_path)
    if not isinstance(config, Mapping):
        msg = f"config.yaml must contain a top-level mapping: {config_path}"
        raise TypeError(msg)
    return config


def load_run_artifact_plan(run_dir: Path) -> artifact_plan.ArtifactPlan:
    """
    Admit an evaluable run of either task and return its artifact roles from the saved split.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError, RuntimeError
        If the run is incomplete, an Optuna trial, or its config disagrees with ``split_indices.pt``.

    """
    run_dir = Path(run_dir).expanduser().resolve()
    return artifact_plan.artifact_plan_from_completed(experiments.run.validate_evaluable_run(run_dir), run_dir=run_dir)


def _artifact_save_root(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    evaluation_spatial_stride: int = 1,
) -> Path:
    """Resolve one split- and Evaluation-grid-qualified artifact save root."""
    stride = transient_contract.validate_spatial_stride(evaluation_spatial_stride)
    if split not in artifact_plan.ARTIFACT_SPLITS:
        msg = f"Unknown artifact split {split!r}."
        raise ValueError(msg)
    if stride == 1:
        if split == "eval":
            return common.paths.resolve_id_analysis_dir(run_dir)
        return common.paths.resolve_ood_analysis_dir(run_dir, split, dataset_name)
    variant_root = common.paths.resolve_analysis_root(run_dir) / f"grid_s{stride}"
    return variant_root / "id" if split == "eval" else variant_root / split / dataset_name


def _grid_token_stride(name: str) -> int | None:
    """Return one exact noncanonical Evaluation stride from a path token."""
    suffix = name.removeprefix("grid_s")
    if not name.startswith("grid_s") or not suffix.isdigit() or suffix.startswith("0"):
        return None
    stride = int(suffix)
    return stride if stride >= _MINIMUM_GRID_VARIANT_STRIDE else None


def _artifact_target_stride(
    target: Path,
    *,
    analysis_root: Path,
    split: ArtifactSplit,
    dataset_name: str,
) -> int | None:
    """Resolve a canonical or exact sibling grid-variant role target."""
    canonical = analysis_root / "id" if split == "eval" else analysis_root / split / dataset_name
    if target == canonical:
        return 1
    if split == "eval":
        variant_root = target.parent
        exact_leaf = target.name == "id"
    else:
        variant_root = target.parent.parent
        exact_leaf = target.parent.name == split and target.name == dataset_name
    stride = _grid_token_stride(variant_root.name)
    if variant_root.parent != analysis_root or not exact_leaf:
        return None
    return stride


def _normalise_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving symbolic links."""
    return Path(os.path.abspath(path.expanduser()))  # noqa: PTH100 -- lexical normalization must not follow symlinks


def _validated_artifact_target(*, run_dir: Path, save_root: Path) -> tuple[Path, Path]:
    """
    Admit one exact run-owned artifact leaf without following symlink aliases.

    Only ``analysis/id`` and one named ``analysis/<material_ood|parameter_ood>/<dataset>`` leaf are
    accepted. Lexical containment and every existing path component are checked
    before a destructive rebuild or publication lock path is derived.
    """
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    analysis_root = _normalise_path(common.paths.resolve_analysis_root(run_root))
    target = _normalise_path(Path(save_root))
    if not analysis_root.is_relative_to(run_root) or analysis_root == run_root or not target.is_relative_to(analysis_root):
        msg = f"Refusing rebuild outside one exact artifact target: {target}"
        raise ValueError(msg)
    id_target = analysis_root / "id"
    role_roots = {analysis_root / role for role in common.paths.ANALYSIS_OOD_ROLES}
    is_named_role_target = target.parent in role_roots and target.name not in {"", ".", ".."}
    is_id_grid_target = target.name == "id" and target.parent.parent == analysis_root and _grid_token_stride(target.parent.name) is not None
    is_named_grid_target = (
        target.parent.name in common.paths.ANALYSIS_OOD_ROLES
        and target.parent.parent.parent == analysis_root
        and target.name not in {"", ".", ".."}
        and _grid_token_stride(target.parent.parent.name) is not None
    )
    if target != id_target and not is_named_role_target and not is_id_grid_target and not is_named_grid_target:
        msg = f"Refusing rebuild outside one exact artifact target: {target}"
        raise ValueError(msg)

    current = analysis_root
    for part in (Path(), *target.relative_to(analysis_root).parents[::-1], target.relative_to(analysis_root)):
        candidate = current if part == Path() else analysis_root / part
        if candidate.is_symlink():
            msg = f"Refusing rebuild outside one exact artifact target through symbolic link: {candidate}"
            raise ValueError(msg)
    return analysis_root, target


def _artifact_lock_path(*, run_dir: Path, save_root: Path) -> Path:
    """Return one centralized lock path outside an exact deletable artifact target."""
    _analysis_root, target = _validated_artifact_target(run_dir=run_dir, save_root=save_root)
    return common.paths.resolve_artifact_lock_path(target)


def _completion_marker_identity(path: Path) -> tuple[int, int, int, int] | None:
    """Return replacement-sensitive identity for one cache completion marker."""
    try:
        result = path.stat()
    except FileNotFoundError:
        return None
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def _indices_sha256(indices: Iterable[int]) -> str:
    """Hash ordered split membership without hashing dataset tensor contents."""
    return contracts.ordered_indices_sha256(indices)


def _build_artifact_request(
    *,
    run_dir: Path,
    role: artifact_plan.ArtifactRolePlan,
    device_resolution: DeviceResolution,
) -> ArtifactRequest:
    """
    Build and validate the complete semantic request for one artifact cache.

    This boundary admits the evaluable run, its exact saved split membership,
    the saved source dataset identity, the resolved task/objective, model capacity,
    normalizer, physics semantics, and runtime device decision. The current sources
    are opened and checked against that identity by inference when they are generated.
    Filesystem locations, mtimes, and byte sizes are intentionally excluded from the
    scientific identity so equivalent relocations remain comparable.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluable run whose immutable evidence owns the artifacts.
    role : ArtifactRolePlan
        Saved ID-test, material-OOD or parameter-OOD role to materialize, all sources in split order.
    device_resolution : DeviceResolution
        Already resolved execution-device decision.

    Returns
    -------
    ArtifactRequest
        Exact provenance document and complete ordered source membership.

    Raises
    ------
    RuntimeError, TypeError, ValueError
        If run evidence, dataset identity, split membership, or resolved
        semantics disagree or are incomplete.

    """
    run_dir = Path(run_dir).expanduser().resolve()
    admitted = experiments.run.validate_evaluable_run(run_dir)
    if artifact_plan.artifact_plan_from_completed(admitted, run_dir=run_dir).role(role.split) != role:
        msg = f"Requested {role.split!r} artifact role does not match the saved split of this run."
        raise RuntimeError(msg)
    run_config = admitted["config"]
    summary = admitted["summary"]
    data_cfg = run_config.get("data")
    run_cfg = run_config.get("run")
    model_cfg = run_config.get("model")
    loss_cfg = run_config.get("loss")
    evaluation_cfg = run_config.get("evaluation")
    if (
        not isinstance(data_cfg, Mapping)
        or not isinstance(run_cfg, Mapping)
        or not isinstance(model_cfg, Mapping)
        or not isinstance(loss_cfg, Mapping)
        or not isinstance(evaluation_cfg, Mapping)
    ):
        msg = f"config.yaml must contain data, run, model, loss, and evaluation mappings: {common.paths.resolve_run_config_path(run_dir)}"
        raise TypeError(msg)
    from src import learning  # noqa: PLC0415

    if not isinstance(device_resolution, learning.device.DeviceResolution):
        msg = "Artifact provenance requires one resolved runtime device decision."
        raise TypeError(msg)
    raw_batch_size = data_cfg.get("batch_size")
    if isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, Integral) or int(raw_batch_size) <= 0:
        msg = "Evaluable run config data.batch_size must be a positive integer."
        raise TypeError(msg)
    batch_size = int(raw_batch_size)
    task = experiments.config.loader.validate_resolved_task_contract(run_config)
    output_groups = contracts.output_group_payload(task.output_groups)
    normalizer_state = admitted.get("normalizer_state")
    if not isinstance(normalizer_state, Mapping):
        msg = "Evaluable run must contain the validated saved normalizer state."
        raise TypeError(msg)
    train_standard_deviations = contracts.output_standard_deviations_from_state(
        normalizer_state,
        output_fields=task.output_names,
    )
    source_indices = role.source_indices
    metrics = evaluation_cfg.get("metrics")
    if not isinstance(metrics, list):
        msg = "config.yaml evaluation.metrics must be a list."
        raise TypeError(msg)
    objective = experiments.config.loader.get_resolved_objective(run_config)
    physics_cfg = loss_cfg.get("physics")
    if not isinstance(physics_cfg, Mapping):
        msg = "config.yaml loss.physics must be a resolved mapping."
        raise TypeError(msg)
    physics_provenance: dict[str, Any] | None = None
    if task.physics.kind == domain.physics.contracts.STEADY_BRINKMAN_KIND:
        physics_provenance = contracts.steady_physics_provenance(task)

    raw_parameter_counts = summary.get("model_parameter_counts")
    parameter_counts: dict[str, int] | None = None
    if raw_parameter_counts is not None:
        if not isinstance(raw_parameter_counts, Mapping):
            msg = "Run summary model_parameter_counts must be a mapping when recorded."
            raise TypeError(msg)
        parameter_counts = {}
        for name in ("total", "trainable"):
            value = raw_parameter_counts.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                msg = "Run model_parameter_counts must contain positive integer total and trainable values."
                raise TypeError(msg)
            parameter_counts[name] = int(value)
    architecture = model_cfg.get("params")
    if not isinstance(architecture, Mapping):
        msg = "Resolved config model.params must be a mapping."
        raise TypeError(msg)

    model_provenance: dict[str, Any] = {
        "kind": model_cfg.get("kind"),
        "architecture": dict(architecture),
        "physics_enabled": physics_cfg.get("enabled") is True,
    }
    if parameter_counts is not None:
        model_provenance["parameter_counts"] = parameter_counts
    provenance: dict[str, Any] = {
        "producer": contracts.analysis_producer(task.id),
        "run": {
            "name": admitted["scientific_run_name"],
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "effective_config_digest": admitted["effective_config_digest"],
            "best_checkpoint_sha256": admitted["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": admitted["selected_checkpoint_epoch"],
            "normalizer_sha256": admitted["normalizer_sha256"],
            "selected_checkpoint_role": admitted["selected_checkpoint_role"],
        },
        "model": model_provenance,
        "split_role": role.split,
        "dataset": role.dataset_provenance(),
        "selection": {
            "index_key": f"{role.split_role}_indices",
            "full_selected_case_count": len(source_indices),
            "effective_case_count": len(source_indices),
            "generation_limit": None,
            "full_ordered_source_indices_sha256": _indices_sha256(source_indices),
            "effective_ordered_source_indices_sha256": _indices_sha256(source_indices),
        },
        "normalizer": {
            "sha256": admitted["normalizer_sha256"],
            "identity": "saved_run_normalizer.pt",
            "fit_split": task.preprocessing.fit_split,
            "output_normalization": task.preprocessing.output_normalization,
            "denominator_floor": 1e-7,
            "output_standard_deviations": train_standard_deviations,
        },
        "evaluator": {
            "metrics": metrics,
            "objective": objective,
            "input_fields": list(task.input_names),
            "input_units": {field.name: field.unit for field in task.inputs},
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
            "output_groups": output_groups,
            "physics_kind": task.physics.kind,
            "group_objective_evidence": {
                "squared_error_accumulation_dtype": "float64",
                "per_case_physical_columns": {field: list(contracts.physical_statistic_columns(field)) for field in task.output_names},
                "per_case_normalized_columns": {field: list(contracts.normalized_statistic_columns(field)) for field in task.output_names},
                "train_scale_source": "saved_run_normalizer.output_standard_deviations",
                "dataset_reduction": ("sum physical field SSE/count, finalize shared-scale group RMSE, equal macro mean over output groups"),
            },
            "predictive_metrics": {
                "rel_l2": "per-case arithmetic mean of physical per-field relative L2 ratios",
                "rel_h1": "per-case arithmetic mean of physical per-field relative H1 ratios on the declared artifact region where available",
                "physical_rmse_columns": {field: contracts.physical_statistic_columns(field)[2] for field in task.output_names},
            },
        },
        "generation": {
            "effective_case_limit": None,
            "inference_batch_size": batch_size,
            "compression": "numpy savez_compressed",
        },
        "runtime": contracts.runtime_provenance(device_resolution.as_dict(), inference_batch_size=batch_size),
        "artifact_kind": contracts.STEADY_ARTIFACT_KIND,
    }
    if physics_provenance is not None:
        provenance["physics"] = physics_provenance
    return ArtifactRequest(
        provenance=provenance,
        source_indices=source_indices,
        batch_size=batch_size,
        case_ids=role.case_ids,
    )


def _runtime_identities(request: ArtifactRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the minimal dataset and model identities bound to timing."""
    run = request.provenance.get("run")
    if not isinstance(run, Mapping):
        msg = "Artifact request lacks timing identity provenance."
        raise TypeError(msg)
    return (
        timing.runtime_dataset_identity(request.provenance),
        {
            "run_name": run.get("name"),
            "effective_config_digest": run.get("effective_config_digest"),
            "best_checkpoint_sha256": run.get("best_checkpoint_sha256"),
        },
    )


def _report_admitted_target(frame: pd.DataFrame) -> None:
    """Print the warning and timing lines both tasks share for one admitted target."""
    provenance = frame.attrs["artifact_provenance"]
    report = provenance[timing.TIMING_KEY]["report"]
    validity_contract.report_invalid_predictions(provenance["evaluation"]["prediction_validity"])
    if frame.attrs["artifact_kind"] == "transient_sequence":
        measured = len(report["cases"])
    else:
        measured = report["aggregates"]["batch_amortized_neural_operator_forward_s"]["count"]
    print(f"timing validated: neural-operator runtime measured for {measured} cases")


def _cache_has_outputs(*, save_root: Path, parquet_path: Path, npz_dir: Path) -> bool:
    """
    Detect any content that requires fail-closed cache admission.

    A non-empty target counts even when its files are unrecognized, preventing
    normal generation from overwriting interrupted or foreign content.
    """
    return any(
        (
            save_root.is_dir() and any(save_root.iterdir()),
            parquet_path.exists(),
            contracts.artifact_provenance_path(save_root).exists(),
            any(npz_dir.glob("*.npz")),
            any(save_root.glob(".*.tmp")),
            any(npz_dir.glob(".*.tmp")),
        )
    )


def _require_metadata_without_identity(raw_meta: Any, *, label: str) -> dict[str, Any]:
    """
    Parse one case metadata object while protecting authoritative identity fields.

    JSON text and mappings are accepted, but case/source/split identity keys are
    rejected because those values must come only from validated Parquet and NPZ
    scalar fields.
    """
    if not isinstance(raw_meta, str):
        msg = f"{label} metadata must be a JSON string."
        raise ArtifactCacheError(msg)
    try:
        metadata = json.loads(raw_meta)
    except json.JSONDecodeError as error:
        msg = f"{label} metadata is invalid JSON: {error}"
        raise ArtifactCacheError(msg) from error
    if not isinstance(metadata, dict):
        msg = f"{label} metadata must decode to an object."
        raise ArtifactCacheError(msg)
    reserved = {"case_index", "source_index", "split_local_index"}.intersection(metadata)
    if reserved:
        msg = f"{label} metadata duplicates reserved top-level identity fields: {sorted(reserved)}."
        raise ArtifactCacheError(msg)
    return metadata


def _require_provenance_string_list(value: Any, *, label: str) -> tuple[str, ...]:
    """Return one non-empty, duplicate-free JSON string list."""
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        msg = f"Artifact provenance {label} must be a non-empty list of strings."
        raise ArtifactCacheError(msg)
    values = tuple(value)
    if len(set(values)) != len(values):
        msg = f"Artifact provenance {label} contains duplicate field names."
        raise ArtifactCacheError(msg)
    return values


def _evaluator_artifact_contract(provenance: Mapping[str, Any]) -> _EvaluatorArtifactContract:
    """
    Resolve the exact evaluator payload contract from admitted provenance.

    Task identity, unique input/output order, units, task-owned output groups,
    raw saved training scales, declared group diagnostics, and physics kind are
    validated once and frozen for every subsequent Parquet/NPZ check.
    """
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        msg = "Artifact provenance must contain an evaluator mapping."
        raise ArtifactCacheError(msg)
    run = provenance.get("run")
    if not isinstance(run, Mapping):
        msg = "Artifact provenance must contain a run mapping."
        raise ArtifactCacheError(msg)
    task_id = run.get("task")
    if not isinstance(task_id, str) or not task_id:
        msg = "Artifact provenance run.task must be a non-empty string."
        raise ArtifactCacheError(msg)
    input_fields = _require_provenance_string_list(evaluator.get("input_fields"), label="evaluator.input_fields")
    output_fields = _require_provenance_string_list(evaluator.get("output_fields"), label="evaluator.output_fields")
    raw_units = evaluator.get("output_units")
    if not isinstance(raw_units, Mapping) or set(raw_units) != set(output_fields):
        msg = "Artifact provenance evaluator.output_units must map exactly the declared output fields."
        raise ArtifactCacheError(msg)
    if any(not isinstance(raw_units[name], str) or not raw_units[name] for name in output_fields):
        msg = "Artifact provenance evaluator.output_units values must be non-empty strings."
        raise ArtifactCacheError(msg)
    try:
        group_payload = contracts.output_group_payload(evaluator.get("output_groups", ()))
    except (TypeError, ValueError) as error:
        msg = f"Artifact evaluator output groups are invalid: {error}"
        raise ArtifactCacheError(msg) from error
    grouped_fields = tuple(field for group in group_payload for field in group["fields"])
    if grouped_fields != output_fields:
        msg = "Artifact evaluator output groups do not partition declared outputs in order."
        raise ArtifactCacheError(msg)

    normalizer = provenance.get("normalizer")
    raw_scales = normalizer.get("output_standard_deviations") if isinstance(normalizer, Mapping) else None
    denominator_floor = normalizer.get("denominator_floor") if isinstance(normalizer, Mapping) else None
    if (
        isinstance(denominator_floor, bool)
        or not isinstance(denominator_floor, Real)
        or not np.isfinite(float(denominator_floor))
        or float(denominator_floor) < 0.0
    ):
        msg = "Artifact normalizer denominator_floor must be finite and non-negative."
        raise ArtifactCacheError(msg)
    if not isinstance(raw_scales, Mapping) or set(raw_scales) != set(output_fields):
        msg = "Artifact normalizer output_standard_deviations must map exactly the output fields."
        raise ArtifactCacheError(msg)
    train_standard_deviations: dict[str, float] = {}
    for field in output_fields:
        value = raw_scales[field]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) <= 0.0:
            msg = f"Artifact train standard deviation for {field!r} must be finite and strictly positive."
            raise ArtifactCacheError(msg)
        train_standard_deviations[field] = float(value)

    raw_metrics = evaluator.get("metrics")
    if not isinstance(raw_metrics, list):
        msg = "Artifact evaluator metrics must be a list."
        raise ArtifactCacheError(msg)
    group_fields = {tuple(group["fields"]) for group in group_payload}
    group_metric_columns: list[str] = []
    for metric in raw_metrics:
        if not isinstance(metric, Mapping) or metric.get("kind") not in {"group_rmse", "vector_rmse"}:
            continue
        metric_id = metric.get("id")
        raw_metric_fields = metric.get("fields")
        metric_fields = output_fields if raw_metric_fields == "all" else tuple(raw_metric_fields or ())
        if not isinstance(metric_id, str) or not metric_id or metric_fields not in group_fields:
            msg = "Artifact evaluator contains an invalid output-group diagnostic metric."
            raise ArtifactCacheError(msg)
        group_metric_columns.append(metric_id)
    if len(group_metric_columns) != len(set(group_metric_columns)):
        msg = "Artifact evaluator contains duplicate output-group diagnostic ids."
        raise ArtifactCacheError(msg)

    physics_kind = evaluator.get("physics_kind")
    if not isinstance(physics_kind, str) or not physics_kind:
        msg = "Artifact provenance evaluator.physics_kind must be a non-empty string."
        raise ArtifactCacheError(msg)
    return _EvaluatorArtifactContract(
        task_id=task_id,
        input_fields=input_fields,
        output_fields=output_fields,
        output_units=tuple(raw_units[name] for name in output_fields),
        output_groups=tuple((str(group["id"]), tuple(group["fields"])) for group in group_payload),
        train_standard_deviations=train_standard_deviations,
        normalization_denominator_floor=float(denominator_floor),
        group_metric_columns=tuple(group_metric_columns),
        physics_kind=physics_kind,
    )


def _require_npz_string_vector(value: np.ndarray, *, label: str) -> tuple[str, ...]:
    """Return an exact one-dimensional string vector from an NPZ payload."""
    if value.ndim != 1:
        msg = f"{label} must be a one-dimensional string array."
        raise ArtifactCacheError(msg)
    items = value.tolist()
    if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
        msg = f"{label} must contain only non-empty strings without object pickles."
        raise ArtifactCacheError(msg)
    return tuple(items)


def _require_finite_npz_array(value: np.ndarray, *, label: str, rank: int, finite: bool = True) -> np.ndarray:
    """Return a numeric NPZ array with the required rank; ``finite=False`` admits a non-finite prediction's values."""
    if value.ndim != rank or not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        msg = f"{label} must be a rank-{rank} numeric array."
        raise ArtifactCacheError(msg)
    if finite and not np.isfinite(value).all():
        msg = f"{label} contains non-finite values."
        raise ArtifactCacheError(msg)
    return value


def _validate_npz_payload(
    path: Path,
    *,
    case_index: int,
    source_index: int,
    split_local_index: int,
    contract: _EvaluatorArtifactContract,
    prediction_validity: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...] | None]:
    """
    Validate one per-case NPZ against its row identity, validity document and evaluator contract.

    Generic tasks require exact input/output field declarations, finite raw and
    target tensors, and a stored prediction whose recomputed validity equals the
    row document. Prediction, error and residual arrays may be non-finite only for
    a NONFINITE case. Steady Brinkman tasks additionally require permeability,
    boundary pressure, coordinates, and residual arrays with the declared grid
    shape. Object arrays and undeclared keys are rejected.

    Returns
    -------
    tuple[dict[str, Any], tuple[str, ...] | None]
        Identity-free case metadata and, for Brinkman artifacts, permeability
        channel names used for cross-checking the corresponding Parquet row.

    Raises
    ------
    ArtifactCacheError
        If the file is unreadable or any schema, identity, field, unit, shape,
        finiteness, or numerical-consistency check fails.

    """
    common_fields = {
        "case_index",
        "source_index",
        "split_local_index",
        "meta",
        "pred",
        "gt",
        "err",
        "artifact_fields",
        "artifact_units",
        "x_raw",
        "y_raw",
        "input_fields",
        "output_fields",
        "output_units",
    }
    steady_fields = {
        "kappa_encoded",
        "kappa",
        "kappa_names",
        "p_in_bc",
        "coordinates",
        "Rx",
        "Ry",
        "flux_balance",
    }
    required_fields = common_fields | steady_fields if contract.is_steady_brinkman else common_fields
    try:
        with np.load(path, allow_pickle=False) as artifact:
            payload = {name: np.asarray(artifact[name]) for name in artifact.files}
    except (OSError, TypeError, ValueError, KeyError) as error:
        msg = f"Cached NPZ is unreadable or incompatible: {path}: {error}"
        raise ArtifactCacheError(msg) from error

    fields = set(payload)
    missing = required_fields.difference(fields)
    unexpected = fields.difference(required_fields)
    if missing or unexpected:
        msg = f"Cached NPZ schema mismatch for {path}: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        raise ArtifactCacheError(msg)

    actual_identity: dict[str, int] = {}
    for name in ("case_index", "source_index", "split_local_index"):
        value = payload[name]
        if value.shape != () or not np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_):
            msg = f"Cached NPZ {path} field {name!r} must be one integer scalar."
            raise ArtifactCacheError(msg)
        actual_identity[name] = int(value.item())
    expected_identity = {
        "case_index": case_index,
        "source_index": source_index,
        "split_local_index": split_local_index,
    }
    if actual_identity != expected_identity:
        msg = f"Cached NPZ identity mismatch for {path}: expected {expected_identity}, got {actual_identity}."
        raise ArtifactCacheError(msg)

    metadata = _require_metadata_without_identity(
        payload["meta"].item(),
        label=f"Cached NPZ {path}",
    )
    input_fields = _require_npz_string_vector(
        payload["input_fields"],
        label=f"Cached NPZ {path} input_fields",
    )
    output_fields = _require_npz_string_vector(
        payload["output_fields"],
        label=f"Cached NPZ {path} output_fields",
    )
    output_units = _require_npz_string_vector(
        payload["output_units"],
        label=f"Cached NPZ {path} output_units",
    )
    artifact_fields = _require_npz_string_vector(
        payload["artifact_fields"],
        label=f"Cached NPZ {path} artifact_fields",
    )
    artifact_units = _require_npz_string_vector(
        payload["artifact_units"],
        label=f"Cached NPZ {path} artifact_units",
    )
    expected_artifact_fields = (*contract.output_fields, "U") if contract.is_steady_brinkman else contract.output_fields
    expected_artifact_units = (*contract.output_units, contract.velocity_unit) if contract.is_steady_brinkman else contract.output_units
    if (
        input_fields != contract.input_fields
        or output_fields != contract.output_fields
        or output_units != contract.output_units
        or artifact_fields != expected_artifact_fields
        or artifact_units != expected_artifact_units
    ):
        msg = f"Cached NPZ declared fields or units do not match evaluator provenance: {path}"
        raise ArtifactCacheError(msg)

    finite = prediction_validity["status"] == validity_contract.VALID
    prediction = _require_finite_npz_array(payload["pred"], label=f"Cached NPZ {path} pred", rank=3, finite=finite)
    target = _require_finite_npz_array(payload["gt"], label=f"Cached NPZ {path} gt", rank=3)
    error_array = _require_finite_npz_array(payload["err"], label=f"Cached NPZ {path} err", rank=3, finite=finite)
    inputs = _require_finite_npz_array(payload["x_raw"], label=f"Cached NPZ {path} x_raw", rank=3)
    raw_targets = _require_finite_npz_array(payload["y_raw"], label=f"Cached NPZ {path} y_raw", rank=3)
    expected_prediction_channels = len(expected_artifact_fields)
    if prediction.shape != target.shape or prediction.shape != error_array.shape:
        msg = f"Cached NPZ pred/gt/err shapes differ: {path}"
        raise ArtifactCacheError(msg)
    if prediction.shape[0] != expected_prediction_channels:
        msg = f"Cached NPZ prediction channels do not match the task artifact schema: {path}"
        raise ArtifactCacheError(msg)
    recomputed = validity_contract.build_field_prediction_validity(prediction[: len(contract.output_fields)], channels=contract.output_fields)
    if recomputed != prediction_validity:
        msg = f"Cached NPZ prediction contradicts its prediction validity document: {path}"
        raise ArtifactCacheError(msg)
    if inputs.shape[0] != len(contract.input_fields) or raw_targets.shape[0] != len(contract.output_fields):
        msg = f"Cached NPZ raw tensor channels do not match declared task fields: {path}"
        raise ArtifactCacheError(msg)
    if inputs.shape[1:] != raw_targets.shape[1:] or prediction.shape[1:] != raw_targets.shape[1:]:
        msg = f"Cached NPZ raw and prediction spatial shapes differ: {path}"
        raise ArtifactCacheError(msg)
    if not np.allclose(error_array, prediction - target, rtol=1e-6, atol=1e-8, equal_nan=True):
        msg = f"Cached NPZ err is not pred - gt: {path}"
        raise ArtifactCacheError(msg)

    kappa_names: tuple[str, ...] | None = None
    if contract.is_steady_brinkman:
        spatial_shape = raw_targets.shape[1:]
        kappa_encoded = _require_finite_npz_array(
            payload["kappa_encoded"],
            label=f"Cached NPZ {path} kappa_encoded",
            rank=3,
        )
        kappa = _require_finite_npz_array(payload["kappa"], label=f"Cached NPZ {path} kappa", rank=3)
        kappa_names = _require_npz_string_vector(
            payload["kappa_names"],
            label=f"Cached NPZ {path} kappa_names",
        )
        if kappa.shape != kappa_encoded.shape or kappa.shape != (len(kappa_names), *spatial_shape):
            msg = f"Cached NPZ permeability arrays do not match kappa_names or spatial shape: {path}"
            raise ArtifactCacheError(msg)
        p_in_bc = _require_finite_npz_array(payload["p_in_bc"], label=f"Cached NPZ {path} p_in_bc", rank=3)
        if p_in_bc.shape != (1, *spatial_shape):
            msg = f"Cached NPZ p_in_bc shape does not match the task grid: {path}"
            raise ArtifactCacheError(msg)
        coordinates = _require_finite_npz_array(
            payload["coordinates"],
            label=f"Cached NPZ {path} coordinates",
            rank=3,
        )
        if coordinates.shape != (2, *spatial_shape):
            msg = f"Cached NPZ coordinates must contain x/y fields on the task grid: {path}"
            raise ArtifactCacheError(msg)
        for name in ("Rx", "Ry"):
            residual = _require_finite_npz_array(
                payload[name],
                label=f"Cached NPZ {path} {name}",
                rank=2,
                finite=finite,
            )
            if residual.shape != spatial_shape:
                msg = f"Cached NPZ {name} shape does not match the task grid: {path}"
                raise ArtifactCacheError(msg)
        flux_balance = _require_finite_npz_array(payload["flux_balance"], label=f"Cached NPZ {path} flux_balance", rank=1, finite=finite)
        if flux_balance.shape != spatial_shape[:1]:
            msg = f"Cached NPZ flux_balance must hold one value per grid row: {path}"
            raise ArtifactCacheError(msg)

    return metadata, kappa_names


def _require_parquet_string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    """Normalize a Parquet nested string sequence without accepting scalar text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        msg = f"{label} must contain a sequence of non-empty strings."
        raise ArtifactCacheError(msg)
    return tuple(value)


def _validate_staged_case_payloads(stage: Path, *, request: ArtifactRequest) -> None:
    """
    Open every staged steady NPZ once and bind its arrays and metadata to the staged rows.

    This writer-side content check complements the shared role validator, which
    binds schemas, digests, provenance and membership without opening arrays; the
    published digests then carry the checked bytes to every later cache hit and read.
    """
    contract = _evaluator_artifact_contract(request.provenance)
    rows = pd.read_parquet(stage / common.paths.ANALYSIS_ARTIFACTS_FILENAME)
    if len(rows) != len(request.source_indices):
        msg = f"Staged Parquet has {len(rows)} rows, but {len(request.source_indices)} were expected."
        raise ArtifactCacheError(msg)
    names = contracts.dataset_case_payload_filenames(request.case_ids)
    for position, (source_index, name) in enumerate(zip(request.source_indices, names, strict=True)):
        row = rows.iloc[position]
        metadata, kappa_names = _validate_npz_payload(
            stage / "npz" / name,
            case_index=source_index + 1,
            source_index=source_index,
            split_local_index=position,
            contract=contract,
            prediction_validity=validity_contract.admit_field_prediction_validity(row.loc["prediction_validity"], channels=contract.output_fields),
        )
        if _require_metadata_without_identity(row.loc["meta"], label=f"Staged Parquet row {position}") != metadata:
            msg = f"Staged Parquet and NPZ metadata differ at row {position}."
            raise ArtifactCacheError(msg)
        if contract.is_steady_brinkman and _require_parquet_string_sequence(row.loc["kappa_names"], label=f"Staged row {position}") != kappa_names:
            msg = f"Staged Parquet and NPZ kappa_names differ at row {position}."
            raise ArtifactCacheError(msg)


def _artifact_stage_root(save_root: Path) -> Path:
    """Return the one deterministic sibling stage of a target; it survives a failed build for inspection or reuse."""
    save_root.parent.mkdir(parents=True, exist_ok=True)
    return contracts.artifact_stage_path(save_root)


def _recorded_marker(marker: Path) -> object:
    """Return one parsed completion marker, or None when it is unreadable, so the freshness guard reports it as malformed."""
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clear_artifact_stage(stage: Path) -> None:
    """Remove one exact stage after refusing symbolic indirection."""
    if stage.is_symlink():
        message = f"Refusing to rebuild through a symbolic artifact stage: {stage}"
        raise ValueError(message)
    if stage.exists():
        shutil.rmtree(stage)


def _publish_staged_artifact(*, run_dir: Path, save_root: Path, staging_root: Path) -> None:
    """
    Publish one validated sibling stage at an exact run-owned target.

    The stage must carry its provenance completion marker. A new target appears
    in one atomic rename. An existing target is first renamed to a unique backup,
    so it is briefly absent but never mixed. If publication fails, that backup is
    moved back before the exception escapes. A successful replacement removes the
    backup on a best-effort basis.

    Raises
    ------
    ValueError
        If the stage is not the deterministic stage sibling of the target.
    ArtifactCacheError
        If the stage lacks its completion marker.

    """
    _analysis_root, target = _validated_artifact_target(run_dir=run_dir, save_root=save_root)
    stage = _normalise_path(staging_root)
    if stage != contracts.artifact_stage_path(target):
        msg = f"Artifact staging root is not the expected sibling of the exact target: {stage}"
        raise ValueError(msg)
    if not contracts.artifact_provenance_path(stage).is_file():
        msg = f"Refusing to publish an artifact stage without its completion marker: {stage}"
        raise ArtifactCacheError(msg)

    backup = target.with_name(f".{target.name}.backup.{uuid4().hex}")
    moved_previous = False
    try:
        if target.exists():
            target.replace(backup)
            moved_previous = True
        stage.replace(target)
    except BaseException:
        if moved_previous and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    else:
        if moved_previous:
            with suppress(OSError):
                shutil.rmtree(backup)


def _progress_reporter(
    *,
    completed: Mapping[str, Any],
    task: str,
    total_cases: int,
    split: str,
    device_resolution: DeviceResolution,
    output_root: Path,
) -> artifact_performance.ArtifactProgressReporter:
    """Build and start the one progress reporter both tasks share from admitted run metadata."""
    config = completed.get("config")
    training = config.get("training") if isinstance(config, Mapping) else None
    raw_stage = training.get("comparison_arm") or training.get("stage") if isinstance(training, Mapping) else None
    reporter = artifact_performance.ArtifactProgressReporter(
        task=task,
        run=str(completed.get("scientific_run_name") or "run"),
        stage_label=str(raw_stage or "run"),
        checkpoint_label="best",
        device=device_resolution.device,
        dtype=contracts.INFERENCE_PRECISION,
        total_cases=total_cases,
        split=split,
        output_root=output_root,
    )
    reporter.startup()
    return reporter


def _transient_target_build(
    *,
    run_dir: Path,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    evaluation_spatial_stride: int,
) -> _TargetBuild:
    """Bind the transient sequence writer, which resumes an incomplete stage case by case."""
    from src.analysis.evaluation import evaluation_transient_artifact as sequence_artifact  # noqa: PLC0415

    def generate(stage: Path, reporter: artifact_performance.ArtifactProgressReporter) -> None:
        transient.generate_transient_role_artifact(
            run_dir=run_dir,
            role=role,
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            staging_root=stage,
            completed=completed,
            evaluation_spatial_stride=evaluation_spatial_stride,
            progress_reporter=reporter,
        )

    return _TargetBuild(
        case_ids=role.case_ids,
        generate=generate,
        finish_stage=sequence_artifact.cleanup_transient_resume_metadata,
    )


def _steady_target_build(
    *,
    run_dir: Path,
    task: domain.tasks.spec.TaskSpec,
    role: artifact_plan.ArtifactRolePlan,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    save_root: Path,
) -> _TargetBuild:
    """Bind the steady writer; batch-amortized timing cannot resume per case, so an incomplete stage restarts whole."""
    request = _build_artifact_request(run_dir=run_dir, role=role, device_resolution=device_resolution)
    split = role.split

    def generate(stage: Path, reporter: artifact_performance.ArtifactProgressReporter) -> None:
        from src import learning  # noqa: PLC0415

        _clear_artifact_stage(stage)
        timing_cases: list[dict[str, Any]] = []
        with reporter.phase("model_setup"):
            model, loader, processor, device = learning.inference.context.load_inference_context_with_resolution(
                run_dir=run_dir,
                device_resolution=device_resolution,
                dataset_root=dataset_root,
                split=role.split_role,
                batch_size=request.batch_size,
            )
        try:
            with reporter.phase("inference"):
                timing.warm_up_forward(
                    representative_batch=next(iter(loader)),
                    model=model,
                    processor=processor,
                    device=device,
                    passes=timing.WARMUP_PASSES,
                )
                generation.generate_artifacts(
                    task=task,
                    model=model,
                    loader=loader,
                    processor=processor,
                    device=device,
                    save_root=stage,
                    publication_root=save_root,
                    provenance=request.provenance,
                    case_ids=request.case_ids,
                    timing_cases=timing_cases,
                    progress_reporter=reporter,
                )
            dataset_identity, model_identity = _runtime_identities(request)
            report = timing.build_runtime_comparison(
                split_role=split,
                dataset_identity=dataset_identity,
                model_identity=model_identity,
                neural_runtime=timing.neural_runtime_metadata(device_metadata=device_resolution.as_dict(), model=model),
                batch_size=request.batch_size,
                cases=timing_cases,
                comsol_timing=None,
                batch_manifest_sha256=None,
                unavailable_reason="current Dataset selection has no aggregate simulation timing snapshot",
            )
        finally:
            del model, loader, processor
            cleanup_runtime(device)
        timing.write_stage_timing(stage, {"report": report})

    return _TargetBuild(
        case_ids=request.case_ids,
        generate=generate,
        finish_stage=lambda stage: _validate_staged_case_payloads(stage, request=request),
    )


def _run_or_load_artifacts_locked(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    rebuild: bool = False,
    evaluation_spatial_stride: int = 1,
) -> pd.DataFrame:
    """
    Load or generate artifacts for one run and saved split.

    Reuses existing Parquet+NPZ artifacts only after exact provenance and
    payload validation. If artifacts do not exist, runs split-aware inference via
    learning.inference.context.load_inference_context() and generates
    Parquet+NPZ artifacts via generation.generate_artifacts().

    Parameters
    ----------
    run_dir : Path
        Current run directory satisfying the saved-run contract.
    dataset_name : str
        Logical dataset name used for artifact file naming.
    split : {"eval", "material_ood", "parameter_ood"}
        Saved split role to load: "eval" is the ID-test membership, "material_ood" and "parameter_ood"
        the declared material and parameter holdouts.
    device_resolution : DeviceResolution
        Immutable device decision resolved once at the artifact boundary.
    dataset_root : Path
        Current final-dataset root.
    rebuild : bool, optional
        Force staged regeneration and atomically replace only this target.
    evaluation_spatial_stride : int, optional
        Transient artifact Evaluation-grid stride. The default one retains the canonical source grid.

    Returns
    -------
    pandas.DataFrame
        Validated non-empty artifact summary DataFrame.

    Notes
    -----
    - Artifacts cached in analysis/id or analysis/<material_ood|parameter_ood>/<dataset_name>/, as artifacts.parquet plus npz/.
    - ID and OOD evaluation always use saved split membership.
    - One validator (``evaluation_artifact_loader.admit_role_artifact``) admits the stage, a cache hit and the published target.
    - The deterministic stage ``.<target>.stage`` survives a failure; a complete stage is reused only after admission,
      an incomplete one resumes per case (transient) or restarts whole (steady).
    - The timing block is written into the stage before validation, so a timing failure publishes nothing.

    """
    from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader  # noqa: PLC0415

    dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    if split not in artifact_plan.ARTIFACT_SPLITS:
        msg = f"Artifact split must be one of {list(artifact_plan.ARTIFACT_SPLITS)}."
        raise ValueError(msg)
    completed = experiments.run.validate_completed_run(run_dir)
    task = experiments.config.loader.validate_resolved_task_contract(completed["config"])
    if task.id != "transient_drying" and evaluation_spatial_stride != 1:
        msg = "Non-unit --evaluation-spatial-stride is supported only for transient_drying artifacts."
        raise ValueError(msg)
    plan = artifact_plan.artifact_plan_from_completed(completed, run_dir=run_dir)
    role = plan.role(split)
    if role.dataset_name != dataset_name:
        msg = f"Requested dataset {dataset_name!r} does not match the saved {split!r} artifact Dataset {role.dataset_name!r}."
        raise RuntimeError(msg)
    save_root = _artifact_save_root(run_dir=run_dir, dataset_name=dataset_name, split=split, evaluation_spatial_stride=evaluation_spatial_stride)
    label = artifact_performance.split_label(split)
    print(f"artifacts for {run_dir.name} | {label} | dataset {dataset_name}")
    print(f"      run_dir={run_dir}")
    print(f"      save_root={save_root}")

    def admit(root: Path) -> pd.DataFrame:
        try:
            return artifact_loader.admit_role_artifact(
                run_dir,
                completed,
                selection_role=artifact_plan.dataset_role_of(split),
                root=root,
                evaluation_spatial_stride=evaluation_spatial_stride,
            ).frame
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
            msg = (
                f"Artifact cache is incompatible for {run_dir.name}/{split}/{dataset_name}: {type(error).__name__}: {error}. "
                "Rerun only this exact target with --rebuild."
            )
            raise ArtifactCacheError(msg) from error

    unit = f"{run_dir.name}/{split}/{dataset_name}"
    if not rebuild and _cache_has_outputs(
        save_root=save_root, parquet_path=save_root / common.paths.ANALYSIS_ARTIFACTS_FILENAME, npz_dir=save_root / "npz"
    ):
        print(f"[VALIDATE] {run_dir.name} | {label} | {dataset_name} (existing cache)")
        frame = admit(save_root)
        # Admission validated the recorded producer as evidence; only this requested result is compared with the running code.
        contracts.require_current_analysis(frame.attrs.get("artifact_provenance"), task_id=task.id, unit=f"Artifact {unit}")
        print(f"[LOAD] {run_dir.name} | {label} | {dataset_name} (validated cache)")
        _report_admitted_target(frame)
        return frame
    if not rebuild and save_root.exists() and (save_root.is_symlink() or not save_root.is_dir()):
        msg = f"Artifact target is unsafe; rerun only this exact target with --rebuild: {save_root}"
        raise ArtifactCacheError(msg)
    # Generating beside a published role of other analysis code would leave the run with two producers; --rebuild replaces them all.
    if not rebuild and (
        sibling := _sibling_from_other_analysis(
            run_dir=run_dir, plan=plan, task_id=task.id, split=split, evaluation_spatial_stride=evaluation_spatial_stride
        )
    ):
        msg = (
            f"Artifact {unit} would leave {run_dir.name} with two analysis producers: its published {sibling} artifact "
            "comes from other analysis code; rebuild the run's artifacts with --rebuild."
        )
        raise ArtifactCacheError(msg)
    stage_marker = contracts.artifact_provenance_path(contracts.artifact_stage_path(save_root))
    if not rebuild and stage_marker.is_file():
        # A complete stage would be published as this requested result, so it must come from the running analysis code.
        contracts.require_current_analysis(_recorded_marker(stage_marker), task_id=task.id, unit=f"Artifact stage {unit}")

    build = (
        _transient_target_build(
            run_dir=run_dir,
            completed=completed,
            role=role,
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            evaluation_spatial_stride=evaluation_spatial_stride,
        )
        if task.id == "transient_drying"
        else _steady_target_build(
            run_dir=run_dir,
            task=task,
            role=role,
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            save_root=save_root,
        )
    )
    reporter = _progress_reporter(
        completed=completed,
        task=task.id,
        total_cases=len(build.case_ids),
        split=split,
        device_resolution=device_resolution,
        output_root=save_root,
    )
    stage = _artifact_stage_root(save_root)
    try:
        if rebuild:
            _clear_artifact_stage(stage)
        if contracts.artifact_provenance_path(stage).is_file():
            with reporter.phase("finalization"):
                for case_id in build.case_ids:
                    reporter.case_reused(case_id=case_id, material=case_id.split("__", 1)[0])
        else:
            build.generate(stage, reporter)
        timing.write_stage_timing(stage, {"operational_performance": reporter.final_snapshot()})
        with reporter.phase("validation"):
            build.finish_stage(stage)
            admit(stage)
        with reporter.phase("publication"):
            _publish_staged_artifact(run_dir=run_dir, save_root=save_root, staging_root=stage)
            frame = admit(save_root)
        reporter.done()
    except BaseException:
        reporter.failure()
        raise
    finally:
        cleanup_runtime(device_resolution.device)
    _report_admitted_target(frame)
    return frame


def validate_artifact_upload_source(
    *,
    run_dir: Path,
    artifact_root: Path,
) -> Mapping[str, Any]:
    """
    Validate one explicit complete current artifact target for bounded upload.

    Parameters
    ----------
    run_dir : pathlib.Path
        Authoritative completed run that must own ``artifact_root``.
    artifact_root : pathlib.Path
        Exact ``analysis/id`` or named ``analysis/<material_ood|parameter_ood>/<dataset>`` target.

    Returns
    -------
    Mapping[str, Any]
        Admitted current provenance, including the verified payload manifest.

    Raises
    ------
    ArtifactCacheError
        If schemas, digests, payloads, or completed-run identities disagree.
    ValueError
        If the target escapes the run analysis tree or is not an exact leaf.

    Notes
    -----
    The gate never scans sibling targets, regenerates artifacts, or contacts W&B.

    """
    run_path = Path(run_dir).resolve()
    root = Path(artifact_root).resolve()
    _analysis_root, validated_target = _validated_artifact_target(
        run_dir=run_path,
        save_root=root,
    )
    if _normalise_path(validated_target) != _normalise_path(root):
        msg = f"Artifact upload target does not resolve to the explicit artifact root: {root}"
        raise ArtifactCacheError(msg)
    from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader  # noqa: PLC0415

    completed = experiments.run.validate_completed_run(run_path)
    plan = artifact_plan.artifact_plan_from_completed(completed, run_dir=run_path)
    analysis_root = common.paths.resolve_analysis_root(run_path).resolve()
    strides: dict[ArtifactSplit, int | None] = {
        role.split: _artifact_target_stride(root, analysis_root=analysis_root, split=role.split, dataset_name=role.dataset_name)
        for role in plan.roles
    }
    matched: list[tuple[ArtifactSplit, int]] = [(split, stride) for split, stride in strides.items() if stride is not None]
    if not matched:
        msg = "Artifact upload target does not match a saved Evaluation-grid role."
        raise ArtifactCacheError(msg)
    split, stride = matched[0]
    try:
        loaded = artifact_loader.admit_role_artifact(
            run_path,
            completed,
            selection_role=artifact_plan.dataset_role_of(split),
            root=root,
            evaluation_spatial_stride=stride,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        msg = f"Artifact upload source is incompatible: {type(error).__name__}: {error}"
        raise ArtifactCacheError(msg) from error
    return cast("Mapping[str, Any]", loaded.frame.attrs["artifact_provenance"])


def run_or_load_artifacts(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    rebuild: bool = False,
    evaluation_spatial_stride: int = 1,
) -> pd.DataFrame:
    """
    Validate, reuse, or atomically generate one split-qualified artifact cache.

    Parameters
    ----------
    run_dir : pathlib.Path
        Current evaluable run with immutable config/checkpoint/split identity.
    dataset_name : str
        Logical dataset required by the saved split metadata.
    split : {"eval", "material_ood", "parameter_ood"}
        Saved membership role. ``eval`` publishes under ``analysis/id``, an OOD role under ``analysis/<role>/<dataset>``.
    device_resolution : DeviceResolution
        Device decision resolved once before any inference allocation.
    dataset_root : pathlib.Path
        Final training-dataset root.
    rebuild : bool, optional
        Replace only the observed target. A concurrent newer publication wins and
        is validated instead of being deleted.
    evaluation_spatial_stride : int, optional
        Transient artifact Evaluation-grid stride. The default is one.

    Returns
    -------
    pandas.DataFrame
        The role frame admitted by the shared role validator, as
        ``load_run_artifacts`` returns it, with validated provenance attrs.

    Raises
    ------
    ArtifactCacheError
        If existing schema, identity, aggregate, or payload evidence is invalid.
    FileNotFoundError
        If required evaluable-run evidence or a final-dataset artifact is absent.
    TypeError, ValueError, RuntimeError
        If request semantics, saved membership, device resolution, or generated
        payloads violate the current contract.

    Notes
    -----
    The shared run reader lease excludes lifecycle writers; one run-wide artifact
    lock serializes the builds of one run, and the target-specific lock marks the
    target's active writer for run inspection. Rebuilds generate and validate a sibling stage before replacing the target.
    Provenance remains the final completion marker. Without ``rebuild``, a target
    is not generated beside the run's other published role when that role
    records other analysis code, so one run never gains a second producer.

    """
    logical_dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    resolved_evaluation_stride = transient_contract.validate_spatial_stride(evaluation_spatial_stride)
    save_root = _artifact_save_root(
        run_dir=run_dir,
        dataset_name=logical_dataset_name,
        split=split,
        evaluation_spatial_stride=resolved_evaluation_stride,
    )
    completion_path = contracts.artifact_provenance_path(save_root)
    observed_completion = _completion_marker_identity(completion_path) if rebuild else None
    lock_path = _artifact_lock_path(run_dir=run_dir, save_root=save_root)
    with (
        experiments.run.run_reader_lease(run_dir),
        # Builds of one run are serialized, so the sibling producer check still holds when this target is published.
        common.locking.exclusive_file_lock(common.paths.resolve_artifact_lock_path(common.paths.resolve_analysis_root(run_dir)), blocking=True),
        common.locking.exclusive_file_lock(lock_path, blocking=True),
    ):
        current_completion = _completion_marker_identity(completion_path)
        effective_rebuild = rebuild and (current_completion is None or current_completion == observed_completion)
        return _run_or_load_artifacts_locked(
            run_dir=run_dir,
            dataset_name=logical_dataset_name,
            split=split,
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            rebuild=effective_rebuild,
            evaluation_spatial_stride=resolved_evaluation_stride,
        )


def _validated_upload_roots(
    *,
    plan: artifact_plan.ArtifactPlan,
    id_frame: pd.DataFrame,
    ood_frames: Mapping[ArtifactSplit, pd.DataFrame],
    evaluation_spatial_stride: int,
) -> dict[ArtifactSplit, Path]:
    """Validate every published role and bind its frame before any observer starts."""
    if set(plan.ood_roles) != set(ood_frames):
        msg = "W&B publication requires matching saved OOD Evaluation identity and frame availability."
        raise ArtifactCacheError(msg)
    role_specs: list[tuple[ArtifactSplit, str, pd.DataFrame]] = [("eval", plan.id_role.dataset_name, id_frame)]
    role_specs.extend((split, role.dataset_name, ood_frames[split]) for split, role in plan.ood_roles.items())
    roots: dict[ArtifactSplit, Path] = {}
    for split, dataset_name, frame in role_specs:
        root = _artifact_save_root(
            run_dir=plan.run_dir,
            dataset_name=dataset_name,
            split=split,
            evaluation_spatial_stride=evaluation_spatial_stride,
        )
        validated = validate_artifact_upload_source(run_dir=plan.run_dir, artifact_root=root)
        frame_provenance = frame.attrs.get("artifact_provenance")
        if not isinstance(frame_provenance, Mapping) or common.serialization.canonical_json_sha256(
            frame_provenance
        ) != common.serialization.canonical_json_sha256(validated):
            msg = f"The {split} frame provenance is not the validated upload source."
            raise ArtifactCacheError(msg)
        roots[split] = root
    return roots


def _transient_wandb_summary(frames: Mapping[str, pd.DataFrame]) -> dict[str, bool | float | int | str]:
    """Reduce validated transient role frames to the bounded W&B summary."""
    from src.analysis.evaluation import transient_session  # noqa: PLC0415

    evaluation = transient_session.TransientEvaluationSession(frames)
    try:
        return evaluation.wandb_publication_summary()
    finally:
        evaluation.close()


def _steady_wandb_summary(frames: Mapping[str, pd.DataFrame]) -> dict[str, bool | float | int | str]:
    """
    Reduce validated steady role frames to the bounded W&B summary structure both tasks share.

    Per role it carries case validity and the objective's finite-support counts; the objective and
    per-field RMSE appear only when available, so a missing value never reads as zero. Identity and
    timing keys follow the transient publication vocabulary.
    """
    provenances = {role: frame.attrs["artifact_provenance"] for role, frame in frames.items()}
    reports = [provenance[timing.TIMING_KEY]["report"] for provenance in provenances.values()]
    run = provenances["id"]["run"]
    forward = [case["batch_amortized_neural_operator_forward_s"] for report in reports for case in report["cases"]]
    component = "evaluation/timing/component/batch_amortized_neural_operator_forward_seconds"
    values: dict[str, bool | float | int | str] = {
        "evaluation/identity/checkpoint_sha256": run["best_checkpoint_sha256"],
        "evaluation/identity/checkpoint_epoch": run["best_checkpoint_epoch"],
        "evaluation/identity/model_kind": provenances["id"]["model"]["kind"],
        "evaluation/identity/dataset_sha256": common.serialization.canonical_json_sha256({role: p["dataset"] for role, p in provenances.items()}),
        "evaluation/timing/device": "|".join(sorted({report["neural_runtime"]["resolved_device"] for report in reports})),
        "evaluation/timing/precision": "|".join(sorted({report["neural_runtime"]["inference_dtype"] for report in reports})),
        "evaluation/timing/warmup_passes": reports[0]["measurement"]["warmup_passes"],
        "evaluation/timing/hardware_identity_sha256": common.serialization.canonical_json_sha256([report["neural_runtime"] for report in reports]),
        f"{component}/available_count": len(forward),
        f"{component}/median_seconds": float(np.median(forward)),
    }
    for role, provenance in provenances.items():
        aggregate, counts = provenance["aggregate"], provenance["evaluation"]["prediction_validity"]["status_counts"]
        values[f"evaluation/identity/dataset/{role}/name"] = provenance["dataset"]["name"]
        values[f"evaluation/{role}/valid_case_count"] = counts[validity_contract.VALID]
        values[f"evaluation/{role}/nonfinite_case_count"] = counts[validity_contract.NONFINITE]
        values.update({f"evaluation/{role}/{name}": aggregate[name] for name in validity_contract.SUPPORT_COUNT_FIELDS})
        if aggregate["available"]:
            values[f"evaluation/{role}/normalized_group_macro_rmse"] = aggregate["value"]
            for field, statistics in aggregate["field_statistics"].items():
                values[f"evaluation/{role}/normalized_rmse_{field}"] = statistics["normalized_rmse"]
                values[f"evaluation/{role}/physical_rmse_{field}"] = statistics["physical_rmse"]
    return values


def _upload_published_artifacts(
    *,
    plan: artifact_plan.ArtifactPlan,
    device_resolution: DeviceResolution,
    id_frame: pd.DataFrame,
    ood_frames: Mapping[ArtifactSplit, pd.DataFrame] | None = None,
    evaluation_spatial_stride: int = 1,
) -> None:
    """Publish both tasks' provenance files and bounded Evaluation summary, then steady curated media."""
    config = _load_run_config(plan.run_dir)
    settings = config.get("tracking", {}).get("wandb")
    if not isinstance(settings, Mapping):
        msg = "Completed run config must contain tracking.wandb."
        raise TypeError(msg)
    upload = settings.get("upload")
    if settings.get("mode") == "disabled" or not isinstance(upload, Mapping) or not bool(upload.get("evaluation_artifacts")):
        return

    task = experiments.config.loader.validate_resolved_task_contract(config)
    ood_frames = dict(ood_frames or {})
    artifact_roots = _validated_upload_roots(
        plan=plan,
        id_frame=id_frame,
        ood_frames=ood_frames,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
    is_transient = task.id == "transient_drying"
    frames = {"id": id_frame} | ood_frames
    evidence = (_transient_wandb_summary if is_transient else _steady_wandb_summary)(frames)

    started_at = datetime.now(UTC)
    runtime_session_id = uuid4().hex
    experiments.run.append_runtime_session(
        plan.run_dir,
        device_resolution,
        started_at=started_at,
        session_id=runtime_session_id,
        tracking_state=experiments.run.initial_tracking_state(config),
    )
    summary = experiments.run.read_run_summary(plan.run_dir)
    persisted_run_id, last_logged_epoch = experiments.tracking.persisted_wandb_identity(summary)

    def state_updater(updates: Mapping[str, Any]) -> None:
        """Persist observer-only facts in this artifact-upload runtime session."""
        experiments.run.update_runtime_session(
            plan.run_dir,
            runtime_session_id,
            updates,
        )

    session = experiments.tracking.initialize_wandb(
        config,
        run_dir=plan.run_dir,
        semantic_config={},
        resume=True,
        persisted_run_id=persisted_run_id,
        previous_last_logged_epoch=last_logged_epoch,
        previous_evaluation_summary_keys=experiments.tracking.persisted_evaluation_summary_keys(summary),
        state_updater=state_updater,
        ood_roles=tuple(plan.ood_roles),
    )
    upload_error: BaseException | None = None
    try:
        for artifact_root in artifact_roots.values():
            session.upload_files({"artifact_provenance": contracts.artifact_provenance_path(artifact_root)})
        session.log_evaluation_summary(evidence)
        if not is_transient:
            from src.analysis.presentation import analysis_display_labels as display_labels  # noqa: PLC0415
            from src.analysis.presentation import curated  # noqa: PLC0415

            # Admitted role frames are already the evaluation frames the loader returns.
            datasets_eval = {
                f"{plan.run_dir.name} ID": id_frame,
                **{f"{plan.run_dir.name} {display_labels.evaluation_role_display_label(role)}": frame for role, frame in ood_frames.items()},
            }
            with tempfile.TemporaryDirectory(prefix="grainlegumes-curated-analysis-") as temporary_directory:
                bundle = curated.render_curated_analysis(
                    datasets=datasets_eval,
                    output_dir=temporary_directory,
                )
                session.upload_post_artifact(
                    artifact_root=artifact_roots["eval"],
                    media_files=bundle.media_files,
                    tables=bundle.tables,
                )
    except BaseException as error:
        upload_error = error
        raise
    finally:
        local_summary: Mapping[str, Any] | None = None
        with suppress(Exception):
            local_summary = experiments.run.read_run_summary(plan.run_dir)
        status = str(local_summary.get("status", "completed")) if local_summary is not None else "completed"
        result = None
        if local_summary is not None:
            result = {
                "completed_epoch": local_summary.get("completed_epoch"),
                "global_step": local_summary.get("global_step"),
                "selected_epoch": local_summary.get("selected_epoch"),
                "selected_metrics": local_summary.get("selected_metrics", {}),
                "terminal_epoch": local_summary.get("terminal_epoch"),
                "terminal_metrics": local_summary.get("terminal_metrics", {}),
            }
        try:
            session.finish(
                status=status,
                result=result,
                local_summary=local_summary,
                error=upload_error,
            )
        except experiments.tracking.TrackingError:
            if upload_error is None:
                raise


@dataclass(frozen=True)
class ScopedTransientArtifact:
    """Report one validated noncanonical transient debug artifact."""

    root: Path
    split: ArtifactDatasetRole
    dataset_name: str
    case_ids: tuple[str, ...]
    identity_sha256: str


@dataclass(frozen=True)
class PreparedRunArtifacts:
    """
    Report path-based artifact loading and any explicit local generation.

    Parameters
    ----------
    loaded_run : LoadedRunArtifacts
        Validated run and the selected role-local artifacts.
    role_actions : dict[str, str]
        Selected role to ``reused``, ``generated``, or ``rebuilt`` action.
    artifact_device : str | None
        Exact concrete device supplied to or resolved for generation. ``None``
        means every role was reused without a supplied device decision.

    """

    loaded_run: LoadedRunArtifacts
    role_actions: dict[str, str]
    artifact_device: str | None


def invalid_prediction_note(*runs: PreparedRunArtifacts) -> str | None:
    """Return ``valid V of N`` over the selected roles of ``runs`` when any prediction is invalid, else ``None``."""
    return validity_contract.invalid_prediction_note(
        artifact.frame.attrs["artifact_provenance"] for run in runs for artifact in run.loaded_run.role_artifacts().values()
    )


def load_or_build_run_artifacts(
    run_dir: Path | str,
    *,
    artifact_roles: tuple[ArtifactDatasetRole, ...] | None = None,
    artifact_root_overrides: Mapping[ArtifactDatasetRole, Path | str] | None = None,
    dataset_root: Path | str | None = None,
    auto_build_missing: bool = True,
    rebuild_incompatible: bool = False,
    device_policy: str = "cpu",
    device_resolution: DeviceResolution | None = None,
) -> PreparedRunArtifacts:
    """
    Load selected valid artifacts or generate only missing roles locally.

    Existing valid roles are admitted through the read-only path loader before
    any model reconstruction. Missing roles use ``device_resolution`` unchanged
    when supplied; otherwise the explicit device policy is resolved once. Partial,
    corrupt, stale, or incompatible roles fail unless ``rebuild_incompatible``
    deliberately authorizes exact-target replacement. Existing roles are read as
    historical evidence, but the selected roles of one run must share one analysis
    producer: when they disagree, or when a missing role would be generated beside
    a role of other analysis code, that role is rebuilt first under
    ``rebuild_incompatible`` and the preparation is refused before any build
    otherwise. This service
    never uploads to W&B or changes lifecycle status.
    """
    from src import learning  # noqa: PLC0415
    from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader  # noqa: PLC0415

    path = Path(run_dir).expanduser().resolve()
    if not auto_build_missing and not rebuild_incompatible:
        loaded = artifact_loader.load_run_artifacts(
            path,
            artifact_roles=artifact_roles,
            artifact_root_overrides=artifact_root_overrides,
        )
        overrides = {} if artifact_root_overrides is None else dict(artifact_root_overrides)
        return PreparedRunArtifacts(
            loaded_run=loaded,
            role_actions={role: "reused_scoped_partial" if role in overrides else "reused" for role in loaded.role_artifacts()},
            artifact_device=None,
        )
    plan = load_run_artifact_plan(path)
    roles: tuple[ArtifactDatasetRole, ...] = artifact_roles if artifact_roles is not None else plan.dataset_roles
    if not roles or len(set(roles)) != len(roles) or set(roles).difference(artifact_plan.ARTIFACT_DATASET_ROLES):
        msg = f"artifact_roles must contain unique values from {list(artifact_plan.ARTIFACT_DATASET_ROLES)}."
        raise ValueError(msg)
    unplanned = [role for role in roles if role not in plan.dataset_roles]
    if unplanned:
        msg = f"Completed run has no saved {unplanned[0]} artifact role."
        raise ValueError(msg)
    overrides = {} if artifact_root_overrides is None else dict(artifact_root_overrides)
    if set(overrides).difference(roles):
        msg = "Artifact root overrides must belong to selected roles."
        raise ValueError(msg)
    if overrides:
        if auto_build_missing or rebuild_incompatible:
            msg = "Scoped artifact overrides require read-only load policy."
            raise ValueError(msg)
        loaded = artifact_loader.load_run_artifacts(
            path,
            artifact_roles=roles,
            artifact_root_overrides=overrides,
        )
        return PreparedRunArtifacts(
            loaded_run=loaded,
            role_actions={role: ("reused_scoped_partial" if role in overrides else "reused") for role in roles},
            artifact_device=None,
        )
    current_dataset_root = Path(dataset_root) if dataset_root is not None else common.paths.get_datasets_root()
    actions: dict[str, str] = {}
    if device_resolution is not None and not isinstance(device_resolution, learning.device.DeviceResolution):
        msg = f"device_resolution must be a DeviceResolution, got {device_resolution!r}."
        raise TypeError(msg)
    resolution = device_resolution

    for _attempt in range(len(roles) + 1):
        selected_role: ArtifactDatasetRole | None
        try:
            loaded = artifact_loader.load_run_artifacts(path, artifact_roles=roles)
        except artifact_loader.MissingEvaluationArtifactsError as error:
            selected_role = _failed_role(error)
            if not auto_build_missing or selected_role is None or selected_role not in roles:
                raise
            rebuild = False
            action = "generated"
            older = _older_sibling_to_rebuild(
                path=path, plan=plan, split=artifact_plan.split_of(selected_role), roles=roles, rebuild_incompatible=rebuild_incompatible
            )
            if older is not None:
                selected_role, rebuild, action = older, True, "rebuilt"
        except artifact_loader.IncompatibleEvaluationArtifactsError as error:
            selected_role = _failed_role(error)
            if not rebuild_incompatible or selected_role not in roles:
                raise
            rebuild = True
            action = "rebuilt"
        else:
            # Existing roles are historical evidence; building beside them must not leave one run with two analysis producers.
            selected_role = _role_from_other_analysis(loaded, plan.task)
            if selected_role is None:
                for role in roles:
                    actions.setdefault(role, "reused")
                return PreparedRunArtifacts(
                    loaded_run=loaded,
                    role_actions=actions,
                    artifact_device=(str(resolution.device) if resolution is not None else None),
                )
            if not rebuild_incompatible:
                raise _two_producers_error(path, selected_role)
            rebuild = True
            action = "rebuilt"

        if selected_role is None:
            msg = "Artifact load failure did not identify a selected role."
            raise RuntimeError(msg)
        if resolution is None:
            resolution = learning.device.resolve_device(device_policy, path="device_policy")
        run_or_load_artifacts(
            run_dir=path,
            dataset_name=plan.role(artifact_plan.split_of(selected_role)).dataset_name,
            split=artifact_plan.split_of(selected_role),
            device_resolution=resolution,
            dataset_root=current_dataset_root,
            rebuild=rebuild,
        )
        actions[selected_role] = action
    msg = f"Artifact preparation did not converge for selected roles {roles}: {path}"
    raise RuntimeError(msg)


def _role_from_other_analysis(loaded: LoadedRunArtifacts, task_id: str) -> ArtifactDatasetRole | None:
    """Return a loaded role not produced by the running analysis code when the loaded roles disagree, otherwise None."""
    recorded: dict[ArtifactDatasetRole, str] = {
        role: contracts.validate_analysis_producer(artifact.frame.attrs["artifact_provenance"].get("producer"))["analysis_sha256"]
        for role, artifact in loaded.role_artifacts().items()
    }
    if len(set(recorded.values())) <= 1:
        return None
    current = contracts.ANALYSIS_PROJECTIONS[task_id].sha256()
    return next(role for role, digest in recorded.items() if digest != current)


def _failed_role(error: Exception) -> ArtifactDatasetRole | None:
    """Return the Dataset role a role-local artifact load failure names, if it names one."""
    split_role = getattr(error, "role", None)
    return artifact_plan.dataset_role_of(split_role) if split_role in artifact_plan.ARTIFACT_SPLITS else None


def _sibling_from_other_analysis(
    *,
    run_dir: Path,
    plan: artifact_plan.ArtifactPlan,
    task_id: str,
    split: ArtifactSplit,
    evaluation_spatial_stride: int = 1,
) -> ArtifactSplit | None:
    """Return another published split of the run whose recorded analysis implementation is not the running one, otherwise None."""
    current = contracts.ANALYSIS_PROJECTIONS[task_id].sha256()
    for sibling in (role for role in plan.roles if role.split != split):
        root = _artifact_save_root(
            run_dir=run_dir, dataset_name=sibling.dataset_name, split=sibling.split, evaluation_spatial_stride=evaluation_spatial_stride
        )
        marker = contracts.artifact_provenance_path(root)
        if not marker.is_file():
            continue
        recorded = _recorded_marker(marker)
        producer = recorded.get("producer") if isinstance(recorded, Mapping) else None
        if (producer.get("analysis_sha256") if isinstance(producer, Mapping) else None) != current:
            return sibling.split
    return None


def _older_sibling_to_rebuild(
    *,
    path: Path,
    plan: artifact_plan.ArtifactPlan,
    split: ArtifactSplit,
    roles: tuple[ArtifactDatasetRole, ...],
    rebuild_incompatible: bool,
) -> ArtifactDatasetRole | None:
    """Return the selected role of other analysis code to rebuild before a missing split is generated beside it; refuse when not authorized."""
    sibling = _sibling_from_other_analysis(run_dir=path, plan=plan, task_id=plan.task, split=split)
    if sibling is None:
        return None
    role = artifact_plan.dataset_role_of(sibling)
    if not rebuild_incompatible or role not in roles:
        raise _two_producers_error(path, role)
    return role


def _two_producers_error(path: Path, role: str) -> ArtifactCacheError:
    """Return the refusal of a preparation that would leave one run with two analysis producers."""
    return ArtifactCacheError(
        f"Preparing the artifacts of {path} would leave the run with two analysis producers; rebuild its {role} role with --rebuild."
    )


def _publish_scoped_transient_stage(*, target: Path, stage: Path) -> None:
    """Publish one validated scoped stage without overwriting a concurrent target."""
    if target.exists() or target.is_symlink():
        message = f"Scoped artifact output appeared concurrently: {target}"
        raise FileExistsError(message)
    stage.replace(target)


def _require_staged_selection(stage: Path, selected_role: artifact_plan.ArtifactRolePlan) -> None:
    """Refuse a complete scoped stage of an earlier attempt whose persisted selection or analysis code differs from this request."""
    document = json.loads(contracts.artifact_provenance_path(stage).read_text(encoding="utf-8"))
    evaluation = document.get("evaluation") if isinstance(document, dict) else None
    scope = evaluation.get("scope") if isinstance(evaluation, dict) else None
    staged = scope.get("selected_case_ids") if isinstance(scope, dict) else None
    requested = [case_id for source_case_ids in selected_role.case_ids_by_source for case_id in source_case_ids]
    remedy = "delete the stale stage and rerun (bounded generation takes no --rebuild)."
    if staged != requested:
        msg = f"Scoped artifact stage {stage} holds cases {staged}, but this request selects {requested}; {remedy}"
        raise ValueError(msg)
    # The complete stage would be published as this request's result, so it must come from the running analysis code.
    producer = document.get("producer") if isinstance(document, dict) else None
    common.code_identity.require_current_implementation(
        contracts.ANALYSIS_PROJECTIONS["transient_drying"],
        producer.get("analysis_sha256") if isinstance(producer, Mapping) else None,
        unit=f"Scoped artifact stage {stage}",
        remedy=remedy,
    )


def build_scoped_transient_artifact(
    *,
    run_dir: Path | str,
    dataset_root: Path | str,
    output_root: Path | str,
    split: ArtifactDatasetRole,
    case_ids: Sequence[str] | None = None,
    one_case: bool = False,
    device_policy: str = "auto",
    evaluation_spatial_stride: int = 1,
) -> ScopedTransientArtifact:
    """Generate selected transient cases into one explicit noncanonical root."""
    from src import learning  # noqa: PLC0415
    from src.analysis.evaluation import (  # noqa: PLC0415
        evaluation_transient_artifact as sequence_artifact,
    )

    resolved_evaluation_stride = transient_contract.validate_spatial_stride(evaluation_spatial_stride)
    if split not in artifact_plan.ARTIFACT_DATASET_ROLES:
        msg = f"Scoped transient split must be one of {list(artifact_plan.ARTIFACT_DATASET_ROLES)}."
        raise ValueError(msg)
    if not isinstance(one_case, bool):
        msg = "one_case must be boolean."
        raise TypeError(msg)
    requested = () if case_ids is None else tuple(case_ids)
    if one_case and requested:
        msg = "Select --one-case or explicit case IDs, not both."
        raise ValueError(msg)
    if not one_case and not requested:
        msg = "Scoped transient generation requires --one-case or case IDs."
        raise ValueError(msg)

    path = Path(run_dir).expanduser().resolve(strict=True)
    _refuse_optuna_trial(path)
    plan = transient.load_transient_artifact_plan(path)
    saved_role = plan.role(artifact_plan.split_of(split))
    selected_ids = saved_role.case_ids[:1] if one_case else requested
    selected_role = transient.select_transient_role_cases(
        saved_role,
        selected_ids,
    )

    target = _normalise_path(Path(output_root))
    if target == Path(target.anchor) or path.is_relative_to(target) or target.is_relative_to(path):
        msg = f"Scoped artifact output must be one narrow directory outside the run: {target}"
        raise ValueError(msg)
    canonical_analysis = _normalise_path(common.paths.resolve_analysis_root(path))
    if target.is_relative_to(canonical_analysis):
        msg = "Scoped artifacts cannot be written below the canonical run analysis root."
        raise ValueError(msg)
    if target.exists() or target.is_symlink():
        msg = f"Scoped artifact output already exists; refusing to overwrite it: {target}"
        raise FileExistsError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or target.parent.resolve() != target.parent:
        msg = f"Scoped artifact output parent must be one exact regular directory: {target.parent}"
        raise ValueError(msg)

    resolution = learning.device.resolve_device(
        device_policy,
        path="device_policy",
    )
    completed = experiments.run.validate_completed_run(path)
    reporter = _progress_reporter(
        completed=completed,
        task="transient_drying",
        total_cases=len(selected_ids),
        split=split,
        device_resolution=resolution,
        output_root=target,
    )
    stage = _artifact_stage_root(target)
    loaded: sequence_artifact.TransientSequenceArtifactIndex | None = None
    try:
        with experiments.run.run_reader_lease(path):
            if not contracts.artifact_provenance_path(stage).is_file():
                transient.generate_transient_role_artifact(
                    run_dir=path,
                    role=selected_role,
                    saved_role=saved_role,
                    device_resolution=resolution,
                    dataset_root=dataset_root,
                    staging_root=stage,
                    completed=completed,
                    evaluation_spatial_stride=resolved_evaluation_stride,
                    progress_reporter=reporter,
                )
            else:
                _require_staged_selection(stage, selected_role)
            timing.write_stage_timing(stage, {"operational_performance": reporter.final_snapshot()})
            with reporter.phase("validation"):
                sequence_artifact.cleanup_transient_resume_metadata(stage)
                transient.validate_scoped_transient_role_artifact_index(
                    stage,
                    completed=completed,
                    saved_role=saved_role,
                    evaluation_spatial_stride=resolved_evaluation_stride,
                )
        lock_path = target.parent / f".{target.name}.scoped.lock"
        with (
            reporter.phase("publication"),
            common.locking.exclusive_file_lock(lock_path, blocking=True),
        ):
            _publish_scoped_transient_stage(target=target, stage=stage)
            loaded = transient.validate_scoped_transient_role_artifact_index(
                target,
                completed=completed,
                saved_role=saved_role,
                evaluation_spatial_stride=resolved_evaluation_stride,
            )
        reporter.done()
    except BaseException:
        reporter.failure()
        raise
    finally:
        cleanup_runtime(resolution.device)
    if loaded is None:
        msg = "Scoped transient artifact validation did not complete."
        raise RuntimeError(msg)
    return ScopedTransientArtifact(
        root=target,
        split=split,
        dataset_name=selected_role.dataset_name,
        case_ids=tuple(case_id for source_case_ids in selected_role.case_ids_by_source for case_id in source_case_ids),
        identity_sha256=loaded.identity_sha256,
    )


def build_artifacts(
    *,
    runs_root: Path,
    dataset_root: Path,
    run_names: Iterable[str] | None = None,
    device_policy: str = "auto",
    rebuild: bool = False,
    evaluation_spatial_stride: int = 1,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build or validate saved ID/OOD roles keyed by each exact resolved run path.

    Parameters
    ----------
    runs_root : Path
        One evaluable run or a container of run directories.
    dataset_root : Path
        Derived raw root containing immutable final task datasets.
    run_names : Iterable[str] | None, optional
        Persisted scientific run names below ``runs_root``.
    device_policy : {"auto", "cuda", "cpu"}, optional
        Runtime policy. Auto selects usable CUDA and then CPU. CUDA is strict.
        CPU avoids CUDA queries.
    rebuild : bool, optional
        Stage and validate a replacement for each exact selected target. A newer
        concurrent publication is preserved and validated instead.
    evaluation_spatial_stride : int, optional
        Transient artifact Evaluation-grid stride. The default one uses the original grid.

    Returns
    -------
    dict[str, dict[str, pandas.DataFrame]]
        Validated ``eval`` and any saved ``material_ood`` and ``parameter_ood`` frames keyed by run name.

    Raises
    ------
    FileNotFoundError, ArtifactCacheError, TypeError, ValueError, RuntimeError
        If device resolution, run admission, saved dataset identity, cache
        validation, inference, or publication fails.

    Notes
    -----
    Each saved cache role is locally authoritative. Persisted W&B settings are
    the sole publication opt-in. Both tasks validate every published role before
    observer initialization and upload its provenance file; steady runs add curated
    media, transient runs one bounded aggregate scalar/string summary. Any requested
    online or offline observer failure propagates.

    """
    from src import learning  # noqa: PLC0415

    resolved_evaluation_stride = transient_contract.validate_spatial_stride(evaluation_spatial_stride)
    device_resolution = learning.device.resolve_device(
        device_policy,
        path="device_policy",
    )
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for run_dir in iter_run_dirs(runs_root, run_names=run_names):
        plan = load_run_artifact_plan(run_dir)
        common.paths.record_job_log_owner(plan.run_dir)
        id_frame = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.id_role.dataset_name,
            split="eval",
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            rebuild=rebuild,
            evaluation_spatial_stride=resolved_evaluation_stride,
        )
        ood_frames: dict[ArtifactSplit, pd.DataFrame] = {
            split: run_or_load_artifacts(
                run_dir=plan.run_dir,
                dataset_name=role.dataset_name,
                split=split,
                device_resolution=device_resolution,
                dataset_root=dataset_root,
                rebuild=rebuild,
                evaluation_spatial_stride=resolved_evaluation_stride,
            )
            for split, role in plan.ood_roles.items()
        }
        results[str(run_dir.resolve())] = {"eval": id_frame} | ood_frames
        _upload_published_artifacts(
            plan=plan,
            device_resolution=device_resolution,
            id_frame=id_frame,
            ood_frames=ood_frames,
            evaluation_spatial_stride=resolved_evaluation_stride,
        )
        cleanup_runtime(device_resolution.device)
    return results
