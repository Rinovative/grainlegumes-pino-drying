"""Load strict path-selected evaluation artifacts without generation or inference.

Responsibilities:
  - Admit one explicit current run directory independently of its storage alias
  - Derive exact ID and OOD roots from authoritative saved split identity
  - Validate artifact schemas, payload digests, provenance, membership, and the required timing block
  - Read artifact roots only inside the admitting path, since DataFrame provenance does not prove run, timing, or role
  - Serve as the one role validator for private stages, cache hits, and reads of both tasks
  - Return comparison-ready DataFrames with the same role attrs and stable canonical identity digests
  - Report actionable host commands for missing or incompatible artifacts

Design principles:
  - Scientific run and saved-split evidence remain the source of artifact identity
  - ID and OOD roles are explicit and cannot substitute for one another
  - Artifact admission is read-only, exact, and fail-closed
  - Presentation labels and operational runtime facts remain outside scientific identity

This module does NOT:
  - Import artifact generation, reconstruct models, or execute inference
  - Create, repair, rebuild, publish, upload, or remove artifact payloads
  - Open steady NPZ arrays; the writer checks their contents before the stage is admitted
  - Render figures or manage evaluation-session caches

Notes:
  Role admission:
    - Steady ID and OOD roles come from the saved split contract and are admitted independently under their exact run-owned roots.
    - Transient drying uses a separate sequence-index path for ID, material-OOD and parameter-OOD roles; explicit scoped roots remain marked partial.
  Timing:
    - Both tasks carry a required top-level timing block outside scientific identity. A missing or contradictory block refuses the role.

"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from src import common, datasets, domain, experiments
from src.analysis.artifacts import contracts
from src.analysis.artifacts import plan as artifact_plan
from src.analysis.artifacts import timing as artifact_timing

from . import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec

ArtifactRole = artifact_plan.ArtifactSplit
ArtifactSelectionRole = artifact_plan.ArtifactDatasetRole

__all__ = [
    "ArtifactRole",
    "ArtifactSelectionRole",
    "EvaluationArtifactLoadError",
    "IncompatibleEvaluationArtifactsError",
    "LoadedEvaluationArtifact",
    "LoadedRunArtifacts",
    "MissingEvaluationArtifactsError",
    "admit_role_artifact",
    "load_run_artifacts",
]


class EvaluationArtifactLoadError(RuntimeError):
    """
    Represent failure to admit run-owned evaluation artifacts.

    Parameters
    ----------
    message : str
        Human-readable admission failure and recovery guidance.
    role : {"eval", "material_ood", "parameter_ood"} | None
        Rejected artifact role when the failure is role-local.
    run_dir : pathlib.Path | None
        Exact current run directory when it is known.

    Notes
    -----
    Missing and incompatible targets use distinct subclasses so notebook callers
    can present the appropriate host-side recovery command without inspecting
    exception text.

    """

    def __init__(self, message: str, *, role: ArtifactRole | None = None, run_dir: Path | None = None) -> None:
        """Initialize the message and structured recovery context."""
        super().__init__(message)
        self.role = role
        self.run_dir = run_dir


class MissingEvaluationArtifactsError(EvaluationArtifactLoadError):
    """
    Represent absence of a required ID or OOD artifact target.

    Notes
    -----
    This error is reserved for an absent or empty target. Partial targets are
    incompatible because normal artifact generation must not trust them.

    """


class IncompatibleEvaluationArtifactsError(EvaluationArtifactLoadError):
    """
    Represent a stale, partial, changed, or contradictory artifact target.

    Notes
    -----
    The associated message identifies the rejected role and supplies the exact
    host-side rebuild command for the selected task and canonical run name.

    """


@dataclass(frozen=True, slots=True)
class LoadedEvaluationArtifact:
    """
    Store one validated role-local artifact owned by an evaluable run.

    Parameters
    ----------
    split_role : {"eval", "material_ood", "parameter_ood"}
        Exact saved split role represented by the artifact.
    dataset_name : str
        Canonical saved dataset identity for the role.
    root : pathlib.Path
        Exact resolved run-owned artifact root.
    frame : pandas.DataFrame
        Comparison-ready table with complete validated provenance attributes.
    identity_sha256 : str
        Canonical SHA-256 digest of the complete validated provenance document.

    """

    split_role: ArtifactRole
    dataset_name: str
    root: Path
    frame: pd.DataFrame
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedRunArtifacts:
    """
    Store validated ID and OOD artifacts for one scientifically identified run.

    Parameters
    ----------
    task : str
        Canonical registered task identifier.
    run_name : str
        Immutable scientific run name used for persisted identity.
    run_dir : pathlib.Path
        Exact resolved current run directory.
    id_artifact : LoadedEvaluationArtifact
        Artifact bound to saved ID-test membership and ``analysis/id``.
    material_ood_artifact, parameter_ood_artifact : LoadedEvaluationArtifact
        Artifacts bound to saved material-OOD and parameter-OOD membership and their named roots.

    """

    task: str
    run_name: str
    run_dir: Path
    id_artifact: LoadedEvaluationArtifact | None
    material_ood_artifact: LoadedEvaluationArtifact | None = None
    parameter_ood_artifact: LoadedEvaluationArtifact | None = None
    selected_checkpoint_role: str = "best"
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_sha256: str = ""
    summary: Mapping[str, Any] | None = None

    @property
    def scientific_run_name(self) -> str:
        """Return the immutable scientific identity saved in config.yaml."""
        return self.run_name

    @property
    def storage_alias(self) -> str:
        """Return the mutable current directory leaf used for storage."""
        return self.run_dir.name

    def artifact(self, role: ArtifactSelectionRole) -> LoadedEvaluationArtifact | None:
        """Return the loaded artifact of one Dataset role, ``None`` when it was not selected or does not exist."""
        return cast("LoadedEvaluationArtifact | None", getattr(self, f"{role}_artifact"))

    def role_artifacts(self) -> dict[ArtifactSelectionRole, LoadedEvaluationArtifact]:
        """Return every loaded artifact keyed by its Dataset role, in canonical role order."""
        return {role: artifact for role in artifact_plan.ARTIFACT_DATASET_ROLES if (artifact := self.artifact(role)) is not None}


@dataclass(frozen=True, slots=True)
class _SavedRole:
    """
    Store authoritative saved-run evidence for one artifact role.

    Parameters
    ----------
    plan : ArtifactRolePlan
        Saved role with every source Dataset identity and ordered membership, in split order.
    root : pathlib.Path
        Exact resolved run-owned artifact root for the role.

    """

    plan: artifact_plan.ArtifactRolePlan
    root: Path

    @property
    def split_role(self) -> ArtifactRole:
        """Return the persisted artifact split of the role."""
        return self.plan.split

    @property
    def dataset_name(self) -> str:
        """Return the role's artifact Dataset name."""
        return self.plan.dataset_name

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return the exact ordered saved membership as positions in the concatenated sources."""
        return self.plan.source_indices


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """
    Return one mapping with a path-qualified error.

    Parameters
    ----------
    value : Any
        Candidate mapping value.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    collections.abc.Mapping
        The admitted mapping without copying it.

    Raises
    ------
    TypeError
        If ``value`` is not a mapping.

    """
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _positive_int(value: Any, *, label: str) -> int:
    """
    Return one positive integer while rejecting boolean aliases.

    Parameters
    ----------
    value : Any
        Candidate integral value.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    int
        Exact positive integer value.

    Raises
    ------
    TypeError
        If the value is boolean, non-integral, zero, or negative.

    """
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        msg = f"{label} must be a positive integer."
        raise TypeError(msg)
    return int(value)


def _nonempty_string(value: Any, *, label: str) -> str:
    """
    Return one non-empty string.

    Parameters
    ----------
    value : Any
        Candidate string value.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    str
        The admitted non-empty string.

    Raises
    ------
    TypeError
        If the value is not non-empty text.

    """
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string."
        raise TypeError(msg)
    return value


def _artifact_command(*, run_dir: Path, rebuild: bool) -> str:
    """Return the host wrapper command for one exact current run directory."""
    suffix = " --rebuild" if rebuild else ""
    return (
        "./scripts/slurm_ml.sh --mode cpu --partition standard --cpus-per-task 4 "
        "--mem 16G --time 02:00:00 artifacts --run-dir "
        f"{shlex.quote(str(run_dir))}{suffix}"
    )


def _missing(*, task: str, run_name: str, run_dir: Path, role: ArtifactRole, root: Path) -> MissingEvaluationArtifactsError:
    """
    Build an actionable missing-artifact failure.

    Parameters
    ----------
    task, run_name : str
        Canonical completed-run selection.
    run_dir : pathlib.Path
        Exact current run directory used by the recovery command.
    role : {"eval", "material_ood", "parameter_ood"}
        Missing saved split role.
    root : pathlib.Path
        Exact expected artifact root.

    Returns
    -------
    MissingEvaluationArtifactsError
        Failure carrying the generic host-side build command.

    """
    command = _artifact_command(run_dir=run_dir, rebuild=False)
    return MissingEvaluationArtifactsError(
        f"Evaluation artifacts are missing for run {task}/{run_name} ({role} expected at {root}).\nBuild them locally:\n  {command}",
        role=role,
        run_dir=run_dir,
    )


def _incompatible(
    *,
    task: str,
    run_name: str,
    run_dir: Path,
    role: ArtifactRole,
    root: Path,
    reason: str,
) -> IncompatibleEvaluationArtifactsError:
    """
    Build an actionable incompatible-artifact failure.

    Parameters
    ----------
    task, run_name : str
        Canonical completed-run selection.
    run_dir : pathlib.Path
        Exact current run directory used by the recovery command.
    role : {"eval", "material_ood", "parameter_ood"}
        Rejected saved split role.
    root : pathlib.Path
        Exact observed artifact root.
    reason : str
        Path-qualified admission failure.

    Returns
    -------
    IncompatibleEvaluationArtifactsError
        Failure carrying the generic host-side rebuild command.

    """
    command = _artifact_command(run_dir=run_dir, rebuild=True)
    return IncompatibleEvaluationArtifactsError(
        f"Evaluation artifacts are incompatible with run {task}/{run_name} "
        f"({role} at {root}): {reason}\n"
        f"Rebuild only the exact run-owned target explicitly:\n  {command}",
        role=role,
        run_dir=run_dir,
    )


def _saved_roles(
    *,
    completed: Mapping[str, Any],
    run_dir: Path,
    task: str,
    task_spec: TaskSpec,
) -> dict[ArtifactSelectionRole, _SavedRole]:
    """Resolve the admitted ID-test role and every planned OOD role, keyed by Dataset role, from the shared artifact plan."""
    plan = artifact_plan.artifact_plan_from_completed(completed, run_dir=run_dir)
    if plan.task != task or task_spec.id != task:
        msg = "Saved split task identity contradicts the evaluable run task contract."
        raise ValueError(msg)
    return {role.dataset_role: _SavedRole(plan=role, root=role.root(run_dir).absolute()) for role in plan.roles}


def _expected_run_provenance(
    *,
    summary: Mapping[str, Any],
    task_spec: TaskSpec,
    run_name: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Project immutable scientific and lifecycle evaluation evidence."""
    return {
        "name": run_name,
        "task": task_spec.id,
        "task_contract_digest": task_spec.contract_digest,
        "effective_config_digest": summary["effective_config_digest"],
        "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
        "normalizer_sha256": summary["normalizer_sha256"],
        "best_checkpoint_epoch": evaluation["selected_checkpoint_epoch"],
        "selected_checkpoint_role": evaluation["selected_checkpoint_role"],
    }


def _expected_model_provenance(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Project architecture, exact capacity, and physics selection.

    Parameters
    ----------
    config : collections.abc.Mapping
        Validated resolved run configuration.
    summary : collections.abc.Mapping
        Validated completed-run summary carrying parameter counts.

    Returns
    -------
    dict[str, Any]
        Exact model provenance required from the artifact.

    Raises
    ------
    TypeError, ValueError
        If model, physics, or parameter-count evidence is malformed.

    """
    model = _mapping(config.get("model"), label="config.yaml model")
    architecture = _mapping(model.get("params"), label="config.yaml model.params")
    loss = _mapping(config.get("loss"), label="config.yaml loss")
    physics = _mapping(loss.get("physics"), label="config.yaml loss.physics")
    result = {
        "kind": model.get("kind"),
        "architecture": dict(architecture),
        "physics_enabled": physics.get("enabled") is True,
    }
    raw_counts = summary.get("model_parameter_counts")
    if raw_counts is not None:
        counts = _mapping(raw_counts, label="summary model_parameter_counts")
        exact_counts = {
            "total": _positive_int(counts.get("total"), label="summary model_parameter_counts.total"),
            "trainable": _positive_int(counts.get("trainable"), label="summary model_parameter_counts.trainable"),
        }
        if exact_counts["trainable"] > exact_counts["total"]:
            msg = "Run trainable parameter count exceeds total parameter count."
            raise ValueError(msg)
        result["parameter_counts"] = exact_counts
    return result


def _expected_normalizer_provenance(
    *,
    task_spec: TaskSpec,
    normalizer_sha256: Any,
    normalizer_state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Project current saved-normalizer semantics.

    Parameters
    ----------
    task_spec : TaskSpec
        Validated persisted task contract.
    normalizer_sha256 : Any
        Completed-run digest of ``normalizer.pt``.
    normalizer_state : collections.abc.Mapping[str, Any]
        Validated saved normalizer tensors from completed-run admission.

    Returns
    -------
    dict[str, Any]
        Exact normalizer provenance required from the artifact.

    """
    return {
        "sha256": normalizer_sha256,
        "identity": "saved_run_normalizer.pt",
        "fit_split": task_spec.preprocessing.fit_split,
        "output_normalization": task_spec.preprocessing.output_normalization,
        "denominator_floor": 1e-7,
        "output_standard_deviations": contracts.output_standard_deviations_from_state(
            normalizer_state,
            output_fields=task_spec.output_names,
        ),
    }


def _expected_evaluator_provenance(
    *,
    config: Mapping[str, Any],
    task_spec: TaskSpec,
) -> dict[str, Any]:
    """
    Project the task-resolved evaluator contract without runtime workloads.

    Parameters
    ----------
    config : collections.abc.Mapping
        Validated resolved run configuration.
    task_spec : TaskSpec
        Validated persisted task contract.

    Returns
    -------
    dict[str, Any]
        Exact evaluator fields, units, metrics, objective, and formulas.

    Raises
    ------
    TypeError, ValueError
        If evaluation configuration is malformed or inconsistent.

    """
    evaluation = _mapping(config.get("evaluation"), label="config.yaml evaluation")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        msg = "config.yaml evaluation.metrics must be a list."
        raise TypeError(msg)
    return {
        "metrics": metrics,
        "objective": experiments.config.loader.get_resolved_objective(config),
        "input_fields": list(task_spec.input_names),
        "input_units": {field.name: field.unit for field in task_spec.inputs},
        "output_fields": list(task_spec.output_names),
        "output_units": {field.name: field.unit for field in task_spec.outputs},
        "output_groups": contracts.output_group_payload(task_spec.output_groups),
        "physics_kind": task_spec.physics.kind,
        "group_objective_evidence": {
            "squared_error_accumulation_dtype": "float64",
            "per_case_physical_columns": {field: list(contracts.physical_statistic_columns(field)) for field in task_spec.output_names},
            "per_case_normalized_columns": {field: list(contracts.normalized_statistic_columns(field)) for field in task_spec.output_names},
            "train_scale_source": "saved_run_normalizer.output_standard_deviations",
            "dataset_reduction": ("sum physical field SSE/count, finalize shared-scale group RMSE, equal macro mean over output groups"),
        },
        "predictive_metrics": {
            "rel_l2": "per-case arithmetic mean of physical per-field relative L2 ratios",
            "rel_h1": ("per-case arithmetic mean of physical per-field relative H1 ratios on the declared artifact region where available"),
            "physical_rmse_columns": {field: contracts.physical_statistic_columns(field)[2] for field in task_spec.output_names},
        },
    }


def _expected_physics_provenance(task_spec: TaskSpec) -> dict[str, Any] | None:
    """Return the exact current steady-Brinkman diagnostic contract, if any."""
    if task_spec.physics.kind != domain.physics.contracts.STEADY_BRINKMAN_KIND:
        return None
    return contracts.steady_physics_provenance(task_spec)


def _effective_selection(
    role: _SavedRole,
    provenance: Mapping[str, Any],
    *,
    evaluation_batch_size: int,
) -> tuple[int, tuple[int, ...]]:
    """Validate and return the complete saved membership and batching contract."""
    selection = _mapping(provenance.get("selection"), label="artifact provenance selection")
    membership_digest = contracts.ordered_indices_sha256(role.source_indices)
    expected = {
        "index_key": f"{role.plan.split_role}_indices",
        "full_selected_case_count": len(role.source_indices),
        "effective_case_count": len(role.source_indices),
        "generation_limit": None,
        "full_ordered_source_indices_sha256": membership_digest,
        "effective_ordered_source_indices_sha256": membership_digest,
    }
    if dict(selection) != expected:
        msg = "Artifact selection must match the complete saved membership without a generation limit."
        raise ValueError(msg)
    generation = _mapping(provenance.get("generation"), label="artifact provenance generation")
    expected_generation = {
        "effective_case_limit": None,
        "inference_batch_size": evaluation_batch_size,
        "compression": "numpy savez_compressed",
    }
    if dict(generation) != expected_generation:
        msg = "Artifact generation provenance must use complete membership and the saved evaluation batch size."
        raise ValueError(msg)
    return len(role.source_indices), role.source_indices


def _validate_payload_membership(
    *,
    frame: pd.DataFrame,
    provenance: Mapping[str, Any],
    role: _SavedRole,
    effective_indices: tuple[int, ...],
) -> None:
    """
    Bind table rows and manifest NPZ paths to exact saved membership.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema evaluation table.
    provenance : collections.abc.Mapping
        Complete admitted artifact provenance.
    role : _SavedRole
        Authoritative role identity and root.
    effective_indices : tuple[int, ...]
        Exact saved-order membership represented by the artifact.

    Raises
    ------
    TypeError, ValueError
        If row identities, case paths, or output-manifest membership disagree.

    Notes
    -----
    Validation uses persisted scalar identities and file digests. It never opens
    or decompresses the NPZ arrays.

    """
    source_indices = tuple(int(value) for value in frame["source_index"].tolist())
    split_local_indices = tuple(int(value) for value in frame["split_local_index"].tolist())
    case_indices = tuple(int(value) for value in frame["case_index"].tolist())
    if source_indices != effective_indices:
        msg = "Artifact Parquet ordered source_index values do not match saved membership."
        raise ValueError(msg)
    if split_local_indices != tuple(range(len(effective_indices))):
        msg = "Artifact Parquet split_local_index values do not preserve saved membership order."
        raise ValueError(msg)
    expected_case_indices = tuple(source_index + 1 for source_index in effective_indices)
    if case_indices != expected_case_indices:
        msg = "Artifact Parquet case_index values do not match saved source identities."
        raise ValueError(msg)
    if tuple(frame["source_dataset_id"].tolist()) != role.plan.case_source_dataset_ids:
        msg = "Artifact Parquet source_dataset_id values do not name each case's saved source Dataset."
        raise ValueError(msg)

    case_files = contracts.dataset_case_payload_filenames(role.plan.case_ids)
    expected_paths = tuple((role.root / "npz" / name).absolute() for name in case_files)
    raw_paths = frame["npz_path"].tolist()
    if any(not isinstance(value, str) or not value for value in raw_paths):
        msg = "Artifact Parquet npz_path values must be non-empty strings."
        raise TypeError(msg)
    actual_paths = tuple(Path(value).expanduser().absolute() for value in raw_paths)
    if actual_paths != expected_paths:
        msg = "Artifact Parquet NPZ paths do not identify exact run-owned case payloads."
        raise ValueError(msg)
    for expected_path in expected_paths:
        if not expected_path.is_file() or expected_path.is_symlink() or expected_path.resolve() != expected_path:
            msg = f"Artifact NPZ payload is missing or uses a symbolic-link alias: {expected_path}"
            raise ValueError(msg)

    outputs = _mapping(provenance.get("outputs"), label="artifact provenance outputs")
    parquet_output = _mapping(outputs.get("parquet"), label="artifact provenance outputs.parquet")
    if parquet_output.get("path") != common.paths.ANALYSIS_ARTIFACTS_FILENAME:
        msg = f"Artifact manifest Parquet path must be {common.paths.ANALYSIS_ARTIFACTS_FILENAME}."
        raise ValueError(msg)
    npz_outputs = outputs.get("npz")
    if not isinstance(npz_outputs, list):
        msg = "Artifact provenance outputs.npz must be a list."
        raise TypeError(msg)
    manifest_paths = tuple(_mapping(item, label=f"artifact provenance outputs.npz[{index}]").get("path") for index, item in enumerate(npz_outputs))
    expected_manifest_paths = tuple(sorted(f"npz/{name}" for name in case_files))
    if manifest_paths != expected_manifest_paths:
        msg = "Artifact manifest NPZ membership does not match saved source membership."
        raise ValueError(msg)


def _bind_artifact_to_run(
    *,
    frame: pd.DataFrame,
    role: _SavedRole,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
    task_spec: TaskSpec,
    run_name: str,
    evaluation: Mapping[str, Any],
) -> str:
    """
    Bind one admitted artifact to its authoritative evaluable run.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluation table with complete admitted provenance.
    role : _SavedRole
        Authoritative saved role and artifact root.
    config : collections.abc.Mapping
        Validated resolved run configuration.
    summary : collections.abc.Mapping
        Validated completed-run summary.
    normalizer_state : collections.abc.Mapping
        Validated saved training normalizer state.
    task_spec : TaskSpec
        Validated persisted task contract.
    run_name : str
        Canonical completed-run name.
    evaluation : collections.abc.Mapping
        Validated resolved evaluation configuration.

    Returns
    -------
    str
        Canonical SHA-256 digest of complete validated artifact provenance.

    Raises
    ------
    TypeError, ValueError, ComparisonCompatibilityError
        If any schema, run, model, normalizer, evaluator, physics, role,
        dataset, selection, runtime, or payload identity disagrees.

    """
    provenance = dataframe.require_complete_provenance(frame)
    expected_physics = _expected_physics_provenance(task_spec)
    expected_keys = {
        "artifact_kind",
        "producer",
        "timing",
        "run",
        "model",
        "split_role",
        "dataset",
        "selection",
        "normalizer",
        "evaluator",
        "generation",
        "runtime",
        "aggregate",
        "evaluation",
        "outputs",
    }
    if expected_physics is not None:
        expected_keys.add("physics")
    if set(provenance) != expected_keys:
        missing = sorted(expected_keys.difference(provenance))
        unexpected = sorted(set(provenance).difference(expected_keys))
        msg = f"Artifact provenance schema mismatch: missing={missing}, unexpected={unexpected}."
        raise ValueError(msg)
    if provenance.get("artifact_kind") != contracts.STEADY_ARTIFACT_KIND:
        msg = f"Artifact kind must be {contracts.STEADY_ARTIFACT_KIND!r} for a steady role."
        raise ValueError(msg)

    expected_run = _expected_run_provenance(
        summary=summary,
        task_spec=task_spec,
        run_name=run_name,
        evaluation=evaluation,
    )
    actual_run = dict(_mapping(provenance.get("run"), label="artifact provenance run"))
    semantic_run_keys = (
        "task",
        "task_contract_digest",
        "effective_config_digest",
        "best_checkpoint_sha256",
        "normalizer_sha256",
    )
    if {key: actual_run.get(key) for key in semantic_run_keys} != {key: expected_run.get(key) for key in semantic_run_keys}:
        msg = "Artifact run identity does not match the authoritative config, checkpoint, and normalizer."
        raise ValueError(msg)
    expected_model = _expected_model_provenance(config=config, summary=summary)
    if dict(_mapping(provenance.get("model"), label="artifact provenance model")) != expected_model:
        msg = "Artifact model provenance does not match the evaluable run configuration and capacity."
        raise ValueError(msg)
    expected_normalizer = _expected_normalizer_provenance(
        task_spec=task_spec,
        normalizer_sha256=summary.get("normalizer_sha256"),
        normalizer_state=normalizer_state,
    )
    if dict(_mapping(provenance.get("normalizer"), label="artifact provenance normalizer")) != expected_normalizer:
        msg = "Artifact normalizer provenance does not match the evaluable run."
        raise ValueError(msg)
    expected_evaluator = _expected_evaluator_provenance(config=config, task_spec=task_spec)
    if dict(_mapping(provenance.get("evaluator"), label="artifact provenance evaluator")) != expected_evaluator:
        msg = "Artifact evaluator provenance does not match the resolved task and run evaluation contract."
        raise ValueError(msg)
    if expected_physics is None:
        if "physics" in provenance:
            msg = "Artifact physics provenance is unexpected for this task."
            raise ValueError(msg)
    elif dict(_mapping(provenance.get("physics"), label="artifact provenance physics")) != expected_physics:
        msg = "Artifact physics provenance does not match the current task diagnostic contract."
        raise ValueError(msg)

    if provenance.get("split_role") != role.split_role:
        msg = "Artifact split role does not match its exact ID/OOD root."
        raise ValueError(msg)
    if dict(_mapping(provenance.get("dataset"), label="artifact provenance dataset")) != role.plan.dataset_provenance():
        msg = "Artifact dataset provenance does not match saved dataset identity and membership."
        raise ValueError(msg)
    data_config = _mapping(config.get("data"), label="config.yaml data")
    evaluation_batch_size = _positive_int(data_config.get("batch_size"), label="config.yaml data.batch_size")
    effective_count, effective_indices = _effective_selection(
        role,
        provenance,
        evaluation_batch_size=evaluation_batch_size,
    )
    if len(frame) != effective_count:
        msg = "Artifact Parquet row count contradicts the complete saved membership."
        raise ValueError(msg)
    _validate_payload_membership(
        frame=frame,
        provenance=provenance,
        role=role,
        effective_indices=effective_indices,
    )
    if contracts.validate_runtime_provenance(provenance.get("runtime")) != evaluation_batch_size:
        msg = "Artifact runtime inference batch size does not match the evaluable run evaluation batch size."
        raise ValueError(msg)
    timing = artifact_timing.timing_block(provenance, records=frozenset({"report"}))
    _validate_runtime_comparison_binding(timing["report"], frame=frame, role=role)
    return contracts.artifact_identity_digest(provenance)


def _validate_runtime_comparison_binding(
    value: Any,
    *,
    frame: pd.DataFrame,
    role: _SavedRole,
) -> dict[str, Any]:
    """Bind the required timing report to exact run, Dataset, role, and cases."""
    payload = artifact_timing.validate_runtime_comparison(value)
    provenance = dataframe.require_complete_provenance(frame)
    run = _mapping(provenance.get("run"), label="artifact provenance run")
    runtime = _mapping(provenance.get("runtime"), label="artifact provenance runtime")
    expected_dataset = artifact_timing.runtime_dataset_identity(provenance)
    expected_model = {
        "run_name": run.get("name"),
        "effective_config_digest": run.get("effective_config_digest"),
        "best_checkpoint_sha256": run.get("best_checkpoint_sha256"),
    }
    cases = payload["cases"]
    if payload["dataset_identity"] != expected_dataset or payload["model_identity"] != expected_model:
        msg = "Runtime comparison disagrees with exact artifact Dataset or checkpoint identity."
        raise ValueError(msg)
    if payload["split_role"] != role.split_role or payload["measurement"]["batch_size"] != runtime.get("inference_batch_size"):
        msg = "Runtime comparison disagrees with the artifact role or evaluation batch size."
        raise ValueError(msg)
    if tuple(case["source_index"] for case in cases) != tuple(int(value) for value in frame["source_index"]):
        msg = "Runtime comparison source membership disagrees with the artifact."
        raise ValueError(msg)
    if tuple(case["case_id"] for case in cases) != role.plan.case_ids:
        msg = "Runtime comparison case identity disagrees with the saved Dataset."
        raise ValueError(msg)
    return payload


def _read_artifact_root(artifact_root: str | Path) -> pd.DataFrame:
    """
    Read one explicit artifact root and its manifest-declared Parquet payload.

    The frame carries complete provenance (table and provenance agree) but is not
    admitted against a run, timing, or role; callers bind it with
    ``_bind_artifact_to_run``. Never scans for alternate tables or sidecars.

    Raises
    ------
    OSError, TypeError, ValueError, ComparisonCompatibilityError
        If provenance is unreadable, the manifest does not identify one contained
        Parquet file, or table/provenance contracts disagree.

    """
    root = Path(artifact_root).resolve()
    provenance_path = contracts.artifact_provenance_path(root)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Artifact provenance is unreadable: {provenance_path}: {error}"
        raise ValueError(msg) from error
    if not isinstance(provenance, Mapping):
        msg = "Artifact provenance must contain one JSON object."
        raise TypeError(msg)
    outputs = _mapping(provenance.get("outputs"), label="provenance.outputs")
    parquet = _mapping(outputs.get("parquet"), label="provenance.outputs.parquet")
    parquet_name = parquet.get("path")
    if not isinstance(parquet_name, str) or not parquet_name or Path(parquet_name).suffix != ".parquet":
        msg = "Artifact output manifest must declare one Parquet path."
        raise ValueError(msg)
    parquet_path = (root / parquet_name).resolve()
    if not parquet_path.is_relative_to(root) or not parquet_path.is_file():
        msg = "Declared artifact Parquet payload is missing or escapes its root."
        raise ValueError(msg)
    contracts.require_output_structure(root, outputs)
    frame_raw = pd.read_parquet(parquet_path)
    frame_raw.attrs["artifact_root"] = str(root)
    frame_raw.attrs["artifact_provenance"] = dict(provenance)
    frame = dataframe.build_eval_df(frame_raw)
    resolved_paths = [
        contracts.resolve_case_payload_path(
            root,
            stored_path,
            expected_filename=contracts.case_payload_filename(Path(str(stored_path)).stem),
        )
        for stored_path in frame["npz_path"].tolist()
    ]
    if len(set(resolved_paths)) != len(resolved_paths):
        msg = "Artifact Parquet rows share one NPZ payload."
        raise ValueError(msg)
    frame.loc[:, "npz_path"] = [str(path) for path in resolved_paths]
    return frame


def _apply_role_attrs(
    frame: pd.DataFrame,
    *,
    root: Path,
    provenance: Mapping[str, Any],
    kind: str,
    scope: str,
    config: Mapping[str, Any],
) -> None:
    """Attach the role attrs both tasks share; task-specific attrs stay with their admission path."""
    frame.attrs["artifact_root"] = str(root.resolve())
    frame.attrs["artifact_provenance"] = deepcopy(dict(provenance))
    frame.attrs["artifact_kind"] = kind
    frame.attrs["artifact_scope"] = scope
    frame.attrs[dataframe.COMPLETED_RUN_CONFIG_ATTR] = deepcopy(dict(config))


def _load_role(
    *,
    role: _SavedRole,
    task: str,
    run_name: str,
    run_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
    task_spec: TaskSpec,
    evaluation: Mapping[str, Any],
) -> LoadedEvaluationArtifact:
    """
    Load and bind one exact role without NPZ array access or mutation.

    Parameters
    ----------
    role : _SavedRole
        Authoritative saved role identity and target root.
    task, run_name : str
        Canonical completed-run selection.
    run_dir : pathlib.Path
        Exact current run directory owning the artifacts.
    config : collections.abc.Mapping
        Validated resolved run configuration.
    summary : collections.abc.Mapping
        Validated completed-run summary.
    normalizer_state : collections.abc.Mapping
        Validated saved training normalizer state.
    task_spec : TaskSpec
        Validated persisted task contract.
    evaluation : collections.abc.Mapping
        Validated resolved evaluation configuration.

    Returns
    -------
    LoadedEvaluationArtifact
        Role-local frame, root, dataset identity, and canonical digest.

    Raises
    ------
    MissingEvaluationArtifactsError
        If the exact role target is absent or empty.
    IncompatibleEvaluationArtifactsError
        If the target is partial, unreadable, changed, or contradictory.

    """
    root = role.root
    if not root.exists():
        raise _missing(task=task, run_name=run_name, run_dir=run_dir, role=role.split_role, root=root)
    if not root.is_dir():
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason="expected artifact root is not a directory",
        )
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason=f"artifact root cannot be resolved exactly: {error}",
        ) from error
    if resolved_root != root:
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason="artifact root uses a symbolic-link alias outside its exact run-owned path",
        )
    provenance_path = contracts.artifact_provenance_path(root)
    if not provenance_path.is_file():
        try:
            has_content = any(root.iterdir())
        except OSError as error:
            raise _incompatible(
                task=task,
                run_name=run_name,
                run_dir=run_dir,
                role=role.split_role,
                root=root,
                reason=f"artifact root cannot be inspected: {error}",
            ) from error
        if not has_content:
            raise _missing(task=task, run_name=run_name, run_dir=run_dir, role=role.split_role, root=root)
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason=f"partial artifact target has no completion marker {provenance_path.name}",
        )
    try:
        frame = _read_artifact_root(root)
        identity_sha256 = _bind_artifact_to_run(
            frame=frame,
            role=role,
            config=config,
            summary=summary,
            normalizer_state=normalizer_state,
            task_spec=task_spec,
            run_name=run_name,
            evaluation=evaluation,
        )
    except EvaluationArtifactLoadError:
        raise
    except Exception as error:
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason=f"{type(error).__name__}: {error}",
        ) from error
    _apply_role_attrs(
        frame,
        root=root,
        provenance=dataframe.require_complete_provenance(frame),
        kind=contracts.STEADY_ARTIFACT_KIND,
        scope="canonical",
        config=config,
    )
    return LoadedEvaluationArtifact(
        split_role=role.split_role,
        dataset_name=role.dataset_name,
        root=root,
        frame=frame,
        identity_sha256=identity_sha256,
    )


def _physical_split(split: Mapping[str, Any]) -> dict[str, Any]:
    """Return the admitted split's Dataset identities and role memberships, without the task's item expansion."""
    payload = datasets.preprocessing.splits.admit_split(split).as_payload()
    return {"dataset_identity": payload["dataset_identity"], "roles": payload["roles"]}


def _transient_role_artifact(
    run_dir: Path,
    admitted: Mapping[str, Any],
    plan: artifact_plan.ArtifactPlan,
    *,
    selection_role: ArtifactSelectionRole,
    root: Path | str | None,
    scoped: bool,
    evaluation_spatial_stride: int = 1,
) -> LoadedEvaluationArtifact:
    """Admit one transient sequence role at its canonical root, a private stage, a grid variant or a scoped root."""
    from src.analysis.artifacts import transient  # noqa: PLC0415

    split_role = artifact_plan.split_of(selection_role)
    if split_role != "eval" and split_role not in plan.ood_roles:
        raise _missing(
            task="transient_drying",
            run_name=plan.run_name,
            run_dir=run_dir,
            role=split_role,
            root=common.paths.resolve_analysis_root(run_dir) / split_role,
        )
    role_plan = plan.role(split_role)
    canonical_root = role_plan.root(run_dir).absolute()
    target = canonical_root if root is None else Path(root).expanduser().resolve(strict=False)
    if not target.exists():
        raise _missing(task="transient_drying", run_name=plan.run_name, run_dir=run_dir, role=split_role, root=target)
    marker = contracts.artifact_provenance_path(target)
    reason = None
    if not target.is_dir() or target.resolve(strict=True) != target:
        reason = "artifact root is not one exact regular run-owned directory"
    elif not marker.is_file():
        if not any(target.iterdir()):
            raise _missing(task="transient_drying", run_name=plan.run_name, run_dir=run_dir, role=split_role, root=target)
        reason = f"partial sequence artifact has no completion marker {marker.name}"
    if reason is not None:
        raise _incompatible(task="transient_drying", run_name=plan.run_name, run_dir=run_dir, role=split_role, root=target, reason=reason)
    try:
        loaded = (
            transient.validate_scoped_transient_role_artifact_index(target, completed=admitted, saved_role=role_plan)
            if scoped
            else transient.validate_transient_role_artifact_index(
                target,
                completed=admitted,
                role=role_plan,
                evaluation_spatial_stride=evaluation_spatial_stride,
            )
        )
    except Exception as error:
        raise _incompatible(
            task="transient_drying",
            run_name=plan.run_name,
            run_dir=run_dir,
            role=split_role,
            root=target,
            reason=f"{type(error).__name__}: {error}",
        ) from error
    frame = loaded.frame
    _apply_role_attrs(
        frame,
        root=target,
        provenance=loaded.provenance,
        kind="transient_sequence",
        scope="scoped_partial" if scoped else "canonical",
        config=cast("Mapping[str, Any]", admitted["config"]),
    )
    frame.attrs["transient_sequence_index"] = loaded
    frame.attrs["transient_unavailable_horizons"] = loaded.unavailable_horizons
    frame.attrs["transient_scaling_state"] = deepcopy(cast("Mapping[str, Any]", admitted["normalizer_state"]))
    frame.attrs["transient_physical_split"] = _physical_split(_mapping(admitted.get("split_indices"), label="admitted transient split"))
    return LoadedEvaluationArtifact(
        split_role=split_role,
        dataset_name=role_plan.dataset_name,
        root=target,
        frame=frame,
        identity_sha256=loaded.identity_sha256,
    )


def _require_role_selection(artifact_roles: tuple[ArtifactSelectionRole, ...]) -> None:
    """Refuse an empty, repeated or unknown artifact role selection."""
    if not artifact_roles or len(set(artifact_roles)) != len(artifact_roles) or set(artifact_roles).difference(artifact_plan.ARTIFACT_DATASET_ROLES):
        msg = f"artifact_roles must contain unique values from {list(artifact_plan.ARTIFACT_DATASET_ROLES)}."
        raise ValueError(msg)


def _load_admitted_transient_run_artifacts(
    run_dir: Path,
    admitted: Mapping[str, Any],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...],
    artifact_root_overrides: Mapping[ArtifactSelectionRole, Path | str] | None = None,
) -> LoadedRunArtifacts:
    """Load completed transient sequence roles without static artifact assumptions."""
    plan = artifact_plan.artifact_plan_from_completed(admitted, run_dir=run_dir)
    _require_role_selection(artifact_roles)
    overrides = {} if artifact_root_overrides is None else dict(artifact_root_overrides)
    if set(overrides).difference(artifact_roles):
        msg = "Artifact root overrides must belong to selected transient roles."
        raise ValueError(msg)
    loaded = {
        selection_role: _transient_role_artifact(
            run_dir,
            admitted,
            plan,
            selection_role=selection_role,
            root=overrides.get(selection_role),
            scoped=selection_role in overrides,
        )
        for selection_role in artifact_plan.ARTIFACT_DATASET_ROLES
        if selection_role in artifact_roles
    }
    summary = _mapping(admitted.get("summary"), label="completed transient summary")
    return LoadedRunArtifacts(
        task="transient_drying",
        run_name=plan.run_name,
        run_dir=run_dir,
        id_artifact=loaded.get("id"),
        material_ood_artifact=loaded.get("material_ood"),
        parameter_ood_artifact=loaded.get("parameter_ood"),
        selected_checkpoint_role=str(admitted["selected_checkpoint_role"]),
        selected_checkpoint_epoch=int(admitted["selected_checkpoint_epoch"]),
        selected_checkpoint_sha256=str(admitted["selected_checkpoint_sha256"]),
        summary=deepcopy(dict(summary)),
    )


def _load_admitted_run_artifacts(
    run_dir: Path,
    admitted: Mapping[str, Any],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...] | None = None,
    artifact_root_overrides: Mapping[ArtifactSelectionRole, Path | str] | None = None,
) -> LoadedRunArtifacts:
    """Load both artifact roles from one already admitted current run path."""
    run_dir = run_dir.expanduser().resolve()
    summary = _mapping(admitted.get("summary"), label="evaluable summary")
    task_spec = experiments.config.loader.validate_resolved_task_contract(_mapping(admitted.get("config"), label="evaluable config"))
    if task_spec.id == "transient_drying":
        selected_roles: tuple[ArtifactSelectionRole, ...]
        if artifact_roles is None:
            selected_roles = artifact_plan.artifact_plan_from_completed(admitted, run_dir=run_dir).dataset_roles
        else:
            selected_roles = artifact_roles
        return _load_admitted_transient_run_artifacts(
            run_dir,
            admitted,
            artifact_roles=selected_roles,
            artifact_root_overrides=artifact_root_overrides,
        )
    if artifact_root_overrides:
        msg = "Explicit artifact root overrides are supported only for transient scoped artifacts."
        raise ValueError(msg)
    context, saved_roles = _steady_role_context(run_dir, admitted, task_spec)
    if artifact_roles is None:
        artifact_roles = tuple(saved_roles)
    _require_role_selection(artifact_roles)
    unplanned = [role for role in artifact_roles if role not in saved_roles]
    if unplanned:
        msg = f"Evaluable run has no saved {unplanned[0]} role."
        raise ValueError(msg)
    loaded = {role: _load_role(role=saved_roles[role], **context) for role in artifact_roles}
    evaluation = context["evaluation"]
    return LoadedRunArtifacts(
        task=task_spec.id,
        run_name=context["run_name"],
        run_dir=run_dir,
        id_artifact=loaded.get("id"),
        material_ood_artifact=loaded.get("material_ood"),
        parameter_ood_artifact=loaded.get("parameter_ood"),
        selected_checkpoint_role=str(evaluation["selected_checkpoint_role"]),
        selected_checkpoint_epoch=(int(evaluation["selected_checkpoint_epoch"]) if evaluation["selected_checkpoint_epoch"] is not None else None),
        selected_checkpoint_sha256=str(evaluation["selected_checkpoint_sha256"] or ""),
        summary=deepcopy(dict(summary)),
    )


def admit_role_artifact(
    run_dir: Path | str,
    admitted: Mapping[str, Any],
    *,
    selection_role: ArtifactSelectionRole,
    root: Path | str | None = None,
    evaluation_spatial_stride: int = 1,
) -> LoadedEvaluationArtifact:
    """
    Admit one role artifact through the authoritative validator of its task.

    Parameters
    ----------
    run_dir : pathlib.Path | str
        Completed run that owns the role.
    admitted : collections.abc.Mapping
        Completed-run evidence from the run admission gate.
    selection_role : {"id", "material_ood", "parameter_ood"}
        Saved role whose exact membership the artifact must carry.
    root : pathlib.Path | str | None, optional
        Private build stage or transient Evaluation-grid variant to admit instead
        of the canonical role root; the saved membership still applies unchanged.
    evaluation_spatial_stride : int, optional
        Transient Evaluation-grid stride of ``root``. Steady artifacts use one.

    Returns
    -------
    LoadedEvaluationArtifact
        The role with the same frame attrs the read path returns.

    Raises
    ------
    MissingEvaluationArtifactsError, IncompatibleEvaluationArtifactsError
        If the target is absent, partial, or contradicts any bound evidence.

    Notes
    -----
    The artifact service admits each stage before publication, each cache hit and
    each published target through this function, so write, cache and read share
    one validator.

    """
    path = Path(run_dir).expanduser().resolve()
    task_spec = experiments.config.loader.validate_resolved_task_contract(_mapping(admitted.get("config"), label="evaluable config"))
    if task_spec.id == "transient_drying":
        plan = artifact_plan.artifact_plan_from_completed(admitted, run_dir=path)
        return _transient_role_artifact(
            path,
            admitted,
            plan,
            selection_role=selection_role,
            root=root,
            scoped=False,
            evaluation_spatial_stride=evaluation_spatial_stride,
        )
    if evaluation_spatial_stride != 1:
        msg = "Steady artifacts are admitted only on the canonical Evaluation grid."
        raise ValueError(msg)
    context, saved_roles = _steady_role_context(path, admitted, task_spec)
    role = saved_roles.get(selection_role)
    if role is None:
        msg = f"Evaluable run has no saved {selection_role} role."
        raise ValueError(msg)
    return _load_role(role=role if root is None else replace(role, root=Path(root).expanduser().absolute()), **context)


def _steady_role_context(
    run_dir: Path,
    admitted: Mapping[str, Any],
    task_spec: TaskSpec,
) -> tuple[dict[str, Any], dict[ArtifactSelectionRole, _SavedRole]]:
    """Validate steady run evidence once and return the role-binding context with every saved role."""
    config = _mapping(admitted.get("config"), label="evaluable config")
    summary = _mapping(admitted.get("summary"), label="evaluable summary")
    normalizer_state = _mapping(admitted.get("normalizer_state"), label="evaluable normalizer state")
    run_config = _mapping(config.get("run"), label="config.yaml run")
    scientific_run_name = _nonempty_string(run_config.get("name"), label="config.yaml run.name")
    if config.get("task") != task_spec.id or summary.get("task") != task_spec.id:
        msg = "Saved task identity contradicts the authoritative evaluable run."
        raise ValueError(msg)
    if summary.get("run_name") != scientific_run_name:
        msg = "summary.json run_name contradicts config.yaml scientific run identity."
        raise ValueError(msg)
    admitted_dir = admitted.get("run_dir")
    if admitted_dir is not None and Path(admitted_dir).resolve() != run_dir:
        msg = "Run validator returned a different current run directory."
        raise ValueError(msg)

    evidence_summary = dict(summary)
    evidence_summary["effective_config_digest"] = admitted["effective_config_digest"]
    evidence_summary["best_checkpoint_sha256"] = admitted["selected_checkpoint_sha256"]
    evidence_summary["normalizer_sha256"] = admitted["normalizer_sha256"]
    evaluation = {
        "selected_checkpoint_role": admitted["selected_checkpoint_role"],
        "selected_checkpoint_epoch": admitted["selected_checkpoint_epoch"],
        "selected_checkpoint_sha256": admitted["selected_checkpoint_sha256"],
    }
    try:
        saved_roles = _saved_roles(
            completed=admitted,
            run_dir=run_dir,
            task=task_spec.id,
            task_spec=task_spec,
        )
    except Exception as error:
        msg = f"Evaluable run has incompatible saved artifact identity: {type(error).__name__}: {error}"
        raise ValueError(msg) from error
    context = {
        "task": task_spec.id,
        "run_name": scientific_run_name,
        "run_dir": run_dir,
        "config": config,
        "summary": evidence_summary,
        "normalizer_state": normalizer_state,
        "task_spec": task_spec,
        "evaluation": evaluation,
    }
    return context, saved_roles


def load_run_artifacts(
    run_dir: Path | str,
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...] | None = None,
    artifact_root_overrides: Mapping[ArtifactSelectionRole, Path | str] | None = None,
) -> LoadedRunArtifacts:
    """
    Load strict ID and OOD artifacts from one explicit current run directory.

    The directory may be a direct run or Optuna trial at any current location.
    Its leaf is a storage alias; immutable scientific identity comes only from
    validated ``config.yaml`` and ``summary.json`` evidence.
    """
    path = Path(run_dir).expanduser().resolve()
    with experiments.run.saved_artifact_run_lease(path) as admitted:
        return _load_admitted_run_artifacts(
            path,
            admitted,
            artifact_roles=artifact_roles,
            artifact_root_overrides=artifact_root_overrides,
        )
