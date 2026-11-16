"""Discover persisted Evaluation runs without opening models or numerical arrays.

Responsibilities:
  - Recursively identify config-and-summary run leaves below an experiments root, skipping Optuna trials by their recorded layout
  - Bind current transient children through validated immutable parent records
  - Inspect checkpoint and analysis-artifact availability without mutation, taking the recorded best digest without opening checkpoints
  - Admit available artifacts through exact run, Dataset, and payload evidence
  - Return deterministic catalog evidence and bounded per-candidate issues
  - Resolve each run's model family from its saved config, or record an issue and keep the run listed without one
  - Carry the run name of the A0 that a B or A1 run starts from, as its saved config records it
  - Read one completed stage's work quantities (time, runtime, work, units, budget) from its summary
  - Name the distinct completed runs of labelled frames by their first frame, a run being its name and end time
  - Read one completed stage's training history through its run admission and the history's own validator, with the
    stage's start epoch and its selected validation point; a missing or unusable history is unavailable with its reason
  - Read the keys of an Optuna study summary that the study curve draws

Design principles:
  - Persisted config, summary, saved split, and parent records remain authoritative; current Dataset manifests are never consulted
  - Discovery never contacts tracking services or infers scientific identity
  - One malformed candidate cannot prevent inspection of unrelated runs
  - A work quantity the summary does not record is ``None``, never zero
  - A history is read only from an admitted completed run, bound to the epochs and points its last checkpoint counts

This module does NOT:
  - Load models, source datasets, numerical arrays, widgets, plots, or notebooks; a history read leaves checkpoint loading to the
    run admission (``experiments.run``)
  - Generate, repair, publish, or mutate any persisted evidence
  - Validate history records or study summaries beyond the keys its readers return; their producers own the schemas
  - Create storage layouts or use directory names as scientific identity
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import yaml

from src import common, datasets, experiments, learning
from src.analysis.artifacts import analysis_artifact_contracts as artifact_contracts
from src.experiments import experiments_run_identity as run_identity
from src.experiments.config import experiments_config_naming as config_naming
from src.learning.models import naming
from src.learning.training import learning_training_budget as stage_budget

from . import evaluation_artifact_loader as artifact_loader
from . import evaluation_dataframe as dataframe
from . import evaluation_transient_artifact as transient_artifact

if TYPE_CHECKING:
    import pandas as pd

ArtifactState = Literal["ready", "generating", "scoped_partial", "missing", "invalid"]
RunAvailability = Literal["available", "unavailable"]

_SHA256_HEX_LENGTH = 64
# The text of a work quantity a run did not record, also inside a value joined over its attempts.
UNAVAILABLE: Final = "unavailable"
_HANDOFF_STAGES: Final = frozenset({"B", "A1"})  # stages that start from the A0 teacher handoff
_OBJECTIVE_DIRECTIONS: Final = frozenset({"minimize", "maximize"})
# The finished trial statuses: the only ones whose values the study summary counts towards the best so far
# (``experiments_tuning_optuna._BEST_STATUSES`` and its use in ``_study_lifecycle_summary``).
STUDY_FINISHED_STATUSES: Final = ("completed", "duplicate")


@dataclass(frozen=True, slots=True)
class StageHistory:
    """
    Hold one completed stage's admitted training history and what places it on a shared epoch axis.

    ``stage`` is ``steady``, ``A0``, ``B`` or ``A1``; ``pipeline`` is the run name of the A0 a B or A1 starts from (else the run's
    own) and ``start_epoch`` that A0's handoff epoch (else 0). ``selected_point`` is the validation point of the selected (best)
    checkpoint. ``epochs`` and ``validations`` are the history's records in order.
    """

    run_name: str
    stage: str
    model_family: str
    pipeline: str
    start_epoch: int
    objective_id: str
    objective_direction: str
    selected_point: int
    epochs: tuple[Mapping[str, float], ...]
    validations: tuple[Mapping[str, float], ...]


@dataclass(frozen=True, slots=True)
class UnavailableHistory:
    """Name a completed run whose training history cannot be shown and why."""

    run_dir: Path
    reason: str


@dataclass(frozen=True, slots=True)
class StudyTrial:
    """One Optuna trial of a study summary: its number, terminal status, objective value and the best value so far."""

    number: int
    status: str
    value: float | None
    best_so_far: float | None


@dataclass(frozen=True, slots=True)
class StudySummary:
    """The keys of one study summary the study curve reads."""

    study_name: str
    objective_id: str
    direction: str
    trial_counts: Mapping[str, int]
    trials: tuple[StudyTrial, ...]


@dataclass(frozen=True, slots=True)
class EvaluationArtifactInspection:
    """Describe one read-only task-specific artifact target."""

    role: transient_artifact.DatasetRole
    root: Path
    state: ArtifactState
    digest: str | None
    capabilities: frozenset[str]
    case_count: int | None = None


@dataclass(frozen=True, slots=True)
class _ScopedArtifactCandidate:
    """Bind one marker-complete selected-case artifact to exact run evidence."""

    inspection: EvaluationArtifactInspection
    task: str
    run_name: str
    effective_config_digest: str
    checkpoint_sha256: str
    dataset_name: str


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointInspection:
    """Describe the best and latest checkpoint paths and the producer-recorded best digest (verified at selection)."""

    best_path: Path
    best_digest: str | None
    latest_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationRunDiscovery:
    """Describe one persisted run leaf selected only from saved metadata."""

    task: str
    run_name: str
    run_identity_sha256: str | None
    model_label: str
    model_kind: str | None
    model_family: str | None  # Network and physics-loss family (fno, uno, rno, pi-fno, pi-uno); None when the saved config does not resolve it.
    architecture: tuple[tuple[str, str], ...]
    seed: int | None
    stage: str | None
    status: str
    created_at: str | None
    updated_at: str | None
    run_dir: Path
    dataset_id: str | None
    evaluation_protocol: str | None
    evaluation_config_identity: str | None
    spatial_stride: int | None
    parent_identity_sha256: str | None
    parent_label: str | None
    identity_format: Literal["grouped", "independent"]
    label: str
    availability: RunAvailability
    evaluable: bool
    action_enabled: bool
    checkpoints: EvaluationCheckpointInspection
    id_artifact: EvaluationArtifactInspection
    artifact_command: str
    # Per non-empty OOD role (material_ood, parameter_ood): ordered source Datasets, artifact target and assignments.
    ood_dataset_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    ood_artifacts: Mapping[str, EvaluationArtifactInspection] = field(default_factory=dict)
    evaluation_assignments: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    metadata_issues: tuple[str, ...] = ()
    config_label: str | None = None
    pair_label: str | None = None
    a0_run_name: str | None = None  # The run name of the A0 a B or A1 run starts from, as its saved teacher handoff names it.

    def artifacts(self) -> tuple[EvaluationArtifactInspection, ...]:
        """Return the ID artifact target and every OOD artifact target in canonical role order."""
        return (self.id_artifact, *self.ood_artifacts.values())


@dataclass(frozen=True, slots=True)
class EvaluationRunGroup:
    """Describe an exact current parent identity and its discovered children."""

    identity_sha256: str
    label: str
    task: str
    children: tuple[EvaluationRunDiscovery, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDiscoveryIssue:
    """Describe one bounded candidate-read failure without a traceback."""

    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationRunCatalog:
    """Return a complete deterministic run catalog and its compact audit counts."""

    runs: tuple[EvaluationRunDiscovery, ...]
    groups: tuple[EvaluationRunGroup, ...]
    issues: tuple[EvaluationDiscoveryIssue, ...]
    counts_by_task: Mapping[str, int]
    counts_by_status: Mapping[str, int]
    counts_by_identity: Mapping[str, int]
    counts_by_artifact_state: Mapping[str, int]


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    """Return one persisted mapping or raise a concise contract error."""
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping."
        raise TypeError(message)
    return value


def _optional_mapping(value: object) -> Mapping[str, Any]:
    """Return one optional persisted mapping or an empty view."""
    return value if isinstance(value, Mapping) else {}


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """Read one persisted resolved config without invoking the config loader."""
    with path.open(encoding="utf-8") as stream:
        return _mapping(yaml.safe_load(stream), label="config.yaml")


def _read_json(path: Path) -> Mapping[str, Any]:
    """Read one persisted JSON mapping without mutation."""
    with path.open(encoding="utf-8") as stream:
        return _mapping(json.load(stream), label=path.name)


def _text(value: object) -> str | None:
    """Return one non-empty persisted text value."""
    return value if isinstance(value, str) and value else None


def _model_label(config: Mapping[str, Any]) -> str:
    """Use the producer-owned architecture label without stage or seed conflation."""
    return naming.architecture_label(config)


def _manifest_paths_are_present(outputs: object, *, root: Path) -> bool:
    """Validate manifest structure and regular-file presence without hashing bytes."""
    if not isinstance(outputs, Mapping) or set(outputs) != {"parquet", "npz"}:
        return False
    parquet = outputs["parquet"]
    npz = outputs["npz"]
    if not isinstance(parquet, Mapping) or not isinstance(npz, list) or not npz:
        return False
    entries = (parquet, *npz)
    admitted: list[Path] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            return False
        relative = entry["path"]
        digest = entry["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
        path = root / relative
        if not path.is_file() or path.is_symlink():
            return False
        admitted.append(path)
    parquet_files = tuple(path for path in admitted if path.suffix == ".parquet")
    npz_files = tuple(path for path in admitted if path.suffix == ".npz" and path.parent == root / "npz")
    return len(parquet_files) == 1 and len(npz_files) == len(npz)


def _artifact_payload_is_current(payload: Mapping[str, Any], *, root: Path, task: str) -> bool:
    """Return whether one marker has the recognized task and output paths."""
    run = payload.get("run")
    run_task = _text(run.get("task")) if isinstance(run, Mapping) else None
    if task not in {value for value in (_text(payload.get("task")), run_task) if value is not None}:
        return False
    return _manifest_paths_are_present(payload.get("outputs"), root=root)


def _artifact_writer_active(root: Path) -> bool:
    """Return whether the exact canonical target currently has a writer lease."""
    lock_path = common.paths.resolve_artifact_lock_path(root)
    if not lock_path.is_file() or lock_path.is_symlink():
        return False
    try:
        with common.locking.exclusive_file_lock(lock_path, blocking=False):
            return False
    except common.locking.FileLockUnavailableError:
        return True


def _artifact(root: Path, *, role: transient_artifact.DatasetRole, task: str) -> EvaluationArtifactInspection:
    """Inspect one canonical marker, staging lease, and payload inventory without arrays."""
    stage = artifact_contracts.artifact_stage_path(root)
    staged = stage.is_dir() and not stage.is_symlink()
    if not root.exists():
        if staged:
            state: ArtifactState = "generating" if _artifact_writer_active(root) else "invalid"
            return EvaluationArtifactInspection(
                role,
                root,
                state,
                None,
                frozenset(),
            )
        return EvaluationArtifactInspection(
            role,
            root,
            "missing",
            None,
            frozenset(),
        )
    if not root.is_dir() or root.is_symlink():
        return EvaluationArtifactInspection(
            role,
            root,
            "invalid",
            None,
            frozenset(),
        )
    marker = artifact_contracts.artifact_provenance_path(root)
    if not marker.is_file() or marker.is_symlink():
        state = "generating" if staged and _artifact_writer_active(root) else "invalid"
        return EvaluationArtifactInspection(
            role,
            root,
            state,
            None,
            frozenset(),
        )
    try:
        payload = _read_json(marker)
        if not _artifact_payload_is_current(payload, root=root, task=task):
            return EvaluationArtifactInspection(
                role,
                root,
                "invalid",
                None,
                frozenset(),
            )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return EvaluationArtifactInspection(
            role,
            root,
            "invalid",
            None,
            frozenset(),
        )
    capabilities = {"aggregate", "timing"}
    if role == "id":
        capabilities.add("case_fields")
    if payload.get("artifact_kind") == "transient_sequence":
        capabilities.update({"trajectory", "rollout", "case_fields"})
    return EvaluationArtifactInspection(
        role,
        root,
        "ready",
        common.serialization.file_sha256(marker),
        frozenset(capabilities),
    )


def _scoped_artifact_candidate(marker: Path) -> _ScopedArtifactCandidate:
    """Admit one exact marker-complete selected-case artifact candidate."""
    payload = _read_json(marker)
    if payload.get("artifact_kind") != transient_artifact.TRANSIENT_ARTIFACT_KIND:
        message = "Scoped candidate is not a transient sequence artifact."
        raise ValueError(message)
    task = _text(payload.get("task"))
    run = _mapping(payload.get("run"), label="scoped artifact run")
    dataset = _mapping(
        payload.get("dataset"),
        label="scoped artifact Dataset",
    )
    evaluation = _mapping(
        payload.get("evaluation"),
        label="scoped artifact Evaluation",
    )
    scope = _mapping(
        evaluation.get("scope"),
        label="scoped artifact scope",
    )
    role = dataset.get("role")
    selected = scope.get("selected_case_ids")
    if (
        task != "transient_drying"
        or role not in transient_artifact.DATASET_ROLES
        or scope.get("kind") != "selected_cases"
        or scope.get("canonical_publication_eligible") is not False
        or not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or any(not isinstance(case_id, str) or not case_id for case_id in selected)
    ):
        message = "Scoped artifact scope is invalid."
        raise ValueError(message)
    root = marker.parent
    inspected = _artifact(
        root,
        role=role,
        task=task,
    )
    if inspected.state != "ready":
        message = "Scoped artifact payload is not marker-complete."
        raise ValueError(message)
    run_name = _text(run.get("name"))
    effective = _text(run.get("effective_config_digest"))
    checkpoint = _text(run.get("best_checkpoint_sha256"))
    dataset_name = _text(dataset.get("name"))
    if run_name is None or effective is None or checkpoint is None or dataset_name is None:
        message = "Scoped artifact binding evidence is incomplete."
        raise ValueError(message)
    capabilities = inspected.capabilities.difference({"aggregate"})
    return _ScopedArtifactCandidate(
        inspection=replace(
            inspected,
            state="scoped_partial",
            capabilities=frozenset(capabilities),
            case_count=len(selected),
        ),
        task=task,
        run_name=run_name,
        effective_config_digest=effective,
        checkpoint_sha256=checkpoint,
        dataset_name=dataset_name,
    )


def _scoped_artifact_candidates(
    experiments_root: Path,
) -> tuple[
    tuple[_ScopedArtifactCandidate, ...],
    tuple[EvaluationDiscoveryIssue, ...],
]:
    """Discover marker-complete scoped transient artifacts from exact provenance."""
    candidates: list[_ScopedArtifactCandidate] = []
    issues: list[EvaluationDiscoveryIssue] = []
    for marker in sorted(
        experiments_root.rglob(
            artifact_contracts.ARTIFACT_PROVENANCE_FILENAME,
        )
    ):
        if "scoped_artifacts" not in marker.parts:
            continue
        try:
            candidates.append(_scoped_artifact_candidate(marker))
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            issues.append(
                EvaluationDiscoveryIssue(
                    marker.parent,
                    f"invalid scoped artifact: {type(error).__name__}",
                )
            )
    return tuple(candidates), tuple(issues)


def _recorded_checkpoint_digest(path: Path, recorded: object) -> str | None:
    """Return the producer-recorded digest only for an exact regular file; selection verifies the bytes."""
    if not path.is_file() or path.is_symlink():
        return None
    if not isinstance(recorded, str) or len(recorded) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in recorded):
        return None
    return recorded


def _stage_label(training: Mapping[str, Any]) -> str | None:
    """Normalize persisted comparison arms before generic training stages."""
    arm = _text(training.get("comparison_arm"))
    if arm is not None:
        normalized = arm.lower()
        if normalized == "a0":
            return "A0"
        if normalized == "a1":
            return "A1"
        if normalized == "b":
            return "B"
        return "A"
    stage = _text(training.get("stage"))
    return stage.upper() if stage is not None else None


def _independent_group_identity(record: EvaluationRunDiscovery) -> str:
    """Return a stable exact synthetic group identity for one independent leaf."""
    return common.serialization.canonical_json_sha256({"task": record.task, "run_name": record.run_name, "run_dir": str(record.run_dir)})


def _evaluation_config_identity(task: str, evaluation: Mapping[str, Any]) -> str | None:
    """Return the task-aware complete persisted Evaluation protocol identity."""
    if not evaluation:
        return None
    if task == "transient_drying":
        return transient_artifact.evaluation_protocol_identity(evaluation)
    return common.serialization.canonical_json_sha256(evaluation)


def _run_from_leaf(
    run_dir: Path,
    *,
    scoped_candidates: tuple[_ScopedArtifactCandidate, ...] = (),
) -> EvaluationRunDiscovery | None:
    """Build one leaf record from its authoritative config and summary files; Optuna trials yield none."""
    config = _read_yaml(run_dir / common.paths.RUN_CONFIG_FILENAME)
    summary = _read_json(run_dir / common.paths.RUN_SUMMARY_FILENAME)
    task = _text(config.get("task"))
    run = _mapping(config.get("run"), label="config.yaml run")
    common.paths.resolve_config_run_dir(config)  # Validate the complete current declared layout independently of physical placement.
    layout = _mapping(run.get("layout"), label="config.yaml run.layout")
    if artifact_contracts.is_optuna_trial_config(config):
        return None  # The validated layout identifies an Optuna trial: search evidence whose checkpoints are not kept.
    try:
        common.paths.resolve_config_output_root(config, run_dir)
    except ValueError:
        if layout["stage"] != "single":
            raise  # Only a relocated independent bundle may lack a resolvable output root.
    run_name = _text(run.get("name"))
    if task is None or run_name is None:
        message = "config.yaml must define task and run.name."
        raise ValueError(message)
    if summary.get("task") != task or summary.get("run_name") != run_name:
        message = "summary.json contradicts persisted config task or run identity."
        raise ValueError(message)
    status = _text(summary.get("status"))
    if status is None:
        message = "summary.json must define status."
        raise ValueError(message)
    data_mapping = _optional_mapping(config.get("data"))
    ood_dataset_ids: dict[str, tuple[str, ...]] = {}
    assignments: dict[str, tuple[Mapping[str, Any], ...]] = {}
    ood_names: dict[transient_artifact.DatasetRole, str] = {}
    metadata_issues: tuple[str, ...] = ()
    if common.paths.resolve_split_indices_path(run_dir).is_file():  # A run that ended before its split was persisted has no OOD role.
        try:
            split = datasets.preprocessing.splits.read_saved_split(run_dir)
        except (OSError, ValueError, TypeError) as error:
            metadata_issues = (f"Saved split unavailable: {type(error).__name__}: {error}",)
        else:
            for ood_role in split.ood_roles:
                ood_dataset_ids[ood_role] = tuple(part.dataset_id for part in split.ood_parts(ood_role))
                assignments[ood_role] = split.evaluation_assignments[ood_role]
                ood_names[ood_role] = str(split.ood_name(ood_role))
    model_family: str | None = None
    try:
        model_family = config_naming.model_family(config)
    except (KeyError, TypeError) as error:  # The run stays listed without an architecture group; only its actions are withheld.
        metadata_issues = (*metadata_issues, f"Model family unavailable: {type(error).__name__}: {error}")
    model_label = run_name  # The run name labels a run whose model block does not resolve.
    try:
        model_label = _model_label(config)
    except (KeyError, TypeError, ValueError) as error:
        metadata_issues = (*metadata_issues, f"Model label unavailable: {type(error).__name__}: {error}")
    training_mapping = _optional_mapping(config.get("training"))
    model_mapping = _optional_mapping(config.get("model"))
    parameter_mapping = _optional_mapping(model_mapping.get("params"))
    evaluation_mapping = _optional_mapping(config.get("evaluation"))
    objective_mapping = _optional_mapping(evaluation_mapping.get("objective"))
    stage = _stage_label(training_mapping)
    best = run_dir / common.paths.RUN_BEST_CHECKPOINT_FILENAME
    latest = run_dir / common.paths.RUN_LAST_CHECKPOINT_FILENAME
    best_digest = _recorded_checkpoint_digest(best, summary.get("best_checkpoint_sha256"))
    id_artifact = _artifact(common.paths.resolve_id_analysis_dir(run_dir), role="id", task=task)
    run_digest = _text(summary.get("effective_config_digest"))

    def scoped_for(
        inspection: EvaluationArtifactInspection,
        *,
        role: transient_artifact.DatasetRole,
        dataset_name: str | None,
    ) -> EvaluationArtifactInspection:
        """Use one unambiguous exact scoped artifact only when canonical is missing."""
        if inspection.state != "missing" or dataset_name is None or best_digest is None or run_digest is None:
            return inspection
        matches = tuple(
            candidate
            for candidate in scoped_candidates
            if candidate.task == task
            and candidate.run_name == run_name
            and candidate.effective_config_digest == run_digest
            and candidate.checkpoint_sha256 == best_digest
            and candidate.dataset_name == dataset_name
            and candidate.inspection.role == role
        )
        return matches[0].inspection if len(matches) == 1 else inspection

    id_artifact = scoped_for(
        id_artifact,
        role="id",
        dataset_name=_text(data_mapping.get("train_dataset")),
    )
    ood_artifacts: dict[str, EvaluationArtifactInspection] = {
        role: scoped_for(_artifact(common.paths.resolve_ood_analysis_dir(run_dir, role, name), role=role, task=task), role=role, dataset_name=name)
        for role, name in ood_names.items()
    }
    evaluable = status == "completed" and best_digest is not None
    action_enabled = (
        evaluable
        and not metadata_issues
        and any(artifact.state in {"ready", "scoped_partial"} for artifact in (id_artifact, *ood_artifacts.values()))
    )
    return EvaluationRunDiscovery(
        task=task,
        run_name=run_name,
        run_identity_sha256=run_digest,
        model_label=model_label,
        model_kind=_text(model_mapping.get("kind")),
        model_family=model_family,
        architecture=tuple(sorted((str(key), str(value)) for key, value in parameter_mapping.items())),
        seed=run.get("seed") if isinstance(run.get("seed"), int) and not isinstance(run.get("seed"), bool) else None,
        stage=stage,
        status=status,
        created_at=_text(summary.get("created_at")),
        updated_at=_text(summary.get("updated_at")),
        run_dir=run_dir,
        dataset_id=_text(data_mapping.get("train_dataset")),
        evaluation_protocol=_text(objective_mapping.get("id")),
        evaluation_config_identity=_evaluation_config_identity(task, evaluation_mapping),
        spatial_stride=data_mapping.get("spatial_stride")
        if isinstance(data_mapping.get("spatial_stride"), int) and not isinstance(data_mapping.get("spatial_stride"), bool)
        else None,
        parent_identity_sha256=None,
        parent_label=None,
        identity_format="independent",
        label=model_label,
        availability="available" if action_enabled else "unavailable",
        evaluable=evaluable,
        action_enabled=action_enabled,
        checkpoints=EvaluationCheckpointInspection(best, best_digest, latest),
        id_artifact=id_artifact,
        ood_dataset_ids=ood_dataset_ids,
        ood_artifacts=ood_artifacts,
        evaluation_assignments=assignments,
        metadata_issues=metadata_issues,
        config_label=f"{layout['family']}/{layout['config_name']}",
        pair_label=layout["pair"],
        a0_run_name=_text(_optional_mapping(training_mapping.get("teacher_handoff")).get("source_run_name")),
        # Full coverage names every planned held-out role; running it is their explicit release.
        artifact_command=artifact_loader.artifact_command(run_dir=run_dir, roles=("id", *ood_names)),
    )


def _reject_parent_record(message: str) -> None:
    """Reject malformed or contradictory immutable parent records."""
    raise ValueError(message)


def _parent_records(experiments_root: Path) -> tuple[dict[Path, Mapping[str, Any]], tuple[EvaluationDiscoveryIssue, ...]]:
    """Read immutable transient parents through admitted exact-path mappings."""
    children: dict[Path, Mapping[str, Any]] = {}
    conflicting: set[Path] = set()
    portable: set[Path] = set()
    issues: list[EvaluationDiscoveryIssue] = []
    for path in sorted(experiments_root.rglob("experiment.json")):
        try:
            located = run_identity.load_readonly_transient_experiment_record(path, experiments_root=experiments_root)
            record = located["record"]
            layout_root = located["output_root"]
            pending: dict[Path, Mapping[str, Any]] = {}
            for role, raw_child in record["children"].items():
                child = _mapping(raw_child, label="parent child")
                child_path = run_identity.resolve_readonly_experiment_child(
                    record,
                    child,
                    marker=path,
                    output_root=layout_root,
                )
                if role == "stage_a0":
                    continue
                remapped = not Path(child["path"]).is_absolute()
                existing = pending.get(child_path) or children.get(child_path)
                if child_path in conflicting or (
                    (remapped or child_path in portable)
                    and existing is not None
                    and existing["parent_identity_sha256"] != record["parent_identity_sha256"]
                ):
                    conflicting.add(child_path)
                    children.pop(child_path, None)
                    message = "Portable child has conflicting immutable parent identities."
                    _reject_parent_record(message)
                if remapped:
                    portable.add(child_path)
                pending[child_path] = record
            children.update(pending)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, yaml.YAMLError) as error:
            issues.append(EvaluationDiscoveryIssue(path, f"invalid parent record: {type(error).__name__}: {error}"))
    return children, tuple(issues)


def _ordered(runs: list[EvaluationRunDiscovery]) -> tuple[EvaluationRunDiscovery, ...]:
    """Order newest useful records first with stable exact-path ties."""
    return tuple(
        sorted(
            runs,
            key=lambda item: (
                item.availability == "available",
                item.updated_at or item.created_at or "",
                str(item.run_dir),
            ),
            reverse=True,
        )
    )


def _require_current_parent(record: EvaluationRunDiscovery, parent: Mapping[str, Any] | None) -> None:
    """Reject a transient leaf without its current parent Experiment record."""
    if record.task == "transient_drying" and record.stage in {"B", "A1"} and parent is None:
        message = "Current transient Evaluation run lacks its parent Experiment record."
        raise ValueError(message)


def discover_evaluation_runs(experiments_root: Path | str) -> EvaluationRunCatalog:
    """Discover all persisted Evaluation leaves below one experiments root read-only."""
    root = Path(experiments_root).expanduser().resolve(strict=False)
    if not root.exists():
        return EvaluationRunCatalog((), (), (), {}, {}, {}, {})
    parent_by_child, parent_issues = _parent_records(root)
    scoped_candidates, scoped_issues = _scoped_artifact_candidates(root)
    issues = [*parent_issues, *scoped_issues]
    records: list[EvaluationRunDiscovery] = []
    for config_path in sorted(root.rglob(common.paths.RUN_CONFIG_FILENAME)):
        run_dir = config_path.parent
        if not (run_dir / common.paths.RUN_SUMMARY_FILENAME).is_file():
            continue
        try:
            record = _run_from_leaf(run_dir.resolve(), scoped_candidates=scoped_candidates)
            if record is None:
                continue
            parent = parent_by_child.get(record.run_dir)
            _require_current_parent(record, parent)
            if parent is not None:
                record = replace(
                    record,
                    parent_identity_sha256=str(parent["parent_identity_sha256"]),
                    parent_label=str(parent["parent_label"]),
                    identity_format="grouped",
                    label=str(parent["parent_label"]),
                )
            records.append(record)
            issues.extend(EvaluationDiscoveryIssue(record.run_dir, reason) for reason in record.metadata_issues)
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as error:
            issues.append(EvaluationDiscoveryIssue(run_dir, f"invalid run candidate: {type(error).__name__}"))
    runs = _ordered(records)
    grouped: dict[str, list[EvaluationRunDiscovery]] = {}
    for record in runs:
        identity = record.parent_identity_sha256 or _independent_group_identity(record)
        grouped.setdefault(identity, []).append(record)
    groups = tuple(
        EvaluationRunGroup(identity, values[0].parent_label or values[0].label, values[0].task, tuple(values))
        for identity, values in sorted(grouped.items())
    )
    artifact_counts: Counter[str] = Counter(artifact.state for record in runs for artifact in record.artifacts())
    return EvaluationRunCatalog(
        runs,
        groups,
        tuple(issues),
        dict(Counter(record.task for record in runs)),
        dict(Counter(record.status for record in runs)),
        dict(Counter(record.parent_identity_sha256 or _independent_group_identity(record) for record in runs)),
        dict(artifact_counts),
    )


def _distinct(values: Iterable[object]) -> str | None:
    """Join the distinct values a run's attempts recorded, in order, or ``None`` when no attempt recorded one."""
    return "; ".join(dict.fromkeys(str(value) for value in values if value is not None)) or None


def _attempt_evidence(sessions: Iterable[Mapping[str, Any]], read: Callable[[Mapping[str, Any]], object]) -> str | None:
    """
    Join one runtime fact over the attempts that computed optimisation seconds or may have (no end entry).

    An attempt that computed none is skipped. A counted attempt that did not record the fact
    contributes ``unavailable``, so another attempt's value never reads as the run's complete evidence.
    """
    counted = (session for session in sessions if _optional_mapping(session.get("attempt")).get("train_seconds") != 0.0)
    return _distinct(UNAVAILABLE if (value := read(session)) is None else value for session in counted)


def _precision_text(precision: object) -> str | None:
    """Describe one attempt's recorded precision, or ``None`` when it ended before its model existed."""
    if not isinstance(precision, Mapping):
        return None
    flags = "/".join("on" if precision[f"tf32_{kind}"] else "off" for kind in ("matmul", "cudnn"))
    autotune = "on" if precision["cudnn_autotune"] else "off"
    dtypes = "+".join(precision["parameter_dtypes"])
    return f"{dtypes}, autocast {precision['autocast_dtype'] or 'off'}, TF32 matmul/cuDNN {flags}, cuDNN autotune {autotune}"


def stage_work_row(summary: Mapping[str, Any], *, inference_seconds_per_case: float | None = None) -> dict[str, Any]:
    """
    Return one finished training stage's end status and work quantities from its persisted summary.

    Parameters
    ----------
    summary : Mapping[str, Any]
        ``summary.json`` of one finished run, which is one stage (steady, A0, B or A1).
    inference_seconds_per_case : float | None, optional
        Mean inference seconds per case of the run's evaluation artifact, when it has a timing report.

    Returns
    -------
    dict[str, Any]
        End status (the training outcome's), optimisation, total and attempted seconds with their coverage, device, driver, CUDA
        build, precision, trainable parameters, optimizer steps, supervised and forward transitions,
        units, the full Train set's transitions and cases, k and memory batch, budget, overrun and inference time. A value the summary
        does not record is ``None``; device, driver, CUDA build and precision join the attempts that
        computed optimisation seconds, each without its value as ``unavailable``.

    Notes
    -----
    A steady unit is one case whose single forward pass is its supervised prediction, so its forward
    transitions are its supervised ones; a transient stage counts them in its terminal controller.

    """
    sessions = [session for session in summary.get("runtime_sessions", []) if isinstance(session, Mapping)]
    work = _optional_mapping(summary.get("training_work"))
    units = _optional_mapping(summary.get("training_units"))
    controller = _optional_mapping(summary.get("terminal_controller"))
    raw_budget = summary.get("terminal_budget")
    budget = None if raw_budget is None else stage_budget.StageBudget(**stage_budget.validate_state(raw_budget))
    steady = summary.get("task") == "steady_flow"
    return {
        "run_name": summary.get("run_name"),
        "stage": "steady" if steady else controller.get("arm"),
        # The training outcome is the one persisted end: completed, diverged or no_eligible_checkpoint.
        "status": _optional_mapping(summary.get("training_outcome")).get("status"),
        "model_kind": summary.get("model_kind"),
        "trainable_parameters": _optional_mapping(summary.get("model_parameter_counts")).get("trainable"),
        "optimisation_seconds": None if budget is None else budget.train_seconds,
        "total_seconds": summary.get("elapsed_seconds"),
        "attempted_seconds": summary.get("attempted_train_seconds"),
        "time_coverage": summary.get("duration_coverage"),
        "device": _attempt_evidence(sessions, lambda session: session.get("cuda_device_name") or session.get("device_type")),
        "driver": _attempt_evidence(sessions, lambda session: session.get("cuda_driver_version")),
        "cuda_build": _attempt_evidence(sessions, lambda session: session.get("cuda_build_version")),
        "precision": _attempt_evidence(sessions, lambda session: _precision_text(session.get("precision"))),
        "optimizer_steps": work.get("optimizer_steps"),
        "supervised_transitions": work.get("gradient_predictions"),
        "forward_transitions": work.get("gradient_predictions") if steady else controller.get("forward_transitions"),
        "unit_kind": units.get("unit_kind"),
        "seen_units": None if budget is None else budget.units,
        "train_units": units.get("train_units"),
        "full_set_units": units.get("full_set_units"),
        "full_set_transitions": units.get("full_set_transitions"),
        "seen_cases": units.get("seen_cases"),
        "train_cases": units.get("train_cases"),
        "units_per_update": units.get("units_per_update"),
        "memory_batch": _distinct(session.get("memory_batch") for session in sessions),
        "budget": None if budget is None else budget.total,
        "budget_unit": None if budget is None else budget.unit,
        "overrun": None if budget is None else budget.overrun,
        "inference_seconds_per_case": inference_seconds_per_case,
    }


def distinct_runs(frames: Mapping[str, pd.DataFrame]) -> dict[str, Mapping[str, Any]]:
    """Return the completed-run summary of each distinct run of the labelled frames (a run is its name and end time), keyed by its first frame."""
    runs: dict[tuple[object, object], tuple[str, Mapping[str, Any]]] = {}
    for label, frame in frames.items():
        summary = frame.attrs[dataframe.COMPLETED_RUN_SUMMARY_ATTR]
        runs.setdefault((summary.get("run_name"), summary.get("ended_at")), (label, summary))
    return dict(runs.values())


def _stage_placement(config: Mapping[str, Any], summary: Mapping[str, Any]) -> tuple[str, str, int]:
    """Return a completed run's stage, the pipeline it belongs to (its A0's run name) and the epoch it starts at."""
    # The run admission already validated the summary's terminal controller and teacher handoff.
    stage = str(stage_work_row(summary)["stage"])
    if stage not in _HANDOFF_STAGES:
        return stage, str(config["run"]["name"]), 0
    handoff = summary["teacher_handoff"]
    return stage, str(handoff["source_run_name"]), int(handoff["best_epoch"])


def _selected_point(best: Mapping[str, Any], last: Mapping[str, Any]) -> int:
    """Return the selected checkpoint's validation point: the first entry of the last checkpoint's objective history at its epoch and value."""
    selected = (best["best_epoch"], float(best["best_metric"]))
    for entry in last["objective_history"]:
        if entry["value"] is not None and (entry["epoch"], float(entry["value"])) == selected:
            return int(entry["point"])
    message = "The last checkpoint's objective history does not hold the selected checkpoint's epoch and value."
    raise ValueError(message)


def stage_history(run_dir: Path | str, *, expected_checkpoint_sha256: str) -> StageHistory | UnavailableHistory:
    """
    Read one completed stage's training history, or say why it is unavailable.

    The run is admitted first (``experiments.run.saved_artifact_run_lease``: completed, digests, identities and checkpoints, with
    the numerical environment its checkpoints recorded). Under that same lease its selected checkpoint's digest must be
    ``expected_checkpoint_sha256`` (the one the loaded artifact records), so a replaced run is refused, not read. The history is
    read under the lease by its own validator,
    bound to the run's task, name and checkpoint identity, to the completed epochs of its last checkpoint and to every entry of
    that checkpoint's objective history (point, epoch, eligibility, value), with ordered steps. The selected point is the
    first entry at the best checkpoint's epoch and value; B and A1 start at their A0 handoff epoch.

    The two tasks differ at admission. A transient run's admission already validates its history, so a transient stage
    without ``history.json`` or with a refused one fails the lease and raises ``RunLifecycleError``. A steady run is
    admitted without its history, so a steady run without one (trained before the durable history) or with one the
    validator refuses returns ``UnavailableHistory`` with the reason.

    Parameters
    ----------
    run_dir : Path | str
        Completed run leaf of one steady run or one transient stage (A0, B or A1).
    expected_checkpoint_sha256 : str
        Digest of the selected checkpoint the caller's loaded artifact records.

    Returns
    -------
    StageHistory | UnavailableHistory
        The admitted records with the stage, its pipeline and start epoch, the objective and the selected validation point.

    Raises
    ------
    experiments.run.RunLifecycleError
        If the run is not an admitted completed run, also when a transient stage's history is missing or refused.
    RuntimeError
        If the admitted run's selected checkpoint digest is not ``expected_checkpoint_sha256``.

    """
    with experiments.run.saved_artifact_run_lease(run_dir) as admitted:
        if admitted["selected_checkpoint_sha256"] != expected_checkpoint_sha256:
            message = (
                f"Run {Path(admitted['run_dir']).name!r} selects checkpoint {admitted['selected_checkpoint_sha256']}, "
                f"not the expected {expected_checkpoint_sha256}; load the run again."
            )
            raise RuntimeError(message)
        path, config, last = Path(admitted["run_dir"]), admitted["config"], admitted["last_checkpoint"]
        identity, best = admitted["checkpoint_identity"], admitted["best_checkpoint"]
        try:
            history = learning.training.history.validate_completed_history(
                path,
                task=str(config["task"]),
                run_name=str(config["run"]["name"]),
                checkpoint_identity=identity,
                completed_epoch=int(last["completed_epoch"]),
                validations=last["objective_history"],
            )
        except FileNotFoundError as error:
            return UnavailableHistory(path, f"{error} Runs trained before the durable training history have none.")
        except (TypeError, ValueError) as error:
            return UnavailableHistory(path, f"Training history refused: {error}")
    stage, pipeline, start_epoch = _stage_placement(config, admitted["summary"])
    return StageHistory(
        run_name=str(config["run"]["name"]),
        stage=stage,
        model_family=config_naming.model_family(config),
        pipeline=pipeline,
        start_epoch=start_epoch,
        objective_id=str(identity["objective"]["id"]),
        objective_direction=str(identity["objective"]["direction"]),
        selected_point=_selected_point(best, last),
        epochs=tuple(dict(record) for record in history["epochs"]),
        validations=tuple(dict(record) for record in history["validations"]),
    )


def _finite_or_none(value: object, *, label: str) -> float | None:
    """Return one finite study value, or ``None`` for a trial without one."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        message = f"{label} must be a finite number or null, got {value!r}."
        raise ValueError(message)
    return float(value)


def _study_trial(record: object, *, previous: int | None) -> StudyTrial:
    """Admit one trial record after the trial numbered ``previous``."""
    trial = _mapping(record, label="study_summary.json trial")
    number, status = trial.get("number"), trial.get("status")
    if isinstance(number, bool) or not isinstance(number, int) or (previous is not None and number <= previous) or _text(status) is None:
        message = "Study summary trials need ascending integer numbers and a status text."
        raise ValueError(message)
    if not {"value", "best_objective_so_far"} <= trial.keys():
        message = f"Study summary trial {number} must record its value and best_objective_so_far, null when it has none."
        raise ValueError(message)
    return StudyTrial(
        number=number,
        status=str(status),
        value=_finite_or_none(trial["value"], label=f"Trial {number} value"),
        best_so_far=_finite_or_none(trial["best_objective_so_far"], label=f"Trial {number} best_objective_so_far"),
    )


def _require_running_best(trials: Iterable[StudyTrial], *, direction: str) -> None:
    """Require every trial's best so far to be the producer's: the strictly better values of completed and duplicate trials so far."""
    best: float | None = None
    minimize = direction == "minimize"
    for trial in trials:
        value = trial.value
        # Strictly better only, so the earlier trial keeps a tie.
        if value is not None and trial.status in STUDY_FINISHED_STATUSES and (best is None or (value < best if minimize else value > best)):
            best = value
        if trial.best_so_far != best:
            message = f"Study summary trial {trial.number} best_objective_so_far {trial.best_so_far!r} is not the running best {best!r}."
            raise ValueError(message)


def read_study_summary(path: Path | str) -> StudySummary:
    """
    Read the study name, objective, trial counts and trials of one Optuna ``study_summary.json``.

    Only the keys the study curve draws are checked; the tuning service owns the full schema. Every trial records its value
    and best so far (null when it has none). The counts must equal the trials' statuses, so the curve's points and its
    counts describe the same trials, and every best so far must be the running best of the completed and duplicate trials
    in the objective's direction, as the tuning service derives it.

    Raises
    ------
    TypeError
        If the summary, its objective, counts or a trial is not a mapping, or its trials are not a list.
    ValueError
        If a checked key is missing or invalid, or the counts disagree with the trials.

    """
    payload = _read_json(Path(path))
    objective = _mapping(payload.get("objective"), label="study_summary.json objective")
    study_name, objective_id, direction = _text(payload.get("study_name")), _text(objective.get("id")), objective.get("direction")
    if study_name is None or objective_id is None or direction not in _OBJECTIVE_DIRECTIONS:
        message = "Study summary needs a study name, an objective id and a direction minimize or maximize."
        raise ValueError(message)
    counts = _mapping(payload.get("trial_counts"), label="study_summary.json trial_counts")
    if any(not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0 for key, count in counts.items()):
        message = "Study summary trial_counts must map status texts to non-negative integers."
        raise ValueError(message)
    records = payload.get("trials")
    if not isinstance(records, list):
        message = "Study summary trials must be a list."
        raise TypeError(message)
    trials: list[StudyTrial] = []
    for record in records:
        trials.append(_study_trial(record, previous=trials[-1].number if trials else None))
    if Counter(trial.status for trial in trials) != Counter({status: count for status, count in counts.items() if count}):
        message = "Study summary trial_counts disagree with the statuses of its trials."
        raise ValueError(message)
    _require_running_best(trials, direction=str(direction))
    return StudySummary(study_name, objective_id, str(direction), dict(counts), tuple(trials))
