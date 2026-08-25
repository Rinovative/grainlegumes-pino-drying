"""
evaluation_run_discovery.py

Discover persisted Evaluation runs without opening models or numerical arrays.

Responsibilities:
  - Recursively identify config-and-summary run leaves below an experiments root
  - Bind current transient children through validated immutable parent records
  - Inspect checkpoint and analysis-artifact availability without mutation
  - Admit available artifacts through exact run, Dataset, and payload evidence
  - Return deterministic catalog evidence and bounded per-candidate issues

Design principles:
  - Persisted config, summary, and parent records remain authoritative
  - Discovery never contacts tracking services or infers scientific identity
  - One malformed candidate cannot prevent inspection of unrelated runs

This module does NOT:
  - Load models, source datasets, numerical arrays, widgets, plots, or notebooks
  - Generate, repair, publish, or mutate any persisted evidence
  - Change legacy layouts or use directory names as scientific identity
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from src import common, datasets
from src.analysis.artifacts import analysis_artifact_contracts as artifact_contracts
from src.experiments import experiments_run_identity as run_identity

from . import evaluation_transient_artifact as transient_artifact

ArtifactState = Literal["ready", "generating", "scoped_partial", "missing", "invalid"]
RunAvailability = Literal["available", "unavailable"]

_EXACT_DATASET_DIGEST_TOKEN = re.compile(r"[0-9a-f]{16,64}\Z")
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class EvaluationArtifactInspection:
    """Describe one read-only ID or OOD artifact target."""

    role: Literal["id", "ood"]
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
    """Describe exact best and latest checkpoint presence and byte digests."""

    best_path: Path
    best_digest: str | None
    latest_path: Path
    latest_digest: str | None


@dataclass(frozen=True, slots=True)
class EvaluationRunDiscovery:
    """Describe one persisted run leaf selected only from saved metadata."""

    task: str
    run_name: str
    run_revision: int | None
    run_identity_sha256: str | None
    model_label: str
    model_kind: str | None
    architecture: tuple[tuple[str, str], ...]
    seed: int | None
    stage: str | None
    status: str
    created_at: str | None
    updated_at: str | None
    run_dir: Path
    dataset_id: str | None
    dataset_label: str | None
    dataset_reference: str | None
    dataset_revision: int | None
    ood_dataset_ids: tuple[str, ...]
    evaluation_protocol: str | None
    evaluation_config_identity: str | None
    spatial_stride: int | None
    parent_identity_sha256: str | None
    parent_label: str | None
    identity_format: Literal["grouped", "legacy"]
    label: str
    availability: RunAvailability
    evaluable: bool
    action_enabled: bool
    checkpoints: EvaluationCheckpointInspection
    id_artifact: EvaluationArtifactInspection
    ood_artifact: EvaluationArtifactInspection
    artifact_command: str


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
    """Derive one concise architecture label from persisted model metadata."""
    model = config.get("model")
    if not isinstance(model, Mapping):
        return "Model"
    kind = _text(model.get("kind")) or _text(model.get("name")) or "model"
    parameters = model.get("params")
    if not isinstance(parameters, Mapping):
        return kind.upper()
    parts = [kind.upper()]
    modes_x = parameters.get("modes_x")
    modes_y = parameters.get("modes_y")
    if all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (modes_x, modes_y)):
        parts.append(f"m{modes_x}×{modes_y}")  # noqa: RUF001 -- human architecture label
    hidden = parameters.get("hidden_channels")
    if isinstance(hidden, int) and not isinstance(hidden, bool) and hidden > 0:
        parts.append(f"h{hidden}")
    layers = parameters.get("n_layers")
    if isinstance(layers, int) and not isinstance(layers, bool) and layers > 0:
        parts.append(f"l{layers}")
    ratio = parameters.get("mode_ratio")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        parts.append(f"r{float(ratio):.3f}")
    return " ".join(parts)


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
    """Return whether one marker has recognized schema, task, and output paths."""
    artifact_schema = payload.get("artifact_schema_version")
    if payload.get("provenance_schema_version") != artifact_contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION:
        return False
    run = payload.get("run")
    run_task = _text(run.get("task")) if isinstance(run, Mapping) else None
    if task not in {value for value in (_text(payload.get("task")), run_task) if value is not None}:
        return False
    expected_schema = (
        transient_artifact.TRANSIENT_SEQUENCE_SCHEMA_VERSION
        if payload.get("artifact_kind") == transient_artifact.TRANSIENT_ARTIFACT_KIND
        else artifact_contracts.ARTIFACT_SCHEMA_VERSION
    )
    return artifact_schema == expected_schema and _manifest_paths_are_present(
        payload.get("outputs"),
        root=root,
    )


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


def _artifact(root: Path, *, role: Literal["id", "ood"], task: str) -> EvaluationArtifactInspection:
    """Inspect one canonical marker, staging lease, and payload inventory without arrays."""
    stages = tuple(candidate for candidate in root.parent.glob(f".{root.name}.staging.*") if candidate.is_dir() and not candidate.is_symlink())
    if not root.exists():
        if stages:
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
        state = "generating" if stages and _artifact_writer_active(root) else "invalid"
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
    capabilities = {"aggregate"}
    if role == "id":
        capabilities.add("case_fields")
    if payload.get("artifact_kind") == "transient_sequence":
        capabilities.update({"trajectory", "rollout", "timing", "case_fields"})
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
        or role not in {"id", "ood"}
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


def _checkpoint(path: Path) -> str | None:
    """Return a checkpoint byte digest only for an exact regular file."""
    if not path.is_file() or path.is_symlink():
        return None
    return common.serialization.file_sha256(path)


def _stage_label(training: Mapping[str, Any]) -> str | None:
    """Normalize persisted comparison arms before generic training stages."""
    arm = _text(training.get("comparison_arm"))
    if arm is not None:
        normalized = arm.lower()
        if normalized == "a0":
            return "A0"
        if normalized == "a_plus":
            return "A+"
        if normalized == "b":
            return "B"
        return "A"
    stage = _text(training.get("stage"))
    return stage.upper() if stage is not None else None


def _legacy_group_identity(record: EvaluationRunDiscovery) -> str:
    """Return a stable exact synthetic group identity for one legacy leaf."""
    return common.serialization.canonical_json_sha256({"task": record.task, "run_name": record.run_name, "run_dir": str(record.run_dir)})


def dataset_display_projection(
    task: str,
    dataset_id: str,
) -> str:
    """Remove task and digest tokens from one exact Dataset identity for display."""
    resolved_task = common.paths.validate_logical_name(
        task,
        label="Evaluation task display projection",
    )
    resolved_dataset = common.paths.validate_logical_name(
        dataset_id,
        label="Evaluation Dataset display projection",
    )
    parts = resolved_dataset.split("__")
    if parts and parts[0] == resolved_task:
        parts.pop(0)
    if len(parts) > 1 and _EXACT_DATASET_DIGEST_TOKEN.fullmatch(parts[-1]) is not None:
        parts.pop()
    return common.paths.validate_logical_name(
        "_".join(parts),
        label="Evaluation Dataset display label",
    )


def _dataset_label(
    task: str,
    data: Mapping[str, Any],
    train_reference: Mapping[str, Any],
) -> str | None:
    """Return a concise persisted Dataset label without exact digest tokens."""
    if train_reference:
        name = common.paths.validate_logical_name(
            train_reference.get("name"),
            label="config.yaml data.dataset_references.train.name",
        )
        revision = train_reference.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            message = "config.yaml data.dataset_references.train.revision must be a non-negative integer."
            raise ValueError(message)
        return name if revision == 0 else f"{name}_d{revision}"

    raw_dataset_id = data.get("train_dataset")
    if raw_dataset_id is None:
        return None
    dataset_id = common.paths.validate_logical_name(
        raw_dataset_id,
        label="config.yaml data.train_dataset",
    )
    return dataset_display_projection(task, dataset_id)


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
) -> EvaluationRunDiscovery:
    """Build one leaf record from its authoritative config and summary files."""
    config = _read_yaml(run_dir / common.paths.RUN_CONFIG_FILENAME)
    summary = _read_json(run_dir / common.paths.RUN_SUMMARY_FILENAME)
    task = _text(config.get("task"))
    run = _mapping(config.get("run"), label="config.yaml run")
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
    reference_mapping = _optional_mapping(data_mapping.get("dataset_references"))
    train_reference = _optional_mapping(reference_mapping.get("train"))
    ood_raw = data_mapping.get("ood_datasets")
    ood = tuple(value for value in ood_raw if isinstance(value, str)) if isinstance(ood_raw, list) else ()
    training_mapping = _optional_mapping(config.get("training"))
    model_mapping = _optional_mapping(config.get("model"))
    parameter_mapping = _optional_mapping(model_mapping.get("params"))
    evaluation_mapping = _optional_mapping(config.get("evaluation"))
    objective_mapping = _optional_mapping(evaluation_mapping.get("objective"))
    stage = _stage_label(training_mapping)
    best = run_dir / common.paths.RUN_BEST_CHECKPOINT_FILENAME
    latest = run_dir / common.paths.RUN_LAST_CHECKPOINT_FILENAME
    best_digest = _checkpoint(best)
    id_root = common.paths.resolve_id_analysis_dir(run_dir)
    ood_identity = datasets.contracts.identity.combined_dataset_id(ood) if ood else None
    ood_root = (
        common.paths.resolve_ood_analysis_dir(run_dir, ood_identity)
        if ood_identity is not None
        else common.paths.resolve_analysis_root(run_dir) / "ood"
    )
    id_artifact = _artifact(id_root, role="id", task=task)
    ood_artifact = _artifact(ood_root, role="ood", task=task)
    run_digest = _text(summary.get("effective_config_digest"))

    def scoped_for(
        inspection: EvaluationArtifactInspection,
        *,
        role: Literal["id", "ood"],
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
    ood_artifact = scoped_for(
        ood_artifact,
        role="ood",
        dataset_name=ood_identity,
    )
    evaluable = status == "completed" and best_digest is not None
    action_enabled = evaluable and any(artifact.state in {"ready", "scoped_partial"} for artifact in (id_artifact, ood_artifact))
    return EvaluationRunDiscovery(
        task=task,
        run_name=run_name,
        run_revision=run.get("revision") if isinstance(run.get("revision"), int) and not isinstance(run.get("revision"), bool) else None,
        run_identity_sha256=run_digest,
        model_label=_model_label(config),
        model_kind=_text(model_mapping.get("kind")),
        architecture=tuple(sorted((str(key), str(value)) for key, value in parameter_mapping.items())),
        seed=run.get("seed") if isinstance(run.get("seed"), int) and not isinstance(run.get("seed"), bool) else None,
        stage=stage,
        status=status,
        created_at=_text(summary.get("created_at")),
        updated_at=_text(summary.get("updated_at")),
        run_dir=run_dir,
        dataset_id=_text(data_mapping.get("train_dataset")),
        dataset_label=_dataset_label(task, data_mapping, train_reference),
        dataset_reference=_text(train_reference.get("id")) or _text(train_reference.get("name")),
        dataset_revision=train_reference.get("revision")
        if isinstance(train_reference.get("revision"), int) and not isinstance(train_reference.get("revision"), bool)
        else None,
        ood_dataset_ids=ood,
        evaluation_protocol=_text(objective_mapping.get("id")),
        evaluation_config_identity=_evaluation_config_identity(task, evaluation_mapping),
        spatial_stride=data_mapping.get("spatial_stride")
        if isinstance(data_mapping.get("spatial_stride"), int) and not isinstance(data_mapping.get("spatial_stride"), bool)
        else None,
        parent_identity_sha256=None,
        parent_label=None,
        identity_format="legacy",
        label=_model_label(config),
        availability="available" if action_enabled else "unavailable",
        evaluable=evaluable,
        action_enabled=action_enabled,
        checkpoints=EvaluationCheckpointInspection(best, best_digest, latest, _checkpoint(latest)),
        id_artifact=id_artifact,
        ood_artifact=ood_artifact,
        artifact_command="./scripts/docker_job.sh artifacts --run-dir " + shlex.quote(str(run_dir)),
    )


def _parent_records(experiments_root: Path) -> tuple[dict[Path, Mapping[str, Any]], tuple[EvaluationDiscoveryIssue, ...]]:
    """Read valid transient parents keyed by their exact declared child paths."""
    children: dict[Path, Mapping[str, Any]] = {}
    issues: list[EvaluationDiscoveryIssue] = []
    for path in sorted(experiments_root.rglob("experiment.json")):
        try:
            record = run_identity.validate_transient_experiment_record(dict(_read_json(path)))
            for child in record["children"].values():
                child_path = Path(child["path"]).expanduser().resolve(strict=False)
                children[child_path] = record
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:  # noqa: PERF203
            issues.append(EvaluationDiscoveryIssue(path, f"invalid parent record: {type(error).__name__}"))
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
            record = _run_from_leaf(
                run_dir.resolve(),
                scoped_candidates=scoped_candidates,
            )
            parent = parent_by_child.get(record.run_dir)
            if parent is not None:
                record = replace(
                    record,
                    parent_identity_sha256=str(parent["parent_identity_sha256"]),
                    parent_label=str(parent["parent_label"]),
                    identity_format="grouped",
                    label=str(parent["parent_label"]),
                )
            records.append(record)
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as error:
            issues.append(EvaluationDiscoveryIssue(run_dir, f"invalid run candidate: {type(error).__name__}"))
    runs = _ordered(records)
    grouped: dict[str, list[EvaluationRunDiscovery]] = {}
    for record in runs:
        identity = record.parent_identity_sha256 or _legacy_group_identity(record)
        grouped.setdefault(identity, []).append(record)
    groups = tuple(
        EvaluationRunGroup(identity, values[0].parent_label or values[0].label, values[0].task, tuple(values))
        for identity, values in sorted(grouped.items())
    )
    artifact_counts: Counter[str] = Counter(artifact.state for record in runs for artifact in (record.id_artifact, record.ood_artifact))
    return EvaluationRunCatalog(
        runs,
        groups,
        tuple(issues),
        dict(Counter(record.task for record in runs)),
        dict(Counter(record.status for record in runs)),
        dict(Counter(record.parent_identity_sha256 or _legacy_group_identity(record) for record in runs)),
        dict(artifact_counts),
    )
