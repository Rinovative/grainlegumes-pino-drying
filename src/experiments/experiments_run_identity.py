"""
experiments_run_identity.py

Build and publish exact experiment identity evidence apart from human labels.

Responsibilities:
  - Digest complete resolved run configurations and authored config sources
  - Record exact Dataset IDs, logical references, revisions, and repository provenance
  - Bind transient Stage A0 and Stage B leaves under one immutable parent record
  - Recover an exact childless parent after a failed pre-allocation launch
  - Reject reuse of one parent label and run revision for different exact inputs

Design principles:
  - Human labels locate records but never substitute for exact identity evidence
  - Repository state is provenance and is excluded from the parent identity digest
  - Parent metadata references child artifacts without copying or linking them

This module does NOT:
  - Allocate child run leaves, execute training, or admit checkpoints
  - Change legacy run layouts or infer Dataset or run revisions
  - Rewrite an existing experiment record
"""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from src import common

from .config import experiments_config_loader as config_loader

EXPERIMENT_RECORD_SCHEMA_KIND: Final = "vp2_transient_experiment"
EXPERIMENT_RECORD_SCHEMA_VERSION: Final = 1
_RUN_IDENTITY_SCHEMA_VERSION: Final = 1
_RECORD_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "task",
        "parent_label",
        "parent_identity_sha256",
        "run_revision",
        "seed",
        "authored_config",
        "dataset_identity",
        "children",
        "handoff",
        "source_repository",
    }
)


def resolved_config_digest(config: Mapping[str, Any]) -> str:
    """Return the complete resolved-config identity digest."""
    return common.serialization.canonical_json_sha256(copy.deepcopy(dict(config)))


def source_repository_evidence() -> dict[str, str | bool | None]:
    """Return bounded read-only Git commit and dirty-state provenance."""
    executable = shutil.which("git")
    if executable is None:
        return {"commit": None, "dirty": None}
    try:
        revision = subprocess.run(  # noqa: S603 -- trusted absolute executable and fixed arguments
            [executable, "rev-parse", "HEAD"],
            cwd=common.paths.get_project_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(  # noqa: S603 -- trusted absolute executable and fixed arguments
            [executable, "status", "--porcelain", "--untracked-files=normal"],
            cwd=common.paths.get_project_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"commit": commit or None, "dirty": dirty}


def authored_config_evidence(config_path: Path | str) -> dict[str, str]:
    """Return exact source path, basename, and content hash for one authored YAML."""
    requested = Path(config_path).expanduser()
    if requested.is_symlink():
        raise ValueError(f"Authored experiment config is missing or unsafe: {requested}.")
    source = requested.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Authored experiment config is missing or unsafe: {source}.")
    project_root = common.paths.get_project_root().expanduser().resolve(strict=False)
    try:
        label = source.relative_to(project_root).as_posix()
    except ValueError:
        label = str(source)
    return {
        "path": label,
        "basename": source.name,
        "sha256": common.serialization.file_sha256(source),
    }


def dataset_identity_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact Dataset selections and their optional immutable logical records."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("Resolved experiment config must contain data identity.")
    train = common.paths.validate_logical_name(data.get("train_dataset"), label="data.train_dataset")
    raw_ood = data.get("ood_datasets")
    if not isinstance(raw_ood, list):
        raise TypeError("Resolved data.ood_datasets must be a list.")
    ood = [common.paths.validate_logical_name(value, label="data.ood_datasets") for value in raw_ood]
    references = copy.deepcopy(data.get("dataset_references"))
    return {"train_dataset": train, "ood_datasets": ood, "references": references}


def run_identity_evidence(config: Mapping[str, Any], *, config_path: Path | str) -> dict[str, Any]:
    """Build one exact run identity record for local summary publication."""
    run = config.get("run")
    tracking = config.get("tracking")
    wandb = tracking.get("wandb") if isinstance(tracking, Mapping) else None
    if not isinstance(run, Mapping) or not isinstance(wandb, Mapping):
        raise TypeError("Resolved experiment config must contain run and tracking identity.")
    return {
        "schema_version": _RUN_IDENTITY_SCHEMA_VERSION,
        "parent_label": config_loader.generate_parent_experiment_label(config),
        "run_name": run["name"],
        "resolved_config_sha256": resolved_config_digest(config),
        "authored_config": authored_config_evidence(config_path),
        "dataset_identity": dataset_identity_evidence(config),
        "seed": run["seed"],
        "run_revision": run.get("revision", 0),
        "wandb_metric_schema_version": wandb.get("metric_schema_version", 1),
        "source_repository": source_repository_evidence(),
    }


def _parent_record_payload(
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> dict[str, Any]:
    """Build the deterministic identity-bearing portion of one parent record."""
    task = common.paths.validate_logical_name(stage_a.get("task"), label="task")
    if stage_b.get("task") != task or task != "transient_drying":
        raise ValueError("Transient parent children must share task='transient_drying'.")
    parent_a = config_loader.generate_parent_experiment_label(stage_a)
    parent_b = config_loader.generate_parent_experiment_label(stage_b)
    if parent_a != parent_b:
        raise ValueError("Transient Stage A0 and Stage B must share one parent label.")
    run_a = stage_a.get("run")
    run_b = stage_b.get("run")
    paths_a = stage_a.get("paths")
    paths_b = stage_b.get("paths")
    if not isinstance(run_a, Mapping) or not isinstance(run_b, Mapping) or not isinstance(paths_a, Mapping) or not isinstance(paths_b, Mapping):
        raise TypeError("Transient parent children require resolved run and path mappings.")
    output_root = Path(paths_a["output_root"]).expanduser()
    if Path(paths_b["output_root"]).expanduser() != output_root:
        raise ValueError("Transient parent children must share one output root.")
    a_name = common.paths.validate_logical_name(run_a["name"], label="Stage A run name")
    b_name = common.paths.validate_logical_name(run_b["name"], label="Stage B run name")
    a_path = common.paths.resolve_run_output_dir(task, a_name, output_root=output_root)
    b_path = common.paths.resolve_run_output_dir(task, b_name, output_root=output_root)
    a_digest = resolved_config_digest(stage_a)
    b_digest = resolved_config_digest(stage_b)
    return {
        "schema_kind": EXPERIMENT_RECORD_SCHEMA_KIND,
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "task": task,
        "parent_label": parent_a,
        "run_revision": run_a.get("revision", 0),
        "seed": run_a["seed"],
        "authored_config": authored_config_evidence(config_path),
        "dataset_identity": dataset_identity_evidence(stage_a),
        "children": {
            "stage_a0": {
                "run_name": a_name,
                "path": str(a_path),
                "resolved_config_sha256": a_digest,
            },
            "stage_b": {
                "run_name": b_name,
                "path": str(b_path),
                "resolved_config_sha256": b_digest,
            },
        },
        "handoff": {
            "source_run_name": a_name,
            "target_run_name": b_name,
            "source_resolved_config_sha256": a_digest,
            "target_resolved_config_sha256": b_digest,
            "checkpoint_contract": "stage_a_best_checkpoint_to_stage_b_teacher_handoff",
        },
    }


def build_transient_experiment_record(
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> dict[str, Any]:
    """Build one complete transient parent record with non-semantic repository provenance."""
    payload = _parent_record_payload(stage_a, stage_b, config_path=config_path)
    return {
        **payload,
        "parent_identity_sha256": common.serialization.canonical_json_sha256(payload),
        "source_repository": source_repository_evidence(),
    }


def validate_transient_experiment_record(value: object) -> dict[str, Any]:
    """Validate one parent record and recompute its exact identity digest."""
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise ValueError("Transient experiment record keys do not match the current schema.")
    if value["schema_kind"] != EXPERIMENT_RECORD_SCHEMA_KIND or value["schema_version"] != EXPERIMENT_RECORD_SCHEMA_VERSION:
        raise ValueError("Transient experiment record schema is unsupported.")
    payload = {key: copy.deepcopy(value[key]) for key in value if key not in {"parent_identity_sha256", "source_repository"}}
    expected = common.serialization.canonical_json_sha256(payload)
    if value["parent_identity_sha256"] != expected:
        raise ValueError("Transient experiment parent identity digest does not match its payload.")
    common.paths.validate_logical_name(value["task"], label="task")
    common.paths.validate_logical_name(value["parent_label"], label="parent_label")
    if isinstance(value["run_revision"], bool) or not isinstance(value["run_revision"], int) or value["run_revision"] < 0:
        raise ValueError("Transient experiment run_revision must be an integer >= 0.")
    return copy.deepcopy(value)


def experiment_record_path(record: Mapping[str, Any], *, output_root: Path | str) -> Path:
    """Return the visible parent metadata path for one transient experiment."""
    task = common.paths.validate_logical_name(record.get("task"), label="task")
    parent = common.paths.validate_logical_name(record.get("parent_label"), label="parent_label")
    return Path(output_root).expanduser() / task / "runs" / parent / "experiment.json"


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one JSON record without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = f"{json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)}\n".encode()
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _transient_child_exists(record: Mapping[str, Any]) -> bool:
    """Return whether either exact child leaf already exists or is a symlink."""
    children = record.get("children")
    if not isinstance(children, Mapping):
        raise TypeError("Transient experiment children metadata is invalid.")
    for key in ("stage_a0", "stage_b"):
        child = children.get(key)
        child_path = child.get("path") if isinstance(child, Mapping) else None
        if not isinstance(child_path, str) or not child_path:
            raise TypeError("Transient experiment child path metadata is invalid.")
        path = Path(child_path)
        if path.exists() or path.is_symlink():
            return True
    return False


def _publish_transient_experiment_record(
    record: Mapping[str, Any],
    *,
    output_root: Path | str,
    reuse_matching_without_children: bool,
) -> Path:
    """Publish one parent, optionally recovering an exact childless marker."""
    requested = validate_transient_experiment_record(dict(record))
    path = experiment_record_path(requested, output_root=output_root)
    lock_path = common.paths.resolve_run_lock_path(path.parent)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if path.exists() or path.is_symlink():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Transient experiment record is unsafe: {path}.")
            existing = validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))
            if existing["parent_identity_sha256"] == requested["parent_identity_sha256"]:
                if reuse_matching_without_children and not _transient_child_exists(requested):
                    return path
                raise FileExistsError(f"Matching experiment already exists for {requested['parent_label']!r}; use explicit resume.")
            raise FileExistsError(
                "Run revision conflict: "
                f"experiment={requested['parent_label']!r}, revision={requested['run_revision']}, "
                f"existing config identity={existing['parent_identity_sha256']}, "
                f"requested config identity={requested['parent_identity_sha256']}. "
                "Resume the matching run or set a new explicit run.revision."
            )
        try:
            _atomic_create_json(path, requested)
        except FileExistsError as error:
            raise FileExistsError(f"Transient experiment parent was concurrently published: {path}.") from error
    return path


def publish_transient_experiment_record(
    record: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> Path:
    """Publish one immutable parent record or reject exact reuse and identity conflict."""
    return _publish_transient_experiment_record(
        record,
        output_root=output_root,
        reuse_matching_without_children=False,
    )


def admit_fresh_transient_experiment_record(
    record: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> Path:
    """Publish a fresh parent or recover its exact childless pre-allocation marker."""
    return _publish_transient_experiment_record(
        record,
        output_root=output_root,
        reuse_matching_without_children=True,
    )


def load_transient_experiment_record(
    task: str,
    parent_label: str,
    *,
    output_root: Path | str,
) -> tuple[Path, dict[str, Any]]:
    """Load and validate one visible transient parent record without child mutation."""
    path = experiment_record_path(
        {"task": task, "parent_label": parent_label},
        output_root=output_root,
    )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Transient experiment parent record is missing or unsafe: {path}.")
    return path, validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))


def validate_persisted_transient_experiment_record(
    requested: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> Path:
    """Require an existing parent record to match an explicit-resume request exactly."""
    expected = validate_transient_experiment_record(dict(requested))
    path = experiment_record_path(expected, output_root=output_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Transient experiment parent record is missing or unsafe: {path}.")
    actual = validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))
    if actual["parent_identity_sha256"] != expected["parent_identity_sha256"]:
        raise ValueError("Explicit resume does not match the persisted transient parent identity.")
    return path
