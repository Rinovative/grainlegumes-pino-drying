"""Build and publish exact experiment identity evidence apart from human labels.

Responsibilities:
  - Digest complete resolved run configurations and authored config sources
  - Record exact Dataset IDs, content identities, grouping, and repository provenance
  - Read a frozen job's commit and dirty state from its source snapshot instead of Git
  - Name the frozen source tree the executing process runs in, inside a Slurm ML container or in tree mode
  - Bind matched A0/B/A1 comparison children under one immutable parent record
  - Recover an exact childless parent after a failed pre-allocation launch
  - Reject reuse of a content group with different exact inputs

Design principles:
  - Human labels locate records but never substitute for exact identity evidence
  - Repository state is provenance and is excluded from the parent identity digest
  - A Slurm ML job runs from a Git-free source tree, so its evidence comes from the submitted provenance and the retained snapshot
  - Parent metadata references child artifacts without copying or linking them

This module does NOT:
  - Allocate child run leaves, execute training, or admit checkpoints
  - Infer scientific compatibility from human labels
  - Rewrite an existing experiment record

Notes:
  Parent records live in a pair directory and refer to the seed group's shared A0
  as ``../a0`` and to the owned B/A1 leaves as ``b`` and ``a1``. Resolution admits
  only those exact paths within the declared scientific experiment group.
  Producing Git evidence is provenance, not a compatibility predicate or a claim
  that corresponding bytes are retained. Inside a Slurm ML job (ML_SLURM_PROVENANCE
  set) the commit is the submitted one and the run is dirty when the snapshot
  ``source_snapshots/<commit12>-<fingerprint12>/snapshot.json`` lists untracked
  files or a non-empty tracked diff; a missing snapshot fails.

"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from src import common
from src.learning.training import learning_training_checkpoint as training_checkpoint

from .config import experiments_config_loader as config_loader
from .config import experiments_config_naming as config_naming

EXPERIMENT_RECORD_SCHEMA_KIND: Final = "experiment_transient_parent"
SLURM_PROVENANCE_VARIABLE: Final = "ML_SLURM_PROVENANCE"
_SNAPSHOT_MANIFEST_FILENAME: Final = "snapshot.json"
_EMPTY_DIFF_SHA256: Final = hashlib.sha256(b"").hexdigest()
_TREE_NAME_PART_LENGTH: Final = 12
_SHARED_A0_PATH: Final = "../a0"
_RECORD_KEYS: Final = frozenset(
    {
        "schema_kind",
        "topology",
        "task",
        "parent_label",
        "parent_identity_sha256",
        "layout",
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


def source_tree_name(source_commit: object, source_fingerprint_sha256: object) -> str:
    """
    Return the ``<commit12>-<fingerprint12>`` name of a frozen source tree and its snapshot.

    Raises
    ------
    ValueError
        If the commit is not a full lowercase Git object ID or the fingerprint is not a lowercase SHA256.

    """
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        msg = f"Source commit must be a full lowercase 40-character Git object ID, got {source_commit!r}."
        raise ValueError(msg)
    if not isinstance(source_fingerprint_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_fingerprint_sha256) is None:
        msg = f"Source fingerprint must be a lowercase SHA256, got {source_fingerprint_sha256!r}."
        raise ValueError(msg)
    return f"{source_commit[:_TREE_NAME_PART_LENGTH]}-{source_fingerprint_sha256[:_TREE_NAME_PART_LENGTH]}"


def _submitted_source(raw_provenance: str) -> tuple[Any, Any]:
    """Return the unvalidated commit and fingerprint every worker operation records; ``source_tree_name`` validates them."""
    try:
        provenance = json.loads(raw_provenance)
    except json.JSONDecodeError as error:
        msg = f"{SLURM_PROVENANCE_VARIABLE} must contain valid JSON from the Slurm worker."
        raise ValueError(msg) from error
    if not isinstance(provenance, Mapping):
        msg = f"{SLURM_PROVENANCE_VARIABLE} must contain a JSON object."
        raise TypeError(msg)
    return provenance.get("source_commit"), provenance.get("source_worktree_sha256")


def _frozen_source_evidence(raw_provenance: str) -> dict[str, str | bool | None]:
    """Describe the frozen source a Slurm ML job runs from using its retained snapshot, never Git."""
    commit, fingerprint = _submitted_source(raw_provenance)
    manifest_path = common.paths.get_source_snapshots_root() / source_tree_name(commit, fingerprint) / _SNAPSHOT_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        msg = f"Source snapshot of the submitted job is missing: {manifest_path}"
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("git_commit") != commit
        or manifest.get("source_fingerprint_sha256") != fingerprint
        or not isinstance(manifest.get("untracked"), list)
    ):
        msg = f"Source snapshot does not match the submitted commit and fingerprint: {manifest_path}"
        raise ValueError(msg)
    return {"commit": commit, "dirty": bool(manifest["untracked"]) or manifest.get("tracked_diff_sha256") != _EMPTY_DIFF_SHA256}


def source_repository_evidence() -> dict[str, str | bool | None]:
    """
    Return bounded read-only commit and dirty-state provenance of the executing source.

    A Slurm ML job runs from a frozen source tree without Git, so inside one
    (``ML_SLURM_PROVENANCE`` set) the evidence comes from the submitted provenance
    and the retained snapshot; elsewhere it comes from the live Git checkout.

    Raises
    ------
    FileNotFoundError
        Inside a Slurm ML job if the source snapshot named by the provenance is missing.
    ValueError
        Inside a Slurm ML job if the provenance or its snapshot is malformed or inconsistent.

    """
    raw_provenance = os.environ.get(SLURM_PROVENANCE_VARIABLE)
    if raw_provenance is not None:
        return _frozen_source_evidence(raw_provenance)
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


def executing_source_tree_id() -> str | None:
    """
    Return the frozen source tree this process executes in, or ``None`` outside one.

    A Slurm ML job sees its tree only as the container's repository mount and never
    ``SOURCE_TREE_ID``, so it is named from the submitted commit and fingerprint that
    every worker operation records, artifact jobs included; any other process names
    it through ``SOURCE_TREE_ID``.

    Raises
    ------
    ValueError
        If the job's provenance or ``SOURCE_TREE_ID`` cannot name a tree.
    TypeError
        If the job's provenance is not a JSON object.

    """
    raw_provenance = os.environ.get(SLURM_PROVENANCE_VARIABLE)
    if raw_provenance is None:
        return common.paths.source_tree_id()
    return source_tree_name(*_submitted_source(raw_provenance))


def authored_config_evidence(config_path: Path | str) -> dict[str, str]:
    """Return exact source path, basename, and content hash for one authored YAML."""
    requested = Path(config_path).expanduser()
    if requested.is_symlink():
        msg = f"Authored experiment config is missing or unsafe: {requested}."
        raise ValueError(msg)
    source = requested.resolve(strict=True)
    if not source.is_file():
        msg = f"Authored experiment config is missing or unsafe: {source}."
        raise ValueError(msg)
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
        msg = "Resolved experiment config must contain data identity."
        raise TypeError(msg)
    train = common.paths.validate_logical_name(data.get("train_dataset"), label="data.train_dataset")
    selection_key = "material_ood_datasets"
    raw_selected = data.get(selection_key)
    if not isinstance(raw_selected, list):
        msg = f"Resolved data.{selection_key} must be a list."
        raise TypeError(msg)
    selected = [common.paths.validate_logical_name(value, label=f"data.{selection_key}") for value in raw_selected]
    return {"train_dataset": train, selection_key: selected}


def run_identity_evidence(config: Mapping[str, Any], *, config_path: Path | str) -> dict[str, Any]:
    """Build one exact run identity record for local summary publication."""
    run = config.get("run")
    if not isinstance(run, Mapping):
        msg = "Resolved experiment config must contain run identity."
        raise TypeError(msg)
    return {
        "schema_kind": "experiment_run_identity",
        "parent_label": config_naming.parent_label(run["layout"]),
        "run_name": run["name"],
        "resolved_config_sha256": resolved_config_digest(config),
        "authored_config": authored_config_evidence(config_path),
        "dataset_identity": dataset_identity_evidence(config),
        "seed": run["seed"],
        "layout": copy.deepcopy(run["layout"]),
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
        msg = "Transient parent children must share task='transient_drying'."
        raise ValueError(msg)
    parent_a = config_naming.parent_label(stage_a["run"]["layout"])
    parent_b = config_naming.parent_label(stage_b["run"]["layout"])
    if parent_a != parent_b:
        msg = "Transient Stage A0 and Stage B must share one parent label."
        raise ValueError(msg)
    run_a = stage_a.get("run")
    run_b = stage_b.get("run")
    paths_a = stage_a.get("paths")
    paths_b = stage_b.get("paths")
    if not isinstance(run_a, Mapping) or not isinstance(run_b, Mapping) or not isinstance(paths_a, Mapping) or not isinstance(paths_b, Mapping):
        msg = "Transient parent children require resolved run and path mappings."
        raise TypeError(msg)
    output_root = Path(paths_a["output_root"]).expanduser()
    if Path(paths_b["output_root"]).expanduser() != output_root:
        msg = "Transient parent children must share one output root."
        raise ValueError(msg)
    a_name = common.paths.validate_logical_name(run_a["name"], label="Stage A run name")
    b_name = common.paths.validate_logical_name(run_b["name"], label="Stage B run name")
    a_digest = training_checkpoint.resume_contract_digest(stage_a)
    b_digest = training_checkpoint.resume_contract_digest(stage_b)
    return {
        "schema_kind": EXPERIMENT_RECORD_SCHEMA_KIND,
        "topology": "a0_to_b",
        "task": task,
        "parent_label": parent_a,
        "layout": copy.deepcopy(run_b["layout"]),
        "seed": run_a["seed"],
        "authored_config": authored_config_evidence(config_path),
        "dataset_identity": dataset_identity_evidence(stage_a),
        "children": {
            "stage_a0": {
                "run_name": a_name,
                "path": Path(
                    os.path.relpath(common.paths.resolve_config_run_dir(stage_a), common.paths.resolve_config_branch_dir(stage_b))
                ).as_posix(),
                "semantic_config_sha256": a_digest,
            },
            "stage_b": {
                "run_name": b_name,
                "path": "b",
                "semantic_config_sha256": b_digest,
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
    stage_a1: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one complete matched A0/B/A1 parent record with non-semantic repository provenance."""
    payload = _parent_record_payload(stage_a, stage_b, config_path=config_path)
    parent = payload["parent_label"]
    if config_naming.parent_label(stage_a1["run"]["layout"]) != parent:
        msg = "Matched A1 must share the exact A0/B parent label."
        raise ValueError(msg)
    a1_name = common.paths.validate_logical_name(stage_a1["run"]["name"], label="Stage A1 run name")
    output_root = Path(stage_a1["paths"]["output_root"])
    if output_root != Path(stage_a["paths"]["output_root"]):
        msg = "Matched A1 output root differs from A0/B."
        raise ValueError(msg)
    continuation = stage_a1["training"]["comparison_arm"]
    payload["topology"] = f"a0_to_b_and_{continuation}"
    payload["children"][f"stage_{continuation}"] = {
        "run_name": a1_name,
        "path": continuation,
        "semantic_config_sha256": training_checkpoint.resume_contract_digest(stage_a1),
    }
    for key, config in (("stage_a0", stage_a), ("stage_b", stage_b)):
        payload["children"][key].pop("semantic_config_sha256")
        payload["children"][key]["semantic_config_sha256"] = training_checkpoint.resume_contract_digest(config)
    payload["handoff"] = {
        "source_run_name": stage_a["run"]["name"],
        "target_run_names": [stage_b["run"]["name"], a1_name],
        "initialization": "a0_best_full_common_fresh_controller",
    }
    identity_payload = {key: value for key, value in payload.items() if key != "authored_config"}
    return {
        **payload,
        "parent_identity_sha256": common.serialization.canonical_json_sha256(identity_payload),
        "source_repository": source_repository_evidence(),
    }


def validate_transient_experiment_record(value: object) -> dict[str, Any]:
    """Validate one parent record and recompute its exact identity digest."""
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        msg = "Transient experiment record keys do not match the current schema."
        raise ValueError(msg)
    if value["schema_kind"] != EXPERIMENT_RECORD_SCHEMA_KIND:
        msg = "Transient experiment record schema is unsupported."
        raise ValueError(msg)
    if value["topology"] != "a0_to_b_and_a1" or set(value["children"]) != {"stage_a0", "stage_b", "stage_a1"}:
        msg = "Transient experiment parent topology must be the matched A0 -> {B, A1} comparison."
        raise ValueError(msg)
    payload = {key: copy.deepcopy(value[key]) for key in value if key not in {"parent_identity_sha256", "source_repository", "authored_config"}}
    children = value["children"]
    for key, child in children.items():
        if not isinstance(child, Mapping) or set(child) != {"run_name", "path", "semantic_config_sha256"}:
            msg_0 = "Parent child metadata keys are invalid."
            raise ValueError(msg_0)
        common.paths.validate_logical_name(child["run_name"], label="parent child run_name")
        if not isinstance(child["semantic_config_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", child["semantic_config_sha256"]) is None:
            msg_0 = "Parent child requires an exact saved scientific identity."
            raise ValueError(msg_0)
        expected_path = _SHARED_A0_PATH if key == "stage_a0" else key.removeprefix("stage_")
        if child["path"] != expected_path:
            msg_0 = "Parent child path contradicts its shared A0/pair layout."
            raise ValueError(msg_0)
    expected = common.serialization.canonical_json_sha256(payload)
    if value["parent_identity_sha256"] != expected:
        msg = "Transient experiment parent identity digest does not match its payload."
        raise ValueError(msg)
    common.paths.resolve_config_run_dir({"run": {"layout": value["layout"]}}, output_root=Path())
    if value["layout"]["stage"] != "b":
        msg_0 = "Parent layout must be the matched B stage that owns the pair directory."
        raise ValueError(msg_0)
    if value["seed"] != value["layout"]["seed"]:
        msg_0 = "Parent seed disagrees with its saved layout."
        raise ValueError(msg_0)
    common.paths.validate_logical_name(value["task"], label="task")
    common.paths.validate_logical_name(value["parent_label"], label="parent_label")
    return copy.deepcopy(value)


def experiment_record_path(record: Mapping[str, Any], *, output_root: Path | str) -> Path:
    """Return the visible parent metadata path for one transient experiment."""
    layout = record["layout"]
    config = {"run": {"layout": layout}, "paths": {"output_root": str(output_root)}}
    return common.paths.resolve_config_branch_dir(config) / common.paths.EXPERIMENT_RECORD_FILENAME


def load_readonly_transient_experiment_record(
    marker: Path | str,
    *,
    experiments_root: Path | str | None = None,
) -> dict[str, Any]:
    """Read parent metadata and locate its actual root without model/tensor reads.

    ``experiments_root`` bounds catalog discovery. Explicit callers may locate a
    previously validated marker directly. No child completion or saved
    prediction admission follows from this step.
    """
    path = Path(marker).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        msg = f"Experiment parent must be one regular file: {path}"
        raise ValueError(msg)
    record = validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))
    output_root = common.paths.resolve_experiment_record_root(path, record["layout"], experiments_root=experiments_root)
    if experiment_record_path(record, output_root=output_root).resolve(strict=False) != path.resolve(strict=True):
        msg = "Parent metadata contradicts its physical layout."
        raise ValueError(msg)
    return {"record": record, "output_root": output_root}


def resolve_readonly_experiment_child(
    record: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    marker: Path,
    output_root: Path | str,
) -> Path:
    """Resolve saved child metadata without admitting model execution or completion."""
    experiments_root = Path(output_root).expanduser().resolve(strict=False)
    expected_marker = experiment_record_path(record, output_root=experiments_root)
    if marker.is_symlink() or marker.resolve(strict=True) != expected_marker.resolve(strict=False):
        msg = "Parent record is not at its declared experiment branch location."
        raise ValueError(msg)
    validated_record = validate_transient_experiment_record(dict(record))
    if child not in validated_record["children"].values():
        msg_0 = "Requested child is not declared by the immutable parent."
        raise ValueError(msg_0)
    declared = Path(child["path"])
    if declared.as_posix() not in {_SHARED_A0_PATH, "b", "a1"}:
        msg = "Parent child must identify the shared A0 or an owned branch leaf."
        raise ValueError(msg)
    group = common.paths.resolve_config_experiment_dir({"run": {"layout": record["layout"]}}, output_root=experiments_root).resolve(strict=False)
    expected_child = (
        group / "a0" if child is record["children"].get("stage_a0") or child == record["children"].get("stage_a0") else marker.parent / declared.name
    )
    # ``..`` must normalize exactly to the one declared shared A0.
    if (marker.parent / declared).resolve(strict=False) != expected_child.resolve(strict=False):
        msg = "Parent child does not identify its shared A0 or owned stage leaf."
        raise ValueError(msg)
    resolved = (marker.parent / declared).resolve(strict=False)
    common.paths.resolve_config_output_root(
        {"run": {"layout": {**record["layout"], "stage": "a0", "branch_id": None, "pair": None}}}
        if declared.name == "a0"
        else {"run": {"layout": {**record["layout"], "stage": declared.name}}},
        marker.parent / declared,
    )
    if not group.is_relative_to(experiments_root) or not resolved.is_relative_to(group):
        msg = "Parent child escapes its scientific experiment group."
        raise ValueError(msg)
    config_path = resolved / common.paths.RUN_CONFIG_FILENAME
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or not config_path.is_file():
            msg = "Parent child config must be a regular file."
            raise ValueError(msg)
        config = config_loader.validate_resolved_config(config_loader.load_yaml(config_path))
        if config["task"] != record["task"] or config["run"]["name"] != child["run_name"]:
            msg = "Parent child task/run identity contradicts its saved config."
            raise ValueError(msg)
        if common.paths.resolve_config_run_dir(config, output_root=experiments_root).resolve(strict=False) != resolved:
            msg = "Parent child layout contradicts its saved config."
            raise ValueError(msg)
        from . import experiments_run as run_service  # noqa: PLC0415 -- saved summary access avoids the run/identity import cycle

        summary_path = resolved / common.paths.RUN_SUMMARY_FILENAME
        if summary_path.is_symlink() or not summary_path.is_file():
            msg = "Parent child requires one regular saved summary."
            raise ValueError(msg)
        summary = run_service.read_run_summary(resolved)
        if summary.get("status") == "completed":
            if summary.get("config_sha256") != common.serialization.file_sha256(config_path):
                msg = "Parent child config bytes disagree with its saved summary."
                raise ValueError(msg)
            identity = summary.get("checkpoint_identity")
            saved_digest = identity.get("resume_contract_digest") if isinstance(identity, Mapping) else None
            if not isinstance(saved_digest, str) or re.fullmatch(r"[0-9a-f]{64}", saved_digest) is None:
                msg = "Completed parent child requires a valid saved resume identity."
                raise ValueError(msg)
            if saved_digest != child["semantic_config_sha256"]:
                msg = "Parent child scientific identity disagrees with its immutable parent."
                raise ValueError(msg)
        else:
            evidence = summary.get("run_identity")
            if not isinstance(evidence, Mapping) or evidence.get("resolved_config_sha256") != resolved_config_digest(config):
                msg = "Incomplete parent child config disagrees with its persisted run identity."
                raise ValueError(msg)
    return resolved


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


def _transient_child_exists(record: Mapping[str, Any], marker: Path) -> bool:
    """Return whether either exact child leaf already exists or is a symlink."""
    children = record.get("children")
    if not isinstance(children, Mapping):
        msg = "Transient experiment children metadata is invalid."
        raise TypeError(msg)
    for key in children:
        child = children.get(key)
        child_path = child.get("path") if isinstance(child, Mapping) else None
        if not isinstance(child_path, str) or not child_path:
            msg = "Transient experiment child path metadata is invalid."
            raise TypeError(msg)
        path = marker.parent / child_path
        if path.exists() or path.is_symlink():
            return True
    return False


def admit_fresh_transient_experiment_record(
    record: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> Path:
    """Publish a fresh parent or recover its exact childless pre-allocation marker."""
    requested = validate_transient_experiment_record(dict(record))
    path = experiment_record_path(requested, output_root=output_root)
    lock_path = common.paths.resolve_run_lock_path(path.parent)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if path.exists() or path.is_symlink():
            if not path.is_file() or path.is_symlink():
                msg = f"Transient experiment record is unsafe: {path}."
                raise ValueError(msg)
            existing = validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))
            if existing["parent_identity_sha256"] == requested["parent_identity_sha256"]:
                if not _transient_child_exists(requested, path):
                    return path
                msg = f"Matching experiment already exists for {requested['parent_label']!r}; rerun the same command to continue it."
                raise FileExistsError(msg)
            msg = (
                "Scientific parent identity conflict: "
                f"experiment={requested['parent_label']!r}, "
                f"existing config identity={existing['parent_identity_sha256']}, "
                f"requested config identity={requested['parent_identity_sha256']}. "
                "Resume the matching run, or pass --revision <label> to start a deliberately separate experiment."
            )
            raise FileExistsError(msg)
        try:
            _atomic_create_json(path, requested)
        except FileExistsError as error:
            msg = f"Transient experiment parent was concurrently published: {path}."
            raise FileExistsError(msg) from error
    return path


def validate_persisted_transient_experiment_record(
    requested: Mapping[str, Any],
    *,
    output_root: Path | str,
) -> Path:
    """Require an existing parent record to match a continuing request exactly."""
    expected = validate_transient_experiment_record(dict(requested))
    path = experiment_record_path(expected, output_root=output_root)
    if not path.is_file() or path.is_symlink():
        msg = f"Transient experiment parent record is missing or unsafe: {path}."
        raise FileNotFoundError(msg)
    actual = validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))
    if actual["parent_identity_sha256"] != expected["parent_identity_sha256"]:
        msg = "The continuing request does not match the persisted transient parent identity."
        raise ValueError(msg)
    return path
