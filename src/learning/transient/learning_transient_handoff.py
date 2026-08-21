# ruff: noqa: EM101, EM102, TRY003
"""
learning_transient_handoff.py

Publish and validate immutable transient teacher handoff snapshots.

Responsibilities:
  - Copy selected Stage-A checkpoint and scaling artifacts immutably
  - Bind copied bytes and semantic compatibility evidence in a strict manifest
  - Validate target admission before any continuation state is restored

Design principles:
  - Publication uses a sibling temporary directory and atomic rename
  - Existing snapshots are accepted only when byte-identical and valid
  - Compatibility separates scientific state from comparison execution policy

This module does NOT:
  - Construct models, fit scalers, or restore checkpoint state
  - Choose source runs or infer target budgets
"""

from __future__ import annotations

import copy
import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src import common

_HANDOFF_DIRECTORY = "stage_a_handoff"
_MANIFEST_FILENAME = "manifest.json"
_CHECKPOINT_FILENAME = "best_checkpoint.pt"
_SCALING_FILENAME = "normalizer.pt"
_SCHEMA_VERSION = 1


def _canonical_digest(value: Mapping[str, Any]) -> str:
    """Return the stable digest of one JSON-compatible compatibility payload."""
    return common.serialization.canonical_json_sha256(dict(value))


def _device_evidence(device: Mapping[str, Any]) -> dict[str, str | None]:
    """Admit exact device type and CUDA GPU model evidence."""
    device_type = device.get("device_type")
    gpu_model = device.get("cuda_device_name", device.get("cuda_gpu_model"))
    if (
        device_type not in {"cpu", "cuda"}
        or (device_type == "cuda" and (not isinstance(gpu_model, str) or not gpu_model))
        or (device_type == "cpu" and gpu_model is not None)
    ):
        raise ValueError("Teacher handoff device evidence must be CPU or CUDA with an exact GPU model.")
    return {"device_type": device_type, "cuda_gpu_model": gpu_model}


def compatibility_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract transient scientific compatibility while excluding comparison policy."""
    required = {"task", "task_contract", "input_profile", "temporal", "model", "loss", "optimizer", "scheduler", "scaling", "run", "data", "training"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Transient handoff config lacks compatibility roots {sorted(missing)}.")
    run = config["run"]
    training = config["training"]
    data = config["data"]
    if not all(isinstance(item, Mapping) for item in (run, training, data)):
        raise TypeError("Transient handoff config run, data, and training must be mappings.")
    return {
        "task": copy.deepcopy(config["task"]),
        "task_contract": copy.deepcopy(config["task_contract"]),
        "input_profile": copy.deepcopy(config["input_profile"]),
        "temporal": {key: copy.deepcopy(value) for key, value in config["temporal"].items() if key != "sampling"},
        "model": copy.deepcopy(config["model"]),
        "loss": copy.deepcopy(config["loss"]),
        "optimizer": copy.deepcopy(config["optimizer"]),
        "scheduler": copy.deepcopy(config["scheduler"]),
        "scaling": copy.deepcopy(config["scaling"]),
        "deterministic": copy.deepcopy(run.get("deterministic")),
        "seed": copy.deepcopy(run.get("seed")),
        "amp": copy.deepcopy(training.get("mixed_precision")),
        "datasets": {"train_dataset": copy.deepcopy(data.get("train_dataset")), "ood_datasets": copy.deepcopy(data.get("ood_datasets"))},
    }


def publish_stage_a_handoff(
    run_dir: Path | str,
    *,
    source_run_name: str,
    checkpoint_identity: Mapping[str, Any],
    best_epoch: int,
    global_step: int,
    scaling_semantic_digest: str,
    task_contract_digest: str,
    tensorizer_digest: str,
    model_kind: str,
    input_profile: str,
    device: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one immutable Stage-A best-state handoff snapshot."""
    source = Path(run_dir)
    checkpoint = common.paths.resolve_best_checkpoint_file(source)
    scaling = common.paths.resolve_normalizer_path(source)
    if not checkpoint.is_file() or not scaling.is_file():
        raise FileNotFoundError("Stage-A handoff requires selected best_checkpoint.pt and normalizer.pt.")
    if not isinstance(source_run_name, str) or not source_run_name:
        raise ValueError("source_run_name must be non-empty.")
    if not isinstance(best_epoch, int) or best_epoch < 1 or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("Stage-A handoff best epoch and global step are invalid.")
    compatibility = compatibility_payload(config)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "source_run_name": source_run_name,
        "files": {
            "checkpoint": {"filename": _CHECKPOINT_FILENAME, "sha256": common.serialization.file_sha256(checkpoint)},
            "scaling": {"filename": _SCALING_FILENAME, "sha256": common.serialization.file_sha256(scaling)},
        },
        "scaling_semantic_digest": scaling_semantic_digest,
        "task_contract_digest": task_contract_digest,
        "tensorizer_digest": tensorizer_digest,
        "model_kind": model_kind,
        "input_profile": input_profile,
        "checkpoint_identity": copy.deepcopy(dict(checkpoint_identity)),
        "best_epoch": best_epoch,
        "global_step": global_step,
        "device": _device_evidence(device),
        "compatibility": compatibility,
        "compatibility_digest": _canonical_digest(compatibility),
        "target_reset_policy": {"adapter": "fresh", "selection": "fresh", "global_step": "fresh", "history": "fresh"},
    }
    destination = source / _HANDOFF_DIRECTORY
    if destination.exists():
        existing = validate_stage_a_handoff(destination, target_config=config, device=device, expected_source_run_name=source_run_name)
        if existing["files"]["checkpoint"]["sha256"] != common.serialization.file_sha256(checkpoint) or existing["files"]["scaling"][
            "sha256"
        ] != common.serialization.file_sha256(scaling):
            raise FileExistsError("Existing Stage-A handoff is not byte-identical to the selected source artifacts.")
        return existing
    temporary = source / f".{_HANDOFF_DIRECTORY}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        shutil.copyfile(checkpoint, temporary / _CHECKPOINT_FILENAME)
        shutil.copyfile(scaling, temporary / _SCALING_FILENAME)
        common.serialization.atomic_write_json(temporary / _MANIFEST_FILENAME, manifest)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_stage_a_handoff(destination, target_config=config, device=device, expected_source_run_name=source_run_name)


def validate_local_teacher_handoff(
    manifest_path: Path | str,
    *,
    local_normalizer_path: Path | str,
    target_config: Mapping[str, Any],
    device: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a target-local teacher identity without reopening its source run."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError("Local teacher handoff manifest is invalid JSON.") from error
    required = {
        "schema_version",
        "source_run_name",
        "files",
        "scaling_semantic_digest",
        "task_contract_digest",
        "tensorizer_digest",
        "model_kind",
        "input_profile",
        "checkpoint_identity",
        "best_epoch",
        "global_step",
        "device",
        "compatibility",
        "compatibility_digest",
        "target_reset_policy",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required or manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Local teacher handoff manifest schema is invalid.")
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != {"checkpoint", "scaling"}:
        raise ValueError("Local teacher handoff file manifest is invalid.")
    scaling = files["scaling"]
    checkpoint = files["checkpoint"]
    if (
        not isinstance(scaling, Mapping)
        or scaling.get("filename") != _SCALING_FILENAME
        or not isinstance(scaling.get("sha256"), str)
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("filename") != _CHECKPOINT_FILENAME
        or not isinstance(checkpoint.get("sha256"), str)
    ):
        raise ValueError("Local teacher handoff file entries are invalid.")
    if common.serialization.file_sha256(Path(local_normalizer_path)) != scaling["sha256"]:
        raise ValueError("Local teacher scaling bytes do not match the teacher handoff manifest.")
    compatibility = manifest["compatibility"]
    if not isinstance(compatibility, Mapping) or manifest["compatibility_digest"] != _canonical_digest(compatibility):
        raise ValueError("Local teacher handoff compatibility digest is invalid.")
    if dict(compatibility) != compatibility_payload(target_config):
        raise ValueError("Local teacher handoff is scientifically incompatible with the target configuration.")
    if _device_evidence(manifest["device"]) != _device_evidence(device):
        raise ValueError("Local teacher handoff device type or CUDA GPU model does not match the target runtime.")
    return copy.deepcopy(dict(manifest))


def validate_stage_a_handoff(
    handoff_dir: Path | str,
    *,
    target_config: Mapping[str, Any],
    device: Mapping[str, Any],
    expected_source_run_name: str | None = None,
) -> dict[str, Any]:
    """Validate immutable snapshot bytes, device, and scientific compatibility."""
    path = Path(handoff_dir)
    manifest_path = path / _MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError("Transient handoff manifest is invalid JSON.") from error
    required = {
        "schema_version",
        "source_run_name",
        "files",
        "scaling_semantic_digest",
        "task_contract_digest",
        "tensorizer_digest",
        "model_kind",
        "input_profile",
        "checkpoint_identity",
        "best_epoch",
        "global_step",
        "device",
        "compatibility",
        "compatibility_digest",
        "target_reset_policy",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required or manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Transient handoff manifest schema is invalid.")
    if expected_source_run_name is not None and manifest["source_run_name"] != expected_source_run_name:
        raise ValueError("Transient handoff source run name disagrees with target configuration.")
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != {"checkpoint", "scaling"}:
        raise ValueError("Transient handoff file manifest is invalid.")
    for name, filename in (("checkpoint", _CHECKPOINT_FILENAME), ("scaling", _SCALING_FILENAME)):
        entry = files[name]
        if not isinstance(entry, Mapping) or entry.get("filename") != filename or not isinstance(entry.get("sha256"), str):
            raise ValueError("Transient handoff file entry is invalid.")
        if common.serialization.file_sha256(path / filename) != entry["sha256"]:
            raise ValueError(f"Transient handoff {name} bytes do not match its manifest.")
    compatibility = manifest["compatibility"]
    if not isinstance(compatibility, Mapping) or manifest["compatibility_digest"] != _canonical_digest(compatibility):
        raise ValueError("Transient handoff compatibility digest is invalid.")
    if dict(compatibility) != compatibility_payload(target_config):
        raise ValueError("Transient handoff is scientifically incompatible with the target configuration.")
    if _device_evidence(manifest["device"]) != _device_evidence(device):
        raise ValueError("Transient handoff device type or CUDA GPU model does not match the target runtime.")
    return copy.deepcopy(dict(manifest))
