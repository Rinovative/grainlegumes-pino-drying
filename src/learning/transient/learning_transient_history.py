# ruff: noqa: EM101, EM102, TRY003, PLR2004
"""
learning_transient_history.py

Persist durable completed-epoch transient training history.

Responsibilities:
  - Bind epoch records to one task, run, and checkpoint identity
  - Publish ordered finite metric records through atomic JSON replacement
  - Reconcile only a crash-ahead final epoch against restored checkpoints

Design principles:
  - History is evidence written before last-checkpoint publication
  - Records have fixed scalar-only keys and reject duplicate epochs
  - Resume never silently repairs incompatible persisted evidence

This module does NOT:
  - Select checkpoints, drive training, or send observer telemetry
  - Store arbitrary batch-level values or mutable runtime objects
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src import common

_HISTORY_SCHEMA_VERSION = 1
_HISTORY_FILENAME = "history.json"


def history_path(run_dir: Path | str) -> Path:
    """Return the task-owned durable history path for one run."""
    return Path(run_dir) / _HISTORY_FILENAME


def _identity_digest(identity: Mapping[str, Any]) -> str:
    """Return the stable digest that binds history to checkpoint identity."""
    return common.serialization.canonical_json_sha256(dict(identity))


def _admit_record(record: Any, *, expected_epoch: int) -> dict[str, float]:
    """Validate one bounded scalar completed-epoch record."""
    if not isinstance(record, Mapping) or not record:
        raise ValueError("Transient history epoch record must be a non-empty mapping.")
    admitted: dict[str, float] = {}
    for key, value in record.items():
        if not isinstance(key, str) or not key or len(key) > 160:
            raise ValueError("Transient history metric keys must be bounded non-empty strings.")
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"Transient history metric {key!r} must be finite numeric evidence.")
        admitted[key] = float(value)
    epoch = admitted.get("epoch")
    if epoch != float(expected_epoch):
        raise ValueError("Transient history records must have ordered unique completed epochs.")
    return admitted


def _admit_history(payload: Any, *, task: str, run_name: str, checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate history identity and gap-free ordered record sequence."""
    required = {"schema_version", "task", "run_name", "checkpoint_identity_digest", "epochs"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload["schema_version"] != _HISTORY_SCHEMA_VERSION:
        raise ValueError("Transient history schema is invalid.")
    if payload["task"] != task or payload["run_name"] != run_name:
        raise ValueError("Transient history task or run identity disagrees with the current run.")
    if payload["checkpoint_identity_digest"] != _identity_digest(checkpoint_identity):
        raise ValueError("Transient history checkpoint identity disagrees with the current run.")
    records = payload["epochs"]
    if not isinstance(records, list):
        raise TypeError("Transient history epochs must be a list.")
    return {
        "schema_version": _HISTORY_SCHEMA_VERSION,
        "task": task,
        "run_name": run_name,
        "checkpoint_identity_digest": _identity_digest(checkpoint_identity),
        "epochs": [_admit_record(record, expected_epoch=index) for index, record in enumerate(records, start=1)],
    }


def validate_completed_history(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
) -> dict[str, Any]:
    """Read one completed run's exact, immutable epoch-history evidence."""
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int) or completed_epoch < 0:
        raise ValueError("completed_epoch must be a non-negative integer.")
    path = history_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError("Completed transient run has no durable history.json.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Transient history.json is not valid JSON.") from error
    history = _admit_history(raw, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    records = history["epochs"]
    if len(records) != completed_epoch:
        raise ValueError("Completed transient history and last checkpoint completed epoch disagree.")
    return history


def reconcile_history(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
) -> dict[str, Any]:
    """Read and reconcile durable history against one restored last checkpoint."""
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int) or completed_epoch < 0:
        raise ValueError("completed_epoch must be a non-negative integer.")
    path = history_path(run_dir)
    if not path.exists():
        if completed_epoch:
            raise FileNotFoundError("Transient resume checkpoint has no durable history.json.")
        return {
            "schema_version": _HISTORY_SCHEMA_VERSION,
            "task": task,
            "run_name": run_name,
            "checkpoint_identity_digest": _identity_digest(checkpoint_identity),
            "epochs": [],
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Transient history.json is not valid JSON.") from error
    history = _admit_history(raw, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    count = len(history["epochs"])
    if count == completed_epoch:
        return history
    if count == completed_epoch + 1:
        history["epochs"] = history["epochs"][:completed_epoch]
        common.serialization.atomic_write_json(path, history)
        return history
    raise ValueError("Transient history and last checkpoint disagree by more than one completed epoch.")


def make_epoch_state_callback(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    initial_history: Mapping[str, Any] | None = None,
) -> Callable[[int, dict[str, float]], None]:
    """Build an atomic pre-checkpoint callback for one fresh or reconciled run."""
    initial = _admit_history(
        initial_history
        if initial_history is not None
        else {
            "schema_version": _HISTORY_SCHEMA_VERSION,
            "task": task,
            "run_name": run_name,
            "checkpoint_identity_digest": _identity_digest(checkpoint_identity),
            "epochs": [],
        },
        task=task,
        run_name=run_name,
        checkpoint_identity=checkpoint_identity,
    )
    path = history_path(run_dir)

    def write_epoch(epoch: int, metrics: dict[str, float]) -> None:
        """Append one complete record before last-checkpoint publication."""
        record = _admit_record({"epoch": epoch, **copy.deepcopy(metrics)}, expected_epoch=epoch)
        expected_epoch = len(initial["epochs"]) + 1
        if epoch != expected_epoch:
            raise ValueError("Transient history callback received a duplicate or gapped epoch.")
        initial["epochs"].append(record)
        common.serialization.atomic_write_json(path, initial)

    return write_epoch
