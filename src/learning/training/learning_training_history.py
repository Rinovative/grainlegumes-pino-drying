"""Persist the durable training history of both tasks: one record per completed epoch and per validation point.

Responsibilities:
  - Bind epoch and validation records to one task, run, and checkpoint identity
  - Publish ordered finite metric records through one atomic JSON replacement per training event
  - Raise FloatingPointError for non-finite live metrics before admission
  - Bind every validation record to its checkpoint objective-history entry: point, epoch, eligibility and value
  - Truncate records ahead of a restored last checkpoint; reject history behind it or disagreeing with it

Design principles:
  - History is evidence written before the last-checkpoint publication of its event
  - Records have scalar-only keys; epoch records are gap-free and validation points strictly ascending
  - Optimizer steps order the records: epoch end steps strictly increase, and validation steps strictly
    increase inside their epoch, so a reader can place every validation between two epoch ends
  - A validation record holds its event's complete values, so every validation point appears exactly once,
    also inside an epoch; an epoch record holds its last event's training values
  - Resume never silently repairs incompatible persisted evidence
  - One task-agnostic contract: steady and transient runs write and admit the same file

This module does NOT:
  - Select checkpoints, drive training, or send observer telemetry
  - Store arbitrary batch-level values or mutable runtime objects
  - Decide which lifecycle requires the history or derive what a checkpoint counts; run admission and the
    checkpoint owner do
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src import common

_HISTORY_FILENAME = "history.json"
_MAX_METRIC_KEY_LENGTH = 160
_POINT_KEY = "validation/point"
_ELIGIBLE_KEY = "validation/eligible"
_STEP_KEY = "global_step"
# The loop records the ID evaluation metrics, the objective among them, under this prefix.
_ID_METRIC_PREFIX = "id/"
_HISTORY_KEYS = frozenset({"task", "run_name", "checkpoint_identity_digest", "epochs", "validations"})


def history_path(run_dir: Path | str) -> Path:
    """Return the durable history path for one run."""
    return Path(run_dir) / _HISTORY_FILENAME


def _identity_digest(identity: Mapping[str, Any]) -> str:
    """Return the stable digest that binds history to checkpoint identity."""
    return common.serialization.canonical_json_sha256(dict(identity))


def _empty_history(*, task: str, run_name: str, checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the history of a run that recorded nothing yet."""
    return {"task": task, "run_name": run_name, "checkpoint_identity_digest": _identity_digest(checkpoint_identity), "epochs": [], "validations": []}


def _admit_scalars(record: Any) -> dict[str, float]:
    """Validate one bounded non-empty mapping of finite scalar evidence."""
    if not isinstance(record, Mapping) or not record:
        msg = "Training history record must be a non-empty mapping."
        raise ValueError(msg)
    admitted: dict[str, float] = {}
    for key, value in record.items():
        if not isinstance(key, str) or not key or len(key) > _MAX_METRIC_KEY_LENGTH:
            msg = "Training history metric keys must be bounded non-empty strings."
            raise ValueError(msg)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            msg = f"Training history metric {key!r} must be finite numeric evidence."
            raise ValueError(msg)
        admitted[key] = float(value)
    return admitted


def _end_step(history: Mapping[str, Any], epoch: int) -> float:
    """Return the optimizer step at which a completed ``epoch`` of ``history`` ended; training starts at step 0."""
    return history["epochs"][epoch - 1][_STEP_KEY] if epoch else 0.0


def _admit_record(record: Any, *, history: Mapping[str, Any]) -> dict[str, float]:
    """Validate the next gap-free epoch record after ``history``, ending after the previous epoch and every recorded validation."""
    admitted = _admit_scalars(record)
    completed = len(history["epochs"])
    if admitted.get("epoch") != float(completed + 1):
        msg = "Training history epoch records must have ordered unique completed epochs."
        raise ValueError(msg)
    step = admitted.get(_STEP_KEY)
    # The writer records an event's validation before its epoch record, which therefore closes it.
    last_validation = history["validations"][-1][_STEP_KEY] if history["validations"] else 0.0
    if step is None or step <= _end_step(history, completed) or step < last_validation:
        msg = "Training history epoch end steps must strictly increase and close the validations recorded before them."
        raise ValueError(msg)
    return admitted


def _admit_validation(record: Any, *, history: Mapping[str, Any]) -> dict[str, float]:
    """Validate one validation record after ``history``: a later integral point and step at a non-decreasing epoch up to the running one."""
    admitted = _admit_scalars(record)
    point, epoch, step = admitted.get(_POINT_KEY), admitted.get("epoch"), admitted.get(_STEP_KEY)
    completed = len(history["epochs"])
    previous = history["validations"][-1] if history["validations"] else {_POINT_KEY: 0.0, "epoch": 1.0, _STEP_KEY: 0.0}
    # A validation inside an epoch belongs to the running epoch, one after the last completed one.
    if (
        point is None
        or epoch is None
        or not (point.is_integer() and epoch.is_integer())
        or point <= previous[_POINT_KEY]
        or not previous["epoch"] <= epoch <= completed + 1
    ):
        msg = "Training history validation records need strictly ascending integral points at non-decreasing epochs up to the running one."
        raise ValueError(msg)
    # Its step lies after the previous epoch's end and at most at its own epoch's end, unbounded in the running epoch.
    upper = _end_step(history, int(epoch)) if epoch <= completed else math.inf
    if step is None or step <= previous[_STEP_KEY] or not _end_step(history, int(epoch) - 1) < step <= upper:
        msg = "Training history validation steps must strictly increase and lie inside their epoch."
        raise ValueError(msg)
    return admitted


def _require_bound_validations(records: Sequence[Mapping[str, float]], validations: Sequence[Mapping[str, Any] | int]) -> None:
    """Require each validation record to match its checkpoint entry, or its bare point; refuse naming the first mismatch."""
    for index, (record, entry) in enumerate(zip(records, validations, strict=True), start=1):
        # (field, recorded, checkpoint) in the order the first mismatch is reported.
        if isinstance(entry, Mapping):
            value = entry["value"]
            fields = [
                ("validation point", record[_POINT_KEY], float(entry["point"])),
                ("epoch", record["epoch"], float(entry["epoch"])),
                ("eligibility", record.get(_ELIGIBLE_KEY), float(value is not None)),
            ]
            if value is not None:
                key = f"{_ID_METRIC_PREFIX}{entry['objective_id']}"
                fields.append((key, record.get(key), float(value)))
        else:
            fields = [("validation point", record[_POINT_KEY], float(entry))]
        for field, recorded, expected in fields:
            if recorded != expected:
                msg = f"Training history validation record {index} disagrees with its checkpoint in {field}: {recorded!r}, expected {expected!r}."
                raise ValueError(msg)


def _reject_nonfinite_live_metrics(epoch: int, metrics: Mapping[str, Any]) -> None:
    """Raise FloatingPointError: a non-finite live metric is a numerical training failure, not a persistence violation."""
    for key, value in metrics.items():
        if isinstance(value, int | float) and not isinstance(value, bool) and not math.isfinite(value):
            msg = f"Training history metric {key!r} is non-finite at epoch {epoch}: {value}."
            raise FloatingPointError(msg)


def _admit_history(payload: Any, *, task: str, run_name: str, checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate history identity, the gap-free epoch records and the ascending validation records."""
    if not isinstance(payload, Mapping) or set(payload) != _HISTORY_KEYS:
        msg = "Training history schema is invalid."
        raise ValueError(msg)
    if payload["task"] != task or payload["run_name"] != run_name:
        msg = "Training history task or run identity disagrees with the current run."
        raise ValueError(msg)
    if payload["checkpoint_identity_digest"] != _identity_digest(checkpoint_identity):
        msg = "Training history checkpoint identity disagrees with the current run."
        raise ValueError(msg)
    epochs, validations = payload["epochs"], payload["validations"]
    if not isinstance(epochs, list) or not isinstance(validations, list):
        msg = "Training history epochs and validations must be lists."
        raise TypeError(msg)
    history = _empty_history(task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    # All epochs first, so each validation is admitted against the end steps of its own and the previous epoch.
    for record in epochs:
        history["epochs"].append(_admit_record(record, history=history))
    for record in validations:
        history["validations"].append(_admit_validation(record, history=history))
    return history


def _read_history(path: Path, *, task: str, run_name: str, checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Read and admit one persisted history file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = "Training history.json is not valid JSON."
        raise ValueError(msg) from error
    return _admit_history(raw, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)


def _require_completed_epoch(completed_epoch: int) -> None:
    """Validate the number of epoch records a checkpoint counts."""
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int) or completed_epoch < 0:
        msg = "completed_epoch must be a non-negative integer."
        raise ValueError(msg)


def validate_completed_history(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
    validations: Sequence[Mapping[str, Any]] | Sequence[int],
) -> dict[str, Any]:
    """
    Read one finished run's immutable history: exactly the epochs and validations its checkpoint or terminal budget counts.

    Parameters
    ----------
    run_dir : Path | str
        Finished run directory holding ``history.json``.
    task, run_name : str
        Run identity the history must name.
    checkpoint_identity : Mapping[str, Any]
        Checkpoint identity whose digest the history must carry.
    completed_epoch : int
        Number of epoch records the checkpoint counts.
    validations : Sequence[Mapping[str, Any]] | Sequence[int]
        The last checkpoint's ``objective_history`` wherever the run keeps one: each validation record then matches
        its entry in point, epoch, eligibility (``value is not None``) and, when eligible, the objective value. A
        finished Optuna trial deletes its checkpoints and keeps only its terminal budget, so it passes the budget's
        validated points and binds the points alone: no persisted evidence holds its validation values.

    Returns
    -------
    dict[str, Any]
        The admitted history.

    Raises
    ------
    FileNotFoundError
        If the run has no ``history.json``.
    ValueError
        If the history is malformed, unordered, foreign, or disagrees with what the checkpoint counts; a
        disagreement names the first mismatching validation record.

    """
    _require_completed_epoch(completed_epoch)
    path = history_path(run_dir)
    if not path.is_file():
        msg = "Finished run has no durable history.json."
        raise FileNotFoundError(msg)
    history = _read_history(path, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    if len(history["epochs"]) != completed_epoch or len(history["validations"]) != len(validations):
        msg = "Finished training history disagrees with the number of epochs and validations its checkpoint counts."
        raise ValueError(msg)
    _require_bound_validations(history["validations"], validations)
    return history


def reconcile_history(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
    validations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Read and reconcile durable history against one restored last checkpoint.

    History is written at every event while ``last`` follows a wall-time cadence, so
    epoch and validation records ahead of the checkpoint are truncated atomically.
    The records it counts must match ``validations``, its objective history, exactly as
    in :func:`validate_completed_history`; history behind it, or disagreeing with it, is
    incompatible evidence and raises ``ValueError``.
    """
    _require_completed_epoch(completed_epoch)
    path = history_path(run_dir)
    if not path.exists():
        if completed_epoch or validations:
            msg = "Resume checkpoint has no durable history.json."
            raise FileNotFoundError(msg)
        return _empty_history(task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    history = _read_history(path, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    if len(history["epochs"]) < completed_epoch or len(history["validations"]) < len(validations):
        msg = "Training history is behind the restored last checkpoint."
        raise ValueError(msg)
    _require_bound_validations(history["validations"][: len(validations)], validations)
    if len(history["epochs"]) > completed_epoch or len(history["validations"]) > len(validations):
        # Events after the newest last checkpoint are trained again, so their records are replaced.
        history["epochs"] = history["epochs"][:completed_epoch]
        history["validations"] = history["validations"][: len(validations)]
        common.serialization.atomic_write_json(path, history)
    return history


def make_epoch_state_callback(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    initial_history: Mapping[str, Any] | None = None,
) -> Callable[[int, dict[str, float] | None, dict[str, float] | None], None]:
    """Build the writer of one event's validation and epoch records, replacing the file once per event before its last checkpoint."""
    initial = _admit_history(
        initial_history if initial_history is not None else _empty_history(task=task, run_name=run_name, checkpoint_identity=checkpoint_identity),
        task=task,
        run_name=run_name,
        checkpoint_identity=checkpoint_identity,
    )
    path = history_path(run_dir)

    def write_event(epoch: int, training: dict[str, float] | None, validation: dict[str, float] | None) -> None:
        """Append one event's records before its last-checkpoint publication."""
        for metrics in (training, validation):
            if metrics is not None:
                _reject_nonfinite_live_metrics(epoch, metrics)
        # Both records are admitted before either is kept: the epoch record against the event's validation.
        validations = list(initial["validations"])
        if validation is not None:
            validations.append(_admit_validation({**copy.deepcopy(validation), "epoch": epoch}, history=initial))
        if training is not None:
            record = _admit_record({**copy.deepcopy(training), "epoch": epoch}, history={**initial, "validations": validations})
            initial["epochs"].append(record)
        initial["validations"] = validations
        common.serialization.atomic_write_json(path, initial)

    return write_event
