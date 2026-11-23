"""Persist the durable training history of both tasks: one record per completed epoch and per validation point.

Responsibilities:
  - Bind epoch and validation records to one task, run, and checkpoint identity
  - Publish ordered finite metric records through one atomic JSON replacement per training event
  - Convert read and live values through one guarded float conversion; raise FloatingPointError for non-finite
    live metrics before admission
  - Bind every validation record to its checkpoint objective-history entry: point, epoch, eligibility and value
  - Validate history against a restored last checkpoint without writing, so a refused resume changes nothing,
    then truncate the records after it; reject history behind it, disagreeing with it, holding a validation inside
    the epochs it counts that it does not count, or not ending its counted records exactly at its global step

Design principles:
  - History is evidence written before the last-checkpoint publication of its event
  - Only a regular run-local ``history.json`` is evidence, never a link
  - Records have scalar-only keys; epoch records are gap-free and validation points strictly ascending
  - Integral optimizer steps order the records: epoch end steps strictly increase, and validation steps strictly
    increase inside their epoch, so a reader can place every validation between two epoch ends; epochs, points
    and steps are integral as written, not only after rounding to a float
  - A validation record holds its event's complete values, so every validation point appears exactly once,
    also inside an epoch; an epoch record holds its event's training values, or, for an epoch end that committed
    no update after a validation inside its epoch, that validation's
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src import common

_HISTORY_FILENAME = "history.json"
MAX_METRIC_KEY_LENGTH = 160
_POINT_KEY = "validation/point"
_ELIGIBLE_KEY = "validation/eligible"
_STEP_KEY = "global_step"
# The loop records the ID evaluation metrics, the objective among them, under this prefix.
_ID_METRIC_PREFIX = "id/"
# The loop records an update stretch's training values under these prefixes (``train_update_groups``).
_TRAINING_PREFIXES = ("train/", "physics/train/", "system/train_")
_INTEGRAL_KEYS = frozenset({"epoch", _POINT_KEY, _STEP_KEY})
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
    """Validate one bounded non-empty mapping of finite scalar evidence, the one float conversion of read and live values."""
    if not isinstance(record, Mapping) or not record:
        msg = "Training history record must be a non-empty mapping."
        raise ValueError(msg)
    admitted: dict[str, float] = {}
    for key, value in record.items():
        if not isinstance(key, str) or not key or len(key) > MAX_METRIC_KEY_LENGTH:
            msg = "Training history metric keys must be bounded non-empty strings."
            raise ValueError(msg)
        msg = f"Training history metric {key!r} must be finite numeric evidence."
        try:
            number = float(value) if isinstance(value, int | float | Decimal) and not isinstance(value, bool) else math.nan
        except OverflowError as error:
            # Only an integer beyond the float range overflows; whatever converts to a finite float is evidence.
            raise ValueError(msg) from error
        if not math.isfinite(number):
            raise ValueError(msg)
        # The reader parses decimals exactly, so a count is integral as written, not only after rounding to a float.
        if key in _INTEGRAL_KEYS and not (value == value.to_integral_value() if isinstance(value, Decimal) else number.is_integer()):
            msg = f"Training history count {key!r} must be integral as written."
            raise ValueError(msg)
        admitted[key] = number
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
    if point is None or epoch is None or point <= previous[_POINT_KEY] or not previous["epoch"] <= epoch <= completed + 1:
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
    # Only a float is NaN or infinite; an integer beyond the float range is refused by the scalar admission.
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            msg = f"Training history metric {key!r} is non-finite at epoch {epoch}: {value}."
            raise FloatingPointError(msg)


def _admit_history(payload: Any, *, task: str, run_name: str, checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate history identity, the gap-free epoch records and the ascending validation records."""
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _HISTORY_KEYS
        or not all(isinstance(payload[key], list) for key in ("epochs", "validations"))
    ):
        msg = "Training history schema is invalid: it needs exactly its keys, with epoch and validation record lists."
        raise ValueError(msg)
    if payload["task"] != task or payload["run_name"] != run_name:
        msg = "Training history task or run identity disagrees with the current run."
        raise ValueError(msg)
    if payload["checkpoint_identity_digest"] != _identity_digest(checkpoint_identity):
        msg = "Training history checkpoint identity disagrees with the current run."
        raise ValueError(msg)
    history = _empty_history(task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    # All epochs first, so each validation is admitted against the end steps of its own and the previous epoch.
    for record in payload["epochs"]:
        history["epochs"].append(_admit_record(record, history=history))
    for record in payload["validations"]:
        history["validations"].append(_admit_validation(record, history=history))
    return history


def _read_history(path: Path, *, task: str, run_name: str, checkpoint_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Read and admit one persisted history file, a regular file of the run."""
    if path.is_symlink() or not path.is_file():
        msg = "Training history.json must be a regular run-local file, not a symbolic link or another file type."
        raise ValueError(msg)
    try:
        # Exact decimals let the scalar admission see a count's fraction that a float would round away.
        raw = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (json.JSONDecodeError, InvalidOperation) as error:
        # The decimal parser refuses an exponent beyond its range with InvalidOperation, an ArithmeticError.
        msg = "Training history.json is not valid JSON of representable numbers."
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
        validated points and binds the points alone: its run keeps no checkpoint evidence of the validation values,
        and the objective values ``trial.report`` stored in its study are not bound here.

    Returns
    -------
    dict[str, Any]
        The admitted history.

    Raises
    ------
    FileNotFoundError
        If the run has no ``history.json``.
    ValueError
        If the history is malformed, unordered, foreign, holds a validation after its last epoch, or disagrees
        with what the checkpoint counts; a disagreement names the first mismatching validation record.

    """
    _require_completed_epoch(completed_epoch)
    path = history_path(run_dir)
    if not path.is_file() and not path.is_symlink():
        msg = "Finished run has no durable history.json."
        raise FileNotFoundError(msg)
    history = _read_history(path, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    if len(history["epochs"]) != completed_epoch or len(history["validations"]) != len(validations):
        msg = "Finished training history disagrees with the number of epochs and validations its checkpoint counts."
        raise ValueError(msg)
    # A finished run has no running epoch: the event that ended it recorded the epoch of its last validation.
    if history["validations"] and history["validations"][-1]["epoch"] > completed_epoch:
        msg = "Finished training history records a validation after its last completed epoch."
        raise ValueError(msg)
    _require_bound_validations(history["validations"], validations)
    return history


def validate_resume_history(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
    validations: Sequence[Mapping[str, Any]],
    global_step: int,
) -> dict[str, Any]:
    """
    Read durable history and validate it against one restored last checkpoint without writing.

    The records the checkpoint counts must match ``validations``, its objective history,
    exactly as in :func:`validate_completed_history`; history behind it, or disagreeing with
    it, is incompatible evidence. So is a validation inside an epoch it counts that its
    objective history lacks: it lies before the checkpoint and is never trained again, so it
    is refused, not truncated. Every last checkpoint follows an event that recorded at its
    committed ``global_step``, so the newest record it counts lies exactly at that step (a
    fresh run counts none, at step 0) and a record it does not count at or after it, at it
    only for an epoch end that committed no update since (a divergence).

    Returns
    -------
    dict[str, Any]
        The admitted history, the records ahead of the checkpoint included.

    Raises
    ------
    FileNotFoundError
        If the run has no ``history.json`` although the checkpoint counts records or steps.
    ValueError
        If the history is malformed, foreign, behind the checkpoint or contradicts it.

    """
    _require_completed_epoch(completed_epoch)
    path = history_path(run_dir)
    if not path.exists() and not path.is_symlink():
        if completed_epoch or validations or global_step:
            msg = "Resume checkpoint has no durable history.json."
            raise FileNotFoundError(msg)
        return _empty_history(task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    history = _read_history(path, task=task, run_name=run_name, checkpoint_identity=checkpoint_identity)
    if len(history["epochs"]) < completed_epoch or len(history["validations"]) < len(validations):
        msg = "Training history is behind the restored last checkpoint."
        raise ValueError(msg)
    _require_bound_validations(history["validations"][: len(validations)], validations)
    # The checkpoint follows the end of every epoch it counts and holds every validation up to that end.
    surplus = history["validations"][len(validations) :]
    if surplus and surplus[0]["epoch"] <= completed_epoch:
        msg = "Training history records a validation before the restored last checkpoint that the checkpoint does not count."
        raise ValueError(msg)
    counted = (history["epochs"][:completed_epoch], history["validations"][: len(validations)])
    ahead = (history["epochs"][completed_epoch:], history["validations"][len(validations) :])
    newest = max((records[-1][_STEP_KEY] for records in counted if records), default=0.0)
    if newest != global_step or any(records[0][_STEP_KEY] < global_step for records in ahead if records):
        msg = (
            f"Training history contradicts the restored last checkpoint's global step {global_step}: "
            "the newest record it counts must lie exactly at that step and none it does not count before it."
        )
        raise ValueError(msg)
    return history


def reconcile_history(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
    validations: Sequence[Mapping[str, Any]],
    global_step: int,
) -> dict[str, Any]:
    """
    Validate durable history against one restored last checkpoint and truncate the records ahead of it.

    History is written at every event while ``last`` follows a wall-time cadence, so the
    epoch and validation records ahead of the checkpoint that :func:`validate_resume_history`
    admitted are truncated atomically; this is the one writer of that truncation.
    """
    history = validate_resume_history(
        run_dir,
        task=task,
        run_name=run_name,
        checkpoint_identity=checkpoint_identity,
        completed_epoch=completed_epoch,
        validations=validations,
        global_step=global_step,
    )
    if len(history["epochs"]) > completed_epoch or len(history["validations"]) > len(validations):
        # Events after the newest last checkpoint are trained again, so their records are replaced.
        history["epochs"] = history["epochs"][:completed_epoch]
        history["validations"] = history["validations"][: len(validations)]
        common.serialization.atomic_write_json(history_path(run_dir), history)
    return history


def make_epoch_state_callback(
    run_dir: Path | str,
    *,
    task: str,
    run_name: str,
    checkpoint_identity: Mapping[str, Any],
    initial_history: Mapping[str, Any] | None = None,
) -> Callable[[int, dict[str, float] | None, dict[str, float] | None], None]:
    """
    Build the writer of one event's validation and epoch records, replacing the file once per event before its last checkpoint.

    An epoch end at the step of an earlier event's validation in its epoch committed no update since (a divergence in
    its first group, also right after a resume at that validation): its epoch record takes that validation's training
    values, those under the loop's training prefixes, over the event's own.
    """
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
            last = validations[-1] if validations else None
            if validation is None and last is not None and (last["epoch"], last[_STEP_KEY]) == (epoch, training.get(_STEP_KEY)):
                training = {**training, **{key: value for key, value in last.items() if key.startswith(_TRAINING_PREFIXES)}}
            record = _admit_record({**copy.deepcopy(training), "epoch": epoch}, history={**initial, "validations": validations})
            initial["epochs"].append(record)
        initial["validations"] = validations
        common.serialization.atomic_write_json(path, initial)

    return write_event
