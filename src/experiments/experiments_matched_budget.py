"""Publish immutable B budget receipts for matched A0/B/A1 comparisons.

Responsibilities:
  - Bind finished B work, counted in the supervised transitions of its admitted terminal budget, to the exact parent identity
    and A0 handoff
  - Record B's actual supervised transitions and end status (completed, diverged, no eligible checkpoint), so A1 always gets them
  - Report B's optimisation seconds, which budget nothing, and bind the one GPU model of the B/A1 pair
  - Carry B's budget progress and eligible points, the comparison's one source of B's end
  - Validate the exact receipt keys and field types, end status against budget progress, exact transition count (zero only
    after a first-group divergence), device and identity digests
  - Publish one receipt and admit it before its selected A1 continuation

Design principles:
  - A canonical JSON digest binds the complete receipt payload
  - Publication accepts an existing destination only when its bytes are identical
  - Parent, handoff, and B run identities must match the requested continuation

This module does NOT:
  - Measure training time or execute B or A1 optimization
  - Open training artifacts to validate their contents or select a checkpoint
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src import common
from src.learning.training import learning_training_budget as stage_budget
from src.learning.transient import learning_transient_handoff as transient_handoff

from . import experiments_run_identity as run_identity

RECEIPT_NAME = "b_train_budget_receipt.json"
_SCHEMA_KIND = "matched_a1_b_train_budget"
_SHA256_LENGTH = 64
_PROTOCOL_MATCHED = "matched_a1_b"
_BUDGET_UNIT = "supervised_transitions"
# B's end: its budget with an eligible checkpoint, a non-finite training loss, or its budget without an eligible checkpoint.
B_END_STATUSES = ("completed", "diverged", "no_eligible_checkpoint")
# Every field build_b_receipt writes; a receipt holds exactly these.
_RECEIPT_KEYS = frozenset(
    {
        "schema_kind",
        "protocol",
        "budget_unit",
        "supervised_transitions",
        "train_seconds",
        "device",
        "parent_identity_sha256",
        "a0_handoff_sha256",
        "a0_run_name",
        "b_run_name",
        "dataset_identity",
        "input_profile",
        "temporal_conditioning",
        "spatial_stride",
        "model",
        "objective",
        "initialization",
        "b_status",
        "b_budget_progress",
        "b_eligible_points",
        "optimizer_steps",
        "terminal_epoch",
        "best_epoch",
        "best_checkpoint_sha256",
        "last_checkpoint_sha256",
        "receipt_sha256",
    }
)


def _transitions(value: object, *, status: object) -> int:
    """Return B's exact committed supervised transitions; zero only for a B that diverged before its first committed group."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (value == 0 and status != "diverged"):
        msg = "Committed B supervised_transitions must be a positive integer, or zero for a B that diverged in its first group."
        raise ValueError(msg)
    return value


def _count(value: object, *, minimum: int = 0) -> bool:
    """Return whether ``value`` is an integer (not a bool) of at least ``minimum``."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _name(value: object) -> bool:
    """Return whether ``value`` is a non-empty string."""
    return isinstance(value, str) and bool(value)


def _validate_field_types(value: Mapping[str, Any]) -> None:
    """Require each identity and work field of a receipt to have the type build_b_receipt writes."""
    dataset = value["dataset_identity"]
    valid = {
        "a0_run_name": _name(value["a0_run_name"]),
        "b_run_name": _name(value["b_run_name"]),
        "input_profile": _name(value["input_profile"]),
        "initialization": _name(value["initialization"]),
        "dataset_identity": isinstance(dataset, Mapping)
        and set(dataset) == {"train_dataset", "material_ood_datasets"}
        and _name(dataset["train_dataset"])
        and isinstance(dataset["material_ood_datasets"], list)
        and all(_name(name) for name in dataset["material_ood_datasets"]),
        "temporal_conditioning": isinstance(value["temporal_conditioning"], Mapping),
        "model": isinstance(value["model"], Mapping),
        "objective": isinstance(value["objective"], Mapping),
        "spatial_stride": _count(value["spatial_stride"], minimum=1),
        "optimizer_steps": _count(value["optimizer_steps"]),
        "terminal_epoch": _count(value["terminal_epoch"]),
        "best_epoch": value["best_epoch"] is None or _count(value["best_epoch"]),
    }
    invalid = sorted(key for key, ok in valid.items() if not ok)
    if invalid:
        msg = f"B train budget receipt fields {invalid} have an invalid type."
        raise ValueError(msg)


def pair_device(sessions: object) -> dict[str, str | None]:
    """
    Return the one GPU model every attempt of a B ran on, which its A1 must use too.

    A B/A1 pair trains on one GPU model, so its reported time ratio compares like with
    like; run admission refuses a B attempt on another model, and a B whose runtime
    sessions name several is refused here.
    """
    if not isinstance(sessions, list) or not sessions or not all(isinstance(session, Mapping) for session in sessions):
        msg = "Finished B lacks its runtime sessions."
        raise ValueError(msg)
    devices = {json.dumps(transient_handoff.device_evidence(session), sort_keys=True) for session in sessions}
    if len(devices) != 1:
        msg = f"B trained on several devices {sorted(devices)}; a B/A1 pair trains on one GPU model."
        raise ValueError(msg)
    return json.loads(devices.pop())


def build_b_receipt(
    *,
    parent_identity_sha256: str,
    a0_handoff_sha256: str,
    b_config: Mapping[str, Any],
    finished_b: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Bind terminal B optimization work and B's end status to its exact common A0 branch point.

    ``finished_b`` is B as ``experiments_run.validate_finished_run`` admitted it: its summary
    and the terminal budget its training outcome was checked against. The receipt records
    B's actual committed supervised transitions also when B diverged or has no eligible
    checkpoint, so A1 always trains exactly these transitions. Example: B diverges at 30 % of
    its budget; the receipt holds those transitions and A1 trains them; a B that diverged in its
    first group holds 0, and its A1 runs no update. B's optimisation seconds are only reported.
    """
    protocol = b_config["training"].get("matched_a1_b")
    if protocol is None or protocol["role"] != "b":
        msg = "A B budget receipt requires matched B training under the matched_a1_b protocol."
        raise ValueError(msg)
    b_summary = finished_b["summary"]
    budget = finished_b["terminal_budget"]
    outcome = b_summary["training_outcome"]
    controller = b_summary.get("terminal_controller")
    if not isinstance(controller, Mapping) or not isinstance(budget, stage_budget.StageBudget) or budget.unit != _BUDGET_UNIT:
        msg = "Finished B lacks its terminal controller or a terminal budget in supervised transitions."
        raise ValueError(msg)
    payload = {
        "schema_kind": _SCHEMA_KIND,
        "protocol": _PROTOCOL_MATCHED,
        "budget_unit": _BUDGET_UNIT,
        "supervised_transitions": _transitions(budget.supervised_transitions, status=outcome["status"]),
        "train_seconds": budget.train_seconds,
        "device": pair_device(b_summary.get("runtime_sessions")),
        "parent_identity_sha256": parent_identity_sha256,
        "a0_handoff_sha256": a0_handoff_sha256,
        "a0_run_name": b_config["training"]["teacher_handoff"]["source_run_name"],
        "b_run_name": b_config["run"]["name"],
        "dataset_identity": run_identity.dataset_identity_evidence(b_config),
        "input_profile": b_config["input_profile"],
        "temporal_conditioning": b_config["temporal"]["temporal_conditioning"],
        "spatial_stride": b_config["data"]["spatial_stride"],
        "model": b_config["model"],
        "objective": b_config["evaluation"]["objective"],
        "initialization": protocol["initialization"],
        # B's end as the comparison reads it, also when B has no artifact of its own.
        "b_status": outcome["status"],
        "b_budget_progress": outcome["budget_progress"],
        "b_eligible_points": outcome["eligible_points"],
        "optimizer_steps": controller["post_handoff_optimizer_steps"],
        "terminal_epoch": b_summary["terminal_epoch"],
        "best_epoch": b_summary["best_epoch"],
        "best_checkpoint_sha256": b_summary.get("best_checkpoint_sha256"),
        "last_checkpoint_sha256": b_summary["last_checkpoint_sha256"],
    }
    return {**payload, "receipt_sha256": common.serialization.canonical_json_sha256(payload)}


def validate_b_receipt(value: object) -> dict[str, Any]:
    """
    Validate an immutable receipt without opening training artifacts.

    A receipt holds exactly the fields :func:`build_b_receipt` writes, each with its type, and
    B's end status agrees with its budget progress: a completed or no-eligible B spent its
    budget (progress 1), a diverged B did not, and only a B that diverged in its first group
    has progress 0, 0 transitions, no optimizer step and no checkpoint. Example: a receipt that
    claims ``completed`` at progress 0 is refused even with a recomputed digest.
    """
    if (
        not isinstance(value, dict)
        or set(value) != _RECEIPT_KEYS
        or value["schema_kind"] != _SCHEMA_KIND
        or value["protocol"] != _PROTOCOL_MATCHED
        or value["budget_unit"] != _BUDGET_UNIT
    ):
        msg = "B train budget receipt schema, keys, protocol or budget unit are unsupported."
        raise ValueError(msg)
    payload = dict(value)
    digest = payload.pop("receipt_sha256", None)
    if not isinstance(digest, str) or digest != common.serialization.canonical_json_sha256(payload):
        msg = "B train budget receipt digest mismatch."
        raise ValueError(msg)
    _validate_field_types(value)
    status = value.get("b_status")
    # Only B's selected checkpoint may be absent, exactly when B has none.
    if status not in B_END_STATUSES or (status == "completed" and value.get("best_checkpoint_sha256") is None):
        msg = "B train budget receipt lacks B's end status or its selected checkpoint."
        raise ValueError(msg)
    if status == "no_eligible_checkpoint" and value.get("best_checkpoint_sha256") is not None:
        msg = "B train budget receipt names a selected checkpoint although B has no eligible one."
        raise ValueError(msg)
    transitions = _transitions(value.get("supervised_transitions"), status=status)
    seconds = value.get("train_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)) or seconds < 0:
        msg = "B train budget receipt train_seconds must be finite and non-negative."
        raise ValueError(msg)
    device = value.get("device")
    if not isinstance(device, Mapping) or transient_handoff.device_evidence(device) != device:
        msg = "B train budget receipt device must be exactly the pair's device type and GPU model."
        raise ValueError(msg)
    progress, eligible_points = value.get("b_budget_progress"), value.get("b_eligible_points")
    if (
        isinstance(progress, bool)
        or not isinstance(progress, (int, float))
        or not 0.0 <= progress <= 1.0
        or isinstance(eligible_points, bool)
        or not isinstance(eligible_points, int)
        or not 0 <= eligible_points <= stage_budget.VALIDATION_POINTS
        or (eligible_points > 0) != (value.get("best_checkpoint_sha256") is not None)
        or (value["best_epoch"] is None) != (value.get("best_checkpoint_sha256") is None)
    ):
        msg = "B train budget receipt budget progress, eligible points or selected epoch are invalid."
        raise ValueError(msg)
    # Only a diverged B stops before its budget, and only one without a committed group stays at progress 0.
    if (progress == 1.0) != (status != "diverged") or (progress == 0.0) != (transitions == 0):
        msg = f"B train budget receipt end status {status!r} contradicts its budget progress {progress} for {transitions} transitions."
        raise ValueError(msg)
    # A B without a committed group has no optimizer step, no checkpoint and no validation point.
    tokens = ["parent_identity_sha256", "a0_handoff_sha256", *(["last_checkpoint_sha256"] if transitions > 0 else [])]
    checkpoints = (value.get("last_checkpoint_sha256"), value.get("best_checkpoint_sha256"))
    if transitions == 0 and (any(token is not None for token in checkpoints) or value["optimizer_steps"] != 0 or eligible_points != 0):
        msg = "B train budget receipt names a checkpoint, optimizer steps or eligible points although B committed no group."
        raise ValueError(msg)
    if transitions > 0 and value["optimizer_steps"] == 0:
        msg = "B train budget receipt commits supervised transitions without an optimizer step."
        raise ValueError(msg)
    if value.get("best_checkpoint_sha256") is not None:
        tokens.append("best_checkpoint_sha256")
    for key in tokens:
        token = value.get(key)
        if not isinstance(token, str) or len(token) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in token):
            msg = f"B train budget receipt {key} is invalid."
            raise ValueError(msg)
    return dict(value)


def publish_b_receipt(run_dir: Path | str, receipt: Mapping[str, Any]) -> Path:
    """Publish once; a repeat accepts only exact identical receipt bytes."""
    path = Path(run_dir) / RECEIPT_NAME
    validated = validate_b_receipt(dict(receipt))
    encoded = (json.dumps(validated, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
            return path
        msg = f"Immutable B budget receipt differs: {path}"
        raise FileExistsError(msg)
    descriptor, temporary = tempfile.mkstemp(prefix=".b-budget-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            msg = f"Immutable B budget receipt differs: {path}"
            raise FileExistsError(msg) from error
    finally:
        Path(temporary).unlink()
    return path


def load_b_receipt(
    path: Path | str,
    *,
    parent_identity_sha256: str,
    a0_handoff_sha256: str,
    b_run_name: str,
) -> dict[str, Any]:
    """Require an exact parent, branch point and B source before A1 starts."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        msg = f"B budget receipt is absent or unsafe: {source}"
        raise FileNotFoundError(msg)
    receipt = validate_b_receipt(json.loads(source.read_text(encoding="utf-8")))
    if (
        receipt["parent_identity_sha256"] != parent_identity_sha256
        or receipt["a0_handoff_sha256"] != a0_handoff_sha256
        or receipt["b_run_name"] != b_run_name
    ):
        msg = "B budget receipt does not match the selected A0/B parent identity."
        raise ValueError(msg)
    return receipt
