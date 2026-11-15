"""Publish immutable B training-time receipts for matched A0/B/A1 comparisons.

Responsibilities:
  - Bind finished B work, measured on the optimisation clock of its admitted terminal budget, to the exact parent identity and A0 handoff
  - Record B's actual seconds and end status (completed, diverged, no eligible checkpoint), so A1 always gets them
  - Carry B's budget progress and eligible points, the comparison's one source of B's end
  - Validate receipt schema, end status, finite training time (zero only after a first-group divergence), and identity digests
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

from . import experiments_run_identity as run_identity

RECEIPT_NAME = "b_train_budget_receipt.json"
_SCHEMA_KIND = "matched_a1_b_train_budget"
_SHA256_LENGTH = 64
_PROTOCOL_MATCHED = "matched_a1_b"
_CLOCK_SECONDS = "train_seconds"
# B's end: its budget with an eligible checkpoint, a non-finite training loss, or its budget without an eligible checkpoint.
B_END_STATUSES = ("completed", "diverged", "no_eligible_checkpoint")


def _seconds(value: object, *, status: object) -> float:
    """Return B's exact finite measured seconds; zero only for a B that diverged before its first committed group."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (value == 0 and status != "diverged")
    ):
        msg = "Committed B train_seconds must be finite and positive, or zero for a B that diverged in its first group."
        raise ValueError(msg)
    return float(value)


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
    B's actual committed optimisation seconds also when B diverged or has no eligible
    checkpoint, so A1 always trains exactly these seconds. Example: B diverges at 30 % of its
    budget; the receipt holds those seconds and A1 trains them; a B that diverged in its
    first group holds 0 seconds, and its A1 runs no update.
    """
    protocol = b_config["training"].get("matched_a1_b")
    if protocol is None or protocol["role"] != "b":
        msg = "A B budget receipt requires matched B training under the matched_a1_b protocol."
        raise ValueError(msg)
    b_summary = finished_b["summary"]
    budget = finished_b["terminal_budget"]
    outcome = b_summary["training_outcome"]
    controller = b_summary.get("terminal_controller")
    if not isinstance(controller, Mapping) or not isinstance(budget, stage_budget.StageBudget) or budget.unit != _CLOCK_SECONDS:
        msg = "Finished B lacks its terminal controller or a terminal budget measured on the optimisation clock in train_seconds."
        raise ValueError(msg)
    train_seconds = _seconds(budget.train_seconds, status=outcome["status"])
    payload = {
        "schema_kind": _SCHEMA_KIND,
        "protocol": _PROTOCOL_MATCHED,
        "clock_kind": _CLOCK_SECONDS,
        "budget_value": train_seconds,
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
        "train_seconds": train_seconds,
        # B's end as the comparison reads it, also when B has no artifact of its own.
        "b_status": outcome["status"],
        "b_budget_progress": outcome["budget_progress"],
        "b_eligible_points": outcome["eligible_points"],
        "optimizer_steps": controller["post_handoff_optimizer_steps"],
        "optimizer_groups": controller["post_handoff_optimizer_steps"],
        "terminal_epoch": b_summary["terminal_epoch"],
        "best_epoch": b_summary["best_epoch"],
        "best_checkpoint_sha256": b_summary.get("best_checkpoint_sha256"),
        "last_checkpoint_sha256": b_summary["last_checkpoint_sha256"],
    }
    return {**payload, "receipt_sha256": common.serialization.canonical_json_sha256(payload)}


def validate_b_receipt(value: object) -> dict[str, Any]:
    """Validate an immutable receipt without opening training artifacts."""
    if (
        not isinstance(value, dict)
        or value.get("schema_kind") != _SCHEMA_KIND
        or value.get("protocol") != _PROTOCOL_MATCHED
        or value.get("clock_kind") != _CLOCK_SECONDS
    ):
        msg = "B train budget receipt schema, protocol or clock is unsupported."
        raise ValueError(msg)
    payload = dict(value)
    digest = payload.pop("receipt_sha256", None)
    if not isinstance(digest, str) or digest != common.serialization.canonical_json_sha256(payload):
        msg = "B train budget receipt digest mismatch."
        raise ValueError(msg)
    status = value.get("b_status")
    # Only B's selected checkpoint may be absent, exactly when B has none.
    if status not in B_END_STATUSES or (status == "completed" and value.get("best_checkpoint_sha256") is None):
        msg = "B train budget receipt lacks B's end status or its selected checkpoint (receipts without b_status are no longer accepted)."
        raise ValueError(msg)
    if status == "no_eligible_checkpoint" and value.get("best_checkpoint_sha256") is not None:
        msg = "B train budget receipt names a selected checkpoint although B has no eligible one."
        raise ValueError(msg)
    seconds = _seconds(value.get("train_seconds"), status=status)
    if value.get("budget_value") != seconds:
        msg = "Matched A1/B receipt budget_value must equal train_seconds."
        raise ValueError(msg)
    progress, eligible_points = value.get("b_budget_progress"), value.get("b_eligible_points")
    if (
        isinstance(progress, bool)
        or not isinstance(progress, (int, float))
        or not 0.0 <= progress <= 1.0
        or isinstance(eligible_points, bool)
        or not isinstance(eligible_points, int)
        or (eligible_points > 0) != (value.get("best_checkpoint_sha256") is not None)
    ):
        msg = "B train budget receipt budget progress or eligible points are invalid."
        raise ValueError(msg)
    # A B without a committed group has no last checkpoint.
    tokens = ["parent_identity_sha256", "a0_handoff_sha256", *(["last_checkpoint_sha256"] if seconds > 0 else [])]
    if seconds == 0 and value.get("last_checkpoint_sha256") is not None:
        msg = "B train budget receipt names a last checkpoint although B committed no group."
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
