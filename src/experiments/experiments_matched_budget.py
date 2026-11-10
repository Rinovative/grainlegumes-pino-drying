"""Publish immutable B training-time receipts for matched A0/B/A1 comparisons.

Responsibilities:
  - Bind completed B work to the exact parent identity and common A0 handoff
  - Validate receipt schema, positive finite training time, and identity digests
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

from . import experiments_run_identity as run_identity

RECEIPT_NAME = "b_train_budget_receipt.json"
_SCHEMA_KIND = "matched_a1_b_train_budget"
_SHA256_LENGTH = 64
_PROTOCOL_MATCHED = "matched_a1_b"
_CLOCK_SECONDS = "train_seconds"


def _positive_seconds(value: object) -> float:
    """Return an exact finite measured duration, independent of device type."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        msg = "Committed B train_seconds must be positive and finite."
        raise ValueError(msg)
    return float(value)


def build_b_receipt(
    *,
    parent_identity_sha256: str,
    a0_handoff_sha256: str,
    b_config: Mapping[str, Any],
    b_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind terminal B optimization work to its exact common A0 branch point."""
    protocol = b_config["training"].get("matched_a1_b")
    if protocol is None:
        msg = "A B budget receipt requires the matched_a1_b training protocol."
        raise ValueError(msg)
    if protocol["role"] != "b" or b_summary.get("status") != "completed":
        msg = "A B budget receipt requires completed matched B training."
        raise ValueError(msg)
    controller = b_summary.get("terminal_controller")
    if not isinstance(controller, Mapping) or not controller.get("budget_complete"):
        msg = "Completed B lacks a terminal committed budget controller."
        raise ValueError(msg)
    train_seconds = _positive_seconds(controller.get("train_seconds"))
    if controller.get("clock_kind") != _CLOCK_SECONDS:
        msg = "Completed B controller clock differs from its exact protocol."
        raise ValueError(msg)
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
        "optimizer_steps": controller["post_handoff_optimizer_steps"],
        "optimizer_groups": controller["post_handoff_optimizer_steps"],
        "terminal_epoch": b_summary["terminal_epoch"],
        "terminal_batch_cursor": controller.get("batch_cursor"),
        "best_epoch": b_summary["best_epoch"],
        "best_checkpoint_sha256": b_summary["best_checkpoint_sha256"],
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
    seconds = _positive_seconds(value.get("train_seconds"))
    if value.get("budget_value") != seconds:
        msg = "Matched A1/B receipt budget_value must equal train_seconds."
        raise ValueError(msg)
    for key in ("parent_identity_sha256", "a0_handoff_sha256", "best_checkpoint_sha256", "last_checkpoint_sha256"):
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
