# ruff: noqa: TRY003, EM101, EM102, D105, TC003
"""
generation_publication_completion_composite.py

Validate and persist composite completion evidence for partially successful campaigns.

Responsibilities:
  - Bind original individually admitted cases and replacement terminal batches
  - Reject collisions, failed replacements, and incomplete target memberships
  - Persist an immutable receipt and exact replacement transfer inventory

Design principles:
  - Original parent campaigns remain partial and are never synthesized as terminal
  - Every composite membership is identity- and hash-bound before Dataset use
  - Only successful replacement terminal evidence contributes to publication

This module does NOT:
  - Submit replacement campaigns, copy source files, or build Dataset payloads
  - Admit raw CPU source trees or reinterpret unsuccessful case evidence
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import common
from src.generation.contracts import generation_contracts_source as source_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.generation.runtime.generation_runtime_batch import TerminalBatchEvidence, TerminalCaseEvidence

SCHEMA_KIND = "generation_completion_composite"
SCHEMA_VERSION = 1
RECEIPT_FILENAME = "completion_composite.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CompositeCaseSource:
    """Bind one admitted completed case to its scientific source batch."""

    batch_id: str
    batch_name: str
    material_family: str
    material_role: str
    evaluation_regime: str
    sampling_regime: str
    source_run_id: str
    source_git_commit: str
    source_campaign_manifest_sha256: str
    terminal: TerminalBatchEvidence | None
    case: TerminalCaseEvidence
    source_kind: str

    def __post_init__(self) -> None:
        if self.source_kind not in {"parent_partial", "replacement"}:
            raise ValueError("Composite case source_kind must be parent_partial or replacement.")
        if not self.batch_id or not self.batch_name:
            raise ValueError("Composite case source requires non-empty batch identity and name.")
        common.paths.validate_logical_name(self.source_run_id, label="source_run_id")
        if (
            source_service.validate_git_commit(self.source_git_commit) != self.source_git_commit
            or _SHA256.fullmatch(self.source_campaign_manifest_sha256) is None
        ):
            raise ValueError("Composite source run requires exact commit and campaign-manifest digests.")
        if self.source_kind == "parent_partial" and self.terminal is not None:
            raise ValueError("Parent partial sources must be case-local and must not claim terminal evidence.")
        if self.source_kind == "replacement" and self.terminal is None:
            raise ValueError("Replacement sources require admitted terminal batch evidence.")
        if self.terminal is not None and len(self.terminal.cases) != 1:
            raise ValueError("Replacement composite sources must be one-case terminal batches.")


def completion_directory(
    completion_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the completion-owned immutable publication directory."""
    safe = common.paths.validate_logical_name(completion_id, label="completion_id")
    return common.paths.get_generation_meta_root(storage_root=storage_root) / "completions" / safe


def _receipt_path(completion_id: str, *, storage_root: Path | str | None) -> Path:
    return completion_directory(completion_id, storage_root=storage_root) / RECEIPT_FILENAME


def _case_payload(source: CompositeCaseSource, *, storage_root: Path) -> dict[str, Any]:
    case = source.case
    artifact = case.artifact("processed", "case.h5")
    return {
        "source_kind": source.source_kind,
        "batch_id": source.batch_id,
        "batch_name": source.batch_name,
        "material_family": source.material_family,
        "material_role": source.material_role,
        "evaluation_regime": source.evaluation_regime,
        "sampling_regime": source.sampling_regime,
        "source_run_id": source.source_run_id,
        "source_git_commit": source.source_git_commit,
        "source_campaign_manifest_sha256": source.source_campaign_manifest_sha256,
        "case_id": case.case_id,
        "case_index": case.case_index,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "success_sha256": case.success_sha256,
        "provenance_sha256": case.provenance_sha256,
        "case_hdf5_sha256": case.case_hdf5_sha256,
        "case_hdf5_relative": artifact.path.relative_to(storage_root).as_posix(),
        "terminal_manifest_sha256": None if source.terminal is None else source.terminal.manifest_sha256,
        "terminal_batch_id": None if source.terminal is None else source.terminal.batch_id,
        "terminal_batch_storage_name": None if source.terminal is None else getattr(source.terminal, "batch_storage_name", None),
    }


def build_composite_receipt(
    *,
    completion_id: str,
    completion_state: Mapping[str, Any],
    completion_state_sha256: str,
    parent_run_id: str,
    parent_partial_sha256: str,
    targets: Mapping[str, int],
    original_cases: Sequence[CompositeCaseSource],
    replacement_cases: Sequence[CompositeCaseSource],
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build or reuse one immutable schema-v1 completion composite receipt."""
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    if completion_state.get("completion_id") != completion_id:
        raise ValueError("Completion state does not bind the requested composite completion ID.")
    if common.serialization.canonical_json_sha256(dict(completion_state)) != completion_state_sha256:
        raise ValueError("Completion state digest differs from the declared immutable state hash.")
    if completion_state.get("parent_run_id") != parent_run_id or completion_state.get("parent_partial_sha256") != parent_partial_sha256:
        raise ValueError("Completion state parent evidence conflicts with composite parent identity.")
    normalized_targets = {str(key): int(value) for key, value in targets.items()}
    if not normalized_targets or any(value < 1 for value in normalized_targets.values()):
        raise ValueError("Composite targets must contain positive exact successful counts.")
    sources = tuple(original_cases) + tuple(replacement_cases)
    if not sources:
        raise ValueError("Completion composite requires at least one admitted source case.")
    original_kinds_valid = all(source.source_kind == "parent_partial" for source in original_cases)
    replacement_kinds_valid = all(source.source_kind == "replacement" for source in replacement_cases)
    if not original_kinds_valid or not replacement_kinds_valid:
        raise ValueError("Composite source collections disagree with their declared source kind.")
    members = [_case_payload(source, storage_root=storage) for source in sources]
    for key in ("case_input_id", "simulation_case_id"):
        values = [str(member[key]) for member in members]
        if len(values) != len(set(values)):
            raise ValueError(f"Composite source membership contains duplicate {key} values.")
    counts: dict[str, int] = dict.fromkeys(normalized_targets, 0)
    for member in members:
        batch_id = str(member["batch_id"])
        if batch_id not in counts:
            raise ValueError(f"Composite source references undeclared target batch {batch_id!r}.")
        counts[batch_id] += 1
    if counts != normalized_targets:
        raise ValueError(f"Composite successful membership must exactly meet targets; actual={counts}, targets={normalized_targets}.")
    inventory = sorted(members, key=lambda item: (item["batch_id"], item["simulation_case_id"]))
    replacement_inventory = [member for member in inventory if member["source_kind"] == "replacement"]
    receipt = {
        "schema_kind": SCHEMA_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completion_id": completion_id,
        "completion_state_sha256": completion_state_sha256,
        "parent_run_id": parent_run_id,
        "parent_partial_sha256": parent_partial_sha256,
        "targets": dict(sorted(normalized_targets.items())),
        "source_membership": inventory,
        "original_success_membership": [item for item in inventory if item["source_kind"] == "parent_partial"],
        "replacement_success_membership": replacement_inventory,
        "source_git_commits": sorted({str(item["source_git_commit"]) for item in inventory}),
        "combined_inventory_sha256": common.serialization.canonical_json_sha256(inventory),
        "replacement_transfer_inventory_sha256": common.serialization.canonical_json_sha256(replacement_inventory),
    }
    path = _receipt_path(completion_id, storage_root=storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_composite_receipt(completion_id, storage_root=storage)
        if existing != receipt:
            raise FileExistsError("Existing composite receipt conflicts with newly admitted completion evidence.")
        return existing
    common.serialization.atomic_write_json(path, receipt)
    return receipt


def load_composite_receipt(completion_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Load one immutable composite receipt and validate its internal inventory binding."""
    path = _receipt_path(completion_id, storage_root=storage_root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_kind") != SCHEMA_KIND or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Composite receipt has an unsupported schema.")
    members = value.get("source_membership")
    if not isinstance(members, list) or common.serialization.canonical_json_sha256(members) != value.get("combined_inventory_sha256"):
        raise ValueError("Composite receipt source inventory is malformed or changed.")
    replacement = [item for item in members if isinstance(item, dict) and item.get("source_kind") == "replacement"]
    source_git_commits = sorted({str(item.get("source_git_commit")) for item in members if isinstance(item, dict)})
    replacement_digest = common.serialization.canonical_json_sha256(replacement)
    replacement_locators_valid = all(
        isinstance(item.get("terminal_batch_storage_name"), str) and item["terminal_batch_storage_name"] for item in replacement
    )
    if (
        replacement != value.get("replacement_success_membership")
        or source_git_commits != value.get("source_git_commits")
        or replacement_digest != value.get("replacement_transfer_inventory_sha256")
        or not replacement_locators_valid
    ):
        raise ValueError("Composite replacement transfer inventory is malformed or changed.")
    return copy.deepcopy(value)


def replacement_transfer_plan(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return the exact successful replacement-only transfer inventory."""
    if receipt.get("schema_kind") != SCHEMA_KIND:
        raise ValueError("Replacement transfer plan requires a composite receipt.")
    members = receipt.get("replacement_success_membership")
    if not isinstance(members, list):
        raise TypeError("Composite receipt replacement membership is malformed.")
    return tuple(copy.deepcopy(item) for item in members)
