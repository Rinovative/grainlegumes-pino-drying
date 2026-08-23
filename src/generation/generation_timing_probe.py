"""
generation_timing_probe.py

Own one isolated and resumable COMSOL phase-timing diagnostic.

Responsibilities:
  - Execute one bounded transient case outside normal Generation publication
  - Observe complete batch-log and stdout evidence incrementally and durably
  - Classify conservative timing candidates and observed-wall diagnostics
  - Publish one bounded immutable experiment bundle with exact integrity evidence

Design principles:
  - Diagnostic evidence never populates production solver-timing fields
  - Controller interruption does not require a second COMSOL execution
  - Final-file and incremental parses remain separate, comparable evidence
  - Publication is fail-closed for identity, inventory, size, and digest integrity

This module does NOT:
  - Publish generated cases, Dataset packages, PT shards, or readiness evidence
  - Invent source-level solve boundaries hidden inside the MPH executable
  - Claim that synthetic parser tests validate real COMSOL log grammar
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_service
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.runtime import generation_runtime_comsol as comsol_service

if TYPE_CHECKING:
    from collections.abc import Callable

PROBE_SCHEMA_VERSION: Final = 1
PROBE_SCHEMA_KIND: Final = "comsol_phase_timing_probe"
PROBE_SCOPE: Final = "comsol_phase_timing_probe"
POLL_INTERVAL_SECONDS: Final = 0.25
_MAX_LOG_BYTES: Final = 32 * 1024 * 1024
_MAX_BUNDLE_BYTES: Final = 100 * 1024 * 1024
_MAX_LINE_BYTES: Final = 256 * 1024
_MAX_CONTEXT_BYTES: Final = 2_048
_MAX_EVENTS: Final = 10_000
_MAX_CANDIDATES: Final = 20_000
_MIN_DUPLICATE_COUNT: Final = 2
_PHASES: Final = ("stationary", "transient")
_SUPPORTED_TIME_UNITS: Final = {
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3_600.0,
    "hr": 3_600.0,
    "hour": 3_600.0,
    "hours": 3_600.0,
}
_SOLUTION_PREFIX: Final = re.compile(r"solution\s+time\s*:", re.IGNORECASE)
_SOLUTION_VALUE: Final = re.compile(
    r"solution\s+time\s*:\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>[A-Za-z]+)?",
    re.IGNORECASE,
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_EXACT_BUNDLE_INVENTORY: Final = frozenset(
    {
        "manifest.json",
        "method_verdicts.json",
        "exact_command.json",
        "environment.json",
        "comsol_batch.log",
        "stdout.log",
        "stderr.log",
        "phase_events.jsonl",
        "batch_log_candidates.json",
        "stdout_candidates.json",
        "observed_wall_timing.json",
        "parser_summary.json",
        "sha256sums.txt",
        "README.md",
    }
)


@dataclass(frozen=True, slots=True)
class ProbeObservationState:
    """Persisted appended-byte reader state for one probe-owned log."""

    offset: int = 0
    partial: bytes = b""
    partial_offset: int = 0
    device: int | None = None
    inode: int | None = None
    next_line_number: int = 1
    phase: str | None = None
    generation: int = 0


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one persisted JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected one JSON object in {path}."
        raise TypeError(msg)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one probe-owned JSON object."""
    common.serialization.atomic_write_json(path, payload)


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    """Atomically persist bounded diagnostic events as JSON lines."""
    content = "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events)
    common.serialization.atomic_write_text(path, content)


def _state_payload(state: ProbeObservationState) -> dict[str, Any]:
    """Return the complete durable observer-state payload."""
    return {
        "offset": state.offset,
        "partial_hex": state.partial.hex(),
        "partial_offset": state.partial_offset,
        "device": state.device,
        "inode": state.inode,
        "next_line_number": state.next_line_number,
        "phase": state.phase,
        "generation": state.generation,
    }


def _state_from_payload(payload: Mapping[str, Any]) -> ProbeObservationState:
    """Validate and restore one durable observer state."""
    required = {
        "offset",
        "partial_hex",
        "partial_offset",
        "device",
        "inode",
        "next_line_number",
        "phase",
        "generation",
    }
    if set(payload) != required:
        msg = "Probe observer state has an invalid schema."
        raise ValueError(msg)
    integer_names = ("offset", "partial_offset", "next_line_number", "generation")
    if any(isinstance(payload[name], bool) or not isinstance(payload[name], int) for name in integer_names):
        msg = "Probe observer offsets and counters must be integers."
        raise ValueError(msg)
    if payload["offset"] < 0 or payload["partial_offset"] < 0 or payload["next_line_number"] < 1 or payload["generation"] < 0:
        msg = "Probe observer offsets and counters are outside their valid ranges."
        raise ValueError(msg)
    if payload["device"] is not None and not isinstance(payload["device"], int):
        msg = "Probe observer device must be an integer or null."
        raise ValueError(msg)
    if payload["inode"] is not None and not isinstance(payload["inode"], int):
        msg = "Probe observer inode must be an integer or null."
        raise ValueError(msg)
    if payload["phase"] not in {*_PHASES, None}:
        msg = "Probe observer phase is invalid."
        raise ValueError(msg)
    try:
        partial = bytes.fromhex(str(payload["partial_hex"]))
    except ValueError as error:
        msg = "Probe observer partial bytes are malformed."
        raise ValueError(msg) from error
    if len(partial) > _MAX_LINE_BYTES:
        msg = "Probe observer partial line exceeds the bounded line contract."
        raise ValueError(msg)
    return ProbeObservationState(
        offset=int(payload["offset"]),
        partial=partial,
        partial_offset=int(payload["partial_offset"]),
        device=payload["device"],
        inode=payload["inode"],
        next_line_number=int(payload["next_line_number"]),
        phase=payload["phase"],
        generation=int(payload["generation"]),
    )


def _phase_marker(line: str) -> tuple[str | None, str | None]:
    """Return a conservative phase and marker kind detected from one line."""
    lowered = line.casefold()
    stationary = "stationary" in lowered
    transient = "transient" in lowered or "time-dependent" in lowered or "time dependent" in lowered
    if stationary == transient:
        return None, None
    phase = "stationary" if stationary else "transient"
    return phase, f"{phase}_phase_first_observed"


def _bounded_context(lines: list[str], position: int) -> tuple[list[str], list[str], str]:
    """Return bounded preceding, following, and combined candidate context."""
    preceding = [line[:512] for line in lines[max(0, position - 3) : position]]
    following = [line[:512] for line in lines[position + 1 : position + 4]]
    combined = "\n".join([*preceding, lines[position][:512], *following])
    encoded = combined.encode("utf-8", errors="replace")[:_MAX_CONTEXT_BYTES]
    return preceding, following, encoded.decode("utf-8", errors="replace")


def _candidate_record(
    *,
    line: str,
    source: str,
    byte_offset: int,
    line_number: int,
    phase: str | None,
    preceding: list[str],
    following: list[str],
    context: str,
) -> dict[str, Any] | None:
    """Classify one possible COMSOL Solution-time line conservatively."""
    if _SOLUTION_PREFIX.search(line) is None:
        return None
    match = _SOLUTION_VALUE.search(line)
    parsed_value: float | None = None
    parsed_unit: str | None = None
    seconds: float | None = None
    parse_status = "malformed"
    ambiguity_reasons: list[str] = []
    if match is not None:
        try:
            parsed_value = float(match.group("value"))
        except (TypeError, ValueError):
            parsed_value = None
        raw_unit = match.group("unit")
        parsed_unit = raw_unit.casefold() if raw_unit else None
        if parsed_value is not None and math.isfinite(parsed_value) and parsed_unit in _SUPPORTED_TIME_UNITS:
            seconds = parsed_value * _SUPPORTED_TIME_UNITS[parsed_unit]
            parse_status = "parsed"
        elif parsed_value is not None and math.isfinite(parsed_value):
            parse_status = "unsupported_format"
    lowered_context = context.casefold()
    nested = bool(line[:1].isspace()) or "nested solver" in lowered_context or "subsolver" in lowered_context
    classification = "nested" if nested else "top_level"
    if phase is None:
        ambiguity_reasons.append("phase_not_unambiguously_detected")
    if parse_status != "parsed":
        ambiguity_reasons.append(parse_status)
    if seconds is not None and seconds < 0:
        ambiguity_reasons.append("negative_value")
    if classification != "top_level":
        ambiguity_reasons.append("nested_candidate")
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "source": source,
        "byte_offset": byte_offset,
        "line_number": line_number,
        "exact_line": line,
        "preceding_context": preceding,
        "following_context": following,
        "context": context,
        "detected_phase": phase,
        "phase": phase,
        "parsed_value": parsed_value,
        "value": parsed_value,
        "unit": parsed_unit,
        "converted_seconds": seconds,
        "seconds": seconds,
        "parse_status": parse_status,
        "classification": classification,
        "ambiguity_reasons": ambiguity_reasons,
        "duplicate_classification": "unique",
    }


def _mark_duplicates(records: list[dict[str, Any]]) -> None:
    """Mark every member of a repeated exact candidate group."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["detected_phase"],
            record["exact_line"],
            record["parsed_value"],
            record["unit"],
        )
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        if len(group) < _MIN_DUPLICATE_COUNT:
            continue
        for record in group:
            record["duplicate_classification"] = "duplicate"
            if "duplicate_candidate" not in record["ambiguity_reasons"]:
                record["ambiguity_reasons"].append("duplicate_candidate")


def _scan_log_bytes(
    data: bytes,
    *,
    source: str,
    base_offset: int,
    first_line_number: int,
    initial_phase: str | None,
    generation: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, int]:
    """Scan complete supplied lines and return candidates, events, phase, and next line."""
    raw_lines = data.splitlines(keepends=True)
    decoded = [raw.rstrip(b"\r\n").decode("utf-8", errors="replace") for raw in raw_lines]
    offsets: list[int] = []
    cursor = base_offset
    for raw in raw_lines:
        offsets.append(cursor)
        cursor += len(raw)
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    phase = initial_phase
    for position, line in enumerate(decoded):
        marker_phase, marker_kind = _phase_marker(line)
        if marker_phase is not None:
            phase = marker_phase
            events.append(
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "source": source,
                    "byte_offset": offsets[position],
                    "line_number": first_line_number + position,
                    "detected_phase": phase,
                    "phase": phase,
                    "event_kind": marker_kind,
                    "candidate_timing": None,
                    "parse_status": "phase_marker",
                    "exact_line": line,
                    "generation": generation,
                }
            )
        preceding, following, context = _bounded_context(decoded, position)
        candidate = _candidate_record(
            line=line,
            source=source,
            byte_offset=offsets[position],
            line_number=first_line_number + position,
            phase=phase,
            preceding=preceding,
            following=following,
            context=context,
        )
        if candidate is None:
            continue
        candidate["generation"] = generation
        candidates.append(candidate)
        events.append(
            {
                "schema_version": PROBE_SCHEMA_VERSION,
                "source": source,
                "byte_offset": candidate["byte_offset"],
                "line_number": candidate["line_number"],
                "detected_phase": candidate["detected_phase"],
                "phase": candidate["phase"],
                "event_kind": (f"{phase}_completion_observed" if phase in _PHASES else "unassigned_timing_candidate"),
                "candidate_timing": {
                    "value": candidate["parsed_value"],
                    "unit": candidate["unit"],
                    "seconds": candidate["converted_seconds"],
                },
                "parse_status": candidate["parse_status"],
                "exact_line": line,
                "generation": generation,
            }
        )
    if len(candidates) > _MAX_CANDIDATES:
        msg = "Probe log contains more timing candidates than the bounded diagnostic contract permits."
        raise ValueError(msg)
    _mark_duplicates(candidates)
    return candidates, events, phase, first_line_number + len(raw_lines)


def parse_solution_times(data: bytes, *, source: str = "comsol_batch.log") -> list[dict[str, Any]]:
    """
    Parse all possible Solution-time candidates from one complete retained log.

    Synthetic input exercises software semantics only; it does not establish the
    grammar emitted by the unavailable real COMSOL runtime.
    """
    records, _, _, _ = _scan_log_bytes(
        data,
        source=source,
        base_offset=0,
        first_line_number=1,
        initial_phase=None,
        generation=0,
    )
    return records


def summarize_solution_times(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return conservative per-phase Candidate A verdicts and no production timing."""
    phase_results: dict[str, dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    unknown = [record for record in records if record.get("detected_phase") not in _PHASES]
    for phase in _PHASES:
        candidates = [record for record in records if record.get("detected_phase") == phase]
        valid = [
            record
            for record in candidates
            if record.get("parse_status") == "parsed"
            and record.get("classification") == "top_level"
            and record.get("duplicate_classification") == "unique"
            and not record.get("ambiguity_reasons")
            and isinstance(record.get("converted_seconds"), (int, float))
            and math.isfinite(float(record["converted_seconds"]))
            and float(record["converted_seconds"]) >= 0.0
        ]
        parse_statuses = {str(record.get("parse_status")) for record in candidates}
        if not candidates:
            status = "ambiguous" if unknown else "missing"
        elif len(candidates) == 1 and len(valid) == 1 and not unknown:
            status = "confirmed"
        elif parse_statuses == {"malformed"}:
            status = "malformed"
        elif parse_statuses == {"unsupported_format"}:
            status = "unsupported_format"
        else:
            status = "ambiguous"
        reasons = sorted({str(reason) for record in [*candidates, *unknown] for reason in record.get("ambiguity_reasons", [])})
        if unknown:
            reasons.append("unassigned_candidate_present")
        phase_results[phase] = {
            "status": status,
            "candidate_count": len(candidates),
            "confirmed_candidate_count": len(valid),
            "duplicate_count": sum(record.get("duplicate_classification") == "duplicate" for record in candidates),
            "ambiguous_count": sum(bool(record.get("ambiguity_reasons")) for record in candidates),
            "ambiguity_reasons": sorted(set(reasons)),
        }
        if status == "confirmed":
            selected[phase] = valid[0]
    if all(phase_results[phase]["status"] == "confirmed" for phase in _PHASES) and int(selected["stationary"]["line_number"]) >= int(
        selected["transient"]["line_number"]
    ):
        for phase in _PHASES:
            phase_results[phase]["status"] = "ambiguous"
            phase_results[phase]["ambiguity_reasons"] = [
                *phase_results[phase]["ambiguity_reasons"],
                "stationary_not_before_transient",
            ]
    candidate_sum = None
    if all(phase_results[phase]["status"] == "confirmed" for phase in _PHASES):
        candidate_sum = sum(float(selected[phase]["converted_seconds"]) for phase in _PHASES)
    return {
        "candidate": "A",
        "method_status": "confirmed" if candidate_sum is not None else "unresolved",
        "diagnostic_only": True,
        "phases": phase_results,
        "candidate_scientific_sum_seconds": candidate_sum,
        "unassigned_candidate_count": len(unknown),
    }


def _decorate_events(
    events: list[dict[str, Any]],
    *,
    probe_id: str | None,
    case_input_id: str | None,
    simulation_case_id: str | None,
    attempt_id: str | None,
    observed_monotonic_ns: int,
    observed_utc: str,
) -> None:
    """Bind one observation instant and the available probe identities."""
    for event in events:
        event.update(
            {
                "probe_id": probe_id,
                "case_input_id": case_input_id,
                "simulation_case_id": simulation_case_id,
                "attempt_id": attempt_id,
                "observed_monotonic_ns": observed_monotonic_ns,
                "observed_utc": observed_utc,
                "diagnostic_only": True,
            }
        )


def observe_appended_bytes(
    path: Path,
    state: ProbeObservationState,
    *,
    probe_id: str | None = None,
    case_input_id: str | None = None,
    simulation_case_id: str | None = None,
    attempt_id: str | None = None,
    observed_monotonic_ns: int | None = None,
    observed_utc: str | None = None,
) -> tuple[ProbeObservationState, list[dict[str, Any]]]:
    """Read appended bytes exactly once while retaining an incomplete trailing line."""
    if not path.exists():
        return state, []
    with path.open("rb") as stream:
        file_stat = os.fstat(stream.fileno())
        replacement = state.inode is not None and (state.device != file_stat.st_dev or state.inode != file_stat.st_ino)
        truncation = not replacement and file_stat.st_size < state.offset
        reset = replacement or truncation
        read_offset = 0 if reset else state.offset
        stream.seek(read_offset)
        chunk = stream.read()
    prefix = b"" if reset else state.partial
    merged = prefix + chunk
    complete_end = merged.rfind(b"\n") + 1
    complete = merged[:complete_end]
    partial = merged[complete_end:]
    if len(partial) > _MAX_LINE_BYTES:
        msg = f"Probe log has an incomplete line exceeding {_MAX_LINE_BYTES} bytes: {path}."
        raise ValueError(msg)
    generation = state.generation + 1 if reset else state.generation
    first_line = 1 if reset else state.next_line_number
    initial_phase = None if reset else state.phase
    merged_offset = 0 if reset else (state.partial_offset if state.partial else state.offset)
    _, events, phase, next_line = _scan_log_bytes(
        complete,
        source=path.name,
        base_offset=merged_offset,
        first_line_number=first_line,
        initial_phase=initial_phase,
        generation=generation,
    )
    now_ns = time.monotonic_ns() if observed_monotonic_ns is None else observed_monotonic_ns
    now_utc = _utc_now() if observed_utc is None else observed_utc
    if reset:
        events.insert(
            0,
            {
                "schema_version": PROBE_SCHEMA_VERSION,
                "source": path.name,
                "byte_offset": 0,
                "line_number": 1,
                "detected_phase": None,
                "phase": None,
                "event_kind": "log_replaced" if replacement else "log_truncated",
                "candidate_timing": None,
                "parse_status": "observer_reset",
                "exact_line": None,
                "generation": generation,
            },
        )
    _decorate_events(
        events,
        probe_id=probe_id,
        case_input_id=case_input_id,
        simulation_case_id=simulation_case_id,
        attempt_id=attempt_id,
        observed_monotonic_ns=now_ns,
        observed_utc=now_utc,
    )
    consumed_offset = read_offset + len(chunk)
    partial_offset = merged_offset + len(complete)
    return (
        ProbeObservationState(
            offset=consumed_offset,
            partial=partial,
            partial_offset=partial_offset,
            device=file_stat.st_dev,
            inode=file_stat.st_ino,
            next_line_number=next_line,
            phase=phase,
            generation=generation,
        ),
        events,
    )


def _flush_trailing_line(
    path: Path,
    state: ProbeObservationState,
    *,
    metadata: Mapping[str, str],
) -> tuple[ProbeObservationState, list[dict[str, Any]]]:
    """Process one final unterminated line after the child process exits."""
    state, events = observe_appended_bytes(
        path,
        state,
        probe_id=metadata["probe_id"],
        case_input_id=metadata["case_input_id"],
        simulation_case_id=metadata["simulation_case_id"],
        attempt_id=metadata["attempt_id"],
    )
    if not state.partial:
        return state, events
    _, trailing_events, phase, next_line = _scan_log_bytes(
        state.partial,
        source=path.name,
        base_offset=state.partial_offset,
        first_line_number=state.next_line_number,
        initial_phase=state.phase,
        generation=state.generation,
    )
    _decorate_events(
        trailing_events,
        probe_id=metadata["probe_id"],
        case_input_id=metadata["case_input_id"],
        simulation_case_id=metadata["simulation_case_id"],
        attempt_id=metadata["attempt_id"],
        observed_monotonic_ns=time.monotonic_ns(),
        observed_utc=_utc_now(),
    )
    return (
        ProbeObservationState(
            offset=state.offset,
            partial=b"",
            partial_offset=state.offset,
            device=state.device,
            inode=state.inode,
            next_line_number=next_line,
            phase=phase,
            generation=state.generation,
        ),
        [*events, *trailing_events],
    )


def _probe_config(path: Path | str) -> tuple[Path, dict[str, Any]]:
    """Load one exact schema-versioned timing-probe configuration."""
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    required = {
        "schema_kind",
        "schema_version",
        "campaign_config",
        "batch_name",
        "case_index",
        "cores_per_case",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema_kind") != "generation_timing_probe"
        or payload.get("schema_version") != PROBE_SCHEMA_VERSION
    ):
        msg = "Timing-probe configuration has an invalid schema."
        raise ValueError(msg)
    for name in ("case_index", "cores_per_case"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"Timing-probe {name} must be a positive integer."
            raise ValueError(msg)
    if not isinstance(payload["batch_name"], str) or not payload["batch_name"]:
        msg = "Timing-probe batch_name must be one non-empty string."
        raise ValueError(msg)
    campaign_source = (source.parent / str(payload["campaign_config"])).resolve()
    result = dict(payload)
    result["campaign_config"] = str(campaign_source)
    return source, result


def _probe_root(storage_root: Path | str) -> Path:
    """Resolve and create the experiment-owned probe root."""
    root = common.paths.get_experiments_root(storage_root=storage_root).resolve() / PROBE_SCOPE
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scheduler_kind() -> str:
    """Return the exact scheduler context bound into one probe command."""
    return "slurm" if os.environ.get("SLURM_JOB_ID") else "local"


def _requested_work_root(work_root: Path | str | None) -> str | None:
    """Return the canonical external work-root identity used by the run key."""
    return None if work_root is None else str(Path(work_root).expanduser().resolve())


def _timing_run_key(
    *,
    source: Path,
    probe: Mapping[str, Any],
    campaign_digest: str,
    source_commit: str,
    scheduler_kind: str,
    requested_work_root: str | None,
) -> str:
    """Bind the stable controller owner to config, source, scheduler, and work root."""
    if scheduler_kind not in {"local", "slurm"}:
        message = "Timing-probe scheduler kind must be local or slurm."
        raise ValueError(message)
    return common.serialization.canonical_json_sha256(
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "source_commit": source_service.validate_git_commit(source_commit),
            "source_config_sha256": common.serialization.file_sha256(source),
            "campaign_digest": campaign_digest,
            "probe_config": dict(probe),
            "scheduler_kind": scheduler_kind,
            "requested_work_root": requested_work_root,
        }
    )


def _child_control(
    *,
    active: Path,
    work: Path,
    command: list[str],
    run_key: str,
    probe_id: str,
    source_commit: str,
    case_input_id: str,
    simulation_case_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Return the exact child control derived from admitted session evidence."""
    runtime = work / "runtime"
    return {
        "schema_kind": "comsol_phase_timing_probe_child",
        "schema_version": PROBE_SCHEMA_VERSION,
        "run_key": run_key,
        "probe_id": probe_id,
        "source_commit": source_commit,
        "case_input_id": case_input_id,
        "simulation_case_id": simulation_case_id,
        "attempt_id": attempt_id,
        "command": list(command),
        "working_directory": str(work),
        "stdout_path": str(runtime / "stdout.log"),
        "stderr_path": str(runtime / "stderr.log"),
        "started_record": str(active / "child_started.json"),
        "exit_record": str(active / "child_exit.json"),
    }


def _probe_case_bundle_evidence(work: Path, case_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash the exact generated case inputs and copied work model."""
    case_path = work / "case.json"
    model_path = work / comsol_service.WORK_MODEL_FILENAME
    inputs = work / "inputs"
    declared = case_payload.get("input_files")
    if (
        not case_path.is_file()
        or case_path.is_symlink()
        or not model_path.is_file()
        or model_path.is_symlink()
        or not inputs.is_dir()
        or inputs.is_symlink()
        or not isinstance(declared, Mapping)
    ):
        message = "Timing-probe case bundle is missing, unsafe, or malformed."
        raise RuntimeError(message)
    entries = tuple(inputs.iterdir())
    if {entry.name for entry in entries} != set(declared) or any(not entry.is_file() or entry.is_symlink() for entry in entries):
        message = "Timing-probe input directory differs from exact generated membership."
        raise RuntimeError(message)
    admitted_files: dict[str, dict[str, Any]] = {}
    for name, raw_evidence in declared.items():
        path = inputs / str(name)
        if (
            not isinstance(name, str)
            or not isinstance(raw_evidence, Mapping)
            or set(raw_evidence) != {"sha256", "size_bytes"}
            or raw_evidence.get("sha256") != common.serialization.file_sha256(path)
            or raw_evidence.get("size_bytes") != path.stat().st_size
        ):
            message = "Timing-probe generated input evidence changed after preparation."
            raise RuntimeError(message)
        admitted_files[name] = dict(raw_evidence)
    return {
        "case_json_sha256": common.serialization.file_sha256(case_path),
        "work_model_sha256": common.serialization.file_sha256(model_path),
        "input_files": dict(sorted(admitted_files.items())),
    }


def _query_comsol_version(executable: str) -> dict[str, Any]:
    """Record bounded version-command evidence without claiming availability."""
    started = _utc_now()
    try:
        result = subprocess.run(  # noqa: S603 -- executable is admitted by Generation configuration
            [executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "argv": [executable, "-version"],
            "status": "recorded" if result.returncode == 0 else "command_failed",
            "exit_code": result.returncode,
            "stdout": result.stdout[:16_384],
            "stderr": result.stderr[:16_384],
            "started_at": started,
            "ended_at": _utc_now(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "argv": [executable, "-version"],
            "status": "unavailable",
            "exit_code": None,
            "stdout": "",
            "stderr": str(error)[:16_384],
            "started_at": started,
            "ended_at": _utc_now(),
        }


def _create_session(
    *,
    active: Path,
    probe_root: Path,
    work_root: Path | str | None,
    source: Path,
    probe: dict[str, Any],
    campaign: config_service.CampaignConfig,
    config: config_service.GenerationConfig,
    source_commit: str,
    run_key: str,
    scheduler_kind: str,
    requested_work_root: str | None,
) -> dict[str, Any]:
    """Create one fresh isolated probe session and deterministic case bundle."""
    active.mkdir(mode=0o700, parents=True, exist_ok=False)
    probe_id = f"probe-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{run_key[:10]}-{uuid.uuid4().hex[:8]}"
    if work_root is None:
        if requested_work_root is not None:
            message = "Internal timing work unexpectedly carries an external root identity."
            raise RuntimeError(message)
        owner_root = active
        work = active / "work"
    else:
        if requested_work_root is None or requested_work_root != _requested_work_root(work_root):
            message = "Timing work-root identity changed before session creation."
            raise RuntimeError(message)
        owner_root = Path(requested_work_root).resolve() / PROBE_SCOPE
        owner_root.mkdir(parents=True, exist_ok=True)
        work = owner_root / probe_id
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    bundle = case_service.generate_case_input_bundle(config, int(probe["case_index"]), work)
    shutil.copyfile(config.template_path, work / comsol_service.WORK_MODEL_FILENAME)
    runtime = work / "runtime"
    runtime.mkdir(mode=0o700)
    batch_log = runtime / "comsol_batch.log"
    command = comsol_service.build_comsol_command(
        config,
        cores_per_case=int(probe["cores_per_case"]),
        scalar_handoff=bundle.scalar_handoff,
        scheduler_kind=scheduler_kind,
        diagnostic_batchlog=str(batch_log),
    )
    attempt_id = common.serialization.canonical_json_sha256(
        {
            "probe_id": probe_id,
            "source_commit": source_commit,
            "case_input_id": bundle.case_input_id,
            "simulation_case_id": bundle.simulation_case_id,
            "command": command,
        }
    )
    control = _child_control(
        active=active,
        work=work,
        command=command,
        run_key=run_key,
        probe_id=probe_id,
        source_commit=source_commit,
        case_input_id=bundle.case_input_id,
        simulation_case_id=bundle.simulation_case_id,
        attempt_id=attempt_id,
    )
    _write_json(active / "child_control.json", control)
    session = {
        "schema_kind": "comsol_phase_timing_probe_session",
        "schema_version": PROBE_SCHEMA_VERSION,
        "run_key": run_key,
        "probe_id": probe_id,
        "source_commit": source_commit,
        "source_config": str(source),
        "source_config_sha256": common.serialization.file_sha256(source),
        "campaign_config": probe["campaign_config"],
        "campaign_digest": campaign.campaign_digest,
        "config_identity": common.serialization.canonical_json_sha256(probe),
        "batch_name": config.batch_name,
        "case_index": int(probe["case_index"]),
        "cores_per_case": int(probe["cores_per_case"]),
        "case_id": bundle.case_id,
        "case_input_id": bundle.case_input_id,
        "simulation_case_id": bundle.simulation_case_id,
        "attempt_id": attempt_id,
        "template": {
            "relative_path": config.template_relative_path,
            "sha256": config.template_sha256,
        },
        "case_bundle": _probe_case_bundle_evidence(work, bundle.case_payload),
        "command": command,
        "control_sha256": common.serialization.canonical_json_sha256(control),
        "scheduler_kind": scheduler_kind,
        "requested_work_root": requested_work_root,
        "comsol_version": _query_comsol_version(command[0]),
        "active_path": str(active),
        "work_path": str(work),
        "work_owner_root": str(owner_root),
        "probe_root": str(probe_root),
        "started_at": _utc_now(),
    }
    _write_json(active / "session.json", session)
    _write_json(
        active / "status.json",
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe_id": probe_id,
            "lifecycle": "prepared",
            "active_path": str(active),
            "work_path": str(work),
            "updated_at": _utc_now(),
        },
    )
    return session


def _load_exact_control(active: Path, session: Mapping[str, Any]) -> dict[str, Any]:
    """Load the exact child control and match its digest to reconstructed session fields."""
    control_path = active / "child_control.json"
    if not control_path.is_file() or control_path.is_symlink():
        message = "Timing-probe child control is missing or unsafe."
        raise RuntimeError(message)
    command = session.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and "\x00" not in value for value in command):
        message = "Persisted timing-probe command is malformed."
        raise RuntimeError(message)
    expected = _child_control(
        active=active,
        work=Path(str(session["work_path"])).resolve(),
        command=command,
        run_key=str(session["run_key"]),
        probe_id=str(session["probe_id"]),
        source_commit=str(session["source_commit"]),
        case_input_id=str(session["case_input_id"]),
        simulation_case_id=str(session["simulation_case_id"]),
        attempt_id=str(session["attempt_id"]),
    )
    control = _read_json_object(control_path)
    digest = common.serialization.canonical_json_sha256(control)
    if control != expected or digest != session.get("control_sha256"):
        message = "Timing-probe child control differs from its admitted session digest."
        raise RuntimeError(message)
    return control


def _load_session(
    active: Path,
    *,
    expected_run_key: str,
    expected_root: Path,
    source: Path,
    probe: Mapping[str, Any],
    campaign: config_service.CampaignConfig,
    config: config_service.GenerationConfig,
    source_commit: str,
    scheduler_kind: str,
    requested_work_root: str | None,
) -> dict[str, Any]:
    """Reconstruct and admit an interrupted session before any child execution."""
    session_path = active / "session.json"
    if (
        not active.is_dir()
        or active.is_symlink()
        or active.parent != expected_root / ".active"
        or not session_path.is_file()
        or session_path.is_symlink()
    ):
        message = "Persisted timing-probe active owner is missing or unsafe."
        raise RuntimeError(message)
    session = _read_json_object(session_path)
    required = {
        "schema_kind",
        "schema_version",
        "run_key",
        "probe_id",
        "source_commit",
        "source_config",
        "source_config_sha256",
        "campaign_config",
        "campaign_digest",
        "config_identity",
        "batch_name",
        "case_index",
        "cores_per_case",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "attempt_id",
        "template",
        "case_bundle",
        "command",
        "control_sha256",
        "scheduler_kind",
        "requested_work_root",
        "comsol_version",
        "active_path",
        "work_path",
        "work_owner_root",
        "probe_root",
        "started_at",
    }
    source_commit = source_service.validate_git_commit(source_commit)
    computed_run_key = _timing_run_key(
        source=source,
        probe=probe,
        campaign_digest=campaign.campaign_digest,
        source_commit=source_commit,
        scheduler_kind=scheduler_kind,
        requested_work_root=requested_work_root,
    )
    if (
        set(session) != required
        or session.get("schema_kind") != "comsol_phase_timing_probe_session"
        or session.get("schema_version") != PROBE_SCHEMA_VERSION
        or expected_run_key != computed_run_key
        or session.get("run_key") != computed_run_key
        or active.name != computed_run_key
        or Path(str(session.get("probe_root"))).resolve() != expected_root
        or Path(str(session.get("active_path"))).resolve() != active
        or session.get("source_commit") != source_commit
        or session.get("source_config") != str(source)
        or session.get("source_config_sha256") != common.serialization.file_sha256(source)
        or session.get("campaign_config") != probe["campaign_config"]
        or session.get("campaign_digest") != campaign.campaign_digest
        or session.get("config_identity") != common.serialization.canonical_json_sha256(dict(probe))
        or session.get("batch_name") != config.batch_name
        or session.get("case_index") != int(probe["case_index"])
        or session.get("cores_per_case") != int(probe["cores_per_case"])
        or session.get("scheduler_kind") != scheduler_kind
        or session.get("requested_work_root") != requested_work_root
        or not isinstance(session.get("comsol_version"), dict)
        or not isinstance(session.get("started_at"), str)
        or not session["started_at"]
    ):
        message = "Persisted timing-probe session identity is incompatible with this invocation."
        raise RuntimeError(message)
    probe_id = session.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id.startswith("probe-") or f"-{computed_run_key[:10]}-" not in probe_id:
        message = "Persisted timing-probe probe ID is malformed or unbound."
        raise RuntimeError(message)
    owner = active if requested_work_root is None else Path(requested_work_root).resolve() / PROBE_SCOPE
    work = Path(str(session.get("work_path"))).resolve()
    expected_work = active / "work" if requested_work_root is None else owner / probe_id
    if (
        Path(str(session.get("work_owner_root"))).resolve() != owner
        or work != expected_work
        or not work.is_dir()
        or work.is_symlink()
        or not (work == active / "work" or work.parent == owner)
    ):
        message = "Persisted timing-probe work directory escaped its exact owner."
        raise RuntimeError(message)
    template = session.get("template")
    expected_template = {
        "relative_path": config.template_relative_path,
        "sha256": config.template_sha256,
    }
    if template != expected_template or common.serialization.file_sha256(config.template_path) != config.template_sha256:
        message = "Timing-probe source template changed after session admission."
        raise RuntimeError(message)
    case_payload = _read_json_object(work / "case.json")
    case_service.validate_case_payload_schema(case_payload)
    case_input_id = case_service.compute_case_input_id(case_payload)
    simulation_case_id = case_service.compute_simulation_case_id(case_payload)
    case_id = config.case_id(int(probe["case_index"]))
    if (
        case_payload.get("case_id") != case_id
        or case_payload.get("case_index") != int(probe["case_index"])
        or case_payload.get("batch_id") != config.batch_id
        or case_payload.get("simulation_profile") != config.profile.id
        or case_payload.get("git_commit") != source_commit
        or case_payload.get("template")
        != {
            "relative_path": config.template_relative_path,
            "filename": config.template_path.name,
            "sha256": config.template_sha256,
        }
        or case_payload.get("case_input_id") != case_input_id
        or case_payload.get("simulation_case_id") != simulation_case_id
        or session.get("case_id") != case_id
        or session.get("case_input_id") != case_input_id
        or session.get("simulation_case_id") != simulation_case_id
    ):
        message = "Timing-probe generated case identity changed after preparation."
        raise RuntimeError(message)
    case_bundle = _probe_case_bundle_evidence(work, case_payload)
    if session.get("case_bundle") != case_bundle or case_bundle["work_model_sha256"] != config.template_sha256:
        message = "Timing-probe case bundle differs from its persisted inventory."
        raise RuntimeError(message)
    scalar_handoff = scalar_handoff_service.admit_case_scalar_handoff(case_payload, work / "inputs") if "scalar_handoff" in case_payload else None
    command = comsol_service.build_comsol_command(
        config,
        cores_per_case=int(probe["cores_per_case"]),
        scalar_handoff=scalar_handoff,
        scheduler_kind=scheduler_kind,
        diagnostic_batchlog=str(work / "runtime" / "comsol_batch.log"),
    )
    if session.get("command") != command:
        message = "Timing-probe command changed after deterministic reconstruction."
        raise RuntimeError(message)
    attempt_id = common.serialization.canonical_json_sha256(
        {
            "probe_id": probe_id,
            "source_commit": source_commit,
            "case_input_id": case_input_id,
            "simulation_case_id": simulation_case_id,
            "command": command,
        }
    )
    if session.get("attempt_id") != attempt_id:
        message = "Timing-probe attempt identity changed after reconstruction."
        raise RuntimeError(message)
    _load_exact_control(active, session)
    return session


def _pid_is_alive(pid: int) -> bool:
    """Return whether one recorded process identifier is still running."""
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        if waited_pid == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_or_admit_child(active: Path, session: Mapping[str, Any]) -> int:
    """Start one exact-bound probe child or admit its still-running process."""
    _load_exact_control(active, session)
    exit_record = active / "child_exit.json"
    if exit_record.is_file():
        return int(_read_json_object(exit_record)["exit_code"])
    started_record = active / "child_started.json"
    controller_record = active / "controller_child.json"
    for record_path in (started_record, controller_record):
        if not record_path.is_file():
            continue
        payload = _read_json_object(record_path)
        pid = payload.get("child_pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and _pid_is_alive(pid):
            return pid
    if started_record.exists() or controller_record.exists():
        msg = "The timing-probe child ended without durable exit evidence; work was preserved."
        raise RuntimeError(msg)
    command = [
        sys.executable,
        "-m",
        "src.generation.generation_timing_probe",
        "--execute-child",
        str(active / "child_control.json"),
    ]
    process = subprocess.Popen(  # noqa: S603 -- fixed Python module child boundary
        command,
        cwd=common.paths.get_project_root(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_json(
        controller_record,
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "child_pid": process.pid,
            "argv": command,
            "started_at": _utc_now(),
        },
    )
    return process.pid


def _load_observer(active: Path) -> tuple[dict[str, ProbeObservationState], list[dict[str, Any]]]:
    """Load the atomic observer checkpoint or return a fresh two-log state."""
    path = active / "observer.json"
    if not path.exists():
        return {"comsol_batch.log": ProbeObservationState(), "stdout.log": ProbeObservationState()}, []
    payload = _read_json_object(path)
    if set(payload) != {"schema_version", "states", "events"} or payload.get("schema_version") != PROBE_SCHEMA_VERSION:
        msg = "Timing-probe observer checkpoint has an invalid schema."
        raise ValueError(msg)
    states_payload = payload["states"]
    events = payload["events"]
    if not isinstance(states_payload, dict) or set(states_payload) != {"comsol_batch.log", "stdout.log"}:
        msg = "Timing-probe observer checkpoint has invalid log owners."
        raise ValueError(msg)
    if not isinstance(events, list) or len(events) > _MAX_EVENTS or not all(isinstance(event, dict) for event in events):
        msg = "Timing-probe observer events violate the bounded schema."
        raise ValueError(msg)
    return {name: _state_from_payload(value) for name, value in states_payload.items()}, events


def _persist_observer(
    active: Path,
    states: Mapping[str, ProbeObservationState],
    events: list[dict[str, Any]],
) -> None:
    """Atomically persist offsets and their corresponding bounded events together."""
    _write_json(
        active / "observer.json",
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "states": {name: _state_payload(state) for name, state in states.items()},
            "events": events[-_MAX_EVENTS:],
        },
    )


def _monitor_error_event(session: Mapping[str, Any], source: str, error: Exception) -> dict[str, Any]:
    """Return one bounded monitoring failure without affecting the COMSOL child."""
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe_id": session["probe_id"],
        "case_input_id": session["case_input_id"],
        "simulation_case_id": session["simulation_case_id"],
        "attempt_id": session["attempt_id"],
        "observed_monotonic_ns": time.monotonic_ns(),
        "observed_utc": _utc_now(),
        "source": source,
        "byte_offset": None,
        "line_number": None,
        "detected_phase": None,
        "phase": None,
        "event_kind": "monitor_error",
        "candidate_timing": None,
        "parse_status": "monitor_error",
        "exact_line": None,
        "diagnostic_only": True,
        "error_type": type(error).__name__,
        "message": str(error)[:2_048],
    }


def _observe_until_exit(active: Path, session: Mapping[str, Any], child_pid: int) -> tuple[int, list[dict[str, Any]]]:
    """Observe both growing logs until durable child exit evidence appears."""
    states, events = _load_observer(active)
    runtime = Path(str(session["work_path"])) / "runtime"
    sources = {
        "comsol_batch.log": runtime / "comsol_batch.log",
        "stdout.log": runtime / "stdout.log",
    }
    metadata = {
        "probe_id": str(session["probe_id"]),
        "case_input_id": str(session["case_input_id"]),
        "simulation_case_id": str(session["simulation_case_id"]),
        "attempt_id": str(session["attempt_id"]),
    }
    exit_path = active / "child_exit.json"
    while not exit_path.is_file():
        for name, path in sources.items():
            try:
                states[name], added = observe_appended_bytes(
                    path,
                    states[name],
                    probe_id=metadata["probe_id"],
                    case_input_id=metadata["case_input_id"],
                    simulation_case_id=metadata["simulation_case_id"],
                    attempt_id=metadata["attempt_id"],
                )
                events.extend(added)
                events = events[-_MAX_EVENTS:]
            except Exception as error:  # noqa: BLE001, PERF203 -- source failure cannot terminate COMSOL
                events.append(_monitor_error_event(session, name, error))
                events = events[-_MAX_EVENTS:]
        try:
            _persist_observer(active, states, events)
            _write_json(
                active / "status.json",
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "probe_id": session["probe_id"],
                    "lifecycle": "running",
                    "child_pid": child_pid,
                    "event_count": len(events),
                    "active_path": str(active),
                    "work_path": session["work_path"],
                    "updated_at": _utc_now(),
                },
            )
        except Exception as error:  # noqa: BLE001 -- persistence failure must not terminate COMSOL
            events.append(_monitor_error_event(session, "observer_persistence", error))
            events = events[-_MAX_EVENTS:]
        if not _pid_is_alive(child_pid) and not exit_path.is_file():
            time.sleep(POLL_INTERVAL_SECONDS)
            if not exit_path.is_file():
                msg = "Timing-probe child disappeared without durable exit evidence; work was preserved."
                raise RuntimeError(msg)
        time.sleep(POLL_INTERVAL_SECONDS)
    for name, path in sources.items():
        try:
            states[name], added = _flush_trailing_line(path, states[name], metadata=metadata)
            events.extend(added)
            events = events[-_MAX_EVENTS:]
        except Exception as error:  # noqa: BLE001, PERF203 -- final parse remains source-specific
            events.append(_monitor_error_event(session, name, error))
            events = events[-_MAX_EVENTS:]
    _persist_observer(active, states, events)
    exit_record = _read_json_object(exit_path)
    exit_code = exit_record.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        msg = "Timing-probe child exit evidence has an invalid exit code."
        raise TypeError(msg)
    return exit_code, events


def _observed_wall(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive diagnostic-only host-observed intervals from monotonic event times."""
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "diagnostic_only": True,
        "polling_interval_seconds": POLL_INTERVAL_SECONDS,
        "buffering_risk": True,
        "risks": [
            "COMSOL, process, and filesystem buffering can delay markers.",
            "Polling and parser latency are included in host-observed intervals.",
            "Intervals may include non-solver work around phase markers.",
        ],
    }
    ordered = sorted(
        (event for event in events if isinstance(event.get("observed_monotonic_ns"), int)),
        key=lambda event: int(event["observed_monotonic_ns"]),
    )
    for phase in _PHASES:
        start_kind = f"{phase}_phase_first_observed"
        completion_kind = f"{phase}_completion_observed"
        starts = [event for event in ordered if event.get("event_kind") == start_kind]
        start_ns = int(starts[0]["observed_monotonic_ns"]) if starts else None
        completions = [
            event
            for event in ordered
            if event.get("event_kind") == completion_kind and start_ns is not None and int(event["observed_monotonic_ns"]) >= start_ns
        ]
        completion_ns = int(completions[0]["observed_monotonic_ns"]) if completions else None
        seconds = None if start_ns is None or completion_ns is None else (completion_ns - start_ns) / 1_000_000_000.0
        payload[f"{phase}_phase_first_observed_monotonic_ns"] = start_ns
        payload[f"{phase}_completion_observed_monotonic_ns"] = completion_ns
        field = "stationary_airflow_observed_wall_seconds" if phase == "stationary" else "transient_drying_observed_wall_seconds"
        payload[field] = seconds
        payload[f"{phase}_available"] = seconds is not None and math.isfinite(seconds) and seconds >= 0.0
    return payload


def _candidate_keys_from_events(events: list[dict[str, Any]], source: str) -> set[tuple[Any, ...]]:
    """Return stable incremental timing-candidate keys for one current log generation."""
    source_events = [event for event in events if event.get("source") == source]
    current_generation = max((int(event.get("generation", 0)) for event in source_events), default=0)
    return {
        (
            event.get("byte_offset"),
            event.get("line_number"),
            event.get("exact_line"),
            event.get("detected_phase"),
            event.get("parse_status"),
        )
        for event in source_events
        if event.get("generation", 0) == current_generation
        and event.get("event_kind")
        in {
            "stationary_completion_observed",
            "transient_completion_observed",
            "unassigned_timing_candidate",
        }
    }


def _candidate_keys_from_final(records: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    """Return stable final-file timing-candidate keys."""
    return {
        (
            record.get("byte_offset"),
            record.get("line_number"),
            record.get("exact_line"),
            record.get("detected_phase"),
            record.get("parse_status"),
        )
        for record in records
    }


def _comparison(events: list[dict[str, Any]], source: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report both directions of disagreement between incremental and final parses."""
    incremental = _candidate_keys_from_events(events, source)
    final = _candidate_keys_from_final(records)
    return {
        "source": source,
        "exact_match": incremental == final,
        "incremental_candidate_count": len(incremental),
        "final_candidate_count": len(final),
        "missing_from_incremental": [list(item) for item in sorted(final - incremental, key=repr)],
        "absent_from_final": [list(item) for item in sorted(incremental - final, key=repr)],
    }


def _method_verdicts(batch_summary: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact conservative method verdict surface."""
    same_process = {
        "candidate": "B",
        "method_status": "not_implementable_from_current_source_boundary",
        "exact_boundary_proven": False,
        "stationary_available": False,
        "transient_available": False,
        "scientific_equivalence_status": "not_assessed",
        "unavailable_reason": (
            "Exact top-level stationary and transient solve calls are inside the binary MPH execution boundary, "
            "not source-controlled Python code. No second process or shell-marker surrogate was introduced."
        ),
    }
    observed_method = {
        "candidate": "C",
        "method_status": "diagnostic_only",
        "stationary_available": observed["stationary_available"],
        "transient_available": observed["transient_available"],
        "diagnostic_only": True,
        "polling_interval_seconds": POLL_INTERVAL_SECONDS,
        "buffering_risk": True,
    }
    recommendation = "comsol_batch_log_solution_time" if batch_summary["method_status"] == "confirmed" else "unresolved"
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "recommendation": recommendation,
        "batch_log_method": dict(batch_summary),
        "same_process_method": same_process,
        "observed_wall_method": observed_method,
        "production_timing_fields_updated": False,
    }


def _probe_readme() -> str:
    """Return bundle-local interpretation guidance."""
    return (
        "# COMSOL phase timing probe\n\n"
        "This immutable bundle is diagnostic-only. Candidate A conservatively parses one complete case-owned batch "
        "log. Candidate B is unavailable because the two exact solve-call boundaries are hidden inside the MPH "
        "execution. Candidate C contains host-observed marker intervals with polling and buffering risk. No value in "
        "this bundle is automatically a production solver runtime or speedup numerator. Synthetic parser tests do "
        "not establish real COMSOL log grammar. See method_verdicts.json and parser_summary.json before interpreting "
        "any candidate.\n"
    )


def _require_bounded_log(path: Path) -> bytes:
    """Read one complete retained log only when it satisfies the bounded contract."""
    if not path.exists():
        return b""
    size = path.stat().st_size
    if size > _MAX_LOG_BYTES:
        msg = f"Complete probe log exceeds the {_MAX_LOG_BYTES}-byte bundle bound: {path}."
        raise ValueError(msg)
    return path.read_bytes()


def _file_evidence(path: Path) -> dict[str, Any]:
    """Return exact size and digest evidence for one regular file."""
    if path.is_symlink() or not path.is_file():
        msg = f"Probe bundle member is not one regular non-symlink file: {path}."
        raise ValueError(msg)
    return {"size_bytes": path.stat().st_size, "sha256": common.serialization.file_sha256(path)}


def _publish_bundle(
    *,
    root: Path,
    active: Path,
    session: Mapping[str, Any],
    exit_code: int,
    events: list[dict[str, Any]],
) -> Path:
    """Publish one exact immutable diagnostic bundle from retained active evidence."""
    probe_id = str(session["probe_id"])
    destination = root / probe_id
    if destination.exists():
        validate_probe_bundle(destination)
        return destination
    staging_root = root / ".staging"
    staging_root.mkdir(mode=0o700, exist_ok=True)
    staging = staging_root / probe_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    runtime = Path(str(session["work_path"])) / "runtime"
    log_payloads = {
        "comsol_batch.log": _require_bounded_log(runtime / "comsol_batch.log"),
        "stdout.log": _require_bounded_log(runtime / "stdout.log"),
        "stderr.log": _require_bounded_log(runtime / "stderr.log"),
    }
    for name, payload in log_payloads.items():
        common.serialization.atomic_write_bytes(staging / name, payload)
    batch_records = parse_solution_times(log_payloads["comsol_batch.log"], source="comsol_batch.log")
    stdout_records = parse_solution_times(log_payloads["stdout.log"], source="stdout.log")
    batch_summary = summarize_solution_times(batch_records)
    stdout_summary = summarize_solution_times(stdout_records)
    observed = _observed_wall(events)
    parser_summary = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "incremental_event_count": len(events),
        "monitor_error_count": sum(event.get("event_kind") == "monitor_error" for event in events),
        "batch_log": _comparison(events, "comsol_batch.log", batch_records),
        "stdout": _comparison(events, "stdout.log", stdout_records),
        "evidence_policy": "final and incremental parses are retained separately; disagreements are never hidden",
    }
    _write_events(staging / "phase_events.jsonl", events)
    _write_json(
        staging / "batch_log_candidates.json",
        {"schema_version": PROBE_SCHEMA_VERSION, "records": batch_records, "summary": batch_summary},
    )
    _write_json(
        staging / "stdout_candidates.json",
        {"schema_version": PROBE_SCHEMA_VERSION, "records": stdout_records, "summary": stdout_summary},
    )
    _write_json(staging / "observed_wall_timing.json", observed)
    _write_json(staging / "method_verdicts.json", _method_verdicts(batch_summary, observed))
    _write_json(staging / "parser_summary.json", parser_summary)
    _write_json(
        staging / "exact_command.json",
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "argv": session["command"],
            "working_directory": session["work_path"],
            "cores_per_case": session["cores_per_case"],
            "batch_log_path": str(Path(str(session["work_path"])) / "runtime" / "comsol_batch.log"),
        },
    )
    child_exit = _read_json_object(active / "child_exit.json")
    _write_json(
        staging / "environment.json",
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "platform": sys.platform,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "loaded_modules": os.environ.get("LOADEDMODULES"),
            "comsol_version": session["comsol_version"],
        },
    )
    common.serialization.atomic_write_text(staging / "README.md", _probe_readme())
    content_names = _EXACT_BUNDLE_INVENTORY - {"manifest.json", "sha256sums.txt"}
    content_files = {name: _file_evidence(staging / name) for name in sorted(content_names)}
    ended_at = str(child_exit.get("ended_at") or _utc_now())
    _write_json(
        staging / "manifest.json",
        {
            "schema_kind": PROBE_SCHEMA_KIND,
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe_id": probe_id,
            "diagnostic_only": True,
            "source_commit": session["source_commit"],
            "source_config": session["source_config"],
            "source_config_sha256": session["source_config_sha256"],
            "campaign_config": session["campaign_config"],
            "campaign_digest": session["campaign_digest"],
            "config_identity": session["config_identity"],
            "batch_name": session["batch_name"],
            "case_index": session["case_index"],
            "case_id": session["case_id"],
            "case_input_id": session["case_input_id"],
            "simulation_case_id": session["simulation_case_id"],
            "attempt_id": session["attempt_id"],
            "template": session["template"],
            "comsol_version": session["comsol_version"],
            "host": socket.gethostname(),
            "cores_per_case": session["cores_per_case"],
            "exact_command": session["command"],
            "exit_status": {"exit_code": exit_code, "status": "success" if exit_code == 0 else "failed"},
            "started_at": session["started_at"],
            "ended_at": ended_at,
            "files": content_files,
        },
    )
    checksummed = _EXACT_BUNDLE_INVENTORY - {"sha256sums.txt"}
    sums = "".join(f"{common.serialization.file_sha256(staging / name)}  {name}\n" for name in sorted(checksummed))
    common.serialization.atomic_write_text(staging / "sha256sums.txt", sums)
    validate_probe_bundle(staging, require_immutable=False)
    for item in staging.iterdir():
        item.chmod(0o444)
    staging.replace(destination)
    destination.chmod(0o555)
    validate_probe_bundle(destination)
    return destination


def validate_probe_bundle(
    bundle_path: Path | str,
    *,
    require_immutable: bool = True,
) -> dict[str, Any]:
    """Validate exact inventory, size, identity, digest, and immutability evidence."""
    supplied = Path(bundle_path).expanduser()
    if supplied.is_symlink():
        msg = "Probe bundle root cannot be a symbolic link."
        raise ValueError(msg)
    bundle = supplied.resolve()
    if not bundle.is_dir():
        msg = f"Probe bundle is not a directory: {bundle}."
        raise NotADirectoryError(msg)
    entries = list(bundle.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        msg = "Probe bundle may contain only regular non-symlink files."
        raise ValueError(msg)
    actual = {item.name for item in entries}
    if actual != _EXACT_BUNDLE_INVENTORY:
        msg = f"Probe bundle inventory differs from the exact contract: {sorted(actual ^ _EXACT_BUNDLE_INVENTORY)!r}."
        raise ValueError(msg)
    total_size = sum(item.stat().st_size for item in entries)
    if total_size > _MAX_BUNDLE_BYTES:
        msg = f"Probe bundle exceeds the {_MAX_BUNDLE_BYTES}-byte total bound."
        raise ValueError(msg)
    for name in ("comsol_batch.log", "stdout.log", "stderr.log"):
        if (bundle / name).stat().st_size > _MAX_LOG_BYTES:
            msg = f"Probe log {name!r} exceeds the complete-log size bound."
            raise ValueError(msg)
    checksum_lines = (bundle / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    sums: dict[str, str] = {}
    for line in checksum_lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) in sums:
            msg = "Probe checksum inventory is malformed or duplicated."
            raise ValueError(msg)
        sums[match.group(2)] = match.group(1)
    expected_sums = _EXACT_BUNDLE_INVENTORY - {"sha256sums.txt"}
    if set(sums) != expected_sums:
        msg = "Probe checksum inventory is incomplete or contains unexpected members."
        raise ValueError(msg)
    for name, expected_digest in sums.items():
        if common.serialization.file_sha256(bundle / name) != expected_digest:
            msg = f"Probe bundle SHA-256 mismatch for {name!r}."
            raise ValueError(msg)
    manifest = _read_json_object(bundle / "manifest.json")
    if (
        manifest.get("schema_kind") != PROBE_SCHEMA_KIND
        or manifest.get("schema_version") != PROBE_SCHEMA_VERSION
        or manifest.get("diagnostic_only") is not True
        or not isinstance(manifest.get("probe_id"), str)
        or _SHA256.fullmatch(str(manifest.get("case_input_id"))) is None
        or _SHA256.fullmatch(str(manifest.get("simulation_case_id"))) is None
        or _SHA256.fullmatch(str(manifest.get("attempt_id"))) is None
    ):
        msg = "Probe manifest identity evidence is malformed."
        raise ValueError(msg)
    files = manifest.get("files")
    content_names = _EXACT_BUNDLE_INVENTORY - {"manifest.json", "sha256sums.txt"}
    if not isinstance(files, dict) or set(files) != content_names:
        msg = "Probe manifest file-size and digest inventory is invalid."
        raise ValueError(msg)
    for name in content_names:
        evidence = files[name]
        actual_evidence = _file_evidence(bundle / name)
        if not isinstance(evidence, dict) or evidence != actual_evidence:
            msg = f"Probe manifest file evidence disagrees for {name!r}."
            raise ValueError(msg)
    if require_immutable:
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        if bundle.stat().st_mode & write_bits:
            msg = "Published probe bundle directory is writable."
            raise ValueError(msg)
        if any(item.stat().st_mode & write_bits for item in entries):
            msg = "Published probe bundle contains a writable file."
            raise ValueError(msg)
    return {
        "probe_bundle": str(bundle),
        "probe_id": manifest["probe_id"],
        "valid": True,
        "inventory": sorted(_EXACT_BUNDLE_INVENTORY),
        "total_size_bytes": total_size,
    }


def _cleanup_completed_session(active: Path, session: Mapping[str, Any]) -> None:
    """Remove only the validated session-owned active and work directories."""
    work = Path(str(session["work_path"])).resolve()
    owner = Path(str(session["work_owner_root"])).resolve()
    active_resolved = active.resolve()
    if work != active_resolved and not work.is_relative_to(active_resolved):
        if work.parent != owner or work.name != session["probe_id"]:
            msg = "Refusing to clean a probe work directory outside its exact session owner."
            raise RuntimeError(msg)
        shutil.rmtree(work)
    shutil.rmtree(active)


def run_timing_probe(
    config_path: Path | str,
    *,
    storage_root: Path | str,
    work_root: Path | str | None = None,
    announce: Callable[[Mapping[str, str]], None] | None = None,
) -> dict[str, Any]:
    """
    Run or safely resume one isolated transient timing probe.

    The independent child owns the COMSOL process and durable exit record. A
    restarted controller resumes appended-byte observation without launching a
    second case. Successful or failed process evidence is published only through
    the exact immutable diagnostic bundle contract.
    """
    source, probe = _probe_config(config_path)
    campaign = config_service.load_campaign_config(probe["campaign_config"])
    config = campaign.batch(str(probe["batch_name"]))
    case_index = int(probe["case_index"])
    if (
        campaign.campaign_purpose != "technical_runtime_smoke"
        or config.profile.id != profiles.TRANSIENT_DRYING_PROFILE
        or case_index not in config.case_indices
    ):
        msg = "Timing probe requires one configured transient technical-runtime-smoke case."
        raise ValueError(msg)
    source_commit = source_service.required_git_commit()
    scheduler_kind = _scheduler_kind()
    requested_work_root = _requested_work_root(work_root)
    root = _probe_root(storage_root)
    run_key = _timing_run_key(
        source=source,
        probe=probe,
        campaign_digest=campaign.campaign_digest,
        source_commit=source_commit,
        scheduler_kind=scheduler_kind,
        requested_work_root=requested_work_root,
    )
    active = root / ".active" / run_key
    lock = root / ".locks" / f"{run_key}.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        if active.exists():
            session = _load_session(
                active,
                expected_run_key=run_key,
                expected_root=root,
                source=source,
                probe=probe,
                campaign=campaign,
                config=config,
                source_commit=source_commit,
                scheduler_kind=scheduler_kind,
                requested_work_root=requested_work_root,
            )
            resumed = True
        else:
            session = _create_session(
                active=active,
                probe_root=root,
                work_root=work_root,
                source=source,
                probe=probe,
                campaign=campaign,
                config=config,
                source_commit=source_commit,
                run_key=run_key,
                scheduler_kind=scheduler_kind,
                requested_work_root=requested_work_root,
            )
            session = _load_session(
                active,
                expected_run_key=run_key,
                expected_root=root,
                source=source,
                probe=probe,
                campaign=campaign,
                config=config,
                source_commit=source_commit,
                scheduler_kind=scheduler_kind,
                requested_work_root=requested_work_root,
            )
            resumed = False
        if announce is not None:
            announce(
                {
                    "probe_id": str(session["probe_id"]),
                    "probe_active": str(active),
                    "probe_work": str(session["work_path"]),
                }
            )
        child_pid = _start_or_admit_child(active, session)
        exit_code, events = _observe_until_exit(active, session, child_pid)
        bundle = _publish_bundle(
            root=root,
            active=active,
            session=session,
            exit_code=exit_code,
            events=events,
        )
        validate_probe_bundle(bundle)
        _cleanup_completed_session(active, session)
        return {
            "probe_id": session["probe_id"],
            "probe_bundle": str(bundle),
            "exit_code": exit_code,
            "resumed": resumed,
        }


def _session_and_control_for_child(control_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-admit authoritative config, case, and command evidence at the child boundary."""
    candidate = control_path.expanduser()
    if candidate.name != "child_control.json" or not candidate.is_file() or candidate.is_symlink() or candidate.parent.is_symlink():
        message = "Timing-probe child control path is missing or unsafe."
        raise RuntimeError(message)
    active = candidate.parent.resolve()
    if active.parent.name != ".active":
        message = "Timing-probe child control is outside the exact active owner."
        raise RuntimeError(message)
    session_path = active / "session.json"
    if not session_path.is_file() or session_path.is_symlink():
        message = "Timing-probe child session is missing or unsafe."
        raise RuntimeError(message)
    untrusted = _read_json_object(session_path)
    source_value = untrusted.get("source_config")
    requested_work_root = untrusted.get("requested_work_root")
    if not isinstance(source_value, str) or (
        requested_work_root is not None
        and (not isinstance(requested_work_root, str) or _requested_work_root(requested_work_root) != requested_work_root)
    ):
        message = "Timing-probe child session has malformed source or work-root identity."
        raise RuntimeError(message)
    source, probe = _probe_config(source_value)
    campaign = config_service.load_campaign_config(probe["campaign_config"])
    config = campaign.batch(str(probe["batch_name"]))
    case_index = int(probe["case_index"])
    if (
        campaign.campaign_purpose != "technical_runtime_smoke"
        or config.profile.id != profiles.TRANSIENT_DRYING_PROFILE
        or case_index not in config.case_indices
    ):
        message = "Timing-probe child requires the configured transient technical-runtime-smoke case."
        raise RuntimeError(message)
    source_commit = source_service.required_git_commit()
    scheduler_kind = _scheduler_kind()
    session = _load_session(
        active,
        expected_run_key=active.name,
        expected_root=active.parent.parent,
        source=source,
        probe=probe,
        campaign=campaign,
        config=config,
        source_commit=source_commit,
        scheduler_kind=scheduler_kind,
        requested_work_root=requested_work_root,
    )
    return session, _load_exact_control(active, session)


def _execute_child(control_path: Path) -> int:
    """Execute only the exact command reconstructed from admitted session evidence."""
    session, control = _session_and_control_for_child(control_path)
    work = Path(str(session["work_path"])).resolve()
    runtime = work / "runtime"
    if not runtime.is_dir() or runtime.is_symlink():
        message = "Timing-probe runtime directory is missing or unsafe."
        raise RuntimeError(message)
    paths = {
        "stdout_path": runtime / "stdout.log",
        "stderr_path": runtime / "stderr.log",
    }
    started_record = Path(str(control["started_record"])).resolve()
    exit_record = Path(str(control["exit_record"])).resolve()
    started_at = _utc_now()
    exit_code = 125
    error_payload: dict[str, Any] | None = None
    try:
        with paths["stdout_path"].open("wb") as stdout, paths["stderr_path"].open("wb") as stderr:
            process = subprocess.Popen(  # noqa: S603 -- argv is an exact persisted Generation-owned command
                control["command"],
                cwd=work,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            _write_json(
                started_record,
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "child_pid": os.getpid(),
                    "comsol_pid": process.pid,
                    "started_at": started_at,
                },
            )
            exit_code = process.wait()
    except Exception as error:  # noqa: BLE001 -- child must persist subprocess launch failures
        error_payload = {"error_type": type(error).__name__, "message": str(error)[:2_048]}
    _write_json(
        exit_record,
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "child_pid": os.getpid(),
            "exit_code": exit_code,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "child_error": error_payload,
        },
    )
    return exit_code


def _module_main(argv: list[str] | None = None) -> int:
    """Run only the private durable child boundary."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--execute-child", type=Path, required=True)
    args = parser.parse_args(argv)
    return _execute_child(args.execute_child)


if __name__ == "__main__":
    raise SystemExit(_module_main())
