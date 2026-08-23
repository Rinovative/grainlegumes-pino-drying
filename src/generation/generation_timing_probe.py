"""
generation_timing_probe.py

Observe one isolated normal Generation case for COMSOL phase-timing evidence.

Responsibilities:
  - Select one ordinary transient technical-smoke campaign case
  - Use canonical input generation and the normal case runtime lifecycle
  - Retain complete batch-log, stdout, stderr, and normal outcome evidence
  - Publish and transfer one bounded immutable diagnostic bundle

Design principles:
  - Case preparation, COMSOL execution, collection, publication, and cleanup remain normal
  - Diagnostic evidence never populates production solver-timing fields
  - Final-file and incremental parses remain separate, comparable evidence
  - Publication is fail-closed for identity, inventory, size, and digest integrity

This module does NOT:
  - Publish a case or Dataset package outside its temporary isolated storage
  - Invent source-level solve boundaries hidden inside the MPH executable
  - Claim that synthetic parser tests validate real COMSOL log grammar
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.runtime import generation_runtime_batch as batch_service
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_comsol as comsol_service
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

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
_MAX_SHELL_EXIT_CODE: Final = 255
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
_TIMING_PREFIX: Final = re.compile(
    r"(?P<label>solution|elapsed|computation)\s+time\s*:",
    re.IGNORECASE,
)
_TIMING_VALUE: Final = re.compile(
    r"(?P<label>solution|elapsed|computation)\s+time\s*:\s*"
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
    """Classify one possible COMSOL timing line conservatively."""
    prefix = _TIMING_PREFIX.search(line)
    if prefix is None:
        return None
    match = _TIMING_VALUE.search(line)
    timing_expression = f"{prefix.group('label').casefold()}_time"
    parsed_value: float | None = None
    parsed_unit: str | None = None
    seconds: float | None = None
    parse_status = "malformed"
    ambiguity_reasons: list[str] = []
    if match is not None:
        timing_expression = f"{match.group('label').casefold()}_time"
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
        "timing_expression": timing_expression,
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
            record["timing_expression"],
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
    """Return diagnostic per-phase Candidate A evidence without validating real grammar."""
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
            and not isinstance(record.get("converted_seconds"), bool)
            and math.isfinite(float(record["converted_seconds"]))
            and float(record["converted_seconds"]) >= 0.0
        ]
        parse_statuses = {str(record.get("parse_status")) for record in candidates}
        if not candidates:
            status = "ambiguous" if unknown else "missing"
        elif len(candidates) == 1 and len(valid) == 1 and not unknown:
            status = "single_candidate"
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
            "single_candidate_count": len(valid),
            "duplicate_count": sum(record.get("duplicate_classification") == "duplicate" for record in candidates),
            "ambiguous_count": sum(bool(record.get("ambiguity_reasons")) for record in candidates),
            "ambiguity_reasons": sorted(set(reasons)),
        }
        if status == "single_candidate":
            selected[phase] = valid[0]
    ordered = all(phase_results[phase]["status"] == "single_candidate" for phase in _PHASES)
    if ordered and int(selected["stationary"]["line_number"]) >= int(selected["transient"]["line_number"]):
        ordered = False
        for phase in _PHASES:
            phase_results[phase]["status"] = "ambiguous"
            phase_results[phase]["ambiguity_reasons"] = [
                *phase_results[phase]["ambiguity_reasons"],
                "stationary_not_before_transient",
            ]
    candidate_sum = None
    if ordered:
        candidate_sum = sum(float(selected[phase]["converted_seconds"]) for phase in _PHASES)
    return {
        "candidate": "A",
        "method_status": "candidate_evidence_available" if candidate_sum is not None else "unresolved",
        "diagnostic_only": True,
        "real_comsol_grammar_validated": False,
        "phases": phase_results,
        "diagnostic_candidate_sum_seconds": candidate_sum,
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


def _probe_root(storage_root: Path | str) -> Path:
    """Resolve and create the experiment-owned probe root."""
    root = common.paths.get_experiments_root(storage_root=storage_root).resolve() / PROBE_SCOPE
    root.mkdir(parents=True, exist_ok=True)
    return root


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


def _monitor_error_event(session: Mapping[str, Any], source: str, error: Exception) -> dict[str, Any]:
    """Return one bounded monitoring failure without affecting the COMSOL process."""
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
        completions = [event for event in ordered if event.get("event_kind") == completion_kind]
        start_ns = int(starts[0]["observed_monotonic_ns"]) if starts else None
        completion_ns = None
        if start_ns is not None:
            completion_ns = next(
                (int(event["observed_monotonic_ns"]) for event in completions if int(event["observed_monotonic_ns"]) >= start_ns),
                None,
            )
        seconds = None if start_ns is None or completion_ns is None else (completion_ns - start_ns) / 1_000_000_000
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
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "recommendation": "unresolved_pending_real_probe_review",
        "real_comsol_grammar_validated": False,
        "batch_log_method": dict(batch_summary),
        "same_process_method": same_process,
        "observed_wall_method": observed_method,
        "production_timing_fields_updated": False,
    }


def _probe_readme() -> str:
    """Return bundle-local interpretation guidance."""
    return (
        "# COMSOL phase timing probe\n\n"
        "This immutable bundle came from one ordinary transient Generation case executed through canonical input "
        "generation, scratch preparation, the standard COMSOL command and lifecycle, export collection, normal "
        "publication or attempt recording, and normal scratch cleanup. It is diagnostic-only. Candidate A retains "
        "possible Solution time, Elapsed time, and Computation time lines without claiming that synthetic tests "
        "validate real COMSOL grammar. Candidate B is unavailable because the two exact solve-call boundaries are "
        "hidden inside the MPH execution. Candidate C contains host-observed marker intervals with polling and "
        "buffering risk. No value in this bundle is automatically a production solver runtime or speedup numerator. "
        "See method_verdicts.json and parser_summary.json before interpretation.\n"
    )


def _require_bounded_log(path: Path) -> bytes:
    """Read one complete retained log only when it satisfies the bounded contract."""
    if not path.exists():
        return b""
    if path.is_symlink() or not path.is_file():
        msg = f"Probe log is not one regular non-symlink file: {path}."
        raise ValueError(msg)
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


def _bounded_json_evidence(path: Path) -> dict[str, Any]:
    """Return bounded JSON payload and exact file identity."""
    if path.stat().st_size > 8 * 1024 * 1024:
        msg = f"Normal case JSON evidence is unexpectedly large: {path}."
        raise ValueError(msg)
    return {"file": _file_evidence(path), "payload": _read_json_object(path)}


def _admit_embedded_json_evidence(value: object, *, label: str) -> Mapping[str, Any]:
    """Admit one bounded retained JSON payload and its source-file identity."""
    if not isinstance(value, Mapping) or set(value) != {"file", "payload"}:
        msg = f"{label} is not one exact retained JSON evidence object."
        raise ValueError(msg)
    file_evidence = value["file"]
    payload = value["payload"]
    if (
        not isinstance(file_evidence, Mapping)
        or set(file_evidence) != {"size_bytes", "sha256"}
        or isinstance(file_evidence.get("size_bytes"), bool)
        or not isinstance(file_evidence.get("size_bytes"), int)
        or not 0 <= int(file_evidence["size_bytes"]) <= 8 * 1024 * 1024
        or _SHA256.fullmatch(str(file_evidence.get("sha256"))) is None
        or not isinstance(payload, Mapping)
    ):
        msg = f"{label} file identity or JSON payload is malformed."
        raise ValueError(msg)
    return payload


class _RuntimeProbeObserver:
    """Observe one ordinary prepared case without owning its execution lifecycle."""

    def __init__(self, active: Path, session: dict[str, Any]) -> None:
        """Initialize bounded in-memory and active-session evidence."""
        self.active = active
        self.session = session
        self.states = {
            "comsol_batch.log": ProbeObservationState(),
            "stdout.log": ProbeObservationState(),
        }
        self.paths: dict[str, Path] = {}
        self.events: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.finished = False

    def __call__(self, stage: str, prepared: Any, payload: Mapping[str, Any]) -> None:
        """Admit the prepared callback or retain complete final runtime logs."""
        if stage == "prepared":
            self._prepared(prepared, payload)
            return
        if stage == "finished":
            self.finish(payload)
            return
        msg = f"Unsupported timing-probe observer stage: {stage!r}."
        raise ValueError(msg)

    def _prepared(self, prepared: Any, payload: Mapping[str, Any]) -> None:
        """Bind the exact normal workspace, inputs, and command before launch."""
        if self.thread is not None:
            msg = "Timing-probe observer received duplicate prepared evidence."
            raise RuntimeError(msg)
        command_value = payload.get("command")
        if not isinstance(command_value, (list, tuple)) or not all(isinstance(item, str) for item in command_value):
            msg = "Timing-probe command evidence is malformed."
            raise TypeError(msg)
        command = list(command_value)
        batch_log = prepared.runtime_directory / "comsol_batch.log"
        if (
            payload.get("batch_log_path") != str(batch_log)
            or command.count("-batchlog") != 1
            or command.count("-batchlogout") != 1
            or command[command.index("-batchlog") + 1] != str(batch_log)
        ):
            msg = "Normal COMSOL command does not own exactly one case-local batch log."
            raise RuntimeError(msg)
        case_inputs = prepared.bundle.case_payload.get("input_files")
        if not isinstance(case_inputs, Mapping) or not case_inputs:
            msg = "Prepared normal case has no declared input-file identity map."
            raise TypeError(msg)
        workspace_inputs: dict[str, dict[str, Any]] = {}
        for name, identity in sorted(case_inputs.items()):
            if not isinstance(name, str) or not isinstance(identity, Mapping):
                msg = "Prepared input-file identity is malformed."
                raise TypeError(msg)
            candidate = prepared.work_directory / name
            evidence = _file_evidence(candidate)
            expected = {"size_bytes": identity.get("size_bytes"), "sha256": identity.get("sha256")}
            if evidence != expected:
                msg = f"Prepared normal input changed before COMSOL launch: {candidate}."
                raise RuntimeError(msg)
            workspace_inputs[name] = {**evidence, "workspace_path": str(candidate)}
        if {path.name for path in prepared.bundle.input_paths} != set(workspace_inputs):
            msg = "Prepared normal case input membership disagrees with its declared files."
            raise RuntimeError(msg)
        attempt_id = common.serialization.canonical_json_sha256(
            {
                "probe_id": self.session["probe_id"],
                "source_commit": self.session["source_commit"],
                "case_input_id": prepared.bundle.case_input_id,
                "simulation_case_id": prepared.bundle.simulation_case_id,
                "command": command,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
        )
        self.session.update(
            {
                "case_input_id": prepared.bundle.case_input_id,
                "simulation_case_id": prepared.bundle.simulation_case_id,
                "attempt_id": attempt_id,
                "input_generation_id": prepared.input_generation_id,
                "canonical_raw_case": str(prepared.canonical_raw_directory),
                "work_path": str(prepared.work_directory),
                "work_root": str(prepared.work_root),
                "command": command,
                "batch_log_path": str(batch_log),
                "workspace_input_files": workspace_inputs,
            }
        )
        self.paths = {
            "comsol_batch.log": batch_log,
            "stdout.log": prepared.runtime_directory / "stdout.log",
            "stderr.log": prepared.runtime_directory / "stderr.log",
        }
        _write_json(self.active / "session.json", self.session)
        self.thread = threading.Thread(
            target=self._watch,
            name=f"timing-probe-{self.session['probe_id']}",
            daemon=True,
        )
        self.thread.start()

    def _metadata(self) -> dict[str, str]:
        """Return identities bound into every incremental event."""
        return {
            "probe_id": str(self.session["probe_id"]),
            "case_input_id": str(self.session["case_input_id"]),
            "simulation_case_id": str(self.session["simulation_case_id"]),
            "attempt_id": str(self.session["attempt_id"]),
        }

    def _observe_once(self) -> None:
        """Observe each growing source once and persist bounded active evidence."""
        metadata = self._metadata()
        for name in ("comsol_batch.log", "stdout.log"):
            path = self.paths[name]
            try:
                self.states[name], added = observe_appended_bytes(
                    path,
                    self.states[name],
                    probe_id=metadata["probe_id"],
                    case_input_id=metadata["case_input_id"],
                    simulation_case_id=metadata["simulation_case_id"],
                    attempt_id=metadata["attempt_id"],
                )
                self.events.extend(added)
            except Exception as error:  # noqa: BLE001 -- observation cannot terminate the normal case
                self.events.append(_monitor_error_event(self.session, name, error))
            self.events = self.events[-_MAX_EVENTS:]
        try:
            _write_json(
                self.active / "observer_state.json",
                {name: _state_payload(state) for name, state in self.states.items()},
            )
            _write_events(self.active / "phase_events.jsonl", self.events)
        except Exception as error:  # noqa: BLE001 -- active persistence remains diagnostic
            self.events.append(_monitor_error_event(self.session, "observer_persistence", error))
            self.events = self.events[-_MAX_EVENTS:]

    def _watch(self) -> None:
        """Poll only while the normal case runner owns the process."""
        while not self.stop.wait(POLL_INTERVAL_SECONDS):
            self._observe_once()

    def finish(self, payload: Mapping[str, Any]) -> None:
        """Stop observation and retain complete normal runtime logs before cleanup."""
        if self.finished:
            return
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, POLL_INTERVAL_SECONDS * 4.0))
            if self.thread.is_alive():
                self.events.append(
                    _monitor_error_event(
                        self.session,
                        "observer_thread",
                        RuntimeError("Timing-probe observer thread did not stop before final retention."),
                    )
                )
            self._observe_once()
            metadata = self._metadata()
            for name in ("comsol_batch.log", "stdout.log"):
                try:
                    self.states[name], added = _flush_trailing_line(
                        self.paths[name],
                        self.states[name],
                        metadata=metadata,
                    )
                    self.events.extend(added)
                except Exception as error:  # noqa: BLE001 -- retain other complete evidence
                    self.events.append(_monitor_error_event(self.session, name, error))
                self.events = self.events[-_MAX_EVENTS:]
        evidence = self.active / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        for name in ("comsol_batch.log", "stdout.log", "stderr.log"):
            source = self.paths.get(name)
            content = b"" if source is None else _require_bounded_log(source)
            common.serialization.atomic_write_bytes(evidence / name, content)
        exit_code = payload.get("exit_code")
        self.session["normal_runtime_result"] = {
            "exit_code": exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
            "error": None if payload.get("error") is None else str(payload.get("error"))[:4_096],
            "retained_at": _utc_now(),
        }
        self.session["ended_at"] = _utc_now()
        _write_json(self.active / "session.json", self.session)
        _write_events(self.active / "phase_events.jsonl", self.events)
        self.finished = True


def _resolve_probe_owners(
    config_path: Path | str,
) -> tuple[Path, config_service.CampaignConfig, config_service.GenerationConfig, cluster_service.CampaignTask]:
    """Resolve one deterministic ordinary transient technical-smoke task."""
    supplied = Path(config_path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        msg = f"Timing-probe campaign configuration is missing or unsafe: {supplied}."
        raise FileNotFoundError(msg)
    source = supplied.resolve()
    campaign = config_service.load_campaign_config(source)
    tasks = cluster_service.campaign_tasks(campaign)
    if campaign.campaign_purpose != "technical_runtime_smoke" or not tasks:
        msg = "Timing probe requires a non-empty technical-runtime-smoke campaign."
        raise ValueError(msg)
    task = tasks[0]
    config = campaign.batch(task.batch_name)
    if config.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        msg = "Timing probe requires an ordinary transient-drying campaign configuration."
        raise ValueError(msg)
    return source, campaign, config, task


def resolve_timing_probe_plan(config_path: Path | str) -> dict[str, Any]:
    """Resolve the ordinary campaign-owned one-case diagnostic plan."""
    source, campaign, config, task = _resolve_probe_owners(config_path)
    cluster = campaign.execution_values["cluster"]
    site = campaign.execution_values["site"]
    return {
        "schema_kind": PROBE_SCHEMA_KIND,
        "schema_version": PROBE_SCHEMA_VERSION,
        "campaign_config": str(source),
        "campaign_config_sha256": common.serialization.file_sha256(source),
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "batch_name": task.batch_name,
        "batch_id": task.batch_id,
        "batch_identity": config.batch_identity,
        "case_index": task.case_index,
        "case_id": task.case_id,
        "simulation_profile": config.profile.id,
        "template": {
            "relative_path": config.template_relative_path,
            "sha256": config.template_sha256,
        },
        "case_count": 1,
        "resources": {
            "scheduler_kind": cluster["scheduler_kind"],
            "partition": cluster["partition"],
            "cores_per_case": cluster["cores_per_case"],
            "cores_per_node": cluster["cores_per_node"],
            "wall_time": cluster["wall_time"],
            "scheduler_options": list(cluster["scheduler_options"]),
            "cpu_host": site["cpu_host"],
            "python_module": site["python_module"],
            "comsol_module": site["comsol_module"],
            "python_executable": site["python_executable"],
            "comsol_executable": site["comsol_executable"],
        },
    }


def _admit_allocation(campaign: config_service.CampaignConfig) -> tuple[str, int]:
    """Bind actual scheduler context to ordinary configured case resources."""
    cluster = campaign.execution_values["cluster"]
    cores = int(cluster["cores_per_case"])
    if cluster["scheduler_kind"] != "slurm":
        msg = "A timing probe requires an ordinary campaign configured for Slurm."
        raise RuntimeError(msg)
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or not job_id.isdigit() or int(job_id) < 1:
        msg = "A timing probe may execute only inside one numeric Slurm job allocation."
        raise RuntimeError(msg)
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is None or not allocated.isdigit() or int(allocated) != cores:
        msg = "Timing-probe Slurm allocation must exactly match configured execution.cluster.cores_per_case."
        raise RuntimeError(msg)
    return "slurm", cores


def _normal_case_evidence(
    config: config_service.GenerationConfig,
    case_index: int,
    *,
    isolated_storage: Path,
    campaign_run_id: str,
    outcome: batch_service.CaseRunOutcome | None,
) -> dict[str, Any]:
    """Capture bounded authoritative publication or attempt evidence before isolation cleanup."""
    if outcome is not None and outcome.status == "completed":
        destination = outcome.processed_directory.resolve()
        if not destination.is_relative_to(isolated_storage.resolve()):
            msg = "Normal probe case publication escaped isolated storage."
            raise RuntimeError(msg)
        files = {}
        for name in (
            "_SUCCESS",
            "status.json",
            "provenance.json",
            "execution_provenance.json",
            "processing_provenance.json",
            "timing.json",
        ):
            files[name] = _bounded_json_evidence(destination / name)
        return {
            "status": "completed",
            "processed_directory": str(destination.relative_to(isolated_storage)),
            "publication_files": files,
            "failure_attempt": None,
            "license_wait": None,
        }
    attempt = attempt_service.latest_case_attempt_across_campaign_runs(
        config,
        case_index,
        storage_root=isolated_storage,
    )
    attempt_payload = None
    if attempt is not None:
        cleanup_path = attempt.directory / "cleanup.json"
        attempt_payload = {
            "directory": str(attempt.directory.resolve().relative_to(isolated_storage.resolve())),
            "receipt": _bounded_json_evidence(attempt.receipt_path),
            "cleanup": _bounded_json_evidence(cleanup_path) if cleanup_path.is_file() else None,
        }
    license_wait = None
    if outcome is not None and outcome.status == "license_blocked":
        wait_payload = license_service.load_temporary_license_wait(
            config,
            case_index,
            campaign_run_id=campaign_run_id,
            storage_root=isolated_storage,
        )
        if wait_payload is None:
            msg = "Normal runtime reported license_blocked without authoritative license-wait evidence."
            raise RuntimeError(msg)
        wait_directory = license_service.temporary_license_wait_directory(
            campaign_run_id,
            config.batch_id,
            config.case_id(case_index),
            storage_root=isolated_storage,
        )
        wait_path = wait_directory / "license_wait.json"
        license_wait = {
            "directory": str(wait_directory.resolve().relative_to(isolated_storage.resolve())),
            "receipt": _bounded_json_evidence(wait_path),
        }
    if (attempt_payload is None) == (license_wait is None):
        msg = "Failed normal probe case lacks one unique authoritative outcome receipt."
        raise RuntimeError(msg)
    return {
        "status": "failed" if outcome is None else outcome.status,
        "processed_directory": None,
        "publication_files": None,
        "failure_attempt": attempt_payload,
        "license_wait": license_wait,
    }


def _publish_bundle(
    *,
    root: Path,
    active: Path,
    session: Mapping[str, Any],
    exit_code: int,
    events: list[dict[str, Any]],
) -> Path:
    """Publish one exact immutable diagnostic bundle from retained normal-path evidence."""
    probe_id = str(session["probe_id"])
    destination = root / probe_id
    if destination.exists():
        validation = validate_probe_bundle(destination)
        if validation["probe_id"] != probe_id:
            msg = f"Existing timing-probe bundle has another identity: {destination}."
            raise FileExistsError(msg)
        return destination
    staging_root = root / ".staging"
    staging_root.mkdir(mode=0o700, exist_ok=True)
    staging = staging_root / probe_id
    if staging.exists():
        if staging.parent != staging_root or staging.name != probe_id:
            msg = "Refusing to replace timing-probe staging outside its exact owner."
            raise RuntimeError(msg)
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    runtime = active / "evidence"
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
        "real_comsol_grammar_validated": False,
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
    command = list(session["command"])
    _write_json(
        staging / "exact_command.json",
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "command_available": bool(command),
            "argv": command,
            "working_directory": session["work_path"],
            "cores_per_case": session["cores_per_case"],
            "scheduler_kind": session["scheduler_kind"],
            "batch_log_path": session["batch_log_path"],
        },
    )
    _write_json(
        staging / "environment.json",
        {
            "schema_version": PROBE_SCHEMA_VERSION,
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "platform": sys.platform,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "loaded_modules": os.environ.get("LOADEDMODULES"),
            "configured_resources": session["resources"],
            "comsol_version": session["comsol_version"],
        },
    )
    common.serialization.atomic_write_text(staging / "README.md", _probe_readme())
    content_names = _EXACT_BUNDLE_INVENTORY - {"manifest.json", "sha256sums.txt"}
    content_files = {name: _file_evidence(staging / name) for name in sorted(content_names)}
    _write_json(
        staging / "manifest.json",
        {
            "schema_kind": PROBE_SCHEMA_KIND,
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe_id": probe_id,
            "diagnostic_only": True,
            "production_timing_fields_updated": False,
            "source_commit": session["source_commit"],
            "source_config": session["source_config"],
            "source_config_sha256": session["source_config_sha256"],
            "campaign_id": session["campaign_id"],
            "campaign_digest": session["campaign_digest"],
            "campaign_purpose": session["campaign_purpose"],
            "batch_name": session["batch_name"],
            "batch_id": session["batch_id"],
            "batch_identity": session["batch_identity"],
            "case_index": session["case_index"],
            "case_id": session["case_id"],
            "case_input_id": session["case_input_id"],
            "simulation_case_id": session["simulation_case_id"],
            "attempt_id": session["attempt_id"],
            "input_generation_id": session["input_generation_id"],
            "canonical_input_files": session["canonical_input_files"],
            "workspace_input_files": session["workspace_input_files"],
            "template": session["template"],
            "comsol_version": session["comsol_version"],
            "host": socket.gethostname(),
            "resources": session["resources"],
            "scheduler_kind": session["scheduler_kind"],
            "normal_generation_path": {
                "canonical_input_generator": "generation_cases_input.generate_input_cases",
                "case_runner": "generation_runtime_batch.run_case",
                "requested_case_count": 1,
                "worker_slot": 0,
            },
            "normal_case_evidence": session["normal_case_evidence"],
            "exact_command": command,
            "exit_status": {
                "exit_code": exit_code,
                "status": session["probe_case_state"],
                "normal_runtime_result": session["normal_runtime_result"],
                "error": session.get("normal_case_error"),
            },
            "started_at": session["started_at"],
            "ended_at": session["ended_at"],
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


def validate_probe_bundle(  # noqa: C901, PLR0912, PLR0915 -- centralized exact bundle admission
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
        difference = sorted(actual ^ _EXACT_BUNDLE_INVENTORY)
        msg = f"Probe bundle inventory differs from the exact contract: {difference!r}."
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
    exit_status = manifest.get("exit_status")
    case_state = None if not isinstance(exit_status, Mapping) else exit_status.get("status")
    exit_code = None if not isinstance(exit_status, Mapping) else exit_status.get("exit_code")
    if (
        manifest.get("schema_kind") != PROBE_SCHEMA_KIND
        or manifest.get("schema_version") != PROBE_SCHEMA_VERSION
        or manifest.get("diagnostic_only") is not True
        or manifest.get("production_timing_fields_updated") is not False
        or not isinstance(manifest.get("probe_id"), str)
        or manifest.get("probe_id") != bundle.name
        or _SHA256.fullmatch(str(manifest.get("source_config_sha256"))) is None
        or _SHA256.fullmatch(str(manifest.get("case_input_id"))) is None
        or _SHA256.fullmatch(str(manifest.get("simulation_case_id"))) is None
        or _SHA256.fullmatch(str(manifest.get("attempt_id"))) is None
        or case_state not in {"successful", "failed"}
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or (case_state == "successful" and exit_code != 0)
        or (case_state == "failed" and exit_code == 0)
    ):
        msg = "Probe manifest identity or exit evidence is malformed."
        raise ValueError(msg)
    canonical_inputs = manifest.get("canonical_input_files")
    workspace_inputs = manifest.get("workspace_input_files")
    if not isinstance(canonical_inputs, Mapping) or not canonical_inputs:
        msg = "Probe manifest lacks canonical input-file evidence."
        raise ValueError(msg)
    if case_state == "successful":
        if not isinstance(workspace_inputs, Mapping) or set(workspace_inputs) != set(canonical_inputs):
            msg = "Successful probe workspace input membership is incomplete."
            raise ValueError(msg)
        for name, canonical in canonical_inputs.items():
            workspace = workspace_inputs[name]
            if not isinstance(canonical, Mapping) or not isinstance(workspace, Mapping):
                msg = "Probe input identity evidence is malformed."
                raise TypeError(msg)
            if {key: workspace.get(key) for key in ("size_bytes", "sha256")} != dict(canonical):
                msg = f"Probe workspace input identity disagrees for {name!r}."
                raise ValueError(msg)
    normal_evidence = manifest.get("normal_case_evidence")
    if not isinstance(normal_evidence, Mapping):
        msg = "Probe manifest lacks normal case outcome evidence."
        raise TypeError(msg)
    evidence_status = normal_evidence.get("status")
    publication_files = normal_evidence.get("publication_files")
    failure_attempt = normal_evidence.get("failure_attempt")
    license_wait = normal_evidence.get("license_wait")
    if case_state == "successful":
        expected_publication_files = {
            "_SUCCESS",
            "status.json",
            "provenance.json",
            "execution_provenance.json",
            "processing_provenance.json",
            "timing.json",
        }
        if (
            evidence_status != "completed"
            or not isinstance(publication_files, Mapping)
            or set(publication_files) != expected_publication_files
            or failure_attempt is not None
            or license_wait is not None
        ):
            msg = "Successful probe lacks one complete normal case publication receipt set."
            raise ValueError(msg)
        success_payload = _admit_embedded_json_evidence(
            publication_files["_SUCCESS"],
            label="normal case success receipt",
        )
        if success_payload.get("case_id") != manifest.get("case_id"):
            msg = "Normal case success receipt disagrees with the probe case identity."
            raise ValueError(msg)
    else:
        has_failure_attempt = isinstance(failure_attempt, Mapping)
        has_license_wait = isinstance(license_wait, Mapping)
        if evidence_status not in {"failed", "license_blocked"} or publication_files is not None or has_failure_attempt == has_license_wait:
            msg = "Failed probe lacks one unique normal failure or license-wait receipt."
            raise ValueError(msg)
        if isinstance(failure_attempt, Mapping):
            if set(failure_attempt) != {"directory", "receipt", "cleanup"}:
                msg = "Normal failure-attempt evidence has unexpected fields."
                raise ValueError(msg)
            attempt_payload = _admit_embedded_json_evidence(
                failure_attempt["receipt"],
                label="normal failure-attempt receipt",
            )
            if (
                attempt_payload.get("campaign_run_id") != manifest.get("probe_id")
                or attempt_payload.get("case_id") != manifest.get("case_id")
                or attempt_payload.get("case_state") != "failed"
            ):
                msg = "Normal failure-attempt receipt disagrees with the probe outcome."
                raise ValueError(msg)
            cleanup = failure_attempt["cleanup"]
            if cleanup is not None:
                _admit_embedded_json_evidence(cleanup, label="normal failure cleanup receipt")
        else:
            if not isinstance(license_wait, Mapping):
                msg = "Normal license-wait evidence is not an object."
                raise TypeError(msg)
            if set(license_wait) != {"directory", "receipt"}:
                msg = "Normal license-wait evidence has unexpected fields."
                raise ValueError(msg)
            wait_payload = _admit_embedded_json_evidence(
                license_wait["receipt"],
                label="normal license-wait receipt",
            )
            if (
                evidence_status != "license_blocked"
                or wait_payload.get("schema_kind") != "generation_temporary_license_wait"
                or wait_payload.get("campaign_run_id") != manifest.get("probe_id")
                or wait_payload.get("batch_id") != manifest.get("batch_id")
                or wait_payload.get("case_id") != manifest.get("case_id")
            ):
                msg = "Normal license-wait receipt disagrees with the probe outcome identity."
                raise ValueError(msg)
    exact_command = _read_json_object(bundle / "exact_command.json")
    command = exact_command.get("argv")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        msg = "Probe exact command is malformed."
        raise TypeError(msg)
    if command:
        if (
            command.count("-batchlog") != 1
            or command.count("-batchlogout") != 1
            or command[command.index("-batchlog") + 1] != exact_command.get("batch_log_path")
        ):
            msg = "Probe command lacks one exact runtime-owned batch log."
            raise ValueError(msg)
    elif case_state == "successful":
        msg = "Successful probe has no exact COMSOL command."
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
    verdicts = _read_json_object(bundle / "method_verdicts.json")
    if (
        verdicts.get("recommendation") != "unresolved_pending_real_probe_review"
        or verdicts.get("real_comsol_grammar_validated") is not False
        or verdicts.get("production_timing_fields_updated") is not False
    ):
        msg = "Probe method verdicts overstate diagnostic timing evidence."
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


def publish_transferred_probe_bundle(
    probe_id: str,
    *,
    staging_root: Path | str,
    destination_root: Path | str,
) -> dict[str, Any]:
    """Validate and atomically publish one transferred diagnostic-only probe bundle."""
    safe_probe_id = common.paths.validate_logical_name(probe_id, label="probe_id")
    staging = workspace_service.validate_transfer_staging(staging_root, run_id=safe_probe_id)
    marker = workspace_service.TRANSFER_STAGING_MARKER
    source = staging / safe_probe_id
    if {item.name for item in staging.iterdir()} != {marker, safe_probe_id}:
        msg = "Timing-probe transfer staging has unexpected inventory."
        raise ValueError(msg)
    source_validation = validate_probe_bundle(source, require_immutable=False)
    if source_validation["probe_id"] != safe_probe_id:
        msg = "Transferred timing-probe identity disagrees with its staging name."
        raise ValueError(msg)
    destination_storage = workspace_service.resolve_storage_root(destination_root, create=True)
    target = common.paths.get_experiments_root(storage_root=destination_storage) / PROBE_SCOPE / safe_probe_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target_validation = validate_probe_bundle(target)
        for name in _EXACT_BUNDLE_INVENTORY:
            if (target / name).read_bytes() != (source / name).read_bytes():
                msg = f"Existing timing-probe publication conflicts: {target}"
                raise FileExistsError(msg)
        source.chmod(0o700)
        return {
            "probe_id": safe_probe_id,
            "probe_bundle": str(target),
            "reused": True,
            "inventory": target_validation["inventory"],
        }
    source.chmod(0o700)
    source.replace(target)
    for item in target.iterdir():
        item.chmod(0o444)
    target.chmod(0o555)
    target_validation = validate_probe_bundle(target)
    if target_validation["total_size_bytes"] != source_validation["total_size_bytes"]:
        msg = "Timing-probe bundle changed during atomic publication."
        raise RuntimeError(msg)
    return {
        "probe_id": safe_probe_id,
        "probe_bundle": str(target),
        "reused": False,
        "inventory": target_validation["inventory"],
    }


def _sanitized_failure_exit(error: BaseException | None, session: Mapping[str, Any]) -> int:
    """Return one nonzero shell-safe process status without inventing zero success."""
    candidates = [
        None if error is None else getattr(error, "exit_code", None),
        session.get("normal_runtime_result", {}).get("exit_code"),
    ]
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= _MAX_SHELL_EXIT_CODE:
            return value
    return 1


def _cleanup_successful_active(root: Path, active: Path) -> None:
    """Remove only the exact validated probe-owned active directory."""
    active_root = (root / ".active").resolve()
    resolved = active.resolve()
    if resolved.parent != active_root or resolved.name != active.name:
        msg = "Refusing to clean timing-probe evidence outside its exact active owner."
        raise RuntimeError(msg)
    shutil.rmtree(resolved)
    for directory in (active_root, (root / ".staging").resolve()):
        with contextlib.suppress(OSError):
            directory.rmdir()


def run_timing_probe(
    config_path: Path | str,
    *,
    storage_root: Path | str,
    work_root: Path | str | None = None,
    announce: Callable[[Mapping[str, str]], None] | None = None,
) -> dict[str, Any]:
    """Run exactly one normal transient Generation case with diagnostic observation."""
    source, campaign, config, task = _resolve_probe_owners(config_path)
    scheduler_kind, cores_per_case = _admit_allocation(campaign)
    source_commit = source_service.required_git_commit()
    root = _probe_root(storage_root)
    probe_id = f"probe-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    active_root = root / ".active"
    active_root.mkdir(mode=0o700, exist_ok=True)
    active = active_root / probe_id
    active.mkdir(mode=0o700)
    isolated_storage = active / "normal_case_storage"
    isolated_storage.mkdir(mode=0o700)
    started_at = _utc_now()
    initial_attempt_id = common.serialization.canonical_json_sha256(
        {"probe_id": probe_id, "source_commit": source_commit, "state": "normal_case_not_prepared"}
    )
    resources = resolve_timing_probe_plan(source)["resources"]
    session: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe_id": probe_id,
        "source_commit": source_commit,
        "source_config": str(source),
        "source_config_sha256": common.serialization.file_sha256(source),
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "batch_name": task.batch_name,
        "batch_id": task.batch_id,
        "batch_identity": config.batch_identity,
        "case_index": task.case_index,
        "case_id": task.case_id,
        "case_input_id": "0" * 64,
        "simulation_case_id": "0" * 64,
        "attempt_id": initial_attempt_id,
        "input_generation_id": None,
        "canonical_input_files": {},
        "workspace_input_files": {},
        "template": {"relative_path": config.template_relative_path, "sha256": config.template_sha256},
        "resources": resources,
        "cores_per_case": cores_per_case,
        "scheduler_kind": scheduler_kind,
        "comsol_version": _query_comsol_version(comsol_service.resolve_comsol_executable(config)),
        "work_path": None,
        "work_root": None,
        "command": [],
        "batch_log_path": None,
        "normal_runtime_result": {"exit_code": None, "error": None, "retained_at": None},
        "normal_case_evidence": None,
        "probe_case_state": "failed",
        "started_at": started_at,
        "ended_at": None,
    }
    _write_json(active / "session.json", session)
    if announce is not None:
        announce({"probe_id": probe_id})
    generated = input_service.generate_input_cases(
        config,
        1,
        case_start=task.case_index,
        storage_root=isolated_storage,
    )
    if generated.requested_case_indices != (task.case_index,):
        msg = "Timing probe canonical input generation did not select exactly one configured case."
        raise RuntimeError(msg)
    canonical_case = generated.raw_directory / task.case_id
    case_payload = _read_json_object(canonical_case / "case.json")
    canonical_inputs = case_payload.get("input_files")
    if not isinstance(canonical_inputs, Mapping) or not canonical_inputs:
        msg = "Timing-probe canonical case lacks input-file identity evidence."
        raise TypeError(msg)
    session.update(
        {
            "case_input_id": str(case_payload["case_input_id"]),
            "simulation_case_id": str(case_payload["simulation_case_id"]),
            "input_generation_id": generated.input_generation_id,
            "canonical_input_files": {name: dict(identity) for name, identity in sorted(canonical_inputs.items())},
        }
    )
    _write_json(active / "session.json", session)
    observer = _RuntimeProbeObserver(active, session)
    outcome: batch_service.CaseRunOutcome | None = None
    normal_error: Exception | None = None
    previous_campaign_run_id = os.environ.get("GENERATION_CAMPAIGN_RUN_ID")
    os.environ["GENERATION_CAMPAIGN_RUN_ID"] = probe_id
    try:
        try:
            outcome = batch_service.run_case(
                config,
                task.case_index,
                cores_per_case=cores_per_case,
                worker_slot=0,
                scheduler_kind=scheduler_kind,
                allocated_node=os.environ.get("SLURMD_NODENAME"),
                storage_root=isolated_storage,
                work_root=work_root,
                diagnostic_observer=observer,
            )
            observer.finish({"exit_code": 0 if outcome.status == "completed" else None, "error": outcome.message})
        except Exception as error:  # noqa: BLE001 -- normal failure evidence must still become one probe bundle
            normal_error = error
            observer.finish(
                {
                    "exit_code": getattr(error, "exit_code", None),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    finally:
        if previous_campaign_run_id is None:
            os.environ.pop("GENERATION_CAMPAIGN_RUN_ID", None)
        else:
            os.environ["GENERATION_CAMPAIGN_RUN_ID"] = previous_campaign_run_id
    successful = normal_error is None and outcome is not None and outcome.status == "completed"
    exit_code = 0 if successful else _sanitized_failure_exit(normal_error, session)
    session["probe_case_state"] = "successful" if successful else "failed"
    session["normal_case_error"] = None if normal_error is None else {"error_type": type(normal_error).__name__, "message": str(normal_error)[:4_096]}
    session["normal_case_evidence"] = _normal_case_evidence(
        config,
        task.case_index,
        isolated_storage=isolated_storage,
        campaign_run_id=probe_id,
        outcome=outcome,
    )
    session["ended_at"] = session.get("ended_at") or _utc_now()
    _write_json(active / "session.json", session)
    bundle = _publish_bundle(
        root=root,
        active=active,
        session=session,
        exit_code=exit_code,
        events=observer.events,
    )
    validate_probe_bundle(bundle)
    _cleanup_successful_active(root, active)
    return {
        "probe_id": probe_id,
        "probe_case_state": session["probe_case_state"],
        "probe_case_exit_code": exit_code,
        "probe_cpu_bundle": str(bundle),
    }
