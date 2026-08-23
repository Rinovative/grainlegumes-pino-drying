# ruff: noqa: EM101, EM102, TRY003
"""
generation_runtime_comsol_timing.py

Parse and validate COMSOL-reported scientific solver timing evidence.

Responsibilities:
  - Parse structural COMSOL batch-log block boundaries and Solution time records
  - Select only the stationary-airflow and transient-drying top-level solver blocks
  - Represent complete, missing, ambiguous, and not-applicable phase timing explicitly
  - Validate additive persisted solver-timing evidence for downstream consumers

Design principles:
  - Block ownership comes only from matched COMSOL open and close boundaries
  - Missing or changed log grammar produces unavailable metadata, not a solver failure
  - Process wall timing remains independent from COMSOL-reported scientific timing

This module does NOT:
  - Infer solver phases from arbitrary Stationary or Transient text
  - Parse stdout, scheduler timing, polling intervals, or export duration
  - Mutate completed cases or historical storage
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

COMSOL_BATCH_LOG_FILENAME: Final = "comsol_batch.log"
SOLVER_TIMING_SCHEMA_KIND: Final = "comsol_solver_timing"
SOLVER_TIMING_SCHEMA_VERSION: Final = 1
SOLVER_TIMING_METHOD: Final = "comsol_batch_log_top_level_solution_time"
SOLVER_TIMING_SOURCE_KIND: Final = "comsol_batch_log"

_PHASE_STATIONARY: Final = "stationary_airflow"
_PHASE_TRANSIENT: Final = "transient_drying"
_PHASES: Final = (_PHASE_STATIONARY, _PHASE_TRANSIENT)
_MAX_DIAGNOSTICS: Final = 8
_MAX_DIAGNOSTIC_CHARS: Final = 240
_MAX_RETAINED_CANDIDATES: Final = 8

_BLOCK_PATTERNS: Final = {
    _PHASE_STATIONARY: re.compile(r"^Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution \(sol[0-9]+\)$"),
    _PHASE_TRANSIENT: re.compile(r"^Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution \(sol[0-9]+\)$"),
}
_OPEN_BOUNDARY_PREFIX: Final = "<---- "
_CLOSE_BOUNDARY_PATTERN: Final = re.compile(r"^----- (?=\S)")
_OPEN_TERMINATOR_PATTERN: Final = re.compile(r"\s-+\s*$")
_CLOSE_TERMINATOR_PATTERN: Final = re.compile(r"\s(?:-+)?>(?:\s*)$")
_SOLUTION_PREFIX_PATTERN: Final = re.compile(r"^Solution time:\s*(?P<value>.*)$")
_SOLUTION_VALUE_PATTERN: Final = re.compile(r"^(?P<seconds>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s+s\.(?:\s+\([^\n]*\))?\s*$")

PhaseStatus = Literal["complete", "missing", "ambiguous", "not_applicable"]
TimingStatus = Literal["complete", "missing", "ambiguous"]


@dataclass(frozen=True, slots=True)
class SolutionTimeRecord:
    """Identify one COMSOL Solution time and its directly owning block."""

    seconds: float
    block: str
    line_number: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return {
            "seconds": self.seconds,
            "block": self.block,
            "line_number": self.line_number,
        }


@dataclass(frozen=True, slots=True)
class SolverPhaseTiming:
    """Describe the unambiguous or unavailable timing for one scientific phase."""

    status: PhaseStatus
    occurrence_count: int
    candidates: tuple[SolutionTimeRecord, ...]

    @property
    def seconds(self) -> float | None:
        """Return the sole confirmed phase duration when available."""
        if self.status != "complete" or len(self.candidates) != 1:
            return None
        return self.candidates[0].seconds

    def as_dict(self) -> dict[str, Any]:
        """Return compact phase evidence with bounded candidate provenance."""
        record = self.candidates[0] if self.status == "complete" and len(self.candidates) == 1 else None
        return {
            "status": self.status,
            "occurrence_count": self.occurrence_count,
            "seconds": None if record is None else record.seconds,
            "block": None if record is None else record.block,
            "line_number": None if record is None else record.line_number,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ComsolSolverTiming:
    """Represent structural solver timing parsed from one authoritative batch log."""

    simulation_profile: str
    status: TimingStatus
    stationary_airflow: SolverPhaseTiming
    transient_drying: SolverPhaseTiming
    scientific_solver_seconds: float | None
    solution_time_record_count: int
    ignored_non_scientific_timing_count: int
    diagnostics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the compact persisted parser evidence."""
        return {
            "schema_kind": SOLVER_TIMING_SCHEMA_KIND,
            "schema_version": SOLVER_TIMING_SCHEMA_VERSION,
            "method": SOLVER_TIMING_METHOD,
            "source_kind": SOLVER_TIMING_SOURCE_KIND,
            "source_path": COMSOL_BATCH_LOG_FILENAME,
            "simulation_profile": self.simulation_profile,
            "status": self.status,
            "required_phases": list(_required_phases(self.simulation_profile)),
            "phases": {
                _PHASE_STATIONARY: self.stationary_airflow.as_dict(),
                _PHASE_TRANSIENT: self.transient_drying.as_dict(),
            },
            "scientific_solver_seconds": self.scientific_solver_seconds,
            "solution_time_record_count": self.solution_time_record_count,
            "ignored_non_scientific_timing_count": self.ignored_non_scientific_timing_count,
            "diagnostics": list(self.diagnostics),
        }

    def timing_fields(self) -> dict[str, Any]:
        """Return additive fields for Generation's authoritative timing sidecar."""
        return {
            "comsol_stationary_airflow_seconds": self.stationary_airflow.seconds,
            "comsol_transient_drying_seconds": self.transient_drying.seconds,
            "comsol_scientific_solver_seconds": self.scientific_solver_seconds,
            "comsol_solver_timing": self.as_dict(),
        }


@dataclass(slots=True)
class _ActiveBlock:
    """Track one structurally opened COMSOL block until its matching close."""

    title: str
    opening_line_number: int
    solution_times: list[SolutionTimeRecord]


@dataclass(frozen=True, slots=True)
class _Boundary:
    """Represent one complete normalized COMSOL block boundary."""

    kind: Literal["open", "close"]
    title: str
    line_number: int
    final_index: int


def _required_phases(simulation_profile: str) -> tuple[str, ...]:
    """Return the exact scientific phases required by one supported profile."""
    if simulation_profile == profiles.STEADY_FLOW_PROFILE:
        return (_PHASE_STATIONARY,)
    if simulation_profile == profiles.TRANSIENT_DRYING_PROFILE:
        return _PHASES
    message = f"Unsupported COMSOL solver-timing profile: {simulation_profile!r}."
    raise ValueError(message)


def _append_diagnostic(diagnostics: list[str], message: str) -> None:
    """Append one bounded diagnostic without growing persisted evidence unboundedly."""
    if len(diagnostics) >= _MAX_DIAGNOSTICS:
        return
    normalized = " ".join(message.split())
    diagnostics.append(normalized[:_MAX_DIAGNOSTIC_CHARS])


def _phase_for_block(title: str) -> str | None:
    """Return the exact target phase owned by a normalized block title."""
    for phase, pattern in _BLOCK_PATTERNS.items():
        if pattern.fullmatch(title) is not None:
            return phase
    return None


def _boundary_start(line: str) -> tuple[Literal["open", "close"], str] | None:
    """Recognize only exact COMSOL boundary prefixes, never progress dashes."""
    if line.startswith(_OPEN_BOUNDARY_PREFIX):
        return "open", line[len(_OPEN_BOUNDARY_PREFIX) :]
    close = _CLOSE_BOUNDARY_PATTERN.match(line)
    if close is not None:
        return "close", line[close.end() :]
    return None


def _boundary_is_complete(kind: Literal["open", "close"], segment: str) -> bool:
    """Return whether one boundary segment contains its terminal decoration."""
    pattern = _OPEN_TERMINATOR_PATTERN if kind == "open" else _CLOSE_TERMINATOR_PATTERN
    return pattern.search(segment) is not None


def _normalize_boundary_title(kind: Literal["open", "close"], segments: Sequence[str]) -> str:
    """Remove COMSOL boundary decoration and normalize wrapped title whitespace."""
    parts = [segment.strip() for segment in segments]
    pattern = _OPEN_TERMINATOR_PATTERN if kind == "open" else _CLOSE_TERMINATOR_PATTERN
    parts[-1] = pattern.sub("", parts[-1]).strip()
    return " ".join(part for part in parts if part)


def _read_boundary(
    lines: Sequence[str],
    index: int,
    diagnostics: list[str],
) -> _Boundary | None:
    """Read one complete possibly wrapped COMSOL block boundary."""
    started = _boundary_start(lines[index])
    if started is None:
        return None
    kind, first_segment = started
    segments = [first_segment]
    final_index = index
    while not _boundary_is_complete(kind, segments[-1]):
        candidate_index = final_index + 1
        if candidate_index >= len(lines) or not lines[candidate_index][:1].isspace():
            _append_diagnostic(
                diagnostics,
                f"Incomplete COMSOL {kind} boundary at line {index + 1}.",
            )
            return None
        final_index = candidate_index
        segments.append(lines[final_index])
    title = _normalize_boundary_title(kind, segments)
    if not title:
        _append_diagnostic(diagnostics, f"Empty COMSOL {kind} boundary at line {index + 1}.")
        return None
    return _Boundary(
        kind=kind,
        title=title,
        line_number=index + 1,
        final_index=final_index,
    )


def _solution_seconds(line: str, *, line_number: int, diagnostics: list[str]) -> float | None:
    """Parse one exact COMSOL Solution time value expressed in seconds."""
    prefix = _SOLUTION_PREFIX_PATTERN.fullmatch(line)
    if prefix is None:
        return None
    value = _SOLUTION_VALUE_PATTERN.fullmatch(prefix.group("value"))
    if value is None:
        _append_diagnostic(diagnostics, f"Malformed Solution time at line {line_number}.")
        return None
    seconds = float(value.group("seconds"))
    if not math.isfinite(seconds) or seconds < 0.0:
        _append_diagnostic(diagnostics, f"Invalid Solution time at line {line_number}.")
        return None
    return seconds


def _phase_timing(records: Sequence[SolutionTimeRecord], *, applicable: bool) -> SolverPhaseTiming:
    """Classify one phase without selecting arbitrarily among duplicate records."""
    if not applicable:
        return SolverPhaseTiming(status="not_applicable", occurrence_count=0, candidates=())
    occurrence_count = len(records)
    retained = tuple(records[:_MAX_RETAINED_CANDIDATES])
    if occurrence_count == 1:
        status: PhaseStatus = "complete"
    elif occurrence_count == 0:
        status = "missing"
    else:
        status = "ambiguous"
    return SolverPhaseTiming(
        status=status,
        occurrence_count=occurrence_count,
        candidates=retained,
    )


def _result(
    *,
    simulation_profile: str,
    phase_records: Mapping[str, Sequence[SolutionTimeRecord]],
    solution_time_record_count: int,
    ignored_non_scientific_timing_count: int,
    diagnostics: Sequence[str],
) -> ComsolSolverTiming:
    """Build one profile-aware result and scientific sum."""
    required = _required_phases(simulation_profile)
    stationary = _phase_timing(
        phase_records[_PHASE_STATIONARY],
        applicable=_PHASE_STATIONARY in required,
    )
    transient = _phase_timing(
        phase_records[_PHASE_TRANSIENT],
        applicable=_PHASE_TRANSIENT in required,
    )
    required_timings = {
        _PHASE_STATIONARY: stationary,
        _PHASE_TRANSIENT: transient,
    }
    selected = tuple(required_timings[phase] for phase in required)
    resolved_diagnostics = list(diagnostics)
    for phase_name, phase_timing in zip(required, selected, strict=True):
        if phase_timing.status == "missing":
            _append_diagnostic(resolved_diagnostics, f"Required {phase_name} top-level Solution time is missing.")
        elif phase_timing.status == "ambiguous":
            _append_diagnostic(
                resolved_diagnostics,
                f"Required {phase_name} top-level Solution time is ambiguous: {phase_timing.occurrence_count} records.",
            )
    if any(phase.status == "ambiguous" for phase in selected):
        status: TimingStatus = "ambiguous"
    elif all(phase.status == "complete" for phase in selected):
        status = "complete"
    else:
        status = "missing"
    scientific = None
    if status == "complete":
        confirmed = tuple(phase.seconds for phase in selected)
        if any(value is None for value in confirmed):
            message = "Complete COMSOL solver timing lacks a required phase value."
            raise RuntimeError(message)
        scientific = sum(float(value) for value in confirmed if value is not None)
    return ComsolSolverTiming(
        simulation_profile=simulation_profile,
        status=status,
        stationary_airflow=stationary,
        transient_drying=transient,
        scientific_solver_seconds=scientific,
        solution_time_record_count=solution_time_record_count,
        ignored_non_scientific_timing_count=ignored_non_scientific_timing_count,
        diagnostics=tuple(resolved_diagnostics),
    )


def parse_comsol_batch_log_text(
    text: str,
    *,
    simulation_profile: str,
) -> ComsolSolverTiming:
    """
    Parse complete COMSOL batch-log text by matched structural block ownership.

    Parameters
    ----------
    text : str
        Complete text from one runtime-owned COMSOL batch log.
    simulation_profile : str
        Supported Generation profile that determines required scientific phases.

    Returns
    -------
    ComsolSolverTiming
        Immutable complete, missing, or ambiguous timing evidence.

    """
    _required_phases(simulation_profile)
    lines = text.splitlines()
    diagnostics: list[str] = []
    stack: list[_ActiveBlock] = []
    phase_records: dict[str, list[SolutionTimeRecord]] = {phase: [] for phase in _PHASES}
    solution_time_record_count = 0
    ignored_non_scientific_timing_count = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        boundary = _read_boundary(lines, index, diagnostics)
        if boundary is not None:
            if boundary.kind == "open":
                stack.append(
                    _ActiveBlock(
                        title=boundary.title,
                        opening_line_number=boundary.line_number,
                        solution_times=[],
                    )
                )
            elif not stack:
                _append_diagnostic(
                    diagnostics,
                    f"Unmatched COMSOL close boundary at line {boundary.line_number}: {boundary.title}",
                )
            elif stack[-1].title != boundary.title:
                _append_diagnostic(
                    diagnostics,
                    f"Mismatched COMSOL close boundary at line {boundary.line_number}: {boundary.title}",
                )
                stack.clear()
            else:
                block = stack.pop()
                phase = _phase_for_block(block.title)
                if phase is not None:
                    phase_records[phase].extend(block.solution_times)
            index = boundary.final_index + 1
            continue
        if _SOLUTION_PREFIX_PATTERN.fullmatch(line) is not None:
            solution_time_record_count += 1
            owner = stack[-1] if stack else None
            phase = None if owner is None else _phase_for_block(owner.title)
            if phase is None:
                ignored_non_scientific_timing_count += 1
            seconds = _solution_seconds(
                line,
                line_number=index + 1,
                diagnostics=diagnostics,
            )
            if seconds is not None and owner is not None:
                owner.solution_times.append(
                    SolutionTimeRecord(
                        seconds=seconds,
                        block=owner.title,
                        line_number=index + 1,
                    )
                )
        index += 1
    for block in stack:
        _append_diagnostic(
            diagnostics,
            f"Unclosed COMSOL block from line {block.opening_line_number}: {block.title}",
        )
    return _result(
        simulation_profile=simulation_profile,
        phase_records=phase_records,
        solution_time_record_count=solution_time_record_count,
        ignored_non_scientific_timing_count=ignored_non_scientific_timing_count,
        diagnostics=diagnostics,
    )


def parse_comsol_batch_log(
    path: Path | str,
    *,
    simulation_profile: str,
) -> ComsolSolverTiming:
    """
    Parse one finalized runtime-owned COMSOL batch log without changing case status.

    Parameters
    ----------
    path : Path | str
        Attempt-local authoritative batch-log path.
    simulation_profile : str
        Supported Generation profile that determines required scientific phases.

    Returns
    -------
    ComsolSolverTiming
        Timing evidence; unreadable or absent logs produce missing evidence.

    """
    _required_phases(simulation_profile)
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        diagnostics: list[str] = []
        _append_diagnostic(
            diagnostics,
            f"COMSOL batch log is unavailable: {type(error).__name__}: {error}",
        )
        return _result(
            simulation_profile=simulation_profile,
            phase_records=dict.fromkeys(_PHASES, ()),
            solution_time_record_count=0,
            ignored_non_scientific_timing_count=0,
            diagnostics=diagnostics,
        )
    result = parse_comsol_batch_log_text(text, simulation_profile=simulation_profile)
    if "\ufffd" not in text:
        return result
    diagnostics = list(result.diagnostics)
    _append_diagnostic(diagnostics, "COMSOL batch log contained replacement-decoded bytes.")
    return ComsolSolverTiming(
        simulation_profile=result.simulation_profile,
        status=result.status,
        stationary_airflow=result.stationary_airflow,
        transient_drying=result.transient_drying,
        scientific_solver_seconds=result.scientific_solver_seconds,
        solution_time_record_count=result.solution_time_record_count,
        ignored_non_scientific_timing_count=result.ignored_non_scientific_timing_count,
        diagnostics=tuple(diagnostics),
    )


def _finite_non_negative(value: Any, *, label: str) -> float:
    """Require one finite non-negative numeric persisted value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _candidate_from_payload(value: Any, *, phase: str) -> SolutionTimeRecord:
    """Validate one persisted phase candidate."""
    if not isinstance(value, dict) or set(value) != {"seconds", "block", "line_number"}:
        raise ValueError(f"Persisted {phase} timing candidate has invalid fields.")
    block = value["block"]
    line_number = value["line_number"]
    if not isinstance(block, str) or _BLOCK_PATTERNS[phase].fullmatch(block) is None:
        raise ValueError(f"Persisted {phase} timing candidate has an invalid solver block.")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1:
        raise ValueError(f"Persisted {phase} timing candidate has an invalid line number.")
    return SolutionTimeRecord(
        seconds=_finite_non_negative(value["seconds"], label=f"persisted {phase} seconds"),
        block=block,
        line_number=line_number,
    )


def _phase_from_payload(value: Any, *, phase: str, applicable: bool) -> SolverPhaseTiming:
    """Validate one persisted phase status and bounded candidate set."""
    expected = {"status", "occurrence_count", "seconds", "block", "line_number", "candidates"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Persisted {phase} timing evidence has invalid fields.")
    status = value["status"]
    if status not in {"complete", "missing", "ambiguous", "not_applicable"}:
        raise ValueError(f"Persisted {phase} timing has an invalid status.")
    occurrence_count = value["occurrence_count"]
    if isinstance(occurrence_count, bool) or not isinstance(occurrence_count, int) or occurrence_count < 0:
        raise ValueError(f"Persisted {phase} timing has an invalid occurrence count.")
    raw_candidates = value["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) > _MAX_RETAINED_CANDIDATES:
        raise ValueError(f"Persisted {phase} timing candidates are invalid or unbounded.")
    candidates = tuple(_candidate_from_payload(item, phase=phase) for item in raw_candidates)
    expected_status: PhaseStatus
    if not applicable:
        expected_status = "not_applicable"
        if occurrence_count != 0 or candidates:
            raise ValueError(f"Persisted {phase} timing is not applicable but contains candidates.")
    elif occurrence_count == 0:
        expected_status = "missing"
    elif occurrence_count == 1:
        expected_status = "complete"
    else:
        expected_status = "ambiguous"
    if status != expected_status or len(candidates) != min(occurrence_count, _MAX_RETAINED_CANDIDATES):
        raise ValueError(f"Persisted {phase} timing status disagrees with its candidates.")
    timing = SolverPhaseTiming(
        status=expected_status,
        occurrence_count=occurrence_count,
        candidates=candidates,
    )
    record = candidates[0] if expected_status == "complete" else None
    expected_direct = (
        None if record is None else record.seconds,
        None if record is None else record.block,
        None if record is None else record.line_number,
    )
    if (value["seconds"], value["block"], value["line_number"]) != expected_direct:
        raise ValueError(f"Persisted {phase} direct timing fields disagree with candidate evidence.")
    return timing


def admit_persisted_solver_timing(
    timing: Mapping[str, Any],
    *,
    simulation_profile: str,
) -> ComsolSolverTiming | None:
    """
    Validate optional additive solver timing in one admitted Generation sidecar.

    Legacy schema-v1 timing sidecars without any solver timing fields remain valid
    and return ``None``. Any partially present or inconsistent new evidence fails
    closed.
    """
    required = _required_phases(simulation_profile)
    top_fields = {
        "comsol_stationary_airflow_seconds",
        "comsol_transient_drying_seconds",
        "comsol_scientific_solver_seconds",
    }
    evidence_present = "comsol_solver_timing" in timing
    present_top_fields = top_fields.intersection(timing)
    if not evidence_present and not present_top_fields:
        return None
    if not evidence_present or present_top_fields != top_fields:
        raise ValueError("Case timing contains incomplete COMSOL solver-timing fields.")
    payload = timing["comsol_solver_timing"]
    expected_keys = {
        "schema_kind",
        "schema_version",
        "method",
        "source_kind",
        "source_path",
        "simulation_profile",
        "status",
        "required_phases",
        "phases",
        "scientific_solver_seconds",
        "solution_time_record_count",
        "ignored_non_scientific_timing_count",
        "diagnostics",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Persisted COMSOL solver-timing evidence has invalid fields.")
    if (
        payload["schema_kind"] != SOLVER_TIMING_SCHEMA_KIND
        or payload["schema_version"] != SOLVER_TIMING_SCHEMA_VERSION
        or payload["method"] != SOLVER_TIMING_METHOD
        or payload["source_kind"] != SOLVER_TIMING_SOURCE_KIND
        or payload["source_path"] != COMSOL_BATCH_LOG_FILENAME
        or payload["simulation_profile"] != simulation_profile
        or payload["required_phases"] != list(required)
    ):
        raise ValueError("Persisted COMSOL solver-timing identity or method is invalid.")
    phases_payload = payload["phases"]
    if not isinstance(phases_payload, dict) or set(phases_payload) != set(_PHASES):
        raise ValueError("Persisted COMSOL solver-timing phases are invalid.")
    stationary = _phase_from_payload(
        phases_payload[_PHASE_STATIONARY],
        phase=_PHASE_STATIONARY,
        applicable=_PHASE_STATIONARY in required,
    )
    transient = _phase_from_payload(
        phases_payload[_PHASE_TRANSIENT],
        phase=_PHASE_TRANSIENT,
        applicable=_PHASE_TRANSIENT in required,
    )
    phase_map = {_PHASE_STATIONARY: stationary, _PHASE_TRANSIENT: transient}
    selected = tuple(phase_map[phase] for phase in required)
    expected_status: TimingStatus
    if any(phase.status == "ambiguous" for phase in selected):
        expected_status = "ambiguous"
    elif all(phase.status == "complete" for phase in selected):
        expected_status = "complete"
    else:
        expected_status = "missing"
    if payload["status"] != expected_status:
        raise ValueError("Persisted COMSOL solver-timing status is inconsistent.")
    scientific = payload["scientific_solver_seconds"]
    if expected_status == "complete":
        expected_scientific = sum(float(phase.seconds) for phase in selected if phase.seconds is not None)
        scientific_value = _finite_non_negative(
            scientific,
            label="persisted scientific solver seconds",
        )
        if not math.isclose(scientific_value, expected_scientific, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Persisted scientific solver timing disagrees with required phase timings.")
        scientific = scientific_value
    elif scientific is not None:
        raise ValueError("Unavailable persisted COMSOL solver timing cannot contain a scientific sum.")
    record_count = payload["solution_time_record_count"]
    ignored_count = payload["ignored_non_scientific_timing_count"]
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
        or isinstance(ignored_count, bool)
        or not isinstance(ignored_count, int)
        or not 0 <= ignored_count <= record_count
    ):
        raise ValueError("Persisted COMSOL Solution time record counts are invalid.")
    accounted_record_count = stationary.occurrence_count + transient.occurrence_count + ignored_count
    if accounted_record_count > record_count:
        raise ValueError("Persisted COMSOL phase and ignored timing counts exceed the total record count.")
    diagnostics_value = payload["diagnostics"]
    if (
        not isinstance(diagnostics_value, list)
        or len(diagnostics_value) > _MAX_DIAGNOSTICS
        or any(not isinstance(item, str) or len(item) > _MAX_DIAGNOSTIC_CHARS for item in diagnostics_value)
    ):
        raise ValueError("Persisted COMSOL solver-timing diagnostics are invalid or unbounded.")
    result = ComsolSolverTiming(
        simulation_profile=simulation_profile,
        status=expected_status,
        stationary_airflow=stationary,
        transient_drying=transient,
        scientific_solver_seconds=scientific,
        solution_time_record_count=record_count,
        ignored_non_scientific_timing_count=ignored_count,
        diagnostics=tuple(diagnostics_value),
    )
    expected_top = {
        "comsol_stationary_airflow_seconds": result.stationary_airflow.seconds,
        "comsol_transient_drying_seconds": result.transient_drying.seconds,
        "comsol_scientific_solver_seconds": result.scientific_solver_seconds,
    }
    if any(timing[field] != expected for field, expected in expected_top.items()):
        raise ValueError("Case timing COMSOL solver fields disagree with structural evidence.")
    return result
