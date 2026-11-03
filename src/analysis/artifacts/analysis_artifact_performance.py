"""Record bounded operational timing for Evaluation artifact workflows.

Responsibilities:
  - Accumulate non-overlapping stage durations, counters and runtime measurements
  - Report one line per finished phase, a heartbeat at most once per minute, device and process-memory evidence
  - Validate operational performance snapshots through one closed schema

Design principles:
  - Operational telemetry remains separate from scientific artifact identity
  - Missing memory observations remain unavailable rather than fabricated
  - A phase prints when it finishes, never when it starts

This module does NOT:
  - Change predictions, metrics, Dataset membership or numerical payloads
  - Generate artifacts or publish scientific completion markers
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import torch

from src.experiments import experiments_console as console

_SCHEMA_KIND: Final = "evaluation_artifact_operational_performance"
_SCHEMA_VERSION: Final = 1
_KIBIBYTE: Final = 1024
_GIBIBYTE: Final = 1024**3
_MINIMUM_ETA_CASES: Final = 2
_HEARTBEAT_SECONDS: Final = 60.0
_PHASES: Final = frozenset(
    {
        "preflight",
        "model_setup",
        "inference",
        "metrics",
        "serialization",
        "finalization",
        "validation",
        "publication",
    }
)


@dataclass(slots=True)
class ArtifactPerformanceRecorder:
    """Accumulate non-overlapping wall-clock stages and bounded runtime facts."""

    device: str
    dtype: str
    _started: float = field(default_factory=perf_counter, init=False, repr=False)
    _stage_seconds: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _measurements: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _cuda_peak_allocated_bytes: int | None = field(default=None, init=False, repr=False)
    _cuda_peak_reserved_bytes: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the runtime identity retained by operational evidence."""
        if not isinstance(self.device, str) or not self.device:
            message = "Artifact performance device must be non-empty text."
            raise TypeError(message)
        if not isinstance(self.dtype, str) or not self.dtype:
            message = "Artifact performance dtype must be non-empty text."
            raise TypeError(message)

    @contextmanager
    def stage(self, name: str, *, announce: bool = False) -> Iterator[None]:
        """Measure one stage and optionally emit bounded progress lines."""
        if not isinstance(name, str) or not name or any(character.isspace() for character in name):
            message = "Artifact performance stage names must be non-empty tokens."
            raise ValueError(message)
        if announce:
            print(f"Artifact stage started: {name}", flush=True)
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed
            if announce:
                print(f"Artifact stage completed: {name} ({elapsed:.3f} s)", flush=True)

    def increment(self, name: str, value: int = 1) -> None:
        """Add one non-negative operational count."""
        if not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            message = "Artifact performance counts require a name and non-negative integer."
            raise TypeError(message)
        self._counts[name] = self._counts.get(name, 0) + value

    @property
    def elapsed_seconds(self) -> float:
        """Return total recorder wall time so far."""
        return perf_counter() - self._started

    @property
    def stage_seconds(self) -> Mapping[str, float]:
        """Return a read-only copy of accumulated major-phase durations."""
        return dict(self._stage_seconds)

    @property
    def counts(self) -> Mapping[str, int]:
        """Return a read-only copy of operational counts."""
        return dict(self._counts)

    def set_measurement(self, name: str, value: float) -> None:
        """Store one finite non-negative operational measurement."""
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0.0
        ):
            message = "Artifact performance measurements require finite non-negative values."
            raise TypeError(message)
        self._measurements[name] = float(value)

    def set_cuda_peaks(self, *, allocated_bytes: int, reserved_bytes: int) -> None:
        """Store process-local CUDA allocator peaks sampled by the runtime owner."""
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (allocated_bytes, reserved_bytes)):
            message = "CUDA peak memory values must be non-negative integers."
            raise TypeError(message)
        self._cuda_peak_allocated_bytes = allocated_bytes
        self._cuda_peak_reserved_bytes = reserved_bytes

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe operational snapshot with stage fractions."""
        total = perf_counter() - self._started
        stages = [
            {
                "name": name,
                "seconds": seconds,
                "total_fraction": seconds / total if total > 0.0 else 0.0,
            }
            for name, seconds in self._stage_seconds.items()
        ]
        return {
            "schema_kind": _SCHEMA_KIND,
            "schema_version": _SCHEMA_VERSION,
            "total_seconds": total,
            "stages": stages,
            "counts": dict(sorted(self._counts.items())),
            "measurements": dict(sorted(self._measurements.items())),
            "runtime": {
                "device": self.device,
                "dtype": self.dtype,
            },
            "memory": {
                "process_peak_rss_bytes": _peak_process_rss_bytes(),
                "cuda_peak_allocated_bytes": self._cuda_peak_allocated_bytes,
                "cuda_peak_reserved_bytes": self._cuda_peak_reserved_bytes,
            },
            "clocks": {
                "stage_wall_clock": "time.perf_counter",
                "cuda_inference_clock": ("torch.cuda.Event" if self.device.startswith("cuda") else "time.perf_counter"),
            },
        }


def _peak_process_rss_bytes() -> int | None:
    """Return process peak RSS using platform-aware Unix resource units."""
    try:
        from resource import RUSAGE_SELF, getrusage  # noqa: PLC0415
    except ImportError:
        return None
    value = int(getrusage(RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * _KIBIBYTE


def current_process_rss_bytes() -> int | None:
    """Return current Linux RSS with a platform-aware Unix peak fallback."""
    statm = Path("/proc/self/statm")
    try:
        fields = statm.read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError):
        return _peak_process_rss_bytes()


@dataclass(slots=True)
class ArtifactProgressReporter:
    """Emit one line per finished phase, a rate-limited heartbeat, and one completion line."""

    task: str
    run: str
    stage_label: str
    checkpoint_label: str
    device: torch.device
    dtype: str
    total_cases: int
    split: str
    output_root: Path
    recorder: ArtifactPerformanceRecorder = field(init=False)
    _completed_cases: int = field(default=0, init=False, repr=False)
    _last_case: str | None = field(default=None, init=False, repr=False)
    _current_phase: str | None = field(default=None, init=False, repr=False)
    _failed_reported: bool = field(default=False, init=False, repr=False)
    _model_forward_seconds: float = field(default=0.0, init=False, repr=False)
    _timed_forward_calls: int = field(default=0, init=False, repr=False)
    _started_work_phases: set[str] = field(default_factory=set, init=False, repr=False)
    _completed_work_phases: set[str] = field(default_factory=set, init=False, repr=False)
    _last_output: float = field(default_factory=perf_counter, init=False, repr=False)
    _device_label: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate bounded startup metadata and create its telemetry recorder."""
        if not isinstance(self.total_cases, int) or isinstance(self.total_cases, bool) or self.total_cases < 1:
            message = "Artifact progress requires a positive case count."
            raise TypeError(message)
        self.output_root = Path(self.output_root)
        self.recorder = ArtifactPerformanceRecorder(
            device=str(self.device),
            dtype=self.dtype,
        )

    def _print(self, text: str, *, to_stderr: bool = False) -> None:
        """Print and flush one line, restarting the heartbeat interval."""
        print(text, file=sys.stderr if to_stderr else sys.stdout, flush=True)
        self._last_output = perf_counter()

    def startup(self) -> None:
        """Print one bounded startup line without exact manifests."""
        self._print(
            f"artifacts {self.run} | {self.task} {self.stage_label} ({self.checkpoint_label}) | {self.device} {self.dtype}"
            f" | {self.total_cases} cases | split {self.split} | out {console.display_path(self.output_root)}"
        )

    def _phase_done(self, name: str, elapsed: float, detail: str = "") -> None:
        """Print the one line of a finished phase."""
        suffix = f"  {self._device_label}" if name == "model_setup" and self._device_label else detail
        self._print(f"  {name:<13} {elapsed:6.1f} s   total {self.recorder.elapsed_seconds:.1f} s{suffix}")

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Print one line when a major phase finishes and preserve exact failures."""
        if name not in _PHASES:
            message = f"Unknown artifact phase {name!r}."
            raise ValueError(message)
        self._current_phase = name
        started = perf_counter()
        try:
            with self.recorder.stage(name):
                yield
        except BaseException:
            self.failure()
            raise
        else:
            self._phase_done(name, perf_counter() - started)
        finally:
            self._current_phase = None

    @contextmanager
    def work_phase(self, name: str) -> Iterator[None]:
        """Accumulate one interleaved case-pipeline phase; it prints once when finished."""
        if name not in _PHASES:
            message = f"Unknown artifact phase {name!r}."
            raise ValueError(message)
        self._started_work_phases.add(name)
        self._current_phase = name
        try:
            with self.recorder.stage(name):
                yield
        except BaseException:
            self.failure()
            raise
        finally:
            self._current_phase = None

    def finish_work_phase(self, name: str) -> None:
        """Report one accumulated interleaved phase exactly once."""
        if name not in self._started_work_phases or name in self._completed_work_phases:
            message = f"Artifact phase {name!r} was not started or was already completed."
            raise RuntimeError(message)
        elapsed = float(self.recorder.stage_seconds.get(name, 0.0))
        detail = f"  {self._completed_cases} cases, {self.recorder.counts.get('forward_call_count', 0)} forwards" if name == "inference" else ""
        self._phase_done(name, elapsed, detail)
        self._completed_work_phases.add(name)

    def device_summary(
        self,
        *,
        model_device: torch.device,
        scaler_device: torch.device,
    ) -> None:
        """Remember the PyTorch-owned device label that the model_setup line shows."""
        gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else "CPU"
        mismatch = f" (model {model_device}, scaler {scaler_device})" if {model_device, scaler_device} != {self.device} else ""
        self._device_label = f"{gpu_name}{mismatch}"

    def case_started(
        self,
        *,
        case_id: str,
        material: str,
        rollout_steps: int,
    ) -> None:
        """Remember the running case for the failure line."""
        self._last_case = f"{case_id} ({material}, {rollout_steps} rollout steps)"

    def case_completed(
        self,
        *,
        case_id: str,
        material: str,
        forward_calls: int,
        timed_forward_calls: int,
        model_forward_seconds: float,
    ) -> None:
        """Record one case and print a heartbeat only when the last line is older than a minute."""
        self._completed_cases += 1
        self._last_case = case_id
        self.recorder.increment("case_count")
        self.recorder.increment("forward_call_count", forward_calls)
        self.recorder.increment(
            "timed_forward_call_count",
            timed_forward_calls,
        )
        self._timed_forward_calls += timed_forward_calls
        self._model_forward_seconds += model_forward_seconds
        if self._completed_cases == self.total_cases or perf_counter() - self._last_output < _HEARTBEAT_SECONDS:
            return
        elapsed = self.recorder.elapsed_seconds
        rate = self._completed_cases / (elapsed / 60.0) if elapsed > 0.0 else 0.0
        percent = round(100.0 * self._completed_cases / self.total_cases)
        parts = [
            f"  progress {self._completed_cases}/{self.total_cases} cases ({percent}%)",
            material,
            f"{rate:.2f} cases/min",
            f"elapsed {console.format_duration(elapsed)}",
        ]
        if self._completed_cases >= _MINIMUM_ETA_CASES and rate > 0.0:
            remaining = self.total_cases - self._completed_cases
            parts.append(f"eta {console.format_duration(60.0 * remaining / rate)}")
        parts.append(self._memory_text())
        self._print(" | ".join(parts))

    def case_reused(
        self,
        *,
        case_id: str,
        material: str,
    ) -> None:
        """Count one strictly admitted staged case without model operations."""
        self.recorder.increment("reused_case_count")
        self.case_completed(
            case_id=case_id,
            material=material,
            forward_calls=0,
            timed_forward_calls=0,
            model_forward_seconds=0.0,
        )

    def inference_summary(self) -> None:
        """Record operational inference timing measurements that the completion line reports."""
        average_ms = 1000.0 * self._model_forward_seconds / self._timed_forward_calls if self._timed_forward_calls > 0 else 0.0
        self.recorder.set_measurement("model_forward_seconds", self._model_forward_seconds)
        self.recorder.set_measurement("average_forward_milliseconds", average_ms)

    def update_cuda_peaks(self) -> None:
        """Sample allocator peaks once without resetting them."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        self.recorder.set_cuda_peaks(
            allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)),
        )

    def final_snapshot(self) -> dict[str, Any]:
        """Finalize throughput and memory evidence without printing."""
        self.update_cuda_peaks()
        total = self.recorder.elapsed_seconds
        rate = self._completed_cases / (total / 60.0) if total > 0.0 else 0.0
        self.recorder.set_measurement("cases_per_minute", rate)
        return self.recorder.snapshot()

    def done(self, snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Print the one-line success summary and return final telemetry."""
        admitted = self.final_snapshot() if snapshot is None else dict(snapshot)
        validated = validate_operational_performance(admitted)
        memory = validated["memory"]
        measurements = validated["measurements"]
        gpu_peak = memory["cuda_peak_allocated_bytes"]
        rss_peak = memory["process_peak_rss_bytes"]
        peaks = [f"RSS peak {'unavailable' if rss_peak is None else f'{rss_peak / _GIBIBYTE:.2f} GiB'}"]
        if gpu_peak is not None:
            peaks.insert(0, f"GPU peak {gpu_peak / _GIBIBYTE:.2f} GiB")
        self._print(
            f"artifacts done | {self._completed_cases} cases | total {console.format_duration(float(validated['total_seconds']))}"
            f" | {float(measurements.get('cases_per_minute', 0.0)):.2f} cases/min"
            f" | forward avg {float(measurements.get('average_forward_milliseconds', 0.0)):.2f} ms | {' | '.join(peaks)}"
            f" | out {console.display_path(self.output_root)}"
        )
        return validated

    def failure(self, *, phase: str | None = None) -> None:
        """Print one phase-aware failure line to stderr without swallowing the exception."""
        if self._failed_reported:
            return
        last_case = "" if self._last_case is None else f" | last case {self._last_case}"
        self._print(
            f"artifacts FAILED in {phase or self._current_phase or 'unknown'} | {self._completed_cases}/{self.total_cases} cases done"
            f" | elapsed {console.format_duration(self.recorder.elapsed_seconds)}{last_case} | out {console.display_path(self.output_root)}",
            to_stderr=True,
        )
        self._failed_reported = True

    def _memory_text(self) -> str:
        """Return one compact CPU/CUDA memory suffix for progress lines."""
        rss_bytes = current_process_rss_bytes()
        rss_text = "unavailable" if rss_bytes is None else f"{rss_bytes / _GIBIBYTE:.2f} GiB"
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return f"CPU | RSS {rss_text}"
        allocated = torch.cuda.memory_allocated(self.device) / _GIBIBYTE
        reserved = torch.cuda.memory_reserved(self.device) / _GIBIBYTE
        peak = torch.cuda.max_memory_allocated(self.device) / _GIBIBYTE
        return f"GPU {allocated:.2f}/{reserved:.2f} GiB | peak {peak:.2f} GiB | RSS {rss_text}"


def validate_operational_performance(value: object) -> dict[str, Any]:
    """Validate one bounded operational snapshot without treating it as identity."""
    if not isinstance(value, Mapping) or set(value) != {
        "schema_kind",
        "schema_version",
        "total_seconds",
        "stages",
        "counts",
        "measurements",
        "runtime",
        "memory",
        "clocks",
    }:
        message = "Artifact operational performance fields do not match the schema."
        raise ValueError(message)
    if value["schema_kind"] != _SCHEMA_KIND or value["schema_version"] != _SCHEMA_VERSION:
        message = "Artifact operational performance schema is unsupported."
        raise ValueError(message)
    total = value["total_seconds"]
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(float(total)) or total < 0.0:
        message = "Artifact operational total_seconds must be finite and non-negative."
        raise ValueError(message)
    stages = value["stages"]
    if not isinstance(stages, list) or len(stages) != len({item.get("name") for item in stages if isinstance(item, Mapping)}):
        message = "Artifact operational stages must be a unique ordered list."
        raise ValueError(message)
    for item in stages:
        if not isinstance(item, Mapping) or set(item) != {"name", "seconds", "total_fraction"}:
            message = "Artifact operational stage fields are invalid."
            raise ValueError(message)
        name = item["name"]
        seconds = item["seconds"]
        fraction = item["total_fraction"]
        if (
            not isinstance(name, str)
            or not name
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or seconds < 0.0
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 <= fraction <= 1.0
        ):
            message = "Artifact operational stage values are invalid."
            raise ValueError(message)
    counts = value["counts"]
    if not isinstance(counts, Mapping) or any(
        not isinstance(name, str) or not name or isinstance(count, bool) or not isinstance(count, int) or count < 0 for name, count in counts.items()
    ):
        message = "Artifact operational counts are invalid."
        raise ValueError(message)
    measurements = value["measurements"]
    if not isinstance(measurements, Mapping) or any(
        not isinstance(name, str)
        or not name
        or isinstance(measurement, bool)
        or not isinstance(measurement, (int, float))
        or not math.isfinite(float(measurement))
        or measurement < 0.0
        for name, measurement in measurements.items()
    ):
        message = "Artifact operational measurements are invalid."
        raise ValueError(message)
    runtime = value["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"device", "dtype"}
        or any(not isinstance(item, str) or not item for item in runtime.values())
    ):
        message = "Artifact operational runtime identity is invalid."
        raise ValueError(message)
    memory = value["memory"]
    if not isinstance(memory, Mapping) or set(memory) != {
        "process_peak_rss_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    }:
        message = "Artifact operational memory fields are invalid."
        raise ValueError(message)
    if any(item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0) for item in memory.values()):
        message = "Artifact operational memory values are invalid."
        raise ValueError(message)
    clocks = value["clocks"]
    if not isinstance(clocks, Mapping) or set(clocks) != {
        "stage_wall_clock",
        "cuda_inference_clock",
    }:
        message = "Artifact operational clock fields are invalid."
        raise ValueError(message)
    return {
        "schema_kind": value["schema_kind"],
        "schema_version": value["schema_version"],
        "total_seconds": float(total),
        "stages": [dict(item) for item in stages],
        "counts": dict(counts),
        "measurements": dict(measurements),
        "runtime": dict(runtime),
        "memory": dict(memory),
        "clocks": dict(clocks),
    }
