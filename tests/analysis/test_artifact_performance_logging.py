"""Protect operational evidence and failure context of Evaluation artifact runs.

Responsibilities:
  - Check the persisted performance snapshot counts, runtime, and CUDA memory evidence
  - Check that failure context is reported while the original exception escapes
  - Check one line per finished phase and the once-per-minute heartbeat

Design principles:
  - Assertions target recorded values, identifiers and the line cadence, not console line layout

This module does NOT:
  - Measure scientific solver duration or publish artifact tensors
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from src.analysis.artifacts import analysis_artifact_performance as performance


def _reporter(*, total_cases: int, device: str = "cpu") -> performance.ArtifactProgressReporter:
    return performance.ArtifactProgressReporter(
        task="transient_drying",
        run="uno_example",
        stage_label="a0",
        checkpoint_label="best",
        device=torch.device(device),
        dtype="float32",
        total_cases=total_cases,
        split="id",
        output_root=Path("/workspace/artifacts/example"),
    )


def test_snapshot_records_case_forward_and_runtime_evidence() -> None:
    """Accumulate every completed case and forward call across phases."""
    reporter = _reporter(total_cases=4)
    reporter.startup()
    with reporter.phase("inference"):
        for index in range(4):
            reporter.case_started(case_id=f"case_{index:04d}", material="lentil", rollout_steps=161)
            reporter.case_completed(
                case_id=f"case_{index:04d}",
                material="lentil",
                forward_calls=1370,
                timed_forward_calls=1370,
                model_forward_seconds=0.01,
            )
    reporter.inference_summary()
    for phase in ("metrics", "publication"):
        with reporter.phase(phase):
            pass

    snapshot = reporter.done()

    assert snapshot["counts"]["case_count"] == 4
    assert snapshot["counts"]["forward_call_count"] == 4 * 1370
    assert snapshot["runtime"] == {"device": "cpu", "dtype": "float32"}


def test_cuda_snapshot_uses_pytorch_memory_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record allocator peaks without requiring a GPU."""
    gibibyte = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Synthetic GPU")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 2 * gibibyte)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 3 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 4 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 5 * gibibyte)
    reporter = _reporter(total_cases=2, device="cuda:2")
    reporter.startup()
    reporter.device_summary(model_device=torch.device("cuda:2"), scaler_device=torch.device("cuda:2"))
    reporter.case_completed(case_id="case_0001", material="lentil", forward_calls=10, timed_forward_calls=10, model_forward_seconds=0.2)

    snapshot = reporter.final_snapshot()

    assert snapshot["memory"]["cuda_peak_allocated_bytes"] == 4 * gibibyte
    assert snapshot["memory"]["cuda_peak_reserved_bytes"] == 5 * gibibyte


@pytest.mark.parametrize("phase", ["metrics", "finalization"])
def test_failure_reports_phase_and_last_case_and_preserves_exception(
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    """Report where the failure happened while the exact error escapes."""
    reporter = _reporter(total_cases=4)
    reporter.case_started(case_id="case_0007", material="lentil", rollout_steps=161)
    message = "exact scientific failure"

    with pytest.raises(ValueError, match=message), reporter.phase(phase):
        raise ValueError(message)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert phase in captured.err
    assert "case_0007" in captured.err
    assert "/workspace/artifacts/example" in captured.err


class _Clock:
    """Controllable replacement for the module's monotonic clock."""

    def __init__(self) -> None:
        self.now = time.perf_counter()

    def __call__(self) -> float:
        return self.now


def test_startup_line_names_the_split_in_words(capsys: pytest.CaptureFixture[str]) -> None:
    """Show the reader the evaluated set, not the internal split role."""
    _reporter(total_cases=2).startup()

    startup = capsys.readouterr().out
    assert "ID test set" in startup
    assert "split id" not in startup


def test_finished_phases_print_one_line_each_and_fast_cases_print_no_progress(capsys: pytest.CaptureFixture[str]) -> None:
    """Print a line only when a phase finishes, with its duration, and no progress line for fast cases."""
    reporter = _reporter(total_cases=2)
    reporter.startup()
    with reporter.phase("preflight"):
        pass
    with reporter.work_phase("inference"):
        reporter.case_started(case_id="case_0001", material="lentil", rollout_steps=161)
        reporter.case_completed(case_id="case_0001", material="lentil", forward_calls=3, timed_forward_calls=3, model_forward_seconds=0.01)
    start_output = capsys.readouterr().out.splitlines()
    assert len(start_output) == 2  # startup line and the finished preflight phase
    assert "preflight" in start_output[1]
    assert " s " in start_output[1]

    reporter.finish_work_phase("inference")
    reporter.case_completed(case_id="case_0002", material="lentil", forward_calls=3, timed_forward_calls=3, model_forward_seconds=0.01)
    reporter.inference_summary()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "inference" in lines[0]
    assert "progress" not in "".join(lines)


def test_running_phase_prints_a_heartbeat_at_most_once_per_minute(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report progress for a long phase no more often than every sixty seconds."""
    clock = _Clock()
    monkeypatch.setattr(performance, "perf_counter", clock)
    reporter = _reporter(total_cases=5)
    reporter.startup()
    capsys.readouterr()

    def complete(index: int, *, advance: float) -> list[str]:
        clock.now += advance
        reporter.case_completed(case_id=f"case_{index}", material="lentil", forward_calls=1, timed_forward_calls=1, model_forward_seconds=0.01)
        return capsys.readouterr().out.splitlines()

    assert complete(1, advance=61.0) != []
    assert complete(2, advance=30.0) == []
    heartbeat = complete(3, advance=31.0)
    assert len(heartbeat) == 1
    assert "3/5" in heartbeat[0]
    assert "elapsed 2 min 02 s" in heartbeat[0]
    assert complete(4, advance=1.0) == []
