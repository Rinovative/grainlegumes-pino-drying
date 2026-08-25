# ruff: noqa: S101, PLR2004
"""Protect concise operational logging for long-running Evaluation artifacts."""

from __future__ import annotations

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


def test_full_artifact_log_is_bounded_and_reports_major_runtime_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report useful phase/progress state without case-internal log spam."""
    reporter = _reporter(total_cases=100)
    reporter.startup()
    for phase in ("preflight", "model_setup"):
        with reporter.phase(phase):
            pass
    reporter.device_summary(
        model_device=torch.device("cpu"),
        scaler_device=torch.device("cpu"),
    )
    with reporter.phase("inference"):
        for index in range(100):
            reporter.case_started(
                case_id=f"case_{index:04d}",
                material="chickpea" if index < 50 else "lentil",
                rollout_steps=161,
            )
            reporter.case_completed(
                case_id=f"case_{index:04d}",
                material="chickpea" if index < 50 else "lentil",
                forward_calls=1370,
                timed_forward_calls=1370,
                model_forward_seconds=0.01,
            )
    reporter.inference_summary()
    for phase in ("metrics", "serialization", "finalization", "validation", "publication"):
        with reporter.phase(phase):
            pass
    snapshot = reporter.done()

    lines = capsys.readouterr().out.splitlines()
    progress = [line for line in lines if line.startswith("[PROGRESS]")]
    assert any("task=transient_drying" in line and "run=uno_example" in line for line in lines)
    assert any("device=cpu" in line and "dtype=float32" in line for line in lines)
    assert any(line == "[PHASE] preflight" for line in lines)
    assert any(line.startswith("[PHASE] preflight done |") for line in lines)
    assert progress[0].startswith("[PROGRESS] 1/100 (1%)")
    assert progress[-1].startswith("[PROGRESS] 100/100 (100%)")
    assert len(progress) <= 20
    assert "ETA" not in progress[0]
    assert any("ETA" in line for line in progress[1:-1])
    assert all("CPU | RSS " in line and "GPU " not in line for line in progress)
    assert not any("rollout step" in line.lower() for line in lines)
    assert any(line.startswith("[INFERENCE] 100 cases | 137000 forwards |") for line in lines)
    assert any(line.startswith("[DONE] artifact validated |") and "total=" in line for line in lines)
    phase_summary = next(line for line in lines if line.startswith("[DONE] phases |"))
    for phase in ("inference", "metrics", "serialization", "finalization", "validation", "publication"):
        assert f"{phase}=" in phase_summary
    status_lines = [line for line in lines if line.startswith(("[ARTIFACT]", "[DEVICE]", "[PHASE]", "[PROGRESS]", "[INFERENCE]", "[DONE]"))]
    assert len(status_lines) <= 42
    assert snapshot["counts"]["case_count"] == 100
    assert snapshot["counts"]["forward_call_count"] == 137000
    assert snapshot["runtime"] == {"device": "cpu", "dtype": "float32"}


def test_one_case_log_uses_case_summary_without_percentage_spam(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use meaningful case boundaries instead of synthetic percentage updates."""
    reporter = _reporter(total_cases=1)
    reporter.startup()
    with reporter.phase("inference"):
        reporter.case_started(
            case_id="case_0051",
            material="chickpea",
            rollout_steps=161,
        )
        reporter.case_completed(
            case_id="case_0051",
            material="chickpea",
            forward_calls=1370,
            timed_forward_calls=1370,
            model_forward_seconds=1.25,
        )
    reporter.inference_summary()

    lines = capsys.readouterr().out.splitlines()
    assert "[CASE] 1/1 chickpea case_0051 | rollout=161 steps" in lines
    assert any(line.startswith("[CASE] done | forwards=1370 |") for line in lines)
    assert not any(line.startswith("[PROGRESS]") for line in lines)
    assert not any("%" in line for line in lines)
    assert not any("step 1/" in line.lower() for line in lines)


def test_cuda_progress_uses_pytorch_memory_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose bounded CUDA identity and allocator memory without a GPU dependency."""
    gibibyte = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Synthetic GPU")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 2 * gibibyte)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 3 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 4 * gibibyte)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 5 * gibibyte)

    reporter = _reporter(total_cases=2, device="cuda:2")
    reporter.startup()
    reporter.device_summary(
        model_device=torch.device("cuda:2"),
        scaler_device=torch.device("cuda:2"),
    )
    reporter.case_completed(
        case_id="case_0001",
        material="lentil",
        forward_calls=10,
        timed_forward_calls=10,
        model_forward_seconds=0.2,
    )
    snapshot = reporter.final_snapshot()

    output = capsys.readouterr().out
    assert "[DEVICE] cuda:2 | Synthetic GPU | model=cuda:2 | scaler=cuda:2" in output
    assert "GPU 2.00/3.00 GiB | peak 4.00 GiB | RSS " in output
    assert snapshot["memory"]["cuda_peak_allocated_bytes"] == 4 * gibibyte
    assert snapshot["memory"]["cuda_peak_reserved_bytes"] == 5 * gibibyte


def test_interleaved_case_phases_are_announced_and_completed_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Accumulate streamed case phases without repeating phase log lines."""
    reporter = _reporter(total_cases=2)
    for _case in range(2):
        for phase in ("inference", "metrics", "serialization"):
            with reporter.work_phase(phase):
                pass
    for phase in ("inference", "metrics", "serialization"):
        reporter.finish_work_phase(phase)

    lines = capsys.readouterr().out.splitlines()
    for phase in ("inference", "metrics", "serialization"):
        assert lines.count(f"[PHASE] {phase}") == 1
        assert sum(line.startswith(f"[PHASE] {phase} done |") for line in lines) == 1


@pytest.mark.parametrize("phase", ["metrics", "finalization"])
def test_failure_log_reports_phase_progress_and_preserves_exception(
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    """Prepend bounded failure context while allowing the exact error to escape."""
    reporter = _reporter(total_cases=4)
    reporter.case_started(
        case_id="case_0007",
        material="lentil",
        rollout_steps=161,
    )
    message = "exact scientific failure"
    with pytest.raises(ValueError, match=message), reporter.phase(phase):
        raise ValueError(message)

    output = capsys.readouterr().out
    assert f"[FAILED] phase={phase} | completed=0/4 |" in output
    assert "[FAILED] last_case=case_0007" in output
    assert "[FAILED] output=/workspace/artifacts/example" in output
