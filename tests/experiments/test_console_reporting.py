"""Protect the information content and redaction of the console reporting.

Responsibilities:
  - Check the run start block, the epoch line, the end line, the startup line and the Optuna lines
  - Check that paths are relative to the storage root and credentials never reach diagnostics

Design principles:
  - Assertions protect information content without freezing incidental prose

This module does NOT:
  - Publish W&B data or calculate training metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from support import configs

from src import experiments

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> experiments.console.ConsoleReporter:
    """Build a reporter whose run folder lies under a test-owned storage root."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    return experiments.console.ConsoleReporter(
        config=configs.resolve(configs.direct_config()),
        run_dir=tmp_path / "03_experiments" / "run-a",
    )


def test_run_lines_carry_progress_information_with_storage_relative_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show start block, epoch lines, and the end line without absolute paths."""
    reporter = _reporter(tmp_path, monkeypatch)
    target = reporter.config["training"]["epochs"]
    reporter.startup(resolved_device="cpu")
    start_lines = capsys.readouterr().out.splitlines()
    assert len(start_lines) <= 4
    assert reporter.config["run"]["name"] in start_lines[0]
    assert "cpu" in start_lines[0]
    assert "new run" in start_lines[0]
    assert f"seed {reporter.config['run']['seed']}" in start_lines[0]

    reporter.epoch(
        2,
        {
            "train/loss_total": 0.03215,
            "train/loss_data": 0.03215,
            f"id/{reporter.objective_id}": 0.01234,
            "checkpoint/new_best": 1.0,
            "optimization/learning_rate": 3.88e-4,
            "system/epoch_duration_seconds": 12.8,
            "system/train_duration_seconds": 10.0,
            "system/id_evaluation_duration_seconds": 2.0,
            "system/train_samples_per_second": 45.0,
        },
    )
    reporter.epoch(3, {"train/loss_total": 0.03, "train/loss_data": 0.03, "optimization/learning_rate": 3.88e-4})
    reporter.startup_phase("training_start", {"batches_per_epoch": 1})
    reporter.final({"best_epoch": 2, "completed_epoch": 3, "best_metric": 0.01234}, total_wall_seconds=125.0)
    evaluated, plain, _startup, end = capsys.readouterr().out.splitlines()
    for expected in (f"2/{target}", "loss 3.215e-02", "val 1.234e-02 *best", "lr 3.88e-04", "13 s", "45.0 samples/s"):
        assert expected in evaluated
    assert "val" not in plain
    assert "[" not in evaluated  # a single loss term is not repeated as a component
    for expected in (
        f"completed after 3 of {target} epochs (epoch budget)",
        "best epoch 2",
        reporter.objective_id,
        "total 2 min 05 s (startup 0 s, training 10 s, validation 2 s, other 1 min 53 s)",
        "03_experiments/run-a",
    ):
        assert expected in end
    assert str(tmp_path) not in "".join(start_lines) + evaluated + plain + end


def test_start_block_names_the_loader_workers_that_actually_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worker count capped by the allocation's CPUs shows the effective count, not the configured one."""
    reporter = _reporter(tmp_path, monkeypatch)
    reporter.config["data"]["num_workers"] = 8
    monkeypatch.setattr(experiments.config.loader.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    reporter.startup(resolved_device="cpu")
    data_line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("data "))
    assert data_line.endswith("| workers 4")


def test_end_line_names_a_measured_train_seconds_budget_as_the_stop_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matched stage that stops on its measured budget says so, with the epochs it ran of those it could run."""
    reporter = _reporter(tmp_path, monkeypatch)
    target = reporter.config["training"]["epochs"]

    reporter.final({"best_epoch": 3, "completed_epoch": 3, "best_metric": 0.5, "target_train_seconds": 10.0}, total_wall_seconds=30.0)

    assert f"completed after 3 of {target} epochs (train_seconds budget 10 s)" in capsys.readouterr().out


def test_loss_components_share_the_epoch_line_only_when_several_remain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append compact components when more than the total loss exists."""
    reporter = _reporter(tmp_path, monkeypatch)
    reporter.epoch(1, {"train/loss_total": 0.03, "train/loss_data": 0.03, "train/loss_data_T": 0.0168, "train/loss_data_phi": 0.015})

    line = capsys.readouterr().out
    assert len(line.splitlines()) == 1
    assert "T 1.68e-02" in line
    assert "phi 1.50e-02" in line


def test_startup_phases_print_one_line_and_failures_name_the_last_phase(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect startup phases into one line and show details only for a failed startup."""
    reporter = _reporter(tmp_path, monkeypatch)
    reporter.startup_phase("dataset_resolved", {"dataset_id": "train", "cases": 2, "items": 152, "grid": "126x201", "additional_sources": 0})
    reporter.startup_phase("scaler_cache", {"state": "hit"})
    assert capsys.readouterr().out == ""
    reporter.failure(RuntimeError("boom"), status="failed")
    stopped = capsys.readouterr().err
    assert "scaler_cache" in stopped
    assert "boom" in stopped

    reporter.startup_phase("training_start", {"batches_per_epoch": 76})
    line = capsys.readouterr().out
    assert len(line.splitlines()) == 1
    for expected in ("startup", "2 Train cases", "152 items", "126x201", "scaler hit", "76 batches/epoch"):
        assert expected in line


def test_console_failure_does_not_disclose_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redact credentials from both startup and failure diagnostics; the caller owns any traceback."""
    secret = "synthetic-credential-value"  # noqa: S105 - artificial redaction sentinel
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = configs.resolve(configs.direct_config())
    reporter = experiments.console.ConsoleReporter(config=config, run_dir=tmp_path)

    reporter.startup(resolved_device="cpu")
    message = f"api_key={secret}"
    reporter.failure(RuntimeError(message), status="failed")

    captured = capsys.readouterr()
    visible = captured.out + captured.err
    assert secret not in visible
    assert "<redacted>" in captured.err
    assert "FAILED" in captured.err
    assert "Traceback" not in captured.err


def test_optuna_lines_report_each_finished_trial_and_the_study_end(capsys: pytest.CaptureFixture[str]) -> None:
    """Print number, parameters, objective, state and duration per trial; failures go to stderr."""
    sampled = {"batch": 16, "lr": 0.0015389379}
    experiments.console.optuna_trial_event("completed", trial=1, sampled=sampled, duration_seconds=543.0, objective=1.379134)
    experiments.console.optuna_trial_event("pruned", trial=2, sampled=sampled, duration_seconds=60.0, objective=2.0, step=5)
    captured = capsys.readouterr()
    completed, pruned = captured.out.splitlines()
    for expected in ("trial   1", "completed", "1.379e+00", "9 min 03 s", "batch=16", "lr=0.001539"):
        assert expected in completed
    assert "pruned@5" in pruned
    assert captured.err == ""

    experiments.console.optuna_trial_event("failed", trial=3, sampled=sampled, duration_seconds=1.0)
    assert "trial   3  failed" in capsys.readouterr().err

    experiments.console.optuna_study_event(
        "completed",
        study="fno_search",
        attempted_trials=3,
        completed_trials=1,
        pruned_trials=1,
        failed_trials=1,
        best_trial=1,
        best_objective=1.379134,
        best_parameters=sampled,
        duration_seconds=590.0,
        storage="sqlite://<configured>",
    )
    study = capsys.readouterr().out
    for expected in ("fno_search", "3 trials", "1 pruned", "best trial 1", "batch=16"):
        assert expected in study
