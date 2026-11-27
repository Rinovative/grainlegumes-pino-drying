"""Protect the information content and redaction of the console reporting.

Responsibilities:
  - Check the run start block, the event line, the end line, the startup line and the Optuna lines
  - Check that only an interrupted run is advised the rerun that continues it
  - Check that paths are relative to the storage root and credentials never reach diagnostics

Design principles:
  - Assertions protect information content without freezing incidental prose

This module does NOT:
  - Publish W&B data or calculate training metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from support import configs

from src import common, experiments

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
    """Show start block, event lines with budget progress and point, and the end line without absolute paths."""
    reporter = _reporter(tmp_path, monkeypatch)
    budget_epochs = reporter.config["training"]["budget_epochs"]
    reporter.startup(resolved_device="cpu")
    start_lines = capsys.readouterr().out.splitlines()
    assert len(start_lines) <= 4
    assert reporter.config["run"]["name"] in start_lines[0]
    assert "cpu" in start_lines[0]
    assert "new run" in start_lines[0]
    assert f"seed {reporter.config['run']['seed']}" in start_lines[0]
    assert any(f"budget {budget_epochs} epochs" in line for line in start_lines)

    reporter.event(
        7,
        {
            "epoch": 2.0,
            "budget/progress": 0.5,
            "validation/point": 10.0,
            "train/loss_total": 0.03215,
            "train/loss_data": 0.03215,
            f"id/{reporter.objective_id}": 0.01234,
            "checkpoint/new_best": 1.0,
            "optimization/learning_rate": 3.88e-4,
            "system/event_duration_seconds": 12.8,
            "system/train_duration_seconds": 10.0,
            "system/id_evaluation_duration_seconds": 2.0,
            "system/train_samples_per_second": 45.0,
        },
    )
    plain_event = {"epoch": 3.0, "budget/progress": 0.55, "train/loss_total": 0.03, "train/loss_data": 0.03, "optimization/learning_rate": 3.88e-4}
    reporter.event(8, plain_event)
    reporter.startup_phase("training_start", {"updates_per_epoch": 1})
    budget = {
        "unit": "units",
        "total": 40,
        "units": 40,
        "supervised_transitions": 40,
        "train_seconds": 9.5,
        "validated_point": 20,
        "skipped_points": [],
    }
    reporter.final({"best_epoch": 2, "completed_epoch": 3, "best_metric": 0.01234, "budget": budget}, total_wall_seconds=125.0)
    evaluated, plain, _startup, end = capsys.readouterr().out.splitlines()
    for expected in ("epoch 2", "step 7", "50.0%", "point 10/20", "loss 3.215e-02", "val 1.234e-02 *best", "lr 3.88e-04", "13 s", "45.0 units/s"):
        assert expected in evaluated
    assert "val" not in plain
    assert "point" not in plain
    assert "[" not in evaluated  # a single loss term is not repeated as a component
    for expected in (
        "completed after 3 epochs (40 of 40 units, ",
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


def test_end_line_names_a_supervised_transition_budget_and_reports_the_optimisation_seconds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matched stage ends on its supervised transitions; the line shows them against the budget and the measured seconds beside."""
    reporter = _reporter(tmp_path, monkeypatch)
    budget = {
        "unit": "supervised_transitions",
        "total": 40,
        "units": 2,
        "supervised_transitions": 42,
        "train_seconds": 10.2,
        "validated_point": 20,
        "skipped_points": [],
    }

    reporter.final({"best_epoch": 3, "completed_epoch": 3, "best_metric": 0.5, "budget": budget}, total_wall_seconds=30.0)

    assert "completed after 3 epochs (42 of 40 supervised_transitions, 10 s optimisation)" in capsys.readouterr().out


def test_loss_components_share_the_epoch_line_only_when_several_remain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append compact components when more than the total loss exists."""
    reporter = _reporter(tmp_path, monkeypatch)
    reporter.event(1, {"train/loss_total": 0.03, "train/loss_data": 0.03, "train/loss_data_T": 0.0168, "train/loss_data_phi": 0.015})

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

    reporter.startup_phase("training_start", {"updates_per_epoch": 76})
    line = capsys.readouterr().out
    assert len(line.splitlines()) == 1
    for expected in ("startup", "2 Train cases", "152 items", "126x201", "scaler hit", "76 updates/epoch"):
        assert expected in line


def test_only_an_interrupted_run_is_advised_the_rerun_that_continues_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed run never continues, also beside its last checkpoint, so its next step advises no rerun; an interrupted one keeps it."""
    reporter = _reporter(tmp_path, monkeypatch)
    last = common.paths.resolve_last_checkpoint_file(reporter.run_dir)
    last.parent.mkdir(parents=True)
    last.write_bytes(b"")

    reporter.failure(RuntimeError("boom"), status="failed")
    failed = capsys.readouterr().err
    reporter.failure(KeyboardInterrupt(), status="interrupted")
    interrupted = capsys.readouterr().err

    assert "next:" in failed
    assert "rerun" not in failed.lower()
    assert "rerun" in interrupted


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

    experiments.console.optuna_trial_event("diverged", trial=4, sampled=sampled, duration_seconds=1.0, objective=0.9, step=3)
    experiments.console.optuna_trial_event("failed", trial=3, sampled=sampled, duration_seconds=1.0)
    captured = capsys.readouterr()
    assert "diverged@3" in captured.out
    assert "trial   3  failed" in captured.err

    experiments.console.optuna_study_event(
        "completed",
        study="fno_search",
        attempted_trials=4,
        trial_counts={"completed": 1, "pruned": 1, "diverged": 1, "no_eligible_checkpoint": 0, "failed": 1},
        best_trial=1,
        best_objective=1.379134,
        best_parameters=sampled,
        duration_seconds=590.0,
        study_seconds=1200.0,
        storage="sqlite://<configured>",
    )
    study = capsys.readouterr().out
    for expected in ("fno_search", "4 trials", "1 pruned, 1 diverged, 1 failed", "best trial 1", "study 20 min", "batch=16"):
        assert expected in study
    assert "no_eligible_checkpoint" not in study
