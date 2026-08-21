# ruff: noqa: S101
"""Protect two-stage transient CLI reporting and artifact deferral."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import experiments
from src.experiments.cli import cli_train

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_two_stage_cli_reports_both_leaves_without_artifact_builder(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Keep a successful two-stage transient run successful while artifacts are out of scope."""
    a_dir = tmp_path / "stage_a0"
    b_dir = tmp_path / "stage_b"
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": b_dir,
            "result": {"best_epoch": 1, "best_metric": 0.25},
            "device_resolution": object(),
            "stage_runs": {"a": a_dir, "b": b_dir},
        },
    )

    assert cli_train.main(["plan.yaml"]) == 0

    output = capsys.readouterr().out
    assert f"Stage A run directory: {a_dir}" in output
    assert f"Stage B run directory: {b_dir}" in output
    assert "not implemented for transient_drying" in output


def test_single_stage_transient_cli_also_defers_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Do not send standalone transient Stage A, B, or A+ runs to steady artifacts."""
    run_dir = tmp_path / "single_stage_transient"
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": run_dir,
            "result": {"best_epoch": 1, "best_metric": 0.25},
            "device_resolution": object(),
            "task": "transient_drying",
        },
    )

    assert cli_train.main(["stage.yaml"]) == 0

    output = capsys.readouterr().out
    assert "not implemented for transient_drying" in output
