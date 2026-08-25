# ruff: noqa: S101
"""Protect transient CLI reporting and task-aware post-training artifacts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from src import analysis, experiments
from src.experiments.cli import cli_train

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _install_artifact_spies(
    monkeypatch: pytest.MonkeyPatch,
    observed: list[object],
) -> None:
    """Install thin completed-run and artifact-service spies."""
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda run_dir: observed.append(("validate", run_dir)))
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda device: observed.append(("cleanup", device)))
    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_or_build_run_artifacts",
        lambda run_dir, **kwargs: observed.append(("artifacts", run_dir, kwargs)) or SimpleNamespace(role_actions={"id": "reused"}),
    )


def test_two_stage_cli_reports_both_leaves_and_builds_final_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Build artifacts for the terminal B leaf while reporting both stage directories."""
    a_dir = tmp_path / "stage_a0"
    b_dir = tmp_path / "stage_b"
    resolution = SimpleNamespace(device=torch.device("cpu"))
    observed: list[object] = []
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": b_dir,
            "result": {"best_epoch": 1, "best_metric": 0.25},
            "device_resolution": resolution,
            "stage_runs": {"a": a_dir, "b": b_dir},
        },
    )
    _install_artifact_spies(monkeypatch, observed)

    assert cli_train.main(["plan.yaml"]) == 0

    output = capsys.readouterr().out
    assert f"Stage A run directory: {a_dir}" in output
    assert f"Stage B run directory: {b_dir}" in output
    assert "Post-training artifacts: reused" in output
    assert ("validate", b_dir) in observed
    assert ("cleanup", resolution.device) in observed
    assert ("artifacts", b_dir, {"device_resolution": resolution}) in observed


def test_single_stage_transient_cli_builds_task_aware_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Send standalone transient A0, B, or A+ runs through the shared artifact service."""
    run_dir = tmp_path / "single_stage_transient"
    resolution = SimpleNamespace(device=torch.device("cpu"))
    observed: list[object] = []
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": run_dir,
            "result": {"best_epoch": 1, "best_metric": 0.25},
            "device_resolution": resolution,
            "task": "transient_drying",
        },
    )
    _install_artifact_spies(monkeypatch, observed)

    assert cli_train.main(["stage.yaml"]) == 0

    output = capsys.readouterr().out
    assert "Post-training artifacts: reused" in output
    assert "not implemented" not in output
    assert ("artifacts", run_dir, {"device_resolution": resolution}) in observed
