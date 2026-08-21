# ruff: noqa: S101, SLF001, TRY003, EM101
"""Protect automatic authored transient A0-to-B runtime sequencing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import experiments
from src.experiments.config import experiments_config_transient_plan as transient_plan

_CONFIG = Path("configs/learning/transient_drying/experiments/fno_m128x160_h64_l3__lentil_chickpea__s9.yaml")


def _install_sequence_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_a: bool = False,
) -> tuple[list[tuple[str, Path | None]], list[Path]]:
    """Replace leaf execution and handoff admission with deterministic local evidence."""
    executions: list[tuple[str, Path | None]] = []
    validated: list[Path] = []

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        stage = str(config["training"]["stage"])
        resume = kwargs.get("resume")
        executions.append((stage, None if resume is None else Path(resume)))
        if stage == "a" and fail_a:
            raise RuntimeError("stage-a failure")
        run_dir = Path(resume) if resume is not None else experiments.run._stage_destination(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": run_dir, "result": {"stage": stage}, "device_resolution": kwargs["device_resolution"]}

    def fake_validate(run_dir: Path, **_kwargs: Any) -> None:
        validated.append(run_dir)

    monkeypatch.setattr(experiments.run, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(experiments.run, "_validate_reusable_stage_a", fake_validate)
    return executions, validated


def test_fresh_plan_sequences_a_then_b_and_returns_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh authored plan executes independent A then B leaves in strict order."""
    executions, validated = _install_sequence_fakes(monkeypatch)

    outcome = experiments.run.run_experiment(_CONFIG, device="cpu", output_root=tmp_path)

    assert [stage for stage, _resume in executions] == ["a", "b"]
    assert validated == [outcome["stage_runs"]["a"]]
    assert outcome["run_dir"] == outcome["stage_runs"]["b"]
    assert "stage_a0" in outcome["stage_runs"]["a"].name
    assert "stage_b" in outcome["stage_runs"]["b"].name


def test_completed_a_boundary_is_reused_without_retraining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A validated completed A leaf permits B allocation without mutating or rerunning A."""
    executions, validated = _install_sequence_fakes(monkeypatch)
    plan = transient_plan.load_and_resolve_transient_training_plan(_CONFIG)
    a = experiments.run._with_output_root(plan.stage_a, tmp_path)
    experiments.run._stage_destination(a).mkdir(parents=True)

    outcome = experiments.run.run_experiment(_CONFIG, device="cpu", output_root=tmp_path)

    assert executions == [("b", None)]
    assert validated == [outcome["stage_runs"]["a"]]


@pytest.mark.parametrize("stage", ["a", "b"])
def test_explicit_resume_routes_only_to_matching_derived_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Explicit authored-plan resume keeps the selected child leaf authoritative."""
    executions, validated = _install_sequence_fakes(monkeypatch)
    plan = transient_plan.load_and_resolve_transient_training_plan(_CONFIG)
    selected = experiments.run._with_output_root(plan.stage_a if stage == "a" else plan.stage_b, tmp_path)
    resume_dir = experiments.run._stage_destination(selected)
    resume_dir.mkdir(parents=True)
    if stage == "b":
        a = experiments.run._with_output_root(plan.stage_a, tmp_path)
        experiments.run._stage_destination(a).mkdir(parents=True)

    outcome = experiments.run.run_experiment(_CONFIG, device="cpu", output_root=tmp_path, resume=resume_dir)

    if stage == "a":
        assert executions == [("a", resume_dir), ("b", None)]
    else:
        assert executions == [("b", resume_dir)]
    assert validated
    assert outcome["run_dir"] == outcome["stage_runs"]["b"]


def test_handoff_admission_failure_prevents_b_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed A without admissible handoff evidence never permits B allocation."""
    executions, _validated = _install_sequence_fakes(monkeypatch)

    def fail_handoff(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("handoff admission failure")

    monkeypatch.setattr(experiments.run, "_validate_reusable_stage_a", fail_handoff)
    with pytest.raises(RuntimeError, match="handoff admission failure"):
        experiments.run.run_experiment(_CONFIG, device="cpu", output_root=tmp_path)

    assert executions == [("a", None)]


def test_a_admission_failure_prevents_b_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No B leaf is allocated when Stage A cannot complete or publish its handoff."""
    executions, _validated = _install_sequence_fakes(monkeypatch, fail_a=True)

    with pytest.raises(RuntimeError, match="stage-a failure"):
        experiments.run.run_experiment(_CONFIG, device="cpu", output_root=tmp_path)

    assert executions == [("a", None)]
