# ruff: noqa: S101, SLF001, TRY003, EM101
"""Protect automatic authored transient A0-to-B runtime sequencing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from support import configs

from src import experiments
from src.experiments import experiments_run_identity as run_identity
from src.experiments.config import experiments_config_transient_plan as transient_plan


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write one test-owned authored transient plan at its canonical task path."""
    return configs.write_yaml(
        tmp_path / "configs" / "learning" / "transient_drying" / "experiments" / "plan.yaml",
        configs.transient_two_stage_config(),
    )


def _install_sequence_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_a: bool = False,
    reject_first_validation: bool = False,
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
        if reject_first_validation and len(validated) == 1:
            raise experiments.run.RunLifecycleError("Stage A requires checkpoint resume.")

    monkeypatch.setattr(experiments.run, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(experiments.run, "_validate_reusable_stage_a", fake_validate)
    return executions, validated


def _publish_parent_record(
    plan: transient_plan.TransientTrainingPlan,
    output_root: Path,
    config_path: Path,
) -> Path:
    """Publish the current-schema parent required by an explicit child resume."""
    stage_a = experiments.run._with_output_root(plan.stage_a, output_root)
    stage_b = experiments.run._with_output_root(plan.stage_b, output_root)
    record = run_identity.build_transient_experiment_record(stage_a, stage_b, config_path=config_path)
    return run_identity.publish_transient_experiment_record(record, output_root=output_root)


def test_fresh_plan_sequences_a_then_b_and_returns_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
) -> None:
    """A fresh authored plan executes independent A then B leaves in strict order."""
    executions, validated = _install_sequence_fakes(monkeypatch)

    outcome = experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert [stage for stage, _resume in executions] == ["a", "b"]
    assert validated == [outcome["stage_runs"]["a"]]
    assert outcome["run_dir"] == outcome["stage_runs"]["b"]
    assert outcome["stage_runs"]["a"].name.endswith("_a0")
    assert outcome["stage_runs"]["b"].name.endswith("_b")


def test_fresh_plan_rejects_existing_a_and_requires_explicit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
) -> None:
    """Never reinterpret a pre-existing child as a fresh-run continuation."""
    executions, validated = _install_sequence_fakes(monkeypatch)
    plan = transient_plan.load_and_resolve_transient_training_plan(config_path)
    stage_a = experiments.run._with_output_root(plan.stage_a, tmp_path)
    experiments.run._stage_destination(stage_a).mkdir(parents=True)

    with pytest.raises(experiments.run.ExistingRunAdmissionError):
        experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert executions == []
    assert validated == []


@pytest.mark.parametrize("stage", ["a", "b"])
def test_explicit_resume_routes_only_to_matching_derived_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    stage: str,
) -> None:
    """Explicit authored-plan resume keeps the selected child leaf authoritative."""
    executions, validated = _install_sequence_fakes(
        monkeypatch,
        reject_first_validation=stage == "a",
    )
    plan = transient_plan.load_and_resolve_transient_training_plan(config_path)
    _publish_parent_record(plan, tmp_path, config_path)
    selected = experiments.run._with_output_root(plan.stage_a if stage == "a" else plan.stage_b, tmp_path)
    resume_dir = experiments.run._stage_destination(selected)
    resume_dir.mkdir(parents=True)
    if stage == "b":
        a = experiments.run._with_output_root(plan.stage_a, tmp_path)
        experiments.run._stage_destination(a).mkdir(parents=True)

    outcome = experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path, resume=resume_dir)

    if stage == "a":
        assert executions == [("a", resume_dir), ("b", None)]
    else:
        assert executions == [("b", resume_dir)]
    assert validated
    assert outcome["run_dir"] == outcome["stage_runs"]["b"]


def test_explicit_resume_rejects_a_different_persisted_parent_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
) -> None:
    """Bind resume to the complete requested parent rather than child paths alone."""
    executions, validated = _install_sequence_fakes(monkeypatch)
    plan = transient_plan.load_and_resolve_transient_training_plan(config_path)
    alternate_path = configs.write_yaml(tmp_path / "alternate-plan.yaml", configs.transient_two_stage_config())
    _publish_parent_record(plan, tmp_path, alternate_path)
    stage_a = experiments.run._with_output_root(plan.stage_a, tmp_path)
    resume_dir = experiments.run._stage_destination(stage_a)
    resume_dir.mkdir(parents=True)

    with pytest.raises(experiments.run.RunLifecycleError, match="parent identity"):
        experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path, resume=resume_dir)

    assert executions == []
    assert validated == []


def test_handoff_admission_failure_prevents_b_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
) -> None:
    """A completed A without admissible handoff evidence never permits B allocation."""
    executions, _validated = _install_sequence_fakes(monkeypatch)

    def fail_handoff(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("handoff admission failure")

    monkeypatch.setattr(experiments.run, "_validate_reusable_stage_a", fail_handoff)
    with pytest.raises(RuntimeError, match="handoff admission failure"):
        experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert executions == [("a", None)]


def test_a_preallocation_failure_allows_an_exact_fresh_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
) -> None:
    """Recover an exact childless parent while still preventing premature B allocation."""
    failed_executions, _validated = _install_sequence_fakes(monkeypatch, fail_a=True)

    with pytest.raises(RuntimeError, match="stage-a failure"):
        experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert failed_executions == [("a", None)]
    retry_executions, _validated = _install_sequence_fakes(monkeypatch)

    outcome = experiments.run.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert [stage for stage, _resume in retry_executions] == ["a", "b"]
    assert outcome["run_dir"] == outcome["stage_runs"]["b"]
