"""Protect matched A0 -> B -> A1 stage sequencing, reuse, resume routing, and retry admission.

Responsibilities:
  - Check ordered execution, completed-A0 reuse, and automatic continuation of incomplete leaves
  - Check that a completed A0 is extended only while no B/A1 pair pins its handoff
  - Check that an inadmissible A0 handoff prevents B allocation and that a childless failure can retry
  - Check that a source-binding refusal of a stage continuation reaches the caller unwrapped

Design principles:
  - Fake leaf execution and receipt services give deterministic evidence of what the run service dispatches
  - Parent records, saved configs, and leaf directories are real test-owned files

This module does NOT:
  - Train models, measure budgets, or validate receipt contents
  - Cover failure reporting of B/A1 stages, which the matched-budget module owns
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from support import configs

from src.experiments import experiments_matched_budget as budget
from src.experiments import experiments_run as run_service
from src.experiments import experiments_run_identity as identity
from src.experiments.config import experiments_config_loader as config_loader
from src.experiments.config import experiments_config_transient_plan as plan_service
from src.learning.transient.learning_transient_handoff import HANDOFF_DIRECTORY

if TYPE_CHECKING:
    from collections.abc import Mapping

_Execution = tuple[str, Path | None]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write one test-owned authored matched plan at its canonical task path."""
    return configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)


def _install_stage_fakes(monkeypatch: pytest.MonkeyPatch, *, fail_arm: str | None = None) -> tuple[list[_Execution], list[Path]]:
    """Replace leaf execution and receipt services with deterministic local evidence."""
    executions: list[_Execution] = []
    handoff_checks: list[Path] = []
    completed: dict[Path, dict[str, Any]] = {}

    def fake_execute(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        arm = str(config["training"]["comparison_arm"])
        resume = kwargs.get("resume")
        # The real continuation rejects an output-root override: the existing leaf is already located.
        assert resume is None or kwargs.get("output_root") is None
        executions.append((arm, None if resume is None else Path(resume)))
        if arm == fail_arm:
            msg = f"stage {arm} failure"
            raise RuntimeError(msg)
        run_dir = run_service._stage_destination(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        if arm == "a0":
            manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
            manifest.parent.mkdir(exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
        completed[run_dir] = config
        return {
            "run_dir": run_dir,
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "device_resolution": kwargs["device_resolution"],
        }

    def fake_validate_completed(run_dir: Path | str) -> dict[str, Any]:
        if Path(run_dir) not in completed:
            msg = "incomplete synthetic leaf"
            raise run_service.RunLifecycleError(msg)
        return {"config": completed[Path(run_dir)], "summary": {"best_epoch": 1, "best_metric": 0.5, "train_seconds": 1.0}}

    def fake_handoff_check(run_dir: Path, **_kwargs: Any) -> None:
        handoff_checks.append(run_dir)

    monkeypatch.setattr(run_service, "_run_resolved_experiment", fake_execute)
    monkeypatch.setattr(run_service, "validate_completed_run", fake_validate_completed)
    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", fake_handoff_check)
    monkeypatch.setattr(budget, "build_b_receipt", lambda **_kwargs: {"test": "receipt"})
    monkeypatch.setattr(budget, "publish_b_receipt", lambda *_args: None)
    monkeypatch.setattr(budget, "load_b_receipt", lambda *_args, **_kwargs: {"test": "receipt"})
    return executions, handoff_checks


def _requested_plan(output_root: Path) -> plan_service.MatchedA1BPlan:
    """Resolve the authored plan under one output root, as the run service does."""
    plan = configs.resolve_plan(configs.transient_matched_config(), output_root=output_root)
    return plan_service.MatchedA1BPlan(*(run_service._with_output_root(config, output_root) for config in plan.stages))


def _publish_parent_record(plan: plan_service.MatchedA1BPlan, output_root: Path, config_path: Path) -> Path:
    """Publish the current-schema parent record required by an explicit child resume."""
    record = identity.build_transient_experiment_record(plan.a0, plan.b, stage_a1=plan.a1, config_path=config_path)
    return identity.admit_fresh_transient_experiment_record(record, output_root=output_root)


def _create_leaf(config: Mapping[str, Any]) -> Path:
    """Create one persisted stage leaf with its exact saved effective config; A0 also holds its handoff."""
    run_dir = run_service._stage_destination(config)
    configs.write_yaml(run_dir / "config.yaml", config)
    if config["training"]["comparison_arm"] == "a0":
        (run_dir / HANDOFF_DIRECTORY).mkdir(exist_ok=True)
        (run_dir / HANDOFF_DIRECTORY / "manifest.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_fresh_plan_executes_a0_then_b_then_a1_and_returns_a1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    """A fresh matched plan runs the three leaves in strict order and publishes one parent record."""
    executions, handoff_checks = _install_stage_fakes(monkeypatch)

    outcome = run_service.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert [arm for arm, resume in executions] == ["a0", "b", "a1"]
    assert all(resume is None for _arm, resume in executions)
    assert handoff_checks
    assert tuple(outcome["stage_runs"]) == ("a0", "b", "a1")
    assert [path.name for path in outcome["stage_runs"].values()] == ["a0", "b", "a1"]
    assert outcome["run_dir"] == outcome["stage_runs"]["a1"]
    assert outcome["training_status"] == "completed"
    assert outcome["experiment_record"].is_file()


def test_selected_b_reuses_the_completed_shared_a0_without_rerunning_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    """An unselected A0 is admitted through the handoff guard and never executed again."""
    executions, handoff_checks = _install_stage_fakes(monkeypatch)
    a0_dir = _create_leaf(_requested_plan(tmp_path).a0)

    outcome = run_service.run_experiment(config_path, device="cpu", output_root=tmp_path, train_stages=("b",))

    assert [arm for arm, _resume in executions] == ["b"]
    assert handoff_checks
    assert set(handoff_checks) == {a0_dir}
    assert outcome["stage_runs"]["a0"] == a0_dir


@pytest.mark.parametrize("arm", ["a0", "b"])
def test_rerunning_the_plan_continues_the_incomplete_selected_leaf_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path, arm: plan_service.MatchedStageName
) -> None:
    """The same command continues the named incomplete leaf in place instead of allocating a new one."""
    executions, _handoff_checks = _install_stage_fakes(monkeypatch)
    plan = _requested_plan(tmp_path)
    _publish_parent_record(plan, tmp_path, config_path)
    _create_leaf(plan.a0)
    leaf = _create_leaf(plan.stage(arm))

    run_service.run_experiment(config_path, device="cpu", output_root=tmp_path, train_stages=(arm,))

    assert executions == [(arm, leaf)]


def test_source_binding_refusal_of_a_stage_continuation_is_not_wrapped_as_a_stage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """An implicit source-tree switch is an admission decision: it reaches the caller unchanged, with no resume advice."""
    _install_stage_fakes(monkeypatch)
    plan = _requested_plan(tmp_path)
    _publish_parent_record(plan, tmp_path, config_path)
    _create_leaf(plan.a0)

    def refuse(_config: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        msg = "This run is bound to source tree x."
        raise run_service.SourceBindingError(msg)

    monkeypatch.setattr(run_service, "_run_resolved_experiment", refuse)

    with pytest.raises(run_service.SourceBindingError, match="bound to source tree"):
        run_service.run_experiment(config_path, device="cpu", output_root=tmp_path, train_stages=("a0",))


def test_existing_a0_with_changed_science_at_the_same_name_is_refused_with_the_revision_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """Changed A0 science under an existing seed group is a different A0, never a continuation."""
    executions, _handoff_checks = _install_stage_fakes(monkeypatch)
    plan = _requested_plan(tmp_path)
    changed = copy.deepcopy(plan.a0)
    changed["optimizer"]["lr"] = float(plan.a0["optimizer"]["lr"]) * 2
    changed["run"]["layout"]["a0_id"] = config_loader.allocation_digest(changed)[:16]
    assert changed["run"]["layout"]["a0_id"] != plan.a0["run"]["layout"]["a0_id"]
    a0_dir = _create_leaf(changed)

    with pytest.raises(run_service.RunLifecycleError, match=r"another scientific identity.*--revision"):
        run_service.run_experiment(config_path, device="cpu", output_root=tmp_path, train_stages=("a0",))

    assert executions == []
    assert (a0_dir / HANDOFF_DIRECTORY / "manifest.json").is_file()


def test_another_a0_epoch_budget_trains_in_its_own_seed_group_beside_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """A0's epoch budget names its seed group, so a different budget needs no revision and never touches the first A0."""
    executions, _handoff_checks = _install_stage_fakes(monkeypatch)
    budget_epochs = int(_requested_plan(tmp_path).a0["training"]["epochs"])

    first = run_service.run_experiment(config_path, device="cpu", output_root=tmp_path, train_stages=("a0",))
    other = run_service.run_experiment(config_path, device="cpu", output_root=tmp_path, train_stages=("a0",), epochs=budget_epochs + 1)

    assert first["run_dir"].parent.name.endswith(f"__a0ep-{budget_epochs}")
    assert other["run_dir"].parent.name.endswith(f"__a0ep-{budget_epochs + 1}")
    assert first["run_dir"].parent.parent == other["run_dir"].parent.parent
    assert [arm for arm, _resume in executions] == ["a0", "a0"]


def test_inadmissible_a0_handoff_prevents_b_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    """A completed A0 without admissible handoff evidence never permits B allocation."""
    executions, _handoff_checks = _install_stage_fakes(monkeypatch)

    def reject_handoff(*_args: Any, **_kwargs: Any) -> None:
        msg = "handoff admission failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(run_service, "_validate_reusable_stage_a", reject_handoff)

    with pytest.raises(RuntimeError, match="handoff admission failure"):
        run_service.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert [arm for arm, _resume in executions] == ["a0"]


def test_a0_failure_before_allocation_allows_an_exact_fresh_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    """Recover an exact childless parent while still preventing B and A1 allocation after the failure."""
    failed, _handoff_checks = _install_stage_fakes(monkeypatch, fail_arm="a0")

    with pytest.raises(run_service.MatchedStageExecutionError) as captured:
        run_service.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert captured.value.stage == "a0"
    assert [arm for arm, _resume in failed] == ["a0"]
    retried, _handoff_checks = _install_stage_fakes(monkeypatch)

    outcome = run_service.run_experiment(config_path, device="cpu", output_root=tmp_path)

    assert [arm for arm, _resume in retried] == ["a0", "b", "a1"]
    assert outcome["run_dir"] == outcome["stage_runs"]["a1"]
