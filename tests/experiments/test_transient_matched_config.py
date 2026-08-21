# ruff: noqa: S101
"""Protect local transient matched-config derivation semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import experiments
from src.experiments.cli import cli_transient_matched_config as matched
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_transient_plan as transient_plan
from src.learning.transient.learning_transient_curriculum import RolloutCurriculum

_ROLLOUT_HORIZON = 32
_B_STEPS = 7


def _plan(model_kind: str = "fno") -> transient_plan.TransientTrainingPlan:
    """Return one resolved maintained transient two-stage plan."""
    names = {
        "fno": "fno_m128x160_h64_l3__material_pilot__s9.yaml",
        "rno": "rno_m24x24_h16_l3__material_pilot__s9.yaml",
    }
    return transient_plan.load_and_resolve_transient_training_plan(Path("configs/learning/transient_drying/experiments") / names[model_kind])


def _a0_config() -> dict[str, Any]:
    """Return one isolated resolved transient A0 configuration."""
    return dict(_plan().stage_a)


def test_b_and_a_plus_configs_follow_completed_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Derive canonical B windows and A+ only from completed B evidence."""
    a0 = _a0_config()
    source = {"config": a0, "summary": {"resolved_device": "cpu"}}
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: source)
    b_path = tmp_path / "b.yaml"
    matched.generate_matched_config(source_run_dir=tmp_path / "a0", output_path=b_path, arm="b", budget=_B_STEPS)
    b = loader.validate_resolved_config(loader.load_yaml(b_path))
    assert b["temporal"]["sampling"] == {
        "mode": "rollout_window",
        "rollout_length": _ROLLOUT_HORIZON,
        "window_stride": _ROLLOUT_HORIZON,
        "window_offset": 0,
    }
    assert b["training"]["fixed_evaluation_horizon"] == _ROLLOUT_HORIZON
    assert b["training"]["curriculum"]["lengths"] == [2, 4, 8, 16, _ROLLOUT_HORIZON]
    assert b["training"]["curriculum"]["milestone_fractions"] == list(RolloutCurriculum.DEFAULT_MILESTONE_FRACTIONS)
    assert b["training"]["matched_compute"]["planned_steps"] == _B_STEPS
    assert b["training"]["matched_compute"]["rollout_reference_steps"] is None

    completed_b = {
        "config": b,
        "summary": {
            "resolved_device": "cpu",
            "terminal_controller": {"successful_optimizer_steps": 9},
        },
    }
    calls = iter((source, completed_b))
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: next(calls))
    a_plus_path = tmp_path / "a_plus.yaml"
    matched.generate_matched_config(source_run_dir=tmp_path / "a0", output_path=a_plus_path, arm="a_plus", b_run_dir=tmp_path / "b")
    a_plus = loader.validate_resolved_config(loader.load_yaml(a_plus_path))
    assert a_plus["training"]["stage"] == "a"
    assert a_plus["training"]["comparison_arm"] == "a_plus"
    assert a_plus["training"]["matched_compute"] == {
        "planned_seconds": None,
        "planned_steps": 9,
        "rollout_reference_seconds": None,
        "rollout_reference_steps": 9,
    }
    assert a_plus["temporal"] == b["temporal"]
    assert a_plus["training"]["curriculum"] == b["training"]["curriculum"]


def test_a_plus_rejects_caller_budget_that_disagrees_with_completed_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep A+ matched to terminal B evidence rather than a caller-selected count."""
    a0 = _a0_config()
    b = _a0_config()
    b["training"].update(
        {
            "stage": "b",
            "comparison_arm": "b",
            "teacher_handoff": {"source_run_name": a0["run"]["name"]},
            "matched_compute": {"planned_seconds": None, "planned_steps": 2, "rollout_reference_seconds": None, "rollout_reference_steps": None},
        }
    )
    b["temporal"]["sampling"] = {"mode": "rollout_window", "rollout_length": 32, "window_stride": 32, "window_offset": 0}
    b["training"]["fixed_evaluation_horizon"] = 32
    b["training"]["curriculum"] = {
        "lengths": [2, 4, 8, 16, 32],
        "milestone_fractions": list(RolloutCurriculum.DEFAULT_MILESTONE_FRACTIONS),
        "seed": 9,
    }
    source = {"config": a0, "summary": {"resolved_device": "cpu"}}
    completed_b = {"config": b, "summary": {"resolved_device": "cpu", "terminal_controller": {"successful_optimizer_steps": 3}}}
    calls = iter((source, completed_b))
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: next(calls))
    with pytest.raises(ValueError, match="exactly equal"):
        matched.generate_matched_config(
            source_run_dir=tmp_path / "a0", output_path=tmp_path / "a_plus.yaml", arm="a_plus", b_run_dir=tmp_path / "b", budget=2
        )


def test_transient_config_admits_b_without_reference_but_rejects_rno_one_step() -> None:
    """Admit fresh B budgeting while requiring contiguous RNO reference sequences."""
    b = dict(_plan().stage_b)
    assert loader.validate_resolved_config(b)["training"]["comparison_arm"] == "b"
    b["training"]["fixed_evaluation_horizon"] = 3
    with pytest.raises(loader.ConfigError, match="one admitted curriculum horizon"):
        loader.validate_resolved_config(b)

    rno = dict(_plan("rno").stage_a)
    rno["temporal"]["sampling"] = {"mode": "one_step_transition"}
    with pytest.raises(loader.ConfigError, match="Stage B and Stage-A RNO require"):
        loader.validate_resolved_config(rno)
