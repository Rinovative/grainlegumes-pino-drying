# ruff: noqa: S101, SLF001
"""Protect transient adapter evaluation preparation and transactional restore."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.learning.transient.learning_transient_adapter import TransientTrainingAdapter
from src.learning.transient.learning_transient_curriculum import (
    MatchedComputeController,
    RolloutCurriculumState,
    TeacherHandoffIdentity,
    TransientTrainingSpec,
)
from src.learning.transient.learning_transient_tensorizer import TransientBatch


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _spec() -> TransientTrainingSpec:
    handoff = TeacherHandoffIdentity(
        source_run_name="synthetic",
        source_checkpoint_sha256=_digest("checkpoint"),
        source_scaling_sha256=_digest("scaling"),
        task_contract_sha256=_digest("task"),
        tensorizer_sha256=_digest("tensorizer"),
        model_kind="fno",
        input_profile="canonical_physics_complete_v1",
    )
    controller = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        clock_kind="optimizer_steps",
        config_digest=_digest("config"),
        teacher_handoff=handoff,
        planned_teacher_forcing_budget_steps=4,
    )
    return TransientTrainingSpec(
        stage="stage_b_self_fed",
        model_kind="fno",
        controller=controller,
        curriculum_seed=19,
    )


def _batch() -> TransientBatch:
    state = torch.zeros(1, 4, 1, 1)
    target = torch.zeros(1, 2, 4, 1, 1)
    sequence = torch.zeros(1, 2, 5, 1, 1)
    return TransientBatch(
        state=state,
        target=target,
        static=torch.zeros(1, 7, 1, 1),
        boundary=torch.zeros(1, 2, 9),
        scalars=torch.zeros(1, 8),
        t_n=torch.zeros(1, 2),
        t_n_plus_1=torch.ones(1, 2),
        dt=torch.ones(1, 2),
        step_input=sequence[:, 0],
        sequence_input=sequence,
        scaled_target=target,
    )


def _adapter_with_fake_tensorizer() -> TransientTrainingAdapter:
    adapter: Any = object.__new__(TransientTrainingAdapter)
    adapter.spec = _spec()
    adapter.curriculum_state = RolloutCurriculumState.create(adapter.spec.curriculum, seed=adapter.spec.curriculum_seed)
    adapter._epoch_index = 0
    adapter._optimizer_events = 0
    adapter.tensorizer = SimpleNamespace(
        scaling=SimpleNamespace(device=torch.device("cpu")),
        tensorize=lambda raw: raw["batch"],
    )
    return adapter


def test_evaluation_preparation_does_not_consume_curriculum_rng() -> None:
    """Keep full Stage-B source windows intact until deterministic evaluation cropping."""
    adapter = _adapter_with_fake_tensorizer()
    raw = {"batch": _batch()}
    before = adapter.curriculum_state.state_dict()
    prepared = adapter.prepare_batch(raw, device=torch.device("cpu"), training=False)
    assert prepared is raw["batch"]
    assert adapter.curriculum_state.state_dict() == before


def test_adapter_restore_is_transactional_when_nested_curriculum_is_invalid() -> None:
    """Leave all live adapter continuation evidence unchanged after nested failure."""
    adapter = _adapter_with_fake_tensorizer()
    before = copy.deepcopy(adapter.state_dict())
    invalid = copy.deepcopy(before)
    invalid["curriculum"]["completed_milestones"] = ["invalid"]
    with pytest.raises(ValueError, match="completed_milestones"):
        adapter.load_state_dict(invalid)
    assert adapter.state_dict() == before


def test_validation_work_is_persisted_but_excluded_from_budget_progress() -> None:
    """Keep completed validation time as secondary matched-compute evidence."""
    adapter = _adapter_with_fake_tensorizer()
    before = adapter.spec.controller.progress
    adapter.record_validation_work(1.25)
    assert adapter.spec.controller.validation_seconds == pytest.approx(1.25)
    assert adapter.spec.controller.progress == before
    assert adapter.telemetry_state()["validation_seconds"] == pytest.approx(1.25)
    assert adapter.state_dict()["controller"]["validation_seconds"] == pytest.approx(1.25)


def test_adapter_uses_only_current_controller_schema() -> None:
    """Persist version 1 and reject controller state from any other schema."""
    adapter = _adapter_with_fake_tensorizer()
    current = copy.deepcopy(adapter.state_dict())
    assert current["controller"]["schema_version"] == 1

    invalid = copy.deepcopy(current)
    invalid["controller"]["schema_version"] = 0
    with pytest.raises(ValueError, match="strict schema"):
        adapter.load_state_dict(invalid)
