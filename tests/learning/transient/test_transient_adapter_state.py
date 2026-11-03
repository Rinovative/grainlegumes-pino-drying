"""Protect transient adapter evaluation preparation and transactional restore.

Responsibilities:
  - Check evaluation preparation, nested restore rollback, work accounting, and schema admission
  - Check that every stage selects checkpoints on whole autonomous trajectories
  - Check RNO training on consecutive trajectory windows with truncated gradients, warm Stage-B starts, and the carried one-step view

Design principles:
  - Curriculum RNG and budget progress stay unchanged by evaluation preparation

This module does NOT:
  - Select curriculum lengths or execute training campaigns
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.optim.sgd import SGD

from src.learning.transient.learning_transient_adapter import TransientTrainingAdapter
from src.learning.transient.learning_transient_curriculum import (
    MatchedComputeController,
    RolloutCurriculum,
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
        planned_stage_epochs=4,
    )
    return TransientTrainingSpec(
        stage="stage_b_self_fed",
        model_kind="fno",
        controller=controller,
        curriculum_seed=19,
    )


def _batch(length: int = 2) -> TransientBatch:
    state = torch.zeros(1, 4, 1, 1)
    target = torch.zeros(1, length, 4, 1, 1)
    sequence = torch.zeros(1, length, 5, 1, 1)
    return TransientBatch(
        state=state,
        target=target,
        static=torch.zeros(1, 7, 1, 1),
        boundary=torch.zeros(1, length, 9),
        scalars=torch.zeros(1, 8),
        t_n=torch.zeros(1, length),
        t_n_plus_1=torch.ones(1, length),
        dt=torch.ones(1, length),
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
    """Keep evaluation trajectories intact without consuming curriculum randomness."""
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
    """Keep completed validation time as secondary stage-budget evidence."""
    adapter = _adapter_with_fake_tensorizer()
    before = adapter.spec.controller.progress
    adapter.record_validation_work(1.25)
    assert adapter.spec.controller.validation_seconds == pytest.approx(1.25)
    assert adapter.spec.controller.progress == before
    assert adapter.telemetry_state()["validation_seconds"] == pytest.approx(1.25)
    assert adapter.state_dict()["controller"]["validation_seconds"] == pytest.approx(1.25)


def test_adapter_uses_only_current_controller_schema() -> None:
    """Persist schema version 3 and reject controller state from any other schema."""
    adapter = _adapter_with_fake_tensorizer()
    current = copy.deepcopy(adapter.state_dict())
    assert current["controller"]["schema_version"] == 4

    invalid = copy.deepcopy(current)
    invalid["controller"]["schema_version"] = 1
    with pytest.raises(ValueError, match="strict schema"):
        adapter.load_state_dict(invalid)


def test_every_stage_selects_checkpoints_on_the_whole_autonomous_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate teacher-forced Stage A by the self-fed rollout of the whole trajectory, as Stage B."""
    from src.learning.transient import learning_transient_adapter as adapter_module  # noqa: PLC0415

    adapter = _adapter_with_fake_tensorizer()
    controller = MatchedComputeController(
        arm="A0",
        stage="stage_a_teacher_forcing",
        clock_kind="optimizer_steps",
        config_digest=_digest("a0"),
        planned_stage_epochs=2,
    )
    adapter.spec = replace(
        adapter.spec,
        stage="stage_a_teacher_forcing",
        controller=controller,
        curriculum=RolloutCurriculum(lengths=(2, 4), milestone_fractions=(0.0, 0.5)),
    )
    adapter.curriculum_state = RolloutCurriculumState.create(adapter.spec.curriculum, seed=adapter.spec.curriculum_seed)
    batch = _batch(8)
    prepared = adapter.prepare_batch({"batch": batch}, device=torch.device("cpu"), training=True)
    assert prepared is batch
    self_fed_lengths: list[int] = []
    teacher_windows: list[tuple[float, int]] = []

    def self_fed(_model: Any, evaluated: TransientBatch, _tensorizer: Any, *, model_kind: str) -> Any:
        assert model_kind == "fno"
        self_fed_lengths.append(evaluated.rollout_length)
        return SimpleNamespace(
            normalized_prediction=evaluated.scaled_target,
            normalized_target=evaluated.scaled_target,
            physical_prediction=evaluated.target,
            physical_target=evaluated.target,
        )

    def teacher(
        _model: Any, window: TransientBatch, _tensorizer: Any, *, model_kind: str, initial_hidden: object | None, capture: tuple[int, ...]
    ) -> Any:
        assert (model_kind, initial_hidden, capture) == ("fno", None, ())
        teacher_windows.append((float(window.t_n[0, 0]), window.rollout_length))
        return SimpleNamespace(
            normalized_prediction=window.t_n[:, :, None, None, None].expand(-1, -1, 4, 1, 1), physical_prediction=window.target, captured_hidden={}
        )

    monkeypatch.setattr(adapter_module, "teacher_forced_rollout", teacher)
    monkeypatch.setattr(adapter_module, "self_fed_rollout", self_fed)
    monkeypatch.setattr(adapter_module, "_TEACHER_FORCED_CHUNK", 3)
    batch.t_n.copy_(torch.arange(8, dtype=torch.float32))
    before = adapter.curriculum_state.state_dict()
    views = adapter.evaluation_step(torch.nn.Identity(), batch)

    assert self_fed_lengths == [8]
    assert views.processed_target_transitions == 8
    assert views.normalized_prediction.shape[1] == 8
    # The one-step view covers every transition in memory-bounded chunks, in order.
    assert teacher_windows == [(0.0, 3), (3.0, 3), (6.0, 2)]
    assert torch.equal(views.one_step_normalized_prediction[0, :, 0, 0, 0], torch.arange(8, dtype=torch.float32))
    assert views.one_step_physical_prediction.shape == views.physical_target.shape
    assert adapter.curriculum_state.state_dict() == before


class _StateRNO(torch.nn.Module):
    """Accumulate weighted inputs in a one-layer tensor state; each prediction is the new state."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, value: torch.Tensor, *, init_hidden_states: object | None, return_hidden_states: bool) -> Any:
        assert return_hidden_states is True
        step = value[:, 0, :4] * self.weight
        state = step if init_hidden_states is None else init_hidden_states[0] + step  # type: ignore[index]
        return state, [state]


def _trajectory_batch(batch_size: int = 2, length: int = 6, increments: torch.Tensor | None = None) -> TransientBatch:
    """Build reference trajectories whose teacher-forced inputs are the reference states; scalar 0 numbers the samples."""
    state = torch.zeros(batch_size, 4, 1, 1)
    target = torch.ones(batch_size, length, 4, 1, 1) if increments is None else increments[:, :, None, None, None].expand(-1, -1, 4, 1, 1).clone()
    reference = state.unsqueeze(1) + torch.cat((torch.zeros_like(target[:, :1]), target[:, :-1]), dim=1).cumsum(dim=1)
    sequence = torch.cat((reference, torch.ones_like(reference[:, :, :1])), dim=2)
    scalars = torch.zeros(batch_size, 8)
    scalars[:, 0] = torch.arange(batch_size)
    return TransientBatch(
        state=state,
        target=target,
        static=torch.zeros(batch_size, 7, 1, 1),
        boundary=torch.zeros(batch_size, length, 9),
        scalars=scalars,
        t_n=torch.arange(length, dtype=torch.float32).expand(batch_size, -1),
        t_n_plus_1=torch.arange(1, length + 1, dtype=torch.float32).expand(batch_size, -1),
        dt=torch.ones(batch_size, length),
        step_input=sequence[:, 0],
        sequence_input=sequence,
        scaled_target=target,
    )


def _rno_adapter(stage: str) -> TransientTrainingAdapter:
    """Return one RNO adapter on four-step trajectory windows with an identity tensorizer."""
    adapter: Any = _adapter_with_fake_tensorizer()
    if stage == "a":
        controller = MatchedComputeController(
            arm="A0", stage="stage_a_teacher_forcing", clock_kind="optimizer_steps", config_digest=_digest("rno-a0"), planned_stage_epochs=2
        )
        curriculum = RolloutCurriculum(lengths=(2, 4), milestone_fractions=(0.0, 0.5))
    else:
        controller = replace(adapter.spec.controller, config_digest=_digest("rno-b"))
        curriculum = RolloutCurriculum(lengths=(2,), milestone_fractions=(0.0,))
    adapter.spec = TransientTrainingSpec(
        stage=controller.stage,
        model_kind="rno",
        controller=controller,
        curriculum=curriculum,
        curriculum_seed=19,
        trajectory_window=4,
    )
    adapter.curriculum_state = RolloutCurriculumState.create(adapter.spec.curriculum, seed=adapter.spec.curriculum_seed)
    adapter.tensorizer = SimpleNamespace(
        scaling=SimpleNamespace(device=torch.device("cpu"), encode_state=lambda value: value, encode_delta=lambda value: value),
        reconstruct_next_state=lambda state, delta: state + delta,
        assemble_step=lambda state, *_args: torch.cat((state, torch.ones_like(state[:, :1])), dim=1),
        tensorize=lambda raw: raw["batch"],
    )
    return adapter


def test_rno_stage_a_trains_consecutive_windows_with_the_carried_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Window losses equal one long teacher-forced pass; only the last window consumes the loader items."""
    from src.learning.transient import learning_transient_adapter as adapter_module  # noqa: PLC0415
    from src.learning.transient import learning_transient_rollout as rollout_module  # noqa: PLC0415

    adapter = _rno_adapter("a")
    model = _StateRNO()
    batch = _trajectory_batch()
    loss = torch.nn.MSELoss()
    steps = list(adapter.training_steps(model, batch, loss))
    full = rollout_module.teacher_forced_rollout(model, batch, adapter.tensorizer, model_kind="rno")

    assert [step.consumed_items for step in steps] == [0, 2]
    assert [step.processed_target_transitions for step in steps] == [8, 8]
    for step, start in zip(steps, (0, 2), strict=True):
        assert step.loss.item() == pytest.approx(loss(full.scaled_delta[:, start : start + 4], batch.scaled_target[:, start : start + 4]).item())

    monkeypatch.setattr(adapter_module, "_TEACHER_FORCED_CHUNK", 4)
    views = adapter.evaluation_step(model, batch)
    autonomous = rollout_module.self_fed_rollout(model, batch, adapter.tensorizer, model_kind="rno")
    assert views.sample_count == 2
    assert views.processed_target_transitions == 12
    # Selection rolls out the whole trajectory from its start; the one-step view carries the state across chunks.
    assert torch.allclose(views.physical_prediction, autonomous.physical_prediction)
    assert torch.allclose(views.one_step_physical_prediction, full.physical_prediction)
    assert not torch.allclose(views.physical_prediction, views.one_step_physical_prediction)


def _window_loss_and_gradient(carried: float, weight: float, cumulative_inputs: tuple[float, ...]) -> tuple[float, float]:
    """Return the unit-target MSE of the states ``carried + weight * cumulative_input`` and its derivative in ``weight``."""
    residuals = [carried + weight * value - 1.0 for value in cumulative_inputs]
    loss = sum(residual**2 for residual in residuals) / len(residuals)
    gradient = 2.0 * sum(value * residual for value, residual in zip(cumulative_inputs, residuals, strict=True)) / len(residuals)
    return loss, gradient


def test_rno_stage_a_windows_consumed_like_the_training_loop_stop_gradients_at_the_boundary() -> None:
    """Step the optimizer between lazily yielded windows; the carried state keeps no graph of the previous window."""
    adapter = _rno_adapter("a")
    model = _StateRNO()
    optimizer = SGD(model.parameters(), lr=0.01)
    weights: list[float] = []
    losses: list[float] = []
    gradients: list[float] = []
    for step in adapter.training_steps(model, _trajectory_batch(), torch.nn.MSELoss()):
        weights.append(model.weight.item())
        losses.append(step.loss.item())
        step.loss.backward()
        gradient = model.weight.grad
        assert gradient is not None
        gradients.append(gradient.item())
        optimizer.step()
        optimizer.zero_grad()

    # Reference inputs are 0..5, so the states of the first window are w * (0, 1, 3, 6) and the carried state after two
    # steps is the constant w_0 * (0 + 1). The second window adds w_1 * (2, 5, 9, 14) with the weight the first step
    # produced; its gradient is the direct term only, whereas an untruncated carry would add the first window's d(state)/dw = 1.
    first_loss, first_gradient = _window_loss_and_gradient(0.0, weights[0], (0.0, 1.0, 3.0, 6.0))
    second_loss, second_gradient = _window_loss_and_gradient(weights[0], weights[1], (2.0, 5.0, 9.0, 14.0))
    assert len(losses) == 2
    assert weights[1] == pytest.approx(weights[0] - 0.01 * first_gradient)
    assert losses == pytest.approx([first_loss, second_loss], rel=1e-5)
    assert gradients == pytest.approx([first_gradient, second_gradient], rel=1e-5)


def test_rno_stage_b_starts_each_origin_from_its_reference_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Origins at the trajectory start keep the initial state; later origins receive the warm state."""
    from src.learning.transient import learning_transient_adapter as adapter_module  # noqa: PLC0415

    adapter = _rno_adapter("b")
    calls: list[tuple[int, bool]] = []
    real_self_fed = adapter_module.self_fed_rollout

    def recording_self_fed(model: Any, batch: TransientBatch, tensorizer: Any, *, model_kind: str, initial_hidden: object | None = None) -> Any:
        calls.append((batch.batch_size, initial_hidden is not None))
        return real_self_fed(model, batch, tensorizer, model_kind=model_kind, initial_hidden=initial_hidden)

    monkeypatch.setattr(adapter_module, "self_fed_rollout", recording_self_fed)
    monkeypatch.setattr(RolloutCurriculumState, "select", lambda _self, **_kwargs: (2, torch.tensor([0, 1])))
    steps = list(adapter.training_steps(_StateRNO(), _trajectory_batch(), torch.nn.MSELoss()))

    assert len(steps) == 2
    assert all(torch.isfinite(step.loss) for step in steps)
    # First window: one sample still at the initial state and one warm sample; second window: both warm.
    assert calls == [(1, False), (1, True), (2, True)]
    assert [step.forward_transitions for step in steps] == [4 + 8, 4 + 8]


def _reference_state(increments: list[float], position: int, weight: float) -> float | None:
    """Return the ``_StateRNO`` state after the teacher-forced reference inputs before ``position`` (``None`` at the start)."""
    if position == 0:
        return None
    return weight * sum(sum(increments[:step]) for step in range(position))


def _stage_b_start_states(monkeypatch: pytest.MonkeyPatch, increments: torch.Tensor, origins: torch.Tensor) -> list[dict[int, torch.Tensor | None]]:
    """Return per window the hidden state each numbered sample's self-fed rollout starts from (``None``: initial state)."""
    from src.learning.transient import learning_transient_adapter as adapter_module  # noqa: PLC0415
    from src.learning.transient import learning_transient_rollout as rollout_module  # noqa: PLC0415

    adapter = _rno_adapter("b")
    started: dict[int, torch.Tensor | None] = {}

    def recording_self_fed(model: Any, batch: TransientBatch, tensorizer: Any, *, model_kind: str, initial_hidden: Any = None) -> Any:
        for row, sample in enumerate(batch.scalars[:, 0].long().tolist()):
            started[sample] = None if initial_hidden is None else initial_hidden[0][row, :, 0, 0].clone()
        return rollout_module.self_fed_rollout(model, batch, tensorizer, model_kind=model_kind, initial_hidden=initial_hidden)

    monkeypatch.setattr(adapter_module, "self_fed_rollout", recording_self_fed)
    monkeypatch.setattr(RolloutCurriculumState, "select", lambda _self, **_kwargs: (2, origins))
    windows: list[dict[int, torch.Tensor | None]] = []
    batch = _trajectory_batch(increments.shape[0], increments.shape[1], increments)
    for _step in adapter.training_steps(_StateRNO(), batch, torch.nn.MSELoss()):
        windows.append(dict(started))
        started.clear()
    return windows


def test_rno_stage_b_origin_state_is_the_reference_prefix_state_and_ignores_later_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every sample at its own origin from the state of the reference before it, never from later reference."""
    weight = 0.5
    increments = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0], [3.0, 5.0, 1.0, 6.0, 2.0, 4.0]])
    origins = torch.tensor([2, 0, 1])
    # Change every reference increment from each sample's first-window origin on; its state at that origin cannot depend on them.
    later = (torch.arange(increments.shape[1])[None, :] >= origins[:, None]).float()
    changed = increments + 7.0 * later

    windows = _stage_b_start_states(monkeypatch, increments, origins)
    changed_windows = _stage_b_start_states(monkeypatch, changed, origins)

    # Windows start at 0 and 2; a sample's state at origin o of the window starting at s follows the reference before s + o.
    for run, table in ((windows, increments), (changed_windows, changed)):
        assert [set(window) for window in run] == [{0, 1, 2}] * 2
        for start, window in zip((0, 2), run, strict=True):
            for sample, origin in enumerate(origins.tolist()):
                expected = _reference_state(table[sample].tolist(), start + origin, weight)
                actual = window[sample]
                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None
                    assert torch.allclose(actual, torch.full((4,), expected))
    for sample in range(3):
        before, after = windows[0][sample], changed_windows[0][sample]
        assert (before is None and after is None) or (before is not None and after is not None and torch.equal(before, after))
