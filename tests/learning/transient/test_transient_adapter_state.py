"""Protect transient adapter evaluation preparation and transactional restore.

Responsibilities:
  - Check evaluation preparation, nested restore rollback, work accounting, and schema admission
  - Check that every stage selects checkpoints on whole autonomous trajectories, and that a finite rollout whose
    standardized final state overflows is an incomplete case
  - Check that the teacher-forced state loss equals the group-weighted Train-scaled increment error
  - Check RNO training on consecutive trajectory windows with a rest window, truncated gradients, warm Stage-B starts,
    and the carried one-step view
  - Check that a Stage-B case shorter than the window forms one window with a capped horizon, and that a
    Stage-B update steps alike for any memory split
  - Check that the windows of one RNO Stage-B trajectory weigh equally whatever their drawn horizons

Design principles:
  - Curriculum RNG and budget progress stay unchanged by evaluation preparation

This module does NOT:
  - Select curriculum lengths or execute training campaigns
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from src import domain, learning
from src.datasets.runtime.dataset_runtime_units import UnitGroup, UnitSampler
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
        input_profile="canonical_physics_complete",
    )
    controller = MatchedComputeController(
        arm="B",
        stage="stage_b_self_fed",
        config_digest=_digest("config"),
        teacher_handoff=handoff,
    )
    return TransientTrainingSpec(
        stage="stage_b_self_fed",
        model_kind="fno",
        controller=controller,
        curriculum=RolloutCurriculum(),
        curriculum_seed=19,
    )


def _stage_a(adapter: Any, controller: MatchedComputeController, **changes: Any) -> None:
    """Make a fake adapter teacher-forced Stage A, which has no curriculum, seed or curriculum state."""
    adapter.spec = replace(adapter.spec, stage="stage_a_teacher_forcing", controller=controller, curriculum=None, curriculum_seed=None, **changes)
    adapter.curriculum_state = None


def _batch(length: int = 2) -> TransientBatch:
    state = torch.zeros(1, 4, 1, 1)
    next_state = torch.zeros(1, length, 4, 1, 1)
    sequence = torch.zeros(1, length, 5, 1, 1)
    return TransientBatch(
        state=state,
        next_state=next_state,
        static=torch.zeros(1, 7, 1, 1),
        boundary=torch.zeros(1, length, 9),
        scalars=torch.zeros(1, 8),
        t_n=torch.zeros(1, length),
        t_n_plus_1=torch.ones(1, length),
        dt=torch.ones(1, length),
        step_input=sequence[:, 0],
        sequence_input=sequence,
    )


def _adapter_with_fake_tensorizer() -> TransientTrainingAdapter:
    adapter: Any = object.__new__(TransientTrainingAdapter)
    spec = _spec()
    # A Stage B spec draws rollout horizons, so it carries a curriculum and its seed.
    assert spec.curriculum is not None
    assert spec.curriculum_seed is not None
    adapter.spec = spec
    adapter.curriculum_state = RolloutCurriculumState.create(spec.curriculum, seed=spec.curriculum_seed)
    adapter._epoch_index = 0
    adapter._optimizer_events = 0
    adapter.tensorizer = SimpleNamespace(
        scaling=SimpleNamespace(device=torch.device("cpu"), encode_state=lambda value: value),
        tensorize=lambda raw: raw["batch"],
    )
    return adapter


def _group(keys: tuple[int, ...]) -> UnitGroup:
    """Return one update group of equally weighted units with the given transition counts, loaded as one microbatch."""
    positions = tuple(range(len(keys)))
    return UnitGroup(items=positions, keys=keys, weights=(1.0,) * len(keys), cases=positions, microbatches=(positions,))


class _UnitMSE(torch.nn.Module):
    """Return the mean squared state error of every unit as its total."""

    def unit_components(self, prediction: torch.Tensor, *, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``total`` per unit of ``[U, L, C, H, W]`` states."""
        return {"total": (prediction - y).square().flatten(1).mean(dim=1)}


def test_evaluation_preparation_does_not_consume_curriculum_rng() -> None:
    """Keep evaluation trajectories intact without consuming curriculum randomness."""
    adapter = _adapter_with_fake_tensorizer()
    curriculum = adapter.curriculum_state
    assert curriculum is not None
    raw = {"batch": _batch()}
    before = curriculum.state_dict()
    prepared = adapter.prepare_batch(raw, device=torch.device("cpu"), training=False)
    assert prepared is raw["batch"]
    assert curriculum.state_dict() == before


def test_adapter_restore_is_transactional_when_nested_curriculum_is_invalid() -> None:
    """Leave all live adapter continuation evidence unchanged after nested failure."""
    adapter = _adapter_with_fake_tensorizer()
    before = copy.deepcopy(adapter.state_dict())
    invalid = copy.deepcopy(before)
    invalid["curriculum"]["completed_milestones"] = ["invalid"]
    with pytest.raises(ValueError, match="completed_milestones"):
        adapter.load_state_dict(invalid)
    assert adapter.state_dict() == before


def test_validation_work_is_persisted_but_excluded_from_curriculum_progress() -> None:
    """Keep completed validation time as secondary evidence; the curriculum never advances by it."""
    adapter = _adapter_with_fake_tensorizer()
    curriculum = adapter.curriculum_state
    assert curriculum is not None
    before = curriculum.state_dict()
    adapter.record_validation_work(1.25)
    assert adapter.spec.controller.validation_seconds == pytest.approx(1.25)
    assert curriculum.state_dict() == before
    assert adapter.telemetry_state()["validation_seconds"] == pytest.approx(1.25)
    assert adapter.state_dict()["controller"]["validation_seconds"] == pytest.approx(1.25)


def test_a_stage_a_adapter_has_no_curriculum_state_telemetry_or_saved_entry() -> None:
    """Teacher-forced Stage A draws no horizon: no curriculum, seed, state, telemetry or saved entry, and a saved one is refused."""
    adapter = _adapter_with_fake_tensorizer()
    stage_b_curriculum = adapter.state_dict()["curriculum"]
    controller = MatchedComputeController(arm="A0", stage="stage_a_teacher_forcing", config_digest=_digest("a0-state"))
    _stage_a(adapter, controller)
    state = adapter.state_dict()

    assert "curriculum" not in state
    assert not any(key.startswith("curriculum") or key == "selected_rollout_horizon" for key in adapter.telemetry_state())
    adapter.load_state_dict(copy.deepcopy(state))
    with pytest.raises(ValueError, match="conflicts with the configured semantic specification"):
        adapter.load_state_dict({**copy.deepcopy(state), "curriculum": stage_b_curriculum})
    with pytest.raises(ValueError, match="Exactly Stage B"):
        TransientTrainingSpec(
            stage="stage_a_teacher_forcing", model_kind="fno", controller=controller, curriculum=RolloutCurriculum(), curriculum_seed=19
        )


def test_every_stage_selects_checkpoints_on_the_whole_autonomous_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate teacher-forced Stage A by the self-fed rollout of the whole trajectory, as Stage B."""
    from src.learning.transient import learning_transient_adapter as adapter_module  # noqa: PLC0415

    adapter = _adapter_with_fake_tensorizer()
    _stage_a(adapter, MatchedComputeController(arm="A0", stage="stage_a_teacher_forcing", config_digest=_digest("a0")))
    batch = _batch(8)
    prepared = adapter.prepare_batch({"batch": batch}, device=torch.device("cpu"), training=True)
    assert prepared is batch
    self_fed_lengths: list[int] = []
    teacher_windows: list[tuple[float, int]] = []

    def self_fed(_model: Any, _tensorizer: Any, *, boundary: torch.Tensor, model_kind: str, **_inputs: Any) -> Any:
        assert model_kind == "fno"
        self_fed_lengths.append(int(boundary.shape[1]))
        return SimpleNamespace(physical_prediction=batch.next_state)

    def teacher(
        _model: Any,
        window: TransientBatch,
        _tensorizer: Any,
        *,
        model_kind: str,
        initial_hidden: object | None,
        capture: tuple[int, ...],
        require_finite: bool,
    ) -> Any:
        assert (model_kind, initial_hidden, capture, require_finite) == ("fno", None, (), False)
        teacher_windows.append((float(window.t_n[0, 0]), window.rollout_length))
        return SimpleNamespace(physical_prediction=window.t_n[:, :, None, None, None].expand(-1, -1, 4, 1, 1), captured_hidden={})

    monkeypatch.setattr(adapter_module, "teacher_forced_rollout", teacher)
    monkeypatch.setattr(adapter_module, "autonomous_rollout", self_fed)
    monkeypatch.setattr(adapter_module, "_TEACHER_FORCED_CHUNK", 3)
    batch.t_n.copy_(torch.arange(8, dtype=torch.float32))
    views = adapter.evaluation_step(torch.nn.Identity(), batch)

    assert self_fed_lengths == [8]
    assert views.processed_target_transitions == 8
    assert views.physical_prediction.shape[1] == 8
    # The one-step view covers every transition in memory-bounded chunks, in order.
    assert teacher_windows == [(0.0, 3), (3.0, 3), (6.0, 2)]
    assert torch.equal(views.one_step_physical_prediction[0, :, 0, 0, 0], torch.arange(8, dtype=torch.float32))
    assert views.one_step_physical_prediction.shape == views.physical_target.shape


_STATE_MEAN = torch.tensor((300.0, 0.5, 0.1, 0.2), dtype=torch.float64).view(4, 1, 1)
_STATE_SCALE = torch.tensor((2.0, 0.5, 4.0, 8.0), dtype=torch.float64).view(4, 1, 1)


@dataclass(frozen=True)
class _StateScales:
    """Expose Train state channel names and scales as the scaling's state group."""

    names: tuple[str, ...]
    scale: torch.Tensor

    def group(self, name: str) -> _StateScales:
        """Return the state group."""
        assert name == "state"
        return self


class _TanhOperator(torch.nn.Module):
    """Predict one bounded, input-dependent scaled increment per state channel."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 0.3 * torch.tanh(value[:, :4]) + 0.1


def test_stage_a_state_loss_is_the_group_weighted_train_scaled_increment_mse() -> None:
    """
    Train one teacher-forced step with the run's transient loss on Train-scaled reconstructed states.

    Each prediction adds its increment to the stored current state, so the Train-scaled state error
    equals the predicted scaled increment minus the stored increment over the Train scale; the
    centring cancels and the loss is the group-weighted MSE of those increment errors.
    """
    task = domain.tasks.registry.get_task("transient_drying")
    loss = learning.losses.factory.build_training_loss(
        {"task": task.id, "loss": {"data": {"kind": "mse", "weight": 1.0}, "physics": {"enabled": False}}}, device=torch.device("cpu")
    )
    assert isinstance(loss, learning.losses.transient.TransientStateLoss)
    loss.set_scaling(_StateScales(task.metric_names, _STATE_SCALE.flatten()))
    adapter: Any = _adapter_with_fake_tensorizer()
    _stage_a(adapter, MatchedComputeController(arm="A0", stage="stage_a_teacher_forcing", config_digest=_digest("a0-loss")))
    adapter.tensorizer = SimpleNamespace(
        scaling=SimpleNamespace(device=torch.device("cpu"), encode_state=lambda value: (value - _STATE_MEAN) / _STATE_SCALE),
        reconstruct_next_state=lambda state, delta: state + delta * _STATE_SCALE,
        tensorize=lambda raw: raw["batch"],
    )
    stored = _STATE_MEAN + _STATE_SCALE * torch.arange(2 * 4 * 4 * 2 * 3, dtype=torch.float64).view(2, 4, 4, 2, 3).sin()
    batch = _stored_batch(stored)
    model = _TanhOperator()
    adapter.begin_group(_group((3, 3)), progress=0.0)
    (share,) = adapter.unit_losses(model, {"batch": batch}, loss, positions=(0, 1))
    scaled_delta = torch.stack([model(batch.sequence_input[:, index]) for index in range(batch.rollout_length)], dim=1)
    increment_error = scaled_delta - (stored[:, 1:] - stored[:, :-1]) / _STATE_SCALE
    field_mse = increment_error.square().movedim(2, 1).reshape(2, 4, -1).mean(dim=2)
    expected = {term.id: term.weight * field_mse[:, task.metric_names.index(term.fields[0])] for term in task.error_terms}
    assert torch.allclose(share.components["total"], torch.stack(tuple(expected.values())).sum(dim=0), rtol=1e-12)
    for name, value in expected.items():
        assert torch.allclose(share.components[f"data_{name}"], value, rtol=1e-12)
    assert (share.supervised_transitions, share.forward_transitions) == (6, 6)


class _StateRNO(torch.nn.Module):
    """Accumulate weighted inputs in a one-layer tensor state; each prediction is the new state."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, value: torch.Tensor, *, init_hidden_states: object | None, return_hidden_states: bool, keep_states_padded: bool) -> Any:
        assert return_hidden_states is True
        assert keep_states_padded is True
        step = value[:, 0, :4] * self.weight
        state = step if init_hidden_states is None else init_hidden_states[0] + step  # type: ignore[index]
        return state, [state]


def _stored_batch(stored: torch.Tensor) -> TransientBatch:
    """Build reference trajectories over stored states ``[B, L + 1, 4, H, W]``; teacher-forced inputs are the stored current states."""
    batch_size, length = stored.shape[0], stored.shape[1] - 1
    reference = stored[:, :-1]
    sequence = torch.cat((reference, torch.ones_like(reference[:, :, :1])), dim=2)
    scalars = torch.zeros(batch_size, 8, dtype=stored.dtype)
    scalars[:, 0] = torch.arange(batch_size)
    return TransientBatch(
        state=stored[:, 0],
        next_state=stored[:, 1:],
        static=torch.zeros(batch_size, 7, 1, 1, dtype=stored.dtype),
        boundary=torch.zeros(batch_size, length, 9, dtype=stored.dtype),
        scalars=scalars,
        t_n=torch.arange(length, dtype=stored.dtype).expand(batch_size, -1),
        t_n_plus_1=torch.arange(1, length + 1, dtype=stored.dtype).expand(batch_size, -1),
        dt=torch.ones(batch_size, length, dtype=stored.dtype),
        step_input=sequence[:, 0],
        sequence_input=sequence,
    )


def _trajectory_batch(batch_size: int = 2, length: int = 6, increments: torch.Tensor | None = None) -> TransientBatch:
    """Build reference trajectories from zero with the given (default unit) increments; scalar 0 numbers the samples."""
    steps = torch.ones(batch_size, length, 4, 1, 1) if increments is None else increments[:, :, None, None, None].expand(-1, -1, 4, 1, 1)
    state = torch.zeros(batch_size, 1, 4, 1, 1)
    return _stored_batch(torch.cat((state, steps.cumsum(dim=1)), dim=1))


def _rno_adapter(stage: str) -> TransientTrainingAdapter:
    """Return one RNO adapter on four-step trajectory windows with an identity tensorizer."""
    adapter: Any = _adapter_with_fake_tensorizer()
    if stage == "a":
        _stage_a(
            adapter,
            MatchedComputeController(arm="A0", stage="stage_a_teacher_forcing", config_digest=_digest("rno-a0")),
            model_kind="rno",
            trajectory_window=4,
        )
    else:
        adapter.spec = TransientTrainingSpec(
            stage="stage_b_self_fed",
            model_kind="rno",
            controller=replace(adapter.spec.controller, config_digest=_digest("rno-b")),
            curriculum=RolloutCurriculum(lengths=(2,), milestone_fractions=(0.0,)),
            curriculum_seed=19,
            trajectory_window=4,
        )
        adapter.curriculum_state = RolloutCurriculumState.create(RolloutCurriculum(lengths=(2,), milestone_fractions=(0.0,)), seed=19)
    adapter.tensorizer = SimpleNamespace(
        scaling=SimpleNamespace(device=torch.device("cpu"), encode_state=lambda value: value, encode_delta=lambda value: value),
        reconstruct_next_state=lambda state, delta: state + delta,
        reconstruct_next_state_diagnostic=lambda state, delta: (state + delta, delta),
        assemble_step=lambda state, *_args: torch.cat((state, torch.ones_like(state[:, :1])), dim=1),
        tensorize=lambda raw: raw["batch"],
    )
    return adapter


def test_rno_stage_a_trains_consecutive_windows_with_the_carried_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Cut a six-step trajectory into a four-step window and a two-step rest window.

    The second window's carried state equals one teacher-forced pass from hour 0, and
    each window's share is its transition fraction of the unit's mean loss.
    """
    from src.learning.transient import learning_transient_adapter as adapter_module  # noqa: PLC0415
    from src.learning.transient import learning_transient_rollout as rollout_module  # noqa: PLC0415

    adapter = _rno_adapter("a")
    model = _StateRNO()
    batch = _trajectory_batch()
    adapter.begin_group(_group((6, 6)), progress=0.0)
    shares = list(adapter.unit_losses(model, {"batch": batch}, _UnitMSE(), positions=(0, 1)))
    full = rollout_module.teacher_forced_rollout(model, batch, adapter.tensorizer, model_kind="rno")

    assert [share.supervised_transitions for share in shares] == [8, 4]
    for share, (start, length) in zip(shares, ((0, 4), (4, 2)), strict=True):
        window_error = full.normalized_prediction[:, start : start + length] - batch.next_state[:, start : start + length]
        assert torch.allclose(share.components["total"], length / 6 * window_error.square().flatten(1).mean(dim=1))

    monkeypatch.setattr(adapter_module, "_TEACHER_FORCED_CHUNK", 4)
    views = adapter.evaluation_step(model, batch)
    autonomous = rollout_module.self_fed_rollout(model, batch, adapter.tensorizer, model_kind="rno")
    assert views.sample_count == 2
    assert views.case_complete.tolist() == [True, True]
    assert views.processed_target_transitions == 12
    # Selection rolls out the whole trajectory from its start; the one-step view carries the state across chunks.
    assert torch.allclose(views.physical_prediction, autonomous.physical_prediction)
    assert torch.allclose(views.one_step_physical_prediction, full.physical_prediction)
    assert not torch.allclose(views.physical_prediction, views.one_step_physical_prediction)


class _FlaggedIncrement(torch.nn.Module):
    """Predict the scaled increment 2e38 in every state channel of the rows whose fifth input channel is 1, none where it is 0."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 2.0e38 * value[:, 4:5].expand(-1, 4, -1, -1)


def test_a_finite_rollout_whose_standardized_final_state_overflows_is_an_incomplete_case() -> None:
    """
    With a Train scale of 1e-7, two increments of 2e38 leave case 0 at the finite states 2e31 and 4e31.

    The first state standardizes to 2e38 and is fed back; the final one overflows float32 when standardized,
    so case 0 is incomplete although its physical rollout is finite, and case 1, without increments, is complete.
    """
    scale = 1.0e-7
    adapter: Any = _adapter_with_fake_tensorizer()
    _stage_a(adapter, MatchedComputeController(arm="A0", stage="stage_a_teacher_forcing", config_digest=_digest("overflow")))
    # The autonomous input flags case 0 (sample number 0) only; the teacher-forced inputs flag every row.
    adapter.tensorizer = SimpleNamespace(
        scaling=SimpleNamespace(device=torch.device("cpu"), encode_state=lambda value: value / scale),
        assemble_step=lambda state, _static, _boundary, scalars, _time: torch.cat((state / scale, (1.0 - scalars[:, :1]).view(-1, 1, 1, 1)), dim=1),
        reconstruct_next_state_diagnostic=lambda state, delta: (state + delta * scale, delta * scale),
        tensorize=lambda raw: raw["batch"],
    )

    views = adapter.evaluation_step(_FlaggedIncrement(), _trajectory_batch(length=2, increments=torch.zeros(2, 2)))

    assert views.physical_prediction.isfinite().all()
    assert views.physical_prediction[0, :, 0, 0, 0].tolist() == pytest.approx([2.0e31, 4.0e31], rel=1e-6)
    assert views.case_complete.tolist() == [False, True]


def _window_loss_and_gradient(carried: float, weight: float, cumulative_inputs: tuple[float, ...]) -> tuple[float, float]:
    """
    Return the unit-increment MSE of the outputs ``carried + weight * cumulative_input`` and its derivative in ``weight``.

    Under teacher forcing with identity scaling, the state error of each transition is this increment error.
    """
    residuals = [carried + weight * value - 1.0 for value in cumulative_inputs]
    loss = sum(residual**2 for residual in residuals) / len(residuals)
    gradient = 2.0 * sum(value * residual for value, residual in zip(cumulative_inputs, residuals, strict=True)) / len(residuals)
    return loss, gradient


def test_rno_stage_a_windows_consumed_like_the_training_loop_stop_gradients_at_the_boundary() -> None:
    """Backpropagate each lazily yielded window before the next; the carried state keeps no graph of the previous window."""
    adapter = _rno_adapter("a")
    model = _StateRNO()
    weight = model.weight.item()
    losses: list[float] = []
    gradients: list[float] = []
    adapter.begin_group(_group((6,)), progress=0.0)
    for share in adapter.unit_losses(model, {"batch": _trajectory_batch(batch_size=1)}, _UnitMSE(), positions=(0,)):
        total = share.components["total"].sum()
        losses.append(total.item())
        total.backward()
        gradient = model.weight.grad
        assert gradient is not None
        gradients.append(gradient.item())
        model.weight.grad = None

    # Reference inputs are 0..5, so the states of the first window are w * (0, 1, 3, 6) and the carried state after four
    # steps is the constant w * (0 + 1 + 2 + 3). The rest window adds w * (4, 9); its gradient is the direct term only,
    # whereas an untruncated carry would add the first window's d(state)/dw = 6. Shares weigh each window by 4/6 and 2/6.
    first_loss, first_gradient = _window_loss_and_gradient(0.0, weight, (0.0, 1.0, 3.0, 6.0))
    second_loss, second_gradient = _window_loss_and_gradient(6.0 * weight, weight, (4.0, 9.0))
    assert losses == pytest.approx([4 / 6 * first_loss, 2 / 6 * second_loss], rel=1e-5)
    assert gradients == pytest.approx([4 / 6 * first_gradient, 2 / 6 * second_gradient], rel=1e-5)


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
    # One draw per window of each unit, in unit then window order: unit 0 starts at origin 0, unit 1 at origin 1.
    monkeypatch.setattr(RolloutCurriculumState, "draw", lambda _self, **_kwargs: ((2, 0), (2, 0), (2, 1), (2, 1)))
    adapter.begin_group(_group((6, 6)), progress=0.0)
    shares = list(adapter.unit_losses(_StateRNO(), {"batch": _trajectory_batch()}, _UnitMSE(), positions=(0, 1)))

    assert len(shares) == 2
    assert all(torch.isfinite(share.components["total"]).all() for share in shares)
    # First window: one sample still at the initial state and one warm sample; second window: both warm.
    assert calls == [(1, False), (1, True), (2, True)]
    assert [share.forward_transitions for share in shares] == [4 + 8, 4 + 8]
    assert [share.supervised_transitions for share in shares] == [4, 4]


def _collate_units(items: list[dict[str, TransientBatch]]) -> dict[str, TransientBatch]:
    """Stack single-unit transient batches into one microbatch."""
    parts = [item["batch"] for item in items]
    return {"batch": TransientBatch(**{field.name: torch.cat([getattr(part, field.name) for part in parts]) for field in fields(TransientBatch)})}


def _rno_stage_b_update(memory_batch: int) -> float:
    """Train one Stage-B RNO update of two trajectories split into microbatches of ``memory_batch``; return the new weight."""
    batch = _trajectory_batch(2, 6, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]]))
    items: Any = [  # A list serves as the map-style dataset of the two trajectories.
        {"batch": TransientBatch(**{field.name: getattr(batch, field.name)[row : row + 1] for field in fields(TransientBatch)})} for row in range(2)
    ]
    sampler = UnitSampler(
        unit_kind="trajectory", cases=(0, 1), keys=(6, 6), units_per_update=2, memory_batch=memory_batch, seed=0, full_set_transitions=12
    )
    sampler.begin_epoch(0)
    adapter: Any = _rno_adapter("b")
    model = _StateRNO()
    loader = DataLoader(items, batch_sampler=sampler, collate_fn=_collate_units)
    values = learning.training.loop.train_update_groups(
        model,
        loader,
        SGD(model.parameters(), lr=0.01),
        _UnitMSE(),
        torch.device("cpu"),
        adapter,
        budget=learning.training.budget.StageBudget(unit="supervised_transitions", total=12),
        adapter=adapter,
    )
    assert values["train/updates"] == 1.0
    return model.weight.item()


class _UnitConstant(torch.nn.Module):
    """Return a loss of one for every unit, so each yielded share is its window weight."""

    def unit_components(self, prediction: torch.Tensor, *, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``total`` of one per unit of ``[U, L, C, H, W]`` states."""
        return {"total": torch.ones(len(prediction), dtype=prediction.dtype) + 0.0 * (prediction - y).flatten(1).sum(dim=1)}


def test_rno_stage_b_weights_every_window_equally_whatever_its_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two windows of one trajectory with drawn horizons 2 and 32 each carry half of the unit's loss."""
    adapter: Any = _rno_adapter("b")
    adapter.spec = replace(adapter.spec, trajectory_window=32)
    monkeypatch.setattr(RolloutCurriculumState, "draw", lambda _self, **_kwargs: ((2, 0), (32, 0)))
    adapter.begin_group(_group((64,)), progress=0.0)
    assert adapter._group_windows == (((0, 32), (32, 32)),)

    shares = list(adapter.unit_losses(_StateRNO(), {"batch": _trajectory_batch(batch_size=1, length=64)}, _UnitConstant(), positions=(0,)))

    assert [share.supervised_transitions for share in shares] == [2, 32]
    assert [share.components["total"].item() for share in shares] == pytest.approx([0.5, 0.5])


def test_rno_stage_b_update_does_not_depend_on_the_memory_split() -> None:
    """Horizons and origins are drawn per window before the split, so 1 x 2 and 2 x 1 microbatches step alike."""
    assert _rno_stage_b_update(1) == pytest.approx(_rno_stage_b_update(2), rel=1e-12)
    assert _rno_stage_b_update(1) != 0.5


def test_stage_b_short_case_forms_one_window_with_the_capped_horizon() -> None:
    """A one-step case under a four-step window and a two-step horizon trains one window whose horizon is capped at one."""
    adapter: Any = _rno_adapter("b")
    adapter.begin_group(_group((6, 1)), progress=0.0)
    assert adapter._group_windows == (((0, 4), (2, 4)), ((0, 1),))
    assert adapter._group_draws[1] == ((1, 0),)
    assert all(horizon == 2 for horizon, _ in adapter._group_draws[0])

    short = _trajectory_batch(batch_size=1, length=1)
    (share,) = adapter.unit_losses(_StateRNO(), {"batch": short}, _UnitMSE(), positions=(1,))
    assert share.supervised_transitions == 1
    assert torch.isfinite(share.components["total"]).all()


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
    draws = tuple((2, origin) for origin in origins.tolist() for _window in range(2))
    monkeypatch.setattr(RolloutCurriculumState, "draw", lambda _self, **_kwargs: draws)
    windows: list[dict[int, torch.Tensor | None]] = []
    batch = _trajectory_batch(increments.shape[0], increments.shape[1], increments)
    units = tuple(range(batch.batch_size))
    adapter.begin_group(_group((batch.rollout_length,) * batch.batch_size), progress=0.0)
    for _share in adapter.unit_losses(_StateRNO(), {"batch": batch}, _UnitMSE(), positions=units):
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
