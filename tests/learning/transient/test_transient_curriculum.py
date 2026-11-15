"""Protect the transient continuous curriculum and the matched controller evidence.

Responsibilities:
  - Check per-window horizon draws and exact resumed RNG
  - Check that origin strata reach the late part of a window for every group size and that a short
    window caps its horizon
  - Check the controller's work accounting, its best-within-budget record and its strict identity

Design principles:
  - The stage budget owns progress and validation points; the controller only accounts work evidence

This module does NOT:
  - Count budgets or validation points; the stage schedule tests own them
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from src.learning.transient.learning_transient_curriculum import (
    MatchedComputeController,
    RolloutCurriculum,
    RolloutCurriculumState,
    TeacherHandoffIdentity,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _handoff() -> TeacherHandoffIdentity:
    return TeacherHandoffIdentity(
        source_run_name="synthetic-a0",
        source_checkpoint_sha256=_digest("checkpoint"),
        source_scaling_sha256=_digest("scaling"),
        task_contract_sha256=_digest("task"),
        tensorizer_sha256=_digest("tensorizer"),
        model_kind="fno",
        input_profile="canonical_physics_complete",
    )


def test_curriculum_retains_shorter_horizons_and_resumes_exact_draws() -> None:
    """Use compute progress and private RNG state for reproducible valid windows."""
    curriculum = RolloutCurriculum()
    state = RolloutCurriculumState.create(curriculum, seed=17)
    first = state.draw(progress=0.85, available_lengths=[32] * 9)
    assert {horizon for horizon, _ in first} <= set(curriculum.eligible_lengths(0.85))
    assert all(0 <= origin <= 32 - horizon for horizon, origin in first)
    assert len({horizon for horizon, _ in first}) > 1
    assert state.draw_index == 9
    saved = state.state_dict()
    expected = state.draw(progress=0.85, available_lengths=[32] * 6)
    restored = RolloutCurriculumState.create(curriculum, seed=999)
    restored.load_state_dict(saved)
    assert restored.draw(progress=0.85, available_lengths=[32] * 6) == expected
    assert 2 in curriculum.eligible_lengths(0.85)


def _work(**overrides: Any) -> dict[str, Any]:
    """Return one completed optimizer group with test-owned defaults."""
    work: dict[str, Any] = {
        "microbatches": 1,
        "processed_target_transitions": 2,
        "forward_transitions": 2,
    }
    work.update(overrides)
    return work


@pytest.mark.parametrize("windows", [1, 2])
def test_origin_strata_rotate_so_small_groups_reach_every_third_of_the_window(windows: int) -> None:
    """
    Draw origins for groups smaller than the three temporal strata.

    Groups of one or two windows must still start rollouts in the early, middle
    and late third of a window, in equal numbers over three consecutive draws.
    """
    state = RolloutCurriculumState.create(RolloutCurriculum(), seed=5)
    available_length = 32
    horizon = 2
    maximum = available_length - horizon
    strata = ((0, maximum // 3), (maximum // 3 + 1, (2 * maximum) // 3), ((2 * maximum) // 3 + 1, maximum))
    counts = [0, 0, 0]
    for _draw in range(3):
        # The first curriculum stage admits only the shortest horizon.
        for drawn_horizon, origin in state.draw(progress=0.0, available_lengths=[available_length] * windows):
            assert drawn_horizon == horizon
            counts[next(index for index, (low, high) in enumerate(strata) if low <= origin <= high)] += 1

    assert counts == [windows] * 3


def test_a_window_shorter_than_the_drawn_horizon_caps_it_at_its_length() -> None:
    """A short case's single window rolls out over its whole length from origin 0."""
    state = RolloutCurriculumState.create(RolloutCurriculum(), seed=3)
    draws = state.draw(progress=0.85, available_lengths=[1, 32])
    assert draws[0] == (1, 0)
    assert draws[1][0] >= 2


def test_controller_accumulates_committed_work_and_the_best_within_budget() -> None:
    """Sum every committed group's work and keep the best validation metric."""
    controller = MatchedComputeController(arm="B", stage="stage_b_self_fed", config_digest=_digest("work"), teacher_handoff=_handoff())
    controller.record_completed_work(**_work(microbatches=2))
    controller.record_completed_work(**_work())
    controller.record_within_budget_evaluation(0.5, epoch_index=0)
    controller.record_within_budget_evaluation(0.25, epoch_index=1)
    controller.record_within_budget_evaluation(0.75, epoch_index=2)

    assert (controller.post_handoff_optimizer_steps, controller.microbatches, controller.processed_target_transitions) == (2, 3, 4)
    assert (controller.best_within_budget_metric, controller.best_within_budget_epoch) == (0.25, 1)


def test_controller_state_rejects_semantic_drift_foreign_keys_and_a_misplaced_receipt() -> None:
    """Require immutable controller identity and the exact key set; only A1 binds B's receipt."""
    controller = MatchedComputeController(
        arm="A1",
        stage="stage_a_teacher_forcing",
        config_digest=_digest("state"),
        teacher_handoff=_handoff(),
        matched_receipt_sha256=_digest("receipt"),
    )
    saved = controller.state_dict()
    with pytest.raises(ValueError, match="conflicts"):
        controller.load_state_dict(dict(saved, config_digest=_digest("other")))
    with pytest.raises(ValueError, match="strict schema"):
        controller.load_state_dict(dict(saved, unexpected_key=1))
    with pytest.raises(ValueError, match="receipt"):
        MatchedComputeController(arm="A1", stage="stage_a_teacher_forcing", config_digest=_digest("a1"), teacher_handoff=_handoff())
    with pytest.raises(ValueError, match="receipt"):
        MatchedComputeController(
            arm="B",
            stage="stage_b_self_fed",
            config_digest=_digest("b"),
            teacher_handoff=_handoff(),
            matched_receipt_sha256=_digest("receipt"),
        )
