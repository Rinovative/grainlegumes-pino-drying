"""Protect the resumable unit sampler that both tasks draw their optimizer updates from.

Responsibilities:
  - Check that every Train unit is drawn once per epoch in groups of k, the last group shorter
  - Check the case weights that give every Train case equal total weight per epoch
  - Check that a mid-epoch state reproduces the remaining order exactly and that foreign states are refused
  - Check that a continuation adopts only the generator stream of a mid-epoch state and starts a fresh epoch
  - Check the count of distinct cases the committed units cover, below and at the full Train set

Design principles:
  - Synthetic unit lists only; no Dataset, loader or model

This module does NOT:
  - Test loss reduction over a group; the training-loop tests own it
"""

from __future__ import annotations

import pytest
import torch

from src.datasets.runtime import dataset_runtime_units as units


def _sampler(*, seed: int = 3, memory_batch: int = 2) -> units.UnitSampler:
    """Return a sampler over seven units of three cases (4, 2 and 1 units) with two keys."""
    cases = (0, 0, 0, 0, 1, 1, 2)
    keys = (4, 4, 4, 4, 4, 4, 3)
    return units.UnitSampler(
        unit_kind="window", cases=cases, keys=keys, units_per_update=3, memory_batch=memory_batch, seed=seed, full_set_transitions=sum(keys)
    )


def _drain(sampler: units.UnitSampler) -> list[tuple[int, ...]]:
    """Commit every remaining group of the current epoch and return their items."""
    drawn = []
    for group in sampler.groups():
        drawn.append(group.items)
        sampler.advance(len(group.items))
    return drawn


def test_every_unit_is_drawn_once_per_epoch_with_equal_case_weight() -> None:
    """Groups of k cover each unit once, the last group is shorter, and each case sums to the same weight."""
    sampler = _sampler()
    assert sampler.updates_per_epoch == 3
    sampler.begin_epoch(0)
    groups = sampler.groups()
    assert [len(group.items) for group in groups] == [3, 3, 1]
    # The loader length counts the microbatches the sampler yields, not the optimizer updates.
    assert len(sampler) == len(list(sampler)) == sum(len(group.microbatches) for group in groups) > sampler.updates_per_epoch
    assert sorted(item for group in groups for item in group.items) == list(range(7))
    for group in groups:
        assert sorted(position for batch in group.microbatches for position in batch) == list(range(len(group.items)))
        assert all(len(batch) <= 2 and len({group.keys[position] for position in batch}) == 1 for batch in group.microbatches)
    # Seven units over three cases: w = (7/3) / units of the case, so every case sums to 7/3 per epoch.
    totals: dict[object, float] = {}
    for group in groups:
        for case, weight in zip(group.cases, group.weights, strict=True):
            totals[case] = totals.get(case, 0.0) + weight
    assert totals == pytest.approx({0: 7 / 3, 1: 7 / 3, 2: 7 / 3})
    _drain(sampler)
    assert sampler.epoch_complete
    sampler.begin_epoch(1)
    assert sorted(item for group in sampler.groups() for item in group.items) == list(range(7))


def test_a_mid_epoch_state_reproduces_the_remaining_order_and_rejects_foreign_states() -> None:
    """Resume at a group boundary continues the same epoch and the same generator stream."""
    reference = _sampler()
    reference.begin_epoch(0)
    first = reference.groups()[0]
    reference.advance(len(first.items))
    state = reference.state_dict()
    expected = _drain(reference)
    reference.begin_epoch(1)
    expected_next = [group.items for group in reference.groups()]

    resumed = _sampler(seed=99, memory_batch=1)  # the memory split is not part of the order
    resumed.load_state_dict(state)
    resumed.begin_epoch(0)
    assert _drain(resumed) == expected
    resumed.begin_epoch(1)
    assert [group.items for group in resumed.groups()] == expected_next

    with pytest.raises(ValueError, match="update-group boundary"):
        reference.advance(2)
    with pytest.raises(ValueError, match="update-group boundary"):
        units.validate_state({**state, "cursor": 2})
    with pytest.raises(ValueError, match="another unit kind"):
        units.UnitSampler(
            unit_kind="trajectory", cases=range(7), keys=None, units_per_update=3, memory_batch=1, seed=0, full_set_transitions=27
        ).load_state_dict(state)
    with pytest.raises(ValueError, match="keys must be exactly"):
        units.validate_state({key: value for key, value in state.items() if key != "generator_state"})


def test_a_continuation_adopts_the_generator_of_a_mid_epoch_teacher_state_and_starts_a_fresh_epoch() -> None:
    """A B or A1 sampler continues an A0 state selected inside an epoch with that state's generator stream."""
    teacher = _sampler()
    teacher.begin_epoch(0)
    teacher.advance(len(teacher.groups()[0].items))
    state = teacher.state_dict()
    assert not teacher.epoch_complete
    stream = torch.Generator()
    stream.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))

    continued = units.UnitSampler(
        unit_kind="trajectory", cases=range(5), keys=None, units_per_update=2, memory_batch=1, seed=99, full_set_transitions=20
    )
    continued.continue_generator(state)
    assert (continued.epoch, continued.permutation, continued.cursor) == (-1, [], 0)
    continued.begin_epoch(0)
    assert continued.permutation == torch.randperm(5, generator=stream).tolist()


def test_seen_cases_count_the_committed_units_until_an_epoch_covers_the_full_set() -> None:
    """The cases seen come from the epoch order before the cursor, and are all Train cases once an epoch completed, also after a restore."""
    sampler = _sampler()
    assert (sampler.case_count, sampler.seen_cases) == (3, 0)
    sampler.begin_epoch(0)
    first = sampler.groups()[0]
    sampler.advance(len(first.items))
    assert sampler.seen_cases == len(set(first.cases)) < sampler.case_count
    restored = _sampler(seed=99)
    restored.load_state_dict(sampler.state_dict())
    assert restored.seen_cases == sampler.seen_cases
    _drain(sampler)
    sampler.begin_epoch(1)
    assert sampler.seen_cases == sampler.case_count == 3
