"""Draw resumable groups of Train units, one group per optimizer update, for both tasks.

Responsibilities:
  - Shuffle every Train unit once per epoch with one persisted generator
  - Cut the epoch order into groups of ``units_per_update`` units, the last group of an epoch shorter
  - Split every group into memory microbatches of at most ``memory_batch`` units that share one collation key
  - Weight every unit by (mean units per Train case) / (units of its case), so every case counts equally per epoch
  - Persist and restore the epoch order and the committed group position exactly
  - Count the distinct cases the committed units cover, from the epoch, its order and the cursor
  - Carry the full Train role's unit and supervised-transition counts, which a stage budget in passes counts also for a Train subset
  - Partition an item order into key-homogeneous batches for evaluation loaders

Design principles:
  - A group, its units, their weights and its microbatches are fixed before any of them is loaded,
    so the memory split changes only floating-point rounding
  - The cursor moves only when the caller commits a completed optimizer group
  - The same case may appear several times in one group; no unit repeats within an epoch

This module does NOT:
  - Load items, compute losses or step optimizers
  - Choose the unit kind; the task runtime that builds the Train Dataset names it
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import torch
from torch.utils.data import Sampler

UnitKind = Literal["case", "transition", "window", "trajectory"]
UNIT_KINDS: Final[tuple[UnitKind, ...]] = ("case", "transition", "window", "trajectory")
_STATE_KEYS: Final = frozenset({"unit_kind", "size", "units_per_update", "epoch", "permutation", "cursor", "generator_state"})
_BYTE_MAX: Final = 255


@dataclass(frozen=True, slots=True)
class UnitGroup:
    """
    Describe one optimizer update before any of its units is loaded.

    ``items`` are the Dataset indices of the update's units in drawn order; ``keys``,
    ``weights`` and ``cases`` follow that order. ``microbatches`` hold positions into
    ``items``; every microbatch shares one key and holds at most ``memory_batch`` units.
    """

    items: tuple[int, ...]
    keys: tuple[Hashable, ...]
    weights: tuple[float, ...]
    cases: tuple[Hashable, ...]
    microbatches: tuple[tuple[int, ...], ...]

    @property
    def distinct_cases(self) -> int:
        """Return how many different cases the update draws from."""
        return len(set(self.cases))


def _positive_int(value: object, *, label: str) -> int:
    """Return one positive integer or raise a labelled error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{label} must be a positive integer, got {value!r}."
        raise ValueError(message)
    return value


def group_batches(order: Sequence[int], keys: Sequence[Hashable] | None, batch_size: int) -> list[list[int]]:
    """
    Partition an item order into batches whose items share one key.

    Parameters
    ----------
    order : Sequence[int]
        Item indices in the order they should be consumed.
    keys : Sequence[Hashable] or None
        Key of every item index; ``None`` puts all items under one key.
    batch_size : int
        Maximum items per batch.

    Returns
    -------
    list[list[int]]
        Batches that keep the relative item order within each key; a batch
        closes at ``batch_size`` items or at the end, and batches are ordered
        by their first item. With one key this is consecutive chunking.

    Raises
    ------
    ValueError
        If ``batch_size`` is not a positive integer or an index has no key.

    """
    _positive_int(batch_size, label="batch_size")
    items = list(order)
    if keys is None:
        return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    if any(item < 0 or item >= len(keys) for item in items):
        message = "Every batched item index needs one key."
        raise ValueError(message)
    open_batches: dict[Hashable, tuple[int, list[int]]] = {}
    closed: list[tuple[int, list[int]]] = []
    for position, item in enumerate(items):
        first, batch = open_batches.setdefault(keys[item], (position, []))
        batch.append(item)
        if len(batch) == batch_size:
            closed.append((first, batch))
            del open_batches[keys[item]]
    closed.extend(open_batches.values())
    return [batch for _, batch in sorted(closed, key=lambda entry: entry[0])]


class UnitSampler(Sampler[list[int]]):
    """
    Yield the memory microbatches of every remaining update group of one epoch.

    Parameters
    ----------
    unit_kind : {"case", "transition", "window", "trajectory"}
        What one Train item is; recorded in the persisted state.
    cases : Sequence[Hashable]
        Case of every Train item, in Dataset order.
    keys : Sequence[Hashable] | None
        Collation key of every item (items of different keys never share a microbatch); ``None`` for one key.
    units_per_update : int
        ``training.units_per_update`` (k): units of one optimizer update.
    memory_batch : int
        ``training.memory_batch``: largest microbatch loaded at once.
    seed : int
        Seed of the generator that draws every epoch order.
    full_set_units : int | None, optional
        Units of the full Train role when the sampler draws from a Train subset (learning curve);
        ``None`` when it draws from the full role. A stage budget in epochs counts these units.
    full_set_transitions : int | None, optional
        Supervised transitions N of the full Train role, the sum of its trajectories' transitions,
        which a budget in supervised transitions counts (B: K x N). ``None`` only where one unit is
        one supervised prediction (a steady case, a one-step transition), so N is the full-set units;
        window and trajectory units require it.

    """

    def __init__(
        self,
        *,
        unit_kind: UnitKind,
        cases: Sequence[Hashable],
        keys: Sequence[Hashable] | None,
        units_per_update: int,
        memory_batch: int,
        seed: int,
        full_set_units: int | None = None,
        full_set_transitions: int | None = None,
    ) -> None:
        """Bind the units, their case weights and the update grouping."""
        if unit_kind not in UNIT_KINDS:
            message = f"unit_kind must be one of {list(UNIT_KINDS)}, got {unit_kind!r}."
            raise ValueError(message)
        if not cases:
            message = "Unit sampling requires at least one Train unit."
            raise ValueError(message)
        if keys is not None and len(keys) != len(cases):
            message = "Unit keys must cover every Train unit."
            raise ValueError(message)
        if isinstance(seed, bool) or not isinstance(seed, int):
            message = "Unit sampling seed must be an integer."
            raise TypeError(message)
        self.unit_kind: UnitKind = unit_kind
        self.size = len(cases)
        self.full_set_units = self.size if full_set_units is None else _positive_int(full_set_units, label="full_set_units")
        if self.full_set_units < self.size:
            message = "full_set_units cannot be smaller than the sampled Train units."
            raise ValueError(message)
        if full_set_transitions is None and unit_kind in {"window", "trajectory"}:
            message = "A window or trajectory unit holds several transitions; the sampler needs the full Train set's full_set_transitions."
            raise ValueError(message)
        self.full_set_transitions = (
            self.full_set_units if full_set_transitions is None else _positive_int(full_set_transitions, label="full_set_transitions")
        )
        self.units_per_update = _positive_int(units_per_update, label="units_per_update")
        self.memory_batch = _positive_int(memory_batch, label="memory_batch")
        self._cases = tuple(cases)
        self._keys = None if keys is None else tuple(keys)
        counts = Counter(self._cases)
        mean_units = self.size / len(counts)
        self.weights = tuple(mean_units / counts[case] for case in self._cases)
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = -1
        self.permutation: list[int] = []
        self.cursor = 0

    def begin_epoch(self, epoch: int) -> None:
        """Draw one new epoch order only after the previous epoch was fully committed."""
        if epoch == self.epoch and self.permutation:
            return
        if epoch != self.epoch + 1 or (self.permutation and self.cursor != self.size):
            message = "Unit sampler epoch progression conflicts with its committed cursor."
            raise ValueError(message)
        self.permutation = torch.randperm(self.size, generator=self.generator).tolist()
        self.cursor = 0
        self.epoch = epoch

    @property
    def epoch_complete(self) -> bool:
        """Return whether every unit of the current epoch was committed."""
        return bool(self.permutation) and self.cursor == self.size

    def groups(self) -> tuple[UnitGroup, ...]:
        """Return the uncommitted update groups of the current epoch in drawn order."""
        if not self.permutation:
            message = "Unit sampler has no epoch order; call begin_epoch first."
            raise RuntimeError(message)
        remaining = self.permutation[self.cursor :]
        groups: list[UnitGroup] = []
        for start in range(0, len(remaining), self.units_per_update):
            items = tuple(remaining[start : start + self.units_per_update])
            keys = tuple(None if self._keys is None else self._keys[item] for item in items)
            microbatches = group_batches(range(len(items)), None if self._keys is None else keys, self.memory_batch)
            groups.append(
                UnitGroup(
                    items=items,
                    keys=keys,
                    weights=tuple(self.weights[item] for item in items),
                    cases=tuple(self._cases[item] for item in items),
                    microbatches=tuple(tuple(batch) for batch in microbatches),
                )
            )
        return tuple(groups)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield the Dataset indices of every uncommitted microbatch, group after group."""
        return iter([[group.items[position] for position in batch] for group in self.groups() for batch in group.microbatches])

    def __len__(self) -> int:
        """Return how many microbatches :meth:`__iter__` yields: those of the uncommitted groups of the current epoch."""
        return sum(len(group.microbatches) for group in self.groups())

    @property
    def case_count(self) -> int:
        """Return how many distinct cases the Train units belong to."""
        return len(set(self._cases))

    @property
    def seen_cases(self) -> int:
        """Return how many distinct cases the committed units cover: all after a complete epoch, else those before the cursor."""
        if self.epoch > 0 or self.epoch_complete:
            return self.case_count
        return len({self._cases[item] for item in self.permutation[: self.cursor]})

    @property
    def updates_per_epoch(self) -> int:
        """Return the optimizer updates of one full epoch."""
        return math.ceil(self.size / self.units_per_update)

    def advance(self, units: int) -> None:
        """Commit one completed update group of ``units`` units."""
        cursor = self.cursor + _positive_int(units, label="units")
        if not self.permutation or cursor > self.size or (cursor % self.units_per_update and cursor != self.size):
            message = "Unit sampler commit does not end on an update-group boundary of its epoch order."
            raise ValueError(message)
        self.cursor = cursor

    def state_dict(self) -> dict[str, Any]:
        """Return the exact epoch order, committed position and generator state."""
        return {
            "unit_kind": self.unit_kind,
            "size": self.size,
            "units_per_update": self.units_per_update,
            "epoch": self.epoch,
            "permutation": list(self.permutation),
            "cursor": self.cursor,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state of this unit set and grouping; reject any other."""
        validate_state(state)
        if (state["unit_kind"], state["size"], state["units_per_update"]) != (self.unit_kind, self.size, self.units_per_update):
            message = "Saved unit sampler state belongs to another unit kind, Train unit count or units_per_update."
            raise ValueError(message)
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
        self.epoch = state["epoch"]
        self.permutation = list(state["permutation"])
        self.cursor = state["cursor"]

    def continue_generator(self, state: Mapping[str, Any]) -> None:
        """
        Continue another stage's generator stream at a fresh epoch.

        A matched continuation adopts only the generator of its teacher's state,
        which a validation point may have selected inside an epoch; its own units,
        order and position start anew.
        """
        validate_state(state)
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
        self.epoch = -1
        self.permutation = []
        self.cursor = 0


def validate_state(state: Mapping[str, Any]) -> None:
    """
    Reject a malformed unit sampler state before it reaches a sampler.

    Raises
    ------
    ValueError
        If keys, the unit kind, the order, the epoch or the cursor are invalid, or the
        cursor does not lie on an update-group boundary.

    """
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        message = f"Unit sampler state keys must be exactly {sorted(_STATE_KEYS)}."
        raise ValueError(message)
    size, epoch, cursor, permutation = state["size"], state["epoch"], state["cursor"], state["permutation"]
    units_per_update = _positive_int(state["units_per_update"], label="units_per_update")
    if state["unit_kind"] not in UNIT_KINDS or type(size) is not int or size < 1 or type(epoch) is not int or epoch < -1:
        message = "Unit sampler state has an invalid unit kind, size or epoch."
        raise ValueError(message)
    if not isinstance(permutation, list) or (permutation and sorted(permutation) != list(range(size))) or (epoch >= 0) != bool(permutation):
        message = "Unit sampler state order is not one permutation of its units for its epoch."
        raise ValueError(message)
    if type(cursor) is not int or not 0 <= cursor <= len(permutation) or (cursor % units_per_update and cursor != size):
        message = "Unit sampler state cursor does not lie on an update-group boundary."
        raise ValueError(message)
    generator_state = state["generator_state"]
    if not isinstance(generator_state, list) or any(type(value) is not int or not 0 <= value <= _BYTE_MAX for value in generator_state):
        message = "Unit sampler generator state must be a list of bytes."
        raise ValueError(message)
    try:
        torch.Generator().set_state(torch.tensor(generator_state, dtype=torch.uint8))
    except RuntimeError as error:
        message = "Unit sampler generator state is not a valid Torch generator state."
        raise ValueError(message) from error
