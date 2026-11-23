"""Resolve which datasets a view draws, separately per item or pooled per group.

Responsibilities:
  - Describe the selectable datasets of one view as items grouped by role or sampling regime, with unavailable items kept apart with their reason
  - Resolve selected item keys into ordered draw choices: one choice per item by default, one per group when pooled
  - Label an item «material, group» and a pooled choice «group, k von n Materialien» or «group, alle Materialien»

Design principles:
  - Choices follow the inventory and group order, never the order of the selection
  - A pooled choice with a single item is that item, so its label and its draw equal the separate one
  - A pooled label counts the group's unavailable items among its n materials, so it never claims all while one is unavailable
  - Unknown keys, a pooled request on a non-poolable inventory, a choice holding another group's item, and inconsistent inventories
    fail closed

This module does NOT:
  - Load frames, recompute any objective, or decide which views may pool
  - Construct widgets or choose colors
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

EVALUATION_GROUP_ORDER: Final = ("id", "parameter_ood", "material_ood")


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """Bind one selectable dataset key to its material, group, and display label."""

    key: Hashable
    material: str
    group: str
    label: str


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """Hold the selectable and the unavailable items of one view with the group order and group labels."""

    items: Sequence[DatasetItem]
    _: KW_ONLY
    group_order: Sequence[str]
    group_labels: Mapping[str, str]
    poolable: bool
    unavailable: Sequence[tuple[DatasetItem, str]] = ()

    def __post_init__(self) -> None:
        """Freeze the collections and require one consistent group and key space."""
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "group_order", tuple(self.group_order))
        object.__setattr__(self, "group_labels", MappingProxyType(dict(self.group_labels)))
        object.__setattr__(self, "unavailable", tuple((item, str(reason)) for item, reason in self.unavailable))
        if len(set(self.group_order)) != len(self.group_order):
            message = "Dataset group order must not repeat a group."
            raise ValueError(message)
        missing = [group for group in self.group_order if group not in self.group_labels]
        if missing:
            message = f"Dataset group labels are missing for {missing}."
            raise ValueError(message)
        every = (*self.items, *(item for item, _reason in self.unavailable))
        stray = sorted({item.group for item in every} - set(self.group_order))
        if stray:
            message = f"Dataset items name groups outside the group order: {stray}."
            raise ValueError(message)
        keys = [item.key for item in every]
        if len(set(keys)) != len(keys):
            message = "Dataset item keys must be unique across available and unavailable items."
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class DatasetChoice:
    """Describe one dataset a view draws: one item, or the pooled items of one group."""

    group: str
    items: tuple[DatasetItem, ...]
    available: bool
    label: str

    def __post_init__(self) -> None:
        """Require every item to belong to the choice's group, which decides the frame a view cuts."""
        stray = sorted({item.group for item in self.items} - {self.group})
        if stray:
            message = f"Dataset choice of group {self.group!r} holds items of groups {stray}."
            raise ValueError(message)

    @property
    def members(self) -> tuple[Hashable, ...]:
        """Return the item keys the choice draws."""
        return tuple(item.key for item in self.items)


def dataset_label(material: str, group_label: str) -> str:
    """Return the dataset label «lentil, ID» with underscores of the material shown as spaces."""
    if not material or not group_label:
        message = "A dataset label needs a material and a group label."
        raise ValueError(message)
    return f"{material.replace('_', ' ')}, {group_label}"


def resolve_choices(inventory: DatasetInventory, selected: Iterable[Hashable], *, pooled: bool) -> tuple[DatasetChoice, ...]:
    """
    Resolve selected item keys into the choices a view draws, in group and inventory order.

    Parameters
    ----------
    inventory : DatasetInventory
        Selectable and unavailable items of the view.
    selected : Iterable[Hashable]
        Selected item keys; their order and repetition do not matter.
    pooled : bool
        Merge the selected available items of each group into one choice.

    Returns
    -------
    tuple[DatasetChoice, ...]
        Available choices per group, then the selected unavailable items of that group as
        choices with ``available=False`` that are never pooled.

    Raises
    ------
    ValueError
        If a key is unknown or ``pooled`` is requested on an inventory that is not poolable.

    """
    chosen = set(selected)
    refused = {item.key: item for item, _reason in inventory.unavailable}
    unknown = chosen - {item.key for item in inventory.items} - refused.keys()
    if unknown:
        message = f"Unknown dataset keys {sorted(map(repr, unknown))}."
        raise ValueError(message)
    if pooled and not inventory.poolable:
        message = "This dataset inventory cannot be pooled."
        raise ValueError(message)
    choices: list[DatasetChoice] = []
    for group in inventory.group_order:
        members = tuple(item for item in inventory.items if item.group == group and item.key in chosen)
        if pooled and members:
            total = sum(item.group == group for item in (*inventory.items, *refused.values()))
            choices.append(DatasetChoice(group, members, True, _pooled_label(inventory.group_labels[group], members, total)))
        else:
            choices.extend(DatasetChoice(group, (item,), True, item.label) for item in members)
        choices.extend(DatasetChoice(group, (item,), False, item.label) for key, item in refused.items() if key in chosen and item.group == group)
    return tuple(choices)


def _pooled_label(group_label: str, members: tuple[DatasetItem, ...], total: int) -> str:
    """Name a pooled choice by its group and how many of the group's ``total`` materials, unavailable ones included, it merges."""
    if len(members) == 1:
        return members[0].label
    if len(members) == total:
        return f"{group_label}, alle Materialien"
    return f"{group_label}, {len(members)} von {total} Materialien"
