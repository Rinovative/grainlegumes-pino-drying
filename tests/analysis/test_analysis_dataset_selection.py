"""Protect the separate and pooled resolution of selectable datasets into draw choices.

Responsibilities:
  - Check group order, separate and pooled choices, their labels, and the handling of unavailable items
  - Check that a choice holds only items of its own group

Design principles:
  - Choices follow inventory order, so the selection order never changes what is drawn

This module does NOT:
  - Load evaluation frames or recompute objectives
"""

from __future__ import annotations

import pytest

from src.analysis.presentation import dataset_selection as selection

_LABELS = {"id": "ID", "parameter_ood": "Parameter OOD", "material_ood": "Material OOD"}


def _item(material: str, group: str) -> selection.DatasetItem:
    return selection.DatasetItem((material, group), material, group, selection.dataset_label(material, _LABELS[group]))


def _inventory(*, poolable: bool = True) -> selection.DatasetInventory:
    return selection.DatasetInventory(
        [_item("lentil", "id"), _item("chickpea", "id"), _item("kidney_bean", "id"), _item("lentil", "material_ood")],
        group_order=selection.EVALUATION_GROUP_ORDER,
        group_labels=_LABELS,
        poolable=poolable,
        unavailable=[(_item("lentil", "parameter_ood"), "keine gemeinsamen Fälle in run b")],
    )


def test_labels_name_the_material_and_group_with_spaces_for_underscores() -> None:
    assert selection.dataset_label("lentil", "ID") == "lentil, ID"
    assert selection.dataset_label("kidney_bean", "Material OOD") == "kidney bean, Material OOD"


def test_separate_choices_follow_group_and_inventory_order_not_selection_order() -> None:
    inventory = _inventory()
    selected = [("lentil", "material_ood"), ("kidney_bean", "id"), ("lentil", "id")]

    choices = selection.resolve_choices(inventory, selected, pooled=False)

    assert [choice.label for choice in choices] == ["lentil, ID", "kidney bean, ID", "lentil, Material OOD"]
    assert all(choice.available and len(choice.items) == 1 for choice in choices)
    assert choices[0].members == (("lentil", "id"),)


def test_pooled_choice_merges_a_group_and_names_how_many_materials_it_holds() -> None:
    inventory = _inventory()

    partial = selection.resolve_choices(inventory, [("lentil", "id"), ("chickpea", "id"), ("lentil", "material_ood")], pooled=True)
    everything = selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=True)

    assert [choice.label for choice in partial] == ["ID, 2 von 3 Materialien", "lentil, Material OOD"]
    assert partial[0].members == (("lentil", "id"), ("chickpea", "id"))
    assert [choice.label for choice in everything] == ["ID, alle Materialien", "lentil, Material OOD"]
    assert [choice.label for choice in selection.resolve_choices(inventory, [("chickpea", "id")], pooled=True)] == ["chickpea, ID"]


def test_a_pooled_label_counts_the_unavailable_materials_of_its_group() -> None:
    inventory = selection.DatasetInventory(
        [_item("lentil", "id"), _item("chickpea", "id")],
        group_order=selection.EVALUATION_GROUP_ORDER,
        group_labels=_LABELS,
        poolable=True,
        unavailable=[(_item("kidney_bean", "id"), "keine gemeinsamen Fälle in run b")],
    )

    choices = selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=True)

    assert [choice.label for choice in choices] == ["ID, 2 von 3 Materialien"]  # never «alle» while a material of the group is unavailable


def test_a_choice_refuses_items_of_another_group() -> None:
    with pytest.raises(ValueError, match="holds items of groups"):
        selection.DatasetChoice("id", (_item("lentil", "id"), _item("lentil", "material_ood")), True, "lentil")


def test_unavailable_items_stay_apart_and_are_never_pooled() -> None:
    inventory = _inventory()
    refused = ("lentil", "parameter_ood")

    choices = selection.resolve_choices(inventory, [("lentil", "id"), refused], pooled=True)

    assert [(choice.label, choice.available) for choice in choices] == [("lentil, ID", True), ("lentil, Parameter OOD", False)]
    assert refused not in {item.key for item in inventory.items}


def test_unknown_keys_and_pooling_a_non_poolable_inventory_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown dataset keys"):
        selection.resolve_choices(_inventory(), [("rapeseed", "id")], pooled=False)
    with pytest.raises(ValueError, match="cannot be pooled"):
        selection.resolve_choices(_inventory(poolable=False), [("lentil", "id")], pooled=True)


def test_inventory_rejects_repeated_keys_and_groups_outside_its_order() -> None:
    with pytest.raises(ValueError, match="unique"):
        selection.DatasetInventory(
            [_item("lentil", "id")],
            group_order=("id",),
            group_labels=_LABELS,
            poolable=True,
            unavailable=[(_item("lentil", "id"), "reason")],
        )
    with pytest.raises(ValueError, match="outside the group order"):
        selection.DatasetInventory([_item("lentil", "material_ood")], group_order=("id",), group_labels=_LABELS, poolable=True)
