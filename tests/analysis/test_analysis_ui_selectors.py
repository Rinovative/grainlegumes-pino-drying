"""Protect the run and dataset selectors of the notebook control standard.

Responsibilities:
  - Check the summary «k of n selected» for single and multiple choice, observer counts, and single-choice exclusivity
  - Check the architecture grouping and the disabled «Missing metadata» group of the run selector
  - Check the single-choice run header «Run: <name>», the indent of derived runs and the shared button width of a group
  - Check the dataset groups, the unavailable group, and the enabled, disabled, and absent pooling switch

Design principles:
  - One case per contract branch, without a browser

This module does NOT:
  - Check the pooled-label rules, which the dataset-selection service owns
"""

from __future__ import annotations

from typing import cast

import ipywidgets as widgets
import pytest

from src.analysis import presentation
from src.analysis.ui import analysis_ui_selectors as selectors

GROUPS = [("A", [("one", 1), ("two", 2)]), ("B", [("three", 3)])]


def _texts(widget: widgets.Widget) -> list[str]:
    """Return every description and HTML value below ``widget``."""
    found = [str(getattr(widget, "description", "") or ""), str(getattr(widget, "value", "") if isinstance(widget, widgets.HTML) else "")]
    if isinstance(widget, widgets.Box):
        for child in widget.children:
            found.extend(_texts(child))
    return [text for text in found if text]


def test_summary_counts_selectable_items_for_multiple_and_single_choice() -> None:
    multiple = selectors.ChoiceBlock("Runs", GROUPS, multiple=True, selected=[1, 3])
    assert multiple.header.summary == "Runs: 2 of 3 selected"
    single = selectors.ChoiceBlock("Datasets", GROUPS, multiple=False, selected=())
    assert single.header.summary == "Datasets: 1 of 3 selected"
    assert single.selected() == (1,)
    assert multiple.body.layout.display == "none"
    assert not any(text.lower() in {"all", "none"} for text in _texts(multiple.body))


def test_select_notifies_once_and_single_choice_stays_exclusive() -> None:
    block = selectors.ChoiceBlock("Runs", GROUPS, multiple=True, selected=[])
    calls: list[tuple[object, ...]] = []
    block.observe(lambda: calls.append(block.selected()))
    block.select([2, 3])
    assert calls == [(2, 3)]
    block.boxes[2].value = False
    assert calls[-1] == (3,)
    single = selectors.ChoiceBlock("Datasets", GROUPS, multiple=False, selected=[2])
    single.boxes[3].value = True
    assert single.selected() == (3,)
    single.boxes[3].value = False
    assert single.selected() == (3,)
    with pytest.raises(ValueError, match="not selectable"):
        selectors.ChoiceBlock("Runs", GROUPS, multiple=True, selected=[9])
    with pytest.raises(ValueError, match="exactly one"):
        single.select([1, 2])


def test_runs_group_by_architecture_and_missing_metadata_stays_disabled() -> None:
    runs = [
        selectors.RunOption("pi-uno_b", "pi-uno", "pu"),
        selectors.RunOption("uno_a", "uno", "u"),
        selectors.RunOption("fno_z", "fno", "f2"),
        selectors.RunOption("fno_a", "fno", "f1"),
        selectors.RunOption("old_run", None, "x", issue="physics block missing"),
        selectors.RunOption("odd_run", "mlp", "y"),
    ]
    selector = selectors.RunSelector(runs, multiple=True, selected=["f1"])
    texts = _texts(selector.body)
    headings = [text for text in texts if text.startswith("<b>")]
    assert headings == ["<b>FNO</b>", "<b>U-NO</b>", "<b>PI-U-NO</b>", f"<b>{selectors.MISSING_METADATA_GROUP}</b>"]
    assert list(selector.boxes) == ["f1", "f2", "u", "pu"]
    assert selector.header.summary == "Runs: 1 of 4 selected"
    disabled = [text for text in texts if "physics block missing" in text or "mlp" in text]
    assert len(disabled) == 2
    assert all(len(row.children) == 1 for row in cast("widgets.Box", selector.body).children if isinstance(row, widgets.HBox))
    with pytest.raises(ValueError, match="not selectable"):
        selectors.RunSelector(runs, multiple=False, selected=["x"])
    assert selectors.RunSelector(runs, multiple=True).header.value is False
    opened = selectors.RunSelector(runs, multiple=True, opened=True)
    assert opened.header.value is True
    assert opened.body.layout.display != "none"


def test_run_header_names_the_selected_run_for_a_single_choice_and_counts_for_a_multiple_choice() -> None:
    runs = [selectors.RunOption("FNO · A0 · seed 9", "fno", "a"), selectors.RunOption("FNO · B · seed 9", "fno", "b")]
    single = selectors.RunSelector(runs, multiple=False, selected=["a"])
    assert single.header.summary == "Run: FNO · A0 · seed 9"
    single.boxes["b"].value = True
    assert single.header.summary == "Run: FNO · B · seed 9"
    single.select(["a"])
    assert single.header.summary == "Run: FNO · A0 · seed 9"
    assert selectors.RunSelector(runs, multiple=False).header.summary == "Run: FNO · A0 · seed 9"
    assert selectors.RunSelector(runs, multiple=True, selected=["a", "b"]).header.summary == "Runs: 2 of 2 selected"


def test_a_run_with_a_listed_parent_stands_indented_directly_below_it_and_an_orphan_stands_unindented() -> None:
    runs = [
        selectors.RunOption("fno__a0", "fno", "a0"),
        selectors.RunOption("fno__a0x", "fno", "x"),
        selectors.RunOption("fno__b-a1__long_suffix__a1", "fno", "a1", parent="a0"),
        selectors.RunOption("fno__b-a1__long_suffix__b", "fno", "b", parent="a0"),
        selectors.RunOption("fno__z", "fno", "orphan", parent="missing"),
    ]
    selector = selectors.RunSelector(runs, multiple=False)
    rows = [row for row in cast("widgets.Box", selector.body).children if isinstance(row, widgets.HBox)]
    margin = {key: next(row.layout.margin for row in rows if row.children == (box,)) for key, box in selector.boxes.items()}
    assert list(selector.boxes) == ["a0", "a1", "b", "x", "orphan"]
    indent = f"0 0 0 {selectors._INDENT_PX}px"
    assert margin == {"a0": None, "a1": indent, "b": indent, "x": None, "orphan": None}
    widths = {key: int(str(box.layout.width)[:-2]) for key, box in selector.boxes.items()}
    assert widths["a0"] == widths["a1"] + selectors._INDENT_PX == widths["b"] + selectors._INDENT_PX


def test_indented_applies_to_per_line_choices_only() -> None:
    with pytest.raises(ValueError, match="indented"):
        selectors.ChoiceBlock("Runs", GROUPS, multiple=False, selected=[], indented=[2])


def test_close_on_pick_closes_a_single_choice_before_its_observers_run() -> None:
    runs = [selectors.RunOption("fno_a", "fno", "a"), selectors.RunOption("fno_b", "fno", "b")]
    selector = selectors.RunSelector(runs, multiple=False, selected=["a"], opened=True, close_on_pick=True)
    seen: list[bool] = []
    selector.observe(lambda: seen.append(selector.header.value))
    selector.boxes["b"].value = True
    assert seen == [False]
    selector.header.value = True
    selector.boxes["b"].value = False  # pressing the selected button again keeps it and closes the list
    assert selector.selected() == ("b",)
    assert selector.header.value is False
    selector.header.value = True
    selector.select(["a"])
    assert selector.header.value is True
    with pytest.raises(ValueError, match="close_on_pick"):
        selectors.RunSelector(runs, multiple=True, close_on_pick=True)


def _inventory(*, poolable: bool = True) -> presentation.dataset_selection.DatasetInventory:
    item = presentation.dataset_selection.DatasetItem
    items = [item(("lentil", "id"), "lentil", "id", "lentil, ID"), item(("chickpea", "id"), "chickpea", "id", "chickpea, ID")]
    refused = [(item(("lentil", "material_ood"), "lentil", "material_ood", "lentil, Material OOD"), "no shared cases in run_a")]
    return presentation.dataset_selection.DatasetInventory(
        items,
        group_order=("id", "material_ood"),
        group_labels={"id": "ID", "material_ood": "Material OOD"},
        poolable=poolable,
        unavailable=refused,
    )


def test_dataset_selector_groups_and_lists_unavailable_items() -> None:
    selector = selectors.DatasetSelector(_inventory(), multiple=True, selected=[("lentil", "id")])
    texts = _texts(selector.body)
    assert "<b>ID</b>" in texts
    assert f"<b>{selectors.UNAVAILABLE_GROUP}</b>" in texts
    assert any("no shared cases" in text for text in texts)
    assert selector.header.summary == "Datasets: 1 of 2 selected"
    assert [choice.label for choice in selector.choices()] == ["lentil, ID"]


def test_pool_switch_enabled_disabled_with_reason_or_absent() -> None:
    every = [("lentil", "id"), ("chickpea", "id")]
    enabled = selectors.DatasetSelector(_inventory(), multiple=True, selected=every)
    assert enabled.pool_toggle is not None
    assert not enabled.pool_toggle.disabled
    calls: list[bool] = []
    enabled.observe(lambda: calls.append(enabled.pooled))
    enabled.pool_toggle.value = True
    assert calls == [True]
    assert [choice.label for choice in enabled.choices()] == ["ID, all materials"]
    enabled.set_state([("lentil", "id")], pooled=False)
    assert calls == [True]
    assert not enabled.pooled

    disabled = selectors.DatasetSelector(_inventory(), multiple=True, selected=every, pooled=True, pool_reason="nicht materialausgeglichen")
    assert disabled.pool_toggle is not None
    assert disabled.pool_toggle.disabled
    assert not disabled.pooled
    assert any("nicht materialausgeglichen" in text for text in _texts(disabled.body))
    assert len(disabled.choices()) == 2

    absent = selectors.DatasetSelector(_inventory(poolable=False), multiple=False, selected=())
    assert absent.pool_toggle is None
    assert not absent.pool_enabled
