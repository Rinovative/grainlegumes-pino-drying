"""Protect the run and dataset selectors of the notebook control standard.

Responsibilities:
  - Check the summary «k von n gewählt» for single and multiple choice, observer counts, and single-choice exclusivity
  - Check the architecture grouping and the disabled «Metadaten fehlen» group of the run selector
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
    multiple = selectors.ChoiceBlock("Läufe", GROUPS, multiple=True, selected=[1, 3])
    assert multiple.header.summary == "Läufe: 2 von 3 gewählt"
    single = selectors.ChoiceBlock("Datensätze", GROUPS, multiple=False, selected=())
    assert single.header.summary == "Datensätze: 1 von 3 gewählt"
    assert single.selected() == (1,)
    assert multiple.body.layout.display == "none"
    assert not any(text.lower() in {"alle", "keine"} for text in _texts(multiple.body))


def test_select_notifies_once_and_single_choice_stays_exclusive() -> None:
    block = selectors.ChoiceBlock("Läufe", GROUPS, multiple=True, selected=[])
    calls: list[tuple[object, ...]] = []
    block.observe(lambda: calls.append(block.selected()))
    block.select([2, 3])
    assert calls == [(2, 3)]
    block.boxes[2].value = False
    assert calls[-1] == (3,)
    single = selectors.ChoiceBlock("Datensätze", GROUPS, multiple=False, selected=[2])
    single.boxes[3].value = True
    assert single.selected() == (3,)
    single.boxes[3].value = False
    assert single.selected() == (3,)
    with pytest.raises(ValueError, match="not selectable"):
        selectors.ChoiceBlock("Läufe", GROUPS, multiple=True, selected=[9])
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
    assert selector.header.summary == "Läufe: 1 von 4 gewählt"
    disabled = [text for text in texts if "physics block missing" in text or "mlp" in text]
    assert len(disabled) == 2
    assert all(len(row.children) == 1 for row in cast("widgets.Box", selector.body).children if isinstance(row, widgets.HBox))
    with pytest.raises(ValueError, match="not selectable"):
        selectors.RunSelector(runs, multiple=False, selected=["x"])


def _inventory(*, poolable: bool = True) -> presentation.dataset_selection.DatasetInventory:
    item = presentation.dataset_selection.DatasetItem
    items = [item(("lentil", "id"), "lentil", "id", "lentil, ID"), item(("chickpea", "id"), "chickpea", "id", "chickpea, ID")]
    refused = [(item(("lentil", "material_ood"), "lentil", "material_ood", "lentil, Material OOD"), "keine gemeinsamen Fälle in run_a")]
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
    assert any("keine gemeinsamen Fälle" in text for text in texts)
    assert selector.header.summary == "Datensätze: 1 von 2 gewählt"
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
    assert [choice.label for choice in enabled.choices()] == ["ID, alle Materialien"]
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
