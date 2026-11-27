"""Select runs and datasets through one collapsed line, «<noun>: k of n selected», whose body lists the choices by group.

Responsibilities:
  - Show grouped choices as checkboxes (several) or one-of buttons (one), with a summary line that counts the selection
  - List unavailable choices in disabled groups with each item's reason
  - Group runs by architecture in the fixed run-selector order, one per line, left-aligned, each under the name its caller gives
    (the Evaluation workspace gives its short label), a continuing run indented below its parent run
  - Keep runs without valid architecture metadata visible and disabled under «Missing metadata»
  - Group datasets by role or sampling regime and offer the switch «Pool materials per role» where pooling is valid

Design principles:
  - The summary counts selectable items only and reads «k of n selected» for single and multiple choice alike (F3); the single run
    selector is the exception and reads «Run: <name of the selected run>», in the Evaluation workspace its short label (Rino 16.11.)
  - No «all» or «none» buttons
  - Every body starts closed (Q20, Q22), except the Evaluation workspace run selector, which opens at the start (Rino 16.11.)
  - The pooling switch is enabled only where the view's reduction is material-balanced; elsewhere it is disabled with its reason
  - Selection changes notify observers once, programmatic state changes can stay silent

This module does NOT:
  - Load runs or frames, admit roles, or recompute objectives
  - Resolve pooled choices itself; it delegates to the dataset-selection service
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Final

import ipywidgets as widgets

from src.analysis import presentation

from . import analysis_ui_controls as controls
from . import analysis_ui_panel as panel
from . import analysis_ui_style as style

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Iterable, Sequence

    from src.analysis.presentation.analysis_dataset_selection import DatasetChoice, DatasetInventory

    Group = tuple[str, Sequence[tuple[str, Hashable]]]
    DisabledGroup = tuple[str, Sequence[tuple[str, str]]]

MISSING_METADATA_GROUP: Final = "Missing metadata"
UNAVAILABLE_GROUP: Final = "Unavailable"
POOL_LABEL: Final = "Pool materials per role"
_ONE_OF_ALLOWANCE_CH: Final = 4
_INDENT_PX: Final = 3 * style.CH_PX  # indent of a run below its parent run


def _heading(text: str) -> panel.Row:
    """Return the bold heading row of one group."""
    label = widgets.HTML(f"<b>{escape(text)}</b>", layout=widgets.Layout(width=f"{style.GRID.snap(style.text_px(text))}px", flex="0 0 auto"))
    return panel.row(label)


def _one_of_px(label: str) -> int:
    """Return the estimated pixel width of one one-of button with ``label``."""
    return style.text_px(label, allowance_ch=_ONE_OF_ALLOWANCE_CH)


def _one_of(label: str, value: bool, *, width_px: int | None = None) -> widgets.ToggleButton:
    """Return one button of a one-of choice, as wide as its full text or ``width_px``."""
    return widgets.ToggleButton(
        value=value,
        description=label,
        tooltip=label,
        layout=widgets.Layout(width=f"{style.GRID.snap(_one_of_px(label)) if width_px is None else width_px}px", flex="0 0 auto"),
    )


def _disabled_item(label: str, reason: str, *, multiple: bool) -> widgets.Widget:
    """Return a disabled choice that names why it cannot be selected."""
    text = f"{label} — {reason}"
    item = controls.checkbox(text, value=False, tooltip=reason) if multiple else _one_of(text, False)
    item.disabled = True
    return item


class ChoiceBlock:
    """
    The collapsed line «<noun>: k of n selected» whose body lists grouped choices.

    ``header`` is one element of a panel row and ``body`` its popover; pass ``(header, body)`` to the panel's
    ``popovers``. Several choices use checkboxes, one choice uses one-of buttons. ``boxes`` maps each selectable key
    to its widget. ``per_line`` puts every item on its own line (long names; the buttons of a group share the width of its
    longest); ``indented`` keys (per-line only) stand shifted right by ``_INDENT_PX``; ``footer`` rows follow the groups.
    """

    def __init__(
        self,
        noun: str,
        groups: Sequence[Group],
        *,
        multiple: bool,
        selected: Iterable[Hashable],
        disabled_groups: Sequence[DisabledGroup] = (),
        per_line: bool = False,
        indented: Collection[Hashable] = (),
        footer: Sequence[widgets.Widget] = (),
        opened: bool = False,
        close_on_pick: bool = False,
    ) -> None:
        """
        Create the block with ``selected`` checked and the body closed, or open with ``opened``.

        A single choice without a selected key selects its first item; with ``close_on_pick`` a click on one of its
        buttons closes the body before observers run (``select()`` leaves it open).

        Raises
        ------
        ValueError
            If a key repeats, a selected key is not selectable, a single choice gets more than one key, or
            ``close_on_pick`` is set for a multiple choice, or ``indented`` is given without ``per_line``.

        """
        if close_on_pick and multiple:
            message = f"{noun}: close_on_pick applies to a single choice only."
            raise ValueError(message)
        if indented and not per_line:
            message = f"{noun}: indented applies to per-line choices only."
            raise ValueError(message)
        shifted = frozenset(indented)
        self.noun = noun
        self.multiple = multiple
        self._close_on_pick = close_on_pick
        keys = [key for _heading_text, items in groups for _label, key in items]
        if len(set(keys)) != len(keys):
            message = f"{noun}: every choice needs its own key."
            raise ValueError(message)
        chosen = list(dict.fromkeys(selected))
        unknown = [key for key in chosen if key not in keys]
        if unknown:
            message = f"{noun}: selected keys {unknown!r} are not selectable."
            raise ValueError(message)
        if not multiple:
            if len(chosen) > 1:
                message = f"{noun}: a single choice takes at most one selected key, got {chosen!r}."
                raise ValueError(message)
            chosen = chosen or keys[:1]
        self._callbacks: list[Callable[[], None]] = []
        self._silent = False
        self.boxes: dict[Hashable, widgets.Checkbox | widgets.ToggleButton] = {}
        rows: list[widgets.Widget] = []
        for heading, items in groups:
            rows.append(_heading(heading))
            # One width for the buttons of a per-line single-choice group: the longest label, an indented one counted with its indent.
            widest = (
                style.GRID.snap(max((_one_of_px(label) + (_INDENT_PX if key in shifted else 0) for label, key in items), default=0))
                if per_line and not multiple
                else 0
            )
            group_boxes: list[widgets.Widget] = []
            for label, key in items:
                shift = _INDENT_PX if key in shifted else 0
                if multiple:
                    box = controls.checkbox(label, value=key in chosen)
                else:
                    box = _one_of(label, key in chosen, width_px=widest - shift if per_line else None)
                box.observe(lambda change, key=key: self._changed(key, change), names="value")
                self.boxes[key] = box
                group_boxes.append(box)
                if per_line:
                    line = panel.row(box)
                    if shift:
                        line.layout.margin = f"0 0 0 {_INDENT_PX}px"
                    rows.append(line)
            if not per_line:
                rows.append(panel.row(*group_boxes))
        for heading, disabled_items in disabled_groups:
            rows.append(_heading(heading))
            rows.extend(panel.row(_disabled_item(label, reason, multiple=multiple)) for label, reason in disabled_items)
        rows.extend(footer)
        self.header, self.body = panel.disclosure(self._summary(), widgets.VBox(tuple(rows)), opened=opened)

    def selected(self) -> tuple[Hashable, ...]:
        """Return the selected keys in display order."""
        return tuple(key for key, box in self.boxes.items() if box.value)

    def select(self, keys: Iterable[Hashable]) -> None:
        """Select exactly ``keys``; observers are called once."""
        self._set(keys)
        self._notify()

    def observe(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after every change of the selection."""
        self._callbacks.append(callback)

    def _set(self, keys: Iterable[Hashable]) -> None:
        wanted = set(keys)
        unknown = wanted - set(self.boxes)
        if unknown:
            message = f"{self.noun}: keys {sorted(map(repr, unknown))} are not selectable."
            raise ValueError(message)
        if not self.multiple and len(wanted) != 1 and self.boxes:
            message = f"{self.noun}: a single choice needs exactly one key."
            raise ValueError(message)
        self._silent = True
        try:
            for key, box in self.boxes.items():
                box.value = key in wanted
        finally:
            self._silent = False
        self.header.summary = self._summary()

    def _summary(self) -> str:
        count = sum(bool(box.value) for box in self.boxes.values()) if self.boxes else 0
        return f"{self.noun}: {count} of {len(self.boxes)} selected"

    def _changed(self, key: Hashable, change: dict[str, object]) -> None:
        if self._silent:
            return
        if not self.multiple:
            if not change["new"]:
                # A single choice cannot be emptied: pressing the selected button again keeps it selected.
                self._silent = True
                try:
                    self.boxes[key].value = True
                finally:
                    self._silent = False
                self._close_after_pick()
                return
            self._silent = True
            try:
                for other, box in self.boxes.items():
                    if other != key:
                        box.value = False
            finally:
                self._silent = False
            self._close_after_pick()
        self._notify()

    def _close_after_pick(self) -> None:
        if self._close_on_pick:
            self.header.value = False

    def _notify(self) -> None:
        self.header.summary = self._summary()
        for callback in tuple(self._callbacks):
            callback()


@dataclass(frozen=True, slots=True)
class RunOption:
    """
    One selectable run: the name its option shows, its architecture family from the saved config, its key, and a metadata issue.

    ``name`` is what the caller lists the run as and what a single-choice header shows (the Evaluation workspace: its short
    label, the full name stays in its details); ``parent`` is the key of the run this one continues (a B or A1 run its A0),
    which it stands indented below when that run is listed in the same group.
    """

    name: str
    family: str | None
    key: Hashable
    issue: str | None = None
    parent: Hashable | None = None


class RunSelector(ChoiceBlock):
    """
    «Runs: k of n selected» (single choice: «Run: <name>») with one heading per architecture in the order FNO, U-NO, RNO, PI-FNO, PI-U-NO.

    Each run stands under the name its caller gives (the Evaluation workspace: its unique short label), one per line,
    left-aligned, in name order. A run with a listed ``parent`` stands indented directly below it; one whose parent is
    not listed stands in the name order unindented. A run whose family is missing or unknown stays visible and disabled
    under «Missing metadata» with its issue; it never makes the selector fail.
    """

    def __init__(
        self, runs: Sequence[RunOption], *, multiple: bool, selected: Iterable[Hashable] = (), opened: bool = False, close_on_pick: bool = False
    ) -> None:
        """Create the selector over ``runs`` with ``selected`` chosen (a single choice defaults to its first run); see ``ChoiceBlock``."""
        self._labels = {run.key: run.name for run in runs}  # read by ``_summary``, which the base constructor calls
        known = {spec.family: spec.label for spec in presentation.visual_semantics.ARCHITECTURES}
        groups: list[Group] = []
        indented: list[Hashable] = []
        for spec in presentation.visual_semantics.ARCHITECTURES:
            members = sorted((run for run in runs if run.family == spec.family), key=lambda run: run.name)
            anchors = {run.key for run in members if run.parent is None}
            placed: list[RunOption] = []
            for root in (run for run in members if run.parent not in anchors):
                children = [run for run in members if root.key in anchors and run.parent == root.key]
                placed.extend([root, *children])
                indented.extend(child.key for child in children)
            if placed:
                groups.append((spec.label, [(run.name, run.key) for run in placed]))
        missing = [
            (run.name, run.issue or ("architecture missing" if run.family is None else f"unknown architecture {run.family!r}"))
            for run in runs
            if run.family not in known
        ]
        super().__init__(
            "Runs",
            groups,
            multiple=multiple,
            selected=selected,
            disabled_groups=[(MISSING_METADATA_GROUP, missing)] if missing else (),
            per_line=True,
            indented=indented,
            opened=opened,
            close_on_pick=close_on_pick,
        )

    def _summary(self) -> str:
        chosen = self.selected()
        return super()._summary() if self.multiple or not chosen else f"Run: {self._labels[chosen[0]]}"


class DatasetSelector(ChoiceBlock):
    """
    «Datasets: k of n selected» with one heading per group of the inventory (form 2, by role).

    Unavailable items stand disabled under «Unavailable» with their reason. A poolable inventory gets the switch
    «Pool materials per role» as the last body row: enabled when ``pool_reason`` is None, else disabled
    with that one-line reason. ``choices()`` resolves the selection into the datasets the view draws.
    """

    def __init__(
        self,
        inventory: DatasetInventory,
        *,
        multiple: bool,
        selected: Iterable[Hashable],
        pooled: bool = False,
        pool_reason: str | None = None,
    ) -> None:
        """Create the selector; ``pooled`` takes effect only where the switch is enabled."""
        self.inventory = inventory
        self.pool_toggle: widgets.ToggleButton | None = None
        footer: list[widgets.Widget] = []
        if inventory.poolable:
            self.pool_toggle = controls.toggle(POOL_LABEL, value=pooled and pool_reason is None)
            self.pool_toggle.disabled = pool_reason is not None
            elements: list[widgets.Widget] = [self.pool_toggle]
            if pool_reason is not None:
                self.pool_toggle.tooltip = pool_reason
                elements.append(widgets.HTML(escape(pool_reason), layout=widgets.Layout(width=f"{style.GRID.snap(style.text_px(pool_reason))}px")))
            footer.append(panel.row(*elements))
        groups = [
            (inventory.group_labels[group], [(item.label, item.key) for item in inventory.items if item.group == group])
            for group in inventory.group_order
        ]
        unavailable = [(item.label, reason) for item, reason in inventory.unavailable]
        super().__init__(
            "Datasets",
            [group for group in groups if group[1]],
            multiple=multiple,
            selected=selected,
            disabled_groups=[(UNAVAILABLE_GROUP, unavailable)] if unavailable else (),
            footer=footer,
        )
        if self.pool_toggle is not None:
            self.pool_toggle.observe(lambda _change: None if self._silent else self._notify(), names="value")

    @property
    def pool_enabled(self) -> bool:
        """Return whether this view offers the pooling switch enabled."""
        return self.pool_toggle is not None and not self.pool_toggle.disabled

    @property
    def pooled(self) -> bool:
        """Return whether the selected materials of each group are drawn as one dataset."""
        return self.pool_enabled and bool(self.pool_toggle is not None and self.pool_toggle.value)

    def choices(self) -> tuple[DatasetChoice, ...]:
        """Return the datasets the view draws, pooled per group only where the switch is enabled and on."""
        return presentation.dataset_selection.resolve_choices(self.inventory, self.selected(), pooled=self.pooled)

    def set_state(self, selected: Iterable[Hashable], pooled: bool) -> None:
        """Set the selection and the switch without notifying observers; ``pooled`` is ignored where the switch is disabled."""
        self._set(selected)
        if self.pool_enabled and self.pool_toggle is not None:
            self._silent = True
            try:
                self.pool_toggle.value = pooled
            finally:
                self._silent = False
