"""Navigate stored cases, stored times, and case counts with one field and uniform step buttons.

Responsibilities:
  - Step through given case IDs, contiguous or sparse, and replace a typed absent ID by the nearest one with a warning
  - Step through irregular stored times by positions and snap a typed time to the nearest stored time with a warning
  - Show a case's long label and the selected state time on the view's status line
  - Change a case count with a slider and decrement and increment buttons
  - Disable every step button that would leave the available range

Design principles:
  - Steps move by stored positions (one or ten), clamped at the ends, never jumping to the end
  - Callbacks fire once per accepted change, never for a re-entered identical value
  - Rebinding keeps a still valid selection and names a replaced one on the status line

This module does NOT:
  - Interpolate times, load cases, or render figures
  - Compose rows or panels
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final

import ipywidgets as widgets

from . import analysis_ui_controls as controls
from . import analysis_ui_style as style

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

CASE_STEPS: Final = (("◀", -1), ("▶", 1))
TIME_STEPS: Final = (("◀◀", -10), ("◀", -1), ("▶", 1), ("▶▶", 10))
COUNT_STEPS: Final = (("\N{MINUS SIGN}", -1), ("+", 1))
_COUNT_SLIDER_PX: Final = 220  # slider track plus its readout
_STEP_FONT: Final = "11px"


@dataclass(frozen=True, slots=True)
class TimeStepSelection:
    """Describe one selected stored snapshot and its physical time."""

    position: int
    display_index: int
    physical_time: float


def _step_button(text: str, tooltip: str) -> widgets.Button:
    """Return one step button of the uniform size."""
    button = widgets.Button(
        description=text,
        tooltip=tooltip,
        layout=widgets.Layout(width=f"{style.BUTTON_PX}px", height=f"{style.LINE_PX}px", padding="0", flex="0 0 auto"),
    )
    button.style.font_size = _STEP_FONT
    return button


def _compound(first: widgets.DOMWidget, buttons: Sequence[widgets.Button]) -> widgets.HBox:
    """Return the field and its step buttons as one grid-rounded row element."""
    box = widgets.HBox((first, *buttons))
    box.add_class("glp-step")
    total = controls.element_width(first) + len(buttons) * (style.BUTTON_PX + style.STEP_GAP_PX)
    box.layout = widgets.Layout(width=f"{total}px", flex="0 0 auto")
    controls.fit_element(box)
    return box


def _field_width(label_px: int, texts: Sequence[str]) -> int:
    """Return the width of a number field with a label box of ``label_px`` and values like ``texts``."""
    control = max(style.text_px(text, allowance_ch=style.FIELD_ALLOWANCE_CH) for text in texts)
    return label_px + (style.LABEL_GAP_PX if label_px else 0) + control


class _PositionNavigator:
    """Hold one field and step buttons over an ordered tuple of available values."""

    def __init__(
        self,
        values: tuple[Any, ...],
        field: widgets.IntText | widgets.FloatText,
        steps: Sequence[tuple[str, int]],
        *,
        position: int,
        status: controls.StatusLine | None,
    ) -> None:
        self._values = values
        self.field = field
        self._status = status
        self._position = position
        self._disabled = False
        self._updating = False
        self._buttons: list[tuple[widgets.Button, int]] = []
        for text, delta in steps:
            direction = "back" if delta < 0 else "forward"
            button = _step_button(text, f"{abs(delta)} stored position{'s' if abs(delta) > 1 else ''} {direction}")
            button.on_click(lambda _button, delta=delta: self.step(delta))
            self._buttons.append((button, delta))
        field.observe(self._typed, names="value")
        self.widget = _compound(field, [button for button, _delta in self._buttons])
        self._sync()

    @property
    def position(self) -> int:
        """Return the zero-based position of the selected value."""
        return self._position

    @property
    def disabled(self) -> bool:
        """Return whether the whole control is disabled."""
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool) -> None:
        self._disabled = bool(value)
        self._sync()

    def step(self, delta: int) -> None:
        """Move by ``delta`` stored positions, clamped at the ends."""
        if isinstance(delta, bool) or not isinstance(delta, int):
            message = "A navigator step must be an integer number of positions."
            raise TypeError(message)
        self._accept(max(0, min(len(self._values) - 1, self._position + delta)), None)

    def _resolve(self, requested: float) -> tuple[int, str | None]:
        raise NotImplementedError

    def _note(self) -> str:
        """Return the normal status text of the current selection."""
        return ""

    def _emit(self) -> None:
        raise NotImplementedError

    def _typed(self, change: dict[str, Any]) -> None:
        if self._updating:
            return
        try:
            requested = float(change["new"])
        except (TypeError, ValueError):
            self._sync()
            return
        self._accept(*self._resolve(requested))

    def _accept(self, position: int, warning: str | None) -> None:
        changed = position != self._position
        self._position = position
        self._sync()
        self._report(warning)
        if changed:
            self._emit()

    def _report(self, warning: str | None) -> None:
        if self._status is None:
            return
        note = self._note()
        if warning:
            self._status.set(f"{warning} {note}".strip(), "warning")
        else:
            self._status.set(note)

    def _rebound(self, values: tuple[Any, ...], position: int, warning: str | None, field_texts: Sequence[str], *, report: bool = True) -> None:
        """Install new values, widen the field when needed, and report the selection (a replaced one with ``warning``) unless ``report`` is false."""
        self._values = values
        self._position = position
        label = controls.element_label(self.field) or 0
        needed = _field_width(label, field_texts)
        if needed > controls.element_width(self.field):
            rest = controls.element_width(self.widget) - controls.element_width(self.field)
            self.field.layout.width = f"{needed}px"
            self.widget.layout.width = f"{needed + rest}px"
            controls.fit_element(self.widget)
        self._sync()
        if report:
            self._report(warning)

    def _sync(self) -> None:
        self._updating = True
        try:
            self.field.value = self._values[self._position]
        finally:
            self._updating = False
        self.field.disabled = self._disabled
        last = len(self._values) - 1
        for button, delta in self._buttons:
            at_edge = self._position == 0 if delta < 0 else self._position == last
            button.disabled = self._disabled or at_edge


def _nearest_case(case_ids: Sequence[int], requested: float) -> int:
    """Return the position of the nearest case ID, the smaller one on a tie."""
    return min(range(len(case_ids)), key=lambda index: (abs(case_ids[index] - requested), case_ids[index]))


def _case_ids(case_ids: Sequence[int]) -> tuple[int, ...]:
    """Return validated unique integer case IDs."""
    ids = tuple(case_ids)
    if not ids or len(set(ids)) != len(ids) or any(isinstance(case, bool) or not isinstance(case, int) for case in ids):
        message = "A case navigator needs at least one integer case ID and no repeated ID."
        raise ValueError(message)
    return ids


class CaseNavigator(_PositionNavigator):
    """
    Step through given case IDs with ◀ ▶; a typed ID that is not available selects the nearest one.

    ``labels`` maps an ID to its long description, shown on ``status`` instead of in a wide dropdown (Q18).
    ``on_change`` receives the newly selected ID once per accepted change.
    """

    def __init__(
        self,
        case_ids: Sequence[int],
        *,
        value: int | None = None,
        description: str = "Case",
        labels: Mapping[int, str] | None = None,
        status: controls.StatusLine | None = None,
        on_change: Callable[[int], None] | None = None,
    ) -> None:
        """Create the navigator on ``value`` or the first ID."""
        ids = _case_ids(case_ids)
        if value is not None and value not in ids:
            message = f"Case {value} is not one of the navigator's case IDs."
            raise ValueError(message)
        self._labels = dict(labels or {})
        self._on_change = on_change
        label = style.text_px(description) if description else 0
        field = widgets.IntText(
            value=ids[0] if value is None else value,
            description=description,
            continuous_update=False,
            style={"description_width": f"{label}px"},
            layout=widgets.Layout(width=f"{_field_width(label, [str(case) for case in ids])}px", flex="0 0 auto"),
        )
        super().__init__(ids, field, CASE_STEPS, position=0 if value is None else ids.index(value), status=status)
        self._report(None)

    @property
    def value(self) -> int:
        """Return the selected case ID."""
        return int(self._values[self._position])

    @property
    def case_ids(self) -> tuple[int, ...]:
        """Return the available case IDs."""
        return self._values

    def rebind(self, case_ids: Sequence[int], *, preserve: bool = True, notify: bool = True, labels: Mapping[int, str] | None = None) -> int:
        """
        Replace the available IDs and return the selected one.

        With ``preserve`` a still available ID stays selected and a removed one is replaced by the nearest available ID
        with a warning; without it the first ID is selected. ``notify`` calls ``on_change`` once with the result.
        """
        ids = _case_ids(case_ids)
        previous = self.value
        if labels is not None:
            self._labels = dict(labels)
        warning = None
        if not preserve:
            position = 0
        elif previous in ids:
            position = ids.index(previous)
        else:
            position = _nearest_case(ids, previous)
            warning = f"Case {previous} is no longer available; showing the nearest case {ids[position]}."
        self._rebound(ids, position, warning, [str(case) for case in ids])
        if notify:
            self._emit()
        return self.value

    def _resolve(self, requested: float) -> tuple[int, str | None]:
        typed = round(requested)
        if typed in self._values:
            return self._values.index(typed), None
        position = _nearest_case(self._values, typed)
        return position, f"Case {typed} is not available; showing the nearest case {self._values[position]}."

    def _note(self) -> str:
        return self._labels.get(self.value, "")

    def _emit(self) -> None:
        if self._on_change is not None:
            self._on_change(self.value)


def _stored_times(times_h: Sequence[float]) -> tuple[float, ...]:
    """Return finite, strictly increasing stored times."""
    stored = tuple(float(time) for time in times_h)
    if not stored or not all(math.isfinite(time) for time in stored) or any(later <= earlier for earlier, later in pairwise(stored)):
        message = "A time navigator needs finite, strictly increasing stored times."
        raise ValueError(message)
    return stored


def _nearest_time(times: Sequence[float], requested: float) -> int:
    """Return the position of the nearest stored time, the earlier one on a tie."""
    return min(range(len(times)), key=lambda index: (abs(times[index] - requested), index))


class TimeNavigator(_PositionNavigator):
    """
    Step through irregular stored times with ◀◀ ◀ ▶ ▶▶ (ten and one positions); a typed time snaps to the nearest stored one.

    The status line shows the selected state time, and a warning when a typed time was snapped (rule 15).
    ``on_change`` receives the new ``TimeStepSelection`` once per accepted change.
    """

    def __init__(
        self,
        times_h: Sequence[float],
        *,
        value: float | None = None,
        description: str = "Time [h]",
        status: controls.StatusLine | None = None,
        on_change: Callable[[TimeStepSelection], None] | None = None,
    ) -> None:
        """Create the navigator on the stored time nearest to ``value``, or on the first one."""
        stored = _stored_times(times_h)
        self._on_change = on_change
        label = style.text_px(description) if description else 0
        field = widgets.FloatText(
            value=stored[0],
            description=description,
            continuous_update=False,
            style={"description_width": f"{label}px"},
            layout=widgets.Layout(width=f"{_field_width(label, [f'{time:g}' for time in stored])}px", flex="0 0 auto"),
        )
        position = 0 if value is None else _nearest_time(stored, float(value))
        super().__init__(stored, field, TIME_STEPS, position=position, status=status)
        self._report(None)

    @property
    def value(self) -> float:
        """Return the selected stored time in hours."""
        return float(self._values[self._position])

    @property
    def times_h(self) -> tuple[float, ...]:
        """Return the stored times."""
        return self._values

    @property
    def selection(self) -> TimeStepSelection:
        """Return the selected position, its one-based index, and its time."""
        return TimeStepSelection(position=self._position, display_index=self._position + 1, physical_time=self.value)

    def set_callback(self, callback: Callable[[TimeStepSelection], None] | None) -> None:
        """Replace the change callback without changing the selection."""
        self._on_change = callback

    def clear_status(self) -> None:
        """Clear the status line only while it shows this navigator's own time text, never another control's message on it."""
        if self._status is not None and self._status.text.endswith(self._note()):
            self._status.clear()

    def rebind(self, times_h: Sequence[float], *, preserve_time: float | None = None, notify: bool = True, report: bool = True) -> TimeStepSelection:
        """
        Replace the stored times, select the one nearest to ``preserve_time`` (default: the current time), and notify once.

        Without ``report`` (a hidden navigator) neither the time nor a snap warning reaches the status line, so another control's message stays.
        """
        stored = _stored_times(times_h)
        target = self.value if preserve_time is None else float(preserve_time)
        position = _nearest_time(stored, target)
        warning = None if stored[position] == target else f"Time {target:g} h is not stored; showing the nearest stored time {stored[position]:g} h."
        self._rebound(stored, position, warning, [f"{time:g}" for time in stored], report=report)
        if notify:
            self._emit()
        return self.selection

    def _resolve(self, requested: float) -> tuple[int, str | None]:
        if not math.isfinite(requested):
            return self._position, f"Time must be a finite number; kept {self.value:g} h."
        position = _nearest_time(self._values, requested)
        if self._values[position] != requested:
            return position, f"Typed {requested:g} h snapped to the nearest stored time {self._values[position]:g} h."
        return position, None

    def _note(self) -> str:
        return f"Time {self.value:g} h"

    def _emit(self) -> None:
        if self._on_change is not None:
            self._on_change(self.selection)


class CountControl:
    """Choose a count with a slider and decrement and increment buttons; each button is disabled at its bound."""

    def __init__(
        self,
        *,
        value: int,
        minimum: int,
        maximum: int,
        step: int = 1,
        description: str = "Cases",
        on_change: Callable[[int], None] | None = None,
    ) -> None:
        """Create the control; ``on_change`` receives the new count."""
        if step < 1 or minimum > maximum or not minimum <= value <= maximum:
            message = f"A count control needs step >= 1 and minimum <= value <= maximum, got {minimum} <= {value} <= {maximum}."
            raise ValueError(message)
        self._on_change = on_change
        self._disabled = False
        label = style.text_px(description) if description else 0
        self.slider = widgets.IntSlider(
            value=value,
            min=minimum,
            max=maximum,
            step=step,
            description=description,
            continuous_update=False,
            style={"description_width": f"{label}px"},
            layout=widgets.Layout(width=f"{label + style.LABEL_GAP_PX + _COUNT_SLIDER_PX}px", flex="0 0 auto"),
        )
        self.decrement = _step_button(COUNT_STEPS[0][0], f"{step} fewer")
        self.increment = _step_button(COUNT_STEPS[1][0], f"{step} more")
        self.decrement.on_click(lambda _button: self._move(-step))
        self.increment.on_click(lambda _button: self._move(step))
        self.slider.observe(self._changed, names="value")
        self.widget = _compound(self.slider, [self.decrement, self.increment])
        self._sync()

    @property
    def value(self) -> int:
        """Return the current count."""
        return int(self.slider.value)

    @property
    def disabled(self) -> bool:
        """Return whether the whole control is disabled."""
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool) -> None:
        self._disabled = bool(value)
        self._sync()

    def _move(self, delta: int) -> None:
        self.slider.value = min(max(self.slider.value + delta, self.slider.min), self.slider.max)

    def _changed(self, _change: object) -> None:
        self._sync()
        if self._on_change is not None:
            self._on_change(self.value)

    def _sync(self) -> None:
        self.slider.disabled = self._disabled
        self.decrement.disabled = self._disabled or self.slider.value <= self.slider.min
        self.increment.disabled = self._disabled or self.slider.value >= self.slider.max
