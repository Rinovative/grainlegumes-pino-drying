"""Construct the primitive controls of the notebook control standard and size them on the column grid.

Responsibilities:
  - Build dropdowns, toggles, checkboxes, one-line choice groups, and action buttons sized from their full text
  - Provide the shared control vocabularies (error mode, scale mode, kappa scale, scope, display stride, channel, scale lock)
  - Keep direct channel checkboxes whose rebinding retains valid choices and names removed ones
  - Show one wrapping status line, one failure display with a collapsed traceback, and guard busy actions
  - Measure, label-align, and round every element to whole grid columns

Design principles:
  - Widths come from generous text estimates rounded up to the column grid, never from fixed per-site numbers
  - Every element carries a pixel width, so panels can align labels and check grid membership
  - A failure or busy action never leaves buttons disabled or a stale figure in the export state

This module does NOT:
  - Compose rows, panels, tabs, or notebook shells
  - Navigate cases or times, or resolve run and dataset selections
  - Render figures or write export files
"""

from __future__ import annotations

import html
import logging
import traceback
from typing import TYPE_CHECKING, Any, Final, Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.analysis import presentation

from . import analysis_ui_style as style

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

Level = Literal["normal", "warning", "error"]
ActionKind = Literal["open", "close", "export", "compute"]
Option = str | tuple[str, Any]

LEVELS: Final[tuple[Level, ...]] = ("normal", "warning", "error")
COMPUTE_LABEL: Final = "Berechnen"
ACTION_LABELS: Final = ("Close", "Export PDF", "Export GIF", COMPUTE_LABEL)

_LOGGER = logging.getLogger(__name__)
_CHECKBOX_ALLOWANCE_CH: Final = 3  # the box, its gap to the text, and a margin
_BUTTON_STYLES: Final[Mapping[str, str]] = {"open": "primary", "close": "danger", "export": "", "compute": "primary"}
_ACTION_PX: Final = style.GRID.snap(max(style.text_px(label, allowance_ch=style.BUTTON_ALLOWANCE_CH) for label in ACTION_LABELS))
_EXPORT_FIELDS: Final = ("fig", "figures", "filename_stem")
_DEFAULT_CHANNELS: Final = ("p", "u", "v", "|u|")
_NORMALIZED_ERROR: Final = "Normalized absolute error"
_SIGNED_ERROR: Final = "Signed error"


# ----------------------------------------------------------------------------- element geometry


def element_width(widget: widgets.DOMWidget) -> int:
    """Return the pixel width an element declares; raise when it declares none."""
    width = widget.layout.width
    if not isinstance(width, str) or not width.endswith("px"):
        message = f"{type(widget).__name__} declares no pixel width; build panel elements with the ui constructors."
        raise TypeError(message)
    return round(float(width.removesuffix("px")))


def element_label(widget: widgets.Widget) -> int | None:
    """Return the label box width of a described element or of a compound element's first child, else None."""
    if isinstance(widget, widgets.Box):
        return element_label(widget.children[0]) if widget.children else None
    if not getattr(widget, "description", ""):
        return None
    description_width = getattr(getattr(widget, "style", None), "description_width", "")
    if isinstance(description_width, str) and description_width.endswith("px"):
        return round(float(description_width.removesuffix("px")))
    return None


def fit_element(widget: widgets.DOMWidget, *, label: int | None = None) -> None:
    """
    Widen the element's first label to ``label`` when given and round its width up to whole grid columns.

    A compound element (a box) gives the extra width to its first child, so its step buttons or further
    checkboxes keep their own widths and the compound ends on a column line.
    """
    if isinstance(widget, widgets.Box) and widget.children:
        first = widget.children[0]
        rest = element_width(widget) - element_width(first)
        needed = _needed_width(first, label) + rest
        total = style.GRID.snap(needed)
        _relabel(first, label)
        first.layout.width = f"{total - rest}px"
        widget.layout.width = f"{total}px"
        return
    needed = _needed_width(widget, label)
    _relabel(widget, label)
    widget.layout.width = f"{style.GRID.snap(needed)}px"


def _needed_width(widget: widgets.DOMWidget, label: int | None) -> int:
    """Return the width the element needs with its label box widened to ``label``."""
    current = element_label(widget)
    if label is None or current is None:
        return element_width(widget)
    return element_width(widget) + max(0, label - current)


def _relabel(widget: widgets.Widget, label: int | None) -> None:
    """Set the label box of a described element to ``label`` when it is wider."""
    current = element_label(widget)
    if label is not None and current is not None and label > current:
        widget.style.description_width = f"{label}px"  # type: ignore[attr-defined]


def _described_layout(description: str, control_px: int) -> tuple[dict[str, str], widgets.Layout]:
    """Return the label style and the grid-rounded layout of a control with an optional left label."""
    label = style.text_px(description) if description else 0
    gap = style.LABEL_GAP_PX if description else 0
    width = style.GRID.snap(label + gap + control_px)
    return {"description_width": f"{label}px"}, widgets.Layout(width=f"{width}px", flex="0 0 auto")


def _fixed_layout(width_px: int) -> widgets.Layout:
    """Return the grid-rounded layout of an element without a left label."""
    return widgets.Layout(width=f"{style.GRID.snap(width_px)}px", flex="0 0 auto")


def _option_label(option: Option) -> str:
    """Return the visible text of one option."""
    return option[0] if isinstance(option, tuple) else option


def _option_value(option: Option) -> Any:
    """Return the value of one option."""
    return option[1] if isinstance(option, tuple) else option


def _require_options(options: Sequence[Option], *, kind: str) -> list[Option]:
    """Return the options as a list and refuse an empty or repeated set."""
    resolved = list(options)
    values = [_option_value(option) for option in resolved]
    if not resolved or len(set(map(repr, values))) != len(values):
        message = f"A {kind} needs at least one option and no repeated values."
        raise ValueError(message)
    return resolved


def _resolved_value(options: Sequence[Option], value: Any) -> Any:
    """Return ``value`` or the first option's value, refusing a value outside the options."""
    values = [_option_value(option) for option in options]
    if value is None:
        return values[0]
    if value not in values:
        message = f"Value {value!r} is not one of the options {values!r}."
        raise ValueError(message)
    return value


# ----------------------------------------------------------------------------- primitive controls


def dropdown_width(description: str, texts: Iterable[str]) -> int:
    """Return the grid width of a dropdown whose label is ``description`` and whose longest entry is in ``texts``."""
    labels = list(texts)
    widest = max((style.text_px(text, allowance_ch=style.DROPDOWN_ALLOWANCE_CH) for text in labels), default=0)
    label = style.text_px(description) + style.LABEL_GAP_PX if description else 0
    return style.GRID.snap(label + widest)


def dropdown(
    options: Sequence[Option],
    *,
    description: str = "",
    value: Any = None,
    width_texts: Sequence[str] | None = None,
) -> widgets.Dropdown:
    """
    Return a dropdown as wide as its label plus its longest entry, rounded up to whole grid columns (F1).

    ``width_texts`` adds texts the dropdown may show later (for example every view title of a notebook), so the width
    covers them as well.
    """
    resolved = _require_options(options, kind="dropdown")
    texts = [_option_label(option) for option in resolved] + list(width_texts or ())
    label = style.text_px(description) if description else 0
    return widgets.Dropdown(
        options=resolved,
        value=_resolved_value(resolved, value),
        description=description,
        style={"description_width": f"{label}px"},
        layout=widgets.Layout(width=f"{dropdown_width(description, texts)}px", flex="0 0 auto"),
    )


def toggle(description: str, *, value: bool = False) -> widgets.ToggleButton:
    """Return a toggle button as wide as its full text."""
    return widgets.ToggleButton(
        value=value,
        description=description,
        tooltip=description,
        layout=_fixed_layout(style.text_px(description, allowance_ch=style.BUTTON_ALLOWANCE_CH)),
    )


def checkbox(description: str, *, value: bool = True, tooltip: str = "") -> widgets.Checkbox:
    """Return a checkbox whose label is never cut: its width comes from the full text."""
    return widgets.Checkbox(
        value=value,
        description=description,
        indent=False,
        tooltip=tooltip or description,
        layout=_fixed_layout(style.text_px(description, allowance_ch=_CHECKBOX_ALLOWANCE_CH)),
    )


def choice_group(options: Sequence[Option], *, description: str = "", value: Any = None) -> widgets.ToggleButtons:
    """Return a one-line group of equally wide buttons for a small exclusive choice (rule 18)."""
    resolved = _require_options(options, kind="choice group")
    button_px = max(style.text_px(_option_label(option), allowance_ch=style.BUTTON_ALLOWANCE_CH) for option in resolved)
    label_style, layout = _described_layout(description, len(resolved) * button_px)
    return widgets.ToggleButtons(
        options=resolved,
        value=_resolved_value(resolved, value),
        description=description,
        style={**label_style, "button_width": f"{button_px}px"},
        layout=layout,
    )


def error_mode_group(task: Literal["steady", "transient"]) -> widgets.ToggleButtons:
    """Return the error-map choice of a steady or a transient view."""
    if task == "steady":
        return choice_group((_SIGNED_ERROR, "MAE", "Relative [%]", _NORMALIZED_ERROR), description="Error mode", value=_SIGNED_ERROR)
    if task == "transient":
        return choice_group((_SIGNED_ERROR, "Absolute error", _NORMALIZED_ERROR), description="Error mode", value=_SIGNED_ERROR)
    message = f"Error modes exist for 'steady' and 'transient', got {task!r}."
    raise ValueError(message)


def scale_mode_group() -> widgets.ToggleButtons:
    """Return the prediction colour-scale choice: independent scales or the ground-truth scale."""
    return choice_group(("Independent", "Shared (GT)"), description="Pred/GT scale")


def kappa_scale_group() -> widgets.ToggleButtons:
    """Return the permeability scale choice."""
    return choice_group(("kappa", "log10(kappa)"), description="Kappa scale", value="log10(kappa)")


def scope_group(value: Literal["aggregate", "single"] = "aggregate") -> widgets.ToggleButtons:
    """Return the Aggregate or Single case choice."""
    return choice_group((("Aggregate", "aggregate"), ("Single case", "single")), value=value)


def display_stride_dropdown() -> widgets.Dropdown:
    """Return the presentation-only map sampling choice; the full grid stays in the data."""
    return dropdown((("Auto", "auto"), ("Full", 1), ("Stride 2", 2), ("Stride 4", 4)), description="Display stride", value="auto")


def channel_dropdown(channels: Sequence[str] = _DEFAULT_CHANNELS, *, default: str = "|u|") -> widgets.Dropdown:
    """Return one output-channel choice labelled with the canonical field labels."""
    return dropdown(
        [(presentation.field_labels.field_label(channel, control=True), channel) for channel in channels], description="Channel", value=default
    )


def scale_lock_checkbox(*, value: bool, description: str = "Lock color scale") -> widgets.Checkbox:
    """Return the colour-scale lock of a map view."""
    if not isinstance(value, bool):
        message = "A colour-scale lock state must be boolean."
        raise TypeError(message)
    return checkbox(description, value=value)


def action_button(label: str, *, kind: ActionKind) -> widgets.Button:
    """
    Return an action button: open primary, close danger, export neutral, compute primary.

    Close, Export PDF, Export GIF and Berechnen share one width; an open button carries the panel title and is as wide
    as that title, rounded to the grid.
    """
    if kind not in _BUTTON_STYLES:
        message = f"Action kind must be one of {sorted(_BUTTON_STYLES)}, got {kind!r}."
        raise ValueError(message)
    own = style.GRID.snap(style.text_px(label, allowance_ch=style.BUTTON_ALLOWANCE_CH))
    width = own if kind == "open" else max(own, _ACTION_PX)
    return widgets.Button(
        description=label,
        tooltip=label,
        button_style=_BUTTON_STYLES[kind],
        layout=widgets.Layout(width=f"{width}px", flex="0 0 auto"),
    )


# ----------------------------------------------------------------------------- channel checkboxes


class ChannelChecks:
    """
    Show up to six short channel names as direct checkboxes in one row element (F3).

    ``widget`` is the row element: the title, then one checkbox per channel. ``rebind`` replaces the channel set in
    place, keeps every still valid choice, checks channels that are new to the control, and returns the removed
    choices so the view can name them in its status (rule 5).
    """

    def __init__(self, options: Sequence[str], *, labels: Mapping[str, str] | None = None, title: str = "Channels") -> None:
        """Create the control with every channel checked."""
        self._title = widgets.Label(title, layout=_fixed_layout(style.text_px(title)))
        self._title.add_class("glp-label")
        self._callbacks: list[Callable[[], None]] = []
        self._known: tuple[str, ...] = ()
        self.boxes: dict[str, widgets.Checkbox] = {}
        self.widget = widgets.HBox()
        self.widget.add_class("glp-checks")
        self._build(options, labels, checked=None)

    @property
    def selected(self) -> tuple[str, ...]:
        """Return the checked channels in option order."""
        return tuple(name for name, box in self.boxes.items() if box.value)

    @property
    def options(self) -> tuple[str, ...]:
        """Return the current channels in option order."""
        return self._known

    def observe(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after every change of the checked channels."""
        self._callbacks.append(callback)

    def rebind(
        self,
        options: Sequence[str],
        *,
        labels: Mapping[str, str] | None = None,
        retained: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """
        Replace the channels and return the retained choices that the new set no longer offers.

        ``retained`` defaults to the current selection. Observers are not called; the caller redraws once.
        """
        kept = tuple(self.selected if retained is None else retained)
        new = tuple(options)
        checked = {name for name in new if name in kept or name not in self._known}
        self._build(new, labels, checked=checked)
        return tuple(name for name in kept if name not in new)

    def _build(self, options: Sequence[str], labels: Mapping[str, str] | None, *, checked: set[str] | None) -> None:
        """Create one checkbox per channel and lay the row element out on the grid."""
        resolved = tuple(options)
        if not resolved or len(set(resolved)) != len(resolved):
            message = "Channel checkboxes need at least one channel and no repeated channel."
            raise ValueError(message)
        names = dict(labels or {})
        if not set(names) <= set(resolved):
            message = "Channel labels must be keyed by offered channels."
            raise ValueError(message)
        for old in self.boxes.values():
            old.unobserve(self._changed, names="value")
        self.boxes = {
            name: checkbox(
                names.get(name, presentation.field_labels.field_label(name, control=True)), value=checked is None or name in checked, tooltip=name
            )
            for name in resolved
        }
        for box in self.boxes.values():
            box.observe(self._changed, names="value")
        self._known = resolved
        children = (self._title, *self.boxes.values())
        self.widget.children = children
        total = sum(element_width(child) for child in children) + style.GAP_PX * (len(children) - 1)
        self.widget.layout = widgets.Layout(width=f"{total}px", flex="0 0 auto")

    def _changed(self, _change: object = None) -> None:
        """Notify observers once per user change."""
        for callback in tuple(self._callbacks):
            callback()


# ----------------------------------------------------------------------------- status, failure, busy state


class StatusLine:
    """One status line: normal, warning or error text that wraps instead of being cut, and textual progress."""

    def __init__(self, text: str = "", level: Level = "normal") -> None:
        """Create the line with one reserved line of height."""
        self.widget = widgets.HTML()
        self.widget.add_class("glp-status")
        self._text = ""
        self._level: Level = "normal"
        self.set(text, level)

    @property
    def text(self) -> str:
        """Return the plain text currently shown."""
        return self._text

    @property
    def level(self) -> Level:
        """Return the level of the current text."""
        return self._level

    def set(self, text: str, level: Level = "normal") -> None:
        """Replace the line; a long text wraps over as many lines as it needs."""
        if level not in LEVELS:
            message = f"Status level must be one of {LEVELS}, got {level!r}."
            raise ValueError(message)
        self._apply_level(level)
        self._text = text
        self.widget.value = html.escape(text) if text else "&nbsp;"

    def clear(self) -> None:
        """Empty the line; its reserved height stays."""
        self.set("")

    def progress(self, label: str, done: int, total: int) -> None:
        """Show ``label done/total`` with a small progress bar."""
        if total <= 0 or not 0 <= done <= total:
            message = f"Progress needs 0 <= done <= total and total > 0, got {done}/{total}."
            raise ValueError(message)
        self._apply_level("normal")
        self._text = f"{label} {done}/{total}"
        self.widget.value = f'{html.escape(self._text)}<progress value="{done}" max="{total}"></progress>'

    def _apply_level(self, level: Level) -> None:
        """Mark the line with exactly one level class."""
        for name in LEVELS:
            self.widget.remove_class(f"glp-status-{name}")
        self.widget.add_class(f"glp-status-{level}")
        self._level = level


def release_export(export_state: dict[str, Any] | None) -> None:
    """Close the figure an export state holds and clear its target, so no export writes a stale figure."""
    if export_state is None:
        return
    previous = export_state.get("fig")
    if isinstance(previous, Figure):
        plt.close(previous)
    export_state.update(dict.fromkeys(_EXPORT_FIELDS))
    export_state["figures"] = ()


def failure_html(error: BaseException) -> str:
    """Return the escaped markup of a failed view: one visible summary line and the collapsed traceback."""
    detail = html.escape("".join(traceback.format_exception(error)))
    summary = html.escape(f"View failed — {type(error).__name__}: {error}")
    return f"<p><b>{summary}</b></p><details><summary>Traceback</summary><pre>{detail}</pre></details>"


class FailureDisplay:
    """Show a failed view as one visible summary line with the traceback collapsed; take no space when clear."""

    def __init__(self) -> None:
        """Create the display in its clear, hidden state."""
        self.widget = widgets.HTML()
        self.widget.add_class("glp-failure")
        self.clear()

    @property
    def shown(self) -> bool:
        """Return whether a failure is shown."""
        return self.widget.layout.display != "none"

    def show(self, error: BaseException, export_state: dict[str, Any] | None = None) -> None:
        """Show ``error`` and release the export target of the failed view, the release first so it precedes any widget change."""
        release_export(export_state)
        _LOGGER.exception("View failed: %s", error, exc_info=error)
        self.widget.value = failure_html(error)
        set_visible(self.widget, True)

    def clear(self) -> None:
        """Remove the failure; the display takes no space."""
        self.widget.value = ""
        set_visible(self.widget, False)


class BusyGuard:
    """Disable the guarded buttons while one action runs, report on the status line, and always restore them (rule 8)."""

    def __init__(self, buttons: Sequence[widgets.Button], status: StatusLine | None = None) -> None:
        """Guard ``buttons``; busy, outcome and failure texts go to ``status``."""
        self.buttons = tuple(buttons)
        self.status = status
        self.busy = False

    def run(
        self,
        action: Callable[[], str | None],
        *,
        busy_text: str = "Working …",
        on_error: Callable[[BaseException], None] | None = None,
    ) -> bool:
        """Run ``action`` unless one is running and return whether it ran; the action returns its outcome text."""
        if self.busy:
            return False
        self.busy = True
        previous = [button.disabled for button in self.buttons]
        for button in self.buttons:
            button.disabled = True
        if self.status is not None:
            self.status.set(busy_text)
        try:
            outcome = action()
        except Exception as error:
            _LOGGER.exception("Action failed")
            if self.status is not None:
                self.status.set(f"Failed — {type(error).__name__}: {error}", "error")
            if on_error is not None:
                on_error(error)
        else:
            if self.status is not None and outcome is not None:
                self.status.set(outcome)
        finally:
            for button, was_disabled in zip(self.buttons, previous, strict=True):
                button.disabled = was_disabled
            self.busy = False
        return True


def set_visible(widget: widgets.DOMWidget, visible: bool) -> None:
    """Show or hide an element; a hidden element takes no space (rule 7)."""
    widget.layout.display = None if visible else "none"
