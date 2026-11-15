"""Compose control rows, view panels, popovers, on-demand views, and tab sets of the control standard.

Responsibilities:
  - Build one-line control rows and align their first labels and element edges on the column grid (F1)
  - Reserve exactly the panel's rows, then the status line, the plot gap, and the plot slot or the failure in its place
  - Show plot results in the slot: a figure as its PNG image at its own pixel size, a widget as itself, anything else as HTML
  - Open a popover body directly below its own header's row, over the rows below, without moving the plot (F2)
  - Keep hidden views idle and coalesce the redraw requests of a batch of control changes into one (rules 4 and 12)
  - Number the render requests of a view so only the newest publishes, and bind a nested view to its host's render (rule 3)
  - Draw cheap views at once and expensive views on «Berechnen», keeping the last results in kernel memory (rule 2)
  - Build tab sets whose headers are equally wide and fit the longest title

Design principles:
  - A popover body is a child of the panel, never of a row, so a sideways scrolled row cannot misplace or clip it
  - A panel refuses anything but rows and refuses stacked radio groups (rule 18)
  - The plot's top-left corner depends only on the panel's row count, never on what a row shows
  - No output widget and no output capture: a slot swaps its children, so a redraw sends plain widget state only

This module does NOT:
  - Construct primitive controls, navigators, or selectors
  - Write export files or own notebook shells and tab lifecycles

Notes:
  On-demand views:
    - A view is marked expensive, and then waits for «Berechnen», when its measured render time is above about 2 s.

"""

from __future__ import annotations

import html
import io
import math
import time
from collections import OrderedDict
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

import ipywidgets as widgets
from matplotlib.figure import Figure

from . import analysis_ui_controls as controls
from . import analysis_ui_style as style

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence
    from types import ModuleType

VIEW_CACHE_ENTRIES: Final = 8
CACHE_NOTE: Final = "aus dem Cache"
_DISCLOSURE_ALLOWANCE_CH: Final = 2
_TAB_CLASS_PREFIX: Final = "glp-tabw-"


class Row(widgets.HBox):
    """One control row: elements side by side, the first flush left; it never wraps and scrolls when too wide."""

    def __init__(self, elements: Sequence[widgets.Widget]) -> None:
        """Create the row from its elements."""
        super().__init__(tuple(elements))
        self.add_class("glp-row")


def row(*elements: widgets.Widget) -> Row:
    """Return one control row of ``elements``."""
    return Row(elements)


def figure_image(figure: Figure) -> widgets.Image:
    """Return ``figure`` as a PNG image at its own dpi with a tight bounding box, as IPython's inline display encodes it."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=figure.dpi)
    # Not stretched across the slot: the image keeps its pixel size, and the slot scrolls when it is wider.
    return widgets.Image(value=buffer.getvalue(), format="png", layout=widgets.Layout(align_self="flex-start"))


def text_html(text: str) -> widgets.HTML:
    """Return ``text`` escaped in a preformatted block."""
    block = widgets.HTML(f"<pre>{html.escape(text)}</pre>")
    block.add_class("glp-text")
    return block


def item_widget(item: Any) -> widgets.Widget:
    """
    Return the element that shows one plot item: a figure as its PNG image, a widget or view panel as itself.

    Any other item becomes an HTML element of its ``_repr_html_()`` when it has one (tables, styled frames, HTML), else
    of its escaped ``str()``.
    """
    if isinstance(item, Figure):
        return figure_image(item)
    if isinstance(item, ViewPanel):
        return item.widget
    if isinstance(item, widgets.Widget):
        return item
    render = getattr(item, "_repr_html_", None)
    markup = render() if callable(render) else None
    if not isinstance(markup, str):
        return text_html(str(item))
    rendered = widgets.HTML(markup)
    rendered.add_class("jp-RenderedHTMLCommon")  # the table styling a notebook output area gives the same markup
    return rendered


class PlotSlot:
    """The plot place of a view: a box of figure images and HTML items; a figure keeps its pixel size and the slot scrolls sideways."""

    def __init__(self) -> None:
        """Create an empty slot."""
        self.widget = widgets.VBox()
        self.widget.add_class("glp-plot")
        self._built: tuple[widgets.Widget, ...] = ()

    def show(self, items: Sequence[Any]) -> None:
        """
        Replace the slot's content by one element per item, in order; nothing changes when an item cannot be shown.

        The elements the slot built itself (figure images, HTML of tables and text) are closed once replaced, so neither
        the kernel nor the front end keeps them; widgets and view panels passed in stay open for their owner.
        """
        elements = tuple(item_widget(item) for item in items)
        built = tuple(element for item, element in zip(items, elements, strict=True) if not isinstance(item, (widgets.Widget, ViewPanel)))
        self.widget.children = elements
        replaced, self._built = self._built, built
        for element in replaced:
            for part in (element, getattr(element, "layout", None), getattr(element, "style", None)):
                if isinstance(part, widgets.Widget):
                    part.close()


class DisclosureHeader(widgets.ToggleButton):
    """The flush-left line ``▸ summary`` that opens its body; ``summary`` can change while the header is shown."""

    def __init__(self, summary: str, *, opened: bool) -> None:
        """Create the header with its summary."""
        super().__init__(value=opened, tooltip="Show or hide the details", layout=widgets.Layout(width="0px", flex="0 0 auto"))
        self.add_class("glp-disclosure")
        self._summary = summary
        self._render()
        self.observe(lambda _change: self._render(), names="value")

    @property
    def summary(self) -> str:
        """Return the summary text without the disclosure mark."""
        return self._summary

    @summary.setter
    def summary(self, text: str) -> None:
        self._summary = text
        self._render()

    def _render(self) -> None:
        self.description = f"{'▾' if self.value else '▸'} {self._summary}"
        needed = style.GRID.snap(style.text_px(f"▾ {self._summary}", allowance_ch=_DISCLOSURE_ALLOWANCE_CH))
        self.layout.width = f"{max(needed, controls.element_width(self))}px"


def disclosure(summary: str, body: widgets.DOMWidget, *, opened: bool = False) -> tuple[DisclosureHeader, widgets.DOMWidget]:
    """Return a flush ``▸/▾`` header and its body; the body is visible only while the header is open (Q16)."""
    header = DisclosureHeader(summary, opened=opened)
    controls.set_visible(body, opened)
    header.observe(lambda change: controls.set_visible(body, bool(change["new"])), names="value")
    return header, body


def _holds_radio(widget: widgets.Widget) -> bool:
    """Return whether ``widget`` is or contains a stacked radio group."""
    if isinstance(widget, widgets.RadioButtons):
        return True
    return isinstance(widget, widgets.Box) and any(_holds_radio(child) for child in widget.children)


def _contains(widget: widgets.Widget, target: widgets.Widget) -> bool:
    """Return whether ``target`` is ``widget`` or inside it."""
    if widget is target:
        return True
    return isinstance(widget, widgets.Box) and any(_contains(child, target) for child in widget.children)


def control_rows(rows: Sequence[widgets.Widget]) -> widgets.VBox:
    """
    Validate, align, and stack panel rows into a block of exactly one row pitch per row.

    The first labels of all rows get one common width, so the first controls start on one line; every element is
    rounded up to whole grid columns (F1). A row hidden later keeps its pitch, so the plot below never jumps.

    Raises
    ------
    TypeError
        If a row is not a ``Row`` or holds a stacked radio group (use ``controls.choice_group``).

    """
    checked: list[Row] = []
    for index, candidate in enumerate(rows):
        if not isinstance(candidate, Row):
            message = f"Row {index} of a panel must be built with panel.row(...), got {type(candidate).__name__}."
            raise TypeError(message)
        if _holds_radio(candidate):
            message = f"Row {index} of a panel holds a stacked radio group; use controls.choice_group for a one-line choice."
            raise TypeError(message)
        checked.append(candidate)
    firsts = [candidate.children[0] for candidate in checked if candidate.children]
    labels = [label for first in firsts if (label := controls.element_label(first)) is not None]
    common = max(labels, default=None)
    for candidate in checked:
        for index, element in enumerate(candidate.children):
            controls.fit_element(element, label=common if index == 0 else None)
    block = widgets.VBox(tuple(rows), layout=widgets.Layout(height=f"{len(rows) * style.ROW_PITCH_PX}px", flex="0 0 auto"))
    block.add_class("glp-rows")
    return block


def popover_bodies(rows: Sequence[widgets.Widget], popovers: Sequence[tuple[widgets.ToggleButton, widgets.DOMWidget]]) -> tuple[widgets.Widget, ...]:
    """
    Place each popover body directly below its header's row and return the bodies to add to the panel (F2).

    A body gets ``left: 0`` and ``top = (row index + 1) * row pitch``; opening one body closes the others.

    Raises
    ------
    ValueError
        If a header is in none of ``rows``.

    """
    headers = [header for header, _body in popovers]
    bodies: list[widgets.Widget] = []
    for header, body in popovers:
        index = next((position for position, candidate in enumerate(rows) if _contains(candidate, header)), None)
        if index is None:
            message = "A popover header must stand in one of the panel's rows."
            raise ValueError(message)
        body.add_class("glp-popover")
        body.layout.left = "0px"
        body.layout.top = f"{(index + 1) * style.ROW_PITCH_PX}px"
        bodies.append(body)

        def close_others(change: dict[str, Any], header: widgets.ToggleButton = header) -> None:
            if change["new"]:
                for other in headers:
                    if other is not header:
                        other.value = False

        header.observe(close_others, names="value")
    return tuple(bodies)


class ViewPanel:
    """
    One view: its control rows, one status line, the plot gap, then the plot slot or the failure in its place.

    Redraws go through ``request``: an inactive panel (a hidden tab) only remembers the latest request and draws it on
    ``activate`` (rule 12); inside ``batch()`` only the last request runs, once, when the batch ends (rule 4).

    Every render request takes a new ``generation`` with ``supersede``; a request publishes only while ``current`` holds
    for its generation (rule 3). A panel built with a ``host`` is a view nested in the host's plot: it binds to the host
    render that built it and is current only while that render is current too.
    """

    def __init__(
        self,
        rows: Sequence[widgets.Widget],
        *,
        status: controls.StatusLine | None = None,
        slot: PlotSlot | None = None,
        popovers: Sequence[tuple[widgets.ToggleButton, widgets.DOMWidget]] = (),
        on_activate: Callable[[], None] | None = None,
        on_deactivate: Callable[[], None] | None = None,
        host: ViewPanel | None = None,
    ) -> None:
        """Assemble the panel; it starts active and, with a ``host``, bound to the host's current render."""
        self.rows = tuple(rows)
        self.status = status if status is not None else controls.StatusLine()
        self.slot = slot
        self.failure = controls.FailureDisplay()
        children: list[widgets.Widget] = [control_rows(self.rows), self.status.widget, self.failure.widget]
        if slot is not None:
            children.append(slot.widget)
        children.extend(popover_bodies(self.rows, popovers))
        self.widget = widgets.VBox(tuple(children))
        self.widget.add_class(style.ROOT_CLASS)
        self.widget.add_class("glp-panel")
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._active = True
        self._batch_depth = 0
        self._pending: Callable[[], None] | None = None
        self.generation = 0
        self.host = None if host is None else (host, host.generation)

    @property
    def active(self) -> bool:
        """Return whether the panel is visible and draws on request."""
        return self._active

    def supersede(self) -> int:
        """Supersede every earlier render request of the panel and return the generation of the new one."""
        self.generation += 1
        return self.generation

    def current(self, generation: int) -> bool:
        """Return whether a render of ``generation`` may still publish: it is the newest, the panel is shown, the host render too."""
        return self.generation == generation and self._active and (self.host is None or self.host[0].current(self.host[1]))

    def request(self, render: Callable[[], None]) -> None:
        """Run ``render`` now, or remember it as the latest request while batching or inactive."""
        if self._batch_depth or not self._active:
            self._pending = render
            return
        self._pending = None
        render()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Coalesce the requests of several control changes into one redraw at the end."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if not self._batch_depth:
                self._flush()

    def activate(self) -> None:
        """Mark the panel visible and draw the request it missed while hidden."""
        if self._active:
            return
        self._active = True
        if self._on_activate is not None:
            self._on_activate()
        self._flush()

    def deactivate(self) -> None:
        """Mark the panel hidden and supersede the renders still running for it; later requests wait for ``activate``."""
        if not self._active:
            return
        self._active = False
        self.supersede()
        if self._on_deactivate is not None:
            self._on_deactivate()

    def show_failure(self, error: BaseException, export_state: dict[str, Any] | None = None) -> None:
        """Show ``error`` in the plot's place and release the export target (rules 9 and 10)."""
        self.failure.show(error, export_state)
        if self.slot is not None:
            controls.set_visible(self.slot.widget, False)

    def clear_failure(self) -> None:
        """Remove a shown failure and give the plot its place back."""
        self.failure.clear()
        if self.slot is not None:
            controls.set_visible(self.slot.widget, True)

    def _flush(self) -> None:
        pending = self._pending
        if pending is not None and self._active and not self._batch_depth:
            self._pending = None
            pending()


class ViewCache:
    """Keep the results of one expensive view for its most recently used selections, in kernel memory only."""

    def __init__(self, entries: int = VIEW_CACHE_ENTRIES) -> None:
        """Create an empty cache for ``entries`` results."""
        if entries < 1:
            message = "A view cache needs room for at least one entry."
            raise ValueError(message)
        self.entries = entries
        self._results: OrderedDict[Hashable, Any] = OrderedDict()

    def __contains__(self, key: Hashable) -> bool:
        """Return whether ``key`` is cached."""
        return key in self._results

    def __len__(self) -> int:
        """Return the number of cached results."""
        return len(self._results)

    def get(self, key: Hashable) -> Any:
        """Return the result of ``key`` and mark it most recently used."""
        self._results.move_to_end(key)
        return self._results[key]

    def put(self, key: Hashable, result: Any) -> None:
        """Store ``result`` and drop the least recently used entries beyond the capacity."""
        self._results[key] = result
        self._results.move_to_end(key)
        while len(self._results) > self.entries:
            self._results.popitem(last=False)


def compute_button() -> widgets.Button:
    """Return the «Berechnen» button of an expensive view."""
    return controls.action_button(controls.COMPUTE_LABEL, kind="compute")


def _viewers() -> ModuleType:
    """Return the viewer module; imported on use because it builds panels from this module."""
    return import_module(f"{__package__}.analysis_ui_viewers")


class OnDemandView:
    """
    The plot of one view: a cheap view draws on every refresh, an expensive one only on «Berechnen» (rule 2).

    ``draw(key)`` returns the figure for the hashable control state ``selection()``. An expensive view keeps its last
    ``VIEW_CACHE_ENTRIES`` results in a ``ViewCache`` and shows a cached selection at once with the note «aus dem Cache».
    The panel is built before the view, so the «Berechnen» ``button`` is made with ``compute_button()``, placed in one of
    the panel's rows, and passed here; it is shown only for an expensive view.
    """

    def __init__(
        self,
        panel: ViewPanel,
        draw: Callable[[Hashable], Any],
        selection: Callable[[], Hashable],
        *,
        expensive: bool,
        button: widgets.Button | None = None,
        export_state: dict[str, Any] | None = None,
        export_plot_name: str | None = None,
        export_title: str | None = None,
    ) -> None:
        """Create the view; nothing is drawn until ``refresh``."""
        if panel.slot is None:
            message = "An on-demand view needs a panel with a plot slot."
            raise ValueError(message)
        if expensive and button is None:
            message = "An expensive view needs its «Berechnen» button placed in one of the panel's rows."
            raise ValueError(message)
        self.panel = panel
        self.expensive = expensive
        self.cache = ViewCache()
        self.button = button if button is not None else compute_button()
        controls.set_visible(self.button, expensive)
        self.button.on_click(lambda _button: self.compute())
        self._draw = draw
        self._selection = selection
        self._export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
        self._guard = controls.BusyGuard([self.button], panel.status)

    def refresh(self) -> None:
        """Follow a control change: draw a cheap view, show a cached result, or wait for «Berechnen»."""
        self.panel.request(self._refresh_now)

    def compute(self) -> None:
        """Compute the current selection, or take it from the cache, and show it."""
        key = self._selection()

        def run() -> str:
            if key in self.cache:
                return self._show_cached(key)
            started = time.perf_counter()
            result = self._draw(key)
            elapsed = time.perf_counter() - started
            self.cache.put(key, result)
            _viewers().render_view(self.panel, lambda: result, **self._export)
            return f"Berechnet in {elapsed:.1f} s; für diese Auswahl gespeichert ({len(self.cache)} von {self.cache.entries} Einträgen)."

        self._guard.run(run, busy_text="Berechne …", on_error=lambda error: _viewers().fail_view(self.panel, error, **self._export))

    def _refresh_now(self) -> None:
        key = self._selection()
        if not self.expensive:
            _viewers().render_view(self.panel, self._draw, args=(key,), **self._export)
        elif key in self.cache:
            self.panel.status.set(self._show_cached(key))
        else:
            message = f"Diese Auswahl ist noch nicht berechnet: «{controls.COMPUTE_LABEL}» drücken."
            _viewers().clear_view(self.panel, message, **self._export)

    def _show_cached(self, key: Hashable) -> str:
        result = self.cache.get(key)
        _viewers().render_view(self.panel, lambda: result, **self._export)
        return f"Ansicht {CACHE_NOTE} ({len(self.cache)} von {self.cache.entries} Einträgen)."


def tab_width_ch(titles: Sequence[str]) -> int:
    """Return the header width in ch of a tab set whose longest title is in ``titles``."""
    if not titles:
        message = "A tab width needs at least one title."
        raise ValueError(message)
    widest = math.ceil(max(style.text_ch(title) for title in titles))
    return math.ceil((widest + style.TAB_PADDING_CH) * style.TAB_WIDTH_FACTOR)


def set_tab_width(tab: widgets.Tab, width_ch: int) -> None:
    """Give every header of ``tab`` the width ``width_ch``, replacing a previous width."""
    target = style.tab_class(width_ch)
    for name in tuple(tab._dom_classes):  # noqa: SLF001 -- ipywidgets exposes the class list only through this trait
        if name.startswith(_TAB_CLASS_PREFIX) and name != target:
            tab.remove_class(name)
    tab.add_class(target)


def tabs(titles: Sequence[str], children: Sequence[widgets.Widget], *, width_ch: int) -> widgets.Tab:
    """Return a tab set with one title per child and equally wide headers of ``width_ch`` ch."""
    if not titles or len(titles) != len(children):
        message = f"Tabs need at least one child and one title per child, got {len(titles)} titles and {len(children)} children."
        raise ValueError(message)
    tab = widgets.Tab(children=tuple(children))
    for index, title in enumerate(titles):
        tab.set_title(index, title)
    tab.add_class(style.ROOT_CLASS)
    set_tab_width(tab, width_ch)
    return tab
