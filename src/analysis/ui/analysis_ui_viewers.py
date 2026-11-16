"""Build interactive viewers for aggregate analysis plots over the first cases of labelled datasets.

Responsibilities:
  - Render Matplotlib figures and other items into the plot slots of view panels
  - Let only the newest requested render of a view publish its figure and export target
  - Show a failing view in place instead of raising it into the notebook
  - Manage dataset, case-count, and semantic view controls of the case-count viewer
  - Store current figures for notebook export

Design principles:
  - Viewer callbacks receive explicit datasets and render functions
  - Widget construction delegates to the ui control, navigation, and panel modules
  - Panel-local export state is updated only around rendered figures
  - A view panel publishes without a lock: a request supersedes the earlier ones before its plot runs, and its
    publication runs no plot function and no callback, so a request re-entering the view wins and cannot deadlock

This module does NOT:
  - Compose numbered notebook sections or choose scientific control vocabularies
  - Load artifacts directly or implement domain-specific plot mathematics
  - Navigate single cases or draw case-level views; the case-count viewer draws separate datasets only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import analysis_ui_controls as ui_controls
from . import analysis_ui_navigation as navigation
from . import analysis_ui_panel as panel

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import pandas as pd

    from .analysis_ui_selectors import DatasetSelector

type ViewControl = widgets.ValueWidget | ui_controls.ChannelChecks


EMPTY_SELECTION_MESSAGE = "Select at least one dataset."


# =============================================================================
# VIEW-PANEL RENDERING (control standard)
# =============================================================================


@dataclass(frozen=True, slots=True)
class _Export:
    """The export target of one request: the panel-local state and the identity the request publishes under."""

    state: dict[str, Any] | None
    plot_name: str | None
    title: str | None


@dataclass(frozen=True, slots=True)
class _Failure:
    """The outcome of a request whose plot function raised, or that reports a failure directly."""

    error: BaseException


type _Outcome = list[Any] | _Failure


def _require_slot(view: panel.ViewPanel) -> panel.PlotSlot:
    """Return the plot slot of ``view`` or refuse a panel without one."""
    if view.slot is None:
        message = "Rendering a view needs a panel with a plot slot."
        raise ValueError(message)
    return view.slot


def _items(result: Any) -> list[Any]:
    """Return the items of a plot result: the non-``None`` items of a list or tuple in order, else the result itself unless ``None``."""
    return [item for item in (result if isinstance(result, (list, tuple)) else (result,)) if item is not None]


def _close_figures(outcome: _Outcome) -> None:
    """Close the figures of ``outcome``; an exported figure stays readable for the export."""
    if isinstance(outcome, list):
        for item in outcome:
            if isinstance(item, Figure):
                plt.close(item)


def _show_failure(view: panel.ViewPanel, error: BaseException, export: _Export) -> None:
    """Show ``error`` in the plot's place under the request's export identity and release the export target (rule 12)."""
    if export.state is not None:
        export.state.update(plot_name=export.plot_name, title=export.title)
    view.show_failure(error, export.state)


def _show_items(view: panel.ViewPanel, items: list[Any], export: _Export) -> None:
    """Export the first figure as ``fig`` and every figure as ``figures``, show every item in order in the plot slot, then clear a shown failure."""
    figures = tuple(item for item in items if isinstance(item, Figure))
    figure = figures[0] if figures else None
    if export.state is not None:
        if export.state.get("fig") is not figure:
            ui_controls.release_export(export.state)
        export.state.update(fig=figure, figures=figures, plot_name=export.plot_name, title=export.title)
    try:
        _require_slot(view).show(items)
    finally:
        _close_figures(items)
    # Last, so that a render an observer of the failure display starts here publishes after this one.
    view.clear_failure()


def _publish(view: panel.ViewPanel, generation: int, outcome: _Outcome, export: _Export) -> bool:
    """
    Publish ``outcome`` of the request ``generation`` of ``view`` if it is still current; return whether it was published.

    A request that is no longer current only closes its figures. A failure outcome, and any exception of the
    publication itself, ends in the failure display with the export target released; nothing escapes.
    """
    if not view.current(generation):
        _close_figures(outcome)
        return False
    try:
        if isinstance(outcome, _Failure):
            _show_failure(view, outcome.error, export)
        else:
            _show_items(view, outcome, export)
    except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
        _close_figures(outcome)
        _show_failure(view, error, export)
    return True


def render_view(
    view: panel.ViewPanel,
    plot_func: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> bool:
    """
    Invoke ``plot_func`` and show its result in the plot slot of ``view``; only the newest request publishes (rule 3).

    The result is one displayable item (a Matplotlib figure, a widget, a table, a string) or a list or tuple of items,
    shown in their order without the ``None`` entries. The first figure becomes the export target and all figures the
    export bundle; every figure is closed after display. A failure of ``plot_func`` or of the publication replaces the
    plot with the failure display and releases the export target; a later successful render gives the plot its place back.

    The request supersedes every earlier one before ``plot_func`` runs, so a request made while it runs wins. Returns
    whether the result was published: False when a newer request, hiding the view, or the end of its host's render
    superseded it.
    """
    _require_slot(view)
    export = _Export(export_state, export_plot_name, export_title)
    generation = view.supersede()
    outcome: _Outcome
    try:
        outcome = _items(plot_func(*args, **(kwargs or {})))
    except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
        outcome = _Failure(error)
    return _publish(view, generation, outcome, export)


def show_result(
    view: panel.ViewPanel,
    result: Any,
    *,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> bool:
    """Supersede pending renders and show an already computed ``result`` in the plot slot of ``view`` like ``render_view``."""
    _require_slot(view)
    return _publish(view, view.supersede(), _items(result), _Export(export_state, export_plot_name, export_title))


def clear_view(
    view: panel.ViewPanel,
    message: str,
    *,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> bool:
    """Supersede pending renders, release the export target, and explain the empty view in the plot's place (rule 6)."""
    _require_slot(view)
    return _publish(view, view.supersede(), [message], _Export(export_state, export_plot_name, export_title))


def fail_view(
    view: panel.ViewPanel,
    error: BaseException,
    *,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> bool:
    """Supersede pending renders and show ``error`` in the plot's place with the export target released (rule 12)."""
    return _publish(view, view.supersede(), _Failure(error), _Export(export_state, export_plot_name, export_title))


def _control_element(control: ViewControl) -> widgets.Widget:
    """Return the row element of one view control."""
    return control.widget if isinstance(control, ui_controls.ChannelChecks) else control


def _view_control_value(control: ViewControl) -> Any:
    """Return the value one view control passes to the plot."""
    return control.selected if isinstance(control, ui_controls.ChannelChecks) else control.value


def _observe_control(control: ViewControl, callback: Callable[[], None]) -> None:
    """Call ``callback()`` after every change of one view control."""
    if isinstance(control, ui_controls.ChannelChecks):
        control.observe(callback)
    elif isinstance(control, widgets.ValueWidget):
        control.observe(lambda _change: callback(), names="value")
    else:
        message = f"A view control must be a value widget or ChannelChecks, got {type(control).__name__}."
        raise TypeError(message)


def _labelled_frames(datasets: Mapping[Hashable, pd.DataFrame], *, kind: str) -> Mapping[Hashable, pd.DataFrame]:
    """Return the datasets, a plain dict copied and any other mapping (an immutable one carrying labels) kept, refusing empty input."""
    frames = dict(datasets) if isinstance(datasets, dict) else datasets
    if not frames or any(frame.empty for frame in frames.values()):
        message = f"{kind} views require non-empty labelled datasets."
        raise ValueError(message)
    return frames


def _require_separate_selector(selector: DatasetSelector | None, frames: Mapping[Hashable, pd.DataFrame]) -> None:
    """Refuse a selector that may pool or that names datasets the view does not hold."""
    if selector is None:
        return
    if selector.pool_enabled:
        message = "Case viewers draw separate datasets; build their DatasetSelector with a pool_reason."
        raise ValueError(message)
    if not isinstance(frames, dict):
        message = "A dataset selector re-keys the frames by dataset label and would drop their display labels; pass a plain dict."
        raise TypeError(message)
    missing = [item.key for item in selector.inventory.items if item.key not in frames]
    if missing:
        message = f"The dataset selector offers keys without a frame: {missing!r}."
        raise ValueError(message)


def _selected_frames(frames: Mapping[Hashable, pd.DataFrame], selector: DatasetSelector | None) -> Mapping[Hashable, pd.DataFrame]:
    """Return every frame as given (a plain dict copied), or the selected ones keyed by their dataset label (a plain dict)."""
    if selector is None:
        return dict(frames) if isinstance(frames, dict) else frames
    return {choice.label: frames[choice.items[0].key] for choice in selector.choices() if choice.available}


def _selection_rows(selector: DatasetSelector | None) -> tuple[list[widgets.Widget], tuple[tuple[widgets.ToggleButton, widgets.DOMWidget], ...]]:
    """Return the selection row and its popover when a selector is given."""
    if selector is None:
        return [], ()
    return [panel.row(selector.header)], ((selector.header, selector.body),)


def case_count_view(
    plot_func: Callable[..., Any],
    *,
    datasets: Mapping[Hashable, pd.DataFrame],
    start_cases: int,
    step_size: int,
    controls: Mapping[str, ViewControl] | None = None,
    selector: DatasetSelector | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    resume_cases: int | None = None,
    host: panel.ViewPanel | None = None,
    **plot_kwargs: Any,
) -> panel.ViewPanel:
    """
    Build a view that draws the first ``max_cases`` cases of the selected datasets and redraws on every change.

    ``plot_func`` receives ``datasets`` (the selected frames, keyed by dataset label when a ``selector`` is given),
    ``max_cases``, one keyword per entry of ``controls`` and ``plot_kwargs``. An empty selection clears the view with
    an explanation. The count starts at ``start_cases``, limited to the shortest dataset; ``resume_cases``, the count a
    former view of the same datasets showed, wins and is kept up to the longest dataset. Built inside the render of a
    ``host`` view, the view publishes only while that host render is current and the host is shown.
    """
    frames = _labelled_frames(datasets, kind="Case-count")
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        message = "step_size must be a positive integer."
        raise ValueError(message)
    _require_separate_selector(selector, frames)
    semantic = dict(controls or {})
    export: dict[str, Any] = {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}
    view: panel.ViewPanel | None = None

    def request() -> None:
        if view is not None:
            view.request(render_now)

    lengths = [len(frame) for frame in frames.values()]
    count = navigation.CountControl(
        value=max(1, min(start_cases, *lengths) if resume_cases is None else min(resume_cases, max(lengths))),
        minimum=1,
        maximum=max(lengths),
        step=step_size,
        on_change=lambda _count: request(),
    )
    rows, popovers = _selection_rows(selector)
    rows.append(panel.row(count.widget, *(_control_element(control) for control in semantic.values())))
    view = panel.ViewPanel(rows, slot=panel.PlotSlot(), popovers=popovers, host=host)

    def render_now() -> None:
        selected = _selected_frames(frames, selector)
        if not selected:
            clear_view(cast("panel.ViewPanel", view), EMPTY_SELECTION_MESSAGE, **export)
            return
        values = {name: _view_control_value(control) for name, control in semantic.items()}
        render_view(
            cast("panel.ViewPanel", view),
            plot_func,
            kwargs={"datasets": selected, "max_cases": count.value, **values, **plot_kwargs},
            **export,
        )

    for control in semantic.values():
        _observe_control(control, request)
    if selector is not None:
        selector.observe(request)
    request()
    return view
