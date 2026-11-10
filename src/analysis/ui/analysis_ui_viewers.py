"""Build interactive viewers for case-level and aggregate analysis plots.

Responsibilities:
  - Render case-by-case Matplotlib figures inside widgets
  - Show a failing view in place instead of raising it into the notebook
  - Manage dataset and case navigation controls
  - Store current figures for notebook export

Design principles:
  - Viewer callbacks receive explicit datasets and render functions
  - Widget construction delegates to analysis_ui_components
  - Panel-local export state is updated only around rendered figures

This module does NOT:
  - Compose numbered notebook sections or choose scientific control vocabularies
  - Load artifacts directly or implement domain-specific plot mathematics
"""

from __future__ import annotations

import logging
import traceback
from html import escape
from threading import Lock
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.figure import Figure

from . import analysis_ui_components as components

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd


# =============================================================================
# INTERNAL HELPERS (viewer-agnostic, no semantics)
# =============================================================================

_LOGGER = logging.getLogger(__name__)
_RENDER_LOCK = Lock()
_RENDER_GENERATIONS: WeakKeyDictionary[widgets.Output, int] = WeakKeyDictionary()


def show_render_failure(out: widgets.Output, error: BaseException) -> None:
    """Replace the output content with one visible failure line and a collapsed traceback."""
    _LOGGER.exception("View failed: %s", error, exc_info=error)
    detail = escape("".join(traceback.format_exception(error)))
    summary = escape(f"View failed — {type(error).__name__}: {error}")
    with out:
        out.clear_output(wait=True)
        display(HTML(f"<p><b>{summary}</b></p><details><summary>Traceback</summary><pre>{detail}</pre></details>"))


def clear_rendered_figure(
    *,
    out: widgets.Output,
    message: str,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> None:
    """Supersede in-flight rendering before clearing a legitimate empty view."""
    with _RENDER_LOCK:
        _RENDER_GENERATIONS[out] = _RENDER_GENERATIONS.get(out, 0) + 1
        if export_state is not None:
            previous = export_state.get("fig")
            if isinstance(previous, Figure):
                plt.close(previous)
            export_state.update(
                fig=None,
                figures=(),
                plot_name=export_plot_name,
                title=export_title,
                filename_stem=None,
            )
        with out:
            out.clear_output(wait=True)
            print(message)


def render_figure(
    *,
    out: widgets.Output,
    plot_func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> None:
    """
    Invoke and display one plot result inside an output widget.

    Parameters
    ----------
    out : ipywidgets.Output
        Output area updated only after the latest requested render finishes.
    plot_func : Callable[..., Any]
        Callable returning a Matplotlib figure, a tuple whose first item is a
        figure, another displayable value, or ``None``.
    args : tuple[Any, ...], optional
        Positional arguments forwarded unchanged.
    kwargs : dict[str, Any] | None, optional
        Keyword arguments forwarded unchanged.
    export_state : dict[str, Any] | None, optional
        Panel-local mutable state receiving the current rendered figure.
    export_plot_name, export_title : str | None, optional
        Stable export stem and display title owned by the active dropdown entry.

    Notes
    -----
    Recognized figures update only the explicitly supplied panel state, are
    displayed, and are then closed to release GUI resources. A newer request
    supersedes an older result before either display or export publication.
    A failing callable is shown in place of its figure and clears the export
    target; a later successful render replaces the failure.

    """
    with _RENDER_LOCK:
        generation = _RENDER_GENERATIONS.get(out, 0) + 1
        _RENDER_GENERATIONS[out] = generation
    try:
        result = plot_func(*args, **(kwargs or {}))
    except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
        with _RENDER_LOCK:
            if _RENDER_GENERATIONS.get(out) == generation:
                if export_state is not None:
                    previous = export_state.get("fig")
                    if isinstance(previous, Figure):
                        plt.close(previous)
                    export_state.update(fig=None, figures=(), plot_name=export_plot_name, title=export_title)
                show_render_failure(out, error)
        return

    fig: Figure | None = None
    if isinstance(result, Figure):
        fig = result
    elif isinstance(result, tuple) and result and isinstance(result[0], Figure):
        fig = result[0]
    with _RENDER_LOCK:
        if _RENDER_GENERATIONS.get(out) != generation:
            if fig is not None:
                plt.close(fig)
            return
        if export_state is not None:
            previous = export_state.get("fig")
            if isinstance(previous, Figure) and previous is not fig:
                plt.close(previous)
            export_state.update(fig=fig, figures=(fig,) if fig is not None else (), plot_name=export_plot_name, title=export_title)
        with out:
            out.clear_output(wait=True)
            if fig is not None:
                display(fig)
                plt.close(fig)
            elif result is not None:
                display(result)


def _attach_widget_rerender(
    widgets_list: list[widgets.Widget],
    render_func: Callable[[], None],
) -> None:
    """
    Register one render callback on heterogeneous semantic controls.

    Standard value widgets observe their ``value`` trait. Checkbox-group
    containers are recognized through the public ``boxes`` mapping and each
    checkbox is observed. Widgets without either contract are ignored.
    """
    for w in widgets_list:
        # ---------------------------------------------
        # Case 1: standard ValueWidget (Dropdown, Radio)
        # ---------------------------------------------
        if hasattr(w, "observe") and hasattr(w, "value"):
            w.observe(lambda _: render_func(), names="value")
            continue

        # ---------------------------------------------
        # Case 2: checkbox group (VBox with .boxes)
        # ---------------------------------------------
        boxes = getattr(w, "boxes", None)
        if isinstance(boxes, dict):
            for checkbox in boxes.values():
                if isinstance(checkbox, widgets.Checkbox):
                    checkbox.observe(lambda _: render_func(), names="value")


def _control_value(widget: widgets.Widget) -> Any:
    """Return a normal value or an ordered tuple from a checkbox group."""
    boxes = getattr(widget, "boxes", None)
    if isinstance(boxes, dict):
        return tuple(label for label, checkbox in boxes.items() if isinstance(checkbox, widgets.Checkbox) and checkbox.value)
    if hasattr(widget, "value"):
        return cast("widgets.ValueWidget", widget).value
    msg = "Scientific controls must expose value or a checkbox-group boxes mapping."
    raise TypeError(msg)


# =============================================================================
# CASECOUNT VIEWER (multi-case aggregations)
# =============================================================================


def make_casecount_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    start_cases: int = 100,
    step_size: int = 50,
    extra_widgets: list[widgets.Widget] | None = None,
    controls: Mapping[str, widgets.Widget] | None = None,
    allow_dataset_selection: bool = False,
    selector_title: str = "datasets",
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """Build and immediately render the shared-prefix control."""
    if not datasets or any(frame.empty for frame in datasets.values()):
        msg = "Case-count viewers require non-empty labelled datasets."
        raise ValueError(msg)
    if not isinstance(step_size, int) or isinstance(step_size, bool) or step_size <= 0:
        msg = "step_size must be a positive integer."
        raise ValueError(msg)
    default_cases = min(len(df) for df in datasets.values())
    max_cases_global = max(len(df) for df in datasets.values())
    case_count, prev_btn, next_btn = components.ui_step_case_count(
        start_cases=min(start_cases, default_cases),
        min_cases=0,
        max_cases=max_cases_global,
        step_size=step_size,
    )
    selector = (
        cast("components.CheckboxGroup", components.ui_checkbox_datasets(dataset_names=list(datasets)))
        if allow_dataset_selection and len(datasets) > 1
        else None
    )
    output = components.ui_output_plot()
    extra_widgets = list(extra_widgets or [])
    semantic_controls = dict(controls or {})

    def _selected_datasets() -> dict[str, pd.DataFrame]:
        """Return all frames or the historically checked subset in stable order."""
        if selector is None:
            return dict(datasets)
        return {name: datasets[name] for name, checkbox in selector.boxes.items() if checkbox.value}

    def _render() -> None:
        """Render the selected saved-membership prefix without a confirmation step."""
        selected = _selected_datasets()
        if not selected:
            clear_rendered_figure(
                out=output,
                message=f"Select at least one {selector_title.strip().lower()}.",
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            return
        render_figure(
            out=output,
            plot_func=plot_func,
            kwargs={
                "datasets": selected,
                "max_cases": int(case_count.value),
                **{name: _control_value(widget) for name, widget in semantic_controls.items()},
                **plot_kwargs,
            },
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def _step(delta: int) -> None:
        """Change prefix size by one step within the shared bound."""
        new_value = case_count.value + delta * step_size
        case_count.value = max(1, min(max_cases_global, new_value))

    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    case_count.observe(lambda _: _render(), names="value")
    _attach_widget_rerender([*semantic_controls.values(), *extra_widgets], _render)
    if selector is not None:
        for checkbox in selector.boxes.values():
            checkbox.observe(lambda _: _render(), names="value")

    _render()
    header_controls = [case_count, prev_btn, next_btn, *semantic_controls.values(), *extra_widgets]
    if selector is not None:
        header_controls.append(selector)
    header = widgets.HBox(header_controls, layout=widgets.Layout(align_items="center"))
    return widgets.VBox([header, output])


def make_indexed_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    controls: Mapping[str, widgets.ValueWidget] | None = None,
    dataset_selection: str = "dropdown",
    max_positions: int | None = None,
    index_to_kwargs: Callable[[int], Mapping[str, Any]] | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """Build a one-based case navigator over one selected model or all models."""
    if not datasets or any(frame.empty for frame in datasets.values()):
        msg = "Indexed viewers require non-empty labelled datasets."
        raise ValueError(msg)
    if dataset_selection not in {"all", "dropdown"}:
        msg = "dataset_selection must be 'all' or 'dropdown'."
        raise ValueError(msg)
    if max_positions is not None and (isinstance(max_positions, bool) or not isinstance(max_positions, int) or max_positions <= 0):
        msg = "max_positions must be a positive integer when supplied."
        raise ValueError(msg)

    semantic_controls = dict(controls or {})
    dataset_dropdown = components.ui_dropdown_dataset(list(datasets)) if dataset_selection == "dropdown" and len(datasets) > 1 else None

    def _selected_datasets() -> dict[str, pd.DataFrame]:
        """Return one dropdown-selected frame or the complete ordered mapping."""
        if dataset_dropdown is None:
            return dict(datasets)
        selected = dataset_dropdown.value
        if not isinstance(selected, str):
            msg = "Dataset dropdown must contain string values."
            raise TypeError(msg)
        return {selected: datasets[selected]}

    def _position_count() -> int:
        """Return the current safe navigation bound."""
        count = min(len(frame) for frame in _selected_datasets().values())
        return count if max_positions is None else min(count, max_positions)

    case_index, previous, following = components.ui_step_case_index(n_cases=_position_count(), start_idx=0)
    if not isinstance(case_index, widgets.BoundedIntText):
        msg = "Indexed viewers require a bounded one-based case control."
        raise TypeError(msg)
    output = components.ui_output_plot()
    state = {"updating": False}

    def _render(_: object = None) -> None:
        """Render the current index and semantic controls automatically."""
        if state["updating"]:
            return
        zero_based = int(case_index.value) - 1
        index_kwargs = {"row_position": zero_based} if index_to_kwargs is None else dict(index_to_kwargs(zero_based))
        control_kwargs = {name: widget.value for name, widget in semantic_controls.items()}
        render_figure(
            out=output,
            plot_func=plot_func,
            kwargs={"datasets": _selected_datasets(), **index_kwargs, **control_kwargs},
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def _step(delta: int) -> None:
        """Move within the current one-based position bound."""
        case_index.value = max(1, min(case_index.max, case_index.value + delta))

    def _select_dataset(_: object = None) -> None:
        """Rebind the position range before rendering the selected model."""
        state["updating"] = True
        try:
            case_index.max = _position_count()
            case_index.value = min(case_index.value, case_index.max)
        finally:
            state["updating"] = False
        _render()

    previous.on_click(lambda _: _step(-1))
    following.on_click(lambda _: _step(1))
    case_index.observe(_render, names="value")
    for widget in semantic_controls.values():
        widget.observe(_render, names="value")
    if dataset_dropdown is not None:
        dataset_dropdown.observe(_select_dataset, names="value")

    _render()
    header_items: list[widgets.Widget] = [case_index, previous, following, *semantic_controls.values()]
    if dataset_dropdown is not None:
        header_items.append(dataset_dropdown)
    return widgets.VBox([widgets.HBox(header_items), output])
