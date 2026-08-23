"""
eda_viewers.py

Build shared-state live viewers for adaptive generated-output EDA.

Responsibilities:
  - Bind shared dataset, case, channel, and time state to scientific plots
  - Render mixed-capability aggregate, spatial, spectral, and transient views lazily
  - Preserve channel selections and exact physical time across compatible changes
  - Keep active figures synchronized with explicit panel-local export state

Design principles:
  - Registered factories expose explicit export signatures
  - Selection changes load only selected capability-compatible views
  - View deactivation preserves cached scientific selections without playback

This module does NOT:
  - Discover storage, define presentation order, or implement plot mathematics
  - Catch programming errors or reinterpret unavailable scientific capabilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.figure import Figure

from src.analysis import ui

from . import eda_capabilities as capabilities
from . import eda_controls as controls
from . import eda_selection as selection
from . import eda_transient as transient
from . import plots

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd

_TransientCaseKind = Literal["snapshot", "trajectory"]
_SCOPE_ROW_WIDTH = f"{ui.notebook.COMPACT_VIEW_SELECTOR_WIDTH_PX}px"
_WIDGET_DEFAULT_MARGIN_PX = 2.0
_SCOPE_BUTTON_INCREMENT_POINTS = 1.0
_CSS_PIXELS_PER_POINT = 96.0 / 72.0
_SCOPE_BUTTON_WIDTH_PX = (
    ui.notebook.COMPACT_VIEW_SELECTOR_WIDTH_PX - 4.0 * _WIDGET_DEFAULT_MARGIN_PX
) / 2.0 + _SCOPE_BUTTON_INCREMENT_POINTS * _CSS_PIXELS_PER_POINT
_SCOPE_BUTTON_WIDTH = f"{_SCOPE_BUTTON_WIDTH_PX:.12g}px"
_SCOPE_BUTTON_GROUP_WIDTH_PX = 2.0 * _SCOPE_BUTTON_WIDTH_PX + 4.0 * _WIDGET_DEFAULT_MARGIN_PX
_SCOPE_BUTTON_GROUP_WIDTH = f"{_SCOPE_BUTTON_GROUP_WIDTH_PX:.12g}px"
_MINIMUM_COMPARABLE_MAP_COUNT = 2


class ActivatableView(widgets.VBox):
    """Expose explicit activation and deactivation for one cached live view."""

    def __init__(
        self,
        children: Sequence[widgets.Widget],
        *,
        activate: Callable[[], None],
        deactivate: Callable[[], None] | None = None,
    ) -> None:
        """Retain normal VBox construction beside lifecycle callbacks."""
        super().__init__(children=tuple(children))
        self._activate_callback = activate
        self._deactivate_callback = deactivate

    def activate(self) -> None:
        """Restore and rerender this view from current shared selection."""
        self._activate_callback()

    def deactivate(self) -> None:
        """Stop active resources before the view becomes hidden."""
        if self._deactivate_callback is not None:
            self._deactivate_callback()


def _is_active(
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
) -> bool:
    """Return whether a cached view currently owns the panel export identity."""
    return export_state is None or export_state.get("plot_name") == export_plot_name


def _clear_export_figures(export_state: dict[str, Any]) -> None:
    """Close every distinct retained export figure and clear the bundle."""
    figures = export_state.get("figures", ())
    candidates = tuple(figures) if isinstance(figures, (list, tuple)) else ()
    primary = export_state.get("fig")
    if isinstance(primary, Figure):
        candidates = (*candidates, primary)
    seen: set[int] = set()
    for figure in candidates:
        if isinstance(figure, Figure) and id(figure) not in seen:
            plt.close(figure)
            seen.add(id(figure))
    export_state["fig"] = None
    export_state["figures"] = ()


def _set_export_identity(
    export_state: dict[str, Any] | None,
    *,
    export_plot_name: str | None,
    export_title: str | None,
    clear_figure: bool = False,
) -> None:
    """Synchronize explicit export identity and optionally clear its figures."""
    if export_state is None:
        return
    if clear_figure:
        _clear_export_figures(export_state)
    export_state["plot_name"] = export_plot_name
    export_state["title"] = export_title


def _show_message(
    output: widgets.Output,
    message: str,
    *,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> None:
    """Show one legitimate selection message and clear the export target."""
    _set_export_identity(
        export_state,
        export_plot_name=export_plot_name,
        export_title=export_title,
        clear_figure=True,
    )
    with output:
        output.clear_output(wait=True)
        print(message)


def _compact_scope_controls(
    scope: widgets.ToggleButtons | None,
) -> tuple[widgets.HBox, widgets.HBox]:
    """Build one bounded scope row with a replaceable case/count column."""
    detail = widgets.HBox(
        layout=widgets.Layout(
            align_items="center",
            min_width="0",
        )
    )
    if scope is None:
        detail.layout = widgets.Layout(
            align_items="center",
            width=_SCOPE_ROW_WIDTH,
            min_width=_SCOPE_ROW_WIDTH,
            max_width=_SCOPE_ROW_WIDTH,
            flex=f"0 0 {_SCOPE_ROW_WIDTH}",
        )
        children: tuple[widgets.Widget, ...] = (detail,)
    else:
        scope.layout = widgets.Layout(
            display="flex",
            flex_flow="row nowrap",
            align_items="stretch",
            justify_content="flex-start",
            width=_SCOPE_BUTTON_GROUP_WIDTH,
            min_width=_SCOPE_BUTTON_GROUP_WIDTH,
            max_width=_SCOPE_BUTTON_GROUP_WIDTH,
            flex=f"0 0 {_SCOPE_BUTTON_GROUP_WIDTH}",
            margin="0",
        )
        scope.style.button_width = _SCOPE_BUTTON_WIDTH
        detail.layout = widgets.Layout(
            align_items="center",
            width="auto",
            min_width="0",
            flex="0 1 auto",
        )
        children = (scope, detail)
    row = widgets.HBox(
        children,
        layout=widgets.Layout(
            align_items="center",
            flex_flow="row nowrap",
            width="auto",
            min_width=(_SCOPE_ROW_WIDTH if scope is None else _SCOPE_BUTTON_GROUP_WIDTH),
        ),
    )
    return row, detail


def _set_optional_children(
    container: widgets.Box,
    children: Sequence[widgets.Widget],
) -> None:
    """Populate one optional control row or remove it from normal layout flow."""
    resolved = tuple(children)
    container.children = resolved
    container.layout.display = "flex" if resolved else "none"


def _set_scope_detail(
    detail: widgets.HBox,
    controls: Sequence[widgets.Widget],
) -> None:
    """Fit one active case/count navigator into the bounded scope column."""
    resolved = tuple(controls)
    if not resolved:
        message = "Scope detail requires one active case or count control."
        raise ValueError(message)
    detail.children = resolved


def _scale_lock_is_meaningful(
    frames: Mapping[str, pd.DataFrame],
    fields: Sequence[str],
    *,
    view: capabilities.FieldView,
) -> bool:
    """Return whether any selected field has comparable maps in two datasets."""
    if len(frames) < _MINIMUM_COMPARABLE_MAP_COUNT or not fields:
        return False
    try:
        resolution = capabilities.resolve_fields(
            frames,
            view=view,
            requested=fields,
        )
    except ValueError:
        return False
    for field in resolution.fields:
        compatible = resolution.datasets_by_field[field]
        if len(compatible) < _MINIMUM_COMPARABLE_MAP_COUNT:
            continue
        units = {capabilities.field_unit(frames[label], field) for label in compatible}
        if len(units) == 1:
            return True
    return False


def _update_scale_lock_container(
    container: widgets.HBox,
    checkbox: widgets.Checkbox,
    frames: Mapping[str, pd.DataFrame],
    fields: Sequence[str],
    *,
    view: capabilities.FieldView,
) -> None:
    """Show the scale lock only when shared normalization can affect maps."""
    _set_optional_children(
        container,
        (checkbox,) if _scale_lock_is_meaningful(frames, fields, view=view) else (),
    )


def make_statistics_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    plot_function: Callable[..., widgets.Widget],
    required_capabilities: Sequence[selection.GeneratedOutputCapability],
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build one capability-adaptive established statistics viewer."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=required_capabilities,
        include_case=False,
    )
    output = ui.components.ui_output_plot()

    def update() -> None:
        """Rebuild the established case-count viewer for current shared data."""
        if not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        if not frames:
            _show_message(
                output,
                "Select at least one compatible dataset.",
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            return
        with output:
            output.clear_output(wait=True)
            viewer = plot_function(
                datasets=frames,
                allow_dataset_selection=False,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            display(viewer)

    def activate() -> None:
        """Restore export identity before activating shared controls."""
        _set_export_identity(
            export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
        local.activate()

    local.set_callback(update)
    view = ActivatableView((local.widget, output), activate=activate)
    view.activate()
    return view


def make_spectral_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    single_plot_function: Callable[..., Figure],
    aggregate_plot_function: Callable[..., Figure] | None,
    semantic_controls: Mapping[str, widgets.ValueWidget] | None = None,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build one capability-adaptive spectral viewer with shared scope and channels."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("spectral",),
        include_case=True,
    )
    selected_views = local.selected_views()
    initial_maximum = min(view.case_count for view in selected_views)
    case_count, fewer, more = ui.components.ui_step_case_count(
        start_cases=min(100, initial_maximum),
        min_cases=1,
        max_cases=initial_maximum,
        step_size=50,
    )
    default_scope = "aggregate" if aggregate_plot_function is not None else "single"
    retained_scope = selection_state.scope_selection(
        "spectral",
        default=default_scope,
    )
    if aggregate_plot_function is None:
        retained_scope = "single"
    scope = (
        widgets.ToggleButtons(
            options=(("Aggregate", "aggregate"), ("Single case", "single")),
            value=retained_scope,
        )
        if aggregate_plot_function is not None
        else None
    )
    semantic = dict(semantic_controls or {})
    semantic_rows = (widgets.HBox(tuple(semantic.values())),) if semantic else ()
    scope_row, scope_detail = _compact_scope_controls(scope)
    output = ui.components.ui_output_plot()
    state = {"updating": False}

    def render() -> None:
        """Render one aggregate or exact shared-case spectral figure."""
        if state["updating"] or not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        if not frames:
            _show_message(
                output,
                "Select at least one compatible dataset.",
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            return
        selected_channels = channel_state.selected
        if not selected_channels:
            _show_message(
                output,
                "Select at least one compatible spectral channel.",
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            return
        semantic_values = {name: widget.value for name, widget in semantic.items()}
        if scope is not None and scope.value == "aggregate":
            if aggregate_plot_function is None:
                message = "Aggregate spectral plotting is unavailable."
                raise RuntimeError(message)
            plot_function = aggregate_plot_function
            plot_kwargs = {
                "datasets": frames,
                "max_cases": int(case_count.value),
                "channels": selected_channels,
                **semantic_values,
            }
        else:
            case_number = local.selected_case_number
            if case_number is None:
                _show_message(
                    output,
                    "Selected datasets have no shared case number.",
                    export_state=export_state,
                    export_plot_name=export_plot_name,
                    export_title=export_title,
                )
                return
            plot_function = single_plot_function
            plot_kwargs = {
                "datasets": frames,
                "case_number": case_number,
                "channels": selected_channels,
                **semantic_values,
            }
        ui.viewers.render_figure(
            out=output,
            plot_func=plot_function,
            kwargs=plot_kwargs,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    channel_state = _ChannelState(
        title="Spectral channels",
        callback=render,
        selection_state=selection_state,
        capability="spectral",
    )

    def update_scope_row() -> None:
        """Expose only the active aggregate-count or exact-case navigator."""
        if scope is not None and scope.value == "aggregate":
            _set_scope_detail(scope_detail, (case_count, fewer, more))
        else:
            _set_scope_detail(scope_detail, tuple(local.case_row.children))

    def rebind_and_render() -> None:
        """Rebind bounds and task-compatible channels before one render."""
        frames = local.selected_frames()
        selected = local.selected_views()
        if selected:
            maximum = min(view.case_count for view in selected)
            state["updating"] = True
            try:
                case_count.max = maximum
                case_count.value = min(case_count.value, maximum)
            finally:
                state["updating"] = False
        if not frames:
            render()
            return
        resolution = capabilities.resolve_fields(frames, view="spectral")
        options = resolution.fields
        if not options:
            _show_message(
                output,
                "Selected datasets expose no compatible spectral channels.",
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            return
        channel_state.rebind(
            options,
            labels=capabilities.resolved_field_labels(frames, resolution),
        )
        render()

    def step_count(delta: int) -> None:
        """Move the bounded aggregate prefix by one configured increment."""
        case_count.value = max(
            case_count.min,
            min(case_count.max, case_count.value + delta * case_count.step),
        )

    def activate() -> None:
        """Restore export identity before activating shared selection."""
        _set_export_identity(
            export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
        local.activate()

    local.set_callback(rebind_and_render)
    case_count.observe(lambda _change: render(), names="value")
    fewer.on_click(lambda _button: step_count(-1))
    more.on_click(lambda _button: step_count(1))
    for widget in semantic.values():
        widget.observe(lambda _change: render(), names="value")
    if scope is not None:

        def scope_changed(_change: dict[str, Any]) -> None:
            """Retain and render one accepted scientific scope change."""
            selected_scope = str(scope.value)
            selection_state.select_scope("spectral", selected_scope)
            update_scope_row()
            render()

        scope.observe(scope_changed, names="value")
    update_scope_row()
    view = ActivatableView(
        (
            scope_row,
            local.status,
            channel_state.container,
            *semantic_rows,
            output,
        ),
        activate=activate,
    )
    view.activate()
    return view


class _ChannelState:
    """Preserve compatible checkbox channels in shared session state."""

    def __init__(
        self,
        *,
        title: str,
        callback: Callable[[], None],
        selection_state: selection.GeneratedOutputSelectionState,
        capability: str,
    ) -> None:
        """Initialize an empty capability-bound checkbox container."""
        self._title = title
        self._callback = callback
        self._selection_state = selection_state
        self._capability = capability
        self._known: tuple[str, ...] = ()
        self._group: ui.components.CheckboxGroup | None = None
        self.status = widgets.HTML(
            layout=widgets.Layout(display="none"),
        )
        self.container = widgets.VBox(
            layout=widgets.Layout(
                display="none",
                width="max-content",
                max_width="100%",
                align_items="flex-start",
            )
        )

    @property
    def selected(self) -> tuple[str, ...]:
        """Return checked channels in canonical visible order."""
        if self._group is None:
            return ()
        return tuple(name for name, checkbox in self._group.boxes.items() if checkbox.value)

    def _changed(self) -> None:
        """Retain one accepted channel selection before rendering."""
        self._selection_state.select_channels(
            self._capability,
            self.selected,
        )
        self._callback()

    def rebind(
        self,
        options: Sequence[str],
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Keep retained compatible choices and report removed selections."""
        resolved = tuple(options)
        if not resolved or len(resolved) != len(set(resolved)):
            message = "Channel controls require unique compatible options."
            raise ValueError(message)
        retained = self._selection_state.channel_selection(self._capability)
        if retained is None:
            defaults = resolved
            removed: tuple[str, ...] = ()
        else:
            allowed = set(resolved)
            previously_known = set(self._known)
            defaults = tuple(name for name in resolved if name in retained or name not in previously_known)
            removed = tuple(name for name in retained if name not in allowed)
        group = cast(
            "ui.components.CheckboxGroup",
            ui.components.ui_checkbox_channels(
                channels=resolved,
                default_on=defaults,
                labels=labels,
                fixed_columns=3,
            ),
        )
        for checkbox in group.boxes.values():
            checkbox.observe(
                lambda _change: self._changed(),
                names="value",
            )
        self._group = group
        self._known = resolved
        self._selection_state.select_channels(self._capability, defaults)
        self.status.value = "<p>Unavailable channels removed: " + ", ".join(removed) + ".</p>" if removed else ""
        self.status.layout.display = "flex" if removed else "none"
        self.container.children = (
            widgets.HTML(f"<b>{self._title}</b>"),
            cast("widgets.Widget", group),
            self.status,
        )
        self.container.layout.display = "flex"


def _scale_lock(
    selection_state: selection.GeneratedOutputSelectionState,
    *,
    preference_key: str,
) -> widgets.Checkbox:
    """Build one persisted Generation-input-style boolean map-scale lock."""
    return ui.components.ui_checkbox_map_scale_lock(
        value=selection_state.scale_selection(
            preference_key,
            default="individual",
        )
        == "shared"
    )


def _master_case_times(
    frames: Mapping[str, pd.DataFrame],
    case_ids: Mapping[str, str],
) -> tuple[float, ...]:
    """Return the sorted union reaching the longest selected transient case."""
    grids = tuple(
        transient.available_physical_times(frame, case_ids[label]) for label, frame in frames.items() if capabilities.is_transient_frame(frame)
    )
    if not grids:
        return ()
    return ui.time.ordered_time_union(grids)


def make_spatial_case_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build the common steady/transient single-case spatial-field view."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("spatial_fields",),
        include_case=True,
    )
    case_row, case_detail = _compact_scope_controls(None)
    _set_scope_detail(case_detail, tuple(local.case_row.children))
    scale_lock = _scale_lock(
        selection_state,
        preference_key="spatial_field_maps",
    )
    scale_lock_container = widgets.HBox(
        layout=widgets.Layout(display="none"),
    )
    output = ui.components.ui_output_plot()
    time_container = widgets.VBox(
        layout=widgets.Layout(display="none"),
    )
    navigator: ui.time.TimeStepNavigator | None = None

    def show_unavailable(message: str) -> None:
        """Show one legitimate spatial selection message."""
        _show_message(
            output,
            message,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def render() -> None:
        """Render one selected exact case across compatible field subsets."""
        if not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames:
            show_unavailable("Select at least one compatible dataset.")
            return
        if not case_ids:
            show_unavailable("Selected datasets have no shared case number.")
            return
        selected_fields = channel_state.selected
        if not selected_fields:
            show_unavailable("Select at least one compatible spatial field.")
            return
        ui.viewers.render_figure(
            out=output,
            plot_func=plots.transient.plot_spatial_field_comparison,
            kwargs={
                "datasets": frames,
                "case_ids": case_ids,
                "fields": selected_fields,
                "lock_scale": bool(scale_lock.value),
                "physical_time_hours": (None if navigator is None else navigator.selection.physical_time),
            },
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def channels_changed() -> None:
        """Update scale-lock availability before rendering selected fields."""
        frames = local.selected_frames()
        _update_scale_lock_container(
            scale_lock_container,
            scale_lock,
            frames,
            channel_state.selected,
            view="spatial_map",
        )
        render()

    channel_state = _ChannelState(
        title="Spatial fields",
        callback=channels_changed,
        selection_state=selection_state,
        capability="spatial_field_maps",
    )

    def time_changed(chosen: ui.time.TimeStepSelection) -> None:
        """Retain one accepted master physical time before rendering."""
        selection_state.select_physical_time(
            "spatial_field_maps",
            chosen.physical_time,
        )
        render()

    def rebind_and_render() -> None:
        """Rebind field union and optional transient master timeline."""
        nonlocal navigator
        if not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames or not case_ids:
            navigator = None
            _set_optional_children(scale_lock_container, ())
            _set_optional_children(time_container, ())
            render()
            return
        try:
            resolution = capabilities.resolve_fields(
                frames,
                view="spatial_map",
            )
            options = resolution.fields
        except ValueError:
            show_unavailable("Selected datasets expose no spatial fields for this view.")
            return
        channel_state.rebind(
            options,
            labels=capabilities.resolved_field_labels(frames, resolution),
        )
        _update_scale_lock_container(
            scale_lock_container,
            scale_lock,
            frames,
            channel_state.selected,
            view="spatial_map",
        )
        times = _master_case_times(frames, case_ids)
        if times:
            preferred = selection_state.physical_time_selection("spatial_field_maps")
            if navigator is None:
                navigator = ui.time.TimeStepNavigator(
                    times,
                    callback=time_changed,
                    initial_time=preferred,
                )
            else:
                navigator.rebind(
                    times,
                    preserve_time=preferred,
                    notify=False,
                )
            _set_optional_children(time_container, (navigator.widget,))
            selection_state.select_physical_time(
                "spatial_field_maps",
                navigator.selection.physical_time,
            )
        else:
            navigator = None
            _set_optional_children(time_container, ())
        render()

    def activate() -> None:
        """Restore export identity before activating shared selection."""
        _set_export_identity(
            export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
        local.activate()

    def scale_changed(_change: dict[str, Any]) -> None:
        """Retain one accepted map-scale lock before rendering."""
        selection_state.select_scale(
            "spatial_field_maps",
            "shared" if bool(scale_lock.value) else "individual",
        )
        render()

    scale_lock.observe(scale_changed, names="value")
    local.set_callback(rebind_and_render)
    view = ActivatableView(
        (
            case_row,
            local.status,
            channel_state.container,
            scale_lock_container,
            time_container,
            output,
        ),
        activate=activate,
    )
    view.activate()
    return view


def _make_transient_snapshot_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build the single-case schedule-and-state snapshot comparison."""
    required = ("transient_state", "physical_time")
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=required,
        include_case=True,
    )
    case_row, case_detail = _compact_scope_controls(None)
    _set_scope_detail(case_detail, tuple(local.case_row.children))
    scale_lock = _scale_lock(
        selection_state,
        preference_key="transient_snapshot_maps",
    )
    scale_lock_container = widgets.HBox(
        layout=widgets.Layout(display="none"),
    )
    output = ui.components.ui_output_plot()
    time_container = widgets.VBox(
        layout=widgets.Layout(display="none"),
    )
    navigator: ui.time.TimeStepNavigator | None = None

    def show_unavailable(message: str) -> None:
        """Show one legitimate transient selection message."""
        _show_message(
            output,
            message,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def render() -> None:
        """Render schedules and actual held/exact state times once."""
        if not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames:
            show_unavailable("Select at least one transient dataset.")
            return
        if not case_ids:
            show_unavailable("Selected transient datasets have no shared case number.")
            return
        if navigator is None:
            show_unavailable("Selected cases expose no physical-time evidence.")
            return
        channels = channel_state.selected
        if not channels:
            show_unavailable("Select at least one state field.")
            return
        ui.viewers.render_figure(
            out=output,
            plot_func=plots.transient.plot_state_snapshot_comparison,
            kwargs={
                "datasets": frames,
                "case_ids": case_ids,
                "physical_time_hours": navigator.selection.physical_time,
                "channels": channels,
                "lock_scale": bool(scale_lock.value),
            },
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def channels_changed() -> None:
        """Update state-map scale availability before rendering channels."""
        frames = local.selected_frames()
        _update_scale_lock_container(
            scale_lock_container,
            scale_lock,
            frames,
            channel_state.selected,
            view="state_snapshot",
        )
        render()

    channel_state = _ChannelState(
        title="State and conditioning fields",
        callback=channels_changed,
        selection_state=selection_state,
        capability="transient_snapshot_fields",
    )

    def time_changed(chosen: ui.time.TimeStepSelection) -> None:
        """Retain one accepted master time before rendering."""
        selection_state.select_physical_time(
            "transient_snapshot",
            chosen.physical_time,
        )
        render()

    def rebind_and_render() -> None:
        """Rebind the field union and longest-case master timeline."""
        nonlocal navigator
        if not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames or not case_ids:
            navigator = None
            _set_optional_children(scale_lock_container, ())
            _set_optional_children(time_container, ())
            render()
            return
        resolution = capabilities.resolve_fields(
            frames,
            view="state_snapshot",
        )
        channel_state.rebind(
            resolution.fields,
            labels=capabilities.resolved_field_labels(frames, resolution),
        )
        _update_scale_lock_container(
            scale_lock_container,
            scale_lock,
            frames,
            channel_state.selected,
            view="state_snapshot",
        )
        times = _master_case_times(frames, case_ids)
        if not times:
            navigator = None
            _set_optional_children(time_container, ())
            render()
            return
        preferred = selection_state.physical_time_selection("transient_snapshot")
        if navigator is None:
            navigator = ui.time.TimeStepNavigator(
                times,
                callback=time_changed,
                initial_time=preferred,
            )
        else:
            navigator.rebind(
                times,
                preserve_time=preferred,
                notify=False,
            )
        _set_optional_children(time_container, (navigator.widget,))
        selection_state.select_physical_time(
            "transient_snapshot",
            navigator.selection.physical_time,
        )
        render()

    def activate() -> None:
        """Restore export identity before activating shared selection."""
        _set_export_identity(
            export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
        local.activate()

    def scale_changed(_change: dict[str, Any]) -> None:
        """Retain one accepted state-map scale lock."""
        selection_state.select_scale(
            "transient_snapshot_maps",
            "shared" if bool(scale_lock.value) else "individual",
        )
        render()

    scale_lock.observe(scale_changed, names="value")
    local.set_callback(rebind_and_render)
    view = ActivatableView(
        (
            case_row,
            local.status,
            channel_state.container,
            scale_lock_container,
            time_container,
            output,
        ),
        activate=activate,
    )
    view.activate()
    return view


def _make_transient_trajectory_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build schedule-integrated aggregate/single physical trajectories."""
    required = ("transient_state", "physical_time")
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=required,
        include_case=True,
    )
    selected_views = local.selected_views()
    initial_maximum = min(view.case_count for view in selected_views)
    case_count, fewer, more = ui.components.ui_step_case_count(
        start_cases=min(100, initial_maximum),
        min_cases=1,
        max_cases=initial_maximum,
        step_size=50,
    )
    scope = widgets.ToggleButtons(
        options=(("Aggregate", "aggregate"), ("Single case", "single")),
        value=selection_state.scope_selection(
            "transient_trajectories",
            default="aggregate",
        ),
    )
    scope_row, scope_detail = _compact_scope_controls(scope)
    output = ui.components.ui_output_plot()
    updating = {"value": False}

    def show_unavailable(message: str) -> None:
        """Show one legitimate trajectory selection message."""
        _show_message(
            output,
            message,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def update_scope_row() -> None:
        """Expose only the active aggregate-count or exact-case navigator."""
        if scope.value == "aggregate":
            _set_scope_detail(scope_detail, (case_count, fewer, more))
        else:
            _set_scope_detail(scope_detail, tuple(local.case_row.children))

    def render() -> None:
        """Render one exact-coordinate aggregate or selected-case figure."""
        if updating["value"] or not _is_active(
            export_state,
            export_plot_name,
        ):
            return
        frames = local.selected_frames()
        if not frames:
            show_unavailable("Select at least one transient dataset.")
            return
        channels = channel_state.selected
        if not channels:
            show_unavailable("Select at least one transient state channel.")
            return
        plot_function: Callable[..., Figure]
        if scope.value == "aggregate":
            plot_function = plots.transient.plot_state_trajectory_summary
            kwargs = {
                "datasets": frames,
                "max_cases": int(case_count.value),
                "channels": channels,
            }
        else:
            case_ids = local.selected_case_ids()
            if not case_ids:
                show_unavailable("Selected transient datasets have no shared case number.")
                return
            plot_function = plots.transient.plot_state_trajectory_comparison
            kwargs = {
                "datasets": frames,
                "case_ids": case_ids,
                "channels": channels,
            }
        ui.viewers.render_figure(
            out=output,
            plot_func=plot_function,
            kwargs=kwargs,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    channel_state = _ChannelState(
        title="Transient state channels",
        callback=render,
        selection_state=selection_state,
        capability="transient_trajectory_fields",
    )

    def rebind_and_render() -> None:
        """Rebind aggregate bounds and transient output channels."""
        frames = local.selected_frames()
        selected = local.selected_views()
        if selected:
            maximum = min(view.case_count for view in selected)
            updating["value"] = True
            try:
                case_count.max = maximum
                case_count.value = min(case_count.value, maximum)
            finally:
                updating["value"] = False
        if not frames:
            render()
            return
        resolution = capabilities.resolve_fields(
            frames,
            view="state_trajectory",
        )
        channel_state.rebind(
            resolution.fields,
            labels=capabilities.resolved_field_labels(frames, resolution),
        )
        render()

    def step_count(delta: int) -> None:
        """Move the bounded aggregate prefix by one configured increment."""
        case_count.value = max(
            case_count.min,
            min(case_count.max, case_count.value + delta * case_count.step),
        )

    def scope_changed(_change: dict[str, Any]) -> None:
        """Retain and render one accepted trajectory scope."""
        selection_state.select_scope(
            "transient_trajectories",
            str(scope.value),
        )
        update_scope_row()
        render()

    def activate() -> None:
        """Restore export identity before activating shared selection."""
        _set_export_identity(
            export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
        local.activate()

    local.set_callback(rebind_and_render)
    scope.observe(scope_changed, names="value")
    case_count.observe(lambda _change: render(), names="value")
    fewer.on_click(lambda _button: step_count(-1))
    more.on_click(lambda _button: step_count(1))
    update_scope_row()
    view = ActivatableView(
        (scope_row, local.status, channel_state.container, output),
        activate=activate,
    )
    view.activate()
    return view


def make_transient_case_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    kind: _TransientCaseKind,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build one maintained transient snapshot or trajectory view."""
    if kind == "snapshot":
        return _make_transient_snapshot_view(
            catalog=catalog,
            selection_state=selection_state,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
    if kind == "trajectory":
        return _make_transient_trajectory_view(
            catalog=catalog,
            selection_state=selection_state,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
    message = f"Unsupported transient case view kind: {kind!r}."
    raise ValueError(message)


def make_completion_target_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ActivatableView:
    """Build the consolidated completion view from retained structured evidence."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("completion", "physical_time"),
        include_case=False,
    )
    output = ui.components.ui_output_plot()

    def render() -> None:
        """Render the composite and retain its single complete export figure."""
        if not _is_active(export_state, export_plot_name):
            return
        frames = local.selected_frames()
        if not frames:
            _show_message(
                output,
                "Select at least one compatible transient dataset.",
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )
            return
        figure = plots.transient.plot_completion_target_analysis(datasets=frames)
        if export_state is not None:
            _clear_export_figures(export_state)
            export_state.update(
                {
                    "fig": figure,
                    "figures": (figure,),
                    "plot_name": export_plot_name,
                    "title": export_title,
                }
            )
        with output:
            output.clear_output(wait=True)
            display(figure)
        plt.close(figure)

    def activate() -> None:
        """Restore export identity and activate shared transient selection."""
        _set_export_identity(
            export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )
        local.activate()

    local.set_callback(render)
    view = ActivatableView(
        (local.status, output),
        activate=activate,
    )
    view.activate()
    return view
