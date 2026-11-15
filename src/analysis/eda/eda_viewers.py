"""Build shared-state live viewers for adaptive generated-output EDA.

Responsibilities:
  - Bind shared dataset, case, channel, and time state to scientific plots
  - Render mixed-capability aggregate, spatial, spectral, and transient views lazily
  - Preserve channel selections and exact physical time across compatible changes
  - Keep active figures synchronized with explicit panel-local export state
  - Show a view that fails to load, resolve, or render in place and clear its export target

Design principles:
  - Registered factories expose explicit export signatures
  - Selection changes load only selected capability-compatible views
  - View deactivation preserves cached scientific selections, stops all shared-state rendering, and supersedes renders still running
  - Loading, rebinding, and rendering of one change run inside one failure boundary of the view
  - Every control is a control-standard element in a panel row; an element that does not apply is hidden without space

This module does NOT:
  - Discover storage, define presentation order, or implement plot mathematics
  - Catch programming errors outside the failure boundary of the view
  - Reinterpret unavailable scientific capabilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np

from src.analysis import ui
from src.analysis.presentation import analysis_moisture_display as moisture_display

from . import eda_capabilities as capabilities
from . import eda_controls as controls
from . import eda_dataframe as dataframe
from . import eda_selection as selection
from . import eda_transient as transient
from . import plots

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import ipywidgets as widgets
    import pandas as pd
    from matplotlib.figure import Figure

    from src.analysis.presentation import analysis_display_grid as display_grid

    type Action = Callable[[], None]

_TransientCaseKind = Literal["snapshot", "trajectory"]
_MINIMUM_COMPARABLE_MAP_COUNT = 2
_NO_CHANNEL: Final = "—"  # stands in the channel row until the first dataset selection offers real channels
_NO_TENSOR: Final = "—"
_DEFAULT_CASE_COUNT: Final = 100
_CASE_COUNT_STEP: Final = 50


def _export(export_state: dict[str, Any] | None, export_plot_name: str | None, export_title: str | None) -> dict[str, Any]:
    """Return the export identity arguments of the render helpers."""
    return {"export_state": export_state, "export_plot_name": export_plot_name, "export_title": export_title}


def _surface(
    local: controls.GeneratedOutputControls, rows: Sequence[widgets.Widget], export: Mapping[str, Any]
) -> tuple[ui.panel.ViewPanel, Callable[[Action], Action]]:
    """
    Build the panel of one EDA view and the decorator that runs a view action inside the panel's failure boundary.

    A failing action shows its failure in the plot's place and releases the export target, so no figure of the previous
    selection stays exportable; the failure supersedes every render still running in the same step, so none of them can
    clear it or export again. Hiding the panel supersedes them in the same way.
    """
    view = ui.panel.ViewPanel(rows, status=local.status, slot=ui.panel.PlotSlot(), on_activate=local.activate, on_deactivate=local.deactivate)

    def guarded(action: Action) -> Action:
        def run() -> None:
            try:
                action()
            except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
                ui.viewers.fail_view(view, error, **export)

        return run

    return view, guarded


class _Channels:
    """Direct channel checkboxes bound to the retained channel choice of one capability; the row shows once channels exist."""

    def __init__(
        self,
        title: str,
        selection_state: selection.GeneratedOutputSelectionState,
        capability: str,
        status: ui.controls.StatusLine,
    ) -> None:
        """Create the hidden row."""
        self._state = selection_state
        self._capability = capability
        self._status = status
        self.checks = ui.controls.ChannelChecks((_NO_CHANNEL,), labels={_NO_CHANNEL: _NO_CHANNEL}, title=title)
        ui.controls.set_visible(self.checks.widget, False)

    @property
    def selected(self) -> tuple[str, ...]:
        """Return the checked channels in canonical visible order."""
        return self.checks.selected

    def observe(self, on_change: Action) -> None:
        """Call ``on_change()`` after every accepted user change, once the choice is retained."""
        self.checks.observe(lambda: self._changed(on_change))

    def rebind(self, options: Sequence[str], labels: Mapping[str, str]) -> None:
        """Offer ``options``, keep the still valid retained choice, and name removed channels on the status line."""
        retained = self._state.channel_selection(self._capability)
        removed = self.checks.rebind(options, labels=labels, retained=retained or ())
        self._state.select_channels(self._capability, self.checks.selected)
        ui.controls.set_visible(self.checks.widget, True)
        if removed:
            self._status.set(f"Unavailable channels removed: {', '.join(removed)}.", "warning")

    def _changed(self, on_change: Action) -> None:
        """Retain one accepted selection before rendering."""
        self._state.select_channels(self._capability, self.checks.selected)
        on_change()


class _TimeAxis:
    """The master physical-time navigator of a view; it shows only while the selected cases share stored times."""

    def __init__(
        self,
        selection_state: selection.GeneratedOutputSelectionState,
        key: str,
        status: ui.controls.StatusLine,
    ) -> None:
        """Create the hidden navigator."""
        self._state = selection_state
        self._key = key
        self.timed = False
        self.navigator = ui.navigation.TimeNavigator((0.0,), status=status)
        ui.controls.set_visible(self.navigator.widget, False)
        status.clear()  # the placeholder time is not a selection

    @property
    def physical_time(self) -> float | None:
        """Return the selected physical time in hours, or None while no stored times are offered."""
        return self.navigator.selection.physical_time if self.timed else None

    def observe(self, on_change: Action) -> None:
        """Call ``on_change()`` after every accepted user change, once the time is retained."""
        self.navigator.set_callback(lambda chosen: self._chosen(chosen, on_change))

    def rebind(self, times: Sequence[float]) -> None:
        """Offer the stored ``times`` with the retained time selected, or hide the navigator when there are none."""
        self.timed = bool(times)
        ui.controls.set_visible(self.navigator.widget, self.timed)
        if not times:
            return
        preferred = self._state.physical_time_selection(self._key)
        self.navigator.rebind(times, preserve_time=times[0] if preferred is None else preferred, notify=False)
        self._state.select_physical_time(self._key, self.navigator.selection.physical_time)

    def _chosen(self, chosen: ui.navigation.TimeStepSelection, on_change: Action) -> None:
        """Retain one accepted master physical time before rendering."""
        self._state.select_physical_time(self._key, chosen.physical_time)
        on_change()


class _CaseCount:
    """The aggregate case-count control of a view; its bound follows the largest selected dataset."""

    def __init__(self, selected_views: Sequence[selection.EDAView]) -> None:
        """Create the control on the smallest selected dataset, capped at the default count."""
        counts = [item.case_count for item in selected_views]
        self._updating = False
        self._on_change: Action | None = None
        self.control = ui.navigation.CountControl(
            value=min(_DEFAULT_CASE_COUNT, min(counts, default=1)),
            minimum=1,
            maximum=max(counts, default=1),
            step=_CASE_COUNT_STEP,
            on_change=self._changed,
        )

    @property
    def value(self) -> int:
        """Return the current case count."""
        return self.control.value

    def observe(self, on_change: Action) -> None:
        """Call ``on_change()`` after every user change of the count."""
        self._on_change = on_change

    def _changed(self, _count: int) -> None:
        if self._on_change is not None and not self._updating:
            self._on_change()

    def rebind(self, selected_views: Sequence[selection.EDAView]) -> None:
        """Move the bound to the largest selected dataset and cut the count to it, without a redraw."""
        if not selected_views:
            return
        maximum = max(item.case_count for item in selected_views)
        self._updating = True
        try:
            self.control.slider.max = maximum
            self.control.slider.value = min(self.control.value, maximum)
            self.control.disabled = False  # re-evaluates the step buttons against the new bound
        finally:
            self._updating = False


def _scope_group(selection_state: selection.GeneratedOutputSelectionState, capability: str) -> widgets.ToggleButtons:
    """Build the Aggregate or Single case choice with the retained scope of ``capability``."""
    scope = ui.controls.scope_group()
    scope.value = selection_state.scope_selection(capability, default="aggregate")
    return scope


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


def _scale_lock(selection_state: selection.GeneratedOutputSelectionState, *, preference_key: str) -> widgets.Checkbox:
    """Build one persisted Generation-input-style boolean map-scale lock, hidden until it can matter."""
    lock = ui.controls.scale_lock_checkbox(value=selection_state.scale_selection(preference_key, default="individual") == "shared")
    ui.controls.set_visible(lock, False)
    return lock


def _update_scale_lock(
    lock: widgets.Checkbox,
    frames: Mapping[str, pd.DataFrame],
    fields: Sequence[str],
    *,
    view: capabilities.FieldView,
) -> None:
    """Show the scale lock only when shared normalization can affect maps."""
    ui.controls.set_visible(lock, _scale_lock_is_meaningful(frames, fields, view=view))


def _shown_count(nested: ui.panel.ViewPanel) -> int:
    """Return the case count a nested case-count viewer shows: the slider of the first element of its first row."""
    count_row = cast("Any", nested.rows[0])
    return int(count_row.children[0].children[0].value)


def make_statistics_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    plot_function: Callable[..., ui.panel.ViewPanel],
    required_capabilities: Sequence[selection.GeneratedOutputCapability],
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ui.panel.ViewPanel:
    """Build one capability-adaptive established statistics viewer."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=required_capabilities,
        include_case=False,
    )
    export = _export(export_state, export_plot_name, export_title)
    view, guarded = _surface(local, (), export)
    nested: ui.panel.ViewPanel | None = None

    @guarded
    def update() -> None:
        """Rebuild the established case-count viewer for the current datasets, at the case count it showed before."""
        nonlocal nested
        frames = local.selected_frames()
        if not frames:
            ui.viewers.clear_view(view, "Select at least one compatible dataset.", **export)
            return
        carried: dict[str, int] = {} if nested is None else {"resume_cases": _shown_count(nested)}
        built: list[ui.panel.ViewPanel] = []

        def build() -> widgets.Widget:
            # The viewer publishes its own figures to the export state while the render of this view that built it is
            # current, so hiding or rebuilding this view supersedes its renders, those of its own construction included.
            built.append(plot_function(datasets=frames, host=view, **carried, **export))
            return built[0].widget

        if ui.viewers.render_view(view, build) and built:  # a failed build published its failure and left nothing to keep
            nested = built[0]

    local.set_callback(update)
    local.activate()
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
) -> ui.panel.ViewPanel:
    """Build one capability-adaptive spectral viewer with shared scope and channels."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("spectral",),
        include_case=True,
    )
    export = _export(export_state, export_plot_name, export_title)
    case = local.case
    count = _CaseCount(local.selected_views()) if aggregate_plot_function is not None else None
    scope = _scope_group(selection_state, "spectral") if count is not None else None
    semantic = dict(semantic_controls or {})
    channels = _Channels("Spectral channels", selection_state, "spectral", local.status)
    selection_row = [case.widget] if count is None or scope is None else [scope, count.control.widget, case.widget]
    rows = [ui.panel.row(*selection_row), ui.panel.row(channels.checks.widget)]
    if semantic:
        rows.append(ui.panel.row(*semantic.values()))
    view, guarded = _surface(local, rows, export)

    @guarded
    def render() -> None:
        """Render one aggregate or exact shared-case spectral figure."""
        if not view.active:
            return
        frames = local.selected_frames()
        if not frames:
            ui.viewers.clear_view(view, "Select at least one compatible dataset.", **export)
            return
        selected_channels = channels.selected
        if not selected_channels:
            ui.viewers.clear_view(view, "Select at least one compatible spectral channel.", **export)
            return
        semantic_values = {name: widget.value for name, widget in semantic.items()}
        if scope is not None and scope.value == "aggregate":
            if aggregate_plot_function is None or count is None:
                message = "Aggregate spectral plotting is unavailable."
                raise RuntimeError(message)
            plot_function = aggregate_plot_function
            plot_kwargs = {"datasets": frames, "max_cases": count.value, "channels": selected_channels, **semantic_values}
        else:
            case_number = local.selected_case_number
            if case_number is None:
                ui.viewers.clear_view(view, "Selected datasets have no shared case number.", **export)
                return
            plot_function = single_plot_function
            plot_kwargs = {"datasets": frames, "case_number": case_number, "channels": selected_channels, **semantic_values}
        ui.viewers.render_view(view, plot_function, kwargs=plot_kwargs, **export)

    def update_scope_row() -> None:
        """Expose only the active aggregate-count or exact-case navigator."""
        aggregate = scope is not None and scope.value == "aggregate"
        if count is not None:
            ui.controls.set_visible(count.control.widget, aggregate)
        ui.controls.set_visible(case.widget, not aggregate)

    @guarded
    def rebind_and_render() -> None:
        """Rebind bounds and task-compatible channels before one render."""
        frames = local.selected_frames()
        if count is not None:
            count.rebind(local.selected_views())
        if not frames:
            render()
            return
        resolution = capabilities.resolve_fields(frames, view="spectral")
        if not resolution.fields:
            ui.viewers.clear_view(view, "Selected datasets expose no compatible spectral channels.", **export)
            return
        channels.rebind(resolution.fields, capabilities.resolved_field_labels(frames, resolution))
        render()

    def scope_changed(_change: dict[str, Any]) -> None:
        """Retain and render one accepted scientific scope change."""
        if scope is not None:
            selection_state.select_scope("spectral", str(scope.value))
        update_scope_row()
        render()

    channels.observe(render)
    for widget in semantic.values():
        widget.observe(lambda _change: render(), names="value")
    if count is not None and scope is not None:
        count.observe(render)
        scope.observe(scope_changed, names="value")
    update_scope_row()
    local.set_callback(rebind_and_render)
    local.activate()
    return view


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
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> ui.panel.ViewPanel:
    """Build the common steady/transient single-case spatial-field view."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("spatial_fields",),
        include_case=True,
    )
    export = _export(export_state, export_plot_name, export_title)
    scale_lock = _scale_lock(selection_state, preference_key="spatial_field_maps")
    time_axis = _TimeAxis(selection_state, "spatial_field_maps", local.status)
    channels = _Channels("Spatial fields", selection_state, "spatial_field_maps", local.status)
    rows = [
        ui.panel.row(local.case.widget),
        ui.panel.row(channels.checks.widget),
        ui.panel.row(scale_lock),
        ui.panel.row(time_axis.navigator.widget),
    ]
    view, guarded = _surface(local, rows, export)

    @guarded
    def render() -> None:
        """Render one selected exact case across compatible field subsets."""
        if not view.active:
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames:
            ui.viewers.clear_view(view, "Select at least one compatible dataset.", **export)
            return
        if not case_ids:
            ui.viewers.clear_view(view, "Selected datasets have no shared case number.", **export)
            return
        selected_fields = channels.selected
        if not selected_fields:
            ui.viewers.clear_view(view, "Select at least one compatible spatial field.", **export)
            return
        ui.viewers.render_view(
            view,
            plots.transient.plot_spatial_field_comparison,
            kwargs={
                "datasets": frames,
                "display_spatial_stride": display_spatial_stride,
                "case_ids": case_ids,
                "fields": selected_fields,
                "lock_scale": bool(scale_lock.value),
                "physical_time_hours": time_axis.physical_time,
            },
            **export,
        )

    @guarded
    def channels_changed() -> None:
        """Update scale-lock availability before rendering selected fields."""
        _update_scale_lock(scale_lock, local.selected_frames(), channels.selected, view="spatial_map")
        render()

    @guarded
    def rebind_and_render() -> None:
        """Rebind field union and optional transient master timeline."""
        if not view.active:
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames or not case_ids:
            time_axis.rebind(())
            ui.controls.set_visible(scale_lock, False)
            render()
            return
        try:
            resolution = capabilities.resolve_fields(frames, view="spatial_map")
        except ValueError:
            ui.viewers.clear_view(view, "Selected datasets expose no spatial fields for this view.", **export)
            return
        time_axis.rebind(_master_case_times(frames, case_ids))
        channels.rebind(resolution.fields, capabilities.resolved_field_labels(frames, resolution))
        _update_scale_lock(scale_lock, frames, channels.selected, view="spatial_map")
        render()

    def scale_changed(_change: dict[str, Any]) -> None:
        """Retain one accepted map-scale lock before rendering."""
        selection_state.select_scale("spatial_field_maps", "shared" if bool(scale_lock.value) else "individual")
        render()

    channels.observe(channels_changed)
    time_axis.observe(render)
    scale_lock.observe(scale_changed, names="value")
    local.set_callback(rebind_and_render)
    local.activate()
    return view


def _make_transient_snapshot_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> ui.panel.ViewPanel:
    """Build the single-case schedule-and-state snapshot comparison."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("transient_state", "physical_time"),
        include_case=True,
    )
    export = _export(export_state, export_plot_name, export_title)
    scale_lock = _scale_lock(selection_state, preference_key="transient_snapshot_maps")
    time_axis = _TimeAxis(selection_state, "transient_snapshot", local.status)
    channels = _Channels("State and conditioning fields", selection_state, "transient_snapshot_fields", local.status)
    rows = [
        ui.panel.row(local.case.widget),
        ui.panel.row(channels.checks.widget),
        ui.panel.row(scale_lock),
        ui.panel.row(time_axis.navigator.widget),
    ]
    view, guarded = _surface(local, rows, export)

    @guarded
    def render() -> None:
        """Render schedules and actual held/exact state times once."""
        if not view.active:
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames:
            ui.viewers.clear_view(view, "Select at least one transient dataset.", **export)
            return
        if not case_ids:
            ui.viewers.clear_view(view, "Selected transient datasets have no shared case number.", **export)
            return
        if not time_axis.timed:
            ui.viewers.clear_view(view, "Selected cases expose no physical-time evidence.", **export)
            return
        selected_channels = channels.selected
        if not selected_channels:
            ui.viewers.clear_view(view, "Select at least one state field.", **export)
            return
        ui.viewers.render_view(
            view,
            plots.transient.plot_state_snapshot_comparison,
            kwargs={
                "datasets": frames,
                "display_spatial_stride": display_spatial_stride,
                "case_ids": case_ids,
                "physical_time_hours": time_axis.physical_time,
                "channels": selected_channels,
                "lock_scale": bool(scale_lock.value),
            },
            **export,
        )

    @guarded
    def channels_changed() -> None:
        """Update state-map scale availability before rendering channels."""
        _update_scale_lock(scale_lock, local.selected_frames(), channels.selected, view="state_snapshot")
        render()

    @guarded
    def rebind_and_render() -> None:
        """Rebind the field union and longest-case master timeline."""
        if not view.active:
            return
        frames = local.selected_frames()
        case_ids = local.selected_case_ids()
        if not frames or not case_ids:
            time_axis.rebind(())
            ui.controls.set_visible(scale_lock, False)
            render()
            return
        resolution = capabilities.resolve_fields(frames, view="state_snapshot")
        time_axis.rebind(_master_case_times(frames, case_ids))
        channels.rebind(resolution.fields, capabilities.resolved_field_labels(frames, resolution))
        _update_scale_lock(scale_lock, frames, channels.selected, view="state_snapshot")
        render()

    def scale_changed(_change: dict[str, Any]) -> None:
        """Retain one accepted state-map scale lock."""
        selection_state.select_scale("transient_snapshot_maps", "shared" if bool(scale_lock.value) else "individual")
        render()

    channels.observe(channels_changed)
    time_axis.observe(render)
    scale_lock.observe(scale_changed, names="value")
    local.set_callback(rebind_and_render)
    local.activate()
    return view


def _make_transient_trajectory_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
) -> ui.panel.ViewPanel:
    """Build schedule-integrated aggregate/single physical trajectories."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("transient_state", "physical_time"),
        include_case=True,
    )
    export = _export(export_state, export_plot_name, export_title)
    count = _CaseCount(local.selected_views())
    scope = _scope_group(selection_state, "transient_trajectories")
    channels = _Channels("Transient state channels", selection_state, "transient_trajectory_fields", local.status)
    rows = [ui.panel.row(scope, count.control.widget, local.case.widget), ui.panel.row(channels.checks.widget)]
    view, guarded = _surface(local, rows, export)

    @guarded
    def render() -> None:
        """Render one exact-coordinate aggregate or selected-case figure."""
        if not view.active:
            return
        frames = local.selected_frames()
        if not frames:
            ui.viewers.clear_view(view, "Select at least one transient dataset.", **export)
            return
        selected_channels = channels.selected
        if not selected_channels:
            ui.viewers.clear_view(view, "Select at least one transient state channel.", **export)
            return
        plot_function: Callable[..., Figure]
        if scope.value == "aggregate":
            plot_function = plots.transient.plot_state_trajectory_summary
            kwargs: dict[str, Any] = {"datasets": frames, "max_cases": count.value, "channels": selected_channels}
        else:
            case_ids = local.selected_case_ids()
            if not case_ids:
                ui.viewers.clear_view(view, "Selected transient datasets have no shared case number.", **export)
                return
            plot_function = plots.transient.plot_state_trajectory_comparison
            kwargs = {"datasets": frames, "case_ids": case_ids, "channels": selected_channels}
        ui.viewers.render_view(view, plot_function, kwargs=kwargs, **export)

    def update_scope_row() -> None:
        """Expose only the active aggregate-count or exact-case navigator."""
        aggregate = scope.value == "aggregate"
        ui.controls.set_visible(count.control.widget, aggregate)
        ui.controls.set_visible(local.case.widget, not aggregate)

    @guarded
    def rebind_and_render() -> None:
        """Rebind aggregate bounds and transient output channels."""
        frames = local.selected_frames()
        count.rebind(local.selected_views())
        if not frames:
            render()
            return
        resolution = capabilities.resolve_fields(frames, view="state_trajectory")
        channels.rebind(resolution.fields, capabilities.resolved_field_labels(frames, resolution))
        render()

    def scope_changed(_change: dict[str, Any]) -> None:
        """Retain and render one accepted trajectory scope."""
        selection_state.select_scope("transient_trajectories", str(scope.value))
        update_scope_row()
        render()

    channels.observe(render)
    count.observe(render)
    scope.observe(scope_changed, names="value")
    update_scope_row()
    local.set_callback(rebind_and_render)
    local.activate()
    return view


def make_transient_case_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    kind: _TransientCaseKind,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> ui.panel.ViewPanel:
    """Build one maintained transient snapshot or trajectory view."""
    if kind == "snapshot":
        return _make_transient_snapshot_view(
            display_spatial_stride=display_spatial_stride,
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
) -> ui.panel.ViewPanel:
    """Build the consolidated completion view from retained structured evidence."""
    local = controls.GeneratedOutputControls(
        catalog,
        selection_state=selection_state,
        required_capabilities=("completion", "physical_time"),
        include_case=False,
    )
    export = _export(export_state, export_plot_name, export_title)
    view, guarded = _surface(local, (), export)

    @guarded
    def render() -> None:
        """Render the composite and retain its single complete export figure."""
        frames = local.selected_frames()
        if not frames:
            ui.viewers.clear_view(view, "Select at least one compatible transient dataset.", **export)
            return
        ui.viewers.render_view(view, plots.transient.plot_completion_target_analysis, kwargs={"datasets": frames}, **export)

    local.set_callback(render)
    local.activate()
    return view


def make_published_tensor_view(
    *,
    catalog: selection.GeneratedOutputEDACatalog,
    selection_state: selection.GeneratedOutputSelectionState,
    export_state: dict[str, Any] | None,
    export_plot_name: str | None,
    export_title: str | None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> ui.panel.ViewPanel:
    """Inspect one actual package case and retain its selected tensor for export."""
    local = controls.GeneratedOutputControls(catalog, selection_state=selection_state, include_case=True)
    export = _export(export_state, export_plot_name, export_title)
    field = ui.controls.dropdown([(_NO_TENSOR, ("", ""))], description="Tensor")  # a selected placeholder, so new options select their first

    def transition_changed(_chosen: ui.navigation.TimeStepSelection) -> None:
        update()

    transition = ui.navigation.TimeNavigator((0.0,), status=local.status, on_change=transition_changed)
    ui.controls.set_visible(transition.widget, False)
    local.status.clear()  # the placeholder time is not a selection
    binding_controls = False
    rows = [ui.panel.row(local.case.widget), ui.panel.row(field, transition.widget)]
    view, guarded = _surface(local, rows, export)

    def draw(package: selection.PublishedDatasetEDAView, case_id: str) -> Figure:
        """Bind the tensor and transition controls to the admitted cached case and plot the selected tensor."""
        nonlocal binding_controls
        frame = package.load()
        row = frame.loc[case_id]
        is_transient = package.task_id == "transient_drying"
        binding_controls = True
        try:
            ui.controls.set_visible(transition.widget, is_transient)
            if is_transient:
                transition.rebind(tuple(float(time) for time in row["time"]["transition_t_n_hours"]), notify=False)
            values = dataframe.published_tensor_values(frame, case_id, transition=transition.selection.position if is_transient else 0)
            options = tuple(
                (
                    f"{value.group}: {value.name} [{moisture_display.field_display_unit(value.name, value.unit, control=True)}]",
                    (value.group, value.name),
                )
                for value in values
            )
            if field.options != options:
                previous = field.value
                field.options = options
                field.layout.width = f"{ui.controls.dropdown_width('Tensor', [label for label, _key in options])}px"
                if previous in tuple(key for _label, key in options):
                    field.value = previous
        finally:
            binding_controls = False
        selected = next(value for value in values if (value.group, value.name) == field.value)
        coordinates = row["static_fields"] if is_transient else row
        stored: tuple[float, float] | None = None
        context = f"Case {case_id}: {selected.purpose}; {selected.support}. {frame.attrs['normalization']}"
        if is_transient:
            index = transition.selection.position
            stored = (float(row["time"]["transition_t_n_hours"][index]), float(row["time"]["transition_t_n_plus_1_hours"][index]))
            context = f"Time {stored[0]:g} h. {context}"
        local.status.set(context)
        return plots.case_statistics.plot_published_tensor(
            selected,
            display_spatial_stride=display_spatial_stride,
            x=np.asarray(coordinates["x"]),
            y=np.asarray(coordinates["y"]),
            case_label=f"case {local.selected_case_number}",
            transition=stored,
            rho_bu_dry=capabilities.field_values(frame, cast("pd.Series", row), "rho_bu_dry") if is_transient else None,
        )

    @guarded
    def update() -> None:
        """Use the admitted cached case; selection changes never fit a normalizer."""
        if binding_controls or not view.active:
            return
        views = local.selected_views()
        package = views[0] if len(views) == 1 else None
        if not isinstance(package, selection.PublishedDatasetEDAView):
            ui.viewers.clear_view(view, "Select exactly one published Dataset for tensor inspection.", **export)
            return
        case_id = local.selected_case_ids().get(package.label)
        if case_id is not None:
            ui.viewers.render_view(view, draw, args=(package, case_id), **export)

    field.observe(lambda _change: update(), names="value")
    local.set_callback(update)
    local.activate()
    return view
