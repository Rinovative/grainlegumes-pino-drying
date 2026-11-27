"""Catalog every control of the notebook control standard on small synthetic data for the plot-standard notebook.

Responsibilities:
  - Hold the synthetic runs, dataset inventory, case IDs, stored times, and channels the entries use
  - Declare one catalog entry per control, built by the real ui constructors
  - Render an entry as the stylesheet, its collapsed source code, and the built control or figure

Design principles:
  - Entries call the maintained constructors, never copies of them, so the catalog shows the standard as built
  - Synthetic data only; the catalog reads no storage and needs no frontend to build
  - An entry result is a widget or a Matplotlib figure; anything else is refused

This module does NOT:
  - Hold scientific plot entries; plot entries live with the plot catalog
  - Decide which control a real view uses
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Final

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.figure import Figure

from src.analysis import presentation

from . import analysis_ui_controls as controls
from . import analysis_ui_navigation as navigation
from . import analysis_ui_notebook as notebook
from . import analysis_ui_panel as panel
from . import analysis_ui_selectors as selectors
from . import analysis_ui_style as style

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One catalog entry: its key, its heading, the function that builds it, and the control entries it uses."""

    key: str
    title: str
    build: Callable[[], object]
    uses: tuple[str, ...] = ()


# ----------------------------------------------------------------------------- synthetic data

RUNS: Final = (
    selectors.RunOption("fno_m128x160_h64_l3_p0p1__lr-0p000388", "fno", "fno-1"),
    selectors.RunOption("fno_m64x80_h32_l4_p0p1__lr-0p000500", "fno", "fno-2"),
    selectors.RunOption("uno_m64x64_h32_l7_p0p1_s1-05-05-1-1-2-2_r0p495__lr-0p000532", "uno", "uno-1"),
    selectors.RunOption("rno_m24x24_h16_l3_p0p1__lr-0p000388__a0rl-32", "rno", "rno-1"),
    selectors.RunOption("pi-fno_m128x160_h64_l3_p0p1__fd__w-1-1000-100__lr-0p000388", "pi-fno", "pi-fno-1"),
    selectors.RunOption("pi-uno_m64x64_h32_l7_p0p1_s1-05-05-1-1-2-2_r0p495__fd__lr-0p000532", "pi-uno", "pi-uno-1"),
    selectors.RunOption("fno_m32x32_h16_l2_p0p1__lr-0p001", None, "broken-1", issue="physics block missing in the saved config"),
)
_ROLE_MATERIALS: Final = (("id", ("lentil", "chickpea", "field_pea")), ("parameter_ood", ("lentil", "chickpea")), ("material_ood", ("kidney_bean",)))
_GROUP_LABELS: Final = {
    role: presentation.display_labels.evaluation_role_display_label(role) for role in presentation.dataset_selection.EVALUATION_GROUP_ORDER
}


def _item(material: str, role: str) -> presentation.dataset_selection.DatasetItem:
    """Return one synthetic (material, role) dataset item."""
    return presentation.dataset_selection.DatasetItem(
        key=(material, role), material=material, group=role, label=presentation.dataset_selection.dataset_label(material, role)
    )


INVENTORY: Final = presentation.dataset_selection.DatasetInventory(
    [_item(material, role) for role, materials in _ROLE_MATERIALS for material in materials],
    group_order=presentation.dataset_selection.EVALUATION_GROUP_ORDER,
    group_labels=_GROUP_LABELS,
    poolable=True,
    unavailable=[(_item("field_pea", "parameter_ood"), "no common cases in uno_m64x64_h32_l7")],
)
POOL_REASON: Final = "Pooling is not material-balanced in this view"
CASE_IDS: Final = (3, 7, 12, 40)
CASE_LABELS: Final = {
    3: "lentil · case 3 · ε 0.38 · p_in 120 Pa",
    7: "lentil · case 7 · ε 0.41 · p_in 95 Pa",
    12: "chickpea · case 12 · ε 0.36 · p_in 140 Pa",
    40: "kidney bean · case 40 · ε 0.44 · p_in 80 Pa",
}
TIMES_H: Final = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0)
TRANSITION_STARTS_H: Final = (0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0)
CHANNELS: Final = ("T", "phi", "w_surf", "w_int", "w_gr")
VIEW_TITLES: Final = (
    ("1. Overview", "2. Fields", "3. Errors", "4. Physical Consistency", "5. Runtime", "6. Spectral & Representation Analysis"),
    ("Settings", "5 Modes", "6 Stride"),
)


# ----------------------------------------------------------------------------- entry builds


def _figure(label: str) -> Figure:
    """Return a small synthetic line figure."""
    figure = Figure(figsize=(4.0, 2.4), dpi=100)
    axis = figure.add_subplot()
    axis.plot([0, 1, 2, 3], [0.0, 0.6, 0.8, 0.9], label=label)
    axis.set_xlabel("Time [h]")
    axis.set_ylabel("Error [1]")
    axis.set_title(label)
    figure.tight_layout()
    return figure


def _dropdowns() -> widgets.Widget:
    model = controls.dropdown([run.name for run in RUNS[:4]], description="Model")
    channel = controls.channel_dropdown()
    parameter = controls.dropdown(["porosity ε", "inlet pressure p_in", "permeability κxx"], description="Parameter")
    return panel.ViewPanel([panel.row(model), panel.row(channel, parameter)]).widget


def _column_grid() -> widgets.Widget:
    status = controls.StatusLine()
    navigator = navigation.CaseNavigator(CASE_IDS, status=status)
    count = navigation.CountControl(value=20, minimum=1, maximum=50, step=5)
    rows = [
        panel.row(controls.channel_dropdown(), controls.checkbox("Normalise"), controls.toggle("Show residual")),
        panel.row(navigator.widget, controls.scale_mode_group()),
        panel.row(count.widget, controls.scale_lock_checkbox(value=False)),
    ]
    return panel.ViewPanel(rows, status=status).widget


def _channels() -> widgets.Widget:
    checks = controls.ChannelChecks(CHANNELS)
    status = controls.StatusLine()

    def report() -> None:
        status.set(f"Selected: {', '.join(checks.selected) or 'none'}")

    checks.observe(report)
    return panel.ViewPanel([panel.row(checks.widget)], status=status).widget


def _case_navigator() -> widgets.Widget:
    status = controls.StatusLine()
    navigator = navigation.CaseNavigator(CASE_IDS, labels=CASE_LABELS, status=status)
    return panel.ViewPanel([panel.row(navigator.widget)], status=status).widget


def _time_navigator() -> widgets.Widget:
    status = controls.StatusLine()
    navigator = navigation.TimeNavigator(TIMES_H, status=status)
    return panel.ViewPanel([panel.row(navigator.widget)], status=status).widget


def _transition() -> widgets.Widget:
    status = controls.StatusLine()
    task = controls.choice_group((("Transient", "transient"), ("Steady", "steady")), description="Task")
    navigator = navigation.TimeNavigator(TRANSITION_STARTS_H, status=status)

    def task_changed(_change: object) -> None:
        controls.set_visible(navigator.widget, task.value == "transient")

    task.observe(task_changed, names="value")
    return panel.ViewPanel([panel.row(task), panel.row(navigator.widget)], status=status).widget


def _count() -> widgets.Widget:
    status = controls.StatusLine()
    count = navigation.CountControl(value=10, minimum=1, maximum=40, step=10, on_change=lambda value: status.set(f"{value} cases"))
    return panel.ViewPanel([panel.row(count.widget)], status=status).widget


def _choice_groups() -> widgets.Widget:
    rows = [
        panel.row(controls.error_mode_group("steady")),
        panel.row(controls.scale_mode_group(), controls.kappa_scale_group()),
        panel.row(controls.scope_group(), controls.choice_group(("Mean", "Median", "Max"), description="Statistic")),
    ]
    return panel.ViewPanel(rows).widget


def _toggles() -> widgets.Widget:
    row = panel.row(controls.toggle("Show residual"), controls.checkbox("Normalise"), controls.scale_lock_checkbox(value=False))
    return panel.ViewPanel([row]).widget


def _actions() -> widgets.Widget:
    row = panel.row(
        controls.action_button("Steady flow - Open", kind="open"),
        controls.action_button("Close", kind="close"),
        controls.action_button("Export PDF", kind="export"),
        controls.action_button("Export GIF", kind="export"),
        controls.action_button(controls.COMPUTE_LABEL, kind="compute"),
    )
    return panel.ViewPanel([row]).widget


def _status() -> widgets.Widget:
    normal = controls.StatusLine("Case 7 of 4 cases · lentil, ID")
    warning = controls.StatusLine("Case 8 is not available; showing the nearest case 7.", "warning")
    long_error = " ".join(["A long status text wraps over several lines instead of being cut or shown only on hover."] * 4)
    error = controls.StatusLine(long_error, "error")
    progress = controls.StatusLine()
    progress.progress("[GIF] Frame", 12, 48)
    box = widgets.VBox(tuple(line.widget for line in (normal, warning, error, progress)))
    box.add_class(style.ROOT_CLASS)
    return box


def _failure() -> widgets.Widget:
    display_ = controls.FailureDisplay()
    try:
        message = "synthetic failure of a view"
        raise ValueError(message)  # noqa: TRY301 -- the catalog shows a real caught failure with its traceback
    except ValueError as error:
        display_.show(error)
    box = widgets.VBox((display_.widget,))
    box.add_class(style.ROOT_CLASS)
    return box


def _selector_panel(selector: selectors.ChoiceBlock) -> widgets.Widget:
    """Return a one-row panel whose popover is the selector's body."""
    status = controls.StatusLine(", ".join(map(str, selector.selected())))
    selector.observe(lambda: status.set(", ".join(map(str, selector.selected()))))
    return panel.ViewPanel([panel.row(selector.header)], status=status, popovers=[(selector.header, selector.body)]).widget


def _runs_multiple() -> widgets.Widget:
    valid = [run for run in RUNS if run.family is not None]
    return _selector_panel(selectors.RunSelector(valid, multiple=True, selected=[run.key for run in valid]))


def _run_single() -> widgets.Widget:
    return _selector_panel(selectors.RunSelector(RUNS, multiple=False))


def _datasets_multiple() -> widgets.Widget:
    return _selector_panel(selectors.DatasetSelector(INVENTORY, multiple=True, selected=[item.key for item in INVENTORY.items]))


def _datasets_pool_disabled() -> widgets.Widget:
    selector = selectors.DatasetSelector(INVENTORY, multiple=True, selected=[item.key for item in INVENTORY.items], pool_reason=POOL_REASON)
    return _selector_panel(selector)


def _dataset_single() -> widgets.Widget:
    return _selector_panel(selectors.DatasetSelector(INVENTORY, multiple=False, selected=(), pool_reason=POOL_REASON))


def _on_demand() -> widgets.Widget:
    scope = controls.scope_group()
    button = panel.compute_button()
    view_panel = panel.ViewPanel([panel.row(scope, button)], slot=panel.PlotSlot())

    def selection() -> Hashable:
        return scope.value

    view = panel.OnDemandView(view_panel, lambda key: _figure(f"Scope {key}"), selection, expensive=True, button=button)
    scope.observe(lambda _change: view.refresh(), names="value")
    view.refresh()
    return view_panel.widget


def _tabs() -> widgets.Widget:
    widths = notebook.WidthScope()
    sets = []
    for titles in VIEW_TITLES:
        tab = panel.tabs(titles, [widgets.HTML(f"{title}") for title in titles], width_ch=panel.tab_width_ch(titles))
        widths.register_tabs(tab, titles)
        sets.append(tab)
    box = widgets.VBox(tuple(sets))
    box.add_class(style.ROOT_CLASS)
    return box


def _lazy_panel() -> widgets.Widget:
    selector = selectors.DatasetSelector(INVENTORY, multiple=True, selected=[item.key for item in INVENTORY.items], pool_reason=POOL_REASON)
    export_state: dict[str, object] = {}
    sections = [
        notebook.view_section([("Error over time", lambda: _figure("Error over time"), "error_time")], export_state=export_state),
        notebook.view_section(
            [("Runtime", lambda: _figure("Runtime"), "runtime"), ("Speed-up", lambda: _figure("Speed-up"), "speed_up")], export_state=export_state
        ),
    ]
    lazy = notebook.LazyPanel(
        sections,
        ["1. Errors", "2. Runtime"],
        open_label="Evaluation - Open",
        export_state=export_state,
        controls=[panel.row(selector.header)],
        popovers=[(selector.header, selector.body)],
    )
    return lazy.widget


def _view_panel() -> widgets.Widget:
    status = controls.StatusLine()
    runs = selectors.RunSelector([run for run in RUNS if run.family is not None], multiple=True, selected=[RUNS[0].key])
    navigator = navigation.CaseNavigator(CASE_IDS, labels=CASE_LABELS, status=status)
    rows = [panel.row(controls.channel_dropdown()), panel.row(runs.header), panel.row(navigator.widget, controls.scale_lock_checkbox(value=True))]
    slot = panel.PlotSlot()
    view_panel = panel.ViewPanel(rows, status=status, slot=slot, popovers=[(runs.header, runs.body)])
    figure = _figure("Plot directly below the rows")
    slot.show((figure,))
    plt.close(figure)
    return view_panel.widget


_ENTRIES: Final = (
    CatalogEntry("dropdowns", "Dropdowns on one line, as wide as their longest entry", _dropdowns),
    CatalogEntry("column-grid", "Column grid: every row on the same columns", _column_grid),
    CatalogEntry("channels", "Channel checkboxes", _channels),
    CatalogEntry("case-navigator", "Case navigator over sparse case IDs", _case_navigator),
    CatalogEntry("time-navigator", "Time navigator over irregular stored times", _time_navigator),
    CatalogEntry("transition", "Time navigator of a transition view, hidden for steady", _transition),
    CatalogEntry("count", "Case count", _count),
    CatalogEntry("choice-groups", "One-line choice groups", _choice_groups),
    CatalogEntry("toggles", "Toggle and checkboxes", _toggles),
    CatalogEntry("actions", "Action buttons", _actions),
    CatalogEntry("status", "Status line", _status),
    CatalogEntry("failure", "Failure display", _failure),
    CatalogEntry("runs-multiple", "Runs, several", _runs_multiple),
    CatalogEntry("run-single", "Run, one (with «Missing metadata»)", _run_single),
    CatalogEntry("datasets-multiple", "Datasets, several, pooling switch enabled", _datasets_multiple),
    CatalogEntry("datasets-pool-disabled", "Datasets, pooling switch disabled with its reason", _datasets_pool_disabled),
    CatalogEntry("dataset-single", "Dataset, one (with «Unavailable»)", _dataset_single),
    CatalogEntry("on-demand", "Expensive view with «Compute»", _on_demand),
    CatalogEntry("tabs", "Tabs, one width per notebook", _tabs),
    CatalogEntry("lazy-panel", "Lazy panel with a persistent control row", _lazy_panel, ("datasets-pool-disabled", "tabs", "actions")),
    CatalogEntry("view-panel", "View panel: three rows, popover under row 2", _view_panel, ("runs-multiple", "case-navigator")),
)


def control_entries() -> tuple[CatalogEntry, ...]:
    """Return every control entry in notebook order."""
    return _ENTRIES


def entry(key: str) -> CatalogEntry:
    """Return the entry ``key``; raise ``KeyError`` naming the known keys."""
    for candidate in control_entries():
        if candidate.key == key:
            return candidate
    message = f"Unknown catalog entry {key!r}; known entries: {[candidate.key for candidate in control_entries()]}."
    raise KeyError(message)


def render(catalog_entry: CatalogEntry) -> widgets.VBox:
    """
    Return the stylesheet, the entry's source in a collapsed block, and the built result below it.

    Raises
    ------
    TypeError
        If the entry builds something other than a widget or a Matplotlib figure.

    """
    source = escape(inspect.getsource(catalog_entry.build))
    code = widgets.HTML(f"<details><summary>Code</summary><pre>{source}</pre></details>")
    result = catalog_entry.build()
    if isinstance(result, Figure):
        slot = panel.PlotSlot()
        try:
            slot.show((result,))
        finally:
            plt.close(result)
        shown: widgets.Widget = slot.widget
    elif isinstance(result, widgets.Widget):
        shown = result
    else:
        message = f"Catalog entry {catalog_entry.key!r} built {type(result).__name__}; expected a widget or a Matplotlib figure."
        raise TypeError(message)
    box = widgets.VBox((style.stylesheet_widget(), code, shown))
    box.add_class(style.ROOT_CLASS)
    box.add_class("glp-host")
    return box


def show(key: str) -> None:
    """Display the rendered entry ``key``."""
    display(render(entry(key)))
