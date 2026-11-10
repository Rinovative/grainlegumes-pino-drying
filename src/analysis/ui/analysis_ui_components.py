"""Provide reusable widget constructors and UI helpers for analysis notebooks.

Responsibilities:
  - Define checkbox group protocols and constructors
  - Build dropdown, radio and step-control widgets
  - Arrange controls in one uniformly spaced wrapping row
  - Provide output containers

Design principles:
  - Components are small and reusable
  - Widget state is explicit in returned objects

This module does NOT:
  - Compose complete notebook panels or manage figure-export lifecycle
  - Load analysis data or implement domain-specific case rendering
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import ipywidgets as widgets

from src.analysis.presentation.analysis_field_labels import field_label

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

COMPACT_CONTROL_WIDTH_PX = 230
COMPACT_CASE_STEP_WIDTH_PX = 40
COMPACT_CASE_VALUE_WIDTH_PX = COMPACT_CONTROL_WIDTH_PX - 2 * COMPACT_CASE_STEP_WIDTH_PX

_COMPACT_CONTROL_WIDTH = f"{COMPACT_CONTROL_WIDTH_PX}px"
_CONTROL_ROW_MARGIN = "0 0 6px 0"
_CONTROL_ROW_CHILD_MARGIN = "0 12px 6px 0"
_WIDGET_DEFAULT_MARGIN_PX = 2.0
_SCOPE_BUTTON_INCREMENT_POINTS = 1.0
_CSS_PIXELS_PER_POINT = 96.0 / 72.0
_SCOPE_BUTTON_WIDTH_PX = (COMPACT_CONTROL_WIDTH_PX - 4.0 * _WIDGET_DEFAULT_MARGIN_PX) / 2.0 + _SCOPE_BUTTON_INCREMENT_POINTS * _CSS_PIXELS_PER_POINT
_SCOPE_BUTTON_WIDTH = f"{_SCOPE_BUTTON_WIDTH_PX:.12g}px"
_SCOPE_BUTTON_GROUP_WIDTH_PX = 2.0 * _SCOPE_BUTTON_WIDTH_PX + 4.0 * _WIDGET_DEFAULT_MARGIN_PX
_SCOPE_BUTTON_GROUP_WIDTH = f"{_SCOPE_BUTTON_GROUP_WIDTH_PX:.12g}px"

# =============================================================================
# TYPE CONTRACTS
# =============================================================================


class CheckboxGroup(Protocol):
    """
    Define the structural widget contract for labelled checkbox groups.

    Attributes
    ----------
    boxes : dict[str, ipywidgets.Checkbox]
        Public mapping from semantic option label to its checkbox widget.

    Notes
    -----
    The concrete object is a private ``VBox`` subclass that owns this mapping.
    Callers depend only on this protocol and must not infer that implementation.

    """

    boxes: dict[str, widgets.Checkbox]


class _CheckboxGroupVBox(widgets.VBox):
    """Own typed checkbox state while preserving the public ``VBox`` contract."""

    boxes: dict[str, widgets.Checkbox]


class ChannelCheckboxState:
    """Preserve compatible checkbox channels through one external state owner."""

    def __init__(
        self,
        *,
        title: str,
        callback: Callable[[], None],
        selection_getter: Callable[[], Sequence[str] | None],
        selection_setter: Callable[[tuple[str, ...]], None],
    ) -> None:
        """Initialize an empty three-column channel selector."""
        self._title = title
        self._callback = callback
        self._selection_getter = selection_getter
        self._selection_setter = selection_setter
        self._known: tuple[str, ...] = ()
        self._group: _CheckboxGroupVBox | None = None
        self.status = widgets.HTML(layout=widgets.Layout(display="none"))
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

    @property
    def group(self) -> _CheckboxGroupVBox | None:
        """Return the current concrete checkbox group for widget composition."""
        return self._group

    def _changed(self) -> None:
        """Retain one accepted selection before rendering."""
        self._selection_setter(self.selected)
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
        retained_value = self._selection_getter()
        if retained_value is None:
            defaults = resolved
            removed: tuple[str, ...] = ()
        else:
            retained = tuple(retained_value)
            allowed = set(resolved)
            previously_known = set(self._known)
            defaults = tuple(name for name in resolved if name in retained or name not in previously_known)
            removed = tuple(name for name in retained if name not in allowed)
        group = ui_checkbox_channels(
            channels=resolved,
            default_on=defaults,
            labels=labels,
            fixed_columns=3,
        )
        for checkbox in group.boxes.values():
            checkbox.observe(
                lambda _change: self._changed(),
                names="value",
            )
        self._group = group
        self._known = resolved
        self._selection_setter(defaults)
        self.status.value = "<p>Unavailable channels removed: " + ", ".join(removed) + ".</p>" if removed else ""
        self.status.layout.display = "flex" if removed else "none"
        self.container.children = (
            widgets.HTML(f"<b>{self._title}</b>"),
            group,
            self.status,
        )
        self.container.layout.display = "flex"


# =============================================================================
# GENERIC BUILDING BLOCKS (internal use only)
# =============================================================================


def _build_dropdown(
    *,
    options: list[str],
    value: str,
    description: str,
    width: str,
) -> widgets.Dropdown:
    """
    Create internal generic dropdown builder.

    Parameters
    ----------
    options : list[str]
        Dropdown options.
    value : str
        Default selected value.
    description : str
        Dropdown label.
    width : str
        CSS width of the dropdown.

    Returns
    -------
    widgets.Dropdown
        Configured dropdown widget.

    """
    return widgets.Dropdown(
        options=options,
        value=value,
        description=description,
        layout=widgets.Layout(width=width),
    )


def _build_radio(
    *,
    options: list[str],
    value: str,
    width: str,
    margin: str | None = None,
    description: str | None = None,
    description_width: str = "initial",
) -> widgets.RadioButtons:
    """
    Create internal generic radio-button builder.

    Parameters
    ----------
    options : list[str]
        Radio button options.
    value : str
        Default selected value.
    width : str
        CSS width of the radio button group.
    margin : str | None, optional
        CSS margin around the radio button group, by default None.
    description : str | None, optional
        Optional label shown left of the radio group.
    description_width : str, optional
        CSS-like width for the description area, default "initial".

    Returns
    -------
    widgets.RadioButtons
        Configured radio button widget.

    """
    layout_kwargs: dict[str, str] = {"width": width}
    if margin is not None:
        layout_kwargs["margin"] = margin

    return widgets.RadioButtons(
        options=options,
        value=value,
        description=description or "",
        layout=widgets.Layout(**layout_kwargs),
        style={"description_width": description_width},
    )


def _build_int_step_control(
    *,
    value: int,
    minimum: int,
    maximum: int,
    step: int,
    description: str,
    width: str,
    prev_label: str,
    next_label: str,
) -> tuple[widgets.BoundedIntText | widgets.IntSlider, widgets.Button, widgets.Button]:
    """
    Create internal generic integer step control builder.

    Parameters
    ----------
    value : int
        Initial value.
    minimum : int
        Minimum value.
    maximum : int
        Maximum value.
    step : int
        Step size.
    description : str
        Control label.
    width : str
        CSS width of the control.
    prev_label : str
        Label for the "previous" button.
    next_label : str
        Label for the "next" button.

    Returns
    -------
    tuple[widgets.BoundedIntText | widgets.IntSlider, widgets.Button, widgets.Button]
        Control widget, previous button, next button.

    """
    if step == 1:
        # discrete index → text input
        control: widgets.BoundedIntText | widgets.IntSlider = widgets.BoundedIntText(
            value=value,
            min=minimum,
            max=maximum,
            description=description,
            layout=widgets.Layout(width=width),
        )
    else:
        # aggregation / count → slider
        control = widgets.IntSlider(
            value=value,
            min=minimum,
            max=maximum,
            step=step,
            description=description,
            continuous_update=False,
            readout=True,
        )

    prev_btn = widgets.Button(description=prev_label)
    next_btn = widgets.Button(description=next_label)

    return control, prev_btn, next_btn


def _build_checkbox_group(
    *,
    options: list[str],
    defaults: list[str],
    description: str | None = None,
    minimum_column_width: str = "150px",
    tooltips: Mapping[str, str] | None = None,
    natural_width: bool = False,
    one_per_row: bool = False,
    fixed_columns: int | None = None,
    fixed_item_width: str | None = None,
) -> _CheckboxGroupVBox:
    """Create one compact responsive checkbox group with stable option order."""
    tooltip_map = {} if tooltips is None else dict(tooltips)
    unknown_tooltips = set(tooltip_map).difference(options)
    if unknown_tooltips:
        message = f"Checkbox tooltips contain unknown options: {sorted(unknown_tooltips)}."
        raise ValueError(message)
    if (fixed_columns is None) != (fixed_item_width is None):
        message = "Fixed checkbox columns require one shared item width."
        raise ValueError(message)
    if fixed_columns is not None and fixed_columns < 1:
        message = "Fixed checkbox column count must be positive."
        raise ValueError(message)
    if fixed_columns is not None and (natural_width or one_per_row):
        message = "Fixed checkbox columns cannot use another flow layout."
        raise ValueError(message)
    if fixed_columns is not None:
        item_layout = widgets.Layout(
            margin="0",
            width=fixed_item_width,
            min_width=fixed_item_width,
            max_width=fixed_item_width,
            flex="0 0 auto",
        )
    elif one_per_row:
        item_layout = widgets.Layout(
            margin="0",
            width="100%",
            max_width="100%",
            flex="0 0 auto",
        )
    elif natural_width:
        item_layout = widgets.Layout(
            margin="0",
            width="auto",
            max_width="32%",
            flex="0 1 auto",
        )
    else:
        item_layout = widgets.Layout(
            margin="0",
            width="auto",
        )
    boxes = {
        option: widgets.Checkbox(
            value=option in defaults,
            description=option,
            tooltip=tooltip_map.get(option, ""),
            indent=False,
            layout=item_layout,
            style={"description_width": "auto"},
        )
        for option in options
    }
    if fixed_columns is not None:
        option_container = widgets.GridBox(
            children=tuple(boxes.values()),
            layout=widgets.Layout(
                align_items="flex-start",
                grid_template_columns=f"repeat({fixed_columns}, {fixed_item_width})",
                grid_gap="1px 2px",
                width="max-content",
            ),
        )
    elif one_per_row:
        option_container = widgets.VBox(
            children=tuple(boxes.values()),
            layout=widgets.Layout(
                align_items="flex-start",
                grid_gap="2px",
                width="100%",
            ),
        )
    elif natural_width:
        option_container = widgets.Box(
            children=tuple(boxes.values()),
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                justify_content="flex-start",
                align_items="flex-start",
                grid_gap="4px 6px",
                width="100%",
            ),
        )
    else:
        option_container = widgets.GridBox(
            children=tuple(boxes.values()),
            layout=widgets.Layout(
                width="100%",
                grid_template_columns=f"repeat(auto-fit, minmax({minimum_column_width}, 1fr))",
                grid_gap="2px 10px",
            ),
        )
    children: list[widgets.Widget] = []
    if description is not None:
        children.append(widgets.Label(description))
    children.append(option_container)
    box = _CheckboxGroupVBox(
        children,
        layout=widgets.Layout(
            width="max-content" if fixed_columns is not None else "100%",
            margin="0",
            align_items="flex-start",
        ),
    )
    box.boxes = boxes
    return box


# =============================================================================
# SEMANTIC NAVIGATION COMPONENTS
# =============================================================================


def ui_step_case_index(
    *,
    n_cases: int | None = None,
    start_idx: int = 0,
    case_numbers: Sequence[int] | None = None,
) -> tuple[widgets.BoundedIntText | widgets.IntText, widgets.Button, widgets.Button]:
    """Build the maintained numeric case navigator for contiguous indices or sparse IDs."""
    if case_numbers is not None:
        if n_cases is not None:
            msg = "Case navigation accepts n_cases or case_numbers, not both."
            raise ValueError(msg)
        numbers = tuple(case_numbers)
        if not numbers or any(isinstance(number, bool) or not isinstance(number, int) for number in numbers):
            msg = "Case-number navigation requires at least one integer case number."
            raise ValueError(msg)
        if len(numbers) != len(set(numbers)):
            msg = "Case-number navigation cannot contain duplicate values."
            raise ValueError(msg)
        if start_idx < 0 or start_idx >= len(numbers):
            msg = f"start_idx must select an available case number, got {start_idx}."
            raise ValueError(msg)
        sparse_control = widgets.IntText(
            value=numbers[start_idx],
            description="Case:",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="120px"),
        )
        previous = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
        following = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
        return sparse_control, previous, following

    if n_cases is None:
        msg = "Contiguous case navigation requires n_cases."
        raise ValueError(msg)
    control, previous, following = _build_int_step_control(
        value=start_idx + 1,
        minimum=1,
        maximum=n_cases,
        step=1,
        description="Case:",
        width="140px",
        prev_label="←",
        next_label="→",
    )
    previous.layout = widgets.Layout(width="40px")
    following.layout = widgets.Layout(width="40px")
    if not isinstance(control, widgets.BoundedIntText):
        msg = "Unit-step case navigation must construct a bounded integer text control."
        raise TypeError(msg)
    return control, previous, following


def ui_compact_case_row(
    case: widgets.DOMWidget,
    previous: widgets.Button,
    following: widgets.Button,
) -> widgets.HBox:
    """Apply the maintained EDA case navigator dimensions and row behavior."""
    case_width = f"{COMPACT_CASE_VALUE_WIDTH_PX}px"
    step_width = f"{COMPACT_CASE_STEP_WIDTH_PX}px"
    case.layout = widgets.Layout(
        width=case_width,
        min_width=case_width,
        max_width=case_width,
        flex=f"0 0 {case_width}",
        margin="0",
    )
    previous.tooltip = "Previous shared case"
    following.tooltip = "Next shared case"
    for button in (previous, following):
        button.layout = widgets.Layout(
            width=step_width,
            min_width=step_width,
            max_width=step_width,
            flex=f"0 0 {step_width}",
            margin="0",
        )
    return widgets.HBox(
        (case, previous, following),
        layout=widgets.Layout(
            align_items="center",
            flex_flow="row nowrap",
            grid_gap="0",
            width=_COMPACT_CONTROL_WIDTH,
            min_width=_COMPACT_CONTROL_WIDTH,
            max_width=_COMPACT_CONTROL_WIDTH,
            flex=f"0 0 {_COMPACT_CONTROL_WIDTH}",
        ),
    )


def ui_control_row(*children: widgets.Widget) -> widgets.HBox:
    """
    Arrange controls in one wrapping row with the shared notebook spacing.

    ipywidgets has no ``Layout.gap``, so every child receives the margin
    ``0 12px 6px 0``; the row itself adds ``0 0 6px 0`` below. Child order and
    child widths are preserved.
    """
    for child in children:
        child.layout.margin = _CONTROL_ROW_CHILD_MARGIN
    return widgets.HBox(
        children,
        layout=widgets.Layout(
            flex_flow="row wrap",
            align_items="center",
            width="100%",
            margin=_CONTROL_ROW_MARGIN,
        ),
    )


def ui_scope_toggle(*, value: str = "aggregate") -> widgets.ToggleButtons:
    """Build the maintained EDA Aggregate and Single case control."""
    if value not in {"aggregate", "single"}:
        message = "Scope must be 'aggregate' or 'single'."
        raise ValueError(message)
    return widgets.ToggleButtons(
        options=(("Aggregate", "aggregate"), ("Single case", "single")),
        value=value,
    )


def ui_compact_scope_controls(
    scope: widgets.ToggleButtons | None,
) -> tuple[widgets.HBox, widgets.HBox]:
    """Build the maintained bounded scope row and replaceable detail column."""
    detail = widgets.HBox(
        layout=widgets.Layout(
            align_items="center",
            min_width="0",
        )
    )
    if scope is None:
        detail.layout = widgets.Layout(
            align_items="center",
            width=_COMPACT_CONTROL_WIDTH,
            min_width=_COMPACT_CONTROL_WIDTH,
            max_width=_COMPACT_CONTROL_WIDTH,
            flex=f"0 0 {_COMPACT_CONTROL_WIDTH}",
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
            min_width=(_COMPACT_CONTROL_WIDTH if scope is None else _SCOPE_BUTTON_GROUP_WIDTH),
        ),
    )
    return row, detail


def ui_set_scope_detail(
    detail: widgets.HBox,
    controls: Sequence[widgets.Widget],
) -> None:
    """Install exactly one active case or aggregate-count control group."""
    resolved = tuple(controls)
    if not resolved:
        message = "Scope detail requires one active case or count control."
        raise ValueError(message)
    detail.children = resolved


def ui_step_case_count(
    *,
    start_cases: int,
    min_cases: int,
    max_cases: int,
    step_size: int,
) -> tuple[widgets.IntSlider, widgets.Button, widgets.Button]:
    """
    Step control for selecting number of cases to display.

    Parameters
    ----------
    start_cases : int
        Initial number of cases.
    min_cases : int
        Minimum number of cases.
    max_cases : int
        Maximum number of cases.
    step_size : int
        Step size for increasing/decreasing case count.

    Returns
    -------
    tuple[widgets.IntSlider, widgets.Button, widgets.Button]
        Control widget, previous button, next button.

    """
    control, prev_btn, next_btn = _build_int_step_control(
        value=start_cases,
        minimum=min_cases,
        maximum=max_cases,
        step=step_size,
        description="Cases:",
        width="auto",
        prev_label="⟨",
        next_label="⟩",
    )
    if not isinstance(control, widgets.IntSlider):
        msg = "Multi-step case-count navigation must construct an integer slider."
        raise TypeError(msg)
    return control, prev_btn, next_btn


# =============================================================================
# SEMANTIC DROPDOWN SELECTORS
# =============================================================================


def ui_dropdown_dataset(names: list[str]) -> widgets.Dropdown:
    """
    Dropdown selector for dataset names.

    Parameters
    ----------
    names : list[str]
        Available dataset names.

    Returns
    -------
    widgets.Dropdown
        Configured dataset dropdown.

    """
    return _build_dropdown(
        options=names,
        value=names[0],
        description="Select:",
        width="240px",
    )


def ui_dropdown_channel(
    *,
    channels: Sequence[str] | None = None,
    default: str = "|u|",
) -> widgets.Dropdown:
    """Build the compact output-channel dropdown."""
    resolved = list(channels or ("p", "u", "v", "|u|"))
    if not resolved or default not in resolved:
        msg = "Channel dropdown requires a non-empty option list containing its default."
        raise ValueError(msg)
    return widgets.Dropdown(
        options=[(field_label(channel), channel) for channel in resolved],
        value=default,
        description="Channel:",
        layout=widgets.Layout(width="auto"),
    )


# =============================================================================
# SEMANTIC RADIO SELECTORS
# =============================================================================


def ui_radio_error_mode() -> widgets.RadioButtons:
    """
    Radio button selector for the existing field-error location.

    Returns
    -------
    widgets.RadioButtons
        Configured error mode radio buttons.

    """
    return _build_radio(
        options=["MAE", "Relative [%]", "Normalized absolute error"],
        value="Normalized absolute error",
        width="90px",
        margin="0 0 0 12px",
        description="Error mode:",
        description_width="initial",
    )


def ui_radio_transient_error_mode() -> widgets.RadioButtons:
    """Select the one error map in the transient reference/prediction viewer."""
    return _build_radio(
        options=["Signed error", "Absolute error", "Normalized absolute error"],
        value="Normalized absolute error",
        width="auto",
        margin="0 0 0 12px",
        description="Error mode:",
        description_width="initial",
    )


def ui_dropdown_display_spatial_stride() -> widgets.Dropdown:
    """Choose presentation-only map sampling while retaining the full grid."""
    return widgets.Dropdown(
        options=(("Auto", "auto"), ("Full", 1), ("Stride 2", 2), ("Stride 4", 4)),
        value="auto",
        description="Display stride:",
        style={"description_width": "initial"},
    )


def ui_radio_pred_scale_mode() -> widgets.RadioButtons:
    """
    Radio button selector for prediction/GT scaling.

    Options
    -------
    - "Independent" : pred and GT get their own scales (current behaviour)
    - "Shared (GT)" : prediction uses the GT scale, with outliers outside that range masked

    Returns
    -------
    widgets.RadioButtons
        Configured scale-mode radio buttons.

    """
    return _build_radio(
        options=["Independent", "Shared (GT)"],
        value="Independent",
        width="auto",
        margin="0 0 0 12px",
        description="Pred/GT scale:",
        description_width="initial",
    )


def ui_checkbox_map_scale_lock(*, value: bool = False) -> widgets.Checkbox:
    """Build the compact boolean map-normalization lock used by EDA."""
    if not isinstance(value, bool):
        message = "Map color-scale lock state must be boolean."
        raise TypeError(message)
    return widgets.Checkbox(
        value=value,
        description="Lock color scale",
        indent=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )


def ui_radio_kappa_scale() -> widgets.RadioButtons:
    """
    Radio button selector for permeability scaling.

    Options
    -------
    - "kappa"       : physical permeability [m²]
    - "log10(kappa)": logarithmic permeability

    Returns
    -------
    widgets.RadioButtons
        Configured kappa scaling radio buttons.

    """
    return _build_radio(
        options=["kappa", "log10(kappa)"],
        value="log10(kappa)",
        width="100px",
        margin="0 0 0 12px",
        description="Kappa scale:",
        description_width="initial",
    )


# =============================================================================
# SEMANTIC CHECKBOX SELECTORS
# =============================================================================


def ui_checkbox_channels(
    *,
    channels: Sequence[str] | None = None,
    default_on: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    natural_width: bool = False,
    one_per_row: bool = False,
    fixed_columns: int | None = None,
) -> _CheckboxGroupVBox:
    """Build one compact channel checkbox group in caller-selected flow."""
    resolved = list(channels or ("p", "u", "v", "|u|"))
    defaults = list(resolved if default_on is None else default_on)
    if not resolved or not set(defaults).issubset(resolved):
        msg = "Channel checkbox defaults must be drawn from a non-empty option list."
        raise ValueError(msg)
    visible_labels = {} if labels is None else dict(labels)
    if not set(visible_labels).issubset(resolved):
        msg = "Channel display labels must be keyed by available channels."
        raise ValueError(msg)
    if any(not isinstance(label, str) or not label or "\n" in label for label in visible_labels.values()):
        msg = "Channel display labels must be non-empty single-line text."
        raise ValueError(msg)
    resolved_labels = {channel: visible_labels.get(channel, field_label(channel)) for channel in resolved}
    fixed_item_width = None
    if fixed_columns is not None:
        widest_label = max(len(label) for label in resolved_labels.values())
        fixed_item_width = f"{widest_label + 4}ch"
    group = _build_checkbox_group(
        options=resolved,
        defaults=defaults,
        minimum_column_width="110px",
        natural_width=natural_width,
        one_per_row=one_per_row,
        fixed_columns=fixed_columns,
        fixed_item_width=fixed_item_width,
    )
    for channel, checkbox in group.boxes.items():
        checkbox.description = resolved_labels[channel]
    return group


def ui_checkbox_datasets(
    *,
    dataset_names: list[str],
    default_on: list[str] | None = None,
    tooltips: Mapping[str, str] | None = None,
    natural_width: bool = False,
) -> widgets.VBox:
    """
    Checkbox selector for datasets.

    Parameters
    ----------
    dataset_names : list[str]
        Available dataset names.
    default_on : list[str] | None, optional
        Datasets enabled by default. Defaults to all datasets.
    tooltips : Mapping[str, str] | None, optional
        Canonical identity details keyed by visible dataset label.
    natural_width : bool, optional
        Use the left-aligned responsive wrapping selector layout.

    Returns
    -------
    widgets.VBox
        Checkbox group for dataset selection.

    Notes
    -----
    The returned widget exposes a public `boxes` attribute
    mapping dataset name -> Checkbox widget.

    """
    default_on = dataset_names if default_on is None else default_on

    return _build_checkbox_group(
        options=dataset_names,
        defaults=default_on,
        minimum_column_width="190px",
        tooltips=tooltips,
        natural_width=natural_width,
    )


def ui_checkbox_normalise(
    *,
    description: str = "Normalise",
    default: bool = True,
    width: str = "160px",
) -> widgets.Checkbox:
    """
    Checkbox selector for normalisation toggles.

    Parameters
    ----------
    description : str, optional
        Checkbox label.
    default : bool, optional
        Default checkbox state.
    width : str, optional
        CSS width.

    Returns
    -------
    widgets.Checkbox
        Configured checkbox.

    """
    return widgets.Checkbox(
        value=default,
        description=description,
        indent=False,
        layout=widgets.Layout(width=width),
    )


# =============================================================================
# OUTPUT CONTAINER
# =============================================================================


def ui_output_plot() -> widgets.Output:
    """
    Output container for plots.

    Returns
    -------
    widgets.Output
        Configured output widget.

    """
    return widgets.Output()
