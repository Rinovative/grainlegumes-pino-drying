"""
analysis_ui_components.py

Provide reusable widget constructors and UI helpers for analysis notebooks.

Responsibilities:
  - Define checkbox group protocols and constructors
  - Build dropdown, radio and step-control widgets
  - Provide output containers and stable colorbar formatters
  - Compute display-only contour levels, axis labels, and streamline overlays

Design principles:
  - Components are small and reusable
  - Widget state is explicit in returned objects
  - Plotting helpers stay independent of analysis data models

This module does NOT:
  - Compose complete notebook panels or manage figure-export lifecycle
  - Load analysis data or implement domain-specific case rendering
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import ipywidgets as widgets
import matplotlib.ticker as mticker
import numpy as np

from src.analysis.presentation.analysis_field_labels import field_label

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes

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


def ui_dropdown_input_parameter(
    *,
    parameters: list[str],
    default: str | None = None,
) -> widgets.Dropdown:
    """
    Dropdown selector for input parameters (par_*).

    Parameters
    ----------
    parameters : list[str]
        Available input parameters.
    default : str | None, optional
        Default selected parameter. If None, first entry is used.

    Returns
    -------
    widgets.Dropdown
        Configured input-parameter dropdown.

    """
    if not parameters:
        msg = "No input parameters available for dropdown."
        raise ValueError(msg)

    return _build_dropdown(
        options=parameters,
        value=default or parameters[0],
        description="Parameter:",
        width="auto",
    )


# =============================================================================
# SEMANTIC RADIO SELECTORS
# =============================================================================


def ui_radio_error_mode() -> widgets.RadioButtons:
    """
    Radio button selector for error mode (MAE vs. Relative).

    Returns
    -------
    widgets.RadioButtons
        Configured error mode radio buttons.

    """
    return _build_radio(
        options=["MAE", "Relative [%]"],
        value="MAE",
        width="90px",
        margin="0 0 0 12px",
        description="Error mode:",
        description_width="initial",
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


def ui_toggle_map_scale_mode(
    *,
    value: str = "shared",
) -> widgets.ToggleButtons:
    """Build the shared compact map color-scale selector."""
    if value not in {"shared", "individual"}:
        message = "Map scale mode must be 'shared' or 'individual'."
        raise ValueError(message)
    return widgets.ToggleButtons(
        options=(("Shared", "shared"), ("Individual", "individual")),
        value=value,
        description="Color scale:",
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


def ui_checkbox_log_scale(
    *,
    description: str = "log10 for scale parameters",
    default: bool = False,
) -> widgets.Checkbox:
    """
    Checkbox selector for enabling log10 scaling.

    Parameters
    ----------
    description : str, optional
        Checkbox label.
    default : bool, optional
        Default checkbox state.

    Returns
    -------
    widgets.Checkbox
        Configured log-scale checkbox.

    """
    return widgets.Checkbox(
        value=default,
        description=description,
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


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

# COLORBAR FORMATTERS


def choose_colorbar_formatter(
    vmin: float,
    vmax: float,
    *,
    ticks: np.ndarray | None = None,
) -> mticker.Formatter:
    """
    Show consistent tick labels with trailing zeros kept.

    - If scientific notation is used (very small/large magnitude), use "%.2e".
    - Otherwise, choose a fixed number of decimals.
      If `ticks` is provided, decimals are derived from the smallest tick step so that
      labels like 0.20 are shown when 0.18 exists (same decimals across all ticks).

    Parameters
    ----------
    vmin : float
        Minimum colorbar value.
    vmax : float
        Maximum colorbar value.
    ticks : np.ndarray | None
        Optional tick values to derive a stable decimal count.

    Returns
    -------
    matplotlib.ticker.Formatter
        Formatter instance.

    """
    sig = 3
    sci_low, sci_high = -3, 3  # scientific if exponent <= -3 or >= 3

    vr = max(abs(vmin), abs(vmax))

    # Handle all-zero (or invalid) ranges
    if vr == 0 or not math.isfinite(vr):
        return mticker.FormatStrFormatter("%.2f")

    exp = math.floor(math.log10(vr))

    # Scientific notation
    if exp <= sci_low or exp >= sci_high:
        return mticker.FormatStrFormatter(f"%.{sig - 1}e")

    # Fixed decimals
    decimals_default = max(0, (sig - 1) - exp)
    decimals = decimals_default

    if ticks is not None:
        t = np.asarray(ticks, dtype=float)
        t = t[np.isfinite(t)]
        t = np.unique(np.sort(t))

        if t.size >= 2:  # noqa: PLR2004
            diffs = np.diff(t)
            diffs = diffs[diffs > 0]

            if diffs.size > 0:
                step = float(np.min(diffs))
                # decimals so that step is representable (keep trailing zeros)
                # e.g. step=0.05 -> 2 decimals, step=0.1 -> 1 decimal
                decimals_from_step = int(max(0, math.ceil(-math.log10(step) - 1e-12)))
                decimals = max(decimals_default, decimals_from_step)

    def _fmt(x: float, _pos: int | None = None) -> str:
        s = f"{x:.{decimals}f}"

        # Avoid "-0.00"
        if float(s) == 0.0:
            s = s.lstrip("-")

        return s

    return mticker.FuncFormatter(_fmt)


# CONTOUR LEVELS


_MIN_LEVEL_COUNT = 2


def compute_levels(arr: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute contour levels for given data array.

    Parameters
    ----------
    arr : np.ndarray
        Input data array.
    n : int, optional
        Desired number of levels (default: 10).

    Returns
    -------
    np.ndarray
        Computed contour levels.

    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Use 1st and 99th percentiles to avoid outliers
    q_lo, q_hi = np.quantile(arr, [0.01, 0.99])
    vmin, vmax = float(q_lo), float(q_hi)

    if vmin == vmax:
        range_padding = 1e-12
        return np.linspace(vmin - range_padding, vmax + range_padding, n)

    # Use MaxNLocator to get "nice" levels
    locator = mticker.MaxNLocator(nbins=n)

    levels = np.asarray(locator.tick_values(vmin, vmax), dtype=np.float64)
    levels = np.unique(levels).astype(np.float64, copy=False)

    if len(levels) < _MIN_LEVEL_COUNT or not np.all(np.diff(levels) > 0):
        levels = np.linspace(vmin, vmax, n, dtype=np.float64)

    return levels


# AXIS LABEL HELPERS


def apply_axis_labels(
    ax: Axes,
    col: int,
    Lx: float,
    Ly: float,
    *,
    is_last_row: bool,
) -> None:
    """
    Apply consistent axis labels and ticks to subplot axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to modify.
    col : int
        Column index of the subplot.
    Lx : float
        Length of the domain in x-direction.
    Ly : float
        Length of the domain in y-direction.
    is_last_row : bool
        Whether the subplot is in the last row.

    Returns
    -------
    None
        Modifies the Axes in place.

    """
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    yticks = [0.0, 0.25, 0.5, 0.75]
    ax.set_yticks(yticks)

    if col == 0:
        ax.set_ylabel("y [m]")
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    if is_last_row:
        ax.set_xlabel("x [m]")
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


# FLOW OVERLAYS


def overlay_streamlines(ax: Axes, X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """
    Overlay streamlines on the given Axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to modify.
    X : np.ndarray
        X-coordinates meshgrid.
    Y : np.ndarray
        Y-coordinates meshgrid.
    u : np.ndarray
        Velocity component in x-direction.
    v : np.ndarray
        Velocity component in y-direction.

    Returns
    -------
    None
        Modifies the Axes in place.

    """
    ax.streamplot(
        X,
        Y,
        u,
        v,
        color=(0, 0, 0, 0.6),
        density=1.0,
        linewidth=0.6,
        arrowsize=0.6,
        minlength=0.1,
        integration_direction="both",
    )
