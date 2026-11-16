"""Compare bounded dataset spectra without loading model activations or artifacts.

Responsibilities:
  - Compute Hann-windowed isotropic and x/y directional power spectra
  - Show cumulative bandwidth with casewise uncertainty over ordered prefixes
  - Resolve both complementary position-dependent spectral orientations
  - Enforce per-field units, stored representations, and Cartesian grids

Design principles:
  - Frequencies use coordinate-derived units of inverse metres
  - Flow-position-resolved power is normalized within each row before case aggregation
  - Dataset comparisons never combine incompatible stored field representations

This module does NOT:
  - Load model-training datasets or infer undeclared task fields
  - Compare model predictions with references or inspect model activations
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib.ticker as mticker
import numpy as np

from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.presentation.analysis_visual_semantics import SeriesStyle

_DEFAULT_CASE_LIMIT = 100
_PANEL_IN = (4.6, 3.2)  # one spectral panel
_POWER_LINEWIDTH = 1.5
_CUMULATIVE_LINEWIDTH = 0.9
_COLORBAR_LABEL = f"Row-normalized power fraction [{field_labels.display_unit('1')}]"
_HEADER_WRAP_WIDTH = 32  # characters per line of a dataset name above its column
_HEADER_MAX_LINES = 3
_MIN_GRID_SIZE = 2
SpectralEvolutionOrientation = Literal["cross_stream_along_flow", "flow_across_cross_stream"]
_POWER_FRACTION_FLOOR = 1e-12
_CROSS_STREAM_ALONG_FLOW: Final[SpectralEvolutionOrientation] = "cross_stream_along_flow"
_FLOW_ACROSS_CROSS_STREAM: Final[SpectralEvolutionOrientation] = "flow_across_cross_stream"
_CASE_ID_PATTERN = re.compile(r"case_([0-9]{4,})")
_DIMENSIONLESS_REPRESENTATION_LABELS = {
    "dimensionless_log10_ratio_to_1_m2": "log10(k / 1 m²)",
    "dimensionless_cross_component_ratio_to_geometric_mean": "kij / sqrt(kii kjj)",
}
_PHYSICAL_REPRESENTATION_LABELS = {
    "absolute_physical_state": "physical values",
    "identity": "physical values",
    "identity_before_train_normalization": "physical values",
    "derived_speed_magnitude": "derived magnitude |u|",
}
_PHYSICAL_VALUE_REPRESENTATIONS = frozenset(
    {
        "absolute_physical_state",
        "identity",
        "identity_before_train_normalization",
        "derived_speed_magnitude",
    }
)


def available_case_numbers(frame: pd.DataFrame) -> tuple[int, ...]:
    """Return authoritative dataset-local case numbers in DataFrame order."""
    if frame.empty:
        msg = "EDA case navigation requires a non-empty frame."
        raise ValueError(msg)
    numbers: list[int] = []
    for sample_id in frame.index:
        match = _CASE_ID_PATTERN.fullmatch(sample_id) if isinstance(sample_id, str) else None
        if match is None:
            msg = f"EDA case navigation requires canonical case IDs, got {sample_id!r}."
            raise ValueError(msg)
        numbers.append(int(match.group(1)))
    if len(numbers) != len(set(numbers)):
        msg = "EDA case navigation requires unique numeric case IDs."
        raise ValueError(msg)
    return tuple(numbers)


def _case_row(frame: pd.DataFrame, case_number: int) -> pd.Series:
    """Resolve one exact dataset-local case number without positional clamping."""
    if isinstance(case_number, bool) or not isinstance(case_number, int):
        msg = "case_number must be an integer."
        raise TypeError(msg)
    numbers = available_case_numbers(frame)
    for sample_id, number in zip(frame.index, numbers, strict=True):
        if number == case_number:
            row = frame.loc[sample_id]
            if not hasattr(row, "index"):
                msg = f"Case ID {sample_id!r} did not resolve to one row."
                raise RuntimeError(msg)
            return row
    msg = f"Requested case {case_number} is unavailable in this dataset. Choose a shared case number."
    raise ValueError(msg)


def _is_transient(frame: pd.DataFrame) -> bool:
    """Return whether one admitted frame uses nested transient storage."""
    return capabilities.is_transient_frame(frame)


def _final_valid_state_index(row: pd.Series) -> int:
    """Resolve the final valid stored transient state without time interpolation."""
    time = row.get("time")
    if not isinstance(time, dict):
        msg = "Transient EDA spectra require nested time evidence."
        raise TypeError(msg)
    mask = np.asarray(time.get("valid_state_mask"))
    physical_time = np.asarray(time.get("regular_state_hours"), dtype=float)
    if mask.dtype != np.dtype(bool) or mask.ndim != 1 or mask.shape != physical_time.shape or not np.isfinite(physical_time).all():
        msg = "Transient EDA spectra require aligned finite stored time and valid-state evidence."
        raise ValueError(msg)
    indices = np.flatnonzero(mask)
    if not indices.size:
        msg = "Transient EDA spectra require at least one valid stored state."
        raise ValueError(msg)
    return int(indices[-1])


def _field_values(
    frame: pd.DataFrame,
    row: pd.Series,
    field: str,
) -> np.ndarray:
    """Return one direct or final-valid-state 2D field on its display scale, so spectra share the row label's unit."""
    state_index = _final_valid_state_index(row) if _is_transient(frame) and field in row["state_trajectories"] else None
    values = capabilities.field_values(
        frame,
        row,
        field,
        transient_state_index=state_index,
    )
    if values.ndim != _MIN_GRID_SIZE:
        msg = f"Spectral channel {field!r} does not resolve to one stored 2D field."
        raise ValueError(msg)
    return capabilities.field_display_values(frame, row, field, values)


def _coordinate_values(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return direct or nested retained Cartesian coordinates for one case."""
    if "state_trajectories" not in row:
        return np.asarray(row["x"], dtype=float), np.asarray(row["y"], dtype=float)
    static = row["static_fields"]
    if not isinstance(static, dict) or "x" not in static or "y" not in static:
        msg = "Transient EDA spectra require retained nested x and y coordinates."
        raise ValueError(msg)
    return np.asarray(static["x"], dtype=float), np.asarray(static["y"], dtype=float)


def _select_datasets(
    datasets: dict[str, pd.DataFrame],
    dataset_names: Sequence[str] | None,
) -> dict[str, pd.DataFrame]:
    """Select labelled frames in authoritative input order."""
    if not datasets:
        msg = "At least one EDA dataset is required."
        raise ValueError(msg)
    available = tuple(datasets)
    requested = available if dataset_names is None else tuple(dataset_names)
    if not requested:
        msg = "Select at least one dataset."
        raise ValueError(msg)
    if len(requested) != len(set(requested)):
        msg = "EDA dataset selection cannot contain duplicates."
        raise ValueError(msg)
    unknown = tuple(name for name in requested if name not in datasets)
    if unknown:
        msg = f"Unknown EDA dataset selection: {unknown!r}."
        raise ValueError(msg)
    requested_set = set(requested)
    return {name: datasets[name] for name in available if name in requested_set}


def _case_count_text(datasets: dict[str, pd.DataFrame], *, max_cases: int) -> str:
    """Report only actual displayed case counts; selection mechanics stay internal."""
    counts = {label: min(max_cases, len(frame)) for label, frame in datasets.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) == 1:
        return f"{next(iter(unique_counts))} cases"
    return ", ".join(f"{label}: {count} cases" for label, count in counts.items())


def _stored_state_scope_text(datasets: dict[str, pd.DataFrame]) -> str:
    """Disclose final-valid-state extraction for nested transient spectral views."""
    return " — final valid stored state" if any(_is_transient(frame) for frame in datasets.values()) else ""


def _validate_datasets(
    datasets: dict[str, pd.DataFrame],
    *,
    max_cases: int,
    channels: Sequence[str] | None,
) -> capabilities.ResolvedFieldSelection:
    """Admit selected frames and validate comparisons per compatible field subset."""
    if not datasets:
        msg = "At least one EDA dataset is required."
        raise ValueError(msg)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    resolution = capabilities.resolve_fields(
        datasets,
        view="spectral",
        requested=channels,
    )
    for label, frame in datasets.items():
        if not label or frame.empty:
            msg = "EDA datasets require non-empty labels and frames."
            raise ValueError(msg)
        task = frame.attrs.get("task_id")
        contract = frame.attrs.get("task_contract_digest")
        if not isinstance(task, str) or not isinstance(contract, str):
            msg = "EDA spectra require internal task and task-contract metadata."
            raise TypeError(msg)
    for field in resolution.fields:
        compatible = capabilities.compatible_frames(datasets, resolution, field)
        units = {capabilities.field_unit(frame, field) for frame in compatible.values()}
        representations = {capabilities.field_representation(frame, field) for frame in compatible.values()}
        if len(units) != 1 or len(representations) != 1:
            msg = f"EDA spectral field {field!r} requires matching units and stored representations."
            raise ValueError(msg)
    return resolution


def _field_representation(frame: pd.DataFrame, field: str) -> str:
    """Return one authoritative stored or derived field representation."""
    return capabilities.field_representation(frame, field)


def _stored_representation_label(frame: pd.DataFrame, field: str) -> str:
    """Return one explicit human-readable stored-value representation."""
    representation = _field_representation(frame, field)
    labels = {**_DIMENSIONLESS_REPRESENTATION_LABELS, **_PHYSICAL_REPRESENTATION_LABELS}
    try:
        return labels[representation]
    except KeyError as error:
        msg = f"EDA field {field!r} has unsupported stored representation {representation!r}."
        raise ValueError(msg) from error


def _field_row_label(frame: pd.DataFrame, field: str) -> str:
    """Return one unit-bearing row label; it names a dimensionless stored representation itself."""
    return capabilities.field_quantity_label(frame, field, mathtext=True)


def _finish_matrix(
    figure: Figure,
    axes: np.ndarray,
    resolution: capabilities.ResolvedFieldSelection,
    *,
    title: str,
    row_headers: Sequence[str],
    legend_axes: Iterable[Axes] = (),
) -> None:
    """Write the row headers, the title with its availability note and the dataset legend collected from ``legend_axes``."""
    figure_layout.set_row_headers(figure, axes, row_headers)
    figure_layout.set_figure_title(figure, axes, title, note=capabilities.availability_note(resolution) or None)
    handles, labels = figure_layout.legend_entries(legend_axes)
    if labels:
        wrapped = [display_labels.wrapped_dataset_display_label(label, width=transient_plots.LEGEND_WRAP_WIDTH) for label in labels]
        figure_layout.add_legend(figure, axes, handles, wrapped, title="Dataset")


def _hide_x_text(axis: Axes) -> None:
    """Drop the x label and tick labels of a panel above the bottom row."""
    axis.set_xlabel("")
    axis.tick_params(axis="x", labelbottom=False)


def _spectral_power_ylabel(frame: pd.DataFrame, field: str) -> str:
    """Label spectral power of the displayed values, whose mean-removed fluctuations carry the difference unit."""
    representation = _field_representation(frame, field)
    if representation in _DIMENSIONLESS_REPRESENTATION_LABELS:
        return f"Mean spectral power [-]\nStored: {_stored_representation_label(frame, field)}"
    if representation not in _PHYSICAL_VALUE_REPRESENTATIONS:
        msg = f"EDA field {field!r} has unsupported stored representation {representation!r}."
        raise ValueError(msg)
    return f"Mean spectral power [({capabilities.field_display_unit(frame, field, quantity_kind='difference')})²]"


def _spacing(row: pd.Series) -> tuple[float, float, str]:
    """
    Derive positive median Cartesian x/y spacing from one EDA case.

    Explicit finite task inputs ``x`` and ``y`` must each contain at least two
    increasing unique coordinates. The maintained EDA coordinate contract uses
    metres, returned as the disclosed unit string.
    """
    x_grid, y_grid = _coordinate_values(row)
    x_values = np.unique(x_grid)
    y_values = np.unique(y_grid)
    if x_values.size < _MIN_GRID_SIZE or y_values.size < _MIN_GRID_SIZE:
        msg = "EDA coordinate fields must contain at least two unique values per axis."
        raise ValueError(msg)
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    if dx <= 0.0 or dy <= 0.0 or not np.isfinite((dx, dy)).all():
        msg = "EDA coordinate spacing must be finite and positive."
        raise ValueError(msg)
    return dx, dy, "m"


def _power_grid(field: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean-centered Hann-windowed FFT power on physical frequency grids.

    The input must be one finite 2D field with both axes at least length two.
    Power is ``abs(fft2((field-mean)*window))**2 / field.size``. Frequency grids
    use the caller's positive physical ``dx`` and ``dy``.
    """
    values = np.asarray(field, dtype=float)
    if values.ndim != _MIN_GRID_SIZE or min(values.shape) < _MIN_GRID_SIZE or not np.isfinite(values).all():
        msg = "EDA spectra require finite 2D fields."
        raise ValueError(msg)
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    transformed = np.fft.fft2((values - np.mean(values)) * window)
    power = np.abs(transformed) ** 2 / values.size
    kx = np.fft.fftfreq(values.shape[1], d=dx)
    ky = np.fft.fftfreq(values.shape[0], d=dy)
    return power, *np.meshgrid(kx, ky)


def _binned_mean(coordinate: np.ndarray, power: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce frequency-grid power into fixed equal-width coordinate bins.

    Bin centers cover zero through the observed maximum. Empty bins are retained
    with zero mean power so spectra from identical grids stay aligned.
    """
    values = np.asarray(coordinate, dtype=float).ravel()
    energy = np.asarray(power, dtype=float).ravel()
    edges = np.linspace(0.0, float(np.max(values)), bins + 1)
    assignments = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    sums = np.bincount(assignments, weights=energy, minlength=bins)
    counts = np.bincount(assignments, minlength=bins)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def _spectra(field: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return isotropic radial and absolute x/y directional mean-power spectra.

    All three reductions share ``max(2, min(shape)//2)`` equal-width bins and
    preserve physical inverse-coordinate frequencies.
    """
    power, kx, ky = _power_grid(field, dx=dx, dy=dy)
    bins = max(_MIN_GRID_SIZE, min(field.shape) // _MIN_GRID_SIZE)
    radial_k, radial = _binned_mean(np.hypot(kx, ky), power, bins=bins)
    x_k, x_energy = _binned_mean(np.abs(kx), power, bins=bins)
    y_k, y_energy = _binned_mean(np.abs(ky), power, bins=bins)
    return radial_k, radial, x_k, x_energy, y_k, y_energy


def _row_spectra(
    frame: pd.DataFrame,
    row: pd.Series,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute all maintained spectra for one exact EDA case row."""
    dx, dy, coordinate_unit = _spacing(row)
    return (*_spectra(_field_values(frame, row, field), dx=dx, dy=dy), coordinate_unit)


def _case_spectra(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Stack aligned isotropic and directional spectra for one declared field.

    The first bounded saved prefix is used without reranking. Frequency grids
    must remain identical within the frame. The result carries exact selected
    count and coordinate unit for disclosure.
    """
    selected = frame.iloc[: min(max_cases, len(frame))]
    collections: list[list[np.ndarray]] = [[], [], []]
    coordinates: list[np.ndarray | None] = [None, None, None]
    coordinate_unit = "m"
    for _index, row in selected.iterrows():
        radial_k, radial, x_k, x_energy, y_k, y_energy, coordinate_unit = _row_spectra(frame, row, field)
        for axis_index, (k_values, energy) in enumerate(((radial_k, radial), (x_k, x_energy), (y_k, y_energy))):
            reference = coordinates[axis_index]
            if reference is None:
                coordinates[axis_index] = k_values
            elif not np.allclose(reference, k_values):
                msg = "EDA spectral aggregation requires identical grids within each dataset."
                raise ValueError(msg)
            collections[axis_index].append(energy)
    radial_coordinate, x_coordinate, y_coordinate = coordinates
    if radial_coordinate is None or x_coordinate is None or y_coordinate is None:
        msg = "EDA spectral aggregation did not establish frequency grids."
        raise RuntimeError(msg)
    return (
        radial_coordinate,
        np.stack(collections[0]),
        x_coordinate,
        np.stack(collections[1]),
        y_coordinate,
        np.stack(collections[2]),
        len(selected),
        coordinate_unit,
    )


def _spectral_evolution_case_map(
    frame: pd.DataFrame,
    row: pd.Series,
    field: str,
    *,
    orientation: SpectralEvolutionOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return one correctly oriented position-resolved spectral-fraction map."""
    values = _field_values(frame, row, field)
    x_grid, y_grid = _coordinate_values(row)
    if values.ndim != _MIN_GRID_SIZE or x_grid.shape != values.shape or y_grid.shape != values.shape:
        msg = "Spectral evolution requires field, x, and y arrays on the same 2D grid."
        raise ValueError(msg)
    if not np.isfinite(values).all() or not np.isfinite(x_grid).all() or not np.isfinite(y_grid).all():
        msg = "Spectral evolution requires finite field and coordinate arrays."
        raise ValueError(msg)
    x_values = np.median(x_grid, axis=0)
    y_values = np.median(y_grid, axis=1)
    if x_values.size < _MIN_GRID_SIZE or y_values.size < _MIN_GRID_SIZE:
        msg = "Spectral evolution requires at least two grid points per axis."
        raise ValueError(msg)
    dx_values = np.diff(x_values)
    dy_values = np.diff(y_values)
    if np.any(dx_values <= 0.0) or np.any(dy_values <= 0.0):
        msg = "Spectral evolution requires increasing rectilinear coordinates."
        raise ValueError(msg)

    if orientation == _CROSS_STREAM_ALONG_FLOW:
        transform_axis = 1
        spacing = float(np.median(dx_values))
        position = y_values
        window = np.hanning(values.shape[1])[np.newaxis, :]
        centered = values - np.mean(values, axis=1, keepdims=True)
        transformed = np.fft.rfft(centered * window, axis=transform_axis)[:, 1:]
        power = np.abs(transformed) ** 2 / values.shape[1]
    elif orientation == _FLOW_ACROSS_CROSS_STREAM:
        transform_axis = 0
        spacing = float(np.median(dy_values))
        position = x_values
        window = np.hanning(values.shape[0])[:, np.newaxis]
        centered = values - np.mean(values, axis=0, keepdims=True)
        transformed = np.fft.rfft(centered * window, axis=transform_axis)[1:, :]
        power = (np.abs(transformed) ** 2 / values.shape[0]).T
    else:
        msg = f"Unsupported spectral-evolution orientation: {orientation!r}."
        raise ValueError(msg)

    frequency = np.fft.rfftfreq(values.shape[transform_axis], d=spacing)[1:]
    totals = np.sum(power, axis=1, keepdims=True)
    fractions = np.divide(power, totals, out=np.zeros_like(power), where=totals > 0.0)
    return frequency, position, fractions, "m"


def _spectral_evolution_map(
    frame: pd.DataFrame,
    field: str,
    *,
    max_cases: int,
    orientation: SpectralEvolutionOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Return median spectral fractions on one frequency/retained-position grid."""
    selected = frame.iloc[: min(max_cases, len(frame))]
    spectra: list[np.ndarray] = []
    reference_frequency: np.ndarray | None = None
    reference_position: np.ndarray | None = None
    coordinate_unit = "m"
    for _index, row in selected.iterrows():
        frequency, position, fractions, coordinate_unit = _spectral_evolution_case_map(
            frame,
            row,
            field,
            orientation=orientation,
        )
        if reference_frequency is None:
            reference_frequency = frequency
            reference_position = position
        elif not np.allclose(reference_frequency, frequency) or reference_position is None or not np.allclose(reference_position, position):
            msg = "Spectral-evolution aggregation requires identical grids within each dataset."
            raise ValueError(msg)
        spectra.append(fractions)
    if reference_frequency is None or reference_position is None or not spectra:
        msg = "Spectral-evolution aggregation did not establish a frequency-position grid."
        raise RuntimeError(msg)
    return reference_frequency, reference_position, np.median(np.stack(spectra), axis=0), len(selected), coordinate_unit


def _orientation_labels(orientation: SpectralEvolutionOrientation, coordinate_unit: str) -> tuple[str, str, str]:
    """Return figure title and explicit frequency/position axis labels."""
    if orientation == _CROSS_STREAM_ALONG_FLOW:
        return (
            "Cross-stream spectral evolution along flow direction y",
            f"Cross-stream spatial frequency k_x [1/{coordinate_unit}]",
            f"Flow-direction position y [{coordinate_unit}]",
        )
    if orientation == _FLOW_ACROSS_CROSS_STREAM:
        return (
            "Flow-direction spectral evolution across cross-stream direction x",
            f"Flow-direction spatial frequency k_y [1/{coordinate_unit}]",
            f"Cross-stream position x [{coordinate_unit}]",
        )
    msg = f"Unsupported spectral-evolution orientation: {orientation!r}."
    raise ValueError(msg)


def _power_fraction_scale() -> field_maps.FieldMapDefinition:
    """Reuse eleven logarithmic bands for the documented fixed power fraction range."""
    limits = np.asarray((_POWER_FRACTION_FLOOR, 1.0))
    return field_maps.build_field_map(
        limits,
        semantic=field_maps.FieldMapSemantic.LOG_POSITIVE,
        field="row_normalized_power_fraction",
        unit="1",
        comparison_scope="all_spectral_evolution_views",
        locked_boundaries=np.geomspace(_POWER_FRACTION_FLOOR, 1.0, field_maps.N_COLOR_BINS + 1),
        bound_policy="fixed_numerical_floor_1e-12_to_one",
        colormap_name="magma",
    )


def _line(style: SeriesStyle, *, cumulative: bool = False) -> dict[str, Any]:
    """Return the line kwargs of one dataset: its color and batch line style; cumulative energy is the same line, thinner."""
    return {"color": style.color, "linestyle": style.linestyle, "linewidth": _CUMULATIVE_LINEWIDTH if cumulative else _POWER_LINEWIDTH}


def _band(axis: Axes, k_values: np.ndarray, energy: np.ndarray, *, label: str, style: SeriesStyle) -> None:
    """Plot positive median power in the dataset's color and line style with q10-q90 case bands."""
    q10, median, q90 = np.quantile(energy, (0.1, 0.5, 0.9), axis=0)
    valid = (k_values > 0.0) & (median > 0.0)
    axis.plot(k_values[valid], median[valid], label=label, **_line(style))
    band = valid & (q10 > 0.0)
    axis.fill_between(k_values[band], q10[band], q90[band], color=style.color, alpha=0.18)


def _cumulative(energy: np.ndarray) -> np.ndarray:
    """Return casewise cumulative energy after omitting the DC bin."""
    positive = np.maximum(energy[:, 1:], 0.0)
    cumulative = np.cumsum(positive, axis=1)
    totals = cumulative[:, -1:]
    return np.divide(cumulative, totals, out=np.zeros_like(cumulative), where=totals > 0.0)


def _set_log_frequency_axis(axis: Axes) -> None:
    """Use readable major log ticks without dense minor-frequency labels."""
    figure_layout.set_log_scale(axis, "x")
    axis.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
    axis.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(float(value) for value in np.arange(2, 10) * 0.1), numticks=12))


def _isotropic_figure(
    selected_datasets: dict[str, pd.DataFrame],
    resolution: capabilities.ResolvedFieldSelection,
    *,
    title: str,
    styles: Mapping[str, SeriesStyle],
    max_cases: int,
    case_number: int | None = None,
) -> Figure:
    """
    Draw per field the isotropic power and cumulative energy of the selected datasets: the ``max_cases`` prefix or one ``case_number``.

    ``styles`` is resolved over every loaded dataset, so a dataset keeps its style whatever the selection.
    """
    figure, axes = figure_layout.panel_figure(len(resolution.fields), 2, panel_in=_PANEL_IN)
    row_headers: list[str] = []
    for field_index, field in enumerate(resolution.fields):
        field_datasets = capabilities.compatible_frames(selected_datasets, resolution, field)
        reference_frame = next(iter(field_datasets.values()))
        power_axis, cumulative_axis = axes[field_index]
        coordinate_unit = "m"
        for label, frame in field_datasets.items():
            if case_number is None:
                radial_k, radial, _x_k, _x_energy, _y_k, _y_energy, _count, coordinate_unit = _case_spectra(frame, field, max_cases=max_cases)
                _band(power_axis, radial_k, radial, label=label, style=styles[label])
                q10, median, q90 = np.quantile(_cumulative(radial), (0.1, 0.5, 0.9), axis=0)
                cumulative_axis.plot(radial_k[1:], median, **_line(styles[label], cumulative=True))
                cumulative_axis.fill_between(radial_k[1:], q10, q90, color=styles[label].color, alpha=0.18)
            else:
                radial_k, radial, _x_k, _x_energy, _y_k, _y_energy, coordinate_unit = _row_spectra(frame, _case_row(frame, case_number), field)
                valid = (radial_k > 0.0) & (radial > 0.0)
                power_axis.plot(radial_k[valid], radial[valid], label=label, **_line(styles[label]))
                cumulative_axis.plot(radial_k[1:], _cumulative(radial[np.newaxis, :])[0], **_line(styles[label], cumulative=True))
        _set_log_frequency_axis(power_axis)
        figure_layout.set_log_scale(power_axis, "y")
        _set_log_frequency_axis(cumulative_axis)
        cumulative_axis.set_ylim(0.0, 1.02)
        power_axis.set_ylabel(_spectral_power_ylabel(reference_frame, field))
        cumulative_axis.set_ylabel("Cumulative energy [-]")
        for axis in axes[field_index]:
            axis.grid(alpha=0.25, which="both")
            if field_index == len(resolution.fields) - 1:
                axis.set_xlabel(f"Spatial frequency k [1/{coordinate_unit}]")
            else:
                _hide_x_text(axis)
        row_headers.append(_field_row_label(reference_frame, field))
    figure_layout.set_panel_title(axes[0, 0], "Isotropic power")
    figure_layout.set_panel_title(axes[0, 1], "Cumulative energy")
    _finish_matrix(figure, axes, resolution, title=title, row_headers=row_headers, legend_axes=axes[:, 0])
    return figure


def plot_isotropic_spectral_summary(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    dataset_names: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
) -> Figure:
    """
    Compare isotropic field spectra and cumulative bandwidth across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames with physical coordinates, fields, and units.
    styles : collections.abc.Mapping[str, SeriesStyle]
        Color and line style of every dataset, resolved by the viewer over all loaded datasets.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.
    dataset_names : collections.abc.Sequence[str] | None, optional
        Explicit dataset labels to compare. Omission preserves all input labels.
    channels : collections.abc.Sequence[str] | None, optional
        Compatible spectral channels to render. Omission selects all compatible
        task-declared channels in shared semantic order.

    Returns
    -------
    matplotlib.figure.Figure
        Per-field radial median power with q10--q90 case bands and cumulative
        non-DC energy fractions, each disclosing selected case counts.

    Raises
    ------
    ValueError, RuntimeError
        If task/field/unit contracts, finite Cartesian grids, or aligned spectral
        bins cannot establish a comparison.

    Notes
    -----
    Power retains squared stored-value units. Transformed dimensionless
    representations remain dimensionless. Cumulative energy is dimensionless,
    and no interpolation is used when within-frame frequency grids differ.

    """
    selected_datasets = _select_datasets(datasets, dataset_names)
    resolution = _validate_datasets(
        selected_datasets,
        max_cases=max_cases,
        channels=channels,
    )
    case_text = _case_count_text(selected_datasets, max_cases=max_cases)
    return _isotropic_figure(
        selected_datasets,
        resolution,
        title=f"Isotropic spectra — {case_text}{_stored_state_scope_text(selected_datasets)}",
        styles=styles,
        max_cases=max_cases,
    )


def plot_isotropic_spectral_case(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    case_number: int,
    dataset_names: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
) -> Figure:
    """Compare one exact dataset-local case's isotropic spectra."""
    selected_datasets = _select_datasets(datasets, dataset_names)
    resolution = _validate_datasets(
        selected_datasets,
        max_cases=1,
        channels=channels,
    )
    return _isotropic_figure(
        selected_datasets,
        resolution,
        title=f"Isotropic spectra — Case {case_number}{_stored_state_scope_text(selected_datasets)}",
        styles=styles,
        max_cases=1,
        case_number=case_number,
    )


def _directional_axis(
    axis: Axes,
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    field: str,
    direction: str,
    max_cases: int,
    case_number: int | None = None,
    last_column: bool,
) -> Axes:
    """
    Plot one physical directional spectrum and return its cumulative twin axis, whose entries join the shared legend.

    The power y label belongs to the row (``label_rows_once``); only the last column writes the cumulative axis' label and ticks.
    """
    cumulative_axis = axis.twinx()
    coordinate_unit = "m"
    has_positive_power = False
    for label, frame in datasets.items():
        if case_number is None:
            _radial_k, _radial, x_k, x_energy, y_k, y_energy, _count, coordinate_unit = _case_spectra(
                frame,
                field,
                max_cases=max_cases,
            )
        else:
            row = _case_row(frame, case_number)
            _radial_k, _radial, x_k, x_values, y_k, y_values, coordinate_unit = _row_spectra(
                frame,
                row,
                field,
            )
            x_energy = x_values[np.newaxis, :]
            y_energy = y_values[np.newaxis, :]
        if direction == "y":
            k_values, energy = y_k, y_energy
        elif direction == "x":
            k_values, energy = x_k, x_energy
        else:
            message = f"Unsupported spectral direction: {direction!r}."
            raise ValueError(message)
        style = styles[label]
        if case_number is None:
            median_power = np.quantile(energy, 0.5, axis=0)
            has_positive_power = has_positive_power or bool(np.any((k_values > 0.0) & (median_power > 0.0)))
            _band(axis, k_values, energy, label=f"{label} power", style=style)
        else:
            power = energy[0]
            valid = (k_values > 0.0) & (power > 0.0)
            has_positive_power = has_positive_power or bool(np.any(valid))
            axis.plot(k_values[valid], power[valid], label=f"{label} power", **_line(style))
        cumulative = _cumulative(energy)
        curve = np.quantile(cumulative, 0.5, axis=0) if case_number is None else cumulative[0]
        cumulative_axis.plot(k_values[1:], curve, label=f"{label} cumulative", **_line(style, cumulative=True))
    _set_log_frequency_axis(axis)
    figure_layout.set_log_scale(axis, "y")
    if direction == "y":
        axis.set_xlabel(f"Flow-direction spatial frequency k_y [1/{coordinate_unit}]")
    else:
        axis.set_xlabel(f"Cross-stream spatial frequency k_x [1/{coordinate_unit}]")
    cumulative_axis.set_ylim(0.0, 1.02)
    cumulative_axis.set_ylabel("Cumulative energy [-]" if last_column else "", fontsize=figure_layout.LABEL_PT)
    cumulative_axis.tick_params(labelright=last_column, labelsize=figure_layout.TICK_PT)
    if not has_positive_power:
        axis.text(0.5, 0.5, "No positive non-DC spectral power", transform=axis.transAxes, ha="center", va="center")
    axis.grid(alpha=0.25, which="both")
    return cumulative_axis


def _directional_figure(
    selected_datasets: dict[str, pd.DataFrame],
    resolution: capabilities.ResolvedFieldSelection,
    *,
    title: str,
    styles: Mapping[str, SeriesStyle],
    max_cases: int,
    case_number: int | None = None,
) -> Figure:
    """
    Draw per field the flow-y and cross-stream-x spectra: the ``max_cases`` prefix or one ``case_number``.

    The panels of a row share their power axis, which the first column labels. ``styles`` is resolved over every loaded dataset.
    """
    figure, axes = figure_layout.panel_figure(len(resolution.fields), 2, panel_in=_PANEL_IN, sharey="row")
    legend_axes: list[Axes] = []  # each panel, then its cumulative twin: the order of the legend entries
    row_headers: list[str] = []
    for field_index, field in enumerate(resolution.fields):
        field_datasets = capabilities.compatible_frames(selected_datasets, resolution, field)
        for axis_index, direction in enumerate(("y", "x")):
            axis = axes[field_index, axis_index]
            cumulative_axis = _directional_axis(
                axis,
                datasets=field_datasets,
                styles=styles,
                field=field,
                direction=direction,
                max_cases=max_cases,
                case_number=case_number,
                last_column=axis_index == 1,
            )
            legend_axes += [axis, cumulative_axis]
            if field_index < len(resolution.fields) - 1:
                _hide_x_text(axis)
        reference_frame = next(iter(field_datasets.values()))
        figure_layout.label_rows_once(axes[field_index : field_index + 1], _spectral_power_ylabel(reference_frame, field))
        row_headers.append(_field_row_label(reference_frame, field))
    figure_layout.set_panel_title(axes[0, 0], "Flow direction y (k_y)")
    figure_layout.set_panel_title(axes[0, 1], "Cross-stream direction x (k_x)")
    _finish_matrix(figure, axes, resolution, title=title, row_headers=row_headers, legend_axes=legend_axes)
    return figure


def plot_directional_spectral_summary(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    dataset_names: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
) -> Figure:
    """
    Compare flow-direction y and cross-stream x spectra across datasets.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames on internally identical Cartesian grids.
    styles : collections.abc.Mapping[str, SeriesStyle]
        Color and line style of every dataset, resolved by the viewer over all loaded datasets.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.
    dataset_names : collections.abc.Sequence[str] | None, optional
        Explicit dataset labels to compare. Omission preserves all input labels.
    channels : collections.abc.Sequence[str] | None, optional
        Compatible spectral channels to render. Omission selects all compatible
        task-declared channels in shared semantic order.

    Returns
    -------
    matplotlib.figure.Figure
        Separate flow-y/cross-stream-x panels per field, with median directional
        power, q10--q90 bands, cumulative energy, and disclosed case counts.

    Raises
    ------
    ValueError, RuntimeError
        If task/field/unit metadata, finite grids, or aligned frequency bins fail.

    Notes
    -----
    Directional spectra remain separate so anisotropic bandwidth is visible. No
    scalar cross-direction score is calculated.

    """
    selected_datasets = _select_datasets(datasets, dataset_names)
    resolution = _validate_datasets(
        selected_datasets,
        max_cases=max_cases,
        channels=channels,
    )
    case_text = _case_count_text(selected_datasets, max_cases=max_cases)
    return _directional_figure(
        selected_datasets,
        resolution,
        title=f"Directional spectra — {case_text}{_stored_state_scope_text(selected_datasets)}",
        styles=styles,
        max_cases=max_cases,
    )


def plot_directional_spectral_case(
    *,
    datasets: dict[str, pd.DataFrame],
    styles: Mapping[str, SeriesStyle],
    case_number: int,
    dataset_names: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
) -> Figure:
    """Compare one exact case along flow-y and cross-stream-x directions."""
    selected_datasets = _select_datasets(datasets, dataset_names)
    resolution = _validate_datasets(
        selected_datasets,
        max_cases=1,
        channels=channels,
    )
    return _directional_figure(
        selected_datasets,
        resolution,
        title=f"Directional spectra — Case {case_number}{_stored_state_scope_text(selected_datasets)}",
        styles=styles,
        max_cases=1,
        case_number=case_number,
    )


def _evolution_figure(
    selected_datasets: dict[str, pd.DataFrame],
    resolution: capabilities.ResolvedFieldSelection,
    *,
    orientation: SpectralEvolutionOrientation,
    scope: str,
    evolution_map: Callable[[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, str]],
) -> Figure:
    """
    Draw per field and dataset one frequency/position map; ``evolution_map(label, field)`` returns frequency, position, fractions and unit.

    Every colour bar carries its unit; the position label stands once per row and the frequency label once per column.
    """
    figure, axes = figure_layout.panel_figure(len(resolution.fields), len(selected_datasets), panel_in=_PANEL_IN)
    plot_title, xlabel, ylabel = "Spectral evolution", "", ""
    row_headers: list[str] = []
    for field_index, field in enumerate(resolution.fields):
        field_datasets = capabilities.compatible_frames(selected_datasets, resolution, field)
        for dataset_index, label in enumerate(selected_datasets):
            axis = axes[field_index, dataset_index]
            if label not in field_datasets:
                axis.set_axis_off()
                axis.text(0.5, 0.5, f"{field} unavailable", transform=axis.transAxes, ha="center", va="center")
                continue
            frequency, position, fractions, coordinate_unit = evolution_map(label, field)
            scale = _power_fraction_scale()
            image = axis.pcolormesh(
                frequency,
                position,
                np.maximum(fractions, _POWER_FRACTION_FLOOR),
                shading="auto",
                **scale.pcolormesh_kwargs(),
            )
            _set_log_frequency_axis(axis)
            plot_title, xlabel, ylabel = _orientation_labels(orientation, coordinate_unit)
            if not np.any(fractions > 0.0):
                axis.text(0.5, 0.5, "No positive non-DC spectral power", transform=axis.transAxes, ha="center", va="center", color="white")
            colorbar = figure.colorbar(image, ax=axis, **scale.colorbar_kwargs())
            decades = np.arange(0, np.log10(_POWER_FRACTION_FLOOR) - 1, -2, dtype=int)  # round steps 10^0, 10^-2, ... 10^-12
            colorbar.set_ticks((10.0**decades).tolist(), labels=[f"$10^{{{value}}}$" for value in decades])
            colorbar.set_label(_COLORBAR_LABEL, size=figure_layout.LABEL_PT)
            colorbar.ax.tick_params(labelsize=figure_layout.TICK_PT)
        row_headers.append(_field_row_label(next(iter(field_datasets.values())), field))
    for column, label in enumerate(selected_datasets):  # title on the topmost and x text on the lowest available panel of each column
        shown_rows = [axis for axis in axes[:, column] if axis.axison]
        if shown_rows:
            figure_layout.set_panel_title(
                shown_rows[0], display_labels.wrapped_dataset_display_label(label, width=_HEADER_WRAP_WIDTH, max_lines=_HEADER_MAX_LINES)
            )
            shown_rows[-1].set_xlabel(xlabel)
            for axis in shown_rows[:-1]:
                _hide_x_text(axis)
    for row in axes:  # the row's label goes on its first panel that shows data, not on an unavailable one
        shown = next((axis for axis in row if axis.axison), None)
        if shown is not None:
            shown.set_ylabel(ylabel, fontsize=figure_layout.LABEL_PT)
    _finish_matrix(
        figure,
        axes,
        resolution,
        title=f"{plot_title} — {scope}{_stored_state_scope_text(selected_datasets)}",
        row_headers=row_headers,
    )
    return figure


def plot_vertical_spectral_evolution(
    *,
    datasets: dict[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    dataset_names: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    orientation: SpectralEvolutionOrientation = _CROSS_STREAM_ALONG_FLOW,
) -> Figure:
    """
    Plot either position-resolved physical spectral orientation.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Task-compatible EDA frames with identical declared fields/units and an
        internally shared increasing Cartesian grid.
    max_cases : int, optional
        Positive bound on the stored ordered prefix aggregated per dataset.
    dataset_names : collections.abc.Sequence[str] | None, optional
        Explicit dataset labels to compare. Omission preserves all input labels.
    channels : collections.abc.Sequence[str] | None, optional
        Compatible spectral channels to render. Omission selects all compatible
        task-declared channels in shared semantic order.
    orientation : {"cross_stream_along_flow", "flow_across_cross_stream"}, optional
        Spatial transform and retained-position orientation. The default preserves
        cross-stream k_x spectra resolved along flow-direction position y.

    Returns
    -------
    matplotlib.figure.Figure
        One frequency/orthogonal-position map per dataset and field. Values are
        casewise median log10 row-normalized power fractions.

    Raises
    ------
    ValueError, RuntimeError
        If dataset metadata, fields, coordinates, finite values, or within-frame
        grids cannot establish the declared spectral comparison.

    Notes
    -----
    The selected transform axis is mean-centered and Hann-windowed before the
    real FFT. Omitting the DC bin and normalizing at each retained position makes
    the map dimensionless. Both orientations use a fixed 1e-12 numerical floor
    and log10 color range from -12 to 0, including honest zero-energy annotation.

    """
    selected_datasets = _select_datasets(datasets, dataset_names)
    resolution = _validate_datasets(
        selected_datasets,
        max_cases=max_cases,
        channels=channels,
    )
    case_text = _case_count_text(selected_datasets, max_cases=max_cases)

    def evolution_map(label: str, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        frequency, position, fractions, _count, coordinate_unit = _spectral_evolution_map(
            selected_datasets[label],
            field,
            max_cases=max_cases,
            orientation=orientation,
        )
        return frequency, position, fractions, coordinate_unit

    return _evolution_figure(
        selected_datasets,
        resolution,
        orientation=orientation,
        scope=case_text,
        evolution_map=evolution_map,
    )


def plot_vertical_spectral_case(
    *,
    datasets: dict[str, pd.DataFrame],
    case_number: int,
    dataset_names: Sequence[str] | None = None,
    channels: Sequence[str] | None = None,
    orientation: SpectralEvolutionOrientation = _CROSS_STREAM_ALONG_FLOW,
) -> Figure:
    """Compare either position-resolved spectral orientation for one case."""
    selected_datasets = _select_datasets(datasets, dataset_names)
    resolution = _validate_datasets(
        selected_datasets,
        max_cases=1,
        channels=channels,
    )
    return _evolution_figure(
        selected_datasets,
        resolution,
        orientation=orientation,
        scope=f"Case {case_number}",
        evolution_map=lambda label, field: _spectral_evolution_case_map(
            selected_datasets[label],
            _case_row(selected_datasets[label], case_number),
            field,
            orientation=orientation,
        ),
    )
