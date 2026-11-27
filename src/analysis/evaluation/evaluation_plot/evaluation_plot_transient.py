"""Render sequence-aware transient-drying Evaluation evidence.

Responsibilities:
  - Compare reference, prediction, and signed-error fields at exact physical times
  - Draw state maps on the fixed map-grid geometry: state scales from the 5th to the 95th percentile, per drawn time or,
    per field, pooled over all stored times (``fixed_time_fields``; ``default_fixed_time_fields`` names all but moisture), with arrows
    for the cells beyond (on the state bars fixed over the scaled times), error scales (signed by default) fixed over the
    scaled times, tick decimals fixed per colour bar over the scaled times, one scale note under the title and a mean |e|
    line above each error map
  - Draw aggregate error maps and parity at one exact time or drying phase on the same geometry: a row per field, a column per context,
    the signed mean by default (or the absolute or train-sigma-normalized one), one round scale per field and dataset shared by the
    runs compared on it, a unit on every bar, each context's n(t) or n(τ) in its header (also 0), its run on a line of its own
    from the given dataset labels, else from all its given records; parity limits per field and dataset, a case's moisture mean
    over the cells valid in both
  - Plot physical state trajectories and exact grain-moisture derivations, and the error over physical time, in time
    windows (the first 24 h, the middle in days, the last 24 h) or over the drying phase: a prediction-minus-reference row
    under each state mean, one line per distinct target moisture on the grain moisture, its value in a strip right of the last
    window, a case-count row with staggered series markers, one legend outside at the top right
  - Plot rollout error across horizons and reduction scopes, one grey x on the axis line where only some series have no case at
    a horizon, a tick that says when no series has one, one keyed ring for the full-horizon points and one grouped
    missing-horizon note
  - Plot per-field relative norms over physical time or the drying phase, one curve per dataset choice and frame, with their
    supported case counts in a strip under each norm and an x where a non-finite prediction voids a field
  - Present target, matched-compute (one panel per mode and horizon), pipeline-degradation (one panel per selected run in
    its colour on one shared value scale, C's reason in words, each distinct note line once, runs without evidence named), and
    timing evidence; duration ratios as a point at the median of the paired per-case ratios with their 10th to 90th
    percentile as whisker, the ratio of summed durations in the note
  - Name the values an error-vs-magnitude trend leaves out, one line per run in the legend's order, wrapped to the panel width:
    raw non-finite apart from display-masked water

Design principles:
  - Styles are resolved once per comparison from all its loaded runs (``series.comparison_styles``) and passed to every view, which
    looks each series up by its entities (model kind, Dataset role, material, selected checkpoint, read alike from record
    identity and session provenance) and never resolves its own rows: a run (model kind and checkpoint) keeps one colour in every
    view, also where a view holds evidence for only some runs; a figure of run colours says «colour = run» in its context line;
    legends follow ``figure_layout.add_legend``, from two runs on grouped under each run with the entries without it; a point
    legend names a run whose series share one colour but differ in marker once, as a colour swatch, beside a key of the
    material markers in the dataset selector's order
  - A linear axis of run values turns logarithmic when a run's largest value exceeds 20 times the largest median of every other
    run's plotted groups (box, bar, point series); its label says so and the note counts the non-positive values it cannot show.
    A linear error axis never reaches below zero, and a parity plot keeps equal limits on both axes, so a later legend fit
    (``figure_layout``) widens them alike; an endpoint pair with an unavailable or non-finite side is counted, never drawn
  - Dynamic states follow the shared EDA labels, units, and display conversions in the active label style: phi in
    percent with errors in pp, and the water densities w_surf, w_int and w_gr as surface, internal and grain moisture in
    percent of wet basis with errors in pp, converted per case with its dry bulk density; cells with negative or
    non-finite water are masked
  - Normalized errors stay on the stored states and saved train scales
  - Physical times are selected exactly and never nearest-neighbour substituted
  - Cases whose ends differ (exactly compared) share the last window as hours before their own end; the windows only select
    stored times
  - Over the drying phase every view pools by ``time_ui.phase_cohort`` (reference and prediction on the reference's τ) and
    names its cohort label; a case of the error over time without a time after its origin is refused, never dropped
  - A duration ratio is drawn on its log scale only when its median and percentiles are positive and finite
  - A title names the compared runs (``models``); a compared run a view cannot draw stays named, and the view's note or panel says why
  - Endpoint and cumulative reductions remain visually and semantically distinct
  - The material marker tells a run's materials apart where colour and hatch do not (bars, box medians, horizon and
    relative-norm lines, target, matched-budget and runtime points); grouped views list a run's series together in the dataset
    selector's order (role group, then material)
  - Device and compute-route evidence goes to the kernel log, never onto a figure
  - A spread (standard-deviation map, 10th-90th band) needs two cases: with fewer it is masked or not drawn, never shown as zero;
    series whose references alone (whatever their predictions mask) in a window rest on the same cases at every stored time,
    each finite or not alike, and agree exactly share one neutral reference curve there
  - A 10th-90th band is vertical whiskers in its series colour at sparse times of its own on its panel's common time grid, so
    bands never blend into a tint or share a time where avoidable, a reference's wide and pale like its curve, a zero-width one
    a short stroke across its curve; a whisker takes no line style, so it never reads as a series of a role, and one neutral key
    names each kind
  - A trajectory's reference curve is the reference's own mean, so a second run never changes it; its moisture prediction and
    difference use the cells valid in both, its difference is the mean of each case's own difference, and a raw non-finite
    value voids its time (marked); a snapshot keeps that case on the cells valid in both
  - A final time that stands in for an unreached target time is a lower bound and is marked as one, never drawn like an observed time

This module does NOT:
  - Load artifacts, run inference, refit scaling, or calculate speedup formulas
  - Alter steady Evaluation plots or silently fill unavailable evidence
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src import datasets, domain, learning
from src.analysis.evaluation import evaluation_transient_artifact as artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_session as transient_session
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as plot_layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_dataset_selection as dataset_selection
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_moisture_display as moisture_display
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_ui

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

_LOGGER = logging.getLogger(__name__)
_STATE_ARRAY_RANK = 4
_STATE_MAP_COLUMNS = 3  # reference, prediction, error
_SPATIAL_MAP_ERROR_COLUMN = 2
_ERROR_HEADERS = {
    error_diagnostics.SIGNED_ERROR: "Signed error",
    "Absolute error": "Absolute error",
    error_diagnostics.NORMALIZED_ABSOLUTE_ERROR: error_diagnostics.NORMALIZED_ERROR_HEADER,
}
_BEYOND_SCALE_COLOUR = "black"  # owner 14.11.: errors beyond the scale in a colour of their own
# A scale this narrow (relative to its value, at least 1) is the padded range of a constant field, see ``layout.linear_norm``.
_CONSTANT_SPAN = 4.0e-12
_MINIMUM_TRAJECTORY_TIMES = 2
_TARGET_MAGNITUDE_BIN_COUNT = 12
_TARGET_MAGNITUDE_POINT_LIMIT = 50_000
_GRAIN_MOISTURE_FIELD = "w_gr"
_MAX_ERROR_BATCH_INPUT_BYTES = 64 * 1024 * 1024
_ERROR_PAIR_MASK_BYTES = 17
_MIN_BAND_CASES = 2
_SEVERAL_RUNS = 2  # from this many runs a legend groups its entries by run
_WIDE_RANGE = 20.0  # a run's largest value beyond this many times every other run's largest group median puts the axis on a log scale
_LOG_PAD = 1.25  # factor of a log parity axis beyond its smallest and largest value
_WINDOW_PANEL_WIDTH_IN = 4.4  # one time window's column; the legend grows the figure by its own width
_WINDOW_PANELS_MIN_IN = 8.8  # the windows together, so a single (phase) window keeps its data readable beside a wide legend
_WINDOW_ROW_HEIGHT_IN = 2.6
_HALF_ROW = 0.5  # a difference or case-count row against a full row of 1
_FOOTER_LINE_IN = 0.15  # one 8 pt line of a map footer: the band below the maps grows by this per phase-note line
_LINE_SPACING = 1.25  # of a text line against its font size
_POINTS_PER_INCH = 72.0
_REFERENCE_LINE: Final = {"linewidth": 3.2, "alpha": 0.35}  # the reference is wide and pale beside the prediction in its series style
_PREDICTION_WIDTH = 1.6
_WHISKER: Final = {"linewidth": 1.2}  # a prediction's 10th-90th whisker; a reference's is wide and pale like its curve (``_REFERENCE_LINE``)
_WHISKER_KEY_PT = 10.0  # height of a whisker key's stroke
_FLAT_WHISKER_PT = 6.0  # height of the stroke a zero-width whisker shows, taller than its curve is wide, so it shows across it
_CONDITION_COLUMNS = 4  # panels per row of the matched-budget facets, one per mode and horizon
_STAGE_LABELS: Final = {"B/A": "B vs A\ndrying surrogate", "C/A": "C vs A\ncomplete pipeline", "C/B": "C vs B\nairflow substitution"}
_PIPELINE_CONDITIONS = (
    "Pipeline conditions, not training stages: A COMSOL airflow and drying, B drying surrogate on COMSOL airflow, "
    "C drying surrogate on predicted airflow"
)
_STAGE_BAR_WIDTH = 0.4
_NEUTRAL_COLOUR = "0.55"
_GROUPED_PANEL_IN = (8.0, 4.2)  # one channel comparison of bars or boxes per frame and material
_TARGET_COLOUR = "0.35"
_TARGET_LABEL_GAP_PT = 11.0  # target values closer than this are pushed apart
_TARGET_LABEL_OFFSET_PT = 4.0  # frame to the target value
_UNAVAILABLE_LABEL = "non-finite, unavailable"  # the legend shows the x marker beside it
_REASON_TOKEN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")  # a recorded reason token, read in words; any other reason text is prose
_NOTE_WORD = re.compile(r"(?:\$[^$]*\$|\S)+")  # a word of a note line, a mathtext span whole, since a break inside one fails to render
_FULL_HORIZON = "full horizon: each case to its own end"  # the key of the ring on a full-horizon point
_PARTIAL_HORIZON = "no case for a series the note names"  # the key of the x on a horizon some drawn series lack
_UNAVAILABLE_HEIGHT = 0.06  # of the row, above its x axis
_PAD_FRACTION = 0.05  # of a scatter or time axis's span on each side, so no point sits on the frame
_SUPPORT_ROW = 0.35  # a supported-case strip against a norm row of 1
_CURVE_MARKS = 10  # markers per relative-norm curve, about: sparse on a dense time grid, yet several on every curve
_SINGLETON_MARKER_SIZE = 40.0
# A final time is a lower bound on the time to target: an arrow from the point toward later times on that side's axis (x, y).
_BOUND_ARROWS: Final = (("reference", (1.0, 0.0)), ("prediction", (0.0, 1.0)))
_BOUND_ARROW_PX = 18.0  # arrow length on screen, so it never reads as a distance in hours
_RATIO_TICK_SUFFIX = "_speedup"  # a duration ratio's name ends so; its short tick label leaves it off
_OBJECTIVE_LABEL: Final = display_labels.evaluation_protocol_display_label("normalized_drying_group_macro_rmse")  # one label per metric


def _value(record: Any, name: str) -> Any:
    """Read one field from a validated sequence object or mapping."""
    if isinstance(record, Mapping):
        if name not in record:
            msg = f"Transient plot record lacks {name!r}."
            raise ValueError(msg)
        return record[name]
    if not hasattr(record, name):
        msg = f"Transient plot record lacks {name!r}."
        raise ValueError(msg)
    return getattr(record, name)


def _material_label(value: Any) -> str:
    """Return one shared human material label from authoritative identity evidence."""
    if not isinstance(value, str) or not value:
        msg = "Transient plot material_family must be non-empty text."
        raise ValueError(msg)
    return display_labels.material_display_label(value)


def _record_material(record: Any) -> str:
    """Return one record's authoritative material identity."""
    identity = _value(record, "identity")
    material = identity.get("material_family") if isinstance(identity, Mapping) else None
    if not isinstance(material, str) or not material:
        msg = "Transient plot record identity requires material_family."
        raise ValueError(msg)
    return material


def _record_entity(record: Any) -> tuple[Any, ...]:
    """Return one record's entity facts from its identity, ordered as ``series.ENTITY_COLUMNS``; the checkpoint is the selected one."""
    material = _record_material(record)  # also admits the identity as a mapping
    identity = _value(record, "identity")
    checkpoint = identity.get("checkpoint_identity")
    selected = checkpoint.get("best_checkpoint_sha256") if isinstance(checkpoint, Mapping) else None
    return identity.get("model_kind"), identity.get("dataset_role"), material, selected


def _record_styles(groups: Mapping[str, Sequence[Any]], styles: series.SeriesStyles) -> series.SeriesStyles:
    """Return the comparison style of every labelled record series by its records' identity."""
    return series.select_styles({label: [_record_entity(record) for record in records] for label, records in groups.items()}, styles)


def _require_provenance(label: str, rows: pd.DataFrame) -> None:
    """Refuse session summary rows without their provenance columns ``series.ENTITY_COLUMNS``."""
    if missing := [column for column in series.ENTITY_COLUMNS if column not in rows]:
        msg = f"Summary rows of series {label!r} lack the session provenance columns {missing}."
        raise ValueError(msg)


def _frame_styles(groups: Mapping[str, pd.DataFrame], styles: series.SeriesStyles) -> series.SeriesStyles:
    """Return the comparison style of every labelled series of session summary rows by their provenance columns ``series.ENTITY_COLUMNS``."""
    for label, rows in groups.items():
        _require_provenance(label, rows)
    columns = list(series.ENTITY_COLUMNS)
    return series.select_styles({label: list(rows[columns].itertuples(index=False, name=None)) for label, rows in groups.items()}, styles)


def horizon_order(horizon: Any) -> float:
    """Return the display position of a rollout horizon: numeric horizons ascending, ``full`` last."""
    return math.inf if horizon == "full" else float(horizon)


def dataset_order(material: str, role: str) -> tuple[int, str]:
    """Return the display position of a dataset: its role group (``dataset_selection.EVALUATION_GROUP_ORDER``), then its material's display label."""
    roles = dataset_selection.EVALUATION_GROUP_ORDER
    return roles.index(role) if role in roles else len(roles), display_labels.material_display_label(material)


def _summary_series(rows: pd.DataFrame, run_labels: Mapping[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Group summary rows into one series per frame and material, labelled by material and role, after the run where roles do not separate them.

    The run reads as its label of ``run_labels`` (by frame), else as the frame; material and role come from the rows, never from
    the frame, and equal labels of different frames are told apart by the frame. The series of one run (model kind and
    checkpoint, over all its role frames) stand together in the run's first-appearance order, sorted like the dataset selector:
    by role group (``dataset_selection.EVALUATION_GROUP_ORDER``), then material, so every grouped view and its legend list them alike.
    """
    _require_provenance(", ".join(dict.fromkeys(rows["frame"])), rows)
    include_run = rows["frame"].nunique() > rows["dataset_role"].nunique()
    run_of = dict(zip(rows["frame"], zip(rows["model_kind"], rows["checkpoint_sha256"], strict=True), strict=True))  # a frame is one run's
    runs = list(dict.fromkeys(run_of.values()))
    groups = sorted(
        ((frame, material, group) for (frame, material), group in rows.groupby(["frame", "material_family"], sort=False)),
        key=lambda item: (runs.index(run_of[item[0]]), dataset_order(str(item[1]), str(item[2]["dataset_role"].iloc[0]))),
    )
    labels = [
        (f"{(run_labels or {}).get(str(frame), str(frame))} · " if include_run else "")
        + display_labels.material_role_display_label(str(material), str(group["dataset_role"].iloc[0]))
        for frame, material, group in groups
    ]
    shown = display_labels.unique_display_labels(labels, [(str(frame),) for frame, _material, _group in groups])
    return {label: group for label, (_frame, _material, group) in zip(shown, groups, strict=True)}


def _summary_runs(groups: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """Return the run of every ``_summary_series`` label that names one before its material and role (``_series_runs``)."""
    datasets = {
        label: display_labels.material_role_display_label(str(rows["material_family"].iloc[0]), str(rows["dataset_role"].iloc[0]))
        for label, rows in groups.items()
    }
    return _series_runs(datasets)


def _material_marker(material: str) -> str:
    """Return the fixed marker of one registered material (``visual_semantics.MATERIAL_MARKERS``); an unknown one raises ``ValueError``."""
    marker = visual_semantics.MATERIAL_MARKERS.get(material)
    if marker is None:
        msg = f"Unknown material {material!r}; expected one of {sorted(visual_semantics.MATERIAL_MARKERS)}."
        raise ValueError(msg)
    return marker


def _padded(lower: float, upper: float) -> tuple[float, float]:
    """Return ``lower`` to ``upper`` widened by a twentieth of its span on each side, a degenerate range by 5 % of its value (at least 1)."""
    padding = (upper - lower) * _PAD_FRACTION if upper > lower else max(abs(lower) * _PAD_FRACTION, 1.0)
    return lower - padding, upper + padding


def _series_runs(datasets: Mapping[str, str]) -> dict[str, str]:
    """Return the run of every series label written ``run · dataset`` from the label's dataset (``datasets``); a bare dataset names none."""
    return {label: label.removesuffix(f" · {dataset}") for label, dataset in datasets.items() if label.endswith(f" · {dataset}")}


def _run_entries(handles: Sequence[Any], labels: Sequence[str], runs: Mapping[str, str] | None) -> tuple[list[str], dict[str, list[Any]] | None]:
    """
    Return the legend labels and groups of entries whose series ``runs`` maps to a run (``_series_runs``).

    With several runs, an entry of a series (its label, or its label and `` · `` or `` — `` a statistic) stands under its run
    as a group heading and reads without the run; the other entries follow ungrouped. With one run or none nothing changes.
    """
    if not runs or len(set(runs.values())) < _SEVERAL_RUNS:
        return list(labels), None
    groups: dict[str, list[Any]] = {}
    shown: list[str] = []
    names = sorted(runs, key=len, reverse=True)  # the longest series label that an entry starts with is its series
    for handle, label in zip(handles, labels, strict=True):
        series_label = next((name for name in names if label == name or label.startswith((f"{name} · ", f"{name} — "))), None)
        if series_label is not None:
            groups.setdefault(runs[series_label], []).append(handle)
        shown.append(label if series_label is None else label.removeprefix(f"{runs[series_label]} · "))
    return shown, groups


def _legend(figure: Figure, axes: np.ndarray, runs: Mapping[str, str] | None = None, keys: Mapping[str, Artist] | None = None) -> None:
    """Write one legend of every labelled artist of ``axes`` and then ``keys`` (``figure_layout.add_legend``), grouped by run (``_run_entries``)."""
    handles, labels = figure_layout.legend_entries(axes.flat)
    handles, labels = [*handles, *(keys or {}).values()], [*labels, *(keys or {})]
    if labels:
        shown, groups = _run_entries(handles, labels, runs)
        figure_layout.add_legend(figure, axes, handles, shown, groups=groups)


def _dataset_key(records: Sequence[Any]) -> frozenset[tuple[str, str]]:
    """Return the datasets (material, role) a record group holds, from identity: the runs of one dataset share this key."""
    return frozenset((_record_material(record), str(_value(record, "identity").get("dataset_role"))) for record in records)


def _dataset_label(datasets: Collection[tuple[str, str]]) -> str:
    """Return the display label of one series' datasets (material, role), ``Chickpea, ID``; a series pooling materials reads as its role."""
    roles, materials = sorted({role for _material, role in datasets}), sorted({material for material, _role in datasets})
    if len(roles) != 1:
        msg = f"A series holds one Dataset role, not {roles}."
        raise ValueError(msg)
    if len(materials) == 1:
        return display_labels.material_role_display_label(materials[0], roles[0])
    return display_labels.evaluation_role_display_label(roles[0])


def _cohort_note(texts: Mapping[str, tuple[str, str]]) -> str | None:
    """Return the phase note of series (label: dataset, cohort text): one line per dataset whose series pool alike, else per series."""
    lines: dict[str, str] = {}
    for label, (dataset, text) in texts.items():
        lines[dataset if lines.get(dataset, text) == text else label] = text  # the runs of one dataset share its cases
    return time_ui.phase_note(lines)


def _run_material_legend(figure: Figure, axes: np.ndarray, points: Mapping[str, tuple[str, str, str, Any]]) -> None:
    """
    Write the legend of a point figure whose markers name materials; ``points`` maps a series label to its run, material, role and colour.

    A run whose series share one colour but carry different markers gets one entry, a swatch of its colour without any material
    shape, and then every plotted material a separate key of grey markers, in the dataset selector's order (``dataset_order``, a
    material at its first role); every other series keeps its own entry in its colour and marker.
    """
    runs: dict[str, list[str]] = {}
    places: dict[str, tuple[int, str]] = {}
    for label, (run, material, role, _colour) in points.items():
        runs.setdefault(run, []).append(label)
        places[material] = min(places.get(material, dataset_order(material, role)), dataset_order(material, role))
    entries: dict[str, Artist] = {}
    swatched = False
    for run, labels in runs.items():
        shapes = {points[label][1] for label in labels}
        if len(shapes) > 1 and len({to_hex(points[label][3]) for label in labels}) == 1:
            entries[run] = Patch(facecolor=points[labels[0]][3], edgecolor="none")
            swatched = True
            continue
        for label in labels:
            entries[label] = Line2D([], [], color=points[label][3], marker=_material_marker(points[label][1]), linestyle="none")
    materials = sorted(places, key=places.__getitem__) if swatched else []
    key: dict[str, Artist] = {
        _material_label(material): Line2D([], [], color=_NEUTRAL_COLOUR, marker=_material_marker(material), linestyle="none")
        for material in materials
    }
    if entries:
        groups = {"Run": list(entries.values()), "Material": list(key.values())} if key else None  # by handle: a run may share a material's name
        figure_layout.add_legend(figure, axes, [*entries.values(), *key.values()], [*entries, *key], groups=groups)


def _selected_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Return unique compatible stored and derived states in canonical order."""
    canonical = (*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD)
    requested = set(artifact.STATE_ORDER if fields is None else fields)
    if not requested or requested.difference(canonical):
        msg = "Transient state selection must contain known stored or derived states."
        raise ValueError(msg)
    return tuple(field for field in canonical if field in requested)


def _time_index(record: Any, physical_time: float | None) -> tuple[int, float]:
    """Resolve only an exact stored physical time."""
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        msg = "Transient plot times must be finite and strictly increasing."
        raise ValueError(msg)
    selected = float(times[-1]) if physical_time is None else float(physical_time)
    matches = np.flatnonzero(times == selected)
    if matches.size != 1:
        msg = f"Requested physical time {selected:g} is unavailable; valid times are {times.tolist()}."
        raise ValueError(msg)
    return int(matches[0]), selected


def _state_unit(field: str) -> str:
    """Return the exact physical unit for one persisted or derived state."""
    if field == _GRAIN_MOISTURE_FIELD:
        surface_unit = artifact.STATE_UNITS[artifact.STATE_ORDER.index("w_surf")]
        internal_unit = artifact.STATE_UNITS[artifact.STATE_ORDER.index("w_int")]
        if surface_unit != internal_unit:
            msg = "Surface and internal moisture units must agree for grain moisture."
            raise ValueError(msg)
        return surface_unit
    try:
        return artifact.STATE_UNITS[artifact.STATE_ORDER.index(field)]
    except ValueError as error:
        msg = f"Unknown transient state field {field!r}."
        raise ValueError(msg) from error


def _display_state_values(
    field: str,
    values: np.ndarray,
    *,
    error: bool,
) -> np.ndarray:
    """Apply the shared EDA display conversion without changing artifact arrays."""
    return field_labels.display_values(
        values,
        _state_unit(field),
        field=field,
        quantity_kind="difference" if error else "absolute",
    )


def _absolute_display(record: Any, field: str, raw: np.ndarray) -> np.ndarray:
    """Return displayed absolute values: wet-basis % with masked unphysical cells for water densities, else the EDA conversion."""
    if moisture_display.is_moisture_field(field):
        return moisture_display.wet_basis_percent(raw, _dry_bulk_density(record))
    return _display_state_values(field, raw, error=False)


def _difference_display(field: str, values: np.ndarray) -> np.ndarray:
    """Return displayed differences; wet-basis differences are already in percentage points."""
    return values if moisture_display.is_moisture_field(field) else _display_state_values(field, values, error=True)


def _display_unit(field: str, *, difference: bool) -> str:
    """Return the displayed unit of one state's values or differences."""
    return moisture_display.field_display_unit(field, _state_unit(field), difference=difference)


def _display_symbol(field: str) -> str:
    """Return the label of one displayed state in the active style (water as wet-basis moisture)."""
    return moisture_display.display_name(field, mathtext=True)


def _display_label(field: str) -> str:
    """Return the label of one displayed state in the active style with its absolute unit."""
    return f"{_display_symbol(field)} [{_display_unit(field, difference=False)}]"


def _spatial_means(record: Any, field: str, reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return displayed per-time spatial means [time] of reference and prediction [time,Y,X].

    Wet-basis moisture is the bulk value 100 Σw/(Σρ_bu,dry + Σw) over the cells valid in both at that time, so the two
    lines share one cell set; other states average all cells.
    """
    if not moisture_display.is_moisture_field(field):
        with np.errstate(invalid="ignore", over="ignore"):
            return (
                np.mean(_display_state_values(field, reference, error=False), axis=(-2, -1)),
                np.mean(_display_state_values(field, prediction, error=False), axis=(-2, -1)),
            )
    density = _dry_bulk_density(record)
    valid = np.isfinite(_absolute_display(record, field, reference)) & np.isfinite(_absolute_display(record, field, prediction))
    reference_means, prediction_means = (
        np.asarray([moisture_display.bulk_wet_basis_percent(water, density, cells) for water, cells in zip(values, valid, strict=True)])
        for values in (reference, prediction)
    )
    return reference_means, prediction_means


def _voided(means: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the per-time ``means`` void at every time at which ``values`` [time,Y,X] hold a raw non-finite value; negative water is no gap."""
    return np.where(np.isfinite(values).all(axis=(-2, -1)), means, np.nan)


def _reference_means(record: Any, field: str, reference: np.ndarray) -> np.ndarray:
    """
    Return the displayed per-time spatial means [time] of the reference alone, independent of any prediction.

    Wet-basis moisture is the bulk value over the reference's own valid cells, so the runs of one cohort share this curve
    exactly; a time with a raw non-finite reference value is void (``_voided``).
    """
    if not moisture_display.is_moisture_field(field):
        with np.errstate(invalid="ignore", over="ignore"):
            return np.mean(_display_state_values(field, reference, error=False), axis=(-2, -1))
    density = _dry_bulk_density(record)
    valid = np.isfinite(_absolute_display(record, field, reference))
    return _voided(
        np.asarray([moisture_display.bulk_wet_basis_percent(water, density, cells) for water, cells in zip(reference, valid, strict=True)]),
        reference,
    )


def _grain_moisture(record: Any, states: np.ndarray) -> np.ndarray:
    """Derive grain moisture from persisted states and material evidence, unphysical or non-finite water passed through as stored."""
    if states.ndim != _STATE_ARRAY_RANK or states.shape[1] != len(artifact.STATE_ORDER):
        msg = "Grain-moisture trajectories require [time,4,Y,X] state arrays."
        raise ValueError(msg)
    scalars = np.asarray(_value(record, "scalar_conditioning"), dtype=np.float64)
    if scalars.shape != (len(artifact.SCALAR_ORDER),):
        msg = "Grain-moisture trajectories require exact scalar conditioning."
        raise ValueError(msg)
    fraction = scalars[artifact.SCALAR_ORDER.index("f_surf")]
    return domain.moisture.granular_water_content(
        states[:, artifact.STATE_ORDER.index("w_surf")],
        states[:, artifact.STATE_ORDER.index("w_int")],
        fraction,
        require_physical_states=False,
    )


def _field_snapshots(record: Any, field: str, reference: np.ndarray, prediction: np.ndarray, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return raw reference and prediction values [time,Y,X] of one field at the given time indices."""
    selected = list(indices)
    if field == _GRAIN_MOISTURE_FIELD:
        return _grain_moisture(record, reference[selected]), _grain_moisture(record, prediction[selected])
    channel = artifact.STATE_ORDER.index(field)
    return reference[selected, channel], prediction[selected, channel]


def _state_arrays(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the validated reference and prediction state arrays of one record."""
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK or reference.shape[1] != len(artifact.STATE_ORDER):
        msg = "Transient state maps require matching [time,4,Y,X] arrays."
        raise ValueError(msg)
    return reference, prediction


def _selected_error(
    record: Any,
    field: str,
    mode: str,
    raw_reference: np.ndarray,
    raw_prediction: np.ndarray,
    scale: error_diagnostics.ReferenceScale | str | None,
) -> error_diagnostics.SelectedErrorMap:
    """
    Compute one displayed error map over the full grid, with diverging prediction cells masked.

    A signed or absolute water-density error is the difference of the wet-basis fields in pp, masked where either is
    unphysical; the normalized error stays on the stored states, so it needs no ``record``.
    """
    reference, prediction, unit = raw_reference, raw_prediction, _state_unit(field)
    finite = np.isfinite(raw_prediction)
    if moisture_display.is_moisture_field(field) and mode != error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        reference, prediction = _absolute_display(record, field, raw_reference), _absolute_display(record, field, raw_prediction)
        unit = moisture_display.ABSOLUTE_UNIT
        finite = np.isfinite(reference) & np.isfinite(prediction)
    # A fully diverging field keeps the whole grid, so the shared map reports it as non-finite.
    support = finite if finite.any() else np.ones(raw_prediction.shape, dtype=bool)
    return error_diagnostics.selected_error_map(
        reference,
        prediction,
        mode=mode,  # type: ignore[arg-type]
        field=field,
        physical_unit=unit,
        scale=scale,
        support=support,
    )


def _error_display(field: str, error_mode: str, error_result: error_diagnostics.SelectedErrorMap) -> np.ndarray | None:
    """Return the displayed error values of one selected error map."""
    if error_result.map_values is None or error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        return error_result.map_values
    return _difference_display(field, error_result.map_values)


def _error_definition(field: str, error_mode: str, finite_errors: np.ndarray) -> field_maps.FieldMapDefinition:
    """Build one field's error colour scale from its finite displayed errors; errors beyond it set the bar's arrows."""
    signed = error_mode == error_diagnostics.SIGNED_ERROR
    normalized = error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    boundaries = field_maps.error_boundaries(finite_errors, signed=signed)
    return field_maps.build_field_map(
        # The scale's own ends keep a uniform error (all zero included) from being a constant field, whose map refuses locked boundaries.
        np.append(finite_errors, boundaries[[0, -1]]),
        semantic=field_maps.FieldMapSemantic.SIGNED_ERROR if signed else field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
        field=field,
        unit="1" if normalized else _display_unit(field, difference=True),
        display_transform=error_mode,
        locked_boundaries=boundaries,
        bound_policy="rounded 5th to 95th percentile, symmetric" if signed else "0 to the rounded 95th percentile",
    )


@dataclass(frozen=True, slots=True)
class StateMapNorm:
    """
    One field's error colour scale, state tick format and state-bar arrows, fixed over the scaled times; no error scale without finite errors.

    ``arrows`` are the (reference, prediction) bar's arrows: an end is drawn when any scaled time has values beyond it.
    ``fixed`` is the (reference, prediction) state scale pooled over all times, set only for a field of ``fixed_time_fields``.
    """

    error: field_maps.FieldMapDefinition | None
    state_format: layout.TickFormat
    arrows: tuple[field_maps.FieldMapExtend, field_maps.FieldMapExtend]
    fixed: tuple[Normalize | None, Normalize | None] | None = None


def _is_constant(norm: Normalize) -> bool:
    """Return whether ``norm`` is the padded scale of a constant field."""
    lower, upper = np.asarray((norm.vmin, norm.vmax), dtype=np.float64)
    return bool(upper - lower <= _CONSTANT_SPAN * max(abs(lower), 1.0))


def _state_ticks(norm: Normalize) -> np.ndarray:
    """Return the ticks of a state bar: its band boundaries; one tick at the value of a constant field."""
    if isinstance(norm, BoundaryNorm):
        return np.asarray(norm.boundaries, dtype=np.float64)
    lower, upper = np.asarray((norm.vmin, norm.vmax), dtype=np.float64)
    return np.array([(lower + upper) / 2.0])


def _fixed_fields(fields: Sequence[str], fixed_time_fields: Collection[str]) -> frozenset[str]:
    """Return ``fixed_time_fields`` as a set of drawn ``fields``; a bare string or a name that is not drawn is refused."""
    if isinstance(fixed_time_fields, str):
        msg = "fixed_time_fields must be a collection of field names, not one string."
        raise TypeError(msg)
    fixed = frozenset(fixed_time_fields)
    if undrawn := fixed.difference(fields):
        msg = f"Fixed-over-time fields must be drawn fields, not {sorted(undrawn)}."
        raise ValueError(msg)
    return fixed


def default_fixed_time_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Return the ``fields`` whose state scale is fixed over all stored times by default: all but the moisture fields, which follow each time."""
    return tuple(field for field in fields if not moisture_display.is_moisture_field(field))


def state_map_norms(
    record: Any,
    *,
    state_fields: Sequence[str] | None,
    time_indices: Sequence[int],
    error_mode: str,
    reference_scales: Mapping[str, error_diagnostics.ReferenceScale | str] | None = None,
    lock_scale: bool = True,
    fixed_time_fields: Collection[str] = (),
) -> dict[str, StateMapNorm]:
    """
    Return per-field error colour scales and state tick decimals from all given time indices.

    The state colour scales are not fixed here unless the field is in ``fixed_time_fields``: ``plot_state_maps`` derives
    them from each drawn time. Their tick decimals are the most any time's ticks need, at most three (``layout.state_format``);
    a near-uniform time does not set them but is widened (``lock_scale`` as in ``plot_state_maps``). A state bar takes its
    power of ten from the size of its values (``field_maps.state_exponent``); the transient states lie where that is 0, and
    the transient bars carry none, so a field whose state bands would need one raises ``ValueError`` naming it. A fixed
    field's state scale is the one of the 5th to 95th percentile of the displayed values pooled over all given times (the
    prediction's own without ``lock_scale``), and its decimals come from that scale.
    A name in ``fixed_time_fields`` that is not a drawn field raises ``ValueError``, a bare string ``TypeError``.
    """
    fields = _selected_fields(state_fields)
    fixed_fields = _fixed_fields(fields, fixed_time_fields)
    indices = tuple(time_indices)
    if not indices:
        msg = "State-map colour scales require at least one time index."
        raise ValueError(msg)
    if error_mode not in error_diagnostics.TRANSIENT_ERROR_MODES:
        msg = f"Transient error mode must be one of {error_diagnostics.TRANSIENT_ERROR_MODES}."
        raise ValueError(msg)
    reference, prediction = _state_arrays(record)
    norms: dict[str, StateMapNorm] = {}
    for field in fields:
        raw_reference, raw_prediction = _field_snapshots(record, field, reference, prediction, indices)
        scale = None if reference_scales is None else reference_scales.get(field)
        errors = [
            values[np.isfinite(values)]
            for position in range(len(indices))
            if (
                values := _error_display(
                    field, error_mode, _selected_error(record, field, error_mode, raw_reference[position], raw_prediction[position], scale)
                )
            )
            is not None
        ]
        finite_errors = np.concatenate(errors) if errors else np.empty(0)
        frames_reference = [_absolute_display(record, field, raw) for raw in raw_reference]
        frames_prediction = [_absolute_display(record, field, raw) for raw in raw_prediction]
        shown_reference, shown_prediction = frames_reference, frames_prediction
        pooled = field in fixed_fields
        if pooled:  # one pooled "frame" per panel kind
            shown_reference, shown_prediction = [np.concatenate(frames_reference)], [np.concatenate(frames_prediction)]
        steps = []
        for shown in shown_reference if lock_scale else (*shown_reference, *shown_prediction):
            state_range = _frame_range(shown)
            if state_range is None:  # a constant frame keeps one tick and never sets the format
                continue
            # A near-uniform frame records the finer step it needs: it never sets the format, but keeps three decimals if no frame fits.
            bands = field_maps.rounded_boundaries(*state_range, max_decimals=field_maps.MAX_TICK_DECIMALS)
            bands = field_maps.rounded_boundaries(*state_range) if bands is None else bands
            if bands is not None:
                steps.append(float(bands[1] - bands[0]))
        state_format = layout.state_format(steps)
        fixed = _state_norms(field, shown_reference[0], shown_prediction[0], shared=lock_scale, decimals=state_format.decimals) if pooled else None
        frame_norms = [
            fixed or _state_norms(field, shown, other, shared=lock_scale, decimals=state_format.decimals)
            for shown, other in zip(frames_reference, frames_prediction, strict=True)
        ]
        norms[field] = StateMapNorm(
            _error_definition(field, error_mode, finite_errors) if finite_errors.size else None,
            state_format,
            (_union_extend(frames_reference, frame_norms, 0), _union_extend(frames_prediction, frame_norms, 1)),
            fixed,
        )
    return norms


def _union_extend(
    frames: Sequence[np.ndarray], frame_norms: Sequence[tuple[Normalize | None, Normalize | None]], column: int
) -> field_maps.FieldMapExtend:
    """Return the arrows of one state bar over all frames: an end is drawn when any frame has finite values beyond that frame's scale."""
    extends = [
        field_maps.locked_extend(values[np.isfinite(values)], np.array([norm.vmin, norm.vmax]))
        for values, norms in zip(frames, frame_norms, strict=True)
        if (norm := norms[column]) is not None
    ]
    below, above = (any(extend in ends for extend in extends) for ends in (("min", "both"), ("max", "both")))
    return "both" if below and above else "min" if below else "max" if above else "neither"


def _frame_range(values: np.ndarray) -> tuple[float, float] | None:
    """Return the 5th to 95th percentile of the finite ``values``, else their full range; ``None`` without finite values or when constant."""
    scale = field_maps.percentile_scale(values, "state")
    if scale is None or scale[1] > scale[0]:
        return scale
    full = layout.linear_norm(values[np.isfinite(values)])
    return None if _is_constant(full) else (float(np.asarray(full.vmin)), float(np.asarray(full.vmax)))


def _state_norm(field: str, values: np.ndarray, decimals: int) -> Normalize | None:
    """
    Return the discrete scale of ``field`` over the 5th to the 95th percentile of the finite ``values`` (``None`` without any).

    The bands are the round ones of ``field_maps.state_boundaries``, with a ``BoundaryNorm`` of one colour per band.
    A degenerate percentile range falls back to the values' own full finite range, and a constant field to a padded
    continuous scale of one tick. A near-uniform scale whose step is finer than the bar's ``decimals`` is widened to
    11 steps of 10**-``decimals``. The bar is written without a power of ten, so a field whose state bands (a constant:
    its value) or whose widened bands have a nonzero ``field_maps.state_exponent`` (largest magnitude outside 10^-3 to
    10^4) raises ``ValueError``.
    """
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    scale = _frame_range(values)
    banded = np.array([finite.min(), finite.max()]) if scale is None else field_maps.state_boundaries(*scale)
    boundaries = banded
    if scale is not None and layout.is_finer_than_decimals(float(banded[1] - banded[0]), decimals):
        boundaries = np.linspace(*field_maps.widened_range(*scale, decimals), field_maps.N_COLOR_BINS + 1)
    if any(field_maps.state_exponent(float(drawn[0]), float(drawn[-1])) for drawn in (banded, boundaries)):
        msg = f"The transient state {field!r} spans {boundaries[0]!r} to {boundaries[-1]!r}: its bar would need a power of ten it never carries."
        raise ValueError(msg)
    return layout.linear_norm(finite) if scale is None else BoundaryNorm(boundaries, boundaries.size - 1)


def _state_norms(
    field: str, reference: np.ndarray, prediction: np.ndarray, *, shared: bool, decimals: int
) -> tuple[Normalize | None, Normalize | None]:
    """
    Return one time's reference and prediction colour scales (``None`` for a panel without finite values).

    Shared: both panels take the reference's percentile scale, so one colour means one value and the prediction never
    widens it; a reference without finite values leaves each panel its own scale. Otherwise each panel scales to its own
    5th to 95th percentile.
    """
    if shared and (together := _state_norm(field, reference, decimals)) is not None:
        return together, together
    return _state_norm(field, reference, decimals), _state_norm(field, prediction, decimals)


def _static_conditioning(record: Any) -> np.ndarray:
    """Return one record's exact static conditioning [7,Y,X]."""
    static = np.asarray(_value(record, "static_conditioning"), dtype=np.float64)
    if static.shape != (len(artifact.STATIC_ORDER), *np.shape(_value(record, "reference_states"))[-2:]):
        msg = "Transient state maps require exact [7,Y,X] static conditioning."
        raise ValueError(msg)
    return static


def _static_coordinates(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y grid coordinates [Y,X] of one record's exact static conditioning."""
    static = _static_conditioning(record)
    return static[0], static[1]


def _dry_bulk_density(record: Any) -> np.ndarray:
    """Return one record's dry bulk density [Y,X] from its exact static conditioning."""
    return _static_conditioning(record)[artifact.STATIC_ORDER.index("rho_bu_dry")]


def _map_extent(record: Any) -> tuple[float, float, float, float]:
    """Return the physical extent (x0, x1, y0, y1) spanned by the grid coordinates of one record."""
    x_values, y_values = _static_coordinates(record)
    return float(np.min(x_values)), float(np.max(x_values)), float(np.min(y_values)), float(np.max(y_values))


def _fixed_width_time(record: Any, selected_time: float) -> str:
    """Return ``selected_time`` with the decimals and width of the record's widest stored time, padded with figure spaces.

    Digits and figure spaces are equally wide, so a title carrying this text keeps its width over all animation frames.
    """
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    decimals = max(len(np.format_float_positional(value, precision=6, trim="-").partition(".")[2]) for value in times)
    width = max(len(f"{value:.{decimals}f}") for value in times)
    return f"{selected_time:.{decimals}f}".rjust(width, "\N{FIGURE SPACE}")


def state_map_geometry(record: Any, *, state_fields: Sequence[str] | None = None) -> layout.MapGeometry:
    """Return the fixed map-grid geometry that ``plot_state_maps`` draws for one record and state selection."""
    x0, x1, y0, y1 = _map_extent(record)
    return layout.plan_geometry(len(_selected_fields(state_fields)), _STATE_MAP_COLUMNS, (x1 - x0, y1 - y0))


def plot_state_maps(
    record: Any,
    *,
    state_fields: Sequence[str] | None = None,
    physical_time: float | None = None,
    lock_scale: bool = True,
    fixed_time_fields: Collection[str] = (),
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    reference_scales: Mapping[str, error_diagnostics.ReferenceScale | str] | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
    norms: Mapping[str, StateMapNorm] | None = None,
    figure: Figure | None = None,
    title_context: str | None = None,
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot reference, prediction, and one selected full-grid error mode on the fixed map-grid geometry.

    Reference and prediction follow this time: with ``lock_scale`` both share the 5th to 95th percentile of the
    reference's finite displayed values, else each its own; a bar's arrows mark the cells beyond it. ``norms`` (from ``state_map_norms``)
    fixes the error scales, the state tick decimals and the state bars' arrows (drawn when any scaled time has values beyond that
    end) over several times; the default derives them from this time alone.
    A drawn field of ``fixed_time_fields`` has the state scale pooled over all stored times instead of this time's; any
    such field makes the default norms span all stored times, and given ``norms`` must cover every drawn field and carry
    exactly the fixed fields' pooled scales (``state_map_norms`` with the same fields), else ``ValueError``. Names in
    ``fixed_time_fields`` must be drawn fields, as in ``state_map_norms``.
    Only the error maps carry a metric, one mean |e| line above each; one note line under the title names the colour-scale
    rule. A given ``figure`` (from ``layout.map_figure`` with
    ``state_map_geometry``) is cleared and reused so animation frames keep one size and identical
    axes positions. ``title_context`` (the case) replaces the generic figure title before the
    time, which is then written at the fixed width of the record's stored times; the short labels
    ``models`` of the compared run end the title (``layout.models_suffix``), after the time. Panels without
    finite values state that instead of raising.
    """
    fields = _selected_fields(state_fields)
    if not isinstance(lock_scale, bool):
        msg = "lock_scale must be boolean."
        raise TypeError(msg)
    fixed_fields = _fixed_fields(fields, fixed_time_fields)
    index, selected_time = _time_index(record, physical_time)
    reference, prediction = _state_arrays(record)
    x_values, y_values = _static_coordinates(record)
    if norms is None:
        norms = state_map_norms(
            record,
            state_fields=fields,
            time_indices=range(reference.shape[0]) if fixed_fields else (index,),
            error_mode=error_mode,
            reference_scales=reference_scales,
            lock_scale=lock_scale,
            fixed_time_fields=fixed_fields,
        )
    elif missing := set(fields).difference(norms):
        msg = f"State-map colour scales lack the drawn fields {sorted(missing)}."
        raise ValueError(msg)
    elif {field for field in fields if norms[field].fixed is not None} != fixed_fields:
        msg = "State-map colour scales do not fix the requested fixed_time_fields."
        raise ValueError(msg)
    indices = display_grid.display_indices(reference.shape[-2:], display_spatial_stride)
    x_display = display_grid.display_sample(x_values, indices)
    y_display = display_grid.display_sample(y_values, indices)
    geometry = state_map_geometry(record, state_fields=fields)
    figure, axes, slots = layout.map_grid(geometry, extent=_map_extent(record), figure=figure)
    error_texts = []
    normalized_error = error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    signed_error = error_mode == error_diagnostics.SIGNED_ERROR
    for row, field in enumerate(fields):
        snapshot_reference, snapshot_prediction = _field_snapshots(record, field, reference, prediction, (index,))
        raw_reference, raw_prediction = snapshot_reference[0], snapshot_prediction[0]
        scale = None if reference_scales is None else reference_scales.get(field)
        error_result = _selected_error(record, field, error_mode, raw_reference, raw_prediction, scale)
        norm = norms[field]
        values = (
            _absolute_display(record, field, raw_reference),
            _absolute_display(record, field, raw_prediction),
            _error_display(field, error_mode, error_result),
        )
        exponent = layout.error_format(None if norm.error is None else norm.error.boundaries, signed=signed_error).exponent
        error_unit = None if normalized_error else _display_unit(field, difference=True)
        error_texts.append(plot_layout.mean_error_text(values[2], unit=error_unit, exponent=exponent))
        masked_note = "Negative or non-finite water masked" if moisture_display.is_moisture_field(field) else "Non-finite cells masked"
        reference_norm, prediction_norm = (
            _state_norms(field, values[0], values[1], shared=lock_scale, decimals=norm.state_format.decimals) if norm.fixed is None else norm.fixed
        )
        for column, value in enumerate(values):
            axis = axes[row, column]
            is_error = column == _SPATIAL_MAP_ERROR_COLUMN
            error_scale = norm.error if is_error else None
            colormap_name = visual_semantics.field_visual_semantics(field).colormap
            colormap: Colormap | str = colormap_name
            normalizer: Normalize | None = None if is_error else reference_norm if column == 0 else prediction_norm
            if isinstance(normalizer, BoundaryNorm):  # one colour per band of this frame's scale
                colormap = colormaps[colormap_name].resampled(normalizer.boundaries.size - 1)
            if error_scale is not None:
                # Errors beyond the robust scale stand out in one colour of their own instead of the darkest scale colour.
                colormap = error_scale.colormap.with_extremes(over=_BEYOND_SCALE_COLOUR, under=_BEYOND_SCALE_COLOUR)
                normalizer = error_scale.normalizer
            image: ScalarMappable
            if value is None or not np.isfinite(value).any():
                reason = "unavailable: no finite values" if value is not None else f"Unavailable: {error_result.reason}"
                axis.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, transform=axis.transAxes)
                if normalizer is None:
                    continue
                # An error scale fixed over several frames keeps its bar in a frame without finite values.
                image = ScalarMappable(norm=normalizer, cmap=colormap)
            else:
                if is_error and error_scale is None:
                    msg = "State-map colour scales do not cover this time."
                    raise ValueError(msg)
                image = axis.pcolormesh(
                    x_display,
                    y_display,
                    np.ma.masked_invalid(display_grid.display_sample(np.asarray(value), indices)),
                    shading="auto",
                    cmap=colormap,
                    norm=normalizer,
                )
                if not is_error and not np.isfinite(value).all():  # in the note line above its map, never over the data
                    layout.add_map_note(axis, masked_note)
            unit = f"[{_display_unit(field, difference=is_error)}]"
            slot = slots[row, column]
            if error_scale is None:
                extend: field_maps.FieldMapExtend = "neither" if normalizer is None else norm.arrows[column]
                ticks = np.empty(0) if normalizer is None else _state_ticks(normalizer)
                layout.set_fixed_decimal_ticks(layout.add_slot_colorbar(figure, image, slot, label=unit, extend=extend), ticks, norm.state_format)
                continue
            colorbar = layout.add_slot_colorbar(
                figure,
                image,
                slot,
                label=f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} [{field_labels.display_unit('1')}]" if normalized_error else unit,
                extend=error_scale.extend,
            )
            layout.set_fixed_decimal_ticks(colorbar, layout.error_ticks(error_scale.boundaries, signed=signed_error))
    plot_layout.set_error_headers(axes, ("Reference", "Prediction", _ERROR_HEADERS[error_mode]), error_texts)
    layout.set_row_labels(axes, [_display_symbol(field) for field in fields])
    title = (
        f"Transient state comparison at t={selected_time:g} h"
        if title_context is None
        else f"{title_context} · t = {_fixed_width_time(record, selected_time)} h"
    )
    layout.set_map_title(figure, title + plot_layout.models_suffix(models))
    plot_layout.add_scale_note(figure, shared=lock_scale)
    return figure


def _trajectory_values(
    record: Any,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact time, reference, and prediction values of one field on the full grid."""
    if field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Transient trajectory field is unsupported: {field!r}."
        raise ValueError(msg)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim != _STATE_ARRAY_RANK
        or times.shape != (reference.shape[0],)
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0.0)
    ):
        msg = "Transient trajectories require aligned state arrays and physical times."
        raise ValueError(msg)
    if field == _GRAIN_MOISTURE_FIELD:
        reference_values = _grain_moisture(record, reference)
        prediction_values = _grain_moisture(record, prediction)
    else:
        channel = artifact.STATE_ORDER.index(field)
        reference_values = reference[:, channel]
        prediction_values = prediction[:, channel]
    return times, reference_values, prediction_values


def _record_end_hours(records: Sequence[Any], *, exclude_origin: bool) -> np.ndarray:
    """
    Return the end of every record: its last stored time, at most one interval before its exact stop.

    The last stored time, not the exact stop, keeps the offsets before the end on the shared time grid, so cases still aggregate. The
    stored times are validated by the shared time owner; with ``exclude_origin`` every record must hold a time after its origin, so
    none silently leaves the cohort.
    """
    admitted = tuple(records)
    if not admitted:
        msg = "Transient trajectory summaries require at least one record."
        raise ValueError(msg)
    grids = [time_ui.validated_times(_value(record, "physical_times")) for record in admitted]
    short = [str(_value(record, "case_id")) for record, grid in zip(admitted, grids, strict=True) if len(grid) < _MINIMUM_TRAJECTORY_TIMES]
    if exclude_origin and short:
        msg = f"Transient records of cases {short} hold no stored time after its origin; a case is refused, not dropped from the cohort."
        raise ValueError(msg)
    return np.asarray([grid[-1] for grid in grids], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ExactTimeSummary:
    """One exact-time cohort summary with separate presence and invalid counts."""

    times: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    count: np.ndarray
    invalid_count: np.ndarray


def _exact_time_groups(rows: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, tuple[tuple[tuple[int, float], ...], ...]]:
    """Return the sorted present exact times with the ``(row, value)`` contributions at each."""
    by_time: dict[float, list[tuple[int, float]]] = {}
    for row, (times, values) in enumerate(rows):
        if times.ndim != 1 or values.shape != times.shape or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
            msg = "Exact-time aggregate rows require aligned strictly increasing times and values."
            raise ValueError(msg)
        for time, value in zip(times, values, strict=True):
            by_time.setdefault(float(time), []).append((row, float(value)))
    if not by_time:
        msg = "Exact-time aggregate requires at least one supported physical time."
        raise ValueError(msg)
    times = np.asarray(sorted(by_time), dtype=np.float64)
    return times, tuple(tuple(by_time[float(time)]) for time in times)


def _aggregate_exact_time_values(rows: Sequence[tuple[np.ndarray, np.ndarray]]) -> _ExactTimeSummary:
    """Average only present exact times; one nonfinite contributor voids that score."""
    times, groups = _exact_time_groups(rows)
    mean, lower, upper = (np.full(times.shape, np.nan, dtype=np.float64) for _ in range(3))
    count = np.asarray([len(group) for group in groups], dtype=np.int64)
    invalid = np.asarray([sum(not math.isfinite(value) for _row, value in group) for group in groups], dtype=np.int64)
    for index, group in enumerate(groups):
        if invalid[index]:
            continue
        values = np.asarray([value for _row, value in group], dtype=np.float64)
        mean[index] = float(np.mean(values))
        if values.size >= _MIN_BAND_CASES:
            lower[index], upper[index] = np.quantile(values, (0.1, 0.9))
    return _ExactTimeSummary(times, mean, lower, upper, count, invalid)


def _aggregate_exact_time_errors(rows: Sequence[tuple[np.ndarray, np.ndarray]], *, materials: Sequence[str]) -> _ExactTimeSummary:
    """
    Reduce per-case mean squared errors at every exact time like every reported field RMSE.

    The central value is the root of :func:`learning.metrics.metrics.material_case_mean` (equal
    over a material's cases, then over the materials); the band holds the 10th and 90th
    quantiles of the per-case RMSEs. One nonfinite contributor voids that time.
    """
    times, groups = _exact_time_groups(rows)
    central, lower, upper = (np.full(times.shape, np.nan, dtype=np.float64) for _ in range(3))
    count = np.asarray([len(group) for group in groups], dtype=np.int64)
    invalid = np.asarray([sum(not math.isfinite(value) for _row, value in group) for group in groups], dtype=np.int64)
    for index, group in enumerate(groups):
        if invalid[index]:
            continue
        mean_squared = learning.metrics.metrics.material_case_mean((materials[row], value) for row, value in group)
        central[index] = math.nan if mean_squared is None else math.sqrt(mean_squared)
        if len(group) >= _MIN_BAND_CASES:
            lower[index], upper[index] = np.quantile(np.sqrt([value for _row, value in group]), (0.1, 0.9))
    return _ExactTimeSummary(times, central, lower, upper, count, invalid)


def _shared_windows(
    admitted: Mapping[str, tuple[Any, ...]], *, exclude_origin: bool, phase: bool
) -> tuple[tuple[time_ui.TimeWindow, ...], dict[str, np.ndarray]]:
    """
    Return the time windows of all contexts and the final hour of every record, per context.

    ``phase`` gives the one drying-phase window; otherwise cases whose ends (last stored times) are equal share the clock, and
    as soon as any two ends differ at all, as in the Dataset EDA, the last window counts hours before each case's own end.
    """
    ends = {label: _record_end_hours(records, exclude_origin=exclude_origin) for label, records in admitted.items()}
    if phase:
        return (time_ui.PHASE_WINDOW,), ends
    longest = max(float(values.max()) for values in ends.values())
    shortest = min(float(values.min()) for values in ends.values())
    return time_ui.drying_windows(longest, before_end=longest != shortest), ends


def _window_rows(window: time_ui.TimeWindow, rows: Sequence[tuple[np.ndarray, np.ndarray]], ends: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return every row's exact points inside the window on its coordinate; a row keeps its place when none of its points lies inside."""
    shown = []
    for (times, values), end in zip(rows, ends, strict=True):
        inside = window.select(times, end_hours=float(end))
        shown.append((window.hours(times[inside], end_hours=float(end)), values[inside]))
    return shown


def _phase_cohort(records: Sequence[Any]) -> time_ui.PhaseCohort:
    """Return the phase cohort of records over all their stored times, each classified by its reference completion."""
    completion: list[time_ui.Completion] = [transient_session.reference_completion(record) for record in records]
    return time_ui.phase_cohort([_value(record, "physical_times") for record in records], completion)


def _phase_rows(cohort: time_ui.PhaseCohort, rows: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return each pooled case's values at its phase states, on the phase grid, in the exact-time row form."""
    return [(time_ui.PHASE_GRID, values) for values in cohort.take([values for _times, values in rows])]


def _window_figure(windows: Sequence[time_ui.TimeWindow], heights: Sequence[float]) -> tuple[Figure, np.ndarray]:
    """
    Return a figure on the shared panel engine and its axes ``[row, window]``, a row ``heights`` of a full row high.

    The axes take the shared text sizes; the legend grows the figure later. The engine's gaps are paddings in inches only (no
    figure fraction), so a figure grown for its legend keeps every panel's size.
    """
    width = max(_WINDOW_PANEL_WIDTH_IN * len(windows), _WINDOW_PANELS_MIN_IN)  # one phase window gets the room of two
    figure = plt.figure(figsize=(width, sum(heights) * _WINDOW_ROW_HEIGHT_IN), layout=figure_layout.panel_engine())
    axes = layout.window_axes(figure, figure.add_gridspec(1, 1)[0, 0], windows, heights)
    figure_layout.apply_text_sizes(axes.flat)
    return figure, axes


def _count_symbol(windows: Sequence[time_ui.TimeWindow]) -> str:
    """Return the symbol of the case count over the windows' time: n(τ) over normalized time, else n(t)."""
    return "n(τ)" if windows[0].unit == "1" else "n(t)"


def _draw_case_count(axis: Axes, time: np.ndarray, count: np.ndarray, *, style: visual_semantics.SeriesStyle, index: int = 0, total: int = 1) -> None:
    """
    Draw the cases that have a stored time at each exact time (or τ bin) in the series style; every field has the same count.

    The strip's own y label names the count, so the step takes no legend entry beside the series' lines. Series ``index`` of
    ``total`` carries its marker (a circle without one), sparse and staggered (``_staggered``), so equal counts stay apart.
    """
    start, step = _staggered(index, total, time.size)
    marker = style.marker or "o"
    axis.step(
        time, count, where="mid", color=style.color, linestyle=style.linestyle, alpha=0.55, marker=marker, markevery=(start, step), markersize=4
    )


def _whisker_positions(bands: Sequence[_ExactTimeSummary]) -> list[np.ndarray]:
    """
    Return, per 10th-90th band of one panel, the indices of its own times that carry a whisker (``_draw_band``).

    The slots lie on the panel's common time grid, the union of the bands' times: band k of n takes every step-th grid time from
    an offset of its own (``_staggered``), so the bands stand apart also where one of them lacks a time, at most one whisker per
    slot. A band draws at each slot where its bounds are finite, every band's own slots first; at another slot it takes the
    nearest finite time of its own within half a step that no whisker holds, if there is one. A band that no slot gives a
    whisker, as one finite at one time only, keeps one: at a free finite time where there is one, else at a time whose whisker
    its band gives up for another it keeps; two whiskers share a time only where neither band has another.
    """
    if not bands:
        return []
    grid = np.unique(np.concatenate([band.times for band in bands]))
    finite = [set(np.searchsorted(grid, band.times[np.isfinite(band.lower) & np.isfinite(band.upper)]).tolist()) for band in bands]
    step = _staggered(0, len(bands), grid.size)[1]  # the same for every band
    open_slots = [(band, slot) for band in range(len(bands)) for slot in range(_staggered(band, len(bands), grid.size)[0], grid.size, step)]
    held: list[set[int]] = [set() for _band in bands]
    taken: set[int] = set()
    for distance in range(step // 2 + 1):  # a slot's own time first, then times ever farther from it
        for band, slot in list(open_slots):
            if free := [position for position in (slot - distance, slot + distance) if position in finite[band] and position not in taken]:
                held[band].add(free[0])
                taken.add(free[0])
                open_slots.remove((band, slot))
    for band, positions in enumerate(finite):
        if positions and not held[band]:
            owner = {position: other for other, kept in enumerate(held) for position in kept}
            spare = [position for position in sorted(positions) if position not in owner or len(held[owner[position]]) > 1]
            position = min(spare, key=lambda position: position in owner) if spare else min(positions)
            if position in owner and len(held[owner[position]]) > 1:
                held[owner[position]].discard(position)  # its band keeps another whisker
            held[band].add(position)
            taken.add(position)
    return [np.searchsorted(band.times, grid[sorted(positions)]) for band, positions in zip(bands, held, strict=True)]


def _draw_band(axis: Axes, summary: _ExactTimeSummary, place: np.ndarray, *, color: Any, reference: bool = False) -> bool:
    """
    Draw the 10th-90th band of ``summary`` as whiskers at its times ``place`` (``_whisker_positions``); return whether it drew one.

    A whisker spans both bounds; one of zero width, where the cases agree, would show nothing under its key, so a short vertical
    stroke centred on that bound stands for it across its curve. A reference's whiskers are wide and pale like its curve, a
    prediction's thin; a whisker takes no line style, so it never reads as a series of a role, and ``_whisker_key`` keys its kind.
    """
    if not place.size:
        return False
    look = _REFERENCE_LINE if reference else _WHISKER
    times, lower, upper = summary.times[place], summary.lower[place], summary.upper[place]
    axis.vlines(times, lower, upper, color=color, linewidth=look["linewidth"], alpha=look.get("alpha"))
    if (flat := lower == upper).any():
        axis.scatter(times[flat], lower[flat], marker="|", s=_FLAT_WHISKER_PT**2, color=color, linewidths=look["linewidth"], alpha=look.get("alpha"))
    return True


def _whisker_key(*, reference: bool = False) -> Line2D:
    """Return the neutral key of a band's whiskers (``_draw_band``): a vertical stroke without line style, wide and pale for a reference."""
    look = _REFERENCE_LINE if reference else _WHISKER
    return Line2D(
        [],
        [],
        color=_NEUTRAL_COLOUR,
        alpha=look.get("alpha"),
        marker="|",
        markersize=_WHISKER_KEY_PT,
        markeredgewidth=look["linewidth"],
        linestyle="none",
    )


def _shared_references(
    references: Mapping[str, tuple[_ExactTimeSummary, Sequence[str], Sequence[tuple[np.ndarray, np.ndarray]]]],
) -> dict[str, tuple[str, ...]]:
    """
    Return, for every series whose reference another series shares exactly, the labels of all series sharing it (in order).

    ``references`` holds each series' reference summary of one window with the case of every summarized row and those rows. Two
    references are one when at every stored time the same cases contribute, each finite or not alike, and their summaries, invalid
    counts included, agree bit for bit; a series sharing with no other is left out, so it keeps its tinted reference.
    """
    groups: dict[tuple[Any, ...], list[str]] = {}
    for label, (summary, cases, rows) in references.items():
        support = tuple(tuple((cases[row], math.isfinite(value)) for row, value in group) for group in _exact_time_groups(rows)[1])
        arrays = (summary.times, summary.mean, summary.lower, summary.upper, summary.count, summary.invalid_count)
        groups.setdefault((support, *(values.tobytes() for values in arrays)), []).append(label)
    return {label: tuple(group) for group in groups.values() if len(group) > 1 for label in group}


def _reference_keys(shared: Mapping[str, tuple[str, ...]], datasets: Mapping[str, str], runs: Mapping[str, str]) -> dict[str, str]:
    """
    Return the key of every shared reference by its first series: «Reference», with dataset and runs where needed.

    The dataset stands where the series show several, the runs (``_series_runs``, else the series labels) where another shared
    reference of that dataset stands beside it.
    """
    groups = tuple(dict.fromkeys(shared.values()))
    keys = {}
    for group in groups:
        dataset = datasets[group[0]]
        parts = ["Reference", dataset] if len(set(datasets.values())) > 1 else ["Reference"]
        if sum(datasets[other[0]] == dataset for other in groups) > 1:
            parts.append(", ".join(runs.get(label, label) for label in group))
        keys[group[0]] = " · ".join(parts)
    return keys


def _mark_unavailable(axis: Axes, time: np.ndarray, unavailable: np.ndarray, *, color: Any) -> None:
    """Mark on a field's own row the exact times at which a non-finite value voids its score."""
    if unavailable.any():
        axis.plot(
            time[unavailable],
            np.full(int(unavailable.sum()), _UNAVAILABLE_HEIGHT),
            transform=axis.get_xaxis_transform(),
            linestyle="none",
            marker="x",
            markersize=5,
            color=color,
            clip_on=False,
            label=_UNAVAILABLE_LABEL,
        )


def _staggered(index: int, count: int, length: int) -> tuple[int, int]:
    """
    Return the ``markevery`` of curve ``index`` of ``count`` over ``length`` points: sparse, staggered, never without a marker.

    The curves share one step, a multiple of ``count`` that leaves about ``_CURVE_MARKS`` markers each, and start at successive
    offsets within it, so equal values stay apart; an offset beyond a short curve wraps into it.
    """
    spacing = max(1, length // (count * _CURVE_MARKS))
    return index * spacing % max(length, 1), count * spacing


def _draw_target_lines(axes: Sequence[Axes], records: Sequence[Any]) -> list[float]:
    """
    Draw one horizontal line per distinct wet-basis target moisture of the shown cases, in the plot's percent, and return the levels.

    The target is a property of the case: records of one case that carry different targets are refused.
    """
    by_case: dict[str, set[float]] = {}
    for record in records:
        by_case.setdefault(str(_value(record, "case_id")), set()).add(float(_value(_value(record, "target"), "target_moisture_wb")))
    for case_id, case_targets in by_case.items():
        if len(case_targets) > 1:
            msg = f"Transient plot records of case {case_id!r} carry different target_moisture_wb values: {sorted(case_targets)}."
            raise ValueError(msg)
    targets = sorted({target for case_targets in by_case.values() for target in case_targets})
    levels = [float(level) for level in field_labels.display_values(targets, "1", field="target_moisture_wb")]
    for index, level in enumerate(levels):
        for column, axis in enumerate(axes):
            legend = f"Target {_display_symbol(_GRAIN_MOISTURE_FIELD)}" if index == column == 0 else "_nolegend_"
            axis.axhline(level, color=_TARGET_COLOUR, linestyle=":", linewidth=1.2, label=legend)
    return levels


def _label_target_lines(figure: Figure, axis: Axes, levels: Sequence[float]) -> None:
    """
    Write each target's value in a strip right of ``axis``, joined by a leader to the right end of its line; close values are pushed apart.

    The strip lies outside the panel, clear of every curve and band, and takes part in the layout, so the legend stands right of it.
    """
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.execute(figure)  # the final size and y scale of the row
    lines = [float(axis.transData.transform((0.0, level))[1]) for level in levels]
    placed = list(lines)
    gap = _TARGET_LABEL_GAP_PT * figure.dpi / 72.0
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    centring = (sum(lines) - sum(placed)) / len(placed)
    unit = _display_unit(_GRAIN_MOISTURE_FIELD, difference=False)
    for level, line, pixel in zip(levels, lines, placed, strict=True):
        text = axis.annotate(
            f"{level:g} {unit}",
            xy=(1.0, level),
            xycoords=axis.get_yaxis_transform(),
            xytext=(_TARGET_LABEL_OFFSET_PT, (pixel + centring - line) * 72.0 / figure.dpi),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize="small",
            color=_TARGET_COLOUR,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": _TARGET_COLOUR, "linewidth": 0.6, "relpos": (0.0, 0.5), "shrinkA": 0, "shrinkB": 0},
        )
        text.set_in_layout(True)  # the layout reserves the strip, so the legend stands right of it


def _finish_window_figure(
    figure: Figure,
    axes: np.ndarray,
    windows: Sequence[time_ui.TimeWindow],
    title: str,
    *,
    note: str | None,
    target_axis: Axes | None = None,
    target_levels: Sequence[float] = (),
    runs: Mapping[str, str] | None = None,
    keys: Mapping[str, Artist] | None = None,
) -> Figure:
    """
    Title the windows and the figure, style every time axis and write the one legend outside right, top aligned with the top frame.

    Tick numbers and the time label stay under the bottom row; the case-count row (last) gets its own label and integer ticks.
    The target values stand in a strip right of ``target_axis``. The legend (``figure_layout.add_legend``) collects the entries of
    every axis once by label, grouped by ``runs`` (``_legend``), then ``keys``, right of that strip, and grows the figure, never a row.
    """
    last_row = axes.shape[0] - 1
    for column, window in enumerate(windows):
        figure_layout.set_panel_title(axes[0, column], window.title)
        for row in range(axes.shape[0]):
            time_ui.style_window_axis(axes[row, column], window, bottom=row == last_row)
            axes[row, column].grid(axis="y", alpha=0.25)
    axes[last_row, 0].set_ylabel(_count_symbol(windows))
    axes[last_row, 0].set_ylim(bottom=0)
    axes[last_row, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    figure_layout.set_figure_title(figure, axes, title, note=note)
    if target_axis is not None and target_levels:  # a phase view without a pooled case draws no target
        _label_target_lines(figure, target_axis, target_levels)  # before the legend, which stands right of the values
    _legend(figure, axes, runs, keys)
    return figure


def plot_state_trajectory_summary(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    styles: series.SeriesStyles,
    time_axis: Literal["hours", "phase"] = "hours",
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot material-aware aggregate or single-case trajectories on exact times, in time windows, or over the drying phase.

    The title ends with the short labels ``models`` of the compared runs (``layout.models_suffix``, as in every figure below), also of
    one the figure cannot draw: its note or panel names that run's gap.

    Each state has a row of reference and prediction means with a half-height row of prediction minus reference below it,
    where a non-finite value that voids a score is marked; the grain moisture also carries one line per distinct wet-basis
    target moisture of the shown cases. A last row counts the cases with a stored time at each exact time, the same for every state.
    With ``time_axis="phase"`` the series pool by ``time_ui.phase_cohort``, the targets of pooled cases only, the note naming each cohort
    once per dataset. A series takes its style from the comparison's ``styles`` (``series.comparison_styles``) by its records'
    identity; its reference is drawn wide and pale under its prediction, a reference several runs share once in grey under the
    key «Reference» (``_reference_keys``), and every 10th-90th band of a panel as whiskers at times of its own
    (``_whisker_positions``, ``_draw_band``), keyed once per kind. A reference curve is the reference's own mean, so adding a run
    never changes it; moisture predictions and differences use the cells valid in both. The difference row is the mean of each
    case's own difference, so a case enters it with both sides or voids that time, never with one.
    """
    fields = _selected_fields(state_fields)
    phase = time_ui.is_phase(time_axis)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not label or not records for label, records in admitted.items()):
        msg = "Trajectory summaries require labelled non-empty record groups."
        raise ValueError(msg)
    styles = _record_styles(admitted, styles)
    datasets = {label: _dataset_label(_dataset_key(records)) for label, records in admitted.items()}
    runs = _series_runs(datasets)
    windows, ends = _shared_windows(admitted, exclude_origin=False, phase=phase)
    cohorts = {label: _phase_cohort(records) for label, records in admitted.items()} if phase else {}
    shown_cases = [record for label, records in admitted.items() for case, record in enumerate(records) if not phase or case in cohorts[label].cases]
    figure, axes = _window_figure(windows, [height for _field in fields for height in (1.0, _HALF_ROW)] + [_HALF_ROW])
    try:
        target_axis, target_levels = None, []
        case_ids = {label: [str(_value(record, "case_id")) for record in records] for label, records in admitted.items()}
        # The case of every row a window summarizes: each record's, over the phase each pooled one's.
        members = {label: [cases[case] for case in cohorts[label].cases] if phase else cases for label, cases in case_ids.items()}
        order = {label: index for index, label in enumerate(admitted)}  # staggers the count markers
        drawn: set[str] = set()  # the band kinds with a whisker, each keyed once
        for row, field in enumerate(fields):
            quantity_axes, difference_axes = axes[2 * row], axes[2 * row + 1]
            # Per series and window: paired differences and prediction means over the cells valid in both, the reference's own means and rows.
            summaries: dict[tuple[str, int], tuple[_ExactTimeSummary, _ExactTimeSummary, _ExactTimeSummary, list[tuple[np.ndarray, np.ndarray]]]] = {}
            for label, records in admitted.items():
                rows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"difference": [], "prediction": [], "own": []}
                for record in records:
                    record_times, reference, prediction = _trajectory_values(record, field)
                    reference_means, prediction_means = _spatial_means(record, field, reference, prediction)
                    # A raw non-finite value voids its time here, so the trajectory marks it; a snapshot keeps the cells valid in both.
                    prediction_means = _voided(prediction_means, prediction)
                    rows["prediction"].append((record_times, prediction_means))
                    # Each case's own difference, void where either side is, so no case enters the difference with one side alone.
                    rows["difference"].append((record_times, _voided(prediction_means - reference_means, reference)))
                    rows["own"].append((record_times, _reference_means(record, field, reference)))
                for column, window in enumerate(windows):
                    shown_rows = {
                        kind: _phase_rows(cohorts[label], held) if phase else _window_rows(window, held, ends[label]) for kind, held in rows.items()
                    }
                    if not any(times.size for times, _values in shown_rows["own"]):
                        continue
                    difference_summary, prediction_summary, own_summary = (
                        _aggregate_exact_time_values(shown_rows[kind]) for kind in ("difference", "prediction", "own")
                    )
                    summaries[label, column] = (difference_summary, prediction_summary, own_summary, shown_rows["own"])
            for column in range(len(windows)):
                shown = {label: summaries[label, column] for label in admitted if (label, column) in summaries}
                # Series of one cohort share its reference, judged on the reference alone, so a prediction's masked cells never split it.
                shared = _shared_references({label: (held[2], members[label], held[3]) for label, held in shown.items()})
                keys = _reference_keys(shared, datasets, runs)
                # Every band of the panel, a shared reference's once, placed together, so their whiskers stand at times of their own.
                bands = {
                    (label, kind): held[2] if kind == "Reference" else held[1]
                    for label, held in shown.items()
                    for kind in ("Reference", "Prediction")
                    if kind == "Prediction" or shared.get(label, (label,))[0] == label
                }
                places = dict(zip(bands, _whisker_positions(list(bands.values())), strict=True))
                for label, (difference_summary, prediction_summary, own_summary, _own_rows) in shown.items():
                    style = styles[label]
                    line = {"color": style.color, "linestyle": style.linestyle}
                    marked = {**line, "marker": style.marker, "markevery": plot_layout.MARK_EVERY}  # the material of a run's curves
                    time = difference_summary.times
                    axis = quantity_axes[column]
                    group = shared.get(label)
                    # Every reference curve is the reference's own mean, so a second run beside this one never changes it.
                    if group is None:
                        axis.plot(own_summary.times, own_summary.mean, **line, **_REFERENCE_LINE, label=f"{label} · Reference")
                    elif group[0] == label:  # one neutral curve, in the dataset's line style and marker, for the identical references
                        neutral = {**marked, "color": _NEUTRAL_COLOUR}
                        axis.plot(own_summary.times, own_summary.mean, **neutral, **_REFERENCE_LINE, label=keys[label])
                    place = places.get((label, "Reference"))  # none where another series draws the shared reference
                    if place is not None and _draw_band(
                        axis, own_summary, place, color=style.color if group is None else _NEUTRAL_COLOUR, reference=True
                    ):
                        drawn.add("Reference")
                    axis.plot(time, prediction_summary.mean, **marked, linewidth=_PREDICTION_WIDTH, label=f"{label} · Prediction")
                    if _draw_band(axis, prediction_summary, places[label, "Prediction"], color=style.color):
                        drawn.add("Prediction")
                    difference_axes[column].plot(time, difference_summary.mean, **marked, linewidth=1.3)
                    _mark_unavailable(difference_axes[column], time, difference_summary.invalid_count > 0, color=style.color)
                    if row == 0:
                        _draw_case_count(axes[-1, column], time, difference_summary.count, style=style, index=order[label], total=len(admitted))
            if field == _GRAIN_MOISTURE_FIELD:
                target_axis = quantity_axes[-1]  # the values stand right of the last window
                target_levels = _draw_target_lines(quantity_axes, shown_cases)
            quantity_axes[0].set_ylabel(_display_label(field))
            # Two lines, so the label fits the half-height row and clears the neighbouring rows' ticks.
            difference_axes[0].set_ylabel(f"Prediction \N{MINUS SIGN}\nreference [{_display_unit(field, difference=True)}]")
            for axis in difference_axes:
                axis.axhline(0.0, color="0.6", linewidth=0.8)
        where = "spatial means at each case's stored state nearest to each phase point" if phase else "exact-time spatial means"
        note = f"{where}; 10th-90th percentile whiskers require {_count_symbol(windows)} ≥ 2"
        if any(moisture_display.is_moisture_field(field) for field in fields):
            note += "; moisture predictions and differences over the cells valid in both"
        return _finish_window_figure(
            figure,
            axes,
            windows,
            ("Reference vs prediction trajectories over drying phase" if phase else "Reference vs prediction trajectories")
            + plot_layout.models_suffix(models),
            note=plot_layout.series_note(styles, note, _cohort_note({label: (datasets[label], cohort.label()) for label, cohort in cohorts.items()})),
            target_axis=target_axis,
            target_levels=target_levels,
            runs=runs,
            keys={f"{kind} 10th-90th": _whisker_key(reference=kind == "Reference") for kind in ("Reference", "Prediction") if kind in drawn},
        )
    except Exception:
        plt.close(figure)
        raise


def _normalized_squared_error_trajectory(
    record: Any,
    *,
    scaling_state: Mapping[str, Any],
    fields: Sequence[str],
    origin: bool = False,
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    """Return the saved-scale per-channel mean squared error over the grid at exact post-origin times, or at all with ``origin``."""
    selected_fields = tuple(fields)
    if not selected_fields or set(selected_fields).difference(artifact.STATE_ORDER):
        msg = "Normalized error trajectories require stored transient states."
        raise ValueError(msg)
    scaling = datasets.preprocessing.normalization.TrainScaling.from_state_dict(scaling_state)
    # torch.tensor copies: the stored payloads are read-only arrays, which torch.as_tensor would share with a warning.
    reference = torch.tensor(np.asarray(_value(record, "reference_states"), dtype=np.float32))
    prediction = torch.tensor(np.asarray(_value(record, "predicted_states"), dtype=np.float32))
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK:
        msg = "Normalized error trajectories require aligned state arrays."
        raise ValueError(msg)
    first = 0 if origin else 1
    normalized_error = (scaling.encode_state(prediction) - scaling.encode_state(reference)).detach().cpu().numpy()[first:]
    result = {
        field: np.mean(np.square(normalized_error[:, artifact.STATE_ORDER.index(field)], dtype=np.float64), axis=(1, 2)) for field in selected_fields
    }
    return times[first:], result


def plot_error_over_physical_time(
    series: Mapping[str, Sequence[Any]],
    *,
    scaling_states: Mapping[str, Mapping[str, Any]],
    state_fields: Sequence[str],
    styles: series.SeriesStyles,
    time_axis: Literal["hours", "phase"] = "hours",
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot saved-scale channel errors over exact physical time, or over the drying phase, by context, in the comparison's ``styles``.

    In hours the line is the normalized RMSE of the cases present at each exact time, reduced like every reported field RMSE
    (root after the material-case mean of the per-case mean squared error); whiskers at times of each series' own in a panel
    (``_whisker_positions``, ``_draw_band``) span the 10th to 90th quantile of the per-case RMSEs, under one key. With
    ``time_axis="phase"`` the contexts pool by ``time_ui.phase_cohort`` over all stored states, the origin included, and the cases
    are reduced per phase point as per exact time; the note names each context's cohort label. A record without a time after its
    origin raises ``ValueError``, in phase mode only a reached one (``time_ui.phase_cohort``).
    """
    phase = time_ui.is_phase(time_axis)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not fields or not admitted or set(admitted) != set(scaling_states):
        msg = "Error-over-time series require stored fields and matching scaling contexts."
        raise ValueError(msg)
    styles = _record_styles(admitted, styles)
    windows, ends = _shared_windows(admitted, exclude_origin=not phase, phase=phase)
    count_symbol = _count_symbol(windows)
    cohorts: dict[str, time_ui.PhaseCohort] = {}
    figure, axes = _window_figure(windows, [*([1.0] * len(fields)), _HALF_ROW])
    bands: dict[Axes, list[tuple[_ExactTimeSummary, Any]]] = {}  # each panel's bands with their colours
    try:
        for position, (label, records) in enumerate(admitted.items()):
            trajectories = [
                _normalized_squared_error_trajectory(record, scaling_state=scaling_states[label], fields=fields, origin=phase) for record in records
            ]
            materials = [_record_material(record) for record in records]
            if phase:
                cohorts[label] = _phase_cohort(records)
                materials = [materials[case] for case in cohorts[label].cases]
            style = styles[label]
            for row, field in enumerate(fields):
                rows = [(record_times, values[field]) for record_times, values in trajectories]
                for column, window in enumerate(windows):
                    shown = _phase_rows(cohorts[label], rows) if phase else _window_rows(window, rows, ends[label])
                    if not any(times.size for times, _values in shown):
                        continue
                    summary = _aggregate_exact_time_errors(shown, materials=materials)
                    axis = axes[row, column]
                    line = {"color": style.color, "linestyle": style.linestyle, "marker": style.marker, "markevery": plot_layout.MARK_EVERY}
                    axis.plot(summary.times, summary.mean, **line, linewidth=2.0, label=f"{label} · material-case RMSE")
                    bands.setdefault(axis, []).append((summary, style.color))
                    _mark_unavailable(axis, summary.times, summary.invalid_count > 0, color=style.color)
                    if row == 0:
                        _draw_case_count(axes[-1, column], summary.times, summary.count, style=style, index=position, total=len(admitted))
        banded = False  # whether any series drew a whisker, which one key then names
        for axis, held in bands.items():  # a panel's bands placed together, so their whiskers stand at times of their own
            for (summary, colour), place in zip(held, _whisker_positions([summary for summary, _colour in held]), strict=True):
                banded |= _draw_band(axis, summary, place, color=colour)
        for row, field in enumerate(fields):
            axes[row, 0].set_ylabel(f"{field_labels.metric_label('nRMSE', field, mathtext=True)} [{field_labels.display_unit('1')}]")
        where = "at each case's stored state nearest to each phase point" if phase else "at each exact time"
        reduction = f"root of the material-case mean squared error {where}; per-case RMSE 10th-90th whiskers require {count_symbol} ≥ 2"
        scope = "saved-scale nRMSE of the stored states only: the derived grain moisture has none"
        title = ("Error over drying phase" if phase else "Error over physical time") + plot_layout.models_suffix(models)
        datasets = {label: _dataset_label(_dataset_key(records)) for label, records in admitted.items()}
        cohort_texts = {label: (datasets[label], cohort.label()) for label, cohort in cohorts.items()}
        note = plot_layout.series_note(styles, reduction, scope, _cohort_note(cohort_texts))
        keys = {"Per-case RMSE 10th-90th": _whisker_key()} if banded else None
        return _finish_window_figure(figure, axes, windows, title, note=note, runs=_series_runs(datasets), keys=keys)
    except Exception:
        plt.close(figure)
        raise


def plot_relative_norms_over_time(curves: Mapping[str, pd.DataFrame], *, styles: series.SeriesStyles, models: Sequence[str] = ()) -> Figure:
    """
    Plot the per-field relative L2 and H1 of complete rollouts over physical time or the drying phase with their supported case counts.

    ``curves`` maps a series label to :meth:`TransientEvaluationSession.relative_norm_curve` (one material of a frame, or a frame
    pooling its materials), all in hours or all over the phase (a ``phase`` column; the note shows each dataset's cohort label once).
    A gap is a time without a supported case in some material, an x a time at which a non-finite prediction voids the norm; the
    step line counts the supported cases and carries its series' marker. A norm line carries its material marker where its style
    has one (a run over several materials beside other runs, whose curves share colour and line style), its key too; the markers
    of both are sparse and staggered per curve (``_staggered``), so equal values stay apart without moving and a short curve keeps one.
    A curve takes the comparison style of ``styles`` (:meth:`TransientEvaluationSession.series_styles`) that its entity facts
    ``attrs["members"]`` select; a curve without them, or one the comparison lacks, raises ``ValueError``. One legend stands outside right.
    """
    if not curves or any(frame.empty for frame in curves.values()):
        msg = "Relative-norm curves require at least one labelled non-empty curve."
        raise ValueError(msg)
    if missing := [label for label, frame in curves.items() if "members" not in frame.attrs]:
        msg = f"Relative-norm curves {missing} lack the entity facts of their members."
        raise ValueError(msg)
    styles = series.select_styles({label: frame.attrs["members"] for label, frame in curves.items()}, styles)
    phase = {"phase" in frame for frame in curves.values()}
    if len(phase) != 1:
        msg = "Relative-norm curves of one figure must all be in hours or all over the drying phase."
        raise ValueError(msg)
    clock = "phase" if phase == {True} else "physical_time"
    all_times = np.concatenate([frame[clock].to_numpy(dtype=np.float64) for frame in curves.values()])
    display_time = None if clock == "phase" else time_ui.physical_time_display(all_times, preferred_unit="auto")
    heights = (1.0, _SUPPORT_ROW, 1.0, _SUPPORT_ROW)  # each norm row with its supported-case strip below it
    figure, axes = plt.subplots(
        len(heights),
        len(artifact.STATE_ORDER),
        figsize=(5.0 * len(artifact.STATE_ORDER), 3.8 * sum(heights)),
        layout="constrained",
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": heights},
    )
    for row, norm in enumerate(("l2", "h1")):
        for column, field in enumerate(artifact.STATE_ORDER):
            axis, support_axis = axes[2 * row, column], axes[2 * row + 1, column]
            for index, (label, frame) in enumerate(curves.items()):
                selected = frame.loc[frame["metric_id"] == f"physical_relative_{norm}_{field}"].sort_values(clock)
                plotted_time = selected[clock].to_numpy(dtype=np.float64)
                plotted_time = plotted_time if display_time is None else display_time.values(plotted_time)
                style = styles[label]
                start, step = _staggered(index, len(curves), plotted_time.size)
                line = {"color": style.color, "linestyle": style.linestyle, "markevery": (start, step)}
                values = selected["value"].to_numpy(dtype=np.float64)
                finite = np.isfinite(values)
                # A marker on a voided point is invisible: keep the staggered finite points, else the first finite one.
                marks = [point for point in range(start, values.size, step) if finite[point]] or np.flatnonzero(finite)[:1].tolist()
                axis.plot(plotted_time, values, **{**line, "markevery": marks}, marker=style.marker, markersize=6, linewidth=2.0, label=label)
                _mark_unavailable(axis, plotted_time, selected["nonfinite_cases"].to_numpy() > 0, color=style.color)
                support_axis.step(
                    plotted_time, selected["supported_cases"], where="mid", **line, marker=style.marker or "o", markersize=4, alpha=0.55
                )
            for shown, bottom in ((axis, False), (support_axis, True)):
                if display_time is None:
                    time_ui.style_window_axis(shown, time_ui.PHASE_WINDOW, bottom=bottom)
                else:
                    display_time.configure(shown)
                    shown.tick_params(axis="x", labelbottom=bottom)
                    shown.set_xlabel(shown.get_xlabel() if bottom else "")
            figure_layout.set_panel_title(axis, f"Relative {norm.upper()} of {field_labels.field_label(field, mathtext=True)}")
            axis.grid(alpha=0.25)
            support_axis.set_ylim(bottom=0)
            support_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2 * row, 0].set_ylabel(f"Relative {norm.upper()} [-]")
        axes[2 * row + 1, 0].set_ylabel("n(τ)" if clock == "phase" else "n(t)")
    datasets = {label: _dataset_label({(member[2], member[1]) for member in frame.attrs["members"]}) for label, frame in curves.items()}
    labels = _cohort_note({label: (datasets[label], frame.attrs["phase_label"]) for label, frame in curves.items()}) if clock == "phase" else None
    basis = moisture_display.density_basis_note(artifact.STATE_ORDER)
    pooled = any(len({member[2] for member in frame.attrs["members"]}) > 1 for frame in curves.values())
    reduction = "per-case ratios, equal over the cases of a curve's material" + ("; a pooled role weighs its materials equally" if pooled else "")
    note = plot_layout.series_note(styles, f"{reduction}; gaps are unavailable", basis, labels)
    title = ("Relative norms over drying phase" if clock == "phase" else "Relative norms over physical time") + plot_layout.models_suffix(models)
    figure_layout.set_figure_title(figure, axes, title, note=note)
    _legend(figure, axes, _series_runs(datasets))
    return figure


def _aggregate_snapshot_evidence(  # noqa: PLR0915 -- one bounded physical snapshot reduction
    records: Sequence[Any],
    *,
    state_field: str,
    physical_times: Sequence[float],
    compute_device: str = "cpu",
    device_evidence: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool exact-time case evidence without stacking complete trajectories.

    ``physical_times`` holds each record's stored hour. Returns the grid coordinates, the pooled mean absolute, mean signed
    (``error_sum`` over the count) and signed-error standard deviation maps in displayed difference units, and the displayed
    case spatial means (see ``_spatial_means``) of reference and prediction. A cell any case left non-finite is NaN in all three maps,
    and a cell with fewer than two finite cases also in the standard deviation map.
    """
    if state_field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Aggregate transient snapshot field is unsupported: {state_field!r}."
        raise ValueError(msg)
    admitted = tuple(records)
    if not admitted:
        msg = "Aggregate transient snapshots require at least one complete-rollout record."
        raise ValueError(msg)
    error_sum: np.ndarray | None = None
    absolute_error_sum: np.ndarray | None = None
    error_square_sum: np.ndarray | None = None
    counts: np.ndarray | None = None
    divergent: np.ndarray | None = None
    x_values: np.ndarray | None = None
    y_values: np.ndarray | None = None
    reference_means: list[float] = []
    prediction_means: list[float] = []
    cuda_reference: list[np.ndarray] = []
    cuda_prediction: list[np.ndarray] = []
    cuda_masks: list[np.ndarray] = []
    selected_device: compute.AnalysisDevice | None = None
    batch_cases = 1
    routes: set[str] = set()

    def flush_cuda_batch() -> None:
        nonlocal error_sum, absolute_error_sum, error_square_sum, counts, selected_device
        if not cuda_reference or selected_device is None:
            return
        if error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None:
            msg = "Aggregate transient snapshot accumulators were not initialized."
            raise RuntimeError(msg)
        reduced = compute.reduce_error_batch(
            np.stack(cuda_reference),
            np.stack(cuda_prediction),
            np.stack(cuda_masks),
            device=selected_device,
        )
        error_sum += reduced.signed_sum
        absolute_error_sum += reduced.absolute_sum
        error_square_sum += reduced.squared_sum
        counts += reduced.support_count
        routes.add(reduced.device.label)
        selected_device = reduced.device
        cuda_reference.clear()
        cuda_prediction.clear()
        cuda_masks.clear()

    for record, record_time in zip(admitted, physical_times, strict=True):
        if _value(record, "mode") != "autonomous_full" or _value(record, "requested_horizon") != "full":
            msg = "Aggregate transient snapshots require complete autonomous rollouts only."
            raise ValueError(msg)
        time_index, _selected_time = _time_index(record, record_time)
        reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
        prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
        static = np.asarray(_value(record, "static_conditioning"), dtype=np.float64)
        if (
            reference.shape != prediction.shape
            or reference.ndim != _STATE_ARRAY_RANK
            or reference.shape[1] != len(artifact.STATE_ORDER)
            or static.shape != (len(artifact.STATIC_ORDER), *reference.shape[-2:])
        ):
            msg = "Aggregate transient snapshots require aligned state and coordinate arrays."
            raise ValueError(msg)
        # Every case is evaluated on its full grid.
        mask = np.ones(reference.shape[-2:], dtype=bool)
        current_x = static[0]
        current_y = static[1]
        if x_values is None:
            x_values = current_x
            y_values = current_y
            error_sum = np.zeros(mask.shape, dtype=np.float64)
            absolute_error_sum = np.zeros(mask.shape, dtype=np.float64)
            error_square_sum = np.zeros(mask.shape, dtype=np.float64)
            counts = np.zeros(mask.shape, dtype=np.int64)
            divergent = np.zeros(mask.shape, dtype=bool)
            batch_cases = max(1, _MAX_ERROR_BATCH_INPUT_BYTES // (mask.size * _ERROR_PAIR_MASK_BYTES))
            selected_device = compute.resolve_analysis_device(compute_device)
        elif y_values is None or not np.array_equal(current_x, x_values) or not np.array_equal(current_y, y_values):
            msg = "Aggregate transient spatial maps require one exact shared coordinate grid."
            raise ValueError(msg)
        snapshots = _field_snapshots(record, state_field, reference, prediction, (time_index,))
        reference_snapshot, prediction_snapshot = snapshots[0][0], snapshots[1][0]
        if not np.isfinite(reference_snapshot[mask]).all():
            msg = "Aggregate transient snapshots require finite reference values on every grid."
            raise ValueError(msg)
        if error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None or divergent is None:
            msg = "Aggregate transient snapshot accumulators were not initialized."
            raise RuntimeError(msg)
        reference_mean, prediction_mean = _spatial_means(record, state_field, *snapshots)
        reference_means.append(float(reference_mean[0]))
        prediction_means.append(float(prediction_mean[0]))
        # Each case converts to its displayed values before pooling (water with its own dry bulk density), so the pooled
        # errors carry the displayed difference unit, such as pp for phi and water.
        reference_snapshot = _absolute_display(record, state_field, reference_snapshot)
        prediction_snapshot = _absolute_display(record, state_field, prediction_snapshot)
        # A diverging or unphysical cell gets no pooled error; the remaining cells keep their evidence.
        finite = mask & np.isfinite(reference_snapshot) & np.isfinite(prediction_snapshot)
        divergent |= mask & ~finite
        if selected_device is not None and selected_device.backend == "cuda":
            cuda_reference.append(reference_snapshot)
            cuda_prediction.append(prediction_snapshot)
            cuda_masks.append(finite)
            if len(cuda_reference) >= batch_cases:
                flush_cuda_batch()
        else:
            error = prediction_snapshot - reference_snapshot
            error_sum[finite] += error[finite]
            absolute_error_sum[finite] += np.abs(error[finite])
            error_square_sum[finite] += np.square(error[finite], dtype=np.float64)
            counts[finite] += 1
            if selected_device is not None:
                routes.add(selected_device.label)
    if (
        x_values is None
        or y_values is None
        or error_sum is None
        or absolute_error_sum is None
        or error_square_sum is None
        or counts is None
        or divergent is None
    ):
        msg = "Aggregate transient snapshot evidence was not initialized."
        raise RuntimeError(msg)
    flush_cuda_batch()
    if device_evidence is not None:
        device_evidence["route"] = "; ".join(sorted(routes))
    if not all(np.isfinite(values[counts > 0]).all() for values in (error_sum, absolute_error_sum, error_square_sum)):
        msg = "Aggregate transient error statistics are non-finite on required support."
        raise ValueError(msg)
    mean_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    mean_absolute_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    mean_square_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    np.divide(error_sum, counts, out=mean_error, where=counts > 0)
    np.divide(absolute_error_sum, counts, out=mean_absolute_error, where=counts > 0)
    np.divide(error_square_sum, counts, out=mean_square_error, where=counts > 0)
    error_variance = np.maximum(mean_square_error - np.square(mean_error), 0.0)
    mean_absolute_error[divergent] = np.nan
    mean_error[divergent] = np.nan
    error_variance[divergent | (counts < _MIN_BAND_CASES)] = np.nan  # one case has no spread: its zero would read as a measured one
    return (
        x_values,
        y_values,
        mean_absolute_error,
        mean_error,
        np.sqrt(error_variance),
        np.asarray(reference_means, dtype=np.float64),
        np.asarray(prediction_means, dtype=np.float64),
    )


def _labelled_record_series(
    value: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    allow_empty_contexts: bool = False,
) -> dict[str, tuple[Any, ...]]:
    """Normalize one sequence or labelled multi-context aggregate."""
    if isinstance(value, Mapping):
        series: dict[str, tuple[Any, ...]] = {}
        for label, records in value.items():
            if not isinstance(label, str) or not label:
                msg = "Transient aggregate context labels must be non-empty text."
                raise ValueError(msg)
            admitted = tuple(records)
            if not admitted and not allow_empty_contexts:
                msg = "Transient aggregate contexts must contain at least one record."
                raise ValueError(msg)
            series[label] = admitted
    else:
        admitted = tuple(value)
        if not admitted:
            msg = "Transient aggregate plots require at least one record."
            raise ValueError(msg)
        series = {"": admitted}
    if not series or not any(series.values()):
        msg = "Transient aggregate plots require at least one context."
        raise ValueError(msg)
    return series


def _snapshot_selection(
    series: Mapping[str, tuple[Any, ...]], *, physical_time: float | None, phase: float | None
) -> tuple[dict[str, tuple[Any, ...]], dict[str, list[float]], str, str | None]:
    """
    Return each context's records, their stored hours, the title's time and the phase note.

    At a phase the records are those ``time_ui.phase_cohort`` pools, none where every case is censored or unknown: such a context
    still gets its panel, headed n(τ) = 0, and the note names its counts.
    """
    if physical_time is not None and phase is None:
        return dict(series), {label: [physical_time] * len(records) for label, records in series.items()}, f"t={physical_time:g} h", None
    if phase is None or physical_time is not None:
        msg = "Aggregate snapshots select exactly one of an exact physical time and a drying phase."
        raise ValueError(msg)
    index = time_ui.phase_index(phase)
    cohorts = {label: _phase_cohort(records) for label, records in series.items()}
    selected = {label: tuple(series[label][case] for case in cohort.cases) for label, cohort in cohorts.items()}
    note = time_ui.phase_note({label: f"{cohort.hours_label(phase)}; {cohort.label()}" for label, cohort in cohorts.items()})
    return selected, {label: cohort.hours[:, index].tolist() for label, cohort in cohorts.items()}, f"τ={phase:.2f}", note


def _count_headers(
    admitted: Mapping[str, Sequence[Any]], series: Mapping[str, Sequence[Any]], *, phase: bool, context_labels: Mapping[str, str] | None = None
) -> list[str]:
    """
    Return each context's header ending with its case count in ``series``, n(t) at an exact time or n(τ) at a phase, also when it is 0.

    A context labelled ``run · dataset`` puts its run on a line of its own above the dataset and count, so neighbouring headers
    never run into each other. The dataset is the context's of ``context_labels`` (context: dataset label), else that of its
    ``admitted`` records before the snapshot selection, so a context without a shown case keeps it either way.
    """
    symbol = "n(τ)" if phase else "n(t)"
    keys = {label: _dataset_key(records) for label, records in admitted.items() if records}
    datasets = {label: _dataset_label(key) for label, key in keys.items() if len({role for _material, role in key}) == 1}
    runs = _series_runs(datasets | dict(context_labels or {}))
    headers = []
    for label, records in series.items():
        shown = f"{runs[label]}\n{label.removeprefix(f'{runs[label]} · ')}" if label in runs else label
        headers.append(f"{shown} · {symbol} = {len(records)}" if label else f"{symbol} = {len(records)}")
    return headers


def _header_lines_in(headers: Sequence[str]) -> float:
    """Return the height in inches the extra lines of the tallest column header take above the maps."""
    return max(header.count("\n") for header in headers) * figure_layout.PANEL_TITLE_PT * _LINE_SPACING / _POINTS_PER_INCH


def _aggregate_divisors(
    series: Mapping[str, Sequence[Any]],
    fields: Sequence[str],
    reference_scales: Mapping[str, Mapping[str, error_diagnostics.ReferenceScale | str]] | None,
) -> dict[str, float]:
    """Return each field's train sigma in displayed difference units, shared by every shown context (none without one), else raise ``ValueError``."""
    divisors: dict[str, float] = {}
    for field in fields:
        if moisture_display.is_moisture_field(field):
            msg = f"Normalized aggregate errors are defined on the stored states, but {field!r} is shown in wet-basis percentage points."
            raise ValueError(msg)
        scales: list[error_diagnostics.ReferenceScale] = []
        for context in (name for name, current_records in series.items() if current_records):
            scale = (reference_scales or {}).get(context, {}).get(field)
            if scale is None or isinstance(scale, str):  # a missing scale or its unavailability reason, "" included
                msg = scale or f"No valid train reference scale was given for {field!r} in context {context!r}."
                raise ValueError(msg)
            if scale.field != field or scale.unit != _state_unit(field):
                msg = f"The reference scale of {field!r} in context {context!r} belongs to a different physical field or unit."
                raise ValueError(msg)
            scales.append(scale)
        if scales:  # a phase no context reached divides nothing
            common = error_diagnostics.require_common_reference_scale(scales)
            divisors[field] = float(_difference_display(field, np.asarray(common.value)))
    return divisors


def plot_aggregate_error_maps(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    statistic: str,
    physical_time: float | None = None,
    phase: float | None = None,
    compute_device: str = "cpu",
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    reference_scales: Mapping[str, Mapping[str, error_diagnostics.ReferenceScale | str]] | None = None,
    models: Sequence[str] = (),
    context_labels: Mapping[str, str] | None = None,
) -> Figure:
    """
    Plot exact-time mean or standard-deviation error maps on the fixed map-grid geometry: a row per field, a column per context.

    The maps pool every record at one exact ``physical_time`` or the ``time_ui.phase_cohort`` of one drying ``phase`` (exactly
    one is given; a footer band under the maps names the resolved hours and cohorts). Every column header ends with its case
    count n(t) (n(τ) at a phase), also 0, and a context without a case keeps its empty column, also where no context has one;
    a ``run · dataset`` header breaks after the run (``_count_headers``, with the dataset labels ``context_labels``).
    The grid of the given records lays the maps out. The title counts nothing. The compute route goes to the log, not onto the figure.
    The mean map follows ``error_mode``: the signed mean of prediction minus reference (default, blue-white-red), the mean absolute
    error, or the mean absolute error over the train sigma, which needs ``reference_scales`` (context, then field) and is refused
    unless every shown context has the same one, and for water, which is shown in wet-basis percentage points. The standard
    deviation map is that of the signed error whatever the mode. Each row has one round scale per dataset (material and role of
    its records), shared by the compared runs of that dataset, so one dataset never sets another's colours; every bar carries its
    unit, and an arrow stands only where that map has values beyond.
    """
    fields = _selected_fields(state_fields)
    if statistic not in {"mean", "std"}:
        msg = "Aggregate error-map statistic must be 'mean' or 'std'."
        raise ValueError(msg)
    if error_mode not in error_diagnostics.TRANSIENT_ERROR_MODES:
        msg = f"Aggregate error mode must be one of {error_diagnostics.TRANSIENT_ERROR_MODES}, not {error_mode!r}."
        raise ValueError(msg)
    admitted = _labelled_record_series(records, allow_empty_contexts=True)
    series, selected_times, time_text, phase_note = _snapshot_selection(admitted, physical_time=physical_time, phase=phase)
    signed = statistic == "mean" and error_mode == error_diagnostics.SIGNED_ERROR
    normalized = statistic == "mean" and error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    divisors = _aggregate_divisors(series, fields, reference_scales) if normalized else {}
    cells: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    routes: set[str] = set()
    for column, (context, current_records) in enumerate(series.items()):
        for row, field in enumerate(fields):
            if current_records:
                device_evidence: dict[str, str] = {}
                x_values, y_values, absolute, signed_mean, std_error, *_ = _aggregate_snapshot_evidence(
                    current_records,
                    state_field=field,
                    physical_times=selected_times[context],
                    compute_device=compute_device,
                    device_evidence=device_evidence,
                )
                routes.update(device_evidence.values())
                displayed = std_error if statistic == "std" else signed_mean if signed else absolute / divisors.get(field, 1.0)
                cells[row, column] = (x_values, y_values, displayed)
    extents = {_map_extent(record) for current_records in admitted.values() for record in current_records}  # also where none is pooled
    if len(extents) != 1:
        msg = "The maps of one figure share one physical extent, but the given contexts have different grids."
        raise ValueError(msg)
    extent = next(iter(extents))
    _LOGGER.info("Aggregate error maps at %s computed on %s", time_text, "; ".join(sorted(routes)) or "no route")
    below = _FOOTER_LINE_IN * (phase_note.count("\n") + 1) if phase_note else 0.0  # the phase note's lines never reach the maps
    headers = _count_headers(admitted, series, phase=phase is not None, context_labels=context_labels)
    domain = (extent[1] - extent[0], extent[3] - extent[2])
    geometry = layout.plan_geometry(len(fields), len(series), domain, width_cap=None, below_in=below, above_in=_header_lines_in(headers))
    figure, axes, slots = layout.map_grid(geometry, extent=extent)
    datasets = [_dataset_key(current_records) for current_records in series.values()]
    top_notes: list[str] = []
    for row, field in enumerate(fields):
        unit = "1" if normalized else _display_unit(field, difference=True)
        definitions: dict[frozenset[tuple[str, str]], field_maps.FieldMapDefinition | None] = {}
        for dataset in dict.fromkeys(datasets):  # one scale per dataset, over the cells of its runs
            pooled = [cells[row, column][2] for column, key in enumerate(datasets) if key == dataset and (row, column) in cells]
            available = np.concatenate([values[np.isfinite(values)] for values in pooled]) if pooled else np.empty(0)
            mode = error_mode if statistic == "mean" else "Absolute error"
            definitions[dataset] = _error_definition(field, mode, available) if available.size else None
        bar_label = f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} [{field_labels.display_unit('1')}]" if normalized else f"[{unit}]"
        for column in range(len(series)):
            definition = definitions[datasets[column]]
            exponent = 0 if definition is None else layout.error_format(definition.boundaries, signed=signed).exponent
            cell = cells.get((row, column))
            # R16: one small mean line above each mean map, "n/a" without one; the std maps have none.
            shown = None if cell is None else cell[2]
            note = "" if statistic == "std" else plot_layout.mean_error_text(shown, unit=None if normalized else unit, exponent=exponent)
            if row == 0:
                top_notes.append(note)
            elif note:
                layout.add_map_note(axes[row, column], note)
            if cell is None:
                layout.empty_cell(axes[row, column], "No case at this exact time" if phase is None else "No reached case to pool")
                continue
            values = cell[2]
            finite = np.isfinite(values)
            if definition is None or not finite.any():
                spread = statistic == "std" and len(series[list(series)[column]]) < _MIN_BAND_CASES
                reason = "Insufficient support: a spread needs 2 cases" if spread else "Prediction non-finite in every cell"
                layout.empty_cell(axes[row, column], reason)
                continue
            image = axes[row, column].pcolormesh(cell[0], cell[1], np.ma.masked_invalid(values), shading="auto", **definition.pcolormesh_kwargs())
            extend = field_maps.locked_extend(values[finite], definition.boundaries)
            layout.add_field_map_colorbar(figure, image, slots[row, column], definition, label=bar_label, extend=extend)
    layout.set_column_headers(axes, headers, top_notes)
    layout.set_row_labels(axes, [_display_symbol(field) for field in fields])
    statistic_label = f"Mean {_ERROR_HEADERS[error_mode].lower()}" if statistic == "mean" else "Standard deviation of signed error"
    layout.set_map_title(figure, f"{statistic_label} at {time_text}{plot_layout.models_suffix(models)}")
    plot_layout.add_scale_note(figure, shared=False)
    if phase_note:
        figure.text(0.5, 0.01, phase_note, ha="center", va="bottom", fontsize=8, color="0.3")
    return figure


def plot_predicted_vs_reference_channels(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    styles: series.SeriesStyles,
    physical_time: float | None = None,
    phase: float | None = None,
    models: Sequence[str] = (),
    context_labels: Mapping[str, str] | None = None,
) -> Figure:
    """
    Plot case spatial means at one exact hour or phase by channel (rows) and context (columns) in comparison styles.

    Every panel title ends with its context's case count n(t) (n(τ) at a phase), also 0, also where no context has a case at that
    phase; such a panel says so and the note keeps the phase cohorts. A ``run · dataset`` title breaks after the run as the map
    headers do (``context_labels``). The compared runs of one dataset share one padded range per channel on both axes, so their
    panels compare by eye; the titles name every series, so no legend repeats them.
    """
    fields = _selected_fields(state_fields)
    admitted = _labelled_record_series(records, allow_empty_contexts=True)
    contexts, selected_times, time_text, phase_note = _snapshot_selection(admitted, physical_time=physical_time, phase=phase)
    styles = _record_styles(contexts, styles)
    figure, raw_axes = plt.subplots(
        len(fields), len(contexts), figsize=(max(7.2, 5.8 * len(contexts)), max(5.2, 4.6 * len(fields))), layout="constrained", squeeze=False
    )
    datasets = [_dataset_key(current_records) for current_records in contexts.values()]
    headers = _count_headers(admitted, contexts, phase=phase is not None, context_labels=context_labels)
    means = {  # the displayed case spatial means of reference and prediction per channel and context
        (row, column): _aggregate_snapshot_evidence(current_records, state_field=field, physical_times=selected_times[context])[-2:]
        for column, (context, current_records) in enumerate(contexts.items())
        if current_records
        for row, field in enumerate(fields)
    }
    for row, field in enumerate(fields):
        limits = {}
        for dataset in dict.fromkeys(datasets):  # a wet-basis reference mean is NaN where no cell of the case is valid
            pairs = [value for (cell, column), pair in means.items() if cell == row and datasets[column] == dataset for value in pair]
            shown = np.concatenate(pairs) if pairs else np.empty(0)
            shown = shown[np.isfinite(shown)]
            # Every range is padded, so no point (a single pair included) sits on the frame.
            limits[dataset] = _padded(float(np.min(shown)), float(np.max(shown))) if shown.size else _padded(0.0, 0.0)
        for column, context in enumerate(contexts):
            axis = raw_axes[row, column]
            if row == 0:
                figure_layout.set_panel_title(axis, headers[column])
            if (row, column) not in means:
                empty = "No case at this exact time" if phase is None else "No reached case to pool"
                axis.text(0.5, 0.5, empty, ha="center", va="center", transform=axis.transAxes)
                continue
            (reference_display, prediction_display), (lower, upper) = means[row, column], limits[datasets[column]]
            style = styles[context]
            axis.scatter(reference_display, prediction_display, s=55, alpha=0.8, color=style.color, marker=style.marker or "o")
            axis.plot((lower, upper), (lower, upper), color="black", linestyle="--", gid=figure_layout.REFERENCE_GID)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            if hidden := int(np.count_nonzero(~np.isfinite(prediction_display))):
                axis.text(0.02, 0.98, f"{hidden} non-finite predicted case mean(s) not shown", va="top", fontsize="small", transform=axis.transAxes)
            label = _display_label(field)
            axis.set_xlabel(f"Reference case spatial mean — {label}")
            axis.set_ylabel(f"Predicted case spatial mean — {label}")
            axis.grid(alpha=0.25)
    note = plot_layout.series_note(styles, phase_note)
    figure_layout.set_figure_title(figure, raw_axes, f"Prediction versus reference at {time_text}{plot_layout.models_suffix(models)}", note=note)
    return figure


def plot_error_distributions(
    case_frame: pd.DataFrame,
    *,
    styles: series.SeriesStyles,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
    models: Sequence[str] = (),
    run_labels: Mapping[str, str] | None = None,
) -> Figure:
    """
    Show complete-rollout case-error distributions by stored channel, one box per frame and material in its comparison colour and role hatch.

    A box's median line takes the series colour; a series of one case has no box to draw and is one point in its colour instead.
    Where a run spans several materials, its boxes share colour and hatch, so each median carries its material's marker, as does
    its legend key. ``run_labels`` name the run of each frame in the series labels (``_summary_series``).
    """
    selected = _full_autonomous_cumulative(case_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient error distributions require at least one stored channel metric."
        raise ValueError(msg)
    columns = [f"normalized_rmse_{field}" for field in fields]
    if missing := [column for column in columns if column not in selected]:
        msg = f"Transient error distributions lack {missing}."
        raise ValueError(msg)
    if not np.isfinite(selected[columns].to_numpy(dtype=np.float64)).all():
        msg = "Transient case errors must be finite."
        raise ValueError(msg)
    groups = _summary_series(selected, run_labels)
    styles = _frame_styles(groups, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_GROUPED_PANEL_IN)
    axis, width = axes[0, 0], 0.8 / len(groups)
    handles: list[Any] = []
    for offset, (label, rows) in enumerate(groups.items()):
        style = styles[label]
        positions = np.arange(len(fields)) - 0.4 + width * (offset + 0.5)
        values = [rows[column].to_numpy(dtype=np.float64) for column in columns]
        if len(rows) == 1:  # a zero-height box would show only its median line
            points = [value[0] for value in values]
            handles.append(axis.scatter(positions, points, s=_SINGLETON_MARKER_SIZE, color=style.color, marker=style.marker or "o", zorder=3))
            continue
        boxes = axis.boxplot(values, positions=positions, widths=0.9 * width, patch_artist=True, medianprops={"color": style.color, "linewidth": 2.0})
        for box in boxes["boxes"]:
            box.set(facecolor=style.color, hatch=style.hatch, alpha=0.45, edgecolor=style.color)
        handles.append(Patch(facecolor=style.color, hatch=style.hatch))
        if style.marker is not None:  # the material marker at each median, as on the bars of the channel errors
            medians = [float(np.median(value)) for value in values]
            marks = axis.scatter(positions, medians, marker=style.marker, color="white", edgecolors="0.2", s=_SINGLETON_MARKER_SIZE, zorder=3)
            handles[-1] = (handles[-1], marks)
    axis.set_xticks(np.arange(len(fields)), [field_labels.field_label(field, mathtext=True) for field in fields])
    axis.set_ylabel(f"Normalized RMSE [{field_labels.display_unit('1')}]")
    axis.grid(axis="y", alpha=0.25)
    wide = _log_if_wide(axis, _run_values(groups, columns))
    _at_least_zero(axis)
    figure_layout.set_figure_title(
        figure,
        axes,
        "Full-rollout case error distributions" + plot_layout.models_suffix(models),
        note=plot_layout.series_note(styles, moisture_display.density_basis_note(fields), wide),
    )
    labels, legend_groups = _run_entries(handles, list(groups), _summary_runs(groups))
    figure_layout.add_legend(figure, axes, handles, labels, groups=legend_groups)
    return figure


def _thinned_finite_errors(record: Any, field: str) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Return one record's thinned finite reference values (signed), absolute errors, element, raw non-finite and masked counts.

    An element is raw non-finite where a stored value of either side is; it is display-masked where both are finite but the
    wet-basis conversion masks it (negative water). Finite errors are kept at fixed element positions up to the point limit;
    index arrays copy them, so the record's full arrays are released when this returns.
    """
    _times, reference, prediction = _trajectory_values(record, field)
    shown_reference = _absolute_display(record, field, reference[1:])
    target = shown_reference.ravel()
    if moisture_display.is_moisture_field(field):
        # Each record converts with its own dry bulk density; a cell unphysical in either field is masked.
        difference = _absolute_display(record, field, prediction[1:]) - shown_reference
    else:
        difference = _display_state_values(field, prediction[1:] - reference[1:], error=True)
    error = np.abs(difference.ravel())
    finite = np.isfinite(error)
    raw = int(np.count_nonzero(~(np.isfinite(reference[1:]) & np.isfinite(prediction[1:]))))
    kept = np.arange(0, error.size, max(1, int(np.ceil(error.size / _TARGET_MAGNITUDE_POINT_LIMIT))))
    kept = kept[finite[kept]]
    return target[kept], error[kept], error.size, raw, error.size - int(np.count_nonzero(finite)) - raw


def _finite_error_trend(records: Sequence[Any], field: str) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Return target-magnitude bin centers and median finite absolute errors, plus the exact raw non-finite, display-masked and total counts.

    The bins and medians describe the thinned finite errors only. The excluded errors are counted over every element, raw
    non-finite (diverging) apart from those the wet-basis display masks, so the plot names them instead of hiding them.
    """
    target_chunks: list[np.ndarray] = []
    error_chunks: list[np.ndarray] = []
    total = diverging = masked = 0
    for record in records:
        target, error, count, non_finite, display_masked = _thinned_finite_errors(record, field)
        target_chunks.append(target)
        error_chunks.append(error)
        total += count
        diverging += non_finite
        masked += display_masked
    targets = np.concatenate(target_chunks)
    errors = np.concatenate(error_chunks)
    if not targets.size:
        return np.empty(0), np.empty(0), diverging, masked, total
    edges = np.unique(np.quantile(targets, np.linspace(0.0, 1.0, _TARGET_MAGNITUDE_BIN_COUNT + 1)))
    count = max(1, len(edges) - 1)
    bins = np.clip(np.digitize(targets, edges[1:-1], right=True), 0, count - 1)
    present = [index for index in range(count) if np.any(bins == index)]
    centers = np.asarray([float(np.median(targets[bins == index])) for index in present])
    medians = np.asarray([float(np.median(errors[bins == index])) for index in present])
    return centers, medians, diverging, masked, total


def _wrapped(figure: Figure, text: str, width_in: float) -> str:
    """Return ``text`` broken at spaces outside its mathtext into lines no wider than ``width_in`` at the note size; a longer word stays whole."""
    lines = [""]
    for word in _NOTE_WORD.findall(text):
        probe = figure.text(0.0, 0.0, f"{lines[-1]} {word}", fontsize=figure_layout.NOTE_PT)
        fits = probe.get_window_extent().width / figure.dpi <= width_in
        probe.remove()
        if fits or not lines[-1]:
            lines[-1] = f"{lines[-1]} {word}".lstrip()
        else:
            lines.append(word)
    return "\n".join(lines)


def plot_error_vs_target_magnitude(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    styles: series.SeriesStyles,
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot bounded trajectory-element error trends against the displayed (signed) reference value, one panel per field.

    Each series takes its comparison style; the exact counts of non-finite values a series leaves out stand in the note, one line
    per run in the legend's order with each of its datasets' counts and their denominator, wrapped to the panel's width.
    """
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not records for records in admitted.values()):
        msg = "Target-magnitude error trends require labelled non-empty record groups."
        raise ValueError(msg)
    styles = _record_styles(admitted, styles)
    figure, axes = figure_layout.panel_figure(len(fields), 1, panel_in=(7.0, 3.6))
    # Per series in the legend's order, its fields' excluded values: raw non-finite apart from display-masked water.
    excluded: dict[str, list[str]] = {label: [] for label in admitted}
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        for label, records in admitted.items():
            centers, medians, diverging, masked, total = _finite_error_trend(records, field)
            style = styles[label]
            axis.plot(centers, medians, color=style.color, linestyle=style.linestyle, marker=style.marker or "o", linewidth=2.0, label=label)
            parts = [f"{count} {kind}" for count, kind in ((diverging, "non-finite"), (masked, "negative water masked")) if count]
            if parts:  # the medians describe finite errors; the exact counts of excluded values are named, never hidden
                excluded[label].append(f"{_display_symbol(field)} {', '.join(parts)} of {total}")
        axis.set_xlabel(f"Reference value — {_display_label(field)}")  # the parity plot's form, so the label keeps its own capital
        axis.set_ylabel(f"Median absolute error [{_display_unit(field, difference=True)}]")
        axis.grid(alpha=0.25)
        _at_least_zero(axis)
    datasets = {label: _dataset_label(_dataset_key(records)) for label, records in admitted.items()}
    runs = _series_runs(datasets)
    # One line per run, each of its datasets with its own counts, a series without a run alone; the legend's run order holds
    # before a run without an excluded value drops out.
    lines: dict[str, list[str]] = {runs.get(label, label): [] for label in excluded}
    for label, parts in excluded.items():
        if parts:
            lines[runs.get(label, label)].append(f"{datasets[label]} ({'; '.join(parts)})" if label in runs else "; ".join(parts))
    lines = {heading: entries for heading, entries in lines.items() if entries}
    width_in = 0.0
    if lines:  # the layout wraps no line with a symbol, so these wrap here, to the panel's final width
        figure.draw_without_rendering()
        width_in = axes[0, 0].get_window_extent().width / figure.dpi
    notes = [_wrapped(figure, f"{heading}: {'; '.join(entries)}", width_in) for heading, entries in lines.items()]
    intro = "all stored post-origin times; deterministic bounded bins" + ("; excluded values:" if notes else "")
    note = plot_layout.series_note(styles, intro, *notes)
    figure_layout.set_figure_title(figure, axes, "Error vs reference value" + plot_layout.models_suffix(models), note=note)
    _legend(figure, axes, runs)
    return figure


def maximum_error_physical_time(
    record: Any,
    *,
    state_fields: Sequence[str],
) -> float:
    """Return the stored post-origin time with largest scale-normalized field error."""
    fields = _selected_fields(state_fields)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim != _STATE_ARRAY_RANK
        or times.shape != (reference.shape[0],)
        or len(times) < _MINIMUM_TRAJECTORY_TIMES
    ):
        msg = "Error-maximizing time requires aligned non-empty transient evidence."
        raise ValueError(msg)
    scores = np.zeros(len(times) - 1, dtype=np.float64)
    for field in fields:
        if field == _GRAIN_MOISTURE_FIELD:
            reference_values = _grain_moisture(record, reference)
            prediction_values = _grain_moisture(record, prediction)
        else:
            channel = artifact.STATE_ORDER.index(field)
            reference_values = reference[:, channel]
            prediction_values = prediction[:, channel]
        selected_error = prediction_values - reference_values
        scale = max(float(np.std(reference_values, dtype=np.float64)), np.finfo(np.float64).eps)
        scores += np.sqrt(np.mean(np.square(selected_error[1:], dtype=np.float64), axis=(1, 2))) / scale
    # A non-finite score is a diverging prediction and counts as the worst time.
    return float(times[1 + int(np.argmax(np.where(np.isfinite(scores), scores, np.inf)))])


def _run_values(groups: Mapping[str, pd.DataFrame], columns: Sequence[str]) -> dict[tuple[str, str], list[np.ndarray]]:
    """Return every run's (model kind, checkpoint) plotted groups: the values of each of ``columns`` in each of its series (``_summary_series``)."""
    values: dict[tuple[str, str], list[np.ndarray]] = {}
    for rows in groups.values():
        run = (str(rows["model_kind"].iloc[0]), str(rows["checkpoint_sha256"].iloc[0]))
        values.setdefault(run, []).extend(rows[column].to_numpy(dtype=np.float64) for column in columns)
    return values


def _log_if_wide(axis: Axes, values: Mapping[Any, Sequence[np.ndarray]], *, sides: Sequence[Literal["x", "y"]] = ("y",)) -> str | None:
    """
    Put the ``sides`` of ``axis`` on a labelled log scale when a run's largest value exceeds ``_WIDE_RANGE`` times every other run's medians.

    ``values`` holds each run's plotted groups (a box, a bar, a series of points); another run's median is the largest median of
    its groups, never the median of its pooled values. The returned note says so and counts the non-positive values the log axis
    cannot show, never clipping them silently. Without another run with a value, or below the ratio, nothing changes (``None``).
    """
    finite = {run: [group[np.isfinite(group)] for group in groups] for run, groups in values.items()}
    finite = {run: shown for run, groups in finite.items() if (shown := [group for group in groups if group.size])}
    if len(finite) < _SEVERAL_RUNS:
        return None
    medians = {run: max(float(np.median(group)) for group in groups) for run, groups in finite.items()}
    largest = {run: max(float(group.max()) for group in groups) for run, groups in finite.items()}
    if not any(largest[run] > _WIDE_RANGE * max(median for other, median in medians.items() if other != run) for run in finite):
        return None
    for side in sides:
        figure_layout.set_log_scale(axis, side)
        label = axis.yaxis.get_label() if side == "y" else axis.xaxis.get_label()
        label.set_text(f"{label.get_text()} (log scale)")
    hidden = sum(int(np.count_nonzero(group <= 0.0)) for groups in finite.values() for group in groups)
    note = f"log scale: one run exceeds {_WIDE_RANGE:g} times the largest group median of every other run"
    return note + (f"; {hidden} non-positive values cannot be shown on it" if hidden else "")


def _at_least_zero(axis: Axes) -> None:
    """Keep the lower limit of a linear non-negative error axis at zero or above, so no later widening shows negative errors."""
    lower, upper = axis.get_ylim()
    if axis.get_yscale() == "linear" and lower < 0.0:
        axis.set_ylim(0.0, upper)


def _full_autonomous_cumulative(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the comparable complete-rollout cumulative rows."""
    required = {
        "frame",
        "material_family",
        "dataset_role",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame):
        msg = "Transient aggregate plot requires complete-rollout summary columns."
        raise ValueError(msg)
    selected = frame.loc[(frame["mode"] == "autonomous_full") & (frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")].copy()
    if selected.empty:
        msg = "Transient aggregate plot requires full-autonomous cumulative evidence."
        raise ValueError(msg)
    return selected


def plot_channel_error(
    summary_frame: pd.DataFrame,
    *,
    styles: series.SeriesStyles,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
    models: Sequence[str] = (),
    run_labels: Mapping[str, str] | None = None,
) -> Figure:
    """
    Compare complete-rollout normalized RMSE for selected stored states, one bar per frame and material in its comparison colour and role hatch.

    Where a run spans several materials, each bar carries its material's marker at its top. A frame of ``summary_frame``
    without a finite full-rollout error has no bar, and the note names it. ``run_labels`` name the runs as in ``_summary_series``.
    """
    selected = _full_autonomous_cumulative(summary_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient channel-error plots require at least one stored state."
        raise ValueError(msg)
    columns = [f"normalized_rmse_{field}" for field in fields]
    if any(column not in selected for column in columns):
        msg = "Transient channel-error plot lacks per-state normalized RMSE."
        raise ValueError(msg)
    groups = _summary_series(selected, run_labels)
    styles = _frame_styles(groups, styles)
    positions = np.arange(len(columns), dtype=np.float64)
    width = 0.8 / len(groups)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_GROUPED_PANEL_IN)
    axis = axes[0, 0]
    handles: list[Any] = []
    for offset, (label, rows) in enumerate(groups.items()):
        if len(rows) != 1:
            msg = f"Transient channel-error plot requires one summary row per frame and material; {label!r} has {len(rows)}."
            raise ValueError(msg)
        style = styles[label]
        values = rows[columns].to_numpy(dtype=np.float64)[0]
        centres = positions - 0.4 + width * (offset + 0.5)
        bars = axis.bar(centres, values, width=width, color=style.color, hatch=style.hatch, edgecolor="0.2", label=label)
        handles.append(bars)
        if style.marker is not None:  # the bars of one run's materials share colour and hatch: the material marker tells them apart
            marks = axis.scatter(centres, values, marker=style.marker, color="white", edgecolors="0.2", s=_SINGLETON_MARKER_SIZE, zorder=3)
            handles[-1] = (bars, marks)  # the legend key overlays the marker on the bar patch
    axis.set_xticks(positions, [field_labels.field_label(field, mathtext=True) for field in fields])
    axis.set_ylabel(f"Normalized RMSE [{field_labels.display_unit('1')}]")
    axis.grid(axis="y", alpha=0.25)
    wide = _log_if_wide(axis, _run_values(groups, columns))
    shown = set(selected.loc[np.asarray(np.isfinite(selected[columns].to_numpy(dtype=np.float64)).any(axis=1), dtype=bool), "frame"])
    missing = [f"{frame}: no finite full-rollout error" for frame in dict.fromkeys(summary_frame["frame"]) if frame not in shown]
    title = "Full-rollout per-state error" + plot_layout.models_suffix(models)
    note = plot_layout.series_note(styles, moisture_display.density_basis_note(fields), wide, *missing)
    figure_layout.set_figure_title(figure, axes, title, note=note)
    labels, legend_groups = _run_entries(handles, list(groups), _summary_runs(groups))
    figure_layout.add_legend(figure, axes, handles, labels, groups=legend_groups)
    return figure


def plot_horizon_error(
    summary_frame: pd.DataFrame, *, styles: series.SeriesStyles, models: Sequence[str] = (), run_labels: Mapping[str, str] | None = None
) -> Figure:
    """
    Plot rollout-index and physical-time views without conflating axes; ``run_labels`` name the runs as in ``_summary_series``.

    A frame and material keeps its comparison colour and role line style and takes its material's marker; the reduction scope is
    the marker fill (cumulative filled, endpoint open). Every requested horizon keeps its place, also one without a supported
    case; both panels break a line at a horizon the series lacks, the elapsed-time panel on padded limits with hour ticks that
    follow the span. A horizon without a case of any series says so under its tick, one that only some drawn series lack carries
    one grey x on the axis line, at no error value, under its own key. In the elapsed-time panel a full-horizon point, which ends
    each case at its own time, carries a keyed ring. A frame of ``summary_frame`` without a finite rolling-origin error, rows or
    not, has no line, and the note names it; the note also lists each frame and material's horizons without a case, a zero-case
    row or no row at all, and names one without a case at any horizon, which has no line either. Without a case of any series at
    any horizon both panels stay empty, every tick saying so; a summary without rolling-origin rows raises ``ValueError``.
    """
    required = {
        "frame",
        "material_family",
        "dataset_role",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
        "elapsed_physical_time_median",
        "contributing_record_count",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Horizon-error plot requires a transient dataset summary frame."
        raise ValueError(msg)
    # A material without records at a horizon has no error or elapsed time there; its missing cases show in the tables.
    rolling = summary_frame.loc[summary_frame["mode"] == "rolling_origin"]
    if rolling.empty:
        msg = "Horizon-error plot requires rolling-origin summaries."
        raise ValueError(msg)
    selected = rolling.loc[rolling["contributing_record_count"] > 0].copy()
    # Every requested horizon keeps its place, also one no series supports, so no line joins across it in either panel.
    horizons = tuple(sorted(rolling["requested_horizon"].unique(), key=horizon_order))
    horizon_positions = {value: position for position, value in enumerate(horizons)}
    elapsed_hours = selected["elapsed_physical_time_median"].to_numpy(dtype=np.float64)
    display_time = time_ui.physical_time_display(elapsed_hours if elapsed_hours.size else (0.0,), preferred_unit="auto")  # hours without a case
    figure, grid = figure_layout.panel_figure(1, 2, panel_in=(5.2, 4.2))
    axes = grid[0]
    groups = _summary_series(rolling, run_labels)  # every frame and material, also one without a case at any horizon
    styles = _frame_styles(groups, styles)
    lacking_by_horizon: dict[Any, list[str]] = {}  # each horizon's series without a case: a row of zero cases or none at all
    unsupported: list[str] = []  # series without a case at any horizon
    for series_label, held in groups.items():
        rows = held.loc[held["contributing_record_count"] > 0]
        if rows.empty:  # no line: the note names it
            unsupported.append(f"{series_label}: no case at any horizon")
            continue
        for value in horizons:
            if value not in set(rows["requested_horizon"]):
                lacking_by_horizon.setdefault(value, []).append(series_label)
        style = styles[series_label]
        marker = _material_marker(str(rows["material_family"].iloc[0]))
        for scope, group in rows.groupby("scope", sort=False):
            ordered = group.assign(_position=group["requested_horizon"].map(horizon_positions)).sort_values("_position")
            line = {"color": style.color, "linestyle": style.linestyle, "marker": marker}
            if str(scope) == "endpoint":
                line["markerfacecolor"] = "white"
            # A horizon between this series' first and last that it lacks breaks its line in both panels instead of joining across it.
            gaps = ordered.set_index("_position").reindex(range(int(ordered["_position"].min()), int(ordered["_position"].max()) + 1))
            errors = gaps["normalized_drying_group_macro_rmse"].to_numpy(dtype=np.float64)
            axes[0].plot(gaps.index, errors, **line, label=f"{series_label} — {scope}")
            elapsed = gaps["elapsed_physical_time_median"].to_numpy(dtype=np.float64) / display_time.divisor_hours
            axes[1].plot(elapsed, errors, **line)
            full = (gaps["requested_horizon"] == "full").to_numpy()
            if full.any():  # the full horizon ends each case at its own time, unlike a fixed number of transitions: a keyed ring marks it
                ring = {"s": 120, "facecolors": "none", "edgecolors": _NEUTRAL_COLOUR, "linewidths": 1.2}
                axes[1].scatter(elapsed[full], errors[full], **ring, label=_FULL_HORIZON)
    # A horizon no series supports says so under its tick; one some drawn series lack gets one grey x on the axis line, which
    # claims no error value; the note names each horizon's drawn series without a case once.
    supported = set(selected["requested_horizon"])
    axes[0].set_xticks(np.arange(len(horizons)), [str(value) if value in supported else f"{value}\nno case" for value in horizons])
    if partial := [horizon_positions[value] for value in horizons if value in supported and value in lacking_by_horizon]:
        axes[0].plot(
            partial,
            np.zeros(len(partial)),
            transform=axes[0].get_xaxis_transform(),
            linestyle="none",
            marker="x",
            color=_NEUTRAL_COLOUR,
            clip_on=False,
            zorder=3,
            label=_PARTIAL_HORIZON,
        )
    drawn = len(groups) - len(unsupported)
    unsupported.extend(
        f"no case at horizon {value}: " + ("every drawn series" if len(lacking) == drawn > 1 else ", ".join(lacking))
        for value in horizons
        if (lacking := lacking_by_horizon.get(value))
    )
    axes[0].set_xlabel("Requested horizon [transitions]")
    if elapsed_hours.size:
        display_time.configure(axes[1])
        axes[1].set_xlim(*_padded(*display_time.values((display_time.minimum_hours, display_time.maximum_hours))))
    else:  # without a case at any horizon both panels stay empty and claim no time span or error value
        axes[1].set_xticks([])
        for axis in axes:
            axis.set_yticks([])
    axes[1].set_xlabel(f"Median elapsed physical time [{display_time.unit}]")
    figure_layout.label_rows_once(grid, f"{_OBJECTIVE_LABEL} [{field_labels.display_unit('1')}]")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    shown = set(selected.loc[np.isfinite(selected["normalized_drying_group_macro_rmse"].to_numpy(dtype=np.float64)), "frame"])
    missing = [f"{frame}: no finite rolling-origin error" for frame in dict.fromkeys(summary_frame["frame"]) if frame not in shown]
    title = "Error vs rollout horizon" + plot_layout.models_suffix(models)
    figure_layout.set_figure_title(figure, grid, title, note=plot_layout.series_note(styles, *missing, *unsupported))
    _legend(figure, grid, _summary_runs(groups))
    return figure


def plot_endpoint_vs_cumulative(
    summary_frame: pd.DataFrame, *, styles: series.SeriesStyles, models: Sequence[str] = (), run_labels: Mapping[str, str] | None = None
) -> Figure:
    """
    Compare endpoint and cumulative central error without conflating scopes: a point per mode and horizon of a frame and material, in its style.

    A pair with an unavailable (a missing scope row included) or non-finite side is left out and counted in the note; a frame of
    ``summary_frame`` without a finite pair has no point, and the note names it. Both axes share one range, from zero on the
    linear scale. ``run_labels`` name the runs as in ``_summary_series``.
    """
    required = {
        "frame",
        "material_family",
        "dataset_role",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Endpoint comparison requires transient summary columns."
        raise ValueError(msg)
    groups = _summary_series(summary_frame, run_labels)
    # The reindex restores every pair and scope the pivot drops as all-NaN, so a pair without a finite side is counted, never lost unseen.
    keys = ["mode", "requested_horizon"]
    pairs = {
        label: rows.pivot_table(index=keys, columns="scope", values="normalized_drying_group_macro_rmse", aggfunc="first").reindex(
            index=pd.MultiIndex.from_frame(rows[keys].drop_duplicates()), columns=["cumulative", "endpoint"]
        )
        for label, rows in groups.items()
    }
    finite = {label: np.asarray(np.isfinite(pivot.to_numpy(dtype=np.float64)).all(axis=1), dtype=bool) for label, pivot in pairs.items()}
    left_out = sum(int(np.count_nonzero(~kept)) for kept in finite.values())  # a pair with an unavailable or non-finite side, never drawn or scaled
    pairs = {label: pairs[label].loc[kept] for label, kept in finite.items() if kept.any()}
    if not pairs:
        msg = "Endpoint comparison requires paired cumulative and endpoint evidence."
        raise ValueError(msg)
    styles = _frame_styles({label: groups[label] for label in pairs}, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(5.5, 5.0))
    axis = axes[0, 0]
    values: dict[tuple[str, str], list[np.ndarray]] = {}
    for label, pivot in pairs.items():
        style = styles[label]
        axis.scatter(pivot["cumulative"], pivot["endpoint"], color=style.color, marker=style.marker or "o", label=label)
        run = (str(groups[label]["model_kind"].iloc[0]), str(groups[label]["checkpoint_sha256"].iloc[0]))
        values.setdefault(run, []).append(pivot[["cumulative", "endpoint"]].to_numpy(dtype=np.float64).ravel())
    shown_values = np.concatenate([part for parts in values.values() for part in parts])
    axis.set_xlabel(f"Cumulative central error [{field_labels.display_unit('1')}]")
    axis.set_ylabel(f"Endpoint central error [{field_labels.display_unit('1')}]")
    wide = _log_if_wide(axis, values, sides=("x", "y"))
    positive = shown_values[shown_values > 0.0]
    lower = float(positive.min()) if wide and positive.size else 0.0  # a log diagonal starts at the smallest positive value
    maximum = float(np.max(shown_values))
    axis.plot((lower, maximum), (lower, maximum), linestyle="--", color="black", gid=figure_layout.REFERENCE_GID)
    # Equal limits on either scale keep the diagonal the parity line, the linear ones from zero, so a legend fit widens both alike.
    limits = (lower / _LOG_PAD, maximum * _LOG_PAD) if wide else (0.0, _padded(0.0, maximum)[1])
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.grid(alpha=0.25)
    shown = {groups[label]["frame"].iloc[0] for label in pairs}
    missing = [f"{frame}: no paired endpoint and cumulative evidence" for frame in dict.fromkeys(summary_frame["frame"]) if frame not in shown]
    dropped = f"{left_out} pair{'s' if left_out > 1 else ''} with an unavailable or non-finite error not shown" if left_out else None
    note = plot_layout.series_note(styles, wide, dropped, *missing)
    figure_layout.set_figure_title(figure, axes, "Endpoint vs cumulative error" + plot_layout.models_suffix(models), note=note)
    _legend(figure, axes, _summary_runs({label: groups[label] for label in pairs}))
    return figure


def plot_target_time(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    styles: series.SeriesStyles,
    models: Sequence[str] = (),
    run_labels: Mapping[str, str] | None = None,
) -> Figure:
    """
    Plot target pairs by saved model and material, each in its comparison style by its records' identity and its material's marker.

    ``records`` maps a frame to its records; a series reads as its material and role (``Chickpea, ID``), after the run of
    ``run_labels`` (by frame, else the frame) where several runs are drawn. The style is resolved per model and material, so a
    single run's points take their material colours and several runs' their run shades. A reached pair is a filled marker at its
    two times to target, in hours like every time axis. A censored pair (both sides available, one or both without reaching the
    target) is an open marker at the bounds it has: a side that did not reach the target stands at its final time, a lower bound
    marked by a grey arrow toward later times on that side's axis (right for the reference, up for the prediction), so neither a
    bound nor two equal bounds read as observed times; a censored pair without a final time is only counted. Both axes share
    padded limits, and a panel without any reached pair says so. The note keeps every frame's reached, censored and unavailable
    counts, also of a frame without a drawn pair.
    """
    groups = {str(label): tuple(selected) for label, selected in (records if isinstance(records, Mapping) else {"": records}).items()}
    runs = {label: (run_labels or {}).get(label, label) for label in groups}
    several = len(set(runs.values())) > 1
    by_material: dict[str, list[Any]] = {}  # the records of each model and material, the series a style is resolved for
    points: dict[tuple[str, str, bool], list[tuple[float, float]]] = {}  # (series, material, censored): (reference, prediction) hours
    bounded: dict[str, list[tuple[float, float]]] = {side: [] for side, _direction in _BOUND_ARROWS}  # the pairs whose side is a bound
    notices = []
    datasets: dict[str, str] = {}  # each series' dataset label, which its name ends with
    for label, selected in groups.items():
        reached = censored = unavailable = 0
        for record in sorted(selected, key=lambda held: _material_label(_record_material(held))):  # the views' material order
            material = _record_material(record)
            dataset = _dataset_label({(material, str(_value(record, "identity").get("dataset_role")))})
            name = f"{runs[label]} · {dataset}" if several else dataset
            datasets[name] = dataset
            by_material.setdefault(name, []).append(record)
            target = _value(record, "target")
            if not isinstance(target, Mapping):
                msg = "Transient target plot requires mapping evidence."
                raise TypeError(msg)
            if target.get("reference_available") is not True or target.get("predicted_available") is not True:
                unavailable += 1
                continue
            sides = ("reference", "predicted")
            times = [target.get(f"{side}_time_to_target") for side in sides]
            pair_reached = all(isinstance(time, Real) for time in times)
            reached += pair_reached
            censored += not pair_reached
            reference_value, predicted_value = (
                time if isinstance(time, Real) else target.get(f"{side}_final_time") for time, side in zip(times, sides, strict=True)
            )
            if isinstance(reference_value, Real) and isinstance(predicted_value, Real):
                pair = (float(reference_value), float(predicted_value))
                points.setdefault((name, material, not pair_reached), []).append(pair)
                for (side, _direction), time in zip(_BOUND_ARROWS, times, strict=True):  # reference, then prediction, as ``sides``
                    if not isinstance(time, Real):
                        bounded[side].append(pair)
        role = display_labels.evaluation_role_display_label(str(_value(selected[0], "identity").get("dataset_role"))) if selected else label
        notices.append(f"{f'{runs[label]} · ' if several else ''}{role}: paired reached {reached}, censored {censored}, unavailable {unavailable}")
    styles = _record_styles(by_material, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(6.5, 6.0))
    axis = axes[0, 0]
    all_times = [time for pairs in points.values() for pair in pairs for time in pair]
    display_time = time_ui.physical_time_display(all_times or (0.0,), preferred_unit="h")
    names = list(by_material)  # each series' reached entry before its censored one, whichever case came first
    for (name, material, is_censored), pairs in sorted(points.items(), key=lambda item: (names.index(item[0][0]), item[0][2])):
        style = styles[name]
        axis.scatter(
            display_time.values(pair[0] for pair in pairs),
            display_time.values(pair[1] for pair in pairs),
            marker=_material_marker(material),
            facecolors="none" if is_censored else style.color,
            edgecolors=style.color,
            label=name + (" · censored, final time as bound" if is_censored else ""),
        )
    for (side, (towards_x, towards_y)), glyph in zip(_BOUND_ARROWS, (r"$\rightarrow$", r"$\uparrow$"), strict=True):
        if pairs := bounded[side]:
            x_values, y_values = display_time.values(pair[0] for pair in pairs), display_time.values(pair[1] for pair in pairs)
            arrows = {"angles": "uv", "scale_units": "dots", "scale": 1.0 / _BOUND_ARROW_PX, "units": "dots", "width": 1.5}
            axis.quiver(
                x_values, y_values, np.full(len(pairs), towards_x), np.full(len(pairs), towards_y), color=_NEUTRAL_COLOUR, zorder=0.9, **arrows
            )
            label = f"{side.capitalize()} not reached: final time, a lower bound"  # the arrow's legend key, under the open marker
            axis.plot([], [], color=_NEUTRAL_COLOUR, marker=glyph, markersize=10, linestyle="none", label=label)
    display_time.configure(axis)
    display_time.configure(axis, dimension="y")
    lower, upper = _padded(*display_time.values((display_time.minimum_hours, display_time.maximum_hours)))
    lower = max(lower, 0.0)  # equal limits, never below zero: a legend fit widens both axes alike and shows no negative time
    if all_times:
        axis.plot((lower, upper), (lower, upper), linestyle="--", color="black", gid=figure_layout.REFERENCE_GID)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    if not any(not is_censored for _label, _material, is_censored in points):
        axis.text(0.5, 0.5, "No case pair reached the target", transform=axis.transAxes, ha="center", va="center")
    axis.set_xlabel(f"Reference time to target [{display_time.unit}]")
    axis.set_ylabel(f"Predicted time to target [{display_time.unit}]")
    axis.grid(alpha=0.25)
    figure_layout.set_figure_title(
        figure, axes, "Time to target" + plot_layout.models_suffix(models), note=plot_layout.series_note(styles, "\n".join(notices))
    )
    _legend(figure, axes, _series_runs(datasets))
    return figure


def plot_training_performance_vs_compute(
    performance: pd.DataFrame,
    *,
    lineage: comparison.LineageComparison,
    styles: series.SeriesStyles | None = None,
    run_labels: Mapping[str, str] | None = None,
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot the admitted A0/B/A1 arms with a result at their actual committed transitions, one panel per mode and horizon.

    A B or A1 without an eligible checkpoint has no rows; the note names its status, budget progress and eligible points, so A0
    alone still plots. The arms are one lineage evaluated on its ID artifacts: a point takes its material's marker and, with
    ``styles`` (the session's, by frame name) and rows of several runs (``frame``), its run's colour, else its material's colour.
    A thin grey connector joins A0 to each later arm of a material, never two arms at the same budget. The legend names each run
    once (by ``run_labels``, else the frame) beside a key of the material markers (``_run_material_legend``); a single run's points
    are named by material. The x axis is each arm's committed post-handoff transitions as persisted, and the title ends with ``models``.
    """
    unavailable = [arm for arm in (lineage.b, lineage.a1) if not isinstance(arm, comparison.TrainingArmEvidence)]
    required = {
        "comparison_arm",
        "material_family",
        "optimizer_device_compute",
        "normalized_drying_group_macro_rmse",
        "checkpoint_role",
        "mode",
        "requested_horizon",
        "scope",
    }
    if not isinstance(performance, pd.DataFrame) or not required.issubset(performance):
        msg = "Matched-compute plot requires arm, compute, metric, checkpoint, mode, horizon, and scope."
        raise ValueError(msg)
    arms = set(performance["comparison_arm"])
    if arms != {"A0", "A1", "B"}.difference(arm.arm for arm in unavailable):
        msg = "Matched-compute plot requires rows for exactly the A0/A1/B arms that have a result."
        raise ValueError(msg)
    matched_rows = performance["comparison_arm"].isin(("A1", "B"))
    clocks = set(performance.loc[matched_rows, "compute_clock"]) if "compute_clock" in performance else set()
    if matched_rows.any() and clocks != {"supervised_transitions"}:
        msg = "Matched A1/B plot requires persisted committed supervised transitions for every arm with a result."
        raise ValueError(msg)
    selected = performance.loc[performance["scope"] == "cumulative"].copy()
    if selected.empty:
        msg = "Matched-compute plot requires cumulative Evaluation evidence."
        raise ValueError(msg)
    spent, error = ("training_compute" if "training_compute" in selected else "optimizer_device_compute"), "normalized_drying_group_macro_rmse"
    modes = list(dict.fromkeys(selected["mode"]))
    conditions = sorted(  # by mode, then horizon ascending with «full» last, as every view orders them
        dict.fromkeys(zip(selected["mode"], selected["requested_horizon"], strict=True)),
        key=lambda item: (modes.index(item[0]), horizon_order(item[1])),
    )
    columns = min(len(conditions), _CONDITION_COLUMNS)
    figure, axes = figure_layout.panel_figure(math.ceil(len(conditions) / columns), columns, panel_in=(4.2, 3.4))
    run_styles = styles if styles is not None and "frame" in selected and len(set(selected["frame"])) > 1 else None
    materials = sorted(set(selected["material_family"]), key=lambda material: dataset_order(material, "id"))  # the arms are ID artifacts
    single = series.resolve_styles([series.SeriesEntity(material, "", False, "id", (material,), "lineage") for material in materials])
    points: dict[str, tuple[str, str, str, Any]] = {}
    for index, (mode, horizon) in enumerate(conditions):
        axis = axes.flat[index]
        for material in materials:
            group = selected.loc[(selected["mode"] == mode) & (selected["requested_horizon"] == horizon) & (selected["material_family"] == material)]
            if group["comparison_arm"].duplicated().any():
                msg = "Matched-compute series requires at most one value per arm, material, mode, and horizon."
                raise ValueError(msg)
            arms = group.set_index("comparison_arm")
            for later in arms.index.intersection(["A1", "B"]) if "A0" in arms.index else ():  # lineage A0 to each later arm, never A1 to B
                if arms.loc["A0", spent] != arms.loc[later, spent]:  # arms at the same budget stay unjoined
                    axis.plot(arms.loc[["A0", later], spent], arms.loc[["A0", later], error], color=_NEUTRAL_COLOUR, linewidth=0.8, zorder=1)
            for frame, row in arms.iterrows() if run_styles is None else arms.set_index("frame").iterrows():
                run = _material_label(material) if run_styles is None else (run_labels or {}).get(str(frame), str(frame))
                colour = single[material].color if run_styles is None else run_styles[str(frame)].color
                axis.scatter(row[spent], row[error], color=colour, marker=_material_marker(material), zorder=3)
                points[run if run_styles is None else f"{run} · {_material_label(material)}"] = (run, material, "id", colour)
        mode_label = display_labels.evaluation_mode_display_label(str(mode))
        figure_layout.set_panel_title(axis, mode_label if mode != "rolling_origin" else f"{mode_label}, horizon {horizon}")
        if index + columns >= len(conditions):  # the lowest panel of its column
            axis.set_xlabel(f"Committed post-handoff supervised transitions [{field_labels.display_unit('1')}]")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(conditions) :]:
        axis.set_visible(False)
    figure_layout.label_rows_once(axes, f"{display_labels.qualified_label('Cumulative', _OBJECTIVE_LABEL)} [{field_labels.display_unit('1')}]")
    status = [
        f"{arm.arm}: {display_labels.token_display_label(arm.training_status)} at {100.0 * arm.budget_progress:.0f} % of its budget, "
        f"{arm.eligible_points} eligible points, no result"
        for arm in unavailable
    ]
    title = "Shared-A0 performance vs committed budget" + plot_layout.models_suffix(models)
    note = "\n".join(status) if run_styles is None else plot_layout.series_note(run_styles, *status)
    figure_layout.set_figure_title(figure, axes, title, note=note or None)
    _run_material_legend(figure, axes, points)
    return figure


def unavailable_reason_text(reason: str | None) -> str:
    """Return the display text of a recorded unavailability reason: each ``;``-separated identifier token in words, recorded prose as written."""
    parts = [part.strip() for part in (reason or "").split(";") if part.strip()]
    return "; ".join(display_labels.token_display_label(part) if _REASON_TOKEN.fullmatch(part) else part for part in parts) or "no reason recorded"


def plot_pipeline_degradation(
    evidence: Sequence[comparison.AirflowDegradationMetrics],
    *,
    role: str | None = None,
    models: Sequence[str] = (),
    labels: Sequence[str] = (),
    styles: Sequence[visual_semantics.SeriesStyle] = (),
    unavailable: Sequence[str] = (),
) -> Figure:
    """
    Plot the errors of B against A, C against A and C against B as narrow bars, one panel per evidence, titled by ``labels``.

    A, B and C are the pipeline conditions, not the training stages: the note defines them, names the scope (the complete
    autonomous rollouts of the ``role`` cases, cumulative, reduced as checkpoint selection does) and each evidence's upstream
    airflow error, and names an unavailable C with its reason as display text, each distinct line once (without names where every
    panel shares it). A panel's bars take its series style of ``styles``, colour and role hatch (plain grey without), so the
    stages are named only by their tick labels; only a stage with a value is a bar, so an unavailable stage has neither bar nor
    key. All panels share one value scale, so their heights compare. ``unavailable`` names each selected run without evidence
    and why, so every compared run stays named. The y axis names the metric by its shared display label; the title ends with
    ``models``.
    """
    values = tuple(evidence)
    if not values:
        msg = "Pipeline degradation plot requires metric evidence."
        raise ValueError(msg)
    if (labels and len(labels) != len(values)) or (styles and len(styles) != len(values)):
        msg = "Pipeline degradation labels and styles need one entry per evidence."
        raise ValueError(msg)
    figure, axes = figure_layout.panel_figure(len(values), 1, panel_in=(5.0, 3.2), sharey="all")  # one value scale: heights compare
    cases = "" if role is None else f" of the {display_labels.evaluation_role_display_label(role)} cases"
    notes = [_PIPELINE_CONDITIONS, f"Complete autonomous rollouts{cases}, cumulative"]
    lines: tuple[dict[str, list[str]], ...] = ({}, {})  # each distinct upstream error, then C reason, with the panels it holds for
    for number, (axis, item) in enumerate(zip(axes[:, 0], values, strict=True), start=1):
        heights = {"B/A": item.drying_surrogate_error, "C/A": item.complete_pipeline_error, "C/B": item.airflow_substitution_discrepancy}
        drawn = [stage for stage, height in heights.items() if height is not None]
        look = {"color": styles[number - 1].color, "hatch": styles[number - 1].hatch} if styles else {"color": _NEUTRAL_COLOUR}
        axis.bar(range(len(drawn)), [heights[stage] for stage in drawn], width=_STAGE_BAR_WIDTH, edgecolor="0.2", **look)
        axis.set_xticks(range(len(drawn)), [_STAGE_LABELS[stage] for stage in drawn])
        axis.set_xlim(-0.75, len(drawn) - 0.25)  # a narrow bar with room beside it, however few stages have a value
        axis.set_ylabel(f"{display_labels.evaluation_protocol_display_label(item.metric_id)} [{field_labels.display_unit('1')}]")
        if labels:
            figure_layout.set_panel_title(axis, labels[number - 1])
        upstream = "unavailable" if item.upstream_airflow_error is None else plot_layout.significant(item.upstream_airflow_error)
        name = labels[number - 1] if labels else f"Panel {number}"
        lines[0].setdefault(f"upstream airflow error {upstream}", []).append(name)
        if not item.c_available:
            lines[1].setdefault(f"C unavailable: {unavailable_reason_text(item.c_unavailable_reason)}", []).append(name)
    # Each distinct line stands once: without names where every panel shares it, else after the panels it holds for.
    notes.extend(
        line[:1].upper() + line[1:] if len(names) == len(values) else f"{', '.join(names)}: {line}"
        for group in lines
        for line, names in group.items()
    )
    notes.extend(unavailable)
    figure_layout.set_figure_title(figure, axes, "Airflow substitution evidence" + plot_layout.models_suffix(models), note="\n".join(notes))
    return figure


def plot_timing_speedups(
    report: timing.TransientTimingReport, *, role: str | None = None, models: Sequence[str] = (), others: str | None = None
) -> Figure:
    """
    Plot every component-composed duration ratio as one neutral point on a log scale, a formula without paired cases as a note.

    The point is the median of the paired per-case ratios and its whisker spans their 10th to 90th percentile, both the timing
    report's own statistics (NumPy's linear quantile); a point has no baseline, so no length suggests a size on the log scale. A
    row's short label names the ratio in sentence case and its paired count; the note under the title gives each ratio's ratio
    of the summed durations in one line, written positionally (the panel's status names the full boundaries). A ratio whose
    median or percentiles are not all positive and finite has no place on the log scale and is a note as well; without any
    drawn ratio the scale stays linear. Every row keeps half a row of margin, also one without a point. The title names the
    report's Dataset role (``role``) and ends with the short label ``models`` of its run (``layout.models_suffix``);
    ``others``, the caller's line naming the other selected runs with their paired ratios, ends the note, so the figure
    carries them wherever it goes.
    """
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Timing plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    summaries = [report.speedups[name] for name in report.speedups]
    names = [display_labels.token_display_label(summary.name.removesuffix(_RATIO_TICK_SUFFIX)) for summary in summaries]
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(7.0, 0.6 * len(summaries) + 1.0))
    axis = axes[0, 0]
    drawn = False
    for position, summary in enumerate(summaries):
        lower, median, upper = summary.q10, summary.median, summary.q90
        if lower is None or median is None or upper is None:
            note = "no paired case"
        elif not all(math.isfinite(value) and value > 0.0 for value in (lower, median, upper)):
            note = "no positive finite ratio"
        else:
            axis.errorbar(median, position, xerr=[[median - lower], [upper - median]], fmt="o", color=_NEUTRAL_COLOUR, capsize=4.0, elinewidth=1.0)
            drawn = True
            continue
        axis.text(0.02, position, note, transform=axis.get_yaxis_transform(), va="center", fontsize="small")
    axis.set_yticks(
        range(len(summaries)),
        [f"{name}\n{summary.available_count}/{summary.total_count} paired" for name, summary in zip(names, summaries, strict=True)],
        fontsize="small",
    )
    axis.set_ylim(len(summaries) - 0.5, -0.5)  # top to bottom, half a row of margin around every row, with or without its bar
    if drawn:  # every drawn median and percentile is positive and finite
        figure_layout.set_log_scale(axis, "x")
    axis.set_xlabel(f"Duration ratio, median and 10th-90th percentile of the paired cases [{field_labels.display_unit('1')}]")
    axis.grid(axis="x", which="both", alpha=0.25)
    sums = "; ".join(
        f"{name} {plot_layout.significant(summary.ratio_of_sums)}"
        for name, summary in zip(names, summaries, strict=True)
        if summary.ratio_of_sums is not None
    )
    scope = "" if role is None else f" · {display_labels.evaluation_role_display_label(role)} cases"
    note = "\n".join(filter(None, (f"Ratio of summed durations: {sums}" if sums else None, others)))  # the other runs, when given, end it
    figure_layout.set_figure_title(figure, axes, f"Boundary-specific duration ratios{scope}{plot_layout.models_suffix(models)}", note=note)
    return figure


def _case_accuracy_values(accuracy: pd.DataFrame, *, label: str) -> Mapping[str, float]:
    """Admit one finite non-negative central-accuracy value per case."""
    required = {"case_id", "normalized_drying_group_macro_rmse"}
    if not isinstance(accuracy, pd.DataFrame) or not required.issubset(accuracy):
        msg = f"{label} requires case identity and central metric."
        raise ValueError(msg)
    values: dict[str, float] = {}
    for case_id, value in zip(
        accuracy["case_id"],
        accuracy["normalized_drying_group_macro_rmse"],
        strict=True,
    ):
        if not isinstance(case_id, str) or not case_id or case_id in values:
            msg = f"{label} requires unique non-empty case identities."
            raise ValueError(msg)
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = f"{label} central metrics must be real scalars."
            raise TypeError(msg)
        admitted = float(value)
        if not np.isfinite(admitted) or admitted < 0.0:
            msg = f"{label} central metrics must be finite and non-negative."
            raise ValueError(msg)
        values[case_id] = admitted
    return values


def plot_accuracy_runtime_comparison(
    accuracy: pd.DataFrame,
    reports: Mapping[str, timing.TransientTimingReport],
    *,
    styles: series.SeriesStyles,
    measure: str = "prepared_wall",
    models: Sequence[str] = (),
    run_labels: Mapping[str, str] | None = None,
) -> Figure:
    """
    Compare saved models using exact paired case accuracy and a named measured clock, each series in its comparison style.

    A series is one frame's cases of one material, read as its material and role (``Chickpea, ID``), after the run of
    ``run_labels`` (by frame, else the frame) where several runs are drawn; a case keeps its material through the pairing, so its
    point takes the material's marker. The style is resolved per frame and material, so a single run's points take their
    material colours and several runs' their run shades; the legend names a run whose series share one shade once, over all its
    roles, beside a key of the material markers (``_run_material_legend``). The note counts each frame's paired cases by run and
    role and names a frame without any, which has no point.
    """
    labels = {
        "prepared_wall": "Prepared rollout request wall, warmed median [s]",
        "model": "Rollout model calls only, warmed median [s]",
        "ratio": "COMSOL reported drying solution / model-call time [ratio]",
    }
    if measure not in labels or "frame" not in accuracy:
        msg = "Accuracy/runtime comparison requires a named clock and frame identity."
        raise ValueError(msg)
    frames = {frame_name: accuracy.loc[accuracy["frame"] == frame_name] for frame_name in reports}
    runs = {frame_name: (run_labels or {}).get(frame_name, frame_name) for frame_name in reports}
    several = len(set(runs.values())) > 1
    by_material: dict[str, pd.DataFrame] = {}
    names: dict[tuple[str, str], str] = {}  # each frame's series of one material
    keys: dict[str, str] = {}  # each frame as the legend and the note name it: its role, after its run where several runs are drawn
    for frame_name, frame in frames.items():
        _require_provenance(frame_name, frame)
        role = str(frame["dataset_role"].iloc[0]) if len(frame) else None
        keys[frame_name] = " · ".join(
            part
            for part in (runs[frame_name] if several or role is None else "", display_labels.evaluation_role_display_label(role) if role else "")
            if part
        )
        for material, rows in sorted(frame.groupby("material_family"), key=lambda item: str(item[0])):  # the selector's material order
            dataset = display_labels.material_role_display_label(str(material), str(role))
            names[frame_name, str(material)] = f"{runs[frame_name]} · {dataset}" if several else dataset
            by_material[names[frame_name, str(material)]] = rows
    styles = _frame_styles(by_material, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(8.0, 5.5))
    axis = axes[0, 0]
    notices = []
    points: dict[str, tuple[str, str, str, Any]] = {}
    for frame_name, report in reports.items():
        values = _case_accuracy_values(frames[frame_name], label=f"Accuracy/runtime comparison {frame_name}")
        durations: Mapping[str, float]
        if measure == "ratio":
            durations = {case.case_id: case.speedup for case in report.speedups["drying_only_solver_speedup"].cases if case.speedup is not None}
        else:
            component = "drying_no_prepared_rollout_wall_seconds" if measure == "prepared_wall" else "drying_no_rollout_model_seconds"
            durations = timing.component_case_medians(report, component)
        materials = dict(zip(frames[frame_name]["case_id"], frames[frame_name]["material_family"], strict=True))
        paired = [(durations[case_id], value, str(materials[case_id])) for case_id, value in values.items() if case_id in durations]
        for material in sorted({point[2] for point in paired}):  # the dataset selector's material order
            shown = [point for point in paired if point[2] == material]
            name = names[frame_name, material]
            axis.scatter([point[0] for point in shown], [point[1] for point in shown], color=styles[name].color, marker=_material_marker(material))
            role = str(frames[frame_name]["dataset_role"].iloc[0])  # a frame holds one role
            points[name] = (runs[frame_name] if several else keys[frame_name], material, role, styles[name].color)  # a run once, whatever its roles
        notices.append(
            f"{keys[frame_name]}: {len(paired)}/{len(values)} cases paired"
            if paired
            else f"{keys[frame_name]}: no paired accuracy and runtime evidence"
        )
    axis.set_xlabel(labels[measure])
    axis.set_ylabel(f"{display_labels.qualified_label('Cumulative full-rollout', _OBJECTIVE_LABEL)} [{field_labels.display_unit('1')}]")
    if not axis.collections:
        axis.text(0.5, 0.5, "No paired saved accuracy/runtime evidence", transform=axis.transAxes, ha="center")
    axis.grid(alpha=0.25)
    figure_layout.set_figure_title(
        figure, axes, "Accuracy vs measured runtime" + plot_layout.models_suffix(models), note=plot_layout.series_note(styles, "\n".join(notices))
    )
    _run_material_legend(figure, axes, points)
    return figure
