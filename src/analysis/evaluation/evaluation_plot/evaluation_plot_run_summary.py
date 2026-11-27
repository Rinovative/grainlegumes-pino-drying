"""Summarize authoritative model accuracy, architecture, and physical tradeoffs.

Responsibilities:
  - Build provenance-backed run, runtime and architecture summary tables
  - Build the work table of completed stages from their summaries and their artifacts' inference times
  - Read normalized_group_macro_rmse only from validated aggregate evidence
  - Plot comparison-set-relative task and output-group diagnostics
  - Compare exact trainable capacity with authoritative aggregate accuracy
  - Plot accuracy against each separately named physical diagnostic, with a marker key per model, one per material shape, and
    padded axes
  - Show the tables of both tasks with display headings and names (``display_table``); the builders keep raw keys and tokens
  - Show the configuration with one spectral-modes row, tokens as names and an absent value as a dash or no row, and the work
    counts as whole numbers

Design principles:
  - Per-case quantiles remain secondary and never reconstruct the objective
  - Only error columns are shaded, each against its own values; identities, counts, work and configuration values rank nothing
  - Comparison-relative normalization is explicitly descriptive, not a ranking
  - A pooled dataset weights its materials equally and keeps each material visible as a marker beside the pooled value
  - A finite value whose ratio is undefined (displayed best zero or unavailable) is named in the note with its raw value, never dropped
  - Exact parameter counts and architecture values come only from provenance
  - An exact Dataset identity reads short in its group and whole among the runtime evidence
  - Incompatible physical diagnostics retain distinct axes, names, and units

This module does NOT:
  - Infer model facts from run names, paths, or hidden configuration defaults
  - Invent parameter-efficiency or cross-diagnostic composite scores
  - Admit incompatible frames or serialize tracking media
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_run_discovery as run_discovery
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_tables as tables

if TYPE_CHECKING:
    import ipywidgets as widgets
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_datasets as dataset_cuts
    from src.analysis.evaluation import evaluation_transient_timing as transient_timing

_PHYSICS_LABELS = {
    "momentum_residual_mse": (r"momentum mean($R_x^2+R_y^2$)", "1"),
    "flux_balance_mse": (r"flux balance mean($((q-\bar q)/Q)^2$)", "1"),
    "pressure_boundary_mse": (r"pressure boundary diagnostic$/\Delta p^2$", "1"),
}
_PHYSICS_TITLES = {"momentum_residual_mse": "Momentum residual", "flux_balance_mse": "Flux balance", "pressure_boundary_mse": "Pressure boundary"}
_MODE_AXES: Final = frozenset({"modes_x", "modes_y"})  # sorted, they give the x and then the y modes
_PARETO_ERROR_TICKS = 4
_PARETO_PAD = 0.12  # of each axis's (transformed) range on each side, so no point sits on the frame
_MINIMUM_SCOREBOARD_MODELS = 2
_SPATIAL_PAIR_LENGTH = 2
_FAMILY_LABELS: Final = {spec.family: spec.label for spec in visual_semantics.ARCHITECTURES}
_CONFIGURATION_TOKENS: Final = (  # configuration rows whose stored values are maintained tokens
    "Architecture · Channel-MLP skip",
    "Architecture · Factorization",
    "Architecture · FNO skip",
    "Architecture · Implementation",
    "Optimization · Data loss",
    "Optimization · Optimizer",
    "Optimization · Scheduler",
    "Physics-informed loss · Derivative operator",
    "Runtime/evidence · Requested device",
)
# The tokens of these columns read as their display names in a shown table (``display_table``).
_VALUE_NAMES: Final[Mapping[str, Callable[[str], str]]] = {
    "dataset_role": display_labels.evaluation_role_display_label,
    "material_family": display_labels.material_display_label,
    "mode": display_labels.evaluation_mode_display_label,
    **dict.fromkeys(("architecture_family", "model_kind"), lambda kind: _FAMILY_LABELS.get(kind, kind.upper())),
    **dict.fromkeys(
        ("budget_unit", "device", "stage", "unit_kind", "status", "time_coverage", *_CONFIGURATION_TOKENS), display_labels.token_display_label
    ),
}
_CHANNEL_FIELDS: Final = (("in_channels", "Data · Input channels"), ("out_channels", "Data · Output channels"))
_ERROR_PREFIXES: Final = ("normalized_", "physical_", *dataframe.STEADY_PHYSICS_METRICS)  # the summary's error columns, lower is better


def _provenance(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Return validated comparison provenance."""
    return dataframe.require_complete_provenance(frame)


def _finite_quantile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    """Return one per-case metric quantile over the finite cases, or NaN (never plotted) when no case is finite."""
    values = validity_contract.finite_case_values(pd.to_numeric(frame[column], errors="raise"))
    return float(np.quantile(values, quantile)) if values.size else float("nan")


def _optional_value(value: float | None) -> float | str:
    """Return a saved value, or ``unavailable`` (never zero) when its evidence is missing."""
    return "unavailable" if value is None else float(value)


def _display_value(value: object, names: Callable[[str], str] | None, absent: str = "") -> object:
    """Return one table cell as shown: a missing value as ``absent``, a flag as Yes or No, a lower-case token of a named column as its name."""
    if value is None or (isinstance(value, float | np.floating) and np.isnan(value)):
        return absent
    if isinstance(value, bool | np.bool_):
        return "Yes" if value else "No"
    if names is not None and isinstance(value, str) and value.islower() and value != run_discovery.UNAVAILABLE:
        return names(value)
    return value


def display_table(table: pd.DataFrame, *, absent: str = "") -> pd.DataFrame:
    """
    Return a copy of ``table`` as a view shows it, whose builders keep the raw keys and tokens.

    The headings read as ``display_labels.table_heading`` writes them, a role, material, mode, model-family, work-unit or status
    token as its display name, a flag as Yes or No and a missing value as ``absent`` (blank); numbers, counts, the ``unavailable``
    marker and every other value stay exact.
    """
    shown = table.copy()
    for column in shown.columns:
        shown[column] = shown[column].map(partial(_display_value, names=_VALUE_NAMES.get(str(column)), absent=absent))
    return shown.rename(columns=lambda column: display_labels.table_heading(str(column)))


def build_run_summary_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build one combined run summary from authoritative artifact evidence.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Labelled current artifacts. Runs inside each ID/OOD role must pass the
        shared comparison contract.

    Returns
    -------
    pandas.DataFrame
        One row per labelled artifact role. ``normalized_group_macro_rmse`` and global
        field RMSE use exact SSE/count evidence. Physical field errors and physics
        diagnostics remain separate unit-labelled quantiles.

    Raises
    ------
    ComparisonCompatibilityError, KeyError, TypeError, ValueError
        If provenance, comparison identity, model facts, aggregate evidence, or
        finite per-case metrics are incomplete.

    Notes
    -----
    The table includes run/config/checkpoint identity, exact trainable parameters,
    and physics enablement without deriving a score.

    """
    dataframe.validate_comparison(datasets)
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        provenance = _provenance(frame)
        run = provenance["run"]
        dataset = provenance["dataset"]
        aggregate = frame.attrs["normalized_group_macro_rmse"]
        unavailable = None if aggregate["available"] else validity_contract.unavailable_text(aggregate)
        fields = tuple(frame.attrs["output_fields"])
        units = dataframe.field_units(frame)
        family, trainable, physics_enabled = series.model_metadata(frame)
        row: dict[str, Any] = {
            "label": label,
            "dataset_role": dataframe.dataset_role(frame),
            "task_id": frame.attrs["task_id"],
            "run_name": run["name"],
            "architecture_family": _family_token(family, physics_enabled=physics_enabled),
            "trainable_parameters": trainable,
            "physics_enabled": physics_enabled,
            "normalized_group_macro_rmse": unavailable or float(aggregate["value"]),
            "dataset_name": dataset["name"],
            "sample_count": len(frame),
            "config_digest": str(run["effective_config_digest"]),
            "checkpoint_digest": str(run["best_checkpoint_sha256"]),
        }
        field_metrics, unavailable_cases = aggregate["field_metrics"], aggregate["relative_unavailable_cases"]
        for field in fields:
            row[f"normalized_rmse_{field}"] = unavailable or float(field_metrics[f"normalized_rmse_{field}"])
            row[f"physical_mae_{field} [{units[field]}]"] = unavailable or float(field_metrics[f"physical_mae_{field}"])
            row[f"physical_rmse_{field}_median [{units[field]}]"] = _finite_quantile(frame, f"physical_rmse_{field}", 0.5)
            row[f"physical_rmse_{field}_q90 [{units[field]}]"] = _finite_quantile(frame, f"physical_rmse_{field}", 0.9)
        # A relative norm is unavailable, never zero, when a material has no case whose reference reaches the floor.
        for norm in ("l2", "h1"):
            for metric_id in (f"physical_relative_{norm}", *(f"physical_relative_{norm}_{field}" for field in fields)):
                row[metric_id] = unavailable or _optional_value(field_metrics[metric_id])
                row[f"{metric_id}_unavailable_cases"] = unavailable or int(unavailable_cases[metric_id])
        for group_id, group_fields in frame.attrs["output_groups"]:
            (group_unit,) = {units[field] for field in group_fields}  # a group error is one quantity: its fields share one unit
            for space, key in (("normalized", group_id), ("physical", f"{group_id} [{group_unit}]")):
                row[f"{space}_group_rmse_{key}"] = unavailable or float(aggregate["group_statistics"][group_id][f"{space}_rmse"])
        if "physical_rmse_speed_magnitude" in frame.columns:
            row["physical_rmse_speed_magnitude_median [m/s]"] = _finite_quantile(
                frame,
                "physical_rmse_speed_magnitude",
                0.5,
            )
            row["physical_rmse_speed_magnitude_q90 [m/s]"] = _finite_quantile(
                frame,
                "physical_rmse_speed_magnitude",
                0.9,
            )
        for metric in dataframe.STEADY_PHYSICS_METRICS:
            if metric in frame.columns:
                unit = field_labels.display_unit(_PHYSICS_LABELS[metric][1])
                row[f"{metric}_median [{unit}]"] = _finite_quantile(frame, metric, 0.5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def plot_run_summary_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Show current authoritative metrics, each error column shaded against its own values (lower is better), identities and counts unshaded."""
    summary = build_run_summary_table(datasets)
    hidden_identity = {
        "task_id",
        "run_name",
        "dataset_name",
        "config_digest",
        "checkpoint_digest",
    }
    visible = summary[[column for column in summary.columns if column not in hidden_identity]]
    if not isinstance(visible, pd.DataFrame):
        msg = "Run-summary list-column selection must produce a DataFrame."
        raise TypeError(msg)
    title, count_headings = layout.aggregate_title_context(
        "Global summary",
        layout.effective_case_counts(datasets),
    )
    if any(count_headings.values()):
        visible = visible.rename(index={label: count_headings[label] for label in datasets})
    errors = [str(column) for column in visible.columns if str(column).startswith(_ERROR_PREFIXES) and not str(column).endswith("_cases")]
    shaded = [display_labels.table_heading(column) for column in errors]
    return tables.styled_dataframe(display_table(visible), title=title, shade_columns=shaded)


def runtime_hardware(report: Mapping[str, Any]) -> tuple[str, str]:
    """Return the surrogate and the solver hardware text of one steady runtime report, as the runtime table and the break-even name them."""
    runtime = report["neural_runtime"]
    surrogate = f"{runtime['resolved_device']} · {runtime.get('cuda_device_name') or runtime['processor']}"
    solver = "; ".join(f"{item['hostname']}, {item['requested_cores']} cores ({item['case_count']} cases)" for item in report["solver_hardware"])
    return surrogate, solver


def build_runtime_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build one row per labelled steady artifact from its admitted runtime comparison.

    Each speed-up is Σ COMSOL time / Σ batch-amortized forward time over the cases that have that COMSOL clock; the
    unpaired cases are counted, and a speed-up without any paired case reads ``unavailable``, never zero.
    """
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        report = _provenance(frame)["timing"]["report"]
        surrogate_hardware, solver_hardware = runtime_hardware(report)
        row: dict[str, Any] = {
            "label": label,
            "dataset_role": dataframe.dataset_role(frame),
            "batch_size": report["measurement"]["batch_size"],
            "surrogate_hardware": surrogate_hardware,
            "solver_hardware": solver_hardware,
        }
        for name, speedup in report["speedups"].items():
            row[f"{name} [{field_labels.display_unit('1')}]"] = _optional_value(speedup["ratio_of_sums"])
            row[f"{name} paired cases"] = speedup["paired_case_count"]
            row[f"{name} unavailable cases"] = speedup["unavailable_case_count"]
            row[f"{name} Σ {speedup['reference_component']}"] = _optional_value(speedup["reference_seconds_sum"])
            row[f"{name} Σ surrogate forward [s]"] = _optional_value(speedup["surrogate_seconds_sum"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def plot_runtime_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Show the steady speed-ups with their paired and unavailable case counts, batch size and hardware, the role only where a label lacks it."""
    runtime = build_runtime_table(datasets)
    named = runtime["dataset_role"].to_numpy() == runtime.index.to_numpy()  # the row label is already the role's display name
    if named.all():
        runtime = runtime.drop(columns="dataset_role")
    else:
        runtime["dataset_role"] = runtime["dataset_role"].astype(object).where(~named, None)  # a blank cell, not a repeat
    return tables.styled_dataframe(display_table(runtime), title="Runtime and speed-up (ratio of sums over paired cases)")


def transient_inference_seconds(report: transient_timing.TransientTimingReport) -> float | None:
    """Return the mean over cases of each case's median drying-rollout model-call seconds, or ``None`` when no case has one."""
    medians = [float(np.median(values)) for case in report.cases if (values := case.repetitions.get("drying_no_rollout_model_seconds"))]
    return float(np.mean(medians)) if medians else None


def build_work_table(stages: Mapping[str, tuple[Mapping[str, Any], float | None]]) -> pd.DataFrame:
    """
    Build one row per labelled completed stage from its summary and inference seconds per case.

    The rows are ``run_discovery.stage_work_row``; a value the run did not record reads
    ``unavailable``, never zero.
    """
    rows = []
    for label, (summary, seconds) in stages.items():
        row = run_discovery.stage_work_row(summary, inference_seconds_per_case=seconds)
        rows.append({"label": label, **{name: run_discovery.UNAVAILABLE if value is None else value for name, value in row.items()}})
    return pd.DataFrame.from_records(rows).set_index("label")


def build_frame_work_table(frames: Mapping[str, pd.DataFrame], inference: Callable[[str, pd.DataFrame], float | None]) -> pd.DataFrame:
    """
    Build one work row per completed run of the labelled frames, named by its first frame.

    The frames of one run (its roles) carry the summary the loader attached; the row takes the
    inference seconds per case of that first frame's artifact. The model family of a run whose complete provenance records a
    physics-informed model is its ``pi-`` token.
    """
    table = build_work_table({label: (summary, inference(label, frames[label])) for label, summary in run_discovery.distinct_runs(frames).items()})
    for label, kind in table["model_kind"].items():
        if kind != run_discovery.UNAVAILABLE and _physics_informed(frames[str(label)]):
            table.loc[label, "model_kind"] = _family_token(str(kind), physics_enabled=True)
    return table


def _physics_informed(frame: pd.DataFrame) -> bool:
    """Return whether the complete provenance of ``frame`` records a physics-informed model; a frame without provenance is not."""
    provenance = frame.attrs.get("artifact_provenance") if frame.attrs.get("provenance_complete") is True else None
    model = provenance.get("model") if isinstance(provenance, Mapping) else None
    return isinstance(model, Mapping) and model.get("physics_enabled") is True


def _steady_inference_seconds(_label: str, frame: pd.DataFrame) -> float:
    """Return the mean batch-amortized forward seconds per case of a steady artifact."""
    cases = _provenance(frame)["timing"]["report"]["cases"]
    return float(np.mean([case["batch_amortized_neural_operator_forward_s"] for case in cases]))


def plot_work_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Show the work quantities of every steady run, one column per run, with its mean batch-amortized forward seconds per case."""
    return show_work_table(build_frame_work_table(datasets, _steady_inference_seconds))


def show_work_table(table: pd.DataFrame) -> widgets.VBox:
    """Show a work table (``build_work_table``) of both tasks transposed: a column per run, a row per quantity, so no row runs wide."""
    counts = table.copy()
    for column in counts.columns:
        if "seconds" not in str(column):  # every other number is a count, which the generic cell format would write as 2.497e+05
            counts[column] = counts[column].map(_whole_number)
    return tables.styled_dataframe(display_table(counts).transpose(), title="Work per stage")


def _whole_number(value: object) -> object:
    """Return an integer-valued number as grouped whole-number text (``249,700``), any other value as it is."""
    if isinstance(value, Integral) and not isinstance(value, bool):  # exact, without a float round trip
        return f"{int(value):,}"
    if isinstance(value, Real) and float(value).is_integer():  # false for NaN and infinity
        return f"{int(float(value)):,}"
    return value


def _diagnostics(frame: pd.DataFrame, group_ids: tuple[str, ...]) -> list[float]:
    """Return the task macro and the group values of one frame's aggregate, all NaN (never zero) when it is unavailable."""
    aggregate = frame.attrs["normalized_group_macro_rmse"]
    if not aggregate["available"]:
        return [np.nan] * (1 + len(group_ids))
    statistics = aggregate["group_statistics"]
    if tuple(str(group_id) for group_id in statistics) != group_ids:
        msg = "Relative comparison scoreboard requires identical ordered task groups."
        raise dataframe.ComparisonCompatibilityError(msg)
    return [float(aggregate["value"]), *(float(statistics[group_id]["normalized_rmse"]) for group_id in group_ids)]


def _relative_to_best(values: np.ndarray, best: np.ndarray) -> np.ndarray:
    """Return ``values`` as ratios to the best of each diagnostic (last axis); a best of zero ranks only exact zeros, at 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = values / best
    return np.where(best == 0.0, np.where(values == 0.0, 1.0, np.nan), ratios)


def plot_relative_comparison_scoreboard(*, datasets: dataset_cuts.DisplayFrames) -> Figure:
    """
    Plot each current normalized aggregate relative to its displayed best: a panel per dataset, the diagnostics as groups of one bar per model.

    A pooled dataset (one frame per material) draws one bar per model and diagnostic at the equal-weight mean of its materials' values
    (``series.material_mean``), relative to the best of those means, with each material's own value as a small marker on the same scale.
    The materials of a pooled series must share one exact grid (``dataframe.validate_pooled_grids``). Where a diagnostic's displayed
    best is zero (only exact zeros rank, at 1) or unavailable, a finite value has no ratio: the note says so and lists it as a raw value.
    """
    dataframe.validate_comparison(datasets)
    groups = series.panel_series(datasets.labels)
    pools = [keys for models in groups.values() for keys in models.values() if len(keys) > 1]
    dataframe.validate_pooled_grids(datasets, pools)
    styles, material_styles = series.series_styles(datasets, pools), series.series_styles(datasets)
    if not groups or any(len(models) < _MINIMUM_SCOREBOARD_MODELS for models in groups.values()):
        msg = "Relative comparison scoreboard requires at least two models per dataset."
        raise ValueError(msg)
    figure_title, count_headings = layout.dataset_title_context(
        "Comparison-relative normalized error", datasets, {key: len(frame) for key, frame in datasets.items()}
    )
    group_ids = tuple(str(group_id) for group_id, _fields in datasets[next(iter(datasets))].attrs["output_groups"])
    diagnostic_labels = ("Task macro", *(f"Group: {group_id}" for group_id in group_ids))
    figure, axes = figure_layout.panel_figure(1, len(groups), sharey="row")
    positions = np.arange(len(diagnostic_labels), dtype=float)
    width = 0.8 / max(len(models) for models in groups.values())
    unavailable: list[str] = []
    undefined: list[str] = []
    tied_panels = []
    for axis, (dataset, models) in zip(axes[0], groups.items(), strict=True):
        # An unavailable aggregate stays NaN: it gets no bar and the note names its support counts instead.
        raw = np.full((len(models), len(diagnostic_labels)), np.nan)
        per_material = []
        suffix = {key: f" ({series.material_name(datasets[key])})" if len(keys) > 1 else "" for keys in models.values() for key in keys}
        for row, (model, keys) in enumerate(models.items()):
            raw[row], values = series.material_mean(keys, lambda key: _diagnostics(datasets[key], group_ids))
            per_material.append(values)
            for key in keys:
                aggregate = datasets[key].attrs["normalized_group_macro_rmse"]
                if not aggregate["available"]:
                    unavailable.append(f"{model} on {dataset}{suffix[key]}: {validity_contract.unavailable_text(aggregate)}")
        available = np.isfinite(raw)
        if np.any(raw[available] < 0.0):
            msg = "Relative comparison scoreboard requires non-negative normalized errors."
            raise ValueError(msg)
        best = np.where(available.any(axis=0), np.where(available, raw, np.inf).min(axis=0), np.nan)
        relative = _relative_to_best(raw, best)
        tied_panels.append(bool(available.all() and np.allclose(relative, 1.0, rtol=1e-6, atol=1e-12, equal_nan=False)))
        raw_values: dict[int, list[str]] = {}  # per diagnostic, the finite values whose ratio is undefined
        for row, (model, keys) in enumerate(models.items()):
            shown = np.isfinite(relative[row])
            offset = (row - (len(models) - 1) / 2.0) * width
            style = styles[keys[0]]
            axis.bar(positions[shown] + offset, relative[row, shown], width, color=style.color, edgecolor="white", hatch=style.hatch)
            for key, values, ratios in zip(keys, per_material[row], _relative_to_best(per_material[row], best), strict=True):
                kept = np.isfinite(ratios)
                for column in np.flatnonzero(np.isfinite(values) & ~kept):
                    raw_values.setdefault(int(column), []).append(f"{model}{suffix[key]} {values[column]:.3g}")
                if len(keys) > 1:
                    marker_style = material_styles[key]
                    axis.scatter(
                        positions[kept] + offset,
                        ratios[kept],
                        marker=marker_style.marker or "o",
                        s=24,
                        color=marker_style.color,
                        edgecolors="black",
                        zorder=3,
                    )
        undefined += [
            f"Ratio undefined for {diagnostic_labels[column]} on {dataset} (displayed best is {'zero' if best[column] == 0.0 else 'unavailable'}); "
            f"raw values: {', '.join(entries)}"
            for column, entries in sorted(raw_values.items())
        ]
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        axis.set_xticks(positions, diagnostic_labels, rotation=25, ha="right")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        figure_layout.set_panel_title(axis, count_headings[dataset] or dataset)
    figure_layout.label_rows_once(axes, f"Ratio to displayed best per metric [{field_labels.display_unit('1')}]\n(lower is better)")
    normalization = (
        "Displayed models are tied across all shown normalized diagnostics"
        if all(tied_panels)
        else "Each diagnostic is normalized independently to its displayed best"
    )
    extra = layout.material_legend(datasets, groups, material_styles)
    layout.add_series_legend(figure, axes, datasets, series.series_heads(groups), styles, bars=True, extra=extra)
    not_plotted = ["Not plotted: " + "; ".join(unavailable)] if unavailable else []
    pooled_note = layout.POOLED_NOTE if pools else None
    note = layout.series_note(
        styles, normalization, *not_plotted, *undefined, pooled_note, "This is a descriptive comparison, not an overall ranking"
    )
    figure_layout.set_figure_title(figure, axes, figure_title, note=note)
    return figure


def _config_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    """Require one mapping from already admitted completed-run evidence."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be available in sealed completed-run configuration evidence."
        raise dataframe.ComparisonCompatibilityError(msg)
    return value


def _compact_value(value: object) -> object:
    """Format curated sequence values compactly without dumping raw config nodes."""
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple)) and len(item) == _SPATIAL_PAIR_LENGTH for item in value):
            return " → ".join(f"{float(item[0]):g}x{float(item[1]):g}" for item in value)
        return " x ".join(str(item) for item in value)
    return value


def _family_token(kind: str, *, physics_enabled: bool) -> str:
    """Return the family token with its PI status (``uno``, ``pi-uno``), which ``display_table`` shows as ``U-NO`` or ``PI-U-NO``."""
    return f"pi-{kind}" if physics_enabled else kind


def _family_label(kind: str, *, physics_enabled: bool) -> str:
    """Return the model family's display name with its PI status (``U-NO``, ``PI-U-NO``); an unregistered family reads upper case."""
    family = _family_token(kind, physics_enabled=physics_enabled)
    return _FAMILY_LABELS.get(family, family.upper())


def _configuration_float(value: object, *, label: str) -> float:
    """Require one finite real completed-run configuration value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a finite real value."
        raise dataframe.ComparisonCompatibilityError(msg)
    numeric = float(value)
    if not np.isfinite(numeric):
        msg = f"{label} must be a finite real value."
        raise dataframe.ComparisonCompatibilityError(msg)
    return numeric


def _configuration_integer(value: object, *, label: str) -> int:
    """Require one non-negative integer completed-run configuration value."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        msg = f"{label} must be a non-negative integer."
        raise dataframe.ComparisonCompatibilityError(msg)
    return int(value)


def _weight_display(value: object) -> object:
    """Disclose zero-weight terms as disabled and retain active exact targets."""
    numeric = _configuration_float(value, label="physics target weight")
    return "disabled (target 0)" if numeric == 0.0 else numeric


def _warmup_display(weight: Mapping[str, Any]) -> str:
    """Describe the implemented zero-to-target schedule and post-ramp behavior."""
    target = _configuration_float(weight.get("target"), label="physics target weight")
    warmup = _config_mapping(weight.get("warmup"), label="physics weight warmup")
    kind_value = warmup.get("kind")
    if not isinstance(kind_value, str) or not kind_value:
        msg = "Physics warmup kind must be non-empty text."
        raise dataframe.ComparisonCompatibilityError(msg)
    kind = kind_value
    epochs = _configuration_integer(warmup.get("epochs"), label="physics warmup epochs")
    if target == 0.0:
        return "disabled (target weight 0)"
    if epochs == 0:
        return f"{kind}: target active from epoch 0"
    return f"{kind}: 0 → target over {epochs} epochs; target thereafter"


def architecture_items(architecture: Mapping[str, Any]) -> list[tuple[str, object]]:
    """Return family-applicable architecture values in reconstructable order, as the configuration tables of both tasks show them."""
    labels = {
        "n_modes": "Spectral modes",
        "hidden_channels": "Hidden channels",
        "n_layers": "Layers",
        "lifting_channel_ratio": "Lifting ratio",
        "projection_channel_ratio": "Projection ratio",
        "implementation": "Implementation",
        "factorization": "Factorization",
        "fno_skip": "FNO skip",
        "channel_mlp_skip": "Channel-MLP skip",
        "mode_ratio": "U-NO mode ratio",
        "uno_scalings": "U-NO resolution schedule",
        "domain_padding": "Domain padding",
        "resolution_scaling_factor": "Resolution scaling",
    }
    # A grid stored as x and y modes is the same quantity as a stored pair, so every family shows one ``Spectral modes`` row.
    values = {key: value for key, value in architecture.items() if key not in _MODE_AXES}
    if architecture.keys() >= _MODE_AXES:
        values["n_modes"] = [architecture[axis] for axis in sorted(_MODE_AXES)]
    ordered_keys = tuple(labels)
    result = [(f"Architecture · {labels[key]}", _compact_value(values[key])) for key in ordered_keys if key in values]
    excluded = {*ordered_keys, "in_channels", "out_channels"}
    for key, value in values.items():
        if key not in excluded:
            result.append((f"Architecture · {display_labels.token_display_label(str(key))}", _compact_value(value)))
    return result


def _copy_configuration_fields(
    row: dict[str, Any],
    source: Mapping[str, Any],
    fields: tuple[tuple[str, str], ...],
) -> None:
    """Copy curated present fields into grouped presentation columns; an empty configured list or mapping names nothing and has no row."""
    for key, column in fields:
        if key in source and source[key] not in ([], {}, None):
            row[column] = _compact_value(source[key])


def channel_items(architecture: Mapping[str, Any]) -> dict[str, object]:
    """Return the input and output channel counts an architecture records, as the configuration tables of both tasks show them."""
    row: dict[str, Any] = {}
    _copy_configuration_fields(row, architecture, _CHANNEL_FIELDS)
    return row


def _add_data_configuration(
    row: dict[str, Any],
    config: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> None:
    """Add applicable dataset and channel configuration; the training Dataset reads short, its exact identity among the evidence."""
    data = config.get("data")
    if isinstance(data, Mapping):
        # The configured training data; the Datasets evaluated as material OOD are the admitted roles, not these rows.
        if train_dataset := _train_dataset(config):
            row["Data · Configured training Dataset"] = display_labels.compact_identity_label(train_dataset)
        _copy_configuration_fields(row, data, (("material_ood_datasets", "Data · Configured material-OOD Datasets"),))
    row.update(channel_items(architecture))


def _train_dataset(config: Mapping[str, Any]) -> str | None:
    """Return the exact configured training Dataset identity, or ``None`` when the configuration names none."""
    data = config.get("data")
    value = data.get("train_dataset") if isinstance(data, Mapping) else None
    return value if isinstance(value, str) and value else None


def _add_optimization_configuration(row: dict[str, Any], config: Mapping[str, Any]) -> None:
    """Add applicable loss, optimizer, scheduler, and duration fields."""
    loss = config.get("loss")
    if isinstance(loss, Mapping):
        data_loss = loss.get("data")
        if isinstance(data_loss, Mapping):
            _copy_configuration_fields(
                row,
                data_loss,
                (
                    ("kind", "Optimization · Data loss"),
                    ("weight", "Optimization · Data-loss weight"),
                ),
            )
    optimizer = config.get("optimizer")
    if isinstance(optimizer, Mapping):
        _copy_configuration_fields(
            row,
            optimizer,
            (
                ("kind", "Optimization · Optimizer"),
                ("lr", "Optimization · Learning rate"),
                ("weight_decay", "Optimization · Weight decay"),
                ("second_moment_floor", "Optimization · Second-moment floor"),
            ),
        )
        betas = optimizer.get("betas")
        if isinstance(betas, (list, tuple)) and betas:  # an ordered pair of decay rates; «x» is reserved for spatial sizes
            row["Optimization · Betas"] = f"({', '.join(f'{float(beta):g}' for beta in betas)})"
    scheduler = config.get("scheduler")
    if isinstance(scheduler, Mapping):
        _copy_configuration_fields(
            row,
            scheduler,
            (
                ("kind", "Optimization · Scheduler"),
                ("factor", "Optimization · Scheduler factor"),
                ("patience", "Optimization · Scheduler patience"),
            ),
        )
    training = config.get("training")
    if isinstance(training, Mapping):
        _copy_configuration_fields(
            row,
            training,
            (
                ("budget_epochs", "Optimization · Budget epochs"),
                ("units_per_update", "Optimization · Units per update"),
            ),
        )


def _add_physics_configuration(
    row: dict[str, Any],
    config: Mapping[str, Any],
    *,
    physics_enabled: bool,
) -> None:
    """Add each applicable PI term, schedule, and numerical setting."""
    loss = config.get("loss")
    physics_value = loss.get("physics") if isinstance(loss, Mapping) else None
    physics = _config_mapping(physics_value, label="completed-run loss.physics")
    row["Physics-informed loss · Enabled"] = physics_enabled
    if not physics_enabled:
        return
    derivatives = _config_mapping(physics.get("derivatives"), label="loss.physics.derivatives")
    for term, title in (("momentum", "Momentum residual"), ("continuity", "Flux-balance continuity"), ("boundary", "Pressure-boundary")):
        weight = _config_mapping(physics.get(f"{term}_weight"), label=f"loss.physics.{term}_weight")
        row[f"Physics-informed loss · {title} weight"] = _weight_display(weight.get("target"))
        row[f"Physics-informed loss · {title} schedule"] = _warmup_display(weight)
    row.update(
        {
            "Physics-informed loss · Derivative operator": derivatives.get("kind"),
            "Physics-informed loss · Interior crop": physics.get("interior_crop"),
        }
    )
    _copy_configuration_fields(
        row,
        physics,
        (
            ("residual_normalization", "Physics-informed loss · Residual normalization"),
            ("residual_scale", "Physics-informed loss · Residual scaling"),
            ("velocity_representation", "Physics-informed loss · Configured velocity representation"),
        ),
    )


def _add_runtime_configuration(
    row: dict[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Add declared runtime values and sealed evidence digests."""
    run_config = config.get("run")
    if isinstance(run_config, Mapping):
        _copy_configuration_fields(
            row,
            run_config,
            (
                ("name", "Runtime/evidence · Run name"),
                ("seed", "Runtime/evidence · Seed"),
                ("deterministic", "Runtime/evidence · Deterministic"),
                ("device", "Runtime/evidence · Requested device"),
            ),
        )
    if train_dataset := _train_dataset(config):
        row["Runtime/evidence · Configured training Dataset ID"] = train_dataset
    run = _config_mapping(provenance.get("run"), label="artifact run provenance")
    row["Runtime/evidence · Config digest"] = run.get("effective_config_digest")
    row["Runtime/evidence · Checkpoint digest"] = run.get("best_checkpoint_sha256")


def _configuration_row(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Build one compact grouped model/run configuration row."""
    provenance = _provenance(frame)
    model = _config_mapping(provenance.get("model"), label="artifact model provenance")
    architecture = _config_mapping(model.get("architecture"), label="artifact architecture provenance")
    counts = _config_mapping(model.get("parameter_counts"), label="artifact parameter counts")
    physics_enabled = model.get("physics_enabled")
    kind = model.get("kind")
    if type(physics_enabled) is not bool or not isinstance(kind, str):
        msg = "Model configuration requires typed model kind and physics enablement."
        raise dataframe.ComparisonCompatibilityError(msg)

    config_value = frame.attrs.get(dataframe.COMPLETED_RUN_CONFIG_ATTR)
    if config_value is None:
        config_value = {
            "model": {"kind": kind, "params": architecture},
            "loss": {"physics": {"enabled": physics_enabled}},
        }
    config = _config_mapping(config_value, label="completed-run configuration")
    model_config = _config_mapping(config.get("model"), label="completed-run model")
    config_architecture = _config_mapping(model_config.get("params"), label="completed-run model.params")
    if dict(config_architecture) != dict(architecture):
        msg = "Sealed completed-run architecture contradicts artifact provenance."
        raise dataframe.ComparisonCompatibilityError(msg)

    row: dict[str, Any] = {
        "label": label,
        "Architecture · Model family": _family_label(kind, physics_enabled=physics_enabled),
        **dict(architecture_items(architecture)),
        "Runtime/evidence · Total parameters": counts.get("total"),
        "Runtime/evidence · Trainable parameters": counts.get("trainable"),
    }
    _add_data_configuration(row, config, architecture)
    _add_optimization_configuration(row, config)
    _add_physics_configuration(row, config, physics_enabled=physics_enabled)
    _add_runtime_configuration(row, config, provenance)
    return row


def build_architecture_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one grouped, family-specific configuration row per completed run."""
    dataframe.validate_comparison(datasets)
    rows: list[dict[str, Any]] = []
    seen_runs: set[tuple[object, object]] = set()
    for label, frame in datasets.items():
        provenance = _provenance(frame)
        run = _config_mapping(provenance.get("run"), label="artifact run provenance")
        identity = (run.get("name"), run.get("effective_config_digest"))
        if identity in seen_runs:
            continue
        seen_runs.add(identity)
        rows.append(_configuration_row(label, frame))
    if not rows:
        msg = "Architecture overview requires at least one completed-run configuration."
        raise ValueError(msg)
    table = pd.DataFrame(rows)
    ordered_columns = ["label", *(column for row in rows for column in row if column != "label")]
    ordered_columns = list(dict.fromkeys(ordered_columns))
    return table.reindex(columns=ordered_columns).set_index("label")


def plot_architecture_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Show compact grouped architecture, optimization, and PI evidence."""
    return tables.styled_dataframe(display_configuration_table(build_architecture_table(datasets)), title="Model-family configuration")


def display_configuration_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a configuration table (a row per run) as shown, a row per setting: one no run has is left out, a value some lack is a dash."""
    return display_table(table.dropna(axis="columns", how="all"), absent="\N{EN DASH}").transpose()


def plot_accuracy_physics_pareto(*, datasets: dataset_cuts.DisplayFrames) -> Figure:
    """
    Plot authoritative accuracy against four separate physics diagnostics.

    Parameters
    ----------
    datasets : DisplayFrames
        Compatible steady-flow frames with exact aggregate and residual evidence, labelled with their dataset and model.

    Returns
    -------
    matplotlib.figure.Figure
        A row per dataset of three panels pairing ``normalized_group_macro_rmse`` with the median
        dimensionless momentum, flux-balance, or pressure-boundary diagnostic.

    Raises
    ------
    ComparisonCompatibilityError, KeyError, TypeError, ValueError
        If physics provenance, model metadata, aggregates, or finite non-negative
        metrics are invalid.

    Notes
    -----
    Axes remain separate because diagnostic units/meanings differ. Filled markers
    disclose physics-enabled models. The marker is the material where a run spans several. No
    composite Pareto rank or parameter-efficiency score is calculated.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    figure_title, count_headings = layout.dataset_title_context(
        "Accuracy versus physical consistency", datasets, {key: len(frame) for key, frame in datasets.items()}
    )
    figure, axes = figure_layout.panel_figure(
        len(groups), len(dataframe.STEADY_PHYSICS_METRICS), panel_in=(4.2, 3.6)
    )  # taller for the long diagnostic labels
    unavailable: list[str] = []
    handles: dict[str, Line2D] = {}  # one entry per model in its colour, filled when physics-informed, as the points
    model_markers: dict[str, set[str]] = {}  # the markers of each model's points
    shapes: dict[str, Line2D] = {}  # one hollow black marker per material whose points take its shape
    for row, (dataset, models) in enumerate(groups.items()):
        for model, key in models.items():
            _family, _count, physics_enabled = series.model_metadata(datasets[key])
            style = styles[key]
            fill = style.color if physics_enabled else "none"
            handles.setdefault(model, Line2D([], [], linestyle="none", markersize=8, markerfacecolor=fill, markeredgecolor=style.color))
            model_markers.setdefault(model, set()).add(style.marker or "o")
            if style.marker:
                (material,) = _provenance(datasets[key])["selection"]["material_families"]
                shapes.setdefault(
                    display_labels.material_display_label(str(material)),
                    Line2D([], [], marker=style.marker, linestyle="none", markersize=8, markerfacecolor="none", markeredgecolor="black"),
                )
            aggregate = datasets[key].attrs["normalized_group_macro_rmse"]
            if not aggregate["available"]:
                unavailable.append(f"{model} on {dataset}: {validity_contract.unavailable_text(aggregate)}")
                continue
            primary = float(aggregate["value"])
            marker = style.marker or "o"
            for axis, metric in zip(axes[row], dataframe.STEADY_PHYSICS_METRICS, strict=True):
                physics_value = _finite_quantile(datasets[key], metric, 0.5)
                if primary < 0.0 or physics_value < 0.0:
                    msg = f"Pareto metrics must be non-negative. {key!r} supplied {primary}, {physics_value}."
                    raise ValueError(msg)
                if np.isnan(physics_value):
                    unavailable.append(f"{model} on {dataset}: no finite {metric}")
                axis.scatter(
                    primary,
                    physics_value,
                    marker=marker,
                    s=80,
                    facecolors=fill,
                    edgecolors=style.color,
                    linewidths=1.5,
                )
        for axis, metric in zip(axes[row], dataframe.STEADY_PHYSICS_METRICS, strict=True):
            metric_label, unit = _PHYSICS_LABELS[metric]
            figure_layout.set_log_scale(axis, "y")
            axis.margins(_PARETO_PAD)  # in the transformed range, so the log axis pads alike; two points never sit on the corners
            axis.set_xlabel(f"Normalized task error [{field_labels.display_unit('1')}]")
            axis.set_ylabel(f"Median {metric_label}\n[{field_labels.display_unit(unit)}]")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=_PARETO_ERROR_TICKS))  # long decimals crowd a narrow error range
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
            if row == 0:
                figure_layout.set_panel_title(axis, _PHYSICS_TITLES[metric])
    for model, handle in handles.items():  # the marker a model's points share, a neutral circle where they carry several
        handle.set_marker(next(iter(model_markers[model])) if len(model_markers[model]) == 1 else "o")
    figure_layout.set_row_headers(figure, axes, [count_headings[dataset] or dataset for dataset in groups])
    figure_layout.add_legend(  # the panel row names the dataset and so its role; a model named like a material keeps its own entry
        figure,
        axes,
        [*handles.values(), *shapes.values()],
        [*handles, *shapes],
        groups={"Model": list(handles.values()), "Material": list(shapes.values())} if shapes else None,
    )
    markers = "Filled marker: physics-informed loss, hollow: data loss only" + ("; marker shape: material" if shapes else "")
    not_plotted = "Not plotted: " + "; ".join(unavailable) if unavailable else None
    figure_layout.set_figure_title(figure, axes, figure_title, note=layout.series_note(styles, markers, not_plotted))
    return figure
