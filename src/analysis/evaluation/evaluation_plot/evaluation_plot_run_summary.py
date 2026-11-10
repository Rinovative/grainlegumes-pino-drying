"""Summarize authoritative model accuracy, architecture, and physical tradeoffs.

Responsibilities:
  - Build provenance-backed run and architecture summary tables
  - Read normalized_group_macro_rmse only from validated aggregate evidence
  - Plot comparison-set-relative task and output-group diagnostics
  - Compare exact trainable capacity with authoritative aggregate accuracy
  - Plot accuracy against each separately named physical diagnostic

Design principles:
  - Per-case quantiles remain secondary and never reconstruct the objective
  - Comparison-relative normalization is explicitly descriptive, not a ranking
  - Exact parameter counts and architecture values come only from provenance
  - Incompatible physical diagnostics retain distinct axes, names, and units

This module does NOT:
  - Infer model facts from run names, paths, or hidden configuration defaults
  - Invent parameter-efficiency or cross-diagnostic composite scores
  - Admit incompatible frames or serialize tracking media
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.artifacts import contracts
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_PHYSICS_LABELS = {
    "momentum_residual_mse": (r"momentum mean($(R_x^2+R_y^2)/G^2$)", "1"),
    "flux_balance_mse": (r"flux balance mean($((q-\bar q)/Q)^2$)", "1"),
    "pressure_boundary_mse": (r"pressure boundary diagnostic$/\Delta p^2$", "1"),
}
_ROLE_MARKERS = {"ID": "o", "OOD": "s", "unspecified": "D"}
_MINIMUM_SCOREBOARD_DATASETS = 2
_SPATIAL_PAIR_LENGTH = 2


def _provenance(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Return validated comparison provenance."""
    return dataframe.require_complete_provenance(frame)


def _finite_quantile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    """Return one per-case metric quantile over the finite cases, or NaN (never plotted) when no case is finite."""
    values = validity_contract.finite_case_values(pd.to_numeric(frame[column], errors="raise"))
    return float(np.quantile(values, quantile)) if values.size else float("nan")


def _model_metadata(frame: pd.DataFrame) -> tuple[str, int, bool]:
    """
    Read model family, exact trainable count, and PI flag.

    Values come only from complete provenance. Missing/non-typed parameter counts
    or physics enablement fail rather than being inferred from labels,
    configuration paths, or architecture names.
    """
    provenance = _provenance(frame)
    model = provenance.get("model")
    if not isinstance(model, Mapping):
        msg = "Artifact provenance model must be a mapping."
        raise TypeError(msg)
    counts = model.get("parameter_counts")
    if not isinstance(counts, Mapping):
        msg = "Artifact provenance model.parameter_counts must be a mapping."
        raise TypeError(msg)
    family = model.get("kind")
    trainable = counts.get("trainable")
    if not isinstance(family, str) or not family or isinstance(trainable, bool) or not isinstance(trainable, int):
        msg = "Run summary requires model kind and exact trainable parameter count."
        raise TypeError(msg)
    physics_enabled = model.get("physics_enabled")
    if type(physics_enabled) is not bool:
        msg = "Run summary requires a boolean model.physics_enabled provenance value."
        raise TypeError(msg)
    return family, trainable, physics_enabled


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
        family, trainable, physics_enabled = _model_metadata(frame)
        row: dict[str, Any] = {
            "label": label,
            "dataset_role": dataframe.dataset_role(frame),
            "task_id": frame.attrs["task_id"],
            "run_name": run["name"],
            "architecture_family": family,
            "trainable_parameters": trainable,
            "physics_enabled": physics_enabled,
            "normalized_group_macro_rmse": unavailable or float(aggregate["value"]),
            "relative_l2_median": _finite_quantile(frame, "rel_l2", 0.5),
            "relative_l2_q90": _finite_quantile(frame, "rel_l2", 0.9),
            "relative_h1_median": _finite_quantile(frame, "rel_h1", 0.5),
            "relative_h1_q90": _finite_quantile(frame, "rel_h1", 0.9),
            "dataset_name": dataset["name"],
            "sample_count": len(frame),
            "config_digest": str(run["effective_config_digest"]),
            "checkpoint_digest": str(run["best_checkpoint_sha256"]),
        }
        for field in fields:
            row[f"normalized_rmse_{field}"] = unavailable or float(aggregate["field_statistics"][field]["normalized_rmse"])
            physical_column = contracts.physical_statistic_columns(field)[2]
            row[f"physical_rmse_{field}_median [{units[field]}]"] = _finite_quantile(frame, physical_column, 0.5)
            row[f"physical_rmse_{field}_q90 [{units[field]}]"] = _finite_quantile(frame, physical_column, 0.9)
        for group_id, _group_fields in frame.attrs["output_groups"]:
            for space in ("normalized", "physical"):
                row[f"{space}_group_rmse_{group_id}"] = unavailable or float(aggregate["group_statistics"][group_id][f"{space}_rmse"])
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
                unit = _PHYSICS_LABELS[metric][1]
                row[f"{metric}_median [{unit}]"] = _finite_quantile(frame, metric, 0.5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def _blue_style(table: pd.DataFrame) -> pd.DataFrame:
    """Return quantile-bounded blue numeric cell fills."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    cmap = plt.get_cmap("Blues")
    for column in table.columns:
        numeric = np.asarray(pd.to_numeric(table[column], errors="coerce"), dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            continue
        low, high = np.quantile(finite, (0.05, 0.95))
        for row, value in enumerate(numeric):
            if not np.isfinite(value):
                continue
            fraction = 0.0 if np.isclose(low, high) else float(np.clip((value - low) / (high - low), 0.0, 1.0))
            red, green, blue, _alpha = cmap(0.05 + 0.90 * fraction)
            column_index = cast("int", styles.columns.get_loc(column))
            styles.iloc[row, column_index] = f"background-color: rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, 0.55)"
    return styles


def _cell_text(value: object) -> str:
    """Format a number to four significant digits and keep text, such as an unavailable metric, verbatim."""
    return f"{value:.4g}" if isinstance(value, Real) and not isinstance(value, bool) else str(value)


def _styled_table(table: pd.DataFrame, *, title: str) -> widgets.VBox:
    """Render one automatically displayed blue table."""
    styles = _blue_style(table)
    styler = table.style
    styler.format(_cell_text)
    styler.apply(lambda _table: styles, axis=None)
    return widgets.VBox((widgets.HTML(f"<h2>{title}</h2>"), widgets.HTML(styler.to_html())))


def plot_run_summary_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Show current authoritative metrics using the defined colored table."""
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
    return _styled_table(visible, title=title)


def plot_relative_comparison_scoreboard(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot each current normalized aggregate relative to its displayed best."""
    dataframe.validate_comparison(datasets)
    if len(datasets) < _MINIMUM_SCOREBOARD_DATASETS:
        msg = "Relative comparison scoreboard requires at least two datasets."
        raise ValueError(msg)

    labels = tuple(datasets)
    figure_title, count_headings = layout.aggregate_title_context(
        "Comparison-relative normalized error",
        layout.effective_case_counts(datasets),
    )
    group_ids = tuple(str(group_id) for group_id, _fields in datasets[labels[0]].attrs["output_groups"])
    diagnostic_labels = ("Task macro", *(f"Group: {group_id}" for group_id in group_ids))
    # An unavailable aggregate stays NaN: it gets no bar and its tick names the support counts instead.
    raw = np.full((len(labels), len(diagnostic_labels)), np.nan)
    tick_labels = [count_headings[label] or label for label in labels]
    for row, label in enumerate(labels):
        aggregate = datasets[label].attrs["normalized_group_macro_rmse"]
        if not aggregate["available"]:
            tick_labels[row] += f"\n{validity_contract.unavailable_text(aggregate)}"
            continue
        groups = aggregate["group_statistics"]
        if tuple(str(group_id) for group_id in groups) != group_ids:
            msg = "Relative comparison scoreboard requires identical ordered task groups."
            raise dataframe.ComparisonCompatibilityError(msg)
        raw[row] = [float(aggregate["value"]), *(float(groups[group_id]["normalized_rmse"]) for group_id in group_ids)]
    available = np.isfinite(raw)
    if np.any(raw[available] < 0.0):
        msg = "Relative comparison scoreboard requires non-negative normalized errors."
        raise ValueError(msg)

    relative = np.full_like(raw, np.nan)
    for column in range(raw.shape[1]):
        if not available[:, column].any():
            continue
        best = float(np.nanmin(raw[:, column]))
        if best == 0.0:
            relative[raw[:, column] == 0.0, column] = 1.0
        else:
            relative[:, column] = raw[:, column] / best
    tied_diagnostics = bool(available.all() and np.allclose(relative, 1.0, rtol=1e-6, atol=1e-12, equal_nan=False))

    figure = plt.figure(figsize=(max(12.5, 3.5 * len(labels) + 2.5), 5.3))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 0.28), wspace=0.18)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    positions = np.arange(len(labels), dtype=float)
    width = min(0.8 / len(diagnostic_labels), 0.24)
    offsets = (np.arange(len(diagnostic_labels)) - (len(diagnostic_labels) - 1) / 2.0) * width
    for offset, diagnostic, values in zip(offsets, diagnostic_labels, relative.T, strict=True):
        shown = np.isfinite(values)
        axis.bar(positions[shown] + offset, values[shown], width, label=diagnostic)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    axis.set_xticks(positions)
    axis.set_xticklabels(tick_labels, rotation=25, ha="right")
    axis.set_ylabel("Ratio to displayed best per metric (lower is better)")
    subtitle = (
        "Displayed models are tied across all shown normalized diagnostics\nThis is a descriptive comparison, not an overall ranking"
        if tied_diagnostics
        else "Each diagnostic is normalized independently to its displayed best\nThis is a descriptive comparison, not an overall ranking"
    )
    axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    handles, legend_labels = axis.get_legend_handles_labels()
    legend_axis.legend(handles, legend_labels, title="Normalized diagnostics", loc="upper left", frameon=True)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.82)
    layout.title_over_axes(figure, figure_title, (axis,), y=0.975)
    layout.text_over_axes(
        figure,
        subtitle,
        (axis,),
        y=0.91,
        ha="center",
        va="top",
        fontsize=9.5,
        linespacing=1.25,
    )
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


def _family_label(kind: str, *, physics_enabled: bool) -> str:
    """Return the maintained FNO/UNO family label with explicit PI status."""
    base = kind.upper()
    return f"PI-{base}" if physics_enabled else base


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


def _architecture_items(architecture: Mapping[str, Any]) -> list[tuple[str, object]]:
    """Return family-applicable architecture values in reconstructable order."""
    labels = {
        "n_modes": "Spectral modes",
        "modes_x": "Spectral modes x",
        "modes_y": "Spectral modes y",
        "hidden_channels": "Hidden channels",
        "n_layers": "Layers",
        "lifting_channel_ratio": "Lifting ratio",
        "projection_channel_ratio": "Projection ratio",
        "implementation": "Implementation",
        "factorization": "Factorization",
        "fno_skip": "FNO skip",
        "channel_mlp_skip": "Channel-MLP skip",
        "mode_ratio": "UNO mode ratio",
        "uno_scalings": "UNO resolution schedule",
        "domain_padding": "Domain padding",
        "resolution_scaling_factor": "Resolution scaling",
    }
    ordered_keys = tuple(labels)
    result = [(f"Architecture · {labels[key]}", _compact_value(architecture[key])) for key in ordered_keys if key in architecture]
    excluded = {*ordered_keys, "in_channels", "out_channels"}
    for key, value in architecture.items():
        if key not in excluded:
            result.append((f"Architecture · {str(key).replace('_', ' ').title()}", _compact_value(value)))
    return result


def _copy_configuration_fields(
    row: dict[str, Any],
    source: Mapping[str, Any],
    fields: tuple[tuple[str, str], ...],
) -> None:
    """Copy curated present fields into grouped presentation columns."""
    for key, column in fields:
        if key in source:
            row[column] = _compact_value(source[key])


def _add_data_configuration(
    row: dict[str, Any],
    config: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> None:
    """Add applicable dataset and channel configuration."""
    data = config.get("data")
    if isinstance(data, Mapping):
        _copy_configuration_fields(
            row,
            data,
            (
                ("train_dataset", "Data · Training dataset"),
                ("material_ood_datasets", "Data · Material-OOD datasets"),
                ("batch_size", "Data · Batch size"),
            ),
        )
    _copy_configuration_fields(
        row,
        architecture,
        (
            ("in_channels", "Data · Input channels"),
            ("out_channels", "Data · Output channels"),
        ),
    )


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
                    ("space", "Optimization · Data-loss space"),
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
                ("betas", "Optimization · Betas"),
                ("second_moment_floor", "Optimization · Second-moment floor"),
            ),
        )
    scheduler = config.get("scheduler")
    if isinstance(scheduler, Mapping):
        _copy_configuration_fields(
            row,
            scheduler,
            (
                ("kind", "Optimization · Scheduler"),
                ("factor", "Optimization · Scheduler factor"),
                ("patience", "Optimization · Scheduler patience"),
                ("min_lr", "Optimization · Minimum learning rate"),
            ),
        )
    training = config.get("training")
    if isinstance(training, Mapping):
        _copy_configuration_fields(row, training, (("epochs", "Optimization · Training epochs"),))


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
        **dict(_architecture_items(architecture)),
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
    table = build_architecture_table(datasets)
    return _styled_table(table.transpose(), title="Model-family configuration")


def plot_accuracy_physics_pareto(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot authoritative accuracy against four separate physics diagnostics.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible steady-flow frames with exact aggregate and residual evidence.

    Returns
    -------
    matplotlib.figure.Figure
        Three panels pairing ``normalized_group_macro_rmse`` with the median
        dimensionless momentum, flux-balance, or pressure-boundary diagnostic.

    Raises
    ------
    ComparisonCompatibilityError, KeyError, TypeError, ValueError
        If physics provenance, model metadata, aggregates, or finite non-negative
        metrics are invalid.

    Notes
    -----
    Axes remain separate because diagnostic units/meanings differ. Filled markers
    disclose physics-enabled models. Marker shapes disclose ID/OOD role. No
    composite Pareto rank or parameter-efficiency score is calculated.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    figure_title, count_headings = layout.aggregate_title_context(
        "Accuracy versus physical consistency",
        layout.effective_case_counts(datasets),
    )
    figure = plt.figure(figsize=(19.0, 5.5))
    grid = figure.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.32), wspace=0.30)
    axes = tuple(figure.add_subplot(grid[0, column]) for column in range(len(dataframe.STEADY_PHYSICS_METRICS)))
    legend_axis = figure.add_subplot(grid[0, 3])
    legend_axis.axis("off")
    colors = plt.get_cmap("tab10")
    aggregates = {label: frame.attrs["normalized_group_macro_rmse"] for label, frame in datasets.items()}
    unavailable = [f"{label}: {validity_contract.unavailable_text(item)}" for label, item in aggregates.items() if not item["available"]]
    for axis, metric in zip(axes, dataframe.STEADY_PHYSICS_METRICS, strict=True):
        metric_label, unit = _PHYSICS_LABELS[metric]
        for index, (label, frame) in enumerate(datasets.items()):
            if not aggregates[label]["available"]:
                continue
            primary = float(aggregates[label]["value"])
            physics_value = _finite_quantile(frame, metric, 0.5)
            if primary < 0.0 or physics_value < 0.0:
                msg = f"Pareto metrics must be non-negative. {label!r} supplied {primary}, {physics_value}."
                raise ValueError(msg)
            _family, _count, physics_enabled = _model_metadata(frame)
            role = dataframe.dataset_role(frame)
            color = colors(index % colors.N)
            axis.scatter(
                primary,
                physics_value,
                marker=_ROLE_MARKERS[role],
                s=80,
                facecolors=color if physics_enabled else "none",
                edgecolors=color,
                linewidths=1.5,
                label=f"{count_headings[label] or label} ({role})",
            )
            axis.annotate(
                label,
                (primary, physics_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Normalized task error [1]")
        axis.set_ylabel(f"Median {metric_label} [{unit}]")
        axis.set_title(metric)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend_axis.legend(handles, labels, loc="upper left", title="Model / dataset", frameon=True)
    if unavailable:
        legend_axis.text(0.0, 0.0, "Not plotted:\n" + "\n".join(unavailable), transform=legend_axis.transAxes, fontsize=8, va="bottom", wrap=True)
    figure.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.98)
    layout.title_over_axes(figure, figure_title, axes, y=0.96)
    return figure
