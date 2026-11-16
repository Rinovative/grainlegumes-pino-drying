"""Summarize authoritative model accuracy, architecture, and physical tradeoffs.

Responsibilities:
  - Build provenance-backed run, runtime and architecture summary tables
  - Build the work table of completed stages from their summaries and their artifacts' inference times
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

from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_run_discovery as run_discovery
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_field_labels as field_labels
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
_PARETO_ERROR_TICKS = 4
_MINIMUM_SCOREBOARD_MODELS = 2
_SPATIAL_PAIR_LENGTH = 2


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
            "architecture_family": family,
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
                unit = field_labels.display_unit(_PHYSICS_LABELS[metric][1])
                row[f"{metric}_median [{unit}]"] = _finite_quantile(frame, metric, 0.5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


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
    return tables.styled_dataframe(visible, title=title)


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
    """Show the steady speed-ups with their paired and unavailable case counts, batch size and hardware."""
    return tables.styled_dataframe(build_runtime_table(datasets), title="Runtime and speed-up (ratio of sums over paired cases)")


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
    inference seconds per case of that first frame's artifact.
    """
    return build_work_table({label: (summary, inference(label, frames[label])) for label, summary in run_discovery.distinct_runs(frames).items()})


def _steady_inference_seconds(_label: str, frame: pd.DataFrame) -> float:
    """Return the mean batch-amortized forward seconds per case of a steady artifact."""
    cases = _provenance(frame)["timing"]["report"]["cases"]
    return float(np.mean([case["batch_amortized_neural_operator_forward_s"] for case in cases]))


def plot_work_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Show the work quantities of every steady run, with its mean batch-amortized forward seconds per case."""
    return tables.styled_dataframe(build_frame_work_table(datasets, _steady_inference_seconds), title="Work per stage")


def plot_relative_comparison_scoreboard(*, datasets: dataset_cuts.DisplayFrames) -> Figure:
    """Plot each current normalized aggregate relative to its displayed best: a panel per dataset, the diagnostics as groups of one bar per model."""
    dataframe.validate_comparison(datasets)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    if not groups or any(len(models) < _MINIMUM_SCOREBOARD_MODELS for models in groups.values()):
        msg = "Relative comparison scoreboard requires at least two models per dataset."
        raise ValueError(msg)
    figure_title, count_headings = layout.dataset_title_context(
        "Comparison-relative normalized error", groups, {key: len(frame) for key, frame in datasets.items()}
    )
    group_ids = tuple(str(group_id) for group_id, _fields in datasets[next(iter(datasets))].attrs["output_groups"])
    diagnostic_labels = ("Task macro", *(f"Group: {group_id}" for group_id in group_ids))
    figure, axes = figure_layout.panel_figure(1, len(groups), sharey="row")
    positions = np.arange(len(diagnostic_labels), dtype=float)
    width = 0.8 / max(len(models) for models in groups.values())
    unavailable: list[str] = []
    tied_panels = []
    for axis, (dataset, models) in zip(axes[0], groups.items(), strict=True):
        # An unavailable aggregate stays NaN: it gets no bar and the note names its support counts instead.
        raw = np.full((len(models), len(diagnostic_labels)), np.nan)
        for row, (model, key) in enumerate(models.items()):
            aggregate = datasets[key].attrs["normalized_group_macro_rmse"]
            if not aggregate["available"]:
                unavailable.append(f"{model} on {dataset}: {validity_contract.unavailable_text(aggregate)}")
            else:
                statistics = aggregate["group_statistics"]
                if tuple(str(group_id) for group_id in statistics) != group_ids:
                    msg = "Relative comparison scoreboard requires identical ordered task groups."
                    raise dataframe.ComparisonCompatibilityError(msg)
                raw[row] = [float(aggregate["value"]), *(float(statistics[group_id]["normalized_rmse"]) for group_id in group_ids)]
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
        tied_panels.append(bool(available.all() and np.allclose(relative, 1.0, rtol=1e-6, atol=1e-12, equal_nan=False)))
        for row, key in enumerate(models.values()):
            shown = np.isfinite(relative[row])
            offset = (row - (len(models) - 1) / 2.0) * width
            axis.bar(positions[shown] + offset, relative[row, shown], width, color=styles[key].color, edgecolor="white", hatch=styles[key].hatch)
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
    layout.add_series_legend(figure, axes, datasets, groups, styles, bars=True)
    not_plotted = ["Not plotted: " + "; ".join(unavailable)] if unavailable else []
    note = layout.series_note(styles, normalization, *not_plotted, "This is a descriptive comparison, not an overall ranking")
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
    return tables.styled_dataframe(table.transpose(), title="Model-family configuration", row_local=True)


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
        "Accuracy versus physical consistency", groups, {key: len(frame) for key, frame in datasets.items()}
    )
    figure, axes = figure_layout.panel_figure(
        len(groups), len(dataframe.STEADY_PHYSICS_METRICS), panel_in=(4.2, 3.6)
    )  # taller for the long diagnostic labels
    unavailable: list[str] = []
    for row, (dataset, models) in enumerate(groups.items()):
        for model, key in models.items():
            aggregate = datasets[key].attrs["normalized_group_macro_rmse"]
            if not aggregate["available"]:
                unavailable.append(f"{model} on {dataset}: {validity_contract.unavailable_text(aggregate)}")
                continue
            primary = float(aggregate["value"])
            _family, _count, physics_enabled = series.model_metadata(datasets[key])
            style = styles[key]
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
                    facecolors=style.color if physics_enabled else "none",
                    edgecolors=style.color,
                    linewidths=1.5,
                )
        for axis, metric in zip(axes[row], dataframe.STEADY_PHYSICS_METRICS, strict=True):
            metric_label, unit = _PHYSICS_LABELS[metric]
            figure_layout.set_log_scale(axis, "y")
            axis.set_xlabel(f"Normalized task error [{field_labels.display_unit('1')}]")
            axis.set_ylabel(f"Median {metric_label}\n[{field_labels.display_unit(unit)}]")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=_PARETO_ERROR_TICKS))  # long decimals crowd a narrow error range
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
            if row == 0:
                figure_layout.set_panel_title(axis, _PHYSICS_TITLES[metric])
    figure_layout.set_row_headers(figure, axes, [count_headings[dataset] or dataset for dataset in groups])
    layout.add_series_legend(figure, axes, datasets, groups, styles, name_roles=False)  # the panel row names the dataset and so its role
    not_plotted = "Not plotted: " + "; ".join(unavailable) if unavailable else None
    figure_layout.set_figure_title(figure, axes, figure_title, note=layout.series_note(styles, not_plotted))
    return figure
