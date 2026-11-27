"""Plot how models train and what they cost: training curves, study progress, learning curves and the break-even case count.

Responsibilities:
  - Draw each stage's training loss per completed epoch and its validation objective at every validation point, marking the
    selected checkpoint, with B and A1 continuing from their A0 handoff epoch
  - Draw an Optuna study's trial values by outcome group and its best value so far as a step line, with the counts per group
  - Reduce a learning-curve table to the median and the seed range per material, architecture and train size, and draw it
  - Pair per-case COMSOL and surrogate seconds into one cost per model and draw cumulative COMSOL against surrogate cost with
    the break-even case count

Design principles:
  - Colour comes from ``plots.series.resolve_styles``; in training curves and the break-even a run takes the colour the
    evaluation comparison gives it when given, and a training curve's line style is the stage (A0 and steady solid, B dashed,
    A1 dash-dot)
  - A positive axis spanning less than a decade stays linear with plain decimal ticks
  - Histories are read only through ``run_discovery.stage_history``; this module never opens run files
  - Break-even n* = (C_gen + C_train) / (t_c - t_s) with C_gen = t_c x train cases; a surrogate not faster per case never breaks even
  - Values that cannot be drawn on the chosen axes raise ``ValueError`` instead of being clipped or dropped

This module does NOT:
  - Discover or admit runs, read summaries, timing reports or study files, or decide which runs a view compares
  - Count failed candidates or validation cases in the generation cost
  - Own figure geometry, legend placement or the series colours
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

from src.analysis.evaluation import evaluation_run_discovery as run_discovery
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_STAGE_LINESTYLES: Final = {"steady": "-", "A0": "-", "B": "--", "A1": "-."}
_LOSS_KEY: Final = "train/loss_total"
_ELIGIBLE_KEY: Final = "validation/eligible"
_SELECTED_MARKER_SIZE: Final = 13.0
_VALIDATION_MARKER_SIZE: Final = 3.5
# Outcome groups of the study curve: finished trials trained to their budget (a duplicate reuses a completed training).
STUDY_GROUPS: Final[Mapping[str, tuple[str, ...]]] = {
    "Finished": run_discovery.STUDY_FINISHED_STATUSES,
    "Pruned": ("pruned", "nonfinite_pruned", "oom_pruned", "duplicate_pruned"),
    "Not eligible": ("no_eligible_checkpoint",),
    "Diverged": ("diverged",),
}
_STUDY_STYLES: Final = {"Finished": ("o", "#0072B2"), "Pruned": ("x", "#7F7F7F"), "Not eligible": ("^", "#E69F00"), "Diverged": ("v", "#D62728")}
LEARNING_CURVE_COLUMNS: Final = ("material", "architecture", "train_cases_per_material", "seed", "error")
_BAND_ALPHA: Final = 0.2
_SECONDS_PER_HOUR: Final = 3600.0
_BREAK_EVEN_SPAN: Final = 2.0  # the cost axis runs to twice the largest break-even (or train) case count
_LOG_SPAN: Final = 10.0  # a positive range narrower than a decade reads better on a linear axis with plain decimal ticks
_ROUND_TRIP_DECIMALS: Final = 17  # decimals that write any float64 n* exactly, so its integer part is never rounded up
_NBSP: Final = "\N{NO-BREAK SPACE}"  # keeps a number with its unit and a count with its noun on one line of the note
_T_C: Final = r"$t_\mathrm{c}$"
_T_S: Final = r"$t_\mathrm{s}$"
_C_GEN: Final = r"$C_\mathrm{gen}$"
_C_TRAIN: Final = r"$C_\mathrm{train}$"
_N_STAR: Final = r"$n^*$"


@dataclass(frozen=True, slots=True)
class BreakEvenCost:
    """
    Hold one model's paired per-case seconds and fixed costs; the entity's label names the model.

    ``comsol_seconds`` (t_c) and ``surrogate_seconds`` (t_s) are means over the ``paired_cases`` cases that have both clocks;
    ``unpaired_cases`` had only one. ``clocks`` names the clocks and hardware for the runtime details below the figure.
    """

    entity: series.SeriesEntity
    comsol_seconds: float
    surrogate_seconds: float
    training_seconds: float
    train_cases: int
    paired_cases: int
    unpaired_cases: int
    clocks: str

    @property
    def generation_seconds(self) -> float:
        """Return C_gen: the mean paired COMSOL seconds times the run's train cases."""
        return self.comsol_seconds * self.train_cases

    @property
    def fixed_seconds(self) -> float:
        """Return C_gen + C_train, the cost the surrogate pays before its first prediction."""
        return self.generation_seconds + self.training_seconds

    @property
    def break_even_cases(self) -> float | None:
        """Return n* = (C_gen + C_train) / (t_c - t_s), or ``None`` (never) when the surrogate is not faster per case."""
        saving = self.comsol_seconds - self.surrogate_seconds
        return self.fixed_seconds / saving if saving > 0.0 else None


def _positive_seconds(values: Mapping[str, float], *, label: str) -> None:
    """Refuse a per-case time that is not finite and positive."""
    for case, seconds in values.items():
        if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not math.isfinite(seconds) or seconds <= 0.0:
            message = f"{label} seconds of case {case!r} must be finite and positive, got {seconds!r}."
            raise ValueError(message)


def break_even_cost(
    entity: series.SeriesEntity,
    *,
    comsol_case_seconds: Mapping[str, float],
    surrogate_case_seconds: Mapping[str, float],
    training_seconds: float,
    train_cases: int,
    clocks: str,
) -> BreakEvenCost:
    """
    Pair per-case COMSOL and surrogate seconds by case id and return one model's break-even cost.

    t_c and t_s are the means over the cases that have both clocks; the cases with one clock only are counted as unpaired.

    Raises
    ------
    ValueError
        If a time is not finite and positive, the training seconds are negative or not finite, the train cases are not a
        positive integer, or no case has both clocks.

    """
    _positive_seconds(comsol_case_seconds, label="COMSOL")
    _positive_seconds(surrogate_case_seconds, label="Surrogate")
    if not math.isfinite(training_seconds) or training_seconds < 0.0:
        message = f"Training seconds must be finite and non-negative, got {training_seconds!r}."
        raise ValueError(message)
    if isinstance(train_cases, bool) or not isinstance(train_cases, int) or train_cases < 1:
        message = f"Train cases must be a positive integer, got {train_cases!r}."
        raise ValueError(message)
    paired = sorted(set(comsol_case_seconds) & set(surrogate_case_seconds))
    if not paired:
        message = f"{entity.label}: no case has both a COMSOL and a surrogate time."
        raise ValueError(message)
    return BreakEvenCost(
        entity=entity,
        comsol_seconds=math.fsum(comsol_case_seconds[case] for case in paired) / len(paired),
        surrogate_seconds=math.fsum(surrogate_case_seconds[case] for case in paired) / len(paired),
        training_seconds=float(training_seconds),
        train_cases=train_cases,
        paired_cases=len(paired),
        unpaired_cases=len(set(comsol_case_seconds) ^ set(surrogate_case_seconds)),
        clocks=clocks,
    )


def _log_scale_when_positive(axis: Axes, values: Sequence[float]) -> None:
    """Use a logarithmic y axis when every drawn value is positive over at least a decade, else a linear one with plain decimals."""
    if values and min(values) > 0.0 and max(values) >= _LOG_SPAN * min(values):
        figure_layout.set_log_scale(axis, "y")
        return
    plain = ScalarFormatter(useOffset=False)
    plain.set_scientific(False)
    axis.yaxis.set_major_formatter(plain)


def _three_digits(value: float) -> str:
    """Return ``value`` positionally with three significant digits and their trailing zeros (``2.00``, ``0.00222``, ``600``)."""
    return np.format_float_positional(value, precision=3, unique=False, fractional=False, trim="k").removesuffix(".")


def _crossing_text(n_star: float) -> str:
    """Return n* with one decimal, or as many more as keep it below the integer it lies below (9.96, never 10.0)."""
    return next(text for digits in range(1, _ROUND_TRIP_DECIMALS + 1) if math.floor(float(text := f"{n_star:.{digits}f}")) == math.floor(n_star))


def _entity(label: str, architecture: str, run_key: str) -> series.SeriesEntity:
    """Describe one ID series of an architecture (``fno``, ``pi-fno``, ...) and the run it belongs to."""
    family = architecture.removeprefix("pi-")
    return series.SeriesEntity(label=label, family=family, physics_enabled=family != architecture, role="id", materials=(), run_key=run_key)


def _objective_label(objective_id: str, direction: str) -> str:
    """Return the y label of a (normalized, dimensionless) objective in its shared display label, with the direction that is better."""
    label = display_labels.evaluation_protocol_display_label(objective_id)
    return f"{label} [{field_labels.display_unit('1')}]\n({'lower' if direction == 'minimize' else 'higher'} is better)"


def validation_positions(history: run_discovery.StageHistory) -> np.ndarray:
    """
    Return the epoch position of every validation record on the shared axis.

    A validation sits at its stage's start epoch plus the completed epochs before it and the share of its own epoch's optimizer
    steps: the global step is interpolated between the steps at the ends of the completed epochs (0 at the stage start).
    """
    ends = np.array([0.0, *(record["global_step"] for record in history.epochs)])
    steps = np.array([record["global_step"] for record in history.validations], dtype=float)
    return history.start_epoch + np.interp(steps, ends, np.arange(ends.size, dtype=float))


def plot_training_curves(
    histories: Mapping[str, run_discovery.StageHistory],
    *,
    styles: series.SeriesStyles | None = None,
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot the training loss per epoch and the validation objective per validation point of every labelled stage.

    The left panel draws ``train/loss_total`` at the end of every completed epoch, the right one the objective at every
    eligible validation point (a point) and at the selected checkpoint (a star). The x axis is the epoch counted from the
    start of A0 (or of the steady run): B and A1 continue from their A0 handoff epoch. An ineligible validation's value
    is not bound to the checkpoint, so it is not drawn; the note counts such validations. A stage's colour is its run's:
    ``styles`` (by history label) gives the colours the evaluation views of the same comparison use; without it every stage
    run is its own series. The line style is always the stage. The title ends with the short labels ``models`` of the compared
    runs (``layout.models_suffix``).

    Raises
    ------
    ValueError
        If no history is given, the histories validate different objectives, or ``styles`` lacks a history label.

    """
    if not histories:
        message = "No training history is available."
        raise ValueError(message)
    objectives = {(item.objective_id, item.objective_direction) for item in histories.values()}
    if len(objectives) != 1:
        message = f"Training curves compare stages of one validation objective; got {sorted(objectives)}."
        raise ValueError(message)
    ((objective_id, direction),) = objectives
    objective_key = f"id/{objective_id}"
    if styles is None:  # each stage is its own run, as in the evaluation views, where B and A1 differ by checkpoint
        styles = series.resolve_styles([_entity(label, item.model_family, item.run_name) for label, item in histories.items()])
    elif missing := sorted(set(histories).difference(styles)):
        message = f"Training-curve styles lack the histories {missing}."
        raise ValueError(message)
    figure, axes = figure_layout.panel_figure(1, 2, sharex=True)
    loss_axis, objective_axis = axes[0]
    handles: list[Artist] = []
    losses: list[float] = []
    objectives_drawn: list[float] = []
    ineligible = 0
    for label, history in histories.items():
        color, linestyle = styles[label].color, _STAGE_LINESTYLES[history.stage]
        loss = [record[_LOSS_KEY] for record in history.epochs]
        loss_axis.plot([history.start_epoch + record["epoch"] for record in history.epochs], loss, color=color, linestyle=linestyle)
        placed = list(zip(validation_positions(history), history.validations, strict=True))
        # The validator binds the objective only of an eligible validation; the selected point is always eligible.
        drawn = [(float(x), record[objective_key]) for x, record in placed if record[_ELIGIBLE_KEY] == 1.0]
        ineligible += len(placed) - len(drawn)
        selected = next((float(x), record[objective_key]) for x, record in placed if record["validation/point"] == history.selected_point)
        (line,) = objective_axis.plot(*zip(*drawn, strict=True), color=color, linestyle=linestyle, marker="o", markersize=_VALIDATION_MARKER_SIZE)
        objective_axis.plot(*selected, marker="*", markersize=_SELECTED_MARKER_SIZE, color=color, markeredgecolor="black", linestyle="none", zorder=3)
        handles.append(line)
        losses.extend(loss)
        objectives_drawn.extend(value for _x, value in drawn)
    _log_scale_when_positive(loss_axis, losses)
    _log_scale_when_positive(objective_axis, objectives_drawn)
    figure_layout.set_panel_title(loss_axis, "Training loss per epoch")
    figure_layout.set_panel_title(objective_axis, "Validation objective")
    loss_axis.set_ylabel(f"Total training loss [{field_labels.display_unit('1')}]")
    objective_axis.set_ylabel(_objective_label(objective_id, direction))
    for axis in axes[0]:
        axis.set_xlabel("Epoch from the start of A0" if any(item.stage != "steady" for item in histories.values()) else "Epoch")
        axis.grid(True, linestyle="--", alpha=0.3)
    selected_handle = Line2D([], [], marker="*", markersize=_SELECTED_MARKER_SIZE, color="white", markeredgecolor="black", linestyle="none")
    figure_layout.add_legend(figure, axes, [*handles, selected_handle], [*histories, "Selected checkpoint"])
    note = "Line style: stage (steady and A0 solid, B dashed, A1 dash-dot); points: validations; B and A1 start at the A0 handoff"
    skipped = [f"{ineligible} ineligible validation{'s' if ineligible > 1 else ''} not drawn"] if ineligible else []
    figure_layout.set_figure_title(figure, axes, "Training curves" + layout.models_suffix(models), note=layout.series_note(styles, note, *skipped))
    return figure


def study_counts(summary: run_discovery.StudySummary) -> tuple[dict[str, int], dict[str, int]]:
    """Return the trial count of every outcome group (``STUDY_GROUPS``) and of every other status with trials."""
    grouped = {group: sum(summary.trial_counts.get(status, 0) for status in statuses) for group, statuses in STUDY_GROUPS.items()}
    named = {status for statuses in STUDY_GROUPS.values() for status in statuses}
    other = {status: count for status, count in summary.trial_counts.items() if status not in named and count}
    return grouped, other


def plot_study_curve(summary: run_discovery.StudySummary) -> Figure:
    """
    Plot every trial's objective value by outcome group over its trial number and the best value so far as a step line.

    Trials without a value (most pruned or diverged ones) have no point; the note counts every group and every other status.

    Raises
    ------
    ValueError
        If no trial has an objective value.

    """
    groups = {status: group for group, statuses in STUDY_GROUPS.items() for status in statuses}
    best = [(trial.number, trial.best_so_far) for trial in summary.trials if trial.best_so_far is not None]
    valued = [trial for trial in summary.trials if trial.value is not None]
    if not valued:
        message = f"No trial of study {summary.study_name!r} has an objective value."
        raise ValueError(message)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(6.4, 3.6))
    axis = axes[0, 0]
    grouped, other = study_counts(summary)
    handles: list[Artist] = []
    labels: list[str] = []
    for group, (marker, color) in _STUDY_STYLES.items():
        members = [trial for trial in valued if groups.get(trial.status) == group]
        if members:
            numbers, values = [trial.number for trial in members], [trial.value for trial in members]
            handles.append(axis.scatter(numbers, values, marker=marker, color=color, s=18, zorder=2))
            labels.append(f"{group} ({grouped[group]})")
    if best:
        (step,) = axis.step(*zip(*best, strict=True), where="post", color="black", linewidth=1.2)
        handles.append(step)
        labels.append("Best so far")
    axis.set_xlabel("Trial number")
    axis.set_ylabel(_objective_label(summary.objective_id, summary.direction))
    axis.grid(True, linestyle="--", alpha=0.3)
    figure_layout.add_legend(figure, axes, handles, labels)
    counts = ", ".join(f"{group.lower()} {count}" for group, count in grouped.items())
    rest = "; other: " + ", ".join(f"{status} {count}" for status, count in other.items()) if other else ""
    figure_layout.set_figure_title(figure, axes, f"Study progress: {summary.study_name}", note=f"Trials: {counts}{rest}")
    return figure


def learning_curve_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return the median, minimum, maximum and count of the error over seeds per material, architecture and train size.

    ``frame`` has one row per run with ``LEARNING_CURVE_COLUMNS``; errors must be finite and positive (log axes), train sizes
    positive integers, and every (material, architecture, train size, seed) unique.

    Raises
    ------
    ValueError
        If a column is missing or a value or row violates the rules above.

    """
    missing = [column for column in LEARNING_CURVE_COLUMNS if column not in frame.columns]
    if missing:
        message = f"Learning-curve frame lacks the columns {missing}."
        raise ValueError(message)
    errors = pd.to_numeric(frame["error"], errors="coerce").to_numpy(dtype=float)
    sizes = frame["train_cases_per_material"]
    if not (np.isfinite(errors).all() and (errors > 0.0).all()):
        message = "Learning-curve errors must be finite and positive."
        raise ValueError(message)
    if not all(isinstance(size, int | np.integer) and not isinstance(size, bool) and size >= 1 for size in sizes):
        message = "Learning-curve train sizes must be positive integers."
        raise ValueError(message)
    keys = ["material", "architecture", "train_cases_per_material"]
    if frame.duplicated([*keys, "seed"]).any():
        message = "Learning-curve frame repeats a seed of one material, architecture and train size."
        raise ValueError(message)
    statistics = frame.groupby(keys, sort=False)["error"].agg(median="median", low="min", high="max", seeds="count")
    return statistics.reset_index().sort_values(keys, kind="stable", ignore_index=True)


def plot_learning_curve(frame: pd.DataFrame, *, error_label: str) -> Figure:
    """
    Plot the error over train cases per material: one panel per material, one line per architecture with its seed band.

    The line is the median over seeds and the band runs from the smallest to the largest seed error, both on log axes
    (base 2 for the nested train sizes). ``error_label`` names the error with its unit on the y axis.
    """
    statistics = learning_curve_statistics(frame)
    materials = list(dict.fromkeys(frame["material"]))
    architectures = list(dict.fromkeys(frame["architecture"]))
    styles = series.resolve_styles([_entity(architecture, architecture, architecture) for architecture in architectures])
    figure, axes = figure_layout.panel_figure(1, len(materials), sharey="row")
    handles: dict[str, Artist] = {}
    for axis, material in zip(axes[0], materials, strict=True):
        for architecture in architectures:
            rows = statistics[(statistics["material"] == material) & (statistics["architecture"] == architecture)]
            if rows.empty:
                continue
            color = styles[architecture].color
            (line,) = axis.plot(rows["train_cases_per_material"], rows["median"], color=color, marker="o", markersize=4)
            axis.fill_between(rows["train_cases_per_material"], rows["low"], rows["high"], color=color, alpha=_BAND_ALPHA, linewidth=0)
            handles.setdefault(visual_semantics.architecture_label(architecture), line)
        axis.set_xscale("log", base=2)
        figure_layout.set_log_scale(axis, "y")
        axis.set_xlabel("Train cases per material")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
        figure_layout.set_panel_title(axis, display_labels.material_display_label(str(material)))
    figure_layout.label_rows_once(axes, error_label)
    band = Patch(color="grey", alpha=_BAND_ALPHA)
    figure_layout.add_legend(figure, axes, [*handles.values(), band], [*handles, "Seed range (min to max)"])
    seeds = statistics["seeds"]
    note = f"Line: median over seeds; band: smallest to largest seed error; {seeds.min()} to {seeds.max()} seeds per point"
    figure_layout.set_figure_title(figure, axes, "Learning curve", note=note)
    return figure


def plot_break_even(
    costs: Sequence[BreakEvenCost],
    *,
    styles: series.SeriesStyles | None = None,
    unavailable: Sequence[str] = (),
    models: Sequence[str] = (),
) -> Figure:
    """
    Plot cumulative COMSOL cost n t_c against the surrogate's C_gen + C_train + n t_s, one panel per model.

    The dotted line marks the break-even n*, where the costs cross; the note names each model's role and gives n* and the
    first case from which the surrogate is cheaper (n* keeps the decimals that hold it below that case), the seconds and hours
    three significant digits; a surrogate not faster per case is marked «never». The cost axis runs to twice the largest
    finite n* (or twice the largest train case count when no model breaks even). ``styles`` (by label) are the colours the
    other views give the runs, else each model is its own series. Below the figure stand the runs ``unavailable`` names as
    lacking evidence; the clocks and hardware (``BreakEvenCost.clocks``) are the caller's runtime details, never figure text.
    The title ends with the short labels ``models``.

    Raises
    ------
    ValueError
        If no cost is given, two share a label, or ``styles`` lacks one.

    """
    labels = [cost.entity.label for cost in costs]
    if not costs or len(set(labels)) != len(costs):
        message = "Break-even needs at least one model cost, each with its own label."
        raise ValueError(message)
    if styles is None:
        styles = series.resolve_styles([cost.entity for cost in costs])
    elif missing := sorted(set(labels).difference(styles)):
        message = f"Break-even styles lack the models {missing}."
        raise ValueError(message)
    finite = [n for cost in costs if (n := cost.break_even_cases) is not None]
    span = _BREAK_EVEN_SPAN * (max(finite) if finite else max(cost.train_cases for cost in costs))
    cases = np.linspace(0.0, span, 200)
    figure, axes = figure_layout.panel_figure(1, len(costs), sharey="row")
    handles: list[Artist] = []
    markers: list[Artist] = []
    notes: list[str] = []
    for axis, cost in zip(axes[0], costs, strict=True):
        style = styles[cost.entity.label]
        (comsol,) = axis.plot(cases, cases * cost.comsol_seconds / _SECONDS_PER_HOUR, color="black")
        (surrogate,) = axis.plot(
            cases, (cost.fixed_seconds + cases * cost.surrogate_seconds) / _SECONDS_PER_HOUR, color=style.color, linestyle=style.linestyle
        )
        handles.append(surrogate)
        n_star = cost.break_even_cases
        if n_star is None:
            axis.text(0.98, 0.04, "Break-even: never", transform=axis.transAxes, ha="right", va="bottom", fontsize=figure_layout.TICK_PT)
            outcome = r"never ($t_\mathrm{s} \geq t_\mathrm{c}$)"
        else:
            # Marked as a reference line: it spans the view, so the key clears it without counting it as data extent.
            markers.append(axis.axvline(n_star, color="grey", linestyle=":", linewidth=1.2, gid=figure_layout.REFERENCE_GID))
            outcome = rf"{_N_STAR} = {_crossing_text(n_star)}, cheaper from case{_NBSP}{math.floor(n_star) + 1}"
        axis.set_xlabel("Predicted cases n")
        axis.grid(True, linestyle="--", alpha=0.3)
        if len(costs) > 1:  # one panel is named by the figure title and the legend
            figure_layout.set_panel_title(axis, cost.entity.label)
        generation, training = (_three_digits(seconds / _SECONDS_PER_HOUR) for seconds in (cost.generation_seconds, cost.training_seconds))
        notes.append(
            f"{cost.entity.label} · {display_labels.evaluation_role_display_label(cost.entity.role)}: "
            f"{_T_C} = {_three_digits(cost.comsol_seconds)}{_NBSP}s, {_T_S} = {_three_digits(cost.surrogate_seconds)}{_NBSP}s per case over "
            f"{cost.paired_cases}{_NBSP}paired{_NBSP}cases ({cost.unpaired_cases}{_NBSP}unpaired)\n"
            f"{_C_GEN} = {generation}{_NBSP}h ({cost.train_cases}{_NBSP}train{_NBSP}cases{_NBSP}\N{MULTIPLICATION SIGN}{_NBSP}{_T_C}), "
            f"{_C_TRAIN} = {training}{_NBSP}h; {outcome}"
        )
    figure_layout.label_rows_once(axes, f"Cumulative wall time [{field_labels.display_unit('h')}]")
    formula = r"$C_\mathrm{gen} + C_\mathrm{train} + n\,t_\mathrm{s}$"
    legend_labels = [r"COMSOL: $n\,t_\mathrm{c}$", *(f"{label}: {formula}" for label in labels), *([r"Break-even $n^*$"] if markers else [])]
    figure_layout.add_legend(figure, axes, [comsol, *handles, *markers[:1]], legend_labels)
    notes.append("Generation cost counts neither failed candidates nor validation cases")
    title = "Break-even against COMSOL" + layout.models_suffix(models)
    figure_layout.set_figure_title(figure, axes, title, note=layout.series_note(styles, *notes))
    if unavailable:
        figure.supxlabel("Without break-even evidence: " + "; ".join(unavailable), fontsize=figure_layout.NOTE_PT)
    return figure
