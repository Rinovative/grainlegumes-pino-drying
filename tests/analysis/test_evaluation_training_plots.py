"""Protect the numerics and structure of the training, study, learning-curve and break-even plots.

Responsibilities:
  - Check validation positions on the shared epoch axis, the selected-checkpoint star
  - Check that both training-curve axes are log only over a positive decade and write plain decimals when linear
  - Check the study counts per outcome group and the best-so-far step line
  - Check the learning-curve median and seed range per material, architecture and train size
  - Check the break-even pairing and n* against hand values, including «never», with n* kept on its side of the integer and
    the clocks in three significant digits, its note typeset with numbers and units kept together

Design principles:
  - Inputs are tiny synthetic records; expected numbers are computed by hand in the tests

This module does NOT:
  - Read run directories or study files; the run-discovery tests cover the readers
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from src.analysis.evaluation import evaluation_run_discovery as discovery
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.evaluation.evaluation_plot import evaluation_plot_training as training

_NBSP = "\N{NO-BREAK SPACE}"


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """Close every figure a test opened."""
    yield
    plt.close("all")


def _values(array_like: Any) -> np.ndarray:
    """Return matplotlib's ``ArrayLike`` artist data as a float array."""
    return np.asarray(array_like, dtype=float)


def _history(stage: str, *, start_epoch: int = 0, pipeline: str = "a0", last_eligible: float = 1.0) -> discovery.StageHistory:
    """Return a two-epoch history of 10 steps per epoch, validated mid epoch 2 (the best) and at its end (eligible as given)."""
    return discovery.StageHistory(
        run_name=f"{pipeline}-{stage}",
        stage=stage,
        model_family="fno",
        pipeline=pipeline,
        start_epoch=start_epoch,
        objective_id="rmse",
        objective_direction="minimize",
        selected_point=10,
        epochs=({"epoch": 1.0, "global_step": 10.0, "train/loss_total": 1.0}, {"epoch": 2.0, "global_step": 20.0, "train/loss_total": 0.5}),
        validations=(
            {"epoch": 2.0, "global_step": 14.0, "validation/point": 10.0, "validation/eligible": 1.0, "id/rmse": 0.2},
            {"epoch": 2.0, "global_step": 20.0, "validation/point": 20.0, "validation/eligible": last_eligible, "id/rmse": 0.3},
        ),
    )


def test_validations_sit_at_their_epoch_share_after_the_handoff() -> None:
    """A validation 4 of 10 steps into epoch 2 of a stage starting at epoch 5 sits at 6.4; one at the epoch end at 7."""
    np.testing.assert_allclose(training.validation_positions(_history("B", start_epoch=5)), [6.4, 7.0])


def test_training_curves_mark_the_selected_checkpoint_and_refuse_no_history() -> None:
    """The star sits at the selected point; each stage is its own run, in its stage line style; no history is refused."""
    histories = {"A0": _history("A0"), "B": _history("B", start_epoch=2)}
    figure = training.plot_training_curves(histories)

    loss_axis, objective_axis = figure.axes[:2]
    stars = [line for line in objective_axis.get_lines() if line.get_marker() == "*"]
    assert sorted((float(_values(line.get_xdata())[0]), float(_values(line.get_ydata())[0])) for line in stars) == [(1.4, 0.2), (3.4, 0.2)]
    a0_loss, b_loss = loss_axis.get_lines()
    assert a0_loss.get_color() != b_loss.get_color()  # two runs, as in the evaluation views
    assert (a0_loss.get_linestyle(), b_loss.get_linestyle()) == ("-", "--")
    np.testing.assert_allclose(_values(b_loss.get_xdata()), [3.0, 4.0])
    assert objective_axis.get_ylabel().startswith("RMSE [-]\n")  # the shared metric label with its unit, not the raw id

    with pytest.raises(ValueError, match="No training history"):
        training.plot_training_curves({})


def test_training_curves_take_the_evaluation_colour_of_each_run_and_keep_the_stage_line_style() -> None:
    """Given comparison styles, B and A1 of one A0 take the colours the evaluation views give their runs; a missing label is refused."""
    histories = {"B": _history("B", start_epoch=2), "A1": _history("A1", start_epoch=2)}
    styles = series.resolve_styles([training._entity(label, "fno", label) for label in histories])
    figure = training.plot_training_curves(histories, styles=styles)

    b_loss, a1_loss = figure.axes[0].get_lines()
    assert (b_loss.get_color(), a1_loss.get_color()) == (styles["B"].color, styles["A1"].color)
    assert b_loss.get_color() != a1_loss.get_color()
    assert (b_loss.get_linestyle(), a1_loss.get_linestyle()) == ("--", "-.")
    with pytest.raises(ValueError, match="lack the histories"):
        training.plot_training_curves(histories, styles=series.SeriesStyles({"B": styles["B"]}))


@pytest.mark.parametrize(
    ("values", "scale"),
    [((2.0e-5, 1.5e-5), "linear"), ((1.0e-4, 1.0e-5), "log"), ((2.0e-5, 0.0), "linear"), ((2.0e-5, -1.0e-5), "linear")],
    ids=["below a decade", "exactly a decade", "zero", "negative"],
)
def test_both_training_curve_axes_take_one_decade_rule_and_a_linear_axis_writes_plain_decimals(values: tuple[float, float], scale: str) -> None:
    """The loss and the validation axis are log only over a positive decade or more; linear, their ticks are plain decimals."""
    first, second = values
    history = replace(
        _history("A0"),
        epochs=({"epoch": 1.0, "global_step": 10.0, "train/loss_total": first}, {"epoch": 2.0, "global_step": 20.0, "train/loss_total": second}),
        validations=(
            {"epoch": 2.0, "global_step": 14.0, "validation/point": 10.0, "validation/eligible": 1.0, "id/rmse": first},
            {"epoch": 2.0, "global_step": 20.0, "validation/point": 20.0, "validation/eligible": 1.0, "id/rmse": second},
        ),
    )
    figure = training.plot_training_curves({"A0": history})
    figure.canvas.draw()

    for axis in figure.axes[:2]:
        assert axis.get_yscale() == scale
        if scale == "linear":
            assert not axis.yaxis.get_offset_text().get_text()  # neither an offset nor a power of ten beside the ticks
            assert all("e" not in label.get_text() for label in axis.get_yticklabels())


def test_a_break_even_note_keeps_each_crossing_on_its_side_of_the_integer_and_the_clocks_three_digits() -> None:
    """Crossings at 9.96 and 10.0 read apart, each before its own first cheaper case; the clock values keep their trailing zeros."""
    near, exact = (
        training.break_even_cost(
            _entity(label, family),
            comsol_case_seconds={"a": 2.0},
            surrogate_case_seconds={"a": 1.0},
            training_seconds=seconds,
            train_cases=cases,
            clocks="",
        )
        for label, family, seconds, cases in (("Near", "fno", 1.96, 4), ("Exact", "uno", 0.0, 5))  # n* = (8 + 1.96) / 1 and (10 + 0) / 1
    )
    note = " ".join(" ".join(text.get_text() for text in training.plot_break_even([near, exact]).findobj(Text)).split())

    assert r"$n^*$ = 9.96, cheaper from case 10" in note
    assert r"$n^*$ = 10.0, cheaper from case 11" in note
    assert r"$t_\mathrm{c}$ = 2.00 s, $t_\mathrm{s}$ = 1.00 s per case" in note


def test_a_single_break_even_panel_keeps_its_key_off_the_break_even_line_spanning_the_view() -> None:
    """The demo's U-NO costs put n* near the middle; the key takes a corner the vertical break-even line does not cross."""
    cost = training.break_even_cost(
        _entity("U-NO", "uno"),
        comsol_case_seconds={"a": 28.7},
        surrogate_case_seconds={"a": 0.00429},
        training_seconds=9936.0,
        train_cases=1024,
        clocks="",
    )
    figure = training.plot_break_even([cost])
    figure.canvas.draw()
    (axis,) = figure.axes
    (line,) = [line for line in axis.get_lines() if line.get_linestyle() == ":"]
    vertices = np.asarray(line.get_transform().transform_path(line.get_path()).vertices, dtype=np.float64)
    (crossing,) = {float(x) for x in vertices[:, 0]}
    legend = axis.get_legend()
    assert legend is not None
    key = legend.get_window_extent()

    assert not key.x0 <= crossing <= key.x1


def test_a_break_even_note_typesets_its_symbols_and_never_breaks_a_number_from_its_unit() -> None:
    """No raw formula token reaches the figure; a number, its unit and a case count stay together; one run has no panel title."""
    cost = training.break_even_cost(
        _entity("FNO", "fno"),
        comsol_case_seconds={"a": 2.93},
        surrogate_case_seconds={"a": 1.0},
        training_seconds=3600.0,
        train_cases=1393,
        clocks="",
    )
    figure = training.plot_break_even([cost], models=["FNO"])
    texts = [text.get_text() for text in figure.findobj(Text)]
    note = next(text for text in texts if f"paired{_NBSP}cases" in text)
    outside_math = "".join(note.split("$")[::2])

    assert not any(token in outside_math for token in ("t_c", "t_s", "C_gen", "C_train", "n*"))
    assert f"2.93{_NBSP}s" in note
    assert f"{_NBSP}h" in note
    assert f"1393{_NBSP}train{_NBSP}cases" in note
    assert f"from case{_NBSP}" in note
    assert not [axis for axis in figure.axes if axis.get_title()]  # the figure title names the single run


def test_training_curves_draw_only_eligible_validations_and_count_the_others() -> None:
    """A finite objective at an ineligible validation is unbound evidence: it gets no point and the note counts it."""
    figure = training.plot_training_curves({"A0": _history("A0", last_eligible=0.0)})

    (curve,) = [line for line in figure.axes[1].get_lines() if line.get_marker() == "o"]
    np.testing.assert_allclose(_values(curve.get_xdata()), [1.4])
    assert any("1 ineligible validation" in text.get_text() for text in figure.findobj(Text))


def _summary(trials: list[tuple[int, str, float | None, float | None]]) -> discovery.StudySummary:
    """Return a study summary whose counts are the trials' statuses."""
    counts: dict[str, int] = {}
    for _number, status, _value, _best in trials:
        counts[status] = counts.get(status, 0) + 1
    return discovery.StudySummary("study", "rmse", "minimize", counts, tuple(discovery.StudyTrial(*trial) for trial in trials))


def test_study_counts_group_outcomes_and_the_step_line_follows_the_best_so_far() -> None:
    """Completed and duplicate trials are finished, every pruned kind is pruned, other statuses stay apart."""
    summary = _summary(
        [
            (0, "completed", 0.5, 0.5),
            (1, "oom_pruned", None, 0.5),
            (2, "duplicate", 0.4, 0.4),
            (3, "no_eligible_checkpoint", None, 0.4),
            (4, "diverged", None, 0.4),
            (5, "failed", None, 0.4),
            (6, "pruned", 0.9, 0.4),
        ]
    )
    grouped, other = training.study_counts(summary)
    assert grouped == {"Finished": 2, "Pruned": 2, "Not eligible": 1, "Diverged": 1}
    assert other == {"failed": 1}

    figure = training.plot_study_curve(summary)
    (step,) = figure.axes[0].get_lines()
    np.testing.assert_allclose(_values(step.get_ydata()), [0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4])
    assert sum(len(_values(collection.get_offsets())) for collection in figure.axes[0].collections) == 3  # trials with a value


def _learning_frame() -> pd.DataFrame:
    """Return two seeds of two architectures at two train sizes for lentil, and one seed of FNO for chickpea."""
    rows = [
        ("lentil", "fno", 32, 1, 0.4),
        ("lentil", "fno", 32, 2, 0.2),
        ("lentil", "fno", 64, 1, 0.1),
        ("lentil", "fno", 64, 2, 0.3),
        ("lentil", "pi-fno", 32, 1, 0.5),
        ("lentil", "pi-fno", 32, 2, 0.6),
        ("chickpea", "fno", 32, 1, 0.7),
    ]
    return pd.DataFrame(rows, columns=list(training.LEARNING_CURVE_COLUMNS))


def test_learning_curve_reduces_seeds_to_median_and_range() -> None:
    """Median, minimum and maximum over seeds per material, architecture and size; a repeated seed is refused."""
    statistics = training.learning_curve_statistics(_learning_frame()).set_index(["material", "architecture", "train_cases_per_material"])
    np.testing.assert_allclose(statistics.loc[("lentil", "fno", 32), ["median", "low", "high"]].to_numpy(dtype=float), [0.3, 0.2, 0.4])
    np.testing.assert_allclose(statistics.loc[("lentil", "pi-fno", 32), ["median", "low", "high"]].to_numpy(dtype=float), [0.55, 0.5, 0.6])
    assert statistics.loc[("chickpea", "fno", 32), "seeds"] == 1

    figure = training.plot_learning_curve(_learning_frame(), error_label="Relative L2 [-]")
    assert len([axis for axis in figure.axes if axis.get_visible() and axis.lines]) == 2  # one panel per material

    with pytest.raises(ValueError, match="repeats a seed"):
        training.learning_curve_statistics(pd.concat([_learning_frame(), _learning_frame().iloc[:1]]))


def _entity(label: str, family: str) -> series.SeriesEntity:
    """Return the series entity of one model run."""
    return series.SeriesEntity(label=label, family=family, physics_enabled=False, role="id", materials=(), run_key=label)


def test_break_even_pairs_cases_and_matches_the_hand_value() -> None:
    """t_c and t_s are paired means; n* = (t_c x train + C_train) / (t_c - t_s); a slower surrogate never breaks even."""
    cost = training.break_even_cost(
        _entity("FNO", "fno"),
        comsol_case_seconds={"a": 100.0, "b": 140.0, "c": 500.0},
        surrogate_case_seconds={"a": 1.0, "b": 3.0, "d": 9.0},
        training_seconds=3600.0,
        train_cases=10,
        clocks="test clocks",
    )
    assert (cost.paired_cases, cost.unpaired_cases) == (2, 2)
    assert (cost.comsol_seconds, cost.surrogate_seconds) == (120.0, 2.0)
    assert math.isclose(cost.break_even_cases or math.nan, (120.0 * 10 + 3600.0) / (120.0 - 2.0))

    slow = training.break_even_cost(
        _entity("U-NO", "uno"),
        comsol_case_seconds={"a": 1.0},
        surrogate_case_seconds={"a": 1.0},
        training_seconds=0.0,
        train_cases=1,
        clocks="test clocks",
    )
    assert slow.break_even_cases is None

    figure = training.plot_break_even([cost, slow])
    panels = [axis for axis in figure.axes if axis.lines]
    assert len(panels) == 2
    (marker,) = [line for line in panels[0].get_lines() if line.get_linestyle() == ":"]
    assert math.isclose(float(_values(marker.get_xdata())[0]), cost.break_even_cases or math.nan)

    with pytest.raises(ValueError, match="both a COMSOL and a surrogate"):
        training.break_even_cost(
            _entity("FNO", "fno"), comsol_case_seconds={"a": 1.0}, surrogate_case_seconds={"b": 1.0}, training_seconds=0.0, train_cases=1, clocks=""
        )
