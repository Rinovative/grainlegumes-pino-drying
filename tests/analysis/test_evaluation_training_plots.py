"""Protect the numerics and structure of the training, study, learning-curve and break-even plots.

Responsibilities:
  - Check validation positions on the shared epoch axis, the selected-checkpoint star and unavailable histories
  - Check the study counts per outcome group and the best-so-far step line
  - Check the learning-curve median and seed range per material, architecture and train size
  - Check the break-even pairing and n* against hand values, including «never»

Design principles:
  - Inputs are tiny synthetic records; expected numbers are computed by hand in the tests

This module does NOT:
  - Read run directories or study files; the run-discovery tests cover the readers
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from src.analysis.evaluation import evaluation_run_discovery as discovery
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.evaluation.evaluation_plot import evaluation_plot_training as training


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


def test_training_curves_mark_the_selected_checkpoint_and_name_unavailable_runs() -> None:
    """The star sits at the selected point; stages of one pipeline share a colour and differ by line style; unavailable runs are named."""
    histories: dict[str, Any] = {
        "A0": _history("A0"),
        "B": _history("B", start_epoch=2),
        "old": discovery.UnavailableHistory(Path("old"), "no history"),
    }
    figure = training.plot_training_curves(histories)

    loss_axis, objective_axis = figure.axes[:2]
    stars = [line for line in objective_axis.get_lines() if line.get_marker() == "*"]
    assert sorted((float(_values(line.get_xdata())[0]), float(_values(line.get_ydata())[0])) for line in stars) == [(1.4, 0.2), (3.4, 0.2)]
    a0_loss, b_loss = loss_axis.get_lines()
    assert a0_loss.get_color() == b_loss.get_color()
    assert (a0_loss.get_linestyle(), b_loss.get_linestyle()) == ("-", "--")
    np.testing.assert_allclose(_values(b_loss.get_xdata()), [3.0, 4.0])
    assert any("old: no history" in text.get_text() for text in figure.findobj(Text))

    with pytest.raises(ValueError, match="No training history"):
        training.plot_training_curves({"old": histories["old"]})


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
