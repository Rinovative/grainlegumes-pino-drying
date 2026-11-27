"""Protect that a steady run keeps its style across the Evaluation views and that a legend keys every marker it draws.

Responsibilities:
  - Check that the break-even of a steady comparison draws its costable run in the colour the dataset views give it, with
    the run that has no training evidence named apart
  - Check that the Pareto legend keeps the filled or hollow run handles and keys each material by the marker its points take

Design principles:
  - Inputs are two runs of the small steady fixture; the panel's own scope cuts the datasets as a notebook does

This module does NOT:
  - Check numbers, figure geometry or the transient views
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from support.steady_evaluation import steady_frame

from src import analysis
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.presentation import dataset_selection as selection

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def _two_runs(tmp_path: Path) -> dict[str, pd.DataFrame]:
    """Return two FNO runs «A» and «B» of the steady fixture, each with one lentil and one chickpea case, «A» with its training evidence."""
    frame, _grid = steady_frame(tmp_path)
    frame.loc[1, "material_family"] = "chickpea"
    peer = frame.copy(deep=True)
    peer.attrs = copy.deepcopy(frame.attrs)
    peer.attrs["artifact_provenance"]["run"].update(name="steady-peer", effective_config_digest="f" * 64, best_checkpoint_sha256="1" * 64)
    peer.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR]["run_name"] = "steady-peer"
    frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR].update(
        model_kind="fno", elapsed_seconds=3600.0, training_units={"train_cases": 10}
    )
    return {"A": frame, "B": peer}


def _scope(tmp_path: Path) -> panel._SteadyScope:
    """Return the comparison scope of the two runs over their ID role."""
    return panel._SteadyScope([panel.EvaluationContext(key="id", label="ID", datasets=_two_runs(tmp_path))], comparison=True, refused_roles={})


def test_a_steady_comparisons_break_even_draws_its_costable_run_in_the_colour_of_the_dataset_views(tmp_path: Path) -> None:
    """Of two runs only «A» has its training evidence: its cost takes the colour every dataset view gives «A»; «B» is named apart."""
    scope = _scope(tmp_path)
    registry = panel._build_sections(scope, comparison=True)

    figure, details = {title: factory for title, factory, _stem in registry["runtime"][0]}["Break-even"]()

    cuts = scope.frames(selection.resolve_choices(scope.inventory, [item.key for item in scope.inventory.items], pooled=False))
    styles = analysis.evaluation.plots.series.series_styles(cuts)
    colours = {to_hex(styles[key].color) for key, (_dataset, model) in cuts.labels.items() if model == "A"}
    (drawn,) = [axis for axis in figure.axes if axis.lines]
    assert colours == {to_hex(drawn.lines[1].get_color())}
    assert "B (no " in figure.get_supxlabel()
    assert details.startswith("A: COMSOL solver time")
    assert "comsol_solver_s" not in details  # the clock reads as its name, not as its storage key


def test_a_single_steady_runs_break_even_takes_its_architecture_colour_not_a_dataset_colour(tmp_path: Path) -> None:
    """One FNO run over two materials: its cost line is the FNO hue, as in a comparison, never the pooled role colour of its datasets."""
    context = panel.EvaluationContext(key="id", label="ID", datasets={"A": _two_runs(tmp_path)["A"]})
    scope = panel._SteadyScope([context], comparison=False, refused_roles={})
    registry = panel._build_sections(scope, comparison=False)

    figure, _details = {title: factory for title, factory, _stem in registry["runtime"][0]}["Break-even"]()

    (drawn,) = [axis for axis in figure.axes if axis.lines]
    assert to_hex(drawn.lines[1].get_color()) == to_hex(analysis.presentation.visual_semantics.architecture_color("fno"))


def test_the_pareto_legend_keys_each_material_by_the_marker_its_points_take(tmp_path: Path) -> None:
    """Both runs span two materials, so their points take material markers: the key names each material with that marker beside the runs."""
    scope = _scope(tmp_path)
    cuts = scope.frames(selection.resolve_choices(scope.inventory, [("lentil", "id"), ("chickpea", "id")], pooled=False))

    figure = analysis.evaluation.plots.run_summary.plot_accuracy_physics_pareto(datasets=cuts)

    legends = [*figure.legends, *(legend for axis in figure.axes if (legend := axis.get_legend()) is not None)]
    entries = {
        text.get_text(): handle
        for legend in legends
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        if isinstance(handle, Line2D)
    }
    markers = analysis.presentation.visual_semantics.MATERIAL_MARKERS
    assert entries["Lentil"].get_marker() == markers["lentil"]
    assert entries["Chickpea"].get_marker() == markers["chickpea"]
    assert [entries[run].get_marker() for run in ("A", "B")] == ["o", "o"]  # the run handles stay filled or hollow circles
