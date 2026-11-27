"""Protect how steady Evaluation views count a bounded case prefix, on which datasets their maps open and how they name a case.

Responsibilities:
  - Check that a title counting fewer cases than its datasets hold names the prefix («first 50 of 128 cases»), for the figure
    title and for each dataset heading that differs
  - Check that every map view of a steady comparison opens on every selected dataset and draws each model's panel on it, like
    the other comparison views, with the frames bound in one session as the workflow binds them
  - Check that the case navigator of a steady case view names each row's generation case in its status line

Design principles:
  - A count comes from the drawn and the eligible cases of each dataset, never from a view control

This module does NOT:
  - Check the transient views, whose opening, times and counts the transient session tests protect
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
from support.steady_evaluation import steady_frame

from src import analysis
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from pathlib import Path

    import ipywidgets as widgets
    import pandas as pd
    import pytest


def _model_copies(frame: pd.DataFrame, labels: tuple[str, ...], *, role: str) -> dict[str, pd.DataFrame]:
    """Return independent copies of ``frame`` as the runs of one split role, each its own run so that one session binds them all."""
    copies = {label: frame.copy(deep=True) for label in labels}
    for label, copied in copies.items():
        copied.attrs = copy.deepcopy(frame.attrs)
        copied.attrs["dataset_role"] = role
        copied.attrs["artifact_provenance"]["run"]["name"] = f"steady-{label}"
        copied.attrs["artifact_provenance"]["split_role"] = role
    return copies


def test_a_title_counting_a_bounded_prefix_names_it_with_the_eligible_cases() -> None:
    """50 of 128 cases read «first 50 of 128 cases»; all of them, or a count without eligible cases, keep the plain count."""
    assert layout.case_title("Error maps", case_count=50, eligible=128) == "Error maps — first 50 of 128 cases"
    assert layout.case_title("Error maps", case_count=128, eligible=128) == "Error maps — 128 cases"
    assert layout.case_title("Error maps", case_count=1) == "Error maps — 1 case"


def test_aggregate_titles_share_one_prefix_count_or_give_each_differing_dataset_its_own() -> None:
    """Datasets alike in drawn and eligible cases share the title's count; otherwise each heading names its own prefix."""
    alike = layout.aggregate_title_context("Error maps", {"ID": 50, "OOD": 50}, eligible={"ID": 128, "OOD": 128})
    assert alike == ("Error maps — first 50 of 128 cases", {"ID": "", "OOD": ""})
    title, headings = layout.aggregate_title_context("Error maps", {"ID": 50, "OOD": 30}, models=("A", "B"), eligible={"ID": 128, "OOD": 30})
    assert title == "Error maps (A vs B)"
    assert headings == {"ID": "ID — first 50 of 128 cases", "OOD": "OOD — 30 cases"}


def test_a_dataset_title_names_the_prefix_of_its_first_models_cases_once(tmp_path: Path) -> None:
    """Models A and B of one dataset drawn at one of their cases: the title names that prefix once for the dataset, not per model."""
    frame, _x_grid = steady_frame(tmp_path)
    frames = {"A": frame, "B": frame.copy(deep=True)}
    datasets = dataset_cuts.DisplayFrames(frames, {model: ("Lentil, ID", model) for model in frames})
    title, headings = layout.dataset_title_context("Error maps", datasets, {"A": 1, "B": 1})
    assert title.startswith(f"Error maps — first 1 of {len(frame)} cases")
    assert set(headings.values()) == {""}


def test_every_map_view_of_a_steady_comparison_opens_on_every_selected_dataset(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The mean, std, residual and operator maps of models A and B open on the ID and the material-OOD dataset and draw both models on each."""
    frame, _x_grid = steady_frame(tmp_path)
    by_role = {"id": _model_copies(frame, ("A", "B"), role="eval"), "material_ood": _model_copies(frame, ("A", "B"), role="material_ood")}
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets=by_role["id"]),
        panel.EvaluationContext(key="material_ood", label="Material OOD", datasets=by_role["material_ood"]),
    ]
    maps = {"Mean error maps", "Std error maps", "Spatial residuals", "Darcy–Brinkman operator residual"}  # noqa: RUF001
    bound = {f"{model}:{role}": copied for role, models in by_role.items() for model, copied in models.items()}
    drawn: dict[str, set[str]] = {}  # each view's panel titles
    with analysis.evaluation.session.EvaluationSession(bound):  # bound like the workflow's frames, so the cuts bind beside them
        registry = panel._build_sections(panel._SteadyScope(contexts, comparison=True, refused_roles={}), comparison=True)
        entries = [(title, factory) for section, _name in registry.values() for title, factory, _stem in section if title in maps]
        assert {title for title, _factory in entries} == maps
        try:
            for title, factory in entries:
                state: dict[str, Any] = {}
                analysis.ui.notebook._invoke_dropdown_entry(factory, export_state=state, export_plot_name=title, export_title=title)
                drawn[title] = {axis.get_title() for axis in state["fig"].axes}
        finally:
            plt.close("all")
    columns = {f"{model} · Lentil, {role}" for model in ("A", "B") for role in ("ID", "Material OOD")}
    assert {title: columns - titles for title, titles in drawn.items()} == {title: set() for title in maps}, drawn
    assert not [record.getMessage() for record in caplog.records if "View failed" in record.getMessage()]


def test_the_case_navigator_names_the_generation_case_of_each_row_and_none_where_the_plot_picks_its_cases(tmp_path: Path) -> None:
    """Stepping the rows of a steady case view shows each row's generation case beside the navigator; a ranked view names none."""
    frame, _x_grid = steady_frame(tmp_path)
    scope = panel._SteadyScope([panel.EvaluationContext(key="id", label="ID", datasets={"M": frame})], comparison=False, refused_roles={})
    export = {"export_state": None, "export_plot_name": "steady_case", "export_title": None}
    arguments: dict[str, Any] = {"plot_func": lambda **_kwargs: None, "controls": {}, "selection_state": None, "export": export, "single_model": True}
    view = panel._indexed_view(scope, **arguments)
    field = cast("widgets.Box", cast("widgets.Box", view.rows[1]).children[0]).children[0]
    for position in range(1, len(frame) + 1):
        cast("widgets.IntText", field).value = position
        assert view.status.text == f"generation case {int(frame['case_index'].iloc[position - 1])}"
    ranked = panel._indexed_view(scope, **arguments, offered=lambda rows: rows)  # positions are ranks, not rows
    assert ranked.status.text == ""
