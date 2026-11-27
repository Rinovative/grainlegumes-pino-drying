"""Protect the model, material and role entries that the series legend of the steady Evaluation figures hands to the figure legend.

Responsibilities:
  - Check that entries of different groups keep their own handles under equal text, so a model named like a material keeps its marker
  - Check that a model whose points carry several material markers gets a neutral circle beside a grey key of the shapes

Design principles:
  - The series legend is checked where it hands its entries to the shared figure legend, whose placement its own tests protect

This module does NOT:
  - Check legend placement or the pooled-material key of the steady views
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from support.steady_evaluation import steady_frame

from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.presentation import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest


def _material_frame(frame: pd.DataFrame, material: str) -> pd.DataFrame:
    """Return an independent copy of ``frame`` whose provenance selects ``material``."""
    copied = frame.copy(deep=True)
    copied.attrs = copy.deepcopy(frame.attrs)
    copied.attrs["artifact_provenance"]["selection"]["material_families"] = [material]
    return copied


def test_a_model_over_several_markers_is_a_neutral_circle_and_a_model_named_like_a_material_keeps_its_own_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model «lentil» draws chickpea squares and lentil circles, model B chickpea squares: «lentil» is a model and a material entry."""
    frame, _x_grid = steady_frame(tmp_path)
    frames = {f"{run} {material}": _material_frame(frame, material) for run, material in (("m", "chickpea"), ("m", "lentil"), ("b", "chickpea"))}
    groups = {"Chickpea, ID": {"lentil": "m chickpea", "B": "b chickpea"}, "Lentil, ID": {"lentil": "m lentil"}}
    marker = visual_semantics.MATERIAL_MARKERS
    styles = {
        "m chickpea": visual_semantics.SeriesStyle("#0072b2", "-", marker["chickpea"]),
        "m lentil": visual_semantics.SeriesStyle("#0072b2", "-", marker["lentil"]),
        "b chickpea": visual_semantics.SeriesStyle("#d55e00", "-", marker["chickpea"]),
    }
    handed: dict[str, Any] = {}

    def recording(_figure: Any, _axes: Any, handles: list[Any], labels: list[str], **kwargs: Any) -> None:
        handed.update(kwargs, pairs=list(zip(labels, handles, strict=True)))

    monkeypatch.setattr(layout.figure_layout, "add_legend", recording)
    figure, axes = plt.subplots(1, 2, squeeze=False)
    layout.add_series_legend(figure, axes, frames, groups, styles, points=True, name_roles=False)
    plt.close(figure)
    labels_of = {id(handle): label for label, handle in handed["pairs"]}  # the groups own their entries by handle
    assert {title: [labels_of[id(handle)] for handle in members] for title, members in handed["groups"].items()} == {
        "Model": ["lentil", "B"],
        "Material": ["Chickpea", "Lentil"],
    }
    assert handed["groups"]["Model"][0] is not handed["groups"]["Material"][1]  # model «lentil» and material «lentil» stay two entries
    grey, square = to_hex("0.35"), marker["chickpea"]
    entries = [(label, handle.get_marker(), to_hex(handle.get_color())) for label, handle in handed["pairs"]]
    assert entries == [("lentil", "o", "#0072b2"), ("B", square, "#d55e00"), ("Chickpea", square, grey), ("Lentil", "o", grey)]
