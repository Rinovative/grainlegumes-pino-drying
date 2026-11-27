"""Protect the fixed series styles of entities and the grouping of steady frames into panels.

Responsibilities:
  - Check colour, line style, hatch and marker per entity rule: run shades of one architecture, single material, pooled role
  - Check that a run keeps its colour across materials, the material markers and the shade limit
  - Check the panel grouping of keys and the styles read from steady provenance, also of a pooled series
  - Check that a pooled series is the unweighted mean of its materials

Design principles:
  - Entities are tiny and synthetic; provenance reading uses the steady fixture only

This module does NOT:
  - Draw figures or check legends
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from support.steady_evaluation import steady_frame

from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from pathlib import Path


def _entity(
    label: str, *, family: str = "fno", physics: bool = False, role: str = "id", materials: tuple[str, ...] = (), run: str = "a"
) -> series.SeriesEntity:
    return series.SeriesEntity(label, family, physics, role, materials, run)


def test_one_run_takes_the_single_material_else_the_pooled_role_colour() -> None:
    styles = series.resolve_styles([_entity("lentil", materials=("lentil",)), _entity("all", role="parameter_ood", materials=("lentil", "chickpea"))])

    assert styles["lentil"] == visual_semantics.series_style(material="lentil", role="id")
    assert styles["all"] == visual_semantics.series_style(role="parameter_ood", pooled=True)


def test_a_bare_string_of_materials_is_refused_and_a_list_becomes_a_tuple() -> None:
    with pytest.raises(TypeError, match="not the string"):
        series.SeriesEntity("x", "fno", False, "id", "lentil", "a")
    assert series.SeriesEntity("x", "fno", False, "id", ["lentil"], "a").materials == ("lentil",)


def test_several_runs_take_the_architecture_hue_and_runs_of_one_architecture_one_shade_each_by_sorted_run_key() -> None:
    entities = [_entity("C", run="c"), _entity("B", family="uno", physics=True, run="b"), _entity("A", run="a")]
    styles = series.resolve_styles(entities)

    assert styles["B"].color == visual_semantics.architecture_color("pi-uno")  # the only run of its architecture keeps the hue
    assert styles["A"].color != styles["C"].color
    assert styles["A"].marker is None
    assert styles["A"].color == series.resolve_styles(sorted(entities, key=lambda entity: entity.label))["A"].color  # not the order given


def test_a_run_keeps_its_colour_across_materials_and_its_materials_differ_by_marker() -> None:
    entities = [
        _entity("a lentil", materials=("lentil",), run="a"),
        _entity("a chickpea", materials=("chickpea",), run="a"),
        _entity("b lentil", materials=("lentil",), run="b"),
        _entity("b chickpea", materials=("chickpea",), run="b"),
    ]
    styles = series.resolve_styles(entities)  # two runs of one architecture on two materials: no error
    assert styles["a lentil"].color == styles["a chickpea"].color
    assert styles["b lentil"].color == styles["b chickpea"].color
    assert styles["a lentil"].color != styles["b lentil"].color
    assert styles["a lentil"].marker == visual_semantics.MATERIAL_MARKERS["lentil"]
    assert styles["a chickpea"].marker == visual_semantics.MATERIAL_MARKERS["chickpea"]


def test_runs_beyond_the_shade_limit_switch_the_figure_to_run_colours_and_beyond_the_palette_are_refused() -> None:
    entities = [_entity(f"run {index:02d}", run=f"{index:02d}") for index in range(visual_semantics.SHADE_LIMIT)]
    shaded = series.resolve_styles(entities)
    assert len({style.color for style in shaded.values()}) == visual_semantics.SHADE_LIMIT
    assert not shaded.run_colours
    entities.append(_entity("run more", run="zz", family="uno", role="parameter_ood"))  # another architecture joins the figure
    entities.append(_entity("run extra", run="yy"))  # one more run of the first architecture
    switched = series.resolve_styles(entities)
    assert switched.run_colours
    assert {style.color for style in switched.values()} == set(visual_semantics.RUN_COLORS[: len(entities)])
    assert switched["run 00"].color == visual_semantics.RUN_COLORS[0]  # sorted run key order
    assert switched["run more"].linestyle == visual_semantics.ROLE_LINESTYLES["parameter_ood"]  # the role keeps its meaning
    many = [_entity(f"run {index:02d}", run=f"{index:02d}") for index in range(len(visual_semantics.RUN_COLORS) + 1)]
    with pytest.raises(ValueError, match="at most 10 runs"):
        series.resolve_styles(many)


def test_panel_groups_give_the_original_keys_in_order_of_appearance_and_refuse_a_repeated_pair() -> None:
    labels = {"m1 · d1": ("d1", "m1"), "m1 · d2": ("d2", "m1"), "m2 · d1": ("d1", "m2")}
    assert series.panel_groups(labels) == {"d1": {"m1": "m1 · d1", "m2": "m2 · d1"}, "d2": {"m1": "m1 · d2"}}
    with pytest.raises(ValueError, match=r"'first' and 'second'"):
        series.panel_groups({"first": ("d", "m"), "second": ("d", "m")})


def test_panel_series_group_the_keys_of_a_pooled_series_under_one_pair_and_its_head_stands_for_it() -> None:
    labels = {"lentil": ("ID, all materials", "m1"), "chickpea": ("ID, all materials", "m1"), "m2": ("ID, all materials", "m2")}
    groups = series.panel_series(labels)

    assert groups == {"ID, all materials": {"m1": ("lentil", "chickpea"), "m2": ("m2",)}}
    assert series.series_heads(groups) == {"ID, all materials": {"m1": "lentil", "m2": "m2"}}
    with pytest.raises(ValueError, match="both model"):
        series.panel_groups(labels)  # the views that cannot pool still refuse two frames of one series


def test_a_pooled_series_is_the_unweighted_mean_of_its_materials_and_unavailable_when_one_is() -> None:
    values = {"a": [1.0, 10.0], "b": [3.0, np.nan], "c": [8.0, 2.0]}

    pooled, per_key = series.material_mean(["a", "b"], values.__getitem__)
    alone, _ = series.material_mean(["c"], values.__getitem__)

    assert pooled[0] == 2.0
    assert np.isnan(pooled[1])  # never the mean of the materials that remain
    np.testing.assert_array_equal(per_key[0], [1.0, 10.0])
    np.testing.assert_array_equal(alone, [8.0, 2.0])  # a series of one key is its own value


def test_the_keys_of_a_pooled_series_take_the_series_style_not_the_style_of_their_material(tmp_path: Path) -> None:
    frame, _grid = steady_frame(tmp_path)
    frames = {}
    for material in ("lentil", "chickpea"):
        frames[material] = frame.copy(deep=True)
        frames[material].attrs = copy.deepcopy(frame.attrs)
        frames[material].attrs["artifact_provenance"]["selection"]["material_families"] = [material]

    separate = series.series_styles(frames)
    pooled = series.series_styles(frames, [("lentil", "chickpea")])

    assert separate["lentil"] == visual_semantics.series_style(material="lentil", role="id")
    assert separate["lentil"].color != separate["chickpea"].color
    assert pooled == dict.fromkeys(frames, visual_semantics.series_style(role="id", pooled=True))


def test_series_styles_read_the_provenance_and_need_the_model_only_of_several_runs(tmp_path: Path) -> None:
    frame, _grid = steady_frame(tmp_path)
    peer = frame.copy(deep=True)
    peer.attrs = copy.deepcopy(frame.attrs)
    peer.attrs["artifact_provenance"]["run"].update(name="zz-peer", effective_config_digest="f" * 64)
    peer.attrs["artifact_provenance"]["model"].update(parameter_counts={"total": 16, "trainable": 16})
    assert series.series_styles({"ID": frame}) == {"ID": visual_semantics.series_style(role="id", pooled=True)}
    frame.attrs["artifact_provenance"]["selection"]["material_families"] = ["lentil"]
    assert series.series_styles({"ID": frame}) == {"ID": visual_semantics.series_style(material="lentil", role="id")}
    styles = series.series_styles({"A": frame, "B": peer})
    assert styles["A"].color != styles["B"].color  # two runs of one architecture
    del peer.attrs["artifact_provenance"]["model"]["parameter_counts"]
    assert series.series_styles({"A": frame}) == {"A": visual_semantics.series_style(material="lentil", role="id")}
