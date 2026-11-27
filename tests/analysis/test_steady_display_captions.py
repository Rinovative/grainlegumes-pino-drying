"""Protect the identity, availability and scale text of the steady evaluation views.

Responsibilities:
  - Check that case tables and case-map captions name a case by its display case number, its material and its role
  - Check that every affected panel carries its support lines under it, that the parity fits are the RMSE and R² of the case
    means, and that three significant digits come from one rounding
  - Check the permeability overlay headers, that its contour note names only the drawn red lines, the log10 permeability ticks
    and the single-capacity axis
  - Check that table row headings and comparison map titles name what the rows and columns are, that a zero pressure median is
    drawn, that the pressure summary keeps its column keys and names a component without a finite median, that the profile key
    comes from the shared single-panel legend on the maps' y range, and that the spectral keys match their curves
  - Check that the profile key stands in a free corner or clears the data on a widened value axis that the data fill to 60 %,
    takes columns where one would pass the panel, and never moves a map's extent, and that a dotted profile is thinner with half
    the markers between the others
  - Check that error and residual maps share the scale of the concatenated values of each dataset's runs, named in the note, that
    a dataset without a finite residual is unavailable beside the drawn ones, that the worst-case tables shade only their error
    columns and show no index, that same-unit error bars of a case map and of all datasets and rows of an error-map figure share
    one power of ten, a bar only within two powers of its tick step, and that a magnitude outlier view adds a row of the ranked |u|
  - Check that coincident short spectral traces mark bins of their own, that every band is two edges in its channel's colour with
    its open marker and no fill, and that a sparse heatmap grid binds every run and dataset to its panel with parameter labels on
    every row

Design principles:
  - Small synthetic steady frames from the shared support fixture, compared by display contract rather than pixels

This module does NOT:
  - Recompute metrics, rankings or colour scales; the steady evaluation tests own those values
"""

from __future__ import annotations

import copy
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from support.steady_evaluation import steady_frame

from src import analysis, common
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.figure import Figure

_PLOTS = analysis.evaluation.plots
_GENERATION_IDS = ("case_0577", "case_0578")  # compact generation case IDs that differ from the display case numbers 1 and 2
_SUPPORT_LINE = re.compile(r".+: \d+ (of \d+ cases, \d+ unavailable|cases? below the reference floor, left out)")
_PRESSURE_KEYS = ("pressure inlet MSE / Δp²", "squared outlet mean / Δp²", "pressure boundary MSE / Δp²", "pressure drop absolute error")


def _peer(frame: pd.DataFrame, *, trainable: int = 16) -> pd.DataFrame:
    """Return the fixture's cases as another run with ``trainable`` parameters."""
    peer = frame.copy(deep=True)
    peer.attrs = copy.deepcopy(frame.attrs)
    provenance = peer.attrs["artifact_provenance"]
    provenance["run"].update({"name": "steady-peer", "effective_config_digest": "f" * 64, "best_checkpoint_sha256": "1" * 64})
    provenance["model"].update({"parameter_counts": {"total": trainable, "trainable": trainable}, "architecture": {"modes": 8}})
    return peer


def _refit(frame: pd.DataFrame, folder: Path, factor: float, *, residual: float = 1.0) -> pd.DataFrame:
    """Return the fixture's cases as another run predicting ``factor`` times every reference field and ``residual`` times its momentum residual."""
    peer = _peer(frame)
    folder.mkdir()
    peer.attrs["artifact_root"] = str(folder)
    table = peer.attrs["artifact_provenance"]["outputs"]["parquet"]["path"]
    shutil.copyfile(Path(frame.attrs["artifact_root"]) / table, folder / table)
    for position, source in enumerate(frame["npz_path"]):
        with np.load(source) as payload:
            arrays = dict(payload)
        arrays["pred"] = factor * arrays["gt"]
        arrays["err"] = arrays["pred"] - arrays["gt"]
        arrays["Rx"], arrays["Ry"] = residual * arrays["Rx"], residual * arrays["Ry"]
        path = folder / Path(source).name
        np.savez_compressed(path, **arrays)
        peer.loc[peer.index[position], "npz_path"] = str(path)
    peer.attrs["artifact_provenance"]["outputs"]["npz"] = [
        {"path": Path(path).name, "sha256": common.serialization.cached_file_sha256(Path(path))} for path in peer["npz_path"]
    ]
    return peer


def _pressure_fit(frame: pd.DataFrame) -> tuple[float, float]:
    """Return the RMSE and R² of the pressure case means of ``frame``, computed here from its stored reference and prediction."""
    means = []
    for path in frame["npz_path"]:
        with np.load(path) as payload:
            means.append((float(np.mean(payload["gt"][0])), float(np.mean(payload["pred"][0]))))
    reference, prediction = np.asarray(means).T
    residual = prediction - reference
    return float(np.sqrt(np.mean(residual**2))), float(1.0 - np.sum(residual**2) / np.sum((reference - np.mean(reference)) ** 2))


def _models_of(frames: Mapping[str, pd.DataFrame], dataset: str) -> dataset_cuts.DisplayFrames:
    """Label ``frames`` as the models of one dataset."""
    return dataset_cuts.DisplayFrames(frames, {key: (dataset, key) for key in frames})


def _compared(frame: pd.DataFrame, root: Path, cells: Mapping[tuple[str, str], tuple[float, float]]) -> dataset_cuts.DisplayFrames:
    """Return per (dataset, model) cell a run refitted by its (prediction, residual) factors, keyed «model · dataset»; ID is eval, else OOD."""
    frames, labels = {}, {}
    for (dataset, model), (factor, residual) in cells.items():
        run = _refit(frame, root / f"{model}_{dataset}", factor, residual=residual)
        role = "eval" if dataset == "ID" else "material_ood"
        run.attrs["dataset_role"] = role
        run.attrs["artifact_provenance"]["split_role"] = role
        digest = model.lower() * 64
        run.attrs["artifact_provenance"]["run"].update({"name": f"run {model}", "effective_config_digest": digest, "best_checkpoint_sha256": digest})
        frames[f"{model} · {dataset}"], labels[f"{model} · {dataset}"] = run, (dataset, model)
    return dataset_cuts.DisplayFrames(frames, labels)


def _drawn_maps(figure: Figure, count: int) -> list[Any]:
    """Return the mesh of each of the first ``count`` axes of a map figure (its maps, row by row), ``None`` where a map is unavailable."""
    return [next((item for item in axis.collections if isinstance(item, QuadMesh)), None) for axis in figure.axes[:count]]


def _scale_note(figure: Figure) -> str:
    """Return the colour-scale note under the title of a map figure."""
    return next(text.get_text() for text in figure.texts if text.get_text().startswith("Colour scales:"))


def _below(figure: Figure, text: Any, axis: Any) -> bool:
    """Return whether ``text`` lies wholly under the frame of ``axis`` after a draw."""
    renderer = cast("Any", figure.canvas).get_renderer()
    return bool(text.get_window_extent(renderer).y1 <= axis.get_window_extent(renderer).y0)


def _figure_notes(figure: Figure) -> list[str]:
    """Return the figure title and its note, whitespace collapsed (a long note wraps)."""
    return [" ".join(artist.get_text().split()) for artist in figure.artists if isinstance(artist, Text)]


def test_the_outlier_table_writes_case_numbers_whole_and_keeps_the_reference_marker(tmp_path: Path) -> None:
    """
    Case numbers of four digits stay whole (never 1.4e+03), and every row of the reference case carries its marker.

    With ``top_k`` 2 of two cases the reference is already among the ranked rows; its own row still follows them, last.
    """
    frame, _grid = steady_frame(tmp_path)
    frame["case_index"] = [1400, 1310]
    samples = _PLOTS.samples_outliers
    table = samples.plot_outlier_table(datasets={"lentil, ID": frame}, top_k=2)
    pages = [child.value for child in table.children if "<table" in child.value]
    fields = analysis.evaluation.presentation.display_fields(frame)
    assert len(pages) == len(fields)
    reference = analysis.evaluation.presentation.reference_case_position(frame)
    for page, field in zip(pages, fields, strict=True):
        ranked = samples._outlier_positions(frame, field, top_k=2)[:-1]
        assert reference in ranked
        expected = [f"{frame['case_index'].iloc[position]}{' (Ref)' if position == reference else ''}" for position in (*ranked, reference)]
        assert re.findall(r'class="data row\d+ col0" >([^<]*)</td>', page) == expected  # the Case column, top to bottom
        assert not re.search(r"\d\.\d+e\+0\d", page)


def test_every_case_map_names_the_display_case_number_its_material_and_its_role(tmp_path: Path) -> None:
    """
    The overview, the outlier view, the permeability overlay and the model comparison caption a case alike.

    The number is the one the case navigator and the outlier table show (``case_index``), never the generation case ID, and the
    comparison heads the reference column «Reference».
    """
    frame, _grid = steady_frame(tmp_path)
    frame["source_case_id"] = list(_GENERATION_IDS)
    samples = _PLOTS.samples_outliers
    single = {"lentil, ID": frame}
    figures = {
        "overview": samples.plot_sample_prediction_overview(datasets=single, row_position=1),
        "outlier": samples.plot_linked_outlier_cases(datasets=single, channel="p", selection_index=0),
        "kappa": samples.plot_permeability_error_overlay(datasets=single, row_position=1),
        "comparison": samples.plot_pressure_velocity_comparison(datasets=_models_of({"A": frame, "B": _peer(frame)}, "ID"), row_position=1),
    }
    for name, figure in figures.items():
        title = figure.get_suptitle()
        assert title.endswith("Lentil, ID — Case 2") or "Lentil, ID — Case 2 (" in title, name  # the stored «eval» role reads ID
        assert not any(identity in title for identity in _GENERATION_IDS), name
    assert figures["comparison"].axes[0].get_title() == "Reference"
    for figure in figures.values():
        plt.close(figure)


def test_the_overlay_headers_of_the_lower_row_keep_clear_of_the_upper_maps(tmp_path: Path) -> None:
    """The error and its mean line head the top row; every lower-row header stands in the row gap, under the upper maps."""
    frame, _grid = steady_frame(tmp_path)
    figure = _PLOTS.samples_outliers.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
    figure.canvas.draw()
    renderer = cast("Any", figure.canvas).get_renderer()
    upper, lower = figure.axes[:3], figure.axes[3:6]
    for top, bottom in zip(upper, lower, strict=True):
        assert bottom.title.get_window_extent(renderer).y1 <= top.get_window_extent(renderer).y0
    (mean_line,) = (text for text in upper[2].texts if text.get_text().startswith("mean "))
    assert mean_line.get_window_extent(renderer).y0 >= upper[2].get_window_extent(renderer).y1
    assert upper[2].get_title().startswith("Signed error")
    assert lower[2].get_title().startswith("Reference")
    note = next(text.get_text() for text in figure.texts if text.get_text().startswith("Colour scales:"))
    # The fixture's pressure error is 0.5 x: its 95th percentile is its maximum of 1 Pa, where no line is drawn.
    assert note.endswith("; red line: |e| of $p$ at its 75th percentile, 0.800 Pa")
    plt.close(figure)


def _one_cell(values: np.ndarray) -> np.ndarray:
    """Return an exact field but for one cell with an error of 1."""
    field = np.zeros_like(values)
    field.flat[0] = 1.0
    return field


def _three_at_the_top(values: np.ndarray) -> np.ndarray:
    """Return a ramp from 0 to 0.5 whose last three cells hold the maximum 1, so the 95th percentile is that maximum."""
    field = np.linspace(0.0, 0.5, values.size).reshape(values.shape)
    field.flat[-3:] = 1.0
    return field


@pytest.mark.parametrize(
    ("error", "sentence"),
    [
        (np.zeros_like, None),
        (_one_cell, r"red line: \|e\| of \$p\$ at its 75th and 95th percentile, both 0\.00 Pa"),
        (_three_at_the_top, r"red line: \|e\| of \$p\$ at its 75th percentile, [0-9.]+ Pa"),
    ],
    ids=["exact", "shared-level", "top-level-undrawn"],
)
def test_the_contour_note_names_only_the_red_lines_that_are_drawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Any, sentence: str | None
) -> None:
    """
    The note names the error levels whose red line is drawn: none for an exact prediction, a value both percentiles share once.

    An exact prediction has the one level 0, which draws no line; one erroneous cell puts both percentiles at 0, whose line circles
    it; three cells at the maximum put the 95th percentile there, where no line is drawn, so only the 75th is named.
    """
    frame, _grid = steady_frame(tmp_path)
    samples = _PLOTS.samples_outliers
    real_error_values = samples._error_values

    def synthetic(*args: Any, **kwargs: Any) -> Any:
        result = real_error_values(*args, **kwargs)
        return replace(result, map_values=error(np.asarray(result.map_values, dtype=float)))

    monkeypatch.setattr(samples, "_error_values", synthetic)
    figure = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
    note = next(text.get_text() for text in figure.texts if text.get_text().startswith("Colour scales:"))
    if sentence is None:
        assert note.endswith("arrows mark values outside")  # no contour sentence
    else:
        assert re.search(f"; {sentence}$", note), note
    plt.close(figure)


def test_an_unavailable_error_map_draws_no_red_line_and_adds_no_contour_sentence(tmp_path: Path) -> None:
    """The first fixture case lies below the pressure reference floor, so its relative error map is unavailable: no red line, no sentence."""
    frame, _grid = steady_frame(tmp_path)
    samples = _PLOTS.samples_outliers
    figure = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p", error_mode="Relative [%]")
    assert any(text.get_text().startswith("Unavailable: ") for text in figure.axes[2].texts)  # the error panel says why
    note = next(text.get_text() for text in figure.texts if text.get_text().startswith("Colour scales:"))
    assert note.endswith("arrows mark values outside")
    assert not [item for axis in figure.axes for item in axis.collections if hasattr(item, "levels") and not cast("Any", item).filled]
    plt.close(figure)


def test_the_log10_permeability_bars_label_round_values_on_unchanged_bands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The log10 Kxx and Kyy bars are labelled at round values with the decimals of their step; the bands stay log10 of the kappa bands."""
    frame, _grid = steady_frame(tmp_path)
    samples = _PLOTS.samples_outliers
    real_load_case = samples.cases.load_case

    def varying_kappa(*args: Any) -> Any:
        case = real_load_case(*args)
        x, y = case.coordinates
        kxx, kyy = 1.0e-9 * np.exp(np.sin(3.0 * x) * np.cos(4.0 * y)), 2.0e-9 * (1.5 + np.cos(2.0 * x + y))
        return replace(case, permeability=np.stack([kxx, 3.0e-10 * np.sin(2.0 * x - 3.0 * y), kyy]))

    monkeypatch.setattr(samples.cases, "load_case", varying_kappa)
    kappa = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p", kappa_scale="kappa")
    log = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
    log.canvas.draw()
    for index in (0, 4):  # Kxx and Kyy; their bars are the slots six places later
        levels = cast("Any", log.axes[index].collections[0].norm).boundaries
        np.testing.assert_allclose(levels, np.log10(cast("Any", kappa.axes[index].collections[0].norm).boundaries))
        ticks = np.asarray(log.axes[6 + index].get_yticks(), dtype=float)
        step = float(np.min(np.diff(ticks)))
        decimals = analysis.presentation.field_maps.step_decimals(step)
        assert ticks.size >= 2
        assert decimals <= 2
        np.testing.assert_allclose(ticks / step, np.round(ticks / step), atol=1e-6)
        assert levels[0] - 1e-9 <= ticks[0]
        assert ticks[-1] <= levels[-1] + 1e-9
        labels = [text.get_text() for text in log.axes[6 + index].get_yticklabels() if text.get_text()]
        assert labels
        assert {len(label.partition(".")[2]) for label in labels} == {decimals}
    plt.close(kappa)
    plt.close(log)


def test_the_mean_error_line_keeps_three_significant_digits_with_their_trailing_zeros() -> None:
    """0.01 is 0.0100 and 0.5 is 0.500, a bar's power of ten stays, and a whole number loses only a bare decimal point."""
    mean_error_text = _PLOTS.layout.mean_error_text
    power = "10\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT FOUR}"
    assert mean_error_text(np.array([0.01, -0.01]), unit="Pa") == "mean |e| = 0.0100 Pa"
    assert mean_error_text(np.array([6.7e-4]), unit="m/s", exponent=-4) == f"mean |e| = 6.70 \N{MULTIPLICATION SIGN} {power} m/s"
    assert mean_error_text(np.array([0.5]), unit="Pa") == "mean |e| = 0.500 Pa"
    assert mean_error_text(np.array([0.0999999]), unit="Pa") == "mean |e| = 0.100 Pa"
    assert mean_error_text(np.array([150.0]), unit="Pa") == "mean |e| = 150 Pa"
    assert mean_error_text(np.array([0.31]), unit=None) == f"mean {error_diagnostics.NORMALIZED_ERROR_SYMBOL} = 0.310"


def test_three_significant_digits_come_from_one_rounding_written_out_positionally() -> None:
    """
    A value is rounded once to three significant digits, whose digits are written out without another float conversion.

    Negative values keep their sign, zero its two decimals, a carry across a decade three digits, and 1.234e24 and the largest finite
    float exact zeros, never a binary tail or an overflow. 9.995 is stored as 9.99499…, so its one rounding gives 9.99.
    """
    significant = _PLOTS.layout.significant
    assert significant(0.0) == "0.00"
    assert significant(-0.0125) == "-0.0125"
    assert significant(-150.0) == "-150"
    assert significant(9.996) == "10.0"
    assert significant(-9.996) == "-10.0"
    assert significant(9.995) == "9.99"
    assert significant(1.234e24) == "123" + "0" * 22
    assert significant(sys.float_info.max) == "18" + "0" * 307


def test_support_counts_and_fit_statistics_stand_under_their_panels(tmp_path: Path) -> None:
    """
    Every affected panel carries its expected support lines under it; the parity fits are the RMSE and R² of each model's case means.

    The first case's pressure reference lies below its floor: each model shows 1 of 2 cases in the relative norms, and its pressure
    row of local errors leaves 1 case out. Model A predicts twice the reference and model B 1.1 times it, so the two fits differ; the
    RMSE is written in its unit with three significant digits, R² with four decimals.
    """
    frame, _grid = steady_frame(tmp_path)
    shown = _models_of({"A": frame, "B": _refit(frame, tmp_path / "refit", 1.1)}, "lentil, ID")
    support = [f"{model}: 1 of 2 cases, 1 unavailable" for model in ("A", "B")]
    below_floor = [f"{model}: 1 case below the reference floor, left out" for model in ("A", "B")]
    behavior = _PLOTS.error_behavior
    for figure, expected in (
        (behavior.plot_global_error_metrics(datasets=shown), {2: support, 5: support}),  # the CDF panel of each norm
        (behavior.plot_predictive_error_distributions(datasets=shown), {0: support, 1: below_floor}),  # casewise row, pressure row
    ):
        figure.canvas.draw()
        captions = {index: [text for text in axis.texts if _SUPPORT_LINE.fullmatch(text.get_text())] for index, axis in enumerate(figure.axes)}
        assert {index: [text.get_text() for text in lines] for index, lines in captions.items() if lines} == expected
        assert all(_below(figure, text, figure.axes[index]) for index, lines in captions.items() for text in lines)
        plt.close(figure)
    parity = behavior.plot_mean_field_bias(datasets=shown)
    parity.canvas.draw()
    first = parity.axes[0]  # the pressure row of the one dataset
    fits = [text for text in first.texts if "RMSE" in text.get_text()]
    expected_fits = {model: _pressure_fit(shown[model]) for model in ("A", "B")}
    assert expected_fits["A"] != pytest.approx(expected_fits["B"])
    for text, (model, (rmse, r2)) in zip(fits, expected_fits.items(), strict=True):
        match = re.fullmatch(rf"{model}: RMSE\(\$p\$\) = ([0-9.]+) Pa, R² = (-?[0-9]+\.[0-9]{{4}})", text.get_text())
        assert match is not None, text.get_text()
        assert float(match[1]) == pytest.approx(rmse, rel=5e-3)
        assert float(match[2]) == pytest.approx(r2, abs=5e-5)
        assert _below(parity, text, first)
    assert len(first.get_xticks()) <= 6
    plt.close(parity)


def test_one_shared_capacity_is_one_centred_tick_with_its_exact_count(tmp_path: Path) -> None:
    """Two runs of one capacity stand at the centre of a padded log axis whose one tick is that count; the key shows markers."""
    frame, _grid = steady_frame(tmp_path)
    figure = _PLOTS.sensitivity_capacity.plot_capacity_accuracy(datasets=_models_of({"A": frame, "B": _peer(frame, trainable=8)}, "ID"))
    (axis,) = figure.axes
    assert list(axis.get_xticks()) == [8]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["8"]
    low, high = axis.get_xlim()
    assert np.isclose(np.log10(8.0) - np.log10(low), np.log10(high) - np.log10(8.0))
    assert axis.get_xlabel() == "Trainable parameter count [-]"
    legend = axis.get_legend()
    assert legend is not None
    assert all(cast("Any", handle).get_marker() == "o" and cast("Any", handle).get_linestyle() == "None" for handle in legend.legend_handles)
    plt.close(figure)


def test_table_row_headings_and_comparison_map_titles_name_what_they_show(tmp_path: Path) -> None:
    """The physics table heads datasets «Dataset» and the models of one dataset «Model»; residual maps of models name their dataset."""
    frame, _grid = steady_frame(tmp_path)
    physical = _PLOTS.physical_consistency
    datasets = dataset_cuts.DisplayFrames({"lentil, ID": frame}, {"lentil, ID": ("lentil, ID", "A")})
    models = _models_of({"A": frame, "B": _peer(frame)}, "lentil, ID")
    for shown, heading, other in ((datasets, "Dataset", "Model"), (models, "Model", "Dataset")):
        html = physical.plot_physical_consistency_summary_table(datasets=shown).children[-1].value
        assert f">{heading}</th>" in html
        assert f">{other}</th>" not in html
    for plot in (physical.plot_brinkman_residual, physical.plot_spatial_residuals):
        figure = plot(datasets=models, max_cases=2)
        assert " · lentil, ID" in figure.get_suptitle(), plot.__name__
        plt.close(figure)


def _key_clear_of_the_profiles(figure: Figure, profile: Any) -> Any:
    """Draw ``figure`` and return the key of its ``profile`` panel after checking that it covers none of the profiles' points."""
    figure.canvas.draw()
    legend = profile.get_legend()
    assert legend is not None
    points = np.concatenate([profile.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()])) for line in profile.lines])
    assert not legend.get_window_extent().count_contains(points[np.isfinite(points).all(axis=1)])
    return legend


@pytest.mark.parametrize(
    ("peaks", "free_corner", "margin"),
    [((0.0, 1.0), False, 0.05), ((0.0,), True, 0.05), ((0.0, 1.0), False, 0.5)],
    ids=["peaks-at-both-ends", "upper-right-free", "wide-autoscale-margins"],
)
def test_the_profile_key_clears_the_data_through_a_free_corner_or_the_value_axis_and_never_moves_the_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peaks: tuple[float, ...], free_corner: bool, margin: float
) -> None:
    """
    The shared single-panel legend places the key, and every map and the profile keep the grid extent recorded before it is placed.

    The long run labels make the key wide. Peaks at the bottom and the top leave no corner free, so the log value axis widens on its
    high side, and the profiles' finite extrema still fill 60 % of it, its autoscale margins not counted; with a peak at the bottom
    only, the upper right corner is free and the axis keeps the profiles' own range.
    """
    frame, _grid = steady_frame(tmp_path)
    placed: list[list[Any]] = []
    extents: list[tuple[float, float, float, float]] = []
    add_legend = analysis.ui.figure_layout.add_legend

    def recording(figure: Any, axes: Any, *args: Any, **kwargs: Any) -> None:
        placed.append(list(np.asarray(axes, dtype=object).flat))
        add_legend(figure, axes, *args, **kwargs)

    def peaked(axis: Any, summary: Any, style: Any, label: str | None = None) -> None:
        extents.append(summary.grid.extent)
        heights = summary.grid.coordinates[1, :, 0]
        share = (heights - heights[0]) / (heights[-1] - heights[0])
        values = 2e-3 + 3e-2 * sum(np.exp(-(((share - peak) / 0.05) ** 2)) for peak in peaks)
        axis.plot(values, heights, color=style.color, linestyle=style.linestyle, label=label)

    monkeypatch.setattr(analysis.ui.figure_layout, "add_legend", recording)
    monkeypatch.setattr(_PLOTS.physical_consistency, "_plot_profile", peaked)
    runs = {"FNO · baseline · seed 1": frame, "FNO · physics-informed · seed 2": _peer(frame)}
    with plt.rc_context({"axes.xmargin": margin}):  # the widened limits come from the data, whatever the autoscale margins
        figure = _PLOTS.physical_consistency.plot_spatial_residuals(datasets=_models_of(runs, "lentil, ID"))
    maps, profile = figure.axes[:2], figure.axes[-1]
    assert placed == [[profile]]
    x0, x1, y0, y1 = extents[0]
    assert profile in maps[0].get_shared_y_axes().get_siblings(maps[0])  # the legend helper's shared-axis guard applies
    assert [(axis.get_xlim(), axis.get_ylim()) for axis in maps] == [((x0, x1), (y0, y1))] * 2
    assert profile.get_ylim() == (y0, y1)
    assert profile.get_autoscalex_on() is free_corner
    values = np.log10(np.concatenate([line.get_xdata() for line in profile.lines]))
    low, high = np.log10(profile.get_xlim())
    assert (values.max() - values.min()) / (high - low) >= 0.6
    _key_clear_of_the_profiles(figure, profile)
    plt.close(figure)


def test_a_key_taller_than_the_profile_panel_takes_more_columns_and_stays_inside_it(tmp_path: Path) -> None:
    """Twelve datasets need more key rows than the panel's height holds: the key takes columns, stands inside the frame and covers no data."""
    frame, _grid = steady_frame(tmp_path)
    names = [f"dataset {index}" for index in range(12)]
    shown = dataset_cuts.DisplayFrames(dict.fromkeys(names, frame), {name: (name, "A") for name in names})
    figure = _PLOTS.physical_consistency.plot_spatial_residuals(datasets=shown, max_cases=2)
    profile = figure.axes[-1]
    legend = _key_clear_of_the_profiles(figure, profile)
    key, panel = legend.get_window_extent(), profile.get_window_extent()
    assert len(legend.get_texts()) == len(names)  # every dataset named, in a key no taller than the panel
    assert (panel.x0 <= key.x0, key.x1 <= panel.x1, panel.y0 <= key.y0, key.y1 <= panel.y1) == (True,) * 4
    assert profile.get_ylim() == figure.axes[0].get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_a_dotted_profile_is_drawn_thinner_with_half_the_markers_set_off_from_a_solid_one() -> None:
    """The real profile drawer: a material-OOD (dotted) line is thinner than an ID line and marks half as many rows, each between two of its."""
    heights = np.linspace(0.0, 1.0, 40)
    grid = SimpleNamespace(coordinates=np.stack(np.meshgrid(np.linspace(0.0, 2.0, 3), heights)))
    summary = cast("Any", SimpleNamespace(grid=grid, flux_balance_mean=np.geomspace(1e-3, 1e-2, heights.size)))
    figure, axis = plt.subplots()
    semantics = analysis.presentation.visual_semantics
    for role in ("id", "material_ood"):
        _PLOTS.physical_consistency._plot_profile(axis, summary, semantics.series_style(architecture="fno", material="lentil", role=role))
    solid, dotted = axis.lines

    def marked(line: Any) -> set[int]:
        every = line.get_markevery()
        start, step = every if isinstance(every, tuple) else (0, every)
        return set(range(start, heights.size, step))

    assert (solid.get_linestyle(), dotted.get_linestyle()) == ("-", ":")
    assert dotted.get_linewidth() < solid.get_linewidth()
    assert len(marked(dotted)) == len(marked(solid)) // 2
    assert not marked(dotted) & marked(solid)
    plt.close(figure)


def test_a_zero_pressure_median_is_a_point_at_zero_on_a_linear_axis(tmp_path: Path) -> None:
    """A log axis cannot place a zero median, so its panel stays linear and the zero is a drawn point."""
    frame, _grid = steady_frame(tmp_path)
    frame["pressure_outlet_mean_square"] = 0.0
    figure = _PLOTS.physical_consistency.plot_pressure_boundary_summary(datasets=_models_of({"A": frame}, "ID"), max_cases=2)
    dimensionless = figure.axes[0]
    assert dimensionless.get_yscale() == "linear"
    (points,) = dimensionless.lines
    assert np.asarray(points.get_ydata(), dtype=float)[1] == 0.0
    low, high = dimensionless.get_ylim()
    assert low < 0.0 < high
    plt.close(figure)


def test_a_pressure_component_without_a_finite_median_is_named_and_the_finite_points_keep_their_scale(tmp_path: Path) -> None:
    """Model B has no finite outlet case: its inlet and boundary points stay on the log axis, and the note names B and that component."""
    frame, _grid = steady_frame(tmp_path)
    partial = _peer(frame)
    partial["pressure_outlet_mean_square"] = np.nan
    figure = _PLOTS.physical_consistency.plot_pressure_boundary_summary(datasets=_models_of({"A": frame, "B": partial}, "ID"), max_cases=2)
    dimensionless = figure.axes[0]
    _first, second = dimensionless.lines
    assert np.isfinite(np.asarray(second.get_ydata(), dtype=float)).tolist() == [True, False, True]
    assert dimensionless.get_yscale() == "log"
    assert any("Not plotted, no finite median: B on ID (Squared outlet mean / Δp²)" in note for note in _figure_notes(figure))
    plt.close(figure)


def test_the_pressure_summary_keeps_its_column_keys_and_capitalizes_only_its_tick_labels(tmp_path: Path) -> None:
    """A caller selects the summary's columns by their keys; only the drawn tick labels start with a capital letter."""
    frame, _grid = steady_frame(tmp_path)
    shown = _models_of({"A": frame}, "ID")
    physical = _PLOTS.physical_consistency
    assert tuple(physical.build_pressure_boundary_summary(shown).columns) == _PRESSURE_KEYS
    figure = physical.plot_pressure_boundary_summary(datasets=shown)
    assert [label.get_text() for axis in figure.axes for label in axis.get_xticklabels()] == [key[0].upper() + key[1:] for key in _PRESSURE_KEYS]
    plt.close(figure)


def test_the_spectral_keys_match_the_drawn_curves(tmp_path: Path) -> None:
    """Every curve kind has its own line style, the error included, every channel swatch has the width of its lines, and the note names the bands."""
    frame, _grid = steady_frame(tmp_path)
    figure = _PLOTS.spectral_fidelity.plot_spectral_demand_prediction_error(datasets=_models_of({"A": frame}, "ID"), max_cases=2)
    legends = {legend.get_title().get_text(): legend for legend in figure.legends}
    curves = dict(zip((text.get_text() for text in legends["Curves"].get_texts()), legends["Curves"].legend_handles, strict=True))
    assert len({cast("Any", handle).get_linestyle() for handle in curves.values()}) == len(curves) == 3
    # The medians; the thin lines with open markers are the edges of their q10-q90 bands.
    spectrum, error = ([line for line in axis.lines if line.get_markerfacecolor() != "none"] for axis in figure.axes[:2])
    assert {line.get_linestyle() for line in error} == {cast("Any", curves["Error"]).get_linestyle()}
    widths = [cast("Any", handle).get_linewidth() for handle in legends["Channels"].legend_handles]
    assert [line.get_linewidth() for line in error] == widths
    assert widths == sorted(widths, reverse=True)
    assert len(set(widths)) == len(widths)
    assert [line.get_linewidth() for line in spectrum[::2]] == widths
    markers = [cast("Any", handle).get_marker() for handle in legends["Channels"].legend_handles]  # coincident medians stay told apart
    assert len(set(markers)) == len(markers)
    assert [line.get_marker() for line in error] == markers
    assert any("q10\N{EN DASH}q90 band" in note for note in _figure_notes(figure))  # the band edges have no key entry, the note names them
    plt.close(figure)


def test_coincident_short_spectral_traces_never_mark_one_bin_twice_and_key_their_bands_by_channel() -> None:
    """
    Four channels with one median over 6 to 11 valid bins each mark bins of their own, and every band is keyed by its channel.

    Every q10-q90 band is its two edges, thinner lines in its curve's style and its channel's colour with the channel's marker open
    at its median's bins, and no fill mixes their colours.
    """
    spectral = _PLOTS.spectral_fidelity
    keys = ("p", "u", "v", "velocity_magnitude")
    marks = spectral._channel_marks(cast("Any", [SimpleNamespace(key=key) for key in keys]))
    colors = [to_rgba(f"C{index}") for index in range(len(keys))]
    for bins in range(6, 12):
        figure, axis = plt.subplots()
        values = np.outer((1.0, 2.0, 3.0), np.ones(bins))  # three cases, one median for every channel
        for key, color in zip(keys, colors, strict=True):
            spectral._plot_band(axis, np.arange(1.0, bins + 1.0), values, color=color, linestyle="--", linewidth=1.0, marks=marks[key])
        medians = [line for line in axis.lines if line.get_markerfacecolor() != "none"]
        marked = []
        for line in medians:
            start, step = cast("tuple[int, int]", line.get_markevery())
            marked.append(set(range(start, bins, step)))
        assert all(marked), bins
        assert sum(map(len, marked)) == len(set().union(*marked)), bins
        assert not axis.collections  # no band fill, whose overlapping colours the key lacks
        for median, color, key in zip(medians, colors, keys, strict=True):
            edges = [line for line in axis.lines if line.get_markerfacecolor() == "none" and to_rgba(line.get_color())[:3] == color[:3]]
            assert len(edges) == 2, key
            for edge in edges:
                assert (edge.get_linestyle(), edge.get_marker(), edge.get_markevery()) == ("--", marks[key][0], median.get_markevery())
                assert edge.get_linewidth() < median.get_linewidth()
        plt.close(figure)


def test_a_sparse_heatmap_grid_puts_every_run_and_dataset_in_place_and_labels_the_parameters_of_every_row(tmp_path: Path) -> None:
    """
    Run B has only the second dataset: its row shows the parameter labels on its one panel, and every header and matrix stands at its run and dataset.

    The parameter's order sets the sign of every correlation: +1 for A on ID and B on OOD, -1 for A on OOD, so a swapped run or
    dataset puts a matrix of the other sign in its place.
    """
    frame, _grid = steady_frame(tmp_path)
    cells = {"A · ID": ("ID", "A", frame, (0.1, 0.2)), "A · OOD": ("OOD", "A", frame, (0.2, 0.1)), "B · OOD": ("OOD", "B", _peer(frame), (0.1, 0.2))}
    frames = {}
    for key, (dataset, _model, source, parameter) in cells.items():
        cut = source.copy(deep=True)
        cut.attrs = copy.deepcopy(source.attrs)
        cut["parameters_bed.structure.coarse_len_rel"] = parameter
        cut.attrs["dataset_role"] = cut.attrs["artifact_provenance"]["split_role"] = "eval" if dataset == "ID" else "material_ood"
        frames[key] = cut
    shown = dataset_cuts.DisplayFrames(frames, {key: (dataset, model) for key, (dataset, model, _source, _parameter) in cells.items()})
    figure = _PLOTS.sensitivity_capacity.plot_metadata_error_heatmap(datasets=shown, max_cases=2)
    figure.canvas.draw()
    panels = {(spec.rowspan.start, spec.colspan.start): axis for axis in figure.axes if (spec := axis.get_subplotspec()) is not None and axis.images}
    assert sorted(panels) == [(0, 0), (0, 1), (1, 1)]
    signs = {place: set(np.sign(np.asarray(axis.images[0].get_array())).flat) for place, axis in panels.items()}
    assert signs == {(0, 0): {1.0}, (0, 1): {-1.0}, (1, 1): {1.0}}
    assert [panels[place].get_title() for place in ((0, 0), (0, 1), (1, 1))] == ["ID", "OOD", ""]
    headers = {place: [text.get_text() for text in axis.texts if text.get_text() in {"A", "B"}] for place, axis in panels.items()}
    assert headers == {(0, 0): ["A"], (0, 1): [], (1, 1): ["B"]}
    parameters = [label.get_text() for label in panels[0, 0].get_yticklabels()]
    assert parameters
    assert [label.get_text() for label in panels[1, 1].get_yticklabels()] == parameters
    plt.close(figure)


def test_error_maps_share_the_scale_of_the_concatenated_values_of_each_dataset_and_name_it(tmp_path: Path) -> None:
    """
    The runs on one dataset share the boundaries of their concatenated finite errors, never one run's own; the note names that rule.

    B's errors are twice A's on each dataset, so A's own scale differs from the shared one; OOD's errors are about 100 times ID's.
    """
    frame, _grid = steady_frame(tmp_path)
    shown = _compared(frame, tmp_path, {("ID", "A"): (1.01, 1.0), ("ID", "B"): (1.02, 1.0), ("OOD", "A"): (2.0, 1.0), ("OOD", "B"): (3.0, 1.0)})
    assert _PLOTS.error_behavior.dataset_scale_groups(shown) == [["A · ID", "B · ID"], ["A · OOD", "B · OOD"]]
    assert _PLOTS.error_behavior.dataset_scale_groups({"x": frame, "y": frame}) == [["x"], ["y"]]  # a bare mapping is keyed by dataset
    figure = _PLOTS.error_behavior.plot_mean_error_maps(datasets=shown, max_cases=2, error_mode="MAE", display_spatial_stride=1)
    meshes = _drawn_maps(figure, 4)  # the pressure row: A and B on ID, then on OOD
    error_boundaries = analysis.presentation.field_maps.error_boundaries
    for runs in (meshes[:2], meshes[2:]):
        values = [np.ma.compressed(mesh.get_array()) for mesh in runs]
        pooled = error_boundaries(np.concatenate(values), signed=False)
        assert not np.array_equal(pooled, error_boundaries(values[0], signed=False))
        for mesh in runs:
            np.testing.assert_array_equal(mesh.norm.boundaries, pooled)
    assert meshes[2].norm.boundaries[-1] > 10.0 * meshes[0].norm.boundaries[-1]
    note = _scale_note(figure)
    assert note.startswith("Colour scales: 0 to the 95th percentile")
    assert "one per field and dataset, shared by its runs" in note
    plt.close(figure)


@pytest.mark.parametrize("builder", ["plot_brinkman_residual", "plot_spatial_residuals"])
def test_a_residual_builder_shares_one_scale_over_the_runs_of_a_dataset_beside_a_single_run_dataset(tmp_path: Path, builder: str) -> None:
    """
    ID's two runs share the scale of their concatenated finite residuals; OOD's one run, with 100 times A's residual, has its own scale.

    B's residual is three times A's, so A's own scale differs from the shared one; the note names the rule and the grouping.
    """
    frame, _grid = steady_frame(tmp_path)
    shown = _compared(frame, tmp_path, {("ID", "A"): (2.0, 1.0), ("ID", "B"): (2.0, 3.0), ("OOD", "A"): (2.0, 100.0)})
    figure = getattr(_PLOTS.physical_consistency, builder)(datasets=shown, max_cases=2)
    meshes = _drawn_maps(figure, 3)
    values = [np.ma.compressed(mesh.get_array()) for mesh in meshes]
    error_boundaries = analysis.presentation.field_maps.error_boundaries
    pooled = error_boundaries(np.concatenate(values[:2]), signed=False)
    assert not np.array_equal(pooled, error_boundaries(values[0], signed=False))
    for mesh in meshes[:2]:
        np.testing.assert_array_equal(mesh.norm.boundaries, pooled)
    np.testing.assert_array_equal(meshes[2].norm.boundaries, error_boundaries(values[2], signed=False))
    note = _scale_note(figure)
    assert note.startswith("Colour scales: 0 to the 95th percentile")
    assert "one per field and dataset, shared by its runs" in note
    plt.close(figure)


@pytest.mark.parametrize("builder", ["plot_brinkman_residual", "plot_spatial_residuals"])
def test_a_dataset_without_a_finite_residual_is_unavailable_beside_the_drawn_datasets(tmp_path: Path, builder: str) -> None:
    """OOD's one run has a non-finite residual in every case: its map says so, and ID's two runs are still drawn."""
    frame, _grid = steady_frame(tmp_path)
    shown = _compared(frame, tmp_path, {("ID", "A"): (2.0, 1.0), ("ID", "B"): (2.0, 3.0), ("OOD", "A"): (2.0, np.nan)})
    figure = getattr(_PLOTS.physical_consistency, builder)(datasets=shown, max_cases=2)
    assert [mesh is not None for mesh in _drawn_maps(figure, 3)] == [True, True, False]
    assert [text.get_text() for text in figure.axes[2].texts] == ["Unavailable: no finite values"]
    plt.close(figure)


def test_a_dataset_without_a_finite_error_leaves_the_error_maps_of_the_other_datasets_drawn(tmp_path: Path) -> None:
    """OOD's one run predicts nothing finite: the error figure still draws ID's two runs instead of refusing the whole figure."""
    frame, _grid = steady_frame(tmp_path)
    shown = _compared(frame, tmp_path, {("ID", "A"): (1.01, 1.0), ("ID", "B"): (1.02, 1.0), ("OOD", "A"): (np.nan, 1.0)})
    figure = _PLOTS.error_behavior.plot_mean_error_maps(datasets=shown, max_cases=2, error_mode="MAE", display_spatial_stride=1)
    assert [mesh is not None for mesh in _drawn_maps(figure, 3)][:2] == [True, True]
    plt.close(figure)


@pytest.mark.parametrize("builder", ["plot_std_error_maps", "plot_mean_error_maps"])
def test_same_unit_error_bars_of_every_dataset_and_row_share_one_power_of_ten(tmp_path: Path, builder: str) -> None:
    """
    Every error bar of one unit takes the one power of ten its finest bar needs, over the datasets and the field rows.

    OOD's errors are ten times ID's, so on their own some m/s bars read plain and others in a power of ten; each dataset keeps its own
    scale, and each mean line is written in the power of its bar.
    """
    frame, _grid = steady_frame(tmp_path)
    shown = _compared(frame, tmp_path, {("ID", "A"): (1.002, 1.0), ("OOD", "A"): (1.02, 1.0)})
    figure = getattr(_PLOTS.error_behavior, builder)(datasets=shown, max_cases=2, display_spatial_stride=1)
    powers: dict[str, set[str]] = {}
    for mesh in _drawn_maps(figure, 8):  # the p, u, v and |u| rows of ID and OOD
        power, unit = cast("re.Match[str]", re.fullmatch(r"\[(?:(10\S+) )?(.+)\]", mesh.colorbar.ax.get_ylabel())).groups()
        powers.setdefault(unit, set()).add(power or "")
        means = [text.get_text() for text in mesh.axes.texts if text.get_text().startswith("mean ")]
        assert len(means) == (1 if builder == "plot_mean_error_maps" else 0), means  # every drawn mean map writes its one mean line
        for line in means:
            assert (f"\N{MULTIPLICATION SIGN} {power} {unit}" in line) if power else ("\N{MULTIPLICATION SIGN}" not in line), (line, power)
    assert all(len(found) == 1 for found in powers.values()), powers
    assert powers["m/s"] != {""}  # the finest velocity bar needs a power, which the plainer ones take
    plt.close(figure)


def test_outlier_tables_shade_only_their_error_columns_and_show_no_index(tmp_path: Path) -> None:
    """The worst-case tables shade the nRMSE and relative-L2 columns, never an input parameter, and hide the pandas row index."""
    frame, _grid = steady_frame(tmp_path)
    table = _PLOTS.samples_outliers.plot_outlier_table(datasets={"lentil, ID": frame}, top_k=2)
    pages = [child.value for child in table.children if "<table" in child.value]
    assert pages
    for page in pages:
        assert "row_heading" not in page
        shaded = {int(column) for column in re.findall(r"_row\d+_col(\d+) \{\n  background-color: rgba", page)}
        assert shaded
        assert shaded <= {1, 2}  # the columns after «Case»: nRMSE and relative L2; the parameters stand from column 3 on


@pytest.mark.parametrize(
    ("sizes", "shared"),
    [({"u": 6.0e-4, "v": 6.0e-5}, True), ({"u": 0.01, "v": 0.005}, True), ({"u": 0.02, "v": 1.0e-12}, False)],
    ids=["within-two-powers", "plain-beside-a-factor", "beyond-two-powers"],
)
def test_error_bars_of_rows_with_one_unit_share_a_power_of_ten_only_within_two_powers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sizes: dict[str, float], shared: bool
) -> None:
    """
    The u and v error bars (m/s) share the finer row's power where their tick steps lie within two powers of it, else each keeps its own.

    Errors near 6e-4 and 6e-5 m/s are both written in 10⁻⁶ m/s, and a plain 0.001 m/s step beside a 0.0005 m/s one reads 10 beside 5
    in 10⁻⁴ m/s, never 0.010 beside 5 in 10⁻⁴; 0.02 m/s beside 1e-12 m/s keeps its plain decimals instead of 200000000000 in
    10⁻¹³ m/s. Each mean line is written in the power of its own bar.
    """
    frame, _grid = steady_frame(tmp_path)
    samples = _PLOTS.samples_outliers
    real_error_values = samples._error_values

    def sized(*args: Any, **kwargs: Any) -> Any:
        result = real_error_values(*args, **kwargs)
        values = np.asarray(result.map_values)
        ramp = np.linspace(0.0, 1.0, values.size).reshape(values.shape)
        return replace(result, map_values={"p": 1.0, **sizes}[kwargs["field"].key] * ramp)

    monkeypatch.setattr(samples, "_error_values", sized)
    figure = samples.plot_sample_prediction_overview(datasets={"lentil, ID": frame}, row_position=0, error_mode="MAE")
    powers = []
    for error_map in figure.axes[5:9:3]:  # the error maps of the u and the v row, the last of three columns
        bar = next(item for item in error_map.collections if isinstance(item, QuadMesh)).colorbar
        assert bar is not None
        (mean,) = (text.get_text() for text in error_map.texts if text.get_text().startswith("mean |e|"))
        power = match[1] if (match := re.search(r"\[(10\S*) m/s\]", bar.ax.get_ylabel())) else None
        assert (f"\N{MULTIPLICATION SIGN} {power} m/s" in mean) if power else ("\N{MULTIPLICATION SIGN}" not in mean), (power, mean)
        powers.append(power)
    assert powers[1] is not None
    assert (powers[0] == powers[1]) is shared
    plt.close(figure)


def test_a_magnitude_outlier_view_adds_a_row_of_the_ranked_speed_after_the_learned_fields(tmp_path: Path) -> None:
    """The |u| ranking channel adds a last row headed |u| mapping the case's reference and predicted speed; a learned one keeps the learned rows."""
    frame, _grid = steady_frame(tmp_path)
    samples = _PLOTS.samples_outliers
    fields = analysis.evaluation.presentation.display_fields(frame)
    learned = [field for field in fields if not field.is_magnitude]
    speed = next(field for field in fields if field.is_magnitude)
    bold = analysis.ui.figure_layout.bold_label
    labels = {}
    for channel in ("p", "U"):
        figure = samples.plot_linked_outlier_cases(datasets={"lentil, ID": frame}, channel=channel, selection_index=0, display_spatial_stride=1)
        labels[channel] = [text.get_text() for axis in figure.axes for text in axis.texts if text.get_gid() == "channel-row-label"]
        if channel == "U":
            position = samples._outlier_positions(frame, speed, top_k=5)[0]
            components = [list(frame.attrs["output_fields"]).index(key) for key in speed.component_fields]
            with np.load(frame["npz_path"].iloc[position]) as payload:
                expected = [np.hypot(*payload[source][components]) for source in ("gt", "pred")]
            for axis, values in zip(figure.axes[3 * len(learned) : 3 * len(learned) + 2], expected, strict=True):
                shown = next(item for item in axis.collections if isinstance(item, QuadMesh)).get_array()
                assert shown is not None
                np.testing.assert_allclose(np.ma.getdata(shown).reshape(values.shape), values)
        plt.close(figure)
    assert labels["p"] == [bold(analysis.presentation.field_labels.field_label(field.key, mathtext=True)) for field in learned]
    assert labels["U"] == [*labels["p"], bold(speed.matplotlib_label)]
