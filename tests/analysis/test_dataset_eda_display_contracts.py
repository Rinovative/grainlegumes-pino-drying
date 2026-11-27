"""Protect the display contracts of the published-Dataset EDA views.

Responsibilities:
  - Check that the inlet boundary input leaves the 2-D map views and is named in their availability note, also when a
    request names it beside a drawable field, while a channel no selected dataset holds still raises
  - Check the case counts of the summary titles, named per Dataset when the draws differ, and the note of the eligible
    cases a bound leaves out
  - Check the colormap of a one-signed retained velocity and the diverging scale of a signed one, and that a round-off
    cell sets only the scale while the map and its range note keep the stored value
  - Check that figure text names a dataset by its bare label only where no other plotted label equals it, while frames,
    styles, case ids and subsets stay keyed by the dataset key
  - Check that datasets holding one constant show it once, other histograms outlines on common bins, and that coincident
    outlines stay visible: nested in other colours, named in one colour

Design principles:
  - Small synthetic frames and arrays; the catalog's synthetic Source frames are read, never changed

This module does NOT:
  - Check figure geometry, legend placement or the time windows of the trajectory view
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.text import Text

from src import analysis
from src.analysis.catalog import analysis_catalog_transient as catalog_transient
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_case_statistics as statistics
from src.analysis.eda.plots import eda_plot_spectral_analysis as spectra
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.analysis.presentation import analysis_field_maps as field_maps

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.patches import Rectangle


def test_the_inlet_boundary_input_leaves_the_map_views_and_their_note_names_its_profile() -> None:
    """
    ``p_in_bc`` holds values on the inlet row only: the value statistics keep it, the map views omit it and say why.

    A map request that names it beside T draws T with that note; a channel that no selected dataset holds still raises.
    """
    frames = catalog_transient._eda_datasets()
    frame = next(iter(frames.values()))
    assert "p_in_bc" in capabilities.available_fields(frame, view="field_statistics")
    for view in ("spatial_map", "spectral", "state_snapshot"):
        for requested in (None, ("T", "p_in_bc")):
            resolution = capabilities.resolve_fields(frames, view=view, requested=requested)
            assert "p_in_bc" not in resolution.fields
            assert resolution.boundary_profiles == ("p_in_bc",)
            note = capabilities.availability_note(resolution, plotted=capabilities.plotted_labels(frames))
            assert "inlet boundary profile over x, not drawn as a 2-D field." in note
            assert "p_in,bc" not in note  # the profile is named by its typeset symbol
        assert capabilities.resolve_fields(frames, view=view, requested=("T", "p_in_bc")).fields == ("T",)
        with pytest.raises(ValueError, match="unavailable in every selected dataset"):
            capabilities.resolve_fields(frames, view=view, requested=("T", "no_such_channel"))
    assert capabilities.resolve_fields(frames, view="field_statistics").boundary_profiles == ()
    figure = spectra.plot_isotropic_spectral_case(datasets=frames, styles=catalog_transient._eda_styles(), case_number=1, channels=("T", "p_in_bc"))
    try:
        assert [axis.get_ylabel() for axis in figure.axes if axis.get_ylabel().startswith("Mean spectral power")] == ["Mean spectral power [(°C)²]"]
        assert any("inlet boundary profile over x" in text.get_text() for text in figure.findobj(Text))
    finally:
        plt.close(figure)


def _frame(cases: int, eligible: int | None) -> pd.DataFrame:
    """Return an empty frame of ``cases`` loaded cases out of ``eligible`` members."""
    frame = pd.DataFrame(index=pd.RangeIndex(cases))
    if eligible is not None:
        frame.attrs["available_case_count"] = eligible
    return frame


def test_case_counts_name_the_drawn_cases_per_dataset_and_the_eligible_ones_the_bound_leaves_out() -> None:
    """
    A title counts the cases drawn, never the bound as a population, and names each dataset when the draws differ.

    The note names each dataset the bound cuts, both by the label the figure writes for it.
    """
    keys = {"A": "A", "B": "B"}
    assert transient_plots.case_count_text({"A": _frame(5, 5)}, 8, plotted=keys) == ("5 cases", "")
    count, note = transient_plots.case_count_text({"A": _frame(20, 20), "B": _frame(8, 8)}, 8, plotted=keys)
    assert (count, note) == ("8 cases per Dataset", "First cases drawn of the eligible ones — A: 8 of 20.")
    assert transient_plots.case_count_text({"A": _frame(3, 3), "B": _frame(12, None)}, None, plotted=keys) == ("A: 3 cases, B: 12 cases", "")
    named = transient_plots.case_count_text({"A": _frame(20, 20), "B": _frame(8, 8)}, None, plotted={"A": "Lentil, ID", "B": "B"})
    assert named == ("Lentil, ID: 20 cases, B: 8 cases", "")


def test_parameter_grids_count_only_the_cases_that_give_a_finite_visible_value() -> None:
    """A Dataset counts its drawn cases with a finite value of a shown key; a Dataset with none, or only non-finite ones, is left out."""
    datasets = {"A": _frame(3, 20), "B": _frame(2, None)}
    cases = {"A": [{"p": 1.0}, {"p": float("nan")}, {"q": 2.0}], "B": [{"p": float("nan")}, {"p": float("inf")}]}
    drawn = statistics._contributing_frames(datasets, cases, ("p",))
    assert list(drawn) == ["A"]
    assert len(drawn["A"]) == 1
    assert drawn["A"].attrs["available_case_count"] == 20


def test_a_one_signed_retained_velocity_takes_the_velocity_colormap_and_a_signed_one_stays_diverging() -> None:
    """A retained velocity component of one sign is a state on turbo; with both signs it keeps the diverging scale of its direction."""
    one_signed = transient_plots.spatial_field_scale("v", "m/s", "physical", np.array([[0.1, 0.4], [0.2, 0.3]]), velocity_state=True)
    signed = transient_plots.spatial_field_scale("v", "m/s", "physical", np.array([[-0.1, 0.4], [0.2, 0.3]]), velocity_state=True)
    assert one_signed.colormap.name == "turbo"
    assert one_signed.semantic is not field_maps.FieldMapSemantic.LINEAR_SIGNED
    assert signed.semantic is field_maps.FieldMapSemantic.LINEAR_SIGNED
    assert signed.colormap.name != "turbo"


def test_velocity_noise_at_round_off_gives_a_one_signed_map_no_second_sign() -> None:
    """Cells of -1e-13 beside a peak of 0.3 are solver round-off: the map stays one-signed, while a genuine reversal keeps both signs."""
    noisy = np.array([[-1e-13, 0.3], [0.2, 0.1]])
    reversal = np.array([[-0.05, 0.3], [0.2, 0.1]])
    assert (
        transient_plots.spatial_field_scale("v", "m/s", "physical", transient_plots._without_roundoff(noisy), velocity_state=True).colormap.name
        == "turbo"
    )
    kept = transient_plots._without_roundoff(reversal)
    np.testing.assert_array_equal(kept, reversal)
    assert (
        transient_plots.spatial_field_scale("v", "m/s", "physical", kept, velocity_state=True).semantic is field_maps.FieldMapSemantic.LINEAR_SIGNED
    )


def test_a_round_off_velocity_cell_keeps_its_stored_value_in_the_map_while_the_scale_stays_one_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A -1e-13 m/s cell beside 2 m/s only sets the scale's sign to one (turbo); the map and its range note read the stored value."""
    frame = next(iter(catalog_transient._eda_datasets().values()))
    row = dict(frame.iloc[0])
    velocity = np.array(row["static_fields"]["v"], copy=True)
    velocity[0, 0] = -1e-13
    row["static_fields"] = {**row["static_fields"], "v": velocity}
    noisy = pd.DataFrame([row], index=frame.index[:1])
    noisy.attrs = dict(frame.attrs)
    noted: list[np.ndarray] = []
    note = transient_plots._near_uniform_range_note
    monkeypatch.setattr(transient_plots, "_near_uniform_range_note", lambda values, unit: noted.append(values) or note(values, unit))
    figure = transient_plots.plot_spatial_field_comparison(datasets={"A": noisy}, case_ids={"A": "case_0001"}, fields=("v",))
    try:
        meshes = [(artist, artist.get_array()) for axis in figure.axes for artist in axis.collections if isinstance(artist, QuadMesh)]
        ((mesh, drawn),) = [(artist, array) for artist, array in meshes if array is not None and array.shape == velocity.shape]  # not the bar
        assert mesh.get_cmap().name == "turbo"
        assert float(np.min(drawn)) == -1e-13
        assert [float(np.min(values)) for values in noted] == [-1e-13]
    finally:
        plt.close(figure)


def _packages(*bases: str) -> dict[str, pd.DataFrame]:
    """Return copies of the catalog's two Source frames keyed «bare label · a» and «· b» that record their bare labels."""
    packages = {}
    for alias, base, frame in zip("ab", bases, catalog_transient._eda_datasets().values(), strict=True):
        copied = frame.copy()  # with its own attrs, so the cached catalog frame records no label
        copied.attrs[capabilities.DISPLAY_LABEL_ATTRIBUTE] = base
        packages[f"{base} · {alias}"] = copied
    return packages


def test_figure_text_drops_a_package_alias_only_where_no_plotted_label_meets_it_and_lookups_keep_the_keys() -> None:
    """
    Only figure text takes the bare label; the frame, its style, its case id and the subset stay keyed by the dataset key.

    A style at the bare label never restyles the package, one of two colliding packages reads bare while both keep their
    alias, and an unknown bare key or a case-id map keyed by it is refused.
    """
    packages = _packages("Lentil, ID", "Lentil, ID")
    first, second = packages
    styles = transient_plots.batch_styles(
        {first: "lentil", second: "lentil", "Lentil, ID": "chickpea"}, roles={first: "id", second: "parameter_ood", "Lentil, ID": "material_ood"}
    )

    def drawn(**options: Any) -> tuple[set[str], set[tuple[str, str]]]:
        """Return the legend entries and the (colour, line style) of every line of one isotropic T spectrum."""
        figure = spectra.plot_isotropic_spectral_case(styles=styles, case_number=1, channels=("T",), **options)
        try:
            lines = {(str(line.get_color()), str(line.get_linestyle())) for axis in figure.axes for line in axis.get_lines()}
            return {text.get_text() for text in figure.legends[0].get_texts()}, lines
        finally:
            plt.close(figure)

    own_style = {(styles[first].color, styles[first].linestyle)}
    assert drawn(datasets={first: packages[first]}) == ({"Lentil, ID"}, own_style)
    assert drawn(datasets=packages, dataset_names=(first,)) == ({"Lentil, ID"}, own_style)
    assert drawn(datasets=packages)[0] == {first, second}
    with pytest.raises(ValueError, match="Unknown EDA dataset selection"):
        drawn(datasets={first: packages[first]}, dataset_names=("Lentil, ID",))
    alone, bare_case = {first: packages[first]}, {"Lentil, ID": "case_0001"}
    with pytest.raises(ValueError, match="must match"):
        transient_plots.plot_spatial_field_comparison(datasets=alone, case_ids=bare_case, fields=("T",), physical_time_hours=1.0)
    with pytest.raises(ValueError, match="must match"):
        transient_plots.plot_state_trajectory_comparison(datasets=alone, case_ids=bare_case, styles=styles)


def test_trajectory_and_value_distribution_figures_name_their_datasets_by_the_bare_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two packages whose bare labels differ read bare in the trajectory legend and in the value-distribution legend and title."""
    packages = _packages("Lentil, ID", "Chickpea, ID")
    styles = transient_plots.batch_styles(dict(zip(packages, ("lentil", "chickpea"), strict=True)))
    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", lambda plot, **_options: plot)
    draw = cast("Callable[..., Figure]", statistics.plot_field_value_distributions(datasets=packages, styles=styles))
    figures = (
        transient_plots.plot_state_trajectory_comparison(
            datasets=packages, case_ids=dict.fromkeys(packages, "case_0001"), styles=styles, channels=("T",)
        ),
        draw(1, datasets=packages),
    )
    try:
        for figure in figures:
            texts = {line for text in figure.findobj(Text) for line in text.get_text().split("\n")}
            assert {"Lentil, ID", "Chickpea, ID"} <= texts
            assert not any(key in text for key in packages for text in texts)
    finally:
        for figure in figures:
            plt.close(figure)


def test_coincident_outlines_stay_visible_nested_in_other_colours_and_named_in_one_colour() -> None:
    """
    Constant lines nest like bar outlines, the earlier one wider, so one of another colour shows around a coincident later one.

    In one colour, coincident outlines merge into one line, so the panel names the datasets under its x label; every dataset
    keeps its bars on the common bin edges with its own counts.
    """
    first, other, second, varying = "Lentil, ID · a", "Chickpea, ID", "Lentil, ID · b", "Kidney bean, Material OOD"
    materials = {first: "lentil", other: "chickpea", second: "lentil", varying: "kidney_bean"}
    styles = transient_plots.batch_styles(materials, roles={first: "id", other: "id", second: "parameter_ood", varying: "material_ood"})
    spread = np.array([0.0, 1.0, 1.0, 3.0])
    data = {
        first: {"constant": np.full(3, 0.4), "spread": spread},
        other: {"constant": np.full(3, 0.4), "spread": np.array([0.5, 2.0, 2.5])},
        second: {"constant": np.full(2, 0.4), "spread": spread.copy()},
        varying: {"constant": np.array([0.0, 0.5, 1.0]), "spread": np.array([0.2, 0.9])},
    }
    figure = statistics._hist_grid(data_by_dataset=data, active_datasets=list(data), columns=["constant", "spread"], title="t", styles=styles)
    assert isinstance(figure, Figure)
    try:
        constant, spread_panel = figure.axes[:2]
        widths = [line.get_linewidth() for line in constant.get_lines()]  # the constant lines of the first three datasets
        assert len(widths) == 3
        assert widths == sorted(widths, reverse=True)
        assert len(set(widths)) == 3
        for axis in (constant, spread_panel):
            notes = [text for text in axis.texts if first in text.get_text()]
            assert len(notes) == 1
            assert second in notes[0].get_text()
            assert notes[0].get_color() == styles[second].color
            assert not any(other in text.get_text() or varying in text.get_text() for text in axis.texts)
        count = len(spread_panel.patches) // len(data)  # the bars of each dataset, in the order drawn
        bars = [[cast("Rectangle", bar) for bar in spread_panel.patches[index * count : (index + 1) * count]] for index in range(len(data))]
        assert len({tuple(bar.get_x() for bar in group) for group in bars}) == 1  # the common bin edges
        heights = [[bar.get_height() for bar in group] for group in bars]
        assert heights[0] == heights[2]
        assert [sum(group) for group in heights] == [4, 3, 4, 2]
    finally:
        plt.close(figure)


def test_a_partial_outline_overlap_reads_part_of_the_fuller_dataset_and_equal_counts_coincide() -> None:
    """Counts [1, 1, 0] lie on part of [1, 1, 4], so the note says «part of», never a full coincidence; equal counts coincide."""
    full, sparse, twin = "Lentil, ID · a", "Lentil, ID · b", "Lentil, ID · c"
    names = (full, sparse, twin)
    styles = transient_plots.batch_styles(dict.fromkeys(names, "lentil"), roles=dict.fromkeys(names, "id"))
    filled = np.array([0.5, 1.5, 2.5, 2.5, 2.5, 2.5])
    series = {full: filled, sparse: np.array([0.5, 1.5]), twin: filled.copy()}

    lines = statistics._merged_outlines(series, np.array([0.0, 1.0, 2.0, 3.0]), styles, {name: name for name in names})

    assert [text for text, _colour in lines] == [f"{sparse} coincides with part of {full}", f"{twin} coincides with {full}"]


def test_datasets_holding_one_constant_show_it_once_and_other_histograms_draw_outlines_on_common_bins() -> None:
    """One constant of every dataset is one text with unit and coverage; spread values are unfilled bars on shared edges in role styles."""
    styles = transient_plots.batch_styles({"A": "lentil", "B": "lentil"}, roles={"A": "id", "B": "parameter_ood"})
    keys = {"A": "A", "B": "B"}
    figure, (constant, spread) = plt.subplots(1, 2)
    try:
        statistics._draw_panel_histograms(constant, {"A": np.full(4, 0.4), "B": np.full(3, 0.4)}, styles, "m", keys)
        assert [text.get_text() for text in constant.texts] == ["constant 0.4 m\nin all 2 Datasets"]
        assert not constant.lines
        assert not constant.patches
        statistics._draw_panel_histograms(spread, {"A": np.array([0.0, 1.0, 2.0, 3.0]), "B": np.array([1.0, 2.0, 5.0])}, styles, None, keys)
        bars = spread.patches
        assert bars
        assert not any(bar.get_fill() for bar in bars)
        first, second = bars[: len(bars) // 2], bars[len(bars) // 2 :]
        assert [bar.get_x() for bar in first] == [bar.get_x() for bar in second]
        assert all(bar.get_linestyle() == first[0].get_linestyle() for bar in first)
        assert all(bar.get_linestyle() == second[0].get_linestyle() for bar in second)
        assert first[0].get_linestyle() != second[0].get_linestyle()  # the roles' solid and dashed outlines
        assert first[0].get_linewidth() > second[0].get_linewidth()  # coincident outlines nest: the later one over the wider earlier one
    finally:
        plt.close(figure)
