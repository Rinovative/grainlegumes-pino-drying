"""Protect the shared geometry, legend placement and titles of line, bar and distribution figures.

Responsibilities:
  - Check that panels of one grid have equal size, a smaller gap than Matplotlib's default and the shared text sizes
  - Check the legend of one panel in a corner of it (outside right on request), clear of the data of the panel and its
    twin axes through a free corner or headroom, never below bars on a log axis, and starting again from the limits
    before the headroom when replaced unless they are shared since, with a truthful scale (no negative range for a
    non-negative axis, also an inverted one, none wider than the autoscale margin of non-negative data, both limits
    together for a parity plot, a shared axis untouched, the widening left to the axes whose data the key covers, y
    widened before x, the data in view filling 60 % of a widened axis over both passes, a marked reference line, also an
    ``axline``, cleared without counting as data, a second widening checked again), else a key rebuilt in more rendered
    columns, whole groups with their headings per column, else a panel grown at most twice, else the key outside right of
    the panel, also of a hand-placed one; and the legend of several panels outside right, top at the top plot frame,
    without wrapped entries
  - Check legend groups, each owning the entries whose handles it lists whatever their text and order, the figure growing
    for a tall legend, and the refusals of a bad groups or axes argument or of a label listed as a group member
  - Check the y label once per row, row headers attached to their own row above its scientific factor, and the figure title
    with its note, wrapped to the width a later legend adds, as wide in either order, and whole in a tight PNG export
  - Check that the legend entries of a figure's axes are collected in order without repeated labels

Design principles:
  - Geometry is read from drawn figures, never from layout fractions

This module does NOT:
  - Check colours, scientific labels or the entities a legend names
"""

from __future__ import annotations

import copy
import gc
import io
import pickle
import weakref
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text

from src.analysis.ui import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend


def _handles(count: int) -> tuple[list[Line2D], list[str]]:
    handles = [Line2D([], [], color=f"C{index}") for index in range(count)]
    return handles, [f"a rather long entry label number {index}" for index in range(count)]


def _inside_key(figure: Figure, axes: np.ndarray, count: int = 3) -> Legend:
    """Write a key of ``count`` short entries into the single panel of ``axes``, draw the figure and return the key."""
    handles = [Line2D([], [], color=f"C{index}") for index in range(count)]
    figure_layout.add_legend(figure, axes, handles, [f"legend entry {index + 1}" for index in range(count)])
    figure.canvas.draw()
    legend = axes[0, 0].get_legend()
    assert legend is not None
    return legend


def _corner(axis: Axes) -> str:
    """Return where the key of ``axis`` stands, as the frame edges it lies along (within twice its border pad) or "centre"."""
    legend = axis.get_legend()
    assert legend is not None
    box, frame, reach = legend.get_window_extent(), axis.get_window_extent(), 0.1 * axis.figure.dpi
    vertical = "upper" if frame.y1 - box.y1 < reach else "lower" if box.y0 - frame.y0 < reach else "centre"
    horizontal = "right" if frame.x1 - box.x1 < reach else "left" if box.x0 - frame.x0 < reach else "centre"
    return f"{vertical} {horizontal}"


def test_legend_entries_collect_the_labelled_artists_of_all_axes_in_order_without_repeated_labels() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    (first,) = axes[0, 0].plot([0, 1], label="a")
    axes[0, 0].plot([0, 1], label="b")
    axes[0, 1].plot([0, 1], label="a")
    axes[0, 1].plot([0, 1], label="c")
    axes[0, 1].plot([0, 1])  # unlabelled artists name no entry
    handles, labels = figure_layout.legend_entries(axes.flat)
    assert labels == ["a", "b", "c"]
    assert handles[0] is first
    plt.close(figure)


def test_panels_have_equal_size_a_smaller_gap_than_the_default_and_the_shared_text_sizes() -> None:
    figure, axes = figure_layout.panel_figure(2, 3, sharex=True, sharey="all")
    default, default_axes = plt.subplots(2, 3, figsize=figure.get_size_inches(), sharex=True, sharey="all", layout="constrained")
    for fig in (figure, default):
        fig.canvas.draw()

    def gap(grid: object) -> float:
        return float(grid[0, 1].get_window_extent().x0 - grid[0, 0].get_window_extent().x1)  # type: ignore[index]

    sizes = {(round(axis.get_position().width, 6), round(axis.get_position().height, 6)) for axis in axes.flat}
    assert len(sizes) == 1
    assert 0.0 < gap(axes) < gap(default_axes)
    assert not axes[0, 0].xaxis.get_tick_params()["labelbottom"]
    assert axes[1, 0].xaxis.get_tick_params()["labelbottom"]
    assert not axes[0, 1].yaxis.get_tick_params()["labelleft"]
    axis = axes[0, 0]
    figure_layout.set_panel_title(axis, "title")
    axis.set_xlabel("x")
    assert axis.title.get_fontsize() == figure_layout.PANEL_TITLE_PT
    assert axis.xaxis.label.get_fontsize() == figure_layout.LABEL_PT
    assert axis.get_yticklabels()[0].get_fontsize() == figure_layout.TICK_PT
    plt.close(figure)
    plt.close(default)


def test_the_legend_of_one_panel_sits_inside_it() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    handles, labels = _handles(2)
    figure_layout.add_legend(figure, axes, handles, labels, title="Models")
    legend = axes[0, 0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Models"
    assert figure.legends == []
    plt.close(figure)


def test_an_outside_legend_of_one_panel_sits_right_of_it_with_its_top_at_the_top_frame() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    width_before = figure.get_figwidth()
    figure_layout.add_legend(figure, axes, *_handles(2), outside=True)
    figure.canvas.draw()
    (legend,) = figure.legends
    box, panel = legend.get_window_extent(), axes[0, 0].get_window_extent()
    assert axes[0, 0].get_legend() is None
    assert box.x0 > panel.x1
    assert box.y1 == pytest.approx(panel.y1, abs=1.0)
    assert figure.get_figwidth() > width_before  # the figure grows, the panel keeps its size
    plt.close(figure)


def test_the_legend_of_several_panels_sits_outside_right_with_its_top_at_the_top_frame_and_no_wrapped_entry() -> None:
    figure, axes = figure_layout.panel_figure(2, 2)
    width_before = figure.get_figwidth()
    handles, labels = _handles(3)
    twin = Line2D([], [], color="black")
    figure_layout.add_legend(figure, axes, [*handles, twin], [*labels, labels[0]])
    figure.canvas.draw()
    (legend,) = figure.legends
    box = legend.get_window_extent()
    top = max(axis.get_window_extent().y1 for axis in axes.flat)
    assert box.y1 == pytest.approx(top, abs=1.0)
    assert box.x0 >= max(axis.get_window_extent().x1 for axis in axes.flat)
    assert box.x1 <= figure.bbox.x1 + 1.0
    assert figure.get_figwidth() > width_before
    assert len(legend.get_texts()) == len(labels)  # the repeated label collapsed
    assert legend.get_lines()[0].get_color() == "C0"  # onto its first handle, not the last
    assert all("\n" not in text.get_text() for text in legend.get_texts())
    plt.close(figure)


def test_grouped_legends_stack_and_unlisted_labels_form_a_final_untitled_group() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    handles, labels = _handles(4)
    figure_layout.add_legend(figure, axes, handles, labels, groups={"First": handles[:2], "Second": handles[2:3]})
    figure.canvas.draw()
    first, second, rest = figure.legends
    assert rest.get_title().get_text() == ""
    assert [text.get_text() for text in rest.get_texts()] == labels[3:]
    for upper, lower in ((first, second), (second, rest)):
        assert lower.get_window_extent().y1 <= upper.get_window_extent().y0
        assert lower.get_window_extent().x0 == pytest.approx(upper.get_window_extent().x0, abs=1.0)
    plt.close(figure)


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("material_first", [False, True], ids=["models-first", "material-first"])
def test_a_grouped_entry_belongs_to_the_group_that_lists_its_handle_whatever_its_text_and_order(columns: int, material_first: bool) -> None:
    """
    Two model handles named «Lentil» like the material's: Model keeps one circle, Material its square, inside and outside.

    Membership follows the handles each group lists, never the text or the order of the pairs, and a repeated label merges within
    its group only.
    """
    figure, axes = figure_layout.panel_figure(1, columns)
    models = [Line2D([], [], marker="o", linestyle="none") for _ in range(2)]
    material = Line2D([], [], marker="s", linestyle="none")
    handles = [material, *models] if material_first else [*models, material]
    figure_layout.add_legend(figure, axes, handles, ["Lentil"] * 3, groups={"Model": models, "Material": [material]})
    legends = [*figure.legends, *(legend for axis in axes.flat if (legend := axis.get_legend()) is not None)]
    shown = [
        (text.get_text(), cast("Line2D", handle).get_marker())
        for legend in legends
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    ]
    assert [entry for entry in shown if entry[0] == "Lentil"] == [("Lentil", "o"), ("Lentil", "s")]
    plt.close(figure)


def test_a_group_that_lists_a_label_instead_of_a_handle_is_refused() -> None:
    """Equal text can name two entries, so a group lists its entries by their handles and a label as member is refused."""
    figure, axes = figure_layout.panel_figure(1, 2)
    handles = [Line2D([], [], marker=marker, linestyle="none") for marker in ("o", "s")]
    with pytest.raises(TypeError, match="by their handles, not by the label 'Lentil'"):
        figure_layout.add_legend(figure, axes, handles, ["Lentil"] * 2, groups={"Material": ["Lentil"]})  # type: ignore[list-item]
    plt.close(figure)


def test_a_legend_taller_than_the_panels_grows_the_figure_and_stays_inside_it() -> None:
    figure, axes = figure_layout.panel_figure(1, 2, panel_in=(3.0, 1.0))
    height = figure.get_figheight()
    figure.canvas.draw()
    panel_in = axes[0, 0].get_window_extent().height / figure.dpi
    handles, labels = _handles(25)
    figure_layout.add_legend(figure, axes, handles, labels)
    figure.canvas.draw()
    assert figure.get_figheight() > height
    assert figure.legends[0].get_window_extent().y0 >= -1.0
    assert axes[0, 0].get_window_extent().height / figure.dpi == pytest.approx(panel_in, rel=0.02)  # the panels kept their size
    for axis in axes.flat:
        box = axis.get_window_extent()
        assert box.y0 >= 0.0
        assert box.y1 <= figure.bbox.y1
        assert box.x1 <= figure.bbox.x1
    plt.close(figure)


def test_the_legend_starts_right_of_every_axis_decoration_such_as_twin_axis_tick_labels() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    twin = axes[0, 1].twinx()
    twin.plot([0.0, 1.0], [0.0, 1.0e6])
    twin.set_ylabel("n(t)")
    figure_layout.add_legend(figure, axes, *_handles(2))
    figure.canvas.draw()
    box = figure.legends[0].get_window_extent()
    tight_boxes = [axis.get_tightbbox() for axis in figure.axes]
    assert all(tight is not None for tight in tight_boxes)
    assert box.x0 >= max(tight.x1 for tight in tight_boxes if tight is not None)
    assert box.y1 == pytest.approx(axes[0, 0].get_window_extent().y1, abs=1.0)
    plt.close(figure)


def test_the_legend_starts_right_of_the_rightmost_panel_even_when_the_top_right_one_is_hidden() -> None:
    figure, axes = figure_layout.panel_figure(2, 2)
    axes[0, 1].set_visible(False)
    figure_layout.add_legend(figure, axes, *_handles(2))
    figure.canvas.draw()
    box = figure.legends[0].get_window_extent()
    assert box.x0 >= axes[1, 1].get_window_extent().x1
    assert box.y1 == pytest.approx(axes[0, 0].get_window_extent().y1, abs=1.0)
    plt.close(figure)


def test_a_titled_panel_figure_pickles() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    figure_layout.set_panel_title(axes[0, 0], "title")
    figure_layout.set_figure_title(figure, axes, "Title", note="note")
    figure_layout.add_legend(figure, axes, *_handles(2))
    assert pickle.loads(pickle.dumps(figure)).axes  # noqa: S301
    plt.close(figure)


def test_a_reloaded_or_deep_copied_untitled_figure_takes_a_title_with_its_own_axes() -> None:
    figure, _axes = figure_layout.panel_figure(1, 2)
    for clone in (pickle.loads(pickle.dumps(figure)), copy.deepcopy(figure)):  # noqa: S301
        own = np.array(clone.axes, dtype=object).reshape(1, 2)
        figure_layout.set_figure_title(clone, own, "Title")
        assert [text.get_text() for text in clone.texts + clone.artists if isinstance(text, Text)] == ["Title"]
    plt.close(figure)


def test_a_closed_figure_with_title_and_legend_is_garbage_collected() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    figure_layout.set_figure_title(figure, axes, "Title")
    figure_layout.add_legend(figure, axes, *_handles(2))
    reference = weakref.ref(figure)
    plt.close(figure)
    del figure, axes
    gc.collect()
    assert reference() is None


def test_a_legend_without_entries_is_refused_for_one_and_for_several_panels() -> None:
    for columns in (1, 2):
        figure, axes = figure_layout.panel_figure(1, columns)
        with pytest.raises(ValueError, match="at least one entry"):
            figure_layout.add_legend(figure, axes, [], [])
        plt.close(figure)


def test_a_legend_refuses_no_visible_panel_and_the_single_panel_keeps_group_headings_only_for_several_groups() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    handles, labels = _handles(3)
    for axis in axes.flat:
        axis.set_visible(False)
    with pytest.raises(ValueError, match="visible panel"):
        figure_layout.add_legend(figure, axes, handles, labels)
    axes[0, 0].set_visible(True)
    figure_layout.add_legend(figure, axes, handles, labels, groups={"Bands": handles[:1], "Models": handles[1:]})
    legend = axes[0, 0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Bands", labels[0], " ", "Models", labels[1], labels[2]]
    assert legend.get_texts()[0].get_fontweight() == "bold"
    figure_layout.add_legend(figure, axes, handles, labels, groups={"All": handles})  # one named group titles the plain list
    legend = axes[0, 0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == labels
    assert legend.get_title().get_text() == "All"
    plt.close(figure)


def test_unlisted_labels_form_their_own_block_and_only_heading_rows_are_bold_on_a_single_panel() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    handles, _labels = _handles(3)
    figure_layout.add_legend(figure, axes, handles, ["Models", "b", "c"], groups={"Models": handles[1:2]})
    legend = axes[0, 0].get_legend()
    assert legend is not None
    texts = legend.get_texts()
    assert [text.get_text() for text in texts] == ["Models", "b", " ", "Models", "c"]  # a gap row, then the unlisted entries
    assert [text.get_fontweight() for text in texts] == ["bold", "normal", "normal", "normal", "normal"]  # an entry named like a group stays plain
    plt.close(figure)


def test_the_y_label_is_once_per_row_on_the_first_visible_panel_and_axes_must_be_two_dimensional() -> None:
    figure, axes = figure_layout.panel_figure(2, 2)
    axes[1, 0].set_visible(False)
    figure_layout.label_rows_once(axes, "Counts")
    assert [[axis.get_ylabel() for axis in row] for row in axes] == [["Counts", ""], ["", "Counts"]]
    for call in (lambda: figure_layout.label_rows_once(axes[:, 0], "x"), lambda: figure_layout.set_row_headers(figure, axes[0], ["a", "b"])):
        with pytest.raises(ValueError, match="two-dimensional"):
            call()
    with pytest.raises(ValueError, match="zip"):
        figure_layout.set_row_headers(figure, axes, ["only one"])
    plt.close(figure)


def test_a_note_of_tall_mathtext_lines_keeps_a_gap_above_the_panel_titles_before_and_after_a_wrap() -> None:
    """Lines with superscripts and subscripts render taller than plain ones: the strip holds the note as rendered, also once it wraps."""
    note = "\n".join(rf"$n^*_{{{index}}}$ = $C_\mathrm{{train}}$ / ($t_\mathrm{{c}}$ $-$ $t_\mathrm{{s}}$) cases" for index in range(4))
    for wrap in (False, True):
        figure, axes = figure_layout.panel_figure(1, 3)
        for axis in axes.flat:
            figure_layout.set_panel_title(axis, "Model · seed 9")
        figure_layout.set_figure_title(figure, axes, "Break-even", note=note + (" and a plain tail that wraps" * 12 if wrap else ""))
        figure.canvas.draw()
        note_box = next(artist for artist in figure.artists if getattr(artist, "unwrapped", None) is not None).get_window_extent()
        assert note_box.y0 > max(axis.title.get_window_extent().y1 for axis in axes.flat), wrap
        plt.close(figure)


def test_a_long_title_and_note_stay_inside_the_figure_and_a_short_one_stays_centred_over_the_panel() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    figure_layout.set_figure_title(figure, axes, "Short")
    figure.canvas.draw()
    box = figure.artists[0].get_window_extent()
    panel = axes[0, 0].get_window_extent()
    assert (box.x0 + box.x1) / 2.0 == pytest.approx((panel.x0 + panel.x1) / 2.0, abs=1.0)
    plt.close(figure)

    figure, axes = figure_layout.panel_figure(1, 1)
    panel_in = axes[0, 0].get_window_extent().width / figure.dpi
    figure_layout.set_figure_title(figure, axes, "A very long figure title " * 8, note="and an even longer note " * 12)
    figure.savefig(io.BytesIO())  # plain savefig: whatever lies outside the figure is clipped
    for artist in figure.artists:
        box = artist.get_window_extent()
        assert box.x0 >= -1.0
        assert box.x1 <= figure.bbox.x1 + 1.0
    assert figure.get_figwidth() > panel_in
    plt.close(figure)


def test_a_long_note_wraps_to_the_figure_width_and_a_mathtext_line_stays_whole() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    width_in = figure.get_figwidth()
    mathtext = r"$t_\mathrm{c}$ = 34 s"
    figure_layout.set_figure_title(figure, axes, "Title", note="evidence item; " * 40 + f"\n{mathtext}")
    note = figure.artists[1]
    assert isinstance(note, Text)
    assert figure.get_figwidth() == pytest.approx(width_in)
    assert note.get_text().count("\n") > 1
    assert note.get_text().split("\n")[-1] == mathtext
    figure.savefig(io.BytesIO())
    assert note.get_window_extent().width <= figure.bbox.width + 1.0
    plt.close(figure)

    # Glyph widths differ: a character count alone wrapped wide and narrow words too wide; the wrap measures and narrows.
    figure, axes = figure_layout.panel_figure(1, 1)
    width_in = figure.get_figwidth()
    figure_layout.set_figure_title(figure, axes, "Title", note="ii " * 150 + " ".join(["WWWWWWWWWW"] * 20))
    assert figure.get_figwidth() == pytest.approx(width_in)
    plt.close(figure)

    # A mathtext line wider than the figure widens it; the other lines fill that width instead of shrinking to single words.
    figure, axes = figure_layout.panel_figure(1, 1)
    width_in = figure.get_figwidth()
    mathtext = r"$x_\mathrm{" + "a" * 200 + "}$"
    figure_layout.set_figure_title(figure, axes, "Title", note="evidence item; " * 40 + f"\n{mathtext}")
    note = figure.artists[1]
    assert isinstance(note, Text)
    *plain, last = note.get_text().split("\n")
    assert last == mathtext
    assert figure.get_figwidth() > width_in
    assert all(len(line.split()) > 2 for line in plain)
    plt.close(figure)


def test_panel_engine_is_the_engine_of_panel_figure() -> None:
    figure, _axes = figure_layout.panel_figure(1, 1)
    plt.close(figure)
    engine = figure.get_layout_engine()
    assert engine is not None
    assert engine.get() == figure_layout.panel_engine().get()


def test_a_row_header_stays_above_a_panel_title_of_several_lines_and_closer_to_its_own_row_than_to_the_row_above() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    figure_layout.set_panel_title(axes[0, 0], "first line+\nsecond line+\nthird line")
    figure_layout.set_row_headers(figure, axes, ["T [°C]"])
    figure.canvas.draw()
    header = next(text for text in axes[0, 0].texts if text.get_text() == "T [°C]")
    assert header.get_window_extent().y0 >= axes[0, 0].title.get_window_extent().y1
    plt.close(figure)

    figure, axes = figure_layout.panel_figure(2, 1)
    axes[0, 0].set_xlabel("Value [-]")
    figure_layout.set_row_headers(figure, axes, ["first", "second"])
    figure.canvas.draw()
    header = next(text for text in axes[1, 0].texts if text.get_text() == "second").get_window_extent()
    above = axes[0, 0].xaxis.label.get_window_extent().y0 - header.y1
    below = header.y0 - axes[1, 0].get_window_extent().y1
    assert above > 2.0 * below > 0.0
    plt.close(figure)


def test_a_row_header_stands_above_the_scientific_factor_of_its_rows_y_axis() -> None:
    figure, axes = figure_layout.panel_figure(2, 1)
    for axis in axes.flat:
        axis.plot([0.0, 1.0], [0.0, 1.0e8])
        axis.set_xlabel("Value")
    figure_layout.set_row_headers(figure, axes, ["first", "second"])
    figure.canvas.draw()
    for axis, name in zip(axes[:, 0], ("first", "second"), strict=True):
        offset = axis.yaxis.get_offset_text()
        assert offset.get_text()
        header = next(text for text in axis.texts if text.get_text() == name)
        assert header.get_window_extent().y0 >= offset.get_window_extent().y1
    plt.close(figure)


def test_a_long_title_and_note_set_before_an_outside_legend_stay_whole_in_a_tight_png_export() -> None:
    """The texts keep the place the tight bounding box measured when the cropped figure is drawn, so no ink reaches an image edge."""
    figure, axes = figure_layout.panel_figure(2, 1, panel_in=(3.0, 1.5))
    figure_layout.set_figure_title(figure, axes, "A figure title much wider than its single column of panels " * 2, note="a long note line " * 12)
    figure_layout.add_legend(figure, axes, *_handles(3))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image = plt.imread(buffer)
    assert min(image[:, :2, :3].min(), image[:, -2:, :3].min()) > 0.9  # only background in the outer columns
    plt.close(figure)


@pytest.mark.parametrize("note", ["evidence item; " * 30, "M" * 36])  # a note that wraps, and one word wider than the panels
def test_a_note_set_before_an_outside_legend_wraps_to_the_width_the_legend_adds(note: str) -> None:
    """Title first or legend first: the figure ends as wide, the note with the same lines and the strip above the panels as high."""

    def build(*, title_first: bool) -> tuple[str, float, float]:
        figure, axes = figure_layout.panel_figure(2, 1, panel_in=(3.0, 1.5))
        steps = [
            lambda: figure_layout.set_figure_title(figure, axes, "Title", note=note),
            lambda: figure_layout.add_legend(figure, axes, *_handles(3)),
        ]
        for step in steps if title_first else steps[::-1]:
            step()
        figure.canvas.draw()
        text = figure.artists[1]
        assert isinstance(text, Text)
        strip_in = (figure.bbox.y1 - axes[0, 0].get_window_extent().y1) / figure.dpi
        plt.close(figure)
        return text.get_text(), strip_in, figure.get_figwidth()

    (text, strip_in, width_in), (expected_text, expected_strip_in, expected_width_in) = build(title_first=True), build(title_first=False)
    assert text == expected_text
    assert strip_in == pytest.approx(expected_strip_in, abs=0.01)
    assert width_in == pytest.approx(expected_width_in, abs=0.01)


def test_a_single_panel_key_over_data_gets_headroom_on_its_side_and_a_key_beside_the_data_changes_no_limit() -> None:
    """Bars with error bars fill the panel: the upper limit rises until the key clears them, the lower stays; a clear key moves nothing."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.bar(np.arange(6), np.ones(6), yerr=0.1, capsize=3.0, label="bars")
    axis.plot([], [], label="a second entry")
    bottom = axis.get_ylim()[0]
    figure_layout.add_legend(figure, axes, *figure_layout.legend_entries(axes.flat))
    figure.canvas.draw()
    legend = axis.get_legend()
    assert legend is not None
    assert legend.get_window_extent().y0 > axis.transData.transform((0.0, 1.1))[1]  # above the bars and their error bars
    assert axis.get_ylim()[0] == bottom
    plt.close(figure)

    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.plot([0.0, 1.0], [0.0, 1.0], label="diagonal")
    limits = axis.get_ylim()
    figure_layout.add_legend(figure, axes, *figure_layout.legend_entries(axes.flat))
    assert axis.get_ylim() == limits
    plt.close(figure)


def test_a_single_panel_key_clears_lines_without_a_vertex_in_view_and_takes_no_headroom_below_bars_on_a_log_axis() -> None:
    """
    Lines crossing the panel from outside its view cover every place of the key: it gets headroom above them all.

    Bars on a log axis reach the panel's bottom whatever its limit, so headroom below them cannot clear the key; it goes above.
    """
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    for height in np.linspace(0.05, 0.95, 10):
        axis.plot([-1.0, 2.0], [height, height])
    assert not _crosses(axis, _inside_key(figure, axes))
    assert axis.get_ylim()[0] == 0.0
    plt.close(figure)

    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set_yscale("log")
    axis.bar(np.arange(6), np.ones(6), yerr=0.1, capsize=3.0)
    bottom = axis.get_ylim()[0]
    assert _inside_key(figure, axes).get_window_extent().y0 > axis.transData.transform((0.0, 1.1))[1]  # above the error bars
    assert axis.get_ylim()[0] == bottom
    plt.close(figure)


def test_a_single_panel_key_clears_the_bars_of_a_twin_axis_with_headroom_on_that_axis_only() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    twin = axis.twinx()
    twin.bar(np.arange(6), np.ones(6), yerr=0.1, capsize=3.0)
    limits = axis.get_ylim()
    assert _inside_key(figure, axes).get_window_extent().y0 > twin.transData.transform((0.0, 1.1))[1]
    assert axis.get_ylim() == limits
    plt.close(figure)


def test_a_single_panel_key_ends_in_a_corner_where_matplotlib_would_put_it_between_markers_in_every_corner() -> None:
    """Matplotlib's best place is centre right; the key takes the upper right corner with headroom above the markers on both parity axes."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.scatter([0.12, 0.12, 0.88, 0.88], [0.1, 0.9, 0.1, 0.9])
    legend = _inside_key(figure, axes)
    assert _corner(axis) == "upper right"
    assert legend.get_window_extent().y0 > axis.transData.transform((0.0, 0.9))[1]
    assert axis.get_ylim()[0] == 0.0
    assert axis.get_ylim()[1] > 1.0
    assert axis.get_xlim() == axis.get_ylim()
    plt.close(figure)


def test_a_single_panel_key_over_data_in_its_best_corner_moves_to_a_free_corner_without_changing_a_limit() -> None:
    """Matplotlib's best place ignores line collections such as these vertical lines; the key leaves them for the upper left."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.vlines([0.85, 0.9, 0.95], 0.0, 1.0)
    _inside_key(figure, axes)
    assert _corner(axis) == "upper left"
    assert axis.get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_a_replaced_single_panel_key_starts_from_the_limits_before_the_first_headroom_unless_the_caller_set_others() -> None:
    """A one-entry key after a three-entry one gets the smaller headroom, autoscaling stays on; limits the caller set stay its start."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.bar(np.arange(6), np.ones(6), yerr=0.1, capsize=3.0)
    top = axis.get_ylim()[1]
    _inside_key(figure, axes, 3)
    raised = axis.get_ylim()[1]
    _inside_key(figure, axes, 1)
    assert top < axis.get_ylim()[1] < raised
    assert axis.get_autoscaley_on()
    axis.set_ylim(0.0, 3.0)
    _inside_key(figure, axes, 1)
    assert axis.get_ylim() == (0.0, 3.0)
    plt.close(figure)


def _points_under_key(axis: Axes, points: np.ndarray, legend: Legend) -> int:
    """Return how many of the data ``points`` of ``axis`` lie under its key."""
    return int(legend.get_window_extent().count_contains(axis.transData.transform(points)))


def test_a_single_panel_key_never_widens_a_non_negative_axis_below_zero_and_takes_its_headroom_above() -> None:
    """Lines at the upper part make the lower corners cheaper; a lower limit of zero still keeps them closed."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 2.0), ylim=(0.0, 1.0))
    for height in np.linspace(0.2, 0.98, 8):
        axis.plot([-1.0, 3.0], [height, height])
    legend = _inside_key(figure, axes)
    assert axis.get_ylim()[0] == 0.0
    assert axis.get_ylim()[1] > 1.0
    assert not _crosses(axis, legend)
    plt.close(figure)


def test_a_single_panel_key_never_widens_non_negative_data_past_the_margin_autoscaling_gave_them_below_zero() -> None:
    """Values 0 to 100 autoscale to -5 to 105; the lower corners are cheaper, yet the axis keeps -5 as its floor."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    points = np.array([[0.0, 99.0], [1.0, 100.0], [2.0, 99.0], [3.0, 100.0], [0.0, 25.0], [1.0, 30.0], [2.0, 26.0], [3.0, 28.0], [1.0, 0.0]])
    axis.scatter(points[:, 0], points[:, 1])
    lower = axis.get_ylim()[0]
    handles = [Line2D([], [], color=f"C{index}") for index in range(5)]
    figure_layout.add_legend(figure, axes, handles, [f"legend entry {index + 1}" for index in range(5)])
    figure.canvas.draw()
    legend = axis.get_legend() or figure.legends[0]
    assert lower < 0.0
    assert axis.get_ylim()[0] == lower
    assert _points_under_key(axis, points, legend) == 0
    plt.close(figure)


def test_a_single_panel_key_leaves_an_axis_shared_with_other_panels_alone_and_widens_the_other_axis() -> None:
    """The visible panel shares its y axis with a hidden one: the key clears the bars by moving the x limit."""
    figure, axes = figure_layout.panel_figure(1, 2, sharey="all")
    axes[0, 1].set_visible(False)
    axis = axes[0, 0]
    axis.bar(np.arange(6), np.ones(6))
    limits, left = axis.get_ylim(), axis.get_xlim()
    box = _inside_key(figure, axes[:, :1]).get_window_extent()
    assert axis.get_ylim() == limits
    assert axis.get_xlim()[1] > left[1]
    assert axis.get_xlim()[0] == left[0]
    assert not any(patch.get_window_extent().overlaps(box) for patch in axis.patches)
    plt.close(figure)


def _column_starts(legend: Legend) -> list[float]:
    """Return the distinct left edges in display pixels at which the texts of ``legend`` stand, one per rendered column."""
    return sorted({round(text.get_window_extent().x0, 1) for text in legend.get_texts()})


def _inside_panel(legend: Legend, axis: Axes) -> bool:
    """Return whether the key ``legend`` lies inside the frame of ``axis``."""
    box, frame = legend.get_window_extent(), axis.get_window_extent()
    return frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1


def _crossing_lines(axis: Axes) -> None:
    """Draw ten lines across ``axis`` (x 0 to 2, y 0 to 1) from outside its view at every height: only its y axis can widen to clear them."""
    axis.set(xlim=(0.0, 2.0), ylim=(0.0, 1.0))
    for height in np.linspace(0.05, 0.95, 10):
        axis.plot([-1.0, 3.0], [height, height])


def _crosses(axis: Axes, legend: Legend) -> bool:
    """Return whether a line of ``axis`` crosses its key ``legend``."""
    box = legend.get_window_extent()
    return any(line.get_transform().transform_path(line.get_path()).intersects_bbox(box) for line in axis.get_lines())


def _grouped_key(figure: Figure, axes: np.ndarray, runs: int, materials: int) -> Legend:
    """Write a key of a «Run» and a «Material» group into the single panel of ``axes``, draw the figure and return the key."""
    run_handles = [Line2D([], [], color=f"C{index}", marker="o", linestyle="none") for index in range(runs)]
    material_handles = [Line2D([], [], color="grey", marker="s", linestyle="none") for _ in range(materials)]
    labels = [f"run label {index + 1}" for index in range(runs)] + [f"material label {index + 1}" for index in range(materials)]
    groups = {"Run": run_handles, "Material": material_handles}
    figure_layout.add_legend(figure, axes, [*run_handles, *material_handles], labels, groups=groups)
    figure.canvas.draw()
    legend = axes[0, 0].get_legend()
    assert legend is not None
    return legend


def _outside_right(figure: Figure, axis: Axes) -> bool:
    """Draw ``figure`` and return whether its only key stands outside right of the panel ``axis``, top aligned with its frame."""
    figure.canvas.draw()
    (legend,) = figure.legends
    box, frame = legend.get_window_extent(), axis.get_window_extent()
    return axis.get_legend() is None and box.x0 >= frame.x1 and abs(box.y1 - frame.y1) <= 1.0


def test_a_single_panel_key_over_a_parity_plot_keeps_sixty_percent_of_both_axes_over_both_passes() -> None:
    """
    The first pass brings two points from right of the view under the key; a second pass beyond the cap is undone, not added.

    The one growth tried, the least any column count needs, clears nothing over both passes either, so the key stands outside
    right and both limits are the caller's again.
    """
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    points = np.array([[0.1, 0.87], [0.1, 0.05], [0.85, 0.8], [1.12, 1.0], [0.85, 0.1], [1.12, 0.05]])
    axis.scatter(points[:, 0], points[:, 1])
    handles = [Line2D([], [], color=f"C{index}") for index in range(8)]
    figure_layout.add_legend(figure, axes, handles, [f"legend entry {index + 1}" for index in range(8)])
    assert _outside_right(figure, axis)
    assert axis.get_xlim() == axis.get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_a_single_panel_key_is_checked_again_after_its_second_widening_and_never_left_over_a_point_it_brought_in() -> None:
    """A parity plot: each widening of the one-column key brings a point from right of the view under it, the second one unchecked."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    points = np.array([[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.9, 0.8], [1.05, 1.02], [1.3, 1.3]])
    axis.scatter(points[:, 0], points[:, 1])
    legend = _inside_key(figure, axes)
    assert _points_under_key(axis, points, legend) == 0
    assert axis.get_xlim() == axis.get_ylim()
    plt.close(figure)


def test_a_single_panel_key_clears_a_marked_reference_line_whose_span_never_counts_as_data_for_the_sixty_percent() -> None:
    """A marked diagonal spans the view, the points only 0.2 to 1.8 and 0.1 to 0.9: every widened axis keeps 60 % of the points."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 2.0), ylim=(0.0, 1.0))
    axis.plot([0.0, 2.0], [0.0, 1.0], color="black", gid=figure_layout.REFERENCE_GID)
    points = np.array([[0.2, 0.1], [0.2, 0.9], [1.8, 0.1]])
    axis.scatter(points[:, 0], points[:, 1])
    legend = _inside_key(figure, axes, 5)
    assert 1.6 / np.ptp(axis.get_xlim()) >= 0.6
    assert 0.8 / np.ptp(axis.get_ylim()) >= 0.6
    assert _points_under_key(axis, points, legend) == 0
    assert not _crosses(axis, legend)  # the key clears the reference line all the same
    plt.close(figure)


def test_a_single_panel_key_clears_a_marked_axline_diagonal_drawn_through_its_own_transform() -> None:
    """An ``axline`` runs corner to corner of a parity view; points block the other two corners, so the key widens both axes there."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 11.0), ylim=(0.0, 11.0))
    points = np.array([[1.0, 1.0], [1.0, 10.0], [10.0, 1.0]])
    axis.scatter(points[:, 0], points[:, 1])
    axis.axline((0.0, 0.0), slope=1.0, color="black", gid=figure_layout.REFERENCE_GID)
    legend = _inside_key(figure, axes, 2)
    assert not _crosses(axis, legend)
    assert _points_under_key(axis, points, legend) == 0
    assert axis.get_xlim() == axis.get_ylim()
    plt.close(figure)


def test_a_single_panel_key_widens_the_value_axis_where_it_can_and_leaves_the_time_axis_though_x_would_need_less() -> None:
    """Points in every corner of a wide panel: widening x would clear the key by less, but y may widen within its caps, so x stays."""
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(8.0, 3.0))
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 10.0), ylim=(0.0, 1.0))
    points = np.array([[0.2, 0.05], [0.2, 0.95], [9.8, 0.05], [9.8, 0.9]])
    axis.scatter(points[:, 0], points[:, 1])
    legend = _inside_key(figure, axes)
    assert axis.get_xlim() == (0.0, 10.0)
    assert axis.get_ylim()[0] == 0.0
    assert axis.get_ylim()[1] > 1.0
    assert _points_under_key(axis, points, legend) == 0
    plt.close(figure)


def test_a_single_panel_key_that_needs_its_panel_more_than_twice_as_tall_stands_outside_right_and_the_panel_keeps_its_size() -> None:
    """A short panel with lines at every height: twelve long entries are too tall for headroom and too wide for two columns."""
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(3.0, 1.5))
    axis = axes[0, 0]
    _crossing_lines(axis)
    figure.canvas.draw()
    height = axis.get_window_extent().height
    figure_layout.add_legend(figure, axes, *_handles(12))
    assert _outside_right(figure, axis)
    assert axis.get_window_extent().height == pytest.approx(height, rel=0.01)
    assert axis.get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_a_hand_placed_panel_whose_data_no_key_inside_can_clear_gets_its_key_outside_right_and_keeps_the_figure_size() -> None:
    """An add_axes panel cannot grow: lines at every height, and a key too tall for headroom and too wide for two columns."""
    figure = plt.figure(figsize=(5.0, 3.0))
    axis = figure.add_axes((0.1, 0.15, 0.6, 0.75))
    _crossing_lines(axis)
    figure_layout.add_legend(figure, np.array([[axis]], dtype=object), *_handles(6))
    assert _outside_right(figure, axis)
    assert axis.get_ylim() == (0.0, 1.0)  # no headroom left behind
    assert tuple(figure.get_size_inches()) == (5.0, 3.0)  # without a constrained layout, the figure keeps its size
    plt.close(figure)


def test_a_single_panel_key_beyond_the_cap_is_built_again_in_more_rendered_columns_inside_the_panel() -> None:
    """Lines at every height; one column would leave them under 60 % of the y axis, so the key is rebuilt in columns that fit."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    _crossing_lines(axis)
    size = tuple(figure.get_size_inches())
    legend = _inside_key(figure, axes, 6)
    assert len(_column_starts(legend)) >= 2
    assert _inside_panel(legend, axis)
    assert not _crosses(axis, legend)
    assert axis.get_ylim()[0] == 0.0
    assert 0.9 / axis.get_ylim()[1] >= 0.6
    assert tuple(figure.get_size_inches()) == size  # more columns sufficed, the panel kept its size
    plt.close(figure)


def test_a_grouped_single_panel_key_packs_whole_groups_into_columns_with_their_headings_on_top() -> None:
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(6.0, 4.0))
    axis = axes[0, 0]
    _crossing_lines(axis)
    legend = _grouped_key(figure, axes, 3, 3)
    columns: dict[float, list[Text]] = {}
    for text in legend.get_texts():
        if text.get_text().strip():
            columns.setdefault(round(text.get_window_extent().x0, 1), []).append(text)
    rows = [[text.get_text() for text in sorted(texts, key=lambda text: -text.get_window_extent().y0)] for _start, texts in sorted(columns.items())]
    assert rows == [["Run", *(f"run label {index}" for index in (1, 2, 3))], ["Material", *(f"material label {index}" for index in (1, 2, 3))]]
    assert [text.get_text() for text in legend.get_texts() if text.get_fontweight() == "bold"] == ["Run", "Material"]
    assert _inside_panel(legend, axis)
    assert not _crosses(axis, legend)
    plt.close(figure)


def test_a_grouped_single_panel_key_no_column_count_clears_grows_its_panel_and_keeps_its_groups_whole() -> None:
    """A narrow panel: the groups side by side are wider than it, stacked too tall for headroom, so the panel grows taller."""
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(3.0, 3.0))
    axis = axes[0, 0]
    _crossing_lines(axis)
    width, height = figure.get_size_inches()
    legend = _grouped_key(figure, axes, 3, 2)
    assert figure.get_figheight() > height
    assert figure.get_figwidth() == width
    assert len(_column_starts(legend)) == 1
    assert [text.get_text() for text in legend.get_texts()] == ["Run", "run label 1", "run label 2", "run label 3", " ", "Material"] + [
        f"material label {index}" for index in (1, 2)
    ]
    assert _inside_panel(legend, axis)
    assert not _crosses(axis, legend)
    assert 0.9 / axis.get_ylim()[1] >= 0.6
    plt.close(figure)


def test_a_single_panel_key_over_an_inverted_non_negative_axis_moves_its_numerically_upper_limit_only() -> None:
    """The y axis runs from 1 at the bottom to 0 at the top: the top stays at zero, the key gets its room past 1 at the bottom."""
    figure, axes = figure_layout.panel_figure(1, 1)
    axis = axes[0, 0]
    axis.set(xlim=(0.0, 1.0), ylim=(1.0, 0.0))
    for height in np.linspace(0.02, 0.98, 10):
        axis.plot([-4.0, 4.0], [height, height])
    legend = _inside_key(figure, axes)
    bottom, top = axis.get_ylim()
    assert top == 0.0
    assert bottom > 1.0
    assert 0.96 / (bottom - top) >= 0.6
    assert not _crosses(axis, legend)
    plt.close(figure)


def test_a_single_panel_key_widens_the_twin_whose_bars_it_covers_where_the_panel_itself_may_widen_neither_axis() -> None:
    """The panel shares x with its twin and y with a hidden panel; only the twin's bars lie under the key, so the twin's y widens."""
    figure, axes = figure_layout.panel_figure(1, 2, sharey="all")
    axes[0, 1].set_visible(False)
    axis = axes[0, 0]
    axis.scatter([2.5], [0.5])  # a central point, clear of every corner
    twin = axis.twinx()
    twin.bar(np.arange(6), np.ones(6))
    limits, twin_bottom = (axis.get_xlim(), axis.get_ylim()), twin.get_ylim()[0]
    box = _inside_key(figure, axes[:, :1]).get_window_extent()
    assert (axis.get_xlim(), axis.get_ylim()) == limits
    assert twin.get_ylim()[0] == twin_bottom
    assert twin.get_ylim()[1] > 1.05
    assert not any(patch.get_window_extent().overlaps(box) for patch in twin.patches)
    plt.close(figure)


def test_a_replaced_single_panel_key_leaves_the_limits_of_an_axis_shared_since_and_forgets_their_headroom() -> None:
    """A key widened y; a panel linked to that y axis since keeps the limits it took, and the next key widens x instead."""
    figure, axes = figure_layout.panel_figure(1, 2)
    axis, other = axes[0, 0], axes[0, 1]
    other.set_visible(False)
    axis.bar(np.arange(6), np.ones(6))
    _inside_key(figure, axes[:, :1])
    widened = axis.get_ylim()
    assert widened[1] > 1.05  # the first key took headroom above the bars
    other.sharey(axis)
    _inside_key(figure, axes[:, :1])
    assert axis.get_ylim() == other.get_ylim() == widened
    plt.close(figure)


def test_the_title_is_bold_and_centred_over_the_axes_with_its_note_close_below_and_is_set_once() -> None:
    figure, axes = figure_layout.panel_figure(2, 2)
    figure_layout.set_row_headers(figure, axes, ["Relative L2", "H1"])
    height = figure.get_figheight()
    figure_layout.set_figure_title(figure, axes, "Title", note="context")
    figure_layout.add_legend(figure, axes, *_handles(3))  # widens the figure after the title; the title stays centred over the axes
    figure.add_axes((0.96, 0.1, 0.03, 0.5))  # a later axis (colour bar, widget) does not shift the title
    figure.canvas.draw()
    title, note = (artist for artist in figure.artists)
    assert isinstance(title, Text)
    assert figure.get_figheight() > height
    assert title.get_fontweight() == "bold"
    left, right = min(a.get_window_extent().x0 for a in axes.flat), max(a.get_window_extent().x1 for a in axes.flat)
    for text in (title, note):
        box = text.get_window_extent()
        assert (box.x0 + box.x1) / 2.0 == pytest.approx((left + right) / 2.0, abs=1.0)
    gap_pt = (title.get_window_extent().y0 - note.get_window_extent().y1) * 72.0 / figure.dpi
    assert 0.0 <= gap_pt < figure_layout.TITLE_PT  # closer than one title line
    assert note.get_window_extent().y0 >= max(a.get_window_extent().y1 for a in axes.flat)
    assert figure.bbox.y1 - title.get_window_extent().y1 >= 0.1 * figure.dpi  # no title ink at the export boundary
    with pytest.raises(ValueError, match="already set"):
        figure_layout.set_figure_title(figure, axes, "Again")
    plt.close(figure)


def test_log_scale_labels_spaced_minor_ticks_inside_one_decade_and_none_over_many() -> None:
    figure, axis = plt.subplots()
    figure_layout.set_log_scale(axis, "x")

    def labelled(low: float, high: float) -> set[float]:
        axis.set_xlim(low, high)
        figure.savefig(io.BytesIO())
        formatter = axis.xaxis.get_minor_formatter()
        return {float(f"{tick:.6g}") for tick in axis.xaxis.get_minorticklocs() if low <= tick <= high and formatter(tick)}

    # 5 comes first, then 2; 3 would sit too close to 2 and 5. The decade formatter on the axis keeps a tick subset that
    # left out 5; the minor labels do not use it.
    assert labelled(1.0, 9.0) == {2.0, 5.0}
    assert labelled(1e-3, 4e-3) == {2e-3, 4e-3}
    # Inside one decade without a decade tick, several spaced ticks are labelled instead of one or none.
    assert labelled(0.03, 0.095) == {0.03, 0.05, 0.08}
    assert labelled(3e-6, 9e-6) == {3e-6, 5e-6, 7e-6}
    # Few ticks in view (8, 9, 10): the gap halves once, so 8 joins the decade instead of a lone label.
    assert labelled(7.7, 16.6) == {8.0}
    # No integer mantissa in view: Matplotlib places linear ticks, which are labelled with their own mantissa, not rounded.
    assert len(labelled(1.1, 1.2)) >= 2
    formatter = axis.xaxis.get_minor_formatter()
    assert all(f"{tick:.3g}" in formatter(tick) for tick in axis.xaxis.get_minorticklocs() if 1.1 <= tick <= 1.2 and formatter(tick))
    # Still narrower, three digits would print every label as 1.1; the labels take the digits that keep them distinct.
    axis.set_xlim(1.1001, 1.1002)
    figure.savefig(io.BytesIO())
    formatter = axis.xaxis.get_minor_formatter()
    texts = [text for tick in axis.xaxis.get_minorticklocs() if 1.1001 <= tick <= 1.1002 and (text := formatter(tick))]
    assert len(texts) >= 2
    assert len(set(texts)) == len(texts)
    # Over several decades the decades alone are labelled.
    assert labelled(1e-6, 1e-2) == set()
    plt.close(figure)
