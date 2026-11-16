"""Protect the shared geometry, legend placement and titles of line, bar and distribution figures.

Responsibilities:
  - Check that panels of one grid have equal size, a smaller gap than Matplotlib's default and the shared text sizes
  - Check the legend of one panel inside it (outside right on request) and of several outside right, top at the top plot frame,
    without wrapped entries
  - Check legend groups, the figure growing for a tall legend, and the refusals of a bad groups or axes argument
  - Check the y label once per row, row headers and the figure title with its note
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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text

from src.analysis.ui import analysis_ui_figure_layout as figure_layout


def _handles(count: int) -> tuple[list[Line2D], list[str]]:
    handles = [Line2D([], [], color=f"C{index}") for index in range(count)]
    return handles, [f"a rather long entry label number {index}" for index in range(count)]


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
    figure_layout.add_legend(figure, axes, handles, labels, groups={"First": labels[:2], "Second": labels[2:3]})
    figure.canvas.draw()
    first, second, rest = figure.legends
    assert rest.get_title().get_text() == ""
    assert [text.get_text() for text in rest.get_texts()] == labels[3:]
    for upper, lower in ((first, second), (second, rest)):
        assert lower.get_window_extent().y1 <= upper.get_window_extent().y0
        assert lower.get_window_extent().x0 == pytest.approx(upper.get_window_extent().x0, abs=1.0)
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
    figure_layout.add_legend(figure, axes, handles, labels, groups={"Bands": labels[:1], "Models": labels[1:]})
    legend = axes[0, 0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Bands", labels[0], " ", "Models", labels[1], labels[2]]
    assert legend.get_texts()[0].get_fontweight() == "bold"
    figure_layout.add_legend(figure, axes, handles, labels, groups={"All": labels})  # one named group titles the plain list
    legend = axes[0, 0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == labels
    assert legend.get_title().get_text() == "All"
    plt.close(figure)


def test_unlisted_labels_form_their_own_block_and_only_heading_rows_are_bold_on_a_single_panel() -> None:
    figure, axes = figure_layout.panel_figure(1, 1)
    handles, _labels = _handles(3)
    figure_layout.add_legend(figure, axes, handles, ["Models", "b", "c"], groups={"Models": ["b"]})
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


def test_a_row_header_stays_above_a_panel_title_of_several_lines() -> None:
    figure, axes = figure_layout.panel_figure(1, 2)
    figure_layout.set_panel_title(axes[0, 0], "first line+\nsecond line+\nthird line")
    figure_layout.set_row_headers(figure, axes, ["T [°C]"])
    figure.canvas.draw()
    header = next(text for text in axes[0, 0].texts if text.get_text() == "T [°C]")
    assert header.get_window_extent().y0 >= axes[0, 0].title.get_window_extent().y1
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
