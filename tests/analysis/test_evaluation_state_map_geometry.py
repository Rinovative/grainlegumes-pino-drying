"""Protect the fixed map-grid geometry, units and colour-bar arrows of the transient state maps.

Responsibilities:
  - Check equal map sizes with the domain aspect, colour bars as tall as their maps, edge-only labels and the title
  - Check that the error scale runs from 0 to a rounded top over the largest error, fixed over the scaled times
  - Check that the signed error scale is symmetric to the larger percentile and fixed over all frames, and that tick
    decimals stay constant over all frames
  - Check that a reused animation figure keeps its size, axes positions and title width
  - Check that every decoded GIF delta frame equals its source within the decoding bound, that pixels equal in every
    frame decode alike in every frame, that colours shown in one frame only keep fine steps, and that a rare colour
    decodes like the same colour filling a later frame
  - Check that a frame without finite values keeps every colour bar and the error scale fixed over all frames
  - Check that the GIF admission estimate covers the measured peak of all live frame images
  - Check that the sample title names the compact case, the frame only when comparing, and fits the frame
  - Check that a field fixed over time draws the pooled state bands in every frame while a free field follows its frame,
    that the norms and the requested fixed fields must agree, that a state bar draws the union of its norms' arrows, and
    that the mean-error line writes its mean positionally in its bar's power of ten

Design principles:
  - Geometry is measured on drawn figures of a small synthetic record

This module does NOT:
  - Check the per-time state colour scales and their values in the viewers (see the transient session tests) or the
    notebook controls
"""

from __future__ import annotations

import tracemalloc
import weakref
from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Normalize
from PIL import Image
from support.transient_evaluation import scaling_state, sequence_record

from src import analysis
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from pathlib import Path

transient = analysis.evaluation.plots.transient
layout = analysis.ui.plot_layout
field_maps = analysis.presentation.field_maps
FIELDS = ("T", "phi", "w_gr")
_MAX_DECODING_ERROR_RGB = 8.0  # Euclidean RGB distance of a decoded pixel from its source; a pale-yellow-to-pink flip is about 27


def _record() -> Any:
    """Return a three-time record with spatially varying states and growing prediction errors."""
    record = sequence_record("autonomous_full", "full")
    ramp = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    reference = np.array([300.0, 0.4, 0.5, 0.6]).reshape(1, 4, 1, 1) + np.array([10.0, 0.1, 0.2, 0.2]).reshape(1, 4, 1, 1) * ramp
    reference = np.repeat(reference, 3, axis=0)
    growth = np.arange(3.0).reshape(3, 1, 1, 1)
    record.reference_states = reference.astype(np.float32)
    record.predicted_states = (reference + growth * np.array([1.0, 0.01, 0.02, 0.02]).reshape(1, 4, 1, 1) * ramp).astype(np.float32)
    return record


def _colorbar_extend(axis: Any) -> Any:
    """Return the extend mode of the colour bar drawn on ``axis``, or None when it has none."""
    colorbar = getattr(axis, "_colorbar", None)
    return colorbar and colorbar.extend


def _inches(figure: Any, axis: Any) -> tuple[float, float, float, float]:
    """Return x0, x1, width and height of one drawn axes in inches."""
    box = axis.get_position()
    width, height = figure.get_size_inches()
    return box.x0 * width, box.x1 * width, box.width * width, box.height * height


def _bar_height(figure: Any, slot: Any) -> float:
    """Return the drawn height in inches of the colour bar on ``slot`` with its arrows, which shrink the bar axes itself."""
    return float(slot._colorbar.outline.get_window_extent().height / figure.dpi)


def test_state_maps_draw_equal_maps_with_bars_as_tall_as_their_maps_units_and_one_title() -> None:
    """Every map has the planned size and aspect, every bar its map's height and a unit; labels stand once."""
    record = _record()
    context = "Lentil · case 7 · FNO"
    figure = transient.plot_state_maps(record, state_fields=FIELDS, physical_time=1.0, error_mode="Absolute error", title_context=context)
    figure.canvas.draw()
    geometry = transient.state_map_geometry(record, state_fields=FIELDS)
    assert figure.get_layout_engine() is None
    np.testing.assert_allclose(figure.get_size_inches(), (geometry.figure_width, geometry.figure_height))
    maps, slots = figure.axes[:9], figure.axes[9:]
    assert geometry.map_height / geometry.map_width == pytest.approx(0.04 / 0.08)
    for map_axis, slot in zip(maps, slots, strict=True):
        _x0, map_x1, map_width, map_height = _inches(figure, map_axis)
        slot_x0, _x1, slot_width, _slot_height = _inches(figure, slot)
        assert (map_width, map_height) == pytest.approx((geometry.map_width, geometry.map_height), rel=1e-3)
        assert (slot_width, _bar_height(figure, slot)) == pytest.approx((layout.COLORBAR_WIDTH_IN, map_height), rel=1e-2)
        assert slot_x0 - map_x1 == pytest.approx(layout.COLORBAR_PAD_IN, rel=1e-3)
        assert slot.get_ylabel().endswith("]")
    for row in range(3):
        for column in range(3):
            axis = maps[3 * row + column]
            assert any(label.get_visible() and label.get_text() for label in axis.get_xticklabels()) is (row == 2)
            assert bool(axis.get_ylabel()) is (column == 0)
    assert [axis.get_title() for axis in maps[3:5]] == ["", ""]
    row_labels = [text.get_text() for axis in maps for text in axis.texts if text.get_gid() == "channel-row-label"]
    assert len(row_labels) == len(FIELDS)
    assert not any("[" in label for label in row_labels)
    assert figure.get_suptitle() == "Lentil · case 7 · FNO · t = 1 h"
    plt.close(figure)
    plain = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    assert plain.get_suptitle() == "Transient state comparison at t=1 h"
    plt.close(plain)


def test_error_scale_runs_from_zero_to_a_rounded_top_over_the_largest_error_and_fixed_norms_give_every_frame_the_worst_time() -> None:
    """A view scales its error to its own time; norms over all times give every frame the scale of the worst time, never an arrow."""
    record = sequence_record("autonomous_full", "full")
    record.reference_states[:, 0] = 300.0
    record.predicted_states = record.reference_states.copy()
    record.predicted_states[1, 0] += 1.0  # |e|/s = 0.5 with the two-kelvin train scale
    record.predicted_states[2, 0] += 4.0  # |e|/s = 2
    scale = error_diagnostics.transient_reference_scale(scaling_state(), field="T", unit="K", dataset_identity="a" * 64)
    common: dict[str, Any] = {
        "state_fields": ("T",),
        "error_mode": error_diagnostics.NORMALIZED_ABSOLUTE_ERROR,
        "reference_scales": {"T": scale},
    }

    def scale_of(figure: Any) -> tuple[str, float, float, float]:
        image = figure.axes[2].collections[0]
        boundaries = np.asarray(image.norm.boundaries)
        return image.colorbar.extend, float(boundaries[0]), float(boundaries[1] - boundaries[0]), float(boundaries[-1])

    own = transient.plot_state_maps(record, physical_time=1.0, **common)
    worst = transient.plot_state_maps(record, physical_time=2.0, **common)
    norms = transient.state_map_norms(record, time_indices=range(3), **common)
    frame = transient.plot_state_maps(record, physical_time=1.0, norms=norms, **common)
    # 9 to 13 equal steps of m times a power of ten (m in 1, 2, 2.5, 5), the least span reaching the error.
    assert scale_of(own) == pytest.approx(("neither", 0.0, 0.05, 0.5))
    assert scale_of(worst) == pytest.approx(("neither", 0.0, 0.2, 2.0))
    assert scale_of(frame) == scale_of(worst)
    for figure in (own, worst, frame):
        plt.close(figure)


def test_error_scales_reach_the_95th_percentile_symmetric_when_signed_and_mark_the_errors_beyond() -> None:
    """A uniform signed error gets a symmetric scale; a few large errors lie beyond a top that float rounding never undercuts."""
    signed = transient._error_definition("T", "Signed error", np.full(4, 2.0))
    assert signed.boundaries[0] == pytest.approx(-signed.boundaries[-1])
    assert (signed.boundaries[-1] >= 2.0, signed.extend) == (True, "neither")
    # Symmetric to the larger absolute one of the 5th and 95th percentile of the signed error, so a lopsided error keeps white at 0.
    lopsided = transient._error_definition("T", "Signed error", np.concatenate((np.full(940, 0.1), np.full(60, -3.0))))
    assert lopsided.boundaries[0] == pytest.approx(-lopsided.boundaries[-1])
    assert (lopsided.boundaries[-1] >= 3.0, lopsided.extend) == (True, "neither")
    errors = np.concatenate((np.full(960, 0.1), np.full(40, 5.0)))  # four percent of cells far above the rest
    robust = transient._error_definition("T", "Absolute error", errors)
    assert (robust.boundaries[0], robust.boundaries[-1] < 5.0, robust.extend) == (0.0, True, "max")
    exact = transient._error_definition("T", "Absolute error", np.concatenate((np.full(960, 5.5), np.full(40, 50.0))))  # top exactly 5.5
    assert exact.extend == "max"
    assert exact.normalizer(np.array([5.5, 50.0])).tolist() == [exact.colormap.N - 1, exact.colormap.N]
    reach = 0.011000000000000001  # reach / 11 rounds to exactly 0.001, whose eleven steps end at 0.011
    rounded = field_maps.rounded_boundaries(0.0, reach)
    assert rounded is not None
    assert rounded[-1] >= reach
    assert field_maps.rounded_boundaries(0.0, 0.0) is None


def test_the_signed_error_scale_is_blue_white_red_symmetric_to_the_larger_percentile_and_fixed_over_all_frames() -> None:
    """Norms over all times give every frame the one scale of the rollout's 5th and 95th percentile of the signed error."""
    record = _record()
    ramp = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    growth = np.arange(3.0).reshape(3, 1, 1, 1)
    steps = np.array([2.0, 0.01, 0.02, 0.02]).reshape(1, 4, 1, 1)
    record.predicted_states = (record.reference_states + growth * steps * (ramp - 0.8)).astype(np.float32)
    record.predicted_states[2, 0, 0, 0] -= 50.0  # one extreme cell beyond the percentile reach
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    shown = np.asarray(record.predicted_states[:, 0], dtype=np.float64) - np.asarray(record.reference_states[:, 0], dtype=np.float64)
    reach = max(abs(np.percentile(shown, 5.0)), abs(np.percentile(shown, 95.0)))
    boundaries = cast("Any", norms["T"].error).boundaries
    assert boundaries[0] == pytest.approx(-boundaries[-1])
    assert boundaries[-2] < reach <= boundaries[-1]  # the round top just covers the percentile reach, not the extremes
    scales = []
    for time in (1.0, 2.0):
        figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=time, norms=norms)
        image = cast("Any", figure.axes[2].collections[0])
        scales.append((tuple(np.asarray(image.norm.boundaries)), image.colorbar.extend))
        centres = (np.asarray(image.norm.boundaries)[:-1] + np.asarray(image.norm.boundaries)[1:]) / 2.0
        error_bar = cast("Any", figure.axes[5])._colorbar
        np.testing.assert_allclose(error_bar.get_ticks(), centres, atol=1e-9)  # the white band is centred on a labelled 0
        assert 0.0 in np.round(error_bar.get_ticks(), 9)
        assert image.get_cmap().name == visual_semantics.field_visual_semantics("T", role="signed_error").colormap
        plt.close(figure)
    assert scales[0] == scales[1]
    assert scales[0][1] != "neither"  # the extremes beyond the percentile reach get arrows


def test_tick_labels_keep_their_decimals_over_all_frames_one_by_default_and_more_for_a_fine_step() -> None:
    """A state bar writes every frame at the decimals of the channel's finest band step over all frames, none for a whole-number step."""
    record = _record()
    ramp = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    record.reference_states[0, 0] = 300.0 + 0.04 * ramp  # frame 0 has a band step of 0.005, frame 1 of 0.05; phi (%) has 1
    record.reference_states[1, 0] = 300.0 + 0.4 * ramp
    record.predicted_states = record.reference_states.copy()
    norms = transient.state_map_norms(record, state_fields=("T", "phi"), time_indices=range(3), error_mode="Signed error")
    decimals = []
    for time in (0.0, 1.0):
        figure = transient.plot_state_maps(record, state_fields=("T", "phi"), physical_time=time, norms=norms)
        figure.canvas.draw()
        bars = (figure.axes[6], figure.axes[9])  # the reference bars of T and phi
        decimals.append([{len(text.partition(".")[2]) for label in bar.get_yticklabels() if (text := label.get_text())} for bar in bars])
        plt.close(figure)
    assert decimals[0] == decimals[1] == [{3}, {0}]


def test_exact_predictions_keep_a_round_error_scale() -> None:
    """All-zero errors get an explicit nondegenerate scale (signed: symmetric about 0 with a centred white band), never a one-colour constant map."""
    sizes = {}
    for signed in (True, False):
        boundaries = field_maps.error_boundaries(np.zeros(4), signed=signed)
        sizes[signed] = boundaries.size - 1
        assert boundaries[0] == pytest.approx(-boundaries[-1] if signed else 0.0)
    assert sizes == {True: 11, False: 10}  # reach 1: w = 0.2 (11 bands, span 2.2) and step 0.1 (10 bands)
    record = _record()
    record.predicted_states = record.reference_states.copy()
    common: dict[str, Any] = {"state_fields": ("T",), "error_mode": "Signed error"}
    definition = cast("Any", transient.state_map_norms(record, time_indices=range(3), **common)["T"].error)
    assert definition.boundaries.size == 12
    assert definition.color_count == 11
    assert definition.semantic is not transient.field_maps.FieldMapSemantic.CONSTANT
    figure = transient.plot_state_maps(record, physical_time=1.0, **common)
    assert cast("Any", figure.axes[2].collections[0]).norm.boundaries.size == 12
    plt.close(figure)


def _drifting_record() -> Any:
    """Return ``_record`` whose temperature drifts by 6 K per stored time in both states, so frame and pooled scales differ."""
    record = _record()
    drift = (6.0 * np.arange(3.0)).reshape(3, 1, 1)
    for states in (record.reference_states, record.predicted_states):
        states[:, 0] += drift.astype(np.float32)
    return record


def _pooled_bands(states: Any) -> Any:
    """Return the round state bands of the 5th to 95th percentile of the temperature pooled over all stored times."""
    pooled = np.concatenate([transient._display_state_values("T", np.asarray(frame[0], dtype=np.float64), error=False).ravel() for frame in states])
    return field_maps.state_boundaries(*np.percentile(pooled, (5.0, 95.0)))


def _frame_bands(states: Any, time: int, *, field: str = "phi", channel: int = 1) -> Any:
    """Return the round state bands of the 5th to 95th percentile of one field at one stored time."""
    shown = transient._display_state_values(field, np.asarray(states[time, channel], dtype=np.float64), error=False)
    return field_maps.state_boundaries(*np.percentile(shown, (5.0, 95.0)))


def test_fixed_time_fields_draw_the_pooled_bands_in_every_frame_and_the_prediction_follows_the_lock() -> None:
    """A fixed field gives every frame the bands of the pooled reference percentiles; without the lock the prediction gets its pooled own."""
    record = _drifting_record()
    reference_bands, prediction_bands = (_pooled_bands(states) for states in (record.reference_states, record.predicted_states))
    assert not np.array_equal(reference_bands, prediction_bands)
    for lock_scale, expected_prediction in ((True, reference_bands), (False, prediction_bands)):
        norms = transient.state_map_norms(
            record, state_fields=("T",), time_indices=range(3), error_mode="Signed error", lock_scale=lock_scale, fixed_time_fields=("T",)
        )
        for time in (0.0, 1.0, 2.0):
            figure = transient.plot_state_maps(
                record, state_fields=("T",), physical_time=time, lock_scale=lock_scale, fixed_time_fields=("T",), norms=norms
            )
            figure.canvas.draw()
            for column, expected in ((0, reference_bands), (1, expected_prediction)):
                image = cast("Any", figure.axes[column].collections[0])
                assert isinstance(image.norm, BoundaryNorm)
                np.testing.assert_allclose(image.norm.boundaries, expected)
                assert expected.size - 1 == image.get_cmap().N
            assert len(figure.axes[3].get_yticklabels()) == reference_bands.size
            plt.close(figure)
    # The viewer path (no norms) pools over all stored times although it draws one.
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=0.0, fixed_time_fields=("T",))
    np.testing.assert_allclose(cast("Any", figure.axes[0].collections[0]).norm.boundaries, reference_bands)
    plt.close(figure)


def test_a_mixed_set_pools_the_fixed_field_and_lets_the_free_field_follow_each_frame() -> None:
    """The fixed field has one pooled band set in every frame; the free field keeps its frame's bands and the union arrows it has alone."""
    record = _drifting_record()
    for states in (record.reference_states, record.predicted_states):
        states[:, 1] += (0.2 * np.arange(3.0)).reshape(3, 1, 1).astype(np.float32)  # the free field drifts too, so its frames differ
    record.predicted_states[0, 1] = record.reference_states[0, 1].mean()  # constant, inside its frame's bands
    record.predicted_states[2, 1] = record.reference_states[2, 1].max() + 0.5  # every cell above its frame's top
    common: dict[str, Any] = {"error_mode": "Signed error", "state_fields": ("T", "phi")}
    norms = transient.state_map_norms(record, time_indices=range(3), fixed_time_fields=("T",), **common)
    alone = transient.state_map_norms(record, time_indices=range(3), **{**common, "state_fields": ("phi",)})
    assert norms["T"].fixed is not None
    assert norms["phi"].fixed is None
    assert (norms["phi"].arrows, norms["phi"].state_format) == (alone["phi"].arrows, alone["phi"].state_format)
    pooled = _pooled_bands(record.reference_states)
    free_bands = []
    for index, time in enumerate((0.0, 1.0, 2.0)):
        figure = transient.plot_state_maps(record, physical_time=time, fixed_time_fields=("T",), norms=norms, **common)
        fixed_image, free_image = (cast("Any", figure.axes[3 * row].collections[0]) for row in (0, 1))
        np.testing.assert_allclose(fixed_image.norm.boundaries, pooled)
        np.testing.assert_allclose(free_image.norm.boundaries, _frame_bands(record.reference_states, index))
        assert free_image.colorbar.extend == norms["phi"].arrows[0]
        # the free prediction bar carries the top arrow frame 2 needs, also in frame 0, whose own values stay inside
        assert cast("Any", figure.axes[4].collections[0]).colorbar.extend in ("max", "both")
        free_bands.append(free_image.norm.boundaries)
        plt.close(figure)
    assert not np.array_equal(free_bands[0], free_bands[2])
    alone_frame = transient.plot_state_maps(record, physical_time=0.0, **common)
    assert cast("Any", alone_frame.axes[4].collections[0]).colorbar.extend not in ("max", "both")
    plt.close(alone_frame)


def test_without_a_fixed_field_every_frame_scales_to_itself_and_norms_must_agree_with_the_request() -> None:
    """No fixed field leaves the norms free and each frame on its own bands; norms and requested fixed fields must name the same drawn fields."""
    record = _drifting_record()
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    assert norms["T"].fixed is None
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=2.0, norms=norms)
    own = _frame_bands(record.reference_states, 2, field="T", channel=0)
    np.testing.assert_allclose(cast("Any", figure.axes[0].collections[0]).norm.boundaries, own)
    plt.close(figure)
    fixed = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error", fixed_time_fields=("T",))
    for given, requested in ((norms, ("T",)), (fixed, ())):
        with pytest.raises(ValueError, match="fixed_time_fields"):
            transient.plot_state_maps(record, state_fields=("T",), physical_time=2.0, fixed_time_fields=requested, norms=given)
    with pytest.raises(ValueError, match="lack the drawn fields"):
        transient.plot_state_maps(record, state_fields=("T", "phi"), physical_time=2.0, fixed_time_fields=("T",), norms=fixed)
    # Both entry points refuse a fixed name that is not drawn, and a bare string instead of a collection of names.
    common: dict[str, Any] = {"state_fields": ("T",), "error_mode": "Signed error"}
    for build in (
        lambda fields: transient.state_map_norms(record, time_indices=range(3), fixed_time_fields=fields, **common),
        lambda fields: transient.plot_state_maps(record, physical_time=2.0, fixed_time_fields=fields, **common),
    ):
        with pytest.raises(ValueError, match="drawn fields"):
            build(("T", "phi"))
        with pytest.raises(TypeError, match="one string"):
            build("T")


def test_the_default_fixed_time_fields_leave_out_the_moisture_fields_and_keep_the_order() -> None:
    """Temperature and relative humidity are fixed over time by default, the moisture fields are not."""
    assert transient.default_fixed_time_fields(("T", "phi", "w_surf", "w_int", "w_gr")) == ("T", "phi")
    assert transient.default_fixed_time_fields(("w_gr", "phi", "T")) == ("phi", "T")
    assert transient.default_fixed_time_fields(("w_surf",)) == ()


def test_the_scale_note_says_nothing_about_times_or_fixed_fields() -> None:
    """The note line is the same with and without a fixed field, with the reference clause only under the lock."""
    record = _drifting_record()
    texts = {}
    for lock_scale, fixed in ((True, ("T",)), (False, ("T",)), (True, ())):
        figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0, lock_scale=lock_scale, fixed_time_fields=fixed)
        (note,) = (text.get_text() for text in figure.texts if text.get_text().startswith("Colour scales:"))
        texts[lock_scale, fixed] = note
        plt.close(figure)
    head, tail = "Colour scales: 5th\N{EN DASH}95th percentile", "arrows mark values outside"
    assert texts[True, ("T",)] == texts[True, ()] == f"{head}; prediction uses the reference scale; {tail}"
    assert texts[False, ("T",)] == f"{head}; {tail}"


def test_the_prediction_never_sets_the_state_scale_when_the_reference_range_is_degenerate() -> None:
    """A degenerate 5th to 95th percentile range falls back to the reference's own full range, else a padded reference-only scale."""
    record = _record()
    record.reference_states[1, 0] = 300.0
    record.reference_states[1, 0, 0, 0] = 305.0  # one outlier cell: the percentile range is degenerate, the full range is not
    record.predicted_states[1, 0] = 400.0  # far outside the reference
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    for column in (0, 1):
        image = cast("Any", figure.axes[column].collections[0])
        assert (image.norm.vmin, image.norm.vmax) == pytest.approx((26.5, 32.0))  # round bands of 0.5 °C covering the full range 26.85 to 31.85
    plt.close(figure)
    record.reference_states[1, 0] = 300.0  # constant: a padded reference-only scale
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=1.0)
    for column in (0, 1):
        image = cast("Any", figure.axes[column].collections[0])
        assert (image.norm.vmin, image.norm.vmax) == pytest.approx((300.0 - 273.15, 300.0 - 273.15), abs=1e-6)
    assert _colorbar_extend(figure.axes[3 + 1]) == "max"
    plt.close(figure)


def test_a_label_that_rounds_to_zero_prints_as_zero_never_minus_zero() -> None:
    """A single tick and a small negative tick are written at one decimal as 0.0."""
    figure, axis = plt.subplots()
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1.0, 1.0)), ax=axis)
    assert layout.tick_format(layout.tick_step(np.array([-0.04]))) == layout.TickFormat(1, 0)
    layout.set_fixed_decimal_ticks(bar, np.array([-0.04]))
    figure.canvas.draw()
    assert [text.get_text() for text in bar.ax.get_yticklabels()] == ["0.0"]
    layout.set_fixed_decimal_ticks(bar, np.array([-0.5, -0.04, 0.5]), layout.TickFormat(1))
    figure.canvas.draw()
    assert [text.get_text() for text in bar.ax.get_yticklabels()] == ["\N{MINUS SIGN}0.5", "0.0", "0.5"]
    plt.close(figure)


def test_a_transient_state_outside_a_thousandth_to_ten_thousand_is_refused_instead_of_drawn_without_its_power_of_ten() -> None:
    """A porosity of about 1e-5 % would need a factor the transient state bars do not carry: ValueError naming the field."""
    record = _record()
    record.reference_states[:, 1] = 1.0e-7 * (1.0 + np.linspace(0.0, 1.0, 25).reshape(5, 5))
    record.predicted_states = record.reference_states.copy()
    with pytest.raises(ValueError, match=r"'phi'.*power of ten"):
        transient.plot_state_maps(record, state_fields=("phi",), physical_time=0.0)


def test_a_near_uniform_frame_widened_across_ten_thousand_by_the_bar_decimals_is_refused() -> None:
    """Around 9999.89 °C its own bands stay below 10^4, but the whole-number decimals of the other frames widen it to 9994-10005."""
    record = _record()
    ramp = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    record.reference_states[0, 0] = 273.15 + 9999.887 + 0.005 * ramp
    record.reference_states[1:, 0] = 273.15 + 9990.0 + 8.0 * ramp
    record.predicted_states = record.reference_states.copy()
    with pytest.raises(ValueError, match=r"'T'.*power of ten"):
        transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")


def test_a_constant_frame_never_sets_the_decimals_of_a_bar() -> None:
    """A constant first frame falls back to a padded range of tiny steps, which must not give the whole GIF 13 decimals."""
    record = _record()
    record.reference_states[0, 0] = 300.0
    record.predicted_states = record.reference_states.copy()
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    assert norms["T"].state_format.decimals == 0  # the other frames' band step is 1 °C


def test_the_mean_error_line_writes_the_power_of_ten_of_its_bar() -> None:
    """A signed bar of band width 5e-5 m/s labels in [10⁻⁵ m/s]; its mean line says 15 times 10⁻⁵ m/s, a coarse bar the plain value."""
    plot_layout = analysis.evaluation.plots.layout
    errors = np.array([1.5e-4, -1.5e-4, np.nan])
    fine = layout.error_format(field_maps.error_boundaries(2e-4 * np.linspace(-1.0, 1.0, 21), signed=True), signed=True)
    assert fine.exponent == -5
    power = "10\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT FIVE}"
    assert plot_layout.mean_error_text(errors, unit="m/s", exponent=fine.exponent) == f"mean |e| = 15 \N{MULTIPLICATION SIGN} {power} m/s"
    assert plot_layout.mean_error_text(errors * 1e4, unit="Pa") == "mean |e| = 1.5 Pa"
    assert plot_layout.mean_error_text(np.array([np.nan]), unit="Pa") == "mean |e| n/a"
    # Three significant digits stay positional: no e-notation for a large or a tiny mean.
    assert plot_layout.mean_error_text(np.array([1234.0]), unit="Pa") == "mean |e| = 1230 Pa"
    assert plot_layout.mean_error_text(np.array([2.3456e-5]), unit="Pa") == "mean |e| = 0.0000235 Pa"


@pytest.mark.parametrize("reach", [3e-4, 2e-3])
def test_a_fine_error_bar_writes_distinct_labels_of_at_most_three_decimals_with_a_power_of_ten_in_its_unit(reach: float) -> None:
    """A step needing more than three decimals scales the ticks by a power of ten named in the unit, so no two labels agree."""
    boundaries = field_maps.error_boundaries(np.linspace(-reach, reach, 101), signed=True)
    figure, axis = plt.subplots()
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(boundaries[0], boundaries[-1])), ax=axis)
    bar.set_label("u [m/s]")
    layout.set_fixed_decimal_ticks(bar, layout.error_ticks(boundaries, signed=True))
    figure.canvas.draw()
    labels = [text.get_text() for text in bar.ax.get_yticklabels()]
    assert len(labels) == len(set(labels)) == boundaries.size - 1  # one label per band centre
    assert all(len(label.partition(".")[2]) <= 3 for label in labels)
    assert bar.ax.get_ylabel().startswith("u [10")
    assert bar.ax.get_ylabel().endswith(" m/s]")
    plt.close(figure)


@pytest.mark.parametrize("fixed_time_fields", [(), ("T",)])
def test_state_arrows_are_fixed_over_the_frames_of_a_gif_and_follow_the_frame_of_a_viewer(fixed_time_fields: tuple[str, ...]) -> None:
    """A GIF bar draws an arrow when any frame passes that end, in every frame; a single frame draws only its own arrows."""
    record = _record()
    record.predicted_states[0, 0] = record.reference_states[0, 0].mean()  # constant, inside the reference's bands
    record.predicted_states[1, 0] = record.reference_states[1, 0].max() + 50.0  # every cell above the reference's top
    common: dict[str, Any] = {"state_fields": ("T",), "error_mode": "Signed error", "fixed_time_fields": fixed_time_fields}
    norms = transient.state_map_norms(record, time_indices=range(3), **common)
    figure, _grid = layout.map_figure(transient.state_map_geometry(record, state_fields=("T",)))
    positions, arrows = [], []
    for time in (0.0, 1.0):
        transient.plot_state_maps(record, physical_time=time, figure=figure, norms=norms, **common)
        figure.canvas.draw()
        positions.append([tuple(axis.get_position().bounds) for axis in figure.axes])
        arrows.append([_colorbar_extend(axis) for axis in figure.axes[3:5]])
        for axis in figure.axes[3:5]:
            colormap = cast("Any", axis)._colorbar.cmap
            assert all(tuple(getter()) != tuple(figure.get_facecolor()) for getter in (colormap.get_under, colormap.get_over))
    plt.close(figure)
    assert positions[0] == positions[1]
    assert arrows[0] == arrows[1] == [norms["T"].arrows[0], norms["T"].arrows[1]]
    assert norms["T"].arrows[1] in ("max", "both")  # the frame without values above still shows the arrow frame 1 needs
    viewer = transient.plot_state_maps(record, physical_time=0.0, **{**common, "fixed_time_fields": ()})
    assert _colorbar_extend(viewer.axes[4]) != norms["T"].arrows[1]  # the constant frame passes no end of its own scale
    plt.close(viewer)


def test_a_step_of_a_quarter_needs_two_decimals_and_a_whole_step_none() -> None:
    """The format follows the decimals the step needs exactly: 0.25 gives 2, 2.5 gives 1, 25 gives 0; a fine error step moves its power of ten out."""
    assert layout.tick_format(0.25) == layout.TickFormat(2)
    assert layout.tick_format(2.5) == layout.TickFormat(1)
    assert layout.tick_format(25.0) == layout.TickFormat(0)
    assert layout.state_format([25.0, 0.25, None]) == layout.TickFormat(2)
    assert layout.tick_format(2.5e-5) == layout.TickFormat(1, -5)


def test_state_maps_are_discrete_on_the_rule_boundaries_with_the_decimals_fixed_over_all_frames() -> None:
    """Each frame draws one colour per band of its own percentile bands; the bar labels its boundaries at one decimal count."""
    record = _record()
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    decimals = set()
    for time, index in ((0.0, 0), (1.0, 1)):
        figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=time, norms=norms)
        figure.canvas.draw()
        image = cast("Any", figure.axes[0].collections[0])
        shown = transient._display_state_values("T", record.reference_states[index, 0], error=False)
        expected = field_maps.state_boundaries(*np.percentile(shown, (5.0, 95.0)))
        assert isinstance(image.norm, BoundaryNorm)
        np.testing.assert_allclose(image.norm.boundaries, expected)
        assert expected.size - 1 == image.get_cmap().N  # one colour per band
        bar = figure.axes[3]
        assert [float(text.get_text().replace("\N{MINUS SIGN}", "-")) for text in bar.get_yticklabels()] == pytest.approx(expected.tolist())
        decimals.add(len(bar.get_yticklabels()[0].get_text().partition(".")[2]))
        plt.close(figure)
    assert decimals == {norms["T"].state_format.decimals}


def test_a_near_uniform_frame_is_widened_and_never_sets_the_format_or_a_factor_of_a_state_bar() -> None:
    """A frame whose round step is below the bar's decimals is widened to 11 steps of 10**-d; the other frames keep the format they need."""
    record = _record()
    ramp = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    record.reference_states[0, 0] = 300.0 + 4e-5 * ramp  # near uniform: a round step of 1e-5 K
    record.reference_states[1, 0] = 300.0 + 4.0 * ramp  # a round step of 1 K
    record.reference_states[2, 0] = 300.0 + 4.0 * ramp
    record.predicted_states = record.reference_states.copy()
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    assert norms["T"].state_format == layout.TickFormat(1, 0)
    widths, labels = [], []
    for time in (0.0, 1.0):
        figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=time, norms=norms)
        figure.canvas.draw()
        image = cast("Any", figure.axes[0].collections[0])
        widths.append(image.norm.vmax - image.norm.vmin)
        bar = figure.axes[3]  # the reference bar
        labels.append((bar.get_ylabel(), {len(text.get_text().partition(".")[2]) for text in bar.get_yticklabels() if text.get_text()}))
        plt.close(figure)
    assert widths[0] == pytest.approx(1.1)  # 11 steps of 0.1 K about the frame's centre
    assert widths[1] > 3.0  # the other frame keeps its own percentile range
    assert labels == [("[°C]", {1}), ("[°C]", {1})]  # no factor anywhere, one decimal


def test_near_uniform_frames_alone_keep_three_decimals_and_a_degenerate_frame_counts_with_its_full_range() -> None:
    """Only near-uniform frames: three decimals and 11 steps of 0.001; one outlier cell: the full range's step of 0.005 sets three."""
    record = _record()
    record.reference_states[:, 0] = 300.0 + 4e-5 * np.linspace(0.0, 1.0, 25).reshape(5, 5)
    record.predicted_states = record.reference_states.copy()
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    assert norms["T"].state_format == layout.TickFormat(3)
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=0.0, norms=norms)
    image = cast("Any", figure.axes[0].collections[0])
    assert image.norm.vmax - image.norm.vmin == pytest.approx(0.011)
    plt.close(figure)
    record.reference_states[:, 0] = 300.0
    record.reference_states[:, 0, 0, 0] = 300.05  # the 5th and 95th percentiles coincide, the full range is 0.05 K
    record.predicted_states = record.reference_states.copy()
    norms = transient.state_map_norms(record, state_fields=("T",), time_indices=range(3), error_mode="Signed error")
    assert norms["T"].state_format == layout.TickFormat(3)


def test_a_near_uniform_steady_state_map_gets_at_most_three_decimals_and_no_factor() -> None:
    """A state map of 26.85 ± 5e-5 is widened to 11 steps of 0.001 and its bar labels three decimals, never a power of ten."""
    boundaries = analysis.evaluation.plots.samples_outliers._state_boundaries(26.85 + 5e-5 * np.linspace(-1.0, 1.0, 101))
    assert boundaries is not None
    assert boundaries.size == 12
    np.testing.assert_allclose(np.diff(boundaries), 1e-3)
    assert boundaries[0] < 26.85 - 5e-5
    assert boundaries[-1] > 26.85 + 5e-5
    figure, axis = plt.subplots()
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(boundaries[0], boundaries[-1])), ax=axis)
    bar.set_label("T [°C]")
    layout.set_fixed_decimal_ticks(bar, layout.error_ticks(boundaries, signed=False))
    figure.canvas.draw()
    assert bar.ax.get_ylabel() == "T [°C]"
    assert {len(text.get_text().partition(".")[2]) for text in bar.ax.get_yticklabels()} == {3}
    plt.close(figure)


def test_a_constant_reference_frame_gets_one_tick_at_its_value() -> None:
    """A uniform temperature reference has one tick on its colour bar, not several equal ones."""
    record = _record()
    record.reference_states[0, 0] = 293.15
    record.predicted_states = record.reference_states.copy()
    figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=0.0)
    figure.canvas.draw()
    bar = figure.axes[3]
    assert [text.get_text() for text in bar.get_yticklabels()] == ["20.0"]
    plt.close(figure)


def test_a_reused_frame_figure_keeps_its_size_and_axes_positions_and_refuses_another_size() -> None:
    """Animation frames redraw on one fixed figure without moving any axes."""
    record = _record()
    geometry = transient.state_map_geometry(record, state_fields=FIELDS)
    figure, _grid = layout.map_figure(geometry)
    positions = []
    norms = transient.state_map_norms(record, state_fields=FIELDS, time_indices=range(3), error_mode=error_diagnostics.DEFAULT_ERROR_MODE)
    for time in (0.0, 2.0):  # GIF-style norms: the arrows, and with them every bar, are the same in all frames
        assert transient.plot_state_maps(record, state_fields=FIELDS, physical_time=time, figure=figure, norms=norms) is figure
        figure.canvas.draw()
        positions.append([tuple(axis.get_position().bounds) for axis in figure.axes])
    assert positions[0] == positions[1]
    np.testing.assert_allclose(figure.get_size_inches(), (geometry.figure_width, geometry.figure_height))
    plt.close(figure)
    other, _grid = layout.map_figure(layout.plan_geometry(1, 3, (1.0, 1.0)))
    with pytest.raises(ValueError, match="geometry's size"):
        transient.plot_state_maps(record, state_fields=FIELDS, physical_time=0.0, figure=other)
    plt.close(other)


def test_frame_titles_keep_one_text_width_over_stored_times_of_different_lengths() -> None:
    """The time is written at the width of the widest stored time, so the centred title never shifts."""
    record = _record()
    record.physical_times = np.array((0.0, 1.5, 10.0))
    widths = []
    for time in (1.5, 10.0):
        figure = transient.plot_state_maps(record, state_fields=("T",), physical_time=time, title_context="Lentil · case 7")
        figure.canvas.draw()
        (title,) = (text for text in figure.texts if text.get_text() == figure.get_suptitle())
        widths.append(title.get_window_extent().width)
        plt.close(figure)
    assert widths[0] == pytest.approx(widths[1], abs=0.5)


def test_a_colour_decodes_alike_in_a_small_patch_and_filling_a_later_frame() -> None:
    """A colour in a 3 x 3 patch beside thousands of others decodes like the same colour filling a later frame."""
    salmon = (252, 138, 106)
    x, y = np.meshgrid(np.linspace(0, 255, 200), np.linspace(0, 255, 200))
    gradient = np.stack((x, y, np.full_like(x, 128.0)), axis=-1).astype(np.uint8)
    first = gradient.copy()
    first[10:13, 10:13] = salmon
    last = np.empty_like(gradient)
    last[...] = salmon
    last[:20] = gradient[:20]
    handle = BytesIO()
    writer = panel._HandleGifWriter(handle, panel._GIF_FPS)
    writer._frames = [Image.fromarray(frame) for frame in (first, gradient, last)]
    writer.finish()
    handle.seek(0)
    with Image.open(handle) as image:
        assert image.info["loop"] == 0
        assert image.info["duration"] == pytest.approx(1000 / panel._GIF_FPS, abs=10)
        image.seek(0)
        small = image.convert("RGB").getpixel((11, 11))
        image.seek(2)
        large = image.convert("RGB").getpixel((100, 150))
    assert small == large


def _colormap_frames() -> tuple[list[np.ndarray], dict[str, np.ndarray], tuple[slice, slice]]:
    """
    Return three RGB frames of real state and error colormaps, the colormaps' colours, and one late patch.

    Each 1000 x 1400 frame (4.2 Mpx in all, the size of a real export) holds two-dimensional gradients of the
    three state colormaps without their pale ends beside a large pale error area and a white margin; only the
    last frame has a patch of 8 pixels on one row at the pale end of YlGnBu.
    """
    state = {field: colormaps[visual_semantics.field_visual_semantics(field).colormap] for field in FIELDS}
    error = colormaps[visual_semantics.field_visual_semantics("T", role="absolute_error").colormap].resampled(11)
    palettes = {name: (colormap(np.linspace(0.0, 1.0, colormap.N))[:, :3] * 255).round() for name, colormap in state.items()}
    palettes["error"] = (error(np.arange(11))[:, :3] * 255).round()
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 300), np.linspace(0.0, 1.0, 1000))
    patch = (slice(500, 501), slice(400, 408))
    frames = []
    for shift in (0.0, 0.1, 0.2):
        values = 0.15 + 0.85 * ((0.8 * x + 0.2 * y + shift) % 1.0)
        panels = [colormap(values)[..., :3] for colormap in state.values()]
        pale_errors = error(np.minimum((3.0 * (x[:, :225] + y[:, :225] + shift)) % 3.0, 2.0).astype(int))[..., :3]
        frame = np.concatenate((*panels, pale_errors, pale_errors, np.ones((1000, 50, 3))), axis=1)
        frames.append((frame * 255).round().astype(np.uint8))
    frames[-1][patch] = palettes["phi"][0]
    return frames, palettes, patch


def _decoded_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Write ``frames`` through the GIF writer and return the decoded RGB frames."""
    handle = BytesIO()
    writer = panel._HandleGifWriter(handle, panel._GIF_FPS)
    writer._frames = [Image.fromarray(frame) for frame in frames]
    writer.finish()
    handle.seek(0)
    decoded = []
    with Image.open(handle) as image:
        for index in range(len(frames)):
            image.seek(index)
            decoded.append(np.asarray(image.convert("RGB"), dtype=np.float64))
    return decoded


def test_gif_palette_keeps_a_rare_pale_state_colour_in_its_colormap_and_bounds_every_decoding_error() -> None:
    """A pale YlGnBu colour seen only in one late 8-pixel patch decodes near itself, never as a pale error colour."""
    frames, palettes, patch = _colormap_frames()
    decoded = _decoded_frames(frames)
    errors = np.concatenate([np.linalg.norm(out - source, axis=-1).ravel() for out, source in zip(decoded, frames, strict=True)])
    patch_colours = decoded[-1][patch].reshape(-1, 3)
    assert np.linalg.norm(patch_colours - palettes["phi"][0], axis=-1).max() <= _MAX_DECODING_ERROR_RGB
    owners = list(palettes)
    for colour in patch_colours:
        assert owners[int(np.argmin([np.linalg.norm(table - colour, axis=-1).min() for table in palettes.values()]))] == "phi"
    assert errors.max() <= _MAX_DECODING_ERROR_RGB


def test_gif_palette_follows_a_large_smooth_ramp_beside_many_scattered_colours() -> None:
    """A large YlGnBu ramp of several hundred colours decodes without terraces although 38,400 scattered colours compete."""
    ramp_colours = colormaps[visual_semantics.field_visual_semantics("w_surf").colormap].resampled(1024)
    ramp = (ramp_colours(np.linspace(0.0, 1.0, 800))[:, :3] * 255).round().astype(np.uint8)
    panels = [np.repeat(ramp[None], 600, axis=0)]
    for field in ("T", "w_gr"):
        gradient = colormaps[visual_semantics.field_visual_semantics(field).colormap](np.linspace(0.0, 1.0, 200))[:, :3]
        panels.append(np.repeat((gradient[None] * 255).round().astype(np.uint8), 600, axis=0))
    panels.append(np.random.default_rng(7).integers(0, 256, (600, 64, 3), dtype=np.uint8))
    frame = np.concatenate(panels, axis=1)
    mirrored = frame[::-1, ::-1].copy()  # every pixel changes, so ramp and scattered colours compete in each frame's palette
    assert len(np.unique(ramp, axis=0)) > 300
    first, second = _decoded_frames([frame, mirrored])
    errors = np.linalg.norm(np.stack((first[:, :800], second[:, -800:])) - np.stack((frame[:, :800], mirrored[:, -800:])), axis=-1)
    assert errors.max() <= 4.0
    assert errors.mean() <= 2.0


def _bars_and_moving_ramps() -> tuple[list[np.ndarray], np.ndarray, tuple[slice, slice]]:
    """
    Return twelve frames with static colour bars, grey text and margin, their static mask and their map area.

    The map area's colormap ramp differs in every frame; only frame 0's holds the dark half of YlGnBu (1024 levels),
    the colours a drying state shows only early.
    """
    rows = 120
    area = (slice(0, rows), slice(0, 800))

    def ramp(name: str, start: float, count: int) -> np.ndarray:
        colours = colormaps[name].resampled(1024)(np.linspace(start, 1.0, count))[:, :3]
        return np.repeat((colours[None] * 255).round().astype(np.uint8), rows, axis=0)

    bars = np.stack([colormaps[name].resampled(1024)(np.linspace(0.0, 1.0, rows))[:, :3] for name in ("YlGnBu", "cividis")], axis=1)
    static = np.concatenate(
        (
            (np.repeat(bars, 8, axis=1) * 255).round().astype(np.uint8),  # two vertical colour bars, 8 pixels wide
            np.random.default_rng(5).integers(0, 256, (rows, 24, 1), dtype=np.uint8).repeat(3, axis=2),  # grey text
            np.full((rows, 40, 3), 255, dtype=np.uint8),
        ),
        axis=1,
    )
    names = ("inferno", "cividis", "viridis", "plasma", "magma", "Reds", "Greens", "Purples", "Oranges", "cubehelix", "turbo")
    maps = [ramp("YlGnBu", 0.5, 800), *(ramp(name, 0.0, 800) for name in names)]
    frames = [np.concatenate((values, static), axis=1) for values in maps]
    mask = np.all(np.stack(frames) == frames[0], axis=(0, 3))
    return frames, mask, area


def test_static_pixels_decode_to_one_colour_in_every_frame_and_near_their_source() -> None:
    """Colour bars, text and margin, equal in every frame, decode identically in every frame, so bars never shimmer."""
    frames, static, _area = _bars_and_moving_ramps()
    decoded = np.stack(_decoded_frames(frames))
    assert static[:, 800:].all()
    assert not static[:, :800].any()
    assert (decoded[:, static] == decoded[0, static]).all()
    assert np.linalg.norm(decoded[0, static] - frames[0][static], axis=-1).max() <= _MAX_DECODING_ERROR_RGB


def test_a_ramp_shown_only_in_the_first_frame_keeps_fine_steps_beside_other_colormaps_in_later_frames() -> None:
    """The dark YlGnBu half seen only at t = 0 decodes without terraces although eleven other ramps fill later frames."""
    frames, _static, area = _bars_and_moving_ramps()
    assert len(np.unique(frames[0][area].reshape(-1, 3), axis=0)) > 256  # more colours than any palette holds
    errors = np.linalg.norm(_decoded_frames(frames)[0][area] - frames[0][area], axis=-1)
    assert errors.max() <= 4.0
    assert errors.mean() <= 2.0


def test_a_later_frame_whose_changed_pixels_hold_at_most_255_colours_decodes_them_exactly() -> None:
    """A later frame's palette serves only its changed pixels, so unchanged colour bars and text take none of it."""
    frames, _static, area = _bars_and_moving_ramps()
    later = frames[1].copy()
    later[area] = (colormaps["viridis"].resampled(250)(np.linspace(0.0, 1.0, 800))[None, :, :3] * 255).round().astype(np.uint8)
    assert 200 < len(np.unique(later[area].reshape(-1, 3), axis=0)) <= 255
    np.testing.assert_array_equal(_decoded_frames([frames[0], later])[1][area], later[area])


def test_every_decoded_frame_of_a_delta_gif_equals_its_source() -> None:
    """Pillow's decoder, compositing transparency and disposal, returns every frame exactly when each has few colours."""
    height, width = 60, 90
    blocks = np.random.default_rng(11).integers(0, 256, (4, 3), dtype=np.uint8)
    ramp = colormaps["YlGnBu"]
    frames = []
    for index in range(6):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        frame[:, :8] = (colormaps["cividis"](np.linspace(0.0, 1.0, height))[:, None, :3] * 255).round()  # static bar
        frame[10:40, 12:60] = (ramp((np.linspace(0.0, 0.5, 48) + 0.1 * index) % 1.0)[None, :, :3] * 255).round()  # moving ramp
        frame[45:55, 12:30] = blocks[index % 2]  # toggles back to an earlier colour
        frame[45:55, 40:60] = blocks[2] if index < 5 else blocks[3]  # changes only in the last frame
        frame[2, 70 + index] = 0  # one pixel turns black while the previous one turns white again
        frames.append(frame)
    decoded = _decoded_frames(frames)
    for frame, result in zip(frames, decoded, strict=True):
        np.testing.assert_array_equal(result, frame)


def test_gif_frames_map_each_encoded_pixel_to_its_nearest_entry_and_leave_unchanged_pixels_transparent() -> None:
    """Brute force confirms every encoded pixel's nearest entry; later frames mark unchanged pixels transparent."""
    rng = np.random.default_rng(3)
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    frame[50:150, 50:150] = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    sources = [frame, frame[::-1].copy(), frame[::-1].copy()]  # the last frame changes nothing

    def indexed(arrays: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, Any]]:
        images = [Image.fromarray(array) for array in arrays]
        panel._index_gif_frames(images)
        palettes = [np.asarray(image.getpalette(), dtype=np.int64).reshape(-1, 3) for image in images]
        return [(np.asarray(image), palette, image.info.get("transparency")) for image, palette in zip(images, palettes, strict=True)]

    first, again = indexed(sources), indexed(sources)
    for index, ((indices, palette, transparent), repeated, source) in enumerate(zip(first, again, sources, strict=True)):
        np.testing.assert_array_equal(indices, repeated[0])
        np.testing.assert_array_equal(palette, repeated[1])
        encoded = np.ones(source.shape[:2], dtype=bool) if index == 0 else np.any(source != sources[index - 1], axis=-1)
        assert transparent == (None if index == 0 else 255)
        assert (indices[~encoded] == 255).all()
        entries = palette if index == 0 else palette[:255]
        colours = source[encoded].astype(np.int64)
        distances = ((colours[:, None, :] - entries[None, :, :]) ** 2).sum(axis=-1)
        np.testing.assert_array_equal(distances[np.arange(len(colours)), indices[encoded]], distances.min(axis=1))
    assert not np.any(first[2][0] != 255)
    assert tuple(first[0][1][first[0][0][0, 0]]) == (255, 255, 255)
    few = [(source // 64) * 64 for source in sources]  # 64 colours in all
    for index, ((indices, palette, _transparent), source) in enumerate(zip(indexed(few), few, strict=True)):
        encoded = np.ones(source.shape[:2], dtype=bool) if index == 0 else np.any(source != few[index - 1], axis=-1)
        np.testing.assert_array_equal(palette[indices[encoded]], source[encoded])


def test_gif_admission_admits_a_production_record_of_169_hourly_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GIF states T, phi and w_gr at 1740 x 996 pixels over 169 stored times (0 to 168 h) stay under the memory limit."""
    y, x = np.meshgrid(np.linspace(0.0, 0.75, 4), np.linspace(0.0, 1.2, 4), indexing="ij")
    record = SimpleNamespace(
        physical_times=np.arange(169.0),
        reference_states=np.zeros((169, 4, 4, 4), dtype=np.float32),
        static_conditioning=np.stack((x, y, *np.zeros((5, 4, 4)))),
    )
    fields = ("T", "phi", "w_gr")
    geometry = transient.state_map_geometry(record, state_fields=fields)
    assert (round(geometry.figure_width * panel._GIF_DPI), round(geometry.figure_height * panel._GIF_DPI)) == (1740, 996)
    started: list[str] = []
    monkeypatch.setattr(analysis.ui.notebook, "write_export", lambda _status, *, stem, **_kwargs: started.append(stem))
    panel._export_state_map_gif(
        analysis.ui.controls.StatusLine(), stem="production", record=record, plot_kwargs={"state_fields": fields}, fps=panel._GIF_FPS
    )
    assert started == ["production"]


def test_a_frame_without_finite_values_keeps_every_colour_bar_and_the_error_scale_fixed_over_all_frames() -> None:
    """A diverged prediction frame shows the same bars, limits and header notes as a finite frame of the same reference."""
    record = sequence_record("autonomous_full", "full")
    record.reference_states[:, 0] = 300.0 + np.linspace(0.0, 4.0, 25).reshape(5, 5)
    record.predicted_states = record.reference_states.copy()
    record.predicted_states[1, 0] = np.nan  # prediction and error map unavailable at t = 1
    # The reference mirrored: the same value range, so the same shared state scale, and |e|/s up to 2, the largest error.
    record.predicted_states[2, 0] = record.reference_states[2, 0][::-1, ::-1]
    scale = error_diagnostics.transient_reference_scale(scaling_state(), field="T", unit="K", dataset_identity="a" * 64)
    common: dict[str, Any] = {"state_fields": ("T",), "error_mode": error_diagnostics.NORMALIZED_ABSOLUTE_ERROR, "reference_scales": {"T": scale}}
    norms = transient.state_map_norms(record, time_indices=range(3), **common)
    frames, extends = [], []
    for time in (1.0, 2.0):
        figure = transient.plot_state_maps(record, physical_time=time, norms=norms, **common)
        bars = [(slot.get_visible(), tuple(slot.get_ylim())) for slot in figure.axes[3:]]
        notes = [text.get_text() for axis in figure.axes[1:2] for text in axis.texts if "colour" in text.get_text()]
        frames.append((bars, notes))
        extends.append([_colorbar_extend(slot) for slot in figure.axes[3:6]])
        plt.close(figure)
    assert frames[0] == frames[1]
    assert all(visible for visible, _limits in frames[0][0])
    assert frames[0][0][2][1] == pytest.approx((0.0, 2.0))
    # State arrows are the union over the frames: the mirrored prediction holds the reference's values, so its bar gets the
    # arrow (the lowest cells pass 27 °C) in every frame, also in the diverged frame that has no value beyond.
    assert extends[0] == extends[1] == ["min", "min", "neither"]


def test_sample_title_names_the_compact_case_and_the_frame_only_when_comparing_and_fits_the_frame() -> None:
    """A real case key and run frame name stay out of a single-model title, which then lies inside the drawn frame."""
    entry = SimpleNamespace(
        material_family="lentil",
        dataset_role="id",
        evaluation_assignment=None,
        case_id="lentil__case_0041__ab0c122b4995",
        case_label="case_0041",
        frame_name="architecture_comparison__uno_m64x64_h32_l7_p0p1_s1-05-05-1-1-2-2_r0p495__lr-0p000532__a0smp-one-step-transition TEST",
    )
    single = panel._transient_sample_title(entry, model_label=None)
    assert "case_0041" in single
    assert entry.case_id not in single
    assert entry.frame_name not in single
    assert "U-NO · A0" in panel._transient_sample_title(entry, model_label="U-NO · A0")
    record = _record()
    geometry = transient.state_map_geometry(record, state_fields=FIELDS)
    figure, _grid = layout.map_figure(geometry)
    transient.plot_state_maps(record, state_fields=FIELDS, physical_time=1.0, figure=figure, title_context=single)
    figure.canvas.draw()
    (title,) = (text for text in figure.texts if text.get_text() == figure.get_suptitle())
    extent = title.get_window_extent(cast("Any", figure.canvas).get_renderer())
    assert figure.bbox.x0 <= extent.x0 <= extent.x1 <= figure.bbox.x1
    assert extent.y1 <= figure.bbox.y1
    plt.close(figure)


def test_gif_admission_refuses_an_export_whose_measured_image_peak_exceeds_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live Pillow images plus the arrays of the palette step stay under the estimate: a limit just below them refuses."""
    live: weakref.WeakValueDictionary[int, Image.Image] = weakref.WeakValueDictionary()  # Pillow images are unhashable
    peak = [0]
    original_init = Image.Image.__init__

    def tracked_init(self: Image.Image) -> None:
        original_init(self)
        live[id(self)] = self

    def measured(name: str) -> Any:
        method = getattr(Image.Image, name)

        def call(self: Image.Image, *args: Any, **kwargs: Any) -> Any:
            result = method(self, *args, **kwargs)
            images = list(live.values())
            peak[0] = max(peak[0], sum(image.width * image.height * (1 if image.mode in {"1", "L", "P"} else 4) for image in images))
            return result

        return call

    finish = panel._HandleGifWriter.finish
    arrays = [0]

    def traced_finish(self: Any) -> None:
        # The colour tables and keys are numpy arrays, which tracemalloc sees and the image census does not.
        tracemalloc.start()
        try:
            finish(self)
            arrays[0] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    monkeypatch.setattr(Image.Image, "__init__", tracked_init)
    for name in ("convert", "copy"):
        monkeypatch.setattr(Image.Image, name, measured(name))
    monkeypatch.setattr(panel._HandleGifWriter, "finish", traced_finish)
    monkeypatch.setattr(analysis.ui.notebook, "_export_destination", lambda: tmp_path)
    plot_kwargs: dict[str, Any] = {
        "state_fields": FIELDS,
        "lock_scale": True,
        "fixed_time_fields": (),
        "error_mode": "Absolute error",
        "reference_scales": None,
        "display_spatial_stride": "auto",
        "title_context": "Lentil · case 7",
    }
    record = _record()
    status = analysis.ui.controls.StatusLine()
    panel._export_state_map_gif(status, stem="admitted", record=record, plot_kwargs=plot_kwargs, fps=panel._GIF_FPS)
    assert len(list(tmp_path.glob("*.gif"))) == 1
    assert peak[0] > 0
    assert arrays[0] > 0
    monkeypatch.setattr(panel, "_GIF_MEMORY_LIMIT_BYTES", peak[0] + arrays[0] - 1)
    panel._export_state_map_gif(status, stem="refused", record=record, plot_kwargs=plot_kwargs, fps=panel._GIF_FPS)
    assert "Refused" in status.text
    assert len(list(tmp_path.glob("*.gif"))) == 1
