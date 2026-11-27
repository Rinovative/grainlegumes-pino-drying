"""Protect unequal trajectory support without synthetic tails or interpolation.

Responsibilities:
  - Check exact-time aggregate support, case weighting, scale contracts, and spectral intervals

Design principles:
  - Unequal trajectories retain their tails without interpolation or synthetic contributors

This module does NOT:
  - Extend source trajectories or fit training normalizers
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import BoundaryNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src import analysis, domain
from src.analysis.catalog import analysis_catalog_transient as catalog_transient
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda import eda_dataframe
from src.analysis.eda import eda_transient as transient
from src.analysis.eda.eda_dataframe import PublishedTensorValue
from src.analysis.eda.plots import eda_plot_case_statistics as statistics
from src.analysis.eda.plots import eda_plot_spectral_analysis as spectra
from src.analysis.eda.plots import eda_plot_transient as plots
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_visual_semantics as semantics
from src.analysis.ui import analysis_ui_time as time_ui

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.patches import Rectangle


def test_aggregate_preserves_long_tail_and_equal_case_weight() -> None:
    """Unequal trajectory lengths retain observed tails and unweighted cases."""
    times, low, median, high, counts = transient.aggregate_exact_series(
        [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])],
        [np.array([10.0, 20.0, 80.0]), np.array([30.0, 40.0])],
    )
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(counts, [2, 2, 1])
    np.testing.assert_allclose(median, [20.0, 30.0, 80.0])
    np.testing.assert_allclose(low, [12.0, 22.0, 80.0])
    np.testing.assert_allclose(high, [28.0, 38.0, 80.0])


def test_disjoint_timestamps_do_not_acquire_interpolated_contributors() -> None:
    """Different time coordinates contribute only at their stored times."""
    times, _low, median, _high, counts = transient.aggregate_exact_series(
        [np.array([0.0, 2.0]), np.array([1.0, 3.0])],
        [np.array([4.0, 8.0]), np.array([11.0, 15.0])],
    )
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(counts, [1, 1, 1, 1])
    np.testing.assert_array_equal(median, [4.0, 11.0, 8.0, 15.0])


@pytest.mark.parametrize(
    ("times", "values"),
    [
        (np.array([0.0, 1.0]), np.array([0.0, np.nan])),
        (np.array([0.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([2, 1], dtype=np.uint32), np.array([0.0, 1.0])),
    ],
)
def test_invalid_source_support_fails_instead_of_shrinking_score(
    times: np.ndarray,
    values: np.ndarray,
) -> None:
    """Invalid values or duplicate/out-of-order times fail scientific admission."""
    with pytest.raises(ValueError, match="finite aligned values"):
        transient.aggregate_exact_series([times], [values])


def test_the_master_time_marker_at_the_main_windows_start_is_drawn_unclipped() -> None:
    """The master-time marker at the main window's 1 h start sits on its frame, so it is drawn unclipped and shows its whole disc."""
    figure, axes = plt.subplots(2, 2)
    try:
        label, frame = next(iter(catalog_transient._eda_datasets().items()))
        schedules = list(axes[:, 0])
        plots._draw_exact_schedule_split_axes(
            schedules,
            list(axes[:, 1]),
            datasets={label: frame},
            case_ids={label: "case_0001"},
            colors={label: "black"},
            linestyles={label: "-"},
            master_time_hours=1.0,
            case_times={label: time_ui.ResolvedPhysicalTime(1.0, exact_master_match=True, final_hold=False)},
        )
        markers = [marker for axis in schedules for marker in axis.collections]
        assert [axis.get_xlim()[0] for axis in schedules] == [marker.get_offsets()[0][0] for marker in markers] == [1.0, 1.0]
        assert not any(marker.get_clip_on() for marker in markers)
    finally:
        plt.close(figure)


def test_transitions_keep_the_stored_states_where_float32_increments_do_not_add_back() -> None:
    """Every transition starts and ends at stored states, never at a state plus a rounded increment."""
    states = np.asarray([np.full((4, 2, 2), value) for value in (0.35, 0.000123, 0.0001)], dtype=np.float32)
    increments = states[1:] - states[:-1]
    assert not np.array_equal(states[0] + increments[0], states[1])
    item = {
        "state": states[0],
        "next_state": states[1:],
        "boundary": np.zeros((2, 4), dtype=np.float32),
        "time": {"t_n": np.array([0.0, 1.0]), "t_n_plus_1": np.array([1.0, 2.0]), "dt": np.ones(2)},
    }
    records = transient._runtime_transitions(item)
    for index, record in enumerate(records):
        np.testing.assert_array_equal(record["state"], states[index])
        np.testing.assert_array_equal(record["next_state"], states[index + 1])


def test_published_tensor_scales_follow_signed_and_encoded_field_contracts() -> None:
    """An all-positive signed field stays centered; log-encoded K is sequential over its negative range; physical K stays geometric."""
    x, y = np.meshgrid(np.arange(3), np.arange(2))
    values = np.arange(1.0, 7.0).reshape(2, 3)
    for name, unit, representation, array, kind in (
        ("Kxy", "1", "dimensionless_cross_component_ratio_to_geometric_mean", values, "signed"),
        ("u", "m/s", "identity", values, "signed"),
        ("delta_T", "K", "physical_increment", values, "signed"),
        ("Kxx", "1", "dimensionless_log10_ratio_to_1_m2", -values, "sequential"),
        ("Kxx", "m^2", "identity", values * 1e-9, "geometric"),
    ):
        value = PublishedTensorValue("static", "input", name, unit, representation, "spatial", array)
        figure = statistics.plot_published_tensor(value, x=x, y=y, display_spatial_stride=2)
        try:
            image = figure.axes[0].collections[0]
            assert isinstance(image.norm, BoundaryNorm)
            boundaries = image.norm.boundaries
            assert len(boundaries) - 1 == image.cmap.N
            if kind == "signed":
                centre = image.cmap.N // 2  # an odd count of round bands, the neutral one around 0
                assert image.cmap.N in {9, 11, 13}
                assert boundaries[0] == -boundaries[-1]
                assert boundaries[centre] < 0 < boundaries[centre + 1]
            elif kind == "sequential":
                assert boundaries[0] <= array.min()  # the values' own range, not centred on 0
                assert array.max() <= boundaries[-1] < 0
            else:
                assert len(boundaries) == 12
                np.testing.assert_allclose(np.diff(np.log(boundaries)), np.repeat(np.diff(np.log(boundaries))[0], 11))
        finally:
            plt.close(figure)


def test_published_inlet_tensor_is_plotted_over_inlet_x_coordinates() -> None:
    """The inlet edge y = min(y) is plotted against its x coordinates."""
    x, y = np.meshgrid(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0]))
    inlet = np.array([5.0, 6.0, 7.0, 8.0])
    value = PublishedTensorValue("boundary", "input", "p_in_bc", "Pa", "identity", "inlet boundary", inlet)
    figure = statistics.plot_published_tensor(value, x=x, y=y)
    try:
        line = figure.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), x[0])
        np.testing.assert_array_equal(line.get_ydata(), inlet)
        assert figure.axes[0].get_xlabel() == "Inlet x [m]"
        assert figure.axes[0].get_ylabel() == r"$p_{\mathrm{in,bc}}$ [Pa]"
        assert figure.axes[0].get_title() == r"boundary: $p_{\mathrm{in,bc}}$"
    finally:
        plt.close(figure)


def test_published_tensors_show_display_units_and_wet_basis_moisture() -> None:
    """Water maps become wet-basis percent with the case density and require it; scalar phi and T read % and °C."""
    x, y = np.meshgrid(np.arange(3.0), np.arange(2.0))
    water = np.array([[100.0, 50.0, 0.0], [25.0, 100.0, 50.0]])
    density = np.array([[400.0, 450.0, 500.0], [475.0, 400.0, 450.0]])
    value = PublishedTensorValue("state", "input", "w_surf", "kg/m^3", "identity", "spatial", water)
    figure = statistics.plot_published_tensor(value, x=x, y=y, rho_bu_dry=density)
    try:
        np.testing.assert_allclose(np.asarray(figure.axes[0].collections[0].get_array()).ravel(), (100.0 * water / (density + water)).ravel())
        assert figure.axes[1].get_ylabel() == r"$X_{\mathrm{wb,surf}}$ [%]"
        assert figure.axes[0].get_title() == r"state: $\mathbf{X_{\mathbf{wb,surf}}}$"  # the column header, mathtext set bold
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="dry bulk density"):
        statistics.plot_published_tensor(value, x=x, y=y)
    for name, unit, stored, shown in (
        ("phi_in_bc", "1", 0.2, r"$\varphi_{\mathrm{in,bc}}$ = 20 [%]"),
        ("T_in_bc", "K", 300.0, r"$T_{\mathrm{in,bc}}$ = 26.85 [°C]"),
    ):
        scalar = PublishedTensorValue("boundary", "input", name, unit, "identity", "transition boundary conditioning", np.asarray(stored))
        figure = statistics.plot_published_tensor(scalar, x=x, y=y)
        try:
            assert figure.axes[0].texts[0].get_text().startswith(shown)
        finally:
            plt.close(figure)


def test_spectral_fraction_maps_share_eleven_fixed_logarithmic_intervals() -> None:
    """Both spectral orientations share the established numerical floor and range."""
    scale = spectra._power_fraction_scale()
    assert scale.color_count == 11
    np.testing.assert_allclose(scale.boundaries, np.geomspace(1e-12, 1.0, 12))
    assert scale.is_locked


def test_published_inlet_profile_follows_the_edge_of_smallest_y_along_x() -> None:
    """With the inlet row stored last (y descending), the profile still runs along that row's x coordinates."""
    x, y = np.meshgrid(np.array([0.0, 0.5, 1.2]), np.array([0.75, 0.4, 0.0]))
    inlet = np.array([3.0, 2.0, 1.0])
    value = PublishedTensorValue("boundary", "input", "p_in_bc", "Pa", "identity", "inlet boundary", inlet)
    figure = statistics.plot_published_tensor(value, x=x, y=y)
    try:
        line = figure.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), x[-1])
        np.testing.assert_array_equal(line.get_ydata(), inlet)
        assert figure.axes[0].get_xlabel() == "Inlet x [m]"
    finally:
        plt.close(figure)


def test_histogram_grid_labels_x_on_the_lowest_panel_of_every_column_and_draws_dashes_in_the_legend() -> None:
    """A column that ends a row early still gets its x label; a dashed legend line is thin enough to show its dashes."""
    styles = plots.batch_styles({"A": "lentil", "B": "lentil"}, roles={"A": "id", "B": "parameter_ood"})
    rng = np.random.default_rng(0)
    data = {name: {key: rng.normal(size=20) for key in "abcde"} for name in styles}
    figure = statistics._hist_grid(data_by_dataset=data, active_datasets=list(styles), columns=list("abcde"), title="t", styles=styles)
    assert isinstance(figure, Figure)
    try:
        assert [axis.get_xlabel() for axis in figure.axes[:5]] == ["", "", "Value", "Value", "Value"]  # 3 columns: the last row has two panels
        handles = figure.legends[0].legend_handles
        assert all(isinstance(handle, Line2D) for handle in handles)
        lines = [handle for handle in handles if isinstance(handle, Line2D)]
        assert {line.get_linestyle() for line in lines} == {"-", "--"}
        assert all(line.get_linewidth() <= 3.0 for line in lines)
    finally:
        plt.close(figure)


def test_histogram_grid_writes_counts_once_per_row_and_colors_datasets_by_material() -> None:
    """Only the first panel of each row says "Counts"; a dataset of one material is drawn in that material's color."""
    styles = plots.batch_styles({"Lentil": "lentil", "Mix": None})
    assert styles["Lentil"].color == semantics.material_color("lentil")
    rng = np.random.default_rng(0)
    data = {name: {key: rng.normal(size=20) for key in "abcd"} for name in styles}
    figure = statistics._hist_grid(data_by_dataset=data, active_datasets=list(styles), columns=list("abcd"), title="t", styles=styles)
    assert isinstance(figure, Figure)
    try:
        panels = figure.axes[:4]
        assert [axis.get_ylabel() for axis in panels] == ["Counts", "", "Counts", ""]
        assert to_rgb(panels[0].patches[0].get_facecolor()) == to_rgb(styles["Lentil"].color)
        assert len(figure.legends) == 1
    finally:
        plt.close(figure)


def test_dimensionless_representations_are_named_in_the_displayed_quantity() -> None:
    """A log10 permeability and a cross ratio name what they store; a physical field keeps its plain name; a published tensor shows it."""
    log10, cross = "dimensionless_log10_ratio_to_1_m2", "dimensionless_cross_component_ratio_to_geometric_mean"
    assert capabilities.representation_display_name("Kxx", log10) == "log₁₀(κₓₓ / 1 m²)"
    assert capabilities.representation_display_name("Kxy", cross) == "κₓᵧ / √(κₓₓ κᵧᵧ)"
    assert capabilities.representation_display_name("Kxx", "identity") == "κₓₓ"
    assert r"\log_{10}(\kappa_{xx}" in capabilities.representation_display_name("Kxx", log10, mathtext=True)
    assert r"\sqrt{\kappa_{xx}\,\kappa_{yy}}" in capabilities.representation_display_name("Kxy", cross, mathtext=True)
    x, y = np.meshgrid(np.arange(3.0), np.arange(2.0))
    value = PublishedTensorValue("static", "input", "Kxx", "1", log10, "spatial", -np.ones((2, 3)))
    figure = statistics.plot_published_tensor(value, x=x, y=y)
    try:
        assert r"\log_{10}" in figure.axes[0].get_title()
        assert figure.axes[1].get_ylabel().endswith(" [-]")
    finally:
        plt.close(figure)


def test_long_dataset_name_wraps_at_its_separators_without_losing_characters() -> None:
    """A name without spaces breaks after ``+`` and ``__``, keeps every character, and honours a line limit by widening the lines."""
    name = "lentil-train1-val1-pood1+chickpea-train1-val1-pood1+kidney-bean-ood2__ref-param-ext__sobol-seed-9300"
    lines = display_labels.wrapped_dataset_display_label(name, width=26).split("\n")
    assert "".join(lines) == name
    assert all(line.endswith(("+", "__")) for line in lines[:-1])
    assert len(lines) > 2
    two = display_labels.wrapped_dataset_display_label(name, width=26, max_lines=2).split("\n")
    assert len(two) == 2
    assert "".join(two) == name
    assert display_labels.wrapped_dataset_display_label("a short name", width=26) == "a short name"
    with pytest.raises(ValueError, match="line limit"):  # a limit below one line would widen forever
        display_labels.wrapped_dataset_display_label(name, width=26, max_lines=0)


def _steady_frame(offsets: Sequence[float], values: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Return one steady frame with a case per offset; every field is constant per case, at its ``values`` entry (else position + 1) + offset."""
    task = domain.tasks.registry.get_task("steady_flow")
    names = (*task.input_names, *task.output_names)
    ids = [f"case_{number:04d}" for number in range(1, len(offsets) + 1)]
    rows = [
        {
            **{name: np.full((2, 3), (values or {}).get(name, index + 1.0) + offset, dtype=np.float32) for index, name in enumerate(names)},
            "meta": {"case_id": case_id},
        }
        for offset, case_id in zip(offsets, ids, strict=True)
    ]
    loaded = {
        "task": task,
        "sample_ids": ids,
        "rows": rows,
        "available_case_count": len(ids),
        "total_discovered_case_count": len(ids),
        "failed_case_count": 0,
        "incomplete_case_count": 0,
        "invalid_case_count": 0,
        "exclusion_reasons": {},
        "case_accounting_scope": "test_owned_discovery",
        "analysis_representation": "steady_field_rows",
        "dataset_backend": "test_owned_backend",
        "generated_batch_identity": {"batch_manifest_identity_sha256": "f" * 64},
        "manifest_sha256": "d" * 64,
        "generation_root": Path("/synthetic"),
    }
    frame, _logs = eda_dataframe._steady_dataframe(loaded, batch_name="synthetic_steady", task=task)
    return frame


def _histogram_count(axis: Axes) -> int:
    """Return how many values the bars of one histogram panel count."""
    return round(sum(cast("Rectangle", patch).get_height() for patch in axis.patches))


def test_field_distribution_panels_write_their_power_of_ten_into_the_x_label_not_an_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A panel of values near 1e-13 shows plain tick labels and ``[10⁻¹³]`` in its x label; a panel of ordinary values keeps its label."""
    frame = _steady_frame([0.0], {"v": 3.0e-13, "Kxx": -9.0})
    name = "lentil-train1-val1-pood1+chickpea-train1-val1-pood1+kidney-bean-ood2__ref-param-ext__sobol-seed-9300"
    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", lambda plot, **_options: plot)
    # The monkeypatched viewer hands back the figure function itself.
    draw = cast("Callable[..., Figure]", statistics.plot_field_value_distributions(datasets={name: frame}, styles=plots.batch_styles({name: None})))
    figure = draw(1, datasets={name: frame})
    try:
        specs = [(axis, spec) for axis in figure.axes if axis.get_visible() and (spec := axis.get_subplotspec()) is not None]
        grid = [axis for axis, _spec in specs]
        panels = {spec.rowspan.start: axis for axis, spec in specs if spec.colspan.start == 0}
        assert len(panels) > 1
        exponents = [axis.get_xlabel() for axis in panels.values()]
        assert any("10⁻¹³" in label for label in exponents)
        assert all(axis.get_xlabel().endswith("]") for axis in grid)  # every panel states its unit, rescaled or not
        assert all(not axis.xaxis.get_offset_text().get_text() for axis in grid)
        assert "\n" in figure.legends[0].get_texts()[0].get_text()
    finally:
        plt.close(figure)


def test_field_distributions_draw_only_the_requested_case_prefix_and_name_the_cases_clipping_leaves_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """At 100 cases every panel names, in the series color, the 2 cases its 1st-99th percentile display omits; 8 cases after it draw just 8."""
    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", lambda plot, **_options: plot)
    frame = _steady_frame([float(offset) for offset in range(100)])
    styles = plots.batch_styles({"S": None})
    draw = cast("Callable[..., Figure]", statistics.plot_field_value_distributions(datasets={"S": frame}, styles=styles))
    for cases, drawn, notes in ((100, 98, ["2 of 100 cases left out"]), (8, 8, [])):
        figure = draw(cases, datasets={"S": frame})
        try:
            panels = [axis for axis in figure.axes if axis.get_subplotspec() is not None]
            assert len(panels) > 1
            for axis in panels:
                tails = [text for text in axis.texts if text.get_text().startswith("shown ")]  # not a row header
                assert _histogram_count(axis) == drawn
                assert [text.get_text().split("; ")[-1] for text in tails] == notes
                assert all(text.get_color() == styles["S"].color for text in tails)
        finally:
            plt.close(figure)


def test_a_clipped_range_note_reads_in_the_scaled_values_of_its_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values near 1.5e4 show in units of 10⁴ (x label); the note's shown range is written on that scale, inside the axis limits."""
    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", lambda plot, **_options: plot)
    frame = _steady_frame([1.0e4 + 100.0 * offset for offset in range(100)])
    draw = cast("Callable[..., Figure]", statistics.plot_field_value_distributions(datasets={"S": frame}, styles=plots.batch_styles({"S": None})))
    figure = draw(100, datasets={"S": frame})
    try:
        scaled = [axis for axis in figure.axes if "10⁴" in axis.get_xlabel()]
        assert scaled
        for axis in scaled:
            (note,) = [text.get_text() for text in axis.texts if text.get_text().startswith("shown ")]
            lower, upper = (float(value) for value in note.removeprefix("shown ").split(";")[0].split(" to "))
            assert axis.get_xlim()[0] <= lower < upper <= axis.get_xlim()[1]
    finally:
        plt.close(figure)


def test_parameter_histograms_draw_only_the_requested_case_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """After four cases, three draw the values of cases 1 to 3 only: generated parameters and material scalars alike."""
    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", lambda plot, **_options: plot)
    frame = pd.DataFrame(
        {
            "meta": [{"case_id": f"c{case}", "parameters": {"a": float(case)} if case else {}} for case in range(4)],
            "scalar_conditioning": [{"A_osw": float(case)} if case else {} for case in range(4)],
        }
    )
    styles = plots.batch_styles({"P": None})
    for build in (statistics.plot_meta_statistics, statistics.plot_meta_parameters):
        draw = cast("Callable[..., Figure]", build(datasets={"P": frame}, styles=styles))
        plt.close(draw(4, datasets={"P": frame}))
        figure = draw(3, datasets={"P": frame})
        try:
            assert _histogram_count(figure.axes[0]) == 2  # the first case holds no value
        finally:
            plt.close(figure)


def test_parameter_histogram_writes_its_power_of_ten_into_the_x_label_not_an_offset() -> None:
    """A parameter near 1e-8 m² shows plain tick labels and ``[10⁻⁸ m²]`` in its x label; an ordinary parameter keeps ``Value``."""
    styles = plots.batch_styles({"A": "lentil"})
    rng = np.random.default_rng(0)
    data = {"A": {"kappa": 1.0e-8 * rng.uniform(1.0, 2.0, size=20), "porosity": rng.uniform(0.3, 0.5, size=20)}}
    figure = statistics._hist_grid(
        data_by_dataset=data,
        active_datasets=["A"],
        columns=["kappa", "porosity"],
        title="t",
        styles=styles,
        units_by_column={"kappa": "m²", "porosity": "1"},
    )
    assert isinstance(figure, Figure)
    try:
        kappa, porosity = figure.axes[:2]
        assert kappa.get_xlabel() == "Value [10⁻⁸ m²]"
        assert porosity.get_xlabel() == "Value [-]"
        assert not kappa.xaxis.get_offset_text().get_text()
        assert max(float(tick.get_text()) for tick in kappa.get_xticklabels() if tick.get_text()) < 10.0  # shown in units of 1e-8
    finally:
        plt.close(figure)


def test_scalar_conditioning_panels_take_registered_symbols_and_keep_the_unit_only_on_the_x_axis() -> None:
    """Every scalar material input has a declared symbol; a panel title names it without unit, the x label holds unit and power of ten."""
    fields = statistics._SCALAR_MATERIAL_FIELDS
    assert all(field_labels.has_declared_field_metadata(field.name) for field in fields)
    oswin = [field_labels.field_label(name, mathtext=True, control=True) for name in ("A_osw", "B_osw", "C_osw")]
    assert oswin == ["$A$", "$B$", "$C$"]  # the symbols of docs/02_physical_model.md
    styles = plots.batch_styles({"A": "lentil"})
    data = {"A": {"r_surf_0": np.linspace(2.5e-4, 3.5e-4, 8), "cp_gr_dry": np.linspace(1400.0, 1600.0, 8)}}
    figure = statistics._hist_grid(
        data_by_dataset=data,
        active_datasets=["A"],
        columns=["r_surf_0", "cp_gr_dry"],
        title="t",
        styles=styles,
        units_by_column=statistics._SCALAR_MATERIAL_UNITS,
    )
    try:
        assert isinstance(figure, Figure)
        rate, heat = figure.axes[:2]
        assert rate.get_title() == field_labels.field_label("r_surf_0", mathtext=True)
        assert all("[" not in axis.get_title() for axis in (rate, heat))
        assert rate.get_xlabel() == "Value [10⁻⁴ 1/s]"
        assert heat.get_xlabel() == "Value [J/(kg·K)]"
    finally:
        plt.close(figure)


def test_sparse_histograms_keep_every_case_and_draw_a_kernel_density_only_from_enough_cases() -> None:
    """Eight cases keep their extremes and get no density curve; twenty cases get one."""
    sparse = np.linspace(0.0, 1.0, 8)
    np.testing.assert_array_equal(statistics._clip_for_plot(sparse), sparse)
    style = semantics.SeriesStyle("black", "-", None)
    for size, curves in ((8, 0), (statistics._KDE_MINIMUM_OBSERVATIONS, 1)):
        figure, axis = plt.subplots()
        try:
            statistics._draw_dataset_histogram(axis, np.linspace(0.0, 1.0, size) ** 2, style)
            assert len(axis.get_lines()) == curves
        finally:
            plt.close(figure)


def test_generated_case_parameters_without_any_parameter_return_a_one_line_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cases whose metadata carries no sampled parameter (a published runtime view) give a note instead of an empty figure."""
    frame = pd.DataFrame({"meta": [{"case_id": "case_0001"}]}, index=pd.Index(["case_0001"]))
    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", lambda plot, **_options: plot)
    draw = cast("Callable[..., object]", statistics.plot_meta_statistics(datasets={"P": frame}, styles=plots.batch_styles({"P": None})))
    note = draw(1, datasets={"P": frame})
    assert isinstance(note, str)
    assert note
