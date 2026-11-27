"""Protect transient spectral reductions and nonuniform-time admission.

Responsibilities:
  - Check mixed-task channel selection and spectra from final valid stored states
  - Check the spectral display: power and cumulative energy in adjacent panels of their own per direction, squared
    display units, row headers without unit, one colour-bar label per row of maps, and the typeset $k_x$ and $k_y$ of
    the directional and both evolution frequency labels
  - Check that a stored ratio header is one formula of symbols with the name outside it in every label style, and
    that a summary counts only the cases of the Datasets it draws

Design principles:
  - Selected plots preserve temperature arrays and authoritative temporal support

This module does NOT:
  - Interpolate transient times or generate reference trajectories
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_spectral_analysis as spectra
from src.analysis.eda.plots import eda_plot_transient as transient_plots
from src.analysis.presentation import analysis_field_labels as field_labels


def _transient_frame() -> pd.DataFrame:
    shape = (4, 4)
    x, y = np.meshgrid(np.arange(shape[1], dtype=float), np.arange(shape[0], dtype=float))
    first = np.zeros(shape)
    final_valid = np.add.outer(np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float))
    invalid_tail = np.full(shape, 99.0)
    row = {
        "state_trajectories": {
            "T": np.stack((first, final_valid, invalid_tail)),
            "phi": np.stack((first + 1.0, final_valid + 2.0, invalid_tail + 3.0)),
        },
        "static_fields": {
            "x": x,
            "y": y,
            "u": x + 1.0,
            "v": y + 2.0,
            "p": x - y,
            "eps_bed": np.full(shape, 0.4),
            "X_0_db_field": np.full(shape, 0.2),
            "rho_bu_dry": np.full(shape, 650.0),
        },
        "time": {
            "regular_state_hours": np.array((0.0, 1.0, 2.0)),
            "valid_state_mask": np.array((True, True, False)),
        },
    }
    frame = pd.DataFrame((row,), index=pd.Index(("case_0001",), name="sample_id"))
    frame.attrs.update(
        {
            "task_id": "transient_drying",
            "task_contract_digest": "synthetic-transient-contract",
            "field_names": ("T", "phi"),
            "field_categories": {
                "dynamic_state": ("T", "phi"),
                "static_spatial": (
                    "x",
                    "y",
                    "u",
                    "v",
                    "p",
                    "eps_bed",
                    "X_0_db_field",
                    "rho_bu_dry",
                ),
            },
            "field_units": {
                "T": "K",
                "phi": "1",
                "x": "m",
                "y": "m",
                "u": "m/s",
                "v": "m/s",
                "p": "Pa",
                "eps_bed": "1",
                "X_0_db_field": "kg/kg",
                "rho_bu_dry": "kg/m^3",
            },
            "field_representations": {"T": "absolute_physical_state", "phi": "absolute_physical_state"},
            "source_manifest_sha256": "a" * 64,
        }
    )
    return frame


def _steady_frame() -> pd.DataFrame:
    """Return one steady case on the grid of ``_transient_frame``, every field in its identity representation."""
    static = _transient_frame().iloc[0]["static_fields"]
    x, y = static["x"], static["y"]
    steady = pd.DataFrame(
        (
            {
                "x": x,
                "y": y,
                "Kxx": np.full_like(x, 1.0e-8),
                "eps_bed": np.full_like(x, 0.4),
                "p": x - y,
                "u": x + 1.0,
                "v": y + 2.0,
                "U": np.hypot(x + 1.0, y + 2.0),
            },
        ),
        index=pd.Index(("case_0001",), name="sample_id"),
    )
    steady.attrs.update(
        {
            "task_id": "steady_flow",
            "field_names": ("x", "y", "Kxx", "eps_bed", "p", "u", "v", "U"),
            "field_roles": {
                "x": "coordinate",
                "y": "coordinate",
                "Kxx": "conditioning",
                "eps_bed": "conditioning",
                "p": "state",
                "u": "state",
                "v": "state",
                "U": "derived_speed",
            },
            "field_units": {
                "x": "m",
                "y": "m",
                "Kxx": "m^2",
                "eps_bed": "1",
                "p": "Pa",
                "u": "m/s",
                "v": "m/s",
                "U": "m/s",
            },
            "field_representations": dict.fromkeys(
                ("Kxx", "eps_bed", "p", "u", "v", "U"),
                "identity",
            ),
            "task_contract_digest": "synthetic-steady-contract",
        }
    )
    return steady


def _published_permeabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a steady ``frame`` with Kxy that stores the permeabilities like a published Dataset: log₁₀(κₓₓ / 1 m²) and κₓᵧ / √(κₓₓ κᵧᵧ)."""
    encoded = frame.copy()
    encoded["Kxy"] = pd.Series([0.1 * np.asarray(frame.iloc[0]["x"])], index=frame.index)
    encoded.attrs = {
        **frame.attrs,
        "field_names": (*frame.attrs["field_names"], "Kxy"),
        "field_roles": {**frame.attrs["field_roles"], "Kxy": "conditioning"},
        "field_units": {**frame.attrs["field_units"], "Kxx": "1", "Kxy": "1"},
        "field_representations": {
            **frame.attrs["field_representations"],
            "Kxx": "dimensionless_log10_ratio_to_1_m2",
            "Kxy": "dimensionless_cross_component_ratio_to_geometric_mean",
        },
    }
    return encoded


def _cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Return ``count`` copies of the one case of ``frame`` as its cases 1 to ``count``."""
    index = pd.Index([f"case_{number:04d}" for number in range(1, count + 1)], name="sample_id")
    cases = pd.DataFrame([frame.iloc[0].to_dict()] * count, index=index)
    cases.attrs = dict(frame.attrs)
    return cases


def test_mixed_steady_transient_field_union_keeps_compatible_subsets() -> None:
    """Expose transient fields in mixed selection without zero-filled steady data."""
    transient_frame = _transient_frame()
    steady = _steady_frame()
    selected = {"Steady": steady, "Drying": transient_frame}
    resolution = capabilities.resolve_fields(selected, view="spectral")
    assert resolution.fields == (
        "Kxx",
        "eps_bed",
        "p",
        "u",
        "v",
        "U",
        "X_0_db_field",
        "rho_bu_dry",
        "T",
        "phi",
    )
    assert resolution.datasets_by_field["T"] == ("Drying",)
    assert resolution.omitted_by_field["T"] == ("Steady",)
    assert tuple(capabilities.compatible_frames(selected, resolution, "T")) == ("Drying",)


def test_transient_spectra_use_final_valid_stored_state_and_shared_channels() -> None:
    frame = _transient_frame()

    expected_order = (
        "eps_bed",
        "p",
        "u",
        "v",
        "U",
        "X_0_db_field",
        "rho_bu_dry",
        "T",
        "phi",
    )
    for view in ("field_statistics", "spatial_map", "spectral", "state_snapshot"):
        assert capabilities.resolve_fields({"Drying": frame}, view=view).fields == expected_order
    assert tuple(capabilities.field_group(frame, field) for field in expected_order) == (
        "airflow_input",
        "airflow_output",
        "airflow_output",
        "airflow_output",
        "airflow_output",
        "transient_input",
        "transient_input",
        "transient_output",
        "transient_output",
    )
    row = frame.iloc[0]
    np.testing.assert_allclose(
        spectra._field_values(frame, row, "T"),
        row["state_trajectories"]["T"][1] - 273.15,  # the final valid state on its display scale; the mean removal drops the offset
    )
    np.testing.assert_allclose(
        spectra._field_values(frame, row, "U"),
        np.hypot(row["static_fields"]["u"], row["static_fields"]["v"]),
    )


def test_position_resolved_spectral_plot_builds_without_mutating_temperature() -> None:
    """Build 2-3 without drawing and preserve its authoritative stored field."""
    frame = _transient_frame()
    source = np.array(frame.iloc[0]["state_trajectories"]["T"], copy=True)

    figure = spectra.plot_vertical_spectral_case(
        datasets={"Drying · Lentil · ID": frame},
        case_number=1,
        channels=("T",),
    )
    try:
        np.testing.assert_array_equal(
            frame.iloc[0]["state_trajectories"]["T"],
            source,
        )
    finally:
        plt.close(figure)


def test_directional_spectra_label_each_axis_once_per_row_and_keep_one_color_per_dataset() -> None:
    """Power and cumulative y labels appear once per row; both line styles of a dataset share its color; one legend sits at the right."""
    label = "Drying · Lentil · ID"
    figure = spectra.plot_directional_spectral_case(
        datasets={label: _transient_frame()}, styles=transient_plots.batch_styles({label: None}), case_number=1, channels=("T", "phi")
    )
    try:
        labels = [axis.get_ylabel() for axis in figure.axes]
        assert sum(label.startswith("Mean spectral power") for label in labels) == 2
        assert labels.count("Cumulative energy [-]") == 2
        lines = [line for axis in figure.axes for line in axis.get_lines()]
        assert {line.get_color() for line in lines} == {lines[0].get_color()}
        assert {line.get_linestyle() for line in lines} == {"-"}
        assert len({line.get_linewidth() for line in lines}) == 2  # power and the thinner cumulative line
        assert len(figure.legends) == 1
    finally:
        plt.close(figure)


def test_directional_frequency_labels_typeset_k_x_and_k_y_in_the_spectra_and_both_evolution_orientations() -> None:
    """The directional spectra and either spectral-evolution orientation write their frequency as the mathtext $k_y$ or $k_x$."""
    label = "Drying · Lentil · ID"
    frame = _transient_frame()
    figure = spectra.plot_directional_spectral_case(
        datasets={label: frame}, styles=transient_plots.batch_styles({label: None}), case_number=1, channels=("T",)
    )
    try:
        labels = {axis.get_xlabel() for axis in figure.axes if axis.get_xlabel()}
        assert len(labels) == 2  # one frequency label per direction, under its power and cumulative panel
        assert {symbol for text in labels for symbol in ("$k_y$", "$k_x$") if symbol in text} == {"$k_y$", "$k_x$"}
    finally:
        plt.close(figure)
    orientations: tuple[tuple[spectra.SpectralEvolutionOrientation, str], ...] = (
        ("cross_stream_along_flow", "$k_x$"),
        ("flow_across_cross_stream", "$k_y$"),
    )
    for orientation, symbol in orientations:
        figure = spectra.plot_vertical_spectral_case(datasets={label: frame}, case_number=1, channels=("T",), orientation=orientation)
        try:
            labels = {axis.get_xlabel() for axis in figure.axes if axis.get_xlabel()}
            assert len(labels) == 1
            assert symbol in labels.pop()
        finally:
            plt.close(figure)


def test_spectral_power_squares_the_displayed_unit_and_row_headers_name_the_quantity_without_unit() -> None:
    """Power reads (°C)² for a temperature and - (never (-)²) for a dimensionless field; the bold row headers carry no unit."""
    label = "Drying · Lentil · ID"
    figure = spectra.plot_isotropic_spectral_case(
        datasets={label: _transient_frame()}, styles=transient_plots.batch_styles({label: None}), case_number=1, channels=("T", "eps_bed")
    )
    try:
        labels = [axis.get_ylabel() for axis in figure.axes if axis.get_ylabel().startswith("Mean spectral power")]
        assert sorted(labels) == ["Mean spectral power [(°C)²]", "Mean spectral power [-]"]
        headers = [text.get_text() for axis in figure.axes for text in axis.texts if text.get_fontweight() == "bold"]
        assert len(headers) == 2
        assert not any("[" in header for header in headers)
    finally:
        plt.close(figure)


def test_a_stored_ratio_header_is_one_formula_of_symbols_with_the_name_outside_it_in_every_label_style() -> None:
    """
    The row headers of the published permeability ratios (``field_display_name``) render as one math formula of symbols in every label style.

    The name and name-and-symbol styles write the field name before the formula, never inside it.
    """
    frame = _published_permeabilities(_steady_frame())
    probe = plt.figure()
    try:
        for style in ("symbol", "name", "name_symbol"):
            field_labels.set_label_style(style)
            for field in ("Kxx", "Kxy"):
                name = field_labels.field_display_metadata(field).name
                header = capabilities.field_display_name(frame, field, mathtext=True)
                probe.text(0.0, 0.5, header)
                prefix, formula, rest = header.split("$")  # exactly one formula
                assert prefix == ("" if style == "symbol" else f"{name} ")
                assert rest == ""
                assert field_labels.field_label(field, mathtext=True, control=True).strip("$") in formula
                assert name not in formula
        probe.canvas.draw()  # renders the header of every style
    finally:
        field_labels.set_label_style("symbol")
        plt.close(probe)


def test_a_summary_counts_the_cases_of_the_datasets_it_draws_and_names_each_when_their_draws_differ() -> None:
    """
    A T-only spectrum over two transient Datasets and a steady one without T counts only the drawn transient cases.

    Their draws differ, so the title names each Dataset with its count; the note gives the drawn of the pre-load membership.
    """
    two, three = _cases(_transient_frame(), 2), _cases(_transient_frame(), 3)
    two.attrs["available_case_count"] = 5  # the membership known before the load, of which the frame holds the first two
    datasets = {"A": two, "B": three, "Steady": _steady_frame()}
    figure = spectra.plot_isotropic_spectral_summary(datasets=datasets, styles=transient_plots.batch_styles(dict.fromkeys(datasets)), channels=("T",))
    try:
        texts = [" ".join(text.get_text().split()) for text in figure.findobj(Text)]
        assert "Isotropic spectra — A: 2 cases, B: 3 cases — final valid stored state" in texts
        assert any(text.startswith("First cases drawn of the eligible ones — A: 2 of 5.") for text in texts)
    finally:
        plt.close(figure)


def _without_phi(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` that stores only T."""
    row = dict(frame.iloc[0])
    row["state_trajectories"] = {"T": row["state_trajectories"]["T"]}
    reduced = pd.DataFrame((row,), index=frame.index)
    reduced.attrs.update(frame.attrs)
    reduced.attrs["field_names"] = ("T",)
    reduced.attrs["field_categories"] = {**frame.attrs["field_categories"], "dynamic_state": ("T",)}
    return reduced


def test_directional_colors_are_resolved_once_per_figure_and_batches_differ_in_line_style() -> None:
    """
    A field that one dataset lacks does not shift the others' colors; batches of one material share its color and differ in line style.

    Each direction's power and cumulative energy are two adjacent panels on one frequency range, checked pair by pair.
    """
    full = _transient_frame()
    datasets = {"A": _without_phi(full), "B": full, "C": full.copy()}
    figure = spectra.plot_directional_spectral_case(
        datasets=datasets, styles=transient_plots.batch_styles(dict.fromkeys(datasets)), case_number=1, channels=("T", "phi")
    )
    try:
        power = [axis for axis in figure.axes if axis.get_ylabel().startswith("Mean spectral power")]
        colors = [{line.get_label(): line.get_color() for line in axis.get_lines()} for axis in power]
        assert colors[0]["B"] == colors[1]["B"]
        assert colors[0]["C"] == colors[1]["C"]
    finally:
        plt.close(figure)
    styles = transient_plots.batch_styles({"A": "lentil", "B": "lentil"}, roles={"A": "id", "B": "parameter_ood"})
    figure = spectra.plot_directional_spectral_case(datasets={"A": full, "B": full.copy()}, styles=styles, case_number=1, channels=("T",))
    try:
        assert len(figure.axes) == 4  # one row: the power and cumulative panel of the flow direction y, then of the cross-stream x
        for power_axis, cumulative_axis in (figure.axes[0:2], figure.axes[2:4]):
            columns = [cast("Any", axis.get_subplotspec()).colspan.start for axis in (power_axis, cumulative_axis)]
            assert columns[1] == columns[0] + 1  # distinct adjacent panels, never twins
            assert power_axis.get_xlim() == cumulative_axis.get_xlim()
            assert cumulative_axis.get_title() == power_axis.get_title().replace(" power", " cumulative energy")  # one direction
            power = {line.get_label(): line for line in power_axis.get_lines()}
            assert power["A"].get_color() == power["B"].get_color()
            assert power["A"].get_linestyle() != power["B"].get_linestyle()
            cumulative = cumulative_axis.get_lines()  # A, then B, in the order drawn
            assert [line.get_linestyle() for line in cumulative] == [power["A"].get_linestyle(), power["B"].get_linestyle()]
            assert all(line.get_linewidth() < power["A"].get_linewidth() for line in cumulative)
    finally:
        plt.close(figure)


def test_a_dataset_is_drawn_in_the_style_it_is_given_whatever_the_selection() -> None:
    """The builder never resolves styles from the frames it gets: selecting batch B alone draws it in B's given dashed style."""
    full = _transient_frame()
    datasets = {"A": full, "B": full.copy()}
    styles = transient_plots.batch_styles({"A": "lentil", "B": "lentil"}, roles={"B": "parameter_ood"})
    figure = spectra.plot_directional_spectral_case(datasets=datasets, styles=styles, case_number=1, dataset_names=["B"], channels=("T",))
    try:
        lines = {line.get_label(): line.get_linestyle() for axis in figure.axes for line in axis.get_lines()}
        assert lines["B"] == styles["B"].linestyle != "-"
    finally:
        plt.close(figure)


def test_spectral_evolution_row_label_skips_an_unavailable_panel() -> None:
    """A panel of a missing field is axis-off and carries no y label; the row's label stands on the first panel that shows data."""
    full = _transient_frame()
    figure = spectra.plot_vertical_spectral_case(datasets={"A": _without_phi(full), "B": full}, case_number=1, channels=("T", "phi"))
    try:
        assert all(axis.get_ylabel() == "" for axis in figure.axes if not axis.axison)
        row_labels = [axis for axis in figure.axes if axis.axison and axis.get_ylabel() and "power fraction" not in axis.get_ylabel()]
        assert len(row_labels) == 2  # one per field row, though phi is missing in the first column
    finally:
        plt.close(figure)


def test_spectral_evolution_column_title_and_x_label_stand_on_available_panels() -> None:
    """A column whose lowest cell is unavailable still carries its x label, and each column its title, on available panels."""
    full = _transient_frame()
    figure = spectra.plot_vertical_spectral_case(datasets={"A": _without_phi(full), "B": full}, case_number=1, channels=("T", "phi"))
    try:
        shown = [axis for axis in figure.axes if axis.axison]
        assert sum(bool(axis.get_xlabel()) for axis in shown) == 2  # one per dataset column, A's on its only panel
        assert sum(bool(axis.get_title()) for axis in shown) == 2
    finally:
        plt.close(figure)


def test_spectral_evolution_keeps_a_colorbar_per_map_and_names_the_rows_shared_quantity_once_at_the_right() -> None:
    """Every map of the position-resolved spectra keeps its colour bar; the bars of a row share quantity and unit, which the rightmost names."""
    frame = _transient_frame()
    figure = spectra.plot_vertical_spectral_case(datasets={"A": frame, "B": frame.copy()}, case_number=1, channels=("T",))
    try:
        bars = [axis for axis in figure.axes if axis.get_label() == "<colorbar>"]
        assert [axis.get_ylabel() for axis in bars] == ["", "Row-normalized power fraction [-]"]
    finally:
        plt.close(figure)
