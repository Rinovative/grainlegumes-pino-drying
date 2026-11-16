"""Protect transient spectral reductions and nonuniform-time admission.

Responsibilities:
  - Check mixed-task channel selection and spectra from final valid stored states

Design principles:
  - Selected plots preserve temperature arrays and authoritative temporal support

This module does NOT:
  - Interpolate transient times or generate reference trajectories
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_spectral_analysis as spectra
from src.analysis.eda.plots import eda_plot_transient as transient_plots


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


def test_mixed_steady_transient_field_union_keeps_compatible_subsets() -> None:
    """Expose transient fields in mixed selection without zero-filled steady data."""
    transient_frame = _transient_frame()
    x = transient_frame.iloc[0]["static_fields"]["x"]
    y = transient_frame.iloc[0]["static_fields"]["y"]
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
        }
    )
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
    """A field that one dataset lacks does not shift the others' colors; batches of one material share its color and differ in line style."""
    full = _transient_frame()
    datasets = {"A": _without_phi(full), "B": full, "C": full.copy()}
    figure = spectra.plot_directional_spectral_case(
        datasets=datasets, styles=transient_plots.batch_styles(dict.fromkeys(datasets)), case_number=1, channels=("T", "phi")
    )
    try:
        power = [axis for axis in figure.axes if axis.get_ylabel().startswith("Mean spectral power")]
        colors = [{line.get_label(): line.get_color() for line in axis.get_lines()} for axis in power]
        assert colors[0]["B power"] == colors[1]["B power"]
        assert colors[0]["C power"] == colors[1]["C power"]
    finally:
        plt.close(figure)
    figure = spectra.plot_directional_spectral_case(
        datasets={"A": full, "B": full.copy()}, styles=transient_plots.batch_styles({"A": "lentil", "B": "lentil"}), case_number=1, channels=("T",)
    )
    try:
        lines = {line.get_label(): line for axis in figure.axes for line in axis.get_lines()}
        assert lines["A power"].get_color() == lines["B power"].get_color()
        assert lines["A power"].get_linestyle() != lines["B power"].get_linestyle()
        assert lines["A cumulative"].get_linestyle() == lines["A power"].get_linestyle()
        assert lines["A cumulative"].get_linewidth() < lines["A power"].get_linewidth()
    finally:
        plt.close(figure)


def test_a_dataset_is_drawn_in_the_style_it_is_given_whatever_the_selection() -> None:
    """The builder never resolves styles from the frames it gets: selecting batch B alone draws it in B's given dashed style."""
    full = _transient_frame()
    datasets = {"A": full, "B": full.copy()}
    styles = transient_plots.batch_styles({"A": "lentil", "B": "lentil"})
    figure = spectra.plot_directional_spectral_case(datasets=datasets, styles=styles, case_number=1, dataset_names=["B"], channels=("T",))
    try:
        lines = {line.get_label(): line.get_linestyle() for axis in figure.axes for line in axis.get_lines()}
        assert lines["B power"] == styles["B"].linestyle != "-"
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


def test_spectral_evolution_colorbars_each_carry_the_power_fraction_unit() -> None:
    """Every colour bar of the position-resolved spectra names its quantity and unit."""
    frame = _transient_frame()
    figure = spectra.plot_vertical_spectral_case(datasets={"A": frame, "B": frame.copy()}, case_number=1, channels=("T",))
    try:
        labels = [axis.get_ylabel() for axis in figure.axes if "power fraction" in axis.get_ylabel()]
        assert labels == ["Row-normalized power fraction [-]"] * 2
    finally:
        plt.close(figure)
