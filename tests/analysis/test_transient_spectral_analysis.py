# ruff: noqa: S101, D100, D103, SLF001
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_spectral_analysis as spectra


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
        row["state_trajectories"]["T"][1],
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
