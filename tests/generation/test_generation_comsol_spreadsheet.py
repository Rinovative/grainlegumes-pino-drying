# ruff: noqa: PLR2004, S101, TC003
"""COMSOL Spreadsheet parsing and mapping regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.generation.contracts import generation_contracts_comsol_spreadsheet as spreadsheet

_TEMPORAL_UNITS = {
    "t": "h",
    "x": "m",
    "y": "m",
    "T": "K",
    "mt.phi": "1",
    "w_surf": "kg/m^3",
    "w_int": "kg/m^3",
}


def _wide_header(times: tuple[float, ...]) -> list[str]:
    """Return compact native COMSOL temporal descriptors."""
    return [f"{source} ({unit}) @ t={state_time:g}" for state_time in times for source, unit in _TEMPORAL_UNITS.items()]


_STEADY_RAW_HEADER = (
    "x (m)",
    "y (m)",
    "Kxx (m^2)",
    "Kxy (m^2)",
    "Kyy (m^2)",
    "eps_bed",
    "p_in_bc (Pa)",
    "p (Pa)",
    "u (m/s)",
    "v (m/s)",
)
_STEADY_CANONICAL_HEADER = (
    "x",
    "y",
    "Kxx",
    "Kxy",
    "Kyy",
    "eps_bed",
    "p_in_bc",
    "p",
    "u",
    "v",
)
_STEADY_UNITS: dict[str, str] = dict(
    zip(
        _STEADY_CANONICAL_HEADER,
        ("m", "m", "m^2", "m^2", "m^2", "1", "Pa", "Pa", "m/s", "m/s"),
        strict=True,
    )
)


def test_real_steady_spreadsheet_retains_every_numeric_row(tmp_path: Path) -> None:
    """Reproduce the real percent-prefixed header and row-count failure."""
    path = tmp_path / "stationary_fields.csv"
    path.write_text(
        """% Model,model.mph
% Version,COMSOL 6.4.0.293
% Date,"Aug 12 2026, 05:58"
% Dimension,2
% Nodes,3
% Expressions,10
% Description,"x-coordinate,y-coordinate,..."
% Length unit,m
% x (m),y (m),Kxx (m^2),Kxy (m^2),Kyy (m^2),eps_bed,p_in_bc (Pa),p (Pa),u (m/s),v (m/s)
0,0,1e-9,0,2e-9,0.35,750,0,0.5,0
0.003,0,1.1e-9,0,2.1e-9,0.36,751,1,0.6,0
0.006,0,1.2e-9,0,2.2e-9,0.37,752,2,0.7,0
""",
        encoding="utf-8",
    )

    table = spreadsheet.read_comsol_spreadsheet(
        path,
        delimiter=",",
        expected_units=_STEADY_UNITS,
    )

    assert table.raw_header == _STEADY_RAW_HEADER
    assert table.canonical_header == _STEADY_CANONICAL_HEADER
    assert table.shape == (3, 10)
    assert table.metadata["Nodes"] == 3
    assert table.metadata["Expressions"] == 10
    assert table.values is not None
    np.testing.assert_array_equal(
        table.values[0],
        np.asarray([0, 0, 1e-9, 0, 2e-9, 0.35, 750, 0, 0.5, 0]),
    )
    np.testing.assert_array_equal(
        table.values[-1],
        np.asarray([0.006, 0, 1.2e-9, 0, 2.2e-9, 0.37, 752, 2, 0.7, 0]),
    )


def test_unit_normalization_preserves_expression_parentheses_and_conflicts() -> None:
    """Strip only exact declared units and preserve dimensionless expressions."""
    raw = (
        "comp1.maxop_bed(comp1.T) (K)",
        "comp1.minop_bed(comp1.mt.phi)",
        "mt.phi (1)",
        "comp1.X_wb_bulk",
        "x (cm)",
    )
    canonical = spreadsheet.canonicalize_header(
        raw,
        expected_units={
            "comp1.maxop_bed(comp1.T)": "K",
            "comp1.minop_bed(comp1.mt.phi)": "1",
            "mt.phi": "1",
            "comp1.X_wb_bulk": "1",
            "x": "m",
        },
    )
    assert canonical == (
        "comp1.maxop_bed(comp1.T)",
        "comp1.minop_bed(comp1.mt.phi)",
        "mt.phi",
        "comp1.X_wb_bulk",
        "x (cm)",
    )


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ("% Nodes,2\n% Expressions,1\n", "Nodes metadata"),
        ("% Nodes,3\n% Expressions,2\n", "Expressions metadata"),
    ],
)
def test_metadata_integrity_mismatches_fail_clearly(
    tmp_path: Path,
    metadata: str,
    match: str,
) -> None:
    """Reject COMSOL row-count and width metadata that contradict the table."""
    path = tmp_path / "invalid.csv"
    path.write_text(
        f"% Model,model.mph\n{metadata}% x (m)\n0\n1\n2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        spreadsheet.read_comsol_spreadsheet(
            path,
            delimiter=",",
            expected_units={"x": "m"},
        )


def test_metadata_without_a_column_header_fails_clearly(tmp_path: Path) -> None:
    """Do not reinterpret a final metadata record as the column header."""
    path = tmp_path / "missing_header.csv"
    path.write_text(
        """% Model,model.mph
% Description,"x-coordinate,y-coordinate"
0,1
2,3
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Could not identify"):
        spreadsheet.read_comsol_spreadsheet(path, delimiter=",")


def test_comment_after_numeric_data_fails_clearly(tmp_path: Path) -> None:
    """Reject an unexpected comment after the numeric section begins."""
    path = tmp_path / "ambiguous.csv"
    path.write_text(
        """% Model,model.mph
% x (m)
0
% unexpected
1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unexpected comment record"):
        spreadsheet.read_comsol_spreadsheet(
            path,
            delimiter=",",
            expected_units={"x": "m"},
        )


def test_temporal_descriptor_preserves_expression_and_header_precision() -> None:
    """Parse only the trailing declared unit and temporal suffix."""
    descriptor = spreadsheet.parse_temporal_column_descriptor(
        "comp1.some_expression(a(b)) (kg/s) @ t=50.601",
        expected_units={"comp1.some_expression(a(b))": "kg/s"},
    )
    assert descriptor.source == "comp1.some_expression(a(b))"
    assert descriptor.unit == "kg/s"
    assert descriptor.state_time == 50.601
    assert descriptor.state_time_text_atol == pytest.approx(0.0005)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (_wide_header((0.0,))[:-1], "incomplete"),
        ([*_wide_header((0.0,)), "T (K) @ t=0"], "repeats logical"),
        (_wide_header((1.0, 0.0)), "strictly increasing"),
        (["T (degC) @ t=0"], "exact declared unit"),
        (["T (K) @ t=broken"], "Malformed"),
        (
            [name.replace("w_int (kg/m^3) @ t=1", "w_int (kg/m^3) @ t=1.5") for name in _wide_header((1.0,))],
            "incomplete",
        ),
    ],
)
def test_temporal_grouping_rejects_malformed_or_incomplete_states(
    header: list[str],
    match: str,
) -> None:
    """Fail closed for every malformed temporal grouping branch."""
    with pytest.raises(ValueError, match=match):
        spreadsheet.group_temporal_columns(header, expected_units=_TEMPORAL_UNITS)
