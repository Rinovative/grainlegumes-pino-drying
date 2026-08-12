# ruff: noqa: PLR2004, S101, SLF001, TC003
"""COMSOL Spreadsheet parsing and mapping regressions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from src.generation.contracts import generation_contracts_comsol_spreadsheet as spreadsheet
from src.generation.publication import generation_publication_storage as storage
from src.generation.runtime import generation_runtime_mapping_probe as mapping_probe

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


def test_hdf5_admission_uses_canonical_unit_aware_headers(tmp_path: Path) -> None:
    """Use the shared parser for mapped numeric values without row loss."""
    path = tmp_path / "fields.csv"
    path.write_text(
        """% Model,model.mph
% Nodes,2
% Expressions,2
% x (m),p (Pa)
0,10
0.5,11
""",
        encoding="utf-8",
    )
    mapped = storage._mapped_table(
        [path],
        {
            "role": "steady_flow_fields",
            "units": {"x": "m", "p": "Pa"},
            "columns": {"x": "x", "p": "p"},
            "delimiter": ",",
        },
    )
    np.testing.assert_array_equal(mapped["x"], np.asarray([0.0, 0.5]))
    np.testing.assert_array_equal(mapped["p"], np.asarray([10.0, 11.0]))


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


def test_all_export_roles_use_unit_aware_exact_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover long Data metadata and shorter Table metadata for every role."""
    monkeypatch.setattr(mapping_probe.common.paths, "get_project_root", lambda: tmp_path)
    profile = {
        "schema_kind": "generation_profile",
        "schema_version": 2,
        "simulation_profile": "transient_drying",
        "steady_flow_conditioning": None,
        "exports": [
            {
                "role": "steady_flow_fields",
                "temporal_kind": "stationary",
                "source": "steady.csv",
                "delimiter": ",",
                "columns": {"x": "x", "p": "p"},
            },
            {
                "role": "transient_fields",
                "temporal_kind": "regular_time_series",
                "source": "transient.csv",
                "delimiter": ",",
                "columns": {"t": "t", "T": "T", "phi": "mt.phi"},
            },
            {
                "role": "global_time_series",
                "temporal_kind": "regular_time_series",
                "source": "globals.csv",
                "delimiter": ",",
                "columns": {"t": "t", "X_wb_bulk": "X_wb_bulk", "T_out_mean": "T_out_mean"},
            },
            {
                "role": "final_status",
                "temporal_kind": "final_status",
                "source": "final.csv",
                "delimiter": ",",
                "columns": {
                    "t_final": "t_final",
                    "T_max_final": "T_max_final",
                    "phi_min_final": "phi_min_final",
                },
            },
        ],
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    exports = {
        "steady.csv": """% Model,model.mph
% Dimension,2
% Nodes,1
% Expressions,2
% Length unit,m
% x (m),p (Pa)
0,0
""",
        "transient.csv": """% Model,model.mph
% Dimension,2
% Nodes,1
% Expressions,3
% t (h) @ t=0,T (K) @ t=0,mt.phi (1) @ t=0
0,296,0.5
""",
        "globals.csv": """% Model,model.mph
% Expressions,4
% Time,t (h),X_wb_bulk (1),T_out_mean (K)
0,0,0.2,296
""",
        "final.csv": """% Model,model.mph
% Time,t_final (h),T_max_final (K),phi_min_final (1)
0,1,295,0.4
""",
    }
    inventory = []
    for name, content in exports.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        inventory.append({"relative_path": name, "table": mapping_probe._table_observation(path)})

    comparison = mapping_probe._mapping_comparison(
        mapping_probe._profile_mapping(profile_path),
        inventory,
        profile_path=profile_path,
    )

    assert comparison["required_corrections"] == []
    assert comparison["required_missing_exports"] == []
    assert (
        mapping_probe._mapping_probe_status(
            comparison,
            exit_code=0,
            timed_out=False,
            start_error=None,
        )
        == "mapping_observation_complete"
    )
    assert [observation["role"] for observation in comparison["observations"]] == [
        "steady_flow_fields",
        "transient_fields",
        "global_time_series",
        "final_status",
    ]
    for observation in comparison["observations"]:
        assert observation["raw_header"]
        assert observation["canonical_header"] == observation["observed_header"]
        assert observation["parsed_shape"][0] == 1


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


def _write_wide_fixture(
    path: Path,
    *,
    numeric_times: tuple[float, ...] = (0.0, 1.0, 1.5),
    shifted_state: int | None = None,
) -> None:
    """Write a three-node native-wide table with one exact-stop state."""
    times = (0.0, 1.0, 1.5)
    header = _wide_header(times)
    records = [
        "% Model,model.mph",
        "% Dimension,2",
        "% Nodes,3",
        f"% Expressions,{len(header)}",
        "% " + ",".join(header),
    ]
    for x_value in (0.0, 1.0, 2.0):
        row: list[float] = []
        for state_index, (header_time, numeric_time) in enumerate(zip(times, numeric_times, strict=True)):
            state_x = x_value + (0.01 if state_index == shifted_state else 0.0)
            row.extend(
                (
                    numeric_time,
                    state_x,
                    0.0,
                    300.0 + header_time + x_value,
                    0.4 + 0.01 * header_time,
                    10.0 - header_time - x_value,
                    20.0 - header_time - x_value,
                )
            )
        records.append(",".join(format(value, ".17g") for value in row))
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _wide_storage_config() -> Any:
    """Return the minimal resolved contract consumed by wide admission."""
    return SimpleNamespace(
        scientific_values={
            "time": {
                "start": 0.0,
                "interval": 1.0,
                "stop": 2.0,
                "regular_times": [0.0, 1.0, 2.0],
            },
            "output_contract": {
                "exports": [
                    {
                        "role": "transient_fields",
                        "delimiter": ",",
                        "columns": {
                            "t": "t",
                            "x": "x",
                            "y": "y",
                            "T": "T",
                            "phi": "mt.phi",
                            "w_surf": "w_surf",
                            "w_int": "w_int",
                        },
                        "units": {
                            "t": "h",
                            "x": "m",
                            "y": "m",
                            "T": "K",
                            "phi": "1",
                            "w_surf": "kg/m^3",
                            "w_int": "kg/m^3",
                        },
                    }
                ]
            },
        }
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


def test_wide_transient_conversion_preserves_regular_and_exact_stop_states(tmp_path: Path) -> None:
    """Stream native wide rows directly into canonical ordered state arrays."""
    path = tmp_path / "transient.csv"
    _write_wide_fixture(path, numeric_times=(0.0, 1.0, 1.5004))
    regular_time, regular, exact_time, exact = storage._transient_fields(
        _wide_storage_config(),
        [SimpleNamespace(role="transient_fields", source_path=path)],
        x_axis=np.asarray([0.0, 1.0, 2.0]),
        y_axis=np.asarray([0.0]),
    )
    np.testing.assert_array_equal(regular_time, np.asarray([0.0, 1.0]))
    assert regular.shape == (2, 4, 1, 3)
    np.testing.assert_array_equal(regular[0, 0, 0], np.asarray([300.0, 301.0, 302.0]))
    np.testing.assert_array_equal(regular[1, 1, 0], np.full(3, 0.4 + 0.01))
    assert exact_time == 1.5004
    assert exact is not None
    np.testing.assert_array_equal(exact[2, 0], np.asarray([8.5, 7.5, 6.5]))


@pytest.mark.parametrize(
    ("numeric_times", "shifted_state", "match"),
    [((0.0, 1.0, 1.4), None, "numeric t disagrees"), ((0.0, 1.0, 1.5), 1, "coordinates disagree")],
)
def test_wide_transient_conversion_rejects_numeric_time_or_grid_disagreement(
    tmp_path: Path,
    numeric_times: tuple[float, ...],
    shifted_state: int | None,
    match: str,
) -> None:
    """Require independent numeric-time and repeated-grid evidence."""
    path = tmp_path / "invalid.csv"
    _write_wide_fixture(path, numeric_times=numeric_times, shifted_state=shifted_state)
    with pytest.raises(ValueError, match=match):
        storage._transient_fields(
            _wide_storage_config(),
            [SimpleNamespace(role="transient_fields", source_path=path)],
            x_axis=np.asarray([0.0, 1.0, 2.0]),
            y_axis=np.asarray([0.0]),
        )


@pytest.mark.parametrize(
    "times",
    [np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.5, 1.0, 1.5])],
)
def test_time_classification_rejects_nonfinal_or_multiple_irregular_states(times: np.ndarray) -> None:
    """Keep irregular solver stops final, singular, and diagnostic-only."""
    with pytest.raises(ValueError, match="irregular"):
        storage._classify_transient_times(
            times,
            {"start": 0.0, "interval": 1.0, "stop": 2.0, "regular_times": [0.0, 1.0, 2.0]},
        )


def test_unit_interval_admission_allows_only_binary64_roundoff() -> None:
    """Admit COMSOL unit-fraction residue without hiding real violations."""
    values = np.asarray([np.nextafter(0.0, -1.0), 0.0, 1.0, np.nextafter(1.0, 2.0), -1e-6, 1.0 + 1e-6])
    np.testing.assert_array_equal(
        storage._outside_unit_interval(values),
        np.asarray([False, False, False, False, True, True]),
    )
