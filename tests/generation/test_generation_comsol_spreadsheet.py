# ruff: noqa: PLR2004, S101, SLF001, TC003
"""COMSOL Spreadsheet parsing and mapping regressions."""

from __future__ import annotations

from pathlib import Path

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
        "simulation_profile": "transient_drying",
        "steady_flow_conditioning": None,
        "exports": [
            {
                "role": "steady_flow_fields",
                "temporal_kind": "stationary",
                "source": {"state": "runtime_confirmed", "pattern": "steady.csv"},
                "delimiter": ",",
                "columns": {
                    "x": {"state": "runtime_confirmed", "source_header": "x"},
                    "p": {"state": "runtime_confirmed", "source_header": "p"},
                },
            },
            {
                "role": "transient_fields",
                "temporal_kind": "regular_time_series",
                "source": {"state": "runtime_confirmed", "pattern": "transient.csv"},
                "delimiter": ",",
                "columns": {
                    "t": {"state": "runtime_confirmed", "source_header": "t"},
                    "T": {"state": "runtime_confirmed", "source_header": "T"},
                    "phi": {"state": "runtime_confirmed", "source_header": "mt.phi"},
                },
            },
            {
                "role": "global_time_series",
                "temporal_kind": "regular_time_series",
                "source": {"state": "runtime_confirmed", "pattern": "globals.csv"},
                "delimiter": ",",
                "columns": {
                    "t": {"state": "runtime_confirmed", "source_header": "t"},
                    "X_wb_bulk": {"state": "runtime_confirmed", "source_header": "comp1.X_wb_bulk"},
                    "T_out_mean": {"state": "runtime_confirmed", "source_header": "comp1.T_out_mean"},
                },
            },
            {
                "role": "final_status",
                "temporal_kind": "final_status",
                "source": {"state": "runtime_confirmed", "pattern": "final.csv"},
                "delimiter": ",",
                "columns": {
                    "t_final": {"state": "runtime_confirmed", "source_header": "t"},
                    "T_max_final": {
                        "state": "runtime_confirmed",
                        "source_header": "comp1.maxop_bed(comp1.T)",
                    },
                    "phi_min_final": {
                        "state": "runtime_confirmed",
                        "source_header": "comp1.minop_bed(comp1.mt.phi)",
                    },
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
% t (h),T (K),mt.phi
0,296,0.5
""",
        "globals.csv": """% Model,model.mph
% Expressions,3
% t (h),comp1.X_wb_bulk,comp1.T_out_mean (K)
0,0.2,296
""",
        "final.csv": """% Model,model.mph
% t (h),comp1.maxop_bed(comp1.T) (K),comp1.minop_bed(comp1.mt.phi)
1,295,0.4
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
