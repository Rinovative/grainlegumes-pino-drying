# ruff: noqa: S101
"""Focused persisted diagnostics for failed transient initial-state reconstruction."""

from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff
from src.generation.publication import generation_publication_storage as storage
from src.generation.runtime import generation_runtime_diagnostics as diagnostics

if TYPE_CHECKING:
    from src.generation.cases.generation_cases_config import GenerationConfig


_GRID_NODE_COUNT = 4
_SEMANTIC_RTOL = 1.0e-4
_SEMANTIC_ATOL = 1.0e-10
_FLOAT32_RTOL = 1.0e-6
_FLOAT32_ATOL = 1.0e-12
_F_SURF = 0.4


def _reconstructed(
    *,
    surface_scale: np.ndarray | float,
    interior_scale: np.ndarray | float,
) -> storage.ReconstructedTransientInitialState:
    """Return compact arrays shaped exactly like production reconstruction."""
    rho = np.asarray(((500.0, 600.0), (700.0, 800.0)), dtype=np.float64)
    x_initial = np.asarray(((0.10, 0.20), (0.15, 0.25)), dtype=np.float64)
    expected = rho * x_initial
    stationary = np.zeros(
        (len(profiles.TRANSIENT_STATIC_FIELD_NAMES), 2, 2),
        dtype=np.float64,
    )
    stationary[profiles.TRANSIENT_STATIC_FIELD_NAMES.index("rho_bu_dry")] = rho
    stationary[profiles.TRANSIENT_STATIC_FIELD_NAMES.index("X_0_db_field")] = x_initial
    states = np.zeros(
        (2, len(profiles.TRANSIENT_FIELD_NAMES), 2, 2),
        dtype=np.float64,
    )
    states[0, profiles.TRANSIENT_FIELD_NAMES.index("w_surf")] = expected * surface_scale
    states[0, profiles.TRANSIENT_FIELD_NAMES.index("w_int")] = expected * interior_scale
    entries = [
        scalar_handoff.ScalarHandoffEntry(
            name=name,
            value=_F_SURF if name == "f_surf" else 1.0,
            unit=unit,
            owner="case_dependent",
        )
        for name, unit in zip(
            profiles.scalar_input_fields(profiles.TRANSIENT_DRYING_PROFILE),
            profiles.scalar_input_units(profiles.TRANSIENT_DRYING_PROFILE),
            strict=True,
        )
    ]
    admission = scalar_handoff.admit_transient_scalar_handoff(
        entries,
        source_path=Path("scalars.csv"),
        source_filename="scalars.csv",
        sha256="d" * 64,
        size_bytes=1,
    )
    return storage.ReconstructedTransientInitialState(
        stationary_fields=stationary,
        transient_states=states,
        time=np.asarray((0.0, 1.0, 1.5), dtype=np.float64),
        x_axis=np.asarray((0.0, 0.1), dtype=np.float64),
        y_axis=np.asarray((0.0, 0.1), dtype=np.float64),
        scalar_handoff=admission,
        source_artifacts={
            "stationary_fields": {
                "path": "exports/stationary_fields.csv",
                "sha256": "a" * 64,
                "size_bytes": 101,
            },
            "transient_states": {
                "path": "exports/transient_states.csv",
                "sha256": "b" * 64,
                "size_bytes": 202,
            },
        },
    )


def _diagnose(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    surface_scale: np.ndarray | float = 1.0,
    interior_scale: np.ndarray | float = 1.0,
) -> diagnostics.InitialStateDiagnostic:
    """Persist one synthetic canonical reconstruction through the real metric owner."""
    reconstructed = _reconstructed(
        surface_scale=surface_scale,
        interior_scale=interior_scale,
    )
    monkeypatch.setattr(
        diagnostics.storage,
        "reconstruct_transient_initial_state",
        lambda *_args, **_kwargs: reconstructed,
    )
    config = cast(
        "GenerationConfig",
        cast(
            "object",
            type(
                "SyntheticGenerationConfig",
                (),
                {
                    "profile": type(
                        "SyntheticProfile",
                        (),
                        {"id": profiles.TRANSIENT_DRYING_PROFILE},
                    )(),
                    "batch_id": "transient_drying__lentil__natural",
                    "scientific_values": {
                        "validation": {
                            "transient_initial_state": {
                                "rtol": _SEMANTIC_RTOL,
                                "atol": _SEMANTIC_ATOL,
                            },
                        },
                        "storage": {
                            "float32_rtol": _FLOAT32_RTOL,
                            "float32_atol": _FLOAT32_ATOL,
                        },
                        "time": {
                            "start": 0.0,
                            "stop": 1.5,
                            "interval": 1.0,
                        },
                    },
                },
            )(),
        ),
    )
    case_payload = {
        "case_id": "case_0001",
        "template": {"sha256": "c" * 64},
    }
    return diagnostics.write_initial_state_diagnostic(
        config,
        case_payload,
        stationary_export=tmp_path / "stationary_fields.csv",
        transient_export=tmp_path / "transient_states.csv",
        work_directory=tmp_path,
        output_directory=tmp_path / "diagnostics",
        campaign_run_id="technical-smoke__0123456789abcdef",
    )


def test_initial_state_diagnostic_classifies_canonical_and_persists_full_grid(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Persist exact canonical evidence and one row per canonical spatial node."""
    result = _diagnose(tmp_path, monkeypatch)
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    with result.csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert payload["schema_kind"] == diagnostics.DIAGNOSTIC_SCHEMA_KIND
    assert payload["schema_version"] == diagnostics.DIAGNOSTIC_SCHEMA_VERSION
    assert payload["diagnostic_classification"] == "approximately_canonical"
    assert payload["validator"] == {
        "rtol": _SEMANTIC_RTOL,
        "atol": _SEMANTIC_ATOL,
        "w_surf_allclose": True,
        "w_int_allclose": True,
    }
    assert payload["time_axis"] == {
        "first_exported_time": 0.0,
        "second_exported_time": 1.0,
        "temporal_group_count": 3,
    }
    assert payload["weighted_total"]["error"]["max_abs"] == 0.0
    assert payload["w_surf"]["ratio"]["coefficient_of_variation"] == 0.0
    assert payload["w_int"]["worst_locations"]["maximum_relative_error"]["x"] == 0.0
    assert len(rows) == _GRID_NODE_COUNT
    assert tuple(rows[0]) == (
        "x",
        "y",
        "rho_bu_dry",
        "X_0_db_field",
        "expected_w_gr_0",
        "w_surf_t0",
        "w_int_t0",
        "w_surf_error",
        "w_int_error",
        "w_surf_rel_error",
        "w_int_rel_error",
        "w_surf_ratio",
        "w_int_ratio",
        "weighted_w_gr_t0",
        "weighted_w_gr_error",
        "weighted_w_gr_rel_error",
        "weighted_w_gr_ratio",
    )


def test_initial_state_diagnostic_distinguishes_split_and_redistribution(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Distinguish explicit splitting from redistributed conserved total water."""
    split = _diagnose(
        tmp_path / "split",
        monkeypatch,
        surface_scale=_F_SURF,
        interior_scale=1.0 - _F_SURF,
    )
    redistributed = _diagnose(
        tmp_path / "redistributed",
        monkeypatch,
        surface_scale=0.5,
        interior_scale=(1.0 - _F_SURF * 0.5) / (1.0 - _F_SURF),
    )

    assert split.payload["diagnostic_classification"] == "compartment_split"
    assert split.payload["w_surf"]["split_hypothesis"]["rmse"] == 0.0
    assert split.payload["w_int"]["split_hypothesis"]["rmse"] == 0.0
    assert redistributed.payload["diagnostic_classification"] == "redistributed_conserved_total"
    one_compartment = _diagnose(
        tmp_path / "one-compartment",
        monkeypatch,
        surface_scale=_F_SURF,
        interior_scale=2.0,
    )
    assert one_compartment.payload["diagnostic_classification"] == "other"
    np.testing.assert_allclose(
        redistributed.payload["weighted_total"]["error"]["max_abs"],
        0.0,
        rtol=0.0,
        atol=_SEMANTIC_ATOL,
    )


def test_initial_state_diagnostic_uses_exact_configured_tolerance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Distinguish perturbations immediately below and above production tolerance."""
    below = _diagnose(
        tmp_path / "below",
        monkeypatch,
        surface_scale=1.0 + 0.5 * _SEMANTIC_RTOL,
        interior_scale=1.0 + 0.5 * _SEMANTIC_RTOL,
    )
    above = _diagnose(
        tmp_path / "above",
        monkeypatch,
        surface_scale=1.0 + 2.0 * _SEMANTIC_RTOL,
        interior_scale=1.0 + 2.0 * _SEMANTIC_RTOL,
    )

    assert below.payload["diagnostic_classification"] == "approximately_canonical"
    assert below.payload["validator"]["w_surf_allclose"] is True
    assert above.payload["diagnostic_classification"] == "other"
    assert above.payload["validator"]["w_surf_allclose"] is False
    assert above.payload["w_surf"]["relative_error"]["median"] > _SEMANTIC_RTOL


def test_real_solver_scale_discrepancy_is_approximately_canonical(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Admit the observed solver-initialization regime with the semantic tolerance."""
    result = _diagnose(
        tmp_path,
        monkeypatch,
        surface_scale=np.asarray(((1.0 - 2.1043e-5, 1.0 - 1.33e-5), (1.0 - 8.0e-6, 1.0 - 3.6e-6))),
        interior_scale=np.asarray(((1.0 - 1.1296e-5, 1.0), (1.0 + 7.93e-6, 1.0 - 2.0e-11))),
    )

    assert result.payload["diagnostic_classification"] == "approximately_canonical"
    assert result.payload["validator"]["w_surf_allclose"] is True
    assert result.payload["validator"]["w_int_allclose"] is True
    assert result.payload["validator"] == {
        "rtol": _SEMANTIC_RTOL,
        "atol": _SEMANTIC_ATOL,
        "w_surf_allclose": True,
        "w_int_allclose": True,
    }


def test_diagnostic_module_cannot_launch_solver_or_scheduler() -> None:
    """Keep diagnostics structurally unable to execute COMSOL or scheduler work."""
    parsed = ast.parse(inspect.getsource(diagnostics))
    imported_modules = {node.names[0].name for node in ast.walk(parsed) if isinstance(node, ast.Import) and node.names}
    imported_from_modules = {node.module for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert "subprocess" not in imported_modules
    assert not any(name.endswith(("generation_runtime_batch", "generation_runtime_comsol")) for name in imported_from_modules)


def test_bulk_moisture_diagnostic_uses_production_result_and_tolerance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Persist compact metrics from the exact production comparison result."""
    discrepancy = 1.8912875530963102e-7
    time_axis = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
    expected = np.asarray((0.12, 0.11, 0.10), dtype=np.float64)
    exported = expected + np.asarray(
        (2.0e-8, discrepancy, -4.0e-8),
        dtype=np.float64,
    )
    consistency = storage.TransientBulkMoistureConsistency(
        time=time_axis,
        exported=exported,
        reconstructed=expected,
        rtol=1.0e-5,
        atol=1.0e-9,
        matches=True,
    )
    reconstructed = storage.ReconstructedTransientBulkMoisture(
        consistency=consistency,
        source_artifacts={
            "stationary_fields": {
                "relative_path": "exports/stationary_fields.csv",
                "sha256": "a" * 64,
                "size_bytes": 101,
            },
            "transient_states": {
                "relative_path": "exports/transient_states.csv",
                "sha256": "b" * 64,
                "size_bytes": 202,
            },
            "global_time_series": {
                "relative_path": "exports/global_timeseries.csv",
                "sha256": "c" * 64,
                "size_bytes": 303,
            },
        },
    )
    monkeypatch.setattr(
        diagnostics.storage,
        "reconstruct_transient_bulk_moisture",
        lambda *_args, **_kwargs: reconstructed,
    )
    config = cast(
        "GenerationConfig",
        cast(
            "object",
            type(
                "SyntheticGenerationConfig",
                (),
                {
                    "profile": type(
                        "SyntheticProfile",
                        (),
                        {"id": profiles.TRANSIENT_DRYING_PROFILE},
                    )(),
                    "batch_id": "transient_drying__lentil__natural",
                },
            )(),
        ),
    )
    result = diagnostics.write_bulk_moisture_consistency_diagnostic(
        config,
        {
            "case_id": "case_0001",
            "template": {"sha256": "d" * 64},
        },
        stationary_export=tmp_path / "stationary_fields.csv",
        transient_export=tmp_path / "transient_states.csv",
        global_export=tmp_path / "global_timeseries.csv",
        work_directory=tmp_path,
        output_directory=tmp_path / "diagnostics",
        campaign_run_id="technical-smoke__0123456789abcdef",
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert result.json_path.name == ("bulk_moisture_consistency_diagnostic.json")
    assert payload["schema_kind"] == (diagnostics.BULK_MOISTURE_DIAGNOSTIC_SCHEMA_KIND)
    assert payload["schema_version"] == 1
    assert payload["validator"] == {
        "rtol": 1.0e-5,
        "atol": 1.0e-9,
        "allclose": True,
    }
    assert payload["time_axis"] == {
        "number_of_time_points": 3,
        "first_time": 0.0,
        "final_time": 2.0,
    }
    assert payload["error"]["maximum_absolute_error"] == pytest.approx(
        discrepancy,
    )
    assert payload["error"]["time_of_maximum_error"] == 1.0
    assert payload["error"]["exported_X_wb_bulk_at_max_error"] == pytest.approx(exported[1])
    assert payload["error"]["reconstructed_X_wb_bulk_at_max_error"] == pytest.approx(expected[1])
    assert set(payload["error"]["absolute_error_quantiles"]) == {
        "q50",
        "q95",
        "q99",
    }
