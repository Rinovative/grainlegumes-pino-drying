"""
===============================================================================
generation_runtime_diagnostics.py
===============================================================================
Measure failed transient initial states from already-produced raw exports.
Responsibilities:
  - Reuse production canonical reconstruction before failed HDF5 publication
  - Quantify canonical, compartment-split, and conserved-total hypotheses
  - Persist hash-bound JSON and full-grid CSV diagnostic evidence
Design principles:
  - Diagnostics never execute solvers or scheduler commands
  - Configured float32 validation tolerances define numerical interpretation
  - Every metric is derived from canonical production parser outputs
This module does NOT:
  - Modify model, solver, or scientific initialization semantics
  - Publish canonical cases or retain runtime artifacts
===============================================================================
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src import common
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.publication import generation_publication_storage as storage

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.generation.cases.generation_cases_config import GenerationConfig

DIAGNOSTIC_SCHEMA_KIND = "vp2_transient_initial_state_diagnostic"
DIAGNOSTIC_SCHEMA_VERSION = 1
_NUMERICAL_FLOOR = np.finfo(np.float64).tiny
_MINIMUM_TIMES_WITH_SECOND = 2
_MATERIAL_HYPOTHESIS_IMPROVEMENT = 0.1
_CSV_COLUMNS = (
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


@dataclass(frozen=True, slots=True)
class InitialStateDiagnostic:
    """Published paths and JSON-compatible measurements for one failed case."""

    json_path: Path
    csv_path: Path
    payload: dict[str, Any]


def _basic(values: np.ndarray) -> dict[str, float | int]:
    """Return required finite field statistics."""
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(value.size),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    """Return the required relative-error or ratio quantiles."""
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        name: float(number)
        for name, number in zip(("q01", "q05", "q50", "q95", "q99"), np.quantile(value, (0.01, 0.05, 0.50, 0.95, 0.99)), strict=True)
    }


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """Return complete field comparison metrics against one hypothesis."""
    error = actual - expected
    relative = np.abs(error) / np.maximum(np.abs(expected), _NUMERICAL_FLOOR)
    ratio = actual / expected
    error_metrics = {
        **_basic(error),
        "mean_abs": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
    ratio_metrics = {**_basic(ratio), "std": float(np.std(ratio)), **_quantiles(ratio)}
    ratio_metrics["coefficient_of_variation"] = (
        float(ratio_metrics["std"] / abs(ratio_metrics["mean"])) if ratio_metrics["mean"] != 0.0 else float("nan")
    )
    return {"actual": _basic(actual), "error": error_metrics, "relative_error": {**_basic(relative), **_quantiles(relative)}, "ratio": ratio_metrics}


def _hypothesis(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Return the direct canonical or split-hypothesis comparison summary."""
    metrics = _comparison(actual, expected)
    return {
        "max_abs_error": float(metrics["error"]["max_abs"]),
        "rmse": float(metrics["error"]["rmse"]),
        "median_relative_error": float(metrics["relative_error"]["median"]),
        "max_relative_error": float(metrics["relative_error"]["max"]),
    }


def _worst(
    actual: np.ndarray, expected: np.ndarray, rho: np.ndarray, x0: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray
) -> dict[str, dict[str, float | int]]:
    """Locate maximum absolute and relative error points on the canonical grid."""
    error = actual - expected
    relative = np.abs(error) / np.maximum(np.abs(expected), _NUMERICAL_FLOOR)
    result: dict[str, dict[str, float | int]] = {}
    for name, index in (("maximum_absolute_error", int(np.argmax(np.abs(error)))), ("maximum_relative_error", int(np.argmax(relative)))):
        y_index, x_index = np.unravel_index(index, error.shape)
        result[name] = {
            "x": float(x_axis[x_index]),
            "y": float(y_axis[y_index]),
            "rho_bu_dry": float(rho[y_index, x_index]),
            "X_0_db_field": float(x0[y_index, x_index]),
            "expected": float(expected[y_index, x_index]),
            "actual": float(actual[y_index, x_index]),
            "error": float(error[y_index, x_index]),
            "relative_error": float(relative[y_index, x_index]),
            "ratio": float(actual[y_index, x_index] / expected[y_index, x_index]),
        }
    return result


def _csv_text(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    rho: np.ndarray,
    x0: np.ndarray,
    expected: np.ndarray,
    surface: np.ndarray,
    interior: np.ndarray,
    f_surf: float,
) -> str:
    """Serialize full-grid canonical diagnostic evidence with round-trip precision."""
    weighted = f_surf * surface + (1.0 - f_surf) * interior
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(_CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for y_index, y in enumerate(y_axis):
        for x_index, x in enumerate(x_axis):
            expected_value, surface_value, interior_value, weighted_value = (
                array[y_index, x_index] for array in (expected, surface, interior, weighted)
            )
            values = (
                x,
                y,
                rho[y_index, x_index],
                x0[y_index, x_index],
                expected_value,
                surface_value,
                interior_value,
                surface_value - expected_value,
                interior_value - expected_value,
                abs(surface_value - expected_value) / max(abs(expected_value), _NUMERICAL_FLOOR),
                abs(interior_value - expected_value) / max(abs(expected_value), _NUMERICAL_FLOOR),
                surface_value / expected_value,
                interior_value / expected_value,
                weighted_value,
                weighted_value - expected_value,
                abs(weighted_value - expected_value) / max(abs(expected_value), _NUMERICAL_FLOOR),
                weighted_value / expected_value,
            )
            writer.writerow(dict(zip(_CSV_COLUMNS, (format(float(value), ".17g") for value in values), strict=True)))
    return stream.getvalue()


def write_initial_state_diagnostic(
    config: GenerationConfig,
    case_payload: Mapping[str, Any],
    *,
    stationary_export: Path,
    transient_export: Path,
    work_directory: Path,
    output_directory: Path,
    campaign_run_id: str,
) -> InitialStateDiagnostic:
    """
    Measure and persist one failed transient case's initial-state diagnostic.

    Parameters
    ----------
    config : GenerationConfig
        Resolved transient Technical-Smoke batch configuration.
    case_payload : Mapping[str, Any]
        Prepared case provenance used for scalar-handoff admission.
    stationary_export : Path
        Already-produced configured stationary Spreadsheet export.
    transient_export : Path
        Already-produced configured wide transient Spreadsheet export.
    work_directory : Path
        Case workspace containing the original scalar handoff.
    output_directory : Path
        Staging directory for JSON and full-grid CSV evidence.
    campaign_run_id : str
        Current durable campaign execution identity.

    Returns
    -------
    InitialStateDiagnostic
        Published diagnostic paths and the measured JSON payload.

    Raises
    ------
    FileNotFoundError
        If a required raw source is missing or unsafe.
    ValueError
        If production reconstruction or the zero-time contract fails.

    Notes
    -----
    This function only reads existing case outputs. It has no solver or
    scheduler dependency and cannot launch another simulation.

    """
    reconstructed = storage.reconstruct_transient_initial_state(
        config, case_payload, stationary_export=stationary_export, transient_export=transient_export, work_directory=work_directory
    )
    time_tolerance = storage.time_classification_tolerance(config.scientific_values["time"])
    if reconstructed.time.size < 1 or abs(float(reconstructed.time[0])) > time_tolerance:
        message = "Initial-state diagnostic requires an exported zero-time transient group."
        raise ValueError(message)
    fields, states = profiles.TRANSIENT_STATIC_FIELD_NAMES, profiles.TRANSIENT_FIELD_NAMES
    rho, x0 = reconstructed.stationary_fields[fields.index("rho_bu_dry")], reconstructed.stationary_fields[fields.index("X_0_db_field")]
    expected = rho * x0
    surface, interior = reconstructed.transient_states[0, states.index("w_surf")], reconstructed.transient_states[0, states.index("w_int")]
    scalar_names = reconstructed.scalar_handoff.field_names
    f_surf = float(reconstructed.scalar_handoff.values[scalar_names.index("f_surf")])
    rtol, atol = (float(config.scientific_values["storage"][name]) for name in ("float32_rtol", "float32_atol"))
    weighted = f_surf * surface + (1.0 - f_surf) * interior
    surface_canonical, interior_canonical = _comparison(surface, expected), _comparison(interior, expected)
    surface_split, interior_split = _hypothesis(surface, f_surf * expected), _hypothesis(interior, (1.0 - f_surf) * expected)
    canonical_pass = bool(np.allclose(surface, expected, rtol=rtol, atol=atol) and np.allclose(interior, expected, rtol=rtol, atol=atol))
    weighted_pass = bool(np.allclose(weighted, expected, rtol=rtol, atol=atol))
    surface_split_match = bool(np.allclose(surface, f_surf * expected, rtol=rtol, atol=atol))
    interior_split_match = bool(
        np.allclose(
            interior,
            (1.0 - f_surf) * expected,
            rtol=rtol,
            atol=atol,
        )
    )
    surface_split_better = (
        surface_canonical["error"]["rmse"] > 0.0 and surface_split["rmse"] <= _MATERIAL_HYPOTHESIS_IMPROVEMENT * surface_canonical["error"]["rmse"]
    )
    interior_split_better = (
        interior_canonical["error"]["rmse"] > 0.0 and interior_split["rmse"] <= _MATERIAL_HYPOTHESIS_IMPROVEMENT * interior_canonical["error"]["rmse"]
    )
    split_better = surface_split_match and surface_split_better and interior_split_match and interior_split_better
    classification = (
        "approximately_canonical"
        if canonical_pass
        else "compartment_split"
        if split_better
        else "redistributed_conserved_total"
        if weighted_pass
        else "other"
    )
    payload: dict[str, Any] = {
        "schema_kind": DIAGNOSTIC_SCHEMA_KIND,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "campaign_run_id": campaign_run_id,
        "batch_id": config.batch_id,
        "case_id": case_payload["case_id"],
        "template_sha256": case_payload["template"]["sha256"],
        "f_surf": f_surf,
        "source_artifacts": reconstructed.source_artifacts,
        "time_axis": {
            "first_exported_time": float(reconstructed.time[0]),
            "second_exported_time": (None if reconstructed.time.size < _MINIMUM_TIMES_WITH_SECOND else float(reconstructed.time[1])),
            "temporal_group_count": int(reconstructed.time.size),
        },
        "validator": {
            "rtol": rtol,
            "atol": atol,
            "w_surf_allclose": bool(np.allclose(surface, expected, rtol=rtol, atol=atol)),
            "w_int_allclose": bool(np.allclose(interior, expected, rtol=rtol, atol=atol)),
        },
        "expected_w_gr_0": _basic(expected),
        "w_surf": {
            "canonical_hypothesis": _hypothesis(surface, expected),
            "split_hypothesis": surface_split,
            **surface_canonical,
            "worst_locations": _worst(surface, expected, rho, x0, reconstructed.x_axis, reconstructed.y_axis),
        },
        "w_int": {
            "canonical_hypothesis": _hypothesis(interior, expected),
            "split_hypothesis": interior_split,
            **interior_canonical,
            "worst_locations": _worst(interior, expected, rho, x0, reconstructed.x_axis, reconstructed.y_axis),
        },
        "weighted_total": _comparison(weighted, expected),
        "diagnostic_classification": classification,
    }
    json_path = common.serialization.atomic_write_json(output_directory / "initial_state_diagnostic.json", payload)
    csv_path = common.serialization.atomic_write_text(
        output_directory / "initial_state_diagnostic.csv",
        _csv_text(reconstructed.x_axis, reconstructed.y_axis, rho, x0, expected, surface, interior, f_surf),
    )
    return InitialStateDiagnostic(json_path=json_path, csv_path=csv_path, payload=payload)
