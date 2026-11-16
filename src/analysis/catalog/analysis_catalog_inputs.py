"""Catalog the generation-input and Dataset EDA plots on synthetic inputs.

Responsibilities:
  - Declare one catalog entry per generation-input and Dataset EDA sub-type
  - Build the synthetic inputs of those entries: admitted-case diagnostics, candidate rows and compact transient EDA frames

Design principles:
  - Entries call the real builders on synthetic inputs
  - Every input is built once per process and shared between entries; none is read from storage

This module does NOT:
  - Read storage or run the generation pipeline; the admitted cases are assembled in memory from their persisted contract
  - Own a plot or table implementation
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.analysis.eda import eda_dataframe
from src.analysis.eda.plots import eda_plot_case_statistics, eda_plot_spectral_analysis, eda_plot_transient
from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.generation_inputs import generation_input_sources as sources
from src.analysis.generation_inputs import generation_input_template_comparison as template
from src.analysis.generation_inputs.plots import (
    generation_input_plot_atlas,
    generation_input_plot_boundaries,
    generation_input_plot_moisture,
    generation_input_plot_overview,
    generation_input_plot_spatial,
    generation_input_plot_template,
)
from src.generation.cases import generation_cases_admission as admission
from src.generation.contracts import generation_contracts_profiles as profiles

from .analysis_catalog_entries import CatalogPlot

_SHAPE: Final = (10, 16)  # grid rows (y) x columns (x)
_EDA_CASES: Final = 6  # cases of the transient Dataset EDA frame
_SCHEDULE_HOURS: Final = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)  # a startup rejoin row at 0.25 h, then a regular 0.5 h grid
_BUILDERS: Final = "src.analysis.generation_inputs.plots"
_OVERVIEW: Final = f"{_BUILDERS}.generation_input_plot_overview"
_SPECTRAL: Final = "src.analysis.eda.plots.eda_plot_spectral_analysis"
_STATISTICS: Final = "src.analysis.eda.plots.eda_plot_case_statistics"
_CAMPAIGNS: Final = (("transient_drying", "campaign_a"), ("transient_drying", "campaign_b"))
_PARAMETERS: Final = (("kappa_mean", "m^2"), ("initial_moisture.mean_db", "kg/kg"), ("schedule.corr", "1"))
_DURATION: Final = {"duration_seconds": "Duration [s]"}


def _case_diagnostics(index: int, batch: str) -> diagnostics.GenerationInputDiagnostics:
    """Return the diagnostics of one transient input case assembled from its persisted contract, varied by ``index``."""
    ny, nx = _SHAPE
    x, y = np.meshgrid(np.linspace(0.0, 0.3, nx), np.linspace(0.0, 0.1, ny))
    wave = np.sin(2.0 * np.pi * (x / 0.3 + 0.1 * index) * (1 + index % 2)) * np.cos(np.pi * y / 0.1)
    inlet = np.zeros(_SHAPE)
    inlet[0] = 60.0 + 15.0 * index + 10.0 * np.sin(2.0 * np.pi * x[0] / 0.3)
    fields = {
        "x": x,
        "y": y,
        "Kxx": 1.0e-9 * (1.5 + 0.4 * wave),
        "Kxy": 1.0e-10 * (0.5 + 0.3 * wave),
        "Kyy": 1.0e-9 * (1.0 - 0.3 * wave),
        "eps_bed": 0.4 + 0.02 * index + 0.03 * wave,
        "p_in_bc": inlet,
        "X_0_db_field": 0.15 + 0.01 * index + 0.03 * wave,
    }
    scalars: dict[str, float] = dict(
        zip(
            profiles.TRANSIENT_SCALAR_INPUT_FIELDS,
            (295.0, 0.4, 780.0, 0.2, 1800.0, 0.12, 1e-4, 1.0, 0.5, 12.06202053, -0.0573838, 0.34338283),
            strict=True,
        )
    )
    temperature = np.array((295.0, 300.0, 303.0, 305.0, 304.0, 302.0)) + 2.0 * index
    humidity_ratio = np.array((0.006, 0.007, 0.008, 0.009, 0.0085, 0.008)) + 0.0005 * index
    schedule = np.column_stack((_SCHEDULE_HOURS, temperature, humidity_ratio))
    handoff = {
        "canonical_regular_grid": {"start_h": 0.0, "stop_h": 2.0, "interval_h": 0.5, "node_count": 5},
        "startup_ramp": {"enabled": True, "duration_h": 0.25},
        "rejoin_row": schedule[1].tolist(),
        "canonical_start_row": schedule[0].tolist(),
        "regular_output_time_policy": "common.time.regular_times_unchanged",
    }
    payload = {
        "simulation_profile": profiles.TRANSIENT_DRYING_PROFILE,
        "case_input_id": f"input-{batch}-{index}",
        "simulation_case_id": f"simulation-{batch}-{index}",
        "git_commit": "a" * 40,
        "input_files": ["case.json", "schedule.csv"],
        "batch_id": batch,
        "batch_identity": f"identity-{batch}",
        "material_family": "lentil",
        "sampling_regime": "natural_support",
        "sampled_values": {
            "kappa_mean": 1.0e-9 * (1.0 + 0.2 * index),
            "T_init": 293.0,
            "schedule.component_weights": {"smooth": 0.5, "event": 0.3 - 0.02 * index, "trend": 0.2 + 0.02 * index},
        },
        "sampled_units": {"kappa_mean": "m^2", "T_init": "K", "schedule.component_weights": "1"},
        "stationary_fixed_values": [
            {"name": name, "value": value, "unit": unit}
            for name, value, unit in zip(profiles.STATIONARY_FIXED_FIELDS, (300.0, 101325.0, 0.0), profiles.STATIONARY_FIXED_UNITS, strict=True)
        ],
        "schedule_diagnostics": {"boundary_handoff": handoff},
    }
    case = admission.AdmittedInputCase(
        source_id=f"input-{batch}",
        source_kind="input_generation",
        profile_id=profiles.TRANSIENT_DRYING_PROFILE,
        batch_storage_name=f"storage-{batch}",
        campaign_purpose="source_generation",
        label=f"Case {index}",
        case_id=f"case_{index:04d}",
        case_index=index,
        directory=Path(f"synthetic/{batch}/case_{index:04d}"),
        payload=payload,
        parameter_metadata={},
        fields=fields,
        scalars=scalars,
        schedule=schedule,
    )
    return diagnostics.build_case_diagnostics(case)


@cache
def _datasets() -> tuple[diagnostics.DatasetDiagnostics, diagnostics.DatasetDiagnostics]:
    """Return two input Datasets of two cases each, A holding cases 1 and 2 and B cases 3 and 4."""
    return (
        diagnostics.build_dataset_diagnostics([_case_diagnostics(index, "batch-a") for index in (1, 2)]),
        diagnostics.build_dataset_diagnostics([_case_diagnostics(index, "batch-b") for index in (3, 4)]),
    )


def _pair() -> tuple[Any, ...]:
    """Return the arguments ``first, mean_a, second, mean_b`` of the two-case comparison builders."""
    mean_a, mean_b = _datasets()
    return mean_a.records[0], mean_a, mean_b.records[0], mean_b


@cache
def _candidate_rows() -> list[dict[str, Any]]:
    """Return the candidates of two campaigns over eight designs of two materials, paired as the template comparison needs."""
    designs = (
        ("lentil", 1, ("successful", 10.0), ("successful", 12.0)),
        ("lentil", 2, ("successful", 20.0), ("successful", 18.0)),
        ("lentil", 3, ("successful", 5.0), ("failed", None)),
        ("lentil", 4, ("failed", None), ("successful", 30.0)),
        ("chickpea", 1, ("successful", 40.0), ("successful", 44.0)),
        ("chickpea", 2, ("failed", None), ("successful", 8.0)),
        ("chickpea", 3, ("pending", None), ("successful", 9.0)),
        ("chickpea", 4, ("successful", 6.0), ("failed", None)),
    )
    rows = []
    for material, index, *outcomes in designs:
        for (task, source), (outcome, duration) in zip(_CAMPAIGNS, outcomes, strict=True):
            scale = 1.0 if material == "lentil" else 1.5
            rows.append(
                {
                    "task": task,
                    "source_name": source,
                    "material_family": material,
                    "design_key": f"sobol_reference:{index}",
                    "case_id": f"case_{index:04d}",
                    "origin": "primary",
                    "outcome": outcome,
                    "failure": {"reason_code": "stalled" if index % 2 else "timed_out"} if outcome == "failed" else None,
                    "duration_seconds": duration,
                    "parameters": {
                        "kappa_mean": scale * index * 1.0e-9,
                        "initial_moisture.mean_db": 0.1 + 0.03 * index,
                        "schedule.corr": 0.2 * index - 0.4,
                    },
                    "parameter_units": dict(_PARAMETERS),
                }
            )
    return rows


def _eda_frame() -> pd.DataFrame:
    """Return a transient Dataset frame of ``_EDA_CASES`` cases with sampled parameters, two states and the static input fields."""
    ny, nx = _SHAPE
    x, y = np.meshgrid(np.linspace(0.0, 0.3, nx), np.linspace(0.0, 0.1, ny))
    rows = []
    for case in range(_EDA_CASES):
        rng = np.random.default_rng(case)
        smooth = (1.0 + 0.3 * case) * np.sin(2.0 * np.pi * (x / 0.3) * (case % 3 + 1)) * np.cos(np.pi * y / 0.1)
        states = {
            "T": 300.0 + 3.0 * np.arange(3)[:, None, None] * (1.0 + smooth) + rng.normal(0.0, 0.2, (3, ny, nx)),
            "phi": 0.5 + 0.1 * smooth * np.ones((3, 1, 1)),
        }
        rows.append(
            {
                "meta": {
                    "parameters": {
                        "kappa_mean": 1.0e-9 * (1 + case),
                        "kappa_cv": 0.3 + 0.1 * case,
                        "T_in_base": 300.0 + 2.0 * case,
                        "omega_in_base": 0.008 + 0.001 * case,
                    },
                    "parameter_units": {"kappa_mean": "m^2", "kappa_cv": "1", "T_in_base": "K", "omega_in_base": "kg/kg"},
                },
                "state_trajectories": states,
                "static_fields": {
                    "x": x,
                    "y": y,
                    "u": 0.5 + 0.1 * smooth,
                    "v": 0.2 * smooth,
                    "p": 50.0 * (1.0 - x / 0.3) + case,
                    "eps_bed": 0.4 + 0.02 * smooth,
                    "X_0_db_field": 0.2 + 0.02 * smooth,
                    "rho_bu_dry": np.full(_SHAPE, 650.0 + 10.0 * case),
                },
                "time": {"regular_state_hours": np.array((0.0, 1.0, 2.0)), "valid_state_mask": np.array((True, True, True))},
            }
        )
    frame = pd.DataFrame(rows, index=pd.Index([f"case_{case + 1:04d}" for case in range(_EDA_CASES)], name="sample_id"))
    static = ("x", "y", "u", "v", "p", "eps_bed", "X_0_db_field", "rho_bu_dry")
    frame.attrs.update(
        {
            "task_id": "transient_drying",
            "task_contract_digest": "synthetic-transient-contract",
            "field_names": ("T", "phi"),
            "field_categories": {"dynamic_state": ("T", "phi"), "static_spatial": static},
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


@cache
def _eda_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return one labelled Dataset frame and its series styles."""
    label = "Drying · Lentil · Synthetic"
    return {label: _eda_frame()}, eda_plot_transient.batch_styles({label: "lentil"})


def _draw_map_grid_01() -> object:
    atlas = generation_input_plot_atlas.prepare_field_atlas(_pair()[0])
    return generation_input_plot_atlas.field_atlas(atlas, ("Kxx", "Kxy", "eps_bed", "Kyy", "K_orientation", "X_0_db_field"))


def _draw_map_plus_lines_01() -> object:
    return generation_input_plot_spatial.basic_comparison(*_pair(), lock_scale=True)


def _draw_map_plus_lines_02() -> object:
    return generation_input_plot_moisture.moisture_comparison(*_pair(), lock_scale=True)


def _draw_time_series_01() -> object:
    return generation_input_plot_boundaries.schedule_comparison(*_pair(), same_dataset=False)


def _draw_distribution_01() -> object:
    rows = [row for row in _candidate_rows() if row["source_name"] == "campaign_a"]
    values = {parameter: sources.outcome_parameter_values(rows, *parameter) for parameter in _PARAMETERS}
    counts = {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in ("successful", "failed", "pending")}
    return generation_input_plot_overview.outcome_parameter_distributions(values, counts)


def _draw_distribution_02() -> object:
    datasets, styles = _eda_inputs()
    return eda_plot_case_statistics.plot_meta_statistics(datasets=datasets, styles=styles).widget


def _draw_distribution_03() -> object:
    datasets, styles = _eda_inputs()
    return eda_plot_case_statistics.plot_field_value_distributions(datasets=datasets, styles=styles).widget


def _draw_scatter_parity_01() -> object:
    rows = [row for row in _candidate_rows() if row["source_name"] == "campaign_a"]
    return generation_input_plot_overview.failures_by_parameter(rows, _PARAMETERS)


def _draw_scatter_parity_09() -> object:
    comparison = template.compare_templates(_candidate_rows(), *_CAMPAIGNS, quantities=_DURATION)
    return generation_input_plot_template.template_comparison(comparison)


def _draw_spectral_01() -> object:
    datasets, styles = _eda_inputs()
    return eda_plot_spectral_analysis.plot_isotropic_spectral_summary(datasets=datasets, styles=styles, channels=("T", "phi"))


def _draw_spectral_02() -> object:
    datasets, styles = _eda_inputs()
    return eda_plot_spectral_analysis.plot_directional_spectral_case(datasets=datasets, styles=styles, case_number=1, channels=("T", "phi"))


def _draw_spectral_03() -> object:
    datasets, _styles = _eda_inputs()
    return eda_plot_spectral_analysis.plot_vertical_spectral_evolution(datasets=datasets, channels=("T",))


def _draw_table_01() -> object:
    first, mean_a, second, mean_b = _pair()
    return (
        generation_input_plot_overview.case_context(first, mean_a, second, mean_b),
        generation_input_plot_overview.parameters(first, mean_a, second, mean_b),
        generation_input_plot_overview.field_summaries(first, mean_a, second, mean_b),
        generation_input_plot_overview.dataset_cases_parameters(mean_a),
        generation_input_plot_overview.dataset_field_summaries(mean_a),
    )


def _draw_table_02() -> object:
    return generation_input_plot_boundaries.boundary_comparison_table(*_pair())


def _draw_own_02() -> object:
    x, y = np.meshgrid(np.linspace(0.0, 0.3, _SHAPE[1]), np.linspace(0.0, 0.1, _SHAPE[0]))
    kxx = eda_dataframe.PublishedTensorValue("static", "input", "Kxx", "m^2", "identity", "spatial", 1.0e-9 * (1.5 + np.sin(2.0 * np.pi * x / 0.3)))
    inlet = eda_dataframe.PublishedTensorValue(
        "boundary", "input", "p_in_bc", "Pa", "identity", "inlet boundary", 60.0 + 10.0 * np.sin(2.0 * np.pi * x[0] / 0.3)
    )
    return tuple(eda_plot_case_statistics.plot_published_tensor(value, x=x, y=y, case_label="Case 1") for value in (kxx, inlet))


ENTRIES: Final[tuple[CatalogPlot, ...]] = (
    CatalogPlot("map-grid_01", "Single-case field atlas", (f"{_BUILDERS}.generation_input_plot_atlas.field_atlas",), _draw_map_grid_01),
    CatalogPlot(
        "map-plus-lines_01", "Basic two-case comparison", (f"{_BUILDERS}.generation_input_plot_spatial.basic_comparison",), _draw_map_plus_lines_01
    ),
    CatalogPlot(
        "map-plus-lines_02",
        "Multi-field two-case comparison",
        (f"{_BUILDERS}.generation_input_plot_moisture.moisture_comparison",),
        _draw_map_plus_lines_02,
    ),
    CatalogPlot(
        "time-series_01", "Operating schedules", (f"{_BUILDERS}.generation_input_plot_boundaries.schedule_comparison",), _draw_time_series_01
    ),
    CatalogPlot("distribution_01", "Outcome histogram", (f"{_OVERVIEW}.outcome_parameter_distributions",), _draw_distribution_01),
    CatalogPlot("distribution_02", "Dataset metadata histograms", (f"{_STATISTICS}.plot_meta_statistics",), _draw_distribution_02),
    CatalogPlot("distribution_03", "Dataset field-value distributions", (f"{_STATISTICS}.plot_field_value_distributions",), _draw_distribution_03),
    CatalogPlot("scatter-parity_01", "Outcome strip", (f"{_OVERVIEW}.failures_by_parameter",), _draw_scatter_parity_01),
    CatalogPlot(
        "scatter-parity_09",
        "Paired template comparison (NV5)",
        (f"{_BUILDERS}.generation_input_plot_template.template_comparison",),
        _draw_scatter_parity_09,
    ),
    CatalogPlot("spectral_01", "Isotropic curves", (f"{_SPECTRAL}.plot_isotropic_spectral_summary",), _draw_spectral_01),
    CatalogPlot("spectral_02", "Directional curves", (f"{_SPECTRAL}.plot_directional_spectral_case",), _draw_spectral_02),
    CatalogPlot("spectral_03", "Position-frequency images", (f"{_SPECTRAL}.plot_vertical_spectral_evolution",), _draw_spectral_03),
    CatalogPlot(
        "table_01",
        "Generation case and Dataset information",
        tuple(
            f"{_OVERVIEW}.{function}"
            for function in ("case_context", "parameters", "field_summaries", "dataset_cases_parameters", "dataset_field_summaries")
        ),
        _draw_table_01,
    ),
    CatalogPlot("table_02", "Boundary comparison", (f"{_BUILDERS}.generation_input_plot_boundaries.boundary_comparison_table",), _draw_table_02),
    CatalogPlot("own_02", "Published tensor dispatch", (f"{_STATISTICS}.plot_published_tensor",), _draw_own_02),
)
