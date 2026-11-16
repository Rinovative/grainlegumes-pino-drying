"""Catalog the component-check plots, the training-inspection table and the training plots on synthetic inputs.

Responsibilities:
  - Declare one catalog entry per component-check and training sub-type
  - Build the synthetic inputs: an analytic Darcy-flow reference with a smooth error, small result tables, and training histories

Design principles:
  - Entries call the real builders on synthetic inputs; the residual, derivative and sensitivity tables come from the real producers
  - Grids and tables stay as small as the builders allow

This module does NOT:
  - Read storage or published Datasets
  - Reimplement a plot builder or a table producer
"""

from __future__ import annotations

import dataclasses
import math
from functools import cache
from pathlib import Path
from typing import Final

import pandas as pd
import torch

from src import domain
from src.analysis.component_checks import component_checks_increments as increments
from src.analysis.component_checks import component_checks_plots as plots
from src.analysis.component_checks import component_checks_sources as sources
from src.analysis.component_checks import component_checks_steady as steady
from src.analysis.component_checks import component_checks_summary as summary
from src.analysis.evaluation import evaluation_run_discovery as discovery
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.evaluation.evaluation_plot import evaluation_plot_training as training
from src.experiments import experiments_notebook_support as notebook_support

from .analysis_catalog_entries import CatalogPlot

_VISCOSITY: Final = domain.physics.contracts.AIR_DYNAMIC_VISCOSITY
_PERMEABILITY: Final = 1e-8
_INLET_PRESSURE: Final = 3.0
_EXTENT: Final = (0.0, 2.0, 0.0, 1.0)
_SETTINGS: Final = (("fd", 0), ("fd", 2), ("fc", 2))


@cache
def _reference() -> sources.SteadyReference:
    """Return two cases of uniform Darcy flow on a 36 x 40 grid with an interior band, whose pressure and v carry a seeded smooth 5 % error."""
    height, width, cases = 36, 40, 2
    task = domain.tasks.registry.get_task("steady_flow")
    y_grid, x_grid = torch.meshgrid(torch.linspace(0.0, 1.0, height), torch.linspace(0.0, 2.0, width), indexing="ij")
    log_permeability = torch.full_like(x_grid, math.log10(_PERMEABILITY))
    by_input = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": log_permeability,
        "Kxy": torch.zeros_like(x_grid),
        "Kyy": log_permeability,
        "eps_bed": torch.full_like(x_grid, 0.5),
        "p_in_bc": torch.where(y_grid == 0.0, _INLET_PRESSURE, 0.0),
    }
    by_output = {
        "p": _INLET_PRESSURE * (1.0 - y_grid),
        "u": torch.zeros_like(x_grid),
        "v": torch.full_like(x_grid, _PERMEABILITY * _INLET_PRESSURE / _VISCOSITY),
    }
    inputs = torch.stack([by_input[name] for name in task.input_names]).expand(cases, -1, -1, -1).clone()
    outputs = torch.stack([by_output[name] for name in task.output_names]).expand(cases, -1, -1, -1).clone()
    exact = sources.SteadyReference(
        dataset_id="synthetic",
        membership="train",
        case_ids=tuple(f"lentil__case_{index:04d}__digest" for index in range(cases)),
        material_families=("lentil",) * cases,
        input_fields=tuple(task.input_names),
        output_fields=tuple(task.output_names),
        inputs=inputs,
        outputs=outputs,
    )
    _, perturbed = steady.perturb_outputs(exact, steady.Perturbation("smooth", ("p", "v"), 0.05), dtype=torch.float32, seed=1)
    return dataclasses.replace(exact, outputs=perturbed)


@cache
def _backend_fields() -> dict[str, steady.SteadyResidualFields]:
    """Return the residual fields of the reference per derivative backend."""
    return steady.evaluate_backends(_reference())


def _draw_map_grid_03() -> object:
    fields = list(_backend_fields().values())
    return plots.plot_residual_maps(fields, case_index=0, case_label=_reference().case_labels[0], extent=_EXTENT)


def _draw_time_series_03() -> object:
    rows = [
        {"rule": rule, "end_h": hour, **{f"{name}_relative": scale * hour for name in ("internal", "surface", "grain")}}
        for rule, scale in (("trapezoid", 1e-6), ("left", 1e-4))
        for hour in (1.0, 2.0, 3.0)
    ]
    return plots.plot_sorption_residuals({"case 1": pd.DataFrame(rows)})


def _history(stage: str, *, start_epoch: int = 0, pipeline: str = "a0") -> discovery.StageHistory:
    """Return a four-epoch history validated twice per epoch at 10 steps per epoch."""
    epochs = tuple({"epoch": float(epoch), "global_step": 10.0 * epoch, "train/loss_total": 1.0 / epoch} for epoch in range(1, 5))
    validations = tuple(
        {
            "epoch": epoch + 0.5,
            "global_step": 10.0 * epoch + 5.0,
            "validation/point": 10.0 * epoch,
            "validation/eligible": 1.0,
            "id/rmse": 0.8 / epoch,
        }
        for epoch in range(1, 5)
    )
    return discovery.StageHistory(
        run_name=f"{pipeline}-{stage}",
        stage=stage,
        model_family="fno",
        pipeline=pipeline,
        start_epoch=start_epoch,
        objective_id="rmse",
        objective_direction="minimize",
        selected_point=40,
        epochs=epochs,
        validations=validations,
    )


def _draw_time_series_08() -> object:
    histories: dict[str, discovery.StageHistory | discovery.UnavailableHistory] = {
        "A0": _history("A0"),
        "B": _history("B", start_epoch=2),
        "old": discovery.UnavailableHistory(Path("old"), "no history"),
    }
    return training.plot_training_curves(histories)


def _draw_time_series_09() -> object:
    trials = [(0, "completed", 0.5, 0.5), (1, "pruned", 0.9, 0.5), (2, "completed", 0.4, 0.4), (3, "diverged", None, 0.4), (4, "completed", 0.3, 0.3)]
    counts = {status: sum(trial[1] == status for trial in trials) for status in {trial[1] for trial in trials}}
    return training.plot_study_curve(
        discovery.StudySummary("study", "rmse", "minimize", counts, tuple(discovery.StudyTrial(*trial) for trial in trials))
    )


def _draw_line_relationship_01() -> object:
    profiles = {label: steady.edge_profiles(steady.residual_magnitude(fields), max_distance=2) for label, fields in _backend_fields().items()}
    return plots.plot_edge_profiles(profiles)


def _draw_line_relationship_02() -> object:
    return plots.plot_derivative_difference(steady.compare_derivatives(_reference()))


def _draw_line_relationship_05() -> object:
    rows = [
        (material, architecture, size, seed, scale * factor * jitter / size**0.5)
        for material, scale in (("lentil", 1.0), ("chickpea", 1.5))
        for architecture, factor in (("fno", 1.0), ("pi-fno", 0.7))
        for size in (16, 32, 64)
        for seed, jitter in ((1, 0.9), (2, 1.1))
    ]
    return training.plot_learning_curve(pd.DataFrame(rows, columns=list(training.LEARNING_CURVE_COLUMNS)), error_label="Relative L2 [-]")


def _draw_line_relationship_06() -> object:
    costs = [
        training.break_even_cost(
            series.SeriesEntity(label, family, physics_enabled=False, role="id", materials=(), run_key=label),
            comsol_case_seconds={"a": 100.0, "b": 140.0},
            surrogate_case_seconds={"a": seconds, "b": seconds},
            training_seconds=3600.0,
            train_cases=10,
            clocks="synthetic clocks",
        )
        for label, family, seconds in (("FNO", "fno", 1.0), ("U-NO", "uno", 3.0))
    ]
    return training.plot_break_even(costs)


def _draw_scatter_parity_02() -> object:
    residuals = steady.summarize_reference_residuals(_reference(), crops=(0, 1, 2))
    return plots.plot_reference_terms(residuals, settings=_SETTINGS)


def _draw_scatter_parity_03() -> object:
    return plots.plot_sensitivity(steady.evaluate_sensitivity(_reference(), settings=_SETTINGS, seeds=(1,)))


def _draw_bar_compare_01() -> object:
    terms = steady.momentum_term_table(_backend_fields(), case_labels=_reference().case_labels, crops=(2,))
    return plots.plot_term_decomposition(terms)


def _draw_spectral_04() -> object:
    rows = [
        {"field": field, "axis": axis, "padding": padding, "modes_y": modes, "modes_x": modes, "relative_l2": (1.0 + offset) * 0.5**modes}
        for offset, field in enumerate(("p", "v"))
        for axis in ("y", "x")
        for padding in (0.0, 0.1)
        for modes in (2, 4, 8, 16)
    ]
    return plots.plot_mode_curves(pd.DataFrame(rows), pd.DataFrame({"modes_y": [8], "modes_x": [16]}), title="States")


def _draw_table_03() -> object:
    index = pd.MultiIndex.from_product([("fd", "fc"), (0, 1, 2)], names=["backend", "crop"])
    crop = pd.DataFrame(
        {"momentum_floor_median": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0], "smooth_1pct_ratio_median": [10.0, 100.0, 300.0, 5.0, 50.0, 99.9]}, index=index
    )
    timed = pd.DataFrame({"device": ["cpu", "cpu"], "backend": ["fd", "fc"], "status": ["measured"] * 2, "median_ms": [10.0, 20.0]})
    modes = pd.DataFrame(
        [
            {"field": "p", "axis": axis, "config": None, "modes_y": count if axis == "y" else None, "modes_x": count if axis == "x" else None}
            | {"padding": 0.0, "relative_l2": error, "edge_relative": error}
            for axis in ("y", "x")
            for count, error in ((8, 2e-3), (16, 1e-3), (32, 1e-4))
        ]
    )
    stride = pd.DataFrame(
        {"stride": [1, 2, 4], "status": ["admissible", "admissible", "refused: endpoint"], "max_normalized_rmse": [0.0, 1e-3, math.nan]}
    )
    tables = increments.TransientComponentTables(spectral=modes, stride=stride, increments=pd.DataFrame(), time_steps=pd.DataFrame())
    return summary.summary_table(crop=crop, timing=timed, steady_modes=modes, transient=tables)


def _draw_table_04() -> object:
    run_dir = Path("storage/experiments/steady_flow/example_run")
    inspection = notebook_support.RunInspection(
        run_dir=run_dir,
        summary_rows=(("Run directory", run_dir), ("Status", "completed"), ("Objective", "rmse"), ("Completed epoch", 40)),
        existence_rows=(("config.yaml", True), ("summary.json", True), ("history.json", False)),
    )
    tables = notebook_support.prepare_run_inspection_tables(inspection, project_root=Path(), storage_root=Path("storage"))
    return tuple(pd.DataFrame(table.rows, columns=table.columns) for table in tables)


_PLOTS: Final = "src.analysis.component_checks.component_checks_plots"
_TRAINING: Final = "src.analysis.evaluation.evaluation_plot.evaluation_plot_training"
ENTRIES: Final[tuple[CatalogPlot, ...]] = (
    CatalogPlot("map-grid_03", "Backend residual maps", (f"{_PLOTS}.plot_residual_maps",), _draw_map_grid_03),
    CatalogPlot("time-series_03", "Component-check time histories", (f"{_PLOTS}.plot_sorption_residuals",), _draw_time_series_03),
    CatalogPlot("time-series_08", "Training curves", (f"{_TRAINING}.plot_training_curves",), _draw_time_series_08),
    CatalogPlot("time-series_09", "Study curve", (f"{_TRAINING}.plot_study_curve",), _draw_time_series_09),
    CatalogPlot("line-relationship_01", "Residual versus spatial distance", (f"{_PLOTS}.plot_edge_profiles",), _draw_line_relationship_01),
    CatalogPlot(
        "line-relationship_02", "Derivative difference by boundary band", (f"{_PLOTS}.plot_derivative_difference",), _draw_line_relationship_02
    ),
    CatalogPlot("line-relationship_05", "Learning curve", (f"{_TRAINING}.plot_learning_curve",), _draw_line_relationship_05),
    CatalogPlot("line-relationship_06", "Break-even", (f"{_TRAINING}.plot_break_even",), _draw_line_relationship_06),
    CatalogPlot("scatter-parity_02", "Reference residual terms", (f"{_PLOTS}.plot_reference_terms",), _draw_scatter_parity_02),
    CatalogPlot("scatter-parity_03", "Perturbation sensitivity ranges", (f"{_PLOTS}.plot_sensitivity",), _draw_scatter_parity_03),
    CatalogPlot("bar-compare_01", "Reference momentum decomposition", (f"{_PLOTS}.plot_term_decomposition",), _draw_bar_compare_01),
    CatalogPlot("spectral_04", "Retained-mode error", (f"{_PLOTS}.plot_mode_curves",), _draw_spectral_04),
    CatalogPlot("table_03", "Component checks", ("src.analysis.component_checks.component_checks_summary.summary_table",), _draw_table_03),
    CatalogPlot("table_04", "Training inspection", ("src.experiments.experiments_notebook_support.prepare_run_inspection_tables",), _draw_table_04),
)
