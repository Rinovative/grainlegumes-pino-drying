"""
analysis_presentation_curated.py

Render the fixed scientific media inventory accepted by optional W&B tracking.

Responsibilities:
  - Build the fixed steady figure/table bundle from validated aggregate frames
  - Build the fixed transient sequence, process, comparison, and timing report
  - Save rendered files below an explicit caller-owned output directory

Design principles:
  - Input frames already satisfy current artifact/provenance compatibility
  - Output media remains separate from immutable ID/OOD artifact cache directories
  - Rendering never mutates artifact caches or run state
  - Every task returns an exact bounded inventory without importing W&B

This module does NOT:
  - Import, initialize, or call W&B
  - Admit artifact provenance or decide whether frames are comparable
  - Own the scientific mathematics implemented by evaluation plot modules
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation.evaluation_plot import (
    evaluation_plot_physical_consistency as physical_consistency,
)
from src.analysis.evaluation.evaluation_plot import evaluation_plot_run_summary as run_summary
from src.analysis.evaluation.evaluation_plot import evaluation_plot_spectral_fidelity as spectral_fidelity

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.figure import Figure

    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession

CURATED_ANALYSIS_KEYS = frozenset(
    {
        "run_summary_table",
        "accuracy_physics_pareto",
        "dual_continuity_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    }
)
TRANSIENT_CURATED_ANALYSIS_KEYS = frozenset(
    {
        "transient_summary_table",
        "transient_state_maps",
        "transient_central_error_time",
        "transient_horizon_error",
        "transient_endpoint_cumulative",
        "transient_target_time",
        "transient_pipeline_degradation",
        "transient_timing_distributions",
        "transient_timing_speedups",
        "transient_accuracy_inference_time",
        "transient_accuracy_speedup",
    }
)
_TRANSIENT_OPTIONAL_ANALYSIS_KEYS = frozenset({"transient_training_performance_compute"})


@dataclass(frozen=True, slots=True)
class CuratedAnalysisBundle:
    """
    Hold the exact local scientific bundle accepted by W&B tracking.

    Parameters
    ----------
    media_files : Mapping[str, pathlib.Path]
        Four explicitly rendered files outside immutable artifact caches.
    tables : Mapping[str, Any]
        Neutral ``run_summary_table`` columns/data payload.

    Raises
    ------
    ValueError
        If keys are missing, additional, or shared by both mappings.

    Notes
    -----
    The dataclass is frozen and slotted, but the supplied mappings are not copied
    or deeply frozen. Callers retain ownership of rendered files and mappings.

    """

    media_files: Mapping[str, Path]
    tables: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the exact fixed five-key inventory."""
        names = set(self.media_files).union(self.tables)
        if names != CURATED_ANALYSIS_KEYS or set(self.media_files).intersection(self.tables):
            msg = f"Curated analysis bundle must contain exactly {sorted(CURATED_ANALYSIS_KEYS)}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TransientCuratedAnalysisBundle:
    """Hold the fixed local transient report inventory outside artifact caches."""

    media_files: Mapping[str, Path]
    tables: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the exact sequence-aware report inventory."""
        names = set(self.media_files).union(self.tables)
        allowed_inventories = {
            TRANSIENT_CURATED_ANALYSIS_KEYS,
            TRANSIENT_CURATED_ANALYSIS_KEYS.union(_TRANSIENT_OPTIONAL_ANALYSIS_KEYS),
        }
        if frozenset(names) not in allowed_inventories or set(self.media_files).intersection(self.tables):
            msg = "Transient curated bundle has an invalid required or matched-compute inventory."
            raise ValueError(msg)


def _neutral_table(frame: pd.DataFrame) -> dict[str, object]:
    """
    Convert one summary frame to a path-free, W&B-neutral table payload.

    The index is materialized as columns, and NumPy scalars become Python scalars,
    supported primitives remain unchanged, and other values use readable text.
    No W&B type is imported or constructed.
    """
    table = frame.reset_index()
    columns = [str(column) for column in table.columns]
    data: list[list[object]] = []
    for row in table.itertuples(index=False, name=None):
        values: list[object] = []
        for value in row:
            scalar = value.item() if isinstance(value, np.generic) else value
            if scalar is None or isinstance(scalar, (str, bool, int, float)):
                values.append(scalar)
            else:
                values.append(str(scalar))
        data.append(values)
    return {"columns": columns, "data": data}


def _save_figure(figure: Figure, path: Path) -> None:
    """
    Save or replace one figure path, then close the Matplotlib figure.

    A non-figure-like object fails before writing. Matplotlib owns format and
    overwrite behavior for an existing caller-owned target.
    """
    savefig = getattr(figure, "savefig", None)
    if not callable(savefig):
        msg = f"Curated renderer expected a Matplotlib figure for {path.name}."
        raise TypeError(msg)
    savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_curated_analysis(
    *,
    datasets: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> CuratedAnalysisBundle:
    """
    Render the exact curated inventory without importing W&B or mutating caches.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible ID or OOD evaluation frames with physics evidence.
    output_dir : str | pathlib.Path
        Caller-owned render directory outside every immutable artifact root. It
        is created when absent. Existing fixed-name PNG targets are overwritten.

    Returns
    -------
    CuratedAnalysisBundle
        Four PNG paths and one path-neutral summary-table payload under the exact
        five-key tracking allowlist.

    Raises
    ------
    ComparisonCompatibilityError
        If frames are scientifically incompatible or lack required physics data.
    ValueError
        If the output directory is inside an immutable artifact cache.
    TypeError
        If a renderer does not return a Matplotlib figure-like object.

    Notes
    -----
    Rendering owns only the four output figures. The caller owns directory
    lifecycle and cleanup. Evaluation readers and plot modules own admission and
    scientific calculations.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    target = Path(output_dir).resolve()
    artifact_roots = {
        Path(str(frame.attrs["artifact_root"])).resolve() for frame in datasets.values() if frame.attrs.get("artifact_root") is not None
    }
    if any(target == root or target.is_relative_to(root) for root in artifact_roots):
        msg = "Curated analysis output must be outside every immutable artifact cache."
        raise ValueError(msg)
    target.mkdir(parents=True, exist_ok=True)

    media_files = {
        "accuracy_physics_pareto": target / "accuracy_physics_pareto.png",
        "dual_continuity_diagnostics": target / "dual_continuity_diagnostics.png",
        "pressure_boundary_summary": target / "pressure_boundary_summary.png",
        "spectral_fidelity": target / "spectral_fidelity.png",
    }
    renderers = {
        "accuracy_physics_pareto": run_summary.plot_accuracy_physics_pareto,
        "dual_continuity_diagnostics": physical_consistency.plot_spatial_residuals,
        "pressure_boundary_summary": physical_consistency.plot_pressure_boundary_summary,
        "spectral_fidelity": spectral_fidelity.plot_spectral_fidelity,
    }
    for name, renderer in renderers.items():
        _save_figure(renderer(datasets=datasets), media_files[name])

    summary = run_summary.build_run_summary_table(datasets)
    return CuratedAnalysisBundle(
        media_files=media_files,
        tables={"run_summary_table": _neutral_table(summary)},
    )


def render_curated_transient_analysis(
    *,
    session: TransientEvaluationSession,
    output_dir: str | Path,
    training_performance: pd.DataFrame | None = None,
) -> TransientCuratedAnalysisBundle:
    """Render one fixed sequence-aware local report without mutating artifacts."""
    from src.analysis.evaluation.evaluation_plot import evaluation_plot_transient as transient_plot  # noqa: PLC0415

    if not session.frame_names:
        msg = "Transient curated rendering requires one open Evaluation session."
        raise ValueError(msg)
    target = Path(output_dir).resolve()
    if any(target == root or target.is_relative_to(root) for root in session.artifact_roots):
        msg = "Transient curated output must be outside every immutable artifact cache."
        raise ValueError(msg)
    target.mkdir(parents=True, exist_ok=True)
    primary_frame = session.frame_names[0]
    primary_record = session.full_autonomous_records(primary_frame)[0]
    target_records = session.full_autonomous_records()
    summary = session.dataset_dataframe()
    timing_report = session.timing_report(primary_frame)
    primary_accuracy = session.case_dataframe(modes=("autonomous_full",))
    primary_accuracy = primary_accuracy.loc[
        (primary_accuracy["frame"] == primary_frame) & (primary_accuracy["scope"] == "cumulative") & (primary_accuracy["requested_horizon"] == "full")
    ]
    media_files = {
        "transient_state_maps": target / "transient_state_maps.png",
        "transient_central_error_time": target / "transient_central_error_time.png",
        "transient_horizon_error": target / "transient_horizon_error.png",
        "transient_endpoint_cumulative": target / "transient_endpoint_cumulative.png",
        "transient_target_time": target / "transient_target_time.png",
        "transient_pipeline_degradation": target / "transient_pipeline_degradation.png",
        "transient_timing_distributions": target / "transient_timing_distributions.png",
        "transient_timing_speedups": target / "transient_timing_speedups.png",
        "transient_accuracy_inference_time": target / "transient_accuracy_inference_time.png",
        "transient_accuracy_speedup": target / "transient_accuracy_speedup.png",
    }
    figures = {
        "transient_state_maps": transient_plot.plot_state_maps(primary_record),
        "transient_central_error_time": transient_plot.plot_central_error_vs_time(
            primary_record,
            scaling_state=session.scaling_state(primary_frame),
        ),
        "transient_horizon_error": transient_plot.plot_horizon_error(summary),
        "transient_endpoint_cumulative": transient_plot.plot_endpoint_vs_cumulative(summary),
        "transient_target_time": transient_plot.plot_target_time(target_records),
        "transient_pipeline_degradation": transient_plot.plot_pipeline_degradation(session.pipeline_degradation()),
        "transient_timing_distributions": transient_plot.plot_timing_distributions(timing_report),
        "transient_timing_speedups": transient_plot.plot_timing_speedups(timing_report),
        "transient_accuracy_inference_time": transient_plot.plot_accuracy_vs_inference_time(primary_accuracy, timing_report),
        "transient_accuracy_speedup": transient_plot.plot_accuracy_vs_speedup(primary_accuracy, timing_report),
    }
    if training_performance is not None:
        name = "transient_training_performance_compute"
        media_files[name] = target / f"{name}.png"
        figures[name] = transient_plot.plot_training_performance_vs_compute(training_performance)
    for name, figure in figures.items():
        _save_figure(figure, media_files[name])
    return TransientCuratedAnalysisBundle(
        media_files=media_files,
        tables={"transient_summary_table": _neutral_table(summary)},
    )
