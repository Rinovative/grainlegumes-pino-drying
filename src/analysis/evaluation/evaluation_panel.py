"""
evaluation_panel.py

Compose curated evaluation panels over admitted artifacts.

Responsibilities:
  - Keep the approved single-model and comparison view order explicit
  - Bind every view to its defined local controls and defaults
  - Accept caller-provided context labels and session-owned canonical frames
  - Keep comparison roles separate without an outer dataset selector
  - Preserve lazy first rendering and panel-local figure export

Design principles:
  - Strict artifact loading and EvaluationSession binding precede construction
  - Dataset, model, channel, case, and parameter controls remain view-specific
  - Presentation labels never alter canonical identity or numerical cache keys
  - Panel construction opens no case payloads and computes no plot mathematics

This module does NOT:
  - Discover, generate, repair, rebuild, or admit artifact payloads
  - Create an alternate dataframe or generic interaction registry
  - Infer dataset identity from display names, paths, or row positions
  - Cache figures, case arrays, or numerical reductions
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src import analysis

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.analysis.evaluation.evaluation_selection import EvaluationSelectionState
    from src.analysis.evaluation.evaluation_session import EvaluationSession
    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession

_MINIMUM_COMPARISON_RUNS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_FRAME_MATERIAL_CONTEXT_SIZE = 2
_OUTLIER_COUNT = 5

SINGLE_MODEL_EVALUATION_SECTION_KEYS = (
    "overview",
    "global_error",
    "architecture",
    "error_decomposition",
    "physical_consistency",
    "spectral_analysis",
    "parameter_sensitivity",
    "sample_viewer",
    "outlier_analysis",
)
COMPARISON_EVALUATION_SECTION_KEYS = SINGLE_MODEL_EVALUATION_SECTION_KEYS
TRANSIENT_EVALUATION_SECTION_KEYS = (
    *SINGLE_MODEL_EVALUATION_SECTION_KEYS,
    "temporal_rollout",
)

EVALUATION_SECTION_TITLES = {
    "overview": "Overview",
    "global_error": "Global Error Analysis",
    "architecture": "Architecture Sensitivity",
    "error_decomposition": "Error Decomposition",
    "physical_consistency": "Physical Consistency",
    "spectral_analysis": "Spectral & Representation Analysis",
    "parameter_sensitivity": "Error Sensitivity",
    "sample_viewer": "Sample Viewer",
    "outlier_analysis": "Outlier & Extreme Case Analysis",
    "temporal_rollout": "Temporal & Rollout Analysis",
}


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Define one notebook-provided dataset context and its ordered model frames."""

    key: str
    label: str
    datasets: Mapping[str, pd.DataFrame]


def _dropdown(
    options: Sequence[str] | Sequence[tuple[str, Any]],
    *,
    description: str,
    value: Any = None,
) -> widgets.Dropdown:
    """Build one compact scientific dropdown."""
    resolved = tuple(options)
    if not resolved:
        msg = f"No options are available for {description.rstrip(':').lower()}."
        raise ValueError(msg)
    first = resolved[0][1] if isinstance(resolved[0], tuple) else resolved[0]
    return widgets.Dropdown(
        options=resolved,
        value=first if value is None else value,
        description=description,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
    )


def _physics_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether every frame supplies compatible maintained physics evidence."""
    try:
        analysis.evaluation.dataframe.validate_comparison(datasets, require_physics=True)
    except analysis.evaluation.dataframe.ComparisonCompatibilityError:
        return False
    return True


def _permeability_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether every frame declares the required permeability tensor inputs."""
    required = {"Kxx", "Kxy", "Kyy"}
    return all(required.issubset(frame.attrs.get("input_fields", ())) for frame in datasets.values())


def _pressure_velocity_available(datasets: Mapping[str, pd.DataFrame]) -> bool:
    """Return whether the synchronized two-model pressure/velocity view is valid."""
    if len(datasets) < _MINIMUM_COMPARISON_RUNS:
        return False
    roles = {analysis.evaluation.dataframe.dataset_role(frame) for frame in datasets.values()}
    if len(roles) != 1:
        return False
    try:
        for frame in datasets.values():
            analysis.evaluation.dataframe.single_output_group_field(frame, group_id="pressure")
            velocity = analysis.evaluation.dataframe.output_group_fields(frame, group_id="velocity")
            if len(velocity) < _MINIMUM_VELOCITY_COMPONENTS:
                return False
    except analysis.evaluation.dataframe.ComparisonCompatibilityError:
        return False
    return True


def _make_outlier_table_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    top_k: int = _OUTLIER_COUNT,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """Render the automatic outlier tables with one local dropdown."""
    if not datasets:
        msg = "Outlier tables require at least one labelled dataset."
        raise ValueError(msg)
    selector = analysis.ui.components.ui_dropdown_dataset(list(datasets)) if len(datasets) > 1 else None
    if selector is not None:
        selector.layout.width = "229px"
    output = analysis.ui.components.ui_output_plot()

    def render(_: object = None) -> None:
        """Rebuild the tables immediately after a local dataset change."""
        if selector is None:
            selected_name = next(iter(datasets))
        else:
            selected_name = selector.value
            if not isinstance(selected_name, str):
                msg = "Dataset dropdown values must be strings."
                raise TypeError(msg)
        analysis.ui.viewers.render_figure(
            out=output,
            plot_func=plot_func,
            kwargs={"datasets": {selected_name: datasets[selected_name]}, "top_k": top_k},
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    if selector is not None:
        selector.observe(render, names="value")
    render()
    children: list[widgets.Widget] = [] if selector is None else [selector]
    children.append(output)
    return widgets.VBox(children, layout=widgets.Layout(align_items="flex-start"))


def _make_steady_indexed_viewer(  # noqa: C901
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    controls: Mapping[str, widgets.ValueWidget] | None = None,
    dataset_selection: str = "dropdown",
    selection_state: EvaluationSelectionState | None = None,
    channel_fields: Sequence[Any] = (),
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> widgets.VBox:
    """Wrap the maintained steady case viewer with shared case/channel state."""
    semantic_controls = dict(controls or {})
    viewer = analysis.ui.viewers.make_indexed_viewer(
        plot_func,
        datasets=datasets,
        controls=semantic_controls,
        dataset_selection=dataset_selection,
        export_state=export_state,
        export_plot_name=export_plot_name,
        export_title=export_title,
    )
    header = viewer.children[0]
    if not isinstance(header, widgets.HBox) or not header.children or not isinstance(header.children[0], widgets.BoundedIntText):
        msg = "Steady shared-case binding requires the maintained indexed viewer layout."
        raise TypeError(msg)
    case_control = header.children[0]
    dataset_selector = next(
        (control for control in header.children if isinstance(control, widgets.Dropdown) and control.description == "Select:"),
        None,
    )
    raw_channel_control = semantic_controls.get("channel")
    if raw_channel_control is not None and not isinstance(raw_channel_control, widgets.Dropdown):
        msg = "Steady shared-channel binding requires a dropdown control."
        raise TypeError(msg)
    channel_control = raw_channel_control
    key_by_value: dict[str, str] = {}
    value_by_key: dict[str, str] = {}
    for field in channel_fields:
        key = str(field.key)
        label = str(field.label)
        key_by_value.update({key: key, label: key})
        value_by_key[key] = label
    updating = {"value": False}

    def selected_label() -> str:
        """Return the current local model label behind row-position navigation."""
        if dataset_selector is None:
            return next(iter(datasets))
        value = dataset_selector.value
        if not isinstance(value, str) or value not in datasets:
            msg = "Steady dataset selector lost its labelled frame binding."
            raise ValueError(msg)
        return value

    def case_id_at_position(label: str) -> str:
        """Return the exact persisted case identity at the current row position."""
        position = int(case_control.value) - 1
        frame = datasets[label]
        if not 0 <= position < len(frame):
            msg = "Steady case position is outside the selected artifact membership."
            raise IndexError(msg)
        return str(int(frame.iloc[position]["case_index"]))

    def position_for_case(label: str, case_id: str) -> int | None:
        """Return the one-based position for an exact persisted case identity."""
        values = tuple(str(int(value)) for value in datasets[label]["case_index"])
        return values.index(case_id) + 1 if case_id in values else None

    def update_export_stem() -> None:
        """Bind steady task, model, plot, case, and optional channel context."""
        if export_state is None:
            return
        label = selected_label()
        case_id = case_id_at_position(label)
        channel = None if channel_control is None else str(channel_control.value)
        parts = (
            "steady_flow",
            label,
            str(export_plot_name or "sample"),
            f"case_{case_id}",
            None if channel is None else f"channel_{channel}",
        )
        export_state["filename_stem"] = "__".join(part for part in parts if part is not None)

    def publish_case(*_args: object) -> None:
        """Publish one exact case after the maintained viewer accepts its position."""
        if updating["value"]:
            return
        update_export_stem()
        if selection_state is not None:
            selection_state.select_case(case_id_at_position(selected_label()))

    def publish_channel(*_args: object) -> None:
        """Publish a canonical steady channel behind its concise widget label."""
        if updating["value"] or channel_control is None:
            return
        update_export_stem()
        if selection_state is not None:
            value = str(channel_control.value)
            selection_state.select_channels((key_by_value.get(value, value),))

    def sync_from_selection(*_args: object) -> None:
        """Apply the shared case/channel snapshot when this cached view activates."""
        if selection_state is None:
            update_export_stem()
            return
        shared = selection_state.selection
        label = selected_label()
        position = None if shared.case_id is None else position_for_case(label, shared.case_id)
        updating["value"] = True
        try:
            if position is not None:
                case_control.value = position
            if channel_control is not None and shared.channels:
                selected_value = value_by_key.get(shared.channels[0], shared.channels[0])
                available_values = tuple(value for _label, value in channel_control.options)
                if selected_value in available_values:
                    channel_control.value = selected_value
        finally:
            updating["value"] = False
        update_export_stem()

    case_control.observe(publish_case, names="value")
    if dataset_selector is not None:
        dataset_selector.observe(sync_from_selection, names="value")
    if channel_control is not None:
        channel_control.observe(publish_channel, names="value")
    sync_from_selection()
    cast("Any", viewer).activate = sync_from_selection
    return viewer


def _build_sections(
    datasets: Mapping[str, pd.DataFrame],
    *,
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
) -> dict[str, tuple[list[tuple[str, Callable[..., Any], str]], str]]:
    """Bind the approved views directly to their view-specific controls."""
    plots = analysis.evaluation.plots
    viewers = analysis.ui.viewers
    controls = analysis.ui.components
    dataset_map = dict(datasets)
    analysis.evaluation.dataframe.validate_comparison(dataset_map)
    frames = tuple(dataset_map.values())
    display_fields = analysis.evaluation.presentation.shared_display_fields(frames)
    channels = tuple(field.label for field in display_fields)
    parameters = analysis.evaluation.presentation.metadata_parameters(frames)
    parameter_options = tuple((analysis.evaluation.presentation.metadata_label(parameter), parameter) for parameter in parameters)
    toggle = analysis.ui.notebook.make_toggle_shortcut(dataset_map)

    overview_entries = [
        toggle(
            "Overview: Summary table",
            plots.run_summary.plot_run_summary_table,
            plot_name="Overview: Summary table",
        ),
    ]
    if comparison:
        overview_entries.extend(
            (
                toggle(
                    "Overview: Global comparison summary",
                    plots.run_summary.plot_relative_comparison_scoreboard,
                    plot_name="Overview: Global comparison summary",
                ),
                toggle(
                    "Overview: Pareto (Error vs Physics)",
                    plots.run_summary.plot_accuracy_physics_pareto,
                    plot_name="Overview: Pareto (Error vs Physics)",
                ),
            )
        )
    global_error_entries = [
        toggle(
            "1-1. Global error metrics",
            plots.error_behavior.plot_global_error_metrics,
            plot_name="1-1. Global error metrics",
        ),
        toggle(
            "1-2. Global error distribution",
            viewers.make_casecount_viewer,
            plot_name="1-2. Global error distribution",
            plot_func=plots.error_behavior.plot_predictive_error_distributions,
            start_cases=50,
            step_size=50,
        ),
        toggle(
            "1-3. GT vs Prediction (mean)",
            viewers.make_casecount_viewer,
            plot_name="1-3. GT vs Prediction (mean)",
            plot_func=plots.error_behavior.plot_mean_field_bias,
            start_cases=50,
            step_size=50,
        ),
        toggle(
            "1-4. Mean error maps",
            viewers.make_casecount_viewer,
            plot_name="1-4. Mean error maps",
            plot_func=plots.error_behavior.plot_mean_error_maps,
            start_cases=100,
            step_size=50,
            controls={"error_mode": controls.ui_radio_error_mode()},
        ),
        toggle(
            "1-5. Std error maps",
            viewers.make_casecount_viewer,
            plot_name="1-5. Std error maps",
            plot_func=plots.error_behavior.plot_std_error_maps,
            start_cases=100,
            step_size=50,
        ),
    ]

    architecture_entries = [
        toggle(
            "2-1. Model-family configuration",
            plots.run_summary.plot_architecture_table,
            plot_name="2-1. Model-family configuration",
        ),
        toggle(
            "2-2. Exact trainable parameters vs performance",
            viewers.make_casecount_viewer,
            plot_name="2-2. Exact trainable parameters vs performance",
            plot_func=plots.sensitivity_capacity.plot_capacity_accuracy,
            start_cases=50,
            step_size=50,
        ),
    ]

    error_decomposition_entries = [
        toggle(
            "3-1. Error vs |GT| magnitude",
            viewers.make_casecount_viewer,
            plot_name="3-1. Error vs |GT| magnitude",
            plot_func=plots.error_behavior.plot_error_vs_target_magnitude,
            start_cases=50,
            step_size=50,
            allow_dataset_selection=True,
            selector_title="datasets",
        ),
        toggle(
            "3-2. Boundary vs interior error",
            viewers.make_casecount_viewer,
            plot_name="3-2. Boundary vs interior error",
            plot_func=plots.error_behavior.plot_boundary_error_decomposition,
            start_cases=100,
            step_size=50,
            controls={"channels": controls.ui_checkbox_channels(channels=channels)},
        ),
    ]

    physical_entries: list[tuple[str, Callable[..., Any], str]] = []
    if _physics_available(dataset_map):
        physical_entries = [
            toggle(
                "4-1. Physical consistency summary table",
                plots.physical_consistency.plot_physical_consistency_summary_table,
                plot_name="4-1. Physical consistency summary table",
            ),
            toggle(
                "4-2. Physical consistency CDF grid (2x2)",
                viewers.make_casecount_viewer,
                plot_name="4-2. Physical consistency CDF grid (2x2)",
                plot_func=plots.physical_consistency.plot_physical_consistency_cdf_grid,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-3. Velocity continuity residual (∇·u)",
                viewers.make_casecount_viewer,
                plot_name="4-3. Velocity continuity residual (∇·u)",
                plot_func=plots.physical_consistency.plot_velocity_divergence,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-4. Porosity-weighted continuity residual (∇·(εu))",
                viewers.make_casecount_viewer,
                plot_name="4-4. Porosity-weighted continuity residual (∇·(εu))",
                plot_func=plots.physical_consistency.plot_div_eps_u_consistency,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-5. Darcy–Brinkman operator residual",  # noqa: RUF001
                viewers.make_casecount_viewer,
                plot_name="4-5. Darcy–Brinkman operator residual",  # noqa: RUF001
                plot_func=plots.physical_consistency.plot_brinkman_residual,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "4-6. Pressure and boundary-condition consistency",
                viewers.make_casecount_viewer,
                plot_name="4-6. Pressure and boundary-condition consistency",
                plot_func=plots.physical_consistency.plot_pressure_consistency,
                start_cases=100,
                step_size=50,
            ),
        ]

    spectral_entries = [
        toggle(
            "5-1. Demand vs prediction + error",
            viewers.make_casecount_viewer,
            plot_name="5-1. Demand vs prediction + error",
            plot_func=plots.spectral_fidelity.plot_spectral_demand_prediction_error,
            start_cases=50,
            step_size=50,
            controls={
                "channels": controls.ui_checkbox_channels(channels=channels),
                "normalize": controls.ui_checkbox_normalise(default=True),
            },
        ),
        toggle(
            "5-2. Log spectral transfer with support",
            viewers.make_casecount_viewer,
            plot_name="5-2. Log spectral transfer with support",
            plot_func=plots.spectral_fidelity.plot_spectral_transfer_ratio,
            start_cases=50,
            step_size=50,
            controls={"channels": controls.ui_checkbox_channels(channels=channels)},
        ),
    ]

    sensitivity_entries: list[tuple[str, Callable[..., Any], str]] = []
    if parameters:
        sensitivity_entries = [
            toggle(
                "6-1. Parameter-error correlation (heatmap)",
                viewers.make_casecount_viewer,
                plot_name="6-1. Parameter-error correlation (heatmap)",
                plot_func=plots.sensitivity_capacity.plot_metadata_error_heatmap,
                start_cases=100,
                step_size=50,
            ),
            toggle(
                "6-2. Error vs input parameter (binned trend)",
                viewers.make_casecount_viewer,
                plot_name="6-2. Error vs input parameter (binned trend)",
                plot_func=plots.sensitivity_capacity.plot_metadata_error_trends,
                start_cases=100,
                step_size=50,
                controls={"channels": controls.ui_checkbox_channels(channels=channels)},
            ),
        ]

    sample_entries = [
        toggle(
            "7-1. Sample GT vs Prediction",
            _make_steady_indexed_viewer,
            plot_name="7-1. Sample GT vs Prediction",
            plot_func=plots.samples_outliers.plot_sample_prediction_overview,
            controls={
                "scale_mode": controls.ui_radio_pred_scale_mode(),
                "error_mode": controls.ui_radio_error_mode(),
            },
            selection_state=selection_state,
            channel_fields=display_fields,
            dataset_selection="dropdown",
        ),
    ]
    if _permeability_available(dataset_map):
        sample_entries.append(
            toggle(
                "7-2. Kappa tensor with error overlay",
                _make_steady_indexed_viewer,
                plot_name="7-2. Kappa tensor with error overlay",
                plot_func=plots.samples_outliers.plot_permeability_error_overlay,
                controls={
                    "kappa_scale": controls.ui_radio_kappa_scale(),
                    "channel": controls.ui_dropdown_channel(channels=channels),
                    "error_mode": controls.ui_radio_error_mode(),
                },
                selection_state=selection_state,
                channel_fields=display_fields,
                dataset_selection="dropdown",
            )
        )
    if comparison and _pressure_velocity_available(dataset_map):
        labels = tuple(dataset_map)
        sample_entries.append(
            toggle(
                "7-3. Pressure & velocity field comparison",
                _make_steady_indexed_viewer,
                plot_name="7-3. Pressure & velocity field comparison",
                plot_func=plots.samples_outliers.plot_pressure_velocity_comparison,
                controls={
                    "model_1": _dropdown(labels, description="Model 1:", value=labels[0]),
                    "model_2": _dropdown(labels, description="Model 2:", value=labels[1]),
                    "scale_mode": controls.ui_radio_pred_scale_mode(),
                },
                selection_state=selection_state,
                channel_fields=display_fields,
                dataset_selection="all",
            )
        )

    outlier_entries = [
        toggle(
            "8-1. Worst per-channel cases (tables)",
            _make_outlier_table_viewer,
            plot_name="8-1. Worst per-channel cases (tables)",
            plot_func=plots.samples_outliers.plot_outlier_table,
            top_k=_OUTLIER_COUNT,
        ),
        toggle(
            "8-2. Worst per-channel cases (field plots)",
            viewers.make_indexed_viewer,
            plot_name="8-2. Worst per-channel cases (field plots)",
            plot_func=plots.samples_outliers.plot_linked_outlier_cases,
            controls={
                "channel": controls.ui_dropdown_channel(channels=channels),
                "error_mode": controls.ui_radio_error_mode(),
            },
            dataset_selection="dropdown",
            max_positions=_OUTLIER_COUNT + 1,
            index_to_kwargs=lambda index: {"selection_index": index},
        ),
    ]
    if parameters:
        outlier_entries.extend(
            (
                toggle(
                    "8-3. Extreme input parameters (table view)",
                    plots.samples_outliers.plot_input_extremes_table,
                    plot_name="8-3. Extreme input parameters (table view)",
                ),
                toggle(
                    "8-4. Extreme input parameter cases (field plots)",
                    viewers.make_indexed_viewer,
                    plot_name="8-4. Extreme input parameter cases (field plots)",
                    plot_func=plots.samples_outliers.plot_linked_input_extreme_cases,
                    controls={
                        "parameter": _dropdown(parameter_options, description="Parameter:"),
                        "error_mode": controls.ui_radio_error_mode(),
                    },
                    dataset_selection="dropdown",
                    max_positions=2,
                    index_to_kwargs=lambda index: {"selection_index": index},
                ),
            )
        )

    registry = {
        "overview": (overview_entries, EVALUATION_SECTION_TITLES["overview"]),
        "global_error": (global_error_entries, EVALUATION_SECTION_TITLES["global_error"]),
        "architecture": (architecture_entries, EVALUATION_SECTION_TITLES["architecture"]),
        "error_decomposition": (error_decomposition_entries, EVALUATION_SECTION_TITLES["error_decomposition"]),
        "physical_consistency": (physical_entries, EVALUATION_SECTION_TITLES["physical_consistency"]),
        "spectral_analysis": (spectral_entries, EVALUATION_SECTION_TITLES["spectral_analysis"]),
        "parameter_sensitivity": (sensitivity_entries, EVALUATION_SECTION_TITLES["parameter_sensitivity"]),
        "sample_viewer": (sample_entries, EVALUATION_SECTION_TITLES["sample_viewer"]),
        "outlier_analysis": (outlier_entries, EVALUATION_SECTION_TITLES["outlier_analysis"]),
    }
    return {key: value for key, value in registry.items() if value[0]}


def _selected_section_keys(
    sections: Sequence[str] | str,
    *,
    registry: Mapping[str, object],
) -> list[str]:
    """Validate explicit section selection against the fixed approved order."""
    approved = SINGLE_MODEL_EVALUATION_SECTION_KEYS
    if sections == "all":
        return [key for key in approved if key in registry]
    if isinstance(sections, str) or not isinstance(sections, Sequence) or not sections:
        msg = "sections must be 'all' or a non-empty sequence of section keys."
        raise TypeError(msg)
    selected = tuple(sections)
    if not all(isinstance(key, str) for key in selected):
        msg = "sections must contain only string keys."
        raise TypeError(msg)
    if len(selected) != len(set(selected)):
        msg = "sections must not contain duplicate keys."
        raise ValueError(msg)
    unknown = sorted(set(selected).difference(approved))
    if unknown:
        msg = f"Unknown evaluation sections: {unknown}."
        raise ValueError(msg)
    unavailable = [key for key in selected if key not in registry]
    if unavailable:
        msg = f"Evaluation sections are unavailable for these artifacts: {unavailable}."
        raise ValueError(msg)
    return list(selected)


def _build_panel(
    *,
    datasets: Mapping[str, pd.DataFrame],
    comparison: bool,
    sections: Sequence[str] | str,
    selection_state: EvaluationSelectionState | None = None,
) -> widgets.Widget:
    """Build one lazy panel without an outer dataset selector."""
    dataset_map = dict(datasets)
    registry = _build_sections(
        dataset_map,
        comparison=comparison,
        selection_state=selection_state,
    )
    section_keys = _selected_section_keys(sections, registry=registry)
    if not section_keys:
        msg = "The selected artifacts provide no approved evaluation sections."
        raise ValueError(msg)
    export_state = {
        "fig": None,
        "plot_name": None,
        "title": None,
        "filename_stem": None,
        "filename_prefix": "steady_flow",
    }
    ui_sections = [
        analysis.ui.notebook.make_dropdown_section(
            registry[key][0],
            export_state=export_state,
            select_first=True,
        )
        for key in section_keys
    ]
    open_text = "Model comparison - Open Evaluation" if comparison else "Single model - Open Evaluation"
    return analysis.ui.notebook.make_lazy_panel_with_tabs(
        ui_sections,
        tab_titles=[registry[key][1] for key in section_keys],
        open_btn_text=open_text,
        close_btn_text="Close",
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )


def _normalize_contexts(
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    *,
    comparison: bool,
) -> tuple[EvaluationContext, ...]:
    """Validate ordered contexts and exact live-session frame ownership."""
    if session.closed:
        msg = "Cannot build an evaluation panel from a closed session."
        raise analysis.evaluation.session.EvaluationSessionClosedError(msg)
    if isinstance(contexts, (str, bytes)) or not isinstance(contexts, Sequence) or not contexts:
        msg = "contexts must be a non-empty ordered sequence of EvaluationContext values."
        raise TypeError(msg)
    canonical_frames = tuple(session.canonical_frames.values())
    normalized = []
    for position, context in enumerate(contexts):
        if not isinstance(context, EvaluationContext):
            msg = f"contexts[{position}] must be an EvaluationContext."
            raise TypeError(msg)
        key = context.key.strip() if isinstance(context.key, str) else ""
        label = context.label.strip() if isinstance(context.label, str) else ""
        if not key or not label:
            msg = "Every evaluation context requires non-blank key and label text."
            raise ValueError(msg)
        datasets = dict(context.datasets)
        if comparison and len(datasets) < _MINIMUM_COMPARISON_RUNS:
            msg = f"Evaluation context {key!r} must contain at least two models."
            raise ValueError(msg)
        if not comparison and len(datasets) != 1:
            msg = f"Evaluation context {key!r} must contain exactly one model."
            raise ValueError(msg)
        if any(not isinstance(name, str) or not name.strip() for name in datasets):
            msg = f"Evaluation context {key!r} contains a blank or non-string model label."
            raise TypeError(msg)
        for model_label, frame in datasets.items():
            if not any(frame is canonical for canonical in canonical_frames):
                msg = f"Evaluation context {key!r} frame {model_label!r} is not an exact canonical frame owned by the supplied session."
                raise ValueError(msg)
        analysis.evaluation.dataframe.validate_comparison(datasets)
        normalized.append(EvaluationContext(key=key, label=label, datasets=datasets))
    keys = tuple(context.key for context in normalized)
    labels = tuple(context.label for context in normalized)
    if len(keys) != len(set(keys)):
        msg = "Evaluation context keys must be unique."
        raise ValueError(msg)
    if len(labels) != len(set(labels)):
        msg = "Evaluation context labels must be unique."
        raise ValueError(msg)
    return tuple(normalized)


def build_single_model_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
) -> widgets.Widget:
    """Build one single-model panel containing every supplied dataset."""
    normalized = _normalize_contexts(session, contexts, comparison=False)
    model_labels = tuple(next(iter(context.datasets)) for context in normalized)
    if len(set(model_labels)) != 1:
        msg = "Single-model contexts must all refer to the same presentation model label."
        raise ValueError(msg)
    datasets = {context.label: next(iter(context.datasets.values())) for context in normalized}
    return _build_panel(
        datasets=datasets,
        comparison=False,
        sections=sections,
        selection_state=selection_state,
    )


def build_comparison_panel(
    *,
    session: EvaluationSession,
    contexts: Sequence[EvaluationContext],
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
) -> widgets.Widget:
    """Build separate role-local comparison panels without a selector."""
    normalized = _normalize_contexts(session, contexts, comparison=True)
    panels = [
        _build_panel(
            datasets=context.datasets,
            comparison=True,
            sections=sections,
            selection_state=selection_state,
        )
        for context in normalized
    ]
    if len(panels) == 1:
        return panels[0]
    children: list[widgets.Widget] = []
    for context, panel in zip(normalized, panels, strict=True):
        children.extend((widgets.HTML(f"<h3>{escape(context.label)}</h3>"), panel))
    return widgets.VBox(children)


def _selected_widget_index(value: Any, *, option_count: int) -> int:
    """Admit one integer widget selection within its current option range."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "Transient sequence selection must be an integer index."
        raise TypeError(msg)
    if not 0 <= value < option_count:
        msg = "Transient sequence selection is outside the current record range."
        raise ValueError(msg)
    return value


class _ActivatableVBox(widgets.VBox):
    """Refresh one cached state-bound view when its dropdown becomes active."""

    def __init__(
        self,
        children: Sequence[widgets.Widget],
        *,
        activate: Callable[[], None],
    ) -> None:
        """Retain the normal VBox contract beside one activation callback."""
        super().__init__(
            children=tuple(children),
            layout=widgets.Layout(align_items="flex-start"),
        )
        self._activate_callback = activate

    def activate(self) -> None:
        """Synchronize this cached view from the shared Evaluation state."""
        self._activate_callback()


def _shared_transient_physical_times(records: Sequence[Any]) -> tuple[float, ...]:
    """Return exact stored physical times shared by every selected case."""
    admitted = tuple(records)
    if not admitted:
        msg = "Aggregate transient spatial views require complete-rollout records."
        raise ValueError(msg)
    shared = {float(value) for value in admitted[0].physical_times}
    for record in admitted[1:]:
        shared.intersection_update(float(value) for value in record.physical_times)
    times = tuple(sorted(shared))
    if not times:
        msg = "Aggregate transient cases share no exact stored physical time."
        raise ValueError(msg)
    return times


def _transient_section_keys(
    sections: Sequence[str] | str,
    *,
    registry: Mapping[str, object],
) -> tuple[str, ...]:
    """Validate transient selection against baseline order plus one final extension."""
    if sections == "all":
        return tuple(key for key in TRANSIENT_EVALUATION_SECTION_KEYS if key in registry)
    if isinstance(sections, str) or not isinstance(sections, Sequence) or not sections:
        msg = "Transient sections must be 'all' or a non-empty sequence."
        raise TypeError(msg)
    selected = tuple(sections)
    if any(not isinstance(key, str) for key in selected) or len(set(selected)) != len(selected):
        msg = "Transient sections must contain unique string keys."
        raise ValueError(msg)
    unknown = sorted(set(selected).difference(TRANSIENT_EVALUATION_SECTION_KEYS))
    if unknown:
        msg = f"Unknown transient Evaluation sections: {unknown}."
        raise ValueError(msg)
    unavailable = tuple(key for key in selected if key not in registry)
    if unavailable:
        msg = f"Transient Evaluation sections are unavailable for these artifacts: {list(unavailable)}."
        raise ValueError(msg)
    return selected


def _transient_channel_fields(
    session: TransientEvaluationSession,
) -> tuple[Any, ...]:
    """Return comparable fields from exact session-owned frame provenance."""
    return analysis.evaluation.presentation.transient_channel_resolution(tuple(session.canonical_frames.values())).fields


def _supported_transient_fields(fields: Sequence[Any]) -> tuple[Any, ...]:
    """Retain channels representable by the current admitted sequence arrays."""
    supported = {
        *analysis.evaluation.transient_artifact.STATE_ORDER,
        "w_gr",
    }
    return tuple(field for field in fields if field.key in supported)


def _stored_transient_fields(fields: Sequence[Any]) -> tuple[Any, ...]:
    """Retain stored channels with persisted normalized per-channel metrics."""
    return tuple(field for field in fields if field.stored and field.key in analysis.evaluation.transient_artifact.STATE_ORDER)


def _full_autonomous_summary(
    session: TransientEvaluationSession,
    *,
    frame_name: str,
    case_id: str,
) -> Any:
    """Return the sole complete-rollout summary without opening its NPZ payload."""
    matches = tuple(
        summary
        for summary in session.record_summaries_for_case(frame_name, case_id)
        if summary.mode == "autonomous_full" and summary.requested_horizon == "full"
    )
    if len(matches) != 1:
        msg = "Transient field views require exactly one complete autonomous rollout per case."
        raise ValueError(msg)
    return matches[0]


def _full_autonomous_record(
    session: TransientEvaluationSession,
    *,
    frame_name: str,
    case_id: str,
) -> Any:
    """Load one exact complete-rollout record through the bounded session owner."""
    summary = _full_autonomous_summary(
        session,
        frame_name=frame_name,
        case_id=case_id,
    )
    return session.record_for_coordinates(
        frame_name,
        case_id,
        mode="autonomous_full",
        origin_index=int(summary.origin_index),
        requested_horizon="full",
    )


def _transient_inventory(
    session: TransientEvaluationSession,
    *,
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
) -> tuple[Any, ...]:
    """Return disjoint single-model partitions or comparison-owned case entries."""
    inventory = session.case_inventory() if comparison else session.partitioned_case_inventory()
    if comparison and selection_state is not None and selection_state.capabilities is not None:
        allowed = set(selection_state.capabilities.case_ids)
        inventory = tuple(entry for entry in inventory if entry.case_id in allowed)
    if not inventory:
        msg = "Transient field views require at least one exact case."
        raise ValueError(msg)
    return inventory


def _transient_case_label(entry: Any, *, comparison: bool) -> str:
    """Return the compact EDA material/case label while ownership stays internal."""
    material = analysis.presentation.display_labels.material_display_label(
        entry.material_family,
    )
    parts = (material, entry.frame_name if comparison else None, entry.case_label)
    return " · ".join(part for part in parts if part is not None)


def _compact_transient_case_row(
    inventory: Sequence[Any],
    *,
    comparison: bool,
    initial_position: int,
) -> tuple[widgets.Dropdown, widgets.HBox, widgets.Button, widgets.Button]:
    """Build the exact EDA-width case value and step-button row."""
    if not 0 <= initial_position < len(inventory):
        msg = "Initial transient case position is outside the inventory."
        raise IndexError(msg)
    selector = _dropdown(
        tuple(
            (
                _transient_case_label(entry, comparison=comparison),
                position,
            )
            for position, entry in enumerate(inventory)
        ),
        description="Case:",
        value=initial_position,
    )
    _unused_case, previous, following = analysis.ui.components.ui_step_case_index(
        n_cases=len(inventory),
        start_idx=initial_position,
    )
    row = analysis.ui.components.ui_compact_case_row(
        selector,
        previous,
        following,
    )
    return selector, row, previous, following


def _clear_transient_export(
    export_state: dict[str, Any] | None,
    *,
    plot_name: str | None,
    title: str | None,
) -> None:
    """Clear one stale panel-local figure after a legitimate empty selection."""
    if export_state is None:
        return
    previous = export_state.get("fig")
    if isinstance(previous, Figure):
        plt.close(previous)
    export_state.update(
        {
            "fig": None,
            "figures": (),
            "plot_name": plot_name,
            "title": title,
            "filename_stem": None,
        }
    )


def _transient_channel_state(
    *,
    title: str,
    fields: Sequence[Any],
    callback: Callable[[], None],
    selection_state: EvaluationSelectionState | None,
) -> analysis.ui.components.ChannelCheckboxState:
    """Bind exact three-column EDA channels to local and shared selection state."""
    retained: dict[str, tuple[str, ...] | None] = {"value": None}

    def selected() -> tuple[str, ...] | None:
        if selection_state is not None and selection_state.selection.channels:
            return selection_state.selection.channels
        return retained["value"]

    def install(values: tuple[str, ...]) -> None:
        retained["value"] = values
        if (
            selection_state is not None
            and selection_state.capabilities is not None
            and values
            and set(values).issubset(selection_state.capabilities.channels)
        ):
            selection_state.select_channels(values)

    state = analysis.ui.components.ChannelCheckboxState(
        title=title,
        callback=callback,
        selection_getter=selected,
        selection_setter=install,
    )
    state.rebind(
        tuple(field.key for field in fields),
        labels={field.key: f"{field.label} [{analysis.presentation.field_labels.display_unit(field.unit)}]" for field in fields},
    )
    return state


def _transient_outlier_inventory(
    session: TransientEvaluationSession,
    *,
    fields: Sequence[Any],
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
) -> tuple[Any, ...]:
    """Return the stable union of per-channel worst complete-rollout cases."""
    inventory = _transient_inventory(
        session,
        comparison=comparison,
        selection_state=selection_state,
    )
    by_key = {(entry.frame_name, entry.case_id): entry for entry in inventory}
    frame = session.case_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    ordered: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for field in fields:
        column = f"normalized_rmse_{field.key}" if field.stored else "physical_w_gr_rmse"
        if column not in selected:
            continue
        for row in selected.nlargest(_OUTLIER_COUNT, column).itertuples(index=False):
            key = (str(row.frame), str(row.case_id))
            if key in by_key and key not in seen:
                ordered.append(by_key[key])
                seen.add(key)
    return tuple(ordered) or inventory


def _make_transient_sample_viewer(
    *,
    session: TransientEvaluationSession,
    fields: Sequence[Any],
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
    outlier: bool = False,
) -> _ActivatableVBox:
    """Build the EDA-native selected-case Reference/Prediction/Error view."""
    admitted_fields = _supported_transient_fields(fields)
    if not admitted_fields:
        msg = "Transient sample views require representable channels."
        raise ValueError(msg)
    inventory = (
        _transient_outlier_inventory(
            session,
            fields=admitted_fields,
            comparison=comparison,
            selection_state=selection_state,
        )
        if outlier
        else _transient_inventory(
            session,
            comparison=comparison,
            selection_state=selection_state,
        )
    )
    shared_case = None if selection_state is None else selection_state.selection.case_id
    initial_position = next(
        (position for position, entry in enumerate(inventory) if entry.case_id == shared_case),
        0,
    )
    case_selector, case_row, previous, following = _compact_transient_case_row(
        inventory,
        comparison=comparison,
        initial_position=initial_position,
    )
    status = widgets.HTML()
    scale_lock = analysis.ui.components.ui_checkbox_map_scale_lock(value=True)
    scale_container = widgets.HBox((scale_lock,))
    time_container = widgets.VBox()
    output = analysis.ui.components.ui_output_plot()
    navigator: analysis.ui.time.TimeStepNavigator | None = None
    state = {"updating": False}

    def current_entry() -> Any:
        value = case_selector.value
        if isinstance(value, bool) or not isinstance(value, int):
            msg = "Transient case selector must retain an integer inventory position."
            raise TypeError(msg)
        return inventory[_selected_widget_index(value, option_count=len(inventory))]

    def selected_record() -> Any:
        entry = current_entry()
        return _full_autonomous_record(
            session,
            frame_name=entry.frame_name,
            case_id=entry.case_id,
        )

    def publish(record: Any, physical_time: float, channels: tuple[str, ...]) -> None:
        if selection_state is None:
            return
        entry = current_entry()
        case_ids = tuple(dict.fromkeys(item.case_id for item in inventory))
        selection_state.bind_capabilities(
            analysis.evaluation.selection.EvaluationViewCapabilities(
                task="transient_drying",
                channels=tuple(field.key for field in admitted_fields),
                case_ids=case_ids,
                physical_times=tuple(float(value) for value in record.physical_times),
                protocols=("autonomous_full",),
                horizons=("full",),
            )
        )
        selection_state.select_case(entry.case_id)
        selection_state.select_channels(channels)
        selection_state.select_protocol("autonomous_full")
        selection_state.select_horizon("full")
        selection_state.select_physical_time(physical_time)

    def render() -> None:
        if state["updating"]:
            return
        channels = channel_state.selected
        if not channels:
            _clear_transient_export(
                export_state,
                plot_name=export_plot_name,
                title=export_title,
            )
            with output:
                output.clear_output(wait=True)
                print("Select at least one compatible channel.")
            return
        record = selected_record()
        if outlier:
            physical_time = analysis.evaluation.plots.transient.maximum_error_physical_time(
                record,
                state_fields=channels,
            )
        else:
            if navigator is None:
                msg = "Transient sample time navigation was not initialized."
                raise RuntimeError(msg)
            physical_time = navigator.selection.physical_time
        publish(record, physical_time, channels)
        entry = current_entry()
        context = analysis.presentation.display_labels.material_role_display_label(
            entry.material_family,
            entry.dataset_role,
        )
        suffix = f" · selected-channel error-max time: {physical_time:g} h" if outlier else ""
        status.value = f"<span style='color:#555'><b>{escape(context)}</b> · Exact case: <code>{escape(entry.case_id)}</code>{escape(suffix)}</span>"
        if export_state is not None:
            export_state["filename_stem"] = "__".join(
                (
                    "transient_drying",
                    entry.frame_name,
                    str(export_plot_name or "sample"),
                    f"case_{entry.case_id}",
                    f"channel_{'-'.join(channels)}",
                    f"t_{physical_time:g}h",
                )
            )
        analysis.ui.viewers.render_figure(
            out=output,
            plot_func=analysis.evaluation.plots.transient.plot_state_maps,
            args=(record,),
            kwargs={
                "state_fields": channels,
                "physical_time": physical_time,
                "lock_scale": bool(scale_lock.value),
            },
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    def channels_changed() -> None:
        render()

    channel_state = _transient_channel_state(
        title="Channels",
        fields=admitted_fields,
        callback=channels_changed,
        selection_state=selection_state,
    )

    def rebind_case(*_args: object) -> None:
        nonlocal navigator
        if state["updating"]:
            return
        record = selected_record()
        times = tuple(float(value) for value in record.physical_times)
        preferred = None if selection_state is None else selection_state.selection.physical_time
        initial_time = preferred if preferred in times else times[-1]
        state["updating"] = True
        try:
            if outlier:
                navigator = None
                time_container.children = ()
            elif navigator is None:
                navigator = analysis.ui.time.TimeStepNavigator(
                    times,
                    callback=lambda _selection: render(),
                    initial_time=initial_time,
                )
                time_container.children = (navigator.widget,)
            else:
                navigator.rebind(
                    times,
                    preserve_time=initial_time,
                    notify=False,
                )
        finally:
            state["updating"] = False
        render()

    def step_case(offset: int) -> None:
        current = int(cast("int", case_selector.value))
        case_selector.value = min(max(current + offset, 0), len(inventory) - 1)

    case_selector.observe(rebind_case, names="value")
    previous.on_click(lambda _button: step_case(-1))
    following.on_click(lambda _button: step_case(1))
    scale_lock.observe(lambda _change: render(), names="value")
    rebind_case()
    children: tuple[widgets.Widget, ...] = (
        case_row,
        status,
        channel_state.container,
        scale_container,
        time_container,
        output,
    )
    return _ActivatableVBox(children, activate=rebind_case)


def _transient_material_contexts(
    session: TransientEvaluationSession,
    *,
    comparison: bool,
) -> tuple[tuple[str, tuple[str, str]], ...]:
    """Return material-first labels and exact frame/material coordinates."""
    return tuple(
        (
            " · ".join(
                part
                for part in (
                    analysis.presentation.display_labels.material_role_display_label(
                        material,
                        session.dataset_role(frame_name),
                    ),
                    frame_name if comparison else None,
                )
                if part is not None
            ),
            (frame_name, material),
        )
        for frame_name in session.frame_names
        for material in session.material_families(frame_name)
    )


def _make_transient_aggregate_viewer(
    *,
    session: TransientEvaluationSession,
    fields: Sequence[Any],
    plot_kind: str,
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> _ActivatableVBox:
    """Build one exact-time multi-material aggregate with EDA-native controls."""
    if plot_kind not in {"predicted_reference", "mean_error", "std_error"}:
        msg = "Unknown transient aggregate field-view kind."
        raise ValueError(msg)
    admitted_fields = _supported_transient_fields(fields)
    contexts = _transient_material_contexts(session, comparison=comparison)
    maximum = max(len(session.case_ids(frame_name, material_family=material)) for _label, (frame_name, material) in contexts)
    case_count, fewer, more = analysis.ui.components.ui_step_case_count(
        start_cases=min(50, maximum),
        min_cases=1,
        max_cases=maximum,
        step_size=50,
    )
    count_row, count_detail = analysis.ui.components.ui_compact_scope_controls(None)
    analysis.ui.components.ui_set_scope_detail(
        count_detail,
        (case_count, fewer, more),
    )
    status = widgets.HTML()
    time_container = widgets.VBox()
    output = analysis.ui.components.ui_output_plot()
    active: dict[str, Any] = {"key": None, "series": None}
    navigator: analysis.ui.time.TimeStepNavigator | None = None
    state = {"updating": False}

    def record_series() -> dict[str, tuple[Any, ...]]:
        limit = int(case_count.value)
        if active["key"] != limit:
            series: dict[str, tuple[Any, ...]] = {}
            for label, (frame_name, material) in contexts:
                case_ids = session.case_ids(
                    frame_name,
                    material_family=material,
                )[:limit]
                series[label] = tuple(
                    _full_autonomous_record(
                        session,
                        frame_name=frame_name,
                        case_id=case_id,
                    )
                    for case_id in case_ids
                )
            active["key"] = limit
            active["series"] = series
        value = active["series"]
        if not isinstance(value, dict):
            msg = "Transient aggregate record cache has an invalid type."
            raise TypeError(msg)
        return cast("dict[str, tuple[Any, ...]]", value)

    def render() -> None:
        if state["updating"] or navigator is None:
            return
        channels = channel_state.selected
        if not channels:
            _clear_transient_export(
                export_state,
                plot_name=export_plot_name,
                title=export_title,
            )
            with output:
                output.clear_output(wait=True)
                print("Select at least one compatible channel.")
            return
        series = record_series()
        physical_time = navigator.selection.physical_time
        coverage = "; ".join(f"{label}: {len(records)}" for label, records in series.items())
        status.value = f"<span style='color:#555'><b>All material-role partitions</b> · {escape(coverage)} exact cases</span>"
        if selection_state is not None:
            case_ids = tuple(
                dict.fromkeys(
                    case_id
                    for _label, (frame_name, material) in contexts
                    for case_id in session.case_ids(
                        frame_name,
                        material_family=material,
                    )
                )
            )
            selection_state.bind_capabilities(
                analysis.evaluation.selection.EvaluationViewCapabilities(
                    task="transient_drying",
                    channels=tuple(field.key for field in admitted_fields),
                    case_ids=case_ids,
                    physical_times=tuple(float(value) for value in navigator.physical_times),
                    protocols=("autonomous_full",),
                    horizons=("full",),
                )
            )
            selection_state.select_channels(channels)
            selection_state.select_protocol("autonomous_full")
            selection_state.select_horizon("full")
            selection_state.select_physical_time(physical_time)
        if export_state is not None:
            export_state["filename_stem"] = "__".join(
                (
                    "transient_drying",
                    str(export_plot_name or plot_kind),
                    f"cases_{int(case_count.value)}",
                    f"channel_{'-'.join(channels)}",
                    f"t_{physical_time:g}h",
                    "aggregate",
                )
            )
        plot_func: Callable[..., Figure]
        if plot_kind == "predicted_reference":
            plot_func = analysis.evaluation.plots.transient.plot_predicted_vs_reference_channels
            kwargs = {
                "state_fields": channels,
                "physical_time": physical_time,
            }
        else:
            plot_func = analysis.evaluation.plots.transient.plot_aggregate_error_maps
            kwargs = {
                "state_fields": channels,
                "physical_time": physical_time,
                "statistic": "mean" if plot_kind == "mean_error" else "std",
            }
        analysis.ui.viewers.render_figure(
            out=output,
            plot_func=plot_func,
            args=(series,),
            kwargs=kwargs,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    channel_state = _transient_channel_state(
        title="Channels",
        fields=admitted_fields,
        callback=render,
        selection_state=selection_state,
    )

    def rebind(*_args: object) -> None:
        nonlocal navigator
        series = record_series()
        records = tuple(record for current in series.values() for record in current)
        times = _shared_transient_physical_times(records)
        preferred = None if selection_state is None else selection_state.selection.physical_time
        initial_time = preferred if preferred in times else times[-1]
        state["updating"] = True
        try:
            if navigator is None:
                navigator = analysis.ui.time.TimeStepNavigator(
                    times,
                    callback=lambda _selection: render(),
                    initial_time=initial_time,
                )
                time_container.children = (navigator.widget,)
            else:
                navigator.rebind(
                    times,
                    preserve_time=initial_time,
                    notify=False,
                )
        finally:
            state["updating"] = False
        render()

    def step_count(offset: int) -> None:
        case_count.value = max(
            case_count.min,
            min(
                case_count.max,
                case_count.value + offset * case_count.step,
            ),
        )

    case_count.observe(rebind, names="value")
    fewer.on_click(lambda _button: step_count(-1))
    more.on_click(lambda _button: step_count(1))
    rebind()
    return _ActivatableVBox(
        (
            count_row,
            status,
            channel_state.container,
            time_container,
            output,
        ),
        activate=rebind,
    )


def _make_transient_scope_viewer(  # noqa: C901, PLR0915 -- one coordinated EDA scope composition
    *,
    session: TransientEvaluationSession,
    fields: Sequence[Any],
    plot_kind: str,
    comparison: bool,
    selection_state: EvaluationSelectionState | None = None,
    export_state: dict[str, Any] | None = None,
    export_plot_name: str | None = None,
    export_title: str | None = None,
) -> _ActivatableVBox:
    """Build one local Aggregate/Single case trajectory-style view."""
    if plot_kind not in {"trajectory", "error_time", "target_magnitude"}:
        msg = "Unknown transient scoped view kind."
        raise ValueError(msg)
    admitted_fields = _stored_transient_fields(fields) if plot_kind == "error_time" else _supported_transient_fields(fields)
    inventory = _transient_inventory(
        session,
        comparison=comparison,
        selection_state=selection_state,
    )
    shared_case = None if selection_state is None else selection_state.selection.case_id
    initial_position = next(
        (position for position, entry in enumerate(inventory) if entry.case_id == shared_case),
        0,
    )
    case_selector, case_row, previous, following = _compact_transient_case_row(
        inventory,
        comparison=comparison,
        initial_position=initial_position,
    )
    contexts = _transient_material_contexts(session, comparison=comparison)
    maximum = min(len(session.case_ids(frame_name, material_family=material)) for _label, (frame_name, material) in contexts)
    case_count, fewer, more = analysis.ui.components.ui_step_case_count(
        start_cases=min(100, maximum),
        min_cases=1,
        max_cases=maximum,
        step_size=50,
    )
    initial_scope = selection_state.selection.scope if selection_state is not None else "aggregate"
    scope = analysis.ui.components.ui_scope_toggle(value=initial_scope)
    scope_row, scope_detail = analysis.ui.components.ui_compact_scope_controls(scope)
    status = widgets.HTML()
    output = analysis.ui.components.ui_output_plot()
    active: dict[str, Any] = {
        "key": None,
        "series": None,
        "scaling_states": None,
    }
    state = {"updating": False}

    def current_entry() -> Any:
        value = case_selector.value
        if isinstance(value, bool) or not isinstance(value, int):
            msg = "Transient scoped case selector must retain an integer position."
            raise TypeError(msg)
        return inventory[
            _selected_widget_index(
                value,
                option_count=len(inventory),
            )
        ]

    def selected_series() -> tuple[
        dict[str, tuple[Any, ...]],
        dict[str, Mapping[str, Any]],
    ]:
        if scope.value == "single":
            entry = current_entry()
            key: tuple[Any, ...] = (
                "single",
                entry.frame_name,
                entry.case_id,
            )
        else:
            key = ("aggregate", int(case_count.value))
        if active["key"] != key:
            if scope.value == "single":
                entry = current_entry()
                label = _transient_case_label(
                    entry,
                    comparison=comparison,
                )
                series = {
                    label: (
                        _full_autonomous_record(
                            session,
                            frame_name=entry.frame_name,
                            case_id=entry.case_id,
                        ),
                    )
                }
                scaling_states = {label: session.scaling_state(entry.frame_name)}
            else:
                limit = int(case_count.value)
                series = {}
                scaling_states = {}
                for label, (frame_name, material) in contexts:
                    case_ids = session.case_ids(
                        frame_name,
                        material_family=material,
                    )[:limit]
                    series[label] = tuple(
                        _full_autonomous_record(
                            session,
                            frame_name=frame_name,
                            case_id=case_id,
                        )
                        for case_id in case_ids
                    )
                    scaling_states[label] = session.scaling_state(frame_name)
            active["key"] = key
            active["series"] = series
            active["scaling_states"] = scaling_states
        series_value = active["series"]
        scaling_value = active["scaling_states"]
        if not isinstance(series_value, dict) or not isinstance(
            scaling_value,
            dict,
        ):
            msg = "Transient scoped record cache has an invalid type."
            raise TypeError(msg)
        return (
            cast("dict[str, tuple[Any, ...]]", series_value),
            cast("dict[str, Mapping[str, Any]]", scaling_value),
        )

    def render() -> None:
        if state["updating"]:
            return
        channels = channel_state.selected
        if not channels:
            _clear_transient_export(
                export_state,
                plot_name=export_plot_name,
                title=export_title,
            )
            with output:
                output.clear_output(wait=True)
                print("Select at least one compatible channel.")
            return
        series, scaling_states = selected_series()
        if scope.value == "single":
            entry = current_entry()
            status.value = (
                f"<span style='color:#555'><b>{escape(_transient_case_label(entry, comparison=comparison))}</b> · "
                f"Exact case: <code>{escape(entry.case_id)}</code></span>"
            )
        else:
            status.value = (
                f"<span style='color:#555'><b>All material-role partitions</b> · up to {int(case_count.value)} exact cases per partition</span>"
            )
        if selection_state is not None:
            case_ids = tuple(dict.fromkeys(entry.case_id for entry in inventory))
            selection_state.bind_capabilities(
                analysis.evaluation.selection.EvaluationViewCapabilities(
                    task="transient_drying",
                    channels=tuple(field.key for field in admitted_fields),
                    case_ids=case_ids,
                    protocols=("autonomous_full",),
                    horizons=("full",),
                )
            )
            selection_state.select_scope(cast("Any", scope.value))
            selection_state.select_channels(channels)
            if scope.value == "single":
                selection_state.select_case(current_entry().case_id)
            selection_state.select_protocol("autonomous_full")
            selection_state.select_horizon("full")
        if export_state is not None:
            case_part = f"case_{current_entry().case_id}" if scope.value == "single" else f"cases_{int(case_count.value)}"
            export_state["filename_stem"] = "__".join(
                (
                    "transient_drying",
                    str(export_plot_name or plot_kind),
                    str(scope.value),
                    case_part,
                    f"channel_{'-'.join(channels)}",
                )
            )
        plot_func: Callable[..., Figure]
        if plot_kind == "trajectory":
            plot_func = analysis.evaluation.plots.transient.plot_state_trajectory_summary
            kwargs: dict[str, Any] = {"state_fields": channels}
        elif plot_kind == "error_time":
            plot_func = analysis.evaluation.plots.transient.plot_error_over_physical_time
            kwargs = {
                "scaling_states": scaling_states,
                "state_fields": channels,
            }
        else:
            plot_func = analysis.evaluation.plots.transient.plot_error_vs_target_magnitude
            kwargs = {"state_fields": channels}
        analysis.ui.viewers.render_figure(
            out=output,
            plot_func=plot_func,
            args=(series,),
            kwargs=kwargs,
            export_state=export_state,
            export_plot_name=export_plot_name,
            export_title=export_title,
        )

    channel_state = _transient_channel_state(
        title=("Stored channels" if plot_kind == "error_time" else "Channels"),
        fields=admitted_fields,
        callback=render,
        selection_state=selection_state,
    )

    def update_scope_detail() -> None:
        if scope.value == "aggregate":
            analysis.ui.components.ui_set_scope_detail(
                scope_detail,
                (case_count, fewer, more),
            )
        else:
            analysis.ui.components.ui_set_scope_detail(
                scope_detail,
                tuple(case_row.children),
            )

    def scope_changed(_change: object) -> None:
        update_scope_detail()
        render()

    def step_case(offset: int) -> None:
        current = int(cast("int", case_selector.value))
        case_selector.value = min(max(current + offset, 0), len(inventory) - 1)

    def step_count(offset: int) -> None:
        case_count.value = max(
            case_count.min,
            min(case_count.max, case_count.value + offset * case_count.step),
        )

    scope.observe(scope_changed, names="value")
    case_selector.observe(lambda _change: render(), names="value")
    previous.on_click(lambda _button: step_case(-1))
    following.on_click(lambda _button: step_case(1))
    case_count.observe(lambda _change: render(), names="value")
    fewer.on_click(lambda _button: step_count(-1))
    more.on_click(lambda _button: step_count(1))
    update_scope_detail()
    render()
    return _ActivatableVBox(
        (
            scope_row,
            status,
            channel_state.container,
            output,
        ),
        activate=render,
    )


def _transient_summary_table(session: TransientEvaluationSession) -> pd.DataFrame:
    """Return a compact aggregate table without exposing identity hashes."""
    frame = session.dataset_dataframe()
    columns = tuple(
        column
        for column in (
            "frame",
            "material_family",
            "dataset_role",
            "mode",
            "requested_horizon",
            "scope",
            "contributing_record_count",
            "normalized_drying_group_macro_rmse",
            *(f"normalized_rmse_{field}" for field in analysis.evaluation.transient_artifact.STATE_ORDER),
        )
        if column in frame
    )
    return frame.loc[:, list(columns)]


def _transient_comparison_summary(session: TransientEvaluationSession) -> pd.DataFrame:
    """Return one ranked full-rollout model comparison scoreboard."""
    frame = session.dataset_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    columns = (
        "normalized_drying_group_macro_rmse",
        *(f"normalized_rmse_{field}" for field in analysis.evaluation.transient_artifact.STATE_ORDER),
    )
    return selected.groupby("frame", sort=False)[list(columns)].mean().sort_values("normalized_drying_group_macro_rmse").reset_index()


def _transient_architecture_table(session: TransientEvaluationSession) -> pd.DataFrame:
    """Return exact model-family configuration from compact record identity."""
    rows = []
    for frame_name in session.frame_names:
        case_id = session.case_ids(frame_name)[0]
        identity = _full_autonomous_summary(
            session,
            frame_name=frame_name,
            case_id=case_id,
        ).identity
        if not isinstance(identity, Mapping):
            msg = "Transient architecture presentation requires record identity."
            raise TypeError(msg)
        parameters = identity.get("model_parameters")
        rows.append(
            {
                "Artifact": frame_name,
                "Model family": identity.get("model_kind"),
                "Configuration": (dict(parameters) if isinstance(parameters, Mapping) else "Unavailable"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _transient_outlier_table(
    session: TransientEvaluationSession,
    *,
    fields: Sequence[Any],
) -> pd.DataFrame:
    """Return per-channel worst complete-rollout cases from persisted metrics."""
    frame = session.case_dataframe(modes=("autonomous_full",))
    selected = frame.loc[(frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")]
    rows: list[dict[str, Any]] = []
    for field in fields:
        column = f"normalized_rmse_{field.key}" if field.stored else "physical_w_gr_rmse"
        if column not in selected:
            continue
        for rank, row in enumerate(
            selected.nlargest(_OUTLIER_COUNT, column).itertuples(index=False),
            start=1,
        ):
            rows.append(
                {
                    "Channel": field.label,
                    "Rank": rank,
                    "Material · Role": analysis.presentation.display_labels.material_role_display_label(
                        str(row.material_family),
                        str(row.dataset_role),
                    ),
                    "Artifact": str(row.frame),
                    "Exact case": str(row.case_id),
                    "Error": float(getattr(row, column)),
                    "Error unit": "1" if field.stored else field.unit,
                }
            )
    if not rows:
        msg = "Transient outlier tables require at least one per-channel metric."
        raise ValueError(msg)
    return pd.DataFrame.from_records(rows)


def _transient_temporal_capabilities(
    session: TransientEvaluationSession,
) -> tuple[bool, bool, bool]:
    """Return full-trajectory, rollout-horizon, and final-target availability."""
    summaries = tuple(
        summary
        for frame_name in session.frame_names
        for case_id in session.case_ids(frame_name)
        for summary in session.record_summaries_for_case(frame_name, case_id)
    )
    full = any(summary.mode == "autonomous_full" and summary.requested_horizon == "full" for summary in summaries)
    horizon = any(
        summary.mode in {"rolling_origin", "autonomous_full"}
        and isinstance(summary.requested_horizon, int)
        and not isinstance(summary.requested_horizon, bool)
        for summary in summaries
    )
    target = any(
        isinstance(summary.target, Mapping)
        and (summary.target.get("reference_available") is True or summary.target.get("predicted_available") is True)
        for summary in summaries
        if summary.mode == "autonomous_full" and summary.requested_horizon == "full"
    )
    return full, horizon, target


TRANSIENT_PLOT_CLASSIFICATION = {
    "Overview: Summary table": "AGGREGATE_ONLY",
    "Overview: Global comparison summary": "AGGREGATE_ONLY",
    "1-1. Global error metrics": "AGGREGATE_ONLY",
    "1-2. Global error distribution": "AGGREGATE_ONLY",
    "1-3. GT vs Prediction (mean)": "AGGREGATE_ONLY",
    "1-4. Mean error maps": "AGGREGATE_ONLY",
    "1-5. Std error maps": "AGGREGATE_ONLY",
    "2-1. Model-family configuration": "AGGREGATE_ONLY",
    "3-1. Error vs |GT| magnitude": "SINGLE_AND_AGGREGATE",
    "7-1. Sample GT vs Prediction": "SINGLE_ONLY",
    "8-1. Worst per-channel cases (tables)": "AGGREGATE_ONLY",
    "8-2. Worst per-channel cases (field plots)": "SINGLE_ONLY",
    "9-1. Reference vs prediction trajectories": "SINGLE_AND_AGGREGATE",
    "9-2. Error over physical time": "SINGLE_AND_AGGREGATE",
    "9-3. Error vs rollout horizon": "AGGREGATE_ONLY",
    "9-4. Final-state drying summary": "AGGREGATE_ONLY",
}


def _build_transient_sections(
    *,
    session: TransientEvaluationSession,
    comparison: bool,
    selection_state: EvaluationSelectionState | None,
) -> dict[str, tuple[list[tuple[str, Callable[..., Any], str]], str]]:
    """Bind compatible transient evidence under the established Evaluation sections."""
    fields = _transient_channel_fields(session)
    supported_fields = _supported_transient_fields(fields)
    stored_fields = _stored_transient_fields(fields)
    plots = analysis.evaluation.plots.transient

    def summary_table() -> pd.DataFrame:
        return _transient_summary_table(session)

    def comparison_summary() -> pd.DataFrame:
        return _transient_comparison_summary(session)

    def global_metrics() -> Figure:
        return plots.plot_channel_error(
            session.dataset_dataframe(),
            state_fields=tuple(field.key for field in stored_fields),
        )

    def distributions() -> Figure:
        return plots.plot_error_distributions(
            session.case_dataframe(modes=("autonomous_full",)),
            state_fields=tuple(field.key for field in stored_fields),
        )

    def architecture_table() -> pd.DataFrame:
        return _transient_architecture_table(session)

    def outlier_table() -> pd.DataFrame:
        return _transient_outlier_table(
            session,
            fields=supported_fields,
        )

    def scoped_view(plot_kind: str) -> Callable[..., widgets.Widget]:
        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> widgets.Widget:
            return _make_transient_scope_viewer(
                session=session,
                fields=supported_fields,
                plot_kind=plot_kind,
                comparison=comparison,
                selection_state=selection_state,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )

        return build

    def aggregate_view(plot_kind: str) -> Callable[..., widgets.Widget]:
        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> widgets.Widget:
            return _make_transient_aggregate_viewer(
                session=session,
                fields=supported_fields,
                plot_kind=plot_kind,
                comparison=comparison,
                selection_state=selection_state,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
            )

        return build

    def sample_view(outlier: bool) -> Callable[..., widgets.Widget]:
        def build(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> widgets.Widget:
            return _make_transient_sample_viewer(
                session=session,
                fields=supported_fields,
                comparison=comparison,
                selection_state=selection_state,
                export_state=export_state,
                export_plot_name=export_plot_name,
                export_title=export_title,
                outlier=outlier,
            )

        return build

    overview_entries = [
        (
            "Overview: Summary table",
            summary_table,
            "overview_summary_table",
        )
    ]
    if comparison:
        overview_entries.append(
            (
                "Overview: Global comparison summary",
                comparison_summary,
                "overview_global_comparison_summary",
            )
        )
    registry: dict[
        str,
        tuple[list[tuple[str, Callable[..., Any], str]], str],
    ] = {
        "overview": (
            overview_entries,
            EVALUATION_SECTION_TITLES["overview"],
        ),
        "global_error": (
            [
                (
                    "1-1. Global error metrics",
                    global_metrics,
                    "1_1_global_error_metrics",
                ),
                (
                    "1-2. Global error distribution",
                    distributions,
                    "1_2_global_error_distribution",
                ),
                (
                    "1-3. GT vs Prediction (mean)",
                    aggregate_view("predicted_reference"),
                    "1_3_gt_vs_prediction_mean",
                ),
                (
                    "1-4. Mean error maps",
                    aggregate_view("mean_error"),
                    "1_4_mean_error_maps",
                ),
                (
                    "1-5. Std error maps",
                    aggregate_view("std_error"),
                    "1_5_std_error_maps",
                ),
            ],
            EVALUATION_SECTION_TITLES["global_error"],
        ),
        "architecture": (
            [
                (
                    "2-1. Model-family configuration",
                    architecture_table,
                    "2_1_model_family_configuration",
                )
            ],
            EVALUATION_SECTION_TITLES["architecture"],
        ),
        "error_decomposition": (
            [
                (
                    "3-1. Error vs |GT| magnitude",
                    scoped_view("target_magnitude"),
                    "3_1_error_vs_gt_magnitude",
                )
            ],
            EVALUATION_SECTION_TITLES["error_decomposition"],
        ),
        "sample_viewer": (
            [
                (
                    "7-1. Sample GT vs Prediction",
                    sample_view(False),
                    "7_1_sample_gt_vs_prediction",
                )
            ],
            EVALUATION_SECTION_TITLES["sample_viewer"],
        ),
        "outlier_analysis": (
            [
                (
                    "8-1. Worst per-channel cases (tables)",
                    outlier_table,
                    "8_1_worst_per_channel_tables",
                ),
                (
                    "8-2. Worst per-channel cases (field plots)",
                    sample_view(True),
                    "8_2_worst_per_channel_fields",
                ),
            ],
            EVALUATION_SECTION_TITLES["outlier_analysis"],
        ),
    }
    has_full, has_horizon, has_target = _transient_temporal_capabilities(session)
    temporal_entries: list[tuple[str, Callable[..., Any], str]] = []
    if has_full:
        temporal_entries.extend(
            (
                (
                    "9-1. Reference vs prediction trajectories",
                    scoped_view("trajectory"),
                    "9_1_reference_prediction_trajectories",
                ),
                (
                    "9-2. Error over physical time",
                    scoped_view("error_time"),
                    "9_2_error_over_physical_time",
                ),
            )
        )
    if has_horizon:
        temporal_entries.append(
            (
                "9-3. Error vs rollout horizon",
                lambda: plots.plot_horizon_error(session.dataset_dataframe()),
                "9_3_error_vs_rollout_horizon",
            )
        )
    if has_target:
        temporal_entries.append(
            (
                "9-4. Final-state drying summary",
                lambda: plots.plot_target_time(session.full_autonomous_summaries()),
                "9_4_final_state_drying_summary",
            )
        )
    if temporal_entries:
        registry["temporal_rollout"] = (
            temporal_entries,
            EVALUATION_SECTION_TITLES["temporal_rollout"],
        )
    return registry


def build_transient_panel(
    *,
    session: TransientEvaluationSession,
    sections: Sequence[str] | str = "all",
    comparison: bool = False,
    training_performance: Callable[[], pd.DataFrame | None] | None = None,
    selection_state: EvaluationSelectionState | None = None,
) -> widgets.Widget:
    """Build the baseline Evaluation architecture with transient capabilities."""
    del training_performance
    if not session.frame_names:
        msg = "Transient panel requires one open session with frames."
        raise ValueError(msg)
    registry = _build_transient_sections(
        session=session,
        comparison=comparison,
        selection_state=selection_state,
    )
    selected = _transient_section_keys(sections, registry=registry)
    if not selected:
        msg = "The selected transient artifacts provide no approved Evaluation sections."
        raise ValueError(msg)
    export_state = {
        "fig": None,
        "figures": (),
        "plot_name": None,
        "title": None,
        "filename_stem": None,
        "filename_prefix": "transient_drying",
    }
    ui_sections = [
        analysis.ui.notebook.make_dropdown_section(
            registry[key][0],
            export_state=export_state,
            select_first=True,
        )
        for key in selected
    ]
    return analysis.ui.notebook.make_lazy_panel_with_tabs(
        ui_sections,
        tab_titles=[registry[key][1] for key in selected],
        open_btn_text=("Model comparison - Open Evaluation" if comparison else "Single model - Open Evaluation"),
        close_btn_text="Close",
        export_state=export_state,
        export_dir="",
        export_btn_text="Export PDF",
    )
