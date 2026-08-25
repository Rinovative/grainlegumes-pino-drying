"""
evaluation_workflow.py

Prepare portable evaluation sessions and panels from concise run selections.

Responsibilities:
  - Normalize explicit current run paths and optional presentation labels
  - Invoke shared artifact load-or-build orchestration for selected roles
  - Compose session-owned role contexts for single-run or comparison panels
  - Return concise lifecycle, checkpoint, artifact, and device reporting

Design principles:
  - Notebook cells expose choices while reusable orchestration remains in source
  - Scientific run identity and mutable storage identity remain visibly separate
  - Presentation labels never participate in artifact or numerical identity
  - Failed preparation closes any session created before the failure

This module does NOT:
  - Admit runs, reconstruct models, generate payloads, or publish artifacts itself
  - Choose datasets, roles, labels, sections, devices, or rebuild policy for users
  - Contact tracking services or mutate run lifecycle evidence
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd

from src import analysis, common

if TYPE_CHECKING:
    import ipywidgets as widgets

    from src.analysis.artifacts.analysis_artifact_service import PreparedRunArtifacts
    from src.analysis.evaluation.evaluation_artifact_loader import LoadedEvaluationArtifact, LoadedRunArtifacts
    from src.analysis.evaluation.evaluation_panel import EvaluationContext
    from src.analysis.evaluation.evaluation_selection import EvaluationProtocol, EvaluationSelectionState
    from src.analysis.evaluation.evaluation_session import EvaluationSession
    from src.analysis.evaluation.evaluation_transient_comparison import LineageComparison
    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession

ArtifactSelectionRole = Literal["id", "ood"]
_TRAINING_COMPARISON_ARM_COUNT = 3
_MINIMUM_COMPARISON_RUNS = 2


@dataclass(frozen=True, slots=True)
class EvaluationRunSelection:
    """
    Select one current run directory and optional presentation label.

    Parameters
    ----------
    run_dir : pathlib.Path | str
        Exact current direct-run or Optuna-trial directory.
    label : str | None, optional
        Presentation-only label. The storage alias is used when omitted.

    """

    run_dir: Path | str
    label: str | None = None
    artifact_roots: Mapping[ArtifactSelectionRole, Path | str] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationContextSpec:
    """
    Describe one ordered dataset context backed by an artifact role.

    Parameters
    ----------
    key : str
        Stable panel-local context key.
    label : str
        User-facing dataset context label.
    artifact_role : {"id", "ood"}
        Selected run artifact role supplying each model frame.

    """

    key: str
    label: str
    artifact_role: ArtifactSelectionRole


@dataclass(frozen=True, slots=True)
class PreparedEvaluationWorkflow:
    """Hold task-aware prepared artifacts, live session, panel, and report evidence."""

    prepared_runs: tuple[PreparedRunArtifacts, ...]
    task: str
    session: EvaluationSession | TransientEvaluationSession
    contexts: tuple[EvaluationContext, ...]
    panel: widgets.Widget
    report: tuple[dict[str, Any], ...]
    summary_text: str
    transient_lineage: LineageComparison | None = None
    _transient_performance_loader: Callable[[], pd.DataFrame | None] | None = None

    @property
    def transient_performance(self) -> pd.DataFrame | None:
        """Load and cache matched-compute performance only when explicitly requested."""
        if self._transient_performance_loader is None:
            return None
        return self._transient_performance_loader()

    def close(self) -> None:
        """Release all session-owned case and numerical state."""
        self.session.close()

    def render_report(self, output_dir: Path | str) -> Any:
        """Render the task-owned local report bundle below a caller-owned directory."""
        if self.task == "transient_drying":
            return analysis.presentation.curated.render_curated_transient_analysis(
                session=cast("TransientEvaluationSession", self.session),
                output_dir=output_dir,
                training_performance=self.transient_performance,
            )
        datasets = {f"{context.label} / {model_label}": frame for context in self.contexts for model_label, frame in context.datasets.items()}
        return analysis.presentation.curated.render_curated_analysis(
            datasets=datasets,
            output_dir=output_dir,
        )


def _normalized_selections(
    run_selections: Sequence[EvaluationRunSelection],
) -> tuple[tuple[EvaluationRunSelection, Path], ...]:
    """Validate selections and resolve explicit current paths."""
    if isinstance(run_selections, (str, bytes)) or not isinstance(run_selections, Sequence) or not run_selections:
        msg = "run_selections must be a non-empty ordered sequence."
        raise TypeError(msg)
    normalized: list[tuple[EvaluationRunSelection, Path]] = []
    for position, selection in enumerate(run_selections):
        if not isinstance(selection, EvaluationRunSelection):
            msg = f"run_selections[{position}] must be an EvaluationRunSelection."
            raise TypeError(msg)
        if selection.label is not None and (not isinstance(selection.label, str) or not selection.label.strip()):
            msg = f"run_selections[{position}].label must be non-blank text or None."
            raise ValueError(msg)
        roots = {} if selection.artifact_roots is None else dict(selection.artifact_roots)
        if set(roots).difference({"id", "ood"}):
            msg = f"run_selections[{position}].artifact_roots contains an unsupported role."
            raise ValueError(msg)
        normalized.append(
            (
                selection,
                Path(selection.run_dir).expanduser().resolve(),
            )
        )
    return tuple(normalized)


def _normalized_context_specs(
    context_specs: Sequence[EvaluationContextSpec],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...],
) -> tuple[EvaluationContextSpec, ...]:
    """Validate ordered context declarations and retain selected roles."""
    if isinstance(context_specs, (str, bytes)) or not isinstance(context_specs, Sequence) or not context_specs:
        msg = "context_specs must be a non-empty ordered sequence."
        raise TypeError(msg)
    normalized: list[EvaluationContextSpec] = []
    for position, spec in enumerate(context_specs):
        if not isinstance(spec, EvaluationContextSpec):
            msg = f"context_specs[{position}] must be an EvaluationContextSpec."
            raise TypeError(msg)
        if not spec.key.strip() or not spec.label.strip():
            msg = f"context_specs[{position}] requires non-blank key and label text."
            raise ValueError(msg)
        if spec.artifact_role in artifact_roles:
            normalized.append(spec)
    if not normalized:
        msg = "No evaluation context selects one of the requested artifact roles."
        raise ValueError(msg)
    keys = tuple(spec.key for spec in normalized)
    labels = tuple(spec.label for spec in normalized)
    if len(keys) != len(set(keys)) or len(labels) != len(set(labels)):
        msg = "Selected evaluation context keys and labels must each be unique."
        raise ValueError(msg)
    return tuple(normalized)


def _selected_artifact(
    loaded: LoadedRunArtifacts,
    role: ArtifactSelectionRole,
) -> LoadedEvaluationArtifact:
    """Return one role guaranteed by artifact preparation."""
    artifact = loaded.id_artifact if role == "id" else loaded.ood_artifact
    if artifact is None:
        msg = f"Prepared run is missing its selected {role!r} artifact: {loaded.run_dir}"
        raise RuntimeError(msg)
    return artifact


def _display_label(
    selection: EvaluationRunSelection,
    loaded: LoadedRunArtifacts,
) -> str:
    """Return the presentation label with truthful provisional status."""
    base_label = selection.label.strip() if selection.label is not None else loaded.storage_alias
    return f"{base_label} [provisional]" if loaded.is_provisional else base_label


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one required mapping from admitted transient provenance."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _required_text(value: Any, *, label: str) -> str:
    """Return one required non-empty identity string."""
    if not isinstance(value, str) or not value:
        msg = f"{label} must be non-empty text."
        raise TypeError(msg)
    return value


def _required_number(value: Any, *, label: str) -> float:
    """Return one real matched-compute value without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be one real compute value."
        raise TypeError(msg)
    return float(value)


def _required_nonnegative_int(value: Any, *, label: str) -> int:
    """Return one exact non-negative persisted counter."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{label} must be one non-negative integer."
        raise TypeError(msg)
    return value


def _transient_arm_evidence(loaded: LoadedRunArtifacts) -> Any:
    """Extract one strict training-arm comparison record from admitted ID evidence."""
    artifact = _selected_artifact(loaded, "id")
    provenance = _required_mapping(artifact.frame.attrs.get("artifact_provenance"), label="transient artifact provenance")
    lineage = _required_mapping(provenance.get("lineage"), label="transient lineage")
    stage = _required_mapping(lineage.get("stage_identity"), label="transient stage identity")
    arm_id = _required_text(stage.get("comparison_arm"), label="transient comparison arm")
    arm = {"a0": "A0", "a_plus": "A+", "b": "B"}.get(arm_id)
    if arm is None:
        msg = f"Unsupported transient comparison arm {arm_id!r}."
        raise ValueError(msg)
    records = artifact.frame.attrs.get("transient_sequence_records")
    index = artifact.frame.attrs.get("transient_sequence_index")
    if isinstance(records, tuple) and records:
        identity_evidence = records[0].identity
    elif (
        isinstance(
            index,
            analysis.evaluation.transient_artifact.TransientSequenceArtifactIndex,
        )
        and index.summaries
    ):
        identity_evidence = index.summaries[0].identity
    else:
        msg = "Transient comparison requires admitted sequence evidence."
        raise TypeError(msg)
    identity = _required_mapping(identity_evidence, label="transient sequence identity")
    parent = _required_mapping(lineage.get("parent_checkpoint"), label="transient parent checkpoint")
    dataset = _required_mapping(provenance.get("dataset"), label="transient Dataset provenance")
    scaling = _required_mapping(identity.get("scaling_identity"), label="transient scaling identity")
    matched = _required_mapping(lineage.get("matched_compute_manifest"), label="transient matched-compute manifest")
    controller = _required_mapping(matched.get("actual"), label="transient terminal controller")
    clock_kind = _required_text(controller.get("clock_kind"), label="transient compute clock")
    expected_controller = {
        "A0": ("A0", "stage_a_teacher_forcing", "stage_epochs"),
        "A+": ("A+", "stage_a_teacher_forcing", "matched_compute"),
        "B": ("B", "stage_b_self_fed", "matched_compute"),
    }[arm]
    observed_controller = (
        controller.get("arm"),
        controller.get("stage"),
        controller.get("budget_control"),
    )
    if observed_controller != expected_controller:
        msg = f"{arm} terminal controller identity is incompatible."
        raise ValueError(msg)
    best_epoch = _required_nonnegative_int(
        controller.get("best_within_budget_epoch"),
        label=f"{arm} best-within-budget epoch",
    )
    if controller.get("budget_complete") is not True or loaded.selected_checkpoint_epoch != best_epoch + 1:
        msg = f"{arm} selected checkpoint does not match completed best-within-budget evidence."
        raise ValueError(msg)
    best_checkpoint_id = f"{loaded.selected_checkpoint_sha256}@epoch={loaded.selected_checkpoint_epoch}"
    if arm == "A0":
        planned_epochs = _required_nonnegative_int(
            controller.get("planned_stage_epochs"),
            label="A0 planned stage epochs",
        )
        completed_epochs = _required_nonnegative_int(
            controller.get("completed_stage_epochs"),
            label="A0 completed stage epochs",
        )
        if planned_epochs < 1 or completed_epochs != planned_epochs:
            msg = "A0 must complete its persisted Stage-A epoch budget before comparison."
            raise ValueError(msg)
        planned = 0.0
        actual = 0.0
        overrun = 0.0
    else:
        planned_key = "planned_teacher_forcing_budget_seconds" if clock_kind == "cuda_device_seconds" else "planned_teacher_forcing_budget_steps"
        actual_key = "post_handoff_optimizer_device_seconds" if clock_kind == "cuda_device_seconds" else "successful_optimizer_steps"
        planned = _required_number(controller.get(planned_key), label=f"{arm} planned compute")
        actual = _required_number(controller.get(actual_key), label=f"{arm} actual compute")
        overrun = actual - planned
    budget_complete = True
    comparison = analysis.evaluation.transient_comparison
    return comparison.TrainingArmEvidence(
        arm=arm,
        parent_id=common.serialization.canonical_json_sha256(parent),
        architecture_id=common.serialization.canonical_json_sha256(
            {
                "kind": identity.get("model_kind"),
                "parameters": identity.get("model_parameters"),
            }
        ),
        dataset_id=common.serialization.canonical_json_sha256(
            {
                "name": dataset.get("name"),
                "source_dataset_ids": dataset.get("source_dataset_ids"),
            }
        ),
        split_id=common.serialization.canonical_json_sha256(
            {
                "source_identities": dataset.get("source_identities"),
                "membership_digests": dataset.get("membership_digests"),
            }
        ),
        input_profile_id=_required_text(identity.get("input_profile"), label="transient input profile"),
        scaling_id=_required_text(scaling.get("semantic_digest"), label="transient scaling digest"),
        boundary_representation_id=_required_text(
            identity.get("boundary_representation"),
            label="transient boundary representation",
        ),
        strategy=_required_text(lineage.get("training_strategy"), label="transient training strategy"),
        planned_post_handoff_compute=planned,
        actual_post_handoff_compute=actual,
        safe_boundary_overrun=overrun,
        clock_kind=clock_kind,
        budget_complete=budget_complete,
        best_within_budget_checkpoint_id=best_checkpoint_id,
    )


def _transient_role_membership(artifact: LoadedEvaluationArtifact) -> Any:
    """Extract exact transient dataset and case membership from one admitted artifact."""
    provenance = _required_mapping(artifact.frame.attrs.get("artifact_provenance"), label="transient artifact provenance")
    dataset = _required_mapping(provenance.get("dataset"), label="transient Dataset provenance")
    source_dataset_ids = dataset.get("source_dataset_ids")
    membership_digests = dataset.get("membership_digests")
    records = artifact.frame.attrs.get("transient_sequence_records")
    index = artifact.frame.attrs.get("transient_sequence_index")
    if (
        not isinstance(source_dataset_ids, list)
        or not all(isinstance(value, str) and value for value in source_dataset_ids)
        or not isinstance(membership_digests, list)
        or not all(isinstance(value, str) and value for value in membership_digests)
    ):
        msg = "Transient artifact has malformed shared-membership evidence."
        raise ValueError(msg)
    if isinstance(records, tuple) and records:
        case_ids = tuple(
            sorted(
                {
                    _required_text(
                        getattr(record, "case_id", None),
                        label="transient sequence case ID",
                    )
                    for record in records
                }
            )
        )
    elif isinstance(
        index,
        analysis.evaluation.transient_artifact.TransientSequenceArtifactIndex,
    ):
        case_ids = tuple(sorted(index.case_ids))
    else:
        msg = "Transient artifact has no admitted sequence inventory."
        raise ValueError(msg)
    return analysis.evaluation.transient_comparison.SharedRoleMembership(
        dataset_name=_required_text(dataset.get("name"), label="transient dataset name"),
        source_dataset_ids=tuple(source_dataset_ids),
        membership_digests=tuple(membership_digests),
        case_ids=case_ids,
    )


def _validate_transient_shared_membership(
    loaded_runs: Sequence[LoadedRunArtifacts],
    *,
    roles: Sequence[ArtifactSelectionRole],
) -> None:
    """Require exact role membership before any transient multi-run comparison panel."""
    if len(loaded_runs) < _MINIMUM_COMPARISON_RUNS:
        return
    comparison = analysis.evaluation.transient_comparison
    for role in roles:
        comparison.validate_shared_role_membership(
            role=role,
            memberships=tuple(_transient_role_membership(_selected_artifact(loaded, role)) for loaded in loaded_runs),
        )


def _transient_lineage(loaded_runs: Sequence[LoadedRunArtifacts]) -> Any | None:
    """Validate an exact selected A0/A+/B set while leaving other comparisons generic."""
    if len(loaded_runs) != _TRAINING_COMPARISON_ARM_COUNT:
        return None
    arms = tuple(_transient_arm_evidence(loaded) for loaded in loaded_runs)
    by_name = {arm.arm: arm for arm in arms}
    if set(by_name) != {"A0", "A+", "B"}:
        return None
    return analysis.evaluation.transient_comparison.validate_lineage_comparison(
        a0=by_name["A0"],
        a_plus=by_name["A+"],
        b=by_name["B"],
    )


def _transient_performance_frame(
    session: TransientEvaluationSession,
    *,
    loaded_runs: Sequence[LoadedRunArtifacts],
    display_labels: Sequence[str],
    lineage: LineageComparison | None,
) -> pd.DataFrame | None:
    """Attach factual arm and compute evidence to every transient ID summary row."""
    if lineage is None:
        return None
    summaries = session.dataset_dataframe()
    frames: list[pd.DataFrame] = []
    for loaded, display_label in zip(loaded_runs, display_labels, strict=True):
        evidence = _transient_arm_evidence(loaded)
        selected = summaries.loc[summaries["frame"] == f"{display_label} ID"].copy()
        if selected.empty:
            msg = f"Transient comparison lacks ID summaries for {display_label!r}."
            raise RuntimeError(msg)
        selected["comparison_arm"] = evidence.arm
        selected["training_strategy"] = evidence.strategy
        selected["planned_post_handoff_compute"] = evidence.planned_post_handoff_compute
        selected["optimizer_device_compute"] = evidence.actual_post_handoff_compute
        selected["safe_boundary_overrun"] = evidence.safe_boundary_overrun
        selected["compute_clock"] = evidence.clock_kind
        selected["checkpoint_role"] = "stage_a_handoff" if evidence.arm == "A0" else "best_within_budget"
        selected["checkpoint_identity"] = evidence.best_within_budget_checkpoint_id
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    result.attrs["primary_comparison"] = lineage.primary_comparison
    result.attrs["separate_comparison"] = lineage.separate_comparison
    return result


def _lazy_transient_performance(
    session: TransientEvaluationSession,
    *,
    loaded_runs: Sequence[LoadedRunArtifacts],
    display_labels: Sequence[str],
    lineage: LineageComparison | None,
) -> Callable[[], pd.DataFrame | None] | None:
    """Return one cached explicit loader without opening sequence payloads."""
    if lineage is None:
        return None

    @cache
    def load() -> pd.DataFrame | None:
        return _transient_performance_frame(
            session,
            loaded_runs=loaded_runs,
            display_labels=display_labels,
            lineage=lineage,
        )

    return load


def _report_row(
    prepared: PreparedRunArtifacts,
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...],
    display_label: str,
) -> dict[str, Any]:
    """Build one concise JSON-friendly notebook report row."""
    loaded = prepared.loaded_run
    artifacts = {role: _selected_artifact(loaded, role) for role in artifact_roles}
    return {
        "display_label": display_label,
        "scientific_run_name": loaded.scientific_run_name,
        "storage_alias": loaded.storage_alias,
        "run_dir": str(loaded.run_dir),
        "lifecycle_status": loaded.lifecycle_status,
        "is_completed": loaded.is_completed,
        "is_provisional": loaded.is_provisional,
        "selected_checkpoint_role": loaded.selected_checkpoint_role,
        "selected_checkpoint_epoch": loaded.selected_checkpoint_epoch,
        "selected_checkpoint_sha256": loaded.selected_checkpoint_sha256,
        "artifact_roles": list(artifact_roles),
        "artifact_roots": {role: str(artifact.root) for role, artifact in artifacts.items()},
        "artifact_actions": dict(prepared.role_actions),
        "artifact_device": prepared.artifact_device,
    }


def _require_shared_case_ids(case_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return one non-empty exact shared case inventory."""
    if not case_ids:
        msg = "Selected Evaluation artifacts share no exact case identities."
        raise ValueError(msg)
    return case_ids


def _transient_selection_case_ids(
    session: TransientEvaluationSession,
    *,
    display_labels: Sequence[str],
    roles: Sequence[ArtifactSelectionRole],
    single_model: bool,
) -> tuple[str, ...]:
    """Return a partitioned union for one run or role-local intersections for models."""
    if single_model:
        return tuple(entry.case_id for entry in session.partitioned_case_inventory())
    selected: list[str] = []
    for role in roles:
        frame_names = tuple(f"{label} {role.upper()}" for label in display_labels)
        shared = set(session.case_ids(frame_names[0]))
        for frame_name in frame_names[1:]:
            shared.intersection_update(session.case_ids(frame_name))
        selected.extend(_require_shared_case_ids(tuple(case_id for case_id in session.case_ids(frame_names[0]) if case_id in shared)))
    if len(selected) != len(set(selected)):
        msg = "Comparison artifact roles contain ambiguous duplicate exact case identities."
        raise ValueError(msg)
    return tuple(selected)


def _bind_steady_capabilities(
    selection_state: EvaluationSelectionState,
    artifacts: Mapping[tuple[Path, ArtifactSelectionRole], LoadedEvaluationArtifact],
) -> None:
    """Bind exact steady channels and shared case membership once."""
    frames = tuple(artifact.frame for artifact in artifacts.values())
    display_fields = analysis.evaluation.presentation.shared_display_fields(frames)
    shared_cases = {int(value) for value in frames[0]["case_index"]}
    for frame in frames[1:]:
        shared_cases.intersection_update(int(value) for value in frame["case_index"])
    selection_state.bind_capabilities(
        analysis.evaluation.selection.EvaluationViewCapabilities(
            task="steady_flow",
            channels=tuple(field.key for field in display_fields),
            case_ids=_require_shared_case_ids(tuple(str(value) for value in sorted(shared_cases))),
        )
    )


def prepare_evaluation_workflow(
    run_selections: Sequence[EvaluationRunSelection],
    context_specs: Sequence[EvaluationContextSpec],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...] = ("id", "ood"),
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
    auto_build_missing: bool = True,
    rebuild_incompatible: bool = False,
    device_policy: str = "cpu",
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
) -> PreparedEvaluationWorkflow:
    """Prepare one steady or transient Evaluation workflow through task-owned sessions."""
    normalized_selections = _normalized_selections(run_selections)
    requested_roles = tuple(artifact_roles)
    prepared_runs = tuple(
        analysis.artifacts.service.load_or_build_run_artifacts(
            path,
            artifact_roles=requested_roles,
            artifact_root_overrides=selection.artifact_roots,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
            auto_build_missing=auto_build_missing,
            rebuild_incompatible=rebuild_incompatible,
            device_policy=device_policy,
        )
        for selection, path in normalized_selections
    )
    loaded_runs = tuple(prepared.loaded_run for prepared in prepared_runs)
    tasks = {loaded.task for loaded in loaded_runs}
    if len(tasks) != 1:
        msg = "One Evaluation workflow cannot mix task identities."
        raise ValueError(msg)
    task = next(iter(tasks))
    if task not in {"steady_flow", "transient_drying"}:
        msg = f"Unsupported Evaluation task {task!r}."
        raise ValueError(msg)

    effective_roles: list[ArtifactSelectionRole] = []
    for role in requested_roles:
        available = tuple((loaded.id_artifact if role == "id" else loaded.ood_artifact) is not None for loaded in loaded_runs)
        if any(available) and not all(available):
            msg = f"Selected runs disagree on {role!r} artifact-role availability."
            raise ValueError(msg)
        if all(available):
            effective_roles.append(role)
    roles = tuple(effective_roles)
    if not roles:
        msg = "No requested Evaluation artifact role is available across every selected run."
        raise ValueError(msg)
    specs = _normalized_context_specs(context_specs, artifact_roles=roles)

    display_labels = tuple(_display_label(selection, loaded) for (selection, _path), loaded in zip(normalized_selections, loaded_runs, strict=True))
    if len(display_labels) != len(set(display_labels)):
        msg = "Evaluation presentation labels must be unique after provisional labelling."
        raise ValueError(msg)
    artifacts: dict[tuple[Path, ArtifactSelectionRole], LoadedEvaluationArtifact] = {
        (loaded.run_dir, role): _selected_artifact(loaded, role) for loaded in loaded_runs for role in roles
    }
    contexts = tuple(
        analysis.evaluation.panel.EvaluationContext(
            key=spec.key,
            label=spec.label,
            datasets={
                display_label: artifacts[(loaded.run_dir, spec.artifact_role)].frame
                for display_label, loaded in zip(display_labels, loaded_runs, strict=True)
            },
        )
        for spec in specs
    )

    lineage = None
    transient_performance_loader: Callable[[], pd.DataFrame | None] | None = None
    session: EvaluationSession | TransientEvaluationSession
    if task == "transient_drying":
        session = analysis.evaluation.transient_session.TransientEvaluationSession(
            {
                f"{display_label} {role.upper()}": artifacts[(loaded.run_dir, role)].frame
                for display_label, loaded in zip(display_labels, loaded_runs, strict=True)
                for role in roles
            }
        )
    else:
        session = analysis.evaluation.session.EvaluationSession(
            {f"{loaded.run_dir}:{role}": artifacts[(loaded.run_dir, role)].frame for loaded in loaded_runs for role in roles}
        )
    try:
        if task == "transient_drying":
            transient_session = cast("TransientEvaluationSession", session)
            _validate_transient_shared_membership(loaded_runs, roles=roles)
            lineage = _transient_lineage(loaded_runs)
            transient_performance_loader = _lazy_transient_performance(
                transient_session,
                loaded_runs=loaded_runs,
                display_labels=display_labels,
                lineage=lineage,
            )
            if selection_state is not None:
                ordered_cases = _transient_selection_case_ids(
                    transient_session,
                    display_labels=display_labels,
                    roles=roles,
                    single_model=len(loaded_runs) == 1,
                )
                indexes = tuple(artifact.frame.attrs.get("transient_sequence_index") for artifact in artifacts.values())
                summaries = tuple(
                    summary
                    for index in indexes
                    if isinstance(
                        index,
                        analysis.evaluation.transient_artifact.TransientSequenceArtifactIndex,
                    )
                    for summary in index.summaries
                )
                protocols = cast(
                    "tuple[EvaluationProtocol, ...]",
                    tuple(dict.fromkeys(summary.mode for summary in summaries)),
                )
                horizons = tuple(dict.fromkeys(summary.requested_horizon for summary in summaries))
                channel_resolution = analysis.evaluation.presentation.transient_channel_resolution(
                    tuple(artifact.frame for artifact in artifacts.values())
                )
                selection_state.bind_capabilities(
                    analysis.evaluation.selection.EvaluationViewCapabilities(
                        task="transient_drying",
                        channels=channel_resolution.keys,
                        case_ids=ordered_cases,
                        protocols=protocols,
                        horizons=horizons,
                    )
                )
            panel = analysis.evaluation.panel.build_transient_panel(
                session=transient_session,
                sections=sections,
                comparison=len(loaded_runs) > 1,
                training_performance=transient_performance_loader,
                selection_state=selection_state,
            )
        elif len(loaded_runs) == 1:
            if selection_state is not None:
                _bind_steady_capabilities(selection_state, artifacts)
            panel = analysis.evaluation.panel.build_single_model_panel(
                session=cast("EvaluationSession", session),
                contexts=contexts,
                sections=sections,
                selection_state=selection_state,
            )
        else:
            if selection_state is not None:
                _bind_steady_capabilities(selection_state, artifacts)
            panel = analysis.evaluation.panel.build_comparison_panel(
                session=cast("EvaluationSession", session),
                contexts=contexts,
                sections=sections,
                selection_state=selection_state,
            )
        report = tuple(
            _report_row(prepared, artifact_roles=roles, display_label=display_label)
            for prepared, display_label in zip(prepared_runs, display_labels, strict=True)
        )
    except Exception:
        session.close()
        raise
    return PreparedEvaluationWorkflow(
        prepared_runs=prepared_runs,
        task=task,
        session=session,
        contexts=contexts,
        panel=panel,
        report=report,
        summary_text="",
        transient_lineage=lineage,
        _transient_performance_loader=transient_performance_loader,
    )


def _workflow_summary(workflow: PreparedEvaluationWorkflow) -> str:
    """Build concise plain Evaluation preparation evidence for notebook display."""
    roles = ", ".join(workflow.report[0]["artifact_roles"])
    rows = [
        f"- {row['display_label']}: scientific run {row['scientific_run_name']}; "
        f"{row['lifecycle_status']}; checkpoint {row['selected_checkpoint_role']}; actions {row['artifact_actions']}"
        for row in workflow.report
    ]
    header = (
        f"Evaluation workspace summary\n\nTask: {workflow.task}\nSelected runs: {len(workflow.prepared_runs)}\nArtifact roles: {roles}\n\nRuns:\n"
    )
    return header + "\n".join(rows)


def _validated_expected_task(expected_task: str) -> str:
    """Return one supported task expectation before artifact preparation starts."""
    if expected_task not in {"steady_flow", "transient_drying"}:
        msg = f"Evaluation workspace expected_task is unsupported: {expected_task!r}."
        raise ValueError(msg)
    return expected_task


def _default_context_specs() -> tuple[EvaluationContextSpec, EvaluationContextSpec]:
    """Return the task-owned ID and optional OOD notebook contexts."""
    return (
        EvaluationContextSpec("id", "ID", "id"),
        EvaluationContextSpec("ood", "OOD", "ood"),
    )


def _with_expected_task(workflow: PreparedEvaluationWorkflow, *, expected_task: str) -> PreparedEvaluationWorkflow:
    """Bind one notebook workspace to its expected saved task and close on mismatch."""
    if workflow.task != expected_task:
        workflow.close()
        msg = f"Evaluation workspace expected task {expected_task!r}, got {workflow.task!r}."
        raise ValueError(msg)
    return replace(
        workflow,
        summary_text=_workflow_summary(workflow),
    )


def prepare_single_model_evaluation_workspace(
    run_dir: Path | str,
    *,
    label: str | None = None,
    expected_task: str = "transient_drying",
    artifact_roles: tuple[ArtifactSelectionRole, ...] = ("id", "ood"),
    artifact_roots: Mapping[ArtifactSelectionRole, Path | str] | None = None,
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
    auto_build_missing: bool = True,
    rebuild_incompatible: bool = False,
    device_policy: str = "cpu",
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
) -> PreparedEvaluationWorkflow:
    """Prepare one notebook-facing Evaluation workspace with task-owned contexts."""
    task = _validated_expected_task(expected_task)
    workflow = prepare_evaluation_workflow(
        (
            EvaluationRunSelection(
                run_dir=run_dir,
                label=label,
                artifact_roots=artifact_roots,
            ),
        ),
        _default_context_specs(),
        artifact_roles=artifact_roles,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        auto_build_missing=auto_build_missing,
        rebuild_incompatible=rebuild_incompatible,
        device_policy=device_policy,
        sections=sections,
        selection_state=selection_state,
    )
    return _with_expected_task(workflow, expected_task=task)


def prepare_model_comparison_evaluation_workspace(
    run_dirs: Sequence[Path | str],
    *,
    labels: Sequence[str | None] | None = None,
    expected_task: str = "transient_drying",
    artifact_roles: tuple[ArtifactSelectionRole, ...] = ("id", "ood"),
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
    auto_build_missing: bool = True,
    rebuild_incompatible: bool = False,
    device_policy: str = "cpu",
    sections: Sequence[str] | str = "all",
    selection_state: EvaluationSelectionState | None = None,
) -> PreparedEvaluationWorkflow:
    """Prepare an ordered multi-run notebook workspace with task-owned contexts."""
    task = _validated_expected_task(expected_task)
    if isinstance(run_dirs, (str, bytes)) or not isinstance(run_dirs, Sequence) or len(run_dirs) < _MINIMUM_COMPARISON_RUNS:
        msg = "Model-comparison Evaluation workspace requires at least two ordered run paths."
        raise ValueError(msg)
    if isinstance(labels, (str, bytes)):
        msg = "Model-comparison labels must be an ordered sequence or None."
        raise TypeError(msg)
    resolved_labels = (None,) * len(run_dirs) if labels is None else tuple(labels)
    if len(resolved_labels) != len(run_dirs):
        msg = "Model-comparison labels must match the ordered run-path count."
        raise ValueError(msg)
    selections = tuple(EvaluationRunSelection(run_dir=run_dir, label=label) for run_dir, label in zip(run_dirs, resolved_labels, strict=True))
    workflow = prepare_evaluation_workflow(
        selections,
        _default_context_specs(),
        artifact_roles=artifact_roles,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
        auto_build_missing=auto_build_missing,
        rebuild_incompatible=rebuild_incompatible,
        device_policy=device_policy,
        sections=sections,
        selection_state=selection_state,
    )
    return _with_expected_task(workflow, expected_task=task)
