"""Generate and validate transient sequence artifacts through current project owners.

Responsibilities:
  - Reopen the parts of a shared artifact role plan and enforce their saved identity and complete-case membership
  - Generate bounded case inference, metric and timing evidence through maintained owners
  - Delegate sequence staging and validate canonical or explicit selected-case artifacts
  - Bind role-level prediction-validity evidence to exact role membership
  - Stamp the running analysis producer and resume a partial stage only under the analysis code that started it

Design principles:
  - Saved membership order and training identity bind every artifact role
  - Training and Evaluation spatial representations remain explicit
  - Scoped subsets never claim complete canonical publication eligibility

This module does NOT:
  - Implement models, scaling or alternative Dataset storage readers
  - Publish the final run-owned target or contact tracking services
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from src import common, datasets, domain, experiments, generation, learning
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_transient_artifact as sequence_artifact
from src.analysis.evaluation import evaluation_transient_metrics as transient_metrics
from src.analysis.evaluation import evaluation_transient_rollout as rollout
from src.analysis.evaluation import evaluation_transient_timing as transient_timing
from src.analysis.evaluation import evaluation_transient_validity as transient_validity
from src.datasets.packages import dataset_packages_generated_batch as generated_batch
from src.experiments import experiments_matched_budget as matched_budget
from src.experiments import experiments_run_identity as run_identity
from src.learning.transient import learning_transient_handoff as transient_handoff
from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

from . import analysis_artifact_contracts as contracts
from . import analysis_artifact_performance as artifact_performance
from . import analysis_artifact_plan as artifact_plan
from . import analysis_artifact_timing as artifact_timing

if TYPE_CHECKING:
    from src.datasets.runtime.dataset_runtime_transient import TransientPhysicalDataset
    from src.learning.learning_device import DeviceResolution

_SOURCE_SELECTOR_KEYS = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)
_TIMING_REPETITIONS = 3
_RESUME_EVIDENCE_SCHEMA_KIND = "evaluation_transient_resume_case"
# Per-case component evidence the generator writes next to its timing report; resume and role admission both require exactly these fields.
_COMPONENT_TEXT_FIELDS = ("model_clock", "wall_clock")
_COMPONENT_MAPPING_FIELDS = (
    "generation",
    "timing_case",
    "prediction_validity",
    "diagnostic_benchmark",
    "airflow_model",
    "drying_model",
    "surrogate_pipeline",
)
_COMPONENT_EVIDENCE_FIELDS = frozenset({"dataset_materialization_seconds", *_COMPONENT_TEXT_FIELDS, *_COMPONENT_MAPPING_FIELDS})
_CANONICAL_COMPLETION_CONTEXT_FIELDS = frozenset(
    {
        "final_bulk_moisture_wb",
        "target_moisture_wb",
    }
)


def _generation_target_completion(completion: Mapping[str, Any]) -> dict[str, Any]:
    """Project Generation-owned target evidence from one canonical case record."""
    return {field: value for field, value in completion.items() if field not in _CANONICAL_COMPLETION_CONTEXT_FIELDS}


def _process_diagnostic_policy() -> dict[str, Any]:
    """Return the immutable process-diagnostic availability and reduction policy."""
    return {
        "bulk_moisture": {
            "available": True,
            "cell_weighting": transient_metrics.BULK_MOISTURE_CELL_WEIGHTING,
            "invalid_state_policy": transient_metrics.BULK_MOISTURE_INVALID_POLICY,
        },
        "temperature_plausibility_range_kelvin": list(transient_metrics.TEMPERATURE_PLAUSIBILITY_RANGE_K),
        "stability_increment_growth_factor": transient_metrics.STABILITY_GROWTH_FACTOR,
        "mass_balance": {
            "available": False,
            "reason": "sequence_artifact_lacks_gas_water_storage_and_boundary_mass_flux_series",
        },
    }


def _reject_trial(completed: Mapping[str, Any]) -> None:
    """Refuse Optuna trial evidence through the shared artifact refusal of both tasks."""
    contracts.refuse_optuna_trial(completed.get("config"), label=f"Run {completed.get('scientific_run_name')}")


def _require_completed_transient(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require already-admitted completed transient-run evidence without reopening files."""
    config = value.get("config")
    if not isinstance(config, Mapping) or config.get("task") != "transient_drying":
        message = "Transient artifact generation requires admitted completed transient_drying evidence."
        raise RuntimeError(message)
    _reject_trial(value)
    return value


def _completed_transient(run_dir: Path | str) -> dict[str, Any]:
    """Admit one completed transient run and reject static bundles."""
    completed = experiments.run.validate_completed_run(run_dir)
    _require_completed_transient(completed)
    return completed


def load_transient_artifact_plan(run_dir: Path | str) -> artifact_plan.ArtifactPlan:
    """Admit a completed transient run and return its artifact role plan."""
    path = Path(run_dir).expanduser().resolve()
    return artifact_plan.artifact_plan_from_completed(_completed_transient(path), run_dir=path)


def select_transient_role_cases(
    role: artifact_plan.ArtifactRolePlan,
    case_ids: Sequence[str],
) -> artifact_plan.ArtifactRolePlan:
    """Return one saved-order role subset for disposable scoped generation."""
    selected = tuple(case_ids)
    if not selected or any(not isinstance(case_id, str) or not case_id for case_id in selected) or len(selected) != len(set(selected)):
        message = "Scoped transient case IDs must be unique non-empty strings."
        raise ValueError(message)
    available = {case_id for source_case_ids in role.case_ids_by_source for case_id in source_case_ids}
    missing = tuple(case_id for case_id in selected if case_id not in available)
    if missing:
        message = f"Scoped transient cases are absent from saved {role.dataset_role} membership: {missing}."
        raise ValueError(message)
    requested = set(selected)
    groups = tuple(tuple(case_id for case_id in source_case_ids if case_id in requested) for source_case_ids in role.case_ids_by_source)
    return replace(role, case_ids_by_source=groups)


def _training_spatial_representation(
    completed: Mapping[str, Any],
) -> datasets.contracts.transient.TransientSpatialRepresentation:
    """Reconcile the resolved Training stride, the admitted split's spatial view and the fitted scaling shape."""
    _require_completed_transient(completed)
    config = cast("Mapping[str, Any]", completed["config"])
    data = config.get("data")
    normalizer_state = completed.get("normalizer_state")
    if not isinstance(data, Mapping) or not isinstance(normalizer_state, Mapping):
        message = "Transient artifact grid preflight requires config and scaling evidence."
        raise TypeError(message)
    stride = datasets.contracts.transient.validate_spatial_stride(data["spatial_stride"])
    split = datasets.preprocessing.splits.admit_split(completed["split_indices"])
    view = cast("Mapping[str, Any]", split.extension)["spatial_view"]
    if view["spatial_stride"] != stride:
        message = "Resolved Training stride contradicts the admitted split spatial view."
        raise ValueError(message)
    representation = datasets.contracts.transient.resolve_spatial_representation((view["canonical_ny"], view["canonical_nx"]), stride)
    represented_shape = (view["effective_ny"], view["effective_nx"])
    scaling_shape = TransientScalingArtifact.from_state_dict(normalizer_state).spatial_shape
    if representation.represented_shape != represented_shape or scaling_shape != represented_shape:
        message = "Training-grid shape contradicts the split spatial view or the fitted scaling shape."
        raise ValueError(message)
    return representation


def resolve_transient_artifact_spatial_representations(
    completed: Mapping[str, Any],
    *,
    evaluation_spatial_stride: int = 1,
) -> tuple[
    datasets.contracts.transient.TransientSpatialRepresentation,
    datasets.contracts.transient.TransientSpatialRepresentation,
]:
    """Resolve authoritative Training and operator-selected Evaluation grids."""
    training = _training_spatial_representation(completed)
    evaluation = datasets.contracts.transient.resolve_spatial_representation(
        training.source_shape,
        datasets.contracts.transient.validate_spatial_stride(evaluation_spatial_stride),
    )
    return training, evaluation


def _load_role_datasets(
    role: artifact_plan.ArtifactRolePlan,
    *,
    storage_root: Path,
    spatial_representation: datasets.contracts.transient.TransientSpatialRepresentation,
    saved_role: artifact_plan.ArtifactRolePlan | None = None,
) -> tuple[TransientPhysicalDataset, ...]:
    """
    Recreate one-step physical Datasets, check each full package against its saved split part, then select its cases.

    ``saved_role`` is the complete saved role of a scoped ``role`` subset; each package is checked against its full
    saved membership, never against the subset.
    """
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    loaded: list[TransientPhysicalDataset] = []
    for dataset_id, expected_identity, saved_case_ids, case_ids in zip(
        role.source_dataset_ids,
        role.source_identities,
        (role if saved_role is None else saved_role).case_ids_by_source,
        role.case_ids_by_source,
        strict=True,
    ):
        if not case_ids:
            continue
        request = datasets.runtime.factory.DatasetRequest(
            dataset_id=dataset_id,
            dataset_view="transient_drying",
            transient_sampling=sampling,
            storage_root=storage_root,
            spatial_stride=spatial_representation.spatial_stride,
        )
        candidate = datasets.runtime.factory.create_dataset(request)
        if not isinstance(candidate, datasets.runtime.transient.TransientPhysicalDataset):
            message = "Transient artifact Dataset factory returned a non-transient runtime."
            raise TypeError(message)
        # The full package is checked before selection, so a drifted Dataset is refused like a steady one.
        try:
            datasets.preprocessing.splits.require_source_identity(candidate, expected_identity, role=role.split_role, case_ids=saved_case_ids)
            selected = datasets.runtime.transient.select_transient_cases(candidate, case_ids)
        except BaseException:
            candidate.close()
            raise
        if selected.spatial_representation() != spatial_representation:
            selected.close()
            message = f"Current package {dataset_id!r} cannot materialize the requested exact Evaluation grid."
            raise RuntimeError(message)
        selected_case_ids = tuple(
            dict.fromkeys(
                str(selected.payload["cases"][int(selected.payload["samples"][position]["case_index"])]["package_case_id"])
                for position in selected.sample_indices
            )
        )
        if selected_case_ids != case_ids:
            selected.close()
            message = f"Current package {dataset_id!r} contradicts saved complete-case membership."
            raise RuntimeError(message)
        loaded.append(selected)
    return tuple(loaded)


@dataclass(frozen=True, slots=True)
class _TransientCaseMaterialization:
    """Bind one compact selected-case plan to its lazy Dataset positions."""

    dataset: TransientPhysicalDataset
    package_case_id: str
    case_record: Mapping[str, Any]
    item_positions: tuple[int, ...]
    material_family: str
    pt_identity: str


def _case_materialization_specs(
    sources: Sequence[TransientPhysicalDataset],
) -> tuple[_TransientCaseMaterialization, ...]:
    """Build exact selected-case plans without loading numerical item payloads."""
    result: list[_TransientCaseMaterialization] = []
    for dataset in sources:
        positions: dict[str, list[int]] = defaultdict(list)
        case_records = {str(case_record["package_case_id"]): case_record for case_record in dataset.payload["cases"]}
        for item_index, sample_id in enumerate(dataset.runtime_item_ids()):
            separator = "__step_"
            if separator not in sample_id:
                message = "One-step transient runtime item identity lacks its package-case prefix."
                raise RuntimeError(message)
            package_case_id = sample_id.rsplit(separator, maxsplit=1)[0]
            if package_case_id not in case_records:
                message = f"Runtime item {sample_id!r} does not map to its package case record."
                raise RuntimeError(message)
            positions[package_case_id].append(item_index)
        pt_identity = common.serialization.canonical_json_sha256(
            datasets.packages.transient_shards.load_transient_shard_receipt(
                dataset.dataset_id,
                storage_root=dataset.source_root,
                validation_depth="evidence",
            )
        )
        for package_case_id, item_positions in positions.items():
            case_record = case_records[package_case_id]
            material = case_record.get("material_family")
            if not isinstance(material, str) or not material:
                message = f"Transient package case {package_case_id!r} lacks material_family."
                raise TypeError(message)
            result.append(
                _TransientCaseMaterialization(
                    dataset=dataset,
                    package_case_id=package_case_id,
                    case_record=case_record,
                    item_positions=tuple(item_positions),
                    material_family=material,
                    pt_identity=pt_identity,
                )
            )
    if not result:
        message = "Saved transient artifact membership selected no complete cases."
        raise RuntimeError(message)
    return tuple(result)


def _materialize_case(
    spec: _TransientCaseMaterialization,
    *,
    dataset_role: sequence_artifact.DatasetRole,
) -> tuple[str, Mapping[str, Any], rollout.TransientEvaluationCase, str]:
    """Load and assemble one selected complete case from its compact plan."""
    materialization_started = perf_counter()
    items = [spec.dataset[position] for position in spec.item_positions]
    materialization_seconds = perf_counter() - materialization_started
    assembled = rollout.assemble_transient_evaluation_case(
        items,
        dataset_role=dataset_role,
    )
    metadata = {
        **dict(assembled.metadata),
        "dataset_materialization_seconds": materialization_seconds,
    }
    return (
        spec.package_case_id,
        spec.case_record,
        rollout.TransientEvaluationCase(
            case_id=spec.package_case_id,
            dataset_role=assembled.dataset_role,
            physical_times=assembled.physical_times,
            reference_states=assembled.reference_states,
            static_conditioning=assembled.static_conditioning,
            boundary_conditioning=assembled.boundary_conditioning,
            scalar_conditioning=assembled.scalar_conditioning,
            spatial_mask=assembled.spatial_mask,
            metadata=metadata,
        ),
        spec.pt_identity,
    )


def _generation_case_sources(
    role: artifact_plan.ArtifactRolePlan,
    *,
    storage_root: Path,
) -> dict[str, Any]:
    """Re-admit saved package members through their exact current Source selectors."""
    result: dict[str, Any] = {}
    for dataset_id, selected_case_ids in zip(role.source_dataset_ids, role.case_ids_by_source, strict=True):
        manifest = datasets.packages.load_package_manifest_evidence("transient_drying", dataset_id, storage_root=storage_root)
        entries = manifest.get("source_case_identities")
        if not isinstance(entries, list):
            message = "Transient Dataset manifest lacks current Source case identities."
            raise TypeError(message)
        by_case = {entry["package_case_id"]: entry for entry in entries}
        if len(by_case) != len(entries):
            message = "Transient Dataset manifest repeats a package case identity."
            raise ValueError(message)
        for package_case_id in selected_case_ids:
            if package_case_id in result or package_case_id not in by_case:
                message = f"Transient artifact package case has no unique saved Source identity: {package_case_id!r}."
                raise ValueError(message)
            entry = by_case[package_case_id]
            source = entry["source"]
            admitted = generation.publication.catalog.resolve_raw_case_reference(
                "transient_drying",
                {key: source[key] for key in _SOURCE_SELECTOR_KEYS},
                storage_root=storage_root,
            )
            if admitted.source_identity != source or admitted.design_provenance != entry["design"]:
                message = f"Transient package Source or design evidence drifted for {package_case_id!r}."
                raise ValueError(message)
            result[package_case_id] = admitted
    return result


def _canonical_case_evidence(
    case: rollout.TransientEvaluationCase,
    *,
    package_case_id: str,
    case_record: Mapping[str, Any],
    generation_sources: Mapping[str, Any],
    spatial_representation: datasets.contracts.transient.TransientSpatialRepresentation | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-check Dataset items against canonical Generation case semantics."""
    admitted = generation_sources.get(package_case_id)
    if admitted is None:
        message = f"Transient package case lacks its exact admitted Source identity: {package_case_id!r}."
        raise ValueError(message)
    source = admitted.source_identity
    if (
        case_record.get("source_hdf5_sha256") != source["case_hdf5_sha256"]
        or case_record.get("case_input_id") != source["case_input_id"]
        or case_record.get("simulation_case_id") != source["simulation_case_id"]
        or case_record.get("source_batch_id") != source["source_batch_id"]
        or case_record.get("source_relative_path") != source["case_hdf5_relative"]
    ):
        message = f"Transient package case contradicts its admitted Source identity: {package_case_id!r}."
        raise ValueError(message)
    task = domain.tasks.registry.get_task("transient_drying")
    canonical = generated_batch.interpret_admitted_transient_source_case(admitted, task=task)
    canonical_reference = np.stack(
        [canonical["state_trajectories"][name] for name in sequence_artifact.STATE_ORDER],
        axis=1,
    )
    canonical_static = np.stack(
        [canonical["static_fields"][name] for name in sequence_artifact.STATIC_ORDER],
        axis=0,
    )
    representation = spatial_representation
    if representation is None:
        representation = datasets.contracts.transient.resolve_spatial_representation(
            cast("tuple[int, int]", canonical_reference.shape[-2:]),
            1,
        )
    if canonical_reference.shape[-2:] != representation.source_shape or canonical_static.shape[-2:] != representation.source_shape:
        message = f"Canonical Generation grid contradicts artifact source-grid identity for {package_case_id!r}."
        raise RuntimeError(message)
    reference = np.take(
        np.take(canonical_reference, representation.y_indices, axis=-2),
        representation.x_indices,
        axis=-1,
    )
    static = np.take(
        np.take(canonical_static, representation.y_indices, axis=-2),
        representation.x_indices,
        axis=-1,
    )
    boundary = np.stack(
        [canonical["boundary_intervals"][name] for name in sequence_artifact.BOUNDARY_ORDER],
        axis=1,
    )
    scalars = np.asarray(
        [canonical["scalar_conditioning"][name] for name in sequence_artifact.SCALAR_ORDER],
        dtype=np.float32,
    )
    times = np.asarray(canonical["time"]["regular_state_hours"], dtype=np.float64)
    comparisons = (
        (reference, case.reference_states, "absolute states"),
        (static, case.static_conditioning, "static conditioning"),
        (boundary, case.boundary_conditioning, "boundary intervals"),
        (scalars, case.scalar_conditioning, "material scalars"),
        (times, case.physical_times, "physical times"),
    )
    for observed, expected, label in comparisons:
        if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=0.0, atol=2.0e-5):
            message = f"Transient Dataset and canonical Generation {label} disagree for {package_case_id!r}."
            raise RuntimeError(message)
    return canonical, canonical["runtime"]


def _matched_budget_receipt(completed: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a matched B/A1 leaf to the immutable B receipt and common A0 manifest."""
    config = cast("Mapping[str, Any]", completed["config"])
    run_dir = cast("Path", completed["run_dir"])
    training = cast("Mapping[str, Any]", config["training"])
    protocol = cast("Mapping[str, Any]", training["matched_a1_b"])
    parent_evidence = _parent_experiment_evidence(completed)
    if parent_evidence["kind"] != "grouped":
        msg = "Matched branch artifact requires its validated grouped parent experiment."
        raise ValueError(msg)
    record_path = run_dir.parent / "experiment.json"
    if record_path.is_symlink() or not record_path.is_file():
        msg = "Matched branch parent experiment record is absent or unsafe."
        raise FileNotFoundError(msg)
    output_root = common.paths.resolve_config_output_root(config, run_dir)
    located = run_identity.load_readonly_transient_experiment_record(record_path, experiments_root=output_root)
    record = located["record"]
    if record.get("topology") != "a0_to_b_and_a1":
        msg = "Matched branch artifact requires the declared A0 matched-branch topology."
        raise ValueError(msg)
    children = cast("Mapping[str, Mapping[str, Any]]", record["children"])
    child_paths = {
        role: run_identity.resolve_readonly_experiment_child(record, child, marker=record_path, output_root=output_root)
        for role, child in children.items()
    }
    a0_run = child_paths["stage_a0"]
    b_run = child_paths["stage_b"]
    manifest_path = a0_run / transient_handoff.HANDOFF_DIRECTORY / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        msg = "Matched branch common A0 handoff manifest is absent or unsafe."
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_run_name") != parent.get("source_run_name") or manifest.get("files", {}).get("checkpoint", {}).get(
        "sha256"
    ) != parent.get("source_checkpoint_sha256"):
        msg = "Matched branch teacher identity contradicts the common A0 handoff manifest."
        raise ValueError(msg)
    receipt = matched_budget.load_b_receipt(
        b_run / matched_budget.RECEIPT_NAME,
        parent_identity_sha256=str(record["parent_identity_sha256"]),
        a0_handoff_sha256=common.serialization.file_sha256(manifest_path),
        b_run_name=str(children["stage_b"]["run_name"]),
    )
    arm = training["comparison_arm"]
    expected_child_path = child_paths["stage_b" if arm == "b" else "stage_a1"]
    if run_dir.resolve() != expected_child_path or receipt["a0_run_name"] != children["stage_a0"]["run_name"]:
        msg = "Matched branch receipt does not bind the selected parent child and A0 source."
        raise ValueError(msg)
    if arm == "a1":
        local = matched_budget.load_b_receipt(
            run_dir / matched_budget.RECEIPT_NAME,
            parent_identity_sha256=str(record["parent_identity_sha256"]),
            a0_handoff_sha256=common.serialization.file_sha256(manifest_path),
            b_run_name=str(children["stage_b"]["run_name"]),
        )
        if local != receipt:
            msg = "Companion local matched budget receipt differs from immutable B source."
            raise ValueError(msg)
    if (
        receipt["dataset_identity"] != run_identity.dataset_identity_evidence(config)
        or receipt["input_profile"] != config["input_profile"]
        or receipt["temporal_conditioning"] != config["temporal"]["temporal_conditioning"]
        or receipt["spatial_stride"] != config["data"]["spatial_stride"]
        or receipt["model"] != config["model"]
        or receipt["objective"] != config["evaluation"]["objective"]
        or receipt["initialization"] != protocol["initialization"]
    ):
        msg = "Matched branch receipt contradicts saved scientific configuration."
        raise ValueError(msg)
    if arm == "b" and receipt["best_checkpoint_sha256"] != completed["selected_checkpoint_sha256"]:
        msg = "B receipt selected checkpoint differs from artifact checkpoint."
        raise ValueError(msg)
    return receipt


def _lineage(completed: Mapping[str, Any]) -> dict[str, Any]:
    """Project immutable Training strategy, handoff, curriculum, and compute evidence."""
    config = completed.get("config")
    summary = completed.get("summary")
    last_checkpoint = completed.get("last_checkpoint")
    if not isinstance(config, Mapping) or not isinstance(summary, Mapping) or not isinstance(last_checkpoint, Mapping):
        message = "Completed transient lineage requires admitted config, summary, and last checkpoint."
        raise TypeError(message)
    training = config.get("training")
    adapter_state = last_checkpoint.get("adapter_state_dict")
    if not isinstance(training, Mapping) or not isinstance(adapter_state, Mapping):
        message = "Completed transient lineage lacks Training or adapter evidence."
        raise TypeError(message)
    checkpoint_controller = adapter_state.get("controller")
    terminal_controller = summary.get("terminal_controller")
    terminal_curriculum = summary.get("terminal_curriculum")
    if not isinstance(checkpoint_controller, Mapping) or not isinstance(terminal_controller, Mapping) or not isinstance(terminal_curriculum, Mapping):
        message = "Completed transient summary lacks terminal controller/curriculum evidence."
        raise TypeError(message)
    stage = training.get("stage")
    arm = training.get("comparison_arm")
    expected_arm = {"a0": "A0", "a1": "A1", "b": "B"}
    if stage not in {"a", "b"} or arm not in expected_arm or checkpoint_controller.get("arm") != expected_arm[arm]:
        message = "Completed transient stage/comparison arm identity is unsupported or contradictory."
        raise ValueError(message)
    matched = training.get("matched_a1_b")
    if matched is not None and (not isinstance(matched, Mapping) or matched.get("role") != arm):
        msg = "Completed matched branch role contradicts its comparison arm."
        raise ValueError(msg)
    if arm == "a1" and matched is None:
        msg = "A1 lineage requires its explicitly selected matched_a1_b protocol."
        raise ValueError(msg)
    teacher_handoff = checkpoint_controller.get("teacher_handoff")
    if arm == "a0":
        if teacher_handoff is not None:
            message = "A0 lineage must not consume a teacher handoff."
            raise ValueError(message)
        task = experiments.config.loader.validate_resolved_task_contract(config)
        scaling = TransientScalingArtifact.from_state_dict(cast("Mapping[str, Any]", completed["normalizer_state"]))
        model = cast("Mapping[str, Any]", config["model"])
        parent = {
            "source_run_name": completed["scientific_run_name"],
            "source_checkpoint_sha256": completed["selected_checkpoint_sha256"],
            "source_scaling_sha256": completed["normalizer_sha256"],
            "task_contract_sha256": task.contract_digest,
            "tensorizer_sha256": common.serialization.canonical_json_sha256(scaling.tensorizer.as_dict()),
            "model_kind": model["kind"],
            "input_profile": config["input_profile"],
        }
    else:
        if not isinstance(teacher_handoff, Mapping):
            message = "A1 and B lineage require checkpoint-persisted teacher-handoff identity."
            raise TypeError(message)
        parent = dict(teacher_handoff)
    strategy = "rollout" if arm == "b" else "teacher_forced_matched_train_seconds" if arm == "a1" else "stage_a_teacher_forced"
    manifest: dict[str, Any] = {
        "planned": None if matched is None else dict(matched),
        "actual": dict(terminal_controller),
    }
    if matched is not None:
        if arm == "a0":
            if checkpoint_controller.get("clock_kind") == "train_seconds" or checkpoint_controller.get("budget_control") != "stage_epochs":
                msg = "Matched A0 must retain its completed Stage-A epoch budget."
                raise ValueError(msg)
            parent_evidence = _parent_experiment_evidence(completed)
            if parent_evidence["kind"] != "independent":
                msg = "Matched A0 requires its exact shared A0 content identity."
                raise ValueError(msg)
            manifest["a0_semantic_config_sha256"] = parent_evidence["semantic_config_sha256"]
        else:
            receipt = _matched_budget_receipt(completed, parent)
            planned = checkpoint_controller.get("planned_train_seconds")
            actual = checkpoint_controller.get("train_seconds")
            if (
                checkpoint_controller.get("clock_kind") != "train_seconds"
                or checkpoint_controller.get("budget_control") != "train_seconds"
                or not isinstance(planned, (int, float))
                or not isinstance(actual, (int, float))
                or isinstance(planned, bool)
                or isinstance(actual, bool)
                or not (0 < planned <= actual)
                or (arm == "b" and actual != receipt["budget_value"])
                or (
                    arm == "a1"
                    and (planned != receipt["budget_value"] or checkpoint_controller.get("matched_receipt_sha256") != receipt["receipt_sha256"])
                )
            ):
                msg = "Matched branch terminal budget contradicts the immutable B receipt."
                raise ValueError(msg)
            manifest["parent_experiment_sha256"] = receipt["parent_identity_sha256"]
            manifest["b_train_budget_receipt"] = receipt
    return {
        "stage_identity": {"stage": stage, "comparison_arm": arm},
        "training_strategy": strategy,
        "curriculum_identity": dict(terminal_curriculum),
        "parent_checkpoint": parent,
        "stage_a_handoff": parent,
        "matched_compute_manifest": manifest,
    }


def _qualified_case_membership(
    *,
    planned: Sequence[tuple[str, str]],
    produced: Sequence[Mapping[str, Any]] | None = None,
) -> frozenset[tuple[str, str]]:
    """Require exact material-qualified produced cases before metric serialization."""

    def admit_pairs(
        values: Sequence[tuple[str, str]],
        *,
        label: str,
    ) -> frozenset[tuple[str, str]]:
        if any(not isinstance(case_id, str) or not case_id or not isinstance(material, str) or not material for case_id, material in values):
            message = f"Transient artifact {label} case membership must use non-empty qualified identities and materials."
            raise TypeError(message)
        admitted = frozenset(values)
        if len(admitted) != len(values):
            message = f"Transient artifact {label} case membership contains duplicate qualified identities."
            raise ValueError(message)
        return admitted

    expected = admit_pairs(planned, label="planned")
    if produced is None:
        return expected
    actual_values: list[tuple[str, str]] = []
    for identity in produced:
        if not isinstance(identity, Mapping):
            message = "Transient artifact produced case identity must be a mapping."
            raise TypeError(message)
        case_id = identity.get("case_id")
        material = identity.get("material_family")
        simulation = identity.get("simulation_identity")
        package_case_id = simulation.get("package_case_id") if isinstance(simulation, Mapping) else None
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(package_case_id, str)
            or package_case_id != case_id
            or not isinstance(material, str)
            or not material
        ):
            message = "Transient artifact produced case lost its qualified Dataset package identity or material."
            raise ValueError(message)
        actual_values.append((package_case_id, material))
    actual = admit_pairs(actual_values, label="produced")
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        message = f"Transient artifact produced case membership contradicts its exact selected split: missing={missing}, unexpected={unexpected}."
        raise ValueError(message)
    return actual


def _case_dataset_identity(role: artifact_plan.ArtifactRolePlan, case_id: str) -> dict[str, Any]:
    """Derive one case's persisted Dataset identity from its owning saved split part; the producer and the role admission share it."""
    for dataset_id, source_identity, case_ids in zip(role.source_dataset_ids, role.source_identities, role.case_ids_by_source, strict=True):
        if case_id in case_ids:
            return {
                "artifact_dataset_name": role.dataset_name,
                "source_dataset_id": dataset_id,
                "saved_membership_digests": list(role.membership_digests),
                "source_index_digest": source_identity["index_digest"],
            }
    message = f"Case {case_id!r} is absent from saved {role.dataset_role} membership."
    raise ValueError(message)


def _base_identity(
    *,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    case: rollout.TransientEvaluationCase,
    canonical: Mapping[str, Any],
    pt_identity: str,
    runtime_identity: Mapping[str, Any],
    spatial_representation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact per-case sequence identity from completed owners."""
    config = cast("Mapping[str, Any]", completed["config"])
    task = experiments.config.loader.validate_resolved_task_contract(config)
    profile_id = config.get("input_profile")
    if not isinstance(profile_id, str):
        message = "Completed transient config lacks input-profile identity."
        raise TypeError(message)
    profile = task.input_profile(profile_id)
    scaling = TransientScalingArtifact.from_state_dict(cast("Mapping[str, Any]", completed["normalizer_state"]))
    model = config.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("params"), Mapping):
        message = "Completed transient model identity is invalid."
        raise TypeError(message)
    lineage = _lineage(completed)
    source = canonical["source"]
    metadata = canonical["meta"]
    dataset_identity = _case_dataset_identity(role, case.case_id)
    return {
        "case_id": case.case_id,
        "dataset_identity": dataset_identity,
        "dataset_role": role.dataset_role,
        "evaluation_assignment": next(
            (dict(assignment) for assignment in role.evaluation_assignments if assignment["dataset_id"] == dataset_identity["source_dataset_id"]),
            None,
        ),
        "material_family": str(case.metadata["material_family"]),
        "simulation_identity": {
            "simulation_case_id": source["simulation_case_id"],
            "case_input_id": source["case_input_id"],
            "source_batch_id": case.metadata["source_batch_id"],
            "package_case_id": case.case_id,
            "generation_case_id": source["case_id"],
            "simulation_profile": metadata["simulation_profile"],
            "template_sha256": metadata["template_sha256"],
        },
        "model_kind": model["kind"],
        "model_parameters": dict(model["params"]),
        "checkpoint_identity": {
            "identity": dict(cast("Mapping[str, Any]", completed["checkpoint_identity"])),
            "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
        },
        "input_profile": profile_id,
        "coordinate_policy": profile.coordinate_policy,
        "boundary_representation": datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.boundary_interval_representation,
        "scaling_identity": {
            "semantic_digest": scaling.digest,
            "normalizer_sha256": completed["normalizer_sha256"],
            "scale_mode": scaling.scale_mode,
            "spatial_shape": list(scaling.spatial_shape),
            "horizon_hours": scaling.horizon,
            "tensorizer": scaling.tensorizer.as_dict(),
        },
        "spatial_representation": dict(spatial_representation),
        "training_airflow_source": task.training_airflow_source,
        "inference_airflow_source": "comsol_reference",
        **lineage,
        "pt_payload_identity": pt_identity,
        "evaluation_config_identity": sequence_artifact.evaluation_protocol_identity(cast("Mapping[str, Any]", config["evaluation"])),
        "timing_evidence_identity": common.serialization.canonical_json_sha256(runtime_identity),
    }


def _timing_case(
    *,
    case_id: str,
    benchmark: rollout.TransientRolloutBenchmark,
    expected_model_calls: int,
    generation_timing: Mapping[str, Any],
    pt_identity: str,
    runtime_metadata: Mapping[str, Any],
) -> transient_timing.TransientTimingCase:
    """Adapt only complete public rollout clocks plus stable Generation timing."""
    if (
        isinstance(expected_model_calls, bool)
        or not isinstance(expected_model_calls, int)
        or not 1 <= benchmark.model_calls_per_repetition <= expected_model_calls
    ):
        message = "Transient timing adaptation requires bounded factual model-call support."
        raise ValueError(message)
    complete_rollout = benchmark.model_calls_per_repetition == expected_model_calls
    repetitions: dict[str, tuple[float, ...]] = {}
    cold: dict[str, float] = {}
    if complete_rollout:
        repetitions["drying_no_prepared_rollout_wall_seconds"] = benchmark.warmed_prepared_rollout_wall_seconds
        cold["drying_no_prepared_rollout_wall_seconds"] = benchmark.cold_prepared_rollout_wall_seconds
    if complete_rollout and all(value > 0.0 for value in benchmark.warmed_model_seconds):
        repetitions["drying_no_rollout_model_seconds"] = benchmark.warmed_model_seconds
        if benchmark.cold_model_seconds > 0.0:
            cold["drying_no_rollout_model_seconds"] = benchmark.cold_model_seconds

    source_by_component = {
        "comsol_transient_drying_seconds": "transient_drying_solver_seconds",
        "comsol_scientific_solver_seconds": "scientific_solver_seconds",
        "comsol_stationary_airflow_seconds": "stationary_airflow_solver_seconds",
        "comsol_process_seconds": "comsol_process_seconds",
        "generation_compute_end_to_end_seconds": "generation_compute_end_to_end_seconds",
    }
    source_availability = generation_timing.get("component_timing_availability")
    availability = source_availability if isinstance(source_availability, Mapping) else {}
    unavailable = {
        component: "not_available_from_current_evidence" for component in transient_timing.TIMING_COMPONENTS if component not in repetitions
    }
    if not complete_rollout:
        reason = "diagnostic_rollout_stopped_after_nonfinite_prediction"
        unavailable["drying_no_prepared_rollout_wall_seconds"] = reason
        unavailable["drying_no_rollout_model_seconds"] = reason
    for component, source in source_by_component.items():
        value = generation_timing.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0:
            repetitions[component] = (float(value),)
            unavailable.pop(component, None)
        else:
            unavailable[component] = str(availability.get(source, f"{source}_unavailable"))
    unavailable["airflow_no_model_seconds"] = "compatible_airflow_model_not_selected"
    if "drying_no_rollout_model_seconds" not in repetitions:
        unavailable.setdefault(
            "drying_no_rollout_model_seconds",
            "model_clock_resolution_did_not_produce_a_positive_duration",
        )
    unavailable["surrogate_pipeline_end_to_end_seconds"] = "complete_airflow_to_drying_pipeline_not_selected"

    processor = runtime_metadata.get("processor")
    cpu = str(processor) if isinstance(processor, str) and processor else "unknown"
    gpu_value = runtime_metadata.get("cuda_device_name")
    gpu = str(gpu_value) if isinstance(gpu_value, str) and gpu_value else None
    return transient_timing.TransientTimingCase(
        case_id=case_id,
        repetitions=repetitions,
        device=str(runtime_metadata["resolved_device"]),
        precision="float32",
        pt_payload_identity=pt_identity,
        warmup_passes=benchmark.warmup_passes,
        batch_size=1,
        cpu=cpu,
        gpu=gpu,
        software_versions={
            "python": str(runtime_metadata["python_version"]),
            "pytorch": str(runtime_metadata["pytorch_version"]),
            "numpy": str(np.__version__),
        },
        cold_timings=cold,
        unavailable_reasons=unavailable,
    )


def _parent_experiment_evidence(completed: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact admitted parent Experiment for a current transient run."""
    run_dir = completed.get("run_dir")
    summary = completed.get("summary")
    config = completed.get("config")
    if not isinstance(run_dir, Path) or not isinstance(summary, Mapping) or not isinstance(config, Mapping):
        message = "Completed transient evidence lacks its canonical run path, summary, or config."
        raise TypeError(message)
    checkpoint_identity = completed.get("checkpoint_identity")
    saved_digest = checkpoint_identity.get("resume_contract_digest") if isinstance(checkpoint_identity, Mapping) else None
    if not isinstance(saved_digest, str) or re.fullmatch(r"[0-9a-f]{64}", saved_digest) is None:
        message = "Completed transient parent binding requires a valid saved resume identity."
        raise ValueError(message)
    run_evidence = summary.get("run_identity")
    child_source: dict[str, Any] | None = None
    if isinstance(run_evidence, Mapping):
        source = run_evidence.get("source_repository")
        if isinstance(source, Mapping):
            child_source = dict(source)
    layout = config.get("run", {}).get("layout")
    if not isinstance(layout, Mapping):
        message = "Completed transient run lacks its derived storage layout."
        raise TypeError(message)
    if layout["stage"] in {"a0", "single"}:
        return {
            "kind": "independent",
            "layout": dict(layout),
            "run_name": completed["scientific_run_name"],
            "semantic_config_sha256": saved_digest,
            "source_repository": child_source,
        }
    output_root = common.paths.resolve_config_output_root(config, run_dir)
    marker = run_dir.parent / "experiment.json"
    if not marker.exists():
        message = "Current transient parent Experiment record is absent."
        raise FileNotFoundError(message)
    if not marker.is_file() or marker.is_symlink():
        message = f"Transient parent experiment record is unsafe: {marker}."
        raise ValueError(message)
    located = run_identity.load_readonly_transient_experiment_record(marker, experiments_root=output_root)
    record = located["record"]
    matching_children = [
        child
        for child in record["children"].values()
        if run_identity.resolve_readonly_experiment_child(record, child, marker=marker, output_root=output_root) == run_dir.resolve()
    ]
    if len(matching_children) != 1:
        message = "Transient parent experiment does not bind the exact completed child path."
        raise ValueError(message)
    child = matching_children[0]
    digest_key = "semantic_config_sha256"
    expected_digest = saved_digest
    if child["run_name"] != completed.get("scientific_run_name") or child.get(digest_key) != expected_digest:
        message = "Transient parent experiment child identity contradicts the completed run."
        raise ValueError(message)
    return {
        "kind": "grouped",
        "parent_available": True,
        "parent_label": record["parent_label"],
        "parent_identity_sha256": record["parent_identity_sha256"],
        "source_repository": dict(record["source_repository"]),
        "child_source_repository": child_source,
    }


def _record_metric_statistics(
    record: sequence_artifact.TransientSequenceRecord,
    *,
    scaling: TransientScalingArtifact,
) -> dict[str, Mapping[str, object]]:
    """Build full-support errors or explicit unavailability plus raw diagnostics."""
    result: dict[str, Mapping[str, object]] = {}
    density_index = sequence_artifact.STATIC_ORDER.index("rho_bu_dry")
    cumulative_prediction = record.predicted_states[1:]
    cumulative_reference = record.reference_states[1:]
    cell_weights = transient_metrics.trapezoidal_cell_weights(record.spatial_mask)
    full_support = bool(record.prediction_available.all()) and bool(np.isfinite(cumulative_prediction).all())
    normalized_cumulative_prediction = _normalized_metric_state(cumulative_prediction, scaling=scaling) if full_support else None
    normalized_cumulative_reference = _normalized_metric_state(cumulative_reference, scaling=scaling) if full_support else None
    for scope in ("cumulative", "endpoint"):
        selection = np.arange(record.available_horizon) if scope == "cumulative" else np.asarray([record.available_horizon - 1])
        prediction = cumulative_prediction[selection]
        reference = cumulative_reference[selection]
        available_steps = record.prediction_available[selection]
        required_count = int(prediction.size)
        computed_count = int(available_steps.sum() * prediction.shape[1] * prediction.shape[2] * prediction.shape[3])
        finite_count = int(np.isfinite(prediction[available_steps]).sum())
        support = validity_contract.support_evidence(required=required_count, computed=computed_count, finite=finite_count)
        statistics: Mapping[str, object] | None = None
        if support["available"]:
            if normalized_cumulative_prediction is None or normalized_cumulative_reference is None:
                message = "Available transient metrics unexpectedly lack normalized full support."
                raise RuntimeError(message)
            normalized_prediction = normalized_cumulative_prediction[selection]
            normalized_reference = normalized_cumulative_reference[selection]
            accumulator = transient_metrics.TransientMetricAccumulator(
                scope=scope,
            )
            accumulator.update(
                normalized_prediction=normalized_prediction,
                normalized_reference=normalized_reference,
                physical_prediction=prediction,
                physical_reference=reference,
                f_surf=np.asarray(record.scalar_conditioning[2]),
                rho_bu_dry=record.static_conditioning[density_index],
                cell_weights=cell_weights,
                valid_mask=record.spatial_mask[None, None],
            )
            statistics = accumulator.state_dict()
        result[scope] = {**support, "statistics": statistics}
    computed_steps = int(record.prediction_available.sum())
    computed_states = record.predicted_states[: computed_steps + 1]
    result["diagnostics"] = {
        "classification": "PHYSICAL_VALIDITY_METRIC",
        "plausibility": asdict(
            transient_metrics.derive_plausibility_diagnostics(
                computed_states[1:],
                temperature_range=transient_metrics.TEMPERATURE_PLAUSIBILITY_RANGE_K,
            )
        ),
        "stability": asdict(transient_metrics.derive_stability_diagnostics(computed_states)),
    }
    return result


def _normalized_metric_state(
    value: np.ndarray,
    *,
    scaling: TransientScalingArtifact,
) -> np.ndarray:
    """Normalize one portable state array through the scaler runtime placement."""
    with torch.inference_mode():
        source = torch.from_numpy(np.ascontiguousarray(value))
        runtime = source.to(device=scaling.device, dtype=scaling.dtype)
        return scaling.encode_state(runtime).detach().to(device="cpu").numpy()


def _timing_case_from_resume(
    value: Any,
    *,
    case_id: str,
) -> transient_timing.TransientTimingCase:
    """Restore one strict timing case from JSON-safe resume evidence."""
    required = {
        "case_id",
        "repetitions",
        "device",
        "precision",
        "pt_payload_identity",
        "warmup_passes",
        "batch_size",
        "cpu",
        "gpu",
        "software_versions",
        "cold_timings",
        "unavailable_reasons",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        message = "Resumed transient timing-case fields are invalid."
        raise ValueError(message)
    repetitions = value["repetitions"]
    if not isinstance(repetitions, Mapping):
        message = "Resumed transient timing repetitions must be a mapping."
        raise TypeError(message)
    if any(not isinstance(value[name], str) or not value[name] for name in ("case_id", "device", "precision", "pt_payload_identity", "cpu")):
        message = "Resumed transient timing-case text fields are invalid."
        raise TypeError(message)
    restored = transient_timing.TransientTimingCase(
        case_id=str(value["case_id"]),
        repetitions={str(name): tuple(float(item) for item in values) for name, values in repetitions.items()},
        device=str(value["device"]),
        precision=str(value["precision"]),
        pt_payload_identity=str(value["pt_payload_identity"]),
        warmup_passes=int(value["warmup_passes"]),
        batch_size=int(value["batch_size"]),
        cpu=str(value["cpu"]),
        gpu=None if value["gpu"] is None else str(value["gpu"]),
        software_versions={
            str(name): str(item)
            for name, item in cast(
                "Mapping[str, Any]",
                value["software_versions"],
            ).items()
        },
        cold_timings={
            str(name): float(item)
            for name, item in cast(
                "Mapping[str, Any]",
                value["cold_timings"],
            ).items()
        },
        unavailable_reasons={
            str(name): str(item)
            for name, item in cast(
                "Mapping[str, Any]",
                value["unavailable_reasons"],
            ).items()
        },
    )
    if restored.case_id != case_id:
        message = "Resumed transient timing evidence contradicts its case."
        raise ValueError(message)
    transient_timing.build_transient_timing_report((restored,))
    return restored


def _bind_case_timing_evidence(
    component: object,
    *,
    case_id: str,
    pt_payload_identities: set[object],
    timing_case: transient_timing.TransientTimingCase,
    prediction_validity: Mapping[str, Any],
) -> None:
    """Require one case's component evidence to carry the producer's fields and agree with its timing case, sequence identity and validity."""
    seconds = component.get("dataset_materialization_seconds") if isinstance(component, Mapping) else None
    if (
        not isinstance(component, Mapping)
        or set(component) != _COMPONENT_EVIDENCE_FIELDS
        or any(not isinstance(component[name], Mapping) for name in _COMPONENT_MAPPING_FIELDS)
        or any(not isinstance(component[name], str) or not component[name] for name in _COMPONENT_TEXT_FIELDS)
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        message = f"Transient case {case_id!r} component timing evidence lacks the producer's fields or a finite non-negative duration."
        raise ValueError(message)
    if (
        timing_case.case_id != case_id
        or pt_payload_identities != {timing_case.pt_payload_identity}
        or component["prediction_validity"] != prediction_validity
        or common.serialization.canonical_json_sha256(component["timing_case"]) != common.serialization.canonical_json_sha256(asdict(timing_case))
    ):
        message = f"Transient case {case_id!r} contradicts its timing evidence: sequence identity, timing case and component evidence must agree."
        raise ValueError(message)


def _validate_role_timing_evidence(
    timing_block: Mapping[str, Any],
    *,
    expected_cases: set[str],
    pt_identities: Mapping[str, set[object]],
    expected_validity: Mapping[str, Mapping[str, Any]],
) -> None:
    """Admit the role's timing report and bind every report case to its component evidence and sequence record identity."""
    component_evidence = timing_block["component_availability"]
    if not isinstance(component_evidence, Mapping) or set(component_evidence) != expected_cases:
        message = "Transient artifact timing-component evidence does not cover exact case membership."
        raise ValueError(message)
    report = transient_timing.admit_transient_timing_report(cast("Mapping[str, object]", timing_block["report"]))
    if {case.case_id for case in report.cases} != expected_cases:
        message = "Transient timing report does not cover exact saved complete-case membership."
        raise ValueError(message)
    for case in report.cases:
        _bind_case_timing_evidence(
            component_evidence[case.case_id],
            case_id=case.case_id,
            pt_payload_identities=pt_identities[case.case_id],
            timing_case=case,
            prediction_validity=expected_validity[case.case_id],
        )


@dataclass(frozen=True, slots=True)
class _ResumedCaseEvidence:
    """Hold strictly restored generator state for one completed staged case."""

    identity: Mapping[str, Any]
    spatial_identity: Mapping[str, Any]
    component_availability: Mapping[str, Any]
    timing_case: transient_timing.TransientTimingCase
    prediction_validity: Mapping[str, Any]
    spatial_compatibility: Mapping[str, Any]
    material: str
    rollout_steps: int


def _restore_resumed_case_evidence(
    value: Any,
    *,
    case_id: str,
    expected_material: str | None,
    dataset_role: sequence_artifact.DatasetRole,
) -> _ResumedCaseEvidence:
    """Restore exact compact case evidence without Dataset or model execution."""
    required = {
        "schema_kind",
        "case_id",
        "material_family",
        "identity",
        "spatial_identity",
        "component_availability",
        "timing_case",
        "prediction_validity_records",
        "spatial_compatibility",
        "progress",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        message = "Transient staged case resume evidence is incompatible."
        raise ValueError(message)
    material = value["material_family"]
    if (
        value["schema_kind"] != _RESUME_EVIDENCE_SCHEMA_KIND
        or value["case_id"] != case_id
        or not isinstance(material, str)
        or not material
        or (expected_material is not None and material != expected_material)
    ):
        message = "Transient staged case resume identity is incompatible."
        raise ValueError(message)
    identity = value["identity"]
    spatial_identity = value["spatial_identity"]
    component = value["component_availability"]
    raw_validity = value["prediction_validity_records"]
    spatial_compatibility = value["spatial_compatibility"]
    progress = value["progress"]
    if (
        not isinstance(identity, Mapping)
        or not isinstance(spatial_identity, Mapping)
        or not isinstance(component, Mapping)
        or not isinstance(raw_validity, list)
        or not raw_validity
        or not all(isinstance(item, Mapping) for item in raw_validity)
        or not isinstance(spatial_compatibility, Mapping)
        or set(spatial_compatibility) != {"architecture", "scaling"}
        or any(not isinstance(spatial_compatibility[name], Mapping) or not spatial_compatibility[name] for name in ("architecture", "scaling"))
        or not isinstance(progress, Mapping)
        or set(progress) != {"rollout_steps"}
    ):
        message = "Transient staged case resume payload is incompatible."
        raise ValueError(message)
    _qualified_case_membership(
        planned=((case_id, material),),
        produced=(identity,),
    )
    identity = sequence_artifact.validate_sequence_identity(identity, case_id=case_id, dataset_role=dataset_role)
    rollout_steps = progress["rollout_steps"]
    if isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int) or rollout_steps < 1:
        message = "Resumed transient rollout-step evidence is invalid."
        raise ValueError(message)
    prediction_validity = transient_validity.aggregate_case_prediction_validity(
        case_id=case_id,
        records=cast(
            "Sequence[Mapping[str, Any]]",
            raw_validity,
        ),
    )
    timing_case = _timing_case_from_resume(
        value["timing_case"],
        case_id=case_id,
    )
    if spatial_identity != identity.get("spatial_representation"):
        message = "Resumed transient spatial identity contradicts its case identity."
        raise ValueError(message)
    _bind_case_timing_evidence(
        component,
        case_id=case_id,
        pt_payload_identities={identity["pt_payload_identity"]},
        timing_case=timing_case,
        prediction_validity=prediction_validity,
    )
    return _ResumedCaseEvidence(
        identity=dict(identity),
        spatial_identity=dict(spatial_identity),
        component_availability=dict(component),
        timing_case=timing_case,
        prediction_validity=prediction_validity,
        spatial_compatibility={name: dict(cast("Mapping[str, Any]", spatial_compatibility[name])) for name in ("architecture", "scaling")},
        material=material,
        rollout_steps=rollout_steps,
    )


def _role_prediction_validity_evidence(
    cases: Mapping[str, Mapping[str, Any]],
    *,
    expected_case_ids: set[str],
) -> dict[str, Any]:
    """Build exact role-level prediction status counts and per-case evidence."""
    if set(cases) != expected_case_ids:
        message = "Transient prediction-validity cases contradict role membership."
        raise ValueError(message)
    return validity_contract.role_prediction_validity(cases)


def _role_spatial_representation_evidence(
    *,
    completed: Mapping[str, Any],
    training: datasets.contracts.transient.TransientSpatialRepresentation,
    evaluation: datasets.contracts.transient.TransientSpatialRepresentation,
    spatial_compatibility: Mapping[str, Any],
    case_grids: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build exact role-level Training/Evaluation grid and compatibility evidence."""
    if training.source_shape != evaluation.source_shape:
        message = "Training and Evaluation representations must share one canonical source grid."
        raise ValueError(message)
    if (
        not isinstance(spatial_compatibility, Mapping)
        or set(spatial_compatibility) != {"architecture", "scaling"}
        or any(not isinstance(spatial_compatibility[name], Mapping) or not spatial_compatibility[name] for name in ("architecture", "scaling"))
    ):
        message = "Transient spatial compatibility evidence is incomplete."
        raise ValueError(message)
    return {
        "inference_contract": "transient_physical_rollout_evaluation_grid",
        "source_grid": {
            "shape": list(training.source_shape),
            "owner": "canonical_generation_dataset_package",
        },
        "training_grid": training.as_dict(),
        "evaluation_grid": evaluation.as_dict(),
        "checkpoint_identity": {
            "identity": dict(cast("Mapping[str, Any]", completed["checkpoint_identity"])),
            "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
        },
        "architecture_compatibility": dict(cast("Mapping[str, Any]", spatial_compatibility["architecture"])),
        "scaling_compatibility": dict(cast("Mapping[str, Any]", spatial_compatibility["scaling"])),
        "field_alignment": {
            "dynamic_fields": list(sequence_artifact.STATE_ORDER),
            "static_fields": list(sequence_artifact.STATIC_ORDER),
            "coordinates": ["x", "y"],
            "spatial_mask": "same_evaluation_index_map",
            "model_input": "evaluation_grid",
            "reference": "evaluation_grid",
            "prediction": "evaluation_grid",
            "interpolation": "forbidden",
        },
        "case_grids": {case_id: dict(case_grids[case_id]) for case_id in sorted(case_grids)},
    }


def _role_provenance(
    *,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    device_resolution: DeviceResolution,
    component_availability: Mapping[str, Any],
    timing_report: Mapping[str, Any],
    prediction_validity: Mapping[str, Any],
    spatial_representation: Mapping[str, Any],
    saved_role: artifact_plan.ArtifactRolePlan | None = None,
) -> dict[str, Any]:
    """Build role-level run, Dataset, Evaluation, runtime, and lineage provenance."""
    config = cast("Mapping[str, Any]", completed["config"])
    scope = None
    if saved_role is not None:
        selected_case_ids = [case_id for source_case_ids in role.case_ids_by_source for case_id in source_case_ids]
        scope = {
            "kind": "selected_cases",
            "selected_case_ids": selected_case_ids,
            "saved_case_ids_by_source": [list(values) for values in saved_role.case_ids_by_source],
            "canonical_publication_eligible": False,
        }
    result = {
        "producer": contracts.analysis_producer("transient_drying"),
        "run": {
            "name": completed["scientific_run_name"],
            "effective_config_digest": completed["effective_config_digest"],
            "checkpoint_identity": dict(cast("Mapping[str, Any]", completed["checkpoint_identity"])),
            "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
            "normalizer_sha256": completed["normalizer_sha256"],
            "parent_experiment": _parent_experiment_evidence(completed),
        },
        "dataset": role.dataset_provenance(),
        "evaluation": {
            "config_identity": sequence_artifact.evaluation_protocol_identity(cast("Mapping[str, Any]", config["evaluation"])),
            "metrics": config["evaluation"]["metrics"],
            "objective": config["evaluation"]["objective"],
            "modes": list(sequence_artifact.EVALUATION_MODES),
            "fixed_horizons": list(sequence_artifact.FIXED_HORIZONS),
            "rolling_origin_policy": sequence_artifact.ROLLING_ORIGIN_POLICY,
            "recurrent_state_policy": sequence_artifact.RECURRENT_STATE_POLICY,
            "process_diagnostic_policy": _process_diagnostic_policy(),
            "prediction_validity": dict(prediction_validity),
            "pipeline_analysis": {
                "conditions": ["A_comsol_reference", "B_drying_on_comsol_airflow", "C_drying_on_airflow_no"],
                "c_available": False,
                "c_unavailable_reasons": [
                    "compatible_airflow_checkpoint_not_selected",
                    "compatible_airflow_normalizer_not_selected",
                    "compatible_airflow_task_and_case_mapping_not_selected",
                ],
                "fabricated_prediction_count": 0,
            },
        },
        "spatial_representation": dict(spatial_representation),
        "runtime": contracts.runtime_provenance(device_resolution.as_dict(), inference_batch_size=1),
        "lineage": _lineage(completed),
        "timing": {"report": dict(timing_report), "component_availability": dict(component_availability)},
    }
    if scope is not None:
        result["evaluation"]["scope"] = scope
    return result


def generate_transient_role_artifact(  # noqa: C901, PLR0912, PLR0915
    *,
    run_dir: Path | str,
    role: artifact_plan.ArtifactRolePlan,
    device_resolution: DeviceResolution,
    dataset_root: Path | str,
    staging_root: Path | str,
    completed: Mapping[str, Any] | None = None,
    evaluation_spatial_stride: int = 1,
    saved_role: artifact_plan.ArtifactRolePlan | None = None,
    progress_reporter: artifact_performance.ArtifactProgressReporter | None = None,
) -> sequence_artifact.TransientSequenceArtifactIndex:
    """Generate one role through bounded case inference, metrics, and staging."""
    preflight_phase = nullcontext() if progress_reporter is None else progress_reporter.phase("preflight")
    with preflight_phase:
        admitted_completed = _completed_transient(run_dir) if completed is None else _require_completed_transient(completed)
        config = cast("Mapping[str, Any]", admitted_completed["config"])
        training_representation, evaluation_representation = resolve_transient_artifact_spatial_representations(
            admitted_completed,
            evaluation_spatial_stride=evaluation_spatial_stride,
        )
        print(
            f"artifact grid {evaluation_representation.source_shape} -> {evaluation_representation.represented_shape}"
            f" (stride {evaluation_representation.spatial_stride})"
        )
        storage_root = common.paths.storage_root_for_datasets_root(dataset_root)
    datasets_for_role: tuple[TransientPhysicalDataset, ...] = ()
    context: Any | None = None
    generation_sources: Mapping[str, Any] | None = None
    runtime_metadata: Mapping[str, Any] | None = None
    component_availability: dict[str, Any] = {}
    timing_cases: list[transient_timing.TransientTimingCase] = []
    produced_case_identities: list[Mapping[str, Any]] = []
    case_spatial_identities: dict[str, Mapping[str, Any]] = {}
    case_prediction_validity: dict[str, Mapping[str, Any]] = {}
    index: sequence_artifact.TransientSequenceArtifactIndex | None = None
    try:
        expected_case_ids = {case_id for case_ids in role.case_ids_by_source for case_id in case_ids}
        if len(expected_case_ids) != sum(len(case_ids) for case_ids in role.case_ids_by_source):
            message = "Transient artifact saved role contains duplicate case identities."
            raise ValueError(message)
        resume_identity = {
            "run_name": admitted_completed["scientific_run_name"],
            "resolved_config_sha256": run_identity.resolved_config_digest(config),
            "effective_config_digest": admitted_completed["effective_config_digest"],
            "checkpoint": {
                "identity": dict(
                    cast(
                        "Mapping[str, Any]",
                        admitted_completed["checkpoint_identity"],
                    )
                ),
                "selected_checkpoint_sha256": admitted_completed["selected_checkpoint_sha256"],
                "selected_checkpoint_epoch": admitted_completed["selected_checkpoint_epoch"],
            },
            "normalizer_sha256": admitted_completed["normalizer_sha256"],
            "role": {
                "split": role.split,
                "dataset_name": role.dataset_name,
                "dataset_role": role.dataset_role,
                "source_dataset_ids": list(role.source_dataset_ids),
                "source_identities": [dict(value) for value in role.source_identities],
                "case_ids_by_source": [list(case_ids) for case_ids in role.case_ids_by_source],
                "membership_digests": list(role.membership_digests),
            },
            "evaluation_protocol_identity": (
                sequence_artifact.evaluation_protocol_identity(
                    cast(
                        "Mapping[str, Any]",
                        config["evaluation"],
                    )
                )
            ),
            "evaluation_spatial_representation": (evaluation_representation.as_dict()),
            # Kept cases carry their timing, so a resumed stage must run on the same hardware and PyTorch build.
            "timing_hardware": {
                "device_type": device_resolution.metadata.device_type,
                "cuda_device_name": device_resolution.metadata.cuda_device_name,
                "pytorch_version": device_resolution.metadata.pytorch_version,
            },
            # Kept cases were computed by one analysis implementation; a stage started by other analysis code is not resumed.
            "producer": contracts.analysis_producer("transient_drying"),
        }
        recorded_identity = sequence_artifact.recorded_resume_identity(staging_root)
        if recorded_identity is not None:
            contracts.require_current_analysis(recorded_identity, task_id="transient_drying", unit=f"Partial transient artifact stage {staging_root}")
        stager = sequence_artifact.TransientSequenceArtifactStager(
            staging_root,
            dataset_name=role.dataset_name,
            dataset_role=role.dataset_role,
            resume_identity=resume_identity,
        )
        completed_case_ids = stager.completed_case_ids
        if not completed_case_ids.issubset(expected_case_ids):
            message = "Transient resume manifests contain cases outside the exact saved role; rerun with --rebuild."
            raise ValueError(message)
        all_cases_resumed = completed_case_ids == expected_case_ids
        resumed_cases: dict[str, _ResumedCaseEvidence] = {}
        spatial_compatibility: Mapping[str, Any] | None = None
        if all_cases_resumed:
            for case_id in sorted(expected_case_ids):
                resumed = _restore_resumed_case_evidence(
                    stager.completed_case_evidence(case_id),
                    case_id=case_id,
                    expected_material=None,
                    dataset_role=role.dataset_role,
                )
                resumed_cases[case_id] = resumed
            planned_membership = [(case_id, resumed_cases[case_id].material) for case_id in sorted(expected_case_ids)]
            specs: tuple[_TransientCaseMaterialization, ...] = ()
        else:
            model_phase = nullcontext() if progress_reporter is None else progress_reporter.phase("model_setup")
            with model_phase:
                context = learning.inference.transient.build_transient_inference_context(
                    completed=admitted_completed,
                    device=device_resolution.device,
                    precision="float32",
                    evaluation_spatial_shape=evaluation_representation.represented_shape,
                )
                generation_sources = _generation_case_sources(role, storage_root=storage_root)
                datasets_for_role = _load_role_datasets(
                    role,
                    storage_root=storage_root,
                    spatial_representation=evaluation_representation,
                    saved_role=saved_role,
                )
                runtime_metadata = artifact_timing.neural_runtime_metadata(
                    device_metadata=device_resolution.as_dict(),
                    model=context.model,
                )
                if progress_reporter is not None:
                    model_parameter = next(context.model.parameters(), None)
                    model_device = context.device if model_parameter is None else model_parameter.device
                    progress_reporter.device_summary(model_device=model_device, scaler_device=context.scaling.device)
            specs = _case_materialization_specs(datasets_for_role)
            planned_membership = [(spec.package_case_id, spec.material_family) for spec in specs]
        admitted_planned = _qualified_case_membership(planned=planned_membership)
        if {case_id for case_id, _material in admitted_planned} != expected_case_ids:
            message = "Transient artifact selected cases contradict the exact saved split before inference."
            raise ValueError(message)
        if all_cases_resumed:
            for case_id in sorted(expected_case_ids):
                resumed = resumed_cases[case_id]
                if spatial_compatibility is None:
                    spatial_compatibility = resumed.spatial_compatibility
                elif spatial_compatibility != resumed.spatial_compatibility:
                    message = "Resumed transient spatial compatibility evidence conflicts across cases."
                    raise ValueError(message)
                produced_case_identities.append(resumed.identity)
                case_spatial_identities[case_id] = resumed.spatial_identity
                component_availability[case_id] = dict(resumed.component_availability)
                timing_cases.append(resumed.timing_case)
                case_prediction_validity[case_id] = dict(resumed.prediction_validity)
                if progress_reporter is not None:
                    progress_reporter.case_reused(case_id=case_id, material=resumed.material)
        for spec in specs:
            package_case_id = spec.package_case_id
            if package_case_id in stager.completed_case_ids:
                stored = stager.completed_case_evidence(package_case_id)
                resumed = _restore_resumed_case_evidence(
                    stored,
                    case_id=package_case_id,
                    expected_material=spec.material_family,
                    dataset_role=role.dataset_role,
                )
                if package_case_id in case_spatial_identities:
                    message = "Transient resume produced duplicate case evidence."
                    raise ValueError(message)
                if spatial_compatibility is None:
                    spatial_compatibility = resumed.spatial_compatibility
                elif spatial_compatibility != resumed.spatial_compatibility:
                    message = "Resumed transient spatial compatibility evidence conflicts across cases."
                    raise ValueError(message)
                produced_case_identities.append(resumed.identity)
                case_spatial_identities[package_case_id] = resumed.spatial_identity
                component_availability[package_case_id] = dict(resumed.component_availability)
                timing_cases.append(resumed.timing_case)
                case_prediction_validity[package_case_id] = dict(resumed.prediction_validity)
                if progress_reporter is not None:
                    progress_reporter.case_reused(
                        case_id=package_case_id,
                        material=resumed.material,
                    )
                continue

            if context is None or generation_sources is None or runtime_metadata is None:
                message = "Incomplete transient resume lacks inference owners."
                raise RuntimeError(message)
            fresh_spatial_compatibility = {
                "architecture": dict(context.architecture_spatial_compatibility),
                "scaling": dict(context.scaling_spatial_compatibility),
            }
            if spatial_compatibility is None:
                spatial_compatibility = fresh_spatial_compatibility
            elif spatial_compatibility != fresh_spatial_compatibility:
                message = "Transient spatial compatibility evidence conflicts across resumed and new cases."
                raise ValueError(message)
            inference_phase = nullcontext() if progress_reporter is None else progress_reporter.work_phase("inference")
            with inference_phase:
                (
                    package_case_id,
                    case_record,
                    case,
                    pt_identity,
                ) = _materialize_case(
                    spec,
                    dataset_role=role.dataset_role,
                )
                canonical, runtime_identity = _canonical_case_evidence(
                    case,
                    package_case_id=package_case_id,
                    case_record=case_record,
                    generation_sources=generation_sources,
                    spatial_representation=evaluation_representation,
                )
                completion = canonical["completion"]
                target_limit = completion.get("target_wet_fraction_limit")
                target_wet_basis = canonical["scalar_conditioning"].get("X_target_wb")
                if (
                    isinstance(target_limit, bool)
                    or not isinstance(target_limit, (int, float))
                    or isinstance(target_wet_basis, bool)
                    or not isinstance(target_wet_basis, (int, float))
                ):
                    message = f"Canonical target evidence is unavailable for transient case {package_case_id!r}."
                    raise TypeError(message)
                case_spatial_identity = sequence_artifact.build_transient_spatial_identity(
                    evaluation_representation,
                    reference_states=case.reference_states,
                    static_conditioning=case.static_conditioning,
                    spatial_mask=case.spatial_mask,
                )
                identity = _base_identity(
                    completed=admitted_completed,
                    role=role,
                    case=case,
                    canonical=canonical,
                    pt_identity=pt_identity,
                    runtime_identity=runtime_identity,
                    spatial_representation=case_spatial_identity,
                )
                if case.case_id in case_spatial_identities:
                    message = f"Transient artifact produced duplicate spatial identity for case {case.case_id!r}."
                    raise ValueError(message)
                case_spatial_identities[case.case_id] = case_spatial_identity
                _qualified_case_membership(
                    planned=(
                        (
                            spec.package_case_id,
                            spec.material_family,
                        ),
                    ),
                    produced=(identity,),
                )
                material = str(identity["material_family"])
                produced_case_identities.append(identity)
                if progress_reporter is not None:
                    progress_reporter.case_started(
                        case_id=case.case_id,
                        material=material,
                        rollout_steps=case.transition_count,
                    )
                prepared_case = rollout.prepare_transient_evaluation_case(
                    context,
                    case,
                )
                benchmark = rollout.benchmark_transient_full_rollout(
                    context,
                    case,
                    prepared_case=prepared_case,
                    repetitions=_TIMING_REPETITIONS,
                )
                timing_case = _timing_case(
                    case_id=case.case_id,
                    benchmark=benchmark,
                    expected_model_calls=case.transition_count,
                    generation_timing=runtime_identity,
                    pt_identity=pt_identity,
                    runtime_metadata=runtime_metadata,
                )
                timing_cases.append(timing_case)
                evaluated = rollout.evaluate_transient_case(
                    context,
                    case,
                    identity=identity,
                    reference_completion=(_generation_target_completion(completion)),
                    target_wet_basis=float(target_wet_basis),
                    target_fraction_limit=float(target_limit),
                    prepared_case=prepared_case,
                )
                case_prediction_validity[case.case_id] = dict(evaluated.prediction_validity)
                component_availability[case.case_id] = {
                    "dataset_materialization_seconds": case.metadata["dataset_materialization_seconds"],
                    "generation": dict(runtime_identity),
                    "timing_case": asdict(timing_case),
                    "prediction_validity": dict(evaluated.prediction_validity),
                    "model_clock": benchmark.model_clock,
                    "wall_clock": benchmark.wall_clock,
                    "diagnostic_benchmark": {
                        "requested_model_calls_per_repetition": case.transition_count,
                        "actual_model_calls_per_repetition": (benchmark.model_calls_per_repetition),
                        "complete_rollout": (benchmark.model_calls_per_repetition == case.transition_count),
                        "cold_model_seconds": benchmark.cold_model_seconds,
                        "cold_prepared_rollout_wall_seconds": (benchmark.cold_prepared_rollout_wall_seconds),
                        "warmed_model_seconds": list(benchmark.warmed_model_seconds),
                        "warmed_prepared_rollout_wall_seconds": list(benchmark.warmed_prepared_rollout_wall_seconds),
                    },
                    "airflow_model": {
                        "available": False,
                        "reason": ("compatible_airflow_model_not_selected"),
                    },
                    "drying_model": {
                        "available": True,
                        "source": ("bounded_public_transient_full_rollout_benchmark"),
                    },
                    "surrogate_pipeline": {
                        "available": False,
                        "reason": ("complete_airflow_to_drying_pipeline_not_selected"),
                    },
                }
                benchmark_calls = benchmark.model_calls_per_repetition * (1 + benchmark.warmup_passes + benchmark.repetitions)
                benchmark_timed_calls = benchmark.model_calls_per_repetition * (1 + benchmark.repetitions)
                evaluated_calls = evaluated.model_calls
                unique_timed_records = tuple(
                    record for record in evaluated.records if record.mode != "rolling_origin" or record.requested_horizon == "full"
                )
                evaluated_model_seconds = sum(
                    float(seconds)
                    for record in unique_timed_records
                    if isinstance(
                        seconds := record.timing.get("seconds"),
                        (int, float),
                    )
                )
                benchmark_model_seconds = benchmark.cold_model_seconds + sum(benchmark.warmed_model_seconds)
                unique_validity_records = [dict(record.prediction_validity) for record in unique_timed_records]
                rollout_steps = case.transition_count
                del prepared_case, canonical
            metrics_phase = nullcontext() if progress_reporter is None else progress_reporter.work_phase("metrics")
            with metrics_phase:
                case_statistics = {
                    record.record_id: _record_metric_statistics(
                        record,
                        scaling=context.scaling,
                    )
                    for record in evaluated.records
                }
            serialization_phase = nullcontext() if progress_reporter is None else progress_reporter.work_phase("serialization")
            with serialization_phase:
                stager.write_case(
                    evaluated.records,
                    unavailable_horizons=(evaluated.unavailable_horizons),
                    metric_statistics=case_statistics,
                    resume_evidence={
                        "schema_kind": _RESUME_EVIDENCE_SCHEMA_KIND,
                        "case_id": package_case_id,
                        "material_family": material,
                        "identity": dict(identity),
                        "spatial_identity": dict(case_spatial_identity),
                        "component_availability": dict(component_availability[package_case_id]),
                        "timing_case": asdict(timing_case),
                        "prediction_validity_records": (unique_validity_records),
                        "spatial_compatibility": dict(cast("Mapping[str, Any]", spatial_compatibility)),
                        "progress": {
                            "rollout_steps": rollout_steps,
                        },
                    },
                )
            if progress_reporter is not None:
                progress_reporter.case_completed(
                    case_id=case.case_id,
                    material=material,
                    forward_calls=(benchmark_calls + evaluated_calls),
                    timed_forward_calls=(benchmark_timed_calls + evaluated_calls),
                    model_forward_seconds=(benchmark_model_seconds + evaluated_model_seconds),
                )
            del evaluated, case_statistics, case
        finalization_phase = nullcontext() if progress_reporter is None else progress_reporter.phase("finalization")
        with finalization_phase:
            _qualified_case_membership(
                planned=planned_membership,
                produced=produced_case_identities,
            )
            timing_report = asdict(
                transient_timing.build_transient_timing_report(timing_cases),
            )
            prediction_validity = _role_prediction_validity_evidence(
                case_prediction_validity,
                expected_case_ids=expected_case_ids,
            )
            if spatial_compatibility is None:
                message = "Transient role has no spatial compatibility evidence."
                raise RuntimeError(message)
            role_spatial_evidence = _role_spatial_representation_evidence(
                completed=admitted_completed,
                training=training_representation,
                evaluation=evaluation_representation,
                spatial_compatibility=cast("Mapping[str, Any]", spatial_compatibility),
                case_grids=case_spatial_identities,
            )
            provenance = _role_provenance(
                completed=admitted_completed,
                role=role,
                device_resolution=device_resolution,
                component_availability=component_availability,
                timing_report=timing_report,
                prediction_validity=prediction_validity,
                spatial_representation=role_spatial_evidence,
                saved_role=saved_role,
            )
            index = stager.finalize(provenance=provenance)
    finally:
        for dataset in datasets_for_role:
            dataset.close()
    if progress_reporter is not None:
        started_phases = progress_reporter.recorder.stage_seconds
        for phase_name in ("inference", "metrics", "serialization"):
            if phase_name in started_phases:
                progress_reporter.finish_work_phase(phase_name)
        progress_reporter.inference_summary()
        progress_reporter.update_cuda_peaks()
    if index is None:
        message = "Transient artifact staging did not finalize."
        raise RuntimeError(message)
    return index


def _sequence_value(record: object, name: str) -> Any:
    """Read one common eager-record or manifest-summary field."""
    try:
        return getattr(record, name)
    except AttributeError as error:
        message = f"Transient sequence evidence lacks {name!r}."
        raise TypeError(message) from error


def _role_prediction_validity_from_records(
    records: Sequence[object],
    *,
    expected_case_ids: set[str],
) -> dict[str, Any]:
    """Rebuild role validity from the unique persisted prediction chains."""
    by_case: dict[str, list[Mapping[str, Any]]] = {case_id: [] for case_id in expected_case_ids}
    for record in records:
        case_id = str(_sequence_value(record, "case_id"))
        mode = str(_sequence_value(record, "mode"))
        requested_horizon = _sequence_value(record, "requested_horizon")
        if mode == "rolling_origin" and requested_horizon != "full":
            continue
        if case_id not in by_case:
            message = "Transient prediction validity includes an unexpected case."
            raise ValueError(message)
        validity = _sequence_value(record, "prediction_validity")
        if not isinstance(validity, Mapping):
            message = "Transient sequence prediction validity must be a mapping."
            raise TypeError(message)
        by_case[case_id].append(validity)
    case_evidence = {
        case_id: transient_validity.aggregate_case_prediction_validity(
            case_id=case_id,
            records=values,
        )
        for case_id, values in by_case.items()
    }
    return _role_prediction_validity_evidence(
        case_evidence,
        expected_case_ids=expected_case_ids,
    )


def _validate_sequence_inventory(
    records: Sequence[object],
    unavailable_horizons: Sequence[Mapping[str, Any]],
    *,
    expected_cases: set[str],
    dataset_role: sequence_artifact.DatasetRole,
) -> None:
    """Require the exact generated mode, origin, horizon, and unavailability inventory."""
    if {str(_sequence_value(record, "case_id")) for record in records} != expected_cases or any(
        _sequence_value(record, "dataset_role") != dataset_role for record in records
    ):
        message = "Transient artifact sequence inventory contradicts saved case membership or role."
        raise ValueError(message)
    expected_records: set[tuple[str, str, int, int | str, int]] = set()
    expected_unavailable: set[tuple[str, str, int, int, int, str]] = set()
    for case_id in expected_cases:
        case_records = tuple(record for record in records if _sequence_value(record, "case_id") == case_id)
        trajectory_lengths = {int(_sequence_value(record, "trajectory_length")) for record in case_records}
        if len(trajectory_lengths) != 1:
            message = f"Transient artifact sequence inventory has inconsistent trajectory lengths for {case_id!r}."
            raise ValueError(message)
        transition_count = next(iter(trajectory_lengths)) - 1
        expected_records.update((case_id, "teacher_forced_one_step", origin, 1, 1) for origin in range(transition_count))
        expected_records.add((case_id, "autonomous_full", 0, "full", transition_count))
        for origin in rollout.default_rolling_origins(transition_count):
            remaining = transition_count - origin
            expected_records.add((case_id, "rolling_origin", origin, "full", remaining))
            for horizon in sequence_artifact.FIXED_HORIZONS:
                if horizon <= remaining:
                    expected_records.add((case_id, "rolling_origin", origin, horizon, horizon))
                else:
                    expected_unavailable.add(
                        (
                            case_id,
                            dataset_role,
                            origin,
                            horizon,
                            remaining,
                            "requested_fixed_horizon_exceeds_case_future_support",
                        )
                    )
    actual_records = {
        (
            str(_sequence_value(record, "case_id")),
            str(_sequence_value(record, "mode")),
            int(_sequence_value(record, "origin_index")),
            _sequence_value(record, "requested_horizon"),
            int(_sequence_value(record, "available_horizon")),
        )
        for record in records
    }
    if len(actual_records) != len(records) or actual_records != expected_records:
        message = "Transient artifact sequence inventory lacks or adds required mode/origin/horizon records."
        raise ValueError(message)
    actual_unavailable = {
        (
            str(item["case_id"]),
            str(item["dataset_role"]),
            int(item["origin_index"]),
            int(item["requested_horizon"]),
            int(item["available_transitions"]),
            str(item["reason"]),
        )
        for item in unavailable_horizons
    }
    if len(actual_unavailable) != len(unavailable_horizons) or actual_unavailable != expected_unavailable:
        message = "Transient artifact sequence inventory has incomplete or contradictory horizon-unavailability evidence."
        raise ValueError(message)


def _validate_role_spatial_representation(
    loaded: sequence_artifact.LoadedTransientSequenceArtifact | sequence_artifact.TransientSequenceArtifactIndex,
    *,
    completed: Mapping[str, Any],
    expected_cases: set[str],
    evaluation_spatial_stride: int,
) -> None:
    """Bind role and record grid evidence to exact completed-run spatial owners."""
    evidence = loaded.provenance.get("spatial_representation")
    required = {
        "inference_contract",
        "source_grid",
        "training_grid",
        "evaluation_grid",
        "checkpoint_identity",
        "architecture_compatibility",
        "scaling_compatibility",
        "field_alignment",
        "case_grids",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required:
        message = "Transient artifact role spatial evidence does not match the current schema."
        raise ValueError(message)
    training, evaluation = resolve_transient_artifact_spatial_representations(
        completed,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
    expected_checkpoint = {
        "identity": dict(cast("Mapping[str, Any]", completed["checkpoint_identity"])),
        "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
        "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
    }
    expected_fields = {
        "dynamic_fields": list(sequence_artifact.STATE_ORDER),
        "static_fields": list(sequence_artifact.STATIC_ORDER),
        "coordinates": ["x", "y"],
        "spatial_mask": "same_evaluation_index_map",
        "model_input": "evaluation_grid",
        "reference": "evaluation_grid",
        "prediction": "evaluation_grid",
        "interpolation": "forbidden",
    }
    if (
        evidence["inference_contract"] != "transient_physical_rollout_evaluation_grid"
        or evidence["source_grid"] != {"shape": list(training.source_shape), "owner": "canonical_generation_dataset_package"}
        or evidence["training_grid"] != training.as_dict()
        or evidence["evaluation_grid"] != evaluation.as_dict()
        or evidence["checkpoint_identity"] != expected_checkpoint
        or evidence["field_alignment"] != expected_fields
    ):
        message = "Transient artifact role spatial identity is stale or incompatible."
        raise ValueError(message)
    scaling = TransientScalingArtifact.from_state_dict(cast("Mapping[str, Any]", completed["normalizer_state"]))
    expected_scaling = scaling.assess_spatial_compatibility(evaluation.represented_shape)
    if evidence["scaling_compatibility"] != expected_scaling:
        message = "Transient artifact scaling compatibility decision is stale or incompatible."
        raise ValueError(message)
    config = cast("Mapping[str, Any]", completed["config"])
    expected_architecture = learning.models.factory.assess_transient_model_spatial_compatibility(
        config,
        training_shape=training.represented_shape,
        evaluation_shape=evaluation.represented_shape,
    )
    architecture = evidence["architecture_compatibility"]
    if not isinstance(architecture, Mapping):
        message = "Transient artifact architecture compatibility evidence must be a mapping."
        raise TypeError(message)
    if architecture.get("decision") not in {"SUPPORTED_EXACTLY", "SUPPORTED_WITH_CONTRACT"}:
        message = "Transient artifact architecture compatibility is unresolved or unsupported."
        raise ValueError(message)
    expected_without_proof = {key: value for key, value in expected_architecture.items() if key != "forward_output_shape_proof"}
    actual_without_proof = {key: value for key, value in architecture.items() if key != "forward_output_shape_proof"}
    if actual_without_proof != expected_without_proof:
        message = "Transient artifact architecture compatibility decision is stale or incompatible."
        raise ValueError(message)
    proof = architecture.get("forward_output_shape_proof")
    if evaluation.represented_shape == training.represented_shape:
        if proof != "checkpoint_training_shape_contract":
            message = "Exact-shape artifact inference lacks its checkpoint Training-shape proof."
            raise ValueError(message)
    else:
        model = config.get("model")
        params = model.get("params") if isinstance(model, Mapping) else None
        input_channels = params.get("in_channels") if isinstance(params, Mapping) else None
        if not isinstance(model, Mapping) or isinstance(input_channels, bool) or not isinstance(input_channels, int) or input_channels < 1:
            message = "Cross-resolution artifact proof requires resolved model kind and positive in_channels."
            raise TypeError(message)
        expected_proof = {
            "kind": "synthetic_single_step_exact_output",
            "input_shape": [1, input_channels, *evaluation.represented_shape],
            "output_shape": [1, len(sequence_artifact.STATE_ORDER), *evaluation.represented_shape],
            "hidden_state_scope": "single_request" if model.get("kind") == "rno" else "not_applicable",
        }
        if proof != expected_proof:
            message = "Cross-resolution artifact inference lacks exact requested-grid output-shape proof."
            raise ValueError(message)
    case_grids = evidence["case_grids"]
    if not isinstance(case_grids, Mapping) or set(case_grids) != expected_cases:
        message = "Transient artifact case-grid provenance does not cover exact case membership."
        raise ValueError(message)
    records: Sequence[object] = loaded.records if isinstance(loaded, sequence_artifact.LoadedTransientSequenceArtifact) else loaded.summaries
    for record in records:
        case_id = str(_sequence_value(record, "case_id"))
        identity = cast("Mapping[str, Any]", _sequence_value(record, "identity"))
        case_grid = identity.get("spatial_representation")
        if case_grid != case_grids.get(case_id):
            message = "Transient artifact record and role-level case-grid identities disagree."
            raise ValueError(message)
        if (
            not isinstance(case_grid, Mapping)
            or case_grid.get("source_shape") != list(evaluation.source_shape)
            or case_grid.get("evaluation_spatial_stride") != evaluation.spatial_stride
            or case_grid.get("evaluation_shape") != list(evaluation.represented_shape)
            or case_grid.get("index_identity_sha256") != evaluation.index_identity_sha256
            or case_grid.get("reference_grid_identity_sha256") != case_grid.get("prediction_grid_identity_sha256")
        ):
            message = "Transient artifact case grid contradicts the requested Evaluation representation."
            raise ValueError(message)


def _validate_transient_role_evidence(
    loaded: (sequence_artifact.LoadedTransientSequenceArtifact | sequence_artifact.TransientSequenceArtifactIndex),
    *,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    evaluation_spatial_stride: int = 1,
) -> None:
    """Bind eager or indexed sequence evidence to one exact completed run role."""
    _reject_trial(completed)
    loaded_role = (loaded.dataset_name, loaded.dataset_role)
    if loaded.provenance["dataset"] != role.dataset_provenance() or loaded_role != (role.dataset_name, role.dataset_role):
        message = "Transient artifact Dataset provenance contradicts saved complete-case membership."
        raise ValueError(message)
    run = loaded.provenance["run"]
    expected_run = {
        "name": completed["scientific_run_name"],
        "effective_config_digest": completed["effective_config_digest"],
        "checkpoint_identity": dict(cast("Mapping[str, Any]", completed["checkpoint_identity"])),
        "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
        "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
        "normalizer_sha256": completed["normalizer_sha256"],
        "parent_experiment": _parent_experiment_evidence(completed),
    }
    if run != expected_run:
        message = "Transient artifact run/checkpoint/scaling identity is stale or incompatible."
        raise ValueError(message)
    if loaded.provenance.get("lineage") != _lineage(completed):
        message = "Transient artifact Training lineage is stale or incompatible."
        raise ValueError(message)
    expected_cases = {case_id for case_ids in role.case_ids_by_source for case_id in case_ids}
    _validate_role_spatial_representation(
        loaded,
        completed=completed,
        expected_cases=expected_cases,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
    records: Sequence[object] = loaded.records if isinstance(loaded, sequence_artifact.LoadedTransientSequenceArtifact) else loaded.summaries
    _validate_sequence_inventory(
        records,
        loaded.unavailable_horizons,
        expected_cases=expected_cases,
        dataset_role=role.dataset_role,
    )
    simulation_records: set[str] = set()
    pt_identities: defaultdict[str, set[object]] = defaultdict(set)
    for record in records:
        record_case_id = str(_sequence_value(record, "case_id"))
        identity = cast("Mapping[str, Any]", _sequence_value(record, "identity"))
        simulation = identity.get("simulation_identity")
        package_case_id = simulation.get("package_case_id") if isinstance(simulation, Mapping) else None
        if package_case_id != record_case_id:
            message = "Transient artifact sequence record lost its qualified Dataset package identity."
            raise ValueError(message)
        if identity.get("dataset_identity") != _case_dataset_identity(role, record_case_id):
            message = "Transient artifact sequence record names a Dataset source that contradicts its saved split part."
            raise ValueError(message)
        simulation_records.add(record_case_id)
        pt_identities[record_case_id].add(identity.get("pt_payload_identity"))
    if simulation_records != expected_cases:
        message = "Transient artifact sequence records do not cover exact saved complete-case membership."
        raise ValueError(message)
    evaluation = loaded.provenance.get("evaluation")
    if not isinstance(evaluation, Mapping):
        message = "Transient artifact Evaluation provenance must be a mapping."
        raise TypeError(message)
    expected_prediction_validity = _role_prediction_validity_from_records(
        records,
        expected_case_ids=expected_cases,
    )
    if evaluation.get("prediction_validity") != expected_prediction_validity:
        message = "Transient role prediction validity contradicts persisted sequence evidence."
        raise ValueError(message)
    completed_config = completed.get("config")
    if not isinstance(completed_config, Mapping):
        message = "Completed transient evidence must retain its resolved configuration."
        raise TypeError(message)
    evaluation_config = completed_config.get("evaluation")
    if not isinstance(evaluation_config, Mapping):
        message = "Completed transient configuration must retain Evaluation settings."
        raise TypeError(message)
    expected_protocol_identity = sequence_artifact.evaluation_protocol_identity(cast("Mapping[str, Any]", evaluation_config))
    expected_protocol = {
        "config_identity": expected_protocol_identity,
        "metrics": evaluation_config.get("metrics"),
        "objective": evaluation_config.get("objective"),
        "modes": list(sequence_artifact.EVALUATION_MODES),
        "fixed_horizons": list(sequence_artifact.FIXED_HORIZONS),
        "rolling_origin_policy": sequence_artifact.ROLLING_ORIGIN_POLICY,
        "recurrent_state_policy": sequence_artifact.RECURRENT_STATE_POLICY,
    }
    if any(evaluation.get(key) != value for key, value in expected_protocol.items()):
        message = "Transient artifact Evaluation protocol is stale or incompatible."
        raise ValueError(message)
    record_protocol_identities = {
        cast("Mapping[str, Any]", _sequence_value(record, "identity")).get("evaluation_config_identity") for record in records
    }
    if record_protocol_identities != {expected_protocol_identity}:
        message = "Transient sequence records contradict the completed-run Evaluation protocol."
        raise ValueError(message)
    if evaluation.get("process_diagnostic_policy") != _process_diagnostic_policy():
        message = "Transient artifact process-diagnostic policy is stale or incompatible."
        raise ValueError(message)
    pipeline = evaluation.get("pipeline_analysis")
    expected_conditions = ["A_comsol_reference", "B_drying_on_comsol_airflow", "C_drying_on_airflow_no"]
    if (
        not isinstance(pipeline, Mapping)
        or pipeline.get("conditions") != expected_conditions
        or not isinstance(pipeline.get("c_available"), bool)
        or pipeline.get("fabricated_prediction_count") != 0
    ):
        message = "Transient pipeline provenance must declare exact A/B/C conditions and zero fabricated predictions."
        raise ValueError(message)
    if pipeline["c_available"]:
        if not isinstance(pipeline.get("metrics"), Mapping) or pipeline.get("c_unavailable_reasons") not in (None, []):
            message = "Available pipeline C requires measured metric evidence and no unavailable reasons."
            raise ValueError(message)
    else:
        reasons = pipeline.get("c_unavailable_reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or "metrics" in pipeline
        ):
            message = "Unavailable pipeline C requires exact reasons and no fabricated metric evidence."
            raise ValueError(message)
    if contracts.validate_runtime_provenance(loaded.provenance.get("runtime")) != 1:
        message = "Transient artifacts are generated one rollout case at a time (inference_batch_size 1)."
        raise ValueError(message)
    timing_block = artifact_timing.timing_block(loaded.provenance, records=frozenset({"report", "component_availability"}))
    _validate_role_timing_evidence(
        timing_block,
        expected_cases=expected_cases,
        pt_identities=pt_identities,
        expected_validity=expected_prediction_validity["cases"],
    )


def validate_transient_role_artifact(
    root: Path | str,
    *,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    evaluation_spatial_stride: int = 1,
) -> sequence_artifact.LoadedTransientSequenceArtifact:
    """Admit every payload and bind it to exact completed-run role evidence."""
    loaded = sequence_artifact.load_transient_sequence_artifact(root)
    _validate_transient_role_evidence(
        loaded,
        completed=completed,
        role=role,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
    return loaded


def validate_staged_transient_role_artifact(
    index: sequence_artifact.TransientSequenceArtifactIndex,
    *,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    evaluation_spatial_stride: int = 1,
) -> sequence_artifact.TransientSequenceArtifactIndex:
    """Bind writer-admitted private staging evidence without rereading payload bytes."""
    if not isinstance(
        index,
        sequence_artifact.TransientSequenceArtifactIndex,
    ):
        message = "Transient staging validation requires one admitted sequence index."
        raise TypeError(message)
    sequence_artifact.validate_transient_sequence_payload_inventory(index)
    _validate_transient_role_evidence(
        index,
        completed=completed,
        role=role,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
    return index


def validate_scoped_transient_role_artifact_index(
    root: Path | str,
    *,
    completed: Mapping[str, Any],
    saved_role: artifact_plan.ArtifactRolePlan,
    evaluation_spatial_stride: int = 1,
) -> sequence_artifact.TransientSequenceArtifactIndex:
    """Admit one exact selected-case subset without treating it as canonical coverage."""
    loaded = sequence_artifact.load_transient_sequence_artifact_index(root)
    evaluation = loaded.provenance.get("evaluation")
    scope = evaluation.get("scope") if isinstance(evaluation, Mapping) else None
    if not isinstance(scope, Mapping) or set(scope) != {
        "kind",
        "canonical_publication_eligible",
        "selected_case_ids",
        "saved_case_ids_by_source",
    }:
        message = "Scoped transient artifact lacks its exact selected-case scope."
        raise ValueError(message)
    selected_case_ids = scope["selected_case_ids"]
    saved_case_ids = scope["saved_case_ids_by_source"]
    if (
        scope["kind"] != "selected_cases"
        or scope["canonical_publication_eligible"] is not False
        or saved_case_ids != [list(values) for values in saved_role.case_ids_by_source]
        or not isinstance(selected_case_ids, list)
        or not selected_case_ids
        or len(selected_case_ids) != len(set(selected_case_ids))
        or any(not isinstance(case_id, str) or not case_id for case_id in selected_case_ids)
    ):
        message = "Scoped transient artifact scope contradicts saved role evidence."
        raise ValueError(message)
    selected_role = select_transient_role_cases(
        saved_role,
        selected_case_ids,
    )
    sequence_artifact.validate_transient_sequence_payload_inventory(loaded)
    _validate_transient_role_evidence(
        loaded,
        completed=completed,
        role=selected_role,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
    return loaded


def validate_transient_role_artifact_index(
    root: Path | str,
    *,
    completed: Mapping[str, Any],
    role: artifact_plan.ArtifactRolePlan,
    evaluation_spatial_stride: int = 1,
) -> sequence_artifact.TransientSequenceArtifactIndex:
    """Admit manifest evidence without opening numerical arrays until selected."""
    loaded = sequence_artifact.load_transient_sequence_artifact_index(root)
    return validate_staged_transient_role_artifact(
        loaded,
        completed=completed,
        role=role,
        evaluation_spatial_stride=evaluation_spatial_stride,
    )
