"""
Generate and validate transient sequence artifacts through current project owners.

This module is the transient task branch of the existing artifact lifecycle. It
selects saved complete-case membership, calls the public transient inference
service, and delegates sequence persistence to Evaluation's strict artifact
contract. It does not implement models, scaling, Dataset readers, or tracking.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from src import common, datasets, domain, experiments, generation, learning
from src.analysis.evaluation import evaluation_transient_artifact as sequence_artifact
from src.analysis.evaluation import evaluation_transient_metrics as transient_metrics
from src.analysis.evaluation import evaluation_transient_rollout as rollout
from src.analysis.evaluation import evaluation_transient_timing as transient_timing
from src.datasets.packages import dataset_packages_builder as package_builder
from src.datasets.packages import dataset_packages_generated_batch as generated_batch
from src.experiments import experiments_run_identity as run_identity
from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

from . import analysis_artifact_performance as artifact_performance
from . import analysis_artifact_timing as artifact_timing

if TYPE_CHECKING:
    from src.datasets.runtime.dataset_runtime_transient import TransientPhysicalDataset
    from src.generation.runtime.generation_runtime_batch import TerminalCaseEvidence
    from src.learning.learning_device import DeviceResolution

TransientArtifactSplit = Literal["eval", "ood"]
_TIMING_REPETITIONS = 3
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


@dataclass(frozen=True, slots=True)
class TransientArtifactRolePlan:
    """Bind one artifact role to immutable package and complete-case membership."""

    split: TransientArtifactSplit
    dataset_role: sequence_artifact.DatasetRole
    dataset_name: str
    source_dataset_ids: tuple[str, ...]
    source_identities: tuple[Mapping[str, Any], ...]
    case_ids_by_source: tuple[tuple[str, ...], ...]
    membership_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransientArtifactPlan:
    """Bind one completed transient run to its available ID and OOD role plans."""

    run_dir: Path
    run_name: str
    id_role: TransientArtifactRolePlan
    ood_role: TransientArtifactRolePlan | None


def _storage_root_from_packages_root(dataset_root: Path | str) -> Path:
    """Resolve and verify the storage root represented by one package-root path."""
    packages_root = Path(dataset_root).expanduser().resolve()
    if packages_root.name != "packages" or packages_root.parent.name != "02_datasets":
        message = "Transient artifact generation requires dataset_root to identify the authoritative <storage>/02_datasets/packages directory."
        raise ValueError(message)
    storage_root = packages_root.parents[1]
    if common.paths.get_dataset_packages_root(storage_root=storage_root).resolve() != packages_root:
        message = "Transient Dataset package root does not map to one authoritative storage root."
        raise ValueError(message)
    return storage_root


def _completed_transient(run_dir: Path | str) -> dict[str, Any]:
    """Admit one completed transient run and reject provisional/static bundles."""
    completed = experiments.run.validate_completed_run(run_dir)
    config = completed.get("config")
    if not isinstance(config, Mapping) or config.get("task") != "transient_drying":
        message = "Transient artifact generation requires one completed transient_drying run."
        raise RuntimeError(message)
    return completed


def _role_plan(
    *,
    split: TransientArtifactSplit,
    dataset_role: sequence_artifact.DatasetRole,
    dataset_ids: Sequence[str],
    identities: Sequence[Mapping[str, Any]],
    role_evidence: Sequence[Mapping[str, Any]],
) -> TransientArtifactRolePlan:
    """Build one strict package-part role plan from admitted transient split evidence."""
    names = tuple(common.paths.validate_logical_name(value, label=f"{dataset_role} dataset_id") for value in dataset_ids)
    if not names or len(names) != len(identities) or len(names) != len(role_evidence):
        message = f"Transient {dataset_role} package identity and membership counts disagree."
        raise ValueError(message)
    case_groups: list[tuple[str, ...]] = []
    digests: list[str] = []
    copied_identities: list[dict[str, Any]] = []
    for index, (dataset_id, identity, evidence) in enumerate(zip(names, identities, role_evidence, strict=True)):
        if not isinstance(identity, Mapping) or identity.get("dataset_id") != dataset_id:
            message = f"Transient {dataset_role}[{index}] Dataset identity disagrees with its configured package."
            raise ValueError(message)
        if not isinstance(evidence, Mapping):
            message = f"Transient {dataset_role}[{index}] membership evidence must be a mapping."
            raise TypeError(message)
        case_ids = evidence.get("case_ids")
        membership_digest = evidence.get("membership_digest")
        if (
            not isinstance(case_ids, list)
            or not case_ids
            or not all(isinstance(case_id, str) and case_id for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
            or not isinstance(membership_digest, str)
        ):
            message = f"Transient {dataset_role}[{index}] complete-case membership is invalid."
            raise ValueError(message)
        case_groups.append(tuple(case_ids))
        digests.append(membership_digest)
        copied_identities.append(dict(identity))
    flattened_case_ids = tuple(case_id for case_ids in case_groups for case_id in case_ids)
    if len(flattened_case_ids) != len(set(flattened_case_ids)):
        message = f"Transient {dataset_role} package parts contain duplicate package-case identities."
        raise ValueError(message)
    dataset_name = names[0] if len(names) == 1 else datasets.contracts.identity.combined_dataset_id(names)
    return TransientArtifactRolePlan(
        split=split,
        dataset_role=dataset_role,
        dataset_name=dataset_name,
        source_dataset_ids=names,
        source_identities=tuple(copied_identities),
        case_ids_by_source=tuple(case_groups),
        membership_digests=tuple(digests),
    )


def transient_artifact_plan_from_completed(
    completed: Mapping[str, Any],
    *,
    run_dir: Path | str,
) -> TransientArtifactPlan:
    """Project exact ID-test and OOD complete-case roles from completed evidence."""
    if completed.get("is_completed") is not True:
        message = "Transient artifact planning requires an admitted completed run."
        raise ValueError(message)
    config = completed.get("config")
    split = completed.get("split_indices")
    if not isinstance(config, Mapping) or config.get("task") != "transient_drying" or not isinstance(split, Mapping):
        message = "Transient artifact planning requires admitted completed-run config and split evidence."
        raise TypeError(message)
    data = config.get("data")
    identities = split.get("dataset_identity")
    roles = split.get("roles")
    if not isinstance(data, Mapping) or not isinstance(identities, Mapping) or not isinstance(roles, Mapping):
        message = "Completed transient config/split lacks data, Dataset identity, or role evidence."
        raise TypeError(message)
    train_dataset = common.paths.validate_logical_name(data.get("train_dataset"), label="data.train_dataset")
    train_identity = identities.get("train")
    id_test = roles.get("id_test")
    id_role = _role_plan(
        split="eval",
        dataset_role="id",
        dataset_ids=(train_dataset,),
        identities=(cast("Mapping[str, Any]", train_identity),),
        role_evidence=(cast("Mapping[str, Any]", id_test),),
    )
    raw_ood_ids = data.get("ood_datasets")
    raw_ood_identities = identities.get("ood")
    raw_ood_role = roles.get("ood")
    ood_role: TransientArtifactRolePlan | None = None
    if raw_ood_ids:
        if (
            not isinstance(raw_ood_ids, list)
            or not isinstance(raw_ood_identities, list)
            or not isinstance(raw_ood_role, Mapping)
            or not isinstance(raw_ood_role.get("parts"), list)
        ):
            message = "Completed transient OOD package and membership evidence is invalid."
            raise TypeError(message)
        ood_role = _role_plan(
            split="ood",
            dataset_role="ood",
            dataset_ids=tuple(raw_ood_ids),
            identities=tuple(raw_ood_identities),
            role_evidence=tuple(raw_ood_role["parts"]),
        )
    run_config = config.get("run")
    if not isinstance(run_config, Mapping) or not isinstance(run_config.get("name"), str):
        message = "Completed transient config lacks scientific run identity."
        raise TypeError(message)
    return TransientArtifactPlan(
        run_dir=Path(run_dir).expanduser().resolve(),
        run_name=str(run_config["name"]),
        id_role=id_role,
        ood_role=ood_role,
    )


def load_transient_artifact_plan(run_dir: Path | str) -> TransientArtifactPlan:
    """Admit a completed run and return exact transient artifact role plans."""
    path = Path(run_dir).expanduser().resolve()
    return transient_artifact_plan_from_completed(_completed_transient(path), run_dir=path)


def select_transient_role_cases(
    role: TransientArtifactRolePlan,
    case_ids: Sequence[str],
) -> TransientArtifactRolePlan:
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
    return TransientArtifactRolePlan(
        split=role.split,
        dataset_role=role.dataset_role,
        dataset_name=role.dataset_name,
        source_dataset_ids=role.source_dataset_ids,
        source_identities=role.source_identities,
        case_ids_by_source=groups,
        membership_digests=role.membership_digests,
    )


def _dataset_identity(dataset: TransientPhysicalDataset) -> dict[str, Any]:
    """Project the same published package identity persisted by Training."""
    payload = dataset.payload
    return {
        "dataset_id": str(payload["dataset_id"]),
        "data_contract_digest": str(payload["contract_digest"]),
        "index_digest": str(payload["index_digest"]),
        "configured_regular_horizon": dict(payload["configured_regular_horizon"]),
    }


def _load_role_datasets(
    role: TransientArtifactRolePlan,
    *,
    config: Mapping[str, Any],
    storage_root: Path,
) -> tuple[TransientPhysicalDataset, ...]:
    """Recreate one-step physical Datasets and enforce saved case membership."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        message = "Completed transient config.data must be a mapping."
        raise TypeError(message)
    backend = data.get("transient_backend_preference", "pt_shards")
    required = data.get("transient_backend_required", False)
    allow_smoke = data.get("allow_technical_smoke", False)
    cache_size = data.get("hdf5_cache_size", 0)
    sampling = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    loaded: list[TransientPhysicalDataset] = []
    for dataset_id, expected_identity, case_ids in zip(
        role.source_dataset_ids,
        role.source_identities,
        role.case_ids_by_source,
        strict=True,
    ):
        if not case_ids:
            continue
        manifest = datasets.packages.manifest.load_package_manifest(
            dataset_id,
            storage_root=storage_root,
        )
        regime = manifest.get("evaluation_regime")
        membership = "id_test" if role.dataset_role == "id" else None
        request = datasets.runtime.factory.DatasetRequest(
            dataset_id=dataset_id,
            dataset_view="transient_drying",
            evaluation_regime=cast("Any", regime),
            membership=cast("Any", membership),
            transient_sampling=sampling,
            storage_root=storage_root,
            allow_technical_smoke=bool(allow_smoke),
            transient_backend_preference=cast("Any", backend),
            transient_backend_required=bool(required),
        )
        candidate = datasets.runtime.factory.create_dataset(
            request,
            hdf5_cache_size=int(cache_size),
        )
        if not isinstance(candidate, datasets.runtime.transient.TransientPhysicalDataset):
            message = "Transient artifact Dataset factory returned a non-transient runtime."
            raise TypeError(message)
        if _dataset_identity(candidate) != dict(expected_identity):
            candidate.close()
            message = f"Current package {dataset_id!r} contradicts saved transient Dataset identity."
            raise RuntimeError(message)
        selected = datasets.runtime.transient.select_transient_cases(candidate, case_ids)
        candidate.close()
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
    dataset_backend: str
    pt_identity: Mapping[str, Any] | None


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
        pt_identity: Mapping[str, Any] | None = None
        if dataset.storage_backend == "pt_shards":
            receipt = datasets.packages.transient_shards.load_transient_shard_receipt(
                dataset.dataset_id,
                storage_root=dataset.source_root,
                validation_depth="evidence",
            )
            pt_identity = {
                "receipt_digest": common.serialization.canonical_json_sha256(receipt),
                "index_digest": receipt["index_digest"],
            }
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
                    dataset_backend=dataset.storage_backend,
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
) -> tuple[str, Mapping[str, Any], rollout.TransientEvaluationCase, str, Mapping[str, Any] | None]:
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
        spec.dataset_backend,
        spec.pt_identity,
    )


def _materialize_cases(
    sources: Sequence[TransientPhysicalDataset],
    *,
    dataset_role: sequence_artifact.DatasetRole,
) -> tuple[tuple[str, Mapping[str, Any], rollout.TransientEvaluationCase, str, Mapping[str, Any] | None], ...]:
    """Compatibility wrapper that materializes selected cases in exact plan order."""
    return tuple(_materialize_case(spec, dataset_role=dataset_role) for spec in _case_materialization_specs(sources))


def _terminal_case(
    *,
    package_case_id: str,
    case_record: Mapping[str, Any],
    storage_root: Path,
) -> tuple[Any, TerminalCaseEvidence]:
    """Resolve the Generation terminal evidence named by one Dataset case."""
    batch_id = case_record.get("source_batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        message = f"Transient package case {package_case_id!r} lacks source_batch_id."
        raise TypeError(message)
    source_relative_path = case_record.get("source_relative_path")
    if not isinstance(source_relative_path, str):
        message = f"Transient package case {package_case_id!r} lacks source_relative_path."
        raise TypeError(message)
    relative_parts = Path(source_relative_path).parts
    match relative_parts:
        case ("01_generation", "processed", batch_storage_name, source_case_id, "case.h5"):
            pass
        case _:
            message = f"Transient package case {package_case_id!r} has an invalid Generation source path."
            raise ValueError(message)
    batch = generation.runtime.admit_terminal_batch(
        batch_storage_name,
        storage_root=storage_root,
        validation_depth="routine",
    )
    if batch.batch_id != batch_id:
        message = f"Transient package case {package_case_id!r} conflicts with its terminal Generation batch identity."
        raise RuntimeError(message)
    matches = tuple(
        case for case in batch.cases if case.case_id == source_case_id and case.simulation_case_id == case_record.get("simulation_case_id")
    )
    if len(matches) != 1:
        message = f"Transient package case {package_case_id!r} does not map uniquely to terminal Generation evidence."
        raise RuntimeError(message)
    return batch, matches[0]


def _generation_case_sources(
    role: TransientArtifactRolePlan,
    *,
    storage_root: Path,
) -> dict[str, tuple[Any, TerminalCaseEvidence]]:
    """Re-admit composite-backed package cases through the Dataset publication owner."""
    result: dict[str, tuple[Any, TerminalCaseEvidence]] = {}
    for dataset_id in role.source_dataset_ids:
        prepared = package_builder.admit_package_composite_sources(
            dataset_id,
            storage_root=storage_root,
        )
        if prepared is None:
            continue
        for candidate in prepared.candidates:
            package_case_id = str(candidate["package_case_id"])
            if package_case_id in result:
                message = f"Transient artifact packages contain duplicate Generation case evidence: {package_case_id!r}."
                raise ValueError(message)
            result[package_case_id] = (
                candidate["terminal_evidence"],
                candidate["case_evidence"],
            )
    return result


def _canonical_case_evidence(
    case: rollout.TransientEvaluationCase,
    *,
    package_case_id: str,
    case_record: Mapping[str, Any],
    storage_root: Path,
    generation_sources: Mapping[str, tuple[Any, TerminalCaseEvidence]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-check Dataset items against canonical Generation case semantics."""
    source = generation_sources.get(package_case_id)
    if source is None:
        source = _terminal_case(
            package_case_id=package_case_id,
            case_record=case_record,
            storage_root=storage_root,
        )
    batch, terminal_case = source
    task = domain.tasks.registry.get_task("transient_drying")
    canonical = generated_batch.interpret_generated_transient_case(
        batch,
        terminal_case,
        task=task,
    )
    reference = np.stack(
        [canonical["state_trajectories"][name] for name in sequence_artifact.STATE_ORDER],
        axis=1,
    )
    static = np.stack(
        [canonical["static_fields"][name] for name in sequence_artifact.STATIC_ORDER],
        axis=0,
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
    expected_arm = {"a0": "A0", "a_plus": "A+", "b": "B"}
    if stage not in {"a", "b"} or arm not in expected_arm or checkpoint_controller.get("arm") != expected_arm[arm]:
        message = "Completed transient stage/comparison arm identity is unsupported or contradictory."
        raise ValueError(message)
    teacher_handoff = checkpoint_controller.get("teacher_handoff")
    if arm == "a0":
        if teacher_handoff is not None:
            message = "A0 lineage must not consume a teacher handoff."
            raise ValueError(message)
        task = experiments.config.loader.validate_resolved_task_contract(config)
        scaling = TransientScalingArtifact.from_state_dict(cast("Mapping[str, Any]", completed["normalizer_state"]))
        model = cast("Mapping[str, Any]", config["model"])
        parent = {
            "schema_version": 1,
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
            message = "A+ and B lineage require checkpoint-persisted teacher-handoff identity."
            raise TypeError(message)
        parent = dict(teacher_handoff)
    strategy = "rollout" if arm == "b" else "teacher_forced_continuation" if arm == "a_plus" else "stage_a_teacher_forced"
    return {
        "stage_identity": {"stage": stage, "comparison_arm": arm},
        "training_strategy": strategy,
        "curriculum_identity": dict(terminal_curriculum),
        "parent_checkpoint": parent,
        "stage_a_handoff": parent,
        "matched_compute_manifest": {
            "planned": training.get("matched_compute"),
            "actual": dict(terminal_controller),
        },
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


def _base_identity(
    *,
    completed: Mapping[str, Any],
    role: TransientArtifactRolePlan,
    case: rollout.TransientEvaluationCase,
    canonical: Mapping[str, Any],
    dataset_backend: str,
    pt_identity: Mapping[str, Any] | None,
    runtime_identity: Mapping[str, Any],
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
    return {
        "case_id": case.case_id,
        "dataset_identity": {
            "artifact_dataset_name": role.dataset_name,
            "source_dataset_id": case.metadata["dataset_id"],
            "saved_membership_digests": list(role.membership_digests),
            "source_index_digest": next(
                identity["index_digest"] for identity in role.source_identities if identity["dataset_id"] == case.metadata["dataset_id"]
            ),
        },
        "dataset_role": role.dataset_role,
        "material_family": str(case.metadata["material_family"]),
        "simulation_identity": {
            "simulation_case_id": source["simulation_case_id"],
            "case_input_id": source["case_input_id"],
            "source_batch_id": case.metadata["source_batch_id"],
            "package_case_id": case.case_id,
            "generation_case_id": source["case_id"],
            "simulation_profile": source["simulation_profile"],
            "template_sha256": source["template_sha256"],
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
        "training_airflow_source": task.training_airflow_source,
        "inference_airflow_source": "comsol_reference",
        **lineage,
        "dataset_backend": dataset_backend,
        "pt_payload_identity": None if pt_identity is None else dict(pt_identity),
        "evaluation_config_identity": sequence_artifact.evaluation_protocol_identity(cast("Mapping[str, Any]", config["evaluation"])),
        "timing_evidence_identity": common.serialization.canonical_json_sha256(runtime_identity),
    }


def _timing_case(
    *,
    case_id: str,
    benchmark: rollout.TransientRolloutBenchmark,
    generation_timing: Mapping[str, Any],
    dataset_backend: str,
    pt_identity: Mapping[str, Any] | None,
    runtime_metadata: Mapping[str, Any],
) -> transient_timing.TransientTimingCase:
    """Adapt stable Generation and public-inference clocks without inference."""
    repetitions: dict[str, tuple[float, ...]] = {
        "drying_no_end_to_end_seconds": benchmark.warmed_end_to_end_seconds,
    }
    cold: dict[str, float] = {
        "drying_no_end_to_end_seconds": benchmark.cold_end_to_end_seconds,
    }
    if all(value > 0.0 for value in benchmark.warmed_model_seconds):
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
    for component, source in source_by_component.items():
        value = generation_timing.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0:
            repetitions[component] = (float(value),)
            unavailable.pop(component, None)
        else:
            unavailable[component] = str(availability.get(source, f"{source}_unavailable"))
    for component in (
        "airflow_no_model_seconds",
        "airflow_no_preprocessing_seconds",
        "airflow_no_device_transfer_seconds",
        "airflow_no_postprocessing_seconds",
        "airflow_no_end_to_end_seconds",
    ):
        unavailable[component] = "compatible_airflow_model_not_selected"
    for component in (
        "drying_no_preprocessing_seconds",
        "drying_no_device_transfer_seconds",
        "drying_no_postprocessing_seconds",
    ):
        unavailable[component] = "public_transient_inference_does_not_expose_separate_component_clock"
    if "drying_no_rollout_model_seconds" not in repetitions:
        unavailable["drying_no_rollout_model_seconds"] = "model_clock_resolution_did_not_produce_a_positive_duration"
    for component in (
        "surrogate_pipeline_model_seconds",
        "surrogate_pipeline_end_to_end_seconds",
    ):
        unavailable[component] = "complete_airflow_to_drying_pipeline_not_selected"

    processor = runtime_metadata.get("processor")
    cpu = str(processor) if isinstance(processor, str) and processor else "unknown"
    gpu_value = runtime_metadata.get("cuda_device_name")
    gpu = str(gpu_value) if isinstance(gpu_value, str) and gpu_value else None
    pt_digest = None if pt_identity is None else common.serialization.canonical_json_sha256(pt_identity)
    return transient_timing.TransientTimingCase(
        case_id=case_id,
        repetitions=repetitions,
        device=str(runtime_metadata["resolved_device"]),
        precision="float32",
        dataset_backend=dataset_backend,
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
        pt_payload_identity=pt_digest,
    )


def _parent_experiment_evidence(completed: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact grouped-parent evidence or explicit persisted legacy absence."""
    run_dir = completed.get("run_dir")
    summary = completed.get("summary")
    config = completed.get("config")
    if not isinstance(run_dir, Path) or not isinstance(summary, Mapping) or not isinstance(config, Mapping):
        message = "Completed transient evidence lacks its canonical run path, summary, or config."
        raise TypeError(message)
    run_evidence = summary.get("run_identity")
    child_source: dict[str, Any] | None = None
    parent_label: str | None = None
    if isinstance(run_evidence, Mapping):
        source = run_evidence.get("source_repository")
        if isinstance(source, Mapping):
            child_source = dict(source)
        raw_parent = run_evidence.get("parent_label")
        if isinstance(raw_parent, str) and raw_parent:
            parent_label = raw_parent
    legacy = {
        "kind": "legacy",
        "parent_available": False,
        "reason": ("child_run_identity_lacks_parent_label" if parent_label is None else "parent_experiment_record_not_persisted"),
        "child_source_repository": child_source,
    }
    if parent_label is None:
        return legacy
    task = config.get("task")
    if not isinstance(task, str) or run_dir.parent.name != "runs" or run_dir.parent.parent.name != task:
        message = "Completed transient run path does not match its persisted task layout."
        raise ValueError(message)
    output_root = run_dir.parents[2]
    marker = run_identity.experiment_record_path(
        {"task": task, "parent_label": parent_label},
        output_root=output_root,
    )
    if not marker.exists():
        return legacy
    if not marker.is_file() or marker.is_symlink():
        message = f"Transient parent experiment record is unsafe: {marker}."
        raise ValueError(message)
    with marker.open("r", encoding="utf-8") as handle:
        record = run_identity.validate_transient_experiment_record(json.load(handle))
    matching_children = [child for child in record["children"].values() if Path(str(child["path"])).expanduser().resolve() == run_dir.resolve()]
    if len(matching_children) != 1:
        message = "Transient parent experiment does not bind the exact completed child path."
        raise ValueError(message)
    child = matching_children[0]
    if child["run_name"] != completed.get("scientific_run_name") or child["resolved_config_sha256"] != completed.get("effective_config_digest"):
        message = "Transient parent experiment child identity contradicts the completed run."
        raise ValueError(message)
    return {
        "kind": "grouped",
        "parent_available": True,
        "parent_label": record["parent_label"],
        "parent_identity_sha256": record["parent_identity_sha256"],
        "run_revision": record["run_revision"],
        "source_repository": dict(record["source_repository"]),
        "child_source_repository": child_source,
    }


def _record_metric_statistics(
    record: sequence_artifact.TransientSequenceRecord,
    *,
    scaling: TransientScalingArtifact,
) -> dict[str, Mapping[str, object]]:
    """Build exact float64 cumulative and endpoint sufficient statistics for one record."""
    result: dict[str, Mapping[str, object]] = {}
    density_index = sequence_artifact.STATIC_ORDER.index("rho_bu_dry")
    cumulative_prediction = record.predicted_states[1:]
    cumulative_reference = record.reference_states[1:]
    normalized_prediction = _normalized_metric_state(
        cumulative_prediction,
        scaling=scaling,
    )
    normalized_reference = _normalized_metric_state(
        cumulative_reference,
        scaling=scaling,
    )
    cell_weights = transient_metrics.trapezoidal_cell_weights(record.spatial_mask)
    for scope in ("cumulative", "endpoint"):
        selection = slice(None) if scope == "cumulative" else slice(-1, None)
        accumulator = transient_metrics.TransientMetricAccumulator(scope=scope)
        accumulator.update(
            normalized_prediction=normalized_prediction[selection],
            normalized_reference=normalized_reference[selection],
            physical_prediction=cumulative_prediction[selection],
            physical_reference=cumulative_reference[selection],
            f_surf=np.asarray(record.scalar_conditioning[2]),
            rho_bu_dry=record.static_conditioning[density_index],
            cell_weights=cell_weights,
            valid_mask=record.spatial_mask[None, None],
        )
        result[scope] = accumulator.state_dict()
    result["diagnostics"] = {
        "plausibility": asdict(
            transient_metrics.derive_plausibility_diagnostics(
                record.predicted_states[1:],
                temperature_range=transient_metrics.TEMPERATURE_PLAUSIBILITY_RANGE_K,
            )
        ),
        "stability": asdict(transient_metrics.derive_stability_diagnostics(record.predicted_states)),
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


def _role_provenance(
    *,
    completed: Mapping[str, Any],
    role: TransientArtifactRolePlan,
    device_resolution: DeviceResolution,
    component_availability: Mapping[str, Any],
    timing_report: Mapping[str, Any],
    saved_role: TransientArtifactRolePlan | None = None,
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
        "run": {
            "name": completed["scientific_run_name"],
            "effective_config_digest": completed["effective_config_digest"],
            "checkpoint_identity": dict(cast("Mapping[str, Any]", completed["checkpoint_identity"])),
            "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
            "normalizer_sha256": completed["normalizer_sha256"],
            "lifecycle_status": "completed",
            "parent_experiment": _parent_experiment_evidence(completed),
        },
        "dataset": {
            "name": role.dataset_name,
            "role": role.dataset_role,
            "source_dataset_ids": list(role.source_dataset_ids),
            "source_identities": [dict(value) for value in role.source_identities],
            "case_ids_by_source": [list(values) for values in role.case_ids_by_source],
            "membership_digests": list(role.membership_digests),
        },
        "evaluation": {
            "config_identity": sequence_artifact.evaluation_protocol_identity(cast("Mapping[str, Any]", config["evaluation"])),
            "metrics": config["evaluation"]["metrics"],
            "objective": config["evaluation"]["objective"],
            "modes": list(sequence_artifact.EVALUATION_MODES),
            "fixed_horizons": list(sequence_artifact.FIXED_HORIZONS),
            "rolling_origin_policy": sequence_artifact.ROLLING_ORIGIN_POLICY,
            "process_diagnostic_policy": _process_diagnostic_policy(),
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
            "component_availability": dict(component_availability),
            "timing_report": dict(timing_report),
        },
        "runtime": {
            **device_resolution.as_dict(),
            "precision": "float32",
            "inference_batch_size": 1,
        },
        "lineage": _lineage(completed),
    }
    if scope is not None:
        result["evaluation"]["scope"] = scope
    return result


def generate_transient_role_artifact(
    *,
    run_dir: Path | str,
    role: TransientArtifactRolePlan,
    device_resolution: DeviceResolution,
    dataset_root: Path | str,
    staging_root: Path | str,
    saved_role: TransientArtifactRolePlan | None = None,
    progress_reporter: artifact_performance.ArtifactProgressReporter | None = None,
) -> sequence_artifact.TransientSequenceArtifactIndex:
    """Generate one role through bounded case inference, metrics, and staging."""
    preflight_phase = nullcontext() if progress_reporter is None else progress_reporter.phase("preflight")
    with preflight_phase:
        completed = _completed_transient(run_dir)
        config = cast("Mapping[str, Any]", completed["config"])
        storage_root = _storage_root_from_packages_root(dataset_root)
        generation_sources = _generation_case_sources(
            role,
            storage_root=storage_root,
        )
    model_phase = nullcontext() if progress_reporter is None else progress_reporter.phase("model_setup")
    with model_phase:
        context = learning.inference.transient.load_transient_inference_context(
            run_dir=run_dir,
            device=device_resolution.device,
            precision="float32",
        )
        datasets_for_role = _load_role_datasets(
            role,
            config=config,
            storage_root=storage_root,
        )
        runtime_metadata = artifact_timing.neural_runtime_metadata(
            device_metadata=device_resolution.as_dict(),
            model=context.model,
        )
        if progress_reporter is not None:
            model_parameter = next(context.model.parameters(), None)
            model_device = context.device if model_parameter is None else model_parameter.device
            progress_reporter.device_summary(
                model_device=model_device,
                scaler_device=context.scaling.device,
            )
    component_availability: dict[str, Any] = {}
    timing_cases: list[transient_timing.TransientTimingCase] = []
    produced_case_identities: list[Mapping[str, Any]] = []
    index: sequence_artifact.TransientSequenceArtifactIndex | None = None
    try:
        specs = _case_materialization_specs(datasets_for_role)
        planned_membership = [(spec.package_case_id, spec.material_family) for spec in specs]
        admitted_planned = _qualified_case_membership(
            planned=planned_membership,
        )
        expected_case_ids = {case_id for case_ids in role.case_ids_by_source for case_id in case_ids}
        if {case_id for case_id, _material in admitted_planned} != expected_case_ids:
            message = "Transient artifact selected cases contradict the exact saved split before inference."
            raise ValueError(message)
        stager = sequence_artifact.TransientSequenceArtifactStager(
            staging_root,
            dataset_name=role.dataset_name,
            dataset_role=role.dataset_role,
        )
        for spec in specs:
            inference_phase = nullcontext() if progress_reporter is None else progress_reporter.work_phase("inference")
            with inference_phase:
                package_case_id, case_record, case, backend, pt_identity = _materialize_case(
                    spec,
                    dataset_role=role.dataset_role,
                )
                canonical, runtime_identity = _canonical_case_evidence(
                    case,
                    package_case_id=package_case_id,
                    case_record=case_record,
                    storage_root=storage_root,
                    generation_sources=generation_sources,
                )
                completion = canonical["completion"]
                target_limit = completion.get("target_wet_fraction_limit")
                target_wet_basis = canonical["scalar_conditioning"].get(
                    "X_target_wb",
                )
                if (
                    isinstance(target_limit, bool)
                    or not isinstance(target_limit, (int, float))
                    or isinstance(target_wet_basis, bool)
                    or not isinstance(target_wet_basis, (int, float))
                ):
                    message = f"Canonical target evidence is unavailable for transient case {package_case_id!r}."
                    raise TypeError(message)
                identity = _base_identity(
                    completed=completed,
                    role=role,
                    case=case,
                    canonical=canonical,
                    dataset_backend=backend,
                    pt_identity=pt_identity,
                    runtime_identity=runtime_identity,
                )
                _qualified_case_membership(
                    planned=((spec.package_case_id, spec.material_family),),
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
                    generation_timing=runtime_identity,
                    dataset_backend=backend,
                    pt_identity=pt_identity,
                    runtime_metadata=runtime_metadata,
                )
                timing_cases.append(timing_case)
                evaluated = rollout.evaluate_transient_case(
                    context,
                    case,
                    identity=identity,
                    reference_completion=_generation_target_completion(
                        completion,
                    ),
                    target_wet_basis=float(target_wet_basis),
                    target_fraction_limit=float(target_limit),
                    prepared_case=prepared_case,
                )
                component_availability[case.case_id] = {
                    "dataset_materialization_seconds": case.metadata["dataset_materialization_seconds"],
                    "generation": dict(runtime_identity),
                    "timing_case": asdict(timing_case),
                    "model_clock": benchmark.model_clock,
                    "wall_clock": benchmark.wall_clock,
                    "airflow_model": {
                        "available": False,
                        "reason": "compatible_airflow_model_not_selected",
                    },
                    "drying_model": {
                        "available": True,
                        "source": "bounded_public_transient_full_rollout_benchmark",
                    },
                    "surrogate_pipeline": {
                        "available": False,
                        "reason": "complete_airflow_to_drying_pipeline_not_selected",
                    },
                }
                benchmark_calls = case.transition_count * (1 + benchmark.warmup_passes + benchmark.repetitions)
                benchmark_timed_calls = case.transition_count * (1 + benchmark.repetitions)
                evaluated_calls = 2 * case.transition_count + sum(
                    case.transition_count - origin
                    for origin in rollout.default_rolling_origins(
                        case.transition_count,
                    )
                )
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
                    unavailable_horizons=evaluated.unavailable_horizons,
                    metric_statistics=case_statistics,
                )
            if progress_reporter is not None:
                progress_reporter.case_completed(
                    case_id=case.case_id,
                    material=material,
                    forward_calls=benchmark_calls + evaluated_calls,
                    timed_forward_calls=(benchmark_timed_calls + evaluated_calls),
                    model_forward_seconds=(benchmark_model_seconds + evaluated_model_seconds),
                )
            del evaluated, case_statistics, case
        _qualified_case_membership(
            planned=planned_membership,
            produced=produced_case_identities,
        )
        timing_report = asdict(
            transient_timing.build_transient_timing_report(timing_cases),
        )
        provenance = _role_provenance(
            completed=completed,
            role=role,
            device_resolution=device_resolution,
            component_availability=component_availability,
            timing_report=timing_report,
            saved_role=saved_role,
        )
        serialization_phase = nullcontext() if progress_reporter is None else progress_reporter.work_phase("serialization")
        with serialization_phase:
            index = stager.finalize(provenance=provenance)
    finally:
        for dataset in datasets_for_role:
            dataset.close()
    if progress_reporter is not None:
        for phase_name in ("inference", "metrics", "serialization"):
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


def _validate_transient_role_evidence(
    loaded: (sequence_artifact.LoadedTransientSequenceArtifact | sequence_artifact.TransientSequenceArtifactIndex),
    *,
    completed: Mapping[str, Any],
    role: TransientArtifactRolePlan,
    allow_legacy_scoped_package_identity: bool = False,
) -> None:
    """Bind eager or indexed sequence evidence to one exact completed run role."""
    if (
        loaded.dataset_name != role.dataset_name
        or loaded.dataset_role != role.dataset_role
        or loaded.provenance["dataset"]["source_dataset_ids"] != list(role.source_dataset_ids)
        or loaded.provenance["dataset"]["source_identities"] != [dict(value) for value in role.source_identities]
        or loaded.provenance["dataset"]["case_ids_by_source"] != [list(values) for values in role.case_ids_by_source]
        or loaded.provenance["dataset"]["membership_digests"] != list(role.membership_digests)
    ):
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
        "lifecycle_status": "completed",
        "parent_experiment": _parent_experiment_evidence(completed),
    }
    if run != expected_run:
        message = "Transient artifact run/checkpoint/scaling identity is stale or incompatible."
        raise ValueError(message)
    if loaded.provenance.get("lineage") != _lineage(completed):
        message = "Transient artifact Training lineage is stale or incompatible."
        raise ValueError(message)
    expected_cases = {case_id for case_ids in role.case_ids_by_source for case_id in case_ids}
    records: Sequence[object] = loaded.records if isinstance(loaded, sequence_artifact.LoadedTransientSequenceArtifact) else loaded.summaries
    _validate_sequence_inventory(
        records,
        loaded.unavailable_horizons,
        expected_cases=expected_cases,
        dataset_role=role.dataset_role,
    )
    simulation_records: set[str] = set()
    for record in records:
        record_case_id = str(_sequence_value(record, "case_id"))
        identity = cast("Mapping[str, Any]", _sequence_value(record, "identity"))
        simulation = identity.get("simulation_identity")
        package_case_id = simulation.get("package_case_id") if isinstance(simulation, Mapping) else None
        legacy_scoped_identity = (
            allow_legacy_scoped_package_identity
            and isinstance(simulation, Mapping)
            and isinstance(package_case_id, str)
            and bool(package_case_id)
            and "generation_case_id" not in simulation
        )
        if package_case_id != record_case_id and not legacy_scoped_identity:
            message = "Transient artifact sequence record lost its qualified Dataset package identity."
            raise ValueError(message)
        simulation_records.add(record_case_id)
    if simulation_records != expected_cases:
        message = "Transient artifact sequence records do not cover exact saved complete-case membership."
        raise ValueError(message)
    evaluation = loaded.provenance.get("evaluation")
    if not isinstance(evaluation, Mapping):
        message = "Transient artifact Evaluation provenance must be a mapping."
        raise TypeError(message)
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
    component_evidence = evaluation.get("component_availability")
    if not isinstance(component_evidence, Mapping) or set(component_evidence) != expected_cases:
        message = "Transient artifact timing-component evidence does not cover exact case membership."
        raise ValueError(message)
    report = transient_timing.admit_transient_timing_report(cast("Mapping[str, object]", evaluation.get("timing_report")))
    if {case.case_id for case in report.cases} != expected_cases:
        message = "Transient timing report does not cover exact saved complete-case membership."
        raise ValueError(message)


def validate_transient_role_artifact(
    root: Path | str,
    *,
    completed: Mapping[str, Any],
    role: TransientArtifactRolePlan,
) -> sequence_artifact.LoadedTransientSequenceArtifact:
    """Admit every payload and bind it to exact completed-run role evidence."""
    loaded = sequence_artifact.load_transient_sequence_artifact(root)
    _validate_transient_role_evidence(
        loaded,
        completed=completed,
        role=role,
    )
    return loaded


def validate_staged_transient_role_artifact(
    index: sequence_artifact.TransientSequenceArtifactIndex,
    *,
    completed: Mapping[str, Any],
    role: TransientArtifactRolePlan,
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
    )
    return index


def validate_scoped_transient_role_artifact_index(
    root: Path | str,
    *,
    completed: Mapping[str, Any],
    saved_role: TransientArtifactRolePlan,
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
        allow_legacy_scoped_package_identity=True,
    )
    return loaded


def validate_transient_role_artifact_index(
    root: Path | str,
    *,
    completed: Mapping[str, Any],
    role: TransientArtifactRolePlan,
) -> sequence_artifact.TransientSequenceArtifactIndex:
    """Admit manifest evidence without opening numerical arrays until selected."""
    loaded = sequence_artifact.load_transient_sequence_artifact_index(root)
    return validate_staged_transient_role_artifact(
        loaded,
        completed=completed,
        role=role,
    )


def load_transient_role_artifact(
    root: Path | str,
    *,
    run_dir: Path | str,
    split: TransientArtifactSplit,
) -> sequence_artifact.LoadedTransientSequenceArtifact:
    """Admit a run-owned transient artifact role through completed-run validation."""
    completed = _completed_transient(run_dir)
    plan = transient_artifact_plan_from_completed(completed, run_dir=run_dir)
    role = plan.id_role if split == "eval" else plan.ood_role
    if role is None:
        message = "Completed transient run has no saved OOD role."
        raise FileNotFoundError(message)
    return validate_transient_role_artifact(root, completed=completed, role=role)
