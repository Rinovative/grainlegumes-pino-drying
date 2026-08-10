"""
===============================================================================
dataset_packages.py
===============================================================================
Assemble immutable dual-view dataset packages from terminal simulation cases.
Responsibilities:
  - Build provenance-bound steady payloads or compact transient indexes
  - Compute immutable package identities and publish exact manifests atomically
  - Load, inspect, smoke-test, and expose the dataset-package CLI
Design principles:
  - Package identity is independent of operational source locations
  - One combined parameter-OOD package owns compact group/parameter indexes
  - Existing valid identities are reused and conflicting publication fails closed
This module does NOT:
  - Select source cases, assign membership, or audit OOD eligibility
  - Fit normalization, train models, duplicate HDF5 cases, or register transient
===============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch

from src import common, domain, generation

from . import dataset_factory as factory
from . import dataset_generated_batch as generated
from . import dataset_identity as identity
from . import dataset_package_manifest as package_manifest
from . import dataset_package_planning as planning
from . import dataset_transient as transient
from . import dataset_transient_contract as transient_contract
from . import dataset_views as views

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

DATASET_PACKAGE_SCHEMA_KIND: Final = package_manifest.DATASET_PACKAGE_SCHEMA_KIND
DATASET_PACKAGE_SCHEMA_VERSION: Final = package_manifest.DATASET_PACKAGE_SCHEMA_VERSION
_TRANSIENT_PROFILE_CONTRACT: Final = generation.contracts.get_profile_contract("transient_drying")


def _channel_contract(dataset_view: str) -> dict[str, Any]:
    """Return one authoritative physical-unit tensor contract for the manifest."""
    if dataset_view == "transient_drying":
        return transient_contract.transient_contract_payload()
    task = domain.tasks.registry.get_task("steady_flow")
    return {
        "input": [field.as_dict() for field in task.inputs],
        "target": [field.as_dict() for field in task.outputs],
        "tensor_layout": list(task.tensor_layout),
    }


def _case_indexes(
    prepared: planning.PreparedPackage,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Build compact included, OOD-group, and OOD-parameter case indexes."""
    included = [str(candidate["package_case_id"]) for candidate in prepared.candidates]
    groups: dict[str, list[str]] = defaultdict(list)
    parameters: dict[str, list[str]] = defaultdict(list)
    relevant: set[str] = set()
    for candidate in prepared.candidates:
        case_id = str(candidate["package_case_id"])
        evidence = candidate.get("ood_evidence")
        if isinstance(evidence, dict):
            group = evidence.get("group")
            if isinstance(group, str):
                groups[group].append(case_id)
        for parameter in candidate.get("task_relevant_ood_parameters", []):
            name = str(parameter)
            parameters[name].append(case_id)
            relevant.add(name)
    return (
        included,
        dict(sorted(groups.items())),
        dict(sorted(parameters.items())),
        sorted(relevant),
    )


def _package_provenance(
    campaign: generation.config.CampaignConfig,
    prepared: planning.PreparedPackage,
) -> dict[str, Any]:
    """Return complete portable provenance before dataset identity binding."""
    plan = prepared.plan
    view = views.get_view(str(plan["dataset_view"]))
    included, group_indexes, parameter_indexes, relevant_parameters = _case_indexes(prepared)
    profiles_found = sorted({str(record["simulation_profile"]) for record in prepared.batch_records})
    template_digests = sorted({str(record["template"]["sha256"]) for record in prepared.batch_records})
    airflow = sorted({str(record["airflow_source"]) for record in prepared.batch_records})
    operation_digests = sorted({str(record["operation_config_digest"]) for record in prepared.batch_records})
    git_commits = sorted({str(record["git_commit"]) for record in prepared.batch_records})
    case_membership = {str(candidate["package_case_id"]): str(candidate["dataset_membership"]) for candidate in prepared.candidates}
    material_by_batch_name = {batch.batch_name: batch.material_family for batch in campaign.batches}
    material_counts: dict[str, int] = defaultdict(int)
    profile_counts: dict[str, int] = defaultdict(int)
    for candidate in prepared.candidates:
        material_counts[str(candidate["material_family"])] += 1
        profile_counts[str(candidate["simulation_profile"])] += 1
    return {
        "schema_kind": DATASET_PACKAGE_SCHEMA_KIND,
        "schema_version": DATASET_PACKAGE_SCHEMA_VERSION,
        "dataset_name": plan["dataset_name"],
        "dataset_view": view.id,
        "registered_task_id": view.registered_task_id,
        "evaluation_regime": plan["evaluation_regime"],
        "materials": list(plan["materials"]),
        "channel_contract": _channel_contract(view.id),
        "channel_contract_digest": view.contract_digest,
        "source_simulation_profiles": profiles_found,
        "source_batches": prepared.batch_records,
        "source_batch_ids": [record["batch_id"] for record in prepared.batch_records],
        "source_template_digests": template_digests,
        "source_git_commits": git_commits,
        "source_case_identities": [
            {
                "package_case_id": candidate["package_case_id"],
                "batch_id": candidate["batch_id"],
                "source_case_id": candidate["case_id"],
                "source_relative_path": candidate["case_hdf5_relative"],
                "case_input_id": candidate["case_input_id"],
                "simulation_case_id": candidate["simulation_case_id"],
                "case_hdf5_sha256": candidate["case_hdf5_sha256"],
                "material_family": candidate["material_family"],
                "material_role": candidate["case_payload"]["material_role"],
                "evaluation_regime": candidate["case_payload"]["evaluation_regime"],
                "natural_support_state": candidate["case_payload"]["natural_support_state"],
                "simulation_profile": candidate["simulation_profile"],
                "membership": candidate["dataset_membership"],
                "ood_group": candidate.get("ood_evidence", {}).get("group"),
                "ood_parameters": candidate.get("ood_evidence", {}).get("selected_units", []),
                "task_relevant_ood_parameters": candidate.get("task_relevant_ood_parameters", []),
                "ood_evidence": copy.deepcopy(candidate.get("ood_evidence", {})),
            }
            for candidate in prepared.candidates
        ],
        "included_source_cases": included,
        "excluded_source_cases": prepared.excluded,
        "source_selection_decisions": prepared.source_decisions,
        "matched_case_input_ids": sorted(
            str(candidate["case_input_id"]) for candidate in prepared.candidates if candidate["simulation_profile"] == _TRANSIENT_PROFILE_CONTRACT.id
        ),
        "airflow_provenance": airflow if view.id == "steady_flow" else [],
        "steady_flow_conditioning": prepared.steady_conditioning,
        "material_file_identities": {
            material_by_batch_name[str(record["batch_name"])]: record["material_config_digest"] for record in prepared.batch_records
        },
        "operation_config_digests": operation_digests,
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "material_roles": {name: list(values) for name, values in campaign.material_roles.items()},
        "evaluation_regimes": list(campaign.evaluation_regimes),
        "material_memberships": {name: list(values) for name, values in campaign.material_memberships.items()},
        "source_role": plan["source_role"],
        "training_eligible": bool(plan["split_eligibility"]["train"]),
        "duplicate_case_input_policy": campaign.duplicate_case_input_policy,
        "case_membership": case_membership,
        "split_membership": copy.deepcopy(prepared.membership),
        "membership_counts": {name: len(case_ids) for name, case_ids in prepared.membership.items()},
        "available_ood_groups": sorted(group_indexes),
        "ood_group_indexes": group_indexes,
        "ood_parameter_indexes": parameter_indexes,
        "task_relevant_ood_parameters": relevant_parameters,
        "material_counts": dict(sorted(material_counts.items())),
        "source_profile_counts": dict(sorted(profile_counts.items())),
        "candidate_source_case_count": len(prepared.candidates) + len(prepared.excluded),
        "builder_identity": "src.datasets.dataset_packages.build_campaign_packages",
        "schema_identity": {
            "package": DATASET_PACKAGE_SCHEMA_VERSION,
            "case_hdf5": storage_schema_version(),
            "transient_index": transient.TRANSIENT_INDEX_SCHEMA_VERSION,
        },
    }


def storage_schema_version() -> int:
    """Return the canonical source-case HDF5 schema identity."""
    return generation.storage.HDF5_SCHEMA_VERSION


def _aggregate_generated_identity(
    campaign: generation.config.CampaignConfig,
    prepared: planning.PreparedPackage,
) -> dict[str, Any]:
    """Build the steady payload's compatibility identity from exact source provenance."""
    profiles_found = {str(record["simulation_profile"]) for record in prepared.batch_records}
    conditioning_profiles = set(prepared.steady_conditioning.get("source_profiles", [])) if isinstance(prepared.steady_conditioning, dict) else set()
    if not profiles_found or conditioning_profiles != profiles_found:
        message = "Steady aggregate sources are not bound to one complete compatible conditioning audit."
        raise ValueError(message)
    representatives = [record for record in prepared.batch_records if record["simulation_profile"] == campaign.profile.id]
    representative_pool = representatives or prepared.batch_records
    first = min(
        representative_pool,
        key=lambda record: (str(record["simulation_profile"]), str(record["batch_id"])),
    )
    cases = [
        {
            "case_id": candidate["package_case_id"],
            "material_family": candidate["material_family"],
            "case_input_id": candidate["record"]["case_input_id"],
            "simulation_case_id": candidate["record"]["simulation_case_id"],
            "case_hdf5_sha256": candidate["record"]["case_hdf5_sha256"],
            "success_sha256": candidate["record"]["success_sha256"],
            "provenance_sha256": candidate["record"]["provenance_sha256"],
        }
        for candidate in prepared.candidates
    ]
    return identity.build_generated_package_identity(
        dataset_name=str(prepared.plan["dataset_name"]),
        simulation_profile=str(first["simulation_profile"]),
        campaign_digest=campaign.campaign_digest,
        template=first["template"],
        export_contract_sha256=str(first["export_contract_sha256"]),
        available_learning_views=first["available_learning_views"],
        airflow_source=str(first["airflow_source"]),
        cases=cases,
    )


def _build_steady_payload(
    campaign: generation.config.CampaignConfig,
    prepared: planning.PreparedPackage,
    *,
    dataset_id: str,
    provenance: Mapping[str, Any],
    destination: Path,
) -> None:
    """Build one eager steady tensor payload from admitted canonical HDF5 views."""
    task = domain.tasks.registry.get_task("steady_flow")
    source_identities: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for candidate in prepared.candidates:
        _shape, case_inputs, case_outputs, metadata, source, _fingerprint = generated.interpret_generated_case(
            candidate["terminal_evidence"],
            candidate["case_evidence"],
            task=task,
        )
        metadata.update(
            {
                "dataset_id": dataset_id,
                "evaluation_regime": prepared.plan["evaluation_regime"],
                "dataset_membership": candidate["dataset_membership"],
                "source_batch_id": candidate["batch_id"],
                "source_case_id": candidate["case_id"],
                "source_simulation_profile": candidate["simulation_profile"],
                "source_hdf5_sha256": candidate["case_hdf5_sha256"],
                "task_relevant_ood_parameters": candidate.get("task_relevant_ood_parameters", []),
            }
        )
        source.update(
            {
                "source_batch_id": candidate["batch_id"],
                "source_case_id": candidate["case_id"],
                "case_id": candidate["package_case_id"],
            }
        )
        fingerprint = identity.compute_case_fingerprint(
            task=task,
            case_id=candidate["package_case_id"],
            source_identity=source,
            source_metadata=metadata,
            inputs=case_inputs,
            outputs=case_outputs,
        )
        source_identities.append(source)
        source_metadata.append(metadata)
        fingerprints.append(fingerprint)
        inputs.append(case_inputs)
        outputs.append(case_outputs)
    payload = identity.build_training_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=[candidate["package_case_id"] for candidate in prepared.candidates],
        generated_batch_identity=_aggregate_generated_identity(campaign, prepared),
        source_identities=source_identities,
        source_metadata=source_metadata,
        source_provenance=provenance,
        case_fingerprints=fingerprints,
        inputs=torch.stack(inputs),
        outputs=torch.stack(outputs),
    )
    common.serialization.atomic_torch_save(payload, destination)


def _transient_sources(prepared: planning.PreparedPackage) -> tuple[transient.TransientSourceCase, ...]:
    """Convert admitted candidates into typed transient index sources."""
    sources: list[transient.TransientSourceCase] = []
    for candidate in prepared.candidates:
        case = candidate["case_evidence"]
        artifact = case.artifact("processed", "case.h5")
        if artifact.sha256 != case.case_hdf5_sha256:
            message = f"Admitted transient HDF5 evidence disagrees for {case.case_id!r}."
            raise RuntimeError(message)
        sources.append(
            transient.TransientSourceCase(
                path=artifact.path,
                package_case_id=str(candidate["package_case_id"]),
                source_batch_id=str(candidate["batch_id"]),
                membership=str(candidate["dataset_membership"]),
                evaluation_regime=str(prepared.plan["evaluation_regime"]),
                expected_sha256=artifact.sha256,
                expected_case_input_id=case.case_input_id,
                expected_simulation_case_id=case.simulation_case_id,
                material_family=case.material_family,
                ood_group=(str(candidate["ood_evidence"]["group"]) if candidate.get("ood_evidence", {}).get("group") is not None else None),
                ood_parameters=tuple(str(name) for name in candidate.get("task_relevant_ood_parameters", [])),
                ood_evidence=copy.deepcopy(candidate.get("ood_evidence", {})),
            )
        )
    return tuple(sources)


def _publish(
    *,
    dataset_id: str,
    payload_filename: str,
    manifest: Mapping[str, Any],
    build_payload: Callable[[Path], None],
    storage_root: Path | str | None,
) -> tuple[Path, Path, bool]:
    """Atomically publish or integrity-reuse one locked immutable package."""
    payload_root = common.paths.get_dataset_payload_root(storage_root=storage_root)
    metadata_root = common.paths.get_dataset_metadata_root(storage_root=storage_root)
    destination_dir = payload_root / dataset_id
    metadata_dir = metadata_root / dataset_id
    destination = destination_dir / payload_filename
    manifest_path = metadata_dir / "dataset_manifest.json"
    lock_path = common.paths.resolve_dataset_build_lock_path(dataset_id, storage_root=storage_root)
    requested_manifest = dict(manifest)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if destination.is_file() and manifest_path.is_file():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                message = f"Existing package manifest is unreadable: {manifest_path}."
                raise ValueError(message) from error
            if not isinstance(existing_manifest, dict):
                message = f"Existing package manifest is malformed: {manifest_path}."
                raise TypeError(message)
            payload_sha256 = existing_manifest.pop("payload_sha256", None)
            if existing_manifest != requested_manifest or payload_sha256 != common.serialization.file_sha256(destination):
                message = f"Existing package content conflicts with {dataset_id!r}."
                raise FileExistsError(message)
            return destination, manifest_path, True
        if destination_dir.exists() or metadata_dir.exists():
            message = f"Partial or conflicting dataset package already exists: {dataset_id!r}."
            raise FileExistsError(message)
        state_root = common.paths.get_dataset_state_root(storage_root=storage_root)
        state_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", suffix=".tmp", dir=state_root))
        staged_payload_dir = staging / "payload"
        staged_metadata_dir = staging / "metadata"
        staged_payload_dir.mkdir()
        staged_metadata_dir.mkdir()
        try:
            staged_payload = staged_payload_dir / payload_filename
            build_payload(staged_payload)
            complete_manifest = {
                **requested_manifest,
                "payload_sha256": common.serialization.file_sha256(staged_payload),
            }
            common.serialization.atomic_write_json(
                staged_metadata_dir / "dataset_manifest.json",
                complete_manifest,
            )
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            metadata_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_payload_dir.replace(destination_dir)
            staged_metadata_dir.replace(metadata_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return destination, manifest_path, False


def _publish_prepared(
    campaign: generation.config.CampaignConfig,
    prepared: planning.PreparedPackage,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Build or exactly reuse one fully prepared package payload."""
    provenance = _package_provenance(campaign, prepared)
    dataset_id, dataset_digest = identity.package_identity_from_provenance(provenance)
    dataset_view = str(prepared.plan["dataset_view"])
    if dataset_view == "steady_flow":
        payload_filename = f"{dataset_id}.pt"
        sample_count = len(prepared.candidates)
        transition_count = 0

        def build_payload(path: Path) -> None:
            _build_steady_payload(
                campaign,
                prepared,
                dataset_id=dataset_id,
                provenance=provenance,
                destination=path,
            )

    elif dataset_view == "transient_drying":
        payload_filename = f"{dataset_id}.json"
        source_root = common.paths.get_storage_root(storage_root=storage_root)
        source_cases = _transient_sources(prepared)
        preview = transient.build_transient_index(
            source_cases,
            None,
            dataset_name=str(prepared.plan["dataset_name"]),
            dataset_id=dataset_id,
            evaluation_regime=str(prepared.plan["evaluation_regime"]),
            source_root=source_root,
        )
        sample_count = int(preview["sample_count"])
        transition_count = int(preview["transition_count"])

        def build_payload(path: Path) -> None:
            built = transient.build_transient_index(
                source_cases,
                path,
                dataset_name=str(prepared.plan["dataset_name"]),
                dataset_id=dataset_id,
                evaluation_regime=str(prepared.plan["evaluation_regime"]),
                source_root=source_root,
            )
            if built["index_digest"] != preview["index_digest"]:
                message = "Transient transition identity changed between preview and publication."
                raise RuntimeError(message)

    else:
        message = f"Unsupported dataset view: {dataset_view!r}."
        raise ValueError(message)
    manifest = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": payload_filename,
        "sample_count": sample_count,
        "source_case_count": len(prepared.candidates),
        "transition_count": transition_count,
    }
    destination, manifest_path, reused = _publish(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest=manifest,
        build_payload=build_payload,
        storage_root=storage_root,
    )
    return {
        "dataset_name": prepared.plan["dataset_name"],
        "dataset_id": dataset_id,
        "dataset_view": dataset_view,
        "evaluation_regime": prepared.plan["evaluation_regime"],
        "payload_path": destination,
        "manifest_path": manifest_path,
        "sample_count": sample_count,
        "source_case_count": len(prepared.candidates),
        "transition_count": transition_count,
        "status": "reused" if reused else "complete",
    }


def build_dataset_package(
    campaign: generation.config.CampaignConfig,
    dataset_view: str,
    evaluation_regime: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build one declared package with its required ID leakage companion."""
    requested_plan = planning.package_plan(campaign, dataset_view, evaluation_regime)
    selected_plans = [requested_plan]
    if evaluation_regime != "id":
        selected_plans.insert(0, planning.package_plan(campaign, dataset_view, "id"))
    prepared = planning.prepare_campaign_packages(
        campaign,
        storage_root=storage_root,
        selected_plans=selected_plans,
    )
    matches = [
        package for package in prepared if package.plan["dataset_view"] == dataset_view and package.plan["evaluation_regime"] == evaluation_regime
    ]
    if len(matches) != 1:
        message = f"Prepared package selection is ambiguous for {dataset_view!r}/{evaluation_regime!r}."
        raise RuntimeError(message)
    return _publish_prepared(campaign, matches[0], storage_root=storage_root)


def build_campaign_packages(
    campaign: generation.config.CampaignConfig,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build every declared package after one shared membership/leakage preflight."""
    prepared = planning.prepare_campaign_packages(campaign, storage_root=storage_root)
    return tuple(_publish_prepared(campaign, package, storage_root=storage_root) for package in prepared)


def load_package_manifest(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate one package manifest and its exact payload hash."""
    return package_manifest.load_package_manifest(
        dataset_id,
        storage_root=storage_root,
    )


def load_dataset_package_manifest(
    dataset_id: str,
    *,
    dataset_identity: identity.DatasetIdentity,
    dataset_path: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Bind a steady package manifest to its validated tensor payload identity."""
    return package_manifest.load_steady_package_manifest(
        dataset_id,
        dataset_identity=dataset_identity,
        dataset_path=dataset_path,
        metadata_root=metadata_root,
    )


def _runtime_request(
    manifest: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    membership: str | None = None,
    ood_group: str | None = None,
    allow_technical_smoke: bool = False,
) -> Any:
    """Build one typed factory request from a validated package manifest."""
    return factory.DatasetRequest(
        dataset_id=str(manifest["dataset_id"]),
        dataset_view=cast("views.DatasetViewId", manifest["dataset_view"]),
        evaluation_regime=cast("views.PackageRegime", manifest["evaluation_regime"]),
        membership=cast("views.IdMembership | None", membership),
        ood_group=ood_group,
        storage_root=storage_root,
        allow_technical_smoke=allow_technical_smoke,
    )


def _tensor_description(tensor: torch.Tensor, channels: Any) -> dict[str, Any]:
    """Return one inspection tensor shape, dtype, and channel declaration."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "channels": copy.deepcopy(channels),
    }


def inspect_dataset_package(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect one package through its validated manifest and runtime object."""
    manifest = load_package_manifest(dataset_id, storage_root=storage_root)
    request = _runtime_request(
        manifest,
        storage_root=storage_root,
        allow_technical_smoke=manifest["campaign_purpose"] == "technical_runtime_smoke",
    )
    runtime = factory.create_dataset(request, hdf5_cache_size=1)
    sample = cast("Mapping[str, Any]", runtime[0])
    if manifest["dataset_view"] == "steady_flow":
        tensor_report = {
            "input": _tensor_description(sample["x"], manifest["channel_contract"]["input"]),
            "target": _tensor_description(sample["y"], manifest["channel_contract"]["target"]),
        }
        metadata = cast("Mapping[str, Any]", sample["meta"])
        sample_identity = {
            key: metadata.get(key)
            for key in (
                "dataset_id",
                "simulation_case_id",
                "case_input_id",
                "source_batch_id",
                "source_simulation_profile",
                "material_family",
                "evaluation_regime",
                "dataset_membership",
                "source_hdf5_sha256",
            )
        }
    else:
        tensor_report = {
            name: _tensor_description(sample[name], manifest["channel_contract"][name])
            for name in ("state", "static", "boundary", "scalars", "target")
        }
        tensor_report["dt"] = _tensor_description(sample["dt"], manifest["channel_contract"]["dt"])
        sample_identity = dict(sample["metadata"])
        if isinstance(runtime, transient.TransientPhysicalDataset):
            runtime.close()
    regime = str(manifest["evaluation_regime"])
    if regime == "id" and manifest["campaign_purpose"] == "technical_runtime_smoke":
        selectors = [views.TECHNICAL_SMOKE_MEMBERSHIP]
    elif regime == "id":
        selectors = [f"id/{membership}" for membership in views.ID_MEMBERSHIPS]
    elif regime == "parameter_ood":
        selectors = ["parameter_ood/all", *[f"parameter_ood/{group}" for group in manifest["available_ood_groups"]]]
    else:
        selectors = [regime]
    return {
        "dataset_name": manifest["dataset_name"],
        "dataset_id": manifest["dataset_id"],
        "dataset_view": manifest["dataset_view"],
        "registered_task_id": manifest["registered_task_id"],
        "evaluation_regime": regime,
        "available_selectors": selectors,
        "available_ood_groups": manifest["available_ood_groups"],
        "membership_counts": manifest["membership_counts"],
        "source_case_count": manifest["source_case_count"],
        "transition_count": manifest["transition_count"],
        "sample_count": manifest["sample_count"],
        "material_counts": manifest["material_counts"],
        "source_profile_counts": manifest["source_profile_counts"],
        "tensors": tensor_report,
        "sample_identity": sample_identity,
    }


def smoke_dataset_package(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
    membership: str | None = None,
    ood_group: str | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    hdf5_cache_size: int = 1,
) -> dict[str, Any]:
    """Load one batch through the unified factory with requested worker settings."""
    manifest = load_package_manifest(dataset_id, storage_root=storage_root)
    request = _runtime_request(
        manifest,
        storage_root=storage_root,
        membership=membership,
        ood_group=ood_group,
        allow_technical_smoke=manifest["campaign_purpose"] == "technical_runtime_smoke",
    )
    settings = factory.LoaderSettings(
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        hdf5_cache_size=hdf5_cache_size,
    )
    loader = factory.create_data_loader(request, settings)
    batch = next(iter(loader))
    tensor_keys = ("x", "y") if manifest["dataset_view"] == "steady_flow" else ("state", "static", "boundary", "scalars", "target", "dt")
    shapes = {key: list(value.shape) for key in tensor_keys if isinstance((value := batch.get(key)), torch.Tensor)}
    return {
        "dataset_id": dataset_id,
        "dataset_view": manifest["dataset_view"],
        "evaluation_regime": manifest["evaluation_regime"],
        "membership": membership,
        "ood_group": ood_group,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "batch_shapes": shapes,
        "status": "loaded",
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the dataset package build, inspection, and smoke CLI."""
    parser = argparse.ArgumentParser(description="Build, inspect, or smoke immutable dataset packages")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build every package declared by one campaign")
    build.add_argument("campaign_config", type=Path)
    build.add_argument("--storage-root", type=Path)

    inspect = commands.add_parser("inspect", help="inspect one immutable package")
    inspect.add_argument("dataset_id")
    inspect.add_argument("--storage-root", type=Path)

    smoke = commands.add_parser("smoke", help="load one package batch through the unified factory")
    smoke.add_argument("dataset_id")
    smoke.add_argument("--storage-root", type=Path)
    smoke.add_argument("--membership", choices=views.ID_MEMBERSHIPS)
    smoke.add_argument("--ood-group", help="parameter-OOD group declared by the selected package")
    smoke.add_argument("--num-workers", type=int, default=0)
    smoke.add_argument("--persistent-workers", action="store_true")
    smoke.add_argument("--prefetch-factor", type=int)
    smoke.add_argument("--hdf5-cache-size", type=int, default=1)
    return parser


def main() -> int:
    """Execute one explicit package build, inspection, or smoke command."""
    arguments = _build_parser().parse_args()
    if arguments.command == "build":
        campaign = generation.config.load_campaign_config(arguments.campaign_config)
        result: dict[str, Any] = {
            "campaign_id": campaign.campaign_id,
            "packages": build_campaign_packages(campaign, storage_root=arguments.storage_root),
        }
    elif arguments.command == "inspect":
        result = inspect_dataset_package(
            arguments.dataset_id,
            storage_root=arguments.storage_root,
        )
    elif arguments.command == "smoke":
        result = smoke_dataset_package(
            arguments.dataset_id,
            storage_root=arguments.storage_root,
            membership=arguments.membership,
            ood_group=arguments.ood_group,
            num_workers=arguments.num_workers,
            persistent_workers=arguments.persistent_workers,
            prefetch_factor=arguments.prefetch_factor,
            hdf5_cache_size=arguments.hdf5_cache_size,
        )
    else:
        message = f"Unsupported dataset package command: {arguments.command!r}."
        raise RuntimeError(message)
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
