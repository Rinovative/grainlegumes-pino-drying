"""
dataset_packages_builder.py

Build immutable dual-view Dataset packages from terminal simulation evidence.
Responsibilities:
  - Assemble content-bound steady payloads and transient indexes
  - Compute package identities and publish exact payloads atomically
  - Reuse only existing packages whose manifests and payload hashes agree
Design principles:
  - Package identity is independent of operational source locations
  - One combined parameter-OOD package owns compact group and parameter indexes
  - Publication fails closed on partial or conflicting existing state
This module does NOT:
  - Provide the supported Dataset package CLI
  - Inspect or smoke-load published runtime Dataset objects
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from src import common, domain, generation
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views

from . import dataset_packages_generated_batch as generated
from . import dataset_packages_manifest as package_manifest
from . import dataset_packages_planning as planning
from . import dataset_packages_trajectory as trajectory

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


def _schema_identity(dataset_view: str) -> dict[str, int | str]:
    """Return view-specific payload schema identity in the package-v1 envelope."""
    if dataset_view == "transient_drying":
        payload_schema_version = trajectory.TRANSIENT_INDEX_SCHEMA_VERSION
    elif dataset_view == "steady_flow":
        payload_schema_version = identity.TRAINING_DATASET_SCHEMA_VERSION
    else:
        message = f"Unsupported Dataset view {dataset_view!r}."
        raise ValueError(message)
    return {
        "package": DATASET_PACKAGE_SCHEMA_VERSION,
        "case_hdf5": storage_schema_version(),
        "generation_case": generation.cases.case.CASE_CONTRACT_DIGEST,
        "transient_index": payload_schema_version,
    }


def _package_provenance(
    campaign: generation.cases.config.CampaignConfig,
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
    git_commits: set[str] = set()
    for record in prepared.batch_records:
        record_commits = record.get("git_commits")
        if isinstance(record_commits, list):
            git_commits.update(str(value) for value in record_commits)
        else:
            git_commits.add(str(record["git_commit"]))
    git_commits.update(
        str(candidate["source_git_commit"]) for candidate in prepared.candidates if isinstance(candidate.get("source_git_commit"), str)
    )
    sorted_git_commits = sorted(git_commits)
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
        "source_git_commits": sorted_git_commits,
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
                **(
                    {
                        "composite_source_kind": candidate["composite_source_kind"],
                        "source_run_id": candidate["source_run_id"],
                        "source_git_commit": candidate["source_git_commit"],
                        "source_campaign_manifest_sha256": candidate["source_campaign_manifest_sha256"],
                        "completion_receipt_sha256": candidate["completion_receipt_sha256"],
                    }
                    if candidate.get("completion_receipt_sha256") is not None
                    else {}
                ),
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
        "builder_identity": identity.dataset_conversion_contract_identity(view.id),
        "schema_identity": _schema_identity(view.id),
    }


def storage_schema_version() -> int:
    """Return the canonical source-case HDF5 schema identity."""
    return generation.publication.storage.HDF5_SCHEMA_VERSION


def _aggregate_generated_identity(
    campaign: generation.cases.config.CampaignConfig,
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
    campaign: generation.cases.config.CampaignConfig,
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


def _transient_sources(prepared: planning.PreparedPackage) -> tuple[trajectory.TransientSourceCase, ...]:
    """Convert admitted candidates into typed transient index sources."""
    sources: list[trajectory.TransientSourceCase] = []
    for candidate in prepared.candidates:
        case = candidate["case_evidence"]
        artifact = case.artifact_evidence("processed", "case.h5")
        if artifact.sha256 != case.case_hdf5_sha256:
            message = f"Admitted transient HDF5 evidence disagrees for {case.case_id!r}."
            raise RuntimeError(message)
        sources.append(
            trajectory.TransientSourceCase(
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
    manifest_prefix: Mapping[str, Any],
    build_payload: Callable[[Path], Mapping[str, int]],
    storage_root: Path | str | None,
) -> tuple[Path, Path, bool, dict[str, Any]]:
    """Atomically publish or integrity-reuse one locked immutable package."""
    payload_root = common.paths.get_dataset_packages_root(storage_root=storage_root)
    metadata_root = common.paths.get_dataset_metadata_root(storage_root=storage_root)
    destination_dir = payload_root / dataset_id
    metadata_dir = metadata_root / dataset_id
    destination = destination_dir / payload_filename
    manifest_path = metadata_dir / "dataset_manifest.json"
    lock_path = common.paths.resolve_dataset_build_lock_path(dataset_id, storage_root=storage_root)
    requested_prefix = dict(manifest_prefix)
    published_manifest: dict[str, Any]
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if destination.is_file() and manifest_path.is_file():
            try:
                raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                message = f"Existing package manifest is unreadable: {manifest_path}."
                raise ValueError(message) from error
            if not isinstance(raw_manifest, dict):
                message = f"Existing package manifest is malformed: {manifest_path}."
                raise TypeError(message)
            if any(raw_manifest.get(key) != value for key, value in requested_prefix.items()):
                message = f"Existing package content conflicts with {dataset_id!r}."
                raise FileExistsError(message)
            try:
                existing_manifest = package_manifest.validate_manifest_content(
                    raw_manifest,
                    dataset_id=dataset_id,
                    payload_path=destination,
                )
            except (FileNotFoundError, TypeError, ValueError) as error:
                message = f"Existing package content conflicts with {dataset_id!r}."
                raise FileExistsError(message) from error
            return destination, manifest_path, True, existing_manifest
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
            payload_metadata = dict(build_payload(staged_payload))
            if set(payload_metadata) != {"sample_count", "transition_count"}:
                message = "Dataset payload builder returned invalid manifest metadata."
                raise RuntimeError(message)
            published_manifest = {
                **requested_prefix,
                **payload_metadata,
                "payload_sha256": common.serialization.file_sha256(staged_payload),
            }
            package_manifest.validate_manifest_content(
                published_manifest,
                dataset_id=dataset_id,
                payload_path=staged_payload,
                validate_payload_hash=False,
            )
            common.serialization.atomic_write_json(
                staged_metadata_dir / "dataset_manifest.json",
                published_manifest,
            )
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            metadata_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_payload_dir.replace(destination_dir)
            staged_metadata_dir.replace(metadata_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    admitted_manifest = package_manifest.load_package_manifest_evidence(
        dataset_id,
        storage_root=storage_root,
    )
    if admitted_manifest != published_manifest:
        message = f"Published package manifest changed during admission: {dataset_id!r}."
        raise RuntimeError(message)
    return destination, manifest_path, False, admitted_manifest


def _publish_prepared(
    campaign: generation.cases.config.CampaignConfig,
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

        def build_payload(path: Path) -> Mapping[str, int]:
            _build_steady_payload(
                campaign,
                prepared,
                dataset_id=dataset_id,
                provenance=provenance,
                destination=path,
            )
            return {
                "sample_count": len(prepared.candidates),
                "transition_count": 0,
            }

    elif dataset_view == "transient_drying":
        payload_filename = f"{dataset_id}.json"

        def build_payload(path: Path) -> Mapping[str, int]:
            source_root = common.paths.get_storage_root(storage_root=storage_root)
            preview = trajectory.build_transient_index(
                _transient_sources(prepared),
                None,
                dataset_name=str(prepared.plan["dataset_name"]),
                dataset_id=dataset_id,
                evaluation_regime=str(prepared.plan["evaluation_regime"]),
                source_root=source_root,
            )
            serialized = json.dumps(
                preview,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            common.serialization.atomic_write_text(path, f"{serialized}\n")
            return {
                "sample_count": int(preview["sample_count"]),
                "transition_count": int(preview["transition_count"]),
            }

    else:
        message = f"Unsupported dataset view: {dataset_view!r}."
        raise ValueError(message)
    manifest_prefix = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": payload_filename,
        "source_case_count": len(prepared.candidates),
    }
    if dataset_view == "steady_flow":
        manifest_prefix.update(
            {
                "sample_count": len(prepared.candidates),
                "transition_count": 0,
            }
        )
    destination, manifest_path, reused, manifest = _publish(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest_prefix=manifest_prefix,
        build_payload=build_payload,
        storage_root=storage_root,
    )
    if reused and dataset_view == "transient_drying":
        existing_index = trajectory.load_transient_index(destination)
        if (
            existing_index["dataset_id"] != dataset_id
            or existing_index["sample_count"] != manifest["sample_count"]
            or existing_index["source_case_count"] != manifest["source_case_count"]
            or existing_index["transition_count"] != manifest["transition_count"]
        ):
            message = f"Existing package content conflicts with {dataset_id!r}."
            raise FileExistsError(message)
    return {
        "dataset_name": prepared.plan["dataset_name"],
        "dataset_id": dataset_id,
        "dataset_view": dataset_view,
        "evaluation_regime": prepared.plan["evaluation_regime"],
        "payload_path": destination,
        "manifest_path": manifest_path,
        "sample_count": int(manifest["sample_count"]),
        "source_case_count": len(prepared.candidates),
        "transition_count": int(manifest["transition_count"]),
        "status": "reused" if reused else "complete",
    }


def build_dataset_package(
    campaign: generation.cases.config.CampaignConfig,
    dataset_view: str,
    evaluation_regime: str,
    *,
    storage_root: Path | str | None = None,
    composite_receipt: Mapping[str, Any] | None = None,
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
        composite_receipt=composite_receipt,
    )
    matches = [
        package for package in prepared if package.plan["dataset_view"] == dataset_view and package.plan["evaluation_regime"] == evaluation_regime
    ]
    if len(matches) != 1:
        message = f"Prepared package selection is ambiguous for {dataset_view!r}/{evaluation_regime!r}."
        raise RuntimeError(message)
    return _publish_prepared(campaign, matches[0], storage_root=storage_root)


def build_campaign_packages(
    campaign: generation.cases.config.CampaignConfig,
    *,
    storage_root: Path | str | None = None,
    composite_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build every declared package after one shared membership/leakage preflight."""
    prepared = planning.prepare_campaign_packages(campaign, storage_root=storage_root, composite_receipt=composite_receipt)
    return tuple(_publish_prepared(campaign, package, storage_root=storage_root) for package in prepared)
