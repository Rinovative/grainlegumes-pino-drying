"""
===============================================================================
dataset_packages.py
===============================================================================
Assemble campaign-owned ID and OOD dataset packages from terminal batches.
Responsibilities:
  - Resolve independent evaluation packages and immutable source membership
  - Assign deterministic family-stratified train, validation, and ID-test roles
  - Publish digest-bound manifests and steady or transient physical payloads
Design principles:
  - Human-readable names exclude versions and source profiles
  - Dataset identities bind exact cases, memberships, and scientific provenance
  - Existing valid packages are reused; conflicting identities fail closed
This module does NOT:
  - Choose generation counts, normalize data, register transient tasks, or train
  - Merge independent OOD roles into one mandatory package
===============================================================================
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from src import common, domain
from src.generation import generation_config as config_service

from . import dataset_generated_batch as generated
from . import dataset_identity as identity
from . import dataset_transient as transient

DATASET_PACKAGE_SCHEMA_KIND = "vp2_dataset_package_manifest"
DATASET_PACKAGE_SCHEMA_VERSION = 1
_PACKAGE_PROVENANCE_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_name",
        "learning_task",
        "evaluation_regime",
        "materials",
        "source_simulation_profiles",
        "source_batches",
        "source_batch_ids",
        "source_template_digests",
        "source_git_commits",
        "source_case_identities",
        "airflow_provenance",
        "material_file_identities",
        "operation_config_digest",
        "campaign_name",
        "campaign_digest",
        "split_membership",
        "builder_identity",
        "schema_identity",
    }
)
_PACKAGE_MANIFEST_KEYS = _PACKAGE_PROVENANCE_KEYS | {
    "dataset_id",
    "dataset_digest",
    "payload_filename",
    "sample_count",
    "payload_sha256",
}
if TYPE_CHECKING:
    from collections.abc import Mapping

_ID_MEMBERSHIP_ROLES: Final = ("train", "validation", "id_test")


def _package_plan(campaign: config_service.CampaignConfig, evaluation_regime: str) -> dict[str, Any]:
    """Return one exact predeclared dataset package plan."""
    matches = [copy.deepcopy(package) for package in campaign.dataset_packages if package["evaluation_regime"] == evaluation_regime]
    if len(matches) != 1:
        message = f"Campaign {campaign.campaign_name!r} must declare exactly one {evaluation_regime!r} package."
        raise ValueError(message)
    return matches[0]


def _source_batches(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
) -> tuple[config_service.GenerationConfig, ...]:
    """Resolve exact source batches for one evaluation regime."""
    regime = str(plan["evaluation_regime"])
    sampling_regime = "parameter_ood" if regime == "parameter_ood" else "natural"
    expected_materials = tuple(plan["materials"])
    return tuple(campaign.batch(f"{campaign.profile.id}__{material_family}__{sampling_regime}") for material_family in expected_materials)


def _global_case_id(batch_name: str, case_id: str) -> str:
    """Return one package-unique readable case identifier."""
    return f"{batch_name}__{case_id}"


def _load_candidates(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load terminal manifests and enumerate exact source case candidates."""
    batch_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for batch in _source_batches(campaign, plan):
        manifest, _manifest_path, manifest_sha256 = generated.load_batch_manifest(
            batch.batch_id,
            storage_root=storage_root,
        )
        validated_manifest = generated.validate_terminal_batch(
            batch.batch_id,
            storage_root=storage_root,
        )
        if validated_manifest != manifest:
            message = f"Terminal batch changed during package admission: {batch.batch_name}."
            raise RuntimeError(message)
        if manifest["batch_identity"] != batch.batch_identity:
            message = f"Terminal batch identity disagrees with campaign plan: {batch.batch_name}."
            raise ValueError(message)
        batch_records.append(
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "batch_identity": batch.batch_identity,
                "manifest_sha256": manifest_sha256,
                "simulation_profile": manifest["simulation_profile"],
                "template": manifest["template"],
                "scientific_config_digest": manifest["scientific_config_digest"],
                "git_commit": manifest["git_commit"],
                "material_config_digest": batch.scientific_values["material_config_digest"],
                "operation_config_digest": batch.scientific_values["operation_config_digest"],
                "airflow_source": manifest["airflow_source"],
                "available_learning_views": manifest["available_learning_views"],
                "export_contract_sha256": manifest["export_contract_sha256"],
            }
        )
        for record in manifest["cases"]:
            case_id = str(record["case_id"])
            candidates.append(
                {
                    "batch": batch,
                    "manifest": manifest,
                    "record": record,
                    "batch_id": batch.batch_id,
                    "batch_name": batch.batch_name,
                    "case_id": case_id,
                    "package_case_id": _global_case_id(batch.batch_name, case_id),
                    "material_family": record["material_family"],
                    "case_hdf5": (
                        common.paths.resolve_generated_batch_dir(
                            batch.batch_id,
                            stage="processed",
                            storage_root=storage_root,
                        )
                        / case_id
                        / "case.h5"
                    ),
                }
            )
    return batch_records, candidates


def _membership_rank(seed: int, material_family: str, simulation_case_id: str) -> str:
    """Return one stable family-local membership rank."""
    payload = f"{seed}|{material_family}|{simulation_case_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _assign_membership(
    plan: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Assign deterministic family-stratified ID membership or one OOD role."""
    regime = str(plan["evaluation_regime"])
    if regime != "id":
        ordered = list(candidates)
        membership = {regime: [str(candidate["package_case_id"]) for candidate in ordered]}
        for candidate in ordered:
            candidate["dataset_membership"] = regime
        return ordered, membership

    seed = plan["membership_seed"]
    counts = plan["membership_counts_per_material"]
    if not isinstance(seed, int) or any(not isinstance(counts.get(role), int) for role in _ID_MEMBERSHIP_ROLES):
        message = "ID package membership seed and per-material counts must be resolved."
        raise ValueError(message)
    selected: list[dict[str, Any]] = []
    membership = {role: [] for role in _ID_MEMBERSHIP_ROLES}
    for material_family in plan["materials"]:
        family = [candidate for candidate in candidates if candidate["material_family"] == material_family]
        family.sort(
            key=lambda candidate: _membership_rank(
                seed,
                material_family,
                str(candidate["record"]["simulation_case_id"]),
            )
        )
        required = sum(int(counts[role]) for role in _ID_MEMBERSHIP_ROLES)
        if len(family) < required:
            message = f"ID package requires {required} {material_family!r} cases but only {len(family)} are terminal."
            raise ValueError(message)
        offset = 0
        for role in _ID_MEMBERSHIP_ROLES:
            role_count = int(counts[role])
            for candidate in family[offset : offset + role_count]:
                candidate["dataset_membership"] = role
                membership[role].append(str(candidate["package_case_id"]))
                selected.append(candidate)
            offset += role_count
    if len({case_id for values in membership.values() for case_id in values}) != len(selected):
        message = "ID train, validation, and id_test membership must be disjoint."
        raise RuntimeError(message)
    return selected, membership


def _package_provenance(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
    batch_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    membership: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return complete source and membership provenance before identity binding."""
    profiles_found = list(dict.fromkeys(record["simulation_profile"] for record in batch_records))
    template_digests = list(dict.fromkeys(record["template"]["sha256"] for record in batch_records))
    airflow = list(dict.fromkeys(record["airflow_source"] for record in batch_records))
    operation_digests = list(dict.fromkeys(record["operation_config_digest"] for record in batch_records))
    git_commits = list(dict.fromkeys(record["git_commit"] for record in batch_records))
    if len(profiles_found) != 1 or len(template_digests) != 1 or len(airflow) != 1 or len(operation_digests) != 1 or len(git_commits) != 1:
        message = "Dataset package sources disagree on profile, template, airflow, operation, or Git identity."
        raise ValueError(message)
    return {
        "schema_kind": DATASET_PACKAGE_SCHEMA_KIND,
        "schema_version": DATASET_PACKAGE_SCHEMA_VERSION,
        "dataset_name": plan["dataset_name"],
        "learning_task": plan["learning_task"],
        "evaluation_regime": plan["evaluation_regime"],
        "materials": list(plan["materials"]),
        "source_simulation_profiles": profiles_found,
        "source_batches": batch_records,
        "source_batch_ids": [record["batch_id"] for record in batch_records],
        "source_template_digests": template_digests,
        "source_git_commits": git_commits,
        "source_case_identities": [
            {
                "package_case_id": candidate["package_case_id"],
                "batch_id": candidate["batch_id"],
                "case_id": candidate["case_id"],
                "case_input_id": candidate["record"]["case_input_id"],
                "simulation_case_id": candidate["record"]["simulation_case_id"],
                "case_hdf5_sha256": candidate["record"]["case_hdf5_sha256"],
                "material_family": candidate["material_family"],
                "membership": candidate["dataset_membership"],
            }
            for candidate in candidates
        ],
        "airflow_provenance": airflow[0],
        "material_file_identities": {record["batch_name"].split("__")[1]: record["material_config_digest"] for record in batch_records},
        "operation_config_digest": operation_digests[0],
        "campaign_name": campaign.campaign_name,
        "campaign_digest": campaign.campaign_digest,
        "split_membership": copy.deepcopy(dict(membership)),
        "builder_identity": "src.datasets.dataset_packages.build_dataset_package",
        "schema_identity": {
            "package": DATASET_PACKAGE_SCHEMA_VERSION,
            "case_hdf5": storage_schema_version(),
        },
    }


def storage_schema_version() -> int:
    """Return the canonical source-case HDF5 schema identity."""
    from src.generation import generation_storage  # noqa: PLC0415

    return generation_storage.HDF5_SCHEMA_VERSION


def _dataset_identity_from_provenance(provenance: Mapping[str, Any]) -> tuple[str, str]:
    """Return immutable dataset ID and full digest for a package manifest."""
    digest = common.serialization.canonical_json_sha256(provenance)
    name = str(provenance["dataset_name"])
    return f"{name}__{digest[:16]}", digest


def load_dataset_package_manifest(
    dataset_id: str,
    *,
    dataset_identity: identity.DatasetIdentity,
    dataset_path: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Load and bind one campaign package manifest to its exact payload identity."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    payload_path = Path(dataset_path)
    if not payload_path.is_file() or payload_path.is_symlink():
        message = f"Dataset package payload is missing or unsafe: {payload_path}."
        raise FileNotFoundError(message)
    manifest_path = Path(metadata_root) / logical_id / "dataset_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        message = f"Dataset package manifest is missing or unsafe: {manifest_path}."
        raise FileNotFoundError(message)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Dataset package manifest is unreadable: {manifest_path}."
        raise ValueError(message) from error
    if not isinstance(manifest, dict) or set(manifest) != _PACKAGE_MANIFEST_KEYS:
        message = f"Dataset package manifest keys do not match the current schema: {manifest_path}."
        raise ValueError(message)
    provenance = {key: manifest[key] for key in _PACKAGE_PROVENANCE_KEYS}
    expected_id, expected_digest = _dataset_identity_from_provenance(provenance)
    if (
        manifest["schema_kind"] != DATASET_PACKAGE_SCHEMA_KIND
        or manifest["schema_version"] != DATASET_PACKAGE_SCHEMA_VERSION
        or manifest["dataset_id"] != logical_id
        or manifest["dataset_id"] != expected_id
        or manifest["dataset_digest"] != expected_digest
        or manifest["payload_filename"] != payload_path.name
        or manifest["payload_sha256"] != common.serialization.file_sha256(payload_path)
        or manifest["learning_task"] != dataset_identity.task
        or manifest["sample_count"] != dataset_identity.sample_count
        or dataset_identity.dataset_id != logical_id
        or dataset_identity.source_provenance != provenance
    ):
        message = f"Dataset package manifest does not bind the exact payload identity: {manifest_path}."
        raise ValueError(message)
    return dict(manifest)


def _aggregate_generated_identity(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
    batch_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the exact aggregate source identity expected by steady payloads."""
    first = batch_records[0]
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
        for candidate in candidates
    ]
    return identity.build_generated_package_identity(
        dataset_name=str(plan["dataset_name"]),
        simulation_profile=str(first["simulation_profile"]),
        campaign_digest=campaign.campaign_digest,
        template=first["template"],
        export_contract_sha256=str(first["export_contract_sha256"]),
        available_learning_views=first["available_learning_views"],
        airflow_source=str(first["airflow_source"]),
        cases=cases,
    )


def _build_steady_payload(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
    batch_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    dataset_id: str,
    provenance: Mapping[str, Any],
    destination: Path,
    storage_root: Path | str | None,
) -> None:
    """Build one steady-flow package payload from exact multi-batch membership."""
    task = domain.tasks.registry.get_task("steady_flow")
    source_identities: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for candidate in candidates:
        _shape, case_inputs, case_outputs, metadata, source, _fingerprint = generated.interpret_generated_case(
            candidate["batch_id"],
            candidate["case_id"],
            task=task,
            manifest=candidate["manifest"],
            record=candidate["record"],
            storage_root=storage_root,
        )
        metadata["dataset_membership"] = candidate["dataset_membership"]
        source["source_batch_id"] = candidate["batch_id"]
        source["source_case_id"] = candidate["case_id"]
        source["case_id"] = candidate["package_case_id"]
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
    generated_identity = _aggregate_generated_identity(
        campaign,
        plan,
        batch_records,
        candidates,
    )
    payload = identity.build_training_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=[candidate["package_case_id"] for candidate in candidates],
        generated_batch_identity=generated_identity,
        source_identities=source_identities,
        source_metadata=source_metadata,
        source_provenance=provenance,
        case_fingerprints=fingerprints,
        inputs=torch.stack(inputs),
        outputs=torch.stack(outputs),
    )
    common.serialization.atomic_torch_save(payload, destination)


def _publish(
    *,
    dataset_id: str,
    payload_filename: str,
    manifest: Mapping[str, Any],
    build_payload: Any,
    storage_root: Path | str | None,
) -> tuple[Path, Path, bool]:
    """Atomically publish or integrity-reuse one locked dataset package."""
    payload_root = common.paths.get_dataset_payload_root(storage_root=storage_root)
    metadata_root = common.paths.get_dataset_metadata_root(storage_root=storage_root)
    destination_dir = payload_root / dataset_id
    metadata_dir = metadata_root / dataset_id
    destination = destination_dir / payload_filename
    manifest_path = metadata_dir / "dataset_manifest.json"
    lock_path = common.paths.resolve_dataset_build_lock_path(
        dataset_id,
        storage_root=storage_root,
    )
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
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{dataset_id}.",
                suffix=".tmp",
                dir=state_root,
            )
        )
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


def build_dataset_package(
    campaign: config_service.CampaignConfig,
    evaluation_regime: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build or reuse one campaign-declared ID or OOD package."""
    plan = _package_plan(campaign, evaluation_regime)
    batch_records, candidates = _load_candidates(
        campaign,
        plan,
        storage_root=storage_root,
    )
    selected, membership = _assign_membership(plan, candidates)
    provenance = _package_provenance(
        campaign,
        plan,
        batch_records,
        selected,
        membership,
    )
    learning_task = str(plan["learning_task"])
    if learning_task == "transient_drying":
        transient_sources = [(candidate["case_hdf5"], str(plan["evaluation_regime"])) for candidate in selected]
        preview = transient.build_transient_index(
            transient_sources,
            None,
            dataset_name=str(plan["dataset_name"]),
            source_provenance=provenance,
        )
        dataset_id = str(preview["dataset_id"])
        dataset_digest = str(preview["dataset_digest"])
        payload_filename = f"{dataset_id}.json"

        def build_payload(path: Path) -> None:
            built = transient.build_transient_index(
                transient_sources,
                path,
                dataset_name=str(plan["dataset_name"]),
                source_provenance=provenance,
            )
            if built["dataset_id"] != dataset_id:
                message = "Transient package identity changed between preview and publication."
                raise RuntimeError(message)

    elif learning_task == "steady_flow":
        dataset_id, dataset_digest = _dataset_identity_from_provenance(provenance)
        payload_filename = f"{dataset_id}.pt"

        def build_payload(path: Path) -> None:
            _build_steady_payload(
                campaign,
                plan,
                batch_records,
                selected,
                dataset_id=dataset_id,
                provenance=provenance,
                destination=path,
                storage_root=storage_root,
            )

    else:
        message = f"Unsupported package learning view: {learning_task!r}."
        raise ValueError(message)

    manifest = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": payload_filename,
        "sample_count": len(selected),
    }
    destination, manifest_path, reused = _publish(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest=manifest,
        build_payload=build_payload,
        storage_root=storage_root,
    )
    return {
        "dataset_name": plan["dataset_name"],
        "dataset_id": dataset_id,
        "evaluation_regime": evaluation_regime,
        "payload_path": destination,
        "manifest_path": manifest_path,
        "sample_count": len(selected),
        "status": "reused" if reused else "complete",
    }


def build_campaign_packages(
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build every independent package declared by one campaign."""
    return tuple(
        build_dataset_package(
            campaign,
            str(plan["evaluation_regime"]),
            storage_root=storage_root,
        )
        for plan in campaign.dataset_packages
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the campaign package-construction command parser."""
    parser = argparse.ArgumentParser(
        description="Build every independent dataset package declared by one campaign",
    )
    parser.add_argument("campaign_config", type=Path)
    parser.add_argument("--storage-root", type=Path)
    return parser


def main() -> int:
    """Build campaign packages and print names, immutable IDs, and reuse state."""
    arguments = _build_parser().parse_args()
    campaign = config_service.load_campaign_config(arguments.campaign_config)
    packages = build_campaign_packages(campaign, storage_root=arguments.storage_root)
    print(
        json.dumps(
            {"campaign_id": campaign.campaign_id, "packages": packages},
            default=str,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
