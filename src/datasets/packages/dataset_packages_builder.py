"""Publish immutable Dataset selections from exact admitted source cases.

Responsibilities:
  - Build steady tensor payloads or transient trajectory indexes and PT shards from prepared plans
  - Bind content-derived Dataset identities to complete source provenance
  - Publish or reuse exact immutable payload and manifest pairs under writer locks
  - Keep identical content from being published under a second name

Design principles:
  - Source admission and compatibility remain owned by package planning
  - Immutable publication rejects conflicting content instead of overwriting it
  - Publication completes only after payload and manifest evidence agree
  - One staged folder holds manifest and complete payload; one rename publishes it, a failure publishes nothing
  - The shared identity registry binds a content digest to its one Dataset directory

This module does NOT:
  - Assign Train, validation, test or OOD membership
  - Fit preprocessing or generate new physical Source cases
  - Register reconstruction output, which lies outside canonical storage
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from src import common, domain
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views

from . import dataset_packages_generated_batch as generated
from . import dataset_packages_manifest as package_manifest
from . import dataset_packages_planning as planning
from . import dataset_packages_reproduction as reproduction
from . import dataset_packages_selection as selection
from . import dataset_packages_trajectory as trajectory
from . import dataset_packages_transient_shards as transient_shards

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_REGISTRY_KIND: Final = "dataset"


def _channel_contract(dataset_view: str) -> dict[str, Any]:
    if dataset_view == "transient_drying":
        return transient_contract.transient_contract_payload()
    task = domain.tasks.registry.get_task("steady_flow")
    return {
        "input": [field.as_dict() for field in task.inputs],
        "target": [field.as_dict() for field in task.outputs],
        "tensor_layout": list(task.tensor_layout),
    }


def _provenance(
    prepared: planning.PreparedPackage,
    *,
    storage_root: Path | str | None = None,
    producer_evidence: Mapping[str, Any] | None = None,
    case_partition: Mapping[str, Any] | None = None,
    material_roles: Mapping[str, str] | None = None,
    plan_invocation: list[str] | None = None,
) -> dict[str, Any]:
    request = prepared.request
    view = views.get_view(request["dataset_view"])
    materials = list(dict.fromkeys(case["source"]["material_family"] for case in prepared.source_case_identities))
    from src.datasets.contracts import dataset_contracts_partition as partition_contract  # noqa: PLC0415

    case_ids = [entry["package_case_id"] for entry in prepared.source_case_identities]
    material_by_case = {entry["package_case_id"]: entry["source"]["material_family"] for entry in prepared.source_case_identities}
    kind_by_case = {entry["package_case_id"]: entry["design"]["design_kind"] for entry in prepared.source_case_identities}
    if material_roles is not None:
        if case_partition is not None:
            msg = "Supply material roles or fixed membership, never both."
            raise ValueError(msg)
        case_partition = partition_contract.build_fixed_partition(
            case_ids, material_by_case=material_by_case, design_kind_by_case=kind_by_case, material_roles=material_roles
        )
    if case_partition is not None:
        case_partition = partition_contract.validate_fixed_partition(
            case_partition, case_ids=case_ids, material_by_case=material_by_case, design_kind_by_case=kind_by_case
        )
    return {
        "schema_kind": identity.DATASET_PACKAGE_SCHEMA_KIND,
        "dataset_name": request["dataset_name"],
        "dataset_view": view.id,
        "registered_task_id": view.registered_task_id,
        "source_task": request["source_task"],
        "materials": materials,
        "channel_contract": _channel_contract(view.id),
        "channel_contract_digest": view.contract_digest,
        "selection": request["selection"],
        "case_partition": case_partition,
        "plan_invocation": plan_invocation,
        "grid": prepared.grid,
        "source_case_identities": list(prepared.source_case_identities),
        "source_outcomes": list(prepared.source_outcomes),
        "builder_identity": {
            **identity.dataset_conversion_contract_identity(view.id),
            "implementation_sha256": reproduction.conversion_implementation_sha256(view.id),
        },
        "producer_evidence": dict(producer_evidence)
        if producer_evidence is not None
        else reproduction.capture_producer_evidence(storage_root=storage_root),
    }


def _build_steady_payload(
    prepared: planning.PreparedPackage,
    *,
    dataset_id: str,
    provenance: Mapping[str, Any],
    destination: Path,
) -> dict[str, int]:
    task = domain.tasks.registry.get_task("steady_flow")
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    sample_ids: list[str] = []
    for admitted, entry in zip(prepared.cases, prepared.source_case_identities, strict=True):
        identity.require_admitted_source_bytes(admitted.hdf5_path, entry["source"]["case_hdf5_sha256"])
        package_case_id = entry["package_case_id"]
        _shape, case_inputs, case_outputs, case_metadata, source, fingerprint = generated.interpret_admitted_source_case(
            admitted,
            package_case_id=package_case_id,
            task=task,
        )
        sample_ids.append(package_case_id)
        inputs.append(case_inputs)
        outputs.append(case_outputs)
        metadata.append(case_metadata)
        sources.append(source)
        fingerprints.append(fingerprint)
    selection_identity = {
        "schema_kind": identity.SOURCE_SELECTION_SCHEMA_KIND,
        "source_cases": list(prepared.source_case_identities),
    }
    payload = identity.build_training_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=sample_ids,
        source_selection_identity=selection_identity,
        source_identities=sources,
        source_metadata=metadata,
        source_provenance=provenance,
        case_fingerprints=fingerprints,
        inputs=torch.stack(inputs),
        outputs=torch.stack(outputs),
    )
    common.serialization.atomic_torch_save(payload, destination)
    return {"sample_count": len(sample_ids), "transition_count": 0}


def _transient_sources(prepared: planning.PreparedPackage) -> tuple[trajectory.TransientSourceCase, ...]:
    return tuple(
        trajectory.TransientSourceCase(
            path=admitted.hdf5_path,
            package_case_id=entry["package_case_id"],
            source_batch_id=entry["source"]["source_batch_id"],
            expected_sha256=entry["source"]["case_hdf5_sha256"],
            expected_case_input_id=entry["source"]["case_input_id"],
            expected_simulation_case_id=entry["source"]["simulation_case_id"],
            material_family=entry["source"]["material_family"],
        )
        for admitted, entry in zip(prepared.cases, prepared.source_case_identities, strict=True)
    )


def _dataset_registry(storage_root: Path | str | None) -> common.identity_registry.IdentityRegistry:
    """Scope the identity registry to the Dataset area of a storage root."""
    return common.identity_registry.IdentityRegistry(common.paths.get_datasets_root(storage_root=storage_root), {_REGISTRY_KIND})


def _identical_content_error(registry: common.identity_registry.IdentityRegistry) -> Callable[[Path, Path], FileExistsError]:
    """Build the refusal for identical content that already lives under another Dataset name."""

    def conflict(existing: Path, _requested: Path) -> FileExistsError:
        return FileExistsError(f"Identical Dataset content already exists as {existing.relative_to(registry.root).as_posix()}; use that name.")

    return conflict


def _publish(
    *,
    task: str,
    dataset_id: str,
    prefix: Mapping[str, Any],
    build_payload: Callable[[Path], Mapping[str, int]],
    selection_request: Mapping[str, Any],
    storage_root: Path | str | None,
    validate_ordered_claim: bool = True,
    verify_producer: Callable[[], None] | None = None,
    expected_payload_sha256: str | None = None,
    output_directory: Path | None = None,
    build_derived_payload: Callable[[Path, Mapping[str, Any]], str] | None = None,
) -> tuple[Path, Path, bool, dict[str, Any]]:
    """Publish once under a lock; incomplete or conflicting identities fail closed.

    Derived payload parts are built in the same staged folder before the manifest, which binds their digest, so the
    manifest, the payload and every derived part become visible in one atomic rename or not at all.
    """
    registry = _dataset_registry(storage_root) if output_directory is None else None
    content_digest = identity.package_content_digest(prefix)
    if output_directory is None:
        payload_dir = common.paths.resolve_dataset_dir(task, dataset_id, dataset_root=common.paths.get_datasets_root(storage_root=storage_root))
        state_root = common.paths.get_dataset_state_root(storage_root=storage_root)
        lock = common.paths.resolve_dataset_build_lock_path(task, dataset_id, storage_root=storage_root)
    else:
        payload_dir = Path(output_directory).absolute()
        dependency_root = common.paths.get_storage_root(storage_root=storage_root).resolve()
        source_root = common.paths.get_project_root().resolve()
        if (
            payload_dir.is_symlink()
            or payload_dir.exists()
            or payload_dir.resolve().is_relative_to(dependency_root)
            or payload_dir.resolve().is_relative_to(source_root)
        ):
            message = "Reconstruction output must be a new directory outside producing source and dependency storage."
            raise ValueError(message)
        state_root = payload_dir.parent
        lock = payload_dir.parent / f".{payload_dir.name}.dataset-build.lock"

    payload_filename = common.paths.get_dataset_payload_filename(task)
    payload_path = payload_dir / payload_filename
    manifest_path = payload_dir / common.paths.DATASET_MANIFEST_FILENAME
    with common.locking.exclusive_file_lock(lock, blocking=False):
        if payload_dir.exists() or payload_dir.is_symlink():
            if output_directory is not None:
                message = "Reconstruction output appeared before acquiring its build lock."
                raise FileExistsError(message)
            if not payload_dir.is_dir() or payload_dir.is_symlink():
                msg = "Dataset publication is incomplete or unsafe; inspect it without automatic repair."
                raise FileExistsError(msg)
            existing = package_manifest.load_package_manifest(task, dataset_id, storage_root=storage_root)
            if identity.package_semantic_identity_payload(existing) != identity.package_semantic_identity_payload(prefix):
                msg = "Dataset name already binds different immutable content; pass --revision <label> for a deliberately separate Dataset."
                raise FileExistsError(msg)
            if validate_ordered_claim and selection_request["selection"] != existing["selection"]:
                selection.validate_ordered_publication_claim(selection_request, storage_root=storage_root)
            if registry is not None:
                registry.ensure_bound(content_digest, payload_dir, kind=_REGISTRY_KIND, conflict=_identical_content_error(registry))
            return payload_path, manifest_path, True, existing
        if verify_producer is not None:
            verify_producer()
        if validate_ordered_claim:
            selection.validate_ordered_publication_claim(selection_request, storage_root=storage_root)
        if registry is not None:
            registry.ensure_available(content_digest, payload_dir, conflict=_identical_content_error(registry))
        state_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{common.paths.dataset_state_key(task, dataset_id)}.", suffix=".tmp", dir=state_root))
        try:
            staged_payload_dir = staging / "payload"
            staged_payload_dir.mkdir()
            staged_payload = staged_payload_dir / payload_filename
            counts = dict(build_payload(staged_payload))
            if set(counts) != {"sample_count", "transition_count"}:
                msg = "Dataset representation returned invalid count metadata."
                raise RuntimeError(msg)
            manifest = {**prefix, **counts, "payload_sha256": common.serialization.file_sha256(staged_payload), "derived_payload_sha256": None}
            if expected_payload_sha256 is not None and manifest["payload_sha256"] != expected_payload_sha256:
                failed = staging.with_name(staging.name.removesuffix(".tmp") + ".rebuild-mismatch")
                common.serialization.atomic_write_json(staged_payload_dir / common.paths.DATASET_MANIFEST_FILENAME, manifest)
                staging.rename(failed)
                message = f"Rebuilt payload bytes differ from the producing manifest; unpublished evidence retained at {failed}."
                raise ValueError(message)
            if build_derived_payload is not None:
                manifest["derived_payload_sha256"] = build_derived_payload(staged_payload_dir, manifest)
            package_manifest.validate_manifest_content(manifest, task=task, dataset_id=dataset_id, payload_path=staged_payload)
            common.serialization.atomic_write_json(staged_payload_dir / common.paths.DATASET_MANIFEST_FILENAME, manifest)
            if verify_producer is not None:
                verify_producer()
            payload_dir.parent.mkdir(parents=True, exist_ok=True)
            binding = (
                contextlib.nullcontext()
                if registry is None
                else registry.reserve(content_digest, payload_dir, kind=_REGISTRY_KIND, conflict=_identical_content_error(registry))
            )
            with binding:
                if payload_dir.exists() or payload_dir.is_symlink():
                    msg = "Dataset publication appeared while holding its build lock."
                    raise FileExistsError(msg)
                staged_payload_dir.rename(payload_dir)
                admitted = (
                    package_manifest.load_package_manifest(task, dataset_id, storage_root=storage_root)
                    if output_directory is None
                    else package_manifest.validate_manifest_content(
                        json.loads(manifest_path.read_text()),
                        task=task,
                        dataset_id=dataset_id,
                        payload_path=payload_path,
                    )
                )
                if admitted != manifest:
                    msg = "Dataset publication no longer matches staged immutable evidence."
                    raise RuntimeError(msg)
            return payload_path, manifest_path, False, admitted
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def build_dataset_selection(
    request: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
    reproduction_manifest: Mapping[str, Any] | None = None,
    output_directory: Path | None = None,
    material_roles: Mapping[str, str] | None = None,
    plan_invocation: list[str] | None = None,
) -> dict[str, Any]:
    """Build one authored ordered Dataset selection from independent source cases."""
    prepared = planning.prepare_dataset_selection(request, storage_root=storage_root)
    producer_evidence = None
    if reproduction_manifest is not None:
        reproduction.verify_current_producer(reproduction_manifest["producer_evidence"], storage_root=storage_root)
        producer_evidence = reproduction_manifest["producer_evidence"]
    provenance = _provenance(
        prepared,
        storage_root=storage_root,
        producer_evidence=producer_evidence,
        material_roles=material_roles,
        case_partition=reproduction_manifest["case_partition"] if reproduction_manifest is not None else None,
        plan_invocation=reproduction_manifest["plan_invocation"] if reproduction_manifest is not None else plan_invocation,
    )
    dataset_id, dataset_digest = identity.package_identity_from_provenance(provenance)
    if reproduction_manifest is not None and provenance["source_outcomes"] != reproduction_manifest["source_outcomes"]:
        message = "Retained Source candidate outcomes differ from the reconstruction manifest."
        raise ValueError(message)
    if reproduction_manifest is not None and (
        dataset_id != reproduction_manifest["dataset_id"]
        or identity.package_semantic_identity_payload(provenance) != identity.package_semantic_identity_payload(reproduction_manifest)
    ):
        message = "Restored producing source or selected Source evidence differs from the reconstruction manifest."
        raise ValueError(message)
    dataset_view = prepared.request["dataset_view"]
    if dataset_view == "steady_flow":

        def build_payload(path: Path) -> Mapping[str, int]:
            return _build_steady_payload(prepared, dataset_id=dataset_id, provenance=provenance, destination=path)

    elif dataset_view == "transient_drying":

        def build_payload(path: Path) -> Mapping[str, int]:
            index = trajectory.build_transient_index(
                _transient_sources(prepared),
                None,
                dataset_name=prepared.request["dataset_name"],
                dataset_id=dataset_id,
                source_root=common.paths.get_storage_root(storage_root=storage_root),
            )
            common.serialization.atomic_write_text(path, json.dumps(index, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            return {"sample_count": index["sample_count"], "transition_count": index["transition_count"]}

    else:
        msg = f"Unsupported Dataset view {dataset_view!r}."
        raise ValueError(msg)
    prefix = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": common.paths.get_dataset_payload_filename(dataset_view),
        "source_case_count": len(prepared.cases),
    }
    payload_path, manifest_path, reused, manifest = _publish(
        task=dataset_view,
        dataset_id=dataset_id,
        prefix=prefix,
        build_payload=build_payload,
        selection_request=prepared.request,
        storage_root=storage_root,
        validate_ordered_claim=reproduction_manifest is None,
        verify_producer=lambda: reproduction.verify_current_producer(provenance["producer_evidence"], storage_root=storage_root),
        expected_payload_sha256=reproduction_manifest["payload_sha256"] if reproduction_manifest is not None else None,
        output_directory=output_directory,
        build_derived_payload=(lambda directory, manifest: transient_shards.write_transient_shards(directory, manifest, storage_root=storage_root))
        if dataset_view == "transient_drying"
        else None,
    )
    return {
        "dataset_name": prepared.request["dataset_name"],
        "dataset_id": dataset_id,
        "dataset_view": dataset_view,
        "payload_path": payload_path,
        "manifest_path": manifest_path,
        "sample_count": manifest["sample_count"],
        "source_case_count": manifest["source_case_count"],
        "transition_count": manifest["transition_count"],
        "status": "reused" if reused else "complete",
        "dependency_storage_root": str(common.paths.get_storage_root(storage_root=storage_root)),
        "independent_source_bytes": False if output_directory is not None else None,
    }
