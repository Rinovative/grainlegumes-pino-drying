"""
===============================================================================
dataset_build.py
===============================================================================
Build and atomically publish one canonical steady-flow training dataset.

Responsibilities:
  - Interpret the steady-flow view of either supported simulation profile
  - Construct bounded tensors in the registered TaskSpec channel order
  - Bind profile/template provenance into dataset and metadata identity
  - Coordinate locking, recovery, validation, and atomic publication

Design principles:
  - Generated-source admission is owned by ``dataset_generated_batch``
  - One task-owned builder handles standalone and coupled airflow references
  - Ready transactions make two-directory publication safely recoverable

This module does NOT:
  - Run COMSOL, define material families, or infer export expressions
  - Admit superseded generated-batch or sampling schemas
  - Define or publish a transient-drying tensor dataset
===============================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from src import common, datasets, domain

from . import dataset_generated_batch as generated

if TYPE_CHECKING:
    from src.datasets.dataset_identity import DatasetIdentity
    from src.datasets.dataset_metadata import DatasetMetadata
    from src.domain.tasks.domain_task_spec import TaskSpec

_PUBLICATION_TRANSACTION_SCHEMA_KIND = "training_dataset_publication_transaction"
_PUBLICATION_TRANSACTION_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_PUBLICATION_TRANSACTION_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_id",
        "phase",
        "staging_root",
        "dataset_sha256",
        "dataset_size",
        "dataset_metadata_sha256",
    }
)


def _timing_summary(sample_count: int) -> dict[str, Any]:
    """Describe the deliberate absence of a batch-level timing snapshot."""
    return {
        "status": "unavailable",
        "intended_case_count": sample_count,
        "measured_case_count": 0,
    }


def _snapshot_entry(path: Path) -> dict[str, Any]:
    """Describe the required current terminal-manifest snapshot."""
    return {
        "sha256": generated.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "required": True,
        "role": "validated_generation_manifest",
    }


def _stage_metadata_package(
    destination: Path,
    *,
    dataset_identity: DatasetIdentity,
    dataset_sha256: str,
    dataset_size: int,
    task: TaskSpec,
    source_manifest: dict[str, Any],
) -> None:
    """Stage one self-contained current model-training metadata package."""
    destination.mkdir(parents=True)
    manifest_path = destination / datasets.metadata.SOURCE_MANIFEST_FILENAME
    common.serialization.atomic_write_json(manifest_path, source_manifest)
    generated_identity = dataset_identity.generated_batch_identity
    if generated_identity is None:
        msg = "Final dataset identity is missing generated-batch provenance."
        raise RuntimeError(msg)
    source_template = source_manifest["template"]
    scientific = {
        "dataset_schema_version": datasets.identity.TRAINING_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_identity.fingerprint,
        "task_id": task.id,
        "data_contract_digest": dataset_identity.data_contract_digest,
        "source_batch_id": source_manifest["batch_id"],
        "source_simulation_profile": source_manifest["simulation_profile"],
        "source_template_sha256": source_template["sha256"],
        "airflow_source": source_manifest["airflow_source"],
        "generated_batch_identity_sha256": generated_identity["batch_manifest_identity_sha256"],
        "sample_count": dataset_identity.sample_count,
        "spatial_shape": list(dataset_identity.spatial_shape),
        "tensors": {
            "inputs": {
                "dtype": "float32",
                "shape": [dataset_identity.sample_count, task.in_channels, *dataset_identity.spatial_shape],
            },
            "outputs": {
                "dtype": "float32",
                "shape": [dataset_identity.sample_count, task.out_channels, *dataset_identity.spatial_shape],
            },
        },
    }
    metadata = {
        "schema_kind": datasets.metadata.METADATA_SCHEMA_KIND,
        "schema_version": datasets.metadata.METADATA_SCHEMA_VERSION,
        "dataset_id": dataset_identity.dataset_id,
        "scientific_identity": scientific,
        "artifacts": {
            "dataset": {
                "filename": f"{dataset_identity.dataset_id}.pt",
                "sha256": dataset_sha256,
                "size_bytes": dataset_size,
            },
            "snapshots": {
                datasets.metadata.SOURCE_MANIFEST_FILENAME: _snapshot_entry(manifest_path),
            },
        },
        "operational_provenance": {
            "builder_module": datasets.metadata.BUILDER_MODULE,
            "publication_method": datasets.metadata.PUBLICATION_METHOD,
            "source_manifest_sha256": generated.sha256_file(manifest_path),
            "timing": _timing_summary(dataset_identity.sample_count),
        },
    }
    common.serialization.atomic_write_json(destination / datasets.metadata.METADATA_FILENAME, metadata)


def _transaction_record(
    *,
    dataset_id: str,
    phase: str,
    staging_root: Path,
    dataset_sha256: str = "",
    dataset_size: int = 0,
    metadata_sha256: str = "",
) -> dict[str, Any]:
    """Build one exact dataset publication transaction marker."""
    if phase not in {"building", "ready"}:
        msg = f"Unsupported dataset publication phase: {phase!r}."
        raise ValueError(msg)
    return {
        "schema_kind": _PUBLICATION_TRANSACTION_SCHEMA_KIND,
        "schema_version": _PUBLICATION_TRANSACTION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "phase": phase,
        "staging_root": str(staging_root.resolve(strict=False)),
        "dataset_sha256": dataset_sha256,
        "dataset_size": dataset_size,
        "dataset_metadata_sha256": metadata_sha256,
    }


def _load_transaction(path: Path, *, datasets_root: Path, dataset_id: str) -> tuple[dict[str, Any], Path]:
    """Load and constrain one recovery marker to this dataset staging area."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Dataset publication transaction is unreadable: {path}"
        raise RuntimeError(msg) from error
    if not isinstance(value, dict) or set(value) != _PUBLICATION_TRANSACTION_KEYS:
        msg = f"Dataset publication transaction has an invalid schema: {path}"
        raise RuntimeError(msg)
    if (
        value["schema_kind"] != _PUBLICATION_TRANSACTION_SCHEMA_KIND
        or value["schema_version"] != _PUBLICATION_TRANSACTION_SCHEMA_VERSION
        or value["dataset_id"] != dataset_id
        or value["phase"] not in {"building", "ready"}
    ):
        msg = f"Dataset publication transaction identity is invalid: {path}"
        raise RuntimeError(msg)
    staging_root = Path(value["staging_root"])
    resolved_root = datasets_root.resolve(strict=False)
    resolved_staging = staging_root.resolve(strict=False)
    if resolved_staging.parent != resolved_root or not resolved_staging.name.startswith(f".{dataset_id}.dataset-build."):
        msg = f"Dataset publication staging root is outside its owned area: {staging_root}"
        raise RuntimeError(msg)
    if value["phase"] == "ready":
        if not isinstance(value["dataset_size"], int) or value["dataset_size"] <= 0:
            msg = "Ready dataset publication transaction has an invalid artifact size."
            raise RuntimeError(msg)
        for key in ("dataset_sha256", "dataset_metadata_sha256"):
            digest = value[key]
            if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in digest):
                msg = f"Ready dataset publication transaction has invalid {key}."
                raise RuntimeError(msg)
    return value, resolved_staging


def _publish_ready(
    *,
    record: dict[str, Any],
    staging_root: Path,
    destination_dir: Path,
    metadata_destination: Path,
    dataset_id: str,
    task: TaskSpec,
) -> tuple[DatasetIdentity, DatasetMetadata]:
    """Publish or finish publishing one validated ready transaction."""
    staged_dataset_dir = staging_root / "payload" / dataset_id
    staged_metadata_dir = staging_root / "metadata" / dataset_id
    dataset_source = destination_dir if destination_dir.is_dir() else staged_dataset_dir
    metadata_source = metadata_destination if metadata_destination.is_dir() else staged_metadata_dir
    dataset_path = dataset_source / f"{dataset_id}.pt"
    metadata_path = metadata_source / datasets.metadata.METADATA_FILENAME
    if not dataset_path.is_file() or dataset_path.is_symlink():
        msg = "Ready transaction is missing its staged or final dataset payload."
        raise RuntimeError(msg)
    if (
        dataset_path.stat().st_size != record["dataset_size"]
        or generated.sha256_file(dataset_path) != record["dataset_sha256"]
        or not metadata_path.is_file()
        or generated.sha256_file(metadata_path) != record["dataset_metadata_sha256"]
    ):
        msg = "Ready transaction artifacts do not match their durable identities."
        raise RuntimeError(msg)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    try:
        identity = datasets.identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    finally:
        del payload
        gc.collect()
    datasets.metadata.validate_dataset_metadata_directory(
        metadata_source,
        dataset_identity=identity,
        dataset_path=dataset_path,
    )
    if destination_dir.exists() and not destination_dir.is_dir():
        msg = f"Final dataset target is not a directory: {destination_dir}"
        raise FileExistsError(msg)
    if metadata_destination.exists() and not metadata_destination.is_dir():
        msg = f"Final metadata target is not a directory: {metadata_destination}"
        raise FileExistsError(msg)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    moved_metadata = False
    try:
        if metadata_source == staged_metadata_dir:
            staged_metadata_dir.replace(metadata_destination)
            moved_metadata = True
        if dataset_source == staged_dataset_dir:
            staged_dataset_dir.replace(destination_dir)
    except BaseException:
        if moved_metadata and metadata_destination.is_dir() and not staged_metadata_dir.exists():
            staged_metadata_dir.parent.mkdir(parents=True, exist_ok=True)
            metadata_destination.replace(staged_metadata_dir)
        raise
    final_path = destination_dir / f"{dataset_id}.pt"
    final_package = datasets.metadata.validate_dataset_metadata_directory(
        metadata_destination,
        dataset_identity=identity,
        dataset_path=final_path,
    )
    return identity, final_package


def _recover_interrupted_publication(
    transaction_path: Path,
    *,
    datasets_root: Path,
    destination_dir: Path,
    metadata_destination: Path,
    dataset_id: str,
    task: TaskSpec,
) -> tuple[DatasetIdentity, DatasetMetadata] | None:
    """Discard interrupted private construction or finish one ready publication."""
    if not transaction_path.exists():
        return None
    if not transaction_path.is_file() or transaction_path.is_symlink():
        msg = f"Dataset publication transaction is unsafe: {transaction_path}"
        raise RuntimeError(msg)
    record, staging_root = _load_transaction(transaction_path, datasets_root=datasets_root, dataset_id=dataset_id)
    if record["phase"] == "building":
        if staging_root.exists():
            shutil.rmtree(staging_root)
        transaction_path.unlink()
        return None
    identity, package = _publish_ready(
        record=record,
        staging_root=staging_root,
        destination_dir=destination_dir,
        metadata_destination=metadata_destination,
        dataset_id=dataset_id,
        task=task,
    )
    transaction_path.unlink()
    if staging_root.exists():
        shutil.rmtree(staging_root)
    return identity, package


def _assert_manifest_current(path: Path, expected_sha256: str) -> None:
    """Fail when terminal generation evidence changes during dataset construction."""
    if generated.sha256_file(path) != expected_sha256:
        msg = "Terminal simulation manifest changed during dataset construction."
        raise RuntimeError(msg)


def build_batch_dataset(  # noqa: PLR0915
    batch_name: str,
    verbose: bool = False,
    *,
    dataset_id: str | None = None,
    task_id: str = "steady_flow",
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build one steady-flow dataset from either supported simulation profile."""
    task = domain.tasks.registry.get_task(task_id)
    if task.id != "steady_flow":
        msg = f"Generated airflow views support only the current steady_flow task, got {task.id!r}."
        raise ValueError(msg)
    source_batch = common.paths.validate_logical_name(batch_name, label="batch_name")
    resolved_dataset_id = common.paths.validate_logical_name(dataset_id or source_batch, label="dataset_id")
    generation_root = common.paths.get_generation_root(storage_root=storage_root)
    datasets_root = common.paths.get_datasets_root(storage_root=storage_root)
    dataset_metadata_root = common.paths.get_dataset_metadata_root(storage_root=storage_root)
    dataset_payload_root = common.paths.get_dataset_payload_root(storage_root=storage_root)
    destination_dir = common.paths.resolve_dataset_dir(resolved_dataset_id, dataset_root=dataset_payload_root)
    destination = common.paths.resolve_dataset_path(resolved_dataset_id, dataset_root=dataset_payload_root)
    metadata_destination = common.paths.resolve_dataset_metadata_dir(resolved_dataset_id, metadata_root=dataset_metadata_root)
    lock_path = common.paths.resolve_dataset_build_lock_path(resolved_dataset_id, storage_root=storage_root)
    transaction_path = common.paths.resolve_dataset_build_transaction_path(resolved_dataset_id, storage_root=storage_root)
    datasets_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any]

    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        recovered = _recover_interrupted_publication(
            transaction_path,
            datasets_root=datasets_root,
            destination_dir=destination_dir,
            metadata_destination=metadata_destination,
            dataset_id=resolved_dataset_id,
            task=task,
        )
        if recovered is not None:
            recovered_identity, recovered_metadata = recovered
            result = {
                "source_batch": source_batch,
                "source_profile": recovered_metadata.source_manifest["simulation_profile"],
                "generation_root": generation_root,
                "dataset_path": destination,
                "metadata_path": metadata_destination,
                "case_count": recovered_identity.sample_count,
                "task": task.id,
                "data_contract_digest": recovered_identity.data_contract_digest,
                "timing_coverage": recovered_metadata.timing_summary,
                "dataset_fingerprint": recovered_identity.fingerprint,
                "status": "complete",
            }
            if verbose:
                for key, value in result.items():
                    print(f"{key}: {value}")
            return result
        if destination_dir.exists() or destination_dir.is_symlink():
            msg = f"Refusing to overwrite existing final dataset: {destination_dir}"
            raise FileExistsError(msg)
        if metadata_destination.exists() or metadata_destination.is_symlink():
            msg = f"Refusing to overwrite existing dataset metadata: {metadata_destination}"
            raise FileExistsError(msg)
        manifest, manifest_path, manifest_sha256 = generated.load_batch_manifest(
            source_batch,
            storage_root=storage_root,
        )
        generated.validate_terminal_batch(source_batch, storage_root=storage_root)
        if "steady_flow" not in manifest["available_learning_views"]:
            msg = f"Simulation batch {source_batch!r} does not expose a validated steady_flow learning view."
            raise ValueError(msg)
        generated_identity = datasets.identity.build_generated_batch_identity(manifest)
        case_ids = list(generated_identity["intended_case_ids"])
        records_by_id = {record["case_id"]: record for record in manifest["cases"]}
        provenance = generated.source_provenance(manifest, manifest_sha256=manifest_sha256)
        staging_root = Path(tempfile.mkdtemp(dir=datasets_root, prefix=f".{resolved_dataset_id}.dataset-build.", suffix=".tmp"))
        staged_dataset_dir = staging_root / "payload" / resolved_dataset_id
        staged_metadata_dir = staging_root / "metadata" / resolved_dataset_id
        staged_dataset_path = staged_dataset_dir / f"{resolved_dataset_id}.pt"
        transaction_active = False
        ready = False
        inputs: torch.Tensor | None = None
        outputs: torch.Tensor | None = None
        try:
            common.serialization.atomic_write_json(
                transaction_path,
                _transaction_record(dataset_id=resolved_dataset_id, phase="building", staging_root=staging_root),
            )
            transaction_active = True
            staged_dataset_dir.mkdir(parents=True)
            source_identities: list[dict[str, Any]] = []
            source_metadata: list[dict[str, Any]] = []
            fingerprints: list[str] = []
            reference_shape: tuple[int, int] | None = None
            iterator = tqdm(case_ids, desc=f"Building {source_batch}", unit="case", disable=not verbose)
            for index, case_id in enumerate(iterator):
                spatial_shape, case_inputs, case_outputs, metadata, source_identity, fingerprint = generated.interpret_generated_case(
                    source_batch,
                    case_id,
                    task=task,
                    manifest=manifest,
                    record=records_by_id[case_id],
                    storage_root=storage_root,
                )
                if reference_shape is None:
                    reference_shape = spatial_shape
                    inputs = torch.empty((len(case_ids), task.in_channels, *spatial_shape), dtype=torch.float32)
                    outputs = torch.empty((len(case_ids), task.out_channels, *spatial_shape), dtype=torch.float32)
                elif spatial_shape != reference_shape:
                    msg = f"Inconsistent case shape for {case_id!r}: {spatial_shape} != {reference_shape}."
                    raise ValueError(msg)
                if inputs is None or outputs is None:
                    msg = "Final tensors were not allocated."
                    raise RuntimeError(msg)
                inputs[index].copy_(case_inputs)
                outputs[index].copy_(case_outputs)
                source_identities.append(source_identity)
                source_metadata.append(metadata)
                fingerprints.append(fingerprint)
            if inputs is None or outputs is None:
                msg = f"No complete generated cases found for {source_batch!r}."
                raise RuntimeError(msg)
            payload = datasets.identity.build_training_dataset_payload(
                task=task,
                dataset_id=resolved_dataset_id,
                sample_ids=case_ids,
                generated_batch_identity=generated_identity,
                source_identities=source_identities,
                source_metadata=source_metadata,
                source_provenance=provenance,
                case_fingerprints=fingerprints,
                inputs=inputs,
                outputs=outputs,
            )
            fingerprint = str(payload["dataset_fingerprint"])
            common.serialization.atomic_torch_save(payload, staged_dataset_path)
            del payload, inputs, outputs
            inputs = None
            outputs = None
            gc.collect()
            staged_payload = torch.load(staged_dataset_path, map_location="cpu", weights_only=False)
            try:
                staged_identity = datasets.identity.validate_training_dataset_payload(
                    staged_payload,
                    task=task,
                    verify_content=True,
                )
            finally:
                del staged_payload
                gc.collect()
            dataset_sha256 = generated.sha256_file(staged_dataset_path)
            dataset_size = staged_dataset_path.stat().st_size
            _stage_metadata_package(
                staged_metadata_dir,
                dataset_identity=staged_identity,
                dataset_sha256=dataset_sha256,
                dataset_size=dataset_size,
                task=task,
                source_manifest=manifest,
            )
            datasets.metadata.validate_dataset_metadata_directory(
                staged_metadata_dir,
                dataset_identity=staged_identity,
                dataset_path=staged_dataset_path,
            )
            _assert_manifest_current(manifest_path, manifest_sha256)
            metadata_sha256 = generated.sha256_file(staged_metadata_dir / datasets.metadata.METADATA_FILENAME)
            record = _transaction_record(
                dataset_id=resolved_dataset_id,
                phase="ready",
                staging_root=staging_root,
                dataset_sha256=dataset_sha256,
                dataset_size=dataset_size,
                metadata_sha256=metadata_sha256,
            )
            common.serialization.atomic_write_json(transaction_path, record)
            ready = True
            _assert_manifest_current(manifest_path, manifest_sha256)
            published_identity, published_metadata = _publish_ready(
                record=record,
                staging_root=staging_root,
                destination_dir=destination_dir,
                metadata_destination=metadata_destination,
                dataset_id=resolved_dataset_id,
                task=task,
            )
            transaction_path.unlink()
            transaction_active = False
        finally:
            del inputs, outputs
            if not ready and transaction_active:
                transaction_path.unlink(missing_ok=True)
            if not ready and staging_root.exists():
                shutil.rmtree(staging_root)
            if not transaction_active and staging_root.exists():
                shutil.rmtree(staging_root)
        result = {
            "source_batch": source_batch,
            "source_profile": manifest["simulation_profile"],
            "generation_root": generation_root,
            "dataset_path": destination,
            "metadata_path": metadata_destination,
            "case_count": published_identity.sample_count,
            "task": task.id,
            "data_contract_digest": task.data_contract_digest,
            "timing_coverage": published_metadata.timing_summary,
            "dataset_fingerprint": fingerprint,
            "status": "complete",
        }
    if verbose:
        for key, value in result.items():
            print(f"{key}: {value}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build the direct dataset-publication command parser."""
    parser = argparse.ArgumentParser(description="Build a steady-flow dataset from one completed simulation batch.")
    parser.add_argument("batch_id", help="Completed source simulation batch identifier")
    parser.add_argument("--dataset-id", default=None, help="Final dataset identifier; defaults to the batch identifier")
    parser.add_argument("--task", default="steady_flow", help="Registered task identifier")
    parser.add_argument("--storage-root", type=Path, default=None, help="Override STORAGE_ROOT for this invocation")
    parser.add_argument("--verbose", action="store_true", help="Show bounded progress and final identity")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the maintained direct final-dataset build command."""
    args = _build_parser().parse_args(argv)
    try:
        result = build_batch_dataset(
            args.batch_id,
            dataset_id=args.dataset_id,
            task_id=args.task,
            storage_root=args.storage_root,
            verbose=args.verbose,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Dataset build failed: {type(error).__name__}: {error}")
        return 1
    print(f"Source batch: {result['source_batch']}")
    print(f"Source profile: {result['source_profile']}")
    print(f"Destination dataset: {result['dataset_path']}")
    print(f"Dataset fingerprint: {result['dataset_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
