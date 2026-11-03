"""Inspect immutable Dataset tensor samples through the unified runtime.

Responsibilities:
  - Bind validated manifests to typed runtime Dataset requests
  - Report representative tensor and source identities

Design principles:
  - Runtime inspection uses the same factory as training consumers
  - Manifest and payload admission precede every Dataset read
  - Transient HDF5 handles are closed after bounded inspection

This module does NOT:
  - Build or publish Dataset packages
  - Implement the supported command-line interface
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.packages import dataset_packages_manifest as package_manifest

from . import dataset_runtime_factory as factory
from . import dataset_runtime_transient as transient

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import torch


def _runtime_request(
    manifest: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    case_ids: tuple[str, ...] | None = None,
) -> Any:
    """Build one typed factory request from a validated package manifest."""
    return factory.DatasetRequest(
        dataset_id=str(manifest["dataset_id"]),
        dataset_view=manifest["dataset_view"],
        case_ids=case_ids,
        transient_sampling=(
            transient_contract.TransientSamplingSpec(mode="one_step_transition") if manifest["dataset_view"] == "transient_drying" else None
        ),
        storage_root=storage_root,
    )


def _tensor_description(tensor: torch.Tensor, channels: Any) -> dict[str, Any]:
    """Return one inspection tensor shape, dtype, and channel declaration."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "channels": copy.deepcopy(channels),
    }


def _inspect_dataset_package(
    manifest: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect one runtime from package evidence admitted by its caller."""
    request = _runtime_request(
        manifest,
        storage_root=storage_root,
    )
    runtime = factory._create_dataset_from_manifest(  # noqa: SLF001 -- shared admitted evidence
        request,
        manifest,
        hdf5_cache_size=1,
    )
    storage_backend = getattr(runtime, "storage_backend", "canonical_tensor")
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
                "source_case_id",
                "simulation_profile",
                "material_family",
                "domain_profile",
            )
        }
    else:
        tensor_report = {
            name: _tensor_description(sample[name], manifest["channel_contract"][name])
            for name in ("state", "static", "boundary", "scalars", "target")
        }
        temporal_contract = {field["name"]: field for field in manifest["channel_contract"]["time"]["fields"]}
        tensor_report["time"] = {name: _tensor_description(sample["time"][name], temporal_contract[name]) for name in ("t_n", "t_n_plus_1", "dt")}
        sample_identity = dict(sample["metadata"])
        if isinstance(runtime, transient.TransientPhysicalDataset):
            runtime.close()
    materials = list(manifest["materials"])
    return {
        "dataset_name": manifest["dataset_name"],
        "dataset_id": manifest["dataset_id"],
        "dataset_view": manifest["dataset_view"],
        "registered_task_id": manifest["registered_task_id"],
        "case_ids": [entry["package_case_id"] for entry in manifest["source_case_identities"]],
        "source_case_count": manifest["source_case_count"],
        "transition_count": manifest["transition_count"],
        "sample_count": manifest["sample_count"],
        "materials": materials,
        "tensors": tensor_report,
        "sample_identity": sample_identity,
        "storage_backend": storage_backend,
        "training_payload_ready": (storage_backend == "pt_shards" if manifest["dataset_view"] == "transient_drying" else None),
    }


def inspect_dataset_package(
    task: str,
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect one package through its validated manifest and runtime object."""
    manifest = package_manifest.load_package_manifest(
        task,
        dataset_id,
        storage_root=storage_root,
    )
    return _inspect_dataset_package(manifest, storage_root=storage_root)
