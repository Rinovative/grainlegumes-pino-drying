"""
===============================================================================
dataset_runtime_package_validation.py
===============================================================================
Inspect and smoke-load immutable Dataset packages through the unified runtime.
Responsibilities:
  - Bind validated manifests to typed runtime Dataset requests
  - Report representative tensor and source identities
  - Exercise bounded DataLoader worker settings for package smoke checks
Design principles:
  - Runtime inspection uses the same factory as training consumers
  - Manifest and payload admission precede every Dataset read
  - Transient HDF5 handles are closed after bounded inspection
This module does NOT:
  - Build or publish Dataset packages
  - Implement the supported command-line interface
===============================================================================
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import torch

from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.packages import dataset_packages_manifest as package_manifest

from . import dataset_runtime_factory as factory
from . import dataset_runtime_transient as transient

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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
    manifest = package_manifest.load_package_manifest(dataset_id, storage_root=storage_root)
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
    manifest = package_manifest.load_package_manifest(dataset_id, storage_root=storage_root)
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
