"""
dataset_packages.py

Expose the canonical Dataset package builder identity and supported package CLI.
Responsibilities:
  - Preserve the persisted package-builder callable identity
  - Delegate package build, admission, inspection, and smoke operations
  - Parse the supported build, inspect, and smoke commands
Design principles:
  - The facade remains thin while its qualified public identity stays stable
  - Package construction and runtime validation each have one responsible owner
  - Command output is deterministic, sorted JSON
This module does NOT:
  - Implement package publication or Dataset loading
  - Duplicate manifest, identity, or runtime validation logic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src import generation

    from .contracts import dataset_contracts_identity as identity

_SCHEMA_EXPORTS = frozenset(
    {
        "DATASET_PACKAGE_SCHEMA_KIND",
        "DATASET_PACKAGE_SCHEMA_VERSION",
    }
)


def __getattr__(name: str) -> object:
    """Resolve one manifest schema constant without introducing a facade cycle."""
    if name not in _SCHEMA_EXPORTS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    from .packages import dataset_packages_manifest as package_manifest  # noqa: PLC0415

    value = getattr(package_manifest, name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    DATASET_PACKAGE_SCHEMA_KIND: str
    DATASET_PACKAGE_SCHEMA_VERSION: int


def storage_schema_version() -> int:
    """Return the canonical source-case HDF5 schema identity."""
    from .packages import dataset_packages_builder as builder  # noqa: PLC0415

    return builder.storage_schema_version()


def build_dataset_package(
    campaign: generation.cases.config.CampaignConfig,
    dataset_view: str,
    evaluation_regime: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build one declared package with its required ID leakage companion."""
    from .packages import dataset_packages_builder as builder  # noqa: PLC0415

    return builder.build_dataset_package(
        campaign,
        dataset_view,
        evaluation_regime,
        storage_root=storage_root,
    )


def build_campaign_packages(
    campaign: generation.cases.config.CampaignConfig,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build every declared package after one shared membership preflight."""
    from .packages import dataset_packages_builder as builder  # noqa: PLC0415

    return builder.build_campaign_packages(campaign, storage_root=storage_root)


def load_package_manifest(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate one package manifest and its exact payload hash."""
    from .packages import dataset_packages_manifest as package_manifest  # noqa: PLC0415

    return package_manifest.load_package_manifest(dataset_id, storage_root=storage_root)


def load_dataset_package_manifest(
    dataset_id: str,
    *,
    dataset_identity: identity.DatasetIdentity,
    dataset_path: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Bind a steady package manifest to its validated tensor payload identity."""
    from .packages import dataset_packages_manifest as package_manifest  # noqa: PLC0415

    return package_manifest.load_steady_package_manifest(
        dataset_id,
        dataset_identity=dataset_identity,
        dataset_path=dataset_path,
        metadata_root=metadata_root,
    )


def inspect_dataset_package(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect one package through its validated manifest and runtime object."""
    from .runtime import dataset_runtime_package_validation as package_validation  # noqa: PLC0415

    return package_validation.inspect_dataset_package(dataset_id, storage_root=storage_root)


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
    """Load one batch through the unified factory with requested workers."""
    from .runtime import dataset_runtime_package_validation as package_validation  # noqa: PLC0415

    return package_validation.smoke_dataset_package(
        dataset_id,
        storage_root=storage_root,
        membership=membership,
        ood_group=ood_group,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        hdf5_cache_size=hdf5_cache_size,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the Dataset package build, inspection, and smoke CLI."""
    from .contracts import dataset_contracts_views as views  # noqa: PLC0415

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
    from src import generation  # noqa: PLC0415

    arguments = _build_parser().parse_args()
    if arguments.command == "build":
        campaign = generation.cases.config.load_campaign_config(arguments.campaign_config)
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
