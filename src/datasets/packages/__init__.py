"""
Dataset source admission, package planning, and immutable publication services.

Provides:
- builder: provenance-bound Dataset package construction and publication
- generated_batch: generated-simulation batch admission
- manifest: package-manifest publication and admission
- planning: deterministic source and membership planning
- trajectory: transient source admission and compact trajectory indexes
- DATASET_PACKAGE_SCHEMA_KIND: canonical package-manifest schema kind
- DATASET_PACKAGE_SCHEMA_VERSION: canonical package-manifest schema version
- build_dataset_package: build one declared campaign package
- build_campaign_packages: build every declared campaign package
- load_package_manifest: load one canonical package manifest
- load_dataset_package_manifest: bind a manifest to a steady payload identity
- inspect_dataset_package: inspect one package and runtime sample
- smoke_dataset_package: run one bounded package loader smoke
- storage_schema_version: return the canonical source-storage schema version
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.datasets.dataset_packages import (
        build_campaign_packages,
        build_dataset_package,
        inspect_dataset_package,
        load_dataset_package_manifest,
        load_package_manifest,
        smoke_dataset_package,
        storage_schema_version,
    )

    from . import dataset_packages_builder as builder
    from . import dataset_packages_generated_batch as generated_batch
    from . import dataset_packages_manifest as manifest
    from . import dataset_packages_planning as planning
    from . import dataset_packages_trajectory as trajectory
    from .dataset_packages_manifest import (
        DATASET_PACKAGE_SCHEMA_KIND,
        DATASET_PACKAGE_SCHEMA_VERSION,
    )

_MODULES = {
    "builder": "dataset_packages_builder",
    "generated_batch": "dataset_packages_generated_batch",
    "manifest": "dataset_packages_manifest",
    "planning": "dataset_packages_planning",
    "trajectory": "dataset_packages_trajectory",
}
_FACADE_EXPORTS = frozenset(
    {
        "build_campaign_packages",
        "build_dataset_package",
        "inspect_dataset_package",
        "load_dataset_package_manifest",
        "load_package_manifest",
        "smoke_dataset_package",
        "storage_schema_version",
    }
)
_MANIFEST_EXPORTS = frozenset(
    {
        "DATASET_PACKAGE_SCHEMA_KIND",
        "DATASET_PACKAGE_SCHEMA_VERSION",
    }
)
__all__ = [
    "DATASET_PACKAGE_SCHEMA_KIND",
    "DATASET_PACKAGE_SCHEMA_VERSION",
    "build_campaign_packages",
    "build_dataset_package",
    "builder",
    "generated_batch",
    "inspect_dataset_package",
    "load_dataset_package_manifest",
    "load_package_manifest",
    "manifest",
    "planning",
    "smoke_dataset_package",
    "storage_schema_version",
    "trajectory",
]


def __getattr__(name: str) -> object:
    """Resolve one declared package module or canonical lifecycle operation."""
    module_name = _MODULES.get(name)
    if module_name is not None:
        value = import_module(f"{__name__}.{module_name}")
    elif name in _FACADE_EXPORTS:
        value = getattr(import_module("src.datasets.dataset_packages"), name)
    elif name in _MANIFEST_EXPORTS:
        value = getattr(import_module(f"{__name__}.dataset_packages_manifest"), name)
    else:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    globals()[name] = value
    return value
