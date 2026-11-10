"""
Dataset source admission, package planning, and immutable publication services.

Provides:
- builder: provenance-bound Dataset package construction and publication
- generated_batch: physical source-case view conversion
- manifest: package-manifest publication and admission
- reproduction: producing source capture and researcher-facing reconstruction evidence
- planning: deterministic exact source-case admission
- trajectory: transient source admission and compact trajectory indexes
- transient_shards: Dataset-bound derived PT shards, written inside the Dataset publication, and their admission
- DATASET_PACKAGE_SCHEMA_KIND: canonical package-manifest schema kind
- DEFAULT_TRANSIENT_PT_SHARD_BYTES: default soft PT-shard packing target
- build_dataset_selection: publish one independent immutable ordered Dataset
- discover_source_collections: inspect successful and failed source outcomes
- select_exact_cases and select_ordered_cases: author exact case references
- load_package_manifest: load one canonical package manifest
- load_package_manifest_evidence: admit manifest metadata without payload rehashing
- declared_evaluation_assignments: project declared Dataset OOD membership
- inspect_dataset_package: inspect one package and runtime sample
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.datasets.dataset_packages import build_dataset_selection, inspect_dataset_package

    from . import dataset_packages_builder as builder
    from . import dataset_packages_generated_batch as generated_batch
    from . import dataset_packages_manifest as manifest
    from . import dataset_packages_planning as planning
    from . import dataset_packages_reproduction as reproduction
    from . import dataset_packages_trajectory as trajectory
    from . import dataset_packages_transient_shards as transient_shards
    from .dataset_packages_manifest import (
        DATASET_PACKAGE_SCHEMA_KIND,
        declared_evaluation_assignments,
        load_package_manifest,
        load_package_manifest_evidence,
    )
    from .dataset_packages_selection import discover_source_collections, select_exact_cases, select_ordered_cases

_MODULES = {
    "builder": "dataset_packages_builder",
    "generated_batch": "dataset_packages_generated_batch",
    "manifest": "dataset_packages_manifest",
    "reproduction": "dataset_packages_reproduction",
    "planning": "dataset_packages_planning",
    "trajectory": "dataset_packages_trajectory",
    "transient_shards": "dataset_packages_transient_shards",
}
DEFAULT_TRANSIENT_PT_SHARD_BYTES: Final = int(1.5 * 1024**3)
_FACADE_EXPORTS = frozenset({"build_dataset_selection", "inspect_dataset_package"})
_EXPORTS = {
    "DATASET_PACKAGE_SCHEMA_KIND": "dataset_packages_manifest",
    "declared_evaluation_assignments": "dataset_packages_manifest",
    "load_package_manifest": "dataset_packages_manifest",
    "load_package_manifest_evidence": "dataset_packages_manifest",
    "discover_source_collections": "dataset_packages_selection",
    "select_exact_cases": "dataset_packages_selection",
    "select_ordered_cases": "dataset_packages_selection",
}
__all__ = [
    "DATASET_PACKAGE_SCHEMA_KIND",
    "DEFAULT_TRANSIENT_PT_SHARD_BYTES",
    "build_dataset_selection",
    "builder",
    "declared_evaluation_assignments",
    "discover_source_collections",
    "generated_batch",
    "inspect_dataset_package",
    "load_package_manifest",
    "load_package_manifest_evidence",
    "manifest",
    "planning",
    "reproduction",
    "select_exact_cases",
    "select_ordered_cases",
    "trajectory",
    "transient_shards",
]


def __getattr__(name: str) -> object:
    """Resolve one declared package module or canonical lifecycle operation."""
    module_name = _MODULES.get(name)
    if module_name is not None:
        value = import_module(f"{__name__}.{module_name}")
    elif name in _FACADE_EXPORTS:
        value = getattr(import_module("src.datasets.dataset_packages"), name)
    elif name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    else:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    globals()[name] = value
    return value
