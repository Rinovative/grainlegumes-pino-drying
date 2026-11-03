"""Admit ordered source cases and prepare one role-neutral Dataset package.

Responsibilities:
  - Resolve exact successful source references through Generation admission
  - Reject repeated physical cases and incompatible grid or view evidence
  - Preserve complete source design provenance and selected membership
  - Bind each contributing Source's replacement policy and all candidate outcomes to the selection

Design principles:
  - Generation alone owns source success, failure, and physical publication
  - Dataset membership is immutable and independent of experiment roles
  - Preview and publication use the same admitted plan

This module does NOT:
  - Assign train, validation, test, or OOD roles
  - Persist package payloads or fit experiment preprocessing
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from src import common
from src.datasets.contracts import dataset_contracts_views as views
from src.generation.publication import generation_publication_catalog as catalog

from . import dataset_packages_selection as selection

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_GRID_AXES = 2


@dataclass(frozen=True, slots=True)
class PreparedPackage:
    """One exact Dataset selection with validated physical source evidence."""

    request: dict[str, Any]
    cases: tuple[catalog.AdmittedRawCase, ...]
    source_case_identities: tuple[dict[str, Any], ...]
    grid: dict[str, Any]
    source_outcomes: tuple[dict[str, Any], ...]


def _package_case_id(source_identity: Mapping[str, Any]) -> str:
    """Return one stable short case ID without losing exact source evidence."""
    digest = common.serialization.canonical_json_sha256(
        {
            "campaign_digest": source_identity["campaign_digest"],
            "case_input_id": source_identity["case_input_id"],
            "simulation_case_id": source_identity["simulation_case_id"],
        }
    )
    return f"{source_identity['material_family']}__{source_identity['case_id']}__{digest[:12]}"


def _grid_identity(path: Path) -> dict[str, Any]:
    """Bind physical coordinate axes without tensorizing a source case."""
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["coords/x"], dtype=np.float64)
        y = np.asarray(handle["coords/y"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < _GRID_AXES or y.size < _GRID_AXES or not np.isfinite(x).all() or not np.isfinite(y).all():
        msg = f"Source case coordinate axes are invalid: {path}."
        raise ValueError(msg)
    digest = hashlib.sha256()
    for axis in (x, y):
        digest.update(len(axis).to_bytes(8, "big"))
        digest.update(np.ascontiguousarray(axis).tobytes())
    return {"shape": [int(y.size), int(x.size)], "axes_sha256": digest.hexdigest()}


def prepare_dataset_selection(
    request: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> PreparedPackage:
    """Admit every requested successful case and bind one exact ordered pool."""
    authored = selection.materialize_selection_request(request, storage_root=storage_root)
    source_task = authored["source_task"]
    dataset_view = authored["dataset_view"]
    cases: list[catalog.AdmittedRawCase] = []
    identities: list[dict[str, Any]] = []
    physical_inputs: set[str] = set()
    simulations: set[str] = set()
    package_ids: set[str] = set()
    first_grid: dict[str, Any] | None = None
    steady_conditioning_digest: str | None = None
    for reference in authored["source_cases"]:
        admitted = catalog.resolve_raw_case_reference(source_task, reference, storage_root=storage_root)
        identity = admitted.source_identity
        design = admitted.design_provenance
        if any(identity.get(key) != expected for key, expected in reference.items()):
            msg = "Generation admitted case differs from the exact Dataset source reference."
            raise ValueError(msg)
        if identity.get("task") != source_task or not isinstance(design, dict):
            msg = "Generation admitted case lacks its source task or design provenance."
            raise ValueError(msg)
        views.validate_view_for_profile(dataset_view, source_task)
        payload = admitted.case.metadata_payload()
        if dataset_view not in payload["available_learning_views"]:
            msg = f"Source case does not provide Dataset view {dataset_view!r}."
            raise ValueError(msg)
        if payload.get("material_role") is not None or payload.get("evaluation_regime") is not None:
            msg = "Historical role-bearing source case is unsupported by the current Dataset contract."
            raise ValueError(msg)
        if dataset_view == "steady_flow":
            conditioning = payload.get("steady_flow_conditioning")
            declared_digest = payload.get("steady_flow_conditioning_digest")
            if not isinstance(conditioning, dict) or declared_digest != common.serialization.canonical_json_sha256(conditioning):
                msg = "Admitted steady source lacks its exact stationary-conditioning contract."
                raise ValueError(msg)
            if steady_conditioning_digest is None:
                steady_conditioning_digest = declared_digest
            elif declared_digest != steady_conditioning_digest:
                msg = "Steady Dataset sources have incompatible package-fixed stationary conditioning."
                raise ValueError(msg)
        case_input_id = str(identity["case_input_id"])
        simulation_case_id = str(identity["simulation_case_id"])
        if case_input_id in physical_inputs or simulation_case_id in simulations:
            msg = "Dataset selection repeats a physical source input or simulation through an alias."
            raise ValueError(msg)
        physical_inputs.add(case_input_id)
        simulations.add(simulation_case_id)
        package_id = _package_case_id(identity)
        if package_id in package_ids:
            msg = "Dataset source case IDs collide."
            raise ValueError(msg)
        package_ids.add(package_id)
        grid = _grid_identity(admitted.hdf5_path)
        if first_grid is None:
            first_grid = grid
        elif grid != first_grid:
            msg = "Dataset source cases have incompatible physical coordinate grids."
            raise ValueError(msg)
        identities.append({"package_case_id": package_id, "source": copy.deepcopy(identity), "design": copy.deepcopy(design)})
        cases.append(admitted)
    if first_grid is None:
        msg = "Dataset selection has no admitted source cases."
        raise ValueError(msg)
    source_names = dict.fromkeys(entry["source"]["source_name"] for entry in identities)
    return PreparedPackage(
        request=authored,
        cases=tuple(cases),
        source_case_identities=tuple(identities),
        grid=first_grid,
        source_outcomes=tuple(catalog.source_outcome_evidence(source_task, name, storage_root=storage_root) for name in source_names),
    )
