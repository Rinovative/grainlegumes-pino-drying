"""Resolve exact source-case selections for immutable Dataset publication.

Responsibilities:
  - Discover source collections with successful and failed case counts
  - Resolve exact or seeded case choices into immutable source references
  - Validate one role-neutral authored Dataset selection request

Design principles:
  - Generation owns source admission and success or failure evidence
  - Selection order and original design provenance remain explicit
  - A Dataset request never assigns experiment evaluation roles

This module does NOT:
  - Create Generation campaigns or edit source cases
  - Publish Dataset payloads or assign train and evaluation membership
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.datasets.contracts import dataset_contracts_views as views
from src.generation.publication import generation_publication_catalog as catalog

if TYPE_CHECKING:
    from pathlib import Path

SELECTION_SCHEMA_KIND: Final = "dataset_selection_request"
SELECTION_SCHEMA_VERSION: Final = 2
SOURCE_REFERENCE_ORDER: Final = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)
SOURCE_REFERENCE_KEYS: Final = frozenset(SOURCE_REFERENCE_ORDER)
_REQUEST_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "source_task",
        "dataset_name",
        "dataset_view",
        "selection",
        "source_cases",
    }
)


def discover_source_collections(
    task: str,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """List source collections with actual case outcomes and design counts."""
    choices: list[dict[str, Any]] = []
    for campaign in catalog.list_campaigns(task, storage_root=storage_root):
        rows = catalog.list_campaign_cases(task, campaign["source_name"], storage_root=storage_root)
        outcomes = {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in ("successful", "failed", "pending")}
        choices.append({**campaign, "outcomes": outcomes, "cases": rows})
    return tuple(choices)


def _reference_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Project one successful catalog row to its exact authored reference."""
    if row.get("outcome") != "successful":
        msg = "Only successful source cases may enter a Dataset selection."
        raise ValueError(msg)
    reference = {key: row.get(key) for key in SOURCE_REFERENCE_ORDER}
    if any(not isinstance(value, str) or not value for value in reference.values()):
        msg = "Successful source catalog entry lacks an exact case identity."
        raise ValueError(msg)
    return {key: str(value) for key, value in reference.items()}


def select_exact_cases(
    task: str,
    picks: Sequence[Mapping[str, str]],
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, str], ...]:
    """Resolve ordered campaign/material/case picks without hiding failures."""
    if not picks:
        msg = "Exact Dataset selection requires at least one source case."
        raise ValueError(msg)
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pick in picks:
        if not isinstance(pick, Mapping) or set(pick) != {"source_name", "material_family", "case_id"}:
            msg = "Exact source pick requires source_name, material_family, and case_id."
            raise ValueError(msg)
        campaign = common.paths.validate_logical_name(pick["source_name"], label="source name")
        material = common.paths.validate_logical_name(pick["material_family"], label="source material")
        case_id = common.paths.validate_logical_name(pick["case_id"], label="source case")
        key = campaign, material, case_id
        if key in seen:
            msg = "Exact Dataset selection repeats one source case."
            raise ValueError(msg)
        seen.add(key)
        matches = [
            row
            for row in catalog.list_campaign_cases(task, campaign, storage_root=storage_root)
            if row["material_family"] == material and row["case_id"] == case_id
        ]
        if len(matches) != 1:
            msg = f"Exact source case {key!r} is missing or ambiguous."
            raise ValueError(msg)
        selected.append(_reference_from_row(matches[0]))
    return tuple(selected)


def select_ranked_cases(
    task: str,
    source_names: Sequence[str],
    target_counts: Mapping[str, int],
    *,
    root_seed: int,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, str], ...]:
    """Select deterministic successful case subsets and retain exact references."""
    names = tuple(source_names)
    if not names or len(names) != len(set(names)):
        msg = "Ranked Dataset selection requires distinct source campaigns."
        raise ValueError(msg)
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or not 0 <= root_seed < 2**32:
        msg = "Ranked Dataset selection root_seed must be uint32."
        raise ValueError(msg)
    if not target_counts or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in target_counts.values()):
        msg = "Ranked Dataset selection requires positive material target counts."
        raise ValueError(msg)
    pools: dict[str, list[dict[str, str]]] = {material: [] for material in target_counts}
    physical_seen: set[str] = set()
    for name in names:
        common.paths.validate_logical_name(name, label="source campaign")
        for row in catalog.list_campaign_cases(task, name, storage_root=storage_root):
            if row["outcome"] != "successful" or row["material_family"] not in pools:
                continue
            reference = _reference_from_row(row)
            if reference["case_input_id"] in physical_seen:
                msg = "Ranked source pool contains a duplicate physical case input; select exact cases."
                raise ValueError(msg)
            physical_seen.add(reference["case_input_id"])
            pools[reference["material_family"]].append(reference)
    selected: list[dict[str, str]] = []
    for material, count in target_counts.items():
        if len(pools[material]) < count:
            msg = f"Ranked Dataset selection lacks {count} successful {material} cases."
            raise ValueError(msg)
        ranked = sorted(
            pools[material],
            key=lambda reference: (
                hashlib.sha256(f"dataset-selection-v1|{root_seed}|{task}|{material}|{reference['case_input_id']}".encode()).hexdigest(),
                reference["campaign_name"],
                reference["case_id"],
            ),
        )
        selected.extend(ranked[:count])
    return tuple(selected)


def validate_selection_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact, role-neutral Dataset publication request."""
    expected_keys = _REQUEST_KEYS
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        msg = "Dataset selection request keys do not match the current schema."
        raise ValueError(msg)
    if value["schema_kind"] != SELECTION_SCHEMA_KIND or value["schema_version"] != SELECTION_SCHEMA_VERSION:
        msg = "Dataset selection request uses an unsupported contract."
        raise ValueError(msg)
    task = common.paths.validate_logical_name(value["source_task"], label="source_task")
    dataset_name = common.paths.validate_logical_name(value["dataset_name"], label="dataset_name")
    view = value["dataset_view"]
    views.validate_view_for_profile(view, task)
    selection = value["selection"]
    if not isinstance(selection, Mapping) or selection.get("kind") not in {"exact", "seed_ranked"}:
        msg = "Dataset selection method must be exact or seed_ranked."
        raise ValueError(msg)
    if selection["kind"] == "exact" and set(selection) != {"kind"}:
        msg = "Exact Dataset selection has unexpected method fields."
        raise ValueError(msg)
    if selection["kind"] == "seed_ranked":
        _ranked_request_inputs(selection)
    sources = value["source_cases"]
    if not isinstance(sources, list) or not sources:
        msg = "Dataset selection requires a non-empty ordered source_cases list."
        raise ValueError(msg)
    source_keys: set[tuple[str, str, str]] = set()
    physical_ids: set[str] = set()
    admitted_sources: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != SOURCE_REFERENCE_KEYS:
            msg = "Dataset source reference keys do not match the current exact contract."
            raise ValueError(msg)
        reference = {key: source[key] for key in SOURCE_REFERENCE_ORDER}
        if any(not isinstance(item, str) or not item for item in reference.values()):
            msg = "Dataset source reference fields must be non-empty strings."
            raise ValueError(msg)
        source_key = reference["campaign_name"], reference["material_family"], reference["case_id"]
        if source_key in source_keys or reference["case_input_id"] in physical_ids:
            msg = "Dataset selection repeats one source case or physical case input."
            raise ValueError(msg)
        source_keys.add(source_key)
        physical_ids.add(reference["case_input_id"])
        admitted_sources.append(reference)
    return {
        "schema_kind": SELECTION_SCHEMA_KIND,
        "schema_version": value["schema_version"],
        "source_task": task,
        "dataset_name": dataset_name,
        "dataset_view": view,
        "selection": copy.deepcopy(dict(selection)),
        "source_cases": admitted_sources,
    }


def _ranked_request_inputs(method: Mapping[str, Any]) -> tuple[list[str], Mapping[str, int], int]:
    """Admit the declared ranked source pool, target counts, and seed."""
    if (
        set(method) != {"kind", "root_seed", "source_names", "target_counts"}
        or method.get("kind") != "seed_ranked"
        or isinstance(method["root_seed"], bool)
        or not isinstance(method["root_seed"], int)
        or not 0 <= method["root_seed"] < 2**32
    ):
        msg = "Seed-ranked Dataset selection method evidence is invalid."
        raise ValueError(msg)
    names = method["source_names"]
    counts = method["target_counts"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        msg = "Seed-ranked source campaign names must be a non-empty list."
        raise ValueError(msg)
    if len(names) != len(set(names)):
        msg = "Seed-ranked source campaign names must be distinct."
        raise ValueError(msg)
    for name in names:
        common.paths.validate_logical_name(name, label="source campaign")
    if not isinstance(counts, Mapping) or not counts or any(not isinstance(material, str) for material in counts):
        msg = "Seed-ranked material counts must be a non-empty mapping."
        raise ValueError(msg)
    for material, count in counts.items():
        common.paths.validate_logical_name(material, label="source material")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            msg = "Seed-ranked material target counts must be positive integers."
            raise ValueError(msg)
    return names, counts, method["root_seed"]


def validate_ranked_publication_claim(
    request: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> None:
    """Require a new Dataset's exact cases to match its declared ranked source pool."""
    method = request["selection"]
    if method["kind"] != "seed_ranked":
        return
    names, counts, seed = _ranked_request_inputs(method)
    expected = list(select_ranked_cases(request["source_task"], names, counts, root_seed=seed, storage_root=storage_root))
    if request["source_cases"] != expected:
        msg = "Seed-ranked Dataset source cases disagree with the declared campaign pool, counts, seed, or order."
        raise ValueError(msg)


def materialize_selection_request(
    value: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve a seeded authored selector once to exact immutable case references."""
    if not isinstance(value, Mapping):
        msg = "Dataset selection request must be a mapping."
        raise TypeError(msg)
    if value.get("source_cases") != []:
        return validate_selection_request(value)
    method = value.get("selection")
    if not isinstance(method, Mapping) or method.get("kind") != "seed_ranked":
        msg = "Only a seeded source selector may omit exact source cases."
        raise ValueError(msg)
    names, counts, seed = _ranked_request_inputs(method)
    expanded = dict(value)
    expanded["source_cases"] = list(
        select_ranked_cases(
            value["source_task"],
            names,
            counts,
            root_seed=seed,
            storage_root=storage_root,
        )
    )
    return validate_selection_request(expanded)
