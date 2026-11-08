"""Resolve exact source-case selections for immutable Dataset publication.

Responsibilities:
  - Discover source collections with successful and failed case counts
  - Resolve exact or design-ordered case choices into immutable source references
  - Validate one role-neutral authored Dataset selection request

Design principles:
  - Generation owns source admission and success or failure evidence
  - Design order is the canonical selection order: no seed influences which cases a Dataset takes or in which order
  - Selection order and original design provenance remain explicit
  - A Dataset request never assigns experiment evaluation roles

This module does NOT:
  - Create Generation campaigns or edit source cases
  - Publish Dataset payloads or assign train and evaluation membership
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.datasets.contracts import dataset_contracts_views as views
from src.generation.publication import generation_publication_catalog as catalog

if TYPE_CHECKING:
    from pathlib import Path

SELECTION_SCHEMA_KIND: Final = "dataset_selection_request"
SELECTION_SCHEMA_VERSION: Final = 1
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
_SINGLE_SOURCE_MESSAGE: Final = (
    "Design-order Dataset selection admits exactly one Source, because design order is defined within one Source; "
    "use an exact selection to combine several Sources."
)
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


def select_ordered_cases(
    task: str,
    source_names: Sequence[str],
    target_counts: Mapping[str, int],
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, str], ...]:
    """Take each material's first successful cases in design order and retain exact references.

    Design order lists the original cases by case index, then the Sobol replacements by
    candidate ordinal, which is the order Generation selects its own success targets in. That order
    is defined within one Source only, so exactly one Source is admitted; ``exact`` selection covers
    several. Generation reserves a replacement only while its success target exceeds the successful
    and pending candidates, so a Source never holds more successes of a material than its target,
    which equals the requested count in a Dataset plan. The first ``count`` successes are then the
    whole pool; a smaller pool raises ``ValueError``.
    """
    names = tuple(source_names)
    if len(names) != 1:
        raise ValueError(_SINGLE_SOURCE_MESSAGE)
    if not target_counts or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in target_counts.values()):
        msg = "Ordered Dataset selection requires positive material target counts."
        raise ValueError(msg)
    pools: dict[str, list[tuple[tuple[bool, int, str, str], dict[str, str]]]] = {material: [] for material in target_counts}
    physical_seen: set[str] = set()
    for name in names:
        common.paths.validate_logical_name(name, label="source campaign")
        for row in catalog.list_campaign_cases(task, name, storage_root=storage_root):
            if row["outcome"] != "successful" or row["material_family"] not in pools:
                continue
            reference = _reference_from_row(row)
            if reference["case_input_id"] in physical_seen:
                msg = "Ordered source pool contains a duplicate physical case input; select exact cases."
                raise ValueError(msg)
            physical_seen.add(reference["case_input_id"])
            order = (row["origin"] != "primary", int(row["candidate_position"]), reference["campaign_name"], reference["case_id"])
            pools[reference["material_family"]].append((order, reference))
    selected: list[dict[str, str]] = []
    for material, count in target_counts.items():
        if len(pools[material]) < count:
            msg = f"Ordered Dataset selection lacks {count} successful {material} cases."
            raise ValueError(msg)
        selected.extend(reference for _order, reference in sorted(pools[material], key=lambda entry: entry[0])[:count])
    return tuple(selected)


def validate_selection_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact, role-neutral Dataset publication request."""
    expected_keys = _REQUEST_KEYS
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        msg = "Dataset selection request keys do not match the current schema."
        raise ValueError(msg)
    if value["schema_kind"] != SELECTION_SCHEMA_KIND or not common.serialization.is_schema_version(value["schema_version"], SELECTION_SCHEMA_VERSION):
        msg = "Dataset selection request uses an unsupported contract."
        raise ValueError(msg)
    task = common.paths.validate_logical_name(value["source_task"], label="source_task")
    dataset_name = common.paths.validate_logical_name(value["dataset_name"], label="dataset_name")
    view = value["dataset_view"]
    views.validate_view_for_profile(view, task)
    selection = value["selection"]
    if not isinstance(selection, Mapping) or selection.get("kind") not in {"exact", "design_order"}:
        msg = "Dataset selection method must be exact or design_order."
        raise ValueError(msg)
    if selection["kind"] == "exact" and set(selection) != {"kind"}:
        msg = "Exact Dataset selection has unexpected method fields."
        raise ValueError(msg)
    ordered = _ordered_request_inputs(selection) if selection["kind"] == "design_order" else None
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
    if ordered is not None:
        (source_name,) = ordered[0]
        designs = {reference["campaign_name"] for reference in admitted_sources}
        # References carry the frozen design name, which a revised Source ``<design>__rev-<label>`` keeps.
        if len(designs) != 1 or not any(source_name == design or source_name.startswith(f"{design}__rev-") for design in designs):
            msg = "Design-order Dataset source cases reference campaigns other than the declared Source."
            raise ValueError(msg)
    return {
        "schema_kind": SELECTION_SCHEMA_KIND,
        "schema_version": value["schema_version"],
        "source_task": task,
        "dataset_name": dataset_name,
        "dataset_view": view,
        "selection": copy.deepcopy(dict(selection)),
        "source_cases": admitted_sources,
    }


def _ordered_request_inputs(method: Mapping[str, Any]) -> tuple[list[str], Mapping[str, int]]:
    """Admit the declared ordered source pool and target counts."""
    if set(method) != {"kind", "source_names", "target_counts"} or method.get("kind") != "design_order":
        msg = "Design-order Dataset selection method evidence is invalid."
        raise ValueError(msg)
    names = method["source_names"]
    counts = method["target_counts"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        msg = "Design-order source campaign names must be a non-empty list."
        raise ValueError(msg)
    if len(names) != 1:
        raise ValueError(_SINGLE_SOURCE_MESSAGE)
    for name in names:
        common.paths.validate_logical_name(name, label="source campaign")
    if not isinstance(counts, Mapping) or not counts or any(not isinstance(material, str) for material in counts):
        msg = "Design-order material counts must be a non-empty mapping."
        raise ValueError(msg)
    for material, count in counts.items():
        common.paths.validate_logical_name(material, label="source material")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            msg = "Design-order material target counts must be positive integers."
            raise ValueError(msg)
    return names, counts


def validate_ordered_publication_claim(
    request: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> None:
    """Require a new Dataset's exact cases to match its declared ordered source pool."""
    method = request["selection"]
    if method["kind"] != "design_order":
        return
    names, counts = _ordered_request_inputs(method)
    expected = list(select_ordered_cases(request["source_task"], names, counts, storage_root=storage_root))
    if request["source_cases"] != expected:
        msg = "Design-order Dataset source cases disagree with the declared campaign pool, counts, or order."
        raise ValueError(msg)


def materialize_selection_request(
    value: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve a design-order selector once to exact immutable case references."""
    if not isinstance(value, Mapping):
        msg = "Dataset selection request must be a mapping."
        raise TypeError(msg)
    if value.get("source_cases") != []:
        return validate_selection_request(value)
    method = value.get("selection")
    if not isinstance(method, Mapping) or method.get("kind") != "design_order":
        msg = "Only a design-order selector may omit exact source cases."
        raise ValueError(msg)
    names, counts = _ordered_request_inputs(method)
    expanded = dict(value)
    expanded["source_cases"] = list(select_ordered_cases(value["source_task"], names, counts, storage_root=storage_root))
    return validate_selection_request(expanded)
