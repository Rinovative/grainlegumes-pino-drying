"""
===============================================================================
generation_provenance.py
===============================================================================
Validate central scientific sources and effective parameter provenance.
Responsibilities:
  - Validate one unique source register with plain canonical locators
  - Validate supplied status, derivation, confidence, and validity structures
  - Expand source references for resolved inspection without duplicating citations
Design principles:
  - Scientific interpretation is supplied by the validated handoff package
  - Missing derivation inputs remain missing rather than being reconstructed
  - Atomic-record components inherit their complete record provenance
This module does NOT:
  - Search for evidence, invent derivations, or change scientific values
  - Decide material roles, sampling supports, or parameter-OOD applicability
===============================================================================
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

EVIDENCE_STATUSES: Final = (
    "literature_direct",
    "literature_fit",
    "literature_transfer",
    "literature_convention_conversion",
    "official_industry_target",
    "project_baseline",
    "engineering_estimate",
    "engineering_conversion",
    "engineering_inversion",
    "calibration_prior",
    "hierarchical_engineering_prior",
    "synthetic_design",
    "derived",
    "coupled_record",
)
DERIVATION_ORIGIN: Final = "supplied_by_handoff"
DERIVATION_VERIFICATIONS: Final = ("declared_only", "mathematically_reproduced")
SOURCE_TYPES: Final = (
    "journal_article",
    "project_report",
    "model_report",
    "official_guidance",
    "government_guidance",
    "book_or_manual",
    "institutional_record",
)
VALIDITY_KEYS: Final = frozenset(
    {
        "material_scope_ref",
        "product_form",
        "temperature",
        "humidity_or_aw",
        "moisture",
        "moisture_basis",
        "packing_or_flow_regime",
        "equation_or_method",
        "transfer_limit",
        "supplied_scope",
    }
)
_DERIVATION_KEYS: Final = frozenset(
    {
        "kind",
        "origin",
        "verification",
        "description",
        "inputs",
        "formula_or_method",
        "assumptions",
        "supplied_status",
    }
)
_SOURCE_KEYS: Final = frozenset(
    {
        "source_key",
        "citation",
        "identifier",
        "canonical_locator",
        "alternate_locators",
        "source_type",
    }
)
_PROVENANCE_KEYS: Final = frozenset(
    {
        "source_refs",
        "status",
        "derivation",
        "confidence",
        "validity",
        "notes",
    }
)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one isolated mapping with string keys."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _exact_keys(value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    """Require all declared keys and reject unknown provenance structure."""
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise ValueError(message)


def _text(value: Any, *, label: str) -> str:
    """Return safe non-empty single-line text."""
    if not isinstance(value, str) or not value or value.strip() != value or any(character in value for character in ("\x00", "\n", "\r")):
        message = f"{label} must be non-empty single-line text."
        raise ValueError(message)
    return value


def _text_sequence(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    """Return one unique ordered sequence of safe text values."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        message = f"{label} must be an ordered text sequence."
        raise TypeError(message)
    result = [_text(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if (not result and not allow_empty) or len(result) != len(set(result)):
        message = f"{label} must contain {'zero or more' if allow_empty else 'one or more'} unique values."
        raise ValueError(message)
    return result


def bind_decision_source(
    value: Any,
    decision_source: Mapping[str, str],
) -> dict[str, Any]:
    """Expand the registry-owned decision identity into its provenance citation."""
    config = _mapping(value, label="generation source registry")
    decision = _mapping(decision_source, label="generation decision source")
    _exact_keys(
        decision,
        required={"artifact", "schema_version", "sha256"},
        optional=set(),
        label="generation decision source",
    )
    if not all(isinstance(item, str) and item for item in decision.values()):
        message = "Generation decision-source values must be non-empty text."
        raise TypeError(message)
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list):
        message = "generation source registry.sources must be a list."
        raise TypeError(message)
    matches = [(index, raw) for index, raw in enumerate(raw_sources) if isinstance(raw, Mapping) and raw.get("source_key") == "vp2_decision_contract"]
    if len(matches) != 1:
        message = "Generation source registry must declare one vp2_decision_contract citation stub."
        raise ValueError(message)
    index, raw = matches[0]
    stub = _mapping(raw, label="generation source registry vp2_decision_contract")
    _exact_keys(
        stub,
        required={"source_key", "decision_date", "alternate_locators", "source_type"},
        optional=set(),
        label="generation source registry vp2_decision_contract",
    )
    decision_date = _text(
        stub["decision_date"],
        label="generation source registry vp2_decision_contract.decision_date",
    )
    raw_sources[index] = {
        "source_key": "vp2_decision_contract",
        "citation": f"VP2 Parameter Decisions, schema {decision['schema_version']} ({decision_date}).",
        "identifier": f"sha256:{decision['sha256']}",
        "canonical_locator": f"artifact:{decision['artifact']}",
        "alternate_locators": copy.deepcopy(stub["alternate_locators"]),
        "source_type": stub["source_type"],
    }
    config["decision_source"] = copy.deepcopy(decision)
    return config


def validate_source_registry(
    value: Any,
    *,
    decision_validator: Callable[[Any], Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Validate one central source registry and return it keyed by source key."""
    config = _mapping(value, label="generation source registry")
    _exact_keys(
        config,
        required={"schema_kind", "schema_version", "decision_source", "sources"},
        optional=set(),
        label="generation source registry",
    )
    if config["schema_kind"] != "generation_sources" or config["schema_version"] != 1:
        message = "Unsupported generation source-registry schema."
        raise ValueError(message)
    decision_validator(config["decision_source"])
    raw_sources = config["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        message = "generation source registry.sources must be a non-empty list."
        raise TypeError(message)
    sources: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_sources):
        label = f"generation source registry.sources[{index}]"
        source = _mapping(raw, label=label)
        _exact_keys(source, required=set(_SOURCE_KEYS), optional=set(), label=label)
        source_key = _text(source["source_key"], label=f"{label}.source_key")
        if source_key in sources:
            message = f"Duplicate scientific source key {source_key!r}."
            raise ValueError(message)
        for name in ("citation", "identifier", "canonical_locator"):
            source[name] = _text(source[name], label=f"{label}.{name}")
        source["alternate_locators"] = _text_sequence(
            source["alternate_locators"],
            label=f"{label}.alternate_locators",
            allow_empty=True,
        )
        if source["source_type"] not in SOURCE_TYPES:
            message = f"{label}.source_type must be one of {list(SOURCE_TYPES)}."
            raise ValueError(message)
        if any(locator.startswith("[") or "](" in locator for locator in [source["canonical_locator"], *source["alternate_locators"]]):
            message = f"{label} locators must be plain strings, not Markdown links."
            raise ValueError(message)
        sources[source_key] = source
    return sources


def validate_provenance(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Validate one handoff-supplied scientific provenance chain."""
    provenance = _mapping(value, label=label)
    _exact_keys(
        provenance,
        required={"source_refs", "status", "derivation", "confidence", "validity"},
        optional={"notes"},
        label=label,
    )
    source_refs = _text_sequence(provenance["source_refs"], label=f"{label}.source_refs")
    unknown_sources = sorted(set(source_refs).difference(sources))
    if unknown_sources:
        message = f"{label}.source_refs contains unknown keys {unknown_sources}."
        raise ValueError(message)
    status = provenance["status"]
    if status not in EVIDENCE_STATUSES:
        message = f"{label}.status must be one of {list(EVIDENCE_STATUSES)}, got {status!r}."
        raise ValueError(message)
    derivation = _mapping(provenance["derivation"], label=f"{label}.derivation")
    _exact_keys(
        derivation,
        required={"kind", "origin", "verification"},
        optional=set(_DERIVATION_KEYS).difference({"kind", "origin", "verification"}),
        label=f"{label}.derivation",
    )
    derivation["kind"] = _text(derivation["kind"], label=f"{label}.derivation.kind")
    if derivation["origin"] != DERIVATION_ORIGIN:
        message = f"{label}.derivation.origin must be {DERIVATION_ORIGIN!r}."
        raise ValueError(message)
    if derivation["verification"] not in DERIVATION_VERIFICATIONS:
        message = f"{label}.derivation.verification must be one of {list(DERIVATION_VERIFICATIONS)}."
        raise ValueError(message)
    for name in ("description", "formula_or_method", "supplied_status"):
        if name in derivation:
            derivation[name] = _text(derivation[name], label=f"{label}.derivation.{name}")
    for name in ("inputs", "assumptions"):
        if name in derivation:
            derivation[name] = _text_sequence(
                derivation[name],
                label=f"{label}.derivation.{name}",
                allow_empty=True,
            )
    confidence = _text(provenance["confidence"], label=f"{label}.confidence")
    validity = _mapping(provenance["validity"], label=f"{label}.validity")
    unknown_validity = sorted(set(validity).difference(VALIDITY_KEYS))
    if unknown_validity:
        message = f"{label}.validity contains unknown structured fields {unknown_validity}."
        raise ValueError(message)
    for name, item in validity.items():
        if item is None:
            message = f"{label}.validity.{name} must be omitted rather than null."
            raise ValueError(message)
        if isinstance(item, str):
            validity[name] = _text(item, label=f"{label}.validity.{name}")
        elif isinstance(item, Mapping):
            nested = _mapping(item, label=f"{label}.validity.{name}")
            if not nested or any(value is None for value in nested.values()):
                message = f"{label}.validity.{name} must be a non-empty mapping without null values."
                raise ValueError(message)
            validity[name] = nested
        elif isinstance(item, list):
            validity[name] = _text_sequence(item, label=f"{label}.validity.{name}")
        else:
            message = f"{label}.validity.{name} has unsupported type {type(item).__name__}."
            raise TypeError(message)
    result = {
        "source_refs": source_refs,
        "status": status,
        "derivation": derivation,
        "confidence": confidence,
        "validity": validity,
    }
    if "notes" in provenance:
        result["notes"] = _text(provenance["notes"], label=f"{label}.notes")
    return result


def resolve_provenance(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Return validated provenance with complete referenced source records."""
    provenance = validate_provenance(value, sources=sources, label=label)
    provenance["sources"] = [copy.deepcopy(dict(sources[source_ref])) for source_ref in provenance["source_refs"]]
    return provenance


def source_citations(sources: Mapping[str, Mapping[str, Any]]) -> Mapping[str, str]:
    """Return a read-only source-key to citation inspection view."""
    return MappingProxyType({key: str(source["citation"]) for key, source in sources.items()})
