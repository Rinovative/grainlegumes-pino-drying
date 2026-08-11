"""
===============================================================================
generation_contracts_provenance.py
===============================================================================
Validate central scientific sources and effective parameter provenance.
Responsibilities:
  - Validate one unique source register with plain canonical locators
  - Validate compact evidence, method, verification, and applicability records
  - Expand source references for resolved inspection without duplicating citations
Design principles:
  - Scientific interpretation is explicit in canonical provenance records
  - Optional detail is admitted only when it has a distinct scientific purpose
  - Atomic-record components inherit their complete record provenance
This module does NOT:
  - Search for evidence, invent derivations, or change scientific values
  - Decide material roles, sampling supports, or parameter-OOD applicability
===============================================================================
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

EVIDENCE_CLASSES: Final = (
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
REPRODUCED_VERIFICATION: Final = "mathematically_reproduced"
SOURCE_REQUIRED_EVIDENCE_CLASSES: Final = frozenset(
    {
        "coupled_record",
        "literature_convention_conversion",
        "literature_direct",
        "literature_fit",
        "literature_transfer",
        "official_industry_target",
    }
)
SOURCE_TYPES: Final = (
    "journal_article",
    "project_report",
    "model_report",
    "official_guidance",
    "government_guidance",
    "book_or_manual",
    "institutional_record",
)
APPLICABILITY_KEYS: Final = frozenset(
    {
        "evidence_scope",
        "general",
        "limitation",
        "moisture_basis",
        "packing",
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


def validate_source_registry(value: Any) -> dict[str, dict[str, Any]]:
    """Validate one central source registry and return it keyed by source key."""
    config = _mapping(value, label="generation source registry")
    _exact_keys(
        config,
        required={"schema_kind", "schema_version", "sources"},
        optional=set(),
        label="generation source registry",
    )
    if config["schema_kind"] != "generation_sources" or config["schema_version"] != 1:
        message = "Unsupported generation source-registry schema."
        raise ValueError(message)
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
    """Validate one compact scientific provenance record."""
    provenance = _mapping(value, label=label)
    _exact_keys(
        provenance,
        required={"evidence", "source_refs"},
        optional={"method", "verification", "applicability", "note"},
        label=label,
    )
    evidence = provenance["evidence"]
    if evidence not in EVIDENCE_CLASSES:
        message = f"{label}.evidence must be one of {list(EVIDENCE_CLASSES)}, got {evidence!r}."
        raise ValueError(message)
    source_refs = _text_sequence(
        provenance["source_refs"],
        label=f"{label}.source_refs",
        allow_empty=True,
    )
    unknown_sources = sorted(set(source_refs).difference(sources))
    if unknown_sources:
        message = f"{label}.source_refs contains unknown keys {unknown_sources}."
        raise ValueError(message)
    if not source_refs and evidence in SOURCE_REQUIRED_EVIDENCE_CLASSES:
        message = f"{label}.source_refs must identify evidence for {evidence!r}."
        raise ValueError(message)
    result: dict[str, Any] = {
        "evidence": evidence,
        "source_refs": source_refs,
    }
    if "method" in provenance:
        result["method"] = _text(provenance["method"], label=f"{label}.method")
    if "verification" in provenance:
        if provenance["verification"] != REPRODUCED_VERIFICATION:
            message = f"{label}.verification must be {REPRODUCED_VERIFICATION!r} when present."
            raise ValueError(message)
        result["verification"] = REPRODUCED_VERIFICATION
    if "applicability" in provenance:
        applicability = _mapping(provenance["applicability"], label=f"{label}.applicability")
        if not applicability:
            message = f"{label}.applicability must be omitted rather than empty."
            raise ValueError(message)
        unknown = sorted(set(applicability).difference(APPLICABILITY_KEYS))
        if unknown:
            message = f"{label}.applicability contains unknown fields {unknown}."
            raise ValueError(message)
        result["applicability"] = {name: _text(item, label=f"{label}.applicability.{name}") for name, item in applicability.items()}
    if "note" in provenance:
        result["note"] = _text(provenance["note"], label=f"{label}.note")
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
