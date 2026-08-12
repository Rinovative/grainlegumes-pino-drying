"""
===============================================================================
generation_contracts_mapping.py
===============================================================================
Define the semantic identity of one resolved Generation export-mapping contract.
Responsibilities:
  - Select every resolved semantic input that technical smoke verifies
  - Canonicalize ordered export roles, source declarations, fields, and units
  - Derive the sole SHA-256 identity used to match runtime smoke evidence
Design principles:
  - Mapping identity excludes unrelated profile metadata and source commits
  - Wide transient identity includes base fields, never observed time suffixes
  - Incomplete executable contracts fail before an identity is produced
This module does NOT:
  - Load YAML, inspect runtime files, execute COMSOL, or discover smoke evidence
  - Decide whether runtime evidence is current
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from src import common

MAPPING_CONTRACT_SCHEMA_KIND: Final = "generation_mapping_contract"
MAPPING_CONTRACT_SCHEMA_VERSION: Final = 1


def mapping_contract_payload(
    simulation_profile: str,
    output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return the canonical semantic export-mapping payload.

    Parameters
    ----------
    simulation_profile : str
        Exact resolved Generation simulation-profile identity.
    output_contract : Mapping[str, Any]
        Complete resolved output contract from the authoritative config loader.

    Returns
    -------
    dict[str, Any]
        JSON-canonicalizable mapping semantics in configured export order.

    Raises
    ------
    TypeError
        If required mapping semantics have invalid container types.
    ValueError
        If any required mapping declaration is incomplete or inconsistent.

    """
    exports_root = output_contract.get("exports_root")
    exports = output_contract.get("exports")
    if not isinstance(simulation_profile, str) or not simulation_profile:
        message = "Mapping contract requires a non-empty simulation_profile."
        raise ValueError(message)
    if not isinstance(exports_root, str) or not exports_root:
        message = "Mapping contract requires a non-empty exports_root."
        raise ValueError(message)
    if not isinstance(exports, list):
        message = "Mapping contract exports must be an ordered list."
        raise TypeError(message)
    canonical_exports: list[dict[str, Any]] = []
    for index, export in enumerate(exports):
        if not isinstance(export, Mapping):
            message = f"Mapping contract export {index} must be a mapping."
            raise TypeError(message)
        role = export.get("role")
        required = export.get("required")
        source_pattern = export.get("pattern")
        allow_multiple = export.get("allow_multiple")
        delimiter = export.get("delimiter")
        temporal_kind = export.get("temporal_kind")
        columns = export.get("columns")
        units = export.get("units")
        if not isinstance(role, str) or not role:
            message = f"Mapping contract export {index} has no role."
            raise ValueError(message)
        if not isinstance(required, bool) or not isinstance(allow_multiple, bool):
            message = f"Mapping contract export {role!r} has invalid required or allow_multiple semantics."
            raise TypeError(message)
        if not isinstance(source_pattern, str) or not source_pattern:
            message = f"Mapping contract export {role!r} has no complete source pattern."
            raise ValueError(message)
        if not isinstance(delimiter, str) or not delimiter or not isinstance(temporal_kind, str) or not temporal_kind:
            message = f"Mapping contract export {role!r} has invalid delimiter or temporal kind."
            raise ValueError(message)
        if not isinstance(columns, Mapping) or not isinstance(units, Mapping):
            message = f"Mapping contract export {role!r} columns and units must be mappings."
            raise TypeError(message)
        if tuple(columns) != tuple(units):
            message = f"Mapping contract export {role!r} columns and units disagree."
            raise ValueError(message)
        fields: list[dict[str, str]] = []
        for logical_name, source_header in columns.items():
            unit = units[logical_name]
            if not isinstance(logical_name, str) or not logical_name:
                message = f"Mapping contract export {role!r} has an invalid logical field name."
                raise ValueError(message)
            if not isinstance(source_header, str) or not source_header:
                message = f"Mapping contract export {role!r} field {logical_name!r} has no source header."
                raise ValueError(message)
            if not isinstance(unit, str) or not unit:
                message = f"Mapping contract export {role!r} field {logical_name!r} has no unit."
                raise ValueError(message)
            fields.append(
                {
                    "logical_name": logical_name,
                    "source_header": source_header,
                    "unit": unit,
                }
            )
        canonical_exports.append(
            {
                "role": role,
                "required": required,
                "source_pattern": source_pattern,
                "allow_multiple": allow_multiple,
                "delimiter": delimiter,
                "temporal_kind": temporal_kind,
                "fields": fields,
            }
        )
    return {
        "schema_kind": MAPPING_CONTRACT_SCHEMA_KIND,
        "schema_version": MAPPING_CONTRACT_SCHEMA_VERSION,
        "simulation_profile": simulation_profile,
        "exports_root": exports_root,
        "exports": canonical_exports,
    }


def mapping_contract_sha256(
    simulation_profile: str,
    output_contract: Mapping[str, Any],
) -> str:
    """Return the SHA-256 identity of resolved export-mapping semantics."""
    return common.serialization.canonical_json_sha256(mapping_contract_payload(simulation_profile, output_contract))
