"""Report the unresolved authored-value and mapping gates of one campaign configuration.

Responsibilities:
  - Enumerate missing configured design counts, seeds, and required profile export mappings
  - Report each gap by repository-relative file and dotted key

Design principles:
  - Static validity never substitutes for native COMSOL runtime evidence
  - Gate inspection is read-only and starts no production work

This module does NOT:
  - Choose counts or seeds, confirm COMSOL mappings, or create runtime receipts
  - Resolve campaigns or run static sentinels
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src import common

from .contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Mapping


def _yaml(path: Path) -> dict[str, Any]:
    """Load one mapping for dotted-key readiness inspection."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"Readiness configuration must be a YAML mapping: {path}."
        raise TypeError(message)
    return value


def _relative(path: Path) -> str:
    """Return one stable repository-relative configuration path."""
    return path.resolve().relative_to(common.paths.get_project_root().resolve()).as_posix()


def _primary_missing(path: Path, *, authored_value: Mapping[str, Any] | None = None) -> list[str]:
    """Return absent explicit source design count or seed keys."""
    raw = _yaml(path) if authored_value is None else authored_value
    prefix = _relative(path)
    missing: list[str] = []
    sampling = raw.get("sampling")
    if not isinstance(sampling, dict):
        return [f"{prefix}:sampling"]
    if sampling.get("seed_base") is None:
        missing.append(f"{prefix}:sampling.seed_base")
    counts = sampling.get("counts")
    if not isinstance(counts, dict):
        missing.append(f"{prefix}:sampling.counts")
    else:
        for regime, by_material in counts.items():
            if not isinstance(by_material, dict):
                missing.append(f"{prefix}:sampling.counts.{regime}")
                continue
            for family, count in by_material.items():
                if count is None:
                    missing.append(f"{prefix}:sampling.counts.{regime}.{family}")
    return missing


def _profile_mapping_missing(campaign_path: Path, *, authored_value: Mapping[str, Any] | None = None) -> list[str]:
    """Return structurally incomplete required export declarations."""
    campaign = _yaml(campaign_path) if authored_value is None else authored_value
    configured = Path(campaign["profile_config"])
    profile_path = configured if configured.is_absolute() else common.paths.get_project_root() / configured
    profile = _yaml(profile_path)
    prefix = _relative(profile_path)
    profile_spec = profiles.resolve_profile(str(profile["simulation_profile"]))
    missing: list[str] = []
    for index, export in enumerate(profile["exports"]):
        role_spec = profile_spec.export_role(str(export["role"]))
        if not role_spec.required:
            continue
        if not isinstance(export.get("source"), str):
            missing.append(f"{prefix}:exports[{index}].source")
        columns = export.get("columns")
        if not isinstance(columns, dict):
            missing.append(f"{prefix}:exports[{index}].columns")
            continue
        missing.extend(
            f"{prefix}:exports[{index}].columns.{logical}" for logical in role_spec.logical_fields if not isinstance(columns.get(logical), str)
        )
    return missing


def campaign_unresolved_gates(
    path: Path | str,
    *,
    authored_value: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Return exact authored-value and profile-mapping gates for one campaign."""
    campaign_path = Path(path).expanduser().resolve()
    raw = _yaml(campaign_path) if authored_value is None else authored_value
    primary_missing = _primary_missing(campaign_path, authored_value=raw) if raw.get("schema_kind") == "generation_source_campaign" else []
    mapping_missing = _profile_mapping_missing(campaign_path, authored_value=raw)
    return {
        "missing_production_keys": sorted(primary_missing),
        "missing_profile_mapping_keys": sorted(mapping_missing),
    }
