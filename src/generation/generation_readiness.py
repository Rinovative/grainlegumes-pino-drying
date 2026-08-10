"""
===============================================================================
generation_readiness.py
===============================================================================
Report fail-closed VP2 scientific, mapping, runtime, and launch readiness.
Responsibilities:
  - Resolve every canonical campaign without requiring production launch values
  - Enumerate exact missing primary counts, seeds, memberships, and mappings
  - Emit the binding seven-line production-readiness status vocabulary
Design principles:
  - Static validity never substitutes for native COMSOL runtime evidence
  - Missing values are reported by repository-relative file and dotted key
  - Readiness inspection is read-only and starts no production work
This module does NOT:
  - Choose counts or seeds, confirm COMSOL mappings, or create runtime receipts
  - Run static sentinels unless the caller explicitly requests them
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

from src import common

from . import generation_config as config_service
from . import generation_materials as materials
from . import generation_sentinels as sentinel_service
from . import generation_smoke as smoke_service

_STATUS_COMPLETE: Final = "COMPLETE"
_STATUS_INCOMPLETE: Final = "INCOMPLETE"
_STATUS_PENDING: Final = "PENDING"
_STATUS_BLOCKED: Final = "BLOCKED"


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


def _primary_missing(path: Path) -> list[str]:
    """Return absent explicit production count, seed, or membership keys."""
    raw = _yaml(path)
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
    membership = raw.get("membership")
    if not isinstance(membership, dict):
        missing.append(f"{prefix}:membership")
        return missing
    if membership.get("seed") is None:
        missing.append(f"{prefix}:membership.seed")
    per_material = membership.get("per_seen_material")
    if not isinstance(per_material, dict):
        missing.append(f"{prefix}:membership.per_seen_material")
    else:
        for split, count in per_material.items():
            if count is None:
                missing.append(f"{prefix}:membership.per_seen_material.{split}")
    return missing


def _profile_mapping_states(
    campaign_path: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Return required and optional typed mapping states separately."""
    campaign = _yaml(campaign_path)
    configured = Path(campaign["profile_config"])
    profile_path = configured if configured.is_absolute() else common.paths.get_project_root() / configured
    profile = _yaml(profile_path)
    prefix = _relative(profile_path)
    probe_required: list[str] = []
    declared_unverified: list[str] = []
    optional_probe_required: list[str] = []
    for index, export in enumerate(profile["exports"]):
        optional = export["role"] == "exact_stop_diagnostics"
        probe_destination = optional_probe_required if optional else probe_required
        source = export["source"]
        source_key = f"{prefix}:exports[{index}].source"
        if source["state"] == "mapping_probe_required":
            probe_destination.append(source_key)
        elif source["state"] == "declared_unverified" and not optional:
            declared_unverified.append(source_key)
        for logical, mapping in export["columns"].items():
            key = f"{prefix}:exports[{index}].columns.{logical}"
            if mapping["state"] == "mapping_probe_required":
                probe_destination.append(key)
            elif mapping["state"] == "declared_unverified" and not optional:
                declared_unverified.append(key)
    return probe_required, declared_unverified, optional_probe_required


def campaign_unresolved_gates(path: Path | str) -> dict[str, list[str]]:
    """Return exact authored-value and profile-mapping gates for one campaign."""
    campaign_path = Path(path).expanduser().resolve()
    raw = _yaml(campaign_path)
    primary_missing = _primary_missing(campaign_path) if raw.get("campaign_purpose") == "family_generalization" else []
    probe_required, declared_unverified, optional_probe_required = _profile_mapping_states(campaign_path)
    return {
        "missing_production_keys": sorted(primary_missing),
        "missing_profile_mapping_keys": sorted(probe_required),
        "declared_unverified_profile_mapping_keys": sorted(declared_unverified),
        "optional_profile_mapping_keys": sorted(optional_probe_required),
    }


def build_readiness_report(
    steady_primary_path: Path | str,
    transient_primary_path: Path | str,
    *,
    run_static_sentinels: bool = False,
    real_runtime_receipt: Path | str | None = None,
) -> dict[str, Any]:
    """Build one read-only, exact-key production-readiness report."""
    steady_path = Path(steady_primary_path).expanduser().resolve()
    transient_path = Path(transient_primary_path).expanduser().resolve()
    campaigns = [config_service.load_campaign_config(path, require_executable=False) for path in (steady_path, transient_path)]
    expected_roles = {
        "seen": ("lentil", "chickpea", "kidney_bean"),
        "near_family_ood": ("field_pea",),
        "far_family_ood": ("rapeseed",),
        "extreme_family_ood": ("sunflower_seed",),
    }
    inventory = tuple(family for role in expected_roles for family in campaigns[0].material_roles[role])
    if inventory != materials.MATERIAL_FAMILIES or any(campaign.material_roles != expected_roles for campaign in campaigns):
        message = "Readiness family inventory or roles do not match the six-family VP2 contract."
        raise ValueError(message)
    steady_gates = campaign_unresolved_gates(steady_path)
    transient_gates = campaign_unresolved_gates(transient_path)
    primary_missing = sorted(set(steady_gates["missing_production_keys"] + transient_gates["missing_production_keys"]))
    mapping_missing = sorted(set(steady_gates["missing_profile_mapping_keys"] + transient_gates["missing_profile_mapping_keys"]))
    mapping_declared_unverified = sorted(
        set(steady_gates["declared_unverified_profile_mapping_keys"] + transient_gates["declared_unverified_profile_mapping_keys"])
    )
    optional_mapping_missing = sorted(set(steady_gates["optional_profile_mapping_keys"] + transient_gates["optional_profile_mapping_keys"]))
    static_report = None
    if run_static_sentinels:
        static_report = sentinel_service.run_static_sentinels(steady_path, transient_path)
    receipt_path = None if real_runtime_receipt is None else Path(real_runtime_receipt).expanduser().resolve()
    real_complete = False
    if receipt_path is not None:
        smoke_service.validate_real_smoke_receipt(receipt_path)
        real_complete = True
    scientific_complete = True
    ownership_complete = True
    documentation_complete = True
    primary_complete = not primary_missing
    mapping_complete = not mapping_missing and (not mapping_declared_unverified or real_complete)
    static_complete = static_report is not None and static_report["status"] == "pass"
    static_status = _STATUS_COMPLETE if static_complete else _STATUS_BLOCKED if static_report is not None else _STATUS_PENDING
    production_ready = all(
        (
            scientific_complete,
            ownership_complete,
            documentation_complete,
            primary_complete,
            mapping_complete,
            static_complete,
            real_complete,
        )
    )
    return {
        "schema_kind": "vp2_production_readiness",
        "schema_version": 1,
        "decision_sha256": materials.VP2_DECISION_SHA256,
        "canonical_family_inventory": list(materials.MATERIAL_FAMILIES),
        "canonical_family_roles": {name: list(values) for name, values in expected_roles.items()},
        "campaign_ids": [campaign.campaign_id for campaign in campaigns],
        "static_scientific_integration_complete": scientific_complete,
        "config_ownership_consolidation_complete": ownership_complete,
        "documentation_consolidation_complete": documentation_complete,
        "static_generator_sentinels_complete": static_complete,
        "real_runtime_validation_complete": real_complete,
        "primary_production_config_complete": primary_complete,
        "profile_mapping_complete": mapping_complete,
        "production_ready_for_user_launch": production_ready,
        "missing_primary_keys": primary_missing,
        "missing_profile_mapping_keys": mapping_missing,
        "declared_unverified_profile_mapping_keys": mapping_declared_unverified,
        "optional_profile_mapping_keys": optional_mapping_missing,
        "real_runtime_receipt": None if receipt_path is None else str(receipt_path),
        "static_sentinel_report": static_report,
        "status_lines": [
            "STATIC_SCIENTIFIC_INTEGRATION_COMPLETE",
            "CONFIG_OWNERSHIP_CONSOLIDATION_COMPLETE",
            "DOCUMENTATION_CONSOLIDATION_COMPLETE",
            f"STATIC_GENERATOR_SENTINELS_{static_status}",
            f"PRIMARY_PRODUCTION_CONFIG_{_STATUS_COMPLETE if primary_complete else _STATUS_INCOMPLETE}",
            f"REAL_RUNTIME_VALIDATION_{_STATUS_COMPLETE if real_complete else _STATUS_PENDING}",
            f"PRODUCTION_READY_FOR_USER_LAUNCH_{_STATUS_COMPLETE if production_ready else _STATUS_BLOCKED}",
        ],
        "production_solve_started": False,
    }
