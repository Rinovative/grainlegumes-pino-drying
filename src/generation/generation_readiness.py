"""
===============================================================================
generation_readiness.py
===============================================================================
Report fail-closed VP2 scientific, mapping, runtime, and launch readiness.
Responsibilities:
  - Resolve both configured primary campaigns without requiring launch values
  - Enumerate missing configured counts, seeds, memberships, and mappings
  - Emit seven evidence-derived production-readiness status lines
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

from . import generation_smoke as smoke_service
from .cases import generation_cases_config as config_service
from .contracts import generation_contracts_profiles as profiles
from .validation import generation_validation_sentinels as sentinel_service

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
) -> tuple[list[str], list[str]]:
    """Return unresolved required typed mapping states."""
    campaign = _yaml(campaign_path)
    configured = Path(campaign["profile_config"])
    profile_path = configured if configured.is_absolute() else common.paths.get_project_root() / configured
    profile = _yaml(profile_path)
    prefix = _relative(profile_path)
    probe_required: list[str] = []
    declared_unverified: list[str] = []
    for index, export in enumerate(profile["exports"]):
        source = export["source"]
        source_key = f"{prefix}:exports[{index}].source"
        if source["state"] == "mapping_probe_required":
            probe_required.append(source_key)
        elif source["state"] == "declared_unverified":
            declared_unverified.append(source_key)
        for logical, mapping in export["columns"].items():
            key = f"{prefix}:exports[{index}].columns.{logical}"
            if mapping["state"] == "mapping_probe_required":
                probe_required.append(key)
            elif mapping["state"] == "declared_unverified":
                declared_unverified.append(key)
    return probe_required, declared_unverified


def campaign_unresolved_gates(path: Path | str) -> dict[str, list[str]]:
    """Return exact authored-value and profile-mapping gates for one campaign."""
    campaign_path = Path(path).expanduser().resolve()
    raw = _yaml(campaign_path)
    primary_missing = _primary_missing(campaign_path) if raw.get("campaign_purpose") == "family_generalization" else []
    probe_required, declared_unverified = _profile_mapping_states(campaign_path)
    return {
        "missing_production_keys": sorted(primary_missing),
        "missing_profile_mapping_keys": sorted(probe_required),
        "declared_unverified_profile_mapping_keys": sorted(declared_unverified),
    }


def _campaign_contract(
    campaign: config_service.CampaignConfig,
) -> dict[str, Any]:
    """Return configured profile, inventory, role, regime, and package evidence."""
    inventory = campaign.material_inventory
    return {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "source_path": _relative(campaign.source_path),
        "simulation_profile": campaign.profile.id,
        "total_case_count": campaign.total_case_count,
        "material_inventory": list(inventory),
        "material_roles": {role: list(material_families) for role, material_families in campaign.material_roles.items()},
        "evaluation_regimes": list(campaign.evaluation_regimes),
        "dataset_packages": [
            {
                "dataset_name": package["dataset_name"],
                "dataset_view": package["dataset_view"],
                "evaluation_regime": package["evaluation_regime"],
                "source_role": package["source_role"],
                "materials": list(package["materials"]),
                "source_case_count": package["source_case_count"],
            }
            for package in campaign.dataset_packages
        ],
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
    campaigns = tuple(config_service.load_campaign_config(path, require_executable=False) for path in (steady_path, transient_path))
    expected_profiles = (
        profiles.STEADY_FLOW_PROFILE,
        profiles.TRANSIENT_DRYING_PROFILE,
    )
    for campaign, expected_profile in zip(campaigns, expected_profiles, strict=True):
        if campaign.campaign_purpose != "family_generalization" or campaign.profile.id != expected_profile:
            message = (
                "Readiness inputs must be family-generalization campaigns for "
                f"{expected_profile!r}; received {campaign.profile.id!r} "
                f"with purpose {campaign.campaign_purpose!r}."
            )
            raise ValueError(message)
    campaign_contracts = {campaign.profile.id: _campaign_contract(campaign) for campaign in campaigns}
    campaign_resolution_complete = len(campaign_contracts) == len(campaigns)
    resolved_ownership_complete = campaign_resolution_complete

    steady_gates = campaign_unresolved_gates(steady_path)
    transient_gates = campaign_unresolved_gates(transient_path)
    primary_missing = sorted(set(steady_gates["missing_production_keys"] + transient_gates["missing_production_keys"]))
    mapping_missing = sorted(set(steady_gates["missing_profile_mapping_keys"] + transient_gates["missing_profile_mapping_keys"]))
    mapping_declared_unverified = sorted(
        set(steady_gates["declared_unverified_profile_mapping_keys"] + transient_gates["declared_unverified_profile_mapping_keys"])
    )

    static_report = None
    if run_static_sentinels:
        static_report = sentinel_service.run_static_sentinels(
            steady_path,
            transient_path,
        )
    receipt_path = None if real_runtime_receipt is None else Path(real_runtime_receipt).expanduser().resolve()
    real_complete = False
    if receipt_path is not None:
        smoke_service.validate_real_smoke_receipt(receipt_path)
        real_complete = True

    primary_complete = not primary_missing
    mapping_complete = not mapping_missing and (not mapping_declared_unverified or real_complete)
    static_complete = static_report is not None and static_report["status"] == "pass"
    static_status = _STATUS_COMPLETE if static_complete else _STATUS_BLOCKED if static_report is not None else _STATUS_PENDING
    production_ready = all(
        (
            campaign_resolution_complete,
            resolved_ownership_complete,
            primary_complete,
            mapping_complete,
            static_complete,
            real_complete,
        )
    )
    return {
        "schema_kind": "vp2_production_readiness",
        "schema_version": 1,
        "campaign_contracts": campaign_contracts,
        "campaign_ids": [campaign.campaign_id for campaign in campaigns],
        "campaign_config_resolution_complete": campaign_resolution_complete,
        "resolved_config_ownership_validation_complete": resolved_ownership_complete,
        "static_generator_sentinels_complete": static_complete,
        "real_runtime_validation_complete": real_complete,
        "primary_production_config_complete": primary_complete,
        "profile_mapping_complete": mapping_complete,
        "production_ready_for_user_launch": production_ready,
        "missing_primary_keys": primary_missing,
        "missing_profile_mapping_keys": mapping_missing,
        "declared_unverified_profile_mapping_keys": mapping_declared_unverified,
        "real_runtime_receipt": (None if receipt_path is None else str(receipt_path)),
        "static_sentinel_report": static_report,
        "status_lines": [
            ("CAMPAIGN_CONFIG_RESOLUTION_COMPLETE" if campaign_resolution_complete else "CAMPAIGN_CONFIG_RESOLUTION_INCOMPLETE"),
            ("RESOLVED_CONFIG_OWNERSHIP_VALIDATION_COMPLETE" if resolved_ownership_complete else "RESOLVED_CONFIG_OWNERSHIP_VALIDATION_INCOMPLETE"),
            f"STATIC_GENERATOR_SENTINELS_{static_status}",
            (f"PRIMARY_PRODUCTION_CONFIG_{_STATUS_COMPLETE if primary_complete else _STATUS_INCOMPLETE}"),
            (f"PROFILE_MAPPING_{_STATUS_COMPLETE if mapping_complete else _STATUS_INCOMPLETE}"),
            (f"REAL_RUNTIME_VALIDATION_{_STATUS_COMPLETE if real_complete else _STATUS_PENDING}"),
            (f"PRODUCTION_READY_FOR_USER_LAUNCH_{_STATUS_COMPLETE if production_ready else _STATUS_BLOCKED}"),
        ],
        "production_solve_started": False,
    }
