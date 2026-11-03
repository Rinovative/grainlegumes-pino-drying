"""Plan the bounded no-COMSOL sentinel workload of a Source campaign.

Responsibilities:
  - Require a Source campaign with natural support for its full configured material inventory
  - Enumerate every configured profile-active parameter-OOD unit by physical group and material

Design principles:
  - Sentinel views are isolated derived copies and never mutate canonical campaigns
  - The workload follows the resolved configuration and never the production case counts

This module does NOT:
  - Start COMSOL, mutate campaign YAML, authorize production, or fit parameters
  - Sample cases, realize fields, or create alternate sampling modes
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.contracts import generation_contracts_materials as materials

if TYPE_CHECKING:
    from src.generation.cases import generation_cases_config as config_service

_ID_SENTINEL_CASE_COUNT: Final = 8


def _ood_candidates(batch: config_service.GenerationConfig) -> dict[str, tuple[str, ...]]:
    """Return every configured profile-active OOD unit by physical group."""
    groups = materials.active_ood_groups(batch.scientific_values["material"]["parameter_registry"], batch.profile.id)
    candidates: dict[str, list[str]] = {group: [] for group in groups}
    for unit in sampling_service.eligible_ood_units(
        batch.scientific_values["material"],
        groups=groups,
    ):
        candidates[unit["ood_group"]].append(unit["unit_id"])
    return {group: tuple(names) for group, names in candidates.items() if names}


def _campaign_inventory(campaign: config_service.CampaignConfig) -> tuple[str, ...]:
    """Return the configured Source materials in authored order."""
    inventory = tuple(campaign.selected_targets)
    if not inventory or len(inventory) != len(set(inventory)):
        message = f"Campaign {campaign.campaign_id!r} has an empty or duplicate material inventory."
        raise ValueError(message)
    return inventory


def _parameter_ood_source(
    natural_batch: config_service.GenerationConfig,
) -> config_service.GenerationConfig:
    """Return a sentinel-only parameter-OOD source independent of production counts."""
    if natural_batch.sampling_regime != "natural":
        message = "Parameter-OOD sentinels require one natural source batch."
        raise ValueError(message)
    scientific = copy.deepcopy(natural_batch.scientific_values)
    scientific["sampling_regime"] = "parameter_ood"
    scientific["domain_profile"] = "parameter_extension"
    scientific["natural_support_state"] = "parameter_ood"
    return replace(
        natural_batch,
        evaluation_regime="parameter_extension",
        sampling_regime="parameter_ood",
        scientific_values=scientific,
    )


def _validate_campaign(
    campaign: config_service.CampaignConfig,
    profile_id: str,
) -> tuple[str, ...]:
    """Require a Source campaign with natural support for its full inventory."""
    if campaign.campaign_purpose != "source_generation" or campaign.profile.id != profile_id:
        message = f"Static sentinel input is not a Source campaign for {profile_id!r}."
        raise ValueError(message)
    inventory = _campaign_inventory(campaign)
    missing_natural = [
        material_family
        for material_family in inventory
        if campaign.find_batch(
            material_family=material_family,
            sampling_regime="natural",
        )
        is None
    ]
    if missing_natural:
        message = f"Static sentinel campaign lacks natural batches for configured materials {missing_natural}."
        raise ValueError(message)
    return inventory


def inspect_sentinel_workload(
    campaign: config_service.CampaignConfig,
) -> dict[str, Any]:
    """Return the bounded config-derived static-sentinel workload plan."""
    inventory = _validate_campaign(campaign, campaign.profile.id)
    parameter_ood: dict[str, Any] = {}
    for material_family in inventory:
        natural = campaign.require_batch(
            material_family=material_family,
            sampling_regime="natural",
        )
        source = campaign.find_batch(
            material_family=material_family,
            sampling_regime="parameter_ood",
        )
        candidates = _ood_candidates(_parameter_ood_source(natural) if source is None else source)
        parameter_ood[material_family] = {
            "case_count": sum(len(names) for names in candidates.values()),
            "eligible_units_by_group": {group: list(names) for group, names in candidates.items()},
        }
    return {
        "natural_materials": list(inventory),
        "natural_cases_per_material": _ID_SENTINEL_CASE_COUNT,
        "natural_case_count": len(inventory) * _ID_SENTINEL_CASE_COUNT,
        "parameter_ood": parameter_ood,
        "parameter_ood_case_count": sum(int(evidence["case_count"]) for evidence in parameter_ood.values()),
        "production_case_count_independent": True,
    }
