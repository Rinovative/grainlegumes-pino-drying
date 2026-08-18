"""
generation_input_labels.py

Build concise generation-input dataset labels from canonical metadata.
Responsibilities:
  - Map maintained simulation profiles to compact presentation labels
  - Format material, regime, and campaign-purpose label components
  - Build complete profile-qualified dataset labels
  - Disambiguate genuine visible-label collisions with stable identities
Design principles:
  - Canonical metadata remains authoritative over storage or filename parsing
  - Presentation labels never replace scientific or persisted identities
  - Identity suffixes appear only for otherwise identical complete labels
This module does NOT:
  - Discover datasets, admit persisted inputs, or render notebook controls
  - Define canonical profile, campaign-purpose, or batch identity values
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from src import common
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Iterable

_COLLISION_IDENTITY_MINIMUM_WIDTH: Final = 8
_COLLISION_MINIMUM_SIZE: Final = 2
_PROFILE_LABELS: Final = MappingProxyType(
    {
        profiles.STEADY_FLOW_PROFILE: "Airflow",
        profiles.TRANSIENT_DRYING_PROFILE: "Drying",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetLabelMetadata:
    """Hold the canonical metadata required for one visible dataset label."""

    profile_id: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str
    batch_identity: str


def profile_display_label(profile_id: str) -> str:
    """Return the compact maintained label for one canonical profile."""
    profiles.resolve_profile(profile_id)
    return _PROFILE_LABELS[profile_id]


def material_display_label(material_family: str) -> str:
    """Return one human-readable material-family label."""
    common.paths.validate_logical_name(material_family, label="material_family")
    return material_family.replace("_", " ").capitalize()


def regime_display_label(sampling_regime: str) -> str:
    """Return one canonical sampling-regime display label."""
    common.paths.validate_logical_name(sampling_regime, label="sampling_regime")
    return sampling_regime


def campaign_purpose_abbreviation(campaign_purpose: str) -> str:
    """Abbreviate a canonical purpose from each underscore component."""
    common.paths.validate_logical_name(
        campaign_purpose,
        label="campaign_purpose",
    )
    components = tuple(component for component in campaign_purpose.split("_") if component)
    if not components or "_".join(components) != campaign_purpose:
        message = "campaign_purpose must use non-empty underscore-separated components."
        raise ValueError(message)
    return "".join(component[0] for component in components).lower()


@dataclass(frozen=True, slots=True)
class CampaignPurposeLegendRow:
    """Describe one canonical campaign-purpose abbreviation."""

    abbreviation: str
    campaign_purpose: str


def campaign_purpose_legend_rows(
    *purpose_groups: Iterable[str],
) -> tuple[CampaignPurposeLegendRow, ...]:
    """Return deduplicated canonical purpose rows in deterministic order."""
    purposes = frozenset(purpose for group in purpose_groups for purpose in group)
    rows = tuple(
        CampaignPurposeLegendRow(
            abbreviation=campaign_purpose_abbreviation(purpose),
            campaign_purpose=purpose,
        )
        for purpose in purposes
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.abbreviation, row.campaign_purpose),
        )
    )


def profile_label_rows(
    profile_ids: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    """Return compact labels and canonical profile tokens separately."""
    unique = tuple(dict.fromkeys(profile_ids))
    return tuple((profile_display_label(profile_id), profile_id) for profile_id in unique)


def dataset_display_label(metadata: DatasetLabelMetadata) -> str:
    """Return one complete compact label without collision evidence."""
    return " · ".join(
        (
            profile_display_label(metadata.profile_id),
            material_display_label(metadata.material_family),
            regime_display_label(metadata.sampling_regime),
            campaign_purpose_abbreviation(metadata.campaign_purpose),
        )
    )


def dataset_display_labels(
    datasets: tuple[DatasetLabelMetadata, ...],
) -> tuple[str, ...]:
    """Disambiguate complete compact labels with the shortest stable suffix."""
    bases = tuple(dataset_display_label(dataset) for dataset in datasets)
    resolved = list(bases)
    for base in set(bases):
        positions = tuple(index for index, label in enumerate(bases) if label == base)
        if len(positions) < _COLLISION_MINIMUM_SIZE:
            continue
        identities = tuple(datasets[index].batch_identity for index in positions)
        maximum_width = max(len(identity) for identity in identities)
        for width in range(_COLLISION_IDENTITY_MINIMUM_WIDTH, maximum_width + 1):
            prefixes = tuple(identity[:width] for identity in identities)
            if len(prefixes) == len(set(prefixes)):
                for index, prefix in zip(positions, prefixes, strict=True):
                    resolved[index] = f"{base} · {prefix}"
                break
        else:
            message = f"Colliding dataset label has no unique batch identity: {base!r}."
            raise ValueError(message)
    return tuple(resolved)
