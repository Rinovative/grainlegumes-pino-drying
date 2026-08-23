"""
analysis_display_labels.py

Own concise scientific display labels while preserving canonical identities separately.

Responsibilities:
  - Humanize maintained task, material, regime, purpose, and role identifiers
  - Hold the authoritative metadata required for one visible dataset label
  - Disambiguate colliding dataset labels with the shortest stable identity prefix
  - Preserve canonical identities as metadata rather than parsing presentation text

Design principles:
  - Presentation labels never become storage, scientific, or persistence identities
  - Task, material or variation, campaign, and role remain distinct visible concepts
  - Collision suffixes are deterministic and appear only when required for uniqueness

This module does NOT:
  - Discover datasets, infer metadata, or validate persisted artifacts
  - Define task capabilities or scientific field semantics
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill
from typing import TYPE_CHECKING

from src.analysis import generation_inputs

if TYPE_CHECKING:
    from collections.abc import Iterable

_COLLISION_MINIMUM_SIZE = 2
_TASK_LABELS = {
    "steady_flow": "Airflow",
    "transient_drying": "Drying",
}
_ROLE_LABELS = {
    "id": "ID",
    "seen": "ID",
    "id_source": "ID",
    "parameter_ood": "P OOD",
    "family_ood": "F OOD",
    "held_out_family_ood": "F OOD",
    "near_family_ood": "NF OOD",
    "far_family_ood": "FF OOD",
    "extreme_family_ood": "S OOD",
    "stress_ood": "S OOD",
}
_ACRONYMS = frozenset({"id", "ood"})


@dataclass(frozen=True, slots=True)
class DatasetDisplayMetadata:
    """Hold the authoritative metadata required to label one dataset."""

    task_id: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str | None
    source_role: str | None
    evaluation_regime: str | None
    canonical_identity: str

    def __post_init__(self) -> None:
        """Validate independently owned display and identity metadata."""
        _require_identifier(self.task_id, label="task_id")
        _require_identifier(self.material_family, label="material_family")
        _require_identifier(self.sampling_regime, label="sampling_regime")
        for label, value in (
            ("campaign_purpose", self.campaign_purpose),
            ("source_role", self.source_role),
            ("evaluation_regime", self.evaluation_regime),
        ):
            if value is not None:
                _require_identifier(value, label=label)
        _require_identity(self.canonical_identity)


def humanize_identifier(identifier: str) -> str:
    """
    Convert one underscore-delimited identifier to a concise display label.

    Parameters
    ----------
    identifier : str
        Non-empty canonical identifier. It is retained unchanged by callers.

    Returns
    -------
    str
        Title-cased display text with underscores replaced by spaces.

    Raises
    ------
    TypeError
        If ``identifier`` is not text.
    ValueError
        If ``identifier`` is blank or has empty underscore-delimited components.

    """
    _require_identifier(identifier, label="identifier")
    return " ".join(component.upper() if component.casefold() in _ACRONYMS else component.capitalize() for component in identifier.split("_"))


def task_display_label(task_id: str) -> str:
    """
    Return the centralized human-readable label for one task identifier.

    Parameters
    ----------
    task_id : str
        Canonical task identifier.

    Returns
    -------
    str
        Humanized task label.

    """
    _require_identifier(task_id, label="task_id")
    return _TASK_LABELS.get(task_id, humanize_identifier(task_id))


def material_display_label(material_family: str) -> str:
    """Return the shared Generation-input human-readable material label."""
    _require_identifier(material_family, label="material_family")
    return generation_inputs.labels.material_display_label(material_family)


def regime_display_label(sampling_regime: str) -> str:
    """Return the centralized human-readable label for one sampling regime."""
    return humanize_identifier(sampling_regime)


def campaign_role_display_label(metadata: DatasetDisplayMetadata) -> str:
    """Return the authoritative concise evaluation role at the end of a label."""
    if not isinstance(metadata, DatasetDisplayMetadata):
        message = "metadata must be DatasetDisplayMetadata."
        raise TypeError(message)
    role = metadata.evaluation_regime
    if role is None and metadata.sampling_regime == "parameter_ood":
        role = "parameter_ood"
    if role is None:
        role = metadata.source_role
    if role is None:
        return "role unspecified"
    known = _ROLE_LABELS.get(role)
    if known is not None:
        return known
    humanized = humanize_identifier(role)
    return humanized if humanized == "ID" else humanized[:1].lower() + humanized[1:]


def dataset_display_label(metadata: DatasetDisplayMetadata) -> str:
    """
    Return the ordinary task-qualified dataset label without collision evidence.

    Parameters
    ----------
    metadata : DatasetDisplayMetadata
        Canonical metadata retained separately from the returned text.

    Returns
    -------
    str
        Task, material or variation, abbreviated campaign, and authoritative role.

    """
    if not isinstance(metadata, DatasetDisplayMetadata):
        message = "metadata must be DatasetDisplayMetadata."
        raise TypeError(message)
    parts = [
        task_display_label(metadata.task_id),
        material_display_label(metadata.material_family),
    ]
    if metadata.campaign_purpose is not None:
        parts.append(
            generation_inputs.labels.campaign_purpose_abbreviation(
                metadata.campaign_purpose,
            )
        )
    parts.append(campaign_role_display_label(metadata))
    return " · ".join(parts)


def wrapped_dataset_display_label(
    label: str,
    *,
    width: int = 32,
) -> str:
    """Wrap one complete visible dataset label without removing identity text."""
    if not isinstance(label, str):
        message = "Dataset display labels must be text."
        raise TypeError(message)
    if not label or label != label.strip():
        message = "Dataset display labels must be non-empty without surrounding whitespace."
        raise ValueError(message)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        message = "Dataset display-label wrap width must be a positive integer."
        raise ValueError(message)
    return fill(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def dataset_display_labels(datasets: Iterable[DatasetDisplayMetadata]) -> tuple[str, ...]:
    """
    Build deterministic concise labels and disambiguate genuine collisions.

    Parameters
    ----------
    datasets : Iterable[DatasetDisplayMetadata]
        Dataset metadata in caller-selected order.

    Returns
    -------
    tuple[str, ...]
        One visible label per input dataset. A shortest unique canonical-identity
        prefix is appended only to members sharing the same ordinary label.

    Raises
    ------
    TypeError
        If an input is not DatasetDisplayMetadata.
    ValueError
        If canonical identities repeat or cannot distinguish a collision.

    Notes
    -----
    Canonical identities are never parsed. They remain separately available in
    each ``DatasetDisplayMetadata`` instance.

    """
    resolved_datasets = tuple(datasets)
    if any(not isinstance(dataset, DatasetDisplayMetadata) for dataset in resolved_datasets):
        message = "datasets must contain only DatasetDisplayMetadata instances."
        raise TypeError(message)
    identities = tuple(dataset.canonical_identity for dataset in resolved_datasets)
    if len(identities) != len(set(identities)):
        message = "Dataset display metadata must have unique canonical identities."
        raise ValueError(message)
    bases = tuple(dataset_display_label(dataset) for dataset in resolved_datasets)
    labels = list(bases)
    for base in sorted(set(bases)):
        positions = tuple(index for index, candidate in enumerate(bases) if candidate == base)
        if len(positions) < _COLLISION_MINIMUM_SIZE:
            continue
        prefixes = _unique_identity_prefixes(tuple(identities[index] for index in positions))
        for position, prefix in zip(positions, prefixes, strict=True):
            labels[position] = f"{base} · {prefix}"
    return tuple(labels)


def _unique_identity_prefixes(identities: tuple[str, ...]) -> tuple[str, ...]:
    """Return the shortest deterministic unique prefix for each identity."""
    maximum_width = max(len(identity) for identity in identities)
    for width in range(1, maximum_width + 1):
        prefixes = tuple(identity[:width] for identity in identities)
        if len(prefixes) == len(set(prefixes)):
            return prefixes
    message = "Colliding dataset labels require distinct canonical identities."
    raise ValueError(message)


def _require_identifier(value: str, *, label: str) -> None:
    """Require one non-empty underscore-delimited presentation identifier."""
    if not isinstance(value, str):
        message = f"{label} must be text."
        raise TypeError(message)
    if not value or value != value.strip() or any(not component for component in value.split("_")):
        message = f"{label} must be non-empty underscore-delimited text."
        raise ValueError(message)


def _require_identity(value: str) -> None:
    """Require canonical identity text without interpreting its structure."""
    if not isinstance(value, str):
        message = "canonical_identity must be text."
        raise TypeError(message)
    if not value or value != value.strip():
        message = "canonical_identity must be non-empty text without surrounding whitespace."
        raise ValueError(message)
