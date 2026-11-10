"""Own concise scientific display labels while preserving canonical identities separately.

Responsibilities:
  - Humanize maintained task, material, physical domain, and run role identifiers
  - Hold the authoritative metadata required for one visible dataset label
  - Disambiguate colliding dataset labels with the shortest stable identity prefix
  - Make short run and Dataset labels unique with one separating facet, shortened by common affixes and a middle ellipsis
  - Preserve canonical identities as metadata rather than parsing presentation text

Design principles:
  - Presentation labels never become storage, scientific, or persistence identities
  - Source domains and Experiment evaluation roles remain distinct visible concepts
  - Collision suffixes are deterministic and appear only when required for uniqueness

This module does NOT:
  - Discover datasets, infer metadata, or validate persisted artifacts
  - Define task capabilities or scientific field semantics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from textwrap import fill
from typing import TYPE_CHECKING, cast

from src.analysis import generation_inputs

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_COLLISION_MINIMUM_SIZE = 2
_FACET_MINIMUM_WIDTH = 24
_FACET_TOKEN_SEPARATOR = re.compile(r"/|__")
_TASK_LABELS = {
    "steady_flow": "Airflow",
    "transient_drying": "Drying",
}
_EVALUATION_ROLE_LABELS = {
    "id": "ID",
    "ood": "OOD",
    "material_ood": "Material OOD",
    "parameter_ood": "Parameter OOD",
}
_EVALUATION_PROTOCOL_LABELS = {
    "normalized_drying_group_macro_rmse": "Drying group macro RMSE",
}
_ACRONYMS = frozenset({"id", "ood"})


@dataclass(frozen=True, slots=True)
class DatasetDisplayMetadata:
    """Hold the authoritative metadata required to label one dataset."""

    task_id: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str | None
    domain_profile: str | None
    canonical_identity: str

    def __post_init__(self) -> None:
        """Validate independently owned display and identity metadata."""
        _require_identifier(self.task_id, label="task_id")
        _require_identifier(self.material_family, label="material_family")
        _require_identifier(self.sampling_regime, label="sampling_regime")
        for label, value in (
            ("campaign_purpose", self.campaign_purpose),
            ("domain_profile", self.domain_profile),
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


def evaluation_role_display_label(dataset_role: str) -> str:
    """Return the run-relative scientific role for one Evaluation artifact."""
    _require_identifier(dataset_role, label="dataset_role")
    return _EVALUATION_ROLE_LABELS.get(dataset_role, humanize_identifier(dataset_role))


def material_role_display_label(material_family: str, dataset_role: str) -> str:
    """Return one material-first label with run-relative Evaluation role."""
    return f"{material_display_label(material_family)} · {evaluation_role_display_label(dataset_role)}"


def evaluation_protocol_display_label(protocol: str) -> str:
    """Return the shared human-facing label for one exact Evaluation objective."""
    _require_identifier(protocol, label="protocol")
    return _EVALUATION_PROTOCOL_LABELS.get(protocol, humanize_identifier(protocol))


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
        Task, material or variation, abbreviated campaign, and physical domain.

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
    if metadata.domain_profile is not None:
        parts.append(humanize_identifier(metadata.domain_profile))
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


def unique_display_labels(bases: Sequence[str], facets: Sequence[Sequence[str | None]]) -> tuple[str, ...]:
    """
    Make equal base labels unique by appending the first separating facet.

    Parameters
    ----------
    bases : Sequence[str]
        One short label per item.
    facets : Sequence[Sequence[str | None]]
        One tuple per item with its facet values in priority order; ``None``
        disqualifies the facet for the whole collision group.

    Returns
    -------
    tuple[str, ...]
        ``base`` alone when unique, otherwise ``base · facet`` with the facet
        reduced to the ``/``- and ``__``-separated tokens that not all items of
        its group share and, beyond 24 characters, shown as ``head…tail`` with
        the smallest unique length. An item without a distinguishing token keeps
        the bare ``base``.

    Raises
    ------
    ValueError
        If the inputs differ in length or no facet separates a collision group.

    """
    if len(bases) != len(facets):
        message = "bases and facets must have one entry per item."
        raise ValueError(message)
    labels = list(bases)
    for base in dict.fromkeys(bases):
        group = tuple(index for index, candidate in enumerate(bases) if candidate == base)
        if len(group) < _COLLISION_MINIMUM_SIZE:
            continue
        for facet_index in range(max(len(facets[index]) for index in group)):
            values = tuple(facets[index][facet_index] if facet_index < len(facets[index]) else None for index in group)
            if None in values:
                continue
            shown_values = _shorten_facet_values(cast("tuple[str, ...]", values))
            if shown_values is not None:
                for index, shown in zip(group, shown_values, strict=True):
                    labels[index] = f"{base} · {shown}" if shown else base
                break
        else:
            message = f"No facet separates the items labelled {base!r}."
            raise ValueError(message)
    return tuple(labels)


def _shorten_facet_values(values: tuple[str, ...]) -> tuple[str, ...] | None:
    """Keep the tokens not shared by all values, ellipsize to the smallest unique length; ``None`` if not separating."""
    tokens = tuple(_FACET_TOKEN_SEPARATOR.split(value) for value in values)
    shared = set.intersection(*(set(item) for item in tokens))
    cores = tuple(" ".join(token for token in item if token not in shared) for item in tokens)
    if len(set(cores)) != len(cores):
        return None
    for width in range(_FACET_MINIMUM_WIDTH, max(len(core) for core in cores) + 1):
        head = width // 2
        shown = tuple(core if len(core) <= width else f"{core[:head]}…{core[len(core) - (width - head - 1) :]}" for core in cores)
        if len(set(shown)) == len(shown):
            return shown
    return cores


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
