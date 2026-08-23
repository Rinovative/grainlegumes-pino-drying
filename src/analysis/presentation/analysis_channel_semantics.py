"""
analysis_channel_semantics.py

Define shared ordering and compatibility rules for scientific display channels.

Responsibilities:
  - Classify channels into four authoritative scientific field groups
  - Preserve schema-declared order within each group
  - Exclude coordinate channels unless a view explicitly requests them
  - Resolve deterministic intersections without inspecting scientific arrays

Design principles:
  - Explicit task/schema metadata owns category and within-group order
  - Airflow inputs precede airflow outputs and transient inputs precede outputs
  - Caller declaration order is the deterministic fallback for unknown fields

This module does NOT:
  - Load fields, calculate derived quantities, or validate numerical units
  - Define widgets, plot layouts, or persistent task schemas
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

ChannelCategory = Literal[
    "airflow_input",
    "airflow_output",
    "transient_input",
    "transient_output",
    "diagnostic",
    "coordinate",
]

_AIRFLOW_OUTPUT_PRIORITY = ("p", "u", "v", "U")
_TRANSIENT_OUTPUT_PRIORITY = ("T", "phi", "w_surf", "w_int")
_COORDINATE_NAMES = frozenset(
    {
        "x",
        "y",
        "z",
        "t",
        "time",
        "time_s",
        "coordinate_x",
        "coordinate_y",
        "coordinate_z",
    }
)
_CATEGORY_RANK: Mapping[ChannelCategory, int] = {
    "airflow_input": 0,
    "airflow_output": 1,
    "transient_input": 2,
    "transient_output": 3,
    "diagnostic": 4,
    "coordinate": 5,
}


@dataclass(frozen=True, slots=True)
class ChannelPresentationMetadata:
    """Describe one authoritative field group and within-group order."""

    category: ChannelCategory
    order: int = 0

    def __post_init__(self) -> None:
        """Validate the bounded shared channel presentation vocabulary."""
        if self.category not in _CATEGORY_RANK:
            message = f"Unsupported channel category {self.category!r}."
            raise ValueError(message)
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            message = "Channel presentation order must be an integer."
            raise TypeError(message)


def ordered_channels(
    channel_names: Iterable[str],
    *,
    metadata: Mapping[str, ChannelPresentationMetadata | ChannelCategory] | None = None,
    include_coordinates: bool = False,
) -> tuple[str, ...]:
    """Return unique channels in the four-group scientific order."""
    names = _unique_channel_names(channel_names)
    normalized_metadata = _normalize_metadata(metadata)
    declaration_order = {name: index for index, name in enumerate(names)}
    visible = tuple(
        name for name in names if include_coordinates or _channel_metadata(name, normalized_metadata, declaration_order).category != "coordinate"
    )
    return tuple(
        sorted(
            visible,
            key=lambda name: _channel_sort_key(
                name,
                normalized_metadata,
                declaration_order,
            ),
        )
    )


def compatible_channels(
    channel_groups: Iterable[Iterable[str]],
    *,
    metadata: Mapping[str, ChannelPresentationMetadata | ChannelCategory] | None = None,
    include_coordinates: bool = False,
) -> tuple[str, ...]:
    """Return the ordered exact intersection of selected channel groups."""
    groups = tuple(tuple(group) for group in channel_groups)
    if not groups:
        return ()
    first = _unique_channel_names(groups[0])
    common = set(first)
    for group in groups[1:]:
        common.intersection_update(_unique_channel_names(group))
    return ordered_channels(
        (name for name in first if name in common),
        metadata=metadata,
        include_coordinates=include_coordinates,
    )


def _unique_channel_names(channel_names: Iterable[str]) -> tuple[str, ...]:
    """Validate names and retain their first declared occurrence."""
    names: list[str] = []
    for name in channel_names:
        if not isinstance(name, str):
            message = "Channel names must be text."
            raise TypeError(message)
        normalized = name.strip()
        if not normalized:
            message = "Channel names must be non-empty text."
            raise ValueError(message)
        if normalized not in names:
            names.append(normalized)
    return tuple(names)


def _normalize_metadata(
    metadata: Mapping[str, ChannelPresentationMetadata | ChannelCategory] | None,
) -> dict[str, ChannelPresentationMetadata]:
    """Normalize compact group declarations to immutable metadata."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        message = "metadata must be a mapping when provided."
        raise TypeError(message)
    normalized: dict[str, ChannelPresentationMetadata] = {}
    for name, declaration in metadata.items():
        if not isinstance(name, str) or not name.strip():
            message = "Channel metadata names must be non-empty text."
            raise TypeError(message)
        resolved = ChannelPresentationMetadata(declaration) if isinstance(declaration, str) else declaration
        if not isinstance(resolved, ChannelPresentationMetadata):
            message = "Channel metadata values must be categories or ChannelPresentationMetadata."
            raise TypeError(message)
        normalized[name] = resolved
    return normalized


def _channel_metadata(
    name: str,
    metadata: Mapping[str, ChannelPresentationMetadata],
    declaration_order: Mapping[str, int],
) -> ChannelPresentationMetadata:
    """Resolve explicit semantics or a conservative display fallback."""
    if name in metadata:
        return metadata[name]
    if name in _COORDINATE_NAMES:
        return ChannelPresentationMetadata("coordinate", declaration_order[name])
    if name in _AIRFLOW_OUTPUT_PRIORITY:
        return ChannelPresentationMetadata(
            "airflow_output",
            _AIRFLOW_OUTPUT_PRIORITY.index(name),
        )
    if name in _TRANSIENT_OUTPUT_PRIORITY:
        return ChannelPresentationMetadata(
            "transient_output",
            _TRANSIENT_OUTPUT_PRIORITY.index(name),
        )
    return ChannelPresentationMetadata("diagnostic", declaration_order[name])


def _channel_sort_key(
    name: str,
    metadata: Mapping[str, ChannelPresentationMetadata],
    declaration_order: Mapping[str, int],
) -> tuple[int, int, int]:
    """Return the deterministic group/order/declaration sort key."""
    resolved = _channel_metadata(name, metadata, declaration_order)
    return (
        _CATEGORY_RANK[resolved.category],
        resolved.order,
        declaration_order[name],
    )
