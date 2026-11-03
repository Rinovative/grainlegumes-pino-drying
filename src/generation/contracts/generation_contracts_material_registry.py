"""Own stable Generation material identifiers, display slugs, and ordering.

Responsibilities:
  - Map supported material IDs to canonical campaign-name slugs
  - Order material membership independently of file enumeration

Design principles:
  - Slugs never contain campaign grammar separators
  - Material identity remains the existing scientific identifier

This module does NOT:
  - Define material properties or discover material configuration files
"""

from __future__ import annotations

from typing import Final

MATERIAL_ORDER: Final[tuple[str, ...]] = (
    "lentil",
    "chickpea",
    "kidney_bean",
    "field_pea",
    "rapeseed",
    "sunflower_seed",
)
MATERIAL_SLUGS: Final[dict[str, str]] = {material: material.replace("_", "-") for material in MATERIAL_ORDER}


def material_slug(material: str) -> str:
    """Return the unique canonical ASCII slug for one supported material."""
    try:
        return MATERIAL_SLUGS[material]
    except KeyError as error:
        msg = f"Unknown campaign material {material!r}."
        raise ValueError(msg) from error


def ordered_materials(materials: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return unique supported materials in scientific registry order."""
    selected = set(materials)
    unknown = selected.difference(MATERIAL_SLUGS)
    if unknown or len(selected) != len(materials):
        msg = f"Campaign materials must be unique and registered; unknown={sorted(unknown)}."
        raise ValueError(msg)
    return tuple(material for material in MATERIAL_ORDER if material in selected)
