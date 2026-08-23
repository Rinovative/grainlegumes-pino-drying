"""
analysis_visual_semantics.py

Define shared semantic colormaps and deterministic dataset colors for analysis views.

Responsibilities:
  - Resolve a field's semantic colormap and centered-scale requirement
  - Keep plot normalization separate from semantic color selection
  - Describe dataset identities used for deterministic presentation colors
  - Assign colorblind-friendly colors independently of selected dataset order

Design principles:
  - Explicit semantic roles take precedence over field-name inference
  - Canonical identities provide stable color assignment across EDA and evaluation
  - Colormap selection communicates field semantics rather than plot ownership

This module does NOT:
  - Construct Matplotlib normalizers, colorbars, axes, or legends
  - Load scientific arrays or derive numerical error fields
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

FieldSemanticRole = Literal[
    "temperature",
    "humidity",
    "granular_moisture",
    "pressure",
    "scalar",
    "signed_velocity",
    "velocity_magnitude",
    "signed_error",
    "absolute_error",
]

_DATASET_COLOR_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)


@dataclass(frozen=True, slots=True)
class FieldVisualSemantics:
    """Describe the colormap and center requirement for one displayed field."""

    colormap: str
    centered: bool = False


@dataclass(frozen=True, slots=True)
class DatasetVisualIdentity:
    """Hold the identity and current concise label for one dataset color."""

    canonical_identity: str
    label: str

    def __post_init__(self) -> None:
        """Require separately preserved identity and non-empty presentation text."""
        _require_text(self.canonical_identity, label="canonical_identity")
        _require_text(self.label, label="label")


def field_visual_semantics(
    field_name: str,
    *,
    role: FieldSemanticRole | None = None,
) -> FieldVisualSemantics:
    """
    Resolve shared visual semantics for one field or explicitly derived quantity.

    Parameters
    ----------
    field_name : str
        Canonical field key or concise display field name.
    role : FieldSemanticRole, optional
        Explicit semantic role for derived or ambiguous quantities.

    Returns
    -------
    FieldVisualSemantics
        Colormap name and whether plot code should center a diverging scale.

    Raises
    ------
    TypeError
        If ``field_name`` or ``role`` have invalid types.
    ValueError
        If ``field_name`` is blank or ``role`` is unsupported.

    Notes
    -----
    This function intentionally does not create Matplotlib normalizers. Plot
    owners apply the returned ``centered`` requirement to their own data range.

    """
    _require_text(field_name, label="field_name")
    resolved_role = _resolve_role(field_name, role)
    return _ROLE_SEMANTICS[resolved_role]


def dataset_colors(datasets: Iterable[DatasetVisualIdentity]) -> dict[str, str]:
    """
    Assign deterministic colorblind-friendly colors keyed by canonical identity.

    Parameters
    ----------
    datasets : Iterable[DatasetVisualIdentity]
        Dataset identities and current labels. Input order does not affect output.

    Returns
    -------
    dict[str, str]
        Canonical-identity to hexadecimal color mapping.

    Raises
    ------
    TypeError
        If an input is not DatasetVisualIdentity.
    ValueError
        If canonical identities repeat.

    """
    resolved = tuple(datasets)
    if any(not isinstance(dataset, DatasetVisualIdentity) for dataset in resolved):
        message = "datasets must contain only DatasetVisualIdentity instances."
        raise TypeError(message)
    identities = tuple(dataset.canonical_identity for dataset in resolved)
    if len(identities) != len(set(identities)):
        message = "Dataset visual identities must have unique canonical identities."
        raise ValueError(message)
    ranked = tuple(sorted(resolved, key=_dataset_rank))
    assigned: dict[str, str] = {}
    occupied: set[int] = set()
    palette_size = len(_DATASET_COLOR_PALETTE)
    for dataset in ranked:
        preferred = _stable_palette_index(dataset, palette_size)
        index = preferred
        while index in occupied and len(occupied) < palette_size:
            index = (index + 1) % palette_size
        occupied.add(index)
        assigned[dataset.canonical_identity] = _DATASET_COLOR_PALETTE[index]
    return assigned


def _resolve_role(field_name: str, role: FieldSemanticRole | None) -> FieldSemanticRole:
    """Resolve an explicit or name-inferred semantic role."""
    if role is not None:
        if role not in _ROLE_SEMANTICS:
            message = f"Unsupported field semantic role {role!r}."
            raise ValueError(message)
        return role
    stripped = field_name.strip()
    normalized = stripped.lower()
    if stripped == "U" or normalized in {
        "u_mag",
        "velocity_magnitude",
        "speed",
        "|u|",
    }:
        return "velocity_magnitude"
    if normalized in {"u", "v", "velocity_x", "velocity_y"} or "residual" in normalized:
        return "signed_velocity"
    if normalized in {"t", "temperature"} or "temperature" in normalized:
        return "temperature"
    if "humidity" in normalized or "vapour" in normalized or "vapor" in normalized or normalized.startswith("omega"):
        return "humidity"
    if normalized in {"phi", "w_surf", "w_int"} or "moisture" in normalized:
        return "granular_moisture"
    if normalized in {"p", "pressure"} or "pressure" in normalized:
        return "pressure"
    return "scalar"


def _dataset_rank(dataset: DatasetVisualIdentity) -> tuple[bytes, str, str]:
    """Return an order-independent deterministic collision-resolution rank."""
    token = f"{dataset.canonical_identity}\x00{dataset.label}".encode()
    return sha256(token).digest(), dataset.canonical_identity, dataset.label


def _stable_palette_index(dataset: DatasetVisualIdentity, palette_size: int) -> int:
    """Map one stable identity and label token to an initial palette position."""
    digest = _dataset_rank(dataset)[0]
    return int.from_bytes(digest[:8], byteorder="big") % palette_size


def _require_text(value: str, *, label: str) -> None:
    """Require non-empty text without interpreting canonical identity structure."""
    if not isinstance(value, str):
        message = f"{label} must be text."
        raise TypeError(message)
    if not value or value != value.strip():
        message = f"{label} must be non-empty text without surrounding whitespace."
        raise ValueError(message)


_ROLE_SEMANTICS: dict[FieldSemanticRole, FieldVisualSemantics] = {
    "temperature": FieldVisualSemantics("inferno"),
    "humidity": FieldVisualSemantics("Blues"),
    "granular_moisture": FieldVisualSemantics("YlGnBu"),
    "pressure": FieldVisualSemantics("viridis"),
    "scalar": FieldVisualSemantics("cividis"),
    "signed_velocity": FieldVisualSemantics("RdBu_r", centered=True),
    "velocity_magnitude": FieldVisualSemantics("viridis"),
    "signed_error": FieldVisualSemantics("RdBu_r", centered=True),
    "absolute_error": FieldVisualSemantics("Reds"),
}
