"""Define shared semantic colormaps, dataset colors, and one fixed series style per entity for analysis views.

Responsibilities:
  - Resolve a field's semantic colormap and centered-scale requirement
  - Keep plot normalization separate from semantic color selection
  - Describe dataset identities used for deterministic presentation colors
  - Assign colorblind-friendly colors independently of selected dataset order
  - Fix one color per material and per architecture, one line style per evaluation role, and one marker per material
  - Resolve the color, line style, and marker of one plotted series from its architecture, material, and role

Design principles:
  - Explicit semantic roles take precedence over field-name inference
  - Canonical identities provide stable color assignment across EDA and evaluation
  - Colormap selection communicates field semantics rather than plot ownership
  - Color encodes the architecture where architectures are compared, else the material, else the pooled role, never several at once
  - A pooled role color is chosen far from every material color and is not grey, so it reads as a different entity

This module does NOT:
  - Construct Matplotlib normalizers, colorbars, axes, or legends
  - Load scientific arrays or derive numerical error fields
  - Decide which entity a plot shows; plot owners pass the entity they draw
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Bind one saved model family to its display label and fixed color."""

    family: str
    label: str
    color: str


@dataclass(frozen=True, slots=True)
class SeriesStyle:
    """Hold the Matplotlib color, line style, and optional marker of one plotted series."""

    color: str
    linestyle: str
    marker: str | None


# Architectures in run-selector order; a physics-informed variant takes a lighter sibling of its network's hue.
# Hues shared with the material colors are safe because no plot colors architecture and material at once.
ARCHITECTURES: Final = (
    ArchitectureSpec("fno", "FNO", "#0072B2"),
    ArchitectureSpec("uno", "U-NO", "#D55E00"),
    ArchitectureSpec("rno", "RNO", "#009E73"),
    ArchitectureSpec("pi-fno", "PI-FNO", "#56B4E9"),
    ArchitectureSpec("pi-uno", "PI-U-NO", "#E69F00"),
)
# Okabe-Ito colors for the registered materials; the first three match the Generation demo.
MATERIAL_COLORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lentil": "#0072B2",
        "chickpea": "#D55E00",
        "kidney_bean": "#009E73",
        "field_pea": "#E69F00",
        "rapeseed": "#CC79A7",
        "sunflower_seed": "#56B4E9",
    }
)
# Line style is the evaluation role in both color modes, so the material marker carries the material next to an architecture color.
ROLE_LINESTYLES: Final[Mapping[str, str]] = MappingProxyType({"id": "-", "parameter_ood": "--", "material_ood": ":"})
# Pooled role colors keep a CIELAB distance of at least 39 to every material color and a chroma well above grey.
ROLE_COLORS: Final[Mapping[str, str]] = MappingProxyType({"id": "#5E3C99", "parameter_ood": "#7A7A00", "material_ood": "#9E0142"})
MATERIAL_MARKERS: Final[Mapping[str, str]] = MappingProxyType(
    {"lentil": "o", "chickpea": "s", "kidney_bean": "^", "field_pea": "D", "rapeseed": "v", "sunflower_seed": "P"}
)
_DEFAULT_LINESTYLE: Final = "-"


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


def material_color(material: str) -> str:
    """Return the fixed color of one registered material; unknown materials raise ``ValueError``."""
    return _lookup(MATERIAL_COLORS, material, kind="material")


def architecture_color(family: str) -> str:
    """Return the fixed color of one model family (``fno``, ``uno``, ``rno``, ``pi-fno``, ``pi-uno``)."""
    return _architecture(family).color


def architecture_label(family: str) -> str:
    """Return the display label of one model family, for example ``U-NO`` for ``uno``."""
    return _architecture(family).label


def series_style(
    *,
    architecture: str | None = None,
    material: str | None = None,
    role: str | None = None,
    pooled: bool = False,
) -> SeriesStyle:
    """
    Resolve the color, line style, and marker of one plotted series.

    Parameters
    ----------
    architecture : str, optional
        Model family when architectures are compared; it owns the color.
    material : str, optional
        Registered material; it owns the color without an architecture and the marker with one.
    role : str, optional
        Evaluation role (``id``, ``parameter_ood``, ``material_ood``); it owns the line style.
    pooled : bool, default False
        Series that pools the materials of ``role``; without an architecture it takes the role color.

    Returns
    -------
    SeriesStyle
        Color (architecture, else material, else pooled role), line style (role, solid without one),
        and marker (material, only next to an architecture color).

    Raises
    ------
    ValueError
        If an entity is unknown, a pooled series names a material, or nothing determines a color.

    """
    if pooled and material is not None:
        message = "A pooled series merges materials and cannot name one."
        raise ValueError(message)
    linestyle = _DEFAULT_LINESTYLE if role is None else _lookup(ROLE_LINESTYLES, role, kind="role")
    if architecture is not None:
        marker = None if material is None else _lookup(MATERIAL_MARKERS, material, kind="material")
        return SeriesStyle(architecture_color(architecture), linestyle, marker)
    if material is not None:
        return SeriesStyle(material_color(material), linestyle, None)
    if pooled and role is not None:
        return SeriesStyle(_lookup(ROLE_COLORS, role, kind="role"), linestyle, None)
    message = "A series needs an architecture, a material, or a pooled role to pick its color."
    raise ValueError(message)


def _lookup(table: Mapping[str, str], key: str, *, kind: str) -> str:
    """Return one fixed presentation value or name the known keys."""
    try:
        return table[key]
    except KeyError:
        message = f"Unknown {kind} {key!r}; expected one of {sorted(table)}."
        raise ValueError(message) from None


def _architecture(family: str) -> ArchitectureSpec:
    """Return the specification of one model family or name the known families."""
    for spec in ARCHITECTURES:
        if spec.family == family:
            return spec
    message = f"Unknown architecture family {family!r}; expected one of {[spec.family for spec in ARCHITECTURES]}."
    raise ValueError(message)


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
