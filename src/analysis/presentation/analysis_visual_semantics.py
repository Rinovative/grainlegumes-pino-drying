"""Define shared semantic colormaps and one fixed series style per entity for analysis views.

Responsibilities:
  - Resolve a field's semantic colormap and centered-scale requirement
  - Keep plot normalization separate from semantic color selection
  - Fix one color per material and per architecture, one line style (``LineStyle``) and one bar hatch per evaluation role, and one
    marker per material
  - Step the color of several runs of one architecture into shades (``architecture_shade``): plain networks darker, physics-informed
    ones lighter, at most ``SHADE_LIMIT`` apart
  - Resolve the color, line style, hatch, and marker of one plotted series from its architecture, material, and role

Design principles:
  - Explicit semantic roles take precedence over field-name inference
  - Fixed identities provide stable color assignment across EDA and evaluation, independent of selection order
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
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

# The Matplotlib line styles a series may take; a subset of Matplotlib's own style type, so plot calls accept it unchanged.
LineStyle = Literal["-", "--", ":", "-."]
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


@dataclass(frozen=True, slots=True)
class FieldVisualSemantics:
    """Describe the colormap and center requirement for one displayed field."""

    colormap: str
    centered: bool = False


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Bind one saved model family to its display label and fixed color."""

    family: str
    label: str
    color: str


@dataclass(frozen=True, slots=True)
class SeriesStyle:
    """Hold the Matplotlib color, line style, optional marker, and bar hatch of one plotted series."""

    color: str
    linestyle: LineStyle
    marker: str | None
    hatch: str = ""


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
ROLE_LINESTYLES: Final[Mapping[str, LineStyle]] = MappingProxyType({"id": "-", "parameter_ood": "--", "material_ood": ":"})
# Pooled role colors keep a CIELAB distance of at least 39 to every material color and a chroma well above grey.
ROLE_COLORS: Final[Mapping[str, str]] = MappingProxyType({"id": "#5E3C99", "parameter_ood": "#7A7A00", "material_ood": "#9E0142"})
# The bar analogue of the line style: bars of one role share its hatch.
ROLE_HATCHES: Final[Mapping[str, str]] = MappingProxyType({"id": "", "parameter_ood": "//", "material_ood": ".."})
MATERIAL_MARKERS: Final[Mapping[str, str]] = MappingProxyType(
    {"lentil": "o", "chickpea": "s", "kidney_bean": "^", "field_pea": "D", "rapeseed": "v", "sunflower_seed": "P"}
)
_DEFAULT_LINESTYLE: Final[LineStyle] = "-"
# Runs of one architecture that stay at least CIELAB 15 apart from every other shade and 25 from white; a third PI shade would
# either sit under 15 from its neighbour or fade into the background, so more runs switch the figure to run colors.
SHADE_LIMIT: Final = 2
# One qualitative color per run once an architecture has more runs than ``SHADE_LIMIT`` (Matplotlib's tab10, spelled out).
RUN_COLORS: Final = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)
_DARKEN_STEP: Final = 0.3  # share of the way to black per run of a plain network
_LIGHTEN_STEP: Final = 0.4  # share of the way to white per run of a physics-informed network


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


def material_color(material: str) -> str:
    """Return the fixed color of one registered material; unknown materials raise ``ValueError``."""
    return _lookup(MATERIAL_COLORS, material, kind="material")


def architecture_color(family: str) -> str:
    """Return the fixed color of one model family (``fno``, ``uno``, ``rno``, ``pi-fno``, ``pi-uno``)."""
    return _architecture(family).color


def architecture_shade(family: str, index: int, count: int) -> str:
    """
    Return the color of run ``index`` of ``count`` runs of one architecture: run 0 keeps its hue, later runs step away from it.

    A plain network steps darker and its physics-informed sibling (``pi-`` family) lighter, so the two shade sets never meet; every
    shade of every architecture keeps a CIELAB distance of at least 15 from the others and 25 from white. The shade of a run does not depend on
    ``count``. ``count`` above ``SHADE_LIMIT`` or an ``index`` outside ``count`` raises ``ValueError``.
    """
    if not 0 <= index < count:
        message = f"Run index {index} is outside {count} runs."
        raise ValueError(message)
    if count > SHADE_LIMIT:
        message = f"Plots tell at most {SHADE_LIMIT} runs of one architecture apart by shade; {count} were given."
        raise ValueError(message)
    base = architecture_color(family)
    lighter = family.startswith("pi-")
    fraction = index * (_LIGHTEN_STEP if lighter else _DARKEN_STEP)
    target = 255.0 if lighter else 0.0
    channels = (int(base[position : position + 2], 16) for position in (1, 3, 5))
    return "#" + "".join(f"{round(channel + (target - channel) * fraction):02X}" for channel in channels)


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
    Resolve the color, line style, hatch, and marker of one plotted series.

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
        marker (material, only next to an architecture color), and bar hatch (role, empty without one).

    Raises
    ------
    ValueError
        If an entity is unknown, a pooled series names a material, or nothing determines a color.

    """
    if pooled and material is not None:
        message = "A pooled series merges materials and cannot name one."
        raise ValueError(message)
    linestyle = _DEFAULT_LINESTYLE if role is None else _lookup(ROLE_LINESTYLES, role, kind="role")
    hatch = "" if role is None else _lookup(ROLE_HATCHES, role, kind="role")
    if architecture is not None:
        marker = None if material is None else _lookup(MATERIAL_MARKERS, material, kind="material")
        return SeriesStyle(architecture_color(architecture), linestyle, marker, hatch)
    if material is not None:
        return SeriesStyle(material_color(material), linestyle, None, hatch)
    if pooled and role is not None:
        return SeriesStyle(_lookup(ROLE_COLORS, role, kind="role"), linestyle, None, hatch)
    message = "A series needs an architecture, a material, or a pooled role to pick its color."
    raise ValueError(message)


def _lookup[V](table: Mapping[str, V], key: str, *, kind: str) -> V:
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
