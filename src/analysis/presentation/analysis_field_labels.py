"""Resolve concise formula symbols and unit-bearing scientific field labels in one switchable style.

Responsibilities:
  - Map canonical field keys to robust plain-text and Matplotlib symbols and to names, from one table
  - Own the label style (symbol, name, or name and symbol) and compose field and metric labels in it at call time
  - Normalize dimensionless units, unit exponents and the products of compound units for scientific presentation
  - Own the display conversions by unit and quantity: absolute Kelvin to Celsius, and the named relative-humidity and
    wet-basis fractions to percent with their differences in percentage points
  - Provide one human-readable fallback for undeclared formula symbols

Design principles:
  - Task and profile schemas remain authoritative for physical units
  - Unit "1" alone never selects percent: only a named fraction quantity does, so other dimensionless values stay unchanged
  - Formula symbols are presentation metadata and never become field identity
  - Plain widget labels and mathtext plot labels share one field inventory
  - Plot labels follow the active style; control and table texts (``control=True``) keep symbols and name the percent basis

This module does NOT:
  - Infer field availability, task roles, or stored representations
  - Change scientific arrays, units, or canonical field names
  - Convert water densities, which need the dry bulk density (``analysis_moisture_display``)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

VELOCITY_MAGNITUDE_KEY: Final = "U"
VELOCITY_MAGNITUDE_LABEL: Final = "|u|"
VELOCITY_MAGNITUDE_MATHTEXT: Final = r"$|\mathbf{u}|$"
LabelStyle = Literal["symbol", "name", "name_symbol"]
_LABEL_STYLES: Final = ("symbol", "name", "name_symbol")
_ACTIVE_STYLE: Final[dict[str, LabelStyle]] = {"style": "symbol"}


@dataclass(frozen=True, slots=True)
class FieldDisplayMetadata:
    """Hold one field's plain symbol, mathtext symbol, name, and the wet-basis variant of a water density."""

    plain_symbol: str | None
    mathtext_symbol: str | None
    name: str
    wet_basis: FieldDisplayMetadata | None = None


_FIELD_DISPLAY: Final = {
    "Kxx": FieldDisplayMetadata("κₓₓ", r"$\kappa_{xx}$", "Permeability xx"),
    "Kxy": FieldDisplayMetadata("κₓᵧ", r"$\kappa_{xy}$", "Permeability xy"),
    "Kyx": FieldDisplayMetadata("κᵧₓ", r"$\kappa_{yx}$", "Permeability yx"),
    "Kyy": FieldDisplayMetadata("κᵧᵧ", r"$\kappa_{yy}$", "Permeability yy"),
    "K_min": FieldDisplayMetadata("κ_min", r"$\kappa_{\mathrm{min}}$", "Minimum principal permeability"),
    "K_max": FieldDisplayMetadata("κ_max", r"$\kappa_{\mathrm{max}}$", "Maximum principal permeability"),
    "K_anisotropy": FieldDisplayMetadata("κ_max/κ_min", r"$\kappa_{\mathrm{max}}/\kappa_{\mathrm{min}}$", "Anisotropy ratio"),
    "K_orientation": FieldDisplayMetadata("θ_κ", r"$\theta_{\kappa}$", "Major permeability direction"),
    "eps_bed": FieldDisplayMetadata("ε", r"$\varepsilon$", "Bed porosity"),
    "p_in_bc": FieldDisplayMetadata("p_in,bc", r"$p_{\mathrm{in,bc}}$", "Inlet pressure boundary"),
    "p": FieldDisplayMetadata("p", r"$p$", "Pressure"),
    "u": FieldDisplayMetadata("u", r"$u$", "Horizontal velocity"),
    "v": FieldDisplayMetadata("v", r"$v$", "Vertical velocity"),
    "U": FieldDisplayMetadata(VELOCITY_MAGNITUDE_LABEL, VELOCITY_MAGNITUDE_MATHTEXT, "Velocity magnitude"),
    "X_0_db_field": FieldDisplayMetadata("X₀,db", r"$X_{0,\mathrm{db}}$", "Initial dry-basis moisture"),
    "rho_bu_dry": FieldDisplayMetadata("\N{GREEK SMALL LETTER RHO}_bu,dry", r"$\rho_{\mathrm{bu,dry}}$", "Dry bulk density"),
    "T": FieldDisplayMetadata("T", r"$T$", "Temperature"),
    "phi": FieldDisplayMetadata("φ", r"$\varphi$", "Relative humidity"),
    "phi_in_bc": FieldDisplayMetadata("φ_in,bc", r"$\varphi_{\mathrm{in,bc}}$", "Inlet relative humidity"),
    "phi_eq": FieldDisplayMetadata("φ_eq", r"$\varphi_{\mathrm{eq}}$", "Equilibrium relative humidity"),
    # A water density shows as wet-basis moisture where the dry bulk density is known, else as itself in kg/m³.
    "w_surf": FieldDisplayMetadata(
        "w_surf", r"$w_{\mathrm{surf}}$", "Surface water content", FieldDisplayMetadata("X_wb,surf", r"$X_{\mathrm{wb,surf}}$", "Surface moisture")
    ),
    "w_int": FieldDisplayMetadata(
        "w_int", r"$w_{\mathrm{int}}$", "Internal water content", FieldDisplayMetadata("X_wb,int", r"$X_{\mathrm{wb,int}}$", "Internal moisture")
    ),
    "w_gr": FieldDisplayMetadata(
        "w_gr", r"$w_{\mathrm{gr}}$", "Grain water content", FieldDisplayMetadata("X_wb", r"$X_{\mathrm{wb}}$", "Grain moisture")
    ),
    "w_gr0": FieldDisplayMetadata("w_gr,0", r"$w_{\mathrm{gr},0}$", "Initial water content"),
    "T_in_bc": FieldDisplayMetadata("T_in,bc", r"$T_{\mathrm{in,bc}}$", "Inlet temperature"),
    "omega_in_bc": FieldDisplayMetadata("ω_in,bc", r"$\omega_{\mathrm{in,bc}}$", "Inlet humidity ratio"),
    "T_amb": FieldDisplayMetadata("T_amb", r"$T_{\mathrm{amb}}$", "Ambient temperature"),
    # Case-global scalar material conditioning, with the symbols of docs/02_physical_model.md.
    "r_surf_0": FieldDisplayMetadata("r_surf,0", r"$r_{\mathrm{surf},0}$", "Surface drying rate"),
    "r_int_surf": FieldDisplayMetadata("r_int/r_surf", r"$r_{\mathrm{int}}/r_{\mathrm{surf}}$", "Interior-to-surface rate ratio"),
    "f_surf": FieldDisplayMetadata("f_s", r"$f_{\mathrm{s}}$", "Surface water fraction"),
    "A_osw": FieldDisplayMetadata("A", r"$A$", "Oswin coefficient A"),
    "B_osw": FieldDisplayMetadata("B", r"$B$", "Oswin coefficient B"),
    "C_osw": FieldDisplayMetadata("C", r"$C$", "Oswin exponent C"),
    "k_gr": FieldDisplayMetadata("k_gr", r"$k_{\mathrm{gr}}$", "Grain thermal conductivity"),
    "cp_gr_dry": FieldDisplayMetadata("c_p,gr,dry", r"$c_{p,\mathrm{gr,dry}}$", "Dry grain heat capacity"),
}


def set_label_style(style: LabelStyle) -> None:
    """Select the style of every plot quantity label: ``symbol`` (default), ``name``, or ``name_symbol``."""
    if style not in _LABEL_STYLES:
        message = f"Label style must be one of {_LABEL_STYLES}, got {style!r}."
        raise ValueError(message)
    _ACTIVE_STYLE["style"] = style


def label_style() -> LabelStyle:
    """Return the active label style."""
    return _ACTIVE_STYLE["style"]


def field_display_metadata(field: str) -> FieldDisplayMetadata:
    """Return declared symbol metadata or a concise key-derived fallback."""
    if not isinstance(field, str):
        message = "Scientific field keys must be text."
        raise TypeError(message)
    if not field or field != field.strip():
        message = "Scientific field keys must be non-empty without surrounding whitespace."
        raise ValueError(message)
    words = field.replace("_", " ")
    return _FIELD_DISPLAY.get(field, FieldDisplayMetadata(None, None, words[:1].upper() + words[1:]))


def has_declared_field_metadata(field: str) -> bool:
    """Return whether a field has an explicit shared presentation declaration."""
    field_display_metadata(field)
    return field in _FIELD_DISPLAY


QuantityKind = Literal["absolute", "difference"]
RELATIVE_HUMIDITY_UNIT: Final = "% RH"
WET_BASIS_PERCENT_UNIT: Final = "% w.b."
PERCENTAGE_POINT_UNIT: Final = "pp"
_PERCENT: Final = 100.0
# Fractions stored with unit "1" that every display shows in percent; differences of them are percentage points.
_PERCENT_UNITS: Final = MappingProxyType(
    {
        "phi": RELATIVE_HUMIDITY_UNIT,
        "phi_in_bc": RELATIVE_HUMIDITY_UNIT,
        "phi_eq": RELATIVE_HUMIDITY_UNIT,
        "X_target_wb": WET_BASIS_PERCENT_UNIT,
        "target_moisture_wb": WET_BASIS_PERCENT_UNIT,
        "final_bulk_moisture_wb": WET_BASIS_PERCENT_UNIT,
    }
)
_TEMPERATURE_UNITS: Final = frozenset({"K", "degC", "°C"})
_UNIT_EXPONENT: Final = re.compile(r"\^(-?\d+)")
_SUPERSCRIPTS: Final = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _validate_quantity_kind(quantity_kind: QuantityKind) -> None:
    """Require one maintained absolute-or-difference display contract."""
    if quantity_kind not in {"absolute", "difference"}:
        message = "Display quantity kind must be 'absolute' or 'difference'."
        raise ValueError(message)


def _percent_unit(field: str | None, unit: str) -> str | None:
    """Return the percent token of a fraction quantity shown in percent, ``None`` for every other quantity."""
    percent = None if field is None else _PERCENT_UNITS.get(field)
    if percent is not None and unit != "1":
        message = f"{field!r} is shown in percent and must be stored as a fraction with unit '1', got {unit!r}."
        raise ValueError(message)
    return percent


def shown_in_percent(field: str) -> bool:
    """Return whether the named quantity is a fraction that every display shows in percent."""
    return field in _PERCENT_UNITS


def percent_unit(basis_token: str, *, control: bool = False) -> str:
    """Return the percent token of a plot label: plain ``%`` where the symbol names the basis, else ``basis_token`` (``% RH``)."""
    return basis_token if control or label_style() == "name" else "%"


def display_unit(
    unit: str,
    *,
    field: str | None = None,
    quantity_kind: QuantityKind = "absolute",
    control: bool = False,
) -> str:
    """
    Return the concise presentation token for one authoritative unit of the quantity ``field``.

    A named relative-humidity or wet-basis fraction reads ``%`` (``% RH`` or ``% w.b.`` in the name style and for
    ``control`` texts), its difference ``pp``. Omit ``field`` only for a quantity that is no named field, such as a
    residual ratio or a correlation.
    """
    if not isinstance(unit, str):
        message = "Scientific field units must be text."
        raise TypeError(message)
    if not unit or unit != unit.strip():
        message = "Scientific field units must be non-empty without surrounding whitespace."
        raise ValueError(message)
    _validate_quantity_kind(quantity_kind)
    percent = _percent_unit(field, unit)
    if percent is not None:
        return PERCENTAGE_POINT_UNIT if quantity_kind == "difference" else percent_unit(percent, control=control)
    if unit in _TEMPERATURE_UNITS:
        return "°C"
    if unit == "1":
        return "-"
    return _UNIT_EXPONENT.sub(lambda match: match.group(1).translate(_SUPERSCRIPTS), unit).replace("*", "·")  # ``W/(m*K)`` reads ``W/(m·K)``


def display_values(
    values: Sequence[float] | np.ndarray,
    unit: str,
    *,
    field: str,
    quantity_kind: QuantityKind = "absolute",
) -> np.ndarray:
    """Return a display-only numeric copy: absolute Kelvin to Celsius, named fractions and their differences times 100."""
    display_unit(unit, field=field, quantity_kind=quantity_kind)
    array = np.array(values, dtype=np.float64, copy=True)
    if _percent_unit(field, unit) is not None:
        array *= _PERCENT
    elif unit == "K" and quantity_kind == "absolute":
        array -= 273.15
    return array


def field_label(field: str, *, mathtext: bool = False, wet_basis: bool = False, control: bool = False) -> str:
    """
    Return the symbol, name, or name and symbol of one field in the active style; an undeclared symbol falls back to the name.

    ``wet_basis`` selects the moisture variant of a water density; ``control`` keeps the symbol whatever the style.
    """
    if not isinstance(mathtext, bool):
        message = "mathtext must be boolean."
        raise TypeError(message)
    metadata = field_display_metadata(field)
    if wet_basis and metadata.wet_basis is not None:
        metadata = metadata.wet_basis
    symbol = metadata.mathtext_symbol if mathtext else metadata.plain_symbol
    style = "symbol" if control else label_style()
    if symbol is None or style == "name":
        return metadata.name
    return symbol if style == "symbol" else f"{metadata.name} {symbol}"


def metric_label(metric: str, field: str, *, mathtext: bool = False) -> str:
    """Return a metric of one field in the active style: ``nRMSE(T)``, ``Temperature nRMSE``, or ``Temperature nRMSE(T)``."""
    metadata = field_display_metadata(field)
    symbol = metadata.mathtext_symbol if mathtext else metadata.plain_symbol
    style = label_style()
    if symbol is None:
        return f"{metric}({metadata.name})" if style == "symbol" else f"{metadata.name} {metric}"
    return {"symbol": f"{metric}({symbol})", "name": f"{metadata.name} {metric}", "name_symbol": f"{metadata.name} {metric}({symbol})"}[style]


def field_label_with_unit(
    field: str,
    unit: str,
    *,
    mathtext: bool = False,
) -> str:
    """Return one field label in the active style with the display unit of its absolute values."""
    return f"{field_label(field, mathtext=mathtext)} [{display_unit(unit, field=field)}]"
