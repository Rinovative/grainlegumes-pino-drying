"""
analysis_field_labels.py

Resolve concise formula symbols and unit-bearing scientific field labels.

Responsibilities:
  - Map canonical field keys to robust plain-text and Matplotlib symbols
  - Normalize dimensionless units for scientific presentation
  - Provide one human-readable fallback for undeclared formula symbols

Design principles:
  - Task and profile schemas remain authoritative for physical units
  - Formula symbols are presentation metadata and never become field identity
  - Plain widget labels and mathtext plot labels share one field inventory

This module does NOT:
  - Infer field availability, task roles, or stored representations
  - Change scientific arrays, units, or canonical field names
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

VELOCITY_MAGNITUDE_KEY: Final = "U"
VELOCITY_MAGNITUDE_LABEL: Final = "|u|"
VELOCITY_MAGNITUDE_MATHTEXT: Final = r"$|\mathbf{u}|$"


@dataclass(frozen=True, slots=True)
class FieldDisplayMetadata:
    """Hold one field's plain symbol, mathtext symbol, and fallback name."""

    plain_symbol: str | None
    mathtext_symbol: str | None
    fallback_name: str


_FIELD_DISPLAY: Final = {
    "Kxx": FieldDisplayMetadata("κₓₓ", r"$\kappa_{xx}$", "Permeability xx"),
    "Kxy": FieldDisplayMetadata("κₓᵧ", r"$\kappa_{xy}$", "Permeability xy"),
    "Kyy": FieldDisplayMetadata("κᵧᵧ", r"$\kappa_{yy}$", "Permeability yy"),
    "eps_bed": FieldDisplayMetadata("ε", r"$\varepsilon$", "Bed porosity"),
    "p_in_bc": FieldDisplayMetadata("p_in,bc", r"$p_{\mathrm{in,bc}}$", "Inlet pressure boundary"),
    "p": FieldDisplayMetadata("p", r"$p$", "Pressure"),
    "u": FieldDisplayMetadata("u", r"$u$", "Flow-direction velocity"),
    "v": FieldDisplayMetadata("v", r"$v$", "Cross-stream velocity"),
    "U": FieldDisplayMetadata(VELOCITY_MAGNITUDE_LABEL, VELOCITY_MAGNITUDE_MATHTEXT, "Velocity magnitude"),
    "X_0_db_field": FieldDisplayMetadata("X₀,db", r"$X_{0,\mathrm{db}}$", "Initial dry-basis moisture"),
    "rho_bu_dry": FieldDisplayMetadata("\N{GREEK SMALL LETTER RHO}_bu,dry", r"$\rho_{\mathrm{bu,dry}}$", "Dry bulk density"),
    "T": FieldDisplayMetadata("T", r"$T$", "Temperature"),
    "phi": FieldDisplayMetadata("φ", r"$\varphi$", "Relative humidity"),
    "w_surf": FieldDisplayMetadata("w_surf", r"$w_{\mathrm{surf}}$", "Surface moisture"),
    "w_int": FieldDisplayMetadata("w_int", r"$w_{\mathrm{int}}$", "Internal moisture"),
    "w_gr": FieldDisplayMetadata("w_gr", r"$w_{\mathrm{gr}}$", "Grain moisture"),
    "T_in_bc": FieldDisplayMetadata("T_in,bc", r"$T_{\mathrm{in,bc}}$", "Inlet temperature"),
    "omega_in_bc": FieldDisplayMetadata("ω_in,bc", r"$\omega_{\mathrm{in,bc}}$", "Inlet humidity ratio"),
    "T_amb": FieldDisplayMetadata("T_amb", r"$T_{\mathrm{amb}}$", "Ambient temperature"),
}


def field_display_metadata(field: str) -> FieldDisplayMetadata:
    """Return declared symbol metadata or a concise key-derived fallback."""
    if not isinstance(field, str):
        message = "Scientific field keys must be text."
        raise TypeError(message)
    if not field or field != field.strip():
        message = "Scientific field keys must be non-empty without surrounding whitespace."
        raise ValueError(message)
    return _FIELD_DISPLAY.get(
        field,
        FieldDisplayMetadata(None, None, field.replace("_", " ")),
    )


def has_declared_field_metadata(field: str) -> bool:
    """Return whether a field has an explicit shared presentation declaration."""
    field_display_metadata(field)
    return field in _FIELD_DISPLAY


TemperatureQuantityKind = Literal["absolute", "difference"]
_TEMPERATURE_UNITS: Final = frozenset({"K", "degC", "°C"})


def _validate_quantity_kind(quantity_kind: TemperatureQuantityKind) -> None:
    """Require one maintained absolute-or-difference display contract."""
    if quantity_kind not in {"absolute", "difference"}:
        message = "Temperature quantity kind must be 'absolute' or 'difference'."
        raise ValueError(message)


def display_unit(
    unit: str,
    *,
    quantity_kind: TemperatureQuantityKind = "absolute",
) -> str:
    """Return the concise presentation token for one authoritative unit."""
    if not isinstance(unit, str):
        message = "Scientific field units must be text."
        raise TypeError(message)
    if not unit or unit != unit.strip():
        message = "Scientific field units must be non-empty without surrounding whitespace."
        raise ValueError(message)
    _validate_quantity_kind(quantity_kind)
    if unit in _TEMPERATURE_UNITS:
        return "°C"
    return "-" if unit == "1" else unit


def display_values(
    values: Sequence[float] | np.ndarray,
    unit: str,
    *,
    quantity_kind: TemperatureQuantityKind = "absolute",
) -> np.ndarray:
    """Return a display-only numeric copy, converting absolute Kelvin to Celsius."""
    display_unit(unit, quantity_kind=quantity_kind)
    array = np.array(values, dtype=np.float64, copy=True)
    if unit == "K" and quantity_kind == "absolute":
        array -= 273.15
    return array


def field_label(field: str, *, mathtext: bool = False) -> str:
    """Return one formula symbol or the concise human-readable fallback."""
    if not isinstance(mathtext, bool):
        message = "mathtext must be boolean."
        raise TypeError(message)
    metadata = field_display_metadata(field)
    symbol = metadata.mathtext_symbol if mathtext else metadata.plain_symbol
    return metadata.fallback_name if symbol is None else symbol


def field_label_with_unit(
    field: str,
    unit: str,
    *,
    mathtext: bool = False,
) -> str:
    """Return one formula-or-fallback label with its authoritative unit."""
    return f"{field_label(field, mathtext=mathtext)} [{display_unit(unit)}]"
