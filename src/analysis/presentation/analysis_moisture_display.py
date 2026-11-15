"""Present grain-moisture water densities as wet-basis moisture content in percent.

Responsibilities:
  - Label the moisture fields shown as wet-basis moisture in the active label style, with their absolute unit and difference unit
  - Give every named field's displayed unit and channel-control text; a control names the percent basis (% w.b.), a plot
    label leaves it to the symbol
  - Convert water densities with the dry bulk density to wet-basis percent, masking unphysical cells
  - Reduce the cells valid in a given mask to one bulk wet-basis percent

Design principles:
  - The domain moisture conversions stay the only formulas; this module only scales them to percent and masks
  - Model output may hold negative or non-finite water: such cells become NaN and never raise or get clipped
  - Differences of wet-basis percent are percentage points, never percent
  - The wet-basis and percentage-point tokens are the shared field-label tokens, so every display spells them alike

This module does NOT:
  - Change the canonical w_* field labels or units, which views without the dry bulk density keep in kg/m^3
  - Convert normalized errors, metrics or persisted arrays
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from src import domain

from . import analysis_field_labels as field_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

ABSOLUTE_UNIT: Final = field_labels.WET_BASIS_PERCENT_UNIT
DIFFERENCE_UNIT: Final = field_labels.PERCENTAGE_POINT_UNIT
_PERCENT: Final = 100.0


def is_moisture_field(field: str) -> bool:
    """Return whether ``field`` is a water density shown as wet-basis moisture."""
    return field_labels.field_display_metadata(field).wet_basis is not None


def display_name(field: str, *, mathtext: bool = False, control: bool = False) -> str:
    """
    Return a water density's wet-basis label in the active style, else the field's shared label.

    ``control`` gives the plain moisture words in every style, and a ``control`` text keeps the symbol of every other
    field. The words start with a capital letter like every label.
    """
    wet_basis = field_labels.field_display_metadata(field).wet_basis
    if wet_basis is None:
        return field_labels.field_label(field, mathtext=mathtext, control=control)
    if control:
        return wet_basis.name
    return field_labels.field_label(field, mathtext=mathtext, wet_basis=True)


def moisture_unit(*, difference: bool, control: bool = False) -> str:
    """Return ``%`` (``% w.b.`` in the name style and for ``control`` texts) for wet-basis values and ``pp`` for differences."""
    return DIFFERENCE_UNIT if difference else field_labels.percent_unit(ABSOLUTE_UNIT, control=control)


def field_display_unit(field: str, unit: str, *, difference: bool = False, control: bool = False) -> str:
    """Return the displayed unit of one named field: the wet-basis percent or ``pp`` for a water density, else its field-aware token."""
    if is_moisture_field(field):
        return moisture_unit(difference=difference, control=control)
    return field_labels.display_unit(unit, field=field, quantity_kind="difference" if difference else "absolute", control=control)


def channel_label(field: str, label: str, unit: str) -> str:
    """Return a channel control's text: the plain moisture words or ``label``, with the displayed unit of the field's values."""
    return f"{display_name(field, control=True) if is_moisture_field(field) else label} [{field_display_unit(field, unit, control=True)}]"


def wet_basis_percent(water: Sequence[float] | np.ndarray, rho_bu_dry: np.ndarray) -> np.ndarray:
    """
    Return ``100 X_wb`` of water densities over a broadcast dry bulk density, NaN where the water is negative or non-finite.

    The dry bulk density is input evidence and must be finite and positive in every cell; the domain conversion raises otherwise.
    """
    values = np.asarray(water, dtype=np.float64)
    density = np.broadcast_to(np.asarray(rho_bu_dry, dtype=np.float64), values.shape)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(values) & (values >= 0.0)
    # Masked cells enter the domain conversion as dry so it still validates the density of every cell.
    converted = domain.moisture.wet_basis_moisture(np.where(valid, values, 0.0), np.ascontiguousarray(density))
    return np.where(valid, _PERCENT * converted, np.nan)


def bulk_wet_basis_percent(water: np.ndarray, rho_bu_dry: np.ndarray, valid: np.ndarray) -> float:
    """Return ``100 Σw/(Σρ_bu,dry + Σw)`` over the ``valid`` cells, NaN without any."""
    selected = np.asarray(valid, dtype=bool)
    if not selected.any():
        return float("nan")
    density = np.broadcast_to(np.asarray(rho_bu_dry, dtype=np.float64), selected.shape)
    return _PERCENT * domain.moisture.bulk_wet_basis_moisture(np.asarray(water, dtype=np.float64)[selected], density[selected])
