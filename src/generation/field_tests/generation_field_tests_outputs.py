"""Read the datasets and the completion evidence of one Source-format case.h5 for the field-test gate and analysis.

Responsibilities:
  - Return one required dataset or one named-field dataset of a case.h5, refusing a missing one with ``KeyError``
  - Return a transient case's completion evidence: exact stop time, final dry-mass fraction above target, its limit,
    the configured horizon and the reached flag

Design principles:
  - One reader owns the reached rule (final fraction at or below ``f_wet_dm_max``) for the gate comparison and the
    analysis; the analysis endpoint code digest covers this module
  - Read only; the Generation writer owns and validates the case.h5 format

This module does NOT:
  - Compare cases, measure endpoints, or validate a case.h5 beyond the members it reads
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class CaseCompletion:
    """Completion evidence of one transient case.h5."""

    t_final: float
    wet_fraction_final: float
    fraction_limit: float
    horizon: float
    reached: bool


def dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required dataset of a case.h5."""
    item = handle.get(name)
    if not isinstance(item, h5py.Dataset):
        message = f"{handle.filename} lacks the dataset {name!r}."
        raise KeyError(message)
    return item


def named_values(handle: h5py.File, name: str, *, axis: int = 0) -> dict[str, np.ndarray]:
    """Return one named-field dataset as float64 arrays by its JSON ``field_names``, which index ``axis``."""
    item = dataset(handle, name)
    values = np.moveaxis(np.asarray(item[()], dtype=np.float64), axis, 0)
    return dict(zip(json.loads(str(item.attrs["field_names"])), values, strict=True))


def completion(handle: h5py.File) -> CaseCompletion | None:
    """Return the completion evidence of a transient case.h5, or None for a case without a final status (steady)."""
    if not isinstance(handle.get("final_status/values"), h5py.Dataset):
        return None
    final = named_values(handle, "final_status/values")
    raw = dataset(handle, "provenance/scientific_config_json")[()]
    config = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    limit, wet = float(config["scientific_fixed_values"]["f_wet_dm_max"]), float(final["f_wet_dm_final"])
    return CaseCompletion(float(final["t_final"]), wet, limit, float(config["time"]["stop"]), wet <= limit)
