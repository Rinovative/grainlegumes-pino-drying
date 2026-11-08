"""Render wall-clock timestamps and durations for human-readable log lines.

Responsibilities:
  - Render one instant as ISO-8601 local time with its UTC offset, to the second
  - Render one wall duration as ``12 s``, ``3 min 05 s`` or ``1 h 02 min``

Design principles:
  - One format for every Generation, Dataset and ML console line
  - Pure functions of their input; the current instant is read only when none is given

This module does NOT:
  - Format timestamps persisted in records. Their owners keep UTC ISO-8601
  - Parse, round for scientific use, or measure time
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

_SECONDS_PER_MINUTE: Final = 60
_SECONDS_PER_HOUR: Final = 3600


def format_timestamp(moment: datetime | None = None) -> str:
    """Return ``moment`` (default now; a naive value counts as local) as local ISO-8601 with offset, e.g. ``2026-11-02T14:11:07+01:00``."""
    return (moment or datetime.now(UTC)).astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    """Return one non-negative wall duration as ``12 s``, ``3 min 05 s`` or ``1 h 02 min``."""
    whole = max(0, round(seconds))
    if whole < _SECONDS_PER_MINUTE:
        return f"{whole} s"
    if whole < _SECONDS_PER_HOUR:
        return f"{whole // _SECONDS_PER_MINUTE} min {whole % _SECONDS_PER_MINUTE:02d} s"
    return f"{whole // _SECONDS_PER_HOUR} h {whole % _SECONDS_PER_HOUR // _SECONDS_PER_MINUTE:02d} min"
