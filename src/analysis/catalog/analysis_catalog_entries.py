"""Declare the plot-catalog entry contract, the base types, and the per-process scratch folder of the synthetic inputs.

Responsibilities:
  - Define one catalog plot entry: key (which names its base type), sub-type title, real builders, and the function that draws it
  - Fix the base types and their tab order in the plot-standard notebook
  - Provide one scratch folder per process for synthetic inputs that must live on disk (npz-backed artifact frames)

Design principles:
  - An entry draws through its real plot builders on synthetic inputs, never through a copy of them
  - The scratch folder belongs to the process that created it: a forked child gets its own and never removes its parent's

This module does NOT:
  - Hold entries or synthetic data; the group modules own them
  - Build the notebook panel; the panel module does
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

BASE_TYPES: Final = (
    "map-grid",
    "map-plus-lines",
    "time-series",
    "line-relationship",
    "distribution",
    "scatter-parity",
    "bar-compare",
    "matrix",
    "spectral",
    "table",
    "own",
)
_ENTRY_DIGITS: Final = 2
_SCRATCH: dict[int, Path] = {}  # scratch folder per process id


@dataclass(frozen=True, slots=True)
class CatalogPlot:
    """
    One plot-catalog entry.

    Attributes
    ----------
    key : str
        ``<base type>_<NN>`` with a base type of ``BASE_TYPES``, unique over the catalog; the base type names the tab.
    title : str
        The function sub-type the entry shows, as its view title.
    builders : tuple[str, ...]
        Every real builder ``draw`` calls, each ``<module>.<function>``, as the docs table names them.
    draw : Callable[[], object]
        Draws the entry: a Matplotlib figure, a table (DataFrame, Styler, HTML), or a tuple of them.

    """

    key: str
    title: str
    builders: tuple[str, ...]
    draw: Callable[[], object]

    def __post_init__(self) -> None:
        """Refuse a key outside the base types or without its two-digit number, and an entry without a builder."""
        base_type, _, number = self.key.rpartition("_")
        if base_type not in BASE_TYPES or not (number.isdigit() and len(number) == _ENTRY_DIGITS):
            message = f"Catalog key {self.key!r} must be '<base type>_<NN>' with a base type of {BASE_TYPES}."
            raise ValueError(message)
        if not self.builders:
            message = f"Catalog entry {self.key!r} names no builder."
            raise ValueError(message)

    @property
    def base_type(self) -> str:
        """Return the base type the key names."""
        return self.key.rpartition("_")[0]


def _remove_scratch(owner: int, folder: Path) -> None:
    """Remove ``folder`` at exit, but only in the process that created it."""
    if os.getpid() == owner:
        shutil.rmtree(folder, ignore_errors=True)


def scratch_directory() -> Path:
    """Return this process's scratch folder for synthetic inputs on disk, created on first use and removed at its exit."""
    owner = os.getpid()
    folder = _SCRATCH.get(owner)
    if folder is None:
        folder = Path(tempfile.mkdtemp(prefix=f"plot-catalog-{owner}-"))
        _SCRATCH[owner] = folder
        atexit.register(_remove_scratch, owner, folder)
    return folder
