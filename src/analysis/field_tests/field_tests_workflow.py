"""Orchestrate one field test's analysis in two calls: freeze the tolerances, then build the verdicts.

Responsibilities:
  - Freeze the tolerance table of a stage-A field test from the named trained runs of the roster's task
  - Build the verdicts of a field test against its frozen table (stage A's table in stage B)

Design principles:
  - Thin: task dispatch, run iteration and the analysis file location only; the services own every rule
  - A field test is named by task and test id and resolved through the storage path authority, never by a directory
  - The tolerance freeze reads only the roster before any member result is read; stage B never freezes a table

This module does NOT:
  - Choose the trained runs, define endpoints or apply verdict rules
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

from src import common, generation

from . import field_tests_tolerances as tolerances
from . import field_tests_verdicts as verdicts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def freeze_field_test_tolerances(
    task: str, test_id: str, *, run_dirs: Sequence[Path | str], storage_root: Path | str | None = None, device: str = "cpu"
) -> str:
    """
    Build the tolerance table from the runs' validation predictions and freeze it in a stage-A field test's analysis folder.

    Transient runs roll out every validation case on the full grid, which needs a Slurm allocation (CPU or GPU) for
    the demo populations; steady runs need one forward per validation case.

    Returns
    -------
    str
        SHA-256 of the frozen ``analysis/tolerances.json``.

    Raises
    ------
    ValueError
        If no run is named or the field test is a stage-B confirmation, which is judged against stage A's table.

    """
    if not run_dirs:
        msg = "The yardstick needs at least one trained run."
        raise ValueError(msg)
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    roster = generation.field_tests.roster.load_roster(directory)
    if roster["stage"] != "A":
        msg = f"Field test {test_id!r} is a stage-B confirmation: it is judged against stage A's frozen table and never freezes one."
        raise ValueError(msg)
    produce = tolerances.steady_validation_pairs if roster["task"] == "steady_flow" else tolerances.transient_validation_pairs
    pairs = chain.from_iterable(produce(run_dir, storage_root=storage_root, device=device) for run_dir in run_dirs)
    table = tolerances.build_tolerance_table(roster["task"], pairs)
    return tolerances.freeze_tolerances(generation.field_tests.record.record_path(directory, "analysis", tolerances.TOLERANCES_FILENAME), table)


def judge_field_test(task: str, test_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Build and publish ``verdicts.json``, ``verdicts.md`` and the summary figure of one field test."""
    return verdicts.build_verdicts(task, test_id, storage_root=storage_root)
