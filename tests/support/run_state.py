"""Publish metadata-only existing-run state for admission tests and simulate later training-code changes.

Responsibilities:
  - Write the config, summary and placeholder files that read-only run admission inspects
  - Replace one task's training projection, as if its training code had changed after a run recorded it

Design principles:
  - Placeholders are empty files, so admission sees existing state without any trained weights
  - Code drift is simulated through the declared projection only, so every owner that reads it sees the same change

This module does NOT:
  - Train, validate checkpoints or allocate registry entries
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src import common, experiments, learning

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def drift_training_code(monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    """Make the running training digest of ``task`` differ from every digest a run recorded before this call."""
    projections = learning.training.checkpoint.TRAINING_PROJECTIONS
    drifted = common.code_identity.CodeProjection(name=projections[task].name, files=("common/common_names.py",))
    monkeypatch.setitem(projections, task, drifted)


def write_lightweight_run_state(
    run_dir: Path,
    config: dict[str, Any],
    *,
    status: str,
    completed_epoch: int,
    include_best: bool = True,
) -> None:
    """Publish only metadata and placeholder state needed by read-only admission inspection."""
    run_dir.mkdir(parents=True)
    experiments.config.loader.save_yaml(config, common.paths.resolve_run_config_path(run_dir))
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {
            "status": status,
            "completed_epoch": completed_epoch,
        },
    )
    common.paths.resolve_split_indices_path(run_dir).touch()
    common.paths.resolve_normalizer_path(run_dir).touch()
    common.paths.resolve_last_checkpoint_file(run_dir).touch()
    if include_best:
        common.paths.resolve_best_checkpoint_file(run_dir).touch()
