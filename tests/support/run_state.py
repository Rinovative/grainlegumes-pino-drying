"""Publish metadata-only existing-run state for admission tests.

Responsibilities:
  - Write the config, summary and placeholder files that read-only run admission inspects

Design principles:
  - Placeholders are empty files, so admission sees existing state without any trained weights

This module does NOT:
  - Train, validate checkpoints or allocate registry entries
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src import common, experiments

if TYPE_CHECKING:
    from pathlib import Path


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
            "schema_version": experiments.run.RUN_SUMMARY_SCHEMA_VERSION,
            "status": status,
            "completed_epoch": completed_epoch,
        },
    )
    common.paths.resolve_split_indices_path(run_dir).touch()
    common.paths.resolve_normalizer_path(run_dir).touch()
    common.paths.resolve_last_checkpoint_file(run_dir).touch()
    if include_best:
        common.paths.resolve_best_checkpoint_file(run_dir).touch()
