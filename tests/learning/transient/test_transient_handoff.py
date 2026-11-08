"""Protect the Stage-A handoff snapshot's single-copy checkpoint and its immutability.

Responsibilities:
  - Check that the published checkpoint shares A0's bytes on disk and survives A0's atomic replacement unchanged
  - Check that a failed hard link publishes nothing and never falls back to a copy

Design principles:
  - Minimal synthetic bytes and configuration exercise the real publication and validation owners

This module does NOT:
  - Restore checkpoint state or check the scientific compatibility of a handoff
"""

from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING, Any

import pytest

from src import common
from src.learning.transient import learning_transient_handoff as handoff

if TYPE_CHECKING:
    from pathlib import Path

_DEVICE = {"device_type": "cpu"}


def _config() -> dict[str, Any]:
    """Return the smallest configuration the compatibility payload admits."""
    return {
        "task": "transient_drying",
        "task_contract": {"digest": "t" * 64},
        "input_profile": "profile",
        "temporal": {"sampling": {}, "horizon": 2},
        "model": {"kind": "fno"},
        "loss": {},
        "optimizer": {},
        "scheduler": None,
        "scaling": {},
        "run": {"deterministic": True, "seed": 1},
        "data": {"train_dataset": "train", "additional_evaluation_datasets": []},
        "training": {},
    }


def _source_run(tmp_path: Path) -> Path:
    """Create an A0 directory holding the two files a handoff snapshots."""
    run_dir = tmp_path / "a0"
    run_dir.mkdir()
    common.paths.resolve_best_checkpoint_file(run_dir).write_bytes(b"selected best state")
    common.paths.resolve_normalizer_path(run_dir).write_bytes(b"fitted scaling")
    return run_dir


def _publish(run_dir: Path) -> dict[str, Any]:
    """Publish the snapshot with placeholder identity evidence."""
    return handoff.publish_stage_a_handoff(
        run_dir,
        source_run_name="a0",
        checkpoint_identity={"effective_config_digest": "c" * 64},
        best_epoch=1,
        global_step=0,
        scaling_semantic_digest="s" * 64,
        task_contract_digest="t" * 64,
        tensorizer_digest="z" * 64,
        model_kind="fno",
        input_profile="profile",
        device=_DEVICE,
        config=_config(),
    )


def test_handoff_checkpoint_shares_a0_bytes_and_survives_its_atomic_replacement(tmp_path: Path) -> None:
    """One copy on disk, and a later atomic publication of A0's best leaves the snapshot valid and unchanged."""
    run_dir = _source_run(tmp_path)
    best = common.paths.resolve_best_checkpoint_file(run_dir)
    published = run_dir / handoff.HANDOFF_DIRECTORY / "best_checkpoint.pt"

    _publish(run_dir)

    assert best.samefile(published)
    common.serialization.atomic_write_bytes(best, b"a later best state")
    assert published.read_bytes() == b"selected best state"
    assert not best.samefile(published)
    handoff.validate_stage_a_handoff(run_dir / handoff.HANDOFF_DIRECTORY, target_config=_config(), device=_DEVICE, expected_source_run_name="a0")


def test_failed_hard_link_publishes_nothing_and_does_not_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem that refuses the link fails the publication closed instead of copying 1 GB."""
    run_dir = _source_run(tmp_path)

    def refuse(_source: object, _destination: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", refuse)

    with pytest.raises(OSError, match="cross-device"):
        _publish(run_dir)

    assert sorted(path.name for path in run_dir.iterdir()) == ["best_checkpoint.pt", "normalizer.pt"]
