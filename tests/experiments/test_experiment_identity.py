# ruff: noqa: S101
"""Protect concise labels and immutable transient parent experiment records."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from support import configs

from src.experiments import experiments_run_identity as identity
from src.experiments.config import experiments_config_transient_plan as transient_plan


def _plan(
    output_root: Path,
    *,
    revision: int = 0,
    seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return isolated current-schema Stage A0/B configs under one temporary root."""
    raw = configs.transient_two_stage_config(revision=revision, seed=seed)
    plan = transient_plan.resolve_transient_training_plan(raw)
    stage_a = copy.deepcopy(dict(plan.stage_a))
    stage_b = copy.deepcopy(dict(plan.stage_b))
    stage_a["paths"]["output_root"] = str(output_root)
    stage_b["paths"]["output_root"] = str(output_root)
    return stage_a, stage_b


def test_parent_label_distinguishes_dataset_and_run_revisions() -> None:
    """Omit zero revisions and place dN/rN before the stage suffix."""
    raw = configs.transient_two_stage_config(revision=1)
    raw["data"]["train_dataset"] = "transient_drying__family__id__0123456789abcdef"
    plan = transient_plan.resolve_transient_training_plan(raw)

    assert plan.stage_a["run"]["name"].endswith("_r1_a0")
    assert "transient_drying" not in plan.stage_a["run"]["name"]
    assert "0123456789abcdef" not in plan.stage_a["run"]["name"]
    assert "__" not in plan.stage_a["run"]["name"]


def test_parent_record_binds_flat_children_and_rejects_reuse(tmp_path: Path) -> None:
    """Publish one visible parent without copying or nesting child artifacts."""
    authored = configs.write_yaml(tmp_path / "synthetic_plan.yaml", configs.transient_two_stage_config())
    stage_a, stage_b = _plan(tmp_path)
    record = identity.build_transient_experiment_record(stage_a, stage_b, config_path=authored)
    path = identity.publish_transient_experiment_record(record, output_root=tmp_path)

    saved = identity.validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))
    assert path == tmp_path / "transient_drying" / "runs" / saved["parent_label"] / "experiment.json"
    assert saved["run_revision"] == 0
    assert "_r0" not in saved["parent_label"]
    assert saved["children"]["stage_a0"]["path"].endswith("_a0")
    assert saved["children"]["stage_b"]["path"].endswith("_b")
    assert Path(saved["children"]["stage_a0"]["path"]).parent == path.parent.parent
    assert not Path(saved["children"]["stage_a0"]["path"]).exists()
    with pytest.raises(FileExistsError, match="Matching experiment already exists"):
        identity.publish_transient_experiment_record(record, output_root=tmp_path)


def test_explicit_new_run_revision_publishes_a_distinct_parent(tmp_path: Path) -> None:
    """Accept an intentional nonzero revision as a distinct immutable experiment."""
    authored_zero = configs.write_yaml(tmp_path / "plan-r0.yaml", configs.transient_two_stage_config())
    stage_a_zero, stage_b_zero = _plan(tmp_path)
    record_zero = identity.build_transient_experiment_record(stage_a_zero, stage_b_zero, config_path=authored_zero)
    path_zero = identity.publish_transient_experiment_record(record_zero, output_root=tmp_path)

    authored_one = configs.write_yaml(tmp_path / "plan-r1.yaml", configs.transient_two_stage_config(revision=1))
    stage_a_one, stage_b_one = _plan(tmp_path, revision=1)
    record_one = identity.build_transient_experiment_record(stage_a_one, stage_b_one, config_path=authored_one)
    path_one = identity.publish_transient_experiment_record(record_one, output_root=tmp_path)

    assert path_zero != path_one
    assert "_r0" not in record_zero["parent_label"]
    assert record_one["parent_label"].endswith("_r1")
    assert record_one["run_revision"] == 1


def test_seed_is_preserved_and_distinguishes_parent_identity(tmp_path: Path) -> None:
    """Keep seed separate from both Dataset and run revisions."""
    stage_a_17, stage_b_17 = _plan(tmp_path, seed=17)
    stage_a_23, stage_b_23 = _plan(tmp_path, seed=23)

    record_17 = identity.build_transient_experiment_record(
        stage_a_17,
        stage_b_17,
        config_path=configs.write_yaml(tmp_path / "plan-s17.yaml", configs.transient_two_stage_config(seed=17)),
    )
    record_23 = identity.build_transient_experiment_record(
        stage_a_23,
        stage_b_23,
        config_path=configs.write_yaml(tmp_path / "plan-s23.yaml", configs.transient_two_stage_config(seed=23)),
    )

    assert record_17["seed"] == stage_a_17["run"]["seed"]
    assert record_23["seed"] == stage_a_23["run"]["seed"]
    assert stage_a_17["run"]["name"] != stage_a_23["run"]["name"]
    assert f"_s{record_17['seed']}_" in stage_a_17["run"]["name"]
    assert f"_s{record_23['seed']}_" in stage_a_23["run"]["name"]


def test_same_parent_revision_with_different_config_is_conflict(tmp_path: Path) -> None:
    """Reject a changed exact config at the same user-selected parent revision."""
    authored = configs.write_yaml(tmp_path / "synthetic_plan.yaml", configs.transient_two_stage_config())
    stage_a, stage_b = _plan(tmp_path)
    original = identity.build_transient_experiment_record(stage_a, stage_b, config_path=authored)
    identity.publish_transient_experiment_record(original, output_root=tmp_path)

    changed_a = copy.deepcopy(stage_a)
    changed_b = copy.deepcopy(stage_b)
    changed_a["data"]["batch_size"] += 1
    changed_b["data"]["batch_size"] += 1
    changed = identity.build_transient_experiment_record(changed_a, changed_b, config_path=authored)
    assert changed["parent_label"] == original["parent_label"]
    with pytest.raises(FileExistsError, match=r"Run revision conflict.*set a new explicit run.revision"):
        identity.publish_transient_experiment_record(changed, output_root=tmp_path)
