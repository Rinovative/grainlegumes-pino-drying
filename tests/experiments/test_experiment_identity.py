"""Protect content-derived run identity and transient parent provenance.

Responsibilities:
  - Check scientific identity, layout paths, and authored-source provenance separation
  - Check that resolution records the producer digests and that saved configs keep that evidence under code drift
  - Check parent-record binding of the matched A0/B/A1 children and of the persisted identity

Design principles:
  - The readable layout binds one scientific content per name while authored presentation remains provenance

This module does NOT:
  - Run comparison stages or create Dataset packages
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest
from support import configs, run_state

from src import common, learning
from src.experiments.config import experiments_config_loader as config_loader

if TYPE_CHECKING:
    from pathlib import Path

from src.experiments import experiments_run as run_service
from src.experiments import experiments_run_identity as identity


def _plan(
    output_root: Path,
    raw: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return isolated A0/B/A1 configs of one matched plan resolved under a temporary output root."""
    plan = configs.resolve_plan(configs.transient_matched_config() if raw is None else raw, output_root=output_root)
    return tuple(copy.deepcopy(dict(config)) for config in plan.stages)  # type: ignore[return-value]


def _record(
    stages: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    source: Path,
) -> dict[str, Any]:
    """Build the parent record of one resolved matched plan."""
    stage_a0, stage_b, stage_a1 = stages
    return identity.build_transient_experiment_record(stage_a0, stage_b, stage_a1=stage_a1, config_path=source)


def test_resolution_records_the_running_producer_digests_and_saved_configs_keep_them_under_code_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stamp producer evidence once, refuse authored, missing or tampered evidence, and keep a saved identity when code moves."""
    checkpoint = learning.training.checkpoint
    config = configs.resolve(configs.direct_config())
    assert config["producer"] == {
        "training_sha256": checkpoint.TRAINING_PROJECTIONS["steady_flow"].sha256(),
        "model_sha256": checkpoint.MODEL_PROJECTIONS["steady_flow"].sha256(),
    }
    authored = configs.direct_config()
    authored["producer"] = copy.deepcopy(config["producer"])
    with pytest.raises(config_loader.ConfigError, match="must not supply it"):
        configs.resolve(authored)
    missing = copy.deepcopy(config)
    del missing["producer"]
    with pytest.raises(config_loader.ConfigError, match=r"Missing: \['producer'\]"):
        config_loader.validate_resolved_config(missing, verify_layout_identity=False)
    for key in sorted(config["producer"]):
        tampered = copy.deepcopy(config)
        tampered["producer"][key] = "0" * 64
        with pytest.raises(config_loader.ConfigError, match="Initial run layout identity"):
            config_loader.validate_resolved_config(tampered)
        tampered["producer"][key] = "0" * 63
        with pytest.raises(config_loader.ConfigError, match="lowercase SHA-256 digests"):
            config_loader.validate_resolved_config(tampered, verify_layout_identity=False)

    with monkeypatch.context() as drift:
        run_state.drift_training_code(drift, "steady_flow")
        saved = config_loader.validate_resolved_config(copy.deepcopy(config))
        fresh = configs.resolve(configs.direct_config())
    assert saved["producer"] == config["producer"]
    assert saved["run"]["layout"] == config["run"]["layout"]
    # New training code is a new experiment; the model digest, and with it the scaling cache, is untouched.
    assert fresh["producer"]["model_sha256"] == config["producer"]["model_sha256"]
    assert fresh["producer"]["training_sha256"] != config["producer"]["training_sha256"]
    assert fresh["run"]["layout"]["a0_id"] != config["run"]["layout"]["a0_id"]


def test_parent_record_uses_content_layout_and_relative_children(tmp_path: Path) -> None:
    """Persist parent metadata in the pair directory and point to the shared A0 and the pair's leaves."""
    authored = configs.write_yaml(tmp_path / "synthetic_plan.yaml", configs.transient_matched_config())
    stages = _plan(tmp_path)
    record = _record(stages, authored)
    path = identity.admit_fresh_transient_experiment_record(record, output_root=tmp_path)
    saved = identity.validate_transient_experiment_record(json.loads(path.read_text(encoding="utf-8")))

    pair = stages[1]["run"]["layout"]["pair"]
    assert path == common.paths.resolve_config_experiment_dir(stages[0]) / pair / "experiment.json"
    assert path.parent == common.paths.resolve_config_run_dir(stages[1]).parent
    assert saved["layout"] == stages[1]["run"]["layout"]
    assert saved["topology"] == "a0_to_b_and_a1"
    assert {key: child["path"] for key, child in saved["children"].items()} == {"stage_a0": "../a0", "stage_b": "b", "stage_a1": "a1"}
    (path.parent / "b").mkdir()
    with pytest.raises(FileExistsError, match="Matching experiment already exists"):
        identity.admit_fresh_transient_experiment_record(record, output_root=tmp_path)


def test_readonly_child_navigation_binds_saved_identity_and_admits_training_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stages = _plan(tmp_path)
    stage_a = stages[0]
    authored = configs.write_yaml(tmp_path / "plan.yaml", configs.transient_matched_config())
    record = _record(stages, authored)
    marker = identity.admit_fresh_transient_experiment_record(record, output_root=tmp_path)
    child = record["children"]["stage_a0"]
    run_dir = common.paths.resolve_config_run_dir(stage_a)
    config_path = configs.write_yaml(run_dir / "config.yaml", stage_a)
    summary = {
        "schema_version": common.paths.RUN_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "config_sha256": common.serialization.file_sha256(config_path),
        "checkpoint_identity": {"resume_contract_digest": child["semantic_config_sha256"]},
    }
    common.serialization.atomic_write_json(run_dir / "summary.json", summary)
    monkeypatch.setattr(identity.training_checkpoint, "resume_contract_digest", lambda _config: pytest.fail("Navigation must use saved identity"))
    assert identity.resolve_readonly_experiment_child(record, child, marker=marker, output_root=tmp_path) == run_dir
    with monkeypatch.context() as drift:
        # Navigation is a historical read: it binds the recorded producer evidence, never the running code.
        run_state.drift_training_code(drift, "transient_drying")
        assert identity.resolve_readonly_experiment_child(record, child, marker=marker, output_root=tmp_path) == run_dir

    changed = copy.deepcopy(summary)
    changed["checkpoint_identity"]["resume_contract_digest"] = "0" * 64
    common.serialization.atomic_write_json(run_dir / "summary.json", changed)
    with pytest.raises(ValueError, match="immutable parent"):
        identity.resolve_readonly_experiment_child(record, child, marker=marker, output_root=tmp_path)
    changed["checkpoint_identity"] = {}
    common.serialization.atomic_write_json(run_dir / "summary.json", changed)
    with pytest.raises(ValueError, match="valid saved resume identity"):
        identity.resolve_readonly_experiment_child(record, child, marker=marker, output_root=tmp_path)
    common.serialization.atomic_write_json(run_dir / "summary.json", summary)
    raw = copy.deepcopy(stage_a)
    raw["tracking"]["wandb"]["mode"] = "offline"
    configs.write_yaml(config_path, raw)
    with pytest.raises(ValueError, match="config bytes"):
        identity.resolve_readonly_experiment_child(record, child, marker=marker, output_root=tmp_path)


def test_incomplete_child_navigation_does_not_admit_completed_evidence(tmp_path: Path) -> None:
    stages = _plan(tmp_path)
    stage_b = stages[1]
    authored = configs.write_yaml(tmp_path / "plan.yaml", configs.transient_matched_config())
    record = _record(stages, authored)
    marker = identity.admit_fresh_transient_experiment_record(record, output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(stage_b)
    configs.write_yaml(run_dir / "config.yaml", stage_b)
    common.serialization.atomic_write_json(
        run_dir / "summary.json",
        {
            "schema_version": common.paths.RUN_SUMMARY_SCHEMA_VERSION,
            "status": "budget_unmet",
            "run_identity": {"resolved_config_sha256": identity.resolved_config_digest(stage_b)},
        },
    )
    assert identity.resolve_readonly_experiment_child(record, record["children"]["stage_b"], marker=marker, output_root=tmp_path) == run_dir
    with pytest.raises(run_service.RunLifecycleError, match="incomplete"):
        run_service.validate_completed_run(run_dir)


def test_scientific_change_changes_content_identity_and_parent_location(tmp_path: Path) -> None:
    """Change the identities, not the readable parent location, when a scientific parameter changes."""
    original_raw = configs.transient_matched_config()
    changed_raw = copy.deepcopy(original_raw)
    changed_raw["data"]["batch_size"] += 1
    source = configs.write_yaml(tmp_path / "plan.yaml", original_raw)
    original = _record(_plan(tmp_path, original_raw), source)
    changed = _record(_plan(tmp_path, changed_raw), source)

    assert original["parent_identity_sha256"] != changed["parent_identity_sha256"]
    assert original["layout"]["a0_id"] != changed["layout"]["a0_id"]
    assert identity.experiment_record_path(original, output_root=tmp_path) == identity.experiment_record_path(changed, output_root=tmp_path)


def test_persisted_parent_must_match_the_requested_identity(tmp_path: Path) -> None:
    """An existing marker admits only the exact identity it recorded; a missing one is not created implicitly."""
    source = configs.write_yaml(tmp_path / "plan.yaml", configs.transient_matched_config())
    record = _record(_plan(tmp_path), source)

    with pytest.raises(FileNotFoundError, match="missing or unsafe"):
        identity.validate_persisted_transient_experiment_record(record, output_root=tmp_path)
    marker = identity.admit_fresh_transient_experiment_record(record, output_root=tmp_path)
    assert identity.validate_persisted_transient_experiment_record(record, output_root=tmp_path) == marker
    # A record at the same location whose scientific child identity differs, with its digest recomputed to stay self-consistent.
    other = copy.deepcopy(record)
    other["children"]["stage_b"]["semantic_config_sha256"] = "f" * 64
    payload = {key: value for key, value in other.items() if key not in {"parent_identity_sha256", "source_repository", "authored_config"}}
    other["parent_identity_sha256"] = common.serialization.canonical_json_sha256(payload)

    with pytest.raises(ValueError, match="persisted transient parent identity"):
        identity.validate_persisted_transient_experiment_record(other, output_root=tmp_path)


def test_authored_config_and_repository_provenance_do_not_change_parent_identity(tmp_path: Path) -> None:
    """Keep source path/hash and repository facts separate from scientific identity."""
    first_yaml = configs.write_yaml(tmp_path / "first.yaml", configs.transient_matched_config())
    second_yaml = configs.write_yaml(tmp_path / "second.yaml", configs.transient_matched_config())
    stages = _plan(tmp_path)
    first = _record(stages, first_yaml)
    second = _record(stages, second_yaml)

    assert first["authored_config"] != second["authored_config"]
    assert first["parent_identity_sha256"] == second["parent_identity_sha256"]


def test_pairs_with_other_budgets_share_the_seed_group_a0(tmp_path: Path) -> None:
    """A second B/A1 pair gets its own pair directory beside the one A0 of the seed group."""
    raw = configs.transient_matched_config()
    first = _plan(tmp_path, raw)
    source = configs.write_yaml(tmp_path / "plan.yaml", raw)
    marker = identity.admit_fresh_transient_experiment_record(_record(first, source), output_root=tmp_path)
    a0 = common.paths.resolve_config_run_dir(first[0])
    assert marker.parent.parent == a0.parent == common.paths.resolve_config_experiment_dir(first[0])
    assert first[0]["run"]["layout"]["pair"] is None
    raw["training"]["b"]["train_seconds"] = 2.0

    second = _plan(tmp_path, raw)

    assert common.paths.resolve_config_run_dir(second[0]) == a0
    assert second[0]["run"]["layout"]["a0_id"] == first[0]["run"]["layout"]["a0_id"]
    assert second[1]["run"]["layout"]["pair"] != first[1]["run"]["layout"]["pair"]
    second_record = _record(second, source)
    second_marker = identity.admit_fresh_transient_experiment_record(second_record, output_root=tmp_path)
    assert second_marker.parent.parent == marker.parent.parent
    assert second_record["children"]["stage_a0"]["path"] == "../a0"
    assert identity.load_readonly_transient_experiment_record(second_marker, experiments_root=tmp_path)["output_root"] == tmp_path
    for key, child in second_record["children"].items():
        located = identity.resolve_readonly_experiment_child(second_record, child, marker=second_marker, output_root=tmp_path)
        assert located == (a0 if key == "stage_a0" else second_marker.parent / key.removeprefix("stage_"))


def test_same_pair_with_other_content_conflicts_at_the_parent_record(tmp_path: Path) -> None:
    """One pair name binds one content: changed B settings under the same pair are refused, not renamed."""
    raw = configs.transient_matched_config()
    first = _plan(tmp_path, raw)
    source = configs.write_yaml(tmp_path / "plan.yaml", raw)
    identity.admit_fresh_transient_experiment_record(_record(first, source), output_root=tmp_path)
    changed = copy.deepcopy(raw)
    changed["training"]["b"]["evaluation_interval"] += 1

    second = _plan(tmp_path, changed)

    assert second[1]["run"]["layout"]["pair"] == first[1]["run"]["layout"]["pair"]
    with pytest.raises(FileExistsError, match="Scientific parent identity conflict"):
        identity.admit_fresh_transient_experiment_record(_record(second, source), output_root=tmp_path)
