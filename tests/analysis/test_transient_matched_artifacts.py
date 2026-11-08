"""Bind matched transient artifacts to the real A0 parent and B time receipt.

Responsibilities:
  - Check semantic parent digests, distinct A1 lineage, and live budget-receipt admission

Design principles:
  - Cached artifacts remain bound to immutable matched-compute evidence

This module does NOT:
  - Allocate A1 budgets or execute matched training
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml

from src import common
from src.analysis.artifacts import analysis_artifact_plan as artifact_plan
from src.analysis.artifacts import analysis_artifact_transient as artifact
from src.experiments import experiments_matched_budget as matched_budget
from src.experiments import experiments_run_identity as run_identity
from src.learning.transient.learning_transient_handoff import HANDOFF_DIRECTORY

if TYPE_CHECKING:
    from pathlib import Path

    from src.analysis.evaluation.evaluation_transient_artifact import (
        LoadedTransientSequenceArtifact,
        TransientSequenceArtifactIndex,
    )


def _matched_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    output_root = tmp_path / "outputs"
    pair = "b-a1__budget-12p5s__rollout-1__lr-1"
    group = output_root / "transient_drying" / "dataset" / "matched" / "fno_m2x2_h4_l1" / "seed-9"
    branch = group / pair
    runs = {"a0": group / "a0", "b": branch / "b", "a1": branch / "a1"}
    for path in runs.values():
        path.mkdir(parents=True)
    monkeypatch.setattr(
        artifact.learning.training.checkpoint,
        "resume_contract_digest",
        lambda config: common.serialization.canonical_json_sha256({"arm": config["training"]["comparison_arm"]}),
    )

    def config(arm: str) -> dict[str, Any]:
        return {
            "task": "transient_drying",
            "run": {
                "name": arm,
                "seed": 9,
                "layout": {
                    "task": "transient_drying",
                    "dataset": "dataset",
                    "family": "matched",
                    "config_name": "fno_m2x2_h4_l1",
                    "model_label": "fno_m2x2_h4_l1",
                    "seed": 9,
                    "revision": None,
                    "budget_epochs": None,
                    "stage": arm,
                    "pair": None if arm == "a0" else pair,
                    "trial": None,
                    "a0_id": "a" * 16,
                    "branch_id": None if arm == "a0" else "b" * 16,
                },
            },
            "paths": {"output_root": str(output_root)},
            "data": {
                "train_dataset": "dataset",
                "spatial_stride": 1,
                "additional_evaluation_datasets": [],
            },
            "input_profile": "profile",
            "temporal": {"temporal_conditioning": "time"},
            "model": {"kind": "fno", "params": {"n_modes": [2, 2], "hidden_channels": 4, "n_layers": 1}},
            "evaluation": {"objective": {"id": "objective"}},
            "training": {
                "stage": "b" if arm == "b" else "a",
                "comparison_arm": arm,
                "matched_a1_b": {"role": arm, "initialization": "a0_best_full_common_fresh_controller"},
                "teacher_handoff": None if arm == "a0" else {"source_run_name": "a0"},
            },
        }

    configs = {arm: config(arm) for arm in runs}
    monkeypatch.setattr(run_identity.config_loader, "validate_resolved_config", lambda config: config)
    payload: dict[str, Any] = {
        "schema_kind": run_identity.EXPERIMENT_RECORD_SCHEMA_KIND,
        "schema_version": run_identity.EXPERIMENT_RECORD_SCHEMA_VERSION,
        "task": "transient_drying",
        "parent_label": "matched-parent",
        "layout": configs["b"]["run"]["layout"],
        "seed": 9,
        "authored_config": {},
        "dataset_identity": {},
        "topology": "a0_to_b_and_a1",
        "children": {
            f"stage_{arm}": {
                "path": "../a0" if arm == "a0" else arm,
                "run_name": arm,
                "semantic_config_sha256": artifact.learning.training.checkpoint.resume_contract_digest(configs[arm]),
            }
            for arm in runs
        },
        "handoff": {"source_run_name": "a0", "target_run_names": ["b", "a1"]},
    }
    identity_payload = {key: value for key, value in payload.items() if key != "authored_config"}
    record = {
        **payload,
        "parent_identity_sha256": common.serialization.canonical_json_sha256(identity_payload),
        "source_repository": {"commit": None, "dirty": None},
    }
    marker = run_identity.experiment_record_path(record, output_root=output_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record), encoding="utf-8")
    manifest = runs["a0"] / HANDOFF_DIRECTORY / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"source_run_name": "a0", "files": {"checkpoint": {"sha256": "a" * 64}}}), encoding="utf-8")
    b_summary = {
        "status": "completed",
        "terminal_controller": {
            "budget_complete": True,
            "clock_kind": "train_seconds",
            "train_seconds": 12.5,
            "post_handoff_optimizer_steps": 4,
        },
        "terminal_epoch": 2,
        "best_epoch": 2,
        "best_checkpoint_sha256": "b" * 64,
        "last_checkpoint_sha256": "c" * 64,
    }
    receipt = matched_budget.build_b_receipt(
        parent_identity_sha256=record["parent_identity_sha256"],
        a0_handoff_sha256=common.serialization.file_sha256(manifest),
        b_config=configs["b"],
        b_summary=b_summary,
    )
    matched_budget.publish_b_receipt(runs["b"], receipt)
    matched_budget.publish_b_receipt(runs["a1"], receipt)
    parent = {"source_run_name": "a0", "source_checkpoint_sha256": "a" * 64}
    completed: dict[str, dict[str, Any]] = {}
    for arm in ("b", "a1"):
        planned, actual = (10.0, 12.5) if arm == "b" else (12.5, 13.0)
        controller = {
            "arm": arm.upper(),
            "stage": "stage_b_self_fed" if arm == "b" else "stage_a_teacher_forcing",
            "budget_control": "train_seconds",
            "clock_kind": "train_seconds",
            "teacher_handoff": parent,
            "planned_train_seconds": planned,
            "train_seconds": actual,
            "matched_receipt_sha256": receipt["receipt_sha256"] if arm == "a1" else None,
            "budget_complete": True,
            "best_within_budget_epoch": 1,
        }
        completed[arm] = {
            "run_dir": runs[arm],
            "scientific_run_name": arm,
            "config": configs[arm],
            "summary": {
                "run_identity": {"parent_label": "matched-parent"},
                "terminal_controller": controller,
                "terminal_curriculum": {"stage": 1},
            },
            "last_checkpoint": {"adapter_state_dict": {"controller": controller}},
            "selected_checkpoint_sha256": "b" * 64 if arm == "b" else "d" * 64,
            "selected_checkpoint_epoch": 2,
            "normalizer_sha256": "e" * 64,
            "effective_config_digest": "f" * 64,
            "checkpoint_identity": {"resume_contract_digest": artifact.learning.training.checkpoint.resume_contract_digest(configs[arm])},
        }
    return completed, receipt


def test_current_parent_binds_semantic_digest_without_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the saved semantic child digest independently of current Training source."""
    completed, receipt = _matched_context(tmp_path, monkeypatch)
    monkeypatch.setattr(
        artifact.learning.training.checkpoint,
        "resume_contract_digest",
        lambda _config: pytest.fail("Saved parent binding must not regenerate current Training identity"),
    )
    parent = artifact._parent_experiment_evidence(completed["a1"])
    assert parent["parent_identity_sha256"] == receipt["parent_identity_sha256"]
    assert parent["kind"] == "grouped"
    changed = deepcopy(completed["a1"])
    changed["checkpoint_identity"]["resume_contract_digest"] = "0" * 64
    with pytest.raises(ValueError, match="child identity"):
        artifact._parent_experiment_evidence(changed)


@pytest.mark.parametrize("saved_identity", [{}, {"resume_contract_digest": None}, {"resume_contract_digest": "not-a-digest"}])
def test_parent_rejects_missing_or_malformed_saved_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_identity: dict[str, Any]) -> None:
    completed, _receipt = _matched_context(tmp_path, monkeypatch)
    completed["a1"]["checkpoint_identity"] = saved_identity
    with pytest.raises(ValueError, match="valid saved resume identity"):
        artifact._parent_experiment_evidence(completed["a1"])


def test_a1_lineage_preserves_distinct_arm_and_actual_b_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist distinct A1 lineage backed by B's exact committed receipt."""
    completed, receipt = _matched_context(tmp_path, monkeypatch)
    b = artifact._lineage(completed["b"])
    a1 = artifact._lineage(completed["a1"])
    assert b["stage_identity"] == {"stage": "b", "comparison_arm": "b"}
    assert a1["stage_identity"] == {"stage": "a", "comparison_arm": "a1"}
    assert b["matched_compute_manifest"]["b_train_budget_receipt"] == receipt
    assert a1["matched_compute_manifest"]["b_train_budget_receipt"] == receipt
    assert a1["training_strategy"] == "teacher_forced_matched_train_seconds"
    assert b["parent_checkpoint"] == a1["parent_checkpoint"]

    changed = deepcopy(completed["a1"])
    changed["summary"]["terminal_controller"]["matched_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable B receipt"):
        artifact._lineage(changed)
    (completed["a1"]["run_dir"] / matched_budget.RECEIPT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt schema"):
        artifact._lineage(completed["a1"])


def test_cached_a1_lineage_rechecks_live_immutable_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revalidate receipt-backed lineage before admitting a cached artifact."""
    completed, _receipt = _matched_context(tmp_path, monkeypatch)
    a1 = completed["a1"]
    role = artifact_plan.ArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="dataset",
        source_dataset_ids=("dataset",),
        source_identities=({"dataset_id": "dataset"},),
        case_ids_by_source=(("case",),),
        membership_digests=("a" * 64,),
    )
    loaded = SimpleNamespace(
        dataset_name="dataset",
        dataset_role="id",
        provenance={
            "dataset": role.dataset_provenance(),
            "run": {
                "name": "a1",
                "effective_config_digest": "f" * 64,
                "checkpoint_identity": deepcopy(a1["checkpoint_identity"]),
                "best_checkpoint_sha256": "d" * 64,
                "best_checkpoint_epoch": 2,
                "normalizer_sha256": "e" * 64,
                "parent_experiment": artifact._parent_experiment_evidence(a1),
            },
            "lineage": artifact._lineage(a1),
        },
    )

    class ReachedSpatialError(Exception):
        pass

    def spatial(*_args: object, **_kwargs: object) -> None:
        raise ReachedSpatialError

    monkeypatch.setattr(artifact, "_validate_role_spatial_representation", spatial)
    with pytest.raises(ReachedSpatialError):
        artifact._validate_transient_role_evidence(
            cast("LoadedTransientSequenceArtifact | TransientSequenceArtifactIndex", loaded),
            completed=a1,
            role=role,
        )
    loaded.provenance["lineage"]["matched_compute_manifest"]["b_train_budget_receipt"]["train_seconds"] = 99.0
    with pytest.raises(ValueError, match="lineage is stale"):
        artifact._validate_transient_role_evidence(
            cast("LoadedTransientSequenceArtifact | TransientSequenceArtifactIndex", loaded),
            completed=a1,
            role=role,
        )


def _portable_matched_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path]:
    """Persist relative branch children with portable saved output-root provenance."""
    completed, receipt = _matched_context(tmp_path, monkeypatch)
    b_run = completed["b"]["run_dir"]
    marker = b_run.parent / "experiment.json"
    for arm in ("a0", "b", "a1"):
        run = b_run.parent.parent / "a0" if arm == "a0" else b_run.parent / arm
        config = deepcopy(completed["b"]["config"] if arm == "a0" else completed[arm]["config"])
        config["run"]["name"] = arm
        config["run"]["layout"]["stage"] = arm
        config["run"]["layout"]["branch_id"] = None if arm == "a0" else "b" * 16
        config["run"]["layout"]["pair"] = None if arm == "a0" else config["run"]["layout"]["pair"]
        config["training"]["comparison_arm"] = arm
        config["paths"] = {"output_root": "/workspace/storage/03_experiments"}
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        common.serialization.atomic_write_json(
            run / "summary.json",
            {
                "schema_version": common.paths.RUN_SUMMARY_SCHEMA_VERSION,
                "status": "completed",
                "config_sha256": common.serialization.file_sha256(run / "config.yaml"),
                "checkpoint_identity": {"resume_contract_digest": artifact.learning.training.checkpoint.resume_contract_digest(config)},
            },
        )
    return completed, receipt, marker


@pytest.mark.parametrize("arm", ["b", "a1"])
def test_native_matched_artifact_admission_uses_exact_portable_sibling_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    """Read the original A0 handoff and B budget through the shared verified root mapping."""
    completed, receipt, marker = _portable_matched_context(tmp_path, monkeypatch)
    original = marker.read_bytes()
    lineage = artifact._lineage(completed[arm])
    assert lineage["matched_compute_manifest"]["b_train_budget_receipt"] == receipt
    assert lineage["matched_compute_manifest"]["parent_experiment_sha256"] == receipt["parent_identity_sha256"]
    assert marker.read_bytes() == original
    sibling_config = completed["b"]["run_dir"].parent.parent / "a0/config.yaml"
    config = yaml.safe_load(sibling_config.read_text())
    config["run"]["name"] = "different-source"
    sibling_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="task/run identity"):
        artifact._lineage(completed[arm])
