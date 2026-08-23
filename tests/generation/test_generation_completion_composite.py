# ruff: noqa: D103, S101, SLF001, TC003
"""Test immutable replacement composite evidence admission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from src import common
from src.datasets.packages import dataset_packages_transient_shards as transient_shards
from src.generation import generation_workflow as workflow
from src.generation.publication import generation_publication_completion_composite as composite

if TYPE_CHECKING:
    from src.generation.runtime.generation_runtime_batch import TerminalBatchEvidence, TerminalCaseEvidence


@dataclass(frozen=True)
class _Artifact:
    path: Path


@dataclass(frozen=True)
class _Case:
    case_id: str
    case_index: int
    case_input_id: str
    simulation_case_id: str
    success_sha256: str
    provenance_sha256: str
    case_hdf5_sha256: str
    artifact_path: Path

    def artifact(self, stage: str, relative_path: str) -> _Artifact:
        assert (stage, relative_path) == ("processed", "case.h5")
        return _Artifact(self.artifact_path)


@dataclass(frozen=True)
class _Terminal:
    batch_id: str
    manifest_sha256: str
    cases: tuple[_Case, ...]
    batch_storage_name: str = "replacement-storage"


def _case(tmp_path: Path, name: str, index: int) -> _Case:
    path = tmp_path / "03_experiments" / name / "case.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    digest = common.serialization.file_sha256(path)
    return _Case(name, index, f"input-{name}", f"simulation-{name}", "a" * 64, "b" * 64, digest, path)


def _source(case: _Case, *, kind: str, terminal: _Terminal | None = None) -> composite.CompositeCaseSource:
    return composite.CompositeCaseSource(
        batch_id="batch-a",
        batch_name="batch-a",
        material_family="family",
        material_role="seen",
        evaluation_regime="id",
        sampling_regime="natural",
        source_run_id="replacement-run" if kind == "replacement" else "parent-run",
        source_git_commit=("e" if kind == "replacement" else "d") * 40,
        source_campaign_manifest_sha256=("f" if kind == "replacement" else "c") * 64,
        terminal=cast("TerminalBatchEvidence | None", terminal),
        case=cast("TerminalCaseEvidence", case),
        source_kind=kind,
    )


def test_composite_receipt_binds_exact_success_membership_and_excludes_parent_transfer(tmp_path: Path) -> None:
    original = _case(tmp_path, "original", 1)
    replacement = _case(tmp_path, "replacement", 1)
    terminal = _Terminal("replacement-batch", "c" * 64, (replacement,))
    state = {"completion_id": "completion__test", "parent_run_id": "run", "parent_partial_sha256": "d" * 64}
    state_digest = common.serialization.canonical_json_sha256(state)
    receipt = composite.build_composite_receipt(
        completion_id="completion__test",
        completion_state=state,
        completion_state_sha256=state_digest,
        parent_run_id="run",
        parent_partial_sha256="d" * 64,
        targets={"batch-a": 2},
        original_cases=(_source(original, kind="parent_partial"),),
        replacement_cases=(_source(replacement, kind="replacement", terminal=terminal),),
        storage_root=tmp_path,
    )
    assert receipt["combined_inventory_sha256"]
    assert receipt["source_git_commits"] == ["d" * 40, "e" * 40]
    transfer = composite.replacement_transfer_plan(receipt)
    assert [item["case_id"] for item in transfer] == ["replacement"]
    assert transfer[0]["terminal_batch_storage_name"] == "replacement-storage"


def test_composite_receipt_rejects_duplicate_physical_identity(tmp_path: Path) -> None:
    first = _case(tmp_path, "first", 1)
    duplicate = _Case("second", 2, first.case_input_id, "simulation-second", "a" * 64, "b" * 64, first.case_hdf5_sha256, first.artifact_path)
    state = {"completion_id": "completion__duplicate", "parent_run_id": "run", "parent_partial_sha256": "d" * 64}
    with pytest.raises(ValueError, match="duplicate case_input_id"):
        composite.build_composite_receipt(
            completion_id="completion__duplicate",
            completion_state=state,
            completion_state_sha256=common.serialization.canonical_json_sha256(state),
            parent_run_id="run",
            parent_partial_sha256="d" * 64,
            targets={"batch-a": 2},
            original_cases=(_source(first, kind="parent_partial"), _source(duplicate, kind="parent_partial")),
            replacement_cases=(),
            storage_root=tmp_path,
        )


def test_composite_lifecycle_revalidates_hdf5_pt_smoke_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_run_id = "parent-run"
    completion_id = "completion__smoke"
    parent_directory = tmp_path / "parent"
    parent_directory.mkdir()
    partial_path = parent_directory / "campaign_partial.json"
    partial_path.write_text("{}\n", encoding="utf-8")
    composite_receipt = {
        "completion_id": completion_id,
        "parent_run_id": parent_run_id,
        "parent_partial_sha256": common.serialization.file_sha256(partial_path),
        "combined_inventory_sha256": "a" * 64,
        "replacement_transfer_inventory_sha256": "b" * 64,
    }
    identity = workflow._composite_publication_identity(composite_receipt)
    plan = {
        "dataset_name": "transient",
        "dataset_view": "transient_drying",
        "evaluation_regime": "id",
        "training_payload": {"required": True},
    }
    record = {
        "dataset_name": "transient",
        "dataset_id": "dataset-id",
        "dataset_view": "transient_drying",
        "evaluation_regime": "id",
        "manifest_relative_path": "datasets/dataset-id/dataset_manifest.json",
    }
    shard_path = tmp_path / "shards" / "receipt.json"
    shard_path.parent.mkdir()
    shard_path.write_text("{}\n", encoding="utf-8")
    smoke = {
        "one_step_transition": {
            "status": "equivalent",
            "storage_backend": "pt_shards",
            "sample_id": "sample-1",
            "rollout_length": 1,
        },
        "rollout_window": {
            "status": "equivalent",
            "storage_backend": "pt_shards",
            "sample_id": "sample-1",
            "rollout_length": 2,
        },
    }
    cleanup_sources = ({"campaign_run_id": "replacement-run"},)
    lifecycle = {
        "schema_kind": workflow.COMPOSITE_LIFECYCLE_SCHEMA_KIND,
        "schema_version": workflow.WORKFLOW_SCHEMA_VERSION,
        "status": "ready",
        "completion_id": completion_id,
        "parent_run_id": parent_run_id,
        "parent_partial_sha256": composite_receipt["parent_partial_sha256"],
        "composite_receipt_sha256": identity["completion_receipt_sha256"],
        "combined_inventory_sha256": composite_receipt["combined_inventory_sha256"],
        "packages": [record],
        "shards": [
            {
                "dataset_id": "dataset-id",
                "required": True,
                "receipt_relative_path": shard_path.relative_to(tmp_path).as_posix(),
                "receipt_sha256": common.serialization.file_sha256(shard_path),
                "derived_payload_id": "payload-id",
                "loader_smoke": {"one_step_transition": {"status": "loaded"}},
                "hdf5_pt_smoke": smoke,
            }
        ],
        "readiness": "ready",
        "replacement_cleanup": {
            "eligible": True,
            "replacement_transfer_inventory_sha256": composite_receipt["replacement_transfer_inventory_sha256"],
            "sources": list(cleanup_sources),
        },
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    monkeypatch.setattr(workflow.workspace_service, "resolve_storage_root", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(workflow.campaign_runtime, "validate_partially_transferred_campaign", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(composite, "load_composite_receipt", lambda *_args, **_kwargs: composite_receipt)
    monkeypatch.setattr(composite, "replacement_transfer_plan", lambda *_args, **_kwargs: cleanup_sources)
    monkeypatch.setattr(workflow.campaign_evidence, "campaign_run_directory", lambda *_args, **_kwargs: parent_directory)
    monkeypatch.setattr(
        workflow.campaign_evidence,
        "campaign_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(dataset_packages=(plan,)),
    )
    monkeypatch.setattr(workflow, "_validate_package_record", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        workflow,
        "_load_json",
        lambda _path, *, label: (
            lifecycle
            if label == "composite completion lifecycle receipt"
            else {"source_case_identities": [{"completion_receipt_sha256": identity["completion_receipt_sha256"]}]}
        ),
    )
    monkeypatch.setattr(
        transient_shards,
        "load_transient_shard_receipt",
        lambda *_args, **_kwargs: {"derived_payload_id": "payload-id"},
    )
    monkeypatch.setattr(workflow, "_smoke_transient_shard_backend", lambda *_args, **_kwargs: smoke)

    validated = workflow.validate_composite_completion_lifecycle(
        parent_run_id,
        completion_id,
        storage_root=tmp_path,
    )
    assert validated["replacement_cleanup"]["eligible"] is True
    lifecycle["shards"][0]["hdf5_pt_smoke"] = {"tampered": True}
    with pytest.raises(RuntimeError, match="semantic smoke"):
        workflow.validate_composite_completion_lifecycle(
            parent_run_id,
            completion_id,
            storage_root=tmp_path,
        )
