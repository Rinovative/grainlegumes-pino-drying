# ruff: noqa: S101
"""Package-only Generation continuation and immutable extension evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation
from src.datasets import packages as package_service
from src.datasets.runtime import dataset_runtime_package_validation as package_validation_service

if TYPE_CHECKING:
    from pathlib import Path

_RUN_ID = "campaign__0123456789abcdef"
_COMMIT = "a" * 40


def _plan(view: str) -> dict[str, Any]:
    """Return one compact ID package request."""
    return {
        "dataset_name": f"{view}__material__id",
        "dataset_view": view,
        "evaluation_regime": "id",
        "source_role": "seen",
    }


def _record(plan: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Return compact immutable package bindings used by the focused test."""
    return {
        **plan,
        "dataset_id": dataset_id,
        "manifest_sha256": "b" * 64,
        "payload_sha256": "c" * 64,
    }


def test_missing_package_extension_is_append_only_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build only the missing view and never rewrite historical gate receipts."""
    storage = (tmp_path / "storage").resolve()
    run_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / _RUN_ID
    run_directory.mkdir(parents=True)
    base_path = run_directory / generation.workflow.DATASET_RECEIPT_FILENAME
    workflow_path = run_directory / generation.workflow.ALL_WORKFLOW_RECEIPT_FILENAME
    base_bytes = b"immutable base receipt\n"
    workflow_bytes = b"immutable all-workflow receipt\n"
    base_path.write_bytes(base_bytes)
    workflow_path.write_bytes(workflow_bytes)

    base_plan = _plan("transient_drying")
    extension_plan = _plan("steady_flow")
    base_record = _record(base_plan, "base-id")
    extension_record = _record(extension_plan, "extension-id")
    base_receipt = {
        "campaign_id": "campaign-id",
        "campaign_digest": "d" * 64,
        "git_commit": _COMMIT,
        "selected_batch_ids": ["batch-id"],
        "transfer_receipt_sha256": "e" * 64,
        "packages": [base_record],
    }
    launch = SimpleNamespace(
        campaign_purpose="family_generalization",
        dataset_packages=(base_plan,),
    )
    current = SimpleNamespace(
        campaign_id="campaign-id",
        campaign_digest="d" * 64,
        package_request_digest="f" * 64,
        dataset_packages=(base_plan, extension_plan),
    )
    terminal = {
        "dataset_packages": [base_plan],
        "git_commit": _COMMIT,
        "batches": [{"batch_id": "batch-id"}],
    }
    workflow_receipt = {
        "workflow_result": "success",
        "cpu_cleanup_complete": {"status": "complete"},
    }
    source_artifact_set = {
        "artifact_set_sha256": "1" * 64,
        "batch_count": 1,
        "case_count": 3,
        "batch_manifests": [{"batch_id": "batch-id", "manifest_sha256": "2" * 64}],
    }
    build_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        generation.workflow.workspace_service,
        "resolve_storage_root",
        lambda *_args, **_kwargs: storage,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_runtime,
        "admit_transferred_campaign",
        lambda *_args, **_kwargs: {"status": "transfer_complete"},
    )
    monkeypatch.setattr(
        generation.workflow.campaign_runtime,
        "validate_terminal_campaign",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "campaign_for_run",
        lambda *_args, **_kwargs: launch,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "current_campaign_for_run",
        lambda *_args, **_kwargs: current,
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_dataset_packages_receipt",
        lambda *_args, **_kwargs: base_receipt,
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_completed_workflow",
        lambda *_args, **_kwargs: workflow_receipt,
    )
    monkeypatch.setattr(
        generation.workflow,
        "_campaign_source_artifact_identity",
        lambda *_args, **_kwargs: source_artifact_set,
    )
    monkeypatch.setattr(
        generation.workflow,
        "_validate_package_record",
        lambda record, **_kwargs: dict(record),
    )
    monkeypatch.setattr(
        generation.workflow,
        "_package_record",
        lambda _result, **_kwargs: extension_record,
    )

    def build_missing(
        _campaign: Any,
        dataset_view: str,
        evaluation_regime: str,
        **_kwargs: Any,
    ) -> dict[str, str]:
        build_calls.append((dataset_view, evaluation_regime))
        return {"dataset_id": "extension-id", "status": "complete"}

    monkeypatch.setattr(
        package_service,
        "build_dataset_package",
        build_missing,
    )

    first = generation.workflow.build_campaign_datasets(
        _RUN_ID,
        storage_root=storage,
        prepare_training_payloads=False,
    )
    second = generation.workflow.build_campaign_datasets(
        _RUN_ID,
        storage_root=storage,
        prepare_training_payloads=False,
    )

    assert first == second
    assert first["status"] == "complete"
    assert [record["dataset_id"] for record in first["packages"]] == [
        "base-id",
        "extension-id",
    ]
    assert build_calls == [("steady_flow", "id")]
    assert base_path.read_bytes() == base_bytes
    assert workflow_path.read_bytes() == workflow_bytes
    extension_paths = tuple((run_directory / generation.workflow.DATASET_EXTENSION_DIRECTORY_NAME).glob("*.json"))
    assert len(extension_paths) == 1
    extension = json.loads(extension_paths[0].read_text(encoding="utf-8"))
    assert extension["package_plan"] == extension_plan
    assert extension["source_artifact_set"] == source_artifact_set
    assert extension["cpu_source_cleanup_reopened"] is False


def test_compatible_source_discovery_rejects_multiple_completed_runs(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an explicit source when multiple completed runs match exactly."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        campaign_purpose="family_generalization",
        natural_count=3,
        executable=fake_comsol,
    )
    campaign = generation.cases.config.load_campaign_config(
        config_path,
        require_executable=True,
    )
    storage = (tmp_path / "storage").resolve()
    campaigns_root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    run_ids = (
        "campaign__1111111111111111",
        "campaign__2222222222222222",
    )
    for run_id in run_ids:
        directory = campaigns_root / run_id
        directory.mkdir(parents=True)
        (directory / generation.workflow.ALL_WORKFLOW_RECEIPT_FILENAME).write_text(
            "{}\n",
            encoding="utf-8",
        )

    manifest = {
        "campaign_config": str(config_path),
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "state": "complete",
    }
    terminal = {
        "git_commit": _COMMIT,
        "batches": [{"batch_id": batch.batch_id} for batch in campaign.batches],
    }
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "resolve_campaign_config_path",
        lambda *_args, **_kwargs: campaign.source_path.resolve(),
    )
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "campaign_for_run",
        lambda *_args, **_kwargs: campaign,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "current_campaign_for_run",
        lambda *_args, **_kwargs: campaign,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_runtime,
        "validate_terminal_campaign",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_runtime,
        "admit_transferred_campaign",
        lambda *_args, **_kwargs: {"transfer_inventory_sha256": "e" * 64},
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_completed_workflow",
        lambda *_args, **_kwargs: {
            "workflow_result": "success",
            "cpu_cleanup_complete": {"status": "complete"},
        },
    )
    monkeypatch.setattr(
        generation.workflow,
        "_campaign_source_artifact_identity",
        lambda *_args, **_kwargs: {"artifact_set_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_campaign_package_state",
        lambda *_args, **_kwargs: {"status": "complete"},
    )

    with pytest.raises(RuntimeError, match="explicit source selection") as caught:
        generation.workflow.find_compatible_completed_campaign_source(
            config_path,
            storage_root=storage,
        )
    assert all(run_id in str(caught.value) for run_id in run_ids)


def test_package_smoke_uses_manifest_training_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Choose a training smoke from package membership, not campaign naming."""
    storage = (tmp_path / "storage").resolve()
    calls: list[tuple[str | None, int]] = []
    manifest = {
        "campaign_purpose": "steady_flow_id_dataset",
        "evaluation_regime": "id",
        "training_eligible": True,
    }
    monkeypatch.setattr(
        package_validation_service.package_manifest,
        "load_package_manifest_evidence",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        package_validation_service,
        "_inspect_dataset_package",
        lambda *_args, **_kwargs: {"status": "inspected"},
    )

    def smoke(
        _dataset_id: str,
        *,
        membership: str | None,
        num_workers: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((membership, num_workers))
        return {"status": "loaded"}

    monkeypatch.setattr(package_validation_service, "_smoke_dataset_package", smoke)

    generation.workflow._package_runtime_evidence(  # noqa: SLF001
        "steady-flow-id",
        storage_root=storage,
    )

    assert calls == [("train", 0), ("train", 2)]
