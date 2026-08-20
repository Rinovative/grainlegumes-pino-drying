# ruff: noqa: S101
"""Post-transfer workflow receipt, cleanup, and storage lifecycle contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import common, generation
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_workspace as workspace

pytestmark = pytest.mark.integration

_RUN_ID = "synthetic_workflow__0123456789abcdef"
_COMMIT = "a" * 40
_AUTH_DIGESTS = {
    "transfer_receipt_sha256": "1" * 64,
    "dataset_receipt_sha256": "2" * 64,
    "workflow_gate_sha256": "3" * 64,
}


def _mock_local_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Install one complete GPU transfer and dataset-gate fixture."""
    storage = tmp_path / "storage"
    run_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / _RUN_ID
    run_directory.mkdir(parents=True)
    transfer_path = run_directory / "transfer_complete.json"
    dataset_path = run_directory / generation.workflow.DATASET_RECEIPT_FILENAME
    transfer_path.write_text("{}\n", encoding="utf-8")
    dataset_path.write_text("{}\n", encoding="utf-8")
    directories = (
        "01_generation/meta/batches/batch_a",
        "01_generation/raw/batch_a",
        "01_generation/processed/batch_a",
        f"01_generation/attempts/batch_a/case_00001/{_RUN_ID}",
    )
    files = [
        {
            "relative_path": f"{directory}/payload-{index}.bin",
            "size_bytes": index + 1,
            "sha256": str(index + 1) * 64,
        }
        for index, directory in enumerate(directories)
    ]
    files.append(
        {
            "relative_path": f"01_generation/meta/campaigns/{_RUN_ID}/campaign_terminal.json",
            "size_bytes": 4,
            "sha256": "4" * 64,
        }
    )
    transfer = {
        "campaign_run_id": _RUN_ID,
        "campaign_id": "campaign-id",
        "git_commit": _COMMIT,
        "source_host": "cpu.example",
        "source_storage_root": "/remote/storage",
        "destination_storage_root": str(storage.resolve()),
        "campaign_terminal_sha256": "f" * 64,
        "transferred_file_count": len(files),
        "transferred_bytes": sum(record["size_bytes"] for record in files),
        "transfer_inventory_sha256": common.serialization.canonical_json_sha256(files),
        "files": files,
    }
    terminal = {
        "campaign_id": "campaign-id",
        "git_commit": _COMMIT,
        "batches": [{"batch_id": "batch_a"}],
    }
    package = {
        "dataset_id": "dataset-id",
        "manifest_sha256": "b" * 64,
        "payload_sha256": "c" * 64,
        "inspection": {"dataset_id": "dataset-id", "status": "valid"},
        "loader_smoke": {"dataset_id": "dataset-id", "status": "loaded"},
    }
    datasets_receipt = {"packages": [package]}
    plan = {
        "campaign_directory": f"01_generation/meta/campaigns/{_RUN_ID}",
        "batches": [
            {
                "batch_id": "batch_a",
                "meta_directory": directories[0],
                "raw_directory": directories[1],
                "processed_directory": directories[2],
                "attempt_directories": [directories[3]],
            }
        ],
    }
    monkeypatch.setattr(
        generation.campaign,
        "validate_transferred_campaign",
        lambda *_args, **_kwargs: transfer,
    )
    monkeypatch.setattr(
        generation.campaign,
        "validate_terminal_campaign",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        generation.campaign,
        "campaign_transfer_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_dataset_packages_receipt",
        lambda *_args, **_kwargs: datasets_receipt,
    )
    return storage, transfer, terminal, datasets_receipt


def test_post_transfer_operational_paths_do_not_change_campaign_transfer_identity(tmp_path: Path) -> None:
    """Exclude declared operational files and progress descendants from identity."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_terminal.json").write_text("{}\n", encoding="utf-8")
    before = campaign_evidence.directory_identity(
        campaign,
        ignored_relative_paths=campaign_evidence.POST_TRANSFER_OPERATIONAL_PATHS,
    )
    (campaign / "dataset_packages_complete.lock").touch()
    (campaign / campaign_evidence.TECHNICAL_SMOKE_EVIDENCE_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    progress_directory = campaign / campaign_evidence.RUNTIME_PROGRESS_DIRECTORY_NAME
    progress_directory.mkdir()
    (progress_directory / "123.json").write_text("{}\n", encoding="utf-8")
    failure_directory = campaign / "workflow_failures"
    failure_directory.mkdir()
    (failure_directory / "failure-0001.json").write_text("{}\n", encoding="utf-8")

    assert (
        campaign_evidence.directory_identity(
            campaign,
            ignored_relative_paths=campaign_evidence.POST_TRANSFER_OPERATIONAL_PATHS,
        )
        == before
    )
    assert campaign_evidence.directory_identity(campaign) != before
    nested = campaign / "nested"
    nested.mkdir()
    (nested / "dataset_packages_complete.lock").touch()
    assert (
        campaign_evidence.directory_identity(
            campaign,
            ignored_relative_paths=campaign_evidence.POST_TRANSFER_OPERATIONAL_PATHS,
        )
        != before
    )


def test_dataset_finalization_lock_uses_generation_local_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep dataset-finalization synchronization outside campaign evidence."""
    storage = tmp_path / "storage"
    run_directory = campaign_evidence.campaign_run_directory(_RUN_ID, storage_root=storage)
    run_directory.mkdir(parents=True)
    (run_directory / generation.workflow.DATASET_RECEIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(generation.campaign, "validate_transferred_campaign", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        generation.campaign,
        "validate_terminal_campaign",
        lambda *_args, **_kwargs: {"dataset_packages": []},
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(dataset_packages=()),
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_dataset_packages_receipt",
        lambda *_args, **_kwargs: {"packages": []},
    )

    assert generation.workflow.build_campaign_datasets(_RUN_ID, storage_root=storage) == {"packages": []}
    expected_lock = common.paths.get_generation_state_root(storage_root=storage) / "dataset-package-locks" / f"{_RUN_ID}.lock"
    assert expected_lock.is_file()
    assert not (run_directory / "dataset_packages_complete.lock").exists()


def test_workflow_failure_receipts_are_visible_verified_campaign_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep durable workflow errors outside private coordination state."""
    storage = tmp_path / "storage"
    run_directory = campaign_evidence.campaign_run_directory(
        _RUN_ID,
        storage_root=storage,
    )
    run_directory.mkdir(parents=True)
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {},
    )

    first = generation.workflow.record_workflow_failure(
        _RUN_ID,
        storage_root=storage,
        stage="synthetic transfer",
        continuation_command="./scripts/generation_workflow.sh run configs/generation/campaigns/steady_flow/id_dataset.yaml",
        cpu_bytes_retained=17,
    )
    first_bytes = first.read_bytes()
    second = generation.workflow.record_workflow_failure(
        _RUN_ID,
        storage_root=storage,
        stage="synthetic publication",
        continuation_command="./scripts/generation_workflow.sh run configs/generation/campaigns/steady_flow/id_dataset.yaml",
        cpu_bytes_retained=17,
    )

    assert first == (run_directory / "workflow_failures" / "failure-0001.json").resolve()
    assert second.name == "failure-0002.json"
    assert ".state" not in first.parts
    assert first.read_bytes() == first_bytes
    assert json.loads(first.read_text(encoding="utf-8"))["schema_version"] == 1
    assert json.loads(second.read_text(encoding="utf-8"))["stage"] == ("synthetic publication")


def test_all_receipt_records_distinct_gates_and_cleanup_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the durable pre-cleanup gate and both terminal cleanup outcomes."""
    storage, _transfer, _terminal, _datasets = _mock_local_gates(tmp_path, monkeypatch)
    ready = generation.workflow.prepare_all_workflow_receipt(
        _RUN_ID,
        storage_root=storage,
    )
    assert ready["workflow_result"] == "ready_for_cpu_cleanup"
    assert ready["cpu_cleanup_complete"]["status"] == "pending"
    for stage in (
        "generation_complete",
        "transfer_complete",
        "gpu_publication_complete",
        "dataset_packages_complete",
        "loader_smokes_complete",
    ):
        assert ready[stage]["status"] == "complete"
    authorization = generation.workflow.cpu_cleanup_authorization(
        _RUN_ID,
        storage_root=storage,
    )
    completed = generation.workflow.record_cpu_cleanup_complete(
        _RUN_ID,
        storage_root=storage,
        authorization_sha256=authorization["authorization_sha256"],
        cleanup_receipt_sha256="d" * 64,
        reclaimed_bytes=authorization["source_bytes"],
    )
    assert completed["workflow_result"] == "success"
    assert completed["cpu_cleanup_complete"]["status"] == "complete"
    assert completed["cpu_bytes_reclaimed"] == authorization["source_bytes"]

    other = tmp_path / "retained"
    retained_storage, _transfer, _terminal, _datasets = _mock_local_gates(other, monkeypatch)
    retained = generation.workflow.prepare_all_workflow_receipt(
        _RUN_ID,
        storage_root=retained_storage,
        cleanup_requested=False,
    )
    assert retained["workflow_result"] == "success"
    assert retained["cpu_cleanup_complete"]["status"] == "skipped_by_request"
    retained_authorization = generation.workflow.cpu_cleanup_authorization(
        _RUN_ID,
        storage_root=retained_storage,
    )
    late_cleanup = generation.workflow.record_cpu_cleanup_complete(
        _RUN_ID,
        storage_root=retained_storage,
        authorization_sha256=retained_authorization["authorization_sha256"],
        cleanup_receipt_sha256="e" * 64,
        reclaimed_bytes=retained_authorization["source_bytes"],
    )
    assert late_cleanup["cleanup_requested"] is True
    assert late_cleanup["cpu_cleanup_complete"]["status"] == "complete"


def _remote_cleanup_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Install one terminal CPU source and return exact cleanup arguments."""
    storage = (tmp_path / "cpu storage").resolve()
    run_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / _RUN_ID
    run_directory.mkdir(parents=True)
    directories = (
        "01_generation/meta/batches/batch_a",
        "01_generation/raw/batch_a",
        "01_generation/processed/batch_a",
        f"01_generation/attempts/batch_a/case_00001/{_RUN_ID}",
    )
    for index, relative in enumerate(directories, start=1):
        directory = storage / relative
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(bytes([index]) * index)
    terminal = {
        "campaign_id": "campaign-id",
        "git_commit": _COMMIT,
        "slurm_job_ids": ["123"],
        "scheduler_job_name": "vp2-synthetic",
        "batches": [{"batch_id": "batch_a"}],
    }
    terminal_path = run_directory / "campaign_terminal.json"
    terminal_path.write_text(
        json.dumps(terminal, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan = {
        "campaign_directory": f"01_generation/meta/campaigns/{_RUN_ID}",
        "batches": [
            {
                "batch_id": "batch_a",
                "meta_directory": directories[0],
                "raw_directory": directories[1],
                "processed_directory": directories[2],
                "attempt_directories": [directories[3]],
            }
        ],
    }
    files = [
        {
            "relative_path": f"{relative}/payload.bin",
            "size_bytes": (storage / relative / "payload.bin").stat().st_size,
            "sha256": common.serialization.file_sha256(storage / relative / "payload.bin"),
        }
        for relative in directories
    ]
    inventory = {
        "file_count": len(files),
        "size_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
        "inventory_sha256": common.serialization.canonical_json_sha256(files),
    }
    monkeypatch.setattr(
        generation.campaign,
        "validate_terminal_campaign",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        generation.campaign,
        "campaign_transfer_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        generation.campaign,
        "campaign_transfer_inventory",
        lambda *_args, **_kwargs: inventory,
    )
    monkeypatch.setattr(
        generation.campaign,
        "campaign_status",
        lambda *_args, **_kwargs: {
            "campaign_state": "successful",
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "123|COMPLETED|0:0", "error": None},
        },
    )
    eligible_files = files[:3]
    directory_records = [
        {
            "relative_path": relative,
            "file_count": 1,
            "size_bytes": (storage / relative / "payload.bin").stat().st_size,
        }
        for relative in directories[:3]
    ]
    payload = {
        "campaign_run_id": _RUN_ID,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "selected_batch_ids": ["batch_a"],
        "source_host": "cpu.example",
        "source_storage_root": str(storage),
        "destination_storage_root": "/gpu/storage",
        **_AUTH_DIGESTS,
        "source_inventory_sha256": common.serialization.canonical_json_sha256(eligible_files),
        "source_file_count": len(eligible_files),
        "source_bytes": sum(record["size_bytes"] for record in eligible_files),
        "source_directories": directory_records,
    }
    arguments = {
        "storage_root": storage,
        "source_host": payload["source_host"],
        "destination_storage_root": payload["destination_storage_root"],
        "transfer_receipt_sha256": payload["transfer_receipt_sha256"],
        "dataset_receipt_sha256": payload["dataset_receipt_sha256"],
        "workflow_gate_sha256": payload["workflow_gate_sha256"],
        "source_inventory_sha256": payload["source_inventory_sha256"],
        "source_file_count": payload["source_file_count"],
        "source_bytes": payload["source_bytes"],
        "authorization_sha256": common.serialization.canonical_json_sha256(payload),
    }
    return storage, arguments, plan


def test_cpu_cleanup_is_dry_run_transactional_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect dry-run retention, exact run deletion, compact evidence, and retry."""
    storage, arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    dry_run = generation.workflow.cleanup_cpu_campaign_source(
        _RUN_ID,
        **arguments,
    )
    assert dry_run["mode"] == "dry-run"
    assert dry_run["reclaimable_bytes"] == arguments["source_bytes"]
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert (storage / plan["batches"][0][key]).is_dir()

    complete = generation.workflow.cleanup_cpu_campaign_source(
        _RUN_ID,
        confirm=True,
        **arguments,
    )
    assert complete["status"] == "complete"
    assert complete["selected_batch_ids"] == ["batch_a"]
    assert complete["slurm_job_ids"] == ["123"]
    assert complete["source_host"] == "cpu.example"
    assert complete["source_storage_root"] == str(storage)
    assert complete["destination_storage_root"] == "/gpu/storage"
    assert complete["transfer_receipt_sha256"] == _AUTH_DIGESTS["transfer_receipt_sha256"]
    assert complete["dataset_receipt_sha256"] == _AUTH_DIGESTS["dataset_receipt_sha256"]
    assert complete["workflow_gate_sha256"] == _AUTH_DIGESTS["workflow_gate_sha256"]
    assert complete["destination_inventory_sha256"] == complete["source_inventory_sha256"]
    assert complete["source_bytes_reclaimed"] == arguments["source_bytes"]
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert not (storage / plan["batches"][0][key]).exists()
    assert (storage / plan["campaign_directory"]).is_dir()
    assert (storage / plan["batches"][0]["attempt_directories"][0]).is_dir()
    assert (storage / plan["campaign_directory"] / generation.workflow.CPU_CLEANUP_RECEIPT_FILENAME).is_file()

    repeated = generation.workflow.cleanup_cpu_campaign_source(
        _RUN_ID,
        confirm=True,
        **arguments,
    )
    assert repeated == complete


def test_cpu_cleanup_rejects_active_failed_and_incomplete_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve every source directory unless scheduler and publication state are safe."""
    storage, arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    unsafe_statuses = (
        (
            {
                "campaign_state": "successful",
                "squeue": {"output": "123|RUNNING|node", "error": None},
                "sacct": {"output": "123|RUNNING|0:0", "error": None},
            },
            "active campaign",
        ),
        (
            {
                "campaign_state": "failed",
                "squeue": {"output": "", "error": None},
                "sacct": {"output": "123|FAILED|1:0", "error": None},
            },
            "requires a successful terminal source publication",
        ),
        (
            {
                "campaign_state": "running",
                "squeue": {"output": "", "error": None},
                "sacct": {"output": "123|COMPLETED|0:0", "error": None},
            },
            "requires a successful terminal source publication",
        ),
    )
    for status, expected_message in unsafe_statuses:
        monkeypatch.setattr(
            generation.campaign,
            "campaign_status",
            lambda *_args, _status=status, **_kwargs: _status,
        )
        with pytest.raises(RuntimeError, match=expected_message):
            generation.workflow.cleanup_cpu_campaign_source(
                _RUN_ID,
                confirm=True,
                **arguments,
            )
    resumed_status = {
        "campaign_state": "transfer_complete",
        "squeue": {"output": "", "error": None},
        "sacct": {
            "output": "123|FAILED|1:0\n124|COMPLETED|0:0",
            "error": None,
        },
    }
    monkeypatch.setattr(
        generation.campaign,
        "campaign_status",
        lambda *_args, **_kwargs: resumed_status,
    )
    assert generation.workflow.cleanup_cpu_campaign_source(_RUN_ID, **arguments)["status"] == "eligible"
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert (storage / plan["batches"][0][key]).is_dir()
    assert not (storage / plan["campaign_directory"] / generation.workflow.CPU_CLEANUP_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize(
    "campaign_state",
    ["successful", "transfer_complete"],
)
def test_campaign_source_status_accepts_authoritative_terminal_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign_state: str,
) -> None:
    """Keep cleanup eligibility aligned with canonical campaign states."""
    storage, _arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    batch = plan["batches"][0]
    manifest = {
        "batches": [
            {
                key: str((storage / batch[key]).resolve())
                for key in (
                    "meta_directory",
                    "raw_directory",
                    "processed_directory",
                )
            }
        ]
    }
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        generation.campaign,
        "campaign_status",
        lambda *_args, **_kwargs: {
            "campaign_state": campaign_state,
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "123|COMPLETED|0:0", "error": None},
        },
    )

    status = generation.workflow.campaign_source_status(
        _RUN_ID,
        storage_root=storage,
        query_scheduler=True,
    )

    assert status["campaign_state"] == campaign_state
    assert status["cleanup_eligibility"] == "requires_gpu_authorization"
    assert status["active_slurm"] is False


def test_cpu_cleanup_rejects_sources_shared_with_another_campaign_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never remove a deterministic batch directory referenced by another run."""
    storage, arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    other_run_id = "other_campaign__fedcba9876543210"
    other_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / other_run_id
    other_directory.mkdir()
    shared_batch = {key: str((storage / plan["batches"][0][key]).resolve()) for key in ("meta_directory", "raw_directory", "processed_directory")}
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {"state": "submitted", "batches": [shared_batch]},
    )

    with pytest.raises(RuntimeError, match="referenced by other campaign runs"):
        generation.workflow.cleanup_cpu_campaign_source(
            _RUN_ID,
            confirm=True,
            **arguments,
        )
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert (storage / plan["batches"][0][key]).is_dir()


def test_cpu_cleanup_rolls_back_every_directory_on_move_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect run-level all-or-nothing cleanup before canonical source removal."""
    storage, arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    second = (storage / plan["batches"][0]["raw_directory"]).resolve()
    original_replace = Path.replace

    def fail_second(source: Path, target: Path) -> Path:
        if source.resolve() == second:
            message = "synthetic move failure"
            raise OSError(message)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError, match="synthetic move failure"):
        generation.workflow.cleanup_cpu_campaign_source(
            _RUN_ID,
            confirm=True,
            **arguments,
        )
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert (storage / plan["batches"][0][key]).is_dir()
    assert not (storage / plan["campaign_directory"] / generation.workflow.CPU_CLEANUP_RECEIPT_FILENAME).exists()


def test_cpu_cleanup_recovers_interrupted_planned_detach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback a partial planned detach before a confirmed retry starts again."""
    storage, arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    second = (storage / plan["batches"][0]["raw_directory"]).resolve()
    first = (storage / plan["batches"][0]["meta_directory"]).resolve()
    original_replace = Path.replace

    def interrupt_second(source: Path, target: Path) -> Path:
        if source.resolve() == second:
            raise KeyboardInterrupt
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        generation.workflow.cleanup_cpu_campaign_source(
            _RUN_ID,
            confirm=True,
            **arguments,
        )
    transaction = common.paths.get_generation_state_root(storage_root=storage) / "source-cleanup" / _RUN_ID
    marker = json.loads((transaction / "transaction.json").read_text(encoding="utf-8"))
    assert marker["status"] == "planned"
    assert not first.exists()
    assert second.is_dir()
    with pytest.raises(RuntimeError, match="explicit confirmed retry"):
        generation.workflow.cleanup_cpu_campaign_source(_RUN_ID, **arguments)

    monkeypatch.setattr(Path, "replace", original_replace)
    complete = generation.workflow.cleanup_cpu_campaign_source(
        _RUN_ID,
        confirm=True,
        **arguments,
    )
    assert complete["status"] == "complete"
    assert not transaction.exists()
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert not (storage / plan["batches"][0][key]).exists()


def test_cpu_cleanup_finishes_interrupted_detached_disposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish an authorized detached disposal and publish its durable receipt."""
    storage, arguments, plan = _remote_cleanup_fixture(tmp_path, monkeypatch)
    transaction = common.paths.get_generation_state_root(storage_root=storage) / "source-cleanup" / _RUN_ID
    payload_root = transaction / "payload"
    original_rmtree = shutil.rmtree
    interrupted = False

    def interrupt_payload(path: Path | str, *args: Any, **kwargs: Any) -> None:
        nonlocal interrupted
        if Path(path) == payload_root and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", interrupt_payload)
    with pytest.raises(KeyboardInterrupt):
        generation.workflow.cleanup_cpu_campaign_source(
            _RUN_ID,
            confirm=True,
            **arguments,
        )
    marker = json.loads((transaction / "transaction.json").read_text(encoding="utf-8"))
    assert marker["status"] == "detached"
    assert payload_root.is_dir()
    for key in ("meta_directory", "raw_directory", "processed_directory"):
        assert not (storage / plan["batches"][0][key]).exists()

    monkeypatch.setattr(shutil, "rmtree", original_rmtree)
    complete = generation.workflow.cleanup_cpu_campaign_source(
        _RUN_ID,
        confirm=True,
        **arguments,
    )
    assert complete["status"] == "complete"
    assert complete["source_bytes_reclaimed"] == arguments["source_bytes"]
    assert not transaction.exists()
    repeated = generation.workflow.cleanup_cpu_campaign_source(
        _RUN_ID,
        confirm=True,
        **arguments,
    )
    assert repeated == complete


def test_storage_status_reports_separate_protected_layers_and_staging(
    tmp_path: Path,
) -> None:
    """Protect the sibling lifecycle roots, usage totals, and staging visibility."""
    storage = (tmp_path / "storage").resolve()
    generation_root = common.paths.get_generation_root(storage_root=storage)
    datasets_root = common.paths.get_datasets_root(storage_root=storage)
    generation_root.mkdir(parents=True)
    datasets_root.mkdir(parents=True)
    (generation_root / "source.bin").write_bytes(b"source")
    (datasets_root / "package.bin").write_bytes(b"package")
    staging = workspace.create_transfer_staging(
        storage_root=storage,
        run_id=_RUN_ID,
    )
    (staging / "partial.bin").write_bytes(b"partial")

    status = generation.workflow.storage_status(
        storage_root=storage,
        role="gpu",
    )
    assert status["roots"]["generation"] == str(generation_root)
    assert status["roots"]["datasets"] == str(datasets_root)
    assert status["generation_total_bytes"] >= len(b"source")
    assert status["datasets_total_bytes"] == len(b"package")
    assert status["packages_by_view_regime"] == []
    assert status["transfer_staging_bytes"] >= len(b"partial")
    assert status["protected_cleanup_targets"] == [str(generation_root), str(datasets_root)]


def test_shared_setup_idle_checks_campaigns_and_benchmarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse shared-installation mutation for any owned active Slurm job."""
    storage = (tmp_path / "storage").resolve()
    campaign_root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    campaign_root.joinpath(_RUN_ID).mkdir(parents=True)
    benchmark_id = "core_scaling_transient__0123456789abcdef"
    benchmark_root = common.paths.get_generation_performance_benchmark_root(storage_root=storage) / "core_scaling"
    benchmark_root.joinpath(benchmark_id).mkdir(parents=True)
    monkeypatch.setattr(
        generation.workflow,
        "_safe_campaign_source_status",
        lambda *_args, **_kwargs: {
            "active_slurm": False,
            "scheduler_error": None,
        },
    )
    monkeypatch.setattr(
        generation.benchmark,
        "core_benchmark_source_status",
        lambda *_args, **_kwargs: {"active_slurm": True},
    )

    with pytest.raises(
        RuntimeError,
        match=r"active dependent Generation jobs.*benchmark",
    ):
        generation.workflow.assert_shared_setup_idle(storage_root=storage)


def test_shared_setup_idle_fails_closed_on_scheduler_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject mutation when persisted campaign scheduler state is unprovable."""
    storage = (tmp_path / "storage").resolve()
    campaign_root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    campaign_root.joinpath(_RUN_ID).mkdir(parents=True)
    monkeypatch.setattr(
        generation.workflow,
        "_safe_campaign_source_status",
        lambda *_args, **_kwargs: {
            "active_slurm": None,
            "scheduler_error": "squeue unavailable",
        },
    )

    with pytest.raises(RuntimeError, match=r"cannot prove.*squeue unavailable"):
        generation.workflow.assert_shared_setup_idle(storage_root=storage)


def test_partial_completion_receipt_keeps_packages_incomplete_and_source_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate partial completion without Dataset IDs or cleanup authorization."""
    storage = (tmp_path / "storage").resolve()
    run_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / _RUN_ID
    run_directory.mkdir(parents=True)
    (run_directory / "transfer_partial.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    successful = [
        {
            "batch_name": "batch",
            "batch_id": "batch-id",
            "case_id": "case_0001",
            "case_index": 1,
            "state": "successful",
            "classified_state": "successful",
        }
    ]
    failed = [
        {
            "batch_name": "batch",
            "batch_id": "batch-id",
            "case_id": "case_0002",
            "case_index": 2,
            "state": "failed",
            "classified_state": "failed",
        }
    ]
    transfer = {
        "campaign_id": "campaign-id",
        "git_commit": _COMMIT,
        "source_host": "cpu.example",
        "source_storage_root": "/remote/storage",
        "successful_cases": successful,
        "failed_cases": failed,
    }
    monkeypatch.setattr(
        generation.campaign,
        "validate_partially_transferred_campaign",
        lambda *_args, **_kwargs: transfer,
    )
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {"campaign_run_id": _RUN_ID},
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: SimpleNamespace(dataset_packages=({"dataset_view": "all"},)),
    )

    datasets = generation.workflow.record_incomplete_campaign_datasets(
        _RUN_ID,
        storage_root=storage,
    )
    partial = generation.workflow.prepare_partial_completion_receipt(
        _RUN_ID,
        storage_root=storage,
    )

    assert datasets["schema_version"] == 1
    assert datasets["status"] == "incomplete"
    assert datasets["packages"] == []
    assert datasets["dataset_ids"] == []
    assert partial["schema_version"] == 1
    assert partial["workflow_result"] == "partial"
    assert partial["campaign_state"] == "completed_with_failures"
    assert partial["successful_cases"] == successful
    assert partial["failed_cases"] == failed
    assert partial["dataset_ids"] == []
    assert partial["cpu_source_retained"] is True
    assert partial["cleanup_requested"] is False
    assert partial["resume_command"] == f"resume {_RUN_ID}"

    receipt_path = run_directory / generation.workflow.PARTIAL_COMPLETION_RECEIPT_FILENAME
    corrupted = json.loads(receipt_path.read_text(encoding="utf-8"))
    corrupted["cleanup_requested"] = True
    receipt_path.write_text(
        json.dumps(corrupted) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Partial completion receipt is invalid"):
        generation.workflow.validate_partial_completion_receipt(
            _RUN_ID,
            storage_root=storage,
        )
