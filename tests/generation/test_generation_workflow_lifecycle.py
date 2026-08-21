# ruff: noqa: S101
"""Post-transfer workflow receipt, cleanup, and storage lifecycle contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import common, datasets, generation
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


def _write_setup_idle_campaign(
    storage: Path,
    run_id: str,
    *,
    job_ids: tuple[str, ...],
    git_commit: str,
    state: str,
) -> Path:
    """Write the scheduler-ownership slice of one historical campaign."""
    directory = campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    )
    directory.mkdir(parents=True)
    submissions = [
        {
            "submission_index": index,
            "mode": "initial",
            "recorded_at": "2026-08-21T00:00:00+00:00",
            "case": {
                "batch_name": "historical_batch",
                "batch_id": "historical_batch_id",
                "case_index": index,
                "case_id": f"case_{index:04d}",
            },
            "job_name": f"setup-idle-{index}",
            "command": [
                "sbatch",
                f"--job-name=setup-idle-{index}",
                "generation_campaign_node.sh",
            ],
            "job_id": job_id,
            "status": "submitted",
            "error": None,
        }
        for index, job_id in enumerate(job_ids, start=1)
    ]
    manifest = {
        "schema_kind": "generation_campaign_run",
        "schema_version": 1,
        "campaign_run_id": run_id,
        "campaign_config": "configs/generation/campaigns/removed_historical.yaml",
        "git_commit": git_commit,
        "state": state,
        "slurm_job_ids": list(job_ids),
        "submissions": submissions,
        "submission_intent": None,
    }
    path = directory / "campaign_run.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_setup_idle_benchmark(
    storage: Path,
    run_id: str,
    *,
    job_ids: tuple[str, ...],
    git_commit: str,
) -> Path:
    """Write the scheduler-ownership slice of one historical benchmark."""
    directory = (
        common.paths.get_generation_performance_benchmark_root(
            storage_root=storage,
        )
        / "core_scaling"
        / run_id
    )
    directory.mkdir(parents=True)
    manifest = {
        "schema_kind": generation.benchmark.BENCHMARK_RUN_SCHEMA_KIND,
        "schema_version": 1,
        "benchmark_run_id": run_id,
        "suite_config": "configs/generation/benchmarks/removed_historical.yaml",
        "git_commit": git_commit,
        "measured_job_ids": list(job_ids),
        "submission_history": [{"role": "measure", "job_id": job_id} for job_id in job_ids],
    }
    path = directory / "benchmark_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _install_setup_idle_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_states: dict[str, str] | None = None,
    error: str | None = None,
) -> list[list[str]]:
    """Install one strict fake current-user squeue response."""
    commands: list[list[str]] = []
    states = {} if active_states is None else dict(active_states)

    def scheduler_output(command: list[str]) -> tuple[str, str | None]:
        commands.append(command)
        assert command[0:2] == ["squeue", "--noheader"]
        assert sum(argument.startswith("--user=") for argument in command) == 1
        selection = next(argument.removeprefix("--jobs=") for argument in command if argument.startswith("--jobs="))
        assert command[-1] == "--format=%i|%T"
        rows = [f"{job_id}|{states[job_id]}" for job_id in selection.split(",") if job_id in states]
        return "\n".join(rows), error

    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_output",
        scheduler_output,
    )
    return commands


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
        "admit_transferred_campaign",
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


def test_transfer_evidence_admission_checks_metadata_without_payload_rehash(tmp_path: Path) -> None:
    """Keep routine receipt admission distinct from explicit deep validation."""
    storage = tmp_path / "storage"
    run_directory = campaign_evidence.campaign_run_directory(_RUN_ID, storage_root=storage)
    run_directory.mkdir(parents=True)
    terminal = {"campaign_id": "campaign-id", "git_commit": _COMMIT, "batches": []}
    terminal_path = run_directory / "campaign_terminal.json"
    terminal_path.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    payload_path = run_directory / "payload.bin"
    payload_path.write_bytes(b"first")
    relative_directory = run_directory.relative_to(storage).as_posix()
    files = [
        {
            "relative_path": path.relative_to(storage).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": common.serialization.file_sha256(path),
        }
        for path in (terminal_path, payload_path)
    ]
    files.sort(key=lambda record: record["relative_path"])
    receipt = {
        "schema_kind": "generation_campaign_transfer",
        "schema_version": 1,
        "status": "transfer_complete",
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "campaign_run_id": _RUN_ID,
        "campaign_id": terminal["campaign_id"],
        "git_commit": terminal["git_commit"],
        "source_host": "cpu.example",
        "source_storage_root": "/remote/storage",
        "destination_storage_root": str(storage.resolve()),
        "campaign_terminal_sha256": common.serialization.file_sha256(terminal_path),
        "transferred_file_count": len(files),
        "transferred_bytes": sum(record["size_bytes"] for record in files),
        "transfer_inventory_sha256": common.serialization.canonical_json_sha256(files),
        "files": files,
        "directories": [{"directory": relative_directory}],
        "terminal_validation": {"status": "pass", "batch_count": 0},
        "source_removed": False,
    }
    (run_directory / "transfer_complete.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    plan = {"campaign_directory": relative_directory, "batches": []}

    admitted = campaign_evidence.admit_transfer_receipt(
        _RUN_ID,
        terminal=terminal,
        plan=plan,
        storage_root=storage,
    )
    assert admitted["transfer_inventory_sha256"] == receipt["transfer_inventory_sha256"]

    payload_path.write_bytes(b"other")
    campaign_evidence.admit_transfer_receipt(
        _RUN_ID,
        terminal=terminal,
        plan=plan,
        storage_root=storage,
    )
    with pytest.raises(ValueError, match=r"(?i)transfer.*invalid"):
        campaign_evidence.validate_transfer_receipt(
            _RUN_ID,
            terminal=terminal,
            plan=plan,
            storage_root=storage,
        )


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
    monkeypatch.setattr(generation.campaign, "admit_transferred_campaign", lambda *_args, **_kwargs: {})
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


def test_metadata_only_status_avoids_tree_sizing_and_package_payload_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep normal status to metadata while exact diagnostics remain explicit."""
    storage = tmp_path / "storage"
    metadata = common.paths.get_dataset_metadata_root(storage_root=storage) / "dataset-id"
    metadata.mkdir(parents=True)
    source_directory = storage / "01_generation/raw/batch"
    source_directory.mkdir(parents=True)
    snapshot_calls = 0

    def reject_expensive_status_work(*_args: Any, **_kwargs: Any) -> Any:
        message = "Metadata-only status attempted recursive or payload work."
        raise AssertionError(message)

    def campaign_snapshot(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return {
            "campaign_run_id": _RUN_ID,
            "campaign_state": "running",
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "", "error": None},
        }

    monkeypatch.setattr(generation.workflow, "_tree_size", reject_expensive_status_work)
    monkeypatch.setattr(
        generation.workflow,
        "_manifest_source_directories",
        lambda *_args, **_kwargs: (source_directory,),
    )
    monkeypatch.setattr(generation.campaign, "campaign_status", campaign_snapshot)
    monkeypatch.setattr(
        datasets.packages,
        "load_package_manifest",
        reject_expensive_status_work,
    )
    monkeypatch.setattr(
        datasets.packages,
        "inspect_dataset_package",
        reject_expensive_status_work,
    )
    monkeypatch.setattr(
        datasets.packages,
        "load_package_manifest_evidence",
        lambda *_args, **_kwargs: {
            "dataset_view": "transient_drying",
            "evaluation_regime": "id",
        },
    )

    source_status = generation.workflow.campaign_source_status(
        _RUN_ID,
        storage_root=storage,
        query_scheduler=True,
        include_sizes=False,
    )
    assert snapshot_calls == 1
    assert source_status["reclaimable_bytes"] is None
    assert source_status["size_bytes"] is None
    assert source_status["source_directories"] == [
        {
            "path": str(source_directory),
            "exists": True,
            "size_bytes": None,
        }
    ]

    status = generation.workflow.storage_status(
        storage_root=storage,
        role="gpu",
        run_id=_RUN_ID,
        include_sizes=False,
        include_runs=False,
    )
    assert snapshot_calls == 1
    assert status["generation_total_bytes"] is None
    assert status["datasets_total_bytes"] is None
    assert status["experiments_total_bytes"] is None
    assert status["runs"] == []
    assert status["packages"] == [
        {
            "dataset_id": "dataset-id",
            "dataset_view": "transient_drying",
            "evaluation_regime": "id",
            "size_bytes": None,
        }
    ]
    assert status["transfer_staging_bytes"] is None


def test_shared_setup_idle_succeeds_without_persisted_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid scheduler work when no persisted Generation owner exists."""
    storage = (tmp_path / "storage").resolve()
    storage.mkdir()
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_output",
        lambda _command: pytest.fail("empty setup-idle check queried Slurm"),
    )

    status = generation.workflow.assert_shared_setup_idle(storage_root=storage)

    assert status["status"] == "idle"
    assert status["campaign_run_count"] == 0
    assert status["benchmark_run_count"] == 0


@pytest.mark.parametrize("state", ["cancel_requested", "complete"])
def test_shared_setup_idle_accepts_idle_historical_terminal_campaign(
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore current provenance when exact historical jobs left squeue."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        _RUN_ID,
        job_ids=("41001",),
        git_commit="b" * 40,
        state=state,
    )
    monkeypatch.delenv("GENERATION_GIT_COMMIT", raising=False)
    commands = _install_setup_idle_scheduler(monkeypatch)
    monkeypatch.setattr(
        campaign_evidence,
        "resolve_campaign_config_path",
        lambda _value: pytest.fail("setup-idle resolved historical campaign provenance"),
    )

    status = generation.workflow.assert_shared_setup_idle(storage_root=storage)

    assert status["status"] == "idle"
    assert status["campaign_run_count"] == 1
    assert len(commands) == 1


def test_shared_setup_idle_preserves_failed_campaign_inputs_without_reading_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep retained canonical inputs byte-identical and reusable when jobs are idle."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        _RUN_ID,
        job_ids=("41002",),
        git_commit="c" * 40,
        state="completed_with_failures",
    )
    payload = (
        common.paths.resolve_generation_input_generation_raw_directory(
            "historical_batch",
            "historical_input_generation",
            storage_root=storage,
        )
        / "case_0001"
        / "inputs"
        / "fields.csv"
    )
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"x,y\n1.0,2.0\n")
    original_open = Path.open
    with original_open(payload, "rb") as stream:
        expected_bytes = stream.read()
    expected_stat = payload.stat()

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == payload:
            pytest.fail("setup-idle read canonical scientific payload", pytrace=False)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        common.serialization,
        "file_sha256",
        lambda _path: pytest.fail("setup-idle hashed scientific payload"),
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: pytest.fail("setup-idle reconstructed campaign provenance"),
    )
    monkeypatch.delenv("GENERATION_GIT_COMMIT", raising=False)
    _install_setup_idle_scheduler(monkeypatch)

    status = generation.workflow.assert_shared_setup_idle(storage_root=storage)

    with original_open(payload, "rb") as stream:
        actual_bytes = stream.read()
    actual_stat = payload.stat()
    assert status["status"] == "idle"
    assert actual_bytes == expected_bytes
    assert actual_stat.st_ino == expected_stat.st_ino
    assert actual_stat.st_size == expected_stat.st_size
    assert actual_stat.st_mtime_ns == expected_stat.st_mtime_ns


@pytest.mark.parametrize(
    ("scheduler_state", "job_id"),
    [("RUNNING", "41003"), ("PENDING", "41004")],
)
def test_shared_setup_idle_blocks_active_or_pending_persisted_job(
    scheduler_state: str,
    job_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block replacement for every exact persisted job still in squeue."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        _RUN_ID,
        job_ids=(job_id,),
        git_commit="d" * 40,
        state="active",
    )
    _install_setup_idle_scheduler(
        monkeypatch,
        active_states={job_id: scheduler_state},
    )

    with pytest.raises(
        RuntimeError,
        match=r"blocked by active dependent Generation jobs.*campaign",
    ):
        generation.workflow.assert_shared_setup_idle(storage_root=storage)


def test_shared_setup_idle_rejects_malformed_persisted_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before scheduler access when persisted ownership is malformed."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        _RUN_ID,
        job_ids=("not-a-job-id",),
        git_commit="e" * 40,
        state="active",
    )
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_output",
        lambda _command: pytest.fail("malformed persisted ID reached Slurm"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"cannot validate persisted Generation scheduler ownership.*malformed",
    ):
        generation.workflow.assert_shared_setup_idle(storage_root=storage)


def test_shared_setup_idle_reports_scheduler_query_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report unavailable liveness evidence without calling the job active."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        _RUN_ID,
        job_ids=("41005",),
        git_commit="f" * 40,
        state="active",
    )
    _install_setup_idle_scheduler(monkeypatch, error="squeue unavailable")

    with pytest.raises(RuntimeError) as captured:
        generation.workflow.assert_shared_setup_idle(storage_root=storage)

    assert "cannot query dependent Generation scheduler liveness" in str(captured.value)
    assert "squeue unavailable" in str(captured.value)
    assert "blocked by active" not in str(captured.value)


def test_shared_setup_idle_batches_historical_campaigns_from_different_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one liveness query without resolving either historical campaign."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        "historical_one__0123456789abcdef",
        job_ids=("41006",),
        git_commit="1" * 40,
        state="complete",
    )
    _write_setup_idle_campaign(
        storage,
        "historical_two__0123456789abcdef",
        job_ids=("41007",),
        git_commit="2" * 40,
        state="completed_with_failures",
    )
    monkeypatch.delenv("GENERATION_GIT_COMMIT", raising=False)
    commands = _install_setup_idle_scheduler(monkeypatch)
    monkeypatch.setattr(
        campaign_evidence,
        "resolve_campaign_config_path",
        lambda _value: pytest.fail("setup-idle resolved historical campaign config"),
    )

    status = generation.workflow.assert_shared_setup_idle(storage_root=storage)

    expected_campaign_count = 2
    assert status["status"] == "idle"
    assert status["campaign_run_count"] == expected_campaign_count
    assert len(commands) == 1
    assert "--jobs=41006,41007" in commands[0]


def test_shared_setup_idle_checks_campaigns_and_benchmarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse shared setup for an exact active benchmark job in the batch."""
    storage = (tmp_path / "storage").resolve()
    _write_setup_idle_campaign(
        storage,
        _RUN_ID,
        job_ids=("41008",),
        git_commit="3" * 40,
        state="complete",
    )
    benchmark_id = "core_scaling_transient__0123456789abcdef"
    _write_setup_idle_benchmark(
        storage,
        benchmark_id,
        job_ids=("41009",),
        git_commit="4" * 40,
    )
    commands = _install_setup_idle_scheduler(
        monkeypatch,
        active_states={"41009": "RUNNING"},
    )

    with pytest.raises(
        RuntimeError,
        match=r"active dependent Generation jobs.*benchmark",
    ):
        generation.workflow.assert_shared_setup_idle(storage_root=storage)
    assert len(commands) == 1
    assert "--jobs=41008,41009" in commands[0]


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
