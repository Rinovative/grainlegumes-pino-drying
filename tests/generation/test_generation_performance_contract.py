# ruff: noqa: C901, PLR2004, S101, SLF001
"""Exact operation-count contracts for Generation planning and monitoring."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import pytest
import yaml
from typing_extensions import Self

from src import common, generation
from src.generation import generation_campaign_status as status_service
from src.generation.cases import generation_cases_admission as admission_service
from src.generation.cases import generation_cases_input as input_service
from src.generation.runtime import generation_runtime_batch as batch_runtime
from src.generation.runtime import generation_runtime_cluster as cluster_service

_COMMIT = "a" * 40
_SCALING_CASE_COUNTS = (4, 8, 16, 32)


class _CountingReader:
    """Proxy one test-opened campaign file and count bytes through EOF."""

    def __init__(
        self,
        stream: Any,
        counts: defaultdict[str, int],
        *,
        canonical_payload: bool,
    ) -> None:
        self._stream = stream
        self._counts = counts
        self._canonical_payload = canonical_payload
        self._complete = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self) -> Self:
        self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> Any:
        return self._stream.__exit__(*args)

    def __iter__(self) -> _CountingReader:
        return self

    def __next__(self) -> Any:
        try:
            value = next(self._stream)
        except StopIteration:
            self._record_complete()
            raise
        self._record_bytes(value)
        return value

    def _record_bytes(self, value: Any) -> None:
        if isinstance(value, bytes):
            size = len(value)
        elif isinstance(value, str):
            size = len(value.encode(getattr(self._stream, "encoding", None) or "utf-8"))
        else:
            return
        self._counts["bytes_read"] += size
        if self._canonical_payload:
            self._counts["canonical_payload_bytes_read"] += size

    def _record_complete(self) -> None:
        if self._complete:
            return
        self._complete = True
        self._counts["complete_file_reads"] += 1
        if self._canonical_payload:
            self._counts["canonical_payload_complete_reads"] += 1

    def read(self, *args: Any, **kwargs: Any) -> Any:
        value = self._stream.read(*args, **kwargs)
        self._record_bytes(value)
        requested = args[0] if args else kwargs.get("size", -1)
        if requested is None or requested < 0 or value in {b"", ""}:
            self._record_complete()
        return value

    def readline(self, *args: Any, **kwargs: Any) -> Any:
        value = self._stream.readline(*args, **kwargs)
        self._record_bytes(value)
        if value in {b"", ""}:
            self._record_complete()
        return value

    def readlines(self, *args: Any, **kwargs: Any) -> Any:
        values = self._stream.readlines(*args, **kwargs)
        for value in values:
            self._record_bytes(value)
        requested = args[0] if args else kwargs.get("hint", -1)
        if requested is None or requested < 0:
            self._record_complete()
        return values


def _inside(path: Path | str, root: Path) -> bool:
    """Return whether one path is lexically contained by test storage."""
    try:
        Path(path).relative_to(root)
    except ValueError:
        return False
    return True


def _canonical_payload(path: Path | str, storage: Path) -> bool:
    """Return whether one path is a canonical adapter payload, not its receipt."""
    candidate = Path(path)
    if not _inside(candidate, storage):
        return False
    relative = candidate.relative_to(storage)
    return "input_generations" in relative.parts and "inputs" in relative.parts


def _install_operation_counters(
    monkeypatch: pytest.MonkeyPatch,
    *,
    storage: Path,
) -> defaultdict[str, int]:
    """Instrument repository-owned I/O and reconstruction without production hooks."""
    counts: defaultdict[str, int] = defaultdict(int)

    original_open = Path.open
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_stat = Path.stat
    original_iterdir = Path.iterdir
    original_rglob = Path.rglob
    original_sha256 = common.serialization.file_sha256
    original_csv_reader = csv.reader
    original_json_loads = json.loads
    original_yaml_load = yaml.safe_load
    original_hdf5_file = h5py.File
    original_batch_references = input_service.admit_configured_input_references
    original_case_reference = admission_service._case_reference
    original_reconciled = generation.campaign._reconciled
    original_completed = batch_runtime.completed_case_is_valid
    original_tasks = cluster_service.campaign_tasks
    original_submission_index = generation.campaign._submission_index
    original_atomic_json = common.serialization.atomic_write_json

    def counted_open(candidate: Path, *args: Any, **kwargs: Any) -> Any:
        inside = _inside(candidate, storage)
        canonical_payload = _canonical_payload(candidate, storage)
        if inside:
            counts["file_opens"] += 1
        if canonical_payload:
            counts["canonical_payload_opens"] += 1
        stream = original_open(candidate, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if inside and isinstance(mode, str) and "r" in mode and "+" not in mode:
            return _CountingReader(
                stream,
                counts,
                canonical_payload=canonical_payload,
            )
        return stream

    def counted_read_bytes(candidate: Path) -> bytes:
        if _inside(candidate, storage):
            counts["read_bytes_calls"] += 1
        if _canonical_payload(candidate, storage):
            counts["canonical_payload_read_bytes"] += 1
        return original_read_bytes(candidate)

    def counted_read_text(candidate: Path, *args: Any, **kwargs: Any) -> str:
        if _inside(candidate, storage):
            counts["read_text_calls"] += 1
        if _canonical_payload(candidate, storage):
            counts["canonical_payload_read_text"] += 1
        return original_read_text(candidate, *args, **kwargs)

    def counted_stat(candidate: Path, *args: Any, **kwargs: Any) -> Any:
        if _inside(candidate, storage):
            counts["metadata_stats"] += 1
        return original_stat(candidate, *args, **kwargs)

    def counted_iterdir(candidate: Path) -> Any:
        if _inside(candidate, storage):
            counts["directory_iterdirs"] += 1
        return original_iterdir(candidate)

    def counted_rglob(candidate: Path, pattern: str) -> Any:
        if _inside(candidate, storage):
            counts["recursive_directory_traversals"] += 1
        return original_rglob(candidate, pattern)

    def counted_sha256(candidate: Path | str) -> str:
        if _inside(candidate, storage):
            counts["sha256_calls"] += 1
        if _canonical_payload(candidate, storage):
            counts["canonical_payload_sha256"] += 1
        return original_sha256(candidate)

    def counted_csv_reader(*args: Any, **kwargs: Any) -> Any:
        counts["csv_parser_calls"] += 1
        return original_csv_reader(*args, **kwargs)

    def counted_json_loads(*args: Any, **kwargs: Any) -> Any:
        counts["json_parser_calls"] += 1
        return original_json_loads(*args, **kwargs)

    def counted_yaml_load(*args: Any, **kwargs: Any) -> Any:
        counts["yaml_parser_calls"] += 1
        return original_yaml_load(*args, **kwargs)

    def counted_hdf5_file(*args: Any, **kwargs: Any) -> Any:
        counts["hdf5_opens"] += 1
        return original_hdf5_file(*args, **kwargs)

    def counted_batch_references(*args: Any, **kwargs: Any) -> Any:
        counts["batch_reference_constructions"] += 1
        return original_batch_references(*args, **kwargs)

    def counted_case_reference(*args: Any, **kwargs: Any) -> Any:
        counts["case_reference_constructions"] += 1
        return original_case_reference(*args, **kwargs)

    def counted_reconciliation(*args: Any, **kwargs: Any) -> Any:
        counts["campaign_reconciliations"] += 1
        return original_reconciled(*args, **kwargs)

    def counted_completed(*args: Any, **kwargs: Any) -> bool:
        counts["completed_case_validations"] += 1
        return original_completed(*args, **kwargs)

    def counted_tasks(*args: Any, **kwargs: Any) -> Any:
        counts["work_unit_constructions"] += 1
        return original_tasks(*args, **kwargs)

    def counted_submission_index(*args: Any, **kwargs: Any) -> Any:
        counts["submission_index_constructions"] += 1
        return original_submission_index(*args, **kwargs)

    def counted_atomic_json(candidate: Path | str, *args: Any, **kwargs: Any) -> Any:
        if _inside(candidate, storage):
            counts["state_writes"] += 1
        if Path(candidate).name == "campaign_run.json":
            counts["plan_serializations"] += 1
        return original_atomic_json(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(Path, "read_text", counted_read_text)
    monkeypatch.setattr(Path, "stat", counted_stat)
    monkeypatch.setattr(Path, "iterdir", counted_iterdir)
    monkeypatch.setattr(Path, "rglob", counted_rglob)
    monkeypatch.setattr(common.serialization, "file_sha256", counted_sha256)
    monkeypatch.setattr(csv, "reader", counted_csv_reader)
    monkeypatch.setattr(json, "loads", counted_json_loads)
    monkeypatch.setattr(yaml, "safe_load", counted_yaml_load)
    monkeypatch.setattr(h5py, "File", counted_hdf5_file)
    monkeypatch.setattr(input_service, "admit_configured_input_references", counted_batch_references)
    monkeypatch.setattr(admission_service, "_case_reference", counted_case_reference)
    monkeypatch.setattr(generation.campaign, "_reconciled", counted_reconciliation)
    monkeypatch.setattr(batch_runtime, "completed_case_is_valid", counted_completed)
    monkeypatch.setattr(cluster_service, "campaign_tasks", counted_tasks)
    monkeypatch.setattr(generation.campaign, "_submission_index", counted_submission_index)
    monkeypatch.setattr(common.serialization, "atomic_write_json", counted_atomic_json)
    return counts


def _fake_slurm(
    job_id: str,
    counts: defaultdict[str, int],
    *,
    active: bool,
) -> Any:
    """Return one exact fake Slurm boundary with batched queue evidence."""

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        owner = arguments[0]
        counts["scheduler_subprocess_calls"] += 1
        counts[f"scheduler_{owner}_calls"] += 1
        if owner == "sbatch":
            return subprocess.CompletedProcess(arguments, 0, stdout=f"{job_id}\n", stderr="")
        if owner == "squeue":
            output = f"{job_id}|PENDING|Resources|(Priority)|2026-01-01T00:00:00|N/A|00:00:00\n" if active else ""
            return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")
        if owner == "sacct":
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
        raise AssertionError(arguments)

    return run


def _fake_submission_sequence(
    job_ids: tuple[str, ...],
    counts: defaultdict[str, int],
) -> Any:
    """Return a fake scheduler that assigns one distinct ID per submission."""
    pending_ids = iter(job_ids)

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        owner = arguments[0]
        counts["scheduler_subprocess_calls"] += 1
        counts[f"scheduler_{owner}_calls"] += 1
        if owner == "sbatch":
            return subprocess.CompletedProcess(arguments, 0, stdout=f"{next(pending_ids)}\n", stderr="")
        raise AssertionError(arguments)

    return run


def _prepared_campaign(
    generation_config_factory: Any,
    *,
    case_count: int,
    storage: Path,
    max_admission_cases: int = 1,
) -> Any:
    """Create one compact immutable input publication before measurements."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=case_count,
        max_admission_cases=max_admission_cases,
        maximum_failed_cases=case_count,
        campaign_purpose="family_generalization",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    input_service.prepare_campaign_inputs(
        campaign,
        git_commit=_COMMIT,
        storage_root=storage,
    )
    return campaign


def _synthetic_reconciliation(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_count: int,
    progress: Any,
) -> tuple[tuple[Any, ...], list[dict[str, Any]], int, int]:
    """Run one payload-free synthetic reconciliation for progress tests."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=case_count,
        maximum_failed_cases=case_count,
        campaign_purpose="family_generalization",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = tuple(cluster_service.campaign_tasks(campaign))
    manifest = {
        "campaign_run_id": "progress-presentation__0123456789abcdef",
        "git_commit": _COMMIT,
        "slurm_job_ids": [],
        "submissions": [],
        "submission_intent": None,
        "admission_reservations": [],
    }
    references: dict[str, dict[int, Any]] = {}
    for task in tasks:
        references.setdefault(task.batch_name, {})[task.case_index] = object()

    def task_state(
        _manifest: Any,
        _campaign: Any,
        task: Any,
        _scheduler: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {**generation.campaign._task_payload(task), "state": "never_started"}

    monkeypatch.setattr(
        generation.campaign,
        "_matching_case_reconciliation_evidence",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(generation.campaign, "_task_state", task_state)
    views, pending_jobs, running_jobs = generation.campaign._reconciled(
        manifest,
        campaign,
        {"active": {}, "accounted": {}},
        storage_root=None,
        input_references=references,
        progress=progress,
        tasks=tasks,
        submission_index={},
    )
    return tasks, views, pending_jobs, running_jobs


def test_fast_600_case_reconciliation_emits_no_milestone_noise(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a normal fast 600-case pass silent before its status block."""
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(generation.campaign, "monotonic", lambda: 0.0)

    tasks, views, pending_jobs, running_jobs = _synthetic_reconciliation(
        generation_config_factory,
        monkeypatch,
        case_count=600,
        progress=lambda event: events.append(dict(event)),
    )

    assert len(tasks) == len(views) == 600
    assert [(view["batch_id"], view["case_index"]) for view in views] == [(task.batch_id, task.case_index) for task in tasks]
    assert pending_jobs == running_jobs == 0
    assert events == []


def test_slow_reconciliation_uses_monotonic_time_rate_limited_progress(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose useful monotonic heartbeats only after reconciliation is slow."""
    clock = {"seconds": -1.0}
    observations: list[tuple[float, dict[str, Any]]] = []

    def slow_clock() -> float:
        clock["seconds"] += 1.0
        return clock["seconds"]

    def capture(event: Any) -> None:
        observations.append((clock["seconds"], dict(event)))

    monkeypatch.setattr(generation.campaign, "monotonic", slow_clock)
    tasks, views, pending_jobs, running_jobs = _synthetic_reconciliation(
        generation_config_factory,
        monkeypatch,
        case_count=64,
        progress=capture,
    )

    emitted_at = [timestamp for timestamp, _event in observations]
    completed = [int(event["cases_completed"]) for _timestamp, event in observations]
    assert len(tasks) == len(views) == 64
    assert pending_jobs == running_jobs == 0
    assert 1 < len(observations) < len(tasks)
    assert completed == sorted(set(completed))
    assert all(event["cases_total"] == len(tasks) for _timestamp, event in observations)
    assert all(event["heartbeat"] == "active" for _timestamp, event in observations)
    assert emitted_at[0] >= generation.campaign._RECONCILIATION_PROGRESS_DELAY_SECONDS
    assert all(later - earlier >= generation.campaign._RECONCILIATION_PROGRESS_HEARTBEAT_SECONDS for earlier, later in pairwise(emitted_at))


def test_unchanged_stage3_reuses_immutable_evidence_without_payload_io(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one prepared input generation without rereading adapter payloads."""
    storage = tmp_path / "stage3-reuse-storage"
    campaign = _prepared_campaign(
        generation_config_factory,
        case_count=8,
        storage=storage,
    )
    with monkeypatch.context() as scoped:
        counts = _install_operation_counters(scoped, storage=storage)
        readiness = input_service.prepare_campaign_inputs(
            campaign,
            git_commit=_COMMIT,
            storage_root=storage,
        )

    assert readiness["generated_case_count"] == 0
    assert readiness["reused_case_count"] == campaign.total_case_count
    assert counts["canonical_payload_opens"] == 0
    assert counts["canonical_payload_read_bytes"] == 0
    assert counts["canonical_payload_read_text"] == 0
    assert counts["canonical_payload_bytes_read"] == 0
    assert counts["canonical_payload_complete_reads"] == 0
    assert counts["canonical_payload_sha256"] == 0
    assert counts["csv_parser_calls"] == 0
    assert counts["hdf5_opens"] == 0
    assert counts["recursive_directory_traversals"] == 0
    assert counts["case_reference_constructions"] == campaign.total_case_count
    assert counts["state_writes"] == 0


def test_stage4_reuses_stage3_evidence_with_linear_metadata_work(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan and submit 4/8/16/32 cases with no canonical adapter read."""
    observations: list[dict[str, int]] = []

    def bind_repository_commit(operation_counts: defaultdict[str, int]) -> Any:
        def repository_commit() -> str:
            operation_counts["repository_checks"] += 1
            return _COMMIT

        return repository_commit

    for case_count in _SCALING_CASE_COUNTS:
        storage = tmp_path / f"stage4-storage-{case_count}"
        campaign = _prepared_campaign(
            generation_config_factory,
            case_count=case_count,
            storage=storage,
        )
        job_id = str(9100 + case_count)
        with monkeypatch.context() as scoped:
            counts = _install_operation_counters(scoped, storage=storage)
            scoped.setattr(
                input_service,
                "prepare_campaign_inputs",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Stage 3 preparation repeated during Stage 4")),
            )

            scoped.setattr(
                generation.campaign,
                "_repository_commit",
                bind_repository_commit(counts),
            )
            scoped.setattr(
                generation.campaign.subprocess,
                "run",
                _fake_slurm(job_id, counts, active=False),
            )
            scoped.setattr(
                generation.campaign,
                "_task_submissions",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("submission list rescanned per task")),
            )
            manifest = generation.campaign.submit_campaign(
                campaign,
                git_commit=_COMMIT,
                storage_root=storage,
                inputs_prepared=True,
            )

        assert manifest["slurm_job_ids"] == [job_id]
        assert counts["canonical_payload_opens"] == 0
        assert counts["canonical_payload_read_bytes"] == 0
        assert counts["canonical_payload_read_text"] == 0
        assert counts["canonical_payload_bytes_read"] == 0
        assert counts["canonical_payload_complete_reads"] == 0
        assert counts["canonical_payload_sha256"] == 0
        assert counts["csv_parser_calls"] == 0
        assert counts["hdf5_opens"] == 0
        assert counts["recursive_directory_traversals"] == 0
        assert counts["yaml_parser_calls"] == 0
        assert counts["batch_reference_constructions"] == len(campaign.batches)
        assert counts["case_reference_constructions"] == campaign.total_case_count
        assert counts["campaign_reconciliations"] == 1
        assert counts["completed_case_validations"] == campaign.total_case_count
        assert counts["work_unit_constructions"] == 1
        assert counts["submission_index_constructions"] == 1
        assert counts["repository_checks"] == 1
        assert counts["scheduler_sbatch_calls"] == 1
        assert counts["scheduler_squeue_calls"] == 0
        assert counts["scheduler_sacct_calls"] == 0
        assert counts["plan_serializations"] == 3
        observations.append(dict(counts))

    for smaller, larger in pairwise(observations):
        assert larger["json_parser_calls"] <= 2 * smaller["json_parser_calls"] + 8
        assert larger["file_opens"] <= 2 * smaller["file_opens"] + 8
        assert larger["read_text_calls"] <= 2 * smaller["read_text_calls"] + 8
        assert larger["read_bytes_calls"] <= 2 * smaller["read_bytes_calls"] + 8
        assert larger["complete_file_reads"] <= 2 * smaller["complete_file_reads"] + 8
        assert larger["bytes_read"] <= 2 * smaller["bytes_read"] + 65_536
        assert larger["metadata_stats"] <= 2 * smaller["metadata_stats"] + 32
        assert larger["directory_iterdirs"] <= 2 * smaller["directory_iterdirs"] + 8


def test_stage4_submission_manifest_loads_remain_bounded_as_work_units_scale(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one admitted manifest while preserving every durable submission write."""
    for case_count in _SCALING_CASE_COUNTS:
        storage = tmp_path / f"stage4-all-submissions-{case_count}"
        campaign = _prepared_campaign(
            generation_config_factory,
            case_count=case_count,
            storage=storage,
            max_admission_cases=case_count,
        )
        job_ids = tuple(str(9300 + case_count * 100 + index) for index in range(case_count))

        with monkeypatch.context() as scoped:
            counts = _install_operation_counters(scoped, storage=storage)
            original_manifest_load = generation.campaign.campaign_evidence.load_campaign_run

            def counted_manifest_load(
                *args: Any,
                _counts: defaultdict[str, int] = counts,
                _load: Any = original_manifest_load,
                **kwargs: Any,
            ) -> Any:
                _counts["campaign_manifest_loads"] += 1
                return _load(*args, **kwargs)

            scoped.setattr(
                generation.campaign.campaign_evidence,
                "load_campaign_run",
                counted_manifest_load,
            )
            scoped.setattr(generation.campaign, "_repository_commit", lambda: _COMMIT)
            scoped.setattr(
                generation.campaign.subprocess,
                "run",
                _fake_submission_sequence(job_ids, counts),
            )
            manifest = generation.campaign.submit_campaign(
                campaign,
                git_commit=_COMMIT,
                storage_root=storage,
                inputs_prepared=True,
            )

        persisted = generation.campaign.campaign_evidence.load_campaign_run(
            str(manifest["campaign_run_id"]),
            storage_root=storage,
        )
        assert persisted == manifest
        assert len(manifest["slurm_job_ids"]) == case_count
        assert counts["scheduler_sbatch_calls"] == case_count
        assert counts["plan_serializations"] == 1 + 2 * case_count
        assert counts["campaign_manifest_loads"] == 1
        assert counts["canonical_payload_opens"] == 0
        assert counts["canonical_payload_bytes_read"] == 0
        assert counts["canonical_payload_sha256"] == 0


def test_unchanged_monitor_scaling_is_linear_metadata_work(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove 4/8/16/32-case polls avoid payload work and batch scheduler I/O."""
    observations: list[dict[str, int]] = []
    for case_count in _SCALING_CASE_COUNTS:
        storage = tmp_path / f"monitor-storage-{case_count}"
        campaign = _prepared_campaign(
            generation_config_factory,
            case_count=case_count,
            storage=storage,
        )
        job_id = str(9200 + case_count)
        setup_counts: defaultdict[str, int] = defaultdict(int)
        with monkeypatch.context() as scoped:
            scoped.setattr(generation.campaign, "_repository_commit", lambda: _COMMIT)
            scoped.setattr(generation.campaign.subprocess, "run", _fake_slurm(job_id, setup_counts, active=False))
            manifest = generation.campaign.submit_campaign(
                campaign,
                git_commit=_COMMIT,
                storage_root=storage,
                inputs_prepared=True,
            )
        run_id = str(manifest["campaign_run_id"])

        with monkeypatch.context() as scoped:
            counts = _install_operation_counters(scoped, storage=storage)
            scoped.setattr(generation.campaign.subprocess, "run", _fake_slurm(job_id, counts, active=True))
            scoped.setattr(
                generation.campaign,
                "_repository_commit",
                lambda: (_ for _ in ()).throw(AssertionError("unchanged poll checked mutable checkout without admission")),
            )
            scoped.setattr(
                generation.campaign,
                "_task_submissions",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("submission list rescanned per task")),
            )
            scoped.setattr(
                generation.workflow,
                "_tree_size",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("monitor recursively sized campaign storage")),
            )
            snapshot = generation.campaign.resume_campaign_monitor_snapshot(
                run_id,
                storage_root=storage,
            )
            source = generation.workflow.campaign_source_status_from_snapshot(
                run_id,
                campaign_status=snapshot["status"],
                storage_root=storage,
            )
            rendered = status_service.format_workflow_monitor(
                snapshot["status"],
                source,
                max_active_cases=8,
            )

        assert rendered.startswith("campaign-monitor\t")
        assert "\nsource-monitor\t" in rendered
        assert counts["canonical_payload_opens"] == 0
        assert counts["canonical_payload_read_bytes"] == 0
        assert counts["canonical_payload_read_text"] == 0
        assert counts["canonical_payload_bytes_read"] == 0
        assert counts["canonical_payload_complete_reads"] == 0
        assert counts["canonical_payload_sha256"] == 0
        assert counts["csv_parser_calls"] == 0
        assert counts["hdf5_opens"] == 0
        assert counts["recursive_directory_traversals"] == 0
        assert counts["batch_reference_constructions"] == len(campaign.batches)
        assert counts["case_reference_constructions"] == case_count
        assert counts["campaign_reconciliations"] == 1
        assert counts["completed_case_validations"] == case_count + 1
        assert counts["work_unit_constructions"] == 1
        assert counts["submission_index_constructions"] == 1
        assert counts["scheduler_squeue_calls"] == 1
        assert counts["scheduler_sacct_calls"] == 1
        assert counts["scheduler_sbatch_calls"] == 0
        assert counts["plan_serializations"] == 0
        assert counts["state_writes"] == 0
        observations.append(dict(counts))

    for smaller, larger in pairwise(observations):
        assert larger["json_parser_calls"] <= 2 * smaller["json_parser_calls"] + 8
        assert larger["file_opens"] <= 2 * smaller["file_opens"] + 8
        assert larger["read_text_calls"] <= 2 * smaller["read_text_calls"] + 8
        assert larger["read_bytes_calls"] <= 2 * smaller["read_bytes_calls"] + 8
        assert larger["complete_file_reads"] <= 2 * smaller["complete_file_reads"] + 8
        assert larger["bytes_read"] <= 2 * smaller["bytes_read"] + 65_536
        assert larger["metadata_stats"] <= 2 * smaller["metadata_stats"] + 32
        assert larger["directory_iterdirs"] <= 2 * smaller["directory_iterdirs"] + 8


def test_prepared_input_size_conflict_fails_before_scheduler_submission(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed on a truncated immutable payload using its cheap size evidence."""
    storage = tmp_path / "truncated-input-storage"
    campaign = _prepared_campaign(
        generation_config_factory,
        case_count=4,
        storage=storage,
    )
    batch = campaign.batches[0]
    generation_id = input_service.configured_input_generation_id(batch)
    raw = common.paths.resolve_generation_input_generation_raw_directory(
        batch.batch_storage_name,
        generation_id,
        storage_root=storage,
    )
    fields = raw / batch.case_id(batch.case_indices[0]) / "inputs" / "fields.csv"
    payload = fields.read_bytes()
    fields.write_bytes(payload[:-1])
    scheduler_calls: list[list[str]] = []

    def unexpected_scheduler(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        scheduler_calls.append(list(command))
        raise AssertionError(command)

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: _COMMIT)
    monkeypatch.setattr(generation.campaign.subprocess, "run", unexpected_scheduler)
    with pytest.raises((FileExistsError, ValueError), match="hash or size"):
        generation.campaign.submit_campaign(
            campaign,
            git_commit=_COMMIT,
            storage_root=storage,
            inputs_prepared=True,
        )
    assert scheduler_calls == []


@pytest.mark.parametrize("case_count", _SCALING_CASE_COUNTS)
def test_compute_case_lookup_and_batch_gate_avoid_campaign_reconstruction(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_count: int,
) -> None:
    """Keep one compute job to direct membership and completion-marker metadata."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=case_count,
        campaign_purpose="family_generalization",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    case_index = batch.case_indices[0]
    expected = cluster_service.CampaignTask(
        batch_name=batch.batch_name,
        batch_id=batch.batch_id,
        case_index=case_index,
        case_id=batch.case_id(case_index),
    )
    marker_root = tmp_path / "processed"
    for configured_index in batch.case_indices:
        directory = marker_root / batch.case_id(configured_index)
        directory.mkdir(parents=True)
        (directory / "_SUCCESS").write_text("{}\n", encoding="utf-8")

    def reject_campaign_reconstruction(*_args: Any, **_kwargs: Any) -> Any:
        message = "One compute case rebuilt or scanned the complete campaign task list."
        raise AssertionError(message)

    def reject_completed_case_validation(*_args: Any, **_kwargs: Any) -> Any:
        message = "One compute case batch-gated through per-case publication validation."
        raise AssertionError(message)

    outcome = object()
    finalized: list[str] = []
    marker_checks = 0
    original_is_file = Path.is_file

    def counted_is_file(candidate: Path) -> bool:
        nonlocal marker_checks
        if candidate.name == "_SUCCESS" and candidate.parent.parent == marker_root:
            marker_checks += 1
        return original_is_file(candidate)

    monkeypatch.setattr(Path, "is_file", counted_is_file)
    monkeypatch.setattr(
        cluster_service,
        "campaign_tasks",
        reject_campaign_reconstruction,
    )
    monkeypatch.setattr(
        cluster_service.runtime_service,
        "processed_case_directory",
        lambda config, current_index, **_kwargs: marker_root / config.case_id(current_index),
    )
    monkeypatch.setattr(
        cluster_service.runtime_service,
        "completed_case_is_valid",
        reject_completed_case_validation,
    )
    monkeypatch.setattr(
        cluster_service.runtime_service,
        "run_case",
        lambda *_args, **_kwargs: outcome,
    )
    monkeypatch.setattr(
        cluster_service.runtime_service,
        "finalize_batch",
        lambda config, **_kwargs: finalized.append(config.batch_id),
    )

    resolved = cluster_service.require_campaign_task(
        campaign,
        batch_name=batch.batch_name,
        case_index=case_index,
    )
    assert resolved == expected
    assert (
        cluster_service.run_campaign_case(
            campaign,
            resolved,
            cores_per_case=1,
            storage_root=tmp_path,
            work_root=tmp_path / "work",
        )
        is outcome
    )
    assert finalized == [batch.batch_id]
    assert marker_checks == case_count


def test_unchanged_posthoc_gpu_preparation_reuses_shard_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep repeated post-hoc preparation source-free after immutable publication."""
    from src.datasets.packages import dataset_packages_transient_shards as shard_service  # noqa: PLC0415

    storage = tmp_path / "posthoc-storage"
    run_id = "posthoc__0123456789abcdef"
    run_directory = generation.workflow.campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    )
    run_directory.mkdir(parents=True)
    terminal_path = run_directory / "campaign_terminal.json"
    transfer_path = run_directory / "transfer_complete.json"
    terminal_path.write_text(
        json.dumps({"campaign_id": "campaign-id", "git_commit": _COMMIT}) + "\n",
        encoding="utf-8",
    )
    transfer_path.write_text("{}\n", encoding="utf-8")
    generation.workflow._dataset_receipt_path(
        run_id,
        storage_root=storage,
    ).write_text("{}\n", encoding="utf-8")
    shard_receipt_path = run_directory / "shard-receipt.json"
    shard_receipt_path.write_text("{}\n", encoding="utf-8")
    plan = {
        "dataset_name": "transient_drying__lentil__id",
        "dataset_view": "transient_drying",
        "evaluation_regime": "id",
        "training_payload": {
            "required": True,
            "target_shard_bytes": 1024,
        },
    }
    record = {
        **plan,
        "dataset_id": "transient-dataset-id",
        "build_status": "reused",
    }
    campaign = SimpleNamespace(
        campaign_id="campaign-id",
        dataset_packages=(plan,),
    )
    compare_modes: list[bool] = []

    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {"state": "complete"},
    )
    monkeypatch.setattr(
        generation.workflow.campaign_evidence,
        "current_campaign_for_run",
        lambda *_args, **_kwargs: campaign,
    )
    monkeypatch.setattr(
        generation.workflow.campaign_runtime,
        "admit_transferred_campaign",
        lambda *_args, **_kwargs: {"transfer_inventory_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        generation.workflow,
        "build_campaign_datasets",
        lambda *_args, **_kwargs: {"packages": [record]},
    )
    monkeypatch.setattr(
        shard_service,
        "build_transient_shards",
        lambda *_args, **_kwargs: {
            "status": "reused",
            "receipt_path": shard_receipt_path,
            "receipt": {
                "derived_payload_id": "c" * 64,
                "shard_count": 1,
                "case_count": 4,
                "total_size_bytes": 4096,
            },
        },
    )

    def smoke(
        _record: Any,
        *,
        storage_root: Path,
        compare_canonical: bool,
    ) -> dict[str, Any]:
        assert storage_root == storage
        compare_modes.append(compare_canonical)
        return {"one_step_transition": {"status": "equivalent"}}

    monkeypatch.setattr(
        generation.workflow,
        "_smoke_transient_shard_backend",
        smoke,
    )
    monkeypatch.setattr(
        generation.workflow,
        "validate_campaign_package_state",
        lambda *_args, **_kwargs: {
            "package_request_digest": "d" * 64,
            "training_payload_ready_count": 1,
        },
    )

    result = generation.workflow.prepare_gpu_datasets(
        run_id,
        storage_root=storage,
    )

    assert result["status"] == "complete"
    assert result["transient_training_payloads"][0]["status"] == "reused"
    assert compare_modes == [False]
