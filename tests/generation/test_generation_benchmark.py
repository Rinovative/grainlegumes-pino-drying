"""Core-selection benchmark contracts on node-job sweeps.

Responsibilities:
  - Check suite v2 resolution, sweep tickets, node-job submission and recovery, and variant-attempt classification
  - Check the abort and retry rules, runner binding, measured-throughput reporting, summary binding, cancel, and status

Design principles:
  - Use the real suite loader on a small test-owned campaign (4 node cores, k = 1, 2, 4, three cases)
  - Fake only Slurm and the solver; tickets, receipts, and manifests are real files

This module does NOT:
  - Run COMSOL, real Slurm jobs, or the node worker process
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation
from src.generation import generation_benchmark_config as benchmark_config
from src.generation import generation_benchmark_report as benchmark_report
from src.generation import generation_campaign_status as status_service
from src.generation.runtime import generation_runtime_node as node_service

if TYPE_CHECKING:
    from pathlib import Path

_COMMIT = "a" * 40
_FINGERPRINT = "f" * 64
_RUN_ID = "core_scaling_transient__0123456789abcdef"
_ORIGIN = datetime(2026, 10, 27, 20, 0, tzinfo=UTC)

benchmark = generation.benchmark


def _exact_node_type_exclusions(*_args: Any, match: str, **_kwargs: Any) -> list[str]:
    """Fake sinfo exclusions that require the benchmark's exact node-type rule."""
    assert match == "exact", "the benchmark measures per node type"
    return ["hpc111"]


def _suite(generation_config_factory: Any, **overrides: Any) -> benchmark_config.CoreBenchmarkSuite:
    """Write and load one small v3 suite (three-case pool, every case needed) over a test-owned transient campaign."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        scheduler_kind="slurm",
        natural_count=3,
        cores_per_case=2,
        cores_per_node=4,
    )
    project_root = common.paths.get_project_root().resolve()
    launcher = project_root / "scripts" / "generation_node.sh"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("#!/bin/bash\n", encoding="utf-8")
    campaign = config_path.resolve().relative_to(project_root).as_posix()
    suite = {
        "schema_kind": benchmark_config.BENCHMARK_SUITE_SCHEMA_KIND,
        "schema_version": 3,
        "suite_name": "small_core_selection",
        "benchmark_mode": "core_selection",
        "case_pool": {
            "campaign_config": campaign,
            "material_family": "lentil",
            "sampling_regime": "natural",
            "first_case_index": 1,
            "case_count": 3,
        },
        "target_successful_cases": 3,
        "sweeps": [{"sweep_id": "sweep_1", "order": "ascending"}, {"sweep_id": "sweep_2", "order": "descending"}],
        "retry": {"maximum_variant_attempts": 2},
        "resources": {"partition": None, "sweep_wall_time": "02:00:00", "scheduler_options": []},
        "production_interpretation": {
            "campaign_config": campaign,
            "cores_config": (config_path.parent / "execution.yaml").resolve().relative_to(project_root).as_posix(),
            "cores_key": "cluster.cores_per_case",
        },
        "variants": [{"variant_id": f"cores_{cores:02d}", "cores_per_case": cores} for cores in (1, 2, 4)],
    }
    suite.update(overrides)
    suite_path = config_path.parent / "suite.yaml"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    return benchmark_config.load_core_benchmark_suite(suite_path, require_executable=False)


def _write_receipt(node_directory: Path, seq: int, *, started: datetime, ended: datetime | None, exit_code: int = 0) -> None:
    """Write the worker receipts of one ticket."""
    receipts = node_directory / node_service.RECEIPTS_DIRECTORY
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f"{seq:06d}.started.json").write_text(
        json.dumps(
            {
                "schema": node_service.STARTED_RECEIPT_SCHEMA,
                "schema_version": 1,
                "seq": seq,
                "started_at": started.isoformat(),
                "host": "hpc117",
                "worker_pid": 1,
            }
        ),
        encoding="utf-8",
    )
    if ended is not None:
        (receipts / f"{seq:06d}.finished.json").write_text(
            json.dumps(
                {
                    "schema": node_service.FINISHED_RECEIPT_SCHEMA,
                    "schema_version": 1,
                    "seq": seq,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "exit_code": exit_code,
                    "host": "hpc117",
                    "outcome": "exited",
                }
            ),
            encoding="utf-8",
        )


class _Bench:
    """One synthetic benchmark directory, manifest, and fake Slurm submission around a real suite."""

    def __init__(self, suite: benchmark_config.CoreBenchmarkSuite, storage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.suite = suite
        self.storage = storage
        self.directory = benchmark.core_benchmark_directory(_RUN_ID, storage_root=storage)
        self.directory.mkdir(parents=True)
        self.manifest_path = self.directory / "benchmark_manifest.json"
        self.manifest: dict[str, Any] = {
            "benchmark_run_id": _RUN_ID,
            "git_commit": _COMMIT,
            "suite_config": "configs/suite.yaml",
            "state": "inputs_ready",
            "measured_job_ids": [],
            "node_jobs": [],
            "node_job_intent": None,
        }
        self.next_job = 5000
        self.submitted: list[list[str]] = []
        self.outcomes: dict[str, tuple[str, str]] = {}
        monkeypatch.setenv("GENERATION_GIT_COMMIT", _COMMIT)
        monkeypatch.setenv("GENERATION_SOURCE_SHA256", _FINGERPRINT)
        monkeypatch.setenv("GENERATION_NATIVE_VENV", str(common.paths.get_runtime_root().resolve() / "venvs" / "native"))
        monkeypatch.setattr(benchmark.cluster_service, "standard_node_exclusions", _exact_node_type_exclusions)
        monkeypatch.setattr(benchmark.cluster_service, "lease_executing_tree", lambda *_a: None)
        monkeypatch.setattr(benchmark.source_service, "validate_admitted_source_before_publication", lambda: None)
        monkeypatch.setattr(benchmark, "_submit", self._submit)
        monkeypatch.setattr(benchmark, "_unit_record", self._unit_record)
        for representative in suite.representative_cases:
            proof = benchmark._canonical_case_proof_path(self.directory, representative)
            proof.parent.mkdir(parents=True, exist_ok=True)
            proof.write_text("{}\n", encoding="utf-8")

    def _submit(self, command: list[str], **_kwargs: Any) -> str:
        assert json.loads(self.manifest_path.read_text(encoding="utf-8"))["node_job_intent"] is not None
        self.submitted.append(command)
        self.next_job += 1
        return str(self.next_job)

    def _unit_record(self, _directory: Path, suite: benchmark_config.CoreBenchmarkSuite, unit: Any) -> dict[str, Any]:
        unit_id = benchmark._unit_id(suite, unit)
        record = benchmark._pending_record(suite, unit)
        if unit_id not in self.outcomes:
            return record
        status, job_id = self.outcomes[unit_id]
        return {
            **record,
            "status": status,
            "resource": {"slurm_job_id": job_id, "peak_memory_bytes": 2**30},
            "timings_seconds": {"comsol_process_seconds": 100.0},
        }

    def submit(self, sweep_index: int, attempts: dict[int, int] | None = None) -> dict[str, Any]:
        sweep = self.suite.sweeps[sweep_index]
        variant_attempts = [(variant, (attempts or {}).get(variant.cores_per_case, 1)) for variant in self.suite.sweep_variants(sweep)]
        benchmark._submit_sweep_node_job(
            self.manifest,
            self.suite,
            storage=self.storage,
            directory=self.directory,
            manifest_path=self.manifest_path,
            sweep=sweep,
            variant_attempts=variant_attempts,
        )
        return self.manifest["node_jobs"][-1]

    def node_directory(self, record: dict[str, Any]) -> Path:
        return benchmark._node_directory(self.directory, record)

    def finish_variant(self, record: dict[str, Any], cores: int, *, failures: int = 0, offset_minutes: int = 0, seconds: float = 600.0) -> None:
        """Mark every unit of one variant of a node job finished: receipts plus success or failure evidence."""
        tickets = [ticket for ticket in record["tickets"] if self.suite.variant(ticket["variant_id"]).cores_per_case == cores]
        for index, ticket in enumerate(tickets):
            started = _ORIGIN + timedelta(minutes=offset_minutes)
            _write_receipt(self.node_directory(record), ticket["seq"], started=started, ended=started + timedelta(seconds=seconds))
            self.outcomes[ticket["work_unit_id"]] = ("failed" if index < failures else "success", str(record["job_id"]))

    def settle(self, record: dict[str, Any]) -> None:
        """Pretend the driver saw this node job leave the queue long enough ago."""
        record["terminal_observed_at"] = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()


@pytest.fixture
def bench(generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Bench:
    """Return a synthetic benchmark around the small real suite."""
    storage = tmp_path / "storage"
    storage.mkdir()
    return _Bench(_suite(generation_config_factory), storage, monkeypatch)


# -- suite and tickets --


def test_suite_derives_node_resources_sweeps_and_work_unit_ids(generation_config_factory: Any) -> None:
    """Concurrency and node memory come from the execution resolver per k; sweeps order the variants."""
    suite = _suite(generation_config_factory)
    resources = {variant.cores_per_case: suite.resources(variant) for variant in suite.variants}
    assert {cores: item.concurrent_cases for cores, item in resources.items()} == {1: 4, 2: 2, 4: 1}
    assert {cores: item.node_job_memory for cores, item in resources.items()} == {1: "5G", 2: "3G", 4: "2G"}
    assert suite.sweep_node_job_memory() == "5G"
    assert [[variant.cores_per_case for variant in suite.sweep_variants(sweep)] for sweep in suite.sweeps] == [[1, 2, 4], [4, 2, 1]]
    variant = suite.variants[1]
    assert suite.work_unit_id(suite.sweeps[1], variant, 2, 3) == f"sweep_2__a02__{suite.execution_id(variant)}__case_03"
    with pytest.raises(ValueError, match="variant attempt"):
        suite.work_unit_id(suite.sweeps[0], variant, 0, 1)
    contract = suite.resource_contract()
    assert (contract["node_jobs"], contract["node_job_memory"], contract["node_job_wall_time"]) == (2, "5G", "02:00:00")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 1}, "no earlier version is read"),
        ({"variants": [{"variant_id": "cores_02", "cores_per_case": 2}, {"variant_id": "cores_03", "cores_per_case": 3}]}, "must divide"),
        ({"sweeps": [{"sweep_id": "a", "order": "ascending"}, {"sweep_id": "b", "order": "ascending"}]}, "distinct orders"),
        ({"resources": {"partition": None, "sweep_wall_time": "25:00:00", "scheduler_options": []}}, "must not exceed 24:00:00"),
        ({"resources": {"partition": None, "sweep_wall_time": "02:00:00", "scheduler_options": ["--exclude=hpc111"]}}, "owned by the launcher"),
        ({"target_successful_cases": 4}, "must not exceed the case pool"),
    ],
)
def test_suite_rejects_unsupported_schema_pool_and_resources(generation_config_factory: Any, overrides: dict[str, Any], message: str) -> None:
    """The loader fails closed on an old schema, a target above the pool, non-dividing k, duplicate orders, long sweeps, launcher options."""
    with pytest.raises(ValueError, match=message):
        _suite(generation_config_factory, **overrides)


def test_sweep_tickets_group_variants_in_order_and_end_with_stop(bench: _Bench) -> None:
    """One group per variant in sweep order, one step per case with k cores, then the stop ticket."""
    suite = bench.suite
    sweep = suite.sweeps[1]
    tickets = benchmark._sweep_tickets(
        suite, run_id=_RUN_ID, sweep=sweep, variant_attempts=[(variant, 1) for variant in suite.sweep_variants(sweep)], token="t" * 16
    )
    steps = [ticket for ticket in tickets if ticket.kind == node_service.STEP_KIND]
    assert [(ticket.group, ticket.cores) for ticket in steps] == [(0, 4)] * 3 + [(1, 2)] * 3 + [(2, 1)] * 3
    assert tickets[-1].kind == node_service.STOP_KIND
    assert [ticket.seq for ticket in tickets] == list(range(1, 11))
    assert len({ticket.label for ticket in steps}) == len(steps)
    timeout = int(suite.case_config.execution_values["runtime"]["timeout_seconds"])
    first = steps[0]
    assert first.argv[2:] == ("benchmark-case", _RUN_ID, "sweep_2", "cores_04", "1", "case_01")
    assert (first.memory, first.required_seconds, dict(first.env)) == (
        "1G",
        timeout + 720,
        {"GENERATION_BENCHMARK_RUN_ID": _RUN_ID, "GENERATION_NODE_TOKEN": "t" * 16},
    )


# -- benchmark-owned inputs --


def test_inputs_are_benchmark_owned_and_never_enter_the_production_campaign(bench: _Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Materialization needs no campaign plan, writes bundles and proofs into the run, and reuses both."""
    suite = bench.suite
    proofs = [benchmark._canonical_case_proof_path(bench.directory, representative) for representative in suite.representative_cases]
    for proof in proofs:
        proof.unlink()
    monkeypatch.setattr(benchmark, "load_core_benchmark_manifest", lambda *_a, **_k: (bench.manifest, suite))
    monkeypatch.setattr(benchmark, "_require_current_checkout", lambda _manifest: None)
    work = tmp_path / "work"

    paths = benchmark._materialize_core_benchmark_inputs(_RUN_ID, storage_root=bench.storage, work_root=work)

    assert list(paths) == proofs
    roles = sorted(representative.case_role for representative in suite.representative_cases)
    assert sorted(path.name for path in (bench.directory / "inputs").iterdir()) == roles
    for position, representative in enumerate(suite.representative_cases, start=1):
        assert (benchmark._benchmark_input_directory(bench.directory, representative) / "case.json").is_file()
        assert benchmark._load_case_proof(_RUN_ID, suite, position, storage_root=bench.storage)["case_role"] == representative.case_role
    parts = common.paths.campaign_batch_parts(suite.case_config.batch_storage_name)
    assert parts is not None
    campaign = common.paths.resolve_generation_campaign_directory(parts[0], parts[1], storage_root=bench.storage)
    assert not campaign.exists()
    assert not list(work.iterdir())

    def forbid_generation(*_args: Any, **_kwargs: Any) -> None:
        message = "benchmark inputs must be reused"
        raise AssertionError(message)

    monkeypatch.setattr(benchmark.case_service, "generate_case_input_bundle", forbid_generation)
    before = {path: path.read_bytes() for path in proofs}
    assert benchmark._materialize_core_benchmark_inputs(_RUN_ID, storage_root=bench.storage, work_root=work) == paths
    assert {path: path.read_bytes() for path in proofs} == before
    proofs[0].unlink()
    benchmark._materialize_core_benchmark_inputs(_RUN_ID, storage_root=bench.storage, work_root=work)
    first = json.loads(proofs[0].read_text(encoding="utf-8"))
    assert first["case_payload_sha256"] == json.loads(before[proofs[0]])["case_payload_sha256"]


# -- node-job submission and recovery --


def test_node_job_submission_persists_intent_and_tickets_before_sbatch(bench: _Bench) -> None:
    """Intent and every ticket exist before sbatch; the job id, token, and seat are recorded afterwards."""
    record = bench.submit(0)
    assert record["status"] == "submitted"
    assert record["job_id"] == "5001"
    assert bench.manifest["measured_job_ids"] == ["5001"]
    assert bench.manifest["node_job_intent"] is None
    assert record["seat"] == 1
    assert (bench.node_directory(record) / "tickets" / f"{record['stop_ticket_seq']:06d}.json").is_file()
    command = bench.submitted[0]
    assert "--job-name=gen-seat-1" in command
    assert "--dependency=singleton" in command
    assert f"--comment={record['token']}" in command
    assert "--exclude=hpc111" in command
    benchmark._validated_node_job_ids(bench.manifest)


def test_failed_sbatch_marks_submission_failed_and_clears_intent(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused submission leaves an explicit failed record and no unresolved intent."""

    def refuse(*_args: Any, **_kwargs: Any) -> str:
        raise benchmark.subprocess.CalledProcessError(1, ["sbatch"], stderr="refused")

    monkeypatch.setattr(benchmark, "_submit", refuse)
    with pytest.raises(benchmark.subprocess.CalledProcessError):
        bench.submit(0)
    assert bench.manifest["node_jobs"][0]["status"] == "submission_failed"
    assert bench.manifest["node_job_intent"] is None
    assert bench.manifest["measured_job_ids"] == []


@pytest.mark.parametrize("source", ["queue", "worker_record", "absent"])
def test_intent_recovery_uses_queue_comment_then_worker_record(bench: _Bench, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    """A submission interrupted after sbatch is recovered by its token, else by node.json, else marked failed."""
    record = bench.submit(0)
    record.update(status="submitting", job_id=None, submitted_at=None, recorded_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat())
    bench.manifest["measured_job_ids"] = []
    bench.manifest["node_job_intent"] = record["token"]
    queue = {"777": record["token"], "778": "other-token"} if source == "queue" else {"778": "other-token"}
    seat_jobs = {job_id: (1, "RUNNING", comment) for job_id, comment in queue.items()}
    monkeypatch.setattr(benchmark.cluster_service, "query_seat_jobs", lambda _seats: seat_jobs)
    if source == "worker_record":
        (bench.node_directory(record) / "node.json").write_text(
            json.dumps({"schema": node_service.NODE_RECORD_SCHEMA, "schema_version": 1, "job_id": "779"}), encoding="utf-8"
        )
    benchmark._recover_node_job_intent(bench.manifest, directory=bench.directory, manifest_path=bench.manifest_path)
    expected = {"queue": ("submitted", "777"), "worker_record": ("submitted", "779"), "absent": ("submission_failed", None)}[source]
    assert (record["status"], record["job_id"]) == expected
    assert bench.manifest["node_job_intent"] is None
    benchmark._validated_node_job_ids(bench.manifest)


# -- variant and sweep classification --


def _variant(bench: _Bench, sweep_index: int, cores: int, active: frozenset[str]) -> dict[str, Any]:
    suite = bench.suite
    variant = next(item for item in suite.variants if item.cores_per_case == cores)
    return benchmark._variant_state(bench.manifest, suite, bench.directory, suite.sweeps[sweep_index], variant, active)


def test_variant_attempt_states_follow_evidence_receipts_and_liveness(bench: _Bench) -> None:
    """Measured, running, interrupted (a started case), and not started (no case started) are distinguished."""
    record = bench.submit(0)
    bench.finish_variant(record, 1)
    one_case = next(ticket for ticket in record["tickets"] if ticket["variant_id"] == "cores_02")
    _write_receipt(bench.node_directory(record), one_case["seq"], started=_ORIGIN, ended=None)
    active = frozenset({record["job_id"]})
    assert _variant(bench, 0, 1, active)["state"] == "measured"
    assert _variant(bench, 0, 2, active)["state"] == "running"
    gone: frozenset[str] = frozenset()
    assert _variant(bench, 0, 2, gone)["state"] == "running", "a node job not yet observed outside the queue is still judged running"
    bench.settle(record)
    assert _variant(bench, 0, 2, gone)["state"] == "interrupted"
    assert _variant(bench, 0, 4, gone)["state"] == "not_started"
    state = benchmark._sweep_state(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], gone)
    assert state["state"] == "ready_to_submit"
    assert [(variant.cores_per_case, attempt) for variant, attempt in state["next_variant_attempts"]] == [(2, 2), (4, 2)], (
        "a new node job gives every variant a new attempt"
    )


def test_failed_step_without_evidence_is_a_terminal_case_failure(bench: _Bench) -> None:
    """A step that exits non-zero without writing evidence ends its unit as failed_without_evidence."""
    record = bench.submit(0)
    tickets = [ticket for ticket in record["tickets"] if ticket["variant_id"] == "cores_01"]
    for ticket in tickets[:-1]:
        bench.outcomes[ticket["work_unit_id"]] = ("success", str(record["job_id"]))
        _write_receipt(bench.node_directory(record), ticket["seq"], started=_ORIGIN, ended=_ORIGIN + timedelta(minutes=5))
    _write_receipt(bench.node_directory(record), tickets[-1]["seq"], started=_ORIGIN, ended=_ORIGIN + timedelta(minutes=1), exit_code=137)
    state = _variant(bench, 0, 1, frozenset({record["job_id"]}))
    assert state["state"] == "measured"
    assert [item["outcome"] for item in state["units"]] == ["success", "success", "failed_without_evidence"]


def test_failed_candidates_are_replaced_and_the_rest_skipped_after_the_target(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a target below the pool, tickets carry it; a failure never stops the sweep, and skipped candidates complete the variant."""
    storage = tmp_path / "storage"
    storage.mkdir()
    bench = _Bench(_suite(generation_config_factory, target_successful_cases=1), storage, monkeypatch)
    record = bench.submit(0)
    tickets = [ticket for ticket in record["tickets"] if bench.suite.variant(ticket["variant_id"]).cores_per_case == 1]
    ticket_files = [
        json.loads(node_service.ticket_path(bench.node_directory(record), ticket["seq"]).read_text(encoding="utf-8")) for ticket in tickets
    ]
    assert {payload["success_target"] for payload in ticket_files} == {1}
    # The first candidate fails, the second succeeds and reaches the target, the worker skips the third.
    first, second, third = tickets
    _write_receipt(bench.node_directory(record), first["seq"], started=_ORIGIN, ended=_ORIGIN + timedelta(minutes=9), exit_code=2)
    bench.outcomes[first["work_unit_id"]] = ("failed", str(record["job_id"]))
    _write_receipt(bench.node_directory(record), second["seq"], started=_ORIGIN, ended=_ORIGIN + timedelta(minutes=10))
    bench.outcomes[second["work_unit_id"]] = ("success", str(record["job_id"]))
    receipts = bench.node_directory(record) / node_service.RECEIPTS_DIRECTORY
    (receipts / f"{third['seq']:06d}.finished.json").write_text(
        json.dumps(
            {
                "schema": node_service.FINISHED_RECEIPT_SCHEMA,
                "schema_version": 1,
                "seq": third["seq"],
                "started_at": None,
                "ended_at": (_ORIGIN + timedelta(minutes=10)).isoformat(),
                "exit_code": None,
                "host": "hpc117",
                "outcome": "skipped",
            }
        ),
        encoding="utf-8",
    )
    variant = _variant(bench, 0, 1, frozenset({record["job_id"]}))
    assert variant["state"] == "measured"
    assert [item["outcome"] for item in variant["units"]] == ["failed", "success", "skipped"]
    sweep = benchmark._sweep_state(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], frozenset({record["job_id"]}))
    assert sweep["state"] == "running"
    measurement = benchmark._measurement_input(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], variant)
    row = benchmark_report.summarize_core_benchmark_results(bench.suite, [measurement])["measurements"][0]
    assert (row["cases"], row["successful_cases"], row["reached_target"], row["failed_case_roles"]) == (2, 1, True, ["case_01"])
    assert row["makespan_seconds"] == 600.0, "the skipped candidate adds no time; the failed one counts"


def test_exhausted_variant_attempts_fail_the_sweep(bench: _Bench) -> None:
    """Two started but interrupted attempts exhaust the variant; an attempt that never started does not count."""
    first = bench.submit(0)
    bench.settle(first)
    state = benchmark._sweep_state(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], frozenset())
    assert state["state"] == "ready_to_submit", "a node job that started nothing consumes no attempt"
    for record in (bench.submit(0, attempts={1: 2, 2: 2, 4: 2}), bench.submit(0, attempts={1: 3, 2: 3, 4: 3})):
        _write_receipt(bench.node_directory(record), record["tickets"][0]["seq"], started=_ORIGIN, ended=None)
        bench.settle(record)
    state = benchmark._sweep_state(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], frozenset())
    assert (state["state"], state["failed_variant_id"]) == ("variant_failed", "cores_01")


def test_evidence_from_another_job_is_refused(bench: _Bench) -> None:
    """A unit's evidence must come from the node job that held its ticket."""
    record = bench.submit(0)
    bench.finish_variant(record, 1)
    record["job_id"] = "9999"
    bench.manifest["measured_job_ids"] = ["9999"]
    with pytest.raises(ValueError, match="not its node job"):
        _variant(bench, 0, 1, frozenset())


def test_advance_submits_due_sweeps_and_derives_the_benchmark_state(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both sweeps get a node job once; a fully measured run is complete."""
    queue: dict[str, str] = {}
    monkeypatch.setattr(
        benchmark,
        "_scheduler_evidence",
        lambda job_ids: {
            "squeue": {"output": "\n".join(f"{job}|RUNNING|None|hpc117|1:00" for job in job_ids if job in queue), "error": None},
            "sacct": {"output": "", "error": None},
        },
    )
    manifest = benchmark._advance_sweeps(bench.manifest, bench.suite, storage=bench.storage)
    assert manifest["state"] == "running"
    assert [record["seat"] for record in manifest["node_jobs"]] == [1, 2]
    queue.update({record["job_id"]: "" for record in manifest["node_jobs"]})
    assert benchmark._advance_sweeps(manifest, bench.suite, storage=bench.storage)["state"] == "running"
    assert len(manifest["node_jobs"]) == 2
    for record in manifest["node_jobs"]:
        for cores in (1, 2, 4):
            bench.finish_variant(record, cores)
    queue.clear()
    manifest = benchmark._advance_sweeps(manifest, bench.suite, storage=bench.storage)
    assert all(record.get("terminal_observed_at") for record in manifest["node_jobs"])
    assert manifest["state"] == "running", "receipts may still be in flight right after the node jobs left the queue"
    for record in manifest["node_jobs"]:
        bench.settle(record)
    manifest = benchmark._advance_sweeps(manifest, bench.suite, storage=bench.storage)
    assert manifest["state"] == "complete"
    assert len(manifest["node_jobs"]) == 2
    measurements = benchmark._measurements(manifest, bench.suite, bench.directory, frozenset())
    assert [(item["sweep_id"], item["cores_per_case"]) for item in measurements] == [
        ("sweep_1", 1),
        ("sweep_1", 2),
        ("sweep_1", 4),
        ("sweep_2", 4),
        ("sweep_2", 2),
        ("sweep_2", 1),
    ]


# -- runner binding --


def test_runner_requires_a_step_of_its_node_job(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """A work unit runs only as a step (SLURM_STEP_ID) of the node job that holds its ticket."""
    record = bench.submit(0)
    monkeypatch.setattr(benchmark, "load_core_benchmark_manifest", lambda *_a, **_k: (bench.manifest, bench.suite))
    monkeypatch.setattr(benchmark, "_require_current_checkout", lambda _manifest: None)
    monkeypatch.setattr(benchmark.workspace_service, "resolve_storage_root", lambda *_a, **_k: bench.storage)
    monkeypatch.setenv("SLURM_JOB_ID", str(record["job_id"]))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM_STEP_ID"):
        benchmark.run_core_benchmark_case(_RUN_ID, "sweep_1", "cores_01", 1, "case_01", storage_root=bench.storage, work_root=bench.storage)
    monkeypatch.setenv("SLURM_STEP_ID", "3")
    with pytest.raises(RuntimeError, match="GENERATION_NODE_TOKEN"):
        benchmark.run_core_benchmark_case(_RUN_ID, "sweep_1", "cores_01", 1, "case_01", storage_root=bench.storage, work_root=bench.storage)
    monkeypatch.setenv("GENERATION_NODE_TOKEN", record["token"])
    monkeypatch.setattr(benchmark, "_BINDING_WAIT_SECONDS", 0)
    with pytest.raises(RuntimeError, match="does not hold exactly one ticket"):
        benchmark.run_core_benchmark_case(_RUN_ID, "sweep_2", "cores_01", 1, "case_01", storage_root=bench.storage, work_root=bench.storage)


def test_dispatch_binds_by_token_before_the_driver_acknowledged_the_job(bench: _Bench) -> None:
    """A step may start before the driver persisted its job ID; the ticket's token binds it to the submitting record."""
    record = bench.submit(0)
    unit_id = record["tickets"][0]["work_unit_id"]
    record.update(status="submitting", job_id=None)
    with pytest.raises(RuntimeError, match="exactly one ticket"):
        benchmark._dispatched_ticket(bench.manifest, job_id="5001", unit_id=unit_id)
    bound, ticket = benchmark._dispatched_ticket(bench.manifest, job_id="5001", unit_id=unit_id, token=record["token"])
    assert (bound["token"], ticket["work_unit_id"]) == (record["token"], unit_id)


@pytest.mark.parametrize(
    ("record", "finished", "step", "known", "expected"),
    [
        ({"status": "failed", "error": {"type": "CaseInterruptedError"}}, None, None, True, "pending"),
        ({"status": "failed", "error": {"type": "SolverError"}}, None, None, True, "failed"),
        ({"status": "pending"}, {"exit_code": 1, "outcome": "exited"}, None, True, "pending"),
        ({"status": "pending"}, {"exit_code": 143, "outcome": "exited"}, {"state": "CANCELLED by 1"}, True, "pending"),
        ({"status": "pending"}, {"exit_code": 137, "outcome": "exited"}, {"state": "OUT_OF_MEMORY"}, True, "failed_without_evidence"),
        ({"status": "pending"}, {"exit_code": -1, "outcome": "launch_failed"}, None, False, "pending"),
    ],
)
def test_interruptions_and_launch_failures_are_not_case_failures(
    record: dict[str, Any], finished: dict[str, Any] | None, step: dict[str, Any] | None, known: bool, expected: str
) -> None:
    """Cancelled, never-started, or launch-failed steps keep the unit pending; a step that ran and failed is a case failure."""
    assert benchmark._unit_outcome(record, finished, step, step_rows_known=known) == expected


def test_unresolved_submission_waits_for_its_grace_and_blocks_ownership_queries(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh intent without queue or worker evidence stays unresolved, and job ownership fails closed meanwhile."""
    record = bench.submit(0)
    record.update(status="submitting", job_id=None, submitted_at=None, recorded_at=datetime.now(UTC).isoformat())
    bench.manifest["measured_job_ids"] = []
    bench.manifest["node_job_intent"] = record["token"]
    monkeypatch.setattr(benchmark.cluster_service, "query_seat_jobs", lambda _seats: {})
    benchmark._recover_node_job_intent(bench.manifest, directory=bench.directory, manifest_path=bench.manifest_path)
    assert (record["status"], bench.manifest["node_job_intent"]) == ("submitting", record["token"])
    bench.manifest.update(schema_kind=benchmark.BENCHMARK_RUN_SCHEMA_KIND, schema_version=benchmark.BENCHMARK_SCHEMA_VERSION)
    bench.manifest_path.write_text(json.dumps(bench.manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unresolved node-job submission"):
        benchmark.load_core_benchmark_scheduler_job_ids(_RUN_ID, storage_root=bench.storage)


# -- report, summary binding --


def _measurement(suite: benchmark_config.CoreBenchmarkSuite, sweep_id: str, cores: int, case_seconds: float, *, failed: int = 0) -> dict[str, Any]:
    """Return one synthetic measured sweep variant whose cases run in rounds of the variant's concurrency."""
    variant = next(item for item in suite.variants if item.cores_per_case == cores)
    concurrency = suite.resources(variant).concurrent_cases
    units = []
    for index in range(suite.representative_case_count):
        started = _ORIGIN + timedelta(seconds=case_seconds * (index // concurrency))
        outcome = "failed" if index < failed else "success"
        units.append(
            {
                "work_unit_id": f"{sweep_id}-{cores}-{index}",
                "case_role": f"case_0{index + 1}",
                "outcome": outcome,
                "started_at": started.isoformat(),
                "ended_at": (started + timedelta(seconds=case_seconds)).isoformat(),
                "step_exit_code": 0,
                "record": {
                    "timings_seconds": {"comsol_process_seconds": case_seconds - 10.0},
                    "resource": {"peak_memory_bytes": 2**30, "node": "hpc117", "cpu_model": "Xeon", "slurm_job_id": "1", "slurm_step_id": str(index)},
                    "error": {"type": "SolverError"} if outcome == "failed" else None,
                },
            }
        )
    return {
        "sweep_id": sweep_id,
        "order": "ascending",
        "variant_id": variant.variant_id,
        "cores_per_case": cores,
        "variant_attempt": 1,
        "node_job_id": "1",
        "node": "hpc117",
        "cpu_model": "Xeon",
        "foreign_jobs_at_start": [],
        "foreign_jobs_at_end": [],
        "units": units,
    }


def test_report_measures_node_throughput_and_recommends_the_fastest_feasible_k(generation_config_factory: Any) -> None:
    """Throughput is successful cases per node-hour over the group makespan; a variant below the target is not recommendable."""
    suite = _suite(generation_config_factory)
    seconds = {1: 1200.0, 2: 500.0, 4: 300.0}
    measurements = [
        _measurement(suite, sweep, cores, seconds[cores], failed=1 if (sweep, cores) == ("sweep_2", 4) else 0)
        for sweep in ("sweep_1", "sweep_2")
        for cores in (1, 2, 4)
    ]
    summary = benchmark_report.summarize_core_benchmark_results(suite, measurements)
    by_key = {(row["sweep_id"], row["cores_per_case"]): row for row in summary["measurements"]}
    assert by_key[("sweep_1", 1)]["makespan_seconds"] == 1200.0
    assert by_key[("sweep_1", 1)]["throughput_cases_per_node_hour"] == pytest.approx(9.0)
    assert by_key[("sweep_1", 2)]["throughput_cases_per_node_hour"] == pytest.approx(3 * 3600 / 1000)
    assert by_key[("sweep_1", 4)]["throughput_cases_per_node_hour"] == pytest.approx(3 * 3600 / 900)
    assert by_key[("sweep_2", 4)]["failed_case_roles"] == ["case_01"]
    variants = {item["cores_per_case"]: item for item in summary["variants"]}
    assert variants[4]["recommendable"] is False
    assert (summary["recommended_cores_per_case"], summary["complete"]) == (2, True)
    assert by_key[("sweep_1", 1)]["node_peak_memory_upper_bound_bytes"] == 4 * 2**30
    assert benchmark_report.results_csv(measurements).count("\n") == 1 + 6 * 3
    markdown = benchmark_report.core_benchmark_markdown(summary)
    assert markdown == benchmark_report.core_benchmark_markdown(benchmark_report.summarize_core_benchmark_results(suite, measurements))
    assert "Recommended cores_per_case: 2" in markdown


def test_report_treats_a_case_failing_everywhere_as_shared(generation_config_factory: Any) -> None:
    """A case that fails in every measurement is shared; with the target reached, its node time only lowers throughput."""
    suite = _suite(generation_config_factory, target_successful_cases=2)
    measurements = [_measurement(suite, sweep, cores, 400.0, failed=1) for sweep in ("sweep_1", "sweep_2") for cores in (1, 2, 4)]
    summary = benchmark_report.summarize_core_benchmark_results(suite, measurements)
    assert all(item["recommendable"] for item in summary["variants"])
    assert all(item["variant_specific_failed_case_roles"] == [] for item in summary["variants"])
    assert summary["recommended_cores_per_case"] == 1


def test_report_recommends_nothing_when_no_case_succeeded(generation_config_factory: Any) -> None:
    """A variant that never reached its success target is never recommended."""
    suite = _suite(generation_config_factory)
    measurements = [_measurement(suite, sweep, cores, 400.0, failed=3) for sweep in ("sweep_1", "sweep_2") for cores in (1, 2, 4)]
    summary = benchmark_report.summarize_core_benchmark_results(suite, measurements)
    assert summary["recommended_cores_per_case"] is None


def test_launch_claims_without_execution_do_not_consume_attempts(bench: _Bench) -> None:
    """With step accounting, an attempt counts only when a scheduler step or case evidence shows it executed."""
    for attempt in (1, 2):
        record = bench.submit(0, attempts={1: attempt, 2: attempt, 4: attempt})
        _write_receipt(bench.node_directory(record), record["tickets"][0]["seq"], started=_ORIGIN, ended=_ORIGIN, exit_code=1)
        bench.settle(record)
    state = benchmark._sweep_state(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], frozenset(), {})
    assert state["state"] == "ready_to_submit"
    executed = {(str(record["job_id"]), str(record["tickets"][0]["label"])): {"state": "FAILED"} for record in bench.manifest["node_jobs"]}
    state = benchmark._sweep_state(bench.manifest, bench.suite, bench.directory, bench.suite.sweeps[0], frozenset(), executed)
    assert (state["state"], state["failed_variant_id"]) == ("variant_failed", "cores_01")


def test_terminal_step_accounting_is_captured_once_and_then_read_from_the_snapshot(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finalization captures sacct steps once; later validation reads the snapshot even when accounting changed."""
    record = bench.submit(0)
    label = record["tickets"][0]["label"]
    output = {"value": f"{record['job_id']}.0|{label}|COMPLETED|0:0|s|s|e|1|hpc117|4|standard|2G"}
    monkeypatch.setattr(
        benchmark, "_scheduler_evidence", lambda _ids: {"squeue": {"output": "", "error": None}, "sacct": {"output": output["value"], "error": None}}
    )
    first = benchmark._terminal_step_rows(bench.directory, bench.manifest, capture=True)
    assert first[(str(record["job_id"]), label)]["max_rss_bytes"] == 2 * 1024**3
    output["value"] = ""
    assert benchmark._terminal_step_rows(bench.directory, bench.manifest, capture=True) == first
    assert benchmark._terminal_step_rows(bench.directory, bench.manifest, capture=False) == first


def test_report_keeps_unknown_makespan_unknown(generation_config_factory: Any) -> None:
    """A unit without a finished receipt leaves makespan and throughput unavailable instead of guessing."""
    suite = _suite(generation_config_factory)
    measurement = _measurement(suite, "sweep_1", 2, 400.0)
    measurement["units"][0]["ended_at"] = None
    summary = benchmark_report.summarize_core_benchmark_results(suite, [measurement])
    row = summary["measurements"][0]
    assert (row["makespan_seconds"], row["throughput_cases_per_node_hour"], summary["complete"]) == (None, None, False)
    assert summary["recommended_cores_per_case"] is None


def test_summary_identity_binds_the_measurements(generation_config_factory: Any) -> None:
    """A summary is refused when the terminal measurements changed after it was written."""
    suite = _suite(generation_config_factory)
    measurements = [_measurement(suite, "sweep_1", 2, 400.0)]
    proofs = {
        representative.case_role: {"case_input_id": "1" * 64, "simulation_case_id": "2" * 64, "case_payload_sha256": "3" * 64}
        for representative in suite.representative_cases
    }
    manifest = {"git_commit": _COMMIT, "source_fingerprint_sha256": _FINGERPRINT, "preflight": {"receipt_sha256": "4" * 64}}
    summary = {
        **benchmark_report.summarize_core_benchmark_results(suite, measurements),
        "benchmark_run_id": _RUN_ID,
        "git_commit": _COMMIT,
        "source_fingerprint_sha256": _FINGERPRINT,
        "template_sha256": suite.case_config.template_sha256,
        "representative_case_identities": benchmark._proof_identities(suite, proofs),
        "preflight": manifest["preflight"],
        "result_set_digest": common.serialization.canonical_json_sha256(measurements),
        "generated_at": _ORIGIN.isoformat(),
    }
    benchmark._validate_summary_identity(summary, run_id=_RUN_ID, suite=suite, manifest=manifest, proofs=proofs, measurements=measurements)
    measurements[0]["units"][0]["outcome"] = "failed"
    with pytest.raises(ValueError, match="not bound"):
        benchmark._validate_summary_identity(summary, run_id=_RUN_ID, suite=suite, manifest=manifest, proofs=proofs, measurements=measurements)


# -- cancel and status --


def test_cancel_marks_active_node_jobs_and_signals_only_their_batch_shells(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graceful cancel writes control/cancel, cancels pending node jobs, and sends TERM to running batch shells."""
    running = bench.submit(0)
    pending = bench.submit(1)
    commands: list[list[str]] = []
    monkeypatch.setattr(benchmark, "load_core_benchmark_manifest", lambda *_a, **_k: (bench.manifest, bench.suite))
    monkeypatch.setattr(benchmark.workspace_service, "resolve_storage_root", lambda *_a, **_k: bench.storage)
    monkeypatch.setattr(
        benchmark,
        "_scheduler_evidence",
        lambda _ids: {"squeue": {"output": f"{running['job_id']}|RUNNING|None\n{pending['job_id']}|PENDING|Priority", "error": None}},
    )
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda command, **_k: (commands.append(command), benchmark.subprocess.CompletedProcess(command, 0, "", ""))[1]
    )
    benchmark.cancel_core_benchmark(_RUN_ID, storage_root=bench.storage)
    assert commands == [["scancel", pending["job_id"]], ["scancel", "--signal=TERM", "--batch", running["job_id"]]]
    assert all((bench.node_directory(record) / "control" / "cancel").is_file() for record in (running, pending))
    assert bench.manifest["state"] == "cancel_requested"


def test_status_and_formatter_report_sweeps(bench: _Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    """Status lists each sweep's progress and the formatter shows it without scheduler access."""
    record = bench.submit(0)
    bench.finish_variant(record, 1)
    monkeypatch.setattr(benchmark, "load_core_benchmark_manifest", lambda *_a, **_k: (bench.manifest, bench.suite))
    monkeypatch.setattr(benchmark_report, "_production_interpretation", lambda _suite: {"current_production_cores_per_case": 2})
    status = benchmark.core_benchmark_status(_RUN_ID, storage_root=bench.storage, query_scheduler=False)
    sweeps = {item["sweep_id"]: item for item in status["sweeps"]}
    assert (sweeps["sweep_1"]["state"], sweeps["sweep_1"]["measured_variants"], sweeps["sweep_1"]["current_cores_per_case"]) == ("running", 1, 2)
    assert sweeps["sweep_2"]["state"] == "ready_to_submit"
    assert status["state"] == "running"
    assert status["successful_work_units"] == 3
    assert status["partial_evaluation"]["measurements"][0]["cores_per_case"] == 1
    text = status_service.format_benchmark_status_summary(status)
    assert "sweep_1 (ascending): state=running variants=1/3" in text


def test_benchmark_scheduler_footprint_reports_sweep_node_jobs(generation_config_factory: Any) -> None:
    """The dry-run footprint of a benchmark comes from its suite: one node job per sweep plus the driver."""
    suite = _suite(generation_config_factory)
    footprint = generation.controller.benchmark_scheduler_footprint(suite.resource_contract(), driver_time="14:00:00")
    assert footprint == {
        "node_jobs": 2,
        "cores_per_node_job": 4,
        "node_job_memory": "5G",
        "node_job_wall_time": "02:00:00",
        "driver_cpus": 2,
        "driver_time": "14:00:00",
    }


def test_attempt_chain_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    """A gap or a rewritten predecessor breaks the attempt chain."""
    first = tmp_path / "attempt-0001.json"
    first.write_text(json.dumps({"attempt": 1, "previous_attempt": None}), encoding="utf-8")
    second = tmp_path / "attempt-0002.json"
    second.write_text(
        json.dumps({"attempt": 2, "previous_attempt": {"attempt": 1, "receipt_sha256": common.serialization.file_sha256(first)}}),
        encoding="utf-8",
    )
    benchmark._validate_benchmark_attempt_chain((first, second))
    first.write_text(json.dumps({"attempt": 1, "previous_attempt": None, "changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="predecessor chain"):
        benchmark._validate_benchmark_attempt_chain((first, second))


@pytest.mark.parametrize(
    ("value", "status", "started_at"),
    [
        (None, "unavailable", None),
        ("1787097600", "available", "2026-08-19T00:00:00+00:00"),
        ("not-a-timestamp", "invalid", None),
    ],
)
def test_slurm_start_time_environment_is_optional_bounded_evidence(
    monkeypatch: pytest.MonkeyPatch, value: str | None, status: str, started_at: str | None
) -> None:
    """SLURM_JOB_START_TIME is recorded when valid and never trusted when malformed."""
    if value is None:
        monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_START_TIME", value)
    evidence = benchmark._slurm_scheduler_start_time_evidence()
    assert (evidence["status"], evidence["started_at"]) == (status, started_at)


def test_unit_success_needs_evidence_and_a_zero_exit_like_the_worker_target() -> None:
    """Evidence then a failed exit is a failure the worker replaced; a zero exit without evidence fails after the grace."""
    now = datetime.now(UTC)

    def receipt(exit_code: int, seconds_ago: float) -> dict[str, Any]:
        return {"outcome": "exited", "exit_code": exit_code, "ended_at": (now - timedelta(seconds=seconds_ago)).isoformat()}

    success = {"status": "success", "work_unit_id": "u"}
    pending = {"status": "pending", "work_unit_id": "u"}
    assert benchmark._unit_outcome(success, receipt(0, 5), None, step_rows_known=True) == "success"
    assert benchmark._unit_outcome(success, receipt(1, 5), None, step_rows_known=True) == "failed"
    assert benchmark._unit_outcome(pending, receipt(0, 10), None, step_rows_known=True) == "pending"
    assert benchmark._unit_outcome(pending, receipt(0, 600), None, step_rows_known=True) == "failed_without_evidence"
