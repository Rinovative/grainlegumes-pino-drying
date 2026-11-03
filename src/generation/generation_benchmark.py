"""Own the isolated transient COMSOL core-scaling benchmark lifecycle on node-job sweeps.

Responsibilities:
  - Plan, submit, resume, and execute sweep node jobs whose steps measure the benchmark work units
  - Validate and persist immutable measurement evidence bound to the frozen source snapshot
  - Finalize and validate benchmark artifacts

Design principles:
  - Slurm-materialized proofs bind every variant and sweep to the same exact inputs
  - The run owns deterministic raw copies of the representative production cases (inputs/<case_role>) and never reserves
    production candidates, freezes a campaign plan, or writes canonical input storage
  - Scientific identity, resource work-unit identity, and the allocation (node job) stay separate
  - A variant attempt is measured inside exactly one node job; an interrupted attempt is rerun as a new attempt
  - Successful work-unit evidence is immutable and failed attempts are append-only

This module does NOT:
  - Resolve benchmark suite configuration or interpret measurements
  - Implement the node worker or publish training cases

Notes:
  Each sweep is one node job in one node-job slot (job name gen-seat-<seat>, Slurm singleton) running the
  generic node worker; the benchmark writes all tickets of the sweep before submission, one ticket group per variant in
  the sweep order, and a stop ticket. Node-job identity is recovered by the submission token in the job comment.

"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource as resource_usage
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np
import yaml

from src import common
from src.generation.cases import generation_cases_case as case_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_node as node_service
from src.generation.runtime import generation_runtime_preparation as preparation_service
from src.generation.runtime import generation_runtime_progress as progress_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

from . import generation_benchmark_config as benchmark_config
from . import generation_benchmark_report as benchmark_report

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BENCHMARK_RUN_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_run"
BENCHMARK_PROOF_SCHEMA_KIND: Final = "generation_core_scaling_case_proof"
BENCHMARK_RESULT_SCHEMA_KIND: Final = "generation_core_scaling_result"
BENCHMARK_SUMMARY_SCHEMA_KIND: Final = benchmark_report.BENCHMARK_SUMMARY_SCHEMA_KIND
BENCHMARK_PREFLIGHT_SCHEMA_KIND: Final = "generation_core_scaling_preflight"
BENCHMARK_CANCELLATION_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_cancellations"
BENCHMARK_SCHEMA_VERSION: Final = benchmark_config.BENCHMARK_SCHEMA_VERSION
BENCHMARK_FAMILY: Final = "core_scaling"
_MAX_COMSOL_VERSION_EVIDENCE_BYTES: Final = 16 * 1024
_MAX_SLURM_ENV_VALUE_CHARACTERS: Final = 128
_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_BENCHMARK_RUN_ID_PATTERN: Final = re.compile(r"core_scaling_transient__[0-9a-f]{16}")


_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]{16}")

_CASE_PUBLICATION_MARGIN_SECONDS: Final = 600

_STEP_LAUNCH_WAIT_SECONDS: Final = 120

_NOT_STARTED_GRACE_SECONDS: Final = 120

_CASE_FAILURE_STEP_STATES: Final = frozenset({"FAILED", "OUT_OF_MEMORY", "TIMEOUT", "COMPLETED"})  # any other step state is an interruption
_STEP_ACCOUNTING_FILENAME: Final = "step_accounting.json"
_PLAN_TOKEN: Final = "0" * 16  # stands for the per-submission token in the deterministic plan digest
_BINDING_WAIT_SECONDS: Final = 180  # NFS visibility of the driver's manifest on the step's node (probe P1: 52 s)
_INTERRUPTION_ERRORS: Final = frozenset({"CaseInterruptedError", "KeyboardInterrupt", "SystemExit"})
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _slurm_scheduler_start_time_evidence() -> dict[str, str | None]:
    """Return bounded optional worker environment corroboration."""
    raw = os.environ.get("SLURM_JOB_START_TIME")
    bounded_raw = None if raw is None else raw[:_MAX_SLURM_ENV_VALUE_CHARACTERS]
    if raw is None:
        return {
            "status": "unavailable",
            "raw_value": None,
            "started_at": None,
            "warning": "environment variable is absent",
        }
    if _JOB_ID_PATTERN.fullmatch(raw) is None:
        return {
            "status": "invalid",
            "raw_value": bounded_raw,
            "started_at": None,
            "warning": "value is not a positive integer timestamp",
        }
    try:
        timestamp = int(raw)
    except ValueError:
        return {
            "status": "invalid",
            "raw_value": bounded_raw,
            "started_at": None,
            "warning": "value cannot be converted to a supported timestamp",
        }
    if timestamp < 1:
        return {
            "status": "invalid",
            "raw_value": bounded_raw,
            "started_at": None,
            "warning": "value is not positive",
        }
    try:
        started = datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return {
            "status": "invalid",
            "raw_value": bounded_raw,
            "started_at": None,
            "warning": "value is outside the supported timestamp range",
        }
    return {
        "status": "available",
        "raw_value": bounded_raw,
        "started_at": started.isoformat(),
        "warning": None,
    }


def _executing_source() -> tuple[str, str]:
    """Return the commit and source fingerprint this process executes: the admitted frozen tree, or the live checkout."""
    if common.paths.source_tree_id() is not None:
        tree = source_service.verify_admitted_source_tree()
        return tree.git_commit, tree.source_fingerprint_sha256
    repository = common.paths.get_project_root()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- site PATH owns Git
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fingerprint = subprocess.run(  # noqa: S603 -- fixed source fingerprint invocation
        [sys.executable, str(repository / "scripts" / "source_fingerprint.py"), str(repository)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        message = f"The live source fingerprint is malformed: {fingerprint!r}."
        raise RuntimeError(message)
    return source_service.validate_git_commit(commit), fingerprint


def resolve_core_benchmark_runtime_identity(
    path: Path | str,
    *,
    git_commit: str,
    comsol_version_output: str,
) -> dict[str, Any]:
    """Resolve the deterministic runtime identity without persistent mutation."""
    suite = benchmark_config.load_core_benchmark_suite(path, require_executable=True)
    commit, fingerprint = _executing_source()
    if commit != source_service.validate_git_commit(git_commit):
        message = f"Benchmark source commit {commit} does not match the requested commit {git_commit}."
        raise RuntimeError(message)
    version = _comsol_version_evidence(
        comsol_version_output,
        configured_executable=suite.resource_contract()["comsol_executable"],
    )
    run_id = core_benchmark_run_id(
        suite,
        git_commit=commit,
        source_fingerprint=fingerprint,
        comsol_version=version,
    )
    return {
        "schema_kind": "generation_core_scaling_benchmark_runtime_identity",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_run_id": run_id,
        "suite_digest": suite.suite_digest,
        "git_commit": commit,
        "source_fingerprint_sha256": fingerprint,
        "comsol_version": version,
    }


def _require_current_checkout(manifest: Mapping[str, Any]) -> None:
    """Require the executing source (commit and fingerprint) to equal the persisted benchmark source."""
    expected = (source_service.validate_git_commit(manifest.get("git_commit")), manifest.get("source_fingerprint_sha256"))
    actual = _executing_source()
    if actual != expected:
        message = f"Benchmark source {actual} does not match the persisted source {expected}."
        raise RuntimeError(message)


def _comsol_version_evidence(
    output: str,
    *,
    configured_executable: str,
) -> dict[str, str]:
    """Return normalized COMSOL version evidence for benchmark identity."""
    if not isinstance(output, str):
        message = "Benchmark COMSOL version evidence must be text."
        raise TypeError(message)
    normalized = output.strip()
    if not normalized or len(normalized.encode("utf-8")) > _MAX_COMSOL_VERSION_EVIDENCE_BYTES or "\x00" in normalized:
        message = "Benchmark COMSOL version evidence is empty or unsafe."
        raise ValueError(message)
    return {
        "configured_executable": configured_executable,
        "output": normalized,
        "digest": common.serialization.canonical_json_sha256(
            {
                "configured_executable": configured_executable,
                "output": normalized,
            }
        ),
    }


def _benchmark_identity(
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    git_commit: str,
    source_fingerprint: str,
    comsol_version: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the actual dependency-scoped standalone benchmark identity, bound to the frozen source snapshot."""
    commit = source_service.validate_git_commit(git_commit)
    if _SHA256_PATTERN.fullmatch(str(source_fingerprint)) is None:
        message = "Benchmark source fingerprint is malformed."
        raise ValueError(message)
    version = benchmark_config.require_mapping(comsol_version, label="benchmark COMSOL version evidence")
    if (
        set(version) != {"configured_executable", "output", "digest"}
        or version.get("configured_executable") != suite.resource_contract()["comsol_executable"]
        or _SHA256_PATTERN.fullmatch(str(version.get("digest"))) is None
        or _comsol_version_evidence(
            str(version.get("output")),
            configured_executable=str(version.get("configured_executable")),
        )
        != version
    ):
        message = "Benchmark COMSOL version evidence is malformed or inconsistent."
        raise ValueError(message)
    cases = suite.case_selections()
    return {
        "schema_kind": BENCHMARK_RUN_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": commit,
        "source_fingerprint_sha256": source_fingerprint,
        "suite_digest": suite.suite_digest,
        "case_selection_digests": [case["selection_digest"] for case in cases],
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "case_input_config_digest": suite.case_config.case_input_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "export_contract_sha256": cases[0]["export_contract_sha256"],
        "execution_config_digest": cases[0]["execution_config_digest"],
        "sweeps": _sweep_records(suite),
        "representative_case_count": suite.representative_case_count,
        "variants": _variant_records(suite),
        "comsol_version": version,
    }


def core_benchmark_run_id(
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    git_commit: str,
    source_fingerprint: str,
    comsol_version: Mapping[str, Any],
) -> str:
    """Return the standalone run identity from actual benchmark dependencies."""
    digest = common.serialization.canonical_json_sha256(
        _benchmark_identity(
            suite,
            git_commit=git_commit,
            source_fingerprint=source_fingerprint,
            comsol_version=comsol_version,
        )
    )
    return f"core_scaling_transient__{digest[:16]}"


def core_benchmark_directory(
    run_id: str,
    *,
    storage_root: Path | str,
) -> Path:
    """Return one dedicated benchmark evidence directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="benchmark_run_id")
    if _BENCHMARK_RUN_ID_PATTERN.fullmatch(safe_id) is None:
        message = f"Malformed core benchmark run ID: {run_id!r}."
        raise ValueError(message)
    return (
        common.paths.get_generation_performance_benchmark_root(
            storage_root=storage_root,
        )
        / BENCHMARK_FAMILY
        / safe_id
    )


def _preflight_path(
    run_id: str,
    *,
    storage_root: Path | str,
) -> Path:
    """Return the immutable standalone preflight receipt path."""
    return core_benchmark_directory(run_id, storage_root=storage_root) / "benchmark_preflight.json"


def _probe_directory_capability(path: Path | str, *, label: str) -> dict[str, Any]:
    """Prove one existing directory supports small same-filesystem writes."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        message = f"{label} must be an absolute directory: {candidate}"
        raise ValueError(message)
    resolved = candidate.resolve()
    if not resolved.is_dir() or not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
        message = f"{label} is not an available readable/writable directory: {resolved}"
        raise FileNotFoundError(message)
    descriptor, probe_name = tempfile.mkstemp(prefix=".benchmark-preflight.", dir=resolved)
    probe = Path(probe_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"benchmark-preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not probe.is_file() or probe.stat().st_size < 1:
            message = f"{label} capability probe did not persist an ordinary file: {resolved}"
            raise RuntimeError(message)
    finally:
        probe.unlink(missing_ok=True)
    usage = shutil.disk_usage(resolved)
    if usage.free < 1:
        message = f"{label} reports no free storage capacity: {resolved}"
        raise RuntimeError(message)
    return {
        "path": str(resolved),
        "free_bytes_observed": usage.free,
        "small_write_probe": "pass",
    }


def _validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    suite: benchmark_config.CoreBenchmarkSuite,
    git_commit: str,
) -> None:
    """Validate immutable standalone preflight identity and owned checks."""
    expected_keys = {
        "schema_kind",
        "schema_version",
        "status",
        "recorded_at",
        "benchmark_preflight_seconds",
        "benchmark_run_id",
        "suite_config",
        "benchmark_identity",
        "comsol_runtime",
        "python_runtime",
        "storage_capabilities",
        "sweep_plan_digest",
        "checks",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_kind") != BENCHMARK_PREFLIGHT_SCHEMA_KIND
        or payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or payload.get("status") != "pass"
        or payload.get("benchmark_run_id") != run_id
        or payload.get("suite_config") != benchmark_config.repository_relative(suite.source_path)
    ):
        message = f"Standalone benchmark preflight receipt is malformed: {run_id}"
        raise ValueError(message)
    identity = benchmark_config.require_mapping(
        payload.get("benchmark_identity"),
        label="benchmark preflight identity",
    )
    fingerprint = str(identity.get("source_fingerprint_sha256"))
    comsol_version = benchmark_config.require_mapping(identity.get("comsol_version"), label="benchmark preflight COMSOL version")
    expected_identity = _benchmark_identity(suite, git_commit=git_commit, source_fingerprint=fingerprint, comsol_version=comsol_version)
    if (
        identity != expected_identity
        or core_benchmark_run_id(suite, git_commit=git_commit, source_fingerprint=fingerprint, comsol_version=comsol_version) != run_id
    ):
        message = f"Standalone benchmark preflight identity conflicts: {run_id}"
        raise ValueError(message)
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or set(checks.values()) != {"pass"}:
        message = f"Standalone benchmark preflight checks are incomplete: {run_id}"
        raise ValueError(message)
    capabilities = payload.get("storage_capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != {
        "scratch",
        "persistent",
    }:
        message = f"Standalone benchmark storage capability evidence is malformed: {run_id}"
        raise ValueError(message)
    runtime = payload.get("comsol_runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"resolved_executable", "version"}
        or not Path(str(runtime.get("resolved_executable"))).is_absolute()
        or runtime.get("version") != identity["comsol_version"]
    ):
        message = f"Standalone benchmark COMSOL runtime evidence is malformed: {run_id}"
        raise ValueError(message)
    python_runtime = payload.get("python_runtime")
    if (
        not isinstance(python_runtime, dict)
        or set(python_runtime) != {"executable", "version", "imports"}
        or not isinstance(python_runtime.get("imports"), dict)
    ):
        message = f"Standalone benchmark Python runtime evidence is malformed: {run_id}"
        raise ValueError(message)
    if _SHA256_PATTERN.fullmatch(str(payload.get("sweep_plan_digest"))) is None:
        message = f"Standalone benchmark sweep plan evidence is malformed: {run_id}"
        raise ValueError(message)
    benchmark_report.parse_benchmark_timestamp(payload.get("recorded_at"), label="benchmark preflight recorded_at")
    preflight_seconds = payload.get("benchmark_preflight_seconds")
    if (
        isinstance(preflight_seconds, bool)
        or not isinstance(preflight_seconds, (int, float))
        or not math.isfinite(float(preflight_seconds))
        or float(preflight_seconds) < 0.0
    ):
        message = "Standalone benchmark preflight duration is malformed."
        raise ValueError(message)


def preflight_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Run inexpensive benchmark-owned source, runtime, storage, and command checks."""
    started = time.perf_counter()
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit, fingerprint = _executing_source()
    if current_commit != requested_commit:
        message = f"Benchmark source commit {current_commit} does not match requested benchmark commit {requested_commit}."
        raise RuntimeError(message)
    suite = benchmark_config.load_core_benchmark_suite(path, require_executable=True)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    persistent = _probe_directory_capability(storage, label="benchmark persistent storage")
    scratch = _probe_directory_capability(scratch_root, label="benchmark scratch")
    executable = Path(comsol_executable_path).expanduser()
    if not executable.is_absolute():
        message = "Resolved benchmark COMSOL executable must be an absolute path."
        raise ValueError(message)
    resolved_executable = executable.resolve()
    if not resolved_executable.is_file() or not os.access(resolved_executable, os.X_OK):
        message = f"Resolved benchmark COMSOL executable is unavailable: {resolved_executable}"
        raise FileNotFoundError(message)
    version = _comsol_version_evidence(
        comsol_version_output,
        configured_executable=suite.resource_contract()["comsol_executable"],
    )
    identity = _benchmark_identity(suite, git_commit=requested_commit, source_fingerprint=fingerprint, comsol_version=version)
    run_id = core_benchmark_run_id(suite, git_commit=requested_commit, source_fingerprint=fingerprint, comsol_version=version)
    directory = core_benchmark_directory(run_id, storage_root=storage)
    payload = {
        "schema_kind": BENCHMARK_PREFLIGHT_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "pass",
        "recorded_at": _utc_now(),
        "benchmark_preflight_seconds": time.perf_counter() - started,
        "benchmark_run_id": run_id,
        "suite_config": benchmark_config.repository_relative(suite.source_path),
        "benchmark_identity": identity,
        "comsol_runtime": {
            "resolved_executable": str(resolved_executable),
            "version": version,
        },
        "python_runtime": {
            "executable": sys.executable,
            "version": sys.version,
            "imports": {
                "h5py": h5py.__version__,
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "storage_capabilities": {
            "scratch": scratch,
            "persistent": persistent,
        },
        "sweep_plan_digest": common.serialization.canonical_json_sha256(_sweep_plan(suite, run_id=run_id)),
        "checks": {
            "frozen_source_snapshot": "pass",
            "suite_and_case": "pass",
            "template_and_export_contract": "pass",
            "comsol_executable_and_version": "pass",
            "python_runtime_and_imports": "pass",
            "node_job_resources": "pass",
            "scratch_storage": "pass",
            "persistent_storage": "pass",
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = _preflight_path(run_id, storage_root=storage)
    if receipt_path.exists():
        existing = _load_json(
            receipt_path,
            label="standalone benchmark preflight receipt",
        )
        _validate_preflight_payload(
            existing,
            run_id=run_id,
            suite=suite,
            git_commit=requested_commit,
        )
        return existing
    source_service.validate_admitted_source_before_publication()
    _write_immutable_json(
        receipt_path,
        payload,
        label="standalone benchmark preflight receipt",
    )
    _validate_preflight_payload(
        payload,
        run_id=run_id,
        suite=suite,
        git_commit=requested_commit,
    )
    return payload


def _load_core_benchmark_preflight(
    run_id: str,
    *,
    suite: benchmark_config.CoreBenchmarkSuite,
    git_commit: str,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate the immutable standalone preflight receipt."""
    payload = _load_json(
        _preflight_path(run_id, storage_root=storage_root),
        label="standalone benchmark preflight receipt",
    )
    _validate_preflight_payload(
        payload,
        run_id=run_id,
        suite=suite,
        git_commit=git_commit,
    )
    return payload


def _sweep_records(suite: benchmark_config.CoreBenchmarkSuite) -> list[dict[str, Any]]:
    """Return persisted sweep identities: seat and variant order of every sweep."""
    return [
        {
            "sweep_id": sweep.sweep_id,
            "order": sweep.order,
            "seat": seat,
            "variant_order": [variant.variant_id for variant in suite.sweep_variants(sweep)],
        }
        for seat, sweep in enumerate(suite.sweeps, start=1)
    ]


def _format_seconds(seconds: int) -> str:
    """Return one Slurm ``HH:MM:SS`` time."""
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _unit_ticket(
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    run_id: str,
    sweep: benchmark_config.CoreBenchmarkSweep,
    variant: benchmark_config.CoreBenchmarkVariant,
    variant_attempt: int,
    case_position: int,
    group: int,
    seq: int,
    token: str,
) -> node_service.NodeTicket:
    """Return the step ticket that measures one work unit inside its sweep node job."""
    repository = common.paths.get_project_root().resolve()
    representative = suite.representative_case(case_position)
    timeout = int(suite.case_config.execution_values["runtime"]["timeout_seconds"])
    seat = suite.sweeps.index(sweep) + 1
    return node_service.NodeTicket(
        seq=seq,
        kind=node_service.STEP_KIND,
        group=group,
        cores=variant.cores_per_case,
        memory=suite.resources(variant).memory_per_case,
        time_limit=_format_seconds(timeout + _CASE_PUBLICATION_MARGIN_SECONDS),
        required_seconds=timeout + _CASE_PUBLICATION_MARGIN_SECONDS + _STEP_LAUNCH_WAIT_SECONDS,
        label=f"bm{seat}-c{variant.cores_per_case:02d}-a{variant_attempt}-{representative.case_role}"[:48],
        argv=(
            str(repository / "scripts" / "generation_node.sh"),
            str(repository),
            "benchmark-case",
            run_id,
            sweep.sweep_id,
            variant.variant_id,
            str(variant_attempt),
            representative.case_role,
        ),
        env={"GENERATION_BENCHMARK_RUN_ID": run_id, "GENERATION_NODE_TOKEN": token},
        success_target=suite.target_successful_cases,
    )


def _sweep_tickets(
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    run_id: str,
    sweep: benchmark_config.CoreBenchmarkSweep,
    variant_attempts: Sequence[tuple[benchmark_config.CoreBenchmarkVariant, int]],
    token: str,
) -> tuple[node_service.NodeTicket, ...]:
    """Return every ticket of one sweep node job: one group per variant in sweep order, then the stop ticket."""
    tickets: list[node_service.NodeTicket] = []
    for group, (variant, attempt) in enumerate(variant_attempts):
        for case_position in range(1, suite.representative_case_count + 1):
            tickets.append(
                _unit_ticket(
                    suite,
                    run_id=run_id,
                    sweep=sweep,
                    variant=variant,
                    variant_attempt=attempt,
                    case_position=case_position,
                    group=group,
                    seq=len(tickets) + 1,
                    token=token,
                )
            )
    tickets.append(node_service.NodeTicket(seq=len(tickets) + 1, kind=node_service.STOP_KIND))
    return tuple(tickets)


def _sweep_plan(suite: benchmark_config.CoreBenchmarkSuite, *, run_id: str) -> list[dict[str, Any]]:
    """Return the deterministic first-attempt plan of every sweep: node-job resources and ticket payloads."""
    return [
        {
            "sweep_id": sweep.sweep_id,
            "seat": seat,
            "node_job": {
                "cores_per_node": suite.cores_per_node,
                "node_job_memory": suite.sweep_node_job_memory(),
                "node_job_wall_time": suite.sweep_wall_time,
                "partition": suite.partition,
            },
            "tickets": [
                ticket.to_payload()
                for ticket in _sweep_tickets(
                    suite,
                    run_id=run_id,
                    sweep=sweep,
                    variant_attempts=[(variant, 1) for variant in suite.sweep_variants(sweep)],
                    token=_PLAN_TOKEN,
                )
            ],
        }
        for seat, sweep in enumerate(suite.sweeps, start=1)
    ]


def _validated_node_job_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the node-job records and return their known job IDs in record order."""
    run_id = str(manifest.get("benchmark_run_id"))
    records = manifest.get("node_jobs")
    job_ids = manifest.get("measured_job_ids")
    if not isinstance(records, list) or not isinstance(job_ids, list):
        message = f"Benchmark node-job records are malformed: {run_id}"
        raise TypeError(message)
    known: list[str] = []
    tokens: set[str] = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or _TOKEN_PATTERN.fullmatch(str(record.get("token"))) is None
            or record.get("status") not in {"submitting", "submitted", "submission_failed", "unknown"}
            or str(record.get("token")) in tokens
        ):
            message = f"Benchmark node-job record is malformed: {run_id}"
            raise ValueError(message)
        tokens.add(str(record["token"]))
        job_id = record.get("job_id")
        if job_id is not None:
            if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None:
                message = f"Benchmark node-job record has a malformed Slurm job ID: {run_id}"
                raise ValueError(message)
            known.append(job_id)
    if known != job_ids or len(set(known)) != len(known):
        message = f"Benchmark job IDs disagree with the node-job records: {run_id}"
        raise ValueError(message)
    intent = manifest.get("node_job_intent")
    unresolved = [str(record["token"]) for record in records if record["status"] == "submitting"]
    if (intent is None and unresolved) or (intent is not None and unresolved != [intent]):
        message = f"Benchmark node-job intent is inconsistent: {run_id}"
        raise ValueError(message)
    return tuple(known)


def _variant_records(suite: benchmark_config.CoreBenchmarkSuite) -> list[dict[str, Any]]:
    """Return persisted variant identities and derived node-job resources from one resolved suite."""
    return [
        {
            "variant_id": variant.variant_id,
            "cores_per_case": variant.cores_per_case,
            "concurrent_cases": suite.resources(variant).concurrent_cases,
            "node_job_memory": suite.resources(variant).node_job_memory,
            "execution_id": suite.execution_id(variant),
        }
        for variant in suite.variants
    ]


def _plan_payload(
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    git_commit: str,
    storage: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical sweep plan: one node job per sweep, one ticket group per variant."""
    run_id = str(preflight["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    return {
        "schema_kind": "generation_core_scaling_benchmark_plan",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "state": "planned",
        "filesystem_mutated": True,
        "benchmark_run_id": run_id,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": benchmark_config.repository_relative(suite.source_path),
        "git_commit": git_commit,
        "source_fingerprint_sha256": preflight["benchmark_identity"]["source_fingerprint_sha256"],
        "preflight": {
            "receipt_sha256": common.serialization.file_sha256(_preflight_path(run_id, storage_root=storage)),
            "benchmark_identity": preflight["benchmark_identity"],
            "comsol_runtime": preflight["comsol_runtime"],
            "checks": preflight["checks"],
        },
        "representative_cases": suite.case_selections(),
        "variants": _variant_records(suite),
        "sweeps": _sweep_records(suite),
        "required_successes_per_sweep": len(suite.variants) * suite.target_successful_cases,
        "resource_contract": suite.resource_contract(),
        "sweep_plan": _sweep_plan(suite, run_id=run_id),
        "paths": {
            "storage_root": str(storage),
            "benchmark_root": str(directory),
            "scheduler_logs": str(common.paths.get_runtime_root().resolve() / "logs" / "generation"),
        },
        "canonical_input_preparation": {
            "execution_environment": "slurm_driver",
            "scheduler_job": True,
            "proof_directory": "canonical_cases",
            "case_count": suite.representative_case_count,
        },
        "isolation": {
            "node_jobs": len(suite.sweeps),
            "variant_groups_overlap": False,
            "scheduler_arrays": False,
            "queue_wait_primary_metric": False,
            "license_wait_primary_metric": False,
        },
        "dataset_membership": "none",
    }


def plan_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Run or reuse standalone preflight and return the canonical plan."""
    requested_commit = source_service.validate_git_commit(git_commit)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    suite = benchmark_config.load_core_benchmark_suite(path, require_executable=True)
    preflight = preflight_core_benchmark(
        path,
        git_commit=requested_commit,
        storage_root=storage,
        scratch_root=scratch_root,
        comsol_version_output=comsol_version_output,
        comsol_executable_path=comsol_executable_path,
    )
    return _plan_payload(
        suite,
        git_commit=requested_commit,
        storage=storage,
        preflight=preflight,
    )


def _manifest_path(run_id: str, *, storage_root: Path | str) -> Path:
    """Return one benchmark run manifest path."""
    return core_benchmark_directory(run_id, storage_root=storage_root) / "benchmark_manifest.json"


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """Write one immutable canonical JSON object or validate an exact retry."""
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.is_symlink():
            message = f"Existing {label} path is unsafe: {path}"
            raise FileExistsError(message)
        if path.read_text(encoding="utf-8") != serialized:
            message = f"Existing {label} conflicts: {path}"
            raise FileExistsError(message)
        return path
    return common.serialization.atomic_write_text(path, serialized)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required symlink-free JSON object."""
    if not path.is_file() or path.is_symlink():
        message = f"Required {label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load {label}: {path}"
        raise ValueError(message) from error
    return benchmark_config.require_mapping(value, label=label)


def load_core_benchmark_scheduler_job_ids(
    run_id: str,
    *,
    storage_root: Path | str,
) -> tuple[str, ...]:
    """Load only exact persisted scheduler ownership (the sweep node jobs) of one benchmark run."""
    manifest = _load_json(
        _manifest_path(run_id, storage_root=storage_root),
        label="core benchmark manifest",
    )
    if (
        _BENCHMARK_RUN_ID_PATTERN.fullmatch(run_id) is None
        or manifest.get("schema_kind") != BENCHMARK_RUN_SCHEMA_KIND
        or manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or manifest.get("benchmark_run_id") != run_id
    ):
        message = f"Core benchmark manifest schema or run identity is invalid: {run_id}"
        raise ValueError(message)
    if manifest.get("node_job_intent") is not None:
        message = f"Core benchmark {run_id} has an unresolved node-job submission; resume it before relying on its job ownership."
        raise RuntimeError(message)
    return _validated_node_job_ids(manifest)


def load_core_benchmark_manifest(
    run_id: str,
    *,
    storage_root: Path | str,
) -> tuple[dict[str, Any], benchmark_config.CoreBenchmarkSuite]:
    """Load one run manifest and re-resolve its exact suite identity."""
    manifest = _load_json(
        _manifest_path(run_id, storage_root=storage_root),
        label="core benchmark manifest",
    )
    benchmark_config.require_exact_keys(
        manifest,
        {
            "schema_kind",
            "schema_version",
            "benchmark_run_id",
            "suite_name",
            "suite_digest",
            "suite_config",
            "git_commit",
            "source_fingerprint_sha256",
            "preflight",
            "representative_cases",
            "variants",
            "sweeps",
            "required_successes_per_sweep",
            "resource_contract",
            "created_at",
            "state",
            "measured_job_ids",
            "node_jobs",
            "node_job_intent",
        },
        label="core benchmark manifest",
    )
    if (
        manifest["schema_kind"] != BENCHMARK_RUN_SCHEMA_KIND
        or manifest["schema_version"] != BENCHMARK_SCHEMA_VERSION
        or manifest["benchmark_run_id"] != run_id
    ):
        message = f"Core benchmark manifest schema or run identity is invalid: {run_id}"
        raise ValueError(message)
    source_service.validate_git_commit(manifest["git_commit"])
    suite_path = benchmark_config.resolve_reference_path(
        manifest["suite_config"],
        label="benchmark manifest suite_config",
    )
    suite = benchmark_config.load_core_benchmark_suite(suite_path, require_executable=True)
    expected = {
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "representative_cases": suite.case_selections(),
        "variants": _variant_records(suite),
        "sweeps": _sweep_records(suite),
        "required_successes_per_sweep": len(suite.variants) * suite.target_successful_cases,
        "resource_contract": suite.resource_contract(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        message = f"Core benchmark manifest no longer matches current suite source: {run_id}"
        raise ValueError(message)
    _validated_node_job_ids(manifest)
    preflight = _load_core_benchmark_preflight(
        run_id,
        suite=suite,
        git_commit=str(manifest["git_commit"]),
        storage_root=storage_root,
    )
    expected_preflight = {
        "receipt_sha256": common.serialization.file_sha256(_preflight_path(run_id, storage_root=storage_root)),
        "benchmark_identity": preflight["benchmark_identity"],
        "comsol_runtime": preflight["comsol_runtime"],
        "checks": preflight["checks"],
    }
    if (
        manifest.get("preflight") != expected_preflight
        or manifest["source_fingerprint_sha256"] != preflight["benchmark_identity"]["source_fingerprint_sha256"]
    ):
        message = f"Core benchmark manifest preflight identity is inconsistent: {run_id}"
        raise ValueError(message)
    return manifest, suite


def _submit(command: Sequence[str], *, git_commit: str, run_id: str) -> str:
    """Submit one typed Slurm command from the executing source tree, lease that tree for it, and return its parsable root job ID."""
    cluster_service.executing_tree_id()
    environment = os.environ.copy()
    environment["GENERATION_GIT_COMMIT"] = git_commit
    environment["GENERATION_BENCHMARK_RUN_ID"] = run_id
    result = subprocess.run(  # noqa: S603 -- typed Slurm service builds the argv
        list(command),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = f"Slurm returned an invalid benchmark job identifier: {result.stdout!r}."
        raise RuntimeError(message)
    cluster_service.lease_executing_tree(job_id)
    return job_id


def _validate_benchmark_attempt_chain(attempts: Sequence[Path]) -> None:
    """Validate contiguous benchmark attempts and immediate receipt digests."""
    stable_identity: dict[str, Any] | None = None
    previous_path: Path | None = None
    identity_keys = (
        "benchmark_run_id",
        "suite_digest",
        "sweep_id",
        "variant_attempt",
        "variant_id",
        "execution_id",
        "case_position",
        "case_role",
        "work_unit_id",
        "git_commit",
        "case_input_id",
        "simulation_case_id",
        "scientific_config_digest",
        "template_sha256",
        "benchmark_preflight_sha256",
        "cores_per_case",
    )
    for expected_index, attempt_path in enumerate(attempts, start=1):
        if attempt_path.name != f"attempt-{expected_index:04d}.json":
            message = f"Benchmark attempt history is not contiguous: {attempt_path}"
            raise ValueError(message)
        payload = _load_json(attempt_path, label="benchmark work-unit attempt")
        previous = payload.get("previous_attempt")
        if expected_index == 1:
            valid_previous = previous is None
        else:
            valid_previous = (
                isinstance(previous, dict)
                and set(previous) == {"attempt", "receipt_sha256"}
                and previous.get("attempt") == expected_index - 1
                and previous_path is not None
                and previous.get("receipt_sha256") == common.serialization.file_sha256(previous_path)
            )
        if payload.get("attempt") != expected_index or not valid_previous:
            message = f"Benchmark attempt predecessor chain is invalid: {attempt_path}"
            raise ValueError(message)
        identity = {key: payload.get(key) for key in identity_keys}
        if stable_identity is None:
            stable_identity = identity
        elif identity != stable_identity:
            message = f"Benchmark attempt identity changed across retry: {attempt_path}"
            raise ValueError(message)
        previous_path = attempt_path


def _benchmark_previous_attempt_reference(attempts: Sequence[Path]) -> dict[str, Any] | None:
    """Return the exact immediate predecessor after admitting the prior chain."""
    _validate_benchmark_attempt_chain(attempts)
    if not attempts:
        return None
    return {
        "attempt": len(attempts),
        "receipt_sha256": common.serialization.file_sha256(attempts[-1]),
    }


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    suite: benchmark_config.CoreBenchmarkSuite,
    unit: _WorkUnit,
    status: str,
) -> None:
    """Validate identity fields shared by successful and failed attempts."""
    expected = {
        "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": status,
        "suite_digest": suite.suite_digest,
        "sweep_id": unit.sweep.sweep_id,
        "variant_attempt": unit.attempt,
        "variant_id": unit.variant.variant_id,
        "execution_id": suite.execution_id(unit.variant),
        "case_position": unit.case_position,
        "case_role": suite.representative_case(unit.case_position).case_role,
        "work_unit_id": _unit_id(suite, unit),
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "cores_per_case": unit.variant.cores_per_case,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        message = f"Benchmark {status} evidence conflicts for {expected['work_unit_id']}."
        raise ValueError(message)
    if _BENCHMARK_RUN_ID_PATTERN.fullmatch(str(result.get("benchmark_run_id"))) is None:
        message = f"Benchmark result has a malformed run ID for {expected['work_unit_id']}."
        raise ValueError(message)
    source_service.validate_git_commit(result.get("git_commit"))
    if _SHA256_PATTERN.fullmatch(str(result.get("benchmark_preflight_sha256"))) is None:
        message = f"Benchmark result has malformed preflight identity for {expected['work_unit_id']}."
        raise ValueError(message)
    for key in ("case_input_id", "simulation_case_id"):
        if _SHA256_PATTERN.fullmatch(str(result.get(key))) is None:
            message = f"Benchmark result has malformed {key} for {expected['work_unit_id']}."
            raise ValueError(message)
    attempt = result.get("attempt")
    previous = result.get("previous_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        message = f"Benchmark result has an invalid attempt index for {expected['work_unit_id']}."
        raise ValueError(message)
    valid_previous = (
        previous is None
        if attempt == 1
        else isinstance(previous, dict)
        and set(previous) == {"attempt", "receipt_sha256"}
        and previous.get("attempt") == attempt - 1
        and _SHA256_PATTERN.fullmatch(str(previous.get("receipt_sha256"))) is not None
    )
    if not valid_previous:
        message = f"Benchmark result has an invalid attempt chain for {expected['work_unit_id']}."
        raise ValueError(message)


def _validate_resource_evidence(
    result: Mapping[str, Any],
    *,
    suite: benchmark_config.CoreBenchmarkSuite,
    unit: _WorkUnit,
) -> None:
    """Validate the node-job allocation, step identity, node, and observed per-case memory and scratch evidence."""
    work_unit_id = _unit_id(suite, unit)
    resource = result.get("resource")
    expected_keys = {
        "node",
        "cpu_model",
        "partition",
        "requested_cpus",
        "allocated_cpus",
        "comsol_np",
        "slurm_job_id",
        "slurm_step_id",
        "peak_memory_bytes",
        "peak_scratch_bytes",
    }
    if not isinstance(resource, dict) or set(resource) != expected_keys:
        message = f"Benchmark resource evidence is missing for {work_unit_id}."
        raise TypeError(message)
    cores = unit.variant.cores_per_case
    if any(resource.get(key) != cores for key in ("requested_cpus", "allocated_cpus", "comsol_np")):
        message = f"Benchmark allocation evidence conflicts for {work_unit_id}."
        raise ValueError(message)
    if resource.get("partition") != suite.partition:
        message = f"Benchmark partition evidence conflicts for {work_unit_id}."
        raise ValueError(message)
    node = resource.get("node")
    if not isinstance(node, str) or not node or any(character in node for character in "\r\n\t"):
        message = f"Benchmark node evidence is malformed for {work_unit_id}."
        raise ValueError(message)
    if resource.get("cpu_model") is not None and not isinstance(resource.get("cpu_model"), str):
        message = f"Benchmark CPU model evidence is malformed for {work_unit_id}."
        raise ValueError(message)
    if _JOB_ID_PATTERN.fullmatch(str(resource.get("slurm_job_id"))) is None or _JOB_ID_PATTERN.fullmatch(str(resource.get("slurm_step_id"))) is None:
        message = f"Benchmark Slurm job or step identity is malformed for {work_unit_id}."
        raise ValueError(message)
    for key in ("peak_memory_bytes", "peak_scratch_bytes"):
        value = resource.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            message = f"Benchmark {key} is malformed for {work_unit_id}."
            raise ValueError(message)


def _finite_nonnegative(value: object) -> bool:
    """Return whether one optional evidence value is finite and non-negative."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def _validate_scheduler_timing(
    result: Mapping[str, Any],
    *,
    work_unit_id: str,
) -> None:
    """Validate scheduler evidence while allowing unavailable worker queue time."""
    timing = result.get("scheduler_timing")
    expected_keys = {"submit_time", "start_time", "completion_time", "queue_wait_s", "turnaround_s"}
    if not isinstance(timing, dict) or set(timing) != expected_keys:
        message = f"Benchmark scheduler timing is incomplete for {work_unit_id}."
        raise ValueError(message)
    submitted = benchmark_report.parse_benchmark_timestamp(timing["submit_time"], label="submit_time")
    completed = benchmark_report.parse_benchmark_timestamp(timing["completion_time"], label="completion_time")
    started = timing["start_time"]
    queue_wait = timing["queue_wait_s"]
    if started is None or queue_wait is None:
        if started is not None or queue_wait is not None:
            message = f"Benchmark scheduler queue evidence is inconsistent for {work_unit_id}."
            raise ValueError(message)
        expected_turnaround = (completed - submitted).total_seconds()
    else:
        started_at = benchmark_report.parse_benchmark_timestamp(started, label="start_time")
        if started_at < submitted or completed < started_at:
            message = f"Benchmark scheduler timestamps are out of order for {work_unit_id}."
            raise ValueError(message)
        expected_queue = (started_at - submitted).total_seconds()
        if not _finite_nonnegative(queue_wait) or not math.isclose(float(queue_wait), expected_queue, rel_tol=1.0e-9, abs_tol=1.0e-6):
            message = f"Benchmark queue_wait_s is inconsistent for {work_unit_id}."
            raise ValueError(message)
        expected_turnaround = (completed - submitted).total_seconds()
    turnaround = timing["turnaround_s"]
    if not _finite_nonnegative(turnaround) or not math.isclose(float(turnaround), expected_turnaround, rel_tol=1.0e-9, abs_tol=1.0e-6):
        message = f"Benchmark turnaround_s is inconsistent for {work_unit_id}."
        raise ValueError(message)


def _scheduler_timing(
    *,
    submit_time: str,
    completion_time: str,
) -> dict[str, Any]:
    """Return worker scheduler evidence without claiming unavailable queue timing."""
    submitted = benchmark_report.parse_benchmark_timestamp(submit_time, label="submit_time")
    completed = benchmark_report.parse_benchmark_timestamp(completion_time, label="completion_time")
    if completed < submitted:
        message = "Benchmark completion time precedes submission time."
        raise RuntimeError(message)
    return {
        "submit_time": submit_time,
        "start_time": None,
        "completion_time": completion_time,
        "queue_wait_s": None,
        "turnaround_s": (completed - submitted).total_seconds(),
    }


def _validate_worker_interval(result: Mapping[str, Any], *, work_unit_id: str) -> None:
    """Validate optional schema-v1 worker UTC and environment provenance."""
    interval = result.get("worker_interval")
    if interval is None:
        return
    if not isinstance(interval, dict) or set(interval) != {"started_at", "ended_at", "slurm_job_start_time"}:
        message = f"Benchmark worker interval is malformed for {work_unit_id}."
        raise ValueError(message)
    if benchmark_report.parse_benchmark_timestamp(interval["ended_at"], label="worker ended_at") < benchmark_report.parse_benchmark_timestamp(
        interval["started_at"], label="worker started_at"
    ):
        message = f"Benchmark worker interval is out of order for {work_unit_id}."
        raise ValueError(message)
    evidence = interval["slurm_job_start_time"]
    if not isinstance(evidence, dict) or set(evidence) != {"status", "raw_value", "started_at", "warning"}:
        message = f"Benchmark Slurm start-time evidence is malformed for {work_unit_id}."
        raise ValueError(message)
    status = evidence["status"]
    raw_value = evidence["raw_value"]
    started_at = evidence["started_at"]
    warning = evidence["warning"]
    if (
        status not in {"available", "unavailable", "invalid"}
        or (status == "available") != isinstance(started_at, str)
        or (status == "unavailable") != (raw_value is None)
        or (status == "available") != (warning is None)
        or (raw_value is not None and (not isinstance(raw_value, str) or len(raw_value) > _MAX_SLURM_ENV_VALUE_CHARACTERS))
        or (warning is not None and (not isinstance(warning, str) or len(warning) > _MAX_SLURM_ENV_VALUE_CHARACTERS))
    ):
        message = f"Benchmark Slurm start-time evidence is inconsistent for {work_unit_id}."
        raise ValueError(message)
    if isinstance(started_at, str):
        benchmark_report.parse_benchmark_timestamp(started_at, label="SLURM_JOB_START_TIME")


def _validate_work_unit_timings(
    result: Mapping[str, Any],
    *,
    work_unit_id: str,
    success: bool,
) -> None:
    """Validate exact separated timings without mixing waits into solve time."""
    timings = result.get("timings_seconds")
    if not isinstance(timings, dict) or set(timings) != benchmark_report.WORK_UNIT_TIMING_FIELDS:
        message = f"Benchmark timings are incomplete for {work_unit_id}."
        raise ValueError(message)
    optional = {"scheduler_queue_seconds", "comsol_process_seconds", "export_conversion_seconds"}
    for key, value in timings.items():
        if key in optional and value is None:
            continue
        if not _finite_nonnegative(value):
            message = f"Benchmark {key} is malformed for {work_unit_id}."
            raise ValueError(message)
    if success and (timings["comsol_process_seconds"] is None or float(timings["comsol_process_seconds"]) <= 0.0):
        message = f"Benchmark successful COMSOL runtime is invalid for {work_unit_id}."
        raise ValueError(message)


def _validate_success_support_evidence(
    result: Mapping[str, Any],
    *,
    work_unit_id: str,
) -> None:
    """Validate successful solver interval, license, and bounded-log evidence."""
    interval = result.get("solver_interval")
    if not isinstance(interval, dict) or set(interval) != {"started_at", "ended_at"}:
        message = f"Benchmark solver interval is missing for {work_unit_id}."
        raise ValueError(message)
    if benchmark_report.parse_benchmark_timestamp(interval["ended_at"], label="solver ended_at") < benchmark_report.parse_benchmark_timestamp(
        interval["started_at"],
        label="solver started_at",
    ):
        message = f"Benchmark solver interval is out of order for {work_unit_id}."
        raise ValueError(message)
    license_evidence = result.get("license")
    expected_license_keys = {
        "license_blocked_submission_count",
        "license_wait_seconds",
        "license_probe_seconds",
        "scheduler_queue_seconds_before_success",
        "detected_feature",
        "detected_error_code",
        "matched_signatures",
        "raw_excerpt",
        "successful_artifacts_override_prior_warning",
    }
    if not isinstance(license_evidence, dict) or set(license_evidence) != expected_license_keys:
        message = f"Benchmark license evidence is missing for {work_unit_id}."
        raise ValueError(message)
    blocked = license_evidence["license_blocked_submission_count"]
    if isinstance(blocked, bool) or not isinstance(blocked, int) or blocked < 0:
        message = f"Benchmark license-block count is malformed for {work_unit_id}."
        raise ValueError(message)
    for key in (
        "license_wait_seconds",
        "license_probe_seconds",
        "scheduler_queue_seconds_before_success",
    ):
        value = license_evidence[key]
        if key == "scheduler_queue_seconds_before_success" and value is None:
            continue
        if not _finite_nonnegative(value):
            message = f"Benchmark {key} is malformed for {work_unit_id}."
            raise ValueError(message)
    if license_evidence["successful_artifacts_override_prior_warning"] is not True:
        message = f"Benchmark success precedence is missing for {work_unit_id}."
        raise ValueError(message)
    signatures = license_evidence["matched_signatures"]
    if not isinstance(signatures, list) or not all(isinstance(item, str) and item for item in signatures):
        message = f"Benchmark license signatures are malformed for {work_unit_id}."
        raise ValueError(message)
    optional_text = (
        license_evidence["detected_feature"],
        license_evidence["detected_error_code"],
        license_evidence["raw_excerpt"],
    )
    if not all(item is None or isinstance(item, str) for item in optional_text):
        message = f"Benchmark license details are malformed for {work_unit_id}."
        raise ValueError(message)
    solver_log = result.get("solver_log")
    if not isinstance(solver_log, dict) or set(solver_log) != {
        "sha256",
        "size_bytes",
        "excerpt",
        "excerpt_truncated",
    }:
        message = f"Benchmark bounded solver-log evidence is missing for {work_unit_id}."
        raise ValueError(message)
    size = solver_log["size_bytes"]
    if (
        _SHA256_PATTERN.fullmatch(str(solver_log["sha256"])) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(solver_log["excerpt"], str)
        or not isinstance(solver_log["excerpt_truncated"], bool)
        or len(solver_log["excerpt"].encode("utf-8")) > benchmark_report.MAX_BENCHMARK_LOG_EXCERPT_BYTES * 3
    ):
        message = f"Benchmark bounded solver-log evidence is malformed for {work_unit_id}."
        raise ValueError(message)


def _validate_success_result(
    result: Mapping[str, Any],
    *,
    suite: benchmark_config.CoreBenchmarkSuite,
    unit: _WorkUnit,
) -> None:
    """Validate one immutable successful representative-case measurement."""
    work_unit_id = _unit_id(suite, unit)
    _validate_result_identity(result, suite=suite, unit=unit, status="success")
    _validate_resource_evidence(result, suite=suite, unit=unit)
    _validate_scheduler_timing(result, work_unit_id=work_unit_id)
    _validate_worker_interval(result, work_unit_id=work_unit_id)
    _validate_work_unit_timings(result, work_unit_id=work_unit_id, success=True)
    _validate_success_support_evidence(result, work_unit_id=work_unit_id)
    hdf5 = result.get("hdf5")
    if not isinstance(hdf5, dict) or hdf5.get("retained_as_scientific_case") is not False:
        message = f"Benchmark HDF5 isolation evidence is missing for {work_unit_id}."
        raise ValueError(message)
    size = hdf5.get("size_bytes")
    identity = hdf5.get("identity")
    if (
        _SHA256_PATTERN.fullmatch(str(hdf5.get("sha256"))) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(identity, dict)
        or identity.get("case_input_id") != result["case_input_id"]
        or identity.get("simulation_case_id") != result["simulation_case_id"]
    ):
        message = f"Benchmark HDF5 evidence is malformed for {work_unit_id}."
        raise ValueError(message)


def _validate_failure_result(
    result: Mapping[str, Any],
    *,
    suite: benchmark_config.CoreBenchmarkSuite,
    unit: _WorkUnit,
) -> None:
    """Validate one append-only failed work-unit attempt."""
    work_unit_id = _unit_id(suite, unit)
    _validate_result_identity(result, suite=suite, unit=unit, status="failed")
    _validate_resource_evidence(result, suite=suite, unit=unit)
    _validate_scheduler_timing(result, work_unit_id=work_unit_id)
    _validate_worker_interval(result, work_unit_id=work_unit_id)
    _validate_work_unit_timings(result, work_unit_id=work_unit_id, success=False)
    error = result.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("type"), str) or not error["type"] or not isinstance(error.get("message"), str):
        message = f"Benchmark failure error evidence is malformed for {work_unit_id}."
        raise ValueError(message)


def _validate_materialized_case_proof(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Require measured scratch inputs to equal one canonical case proof."""
    if actual != expected:
        message = "Measured benchmark materialization differs from the canonical case proof."
        raise RuntimeError(message)


def _validate_hdf5_scientific_identity(
    hdf5_identity: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    """Require admitted HDF5 output to retain benchmark scientific identity."""
    if hdf5_identity["case_input_id"] != proof["case_input_id"] or hdf5_identity["simulation_case_id"] != proof["simulation_case_id"]:
        message = "Benchmark HDF5 admission changed the canonical scientific identity."
        raise RuntimeError(message)


def _active_benchmark_job_ids(scheduler: Mapping[str, Any]) -> frozenset[str]:
    """Return exact active root job IDs from one admitted scheduler query."""
    result: set[str] = set()
    for line in str(scheduler["squeue"]["output"]).splitlines():
        job_id, separator, _remainder = line.partition("|")
        if separator and _JOB_ID_PATTERN.fullmatch(job_id) is not None:
            result.add(job_id)
    return frozenset(result)


def _work_unit_directory(directory: Path, suite: benchmark_config.CoreBenchmarkSuite, unit: _WorkUnit) -> Path:
    """Return one benchmark work-unit evidence directory."""
    return directory / "runs" / unit.sweep.sweep_id / suite.execution_id(unit.variant) / _unit_id(suite, unit)


def _work_unit_attempts(directory: Path, suite: benchmark_config.CoreBenchmarkSuite, unit: _WorkUnit) -> tuple[Path, ...]:
    """Return one sorted immutable attempt chain."""
    return tuple(sorted(_work_unit_directory(directory, suite, unit).glob("attempt-*.json")))


def _persist_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically replace mutable benchmark orchestration state."""
    common.serialization.atomic_write_json(path, dict(manifest))


@dataclass(frozen=True, slots=True)
class _WorkUnit:
    """One measured case of one variant attempt in one sweep."""

    sweep: benchmark_config.CoreBenchmarkSweep
    variant: benchmark_config.CoreBenchmarkVariant
    attempt: int
    case_position: int


def _unit_id(suite: benchmark_config.CoreBenchmarkSuite, unit: _WorkUnit) -> str:
    """Return the stable work-unit identity."""
    return suite.work_unit_id(unit.sweep, unit.variant, unit.attempt, unit.case_position)


def _variant_units(
    suite: benchmark_config.CoreBenchmarkSuite,
    sweep: benchmark_config.CoreBenchmarkSweep,
    variant: benchmark_config.CoreBenchmarkVariant,
    attempt: int,
) -> tuple[_WorkUnit, ...]:
    """Return every work unit of one variant attempt in one sweep."""
    return tuple(_WorkUnit(sweep, variant, attempt, position) for position in range(1, suite.representative_case_count + 1))


def _pending_record(suite: benchmark_config.CoreBenchmarkSuite, unit: _WorkUnit) -> dict[str, Any]:
    """Return the identity of a work unit that has no terminal evidence yet."""
    return {
        "status": "pending",
        "sweep_id": unit.sweep.sweep_id,
        "variant_attempt": unit.attempt,
        "variant_id": unit.variant.variant_id,
        "execution_id": suite.execution_id(unit.variant),
        "case_position": unit.case_position,
        "case_role": suite.representative_case(unit.case_position).case_role,
        "work_unit_id": _unit_id(suite, unit),
        "cores_per_case": unit.variant.cores_per_case,
    }


def _unit_record(directory: Path, suite: benchmark_config.CoreBenchmarkSuite, unit: _WorkUnit) -> dict[str, Any]:
    """Return the validated success, the latest validated failure, or a pending identity of one work unit.

    The whole attempt chain is validated every time, and a success publication must equal its attempt receipt.
    """
    unit_directory = _work_unit_directory(directory, suite, unit)
    attempts = _work_unit_attempts(directory, suite, unit)
    _validate_benchmark_attempt_chain(attempts)
    success_path = unit_directory / "success.json"
    latest = None if not attempts else _load_json(attempts[-1], label="benchmark work-unit attempt")
    if success_path.is_file():
        record = _load_json(success_path, label="benchmark work-unit success")
        if latest != record:
            message = f"Benchmark success does not match its latest attempt receipt: {success_path}"
            raise ValueError(message)
        _validate_success_result(record, suite=suite, unit=unit)
        return record
    if latest is None:
        return _pending_record(suite, unit)
    if latest.get("status") == "failed":
        _validate_failure_result(latest, suite=suite, unit=unit)
        return latest
    if latest.get("status") != "success":
        message = f"Benchmark attempt has an unsupported status: {attempts[-1]}"
        raise ValueError(message)
    _validate_success_result(latest, suite=suite, unit=unit)
    return _pending_record(suite, unit)


def _node_directory(directory: Path, record: Mapping[str, Any]) -> Path:
    """Return the node directory of one sweep node-job record."""
    return directory / "node_jobs" / str(record["token"])


def _sweep_node_jobs(manifest: Mapping[str, Any], sweep: benchmark_config.CoreBenchmarkSweep) -> list[Mapping[str, Any]]:
    """Return the node-job records of one sweep in submission order."""
    return [record for record in manifest["node_jobs"] if record.get("sweep_id") == sweep.sweep_id]


def _node_job_active(record: Mapping[str, Any], active_job_ids: frozenset[str]) -> bool:
    """Return whether one node job may still run tickets or deliver receipts.

    An unresolved submission counts as active, and so does a node job that left the queue less than
    _NOT_STARTED_GRACE_SECONDS ago (or whose departure was not yet observed by the driver): its worker's receipts can
    reach the driver's node over NFS up to about a minute late (probe P1), so it is not yet judged.
    """
    if record["status"] in {"submitting", "unknown"}:
        return True
    if record["status"] != "submitted":
        return False
    if str(record["job_id"]) in active_job_ids:
        return True
    observed = record.get("terminal_observed_at")
    if observed is None:
        return True
    elapsed = (datetime.now(UTC) - benchmark_report.parse_benchmark_timestamp(observed, label="terminal_observed_at")).total_seconds()
    return elapsed < _NOT_STARTED_GRACE_SECONDS


def _observe_terminal_node_jobs(manifest: dict[str, Any], active_job_ids: frozenset[str]) -> bool:
    """Record the first time the driver sees each submitted node job outside the queue; return whether anything changed."""
    changed = False
    for record in manifest["node_jobs"]:
        if record["status"] == "submitted" and str(record["job_id"]) not in active_job_ids and record.get("terminal_observed_at") is None:
            record["terminal_observed_at"] = _utc_now()
            changed = True
    return changed


def _ticket_seq(record: Mapping[str, Any], unit: _WorkUnit, suite: benchmark_config.CoreBenchmarkSuite) -> int | None:
    """Return the ticket sequence of one work unit inside one node-job record, if it holds it."""
    unit_id = _unit_id(suite, unit)
    matches = [int(ticket["seq"]) for ticket in record["tickets"] if ticket["work_unit_id"] == unit_id]
    return matches[0] if len(matches) == 1 else None


def _rss_bytes(value: str) -> int | None:
    """Parse one sacct MaxRSS value (``1234K``, ``2.5G``, plain bytes); None when empty or malformed."""
    text = value.strip()
    if not text:
        return None
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    scale = units.get(text[-1].upper(), 1)
    number = text[:-1] if text[-1].upper() in units else text
    try:
        return int(float(number) * scale)
    except ValueError:
        return None


def _step_rows(scheduler: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]] | None:
    """Return sacct step evidence keyed by (node job id, step name); None when accounting is unavailable."""
    accounting = scheduler.get("sacct", {})
    if accounting.get("error") is not None or "output" not in accounting:
        return None
    return {
        key: {"step_id": row["step_id"], "state": row["state"], "exit_code": row["exit_code"], "max_rss_bytes": _rss_bytes(row["max_rss"])}
        for key, row in cluster_service.parse_step_accounting(str(accounting["output"])).items()
    }


def _terminal_step_rows(directory: Path, manifest: Mapping[str, Any], *, capture: bool) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the step accounting of a finished benchmark from its persisted snapshot; finalization captures it once.

    Validation after finalization reads only the snapshot, so it does not depend on live or expiring Slurm accounting.
    """
    path = directory / _STEP_ACCOUNTING_FILENAME
    job_ids = [str(value) for value in manifest["measured_job_ids"]]
    if capture and not path.exists():
        rows = _step_rows(_scheduler_evidence(job_ids))
        if rows is None:
            message = "Core benchmark step accounting is unavailable; finalization needs it."
            raise RuntimeError(message)
        _write_immutable_json(
            path,
            {
                "schema_kind": "generation_core_scaling_step_accounting",
                "schema_version": 1,
                "job_ids": job_ids,
                "captured_at": _utc_now(),
                "rows": [{"job_id": job_id, "step_name": name, **row} for (job_id, name), row in sorted(rows.items())],
            },
            label="benchmark step accounting",
        )
    payload = _load_json(path, label="benchmark step accounting")
    if (
        payload.get("schema_kind") != "generation_core_scaling_step_accounting"
        or payload.get("schema_version") != 1
        or payload.get("job_ids") != job_ids
        or not isinstance(payload.get("rows"), list)
    ):
        message = f"Core benchmark step accounting snapshot does not match its node jobs: {path}"
        raise ValueError(message)
    return {
        (str(row["job_id"]), str(row["step_name"])): {key: row[key] for key in ("step_id", "state", "exit_code", "max_rss_bytes")}
        for row in payload["rows"]
    }


def _unit_outcome(
    record: Mapping[str, Any],
    finished: Mapping[str, Any] | None,
    step: Mapping[str, Any] | None,
    *,
    step_rows_known: bool,
) -> str:
    """Classify one unit: success, failed (scientific), failed_without_evidence (the step ran and failed), skipped, or pending.

    Interruptions (cancel, node-job end, a step that never started, a launch failure) stay pending, so the variant
    attempt is judged interrupted and measured again instead of counting as a case failure. A candidate the worker
    skipped because its variant already reached the success target is ``skipped``. Success needs the same verdict as
    the worker's target count: success evidence and a zero exit status. Evidence followed by a non-zero exit (for
    example a failed cleanup) is a failure the worker already replaced; a zero exit without success evidence counts as
    a failure once the evidence grace period has passed, since the worker counted it towards the target.
    """
    if finished is not None and finished.get("outcome") == "skipped":
        if record["status"] != "pending":
            message = f"Benchmark work unit {record['work_unit_id']} has case evidence although its ticket was skipped."
            raise ValueError(message)
        return "skipped"
    exited_nonzero = finished is not None and finished.get("outcome") == "exited" and int(finished["exit_code"]) != 0
    if record["status"] == "success":
        return "failed" if exited_nonzero else "success"
    if record["status"] == "failed":
        error = record.get("error") or {}
        return "pending" if error.get("type") in _INTERRUPTION_ERRORS else "failed"
    if finished is None:
        return "pending"
    if int(finished["exit_code"]) == 0:
        ended = benchmark_report.parse_benchmark_timestamp(finished["ended_at"], label="step receipt ended_at")
        return "failed_without_evidence" if (datetime.now(UTC) - ended).total_seconds() >= _BINDING_WAIT_SECONDS else "pending"
    if not step_rows_known:
        return "failed_without_evidence" if finished.get("outcome") == "exited" and int(finished["exit_code"]) not in {130, 143} else "pending"
    if step is None or str(step["state"]) not in _CASE_FAILURE_STEP_STATES:
        return "pending"
    return "failed_without_evidence"


def _variant_holders(
    manifest: Mapping[str, Any], sweep: benchmark_config.CoreBenchmarkSweep, variant: benchmark_config.CoreBenchmarkVariant
) -> list[Mapping[str, Any]]:
    """Return the node-job records of one sweep that hold tickets of one variant, excluding failed submissions."""
    return [
        record
        for record in _sweep_node_jobs(manifest, sweep)
        if record["status"] != "submission_failed" and any(ticket["variant_id"] == variant.variant_id for ticket in record["tickets"])
    ]


def _await_dispatch(
    run_id: str, *, storage: Path, job_id: str, unit_id: str
) -> tuple[dict[str, Any], benchmark_config.CoreBenchmarkSuite, Mapping[str, Any]]:
    """Load the manifest until it shows this step's dispatch, for at most _BINDING_WAIT_SECONDS (NFS visibility delay)."""
    token = os.environ.get("GENERATION_NODE_TOKEN")
    if token is None or _TOKEN_PATTERN.fullmatch(token) is None:
        message = "Benchmark work unit requires the GENERATION_NODE_TOKEN of its node job."
        raise RuntimeError(message)
    deadline = time.monotonic() + _BINDING_WAIT_SECONDS
    while True:
        manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
        try:
            record, _ticket = _dispatched_ticket(manifest, job_id=job_id, unit_id=unit_id, token=token)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(10)
            continue
        return manifest, suite, record


def _variant_state(
    manifest: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    directory: Path,
    sweep: benchmark_config.CoreBenchmarkSweep,
    variant: benchmark_config.CoreBenchmarkVariant,
    active_job_ids: frozenset[str],
    step_rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify the latest attempt of one variant in one sweep from evidence, receipts, step accounting, and liveness.

    measured: every unit has terminal case evidence from the one node job that held the attempt; running: that node job
    may still run or deliver receipts; interrupted: the node job ended before every unit finished and at least one unit
    started; not_started: no unit of the attempt started. Every new node job gives its variants new attempt numbers, so an
    attempt is held by exactly one node job; ``started_attempts`` counts the attempts in which a unit started.
    """
    holders = _variant_holders(manifest, sweep, variant)
    by_attempt: dict[int, list[Mapping[str, Any]]] = {}
    for record in holders:
        for attempt in {int(ticket["variant_attempt"]) for ticket in record["tickets"] if ticket["variant_id"] == variant.variant_id}:
            by_attempt.setdefault(attempt, []).append(record)
    if not by_attempt:
        return {
            "variant_id": variant.variant_id,
            "state": "unplanned",
            "attempt": None,
            "job_id": None,
            "units": [],
            "started_attempts": 0,
            "max_attempt": 0,
        }
    if any(len(owners) != 1 for owners in by_attempt.values()):
        message = f"Benchmark variant {variant.variant_id} of {sweep.sweep_id} has an attempt held by more than one node job."
        raise ValueError(message)
    started_attempts = 0
    for attempt, owners in by_attempt.items():
        holder = owners[0]
        node_directory = _node_directory(directory, holder)
        tickets = [ticket for ticket in holder["tickets"] if ticket["variant_id"] == variant.variant_id and int(ticket["variant_attempt"]) == attempt]
        if step_rows is not None:
            executed = any((str(holder["job_id"]), str(ticket["label"])) in step_rows for ticket in tickets) or any(
                _unit_record(directory, suite, unit)["status"] != "pending" for unit in _variant_units(suite, sweep, variant, attempt)
            )
        else:
            executed = any(node_service.read_node_receipt(node_directory, int(ticket["seq"]), "started") is not None for ticket in tickets)
        started_attempts += int(executed)
    attempt = max(by_attempt)
    holder = by_attempt[attempt][0]
    node_directory = _node_directory(directory, holder)
    units: list[dict[str, Any]] = []
    for unit in _variant_units(suite, sweep, variant, attempt):
        record = _unit_record(directory, suite, unit)
        seq = _ticket_seq(holder, unit, suite)
        ticket = None if seq is None else next(item for item in holder["tickets"] if int(item["seq"]) == seq)
        started = None if seq is None else node_service.read_node_receipt(node_directory, seq, "started")
        finished = None if seq is None else node_service.read_node_receipt(node_directory, seq, "finished")
        step = (
            None if step_rows is None or ticket is None or holder["job_id"] is None else step_rows.get((str(holder["job_id"]), str(ticket["label"])))
        )
        if record["status"] in {"success", "failed"} and str(record["resource"]["slurm_job_id"]) != str(holder["job_id"]):
            message = (
                f"Benchmark evidence of {record['work_unit_id']} comes from job {record['resource']['slurm_job_id']}, "
                f"not its node job {holder['job_id']}."
            )
            raise ValueError(message)
        outcome = _unit_outcome(record, finished, step, step_rows_known=step_rows is not None)
        units.append({"unit": unit, "record": record, "outcome": outcome, "started": started, "finished": finished, "seq": seq, "step": step})
    if all(item["outcome"] != "pending" for item in units):
        state = "measured"
    elif _node_job_active(holder, active_job_ids):
        state = "running"
    elif any(item["started"] is not None or item["record"]["status"] != "pending" for item in units):
        state = "interrupted"
    else:
        state = "not_started"
    return {
        "variant_id": variant.variant_id,
        "state": state,
        "attempt": attempt,
        "job_id": holder["job_id"],
        "token": holder["token"],
        "units": units,
        "started_attempts": started_attempts,
        "max_attempt": attempt,
    }


def _sweep_state(
    manifest: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    directory: Path,
    sweep: benchmark_config.CoreBenchmarkSweep,
    active_job_ids: frozenset[str],
    step_rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify one sweep and, when it needs a node job, the variant attempts that node job must measure."""
    variants = [_variant_state(manifest, suite, directory, sweep, variant, active_job_ids, step_rows) for variant in suite.sweep_variants(sweep)]
    running = any(_node_job_active(record, active_job_ids) for record in _sweep_node_jobs(manifest, sweep))
    result: dict[str, Any] = {"sweep_id": sweep.sweep_id, "variants": variants, "next_variant_attempts": []}
    if running:
        result["state"] = "running"
        return result
    remaining = [item for item in variants if item["state"] != "measured"]
    if not remaining:
        result["state"] = "complete"
        return result
    next_attempts: list[tuple[benchmark_config.CoreBenchmarkVariant, int]] = []
    for item in remaining:
        variant = suite.variant(str(item["variant_id"]))
        if int(item["started_attempts"]) >= suite.maximum_variant_attempts:
            result["state"] = "variant_failed"
            result["failed_variant_id"] = variant.variant_id
            return result
        next_attempts.append((variant, int(item["max_attempt"]) + 1))
    result["state"] = "ready_to_submit"
    result["next_variant_attempts"] = next_attempts
    return result


def _recover_node_job_intent(manifest: dict[str, Any], *, directory: Path, manifest_path: Path) -> None:
    """Resolve an interrupted node-job submission through squeue (name and comment) or the worker's node.json."""
    token = manifest["node_job_intent"]
    if token is None:
        return
    record = next(item for item in manifest["node_jobs"] if item["token"] == token)
    seat_jobs = cluster_service.query_seat_jobs([int(record["seat"])])
    matches = [job_id for job_id, (_seat, _state, comment) in seat_jobs.items() if comment == token]
    if len(matches) > 1:
        message = f"Benchmark node-job recovery for token {token} is ambiguous: {matches}"
        raise RuntimeError(message)
    job_id = matches[0] if matches else None
    if job_id is None:
        node_record = node_service.read_node_record(_node_directory(directory, record), "node")
        job_id = None if node_record is None else str(node_record["job_id"])
    if job_id is None:
        recorded = benchmark_report.parse_benchmark_timestamp(record["recorded_at"], label="node-job recorded_at")
        if (datetime.now(UTC) - recorded).total_seconds() < _NOT_STARTED_GRACE_SECONDS:
            return
        record["status"] = "submission_failed"
        record["error"] = "no queued job and no worker record carry this submission token"
    else:
        record["status"] = "submitted"
        record["job_id"] = job_id
        record["submitted_at"] = record["recorded_at"]
        manifest["measured_job_ids"].append(job_id)
        cluster_service.lease_executing_tree(job_id)
    manifest["node_job_intent"] = None
    _persist_manifest(manifest_path, manifest)


def _submit_sweep_node_job(
    manifest: dict[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    storage: Path,
    directory: Path,
    manifest_path: Path,
    sweep: benchmark_config.CoreBenchmarkSweep,
    variant_attempts: Sequence[tuple[benchmark_config.CoreBenchmarkVariant, int]],
) -> str:
    """Persist the node-job intent and its tickets, submit the sweep node job, and record its job ID."""
    run_id = str(manifest["benchmark_run_id"])
    token = secrets.token_hex(8)
    seat = suite.sweeps.index(sweep) + 1
    tickets = _sweep_tickets(suite, run_id=run_id, sweep=sweep, variant_attempts=variant_attempts, token=token)
    record: dict[str, Any] = {
        "token": token,
        "sweep_id": sweep.sweep_id,
        "seat": seat,
        "status": "submitting",
        "job_id": None,
        "recorded_at": _utc_now(),
        "submitted_at": None,
        "error": None,
        "command": None,
        "tickets": [
            {
                "seq": ticket.seq,
                "variant_id": str(ticket.argv[5]),
                "variant_attempt": int(ticket.argv[6]),
                "case_role": str(ticket.argv[7]),
                "work_unit_id": suite.work_unit_id(
                    sweep, suite.variant(str(ticket.argv[5])), int(ticket.argv[6]), suite.case_position(str(ticket.argv[7]))
                ),
                "label": ticket.label,
            }
            for ticket in tickets
            if ticket.kind == node_service.STEP_KIND
        ],
        "stop_ticket_seq": tickets[-1].seq,
    }
    node_directory = _node_directory(directory, record)
    for subdirectory in (node_service.TICKETS_DIRECTORY, node_service.RECEIPTS_DIRECTORY, node_service.CONTROL_DIRECTORY):
        (node_directory / subdirectory).mkdir(parents=True, exist_ok=True)
    exclusions = cluster_service.standard_node_exclusions(suite.partition, suite.cores_per_node, match="exact")
    site = suite.case_campaign.execution_values["site"]
    record["command"] = cluster_service.build_node_job_submission_command(
        seat=seat,
        submission_token=token,
        node_directory=node_directory,
        storage_root=storage,
        cores_per_node=suite.cores_per_node,
        node_job_memory=suite.sweep_node_job_memory(),
        node_job_wall_time=suite.sweep_wall_time,
        partition=suite.partition,
        exclusions=exclusions,
        comsol_module=str(site["comsol_module"]),
        python_executable=str(site["python_executable"]),
        comsol_executable=str(site["comsol_executable"]),
        scheduler_options=suite.scheduler_options,
    )
    manifest["node_jobs"].append(record)
    manifest["node_job_intent"] = token
    source_service.validate_admitted_source_before_publication()
    _persist_manifest(manifest_path, manifest)
    for ticket in tickets:
        node_service.write_node_ticket(node_directory, ticket)
    try:
        job_id = _submit(record["command"], git_commit=str(manifest["git_commit"]), run_id=run_id)
    except subprocess.CalledProcessError as error:
        record["status"] = "submission_failed"
        record["error"] = error.stderr.strip() or str(error)
        manifest["node_job_intent"] = None
        _persist_manifest(manifest_path, manifest)
        raise
    record["status"] = "submitted"
    record["job_id"] = job_id
    record["submitted_at"] = _utc_now()
    manifest["measured_job_ids"].append(job_id)
    manifest["node_job_intent"] = None
    manifest["state"] = "running"
    _persist_manifest(manifest_path, manifest)
    return job_id


def _advance_sweeps(
    manifest: dict[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    storage: Path,
) -> dict[str, Any]:
    """Submit the next node job of every sweep that needs one and derive the benchmark state."""
    run_id = str(manifest["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    if manifest.get("state") in {"cancel_requested", "force_cancel_requested"}:
        return manifest
    _recover_node_job_intent(manifest, directory=directory, manifest_path=manifest_path)
    if manifest["node_job_intent"] is not None:
        return manifest
    proof_paths = tuple(_canonical_case_proof_path(directory, representative) for representative in suite.representative_cases)
    if not all(path.is_file() for path in proof_paths):
        manifest["state"] = "inputs_ready"
        _persist_manifest(manifest_path, manifest)
        return manifest
    scheduler = _scheduler_evidence(list(manifest["measured_job_ids"]))
    if scheduler["squeue"]["error"] is not None:
        message = f"Could not verify active benchmark node jobs: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    active_job_ids = _active_benchmark_job_ids(scheduler)
    if _observe_terminal_node_jobs(manifest, active_job_ids):
        _persist_manifest(manifest_path, manifest)
    states: list[str] = []
    step_rows = _step_rows(scheduler)
    if step_rows is None:
        message = f"Could not read benchmark step accounting: {scheduler['sacct']['error']}"
        raise RuntimeError(message)
    for sweep in suite.sweeps:
        sweep_state = _sweep_state(manifest, suite, directory, sweep, active_job_ids, step_rows)
        if sweep_state["state"] == "ready_to_submit":
            _submit_sweep_node_job(
                manifest,
                suite,
                storage=storage,
                directory=directory,
                manifest_path=manifest_path,
                sweep=sweep,
                variant_attempts=sweep_state["next_variant_attempts"],
            )
            states.append("running")
        else:
            states.append(sweep_state["state"])
    if all(state == "complete" for state in states):
        manifest["state"] = "complete"
    elif "running" in states:
        manifest["state"] = "running"
    else:
        manifest["state"] = "variant_failed"
    _persist_manifest(manifest_path, manifest)
    return manifest


def _prepare_core_benchmark_locked(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    storage: Path,
    scratch_root: Path | str,
) -> tuple[dict[str, Any], benchmark_config.CoreBenchmarkSuite]:
    """Materialize one manifest and canonical inputs while submission is locked."""
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    manifest_path = directory / "benchmark_manifest.json"
    if manifest_path.exists():
        manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
        if manifest["state"] in {"cancel_requested", "force_cancel_requested"}:
            scheduler = _scheduler_evidence(list(manifest["measured_job_ids"]))
            if scheduler["squeue"]["error"] is not None:
                message = "Could not verify benchmark cancellation before explicit resume."
                raise RuntimeError(message)
            if scheduler["squeue"]["output"]:
                return manifest, suite
            manifest["state"] = "incomplete"
            _persist_manifest(manifest_path, manifest)
    else:
        suite = benchmark_config.load_core_benchmark_suite(path, require_executable=True)
        manifest = {
            "schema_kind": BENCHMARK_RUN_SCHEMA_KIND,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_run_id": run_id,
            "suite_name": suite.suite_name,
            "suite_digest": suite.suite_digest,
            "suite_config": benchmark_config.repository_relative(suite.source_path),
            "git_commit": plan["git_commit"],
            "source_fingerprint_sha256": plan["source_fingerprint_sha256"],
            "preflight": plan["preflight"],
            "representative_cases": suite.case_selections(),
            "variants": _variant_records(suite),
            "sweeps": _sweep_records(suite),
            "required_successes_per_sweep": plan["required_successes_per_sweep"],
            "resource_contract": suite.resource_contract(),
            "created_at": _utc_now(),
            "state": "preparing_inputs",
            "measured_job_ids": [],
            "node_jobs": [],
            "node_job_intent": None,
        }
        _persist_manifest(manifest_path, manifest)
    proof_paths = tuple(_canonical_case_proof_path(directory, representative) for representative in suite.representative_cases)
    if not all(path.is_file() for path in proof_paths):
        try:
            _materialize_core_benchmark_inputs(run_id, storage_root=storage, work_root=scratch_root)
        except Exception:
            manifest["state"] = "input_preparation_failed"
            _persist_manifest(manifest_path, manifest)
            raise
        manifest["state"] = "inputs_ready"
        _persist_manifest(manifest_path, manifest)
    return manifest, suite


def prepare_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Materialize canonical benchmark input without submitting a measurement."""
    plan = plan_core_benchmark(
        path,
        git_commit=git_commit,
        storage_root=storage_root,
        scratch_root=scratch_root,
        comsol_version_output=comsol_version_output,
        comsol_executable_path=comsol_executable_path,
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        manifest, _suite = _prepare_core_benchmark_locked(
            path,
            plan,
            storage=storage,
            scratch_root=scratch_root,
        )
    return manifest


def submit_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Reuse canonical Slurm-prepared inputs, then submit the sweep node jobs that are due."""
    plan = plan_core_benchmark(
        path,
        git_commit=git_commit,
        storage_root=storage_root,
        scratch_root=scratch_root,
        comsol_version_output=comsol_version_output,
        comsol_executable_path=comsol_executable_path,
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    with common.locking.exclusive_file_lock(directory / "submission.lock", blocking=False):
        manifest, suite = _prepare_core_benchmark_locked(path, plan, storage=storage, scratch_root=scratch_root)
        return _advance_sweeps(manifest, suite, storage=storage)


def resume_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
    scratch_root: Path | str | None = None,
) -> dict[str, Any]:
    """Repair missing input readiness using current scratch, then advance every sweep."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    with common.locking.exclusive_file_lock(manifest_path.parent / "submission.lock", blocking=False):
        manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
        _require_current_checkout(manifest)
        proof_paths = tuple(_canonical_case_proof_path(manifest_path.parent, representative) for representative in suite.representative_cases)
        if not all(path.is_file() for path in proof_paths):
            preflight = _load_core_benchmark_preflight(run_id, suite=suite, git_commit=str(manifest["git_commit"]), storage_root=storage)
            scratch = benchmark_config.require_mapping(
                benchmark_config.require_mapping(preflight["storage_capabilities"], label="benchmark preflight storage capabilities")["scratch"],
                label="benchmark preflight scratch capability",
            )
            work_root = Path(str(scratch["path"])) if scratch_root is None else Path(scratch_root)
            if scratch_root is not None:
                _probe_directory_capability(work_root, label="benchmark scratch")
            try:
                _materialize_core_benchmark_inputs(run_id, storage_root=storage, work_root=work_root)
            except Exception:
                manifest["state"] = "input_preparation_failed"
                _persist_manifest(manifest_path, manifest)
                raise
            manifest["state"] = "inputs_ready"
            _persist_manifest(manifest_path, manifest)
        return _advance_sweeps(manifest, suite, storage=storage)


def _canonical_case_proof_path(
    directory: Path,
    representative: benchmark_config.CoreBenchmarkRepresentativeCase,
) -> Path:
    """Return one immutable representative-case proof path."""
    return directory / "canonical_cases" / f"{representative.case_role}.json"


def _benchmark_input_directory(
    directory: Path,
    representative: benchmark_config.CoreBenchmarkRepresentativeCase,
) -> Path:
    """Return the benchmark-owned raw case bundle directory of one representative case."""
    return directory / "inputs" / representative.case_role


def _generate_benchmark_input_bundle(
    suite: benchmark_config.CoreBenchmarkSuite,
    representative: benchmark_config.CoreBenchmarkRepresentativeCase,
    directory: Path,
) -> Path:
    """Generate (or reuse) one deterministic raw case bundle inside the benchmark run, never in canonical storage."""
    final = _benchmark_input_directory(directory, representative)
    if final.exists():
        return final
    staging = final.parent / f".staging-{representative.case_role}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    case_service.generate_case_input_bundle(suite.case_config, representative.case_index, staging)
    staging.rename(final)
    return final


def _proof_payload(
    suite: benchmark_config.CoreBenchmarkSuite,
    representative: benchmark_config.CoreBenchmarkRepresentativeCase,
    prepared: preparation_service.PreparedCase,
    *,
    canonical_input_preparation_seconds: float,
) -> dict[str, Any]:
    """Return one exact scientific byte identity reused across all core waves."""
    if not math.isfinite(canonical_input_preparation_seconds) or canonical_input_preparation_seconds < 0.0:
        message = "Canonical benchmark input preparation timing is invalid."
        raise ValueError(message)
    case = prepared.bundle.case_payload
    return {
        "schema_kind": BENCHMARK_PROOF_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_digest": suite.suite_digest,
        "case_role": representative.case_role,
        "case_id": prepared.bundle.case_id,
        "case_index": representative.case_index,
        "case_input_id": prepared.bundle.case_input_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "scientific_config_digest": case["scientific_config_digest"],
        "case_input_config_digest": case["case_input_config_digest"],
        "material_family": case["material_family"],
        "sampling_regime": case["sampling_regime"],
        "case_seed": case["seed_evidence"]["case_seed"],
        "sampled_values_sha256": common.serialization.canonical_json_sha256(case["sampled_values"]),
        "input_files": case["input_files"],
        "case_payload_sha256": common.serialization.canonical_json_sha256(case),
        "template": case["template"],
        "canonical_input_preparation_seconds": canonical_input_preparation_seconds,
    }


def _cleanup_prepared(
    prepared: preparation_service.PreparedCase,
    *,
    storage: Path,
) -> int:
    """Remove one exact marked benchmark case workspace."""
    return workspace_service.cleanup_case_workspace(
        prepared.work_directory,
        allowed_root=prepared.work_root,
        storage_root=storage,
        expected_run_id=prepared.workspace_run_id,
        expected_case_id=prepared.bundle.case_id,
        allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
    )


def _materialize_core_benchmark_inputs(
    run_id: str,
    *,
    storage_root: Path | str,
    work_root: Path | str,
) -> tuple[Path, ...]:
    """Generate both benchmark-owned raw inputs and their proofs in the allocated CPU worker."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    _require_current_checkout(manifest)
    if source_service.required_git_commit() != manifest["git_commit"]:
        message = "Benchmark preparation checkout does not match the launch commit."
        raise RuntimeError(message)
    directory = core_benchmark_directory(run_id, storage_root=storage)
    paths: list[Path] = []
    for case_position in range(1, suite.representative_case_count + 1):
        representative = suite.representative_case(case_position)
        path = _canonical_case_proof_path(directory, representative)
        if path.is_file():
            _load_case_proof(
                run_id,
                suite,
                case_position,
                storage_root=storage,
            )
            paths.append(path)
            continue
        preparation_start = time.monotonic()
        raw_case_directory = _generate_benchmark_input_bundle(suite, representative, directory)
        prepared: preparation_service.PreparedCase | None = None
        try:
            prepared = preparation_service.prepare_case_work_directory(
                suite.case_config,
                representative.case_index,
                storage_root=storage,
                work_root=work_root,
                raw_case_directory=raw_case_directory,
            )
            proof = _proof_payload(
                suite,
                representative,
                prepared,
                canonical_input_preparation_seconds=(time.monotonic() - preparation_start),
            )
            source_service.validate_admitted_source_before_publication()
            _write_immutable_json(
                path,
                proof,
                label=f"benchmark canonical-case proof {representative.case_role}",
            )
            paths.append(path)
        finally:
            if prepared is not None:
                _cleanup_prepared(prepared, storage=storage)
    return tuple(paths)


def _load_case_proof(
    run_id: str,
    suite: benchmark_config.CoreBenchmarkSuite,
    case_position: int,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate one prepared representative-case proof."""
    representative = suite.representative_case(case_position)
    path = _canonical_case_proof_path(
        core_benchmark_directory(run_id, storage_root=storage_root),
        representative,
    )
    proof = _load_json(path, label="benchmark canonical-case proof")
    template = proof.get("template")
    preparation_seconds = proof.get("canonical_input_preparation_seconds")
    if (
        proof.get("schema_kind") != BENCHMARK_PROOF_SCHEMA_KIND
        or proof.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or proof.get("suite_digest") != suite.suite_digest
        or proof.get("case_role") != representative.case_role
        or proof.get("case_index") != representative.case_index
        or proof.get("case_id") != suite.case_config.case_id(representative.case_index)
        or proof.get("scientific_config_digest") != suite.case_config.scientific_config_digest
        or proof.get("case_input_config_digest") != suite.case_config.case_input_config_digest
        or not isinstance(template, dict)
        or template.get("sha256") != suite.case_config.template_sha256
        or isinstance(preparation_seconds, bool)
        or not isinstance(preparation_seconds, (int, float))
        or not math.isfinite(float(preparation_seconds))
        or float(preparation_seconds) < 0.0
    ):
        message = f"Benchmark canonical-case proof is invalid: {path}"
        raise ValueError(message)
    for key in (
        "case_input_id",
        "simulation_case_id",
        "sampled_values_sha256",
        "case_payload_sha256",
    ):
        if _SHA256_PATTERN.fullmatch(str(proof.get(key))) is None:
            message = f"Benchmark canonical-case proof has malformed {key}: {path}"
            raise ValueError(message)
    return proof


def _benchmark_peak_child_memory_bytes() -> int:
    """Return Linux child-process peak resident memory in bytes."""
    peak_kibibytes = resource_usage.getrusage(resource_usage.RUSAGE_CHILDREN).ru_maxrss
    if not math.isfinite(float(peak_kibibytes)) or peak_kibibytes < 0:
        message = "Benchmark child-process peak memory evidence is invalid."
        raise RuntimeError(message)
    return int(peak_kibibytes * 1024)


def _benchmark_scratch_bytes(directory: Path) -> int:
    """Return regular-file bytes in one symlink-free benchmark workspace."""
    total = 0
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            message = f"Benchmark scratch contains a symbolic link: {candidate}"
            raise RuntimeError(message)
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _bounded_solver_log(path: Path) -> dict[str, Any]:
    """Return a digest and bounded excerpt without retaining the full solver log."""
    if not path.is_file() or path.is_symlink():
        message = f"Benchmark solver log is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    raw = path.read_bytes()
    excerpt = raw[: benchmark_report.MAX_BENCHMARK_LOG_EXCERPT_BYTES].decode(
        "utf-8",
        errors="replace",
    )
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "excerpt": excerpt,
        "excerpt_truncated": len(raw) > benchmark_report.MAX_BENCHMARK_LOG_EXCERPT_BYTES,
    }


def _dispatched_ticket(
    manifest: Mapping[str, Any],
    *,
    job_id: str,
    unit_id: str,
    token: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the node-job record and ticket that bind one work unit to the current node job.

    A submitted record must carry this job ID; with the ticket's token, a record whose submission the driver has not yet
    acknowledged (status submitting) also binds, because its tickets are already executable.
    """
    matches = [
        (record, ticket)
        for record in manifest["node_jobs"]
        if (record.get("status") == "submitted" and str(record.get("job_id")) == job_id)
        or (token is not None and record.get("token") == token and record.get("status") == "submitting")
        for ticket in record["tickets"]
        if ticket["work_unit_id"] == unit_id
    ]
    if len(matches) != 1:
        message = "Current benchmark node job does not hold exactly one ticket for this work unit."
        raise RuntimeError(message)
    return matches[0]


def _successful_license_evidence(wait: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return compact license evidence while letting success remain authoritative."""
    if wait is None:
        return {
            "license_blocked_submission_count": 0,
            "license_wait_seconds": 0.0,
            "license_probe_seconds": 0.0,
            "scheduler_queue_seconds_before_success": None,
            "detected_feature": None,
            "detected_error_code": None,
            "matched_signatures": [],
            "raw_excerpt": None,
            "successful_artifacts_override_prior_warning": True,
        }
    return {
        "license_blocked_submission_count": int(wait["retry_count"]),
        "license_wait_seconds": float(wait["cumulative_wait_seconds"]),
        "license_probe_seconds": float(wait["cumulative_probe_seconds"]),
        "scheduler_queue_seconds_before_success": wait["cumulative_scheduler_queue_seconds"],
        "detected_feature": wait["feature"],
        "detected_error_code": wait["error_code"],
        "matched_signatures": list(wait["matched_signatures"]),
        "raw_excerpt": wait["raw_excerpt"],
        "successful_artifacts_override_prior_warning": True,
    }


def _total_controller_elapsed_seconds(submitted_at: str, completion_time: str) -> float:
    """Measure from the node-job submission through terminal work-unit evidence."""
    submitted = benchmark_report.parse_benchmark_timestamp(submitted_at, label="benchmark submitted_at")
    completed = benchmark_report.parse_benchmark_timestamp(completion_time, label="benchmark completion_time")
    elapsed = (completed - submitted).total_seconds()
    if elapsed < 0.0:
        message = "Benchmark controller elapsed timing is negative."
        raise RuntimeError(message)
    return elapsed


def run_core_benchmark_case(
    run_id: str,
    sweep_id: str,
    variant_id: str,
    variant_attempt: int,
    case_role: str,
    *,
    storage_root: Path | str,
    work_root: Path | str,
) -> dict[str, Any]:
    """Run one work unit as a step of its sweep node job and retain compact performance evidence."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    _require_current_checkout(manifest)
    worker_started_at = _utc_now()
    start_time_evidence = _slurm_scheduler_start_time_evidence()
    unit = _WorkUnit(suite.sweep(sweep_id), suite.variant(variant_id), variant_attempt, suite.case_position(case_role))
    variant = unit.variant
    representative = suite.representative_case(unit.case_position)
    work_unit_id = _unit_id(suite, unit)
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    if job_id is None or _JOB_ID_PATTERN.fullmatch(job_id) is None or step_id is None or _JOB_ID_PATTERN.fullmatch(step_id) is None:
        message = "Benchmark work unit requires one numeric SLURM_JOB_ID and SLURM_STEP_ID (a step of its node job)."
        raise RuntimeError(message)
    manifest, suite, node_record = _await_dispatch(run_id, storage=storage, job_id=job_id, unit_id=work_unit_id)
    submit_time = str(node_record["submitted_at"] or node_record["recorded_at"])
    if allocated_cpus is None or not allocated_cpus.isdigit() or int(allocated_cpus) != variant.cores_per_case:
        message = f"Benchmark step must exactly match cores_per_case; requested={variant.cores_per_case}, allocated={allocated_cpus!r}."
        raise RuntimeError(message)
    proof = _load_case_proof(run_id, suite, unit.case_position, storage_root=storage)
    directory = core_benchmark_directory(run_id, storage_root=storage)
    work_unit_directory = _work_unit_directory(directory, suite, unit)
    work_unit_directory.mkdir(parents=True, exist_ok=True)
    success_path = work_unit_directory / "success.json"
    lock_path = work_unit_directory / "execution.lock"
    node_name = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    cpu_model = node_service.read_cpu_model()
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        attempts = tuple(sorted(work_unit_directory.glob("attempt-*.json")))
        _validate_benchmark_attempt_chain(attempts)
        if success_path.exists():
            existing_success = _load_json(success_path, label="benchmark work-unit success")
            _validate_success_result(existing_success, suite=suite, unit=unit)
            return existing_success
        attempt_number = len(attempts) + 1
        previous_attempt = _benchmark_previous_attempt_reference(attempts)
        license_evidence = _successful_license_evidence(None)
        attempt_path = work_unit_directory / f"attempt-{attempt_number:04d}.json"
        prepared: preparation_service.PreparedCase | None = None
        success: dict[str, Any]
        identity = {
            "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_run_id": run_id,
            "suite_digest": suite.suite_digest,
            "sweep_id": unit.sweep.sweep_id,
            "variant_attempt": unit.attempt,
            "variant_id": variant.variant_id,
            "execution_id": suite.execution_id(variant),
            "case_position": unit.case_position,
            "case_role": representative.case_role,
            "work_unit_id": work_unit_id,
            "attempt": attempt_number,
            "previous_attempt": previous_attempt,
            "git_commit": manifest["git_commit"],
            "case_input_id": proof["case_input_id"],
            "simulation_case_id": proof["simulation_case_id"],
            "scientific_config_digest": proof["scientific_config_digest"],
            "template_sha256": suite.case_config.template_sha256,
            "benchmark_preflight_sha256": manifest["preflight"]["receipt_sha256"],
            "cores_per_case": variant.cores_per_case,
        }
        progress_reporter = _benchmark_progress_reporter(manifest, directory=directory, suite=suite, unit=unit, job_id=job_id)
        progress_reporter.update(phase="preparing", force=True)
        try:
            prepared = preparation_service.prepare_case_work_directory(
                suite.case_config,
                representative.case_index,
                storage_root=storage,
                work_root=work_root,
                raw_case_directory=_benchmark_input_directory(directory, representative),
            )
            actual_proof = _proof_payload(
                suite,
                representative,
                prepared,
                canonical_input_preparation_seconds=float(proof["canonical_input_preparation_seconds"]),
            )
            _validate_materialized_case_proof(actual_proof, proof)
            result = runtime_service.execute_prepared_case(
                suite.case_config,
                prepared,
                cores_per_case=variant.cores_per_case,
                worker_slot=unit.case_position - 1,
                scheduler_kind="slurm",
                allocated_node=node_name,
                progress_reporter=progress_reporter,
            )
            publication_start = time.monotonic()
            progress_reporter.update(phase="validating", force=True)
            hdf5_identity = storage_service.validate_case_hdf5(
                result.canonical_case.path,
                expected_profile=profiles.TRANSIENT_DRYING_PROFILE,
            )
            publication_seconds = time.monotonic() - publication_start
            _validate_hdf5_scientific_identity(hdf5_identity, proof)
            progress_reporter.update(phase="publishing", force=True)
            command_np = result.command[result.command.index("-np") + 1]
            completion_time = _utc_now()
            success = {
                **identity,
                "status": "success",
                "recorded_at": completion_time,
                "resource": {
                    "node": node_name,
                    "cpu_model": cpu_model,
                    "partition": os.environ.get("SLURM_JOB_PARTITION", suite.partition),
                    "requested_cpus": variant.cores_per_case,
                    "allocated_cpus": int(allocated_cpus),
                    "comsol_np": int(command_np),
                    "slurm_job_id": job_id,
                    "slurm_step_id": step_id,
                    "peak_memory_bytes": _benchmark_peak_child_memory_bytes(),
                    "peak_scratch_bytes": int(result.timing["scratch_peak_bytes"]),
                },
                "scheduler_timing": _scheduler_timing(submit_time=submit_time, completion_time=completion_time),
                "worker_interval": {
                    "started_at": worker_started_at,
                    "ended_at": completion_time,
                    "slurm_job_start_time": start_time_evidence,
                },
                "solver_interval": {
                    "started_at": result.timing["started_at"],
                    "ended_at": result.timing["ended_at"],
                },
                "timings_seconds": {
                    "scheduler_queue_seconds": None,
                    "license_wait_seconds": float(license_evidence["license_wait_seconds"]),
                    "license_probe_seconds": float(license_evidence["license_probe_seconds"]),
                    "canonical_input_preparation_seconds": float(proof["canonical_input_preparation_seconds"]),
                    "comsol_process_seconds": float(result.timing["comsol_process_seconds"]),
                    "export_conversion_seconds": float(result.timing["export_conversion_seconds"]),
                    "publication_seconds": publication_seconds,
                    "total_controller_elapsed_seconds": _total_controller_elapsed_seconds(submit_time, completion_time),
                },
                "license": license_evidence,
                "solver_log": _bounded_solver_log(result.solver_log),
                "hdf5": {
                    "sha256": common.serialization.file_sha256(result.canonical_case.path),
                    "size_bytes": result.canonical_case.path.stat().st_size,
                    "identity": hdf5_identity,
                    "retained_as_scientific_case": False,
                },
            }
            source_service.validate_admitted_source_before_publication()
            _write_immutable_json(attempt_path, success, label="benchmark attempt")
            _write_immutable_json(success_path, success, label="benchmark work-unit success")
            progress_reporter.update(phase="completed", terminal=True)
        except BaseException as error:
            progress_reporter.update(phase="failed", terminal=True)
            completion_time = _utc_now()
            failure = {
                **identity,
                "status": "failed",
                "recorded_at": completion_time,
                "resource": {
                    "node": node_name,
                    "cpu_model": cpu_model,
                    "partition": os.environ.get("SLURM_JOB_PARTITION", suite.partition),
                    "requested_cpus": variant.cores_per_case,
                    "allocated_cpus": int(allocated_cpus),
                    "comsol_np": variant.cores_per_case,
                    "slurm_job_id": job_id,
                    "slurm_step_id": step_id,
                    "peak_memory_bytes": _benchmark_peak_child_memory_bytes(),
                    "peak_scratch_bytes": (0 if prepared is None else _benchmark_scratch_bytes(prepared.work_directory)),
                },
                "scheduler_timing": _scheduler_timing(submit_time=submit_time, completion_time=completion_time),
                "worker_interval": {
                    "started_at": worker_started_at,
                    "ended_at": completion_time,
                    "slurm_job_start_time": start_time_evidence,
                },
                "timings_seconds": {
                    "scheduler_queue_seconds": None,
                    "license_wait_seconds": float(license_evidence["license_wait_seconds"]),
                    "license_probe_seconds": float(license_evidence["license_probe_seconds"]),
                    "canonical_input_preparation_seconds": float(proof["canonical_input_preparation_seconds"]),
                    "comsol_process_seconds": None,
                    "export_conversion_seconds": None,
                    "publication_seconds": 0.0,
                    "total_controller_elapsed_seconds": _total_controller_elapsed_seconds(submit_time, completion_time),
                },
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            if not attempt_path.exists():
                _write_immutable_json(attempt_path, failure, label="benchmark failed attempt")
            raise
        finally:
            if prepared is not None:
                _cleanup_prepared(prepared, storage=storage)
        return success


def _load_case_proofs(
    run_id: str,
    suite: benchmark_config.CoreBenchmarkSuite,
    *,
    storage_root: Path | str,
) -> dict[str, dict[str, Any]]:
    """Load both canonical proofs keyed by exact representative-case role."""
    return {
        representative.case_role: _load_case_proof(
            run_id,
            suite,
            case_position,
            storage_root=storage_root,
        )
        for case_position, representative in enumerate(
            suite.representative_cases,
            start=1,
        )
    }


def manifest_case_roles(
    manifest: Mapping[str, Any],
) -> tuple[benchmark_config.CoreBenchmarkRepresentativeCase, ...]:
    """Return minimal typed case roles from an already-admitted run manifest."""
    values = manifest["representative_cases"]
    if not isinstance(values, list):
        message = "Benchmark manifest representative_cases is malformed."
        raise TypeError(message)
    return tuple(
        benchmark_config.CoreBenchmarkRepresentativeCase(
            case_role=str(value["case_role"]),
            case_index=int(value["case_index"]),
        )
        for value in values
        if isinstance(value, dict)
    )


def _scheduler_command_evidence(command: Sequence[str]) -> dict[str, str | None]:
    """Run one optional fixed scheduler query and retain failure evidence."""
    try:
        result = subprocess.run(  # noqa: S603 -- fixed Slurm argv and numeric IDs
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {"output": "", "error": str(error)}
    return {
        "output": result.stdout.strip(),
        "error": (None if result.returncode == 0 else result.stderr.strip() or f"exit status {result.returncode}"),
    }


def _scheduler_evidence(job_ids: Sequence[str]) -> dict[str, Any]:
    """Return optional current Slurm queue and accounting evidence."""
    if not job_ids:
        return {
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "", "error": None},
        }
    selection = ",".join(job_ids)
    commands = {
        "squeue": [
            "squeue",
            "--noheader",
            f"--jobs={selection}",
            "--format=%i|%T|%R|%N|%M",
        ],
        "sacct": [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            selection,
            f"--format={cluster_service.STEP_ACCOUNTING_FORMAT}",
        ],
    }
    return {name: _scheduler_command_evidence(command) for name, command in commands.items()}


def _active_benchmark_jobs(
    manifest: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return exact pending and running job IDs owned by one benchmark."""
    queue = benchmark_config.require_mapping(scheduler["squeue"], label="benchmark squeue evidence")
    if queue.get("error") is not None:
        message = f"Could not query active core benchmark jobs: {queue['error']}"
        raise RuntimeError(message)
    owned = {str(value) for value in manifest["measured_job_ids"]}
    pending: list[str] = []
    running: list[str] = []
    for line in str(queue.get("output", "")).splitlines():
        try:
            job_id, state, _reason = line.split("|", maxsplit=2)
        except ValueError as error:
            message = f"Malformed active benchmark scheduler record: {line!r}."
            raise ValueError(message) from error
        if job_id not in owned or _JOB_ID_PATTERN.fullmatch(job_id) is None:
            message = f"Scheduler returned an unowned benchmark job: {job_id!r}."
            raise ValueError(message)
        if state.strip().upper() in {"PENDING", "CONFIGURING"}:
            pending.append(job_id)
        else:
            running.append(job_id)
    return pending, running


def _active_job_ids_without_scheduler(manifest: Mapping[str, Any], directory: Path) -> frozenset[str]:
    """Approximate liveness without a scheduler query: submitted node jobs whose worker has not written node_exit.json."""
    return frozenset(
        str(record["job_id"])
        for record in manifest["node_jobs"]
        if record["status"] == "submitted" and node_service.read_node_record(_node_directory(directory, record), "exit") is None
    )


def _measurement_input(
    manifest: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    directory: Path,
    sweep: benchmark_config.CoreBenchmarkSweep,
    variant_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the JSON evidence of one measured sweep variant: node, step receipts, step accounting, and records."""
    holder = next(record for record in manifest["node_jobs"] if record["token"] == variant_state["token"])
    node_record = node_service.read_node_record(_node_directory(directory, holder), "node") or {}
    exit_record = node_service.read_node_record(_node_directory(directory, holder), "exit") or {}
    variant = suite.variant(str(variant_state["variant_id"]))
    return {
        "sweep_id": sweep.sweep_id,
        "order": sweep.order,
        "variant_id": variant.variant_id,
        "cores_per_case": variant.cores_per_case,
        "variant_attempt": variant_state["attempt"],
        "node_job_id": holder["job_id"],
        "node": node_record.get("host"),
        "cpu_model": node_record.get("cpu_model"),
        "foreign_jobs_at_start": node_record.get("foreign_jobs_at_start"),
        "foreign_jobs_at_end": exit_record.get("foreign_jobs_at_end"),
        "units": [
            {
                "work_unit_id": _unit_id(suite, item["unit"]),
                "case_role": suite.representative_case(item["unit"].case_position).case_role,
                "outcome": item["outcome"],
                "started_at": None if item["started"] is None else item["started"]["started_at"],
                "ended_at": None if item["finished"] is None or item["outcome"] == "skipped" else item["finished"]["ended_at"],
                "step_exit_code": None if item["finished"] is None else item["finished"]["exit_code"],
                "step_max_rss_bytes": None if item["step"] is None else item["step"]["max_rss_bytes"],
                "record": None if item["record"]["status"] == "pending" else item["record"],
            }
            for item in variant_state["units"]
        ],
    }


def _measurements(
    manifest: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    directory: Path,
    active_job_ids: frozenset[str],
    step_rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the measured sweep variants in sweep and sweep-order sequence."""
    result: list[dict[str, Any]] = []
    for sweep in suite.sweeps:
        for variant in suite.sweep_variants(sweep):
            state = _variant_state(manifest, suite, directory, sweep, variant, active_job_ids, step_rows)
            if state["state"] == "measured":
                result.append(_measurement_input(manifest, suite, directory, sweep, state))
    return result


def _validate_measurements_against_proofs(
    measurements: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind every terminal work-unit record to its exact canonical input proof."""
    if set(proofs) != {item.case_role for item in manifest_case_roles(manifest)}:
        message = "Benchmark canonical proof roles do not match the run manifest."
        raise ValueError(message)
    for measurement in measurements:
        for unit in measurement["units"]:
            record = unit["record"]
            if record is None:
                continue
            proof = proofs.get(str(record.get("case_role")))
            if proof is None:
                message = f"Benchmark result has an unknown representative case role: {record.get('case_role')!r}."
                raise ValueError(message)
            expected = {
                "benchmark_run_id": run_id,
                "git_commit": manifest["git_commit"],
                "case_input_id": proof["case_input_id"],
                "simulation_case_id": proof["simulation_case_id"],
                "scientific_config_digest": proof["scientific_config_digest"],
                "template_sha256": proof["template"]["sha256"],
                "benchmark_preflight_sha256": manifest["preflight"]["receipt_sha256"],
            }
            if any(record.get(key) != value for key, value in expected.items()):
                message = f"Benchmark result is not bound to the canonical proof: {record.get('work_unit_id')!r}."
                raise ValueError(message)


def _unit_view(
    manifest: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    directory: Path,
    unit: _WorkUnit,
    *,
    outcome: str,
    started: Mapping[str, Any] | None,
    holder_active: bool,
    job_id: str | None,
) -> dict[str, Any]:
    """Return one presentation view of a dispatched or planned work unit."""
    if outcome == "success":
        state = "successful"
    elif outcome in {"failed", "failed_without_evidence"}:
        state = "failed"
    elif outcome == "skipped":
        state = "skipped"
    elif job_id is None:
        state = "never_started"
    elif holder_active and started is not None:
        state = "running"
    elif holder_active:
        state = "scheduler_pending"
    else:
        state = "interrupted"
    runtime: dict[str, Any] = {"availability": "unavailable", "reason": "no_job", "age_seconds": None, "stale": None}
    if job_id is not None and state == "running":
        runtime = _load_benchmark_runtime_progress(directory, _benchmark_progress_identity(manifest, suite=suite, unit=unit, job_id=job_id))
    return {
        "work_unit_id": _unit_id(suite, unit),
        "sweep_id": unit.sweep.sweep_id,
        "variant_id": unit.variant.variant_id,
        "cores_per_case": unit.variant.cores_per_case,
        "variant_attempt": unit.attempt,
        "case_role": suite.representative_case(unit.case_position).case_role,
        "state": state,
        "latest_job_id": job_id,
        "node": None,
        "runtime_progress": runtime,
    }


def _terminal_measurements(
    run_id: str,
    *,
    storage_root: Path | str,
    manifest: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    capture_accounting: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return every measurement of a finished benchmark and the case proofs, or refuse when sweeps are incomplete.

    Step accounting comes from the snapshot persisted at finalization, so later validation does not depend on live sacct.
    """
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    proofs = _load_case_proofs(run_id, suite, storage_root=storage_root)
    step_rows = _terminal_step_rows(directory, manifest, capture=capture_accounting)
    measurements = _measurements(manifest, suite, directory, frozenset(), step_rows)
    if len(measurements) != len(suite.sweeps) * len(suite.variants):
        message = f"Core benchmark {run_id!r} has {len(measurements)} of {len(suite.sweeps) * len(suite.variants)} sweep variant measurements."
        raise RuntimeError(message)
    _validate_measurements_against_proofs(measurements, run_id=run_id, manifest=manifest, proofs=proofs)
    return measurements, proofs


def cancel_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
    force: bool = False,
) -> dict[str, Any]:
    """Persist and request cancellation of the benchmark's own node jobs (cancel marker, then Slurm signals).

    An unresolved submission is recovered first; a node job whose submission is still unresolved gets the cancel marker,
    so its worker stops before launching anything.
    """
    if not isinstance(force, bool):
        message = "force must be boolean."
        raise TypeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    directory = manifest_path.parent
    lock_path = directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        manifest, _suite = load_core_benchmark_manifest(run_id, storage_root=storage)
        _recover_node_job_intent(manifest, directory=directory, manifest_path=manifest_path)
        job_ids = [str(value) for value in manifest["measured_job_ids"]]
        scheduler = _scheduler_evidence(job_ids)
        pending_ids, running_ids = _active_benchmark_jobs(manifest, scheduler)
        manifest["state"] = "force_cancel_requested" if force else "cancel_requested"
        _persist_manifest(manifest_path, manifest)
        for record in manifest["node_jobs"]:
            if record["status"] == "submitting" or (record["status"] == "submitted" and str(record["job_id"]) in {*pending_ids, *running_ids}):
                node_service.publish_node_cancel(_node_directory(directory, record))
        commands: list[list[str]] = []
        if force and (pending_ids or running_ids):
            commands.append(["scancel", "--signal=KILL", "--full", *pending_ids, *running_ids])
        elif not force:
            if pending_ids:
                commands.append(["scancel", *pending_ids])
            if running_ids:
                commands.append(["scancel", "--signal=TERM", "--batch", *running_ids])
        receipt_path = directory / "cancellations.json"
        attempts: list[dict[str, Any]] = []
        if receipt_path.exists():
            receipt = _load_json(receipt_path, label="benchmark cancellation receipt")
            if (
                receipt.get("schema_kind") != BENCHMARK_CANCELLATION_SCHEMA_KIND
                or receipt.get("schema_version") != 1
                or receipt.get("benchmark_run_id") != run_id
                or not isinstance(receipt.get("attempts"), list)
            ):
                message = f"Benchmark cancellation receipt is malformed: {receipt_path}"
                raise ValueError(message)
            attempts = list(receipt["attempts"])
        attempt: dict[str, Any] = {
            "recorded_at": _utc_now(),
            "mode": "force" if force else "graceful",
            "pending_job_ids": pending_ids,
            "running_job_ids": running_ids,
            "unresolved_submission": manifest["node_job_intent"],
            "commands": [],
        }
        receipt = {
            "schema_kind": BENCHMARK_CANCELLATION_SCHEMA_KIND,
            "schema_version": 1,
            "benchmark_run_id": run_id,
            "attempts": [*attempts, attempt],
        }
        common.serialization.atomic_write_json(receipt_path, receipt)
        failures: list[str] = []
        for command in commands:
            result = subprocess.run(  # noqa: S603 -- validated persisted numeric Slurm IDs
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            attempt["commands"].append(
                {"command": command, "exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
            )
            if result.returncode != 0:
                failures.append(result.stderr.strip() or f"{command[0]} exit {result.returncode}")
        common.serialization.atomic_write_json(receipt_path, receipt)
        if failures:
            message = f"Slurm benchmark cancellation failed after its request was persisted: {failures}"
            raise RuntimeError(message)
        return receipt


def _benchmark_progress_identity(
    manifest: Mapping[str, Any], *, suite: benchmark_config.CoreBenchmarkSuite, unit: _WorkUnit, job_id: str, token: str | None = None
) -> dict[str, Any]:
    """Bind common runtime progress to one dispatched work unit of one node job."""
    unit_id = _unit_id(suite, unit)
    _dispatched_ticket(manifest, job_id=job_id, unit_id=unit_id, token=token)
    return {
        "schema_kind": "generation_campaign_runtime_progress",
        "schema_version": 1,
        "benchmark_run_id": manifest["benchmark_run_id"],
        "variant_id": unit.variant.variant_id,
        "cores_per_case": unit.variant.cores_per_case,
        "case_role": suite.representative_case(unit.case_position).case_role,
        "work_unit_id": unit_id,
        "slurm_job_id": job_id,
    }


def _benchmark_progress_path(directory: Path, unit_id: str, *, create: bool) -> Path:
    """Return a bounded symlink-free benchmark progress receipt path keyed by work unit."""
    progress = directory / "progress"
    if progress.is_symlink() or (progress.exists() and not progress.is_dir()):
        message = "Benchmark progress directory is unsafe."
        raise ValueError(message)
    if create:
        progress.mkdir(mode=0o700, exist_ok=True)
    path = progress / f"{unit_id}.json"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        message = "Benchmark progress receipt is unsafe."
        raise ValueError(message)
    return path


def _benchmark_progress_reporter(
    manifest: Mapping[str, Any],
    *,
    directory: Path,
    suite: benchmark_config.CoreBenchmarkSuite,
    unit: _WorkUnit,
    job_id: str,
) -> progress_service.RuntimeProgressReporter:
    """Create the shared incremental reporter for one dispatched work unit."""
    identity = _benchmark_progress_identity(manifest, suite=suite, unit=unit, job_id=job_id, token=os.environ.get("GENERATION_NODE_TOKEN"))
    path = _benchmark_progress_path(directory, _unit_id(suite, unit), create=True)
    return progress_service.create_bound_runtime_progress_reporter(
        identity,
        path,
        hostname=os.environ.get("SLURMD_NODENAME", socket.gethostname()),
    )


def _load_benchmark_runtime_progress(directory: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Load common validated reporter evidence without affecting execution."""
    try:
        path = _benchmark_progress_path(directory, str(identity["work_unit_id"]), create=False)
    except (OSError, TypeError, ValueError):
        return {"availability": "unavailable", "reason": "invalid_or_unavailable", "age_seconds": None, "stale": None}
    return progress_service.load_bound_runtime_progress(identity, path)


def core_benchmark_status(
    run_id: str,
    *,
    storage_root: Path | str,
    query_scheduler: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Reconstruct sweep-, variant-, and work-unit state from evidence, receipts, and node-job liveness."""
    del now
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    job_ids = [str(value) for value in manifest["measured_job_ids"]]
    scheduler = _scheduler_evidence(job_ids) if query_scheduler else {"squeue": {"output": "", "error": None}, "sacct": {"output": "", "error": None}}
    if query_scheduler and scheduler["squeue"]["error"] is not None:
        message = f"Could not query active core benchmark node jobs: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    active_job_ids = _active_benchmark_job_ids(scheduler) if query_scheduler else _active_job_ids_without_scheduler(manifest, directory)
    step_rows = _step_rows(scheduler) if query_scheduler else None
    work_units: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    for sweep in suite.sweeps:
        sweep_state = _sweep_state(manifest, suite, directory, sweep, active_job_ids, step_rows)
        for variant_state in sweep_state["variants"]:
            variant = suite.variant(str(variant_state["variant_id"]))
            if variant_state["attempt"] is None:
                work_units.extend(
                    _unit_view(manifest, suite, directory, unit, outcome="pending", started=None, holder_active=False, job_id=None)
                    for unit in _variant_units(suite, sweep, variant, 1)
                )
                continue
            holder = next(record for record in manifest["node_jobs"] if record["token"] == variant_state["token"])
            holder_active = _node_job_active(holder, active_job_ids)
            work_units.extend(
                _unit_view(
                    manifest,
                    suite,
                    directory,
                    item["unit"],
                    outcome=item["outcome"],
                    started=item["started"],
                    holder_active=holder_active,
                    job_id=None if holder["job_id"] is None else str(holder["job_id"]),
                )
                for item in variant_state["units"]
            )
        current = next((item for item in sweep_state["variants"] if item["state"] != "measured"), None)
        sweeps.append(
            {
                "sweep_id": sweep.sweep_id,
                "order": sweep.order,
                "state": sweep_state["state"],
                "node_job_ids": [str(record["job_id"]) for record in _sweep_node_jobs(manifest, sweep) if record["job_id"] is not None],
                "current_variant_id": None if current is None else current["variant_id"],
                "current_cores_per_case": None if current is None else suite.variant(str(current["variant_id"])).cores_per_case,
                "measured_variants": sum(item["state"] == "measured" for item in sweep_state["variants"]),
                "total_variants": len(sweep_state["variants"]),
            }
        )
    categories = ("successful", "running", "scheduler_pending", "interrupted", "never_started", "failed", "skipped")
    counts = {name: sum(item["state"] == name for item in work_units) for name in categories}
    proof_paths = tuple(_canonical_case_proof_path(directory, representative) for representative in suite.representative_cases)
    sweep_states = [item["state"] for item in sweeps]
    if manifest["state"] in {"cancel_requested", "force_cancel_requested"}:
        state = "cancelled"
    elif not all(path.is_file() for path in proof_paths):
        state = "inputs_ready"
    elif all(item == "complete" for item in sweep_states):
        state = "complete"
    elif any(item in {"running", "ready_to_submit"} for item in sweep_states):
        state = "running"
    else:
        state = "variant_failed"
    measurements = _measurements(manifest, suite, directory, active_job_ids, step_rows)
    partial = None
    if measurements and state != "complete":
        partial = benchmark_report.summarize_core_benchmark_results(suite, measurements)
        partial["provisional"] = True
    final_summary: dict[str, Any] | None = None
    if state == "complete" and (directory / "summary.json").is_file():
        summary = load_core_benchmark_summary(run_id, storage_root=storage_root)
        final_summary = {
            "path": str(directory / "summary.md"),
            "markdown": _load_validated_core_benchmark_markdown(directory / "summary.md", summary),
            "recommended_cores_per_case": summary.get("recommended_cores_per_case"),
            "recommended_throughput_cases_per_node_hour": summary.get("recommended_throughput_cases_per_node_hour"),
        }
    progress = [
        str(runtime["updated_at"])
        for unit in work_units
        if isinstance((runtime := unit.get("runtime_progress")), dict)
        and runtime.get("availability") == "available"
        and isinstance(runtime.get("updated_at"), str)
    ]
    return {
        "schema_kind": "generation_run_status",
        "schema_version": 1,
        "run_kind": "benchmark",
        "suite_name": suite.suite_name,
        "benchmark_run_id": run_id,
        "state": state,
        "successful_work_units": counts["successful"],
        "running_work_units": counts["running"],
        "scheduler_pending_work_units": counts["scheduler_pending"],
        "interrupted_work_units": counts["interrupted"],
        "never_started_work_units": counts["never_started"],
        "failed_work_units": counts["failed"],
        "skipped_work_units": counts["skipped"],
        "total_work_units": len(work_units),
        "work_units": work_units,
        "sweeps": sweeps,
        "scheduler": scheduler,
        "last_progress_timestamp": max(progress, default=None),
        "eta": "unavailable",
        "partial_evaluation": partial,
        "final_summary": final_summary,
        "resume_action": (
            "none; terminal benchmark evidence is complete" if state == "complete" else f"rerun ./scripts/generation run {manifest['suite_config']}"
        ),
    }


def _archive_summary(directory: Path) -> None:
    """Preserve an earlier summary revision before a retry-derived replacement."""
    paths = [directory / name for name in ("runs.csv", "summary.json", "summary.md")]
    if not any(path.exists() for path in paths):
        return
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        message = f"Benchmark summary revision is incomplete or unsafe: {directory}"
        raise ValueError(message)
    history = directory / "summary_history"
    history.mkdir(exist_ok=True)
    revision = len(tuple(history.iterdir())) + 1
    destination = history / f"revision-{revision:04d}"
    destination.mkdir()
    for path in paths:
        shutil.copy2(path, destination / path.name)


def _proof_identities(
    suite: benchmark_config.CoreBenchmarkSuite,
    proofs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact ordered identities for both canonical benchmark inputs."""
    return [
        {
            "case_role": representative.case_role,
            "case_input_id": proofs[representative.case_role]["case_input_id"],
            "simulation_case_id": proofs[representative.case_role]["simulation_case_id"],
            "case_payload_sha256": proofs[representative.case_role]["case_payload_sha256"],
        }
        for representative in suite.representative_cases
    ]


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    suite: benchmark_config.CoreBenchmarkSuite,
    manifest: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
) -> None:
    """Bind one summary revision to the inputs, the frozen source, and all terminal measurement evidence."""
    expected_identity = {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
        "template_sha256": suite.case_config.template_sha256,
        "representative_case_identities": _proof_identities(suite, proofs),
        "preflight": manifest["preflight"],
        "result_set_digest": common.serialization.canonical_json_sha256(list(measurements)),
    }
    if any(summary.get(key) != value for key, value in expected_identity.items()):
        message = "Core benchmark summary is not bound to its terminal source evidence."
        raise ValueError(message)
    generated_at = summary.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at or any(character in generated_at for character in "\r\n\t"):
        message = "Core benchmark summary timestamp is malformed."
        raise ValueError(message)


def _summary_metrics_match(
    summary: Mapping[str, Any],
    suite: benchmark_config.CoreBenchmarkSuite,
    measurements: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether derived metrics match a recomputation from the measurements."""
    recomputed = benchmark_report.summarize_core_benchmark_results(suite, measurements)
    return all(summary.get(key) == recomputed[key] for key in benchmark_report.SUMMARY_METRIC_FIELDS)


def _validate_summary_payload(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    suite: benchmark_config.CoreBenchmarkSuite,
    manifest: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
) -> None:
    """Bind one terminal summary to its inputs and recomputed metrics."""
    _validate_summary_identity(summary, run_id=run_id, suite=suite, manifest=manifest, proofs=proofs, measurements=measurements)
    if not _summary_metrics_match(summary, suite, measurements):
        message = "Core benchmark summary metrics are stale or inconsistent."
        raise ValueError(message)


def _validate_or_repair_summary_outputs(
    directory: Path,
    summary: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    *,
    repair_missing: bool,
) -> None:
    """Validate deterministic summary views and create only missing retry output."""
    expected = {
        "runs.csv": benchmark_report.results_csv(measurements),
        "summary.md": benchmark_report.core_benchmark_markdown(summary),
    }
    for name, content in expected.items():
        path = directory / name
        if not path.exists() and repair_missing:
            common.serialization.atomic_write_text(path, content)
            continue
        if not path.is_file() or path.is_symlink() or path.read_text(encoding="utf-8") != content:
            message = f"Core benchmark derived output is stale or unsafe: {path}"
            raise ValueError(message)


def finalize_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Publish aggregate evidence only after every sweep measured every variant and no node job is active."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    scheduler = _scheduler_evidence(list(manifest["measured_job_ids"]))
    if scheduler["squeue"]["error"] is not None:
        message = f"Could not verify terminal benchmark node jobs before finalization: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    if scheduler["squeue"]["output"]:
        message = "Core benchmark cannot finalize while its node jobs remain active."
        raise RuntimeError(message)
    measurements, proofs = _terminal_measurements(run_id, storage_root=storage_root, manifest=manifest, suite=suite, capture_accounting=True)
    result_set_digest = common.serialization.canonical_json_sha256(measurements)
    current = directory / "summary.json"
    manifest_path = _manifest_path(run_id, storage_root=storage_root)
    if current.exists():
        existing = _load_json(current, label="core benchmark summary")
        if existing.get("result_set_digest") == result_set_digest:
            _validate_summary_payload(existing, run_id=run_id, suite=suite, manifest=manifest, proofs=proofs, measurements=measurements)
            _validate_or_repair_summary_outputs(directory, existing, measurements, repair_missing=True)
            manifest["state"] = "complete"
            _persist_manifest(manifest_path, manifest)
            return existing
    summary = benchmark_report.summarize_core_benchmark_results(suite, measurements)
    summary.update(
        {
            "benchmark_run_id": run_id,
            "git_commit": manifest["git_commit"],
            "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
            "template_sha256": suite.case_config.template_sha256,
            "representative_case_identities": _proof_identities(suite, proofs),
            "preflight": manifest["preflight"],
            "result_set_digest": result_set_digest,
            "generated_at": _utc_now(),
        }
    )
    _archive_summary(directory)
    common.serialization.atomic_write_text(directory / "runs.csv", benchmark_report.results_csv(measurements))
    common.serialization.atomic_write_json(current, summary)
    common.serialization.atomic_write_text(directory / "summary.md", benchmark_report.core_benchmark_markdown(summary))
    manifest["state"] = "complete"
    _persist_manifest(manifest_path, manifest)
    return summary


def load_core_benchmark_summary(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate one aggregate benchmark summary."""
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    summary = _load_json(directory / "summary.json", label="core benchmark summary")
    if (
        summary.get("schema_kind") != BENCHMARK_SUMMARY_SCHEMA_KIND
        or summary.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or summary.get("benchmark_run_id") != run_id
    ):
        message = f"Core benchmark summary schema is invalid: {run_id}"
        raise ValueError(message)
    return summary


def _load_validated_core_benchmark_markdown(
    path: Path,
    summary: Mapping[str, Any],
) -> str:
    """Read one safe persisted Markdown summary and verify its canonical bytes."""
    if not path.is_file() or path.is_symlink():
        message = f"Core benchmark Markdown summary is missing or unsafe: {path}"
        raise ValueError(message)
    markdown = path.read_text(encoding="utf-8")
    if markdown != benchmark_report.core_benchmark_markdown(summary):
        message = f"Core benchmark Markdown summary is inconsistent: {path}"
        raise ValueError(message)
    return markdown


def load_core_benchmark_markdown(
    run_id: str,
    *,
    storage_root: Path | str,
) -> str:
    """Load persisted terminal Markdown after validating it against the summary."""
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    summary = load_core_benchmark_summary(run_id, storage_root=storage_root)
    return _load_validated_core_benchmark_markdown(
        directory / "summary.md",
        summary,
    )


def _directory_inventory(directory: Path) -> dict[str, Any]:
    """Return a symlink-free deterministic file inventory."""
    if not directory.is_dir() or directory.is_symlink():
        message = f"Benchmark evidence directory is missing or unsafe: {directory}"
        raise ValueError(message)
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            message = f"Benchmark evidence contains an unsafe symlink: {path}"
            raise ValueError(message)
        if path.is_file():
            files.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": common.serialization.file_sha256(path),
                }
            )
    return {
        "file_count": len(files),
        "size_bytes": sum(int(record["size_bytes"]) for record in files),
        "files": files,
        "inventory_sha256": common.serialization.canonical_json_sha256(files),
    }


def validate_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Validate the case proofs, every sweep variant measurement, the summaries, and isolation."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    measurements, proofs = _terminal_measurements(run_id, storage_root=storage_root, manifest=manifest, suite=suite)
    if manifest["state"] != "complete":
        message = f"Core benchmark manifest is not complete: {manifest['state']!r}."
        raise ValueError(message)
    summary = load_core_benchmark_summary(run_id, storage_root=storage_root)
    _validate_summary_payload(summary, run_id=run_id, suite=suite, manifest=manifest, proofs=proofs, measurements=measurements)
    _validate_or_repair_summary_outputs(directory, summary, measurements, repair_missing=False)
    expected_root = (common.paths.get_generation_performance_benchmark_root(storage_root=storage_root) / BENCHMARK_FAMILY).resolve()
    if directory.parent.resolve() != expected_root:
        message = f"Core benchmark evidence escaped its dedicated metadata namespace: {directory}"
        raise ValueError(message)
    inventory = _directory_inventory(directory)
    return {
        "status": "complete",
        "benchmark_run_id": run_id,
        "suite_digest": suite.suite_digest,
        "representative_case_identities": _proof_identities(suite, proofs),
        "measurements": len(measurements),
        "required_measurements": len(suite.sweeps) * len(suite.variants),
        "namespace": str(directory),
        "dataset_membership": "none",
        "inventory": inventory,
    }


def core_benchmark_source_status(
    run_id: str,
    *,
    storage_root: Path | str,
    query_scheduler: bool = True,
) -> dict[str, Any]:
    """Report the common CPU source-storage state for one benchmark."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    source = core_benchmark_directory(run_id, storage_root=storage)
    if not source.is_dir() or source.is_symlink():
        message = f"Benchmark source is missing without cleanup evidence: {source}"
        raise FileNotFoundError(message)
    status = core_benchmark_status(
        run_id,
        storage_root=storage,
        query_scheduler=query_scheduler,
    )
    inventory = _directory_inventory(source)
    active = bool(status["running_work_units"] or status["scheduler_pending_work_units"])
    return {
        "schema_kind": "generation_run_source_status",
        "schema_version": 1,
        "run_kind": "benchmark",
        "run_id": run_id,
        "run_state": status["state"],
        "source_state": "retained" if status["state"] == "complete" else "active",
        "reclaimable_bytes": inventory["size_bytes"],
        "cleanup_eligibility": ("eligible" if status["state"] == "complete" and not active else "not_terminal_or_active"),
        "active_slurm": active,
    }
