"""
generation_controller.py

Coordinate Generation runs from authoritative service evidence.

Responsibilities:
  - Interpret immutable run plans and configured execution resources
  - Sequence campaign, benchmark, paired smoke, and completion lifecycles
  - Reconcile repeat invocations, monitor work, and gate publication
  - Report failure stages and preserve service-owned recovery evidence

Design principles:
  - Existing Generation services own identities, manifests, and scientific data
  - Every continuation is reconstructed from persisted service evidence
  - Collection and finalization follow exact service validation gates

This module does NOT:
  - Submit individual scientific cases or define solver behavior
  - Define dataset, completion, or publication persistence formats
  - Load environment modules or execute work outside Slurm
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol

from .runtime import generation_runtime_host as host_service


class GenerationControllerError(RuntimeError):
    """A failed controller transition with a stable process exit status."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        """Retain an actionable diagnostic and the public process exit status."""
        super().__init__(message)
        self.exit_code = exit_code


def _fail(message: str, *, exit_code: int = 1) -> NoReturn:
    """Raise one typed controller failure at any lifecycle stage."""
    raise GenerationControllerError(message, exit_code=exit_code)


class GenerationCommandPort(Protocol):
    """Execute one existing service operation at the admitted host boundary."""

    @property
    def repository(self) -> Path:
        """Return the admitted repository root."""
        ...

    @property
    def storage(self) -> Path:
        """Return the durable storage root."""
        ...

    @property
    def commit(self) -> str:
        """Return the admitted source commit."""
        ...

    def call(
        self,
        operation: str,
        *arguments: str,
        scheduled: bool | None = None,
        benchmark_preflight: bool = False,
        partition: str = "standard",
        wall_time: str = "02:00:00",
    ) -> str:
        """Execute one admitted Generation service operation."""
        ...

    def config_file(self, value: str) -> tuple[Path, str]:
        """Admit one repository-owned configuration path."""
        ...

    def comsol_version(self, module: str, executable: str) -> str:
        """Query native COMSOL version evidence."""
        ...

    def verify_execution(self) -> None:
        """Check native dependencies and the shared node launcher."""
        ...

    def verify_source(self) -> None:
        """Recheck the admitted clean source before paired publication."""
        ...


_SCHEDULED_OPERATIONS = frozenset(
    {
        "prepare-campaign-inputs",
        "resume-campaign",
        "build-campaign-datasets",
        "prepare-gpu-datasets",
        "prepare-all-workflow",
        "advance-campaign-completion",
        "build-campaign-completion-composite",
        "build-campaign-completion-lifecycle",
        "finalize-core-benchmark",
        "finalize-technical-smoke-evidence",
        "finalize-real-smoke",
        "repair-transferred-campaign",
        "repair-partial-campaign-publication",
        "record-pilot-source-inventory",
        "record-shared-pilot-staging",
        "prepare-pilot-check",
        "cleanup-pilot-staging",
        "campaign-transfer-authority",
        "validate-published-campaign",
        "validate-campaign-terminal",
        "validate-campaign-package-state",
        "validate-all-workflow",
        "validate-pilot-check",
        "validate-core-benchmark",
        "validate-real-smoke",
        "validate-campaign-completion-lifecycle",
    }
)
_RUN_ID = re.compile(r"[A-Za-z0-9._-]+__[0-9a-f]{16}\Z")
_COMPLETION_ID = re.compile(r"completion__[0-9a-f]{24}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_BATCH_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_REPLACEMENT_ID = re.compile(r"replacement__[0-9a-f]{24}\Z")
_MIN_CAMPAIGN_MONITOR_LINES = 3
_MONITOR_HEADER_FIELDS = 4
_SOURCE_MONITOR_FIELDS = 7
_PILOT_CLEANUP_FIELDS = 5
_PAIRED_SMOKE_CHILDREN = 2


def _payload(output: str, operation: str) -> dict[str, Any]:
    """Decode one JSON service result without lossy shell transport."""
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        _fail(f"Malformed {operation} response")
    if not isinstance(value, dict):
        _fail(f"Malformed {operation} response")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{label} must be an integer >= 1", exit_code=2)
    return value


def _identifier(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"Malformed {label}: {value}")
    return value


@dataclass(frozen=True, slots=True)
class RunPlan:
    """The service-resolved plan fields needed by the lifecycle controller."""

    kind: str
    identity: str
    config_path: str
    purpose: str
    profile: str
    children: tuple[RunPlan, ...]
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> RunPlan:
        """Read identities and ordered child dependencies from a service plan."""
        children = tuple(cls.from_payload(item) for item in value["children"])
        units = value.get("units", [])
        metadata = units[0]["metadata"] if units else {}
        return cls(
            kind=str(value["run_kind"]),
            identity=str(value["identity"]),
            config_path=str(value["config_path"]),
            purpose=str(metadata.get("campaign_purpose", "-")),
            profile=str(metadata.get("simulation_profile", "-")),
            children=children,
            payload=value,
        )


@dataclass(frozen=True, slots=True)
class ExecutionResources:
    """Validated execution fields from one campaign or benchmark config."""

    partition: str
    scheduler: str
    python_module: str
    comsol_module: str
    python_executable: str
    comsol_executable: str
    poll_seconds: int
    wall_time: str
    cores_per_node: int
    purpose: str = "-"

    @classmethod
    def campaign(cls, value: dict[str, Any]) -> ExecutionResources:
        """Project a validated campaign config onto host execution settings."""
        resources = value["execution_resources"]
        cluster = resources["cluster"]
        submission = resources["submission"]
        site = resources["site"]
        cores_per_node = _positive(cluster["cores_per_node"], "cores_per_node")
        cores_per_case = _positive(cluster["cores_per_case"], "cores_per_case")
        if cores_per_case > cores_per_node:
            _fail("cores_per_case exceeds site cores_per_node", exit_code=2)
        _positive(submission["max_admission_cases"], "max_admission_cases")
        max_running = submission["max_running_cases"]
        if max_running is not None:
            _positive(max_running, "max_running_cases")
        return cls(
            partition=str(site["partition"]),
            scheduler=str(site["scheduler"]),
            python_module=str(site["python_module"]),
            comsol_module=str(site["comsol_module"]),
            python_executable=str(site["python_executable"]),
            comsol_executable=str(site["comsol_executable"]),
            poll_seconds=_positive(submission["poll_interval_seconds"], "poll_interval_seconds"),
            wall_time=str(cluster.get("wall_time") or "02:00:00"),
            cores_per_node=cores_per_node,
            purpose=str(value["campaign_purpose"]),
        )

    @classmethod
    def benchmark(cls, value: dict[str, Any]) -> ExecutionResources:
        """Project a validated benchmark suite onto host execution settings."""
        site = value["resource_contract"]
        return cls(
            partition=str(site["partition"]),
            scheduler=str(site["scheduler"]),
            python_module=str(site["python_module"]),
            comsol_module=str(site["comsol_module"]),
            python_executable=str(site["python_executable"]),
            comsol_executable=str(site["comsol_executable"]),
            poll_seconds=_positive(site["poll_interval_seconds"], "poll_interval_seconds"),
            wall_time="02:00:00",
            cores_per_node=_positive(site["cores_per_node"], "cores_per_node"),
        )


@dataclass(slots=True)
class LeafResult:
    """Transient result of reconstructing and executing one leaf lifecycle."""

    run_id: str
    kind: str
    purpose: str
    result: str = "OK"
    partial: bool = False


class GenerationController:
    """Sequence existing Generation services using their persisted state."""

    def __init__(self, port: GenerationCommandPort, *, execute: bool = False) -> None:
        """Initialize transient orchestration state around a service port."""
        self.port = port
        self.execute = execute
        self.stage = "not_started"
        self.run_id = ""
        self.run_kind = ""
        self.config_argument = ""
        self.resources: ExecutionResources | None = None
        self.cpu_bytes_retained: int | None = None
        self.completion_owner_persisted = False
        self.completion_parent_run = ""
        self.pool_option: int | None = None
        self._interrupt_count = 0

    def call(
        self,
        operation: str,
        *arguments: str,
        scheduled: bool | None = None,
        benchmark_preflight: bool = False,
    ) -> str:
        """Use the node boundary for configured computational operations."""
        resources = self.resources
        if scheduled is None:
            scheduled = self.execute and operation in _SCHEDULED_OPERATIONS
        return self.port.call(
            operation,
            *arguments,
            scheduled=scheduled,
            benchmark_preflight=benchmark_preflight,
            partition=resources.partition if resources else "standard",
            wall_time=resources.wall_time if resources else "02:00:00",
        )

    def json(self, operation: str, *arguments: str, scheduled: bool | None = None) -> dict[str, Any]:
        """Call a JSON service command and decode its result."""
        return _payload(self.call(operation, *arguments, scheduled=scheduled), operation)

    def plan(self, config: str, *, allow_incomplete: bool = False) -> RunPlan:
        """Resolve a config through the existing immutable run-plan owner."""
        config_path, _ = self.port.config_file(config)
        arguments = [str(config_path), "--git-commit", self.port.commit]
        if allow_incomplete:
            arguments.append("--allow-incomplete")
        return RunPlan.from_payload(self.json("resolve-generation-run", *arguments, scheduled=False))

    def campaign_resources(self, config: str, *, executable: bool = True) -> ExecutionResources:
        """Read the configured campaign resource contract once in Python."""
        config_path, _ = self.port.config_file(config)
        arguments = [str(config_path)]
        if not executable:
            arguments.append("--allow-incomplete")
        resources = ExecutionResources.campaign(self.json("validate-config", *arguments, scheduled=False))
        self._validate_site(resources)
        self.resources = resources
        return resources

    def benchmark_resources(self, config: str) -> tuple[ExecutionResources, dict[str, Any]]:
        """Read the benchmark suite's resource and identity contract."""
        config_path, _ = self.port.config_file(config)
        inspection = self.json("inspect-core-benchmark", str(config_path), scheduled=False)
        resources = ExecutionResources.benchmark(inspection)
        self._validate_site(resources)
        self.resources = resources
        return resources, inspection

    def _validate_site(self, resources: ExecutionResources) -> None:
        if resources.scheduler != "slurm" or resources.partition not in {"standard", "long"}:
            _fail("Generation requires configured native Slurm CPU execution")

    def _storage(self) -> tuple[str, str]:
        return "--storage-root", str(self.port.storage)

    def _campaign_file(self, config: str) -> tuple[str, str]:
        path, relative = self.port.config_file(config)
        return str(path), relative

    def preflight(self, plan: RunPlan) -> str:
        """Check executable config and native solver without generating data."""
        target = plan.children[0] if plan.kind == "workflow" else plan
        if target.kind == "benchmark":
            resources, _ = self.benchmark_resources(target.config_path)
        elif target.kind == "campaign":
            resources = self.campaign_resources(target.config_path)
        else:
            _fail(f"Unsupported plan kind: {target.kind}", exit_code=2)
        self.port.verify_execution()
        version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
        return f"PREFLIGHT COMPLETE: plan={plan.identity} kind={plan.kind} host=shared-filesystem COMSOL={version}"

    def run(
        self, plan: RunPlan, config_argument: str, *, pool_size: int | None = None, parent_run_id: str | None = None
    ) -> LeafResult | tuple[LeafResult, ...]:
        """Execute or resume one immutable plan from service-owned evidence."""
        self.execute = True
        self.config_argument = config_argument
        self.run_kind = plan.kind
        self.pool_option = pool_size
        self.stage = "common plan resolution"
        print(f"[1/9] Run plan: kind={plan.kind} identity={plan.identity} config={plan.config_path}")
        try:
            result: LeafResult | tuple[LeafResult, ...]
            if plan.kind == "campaign":
                if pool_size:
                    result = self._run_with_completion(plan, pool_size, parent_run_id)
                else:
                    path, _ = self._campaign_file(plan.config_path)
                    parent = self._completion_parent(path)
                    persisted = parent["status"] == "compatible_partial" and parent.get("completion_status") is not None
                    if persisted:
                        high_water = _positive(parent.get("replacement_pool_size"), "persisted replacement pool high-water")
                        result = self._run_with_completion(plan, high_water, None, parent=parent)
                    else:
                        result = self._leaf(plan)
            elif plan.kind == "benchmark":
                result = self._leaf(plan)
            elif plan.kind == "workflow":
                result = self._workflow(plan)
            else:
                _fail(f"Unsupported Generation run kind: {plan.kind}", exit_code=2)
        except BaseException:
            self._record_failure()
            raise
        print(f"GENERATION COMPLETE: run_identity={plan.identity}")
        return result

    def _completion_valid(self, run_id: str, purpose: str) -> bool:
        try:
            self.call("validate-all-workflow", run_id, *self._storage())
            self.call("validate-campaign-package-state", run_id, *self._storage())
            if purpose == "pilot_check":
                self.call("validate-pilot-check", run_id, "--require-cleanup-complete", *self._storage())
        except host_service.GenerationCommandError:
            return False
        return True

    def _existing_campaign(self, plan: RunPlan, path: str) -> tuple[str, str]:
        run_id = _identifier(plan.identity, _RUN_ID, "campaign-run ID")
        purpose = plan.purpose
        if purpose == "technical_runtime_smoke":
            candidate = self.json("find-compatible-technical-smoke-run", path, *self._storage())
            status = candidate["status"]
            if status in {"compatible_complete", "compatible_repairable"}:
                run_id = _identifier(candidate["campaign_run_id"], _RUN_ID, "campaign-run ID")
                if status == "compatible_complete":
                    if not self._completion_valid(run_id, purpose):
                        _fail("Selected Technical Smoke source is not terminally valid")
                    return run_id, "complete"
                return run_id, "transfer_repair"
            if status != "missing":
                _fail(f"Unsupported Technical Smoke source state: {status}")
        if self._completion_valid(run_id, purpose):
            return run_id, "complete"
        if purpose not in {"pilot_check", "technical_runtime_smoke"}:
            source = self.json("find-compatible-campaign-source", path, *self._storage())
            if source["status"] == "compatible_complete":
                run_id = _identifier(source["campaign_run_id"], _RUN_ID, "campaign-run ID")
                _identifier(source["artifact_set_sha256"], _DIGEST, "artifact set digest")
                state = source["package_state"]
                if state == "complete" and self._completion_valid(run_id, purpose):
                    return run_id, "complete"
                if state == "extension_required":
                    return run_id, "package_only"
                _fail("Compatible source has invalid package state")
            if source["status"] != "missing":
                _fail("Unsupported compatible source state")
        return run_id, "running"

    def _leaf(self, plan: RunPlan, *, parent_retention: bool = False) -> LeafResult:
        self.run_kind = plan.kind
        self.stage = "existing state inspection"
        path, _ = self._campaign_file(plan.config_path)
        if plan.kind == "campaign":
            resources = self.campaign_resources(plan.config_path)
            run_id, state = self._existing_campaign(plan, path)
        elif plan.kind == "benchmark":
            resources, _ = self.benchmark_resources(plan.config_path)
            version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
            identity = self.json("resolve-core-benchmark-run", path, "--git-commit", self.port.commit, "--comsol-version-output", version)
            run_id = _identifier(identity["benchmark_run_id"], _RUN_ID, "benchmark-run ID")
            try:
                self.call("validate-core-benchmark", run_id, *self._storage())
            except host_service.GenerationCommandError:
                state = "running"
            else:
                state = "complete"
        else:
            _fail(f"Unsupported leaf kind: {plan.kind}", exit_code=2)
        self.port.verify_execution()
        self.run_id = run_id
        result = LeafResult(run_id=run_id, kind=plan.kind, purpose=plan.purpose)
        if state == "complete":
            result.result = "REUSED"
            print(f"[2/9] Existing state: REUSED run_id={run_id}")
            if plan.kind == "benchmark":
                print(self.call("core-benchmark-summary", run_id, "--format", "markdown", *self._storage()))
            return result
        print(f"[2/9] Existing state: {state} run_id={run_id}")
        if state in {"package_only", "transfer_repair"}:
            if state == "transfer_repair":
                self.stage = "repairable host transfer publication"
                self._collect_campaign(result)
            self.stage = "declared package continuation"
            self._build_packages(result)
            if parent_retention:
                self._prepare_parent_child(result)
            else:
                if state == "transfer_repair":
                    self._retention(result)
                self._validate_leaf(result)
            return result

        return self._fresh_leaf(plan, path, resources, result, parent_retention=parent_retention)

    def _fresh_leaf(
        self,
        plan: RunPlan,
        path: str,
        resources: ExecutionResources,
        result: LeafResult,
        *,
        parent_retention: bool,
    ) -> LeafResult:
        """Advance a fresh or resumable scientific leaf through all terminal gates."""
        run_id = result.run_id
        self.stage = "canonical input readiness"
        if plan.kind == "campaign":
            try:
                input_result = self.call("prepare-campaign-inputs", path, "--git-commit", self.port.commit, *self._storage())
            except host_service.GenerationCommandError as error:
                _fail(f"Canonical campaign input preparation failed before submission: {error}")
            if not input_result.startswith("canonical-inputs\t"):
                _fail("Malformed canonical campaign input result")
        else:
            materialized = self._benchmark_plan_command("materialize-core-benchmark-inputs", path, resources)
            if materialized["benchmark_run_id"] != run_id:
                _fail("Benchmark input identity changed")
        print(f"[3/9] Canonical inputs: OK run_id={run_id}")

        self.stage = "work-unit admission"
        if plan.kind == "campaign":
            try:
                submitted = self.json("submit-campaign", path, "--git-commit", self.port.commit, *self._storage(), "--inputs-prepared")
            except host_service.GenerationCommandError as error:
                _fail(f"Shared campaign submission failed: {error}")
            if submitted.get("campaign_run_id") != run_id:
                _fail("Campaign submission identity changed")
        else:
            submitted = self._benchmark_plan_command("submit-core-benchmark", path, resources)
            if submitted.get("benchmark_run_id") != run_id:
                _fail("Benchmark submission identity changed")
        print(f"[4/9] Work-unit plan: OK run_id={run_id}")

        self.stage = "work-unit monitoring"
        result.partial = self._monitor(result, resources)
        if result.partial:
            result.result = "PARTIAL"
        elif plan.kind == "benchmark":
            self.call("finalize-core-benchmark", run_id, *self._storage())
        print(f"[5/9] Work units: {'PARTIAL' if result.partial else 'OK'} run_id={run_id}")

        self.stage = "host publication"
        if plan.kind == "campaign":
            self._collect_campaign(result)
        else:
            self.call("validate-core-benchmark", run_id, *self._storage())
        print(f"[6/9] Host publication: OK run_id={run_id}")

        self.stage = "declared packages and finalizers"
        self._build_packages(result)
        print(f"[7/9] Packages/finalizer: {'INCOMPLETE' if result.partial else 'OK'} run_id={run_id}")
        if parent_retention and not result.partial:
            self._prepare_parent_child(result)
            return result

        self.stage = "workflow receipt and retention"
        self._retention(result)
        print(f"[8/9] Retention policy: OK run_id={run_id}")
        self.stage = "terminal validation"
        self._validate_leaf(result)
        print(f"[9/9] Final validation: OK run_id={run_id}")
        return result

    def _benchmark_plan_command(self, operation: str, path: str, resources: ExecutionResources) -> dict[str, Any]:
        output = self.port.call(
            operation,
            path,
            "--git-commit",
            self.port.commit,
            *self._storage(),
            scheduled=True,
            benchmark_preflight=True,
            partition=resources.partition,
            wall_time=resources.wall_time,
        )
        return _payload(output, operation)

    def _monitor(self, result: LeafResult, resources: ExecutionResources) -> bool:
        """Resume scheduler reconciliation until terminal service evidence exists."""
        old_handler = signal.getsignal(signal.SIGINT)
        self._interrupt_count = 0

        def interrupt(_signum: int, _frame: Any) -> None:
            self._interrupt_count += 1
            operation = "cancel-core-benchmark" if result.kind == "benchmark" else "cancel-campaign"
            options = ("--force",) if self._interrupt_count > 1 else ()
            try:
                self.call(operation, result.run_id, *options, *self._storage(), scheduled=False)
            except host_service.GenerationCommandError as error:
                print(f"Cancellation request failed; run evidence remains authoritative: {error}", file=sys.stderr)
            if self._interrupt_count > 1:
                raise KeyboardInterrupt
            print("Graceful cancellation requested. Press Ctrl+C again to force cancellation.", file=sys.stderr)

        signal.signal(signal.SIGINT, interrupt)
        try:
            while True:
                if result.kind == "campaign":
                    output = self.call("resume-campaign", result.run_id, "--format", "workflow-monitor", "--max-active-cases", "8", *self._storage())
                    lines = output.splitlines()
                    if len(lines) < _MIN_CAMPAIGN_MONITOR_LINES:
                        _fail("Malformed campaign workflow monitor")
                    header = lines[0].split("\t")
                    source = lines[1].split("\t")
                    if (
                        len(header) != _MONITOR_HEADER_FIELDS
                        or header[0] != "campaign-monitor"
                        or any(_DIGEST.fullmatch(item) is None for item in header[2:])
                        or len(source) != _SOURCE_MONITOR_FIELDS
                        or source[:2] != ["source-monitor", result.run_id]
                        or source[2] != header[1]
                    ):
                        _fail("Malformed campaign/source monitor identity")
                    state = header[1]
                    if source[4].isdigit():
                        self.cpu_bytes_retained = int(source[4])
                    else:
                        self.cpu_bytes_retained = None
                    print("\n".join(lines[2:]))
                    if state in {"successful", "transfer_complete"}:
                        self.call("validate-campaign-terminal", result.run_id, *self._storage())
                        return False
                    if state == "completed_with_failures":
                        return True
                    if state == "cancelled":
                        _fail("Campaign is cancelled; rerun the same config to resume eligible work")
                    if state not in {"running", "feeding", "license_blocked", "submission_pending_or_unknown"}:
                        _fail(f"Unsupported campaign monitor state: {state}")
                else:
                    self.call("resume-core-benchmark", result.run_id, *self._storage(), scheduled=True, benchmark_preflight=True)
                    output = self.call("core-benchmark-status", result.run_id, *self._storage(), "--format", "monitor")
                    benchmark_header, _, detail = output.partition("\n")
                    fields = benchmark_header.split("\t")
                    if len(fields) != _MONITOR_HEADER_FIELDS or fields[0] != "campaign-monitor":
                        _fail("Malformed benchmark monitor")
                    state = fields[1]
                    print(detail)
                    if state == "complete":
                        return False
                    if state in {"canary_failed", "work_unit_failed", "cancelled"}:
                        _fail(f"Benchmark reached terminal state: {state}")
                    if state not in {"inputs_ready", "running", "license_blocked"}:
                        _fail(f"Unsupported benchmark state: {state}")
                time.sleep(resources.poll_seconds)
        finally:
            signal.signal(signal.SIGINT, old_handler)

    def _collect_campaign(self, result: LeafResult) -> None:
        run_id = result.run_id
        if result.partial:
            self.call(
                "repair-partial-campaign-publication",
                run_id,
                "--source-host",
                "shared-filesystem",
                "--source-storage-root",
                str(self.port.storage),
                *self._storage(),
            )
        else:
            try:
                self.call("validate-published-campaign", run_id, *self._storage())
            except host_service.GenerationCommandError:
                authority = self.call("campaign-transfer-authority", run_id, *self._storage())
                self.call(
                    "repair-transferred-campaign",
                    run_id,
                    "--source-host",
                    "shared-filesystem",
                    "--source-storage-root",
                    str(self.port.storage),
                    "--authority-json",
                    authority,
                    *self._storage(),
                )
        if result.purpose == "pilot_check":
            self.call("record-pilot-source-inventory", run_id, *self._storage())
            self.call("record-shared-pilot-staging", run_id, *self._storage())

    def _build_packages(self, result: LeafResult) -> None:
        if result.kind == "benchmark":
            return
        if result.purpose == "pilot_check":
            catalog = self.json("list-campaigns", "--workflow", scheduled=False)
            production = catalog["workflow"]["primary"]["transient"]["repository_path"]
            production_path, _ = self._campaign_file(production)
            self.call("prepare-pilot-check", result.run_id, "--production-campaign", production_path, *self._storage(), "--keep-cpu-source")
        options = ("--partial",) if result.partial else ()
        package = self.json("build-campaign-datasets", result.run_id, *self._storage(), *options)
        expected = "incomplete" if result.partial else "complete"
        if package["status"] != expected:
            _fail(f"Dataset package state {package['status']} differs from {expected}")
        count = package.get("declared_package_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail("Dataset package count is malformed")

    def _prepare_parent_child(self, result: LeafResult) -> None:
        self.call("prepare-all-workflow", result.run_id, *self._storage(), "--keep-cpu-source")
        self.call("validate-campaign-package-state", result.run_id, *self._storage())

    def _retention(self, result: LeafResult) -> None:
        if result.kind == "benchmark":
            self.call("validate-core-benchmark", result.run_id, *self._storage())
            return
        options = ("--partial",) if result.partial else ()
        self.call("prepare-all-workflow", result.run_id, *self._storage(), *options, "--keep-cpu-source")
        source = self.call("campaign-source-status", result.run_id, "--query-scheduler", "--include-sizes", "--format", "tsv", *self._storage())
        fields = source.strip().split("\t")
        if len(fields) != _SOURCE_MONITOR_FIELDS or fields[:2] != ["source-status", result.run_id] or not fields[4].isdigit():
            _fail("Malformed retained CPU source status")
        self.cpu_bytes_retained = int(fields[4])
        if result.purpose == "pilot_check" and not result.partial:
            cleanup = self.call("cleanup-pilot-staging", result.run_id, "--confirm", "--format", "tsv", *self._storage())
            fields = cleanup.strip().split("\t")
            if len(fields) != _PILOT_CLEANUP_FIELDS or fields[:3] != ["pilot-staging-cleanup", "complete", "True"]:
                _fail("Pilot staging cleanup did not complete")
            _identifier(fields[4], _DIGEST, "pilot staging receipt digest")
            self.call(
                "record-pilot-cleanup",
                result.run_id,
                *self._storage(),
                "--cpu-bytes-reclaimed",
                "0",
                "--transfer-staging-removed",
                "--staging-bytes-reclaimed",
                fields[3],
                "--staging-cleanup-receipt-sha256",
                fields[4],
                scheduled=False,
            )

    def _validate_leaf(self, result: LeafResult) -> None:
        if result.kind == "benchmark":
            self.call("validate-core-benchmark", result.run_id, *self._storage())
            print(self.call("core-benchmark-summary", result.run_id, "--format", "markdown", *self._storage()))
            return
        options = ("--partial",) if result.partial else ()
        self.call("validate-all-workflow", result.run_id, *options, *self._storage())
        if not result.partial:
            self.call("validate-campaign-package-state", result.run_id, *self._storage())
        if result.purpose == "pilot_check" and not result.partial:
            self.call("validate-pilot-check", result.run_id, "--require-cleanup-complete", *self._storage())

    def _workflow(self, plan: RunPlan) -> tuple[LeafResult, ...]:
        if len(plan.children) != _PAIRED_SMOKE_CHILDREN or any(child.kind != "campaign" for child in plan.children):
            _fail("Paired Technical Smoke requires exactly two campaign children")
        children = tuple(self._leaf(child, parent_retention=True) for child in plan.children)
        if any(child.partial for child in children):
            return children
        resources = self.resources
        if resources is None:
            _fail("Missing Technical Smoke execution resources")
        self.port.verify_source()
        version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
        for child, child_plan in zip(children, plan.children, strict=True):
            self.run_id = child.run_id
            evidence = self.call("finalize-technical-smoke-evidence", child.run_id, "--comsol-version-output", version, *self._storage()).strip()
            child_path, _ = self._campaign_file(child_plan.config_path)
            self.call("technical-smoke-evidence-status", child_path, *self._storage(), "--comsol-version-output", version)
            evidence_path = Path(evidence)
            if not evidence_path.is_file() or evidence_path.is_symlink() or not evidence_path.is_relative_to(self.port.storage):
                _fail("Technical Smoke evidence is missing")
        receipt = self.call(
            "finalize-real-smoke", children[0].run_id, children[1].run_id, "--comsol-version-output", version, *self._storage()
        ).strip()
        self.call("validate-real-smoke", receipt, *self._storage())
        for child in children:
            self._retention(child)
            self._validate_leaf(child)
        self.call("validate-real-smoke", receipt, *self._storage())
        return children

    def _record_failure(self) -> None:
        """Record the stage and continuation without replacing service recovery state."""
        if not self.run_id or self.run_kind != "campaign":
            return
        arguments = ["./scripts/generation", "run", self.config_argument]
        if self.pool_option is not None:
            arguments.extend(("--replacement-pool-size", str(self.pool_option)))
            if not self.completion_owner_persisted and self.completion_parent_run:
                arguments.extend(("--parent-run-id", self.completion_parent_run))
        continuation = shlex.join(arguments)
        if self.cpu_bytes_retained is None:
            try:
                source = self.call(
                    "campaign-source-status",
                    self.run_id,
                    "--include-sizes",
                    "--format",
                    "tsv",
                    *self._storage(),
                    scheduled=False,
                )
                fields = source.strip().split("\t")
                if len(fields) == _SOURCE_MONITOR_FIELDS and fields[:2] == ["source-status", self.run_id] and fields[4].isdigit():
                    self.cpu_bytes_retained = int(fields[4])
            except host_service.GenerationCommandError:
                pass
        if self.cpu_bytes_retained is None:
            print(f"FAILED: {self.stage}\ncampaign_run_id: {self.run_id}\nNext: {continuation}", file=sys.stderr)
            return
        try:
            self.call(
                "record-workflow-failure",
                self.run_id,
                *self._storage(),
                "--stage",
                self.stage,
                "--continuation-command",
                continuation,
                "--cpu-bytes-retained",
                str(self.cpu_bytes_retained),
                "--format",
                "tsv",
                scheduled=False,
            )
        except (host_service.GenerationCommandError, OSError) as error:
            print(f"Failure evidence could not be recorded: {error}", file=sys.stderr)
        print(f"FAILED: {self.stage}\ncampaign_run_id: {self.run_id}\nNext: {continuation}", file=sys.stderr)

    def _completion_parent(self, path: str, parent_run_id: str | None = None, *, allow_untransferred: bool = False) -> dict[str, Any]:
        arguments = [path, *self._storage()]
        if parent_run_id:
            arguments.extend(("--parent-run-id", parent_run_id))
        if allow_untransferred:
            arguments.append("--allow-untransferred")
        return self.json("find-completion-parent", *arguments, scheduled=False)

    def completion_preview(self, plan: RunPlan, pool_size: int, parent_run_id: str | None = None) -> dict[str, Any]:
        """Attach exact completion-parent evidence to a read-only run plan."""
        path, _ = self._campaign_file(plan.config_path)
        resolution = self._completion_parent(path, parent_run_id)
        payload = dict(plan.payload)
        payload["replacement_completion"] = {
            "enabled": True,
            "replacement_pool_size": pool_size,
            "requested_high_water_mark": pool_size,
            "parent_run_id_override": parent_run_id,
            "compatible_parent_candidates": resolution["compatible_parent_candidates"],
            "selected_parent": resolution["selected_parent"],
            "target_counts": resolution["target_counts"],
            "current_successes": resolution["current_successes"],
            "deficits": resolution["success_deficits"],
            "expected_completion_id": resolution["expected_completion_id"],
            "package_declarations": resolution["package_declarations"],
            "pt_shard_requirements": resolution["pt_shard_requirements"],
            "parent_resolution": resolution,
        }
        return payload

    def _run_with_completion(
        self,
        plan: RunPlan,
        pool_size: int | None,
        parent_run_id: str | None,
        *,
        parent: dict[str, Any] | None = None,
    ) -> LeafResult:
        if pool_size is None or plan.kind != "campaign":
            _fail("Completion requires one campaign and a replacement pool", exit_code=2)
        path, relative = self._campaign_file(plan.config_path)
        if parent is None:
            parent = self._completion_parent(path, parent_run_id)
        status = parent["status"]
        if status == "compatible_partial":
            result = LeafResult(run_id=_identifier(parent["parent_run_id"], _RUN_ID, "parent run ID"), kind="campaign", purpose=plan.purpose)
            self.run_id = result.run_id
        elif status in {"fresh", "compatible_active", "compatible_complete"}:
            if status == "compatible_active" and parent["parent_run_id"] != plan.identity:
                _fail("Compatible active parent belongs to a different source commit; wait for transfer or use its exact commit")
            result = self._leaf(plan)
            if not result.partial:
                return result
            parent = self._completion_parent(path, result.run_id)
            if parent["status"] != "compatible_partial" or parent["parent_run_id"] != result.run_id:
                _fail("Newly partial campaign lacks exact transferred parent evidence")
        else:
            _fail(f"Unsupported completion parent state: {status}")
        parent_run = _identifier(parent["parent_run_id"], _RUN_ID, "parent run ID")
        completion_id = _identifier(parent["completion_id"], _COMPLETION_ID, "completion ID")
        partial_sha = _identifier(parent["parent_partial_sha256"], _DIGEST, "parent partial digest")
        partial_path = self._storage_path(str(parent["parent_partial_path"]))
        self.completion_parent_run = parent_run
        self.run_id = parent_run
        self.campaign_resources(plan.config_path)
        self.stage = "completion owner initialization"
        arguments = [
            relative,
            "--parent-run-id",
            parent_run,
            "--parent-partial",
            str(partial_path),
            "--parent-partial-sha256",
            partial_sha,
            *self._storage(),
        ]
        if self.pool_option is not None:
            arguments.extend(("--replacement-pool-size", str(pool_size)))
        initialized = self.json("initialize-campaign-completion", *arguments)
        if initialized["completion_id"] != completion_id:
            _fail("Completion owner identity changed")
        self.completion_owner_persisted = True
        if initialized["status"] == "failure_circuit_open":
            _fail("Completion failure circuit is open", exit_code=4)
        if initialized["status"] not in {"active", "complete", "pool_exhausted"}:
            _fail("Unsupported completion initialization state")
        print(f"Completion reconciliation: parent={parent_run} completion_id={completion_id}")
        self._advance_completion(relative, completion_id, parent_run)
        transfer = self._completion_transfer(completion_id, parent_run, partial_sha, path)
        self._collect_replacements(transfer)
        self._finalize_completion(transfer, parent_run, completion_id)
        result.run_id = parent_run
        result.result = "OK"
        result.partial = False
        self.run_id = parent_run
        return result

    def _storage_path(self, value: str) -> Path:
        lexical = Path(value)
        path = lexical.resolve(strict=True)
        if lexical != path or not path.is_file() or not path.is_relative_to(self.port.storage):
            _fail(f"Completion evidence escaped durable storage: {value}")
        return path

    def _advance_completion(self, relative: str, completion_id: str, parent_run: str) -> None:
        resources = self.resources
        if resources is None:
            _fail("Completion execution resources are missing")
        self.stage = "replacement completion reconciliation"
        while True:
            state = self.json("advance-campaign-completion", relative, completion_id, "--git-commit", self.port.commit, *self._storage())
            if state["completion_id"] != completion_id:
                _fail("Replacement completion owner identity changed")
            status = state["status"]
            if status == "complete":
                self.run_id = parent_run
                return
            if status == "failure_circuit_open":
                _fail("Completion failure circuit is open", exit_code=4)
            if status == "pool_exhausted":
                required = sum(int(value) for value in state["new_replacements_required"].values())
                if required <= 0:
                    _fail("Pool exhaustion lacks an uncovered deficit")
                current = _positive(state["replacement_pool_size"], "replacement pool high-water")
                next_pool = current + required
                print(
                    f"COMPLETION POOL EXHAUSTED: completion_id={completion_id} "
                    f"remaining={required} next=./scripts/generation run {shlex.quote(self.config_argument)} "
                    f"--replacement-pool-size {next_pool}",
                    file=sys.stderr,
                )
                _fail("Replacement pool exhausted", exit_code=3)
            if status != "active":
                _fail(f"Unsupported completion state: {status}")
            active = state.get("active_round_run_ids") or state.get("active_run_ids") or []
            if active:
                replacement = _identifier(active[0], _RUN_ID, "replacement run ID")
                self.run_id = replacement
                self._monitor(LeafResult(replacement, "campaign", "replacement"), resources)
                self.run_id = parent_run
            else:
                time.sleep(resources.poll_seconds)

    def _completion_transfer(self, completion_id: str, parent_run: str, partial_sha: str, config_path: str) -> dict[str, Any]:
        self.stage = "completion transfer membership validation"
        transfer = self.json("campaign-completion-transfer-plan", completion_id, *self._storage())
        if transfer["completion_id"] != completion_id or transfer["parent_run_id"] != parent_run or transfer["parent_partial_sha256"] != partial_sha:
            _fail("Completion transfer changed immutable parent identity")
        state_path = self._storage_path(str(transfer["completion_state_path"]))
        expected = self.port.storage / "01_generation/meta/completions" / completion_id / "completion.json"
        if state_path != expected or hashlib.sha256(state_path.read_bytes()).hexdigest() != transfer["completion_state_sha256"]:
            _fail("Completion owner state failed exact digest validation")
        if not transfer["replacement_runs"] or not transfer["replacement_campaigns"]:
            _fail("Complete campaign completion lacks replacement membership")
        run_ids = {_identifier(item["campaign_run_id"], _RUN_ID, "replacement run ID") for item in transfer["replacement_runs"]}
        for item in transfer["replacement_campaigns"]:
            _identifier(item["candidate_id"], _REPLACEMENT_ID, "replacement candidate ID")
            _identifier(item["target_batch_id"], _BATCH_ID, "replacement target batch ID")
            _identifier(item["terminal_batch_id"], _BATCH_ID, "replacement terminal batch ID")
            if _identifier(item["campaign_run_id"], _RUN_ID, "replacement run ID") not in run_ids:
                _fail("Completion transfer has a candidate outside its replacement runs")
        self.call("campaign-completion-status", completion_id, "--config", config_path, *self._storage())
        return transfer

    def _collect_replacements(self, transfer: dict[str, Any]) -> None:
        self.stage = "successful replacement publication"
        for item in transfer["replacement_runs"]:
            run_id = _identifier(item["campaign_run_id"], _RUN_ID, "replacement run ID")
            if item["campaign_state"] not in {"complete", "completed_with_failures"}:
                _fail("Replacement run has nonterminal transfer state")
            if not isinstance(item["partial"], bool):
                _fail("Replacement run has malformed partial state")
            self.run_id = run_id
            self._collect_campaign(LeafResult(run_id, "campaign", "replacement", partial=item["partial"]))
        self.run_id = str(transfer["parent_run_id"])

    def _finalize_completion(self, transfer: dict[str, Any], parent_run: str, completion_id: str) -> None:
        self.stage = "completion composite publication"
        self.call("build-campaign-completion-composite", completion_id, *self._storage())
        self.call("build-campaign-completion-lifecycle", parent_run, completion_id, *self._storage())
        self.call("validate-campaign-completion-lifecycle", parent_run, completion_id, *self._storage())
        cleanup = self.json("campaign-completion-cleanup-plan", parent_run, completion_id, *self._storage())
        expected = sorted(_identifier(item["terminal_batch_id"], _BATCH_ID, "replacement batch ID") for item in transfer["replacement_campaigns"])
        observed = sorted(item["terminal_batch_id"] for item in cleanup["sources"])
        if not cleanup.get("eligible") or observed != expected:
            _fail("Completion cleanup plan differs from successful replacement membership")
        self.call("validate-campaign-completion-lifecycle", parent_run, completion_id, *self._storage())

    def status(self, target: str) -> str:
        """Compose read-only service status for a config or persisted run ID."""
        self.execute = False
        lines: list[str] = []
        candidate = Path(target)
        is_config = candidate.is_file() if candidate.is_absolute() else (self.port.repository / candidate).is_file()
        if is_config:
            plan = self.plan(target, allow_incomplete=True)
            lines.extend((f"run_identity={plan.identity}", f"run_kind={plan.kind}", f"config={plan.config_path}"))
            if plan.kind == "workflow":
                for index, child in enumerate(plan.children, 1):
                    lines.append(f"Child {index}/{len(plan.children)}: {child.identity}")
                    lines.extend(self._campaign_status(child.identity, child.config_path))
                try:
                    self.call("validate-real-smoke", *self._storage())
                except host_service.GenerationCommandError:
                    lines.append("Package/finalizer state: absent_or_incomplete")
                else:
                    lines.append("Package/finalizer state: complete")
            elif plan.kind == "campaign":
                lines.extend(self._campaign_status(plan.identity, plan.config_path))
            elif plan.kind == "benchmark":
                resources, _ = self.benchmark_resources(plan.config_path)
                version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
                path, _ = self._campaign_file(plan.config_path)
                identity = self.json("resolve-core-benchmark-run", path, "--git-commit", self.port.commit, "--comsol-version-output", version)
                run_id = _identifier(identity["benchmark_run_id"], _RUN_ID, "benchmark run ID")
                lines.extend(self._benchmark_status(run_id))
            else:
                _fail(f"Unsupported run kind: {plan.kind}")
        elif target.startswith("core_scaling_transient__"):
            run_id = _identifier(target, _RUN_ID, "benchmark run ID")
            lines.extend(self._benchmark_status(run_id))
        else:
            run_id = _identifier(target, _RUN_ID, "campaign run ID")
            lines.extend(self._campaign_status(run_id))
        return "\n".join(lines)

    def _campaign_status(self, run_id: str, config: str | None = None) -> list[str]:
        lines: list[str] = []
        if config is not None:
            path, relative = self._campaign_file(config)
            self.campaign_resources(relative, executable=False)
            resolution = self._completion_parent(path, allow_untransferred=True)
            lines.extend(("Completion status:", json.dumps(resolution, sort_keys=True)))
            if resolution["status"] == "compatible_partial":
                completion_id = _identifier(
                    resolution.get("completion_id") or resolution.get("expected_completion_id"),
                    _COMPLETION_ID,
                    "completion ID",
                )
                lines.append("Completion execution and finalization status:")
                lines.append(self.call("campaign-completion-status", completion_id, "--if-present", "--config", path, *self._storage()))
        lines.append("Campaign status:")
        lines.append(self.call("campaign-status", run_id, "--format", "workflow-monitor", "--max-active-cases", "8", *self._storage()))
        lines.append("Durable storage status:")
        lines.append(
            self.call("storage-status", "--role", "gpu", "--metadata-only", "--omit-run-status", *self._storage(), "--campaign-run-id", run_id)
        )
        lines.append(self.call("validate-pilot-check", run_id, "--if-present", "--format", "summary", *self._storage()))
        return lines

    def _benchmark_status(self, run_id: str) -> list[str]:
        lines = [
            "Benchmark status:",
            self.call("core-benchmark-status", run_id, *self._storage(), "--format", "summary"),
            "CPU source status:",
            self.call("core-benchmark-source-status", run_id, *self._storage()),
        ]
        try:
            self.call("validate-core-benchmark", run_id, *self._storage())
        except host_service.GenerationCommandError:
            lines.append("Host publication state: absent_or_incomplete")
        else:
            lines.append("Host publication state: complete")
        return lines

    def cancel(self, run_id: str, *, force: bool = False) -> str:
        """Delegate cancellation to the campaign or benchmark state owner."""
        _identifier(run_id, _RUN_ID, "run ID")
        operation = "cancel-core-benchmark" if run_id.startswith("core_scaling_transient__") else "cancel-campaign"
        options = ("--force",) if force else ()
        return self.call(operation, run_id, *self._storage(), *options, scheduled=False)
