"""Coordinate Generation runs from authoritative service evidence.

Responsibilities:
  - Interpret immutable run plans and configured execution resources
  - Summarize the Slurm footprint of a run (node jobs and driver) for read-only previews
  - Sequence source campaign and benchmark lifecycles
  - Reconcile repeat invocations, monitor work, and gate source publication
  - Report failure stages of a run, from plan resolution and preflight on, with the command that continues, preserving service-owned state
  - Print each stage with its duration and a live campaign board in the driver log
  - Have the host link the driver's logs into its campaign as soon as the campaign exists, also when its preparation failed

Design principles:
  - Existing Generation services own identities, manifests, and scientific data
  - Every continuation is reconstructed from persisted service evidence
  - Finalization follows exact source and benchmark validation gates
  - A source failure stops the driver only after it persisted through reconciliations for a settle window
  - Printing reads no extra evidence and never changes a decision, a call, or an exit status

This module does NOT:
  - Submit individual scientific cases or define solver behavior
  - Define dataset or publication persistence formats
  - Load environment modules or execute work outside Slurm
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn, Protocol

from src import common
from src.generation.cases import generation_cases_config as config_service

from . import generation_campaign_status as status_service
from .runtime import generation_runtime_host as host_service

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

# Case evidence written on another node can stay invisible to the driver's node for tens of seconds
# (shared-filesystem attribute caching; 25 s observed on ICE), so a failure must persist this long.
_FAILURE_SETTLE_SECONDS: Final = 120
_NODE_JOB_SETTLE_SECONDS: Final = 1800  # longer than the node worker's 900 s idle timeout
# Slurm time of one scheduled Python operation (input generation); it is not the node-job time of the case steps.
_PYTHON_OPERATION_WALL_TIME: Final = "02:00:00"
# Printed stages of a run: the plan, then input readiness, submission with monitoring, and the closing stage of a campaign.
_CAMPAIGN_STAGES: Final = 4
_BENCHMARK_STAGES: Final = 5


class GenerationControllerError(RuntimeError):
    """A failed controller transition with a stable process exit status."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        """Retain an actionable diagnostic and the public process exit status."""
        super().__init__(message)
        self.exit_code = exit_code
        self.stage: str | None = None
        self.next_command: str | None = None

    def report(self) -> str:
        """Return the one console report of this failure: the failed stage with the message, then the command that continues."""
        if self.stage is None:
            return str(self)
        return "\n".join((f"FAILED: {self.stage}: {self}", *([f"Next: {self.next_command}"] if self.next_command else [])))


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
        wall_time: str = _PYTHON_OPERATION_WALL_TIME,
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

    def choose_tree(self, *, bound_snapshot: Path | None, continue_with_current_code: bool) -> tuple[host_service.SourceAdmission, bool]:
        """Choose the bound or the current frozen source tree and whether that is an explicit source transition."""
        ...

    def submit_tree(
        self,
        admission: host_service.SourceAdmission,
        *,
        module: str,
        arguments: Sequence[str],
        explicit_transition: bool,
        job: host_service.LauncherJob,
    ) -> host_service.SubmittedLauncher:
        """Submit the launcher to run from its chosen frozen source tree as its own Slurm job."""
        ...

    def attach_logs(self, campaign_directory: Path) -> None:
        """Link this job's logs and its helper logs into the existing campaign."""
        ...


_SCHEDULED_OPERATIONS = frozenset(
    {
        "prepare-campaign-inputs",
        "resume-campaign",
        "finalize-core-benchmark",
        "validate-core-benchmark",
    }
)
_RUN_ID = re.compile(r"[A-Za-z0-9._-]+__[0-9a-f]{16}\Z")
_MONITOR_HEADER_FIELDS = 4


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


def _campaign_identifier(value: Any) -> str:
    """Admit one structured current source campaign handle."""
    if not isinstance(value, str):
        _fail(f"Malformed campaign-run ID: {value!r}")
    if common.paths.campaign_run_parts(value) is not None:
        return value
    _fail(f"Campaign-run ID is not a current source handle: {value!r}")


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
    parallel_cases: int = 1

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
        node_jobs = _positive(cluster["node_jobs"], "node_jobs")
        return cls(
            partition=str(site["partition"]),
            scheduler=str(site["scheduler"]),
            python_module=str(site["python_module"]),
            comsol_module=str(site["comsol_module"]),
            python_executable=str(site["python_executable"]),
            comsol_executable=str(site["comsol_executable"]),
            poll_seconds=_positive(submission["poll_interval_seconds"], "poll_interval_seconds"),
            wall_time=_PYTHON_OPERATION_WALL_TIME,
            cores_per_node=cores_per_node,
            purpose=str(value["campaign_purpose"]),
            parallel_cases=node_jobs * (cores_per_node // cores_per_case),
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
            wall_time=_PYTHON_OPERATION_WALL_TIME,
            cores_per_node=_positive(site["cores_per_node"], "cores_per_node"),
        )


def plan_execution_values(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the validated execution values of every campaign in a run plan, parent first."""
    values: list[Mapping[str, Any]] = [payload["execution_values"]] if payload.get("execution_values") else []
    for child in payload.get("children", ()):
        values.extend(plan_execution_values(child))
    return values


def scheduler_footprint(
    execution_values: Sequence[Mapping[str, Any]],
    *,
    driver_time: str,
    driver_partition: str = "standard",
) -> dict[str, Any] | None:
    """
    Return the maximum Slurm footprint a run asks for: its node jobs and its driver job.

    Parameters
    ----------
    execution_values : Sequence[Mapping[str, Any]]
        Validated execution values of the campaigns the run executes; with several campaigns each
        field takes the largest value, because the driver runs them one after another.
    driver_time : str
        Time limit of the driver job as given to ``--driver-time``.
    driver_partition : str, optional
        Partition of the driver job.

    Returns
    -------
    dict[str, Any] | None
        ``node_jobs``, ``cores_per_node_job``, ``node_job_memory`` (``<int>G``), ``node_job_wall_time``
        (``HH:MM:SS``), ``driver_cpus`` and ``driver_time``; ``None`` when the plan carries no campaign
        execution values (a benchmark), whose footprint is not defined by this contract.

    Raises
    ------
    ValueError
        If the driver time or partition is not an admissible launcher setting.

    """
    driver = host_service.LauncherJob("generation", "generation", driver_partition, driver_time)
    if not execution_values:
        return None
    memory = max(
        (value["cluster"]["node_job_memory"] for value in execution_values),
        key=lambda text: config_service.parse_slurm_memory_mib(text, label="node_job_memory"),
    )
    wall_time = max(
        (value["cluster"]["node_job_wall_time"] for value in execution_values),
        key=lambda text: config_service.parse_wall_time_seconds(text, label="node_job_wall_time"),
    )
    return {
        "node_jobs": max(int(value["cluster"]["node_jobs"]) for value in execution_values),
        "cores_per_node_job": max(int(value["site"]["cores_per_node"]) for value in execution_values),
        "node_job_memory": memory,
        "node_job_wall_time": wall_time,
        "driver_cpus": driver.cpus_per_task,
        "driver_time": driver.wall_time,
    }


def benchmark_scheduler_footprint(
    resource_contract: Mapping[str, Any],
    *,
    driver_time: str,
    driver_partition: str = "standard",
) -> dict[str, Any]:
    """
    Return the maximum Slurm footprint of a core benchmark: its sweep node jobs and its driver job.

    Parameters
    ----------
    resource_contract : Mapping[str, Any]
        The suite's resource contract (``inspect-core-benchmark``), which owns the sweep node-job resources.
    driver_time : str
        Time limit of the driver job as given to ``--driver-time``.
    driver_partition : str, optional
        Partition of the driver job.

    Returns
    -------
    dict[str, Any]
        The same keys as ``scheduler_footprint``.

    """
    driver = host_service.LauncherJob("generation", "generation", driver_partition, driver_time)
    return {
        "node_jobs": int(resource_contract["node_jobs"]),
        "cores_per_node_job": int(resource_contract["cores_per_node"]),
        "node_job_memory": str(resource_contract["node_job_memory"]),
        "node_job_wall_time": str(resource_contract["node_job_wall_time"]),
        "driver_cpus": driver.cpus_per_task,
        "driver_time": driver.wall_time,
    }


def _require_continuable_state(run_id: str, state: object) -> None:
    """Stop with an actionable message for every campaign state the controller does not continue by itself."""
    if state == "license_wait_exhausted":
        _fail(f"Canonical campaign stopped: the licence wait budget of at least one case is exhausted; inspect status {run_id}")
    if state in {"blocked", "submission_unknown", "cancel_requested", "force_cancel_requested", "failure_threshold_reached"}:
        _fail(f"Canonical campaign requires explicit continuation from state {state!r}; inspect status {run_id}")


@dataclass(slots=True)
class LeafResult:
    """Transient result of reconstructing and executing one leaf lifecycle."""

    run_id: str
    kind: str
    purpose: str
    result: str = "OK"
    partial: bool = False

    @property
    def source_name(self) -> str | None:
        """Return the Source locator for a campaign result."""
        parts = common.paths.campaign_run_parts(self.run_id)
        return None if parts is None else parts[1]


def _campaign_override_arguments(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Render the normal resolved science and execution request without defaults."""
    arguments: list[str] = []
    if payload.get("authored_value") is not None:
        arguments.extend(("--authored-value-json", json.dumps(payload["authored_value"], sort_keys=True)))
    for category in ("scientific_overrides", "execution_overrides"):
        for key, value in payload.get(category, {}).items():
            if value is False:
                continue
            arguments.append("--" + key.replace("_", "-"))
            if value is not True:
                arguments.extend(str(item) for item in value) if isinstance(value, list) else arguments.append(str(value))
    return tuple(arguments)


def _unsettled_failure(run_id: str, cases: Sequence[Mapping[str, Any]], counts: Mapping[str, Any]) -> str | None:
    """Describe a quiescent source campaign that no recorded decision or replacement settles, else return ``None``.

    A failed case with a recorded terminal decision never stops the campaign: its
    slot is either covered by a replacement or counted as an unfilled deficit.
    """
    deficit = counts.get("replacement_deficit")
    if isinstance(deficit, bool) or not isinstance(deficit, int) or any("terminal_decision" not in case for case in cases):
        _fail("Canonical campaign status lacks current candidate evidence")
    undecided = [case for case in cases if case.get("state") == "failed" and case["terminal_decision"] is None]
    if undecided:
        names = ", ".join(f"{case.get('material')} {case.get('case_id')}" for case in undecided)
        return (
            f"Canonical campaign has failed source cases without a recorded decision: {names}. Inspect status {run_id}, "
            f"record each failure with `./scripts/generation fail-case {run_id} MATERIAL CASE_ID --reason-code CODE --detail TEXT`, "
            "then rerun the same command."
        )
    if deficit > 0:
        return (
            f"Canonical campaign has {deficit} unfilled success slot(s) and no pending or running job: the Source cannot reach its target "
            f"without more replacement capacity; add it with `./scripts/generation amend-replacements {run_id} --material MATERIAL "
            "--regime REGIME --capacity N --reason TEXT` or start a new Source."
        )
    return None


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
        self.scientific_arguments: tuple[str, ...] = ()
        self.restart_arguments: tuple[str, ...] = ()
        self.authored_campaign = False
        self.resources: ExecutionResources | None = None
        # A clock of its own for the printed durations and the board, apart from the settle window's time.monotonic.
        self.clock: Callable[[], float] = time.perf_counter
        self._stage_number = 0
        self._stage_total = 0
        self._stage_started: float | None = None
        self._board_diagnostics: set[str] = set()

    def _enter_stage(self, stage: str, description: str | None = None) -> None:
        """Close the running stage with its duration, then announce the next one."""
        self._leave_stage()
        self.stage = stage
        self._stage_number += 1
        self._stage_started = self.clock()
        print(f"[{self._stage_number}/{self._stage_total}] {description or stage}")

    def _leave_stage(self) -> None:
        """Print the duration of the running stage, if one runs."""
        if self._stage_started is not None:
            duration = status_service.format_duration(self.clock() - self._stage_started)
            print(f"[{self._stage_number}/{self._stage_total}] {self.stage} done in {duration}")
            self._stage_started = None

    def _show_board(self, board: status_service.CampaignBoard, status: Mapping[str, Any]) -> None:
        """Print the live board when it is due; a rendering error is reported once per distinct message and never changes the outcome."""
        try:
            text = board.observe(status, now=self.clock())
            if text is not None:
                print(f"\n{text}")
        except Exception as error:  # noqa: BLE001 -- presentation must not change the campaign outcome
            diagnostic = f"status board unavailable: {error}"
            if diagnostic not in self._board_diagnostics:
                self._board_diagnostics.add(diagnostic)
                print(diagnostic, file=sys.stderr)

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
            wall_time=resources.wall_time if resources else _PYTHON_OPERATION_WALL_TIME,
        )

    def json(self, operation: str, *arguments: str, scheduled: bool | None = None) -> dict[str, Any]:
        """Call a JSON service command and decode its result."""
        return _payload(self.call(operation, *arguments, scheduled=scheduled), operation)

    @contextlib.contextmanager
    def _checked_before_run(self, stage: str, config: str, arguments: Sequence[str]) -> Iterator[None]:
        """Bind a failure of a step that precedes ``run`` to its stage and to the dry run that checks the config again.

        An authored campaign has no command line to repeat, so its failure names only the stage.
        """
        self.stage = stage
        try:
            yield
        except BaseException as error:
            authored = "--authored-value-json" in arguments
            self._record_failure(error, None if authored else shlex.join(["./scripts/generation", "run", config, *arguments, "--dry-run"]))
            raise

    def plan(self, config: str, *, allow_incomplete: bool = False, scientific_arguments: tuple[str, ...] = ()) -> RunPlan:
        """Resolve a config for a run; a failure names plan resolution and the dry run that checks the config again."""
        with self._checked_before_run("plan resolution", config, scientific_arguments):
            return self._resolved_plan(config, allow_incomplete=allow_incomplete, scientific_arguments=scientific_arguments)

    def _resolved_plan(self, config: str, *, allow_incomplete: bool, scientific_arguments: tuple[str, ...]) -> RunPlan:
        """Resolve a config through the existing immutable run-plan owner."""
        config_path, _ = self.port.config_file(config)
        arguments = [str(config_path), "--git-commit", self.port.commit, *scientific_arguments]
        if allow_incomplete:
            arguments.append("--allow-incomplete")
        return RunPlan.from_payload(self.json("resolve-generation-run", *arguments, scheduled=False))

    def authored_campaign_plan(
        self,
        plan_path: Path | str,
        authored_value: Mapping[str, Any],
        *,
        execution_overrides: Mapping[str, Any] | None = None,
        scientific_overrides: Mapping[str, Any] | None = None,
    ) -> RunPlan:
        """Resolve one in-memory campaign to its run plan without executing it."""
        arguments = _campaign_override_arguments(
            {
                "authored_value": dict(authored_value),
                "scientific_overrides": dict(scientific_overrides or {}),
                "execution_overrides": dict(execution_overrides or {}),
            }
        )
        plan = self.plan(str(plan_path), scientific_arguments=tuple(arguments))
        if plan.kind != "campaign":
            _fail("Authored values must resolve to a campaign", exit_code=2)
        return plan

    def choose_run_tree(self, plan: RunPlan, *, continue_with_current_code: bool) -> tuple[host_service.SourceAdmission, bool]:
        """
        Choose the source tree a run executes in and whether that is an explicit source transition.

        An existing campaign continues in the tree of its latest execution epoch
        unless ``continue_with_current_code`` asks for the current code, which the
        campaign then records as an explicit source transition. New campaigns and
        other run kinds use the current code.
        """
        bound = None
        if plan.kind == "campaign":
            from . import generation_campaign as campaign_service  # noqa: PLC0415 -- persisted campaign bindings are read only by launchers

            bound = campaign_service.bound_source_snapshot(plan.identity, storage_root=self.port.storage)
        return self.port.choose_tree(bound_snapshot=bound, continue_with_current_code=continue_with_current_code)

    def submit_run_tree(
        self,
        plan: RunPlan,
        *,
        module: str,
        arguments: Sequence[str],
        continue_with_current_code: bool,
        job: host_service.LauncherJob,
    ) -> host_service.SubmittedLauncher:
        """Submit a launcher as its own Slurm job to run from the source tree its run executes in."""
        admission, explicit = self.choose_run_tree(plan, continue_with_current_code=continue_with_current_code)
        return self.port.submit_tree(admission, module=module, arguments=arguments, explicit_transition=explicit, job=job)

    def run_authored_campaign(
        self,
        plan_path: Path | str,
        authored_value: Mapping[str, Any],
        *,
        execution_overrides: Mapping[str, Any] | None = None,
        scientific_overrides: Mapping[str, Any] | None = None,
        max_admission_cases: int | None = None,
    ) -> LeafResult:
        """Run one in-memory campaign through the existing admitted controller.

        ``plan_path`` identifies the repository YAML preserved as recovery evidence.
        The authored mapping contains only Generation campaign values. A ``STAGED``
        result retains incomplete bounded admission; failures raise the ordinary
        controller error and preserve the service-owned continuation evidence.
        """
        plan = self.authored_campaign_plan(
            plan_path, authored_value, execution_overrides=execution_overrides, scientific_overrides=scientific_overrides
        )
        result = self.run(plan, str(plan_path), max_admission_cases=max_admission_cases)
        if not isinstance(result, LeafResult) or result.kind != "campaign":
            _fail("Authored campaign resolved to an unsupported run kind", exit_code=2)
        return result

    def campaign_resources(self, config: str, *, executable: bool = True, override_arguments: tuple[str, ...] = ()) -> ExecutionResources:
        """Read the configured campaign resource contract once in Python."""
        config_path, _ = self.port.config_file(config)
        arguments = [str(config_path), *override_arguments]
        if not executable:
            arguments.append("--allow-incomplete")
        inspection = self.json("validate-config", *arguments, scheduled=False)
        resources = ExecutionResources.campaign(inspection)
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
        overrides = _campaign_override_arguments(plan.payload)
        with self._checked_before_run("preflight", plan.config_path, overrides):
            if plan.kind == "benchmark":
                resources, _ = self.benchmark_resources(plan.config_path)
            elif plan.kind == "campaign":
                resources = self.campaign_resources(plan.config_path, override_arguments=overrides)
            else:
                _fail(f"Unsupported plan kind: {plan.kind}", exit_code=2)
            self.port.verify_execution()
            version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
        return f"PREFLIGHT COMPLETE: plan={plan.identity} kind={plan.kind} host=shared-filesystem COMSOL={version}"

    def run(self, plan: RunPlan, config_argument: str, *, max_admission_cases: int | None = None) -> LeafResult | tuple[LeafResult, ...]:
        """Execute or resume one immutable plan from service-owned evidence."""
        self.execute = True
        self.config_argument = config_argument
        self.restart_arguments = (
            ("--max-admission-cases", str(max_admission_cases))
            if max_admission_cases is not None and "max_admission_cases" not in plan.payload.get("execution_overrides", {})
            else ()
        )
        self.run_kind = plan.kind
        self.authored_campaign = plan.payload.get("authored_value") is not None
        self._stage_number = 0
        self._stage_started = None
        self._stage_total = _CAMPAIGN_STAGES if plan.kind == "campaign" else _BENCHMARK_STAGES
        self._enter_stage("common plan resolution", f"Run plan: kind={plan.kind} identity={plan.identity} config={plan.config_path}")
        try:
            result: LeafResult | tuple[LeafResult, ...]
            if plan.kind == "campaign":
                if common.paths.campaign_run_parts(plan.identity) is None:
                    _fail("Campaign plan is not a current source campaign", exit_code=2)
                result = self._canonical_campaign(plan, max_admission_cases=max_admission_cases)
            elif plan.kind == "benchmark":
                result = self._benchmark(plan)
            else:
                _fail(f"Unsupported Generation run kind: {plan.kind}", exit_code=2)
        except BaseException as error:
            self._record_failure(error, self._continuation())
            raise
        self._leave_stage()
        staged = any(child.result == "STAGED" for child in result) if isinstance(result, tuple) else result.result == "STAGED"
        if staged:
            print(f"GENERATION STAGED: run_identity={plan.identity}")
        else:
            print(f"GENERATION COMPLETE: run_identity={plan.identity}")
        return result

    def _canonical_campaign(self, plan: RunPlan, *, max_admission_cases: int | None = None) -> LeafResult:
        """Advance the source campaign to completion or its recorded admission cap."""
        run_id = _campaign_identifier(plan.identity)
        path, _ = self._campaign_file(plan.config_path)
        scientific_arguments = _campaign_override_arguments(plan.payload)
        resources = self.campaign_resources(plan.config_path, override_arguments=scientific_arguments)
        self.port.verify_execution()
        self.run_id = run_id
        result = LeafResult(run_id=run_id, kind="campaign", purpose=plan.purpose)
        self.scientific_arguments = scientific_arguments
        self._enter_stage("canonical input readiness")
        parts = common.paths.campaign_run_parts(run_id)
        if parts is None:
            _fail("Canonical campaign run identity is malformed")
        run_directory = common.paths.resolve_generation_campaign_directory(*parts, storage_root=self.port.storage)
        persisted = common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(run_directory)).exists()
        try:
            if not persisted:
                prepared = self.call("prepare-campaign-inputs", path, "--git-commit", self.port.commit, *scientific_arguments, *self._storage())
                if not prepared.startswith("canonical-inputs\t"):
                    _fail("Malformed canonical campaign input result")
            self._enter_stage("campaign submission and reconciliation")
            arguments = ("--inputs-prepared",) if not persisted else ()
            if max_admission_cases is not None and "max_admission_cases" not in plan.payload.get("execution_overrides", {}):
                arguments = (*arguments, "--max-admission-cases", str(max_admission_cases))
            manifest = self.json(
                "submit-campaign",
                path,
                "--git-commit",
                self.port.commit,
                *scientific_arguments,
                *self._storage(),
                "--metadata-only",
                *arguments,
                scheduled=False,
            )
        finally:
            # The logs follow the campaign as soon as it exists, also when its inputs or submission failed; never create it for them.
            if run_directory.is_dir():
                self.port.attach_logs(run_directory)
        if manifest.get("campaign_run_id") != run_id:
            _fail("Canonical campaign submission identity changed")
        if manifest.get("continuation_requires_compute") is True:
            manifest = self.json("resume-campaign", run_id, *self._storage())
            if manifest.get("campaign_run_id") != run_id:
                _fail("Canonical campaign continuation identity changed")
        unsettled_since: float | None = None
        board = status_service.CampaignBoard(started=self.clock(), seats=resources.parallel_cases, verify_seconds=_FAILURE_SETTLE_SECONDS)
        while manifest.get("state") != "complete":
            state = manifest.get("state")
            if self._cancellation_settling(run_id, state):
                time.sleep(resources.poll_seconds)
                manifest = self._resume_campaign(run_id)
                continue
            _require_continuable_state(run_id, state)
            if state not in {
                "ready",
                "submitting",
                "active",
                "license_blocked",
                "completed_with_failures",
                "staged",
                "scheduler_unknown",
                "submission_failed",
            }:
                _fail(f"Unsupported canonical campaign state: {state!r}")
            status = self.json("campaign-status", run_id, *self._storage())
            if status.get("campaign_run_id") != run_id:
                _fail("Canonical campaign status identity changed")
            self._show_board(board, status)
            unsettled: str | None = None
            if (
                status.get("campaign_state") in {"staged", "completed_with_failures"}
                and status.get("pending_jobs") == 0
                and status.get("running_jobs") == 0
                and not status.get("open_node_jobs")
                and status.get("submission_unknown") is False
            ):
                cases = status.get("cases")
                counts = status.get("counts")
                if not isinstance(cases, list) or not cases or not isinstance(counts, dict):
                    _fail("Canonical campaign status lacks current candidate evidence")
                unsettled = _unsettled_failure(run_id, cases, counts)
                if unsettled is None:
                    successful = sum(case.get("state") == "successful" for case in cases)
                    if (
                        0 < successful < len(cases)
                        and counts.get("successful") == successful
                        and all(case.get("state") in {"successful", "never_started"} for case in cases)
                    ):
                        result.result = "STAGED"
                        self._enter_stage("bounded source admission")
                        return result
            # A status read right after jobs end can predate their evidence on this node and the feeder's reconciliation.
            if unsettled is None:
                unsettled_since = None
            elif unsettled_since is None:
                unsettled_since = time.monotonic()
            elif time.monotonic() - unsettled_since >= _FAILURE_SETTLE_SECONDS:
                _fail(unsettled)
            time.sleep(resources.poll_seconds)
            manifest = self._resume_campaign(run_id)
        self._show_board(board, self._await_settled_node_jobs(run_id, resources))
        self._enter_stage("selected source publication")
        return result

    def _cancellation_settling(self, run_id: str, state: object) -> bool:
        """Return whether a persisted cancellation still has node jobs or submissions to end.

        Every resume replays the cancellation, so the controller keeps polling until the last node job has ended.
        """
        if state not in {"cancel_requested", "force_cancel_requested"}:
            return False
        status = self.json("campaign-status", run_id, *self._storage())
        return bool(status.get("open_node_jobs") or status.get("submission_unknown"))

    def _await_settled_node_jobs(self, run_id: str, resources: ExecutionResources) -> dict[str, Any]:
        """After completion, poll until every node job has ended and no dispatch is unresolved, at most _NODE_JOB_SETTLE_SECONDS.

        The feeder sends stop tickets on completion; workers end within a poll, an orphan at its idle timeout.
        Returns the status that showed the node jobs settled.
        """
        deadline = time.monotonic() + _NODE_JOB_SETTLE_SECONDS
        while True:
            status = self.json("campaign-status", run_id, *self._storage())
            if status.get("campaign_run_id") != run_id:
                _fail("Canonical campaign status identity changed")
            if not status.get("open_node_jobs") and status.get("pending_jobs") == 0 and status.get("running_jobs") == 0:
                return status
            if time.monotonic() >= deadline:
                _fail(f"Campaign {run_id} is complete but node jobs or dispatches did not settle within {_NODE_JOB_SETTLE_SECONDS} s")
            time.sleep(resources.poll_seconds)
            self._resume_campaign(run_id)

    def _resume_campaign(self, run_id: str) -> dict[str, Any]:
        """Reconcile one source campaign through the feeder, entering the compute path when the feeder asks for it."""
        manifest = self.json("resume-campaign", run_id, *self._storage(), "--metadata-only", scheduled=False)
        if manifest.get("continuation_requires_compute") is True:
            manifest = self.json("resume-campaign", run_id, *self._storage())
        if manifest.get("campaign_run_id") != run_id:
            _fail("Canonical campaign continuation identity changed")
        return manifest

    def _benchmark(self, plan: RunPlan) -> LeafResult:
        """Advance one benchmark through its own source and terminal owner."""
        path, _ = self._campaign_file(plan.config_path)
        resources, _ = self.benchmark_resources(plan.config_path)
        version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
        identity = self.json("resolve-core-benchmark-run", path, "--git-commit", self.port.commit, "--comsol-version-output", version)
        run_id = _identifier(identity["benchmark_run_id"], _RUN_ID, "benchmark-run ID")
        self.run_id = run_id
        result = LeafResult(run_id=run_id, kind="benchmark", purpose=plan.purpose)
        try:
            self.call("validate-core-benchmark", run_id, *self._storage())
        except host_service.GenerationCommandError:
            pass
        else:
            result.result = "REUSED"
            print(self.call("core-benchmark-summary", run_id, "--format", "markdown", *self._storage()))
            return result
        self.port.verify_execution()
        self._enter_stage("benchmark input readiness")
        materialized = self._benchmark_plan_command("materialize-core-benchmark-inputs", path, resources)
        if materialized["benchmark_run_id"] != run_id:
            _fail("Benchmark input identity changed")
        self._enter_stage("benchmark work-unit admission")
        submitted = self._benchmark_plan_command("submit-core-benchmark", path, resources)
        if submitted["benchmark_run_id"] != run_id:
            _fail("Benchmark submission identity changed")
        self._enter_stage("benchmark scheduler reconciliation")
        self._monitor(result, resources)
        self._enter_stage("benchmark terminal publication")
        self.call("finalize-core-benchmark", run_id, *self._storage())
        self.call("validate-core-benchmark", run_id, *self._storage())
        print(self.call("core-benchmark-summary", run_id, "--format", "markdown", *self._storage()))
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

    def _monitor(self, result: LeafResult, resources: ExecutionResources) -> None:
        """Reconcile a benchmark until terminal service evidence exists."""
        while True:
            self.call("resume-core-benchmark", result.run_id, *self._storage(), scheduled=True, benchmark_preflight=True)
            output = self.call("core-benchmark-status", result.run_id, *self._storage(), "--format", "monitor")
            benchmark_header, _, detail = output.partition("\n")
            fields = benchmark_header.split("\t")
            if len(fields) != _MONITOR_HEADER_FIELDS or fields[0] != "campaign-monitor":
                _fail("Malformed benchmark monitor")
            state = fields[1]
            print(detail)
            if state == "complete":
                return
            if state in {"variant_failed", "cancelled"}:
                _fail(f"Benchmark reached terminal state: {state}")
            if state not in {"inputs_ready", "running"}:
                _fail(f"Unsupported benchmark state: {state}")
            time.sleep(resources.poll_seconds)

    def _continuation(self) -> str | None:
        """Return the command that continues the identified run: rerunning its command (an authored campaign's status), else ``None``.

        A campaign's rerun adopts its persisted node jobs; a benchmark's rerun measures interrupted variants again.
        """
        if not self.run_id or self.run_kind not in {"campaign", "benchmark"}:
            return None
        if self.run_kind == "campaign" and self.authored_campaign:
            return shlex.join(["./scripts/generation", "status", self.run_id])
        restart = (*self.scientific_arguments, *self.restart_arguments) if self.run_kind == "campaign" else ()
        return shlex.join(["./scripts/generation", "run", self.config_argument, *restart])

    def _record_failure(self, error: BaseException, continuation: str | None) -> None:
        """Bind the exact failed stage and the continuation to an expected failure, which its CLI reports; source state remains service-owned.

        An unexpected exception is reported here, because it ends in a traceback.
        """
        if isinstance(error, GenerationControllerError):
            error.stage, error.next_command = self.stage, continuation
        elif continuation is not None:
            identity = [f"{self.run_kind}_run_id: {self.run_id}"] if self.run_id else []
            print("\n".join((f"FAILED: {self.stage}", *identity, f"Next: {continuation}")), file=sys.stderr)

    def status(self, target: str) -> str:
        """Compose read-only service status for a config or persisted run ID."""
        self.execute = False
        lines: list[str] = []
        candidate = Path(target)
        is_config = candidate.is_file() if candidate.is_absolute() else (self.port.repository / candidate).is_file()
        if is_config:
            # Status only reads: a failure here suggests no dry run of a run.
            plan = self._resolved_plan(target, allow_incomplete=True, scientific_arguments=())
            lines.extend((f"run_identity={plan.identity}", f"run_kind={plan.kind}", f"config={plan.config_path}"))
            if plan.kind == "campaign":
                _campaign_identifier(plan.identity)
                lines.append(json.dumps(self.json("campaign-status", plan.identity, *self._storage()), sort_keys=True))
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
            run_id = _campaign_identifier(target)
            lines.append(json.dumps(self.json("campaign-status", run_id, *self._storage()), sort_keys=True))
        return "\n".join(lines)

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
        if run_id.startswith("core_scaling_transient__"):
            _identifier(run_id, _RUN_ID, "benchmark run ID")
        else:
            _campaign_identifier(run_id)
        operation = "cancel-core-benchmark" if run_id.startswith("core_scaling_transient__") else "cancel-campaign"
        options = ("--force",) if force else ()
        return self.call(operation, run_id, *self._storage(), *options, scheduled=False)
