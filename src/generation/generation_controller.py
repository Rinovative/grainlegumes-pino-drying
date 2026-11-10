"""Coordinate Generation runs from authoritative service evidence.

Responsibilities:
  - Interpret immutable run plans and configured execution resources
  - Summarize the Slurm footprint of a run (node jobs and driver) for read-only previews
  - Sequence source campaign lifecycles
  - Reconcile repeat invocations, monitor work, and gate source publication
  - Report failure stages of a run, from plan resolution and preflight on, with the command that continues, preserving service-owned state
  - Print each stage with its duration and a live campaign board in the driver log
  - Have the host link the driver's logs into its campaign as soon as the campaign exists, also when its preparation failed

Design principles:
  - Existing Generation services own identities, manifests, and scientific data
  - Every continuation is reconstructed from persisted service evidence
  - Finalization follows exact source validation gates
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


_SCHEDULED_OPERATIONS = frozenset({"prepare-campaign-inputs", "resume-campaign"})


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
    """Validated execution fields from one campaign config."""

    partition: str
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
        node_jobs = _positive(cluster["node_jobs"], "node_jobs")
        return cls(
            partition=str(site["partition"]),
            comsol_module=str(site["comsol_module"]),
            python_executable=str(site["python_executable"]),
            comsol_executable=str(site["comsol_executable"]),
            poll_seconds=_positive(submission["poll_interval_seconds"], "poll_interval_seconds"),
            wall_time=_PYTHON_OPERATION_WALL_TIME,
            cores_per_node=cores_per_node,
            purpose=str(value["campaign_purpose"]),
            parallel_cases=node_jobs * (cores_per_node // cores_per_case),
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
        execution values.

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


def _require_continuable_state(run_id: str, state: object) -> None:
    """Stop with an actionable message for every campaign state the controller does not continue by itself."""
    if state == "license_wait_exhausted":
        _fail(f"Canonical campaign stopped: the licence wait budget of at least one case is exhausted; inspect status {run_id}")
    if state in {"blocked", "submission_unknown", "cancel_requested", "force_cancel_requested"}:
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
    def source_name(self) -> str:
        """Return the Source locator for a campaign result; a run id without campaign parts is refused."""
        parts = common.paths.campaign_run_parts(self.run_id)
        if parts is None:
            message = f"Run {self.run_id!r} is not a canonical campaign run and has no Source."
            raise ValueError(message)
        return parts[1]


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


def _unsettled_failure(run_id: str, status: Mapping[str, Any]) -> str | None:
    """Describe a quiescent source campaign that no recorded decision or replacement settles, else return ``None``.

    A failed case with a recorded terminal decision never stops the campaign: its slot is either covered by a
    replacement or left unfilled by a batch its minimum success rate stopped, which the status names.
    """
    cases, counts = status.get("cases"), status.get("counts")
    if not isinstance(cases, list) or not cases or not isinstance(counts, dict):
        _fail("Canonical campaign status lacks current candidate evidence")
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
        stop = status.get("success_rate_stop")
        return (
            f"Canonical campaign has {deficit} unfilled success slot(s) and no pending or running job. "
            f"{stop if isinstance(stop, str) else f'Inspect status {run_id}.'}"
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
            duration = common.console.format_duration(self.clock() - self._stage_started)
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
    ) -> str:
        """Use the node boundary for configured computational operations."""
        resources = self.resources
        if scheduled is None:
            scheduled = self.execute and operation in _SCHEDULED_OPERATIONS
        return self.port.call(
            operation,
            *arguments,
            scheduled=scheduled,
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
        return self.plan(str(plan_path), scientific_arguments=tuple(arguments))

    def choose_run_tree(self, plan: RunPlan, *, continue_with_current_code: bool) -> tuple[host_service.SourceAdmission, bool]:
        """
        Choose the source tree a run executes in and whether that is an explicit source transition.

        An existing campaign continues in the tree of its latest execution epoch
        unless ``continue_with_current_code`` asks for the current code, which the
        campaign then records as an explicit source transition. New campaigns and
        other run kinds use the current code.
        """
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
    ) -> LeafResult:
        """Run one in-memory campaign through the existing admitted controller.

        ``plan_path`` identifies the repository YAML preserved as recovery evidence.
        The authored mapping contains only Generation campaign values; an explicit
        admission cap is one of the ``execution_overrides``. A ``STAGED`` result
        retains incomplete bounded admission; failures raise the ordinary
        controller error and preserve the service-owned continuation evidence.
        """
        plan = self.authored_campaign_plan(
            plan_path, authored_value, execution_overrides=execution_overrides, scientific_overrides=scientific_overrides
        )
        return self.run(plan, str(plan_path))

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

    def _validate_site(self, resources: ExecutionResources) -> None:
        if resources.partition not in {"standard", "long"}:
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
            resources = self.campaign_resources(plan.config_path, override_arguments=overrides)
            self.port.verify_execution()
            version = self.port.comsol_version(resources.comsol_module, resources.comsol_executable)
        return f"PREFLIGHT COMPLETE: plan={plan.identity} kind={plan.kind} host=shared-filesystem COMSOL={version}"

    def run(self, plan: RunPlan, config_argument: str, *, max_admission_cases: int | None = None) -> LeafResult:
        """Execute or resume one immutable plan from service-owned evidence."""
        self.execute = True
        self.config_argument = config_argument
        self.restart_arguments = (
            ("--max-admission-cases", str(max_admission_cases))
            if max_admission_cases is not None and "max_admission_cases" not in plan.payload.get("execution_overrides", {})
            else ()
        )
        self.authored_campaign = plan.payload.get("authored_value") is not None
        self._stage_number = 0
        self._stage_started = None
        self._stage_total = _CAMPAIGN_STAGES
        self._enter_stage("common plan resolution", f"Run plan: kind={plan.kind} identity={plan.identity} config={plan.config_path}")
        try:
            if common.paths.campaign_run_parts(plan.identity) is None:
                _fail("Campaign plan is not a current source campaign", exit_code=2)
            result = self._canonical_campaign(plan, max_admission_cases=max_admission_cases)
        except BaseException as error:
            self._record_failure(error, self._continuation())
            raise
        self._leave_stage()
        if result.result == "STAGED":
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
        board = status_service.CampaignBoard(started=self.clock(), verify_seconds=_FAILURE_SETTLE_SECONDS)
        while manifest.get("state") != "complete":
            state = manifest.get("state")
            if self._cancellation_settling(run_id, state):
                time.sleep(resources.poll_seconds)
                manifest = self._resume_campaign(run_id)
                continue
            self._require_continuable(run_id, state)
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
                unsettled = _unsettled_failure(run_id, status)
                cases, counts = status["cases"], status["counts"]
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

    def _require_continuable(self, run_id: str, state: object) -> None:
        """Stop on every state the controller does not continue by itself; a blocked campaign names its stopped batches or its cap."""
        if state == "blocked":
            status = self.json("campaign-status", run_id, *self._storage())
            stops = [stop for stop in (status.get("success_rate_stop"), status.get("admission_cap_stop")) if isinstance(stop, str)]
            if stops:
                _fail(f"Canonical campaign is blocked. {' '.join(stops)}")
        _require_continuable_state(run_id, state)

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

    def _continuation(self) -> str | None:
        """Return the command that continues the identified run: its status for an authored campaign, else rerunning its command."""
        if not self.run_id:
            return None
        if self.authored_campaign:
            return shlex.join(["./scripts/generation", "status", self.run_id])
        return shlex.join(["./scripts/generation", "run", self.config_argument, *self.scientific_arguments, *self.restart_arguments])

    def _record_failure(self, error: BaseException, continuation: str | None) -> None:
        """Bind the exact failed stage and the continuation to an expected failure, which its CLI reports; source state remains service-owned.

        An unexpected exception is reported here, because it ends in a traceback.
        """
        if isinstance(error, GenerationControllerError):
            error.stage, error.next_command = self.stage, continuation
        elif continuation is not None:
            identity = [f"campaign_run_id: {self.run_id}"] if self.run_id else []
            print("\n".join((f"FAILED: {self.stage}", *identity, f"Next: {continuation}")), file=sys.stderr)

    def status(self, target: str) -> str:
        """Compose read-only service status for a config or persisted run ID."""
        self.execute = False
        candidate = Path(target)
        if candidate.is_file() if candidate.is_absolute() else (self.port.repository / candidate).is_file():
            # Status only reads: a failure here suggests no dry run of a run.
            plan = self._resolved_plan(target, allow_incomplete=True, scientific_arguments=())
            _campaign_identifier(plan.identity)
            return "\n".join(
                (
                    f"run_identity={plan.identity}",
                    f"run_kind={plan.kind}",
                    f"config={plan.config_path}",
                    json.dumps(self.json("campaign-status", plan.identity, *self._storage()), sort_keys=True),
                )
            )
        return json.dumps(self.json("campaign-status", _campaign_identifier(target), *self._storage()), sort_keys=True)

    def cancel(self, run_id: str, *, force: bool = False) -> str:
        """Delegate cancellation to the campaign state owner."""
        options = ("--force",) if force else ()
        return self.call("cancel-campaign", _campaign_identifier(run_id), *self._storage(), *options, scheduled=False)
