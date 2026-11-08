"""Coordinate Generation node-job Slurm commands.

Responsibilities:
  - Name the frozen source tree a node job runs from, submit the node job, and lease the tree with a visible result
  - Build the fail-closed submission of one node job per node-job slot and the node types it may run on
  - Derive the time limit and required node time of one case step ticket
  - Read the user's queued or running node jobs of the node-job slots and parse job-step accounting rows for dispatchers

Design principles:
  - Node jobs are submitted only from a frozen source tree, and a lease failure never loses a submitted job
  - A node job holds one generic worker; its submission identity is its slot's job name and a unique submission token
  - Scheduler queries fail loudly: an unreadable or malformed answer is never an empty answer
  - Production dispatch buffering is owned by the campaign feeder

This module does NOT:
  - Pack cases into nodes by itself, create Slurm arrays, or impose a cluster running cap
  - Decide when to poll the scheduler, persist feeder state, or implement scientific generation
  - Write tickets or decide which cases a node job runs
"""

from __future__ import annotations

import getpass
import math
import os
import re
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src import common
from src.generation.cases import generation_cases_config as config_service

from . import generation_runtime_batch as runtime_service
from . import generation_runtime_host as host_service
from . import generation_runtime_node as node_service

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from src.generation.cases import generation_cases_admission as admission_service
    from src.generation.cases import generation_cases_config as config_contract

_TERMINATION_SIGNAL_LEAD_SECONDS: Final = 120
CASE_PUBLICATION_MARGIN_SECONDS: Final = 600
_MAXIMUM_STEP_LIMIT_SECONDS: Final = 100 * 3600
_GIT_COMMIT_LENGTH: Final = 40
_SOURCE_SHA256_LENGTH: Final = 64
NODE_JOB_NAME_PREFIX: Final = "gen-seat-"  # with the slot number the fixed Slurm singleton job name of a node-job slot
_NODE_SEATS: Final = tuple(range(1, config_service.MAXIMUM_NODE_JOBS + 1))
_SUBMISSION_TOKEN = re.compile(r"[a-z0-9]{8,32}")
_PARTITION_NAME = re.compile(r"[A-Za-z0-9_-]+")
_NODE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SINFO_TIMEOUT_SECONDS: Final = 60
_SINFO_FIELDS: Final = 2
_SQUEUE_TIMEOUT_SECONDS: Final = 60
_SEAT_JOB_FIELDS: Final = 4
STEP_ACCOUNTING_FORMAT: Final = "JobIDRaw,JobName,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocCPUS,Partition,MaxRSS"
# Row keys of the fields that follow JobIDRaw and JobName in STEP_ACCOUNTING_FORMAT.
_STEP_ACCOUNTING_KEYS: Final = ("state", "exit_code", "submit", "start", "end", "elapsed", "node", "cpus", "partition", "max_rss")
_JOB_NUMBER = re.compile(r"[0-9]+")
_STEP_JOB_ID = re.compile(r"(?P<allocation>[0-9]+)\.(?P<step>[0-9]+)")
# Options the node job submission sets itself; scheduler_options may add nothing that overrides or duplicates them.
_NODE_OWNED_OPTIONS: Final = frozenset(
    {
        "--parsable",
        "--nodes",
        "--ntasks",
        "--cpus-per-task",
        "--mem",
        "--mem-per-cpu",
        "--time",
        "--partition",
        "--exclude",
        "--nodelist",
        "--job-name",
        "--dependency",
        "--comment",
        "--requeue",
        "--no-requeue",
        "--chdir",
        "--export",
        "--output",
        "--error",
        "--open-mode",
        "--signal",
        "--wrap",
        "--exclusive",
        "--gres",
        "--array",
    }
)


@dataclass(frozen=True, slots=True)
class CampaignTask:
    """One exact campaign case eligible for one independent Slurm job."""

    batch_name: str
    batch_id: str
    execution_key: str
    case_index: int
    case_id: str


def _positive_int(value: int, *, label: str) -> int:
    """Require one positive non-boolean integer resource value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{label} must be an integer >= 1, got {value!r}."
        raise ValueError(message)
    return value


def campaign_tasks(campaign: config_contract.CampaignConfig) -> tuple[CampaignTask, ...]:
    """Return every campaign case in deterministic batch and case order."""
    return tuple(
        CampaignTask(
            batch_name=batch.batch_name,
            batch_id=batch.batch_id,
            execution_key=common.paths.generation_execution_key(batch.batch_storage_name),
            case_index=case_index,
            case_id=batch.case_id(case_index),
        )
        for batch in campaign.batches
        for case_index in batch.case_indices
    )


def require_campaign_task(
    campaign: config_contract.CampaignConfig,
    *,
    batch_name: str,
    case_index: int,
) -> CampaignTask:
    """Resolve one exact campaign member without rebuilding all campaign tasks."""
    batch = campaign.batch(batch_name)
    case_id = batch.case_id(case_index)
    return CampaignTask(
        batch_name=batch.batch_name,
        batch_id=batch.batch_id,
        execution_key=common.paths.generation_execution_key(batch.batch_storage_name),
        case_index=case_index,
        case_id=case_id,
    )


def _batch_publication_markers_complete(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None,
) -> bool:
    """Return whether every configured case has an atomic completion marker."""
    return all(
        (
            runtime_service.processed_case_directory(
                config,
                case_index,
                storage_root=storage_root,
            )
            / "_SUCCESS"
        ).is_file()
        for case_index in config.case_indices
    )


def run_campaign_case(
    campaign: config_contract.CampaignConfig,
    task: CampaignTask,
    *,
    cores_per_case: int,
    scheduler_kind: str = "slurm",
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
    input_references: Mapping[int, admission_service.InputCaseReference] | None = None,
) -> runtime_service.CaseRunOutcome:
    """Materialize, solve, publish, and optionally finalize one campaign case."""
    expected = require_campaign_task(
        campaign,
        batch_name=task.batch_name,
        case_index=task.case_index,
    )
    if task != expected:
        message = "Campaign task identity changed after scheduler selection."
        raise ValueError(message)
    config = campaign.batch(task.batch_name)
    outcome = runtime_service.run_case(
        config,
        task.case_index,
        cores_per_case=cores_per_case,
        scheduler_kind=scheduler_kind,
        allocated_node=socket.gethostname(),
        storage_root=storage_root,
        work_root=work_root,
        blocking_lock=False,
        input_reference=None if input_references is None else input_references[task.case_index],
    )
    if _batch_publication_markers_complete(config, storage_root=storage_root):
        runtime_service.finalize_batch(config, storage_root=storage_root, input_references=input_references)
    return outcome


def executing_tree_id() -> str:
    """Return the frozen source tree this process executes from; node jobs are never submitted from live code."""
    tree_id = common.paths.source_tree_id()
    if tree_id is None:
        message = "Generation node jobs are submitted only from a frozen source tree; start the campaign through its launcher."
        raise RuntimeError(message)
    return tree_id


def lease_executing_tree(job_id: str) -> bool:
    """
    Lease the executing tree for one submitted job, so tree pruning keeps it while the job is queued or running.

    A failing lease prints one warning and returns False: the job is already submitted, so its ID must still be
    recorded, and the caller decides whether the unprotected job may run on.

    Returns
    -------
    bool
        Whether the lease was recorded.

    """
    return host_service.lease_submitted_job(
        common.paths.get_source_trees_root(), common.paths.get_project_root() / "scripts/source_fingerprint.py", executing_tree_id(), job_id
    )


def submit_node_job(command: Sequence[str], *, git_commit: str, environment: Mapping[str, str]) -> str:
    """
    Submit one typed node-job sbatch command from the executing source tree and return its parsable root job ID.

    The tree is not leased here: the dispatcher records the job first and then leases its tree with a visible result.

    Raises
    ------
    RuntimeError
        If the process does not run from a frozen source tree or Slurm returns an invalid job identifier.
    subprocess.CalledProcessError
        If sbatch refuses the submission.

    """
    executing_tree_id()
    submit_environment = {**os.environ, **environment, "GENERATION_GIT_COMMIT": git_commit}
    result = subprocess.run(  # noqa: S603 -- node job builder owns the Slurm argv
        list(command),
        check=True,
        capture_output=True,
        text=True,
        env=submit_environment,
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if _JOB_NUMBER.fullmatch(job_id) is None:
        message = f"Slurm returned an invalid parsable job identifier: {result.stdout!r}."
        raise RuntimeError(message)
    return job_id


def case_step_limits(timeout_seconds: float) -> tuple[str, int]:
    """
    Return the Slurm time limit and the required node time of one case step.

    A case step may use its runtime timeout plus the publication margin; the node worker starts it only when that
    time and the step launch wait are still left in the node job.

    Returns
    -------
    tuple[str, int]
        ``HH:MM:SS`` step time limit and the required remaining node-job seconds.

    Raises
    ------
    ValueError
        If ``timeout_seconds`` is not positive or the limit exceeds ``99:59:59``.

    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds < float("inf"):
        message = f"Case timeout must be a positive finite number of seconds, got {timeout_seconds!r}."
        raise ValueError(message)
    limit = math.ceil(timeout_seconds) + CASE_PUBLICATION_MARGIN_SECONDS
    if limit >= _MAXIMUM_STEP_LIMIT_SECONDS:
        message = f"Case step limit {limit} s exceeds the node ticket time format."
        raise ValueError(message)
    return f"{limit // 3600:02d}:{limit % 3600 // 60:02d}:{limit % 60:02d}", limit + node_service.STEP_START_WAIT_SECONDS


def source_worker_environment() -> list[str]:
    """
    Return the admitted-source bindings of a case worker.

    Returns
    -------
    list[str]
        ``KEY=value`` entries naming the admitted snapshot and, in source-tree mode, the
        frozen tree, its project root, and the runtime root. Empty when no snapshot was admitted.

    Raises
    ------
    ValueError
        If the snapshot evidence is incomplete or malformed, or absent in source-tree mode.

    """
    tree_id = common.paths.source_tree_id()
    snapshot = os.environ.get("GENERATION_SOURCE_SNAPSHOT")
    snapshot_sha = os.environ.get("GENERATION_SOURCE_SNAPSHOT_SHA256")
    if (snapshot is None) != (snapshot_sha is None):
        message = "Generation source snapshot admission is incomplete."
        raise ValueError(message)
    if snapshot is None or snapshot_sha is None:
        if tree_id is not None:
            message = "Source-tree workers require GENERATION_SOURCE_SNAPSHOT and GENERATION_SOURCE_SNAPSHOT_SHA256."
            raise ValueError(message)
        return []
    if (
        not Path(snapshot).is_absolute()
        or len(snapshot_sha) != _SOURCE_SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in snapshot_sha)
    ):
        message = "Generation source snapshot admission is malformed."
        raise ValueError(message)
    entries = [f"GENERATION_SOURCE_SNAPSHOT={snapshot}", f"GENERATION_SOURCE_SNAPSHOT_SHA256={snapshot_sha}"]
    if tree_id is not None:
        entries = [
            f"PROJECT_ROOT={common.paths.get_project_root()}",
            f"SOURCE_TREE_ID={tree_id}",
            f"RUNTIME_ROOT={common.paths.get_runtime_root()}",
            *entries,
        ]
    return entries


def _node_worker_environment(*, storage: Path, comsol_module: str, python_executable: str, comsol_executable: str) -> list[str]:
    """Return the admitted-source and site bindings of a node worker; per-ticket bindings travel in each ticket."""
    runtime_root = common.paths.get_runtime_root().resolve()
    source_commit = os.environ.get("GENERATION_GIT_COMMIT", "")
    source_sha = os.environ.get("GENERATION_SOURCE_SHA256", "")
    native_venv = os.environ.get("GENERATION_NATIVE_VENV", "")
    if len(source_commit) != _GIT_COMMIT_LENGTH or any(character not in "0123456789abcdef" for character in source_commit):
        message = "GENERATION_GIT_COMMIT must contain the exact launch commit."
        raise ValueError(message)
    if len(source_sha) != _SOURCE_SHA256_LENGTH or any(character not in "0123456789abcdef" for character in source_sha):
        message = "GENERATION_SOURCE_SHA256 must contain the launch source fingerprint."
        raise ValueError(message)
    if Path(native_venv) != runtime_root / "venvs" / "native":
        message = "GENERATION_NATIVE_VENV must be the runtime Generation venv RUNTIME_ROOT/venvs/native."
        raise ValueError(message)
    for label, value in (("comsol_module", comsol_module), ("python_executable", python_executable), ("comsol_executable", comsol_executable)):
        if not isinstance(value, str) or not value or any(character in value for character in ("\n", "\r", "\0")):
            message = f"Node job site value {label} must be one non-empty line, got {value!r}."
            raise ValueError(message)
    return [
        f"GENERATION_GIT_COMMIT={source_commit}",
        f"GENERATION_SOURCE_SHA256={source_sha}",
        f"GENERATION_NATIVE_VENV={native_venv}",
        f"STORAGE_ROOT={storage}",
        f"GENERATION_COMSOL_MODULE={comsol_module}",
        f"GENERATION_PYTHON_EXECUTABLE={python_executable}",
        f"GENERATION_COMSOL_EXECUTABLE={comsol_executable}",
        *source_worker_environment(),
    ]


def _validated_scheduler_options(options: Sequence[str]) -> list[str]:
    """Return extra sbatch options after rejecting every option the node job submission owns."""
    checked: list[str] = []
    for option in options:
        name = option.partition("=")[0] if isinstance(option, str) else ""
        if not name.startswith("--") or "\n" in option or "\0" in option or name in _NODE_OWNED_OPTIONS:
            message = f"Node job scheduler option is not an allowed long option: {option!r}."
            raise ValueError(message)
        checked.append(option)
    return checked


def node_job_log_templates(runtime_root: Path) -> tuple[Path, Path]:
    """Return the sbatch ``--output`` and ``--error`` templates of a node job, below ``RUNTIME_ROOT/logs/generation/``."""
    return runtime_root / "logs" / "generation" / "node-%j.out", runtime_root / "logs" / "generation" / "node-%j.err"


def build_node_job_submission_command(
    *,
    seat: int,
    submission_token: str,
    node_directory: Path,
    storage_root: Path,
    cores_per_node: int,
    node_job_memory: str,
    node_job_wall_time: str,
    partition: str,
    exclusions: Sequence[str],
    comsol_module: str,
    python_executable: str,
    comsol_executable: str,
    scheduler_options: Sequence[str] = (),
) -> list[str]:
    """
    Build the sbatch command of one node job that runs the generic node worker in one node-job slot.

    Parameters
    ----------
    seat : int
        Node-job slot from 1 to the per-user cap of node jobs; it fixes the job name ``gen-seat-<seat>`` which, with ``--dependency=singleton``,
        lets Slurm run at most one node job per slot and user.
    submission_token : str
        Unique ``[a-z0-9]{8,32}`` token stored as the job comment; the dispatcher recovers a lost submission by it.
    node_directory : Path
        Existing absolute node directory with the tickets the worker consumes.
    storage_root : Path
        Existing Generation storage root, separate from the repository and the runtime root.
    cores_per_node : int
        CPUs of the node job; it is both the request and the worker's core budget.
    node_job_memory, node_job_wall_time : str
        Slurm memory size (``<int>M|G|T``) and positive ``HH:MM:SS`` wall time of the node job.
    partition : str
        Slurm partition name.
    exclusions : Sequence[str]
        Node names the job must avoid (see ``standard_node_exclusions``); no ``--exclude`` is emitted when empty.
    comsol_module, python_executable, comsol_executable : str
        Site bindings the case steps load; the per-ticket ``GENERATION_ATTEMPT_INDEX`` is not part of the node job.
    scheduler_options : Sequence[str], optional
        Extra long options that none of the node job's own options may override.

    Returns
    -------
    list[str]
        The sbatch argument vector; the worker command is the last ``--wrap=`` argument.

    Raises
    ------
    ValueError, FileNotFoundError
        If any input, the launcher, or an admitted-source binding violates its contract.

    """
    if isinstance(seat, bool) or seat not in _NODE_SEATS:
        message = f"Node job seat must be one of {_NODE_SEATS}, got {seat!r}."
        raise ValueError(message)
    if not isinstance(submission_token, str) or _SUBMISSION_TOKEN.fullmatch(submission_token) is None:
        message = f"Node job submission token must match {_SUBMISSION_TOKEN.pattern!r}."
        raise ValueError(message)
    cores = _positive_int(cores_per_node, label="cores_per_node")
    memory = node_service.require_slurm_memory(node_job_memory, label="node_job_memory")
    wall_time = node_service.require_slurm_time(node_job_wall_time, label="node_job_wall_time")
    if not isinstance(partition, str) or _PARTITION_NAME.fullmatch(partition) is None:
        message = f"Node job partition must match {_PARTITION_NAME.pattern!r}, got {partition!r}."
        raise ValueError(message)
    excluded = sorted(set(exclusions))
    if any(not isinstance(name, str) or _NODE_NAME.fullmatch(name) is None for name in excluded):
        message = f"Node job exclusions must be node names, got {excluded!r}."
        raise ValueError(message)
    options = _validated_scheduler_options(scheduler_options)
    directory = Path(node_directory)
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        message = f"Node directory must be an existing absolute directory: {directory}."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    launcher = repository / "scripts" / "generation_node.sh"
    if not launcher.is_file() or launcher.is_symlink():
        message = f"Node compute-node launcher is missing or unsafe: {launcher}"
        raise FileNotFoundError(message)
    runtime_root = common.paths.get_runtime_root().resolve()
    storage = Path(storage_root).resolve()
    if not storage.is_dir() or storage in (repository, runtime_root):
        message = "Generation storage root must be one separate existing directory."
        raise ValueError(message)
    worker_environment = _node_worker_environment(
        storage=storage, comsol_module=comsol_module, python_executable=python_executable, comsol_executable=comsol_executable
    )
    wrapped = shlex.join(["env", *worker_environment, str(launcher), str(repository), "node-worker", str(directory), str(cores)])
    output_template, error_template = node_job_log_templates(runtime_root)
    return [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cores}",
        f"--mem={memory}",
        f"--time={wall_time}",
        f"--partition={partition}",
        *([f"--exclude={','.join(excluded)}"] if excluded else []),
        f"--job-name={NODE_JOB_NAME_PREFIX}{seat}",
        "--dependency=singleton",
        f"--comment={submission_token}",
        "--no-requeue",
        f"--chdir={repository}",
        "--export=ALL",
        f"--output={output_template}",
        f"--error={error_template}",
        # A rerun must never truncate a log that is hard-linked into its campaign.
        "--open-mode=append",
        f"--signal=B:TERM@{_TERMINATION_SIGNAL_LEAD_SECONDS}",
        *options,
        f"--wrap={wrapped}",
    ]


def _run_sinfo(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one sinfo query without a shell."""
    return subprocess.run(  # noqa: S603 -- fixed executable and controlled argument vector; no shell interpolation
        list(arguments),
        capture_output=True,
        text=True,
        check=False,
        timeout=_SINFO_TIMEOUT_SECONDS,
    )


def standard_node_exclusions(
    partition: str,
    cores_per_node: int,
    *,
    sinfo: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_sinfo,
) -> list[str]:
    """
    Return the nodes of ``partition`` that a node job must exclude because they have fewer CPUs than it requests.

    Larger nodes stay admitted: production jobs request ``cores_per_node`` CPUs without ``--exclusive``, so other jobs may use the
    remaining CPUs.

    Parameters
    ----------
    partition : str
        Slurm partition name.
    cores_per_node : int
        The CPUs per node the node job requests.
    sinfo : Callable, optional
        Runs the full sinfo argument vector and returns its completed process; replaced by tests.

    Returns
    -------
    list[str]
        Sorted unique node names with fewer than ``cores_per_node`` CPUs.

    Raises
    ------
    ValueError
        If an argument is invalid.
    RuntimeError
        If sinfo fails, its output is malformed, or no node has at least ``cores_per_node`` CPUs.

    """
    cores = _positive_int(cores_per_node, label="cores_per_node")
    if not isinstance(partition, str) or _PARTITION_NAME.fullmatch(partition) is None:
        message = f"Partition must match {_PARTITION_NAME.pattern!r}, got {partition!r}."
        raise ValueError(message)
    try:
        result = sinfo(["sinfo", "-h", "-p", partition, "-N", "-o", "%N|%c"])
    except (OSError, subprocess.SubprocessError) as error:
        message = f"sinfo could not be run for partition {partition}: {error}"
        raise RuntimeError(message) from error
    if result.returncode != 0:
        message = f"sinfo failed for partition {partition} with status {result.returncode}: {result.stderr.strip()}"
        raise RuntimeError(message)
    cpus_by_node: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if (
            len(fields) != _SINFO_FIELDS
            or _NODE_NAME.fullmatch(fields[0]) is None
            or not fields[1].isdigit()
            or cpus_by_node.get(fields[0], int(fields[1])) != int(fields[1])
        ):
            message = f"sinfo returned a malformed node line for partition {partition}: {line!r}"
            raise RuntimeError(message)
        cpus_by_node[fields[0]] = int(fields[1])
    if not any(cpus >= cores for cpus in cpus_by_node.values()):
        message = f"Partition {partition} has no node with at least {cores} CPUs."
        raise RuntimeError(message)
    return sorted(name for name, cpus in cpus_by_node.items() if cpus < cores)


def _run_squeue(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one squeue query without a shell."""
    return subprocess.run(  # noqa: S603 -- fixed executable and controlled argument vector; no shell interpolation
        list(arguments),
        capture_output=True,
        text=True,
        check=False,
        timeout=_SQUEUE_TIMEOUT_SECONDS,
    )


def query_seat_jobs(seats: Sequence[int]) -> dict[str, tuple[int, str, str]]:
    """
    Return this user's queued or running node jobs of the given node-job slots, whichever dispatcher submitted them.

    Parameters
    ----------
    seats : Sequence[int]
        Non-empty slot numbers (each one of the node-job slots); duplicates are ignored.

    Returns
    -------
    dict[str, tuple[int, str, str]]
        Job ID to ``(seat, state, comment)`` for every own non-terminal job named ``gen-seat-<seat>``; a slot without a
        row is free. The comment is the submission token of the node job.

    Raises
    ------
    ValueError
        If ``seats`` is empty or names a slot that does not exist.
    RuntimeError
        If squeue cannot be run, fails, or returns a malformed or duplicate row (unknown is not free).

    """
    if not seats:
        message = "At least one node-job slot is required."
        raise ValueError(message)
    if any(isinstance(seat, bool) or seat not in _NODE_SEATS for seat in seats):
        message = f"Node-job slots must be among {_NODE_SEATS}, got {list(seats)!r}."
        raise ValueError(message)
    wanted = sorted(set(seats))
    seat_by_name = {f"{NODE_JOB_NAME_PREFIX}{seat}": seat for seat in wanted}
    command = [
        "squeue",
        "--noheader",
        f"--user={getpass.getuser()}",
        f"--name={','.join(seat_by_name)}",
        "--format=%i|%j|%T|%k",
    ]
    try:
        result = _run_squeue(command)
    except (OSError, subprocess.SubprocessError) as error:
        message = f"Could not query the node jobs of seats {wanted}: {error}"
        raise RuntimeError(message) from error
    if result.returncode != 0:
        message = f"Could not query the node jobs of seats {wanted}: {result.stderr.strip() or f'exit status {result.returncode}'}"
        raise RuntimeError(message)
    jobs: dict[str, tuple[int, str, str]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|", _SEAT_JOB_FIELDS - 1)
        if len(fields) != _SEAT_JOB_FIELDS or _JOB_NUMBER.fullmatch(fields[0]) is None or fields[1] not in seat_by_name or not fields[2]:
            message = f"squeue returned a malformed node-job row: {line!r}"
            raise RuntimeError(message)
        if fields[0] in jobs:
            message = f"squeue returned node job {fields[0]} twice."
            raise RuntimeError(message)
        jobs[fields[0]] = (seat_by_name[fields[1]], fields[2], fields[3])
    return jobs


def _normalized_step_state(value: str) -> str:
    """Return the first word of one sacct state without its ``+`` suffix, e.g. ``CANCELLED by 123`` as ``CANCELLED``."""
    words = value.split()
    return words[0].split("+", maxsplit=1)[0] if words else ""


def parse_step_accounting(output: str) -> dict[tuple[str, str], dict[str, str]]:
    """
    Parse the job-step rows of one ``sacct --noheader --parsable2 --format=<STEP_ACCOUNTING_FORMAT>`` answer.

    Parameters
    ----------
    output : str
        The sacct stdout. Only rows whose JobIDRaw is ``<allocation>.<step>`` with numeric parts are steps, so
        allocation rows and the ``batch`` and ``extern`` steps are skipped.

    Returns
    -------
    dict[tuple[str, str], dict[str, str]]
        ``(allocation id, step name)`` to ``step_id``, ``state`` (normalized: ``CANCELLED by 123`` is ``CANCELLED``),
        ``exit_code``, ``submit``, ``start``, ``end``, ``elapsed``, ``node``, ``cpus``, ``partition``, and ``max_rss``
        (the raw MaxRSS text, empty while unknown). When one allocation holds several steps of the same name, the row
        of the highest step ID is kept.

    Raises
    ------
    ValueError
        If a step row does not have exactly the fields of ``STEP_ACCOUNTING_FORMAT``.

    """
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        identity = _STEP_JOB_ID.fullmatch(fields[0])
        if identity is None:
            continue
        if len(fields) != 2 + len(_STEP_ACCOUNTING_KEYS):
            message = f"sacct returned a malformed step row: {line!r}"
            raise ValueError(message)
        key = (identity["allocation"], fields[1])
        previous = rows.get(key)
        if previous is not None and int(previous["step_id"]) > int(identity["step"]):
            continue
        row = dict(zip(_STEP_ACCOUNTING_KEYS, fields[2:], strict=True))
        row["state"] = _normalized_step_state(row["state"])
        rows[key] = {"step_id": identity["step"], **row}
    return rows
