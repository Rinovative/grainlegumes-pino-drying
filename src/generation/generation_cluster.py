"""
===============================================================================
generation_cluster.py
===============================================================================
Coordinate bounded case concurrency within and across CPU-cluster nodes.
Responsibilities:
  - Validate node, process, core, and global case-concurrency limits
  - Run bounded node workers with lock-protected shared case claims
  - Generate explicit scheduler submission commands and a local development path
Design principles:
  - The scheduler owns physical node allocation
  - Filesystem locks enforce node, case, and global-slot caps on shared storage
  - One COMSOL case always remains inside one node-owned subprocess step
This module does NOT:
  - Use SSH fan-out, MPI hostfiles, COMSOL ``-nn``, or login-node production fan-out
  - Guess scheduler account, partition, time, or site-specific directives
===============================================================================
"""

from __future__ import annotations

import math
import os
import shlex
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src import common

from . import generation_config as config_contract
from . import generation_runtime as runtime_service

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_MAX_SCHEDULER_JOB_NAME_LENGTH: Final = 48


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """Validated resource limits and effective concurrency for one selection."""

    max_nodes: int
    cases_per_node: int
    cores_per_case: int
    max_parallel_cases: int
    cores_per_node: int
    remaining_cases: int
    effective_parallel_cases: int
    effective_nodes: int


@dataclass(frozen=True, slots=True)
class NodeWorkerResult:
    """One node worker's case outcomes in deterministic selection order."""

    worker_index: int
    hostname: str
    outcomes: tuple[runtime_service.CaseRunOutcome, ...]
    claimed_elsewhere: tuple[str, ...]


def _positive_int(value: int, *, label: str) -> int:
    """Require one positive non-boolean integer resource value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{label} must be an integer >= 1, got {value!r}."
        raise ValueError(message)
    return value


def build_resource_plan(
    *,
    max_nodes: int,
    cases_per_node: int,
    cores_per_case: int,
    max_parallel_cases: int,
    cores_per_node: int,
    remaining_cases: int,
) -> ResourcePlan:
    """Validate mandatory resource controls and derive exact effective bounds."""
    max_nodes = _positive_int(max_nodes, label="max_nodes")
    cases_per_node = _positive_int(cases_per_node, label="cases_per_node")
    cores_per_case = _positive_int(cores_per_case, label="cores_per_case")
    max_parallel_cases = _positive_int(max_parallel_cases, label="max_parallel_cases")
    cores_per_node = _positive_int(cores_per_node, label="cores_per_node")
    if isinstance(remaining_cases, bool) or not isinstance(remaining_cases, int) or remaining_cases < 0:
        message = f"remaining_cases must be a non-negative integer, got {remaining_cases!r}."
        raise ValueError(message)
    if cases_per_node * cores_per_case > cores_per_node:
        message = f"cases_per_node * cores_per_case must not exceed cores_per_node: {cases_per_node} * {cores_per_case} > {cores_per_node}."
        raise ValueError(message)
    if max_parallel_cases > max_nodes * cases_per_node:
        message = f"max_parallel_cases must not exceed max_nodes * cases_per_node: {max_parallel_cases} > {max_nodes} * {cases_per_node}."
        raise ValueError(message)
    effective_parallel = min(max_parallel_cases, max_nodes * cases_per_node, remaining_cases)
    effective_nodes = 0 if effective_parallel == 0 else math.ceil(effective_parallel / cases_per_node)
    if effective_nodes > max_nodes:
        message = "Derived effective node count exceeds the configured hard maximum."
        raise RuntimeError(message)
    return ResourcePlan(
        max_nodes=max_nodes,
        cases_per_node=cases_per_node,
        cores_per_case=cores_per_case,
        max_parallel_cases=max_parallel_cases,
        cores_per_node=cores_per_node,
        remaining_cases=remaining_cases,
        effective_parallel_cases=effective_parallel,
        effective_nodes=effective_nodes,
    )


def select_case_indices(
    config: config_contract.GenerationConfig,
    *,
    case_start: int | None = None,
    case_stop: int | None = None,
) -> tuple[int, ...]:
    """Return one inclusive configured case-index range in batch order."""
    if case_start is None and case_stop is None:
        return config.case_indices
    start = config.case_indices[0] if case_start is None else case_start
    stop = config.case_indices[-1] if case_stop is None else case_stop
    if isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int) or start > stop:
        message = f"Selected case range must be ordered integer bounds, got {start!r}:{stop!r}."
        raise ValueError(message)
    selected = tuple(index for index in config.case_indices if start <= index <= stop)
    if not selected or selected[0] != start or selected[-1] != stop:
        message = f"Selected range {start}:{stop} must use configured case-index endpoints."
        raise ValueError(message)
    return selected


@contextmanager
def _global_case_slot(
    config: config_contract.GenerationConfig,
    plan: ResourcePlan,
    *,
    storage_root: Path | str | None,
) -> Iterator[int]:
    """Hold one of the batch's hard global concurrent-case slots."""
    slots_root = runtime_service._state_batch_root(config, storage_root=storage_root) / "slots"  # noqa: SLF001
    slots_root.mkdir(parents=True, exist_ok=True)
    while True:
        for slot in range(plan.max_parallel_cases):
            manager = common.locking.exclusive_file_lock(slots_root / f"slot_{slot:04d}.lock", blocking=False)
            try:
                manager.__enter__()
            except common.locking.FileLockUnavailableError:
                continue
            try:
                yield slot
            finally:
                manager.__exit__(None, None, None)
            return
        time.sleep(0.05)


def _attempt_case(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    plan: ResourcePlan,
    worker_slot: int,
    scheduler_kind: str,
    hostname: str,
    storage_root: Path | str | None,
    work_root: Path | str | None,
) -> runtime_service.CaseRunOutcome | None:
    """Run or claim one case while the authoritative runtime owns its lock."""
    if runtime_service.completed_case_is_valid(config, case_index, storage_root=storage_root):
        try:
            return runtime_service.run_case(
                config,
                case_index,
                cores_per_case=plan.cores_per_case,
                worker_slot=worker_slot,
                scheduler_kind=scheduler_kind,
                allocated_node=hostname,
                storage_root=storage_root,
                work_root=work_root,
                blocking_lock=False,
            )
        except common.locking.FileLockUnavailableError:
            return None
    try:
        with _global_case_slot(config, plan, storage_root=storage_root) as global_slot:
            return runtime_service.run_case(
                config,
                case_index,
                cores_per_case=plan.cores_per_case,
                worker_slot=global_slot,
                scheduler_kind=scheduler_kind,
                allocated_node=hostname,
                storage_root=storage_root,
                work_root=work_root,
                blocking_lock=False,
            )
    except common.locking.FileLockUnavailableError:
        return None


def run_node_worker(
    config: config_contract.GenerationConfig,
    selected_indices: Sequence[int],
    *,
    plan: ResourcePlan,
    worker_index: int,
    worker_count: int,
    assignment_mode: str = "shared",
    scheduler_kind: str = "local",
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> NodeWorkerResult:
    """Run at most ``cases_per_node`` independent case processes on one node."""
    if plan.effective_nodes == 0:
        return NodeWorkerResult(worker_index=worker_index, hostname=socket.gethostname(), outcomes=(), claimed_elsewhere=())
    if worker_count != plan.effective_nodes:
        message = f"worker_count must equal the effective node count {plan.effective_nodes}, got {worker_count}."
        raise ValueError(message)
    if isinstance(worker_index, bool) or not isinstance(worker_index, int) or not 0 <= worker_index < worker_count:
        message = f"worker_index must lie in [0, {worker_count}), got {worker_index!r}."
        raise ValueError(message)
    if worker_count > plan.max_nodes:
        message = f"Node worker count {worker_count} exceeds max_nodes={plan.max_nodes}."
        raise ValueError(message)
    if assignment_mode not in {"shared", "deterministic"}:
        message = "assignment_mode must be 'shared' or 'deterministic'."
        raise ValueError(message)
    if scheduler_kind not in {"local", "slurm"}:
        message = "scheduler_kind must be 'local' or 'slurm'."
        raise ValueError(message)
    selected = tuple(selected_indices)
    if len(selected) != len(set(selected)) or any(index not in config.case_indices for index in selected):
        message = "Node worker selection must be duplicate-free configured case membership."
        raise ValueError(message)
    candidates = selected if assignment_mode == "shared" else selected[worker_index::worker_count]
    hostname = socket.gethostname()
    state_root = runtime_service._state_batch_root(config, storage_root=storage_root)  # noqa: SLF001
    node_lock = state_root / "nodes" / f"worker_{worker_index:04d}.lock"
    outcomes_by_index: dict[int, runtime_service.CaseRunOutcome] = {}
    claimed: list[str] = []
    failures: list[tuple[int, BaseException]] = []
    with (
        common.locking.exclusive_file_lock(node_lock, blocking=False),
        ThreadPoolExecutor(max_workers=plan.cases_per_node, thread_name_prefix=f"node-{worker_index}-case") as executor,
    ):
        futures = {
            executor.submit(
                _attempt_case,
                config,
                case_index,
                plan=plan,
                worker_slot=local_slot % plan.cases_per_node,
                scheduler_kind=scheduler_kind,
                hostname=hostname,
                storage_root=storage_root,
                work_root=work_root,
            ): case_index
            for local_slot, case_index in enumerate(candidates)
        }
        for future in as_completed(futures):
            case_index = futures[future]
            try:
                outcome = future.result()
            except Exception as error:  # noqa: BLE001 -- aggregate independent case failures
                failures.append((case_index, error))
                continue
            if outcome is None:
                claimed.append(config.case_id(case_index))
            else:
                outcomes_by_index[case_index] = outcome
    failure_limit = int(config.execution_values["runtime"]["maximum_failures"])
    if len(failures) >= failure_limit:
        details = "; ".join(f"{config.case_id(index)}: {error}" for index, error in sorted(failures))
        message = f"Node worker {worker_index} reached its failure limit after {len(failures)} case(s): {details}"
        raise RuntimeError(message) from failures[0][1]
    if selected == config.case_indices and all(
        runtime_service.completed_case_is_valid(config, case_index, storage_root=storage_root) for case_index in config.case_indices
    ):
        runtime_service.finalize_batch(config, storage_root=storage_root)
    return NodeWorkerResult(
        worker_index=worker_index,
        hostname=hostname,
        outcomes=tuple(outcomes_by_index[index] for index in sorted(outcomes_by_index)),
        claimed_elsewhere=tuple(sorted(claimed)),
    )


def run_local_batch(
    config: config_contract.GenerationConfig,
    selected_indices: Sequence[int],
    *,
    plan: ResourcePlan,
    assignment_mode: str = "shared",
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> tuple[NodeWorkerResult, ...]:
    """Run the scheduler-neutral development fallback with the same hard limits."""
    available_cores = os.cpu_count() or 1
    if plan.cases_per_node * plan.cores_per_case > available_cores:
        message = (
            f"Local development execution exceeds available CPU cores per worker: {plan.cases_per_node} * {plan.cores_per_case} > {available_cores}."
        )
        raise ValueError(message)
    if plan.effective_parallel_cases * plan.cores_per_case > available_cores:
        message = (
            f"Local development execution would oversubscribe this host: {plan.effective_parallel_cases} * {plan.cores_per_case} > {available_cores}."
        )
        raise ValueError(message)
    if plan.effective_nodes == 0:
        return ()
    with ThreadPoolExecutor(max_workers=plan.effective_nodes, thread_name_prefix="local-node") as executor:
        futures = [
            executor.submit(
                run_node_worker,
                config,
                selected_indices,
                plan=plan,
                worker_index=worker_index,
                worker_count=plan.effective_nodes,
                assignment_mode=assignment_mode,
                scheduler_kind="local",
                storage_root=storage_root,
                work_root=work_root,
            )
            for worker_index in range(plan.effective_nodes)
        ]
        results = tuple(future.result() for future in futures)
    if tuple(selected_indices) == config.case_indices:
        runtime_service.finalize_batch(config, storage_root=storage_root)
    return results


@dataclass(frozen=True, slots=True)
class CampaignTask:
    """One case task identified inside a predeclared campaign batch."""

    batch_name: str
    case_index: int


@dataclass(frozen=True, slots=True)
class CampaignWorkerResult:
    """One campaign worker's outcomes across material/regime batches."""

    worker_index: int
    hostname: str
    completed_tasks: tuple[CampaignTask, ...]
    claimed_elsewhere: tuple[CampaignTask, ...]


def campaign_tasks(campaign: config_contract.CampaignConfig) -> tuple[CampaignTask, ...]:
    """Return all campaign cases in declarative batch and case order."""
    return tuple(CampaignTask(batch_name=batch.batch_name, case_index=case_index) for batch in campaign.batches for case_index in batch.case_indices)


def _campaign_state_root(
    campaign: config_contract.CampaignConfig,
    *,
    storage_root: Path | str | None,
) -> Path:
    """Return one persistent state root shared by every campaign subbatch."""
    return common.paths.get_generation_state_root(storage_root=storage_root) / "campaigns" / campaign.campaign_id


@contextmanager
def _campaign_global_case_slot(
    campaign: config_contract.CampaignConfig,
    plan: ResourcePlan,
    *,
    storage_root: Path | str | None,
) -> Iterator[int]:
    """Hold one campaign-wide concurrent-case slot across all subbatches."""
    slots_root = _campaign_state_root(campaign, storage_root=storage_root) / "slots"
    slots_root.mkdir(parents=True, exist_ok=True)
    while True:
        for slot in range(plan.max_parallel_cases):
            manager = common.locking.exclusive_file_lock(
                slots_root / f"slot_{slot:04d}.lock",
                blocking=False,
            )
            try:
                manager.__enter__()
            except common.locking.FileLockUnavailableError:
                continue
            try:
                yield slot
            finally:
                manager.__exit__(None, None, None)
            return
        time.sleep(0.05)


def _attempt_campaign_task(
    campaign: config_contract.CampaignConfig,
    task: CampaignTask,
    *,
    plan: ResourcePlan,
    scheduler_kind: str,
    hostname: str,
    storage_root: Path | str | None,
    work_root: Path | str | None,
) -> runtime_service.CaseRunOutcome | None:
    """Run one campaign task while the runtime owns its authoritative lock."""
    config = campaign.batch(task.batch_name)
    if runtime_service.runtime_cancellation_requested():
        return None
    if campaign.campaign_purpose == config_contract.PILOT_CAMPAIGN_PURPOSE and runtime_service.case_failure_is_recorded(
        config,
        task.case_index,
        storage_root=storage_root,
    ):
        return None
    if runtime_service.completed_case_is_valid(
        config,
        task.case_index,
        storage_root=storage_root,
    ):
        try:
            return runtime_service.run_case(
                config,
                task.case_index,
                cores_per_case=plan.cores_per_case,
                scheduler_kind=scheduler_kind,
                allocated_node=hostname,
                storage_root=storage_root,
                work_root=work_root,
                blocking_lock=False,
            )
        except common.locking.FileLockUnavailableError:
            return None
    try:
        with _campaign_global_case_slot(
            campaign,
            plan,
            storage_root=storage_root,
        ) as campaign_slot:
            if campaign.campaign_purpose == config_contract.PILOT_CAMPAIGN_PURPOSE and runtime_service.case_failure_is_recorded(
                config,
                task.case_index,
                storage_root=storage_root,
            ):
                return None
            return runtime_service.run_case(
                config,
                task.case_index,
                cores_per_case=plan.cores_per_case,
                worker_slot=campaign_slot,
                scheduler_kind=scheduler_kind,
                allocated_node=hostname,
                storage_root=storage_root,
                work_root=work_root,
                blocking_lock=False,
            )
    except common.locking.FileLockUnavailableError:
        return None


def run_campaign_worker(
    campaign: config_contract.CampaignConfig,
    *,
    plan: ResourcePlan,
    worker_index: int,
    worker_count: int,
    scheduler_kind: str = "slurm",
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> CampaignWorkerResult:
    """Run one node-confined worker from the shared campaign case pool."""
    if worker_count != plan.effective_nodes or not 0 <= worker_index < worker_count:
        message = "Campaign worker index/count must match the effective campaign node plan."
        raise ValueError(message)
    if worker_count > plan.max_nodes:
        message = "Campaign worker count exceeds the campaign-wide max_nodes cap."
        raise ValueError(message)
    runtime_service.reset_runtime_cancellation()
    tasks = campaign_tasks(campaign)
    hostname = socket.gethostname()
    node_lock = _campaign_state_root(campaign, storage_root=storage_root) / "nodes" / f"worker_{worker_index:04d}.lock"
    completed: list[CampaignTask] = []
    claimed: list[CampaignTask] = []
    failures: list[tuple[CampaignTask, BaseException]] = []
    with (
        common.locking.exclusive_file_lock(node_lock, blocking=False),
        ThreadPoolExecutor(
            max_workers=plan.cases_per_node,
            thread_name_prefix=f"campaign-node-{worker_index}",
        ) as executor,
    ):
        futures = {
            executor.submit(
                _attempt_campaign_task,
                campaign,
                task,
                plan=plan,
                scheduler_kind=scheduler_kind,
                hostname=hostname,
                storage_root=storage_root,
                work_root=work_root,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as error:  # noqa: BLE001 -- aggregate independent task failures
                failures.append((task, error))
                continue
            if outcome is None:
                claimed.append(task)
            else:
                completed.append(task)
    if runtime_service.runtime_cancellation_requested():
        message = f"Campaign worker {worker_index} was interrupted and remains resumable."
        raise InterruptedError(message)
    failure_limit = int(campaign.execution_values["runtime"]["maximum_failures"])
    if len(failures) >= failure_limit:
        details = "; ".join(f"{task.batch_name}/{task.case_index}: {error}" for task, error in failures)
        message = f"Campaign worker {worker_index} reached its failure limit after {len(failures)} task(s): {details}"
        raise RuntimeError(message) from failures[0][1]
    for config in campaign.batches:
        if config.case_indices and all(
            runtime_service.completed_case_is_valid(
                config,
                case_index,
                storage_root=storage_root,
            )
            for case_index in config.case_indices
        ):
            runtime_service.finalize_batch(config, storage_root=storage_root)
    return CampaignWorkerResult(
        worker_index=worker_index,
        hostname=hostname,
        completed_tasks=tuple(completed),
        claimed_elsewhere=tuple(claimed),
    )


def build_campaign_slurm_submission_command(
    campaign: config_contract.CampaignConfig,
    *,
    plan: ResourcePlan,
    scheduler_log_directory: Path | None = None,
    scheduler_job_name: str | None = None,
) -> list[str]:
    """Build the one shared Slurm worker-pool submission for a campaign."""
    if campaign.execution_values["cluster"]["scheduler_kind"] != "slurm":
        message = "Campaign Slurm submission requires scheduler_kind='slurm'."
        raise ValueError(message)
    if plan.effective_nodes < 1:
        message = "No Slurm nodes are required for an empty or completed campaign."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    launcher = repository / "scripts" / "generation_campaign_node.sh"
    worker_command = [
        str(launcher),
        str(campaign.source_path),
        "--max-nodes",
        str(plan.max_nodes),
        "--cases-per-node",
        str(plan.cases_per_node),
        "--cores-per-case",
        str(plan.cores_per_case),
        "--max-parallel-cases",
        str(plan.max_parallel_cases),
        "--cores-per-node",
        str(plan.cores_per_node),
        "--remaining-cases",
        str(plan.remaining_cases),
    ]
    if campaign.campaign_purpose == config_contract.PILOT_CAMPAIGN_PURPOSE:
        case_counts = {len(batch.case_indices) for batch in campaign.batches}
        if len(case_counts) != 1:
            message = "Pilot worker construction requires one uniform cases-per-material count."
            raise ValueError(message)
        worker_command.extend(["--pilot-cases-per-material", str(next(iter(case_counts)))])
    elif len(campaign.batches) == 1:
        worker_command.extend(["--only-batch", campaign.batches[0].batch_name])
    elif campaign.campaign_purpose == "family_generalization" and not any(
        batch.evaluation_regime == "extreme_family_ood" for batch in campaign.batches
    ):
        worker_command.append("--skip-extreme-family-ood")
    wrapped = f"CAMPAIGN_WORKER_COUNT={plan.effective_nodes} {shlex.join(worker_command)}"
    cluster = campaign.execution_values["cluster"]
    job_name = campaign.campaign_name[:_MAX_SCHEDULER_JOB_NAME_LENGTH] if scheduler_job_name is None else scheduler_job_name
    common.paths.validate_logical_name(job_name, label="scheduler_job_name")
    if len(job_name) > _MAX_SCHEDULER_JOB_NAME_LENGTH:
        message = "scheduler_job_name must contain at most 48 characters."
        raise ValueError(message)
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={plan.cases_per_node * plan.cores_per_case}",
        f"--array=0-{plan.effective_nodes - 1}%{plan.effective_nodes}",
        f"--chdir={repository}",
        f"--job-name={job_name}",
        "--export=ALL",
    ]
    if scheduler_log_directory is not None:
        log_directory = Path(scheduler_log_directory)
        if not log_directory.is_absolute() or log_directory.is_symlink() or (log_directory.exists() and not log_directory.is_dir()):
            message = f"Scheduler log directory must be one safe absolute directory: {log_directory}."
            raise ValueError(message)
        command.extend(
            [
                f"--output={log_directory}/slurm-%A_%a.out",
                f"--error={log_directory}/slurm-%A_%a.err",
            ]
        )
    if cluster["partition"] is not None:
        command.append(f"--partition={cluster['partition']}")
    if cluster["wall_time"] is not None:
        command.append(f"--time={cluster['wall_time']}")
    command.extend(cluster["scheduler_options"])
    command.append(f"--wrap={wrapped}")
    return command
