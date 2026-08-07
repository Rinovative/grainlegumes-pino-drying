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
from typing import TYPE_CHECKING

from src import common

from . import generation_config as config_contract
from . import generation_runtime as runtime_service

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path


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
    """Claim and run one case once, returning none when another worker owns it."""
    lock_path = runtime_service.case_lock_path(config, case_index, storage_root=storage_root)
    try:
        with common.locking.exclusive_file_lock(lock_path, blocking=False):
            if runtime_service.completed_case_is_valid(config, case_index, storage_root=storage_root):
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
    if failures:
        details = "; ".join(f"{config.case_id(index)}: {error}" for index, error in sorted(failures))
        message = f"Node worker {worker_index} failed {len(failures)} case(s): {details}"
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


def build_slurm_submission_command(
    config: config_contract.GenerationConfig,
    selected_indices: Sequence[int],
    *,
    plan: ResourcePlan,
) -> list[str]:
    """Build one dry-run Slurm submission argument vector for node-owned workers."""
    if config.values["cluster"]["scheduler_kind"] != "slurm":
        message = "Slurm command generation requires cluster.scheduler_kind='slurm'."
        raise ValueError(message)
    if plan.effective_nodes < 1:
        message = "No Slurm nodes are required because the selected cases are already complete or empty."
        raise ValueError(message)
    selected = tuple(selected_indices)
    repository = common.paths.get_project_root().resolve()
    cpus_per_worker = plan.cases_per_node * plan.cores_per_case
    launcher = repository / "scripts" / "generation_node.sh"
    worker_command = [
        str(launcher),
        str(config.source_path),
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
        "--case-start",
        str(selected[0]),
        "--case-stop",
        str(selected[-1]),
    ]
    wrapped = f"NODE_WORKER_COUNT={plan.effective_nodes} {shlex.join(worker_command)}"
    return [
        "sbatch",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus_per_worker}",
        f"--array=0-{plan.effective_nodes - 1}%{plan.effective_nodes}",
        f"--chdir={repository}",
        f"--job-name={config.batch_id[:48]}",
        *config.values["cluster"]["scheduler_options"],
        f"--wrap={wrapped}",
    ]
