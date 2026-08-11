"""
===============================================================================
generation_runtime_cluster.py
===============================================================================
Coordinate local development concurrency and one-case Slurm submissions.
Responsibilities:
  - Validate a bounded local-only development execution plan
  - Run local cases without reusing production scheduler controls
  - Build one ordinary non-exclusive Slurm submission per campaign case
Design principles:
  - The scheduler owns cluster concurrency; each Slurm job owns exactly one case
  - Campaign job identity binds one declared batch and case before submission
  - Production submission buffering is owned by the campaign feeder
This module does NOT:
  - Pack cases into nodes, create Slurm arrays, or impose a cluster running cap
  - Poll the scheduler, persist feeder state, or implement scientific generation
===============================================================================
"""

from __future__ import annotations

import os
import shlex
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src import common

from . import generation_runtime_batch as runtime_service

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.generation.cases import generation_cases_config as config_contract

_MAX_SCHEDULER_JOB_NAME_LENGTH: Final = 48


@dataclass(frozen=True, slots=True)
class LocalResourcePlan:
    """Validated concurrency controls for the local development command only."""

    cores_per_case: int
    max_parallel_cases: int
    remaining_cases: int
    effective_parallel_cases: int


@dataclass(frozen=True, slots=True)
class CampaignTask:
    """One exact campaign case eligible for one independent Slurm job."""

    batch_name: str
    batch_id: str
    case_index: int
    case_id: str


def _positive_int(value: int, *, label: str) -> int:
    """Require one positive non-boolean integer resource value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{label} must be an integer >= 1, got {value!r}."
        raise ValueError(message)
    return value


def build_local_resource_plan(
    *,
    cores_per_case: int,
    max_parallel_cases: int,
    remaining_cases: int,
) -> LocalResourcePlan:
    """Build one local-only plan without cluster node-packing semantics."""
    cores = _positive_int(cores_per_case, label="cores_per_case")
    parallel = _positive_int(max_parallel_cases, label="max_parallel_cases")
    if isinstance(remaining_cases, bool) or not isinstance(remaining_cases, int) or remaining_cases < 0:
        message = f"remaining_cases must be a non-negative integer, got {remaining_cases!r}."
        raise ValueError(message)
    return LocalResourcePlan(
        cores_per_case=cores,
        max_parallel_cases=parallel,
        remaining_cases=remaining_cases,
        effective_parallel_cases=min(parallel, remaining_cases),
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


def run_local_batch(
    config: config_contract.GenerationConfig,
    selected_indices: Sequence[int],
    *,
    plan: LocalResourcePlan,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> tuple[runtime_service.CaseRunOutcome, ...]:
    """Run a bounded local development batch without cluster packing controls."""
    selected = tuple(selected_indices)
    if len(selected) != len(set(selected)) or any(index not in config.case_indices for index in selected):
        message = "Local batch selection must be duplicate-free configured case membership."
        raise ValueError(message)
    available_cores = os.cpu_count() or 1
    if plan.effective_parallel_cases * plan.cores_per_case > available_cores:
        message = (
            f"Local development execution would oversubscribe this host: {plan.effective_parallel_cases} * {plan.cores_per_case} > {available_cores}."
        )
        raise ValueError(message)
    if plan.effective_parallel_cases == 0:
        return ()
    outcomes: dict[int, runtime_service.CaseRunOutcome] = {}
    failures: list[tuple[int, BaseException]] = []
    with ThreadPoolExecutor(
        max_workers=plan.effective_parallel_cases,
        thread_name_prefix="local-generation-case",
    ) as executor:
        futures = {
            executor.submit(
                runtime_service.run_case,
                config,
                case_index,
                cores_per_case=plan.cores_per_case,
                worker_slot=slot % plan.effective_parallel_cases,
                scheduler_kind="local",
                storage_root=storage_root,
                work_root=work_root,
                blocking_lock=False,
            ): case_index
            for slot, case_index in enumerate(selected)
        }
        for future in as_completed(futures):
            case_index = futures[future]
            try:
                outcomes[case_index] = future.result()
            except Exception as error:  # noqa: BLE001 -- report independent local failures together
                failures.append((case_index, error))
    failure_limit = int(config.execution_values["runtime"]["maximum_failures"])
    if len(failures) >= failure_limit:
        details = "; ".join(f"{config.case_id(index)}: {error}" for index, error in sorted(failures))
        message = f"Local batch reached its failure limit after {len(failures)} case(s): {details}"
        raise RuntimeError(message) from failures[0][1]
    if selected == config.case_indices and all(
        runtime_service.completed_case_is_valid(config, case_index, storage_root=storage_root) for case_index in config.case_indices
    ):
        runtime_service.finalize_batch(config, storage_root=storage_root)
    return tuple(outcomes[index] for index in sorted(outcomes))


def campaign_tasks(campaign: config_contract.CampaignConfig) -> tuple[CampaignTask, ...]:
    """Return every campaign case in deterministic batch and case order."""
    return tuple(
        CampaignTask(
            batch_name=batch.batch_name,
            batch_id=batch.batch_id,
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
    """Resolve one exact campaign member without relying on material ordering."""
    matches = tuple(task for task in campaign_tasks(campaign) if task.batch_name == batch_name and task.case_index == case_index)
    if len(matches) != 1:
        message = f"Campaign task {batch_name!r}/{case_index!r} is not exact configured membership."
        raise ValueError(message)
    return matches[0]


def run_campaign_case(
    campaign: config_contract.CampaignConfig,
    task: CampaignTask,
    *,
    cores_per_case: int,
    scheduler_kind: str = "slurm",
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
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
    )
    if all(runtime_service.completed_case_is_valid(config, case_index, storage_root=storage_root) for case_index in config.case_indices):
        runtime_service.finalize_batch(config, storage_root=storage_root)
    return outcome


def build_campaign_case_slurm_submission_command(
    campaign: config_contract.CampaignConfig,
    task: CampaignTask,
    *,
    run_id: str,
    scheduler_log_directory: Path,
    scheduler_job_name: str,
) -> list[str]:
    """Build one ordinary non-exclusive Slurm job for one exact campaign case."""
    if campaign.execution_values["cluster"]["scheduler_kind"] != "slurm":
        message = "Campaign Slurm submission requires scheduler_kind='slurm'."
        raise ValueError(message)
    expected = require_campaign_task(
        campaign,
        batch_name=task.batch_name,
        case_index=task.case_index,
    )
    if task != expected:
        message = "Campaign submission task identity is inconsistent."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    launcher = repository / "scripts" / "generation_campaign_node.sh"
    if not launcher.is_file() or launcher.is_symlink():
        message = f"Campaign compute-node launcher is missing or unsafe: {launcher}"
        raise FileNotFoundError(message)
    log_directory = Path(scheduler_log_directory)
    if not log_directory.is_absolute() or log_directory.is_symlink() or (log_directory.exists() and not log_directory.is_dir()):
        message = f"Scheduler log directory must be one safe absolute directory: {log_directory}."
        raise ValueError(message)
    job_name = common.paths.validate_logical_name(
        scheduler_job_name,
        label="scheduler_job_name",
    )
    if len(job_name) > _MAX_SCHEDULER_JOB_NAME_LENGTH:
        message = "scheduler_job_name must contain at most 48 characters."
        raise ValueError(message)
    cluster = campaign.execution_values["cluster"]
    cores_per_case = int(cluster["cores_per_case"])
    site = campaign.execution_values["site"]
    worker_environment = [
        f"GENERATION_PYTHON_MODULE={site['python_module']}",
        f"GENERATION_COMSOL_MODULE={site['comsol_module']}",
        f"GENERATION_PYTHON_EXECUTABLE={site['python_executable']}",
        f"GENERATION_COMSOL_EXECUTABLE={site['comsol_executable']}",
    ]
    worker_command = [
        str(launcher),
        run_id,
        task.batch_name,
        str(task.case_index),
        str(cores_per_case),
    ]
    wrapped = shlex.join(["env", *worker_environment, *worker_command])
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cores_per_case}",
        f"--chdir={repository}",
        f"--job-name={job_name}",
        "--export=ALL",
        f"--output={log_directory}/slurm-%j.out",
        f"--error={log_directory}/slurm-%j.err",
    ]
    if cluster["partition"] is not None:
        command.append(f"--partition={cluster['partition']}")
    if cluster["wall_time"] is not None:
        command.append(f"--time={cluster['wall_time']}")
    command.extend(cluster["scheduler_options"])
    command.append(f"--wrap={wrapped}")
    return command
