"""
generation_runtime_cluster.py

Coordinate local development concurrency and Generation case Slurm commands.
Responsibilities:
  - Validate a bounded local-only development execution plan
  - Run local cases without reusing production scheduler controls
  - Build ordinary non-exclusive Slurm submissions for campaign and benchmark cases
Design principles:
  - The scheduler owns cluster concurrency; each Slurm job owns exactly one case
  - Campaign job identity binds one declared batch and case before submission
  - Production submission buffering is owned by the campaign feeder
This module does NOT:
  - Pack cases into nodes, create Slurm arrays, or impose a cluster running cap
  - Poll the scheduler, persist feeder state, or implement scientific generation
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
    from collections.abc import Mapping, Sequence

    from src.generation.cases import generation_cases_config as config_contract

_MAX_SCHEDULER_JOB_NAME_LENGTH: Final = 48
_GIT_COMMIT_LENGTH: Final = 40
_SOURCE_SHA256_LENGTH: Final = 64


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
    failure_limit = int(config.execution_values["runtime"]["maximum_failed_cases"])
    if len(failures) > failure_limit:
        details = "; ".join(f"{config.case_id(index)}: {error}" for index, error in sorted(failures))
        message = f"Local batch reached its failure limit after {len(failures)} case(s): {details}"
        raise RuntimeError(message) from failures[0][1]
    if selected == config.case_indices and _batch_publication_markers_complete(
        config,
        storage_root=storage_root,
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
    """Resolve one exact campaign member without rebuilding all campaign tasks."""
    batch = campaign.batch(batch_name)
    case_id = batch.case_id(case_index)
    return CampaignTask(
        batch_name=batch.batch_name,
        batch_id=batch.batch_id,
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
    if _batch_publication_markers_complete(config, storage_root=storage_root):
        runtime_service.finalize_batch(config, storage_root=storage_root)
    return outcome


def build_campaign_case_slurm_submission_command(
    campaign: config_contract.CampaignConfig,
    task: CampaignTask,
    *,
    run_id: str,
    storage_root: Path,
    scheduler_log_directory: Path,
    scheduler_job_name: str,
    attempt_index: int,
    task_index: Mapping[tuple[str, int], CampaignTask] | None = None,
) -> list[str]:
    """Build one ordinary non-exclusive Slurm job for one exact campaign case."""
    if campaign.execution_values["cluster"]["scheduler_kind"] != "slurm":
        message = "Campaign Slurm submission requires scheduler_kind='slurm'."
        raise ValueError(message)
    expected = (
        require_campaign_task(
            campaign,
            batch_name=task.batch_name,
            case_index=task.case_index,
        )
        if task_index is None
        else task_index.get((task.batch_id, task.case_index))
    )
    if task != expected:
        message = "Campaign submission task identity is inconsistent."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    launcher = repository / "scripts" / "generation_node.sh"
    if not launcher.is_file() or launcher.is_symlink():
        message = f"Campaign compute-node launcher is missing or unsafe: {launcher}"
        raise FileNotFoundError(message)
    requested_log_directory = Path(scheduler_log_directory)
    if (
        not requested_log_directory.is_absolute()
        or requested_log_directory.is_symlink()
        or (requested_log_directory.exists() and not requested_log_directory.is_dir())
    ):
        message = f"Scheduler log directory must be one safe absolute directory: {requested_log_directory}."
        raise ValueError(message)
    runtime_root = common.paths.get_runtime_root().resolve()
    storage = Path(storage_root).resolve()
    if not storage.is_dir() or storage in (repository, runtime_root):
        message = "Generation storage root must be one separate existing directory."
        raise ValueError(message)
    job_name = common.paths.validate_logical_name(
        scheduler_job_name,
        label="scheduler_job_name",
    )
    if len(job_name) > _MAX_SCHEDULER_JOB_NAME_LENGTH:
        message = "scheduler_job_name must contain at most 48 characters."
        raise ValueError(message)
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        message = "Campaign Slurm attempt_index must be a positive integer."
        raise ValueError(message)
    cluster = campaign.execution_values["cluster"]
    cores_per_case = int(cluster["cores_per_case"])
    site = campaign.execution_values["site"]
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
        message = "GENERATION_NATIVE_VENV must be the sibling runtime Generation venv."
        raise ValueError(message)
    worker_environment = [
        f"GENERATION_GIT_COMMIT={source_commit}",
        f"GENERATION_SOURCE_SHA256={source_sha}",
        f"GENERATION_NATIVE_VENV={native_venv}",
        f"STORAGE_ROOT={storage}",
        f"GENERATION_COMSOL_MODULE={site['comsol_module']}",
        f"GENERATION_PYTHON_EXECUTABLE={site['python_executable']}",
        f"GENERATION_COMSOL_EXECUTABLE={site['comsol_executable']}",
        f"GENERATION_ATTEMPT_INDEX={attempt_index}",
    ]
    worker_command = [
        str(launcher),
        str(repository),
        "campaign-case",
        run_id,
        task.batch_name,
        str(task.case_index),
        str(cores_per_case),
    ]
    wrapped = shlex.join(["env", *worker_environment, *worker_command])
    return build_generation_slurm_command(
        repository=repository,
        job_name=job_name,
        cores_per_task=cores_per_case,
        partition=cluster["partition"],
        wall_time=cluster["wall_time"],
        scheduler_options=cluster["scheduler_options"],
        wrapped=wrapped,
    )


def build_generation_slurm_command(
    *,
    repository: Path,
    job_name: str,
    cores_per_task: int,
    partition: str | None,
    wall_time: str | None,
    scheduler_options: Sequence[str],
    wrapped: str,
) -> list[str]:
    """Build the common native Generation case submission command."""
    runtime_logs = common.paths.get_runtime_root().resolve() / "logs" / "generation"
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cores_per_task}",
        f"--chdir={repository}",
        f"--job-name={job_name}",
        "--export=ALL",
        f"--output={runtime_logs}/slurm-%j.out",
        f"--error={runtime_logs}/slurm-%j.err",
    ]
    if partition is not None:
        command.append(f"--partition={partition}")
    if wall_time is not None:
        command.append(f"--time={wall_time}")
    command.extend(scheduler_options)
    command.append(f"--wrap={wrapped}")
    return command
