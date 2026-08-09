"""
===============================================================================
generation_preflight.py
===============================================================================
Audit the native CPU generation environment without starting a COMSOL solve.
Responsibilities:
  - Validate Python, COMSOL, Slurm, rsync, modules, templates, and resources
  - Validate user-owned persistent roots and node-local scratch containment
  - Exercise one collision-safe marked directory creation and cleanup lifecycle
Design principles:
  - Environment readiness and production scientific readiness remain separate
  - Every command is an argument vector with bounded captured output
  - A preflight probe is self-cleaning and never becomes campaign state
This module does NOT:
  - Submit production work, execute a model, install packages, or change science
  - Treat template hashes or fake executables as runtime COMSOL validation
===============================================================================
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from src import common

from . import generation_cluster as cluster_service
from . import generation_config as config_service
from . import generation_profiles as profiles
from . import generation_workspace as workspace_service

_REQUIRED_IMPORTS = ("numpy", "scipy", "yaml", "h5py")
_REQUIRED_COMMANDS = (
    "python3",
    "comsol",
    "sbatch",
    "squeue",
    "sacct",
    "scancel",
    "rsync",
)
_EXPECTED_MODULE_INITIALIZATION = (
    "module load Python/3.10",
    "module load Comsol/v6.4",
)


def _version_output(command: list[str]) -> dict[str, Any]:
    """Return bounded version output for one required executable."""
    result = subprocess.run(  # noqa: S603 -- required executable paths resolved by shutil.which
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    output = (result.stdout + result.stderr).strip()
    return {
        "arguments": command,
        "exit_code": result.returncode,
        "output": output[:4096],
    }


def _owned_writable_directory(path: Path, *, label: str) -> dict[str, Any]:
    """Validate one existing absolute user-owned writable directory."""
    resolved = path.expanduser().resolve()
    if not path.expanduser().is_absolute() or not resolved.is_dir() or resolved.is_symlink():
        message = f"{label} must be one existing safe absolute directory: {resolved}"
        raise ValueError(message)
    if resolved.stat().st_uid != os.getuid():
        message = f"{label} is not owned by the current user: {resolved}"
        raise PermissionError(message)
    if not os.access(resolved, os.W_OK | os.X_OK):
        message = f"{label} is not writable and searchable: {resolved}"
        raise PermissionError(message)
    return {
        "path": str(resolved),
        "owner_uid": resolved.stat().st_uid,
        "writable": True,
    }


def _command_versions(commands: dict[str, str]) -> dict[str, Any]:
    """Collect non-solving version evidence from native tools."""
    arguments = {
        "python3": [commands["python3"], "--version"],
        "comsol": [commands["comsol"], "-version"],
        "sbatch": [commands["sbatch"], "--version"],
        "rsync": [commands["rsync"], "--version"],
    }
    results = {name: _version_output(command) for name, command in arguments.items()}
    failures = [name for name, result in results.items() if result["exit_code"] != 0]
    if failures:
        message = f"Required version commands failed: {failures}."
        raise RuntimeError(message)
    if "6.4" not in str(results["comsol"]["output"]):
        message = "Native COMSOL must report version 6.4."
        raise RuntimeError(message)
    return results


def _template_evidence() -> dict[str, Any]:
    """Validate both canonical template sidecars and return exact identities."""
    evidence: dict[str, Any] = {}
    for profile_id in profiles.available_profiles():
        profile = profiles.get_profile(profile_id)
        evidence[profile_id] = {
            "path": str(profile.template_path),
            "size_bytes": profile.template_path.stat().st_size,
            "sha256": profile.template_sha256,
            "sidecar_validation": "pass",
            "comsol_internal_contract": "runtime_unverified",
        }
    return evidence


def _path_guard_probe(
    *,
    storage_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    """Exercise one unique marked probe and reject broad cleanup targets."""
    probe, marker = workspace_service.create_cleanup_probe(
        storage_root=storage_root,
        work_root=work_root,
    )
    rejected: list[str] = []
    for candidate in (
        Path("/"),
        Path.home().resolve(),
        common.paths.get_project_root().resolve(),
        storage_root,
        work_root,
    ):
        try:
            workspace_service.cleanup_case_workspace(
                candidate,
                allowed_root=work_root,
                storage_root=storage_root,
                expected_run_id="preflight",
                expected_case_id="preflight-probe",
                allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
            )
        except (ValueError, RuntimeError):  # noqa: PERF203 -- every protected target is independently audited
            rejected.append(str(candidate))
        else:
            message = f"Cleanup guard unexpectedly accepted protected path: {candidate}"
            raise RuntimeError(message)
    reclaimed = workspace_service.cleanup_case_workspace(
        probe,
        allowed_root=work_root,
        storage_root=storage_root,
        expected_run_id="preflight",
        expected_case_id="preflight-probe",
        allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    if probe.exists():
        message = f"Preflight probe directory survived cleanup: {probe}"
        raise RuntimeError(message)
    return {
        "probe_path": str(probe),
        "marker_path": str(marker),
        "reclaimed_bytes": reclaimed,
        "protected_paths_rejected": rejected,
        "probe_removed": True,
    }


def _quota_evidence(home: Path) -> dict[str, Any]:
    """Return optional quota output without making quota support mandatory."""
    executable = shutil.which("quota")
    if executable is None:
        return {"available": False, "output": None}
    result = subprocess.run(  # noqa: S603 -- optional quota executable resolved by shutil.which
        [executable, "-s"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
        cwd=home,
    )
    return {
        "available": True,
        "exit_code": result.returncode,
        "output": (result.stdout + result.stderr).strip()[:4096],
    }


def run_cpu_preflight(
    config_path: Path,
    *,
    only_batch: str | None,
    storage_root: Path | str,
    work_root: Path | str,
    venv_path: Path | str,
    max_nodes: int,
    cases_per_node: int,
    cores_per_case: int,
    max_parallel_cases: int,
    cores_per_node: int,
) -> dict[str, Any]:
    """Run one safe native CPU environment and path preflight."""
    repository = common.paths.get_project_root().resolve()
    home = Path.home().resolve()
    storage = workspace_service.resolve_storage_root(
        storage_root,
        create=False,
    )
    if not storage.is_dir():
        message = f"Preflight requires the prepared storage root: {storage}"
        raise FileNotFoundError(message)
    work = workspace_service.resolve_work_root(
        storage_root=storage,
        work_root=work_root,
        create=False,
    )
    venv = Path(venv_path).expanduser().resolve()
    paths = {
        "home": _owned_writable_directory(home, label="HOME"),
        "repository": _owned_writable_directory(
            repository,
            label="repository",
        ),
        "storage": _owned_writable_directory(storage, label="storage_root"),
        "work": _owned_writable_directory(work, label="work_root"),
        "venv": _owned_writable_directory(venv, label="venv"),
    }
    try:
        Path(sys.executable).resolve().relative_to(venv)
    except ValueError as error:
        message = f"Active Python executable {sys.executable} is not inside venv {venv}."
        raise RuntimeError(message) from error
    if sys.version_info[:2] != (3, 10):
        message = f"Native generation requires Python 3.10 exactly, got {sys.version}."
        raise RuntimeError(message)
    pyproject = (repository / "pyproject.toml").read_text(encoding="utf-8")
    if 'requires-python = ">=3.10"' not in pyproject:
        message = "Package metadata no longer declares the audited Python >=3.10 contract."
        raise RuntimeError(message)
    missing_imports = [name for name in _REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
    if missing_imports:
        message = f"Generation CPU venv is missing imports: {missing_imports}."
        raise ModuleNotFoundError(message)
    commands: dict[str, str] = {}
    for name in _REQUIRED_COMMANDS:
        executable = shutil.which(name)
        if executable is None:
            message = f"Required native CPU command is unavailable: {name}"
            raise FileNotFoundError(message)
        commands[name] = executable
    versions = _command_versions(commands)

    campaign = config_service.load_campaign_config(
        config_path,
        require_executable=False,
    )
    if only_batch is not None:
        campaign = campaign.select_batches((only_batch,))
    modules = tuple(campaign.execution_values["runtime"]["module_initialization"])
    if modules != _EXPECTED_MODULE_INITIALIZATION:
        message = "Execution configuration must load exactly Python/3.10 and Comsol/v6.4 in that order."
        raise ValueError(message)
    site = campaign.execution_values["site"]
    if (
        site["python_module"] != "Python/3.10"
        or site["comsol_module"] != "Comsol/v6.4"
        or site["python_executable"] != "python3"
        or site["comsol_executable"] != "comsol"
        or site["scheduler"] != "slurm"
    ):
        message = "Execution site configuration disagrees with the native ICE contract."
        raise ValueError(message)
    remaining = max(
        sum(len(batch.case_indices) for batch in campaign.batches),
        1,
    )
    resource_plan = cluster_service.build_resource_plan(
        max_nodes=max_nodes,
        cases_per_node=cases_per_node,
        cores_per_case=cores_per_case,
        max_parallel_cases=max_parallel_cases,
        cores_per_node=cores_per_node,
        remaining_cases=remaining,
    )
    production_blocker: str | None = None
    try:
        executable_campaign = config_service.load_campaign_config(
            config_path,
            require_executable=True,
        )
        if only_batch is not None:
            executable_campaign.select_batches((only_batch,))
    except Exception as error:  # noqa: BLE001 -- readiness is reported separately
        production_blocker = str(error)
    disk = shutil.disk_usage(storage)
    return {
        "schema_kind": "generation_cpu_preflight",
        "schema_version": 1,
        "status": ("environment_ready" if production_blocker is None else "environment_ready_production_blocked"),
        "production_configuration_ready": production_blocker is None,
        "production_configuration_blocker": production_blocker,
        "host": socket.gethostname(),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "required_imports": list(_REQUIRED_IMPORTS),
        },
        "paths": paths,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "quota": _quota_evidence(home),
        "commands": commands,
        "versions": versions,
        "configured_modules": list(modules),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
        "templates": _template_evidence(),
        "execution_config": campaign.execution_values,
        "resource_plan": {
            "max_nodes": resource_plan.max_nodes,
            "cases_per_node": resource_plan.cases_per_node,
            "cores_per_case": resource_plan.cores_per_case,
            "max_parallel_cases": resource_plan.max_parallel_cases,
            "cores_per_node": resource_plan.cores_per_node,
            "effective_parallel_cases": resource_plan.effective_parallel_cases,
            "effective_nodes": resource_plan.effective_nodes,
        },
        "path_cleanup_probe": _path_guard_probe(
            storage_root=storage,
            work_root=work,
        ),
        "production_solve_started": False,
    }
