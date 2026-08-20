"""
generation_runtime_preflight.py

Audit the native CPU generation environment without starting a COMSOL solve.
Responsibilities:
  - Validate compute Python, COMSOL, modules, templates, scratch, and storage
  - Validate the configured Generation venv through Python runtime prefixes
  - Validate user-owned persistent roots and node-local scratch containment
  - Exercise one collision-safe marked directory creation and cleanup lifecycle
Design principles:
  - Environment readiness and production scientific readiness remain separate
  - Every command is an argument vector with bounded captured output
  - A preflight probe is self-cleaning and never becomes campaign state
This module does NOT:
  - Submit production work, execute a model, install packages, or change science
  - Treat template hashes or fake executables as runtime COMSOL validation
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from src import common
from src.generation.cases import generation_cases_config as config_service

from . import generation_runtime_workspace as workspace_service

_REQUIRED_IMPORTS = ("numpy", "scipy", "h5py", "yaml", "src.generation.cli.cli_generation")
_MODULE_VERSION_PATTERN = re.compile(r"/v?([0-9]+(?:\.[0-9]+)*)$", re.IGNORECASE)


def validate_generation_venv(
    venv_path: Path | str,
    *,
    domain: str,
) -> dict[str, Any]:
    """Validate the configured Generation venv using Python runtime identity."""
    configured = Path(venv_path).expanduser()
    if not configured.is_absolute() or configured == Path("/"):
        message = f"{domain} prerequisite failed: configured Generation venv must be one safe absolute path: {configured}."
        raise ValueError(message)
    if configured.is_symlink():
        message = f"{domain} prerequisite failed: configured Generation venv root must not be a symlink: {configured}."
        raise ValueError(message)
    try:
        root = configured.resolve(strict=True)
    except FileNotFoundError as error:
        message = f"{domain} prerequisite missing: configured Generation venv root: {configured}."
        raise FileNotFoundError(message) from error
    if not root.is_dir():
        message = f"{domain} prerequisite failed: configured Generation venv root is not a directory: {configured}."
        raise ValueError(message)
    if root.stat().st_uid != os.getuid():
        message = f"{domain} prerequisite failed: configured Generation venv is not owned by the current user: {root}."
        raise PermissionError(message)
    if not os.access(root, os.R_OK | os.X_OK):
        message = f"{domain} prerequisite failed: configured Generation venv is not readable and searchable: {root}."
        raise PermissionError(message)

    metadata = root / "pyvenv.cfg"
    if not metadata.is_file() or metadata.is_symlink() or not os.access(metadata, os.R_OK):
        message = f"{domain} prerequisite missing: regular readable Generation venv metadata: {metadata}."
        raise FileNotFoundError(message)
    launcher = configured / "bin/python"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        message = f"{domain} prerequisite missing: executable Generation venv launcher: {launcher}."
        raise FileNotFoundError(message)

    reported_prefix = Path(sys.prefix).expanduser()
    reported_base_prefix = Path(sys.base_prefix).expanduser()
    reported_exec_prefix = Path(sys.exec_prefix).expanduser()
    if not all(path.is_absolute() for path in (reported_prefix, reported_base_prefix, reported_exec_prefix)):
        message = f"{domain} prerequisite failed: Python reported non-absolute runtime prefix evidence."
        raise RuntimeError(message)
    runtime_prefix = reported_prefix.resolve()
    runtime_base_prefix = reported_base_prefix.resolve()
    runtime_exec_prefix = reported_exec_prefix.resolve()
    if runtime_prefix == runtime_base_prefix:
        message = (
            f"{domain} prerequisite failed: interpreter reports sys.prefix == sys.base_prefix ({reported_prefix}); "
            f"configured Generation venv {root} is not active."
        )
        raise RuntimeError(message)
    if runtime_prefix != root:
        message = f"{domain} prerequisite failed: configured Generation venv is {root}, but interpreter reports sys.prefix={reported_prefix}."
        raise RuntimeError(message)
    if runtime_exec_prefix != root:
        message = (
            f"{domain} prerequisite failed: configured Generation venv is {root}, but interpreter reports sys.exec_prefix={reported_exec_prefix}."
        )
        raise RuntimeError(message)

    runtime_executable = Path(sys.executable).expanduser()
    expected_launcher = launcher
    if not runtime_executable.is_absolute() or runtime_executable != expected_launcher:
        message = (
            f"{domain} prerequisite failed: configured Generation venv launcher is {expected_launcher}, "
            f"but interpreter reports sys.executable={sys.executable}."
        )
        raise RuntimeError(message)

    missing_imports = [name for name in _REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
    if missing_imports:
        message = (
            f"{domain} prerequisite missing: Generation CPU venv imports {missing_imports} "
            "(blocks case materialization and HDF5 conversion/admission)."
        )
        raise ModuleNotFoundError(message)
    return {
        "configured_venv": str(root),
        "launcher": str(expected_launcher),
        "resolved_launcher_target": str(launcher.resolve(strict=True)),
        "pyvenv_cfg": str(metadata),
        "sys_executable": sys.executable,
        "sys_prefix": str(reported_prefix),
        "sys_base_prefix": str(reported_base_prefix),
        "sys_exec_prefix": str(reported_exec_prefix),
        "required_imports": list(_REQUIRED_IMPORTS),
    }


def _version_output(
    command: list[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return bounded version output for one required executable."""
    result = subprocess.run(  # noqa: S603 -- required executable paths resolved by shutil.which
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
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
        message = f"CPU compute-node prerequisite failed: {label} must be one existing safe absolute directory: {resolved}"
        raise ValueError(message)
    if resolved.stat().st_uid != os.getuid():
        message = f"CPU compute-node prerequisite failed: {label} is not owned by the current user: {resolved}"
        raise PermissionError(message)
    if not os.access(resolved, os.W_OK | os.X_OK):
        message = f"CPU compute-node prerequisite failed: {label} is not writable and searchable: {resolved}"
        raise PermissionError(message)
    return {
        "path": str(resolved),
        "owner_uid": resolved.stat().st_uid,
        "writable": True,
    }


def configured_module_version(module_name: str) -> str | None:
    """Return an optional version suffix authored by one module identifier."""
    match = _MODULE_VERSION_PATTERN.search(module_name)
    return None if match is None else match.group(1)


def reported_version_matches(output: object, expected: str) -> bool:
    """Return whether tool output reports the configured version or one patch release."""
    pattern = rf"(?<![0-9.]){re.escape(expected)}(?:\.[0-9]+)*(?![0-9.])"
    return re.search(pattern, str(output)) is not None


def _command_versions(
    commands: dict[str, str],
    *,
    python_module: str,
    comsol_module: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Collect and validate non-solving version evidence from native tools."""
    arguments = {
        "python": [commands["python"], "--version"],
        "comsol": [commands["comsol"], "-version"],
    }
    results = {name: _version_output(command, timeout_seconds=timeout_seconds) for name, command in arguments.items()}
    failures = [name for name, result in results.items() if result["exit_code"] != 0]
    if failures:
        message = f"CPU compute-node prerequisite failed: version checks {failures} (blocks compute)."
        raise RuntimeError(message)
    expected_versions = {
        "python": configured_module_version(python_module),
        "comsol": configured_module_version(comsol_module),
    }
    for name, expected in expected_versions.items():
        if expected is not None and not reported_version_matches(results[name]["output"], expected):
            message = (
                f"CPU compute-node prerequisite failed: Configured {name} module expects version {expected}, "
                f"but the active executable reported {results[name]['output']!r}."
            )
            raise RuntimeError(message)
    return results


def _template_evidence() -> dict[str, Any]:
    """Validate both canonical template sidecars and return exact identities."""
    evidence: dict[str, Any] = {}
    for profile_id, template in config_service.discover_profile_template_identities().items():
        evidence[profile_id] = {
            "path": str(template.absolute_path),
            "size_bytes": template.absolute_path.stat().st_size,
            "sha256": template.sha256,
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


def _quota_evidence(
    home: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return optional quota output without making quota support mandatory."""
    executable = shutil.which("quota")
    if executable is None:
        return {"available": False, "output": None}
    result = subprocess.run(  # noqa: S603 -- optional quota executable resolved by shutil.which
        [executable, "-s"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
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
) -> dict[str, Any]:
    """Run one safe native CPU environment and path preflight."""
    repository = common.paths.get_project_root().resolve()
    home = Path.home().resolve()
    storage = workspace_service.resolve_storage_root(
        storage_root,
        create=False,
    )
    if not storage.is_dir():
        message = f"CPU compute-node prerequisite missing: prepared durable storage root {storage} (blocks compute)."
        raise FileNotFoundError(message)
    work = workspace_service.resolve_work_root(
        storage_root=storage,
        work_root=work_root,
        create=False,
    )
    venv_identity = validate_generation_venv(
        venv_path,
        domain="CPU compute-node",
    )
    venv = Path(venv_identity["configured_venv"])
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
    campaign = config_service.load_campaign_config(
        config_path,
        require_executable=False,
    )
    if only_batch is not None:
        campaign = campaign.select_batches((only_batch,))
    site = campaign.execution_values["site"]
    if site["scheduler"] != "slurm":
        message = "Native CPU cluster preflight requires configured scheduler='slurm'."
        raise ValueError(message)
    modules = tuple(campaign.execution_values["runtime"]["module_initialization"])
    timeout_seconds = float(campaign.execution_values["runtime"]["timeout_seconds"])

    configured_python_version = configured_module_version(str(site["python_module"]))
    if configured_python_version is not None:
        expected_python = tuple(int(component) for component in configured_python_version.split("."))
        active_python = tuple(sys.version_info[: len(expected_python)])
        if active_python != expected_python:
            message = (
                "CPU compute-node prerequisite failed: Configured Python module expects version "
                f"{configured_python_version}, but the active interpreter is {sys.version}."
            )
            raise RuntimeError(message)
    command_names = {
        "python": str(site["python_executable"]),
        "comsol": str(site["comsol_executable"]),
    }
    commands: dict[str, str] = {}
    for label, command_name in command_names.items():
        executable = shutil.which(command_name)
        if executable is None:
            message = (
                f"CPU compute-node prerequisite missing: {command_name} "
                f"(blocks {'native solve' if label == 'comsol' else 'case materialization and admission'})."
            )
            raise FileNotFoundError(message)
        commands[label] = executable
    versions = _command_versions(
        commands,
        python_module=str(site["python_module"]),
        comsol_module=str(site["comsol_module"]),
        timeout_seconds=timeout_seconds,
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
        "domain": "CPU compute-node",
        "host": socket.gethostname(),
        "python": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "venv_runtime": venv_identity,
            "required_imports": list(_REQUIRED_IMPORTS),
        },
        "paths": paths,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "quota": _quota_evidence(home, timeout_seconds=timeout_seconds),
        "commands": commands,
        "versions": versions,
        "checks": {
            "Generation-venv-imports": {"status": "pass", "imports": list(_REQUIRED_IMPORTS)},
            "repository-and-templates": {"status": "pass", "repository": str(repository)},
            "durable-storage": {"status": "pass", "path": str(storage)},
            "owned-scratch": {"status": "pass", "path": str(work)},
        },
        "configured_modules": list(modules),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
        "templates": _template_evidence(),
        "execution_config": campaign.execution_values,
        "submission_plan": {
            "cases_per_job": 1,
            "cores_per_case": campaign.execution_values["cluster"]["cores_per_case"],
            "cores_per_node": campaign.execution_values["cluster"]["cores_per_node"],
            "max_admission_cases": campaign.execution_values["submission"]["max_admission_cases"],
            "poll_interval_seconds": campaign.execution_values["submission"]["poll_interval_seconds"],
            "max_running_cases": campaign.execution_values["submission"]["max_running_cases"],
        },
        "path_cleanup_probe": _path_guard_probe(
            storage_root=storage,
            work_root=work,
        ),
        "production_solve_started": False,
    }
