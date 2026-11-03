"""Own the fixed Generation-side COMSOL process invocation contract.

Responsibilities:
  - Own canonical COMSOL job and case-local model names
  - Build one safe COMSOL batch argument vector for every execution path
  - Select exactly one save mode, one runtime-owned batch log, and case-local recovery and temporary directories

Design principles:
  - Template job identity and workspace filenames are internal conventions
  - Core allocation remains explicit in every invocation
  - Runtime-owned flags cannot be supplied through user configuration
  - Every invocation, local or Slurm, requests the COMSOL batch licence
  - COMSOL general options precede the batch target so recovery files and temporaries never leave the case workspace

This module does NOT:
  - Execute COMSOL, modify templates, or define scientific parameters
  - Expose job tags or model filenames as user configuration
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract

from . import generation_runtime_comsol_timing as comsol_timing

if TYPE_CHECKING:
    from pathlib import Path

    from src.generation.cases import generation_cases_config as config_contract

COMSOL_JOB_TAG: Final = "b1"
WORK_MODEL_FILENAME: Final = "model.mph"
RETAINED_MODEL_FILENAME: Final = "solved.mph"
SCRATCH_DIRNAME: Final = "comsol"
RECOVERY_DIRNAME: Final = "recovery"
TEMPORARY_DIRNAME: Final = "tmp"


def create_scratch_directories(work_directory: Path) -> Path:
    """Create the case-local COMSOL recovery and temporary directories and return their parent."""
    scratch = work_directory / SCRATCH_DIRNAME
    (scratch / RECOVERY_DIRNAME).mkdir(parents=True)
    (scratch / TEMPORARY_DIRNAME).mkdir()
    return scratch


def resolve_comsol_executable(config: config_contract.GenerationConfig) -> str:
    """Return the configured COMSOL executable without shell parsing."""
    executable = config.execution_values["runtime"].get("executable") or os.environ.get("COMSOL_EXECUTABLE")
    if not executable:
        message = "COMSOL executable is unresolved; configure runtime.executable or COMSOL_EXECUTABLE."
        raise FileNotFoundError(message)
    return str(executable)


def _comsol_parameter_arguments(
    config: config_contract.GenerationConfig,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None,
) -> list[str]:
    """Return the exact admitted transient runtime vector or no steady flags."""
    if config.profile.id == profiles.STEADY_FLOW_PROFILE:
        if scalar_handoff is not None:
            message = "Steady-flow COMSOL commands cannot receive a transient scalar handoff."
            raise ValueError(message)
        return []
    if config.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = f"Unsupported simulation profile for COMSOL execution: {config.profile.id!r}."
        raise ValueError(message)
    if scalar_handoff is None:
        message = "Transient COMSOL commands require one admitted scalar handoff."
        raise ValueError(message)
    if scalar_handoff.profile_id != config.profile.id:
        message = "Scalar-handoff profile identity disagrees with the generation configuration."
        raise ValueError(message)
    entries = scalar_handoff.entries
    names = tuple(entry.name for entry in entries)
    if names != profiles.TRANSIENT_SCALAR_INPUT_FIELDS:
        message = "Transient COMSOL runtime overrides do not match the canonical case-dependent contract."
        raise ValueError(message)
    values = tuple(scalar_handoff_contract.format_comsol_parameter(entry) for entry in entries)
    grid = config.scientific_values["grid"]
    time = config.scientific_values["time"]
    if time["start"] != 0.0 or time["regular_times"][0] != 0.0 or time["regular_times"][-1] != time["stop"]:
        message = "Transient COMSOL template requires a zero-based regular output-time grid."
        raise ValueError(message)
    minimum_nodes = 2
    if grid["nx"] < minimum_nodes or grid["ny"] < minimum_nodes or time["interval"] <= 0.0 or time["stop"] <= 0.0:
        message = "Transient COMSOL discretization must match an admitted positive grid and time plan."
        raise ValueError(message)
    fixed_names = ("Nx", "Ny", "Lx", "Ly", "Lz", "dt_out", "t_max")
    fixed_values = (
        str(grid["nx"]),
        str(grid["ny"]),
        *(f"{scalar_handoff_contract.format_scalar_number(float(grid[name]))}[m]" for name in ("Lx", "Ly", "Lz")),
        f"{scalar_handoff_contract.format_scalar_number(float(time['interval']))}[h]",
        f"{scalar_handoff_contract.format_scalar_number(float(time['stop']))}[h]",
    )
    parameter_names = (*names, *fixed_names)
    parameter_values = (*values, *fixed_values)
    indices = tuple(str(index) for index in range(1, len(parameter_names) + 1))
    return [
        "-pname",
        ",".join(parameter_names),
        "-plist",
        ",".join(parameter_values),
        "-pindex",
        ",".join(indices),
    ]


def _save_arguments(config: config_contract.GenerationConfig) -> list[str]:
    """Use the fixed output model required by controlled status-file stopping."""
    if config.execution_values.get("retention_policy") not in {"full", "compact"}:
        message = "execution.retention_policy must be resolved before COMSOL execution."
        raise TypeError(message)
    return ["-outputfile", RETAINED_MODEL_FILENAME]


def build_comsol_command(
    config: config_contract.GenerationConfig,
    *,
    cores_per_case: int,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None = None,
    scheduler_kind: str = "local",
    batch_log_path: str = comsol_timing.COMSOL_BATCH_LOG_FILENAME,
    scratch_directory: str = SCRATCH_DIRNAME,
) -> list[str]:
    """Build the canonical Generation COMSOL batch argument vector.

    The general options come before the ``batch`` target. ``scratch_directory`` locates the case-local
    directory that holds the recovery and temporary directories; a relative value resolves against the
    COMSOL process working directory.
    """
    if isinstance(cores_per_case, bool) or not isinstance(cores_per_case, int) or cores_per_case < 1:
        message = f"cores_per_case must be a positive integer, got {cores_per_case!r}."
        raise ValueError(message)
    if scheduler_kind not in {"local", "slurm"}:
        message = f"Unsupported scheduler kind for case execution: {scheduler_kind!r}."
        raise ValueError(message)
    extra_arguments = config.execution_values["runtime"]["extra_arguments"]
    for label, value in (("batch_log_path", batch_log_path), ("scratch_directory", scratch_directory)):
        if not isinstance(value, str):
            message = f"{label} must be a string."
            raise TypeError(message)
        if not value or "\x00" in value:
            message = f"{label} must be one safe non-empty path."
            raise ValueError(message)
    for value in extra_arguments:
        normalized = str(value).strip().casefold()
        if normalized in {"-batchlog", "-batchlogout"} or normalized.startswith(("-batchlog=", "-batchlogout=")):
            message = "Runtime-owned COMSOL batch logging conflicts with configured extra arguments."
            raise ValueError(message)
    return [
        resolve_comsol_executable(config),
        "-autosave",
        "off",
        "-recoverydir",
        f"{scratch_directory}/{RECOVERY_DIRNAME}",
        "-tmpdir",
        f"{scratch_directory}/{TEMPORARY_DIRNAME}",
        "batch",
        "-inputfile",
        WORK_MODEL_FILENAME,
        "-job",
        COMSOL_JOB_TAG,
        *_save_arguments(config),
        *_comsol_parameter_arguments(config, scalar_handoff),
        "-np",
        str(cores_per_case),
        "-batchlog",
        batch_log_path,
        "-batchlogout",
        # Every launch, local or Slurm, draws the batch licence seat; no configuration can drop or replace it.
        "-usebatchlic",
        *extra_arguments,
    ]
