"""
===============================================================================
generation_runtime_comsol.py
===============================================================================
Own the fixed Generation-side COMSOL process invocation contract.
Responsibilities:
  - Own canonical COMSOL job and case-local model names
  - Build one safe COMSOL batch argument vector for every execution path
  - Select exactly one save mode from the resolved retention policy
Design principles:
  - Template job identity and workspace filenames are internal conventions
  - Core allocation remains explicit in every invocation
  - Runtime-owned flags cannot be supplied through user configuration
This module does NOT:
  - Execute COMSOL, modify templates, or define scientific parameters
  - Expose job tags or model filenames as user configuration
===============================================================================
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract

if TYPE_CHECKING:
    from src.generation.cases import generation_cases_config as config_contract

COMSOL_JOB_TAG: Final = "s1"
WORK_MODEL_FILENAME: Final = "model.mph"
RETAINED_MODEL_FILENAME: Final = "solved.mph"


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
    indices = tuple(str(index) for index in range(1, len(entries) + 1))
    return [
        "-pname",
        ",".join(names),
        "-plist",
        ",".join(values),
        "-pindex",
        ",".join(indices),
    ]


def _save_arguments(config: config_contract.GenerationConfig) -> list[str]:
    """Return exactly one save mode from the validated retention state."""
    retention = config.execution_values.get("retention")
    retain_solved_model = retention.get("retain_solved_model") if isinstance(retention, Mapping) else None
    if not isinstance(retain_solved_model, bool):
        message = "execution.retention.retain_solved_model must be boolean for COMSOL execution."
        raise TypeError(message)
    if retain_solved_model:
        return ["-outputfile", RETAINED_MODEL_FILENAME]
    return ["-nosave"]


def build_comsol_command(
    config: config_contract.GenerationConfig,
    *,
    cores_per_case: int,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None = None,
    scheduler_kind: str = "local",
) -> list[str]:
    """Build the canonical Generation COMSOL batch argument vector."""
    if isinstance(cores_per_case, bool) or not isinstance(cores_per_case, int) or cores_per_case < 1:
        message = f"cores_per_case must be a positive integer, got {cores_per_case!r}."
        raise ValueError(message)
    if scheduler_kind not in {"local", "slurm"}:
        message = f"Unsupported scheduler kind for case execution: {scheduler_kind!r}."
        raise ValueError(message)
    return [
        resolve_comsol_executable(config),
        "batch",
        "-inputfile",
        WORK_MODEL_FILENAME,
        "-job",
        COMSOL_JOB_TAG,
        *_save_arguments(config),
        *_comsol_parameter_arguments(config, scalar_handoff),
        "-np",
        str(cores_per_case),
        *config.execution_values["runtime"]["extra_arguments"],
    ]
