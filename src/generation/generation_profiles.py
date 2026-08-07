"""
===============================================================================
generation_profiles.py
===============================================================================
Define the authoritative reference-simulation profile contracts.
Responsibilities:
  - Register the exact supported simulation-profile identifiers
  - Bind each profile to one immutable COMSOL template identity
  - Declare profile-owned input, export-role, learning-view, and airflow provenance
Design principles:
  - Profile selection is explicit and never inferred from filenames or tasks
  - Shared generation orchestration consumes narrow immutable profile definitions
  - The steady-flow TaskSpec remains authoritative for learned airflow fields
This module does NOT:
  - Load generation YAML, generate fields, execute COMSOL, or build datasets
  - Define transient-drying training channels or temporal learning semantics
===============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from src import common, domain

STEADY_FLOW_PROFILE = "steady_flow"
TRANSIENT_DRYING_PROFILE = "transient_drying"
STEADY_AIRFLOW_SOURCE = "comsol_steady_reference"
COUPLED_AIRFLOW_SOURCE = "comsol_coupled_reference"
STEADY_FLOW_EXPORT_ROLE = "steady_flow_fields"
TRANSIENT_RAW_EXPORT_ROLE = "transient_fields"
STEADY_FLOW_LEARNING_VIEW = "steady_flow"
TRANSIENT_DRYING_LEARNING_VIEW = "transient_drying"
STATIONARITY_TOLERANCE = 1e-10
_SIDECAR_PART_COUNT = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ExportRoleSpec:
    """Describe one profile-owned raw export role."""

    role: str
    required: bool
    allow_multiple: bool
    canonical_fields: tuple[str, ...]
    learning_view: str


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    """Describe one immutable reference-simulation profile."""

    id: str
    template_relative_path: str
    template_sha256_source: str
    spatial_input_filename: str
    spatial_field_mapping: tuple[tuple[str, str], ...]
    scalar_file_allowed: bool
    schedule_file_allowed: bool
    export_roles: tuple[ExportRoleSpec, ...]
    available_learning_views: tuple[str, ...]
    airflow_source: str

    @property
    def template_path(self) -> Path:
        """Return the repository-owned immutable template path."""
        return (common.paths.get_project_root() / self.template_relative_path).resolve()

    @property
    def template_sha256(self) -> str:
        """Return the one authoritative expected template digest."""
        source = self.template_sha256_source
        if _SHA256_PATTERN.fullmatch(source) is not None:
            return source
        sidecar = (common.paths.get_project_root() / source).resolve()
        try:
            line = sidecar.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as error:
            message = f"Could not read template identity sidecar for profile {self.id!r}: {sidecar}"
            raise ValueError(message) from error
        parts = line.split()
        expected_name = Path(self.template_relative_path).name
        if len(parts) != _SIDECAR_PART_COUNT or _SHA256_PATTERN.fullmatch(parts[0]) is None or Path(parts[1]).name != expected_name:
            message = f"Template identity sidecar is malformed for profile {self.id!r}: {sidecar}"
            raise ValueError(message)
        return parts[0]

    @property
    def required_export_roles(self) -> tuple[str, ...]:
        """Return required export-role identifiers in profile order."""
        return tuple(spec.role for spec in self.export_roles if spec.required)

    def export_role(self, role: str) -> ExportRoleSpec:
        """Resolve one exact export role owned by this profile."""
        for spec in self.export_roles:
            if spec.role == role:
                return spec
        available = ", ".join(spec.role for spec in self.export_roles)
        message = f"Unknown export role {role!r} for profile {self.id!r}. Available roles: {available}."
        raise ValueError(message)


def _steady_flow_input_mapping() -> tuple[tuple[str, str], ...]:
    """Bind generated adapter columns to the current steady-flow TaskSpec."""
    task = domain.tasks.registry.get_task(STEADY_FLOW_LEARNING_VIEW)
    generated_names = {
        "x": "x",
        "y": "y",
        "kxx": "Kxx",
        "kxy": "Kxy",
        "kyy": "Kyy",
        "eps": "eps",
        "p_bc": "p_bc",
    }
    if set(task.input_names) != set(generated_names):
        message = "The steady-flow TaskSpec input fields no longer match the Python generator adapter contract."
        raise RuntimeError(message)
    return tuple((field, generated_names[field]) for field in task.input_names)


_STEADY_OUTPUT_FIELDS = tuple(domain.tasks.registry.get_task(STEADY_FLOW_LEARNING_VIEW).output_names)
_PROFILES: Final = MappingProxyType(
    {
        STEADY_FLOW_PROFILE: SimulationProfile(
            id=STEADY_FLOW_PROFILE,
            template_relative_path="simulation/steady_flow/template_brinkman.mph",
            template_sha256_source="c3363528f49a29774cbf7f48948d5216022f1bac14f4f6c635e7b912985ba976",
            spatial_input_filename="case_0001.csv",
            spatial_field_mapping=_steady_flow_input_mapping(),
            scalar_file_allowed=False,
            schedule_file_allowed=False,
            export_roles=(
                ExportRoleSpec(
                    role=STEADY_FLOW_EXPORT_ROLE,
                    required=True,
                    allow_multiple=False,
                    canonical_fields=("x", "y", *_STEADY_OUTPUT_FIELDS),
                    learning_view=STEADY_FLOW_LEARNING_VIEW,
                ),
            ),
            available_learning_views=(STEADY_FLOW_LEARNING_VIEW,),
            airflow_source=STEADY_AIRFLOW_SOURCE,
        ),
        TRANSIENT_DRYING_PROFILE: SimulationProfile(
            id=TRANSIENT_DRYING_PROFILE,
            template_relative_path="simulation/transient_drying/template_brinkman_temp_moist.mph",
            template_sha256_source="simulation/transient_drying/template.sha256",
            spatial_input_filename="case_0001.csv",
            spatial_field_mapping=_steady_flow_input_mapping(),
            scalar_file_allowed=True,
            schedule_file_allowed=True,
            export_roles=(
                ExportRoleSpec(
                    role=STEADY_FLOW_EXPORT_ROLE,
                    required=True,
                    allow_multiple=False,
                    canonical_fields=("x", "y", *_STEADY_OUTPUT_FIELDS),
                    learning_view=STEADY_FLOW_LEARNING_VIEW,
                ),
                ExportRoleSpec(
                    role=TRANSIENT_RAW_EXPORT_ROLE,
                    required=True,
                    allow_multiple=True,
                    canonical_fields=(),
                    learning_view=TRANSIENT_DRYING_LEARNING_VIEW,
                ),
            ),
            available_learning_views=(STEADY_FLOW_LEARNING_VIEW, TRANSIENT_DRYING_LEARNING_VIEW),
            airflow_source=COUPLED_AIRFLOW_SOURCE,
        ),
    }
)


def available_profiles() -> tuple[str, ...]:
    """Return the exact supported simulation-profile identifiers."""
    return tuple(sorted(_PROFILES))


def get_profile(profile_id: str) -> SimulationProfile:
    """Resolve and validate one exact simulation profile."""
    try:
        profile = _PROFILES[profile_id]
    except KeyError as error:
        available = ", ".join(available_profiles())
        message = f"Unknown simulation_profile {profile_id!r}. Available profiles: {available}."
        raise ValueError(message) from error
    path = profile.template_path
    if not path.is_file() or path.suffix.lower() != ".mph":
        message = f"Required COMSOL template is missing for profile {profile.id!r}: {path}"
        raise FileNotFoundError(message)
    expected = profile.template_sha256
    actual = common.serialization.file_sha256(path)
    if actual != expected:
        message = f"COMSOL template SHA-256 mismatch for profile {profile.id!r}: expected {expected}, got {actual}."
        raise ValueError(message)
    return profile
