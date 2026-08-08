"""
===============================================================================
generation_profiles.py
===============================================================================
Define the authoritative reference-simulation profile contracts.
Responsibilities:
  - Register the exact supported profile and immutable template identities
  - Declare canonical input, export-role, unit, and learning-view contracts
  - Validate repository template digests before generation or execution
Design principles:
  - Logical fields are stable while COMSOL headers remain explicit configuration
  - Profile selection is explicit and never inferred from filenames or tasks
  - The registered steady-flow TaskSpec shares the canonical field names
This module does NOT:
  - Guess COMSOL tags, expressions, filenames, or sign conventions
  - Load YAML, generate fields, execute COMSOL, or register transient learning
===============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from src import common

STEADY_FLOW_PROFILE = "steady_flow"
TRANSIENT_DRYING_PROFILE = "transient_drying"
STEADY_AIRFLOW_SOURCE = "comsol_steady_reference"
COUPLED_AIRFLOW_SOURCE = "comsol_coupled_reference"
STEADY_FLOW_EXPORT_ROLE = "steady_flow_fields"
TRANSIENT_RAW_EXPORT_ROLE = "transient_fields"
GLOBAL_EXPORT_ROLE = "global_time_series"
FINAL_STATUS_EXPORT_ROLE = "final_status"
STEADY_FLOW_LEARNING_VIEW = "steady_flow"
TRANSIENT_DRYING_LEARNING_VIEW = "transient_drying"
STATIONARITY_TOLERANCE = 1e-10
_SIDECAR_PART_COUNT = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

SPATIAL_INPUT_FIELDS: Final = (
    "x",
    "y",
    "Kxx",
    "Kxy",
    "Kyy",
    "eps_bed",
    "p_bc",
    "X_0_db_field",
)
SCHEDULE_FIELDS: Final = ("t", "T_in", "omega_in", "phi_in")
SCALAR_INPUT_FIELDS: Final = (
    "T_init",
    "T_amb",
    "T_in_ref",
    "eps_bed_cal_ref",
    "rho_bu_dry_ref",
    "k_gr",
    "cp_gr_dry",
    "X_target_wb",
    "r_surf_0",
    "r_int_surf",
    "f_surf",
    "A_osw",
    "B_osw",
    "C_osw",
    "f_wet_dm_max",
)
SCALAR_INPUT_UNITS: Final = (
    "K",
    "K",
    "K",
    "1",
    "kg/m^3",
    "W/(m*K)",
    "J/(kg*K)",
    "1",
    "1/s",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
)
SCHEDULE_UNITS: Final = ("h", "K", "kg/kg", "1")
STATIC_FIELD_NAMES: Final = (
    "Kxx",
    "Kxy",
    "Kyy",
    "eps_bed",
    "p_bc",
    "X_0_db_field",
    "u",
    "v",
    "p",
    "rho_bu_dry",
)
STATIC_FIELD_UNITS: Final = (
    "m^2",
    "m^2",
    "m^2",
    "1",
    "Pa",
    "kg/kg",
    "m/s",
    "m/s",
    "Pa",
    "kg/m^3",
)
TRANSIENT_FIELD_NAMES: Final = ("T", "phi", "w_surf", "w_int")
TRANSIENT_FIELD_UNITS: Final = ("K", "1", "kg/m^3", "kg/m^3")
GLOBAL_FIELD_NAMES: Final = (
    "t",
    "X_wb_bulk",
    "X_wb_max",
    "X_wb_q95_mass",
    "f_wet_dm",
    "T_out_mean",
    "phi_out_mean",
    "m_w_gr",
    "m_v_gas",
    "m_dot_evap",
    "m_dot_v_in",
    "m_dot_v_out",
)
GLOBAL_FIELD_UNITS: Final = ("h", "1", "1", "1", "1", "K", "1", "kg", "kg", "kg/s", "kg/s", "kg/s")
FINAL_STATUS_FIELDS: Final = (
    "t_final",
    "f_wet_dm_final",
    "X_target_wb",
    "X_wb_bulk",
    "X_wb_max",
    "X_wb_q95_mass",
    "T_min_final",
    "T_max_final",
    "phi_min_final",
    "phi_max_final",
)
FINAL_STATUS_UNITS: Final = ("h", "1", "1", "1", "1", "1", "K", "K", "1", "1")


@dataclass(frozen=True, slots=True)
class ExportRoleSpec:
    """Describe one profile-owned raw export role."""

    role: str
    required: bool
    allow_multiple: bool
    logical_fields: tuple[str, ...]
    units: tuple[str, ...]
    learning_view: str | None


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    """Describe one immutable reference-simulation profile."""

    id: str
    template_relative_path: str
    template_sha256_source: str
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


_STEADY_ROLE = ExportRoleSpec(
    role=STEADY_FLOW_EXPORT_ROLE,
    required=True,
    allow_multiple=False,
    logical_fields=("x", "y", *STATIC_FIELD_NAMES),
    units=("m", "m", *STATIC_FIELD_UNITS),
    learning_view=STEADY_FLOW_LEARNING_VIEW,
)
_PROFILES: Final = MappingProxyType(
    {
        STEADY_FLOW_PROFILE: SimulationProfile(
            id=STEADY_FLOW_PROFILE,
            template_relative_path="simulation/steady_flow/template_brinkman.mph",
            template_sha256_source="c3363528f49a29774cbf7f48948d5216022f1bac14f4f6c635e7b912985ba976",
            export_roles=(_STEADY_ROLE,),
            available_learning_views=(STEADY_FLOW_LEARNING_VIEW,),
            airflow_source=STEADY_AIRFLOW_SOURCE,
        ),
        TRANSIENT_DRYING_PROFILE: SimulationProfile(
            id=TRANSIENT_DRYING_PROFILE,
            template_relative_path="simulation/transient_drying/template_brinkman_temp_moist.mph",
            template_sha256_source="simulation/transient_drying/template.sha256",
            export_roles=(
                _STEADY_ROLE,
                ExportRoleSpec(
                    role=TRANSIENT_RAW_EXPORT_ROLE,
                    required=True,
                    allow_multiple=True,
                    logical_fields=("x", "y", "t", *TRANSIENT_FIELD_NAMES),
                    units=("m", "m", "h", *TRANSIENT_FIELD_UNITS),
                    learning_view=TRANSIENT_DRYING_LEARNING_VIEW,
                ),
                ExportRoleSpec(
                    role=GLOBAL_EXPORT_ROLE,
                    required=True,
                    allow_multiple=False,
                    logical_fields=GLOBAL_FIELD_NAMES,
                    units=GLOBAL_FIELD_UNITS,
                    learning_view=None,
                ),
                ExportRoleSpec(
                    role=FINAL_STATUS_EXPORT_ROLE,
                    required=True,
                    allow_multiple=False,
                    logical_fields=FINAL_STATUS_FIELDS,
                    units=FINAL_STATUS_UNITS,
                    learning_view=None,
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
    """Resolve one exact profile and validate its immutable template bytes."""
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
    actual = common.serialization.file_sha256(path)
    if actual != profile.template_sha256:
        message = f"COMSOL template SHA-256 mismatch for profile {profile.id!r}: expected {profile.template_sha256}, got {actual}."
        raise ValueError(message)
    return profile
