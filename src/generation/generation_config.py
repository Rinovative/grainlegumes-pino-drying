"""
===============================================================================
generation_config.py
===============================================================================
Resolve and validate layered scientific and execution generation configurations.
Responsibilities:
  - Resolve common, material-family, profile, dataset, and execution YAML owners
  - Validate authoritative grid, time, adapter, storage, split, and runtime contracts
  - Derive deterministic scientific and exact case-input identities
Design principles:
  - Scientific and execution settings remain physically and cryptographically separate
  - Resolved scientific configuration contains no inheritance or unresolved defaults
  - Unknown, ambiguous, or non-executable configuration fails before case preparation
This module does NOT:
  - Invent material ranges, coupled sets, production counts, or COMSOL mappings
  - Generate samples or fields, execute COMSOL, or modify templates
===============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src import common

from . import generation_materials as materials
from . import generation_profiles as profiles
from . import generation_provenance as provenance_service

CONFIG_SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
CANONICAL_HDF5_SCHEMA_VERSION = 1
CANONICAL_HDF5_CONVERTER_VERSION = 1
CASE_ID_WIDTH = 4
UINT32_MAX = 2**32 - 1
_EXPECTED_T_IN_MIN = 298.15
_EXPECTED_T_IN_MAX = 313.15
_EXPECTED_F_WET_DM_MAX = 0.05
_EXPECTED_HDF5_COMPRESSION_LEVEL = 4
_MAXIMUM_TIME_CHUNK = 2
PRODUCTION_CASE_COUNTS = {
    profiles.STEADY_FLOW_PROFILE: 1200,
    profiles.TRANSIENT_DRYING_PROFILE: 660,
}
PAIRED_EQUIVALENCE_SEED = 9930
PILOT_CAMPAIGN_PURPOSE = "pilot_check"
PILOT_CAMPAIGN_SEED = 9940
PILOT_CASE_KINDS = ("nominal_reference", "natural_pilot")
NO_EVALUATION_REGIME = "not_applicable"
_FINAL_PHYSICAL_FORMULAS = {
    "w_surf_balance": "f_surf*d(w_surf)/dt = j_int - m_evap",
    "w_int_balance": "(1-f_surf)*d(w_int)/dt = -j_int",
    "w_gr": "f_surf*w_surf + (1-f_surf)*w_int",
    "w_gr_0": "rho_bu_dry*X_0_db_field; w_surf(0)=w_int(0)=w_gr_0",
    "r_surf": "r_surf_0",
    "r_int": "r_int_surf*r_surf",
    "j_int": "(1-f_surf)*r_int*(w_int-w_surf)",
    "m_evap": "f_surf*r_surf*max(w_surf-w_eq,0)",
    "X_db": "w_gr/rho_bu_dry",
    "X_wb": "w_gr/(rho_bu_dry+w_gr)",
    "X_wb_from_X_db": "X_db/(1+X_db)",
    "X_db_from_X_wb": "X_wb/(1-X_wb)",
    "X_wb_bulk": "integral(w_gr)/(integral(rho_bu_dry)+integral(w_gr))",
    "rho_bu_dry": "rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)",
    "solid_phase_density": "rho_bu_dry/(1-eps_bed)",
    "cp_gr_eff": "cp_gr_dry + X_db*cp_w",
    "volumetric_heat_capacity": "rho_bu_dry*cp_gr_eff",
    "k_eff": "k_gr*(2*k_gr+k_air-2*eps_bed*(k_gr-k_air))/(2*k_gr+k_air+eps_bed*(k_gr-k_air))",
    "phi_eff": "min(max(phi,1e-6),0.999)",
    "X_eq_db": "0.01*(A_osw+B_osw*(T-273.15[K]))*(phi_eff/(1-phi_eff))^C_osw",
    "w_eq": "rho_bu_dry*X_eq_db",
    "osw_ratio_0": "(100*X_0_db_field/(A_osw+B_osw*(T_init-273.15[K])))^(1/C_osw)",
    "phi_init": "osw_ratio_0/(1+osw_ratio_0)",
    "Q_evap": "-h_fg*m_evap",
    "f_wet_dm": "integral(rho_bu_dry*indicator(X_wb>X_target_wb))/integral(rho_bu_dry)",
    "total_water_balance": "d/dt(m_w_gr+m_v_gas)=m_dot_v_in-m_dot_v_out",
}
_STEADY_FLOW_AUDIT_DEPENDENCIES = (
    "Kxx",
    "Kxy",
    "Kyy",
    "eps_bed",
    "p_in_bc",
    "T_flow_ref",
    "p_ref",
    "p_out",
)
SPLIT_NAMES = (
    "train",
    "validation",
    "id_test",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
    "technical_smoke",
)
EVALUATION_REGIMES = (
    "id",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
)
MATERIAL_ROLES = (
    "seen",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
)
CAMPAIGN_PURPOSES = ("family_generalization", "technical_runtime_smoke", PILOT_CAMPAIGN_PURPOSE)
_SEEN_SPLITS = frozenset({"train", "validation", "id_test", "parameter_ood"})
_COMSOL_OWNED_ARGUMENTS = ("-inputfile", "-outputfile", "-np", "-nn", "-nnhost", "-mpihosts")
_SCHEDULER_OWNED_OPTIONS = (
    "--parsable",
    "--nodes",
    "--ntasks",
    "--ntasks-per-node",
    "--cpus-per-task",
    "--array",
    "--chdir",
    "--job-name",
    "--partition",
    "--time",
    "--output",
    "--error",
    "--export",
    "--wrap",
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """One resolved material/regime generation batch plus separate execution."""

    source_path: Path
    profile: profiles.SimulationProfile
    material_family: str
    material_role: str
    evaluation_regime: str
    sampling_regime: str
    batch_name: str
    scientific_values: dict[str, Any]
    execution_values: dict[str, Any]
    template_path: Path
    template_sha256: str
    case_indices: tuple[int, ...]
    seed_base: int | None
    assignments: dict[int, dict[str, Any]]
    scientific_config_digest: str
    case_input_config_digest: str
    batch_identity: str
    batch_id: str

    def case_id(self, case_index: int) -> str:
        """Return the canonical directory identifier for one batch member."""
        if case_index not in self.case_indices:
            message = f"Case index {case_index} is not a member of batch {self.batch_id}."
            raise ValueError(message)
        return f"case_{case_index:0{CASE_ID_WIDTH}d}"

    def case_seed(self, case_index: int) -> int:
        """Return the label-derived master seed for one executable case."""
        self.case_id(case_index)
        if self.seed_base is None:
            message = f"Batch {self.batch_name!r} has no resolved seed."
            raise ValueError(message)
        assignment = self.assignments[case_index]
        return derive_seed(
            self.seed_base,
            "case",
            self.profile.id,
            self.material_family,
            self.sampling_regime,
            str(assignment["assignment_role"]),
            str(case_index),
        )

    def case_assignment(self, case_index: int) -> dict[str, Any]:
        """Return an isolated material, regime, and OOD assignment."""
        self.case_id(case_index)
        return copy.deepcopy(self.assignments[case_index])


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """One resolved campaign plan with immutable scientific subbatches."""

    source_path: Path
    campaign_name: str
    campaign_digest: str
    campaign_id: str
    campaign_purpose: str
    evaluation_regimes: tuple[str, ...]
    profile: profiles.SimulationProfile
    material_roles: dict[str, tuple[str, ...]]
    material_memberships: dict[str, tuple[str, ...]]
    membership: dict[str, Any]
    source_registry: dict[str, dict[str, Any]]
    total_case_count: int
    paired_equivalence_seed: int | None
    batches: tuple[GenerationConfig, ...]
    dataset_packages: tuple[dict[str, Any], ...]
    duplicate_case_input_policy: str
    execution_values: dict[str, Any]

    def batch(self, batch_name: str) -> GenerationConfig:
        """Select one predeclared generation batch by its human-readable name."""
        for batch in self.batches:
            if batch.batch_name == batch_name:
                return batch
        available = ", ".join(item.batch_name for item in self.batches)
        message = f"Batch {batch_name!r} is not declared by campaign {self.campaign_name!r}. Available: {available}."
        raise ValueError(message)

    def select_batches(self, batch_names: tuple[str, ...] | None) -> CampaignConfig:
        """Return an internally complete execution view over declared batches."""
        if batch_names is None:
            return self
        if not batch_names or len(batch_names) != len(set(batch_names)):
            message = "Campaign batch selection must be non-empty and duplicate-free."
            raise ValueError(message)
        selected = tuple(self.batch(name) for name in batch_names)
        available_sources = {(batch.material_family, batch.sampling_regime) for batch in selected}
        packages = tuple(
            copy.deepcopy(package)
            for package in self.dataset_packages
            if all(
                (
                    material_family,
                    "parameter_ood" if package["evaluation_regime"] == "parameter_ood" else "natural",
                )
                in available_sources
                for material_family in package["materials"]
            )
        )
        return replace(
            self,
            total_case_count=sum(len(batch.case_indices) for batch in selected),
            batches=selected,
            dataset_packages=packages,
        )

    def without_extreme_family_ood(self) -> CampaignConfig:
        """Return the optional compute-saving view without extreme-family cases."""
        if self.campaign_purpose != "family_generalization":
            message = "Extreme-family skipping is available only for family-generalization execution."
            raise ValueError(message)
        selected = tuple(batch.batch_name for batch in self.batches if batch.evaluation_regime != "extreme_family_ood")
        if len(selected) == len(self.batches):
            message = "Campaign has no extreme-family OOD batch to skip."
            raise ValueError(message)
        return self.select_batches(selected)

    def with_wall_time(self, wall_time: str | None) -> CampaignConfig:
        """Return an execution-only campaign view with one scheduler time limit."""
        if wall_time is None:
            return self
        validated = _safe_text_or_none(wall_time, label="execution.cluster.wall_time")
        execution = copy.deepcopy(self.execution_values)
        execution["cluster"]["wall_time"] = validated
        return replace(self, execution_values=execution)


class GenerationConfigError(ValueError):
    """Report one invalid layered generation configuration."""


def derive_seed(seed_base: int, *labels: str) -> int:
    """Derive one stable uint32 seed from an integer base and semantic labels."""
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or not 0 <= seed_base <= UINT32_MAX:
        message = f"seed_base must be an integer in the uint32 range, got {seed_base!r}."
        raise ValueError(message)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        message = "Seed derivation requires one or more non-empty labels."
        raise ValueError(message)
    payload = f"{GENERATOR_VERSION}|{seed_base}|" + "|".join(labels)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], byteorder="big", signed=False)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one isolated mapping with string keys."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _exact_keys(value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    """Require all declared keys and reject unknown configuration."""
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise GenerationConfigError(message)


def _finite(value: Any, *, label: str) -> float:
    """Return one finite non-boolean real value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must be a finite real value, got {value!r}."
        raise GenerationConfigError(message)
    return float(value)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    """Return one bounded non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{label} must be an integer >= {minimum}, got {value!r}."
        raise GenerationConfigError(message)
    return value


def validate_relative_file(value: Any, *, label: str, suffix: str | None = None) -> str:
    """Return one safe case-local filename."""
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{label} must be one non-empty case-local filename."
        raise GenerationConfigError(message)
    path = Path(value)
    if path.name != value or path.is_absolute() or value in {".", ".."} or "\x00" in value:
        message = f"{label} must be one non-empty case-local filename, got {value!r}."
        raise GenerationConfigError(message)
    if suffix is not None and path.suffix.lower() != suffix:
        message = f"{label} must end in {suffix!r}, got {value!r}."
        raise GenerationConfigError(message)
    return value


def _delimiter(value: Any, *, label: str) -> str:
    """Return one supported deterministic text-table delimiter."""
    if value == "\t":
        return "	"
    if value not in {",", ";", "	"}:
        message = f"{label} must be ',', ';', or '\t'."
        raise GenerationConfigError(message)
    return str(value)


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required YAML mapping."""
    if not path.is_file():
        message = f"Required {label} does not exist: {path}"
        raise FileNotFoundError(message)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not load {label}: {path}"
        raise GenerationConfigError(message) from error
    return _mapping(loaded, label=label)


def _diagnostic_path(path: Path) -> str:
    """Return a repository-relative owner path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(common.paths.get_project_root().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _diagnostic_key(message: str) -> str:
    """Extract the most specific authored key named by one validation error."""
    candidates = re.findall(
        r"(?:campaign|common|operations|profile|execution|registry|material(?:\[[a-z_]+\])?)"
        r"(?:\.[A-Za-z_][A-Za-z0-9_-]*|\[[0-9]+\])+",
        message,
    )
    if candidates:
        return max(candidates, key=len)
    for key in (
        "steady_flow_conditioning",
        "simulation_profile",
        "material_family",
        "campaign_purpose",
    ):
        if key in message:
            layer = "profile" if key in {"steady_flow_conditioning", "simulation_profile"} else "campaign"
            return f"{layer}.{key}"
    return "$"


def _diagnostic_value(value: Any, key: str) -> Any:
    """Look up one dotted/indexed diagnostic key without mutating authored data."""
    if key == "$":
        return value
    current = value
    for name, index in re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)(?:\[([0-9]+)\])?", key):
        if not isinstance(current, Mapping) or name not in current:
            return "<missing>"
        current = current[name]
        if index:
            if not isinstance(current, list) or int(index) >= len(current):
                return "<missing>"
            current = current[int(index)]
    return current


def validation_error_details(path: Path | str, error: Exception) -> dict[str, Any]:
    """Return exact file, key, rule, value, and authored owner for a CLI error."""
    campaign_path = Path(path).expanduser().resolve()
    try:
        campaign_raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        campaign_raw = None
    message = str(error)
    key = _diagnostic_key(message)
    layer = key.split(".", maxsplit=1)[0].split("[", maxsplit=1)[0]
    owner_path = campaign_path
    local_key = key
    owner_raw = campaign_raw
    references = {
        "common": "common_config",
        "sources": "sources_config",
        "registry": "registry_config",
        "operations": "operations_config",
        "profile": "profile_config",
        "execution": "execution_config",
    }
    if isinstance(campaign_raw, Mapping) and layer in references:
        reference = campaign_raw.get(references[layer])
        if isinstance(reference, str):
            owner_path = _reference_path(
                reference,
                source_path=campaign_path,
                label=f"campaign.{references[layer]}",
            )
            try:
                owner_raw = yaml.safe_load(owner_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                owner_raw = None
            local_key = key.removeprefix(f"{layer}.")
    elif layer == "material":
        match = re.match(r"material\[([a-z_]+)\]", key)
        if match is not None:
            family = match.group(1)
            owner_path = common.paths.get_project_root() / "configs" / "generation" / "materials" / f"{family}.yaml"
            try:
                owner_raw = yaml.safe_load(owner_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                owner_raw = None
            local_key = key.removeprefix(f"material[{family}].")
    actual = _diagnostic_value(owner_raw, local_key)
    file_name = _diagnostic_path(owner_path)
    return {
        "error": "generation_config_validation",
        "file": file_name,
        "key": local_key,
        "expected_type_or_rule": message,
        "actual_value": actual,
        "owner_to_edit": file_name,
    }


def _reference_path(value: Any, *, source_path: Path, label: str) -> Path:
    """Resolve one explicit layered configuration reference."""
    if not isinstance(value, str) or not value:
        message = f"{label} must be non-empty path text."
        raise GenerationConfigError(message)
    configured = Path(value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    local = (source_path.parent / configured).resolve()
    if local.is_file():
        return local
    return (common.paths.get_project_root() / configured).resolve()


def _validate_grid(value: Any) -> dict[str, Any]:
    """Validate the authoritative grid and derive its two exact spacings."""
    grid = _mapping(value, label="common.grid")
    authored = {"nx", "ny", "Lx", "Ly", "Lz", "boundaries_included"}
    _exact_keys(grid, required=authored, optional=set(), label="common.grid")
    normalized = {
        "nx": _integer(grid["nx"], label="common.grid.nx", minimum=2),
        "ny": _integer(grid["ny"], label="common.grid.ny", minimum=2),
        "Lx": _finite(grid["Lx"], label="common.grid.Lx"),
        "Ly": _finite(grid["Ly"], label="common.grid.Ly"),
        "Lz": _finite(grid["Lz"], label="common.grid.Lz"),
        "boundaries_included": grid["boundaries_included"],
    }
    normalized["dx"] = normalized["Lx"] / (normalized["nx"] - 1)
    normalized["dy"] = normalized["Ly"] / (normalized["ny"] - 1)
    required = {
        "nx": 401,
        "ny": 251,
        "Lx": 1.2,
        "Ly": 0.75,
        "Lz": 0.8,
        "boundaries_included": True,
        "dx": 0.003,
        "dy": 0.003,
    }
    if normalized != required:
        message = f"Grid must resolve to the authoritative boundary-inclusive 401x251 contract: {required}."
        raise GenerationConfigError(message)
    return normalized


def _validate_time(value: Any) -> dict[str, Any]:
    """Validate the authoritative hourly 0-through-168-hour contract."""
    time = _mapping(value, label="common.time")
    expected = {"start", "stop", "interval", "internal_steps", "irregular_stop_state"}
    _exact_keys(time, required=expected, optional=set(), label="common.time")
    normalized = {
        "start": _finite(time["start"], label="common.time.start"),
        "stop": _finite(time["stop"], label="common.time.stop"),
        "interval": _finite(time["interval"], label="common.time.interval"),
        "internal_steps": time["internal_steps"],
        "irregular_stop_state": time["irregular_stop_state"],
    }
    expected_values = {
        "start": 0.0,
        "stop": 168.0,
        "interval": 1.0,
        "internal_steps": "adaptive",
        "irregular_stop_state": "diagnostic_only",
    }
    if normalized != expected_values:
        message = f"Time contract must be hourly 0..168 h with adaptive internal steps: {expected_values}."
        raise GenerationConfigError(message)
    normalized["regular_times"] = [float(value) for value in range(169)]
    return normalized


def _validate_scientific_fixed(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate fixed thermodynamic, humidity, and stopping records."""
    fixed = _mapping(value, label="common.scientific_fixed_values")
    units = {
        "T_flow_ref": "K",
        "p_ref": "Pa",
        "p_out": "Pa",
        "T_in_min": "K",
        "T_in_max": "K",
        "omega_min": "kg/kg",
        "omega_max": "kg/kg",
        "phi_operational_min": "1",
        "phi_operational_max": "1",
        "phi_clip_min": "1",
        "phi_clip_max": "1",
        "cp_w": "J/(kg*K)",
        "h_fg": "J/kg",
        "D_v_air": "m^2/s",
        "M_v": "kg/mol",
        "d_wall": "m",
        "k_wall": "W/(m*K)",
        "h_ext": "W/(m^2*K)",
        "U_wall": "W/(m^2*K)",
        "f_wet_dm_max": "1",
        "schedule_interpolation": "method",
    }
    _exact_keys(fixed, required=set(units), optional=set(), label="common.scientific_fixed_values")
    records: dict[str, dict[str, Any]] = {}
    resolved_values: dict[str, Any] = {}
    for name, expected_unit in units.items():
        record = materials.resolve_value_record(
            fixed[name],
            sources=sources,
            label=f"common.scientific_fixed_values.{name}",
        )
        optional = {"boundary"} if name in {"omega_min", "omega_max"} else set()
        required = {"unit", "provenance"} if name == "U_wall" else {"value", "unit", "provenance"}
        _exact_keys(
            record,
            required=required,
            optional=optional,
            label=f"common.scientific_fixed_values.{name}",
        )
        if record["unit"] != expected_unit:
            message = f"common.scientific_fixed_values.{name}.unit must be {expected_unit!r}."
            raise GenerationConfigError(message)
        if "boundary" in record:
            boundary = _mapping(record["boundary"], label=f"common.scientific_fixed_values.{name}.boundary")
            _exact_keys(
                boundary,
                required={"boundary_kind", "boundary_basis", "hard_boundary"},
                optional=set(),
                label=f"common.scientific_fixed_values.{name}.boundary",
            )
            if (
                boundary["boundary_kind"] != "engineering_source_air_envelope"
                or boundary["boundary_basis"] != "source_air_dew_point_range"
                or boundary["hard_boundary"] is not True
            ):
                message = f"common.scientific_fixed_values.{name}.boundary violates the supplied hard-envelope contract."
                raise GenerationConfigError(message)
            record["boundary"] = boundary
        if name == "U_wall":
            provenance = record["provenance"]
            derivation = provenance["derivation"]
            if (
                provenance["status"] != "derived"
                or derivation["kind"] != "derived_from_configured_value"
                or derivation["verification"] != "mathematically_reproduced"
            ):
                message = "common.scientific_fixed_values.U_wall must retain its reproduced supplied derivation."
                raise GenerationConfigError(message)
        elif name == "schedule_interpolation":
            if record["value"] != "linear":
                message = "The maintained schedule interpolation must be linear between hourly nodes."
                raise GenerationConfigError(message)
            resolved_values[name] = "linear"
        else:
            resolved_values[name] = _finite(record["value"], label=f"common.scientific_fixed_values.{name}.value")
        records[name] = record

    resolved_values["U_wall"] = 1.0 / (resolved_values["d_wall"] / resolved_values["k_wall"] + 1.0 / resolved_values["h_ext"])
    records["U_wall"]["value"] = resolved_values["U_wall"]
    if resolved_values["T_flow_ref"] != profiles.STATIONARY_FLOW_REFERENCE_TEMPERATURE:
        message = "common.scientific_fixed_values.T_flow_ref must be the package-fixed 300.65 K."
        raise GenerationConfigError(message)
    if resolved_values["p_ref"] != profiles.STATIONARY_FLOW_REFERENCE_PRESSURE:
        message = "common.scientific_fixed_values.p_ref must match the package-fixed 101325 Pa template contract."
        raise GenerationConfigError(message)
    if resolved_values["p_out"] != profiles.STATIONARY_FLOW_OUTLET_PRESSURE:
        message = "common.scientific_fixed_values.p_out must match the package-fixed 0 Pa template contract."
        raise GenerationConfigError(message)
    expected_constants = {
        "T_in_min": _EXPECTED_T_IN_MIN,
        "T_in_max": _EXPECTED_T_IN_MAX,
        "cp_w": 4180.0,
        "h_fg": 2418200.0,
        "D_v_air": 2.811e-05,
        "M_v": 0.01801528,
        "d_wall": 0.019,
        "k_wall": 0.13,
        "h_ext": 8.0,
        "U_wall": 3.687943262411348,
        "f_wet_dm_max": _EXPECTED_F_WET_DM_MAX,
    }
    if any(resolved_values[name] != expected for name, expected in expected_constants.items()):
        message = f"Scientific fixed values must match the binding VP2 constants {expected_constants}."
        raise GenerationConfigError(message)
    humidity = tuple(
        resolved_values[name]
        for name in (
            "omega_min",
            "omega_max",
            "phi_operational_min",
            "phi_operational_max",
            "phi_clip_min",
            "phi_clip_max",
        )
    )
    if not (
        0 <= humidity[0] < humidity[1]
        and (humidity[2], humidity[3]) == (0.05, 0.85)
        and 0 <= humidity[4] < humidity[2] < humidity[3] < humidity[5] <= 1
    ):
        message = "Operational humidity bounds and numerical Oswin clipping limits are invalid or conflated."
        raise GenerationConfigError(message)
    return resolved_values, records


def _validate_input_contract(value: Any) -> dict[str, Any]:
    """Validate canonical profile-specific spatial, scalar, and schedule adapters."""
    contract = _mapping(value, label="common.input_contract")
    _exact_keys(contract, required={"spatial", "scalar", "schedule"}, optional=set(), label="common.input_contract")
    expected_filenames = {"spatial": "fields.csv", "scalar": "scalars.csv", "schedule": "schedule.csv"}
    filenames: set[str] = set()
    normalized: dict[str, Any] = {}
    for name in ("spatial", "scalar", "schedule"):
        adapter = _mapping(contract[name], label=f"common.input_contract.{name}")
        _exact_keys(adapter, required={"filename", "delimiter", "columns"}, optional=set(), label=f"common.input_contract.{name}")
        filename = validate_relative_file(adapter["filename"], label=f"common.input_contract.{name}.filename", suffix=".csv")
        if filename != expected_filenames[name]:
            message = f"{name} adapter must use canonical filename {expected_filenames[name]!r}."
            raise GenerationConfigError(message)
        if filename in filenames:
            msg = "Input adapter filenames must be unique."
            raise GenerationConfigError(msg)
        filenames.add(filename)
        normalized[name] = {
            "filename": filename,
            "delimiter": _delimiter(adapter["delimiter"], label=f"common.input_contract.{name}.delimiter"),
        }
        columns = adapter["columns"]
        if name == "spatial":
            by_profile = _mapping(columns, label="common.input_contract.spatial.columns")
            expected_profiles = set(profiles.available_profiles())
            _exact_keys(by_profile, required=expected_profiles, optional=set(), label="common.input_contract.spatial.columns")
            normalized["spatial"]["columns_by_profile"] = {}
            for profile_id in profiles.available_profiles():
                expected = list(profiles.spatial_input_fields(profile_id))
                if by_profile[profile_id] != expected:
                    message = f"Spatial adapter for {profile_id!r} must use exact columns {expected}."
                    raise GenerationConfigError(message)
                normalized["spatial"]["columns_by_profile"][profile_id] = expected
        else:
            expected = ["name", "value", "unit"] if name == "scalar" else list(profiles.SCHEDULE_FIELDS)
            if columns != expected:
                message = f"{name} adapter must use exact columns {expected}."
                raise GenerationConfigError(message)
            normalized[name]["columns"] = list(columns)
    return normalized


def _validate_storage(value: Any) -> dict[str, Any]:
    """Validate the canonical HDF5 conversion and layout settings."""
    storage = _mapping(value, label="common.storage")
    expected = {
        "schema_version",
        "converter_version",
        "compression",
        "compression_level",
        "shuffle",
        "float32_rtol",
        "float32_atol",
        "chunk_time",
        "chunk_y",
        "chunk_x",
    }
    _exact_keys(storage, required=expected, optional=set(), label="common.storage")
    storage["schema_version"] = _integer(storage["schema_version"], label="common.storage.schema_version", minimum=1)
    storage["converter_version"] = _integer(
        storage["converter_version"],
        label="common.storage.converter_version",
        minimum=1,
    )
    if storage["schema_version"] != CANONICAL_HDF5_SCHEMA_VERSION or storage["converter_version"] != CANONICAL_HDF5_CONVERTER_VERSION:
        msg = f"Canonical storage must use schema {CANONICAL_HDF5_SCHEMA_VERSION} and converter {CANONICAL_HDF5_CONVERTER_VERSION!r}."
        raise GenerationConfigError(msg)
    if storage["compression"] != "gzip" or storage["compression_level"] != _EXPECTED_HDF5_COMPRESSION_LEVEL or storage["shuffle"] is not True:
        msg = "Canonical case fields require gzip level 4 with shuffle enabled."
        raise GenerationConfigError(msg)
    for key in ("float32_rtol", "float32_atol"):
        storage[key] = _finite(storage[key], label=f"common.storage.{key}")
        if storage[key] < 0:
            msg = f"common.storage.{key} must be non-negative."
            raise GenerationConfigError(msg)
    for key in ("chunk_time", "chunk_y", "chunk_x"):
        storage[key] = _integer(storage[key], label=f"common.storage.{key}", minimum=1)
    if storage["chunk_time"] > _MAXIMUM_TIME_CHUNK:
        msg = "common.storage.chunk_time must optimize one or two time-step reads."
        raise GenerationConfigError(msg)
    return storage


def _validate_provenance_owners(
    value: Any,
    *,
    expected_names: set[str],
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Resolve one exact keyed provenance owner without fabricating absent fields."""
    owners = _mapping(value, label=label)
    _exact_keys(owners, required=expected_names, optional=set(), label=label)
    return {
        name: provenance_service.resolve_provenance(
            provenance,
            sources=sources,
            label=f"{label}.{name}",
        )
        for name, provenance in owners.items()
    }


def _validate_common(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the global material-independent scientific owner."""
    common_config = _mapping(value, label="generation common configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "generator_version",
        "decision_source",
        "parameter_values",
        "scientific_fixed_values",
        "physical_formulas",
        "physical_formulas_provenance",
        "grid",
        "grid_provenance",
        "time",
        "time_provenance",
        "input_contract",
        "storage",
    }
    _exact_keys(common_config, required=expected, optional=set(), label="generation common configuration")
    if common_config["schema_kind"] != "generation_common" or common_config["schema_version"] != 1:
        message = "Unsupported generation common configuration schema."
        raise GenerationConfigError(message)
    common_config["generator_version"] = _integer(
        common_config["generator_version"],
        label="generation common generator_version",
        minimum=1,
    )
    if common_config["generator_version"] != GENERATOR_VERSION:
        message = f"generator_version must be {GENERATOR_VERSION!r}."
        raise GenerationConfigError(message)
    common_config["decision_source"] = materials.validate_decision_source(
        common_config["decision_source"],
        label="generation common decision_source",
    )
    common_config["parameter_values"] = _mapping(common_config["parameter_values"], label="common.parameter_values")
    if set(common_config["parameter_values"]) != {"eps_min_global", "eps_max_global"}:
        message = "common.parameter_values must own exactly the two global porosity guards."
        raise GenerationConfigError(message)
    for name, record in common_config["parameter_values"].items():
        materials.resolve_value_record(record, sources=sources, label=f"common.parameter_values.{name}")

    formulas = _mapping(common_config["physical_formulas"], label="common.physical_formulas")
    _exact_keys(formulas, required=set(_FINAL_PHYSICAL_FORMULAS), optional=set(), label="common.physical_formulas")
    if formulas != _FINAL_PHYSICAL_FORMULAS:
        message = "common.physical_formulas must match the frozen final VP2 formula contract exactly."
        raise GenerationConfigError(message)
    common_config["physical_formulas"] = formulas
    common_config["physical_formulas_provenance"] = provenance_service.resolve_provenance(
        common_config["physical_formulas_provenance"],
        sources=sources,
        label="common.physical_formulas_provenance",
    )

    fixed_values, fixed_records = _validate_scientific_fixed(
        common_config["scientific_fixed_values"],
        sources=sources,
    )
    common_config["scientific_fixed_values"] = fixed_values
    common_config["_scientific_fixed_records"] = fixed_records
    common_config["grid"] = _validate_grid(common_config["grid"])
    common_config["grid_provenance"] = _validate_provenance_owners(
        common_config["grid_provenance"],
        expected_names=set(common_config["grid"]),
        sources=sources,
        label="common.grid_provenance",
    )
    common_config["time"] = _validate_time(common_config["time"])
    common_config["time_provenance"] = _validate_provenance_owners(
        common_config["time_provenance"],
        expected_names={"start", "stop", "interval", "internal_steps", "irregular_stop_state"},
        sources=sources,
        label="common.time_provenance",
    )
    common_config["input_contract"] = _validate_input_contract(common_config["input_contract"])
    common_config["storage"] = _validate_storage(common_config["storage"])
    return common_config


def _validate_steady_flow_conditioning(
    value: Any,
    *,
    fixed_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exhaustive package-fixed stationary-airflow conditioning."""
    conditioning = _mapping(
        value,
        label="generation profile steady_flow_conditioning",
    )
    _exact_keys(
        conditioning,
        required={
            "schema_kind",
            "schema_version",
            "exhaustive",
            "stationary_solution_contract_id",
            "dependencies",
            "additional_case_varying_solver_scalars",
        },
        optional=set(),
        label="generation profile steady_flow_conditioning",
    )
    if (
        conditioning["schema_kind"] != "steady_flow_conditioning"
        or conditioning["schema_version"] != 1
        or conditioning["exhaustive"] is not True
        or conditioning["stationary_solution_contract_id"] != "vp2_stationary_airflow_v1"
    ):
        message = "The steady-flow conditioning header is not the binding exhaustive VP2 contract."
        raise GenerationConfigError(message)
    dependencies = conditioning["dependencies"]
    if not isinstance(dependencies, list):
        message = "steady_flow_conditioning.dependencies must be an ordered list."
        raise TypeError(message)
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(dependencies):
        label = f"steady_flow_conditioning.dependencies[{index}]"
        dependency = _mapping(raw, label=label)
        _exact_keys(
            dependency,
            required={"name", "affects_stationary_solution", "owner", "unit"},
            optional={"fixed_value"},
            label=label,
        )
        name = dependency["name"]
        if not isinstance(name, str) or not name:
            message = f"{label}.name must be non-empty text."
            raise TypeError(message)
        if dependency["affects_stationary_solution"] is not True:
            message = f"{label} must explicitly affect the stationary solution."
            raise GenerationConfigError(message)
        owner = dependency["owner"]
        if owner not in {"model_input", "package_fixed"}:
            message = f"{label}.owner must be model_input or package_fixed."
            raise GenerationConfigError(message)
        unit = dependency["unit"]
        if not isinstance(unit, str) or not unit:
            message = f"{label}.unit must be non-empty text."
            raise TypeError(message)
        if owner == "model_input":
            if "fixed_value" in dependency:
                message = f"{label} model inputs cannot author fixed_value."
                raise GenerationConfigError(message)
        elif "fixed_value" not in dependency:
            message = f"{label} package-fixed dependencies require fixed_value."
            raise GenerationConfigError(message)
        else:
            dependency["fixed_value"] = _finite(
                dependency["fixed_value"],
                label=f"{label}.fixed_value",
            )
        validated.append(dependency)
    names = tuple(str(dependency["name"]) for dependency in validated)
    if names != _STEADY_FLOW_AUDIT_DEPENDENCIES:
        message = f"steady_flow_conditioning.dependencies must be exactly {list(_STEADY_FLOW_AUDIT_DEPENDENCIES)} in order."
        raise GenerationConfigError(message)
    additional = _name_list(
        conditioning["additional_case_varying_solver_scalars"],
        label="steady_flow_conditioning.additional_case_varying_solver_scalars",
        allow_empty=True,
    )
    if additional:
        message = "The canonical steady template permits no additional case-varying solver scalars."
        raise GenerationConfigError(message)
    declared = {str(dependency["name"]): dependency for dependency in validated}
    expected_model_inputs = {
        "Kxx": "m^2",
        "Kxy": "m^2",
        "Kyy": "m^2",
        "eps_bed": "1",
        "p_in_bc": "Pa",
    }
    for name, unit in expected_model_inputs.items():
        if declared[name] != {
            "name": name,
            "affects_stationary_solution": True,
            "owner": "model_input",
            "unit": unit,
        }:
            message = f"Steady-flow conditioning for {name!r} must be owned only by fields.csv."
            raise GenerationConfigError(message)
    for name, unit in zip(
        profiles.STATIONARY_FIXED_FIELDS,
        profiles.STATIONARY_FIXED_UNITS,
        strict=True,
    ):
        expected_value = profiles.STATIONARY_FIXED_VALUES[name]
        if float(fixed_values[name]) != expected_value or declared[name] != {
            "name": name,
            "affects_stationary_solution": True,
            "owner": "package_fixed",
            "unit": unit,
            "fixed_value": expected_value,
        }:
            message = f"Steady-flow conditioning for {name!r} disagrees with the canonical template contract."
            raise GenerationConfigError(message)
    conditioning["dependencies"] = validated
    conditioning["additional_case_varying_solver_scalars"] = []
    return conditioning


_MAPPING_STATES = (
    "declared_unverified",
    "runtime_confirmed",
    "mapping_probe_required",
)


def _validate_mapping_node(
    value: Any,
    *,
    label: str,
    value_key: str,
) -> dict[str, str]:
    """Validate one typed source or header mapping without a null sentinel."""
    node = _mapping(value, label=label)
    state = node.get("state")
    if state not in _MAPPING_STATES:
        message = f"{label}.state must be one of {list(_MAPPING_STATES)}."
        raise GenerationConfigError(message)
    required = {"state"} if state == "mapping_probe_required" else {"state", value_key}
    _exact_keys(node, required=required, optional=set(), label=label)
    if state == "mapping_probe_required":
        return {"state": state}
    mapped = node[value_key]
    if not isinstance(mapped, str) or not mapped or mapped.strip() != mapped or any(character in mapped for character in ("\x00", "\n", "\r")):
        message = f"{label}.{value_key} must be safe non-empty text."
        raise GenerationConfigError(message)
    if value_key == "pattern":
        candidate = Path(mapped)
        if (
            candidate.name != mapped
            or candidate.is_absolute()
            or mapped in {".", ".."}
            or any(character in mapped for character in ("*", "?", "[", "]"))
        ):
            message = f"{label}.pattern must be one exact case-local filename."
            raise GenerationConfigError(message)
    return {"state": state, value_key: mapped}


def _mapping_rollup(
    source_mapping: Mapping[str, str],
    column_mappings: Mapping[str, Mapping[str, str]],
) -> str:
    """Return the least-confirmed state across one export role."""
    states = [
        str(source_mapping["state"]),
        *(str(mapping["state"]) for mapping in column_mappings.values()),
    ]
    if "mapping_probe_required" in states:
        return "mapping_probe_required"
    if "declared_unverified" in states:
        return "declared_unverified"
    return "runtime_confirmed"


def _validate_profile_config(
    value: Any,
    *,
    profile: profiles.SimulationProfile,
    fixed_values: Mapping[str, Any],
    require_executable: bool,
) -> dict[str, Any]:
    """Validate typed export mappings without inventing missing headers."""
    if not isinstance(require_executable, bool):
        message = "require_executable must be boolean."
        raise TypeError(message)
    config = _mapping(value, label="generation profile configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "simulation_profile",
        "steady_flow_conditioning",
        "exports",
    }
    _exact_keys(
        config,
        required=expected,
        optional=set(),
        label="generation profile configuration",
    )
    if config["schema_kind"] != "generation_profile" or config["schema_version"] != 1 or config["simulation_profile"] != profile.id:
        message = "Generation profile configuration schema or profile identity is invalid."
        raise GenerationConfigError(message)
    config["steady_flow_conditioning"] = _validate_steady_flow_conditioning(
        config["steady_flow_conditioning"],
        fixed_values=fixed_values,
    )
    exports = config["exports"]
    if not isinstance(exports, list):
        message = "generation profile exports must be an ordered list."
        raise TypeError(message)
    expected_roles = tuple(spec.role for spec in profile.export_roles)
    if tuple(raw.get("role") if isinstance(raw, Mapping) else None for raw in exports) != expected_roles:
        message = f"Profile {profile.id!r} must declare export roles exactly {list(expected_roles)} in order."
        raise GenerationConfigError(message)
    temporal_kinds = {
        profiles.STEADY_FLOW_EXPORT_ROLE: "stationary",
        profiles.TRANSIENT_RAW_EXPORT_ROLE: "regular_time_series",
        profiles.GLOBAL_EXPORT_ROLE: "regular_time_series",
        profiles.FINAL_STATUS_EXPORT_ROLE: "final_status",
        profiles.EXACT_STOP_EXPORT_ROLE: "irregular_stop_diagnostic",
    }
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(exports):
        label = f"generation profile exports[{index}]"
        export = _mapping(raw, label=label)
        _exact_keys(
            export,
            required={
                "role",
                "temporal_kind",
                "source",
                "delimiter",
                "columns",
            },
            optional=set(),
            label=label,
        )
        role = str(export["role"])
        role_spec = profile.export_role(role)
        temporal_kind = export["temporal_kind"]
        if temporal_kind != temporal_kinds[role]:
            message = f"{label}.temporal_kind must be {temporal_kinds[role]!r}."
            raise GenerationConfigError(message)
        source_mapping = _validate_mapping_node(
            export["source"],
            label=f"{label}.source",
            value_key="pattern",
        )
        columns = _mapping(export["columns"], label=f"{label}.columns")
        if tuple(columns) != role_spec.logical_fields:
            message = f"{label}.columns must map exact logical fields {list(role_spec.logical_fields)} in order."
            raise GenerationConfigError(message)
        column_mappings = {
            logical: _validate_mapping_node(
                mapping,
                label=f"{label}.columns.{logical}",
                value_key="source_header",
            )
            for logical, mapping in columns.items()
        }
        configured_sources = [mapping["source_header"] for mapping in column_mappings.values() if "source_header" in mapping]
        if len(configured_sources) != len(set(configured_sources)):
            message = f"{label}.columns must map known fields to distinct source headers."
            raise GenerationConfigError(message)
        probe_columns = [logical for logical, mapping in column_mappings.items() if mapping["state"] == "mapping_probe_required"]
        normalized: dict[str, Any] = {
            "role": role,
            "temporal_kind": temporal_kind,
            "source_mapping": source_mapping,
            "delimiter": _delimiter(
                export["delimiter"],
                label=f"{label}.delimiter",
            ),
            "column_mappings": column_mappings,
            "columns": {logical: mapping["source_header"] for logical, mapping in column_mappings.items() if "source_header" in mapping},
            "mapping_state": _mapping_rollup(
                source_mapping,
                column_mappings,
            ),
            "mapping_probe_required": {
                "source": source_mapping["state"] == "mapping_probe_required",
                "columns": probe_columns,
            },
            "required": role_spec.required,
            "allow_multiple": role_spec.allow_multiple,
            "units": dict(
                zip(
                    role_spec.logical_fields,
                    role_spec.units,
                    strict=True,
                )
            ),
        }
        if "pattern" in source_mapping:
            normalized["pattern"] = source_mapping["pattern"]
        validated.append(normalized)
    if require_executable:
        unresolved = [
            {
                "role": export["role"],
                "mapping_state": export["mapping_state"],
                "source_probe_required": export["mapping_probe_required"]["source"],
                "column_probe_required": export["mapping_probe_required"]["columns"],
            }
            for export in validated
            if export["required"] and export["mapping_state"] != "runtime_confirmed"
        ]
        if unresolved:
            message = f"Executable generation profile has unconfirmed required export mappings: {unresolved}."
            raise GenerationConfigError(message)
    config["exports"] = validated
    return config


def _validate_operations(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate material-independent operating-distribution ownership."""
    operations = _mapping(value, label="generation operations configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "operation_id",
        "decision_source",
        "constraints",
        "parameter_values",
    }
    _exact_keys(operations, required=expected, optional=set(), label="generation operations configuration")
    if operations["schema_kind"] != "generation_operations" or operations["schema_version"] != 1:
        message = "Unsupported generation operations schema."
        raise GenerationConfigError(message)
    if operations["operation_id"] != "fixed_bed":
        message = "The maintained operation configuration must use operation_id fixed_bed."
        raise GenerationConfigError(message)
    operations["decision_source"] = materials.validate_decision_source(
        operations["decision_source"],
        label="generation operations decision_source",
    )
    constraints = _mapping(operations["constraints"], label="operations.constraints")
    expected_constraints = {
        "heater_only": True,
        "humidity_ratio_conserved": "omega_source_air(t)=omega_in_bc(t)",
        "source_relative_humidity": "RH(T_amb,omega_in_bc(t),p_ref)",
        "inlet_relative_humidity": "RH(T_in_bc(t),omega_in_bc(t),p_ref)",
        "infeasible_schedule_policy": "reject_complete_schedule_and_deterministically_resample",
        "porosity_natural_support_policy": ("realized_mean_must_match_material_support_except_active_porosity_ood"),
    }
    _exact_keys(
        constraints,
        required=set(expected_constraints),
        optional=set(),
        label="operations.constraints",
    )
    if constraints != expected_constraints:
        message = "operations.constraints must match the supplied heater, humidity, rejection, and porosity policies."
        raise GenerationConfigError(message)
    operations["constraints"] = constraints

    parameter_values = _mapping(operations["parameter_values"], label="operations.parameter_values")
    material_owned = {
        "kappa_mean",
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "rho_bu_dry_ref",
        "eps_bed_cal_ref",
        "k_gr",
        "cp_gr_dry",
        "X_target_wb",
        "oswin",
        "r_surf_0",
        "r_int_surf",
        "f_surf",
    }
    expected_parameters = (
        set(materials.EXPECTED_PARAMETERS)
        .difference(materials.DERIVED_PARAMETERS)
        .difference(material_owned)
        .difference({"eps_min_global", "eps_max_global"})
    )
    if set(parameter_values) != expected_parameters:
        missing = sorted(expected_parameters.difference(parameter_values))
        unknown = sorted(set(parameter_values).difference(expected_parameters))
        message = f"operations.parameter_values ownership mismatch: missing={missing}, unknown={unknown}."
        raise GenerationConfigError(message)
    for name, record in parameter_values.items():
        materials.resolve_value_record(
            record,
            sources=sources,
            label=f"operations.parameter_values.{name}",
        )
    operations["parameter_values"] = parameter_values
    return operations


def _name_list(value: Any, *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    """Return one unique ordered list of non-empty identifiers."""
    if not isinstance(value, list) or (not value and not allow_empty):
        message = f"{label} must be {'an' if allow_empty else 'a non-'}empty ordered list."
        raise GenerationConfigError(message)
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        message = f"{label} must contain unique non-empty identifiers."
        raise GenerationConfigError(message)
    return names


def _uint32(value: Any, *, label: str) -> int:
    """Return one explicit uint32 seed."""
    number = _integer(value, label=label)
    if number > UINT32_MAX:
        message = f"{label} exceeds the uint32 range."
        raise GenerationConfigError(message)
    return number


def _validate_batch_counts(
    value: Any,
    *,
    materials_in_order: tuple[str, ...],
    seen: tuple[str, ...],
    regimes: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Validate one explicit positive count for every planned batch."""
    counts = _mapping(value, label="campaign.sampling.counts")
    _exact_keys(counts, required=set(regimes), optional=set(), label="campaign.sampling.counts")
    expected = {regime: materials_in_order if regime == "natural" else seen for regime in regimes}
    normalized: dict[str, dict[str, int]] = {}
    for regime, expected_materials in expected.items():
        raw = _mapping(counts[regime], label=f"campaign.sampling.counts.{regime}")
        if tuple(raw) != expected_materials:
            message = f"campaign.sampling.counts.{regime} must follow exact material order {list(expected_materials)}."
            raise GenerationConfigError(message)
        normalized[regime] = {
            material_family: _integer(
                count,
                label=f"campaign.sampling.counts.{regime}.{material_family}",
                minimum=1,
            )
            for material_family, count in raw.items()
        }
    return normalized


def _validate_binding_sampling_contract(
    *,
    campaign_purpose: str,
    profile_id: str,
    campaign_seed: int,
    counts: Mapping[str, Mapping[str, int]],
) -> None:
    """Require the supplied production, smoke, or pilot allocation exactly."""
    if campaign_purpose == "technical_runtime_smoke":
        expected_seed = 9910 if profile_id == profiles.STEADY_FLOW_PROFILE else 9920
        expected_counts = {"natural": {"lentil": 2}}
        if campaign_seed != expected_seed or counts != expected_counts:
            message = f"The {profile_id} technical smoke must use seed {expected_seed} and exactly two natural lentil cases."
            raise GenerationConfigError(message)
        return
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        case_counts = tuple(counts.get("natural", {}).values())
        expected_materials = set(materials.MATERIAL_FAMILIES)
        if (
            profile_id != profiles.TRANSIENT_DRYING_PROFILE
            or campaign_seed != PILOT_CAMPAIGN_SEED
            or set(counts) != {"natural"}
            or set(counts["natural"]) != expected_materials
            or not case_counts
            or len(set(case_counts)) != 1
        ):
            message = "The pilot-check campaign must use transient_drying, seed 9940, and one uniform positive count for all six materials."
            raise GenerationConfigError(message)
        return
    expected_seed = 9100 if profile_id == profiles.STEADY_FLOW_PROFILE else 9200
    expected_counts = (
        {
            "natural": {
                "lentil": 240,
                "chickpea": 240,
                "kidney_bean": 240,
                "field_pea": 80,
                "rapeseed": 80,
                "sunflower_seed": 80,
            },
            "parameter_ood": {
                "lentil": 80,
                "chickpea": 80,
                "kidney_bean": 80,
            },
        }
        if profile_id == profiles.STEADY_FLOW_PROFILE
        else {
            "natural": {
                "lentil": 120,
                "chickpea": 120,
                "kidney_bean": 120,
                "field_pea": 40,
                "rapeseed": 40,
                "sunflower_seed": 40,
            },
            "parameter_ood": {
                "lentil": 60,
                "chickpea": 60,
                "kidney_bean": 60,
            },
        }
    )
    if campaign_seed != expected_seed or counts != expected_counts:
        message = f"The {profile_id} family-generalization campaign does not match the binding final allocation and seed {expected_seed}."
        raise GenerationConfigError(message)
    expected_total = PRODUCTION_CASE_COUNTS[profile_id]
    if sum(count for regime in counts.values() for count in regime.values()) != expected_total:
        message = f"The {profile_id} family-generalization campaign must contain exactly {expected_total} COMSOL source cases."
        raise GenerationConfigError(message)


def _validate_parameter_ood_policy(
    value: Any,
    *,
    groups: tuple[str, ...],
) -> dict[str, Any]:
    """Validate generic eligible-unit allocation over profile-active OOD units."""
    policy = _mapping(value, label="campaign.sampling.parameter_ood")
    expected = {"groups", "units_per_case", "allocation_strategy", "eligibility_source"}
    _exact_keys(policy, required=expected, optional=set(), label="campaign.sampling.parameter_ood")
    if policy["groups"] != list(groups):
        message = f"Parameter-OOD groups for this profile must be exactly {list(groups)}."
        raise GenerationConfigError(message)
    policy["units_per_case"] = _integer(
        policy["units_per_case"],
        label="campaign.sampling.parameter_ood.units_per_case",
        minimum=1,
    )
    if policy["units_per_case"] != 1:
        message = "Exactly one parameter-OOD scalar unit or complete atomic record must be active per case."
        raise GenerationConfigError(message)
    if policy["allocation_strategy"] != "canonical_eligible_unit_round_robin":
        message = "Parameter OOD must use canonical eligible-unit round-robin allocation."
        raise GenerationConfigError(message)
    if policy["eligibility_source"] != "resolved_profile_projected_registry":
        message = "Parameter-OOD eligibility must come from the resolved profile-projected registry."
        raise GenerationConfigError(message)
    return policy


def dataset_package_name(dataset_view: str, material_families: tuple[str, ...], evaluation_regime: str) -> str:
    """Return the canonical human-readable dataset package name."""
    if not dataset_view or not material_families or evaluation_regime not in EVALUATION_REGIMES:
        message = "Dataset package naming inputs are invalid."
        raise GenerationConfigError(message)
    joined_materials = "+".join(material_families)
    return f"{dataset_view}__{joined_materials}__{evaluation_regime}"


def _expected_package_roles(
    material_roles: Mapping[str, tuple[str, ...]],
    *,
    campaign_purpose: str,
) -> tuple[tuple[str, str], ...]:
    """Return ordered evaluation-regime to material-role ownership."""
    if campaign_purpose == "technical_runtime_smoke":
        return (("id", "seen"),)
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        return ()
    expected = [("id", "seen"), ("parameter_ood", "seen")]
    expected.extend((role, role) for role in MATERIAL_ROLES[1:] if material_roles[role])
    return tuple(expected)


def _validate_membership(
    value: Any,
    *,
    campaign_purpose: str,
    profile_id: str,
) -> dict[str, Any]:
    """Validate one campaign-level immutable Seen-family membership contract."""
    if campaign_purpose in {"technical_runtime_smoke", PILOT_CAMPAIGN_PURPOSE}:
        if value is not None:
            message = f"{campaign_purpose} campaigns do not declare learning membership."
            raise GenerationConfigError(message)
        return {}
    membership = _mapping(value, label="campaign.membership")
    _exact_keys(
        membership,
        required={"seed", "per_seen_material"},
        optional=set(),
        label="campaign.membership",
    )
    expected_seed = 9150 if profile_id == profiles.STEADY_FLOW_PROFILE else 9250
    membership["seed"] = _uint32(membership["seed"], label="campaign.membership.seed")
    if membership["seed"] != expected_seed:
        message = f"The {profile_id} family-generalization membership seed must be {expected_seed}."
        raise GenerationConfigError(message)
    per_material = _mapping(
        membership["per_seen_material"],
        label="campaign.membership.per_seen_material",
    )
    expected_counts = (
        {"train": 192, "validation": 24, "id_test": 24}
        if profile_id == profiles.STEADY_FLOW_PROFILE
        else {"train": 96, "validation": 12, "id_test": 12}
    )
    _exact_keys(
        per_material,
        required=set(expected_counts),
        optional=set(),
        label="campaign.membership.per_seen_material",
    )
    normalized = {
        name: _integer(
            count,
            label=f"campaign.membership.per_seen_material.{name}",
            minimum=1,
        )
        for name, count in per_material.items()
    }
    if normalized != expected_counts:
        message = f"Seen-family membership must be exactly {expected_counts} per material."
        raise GenerationConfigError(message)
    membership["per_seen_material"] = normalized
    return membership


def _package_source_count(
    regime: str,
    source_role: str,
    *,
    material_roles: Mapping[str, tuple[str, ...]],
    counts: Mapping[str, Mapping[str, int]],
) -> int:
    """Derive one package source-case count without counting learning views."""
    materials_in_package = material_roles[source_role]
    sampling_regime = "parameter_ood" if regime == "parameter_ood" else "natural"
    return sum(counts[sampling_regime][family] for family in materials_in_package)


def _validate_dataset_packages(
    value: Any,
    *,
    profile: profiles.SimulationProfile,
    material_roles: Mapping[str, tuple[str, ...]],
    campaign_purpose: str,
    counts: Mapping[str, Mapping[str, int]],
    membership: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate concise declarations and derive immutable package semantics."""
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        if value != []:
            message = "Pilot-check campaigns prohibit normal dataset-package publication."
            raise GenerationConfigError(message)
        return ()
    if not isinstance(value, list) or not value:
        message = "campaign.dataset_packages must be a non-empty list."
        raise GenerationConfigError(message)
    expected = _expected_package_roles(material_roles, campaign_purpose=campaign_purpose)
    declarations: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        label = f"campaign.dataset_packages[{index}]"
        package = _mapping(raw, label=label)
        _exact_keys(
            package,
            required={"evaluation_regime", "source_role"},
            optional=set(),
            label=label,
        )
        pair = (package["evaluation_regime"], package["source_role"])
        if index >= len(expected) or pair != expected[index]:
            message = f"campaign.dataset_packages must declare regime/source-role pairs in order {list(expected)}."
            raise GenerationConfigError(message)
        source_role = str(package["source_role"])
        if not material_roles[source_role]:
            message = f"{label}.source_role {source_role!r} has no materials."
            raise GenerationConfigError(message)
        declarations.append(package)
    if len(declarations) != len(expected):
        message = f"campaign.dataset_packages must declare exactly {list(expected)}."
        raise GenerationConfigError(message)

    packages: list[dict[str, Any]] = []
    for dataset_view in profile.available_learning_views:
        for declaration in declarations:
            regime = str(declaration["evaluation_regime"])
            source_role = str(declaration["source_role"])
            package_materials = material_roles[source_role]
            family_campaign = campaign_purpose == "family_generalization"
            is_id = regime == "id" and family_campaign
            package = {
                **copy.deepcopy(declaration),
                "dataset_view": dataset_view,
                "materials": list(package_materials),
                "dataset_name": dataset_package_name(dataset_view, package_materials, regime),
                "source_case_count": _package_source_count(
                    regime,
                    source_role,
                    material_roles=material_roles,
                    counts=counts,
                ),
                "natural_support_only": regime != "parameter_ood",
                "split_eligibility": {
                    "train": is_id,
                    "validation": is_id,
                    "id_test": is_id,
                    "parameter_ood": regime == "parameter_ood" and family_campaign,
                },
            }
            if is_id:
                per_material = copy.deepcopy(dict(membership["per_seen_material"]))
                package["membership"] = {
                    "seed": membership["seed"],
                    "per_seen_material": per_material,
                    "totals": {name: count * len(package_materials) for name, count in per_material.items()},
                }
            packages.append(package)
    return tuple(packages)


def _material_memberships(
    material_roles: Mapping[str, tuple[str, ...]],
    *,
    campaign_purpose: str,
) -> dict[str, tuple[str, ...]]:
    """Derive typed material eligibility without free-text executable policy."""
    result: dict[str, tuple[str, ...]] = dict.fromkeys(SPLIT_NAMES, ())
    if campaign_purpose == "technical_runtime_smoke":
        result["technical_smoke"] = material_roles["seen"]
        return result
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        return result
    for name in _SEEN_SPLITS:
        result[name] = material_roles["seen"]
    for role in MATERIAL_ROLES[1:]:
        result[role] = material_roles[role]
    return result


def _build_assignments(
    material_family: str,
    sampling_regime: str,
    *,
    material_role: str,
    evaluation_regime: str,
    case_count: int,
    campaign_purpose: str,
    ood_allocation: tuple[Mapping[str, str], ...],
    pilot_case_semantics: Mapping[str, str] | None,
) -> dict[int, dict[str, Any]]:
    """Build deterministic material, pilot, evaluation, and OOD assignments."""
    if sampling_regime == "parameter_ood" and len(ood_allocation) != case_count:
        message = "Parameter-OOD assignment count disagrees with its eligible-unit allocation."
        raise GenerationConfigError(message)
    assignments: dict[int, dict[str, Any]] = {}
    for case_index in range(1, case_count + 1):
        regime_index = case_index - 1
        allocated = ood_allocation[regime_index] if sampling_regime == "parameter_ood" else None
        pilot_kind = None
        if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
            if pilot_case_semantics is None:
                message = "Pilot assignments require explicit configured case semantics."
                raise GenerationConfigError(message)
            pilot_kind = pilot_case_semantics["first"] if case_index == 1 else pilot_case_semantics["remaining"]
        ood_group = None if allocated is None else allocated["ood_group"]
        assignment_role = str(allocated["unit_id"]) if allocated is not None else pilot_kind or material_role
        assignment = {
            "case_index": case_index,
            "regime_index": regime_index,
            "material_family": material_family,
            "material_role": material_role,
            "evaluation_regime": evaluation_regime,
            "sampling_regime": sampling_regime,
            "assignment_role": assignment_role,
            "ood_group": ood_group,
            "ood_unit_id": None if allocated is None else allocated["unit_id"],
            "ood_units_per_case": 1 if allocated is not None else 0,
        }
        if pilot_kind is not None:
            assignment["pilot_case_kind"] = pilot_kind
        assignments[case_index] = assignment
    return assignments


def _safe_text_or_none(value: Any, *, label: str) -> str | None:
    """Return optional safe single-line execution text."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(character in value for character in ("\x00", "\n", "\r")):
        msg = f"{label} must be null or safe non-empty text."
        raise GenerationConfigError(msg)
    return value


def _validate_execution(value: Any, *, campaign_purpose: str) -> dict[str, Any]:
    """Validate authored execution owners and derive repeated runtime fields."""
    execution = _mapping(value, label="generation execution configuration")
    _exact_keys(
        execution,
        required={"schema_kind", "schema_version", "runtime", "retention", "cluster", "site"},
        optional=set(),
        label="generation execution configuration",
    )
    if execution["schema_kind"] != "generation_execution" or execution["schema_version"] != 1:
        msg = "Unsupported generation execution configuration schema."
        raise GenerationConfigError(msg)
    if campaign_purpose not in CAMPAIGN_PURPOSES:
        message = f"execution retention has no supported campaign purpose {campaign_purpose!r}."
        raise GenerationConfigError(message)

    site = _mapping(execution["site"], label="execution.site")
    site_keys = {
        "cpu_host",
        "scheduler",
        "partition",
        "cores_per_node",
        "python_module",
        "comsol_module",
        "python_executable",
        "comsol_executable",
    }
    _exact_keys(site, required=site_keys, optional=set(), label="execution.site")
    for key in site_keys.difference({"cores_per_node"}):
        site[key] = _safe_text_or_none(site[key], label=f"execution.site.{key}")
        if site[key] is None:
            msg = f"execution.site.{key} must be configured."
            raise GenerationConfigError(msg)
    if site["scheduler"] not in {"local", "slurm"}:
        msg = "execution.site.scheduler must be local or slurm."
        raise GenerationConfigError(msg)
    site["cores_per_node"] = _integer(site["cores_per_node"], label="execution.site.cores_per_node", minimum=1)

    runtime = _mapping(execution["runtime"], label="execution.runtime")
    _exact_keys(
        runtime,
        required={"timeout_seconds", "maximum_failures", "extra_arguments"},
        optional=set(),
        label="execution.runtime",
    )
    runtime["timeout_seconds"] = _finite(runtime["timeout_seconds"], label="execution.runtime.timeout_seconds")
    if runtime["timeout_seconds"] <= 0:
        msg = "execution.runtime.timeout_seconds must be positive."
        raise GenerationConfigError(msg)
    runtime["maximum_failures"] = _integer(
        runtime["maximum_failures"],
        label="execution.runtime.maximum_failures",
        minimum=1,
    )
    if runtime["maximum_failures"] != 1:
        msg = "The maintained generation runtime supports only fail-on-one maximum_failures=1."
        raise GenerationConfigError(msg)
    arguments = runtime["extra_arguments"]
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) and item and not any(char in item for char in ("\x00", "\n", "\r")) for item in arguments
    ):
        msg = "execution.runtime.extra_arguments must be an ordered list of safe arguments."
        raise GenerationConfigError(msg)
    if any(item == owned or item.startswith(f"{owned}=") for item in arguments for owned in _COMSOL_OWNED_ARGUMENTS):
        msg = "execution.runtime.extra_arguments cannot override case-owned files or one-node execution."
        raise GenerationConfigError(msg)
    runtime["executable"] = site["comsol_executable"]
    runtime["module_initialization"] = [
        f"module load {site['python_module']}",
        f"module load {site['comsol_module']}",
    ]

    retention_profiles = _mapping(execution["retention"], label="execution.retention")
    _exact_keys(retention_profiles, required=set(CAMPAIGN_PURPOSES), optional=set(), label="execution.retention")
    normalized_retention: dict[str, dict[str, bool]] = {}
    for purpose, raw in retention_profiles.items():
        retention = _mapping(raw, label=f"execution.retention.{purpose}")
        _exact_keys(retention, required={"retain_raw_csv", "retain_solved_model"}, optional=set(), label=f"execution.retention.{purpose}")
        if not all(isinstance(retention[key], bool) for key in retention):
            msg = f"execution.retention.{purpose} controls must be boolean."
            raise TypeError(msg)
        normalized_retention[purpose] = retention

    cluster = _mapping(execution["cluster"], label="execution.cluster")
    cluster_keys = {
        "max_nodes",
        "cases_per_node",
        "cores_per_case",
        "max_parallel_cases",
        "wall_time",
        "scheduler_options",
    }
    _exact_keys(cluster, required=cluster_keys, optional=set(), label="execution.cluster")
    for key in ("max_nodes", "cases_per_node", "cores_per_case", "max_parallel_cases"):
        cluster[key] = _integer(cluster[key], label=f"execution.cluster.{key}", minimum=1)
    cluster["wall_time"] = _safe_text_or_none(cluster["wall_time"], label="execution.cluster.wall_time")
    options = cluster["scheduler_options"]
    if not isinstance(options, list) or not all(isinstance(item, str) and item.startswith("--") for item in options):
        msg = "execution.cluster.scheduler_options must be long scheduler arguments."
        raise GenerationConfigError(msg)
    if any(option == owned or option.startswith(f"{owned}=") for option in options for owned in _SCHEDULER_OWNED_OPTIONS):
        msg = "execution.cluster.scheduler_options cannot override pipeline-owned allocation directives."
        raise GenerationConfigError(msg)
    cluster["cores_per_node"] = site["cores_per_node"]
    cluster["scheduler_kind"] = site["scheduler"]
    cluster["partition"] = site["partition"]
    if cluster["cases_per_node"] * cluster["cores_per_case"] > cluster["cores_per_node"]:
        msg = "cases_per_node * cores_per_case must not exceed site.cores_per_node."
        raise GenerationConfigError(msg)
    if cluster["max_parallel_cases"] > cluster["max_nodes"] * cluster["cases_per_node"]:
        msg = "max_parallel_cases must not exceed max_nodes * cases_per_node."
        raise GenerationConfigError(msg)

    execution["site"] = site
    execution["runtime"] = runtime
    execution["retention_profile"] = campaign_purpose
    execution["retention"] = normalized_retention[campaign_purpose]
    execution["retention_profiles"] = normalized_retention
    execution["cluster"] = cluster
    return execution


def _input_identity_config(scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return science that determines adapter inputs, excluding output behavior."""
    value = copy.deepcopy(dict(scientific))
    for key in (
        "simulation_profile",
        "output_contract",
        "available_learning_views",
        "airflow_source",
        "steady_flow_conditioning",
        "campaign_id",
        "campaign_purpose",
        "material_role",
        "evaluation_regime",
        "natural_support_state",
    ):
        value.pop(key, None)
    return value


def _campaign_name(value: Any) -> str:
    """Return one concise configured campaign name."""
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or value.startswith("_")
        or value.endswith("_")
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
    ):
        message = "campaign_name must use lowercase letters, digits, and single-word separators."
        raise GenerationConfigError(message)
    return value


def _campaign_references(campaign: dict[str, Any], *, source_path: Path) -> None:
    """Resolve each explicit layered campaign reference in place."""
    for key in (
        "sources_config",
        "registry_config",
        "common_config",
        "operations_config",
        "profile_config",
        "execution_config",
    ):
        campaign[key] = _reference_path(campaign[key], source_path=source_path, label=f"campaign.{key}")


def _validate_campaign_header(
    value: Any,
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Validate one concise campaign and derive its canonical material union."""
    campaign = _mapping(value, label="generation campaign configuration")
    required = {
        "schema_kind",
        "schema_version",
        "campaign_purpose",
        "sources_config",
        "registry_config",
        "common_config",
        "operations_config",
        "profile_config",
        "execution_config",
        "material_roles",
        "sampling",
        "dataset_packages",
    }
    optional = {"membership", "paired_equivalence_seed"}
    _exact_keys(campaign, required=required, optional=optional, label="generation campaign configuration")
    if campaign["schema_kind"] != "generation_campaign" or campaign["schema_version"] != 1:
        message = "Unsupported generation campaign schema."
        raise GenerationConfigError(message)
    purpose = campaign["campaign_purpose"]
    if purpose not in CAMPAIGN_PURPOSES:
        message = f"campaign.campaign_purpose must be one of {list(CAMPAIGN_PURPOSES)}."
        raise GenerationConfigError(message)
    if purpose == "family_generalization":
        if "membership" not in campaign or "paired_equivalence_seed" in campaign:
            message = "Family-generalization campaigns require membership and prohibit paired_equivalence_seed."
            raise GenerationConfigError(message)
    elif purpose == "technical_runtime_smoke":
        if "membership" in campaign or "paired_equivalence_seed" not in campaign:
            message = "Technical-smoke campaigns require paired_equivalence_seed and prohibit learning membership."
            raise GenerationConfigError(message)
    elif "membership" in campaign or "paired_equivalence_seed" in campaign:
        message = "Pilot-check campaigns prohibit learning membership and paired-equivalence ownership."
        raise GenerationConfigError(message)
    _campaign_references(campaign, source_path=source_path)

    raw_roles = _mapping(campaign["material_roles"], label="campaign.material_roles")
    _exact_keys(raw_roles, required=set(MATERIAL_ROLES), optional=set(), label="campaign.material_roles")
    material_roles = {
        role: _name_list(
            raw_roles[role],
            label=f"campaign.material_roles.{role}",
            allow_empty=role != "seen",
        )
        for role in MATERIAL_ROLES
    }
    if purpose in {"family_generalization", PILOT_CAMPAIGN_PURPOSE}:
        expected_roles = {
            "seen": ("lentil", "chickpea", "kidney_bean"),
            "near_family_ood": ("field_pea",),
            "far_family_ood": ("rapeseed",),
            "extreme_family_ood": ("sunflower_seed",),
        }
    else:
        expected_roles = {
            "seen": ("lentil",),
            "near_family_ood": (),
            "far_family_ood": (),
            "extreme_family_ood": (),
        }
    if material_roles != expected_roles:
        message = f"campaign.material_roles must match the canonical {purpose} family-role contract {expected_roles}."
        raise GenerationConfigError(message)
    campaign["materials"] = tuple(material_family for role in MATERIAL_ROLES for material_family in material_roles[role])
    campaign["material_roles"] = material_roles
    return campaign


def _build_batch(
    *,
    source_path: Path,
    profile: profiles.SimulationProfile,
    material: Mapping[str, Any],
    sampling_regime: str,
    case_count: int,
    campaign_seed: int,
    paired_equivalence_seed: int | None,
    sampling_method: str,
    parameter_ood: Mapping[str, Any],
    common_config: Mapping[str, Any],
    registry_metadata: Mapping[str, Mapping[str, str]],
    operations_digest: str,
    material_digest: str,
    execution: Mapping[str, Any],
    campaign_id: str,
    campaign_purpose: str,
    material_role: str,
    evaluation_regime: str,
    pilot_case_semantics: Mapping[str, str] | None,
) -> GenerationConfig:
    """Build one immutable profile-projected generation batch."""
    material_family = str(material["material_family"])
    batch_kind = PILOT_CAMPAIGN_PURPOSE if campaign_purpose == PILOT_CAMPAIGN_PURPOSE else sampling_regime
    batch_name = f"{profile.id}__{material_family}__{batch_kind}"
    batch_seed = derive_seed(
        campaign_seed,
        "generation_batch",
        profile.id,
        material_family,
        sampling_regime,
        evaluation_regime,
    )
    groups = materials.active_ood_groups(profile.id)
    from . import generation_sampling as sampling_service  # noqa: PLC0415 -- typed config/sampling cycle

    eligible_units = sampling_service.eligible_ood_units(material, groups=groups) if sampling_regime == "parameter_ood" else ()
    ood_allocation = sampling_service.allocate_ood_units(eligible_units, case_count=case_count) if sampling_regime == "parameter_ood" else ()
    resolved_parameter_ood = copy.deepcopy(dict(parameter_ood))
    resolved_parameter_ood["eligible_units"] = [copy.deepcopy(dict(unit)) for unit in eligible_units]
    resolved_parameter_ood["case_allocation"] = [
        {"case_index": index, **copy.deepcopy(dict(unit))} for index, unit in enumerate(ood_allocation, start=1)
    ]
    allocation_counts = dict.fromkeys((unit["unit_id"] for unit in eligible_units), 0)
    for unit in ood_allocation:
        allocation_counts[unit["unit_id"]] += 1
    resolved_parameter_ood["allocation_counts"] = allocation_counts
    assignments = _build_assignments(
        material_family,
        sampling_regime,
        material_role=material_role,
        evaluation_regime=evaluation_regime,
        case_count=case_count,
        campaign_purpose=campaign_purpose,
        ood_allocation=ood_allocation,
        pilot_case_semantics=pilot_case_semantics,
    )
    case_indices = tuple(assignments)
    registry = material["parameter_registry"]
    active_blocks = materials.active_sampling_blocks(profile.id)

    block_seed_bases = {"airflow": paired_equivalence_seed} if paired_equivalence_seed is not None else None
    sampling_plan = sampling_service.build_sampling_plan(
        registry=registry,
        case_count=case_count,
        seed_base=batch_seed,
        method=sampling_method,
        blocks=active_blocks,
        block_seed_bases=block_seed_bases,
    )
    block_dimensions = materials.sampling_block_dimensions(
        registry,
        blocks=active_blocks,
    )
    input_contract = copy.deepcopy(dict(common_config["input_contract"]))
    spatial_contract = input_contract["spatial"]
    spatial_contract["columns"] = spatial_contract.pop("columns_by_profile")[profile.id]
    if profile.id == profiles.STEADY_FLOW_PROFILE:
        input_contract.pop("scalar")
        input_contract.pop("schedule")
    output_contract = {
        "exports_root": "exports",
        "exports": common_config["_profile_exports"],
        "static_field_names": list(profiles.static_field_names(profile.id)),
        "static_field_units": list(profiles.static_field_units(profile.id)),
        "transient_field_names": (list(profiles.TRANSIENT_FIELD_NAMES) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "transient_field_units": (list(profiles.TRANSIENT_FIELD_UNITS) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "global_field_names": (list(profiles.GLOBAL_FIELD_NAMES) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "global_field_units": (list(profiles.GLOBAL_FIELD_UNITS) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
    }
    fixed_names = profiles.STATIONARY_FIXED_FIELDS if profile.id == profiles.STEADY_FLOW_PROFILE else tuple(common_config["scientific_fixed_values"])
    fixed_values = {name: copy.deepcopy(common_config["scientific_fixed_values"][name]) for name in fixed_names}
    fixed_records = {name: copy.deepcopy(common_config["_scientific_fixed_records"][name]) for name in fixed_names}
    active_registry_metadata = {name: copy.deepcopy(registry_metadata[name]) for name in registry}
    scientific: dict[str, Any] = {
        "schema_kind": "resolved_generation_batch",
        "schema_version": CONFIG_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "simulation_profile": profile.id,
        "campaign_id": campaign_id,
        "campaign_purpose": campaign_purpose,
        "campaign_seed": campaign_seed,
        "material_role": material_role,
        "evaluation_regime": evaluation_regime,
        "natural_support_state": ("natural" if sampling_regime == "natural" else "parameter_ood"),
        "material": copy.deepcopy(dict(material)),
        "material_config_digest": material_digest,
        "operation_config_digest": operations_digest,
        "sampling_regime": sampling_regime,
        "case_count": case_count,
        "sampling": {
            "method": sampling_method,
            "seed_base": batch_seed,
            "blocks": sampling_plan,
            "block_dimensions": block_dimensions,
        },
        "parameter_ood": resolved_parameter_ood,
        "assignments": [assignments[index] for index in case_indices],
        "registry_metadata": active_registry_metadata,
        "scientific_fixed_values": fixed_values,
        "scientific_fixed_records": fixed_records,
        "stationary_fixed_ownership": {
            name: {
                "owner": "package_fixed",
                "unit": unit,
                "fixed_value": float(common_config["scientific_fixed_values"][name]),
            }
            for name, unit in zip(
                profiles.STATIONARY_FIXED_FIELDS,
                profiles.STATIONARY_FIXED_UNITS,
                strict=True,
            )
        },
        "grid": copy.deepcopy(dict(common_config["grid"])),
        "grid_provenance": copy.deepcopy(dict(common_config["grid_provenance"])),
        "input_contract": input_contract,
        "output_contract": output_contract,
        "storage": copy.deepcopy(dict(common_config["storage"])),
        "available_learning_views": list(profile.available_learning_views),
        "airflow_source": profile.airflow_source,
    }
    if paired_equivalence_seed is not None:
        scientific["paired_equivalence_seed"] = paired_equivalence_seed
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        if pilot_case_semantics is None:
            message = "Pilot batch construction lost its configured case semantics."
            raise GenerationConfigError(message)
        scientific["pilot_check"] = {
            "cases_per_material": case_count,
            "case_kinds": list(PILOT_CASE_KINDS),
            "case_semantics": copy.deepcopy(dict(pilot_case_semantics)),
            "nominal_case_index": 1,
            "dataset_membership": "none",
            "sampling_semantics": "one_explicit_nominal_then_ordinary_natural_support",
        }
    if profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        scientific.update(
            {
                "physical_formulas": copy.deepcopy(dict(common_config["physical_formulas"])),
                "physical_formulas_provenance": copy.deepcopy(dict(common_config["physical_formulas_provenance"])),
                "time": copy.deepcopy(dict(common_config["time"])),
                "time_provenance": copy.deepcopy(dict(common_config["time_provenance"])),
                "operation_constraints": copy.deepcopy(dict(common_config["_operation_constraints"])),
            }
        )
    else:
        scientific["operation_constraints"] = {
            "porosity_natural_support_policy": common_config["_operation_constraints"]["porosity_natural_support_policy"]
        }
    conditioning = common_config["_steady_flow_conditioning"]
    if conditioning is not None:
        scientific["steady_flow_conditioning"] = copy.deepcopy(conditioning)

    scientific_digest = common.serialization.canonical_json_sha256(scientific)
    case_input_digest = common.serialization.canonical_json_sha256(_input_identity_config(scientific))
    batch_id = f"{batch_name}__{scientific_digest[:16]}"
    return GenerationConfig(
        source_path=source_path,
        profile=profile,
        material_family=material_family,
        material_role=material_role,
        evaluation_regime=evaluation_regime,
        sampling_regime=sampling_regime,
        batch_name=batch_name,
        scientific_values=scientific,
        execution_values=copy.deepcopy(dict(execution)),
        template_path=profile.template_path,
        template_sha256=profile.template_sha256,
        case_indices=case_indices,
        seed_base=batch_seed,
        assignments=assignments,
        scientific_config_digest=scientific_digest,
        case_input_config_digest=case_input_digest,
        batch_identity=scientific_digest,
        batch_id=batch_id,
    )


def load_campaign_config(  # noqa: PLR0912, PLR0915 -- one centralized campaign resolver
    path: Path | str,
    *,
    require_executable: bool = True,
    pilot_cases_per_material: int | None = None,
) -> CampaignConfig:
    """Resolve one campaign, every subbatch, and every immutable dataset plan."""
    source_path = Path(path).expanduser().resolve()
    campaign = _validate_campaign_header(
        _load_yaml(source_path, label="generation campaign configuration"),
        source_path=source_path,
    )
    source_config = _load_yaml(
        campaign["sources_config"],
        label="generation scientific source registry",
    )
    sources = provenance_service.validate_source_registry(
        source_config,
        decision_validator=lambda value: materials.validate_decision_source(
            value,
            label="generation source registry decision_source",
        ),
    )
    registry_config = _load_yaml(
        campaign["registry_config"],
        label="generation parameter registry",
    )
    definitions, registry_metadata = materials.validate_semantic_registry(registry_config)
    common_config = _validate_common(
        _load_yaml(
            campaign["common_config"],
            label="generation common configuration",
        ),
        sources=sources,
    )
    operations = _validate_operations(
        _load_yaml(
            campaign["operations_config"],
            label="generation operations configuration",
        ),
        sources=sources,
    )
    profile_raw = _load_yaml(
        campaign["profile_config"],
        label="generation profile configuration",
    )
    profile_id = profile_raw.get("simulation_profile")
    if not isinstance(profile_id, str):
        message = "generation profile simulation_profile must be text."
        raise TypeError(message)
    profile = profiles.get_profile(profile_id)
    profile_config = _validate_profile_config(
        profile_raw,
        profile=profile,
        fixed_values=common_config["scientific_fixed_values"],
        require_executable=require_executable,
    )
    campaign_purpose = str(campaign["campaign_purpose"])
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE and profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Pilot-check campaigns require the transient_drying profile."
        raise GenerationConfigError(message)
    if pilot_cases_per_material is not None and campaign_purpose != PILOT_CAMPAIGN_PURPOSE:
        message = "pilot_cases_per_material applies only to a dedicated pilot-check campaign."
        raise GenerationConfigError(message)
    execution = _validate_execution(
        _load_yaml(
            campaign["execution_config"],
            label="generation execution configuration",
        ),
        campaign_purpose=campaign_purpose,
    )
    campaign_name = _campaign_name(f"{profile.id}_{campaign_purpose}")
    campaign_schema_version = campaign["schema_version"]

    paired_equivalence_seed: int | None = None
    if campaign_purpose == "technical_runtime_smoke":
        paired_equivalence_seed = _uint32(
            campaign["paired_equivalence_seed"],
            label="campaign.paired_equivalence_seed",
        )
        if paired_equivalence_seed != PAIRED_EQUIVALENCE_SEED:
            message = "Both technical-smoke campaigns must use paired_equivalence_seed 9930."
            raise GenerationConfigError(message)

    resolved_materials: dict[str, dict[str, Any]] = {}
    material_digests: dict[str, str] = {}
    project_root = common.paths.get_project_root()
    for material_family in campaign["materials"]:
        material_path = project_root / "configs" / "generation" / "materials" / f"{material_family}.yaml"
        material_raw = _load_yaml(
            material_path,
            label=f"material {material_family} configuration",
        )
        if material_raw.get("material_family") != material_family:
            message = f"Material filename and material_family disagree for {material_path}."
            raise GenerationConfigError(message)
        full_material = materials.resolve_material_definition(
            definitions,
            registry_metadata,
            common_config["parameter_values"],
            operations["parameter_values"],
            material_raw,
            sources=sources,
        )
        projected = materials.project_material_for_profile(
            full_material,
            profile.id,
        )
        resolved_materials[material_family] = projected
        material_digests[material_family] = common.serialization.canonical_json_sha256(projected)

    sampling = _mapping(campaign["sampling"], label="campaign.sampling")
    pilot_case_semantics: dict[str, str] | None = None
    configured_pilot_cases_per_material: int | None = None
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        _exact_keys(
            sampling,
            required={"method", "seed_base", "cases_per_material", "case_semantics"},
            optional=set(),
            label="campaign.sampling",
        )
        configured_pilot_cases_per_material = _integer(
            sampling["cases_per_material"],
            label="campaign.sampling.cases_per_material",
            minimum=1,
        )
        selected_count = (
            configured_pilot_cases_per_material
            if pilot_cases_per_material is None
            else _integer(
                pilot_cases_per_material,
                label="pilot_cases_per_material",
                minimum=1,
            )
        )
        pilot_case_semantics = _mapping(
            sampling["case_semantics"],
            label="campaign.sampling.case_semantics",
        )
        _exact_keys(
            pilot_case_semantics,
            required={"first", "remaining"},
            optional=set(),
            label="campaign.sampling.case_semantics",
        )
        if pilot_case_semantics != {
            "first": "nominal_reference",
            "remaining": "natural_pilot",
        }:
            message = "Pilot case semantics must be one nominal_reference followed by natural_pilot cases."
            raise GenerationConfigError(message)
        raw_counts = {
            "natural": dict.fromkeys(campaign["materials"], selected_count),
        }
    else:
        _exact_keys(
            sampling,
            required={"method", "seed_base", "counts"},
            optional=set(),
            label="campaign.sampling",
        )
        raw_counts = _mapping(
            sampling["counts"],
            label="campaign.sampling.counts",
        )
    if sampling["method"] not in {"lhs", "sobol"}:
        message = "campaign.sampling.method must be lhs or sobol."
        raise GenerationConfigError(message)
    sampling_regimes = tuple(raw_counts)
    expected_sampling_regimes = ("natural", "parameter_ood") if campaign_purpose == "family_generalization" else ("natural",)
    if sampling_regimes != expected_sampling_regimes:
        message = f"campaign.sampling must resolve regimes in order {list(expected_sampling_regimes)}."
        raise GenerationConfigError(message)
    campaign_seed = _uint32(
        sampling["seed_base"],
        label="campaign.sampling.seed_base",
    )
    material_roles = campaign["material_roles"]
    counts = _validate_batch_counts(
        raw_counts,
        materials_in_order=campaign["materials"],
        seen=material_roles["seen"],
        regimes=sampling_regimes,
    )
    _validate_binding_sampling_contract(
        campaign_purpose=campaign_purpose,
        profile_id=profile.id,
        campaign_seed=campaign_seed,
        counts=counts,
    )
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        cases_per_material = next(iter(counts["natural"].values()))
        campaign_id = _campaign_name(f"{campaign_name}_n{cases_per_material}_v{campaign_schema_version}")
    else:
        cases_per_material = None
        campaign_id = _campaign_name(f"{campaign_name}_v{campaign_schema_version}")

    membership = _validate_membership(
        campaign.get("membership"),
        campaign_purpose=campaign_purpose,
        profile_id=profile.id,
    )
    if campaign_purpose == "family_generalization":
        ood_groups = materials.active_ood_groups(profile.id)
        parameter_ood = _validate_parameter_ood_policy(
            {
                "groups": list(ood_groups),
                "units_per_case": 1,
                "allocation_strategy": "canonical_eligible_unit_round_robin",
                "eligibility_source": "resolved_profile_projected_registry",
            },
            groups=ood_groups,
        )
    else:
        parameter_ood = {
            "groups": [],
            "units_per_case": 0,
            "allocation_strategy": "not_applicable",
            "eligibility_source": "not_applicable",
        }
    dataset_packages = _validate_dataset_packages(
        campaign["dataset_packages"],
        profile=profile,
        material_roles=material_roles,
        campaign_purpose=campaign_purpose,
        counts=counts,
        membership=membership,
    )
    evaluation_regimes = tuple(dict.fromkeys(str(package["evaluation_regime"]) for package in dataset_packages))
    expected_evaluation_regimes = (
        EVALUATION_REGIMES if campaign_purpose == "family_generalization" else ("id",) if campaign_purpose == "technical_runtime_smoke" else ()
    )
    if evaluation_regimes != expected_evaluation_regimes:
        message = f"Resolved evaluation regimes must be exactly {list(expected_evaluation_regimes)}."
        raise GenerationConfigError(message)
    material_memberships = _material_memberships(
        material_roles,
        campaign_purpose=campaign_purpose,
    )
    role_by_material = {material_family: role for role, role_materials in material_roles.items() for material_family in role_materials}

    common_with_profile = copy.deepcopy(common_config)
    common_with_profile["_profile_exports"] = profile_config["exports"]
    common_with_profile["_steady_flow_conditioning"] = profile_config["steady_flow_conditioning"]
    common_with_profile["_operation_constraints"] = operations["constraints"]
    active_names = {name for block in materials.active_sampling_blocks(profile.id) for name in materials.SAMPLING_BLOCKS[block]}
    active_operation = {name: operations["parameter_values"][name] for name in operations["parameter_values"] if name in active_names}
    active_constraints = (
        operations["constraints"]
        if profile.id == profiles.TRANSIENT_DRYING_PROFILE
        else {"porosity_natural_support_policy": operations["constraints"]["porosity_natural_support_policy"]}
    )
    operations_digest = common.serialization.canonical_json_sha256(
        {
            "operation_id": operations["operation_id"],
            "parameter_values": active_operation,
            "constraints": active_constraints,
        }
    )

    batches: list[GenerationConfig] = []
    for sampling_regime in sampling_regimes:
        for material_family, count in counts[sampling_regime].items():
            material_role = role_by_material[material_family]
            evaluation_regime = (
                NO_EVALUATION_REGIME
                if campaign_purpose == PILOT_CAMPAIGN_PURPOSE
                else "parameter_ood"
                if sampling_regime == "parameter_ood"
                else "id"
                if material_role == "seen"
                else material_role
            )
            batches.append(
                _build_batch(
                    source_path=source_path,
                    profile=profile,
                    material=resolved_materials[material_family],
                    sampling_regime=sampling_regime,
                    case_count=count,
                    campaign_seed=campaign_seed,
                    paired_equivalence_seed=paired_equivalence_seed,
                    sampling_method=str(sampling["method"]),
                    parameter_ood=parameter_ood,
                    common_config=common_with_profile,
                    registry_metadata=registry_metadata,
                    operations_digest=operations_digest,
                    material_digest=material_digests[material_family],
                    execution=execution,
                    campaign_id=campaign_id,
                    campaign_purpose=campaign_purpose,
                    material_role=material_role,
                    evaluation_regime=evaluation_regime,
                    pilot_case_semantics=pilot_case_semantics,
                )
            )
    batch_names = [batch.batch_name for batch in batches]
    if len(batch_names) != len(set(batch_names)):
        message = "Campaign planned duplicate batch names."
        raise GenerationConfigError(message)
    total_case_count = sum(len(batch.case_indices) for batch in batches)
    expected_total = (
        2
        if campaign_purpose == "technical_runtime_smoke"
        else len(materials.MATERIAL_FAMILIES) * int(cases_per_material)
        if campaign_purpose == PILOT_CAMPAIGN_PURPOSE and cases_per_material is not None
        else PRODUCTION_CASE_COUNTS[profile.id]
    )
    if total_case_count != expected_total:
        message = f"Resolved campaign has {total_case_count} cases; expected {expected_total}."
        raise GenerationConfigError(message)

    duplicate_case_input_policy = "reject_duplicates"
    campaign_scientific = {
        "schema_kind": "resolved_generation_campaign",
        "schema_version": 1,
        "campaign_name": campaign_name,
        "campaign_id": campaign_id,
        "campaign_purpose": campaign_purpose,
        "simulation_profile": profile.id,
        "materials": list(campaign["materials"]),
        "material_roles": {name: list(values) for name, values in material_roles.items()},
        "evaluation_regimes": list(evaluation_regimes),
        "material_memberships": {name: list(values) for name, values in material_memberships.items()},
        "membership": copy.deepcopy(membership),
        "sampling_method": sampling["method"],
        "sampling_regimes": list(sampling_regimes),
        "parameter_ood_policy": parameter_ood,
        "parameter_ood_allocations": {
            batch.batch_name: copy.deepcopy(batch.scientific_values["parameter_ood"]) for batch in batches if batch.sampling_regime == "parameter_ood"
        },
        "campaign_seed": campaign_seed,
        "paired_equivalence_seed": paired_equivalence_seed,
        "total_case_count": total_case_count,
        "batch_ids": [batch.batch_id for batch in batches],
        "dataset_packages": list(dataset_packages),
        "duplicate_case_input_policy": duplicate_case_input_policy,
        "sources_config_digest": common.serialization.canonical_json_sha256(source_config),
        "registry_config_digest": common.serialization.canonical_json_sha256(registry_config),
        "common_config_digest": common.serialization.canonical_json_sha256(common_config),
        "operation_config_digest": operations_digest,
        "material_config_digests": material_digests,
    }
    if campaign_purpose == PILOT_CAMPAIGN_PURPOSE:
        campaign_scientific["pilot_plan"] = {
            "configured_default_cases_per_material": configured_pilot_cases_per_material,
            "cases_per_material": cases_per_material,
            "total_case_count": total_case_count,
            "case_semantics": copy.deepcopy(pilot_case_semantics),
            "dataset_membership": "none",
            "evaluation_regime": NO_EVALUATION_REGIME,
        }
    campaign_digest = common.serialization.canonical_json_sha256(campaign_scientific)
    return CampaignConfig(
        source_path=source_path,
        campaign_name=campaign_name,
        campaign_digest=campaign_digest,
        campaign_id=campaign_id,
        campaign_purpose=campaign_purpose,
        evaluation_regimes=evaluation_regimes,
        profile=profile,
        material_roles=copy.deepcopy(material_roles),
        material_memberships=material_memberships,
        membership=copy.deepcopy(membership),
        source_registry=copy.deepcopy(sources),
        total_case_count=total_case_count,
        paired_equivalence_seed=paired_equivalence_seed,
        batches=tuple(batches),
        dataset_packages=dataset_packages,
        duplicate_case_input_policy=duplicate_case_input_policy,
        execution_values=execution,
    )


def load_generation_config(
    path: Path | str,
    *,
    require_executable: bool = True,
    only_batch: str | None = None,
) -> GenerationConfig:
    """Resolve one predeclared batch from a campaign configuration."""
    campaign = load_campaign_config(path, require_executable=require_executable)
    if only_batch is not None:
        return campaign.batch(only_batch)
    if len(campaign.batches) != 1:
        message = f"Campaign {campaign.campaign_name!r} declares {len(campaign.batches)} batches; select one with only_batch."
        raise GenerationConfigError(message)
    return campaign.batches[0]
