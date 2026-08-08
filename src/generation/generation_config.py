"""
===============================================================================
generation_config.py
===============================================================================
Resolve and validate layered scientific and execution generation configurations.
Responsibilities:
  - Resolve common, material-family, profile, dataset, and execution YAML owners
  - Validate authoritative grid, time, adapter, storage, split, and runtime contracts
  - Derive deterministic scientific identities and profile-neutral input identity
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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src import common

from . import generation_materials as materials
from . import generation_profiles as profiles

CONFIG_SCHEMA_VERSION = 2
GENERATOR_VERSION = "python_multiscale_v2"
CASE_ID_WIDTH = 4
UINT32_MAX = 2**32 - 1
_EXPECTED_T_IN_MAX = 308.15
_EXPECTED_F_WET_DM_MAX = 0.05
_EXPECTED_HDF5_COMPRESSION_LEVEL = 4
_MAXIMUM_TIME_CHUNK = 2
SPLIT_NAMES = (
    "train",
    "validation",
    "id_test",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
)
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
        return derive_seed(self.seed_base, "case", str(case_index))

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
    profile: profiles.SimulationProfile
    roles: dict[str, tuple[str, ...]]
    batches: tuple[GenerationConfig, ...]
    dataset_packages: tuple[dict[str, Any], ...]
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
        """Return a campaign execution view over predeclared batches only."""
        if batch_names is None:
            return self
        if not batch_names or len(batch_names) != len(set(batch_names)):
            message = "Campaign batch selection must be non-empty and duplicate-free."
            raise ValueError(message)
        return replace(self, batches=tuple(self.batch(name) for name in batch_names))

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
    """Validate the authoritative 401-by-251 boundary-inclusive grid."""
    grid = _mapping(value, label="common.grid")
    expected = {"nx", "ny", "Lx", "Ly", "dx", "dy", "boundaries_included"}
    _exact_keys(grid, required=expected, optional=set(), label="common.grid")
    required = {
        "nx": 401,
        "ny": 251,
        "Lx": 1.2,
        "Ly": 0.75,
        "dx": 0.003,
        "dy": 0.003,
        "boundaries_included": True,
    }
    normalized = {
        "nx": _integer(grid["nx"], label="common.grid.nx", minimum=2),
        "ny": _integer(grid["ny"], label="common.grid.ny", minimum=2),
        "Lx": _finite(grid["Lx"], label="common.grid.Lx"),
        "Ly": _finite(grid["Ly"], label="common.grid.Ly"),
        "dx": _finite(grid["dx"], label="common.grid.dx"),
        "dy": _finite(grid["dy"], label="common.grid.dy"),
        "boundaries_included": grid["boundaries_included"],
    }
    if normalized != required:
        message = f"Grid must be the authoritative boundary-inclusive 401x251 contract: {required}."
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


def _validate_scientific_fixed(value: Any, *, allow_unresolved: bool) -> dict[str, Any]:
    """Validate fixed thermodynamic, humidity, and stopping values."""
    fixed = _mapping(value, label="common.scientific_fixed_values")
    expected = {
        "p_ref",
        "T_in_max",
        "omega_min",
        "omega_max",
        "phi_clip_min",
        "phi_clip_max",
        "f_wet_dm_max",
        "schedule_interpolation",
    }
    _exact_keys(fixed, required=expected, optional=set(), label="common.scientific_fixed_values")
    unresolved_allowed = {"p_ref", "omega_min", "omega_max", "phi_clip_min", "phi_clip_max"}
    for key in expected.difference({"schedule_interpolation"}):
        if fixed[key] is None and allow_unresolved and key in unresolved_allowed:
            continue
        fixed[key] = _finite(fixed[key], label=f"common.scientific_fixed_values.{key}")
    if fixed["p_ref"] is not None and fixed["p_ref"] <= 0:
        msg = "common.scientific_fixed_values.p_ref must be positive."
        raise GenerationConfigError(msg)
    if fixed["T_in_max"] != _EXPECTED_T_IN_MAX or fixed["f_wet_dm_max"] != _EXPECTED_F_WET_DM_MAX:
        msg = "Temperature maximum and dry-mass-weighted stop limit must be 308.15 K and 0.05."
        raise GenerationConfigError(msg)
    humidity = tuple(fixed[name] for name in ("omega_min", "omega_max", "phi_clip_min", "phi_clip_max"))
    if None not in humidity and not (0 <= humidity[0] < humidity[1] and 0 <= humidity[2] < humidity[3] <= 1):
        msg = "Configured humidity bounds are invalid."
        raise GenerationConfigError(msg)
    if fixed["schedule_interpolation"] != "linear":
        msg = "The maintained schedule interpolation must be linear between hourly nodes."
        raise GenerationConfigError(msg)
    return fixed


def _validate_input_contract(value: Any) -> dict[str, Any]:
    """Validate canonical spatial, scalar, and schedule adapters."""
    contract = _mapping(value, label="common.input_contract")
    _exact_keys(contract, required={"spatial", "scalar", "schedule"}, optional=set(), label="common.input_contract")
    expected_columns = {
        "spatial": list(profiles.SPATIAL_INPUT_FIELDS),
        "scalar": ["name", "value", "unit"],
        "schedule": list(profiles.SCHEDULE_FIELDS),
    }
    expected_filenames = {"spatial": "fields.csv", "scalar": "scalars.csv", "schedule": "schedule.csv"}
    filenames: set[str] = set()
    normalized: dict[str, Any] = {}
    for name in ("spatial", "scalar", "schedule"):
        adapter = _mapping(contract[name], label=f"common.input_contract.{name}")
        _exact_keys(adapter, required={"filename", "delimiter", "columns"}, optional=set(), label=f"common.input_contract.{name}")
        filename = validate_relative_file(adapter["filename"], label=f"common.input_contract.{name}.filename", suffix=".csv")
        columns = adapter["columns"]
        if filename != expected_filenames[name] or columns != expected_columns[name]:
            message = f"{name} adapter must use {expected_filenames[name]!r} with columns {expected_columns[name]}."
            raise GenerationConfigError(message)
        if filename in filenames:
            msg = "Input adapter filenames must be unique."
            raise GenerationConfigError(msg)
        filenames.add(filename)
        normalized[name] = {
            "filename": filename,
            "delimiter": _delimiter(adapter["delimiter"], label=f"common.input_contract.{name}.delimiter"),
            "columns": list(columns),
        }
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
    if not isinstance(storage["converter_version"], str) or not storage["converter_version"]:
        msg = "common.storage.converter_version must be non-empty text."
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


def _validate_common(value: Any, *, require_executable: bool) -> dict[str, Any]:
    """Validate the global material-independent scientific owner."""
    common_config = _mapping(value, label="generation common configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "generator_version",
        "executable",
        "parameter_values",
        "scientific_fixed_values",
        "physical_formulas",
        "grid",
        "time",
        "input_contract",
        "storage",
    }
    _exact_keys(common_config, required=expected, optional=set(), label="generation common configuration")
    if common_config["schema_kind"] != "generation_common" or common_config["schema_version"] != 1:
        msg = "Unsupported generation common configuration schema."
        raise GenerationConfigError(msg)
    if common_config["generator_version"] != GENERATOR_VERSION:
        msg = f"generator_version must be {GENERATOR_VERSION!r}."
        raise GenerationConfigError(msg)
    if not isinstance(common_config["executable"], bool):
        msg = "generation common executable must be boolean."
        raise TypeError(msg)
    if require_executable and not common_config["executable"]:
        msg = "The common scientific template is non-executable because required values remain unresolved."
        raise GenerationConfigError(msg)
    common_config["parameter_values"] = _mapping(common_config["parameter_values"], label="common.parameter_values")
    formulas = _mapping(common_config["physical_formulas"], label="common.physical_formulas")
    required_formulas = {
        "w_gr",
        "X_db",
        "X_wb",
        "X_wb_from_X_db",
        "X_db_from_X_wb",
        "X_wb_bulk",
        "rho_bu_dry",
        "w_gr_0",
        "cp_gr_eff",
        "f_wet_dm",
    }
    _exact_keys(formulas, required=required_formulas, optional=set(), label="common.physical_formulas")
    if any(not isinstance(formula, str) or not formula for formula in formulas.values()):
        msg = "Every common physical formula must be explicit non-empty text."
        raise GenerationConfigError(msg)
    common_config["physical_formulas"] = formulas
    common_config["scientific_fixed_values"] = _validate_scientific_fixed(
        common_config["scientific_fixed_values"],
        allow_unresolved=not require_executable,
    )
    common_config["grid"] = _validate_grid(common_config["grid"])
    common_config["time"] = _validate_time(common_config["time"])
    common_config["input_contract"] = _validate_input_contract(common_config["input_contract"])
    common_config["storage"] = _validate_storage(common_config["storage"])
    return common_config


def _validate_profile_config(value: Any, *, profile: profiles.SimulationProfile, require_executable: bool) -> dict[str, Any]:
    """Validate explicit profile mappings without inferring binary template internals."""
    config = _mapping(value, label="generation profile configuration")
    expected = {"schema_kind", "schema_version", "simulation_profile", "template_ready", "exports"}
    _exact_keys(config, required=expected, optional=set(), label="generation profile configuration")
    if config["schema_kind"] != "generation_profile" or config["schema_version"] != 1 or config["simulation_profile"] != profile.id:
        msg = "Generation profile configuration schema or profile identity is invalid."
        raise GenerationConfigError(msg)
    if not isinstance(config["template_ready"], bool):
        msg = "generation profile template_ready must be boolean."
        raise TypeError(msg)
    if require_executable and not config["template_ready"]:
        message = f"Profile {profile.id!r} is fail-closed until its COMSOL template mappings are manually updated and confirmed."
        raise GenerationConfigError(message)
    exports = config["exports"]
    if not isinstance(exports, list):
        msg = "generation profile exports must be an ordered list."
        raise TypeError(msg)
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(exports):
        label = f"generation profile exports[{index}]"
        export = _mapping(raw, label=label)
        _exact_keys(export, required={"role", "pattern", "delimiter", "columns", "time_column"}, optional=set(), label=label)
        role = export["role"]
        if not isinstance(role, str) or role in seen:
            msg = f"{label}.role must be one unique profile role."
            raise GenerationConfigError(msg)
        role_spec = profile.export_role(role)
        seen.add(role)
        pattern = export["pattern"]
        if pattern is None and not require_executable:
            normalized_pattern = None
        elif not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts or "**" in pattern:
            msg = f"{label}.pattern must be one narrow relative pattern."
            raise GenerationConfigError(msg)
        else:
            normalized_pattern = pattern
        columns = _mapping(export["columns"], label=f"{label}.columns")
        if tuple(columns) != role_spec.logical_fields:
            message = f"{label}.columns must map exact logical fields {list(role_spec.logical_fields)} in order."
            raise GenerationConfigError(message)
        normalized_columns: dict[str, str | None] = {}
        for logical, source in columns.items():
            if source is None and not require_executable:
                normalized_columns[logical] = None
            elif not isinstance(source, str) or not source:
                msg = f"{label}.columns.{logical} must be an explicit export header."
                raise GenerationConfigError(msg)
            else:
                normalized_columns[logical] = source
        configured_sources = [source for source in normalized_columns.values() if source is not None]
        if len(configured_sources) != len(set(configured_sources)):
            msg = f"{label}.columns must map each logical field to a distinct source header."
            raise GenerationConfigError(msg)
        time_column = export["time_column"]
        if time_column is not None and (not isinstance(time_column, str) or not time_column):
            msg = f"{label}.time_column must be null or a non-empty header."
            raise GenerationConfigError(msg)
        if role != profiles.STEADY_FLOW_EXPORT_ROLE and time_column is not None:
            msg = f"{label}.time_column applies only to repeated stationary airflow exports."
            raise GenerationConfigError(msg)
        if time_column is not None and time_column in configured_sources:
            msg = f"{label}.time_column must not duplicate a mapped scientific field header."
            raise GenerationConfigError(msg)
        validated.append(
            {
                "role": role,
                "pattern": normalized_pattern,
                "delimiter": _delimiter(export["delimiter"], label=f"{label}.delimiter"),
                "columns": normalized_columns,
                "time_column": time_column,
                "required": role_spec.required,
                "allow_multiple": role_spec.allow_multiple,
                "units": dict(zip(role_spec.logical_fields, role_spec.units, strict=True)),
            }
        )
    missing = sorted(set(profile.required_export_roles).difference(seen))
    extra = sorted(seen.difference(spec.role for spec in profile.export_roles))
    if missing or extra:
        msg = f"Profile {profile.id!r} export roles are incomplete: missing={missing}, extra={extra}."
        raise GenerationConfigError(msg)
    config["exports"] = validated
    return config


def _validate_operations(value: Any, *, require_executable: bool) -> dict[str, Any]:
    """Validate material-independent operating-distribution ownership."""
    operations = _mapping(value, label="generation operations configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "operation_id",
        "executable",
        "parameter_values",
    }
    _exact_keys(operations, required=expected, optional=set(), label="generation operations configuration")
    if operations["schema_kind"] != "generation_operations" or operations["schema_version"] != 1:
        msg = "Unsupported generation operations schema."
        raise GenerationConfigError(msg)
    if operations["operation_id"] != "fixed_bed":
        msg = "The maintained operation configuration must use operation_id 'fixed_bed'."
        raise GenerationConfigError(msg)
    if not isinstance(operations["executable"], bool):
        msg = "generation operations executable must be boolean."
        raise TypeError(msg)
    if require_executable and not operations["executable"]:
        msg = "Operating distributions remain unresolved and non-executable."
        raise GenerationConfigError(msg)
    operations["parameter_values"] = _mapping(operations["parameter_values"], label="operations.parameter_values")
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


def _optional_uint32(value: Any, *, label: str, allow_unresolved: bool) -> int | None:
    """Return one uint32 seed or an allowed unresolved marker."""
    if value is None and allow_unresolved:
        return None
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
    allow_unresolved: bool,
) -> dict[str, dict[str, int | None]]:
    """Validate one explicit count for every planned material/regime batch."""
    counts = _mapping(value, label="campaign.sampling.counts")
    _exact_keys(counts, required={"natural", "parameter_ood"}, optional=set(), label="campaign.sampling.counts")
    expected = {"natural": materials_in_order, "parameter_ood": seen}
    normalized: dict[str, dict[str, int | None]] = {}
    for regime, expected_materials in expected.items():
        raw = _mapping(counts[regime], label=f"campaign.sampling.counts.{regime}")
        if tuple(raw) != expected_materials:
            message = f"campaign.sampling.counts.{regime} must follow exact material order {list(expected_materials)}."
            raise GenerationConfigError(message)
        normalized[regime] = {}
        for material_family, count in raw.items():
            if count is None and allow_unresolved:
                normalized[regime][material_family] = None
            else:
                normalized[regime][material_family] = _integer(
                    count,
                    label=f"campaign.sampling.counts.{regime}.{material_family}",
                    minimum=1,
                )
    return normalized


def _validate_parameter_ood_policy(
    value: Any,
    *,
    counts: Mapping[str, int | None],
    allow_unresolved: bool,
) -> dict[str, Any]:
    """Validate balanced parameter-OOD-group activation within each material batch."""
    policy = _mapping(value, label="campaign.sampling.parameter_ood")
    expected = {"groups", "units_per_case", "balance_groups", "balance_parameters"}
    _exact_keys(policy, required=expected, optional=set(), label="campaign.sampling.parameter_ood")
    if policy["groups"] != list(materials.OOD_GROUPS):
        message = f"Parameter-OOD groups must be exactly {list(materials.OOD_GROUPS)}."
        raise GenerationConfigError(message)
    policy["units_per_case"] = _integer(
        policy["units_per_case"],
        label="campaign.sampling.parameter_ood.units_per_case",
        minimum=1,
    )
    if not isinstance(policy["balance_groups"], bool) or not isinstance(policy["balance_parameters"], bool):
        message = "Parameter-OOD balance controls must be boolean."
        raise TypeError(message)
    if policy["balance_groups"]:
        for material_family, count in counts.items():
            if count is not None and count % len(materials.OOD_GROUPS) != 0:
                message = f"Balanced parameter-OOD count for {material_family!r} must be divisible by {len(materials.OOD_GROUPS)}."
                raise GenerationConfigError(message)
            if count is None and not allow_unresolved:
                message = f"Parameter-OOD count for {material_family!r} is unresolved."
                raise GenerationConfigError(message)
    return policy


def dataset_package_name(learning_task: str, material_families: tuple[str, ...], evaluation_regime: str) -> str:
    """Return the canonical human-readable dataset package name."""
    if (
        not learning_task
        or not material_families
        or evaluation_regime
        not in {
            "id",
            "parameter_ood",
            "near_family_ood",
            "far_family_ood",
        }
    ):
        message = "Dataset package naming inputs are invalid."
        raise GenerationConfigError(message)
    return f"{learning_task}__{'+'.join(material_families)}__{evaluation_regime}"


def _validate_dataset_packages(
    value: Any,
    *,
    profile: profiles.SimulationProfile,
    roles: Mapping[str, tuple[str, ...]],
    allow_unresolved: bool,
) -> tuple[dict[str, Any], ...]:
    """Validate independent campaign-owned ID and OOD package plans."""
    if not isinstance(value, list) or not value:
        message = "campaign.dataset_packages must be a non-empty list."
        raise GenerationConfigError(message)
    packages: list[dict[str, Any]] = []
    seen_regimes: set[str] = set()
    expected_materials = {
        "id": roles["seen"],
        "parameter_ood": roles["seen"],
        "near_family_ood": roles["near_family_ood"],
        "far_family_ood": roles["far_family_ood"],
    }
    for index, raw in enumerate(value):
        package = _mapping(raw, label=f"campaign.dataset_packages[{index}]")
        regime = package.get("evaluation_regime")
        required = {"learning_task", "evaluation_regime", "materials"}
        if regime == "id":
            required |= {"membership_seed", "membership_counts_per_material"}
        _exact_keys(package, required=required, optional=set(), label=f"campaign.dataset_packages[{index}]")
        if regime in seen_regimes or regime not in expected_materials:
            message = f"Dataset evaluation regime {regime!r} is invalid or duplicated."
            raise GenerationConfigError(message)
        seen_regimes.add(str(regime))
        learning_task = package["learning_task"]
        if learning_task not in profile.available_learning_views:
            message = f"Learning view {learning_task!r} is unavailable from profile {profile.id!r}."
            raise GenerationConfigError(message)
        package_materials = _name_list(package["materials"], label=f"campaign.dataset_packages[{index}].materials")
        if package_materials != expected_materials[str(regime)]:
            message = f"Dataset regime {regime!r} must use materials {list(expected_materials[str(regime)])}."
            raise GenerationConfigError(message)
        package["materials"] = list(package_materials)
        if regime == "id":
            package["membership_seed"] = _optional_uint32(
                package["membership_seed"],
                label=f"campaign.dataset_packages[{index}].membership_seed",
                allow_unresolved=allow_unresolved,
            )
            membership = _mapping(
                package["membership_counts_per_material"],
                label=f"campaign.dataset_packages[{index}].membership_counts_per_material",
            )
            if tuple(membership) != ("train", "validation", "id_test"):
                message = "ID membership counts must declare train, validation, and id_test in order."
                raise GenerationConfigError(message)
            for name, count in membership.items():
                if count is None and allow_unresolved:
                    continue
                membership[name] = _integer(
                    count,
                    label=f"campaign.dataset_packages[{index}].membership_counts_per_material.{name}",
                    minimum=1,
                )
            package["membership_counts_per_material"] = membership
        package["dataset_name"] = dataset_package_name(str(learning_task), package_materials, str(regime))
        packages.append(package)
    return tuple(packages)


def _build_assignments(
    material_family: str,
    sampling_regime: str,
    *,
    case_count: int,
    parameter_ood: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Build deterministic material/regime assignments for one generation batch."""
    assignments: dict[int, dict[str, Any]] = {}
    for case_index in range(1, case_count + 1):
        regime_index = case_index - 1
        ood_group = None
        if sampling_regime == "parameter_ood":
            ood_group = materials.OOD_GROUPS[regime_index % len(materials.OOD_GROUPS)]
        assignments[case_index] = {
            "case_index": case_index,
            "regime_index": regime_index,
            "material_family": material_family,
            "sampling_regime": sampling_regime,
            "ood_group": ood_group,
            "ood_units_per_case": parameter_ood["units_per_case"] if ood_group is not None else 0,
        }
    return assignments


def _safe_text_or_none(value: Any, *, label: str) -> str | None:
    """Return optional safe single-line execution text."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(character in value for character in ("\x00", "\n", "\r")):
        msg = f"{label} must be null or safe non-empty text."
        raise GenerationConfigError(msg)
    return value


def _validate_execution(value: Any) -> dict[str, Any]:
    """Validate physically separate site, runtime, retention, and resource settings."""
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
    site["cores_per_node"] = _integer(site["cores_per_node"], label="execution.site.cores_per_node", minimum=1)

    runtime = _mapping(execution["runtime"], label="execution.runtime")
    _exact_keys(
        runtime,
        required={"executable", "module_initialization", "timeout_seconds", "maximum_failures", "extra_arguments"},
        optional=set(),
        label="execution.runtime",
    )
    runtime["executable"] = _safe_text_or_none(runtime["executable"], label="execution.runtime.executable")
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
    for key in ("module_initialization", "extra_arguments"):
        values = runtime[key]
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item and not any(char in item for char in ("\x00", "\n", "\r")) for item in values
        ):
            msg = f"execution.runtime.{key} must be an ordered list of safe arguments."
            raise GenerationConfigError(msg)
    if any(item == owned or item.startswith(f"{owned}=") for item in runtime["extra_arguments"] for owned in _COMSOL_OWNED_ARGUMENTS):
        msg = "execution.runtime.extra_arguments cannot override case-owned files or one-node execution."
        raise GenerationConfigError(msg)

    retention = _mapping(execution["retention"], label="execution.retention")
    _exact_keys(retention, required={"retain_raw_csv", "retain_solved_model"}, optional=set(), label="execution.retention")
    if not all(isinstance(retention[key], bool) for key in retention):
        msg = "execution.retention controls must be boolean."
        raise TypeError(msg)

    cluster = _mapping(execution["cluster"], label="execution.cluster")
    cluster_keys = {
        "max_nodes",
        "cases_per_node",
        "cores_per_case",
        "max_parallel_cases",
        "cores_per_node",
        "scheduler_kind",
        "partition",
        "wall_time",
        "scheduler_options",
    }
    _exact_keys(cluster, required=cluster_keys, optional=set(), label="execution.cluster")
    for key in ("max_nodes", "cases_per_node", "cores_per_case", "max_parallel_cases", "cores_per_node"):
        cluster[key] = _integer(cluster[key], label=f"execution.cluster.{key}", minimum=1)
    if cluster["scheduler_kind"] not in {"local", "slurm"}:
        msg = "execution.cluster.scheduler_kind must be local or slurm."
        raise GenerationConfigError(msg)
    cluster["partition"] = _safe_text_or_none(cluster["partition"], label="execution.cluster.partition")
    cluster["wall_time"] = _safe_text_or_none(cluster["wall_time"], label="execution.cluster.wall_time")
    options = cluster["scheduler_options"]
    if not isinstance(options, list) or not all(isinstance(item, str) and item.startswith("--") for item in options):
        msg = "execution.cluster.scheduler_options must be long scheduler arguments."
        raise GenerationConfigError(msg)
    if any(option == owned or option.startswith(f"{owned}=") for option in options for owned in _SCHEDULER_OWNED_OPTIONS):
        msg = "execution.cluster.scheduler_options cannot override pipeline-owned allocation directives."
        raise GenerationConfigError(msg)
    if cluster["cases_per_node"] * cluster["cores_per_case"] > cluster["cores_per_node"]:
        msg = "cases_per_node * cores_per_case must not exceed cores_per_node."
        raise GenerationConfigError(msg)
    if cluster["max_parallel_cases"] > cluster["max_nodes"] * cluster["cases_per_node"]:
        msg = "max_parallel_cases must not exceed max_nodes * cases_per_node."
        raise GenerationConfigError(msg)
    if cluster["cores_per_node"] != site["cores_per_node"]:
        msg = "Cluster and site cores_per_node must agree."
        raise GenerationConfigError(msg)
    if cluster["scheduler_kind"] != site["scheduler"]:
        msg = "Cluster scheduler_kind and site scheduler must agree."
        raise GenerationConfigError(msg)
    if cluster["partition"] != site["partition"]:
        msg = "Cluster and site partition must agree."
        raise GenerationConfigError(msg)

    execution["site"] = site
    execution["runtime"] = runtime
    execution["retention"] = retention
    execution["cluster"] = cluster
    return execution


def _input_identity_config(scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return the profile-neutral scientific subset that determines case inputs."""
    value = copy.deepcopy(dict(scientific))
    for key in ("simulation_profile", "output_contract", "available_learning_views", "airflow_source"):
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
    require_executable: bool,
) -> dict[str, Any]:
    """Validate campaign ownership before resolving referenced layers."""
    campaign = _mapping(value, label="generation campaign configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "campaign_name",
        "executable",
        "registry_config",
        "common_config",
        "operations_config",
        "profile_config",
        "execution_config",
        "materials",
        "roles",
        "sampling",
        "dataset_packages",
    }
    _exact_keys(campaign, required=expected, optional=set(), label="generation campaign configuration")
    if campaign["schema_kind"] != "generation_campaign" or campaign["schema_version"] != 1:
        message = "Unsupported generation campaign schema."
        raise GenerationConfigError(message)
    campaign["campaign_name"] = _campaign_name(campaign["campaign_name"])
    if not isinstance(campaign["executable"], bool):
        message = "generation campaign executable must be boolean."
        raise TypeError(message)
    if require_executable and not campaign["executable"]:
        message = "Campaign is non-executable because scientific counts, seeds, ranges, or mappings remain unresolved."
        raise GenerationConfigError(message)
    _campaign_references(campaign, source_path=source_path)

    selected = _name_list(campaign["materials"], label="campaign.materials")
    if any(material_family not in materials.MATERIAL_FAMILIES for material_family in selected):
        message = f"campaign.materials contains an unknown material; allowed values are {list(materials.MATERIAL_FAMILIES)}."
        raise GenerationConfigError(message)
    canonical_selected = tuple(material_family for material_family in materials.MATERIAL_FAMILIES if material_family in selected)
    if selected != canonical_selected:
        message = f"campaign.materials must use canonical material order {list(canonical_selected)}."
        raise GenerationConfigError(message)

    raw_roles = _mapping(campaign["roles"], label="campaign.roles")
    _exact_keys(
        raw_roles,
        required={"seen", "near_family_ood", "far_family_ood"},
        optional=set(),
        label="campaign.roles",
    )
    roles = {
        role: _name_list(raw_roles[role], label=f"campaign.roles.{role}", allow_empty=role != "seen")
        for role in ("seen", "near_family_ood", "far_family_ood")
    }
    flattened = (*roles["seen"], *roles["near_family_ood"], *roles["far_family_ood"])
    if flattened != selected or len(set(flattened)) != len(flattened):
        message = "Campaign roles must partition selected materials in canonical order."
        raise GenerationConfigError(message)
    campaign["materials"] = selected
    campaign["roles"] = roles
    return campaign


def _unresolved_sampling_plan(registry: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return block ownership evidence for an unresolved non-executable batch."""
    dimensions = materials.sampling_block_dimensions(registry)
    return {
        block: {
            "label": block,
            "parameters": list(parameters),
            "effective_dimension": dimensions[block],
            "design_seed": None,
            "design_sha256": None,
            "permutation_seed": None,
            "permutation": [],
            "permutation_sha256": None,
        }
        for block, parameters in materials.SAMPLING_BLOCKS.items()
    }


def _build_batch(
    *,
    source_path: Path,
    profile: profiles.SimulationProfile,
    material: Mapping[str, Any],
    sampling_regime: str,
    case_count: int | None,
    campaign_seed: int | None,
    sampling_method: str,
    parameter_ood: Mapping[str, Any],
    common_config: Mapping[str, Any],
    registry_metadata: Mapping[str, Mapping[str, str]],
    operations_digest: str,
    material_digest: str,
    execution: Mapping[str, Any],
) -> GenerationConfig:
    """Build one immutable material/profile/regime generation batch."""
    material_family = str(material["material_family"])
    batch_name = f"{profile.id}__{material_family}__{sampling_regime}"
    input_batch_label = f"{material_family}__{sampling_regime}"
    batch_seed = None if campaign_seed is None else derive_seed(campaign_seed, "generation_batch", input_batch_label)
    normalized_count = 0 if case_count is None else case_count
    assignments = _build_assignments(
        material_family,
        sampling_regime,
        case_count=normalized_count,
        parameter_ood=parameter_ood,
    )
    case_indices = tuple(assignments)
    registry = material["parameter_registry"]
    if batch_seed is None:
        sampling_plan = _unresolved_sampling_plan(registry)
    else:
        from . import generation_sampling as sampling_service  # noqa: PLC0415 -- typed config/sampling cycle

        sampling_plan = sampling_service.build_sampling_plan(
            registry=registry,
            case_count=normalized_count,
            seed_base=batch_seed,
            method=sampling_method,
        )
    output_contract = {
        "exports_root": "exports",
        "exports": common_config["_profile_exports"],
        "static_field_names": list(profiles.STATIC_FIELD_NAMES),
        "static_field_units": list(profiles.STATIC_FIELD_UNITS),
        "transient_field_names": (list(profiles.TRANSIENT_FIELD_NAMES) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "transient_field_units": (list(profiles.TRANSIENT_FIELD_UNITS) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "global_field_names": (list(profiles.GLOBAL_FIELD_NAMES) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
        "global_field_units": (list(profiles.GLOBAL_FIELD_UNITS) if profile.id == profiles.TRANSIENT_DRYING_PROFILE else []),
    }
    scientific = {
        "schema_kind": "resolved_generation_batch",
        "schema_version": CONFIG_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "simulation_profile": profile.id,
        "material": copy.deepcopy(dict(material)),
        "material_config_digest": material_digest,
        "operation_config_digest": operations_digest,
        "sampling_regime": sampling_regime,
        "case_count": case_count,
        "sampling": {
            "method": sampling_method,
            "seed_base": batch_seed,
            "blocks": sampling_plan,
            "block_dimensions": dict(materials.SAMPLING_BLOCK_DIMENSIONS),
        },
        "parameter_ood": copy.deepcopy(dict(parameter_ood)),
        "assignments": [assignments[index] for index in case_indices],
        "registry_metadata": copy.deepcopy(dict(registry_metadata)),
        "scientific_fixed_values": copy.deepcopy(dict(common_config["scientific_fixed_values"])),
        "physical_formulas": copy.deepcopy(dict(common_config["physical_formulas"])),
        "grid": copy.deepcopy(dict(common_config["grid"])),
        "time": copy.deepcopy(dict(common_config["time"])),
        "input_contract": copy.deepcopy(dict(common_config["input_contract"])),
        "output_contract": output_contract,
        "storage": copy.deepcopy(dict(common_config["storage"])),
        "available_learning_views": list(profile.available_learning_views),
        "airflow_source": profile.airflow_source,
    }
    scientific_digest = common.serialization.canonical_json_sha256(scientific)
    case_input_digest = common.serialization.canonical_json_sha256(_input_identity_config(scientific))
    batch_id = f"{batch_name}__{scientific_digest[:16]}"
    return GenerationConfig(
        source_path=source_path,
        profile=profile,
        material_family=material_family,
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


def load_campaign_config(path: Path | str, *, require_executable: bool = True) -> CampaignConfig:
    """Resolve one campaign, every predeclared subbatch, and each dataset plan."""
    source_path = Path(path).expanduser().resolve()
    campaign = _validate_campaign_header(
        _load_yaml(source_path, label="generation campaign configuration"),
        source_path=source_path,
        require_executable=require_executable,
    )
    definitions, registry_metadata = materials.validate_semantic_registry(
        _load_yaml(campaign["registry_config"], label="generation parameter registry")
    )
    common_config = _validate_common(
        _load_yaml(campaign["common_config"], label="generation common configuration"),
        require_executable=require_executable,
    )
    operations = _validate_operations(
        _load_yaml(campaign["operations_config"], label="generation operations configuration"),
        require_executable=require_executable,
    )
    profile_raw = _load_yaml(campaign["profile_config"], label="generation profile configuration")
    profile_id = profile_raw.get("simulation_profile")
    if not isinstance(profile_id, str):
        message = "generation profile simulation_profile must be text."
        raise TypeError(message)
    profile = profiles.get_profile(profile_id)
    profile_config = _validate_profile_config(
        profile_raw,
        profile=profile,
        require_executable=require_executable,
    )
    execution = _validate_execution(_load_yaml(campaign["execution_config"], label="generation execution configuration"))

    resolved_materials: dict[str, dict[str, Any]] = {}
    material_digests: dict[str, str] = {}
    project_root = common.paths.get_project_root()
    for material_family in campaign["materials"]:
        material_path = project_root / "configs" / "generation" / "materials" / f"{material_family}.yaml"
        material_raw = _load_yaml(material_path, label=f"material {material_family} configuration")
        if material_raw.get("material_family") != material_family:
            message = f"Material filename and material_family disagree for {material_path}."
            raise GenerationConfigError(message)
        resolved_materials[material_family] = materials.resolve_material_definition(
            definitions,
            common_config["parameter_values"],
            operations["parameter_values"],
            material_raw,
            allow_unresolved=not require_executable,
        )
        material_digests[material_family] = common.serialization.canonical_json_sha256(material_raw)

    sampling = _mapping(campaign["sampling"], label="campaign.sampling")
    _exact_keys(
        sampling,
        required={"method", "seed_base", "counts", "parameter_ood"},
        optional=set(),
        label="campaign.sampling",
    )
    if sampling["method"] not in {"lhs", "sobol"}:
        message = "campaign.sampling.method must be lhs or sobol."
        raise GenerationConfigError(message)
    campaign_seed = _optional_uint32(
        sampling["seed_base"],
        label="campaign.sampling.seed_base",
        allow_unresolved=not require_executable,
    )
    counts = _validate_batch_counts(
        sampling["counts"],
        materials_in_order=campaign["materials"],
        seen=campaign["roles"]["seen"],
        allow_unresolved=not require_executable,
    )
    parameter_ood = _validate_parameter_ood_policy(
        sampling["parameter_ood"],
        counts=counts["parameter_ood"],
        allow_unresolved=not require_executable,
    )
    dataset_packages = _validate_dataset_packages(
        campaign["dataset_packages"],
        profile=profile,
        roles=campaign["roles"],
        allow_unresolved=not require_executable,
    )
    expected_regimes = {"id", "parameter_ood"}
    if campaign["roles"]["near_family_ood"]:
        expected_regimes.add("near_family_ood")
    if campaign["roles"]["far_family_ood"]:
        expected_regimes.add("far_family_ood")
    actual_regimes = {str(package["evaluation_regime"]) for package in dataset_packages}
    if actual_regimes != expected_regimes:
        message = f"Campaign dataset packages must cover exactly {sorted(expected_regimes)}."
        raise GenerationConfigError(message)

    common_with_profile = copy.deepcopy(common_config)
    common_with_profile["_profile_exports"] = profile_config["exports"]
    operations_digest = common.serialization.canonical_json_sha256(operations)
    batches: list[GenerationConfig] = []
    for sampling_regime in ("natural", "parameter_ood"):
        for material_family, count in counts[sampling_regime].items():
            batches.append(
                _build_batch(
                    source_path=source_path,
                    profile=profile,
                    material=resolved_materials[material_family],
                    sampling_regime=sampling_regime,
                    case_count=count,
                    campaign_seed=campaign_seed,
                    sampling_method=str(sampling["method"]),
                    parameter_ood=parameter_ood,
                    common_config=common_with_profile,
                    registry_metadata=registry_metadata,
                    operations_digest=operations_digest,
                    material_digest=material_digests[material_family],
                    execution=execution,
                )
            )
    batch_names = [batch.batch_name for batch in batches]
    if len(batch_names) != len(set(batch_names)):
        message = "Campaign planned duplicate batch names."
        raise GenerationConfigError(message)

    campaign_scientific = {
        "schema_kind": "resolved_generation_campaign",
        "schema_version": 1,
        "campaign_name": campaign["campaign_name"],
        "simulation_profile": profile.id,
        "roles": {name: list(values) for name, values in campaign["roles"].items()},
        "sampling_method": sampling["method"],
        "campaign_seed": campaign_seed,
        "batch_ids": [batch.batch_id for batch in batches],
        "dataset_packages": list(dataset_packages),
        "registry_config_digest": common.serialization.canonical_json_sha256(
            _load_yaml(campaign["registry_config"], label="generation parameter registry")
        ),
        "common_config_digest": common.serialization.canonical_json_sha256(common_config),
        "operation_config_digest": operations_digest,
        "material_config_digests": material_digests,
    }
    campaign_digest = common.serialization.canonical_json_sha256(campaign_scientific)
    campaign_name = str(campaign["campaign_name"])
    return CampaignConfig(
        source_path=source_path,
        campaign_name=campaign_name,
        campaign_digest=campaign_digest,
        campaign_id=f"{campaign_name}__{campaign_digest[:16]}",
        profile=profile,
        roles=copy.deepcopy(campaign["roles"]),
        batches=tuple(batches),
        dataset_packages=dataset_packages,
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
