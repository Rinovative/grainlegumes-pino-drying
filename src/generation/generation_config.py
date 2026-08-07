"""
===============================================================================
generation_config.py
===============================================================================
Load and validate explicit reference-simulation generation configurations.
Responsibilities:
  - Validate generic case, input-adapter, export, runtime, and cluster settings
  - Bind one authoritative simulation profile and immutable template identity
  - Derive deterministic profile-qualified batch membership and identity
Design principles:
  - Scientific values come only from explicit configuration
  - Runtime resource choices do not alter scientific batch identity
  - Unknown or ambiguous configuration is rejected before case preparation
This module does NOT:
  - Invent drying parameters, ranges, schedules, or output channels
  - Generate fields, run COMSOL, or mutate generation storage
===============================================================================
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from src import common

from . import generation_profiles as profiles

CONFIG_SCHEMA_VERSION = 1
GENERATOR_VERSION = "python_multiscale_v1"
CASE_ID_WIDTH = 4
MIN_SCHEDULE_SIZE = 2
_COMSOL_OWNED_ARGUMENTS = ("-inputfile", "-outputfile", "-np", "-nn", "-nnhost", "-mpihosts")
_SCHEDULER_OWNED_OPTIONS = (
    "--nodes",
    "--ntasks",
    "--ntasks-per-node",
    "--cpus-per-task",
    "--array",
    "--chdir",
    "--job-name",
    "--wrap",
)
PAIR_PARAMETER_SIZE = 2
UINT32_MAX = 2**32 - 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_SCALAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "simulation_profile",
        "cases",
        "generator",
        "sampling",
        "inputs",
        "exports",
        "execution",
        "cluster",
    }
)
REQUIRED_GENERATOR_PARAMETERS = frozenset(
    {
        "base_len_rel",
        "smooth_len_rel",
        "ms_weight",
        "anisotropy",
        "coupling",
        "noise_level",
        "noise_granularity",
        "noise_bias",
        "k_mean",
        "var_rel",
        "a_max",
        "a_gamma",
        "tensor_strength",
        "theta_jitter",
        "theta_smooth_rel",
        "A_rel",
        "eps_min_global",
        "eps_max_global",
        "eps_smooth_rel",
        "texture_amp",
        "p_inlet_mean",
        "a_sin",
        "f_sin",
        "phi_sin",
        "k_gauss",
        "a_gauss",
        "sigma_gauss",
        "gauss_jitter",
        "a_lin",
    }
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """One fully validated profile-qualified generation configuration."""

    source_path: Path
    profile: profiles.SimulationProfile
    values: dict[str, Any]
    template_path: Path
    template_sha256: str
    case_indices: tuple[int, ...]
    seed_base: int
    overrides: dict[int, dict[str, Any]]
    batch_identity: str
    batch_id: str

    def case_id(self, case_index: int) -> str:
        """Return the canonical identifier for one configured case index."""
        if case_index not in self.case_indices:
            message = f"Case index {case_index} is not a member of batch {self.batch_id}."
            raise ValueError(message)
        return f"case_{case_index:0{CASE_ID_WIDTH}d}"

    def case_seed(self, case_index: int) -> int:
        """Return the versioned case-level seed derivation."""
        self.case_id(case_index)
        seed = self.seed_base + case_index
        if seed > UINT32_MAX:
            message = f"Derived case seed exceeds the uint32 range for case {case_index}: {seed}."
            raise ValueError(message)
        return seed


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one string-keyed mapping copy."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(value)


def _exact_keys(value: dict[str, Any], *, required: set[str], optional: set[str], label: str) -> None:
    """Require exact known mapping keys and all required entries."""
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise ValueError(message)


def _positive_int(value: Any, *, label: str, minimum: int = 1) -> int:
    """Return one bounded positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{label} must be an integer >= {minimum}, got {value!r}."
        raise ValueError(message)
    return value


def _finite_real(value: Any, *, label: str) -> float:
    """Return one finite real number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must be a finite real number, got {value!r}."
        raise ValueError(message)
    return float(value)


def validate_relative_file(value: Any, *, label: str, suffix: str | None = None) -> str:
    """Validate one case-local filename with no directory component."""
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{label} must be one non-empty case-local filename."
        raise ValueError(message)
    path = Path(value)
    if path.name != value or path.is_absolute() or value in {".", ".."} or "\x00" in value:
        message = f"{label} must be one non-empty case-local filename, got {value!r}."
        raise ValueError(message)
    if suffix is not None and path.suffix.lower() != suffix:
        message = f"{label} must end in {suffix!r}, got {value!r}."
        raise ValueError(message)
    return value


def _delimiter(value: Any, *, label: str) -> str:
    """Validate one deterministic text-table delimiter."""
    if value == "\\t":
        return "\t"
    if value not in {",", ";", "\t"}:
        message = f"{label} must be ',', ';', or '\\t'."
        raise ValueError(message)
    return str(value)


def _validate_cases(value: Any) -> tuple[tuple[int, ...], int, dict[int, dict[str, Any]]]:
    cases = _mapping(value, label="cases")
    _exact_keys(cases, required={"indices", "seed_base"}, optional={"overrides"}, label="cases")
    raw_indices = cases["indices"]
    if not isinstance(raw_indices, list) or not raw_indices:
        message = "cases.indices must be one non-empty ordered list of positive integers."
        raise ValueError(message)
    indices = tuple(_positive_int(item, label="cases.indices item") for item in raw_indices)
    if len(indices) != len(set(indices)) or list(indices) != sorted(indices):
        message = "cases.indices must be strictly increasing and duplicate-free."
        raise ValueError(message)
    seed_base = _positive_int(cases["seed_base"], label="cases.seed_base", minimum=0)
    if seed_base + indices[-1] > UINT32_MAX:
        message = "cases.seed_base plus the largest case index exceeds the uint32 seed range."
        raise ValueError(message)
    raw_overrides = cases.get("overrides", {})
    if not isinstance(raw_overrides, dict):
        message = "cases.overrides must be a mapping keyed by configured case index."
        raise TypeError(message)
    overrides: dict[int, dict[str, Any]] = {}
    for raw_key, raw_override in raw_overrides.items():
        try:
            index = int(raw_key)
        except (TypeError, ValueError) as error:
            message = f"cases.overrides key must be an integer case index, got {raw_key!r}."
            raise ValueError(message) from error
        if str(index) != str(raw_key) and raw_key != index:
            message = f"cases.overrides key is ambiguous: {raw_key!r}."
            raise ValueError(message)
        if index not in indices:
            message = f"cases.overrides contains non-member case index {index}."
            raise ValueError(message)
        override = _mapping(raw_override, label=f"cases.overrides[{index}]")
        _exact_keys(override, required=set(), optional={"generator", "scalars", "schedule_rows"}, label=f"cases.overrides[{index}]")
        overrides[index] = override
    return indices, seed_base, overrides


def _validate_generator(value: Any) -> dict[str, Any]:
    generator = _mapping(value, label="generator")
    _exact_keys(generator, required={"version", "domain", "parameters"}, optional=set(), label="generator")
    if generator["version"] != GENERATOR_VERSION:
        message = f"generator.version must be {GENERATOR_VERSION!r}."
        raise ValueError(message)
    domain = _mapping(generator["domain"], label="generator.domain")
    _exact_keys(domain, required={"length_x_m", "length_y_m", "resolution_m"}, optional=set(), label="generator.domain")
    for name in ("length_x_m", "length_y_m", "resolution_m"):
        domain[name] = _finite_real(domain[name], label=f"generator.domain.{name}")
        if domain[name] <= 0:
            message = f"generator.domain.{name} must be strictly positive."
            raise ValueError(message)
    if domain["resolution_m"] > min(domain["length_x_m"], domain["length_y_m"]):
        message = "generator.domain.resolution_m cannot exceed the shorter domain length."
        raise ValueError(message)
    parameters = _mapping(generator["parameters"], label="generator.parameters")
    missing = sorted(REQUIRED_GENERATOR_PARAMETERS.difference(parameters))
    unknown = sorted(set(parameters).difference(REQUIRED_GENERATOR_PARAMETERS))
    if missing or unknown:
        message = f"generator.parameters must match the existing spatial generator contract: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    generator["domain"] = domain
    generator["parameters"] = validate_generator_parameters(parameters)
    return generator


def _generator_pair(parameters: dict[str, Any], name: str) -> tuple[float, float]:
    """Return one finite two-value generator parameter."""
    value = parameters[name]
    if not isinstance(value, (list, tuple)) or len(value) != PAIR_PARAMETER_SIZE:
        message = f"generator.parameters.{name} must contain exactly two values."
        raise ValueError(message)
    first = _finite_real(value[0], label=f"generator.parameters.{name}[0]")
    second = _finite_real(value[1], label=f"generator.parameters.{name}[1]")
    return first, second


def validate_generator_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Validate every maintained spatial-generator formula domain."""
    missing = sorted(REQUIRED_GENERATOR_PARAMETERS.difference(parameters))
    unknown = sorted(set(parameters).difference(REQUIRED_GENERATOR_PARAMETERS))
    if missing or unknown:
        message = f"generator.parameters contract mismatch: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    values: dict[str, Any] = {
        name: _finite_real(parameters[name], label=f"generator.parameters.{name}")
        for name in REQUIRED_GENERATOR_PARAMETERS
        if name not in {"ms_weight", "anisotropy"}
    }
    values["ms_weight"] = _generator_pair(parameters, "ms_weight")
    values["anisotropy"] = _generator_pair(parameters, "anisotropy")
    positive = (
        "base_len_rel",
        "smooth_len_rel",
        "k_mean",
        "var_rel",
        "a_max",
        "a_gamma",
        "A_rel",
        "sigma_gauss",
    )
    if any(float(values[name]) <= 0 for name in positive):
        message = f"Generator parameters {positive} must be strictly positive."
        raise ValueError(message)
    nonnegative = (
        "noise_level",
        "noise_granularity",
        "tensor_strength",
        "theta_jitter",
        "theta_smooth_rel",
        "eps_smooth_rel",
        "texture_amp",
        "f_sin",
        "gauss_jitter",
    )
    if any(float(values[name]) < 0 for name in nonnegative):
        message = f"Generator parameters {nonnegative} must be non-negative."
        raise ValueError(message)
    for name in ("coupling", "noise_granularity", "noise_bias"):
        if not 0 <= float(values[name]) <= 1:
            message = f"generator.parameters.{name} must lie in [0, 1]."
            raise ValueError(message)
    eps_min = float(values["eps_min_global"])
    eps_max = float(values["eps_max_global"])
    if not 0 < eps_min < eps_max <= 1:
        message = "Porosity bounds must satisfy 0 < eps_min_global < eps_max_global <= 1."
        raise ValueError(message)
    weights = _generator_pair(parameters, "ms_weight")
    anisotropy = _generator_pair(parameters, "anisotropy")
    if any(value <= 0 for value in weights) or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        message = "generator.parameters.ms_weight must contain positive values summing to one."
        raise ValueError(message)
    if any(value <= 0 for value in anisotropy):
        message = "generator.parameters.anisotropy must contain positive stretch factors."
        raise ValueError(message)
    k_gauss = float(values["k_gauss"])
    if not k_gauss.is_integer() or k_gauss < 0:
        message = "generator.parameters.k_gauss must be a non-negative integer."
        raise ValueError(message)
    return values


def _validate_scalar_entry(value: Any, *, label: str) -> dict[str, Any]:
    entry = _mapping(value, label=label)
    _exact_keys(entry, required={"name", "value"}, optional={"unit"}, label=label)
    name = entry["name"]
    if not isinstance(name, str) or _SAFE_SCALAR_NAME.fullmatch(name) is None:
        message = f"{label}.name must be a safe scalar identifier, got {name!r}."
        raise ValueError(message)
    entry["value"] = _finite_real(entry["value"], label=f"{label}.value")
    unit = entry.get("unit")
    if unit is not None:
        if not isinstance(unit, str) or unit.strip() != unit or any(character in unit for character in ("\x00", "\n", "\r")):
            message = f"{label}.unit must be safe single-line text."
            raise ValueError(message)
        if Path(unit).is_absolute() or ".." in unit or "\\" in unit:
            message = f"{label}.unit cannot embed a filesystem path."
            raise ValueError(message)
    return entry


def validate_schedule_rows(columns: list[str], rows: Any, *, label: str) -> list[list[float]]:
    """Validate explicit schedule endpoints, widths, values, and time order."""
    if not isinstance(rows, list) or len(rows) < MIN_SCHEDULE_SIZE:
        message = f"{label} must contain explicit first and final rows."
        raise ValueError(message)
    validated: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(columns):
            message = f"{label}[{row_index}] has the wrong width."
            raise ValueError(message)
        validated.append([_finite_real(item, label=f"{label}[{row_index}] item") for item in row])
    if any(right[0] <= left[0] for left, right in pairwise(validated)):
        message = f"{label} time values must be strictly increasing."
        raise ValueError(message)
    return validated


def _validate_inputs(value: Any) -> dict[str, Any]:
    inputs = _mapping(value, label="inputs")
    _exact_keys(inputs, required={"spatial_files"}, optional={"scalar_file", "schedule_file"}, label="inputs")
    spatial_files = inputs["spatial_files"]
    if not isinstance(spatial_files, list) or not spatial_files:
        message = "inputs.spatial_files must contain at least one configured adapter."
        raise ValueError(message)
    filenames: list[str] = []
    validated_spatial: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(spatial_files):
        spec = _mapping(raw_spec, label=f"inputs.spatial_files[{index}]")
        _exact_keys(spec, required={"filename", "delimiter", "columns"}, optional=set(), label=f"inputs.spatial_files[{index}]")
        spec["filename"] = validate_relative_file(spec["filename"], label=f"inputs.spatial_files[{index}].filename")
        spec["delimiter"] = _delimiter(spec["delimiter"], label=f"inputs.spatial_files[{index}].delimiter")
        columns = spec["columns"]
        if not isinstance(columns, list) or not columns or not all(isinstance(column, str) and column for column in columns):
            message = f"inputs.spatial_files[{index}].columns must be a non-empty ordered list of field names."
            raise ValueError(message)
        if len(columns) != len(set(columns)):
            message = f"inputs.spatial_files[{index}].columns contains duplicates."
            raise ValueError(message)
        filenames.append(spec["filename"])
        validated_spatial.append(spec)
    scalar_file = inputs.get("scalar_file")
    if scalar_file is not None:
        scalar = _mapping(scalar_file, label="inputs.scalar_file")
        _exact_keys(
            scalar,
            required={"filename", "format", "delimiter", "include_header", "entries"},
            optional={"required_when_empty"},
            label="inputs.scalar_file",
        )
        scalar["filename"] = validate_relative_file(scalar["filename"], label="inputs.scalar_file.filename")
        scalar["delimiter"] = _delimiter(scalar["delimiter"], label="inputs.scalar_file.delimiter")
        if scalar["format"] not in {"long", "wide"}:
            message = "inputs.scalar_file.format must be 'long' or 'wide'."
            raise ValueError(message)
        if not isinstance(scalar["include_header"], bool) or not isinstance(scalar.get("required_when_empty", False), bool):
            message = "inputs.scalar_file header and empty-file controls must be boolean."
            raise TypeError(message)
        raw_entries = scalar["entries"]
        if not isinstance(raw_entries, list):
            message = "inputs.scalar_file.entries must be an ordered list."
            raise TypeError(message)
        entries = [_validate_scalar_entry(item, label=f"inputs.scalar_file.entries[{index}]") for index, item in enumerate(raw_entries)]
        names = [entry["name"] for entry in entries]
        if len(names) != len(set(names)):
            message = "inputs.scalar_file.entries contains duplicate names."
            raise ValueError(message)
        scalar["entries"] = entries
        inputs["scalar_file"] = scalar
        filenames.append(scalar["filename"])
    schedule_file = inputs.get("schedule_file")
    if schedule_file is not None:
        schedule = _mapping(schedule_file, label="inputs.schedule_file")
        _exact_keys(schedule, required={"filename", "delimiter", "columns", "rows"}, optional=set(), label="inputs.schedule_file")
        schedule["filename"] = validate_relative_file(schedule["filename"], label="inputs.schedule_file.filename")
        schedule["delimiter"] = _delimiter(schedule["delimiter"], label="inputs.schedule_file.delimiter")
        columns = schedule["columns"]
        if not isinstance(columns, list) or len(columns) < MIN_SCHEDULE_SIZE or not all(isinstance(column, str) and column for column in columns):
            message = "inputs.schedule_file.columns must contain time followed by at least one value column."
            raise ValueError(message)
        if len(columns) != len(set(columns)):
            message = "inputs.schedule_file.columns contains duplicates."
            raise ValueError(message)
        schedule["rows"] = validate_schedule_rows(
            columns,
            schedule["rows"],
            label="inputs.schedule_file.rows",
        )
        inputs["schedule_file"] = schedule
        filenames.append(schedule["filename"])
    if len(filenames) != len(set(filenames)) or "model.mph" in filenames or "case.json" in filenames:
        message = "Configured input filenames must be unique and cannot replace model.mph or case.json."
        raise ValueError(message)
    inputs["spatial_files"] = validated_spatial
    return inputs


def _validate_exports(value: Any, *, profile: profiles.SimulationProfile) -> dict[str, Any]:
    """Validate raw export collection and profile-owned semantic role mappings."""
    exports = _mapping(value, label="exports")
    _exact_keys(exports, required={"root", "contracts"}, optional=set(), label="exports")
    root = validate_relative_file(exports["root"], label="exports.root")
    contracts = exports["contracts"]
    if not isinstance(contracts, list) or not contracts:
        message = "exports.contracts must contain explicit profile export roles."
        raise ValueError(message)
    validated: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for index, raw_contract in enumerate(contracts):
        label = f"exports.contracts[{index}]"
        contract = _mapping(raw_contract, label=label)
        _exact_keys(
            contract,
            required={"role", "pattern", "required", "allow_multiple", "format"},
            optional={"delimiter", "columns", "time_column"},
            label=label,
        )
        role = contract["role"]
        if not isinstance(role, str) or role in seen_roles:
            message = f"{label}.role must be one unique profile-owned role."
            raise ValueError(message)
        role_spec = profile.export_role(role)
        seen_roles.add(role)
        pattern = contract["pattern"]
        if (
            not isinstance(pattern, str)
            or not pattern
            or Path(pattern).is_absolute()
            or ".." in Path(pattern).parts
            or "**" in pattern
            or "\x00" in pattern
        ):
            message = f"{label}.pattern must be one narrowly scoped relative pattern."
            raise ValueError(message)
        if contract["required"] is not role_spec.required or contract["allow_multiple"] is not role_spec.allow_multiple:
            message = f"{label} cardinality must match profile {profile.id!r} role {role!r}."
            raise ValueError(message)
        if contract["format"] not in {"numeric_table", "text", "binary"}:
            message = f"{label}.format must be numeric_table, text, or binary."
            raise ValueError(message)
        if contract["format"] == "numeric_table":
            contract["delimiter"] = _delimiter(contract.get("delimiter", ";"), label=f"{label}.delimiter")
        elif "delimiter" in contract:
            message = f"{label}.delimiter applies only to numeric_table exports."
            raise ValueError(message)
        if role_spec.canonical_fields:
            if contract["format"] != "numeric_table":
                message = f"{label} must be a numeric_table to supply the steady_flow learning view."
                raise ValueError(message)
            columns = _mapping(contract.get("columns"), label=f"{label}.columns")
            if tuple(columns) != role_spec.canonical_fields or any(not isinstance(value, str) or not value for value in columns.values()):
                message = (
                    f"{label}.columns must map the exact canonical fields {list(role_spec.canonical_fields)} in order to configured export headers."
                )
                raise ValueError(message)
            contract["columns"] = columns
            time_column = contract.get("time_column")
            if time_column is not None and (not isinstance(time_column, str) or not time_column):
                message = f"{label}.time_column must be a non-empty export header when supplied."
                raise ValueError(message)
        elif "columns" in contract or "time_column" in contract:
            message = f"{label} raw transient role cannot define a final training-field mapping."
            raise ValueError(message)
        validated.append(contract)
    missing = sorted(set(profile.required_export_roles).difference(seen_roles))
    extra = sorted(seen_roles.difference(spec.role for spec in profile.export_roles))
    if missing or extra:
        message = f"Profile {profile.id!r} export roles are incomplete: missing={missing}, extra={extra}."
        raise ValueError(message)
    exports["root"] = root
    exports["contracts"] = validated
    return exports


def _validate_profile_inputs(inputs: dict[str, Any], *, profile: profiles.SimulationProfile) -> None:
    """Bind generic input adapters to one profile-owned filename and field contract."""
    spatial = inputs["spatial_files"]
    expected_columns = [source for _canonical, source in profile.spatial_field_mapping]
    if len(spatial) != 1 or spatial[0]["filename"] != profile.spatial_input_filename or spatial[0]["columns"] != expected_columns:
        message = f"Profile {profile.id!r} requires one spatial adapter {profile.spatial_input_filename!r} with columns {expected_columns}."
        raise ValueError(message)
    if not profile.scalar_file_allowed and inputs.get("scalar_file") is not None:
        message = f"Profile {profile.id!r} does not accept a scalar-file adapter."
        raise ValueError(message)
    if not profile.schedule_file_allowed and inputs.get("schedule_file") is not None:
        message = f"Profile {profile.id!r} does not accept a schedule-file adapter."
        raise ValueError(message)


def _validate_execution(value: Any) -> dict[str, Any]:
    execution = _mapping(value, label="execution")
    _exact_keys(
        execution,
        required={"timeout_seconds", "retain_solved_model"},
        optional={"executable", "extra_arguments"},
        label="execution",
    )
    timeout = _finite_real(execution["timeout_seconds"], label="execution.timeout_seconds")
    if timeout <= 0:
        message = "execution.timeout_seconds must be strictly positive."
        raise ValueError(message)
    if not isinstance(execution["retain_solved_model"], bool):
        message = "execution.retain_solved_model must be boolean."
        raise TypeError(message)
    executable = execution.get("executable")
    if executable is not None and (not isinstance(executable, str) or not executable or any(char in executable for char in ("\x00", "\n", "\r"))):
        message = "execution.executable must be safe non-empty text when supplied."
        raise ValueError(message)
    extra_arguments = execution.get("extra_arguments", [])
    if not isinstance(extra_arguments, list) or not all(
        isinstance(item, str) and item and not any(char in item for char in ("\x00", "\n", "\r")) for item in extra_arguments
    ):
        message = "execution.extra_arguments must be an ordered list of safe non-empty arguments."
        raise ValueError(message)
    if any(item == owned or item.startswith(f"{owned}=") for item in extra_arguments for owned in _COMSOL_OWNED_ARGUMENTS):
        message = "execution.extra_arguments cannot override case-owned files, core allocation, or one-node execution."
        raise ValueError(message)
    execution["timeout_seconds"] = timeout
    execution["extra_arguments"] = extra_arguments
    return execution


def _validate_cluster(value: Any) -> dict[str, Any]:
    cluster = _mapping(value, label="cluster")
    _exact_keys(
        cluster,
        required={"cores_per_node", "scheduler_kind"},
        optional={"scheduler_options", "config_path"},
        label="cluster",
    )
    cluster["cores_per_node"] = _positive_int(cluster["cores_per_node"], label="cluster.cores_per_node")
    if cluster["scheduler_kind"] not in {"local", "slurm"}:
        message = "cluster.scheduler_kind must be explicitly 'local' or 'slurm'."
        raise ValueError(message)
    options = cluster.get("scheduler_options", [])
    if not isinstance(options, list) or not all(isinstance(item, str) and item.startswith("--") for item in options):
        message = "cluster.scheduler_options must be an ordered list of long scheduler arguments."
        raise ValueError(message)
    if any(option == owned or option.startswith(f"{owned}=") for option in options for owned in _SCHEDULER_OWNED_OPTIONS):
        message = "cluster.scheduler_options cannot override pipeline-owned allocation or worker directives."
        raise ValueError(message)
    cluster["scheduler_options"] = options
    return cluster


def _load_cluster_config(value: Any) -> dict[str, Any]:
    """Resolve the one durable cluster execution configuration when referenced."""
    reference = _mapping(value, label="cluster")
    if set(reference) != {"config_path"}:
        return _validate_cluster(reference)
    raw_path = reference["config_path"]
    if not isinstance(raw_path, str) or not raw_path:
        message = "cluster.config_path must be non-empty text."
        raise ValueError(message)
    configured = Path(raw_path).expanduser()
    path = configured if configured.is_absolute() else common.paths.get_project_root() / configured
    path = path.resolve()
    if not path.is_file():
        message = f"Cluster execution configuration does not exist: {path}"
        raise FileNotFoundError(message)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not load cluster execution configuration: {path}"
        raise ValueError(message) from error
    cluster = _mapping(loaded, label="cluster execution configuration")
    _exact_keys(
        cluster,
        required={"schema_kind", "schema_version", "cores_per_node", "scheduler_kind", "scheduler_options"},
        optional=set(),
        label="cluster execution configuration",
    )
    if cluster.pop("schema_kind") != "generation_cluster_execution" or cluster.pop("schema_version") != 1:
        message = f"Unsupported cluster execution configuration schema: {path}"
        raise ValueError(message)
    cluster["config_path"] = raw_path
    return _validate_cluster(cluster)


def _validate_sampling(  # noqa: C901, PLR0912 -- one strict schema boundary enumerates transform-specific rules
    value: Any, *, case_count: int, generator: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any] | None:
    if value is None:
        return None
    sampling = _mapping(value, label="sampling")
    _exact_keys(sampling, required={"method", "variation", "parameters"}, optional=set(), label="sampling")
    if sampling["method"] not in {"uniform", "lhs", "sobol"}:
        message = "sampling.method must be uniform, lhs, or sobol."
        raise ValueError(message)
    variation = _finite_real(sampling["variation"], label="sampling.variation")
    if variation < 0:
        message = "sampling.variation must be non-negative."
        raise ValueError(message)
    raw_parameters = sampling["parameters"]
    if not isinstance(raw_parameters, list):
        message = "sampling.parameters must be an ordered list."
        raise TypeError(message)
    scalar_names = {entry["name"] for entry in (inputs.get("scalar_file") or {}).get("entries", [])}
    seen: set[tuple[str, str, int | None]] = set()
    parameters: list[dict[str, Any]] = []
    for index, raw_parameter in enumerate(raw_parameters):
        parameter = _mapping(raw_parameter, label=f"sampling.parameters[{index}]")
        _exact_keys(
            parameter,
            required={"target", "name", "transform"},
            optional={"minimum", "maximum", "scale", "group", "index"},
            label=f"sampling.parameters[{index}]",
        )
        target = parameter["target"]
        name = parameter["name"]
        if target not in {"generator", "scalar"} or not isinstance(name, str):
            message = f"sampling.parameters[{index}] target/name is invalid."
            raise ValueError(message)
        component = parameter.get("index")
        if component is not None and (isinstance(component, bool) or not isinstance(component, int) or component < 0):
            message = f"sampling.parameters[{index}].index must be a non-negative integer."
            raise ValueError(message)
        if target == "generator":
            if name not in generator["parameters"]:
                message = f"sampling.parameters[{index}] references unknown generator parameter {name!r}."
                raise ValueError(message)
            base_value = generator["parameters"][name]
            is_pair = isinstance(base_value, (list, tuple))
            if is_pair and (component is None or component >= len(base_value)):
                message = f"sampling.parameters[{index}] must select a valid index for pair parameter {name!r}."
                raise ValueError(message)
            if not is_pair and component is not None:
                message = f"sampling.parameters[{index}] cannot index scalar generator parameter {name!r}."
                raise ValueError(message)
        elif name not in scalar_names:
            message = f"sampling.parameters[{index}] references unknown scalar {name!r}."
            raise ValueError(message)
        elif component is not None:
            message = f"sampling.parameters[{index}] cannot index scalar input {name!r}."
            raise ValueError(message)
        key = (target, name, component)
        if key in seen:
            message = f"sampling.parameters contains duplicate target {target}.{name}[{component!r}]."
            raise ValueError(message)
        seen.add(key)
        if parameter["transform"] not in {"log", "logit", "linear", "phase", "integer", "softmax"}:
            message = f"sampling.parameters[{index}].transform is unsupported."
            raise ValueError(message)
        transform = parameter["transform"]
        numeric_options = {"minimum", "maximum", "scale"}
        present_options = numeric_options.intersection(parameter)
        if transform == "integer":
            if present_options != numeric_options:
                message = f"sampling.parameters[{index}] integer transform requires minimum, maximum, and scale."
                raise ValueError(message)
            for optional_name in sorted(numeric_options):
                parameter[optional_name] = _finite_real(
                    parameter[optional_name],
                    label=f"sampling.parameters[{index}].{optional_name}",
                )
            if (
                not float(parameter["minimum"]).is_integer()
                or not float(parameter["maximum"]).is_integer()
                or parameter["minimum"] > parameter["maximum"]
                or parameter["scale"] <= 0
            ):
                message = f"sampling.parameters[{index}] integer bounds and scale are invalid."
                raise ValueError(message)
        elif present_options:
            message = f"sampling.parameters[{index}] numeric options apply only to the integer transform."
            raise ValueError(message)
        if transform == "softmax":
            if not isinstance(parameter.get("group"), str) or not parameter["group"]:
                message = f"sampling.parameters[{index}] softmax transform requires a non-empty group."
                raise ValueError(message)
        elif "group" in parameter:
            message = f"sampling.parameters[{index}] group applies only to the softmax transform."
            raise ValueError(message)
        parameters.append(parameter)
    if case_count and sampling["method"] == "sobol" and not raw_parameters:
        message = "Sobol sampling requires at least one configured parameter."
        raise ValueError(message)
    softmax_counts: dict[str, int] = {}
    for parameter in parameters:
        if parameter["transform"] == "softmax":
            group = str(parameter["group"])
            softmax_counts[group] = softmax_counts.get(group, 0) + 1
    if any(count < PAIR_PARAMETER_SIZE for count in softmax_counts.values()):
        message = "Each softmax sampling group must contain at least two parameters."
        raise ValueError(message)
    sampling["variation"] = variation
    sampling["parameters"] = parameters
    return sampling


def load_generation_config(path: Path | str) -> GenerationConfig:
    """
    Load, validate, and identity-bind one explicit simulation-profile YAML.

    The selected profile, rather than a filename, task, directory, or schedule,
    owns the template and scientific export-role contract.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        message = f"Generation configuration does not exist: {source_path}"
        raise FileNotFoundError(message)
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not load generation configuration: {source_path}"
        raise ValueError(message) from error
    values = _mapping(loaded, label="generation configuration")
    _exact_keys(values, required=set(_TOP_LEVEL_KEYS) - {"sampling"}, optional={"sampling"}, label="generation configuration")
    if values["schema_version"] != CONFIG_SCHEMA_VERSION:
        message = f"Generation configuration must declare schema_version={CONFIG_SCHEMA_VERSION}."
        raise ValueError(message)
    profile_id = values["simulation_profile"]
    if not isinstance(profile_id, str):
        message = "generation configuration simulation_profile must be text."
        raise TypeError(message)
    profile = profiles.get_profile(profile_id)

    case_indices, seed_base, overrides = _validate_cases(values["cases"])
    generator = _validate_generator(values["generator"])
    inputs = _validate_inputs(values["inputs"])
    _validate_profile_inputs(inputs, profile=profile)
    exports = _validate_exports(values["exports"], profile=profile)
    values["exports"] = exports
    values["execution"] = _validate_execution(values["execution"])
    values["cluster"] = _load_cluster_config(values["cluster"])
    values["sampling"] = _validate_sampling(values.get("sampling"), case_count=len(case_indices), generator=generator, inputs=inputs)
    values["generator"] = generator
    values["inputs"] = inputs
    values["cases"] = {"indices": list(case_indices), "seed_base": seed_base, "overrides": {str(key): value for key, value in overrides.items()}}

    identity_payload = copy.deepcopy(values)
    identity_payload["profile_contract"] = {
        "simulation_profile": profile.id,
        "template_relative_path": profile.template_relative_path,
        "template_sha256": profile.template_sha256,
        "available_learning_views": list(profile.available_learning_views),
        "airflow_source": profile.airflow_source,
    }
    identity_payload.pop("execution")
    identity_payload.pop("cluster")
    batch_identity = common.serialization.canonical_json_sha256(identity_payload)
    batch_id = f"{profile.id}-{batch_identity[:16]}"
    config = GenerationConfig(
        source_path=source_path,
        profile=profile,
        values=values,
        template_path=profile.template_path,
        template_sha256=profile.template_sha256,
        case_indices=case_indices,
        seed_base=seed_base,
        overrides=overrides,
        batch_identity=batch_identity,
        batch_id=batch_id,
    )

    from . import generation_sampling as sampling_service  # noqa: PLC0415 -- runtime import avoids a contract cycle

    sampled_values = sampling_service.sample_case_overrides(config)
    for case_index in case_indices:
        parameters, _, schedule_rows = sampling_service.resolve_case_values(
            config,
            case_index,
            sampled_values=sampled_values,
        )
        validate_generator_parameters(parameters)
        schedule_spec = inputs.get("schedule_file")
        if schedule_spec is not None and schedule_rows is not None:
            validate_schedule_rows(schedule_spec["columns"], schedule_rows, label=f"case {case_index} schedule")
    return config
