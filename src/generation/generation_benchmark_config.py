"""Resolve the core benchmark suite and its immutable scientific case selections.

Responsibilities:
  - Validate suite, sweep, and resource variant configuration
  - Own immutable suite, sweep, variant, and representative case contracts
  - Derive stable suite, case, execution, and work-unit identities
  - Derive every variant's node-job concurrency and memory through the execution contract's resolver

Design principles:
  - Preserve authored schema and canonical digest inputs
  - Keep configuration validation independent of benchmark lifecycle
  - Node-job resources come from the one execution resolver, never from a second formula
  - Reject unsafe repository references and scheduler options

This module does NOT:
  - Submit jobs or persist benchmark evidence
  - Interpret measurements or render reports

Notes:
  A sweep is one node job (in one node-job slot) that runs every variant in its own order on one node. The
  variant groups of a sweep never overlap, so each core-count comparison inside a sweep is measured on the same node;
  two sweeps in opposite orders are replicates that balance drift. Every variant tries the same candidate pool in
  case-index order until ``target_successful_cases`` succeeded; a failed candidate is replaced by the next one.

"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Mapping

BENCHMARK_SUITE_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_suite"
BENCHMARK_VARIANT_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_variant"
BENCHMARK_SCHEMA_VERSION: Final = 3
SWEEP_ORDERS: Final = ("ascending", "descending")
_MAXIMUM_POOL_CASES: Final = 64
_MAXIMUM_SWEEPS: Final = 2  # one sweep per node-job slot (job names gen-seat-1, gen-seat-2); the per-user cap of node jobs is a cluster-share limit
_MINIMUM_VARIANTS: Final = 2
_RESERVED_SCHEDULER_OPTIONS: Final = (
    "--array",
    "--chdir",
    "--comment",
    "--cpus-per-task",
    "--dependency",
    "--error",
    "--exclude",
    "--exclusive",
    "--export",
    "--gpus",
    "--gres",
    "--job-name",
    "--licenses",
    "--mem",
    "--nodelist",
    "--nodes",
    "--no-requeue",
    "--ntasks",
    "--ntasks-per-node",
    "--output",
    "--parsable",
    "--partition",
    "--requeue",
    "--reservation",
    "--signal",
    "--time",
    "--wrap",
)


@dataclass(frozen=True, slots=True)
class CoreBenchmarkRepresentativeCase:
    """One deterministic scientific case reused across every core-count variant."""

    case_role: str
    case_index: int


@dataclass(frozen=True, slots=True)
class CoreBenchmarkVariant:
    """One resource-only core-count variant declared inline by its suite."""

    variant_id: str
    cores_per_case: int


@dataclass(frozen=True, slots=True)
class CoreBenchmarkSweep:
    """One node job that measures every variant, in ascending or descending core order, on one node."""

    sweep_id: str
    order: str


@dataclass(frozen=True, slots=True)
class CoreBenchmarkVariantResources:
    """Node-job resources of one variant, derived by the execution contract's resolver."""

    concurrent_cases: int
    memory_per_case: str
    node_job_memory: str


@dataclass(frozen=True, slots=True)
class CoreBenchmarkSuite:
    """One resolved benchmark suite sharing deterministic scientific cases across variants and sweeps."""

    source_path: Path
    suite_name: str
    suite_digest: str
    case_campaign_path: Path
    case_campaign: config_service.CampaignConfig
    case_config: config_service.GenerationConfig
    representative_cases: tuple[CoreBenchmarkRepresentativeCase, ...]
    target_successful_cases: int
    maximum_variant_attempts: int
    variants: tuple[CoreBenchmarkVariant, ...]
    variant_resources: tuple[CoreBenchmarkVariantResources, ...]
    sweeps: tuple[CoreBenchmarkSweep, ...]
    cores_per_node: int
    partition: str
    sweep_wall_time: str
    node_memory_limit: str
    scheduler_options: tuple[str, ...]
    production_campaign_path: Path
    production_cores_config_path: Path
    production_cores_key: str
    production_cores_per_case: int

    def variant(self, variant_id: str) -> CoreBenchmarkVariant:
        """Return one configured variant by its stable identifier."""
        safe_id = common.paths.validate_logical_name(
            variant_id,
            label="benchmark variant_id",
        )
        matches = tuple(item for item in self.variants if item.variant_id == safe_id)
        if len(matches) != 1:
            available = ", ".join(item.variant_id for item in self.variants)
            message = f"Unknown benchmark variant {variant_id!r}; available: {available}."
            raise ValueError(message)
        return matches[0]

    def sweep(self, sweep_id: str) -> CoreBenchmarkSweep:
        """Return one configured sweep by its stable identifier."""
        matches = tuple(item for item in self.sweeps if item.sweep_id == sweep_id)
        if len(matches) != 1:
            available = ", ".join(item.sweep_id for item in self.sweeps)
            message = f"Unknown benchmark sweep {sweep_id!r}; available: {available}."
            raise ValueError(message)
        return matches[0]

    def sweep_variants(self, sweep: CoreBenchmarkSweep) -> tuple[CoreBenchmarkVariant, ...]:
        """Return the variants in the measurement order of one sweep."""
        ordered = tuple(sorted(self.variants, key=lambda item: item.cores_per_case))
        return ordered if sweep.order == "ascending" else tuple(reversed(ordered))

    def resources(self, variant: CoreBenchmarkVariant) -> CoreBenchmarkVariantResources:
        """Return the derived node-job resources of one variant."""
        return self.variant_resources[self.variants.index(variant)]

    def sweep_node_job_memory(self) -> str:
        """Return the node-job memory every sweep requests: the largest variant demand, because one job runs all."""
        return max(
            (item.node_job_memory for item in self.variant_resources),
            key=lambda text: config_service.parse_slurm_memory_mib(text, label="benchmark node_job_memory"),
        )

    def execution_id(self, variant: CoreBenchmarkVariant) -> str:
        """Return one core-setting execution identity separate from science."""
        digest = common.serialization.canonical_json_sha256(
            {
                "schema_kind": BENCHMARK_VARIANT_SCHEMA_KIND,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "suite_digest": self.suite_digest,
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
                "resource_contract": self.resource_contract(),
            }
        )
        return f"{variant.variant_id}__{digest[:16]}"

    @property
    def representative_case_count(self) -> int:
        """Return the fixed number of scientific cases measured in each variant."""
        return len(self.representative_cases)

    def representative_case(self, case_position: int) -> CoreBenchmarkRepresentativeCase:
        """Return one representative case by its one-based stable position."""
        if case_position < 1 or case_position > self.representative_case_count:
            message = f"Benchmark representative case position must be in [1, {self.representative_case_count}], got {case_position}."
            raise ValueError(message)
        return self.representative_cases[case_position - 1]

    def case_position(self, case_role: str) -> int:
        """Return the one-based position for an exact representative-case role."""
        safe_role = common.paths.validate_logical_name(
            case_role,
            label="benchmark representative case_role",
        )
        matches = [position for position, representative in enumerate(self.representative_cases, start=1) if representative.case_role == safe_role]
        if len(matches) != 1:
            available = ", ".join(item.case_role for item in self.representative_cases)
            message = f"Unknown benchmark case role {case_role!r}; available: {available}."
            raise ValueError(message)
        return matches[0]

    def work_unit_id(
        self,
        sweep: CoreBenchmarkSweep,
        variant: CoreBenchmarkVariant,
        variant_attempt: int,
        case_position: int,
    ) -> str:
        """Return one sweep-, attempt-, resource-, and science-scoped work-unit identity."""
        if isinstance(variant_attempt, bool) or not isinstance(variant_attempt, int) or variant_attempt < 1:
            message = f"Benchmark variant attempt must be a positive integer, got {variant_attempt!r}."
            raise ValueError(message)
        representative = self.representative_case(case_position)
        return f"{sweep.sweep_id}__a{variant_attempt:02d}__{self.execution_id(variant)}__{representative.case_role}"

    def resource_contract(self) -> dict[str, Any]:
        """Return the common site, scheduler, and node-job contract for every variant and sweep."""
        site = self.case_campaign.execution_values["site"]
        return {
            "cpu_host": site["cpu_host"],
            "scheduler": site["scheduler"],
            "partition": self.partition,
            "cores_per_node": self.cores_per_node,
            "python_module": site["python_module"],
            "comsol_module": site["comsol_module"],
            "python_executable": site["python_executable"],
            "comsol_executable": site["comsol_executable"],
            "scheduler_options": list(self.scheduler_options),
            "case_pool_size": self.representative_case_count,
            "target_successful_cases": self.target_successful_cases,
            "poll_interval_seconds": self.case_campaign.execution_values["submission"]["poll_interval_seconds"],
            "maximum_variant_attempts": self.maximum_variant_attempts,
            "sweeps": [{"sweep_id": sweep.sweep_id, "order": sweep.order} for sweep in self.sweeps],
            "variant_node_jobs": [
                {
                    "variant_id": variant.variant_id,
                    "cores_per_case": variant.cores_per_case,
                    "concurrent_cases": self.resources(variant).concurrent_cases,
                    "memory_per_case": self.resources(variant).memory_per_case,
                    "node_job_memory": self.resources(variant).node_job_memory,
                }
                for variant in self.variants
            ],
            "node_jobs": len(self.sweeps),
            "node_job_memory": self.sweep_node_job_memory(),
            "node_job_wall_time": self.sweep_wall_time,
            "node_memory_limit": self.node_memory_limit,
        }

    def case_selection(self, case_position: int) -> dict[str, Any]:
        """Return one compact deterministic representative-case identity."""
        representative = self.representative_case(case_position)
        case_index = representative.case_index
        assignment = self.case_config.case_assignment(case_index)
        seed = self.case_config.case_seed(case_index)
        return {
            "case_role": representative.case_role,
            "campaign_config": repository_relative(self.case_campaign_path),
            "campaign_id": self.case_campaign.campaign_id,
            "batch_name": self.case_config.batch_name,
            "batch_id": self.case_config.batch_id,
            "simulation_profile": self.case_config.profile.id,
            "material_family": self.case_config.material_family,
            "sampling_regime": self.case_config.sampling_regime,
            "case_index": case_index,
            "case_id": self.case_config.case_id(case_index),
            "case_seed": seed,
            "assignment": assignment,
            "scientific_config_digest": self.case_config.scientific_config_digest,
            "case_input_config_digest": self.case_config.case_input_config_digest,
            "export_contract_sha256": common.serialization.canonical_json_sha256(self.case_config.scientific_values["output_contract"]),
            "execution_config_digest": common.serialization.canonical_json_sha256(self.case_config.execution_values),
            "template": {
                "relative_path": self.case_config.template_relative_path,
                "sha256": self.case_config.template_sha256,
            },
            "selection_digest": common.serialization.canonical_json_sha256(
                {
                    "case_role": representative.case_role,
                    "scientific_config_digest": self.case_config.scientific_config_digest,
                    "case_input_config_digest": self.case_config.case_input_config_digest,
                    "case_index": case_index,
                    "case_seed": seed,
                    "assignment": assignment,
                    "template_sha256": self.case_config.template_sha256,
                }
            ),
        }

    def case_selections(self) -> list[dict[str, Any]]:
        """Return every representative case in stable authored order."""
        return [self.case_selection(case_position) for case_position in range(1, self.representative_case_count + 1)]


def require_mapping(value: object, *, label: str) -> dict[str, Any]:
    """Return one string-keyed mapping or fail clearly."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return dict(value)


def require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    """Require one closed configuration schema."""
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise ValueError(message)


def load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required YAML mapping."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not load {label}: {path}"
        raise ValueError(message) from error
    return require_mapping(value, label=label)


def repository_relative(path: Path) -> str:
    """Return one stable repository-relative path."""
    repository = common.paths.get_project_root().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError as error:
        message = f"Benchmark configuration escapes the repository: {resolved}"
        raise ValueError(message) from error


def resolve_reference_path(value: object, *, label: str) -> Path:
    """Resolve one safe repository-relative benchmark reference."""
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{label} must be non-empty repository-relative text."
        raise TypeError(message)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        message = f"{label} must not be absolute or contain traversal: {value!r}."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    path = (repository / relative).resolve()
    if not path.is_relative_to(repository) or not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    return path


def require_positive_integer(value: object, *, label: str) -> int:
    """Return one positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{label} must be an integer >= 1, got {value!r}."
        raise ValueError(message)
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    """Return safe optional scheduler text."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or value.strip() != value or any(character in value for character in "\r\n\t"):
        message = f"{label} must be null or safe non-empty text."
        raise ValueError(message)
    return value


def _scheduler_options(value: object) -> tuple[str, ...]:
    """Validate benchmark-owned optional scheduler constraints."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        message = "benchmark.resources.scheduler_options must be a list of strings."
        raise TypeError(message)
    options = tuple(value)
    if len(options) != len(set(options)):
        message = "benchmark.resources.scheduler_options must be duplicate-free."
        raise ValueError(message)
    for option in options:
        if not option.startswith("--") or any(character in option for character in "\r\n\t"):
            message = f"Unsafe benchmark scheduler option: {option!r}."
            raise ValueError(message)
        name = option.split("=", maxsplit=1)[0]
        if any(name == reserved or name.startswith(reserved) for reserved in _RESERVED_SCHEDULER_OPTIONS):
            message = f"Benchmark scheduler option is owned by the launcher: {option!r}."
            raise ValueError(message)
    return options


def _production_like_benchmark_config(
    config: config_service.GenerationConfig,
) -> config_service.GenerationConfig:
    """Return the benchmark execution view with compact Production retention."""
    execution = copy.deepcopy(config.execution_values)
    execution["retention_policy"] = "compact"
    return replace(config, execution_values=execution)


def _parse_sweeps(value: object) -> tuple[CoreBenchmarkSweep, ...]:
    """Validate one to two sweeps with unique identifiers and distinct orders."""
    if not isinstance(value, list) or not 1 <= len(value) <= _MAXIMUM_SWEEPS:
        message = f"benchmark.sweeps must list 1 to {_MAXIMUM_SWEEPS} sweeps (one per node-job slot, within the per-user cap of node jobs)."
        raise ValueError(message)
    sweeps: list[CoreBenchmarkSweep] = []
    for index, raw_sweep in enumerate(value):
        raw = require_mapping(raw_sweep, label=f"benchmark.sweeps[{index}]")
        require_exact_keys(raw, {"sweep_id", "order"}, label=f"benchmark.sweeps[{index}]")
        sweep_id = common.paths.validate_logical_name(raw["sweep_id"], label=f"benchmark.sweeps[{index}].sweep_id")
        if raw["order"] not in SWEEP_ORDERS:
            message = f"benchmark.sweeps[{index}].order must be one of {SWEEP_ORDERS}."
            raise ValueError(message)
        sweeps.append(CoreBenchmarkSweep(sweep_id=sweep_id, order=str(raw["order"])))
    if len({sweep.sweep_id for sweep in sweeps}) != len(sweeps) or len({sweep.order for sweep in sweeps}) != len(sweeps):
        message = "Benchmark sweeps require unique identifiers and distinct orders."
        raise ValueError(message)
    return tuple(sweeps)


def _variant_resources(
    variant: CoreBenchmarkVariant,
    *,
    execution_config_path: Path,
) -> CoreBenchmarkVariantResources:
    """Derive one variant's node-job concurrency and memory through the execution resolver."""
    try:
        cluster = config_service.resolve_execution_values(
            execution_config_path,
            execution_overrides={"cores_per_case": variant.cores_per_case},
        )["cluster"]
    except config_service.GenerationConfigError as error:
        message = f"Benchmark variant {variant.variant_id!r} does not fit one node job: {error}"
        raise ValueError(message) from error
    return CoreBenchmarkVariantResources(
        concurrent_cases=int(cluster["concurrent_cases_per_node"]),
        memory_per_case=str(cluster["memory_per_case"]),
        node_job_memory=str(cluster["node_job_memory"]),
    )


def load_core_benchmark_suite(  # noqa: C901, PLR0912, PLR0915 -- centralized suite validation
    path: Path | str,
    *,
    require_executable: bool = True,
) -> CoreBenchmarkSuite:
    """
    Resolve the shared benchmark cases, the resource-only variants, and the sweeps.

    Parameters
    ----------
    path : Path | str
        Repository-owned suite configuration path.
    require_executable : bool, optional
        Require executable campaign inputs while loading the suite.

    Returns
    -------
    CoreBenchmarkSuite
        Validated immutable suite with stable identity digests.

    """
    source_path = Path(path).expanduser().resolve()
    suite = load_yaml_mapping(source_path, label="core benchmark suite")
    if suite.get("schema_kind") != BENCHMARK_SUITE_SCHEMA_KIND or suite.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        message = (
            f"Unsupported core benchmark suite schema: {source_path}; schema_version {BENCHMARK_SCHEMA_VERSION} "
            "(node-job sweeps) is required and no earlier version is read."
        )
        raise ValueError(message)
    require_exact_keys(
        suite,
        {
            "schema_kind",
            "schema_version",
            "suite_name",
            "benchmark_mode",
            "case_pool",
            "target_successful_cases",
            "sweeps",
            "retry",
            "resources",
            "production_interpretation",
            "variants",
        },
        label="core benchmark suite",
    )
    suite_name = common.paths.validate_logical_name(
        suite["suite_name"],
        label="benchmark suite_name",
    )
    if suite["benchmark_mode"] != "core_selection":
        message = "benchmark.benchmark_mode must be 'core_selection'."
        raise ValueError(message)
    pool = require_mapping(suite["case_pool"], label="benchmark.case_pool")
    require_exact_keys(
        pool,
        {"campaign_config", "material_family", "sampling_regime", "first_case_index", "case_count"},
        label="benchmark.case_pool",
    )
    campaign_path = resolve_reference_path(pool["campaign_config"], label="benchmark.case_pool.campaign_config")
    material_family = common.paths.validate_logical_name(pool["material_family"], label="benchmark.case_pool.material_family")
    sampling_regime = common.paths.validate_logical_name(pool["sampling_regime"], label="benchmark.case_pool.sampling_regime")
    first_case_index = require_positive_integer(pool["first_case_index"], label="benchmark.case_pool.first_case_index")
    pool_size = require_positive_integer(pool["case_count"], label="benchmark.case_pool.case_count")
    if pool_size > _MAXIMUM_POOL_CASES:
        message = f"benchmark.case_pool.case_count must be at most {_MAXIMUM_POOL_CASES}, got {pool_size}."
        raise ValueError(message)
    target_successful_cases = require_positive_integer(suite["target_successful_cases"], label="benchmark.target_successful_cases")
    if target_successful_cases > pool_size:
        message = f"benchmark.target_successful_cases ({target_successful_cases}) must not exceed the case pool ({pool_size})."
        raise ValueError(message)
    pool_indices = range(first_case_index, first_case_index + pool_size)
    role_width = max(2, len(str(pool_indices[-1])))
    campaign = config_service.load_campaign_config(
        campaign_path,
        require_executable=require_executable,
    )
    if campaign.campaign_purpose != "source_generation" or campaign.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Core benchmarking requires one transient source-generation campaign."
        raise ValueError(message)
    if campaign.dataset_packages:
        message = "The benchmark case campaign must declare no Dataset packages."
        raise ValueError(message)
    case_config = campaign.require_batch(
        material_family=material_family,
        sampling_regime=sampling_regime,
    )
    # The pool in trial order: every variant tries these candidates in case-index order until its target succeeded.
    representative_cases = tuple(
        CoreBenchmarkRepresentativeCase(case_role=f"case_{case_index:0{role_width}d}", case_index=case_index) for case_index in pool_indices
    )
    for representative in representative_cases:
        case_config.case_assignment(representative.case_index)

    sweeps = _parse_sweeps(suite["sweeps"])
    retry = require_mapping(suite["retry"], label="benchmark.retry")
    require_exact_keys(
        retry,
        {"maximum_variant_attempts"},
        label="benchmark.retry",
    )
    maximum_variant_attempts = require_positive_integer(
        retry["maximum_variant_attempts"],
        label="benchmark.retry.maximum_variant_attempts",
    )
    resources = require_mapping(suite["resources"], label="benchmark.resources")
    require_exact_keys(
        resources,
        {"partition", "sweep_wall_time", "scheduler_options"},
        label="benchmark.resources",
    )
    execution = campaign.execution_values
    cluster = execution["cluster"]
    site = execution["site"]
    partition = _optional_text(
        resources["partition"],
        label="benchmark.resources.partition",
    ) or str(cluster["partition"])
    sweep_wall_time = _optional_text(
        resources["sweep_wall_time"],
        label="benchmark.resources.sweep_wall_time",
    )
    if sweep_wall_time is None:
        message = "benchmark.resources.sweep_wall_time must be the explicit wall time of one sweep node job."
        raise ValueError(message)
    sweep_seconds = config_service.parse_wall_time_seconds(sweep_wall_time, label="benchmark.resources.sweep_wall_time")
    if sweep_seconds > config_service.parse_wall_time_seconds("24:00:00", label="maximum node job wall time"):
        message = "benchmark.resources.sweep_wall_time must not exceed 24:00:00."
        raise ValueError(message)
    scheduler_options = _scheduler_options(resources["scheduler_options"])

    production = require_mapping(
        suite["production_interpretation"],
        label="benchmark.production_interpretation",
    )
    require_exact_keys(
        production,
        {"campaign_config", "cores_config", "cores_key"},
        label="benchmark.production_interpretation",
    )
    production_campaign_path = resolve_reference_path(
        production["campaign_config"],
        label="benchmark.production_interpretation.campaign_config",
    )
    production_cores_config_path = resolve_reference_path(
        production["cores_config"],
        label="benchmark.production_interpretation.cores_config",
    )
    production_cores_key = production["cores_key"]
    if production_cores_key != "cluster.cores_per_case":
        message = "benchmark.production_interpretation.cores_key must identify cluster.cores_per_case."
        raise ValueError(message)
    production_campaign = config_service.load_campaign_config(
        production_campaign_path,
        require_executable=False,
    )
    production_execution = load_yaml_mapping(
        production_cores_config_path,
        label="benchmark production execution config",
    )
    authored_cluster = require_mapping(
        production_execution.get("cluster"),
        label="benchmark production execution cluster",
    )
    authored_cores = require_positive_integer(
        authored_cluster.get("cores_per_case"),
        label="benchmark production cores_per_case",
    )
    if authored_cores != production_campaign.execution_values["cluster"]["cores_per_case"]:
        message = "Benchmark production cores owner disagrees with the production campaign."
        raise ValueError(message)
    if production_campaign.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Core benchmark production interpretation requires a transient campaign."
        raise ValueError(message)
    if production_campaign.execution_values["retention_policy"] != "compact":
        message = "Core benchmark production interpretation requires compact retention."
        raise ValueError(message)
    if common.serialization.canonical_json_sha256(production_campaign.execution_values) != common.serialization.canonical_json_sha256(execution):
        message = "The benchmark case campaign and the production campaign must use the same execution contract."
        raise ValueError(message)

    if site["scheduler"] != "slurm":
        message = "Core benchmarking requires the configured Slurm CPU site."
        raise ValueError(message)
    cores_per_node = require_positive_integer(
        cluster["cores_per_node"],
        label="execution.cluster.cores_per_node",
    )

    variant_values = suite["variants"]
    if not isinstance(variant_values, list) or len(variant_values) < _MINIMUM_VARIANTS:
        message = f"The core benchmark suite must declare at least {_MINIMUM_VARIANTS} variants."
        raise ValueError(message)
    variants: list[CoreBenchmarkVariant] = []
    for index, raw_variant in enumerate(variant_values):
        raw = require_mapping(raw_variant, label=f"benchmark.variants[{index}]")
        require_exact_keys(
            raw,
            {"variant_id", "cores_per_case"},
            label=f"benchmark.variants[{index}]",
        )
        variant_id = common.paths.validate_logical_name(
            raw["variant_id"],
            label="benchmark variant_id",
        )
        cores = require_positive_integer(
            raw["cores_per_case"],
            label=f"benchmark variant {variant_id} cores_per_case",
        )
        if cores > cores_per_node or cores_per_node % cores != 0:
            message = f"Benchmark variant {variant_id!r} requests {cores} cores, which must divide the {cores_per_node}-core node."
            raise ValueError(message)
        variants.append(CoreBenchmarkVariant(variant_id=variant_id, cores_per_case=cores))
    ids = [variant.variant_id for variant in variants]
    core_counts = [variant.cores_per_case for variant in variants]
    if len(ids) != len(set(ids)) or len(core_counts) != len(set(core_counts)):
        message = "Core benchmark variants require distinct IDs and cores_per_case values."
        raise ValueError(message)
    if core_counts != sorted(core_counts):
        message = "Core benchmark variants must be authored in increasing cores_per_case order."
        raise ValueError(message)
    matching_production_variants = [variant for variant in variants if variant.cores_per_case == authored_cores]
    if len(matching_production_variants) != 1:
        message = (
            "Core benchmark requires exactly one variant matching production "
            f"cores_per_case={authored_cores}; found {len(matching_production_variants)}."
        )
        raise ValueError(message)
    variant_resources = tuple(_variant_resources(variant, execution_config_path=production_cores_config_path) for variant in variants)
    digest_payload = {
        "schema_kind": BENCHMARK_SUITE_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite_name,
        "benchmark_mode": "core_selection",
        "representative_cases": [
            {
                "case_role": representative.case_role,
                "campaign_config": repository_relative(campaign_path),
                "campaign_id": campaign.campaign_id,
                "batch_id": case_config.batch_id,
                "scientific_config_digest": case_config.scientific_config_digest,
                "case_input_config_digest": case_config.case_input_config_digest,
                "case_index": representative.case_index,
                "case_seed": case_config.case_seed(representative.case_index),
                "assignment": case_config.case_assignment(representative.case_index),
                "template_sha256": case_config.template_sha256,
            }
            for representative in representative_cases
        ],
        "target_successful_cases": target_successful_cases,
        "sweeps": [{"sweep_id": sweep.sweep_id, "order": sweep.order} for sweep in sweeps],
        "retry": {
            "maximum_variant_attempts": maximum_variant_attempts,
        },
        "resources": {
            "partition": partition,
            "sweep_wall_time": sweep_wall_time,
            "scheduler_options": list(scheduler_options),
            "cores_per_node": cores_per_node,
            "site": site,
        },
        "variants": [
            {
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
                "concurrent_cases": resources_item.concurrent_cases,
                "memory_per_case": resources_item.memory_per_case,
                "node_job_memory": resources_item.node_job_memory,
            }
            for variant, resources_item in zip(variants, variant_resources, strict=True)
        ],
    }
    return CoreBenchmarkSuite(
        source_path=source_path,
        suite_name=suite_name,
        suite_digest=common.serialization.canonical_json_sha256(digest_payload),
        case_campaign_path=campaign_path,
        case_campaign=campaign,
        case_config=_production_like_benchmark_config(case_config),
        representative_cases=representative_cases,
        target_successful_cases=target_successful_cases,
        maximum_variant_attempts=maximum_variant_attempts,
        variants=tuple(variants),
        variant_resources=variant_resources,
        sweeps=sweeps,
        cores_per_node=cores_per_node,
        partition=partition,
        sweep_wall_time=sweep_wall_time,
        node_memory_limit=str(cluster["node_memory_limit"]),
        scheduler_options=scheduler_options,
        production_campaign_path=production_campaign_path,
        production_cores_config_path=production_cores_config_path,
        production_cores_key=production_cores_key,
        production_cores_per_case=authored_cores,
    )


def inspect_core_benchmark(
    path: Path | str,
    *,
    require_executable: bool = False,
) -> dict[str, Any]:
    """
    Return the compact sweep and variant contract without materializing inputs.

    Parameters
    ----------
    path : Path | str
        Repository-owned suite configuration path.
    require_executable : bool, optional
        Require executable campaign inputs while loading the suite.

    Returns
    -------
    dict[str, Any]
        Ordered sweep, variant, and case identities for inspection.

    """
    suite = load_core_benchmark_suite(path, require_executable=require_executable)
    return {
        "schema_kind": "generation_core_scaling_benchmark_inspection",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": repository_relative(suite.source_path),
        "benchmark_mode": "core_selection",
        "representative_cases": suite.case_selections(),
        "required_successes_per_sweep": len(suite.variants) * suite.target_successful_cases,
        "resource_contract": suite.resource_contract(),
        "sweeps": [
            {
                "sweep_id": sweep.sweep_id,
                "order": sweep.order,
                "variant_groups": [
                    {
                        "group": group,
                        "variant_id": variant.variant_id,
                        "cores_per_case": variant.cores_per_case,
                        "concurrent_cases": suite.resources(variant).concurrent_cases,
                        "execution_id": suite.execution_id(variant),
                    }
                    for group, variant in enumerate(suite.sweep_variants(sweep))
                ],
            }
            for sweep in suite.sweeps
        ],
        "scientific_inputs_materialized": False,
        "dataset_membership": "none",
    }
