"""
generation_benchmark_config.py

Resolve the core benchmark suite and its immutable scientific case selections.

Responsibilities:
  - Validate suite and resource variant configuration
  - Own immutable suite, variant, and representative case contracts
  - Derive stable suite, case, and execution identities

Design principles:
  - Preserve authored schema and canonical digest inputs
  - Keep configuration validation independent of benchmark lifecycle
  - Reject unsafe repository references and scheduler options

This module does NOT:
  - Submit jobs or persist benchmark evidence
  - Interpret measurements or render reports
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
BENCHMARK_SCHEMA_VERSION: Final = 1
BENCHMARK_REPRESENTATIVE_CASE_ROLES: Final = ("nominal", "natural")
_BENCHMARK_VARIANT_COUNT: Final = 4
_RESERVED_SCHEDULER_OPTIONS: Final = (
    "--array",
    "--chdir",
    "--cpus-per-task",
    "--dependency",
    "--error",
    "--exclusive",
    "--export",
    "--job-name",
    "--licenses",
    "--nodelist",
    "--nodes",
    "--ntasks",
    "--ntasks-per-node",
    "--output",
    "--parsable",
    "--partition",
    "--reservation",
    "--time",
    "--wrap",
)


@dataclass(frozen=True, slots=True)
class CoreBenchmarkRepresentativeCase:
    """One deterministic scientific case reused across every core-count wave."""

    case_role: str
    case_index: int


@dataclass(frozen=True, slots=True)
class CoreBenchmarkVariant:
    """One resource-only core-count variant declared by a small YAML file."""

    source_path: Path
    variant_id: str
    cores_per_case: int


@dataclass(frozen=True, slots=True)
class CoreBenchmarkSuite:
    """One resolved benchmark suite sharing two deterministic scientific cases."""

    source_path: Path
    suite_name: str
    suite_digest: str
    case_campaign_path: Path
    case_campaign: config_service.CampaignConfig
    case_config: config_service.GenerationConfig
    representative_cases: tuple[CoreBenchmarkRepresentativeCase, ...]
    maximum_work_unit_attempts: int
    variants: tuple[CoreBenchmarkVariant, ...]
    cores_per_node: int
    partition: str | None
    wall_time: str | None
    scheduler_options: tuple[str, ...]
    production_campaign_path: Path
    production_cores_config_path: Path
    production_cores_key: str
    production_cores_per_case: int
    node_memory_limit_bytes: int | None = None
    node_scratch_limit_bytes: int | None = None

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
        """Return the fixed number of scientific cases measured in each wave."""
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
        variant: CoreBenchmarkVariant,
        case_position: int,
    ) -> str:
        """Return one resource-and-science work-unit identity."""
        representative = self.representative_case(case_position)
        return f"{self.execution_id(variant)}__{representative.case_role}"

    def canary_variant(self) -> CoreBenchmarkVariant:
        """Return the unique variant matching the production core setting."""
        matches = tuple(variant for variant in self.variants if variant.cores_per_case == self.production_cores_per_case)
        if len(matches) != 1:
            message = (
                "Core benchmark requires exactly one variant matching production "
                f"cores_per_case={self.production_cores_per_case}; found {len(matches)}."
            )
            raise ValueError(message)
        return matches[0]

    def resource_contract(self) -> dict[str, Any]:
        """Return the common site and scheduler contract for every variant."""
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
            "wall_time": self.wall_time,
            "scheduler_options": list(self.scheduler_options),
            "cases_per_measured_wave": self.representative_case_count,
            "maximum_concurrent_measured_runs": self.representative_case_count,
            "poll_interval_seconds": self.case_campaign.execution_values["submission"]["poll_interval_seconds"],
            "maximum_work_unit_attempts": self.maximum_work_unit_attempts,
            "node_memory_limit_bytes": self.node_memory_limit_bytes,
            "node_scratch_limit_bytes": self.node_scratch_limit_bytes,
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
        """Return both representative cases in stable authored order."""
        return [self.case_selection(case_position) for case_position in range(1, self.representative_case_count + 1)]

    def variant_wave_order(self) -> tuple[CoreBenchmarkVariant, ...]:
        """Return production cores first, then remaining core counts ascending."""
        production = self.canary_variant()
        remaining = tuple(variant for variant in sorted(self.variants, key=lambda item: item.cores_per_case) if variant != production)
        return (production, *remaining)


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
        if any(option == reserved or option.startswith(f"{reserved}=") for reserved in _RESERVED_SCHEDULER_OPTIONS):
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


def load_core_benchmark_suite(  # noqa: C901, PLR0912, PLR0915 -- centralized suite validation
    path: Path | str,
    *,
    require_executable: bool = True,
) -> CoreBenchmarkSuite:
    """
    Resolve the shared benchmark case and four resource-only variants.

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
    require_exact_keys(
        suite,
        {
            "schema_kind",
            "schema_version",
            "suite_name",
            "benchmark_mode",
            "representative_cases",
            "parallel_cases_per_variant",
            "variant_execution",
            "case_execution_within_variant",
            "retry",
            "resources",
            "production_interpretation",
            "variants",
        },
        label="core benchmark suite",
    )
    if suite["schema_kind"] != BENCHMARK_SUITE_SCHEMA_KIND or suite["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        message = f"Unsupported core benchmark suite schema: {source_path}"
        raise ValueError(message)
    suite_name = common.paths.validate_logical_name(
        suite["suite_name"],
        label="benchmark suite_name",
    )
    if suite["benchmark_mode"] != "core_selection":
        message = "benchmark.benchmark_mode must be 'core_selection'."
        raise ValueError(message)
    if suite["variant_execution"] != "sequential":
        message = "benchmark.variant_execution must be 'sequential'."
        raise ValueError(message)
    if suite["case_execution_within_variant"] != "concurrent":
        message = "benchmark.case_execution_within_variant must be 'concurrent'."
        raise ValueError(message)
    parallel_cases = require_positive_integer(
        suite["parallel_cases_per_variant"],
        label="benchmark.parallel_cases_per_variant",
    )
    representative_values = suite["representative_cases"]
    if not isinstance(representative_values, list) or len(representative_values) != len(BENCHMARK_REPRESENTATIVE_CASE_ROLES):
        message = "Core benchmarking requires exactly two representative cases."
        raise ValueError(message)
    if parallel_cases != len(representative_values):
        message = "Core benchmarking must run both representative cases concurrently within each variant."
        raise ValueError(message)

    parsed_cases: list[tuple[str, Path, str, str, int]] = []
    for index, raw_case in enumerate(representative_values):
        case = require_mapping(raw_case, label=f"benchmark.representative_cases[{index}]")
        require_exact_keys(
            case,
            {
                "case_role",
                "campaign_config",
                "material_family",
                "sampling_regime",
                "case_index",
            },
            label=f"benchmark.representative_cases[{index}]",
        )
        case_role = common.paths.validate_logical_name(
            case["case_role"],
            label=f"benchmark.representative_cases[{index}].case_role",
        )
        campaign_path = resolve_reference_path(
            case["campaign_config"],
            label=f"benchmark.representative_cases[{index}].campaign_config",
        )
        material_family = common.paths.validate_logical_name(
            case["material_family"],
            label=f"benchmark.representative_cases[{index}].material_family",
        )
        sampling_regime = common.paths.validate_logical_name(
            case["sampling_regime"],
            label=f"benchmark.representative_cases[{index}].sampling_regime",
        )
        case_index = require_positive_integer(
            case["case_index"],
            label=f"benchmark.representative_cases[{index}].case_index",
        )
        parsed_cases.append(
            (
                case_role,
                campaign_path,
                material_family,
                sampling_regime,
                case_index,
            )
        )
    roles = tuple(item[0] for item in parsed_cases)
    if roles != BENCHMARK_REPRESENTATIVE_CASE_ROLES:
        message = f"Core benchmark representative case roles must be authored as {list(BENCHMARK_REPRESENTATIVE_CASE_ROLES)}."
        raise ValueError(message)
    shared_selection = {(item[1], item[2], item[3]) for item in parsed_cases}
    if len(shared_selection) != 1:
        message = "Core benchmark representative cases must share one pilot campaign batch."
        raise ValueError(message)
    campaign_path, material_family, sampling_regime = next(iter(shared_selection))
    campaign = config_service.load_campaign_config(
        campaign_path,
        require_executable=require_executable,
    )
    if campaign.campaign_purpose != config_service.PILOT_CAMPAIGN_PURPOSE or campaign.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Core benchmarking requires one transient pilot-check campaign."
        raise ValueError(message)
    if campaign.dataset_packages:
        message = "The benchmark case campaign must declare no Dataset packages."
        raise ValueError(message)
    case_config = campaign.require_batch(
        material_family=material_family,
        sampling_regime=sampling_regime,
    )
    representative_cases = tuple(
        CoreBenchmarkRepresentativeCase(case_role=case_role, case_index=case_index)
        for case_role, _path, _material, _regime, case_index in parsed_cases
    )
    if len({representative.case_index for representative in representative_cases}) != len(representative_cases):
        message = "Core benchmark representative cases must use distinct case indices."
        raise ValueError(message)
    expected_pilot_kinds = {"nominal": "nominal_reference", "natural": "natural_pilot"}
    for representative in representative_cases:
        assignment = case_config.case_assignment(representative.case_index)
        if assignment.get("pilot_case_kind") != expected_pilot_kinds[representative.case_role]:
            message = (
                f"Benchmark case role {representative.case_role!r} does not select the required "
                f"{expected_pilot_kinds[representative.case_role]!r} pilot case."
            )
            raise ValueError(message)

    retry = require_mapping(suite["retry"], label="benchmark.retry")
    require_exact_keys(
        retry,
        {"maximum_work_unit_attempts"},
        label="benchmark.retry",
    )
    maximum_work_unit_attempts = require_positive_integer(
        retry["maximum_work_unit_attempts"],
        label="benchmark.retry.maximum_work_unit_attempts",
    )
    resources = require_mapping(suite["resources"], label="benchmark.resources")
    require_exact_keys(
        resources,
        {"partition", "wall_time", "scheduler_options"},
        label="benchmark.resources",
    )
    execution = campaign.execution_values
    cluster = execution["cluster"]
    site = execution["site"]
    partition = _optional_text(
        resources["partition"],
        label="benchmark.resources.partition",
    )
    if partition is None:
        partition = _optional_text(cluster["partition"], label="execution.cluster.partition")
    wall_time = _optional_text(
        resources["wall_time"],
        label="benchmark.resources.wall_time",
    )
    if wall_time is None:
        wall_time = _optional_text(cluster["wall_time"], label="execution.cluster.wall_time")
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

    if site["scheduler"] != "slurm":
        message = "Core benchmarking requires the configured Slurm CPU site."
        raise ValueError(message)
    cores_per_node = require_positive_integer(
        cluster["cores_per_node"],
        label="execution.cluster.cores_per_node",
    )

    variant_values = suite["variants"]
    if not isinstance(variant_values, list) or len(variant_values) != _BENCHMARK_VARIANT_COUNT:
        message = "The maintained core benchmark suite must reference exactly four variants."
        raise ValueError(message)
    variants: list[CoreBenchmarkVariant] = []
    for index, reference in enumerate(variant_values):
        variant_path = resolve_reference_path(
            reference,
            label=f"benchmark.variants[{index}]",
        )
        raw = load_yaml_mapping(variant_path, label="core benchmark variant")
        require_exact_keys(
            raw,
            {
                "schema_kind",
                "schema_version",
                "suite_config",
                "variant_id",
                "cores_per_case",
            },
            label="core benchmark variant",
        )
        if raw["schema_kind"] != BENCHMARK_VARIANT_SCHEMA_KIND or raw["schema_version"] != BENCHMARK_SCHEMA_VERSION:
            message = f"Unsupported core benchmark variant schema: {variant_path}"
            raise ValueError(message)
        owner = resolve_reference_path(
            raw["suite_config"],
            label="benchmark variant suite_config",
        )
        if owner != source_path:
            message = f"Benchmark variant does not reference its owning suite: {variant_path}"
            raise ValueError(message)
        variant_id = common.paths.validate_logical_name(
            raw["variant_id"],
            label="benchmark variant_id",
        )
        cores = require_positive_integer(
            raw["cores_per_case"],
            label=f"benchmark variant {variant_id} cores_per_case",
        )
        if cores > cores_per_node:
            message = f"Benchmark variant {variant_id!r} requests {cores} cores on a {cores_per_node}-core node."
            raise ValueError(message)
        variants.append(
            CoreBenchmarkVariant(
                source_path=variant_path,
                variant_id=variant_id,
                cores_per_case=cores,
            )
        )
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
        "parallel_cases_per_variant": parallel_cases,
        "variant_execution": "sequential",
        "case_execution_within_variant": "concurrent",
        "retry": {
            "maximum_work_unit_attempts": maximum_work_unit_attempts,
        },
        "resources": {
            "partition": partition,
            "wall_time": wall_time,
            "scheduler_options": list(scheduler_options),
            "cores_per_node": cores_per_node,
            "site": site,
        },
        "variants": [
            {
                "source_path": repository_relative(variant.source_path),
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
            }
            for variant in variants
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
        maximum_work_unit_attempts=maximum_work_unit_attempts,
        variants=tuple(variants),
        cores_per_node=cores_per_node,
        partition=partition,
        wall_time=wall_time,
        scheduler_options=scheduler_options,
        production_campaign_path=production_campaign_path,
        production_cores_config_path=production_cores_config_path,
        production_cores_key=production_cores_key,
        production_cores_per_case=authored_cores,
        node_memory_limit_bytes=None,
        node_scratch_limit_bytes=None,
    )


def inspect_core_benchmark(
    path: Path | str,
    *,
    require_executable: bool = False,
) -> dict[str, Any]:
    """
    Return the compact two-case wave contract without materializing inputs.

    Parameters
    ----------
    path : Path | str
        Repository-owned suite configuration path.
    require_executable : bool, optional
        Require executable campaign inputs while loading the suite.

    Returns
    -------
    dict[str, Any]
        Ordered wave and case identities for inspection.

    """
    suite = load_core_benchmark_suite(path, require_executable=require_executable)
    wave_order = suite.variant_wave_order()
    return {
        "schema_kind": "generation_core_scaling_benchmark_inspection",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": repository_relative(suite.source_path),
        "benchmark_mode": "core_selection",
        "representative_cases": suite.case_selections(),
        "parallel_cases_per_variant": suite.representative_case_count,
        "variant_execution": "sequential",
        "case_execution_within_variant": "concurrent",
        "required_successful_measurements": (len(suite.variants) * suite.representative_case_count),
        "resource_contract": suite.resource_contract(),
        "canary_wave": {
            "variant_id": wave_order[0].variant_id,
            "cores_per_case": wave_order[0].cores_per_case,
            "case_roles": [item.case_role for item in suite.representative_cases],
            "included_in_final_measurements": True,
        },
        "variant_waves": [
            {
                "wave_position": position,
                "variant_id": variant.variant_id,
                "source_path": repository_relative(variant.source_path),
                "cores_per_case": variant.cores_per_case,
                "execution_id": suite.execution_id(variant),
            }
            for position, variant in enumerate(wave_order, start=1)
        ],
        "scientific_inputs_materialized": False,
        "dataset_membership": "none",
    }
