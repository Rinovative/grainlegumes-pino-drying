"""
generation_benchmark.py

Own the isolated transient COMSOL core-scaling benchmark lifecycle.
Responsibilities:
  - Resolve two shared scientific cases and exactly four resource-only variants
  - Plan, submit, resume, execute, summarize, and transfer benchmark evidence
  - Keep benchmark measurements outside canonical scientific case publication
Design principles:
  - CPU-materialized proofs bind every core wave to the same two exact inputs
  - Scientific identity and resource work-unit identity stay separate
  - Successful work-unit evidence is immutable and failed attempts are append-only
This module does NOT:
  - Define scientific parameters, publish training cases, or modify production resources
  - Run on the bare control-plane host or treat isolated throughput as contention proof
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import resource as resource_usage
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np
import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_preparation as preparation_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BENCHMARK_SUITE_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_suite"
BENCHMARK_VARIANT_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_variant"
BENCHMARK_RUN_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_run"
BENCHMARK_PROOF_SCHEMA_KIND: Final = "generation_core_scaling_case_proof"
BENCHMARK_RESULT_SCHEMA_KIND: Final = "generation_core_scaling_result"
BENCHMARK_SUMMARY_SCHEMA_KIND: Final = "generation_core_scaling_summary"
BENCHMARK_PREFLIGHT_SCHEMA_KIND: Final = "generation_core_scaling_preflight"
BENCHMARK_CLEANUP_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_cleanup"
BENCHMARK_CANCELLATION_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_cancellations"
BENCHMARK_SCHEMA_VERSION: Final = 1
BENCHMARK_FAMILY: Final = "core_scaling"
BENCHMARK_TRANSFER_FILENAME: Final = "transfer_complete.json"
BENCHMARK_LOCAL_CLEANUP_FILENAME: Final = "cpu_source_cleanup.json"
_MAX_RECENT_JOB_IDS: Final = 16
_BENCHMARK_VARIANT_COUNT: Final = 4
_BENCHMARK_REPRESENTATIVE_CASE_ROLES: Final = ("nominal", "natural")
_MAX_COMSOL_VERSION_EVIDENCE_BYTES: Final = 16 * 1024
_MAX_SLURM_JOB_NAME_LENGTH: Final = 48
_MAX_DIRTY_PATH_PREVIEW: Final = 5
_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_BENCHMARK_RUN_ID_PATTERN: Final = re.compile(r"core_scaling_transient__[0-9a-f]{16}")
_ACTIVE_SCHEDULER_STATES: Final = frozenset(
    {
        "COMPLETING",
        "CONFIGURING",
        "PENDING",
        "REQUEUED",
        "RESIZING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "SUSPENDED",
    }
)
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_WORK_UNIT_TIMING_FIELDS: Final = frozenset(
    {
        "scheduler_queue_seconds",
        "license_wait_seconds",
        "license_probe_seconds",
        "canonical_input_preparation_seconds",
        "comsol_process_seconds",
        "export_conversion_seconds",
        "publication_seconds",
        "total_controller_elapsed_seconds",
    }
)
_MAX_BENCHMARK_LOG_EXCERPT_BYTES: Final = 8 * 1024
_SUMMARY_METRIC_FIELDS: Final = (
    "suite_name",
    "suite_digest",
    "benchmark_mode",
    "representative_cases",
    "cases_per_variant",
    "required_successful_measurements",
    "cores_per_node",
    "variants",
    "fastest_single_case_cores",
    "lowest_core_hours_cores",
    "recommended_cores_per_case",
    "recommended_estimated_cases_per_node",
    "recommended_production",
    "recommendation_basis",
    "timing_contract",
    "resource_limits",
    "license_qualification",
    "production_interpretation",
    "production_configuration_modified",
    "dataset_membership",
    "canary_wave",
)
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
            "campaign_config": _repository_relative(self.case_campaign_path),
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


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _slurm_scheduler_start_time() -> str:
    """Return the scheduler-owned start timestamp for the current job."""
    raw = os.environ.get("SLURM_JOB_START_TIME")
    if raw is None or _JOB_ID_PATTERN.fullmatch(raw) is None or int(raw) < 1:
        message = "Benchmark timing requires Slurm's positive SLURM_JOB_START_TIME timestamp."
        raise RuntimeError(message)
    try:
        started = datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        message = "Benchmark SLURM_JOB_START_TIME is outside the supported timestamp range."
        raise RuntimeError(message) from error
    return started.isoformat()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    """Return one string-keyed mapping or fail clearly."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return dict(value)


def _exact_keys(
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


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required YAML mapping."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not load {label}: {path}"
        raise ValueError(message) from error
    return _mapping(value, label=label)


def _repository_relative(path: Path) -> str:
    """Return one stable repository-relative path."""
    repository = common.paths.get_project_root().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError as error:
        message = f"Benchmark configuration escapes the repository: {resolved}"
        raise ValueError(message) from error


def _reference_path(value: object, *, label: str) -> Path:
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


def _positive_integer(value: object, *, label: str) -> int:
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
    """Resolve the shared benchmark case and four resource-only variants."""
    source_path = Path(path).expanduser().resolve()
    suite = _load_yaml(source_path, label="core benchmark suite")
    _exact_keys(
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
    parallel_cases = _positive_integer(
        suite["parallel_cases_per_variant"],
        label="benchmark.parallel_cases_per_variant",
    )
    representative_values = suite["representative_cases"]
    if not isinstance(representative_values, list) or len(representative_values) != len(_BENCHMARK_REPRESENTATIVE_CASE_ROLES):
        message = "Core benchmarking requires exactly two representative cases."
        raise ValueError(message)
    if parallel_cases != len(representative_values):
        message = "Core benchmarking must run both representative cases concurrently within each variant."
        raise ValueError(message)

    parsed_cases: list[tuple[str, Path, str, str, int]] = []
    for index, raw_case in enumerate(representative_values):
        case = _mapping(raw_case, label=f"benchmark.representative_cases[{index}]")
        _exact_keys(
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
        campaign_path = _reference_path(
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
        case_index = _positive_integer(
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
    if roles != _BENCHMARK_REPRESENTATIVE_CASE_ROLES:
        message = f"Core benchmark representative case roles must be authored as {list(_BENCHMARK_REPRESENTATIVE_CASE_ROLES)}."
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

    retry = _mapping(suite["retry"], label="benchmark.retry")
    _exact_keys(
        retry,
        {"maximum_work_unit_attempts"},
        label="benchmark.retry",
    )
    maximum_work_unit_attempts = _positive_integer(
        retry["maximum_work_unit_attempts"],
        label="benchmark.retry.maximum_work_unit_attempts",
    )
    resources = _mapping(suite["resources"], label="benchmark.resources")
    _exact_keys(
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

    production = _mapping(
        suite["production_interpretation"],
        label="benchmark.production_interpretation",
    )
    _exact_keys(
        production,
        {"campaign_config", "cores_config", "cores_key"},
        label="benchmark.production_interpretation",
    )
    production_campaign_path = _reference_path(
        production["campaign_config"],
        label="benchmark.production_interpretation.campaign_config",
    )
    production_cores_config_path = _reference_path(
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
    production_execution = _load_yaml(
        production_cores_config_path,
        label="benchmark production execution config",
    )
    authored_cluster = _mapping(
        production_execution.get("cluster"),
        label="benchmark production execution cluster",
    )
    authored_cores = _positive_integer(
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
    cores_per_node = _positive_integer(
        cluster["cores_per_node"],
        label="execution.cluster.cores_per_node",
    )

    variant_values = suite["variants"]
    if not isinstance(variant_values, list) or len(variant_values) != _BENCHMARK_VARIANT_COUNT:
        message = "The maintained core benchmark suite must reference exactly four variants."
        raise ValueError(message)
    variants: list[CoreBenchmarkVariant] = []
    for index, reference in enumerate(variant_values):
        variant_path = _reference_path(
            reference,
            label=f"benchmark.variants[{index}]",
        )
        raw = _load_yaml(variant_path, label="core benchmark variant")
        _exact_keys(
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
        owner = _reference_path(
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
        cores = _positive_integer(
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
                "campaign_config": _repository_relative(campaign_path),
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
                "source_path": _repository_relative(variant.source_path),
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
    """Return the compact two-case wave contract without materializing inputs."""
    suite = load_core_benchmark_suite(path, require_executable=require_executable)
    wave_order = suite.variant_wave_order()
    return {
        "schema_kind": "generation_core_scaling_benchmark_inspection",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": _repository_relative(suite.source_path),
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
                "source_path": _repository_relative(variant.source_path),
                "cores_per_case": variant.cores_per_case,
                "execution_id": suite.execution_id(variant),
            }
            for position, variant in enumerate(wave_order, start=1)
        ],
        "scientific_inputs_materialized": False,
        "dataset_membership": "none",
    }


def resolve_core_benchmark_runtime_identity(
    path: Path | str,
    *,
    git_commit: str,
    comsol_version_output: str,
) -> dict[str, Any]:
    """Resolve the deterministic runtime identity without persistent mutation."""
    suite = load_core_benchmark_suite(path, require_executable=True)
    version = _comsol_version_evidence(
        comsol_version_output,
        configured_executable=suite.resource_contract()["comsol_executable"],
    )
    run_id = core_benchmark_run_id(
        suite,
        git_commit=git_commit,
        comsol_version=version,
    )
    return {
        "schema_kind": "generation_core_scaling_benchmark_runtime_identity",
        "schema_version": 1,
        "benchmark_run_id": run_id,
        "suite_digest": suite.suite_digest,
        "git_commit": source_service.validate_git_commit(git_commit),
        "comsol_version": version,
    }


def _repository_commit() -> str:
    """Return the exact commit of the current CPU checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- site PATH owns Git
        cwd=common.paths.get_project_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return source_service.validate_git_commit(result.stdout.strip())


def _require_clean_repository() -> None:
    """Require the benchmark checkout to contain only exact committed source."""
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],  # noqa: S607 -- site PATH owns Git
        cwd=common.paths.get_project_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    changes = [line for line in result.stdout.splitlines() if line]
    if changes:
        preview = ", ".join(changes[:_MAX_DIRTY_PATH_PREVIEW])
        suffix = "" if len(changes) <= _MAX_DIRTY_PATH_PREVIEW else f" (+{len(changes) - _MAX_DIRTY_PATH_PREVIEW} more)"
        message = f"Standalone benchmark requires clean exact committed source; working-tree changes: {preview}{suffix}"
        raise RuntimeError(message)


def _require_current_checkout(manifest: Mapping[str, Any]) -> None:
    """Require the executing repository to equal the persisted source commit."""
    expected = source_service.validate_git_commit(manifest.get("git_commit"))
    actual = _repository_commit()
    if actual != expected:
        message = f"Benchmark checkout commit {actual} does not match persisted commit {expected}."
        raise RuntimeError(message)


def _comsol_version_evidence(
    output: str,
    *,
    configured_executable: str,
) -> dict[str, str]:
    """Return normalized COMSOL version evidence for benchmark identity."""
    if not isinstance(output, str):
        message = "Benchmark COMSOL version evidence must be text."
        raise TypeError(message)
    normalized = output.strip()
    if not normalized or len(normalized.encode("utf-8")) > _MAX_COMSOL_VERSION_EVIDENCE_BYTES or "\x00" in normalized:
        message = "Benchmark COMSOL version evidence is empty or unsafe."
        raise ValueError(message)
    return {
        "configured_executable": configured_executable,
        "output": normalized,
        "digest": common.serialization.canonical_json_sha256(
            {
                "configured_executable": configured_executable,
                "output": normalized,
            }
        ),
    }


def _benchmark_identity(
    suite: CoreBenchmarkSuite,
    *,
    git_commit: str,
    comsol_version: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the actual dependency-scoped standalone benchmark identity."""
    commit = source_service.validate_git_commit(git_commit)
    version = _mapping(comsol_version, label="benchmark COMSOL version evidence")
    if (
        set(version) != {"configured_executable", "output", "digest"}
        or version.get("configured_executable") != suite.resource_contract()["comsol_executable"]
        or _SHA256_PATTERN.fullmatch(str(version.get("digest"))) is None
        or _comsol_version_evidence(
            str(version.get("output")),
            configured_executable=str(version.get("configured_executable")),
        )
        != version
    ):
        message = "Benchmark COMSOL version evidence is malformed or inconsistent."
        raise ValueError(message)
    cases = suite.case_selections()
    return {
        "schema_kind": BENCHMARK_RUN_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "git_commit": commit,
        "suite_digest": suite.suite_digest,
        "case_selection_digests": [case["selection_digest"] for case in cases],
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "case_input_config_digest": suite.case_config.case_input_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "export_contract_sha256": cases[0]["export_contract_sha256"],
        "execution_config_digest": cases[0]["execution_config_digest"],
        "variant_wave_order": [variant.variant_id for variant in suite.variant_wave_order()],
        "representative_case_count": suite.representative_case_count,
        "variants": _variant_records(suite),
        "comsol_version": version,
    }


def core_benchmark_run_id(
    suite: CoreBenchmarkSuite,
    *,
    git_commit: str,
    comsol_version: Mapping[str, Any],
) -> str:
    """Return the standalone run identity from actual benchmark dependencies."""
    digest = common.serialization.canonical_json_sha256(
        _benchmark_identity(
            suite,
            git_commit=git_commit,
            comsol_version=comsol_version,
        )
    )
    return f"core_scaling_transient__{digest[:16]}"


def core_benchmark_directory(
    run_id: str,
    *,
    storage_root: Path | str,
) -> Path:
    """Return one dedicated benchmark evidence directory."""
    safe_id = common.paths.validate_logical_name(run_id, label="benchmark_run_id")
    if _BENCHMARK_RUN_ID_PATTERN.fullmatch(safe_id) is None:
        message = f"Malformed core benchmark run ID: {run_id!r}."
        raise ValueError(message)
    return (
        common.paths.get_generation_performance_benchmark_root(
            storage_root=storage_root,
        )
        / BENCHMARK_FAMILY
        / safe_id
    )


def _preflight_path(
    run_id: str,
    *,
    storage_root: Path | str,
) -> Path:
    """Return the immutable standalone preflight receipt path."""
    return core_benchmark_directory(run_id, storage_root=storage_root) / "benchmark_preflight.json"


def _probe_directory_capability(path: Path | str, *, label: str) -> dict[str, Any]:
    """Prove one existing directory supports small same-filesystem writes."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        message = f"{label} must be an absolute directory: {candidate}"
        raise ValueError(message)
    resolved = candidate.resolve()
    if not resolved.is_dir() or not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
        message = f"{label} is not an available readable/writable directory: {resolved}"
        raise FileNotFoundError(message)
    descriptor, probe_name = tempfile.mkstemp(prefix=".benchmark-preflight.", dir=resolved)
    probe = Path(probe_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"benchmark-preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not probe.is_file() or probe.stat().st_size < 1:
            message = f"{label} capability probe did not persist an ordinary file: {resolved}"
            raise RuntimeError(message)
    finally:
        probe.unlink(missing_ok=True)
    usage = shutil.disk_usage(resolved)
    if usage.free < 1:
        message = f"{label} reports no free storage capacity: {resolved}"
        raise RuntimeError(message)
    return {
        "path": str(resolved),
        "free_bytes_observed": usage.free,
        "small_write_probe": "pass",
    }


def _validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    suite: CoreBenchmarkSuite,
    git_commit: str,
) -> None:
    """Validate immutable standalone preflight identity and owned checks."""
    expected_keys = {
        "schema_kind",
        "schema_version",
        "status",
        "recorded_at",
        "benchmark_preflight_seconds",
        "benchmark_run_id",
        "suite_config",
        "benchmark_identity",
        "comsol_runtime",
        "python_runtime",
        "storage_capabilities",
        "submission_command_digest",
        "checks",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_kind") != BENCHMARK_PREFLIGHT_SCHEMA_KIND
        or payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or payload.get("status") != "pass"
        or payload.get("benchmark_run_id") != run_id
        or payload.get("suite_config") != _repository_relative(suite.source_path)
    ):
        message = f"Standalone benchmark preflight receipt is malformed: {run_id}"
        raise ValueError(message)
    identity = _mapping(
        payload.get("benchmark_identity"),
        label="benchmark preflight identity",
    )
    expected_identity = _benchmark_identity(
        suite,
        git_commit=git_commit,
        comsol_version=_mapping(
            identity.get("comsol_version"),
            label="benchmark preflight COMSOL version",
        ),
    )
    if (
        identity != expected_identity
        or core_benchmark_run_id(
            suite,
            git_commit=git_commit,
            comsol_version=identity["comsol_version"],
        )
        != run_id
    ):
        message = f"Standalone benchmark preflight identity conflicts: {run_id}"
        raise ValueError(message)
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or set(checks.values()) != {"pass"}:
        message = f"Standalone benchmark preflight checks are incomplete: {run_id}"
        raise ValueError(message)
    capabilities = payload.get("storage_capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != {
        "scratch",
        "persistent",
    }:
        message = f"Standalone benchmark storage capability evidence is malformed: {run_id}"
        raise ValueError(message)
    runtime = payload.get("comsol_runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"resolved_executable", "version"}
        or not Path(str(runtime.get("resolved_executable"))).is_absolute()
        or runtime.get("version") != identity["comsol_version"]
    ):
        message = f"Standalone benchmark COMSOL runtime evidence is malformed: {run_id}"
        raise ValueError(message)
    python_runtime = payload.get("python_runtime")
    if (
        not isinstance(python_runtime, dict)
        or set(python_runtime) != {"executable", "version", "imports"}
        or not isinstance(python_runtime.get("imports"), dict)
    ):
        message = f"Standalone benchmark Python runtime evidence is malformed: {run_id}"
        raise ValueError(message)
    if _SHA256_PATTERN.fullmatch(str(payload.get("submission_command_digest"))) is None:
        message = f"Standalone benchmark command evidence is malformed: {run_id}"
        raise ValueError(message)
    _timestamp(payload.get("recorded_at"), label="benchmark preflight recorded_at")
    preflight_seconds = payload.get("benchmark_preflight_seconds")
    if (
        isinstance(preflight_seconds, bool)
        or not isinstance(preflight_seconds, (int, float))
        or not math.isfinite(float(preflight_seconds))
        or float(preflight_seconds) < 0.0
    ):
        message = "Standalone benchmark preflight duration is malformed."
        raise ValueError(message)


def preflight_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Run inexpensive benchmark-owned source, runtime, storage, and command checks."""
    started = time.perf_counter()
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"CPU checkout commit {current_commit} does not match requested benchmark commit {requested_commit}."
        raise RuntimeError(message)
    _require_clean_repository()
    suite = load_core_benchmark_suite(path, require_executable=True)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    persistent = _probe_directory_capability(storage, label="benchmark persistent storage")
    scratch = _probe_directory_capability(scratch_root, label="benchmark scratch")
    executable = Path(comsol_executable_path).expanduser()
    if not executable.is_absolute():
        message = "Resolved benchmark COMSOL executable must be an absolute path."
        raise ValueError(message)
    resolved_executable = executable.resolve()
    if not resolved_executable.is_file() or not os.access(resolved_executable, os.X_OK):
        message = f"Resolved benchmark COMSOL executable is unavailable: {resolved_executable}"
        raise FileNotFoundError(message)
    version = _comsol_version_evidence(
        comsol_version_output,
        configured_executable=suite.resource_contract()["comsol_executable"],
    )
    if any(
        argument == "-usebatchlic" or argument.startswith("-usebatchlic=")
        for argument in suite.case_config.execution_values["runtime"]["extra_arguments"]
    ):
        message = "Standalone benchmark rejects unverified COMSOL -usebatchlic."
        raise ValueError(message)
    identity = _benchmark_identity(
        suite,
        git_commit=requested_commit,
        comsol_version=version,
    )
    run_id = core_benchmark_run_id(
        suite,
        git_commit=requested_commit,
        comsol_version=version,
    )
    directory = core_benchmark_directory(run_id, storage_root=storage)
    logs = directory / "scheduler"
    commands = [
        build_core_benchmark_slurm_command(
            suite,
            run_id=run_id,
            storage_root=storage,
            log_directory=logs,
            variant=variant,
            case_position=case_position,
        )
        for variant, case_position in _measured_sequence(suite)
    ]
    if any(argument == "--licenses" or argument.startswith("--licenses=") for command in commands for argument in command):
        message = "Standalone benchmark Slurm commands must not request --licenses."
        raise ValueError(message)
    payload = {
        "schema_kind": BENCHMARK_PREFLIGHT_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "pass",
        "recorded_at": _utc_now(),
        "benchmark_preflight_seconds": time.perf_counter() - started,
        "benchmark_run_id": run_id,
        "suite_config": _repository_relative(suite.source_path),
        "benchmark_identity": identity,
        "comsol_runtime": {
            "resolved_executable": str(resolved_executable),
            "version": version,
        },
        "python_runtime": {
            "executable": sys.executable,
            "version": sys.version,
            "imports": {
                "h5py": h5py.__version__,
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "storage_capabilities": {
            "scratch": scratch,
            "persistent": persistent,
        },
        "submission_command_digest": common.serialization.canonical_json_sha256(commands),
        "checks": {
            "clean_exact_source": "pass",
            "suite_and_case": "pass",
            "template_and_export_contract": "pass",
            "comsol_executable_and_version": "pass",
            "python_runtime_and_imports": "pass",
            "scheduler_resources": "pass",
            "scratch_storage": "pass",
            "persistent_storage": "pass",
            "slurm_commands": "pass",
            "unsupported_license_flags_absent": "pass",
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = _preflight_path(run_id, storage_root=storage)
    if receipt_path.exists():
        existing = _load_json(
            receipt_path,
            label="standalone benchmark preflight receipt",
        )
        _validate_preflight_payload(
            existing,
            run_id=run_id,
            suite=suite,
            git_commit=requested_commit,
        )
        return existing
    _write_immutable_json(
        receipt_path,
        payload,
        label="standalone benchmark preflight receipt",
    )
    _validate_preflight_payload(
        payload,
        run_id=run_id,
        suite=suite,
        git_commit=requested_commit,
    )
    return payload


def _load_core_benchmark_preflight(
    run_id: str,
    *,
    suite: CoreBenchmarkSuite,
    git_commit: str,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate the immutable standalone preflight receipt."""
    payload = _load_json(
        _preflight_path(run_id, storage_root=storage_root),
        label="standalone benchmark preflight receipt",
    )
    _validate_preflight_payload(
        payload,
        run_id=run_id,
        suite=suite,
        git_commit=git_commit,
    )
    return payload


def _variant_records(suite: CoreBenchmarkSuite) -> list[dict[str, Any]]:
    """Return persisted variant identities from one resolved suite."""
    return [
        {
            "variant_id": variant.variant_id,
            "config": _repository_relative(variant.source_path),
            "cores_per_case": variant.cores_per_case,
            "execution_id": suite.execution_id(variant),
        }
        for variant in suite.variants
    ]


def _node_environment(suite: CoreBenchmarkSuite, run_id: str) -> list[str]:
    """Return exact environment bindings consumed by the compute-node script."""
    site = suite.case_campaign.execution_values["site"]
    return [
        f"GENERATION_BENCHMARK_RUN_ID={run_id}",
        f"GENERATION_PYTHON_MODULE={site['python_module']}",
        f"GENERATION_COMSOL_MODULE={site['comsol_module']}",
        f"GENERATION_PYTHON_EXECUTABLE={site['python_executable']}",
        f"GENERATION_COMSOL_EXECUTABLE={site['comsol_executable']}",
    ]


def _measured_sequence(
    suite: CoreBenchmarkSuite,
) -> tuple[tuple[CoreBenchmarkVariant, int], ...]:
    """Return both cases for each production-first sequential variant wave."""
    return tuple(
        (variant, case_position) for variant in suite.variant_wave_order() for case_position in range(1, suite.representative_case_count + 1)
    )


def build_core_benchmark_slurm_command(
    suite: CoreBenchmarkSuite,
    *,
    run_id: str,
    storage_root: Path,
    log_directory: Path,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> list[str]:
    """Build one ordinary measured benchmark Slurm job."""
    repository = common.paths.get_project_root().resolve()
    launcher = repository / "scripts" / "generation_benchmark_node.sh"
    if not launcher.is_file() or launcher.is_symlink():
        message = f"Benchmark compute-node launcher is missing or unsafe: {launcher}"
        raise FileNotFoundError(message)
    if not storage_root.is_absolute() or not log_directory.is_absolute():
        message = "Benchmark Slurm storage and log roots must be absolute."
        raise ValueError(message)
    suite.work_unit_id(variant, case_position)
    environment = _node_environment(suite, run_id)
    representative = suite.representative_case(case_position)
    worker = [
        str(launcher),
        str(repository),
        run_id,
        variant.variant_id,
        representative.case_role,
    ]
    wrapped = shlex.join(["env", *environment, *worker])
    job_suffix = f"c{variant.cores_per_case:02d}-{representative.case_role[:3]}-{run_id.rsplit('__', maxsplit=1)[-1][:4]}"
    job_name = f"td-bench-{job_suffix}"
    if len(job_name) > _MAX_SLURM_JOB_NAME_LENGTH or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", job_name) is None:
        message = f"Benchmark Slurm job name is unsafe or exceeds 48 characters: {job_name!r}."
        raise ValueError(message)
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={variant.cores_per_case}",
        f"--chdir={repository}",
        f"--job-name={job_name}",
        "--export=ALL",
        f"--output={log_directory}/slurm-%j.out",
        f"--error={log_directory}/slurm-%j.err",
    ]
    if suite.partition is not None:
        command.append(f"--partition={suite.partition}")
    if suite.wall_time is not None:
        command.append(f"--time={suite.wall_time}")
    command.extend(suite.scheduler_options)
    command.append(f"--wrap={wrapped}")
    return command


def _plan_payload(
    suite: CoreBenchmarkSuite,
    *,
    git_commit: str,
    storage: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical two-case, four-wave benchmark plan."""
    run_id = str(preflight["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    logs = directory / "scheduler"
    sequence = _measured_sequence(suite)
    work_unit_commands = [
        {
            "variant_id": variant.variant_id,
            "cores_per_case": variant.cores_per_case,
            "case_position": case_position,
            "case_role": suite.representative_case(case_position).case_role,
            "work_unit_id": suite.work_unit_id(variant, case_position),
            "command": build_core_benchmark_slurm_command(
                suite,
                run_id=run_id,
                storage_root=storage,
                log_directory=logs,
                variant=variant,
                case_position=case_position,
            ),
        }
        for variant, case_position in sequence
    ]
    wave_order = suite.variant_wave_order()
    return {
        "schema_kind": "generation_core_scaling_benchmark_plan",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "state": "planned",
        "filesystem_mutated": True,
        "benchmark_run_id": run_id,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": _repository_relative(suite.source_path),
        "git_commit": git_commit,
        "preflight": {
            "receipt_sha256": common.serialization.file_sha256(_preflight_path(run_id, storage_root=storage)),
            "benchmark_identity": preflight["benchmark_identity"],
            "comsol_runtime": preflight["comsol_runtime"],
            "checks": preflight["checks"],
        },
        "representative_cases": suite.case_selections(),
        "parallel_cases_per_variant": suite.representative_case_count,
        "variants": _variant_records(suite),
        "variant_wave_order": [variant.variant_id for variant in wave_order],
        "required_successful_measurements": len(sequence),
        "canary_wave": {
            "variant_id": wave_order[0].variant_id,
            "cores_per_case": wave_order[0].cores_per_case,
            "case_roles": [item.case_role for item in suite.representative_cases],
            "included_in_final_measurements": True,
            "additional_canary_work_units": 0,
        },
        "measurement_waves": [
            {
                "wave_position": wave_position,
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
                "depends_on_wave": None if wave_position == 1 else wave_position - 1,
                "work_unit_ids": [suite.work_unit_id(variant, case_position) for case_position in range(1, suite.representative_case_count + 1)],
                "case_execution": "concurrent",
            }
            for wave_position, variant in enumerate(wave_order, start=1)
        ],
        "resource_contract": suite.resource_contract(),
        "paths": {
            "storage_root": str(storage),
            "benchmark_root": str(directory),
            "scheduler_logs": str(logs),
        },
        "canonical_input_preparation": {
            "execution_environment": "cpu_login",
            "scheduler_job": False,
            "proof_directory": "canonical_cases",
            "case_count": suite.representative_case_count,
        },
        "submission_commands": {"work_units": work_unit_commands},
        "isolation": {
            "scientific_cases_per_job": 1,
            "maximum_active_benchmark_jobs": suite.representative_case_count,
            "scheduler_arrays": False,
            "ordered_variant_waves": True,
            "case_execution_within_variant": "concurrent",
            "queue_wait_primary_metric": False,
            "license_wait_primary_metric": False,
        },
        "dataset_membership": "none",
    }


def plan_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Run or reuse standalone preflight and return the canonical plan."""
    requested_commit = source_service.validate_git_commit(git_commit)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    suite = load_core_benchmark_suite(path, require_executable=True)
    preflight = preflight_core_benchmark(
        path,
        git_commit=requested_commit,
        storage_root=storage,
        scratch_root=scratch_root,
        comsol_version_output=comsol_version_output,
        comsol_executable_path=comsol_executable_path,
    )
    return _plan_payload(
        suite,
        git_commit=requested_commit,
        storage=storage,
        preflight=preflight,
    )


def _manifest_path(run_id: str, *, storage_root: Path | str) -> Path:
    """Return one benchmark run manifest path."""
    return core_benchmark_directory(run_id, storage_root=storage_root) / "benchmark_manifest.json"


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """Write one immutable canonical JSON object or validate an exact retry."""
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.is_symlink():
            message = f"Existing {label} path is unsafe: {path}"
            raise FileExistsError(message)
        if path.read_text(encoding="utf-8") != serialized:
            message = f"Existing {label} conflicts: {path}"
            raise FileExistsError(message)
        return path
    return common.serialization.atomic_write_text(path, serialized)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required symlink-free JSON object."""
    if not path.is_file() or path.is_symlink():
        message = f"Required {label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not load {label}: {path}"
        raise ValueError(message) from error
    return _mapping(value, label=label)


def load_core_benchmark_manifest(
    run_id: str,
    *,
    storage_root: Path | str,
) -> tuple[dict[str, Any], CoreBenchmarkSuite]:
    """Load one run manifest and re-resolve its exact suite identity."""
    manifest = _load_json(
        _manifest_path(run_id, storage_root=storage_root),
        label="core benchmark manifest",
    )
    _exact_keys(
        manifest,
        {
            "schema_kind",
            "schema_version",
            "benchmark_run_id",
            "suite_name",
            "suite_digest",
            "suite_config",
            "git_commit",
            "preflight",
            "representative_cases",
            "variants",
            "variant_wave_order",
            "canary_wave",
            "measurement_waves",
            "required_successful_measurements",
            "resource_contract",
            "created_at",
            "state",
            "measured_job_ids",
            "submission_history",
        },
        label="core benchmark manifest",
    )
    if (
        manifest["schema_kind"] != BENCHMARK_RUN_SCHEMA_KIND
        or manifest["schema_version"] != BENCHMARK_SCHEMA_VERSION
        or manifest["benchmark_run_id"] != run_id
    ):
        message = f"Core benchmark manifest schema or run identity is invalid: {run_id}"
        raise ValueError(message)
    source_service.validate_git_commit(manifest["git_commit"])
    suite_path = _reference_path(
        manifest["suite_config"],
        label="benchmark manifest suite_config",
    )
    suite = load_core_benchmark_suite(suite_path, require_executable=True)
    wave_order = suite.variant_wave_order()
    expected = {
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "representative_cases": suite.case_selections(),
        "variants": _variant_records(suite),
        "variant_wave_order": [variant.variant_id for variant in wave_order],
        "canary_wave": {
            "variant_id": wave_order[0].variant_id,
            "cores_per_case": wave_order[0].cores_per_case,
            "case_roles": [item.case_role for item in suite.representative_cases],
            "included_in_final_measurements": True,
            "additional_canary_work_units": 0,
        },
        "measurement_waves": [
            {
                "wave_position": wave_position,
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
                "depends_on_wave": None if wave_position == 1 else wave_position - 1,
                "work_unit_ids": [suite.work_unit_id(variant, case_position) for case_position in range(1, suite.representative_case_count + 1)],
                "case_execution": "concurrent",
            }
            for wave_position, variant in enumerate(wave_order, start=1)
        ],
        "required_successful_measurements": (len(suite.variants) * suite.representative_case_count),
        "resource_contract": suite.resource_contract(),
    }

    if any(manifest.get(key) != value for key, value in expected.items()):
        message = f"Core benchmark manifest no longer matches current suite source: {run_id}"
        raise ValueError(message)
    preflight = _load_core_benchmark_preflight(
        run_id,
        suite=suite,
        git_commit=str(manifest["git_commit"]),
        storage_root=storage_root,
    )
    expected_preflight = {
        "receipt_sha256": common.serialization.file_sha256(_preflight_path(run_id, storage_root=storage_root)),
        "benchmark_identity": preflight["benchmark_identity"],
        "comsol_runtime": preflight["comsol_runtime"],
        "checks": preflight["checks"],
    }
    if manifest.get("preflight") != expected_preflight:
        message = f"Core benchmark manifest preflight identity is inconsistent: {run_id}"
        raise ValueError(message)
    return manifest, suite


def _submit(command: Sequence[str], *, git_commit: str, run_id: str) -> str:
    """Submit one typed Slurm command and return its parsable root job ID."""
    environment = os.environ.copy()
    environment["GENERATION_GIT_COMMIT"] = git_commit
    environment["GENERATION_BENCHMARK_RUN_ID"] = run_id
    result = subprocess.run(  # noqa: S603 -- typed Slurm service builds the argv
        list(command),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    if _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = f"Slurm returned an invalid benchmark job identifier: {result.stdout!r}."
        raise RuntimeError(message)
    return job_id


def _success_path(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> Path:
    """Return the immutable success evidence path for one case_position."""
    return directory / "runs" / suite.execution_id(variant) / suite.work_unit_id(variant, case_position) / "success.json"


def _validate_benchmark_attempt_chain(attempts: Sequence[Path]) -> None:
    """Validate contiguous benchmark attempts and immediate receipt digests."""
    stable_identity: dict[str, Any] | None = None
    previous_path: Path | None = None
    identity_keys = (
        "benchmark_run_id",
        "suite_digest",
        "variant_id",
        "execution_id",
        "case_position",
        "case_role",
        "work_unit_id",
        "git_commit",
        "case_input_id",
        "simulation_case_id",
        "scientific_config_digest",
        "template_sha256",
        "benchmark_preflight_sha256",
        "cores_per_case",
    )
    for expected_index, attempt_path in enumerate(attempts, start=1):
        if attempt_path.name != f"attempt-{expected_index:04d}.json":
            message = f"Benchmark attempt history is not contiguous: {attempt_path}"
            raise ValueError(message)
        payload = _load_json(attempt_path, label="benchmark case_position attempt")
        previous = payload.get("previous_attempt")
        if expected_index == 1:
            valid_previous = previous is None
        else:
            valid_previous = (
                isinstance(previous, dict)
                and set(previous) == {"attempt", "receipt_sha256"}
                and previous.get("attempt") == expected_index - 1
                and previous_path is not None
                and previous.get("receipt_sha256") == common.serialization.file_sha256(previous_path)
            )
        if payload.get("attempt") != expected_index or not valid_previous:
            message = f"Benchmark attempt predecessor chain is invalid: {attempt_path}"
            raise ValueError(message)
        identity = {key: payload.get(key) for key in identity_keys}
        if stable_identity is None:
            stable_identity = identity
        elif identity != stable_identity:
            message = f"Benchmark attempt identity changed across retry: {attempt_path}"
            raise ValueError(message)
        previous_path = attempt_path


def _benchmark_previous_attempt_reference(attempts: Sequence[Path]) -> dict[str, Any] | None:
    """Return the exact immediate predecessor after admitting the prior chain."""
    _validate_benchmark_attempt_chain(attempts)
    if not attempts:
        return None
    return {
        "attempt": len(attempts),
        "receipt_sha256": common.serialization.file_sha256(attempts[-1]),
    }


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
    status: str,
) -> None:
    """Validate identity fields shared by successful and failed attempts."""
    expected = {
        "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": status,
        "suite_digest": suite.suite_digest,
        "variant_id": variant.variant_id,
        "execution_id": suite.execution_id(variant),
        "case_position": case_position,
        "case_role": suite.representative_case(case_position).case_role,
        "work_unit_id": suite.work_unit_id(variant, case_position),
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "cores_per_case": variant.cores_per_case,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        message = f"Benchmark {status} evidence conflicts for {expected['work_unit_id']}."
        raise ValueError(message)
    if _BENCHMARK_RUN_ID_PATTERN.fullmatch(str(result.get("benchmark_run_id"))) is None:
        message = f"Benchmark result has a malformed run ID for {expected['work_unit_id']}."
        raise ValueError(message)
    source_service.validate_git_commit(result.get("git_commit"))
    if _SHA256_PATTERN.fullmatch(str(result.get("benchmark_preflight_sha256"))) is None:
        message = f"Benchmark result has malformed preflight identity for {expected['work_unit_id']}."
        raise ValueError(message)
    for key in ("case_input_id", "simulation_case_id"):
        if _SHA256_PATTERN.fullmatch(str(result.get(key))) is None:
            message = f"Benchmark result has malformed {key} for {expected['work_unit_id']}."
            raise ValueError(message)
    attempt = result.get("attempt")
    previous = result.get("previous_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        message = f"Benchmark result has an invalid attempt index for {expected['work_unit_id']}."
        raise ValueError(message)
    valid_previous = (
        previous is None
        if attempt == 1
        else isinstance(previous, dict)
        and set(previous) == {"attempt", "receipt_sha256"}
        and previous.get("attempt") == attempt - 1
        and _SHA256_PATTERN.fullmatch(str(previous.get("receipt_sha256"))) is not None
    )
    if not valid_previous:
        message = f"Benchmark result has an invalid attempt chain for {expected['work_unit_id']}."
        raise ValueError(message)


def _validate_resource_evidence(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> None:
    """Validate allocation plus observed per-case memory and scratch evidence."""
    resource = result.get("resource")
    expected_keys = {
        "node",
        "partition",
        "requested_cpus",
        "allocated_cpus",
        "comsol_np",
        "slurm_job_id",
        "peak_memory_bytes",
        "peak_scratch_bytes",
    }
    if not isinstance(resource, dict) or set(resource) != expected_keys:
        message = f"Benchmark resource evidence is missing for {suite.work_unit_id(variant, case_position)}."
        raise TypeError(message)
    expected = {
        "requested_cpus": variant.cores_per_case,
        "allocated_cpus": variant.cores_per_case,
        "comsol_np": variant.cores_per_case,
    }
    if any(resource.get(key) != value for key, value in expected.items()):
        message = f"Benchmark allocation evidence conflicts for {suite.work_unit_id(variant, case_position)}."
        raise ValueError(message)
    if suite.partition is not None and resource.get("partition") != suite.partition:
        message = f"Benchmark partition evidence conflicts for {suite.work_unit_id(variant, case_position)}."
        raise ValueError(message)
    node = resource.get("node")
    if not isinstance(node, str) or not node or any(character in node for character in "\r\n\t"):
        message = f"Benchmark node evidence is malformed for {suite.work_unit_id(variant, case_position)}."
        raise ValueError(message)
    if _JOB_ID_PATTERN.fullmatch(str(resource.get("slurm_job_id"))) is None:
        message = f"Benchmark slurm_job_id is malformed for {suite.work_unit_id(variant, case_position)}."
        raise ValueError(message)
    for key in ("peak_memory_bytes", "peak_scratch_bytes"):
        value = resource.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            message = f"Benchmark {key} is malformed for {suite.work_unit_id(variant, case_position)}."
            raise ValueError(message)


def _timestamp(value: object, *, label: str) -> datetime:
    """Parse one timezone-aware benchmark scheduler timestamp."""
    if not isinstance(value, str):
        message = f"{label} must be one ISO timestamp."
        raise TypeError(message)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        message = f"{label} must be one ISO timestamp."
        raise ValueError(message) from error
    if parsed.tzinfo is None:
        message = f"{label} must include a timezone."
        raise ValueError(message)
    return parsed


def _validate_scheduler_timing(
    result: Mapping[str, Any],
    *,
    work_unit_id: str,
) -> None:
    """Validate submit/start/completion and derived queue/turnaround timing."""
    timing = result.get("scheduler_timing")
    if not isinstance(timing, dict) or set(timing) != {
        "submit_time",
        "start_time",
        "completion_time",
        "queue_wait_s",
        "turnaround_s",
    }:
        message = f"Benchmark scheduler timing is incomplete for {work_unit_id}."
        raise ValueError(message)
    submitted = _timestamp(timing["submit_time"], label="submit_time")
    started = _timestamp(timing["start_time"], label="start_time")
    completed = _timestamp(timing["completion_time"], label="completion_time")
    if started < submitted or completed < started:
        message = f"Benchmark scheduler timestamps are out of order for {work_unit_id}."
        raise ValueError(message)
    expected_queue = (started - submitted).total_seconds()
    expected_turnaround = (completed - submitted).total_seconds()
    for key, expected in (("queue_wait_s", expected_queue), ("turnaround_s", expected_turnaround)):
        value = timing[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=1.0e-9, abs_tol=1.0e-6)
        ):
            message = f"Benchmark {key} is inconsistent for {work_unit_id}."
            raise ValueError(message)


def _scheduler_timing(
    *,
    submit_time: str,
    start_time: str,
    completion_time: str,
) -> dict[str, Any]:
    """Return derived queue-wait and turnaround evidence."""
    submitted = _timestamp(submit_time, label="submit_time")
    started = _timestamp(start_time, label="start_time")
    completed = _timestamp(completion_time, label="completion_time")
    if started < submitted:
        started = submitted
        start_time = submit_time
    if completed < started:
        message = "Benchmark completion time precedes start time."
        raise RuntimeError(message)
    return {
        "submit_time": submit_time,
        "start_time": start_time,
        "completion_time": completion_time,
        "queue_wait_s": (started - submitted).total_seconds(),
        "turnaround_s": (completed - submitted).total_seconds(),
    }


def _validate_work_unit_timings(
    result: Mapping[str, Any],
    *,
    work_unit_id: str,
    success: bool,
) -> None:
    """Validate exact separated timings without mixing waits into solve time."""
    timings = result.get("timings_seconds")
    if not isinstance(timings, dict) or set(timings) != _WORK_UNIT_TIMING_FIELDS:
        message = f"Benchmark timings are incomplete for {work_unit_id}."
        raise ValueError(message)
    optional = {"comsol_process_seconds", "export_conversion_seconds"}
    for key, value in timings.items():
        if not success and key in optional and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            message = f"Benchmark {key} is malformed for {work_unit_id}."
            raise ValueError(message)
    if success and float(timings["comsol_process_seconds"]) <= 0.0:
        message = f"Benchmark successful COMSOL runtime is invalid for {work_unit_id}."
        raise ValueError(message)


def _validate_success_support_evidence(
    result: Mapping[str, Any],
    *,
    work_unit_id: str,
) -> None:
    """Validate successful solver interval, license, and bounded-log evidence."""
    interval = result.get("solver_interval")
    if not isinstance(interval, dict) or set(interval) != {"started_at", "ended_at"}:
        message = f"Benchmark solver interval is missing for {work_unit_id}."
        raise ValueError(message)
    if _timestamp(interval["ended_at"], label="solver ended_at") < _timestamp(
        interval["started_at"],
        label="solver started_at",
    ):
        message = f"Benchmark solver interval is out of order for {work_unit_id}."
        raise ValueError(message)
    license_evidence = result.get("license")
    expected_license_keys = {
        "license_blocked_submission_count",
        "license_wait_seconds",
        "license_probe_seconds",
        "scheduler_queue_seconds_before_success",
        "detected_feature",
        "detected_error_code",
        "matched_signatures",
        "raw_excerpt",
        "successful_artifacts_override_prior_warning",
    }
    if not isinstance(license_evidence, dict) or set(license_evidence) != expected_license_keys:
        message = f"Benchmark license evidence is missing for {work_unit_id}."
        raise ValueError(message)
    blocked = license_evidence["license_blocked_submission_count"]
    if isinstance(blocked, bool) or not isinstance(blocked, int) or blocked < 0:
        message = f"Benchmark license-block count is malformed for {work_unit_id}."
        raise ValueError(message)
    for key in (
        "license_wait_seconds",
        "license_probe_seconds",
        "scheduler_queue_seconds_before_success",
    ):
        value = license_evidence[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            message = f"Benchmark {key} is malformed for {work_unit_id}."
            raise ValueError(message)
    if license_evidence["successful_artifacts_override_prior_warning"] is not True:
        message = f"Benchmark success precedence is missing for {work_unit_id}."
        raise ValueError(message)
    signatures = license_evidence["matched_signatures"]
    if not isinstance(signatures, list) or not all(isinstance(item, str) and item for item in signatures):
        message = f"Benchmark license signatures are malformed for {work_unit_id}."
        raise ValueError(message)
    optional_text = (
        license_evidence["detected_feature"],
        license_evidence["detected_error_code"],
        license_evidence["raw_excerpt"],
    )
    if not all(item is None or isinstance(item, str) for item in optional_text):
        message = f"Benchmark license details are malformed for {work_unit_id}."
        raise ValueError(message)
    solver_log = result.get("solver_log")
    if not isinstance(solver_log, dict) or set(solver_log) != {
        "sha256",
        "size_bytes",
        "excerpt",
        "excerpt_truncated",
    }:
        message = f"Benchmark bounded solver-log evidence is missing for {work_unit_id}."
        raise ValueError(message)
    size = solver_log["size_bytes"]
    if (
        _SHA256_PATTERN.fullmatch(str(solver_log["sha256"])) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(solver_log["excerpt"], str)
        or not isinstance(solver_log["excerpt_truncated"], bool)
        or len(solver_log["excerpt"].encode("utf-8")) > _MAX_BENCHMARK_LOG_EXCERPT_BYTES * 3
    ):
        message = f"Benchmark bounded solver-log evidence is malformed for {work_unit_id}."
        raise ValueError(message)


def _validate_success_result(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> None:
    """Validate one immutable successful representative-case measurement."""
    work_unit_id = suite.work_unit_id(variant, case_position)
    _validate_result_identity(
        result,
        suite=suite,
        variant=variant,
        case_position=case_position,
        status="success",
    )
    _validate_resource_evidence(
        result,
        suite=suite,
        variant=variant,
        case_position=case_position,
    )
    _validate_scheduler_timing(result, work_unit_id=work_unit_id)
    _validate_work_unit_timings(result, work_unit_id=work_unit_id, success=True)
    _validate_success_support_evidence(result, work_unit_id=work_unit_id)
    hdf5 = result.get("hdf5")
    if not isinstance(hdf5, dict) or hdf5.get("retained_as_scientific_case") is not False:
        message = f"Benchmark HDF5 isolation evidence is missing for {work_unit_id}."
        raise ValueError(message)
    size = hdf5.get("size_bytes")
    identity = hdf5.get("identity")
    if (
        _SHA256_PATTERN.fullmatch(str(hdf5.get("sha256"))) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(identity, dict)
        or identity.get("case_input_id") != result["case_input_id"]
        or identity.get("simulation_case_id") != result["simulation_case_id"]
    ):
        message = f"Benchmark HDF5 evidence is malformed for {work_unit_id}."
        raise ValueError(message)


def _validate_failure_result(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
    attempts: Sequence[Path],
) -> None:
    """Validate one append-only failed scientific work-unit attempt."""
    del attempts
    work_unit_id = suite.work_unit_id(variant, case_position)
    _validate_result_identity(
        result,
        suite=suite,
        variant=variant,
        case_position=case_position,
        status="failed",
    )
    _validate_resource_evidence(
        result,
        suite=suite,
        variant=variant,
        case_position=case_position,
    )
    _validate_scheduler_timing(result, work_unit_id=work_unit_id)
    _validate_work_unit_timings(result, work_unit_id=work_unit_id, success=False)
    error = result.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("type"), str) or not error["type"] or not isinstance(error.get("message"), str):
        message = f"Benchmark failure error evidence is malformed for {work_unit_id}."
        raise ValueError(message)
    if result.get("temporary_license_retry") is not None:
        message = f"Temporary license capacity must remain operationally pending for {work_unit_id}."
        raise ValueError(message)


def _benchmark_license_wait_path(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> Path:
    """Return the sole mutable license-wait record for one case_position."""
    return directory / "runs" / suite.execution_id(variant) / suite.work_unit_id(variant, case_position) / "license_wait.json"


def _benchmark_wait_timestamp(value: object, *, label: str) -> datetime:
    """Return one timezone-aware benchmark wait timestamp."""
    if not isinstance(value, str):
        message = f"{label} must be a timezone-aware timestamp."
        raise TypeError(message)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        message = f"{label} must be a timezone-aware timestamp."
        raise ValueError(message) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        message = f"{label} must be a timezone-aware timestamp."
        raise ValueError(message)
    return parsed.astimezone(timezone.utc)


def _validate_benchmark_license_wait(
    payload: object,
    *,
    run_id: str,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
    path: Path,
) -> dict[str, Any]:
    """Validate one compact benchmark operational wait record."""
    work_unit_id = suite.work_unit_id(variant, case_position)
    expected_keys = {
        "schema_kind",
        "schema_version",
        "benchmark_run_id",
        "work_unit_id",
        "suite_digest",
        "variant_id",
        "case_position",
        "case_role",
        "scientific_config_digest",
        "classification",
        "feature",
        "error_code",
        "matched_signatures",
        "comsol_exit_code",
        "solver_progress_started",
        "expected_exports_exist",
        "first_blocked_at",
        "last_blocked_at",
        "retry_count",
        "latest_job_id",
        "recent_job_ids",
        "hostname",
        "raw_excerpt",
        "delay_before_next_attempt_seconds",
        "cumulative_wait_seconds",
        "cumulative_probe_seconds",
        "cumulative_scheduler_queue_seconds",
        "retry_budget_remaining",
        "next_retry_at",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_kind") != "generation_temporary_license_wait"
        or payload.get("schema_version") != 1
        or payload.get("benchmark_run_id") != run_id
        or payload.get("work_unit_id") != work_unit_id
        or payload.get("suite_digest") != suite.suite_digest
        or payload.get("variant_id") != variant.variant_id
        or payload.get("case_position") != case_position
        or payload.get("case_role") != suite.representative_case(case_position).case_role
        or payload.get("scientific_config_digest") != suite.case_config.scientific_config_digest
        or payload.get("classification") != license_service.TEMPORARY_LICENSE_CAPACITY
        or not isinstance(payload.get("feature"), str)
        or not payload["feature"]
        or (payload.get("error_code") is not None and not isinstance(payload["error_code"], str))
        or not isinstance(payload.get("matched_signatures"), list)
        or not payload["matched_signatures"]
        or not all(isinstance(value, str) and value for value in payload["matched_signatures"])
        or payload.get("solver_progress_started") is not False
        or payload.get("expected_exports_exist") is not False
        or not isinstance(payload.get("raw_excerpt"), str)
        or not payload["raw_excerpt"]
        or not isinstance(payload.get("hostname"), str)
        or not payload["hostname"]
        or not isinstance(payload.get("retry_budget_remaining"), bool)
    ):
        message = f"Benchmark license-wait evidence is malformed: {path}"
        raise ValueError(message)
    exit_code = payload.get("comsol_exit_code")
    retry_count = payload.get("retry_count")
    delay = payload.get("delay_before_next_attempt_seconds")
    cumulative = payload.get("cumulative_wait_seconds")
    cumulative_probe = payload.get("cumulative_probe_seconds")
    cumulative_queue = payload.get("cumulative_scheduler_queue_seconds")
    job_id = payload.get("latest_job_id")
    recent_job_ids = payload.get("recent_job_ids")
    if (
        (exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)))
        or isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or retry_count < 1
        or isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or float(delay) < 0.0
        or isinstance(cumulative, bool)
        or not isinstance(cumulative, (int, float))
        or float(cumulative) < float(delay)
        or isinstance(cumulative_probe, bool)
        or not isinstance(cumulative_probe, (int, float))
        or not math.isfinite(float(cumulative_probe))
        or float(cumulative_probe) < 0.0
        or isinstance(cumulative_queue, bool)
        or not isinstance(cumulative_queue, (int, float))
        or not math.isfinite(float(cumulative_queue))
        or float(cumulative_queue) < 0.0
        or not isinstance(job_id, str)
        or _JOB_ID_PATTERN.fullmatch(job_id) is None
        or not isinstance(recent_job_ids, list)
        or not recent_job_ids
        or len(recent_job_ids) > _MAX_RECENT_JOB_IDS
        or len(recent_job_ids) != len(set(recent_job_ids))
        or recent_job_ids[-1] != job_id
        or not all(isinstance(value, str) and _JOB_ID_PATTERN.fullmatch(value) for value in recent_job_ids)
    ):
        message = f"Benchmark license-wait counters are malformed: {path}"
        raise ValueError(message)
    first = _benchmark_wait_timestamp(payload["first_blocked_at"], label="first_blocked_at")
    last = _benchmark_wait_timestamp(payload["last_blocked_at"], label="last_blocked_at")
    prior_cumulative = float(cumulative) - float(delay)
    policy = suite.case_config.execution_values["runtime"]["temporary_license_retry"]
    expected_delay = license_service.bounded_retry_delay_seconds(
        policy,
        attempt_index=retry_count,
        cumulative_wait_seconds=prior_cumulative,
    )
    if last < first or float(delay) != expected_delay:
        message = f"Benchmark license-wait timing is inconsistent: {path}"
        raise ValueError(message)
    if payload["retry_budget_remaining"]:
        next_retry = _benchmark_wait_timestamp(payload["next_retry_at"], label="next_retry_at")
        if float(delay) <= 0.0 or next_retry != last + timedelta(seconds=float(delay)):
            message = f"Benchmark license-wait eligibility is inconsistent: {path}"
            raise ValueError(message)
    elif payload["next_retry_at"] is not None or float(delay) != 0.0:
        message = f"Benchmark exhausted license-wait evidence is inconsistent: {path}"
        raise ValueError(message)
    return payload


def _load_benchmark_license_wait(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
    *,
    run_id: str,
) -> dict[str, Any] | None:
    """Load one benchmark wait record when present."""
    path = _benchmark_license_wait_path(directory, suite, variant, case_position)
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        message = f"Benchmark license-wait evidence is unsafe: {path}"
        raise ValueError(message)
    return _validate_benchmark_license_wait(
        _load_json(path, label="benchmark license-wait evidence"),
        run_id=run_id,
        suite=suite,
        variant=variant,
        case_position=case_position,
        path=path,
    )


def _record_benchmark_license_wait(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
    error: license_service.TemporaryLicenseCapacityError,
    *,
    run_id: str,
    job_id: str,
    license_probe_seconds: float,
    scheduler_queue_seconds: float,
) -> dict[str, Any]:
    """Update compact wait evidence with excluded probe and queue timings."""
    for label, value in (
        ("license_probe_seconds", license_probe_seconds),
        ("scheduler_queue_seconds", scheduler_queue_seconds),
    ):
        if not math.isfinite(value) or value < 0.0:
            message = f"Benchmark {label} must be finite and non-negative."
            raise ValueError(message)
    current = _load_benchmark_license_wait(
        directory,
        suite,
        variant,
        case_position,
        run_id=run_id,
    )
    if current is not None and job_id in current["recent_job_ids"]:
        message = f"Benchmark license wait already includes job {job_id}."
        raise FileExistsError(message)
    now = datetime.now(timezone.utc)
    retry_count = 1 if current is None else int(current["retry_count"]) + 1
    prior_cumulative = 0.0 if current is None else float(current["cumulative_wait_seconds"])
    prior_probe = 0.0 if current is None else float(current["cumulative_probe_seconds"])
    prior_queue = 0.0 if current is None else float(current["cumulative_scheduler_queue_seconds"])
    policy = suite.case_config.execution_values["runtime"]["temporary_license_retry"]
    delay = license_service.bounded_retry_delay_seconds(
        policy,
        attempt_index=retry_count,
        cumulative_wait_seconds=prior_cumulative,
    )
    recent_job_ids = [] if current is None else [str(value) for value in current["recent_job_ids"]]
    recent_job_ids.append(job_id)
    recent_job_ids = recent_job_ids[-_MAX_RECENT_JOB_IDS:]
    work_unit_id = suite.work_unit_id(variant, case_position)
    payload = {
        "schema_kind": "generation_temporary_license_wait",
        "schema_version": 1,
        "benchmark_run_id": run_id,
        "work_unit_id": work_unit_id,
        "suite_digest": suite.suite_digest,
        "variant_id": variant.variant_id,
        "case_position": case_position,
        "case_role": suite.representative_case(case_position).case_role,
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "classification": error.evidence.classification,
        "feature": error.evidence.feature,
        "error_code": error.evidence.license_code,
        "matched_signatures": list(error.evidence.matched_signatures),
        "comsol_exit_code": error.exit_code,
        "solver_progress_started": error.solver_progress_started,
        "expected_exports_exist": error.expected_exports_exist,
        "first_blocked_at": now.isoformat() if current is None else current["first_blocked_at"],
        "last_blocked_at": now.isoformat(),
        "retry_count": retry_count,
        "latest_job_id": job_id,
        "recent_job_ids": recent_job_ids,
        "hostname": socket.gethostname(),
        "raw_excerpt": error.evidence.raw_excerpt,
        "delay_before_next_attempt_seconds": delay,
        "cumulative_wait_seconds": prior_cumulative + delay,
        "cumulative_probe_seconds": prior_probe + license_probe_seconds,
        "cumulative_scheduler_queue_seconds": prior_queue + scheduler_queue_seconds,
        "retry_budget_remaining": delay > 0.0,
        "next_retry_at": (now + timedelta(seconds=delay)).isoformat() if delay > 0.0 else None,
    }
    path = _benchmark_license_wait_path(directory, suite, variant, case_position)
    common.serialization.atomic_write_json(path, payload)
    admitted = _load_benchmark_license_wait(
        directory,
        suite,
        variant,
        case_position,
        run_id=run_id,
    )
    if admitted != payload:
        message = f"Benchmark license-wait evidence did not re-admit: {path}"
        raise RuntimeError(message)
    return payload


def _validate_materialized_case_proof(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Require measured scratch inputs to equal one canonical case proof."""
    if actual != expected:
        message = "Measured benchmark materialization differs from the canonical case proof."
        raise RuntimeError(message)


def _validate_hdf5_scientific_identity(
    hdf5_identity: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    """Require admitted HDF5 output to retain benchmark scientific identity."""
    if hdf5_identity["case_input_id"] != proof["case_input_id"] or hdf5_identity["simulation_case_id"] != proof["simulation_case_id"]:
        message = "Benchmark HDF5 admission changed the canonical scientific identity."
        raise RuntimeError(message)


def _normalized_scheduler_state(value: str) -> str:
    """Return one Slurm state without suffix or annotation."""
    return value.split("+", maxsplit=1)[0].split(maxsplit=1)[0]


def _accounted_root_state(
    scheduler: Mapping[str, Any],
    *,
    job_id: str,
) -> str | None:
    """Return one exact root-job state from scheduler accounting."""
    for line in str(scheduler["sacct"]["output"]).splitlines():
        accounted_job_id, separator, remainder = line.partition("|")
        if separator and accounted_job_id == job_id:
            state, _separator, _remaining_fields = remainder.partition("|")
            return _normalized_scheduler_state(state)
    return None


def _active_benchmark_job_ids(scheduler: Mapping[str, Any]) -> frozenset[str]:
    """Return exact active root job IDs from one admitted scheduler query."""
    result: set[str] = set()
    for line in str(scheduler["squeue"]["output"]).splitlines():
        job_id, separator, _remainder = line.partition("|")
        if separator and _JOB_ID_PATTERN.fullmatch(job_id) is not None:
            result.add(job_id)
    return frozenset(result)


def _scientific_failure_count(attempts: Sequence[Path]) -> int:
    """Count only terminal scientific failures in one admitted attempt chain."""
    _validate_benchmark_attempt_chain(attempts)
    return sum(_load_json(path, label="benchmark work-unit attempt").get("status") == "failed" for path in attempts)


def _work_unit_directory(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> Path:
    """Return one benchmark work-unit evidence directory."""
    return directory / "runs" / suite.execution_id(variant) / suite.work_unit_id(variant, case_position)


def _work_unit_attempts(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> tuple[Path, ...]:
    """Return one sorted immutable scientific-attempt chain."""
    return tuple(
        sorted(
            _work_unit_directory(
                directory,
                suite,
                variant,
                case_position,
            ).glob("attempt-*.json")
        )
    )


def _latest_work_unit_submission(
    manifest: Mapping[str, Any],
    *,
    variant_id: str,
    case_role: str,
) -> Mapping[str, Any] | None:
    """Return the newest submission for one exact case-role work unit."""
    matches = [
        record
        for record in manifest["submission_history"]
        if record.get("role") == "measure" and record.get("variant_id") == variant_id and record.get("case_role") == case_role
    ]
    return None if not matches else matches[-1]


def _persist_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically replace mutable benchmark orchestration state."""
    common.serialization.atomic_write_json(path, dict(manifest))


def _submit_benchmark_work_unit(
    manifest: dict[str, Any],
    suite: CoreBenchmarkSuite,
    *,
    storage: Path,
    logs: Path,
    variant: CoreBenchmarkVariant,
    case_position: int,
) -> str:
    """Submit and durably bind one ordinary case-role benchmark job."""
    run_id = str(manifest["benchmark_run_id"])
    representative = suite.representative_case(case_position)
    command = build_core_benchmark_slurm_command(
        suite,
        run_id=run_id,
        storage_root=storage,
        log_directory=logs,
        variant=variant,
        case_position=case_position,
    )
    submitted_at = _utc_now()
    job_id = _submit(
        command,
        git_commit=str(manifest["git_commit"]),
        run_id=run_id,
    )
    manifest["measured_job_ids"].append(job_id)
    manifest["submission_history"].append(
        {
            "submitted_at": submitted_at,
            "role": "measure",
            "variant_id": variant.variant_id,
            "case_position": case_position,
            "case_role": representative.case_role,
            "work_unit_id": suite.work_unit_id(variant, case_position),
            "command": command,
            "job_id": job_id,
        }
    )
    manifest["state"] = "running"
    _persist_manifest(_manifest_path(run_id, storage_root=storage), manifest)
    return job_id


def _submit_pending(
    manifest: dict[str, Any],
    suite: CoreBenchmarkSuite,
    *,
    storage: Path,
) -> dict[str, Any]:
    """Submit eligible cases from only the first incomplete variant wave."""
    run_id = str(manifest["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    if manifest.get("state") in {"cancel_requested", "force_cancel_requested"}:
        return manifest
    logs = directory / "scheduler"
    logs.mkdir(parents=True, exist_ok=True)
    persisted_job_ids = [str(value) for value in manifest["measured_job_ids"]]
    if any(_JOB_ID_PATTERN.fullmatch(job_id) is None for job_id in persisted_job_ids):
        message = f"Benchmark manifest contains malformed Slurm job IDs: {run_id}"
        raise ValueError(message)
    scheduler = _scheduler_evidence(persisted_job_ids)
    if scheduler["squeue"]["error"] is not None:
        message = f"Could not verify active benchmark jobs before resume: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    active_job_ids = _active_benchmark_job_ids(scheduler)
    proof_paths = tuple(_canonical_case_proof_path(directory, representative) for representative in suite.representative_cases)
    if not all(path.is_file() for path in proof_paths):
        manifest["state"] = "inputs_ready"
        _persist_manifest(manifest_path, manifest)
        return manifest
    proofs = _load_case_proofs(run_id, suite, storage_root=storage)
    records = _result_records(directory, suite)
    _validate_records_against_proof(
        records,
        run_id=run_id,
        manifest=manifest,
        proofs=proofs,
    )
    for wave_position, variant in enumerate(suite.variant_wave_order(), start=1):
        wave_records = [record for record in records if record.get("variant_id") == variant.variant_id]
        if all(record.get("status") == "success" for record in wave_records):
            continue
        exhausted: list[str] = []
        eligible: list[int] = []
        blocked = False
        active = False
        terminal_without_result = False
        for case_position in range(1, suite.representative_case_count + 1):
            representative = suite.representative_case(case_position)
            record = next(item for item in wave_records if item.get("case_position") == case_position)
            if record.get("status") == "success":
                continue
            attempts = _work_unit_attempts(
                directory,
                suite,
                variant,
                case_position,
            )
            if _scientific_failure_count(attempts) >= suite.maximum_work_unit_attempts:
                exhausted.append(suite.work_unit_id(variant, case_position))
                continue
            latest = _latest_work_unit_submission(
                manifest,
                variant_id=variant.variant_id,
                case_role=representative.case_role,
            )
            if latest is not None and str(latest["job_id"]) in active_job_ids:
                active = True
                continue
            wait = _load_benchmark_license_wait(
                directory,
                suite,
                variant,
                case_position,
                run_id=run_id,
            )
            if wait is not None and (not bool(wait["retry_budget_remaining"]) or not license_service.wait_record_is_eligible(wait)):
                blocked = True
                continue
            if latest is not None and record.get("status") == "pending" and wait is None:
                if scheduler["sacct"]["error"] is not None:
                    message = f"Could not reconcile terminal benchmark work through accounting: {scheduler['sacct']['error']}"
                    raise RuntimeError(message)
                state = _accounted_root_state(
                    scheduler,
                    job_id=str(latest["job_id"]),
                )
                if state is None or state in _ACTIVE_SCHEDULER_STATES:
                    active = True
                    continue
                terminal_without_result = True
                continue
            eligible.append(case_position)
        if exhausted or terminal_without_result:
            manifest["state"] = "canary_failed" if wave_position == 1 else "work_unit_failed"
            _persist_manifest(manifest_path, manifest)
            return manifest
        for case_position in eligible:
            _submit_benchmark_work_unit(
                manifest,
                suite,
                storage=storage,
                logs=logs,
                variant=variant,
                case_position=case_position,
            )
        manifest["state"] = "running" if eligible or active else "license_blocked" if blocked else "running"
        _persist_manifest(manifest_path, manifest)
        return manifest
    manifest["state"] = "complete"
    _persist_manifest(manifest_path, manifest)
    return manifest


def _prepare_core_benchmark_locked(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    storage: Path,
    scratch_root: Path | str,
) -> tuple[dict[str, Any], CoreBenchmarkSuite]:
    """Materialize one manifest and canonical input while submission is locked."""
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    manifest_path = directory / "benchmark_manifest.json"
    if manifest_path.exists():
        manifest, suite = load_core_benchmark_manifest(
            run_id,
            storage_root=storage,
        )
        if manifest["state"] in {"cancel_requested", "force_cancel_requested"}:
            scheduler = _scheduler_evidence(manifest["measured_job_ids"])
            if scheduler["squeue"]["error"] is not None:
                message = "Could not verify benchmark cancellation before explicit resume."
                raise RuntimeError(message)
            if scheduler["squeue"]["output"]:
                return manifest, suite
            manifest["state"] = "incomplete"
            _persist_manifest(manifest_path, manifest)
    else:
        suite = load_core_benchmark_suite(path, require_executable=True)
        manifest = {
            "schema_kind": BENCHMARK_RUN_SCHEMA_KIND,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_run_id": run_id,
            "suite_name": suite.suite_name,
            "suite_digest": suite.suite_digest,
            "suite_config": _repository_relative(suite.source_path),
            "git_commit": plan["git_commit"],
            "preflight": plan["preflight"],
            "representative_cases": suite.case_selections(),
            "variants": _variant_records(suite),
            "variant_wave_order": plan["variant_wave_order"],
            "canary_wave": plan["canary_wave"],
            "measurement_waves": plan["measurement_waves"],
            "required_successful_measurements": plan["required_successful_measurements"],
            "resource_contract": suite.resource_contract(),
            "created_at": _utc_now(),
            "state": "preparing_inputs",
            "measured_job_ids": [],
            "submission_history": [],
        }
        _persist_manifest(manifest_path, manifest)
    proof_paths = tuple(_canonical_case_proof_path(directory, representative) for representative in suite.representative_cases)
    if not all(path.is_file() for path in proof_paths):
        try:
            _materialize_core_benchmark_inputs(
                run_id,
                storage_root=storage,
                work_root=scratch_root,
            )
        except Exception:
            manifest["state"] = "input_preparation_failed"
            _persist_manifest(manifest_path, manifest)
            raise
        manifest["state"] = "inputs_ready"
        _persist_manifest(manifest_path, manifest)
    return manifest, suite


def prepare_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Materialize canonical benchmark input without submitting a measurement."""
    plan = plan_core_benchmark(
        path,
        git_commit=git_commit,
        storage_root=storage_root,
        scratch_root=scratch_root,
        comsol_version_output=comsol_version_output,
        comsol_executable_path=comsol_executable_path,
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        manifest, _suite = _prepare_core_benchmark_locked(
            path,
            plan,
            storage=storage,
            scratch_root=scratch_root,
        )
    return manifest


def submit_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    scratch_root: Path | str,
    comsol_version_output: str,
    comsol_executable_path: Path | str,
) -> dict[str, Any]:
    """Reuse canonical login-node input, then submit one measured work unit."""
    plan = plan_core_benchmark(
        path,
        git_commit=git_commit,
        storage_root=storage_root,
        scratch_root=scratch_root,
        comsol_version_output=comsol_version_output,
        comsol_executable_path=comsol_executable_path,
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        manifest, suite = _prepare_core_benchmark_locked(
            path,
            plan,
            storage=storage,
            scratch_root=scratch_root,
        )
        return _submit_pending(
            manifest,
            suite,
            storage=storage,
        )


def resume_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Repair missing input readiness and submit the next measured work unit."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    lock = manifest_path.parent / "submission.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        manifest, suite = load_core_benchmark_manifest(
            run_id,
            storage_root=storage,
        )
        _require_current_checkout(manifest)
        proof_paths = tuple(_canonical_case_proof_path(manifest_path.parent, representative) for representative in suite.representative_cases)
        if not all(path.is_file() for path in proof_paths):
            preflight = _load_core_benchmark_preflight(
                run_id,
                suite=suite,
                git_commit=str(manifest["git_commit"]),
                storage_root=storage,
            )
            scratch = _mapping(
                _mapping(
                    preflight["storage_capabilities"],
                    label="benchmark preflight storage capabilities",
                )["scratch"],
                label="benchmark preflight scratch capability",
            )
            try:
                _materialize_core_benchmark_inputs(
                    run_id,
                    storage_root=storage,
                    work_root=Path(str(scratch["path"])),
                )
            except Exception:
                manifest["state"] = "input_preparation_failed"
                _persist_manifest(manifest_path, manifest)
                raise
            manifest["state"] = "inputs_ready"
            _persist_manifest(manifest_path, manifest)
        return _submit_pending(
            manifest,
            suite,
            storage=storage,
        )


def _canonical_case_proof_path(
    directory: Path,
    representative: CoreBenchmarkRepresentativeCase,
) -> Path:
    """Return one immutable representative-case proof path."""
    return directory / "canonical_cases" / f"{representative.case_role}.json"


def _proof_payload(
    suite: CoreBenchmarkSuite,
    representative: CoreBenchmarkRepresentativeCase,
    prepared: preparation_service.PreparedCase,
    *,
    canonical_input_preparation_seconds: float,
) -> dict[str, Any]:
    """Return one exact scientific byte identity reused across all core waves."""
    if not math.isfinite(canonical_input_preparation_seconds) or canonical_input_preparation_seconds < 0.0:
        message = "Canonical benchmark input preparation timing is invalid."
        raise ValueError(message)
    case = prepared.bundle.case_payload
    return {
        "schema_kind": BENCHMARK_PROOF_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_digest": suite.suite_digest,
        "case_role": representative.case_role,
        "case_id": prepared.bundle.case_id,
        "case_index": representative.case_index,
        "case_input_id": prepared.bundle.case_input_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "scientific_config_digest": case["scientific_config_digest"],
        "case_input_config_digest": case["case_input_config_digest"],
        "material_family": case["material_family"],
        "sampling_regime": case["sampling_regime"],
        "case_seed": case["seed_evidence"]["case_seed"],
        "sampled_values_sha256": common.serialization.canonical_json_sha256(case["sampled_values"]),
        "input_files": case["input_files"],
        "case_payload_sha256": common.serialization.canonical_json_sha256(case),
        "template": case["template"],
        "canonical_input_preparation_seconds": canonical_input_preparation_seconds,
    }


def _cleanup_prepared(
    prepared: preparation_service.PreparedCase,
    *,
    storage: Path,
) -> int:
    """Remove one exact marked benchmark case workspace."""
    return workspace_service.cleanup_case_workspace(
        prepared.work_directory,
        allowed_root=prepared.work_root,
        storage_root=storage,
        expected_run_id=prepared.workspace_run_id,
        expected_case_id=prepared.bundle.case_id,
        allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
    )


def _materialize_core_benchmark_inputs(
    run_id: str,
    *,
    storage_root: Path | str,
    work_root: Path | str,
) -> tuple[Path, ...]:
    """Materialize both canonical inputs on the CPU login node."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    _require_current_checkout(manifest)
    if source_service.required_git_commit() != manifest["git_commit"]:
        message = "Benchmark preparation checkout does not match the launch commit."
        raise RuntimeError(message)
    directory = core_benchmark_directory(run_id, storage_root=storage)
    paths: list[Path] = []
    for case_position in range(1, suite.representative_case_count + 1):
        representative = suite.representative_case(case_position)
        path = _canonical_case_proof_path(directory, representative)
        if path.is_file():
            _load_case_proof(
                run_id,
                suite,
                case_position,
                storage_root=storage,
            )
            paths.append(path)
            continue
        preparation_start = time.monotonic()
        input_service.generate_input_cases(
            suite.case_config,
            1,
            case_start=representative.case_index,
            storage_root=storage,
        )
        prepared: preparation_service.PreparedCase | None = None
        try:
            prepared = preparation_service.prepare_case_work_directory(
                suite.case_config,
                representative.case_index,
                storage_root=storage,
                work_root=work_root,
            )
            proof = _proof_payload(
                suite,
                representative,
                prepared,
                canonical_input_preparation_seconds=(time.monotonic() - preparation_start),
            )
            _write_immutable_json(
                path,
                proof,
                label=f"benchmark canonical-case proof {representative.case_role}",
            )
            paths.append(path)
        finally:
            if prepared is not None:
                _cleanup_prepared(prepared, storage=storage)
    return tuple(paths)


def _load_case_proof(
    run_id: str,
    suite: CoreBenchmarkSuite,
    case_position: int,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate one CPU-materialized representative-case proof."""
    representative = suite.representative_case(case_position)
    path = _canonical_case_proof_path(
        core_benchmark_directory(run_id, storage_root=storage_root),
        representative,
    )
    proof = _load_json(path, label="benchmark canonical-case proof")
    template = proof.get("template")
    preparation_seconds = proof.get("canonical_input_preparation_seconds")
    if (
        proof.get("schema_kind") != BENCHMARK_PROOF_SCHEMA_KIND
        or proof.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or proof.get("suite_digest") != suite.suite_digest
        or proof.get("case_role") != representative.case_role
        or proof.get("case_index") != representative.case_index
        or proof.get("case_id") != suite.case_config.case_id(representative.case_index)
        or proof.get("scientific_config_digest") != suite.case_config.scientific_config_digest
        or proof.get("case_input_config_digest") != suite.case_config.case_input_config_digest
        or not isinstance(template, dict)
        or template.get("sha256") != suite.case_config.template_sha256
        or isinstance(preparation_seconds, bool)
        or not isinstance(preparation_seconds, (int, float))
        or not math.isfinite(float(preparation_seconds))
        or float(preparation_seconds) < 0.0
    ):
        message = f"Benchmark canonical-case proof is invalid: {path}"
        raise ValueError(message)
    for key in (
        "case_input_id",
        "simulation_case_id",
        "sampled_values_sha256",
        "case_payload_sha256",
    ):
        if _SHA256_PATTERN.fullmatch(str(proof.get(key))) is None:
            message = f"Benchmark canonical-case proof has malformed {key}: {path}"
            raise ValueError(message)
    return proof


def _measured_submission(
    manifest: Mapping[str, Any],
    *,
    job_id: str,
    variant_id: str,
    case_role: str,
) -> Mapping[str, Any]:
    """Return the persisted submission for one exact case-role work unit."""
    matches = [
        record
        for record in manifest["submission_history"]
        if record.get("role") == "measure"
        and record.get("job_id") == job_id
        and record.get("variant_id") == variant_id
        and record.get("case_role") == case_role
    ]
    if len(matches) != 1:
        message = "Current benchmark job is not bound to this exact case-role work unit."
        raise RuntimeError(message)
    return matches[0]


def _benchmark_peak_child_memory_bytes() -> int:
    """Return Linux child-process peak resident memory in bytes."""
    peak_kibibytes = resource_usage.getrusage(resource_usage.RUSAGE_CHILDREN).ru_maxrss
    if not math.isfinite(float(peak_kibibytes)) or peak_kibibytes < 0:
        message = "Benchmark child-process peak memory evidence is invalid."
        raise RuntimeError(message)
    return int(peak_kibibytes * 1024)


def _benchmark_scratch_bytes(directory: Path) -> int:
    """Return regular-file bytes in one symlink-free benchmark workspace."""
    total = 0
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            message = f"Benchmark scratch contains a symbolic link: {candidate}"
            raise RuntimeError(message)
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _bounded_solver_log(path: Path) -> dict[str, Any]:
    """Return a digest and bounded excerpt without retaining the full solver log."""
    if not path.is_file() or path.is_symlink():
        message = f"Benchmark solver log is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    raw = path.read_bytes()
    excerpt = raw[:_MAX_BENCHMARK_LOG_EXCERPT_BYTES].decode(
        "utf-8",
        errors="replace",
    )
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "excerpt": excerpt,
        "excerpt_truncated": len(raw) > _MAX_BENCHMARK_LOG_EXCERPT_BYTES,
    }


def _successful_license_evidence(wait: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return compact license evidence while letting success remain authoritative."""
    if wait is None:
        return {
            "license_blocked_submission_count": 0,
            "license_wait_seconds": 0.0,
            "license_probe_seconds": 0.0,
            "scheduler_queue_seconds_before_success": 0.0,
            "detected_feature": None,
            "detected_error_code": None,
            "matched_signatures": [],
            "raw_excerpt": None,
            "successful_artifacts_override_prior_warning": True,
        }
    return {
        "license_blocked_submission_count": int(wait["retry_count"]),
        "license_wait_seconds": float(wait["cumulative_wait_seconds"]),
        "license_probe_seconds": float(wait["cumulative_probe_seconds"]),
        "scheduler_queue_seconds_before_success": float(wait["cumulative_scheduler_queue_seconds"]),
        "detected_feature": wait["feature"],
        "detected_error_code": wait["error_code"],
        "matched_signatures": list(wait["matched_signatures"]),
        "raw_excerpt": wait["raw_excerpt"],
        "successful_artifacts_override_prior_warning": True,
    }


def _total_controller_elapsed_seconds(
    manifest: Mapping[str, Any],
    *,
    variant_id: str,
    case_role: str,
    completion_time: str,
) -> float:
    """Measure from the first submission through terminal successful evidence."""
    submissions = [
        _timestamp(record["submitted_at"], label="benchmark submitted_at")
        for record in manifest["submission_history"]
        if record.get("role") == "measure" and record.get("variant_id") == variant_id and record.get("case_role") == case_role
    ]
    if not submissions:
        message = "Benchmark work unit lacks a persisted submission timestamp."
        raise RuntimeError(message)
    completed = _timestamp(completion_time, label="benchmark completion_time")
    elapsed = (completed - min(submissions)).total_seconds()
    if elapsed < 0.0:
        message = "Benchmark controller elapsed timing is negative."
        raise RuntimeError(message)
    return elapsed


def run_core_benchmark_case(
    run_id: str,
    variant_id: str,
    case_role: str,
    *,
    storage_root: Path | str,
    work_root: Path | str,
) -> dict[str, Any]:
    """Run one case-role measurement and retain compact performance evidence."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    _require_current_checkout(manifest)
    start_time = _slurm_scheduler_start_time()
    variant = suite.variant(variant_id)
    case_position = suite.case_position(case_role)
    representative = suite.representative_case(case_position)
    work_unit_id = suite.work_unit_id(variant, case_position)
    if source_service.required_git_commit() != manifest["git_commit"]:
        message = "Benchmark worker checkout does not match the launch commit."
        raise RuntimeError(message)
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Benchmark case work unit requires one numeric SLURM_JOB_ID."
        raise RuntimeError(message)
    submission = _measured_submission(
        manifest,
        job_id=job_id,
        variant_id=variant.variant_id,
        case_role=representative.case_role,
    )
    submit_time = str(submission["submitted_at"])
    if allocated_cpus is None or not allocated_cpus.isdigit() or int(allocated_cpus) != variant.cores_per_case:
        message = f"Benchmark Slurm allocation must exactly match cores_per_case; requested={variant.cores_per_case}, allocated={allocated_cpus!r}."
        raise RuntimeError(message)
    proof = _load_case_proof(
        run_id,
        suite,
        case_position,
        storage_root=storage,
    )
    directory = core_benchmark_directory(run_id, storage_root=storage)
    work_unit_directory = directory / "runs" / suite.execution_id(variant) / work_unit_id
    work_unit_directory.mkdir(parents=True, exist_ok=True)
    success_path = work_unit_directory / "success.json"
    lock_path = work_unit_directory / "execution.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        attempts = tuple(sorted(work_unit_directory.glob("attempt-*.json")))
        _validate_benchmark_attempt_chain(attempts)
        if success_path.exists():
            existing_success = _load_json(
                success_path,
                label="benchmark case success",
            )
            _validate_success_result(
                existing_success,
                suite=suite,
                variant=variant,
                case_position=case_position,
            )
            return existing_success
        attempt_number = len(attempts) + 1
        previous_attempt = _benchmark_previous_attempt_reference(attempts)
        prior_wait = _load_benchmark_license_wait(
            directory,
            suite,
            variant,
            case_position,
            run_id=run_id,
        )
        license_evidence = _successful_license_evidence(prior_wait)
        attempt_path = work_unit_directory / f"attempt-{attempt_number:04d}.json"
        prepared: preparation_service.PreparedCase | None = None
        solver_start: float | None = None
        success: dict[str, Any]
        try:
            prepared = preparation_service.prepare_case_work_directory(
                suite.case_config,
                representative.case_index,
                storage_root=storage,
                work_root=work_root,
            )
            actual_proof = _proof_payload(
                suite,
                representative,
                prepared,
                canonical_input_preparation_seconds=float(proof["canonical_input_preparation_seconds"]),
            )
            _validate_materialized_case_proof(actual_proof, proof)
            solver_start = time.monotonic()
            result = runtime_service.execute_prepared_case(
                suite.case_config,
                prepared,
                cores_per_case=variant.cores_per_case,
                worker_slot=case_position - 1,
                scheduler_kind="slurm",
                allocated_node=os.environ.get(
                    "SLURMD_NODENAME",
                    socket.gethostname(),
                ),
            )
            publication_start = time.monotonic()
            hdf5_identity = storage_service.validate_case_hdf5(
                result.canonical_case.path,
                expected_profile=profiles.TRANSIENT_DRYING_PROFILE,
            )
            publication_seconds = time.monotonic() - publication_start
            _validate_hdf5_scientific_identity(hdf5_identity, proof)
            command_np = result.command[result.command.index("-np") + 1]
            completion_time = _utc_now()
            scheduler_timing = _scheduler_timing(
                submit_time=submit_time,
                start_time=start_time,
                completion_time=completion_time,
            )
            scheduler_queue_seconds = float(scheduler_timing["queue_wait_s"]) + float(license_evidence["scheduler_queue_seconds_before_success"])
            timings = {
                "scheduler_queue_seconds": scheduler_queue_seconds,
                "license_wait_seconds": float(license_evidence["license_wait_seconds"]),
                "license_probe_seconds": float(license_evidence["license_probe_seconds"]),
                "canonical_input_preparation_seconds": float(proof["canonical_input_preparation_seconds"]),
                "comsol_process_seconds": float(result.timing["runtime_s"]),
                "export_conversion_seconds": float(result.timing["export_conversion_s"]),
                "publication_seconds": publication_seconds,
                "total_controller_elapsed_seconds": _total_controller_elapsed_seconds(
                    manifest,
                    variant_id=variant.variant_id,
                    case_role=representative.case_role,
                    completion_time=completion_time,
                ),
            }
            success = {
                "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "status": "success",
                "recorded_at": completion_time,
                "benchmark_run_id": run_id,
                "suite_digest": suite.suite_digest,
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "case_position": case_position,
                "case_role": representative.case_role,
                "work_unit_id": work_unit_id,
                "attempt": attempt_number,
                "previous_attempt": previous_attempt,
                "git_commit": manifest["git_commit"],
                "case_input_id": proof["case_input_id"],
                "simulation_case_id": proof["simulation_case_id"],
                "scientific_config_digest": proof["scientific_config_digest"],
                "template_sha256": suite.case_config.template_sha256,
                "benchmark_preflight_sha256": manifest["preflight"]["receipt_sha256"],
                "cores_per_case": variant.cores_per_case,
                "resource": {
                    "node": os.environ.get(
                        "SLURMD_NODENAME",
                        socket.gethostname(),
                    ),
                    "partition": os.environ.get("SLURM_JOB_PARTITION", suite.partition),
                    "requested_cpus": variant.cores_per_case,
                    "allocated_cpus": int(allocated_cpus),
                    "comsol_np": int(command_np),
                    "slurm_job_id": job_id,
                    "peak_memory_bytes": _benchmark_peak_child_memory_bytes(),
                    "peak_scratch_bytes": int(result.timing["scratch_peak_bytes"]),
                },
                "scheduler_timing": scheduler_timing,
                "solver_interval": {
                    "started_at": result.timing["started_at"],
                    "ended_at": result.timing["ended_at"],
                },
                "timings_seconds": timings,
                "license": license_evidence,
                "solver_log": _bounded_solver_log(result.solver_log),
                "hdf5": {
                    "sha256": common.serialization.file_sha256(result.canonical_case.path),
                    "size_bytes": result.canonical_case.path.stat().st_size,
                    "identity": hdf5_identity,
                    "retained_as_scientific_case": False,
                },
            }
            _write_immutable_json(attempt_path, success, label="benchmark attempt")
            _write_immutable_json(
                success_path,
                success,
                label="benchmark case success",
            )
        except BaseException as error:
            completion_time = _utc_now()
            scheduler_timing = _scheduler_timing(
                submit_time=submit_time,
                start_time=start_time,
                completion_time=completion_time,
            )
            if isinstance(
                error,
                license_service.TemporaryLicenseCapacityError,
            ) and bool(suite.case_config.execution_values["runtime"]["temporary_license_retry"]["enabled"]):
                if solver_start is None:
                    message = "Temporary license capacity was reported before solver launch."
                    raise RuntimeError(message) from error
                _record_benchmark_license_wait(
                    directory,
                    suite,
                    variant,
                    case_position,
                    error,
                    run_id=run_id,
                    job_id=job_id,
                    license_probe_seconds=time.monotonic() - solver_start,
                    scheduler_queue_seconds=float(scheduler_timing["queue_wait_s"]),
                )
                raise
            failure = {
                "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "status": "failed",
                "recorded_at": completion_time,
                "benchmark_run_id": run_id,
                "suite_digest": suite.suite_digest,
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "case_position": case_position,
                "case_role": representative.case_role,
                "work_unit_id": work_unit_id,
                "attempt": attempt_number,
                "previous_attempt": previous_attempt,
                "git_commit": manifest["git_commit"],
                "case_input_id": proof["case_input_id"],
                "simulation_case_id": proof["simulation_case_id"],
                "scientific_config_digest": proof["scientific_config_digest"],
                "template_sha256": suite.case_config.template_sha256,
                "benchmark_preflight_sha256": manifest["preflight"]["receipt_sha256"],
                "cores_per_case": variant.cores_per_case,
                "resource": {
                    "node": os.environ.get(
                        "SLURMD_NODENAME",
                        socket.gethostname(),
                    ),
                    "partition": os.environ.get("SLURM_JOB_PARTITION", suite.partition),
                    "requested_cpus": variant.cores_per_case,
                    "allocated_cpus": int(allocated_cpus),
                    "comsol_np": variant.cores_per_case,
                    "slurm_job_id": job_id,
                    "peak_memory_bytes": _benchmark_peak_child_memory_bytes(),
                    "peak_scratch_bytes": (0 if prepared is None else _benchmark_scratch_bytes(prepared.work_directory)),
                },
                "scheduler_timing": scheduler_timing,
                "timings_seconds": {
                    "scheduler_queue_seconds": float(scheduler_timing["queue_wait_s"])
                    + float(license_evidence["scheduler_queue_seconds_before_success"]),
                    "license_wait_seconds": float(license_evidence["license_wait_seconds"]),
                    "license_probe_seconds": float(license_evidence["license_probe_seconds"]),
                    "canonical_input_preparation_seconds": float(proof["canonical_input_preparation_seconds"]),
                    "comsol_process_seconds": None,
                    "export_conversion_seconds": None,
                    "publication_seconds": 0.0,
                    "total_controller_elapsed_seconds": _total_controller_elapsed_seconds(
                        manifest,
                        variant_id=variant.variant_id,
                        case_role=representative.case_role,
                        completion_time=completion_time,
                    ),
                },
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            if not attempt_path.exists():
                _write_immutable_json(
                    attempt_path,
                    failure,
                    label="benchmark failed attempt",
                )
            raise
        finally:
            if prepared is not None:
                _cleanup_prepared(prepared, storage=storage)
        return success


def _result_records(
    directory: Path,
    suite: CoreBenchmarkSuite,
) -> list[dict[str, Any]]:
    """Return one latest terminal-or-pending record per configured case_position."""
    records: list[dict[str, Any]] = []
    for variant in suite.variants:
        for case_position in range(1, suite.representative_case_count + 1):
            work_unit_id = suite.work_unit_id(variant, case_position)
            work_unit_directory = directory / "runs" / suite.execution_id(variant) / work_unit_id
            success_path = work_unit_directory / "success.json"
            if success_path.is_file():
                record = _load_json(
                    success_path,
                    label="benchmark case_position success",
                )
                _validate_success_result(
                    record,
                    suite=suite,
                    variant=variant,
                    case_position=case_position,
                )
                records.append(record)
                continue
            attempts = tuple(sorted(work_unit_directory.glob("attempt-*.json")))
            wait = _load_benchmark_license_wait(
                directory,
                suite,
                variant,
                case_position,
                run_id=directory.name,
            )
            attempt = (
                None
                if not attempts
                else _load_json(
                    attempts[-1],
                    label="benchmark case_position attempt",
                )
            )
            if wait is not None and (
                attempt is None
                or _benchmark_wait_timestamp(
                    wait["last_blocked_at"],
                    label="last_blocked_at",
                )
                > _benchmark_wait_timestamp(
                    attempt["recorded_at"],
                    label="attempt recorded_at",
                )
            ):
                records.append(
                    {
                        "status": "pending",
                        "variant_id": variant.variant_id,
                        "execution_id": suite.execution_id(variant),
                        "case_position": case_position,
                        "case_role": suite.representative_case(case_position).case_role,
                        "work_unit_id": work_unit_id,
                        "cores_per_case": variant.cores_per_case,
                        "temporary_license_retry": wait,
                    }
                )
                continue
            if attempt is not None:
                if attempt.get("status") == "failed":
                    _validate_failure_result(
                        attempt,
                        suite=suite,
                        variant=variant,
                        case_position=case_position,
                        attempts=attempts,
                    )
                    records.append(attempt)
                    continue
                if attempt.get("status") == "success":
                    _validate_success_result(
                        attempt,
                        suite=suite,
                        variant=variant,
                        case_position=case_position,
                    )
                else:
                    message = f"Benchmark attempt has an unsupported status: {attempts[-1]}"
                    raise ValueError(message)
            records.append(
                {
                    "status": "pending",
                    "variant_id": variant.variant_id,
                    "execution_id": suite.execution_id(variant),
                    "case_position": case_position,
                    "case_role": suite.representative_case(case_position).case_role,
                    "work_unit_id": work_unit_id,
                    "cores_per_case": variant.cores_per_case,
                }
            )
    return records


def _load_case_proofs(
    run_id: str,
    suite: CoreBenchmarkSuite,
    *,
    storage_root: Path | str,
) -> dict[str, dict[str, Any]]:
    """Load both canonical proofs keyed by exact representative-case role."""
    return {
        representative.case_role: _load_case_proof(
            run_id,
            suite,
            case_position,
            storage_root=storage_root,
        )
        for case_position, representative in enumerate(
            suite.representative_cases,
            start=1,
        )
    }


def _validate_records_against_proof(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind each terminal result to its exact nominal or natural proof."""
    if set(proofs) != {item.case_role for item in manifest_case_roles(manifest)}:
        message = "Benchmark canonical proof roles do not match the run manifest."
        raise ValueError(message)
    for record in records:
        if record.get("status") == "pending" and "benchmark_run_id" not in record:
            continue
        case_role = record.get("case_role")
        proof = proofs.get(str(case_role))
        if proof is None:
            message = f"Benchmark result has an unknown representative case role: {case_role!r}."
            raise ValueError(message)
        expected = {
            "benchmark_run_id": run_id,
            "git_commit": manifest["git_commit"],
            "case_role": case_role,
            "case_input_id": proof["case_input_id"],
            "simulation_case_id": proof["simulation_case_id"],
            "scientific_config_digest": proof["scientific_config_digest"],
            "template_sha256": proof["template"]["sha256"],
            "benchmark_preflight_sha256": manifest["preflight"]["receipt_sha256"],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            message = f"Benchmark result is not bound to the canonical proof: {record.get('work_unit_id')!r}."
            raise ValueError(message)


def manifest_case_roles(
    manifest: Mapping[str, Any],
) -> tuple[CoreBenchmarkRepresentativeCase, ...]:
    """Return minimal typed case roles from an already-admitted run manifest."""
    values = manifest["representative_cases"]
    if not isinstance(values, list):
        message = "Benchmark manifest representative_cases is malformed."
        raise TypeError(message)
    return tuple(
        CoreBenchmarkRepresentativeCase(
            case_role=str(value["case_role"]),
            case_index=int(value["case_index"]),
        )
        for value in values
        if isinstance(value, dict)
    )


def _production_interpretation(suite: CoreBenchmarkSuite) -> dict[str, Any]:
    """Resolve current production count and authoritative core-setting owner."""
    campaign = config_service.load_campaign_config(
        suite.production_campaign_path,
        require_executable=False,
    )
    execution = _load_yaml(
        suite.production_cores_config_path,
        label="benchmark production execution config",
    )
    cluster = _mapping(execution.get("cluster"), label="benchmark production execution cluster")
    cores = _positive_integer(
        cluster.get("cores_per_case"),
        label="benchmark production cores_per_case",
    )
    if cores != campaign.execution_values["cluster"]["cores_per_case"]:
        message = "Current production campaign and core-setting config disagree."
        raise ValueError(message)
    return {
        "campaign_config": _repository_relative(suite.production_campaign_path),
        "campaign_total_cases": campaign.total_case_count,
        "current_production_cores_per_case": cores,
        "current_estimated_cases_per_node": suite.cores_per_node // cores,
        "current_max_running_cases": campaign.execution_values["submission"]["max_running_cases"],
        "cores_config": _repository_relative(suite.production_cores_config_path),
        "cores_key": suite.production_cores_key,
    }


def _solver_overlap_metrics(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, bool]:
    """Return peak successful solver concurrency and two-case overlap."""
    intervals = [
        (
            _timestamp(record["solver_interval"]["started_at"], label="solver started_at"),
            _timestamp(record["solver_interval"]["ended_at"], label="solver ended_at"),
        )
        for record in records
    ]
    if not intervals:
        return 0, False
    peak = max(sum(start <= instant < end for start, end in intervals) for instant in (start for start, _end in intervals))
    overlapped = len(intervals) == len(_BENCHMARK_REPRESENTATIVE_CASE_ROLES) and max(intervals[0][0], intervals[1][0]) < min(
        intervals[0][1], intervals[1][1]
    )
    return max(1, peak), overlapped


def _projected_resource_feasibility(
    estimate: int,
    limit: int | None,
) -> str:
    """Classify one projected node resource against an authoritative limit."""
    if limit is None:
        return "operator_review_required"
    return "pass" if estimate <= limit else "fail"


def _variant_resource_feasibility(memory: str, scratch: str) -> str:
    """Combine independent memory and scratch feasibility classifications."""
    if "fail" in {memory, scratch}:
        return "fail"
    if "operator_review_required" in {memory, scratch}:
        return "operator_review_required"
    return "pass"


def _ordered_unique_text(values: Sequence[object]) -> list[str]:
    """Return non-empty text values once in stable encounter order."""
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value in result:
            continue
        result.append(value)
    return result


def summarize_core_benchmark_results(
    suite: CoreBenchmarkSuite,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate separated runtime, throughput, resource, and license metrics."""
    expected_count = len(suite.variants) * suite.representative_case_count
    if len(records) != expected_count:
        message = f"Benchmark summary requires {expected_count} work-unit records, got {len(records)}."
        raise ValueError(message)
    production = _production_interpretation(suite)
    by_variant: list[dict[str, Any]] = []
    for variant in suite.variants:
        selected = [record for record in records if record.get("variant_id") == variant.variant_id]
        if len(selected) != suite.representative_case_count:
            message = f"Benchmark records do not cover both cases for {variant.variant_id!r}."
            raise ValueError(message)
        successes = [record for record in selected if record.get("status") == "success"]
        failures = [record for record in selected if record.get("status") == "failed"]
        pending = [record for record in selected if record.get("status") == "pending"]
        solve_times = [float(record["timings_seconds"]["comsol_process_seconds"]) for record in successes]
        if any(not math.isfinite(value) or value <= 0.0 for value in solve_times):
            message = f"Benchmark summary received invalid successful COMSOL timings for {variant.variant_id!r}."
            raise ValueError(message)
        queue_wait = sum(float(record["timings_seconds"]["scheduler_queue_seconds"]) for record in successes)
        license_wait = sum(float(record["timings_seconds"]["license_wait_seconds"]) for record in successes)
        license_probe = sum(float(record["timings_seconds"]["license_probe_seconds"]) for record in successes)
        conversion = sum(float(record["timings_seconds"]["export_conversion_seconds"]) for record in successes)
        publication = sum(float(record["timings_seconds"]["publication_seconds"]) for record in successes)
        preparation = sum(float(record["timings_seconds"]["canonical_input_preparation_seconds"]) for record in successes)
        controller_elapsed = sum(float(record["timings_seconds"]["total_controller_elapsed_seconds"]) for record in successes)
        operational_values = (
            queue_wait,
            license_wait,
            license_probe,
            conversion,
            publication,
            preparation,
            controller_elapsed,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in operational_values):
            message = f"Benchmark summary received invalid separated timing evidence for {variant.variant_id!r}."
            raise ValueError(message)
        license_records = [record["license"] for record in successes]
        blocked_count = sum(int(value["license_blocked_submission_count"]) for value in license_records)
        resources = [record["resource"] for record in successes]
        peak_memory = max((int(value["peak_memory_bytes"]) for value in resources), default=0)
        peak_scratch = max((int(value["peak_scratch_bytes"]) for value in resources), default=0)
        cases_per_node = suite.cores_per_node // variant.cores_per_case
        estimated_memory = cases_per_node * peak_memory
        estimated_scratch = cases_per_node * peak_scratch
        memory_feasibility = _projected_resource_feasibility(estimated_memory, suite.node_memory_limit_bytes)
        scratch_feasibility = _projected_resource_feasibility(estimated_scratch, suite.node_scratch_limit_bytes)
        resource_feasibility = _variant_resource_feasibility(memory_feasibility, scratch_feasibility)
        concurrency, overlapped = _solver_overlap_metrics(successes)
        if solve_times:
            median_solve = float(statistics.median(solve_times))
            median_core_hours = variant.cores_per_case * median_solve / 3600.0
            node_throughput = cases_per_node * 3600.0 / median_solve
            minimum_solve = min(solve_times)
            maximum_solve = max(solve_times)
        else:
            median_solve = None
            median_core_hours = None
            node_throughput = None
            minimum_solve = None
            maximum_solve = None
        raw_excerpts = _ordered_unique_text([value["raw_excerpt"] for value in license_records])
        by_variant.append(
            {
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "cores_per_case": variant.cores_per_case,
                "successful_measurement_count": len(successes),
                "failed_measurement_count": len(failures),
                "pending_measurement_count": len(pending),
                "individual_comsol_process_seconds": solve_times,
                "median_comsol_process_seconds": median_solve,
                "minimum_comsol_process_seconds": minimum_solve,
                "maximum_comsol_process_seconds": maximum_solve,
                "median_core_hours_per_case": median_core_hours,
                "estimated_cases_per_node": cases_per_node,
                "estimated_cases_per_node_hour": node_throughput,
                "throughput_label": "compute-only estimated node throughput",
                "peak_memory_per_case_bytes": peak_memory,
                "estimated_peak_memory_per_node_bytes": estimated_memory,
                "peak_scratch_per_case_bytes": peak_scratch,
                "estimated_peak_scratch_per_node_bytes": estimated_scratch,
                "memory_feasibility": memory_feasibility,
                "scratch_feasibility": scratch_feasibility,
                "resource_feasibility": resource_feasibility,
                "scheduler_queue_seconds": queue_wait,
                "license_wait_seconds": license_wait,
                "license_probe_seconds": license_probe,
                "canonical_input_preparation_seconds": preparation,
                "export_conversion_seconds": conversion,
                "publication_seconds": publication,
                "total_controller_elapsed_seconds": controller_elapsed,
                "license_blocked_submission_count": blocked_count,
                "detected_features": _ordered_unique_text([value["detected_feature"] for value in license_records]),
                "detected_comsol_flexnet_codes": _ordered_unique_text([value["detected_error_code"] for value in license_records]),
                "matched_signatures": _ordered_unique_text([signature for value in license_records for signature in value["matched_signatures"]]),
                "bounded_raw_excerpts": [value[:_MAX_BENCHMARK_LOG_EXCERPT_BYTES] for value in raw_excerpts],
                "observed_peak_solver_concurrency": concurrency,
                "requested_cases_overlapped_in_solver_execution": overlapped,
            }
        )
    complete = [
        record
        for record in by_variant
        if record["successful_measurement_count"] == suite.representative_case_count
        and record["failed_measurement_count"] == 0
        and record["pending_measurement_count"] == 0
    ]
    fastest_measurement = min(
        (
            (runtime, int(record["cores_per_case"]), str(record["variant_id"]))
            for record in complete
            for runtime in record["individual_comsol_process_seconds"]
        ),
        default=None,
    )
    fastest_cores = None if fastest_measurement is None else fastest_measurement[1]
    core_efficient = (
        min(
            complete,
            key=lambda record: (
                float(record["median_core_hours_per_case"]),
                int(record["cores_per_case"]),
            ),
        )
        if complete
        else None
    )
    lowest_core_hours_cores = None if core_efficient is None else int(core_efficient["cores_per_case"])
    feasible = [record for record in complete if record["resource_feasibility"] != "fail"]
    if feasible:
        best_throughput = max(float(record["estimated_cases_per_node_hour"]) for record in feasible)
        throughput_ties = [record for record in feasible if float(record["estimated_cases_per_node_hour"]) >= 0.95 * best_throughput]
        recommended = min(
            throughput_ties,
            key=lambda record: (
                float(record["median_core_hours_per_case"]),
                int(record["cores_per_case"]),
            ),
        )
    else:
        recommended = None
    recommended_cores = None if recommended is None else int(recommended["cores_per_case"])
    recommended_cases_per_node = None if recommended is None else int(recommended["estimated_cases_per_node"])
    incomplete_license_concurrency = any(
        record["license_blocked_submission_count"] > 0 and not record["requested_cases_overlapped_in_solver_execution"] for record in by_variant
    )
    all_overlap_observed = bool(by_variant) and all(record["requested_cases_overlapped_in_solver_execution"] for record in by_variant)
    if incomplete_license_concurrency:
        license_qualification = "compute recommendation valid; concurrent-license observation incomplete"
    elif all_overlap_observed:
        license_qualification = "compute recommendation valid; concurrent-license execution observed"
    else:
        license_qualification = "compute recommendation valid; requested solver overlap not observed"
    production["recommended_difference_from_current_cores_per_case"] = (
        None if recommended_cores is None else recommended_cores - int(production["current_production_cores_per_case"])
    )
    production["recommended_differs_from_current"] = (
        None if recommended_cores is None else recommended_cores != int(production["current_production_cores_per_case"])
    )
    recommended_detail = (
        None
        if recommended is None
        else {
            "variant_id": recommended["variant_id"],
            "cores_per_case": recommended_cores,
            "estimated_cases_per_node": recommended_cases_per_node,
            "estimated_cases_per_node_hour": recommended["estimated_cases_per_node_hour"],
            "median_comsol_process_seconds": recommended["median_comsol_process_seconds"],
            "median_core_hours_per_case": recommended["median_core_hours_per_case"],
            "resource_feasibility": recommended["resource_feasibility"],
            "license_qualification": license_qualification,
            "proposed_configuration": {
                "cores_per_case": recommended_cores,
                "cases_per_node": recommended_cases_per_node,
                "max_running_cases": production["current_max_running_cases"],
            },
            "manual_review_required": True,
        }
    )
    canary = suite.canary_variant()
    return {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "benchmark_mode": "core_selection",
        "representative_cases": suite.case_selections(),
        "cases_per_variant": suite.representative_case_count,
        "required_successful_measurements": expected_count,
        "cores_per_node": suite.cores_per_node,
        "variants": by_variant,
        "fastest_single_case_cores": fastest_cores,
        "fastest_single_case": (
            None
            if fastest_measurement is None
            else {
                "cores_per_case": fastest_measurement[1],
                "variant_id": fastest_measurement[2],
                "comsol_process_seconds": fastest_measurement[0],
            }
        ),
        "lowest_core_hours_cores": lowest_core_hours_cores,
        "lowest_core_hours": (
            None
            if core_efficient is None
            else {
                "cores_per_case": lowest_core_hours_cores,
                "variant_id": core_efficient["variant_id"],
                "median_core_hours_per_case": core_efficient["median_core_hours_per_case"],
            }
        ),
        "recommended_cores_per_case": recommended_cores,
        "recommended_estimated_cases_per_node": recommended_cases_per_node,
        "recommended_production": recommended_detail,
        "recommendation_basis": (
            "maximize compute-only estimated cases per node-hour among resource-feasible variants; "
            "within 5% prefer lower median core-hours, then fewer cores"
        ),
        "timing_contract": {
            "primary_runtime": "successful comsol_process_seconds only",
            "excluded_from_ranking": [
                "scheduler_queue_seconds",
                "license_wait_seconds",
                "license_probe_seconds",
                "canonical_input_preparation_seconds",
                "export_conversion_seconds",
                "publication_seconds",
                "total_controller_elapsed_seconds",
            ],
            "license_only_attempts_contribute_successful_runtime_observations": 0,
        },
        "resource_limits": {
            "node_memory_limit_bytes": suite.node_memory_limit_bytes,
            "node_scratch_limit_bytes": suite.node_scratch_limit_bytes,
            "missing_limit_policy": "operator_review_required",
        },
        "license_qualification": license_qualification,
        "production_interpretation": production,
        "production_configuration_modified": False,
        "dataset_membership": "none",
        "canary_wave": {
            "variant_id": canary.variant_id,
            "cores_per_case": canary.cores_per_case,
            "case_roles": [item.case_role for item in suite.representative_cases],
            "included_in_final_measurements": True,
            "additional_canary_work_units": 0,
        },
    }


def _scheduler_command_evidence(command: Sequence[str]) -> dict[str, str | None]:
    """Run one optional fixed scheduler query and retain failure evidence."""
    try:
        result = subprocess.run(  # noqa: S603 -- fixed Slurm argv and numeric IDs
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {"output": "", "error": str(error)}
    return {
        "output": result.stdout.strip(),
        "error": (None if result.returncode == 0 else result.stderr.strip() or f"exit status {result.returncode}"),
    }


def _scheduler_evidence(job_ids: Sequence[str]) -> dict[str, Any]:
    """Return optional current Slurm queue and accounting evidence."""
    if not job_ids:
        return {
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "", "error": None},
        }
    selection = ",".join(job_ids)
    commands = {
        "squeue": [
            "squeue",
            "--noheader",
            f"--jobs={selection}",
            "--format=%i|%T|%R",
        ],
        "sacct": [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            selection,
            ("--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,TotalCPU,MaxRSS,NodeList,AllocCPUS,Partition"),
        ],
    }
    return {name: _scheduler_command_evidence(command) for name, command in commands.items()}


def _active_benchmark_jobs(
    manifest: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return exact pending and running job IDs owned by one benchmark."""
    queue = _mapping(scheduler["squeue"], label="benchmark squeue evidence")
    if queue.get("error") is not None:
        message = f"Could not query active core benchmark jobs: {queue['error']}"
        raise RuntimeError(message)
    owned = {str(value) for value in manifest["measured_job_ids"]}
    pending: list[str] = []
    running: list[str] = []
    for line in str(queue.get("output", "")).splitlines():
        try:
            job_id, state, _reason = line.split("|", maxsplit=2)
        except ValueError as error:
            message = f"Malformed active benchmark scheduler record: {line!r}."
            raise ValueError(message) from error
        if job_id not in owned or _JOB_ID_PATTERN.fullmatch(job_id) is None:
            message = f"Scheduler returned an unowned benchmark job: {job_id!r}."
            raise ValueError(message)
        if state.strip().upper() in {"PENDING", "CONFIGURING"}:
            pending.append(job_id)
        else:
            running.append(job_id)
    return pending, running


def cancel_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
    force: bool = False,
) -> dict[str, Any]:
    """Persist and request cancellation of exact benchmark-owned Slurm jobs."""
    if not isinstance(force, bool):
        message = "force must be boolean."
        raise TypeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    lock_path = manifest_path.parent / "submission.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        manifest, _suite = load_core_benchmark_manifest(
            run_id,
            storage_root=storage,
        )
        job_ids = [str(value) for value in manifest["measured_job_ids"]]
        if any(_JOB_ID_PATTERN.fullmatch(job_id) is None for job_id in job_ids):
            message = f"Benchmark manifest contains malformed Slurm job IDs: {run_id}"
            raise ValueError(message)
        scheduler = _scheduler_evidence(job_ids)
        pending_ids, running_ids = _active_benchmark_jobs(manifest, scheduler)
        active_ids = [*pending_ids, *running_ids]
        commands: list[list[str]] = []
        if force and active_ids:
            commands.append(["scancel", "--signal=KILL", "--full", *active_ids])
        elif not force:
            if pending_ids:
                commands.append(["scancel", *pending_ids])
            if running_ids:
                commands.append(["scancel", "--signal=TERM", "--batch", *running_ids])

        receipt_path = manifest_path.parent / "cancellations.json"
        attempts: list[dict[str, Any]] = []
        if receipt_path.exists():
            receipt = _load_json(
                receipt_path,
                label="benchmark cancellation receipt",
            )
            if (
                receipt.get("schema_kind") != BENCHMARK_CANCELLATION_SCHEMA_KIND
                or receipt.get("schema_version") != 1
                or receipt.get("benchmark_run_id") != run_id
                or not isinstance(receipt.get("attempts"), list)
            ):
                message = f"Benchmark cancellation receipt is malformed: {receipt_path}"
                raise ValueError(message)
            attempts = list(receipt["attempts"])
        attempt: dict[str, Any] = {
            "recorded_at": _utc_now(),
            "mode": "force" if force else "graceful",
            "pending_job_ids": pending_ids,
            "running_job_ids": running_ids,
            "commands": [],
        }
        receipt = {
            "schema_kind": BENCHMARK_CANCELLATION_SCHEMA_KIND,
            "schema_version": 1,
            "benchmark_run_id": run_id,
            "attempts": [*attempts, attempt],
        }
        common.serialization.atomic_write_json(receipt_path, receipt)
        manifest["state"] = "force_cancel_requested" if force else "cancel_requested"
        _persist_manifest(manifest_path, manifest)

        command_results: list[dict[str, Any]] = []
        failures: list[str] = []
        for command in commands:
            result = subprocess.run(  # noqa: S603 -- validated persisted numeric Slurm IDs
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            command_results.append(
                {
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
            )
            if result.returncode != 0:
                failures.append(result.stderr.strip() or f"{command[0]} exit {result.returncode}")
        attempt["commands"] = command_results
        common.serialization.atomic_write_json(receipt_path, receipt)
        if failures:
            message = f"Slurm benchmark cancellation failed after its request was persisted: {failures}"
            raise RuntimeError(message)
        return receipt


def core_benchmark_status(
    run_id: str,
    *,
    storage_root: Path | str,
    query_scheduler: bool = True,
) -> dict[str, Any]:
    """Reconstruct precise wave-aware work-unit and scheduler state."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    records = _result_records(directory, suite)
    success_count = sum(record["status"] == "success" for record in records)
    failure_count = sum(record["status"] == "failed" for record in records)
    pending_count = sum(record["status"] == "pending" for record in records)
    job_ids = [str(value) for value in manifest["measured_job_ids"]]
    scheduler = (
        _scheduler_evidence(job_ids)
        if query_scheduler
        else {
            "squeue": {"output": "", "error": None},
            "sacct": {"output": "", "error": None},
        }
    )
    if query_scheduler and scheduler["squeue"]["error"] is not None:
        message = f"Could not query active core benchmark jobs: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    active_job_ids = _active_benchmark_job_ids(scheduler) if query_scheduler else frozenset()
    active_work_units = [
        str(submission["work_unit_id"])
        for submission in manifest["submission_history"]
        if submission.get("role") == "measure" and str(submission.get("job_id")) in active_job_ids
    ]
    exhausted: list[dict[str, Any]] = []
    license_waits: list[dict[str, Any]] = []
    terminal_without_result: list[dict[str, Any]] = []
    for record in records:
        if record["status"] == "success":
            continue
        variant = suite.variant(str(record["variant_id"]))
        case_position = int(record["case_position"])
        representative = suite.representative_case(case_position)
        attempts = _work_unit_attempts(directory, suite, variant, case_position)
        failure_attempts = _scientific_failure_count(attempts)
        if failure_attempts >= suite.maximum_work_unit_attempts:
            exhausted.append(
                {
                    "variant_id": variant.variant_id,
                    "case_role": representative.case_role,
                    "work_unit_id": record["work_unit_id"],
                    "failure_attempts": failure_attempts,
                    "maximum_work_unit_attempts": suite.maximum_work_unit_attempts,
                }
            )
            continue
        latest = _latest_work_unit_submission(
            manifest,
            variant_id=variant.variant_id,
            case_role=representative.case_role,
        )
        latest_active = latest is not None and str(latest["job_id"]) in active_job_ids
        wait = _load_benchmark_license_wait(
            directory,
            suite,
            variant,
            case_position,
            run_id=run_id,
        )
        if wait is not None and not latest_active:
            license_waits.append(wait)
        if query_scheduler and latest is not None and not latest_active and record["status"] == "pending" and wait is None:
            if scheduler["sacct"]["error"] is not None:
                message = f"Could not reconcile terminal benchmark work through accounting: {scheduler['sacct']['error']}"
                raise RuntimeError(message)
            accounted = _accounted_root_state(scheduler, job_id=str(latest["job_id"]))
            if accounted is not None and accounted not in _ACTIVE_SCHEDULER_STATES:
                terminal_without_result.append(
                    {
                        "variant_id": variant.variant_id,
                        "case_role": representative.case_role,
                        "work_unit_id": record["work_unit_id"],
                        "scheduler_state": accounted,
                    }
                )
    proof_paths = tuple(_canonical_case_proof_path(directory, representative) for representative in suite.representative_cases)
    inputs_ready = all(path.is_file() for path in proof_paths)
    canary = suite.canary_variant()
    canary_work_units = {suite.work_unit_id(canary, case_position) for case_position in range(1, suite.representative_case_count + 1)}
    canary_failed = any(item["work_unit_id"] in canary_work_units for item in (*exhausted, *terminal_without_result))
    blocked_waits = [wait for wait in license_waits if not bool(wait["retry_budget_remaining"]) or not license_service.wait_record_is_eligible(wait)]
    if manifest["state"] in {"cancel_requested", "force_cancel_requested"}:
        state = "cancelled"
    elif not inputs_ready:
        state = "inputs_ready"
    elif success_count == len(records):
        state = "complete"
    elif canary_failed:
        state = "canary_failed"
    elif exhausted or terminal_without_result:
        state = "work_unit_failed"
    elif active_work_units:
        state = "running"
    elif blocked_waits:
        state = "license_blocked"
    else:
        state = "running"
    wave_order = suite.variant_wave_order()
    current_wave = next(
        (
            {
                "wave_position": position,
                "variant_id": variant.variant_id,
                "cores_per_case": variant.cores_per_case,
            }
            for position, variant in enumerate(wave_order, start=1)
            if not all(record.get("status") == "success" for record in records if record.get("variant_id") == variant.variant_id)
        ),
        None,
    )
    return {
        "schema_kind": "generation_run_status",
        "schema_version": 1,
        "run_kind": "benchmark",
        "benchmark_run_id": run_id,
        "state": state,
        "successful_work_units": success_count,
        "active_work_units": len(set(active_work_units)),
        "pending_work_units": pending_count,
        "license_blocked_work_units": len(blocked_waits),
        "failed_work_units": failure_count,
        "total_work_units": len(records),
        "required_successful_measurements": len(suite.variants) * suite.representative_case_count,
        "work_unit_failures": [*exhausted, *terminal_without_result],
        "license_waits": license_waits,
        "scheduler": scheduler,
        "current_wave": current_wave,
        "canary": {
            "variant_id": canary.variant_id,
            "cores_per_case": canary.cores_per_case,
            "case_roles": [item.case_role for item in suite.representative_cases],
            "work_unit_ids": sorted(canary_work_units),
            "included_in_final_measurements": True,
            "additional_work_units": 0,
            "validated": all(record.get("status") == "success" for record in records if record.get("variant_id") == canary.variant_id),
        },
        "resume_action": (
            "none; terminal benchmark evidence is complete" if state == "complete" else f"rerun generation_workflow.sh run {manifest['suite_config']}"
        ),
    }


def _results_csv(records: Sequence[Mapping[str, Any]]) -> str:
    """Serialize stable per-work-unit timing, resource, and license evidence."""
    stream = StringIO(newline="")
    fields: tuple[str, ...] = (
        "variant_id",
        "cores_per_case",
        "case_position",
        "case_role",
        "work_unit_id",
        "status",
        "attempt",
        "scheduler_queue_seconds",
        "license_wait_seconds",
        "license_probe_seconds",
        "canonical_input_preparation_seconds",
        "comsol_process_seconds",
        "export_conversion_seconds",
        "publication_seconds",
        "total_controller_elapsed_seconds",
        "core_hours",
        "solver_started_at",
        "solver_ended_at",
        "node",
        "partition",
        "requested_cpus",
        "allocated_cpus",
        "comsol_np",
        "slurm_job_id",
        "peak_memory_bytes",
        "peak_scratch_bytes",
        "license_blocked_submission_count",
        "detected_feature",
        "detected_comsol_flexnet_code",
        "matched_signatures",
        "license_raw_excerpt",
        "solver_log_sha256",
        "solver_log_size_bytes",
        "solver_log_excerpt",
        "hdf5_sha256",
        "hdf5_size_bytes",
    )
    writer: csv.DictWriter[str] = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        timings_value = record.get("timings_seconds")
        timings = timings_value if isinstance(timings_value, dict) else {}
        resource_value = record.get("resource")
        resource = resource_value if isinstance(resource_value, dict) else {}
        license_value = record.get("license")
        license_evidence = license_value if isinstance(license_value, dict) else {}
        interval_value = record.get("solver_interval")
        interval = interval_value if isinstance(interval_value, dict) else {}
        solver_log_value = record.get("solver_log")
        solver_log = solver_log_value if isinstance(solver_log_value, dict) else {}
        hdf5_value = record.get("hdf5")
        hdf5 = hdf5_value if isinstance(hdf5_value, dict) else {}
        cores = record.get("cores_per_case")
        solve = timings.get("comsol_process_seconds")
        core_hours = (
            float(solve) * int(cores) / 3600.0
            if isinstance(solve, (int, float)) and not isinstance(solve, bool) and isinstance(cores, int) and not isinstance(cores, bool)
            else None
        )
        writer.writerow(
            {
                "variant_id": record.get("variant_id"),
                "cores_per_case": cores,
                "case_position": record.get("case_position"),
                "case_role": record.get("case_role"),
                "work_unit_id": record.get("work_unit_id"),
                "status": record.get("status"),
                "attempt": record.get("attempt"),
                **{field: timings.get(field) for field in _WORK_UNIT_TIMING_FIELDS},
                "core_hours": core_hours,
                "solver_started_at": interval.get("started_at"),
                "solver_ended_at": interval.get("ended_at"),
                "node": resource.get("node"),
                "partition": resource.get("partition"),
                "requested_cpus": resource.get("requested_cpus"),
                "allocated_cpus": resource.get("allocated_cpus"),
                "comsol_np": resource.get("comsol_np"),
                "slurm_job_id": resource.get("slurm_job_id"),
                "peak_memory_bytes": resource.get("peak_memory_bytes"),
                "peak_scratch_bytes": resource.get("peak_scratch_bytes"),
                "license_blocked_submission_count": license_evidence.get("license_blocked_submission_count"),
                "detected_feature": license_evidence.get("detected_feature"),
                "detected_comsol_flexnet_code": license_evidence.get("detected_error_code"),
                "matched_signatures": ",".join(license_evidence.get("matched_signatures", [])),
                "license_raw_excerpt": license_evidence.get("raw_excerpt"),
                "solver_log_sha256": solver_log.get("sha256"),
                "solver_log_size_bytes": solver_log.get("size_bytes"),
                "solver_log_excerpt": solver_log.get("excerpt"),
                "hdf5_sha256": hdf5.get("sha256"),
                "hdf5_size_bytes": hdf5.get("size_bytes"),
            }
        )
    return stream.getvalue()


def _format_metric(value: object) -> str:
    """Format one optional finite numeric metric compactly for Markdown."""
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"Benchmark metric must be numeric or null, got {value!r}."
        raise TypeError(message)
    return f"{float(value):.6g}"


def core_benchmark_markdown(summary: Mapping[str, Any]) -> str:
    """Render the fast core-selection evidence without mixing operational waits."""
    lines = [
        f"# Core-selection benchmark: {summary['suite_name']}",
        "",
        "Two deterministic scientific cases are measured concurrently in each of four sequential core-count waves.",
        "The first production-core wave is both the canary and two final measurements; there is no additional canary or second phase.",
        "Queue, license wait, and license-probe time are reported separately and do not affect compute ranking.",
        "",
        (
            "| cores | successes | median COMSOL (s) | min (s) | max (s) | "
            "core-hours/case | cases/node | cases/node-hour | queue (s) | license wait (s) | "
            "memory/case (bytes) | scratch/case (bytes) | peak solver concurrency | overlap | feasibility |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    for record in summary["variants"]:
        values = (
            record["cores_per_case"],
            record["successful_measurement_count"],
            _format_metric(record["median_comsol_process_seconds"]),
            _format_metric(record["minimum_comsol_process_seconds"]),
            _format_metric(record["maximum_comsol_process_seconds"]),
            _format_metric(record["median_core_hours_per_case"]),
            record["estimated_cases_per_node"],
            _format_metric(record["estimated_cases_per_node_hour"]),
            _format_metric(record["scheduler_queue_seconds"]),
            _format_metric(record["license_wait_seconds"]),
            record["peak_memory_per_case_bytes"],
            record["peak_scratch_per_case_bytes"],
            record["observed_peak_solver_concurrency"],
            "yes" if record["requested_cases_overlapped_in_solver_execution"] else "no",
            record["resource_feasibility"],
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    recommended = summary["recommended_production"]
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Fastest individual solve: {summary['fastest_single_case_cores']} cores per case.",
            f"- Lowest median core-hours: {summary['lowest_core_hours_cores']} cores per case.",
        ]
    )
    if recommended is None:
        lines.append("- Production recommendation: unavailable until all eight measurements and resource checks are valid.")
    else:
        proposal = recommended["proposed_configuration"]
        lines.extend(
            [
                f"- Compute-based production recommendation: {recommended['cores_per_case']} cores per case.",
                f"- Estimated cases per node: {recommended['estimated_cases_per_node']}.",
                f"- Proposed cores_per_case: {proposal['cores_per_case']}.",
                f"- Proposed cases_per_node: {proposal['cases_per_node']}.",
                f"- Proposed max_running_cases: {proposal['max_running_cases']}.",
                "- Apply only after manual review; no production configuration was edited.",
            ]
        )
    lines.extend(
        [
            f"- License qualification: {summary['license_qualification']}.",
            f"- Basis: {summary['recommendation_basis']}.",
            "- Throughput is a compute-only estimate from per-case solver time, not a fully packed-node measurement.",
            "- Dataset membership: none.",
            "",
        ]
    )
    return chr(10).join(lines)


def _archive_summary(directory: Path) -> None:
    """Preserve an earlier summary revision before a retry-derived replacement."""
    paths = [directory / name for name in ("runs.csv", "summary.json", "summary.md")]
    if not any(path.exists() for path in paths):
        return
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        message = f"Benchmark summary revision is incomplete or unsafe: {directory}"
        raise ValueError(message)
    history = directory / "summary_history"
    history.mkdir(exist_ok=True)
    revision = len(tuple(history.iterdir())) + 1
    destination = history / f"revision-{revision:04d}"
    destination.mkdir()
    for path in paths:
        shutil.copy2(path, destination / path.name)


def _proof_identities(
    suite: CoreBenchmarkSuite,
    proofs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact ordered identities for both canonical benchmark inputs."""
    return [
        {
            "case_role": representative.case_role,
            "case_input_id": proofs[representative.case_role]["case_input_id"],
            "simulation_case_id": proofs[representative.case_role]["simulation_case_id"],
            "case_payload_sha256": proofs[representative.case_role]["case_payload_sha256"],
        }
        for representative in suite.representative_cases
    ]


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    suite: CoreBenchmarkSuite,
    manifest: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind one summary revision to both inputs and all terminal evidence."""
    expected_identity = {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "template_sha256": suite.case_config.template_sha256,
        "representative_case_identities": _proof_identities(suite, proofs),
        "preflight": manifest["preflight"],
        "result_set_digest": common.serialization.canonical_json_sha256(records),
    }
    if any(summary.get(key) != value for key, value in expected_identity.items()):
        message = "Core benchmark summary is not bound to its terminal source evidence."
        raise ValueError(message)
    accounting = summary.get("scheduler_accounting")
    if not isinstance(accounting, dict) or set(accounting) != {"output", "error"}:
        message = "Core benchmark scheduler accounting evidence is malformed."
        raise TypeError(message)
    generated_at = summary.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at or any(character in generated_at for character in "\r\n\t"):
        message = "Core benchmark summary timestamp is malformed."
        raise ValueError(message)


def _summary_metrics_match(
    summary: Mapping[str, Any],
    suite: CoreBenchmarkSuite,
    records: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether derived metrics match the current campaign interpretation."""
    recomputed = summarize_core_benchmark_results(suite, records)
    return all(summary.get(key) == recomputed[key] for key in _SUMMARY_METRIC_FIELDS)


def _validate_summary_payload(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    suite: CoreBenchmarkSuite,
    manifest: Mapping[str, Any],
    proofs: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind one terminal summary to both inputs and recomputed metrics."""
    _validate_summary_identity(
        summary,
        run_id=run_id,
        suite=suite,
        manifest=manifest,
        proofs=proofs,
        records=records,
    )
    if not _summary_metrics_match(summary, suite, records):
        message = "Core benchmark summary metrics are stale or inconsistent."
        raise ValueError(message)


def _validate_or_repair_summary_outputs(
    directory: Path,
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    repair_missing: bool,
) -> None:
    """Validate deterministic summary views and create only missing retry output."""
    expected = {
        "runs.csv": _results_csv(records),
        "summary.md": core_benchmark_markdown(summary),
    }
    for name, content in expected.items():
        path = directory / name
        if not path.exists() and repair_missing:
            common.serialization.atomic_write_text(path, content)
            continue
        if not path.is_file() or path.is_symlink() or path.read_text(encoding="utf-8") != content:
            message = f"Core benchmark derived output is stale or unsafe: {path}"
            raise ValueError(message)


def finalize_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Publish aggregate evidence only after all eight measurements succeed."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    scheduler = _scheduler_evidence(manifest["measured_job_ids"])
    if scheduler["squeue"]["error"] is not None:
        message = f"Could not verify terminal benchmark jobs before finalization: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    if scheduler["squeue"]["output"]:
        message = "Core benchmark cannot finalize while persisted Slurm jobs remain active."
        raise RuntimeError(message)
    proofs = _load_case_proofs(run_id, suite, storage_root=storage_root)
    records = _result_records(directory, suite)
    _validate_records_against_proof(
        records,
        run_id=run_id,
        manifest=manifest,
        proofs=proofs,
    )
    incomplete = [record["work_unit_id"] for record in records if record["status"] != "success"]
    if incomplete:
        message = f"Core benchmark requires eight valid successes before finalization; incomplete={incomplete}."
        raise RuntimeError(message)
    result_set_digest = common.serialization.canonical_json_sha256(records)
    current = directory / "summary.json"
    if current.exists():
        existing = _load_json(current, label="core benchmark summary")
        if existing.get("result_set_digest") == result_set_digest:
            _validate_summary_identity(
                existing,
                run_id=run_id,
                suite=suite,
                manifest=manifest,
                proofs=proofs,
                records=records,
            )
            if _summary_metrics_match(existing, suite, records):
                _validate_or_repair_summary_outputs(
                    directory,
                    existing,
                    records,
                    repair_missing=True,
                )
                manifest["state"] = "complete"
                _persist_manifest(
                    _manifest_path(run_id, storage_root=storage_root),
                    manifest,
                )
                return existing
    summary = summarize_core_benchmark_results(suite, records)
    summary.update(
        {
            "benchmark_run_id": run_id,
            "git_commit": manifest["git_commit"],
            "template_sha256": suite.case_config.template_sha256,
            "representative_case_identities": _proof_identities(suite, proofs),
            "preflight": manifest["preflight"],
            "scheduler_accounting": scheduler["sacct"],
            "result_set_digest": result_set_digest,
            "generated_at": _utc_now(),
        }
    )
    _archive_summary(directory)
    common.serialization.atomic_write_text(directory / "runs.csv", _results_csv(records))
    common.serialization.atomic_write_json(current, summary)
    common.serialization.atomic_write_text(
        directory / "summary.md",
        core_benchmark_markdown(summary),
    )
    manifest["state"] = "complete"
    _persist_manifest(_manifest_path(run_id, storage_root=storage_root), manifest)
    return summary


def load_core_benchmark_summary(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate one aggregate benchmark summary."""
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    summary = _load_json(directory / "summary.json", label="core benchmark summary")
    if (
        summary.get("schema_kind") != BENCHMARK_SUMMARY_SCHEMA_KIND
        or summary.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or summary.get("benchmark_run_id") != run_id
    ):
        message = f"Core benchmark summary schema is invalid: {run_id}"
        raise ValueError(message)
    return summary


def _directory_inventory(
    directory: Path,
    *,
    ignored_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return a symlink-free deterministic file inventory."""
    if not directory.is_dir() or directory.is_symlink():
        message = f"Benchmark evidence directory is missing or unsafe: {directory}"
        raise ValueError(message)
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            message = f"Benchmark evidence contains an unsafe symlink: {path}"
            raise ValueError(message)
        if path.is_file() and path.name not in ignored_names:
            files.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": common.serialization.file_sha256(path),
                }
            )
    return {
        "file_count": len(files),
        "size_bytes": sum(int(record["size_bytes"]) for record in files),
        "files": files,
        "inventory_sha256": common.serialization.canonical_json_sha256(files),
    }


def validate_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Validate both case proofs, eight successes, summaries, and isolation."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    proofs = _load_case_proofs(run_id, suite, storage_root=storage_root)
    records = _result_records(directory, suite)
    _validate_records_against_proof(
        records,
        run_id=run_id,
        manifest=manifest,
        proofs=proofs,
    )
    incomplete = [record["work_unit_id"] for record in records if record["status"] != "success"]
    if incomplete:
        message = f"Core benchmark {run_id!r} does not have eight valid successes: {incomplete}."
        raise RuntimeError(message)
    if manifest["state"] != "complete":
        message = f"Core benchmark manifest is not complete: {manifest['state']!r}."
        raise ValueError(message)
    summary = load_core_benchmark_summary(run_id, storage_root=storage_root)
    _validate_summary_payload(
        summary,
        run_id=run_id,
        suite=suite,
        manifest=manifest,
        proofs=proofs,
        records=records,
    )
    _validate_or_repair_summary_outputs(
        directory,
        summary,
        records,
        repair_missing=False,
    )
    expected_root = (common.paths.get_generation_performance_benchmark_root(storage_root=storage_root) / BENCHMARK_FAMILY).resolve()
    raw_root = common.paths.get_generation_raw_root(storage_root=storage_root).resolve()
    processed_root = common.paths.get_generation_processed_root(storage_root=storage_root).resolve()
    if directory.parent.resolve() != expected_root or directory.is_relative_to(raw_root) or directory.is_relative_to(processed_root):
        message = f"Core benchmark evidence escaped its dedicated metadata namespace: {directory}"
        raise ValueError(message)
    inventory = _directory_inventory(
        directory,
        ignored_names=frozenset({BENCHMARK_TRANSFER_FILENAME, BENCHMARK_LOCAL_CLEANUP_FILENAME}),
    )
    return {
        "status": "complete",
        "benchmark_run_id": run_id,
        "suite_digest": suite.suite_digest,
        "representative_case_identities": _proof_identities(suite, proofs),
        "successful_measurements": len(records),
        "required_successful_measurements": len(suite.variants) * suite.representative_case_count,
        "namespace": str(directory),
        "dataset_membership": "none",
        "inventory": inventory,
    }


def core_benchmark_transfer_plan(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Return the single dedicated benchmark directory eligible for transfer."""
    validated = validate_core_benchmark(run_id, storage_root=storage_root)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    directory = core_benchmark_directory(run_id, storage_root=storage)
    manifest, _suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    return {
        "benchmark_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "relative_directory": directory.relative_to(storage).as_posix(),
        "inventory": validated["inventory"],
    }


def _validate_expected_transfer_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_sha256: str,
    expected_file_count: int,
    expected_size_bytes: int,
) -> None:
    """Require staged evidence to equal the pre-transfer CPU inventory."""
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        message = "Expected benchmark inventory SHA-256 is malformed."
        raise ValueError(message)
    for label, value in (
        ("expected_file_count", expected_file_count),
        ("expected_size_bytes", expected_size_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            message = f"{label} must be an integer >= 0, got {value!r}."
            raise ValueError(message)
    expected = {
        "inventory_sha256": expected_sha256,
        "file_count": expected_file_count,
        "size_bytes": expected_size_bytes,
    }
    if any(inventory.get(key) != value for key, value in expected.items()):
        message = "Staged benchmark evidence differs from the pre-transfer CPU inventory."
        raise RuntimeError(message)


def publish_transferred_core_benchmark(
    run_id: str,
    *,
    staging_root: Path | str,
    destination_root: Path | str,
    source_host: str,
    source_storage_root: str,
    expected_inventory_sha256: str,
    expected_file_count: int,
    expected_size_bytes: int,
) -> dict[str, Any]:
    """Validate staged benchmark evidence and atomically publish it on hpc115."""
    if not source_host or any(character in source_host for character in "\r\n\t"):
        message = "Benchmark transfer source_host must be safe non-empty text."
        raise ValueError(message)
    source_root = Path(source_storage_root)
    if (
        not source_root.is_absolute()
        or source_root == Path("/")
        or ".." in source_root.parts
        or any(character in source_storage_root for character in "\r\n\t")
    ):
        message = "Benchmark source_storage_root must be a safe absolute non-root path."
        raise ValueError(message)
    staging = workspace_service.validate_transfer_staging(staging_root, run_id=run_id)
    destination = workspace_service.resolve_storage_root(destination_root, create=True)
    incoming_root = (destination / ".incoming").resolve()
    if not staging.is_relative_to(incoming_root) or staging.stat().st_dev != destination.stat().st_dev:
        message = "Benchmark staging must be below destination .incoming on the destination filesystem."
        raise ValueError(message)
    source = core_benchmark_directory(run_id, storage_root=staging)
    target = core_benchmark_directory(run_id, storage_root=destination)
    if source.is_dir() and not source.is_symlink():
        inventory = validate_core_benchmark(run_id, storage_root=staging)["inventory"]
    elif target.is_dir() and not target.is_symlink():
        inventory = validate_core_benchmark(run_id, storage_root=destination)["inventory"]
    else:
        message = f"Benchmark incoming and final publications are both missing: {run_id}"
        raise FileNotFoundError(message)
    _validate_expected_transfer_inventory(
        inventory,
        expected_sha256=expected_inventory_sha256,
        expected_file_count=expected_file_count,
        expected_size_bytes=expected_size_bytes,
    )
    if target.exists():
        target_inventory = _directory_inventory(
            target,
            ignored_names=frozenset({BENCHMARK_TRANSFER_FILENAME, BENCHMARK_LOCAL_CLEANUP_FILENAME}),
        )
        if target_inventory != inventory:
            message = f"Existing benchmark publication conflicts: {target}"
            raise FileExistsError(message)
        if source.exists() and _directory_inventory(source) != inventory:
            message = f"Incoming benchmark publication conflicts: {source}"
            raise RuntimeError(message)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        if _directory_inventory(target) != inventory:
            message = "Benchmark inventory changed during atomic incoming publication."
            raise RuntimeError(message)
    validate_core_benchmark(run_id, storage_root=destination)
    receipt_path = target / BENCHMARK_TRANSFER_FILENAME
    if receipt_path.exists():
        existing = _load_json(receipt_path, label="benchmark transfer receipt")
        identity = {
            "benchmark_run_id": run_id,
            "source_host": source_host,
            "source_storage_root": source_storage_root,
            "destination_storage_root": str(destination),
            "inventory": inventory,
            "source_removed": False,
        }
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing
        message = f"Existing benchmark transfer receipt conflicts: {receipt_path}"
        raise FileExistsError(message)
    receipt = {
        "schema_kind": "generation_core_scaling_benchmark_transfer",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "transfer_complete",
        "recorded_at": _utc_now(),
        "benchmark_run_id": run_id,
        "source_host": source_host,
        "source_storage_root": source_storage_root,
        "destination_storage_root": str(destination),
        "inventory": inventory,
        "source_removed": False,
    }
    _write_immutable_json(
        receipt_path,
        receipt,
        label="benchmark transfer receipt",
    )
    return receipt


def _benchmark_cleanup_receipt_path(
    run_id: str,
    *,
    storage_root: Path | str,
) -> Path:
    """Return durable CPU cleanup evidence outside the removable source."""
    safe_id = common.paths.validate_logical_name(run_id, label="benchmark_run_id")
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    return common.paths.get_generation_meta_root(storage_root=storage) / "benchmark_cleanup" / f"{safe_id}.json"


def _benchmark_transfer_receipt(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load one exact host transfer receipt after validating its publication."""
    validated = validate_core_benchmark(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    receipt = _load_json(
        directory / BENCHMARK_TRANSFER_FILENAME,
        label="benchmark transfer receipt",
    )
    if (
        set(receipt)
        != {
            "schema_kind",
            "schema_version",
            "status",
            "recorded_at",
            "benchmark_run_id",
            "source_host",
            "source_storage_root",
            "destination_storage_root",
            "inventory",
            "source_removed",
        }
        or receipt.get("schema_kind") != "generation_core_scaling_benchmark_transfer"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "transfer_complete"
        or receipt.get("benchmark_run_id") != run_id
        or receipt.get("destination_storage_root") != str(workspace_service.resolve_storage_root(storage_root, create=False))
        or receipt.get("inventory") != validated["inventory"]
        or receipt.get("source_removed") is not False
        or not isinstance(receipt.get("source_host"), str)
        or not receipt["source_host"]
        or not isinstance(receipt.get("source_storage_root"), str)
        or not Path(receipt["source_storage_root"]).is_absolute()
    ):
        message = f"Benchmark transfer receipt is invalid: {directory}"
        raise ValueError(message)
    return receipt


def core_benchmark_cleanup_authorization(
    run_id: str,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Build a hash-bound authorization from the validated host publication."""
    receipt = _benchmark_transfer_receipt(run_id, storage_root=storage_root)
    inventory = {key: receipt["inventory"][key] for key in ("inventory_sha256", "file_count", "size_bytes")}
    identity = {
        "benchmark_run_id": run_id,
        "source_host": receipt["source_host"],
        "source_storage_root": receipt["source_storage_root"],
        "destination_storage_root": receipt["destination_storage_root"],
        "inventory": inventory,
    }
    return {
        "schema_kind": "generation_core_scaling_benchmark_cleanup_authorization",
        "schema_version": 1,
        **identity,
        "authorization_sha256": common.serialization.canonical_json_sha256(identity),
    }


def _validate_benchmark_cleanup_identity(
    inventory: Mapping[str, Any],
    *,
    run_id: str,
    storage: Path,
    source_host: str,
    destination_storage_root: str,
    expected_inventory_sha256: str,
    expected_file_count: int,
    expected_size_bytes: int,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Return the canonical cleanup identity after exact argument validation."""
    _validate_expected_transfer_inventory(
        inventory,
        expected_sha256=expected_inventory_sha256,
        expected_file_count=expected_file_count,
        expected_size_bytes=expected_size_bytes,
    )
    exact_inventory = {
        "inventory_sha256": expected_inventory_sha256,
        "file_count": expected_file_count,
        "size_bytes": expected_size_bytes,
    }
    identity = {
        "benchmark_run_id": run_id,
        "source_host": source_host,
        "source_storage_root": str(storage),
        "destination_storage_root": destination_storage_root,
        "inventory": exact_inventory,
    }
    expected_authorization = common.serialization.canonical_json_sha256(identity)
    if (
        not source_host
        or any(character in source_host for character in "\r\n\t")
        or not Path(destination_storage_root).is_absolute()
        or Path(destination_storage_root) == Path("/")
        or authorization_sha256 != expected_authorization
    ):
        message = "Benchmark cleanup authorization identity is invalid."
        raise ValueError(message)
    return identity


def cleanup_core_benchmark_source(
    run_id: str,
    *,
    storage_root: Path | str,
    source_host: str,
    destination_storage_root: str,
    expected_inventory_sha256: str,
    expected_file_count: int,
    expected_size_bytes: int,
    authorization_sha256: str,
    confirm: bool = False,
) -> dict[str, Any]:
    """Plan or transactionally delete one transferred terminal CPU source."""
    if not isinstance(confirm, bool):
        message = "Benchmark cleanup confirm flag must be boolean."
        raise TypeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    receipt_path = _benchmark_cleanup_receipt_path(run_id, storage_root=storage)
    lock_path = receipt_path.with_suffix(".lock")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        existing: dict[str, Any] | None = None
        if receipt_path.exists():
            existing = _load_json(receipt_path, label="benchmark CPU cleanup receipt")
            if (
                existing.get("schema_kind") != BENCHMARK_CLEANUP_SCHEMA_KIND
                or existing.get("schema_version") != 1
                or existing.get("benchmark_run_id") != run_id
                or existing.get("authorization_sha256") != authorization_sha256
                or existing.get("status") not in {"deleting", "complete"}
            ):
                message = f"Benchmark cleanup receipt conflicts: {receipt_path}"
                raise FileExistsError(message)
            if existing["status"] == "complete":
                return {
                    **existing,
                    "receipt_sha256": common.serialization.file_sha256(receipt_path),
                }
        source = core_benchmark_directory(run_id, storage_root=storage)
        quarantine = source.parent / f".cleanup-{run_id}"
        if source.exists() and quarantine.exists():
            message = "Benchmark source and cleanup quarantine both exist."
            raise RuntimeError(message)
        if source.exists():
            validated = validate_core_benchmark(run_id, storage_root=storage)
            identity = _validate_benchmark_cleanup_identity(
                validated["inventory"],
                run_id=run_id,
                storage=storage,
                source_host=source_host,
                destination_storage_root=destination_storage_root,
                expected_inventory_sha256=expected_inventory_sha256,
                expected_file_count=expected_file_count,
                expected_size_bytes=expected_size_bytes,
                authorization_sha256=authorization_sha256,
            )
            manifest, _suite = load_core_benchmark_manifest(run_id, storage_root=storage)
            scheduler = _scheduler_evidence(manifest["measured_job_ids"])
            if scheduler["squeue"]["error"] is not None:
                message = "Benchmark cleanup cannot prove scheduler inactivity."
                raise RuntimeError(message)
            if scheduler["squeue"]["output"]:
                message = "Benchmark cleanup is blocked by active Slurm work."
                raise RuntimeError(message)
        elif existing is not None and quarantine.exists():
            identity = _validate_benchmark_cleanup_identity(
                existing["inventory"],
                run_id=run_id,
                storage=storage,
                source_host=source_host,
                destination_storage_root=destination_storage_root,
                expected_inventory_sha256=expected_inventory_sha256,
                expected_file_count=expected_file_count,
                expected_size_bytes=expected_size_bytes,
                authorization_sha256=authorization_sha256,
            )
        else:
            message = f"Benchmark source is missing without complete cleanup evidence: {source}"
            raise FileNotFoundError(message)
        if not confirm:
            return {
                "schema_kind": BENCHMARK_CLEANUP_SCHEMA_KIND,
                "schema_version": 1,
                "status": "authorized",
                **identity,
                "authorization_sha256": authorization_sha256,
                "reclaimed_bytes": 0,
            }
        pending = {
            "schema_kind": BENCHMARK_CLEANUP_SCHEMA_KIND,
            "schema_version": 1,
            "status": "deleting",
            "recorded_at": _utc_now(),
            "completed_at": None,
            **identity,
            "authorization_sha256": authorization_sha256,
            "reclaimed_bytes": 0,
        }
        common.serialization.atomic_write_json(receipt_path, pending)
        if source.exists():
            source.replace(quarantine)
        if not quarantine.is_dir() or quarantine.is_symlink():
            message = f"Benchmark cleanup quarantine is unsafe: {quarantine}"
            raise RuntimeError(message)
        shutil.rmtree(quarantine)
        completed = {
            **pending,
            "status": "complete",
            "completed_at": _utc_now(),
            "reclaimed_bytes": expected_size_bytes,
        }
        common.serialization.atomic_write_json(receipt_path, completed)
        return {
            **completed,
            "receipt_sha256": common.serialization.file_sha256(receipt_path),
        }


def record_core_benchmark_cleanup(
    run_id: str,
    *,
    storage_root: Path | str,
    authorization_sha256: str,
    cleanup_receipt_sha256: str,
    reclaimed_bytes: int,
) -> dict[str, Any]:
    """Bind remote cleanup evidence to the validated host publication."""
    authorization = core_benchmark_cleanup_authorization(run_id, storage_root=storage_root)
    if (
        authorization_sha256 != authorization["authorization_sha256"]
        or _SHA256_PATTERN.fullmatch(cleanup_receipt_sha256) is None
        or isinstance(reclaimed_bytes, bool)
        or not isinstance(reclaimed_bytes, int)
        or reclaimed_bytes != authorization["inventory"]["size_bytes"]
    ):
        message = "Benchmark cleanup completion evidence is invalid."
        raise ValueError(message)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    receipt = {
        "schema_kind": "generation_core_scaling_benchmark_host_cleanup",
        "schema_version": 1,
        "status": "complete",
        "recorded_at": _utc_now(),
        "benchmark_run_id": run_id,
        "authorization_sha256": authorization_sha256,
        "cleanup_receipt_sha256": cleanup_receipt_sha256,
        "reclaimed_bytes": reclaimed_bytes,
    }
    _write_immutable_json(
        directory / BENCHMARK_LOCAL_CLEANUP_FILENAME,
        receipt,
        label="benchmark host cleanup receipt",
    )
    validate_core_benchmark(run_id, storage_root=storage_root)
    return receipt


def core_benchmark_source_status(
    run_id: str,
    *,
    storage_root: Path | str,
    query_scheduler: bool = True,
) -> dict[str, Any]:
    """Report the common CPU source-storage state for one benchmark."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    cleanup_path = _benchmark_cleanup_receipt_path(run_id, storage_root=storage)
    if cleanup_path.is_file() and not cleanup_path.is_symlink():
        cleanup = _load_json(cleanup_path, label="benchmark CPU cleanup receipt")
        if (
            cleanup.get("schema_kind") != BENCHMARK_CLEANUP_SCHEMA_KIND
            or cleanup.get("schema_version") != 1
            or cleanup.get("benchmark_run_id") != run_id
            or cleanup.get("status") != "complete"
        ):
            message = f"Benchmark CPU cleanup receipt is invalid: {cleanup_path}"
            raise ValueError(message)
        return {
            "schema_kind": "generation_run_source_status",
            "schema_version": 1,
            "run_kind": "benchmark",
            "run_id": run_id,
            "run_state": "complete",
            "source_state": "cleaned",
            "reclaimable_bytes": 0,
            "cleanup_eligibility": "already_cleaned",
            "active_slurm": False,
        }
    source = core_benchmark_directory(run_id, storage_root=storage)
    if not source.is_dir() or source.is_symlink():
        message = f"Benchmark source is missing without cleanup evidence: {source}"
        raise FileNotFoundError(message)
    status = core_benchmark_status(
        run_id,
        storage_root=storage,
        query_scheduler=query_scheduler,
    )
    inventory = _directory_inventory(
        source,
        ignored_names=frozenset({BENCHMARK_TRANSFER_FILENAME, BENCHMARK_LOCAL_CLEANUP_FILENAME}),
    )
    active = bool(status["active_work_units"])
    return {
        "schema_kind": "generation_run_source_status",
        "schema_version": 1,
        "run_kind": "benchmark",
        "run_id": run_id,
        "run_state": status["state"],
        "source_state": "retained" if status["state"] == "complete" else "active",
        "reclaimable_bytes": inventory["size_bytes"],
        "cleanup_eligibility": ("eligible" if status["state"] == "complete" and not active else "not_terminal_or_active"),
        "active_slurm": active,
    }
