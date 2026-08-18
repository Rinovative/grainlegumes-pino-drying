"""
===============================================================================
generation_benchmark.py
===============================================================================
Own the isolated transient COMSOL core-scaling benchmark lifecycle.
Responsibilities:
  - Resolve one shared scientific case and exactly four resource-only variants
  - Plan, submit, resume, execute, summarize, and transfer benchmark evidence
  - Keep measured repetitions outside canonical scientific case publication
Design principles:
  - One CPU-materialized proof binds every repeated solve to identical inputs
  - Scientific identity and resource/repetition execution identity stay separate
  - Successful repetition evidence is immutable and failed attempts are append-only
This module does NOT:
  - Define scientific parameters, publish training cases, or modify production resources
  - Run on the bare control-plane host or treat isolated throughput as contention proof
===============================================================================
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_preparation as preparation_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

from . import generation_smoke as smoke_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BENCHMARK_SUITE_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_suite"
BENCHMARK_VARIANT_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_variant"
BENCHMARK_RUN_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_run"
BENCHMARK_PROOF_SCHEMA_KIND: Final = "generation_core_scaling_case_proof"
BENCHMARK_RESULT_SCHEMA_KIND: Final = "generation_core_scaling_result"
BENCHMARK_SUMMARY_SCHEMA_KIND: Final = "generation_core_scaling_summary"
BENCHMARK_SCHEMA_VERSION: Final = 1
BENCHMARK_FAMILY: Final = "core_scaling"
_BENCHMARK_VARIANT_COUNT: Final = 4
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
_SUCCESS_TIMING_FIELDS: Final = frozenset(
    {
        "case_materialization",
        "comsol_process",
        "export_conversion",
        "hdf5_admission",
        "complete_case",
    }
)
_SUMMARY_METRIC_FIELDS: Final = (
    "suite_name",
    "suite_digest",
    "case",
    "repetitions_per_variant",
    "variants",
    "fastest_single_case",
    "fastest_single_case_cores_per_case",
    "best_parallel_efficiency",
    "best_parallel_efficiency_cores_per_case",
    "recommended_production_cores_per_case",
    "recommended_production",
    "recommendation_basis",
    "production_interpretation",
    "queue_wait_interpretation",
    "production_configuration_modified",
    "dataset_membership",
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
class CoreBenchmarkVariant:
    """One resource-only core-count variant declared by a small YAML file."""

    source_path: Path
    variant_id: str
    cores_per_case: int


@dataclass(frozen=True, slots=True)
class CoreBenchmarkSuite:
    """One resolved benchmark suite sharing a single scientific case."""

    source_path: Path
    suite_name: str
    suite_digest: str
    case_campaign_path: Path
    case_campaign: config_service.CampaignConfig
    case_config: config_service.GenerationConfig
    case_index: int
    repetitions: int
    variants: tuple[CoreBenchmarkVariant, ...]
    cores_per_node: int
    partition: str | None
    wall_time: str | None
    scheduler_options: tuple[str, ...]
    production_campaign_path: Path
    production_cores_config_path: Path
    production_cores_key: str

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

    def repetition_id(
        self,
        variant: CoreBenchmarkVariant,
        repetition: int,
    ) -> str:
        """Return one execution-replicate identity."""
        if repetition < 1 or repetition > self.repetitions:
            message = f"Benchmark repetition must be in [1, {self.repetitions}], got {repetition}."
            raise ValueError(message)
        return f"{self.execution_id(variant)}__rep_{repetition:03d}"

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
            "cases_per_measured_run": 1,
            "maximum_concurrent_measured_runs": 1,
            "poll_interval_seconds": self.case_campaign.execution_values["submission"]["poll_interval_seconds"],
        }

    def case_selection(self) -> dict[str, Any]:
        """Return the compact deterministic case selection identity."""
        assignment = self.case_config.case_assignment(self.case_index)
        seed = self.case_config.case_seed(self.case_index)
        return {
            "campaign_config": _repository_relative(self.case_campaign_path),
            "campaign_id": self.case_campaign.campaign_id,
            "batch_name": self.case_config.batch_name,
            "batch_id": self.case_config.batch_id,
            "simulation_profile": self.case_config.profile.id,
            "material_family": self.case_config.material_family,
            "sampling_regime": self.case_config.sampling_regime,
            "case_index": self.case_index,
            "case_id": self.case_config.case_id(self.case_index),
            "case_seed": seed,
            "assignment": assignment,
            "scientific_config_digest": self.case_config.scientific_config_digest,
            "case_input_config_digest": self.case_config.case_input_config_digest,
            "template": {
                "relative_path": self.case_config.template_relative_path,
                "sha256": self.case_config.template_sha256,
            },
            "selection_digest": common.serialization.canonical_json_sha256(
                {
                    "scientific_config_digest": self.case_config.scientific_config_digest,
                    "case_input_config_digest": self.case_config.case_input_config_digest,
                    "case_index": self.case_index,
                    "case_seed": seed,
                    "assignment": assignment,
                    "template_sha256": self.case_config.template_sha256,
                }
            ),
        }


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


def load_core_benchmark_suite(
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
            "case",
            "repetitions",
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
    case = _mapping(suite["case"], label="benchmark.case")
    _exact_keys(
        case,
        {"campaign_config", "material_family", "sampling_regime", "case_index"},
        label="benchmark.case",
    )
    campaign_path = _reference_path(
        case["campaign_config"],
        label="benchmark.case.campaign_config",
    )
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
    material_family = common.paths.validate_logical_name(
        case["material_family"],
        label="benchmark case material_family",
    )
    sampling_regime = common.paths.validate_logical_name(
        case["sampling_regime"],
        label="benchmark case sampling_regime",
    )
    case_config = campaign.require_batch(
        material_family=material_family,
        sampling_regime=sampling_regime,
    )
    case_index = _positive_integer(case["case_index"], label="benchmark.case.case_index")
    assignment = case_config.case_assignment(case_index)
    if assignment.get("pilot_case_kind") != "nominal_reference":
        message = "Core benchmarking requires the pilot campaign's canonical nominal_reference case."
        raise ValueError(message)

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
    repetitions = _positive_integer(
        suite["repetitions"],
        label="benchmark.repetitions",
    )
    digest_payload = {
        "schema_kind": BENCHMARK_SUITE_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite_name,
        "case": {
            "campaign_config": _repository_relative(campaign_path),
            "campaign_id": campaign.campaign_id,
            "batch_id": case_config.batch_id,
            "scientific_config_digest": case_config.scientific_config_digest,
            "case_input_config_digest": case_config.case_input_config_digest,
            "case_index": case_index,
            "case_seed": case_config.case_seed(case_index),
            "assignment": assignment,
            "template_sha256": case_config.template_sha256,
        },
        "repetitions": repetitions,
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
        case_index=case_index,
        repetitions=repetitions,
        variants=tuple(variants),
        cores_per_node=cores_per_node,
        partition=partition,
        wall_time=wall_time,
        scheduler_options=scheduler_options,
        production_campaign_path=production_campaign_path,
        production_cores_config_path=production_cores_config_path,
        production_cores_key=production_cores_key,
    )


def inspect_core_benchmark(
    path: Path | str,
    *,
    variant_id: str | None = None,
    require_executable: bool = False,
) -> dict[str, Any]:
    """Return the compact same-case and resource contract without materializing inputs."""
    suite = load_core_benchmark_suite(path, require_executable=require_executable)
    variants = suite.variants if variant_id is None else (suite.variant(variant_id),)
    return {
        "schema_kind": "generation_core_scaling_benchmark_inspection",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": _repository_relative(suite.source_path),
        "case": suite.case_selection(),
        "repetitions": suite.repetitions,
        "resource_contract": suite.resource_contract(),
        "variants": [
            {
                "variant_id": variant.variant_id,
                "source_path": _repository_relative(variant.source_path),
                "cores_per_case": variant.cores_per_case,
                "execution_id": suite.execution_id(variant),
            }
            for variant in variants
        ],
        "scientific_inputs_materialized": False,
        "dataset_membership": "none",
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


def _require_current_checkout(manifest: Mapping[str, Any]) -> None:
    """Require the executing repository to equal the persisted source commit."""
    expected = source_service.validate_git_commit(manifest.get("git_commit"))
    actual = _repository_commit()
    if actual != expected:
        message = f"Benchmark checkout commit {actual} does not match persisted commit {expected}."
        raise RuntimeError(message)


def _smoke_gate(*, storage_root: Path | str) -> dict[str, Any]:
    """Validate and compactly bind the current native runtime-smoke evidence."""
    report = smoke_service.validate_current_real_smoke_receipts(
        storage_root=storage_root,
    )
    receipts = [
        {
            "receipt_digest": record["receipt_digest"],
            "campaign_run_ids": record["campaign_run_ids"],
        }
        for record in report["valid_receipts"]
    ]
    return {
        "status": report["status"],
        "valid_receipts": receipts,
        "gate_digest": common.serialization.canonical_json_sha256(receipts),
    }


def core_benchmark_run_id(
    suite: CoreBenchmarkSuite,
    *,
    git_commit: str,
    smoke_gate_digest: str,
) -> str:
    """Return the run identity binding source, suite, and native gate."""
    commit = source_service.validate_git_commit(git_commit)
    if _SHA256_PATTERN.fullmatch(smoke_gate_digest) is None:
        message = "smoke_gate_digest must be one lowercase SHA-256 digest."
        raise ValueError(message)
    digest = common.serialization.canonical_json_sha256(
        {
            "schema_kind": BENCHMARK_RUN_SCHEMA_KIND,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "suite_digest": suite.suite_digest,
            "git_commit": commit,
            "smoke_gate_digest": smoke_gate_digest,
        }
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


def _selected_variants(
    suite: CoreBenchmarkSuite,
    variant_id: str | None,
) -> tuple[CoreBenchmarkVariant, ...]:
    """Return all variants or one explicit recovery selection."""
    return suite.variants if variant_id is None else (suite.variant(variant_id),)


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
    *,
    variant_id: str | None = None,
) -> tuple[tuple[CoreBenchmarkVariant, int], ...]:
    """Return canonical round-robin order: all cores once per repetition."""
    variants = _selected_variants(suite, variant_id)
    return tuple((variant, repetition) for repetition in range(1, suite.repetitions + 1) for variant in variants)


def build_core_benchmark_slurm_command(
    suite: CoreBenchmarkSuite,
    *,
    run_id: str,
    storage_root: Path,
    log_directory: Path,
    role: str,
    variant: CoreBenchmarkVariant | None = None,
    repetition: int | None = None,
) -> list[str]:
    """Build one ordinary preparation or measured Slurm job."""
    if role not in {"prepare", "measure"}:
        message = f"Unsupported core benchmark Slurm role: {role!r}."
        raise ValueError(message)
    repository = common.paths.get_project_root().resolve()
    launcher = repository / "scripts" / "generation_benchmark_node.sh"
    if not launcher.is_file() or launcher.is_symlink():
        message = f"Benchmark compute-node launcher is missing or unsafe: {launcher}"
        raise FileNotFoundError(message)
    if not storage_root.is_absolute() or not log_directory.is_absolute():
        message = "Benchmark Slurm storage and log roots must be absolute."
        raise ValueError(message)
    environment = _node_environment(suite, run_id)
    if role == "prepare":
        if variant is not None or repetition is not None:
            message = "Preparation submission cannot select a variant or repetition."
            raise ValueError(message)
        cpus = 1
        worker = [str(launcher), str(repository), run_id, "prepare"]
        job_suffix = "prep"
    else:
        if variant is None or repetition is None:
            message = "Measured benchmark submission requires one variant and repetition."
            raise ValueError(message)
        suite.repetition_id(variant, repetition)
        cpus = variant.cores_per_case
        worker = [
            str(launcher),
            str(repository),
            run_id,
            "measure",
            variant.variant_id,
            str(repetition),
        ]
        job_suffix = f"{variant.variant_id}-{repetition:03d}"
    wrapped = shlex.join(["env", *environment, *worker])
    job_name = f"vp2-bench-{run_id.rsplit('__', maxsplit=1)[-1]}-{job_suffix}"[:48]
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
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
    smoke_gate: Mapping[str, Any],
    variant_id: str | None,
) -> dict[str, Any]:
    """Build a read-only suite plan including canonical serial commands."""
    run_id = core_benchmark_run_id(
        suite,
        git_commit=git_commit,
        smoke_gate_digest=str(smoke_gate["gate_digest"]),
    )
    directory = core_benchmark_directory(run_id, storage_root=storage)
    logs = directory / "scheduler"
    selected = _selected_variants(suite, variant_id)
    sequence = _measured_sequence(suite, variant_id=variant_id)
    prepare_command = build_core_benchmark_slurm_command(
        suite,
        run_id=run_id,
        storage_root=storage,
        log_directory=logs,
        role="prepare",
    )
    measured_commands = [
        {
            "variant_id": variant.variant_id,
            "cores_per_case": variant.cores_per_case,
            "repetition": repetition,
            "command": build_core_benchmark_slurm_command(
                suite,
                run_id=run_id,
                storage_root=storage,
                log_directory=logs,
                role="measure",
                variant=variant,
                repetition=repetition,
            ),
        }
        for variant, repetition in sequence
    ]
    return {
        "schema_kind": "generation_core_scaling_benchmark_plan",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "state": "planned",
        "filesystem_mutated": False,
        "benchmark_run_id": run_id,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "suite_config": _repository_relative(suite.source_path),
        "git_commit": git_commit,
        "runtime_smoke": dict(smoke_gate),
        "case": suite.case_selection(),
        "repetitions": suite.repetitions,
        "variants": _variant_records(suite),
        "selected_variant_ids": [variant.variant_id for variant in selected],
        "resource_contract": suite.resource_contract(),
        "paths": {
            "storage_root": str(storage),
            "benchmark_root": str(directory),
            "scheduler_logs": str(logs),
        },
        "submission_commands": {
            "prepare": prepare_command,
            "measured_sequence": measured_commands,
        },
        "isolation": {
            "scientific_cases_per_job": 1,
            "maximum_active_benchmark_jobs": 1,
            "scheduler_arrays": False,
            "scheduler_dependencies": False,
            "queue_wait_primary_metric": False,
        },
        "dataset_membership": "none",
    }


def plan_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    variant_id: str | None = None,
) -> dict[str, Any]:
    """Validate native evidence and return a non-mutating benchmark plan."""
    requested_commit = source_service.validate_git_commit(git_commit)
    current_commit = _repository_commit()
    if current_commit != requested_commit:
        message = f"CPU checkout commit {current_commit} does not match requested commit {requested_commit}."
        raise RuntimeError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    if not storage.is_dir():
        message = f"Core benchmark plan requires the prepared storage root: {storage}"
        raise FileNotFoundError(message)
    suite = load_core_benchmark_suite(path, require_executable=True)
    gate = _smoke_gate(storage_root=storage)
    return _plan_payload(
        suite,
        git_commit=requested_commit,
        storage=storage,
        smoke_gate=gate,
        variant_id=variant_id,
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
            "runtime_smoke",
            "case",
            "repetitions",
            "variants",
            "resource_contract",
            "created_at",
            "state",
            "preparation_job_ids",
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
    expected = {
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "case": suite.case_selection(),
        "repetitions": suite.repetitions,
        "variants": _variant_records(suite),
        "resource_contract": suite.resource_contract(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        message = f"Core benchmark manifest no longer matches current suite source: {run_id}"
        raise ValueError(message)
    gate = _mapping(manifest["runtime_smoke"], label="benchmark runtime_smoke")
    expected_run_id = core_benchmark_run_id(
        suite,
        git_commit=manifest["git_commit"],
        smoke_gate_digest=str(gate.get("gate_digest")),
    )
    if expected_run_id != run_id:
        message = f"Core benchmark run identity is inconsistent: {run_id}"
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
    repetition: int,
) -> Path:
    """Return the immutable success evidence path for one repetition."""
    return directory / "runs" / suite.execution_id(variant) / suite.repetition_id(variant, repetition) / "success.json"


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    repetition: int,
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
        "repetition": repetition,
        "repetition_id": suite.repetition_id(variant, repetition),
        "scientific_config_digest": suite.case_config.scientific_config_digest,
        "template_sha256": suite.case_config.template_sha256,
        "cores_per_case": variant.cores_per_case,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        message = f"Benchmark {status} evidence conflicts for {expected['repetition_id']}."
        raise ValueError(message)
    if _BENCHMARK_RUN_ID_PATTERN.fullmatch(str(result.get("benchmark_run_id"))) is None:
        message = f"Benchmark result has a malformed run ID for {expected['repetition_id']}."
        raise ValueError(message)
    source_service.validate_git_commit(result.get("git_commit"))
    for key in ("case_input_id", "simulation_case_id"):
        if _SHA256_PATTERN.fullmatch(str(result.get(key))) is None:
            message = f"Benchmark result has malformed {key} for {expected['repetition_id']}."
            raise ValueError(message)
    attempt = result.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        message = f"Benchmark result has an invalid attempt for {expected['repetition_id']}."
        raise ValueError(message)


def _validate_resource_evidence(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    repetition: int,
) -> None:
    """Validate the exact ordinary-job allocation and solver-core evidence."""
    resource = result.get("resource")
    expected_keys = {
        "node",
        "partition",
        "requested_cpus",
        "allocated_cpus",
        "comsol_np",
        "slurm_job_id",
    }
    if not isinstance(resource, dict) or set(resource) != expected_keys:
        message = f"Benchmark resource evidence is missing for {suite.repetition_id(variant, repetition)}."
        raise TypeError(message)
    expected = {
        "requested_cpus": variant.cores_per_case,
        "allocated_cpus": variant.cores_per_case,
        "comsol_np": variant.cores_per_case,
    }
    if any(resource.get(key) != value for key, value in expected.items()):
        message = f"Benchmark allocation evidence conflicts for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    if suite.partition is not None and resource.get("partition") != suite.partition:
        message = f"Benchmark partition evidence conflicts for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    node = resource.get("node")
    if not isinstance(node, str) or not node or any(character in node for character in "\r\n\t"):
        message = f"Benchmark node evidence is malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    if _JOB_ID_PATTERN.fullmatch(str(resource.get("slurm_job_id"))) is None:
        message = f"Benchmark slurm_job_id is malformed for {suite.repetition_id(variant, repetition)}."
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
    repetition_id: str,
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
        message = f"Benchmark scheduler timing is incomplete for {repetition_id}."
        raise ValueError(message)
    submitted = _timestamp(timing["submit_time"], label="submit_time")
    started = _timestamp(timing["start_time"], label="start_time")
    completed = _timestamp(timing["completion_time"], label="completion_time")
    if started < submitted or completed < started:
        message = f"Benchmark scheduler timestamps are out of order for {repetition_id}."
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
            message = f"Benchmark {key} is inconsistent for {repetition_id}."
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


def _validate_success_result(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    repetition: int,
) -> None:
    """Validate one immutable successful repetition identity and evidence."""
    _validate_result_identity(
        result,
        suite=suite,
        variant=variant,
        repetition=repetition,
        status="success",
    )
    _validate_resource_evidence(
        result,
        suite=suite,
        variant=variant,
        repetition=repetition,
    )
    _validate_scheduler_timing(
        result,
        repetition_id=suite.repetition_id(variant, repetition),
    )
    timings = result.get("timings_s")
    if not isinstance(timings, dict) or set(timings) != _SUCCESS_TIMING_FIELDS:
        message = f"Benchmark timings are incomplete for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0
        for value in timings.values()
    ):
        message = f"Benchmark timings are malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    if float(timings["comsol_process"]) <= 0.0 or float(timings["complete_case"]) < float(timings["comsol_process"]):
        message = f"Benchmark solve/complete timings are inconsistent for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    hdf5 = result.get("hdf5")
    if not isinstance(hdf5, dict) or hdf5.get("retained_as_scientific_case") is not False:
        message = f"Benchmark HDF5 isolation evidence is missing for {suite.repetition_id(variant, repetition)}."
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
        message = f"Benchmark HDF5 evidence is malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)


def _validate_failure_result(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    repetition: int,
    attempts: Sequence[Path],
) -> None:
    """Validate one append-only failed repetition attempt."""
    _validate_result_identity(
        result,
        suite=suite,
        variant=variant,
        repetition=repetition,
        status="failed",
    )
    _validate_resource_evidence(
        result,
        suite=suite,
        variant=variant,
        repetition=repetition,
    )
    _validate_scheduler_timing(
        result,
        repetition_id=suite.repetition_id(variant, repetition),
    )
    timings = result.get("timings_s")
    if not isinstance(timings, dict) or set(timings) != {
        "case_materialization",
        "complete_case",
    }:
        message = f"Benchmark failure timings are malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    materialization = timings["case_materialization"]
    complete = timings["complete_case"]
    if (
        materialization is not None
        and (
            isinstance(materialization, bool)
            or not isinstance(materialization, (int, float))
            or not math.isfinite(float(materialization))
            or float(materialization) < 0.0
        )
    ) or (isinstance(complete, bool) or not isinstance(complete, (int, float)) or not math.isfinite(float(complete)) or float(complete) < 0.0):
        message = f"Benchmark failure timings are malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    error = result.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("type"), str) or not error["type"] or not isinstance(error.get("message"), str):
        message = f"Benchmark failure error evidence is malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    if result.get("temporary_license_retry") is not None:
        retry_count, _, exhausted = _validated_benchmark_license_retry_history(
            suite.case_config,
            attempts,
            repetition_label=suite.repetition_id(variant, repetition),
        )
        if (
            retry_count < 1
            or not exhausted
            or error.get("type") != "TemporaryLicenseCapacityExhausted"
            or error.get("message") != license_service.EXHAUSTED_REASON
        ):
            message = f"Benchmark exhausted temporary-license evidence is malformed for {suite.repetition_id(variant, repetition)}."
            raise ValueError(message)


def _validate_pending_license_result(
    result: Mapping[str, Any],
    *,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
    repetition: int,
    attempts: Sequence[Path],
) -> None:
    """Validate one append-only pending temporary-license attempt."""
    _validate_result_identity(
        result,
        suite=suite,
        variant=variant,
        repetition=repetition,
        status="pending",
    )
    _validate_resource_evidence(
        result,
        suite=suite,
        variant=variant,
        repetition=repetition,
    )
    _validate_scheduler_timing(
        result,
        repetition_id=suite.repetition_id(variant, repetition),
    )
    timings = result.get("timings_s")
    if not isinstance(timings, dict) or set(timings) != {
        "case_materialization",
        "complete_case",
    }:
        message = f"Benchmark temporary-license timings are malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)
    retry_count, _, exhausted = _validated_benchmark_license_retry_history(
        suite.case_config,
        attempts,
        repetition_label=suite.repetition_id(variant, repetition),
    )
    if retry_count < 1 or exhausted:
        message = f"Benchmark temporary-license evidence is malformed for {suite.repetition_id(variant, repetition)}."
        raise ValueError(message)


def _validated_benchmark_license_retry_history(
    config: config_service.GenerationConfig,
    attempts: Sequence[Path],
    *,
    repetition_label: str,
) -> tuple[int, float, bool]:
    """Validate and summarize one append-only benchmark retry chain."""
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    expected_keys = {
        "classification",
        "detected_feature",
        "detected_license_code",
        "matched_signatures",
        "retry_attempt_index",
        "delay_before_next_attempt_seconds",
        "cumulative_wait_seconds",
        "retry_budget_remaining",
        "next_eligible_at",
    }
    retry_count = 0
    cumulative_wait = 0.0
    exhausted = False
    for attempt_path in attempts:
        payload = _load_json(
            attempt_path,
            label="benchmark repetition attempt",
        )
        retry = payload.get("temporary_license_retry")
        if retry is None:
            continue
        if exhausted:
            message = f"Benchmark temporary-license retry history extends past exhaustion: {attempt_path}"
            raise ValueError(message)
        expected_index = retry_count + 1
        expected_delay = license_service.bounded_retry_delay_seconds(
            policy,
            attempt_index=expected_index,
            cumulative_wait_seconds=cumulative_wait,
        )
        expected_cumulative = cumulative_wait + expected_delay
        expected_remaining = expected_delay > 0.0
        detected_code = retry.get("detected_license_code") if isinstance(retry, dict) else None
        signatures = retry.get("matched_signatures") if isinstance(retry, dict) else None
        actual_index = retry.get("retry_attempt_index") if isinstance(retry, dict) else None
        actual_delay = retry.get("delay_before_next_attempt_seconds") if isinstance(retry, dict) else None
        actual_cumulative = retry.get("cumulative_wait_seconds") if isinstance(retry, dict) else None
        valid_delay = (
            not isinstance(actual_delay, bool)
            and isinstance(actual_delay, (int, float))
            and math.isfinite(float(actual_delay))
            and float(actual_delay) == expected_delay
        )
        valid_cumulative = (
            not isinstance(actual_cumulative, bool)
            and isinstance(actual_cumulative, (int, float))
            and math.isfinite(float(actual_cumulative))
            and float(actual_cumulative) == expected_cumulative
        )
        if (
            not isinstance(retry, dict)
            or set(retry) != expected_keys
            or retry.get("classification") != license_service.TEMPORARY_LICENSE_CAPACITY
            or not isinstance(retry.get("detected_feature"), str)
            or not retry["detected_feature"]
            or (detected_code is not None and not isinstance(detected_code, str))
            or not isinstance(signatures, list)
            or not signatures
            or any(not isinstance(signature, str) or not signature for signature in signatures)
            or isinstance(actual_index, bool)
            or actual_index != expected_index
            or not valid_delay
            or not valid_cumulative
            or retry.get("retry_budget_remaining") is not expected_remaining
            or payload.get("status") != ("pending" if expected_remaining else "failed")
        ):
            message = f"Benchmark temporary-license retry history is inconsistent: {attempt_path}"
            raise ValueError(message)
        recorded_at = _timestamp(payload.get("recorded_at"), label="recorded_at")
        if expected_remaining:
            eligible_at = _timestamp(
                retry.get("next_eligible_at"),
                label="temporary_license_retry.next_eligible_at",
            )
            if eligible_at != recorded_at + timedelta(seconds=expected_delay):
                message = f"Benchmark temporary-license eligibility is inconsistent for {repetition_label}."
                raise ValueError(message)
        elif retry.get("next_eligible_at") is not None:
            message = f"Benchmark exhausted retry eligibility is inconsistent for {repetition_label}."
            raise ValueError(message)
        retry_count = expected_index
        cumulative_wait = expected_cumulative
        exhausted = not expected_remaining
    return retry_count, cumulative_wait, exhausted


def _benchmark_license_retry_metadata(
    config: config_service.GenerationConfig,
    attempts: Sequence[Path],
    evidence: license_service.TemporaryLicenseCapacityClassification,
    *,
    recorded_at: str,
) -> dict[str, Any]:
    """Return the next benchmark retry record from append-only prior attempts."""
    policy = config.execution_values["runtime"]["temporary_license_retry"]
    retry_count, prior_cumulative, exhausted = _validated_benchmark_license_retry_history(
        config,
        attempts,
        repetition_label="benchmark repetition",
    )
    if exhausted:
        message = "Benchmark temporary-license retry budget is already exhausted."
        raise RuntimeError(message)
    retry_index = retry_count + 1
    delay = license_service.bounded_retry_delay_seconds(
        policy,
        attempt_index=retry_index,
        cumulative_wait_seconds=prior_cumulative,
    )
    cumulative = prior_cumulative + delay
    recorded = _timestamp(recorded_at, label="recorded_at")
    return {
        "classification": evidence.classification,
        "detected_feature": evidence.feature,
        "detected_license_code": evidence.license_code,
        "matched_signatures": list(evidence.matched_signatures),
        "retry_attempt_index": retry_index,
        "delay_before_next_attempt_seconds": delay,
        "cumulative_wait_seconds": cumulative,
        "retry_budget_remaining": delay > 0.0,
        "next_eligible_at": ((recorded + timedelta(seconds=delay)).isoformat() if delay > 0.0 else None),
    }


def _latest_benchmark_license_retry(
    manifest: Mapping[str, Any],
    suite: CoreBenchmarkSuite,
    directory: Path,
) -> Mapping[str, Any] | None:
    """Return pending retry evidence for the latest measured submission."""
    if not manifest["submission_history"]:
        return None
    latest = manifest["submission_history"][-1]
    repetitions = latest.get("repetitions")
    if latest.get("role") != "measure" or not isinstance(repetitions, list) or len(repetitions) != 1:
        return None
    matches = [
        record
        for record in _result_records(directory, suite)
        if record.get("variant_id") == latest.get("variant_id") and record.get("repetition") == repetitions[0]
    ]
    if len(matches) != 1 or matches[0].get("status") != "pending":
        return None
    retry = matches[0].get("temporary_license_retry")
    return retry if isinstance(retry, dict) else None


def _validate_materialized_case_proof(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Require a measured materialization to equal the canonical case proof."""
    if actual != expected:
        message = "Measured benchmark materialization differs from the canonical case proof."
        raise RuntimeError(message)


def _validate_hdf5_scientific_identity(
    hdf5_identity: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    """Require admitted HDF5 output to retain the benchmark scientific identity."""
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


def _latest_submission_result_state(
    manifest: Mapping[str, Any],
    suite: CoreBenchmarkSuite,
    directory: Path,
) -> str:
    """Return success, failed, pending, or absent for the latest submitted job."""
    history = manifest["submission_history"]
    if not history:
        return "absent"
    latest = history[-1]
    if latest["role"] == "prepare":
        return "success" if (directory / "canonical_case.json").is_file() else "pending"
    variant_id = latest.get("variant_id")
    repetitions = latest.get("repetitions")
    if not isinstance(variant_id, str) or not isinstance(repetitions, list) or len(repetitions) != 1:
        message = "Latest benchmark measured submission identity is malformed."
        raise ValueError(message)
    matches = [
        record for record in _result_records(directory, suite) if record["variant_id"] == variant_id and record["repetition"] == repetitions[0]
    ]
    if len(matches) != 1:
        message = "Latest benchmark submission does not resolve to one repetition."
        raise RuntimeError(message)
    return str(matches[0]["status"])


def _latest_job_is_terminal_without_result(
    manifest: Mapping[str, Any],
    suite: CoreBenchmarkSuite,
    directory: Path,
    scheduler: Mapping[str, Any],
) -> bool:
    """Return whether accounting proves the latest evidence-missing job terminal."""
    if _latest_submission_result_state(manifest, suite, directory) != "pending":
        return False
    sacct = scheduler["sacct"]
    if sacct["error"] is not None:
        message = f"Could not reconcile the latest benchmark job through accounting: {sacct['error']}"
        raise RuntimeError(message)
    latest_job_id = str(manifest["submission_history"][-1]["job_id"])
    state = _accounted_root_state(scheduler, job_id=latest_job_id)
    return state is not None and state not in _ACTIVE_SCHEDULER_STATES


def _pending_repetitions(
    directory: Path,
    suite: CoreBenchmarkSuite,
    variant: CoreBenchmarkVariant,
) -> tuple[int, ...]:
    """Return only repetitions without valid immutable success evidence."""
    pending: list[int] = []
    for repetition in range(1, suite.repetitions + 1):
        path = _success_path(directory, suite, variant, repetition)
        if path.exists():
            _validate_success_result(
                _load_json(path, label="benchmark repetition success"),
                suite=suite,
                variant=variant,
                repetition=repetition,
            )
        else:
            attempts = tuple(
                sorted((directory / "runs" / suite.execution_id(variant) / suite.repetition_id(variant, repetition)).glob("attempt-*.json"))
            )
            _, _, exhausted = _validated_benchmark_license_retry_history(
                suite.case_config,
                attempts,
                repetition_label=suite.repetition_id(variant, repetition),
            )
            if not exhausted:
                pending.append(repetition)
    return tuple(pending)


def _persist_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically replace mutable benchmark orchestration state."""
    common.serialization.atomic_write_json(path, dict(manifest))


def _submit_pending(
    manifest: dict[str, Any],
    suite: CoreBenchmarkSuite,
    *,
    storage: Path,
    variant_id: str | None,
) -> dict[str, Any]:
    """Reconcile and submit at most one job in canonical serial order."""
    run_id = str(manifest["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    logs = directory / "scheduler"
    logs.mkdir(parents=True, exist_ok=True)
    persisted_job_ids = [
        str(value)
        for value in (
            *manifest["preparation_job_ids"],
            *manifest["measured_job_ids"],
        )
    ]
    if any(_JOB_ID_PATTERN.fullmatch(job_id) is None for job_id in persisted_job_ids):
        message = f"Benchmark manifest contains malformed Slurm job IDs: {run_id}"
        raise ValueError(message)
    if persisted_job_ids:
        scheduler = _scheduler_evidence(persisted_job_ids)
        queue = scheduler["squeue"]
        if queue["error"] is not None:
            message = f"Could not verify active benchmark jobs before resume: {queue['error']}"
            raise RuntimeError(message)
        if queue["output"]:
            manifest["state"] = "submitted"
            _persist_manifest(_manifest_path(run_id, storage_root=storage), manifest)
            return manifest
        latest_result = _latest_submission_result_state(manifest, suite, directory)
        latest_retry = _latest_benchmark_license_retry(
            manifest,
            suite,
            directory,
        )
        if latest_retry is not None and not license_service.retry_attempt_is_eligible(latest_retry):
            manifest["state"] = "waiting_retry"
            _persist_manifest(
                _manifest_path(run_id, storage_root=storage),
                manifest,
            )
            return manifest
        if latest_result == "pending" and not _latest_job_is_terminal_without_result(
            manifest,
            suite,
            directory,
            scheduler,
        ):
            manifest["state"] = "scheduler_unknown"
            _persist_manifest(_manifest_path(run_id, storage_root=storage), manifest)
            return manifest
    proof = directory / "canonical_case.json"
    if not proof.is_file():
        command = build_core_benchmark_slurm_command(
            suite,
            run_id=run_id,
            storage_root=storage,
            log_directory=logs,
            role="prepare",
        )
        submitted_at = _utc_now()
        preparation_job = _submit(
            command,
            git_commit=str(manifest["git_commit"]),
            run_id=run_id,
        )
        manifest["preparation_job_ids"].append(preparation_job)
        manifest["submission_history"].append(
            {
                "submitted_at": submitted_at,
                "role": "prepare",
                "variant_id": None,
                "repetitions": [],
                "command": command,
                "job_id": preparation_job,
            }
        )
        manifest["state"] = "submitted"
        _persist_manifest(_manifest_path(run_id, storage_root=storage), manifest)
        return manifest
    next_execution = next(
        (
            (variant, repetition)
            for variant, repetition in _measured_sequence(
                suite,
                variant_id=variant_id,
            )
            if repetition in _pending_repetitions(directory, suite, variant)
        ),
        None,
    )
    if next_execution is None:
        remaining = sum(len(_pending_repetitions(directory, suite, variant)) for variant in suite.variants)
        records = _result_records(directory, suite)
        retry_exhausted = any(
            record.get("status") == "failed"
            and isinstance(record.get("temporary_license_retry"), dict)
            and record["temporary_license_retry"].get("retry_budget_remaining") is False
            for record in records
        )
        manifest["state"] = "retry_exhausted" if retry_exhausted else "complete" if remaining == 0 else "incomplete"
        _persist_manifest(_manifest_path(run_id, storage_root=storage), manifest)
        return manifest
    variant, repetition = next_execution
    command = build_core_benchmark_slurm_command(
        suite,
        run_id=run_id,
        storage_root=storage,
        log_directory=logs,
        role="measure",
        variant=variant,
        repetition=repetition,
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
            "repetitions": [repetition],
            "command": command,
            "job_id": job_id,
        }
    )
    manifest["state"] = "submitted"
    _persist_manifest(_manifest_path(run_id, storage_root=storage), manifest)
    return manifest


def submit_core_benchmark(
    path: Path | str,
    *,
    git_commit: str,
    storage_root: Path | str,
    variant_id: str | None = None,
) -> dict[str, Any]:
    """Persist benchmark intent and submit isolated measured repetitions."""
    plan = plan_core_benchmark(
        path,
        git_commit=git_commit,
        storage_root=storage_root,
        variant_id=variant_id,
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    run_id = str(plan["benchmark_run_id"])
    directory = core_benchmark_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "benchmark_manifest.json"
    lock = directory / "submission.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        if manifest_path.exists():
            manifest, suite = load_core_benchmark_manifest(
                run_id,
                storage_root=storage,
            )
            return _submit_pending(
                manifest,
                suite,
                storage=storage,
                variant_id=variant_id,
            )
        suite = load_core_benchmark_suite(path, require_executable=True)
        manifest = {
            "schema_kind": BENCHMARK_RUN_SCHEMA_KIND,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_run_id": run_id,
            "suite_name": suite.suite_name,
            "suite_digest": suite.suite_digest,
            "suite_config": _repository_relative(suite.source_path),
            "git_commit": plan["git_commit"],
            "runtime_smoke": plan["runtime_smoke"],
            "case": suite.case_selection(),
            "repetitions": suite.repetitions,
            "variants": _variant_records(suite),
            "resource_contract": suite.resource_contract(),
            "created_at": _utc_now(),
            "state": "submitting",
            "preparation_job_ids": [],
            "measured_job_ids": [],
            "submission_history": [],
        }
        _persist_manifest(manifest_path, manifest)
        return _submit_pending(
            manifest,
            suite,
            storage=storage,
            variant_id=variant_id,
        )


def resume_core_benchmark(
    run_id: str,
    *,
    storage_root: Path | str,
    variant_id: str | None = None,
) -> dict[str, Any]:
    """Submit only repetitions without successful immutable evidence."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest_path = _manifest_path(run_id, storage_root=storage)
    lock = manifest_path.parent / "submission.lock"
    with common.locking.exclusive_file_lock(lock, blocking=False):
        manifest, suite = load_core_benchmark_manifest(
            run_id,
            storage_root=storage,
        )
        _require_current_checkout(manifest)
        return _submit_pending(
            manifest,
            suite,
            storage=storage,
            variant_id=variant_id,
        )


def _proof_payload(
    suite: CoreBenchmarkSuite,
    prepared: preparation_service.PreparedCase,
) -> dict[str, Any]:
    """Return the exact scientific byte identity shared by every measured run."""
    case = prepared.bundle.case_payload
    return {
        "schema_kind": BENCHMARK_PROOF_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_digest": suite.suite_digest,
        "case_id": prepared.bundle.case_id,
        "case_index": suite.case_index,
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


def prepare_core_benchmark_case(
    run_id: str,
    *,
    storage_root: Path | str,
    work_root: Path | str,
) -> Path:
    """Materialize on a CPU node and publish the canonical same-case proof."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    _require_current_checkout(manifest)
    if source_service.required_git_commit() != manifest["git_commit"]:
        message = "Benchmark preparation checkout does not match the launch commit."
        raise RuntimeError(message)
    prepared: preparation_service.PreparedCase | None = None
    try:
        prepared = preparation_service.prepare_case_work_directory(
            suite.case_config,
            suite.case_index,
            storage_root=storage,
            work_root=work_root,
        )
        proof = _proof_payload(suite, prepared)
        path = core_benchmark_directory(run_id, storage_root=storage) / "canonical_case.json"
        _write_immutable_json(path, proof, label="benchmark canonical-case proof")
        return path
    finally:
        if prepared is not None:
            _cleanup_prepared(prepared, storage=storage)


def _load_case_proof(
    run_id: str,
    suite: CoreBenchmarkSuite,
    *,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Load and validate the CPU-materialized canonical input proof."""
    path = core_benchmark_directory(run_id, storage_root=storage_root) / "canonical_case.json"
    proof = _load_json(path, label="benchmark canonical-case proof")
    template = proof.get("template")
    if (
        proof.get("schema_kind") != BENCHMARK_PROOF_SCHEMA_KIND
        or proof.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or proof.get("suite_digest") != suite.suite_digest
        or proof.get("case_index") != suite.case_index
        or proof.get("scientific_config_digest") != suite.case_config.scientific_config_digest
        or proof.get("case_input_config_digest") != suite.case_config.case_input_config_digest
        or not isinstance(template, dict)
        or template.get("sha256") != suite.case_config.template_sha256
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
    repetition: int,
) -> Mapping[str, Any]:
    """Return the one persisted submission record for the current measured job."""
    matches = [
        record
        for record in manifest["submission_history"]
        if record.get("role") == "measure"
        and record.get("job_id") == job_id
        and record.get("variant_id") == variant_id
        and record.get("repetitions") == [repetition]
    ]
    if len(matches) != 1:
        message = "Current benchmark job is not bound to this exact repetition."
        raise RuntimeError(message)
    return matches[0]


def run_core_benchmark_repetition(
    run_id: str,
    variant_id: str,
    repetition: int,
    *,
    storage_root: Path | str,
    work_root: Path | str,
) -> dict[str, Any]:
    """Materialize, identity-check, solve, admit, record, and clean one repetition."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage)
    _require_current_checkout(manifest)
    start_time = _slurm_scheduler_start_time()
    variant = suite.variant(variant_id)
    repetition_id = suite.repetition_id(variant, repetition)
    if source_service.required_git_commit() != manifest["git_commit"]:
        message = "Benchmark worker checkout does not match the launch commit."
        raise RuntimeError(message)
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = "Benchmark repetition requires one numeric SLURM_JOB_ID."
        raise RuntimeError(message)
    submission = _measured_submission(
        manifest,
        job_id=job_id,
        variant_id=variant.variant_id,
        repetition=repetition,
    )
    submit_time = str(submission["submitted_at"])
    if allocated_cpus is None or not allocated_cpus.isdigit() or int(allocated_cpus) != variant.cores_per_case:
        message = f"Benchmark Slurm allocation must exactly match cores_per_case; requested={variant.cores_per_case}, allocated={allocated_cpus!r}."
        raise RuntimeError(message)
    proof = _load_case_proof(run_id, suite, storage_root=storage)
    directory = core_benchmark_directory(run_id, storage_root=storage)
    repetition_directory = directory / "runs" / suite.execution_id(variant) / repetition_id
    repetition_directory.mkdir(parents=True, exist_ok=True)
    success_path = repetition_directory / "success.json"
    lock_path = repetition_directory / "execution.lock"
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if success_path.exists():
            existing_success = _load_json(
                success_path,
                label="benchmark repetition success",
            )
            _validate_success_result(
                existing_success,
                suite=suite,
                variant=variant,
                repetition=repetition,
            )
            return existing_success
        attempts = tuple(sorted(repetition_directory.glob("attempt-*.json")))
        attempt_number = len(attempts) + 1
        attempt_path = repetition_directory / f"attempt-{attempt_number:04d}.json"
        prepared: preparation_service.PreparedCase | None = None
        total_start = time.monotonic()
        materialization_start = total_start
        materialization_s: float | None = None
        success: dict[str, Any]
        try:
            prepared = preparation_service.prepare_case_work_directory(
                suite.case_config,
                suite.case_index,
                storage_root=storage,
                work_root=work_root,
            )
            materialization_s = time.monotonic() - materialization_start
            actual_proof = _proof_payload(suite, prepared)
            _validate_materialized_case_proof(actual_proof, proof)
            result = runtime_service.execute_prepared_case(
                suite.case_config,
                prepared,
                cores_per_case=variant.cores_per_case,
                worker_slot=0,
                scheduler_kind="slurm",
                allocated_node=os.environ.get(
                    "SLURMD_NODENAME",
                    socket.gethostname(),
                ),
            )
            admission_start = time.monotonic()
            hdf5_identity = storage_service.validate_case_hdf5(
                result.canonical_case.path,
                expected_profile=profiles.TRANSIENT_DRYING_PROFILE,
            )
            hdf5_admission_s = time.monotonic() - admission_start
            _validate_hdf5_scientific_identity(hdf5_identity, proof)
            command_np = result.command[result.command.index("-np") + 1]
            completion_time = _utc_now()
            success = {
                "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "status": "success",
                "recorded_at": _utc_now(),
                "benchmark_run_id": run_id,
                "suite_digest": suite.suite_digest,
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "repetition": repetition,
                "repetition_id": repetition_id,
                "attempt": attempt_number,
                "git_commit": manifest["git_commit"],
                "case_input_id": proof["case_input_id"],
                "simulation_case_id": proof["simulation_case_id"],
                "scientific_config_digest": proof["scientific_config_digest"],
                "template_sha256": suite.case_config.template_sha256,
                "cores_per_case": variant.cores_per_case,
                "resource": {
                    "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
                    "partition": os.environ.get("SLURM_JOB_PARTITION", suite.partition),
                    "requested_cpus": variant.cores_per_case,
                    "allocated_cpus": int(allocated_cpus),
                    "comsol_np": int(command_np),
                    "slurm_job_id": job_id,
                },
                "scheduler_timing": _scheduler_timing(
                    submit_time=submit_time,
                    start_time=start_time,
                    completion_time=completion_time,
                ),
                "timings_s": {
                    "case_materialization": materialization_s,
                    "comsol_process": float(result.timing["runtime_s"]),
                    "export_conversion": float(result.timing["export_conversion_s"]),
                    "hdf5_admission": hdf5_admission_s,
                    "complete_case": time.monotonic() - total_start,
                },
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
                label="benchmark repetition success",
            )
        except BaseException as error:
            completion_time = _utc_now()
            retry_metadata = None
            if isinstance(
                error,
                license_service.TemporaryLicenseCapacityError,
            ) and bool(suite.case_config.execution_values["runtime"]["temporary_license_retry"]["enabled"]):
                retry_metadata = _benchmark_license_retry_metadata(
                    suite.case_config,
                    attempts,
                    error.evidence,
                    recorded_at=completion_time,
                )
            failure = {
                "schema_kind": BENCHMARK_RESULT_SCHEMA_KIND,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "status": ("pending" if retry_metadata is not None and retry_metadata["retry_budget_remaining"] else "failed"),
                "recorded_at": completion_time,
                "benchmark_run_id": run_id,
                "suite_digest": suite.suite_digest,
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "repetition": repetition,
                "repetition_id": repetition_id,
                "attempt": attempt_number,
                "git_commit": manifest["git_commit"],
                "case_input_id": proof["case_input_id"],
                "simulation_case_id": proof["simulation_case_id"],
                "scientific_config_digest": proof["scientific_config_digest"],
                "template_sha256": suite.case_config.template_sha256,
                "cores_per_case": variant.cores_per_case,
                "resource": {
                    "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
                    "partition": os.environ.get("SLURM_JOB_PARTITION", suite.partition),
                    "requested_cpus": variant.cores_per_case,
                    "allocated_cpus": int(allocated_cpus),
                    "comsol_np": variant.cores_per_case,
                    "slurm_job_id": job_id,
                },
                "scheduler_timing": _scheduler_timing(
                    submit_time=submit_time,
                    start_time=start_time,
                    completion_time=completion_time,
                ),
                "timings_s": {
                    "case_materialization": materialization_s,
                    "complete_case": time.monotonic() - total_start,
                },
                "error": {
                    "type": (
                        "TemporaryLicenseCapacityExhausted"
                        if retry_metadata is not None and not retry_metadata["retry_budget_remaining"]
                        else type(error).__name__
                    ),
                    "message": (
                        license_service.EXHAUSTED_REASON
                        if retry_metadata is not None and not retry_metadata["retry_budget_remaining"]
                        else str(error)
                    ),
                },
            }
            if retry_metadata is not None:
                failure["temporary_license_retry"] = retry_metadata
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
    """Return one latest terminal-or-pending record per configured repetition."""
    records: list[dict[str, Any]] = []
    for variant in suite.variants:
        for repetition in range(1, suite.repetitions + 1):
            repetition_id = suite.repetition_id(variant, repetition)
            repetition_directory = directory / "runs" / suite.execution_id(variant) / repetition_id
            success_path = repetition_directory / "success.json"
            if success_path.is_file():
                record = _load_json(
                    success_path,
                    label="benchmark repetition success",
                )
                _validate_success_result(
                    record,
                    suite=suite,
                    variant=variant,
                    repetition=repetition,
                )
                records.append(record)
                continue
            attempts = tuple(sorted(repetition_directory.glob("attempt-*.json")))
            if attempts:
                attempt = _load_json(
                    attempts[-1],
                    label="benchmark repetition attempt",
                )
                if attempt.get("status") == "failed":
                    _validate_failure_result(
                        attempt,
                        suite=suite,
                        variant=variant,
                        repetition=repetition,
                        attempts=attempts,
                    )
                    records.append(attempt)
                    continue
                if attempt.get("status") == "success":
                    _validate_success_result(
                        attempt,
                        suite=suite,
                        variant=variant,
                        repetition=repetition,
                    )
                elif attempt.get("status") == "pending":
                    _validate_pending_license_result(
                        attempt,
                        suite=suite,
                        variant=variant,
                        repetition=repetition,
                        attempts=attempts,
                    )
                    records.append(attempt)
                    continue
                else:
                    message = f"Benchmark attempt has an unsupported status: {attempts[-1]}"
                    raise ValueError(message)
            records.append(
                {
                    "status": "pending",
                    "variant_id": variant.variant_id,
                    "execution_id": suite.execution_id(variant),
                    "repetition": repetition,
                    "repetition_id": repetition_id,
                    "cores_per_case": variant.cores_per_case,
                }
            )
    return records


def _validate_records_against_proof(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    """Bind every terminal attempt to the canonical case, source, and run."""
    expected = {
        "benchmark_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "case_input_id": proof["case_input_id"],
        "simulation_case_id": proof["simulation_case_id"],
        "scientific_config_digest": proof["scientific_config_digest"],
        "template_sha256": proof["template"]["sha256"],
    }
    for record in records:
        if record.get("status") == "pending":
            continue
        if any(record.get(key) != value for key, value in expected.items()):
            message = f"Benchmark result is not bound to the canonical proof: {record.get('repetition_id')!r}."
            raise ValueError(message)


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
        "cores_config": _repository_relative(suite.production_cores_config_path),
        "cores_key": suite.production_cores_key,
    }


def summarize_core_benchmark_results(
    suite: CoreBenchmarkSuite,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calculate latency, timing, scaling, and core-hour recommendation metrics."""
    expected_count = len(suite.variants) * suite.repetitions
    if len(records) != expected_count:
        message = f"Benchmark summary requires {expected_count} repetition records, got {len(records)}."
        raise ValueError(message)
    production = _production_interpretation(suite)
    campaign_cases = int(production["campaign_total_cases"])
    by_variant: list[dict[str, Any]] = []
    for variant in suite.variants:
        selected = [record for record in records if record.get("variant_id") == variant.variant_id]
        if len(selected) != suite.repetitions:
            message = f"Benchmark records do not cover every repetition for {variant.variant_id!r}."
            raise ValueError(message)
        successes = [record for record in selected if record.get("status") == "success"]
        failed = [record for record in selected if record.get("status") == "failed"]
        pending = [record for record in selected if record.get("status") == "pending"]
        solve_times = [float(record["timings_s"]["comsol_process"]) for record in successes]
        total_times = [float(record["timings_s"]["complete_case"]) for record in successes]
        queue_waits = [float(record["scheduler_timing"]["queue_wait_s"]) for record in successes]
        turnarounds = [float(record["scheduler_timing"]["turnaround_s"]) for record in successes]
        all_values = [*solve_times, *total_times, *queue_waits, *turnarounds]
        if any(not math.isfinite(value) or value < 0.0 for value in all_values) or any(value <= 0.0 for value in solve_times):
            message = f"Benchmark summary received invalid timings for {variant.variant_id!r}."
            raise ValueError(message)
        if solve_times:
            median_solve = statistics.median(solve_times)
            median_total = statistics.median(total_times)
            core_hours = variant.cores_per_case * median_solve / 3600.0
            metrics: dict[str, float | None] = {
                "median_comsol_wall_s": median_solve,
                "mean_comsol_wall_s": statistics.fmean(solve_times),
                "minimum_comsol_wall_s": min(solve_times),
                "maximum_comsol_wall_s": max(solve_times),
                "sample_standard_deviation_comsol_wall_s": (statistics.stdev(solve_times) if len(solve_times) > 1 else None),
                "median_total_case_wall_s": median_total,
                "median_queue_wait_s": statistics.median(queue_waits),
                "median_turnaround_s": statistics.median(turnarounds),
                "median_core_hours_per_case": core_hours,
                "cases_per_100_core_hours": 100.0 / core_hours,
                "estimated_production_campaign_core_hours": campaign_cases * core_hours,
            }
        else:
            metrics = {
                "median_comsol_wall_s": None,
                "mean_comsol_wall_s": None,
                "minimum_comsol_wall_s": None,
                "maximum_comsol_wall_s": None,
                "sample_standard_deviation_comsol_wall_s": None,
                "median_total_case_wall_s": None,
                "median_queue_wait_s": None,
                "median_turnaround_s": None,
                "median_core_hours_per_case": None,
                "cases_per_100_core_hours": None,
                "estimated_production_campaign_core_hours": None,
            }
        by_variant.append(
            {
                "variant_id": variant.variant_id,
                "execution_id": suite.execution_id(variant),
                "cores_per_case": variant.cores_per_case,
                "successful_repetitions": len(successes),
                "failed_repetitions": len(failed),
                "pending_repetitions": len(pending),
                "individual_comsol_wall_s": solve_times,
                "individual_total_case_wall_s": total_times,
                "individual_queue_wait_s": queue_waits,
                "individual_turnaround_s": turnarounds,
                **metrics,
                "speedup": None,
                "parallel_efficiency": None,
            }
        )
    reference = by_variant[0]
    reference_median = reference["median_comsol_wall_s"]
    reference_cores = int(reference["cores_per_case"])
    if isinstance(reference_median, float):
        for record in by_variant:
            median = record["median_comsol_wall_s"]
            if isinstance(median, float):
                speedup = reference_median / median
                record["speedup"] = speedup
                record["parallel_efficiency"] = speedup / (int(record["cores_per_case"]) / reference_cores)
    recommendation_pool = [
        record
        for record in by_variant
        if record["successful_repetitions"] == suite.repetitions and record["failed_repetitions"] == 0 and record["pending_repetitions"] == 0
    ]
    fastest = (
        min(
            recommendation_pool,
            key=lambda record: (
                float(record["median_comsol_wall_s"]),
                int(record["cores_per_case"]),
            ),
        )
        if recommendation_pool
        else None
    )
    efficiency_pool = [record for record in recommendation_pool if isinstance(record["parallel_efficiency"], float)]
    best_efficiency = (
        max(
            efficiency_pool,
            key=lambda record: (
                float(record["parallel_efficiency"]),
                -int(record["cores_per_case"]),
            ),
        )
        if efficiency_pool
        else None
    )
    recommended = (
        min(
            recommendation_pool,
            key=lambda record: (
                float(record["median_core_hours_per_case"]),
                int(record["cores_per_case"]),
            ),
        )
        if recommendation_pool
        else None
    )
    current_cores = int(production["current_production_cores_per_case"])
    fastest_cores = None if fastest is None else int(fastest["cores_per_case"])
    best_efficiency_cores = None if best_efficiency is None else int(best_efficiency["cores_per_case"])
    recommended_cores = None if recommended is None else int(recommended["cores_per_case"])
    production["recommended_difference_from_current_cores_per_case"] = None if recommended_cores is None else recommended_cores - current_cores
    production["recommended_differs_from_current"] = None if recommended_cores is None else recommended_cores != current_cores
    if recommended is None:
        recommended_detail = None
    else:
        selected_cores = int(recommended["cores_per_case"])
        recommended_detail = {
            "variant_id": recommended["variant_id"],
            "cores_per_case": selected_cores,
            "median_comsol_wall_s": recommended["median_comsol_wall_s"],
            "median_total_case_wall_s": recommended["median_total_case_wall_s"],
            "median_core_hours_per_case": recommended["median_core_hours_per_case"],
            "speedup": recommended["speedup"],
            "parallel_efficiency": recommended["parallel_efficiency"],
            "current_production_cores_per_case": current_cores,
            "difference_from_current_cores_per_case": selected_cores - current_cores,
            "fastest_single_case_cores_per_case": fastest_cores,
            "differs_from_fastest_single_case": selected_cores != fastest_cores,
        }
    return {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite_name": suite.suite_name,
        "suite_digest": suite.suite_digest,
        "case": suite.case_selection(),
        "repetitions_per_variant": suite.repetitions,
        "variants": by_variant,
        "fastest_single_case": (
            None
            if fastest is None
            else {
                "variant_id": fastest["variant_id"],
                "cores_per_case": fastest_cores,
                "median_comsol_wall_s": fastest["median_comsol_wall_s"],
            }
        ),
        "fastest_single_case_cores_per_case": fastest_cores,
        "best_parallel_efficiency": (
            None
            if best_efficiency is None
            else {
                "variant_id": best_efficiency["variant_id"],
                "cores_per_case": best_efficiency_cores,
                "parallel_efficiency": best_efficiency["parallel_efficiency"],
            }
        ),
        "best_parallel_efficiency_cores_per_case": best_efficiency_cores,
        "recommended_production_cores_per_case": recommended_cores,
        "recommended_production": recommended_detail,
        "recommendation_basis": ("lowest median COMSOL core-hours per successful case; ties break to fewer cores"),
        "production_interpretation": production,
        "queue_wait_interpretation": ("observed scheduler conditions only; excluded from solve, core-hour, and recommendation metrics"),
        "production_configuration_modified": False,
        "dataset_membership": "none",
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


def core_benchmark_status(
    run_id: str,
    *,
    storage_root: Path | str,
    query_scheduler: bool = True,
) -> dict[str, Any]:
    """Reconstruct repetition completion and optional scheduler state."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    records = _result_records(directory, suite)
    success_count = sum(record["status"] == "success" for record in records)
    failure_count = sum(record["status"] == "failed" for record in records)
    pending_count = sum(record["status"] == "pending" for record in records)
    exhausted_retries = [
        record
        for record in records
        if record.get("status") == "failed"
        and isinstance(record.get("temporary_license_retry"), dict)
        and record["temporary_license_retry"].get("retry_budget_remaining") is False
    ]
    retryable_failure_count = failure_count - len(exhausted_retries)
    job_ids = [*manifest["preparation_job_ids"], *manifest["measured_job_ids"]]
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
    latest_retry = _latest_benchmark_license_retry(
        manifest,
        suite,
        directory,
    )
    if scheduler["squeue"]["output"]:
        state = "running"
    elif latest_retry is not None and not license_service.retry_attempt_is_eligible(latest_retry):
        state = "waiting_retry"
    elif success_count == len(records):
        state = "complete"
    elif retryable_failure_count:
        state = "retry_required"
    elif exhausted_retries and pending_count == 0:
        state = "retry_exhausted"
    elif query_scheduler and manifest["submission_history"] and _latest_submission_result_state(manifest, suite, directory) == "pending":
        state = (
            "retry_required"
            if _latest_job_is_terminal_without_result(
                manifest,
                suite,
                directory,
                scheduler,
            )
            else "scheduler_unknown"
        )
    else:
        state = "incomplete"
    retry_repetitions = [
        {
            "variant_id": record["variant_id"],
            "repetition": record["repetition"],
            "repetition_id": record["repetition_id"],
            "evidence_status": "failed",
        }
        for record in records
        if record["status"] == "failed" and record not in exhausted_retries
    ]
    if state == "retry_required" and not retry_repetitions:
        latest = manifest["submission_history"][-1]
        latest_repetitions = latest.get("repetitions")
        if latest.get("role") == "measure" and isinstance(latest_repetitions, list) and len(latest_repetitions) == 1:
            variant = suite.variant(str(latest.get("variant_id")))
            repetition = int(latest_repetitions[0])
            retry_repetitions.append(
                {
                    "variant_id": variant.variant_id,
                    "repetition": repetition,
                    "repetition_id": suite.repetition_id(variant, repetition),
                    "evidence_status": "terminal_without_result",
                }
            )
    return {
        "benchmark_run_id": run_id,
        "state": state,
        "successful_repetitions": success_count,
        "failed_repetitions": failure_count,
        "pending_repetitions": pending_count,
        "total_repetitions": len(records),
        "retry_repetitions": retry_repetitions,
        "scheduler": scheduler,
        "suggested_next_command": (
            f"finalize-core-benchmark {run_id}"
            if state in {"complete", "retry_exhausted"}
            else f"resume-core-benchmark {run_id}"
            if state in {"retry_required", "incomplete"}
            else f"core-benchmark-status {run_id}"
        ),
    }


def _results_csv(records: Sequence[Mapping[str, Any]]) -> str:
    """Serialize one stable per-repetition timing and resource table."""
    stream = StringIO(newline="")
    fields = (
        "variant_id",
        "cores_per_case",
        "repetition",
        "repetition_id",
        "status",
        "attempt",
        "submit_time",
        "start_time",
        "completion_time",
        "queue_wait_s",
        "turnaround_s",
        "comsol_process_s",
        "case_materialization_s",
        "export_conversion_s",
        "hdf5_admission_s",
        "complete_case_s",
        "core_hours",
        "node",
        "partition",
        "requested_cpus",
        "allocated_cpus",
        "comsol_np",
        "slurm_job_id",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        timings_value = record.get("timings_s")
        timings = timings_value if isinstance(timings_value, dict) else {}
        scheduler_value = record.get("scheduler_timing")
        scheduler = scheduler_value if isinstance(scheduler_value, dict) else {}
        resource_value = record.get("resource")
        resource = resource_value if isinstance(resource_value, dict) else {}
        complete = timings.get("complete_case")
        cores = record.get("cores_per_case")
        solve = timings.get("comsol_process")
        core_hours = (
            float(solve) * int(cores) / 3600.0
            if isinstance(solve, (int, float)) and not isinstance(solve, bool) and isinstance(cores, int) and not isinstance(cores, bool)
            else None
        )
        writer.writerow(
            {
                "variant_id": record.get("variant_id"),
                "cores_per_case": cores,
                "repetition": record.get("repetition"),
                "repetition_id": record.get("repetition_id"),
                "status": record.get("status"),
                "attempt": record.get("attempt"),
                "submit_time": scheduler.get("submit_time"),
                "start_time": scheduler.get("start_time"),
                "completion_time": scheduler.get("completion_time"),
                "queue_wait_s": scheduler.get("queue_wait_s"),
                "turnaround_s": scheduler.get("turnaround_s"),
                "comsol_process_s": timings.get("comsol_process"),
                "case_materialization_s": timings.get("case_materialization"),
                "export_conversion_s": timings.get("export_conversion"),
                "hdf5_admission_s": timings.get("hdf5_admission"),
                "complete_case_s": complete,
                "core_hours": core_hours,
                "node": resource.get("node"),
                "partition": resource.get("partition"),
                "requested_cpus": resource.get("requested_cpus"),
                "allocated_cpus": resource.get("allocated_cpus"),
                "comsol_np": resource.get("comsol_np"),
                "slurm_job_id": resource.get("slurm_job_id"),
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
    """Render one concise human-readable benchmark summary."""
    lines = [
        f"# Core-scaling benchmark: {summary['suite_name']}",
        "",
        "All rows measure the same canonical transient case in globally serial ordinary jobs.",
        "Queue wait records observed scheduler conditions and does not affect the recommendation.",
        "",
        (
            "| cores | successes | failures | median solve (s) | median case (s) | "
            "median queue (s) | median turnaround (s) | speedup | efficiency | "
            "COMSOL core-hours/case | cases/100 core-hours | campaign core-hours |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary["variants"]:
        values = (
            record["cores_per_case"],
            record["successful_repetitions"],
            record["failed_repetitions"],
            _format_metric(record["median_comsol_wall_s"]),
            _format_metric(record["median_total_case_wall_s"]),
            _format_metric(record["median_queue_wait_s"]),
            _format_metric(record["median_turnaround_s"]),
            _format_metric(record["speedup"]),
            _format_metric(record["parallel_efficiency"]),
            _format_metric(record["median_core_hours_per_case"]),
            _format_metric(record["cases_per_100_core_hours"]),
            _format_metric(record["estimated_production_campaign_core_hours"]),
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    production = summary["production_interpretation"]
    recommended = summary["recommended_production"]
    lines.extend(["", "## Recommended production setting", ""])
    if recommended is None:
        lines.append("No recommendation is available until at least one core variant has all configured repetitions successful.")
    else:
        lines.extend(
            [
                f"- cores_per_case = {recommended['cores_per_case']}",
                f"- Basis: {summary['recommendation_basis']}.",
                f"- Median COMSOL runtime: {_format_metric(recommended['median_comsol_wall_s'])} s.",
                f"- Median complete-case runtime: {_format_metric(recommended['median_total_case_wall_s'])} s.",
                f"- Median COMSOL core-hours per case: {_format_metric(recommended['median_core_hours_per_case'])}.",
                f"- Speedup: {_format_metric(recommended['speedup'])}.",
                f"- Parallel efficiency: {_format_metric(recommended['parallel_efficiency'])}.",
                (f"- Difference from current production cores per case: {recommended['difference_from_current_cores_per_case']:+d}."),
                (f"- Fastest-single-case setting: {recommended['fastest_single_case_cores_per_case']} cores per case."),
            ]
        )
    lines.extend(
        [
            "",
            f"- Fastest single case cores per case: {summary['fastest_single_case_cores_per_case']}",
            (f"- Best parallel efficiency cores per case: {summary['best_parallel_efficiency_cores_per_case']}"),
            (f"- Production estimate target: {production['campaign_total_cases']} cases from {production['campaign_config']}"),
            "- Campaign estimates are compute core-hours, not guaranteed wall-clock completion times.",
            (f"- Apply only after review: {production['cores_config']} -> {production['cores_key']}"),
            "- Production configuration changed automatically: no.",
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


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    suite: CoreBenchmarkSuite,
    manifest: Mapping[str, Any],
    proof: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind one summary revision to immutable benchmark source evidence."""
    expected_identity = {
        "schema_kind": BENCHMARK_SUMMARY_SCHEMA_KIND,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_run_id": run_id,
        "git_commit": manifest["git_commit"],
        "template_sha256": suite.case_config.template_sha256,
        "case_input_id": proof["case_input_id"],
        "simulation_case_id": proof["simulation_case_id"],
        "runtime_smoke": manifest["runtime_smoke"],
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
    proof: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind one terminal summary to its source, case, and recomputed metrics."""
    _validate_summary_identity(
        summary,
        run_id=run_id,
        suite=suite,
        manifest=manifest,
        proof=proof,
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
    """Publish deterministic aggregate evidence after every repetition attempted."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    scheduler = _scheduler_evidence([*manifest["preparation_job_ids"], *manifest["measured_job_ids"]])
    if scheduler["squeue"]["error"] is not None:
        message = f"Could not verify terminal benchmark jobs before finalization: {scheduler['squeue']['error']}"
        raise RuntimeError(message)
    if scheduler["squeue"]["output"]:
        message = "Core benchmark cannot finalize while persisted Slurm jobs remain active."
        raise RuntimeError(message)
    proof = _load_case_proof(run_id, suite, storage_root=storage_root)
    records = _result_records(directory, suite)
    _validate_records_against_proof(
        records,
        run_id=run_id,
        manifest=manifest,
        proof=proof,
    )
    pending = [record["repetition_id"] for record in records if record["status"] == "pending"]
    if pending:
        message = f"Core benchmark cannot finalize while repetitions remain pending: {pending}."
        raise RuntimeError(message)
    result_set_digest = common.serialization.canonical_json_sha256(records)
    terminal_state = "complete" if all(record["status"] == "success" for record in records) else "complete_with_failures"
    current = directory / "summary.json"
    if current.exists():
        existing = _load_json(current, label="core benchmark summary")
        if existing.get("result_set_digest") == result_set_digest:
            _validate_summary_identity(
                existing,
                run_id=run_id,
                suite=suite,
                manifest=manifest,
                proof=proof,
                records=records,
            )
            if _summary_metrics_match(existing, suite, records):
                _validate_or_repair_summary_outputs(
                    directory,
                    existing,
                    records,
                    repair_missing=True,
                )
                manifest["state"] = terminal_state
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
            "case_input_id": proof["case_input_id"],
            "simulation_case_id": proof["simulation_case_id"],
            "runtime_smoke": manifest["runtime_smoke"],
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
    manifest["state"] = terminal_state
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
    """Validate same-case proof, terminal records, summaries, and namespace isolation."""
    manifest, suite = load_core_benchmark_manifest(run_id, storage_root=storage_root)
    directory = core_benchmark_directory(run_id, storage_root=storage_root)
    proof = _load_case_proof(run_id, suite, storage_root=storage_root)
    records = _result_records(directory, suite)
    _validate_records_against_proof(
        records,
        run_id=run_id,
        manifest=manifest,
        proof=proof,
    )
    if any(record["status"] == "pending" for record in records):
        message = f"Core benchmark {run_id!r} still has pending repetitions."
        raise RuntimeError(message)
    expected_state = "complete" if all(record["status"] == "success" for record in records) else "complete_with_failures"
    if manifest["state"] != expected_state:
        message = f"Core benchmark manifest is not terminally consistent: {manifest['state']!r}."
        raise ValueError(message)
    summary = load_core_benchmark_summary(run_id, storage_root=storage_root)
    _validate_summary_payload(
        summary,
        run_id=run_id,
        suite=suite,
        manifest=manifest,
        proof=proof,
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
        ignored_names=frozenset({"transfer_complete.json"}),
    )
    return {
        "status": manifest["state"],
        "benchmark_run_id": run_id,
        "suite_digest": suite.suite_digest,
        "case_input_id": proof["case_input_id"],
        "simulation_case_id": proof["simulation_case_id"],
        "successful_repetitions": sum(record["status"] == "success" for record in records),
        "failed_repetitions": sum(record["status"] == "failed" for record in records),
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
    staged_validation = validate_core_benchmark(run_id, storage_root=staging)
    source = core_benchmark_directory(run_id, storage_root=staging)
    target = core_benchmark_directory(run_id, storage_root=destination)
    inventory = staged_validation["inventory"]
    _validate_expected_transfer_inventory(
        inventory,
        expected_sha256=expected_inventory_sha256,
        expected_file_count=expected_file_count,
        expected_size_bytes=expected_size_bytes,
    )
    if target.exists():
        target_inventory = _directory_inventory(
            target,
            ignored_names=frozenset({"transfer_complete.json"}),
        )
        if target_inventory != inventory:
            message = f"Existing benchmark publication conflicts: {target}"
            raise FileExistsError(message)
    else:
        publication_root = common.paths.get_generation_state_root(storage_root=destination) / "benchmark-transfer-publication"
        publication = workspace_service.create_publication_staging(
            storage_root=destination,
            publication_root=publication_root,
            run_id=run_id,
            case_id="benchmark-transfer",
        )
        payload = publication / "payload"
        try:
            shutil.copytree(source, payload)
            copied = _directory_inventory(
                payload,
                ignored_names=frozenset({"transfer_complete.json"}),
            )
            if copied != inventory:
                message = "Benchmark transfer copy changed the staged inventory."
                raise RuntimeError(message)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload.replace(target)
        finally:
            workspace_service.cleanup_publication_staging(
                publication,
                storage_root=destination,
                publication_root=publication_root,
                run_id=run_id,
                case_id="benchmark-transfer",
                allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
            )
    validate_core_benchmark(run_id, storage_root=destination)
    receipt_path = target / "transfer_complete.json"
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
