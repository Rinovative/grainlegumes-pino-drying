"""Resolve immutable Generation run plans.

Responsibilities:
  - Dispatch validated run configuration schemas to their authoritative loaders
  - Describe campaign and benchmark execution units

Design principles:
  - Scientific campaign and benchmark identities remain owned by their existing loaders
  - Workflow orchestration identity never changes a child scientific identity
  - Run plans are deterministic, serializable, and side-effect free

This module does NOT:
  - Submit jobs, materialize inputs, transfer artifacts, or execute solvers
  - Replace campaign, benchmark, or publication persistence authorities
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_source as source_service

from . import generation_benchmark_config as benchmark_config
from . import generation_campaign as campaign_service

CAMPAIGN_SCHEMA_KIND: Final = "generation_source_campaign"
BENCHMARK_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_suite"
CORE_BENCHMARK_SUMMARY_FINALIZER: Final = "core_benchmark_summary"


@dataclass(frozen=True, slots=True)
class GenerationRunUnit:
    """
    One ordered, immutable unit of planned generation work.

    Parameters
    ----------
    unit_id : str
        Stable identifier within the containing run plan.
    unit_kind : str
        Declared unit category: ``campaign_case`` or ``benchmark_case``.
    input_identity : str
        Existing immutable input identity for the work unit.
    metadata : tuple[tuple[str, Any], ...]
        JSON-serializable ownership metadata required by the unit consumer.
    depends_on : tuple[str, ...]
        Work-unit identities that must complete before this unit is eligible.

    """

    unit_id: str
    unit_kind: str
    input_identity: str
    metadata: tuple[tuple[str, Any], ...]
    depends_on: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable unit payload."""
        return {
            "unit_id": self.unit_id,
            "unit_kind": self.unit_kind,
            "input_identity": self.input_identity,
            "metadata": dict(self.metadata),
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True, slots=True)
class GenerationRunFinalizer:
    """
    One declared continuation required after all child runs complete.

    Parameters
    ----------
    finalizer_kind : str
        Stable finalizer discriminator.
    required_child_identities : tuple[str, ...]
        Ordered child identities that must be terminal before finalization.

    """

    finalizer_kind: str
    required_child_identities: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable finalizer payload."""
        return {
            "finalizer_kind": self.finalizer_kind,
            "required_child_identities": list(self.required_child_identities),
        }


@dataclass(frozen=True, slots=True)
class GenerationRunPlan:
    """
    One resolved immutable run or workflow plan.

    Parameters
    ----------
    run_kind : str
        ``campaign`` or ``benchmark``.
    identity : str
        Run-level immutable identity, distinct from display names and paths.
    source_commit : str
        Exact source commit bound to lifecycle execution.
    input_identity : str
        Scientific input identity supplied by the authoritative loader.
    config_identity : str
        SHA-256 digest of the authored top-level configuration bytes.
    config_path : str
        Validated repository-contained configuration path used by the adapter.
    children : tuple[GenerationRunPlan, ...]
        Ordered child plans for a workflow; empty for executable leaf plans.
    units : tuple[GenerationRunUnit, ...]
        Ordered executable work units for a leaf plan.
    retention_policy : str
        Declared source-attempt retention policy.
    collection_policy : str
        Declared collection continuation policy.
    finalizers : tuple[GenerationRunFinalizer, ...]
        Continuations that follow workflow child completion.
    lifecycle_stages : tuple[str, ...]
        Ordered subset of the common lifecycle vocabulary.

    """

    run_kind: str
    identity: str
    source_commit: str
    input_identity: str
    config_identity: str
    config_path: str
    children: tuple[GenerationRunPlan, ...]
    units: tuple[GenerationRunUnit, ...]
    retention_policy: str
    collection_policy: str
    finalizers: tuple[GenerationRunFinalizer, ...]
    lifecycle_stages: tuple[str, ...]
    scientific_overrides: tuple[tuple[str, Any], ...] = ()
    execution_overrides: tuple[tuple[str, Any], ...] = ()
    execution_values: Mapping[str, Any] | None = None
    authored_value: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable plan payload."""
        return {
            "schema_kind": "generation_run_plan",
            "schema_version": 1,
            "run_kind": self.run_kind,
            "identity": self.identity,
            "source_commit": self.source_commit,
            "input_identity": self.input_identity,
            "config_identity": self.config_identity,
            "config_path": self.config_path,
            "scientific_overrides": dict(self.scientific_overrides),
            "execution_overrides": dict(self.execution_overrides),
            "execution_values": dict(self.execution_values or {}),
            "authored_value": None if self.authored_value is None else dict(self.authored_value),
            "children": [child.to_payload() for child in self.children],
            "units": [unit.to_payload() for unit in self.units],
            "retention_policy": self.retention_policy,
            "collection_policy": self.collection_policy,
            "finalizers": [item.to_payload() for item in self.finalizers],
            "lifecycle_stages": list(self.lifecycle_stages),
        }


def resolve_generation_run(
    path: Path | str,
    *,
    source_commit: str,
    repository_root: Path | str | None = None,
    require_executable: bool = True,
    scientific_overrides: Mapping[str, Any] | None = None,
    execution_overrides: Mapping[str, Any] | None = None,
    authored_value: Mapping[str, Any] | None = None,
) -> GenerationRunPlan:
    """
    Resolve one supported configuration into an immutable run plan.

    Parameters
    ----------
    path : Path | str
        Repository-contained campaign, benchmark-suite, or workflow YAML file.
    source_commit : str
        Exact lowercase Git commit bound to execution provenance.
    repository_root : Path | str | None, optional
        Repository root used for containment validation and relative workflow children.
    require_executable : bool, optional
        Forwarded to the authoritative campaign and benchmark loaders.

    scientific_overrides : Mapping[str, Any] | None, optional
        Explicit campaign material, design-size, physical-duration and Source-revision overrides.
    execution_overrides : Mapping[str, Any] | None, optional
        Explicit normal concurrency, CPU, timeout and retry choices.
    authored_value : Mapping[str, Any] | None, optional
        In-memory campaign configuration with ``path`` retained as authored plan evidence.

    Returns
    -------
    GenerationRunPlan
        Side-effect-free resolved plan.

    Raises
    ------
    FileNotFoundError
        If the configuration or a declared workflow child is unavailable.
    ValueError
        If schemas, references, identities, or supported workflow contracts are invalid.

    """
    root = _repository_root(repository_root)
    source_path = _repository_file(path, root=root, label="Generation run configuration")
    commit = source_service.validate_git_commit(source_commit)
    raw, config_identity = _load_top_level(source_path)
    if authored_value is not None:
        raw = MappingProxyType(dict(authored_value))
    schema_kind = raw.get("schema_kind")
    if schema_kind == CAMPAIGN_SCHEMA_KIND:
        _validate_schema_version(
            raw,
            schema_kind=CAMPAIGN_SCHEMA_KIND,
            version=config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION
            if raw.get("schema_version") == config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION
            else 1,
        )
        return _campaign_plan(
            source_path,
            config_identity=config_identity,
            source_commit=commit,
            root=root,
            require_executable=require_executable,
            scientific_overrides=scientific_overrides,
            execution_overrides=execution_overrides,
            authored_value=authored_value,
        )
    if schema_kind == BENCHMARK_SCHEMA_KIND:
        if authored_value is not None or scientific_overrides or execution_overrides:
            message = "Scientific run overrides apply to source campaigns only."
            raise ValueError(message)
        _validate_schema_version(
            raw,
            schema_kind=BENCHMARK_SCHEMA_KIND,
            version=benchmark_config.BENCHMARK_SCHEMA_VERSION,
        )
        return _benchmark_plan(
            source_path,
            config_identity=config_identity,
            source_commit=commit,
            root=root,
            require_executable=require_executable,
        )
    message = (
        f"Unsupported Generation run schema_kind {schema_kind!r} in {source_path}; expected {CAMPAIGN_SCHEMA_KIND!r} or {BENCHMARK_SCHEMA_KIND!r}."
    )
    raise ValueError(message)


def _campaign_plan(
    source_path: Path,
    *,
    config_identity: str,
    source_commit: str,
    root: Path,
    require_executable: bool,
    scientific_overrides: Mapping[str, Any] | None = None,
    execution_overrides: Mapping[str, Any] | None = None,
    authored_value: Mapping[str, Any] | None = None,
) -> GenerationRunPlan:
    """Build one leaf plan from the authoritative campaign resolver."""
    campaign = config_service.load_campaign_config(
        source_path,
        require_executable=require_executable,
        scientific_overrides=scientific_overrides,
        execution_overrides=execution_overrides,
        authored_value=authored_value,
    )
    units: list[GenerationRunUnit] = []
    for batch in campaign.batches:
        for case_index in batch.case_indices:
            case_id = batch.case_id(case_index)
            input_identity = _case_input_identity(batch, case_index)
            units.append(
                GenerationRunUnit(
                    unit_id=f"{batch.batch_id}:{case_id}",
                    unit_kind="campaign_case",
                    input_identity=input_identity,
                    metadata=(
                        ("batch_id", batch.batch_id),
                        ("case_id", case_id),
                        ("case_index", case_index),
                        ("material_family", batch.material_family),
                        ("sampling_regime", batch.sampling_regime),
                        ("simulation_profile", batch.profile.id),
                        ("campaign_purpose", campaign.campaign_purpose),
                    ),
                )
            )
    retention = str(campaign.execution_values["retention_policy"])
    collection = "deferred_allowed"
    finalizers: tuple[GenerationRunFinalizer, ...] = ()
    lifecycle = _campaign_lifecycle()
    return GenerationRunPlan(
        run_kind="campaign",
        identity=campaign_service.campaign_run_id(campaign, git_commit=source_commit),
        source_commit=source_commit,
        input_identity=campaign.campaign_digest,
        config_identity=config_identity,
        config_path=source_path.relative_to(root).as_posix(),
        children=(),
        units=tuple(units),
        retention_policy=retention,
        collection_policy=collection,
        finalizers=finalizers,
        lifecycle_stages=lifecycle,
        scientific_overrides=tuple((campaign.scientific_overrides or {}).items()),
        execution_overrides=tuple((campaign.execution_overrides or {}).items()),
        execution_values=campaign.execution_values,
        authored_value=campaign.authored_value,
    )


def _benchmark_plan(
    source_path: Path,
    *,
    config_identity: str,
    source_commit: str,
    root: Path,
    require_executable: bool,
) -> GenerationRunPlan:
    """Build first-attempt work units per sweep: one group per variant in sweep order, each group after the previous one."""
    suite = benchmark_config.load_core_benchmark_suite(
        source_path,
        require_executable=require_executable,
    )
    units: list[GenerationRunUnit] = []
    for sweep in suite.sweeps:
        previous_group: tuple[str, ...] = ()
        for group, variant in enumerate(suite.sweep_variants(sweep)):
            current_group: list[str] = []
            for case_position in range(1, suite.representative_case_count + 1):
                representative = suite.representative_case(case_position)
                case_index = representative.case_index
                unit_id = suite.work_unit_id(sweep, variant, 1, case_position)
                current_group.append(unit_id)
                units.append(
                    GenerationRunUnit(
                        unit_id=unit_id,
                        unit_kind="benchmark_case",
                        input_identity=_case_input_identity(
                            suite.case_config,
                            case_index,
                        ),
                        metadata=(
                            ("sweep_id", sweep.sweep_id),
                            ("variant_id", variant.variant_id),
                            ("cores_per_case", variant.cores_per_case),
                            ("concurrent_cases", suite.resources(variant).concurrent_cases),
                            ("group", group),
                            ("case_position", case_position),
                            ("case_role", representative.case_role),
                            ("case_index", case_index),
                            ("case_id", suite.case_config.case_id(case_index)),
                            ("success_target", suite.target_successful_cases),
                        ),
                        depends_on=previous_group,
                    )
                )
            previous_group = tuple(current_group)
    return GenerationRunPlan(
        run_kind="benchmark",
        identity=(
            "benchmark_plan__"
            + common.serialization.canonical_json_sha256(
                {
                    "suite_digest": suite.suite_digest,
                    "source_commit": source_commit,
                }
            )[:16]
        ),
        source_commit=source_commit,
        input_identity=suite.suite_digest,
        config_identity=config_identity,
        config_path=source_path.relative_to(root).as_posix(),
        children=(),
        units=tuple(units),
        retention_policy="compact",
        collection_policy="deferred_allowed",
        finalizers=(
            GenerationRunFinalizer(
                finalizer_kind=CORE_BENCHMARK_SUMMARY_FINALIZER,
                required_child_identities=(),
            ),
        ),
        lifecycle_stages=_benchmark_lifecycle(),
    )


def _campaign_lifecycle() -> tuple[str, ...]:
    """Return the ordinary source-campaign stages actually executed."""
    return (
        "resolve_config",
        "preflight",
        "prepare_inputs",
        "submit",
        "feed_scheduler",
        "monitor_scheduler",
        "terminalize",
        "validate_terminal",
        "complete",
    )


def _benchmark_lifecycle() -> tuple[str, ...]:
    """Return benchmark stages without a separate prepare worker unit."""
    return (
        "resolve_config",
        "preflight",
        "prepare_inputs",
        "plan",
        "submit",
        "feed_scheduler",
        "monitor_scheduler",
        "terminalize",
        "validate_terminal",
        "transfer",
        "publish",
        "finalize",
        "record_workflow",
        "authorize_cleanup",
        "cleanup_source",
        "record_cleanup",
        "validate",
        "complete",
    )


def _case_input_identity(batch: Any, case_index: int) -> str:
    """Derive a stable case input identity from resolved authoritative fields."""
    payload = {
        "batch_id": batch.batch_id,
        "case_index": case_index,
        "case_seed": batch.case_seed(case_index),
        "case_input_config_digest": batch.case_input_config_digest,
    }
    return common.serialization.canonical_json_sha256(payload)


def _load_top_level(path: Path) -> tuple[Mapping[str, Any], str]:
    """Read one YAML object while preserving an exact-byte configuration identity."""
    try:
        data = path.read_bytes()
    except OSError as error:
        message = f"Generation run configuration is unreadable: {path}"
        raise FileNotFoundError(message) from error
    try:
        raw = yaml.safe_load(data)
    except yaml.YAMLError as error:
        message = f"Generation run configuration is not valid YAML: {path}"
        raise ValueError(message) from error
    if not isinstance(raw, Mapping):
        message = f"Generation run configuration must be one YAML object: {path}"
        raise TypeError(message)
    return MappingProxyType(dict(raw)), hashlib.sha256(data).hexdigest()


def _repository_root(value: Path | str | None) -> Path:
    """Resolve one non-symlink repository root."""
    root = common.paths.get_project_root().resolve() if value is None else Path(value).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        message = f"Generation repository root is missing or unsafe: {root}"
        raise FileNotFoundError(message)
    return root


def _repository_file(value: Path | str, *, root: Path, label: str) -> Path:
    """Resolve one existing non-symlink YAML path contained by the repository."""
    candidate = Path(value).expanduser()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        message = f"{label} must be contained by the repository root: {candidate}"
        raise ValueError(message) from error
    if not resolved.is_file() or resolved.is_symlink():
        message = f"{label} is missing or unsafe: {resolved}"
        raise FileNotFoundError(message)
    return resolved


def _validate_schema_version(raw: Mapping[str, Any], *, schema_kind: str, version: int) -> None:
    """Require the supported version before dispatching to an authoritative loader."""
    if raw.get("schema_version") != version:
        message = f"Unsupported {schema_kind} schema version {raw.get('schema_version')!r}."
        raise ValueError(message)
