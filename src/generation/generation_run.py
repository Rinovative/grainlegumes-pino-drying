"""
generation_run.py

Resolve immutable Generation run plans and pure lifecycle decisions.

Responsibilities:
  - Dispatch validated run configuration schemas to their authoritative loaders
  - Describe campaign, benchmark, and paired technical-smoke execution units
  - Bind workflow identities to authored bytes, child identities, and source commits
  - Provide fail-closed lifecycle continuation decisions without executing work

Design principles:
  - Scientific campaign and benchmark identities remain owned by their existing loaders
  - Workflow orchestration identity never changes a child scientific identity
  - Lifecycle decisions are deterministic, serializable, and side-effect free

This module does NOT:
  - Submit jobs, materialize inputs, transfer artifacts, or execute solvers
  - Replace campaign, benchmark, publication, or cleanup persistence authorities
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_source as source_service

from . import generation_benchmark as benchmark_service
from . import generation_campaign as campaign_service

WORKFLOW_SCHEMA_KIND: Final = "generation_workflow"
WORKFLOW_SCHEMA_VERSION: Final = 1
CAMPAIGN_SCHEMA_KIND: Final = "generation_campaign"
BENCHMARK_SCHEMA_KIND: Final = "generation_core_scaling_benchmark_suite"
PAIRED_TECHNICAL_SMOKE_FINALIZER: Final = "paired_technical_smoke"
CORE_BENCHMARK_SUMMARY_FINALIZER: Final = "core_benchmark_summary"
MATERIAL_PILOT_FINALIZER: Final = "material_pilot"
_PAIRED_TECHNICAL_SMOKE_CHILD_COUNT: Final = 2

LIFECYCLE_STAGES: Final = (
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
    "build_packages",
    "finalize",
    "record_workflow",
    "authorize_cleanup",
    "cleanup_source",
    "record_cleanup",
    "validate",
    "complete",
)
RUN_STATES: Final = frozenset(
    {
        "planned",
        "preflight_ready",
        "inputs_ready",
        "running",
        "license_blocked",
        "cpu_complete",
        "awaiting_collection",
        "collecting",
        "host_complete",
        "packages_complete",
        "finalizing",
        "complete",
        "failed",
        "cancelled",
    }
)
_TERMINAL_STATES: Final = frozenset({"complete", "failed", "cancelled"})
_COLLECTION_STAGES: Final = frozenset({"transfer", "publish", "build_packages"})


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
        ``campaign``, ``benchmark``, or ``workflow``.
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
    dataset_packages : tuple[str, ...]
        Declared dataset package identities.
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
    dataset_packages: tuple[str, ...]
    finalizers: tuple[GenerationRunFinalizer, ...]
    lifecycle_stages: tuple[str, ...]

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
            "children": [child.to_payload() for child in self.children],
            "units": [unit.to_payload() for unit in self.units],
            "retention_policy": self.retention_policy,
            "collection_policy": self.collection_policy,
            "dataset_packages": list(self.dataset_packages),
            "finalizers": [item.to_payload() for item in self.finalizers],
            "lifecycle_stages": list(self.lifecycle_stages),
        }


@dataclass(frozen=True, slots=True)
class GenerationRunController:
    """
    Pure, fail-closed lifecycle state for one immutable run plan.

    Parameters
    ----------
    plan : GenerationRunPlan
        Immutable run plan being continued.
    completed_stages : frozenset[str]
        Persisted lifecycle stages already completed in their declared order.
    state : str
        Current common lifecycle state.
    defer_collection : bool
        Hold transfer and publication while the validated CPU source is retained.

    """

    plan: GenerationRunPlan
    completed_stages: frozenset[str] = frozenset()
    state: str = "planned"
    defer_collection: bool = False

    def __post_init__(self) -> None:
        """Validate persisted continuation state against the immutable plan."""
        if self.state not in RUN_STATES:
            message = f"Unknown Generation run state {self.state!r}."
            raise ValueError(message)
        unknown = self.completed_stages.difference(self.plan.lifecycle_stages)
        if unknown:
            message = f"Completed lifecycle stages are not declared by the plan: {sorted(unknown)}."
            raise ValueError(message)
        expected = self.plan.lifecycle_stages[: len(self.completed_stages)]
        if set(expected) != set(self.completed_stages):
            message = "Completed lifecycle stages must be an ordered prefix of the declared plan lifecycle."
            raise ValueError(message)
        if self.state == "complete" and "complete" not in self.completed_stages:
            message = "Complete Generation run state requires the complete lifecycle stage."
            raise ValueError(message)
        if "complete" in self.completed_stages and self.state != "complete":
            message = "The complete lifecycle stage requires complete Generation run state."
            raise ValueError(message)
        if self.defer_collection and self.plan.collection_policy == "none":
            message = "Collection cannot be deferred for a run plan without collection stages."
            raise ValueError(message)

    def next_stage(self) -> str | None:
        """Return the next executable stage, or ``None`` when continuation must wait."""
        if self.state in _TERMINAL_STATES:
            return None
        for stage in self.plan.lifecycle_stages:
            if stage in self.completed_stages:
                continue
            if stage in _COLLECTION_STAGES and self.defer_collection:
                return None
            return stage
        return None

    def continuation_state(self) -> str:
        """Return the visible state that explains the next continuation decision."""
        if self.state in _TERMINAL_STATES or self.state == "license_blocked":
            return self.state
        pending = self.next_stage()
        if pending is None:
            if self.defer_collection:
                return "awaiting_collection"
            return self.state
        if pending == "preflight":
            return "preflight_ready"
        if pending == "prepare_inputs":
            return "inputs_ready"
        if pending in _COLLECTION_STAGES:
            return "collecting"
        if pending == "finalize":
            return "finalizing"
        return "running" if self.completed_stages else "planned"

    def advance(self, stage: str) -> GenerationRunController:
        """Return state after exactly the next declared lifecycle stage completes."""
        expected = self.next_stage()
        if expected is None:
            message = f"Generation run cannot advance while state is {self.continuation_state()!r}."
            raise RuntimeError(message)
        if stage != expected:
            message = f"Invalid Generation lifecycle transition: expected {expected!r}, got {stage!r}."
            raise ValueError(message)
        completed = self.completed_stages | {stage}
        state = _state_after_stage(stage)
        return replace(self, completed_stages=completed, state=state)

    def resume(
        self,
        *,
        defer_collection: bool | None = None,
    ) -> GenerationRunController:
        """Return a validated continuation view without changing plan identity."""
        return replace(
            self,
            defer_collection=self.defer_collection if defer_collection is None else defer_collection,
            state=self.state,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return deterministic serializable continuation state."""
        return {
            "schema_kind": "generation_run_controller",
            "schema_version": 1,
            "plan_identity": self.plan.identity,
            "completed_stages": [stage for stage in self.plan.lifecycle_stages if stage in self.completed_stages],
            "state": self.continuation_state(),
            "defer_collection": self.defer_collection,
            "next_stage": self.next_stage(),
        }


def resolve_generation_run(
    path: Path | str,
    *,
    source_commit: str,
    repository_root: Path | str | None = None,
    require_executable: bool = True,
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
    schema_kind = raw.get("schema_kind")
    if schema_kind == CAMPAIGN_SCHEMA_KIND:
        _validate_schema_version(raw, schema_kind=CAMPAIGN_SCHEMA_KIND, version=1)
        return _campaign_plan(
            source_path,
            config_identity=config_identity,
            source_commit=commit,
            root=root,
            require_executable=require_executable,
        )
    if schema_kind == BENCHMARK_SCHEMA_KIND:
        _validate_schema_version(
            raw,
            schema_kind=BENCHMARK_SCHEMA_KIND,
            version=benchmark_service.BENCHMARK_SCHEMA_VERSION,
        )
        return _benchmark_plan(
            source_path,
            config_identity=config_identity,
            source_commit=commit,
            root=root,
            require_executable=require_executable,
        )
    if schema_kind == WORKFLOW_SCHEMA_KIND:
        return _workflow_plan(
            raw=raw,
            source_path=source_path,
            config_identity=config_identity,
            source_commit=commit,
            root=root,
            require_executable=require_executable,
        )
    message = (
        f"Unsupported Generation run schema_kind {schema_kind!r} in {source_path}; "
        f"expected {CAMPAIGN_SCHEMA_KIND!r}, {BENCHMARK_SCHEMA_KIND!r}, or {WORKFLOW_SCHEMA_KIND!r}."
    )
    raise ValueError(message)


def _campaign_plan(
    source_path: Path,
    *,
    config_identity: str,
    source_commit: str,
    root: Path,
    require_executable: bool,
) -> GenerationRunPlan:
    """Build one leaf plan from the authoritative campaign resolver."""
    campaign = config_service.load_campaign_config(source_path, require_executable=require_executable)
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
    package_names = tuple(str(package["dataset_name"]) for package in campaign.dataset_packages)
    if len(package_names) != len(set(package_names)):
        message = "Campaign plan resolved duplicate declared Dataset package names."
        raise ValueError(message)
    retention = str(campaign.execution_values["retention_policy"])
    collection = "deferred_allowed"
    finalizers = (
        (
            GenerationRunFinalizer(
                finalizer_kind=MATERIAL_PILOT_FINALIZER,
                required_child_identities=(),
            ),
        )
        if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE
        else ()
    )
    lifecycle = _campaign_lifecycle(
        has_packages=bool(package_names),
        has_finalizer=bool(finalizers),
    )
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
        dataset_packages=package_names,
        finalizers=finalizers,
        lifecycle_stages=lifecycle,
    )


def _benchmark_plan(
    source_path: Path,
    *,
    config_identity: str,
    source_commit: str,
    root: Path,
    require_executable: bool,
) -> GenerationRunPlan:
    """Build two-case work units in production-first sequential waves."""
    suite = benchmark_service.load_core_benchmark_suite(
        source_path,
        require_executable=require_executable,
    )
    units: list[GenerationRunUnit] = []
    previous_wave: tuple[str, ...] = ()
    for wave_position, variant in enumerate(suite.variant_wave_order(), start=1):
        current_wave: list[str] = []
        for case_position in range(1, suite.representative_case_count + 1):
            representative = suite.representative_case(case_position)
            case_index = representative.case_index
            unit_id = suite.work_unit_id(variant, case_position)
            current_wave.append(unit_id)
            units.append(
                GenerationRunUnit(
                    unit_id=unit_id,
                    unit_kind="benchmark_case",
                    input_identity=_case_input_identity(
                        suite.case_config,
                        case_index,
                    ),
                    metadata=(
                        ("variant_id", variant.variant_id),
                        ("cores_per_case", variant.cores_per_case),
                        ("wave_position", wave_position),
                        ("case_position", case_position),
                        ("case_role", representative.case_role),
                        ("case_index", case_index),
                        ("case_id", suite.case_config.case_id(case_index)),
                        ("canary", wave_position == 1),
                        ("included_in_final_measurements", True),
                    ),
                    depends_on=previous_wave,
                )
            )
        previous_wave = tuple(current_wave)
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
        dataset_packages=(),
        finalizers=(
            GenerationRunFinalizer(
                finalizer_kind=CORE_BENCHMARK_SUMMARY_FINALIZER,
                required_child_identities=(),
            ),
        ),
        lifecycle_stages=_benchmark_lifecycle(),
    )


def _workflow_plan(
    *,
    raw: Mapping[str, Any],
    source_path: Path,
    config_identity: str,
    source_commit: str,
    root: Path,
    require_executable: bool,
) -> GenerationRunPlan:
    """Resolve the sole supported paired technical-smoke workflow contract."""
    _exact_keys(
        raw,
        {"schema_kind", "schema_version", "workflow_name", "children", "finalizers"},
        label="Generation workflow",
    )
    if raw["schema_version"] != WORKFLOW_SCHEMA_VERSION:
        message = f"Unsupported Generation workflow schema version {raw['schema_version']!r}."
        raise ValueError(message)
    workflow_name = _logical_name(raw["workflow_name"], label="workflow_name")
    children_raw = raw["children"]
    if not isinstance(children_raw, list) or len(children_raw) != _PAIRED_TECHNICAL_SMOKE_CHILD_COUNT:
        message = "paired technical-smoke workflow requires exactly two ordered campaign children."
        raise ValueError(message)
    references = tuple(_workflow_reference(value, root=root, index=index) for index, value in enumerate(children_raw))
    if len(references) != len(set(references)):
        message = "Generation workflow children must be duplicate-free."
        raise ValueError(message)
    children = tuple(
        resolve_generation_run(
            reference,
            source_commit=source_commit,
            repository_root=root,
            require_executable=require_executable,
        )
        for reference in references
    )
    if any(child.run_kind != "campaign" for child in children):
        message = "paired technical-smoke workflow children must resolve generation_campaign configurations."
        raise ValueError(message)
    if any(child.children or child.finalizers for child in children):
        message = "Generation workflow nesting is unsupported."
        raise ValueError(message)
    if any(_unit_metadata(unit, "campaign_purpose") != "technical_runtime_smoke" for child in children for unit in child.units):
        message = "paired technical-smoke workflow children must have technical_runtime_smoke purpose."
        raise ValueError(message)
    identities = tuple(child.identity for child in children)
    if len(identities) != len(set(identities)):
        message = "Generation workflow children resolve duplicate campaign identities."
        raise ValueError(message)
    finalizers = _workflow_finalizers(raw["finalizers"], child_identities=identities)
    profiles = tuple(_unit_metadata(child.units[0], "simulation_profile") for child in children)
    if profiles != ("steady_flow", "transient_drying"):
        message = "paired technical-smoke workflow children must be ordered steady_flow then transient_drying."
        raise ValueError(message)
    if any(_unit_metadata(unit, "simulation_profile") != profile for child, profile in zip(children, profiles, strict=True) for unit in child.units):
        message = "paired technical-smoke workflow child units must use exactly one simulation profile."
        raise ValueError(message)
    payload = {
        "schema_kind": WORKFLOW_SCHEMA_KIND,
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_name": workflow_name,
        "config_identity": config_identity,
        "children": [child.identity for child in children],
        "source_commit": source_commit,
        "finalizers": [item.to_payload() for item in finalizers],
    }
    identity = f"workflow__{common.serialization.canonical_json_sha256(payload)}"
    return GenerationRunPlan(
        run_kind="workflow",
        identity=identity,
        source_commit=source_commit,
        input_identity=common.serialization.canonical_json_sha256({"children": [child.input_identity for child in children]}),
        config_identity=config_identity,
        config_path=source_path.relative_to(root).as_posix(),
        children=children,
        units=(),
        retention_policy="full",
        collection_policy="deferred_allowed",
        dataset_packages=tuple(package for child in children for package in child.dataset_packages),
        finalizers=finalizers,
        lifecycle_stages=_workflow_lifecycle(),
    )


def _workflow_finalizers(value: Any, *, child_identities: tuple[str, ...]) -> tuple[GenerationRunFinalizer, ...]:
    """Validate and resolve the only supported workflow finalizer."""
    if value != [PAIRED_TECHNICAL_SMOKE_FINALIZER]:
        message = "Generation workflow finalizers must be exactly ['paired_technical_smoke']."
        raise ValueError(message)
    return (
        GenerationRunFinalizer(
            finalizer_kind=PAIRED_TECHNICAL_SMOKE_FINALIZER,
            required_child_identities=child_identities,
        ),
    )


def _campaign_lifecycle(
    *,
    has_packages: bool,
    has_finalizer: bool,
) -> tuple[str, ...]:
    """Return campaign stages, omitting undeclared package and finalizer work."""
    stages = [
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
    ]
    if has_packages:
        stages.append("build_packages")
    if has_finalizer:
        stages.append("finalize")
    stages.extend(
        (
            "record_workflow",
            "authorize_cleanup",
            "cleanup_source",
            "record_cleanup",
            "validate",
            "complete",
        )
    )
    return tuple(stages)


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


def _workflow_lifecycle() -> tuple[str, ...]:
    """Return the parent continuation after child-owned stages."""
    return (
        "resolve_config",
        "finalize",
        "validate",
        "complete",
    )


def _state_after_stage(stage: str) -> str:
    """Map one completed lifecycle stage to the corresponding common state."""
    states = {
        "resolve_config": "preflight_ready",
        "preflight": "preflight_ready",
        "prepare_inputs": "inputs_ready",
        "terminalize": "cpu_complete",
        "validate_terminal": "cpu_complete",
        "transfer": "collecting",
        "publish": "host_complete",
        "build_packages": "packages_complete",
        "finalize": "host_complete",
        "complete": "complete",
    }
    return states.get(stage, "running")


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


def _workflow_reference(value: Any, *, root: Path, index: int) -> Path:
    """Resolve a repository-root-relative workflow child reference safely."""
    if not isinstance(value, str) or not value:
        message = f"Generation workflow children[{index}] must be a non-empty relative path."
        raise TypeError(message)
    reference = Path(value)
    if reference.is_absolute():
        message = f"Generation workflow children[{index}] must not use an absolute path."
        raise ValueError(message)
    return _repository_file(root / reference, root=root, label=f"Generation workflow children[{index}]")


def _validate_schema_version(raw: Mapping[str, Any], *, schema_kind: str, version: int) -> None:
    """Require the supported version before dispatching to an authoritative loader."""
    if raw.get("schema_version") != version:
        message = f"Unsupported {schema_kind} schema version {raw.get('schema_version')!r}."
        raise ValueError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    """Require one exact YAML mapping key set."""
    actual = set(value)
    if actual != expected:
        message = f"{label} keys must be exactly {sorted(expected)}, got {sorted(actual)}."
        raise ValueError(message)


def _logical_name(value: Any, *, label: str) -> str:
    """Validate one stable workflow display identifier."""
    if not isinstance(value, str):
        message = f"{label} must be text."
        raise TypeError(message)
    return common.paths.validate_logical_name(value, label=label)


def _unit_metadata(unit: GenerationRunUnit, key: str) -> Any:
    """Return one required immutable metadata value."""
    for item_key, value in unit.metadata:
        if item_key == key:
            return value
    message = f"Generation run unit {unit.unit_id!r} lacks required metadata {key!r}."
    raise RuntimeError(message)
