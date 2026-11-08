"""Compose one Dataset plan through the maintained Generation and Dataset services.

Responsibilities:
  - Validate a readable plan and explicit temporary execution overrides
  - Admit the plan's minimum success rate, from which the Generation campaign derives its admission cap unless one is explicit
  - Compute the Dataset name and the physical Source name from the resolved plan, never from authored text
  - Describe the Slurm footprint and Source run identity of a plan run for read-only preview
  - Reuse Generation admission/continuation, design-order selection and task payload publication
  - Submit the Dataset driver from the login node to run from its campaign's frozen source tree
  - Freeze Dataset-owned case roles and membership before training
  - Own the plan's physical horizon override: validated once at load and read from the plan by every action

Design principles:
  - Preview performs no admission, snapshots, solver work or Dataset writes
  - The name states the resolved content, so an override or a different design is a different name
  - A revision labels only the Dataset and reuses the default Source; a new Source is simulated afresh only on explicit request
    (``new_source``), under the same label, and changes neither seeds nor the scientific campaign digest
  - Physical Sources never receive Dataset evaluation roles
  - The minimum success rate is authored policy: it changes neither the Dataset name nor the Generation seeds
  - Membership follows Generation's design order, so only the Generation seed and the split seed shape a Dataset
  - The source is frozen on the login node only; the driver job runs from the tree and never calls Git
  - Existing stage implementations own their atomic evidence and failures
  - A published Dataset folder is complete at its rename; the driver's still-growing job log stays in its log area

This module does NOT:
  - Implement sampling, solver retries, replacement allocation, the admission cap or a stage database
  - Repartition a Dataset for a model or training seed
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import shlex
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src import common, generation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.generation.generation_controller import GenerationController

_CONFIG_KEYS = ("sources_config", "registry_config", "common_config", "operations_config", "profile_config", "execution_config")
_SPLIT_ROLES = ("train_fraction", "validation_fraction", "id_test_fraction")
_PLAN_SCHEMA_VERSION = 1
_PLAN_KEYS = {"schema_kind", "schema_version", "task", "generation", "materials", "split"}
_PLAN_TASKS = ("steady_flow", "transient_drying")
_HORIZON_TASKS = ("transient_drying",)


def load_dataset_plan(
    path: Path | str,
    *,
    counts: Sequence[str] | None = None,
    split_fractions: str | None = None,
    revision: str | None = None,
    new_source: bool = False,
    duration_hours: float | None = None,
) -> dict[str, Any]:
    """
    Resolve one authored plan and compute its Dataset and Source names without touching scientific storage.

    A transient plan names its physical horizon (``__h-<hours>``) only when the
    ``duration_hours`` override differs from the horizon of the referenced common config.

    A ``revision`` label extends only the Dataset name; the physical Source keeps
    the Generation default name and is reused. With ``new_source`` the authored
    campaign also carries ``revision``, so the Source is
    ``<default Source name>__rev-<label>`` and is simulated afresh.
    """
    raw_plan = Path(path).read_bytes()
    value = yaml.safe_load(raw_plan.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema_kind") != "dataset_plan":
        msg = "Dataset plan must be a mapping with schema_kind dataset_plan."
        raise ValueError(msg)
    if not common.serialization.is_schema_version(value.get("schema_version"), _PLAN_SCHEMA_VERSION):
        msg = f"Dataset plan schema_version must be {_PLAN_SCHEMA_VERSION}, got {value.get('schema_version')!r}."
        raise ValueError(msg)
    if set(value) != _PLAN_KEYS:
        msg = f"Dataset plan must use exactly the fields {sorted(_PLAN_KEYS)}."
        raise ValueError(msg)
    plan = copy.deepcopy(value)
    if plan["task"] not in _PLAN_TASKS:
        msg = f"Dataset plan task must be one of {_PLAN_TASKS}."
        raise ValueError(msg)
    generation_plan = plan["generation"]
    if (
        not isinstance(generation_plan, dict)
        or not {*_CONFIG_KEYS, "domain_profile", "minimum_success_rate"}.issubset(generation_plan)
        or set(generation_plan) - {*_CONFIG_KEYS, "sampling", "domain_profile", "minimum_success_rate"}
    ):
        msg = "Generation must reference the six maintained configs, sampling, domain_profile and minimum_success_rate."
        raise ValueError(msg)
    if any(not isinstance(generation_plan[key], str) or not generation_plan[key] for key in _CONFIG_KEYS):
        msg = "Generation configuration references must be non-empty paths."
        raise ValueError(msg)
    _validate_profile_task(generation_plan["profile_config"], plan["task"])
    sampling = generation_plan.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"method", "seed_base"} or sampling["method"] != "lhs":
        msg = "Dataset plan preserves the existing lhs sampling method and seed_base."
        raise ValueError(msg)
    seed = sampling["seed_base"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        msg = "Generation seed_base must be a uint32 integer."
        raise ValueError(msg)
    materials = plan["materials"]
    if not isinstance(materials, dict) or not materials:
        msg = "Dataset plan requires explicit material counts and roles."
        raise ValueError(msg)
    if counts:
        seen: set[str] = set()
        for item in counts:
            material, separator, count = item.partition("=")
            if not separator or material not in materials or material in seen:
                msg = "Count overrides require unique authored MATERIAL=POSITIVE_COUNT entries."
                raise ValueError(msg)
            materials[material]["count"] = int(count)
            seen.add(material)
    for material, specification in materials.items():
        common.paths.validate_logical_name(material, label="material")
        if not isinstance(specification, dict) or set(specification) != {"count", "role"}:
            msg = "Each material requires only count and role (id or ood)."
            raise ValueError(msg)
        if (
            isinstance(specification["count"], bool)
            or not isinstance(specification["count"], int)
            or specification["count"] < 1
            or specification["role"] not in {"id", "ood"}
        ):
            msg = "Material counts must be positive integers and roles explicit id/ood."
            raise ValueError(msg)
    generation_plan["minimum_success_rate"] = generation.cases.config.validate_minimum_success_rate(
        generation_plan["minimum_success_rate"], label="generation.minimum_success_rate"
    )
    _validate_split(plan["split"], materials, split_fractions)
    plan["duration_hours"] = duration_hours
    plan["horizon_override_hours"] = _horizon_override_hours(generation_plan, plan["task"], duration_hours)
    plan["name"] = derive_dataset_name(plan, revision=revision)
    if new_source:
        generation_plan["revision"] = generation.contracts.campaign_identity.revision_label(revision)
    plan["_authored_sha256"] = hashlib.sha256(raw_plan).hexdigest()
    return plan


def _id_split_counts(count: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    """Return the Train, validation and ID-test case counts of one ID material (floors, remainder to test)."""
    train = int(count * fractions[0])
    validation = int(count * fractions[1])
    return train, validation, count - train - validation


def derive_dataset_name(plan: Mapping[str, Any], *, revision: str | None = None) -> str:
    """Name a resolved plan by its materials, split counts, physical domain and Generation seed.

    Parameters
    ----------
    plan : Mapping[str, Any]
        Validated plan with its count, split and fraction overrides applied.
    revision : str | None, optional
        Label appended as ``__rev-<label>`` to deliberately keep a second Dataset
        of otherwise identical content; it does not name or change the Source.

    Returns
    -------
    str
        ``<material>-train<a>-val<b>-test<c>`` (``<material>-ood<n>`` for a held-out
        material) joined by ``+`` in plan order, then ``__<domain>__lhs-seed-<seed>``,
        ``__h-<hours>`` for a deviating transient horizon and the optional revision.

    Raises
    ------
    ValueError
        If the domain profile, material or revision is unsupported or the name is not one safe path component.

    """
    fractions = [plan["split"][key] for key in _SPLIT_ROLES]
    members: list[str] = []
    for material, specification in plan["materials"].items():
        slug = generation.contracts.material_registry.material_slug(material)
        if specification["role"] == "ood":
            members.append(f"{slug}-ood{specification['count']}")
        else:
            train, validation, test = _id_split_counts(specification["count"], fractions)
            members.append(f"{slug}-train{train}-val{validation}-test{test}")
    generation_plan = plan["generation"]
    domain = generation.contracts.campaign_identity.domain_token(generation_plan["domain_profile"])
    sampling = generation_plan["sampling"]
    name = f"{'+'.join(members)}__{domain}__{sampling['method']}-seed-{sampling['seed_base']}"
    if plan["horizon_override_hours"] is not None:
        name += f"__{generation.contracts.campaign_identity.horizon_token(plan['horizon_override_hours'])}"
    if revision is not None:
        name += f"__rev-{generation.contracts.campaign_identity.revision_label(revision)}"
    return common.paths.validate_logical_name(name, label="Dataset name")


def derive_source_name(plan: Mapping[str, Any]) -> str:
    """
    Name the physical Source of a resolved plan like Generation does, following the plan's own new-source choice.

    Parameters
    ----------
    plan : Mapping[str, Any]
        Validated plan with its count overrides applied; ``plan["generation"]["revision"]``
        is present only when the run asked for a new Source.

    Returns
    -------
    str
        The canonical design name ``<material>-<count>+...__<domain>__lhs-seed-<seed>``
        (plus ``__h-<hours>`` for a deviating horizon) that Generation gives an unrevised campaign, then ``__rev-<label>`` when the
        plan asks for a new Source. Seeds and the campaign digest do not change.

    Raises
    ------
    ValueError
        If the design or the revision cannot form a Source name.

    """
    sampling = plan["generation"]["sampling"]
    name = generation.contracts.campaign_identity.canonical_campaign_name(
        {material: specification["count"] for material, specification in plan["materials"].items()},
        domain_profile=plan["generation"]["domain_profile"],
        mode=sampling["method"],
        root_seed=sampling["seed_base"],
        horizon_hours=plan["horizon_override_hours"],
    )
    revision = plan["generation"].get("revision")
    return name if revision is None else generation.contracts.campaign_identity.revised_source_label(name, revision)


def _horizon_override_hours(generation_plan: Mapping[str, Any], task: str, duration_hours: float | None) -> float | None:
    """Return the transient horizon that deviates from the configured one and so enters the names, else ``None``.

    A task without a physical time horizon refuses the override at admission, before anything runs.
    """
    if duration_hours is None:
        return None
    if task not in _HORIZON_TASKS:
        msg = f"--duration-hours sets a physical time horizon, which the Dataset plan task {task!r} does not have."
        raise ValueError(msg)
    if isinstance(duration_hours, bool) or not isinstance(duration_hours, (int, float)) or not 0 < duration_hours < float("inf"):
        msg = "Physical duration override must be finite and positive."
        raise ValueError(msg)
    common_config = yaml.safe_load((common.paths.get_project_root() / generation_plan["common_config"]).read_text(encoding="utf-8"))
    configured_hours = generation.cases.config.physical_horizon_hours(common_config["time"])
    return generation.contracts.campaign_identity.deviating_horizon_hours(float(duration_hours), configured_hours)


def _validate_profile_task(profile_config: str, task: str) -> None:
    """Require the referenced Generation profile to simulate the plan's task."""
    profile_path = common.paths.get_project_root() / profile_config
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("simulation_profile") != task:
        msg = f"Generation profile {profile_config!r} does not simulate the plan task {task!r}."
        raise ValueError(msg)


def _validate_split(split: dict[str, Any], materials: Mapping[str, Any], split_fractions: str | None) -> None:
    """Validate explicit fractions with unchanged per-material floor rounding."""
    if not isinstance(split, dict) or set(split) != {"seed", "train_fraction", "validation_fraction", "id_test_fraction"}:
        msg = "Dataset split requires its own seed and three fractions."
        raise ValueError(msg)
    if isinstance(split["seed"], bool) or not isinstance(split["seed"], int) or split["seed"] < 0:
        msg = "Dataset split seed must be a non-negative integer."
        raise ValueError(msg)
    if split_fractions is not None:
        fraction_texts = split_fractions.split(",")
        if len(fraction_texts) != len(_SPLIT_ROLES):
            msg = "--split-fractions requires TRAIN,VALIDATION,TEST."
            raise ValueError(msg)
        for key, fraction_text in zip(_SPLIT_ROLES, fraction_texts, strict=True):
            split[key] = float(Fraction(fraction_text))
    # Preview checks the existing floor counts without allocating fictional cases.
    fractions = [split[key] for key in _SPLIT_ROLES]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1 for value in fractions
    ) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12, rel_tol=0):
        msg = "Split fractions must be finite, positive and sum to one."
        raise ValueError(msg)
    for material, spec in materials.items():
        if spec["role"] == "id" and min(_id_split_counts(spec["count"], fractions)) < 1:
            msg = f"Existing split rounding leaves {material} without a case in every role; explicitly change counts or fractions."
            raise ValueError(msg)


def _scientific_overrides(plan: Mapping[str, Any]) -> dict[str, float] | None:
    """Return the Generation overrides of the admitted plan horizon."""
    return None if plan["duration_hours"] is None else {"duration_hours": plan["duration_hours"]}


def _split_recipe(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": "seeded_material_fraction", **{key: plan["split"][key] for key in _SPLIT_ROLES}}


def authored_campaign(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Project physical Generation settings without Dataset names, roles or splits."""
    generation_plan = plan["generation"]
    return {
        "schema_kind": "generation_source_campaign",
        "schema_version": generation.cases.config.CAMPAIGN_SCHEMA_VERSION,
        **{key: generation_plan[key] for key in _CONFIG_KEYS},
        "sampling": {**generation_plan["sampling"], "counts": {"natural": {material: spec["count"] for material, spec in plan["materials"].items()}}},
        "materials": list(plan["materials"]),
        "minimum_success_rate": generation_plan["minimum_success_rate"],
        "domain_profile": generation_plan["domain_profile"],
        **({"revision": generation_plan["revision"]} if "revision" in generation_plan else {}),
    }


def effective_execution_overrides(requested: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the explicitly requested execution overrides of one plan run.

    Without an explicit ``max_admission_cases`` the campaign derives its cap from the plan's ``minimum_success_rate``:
    each material's count plus the failures it tolerates at each original batch's current rate. An explicit cap stays as given.
    """
    result = dict(requested or {})
    for key, value in result.items():
        if key == "no_retry":
            if not isinstance(value, bool):
                msg = "no_retry must be boolean."
                raise ValueError(msg)
        elif (
            key not in {"max_admission_cases", "node_jobs", "cores_per_case", "case_timeout_seconds"}
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            msg = "Execution overrides require existing positive integer resource controls."
            raise ValueError(msg)
    return result


def preview_dataset_plan(plan: Mapping[str, Any], *, execution_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe effective intent; do not claim attempted or successful counts."""
    source_name = derive_source_name(plan)
    return {
        "dataset": plan["name"],
        "task": plan["task"],
        "directory": f"02_datasets/{plan['task']}/{plan['name']}",
        "materials": copy.deepcopy(plan["materials"]),
        "minimum_success_rate": plan["generation"]["minimum_success_rate"],
        "generation_seed_base": plan["generation"]["sampling"]["seed_base"],
        "selection": {"kind": "design_order"},
        "split": {
            "seed_input": plan["split"]["seed"],
            "seed": common.random.derive_subseed(plan["split"]["seed"], "split"),
            "recipe": _split_recipe(plan),
        },
        "physical_duration_hours_override": plan["duration_hours"],
        "execution_overrides": effective_execution_overrides(execution_overrides),
        "source_name": source_name,
        **(
            {"new_source": f"Source {source_name} will be simulated afresh (--new-source); the default Source is left untouched."}
            if "revision" in plan["generation"]
            else {}
        ),
        "actual_outcomes": "not measured by preview",
        "collision": (
            "Same name reuses identical content; different content under the same name is refused (pass --revision <label>); "
            "identical content under another name is refused (docs/03_datasets.md, Names and identity)."
        ),
        "execution": (
            "Run in a modest nonexclusive CPU Slurm allocation. Generation submits separate solver jobs; "
            "assembly and the training payload use this allocation. STAGED returns pending without Dataset publication."
        ),
    }


def preview_scheduler_footprint(
    plan: Mapping[str, Any],
    *,
    execution_overrides: Mapping[str, Any] | None = None,
    driver_time: str,
    driver_partition: str = "standard",
) -> dict[str, Any]:
    """
    Return the Slurm footprint and Source run identity a run of this plan would ask for, without admitting anything.

    The footprint resolves the plan's execution configuration with the same overrides as the run
    (``effective_execution_overrides``), so it equals what the submitted Generation campaign validates.
    The identity is the Source campaign handle, which is fixed by the plan before any commit is bound.
    """
    execution = generation.cases.config.resolve_execution_values(
        common.paths.get_project_root() / plan["generation"]["execution_config"],
        execution_overrides=effective_execution_overrides(execution_overrides),
    )
    return {
        "identity": f"source~{plan['task']}~{derive_source_name(plan)}",
        "scheduler_footprint": generation.controller.scheduler_footprint([execution], driver_time=driver_time, driver_partition=driver_partition),
    }


def _controller() -> GenerationController:
    """Freeze the live source, uncommitted changes included, and wrap it in the Generation controller."""
    from src.generation import generation_controller as controller  # noqa: PLC0415
    from src.generation.runtime import generation_runtime_host as host  # noqa: PLC0415

    layout = host.HostLayout.resolve(common.paths.get_project_root())
    admission = host.admit_source(layout, freeze=True)
    return controller.GenerationController(host.GenerationHost(layout, admission))


def _admitted_execution(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    execution_overrides: Mapping[str, Any] | None,
    require_allocation: bool,
) -> dict[str, Any]:
    """Check the shared run guards and return the effective execution overrides."""
    preview_dataset_plan(plan, execution_overrides=execution_overrides)
    if require_allocation and not os.environ.get("SLURM_JOB_ID"):
        msg = (
            "Dataset assembly must run in a CPU Slurm allocation: start `run` on the login node, which submits its own driver job, "
            "or use --preview for read-only login inspection."
        )
        raise RuntimeError(msg)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != plan.get("_authored_sha256"):
        msg = "Authored plan changed since preview; reload and preview the current plan before running."
        raise ValueError(msg)
    return effective_execution_overrides(execution_overrides)


def submit_dataset_plan(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    invocation: Sequence[str],
    driver_partition: str,
    driver_time: str,
    execution_overrides: Mapping[str, Any] | None = None,
    continue_with_current_code: bool = False,
) -> dict[str, Any]:
    """
    Freeze the source on the login node and submit the Dataset driver to run from its campaign's tree.

    An existing campaign continues in its bound tree unless
    ``continue_with_current_code``; a new one uses the current code. The
    submitted ``invocation`` then runs :func:`run_dataset_plan` from the tree
    in its own CPU job, so neither Generation nor Dataset assembly executes live
    code or Git on a compute node. Rerunning the same command continues the plan.
    """
    from src.generation.runtime import generation_runtime_host as host  # noqa: PLC0415

    job = host.LauncherJob(
        name=f"dataset-{plan['task']}",
        log_area="datasets",
        partition=driver_partition,
        wall_time=driver_time,
    )
    execution = _admitted_execution(path, plan, execution_overrides=execution_overrides, require_allocation=False)
    controller = _controller()
    campaign_plan = controller.authored_campaign_plan(
        str(path),
        authored_campaign(plan),
        scientific_overrides=_scientific_overrides(plan),
        execution_overrides=execution,
    )
    submitted = controller.submit_run_tree(
        campaign_plan,
        module="src.datasets.dataset_packages",
        arguments=list(invocation[3:]),
        continue_with_current_code=continue_with_current_code,
        job=job,
    )
    return {"status": "SUBMITTED", "job_id": submitted.job_id, "tree_id": submitted.tree_id, "log_path": str(submitted.log_path)}


def run_dataset_plan(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    invocation: Sequence[str],
    execution_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run/reuse the existing stages from the frozen source tree inside the driver job; incomplete Sources never publish a Dataset."""
    execution = _admitted_execution(path, plan, execution_overrides=execution_overrides, require_allocation=True)
    controller = _controller()
    result = controller.run_authored_campaign(
        str(path),
        authored_campaign(plan),
        scientific_overrides=_scientific_overrides(plan),
        execution_overrides=execution,
    )
    if result.result != "OK":
        return {"status": result.result, "source_run": result.run_id, "dataset": None, "next": shlex.join(list(invocation))}
    from .packages import dataset_packages_builder as builder  # noqa: PLC0415
    from .packages import dataset_packages_selection as selection  # noqa: PLC0415

    request = {
        "schema_kind": selection.SELECTION_SCHEMA_KIND,
        "schema_version": selection.SELECTION_SCHEMA_VERSION,
        "source_task": plan["task"],
        "dataset_view": plan["task"],
        "dataset_name": plan["name"],
        "selection": {
            "kind": "design_order",
            "source_names": [result.source_name],
            "target_counts": {material: spec["count"] for material, spec in plan["materials"].items()},
        },
        "source_cases": [],
    }
    fixed_spec = {
        "seed_input": plan["split"]["seed"],
        "seed": common.random.derive_subseed(plan["split"]["seed"], "split"),
        "recipe": _split_recipe(plan),
        "material_roles": {material: spec["role"] for material, spec in plan["materials"].items()},
    }
    # One atomic publication holds the manifest and the complete payload of either task (transient: index and PT shards).
    # The still-running driver keeps writing its job log, so the log stays in its log area and never enters the immutable folder.
    publication = builder.build_dataset_selection(request, partition_spec=fixed_spec, plan_invocation=list(invocation))
    return {"status": "OK", "source_run": result.run_id, "dataset": publication}
