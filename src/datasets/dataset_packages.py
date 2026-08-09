"""
===============================================================================
dataset_packages.py
===============================================================================
Assemble immutable dual-view dataset packages from terminal simulation cases.
Responsibilities:
  - Assign shared case-level ID membership before view-specific sample expansion
  - Audit steady conditioning and task-specific parameter-OOD eligibility
  - Publish provenance-bound steady tensors or compact transient indexes
Design principles:
  - Dataset views, regimes, memberships, and source decisions are explicit
  - One combined parameter-OOD package owns compact group/parameter indexes
  - Existing valid identities are reused and conflicting publication fails closed
This module does NOT:
  - Fit normalization, train models, duplicate HDF5 cases, or register transient
  - Infer scientific relevance from filenames, filesystem order, or group names
===============================================================================
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch

from src import common, domain
from src.generation import generation_config as config_service
from src.generation import generation_profiles as profiles

from . import dataset_generated_batch as generated
from . import dataset_identity as identity
from . import dataset_transient as transient
from . import dataset_views as views

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

DATASET_PACKAGE_SCHEMA_KIND: Final = "vp2_dataset_package_manifest"
DATASET_PACKAGE_SCHEMA_VERSION: Final = 2
_ID_MEMBERSHIP_ROLES: Final = views.ID_MEMBERSHIPS
_STEADY_SOLVER_INPUTS: Final = ("Kxx", "Kxy", "Kyy", "eps_bed", "p_bc")
_PACKAGE_PROVENANCE_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_name",
        "dataset_view",
        "registered_task_id",
        "evaluation_regime",
        "materials",
        "channel_contract",
        "channel_contract_digest",
        "source_simulation_profiles",
        "source_batches",
        "source_batch_ids",
        "source_template_digests",
        "source_git_commits",
        "source_case_identities",
        "included_source_cases",
        "excluded_source_cases",
        "source_selection_decisions",
        "matched_case_input_ids",
        "airflow_provenance",
        "steady_flow_conditioning",
        "material_file_identities",
        "operation_config_digests",
        "campaign_name",
        "campaign_digest",
        "duplicate_case_input_policy",
        "case_membership",
        "split_membership",
        "membership_counts",
        "available_ood_groups",
        "ood_group_indexes",
        "ood_parameter_indexes",
        "task_relevant_ood_parameters",
        "material_counts",
        "source_profile_counts",
        "candidate_source_case_count",
        "builder_identity",
        "schema_identity",
    }
)
_PACKAGE_MANIFEST_KEYS: Final = _PACKAGE_PROVENANCE_KEYS | {
    "dataset_id",
    "dataset_digest",
    "payload_filename",
    "sample_count",
    "source_case_count",
    "transition_count",
    "payload_sha256",
}


@dataclass(slots=True)
class _PreparedPackage:
    """Hold one completely admitted package before payload publication."""

    plan: dict[str, Any]
    batch_records: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    membership: dict[str, list[str]]
    source_decisions: list[dict[str, Any]]
    steady_conditioning: dict[str, Any] | None


def _package_plan(
    campaign: config_service.CampaignConfig,
    dataset_view: str,
    evaluation_regime: str,
) -> dict[str, Any]:
    """Return one exact declared view/regime package plan."""
    matches = [
        copy.deepcopy(package)
        for package in campaign.dataset_packages
        if package["dataset_view"] == dataset_view and package["evaluation_regime"] == evaluation_regime
    ]
    if len(matches) != 1:
        message = f"Campaign {campaign.campaign_name!r} must declare exactly one {dataset_view!r}/{evaluation_regime!r} package."
        raise ValueError(message)
    return matches[0]


def _source_batches(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
) -> tuple[config_service.GenerationConfig, ...]:
    """Resolve the exact source batches owned by one package regime."""
    regime = str(plan["evaluation_regime"])
    sampling_regime = "parameter_ood" if regime == "parameter_ood" else "natural"
    return tuple(campaign.batch(f"{campaign.profile.id}__{material_family}__{sampling_regime}") for material_family in plan["materials"])


def _global_case_id(batch_name: str, case_id: str) -> str:
    """Return one readable package-unique source case identifier."""
    return f"{batch_name}__{case_id}"


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object from validated generated storage."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{label} is unreadable: {path}."
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain one JSON object: {path}."
        raise TypeError(message)
    return value


def _load_candidates(
    campaign: config_service.CampaignConfig,
    plan: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate terminal batches and enumerate deterministic source candidates."""
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    batch_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for batch in _source_batches(campaign, plan):
        manifest, _manifest_path, manifest_sha256 = generated.load_batch_manifest(
            batch.batch_id,
            storage_root=storage,
        )
        if generated.validate_terminal_batch(batch.batch_id, storage_root=storage) != manifest:
            message = f"Terminal batch changed during package admission: {batch.batch_name}."
            raise RuntimeError(message)
        if manifest["batch_identity"] != batch.batch_identity:
            message = f"Terminal batch identity disagrees with campaign plan: {batch.batch_name}."
            raise ValueError(message)
        batch_records.append(
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "batch_identity": batch.batch_identity,
                "manifest_sha256": manifest_sha256,
                "simulation_profile": manifest["simulation_profile"],
                "template": manifest["template"],
                "scientific_config_digest": manifest["scientific_config_digest"],
                "git_commit": manifest["git_commit"],
                "material_config_digest": batch.scientific_values["material_config_digest"],
                "operation_config_digest": batch.scientific_values["operation_config_digest"],
                "airflow_source": manifest["airflow_source"],
                "available_learning_views": manifest["available_learning_views"],
                "export_contract_sha256": manifest["export_contract_sha256"],
                "steady_flow_conditioning": copy.deepcopy(batch.scientific_values["steady_flow_conditioning"]),
            }
        )
        processed_root = common.paths.resolve_generated_batch_dir(
            batch.batch_id,
            stage="processed",
            storage_root=storage,
        )
        for record in manifest["cases"]:
            case_id = str(record["case_id"])
            case_directory = processed_root / case_id
            case_hdf5 = case_directory / "case.h5"
            case_payload = _json_object(case_directory / "case.json", label="Canonical case provenance")
            candidates.append(
                {
                    "batch": batch,
                    "manifest": manifest,
                    "record": record,
                    "case_payload": case_payload,
                    "batch_id": batch.batch_id,
                    "batch_name": batch.batch_name,
                    "case_id": case_id,
                    "package_case_id": _global_case_id(batch.batch_name, case_id),
                    "material_family": record["material_family"],
                    "simulation_profile": manifest["simulation_profile"],
                    "case_input_id": record["case_input_id"],
                    "simulation_case_id": record["simulation_case_id"],
                    "case_hdf5": case_hdf5,
                    "case_hdf5_relative": case_hdf5.resolve().relative_to(storage).as_posix(),
                    "case_hdf5_sha256": record["case_hdf5_sha256"],
                }
            )
    return batch_records, candidates


def resolve_duplicate_case_inputs(
    candidates: Sequence[Mapping[str, Any]],
    *,
    dataset_view: str,
    policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve repeated physical inputs with one explicit profile source policy."""
    if policy not in {"prefer_transient_source", "prefer_steady_source", "reject_duplicates"}:
        message = f"Unsupported duplicate case-input policy: {policy!r}."
        raise ValueError(message)
    normalized = [dict(candidate) for candidate in candidates]
    simulation_ids = [str(candidate["simulation_case_id"]) for candidate in normalized]
    if len(simulation_ids) != len(set(simulation_ids)):
        message = f"Dataset view {dataset_view!r} contains duplicate simulation-case identities."
        raise ValueError(message)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in normalized:
        grouped[str(candidate["case_input_id"])].append(candidate)
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    preferred_profile = {
        "prefer_transient_source": profiles.TRANSIENT_DRYING_PROFILE,
        "prefer_steady_source": profiles.STEADY_FLOW_PROFILE,
    }
    for case_input_id in sorted(grouped):
        group = grouped[case_input_id]
        if len(group) == 1:
            selected.append(group[0])
            continue
        sources = [
            {
                "package_case_id": candidate["package_case_id"],
                "simulation_case_id": candidate["simulation_case_id"],
                "simulation_profile": candidate["simulation_profile"],
            }
            for candidate in sorted(group, key=lambda item: str(item["simulation_case_id"]))
        ]
        if policy == "reject_duplicates":
            message = f"Repeated case_input_id {case_input_id!r} requires an explicit source preference; candidates={sources}."
            raise ValueError(message)
        preferred = preferred_profile[policy]
        matches = [candidate for candidate in group if candidate["simulation_profile"] == preferred]
        if len(matches) != 1:
            message = (
                f"Duplicate policy {policy!r} requires exactly one {preferred!r} source for case_input_id {case_input_id!r}; candidates={sources}."
            )
            raise ValueError(message)
        chosen = matches[0]
        selected.append(chosen)
        decisions.append(
            {
                "case_input_id": case_input_id,
                "policy": policy,
                "candidates": sources,
                "selected_simulation_case_id": chosen["simulation_case_id"],
                "excluded_simulation_case_ids": sorted(str(candidate["simulation_case_id"]) for candidate in group if candidate is not chosen),
            }
        )
    selected.sort(key=lambda candidate: (str(candidate["material_family"]), str(candidate["case_input_id"])))
    return selected, decisions


def _membership_rank(seed: int, material_family: str, case_input_id: str) -> str:
    """Return one stable family-local physical-input membership rank."""
    payload = f"{seed}|{material_family}|{case_input_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _shared_id_membership(
    plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Assign deterministic material-stratified membership to physical case IDs."""
    seed = plan["membership_seed"]
    counts = plan["membership_counts_per_material"]
    if not isinstance(seed, int) or any(not isinstance(counts.get(role), int) for role in _ID_MEMBERSHIP_ROLES):
        message = "ID package membership seed and per-material counts must be resolved."
        raise ValueError(message)
    assignments: dict[str, str] = {}
    for material_family in plan["materials"]:
        family_by_input: dict[str, Mapping[str, Any]] = {}
        for candidate in candidates:
            if candidate["material_family"] != material_family:
                continue
            case_input_id = str(candidate["case_input_id"])
            if case_input_id in family_by_input:
                message = f"ID membership candidate pool duplicates physical input {case_input_id!r}."
                raise ValueError(message)
            family_by_input[case_input_id] = candidate
        family = sorted(
            family_by_input.values(),
            key=lambda candidate: _membership_rank(
                seed,
                str(material_family),
                str(candidate["case_input_id"]),
            ),
        )
        required = sum(int(counts[role]) for role in _ID_MEMBERSHIP_ROLES)
        if len(family) < required:
            message = f"ID package requires {required} {material_family!r} cases but only {len(family)} are terminal."
            raise ValueError(message)
        offset = 0
        for role in _ID_MEMBERSHIP_ROLES:
            role_count = int(counts[role])
            for candidate in family[offset : offset + role_count]:
                assignments[str(candidate["case_input_id"])] = role
            offset += role_count
    if len(assignments) != len(set(assignments)):
        message = "ID train, validation, and id_test membership must be disjoint."
        raise RuntimeError(message)
    return assignments


def audit_steady_flow_conditioning(
    batch_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require complete task-compatible airflow dependencies for every source."""
    task = domain.tasks.registry.get_task("steady_flow")
    solver_inputs = set(_STEADY_SOLVER_INPUTS)
    if not solver_inputs.issubset(task.input_names):
        message = "The steady-flow TaskSpec no longer contains the audited solver inputs."
        raise RuntimeError(message)
    contracts: list[dict[str, Any]] = []
    contract_digests: set[str] = set()
    contract_ids: set[str] = set()
    profiles_found: set[str] = set()
    for record in batch_records:
        raw_contract = record.get("steady_flow_conditioning")
        if not isinstance(raw_contract, dict) or raw_contract.get("exhaustive") is not True:
            message = f"Source profile {record.get('simulation_profile')!r} has no exhaustive steady-flow conditioning audit."
            raise ValueError(message)
        contract = copy.deepcopy(raw_contract)
        dependencies = contract.get("dependencies")
        if not isinstance(dependencies, list):
            message = "Steady-flow conditioning dependencies are malformed."
            raise TypeError(message)
        dependency_by_name = {str(dependency.get("name")): dependency for dependency in dependencies if isinstance(dependency, dict)}
        for name in solver_inputs:
            dependency = dependency_by_name.get(name)
            if dependency is None or dependency.get("owner") != "model_input" or dependency.get("affects_stationary_solution") is not True:
                message = f"Steady-flow solver dependency {name!r} is not represented by the TaskSpec input contract."
                raise ValueError(message)
        for name, dependency in dependency_by_name.items():
            if dependency.get("affects_stationary_solution") is not True:
                continue
            owner = dependency.get("owner")
            if owner == "model_input" and name not in task.input_names:
                message = (
                    f"Hidden steady-flow conditioning detected: case-varying dependency {name!r} "
                    "affects the reference solution but is absent from the TaskSpec inputs."
                )
                raise ValueError(message)
            if owner not in {"model_input", "package_fixed"}:
                message = f"Steady-flow dependency {name!r} has no admissible model-input or package-fixed owner."
                raise ValueError(message)
        additional = contract.get("additional_case_varying_solver_scalars")
        if not isinstance(additional, list) or any(name not in task.input_names for name in additional):
            message = f"Hidden steady-flow conditioning detected in additional_case_varying_solver_scalars={additional!r}."
            raise ValueError(message)
        contract_id = contract.get("stationary_solution_contract_id")
        if not isinstance(contract_id, str) or not contract_id:
            message = "Steady-flow conditioning lacks a stationary-solution compatibility identity."
            raise ValueError(message)
        digest = common.serialization.canonical_json_sha256(contract)
        contract_ids.add(contract_id)
        contract_digests.add(digest)
        profiles_found.add(str(record["simulation_profile"]))
        contracts.append(contract)
    if len(contract_ids) != 1 or len(contract_digests) != 1:
        message = (
            "Steady-flow sources have incompatible stationary-solution conditioning contracts; "
            f"contract_ids={sorted(contract_ids)}, digests={sorted(contract_digests)}."
        )
        raise ValueError(message)
    contract = contracts[0]
    dependencies = contract["dependencies"]
    fixed = [
        {
            "name": dependency["name"],
            "unit": dependency["unit"],
            "value": dependency["fixed_value"],
        }
        for dependency in dependencies
        if dependency["owner"] == "package_fixed"
    ]
    unused = [dependency["name"] for dependency in dependencies if dependency["owner"] == "not_used"]
    return {
        "audit_schema_version": 1,
        "stationary_solution_contract_id": next(iter(contract_ids)),
        "conditioning_contract_digest": next(iter(contract_digests)),
        "source_profiles": sorted(profiles_found),
        "model_inputs": list(task.input_names),
        "package_fixed_physics": fixed,
        "unused_dependencies": unused,
        "T_flow_ref_owner": next(dependency["owner"] for dependency in dependencies if dependency["name"] == "T_flow_ref"),
        "hidden_conditioning": False,
    }


def _parameter_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return registry-owned OOD group, support, transform, and sampled evidence."""
    case_payload = candidate["case_payload"]
    ood = case_payload.get("ood")
    if not isinstance(ood, dict):
        message = f"Case {candidate['package_case_id']!r} has malformed OOD provenance."
        raise TypeError(message)
    selected = ood.get("selected_units")
    if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
        message = f"Case {candidate['package_case_id']!r} has malformed OOD selected_units."
        raise TypeError(message)
    registry = candidate["batch"].scientific_values["material"]["parameter_registry"]
    sampled_values = case_payload.get("sampled_values")
    coupled = case_payload.get("coupled_selections")
    if not isinstance(sampled_values, dict) or not isinstance(coupled, dict):
        message = f"Case {candidate['package_case_id']!r} lacks sampled OOD evidence."
        raise TypeError(message)
    parameters: list[dict[str, Any]] = []
    for name in selected:
        entry = registry.get(name)
        if not isinstance(entry, dict):
            message = f"OOD parameter {name!r} is absent from the source registry."
            raise TypeError(message)
        block = entry.get("block")
        block_provenance = case_payload.get("block_provenance", {}).get(block) if isinstance(block, str) else None
        parameters.append(
            {
                "name": name,
                "group": entry.get("ood_group"),
                "block": block,
                "kind": entry.get("kind"),
                "transform": entry.get("transform"),
                "unit": entry.get("unit"),
                "id_support": {key: copy.deepcopy(entry[key]) for key in ("lower", "upper", "sets") if key in entry},
                "ood_support": copy.deepcopy(entry.get("ood", entry.get("ood_values", entry.get("ood_sets")))),
                "sampled_value": copy.deepcopy(sampled_values.get(name)),
                "coupled_selection": coupled.get(name),
                "transformed_coordinate_evidence": copy.deepcopy(block_provenance),
            }
        )
    return {
        "group": ood.get("group"),
        "selected_units": list(selected),
        "units_per_case": ood.get("units_per_case"),
        "parameters": parameters,
    }


def _ood_eligibility(
    candidate: dict[str, Any],
    *,
    view: views.DatasetViewSpec,
) -> tuple[bool, tuple[str, ...], dict[str, Any], str | None]:
    """Return view-specific eligibility from registry-owned dependency blocks."""
    evidence = _parameter_evidence(candidate)
    group = evidence["group"]
    if group not in views.OOD_GROUPS:
        message = f"Parameter-OOD case {candidate['package_case_id']!r} has invalid group {group!r}."
        raise ValueError(message)
    selected = evidence["parameters"]
    if view.id == "transient_drying":
        relevant = tuple(str(parameter["name"]) for parameter in selected)
    else:
        relevant = tuple(str(parameter["name"]) for parameter in selected if parameter["block"] in view.parameter_ood_blocks)
    if not relevant:
        reason = f"No selected parameter belongs to the {view.id!r} dependency blocks {list(view.parameter_ood_blocks)}."
        return False, (), evidence, reason
    return True, relevant, evidence, None


def _excluded_case(candidate: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    """Return compact excluded-source evidence with one actionable reason."""
    ood = candidate["case_payload"].get("ood", {})
    return {
        "package_case_id": candidate["package_case_id"],
        "case_input_id": candidate["case_input_id"],
        "simulation_case_id": candidate["simulation_case_id"],
        "simulation_profile": candidate["simulation_profile"],
        "material_family": candidate["material_family"],
        "ood_group": ood.get("group") if isinstance(ood, dict) else None,
        "ood_parameters": ood.get("selected_units", []) if isinstance(ood, dict) else [],
        "reason": reason,
    }


def _prepare_campaign_packages(
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path | str | None,
    selected_plans: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[_PreparedPackage, ...]:
    """Admit selected packages together so membership and leakage are global."""
    plans = [copy.deepcopy(dict(plan)) for plan in (campaign.dataset_packages if selected_plans is None else selected_plans)]
    source_by_regime: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for plan in plans:
        regime = str(plan["evaluation_regime"])
        if regime not in source_by_regime:
            source_by_regime[regime] = _load_candidates(campaign, plan, storage_root=storage_root)
    id_plans = [plan for plan in plans if plan["evaluation_regime"] == "id"]
    if not id_plans:
        message = "Campaign must declare at least one ID package."
        raise ValueError(message)
    _id_records, id_candidates = source_by_regime["id"]
    physical_id_candidates, _id_decisions = resolve_duplicate_case_inputs(
        id_candidates,
        dataset_view="shared_id_membership",
        policy=campaign.duplicate_case_input_policy,
    )
    shared_membership = _shared_id_membership(id_plans[0], physical_id_candidates)
    prepared: list[_PreparedPackage] = []
    for plan in plans:
        dataset_view = str(plan["dataset_view"])
        regime = str(plan["evaluation_regime"])
        view = views.get_view(dataset_view)
        batch_records, raw_candidates = source_by_regime[regime]
        candidates, decisions = resolve_duplicate_case_inputs(
            raw_candidates,
            dataset_view=dataset_view,
            policy=campaign.duplicate_case_input_policy,
        )
        for candidate in candidates:
            views.validate_view_for_profile(dataset_view, str(candidate["simulation_profile"]))
        steady_conditioning = audit_steady_flow_conditioning(batch_records) if dataset_view == "steady_flow" else None
        included: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        membership: dict[str, list[str]] = {role: [] for role in _ID_MEMBERSHIP_ROLES} if regime == "id" else {regime: []}
        for candidate in candidates:
            if regime == "id":
                assigned = shared_membership.get(str(candidate["case_input_id"]))
                if assigned is None:
                    excluded.append(
                        _excluded_case(
                            candidate,
                            reason="Physical case was not selected by the declared material-stratified ID membership counts.",
                        )
                    )
                    continue
                candidate["dataset_membership"] = assigned
                candidate["task_relevant_ood_parameters"] = []
                candidate["ood_evidence"] = {}
            else:
                candidate["dataset_membership"] = regime
                candidate["task_relevant_ood_parameters"] = []
                candidate["ood_evidence"] = {}
                if regime == "parameter_ood":
                    eligible, relevant, evidence, reason = _ood_eligibility(candidate, view=view)
                    if not eligible:
                        excluded.append(_excluded_case(candidate, reason=str(reason)))
                        continue
                    candidate["task_relevant_ood_parameters"] = list(relevant)
                    candidate["ood_evidence"] = evidence
            included.append(candidate)
            membership[str(candidate["dataset_membership"])].append(str(candidate["package_case_id"]))
        if not included:
            message = f"Declared package {dataset_view!r}/{regime!r} has no scientifically eligible source cases."
            raise ValueError(message)
        prepared.append(
            _PreparedPackage(
                plan=plan,
                batch_records=copy.deepcopy(batch_records),
                candidates=included,
                excluded=excluded,
                membership=membership,
                source_decisions=decisions,
                steady_conditioning=steady_conditioning,
            )
        )
    _validate_no_id_ood_overlap(prepared)
    return tuple(prepared)


def _validate_no_id_ood_overlap(prepared: Sequence[_PreparedPackage]) -> None:
    """Reject simulation or physical-input reuse from ID train into any OOD package."""
    id_train_simulation_ids = {
        str(candidate["simulation_case_id"])
        for package in prepared
        if package.plan["evaluation_regime"] == "id"
        for candidate in package.candidates
        if candidate["dataset_membership"] == "train"
    }
    id_train_input_ids = {
        str(candidate["case_input_id"])
        for package in prepared
        if package.plan["evaluation_regime"] == "id"
        for candidate in package.candidates
        if candidate["dataset_membership"] == "train"
    }
    for package in prepared:
        if package.plan["evaluation_regime"] == "id":
            continue
        simulation_overlap = id_train_simulation_ids.intersection(str(candidate["simulation_case_id"]) for candidate in package.candidates)
        input_overlap = id_train_input_ids.intersection(str(candidate["case_input_id"]) for candidate in package.candidates)
        if simulation_overlap or input_overlap:
            message = (
                "ID training and OOD package source overlap detected: "
                f"simulation_case_ids={sorted(simulation_overlap)}, case_input_ids={sorted(input_overlap)}."
            )
            raise ValueError(message)


def _channel_contract(dataset_view: str) -> dict[str, Any]:
    """Return one authoritative physical-unit tensor contract for the manifest."""
    if dataset_view == "transient_drying":
        return transient.transient_contract_payload()
    task = domain.tasks.registry.get_task("steady_flow")
    return {
        "input": [field.as_dict() for field in task.inputs],
        "target": [field.as_dict() for field in task.outputs],
        "tensor_layout": list(task.tensor_layout),
    }


def _case_indexes(
    prepared: _PreparedPackage,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Build compact included, OOD-group, and OOD-parameter case indexes."""
    included = [str(candidate["package_case_id"]) for candidate in prepared.candidates]
    groups: dict[str, list[str]] = defaultdict(list)
    parameters: dict[str, list[str]] = defaultdict(list)
    relevant: set[str] = set()
    for candidate in prepared.candidates:
        case_id = str(candidate["package_case_id"])
        evidence = candidate.get("ood_evidence")
        if isinstance(evidence, dict):
            group = evidence.get("group")
            if isinstance(group, str):
                groups[group].append(case_id)
        for parameter in candidate.get("task_relevant_ood_parameters", []):
            name = str(parameter)
            parameters[name].append(case_id)
            relevant.add(name)
    return (
        included,
        dict(sorted(groups.items())),
        dict(sorted(parameters.items())),
        sorted(relevant),
    )


def _package_provenance(
    campaign: config_service.CampaignConfig,
    prepared: _PreparedPackage,
) -> dict[str, Any]:
    """Return complete portable provenance before dataset identity binding."""
    plan = prepared.plan
    view = views.get_view(str(plan["dataset_view"]))
    included, group_indexes, parameter_indexes, relevant_parameters = _case_indexes(prepared)
    profiles_found = sorted({str(record["simulation_profile"]) for record in prepared.batch_records})
    template_digests = sorted({str(record["template"]["sha256"]) for record in prepared.batch_records})
    airflow = sorted({str(record["airflow_source"]) for record in prepared.batch_records})
    operation_digests = sorted({str(record["operation_config_digest"]) for record in prepared.batch_records})
    git_commits = sorted({str(record["git_commit"]) for record in prepared.batch_records})
    case_membership = {str(candidate["package_case_id"]): str(candidate["dataset_membership"]) for candidate in prepared.candidates}
    material_counts: dict[str, int] = defaultdict(int)
    profile_counts: dict[str, int] = defaultdict(int)
    for candidate in prepared.candidates:
        material_counts[str(candidate["material_family"])] += 1
        profile_counts[str(candidate["simulation_profile"])] += 1
    return {
        "schema_kind": DATASET_PACKAGE_SCHEMA_KIND,
        "schema_version": DATASET_PACKAGE_SCHEMA_VERSION,
        "dataset_name": plan["dataset_name"],
        "dataset_view": view.id,
        "registered_task_id": view.registered_task_id,
        "evaluation_regime": plan["evaluation_regime"],
        "materials": list(plan["materials"]),
        "channel_contract": _channel_contract(view.id),
        "channel_contract_digest": view.contract_digest,
        "source_simulation_profiles": profiles_found,
        "source_batches": prepared.batch_records,
        "source_batch_ids": [record["batch_id"] for record in prepared.batch_records],
        "source_template_digests": template_digests,
        "source_git_commits": git_commits,
        "source_case_identities": [
            {
                "package_case_id": candidate["package_case_id"],
                "batch_id": candidate["batch_id"],
                "source_case_id": candidate["case_id"],
                "source_relative_path": candidate["case_hdf5_relative"],
                "case_input_id": candidate["case_input_id"],
                "simulation_case_id": candidate["simulation_case_id"],
                "case_hdf5_sha256": candidate["case_hdf5_sha256"],
                "material_family": candidate["material_family"],
                "simulation_profile": candidate["simulation_profile"],
                "membership": candidate["dataset_membership"],
                "ood_group": candidate.get("ood_evidence", {}).get("group"),
                "ood_parameters": candidate.get("ood_evidence", {}).get("selected_units", []),
                "task_relevant_ood_parameters": candidate.get("task_relevant_ood_parameters", []),
                "ood_evidence": copy.deepcopy(candidate.get("ood_evidence", {})),
            }
            for candidate in prepared.candidates
        ],
        "included_source_cases": included,
        "excluded_source_cases": prepared.excluded,
        "source_selection_decisions": prepared.source_decisions,
        "matched_case_input_ids": sorted(
            str(candidate["case_input_id"])
            for candidate in prepared.candidates
            if candidate["simulation_profile"] == profiles.TRANSIENT_DRYING_PROFILE
        ),
        "airflow_provenance": airflow if view.id == "steady_flow" else [],
        "steady_flow_conditioning": prepared.steady_conditioning,
        "material_file_identities": {record["batch_name"].split("__")[1]: record["material_config_digest"] for record in prepared.batch_records},
        "operation_config_digests": operation_digests,
        "campaign_name": campaign.campaign_name,
        "campaign_digest": campaign.campaign_digest,
        "duplicate_case_input_policy": campaign.duplicate_case_input_policy,
        "case_membership": case_membership,
        "split_membership": copy.deepcopy(prepared.membership),
        "membership_counts": {name: len(case_ids) for name, case_ids in prepared.membership.items()},
        "available_ood_groups": sorted(group_indexes),
        "ood_group_indexes": group_indexes,
        "ood_parameter_indexes": parameter_indexes,
        "task_relevant_ood_parameters": relevant_parameters,
        "material_counts": dict(sorted(material_counts.items())),
        "source_profile_counts": dict(sorted(profile_counts.items())),
        "candidate_source_case_count": len(prepared.candidates) + len(prepared.excluded),
        "builder_identity": "src.datasets.dataset_packages.build_campaign_packages",
        "schema_identity": {
            "package": DATASET_PACKAGE_SCHEMA_VERSION,
            "case_hdf5": storage_schema_version(),
            "transient_index": transient.TRANSIENT_INDEX_SCHEMA_VERSION,
        },
    }


def storage_schema_version() -> int:
    """Return the canonical source-case HDF5 schema identity."""
    from src.generation import generation_storage  # noqa: PLC0415

    return generation_storage.HDF5_SCHEMA_VERSION


def _dataset_identity_from_provenance(provenance: Mapping[str, Any]) -> tuple[str, str]:
    """Return immutable dataset ID without operational relative source locators."""
    identity_provenance = copy.deepcopy(dict(provenance))
    source_cases = identity_provenance.get("source_case_identities")
    if isinstance(source_cases, list):
        for source_case in source_cases:
            if isinstance(source_case, dict):
                source_case.pop("source_relative_path", None)
    digest = common.serialization.canonical_json_sha256(identity_provenance)
    name = str(provenance["dataset_name"])
    return f"{name}__{digest[:16]}", digest


def _aggregate_generated_identity(
    campaign: config_service.CampaignConfig,
    prepared: _PreparedPackage,
) -> dict[str, Any]:
    """Build the steady payload's compatibility identity from exact source provenance."""
    profiles_found = {str(record["simulation_profile"]) for record in prepared.batch_records}
    conditioning_profiles = set(prepared.steady_conditioning.get("source_profiles", [])) if isinstance(prepared.steady_conditioning, dict) else set()
    if not profiles_found or conditioning_profiles != profiles_found:
        message = "Steady aggregate sources are not bound to one complete compatible conditioning audit."
        raise ValueError(message)
    representatives = [record for record in prepared.batch_records if record["simulation_profile"] == campaign.profile.id]
    representative_pool = representatives or prepared.batch_records
    first = min(
        representative_pool,
        key=lambda record: (str(record["simulation_profile"]), str(record["batch_id"])),
    )
    cases = [
        {
            "case_id": candidate["package_case_id"],
            "material_family": candidate["material_family"],
            "case_input_id": candidate["record"]["case_input_id"],
            "simulation_case_id": candidate["record"]["simulation_case_id"],
            "case_hdf5_sha256": candidate["record"]["case_hdf5_sha256"],
            "success_sha256": candidate["record"]["success_sha256"],
            "provenance_sha256": candidate["record"]["provenance_sha256"],
        }
        for candidate in prepared.candidates
    ]
    return identity.build_generated_package_identity(
        dataset_name=str(prepared.plan["dataset_name"]),
        simulation_profile=str(first["simulation_profile"]),
        campaign_digest=campaign.campaign_digest,
        template=first["template"],
        export_contract_sha256=str(first["export_contract_sha256"]),
        available_learning_views=first["available_learning_views"],
        airflow_source=str(first["airflow_source"]),
        cases=cases,
    )


def _build_steady_payload(
    campaign: config_service.CampaignConfig,
    prepared: _PreparedPackage,
    *,
    dataset_id: str,
    provenance: Mapping[str, Any],
    destination: Path,
    storage_root: Path | str | None,
) -> None:
    """Build one eager steady tensor payload from admitted canonical HDF5 views."""
    task = domain.tasks.registry.get_task("steady_flow")
    source_identities: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for candidate in prepared.candidates:
        _shape, case_inputs, case_outputs, metadata, source, _fingerprint = generated.interpret_generated_case(
            candidate["batch_id"],
            candidate["case_id"],
            task=task,
            manifest=candidate["manifest"],
            record=candidate["record"],
            storage_root=storage_root,
        )
        metadata.update(
            {
                "dataset_id": dataset_id,
                "evaluation_regime": prepared.plan["evaluation_regime"],
                "dataset_membership": candidate["dataset_membership"],
                "source_batch_id": candidate["batch_id"],
                "source_case_id": candidate["case_id"],
                "source_simulation_profile": candidate["simulation_profile"],
                "source_hdf5_sha256": candidate["case_hdf5_sha256"],
                "task_relevant_ood_parameters": candidate.get("task_relevant_ood_parameters", []),
            }
        )
        source.update(
            {
                "source_batch_id": candidate["batch_id"],
                "source_case_id": candidate["case_id"],
                "case_id": candidate["package_case_id"],
            }
        )
        fingerprint = identity.compute_case_fingerprint(
            task=task,
            case_id=candidate["package_case_id"],
            source_identity=source,
            source_metadata=metadata,
            inputs=case_inputs,
            outputs=case_outputs,
        )
        source_identities.append(source)
        source_metadata.append(metadata)
        fingerprints.append(fingerprint)
        inputs.append(case_inputs)
        outputs.append(case_outputs)
    payload = identity.build_training_dataset_payload(
        task=task,
        dataset_id=dataset_id,
        sample_ids=[candidate["package_case_id"] for candidate in prepared.candidates],
        generated_batch_identity=_aggregate_generated_identity(campaign, prepared),
        source_identities=source_identities,
        source_metadata=source_metadata,
        source_provenance=provenance,
        case_fingerprints=fingerprints,
        inputs=torch.stack(inputs),
        outputs=torch.stack(outputs),
    )
    common.serialization.atomic_torch_save(payload, destination)


def _transient_sources(prepared: _PreparedPackage) -> tuple[transient.TransientSourceCase, ...]:
    """Convert admitted candidates into typed transient index sources."""
    return tuple(
        transient.TransientSourceCase(
            path=Path(candidate["case_hdf5"]),
            package_case_id=str(candidate["package_case_id"]),
            source_batch_id=str(candidate["batch_id"]),
            membership=str(candidate["dataset_membership"]),
            evaluation_regime=str(prepared.plan["evaluation_regime"]),
            expected_sha256=str(candidate["case_hdf5_sha256"]),
            expected_case_input_id=str(candidate["case_input_id"]),
            expected_simulation_case_id=str(candidate["simulation_case_id"]),
            material_family=str(candidate["material_family"]),
            ood_group=(str(candidate["ood_evidence"]["group"]) if candidate.get("ood_evidence", {}).get("group") is not None else None),
            ood_parameters=tuple(str(name) for name in candidate.get("task_relevant_ood_parameters", [])),
            ood_evidence=copy.deepcopy(candidate.get("ood_evidence", {})),
        )
        for candidate in prepared.candidates
    )


def _publish(
    *,
    dataset_id: str,
    payload_filename: str,
    manifest: Mapping[str, Any],
    build_payload: Callable[[Path], None],
    storage_root: Path | str | None,
) -> tuple[Path, Path, bool]:
    """Atomically publish or integrity-reuse one locked immutable package."""
    payload_root = common.paths.get_dataset_payload_root(storage_root=storage_root)
    metadata_root = common.paths.get_dataset_metadata_root(storage_root=storage_root)
    destination_dir = payload_root / dataset_id
    metadata_dir = metadata_root / dataset_id
    destination = destination_dir / payload_filename
    manifest_path = metadata_dir / "dataset_manifest.json"
    lock_path = common.paths.resolve_dataset_build_lock_path(dataset_id, storage_root=storage_root)
    requested_manifest = dict(manifest)
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        if destination.is_file() and manifest_path.is_file():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                message = f"Existing package manifest is unreadable: {manifest_path}."
                raise ValueError(message) from error
            if not isinstance(existing_manifest, dict):
                message = f"Existing package manifest is malformed: {manifest_path}."
                raise TypeError(message)
            payload_sha256 = existing_manifest.pop("payload_sha256", None)
            if existing_manifest != requested_manifest or payload_sha256 != common.serialization.file_sha256(destination):
                message = f"Existing package content conflicts with {dataset_id!r}."
                raise FileExistsError(message)
            return destination, manifest_path, True
        if destination_dir.exists() or metadata_dir.exists():
            message = f"Partial or conflicting dataset package already exists: {dataset_id!r}."
            raise FileExistsError(message)
        state_root = common.paths.get_dataset_state_root(storage_root=storage_root)
        state_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", suffix=".tmp", dir=state_root))
        staged_payload_dir = staging / "payload"
        staged_metadata_dir = staging / "metadata"
        staged_payload_dir.mkdir()
        staged_metadata_dir.mkdir()
        try:
            staged_payload = staged_payload_dir / payload_filename
            build_payload(staged_payload)
            complete_manifest = {
                **requested_manifest,
                "payload_sha256": common.serialization.file_sha256(staged_payload),
            }
            common.serialization.atomic_write_json(
                staged_metadata_dir / "dataset_manifest.json",
                complete_manifest,
            )
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
            metadata_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_payload_dir.replace(destination_dir)
            staged_metadata_dir.replace(metadata_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return destination, manifest_path, False


def _publish_prepared(
    campaign: config_service.CampaignConfig,
    prepared: _PreparedPackage,
    *,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Build or exactly reuse one fully prepared package payload."""
    provenance = _package_provenance(campaign, prepared)
    dataset_id, dataset_digest = _dataset_identity_from_provenance(provenance)
    dataset_view = str(prepared.plan["dataset_view"])
    if dataset_view == "steady_flow":
        payload_filename = f"{dataset_id}.pt"
        sample_count = len(prepared.candidates)
        transition_count = 0

        def build_payload(path: Path) -> None:
            _build_steady_payload(
                campaign,
                prepared,
                dataset_id=dataset_id,
                provenance=provenance,
                destination=path,
                storage_root=storage_root,
            )

    elif dataset_view == "transient_drying":
        payload_filename = f"{dataset_id}.json"
        source_root = common.paths.get_storage_root(storage_root=storage_root)
        source_cases = _transient_sources(prepared)
        preview = transient.build_transient_index(
            source_cases,
            None,
            dataset_name=str(prepared.plan["dataset_name"]),
            dataset_id=dataset_id,
            evaluation_regime=str(prepared.plan["evaluation_regime"]),
            contract_digest=views.get_view("transient_drying").contract_digest,
            source_root=source_root,
        )
        sample_count = int(preview["sample_count"])
        transition_count = int(preview["transition_count"])

        def build_payload(path: Path) -> None:
            built = transient.build_transient_index(
                source_cases,
                path,
                dataset_name=str(prepared.plan["dataset_name"]),
                dataset_id=dataset_id,
                evaluation_regime=str(prepared.plan["evaluation_regime"]),
                contract_digest=views.get_view("transient_drying").contract_digest,
                source_root=source_root,
            )
            if built["index_digest"] != preview["index_digest"]:
                message = "Transient transition identity changed between preview and publication."
                raise RuntimeError(message)

    else:
        message = f"Unsupported dataset view: {dataset_view!r}."
        raise ValueError(message)
    manifest = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": payload_filename,
        "sample_count": sample_count,
        "source_case_count": len(prepared.candidates),
        "transition_count": transition_count,
    }
    destination, manifest_path, reused = _publish(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest=manifest,
        build_payload=build_payload,
        storage_root=storage_root,
    )
    return {
        "dataset_name": prepared.plan["dataset_name"],
        "dataset_id": dataset_id,
        "dataset_view": dataset_view,
        "evaluation_regime": prepared.plan["evaluation_regime"],
        "payload_path": destination,
        "manifest_path": manifest_path,
        "sample_count": sample_count,
        "source_case_count": len(prepared.candidates),
        "transition_count": transition_count,
        "status": "reused" if reused else "complete",
    }


def build_dataset_package(
    campaign: config_service.CampaignConfig,
    dataset_view: str,
    evaluation_regime: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build one declared package with its required ID leakage companion."""
    requested_plan = _package_plan(campaign, dataset_view, evaluation_regime)
    selected_plans = [requested_plan]
    if evaluation_regime != "id":
        selected_plans.insert(0, _package_plan(campaign, dataset_view, "id"))
    prepared = _prepare_campaign_packages(
        campaign,
        storage_root=storage_root,
        selected_plans=selected_plans,
    )
    matches = [
        package for package in prepared if package.plan["dataset_view"] == dataset_view and package.plan["evaluation_regime"] == evaluation_regime
    ]
    if len(matches) != 1:
        message = f"Prepared package selection is ambiguous for {dataset_view!r}/{evaluation_regime!r}."
        raise RuntimeError(message)
    return _publish_prepared(campaign, matches[0], storage_root=storage_root)


def build_campaign_packages(
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build every declared package after one shared membership/leakage preflight."""
    prepared = _prepare_campaign_packages(campaign, storage_root=storage_root)
    return tuple(_publish_prepared(campaign, package, storage_root=storage_root) for package in prepared)


def _validate_manifest_content(
    manifest: Any,
    *,
    dataset_id: str,
    payload_path: Path,
) -> dict[str, Any]:
    """Validate one current manifest against its portable identity and payload."""
    if not isinstance(manifest, dict) or set(manifest) != _PACKAGE_MANIFEST_KEYS:
        message = f"Dataset package manifest keys do not match the current schema for {dataset_id!r}."
        raise ValueError(message)
    provenance = {key: manifest[key] for key in _PACKAGE_PROVENANCE_KEYS}
    expected_id, expected_digest = _dataset_identity_from_provenance(provenance)
    if (
        manifest["schema_kind"] != DATASET_PACKAGE_SCHEMA_KIND
        or manifest["schema_version"] != DATASET_PACKAGE_SCHEMA_VERSION
        or manifest["dataset_id"] != dataset_id
        or manifest["dataset_id"] != expected_id
        or manifest["dataset_digest"] != expected_digest
        or manifest["payload_filename"] != payload_path.name
        or manifest["payload_sha256"] != common.serialization.file_sha256(payload_path)
        or manifest["sample_count"] < 1
        or manifest["source_case_count"] < 1
    ):
        message = f"Dataset package manifest does not bind the exact payload identity for {dataset_id!r}."
        raise ValueError(message)
    return dict(manifest)


def load_package_manifest(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate one package manifest and its exact payload hash."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    metadata_path = common.paths.get_dataset_metadata_root(storage_root=storage_root) / logical_id / "dataset_manifest.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        message = f"Dataset package manifest is missing or unsafe: {metadata_path}."
        raise FileNotFoundError(message)
    try:
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Dataset package manifest is unreadable: {metadata_path}."
        raise ValueError(message) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("payload_filename"), str):
        message = f"Dataset package manifest is malformed: {metadata_path}."
        raise TypeError(message)
    payload_path = common.paths.get_dataset_payload_root(storage_root=storage_root) / logical_id / manifest["payload_filename"]
    if not payload_path.is_file() or payload_path.is_symlink():
        message = f"Dataset package payload is missing or unsafe: {payload_path}."
        raise FileNotFoundError(message)
    return _validate_manifest_content(manifest, dataset_id=logical_id, payload_path=payload_path)


def load_dataset_package_manifest(
    dataset_id: str,
    *,
    dataset_identity: identity.DatasetIdentity,
    dataset_path: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Bind a steady package manifest to its validated tensor payload identity."""
    logical_id = common.paths.validate_logical_name(dataset_id, label="dataset_id")
    payload_path = Path(dataset_path)
    manifest_path = Path(metadata_root) / logical_id / "dataset_manifest.json"
    if not payload_path.is_file() or payload_path.is_symlink():
        message = f"Dataset package payload is missing or unsafe: {payload_path}."
        raise FileNotFoundError(message)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        message = f"Dataset package manifest is missing or unsafe: {manifest_path}."
        raise FileNotFoundError(message)
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Dataset package manifest is unreadable: {manifest_path}."
        raise ValueError(message) from error
    manifest = _validate_manifest_content(raw_manifest, dataset_id=logical_id, payload_path=payload_path)
    provenance = {key: manifest[key] for key in _PACKAGE_PROVENANCE_KEYS}
    if (
        manifest["dataset_view"] != "steady_flow"
        or manifest["registered_task_id"] != dataset_identity.task
        or manifest["sample_count"] != dataset_identity.sample_count
        or dataset_identity.dataset_id != logical_id
        or dataset_identity.source_provenance != provenance
    ):
        message = f"Steady package manifest does not bind its validated tensor identity: {manifest_path}."
        raise ValueError(message)
    return manifest


def _runtime_request(
    manifest: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
    membership: str | None = None,
    ood_group: str | None = None,
) -> Any:
    """Build one typed factory request from a validated package manifest."""
    from . import dataset_factory as factory  # noqa: PLC0415

    return factory.DatasetRequest(
        dataset_id=str(manifest["dataset_id"]),
        dataset_view=cast("views.DatasetViewId", manifest["dataset_view"]),
        evaluation_regime=cast("views.PackageRegime", manifest["evaluation_regime"]),
        membership=cast("views.IdMembership | None", membership),
        ood_group=cast("views.OodGroup | None", ood_group),
        storage_root=storage_root,
    )


def _tensor_description(tensor: torch.Tensor, channels: Any) -> dict[str, Any]:
    """Return one inspection tensor shape, dtype, and channel declaration."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "channels": copy.deepcopy(channels),
    }


def inspect_dataset_package(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect one package through its validated manifest and runtime object."""
    from . import dataset_factory as factory  # noqa: PLC0415

    manifest = load_package_manifest(dataset_id, storage_root=storage_root)
    request = _runtime_request(manifest, storage_root=storage_root)
    runtime = factory.create_dataset(request, hdf5_cache_size=1)
    sample = cast("Mapping[str, Any]", runtime[0])
    if manifest["dataset_view"] == "steady_flow":
        tensor_report = {
            "input": _tensor_description(sample["x"], manifest["channel_contract"]["input"]),
            "target": _tensor_description(sample["y"], manifest["channel_contract"]["target"]),
        }
        metadata = cast("Mapping[str, Any]", sample["meta"])
        sample_identity = {
            key: metadata.get(key)
            for key in (
                "dataset_id",
                "simulation_case_id",
                "case_input_id",
                "source_batch_id",
                "source_simulation_profile",
                "material_family",
                "evaluation_regime",
                "dataset_membership",
                "source_hdf5_sha256",
            )
        }
    else:
        tensor_report = {
            name: _tensor_description(sample[name], manifest["channel_contract"][name])
            for name in ("state", "static", "boundary", "scalars", "target")
        }
        tensor_report["dt"] = _tensor_description(sample["dt"], manifest["channel_contract"]["dt"])
        sample_identity = dict(sample["metadata"])
        if isinstance(runtime, transient.TransientPhysicalDataset):
            runtime.close()
    regime = str(manifest["evaluation_regime"])
    if regime == "id":
        selectors = [f"id/{membership}" for membership in views.ID_MEMBERSHIPS]
    elif regime == "parameter_ood":
        selectors = ["parameter_ood/all", *[f"parameter_ood/{group}" for group in manifest["available_ood_groups"]]]
    else:
        selectors = [regime]
    return {
        "dataset_name": manifest["dataset_name"],
        "dataset_id": manifest["dataset_id"],
        "dataset_view": manifest["dataset_view"],
        "registered_task_id": manifest["registered_task_id"],
        "evaluation_regime": regime,
        "available_selectors": selectors,
        "available_ood_groups": manifest["available_ood_groups"],
        "membership_counts": manifest["membership_counts"],
        "source_case_count": manifest["source_case_count"],
        "transition_count": manifest["transition_count"],
        "sample_count": manifest["sample_count"],
        "material_counts": manifest["material_counts"],
        "source_profile_counts": manifest["source_profile_counts"],
        "tensors": tensor_report,
        "sample_identity": sample_identity,
    }


def smoke_dataset_package(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
    membership: str | None = None,
    ood_group: str | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    hdf5_cache_size: int = 1,
) -> dict[str, Any]:
    """Load one batch through the unified factory with requested worker settings."""
    from . import dataset_factory as factory  # noqa: PLC0415

    manifest = load_package_manifest(dataset_id, storage_root=storage_root)
    request = _runtime_request(
        manifest,
        storage_root=storage_root,
        membership=membership,
        ood_group=ood_group,
    )
    settings = factory.LoaderSettings(
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        hdf5_cache_size=hdf5_cache_size,
    )
    loader = factory.create_data_loader(request, settings)
    batch = next(iter(loader))
    tensor_keys = ("x", "y") if manifest["dataset_view"] == "steady_flow" else ("state", "static", "boundary", "scalars", "target", "dt")
    shapes = {key: list(value.shape) for key in tensor_keys if isinstance((value := batch.get(key)), torch.Tensor)}
    return {
        "dataset_id": dataset_id,
        "dataset_view": manifest["dataset_view"],
        "evaluation_regime": manifest["evaluation_regime"],
        "membership": membership,
        "ood_group": ood_group,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "batch_shapes": shapes,
        "status": "loaded",
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the dataset package build, inspection, and smoke CLI."""
    parser = argparse.ArgumentParser(description="Build, inspect, or smoke immutable dataset packages")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build every package declared by one campaign")
    build.add_argument("campaign_config", type=Path)
    build.add_argument("--storage-root", type=Path)

    inspect = commands.add_parser("inspect", help="inspect one immutable package")
    inspect.add_argument("dataset_id")
    inspect.add_argument("--storage-root", type=Path)

    smoke = commands.add_parser("smoke", help="load one package batch through the unified factory")
    smoke.add_argument("dataset_id")
    smoke.add_argument("--storage-root", type=Path)
    smoke.add_argument("--membership", choices=views.ID_MEMBERSHIPS)
    smoke.add_argument("--ood-group", choices=views.OOD_GROUPS)
    smoke.add_argument("--num-workers", type=int, default=0)
    smoke.add_argument("--persistent-workers", action="store_true")
    smoke.add_argument("--prefetch-factor", type=int)
    smoke.add_argument("--hdf5-cache-size", type=int, default=1)
    return parser


def main() -> int:
    """Execute one explicit package build, inspection, or smoke command."""
    arguments = _build_parser().parse_args()
    if arguments.command == "build":
        campaign = config_service.load_campaign_config(arguments.campaign_config)
        result: dict[str, Any] = {
            "campaign_id": campaign.campaign_id,
            "packages": build_campaign_packages(campaign, storage_root=arguments.storage_root),
        }
    elif arguments.command == "inspect":
        result = inspect_dataset_package(
            arguments.dataset_id,
            storage_root=arguments.storage_root,
        )
    elif arguments.command == "smoke":
        result = smoke_dataset_package(
            arguments.dataset_id,
            storage_root=arguments.storage_root,
            membership=arguments.membership,
            ood_group=arguments.ood_group,
            num_workers=arguments.num_workers,
            persistent_workers=arguments.persistent_workers,
            prefetch_factor=arguments.prefetch_factor,
            hdf5_cache_size=arguments.hdf5_cache_size,
        )
    else:
        message = f"Unsupported dataset package command: {arguments.command!r}."
        raise RuntimeError(message)
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
