"""
dataset_packages_planning.py

Plan Dataset package sources, eligibility, and case membership.
Responsibilities:
  - Resolve declared campaign package sources through terminal evidence
  - Select duplicate physical inputs and deterministic ID membership
  - Audit steady conditioning and view-specific parameter-OOD eligibility
  - Reject leakage across ID training and OOD source selections
Design principles:
  - Generation alone admits terminal publication and artifact integrity
  - Selection is deterministic and independent of payload publication
  - View relevance comes from immutable Dataset and Generation contracts
This module does NOT:
  - Build payloads, compute package identities, publish artifacts, or expose a CLI
  - Load runtime Dataset objects or fit preprocessing
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from src import common, domain, generation
from src.datasets.contracts import dataset_contracts_views as views

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_ID_MEMBERSHIP_ROLES: Final = views.ID_MEMBERSHIPS
_STEADY_PROFILE_CONTRACT: Final = generation.contracts.get_profile_contract("steady_flow")
_TRANSIENT_PROFILE_CONTRACT: Final = generation.contracts.get_profile_contract("transient_drying")


@dataclass(slots=True)
class PreparedPackage:
    """Hold one completely admitted package before payload publication."""

    plan: dict[str, Any]
    batch_records: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    membership: dict[str, list[str]]
    source_decisions: list[dict[str, Any]]
    steady_conditioning: dict[str, Any] | None


def package_plan(
    campaign: generation.cases.config.CampaignConfig,
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
    campaign: generation.cases.config.CampaignConfig,
    plan: Mapping[str, Any],
) -> tuple[generation.cases.config.GenerationConfig, ...]:
    """Resolve the exact source batches owned by one package regime."""
    regime = str(plan["evaluation_regime"])
    sampling_regime = "parameter_ood" if regime == "parameter_ood" else "natural"
    return tuple(
        campaign.require_batch(
            material_family=str(material_family),
            sampling_regime=sampling_regime,
        )
        for material_family in plan["materials"]
    )


def _global_case_id(batch_name: str, case_id: str) -> str:
    """Return one readable package-unique source case identifier."""
    return f"{batch_name}__{case_id}"


def _load_candidates(
    campaign: generation.cases.config.CampaignConfig,
    plan: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit terminal batches and enumerate deterministic source candidates."""
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    batch_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for batch in _source_batches(campaign, plan):
        terminal = generation.runtime.admit_terminal_batch(
            batch.batch_storage_name,
            storage_root=storage,
            validation_depth="routine",
        )
        manifest = terminal.manifest_payload()
        if terminal.batch_identity != batch.batch_identity:
            message = f"Terminal batch identity disagrees with campaign plan: {batch.batch_name}."
            raise ValueError(message)
        batch_records.append(
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "batch_identity": batch.batch_identity,
                "manifest_sha256": terminal.manifest_sha256,
                "simulation_profile": terminal.simulation_profile,
                "template": manifest["template"],
                "scientific_config_digest": terminal.scientific_config_digest,
                "git_commit": terminal.git_commit,
                "material_config_digest": batch.scientific_values["material_config_digest"],
                "material_role": batch.material_role,
                "evaluation_regime": batch.evaluation_regime,
                "natural_support_state": batch.scientific_values["natural_support_state"],
                "operation_config_digest": batch.scientific_values["operation_config_digest"],
                "airflow_source": terminal.airflow_source,
                "available_learning_views": list(terminal.available_learning_views),
                "export_contract_sha256": terminal.export_contract_sha256,
                "steady_flow_conditioning": copy.deepcopy(batch.scientific_values["steady_flow_conditioning"]),
            }
        )
        for case in terminal.cases:
            record = case.record_payload()
            case_payload = case.metadata_payload()
            expected_source = {
                "material_family": batch.material_family,
                "material_role": batch.material_role,
                "evaluation_regime": str(plan["evaluation_regime"]),
                "sampling_regime": batch.sampling_regime,
                "natural_support_state": "natural" if batch.sampling_regime == "natural" else "parameter_ood",
            }
            observed_source = {name: case_payload.get(name) for name in expected_source}
            if (
                observed_source != expected_source
                or batch.evaluation_regime != plan["evaluation_regime"]
                or batch.material_role != plan["source_role"]
                or case_payload.get("ood", {}).get("natural_support_state") != expected_source["natural_support_state"]
            ):
                message = (
                    f"Case {batch.batch_name}/{case.case_id} disagrees with package source-role/evaluation ownership: "
                    f"expected={expected_source}, actual={observed_source}."
                )
                raise ValueError(message)
            candidates.append(
                {
                    "batch": batch,
                    "terminal_evidence": terminal,
                    "case_evidence": case,
                    "manifest": manifest,
                    "record": record,
                    "case_payload": case_payload,
                    "batch_id": batch.batch_id,
                    "batch_name": batch.batch_name,
                    "case_id": case.case_id,
                    "package_case_id": _global_case_id(batch.batch_name, case.case_id),
                    "material_family": case.material_family,
                    "simulation_profile": terminal.simulation_profile,
                    "case_input_id": case.case_input_id,
                    "simulation_case_id": case.simulation_case_id,
                    "case_hdf5_relative": case.hdf5_path.relative_to(storage).as_posix(),
                    "case_hdf5_sha256": case.case_hdf5_sha256,
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
        "prefer_transient_source": _TRANSIENT_PROFILE_CONTRACT.id,
        "prefer_steady_source": _STEADY_PROFILE_CONTRACT.id,
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
    task_model_inputs = tuple(field.name for field in task.inputs if field.role != "coordinate")
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
        admitted_model_inputs = tuple(
            str(dependency.get("name")) for dependency in dependencies if isinstance(dependency, dict) and dependency.get("owner") == "model_input"
        )
        if admitted_model_inputs != task_model_inputs:
            message = (
                "Hidden steady-flow conditioning detected: admitted ordered model-input dependencies "
                f"{list(admitted_model_inputs)} do not equal the authoritative steady TaskSpec fields {list(task_model_inputs)}."
            )
            raise ValueError(message)
        dependency_by_name = {str(dependency.get("name")): dependency for dependency in dependencies if isinstance(dependency, dict)}
        for name, dependency in dependency_by_name.items():
            owner = dependency.get("owner")
            if owner == "model_input" and dependency.get("affects_stationary_solution") is not True:
                message = f"Steady-flow model input {name!r} is not declared as affecting the stationary solution."
                raise ValueError(message)
            if dependency.get("affects_stationary_solution") is not True:
                continue
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
    fixed_by_name = {item["name"]: item for item in fixed}
    expected_fixed_units = {field.name: field.unit for field in _STEADY_PROFILE_CONTRACT.stationary_fixed_fields}
    observed_fixed_units = {name: item["unit"] for name, item in fixed_by_name.items()}
    if observed_fixed_units != expected_fixed_units:
        message = "Steady-flow package-fixed field names or units violate the Generation profile contract."
        raise ValueError(message)
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
    """Return scalar or complete-record OOD support and distance evidence."""
    case_payload = candidate["case_payload"]
    ood = case_payload.get("ood")
    if not isinstance(ood, dict):
        message = f"Case {candidate['package_case_id']!r} has malformed OOD provenance."
        raise TypeError(message)
    selected = ood.get("selected_units")
    selection_details = ood.get("selections")
    if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected) or not isinstance(selection_details, dict):
        message = f"Case {candidate['package_case_id']!r} has malformed OOD selections."
        raise TypeError(message)
    material = candidate["batch"].scientific_values["material"]
    registry = material["parameter_registry"]
    coupled_contracts = material["coupled_ood_records"]
    sampled_values = case_payload.get("sampled_values")
    coupled = case_payload.get("coupled_selections")
    if not isinstance(sampled_values, dict) or not isinstance(coupled, dict):
        message = f"Case {candidate['package_case_id']!r} lacks sampled OOD evidence."
        raise TypeError(message)
    parameters: list[dict[str, Any]] = []
    for name in selected:
        entry = registry.get(name)
        if isinstance(entry, dict):
            block = entry.get("block")
            detail = selection_details.get(name)
            transform: Any
            id_support = {key: copy.deepcopy(entry[key]) for key in ("lower", "upper", "sets") if key in entry}
            ood_support = copy.deepcopy(entry.get("ood", entry.get("ood_values", entry.get("ood_sets"))))
            transform = entry.get("transform")
            parameters.append(
                {
                    "name": name,
                    "group": entry.get("ood_group"),
                    "block": block,
                    "kind": entry.get("kind"),
                    "transform": transform,
                    "unit": entry.get("unit"),
                    "id_support": id_support,
                    "ood_support": ood_support,
                    "sampled_value": copy.deepcopy(sampled_values.get(name)),
                    "coupled_selection": coupled.get(name),
                    "transformed_coordinate_evidence": copy.deepcopy(detail),
                    "ood_provenance": copy.deepcopy(entry.get("ood_provenance")),
                }
            )
            continue
        contract = coupled_contracts.get(name)
        if not isinstance(contract, dict):
            message = f"OOD unit {name!r} is absent from scalar and coupled material contracts."
            raise TypeError(message)
        components = list(contract["components"])
        parameters.append(
            {
                "name": name,
                "group": contract["ood_group"],
                "block": contract["block"],
                "kind": "complete_coupled_record",
                "transform": "component_owned",
                "unit": copy.deepcopy(contract["units"]),
                "id_support": {
                    component: {key: copy.deepcopy(registry[component][key]) for key in ("lower", "upper", "value") if key in registry[component]}
                    for component in components
                },
                "ood_support": copy.deepcopy(contract["records"]),
                "sampled_value": {component: copy.deepcopy(sampled_values[component]) for component in components},
                "coupled_selection": coupled.get(name),
                "transformed_coordinate_evidence": copy.deepcopy(selection_details.get(name)),
                "ood_provenance": copy.deepcopy(contract.get("ood_provenance")),
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
    selected = evidence["parameters"]
    relevant = (
        tuple(str(parameter["name"]) for parameter in selected if parameter["block"] in view.parameter_ood_blocks)
        if view.parameter_ood_blocks
        else tuple(str(parameter["name"]) for parameter in selected)
    )
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


def prepare_campaign_packages(
    campaign: generation.cases.config.CampaignConfig,
    *,
    storage_root: Path | str | None,
    selected_plans: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[PreparedPackage, ...]:
    """Admit selected packages together so membership and leakage are global."""
    plans = [copy.deepcopy(dict(plan)) for plan in (campaign.dataset_packages if selected_plans is None else selected_plans)]
    source_by_regime: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for plan in plans:
        regime = str(plan["evaluation_regime"])
        if regime not in source_by_regime:
            source_by_regime[regime] = _load_candidates(campaign, plan, storage_root=storage_root)
    id_plans = [plan for plan in plans if plan["evaluation_regime"] == "id"]
    shared_membership: dict[str, str] = {}
    technical_smoke = campaign.campaign_purpose == "technical_runtime_smoke"
    if id_plans:
        _id_records, id_candidates = source_by_regime["id"]
        physical_id_candidates, _id_decisions = resolve_duplicate_case_inputs(
            id_candidates,
            dataset_view="shared_id_membership",
            policy=campaign.duplicate_case_input_policy,
        )
        if technical_smoke:
            shared_membership = {str(candidate["case_input_id"]): views.TECHNICAL_SMOKE_MEMBERSHIP for candidate in physical_id_candidates}
        else:
            shared_membership = _shared_id_membership(id_plans[0], physical_id_candidates)
    elif campaign.campaign_purpose == "family_generalization":
        message = "Family-generalization campaigns must declare an ID package."
        raise ValueError(message)
    prepared: list[PreparedPackage] = []
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
        membership: dict[str, list[str]] = (
            {views.TECHNICAL_SMOKE_MEMBERSHIP: []}
            if regime == "id" and technical_smoke
            else {role: [] for role in _ID_MEMBERSHIP_ROLES}
            if regime == "id"
            else {regime: []}
        )
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
            PreparedPackage(
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


def _validate_no_id_ood_overlap(prepared: Sequence[PreparedPackage]) -> None:
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
