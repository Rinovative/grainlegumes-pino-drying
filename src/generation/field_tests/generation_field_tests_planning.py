"""Build the stage-A screening roster and the stage-B confirmation roster of one Generation field test.

Responsibilities:
  - Admit base candidates: original first-attempt successes without support or schedule retry, transient ones at target;
    Train only, except for a material without Train cases, which takes any design kind
  - Order each material's candidates deterministically by maximin distance over the transformed keep-row coordinates
  - Build per base the gate, the references R0, R_sin, R_gauss, R_texture, R_event, R_jitter and R_perturb, and the
    challenges of every tested row against each of its references
  - Add the extra bases of the coupled-stream rows, and in stage B the fresh-base challenges of rows screened remove, the joint
    test, and the constant-combination displacements of every prespecified merge whose two rows stage A tested
  - Replan a roster version that replaces every base with a failed gate by its material's next unused base

Design principles:
  - Challenge values come from the sampling hook's feasible range at the reference's fixed companions; an empty contrast keeps
    the registry edge so its member is recorded infeasible instead of dropped
  - Every choice is a pure function of the frozen Source, the registry, and stage A's verdicts, so a replan yields the same digest
  - Every material of the Source design supplies its full quota of bases, or no roster is built
  - The keep-by-prior-evidence rows vary across bases and are never intervened on, except the R_texture partner
  - Merges are prespecified here (E1) and confirmed in stage B regardless of the screening; stage A emits no merge verdict

This module does NOT:
  - Run cases, write member records, or compute tolerances and verdicts

Notes:
  Counts follow plan 223: steady 8 bases per material, transient 2 bases per material plus 2 extra bases for the four
  coupled-stream rows; stage B uses 4 fresh bases per material for the joint test, of which the first 2 carry the challenges
  and the 2 merge displacements (both ends of the jointly feasible segment of the constant combination at R0).
  Bases are ordered from the candidate nearest the material's centroid, then by largest minimum distance to the chosen ones;
  ties follow case order. Coordinates are the registry transforms (log, logit, linear) scaled to the ID support.

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src import common
from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.contracts import generation_contracts_registry as registry_service
from src.generation.publication import generation_publication_catalog as catalog

from . import generation_field_tests_record as record
from . import generation_field_tests_roster as roster_service

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.generation.cases import generation_cases_config as config_contract

DESIGN_KIND: Final = "train"
MAX_ATTEMPTS: Final = 3
_AIRFLOW_KEEP: Final = ("kappa_mean", "kappa_cv", "airflow.superficial_velocity")
KEEP_ROWS: Final = {
    "steady_flow": _AIRFLOW_KEEP,
    "transient_drying": (
        *_AIRFLOW_KEEP,
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "T_in_base",
        "T_in_amp",
        "omega_in_base",
        "rho_bu_dry_ref",
        "r_surf_0",
        "r_int_surf",
        "f_surf",
    ),
}
COUPLED_STREAM_ROWS: Final = (
    "bed.perturbations.amplitude",
    "bed.perturbations.granularity",
    "bed.structure.coarse_len_rel",
    "schedule.event_count",
)
_BASES_PER_MATERIAL: Final = {"steady_flow": 8, "transient_drying": 2}
_COUPLED_EXTRA_BASES: Final = {"steady_flow": 0, "transient_drying": 2}
_FRESH_CHALLENGE_BASES: Final = 2
_FRESH_JOINT_BASES: Final = 4
# Each partner reference moves one partner from R0 to an active value: the largest feasible magnitude, the upper feasible
# edge, or for the schedule simplex the variant with the event weight at its upper bound (plan 223, «Stage A»).
_PARTNERS: Final = {
    "R_sin": ("pressure_bc.sin_amp", "magnitude"),
    "R_gauss": ("pressure_bc.gauss_amp", "magnitude"),
    "R_texture": ("initial_moisture.amplitude_db", "upper"),
    "R_event": ("schedule.component_weights", "event"),
    "R_jitter": ("permeability.orientation.jitter", "upper"),
    "R_perturb": ("bed.perturbations.amplitude", "upper"),
}
_ROW_REFERENCES: Final = {
    "pressure_bc.sin_freq": ("R_sin",),
    "pressure_bc.sin_phase": ("R_sin",),
    "pressure_bc.gauss_count": ("R_gauss",),
    "pressure_bc.gauss_width": ("R_gauss",),
    "pressure_bc.gauss_jitter": ("R_gauss",),
    "schedule.event_duration_rel": ("R_event",),
    "schedule.event_width_rel": ("R_event",),
    "permeability.orientation.smooth_len_rel": ("R_jitter",),
    # Perturbation shape rows are screened at the nominal amplitude and re-checked at the upper amplitude.
    "bed.perturbations.granularity": ("R0", "R_perturb"),
    "bed.perturbations.sign_bias": ("R0", "R_perturb"),
}
_TEXTURE_ROW_PREFIX: Final = "initial_moisture.structure."
# sin_phase levels: 0 and 2 pi give the same shape, so the nominal pi is challenged at pi -/+ pi/2 and at 0.
_PHASE_LEVELS: Final = (("phase_minus", math.pi / 2), ("phase_plus", 3 * math.pi / 2), ("phase_zero", 0.0))
_EMPTY_CONTRAST_TOLERANCE: Final = 1e-9
# Prespecified exact merges (plan 223, E1): the permeability map reads strength and max_ratio only through
# strength * (max_ratio - 1) (generation_cases_fields.py, principal_ratio). Value: (multiplier, ratio) of that product.
_MERGES: Final = {
    "merge:permeability.anisotropy.max_ratio+permeability.anisotropy.strength": (
        "permeability.anisotropy.strength",
        "permeability.anisotropy.max_ratio",
    ),
}


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One admitted base case with its frozen batch, roster base binding, references, and keep-row coordinates."""

    config: config_contract.GenerationConfig
    case_index: int
    base: dict[str, Any]
    references: dict[str, dict[str, Any]]
    coordinates: tuple[float, ...]


def _registry(config: config_contract.GenerationConfig) -> Mapping[str, Mapping[str, Any]]:
    """Return the resolved parameter registry of one batch."""
    return config.scientific_values["material"]["parameter_registry"]


def tested_rows(task: str, config: config_contract.GenerationConfig) -> tuple[str, ...]:
    """Return the sampled primitives of the batch's active blocks that are not kept on prior evidence, in block order."""
    registry = _registry(config)
    return tuple(
        str(name)
        for plan in config.scientific_values["sampling"]["blocks"].values()
        for name in plan["parameters"]
        if registry[name]["kind"] in {"interval", "integer", "simplex"} and name not in KEEP_ROWS[task]
    )


def _nominal(entry: Mapping[str, Any], name: str) -> Any:
    """Return one primitive's configured nominal in its canonical intervention form."""
    if "nominal" not in entry:
        message = f"Tested row {name!r} has no configured nominal."
        raise ValueError(message)
    value = entry["nominal"]
    if entry["kind"] == "simplex":
        weights = value if isinstance(value, dict) else dict(zip(entry["components"], value, strict=True))
        return {str(component): float(weights[component]) for component in entry["components"]}
    return int(value) if entry["kind"] == "integer" else float(value)


def reference_kinds(row: str) -> tuple[str, ...]:
    """Return the references a row's challenges pair with; the first one names the row's plain challenge members."""
    if row.startswith(_TEXTURE_ROW_PREFIX):
        return ("R_texture",)
    return _ROW_REFERENCES.get(row, ("R0",))


def _references(task: str, config: config_contract.GenerationConfig, case_index: int) -> dict[str, dict[str, Any]]:
    """Return every applicable reference intervention of one base; any infeasible one raises ``InterventionError``."""
    registry = _registry(config)
    r0 = {row: _nominal(registry[row], row) for row in tested_rows(task, config)}
    sampling_service.sample_case(config, case_index, intervention=r0)
    references = {"R0": r0}
    active = {str(name) for plan in config.scientific_values["sampling"]["blocks"].values() for name in plan["parameters"]}
    for kind, (partner, rule) in _PARTNERS.items():
        if partner not in active:
            continue
        value: Any
        if registry[partner]["kind"] == "simplex":
            value = _simplex_variant(registry[partner], _nominal(registry[partner], partner), rule)
        else:
            lower, upper = sampling_service.feasible_range(config, case_index, partner, background=r0)
            value = (upper if abs(upper) >= abs(lower) else lower) if rule == "magnitude" else upper
        references[kind] = {**r0, partner: value}
        sampling_service.sample_case(config, case_index, intervention=references[kind])
    return references


def _simplex_variant(entry: Mapping[str, Any], nominal: Mapping[str, float], component: str) -> dict[str, float]:
    """Set one component to its upper bound and share the rest by the nominal ratio, lifting any share below the minimum."""
    lower, upper = float(entry["minimum_each"]), float(entry["maximum_each"])
    rest = [str(name) for name in entry["components"] if name != component]
    remaining = 1.0 - upper
    weights = {name: remaining * nominal[name] / sum(nominal[other] for other in rest) for name in rest}
    lifted = [name for name in rest if weights[name] < lower]
    free = [name for name in rest if name not in lifted]
    free_total = sum(weights[name] for name in free)
    for name in lifted:
        weights[name] = lower
    for name in free:
        weights[name] = (remaining - lower * len(lifted)) * weights[name] / free_total
    return {str(name): (upper if name == component else weights[str(name)]) for name in entry["components"]}


def _challenges(config: config_contract.GenerationConfig, case_index: int, row: str, reference: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return the (level, value) challenges of one row against its reference."""
    entry = _registry(config)[row]
    if entry["kind"] == "simplex":
        return [(f"simplex_{component}", _simplex_variant(entry, reference[row], str(component))) for component in entry["components"]]
    if entry.get("transform") == "phase":
        return list(_PHASE_LEVELS)
    if entry["kind"] == "integer":
        return [("low", int(entry["lower"])), ("high", int(entry["upper"]))]
    lower, upper = sampling_service.feasible_range(config, case_index, row, background=reference)
    nominal = float(reference[row])
    levels = []
    for level, edge, registry_edge in (("low", lower, float(entry["lower"])), ("high", upper, float(entry["upper"]))):
        empty = math.isclose(edge, nominal, rel_tol=_EMPTY_CONTRAST_TOLERANCE, abs_tol=0.0) and edge != registry_edge
        levels.append((level, registry_edge if empty else edge))
    return levels


def _merge_displacements(config: config_contract.GenerationConfig, case_index: int, merge: str, reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the two constant-combination displacements of one merge at R0: both ends of the jointly feasible multiplier segment."""
    multiplier, ratio = _MERGES[merge]
    combination = float(reference[multiplier]) * (float(reference[ratio]) - 1.0)
    if combination <= 0.0:
        message = f"{merge}: the combination is not positive at R0, so a constant-combination displacement is undefined."
        raise ValueError(message)
    low, high = sampling_service.feasible_range(config, case_index, multiplier, background=reference)
    ratio_low, ratio_high = sampling_service.feasible_range(config, case_index, ratio, background=reference)
    low = max(low, combination / (ratio_high - 1.0))
    high = high if ratio_low <= 1.0 else min(high, combination / (ratio_low - 1.0))
    return [{**reference, multiplier: value, ratio: 1.0 + combination / value} for value in (low, high)]


def _slug(text: str) -> str:
    """Return one lowercase member-id component."""
    return text.lower().replace(".", "_")


def _members_for_base(
    candidate: _Candidate, rows: tuple[str, ...], *, joint: Mapping[str, Any] | None = None, merges: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Return the gate, the references the rows and merges need, the challenges, the merge displacements, and an optional joint member."""
    base = candidate.base
    prefix = f"{base['material_family']}-{base['case_id']}"
    gate_id = f"{prefix}-gate"

    def entry(member_id: str, role: str, *, intervention: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {"gate_member_id": gate_id, "reference_member_id": None, "row": None, "level": "nominal"}
        canonical = sampling_service.canonical_intervention(intervention or {}, scientific_values=candidate.config.scientific_values)
        return {"member_id": member_id, "role": role, "base": dict(base), **defaults, **fields, "intervention": canonical}

    members = [entry(gate_id, "gate", gate_member_id=None)]
    kinds = {kind for row in rows for kind in reference_kinds(row)} | ({"R0"} if rows or merges else set())
    missing = sorted(kinds - set(candidate.references))
    if missing:
        message = f"Base {prefix} lacks the references {missing} its rows need."
        raise ValueError(message)
    members.extend(
        entry(f"{prefix}-ref-{_slug(kind)}", "reference", reference_kind=kind, intervention=candidate.references[kind])
        for kind in roster_service.REFERENCE_KINDS
        if kind in kinds
    )
    for number, merge in enumerate(merges, start=1):
        displacements = _merge_displacements(candidate.config, candidate.case_index, merge, candidate.references["R0"])
        members.extend(
            entry(
                f"{prefix}-merge{number}-d{index}", "challenge", reference_member_id=f"{prefix}-ref-r0", row=merge, level="merge", intervention=value
            )
            for index, value in enumerate(displacements, start=1)
        )
    for row in rows:
        for position, kind in enumerate(reference_kinds(row)):
            reference = candidate.references[kind]
            suffix = "" if position == 0 else f"-at-{_slug(kind)}"
            members.extend(
                entry(
                    f"{prefix}-{_slug(row)}-{level}{suffix}",
                    "challenge",
                    reference_member_id=f"{prefix}-ref-{_slug(kind)}",
                    row=row,
                    level=level,
                    intervention={**reference, row: value},
                )
                for level, value in _challenges(candidate.config, candidate.case_index, row, reference)
            )
    if joint is not None:
        members.append(entry(f"{prefix}-joint", "joint", reference_member_id=gate_id, level="joint", intervention=dict(joint)))
    return members


def _admit(
    task: str,
    campaign: config_contract.CampaignConfig,
    row: Mapping[str, Any],
    *,
    source: Path,
    storage: Path,
) -> _Candidate:
    """Admit one catalog row as a base candidate or raise ``ValueError`` naming why it lies in the excluded region."""
    material, case_id, kind = str(row["material_family"]), str(row["case_id"]), str(row["sampling_regime"])
    # A material without Train cases (material OOD) takes its bases from any design kind: bases are inputs, not selection data.
    with_train = any(batch.material_family == material and batch.sampling_regime == DESIGN_KIND for batch in campaign.batches)
    if row["outcome"] != "successful" or row["origin"] != "primary" or row["attempts"] or (with_train and kind != DESIGN_KIND):
        message = "not an original first-attempt success of the material's base design kind"
        raise ValueError(message)
    case_h5 = (storage / str(row["case_hdf5_relative"])).resolve()
    case_json = roster_service.base_inputs_directory(case_h5) / common.paths.GENERATION_CASE_PAYLOAD_FILENAME
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    if roster_service.acceptance_attempts(payload) not in {(1, 1), (1, None)}:
        message = "spatial support or schedule feasibility retry"
        raise ValueError(message)
    if task == "transient_drying":
        status_path = common.paths.resolve_generation_result_evidence_directory(case_h5.parent) / "status.json"
        if json.loads(status_path.read_text(encoding="utf-8")).get("target_reached") is not True:
            message = "drying target not reached (censored)"
            raise ValueError(message)
    case_index = int(case_id.removeprefix("case_"))
    config = roster_service.case_batch(campaign, material, kind, case_index)
    base: dict[str, Any] = {
        "material_family": material,
        "design_kind": kind,
        "design_index": int(config.case_assignment(case_index)["design_index"]),
        "case_id": case_id,
        "case_path": case_h5.relative_to(source).as_posix(),
        "case_sha256": str(row["case_hdf5_sha256"]),
        "case_json_sha256": common.serialization.file_sha256(case_json),
    }
    registry = _registry(config)
    coordinates = []
    for name in KEEP_ROWS[task]:
        entry = registry[name]
        lower, upper = (registry_service.transformed_coordinate(float(entry[edge]), entry) for edge in ("lower", "upper"))
        coordinates.append((registry_service.transformed_coordinate(float(payload["sampled_values"][name]), entry) - lower) / (upper - lower))
    try:
        references = _references(task, config, case_index)
    except sampling_service.InterventionError as error:
        message = f"a reference is infeasible at this base: {error}"
        raise ValueError(message) from error
    return _Candidate(config=config, case_index=case_index, base=base, references=references, coordinates=tuple(coordinates))


def _maximin_order(candidates: list[_Candidate]) -> list[_Candidate]:
    """Order candidates from the one nearest their centroid by largest minimum distance to those already chosen."""
    if not candidates:
        return []
    points = np.asarray([candidate.coordinates for candidate in candidates], dtype=np.float64)
    order = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    while len(order) < len(candidates):
        distances = np.min(np.linalg.norm(points[:, None, :] - points[order][None, :, :], axis=2), axis=1)
        distances[order] = -1.0
        order.append(int(np.argmax(distances)))
    return [candidates[index] for index in order]


def ordered_candidates(
    task: str, base_source: Mapping[str, Any], *, storage_root: Path | str | None
) -> tuple[dict[str, list[_Candidate]], list[dict[str, str]]]:
    """Return every material of the Source design (in batch order) with its admitted candidates in maximin order, and every excluded case."""
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    campaign = roster_service.load_base_campaign(task, base_source, storage_root=storage)
    source = roster_service.source_directory(task, base_source, storage_root=storage).resolve()
    # The required material set is the design's, so a material whose every case is excluded still demands its quota.
    admitted: dict[str, list[_Candidate]] = {batch.material_family: [] for batch in campaign.batches}
    excluded: list[dict[str, str]] = []
    for row in catalog.list_campaign_cases(task, str(base_source["campaign_name"]), storage_root=storage):
        try:
            candidate = _admit(task, campaign, row, source=source, storage=storage)
        except (ValueError, KeyError, OSError) as error:
            excluded.append({"material_family": str(row["material_family"]), "case_id": str(row["case_id"]), "reason": str(error)})
            continue
        admitted[candidate.base["material_family"]].append(candidate)
    return {material: _maximin_order(items) for material, items in admitted.items()}, excluded


def _take(ordered: Mapping[str, list[_Candidate]], material: str, count: int, *, skip: set[tuple[str, str]]) -> list[_Candidate]:
    """Return the next ``count`` unused candidates of one material in maximin order, or raise when too few remain."""
    chosen = [candidate for candidate in ordered.get(material, []) if (material, candidate.base["case_id"]) not in skip][:count]
    if len(chosen) < count:
        message = f"Material {material!r} has {len(chosen)} unused admitted bases; {count} are needed."
        raise ValueError(message)
    return chosen


def build_stage_a(task: str, test_id: str, campaign_name: str, *, storage_root: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return the stage-A screening roster of one Source and its planning evidence (base order and the excluded region).

    Raises
    ------
    ValueError
        If a material has fewer admitted bases than the plan needs.

    """
    base_source = {"campaign_name": campaign_name, "plan_sha256": roster_service.plan_sha256(task, campaign_name, storage_root=storage_root)}
    ordered, excluded = ordered_candidates(task, base_source, storage_root=storage_root)
    members: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for material in ordered:
        for candidate in _take(ordered, material, _BASES_PER_MATERIAL[task], skip=used):
            members.extend(_members_for_base(candidate, tested_rows(task, candidate.config)))
            used.add((material, candidate.base["case_id"]))
    for extra in range(_COUPLED_EXTRA_BASES[task]):
        material = list(ordered)[extra % len(ordered)]
        candidate = _take(ordered, material, 1, skip=used)[0]
        rows = tuple(row for row in COUPLED_STREAM_ROWS if row in tested_rows(task, candidate.config))
        members.extend(_members_for_base(candidate, rows))
        used.add((material, candidate.base["case_id"]))
    roster = roster_service.build_roster(test_id=test_id, task=task, stage="A", base_source=base_source, members=members, max_attempts=MAX_ATTEMPTS)
    evidence = {"base_order": {material: [item.base["case_id"] for item in items] for material, items in ordered.items()}, "excluded": excluded}
    return roster, evidence


def build_stage_b(task: str, test_id: str, stage_a_test_id: str, *, storage_root: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return the stage-B confirmation roster built from stage A's verdicts.

    Rows screened remove get fresh-base challenges and the joint removal test. Every prespecified merge whose two rows
    stage A tested gets two constant-combination displacements per challenge base, whatever their screening verdicts.

    Raises
    ------
    ValueError
        If the verdicts do not belong to stage A's roster or carry a merge verdict, nothing is screened remove and no merge
        applies, or fresh bases run short.

    """
    directory_a = common.paths.resolve_field_test_dir(task, stage_a_test_id, storage_root=storage_root)
    roster_a, version_a = roster_service.load_roster_version(directory_a)
    verdicts_path = record.record_path(directory_a, "analysis", "verdicts.json")
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    if (
        verdicts.get("schema_kind") != "field_test_verdicts"
        or verdicts.get("schema_version") != 1
        or verdicts.get("roster_sha256") != roster_a["roster_sha256"]
        or verdicts.get("roster_version") != version_a
        or verdicts.get("stage") != "A"
        or not isinstance(verdicts.get("tolerances_sha256"), str)
        or any(row["verdict"] == "merge" for row in verdicts["rows"])
    ):
        message = f"Stage-A verdicts do not belong to roster version {version_a} or carry a merge verdict: {verdicts_path}"
        raise ValueError(message)
    removals = tuple(str(row["row"]) for row in verdicts["rows"] if row["verdict"] == "remove")
    tested = {str(member["row"]) for member in roster_a["members"] if member["role"] == "challenge"}
    merges = tuple(name for name, pair in _MERGES.items() if set(pair) <= tested)
    if not removals and not merges:
        message = "Stage A screened no row remove and tested no prespecified merge pair; stage B has nothing to confirm."
        raise ValueError(message)
    ordered, excluded = ordered_candidates(task, roster_a["base_source"], storage_root=storage_root)
    used = {(str(member["base"]["material_family"]), str(member["base"]["case_id"])) for member in roster_a["members"]}
    members: list[dict[str, Any]] = []
    for material in ordered:
        for position, candidate in enumerate(_take(ordered, material, _FRESH_JOINT_BASES, skip=used)):
            unknown = sorted(set(removals) - set(tested_rows(task, candidate.config)))
            if unknown:
                message = f"Rows screened remove are not tested rows of {material}: {unknown}"
                raise ValueError(message)
            registry = _registry(candidate.config)
            joint = {row: _nominal(registry[row], row) for row in removals} or None
            challenged = position < _FRESH_CHALLENGE_BASES
            if joint is None and not challenged:
                continue
            members.extend(_members_for_base(candidate, removals if challenged else (), joint=joint, merges=merges if challenged else ()))
    derived_from = {
        "test_id": stage_a_test_id,
        "roster_sha256": str(roster_a["roster_sha256"]),
        "roster_version": version_a,
        "verdicts_sha256": common.serialization.file_sha256(verdicts_path),
        # Stage B is judged against exactly this frozen stage-A tolerance table and never freezes its own.
        "tolerances_sha256": str(verdicts["tolerances_sha256"]),
    }
    roster = roster_service.build_roster(
        test_id=test_id,
        task=task,
        stage="B",
        base_source=roster_a["base_source"],
        members=members,
        max_attempts=MAX_ATTEMPTS,
        derived_from=derived_from,
    )
    return roster, {"removals": list(removals), "merges": list(merges), "excluded": excluded}


def replace_failed_bases(task: str, test_id: str, *, storage_root: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return the next roster version: every base whose gate sealed a failure is replaced by its material's next unused base.

    The replacement base gets the failed base's members anew (the same rows, merges, and joint test); every earlier member
    stays unchanged, so the failed base keeps its sealed records and the replacement is recorded in ``replacements``.

    Raises
    ------
    ValueError
        If no failed base is left unreplaced or a material has no unused admitted base.

    """
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    roster, version = roster_service.load_roster_version(directory)
    replaced = {(item["material_family"], item["failed_case_id"]) for item in roster.get("replacements", [])}
    failed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for gate in (item for item in roster["members"] if item["role"] == "gate"):
        seal = record.read_seal(directory, str(gate["member_id"]))
        key = (gate["base"]["material_family"], gate["base"]["case_id"])
        if seal is not None and key not in replaced and not record.gate_succeeded(seal):
            failed.append((gate, seal))
    if not failed:
        message = f"Field test {test_id!r} has no failed base left to replace."
        raise ValueError(message)
    ordered, excluded = ordered_candidates(task, roster["base_source"], storage_root=storage_root)
    used = {(str(item["base"]["material_family"]), str(item["base"]["case_id"])) for item in roster["members"]}
    added: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    for gate, seal in failed:
        material = str(gate["base"]["material_family"])
        members = [item for item in roster["members"] if item["base"] == gate["base"]]
        challenges = [item for item in members if item["role"] == "challenge"]
        rows = tuple(dict.fromkeys(str(item["row"]) for item in challenges if item["level"] != "merge"))
        merges = tuple(dict.fromkeys(str(item["row"]) for item in challenges if item["level"] == "merge"))
        joint = next((item["intervention"] for item in members if item["role"] == "joint"), None)
        candidate = _take(ordered, material, 1, skip=used)[0]
        used.add((material, candidate.base["case_id"]))
        added.extend(_members_for_base(candidate, rows, joint=joint, merges=merges))
        replacements.append(
            {
                "material_family": material,
                "failed_case_id": gate["base"]["case_id"],
                "failed_gate_member_id": gate["member_id"],
                "failed_disposition": seal["disposition"],
                "replacement_case_id": candidate.base["case_id"],
            }
        )
    return roster_service.next_version(roster, added, replacements, version=version + 1), {"replacements": replacements, "excluded": excluded}
