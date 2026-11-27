"""Define, validate, and persist the frozen roster of one Generation field test and locate its base Source.

Responsibilities:
  - Define the roster, role, level, reference, and disposition vocabularies shared with the analysis build
  - Validate one roster fail-closed and bind it to the sha256 of its whole canonical payload; digest one member entry
  - Publish each roster version once and load the current version only while the whole consecutive chain holds
  - Reconstruct a base batch from the Source's frozen plan and verify a member's published base files by their digests

Design principles:
  - A roster version is frozen once written: a later version only appends the members of replacement bases and never
    changes, replaces, or removes an earlier member
  - Interventions are canonical JSON, so the roster digest binds every declared value and every replacement record
  - Base files are addressed relative to the Source root and admitted only through their recorded sha256

This module does NOT:
  - Choose bases or interventions, run cases, or write member records
  - Verify a replacement against its failed gate's seal (the record module does, for the members a verdict judges)
  - Read or write anything below a Source except the published plan, case.json, and case.h5 it verifies

Notes:
  roster.json (schema_kind field_test_roster, version 1) holds test_id, task, stage, base_source {campaign_name (the Source
  folder below 01_generation/TASK/), plan_sha256}, max_attempts, roster_sha256 (the canonical digest of every other roster
  field), and members. Each member holds member_id, role, base {material_family, design_kind, design_index, case_id,
  case_path (Source-relative result/case.h5), case_sha256, case_json_sha256 (its inputs/case.json)}, gate_member_id,
  reference_member_id, row, level, intervention, and for a reference its reference_kind. Planned member ids start with
  <material_family>-<case_id>-. A stage-B roster adds derived_from {test_id, roster_sha256, roster_version,
  verdicts_sha256, tolerances_sha256} of stage A. A merge confirmation is a challenge with level merge, row
  merge:<param_a>+<param_b> (names sorted), and the intervention of its reference with exactly both parameters moved along
  their constant combination; level joint stays reserved for the joint removal test.
  Version N >= 2 lives in roster.vN.json and adds supersedes {version, roster_sha256} and replacements [{material_family,
  failed_case_id, failed_gate_member_id, failed_disposition, replacement_case_id}]. The appended members of a version are
  exactly its new replacement bases, each mirroring its failed base's planned members (id suffixes, roles, links, rows,
  levels, reference kinds, and the joint intervention); no other roster.v*.json may exist beside the chain.

"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.generation.cases import generation_cases_config as config_contract

ROSTER_SCHEMA_KIND: Final = "field_test_roster"
ROSTER_SCHEMA_VERSION: Final = 1
ROSTER_FILENAME: Final = "roster.json"
TASKS: Final = ("steady_flow", "transient_drying")
STAGES: Final = ("A", "B")
ROLES: Final = ("gate", "reference", "challenge", "joint")
REFERENCE_KINDS: Final = ("R0", "R_sin", "R_gauss", "R_texture", "R_event", "R_jitter", "R_perturb")
DISPOSITIONS: Final = ("succeeded", "solver_failure", "infrastructure_stop", "gate_blocked", "binding_refused", "infeasible")
_LEVEL = re.compile(r"nominal|low|high|phase_minus|phase_plus|phase_zero|joint|merge|simplex_[a-z]+")
MERGE_ROW_PREFIX: Final = "merge:"
_VERSION_FILE = re.compile(r"roster\.v(?P<version>[0-9]+)\.json")
_MEMBER_ID = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_MEMBER_ID_MAX_LENGTH: Final = 128
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROSTER_KEYS: Final = frozenset(
    {"schema_kind", "schema_version", "test_id", "task", "stage", "base_source", "max_attempts", "roster_sha256", "members"}
)
_MEMBER_KEYS: Final = frozenset({"member_id", "role", "base", "gate_member_id", "reference_member_id", "row", "level", "intervention"})
_BASE_KEYS: Final = frozenset({"material_family", "design_kind", "design_index", "case_id", "case_path", "case_sha256", "case_json_sha256"})
_REPLACEMENT_KEYS: Final = frozenset({"material_family", "failed_case_id", "failed_gate_member_id", "failed_disposition", "replacement_case_id"})


class BaseBindingError(ValueError):
    """A member's base Source, frozen plan, or published case no longer matches the digests its roster binds."""


def roster_digest(payload: Mapping[str, Any]) -> str:
    """Return the roster digest: the sha256 of the canonical JSON of the whole roster payload except the digest itself."""
    return common.serialization.canonical_json_sha256({key: value for key, value in payload.items() if key != "roster_sha256"})


def member_sha256(member: Mapping[str, Any]) -> str:
    """Return the canonical digest of one roster member entry, which binds a submission to exactly that member."""
    return common.serialization.canonical_json_sha256(dict(member))


def _require(condition: bool, message: str) -> None:
    """Fail closed with one roster contract message."""
    if not condition:
        raise ValueError(message)


def _validate_member(member: Any, *, members: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate one member and its links to its gate and reference."""
    _require(isinstance(member, dict) and set(member) - {"reference_kind"} == _MEMBER_KEYS, "Roster member fields differ from the contract.")
    member_id = member["member_id"]
    label = f"Roster member {member_id!r}"
    _require(
        isinstance(member_id, str) and len(member_id) <= _MEMBER_ID_MAX_LENGTH and _MEMBER_ID.fullmatch(member_id) is not None,
        f"{label}: member_id must be a lowercase slug.",
    )
    base = member["base"]
    _require(isinstance(base, dict) and set(base) == _BASE_KEYS, f"{label}: base fields differ from the contract.")
    _require(
        all(_SHA256.fullmatch(str(base[key])) for key in ("case_sha256", "case_json_sha256"))
        and isinstance(base["design_index"], int)
        and not Path(str(base["case_path"])).is_absolute()
        and ".." not in Path(str(base["case_path"])).parts,
        f"{label}: base digests, design index, or Source-relative case path are malformed.",
    )
    role, level = member["role"], member["level"]
    _require(role in ROLES and isinstance(level, str) and _LEVEL.fullmatch(level) is not None, f"{label}: role or level is unknown.")
    # Values are canonicalized by the sampling authority when planned and checked again when generated.
    _require(isinstance(member["intervention"], dict), f"{label}: intervention must map primitive names to values.")
    _require((role == "reference") == ("reference_kind" in member), f"{label}: only a reference carries reference_kind.")
    if role == "gate":
        _require(
            member["gate_member_id"] is None and member["reference_member_id"] is None and member["intervention"] == {} and member["row"] is None,
            f"{label}: a gate reruns its unchanged base without links.",
        )
        _require(level == "nominal", f"{label}: a gate's level is nominal.")
        return
    gate = members.get(str(member["gate_member_id"]))
    _require(gate is not None and gate["role"] == "gate" and gate["base"] == base, f"{label}: gate_member_id must name the gate of its base.")
    _require(bool(member["intervention"]), f"{label}: a variant declares a non-empty intervention.")
    if role == "reference":
        _require(
            member["reference_kind"] in REFERENCE_KINDS and member["reference_member_id"] is None, f"{label}: reference kind or link is invalid."
        )
        _require(member["row"] is None and level == "nominal", f"{label}: a reference has no row and the nominal level.")
        return
    reference = members.get(str(member["reference_member_id"]))
    expected_role = "gate" if role == "joint" else "reference"
    _require(
        reference is not None and reference["role"] == expected_role and reference["base"] == base,
        f"{label}: reference_member_id must name a {expected_role} of the same base.",
    )
    _require((role == "joint") == (level == "joint"), f"{label}: only a joint member has the joint level.")
    _require(role == "joint" or isinstance(member["row"], str), f"{label}: a challenge names its row.")
    _require(
        (level == "merge") == str(member["row"]).startswith(MERGE_ROW_PREFIX), f"{label}: only a merge confirmation has the merge level and row."
    )


def validate_roster(payload: Any) -> dict[str, Any]:
    """
    Validate one roster document fail-closed and return it.

    Raises
    ------
    ValueError
        If any field, member, link, or the roster digest violates the roster contract.

    """
    _require(isinstance(payload, dict), "Field-test roster must be a JSON object.")
    _require(set(payload) - {"derived_from", "supersedes", "replacements"} == _ROSTER_KEYS, "Field-test roster fields differ from the contract.")
    _require(("supersedes" in payload) == ("replacements" in payload), "A roster version names what it supersedes and its replacements.")
    if "supersedes" in payload:
        prior = payload["supersedes"]
        _require(
            isinstance(prior, dict)
            and set(prior) == {"version", "roster_sha256"}
            and isinstance(prior["version"], int)
            and prior["version"] >= 1
            and _SHA256.fullmatch(str(prior["roster_sha256"])) is not None,
            "A roster version supersedes one earlier version by number and digest.",
        )
        _require(
            isinstance(payload["replacements"], list)
            and bool(payload["replacements"])
            and all(isinstance(item, dict) and set(item) == _REPLACEMENT_KEYS for item in payload["replacements"]),
            "A roster version records at least one base replacement.",
        )
    _require(
        payload["schema_kind"] == ROSTER_SCHEMA_KIND and payload["schema_version"] == ROSTER_SCHEMA_VERSION,
        "Field-test roster schema is not the current one.",
    )
    _require(payload["task"] in TASKS and payload["stage"] in STAGES, "Field-test roster task or stage is unknown.")
    _require((payload["stage"] == "B") == ("derived_from" in payload), "Only a stage-B roster names the stage it derives from.")
    common.paths.resolve_field_test_dir(payload["task"], payload["test_id"], storage_root=Path("/"))
    source = payload["base_source"]
    _require(
        isinstance(source, dict) and set(source) == {"campaign_name", "plan_sha256"} and _SHA256.fullmatch(str(source["plan_sha256"])) is not None,
        "Field-test base_source must name the Source folder and its plan digest.",
    )
    common.paths.validate_logical_name(source["campaign_name"], label="base campaign_name")
    _require(isinstance(payload["max_attempts"], int) and not isinstance(payload["max_attempts"], bool), "max_attempts must be an integer.")
    _require(payload["max_attempts"] >= 1, "max_attempts must be at least one.")
    members = payload["members"]
    _require(isinstance(members, list) and bool(members), "Field-test roster has no members.")
    by_id = {str(member.get("member_id")): member for member in members if isinstance(member, dict)}
    _require(len(by_id) == len(members), "Field-test member ids must be unique.")
    for member in members:
        _validate_member(member, members=by_id)
    gate_bases = [json.dumps(member["base"], sort_keys=True) for member in members if member["role"] == "gate"]
    _require(len(gate_bases) == len(set(gate_bases)), "Each base has exactly one gate.")
    _require(payload["roster_sha256"] == roster_digest(payload), "Field-test roster digest does not match its payload.")
    return payload


def build_roster(
    *,
    test_id: str,
    task: str,
    stage: str,
    base_source: Mapping[str, str],
    members: list[dict[str, Any]],
    max_attempts: int,
    derived_from: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one validated roster document whose digest binds ``members`` and every other field."""
    payload: dict[str, Any] = {
        "schema_kind": ROSTER_SCHEMA_KIND,
        "schema_version": ROSTER_SCHEMA_VERSION,
        "test_id": test_id,
        "task": task,
        "stage": stage,
        "base_source": dict(base_source),
        "max_attempts": max_attempts,
        "members": members,
    }
    if derived_from is not None:
        payload["derived_from"] = dict(derived_from)
    payload["roster_sha256"] = roster_digest(payload)
    return validate_roster(json.loads(json.dumps(payload)))


def publish_once(destination: Path, payload: Mapping[str, Any]) -> bool:
    """Publish one JSON object atomically without ever replacing a file; return False when that exact content already exists.

    Raises
    ------
    FileExistsError
        If ``destination`` already holds different content.

    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.once-{os.getpid()}")
    common.serialization.atomic_write_json(staging, payload)
    try:
        os.link(staging, destination)
    except FileExistsError as error:
        if json.loads(destination.read_text(encoding="utf-8")) != json.loads(staging.read_text(encoding="utf-8")):
            message = f"Write-once record already exists with other content: {destination}"
            raise FileExistsError(message) from error
        return False
    finally:
        staging.unlink(missing_ok=True)
    return True


def next_version(previous: Mapping[str, Any], added: list[dict[str, Any]], replacements: list[dict[str, Any]], *, version: int) -> dict[str, Any]:
    """Return the validated roster version that keeps every member of ``previous`` unchanged, appends ``added``, and accumulates replacements."""
    payload: dict[str, Any] = {
        **{key: value for key, value in previous.items() if key not in {"supersedes", "replacements", "roster_sha256"}},
        "members": [*previous["members"], *added],
        "supersedes": {"version": version - 1, "roster_sha256": previous["roster_sha256"]},
        "replacements": [*previous.get("replacements", []), *replacements],
    }
    payload["roster_sha256"] = roster_digest(payload)
    return validate_roster(json.loads(json.dumps(payload)))


def _planned_shape(member: Mapping[str, Any], prefix: str) -> str:
    """Return one member's planned structure without its base prefix: id suffix, role, links, row, level, kind, joint values."""
    links = [None if member[key] is None else str(member[key]).removeprefix(prefix) for key in ("gate_member_id", "reference_member_id")]
    joint = member["intervention"] if member["role"] == "joint" else None
    shape = [
        str(member["member_id"]).removeprefix(prefix),
        member["role"],
        *links,
        member["row"],
        member["level"],
        member.get("reference_kind"),
        joint,
    ]
    return json.dumps(shape, sort_keys=True)


def _require_extension(previous: Mapping[str, Any], later: Mapping[str, Any], *, version: int, directory: Path) -> None:
    """Refuse a version that breaks the chain or appends anything but its new replacement bases, each mirroring its failed base."""
    known, earlier = len(previous["members"]), list(previous.get("replacements", []))
    new = later["replacements"][len(earlier) :]
    _require(
        later.get("supersedes") == {"version": version, "roster_sha256": previous["roster_sha256"]}
        and later["members"][:known] == previous["members"]
        and later["replacements"][: len(earlier)] == earlier
        and bool(new)
        and all(later.get(key) == previous.get(key) for key in ("test_id", "task", "stage", "base_source", "max_attempts", "derived_from")),
        f"Roster version {version + 1} does not extend version {version} of {directory}.",
    )
    added = later["members"][known:]
    used = {(item["base"]["material_family"], item["base"]["case_id"]) for item in previous["members"]}
    replaced = {str(item["failed_gate_member_id"]) for item in earlier}
    claimed = 0
    for item in new:
        material = item["material_family"]
        gate = next((m for m in previous["members"] if m["member_id"] == item["failed_gate_member_id"] and m["role"] == "gate"), None)
        part = [m for m in added if m["base"]["material_family"] == material and m["base"]["case_id"] == item["replacement_case_id"]]
        failed = [] if gate is None else [m for m in previous["members"] if m["base"] == gate["base"]]
        _require(
            gate is not None
            and gate["member_id"] not in replaced
            and (gate["base"]["material_family"], gate["base"]["case_id"]) == (material, item["failed_case_id"])
            and (material, item["replacement_case_id"]) not in used
            and sorted(_planned_shape(m, f"{material}-{item['replacement_case_id']}-") for m in part)
            == sorted(_planned_shape(m, f"{material}-{item['failed_case_id']}-") for m in failed),
            f"Roster version {version + 1} of {directory}: replacement {item} does not mirror the failed base's planned members.",
        )
        replaced.add(str(item["failed_gate_member_id"]))
        claimed += len(part)
    _require(claimed == len(added), f"Roster version {version + 1} of {directory} appends members outside its replacement bases.")


def _version_path(directory: Path, version: int) -> Path:
    """Return the file of one roster version: roster.json for the first, roster.vN.json for later ones."""
    return directory / (ROSTER_FILENAME if version == 1 else f"roster.v{version}.json")


def write_roster(directory: Path, payload: Mapping[str, Any]) -> Path:
    """Publish one validated roster version once; a first version or the version after the current one, never a rewrite."""
    validate_roster(json.loads(json.dumps(payload)))
    version = 1
    if "supersedes" in payload:
        current, version = load_roster_version(directory)
        version += 1
        if payload["supersedes"] != {"version": version - 1, "roster_sha256": current["roster_sha256"]}:
            message = f"Roster version {version} does not supersede the current version of {directory}."
            raise ValueError(message)
    path = _version_path(directory, version)
    publish_once(path, payload)
    return path


def load_roster_version(directory: Path) -> tuple[dict[str, Any], int]:
    """
    Return the current roster and its version after checking the whole chain of versions.

    The version files must be regular files forming one consecutive chain (roster.json, roster.v2.json, ...) with no other
    roster.v*.json beside it. Each later version must supersede its predecessor by digest, keep every earlier member and
    replacement unchanged and in order, and append exactly the planned members of its new replacement bases.

    Raises
    ------
    FileNotFoundError
        If the field test has no regular roster.json.
    ValueError
        If any version is invalid, a symbolic link, outside the consecutive chain, or breaks the chain.

    """
    path = _version_path(directory, 1)
    if path.is_symlink() or not path.is_file():
        message = f"Field test has no regular roster.json: {directory}"
        raise FileNotFoundError(message)
    numbers: list[int] = []
    for candidate in sorted(directory.glob("roster.v*.json")):
        match = _VERSION_FILE.fullmatch(candidate.name)
        if match is None or candidate.name != f"roster.v{int(match['version'])}.json" or candidate.is_symlink() or not candidate.is_file():
            message = f"Roster version file is unsafe or outside the chain: {candidate}"
            raise ValueError(message)
        numbers.append(int(match["version"]))
    _require(sorted(numbers) == list(range(2, len(numbers) + 2)), f"Roster versions of {directory} are not one consecutive chain: {sorted(numbers)}")
    roster = validate_roster(json.loads(path.read_text(encoding="utf-8")))
    _require("supersedes" not in roster, f"The first roster version supersedes nothing: {path}")
    for version in range(1, len(numbers) + 1):
        later = validate_roster(json.loads(_version_path(directory, version + 1).read_text(encoding="utf-8")))
        _require_extension(roster, later, version=version, directory=directory)
        roster = later
    return roster, len(numbers) + 1


def load_roster(directory: Path) -> dict[str, Any]:
    """Load and validate the current frozen roster version of one field-test directory; a missing or changed roster fails closed."""
    return load_roster_version(directory)[0]


def member(roster: Mapping[str, Any], member_id: str) -> dict[str, Any]:
    """Return one roster member by id."""
    matches = [item for item in roster["members"] if item["member_id"] == member_id]
    if len(matches) != 1:
        message = f"Field test {roster['test_id']!r} has no member {member_id!r}."
        raise ValueError(message)
    return matches[0]


def source_directory(task: str, base_source: Mapping[str, Any], *, storage_root: Path | str | None) -> Path:
    """Return the base Source directory ``01_generation/TASK/SOURCE``."""
    return common.paths.resolve_generation_campaign_directory(task, str(base_source["campaign_name"]), storage_root=storage_root)


def plan_sha256(task: str, campaign_name: str, *, storage_root: Path | str | None) -> str:
    """Return the sha256 of one Source's frozen campaign_plan.json."""
    directory = common.paths.resolve_generation_campaign_directory(task, campaign_name, storage_root=storage_root)
    return common.serialization.file_sha256(common.paths.resolve_generation_campaign_plan_path(directory))


def load_base_campaign(task: str, base_source: Mapping[str, Any], *, storage_root: Path | str | None) -> config_contract.CampaignConfig:
    """
    Reconstruct the base Source's campaign from its frozen plan and run record, after checking the plan digest.

    Raises
    ------
    BaseBindingError
        If the frozen plan changed since the roster was planned.

    """
    if plan_sha256(task, str(base_source["campaign_name"]), storage_root=storage_root) != base_source["plan_sha256"]:
        message = f"Base Source plan of {base_source['campaign_name']!r} no longer matches the roster's plan_sha256."
        raise BaseBindingError(message)
    run_id = f"source~{task}~{base_source['campaign_name']}"
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    return lifecycle.campaign_from_frozen_run(manifest, storage_root=storage_root)


def case_batch(campaign: config_contract.CampaignConfig, material_family: str, design_kind: str, case_index: int) -> config_contract.GenerationConfig:
    """Return the unique frozen batch that holds one case of a material and design kind."""
    batches = [
        batch
        for batch in campaign.batches
        if batch.material_family == material_family and batch.sampling_regime == design_kind and case_index in batch.case_indices
    ]
    if len(batches) != 1:
        message = f"Case {case_index} of {material_family}/{design_kind} has no unique frozen batch."
        raise BaseBindingError(message)
    return batches[0]


def base_batch(campaign: config_contract.CampaignConfig, base: Mapping[str, Any]) -> tuple[config_contract.GenerationConfig, int]:
    """Return the frozen batch and case index of one base, after checking its design kind and design index."""
    case_index = int(str(base["case_id"]).removeprefix("case_"))
    config = case_batch(campaign, str(base["material_family"]), str(base["design_kind"]), case_index)
    if config.case_assignment(case_index)["design_index"] != base["design_index"]:
        message = f"Base {base['material_family']}/{base['case_id']} no longer has design index {base['design_index']}."
        raise BaseBindingError(message)
    return config, case_index


def base_inputs_directory(case_h5: Path) -> Path:
    """Return the published inputs directory (case.json and inputs.h5) beside one base case.h5."""
    return common.paths.resolve_generation_case_inputs_directory(case_h5.parent.parent)


def acceptance_attempts(case_payload: Mapping[str, Any]) -> tuple[int, int | None]:
    """Return the complete-case support attempt and the schedule acceptance attempt (None for steady) one case.json records."""
    spatial = int(case_payload["spatial_diagnostics"]["complete_case_support_retry"]["acceptance_attempt"])
    schedule = case_payload.get("schedule_diagnostics")
    return spatial, None if schedule is None else int(schedule["schedule_acceptance_attempt"])


def verified_base_payload(
    task: str, roster: Mapping[str, Any], base: Mapping[str, Any], *, storage_root: Path | str | None
) -> tuple[dict[str, Any], Path]:
    """
    Return the published base case.json payload and the base case.h5 path after verifying both digests.

    Raises
    ------
    BaseBindingError
        If a base file escapes its Source or no longer has the digest the roster binds.

    """
    source = source_directory(task, roster["base_source"], storage_root=storage_root).resolve()
    case_h5 = (source / str(base["case_path"])).resolve()
    case_json = base_inputs_directory(case_h5) / common.paths.GENERATION_CASE_PAYLOAD_FILENAME
    for path, expected in ((case_h5, base["case_sha256"]), (case_json, base["case_json_sha256"])):
        if not path.is_relative_to(source) or path.is_symlink() or not path.is_file() or common.serialization.file_sha256(path) != expected:
            message = f"Published base file is missing, unsafe, or changed: {path}"
            raise BaseBindingError(message)
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("case_id") != base["case_id"]:
        message = f"Published base case.json does not belong to {base['case_id']}: {case_json}"
        raise BaseBindingError(message)
    return payload, case_h5
