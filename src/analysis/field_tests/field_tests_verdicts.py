"""Turn the sealed members of one field test into one verdict per parameter row against the frozen tolerances.

Responsibilities:
  - Admit the effective roster (the newest verified version and verified replacements, through the Generation record),
    the member seals, the base binding to the published Source cases and the frozen table (stage A's in stage B)
  - Measure each base's replicate envelope: its gate's case.h5 against the published base case on every endpoint and
    every shared window end
  - Decide pairing validity: both seals succeeded, the base gate passed and both retry attempts equal the reference's
  - Measure every comparison with the endpoint code, reduce it to ratios against the frozen tolerances, and look each of
    its verdict endpoints up in its base's replicate envelope at the comparison's own hour or window end; after the
    base's last stored hour, carry the largest finite replicate ratio of the same quantity over the base's window
  - Apply the verdict rules: remove, keep, unresolved and, for stage-B constant-combination rows, merge; judge the
    stage-B joint removal test separately and let it gate every stage-B removal
  - Publish ``verdicts.json`` (rows, joint result, replicates and every comparison's measurements), ``verdicts.md`` and
    the summary figure, and admit ``verdicts.json`` for stage B

Design principles:
  - Fail closed: a missing or broken mandatory comparison, a missing value or tolerance, a censored duration within
    tolerance, a replicate difference missing or at or above the margin and an uncleared joint test all block removal;
    equality never clears a margin
  - Keep needs one valid exceedance or a solver failure that reproduced in two attempts while its reference succeeded
  - The tolerance table is admitted and hashed before any member is read, and a verdict file of another table is
    never overwritten; stage B is judged only against the stage-A table its roster binds
  - Coupled-stream rows are generator effects with reassigned downstream draws and need four or more bases to clear

This module does NOT:
  - Run, repair or reseal members, or rebuild the roster
  - Build tolerances or decide which rows stage B tests

Notes:
  Each ``verdicts.json`` comparison carries ``measured`` and ``report_only``: endpoint name to ``[value, contributors]``,
  the value ``"unavailable"`` when the endpoint has no finite value. Each replicate carries ``endpoints``: verdict endpoint
  name, every shared window end included, to ``[value, ratio]``, the ratio ``null`` without a finite value or tolerance.

"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src import common, generation

from . import field_tests_endpoints as endpoints
from . import field_tests_tolerances as tolerances

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

VERDICTS_SCHEMA_KIND: Final = "field_test_verdicts"
SCHEMA_VERSION: Final = 1
VERDICTS_FILENAME: Final = "verdicts.json"
VERDICTS: Final = ("remove", "keep", "unresolved", "merge")
REMOVE_MARGIN: Final = 0.5
KEEP_RATIO: Final = 1.0
COUPLED_STREAM_ROWS: Final = (
    "bed.perturbations.amplitude",
    "bed.perturbations.granularity",
    "bed.structure.coarse_len_rel",
    "schedule.event_count",
)
COUPLED_MIN_BASES: Final = 4
MERGE_MIN_DISPLACEMENTS: Final = 2
MERGE_MIN_BASES_PER_MATERIAL: Final = 2
MERGE_LEVEL: Final = "merge"
MERGE_ROW_PREFIX: Final = "merge:"
JOINT_NOT_CLEARED: Final = "joint removal test not cleared"
UNAVAILABLE: Final = "unavailable"
PROVISIONAL_INLET_ROWS: Final = ("airflow.superficial_velocity",)
PROVISIONAL_INLET_PREFIX: Final = "pressure_bc."
SCOPE: Final = "tested materials and today's template; ID supports only"
_COLOURS: Final = {"remove": "#2a9d55", "merge": "#3a6fb0", "keep": "#c8423b", "unresolved": "#9a9a9a"}
_DISPLAY_FLOOR: Final = 1.0e-3
_NOTE_NAMES: Final = 3
_MERGE_PARAMETERS: Final = 2
# A solver failure counts as an effect only after its identical-input rerun failed too; infrastructure stops never count.
_REPRODUCED_ATTEMPTS: Final = 2
# A challenge pairs with its reference only on the same accepted spatial support and schedule draw.
_PAIRED_ATTEMPTS: Final = ("spatial_support_attempt", "schedule_acceptance_attempt")


@dataclass(slots=True)
class _Comparison:
    """One mandatory comparison of a row or the joint test, reduced to its validity, ratios and persisted measurements."""

    member_id: str
    row: str | None
    role: str
    base: str
    material: str
    problem: str | None = None
    failure_effect: bool = False
    infeasible: bool = False
    worst_ratio: float | None = None
    worst_endpoint: str | None = None
    exceeds: bool = False
    blocked: list[str] = field(default_factory=list)
    censored_within: bool = False
    replicate_blocked: list[str] = field(default_factory=list)
    carried: list[float | None] = field(default_factory=list)
    cpu_models: set[str] = field(default_factory=set)
    measured: dict[str, list[Any]] = field(default_factory=dict)
    report_only: dict[str, list[Any]] = field(default_factory=dict)

    def clears(self) -> bool:
        """Return whether the comparison is valid and below the removal margin on every endpoint, replicate envelope included."""
        return (
            self.problem is None
            and not self.blocked
            and not self.censored_within
            and not self.replicate_blocked
            and self.worst_ratio is not None
            and self.worst_ratio < REMOVE_MARGIN
        )

    def record(self) -> dict[str, Any]:
        """Return the persisted evidence of the comparison."""
        return {
            "member_id": self.member_id,
            "row": self.row,
            "role": self.role,
            "base": self.base,
            "problem": self.problem,
            "worst_ratio": self.worst_ratio,
            "worst_endpoint": self.worst_endpoint,
            "measured": self.measured,
            "report_only": self.report_only,
        }


@dataclass(frozen=True, slots=True)
class _Envelope:
    """One base's replicate ratio per verdict endpoint, its last stored hour, and the largest finite ratio per hourly quantity."""

    ratios: Mapping[str, float | None]
    last_hour: float | None = None
    largest: Mapping[str, float] = field(default_factory=dict)


def base_label(base: Mapping[str, Any]) -> str:
    """Return the label of one base binding: material, design kind and design index."""
    return f"{base['material_family']}/{base['design_kind']}/{base['design_index']}"


def pairing_problem(challenge: Mapping[str, Any] | None, reference: Mapping[str, Any] | None, gate: Mapping[str, Any] | None) -> str | None:
    """
    Return why a challenge is not validly paired with its reference, or ``None`` for a valid pair.

    A valid pair has a successful gate whose regenerated inputs equal the published base, both seals ``succeeded``,
    and the challenge's spatial support and schedule acceptance attempts equal the reference's. Pairs may cross CPU
    models: regeneration differs there only in the last digits, far below the tolerances.
    """
    if challenge is None or reference is None or gate is None:
        return "unsealed"
    if gate.get("disposition") != "succeeded" or not (gate.get("gate") or {}).get("inputs_equal", False):
        return "gate_failed"
    if reference.get("disposition") != "succeeded":
        return f"reference_{reference.get('disposition')}"
    for name in _PAIRED_ATTEMPTS:
        if challenge.get(name) != reference.get(name):
            return f"{name}_differs"
    if challenge.get("disposition") != "succeeded":
        return f"challenge_{challenge.get('disposition')}"
    return None


def _ratio(name: str, endpoint: endpoints.Endpoint, limits: Mapping[str, float]) -> float | None:
    """Return an endpoint's value over its tolerance, or None without a finite value or a positive tolerance."""
    tolerance = limits.get(name)
    if endpoint.value is None or not math.isfinite(endpoint.value) or tolerance is None or not tolerance > 0.0:
        return None
    return endpoint.value / tolerance


def _persisted(endpoint: endpoints.Endpoint) -> list[Any]:
    """Return one endpoint as persisted: its value (``unavailable`` without a finite value) and its contributor count."""
    finite = endpoint.value is not None and math.isfinite(endpoint.value)
    return [endpoint.value if finite else UNAVAILABLE, endpoint.contributors]


def _replicate_ratio(envelope: _Envelope, name: str) -> tuple[float | None, bool]:
    """
    Return the replicate ratio a comparison endpoint is judged with, and whether it was carried from the base window.

    Within the base's stored window the ratio is the replicate's own under the same name. An hour or window end after the
    base's last stored hour has no replicate measurement: it carries the largest finite replicate ratio of the same
    quantity over the base's whole stored window, ``None`` when the base has none.
    """
    quantity, hour = endpoints.split_hour(name)
    if hour is None or envelope.last_hour is None or hour <= envelope.last_hour + endpoints.HOUR_ATOL:
        return envelope.ratios.get(name), False
    return envelope.largest.get(quantity), True


def _measure(
    comparison: _Comparison,
    reference: endpoints.CaseFields,
    other: endpoints.CaseFields,
    limits: Mapping[str, float],
    envelope: _Envelope,
) -> None:
    """
    Persist every endpoint of one valid comparison and reduce its verdict endpoints to the worst ratio and blocked endpoints.

    A verdict endpoint without a ratio is blocked; a measured one whose replicate ratio (``_replicate_ratio``) is missing
    or reaches the removal margin is replicate-blocked.
    """
    for name, endpoint in endpoints.endpoints(reference, other).items():
        (comparison.measured if endpoint.verdict else comparison.report_only)[name] = _persisted(endpoint)
        if not endpoint.verdict:
            continue
        ratio = _ratio(name, endpoint, limits)
        if ratio is None:
            comparison.blocked.append(name)
            continue
        replicate, carried = _replicate_ratio(envelope, name)
        if carried:
            comparison.carried.append(replicate)
        if replicate is None or replicate >= REMOVE_MARGIN:
            comparison.replicate_blocked.append(name)
        if comparison.worst_ratio is None or ratio > comparison.worst_ratio:
            comparison.worst_ratio, comparison.worst_endpoint = ratio, name
        if ratio > KEEP_RATIO:
            comparison.exceeds = True
        elif endpoint.censored:
            comparison.censored_within = True


def _verified_case(path: Path, expected_sha256: object) -> endpoints.CaseFields | None:
    """Read a ``case.h5`` only when its digest equals the bound one."""
    if not path.is_file() or common.serialization.file_sha256(path) != expected_sha256:
        return None
    return endpoints.read_case_fields(path)


def _replicates(
    directory: Path, members: Sequence[Mapping[str, Any]], seals: Mapping[str, Any], limits: Mapping[str, float], *, source_root: Path
) -> tuple[list[dict[str, Any]], dict[str, _Envelope], dict[str, bool]]:
    """
    Measure each base's gate case against its published base case on the envelope endpoints.

    Returns the persisted replicate per base, the replicate envelope per base (ratios ``None`` without a finite value or
    tolerance; a base whose gate case cannot be read has no envelope), and whether each base's published case still has
    its bound digest.
    """
    records: list[dict[str, Any]] = []
    envelopes: dict[str, _Envelope] = {}
    base_ok: dict[str, bool] = {}
    for gate in (member for member in members if member["role"] == "gate"):
        label = base_label(gate["base"])
        published = source_root / gate["base"]["case_path"]
        base_ok[label] = published.is_file() and common.serialization.file_sha256(published) == gate["base"]["case_sha256"]
        seal = seals.get(gate["member_id"])
        entry: dict[str, Any] = {"base": label, "gate_member_id": gate["member_id"], "problem": None, "endpoints": {}}
        records.append(entry)
        if not base_ok[label] or seal is None or not generation.field_tests.record.gate_succeeded(seal):
            entry["problem"] = "gate_or_base_invalid"
            continue
        member_case = generation.field_tests.record.member_directory(directory, gate["member_id"]) / "case.h5"
        replica = _verified_case(member_case, seal.get("case_h5_sha256"))
        if replica is None:
            # Without the replicate the envelope is unknown, so no removal may rest on this base.
            entry["problem"] = "replicate_unavailable"
            continue
        base_case = endpoints.read_case_fields(published)
        ratios: dict[str, float | None] = {}
        largest: dict[str, float] = {}
        for name, endpoint in endpoints.envelope_endpoints(base_case, replica).items():
            if endpoint.verdict:
                ratio = ratios[name] = _ratio(name, endpoint, limits)
                entry["endpoints"][name] = [_persisted(endpoint)[0], ratio]
                quantity, hour = endpoints.split_hour(name)
                if hour is not None and ratio is not None:
                    largest[quantity] = max(ratio, largest.get(quantity, ratio))
        last_hour = None if base_case.times is None or not base_case.times.size else float(base_case.times[-1])
        envelopes[label] = _Envelope(ratios, last_hour, largest)
    return records, envelopes, base_ok


def _solver_failures(directory: Path, member_id: str, reference: Mapping[str, Any]) -> int:
    """Return how many recorded attempts of one member ended as a solver failure on the reference's accepted draws."""
    record = generation.field_tests.record
    outcomes = (record.read_outcome(directory, member_id, number) for number in record.attempt_numbers(directory, member_id))
    return sum(
        1
        for outcome in outcomes
        if outcome is not None
        and outcome["classification"] == "solver_failure"
        and all(outcome[name] == reference.get(name) for name in _PAIRED_ATTEMPTS)
    )


def _comparisons(
    directory: Path,
    members: Sequence[Mapping[str, Any]],
    seals: Mapping[str, Any],
    limits: Mapping[str, float],
    *,
    base_ok: Mapping[str, bool],
    envelopes: Mapping[str, _Envelope],
) -> list[_Comparison]:
    """Validate and measure every challenge and joint member, one reference case in memory at a time."""
    tested = sorted(
        (member for member in members if member["role"] in {"challenge", "joint"}),
        key=lambda member: (str(member["reference_member_id"]), member["member_id"]),
    )
    result: list[_Comparison] = []
    cached: tuple[str, endpoints.CaseFields | None] | None = None
    for member in tested:
        reference_id = str(member["reference_member_id"])
        label = base_label(member["base"])
        comparison = _Comparison(
            member_id=member["member_id"], row=member["row"], role=member["role"], base=label, material=str(member["base"]["material_family"])
        )
        seal, reference_seal, gate_seal = seals.get(member["member_id"]), seals.get(reference_id), seals.get(str(member["gate_member_id"]))
        comparison.cpu_models = {str(item["cpu_model"]) for item in (seal, reference_seal, gate_seal) if item and item.get("cpu_model")}
        comparison.problem = "source_base_changed" if not base_ok.get(label, False) else pairing_problem(seal, reference_seal, gate_seal)
        if seal is not None and seal.get("disposition") == "infeasible":
            comparison.problem, comparison.infeasible = "infeasible", True
        elif (
            comparison.problem == "challenge_solver_failure"
            and reference_seal is not None
            and _solver_failures(directory, member["member_id"], reference_seal) >= _REPRODUCED_ATTEMPTS
        ):
            comparison.failure_effect = True
        elif comparison.problem is None and seal is not None and reference_seal is not None:
            if cached is None or cached[0] != reference_id:
                reference_case = generation.field_tests.record.member_directory(directory, reference_id) / "case.h5"
                cached = (reference_id, _verified_case(reference_case, reference_seal["case_h5_sha256"]))
            other_case = generation.field_tests.record.member_directory(directory, member["member_id"]) / "case.h5"
            other = _verified_case(other_case, seal["case_h5_sha256"])
            if cached[1] is None or other is None:
                comparison.problem = "case_h5_digest_mismatch"
            else:
                _measure(comparison, cached[1], other, limits, envelopes.get(label, _Envelope(ratios={})))
        result.append(comparison)
    return result


def _row_notes(row: str, comparisons: Sequence[_Comparison]) -> list[str]:
    """Return the coupled-stream, provisional, replicate and evidence notes of one row."""
    notes: list[str] = []
    if row in COUPLED_STREAM_ROWS:
        notes.append("coupled-stream row: differences are generator effects including reassigned downstream draws")
    if row in PROVISIONAL_INLET_ROWS or row.startswith(PROVISIONAL_INLET_PREFIX):
        notes.append("inlet row: provisional until the template-63 rerun")
    notes.extend(f"infeasible contrast: {item.member_id}" for item in comparisons if item.infeasible)
    broken = [item for item in comparisons if item.problem and not item.infeasible and not item.failure_effect]
    notes.extend(f"broken comparison {item.member_id}: {item.problem}" for item in broken)
    notes.extend(f"reproduced solver failure with a successful reference: {item.member_id}" for item in comparisons if item.failure_effect)
    notes.extend(f"censored duration bound within tolerance: {item.member_id}" for item in comparisons if item.censored_within)
    cpu_models = sorted(set().union(*(item.cpu_models for item in comparisons)))
    notes.append(f"cpu models (evidence only): {', '.join(cpu_models) if cpu_models else 'unrecorded'}")
    for item in comparisons:
        if item.carried:
            finite = [ratio for ratio in item.carried if ratio is not None]
            shown = f"{max(finite):.3g}" if finite else UNAVAILABLE
            notes.append(f"replicate envelope carried from the base window (max ratio {shown}): {item.member_id}")
        if item.replicate_blocked:
            shown = ", ".join(item.replicate_blocked[:_NOTE_NAMES])
            notes.append(
                f"replicate difference at {item.base} missing or at least {REMOVE_MARGIN} x tolerance for {item.member_id}: "
                f"{len(item.replicate_blocked)} endpoints ({shown})"
            )
        if item.blocked:
            shown = ", ".join(item.blocked[:_NOTE_NAMES])
            notes.append(f"missing value or tolerance at {item.member_id}: {len(item.blocked)} endpoints ({shown})")
    return notes


def _merge_rows(members: Sequence[Mapping[str, Any]]) -> set[str]:
    """
    Return the merge rows of a roster and refuse a malformed merge confirmation member.

    A merge member is a ``challenge`` at level ``merge`` whose row is ``merge:<a>+<b>`` with sorted parameter names and
    whose intervention equals its reference's intervention except for exactly those two parameters.
    """
    by_id = {member["member_id"]: member for member in members}
    rows: set[str] = set()
    for member in members:
        if member["level"] != MERGE_LEVEL:
            continue
        row = str(member["row"])
        parameters = row.removeprefix(MERGE_ROW_PREFIX).split("+")
        reference = by_id.get(str(member.get("reference_member_id")))
        moved, base = member["intervention"], {} if reference is None else reference["intervention"]
        if (
            member["role"] != "challenge"
            or reference is None
            or not row.startswith(MERGE_ROW_PREFIX)
            or len(parameters) != _MERGE_PARAMETERS
            or parameters != sorted(parameters)
            or {name for name in set(moved) | set(base) if moved.get(name) != base.get(name)} != set(parameters)
        ):
            msg = f"Merge member {member['member_id']!r} does not follow the merge:<a>+<b> confirmation convention."
            raise ValueError(msg)
        rows.add(row)
    return rows


def _merge_clears(comparisons: Sequence[_Comparison], materials: set[str]) -> bool:
    """Return whether every merge base has two or more clearing displacements and every material two or more such bases."""
    clearing = Counter(item.base for item in comparisons if item.clears())
    bases = {item.base: item.material for item in comparisons}
    per_material = Counter(bases.values())
    return all(clearing[base] >= MERGE_MIN_DISPLACEMENTS for base in bases) and all(
        per_material[material] >= MERGE_MIN_BASES_PER_MATERIAL for material in materials
    )


def _row_verdict(
    row: str, stage: str, comparisons: Sequence[_Comparison], *, merge_row: bool, materials: set[str], joint_cleared: bool
) -> dict[str, Any]:
    """Apply the verdict rules to the comparisons of one row and return its ``verdicts.json`` entry."""
    valid = [item for item in comparisons if item.problem is None]
    broken = [item for item in comparisons if item.problem is not None and not item.infeasible and not item.failure_effect]
    measured = [item for item in valid if item.worst_ratio is not None]
    worst = max(measured, key=lambda item: float(item.worst_ratio or 0.0), default=None)
    bases = {item.base for item in valid}
    clears = (
        bool(valid) and not broken and all(item.clears() for item in valid) and (row not in COUPLED_STREAM_ROWS or len(bases) >= COUPLED_MIN_BASES)
    )
    notes = _row_notes(row, comparisons)
    if any(item.exceeds for item in valid) or any(item.failure_effect for item in comparisons):
        verdict = "keep"
    elif merge_row:
        verdict = "merge" if clears and _merge_clears(comparisons, materials) else "unresolved"
    else:
        verdict = "remove" if clears else "unresolved"
    if row in COUPLED_STREAM_ROWS and len(bases) < COUPLED_MIN_BASES:
        notes.append(f"removal needs the margin at {COUPLED_MIN_BASES} or more bases, valid at {len(bases)}")
    if verdict == "remove" and stage == "B" and not joint_cleared:
        verdict = "unresolved"
        notes.append(JOINT_NOT_CLEARED)
    if verdict == "remove" and stage == "A":
        notes.append("screening verdict; a removal is recommended only after stage B confirms it")
    return {
        "row": row,
        "verdict": verdict,
        "worst_ratio": None if worst is None else worst.worst_ratio,
        "worst_at": None if worst is None else {"member_id": worst.member_id, "endpoint": worst.worst_endpoint, "base": worst.base},
        "valid_comparisons": len(valid),
        "missing_or_broken": len(broken),
        "notes": notes,
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the human verdict table."""
    joint = payload["joint"]
    lines = [
        f"# Field-test verdicts: {payload['test_id']} (stage {payload['stage']}, {payload['task']})",
        "",
        f"Scope: {payload['scope']}. Tolerances `{payload['tolerances_sha256'][:12]}`, roster `{payload['roster_sha256'][:12]}`.",
        f"Rules: remove below {REMOVE_MARGIN} x tolerance everywhere with every comparison valid; keep above {KEEP_RATIO} x anywhere.",
        *(
            []
            if joint is None
            else [f"Joint removal test: {'cleared' if joint['cleared'] else JOINT_NOT_CLEARED} ({len(joint['member_ids'])} members)."]
        ),
        "",
        "| Row | Verdict | Worst ratio | Where | Valid | Missing or broken | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        ratio = "n/a" if row["worst_ratio"] is None else f"{row['worst_ratio']:.3g}"
        where = "" if row["worst_at"] is None else f"{row['worst_at']['endpoint']} ({row['worst_at']['base']})"
        cells = (row["row"], row["verdict"], ratio, where, row["valid_comparisons"], row["missing_or_broken"], "; ".join(row["notes"]))
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    common.serialization.atomic_write_text(path, "\n".join(lines) + "\n")


def _write_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Draw rows by worst ratio, coloured by verdict, with the removal margin and the keep line."""
    ordered = sorted(rows, key=lambda row: -1.0 if row["worst_ratio"] is None else float(row["worst_ratio"]))
    figure = Figure(figsize=(8.0, max(2.5, 0.28 * len(ordered) + 1.2)), layout="constrained")
    axis = figure.add_subplot()
    values = [_DISPLAY_FLOOR if row["worst_ratio"] is None else max(float(row["worst_ratio"]), _DISPLAY_FLOOR) for row in ordered]
    axis.barh([row["row"] for row in ordered], values, color=[_COLOURS[row["verdict"]] for row in ordered])
    axis.axvline(REMOVE_MARGIN, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(KEEP_RATIO, color="black", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("worst |difference| / tolerance (no ratio drawn at the floor)")
    handles = [Patch(color=colour, label=verdict) for verdict, colour in _COLOURS.items()]
    axis.legend(handles=handles, loc="lower right", fontsize="small")
    common.serialization.atomic_path_write(path, lambda target: figure.savefig(target, format="png", dpi=150))


def _admitted_tolerances(directory: Path, roster: Mapping[str, Any], *, storage_root: Path | str | None) -> tolerances.FrozenTolerances:
    """
    Admit the frozen table a field test is judged against: its own in stage A, stage A's in stage B.

    Raises
    ------
    ValueError
        If stage A's table is not the one the stage-B roster binds.

    """
    if roster["stage"] == "A":
        return tolerances.load_frozen_tolerances(generation.field_tests.record.record_path(directory, "analysis", tolerances.TOLERANCES_FILENAME))
    origin = roster["derived_from"]
    stage_a = common.paths.resolve_field_test_dir(roster["task"], str(origin["test_id"]), storage_root=storage_root)
    frozen = tolerances.load_frozen_tolerances(generation.field_tests.record.record_path(stage_a, "analysis", tolerances.TOLERANCES_FILENAME))
    if frozen.sha256 != origin.get("tolerances_sha256"):
        msg = f"Stage A's frozen tolerance table of {origin['test_id']!r} is not the table this stage-B roster binds."
        raise ValueError(msg)
    return frozen


def build_verdicts(task: str, test_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """
    Build and publish the verdicts of one field test from its frozen tolerances, roster, seals and members.

    Raises
    ------
    FileNotFoundError
        If the tolerance table has not been frozen.
    ValueError
        If the tolerance table or roster changed, a replacement is not backed by its failed gate, the tasks differ, a
        stage-B table is not stage A's, or an existing ``verdicts.json`` binds another tolerance table.

    """
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    roster, version = generation.field_tests.roster.load_roster_version(directory)
    frozen = _admitted_tolerances(directory, roster, storage_root=storage_root)
    analysis = generation.field_tests.record.record_path(directory, "analysis")
    existing = analysis / VERDICTS_FILENAME
    if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")).get("tolerances_sha256") != frozen.sha256:
        msg = f"{existing} was built on another tolerance table; the frozen table must not change."
        raise ValueError(msg)
    if roster["task"] != frozen.payload["task"]:
        msg = f"Tolerances of {frozen.payload['task']} cannot judge a {roster['task']} field test."
        raise ValueError(msg)
    members = generation.field_tests.record.effective_members(directory, roster)
    seals = {member["member_id"]: generation.field_tests.record.read_seal(directory, member["member_id"]) for member in members}
    source_root = common.paths.resolve_generation_campaign_directory(task, roster["base_source"]["campaign_name"], storage_root=storage_root)
    replicates, envelopes, base_ok = _replicates(directory, members, seals, frozen.tolerances, source_root=source_root)
    comparisons = _comparisons(directory, members, seals, frozen.tolerances, base_ok=base_ok, envelopes=envelopes)
    joints = [item for item in comparisons if item.role == "joint"]
    joint_cleared = bool(joints) and all(item.clears() for item in joints)
    by_row: dict[str, list[_Comparison]] = {}
    for item in comparisons:
        if item.role != "joint":
            by_row.setdefault(str(item.row), []).append(item)
    merge_rows = _merge_rows(members)
    materials = {str(member["base"]["material_family"]) for member in members}
    payload = {
        "schema_kind": VERDICTS_SCHEMA_KIND,
        "schema_version": SCHEMA_VERSION,
        "test_id": roster["test_id"],
        "task": roster["task"],
        "stage": roster["stage"],
        "scope": SCOPE,
        "roster_sha256": roster["roster_sha256"],
        "roster_version": version,
        "tolerances_sha256": frozen.sha256,
        "rows": [
            _row_verdict(
                row,
                roster["stage"],
                items,
                merge_row=roster["stage"] == "B" and row in merge_rows,
                materials=materials,
                joint_cleared=joint_cleared,
            )
            for row, items in by_row.items()
        ],
        "joint": {"cleared": joint_cleared, "member_ids": [item.member_id for item in joints]} if joints else None,
        "replicates": replicates,
        "comparisons": [item.record() for item in comparisons],
    }
    validate_verdicts(payload)
    common.serialization.atomic_write_json(existing, payload)
    _write_markdown(analysis / "verdicts.md", payload)
    _write_figure(analysis / "verdicts.png", payload["rows"])
    return payload


def validate_verdicts(payload: Mapping[str, Any]) -> None:
    """Refuse a verdict payload that does not follow the interface schema and its documented additions."""
    if payload.get("schema_kind") != VERDICTS_SCHEMA_KIND or payload.get("schema_version") != SCHEMA_VERSION:
        msg = "Not a field-test verdict payload of the supported schema version."
        raise ValueError(msg)
    for key in ("test_id", "roster_sha256", "tolerances_sha256"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            msg = f"Verdict payload lacks {key}."
            raise ValueError(msg)
    version = payload.get("roster_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        msg = "Verdict payload lacks the roster version it judged."
        raise ValueError(msg)
    joint = payload.get("joint")
    if not (joint is None or (isinstance(joint, dict) and isinstance(joint.get("cleared"), bool) and isinstance(joint.get("member_ids"), list))):
        msg = "Verdict payload's joint removal test result is malformed."
        raise ValueError(msg)
    if not all(isinstance(payload.get(key), list) and all(isinstance(item, dict) for item in payload[key]) for key in ("replicates", "comparisons")):
        msg = "Verdict payload lacks its replicates or comparisons."
        raise ValueError(msg)
    for row in payload.get("rows", ()):
        ratio = row.get("worst_ratio")
        if (
            not isinstance(row.get("row"), str)
            or row.get("verdict") not in VERDICTS
            or not (ratio is None or (isinstance(ratio, float) and math.isfinite(ratio)))
            or not (row.get("worst_at") is None or set(row["worst_at"]) == {"member_id", "endpoint", "base"})
            or not all(isinstance(row.get(key), int) and row[key] >= 0 for key in ("valid_comparisons", "missing_or_broken"))
            or not isinstance(row.get("notes"), list)
        ):
            msg = f"Verdict row {row.get('row')!r} does not follow the interface schema."
            raise ValueError(msg)


def load_verdicts(task: str, test_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Admit the published ``verdicts.json`` of one field test, for example stage A's for the stage-B roster builder."""
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    payload = json.loads(generation.field_tests.record.record_path(directory, "analysis", VERDICTS_FILENAME).read_text(encoding="utf-8"))
    validate_verdicts(payload)
    return payload


def screening_removals(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the rows a stage-A verdict payload calls ``remove``, the only rows stage B may confirm."""
    validate_verdicts(payload)
    if payload.get("stage") != "A":
        msg = "Only stage-A verdicts select the rows stage B confirms."
        raise ValueError(msg)
    return tuple(row["row"] for row in payload["rows"] if row["verdict"] == "remove")
