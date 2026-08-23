# ruff: noqa: TRY003, EM101, EM102, PLC0415
"""
generation_campaign_completion.py

Persist deterministic supplemental replacement planning for partial campaigns.

Responsibilities:
  - Bind completion state to immutable parent evidence and a stable identity
  - Allocate and materialize bounded prefix-stable supplemental candidates
  - Advance one-case replacement campaigns through the normal campaign feeder
  - Construct exact composite membership from independently admitted successes

Design principles:
  - Parent successes always satisfy targets before supplemental candidates
  - Requested pool capacity is monotonic and never silently discarded
  - Persisted state is sufficient to resume controller reconciliation

This module does NOT:
  - Transfer artifacts, build Dataset payloads, or authorize source cleanup
  - Rewrite parent campaign manifests or reinterpret failed original cases
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src import common

from .cases import generation_cases_config as config_service
from .cases import generation_cases_sampling as sampling_service
from .cases import generation_cases_seeding as seeding_service
from .contracts import generation_contracts_source as source_service

_SCHEMA_VERSION = 1
_OWNER = "generation_campaign_completion.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESERVING_STATES = frozenset({"planned", "active", "retry", "replay"})


@dataclass(frozen=True, slots=True)
class CompletionTarget:
    """Exact successful-case target for one parent batch."""

    batch_id: str
    target_successes: int


@dataclass(frozen=True, slots=True)
class ReplacementCandidate:
    """One immutable supplemental candidate identity."""

    ordinal: int
    batch_ordinal: int
    batch_id: str
    parent_failed_case_id: str
    candidate_id: str
    science_id: str
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplacementWave:
    """One ordinary one-case synthetic campaign allocation."""

    wave_ordinal: int
    candidate_ids: tuple[str, ...]
    state: str


def _positive_int(value: object, *, label: str) -> int:
    """Require one non-boolean strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    """Require one non-boolean nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}.")
    return value


def _canonical(value: Mapping[str, Any]) -> str:
    """Return the repository canonical digest for one JSON mapping."""
    return common.serialization.canonical_json_sha256(dict(value))


def completion_id(*, parent_run_id: str, parent_partial_sha256: str) -> str:
    """Derive the stable completion identifier without pool capacity."""
    if not parent_run_id or _SHA256.fullmatch(parent_partial_sha256) is None:
        raise ValueError("Completion identity requires a parent run ID and lowercase SHA-256 partial-evidence digest.")
    digest = _canonical({"owner": _OWNER, "parent_run_id": parent_run_id, "parent_partial_sha256": parent_partial_sha256})
    return f"completion__{digest[:24]}"


def completion_directory(completion_id_value: str, *, storage_root: Path | str | None = None) -> Path:
    """Return the dedicated completion owner directory."""
    if not completion_id_value.startswith("completion__"):
        raise ValueError("Completion ID has the wrong namespace.")
    return common.paths.get_generation_meta_root(storage_root=storage_root) / "completions" / completion_id_value


def _state_path(completion_id_value: str, *, storage_root: Path | str | None) -> Path:
    return completion_directory(completion_id_value, storage_root=storage_root) / "completion.json"


def _targets_payload(targets: Sequence[CompletionTarget]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    payload: list[dict[str, Any]] = []
    for target in targets:
        if not target.batch_id or target.batch_id in seen:
            raise ValueError("Completion targets must have unique non-empty batch IDs.")
        seen.add(target.batch_id)
        payload.append({"batch_id": target.batch_id, "target_successes": _positive_int(target.target_successes, label="target_successes")})
    if not payload:
        raise ValueError("Completion requires at least one batch target.")
    return payload


def _validate_completion_state(  # noqa: C901, PLR0912, PLR0915 -- centralized durable-state integrity gate
    state: Mapping[str, Any],
    *,
    expected_completion_id: str | None = None,
) -> None:
    """Validate exact durable completion identity, membership, and wave accounting."""
    required = {
        "schema_version",
        "owner",
        "completion_id",
        "parent_run_id",
        "parent_partial_sha256",
        "parent_partial_evidence",
        "targets",
        "failure_policy",
        "failure_circuit",
        "pool_high_water",
        "candidates",
        "waves",
    }
    if set(state) != required or state.get("schema_version") != _SCHEMA_VERSION or state.get("owner") != _OWNER:
        raise ValueError("Completion state has an unsupported schema or shape.")
    identifier = common.paths.validate_logical_name(state.get("completion_id"), label="completion_id")
    if expected_completion_id is not None and identifier != expected_completion_id:
        raise ValueError("Completion state identifier differs from its storage owner.")
    parent_run_id = common.paths.validate_logical_name(state.get("parent_run_id"), label="parent_run_id")
    partial_sha256 = str(state.get("parent_partial_sha256", ""))
    if _SHA256.fullmatch(partial_sha256) is None or identifier != completion_id(
        parent_run_id=parent_run_id,
        parent_partial_sha256=partial_sha256,
    ):
        raise ValueError("Completion state parent identity evidence is invalid.")
    if not isinstance(state.get("parent_partial_evidence"), dict):
        raise TypeError("Completion state parent partial evidence must be one object.")
    targets_value = state.get("targets")
    if not isinstance(targets_value, list):
        raise TypeError("Completion state targets must be a list.")
    target_specs: list[CompletionTarget] = []
    for item in targets_value:
        if not isinstance(item, dict) or set(item) != {"batch_id", "target_successes"}:
            raise ValueError("Completion state targets are malformed or non-canonical.")
        target_specs.append(
            CompletionTarget(
                str(item["batch_id"]),
                _positive_int(
                    item["target_successes"],
                    label="target_successes",
                ),
            )
        )
    targets = _targets_payload(tuple(target_specs))
    if targets != targets_value:
        raise ValueError("Completion state targets are malformed or non-canonical.")
    target_ids = {item["batch_id"] for item in targets}
    failure_policy = state.get("failure_policy")
    failure_circuit = state.get("failure_circuit")
    if (
        not isinstance(failure_policy, dict)
        or set(failure_policy) != {"maximum_failed_cases", "counted_failure_classes"}
        or failure_policy.get("counted_failure_classes") != ["solver_failed", "technical_runtime_timed_out"]
        or not isinstance(failure_circuit, dict)
        or set(failure_circuit) != {"counted_failures", "open"}
        or not isinstance(failure_circuit.get("open"), bool)
    ):
        raise ValueError("Completion-local failure policy or circuit has an unsupported shape.")
    maximum_failed_cases = _nonnegative_int(
        failure_policy.get("maximum_failed_cases"),
        label="completion maximum_failed_cases",
    )
    pool = _positive_int(state.get("pool_high_water"), label="pool_high_water")
    candidates = state.get("candidates")
    waves = state.get("waves")
    if not isinstance(candidates, list) or not isinstance(waves, list) or len(candidates) > pool:
        raise ValueError("Completion state candidates, waves, or pool accounting is invalid.")
    candidate_ids: list[str] = []
    batch_counts = dict.fromkeys(target_ids, 0)
    allowed_states = {"planned", "active", "retry", "replay", "successful", "failed"}
    required_candidate = {
        "ordinal",
        "batch_ordinal",
        "batch_id",
        "parent_failed_case_id",
        "candidate_id",
        "science_id",
        "provenance",
        "state",
    }
    required_provenance = {
        "owner",
        "parent_run_id",
        "parent_partial_sha256",
        "parent_failed_case_id",
        "batch_id",
        "candidate_ordinal",
        "batch_ordinal",
        "sampling_algorithm",
        "sampling_version",
        "sampling_seed",
        "block_seeds",
        "unit_point",
        "unit_point_sha256",
    }
    for expected_ordinal, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise TypeError("Completion candidate must be one object.")
        candidate_keys = set(candidate)
        allowed_candidate_shapes = (
            required_candidate,
            required_candidate | {"materialization"},
            required_candidate | {"materialization", "terminal_evidence"},
        )
        if candidate_keys not in allowed_candidate_shapes:
            raise ValueError("Completion candidate has an unsupported shape.")
        ordinal = _positive_int(candidate.get("ordinal"), label="candidate ordinal")
        batch_ordinal = _positive_int(candidate.get("batch_ordinal"), label="candidate batch ordinal")
        batch_id = str(candidate.get("batch_id"))
        if ordinal != expected_ordinal or batch_id not in target_ids:
            raise ValueError("Completion candidate order or target batch is invalid.")
        batch_counts[batch_id] += 1
        if batch_ordinal != batch_counts[batch_id]:
            raise ValueError("Completion candidate batch-local prefix is not stable.")
        provenance = candidate.get("provenance")
        if not isinstance(provenance, dict) or set(provenance) != required_provenance:
            raise ValueError("Completion candidate provenance has an unsupported shape.")
        point = provenance.get("unit_point")
        block_seeds = provenance.get("block_seeds")
        sampling_seed = provenance.get("sampling_seed")
        if (
            provenance.get("owner") != _OWNER
            or provenance.get("parent_run_id") != parent_run_id
            or provenance.get("parent_partial_sha256") != partial_sha256
            or provenance.get("parent_failed_case_id") != candidate.get("parent_failed_case_id")
            or provenance.get("batch_id") != batch_id
            or provenance.get("candidate_ordinal") != ordinal
            or provenance.get("batch_ordinal") != batch_ordinal
            or provenance.get("sampling_algorithm") != "sobol"
            or provenance.get("sampling_version") != 1
            or isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or not isinstance(point, dict)
            or not point
            or not isinstance(block_seeds, dict)
            or set(block_seeds) != set(point)
            or provenance.get("unit_point_sha256") != _canonical({"unit_point": point})
        ):
            raise ValueError("Completion candidate provenance identity is invalid.")
        digest = _canonical(provenance)
        candidate_id = f"replacement__{digest[:24]}"
        science_id = f"supplemental__{digest[:24]}"
        if candidate.get("candidate_id") != candidate_id or candidate.get("science_id") != science_id or candidate.get("state") not in allowed_states:
            raise ValueError("Completion candidate identity or state is invalid.")
        candidate_ids.append(candidate_id)
        materialization = candidate.get("materialization")
        if materialization is not None:
            materialization_keys = {
                "campaign_run_id",
                "extension",
                "batch_id",
                "case_id",
                "case_index",
                "simulation_science_id",
                "git_commit",
            }
            if not isinstance(materialization, dict) or set(materialization) != materialization_keys:
                raise ValueError("Completion candidate materialization has an unsupported shape.")
            common.paths.validate_logical_name(materialization.get("campaign_run_id"), label="replacement campaign_run_id")
            common.paths.validate_logical_name(materialization.get("batch_id"), label="replacement batch_id")
            common.paths.validate_logical_name(materialization.get("case_id"), label="replacement case_id")
            _positive_int(materialization.get("case_index"), label="replacement case_index")
            source_service.validate_git_commit(materialization.get("git_commit"))
            extension = validate_synthetic_manifest_extension(materialization.get("extension"))
            design = extension.get("design")
            if (
                materialization.get("simulation_science_id") != science_id
                or not isinstance(design, dict)
                or design.get("completion_id") != identifier
                or design.get("candidate_id") != candidate_id
                or extension.get("synthetic_batch_id") != materialization.get("batch_id")
                or extension.get("synthetic_case_index") != materialization.get("case_index")
            ):
                raise ValueError("Completion candidate materialization identity is invalid.")
        terminal_evidence = candidate.get("terminal_evidence")
        terminal_state = candidate.get("state") in {"successful", "failed"}
        if terminal_state != (terminal_evidence is not None):
            raise ValueError("Completion candidate terminal state and execution evidence disagree.")
        if terminal_evidence is not None:
            terminal_keys = {
                "campaign_run_manifest_sha256",
                "git_commit",
                "failure_counts",
                "failure_circuit_contribution",
            }
            failure_counts = terminal_evidence.get("failure_counts") if isinstance(terminal_evidence, dict) else None
            if (
                not isinstance(materialization, dict)
                or not isinstance(terminal_evidence, dict)
                or set(terminal_evidence) != terminal_keys
                or _SHA256.fullmatch(str(terminal_evidence.get("campaign_run_manifest_sha256", ""))) is None
                or source_service.validate_git_commit(terminal_evidence.get("git_commit")) != materialization["git_commit"]
                or not isinstance(failure_counts, dict)
                or set(failure_counts) != {"solver_failed", "technical_runtime_timed_out"}
            ):
                raise ValueError("Completion candidate terminal execution evidence is invalid.")
            solver_failed = _nonnegative_int(failure_counts.get("solver_failed"), label="replacement solver_failed")
            timed_out = _nonnegative_int(
                failure_counts.get("technical_runtime_timed_out"),
                label="replacement technical_runtime_timed_out",
            )
            contribution = _nonnegative_int(
                terminal_evidence.get("failure_circuit_contribution"),
                label="replacement failure_circuit_contribution",
            )
            if contribution != solver_failed + timed_out or contribution > 1:
                raise ValueError("One-case replacement failure-circuit accounting is invalid.")
            if candidate.get("state") == "successful" and contribution != 0:
                raise ValueError("Successful replacement cannot contribute to the failure circuit.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise FileExistsError("Completion state contains duplicate candidate identities.")
    wave_candidate_ids: list[str] = []
    by_id = {str(item["candidate_id"]): item for item in candidates}
    for expected_wave, wave in enumerate(waves, start=1):
        if (
            not isinstance(wave, dict)
            or set(wave) != {"wave_ordinal", "candidate_ids", "state"}
            or wave.get("wave_ordinal") != expected_wave
            or wave.get("state") not in {"active", "terminal"}
            or not isinstance(wave.get("candidate_ids"), list)
            or not wave["candidate_ids"]
        ):
            raise ValueError("Completion wave has an unsupported shape or order.")
        ids = [str(value) for value in wave["candidate_ids"]]
        if any(value not in by_id for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Completion wave references unknown or duplicate candidates.")
        expected_state = "terminal" if {by_id[value]["state"] for value in ids} <= {"successful", "failed"} else "active"
        if wave["state"] != expected_state:
            raise ValueError("Completion wave state conflicts with candidate accounting.")
        wave_candidate_ids.extend(ids)
    if wave_candidate_ids != candidate_ids:
        raise ValueError("Completion waves do not preserve exact candidate prefix membership.")
    counted_failures = sum(
        int(candidate["terminal_evidence"]["failure_circuit_contribution"])
        for candidate in candidates
        if isinstance(candidate.get("terminal_evidence"), dict)
    )
    expected_circuit = {
        "counted_failures": counted_failures,
        "open": counted_failures > maximum_failed_cases,
    }
    if failure_circuit != expected_circuit:
        raise ValueError("Completion-local failure circuit conflicts with terminal candidate evidence.")


def create_completion(
    *,
    parent_run_id: str,
    parent_partial_evidence: Mapping[str, Any],
    parent_partial_sha256: str,
    targets: Sequence[CompletionTarget],
    replacement_pool_size: int,
    maximum_failed_cases: int = 0,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create or resume one completion owner under its exclusive state lock."""
    pool = _positive_int(replacement_pool_size, label="replacement_pool_size")
    maximum_failures = _nonnegative_int(maximum_failed_cases, label="maximum_failed_cases")
    identifier = completion_id(parent_run_id=parent_run_id, parent_partial_sha256=parent_partial_sha256)
    path = _state_path(identifier, storage_root=storage_root)
    targets_payload = _targets_payload(targets)
    parent_snapshot = copy.deepcopy(dict(parent_partial_evidence))
    path.parent.mkdir(parents=True, exist_ok=True)
    with common.locking.exclusive_file_lock(path.parent / "completion.lock", blocking=True):
        if path.exists():
            state = load_completion(identifier, storage_root=storage_root)
            expected = {
                "parent_run_id": parent_run_id,
                "parent_partial_sha256": parent_partial_sha256,
                "parent_partial_evidence": parent_snapshot,
                "targets": targets_payload,
                "failure_policy": {
                    "maximum_failed_cases": maximum_failures,
                    "counted_failure_classes": ["solver_failed", "technical_runtime_timed_out"],
                },
            }
            if {key: state[key] for key in expected} != expected:
                raise ValueError("Persisted completion identity collides with different immutable parent evidence or targets.")
            if extend_pool(state, replacement_pool_size=pool, storage_root=storage_root):
                return state
            return state
        state = {
            "schema_version": _SCHEMA_VERSION,
            "owner": _OWNER,
            "completion_id": identifier,
            "parent_run_id": parent_run_id,
            "parent_partial_sha256": parent_partial_sha256,
            "parent_partial_evidence": parent_snapshot,
            "targets": targets_payload,
            "failure_policy": {
                "maximum_failed_cases": maximum_failures,
                "counted_failure_classes": ["solver_failed", "technical_runtime_timed_out"],
            },
            "failure_circuit": {"counted_failures": 0, "open": False},
            "pool_high_water": pool,
            "candidates": [],
            "waves": [],
        }
        save_completion(state, storage_root=storage_root)
        return state


def load_completion(completion_id_value: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Load and minimally validate immutable completion state."""
    path = _state_path(completion_id_value, storage_root=storage_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Completion state is missing or unsafe: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Completion state must contain one JSON object.")
    _validate_completion_state(payload, expected_completion_id=completion_id_value)
    return payload


def save_completion(state: Mapping[str, Any], *, storage_root: Path | str | None = None) -> Path:
    """Atomically persist one fully resolved completion state."""
    identifier = str(state.get("completion_id", ""))
    _validate_completion_state(state, expected_completion_id=identifier)
    path = _state_path(identifier, storage_root=storage_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return common.serialization.atomic_write_json(path, dict(state))


def ensure_pool_capacity(state: Mapping[str, Any], *, replacement_pool_size: int) -> None:
    """Reject a requested pool decrease below durable candidate materialization."""
    requested = _positive_int(replacement_pool_size, label="replacement_pool_size")
    existing = max(int(state["pool_high_water"]), len(state["candidates"]))
    if requested < existing:
        raise ValueError(f"replacement_pool_size {requested} is below persisted completion high-water {existing}.")


def extend_pool(state: dict[str, Any], *, replacement_pool_size: int, storage_root: Path | str | None = None) -> bool:
    """Raise durable pool high-water without allocating candidates."""
    ensure_pool_capacity(state, replacement_pool_size=replacement_pool_size)
    requested = int(replacement_pool_size)
    if requested == state["pool_high_water"]:
        return False
    state["pool_high_water"] = requested
    save_completion(state, storage_root=storage_root)
    return True


def supplemental_unit_point(
    *,
    registry: Mapping[str, Mapping[str, Any]],
    blocks: Sequence[str],
    block_parameters: Mapping[str, Sequence[str]],
    seed: int,
    ordinal: int,
) -> dict[str, list[float]]:
    """Return prefix-stable Sobol points for the authoritative active registry blocks."""
    position = _positive_int(ordinal, label="ordinal") - 1
    dimensions = sampling_service.materials.sampling_block_dimensions(
        registry,
        blocks=tuple(blocks),
        block_parameters={key: tuple(value) for key, value in block_parameters.items()},
    )
    return {
        block: sampling_service.unit_design(
            "sobol",
            count=ordinal,
            dimensions=dimensions[block],
            seed=seeding_service.derive_seed(seed, "supplemental_replacement", block),
        )[position].tolist()
        for block in blocks
    }


def _candidate(
    *,
    state: Mapping[str, Any],
    ordinal: int,
    batch_ordinal: int,
    batch_id: str,
    parent_failed_case_id: str,
    sampling_seed: int,
    unit_point: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    point = {key: list(value) for key, value in sorted(unit_point.items())}
    point_digest = _canonical({"unit_point": point})
    block_seeds = {block: seeding_service.derive_seed(sampling_seed, "supplemental_replacement", block) for block in point}
    provenance = {
        "owner": _OWNER,
        "parent_run_id": state["parent_run_id"],
        "parent_partial_sha256": state["parent_partial_sha256"],
        "parent_failed_case_id": parent_failed_case_id,
        "batch_id": batch_id,
        "candidate_ordinal": ordinal,
        "batch_ordinal": batch_ordinal,
        "sampling_algorithm": "sobol",
        "sampling_version": 1,
        "sampling_seed": sampling_seed,
        "block_seeds": block_seeds,
        "unit_point": point,
        "unit_point_sha256": point_digest,
    }
    digest = _canonical(provenance)
    return {
        "ordinal": ordinal,
        "batch_ordinal": batch_ordinal,
        "batch_id": batch_id,
        "parent_failed_case_id": parent_failed_case_id,
        "candidate_id": f"replacement__{digest[:24]}",
        "science_id": f"supplemental__{digest[:24]}",
        "provenance": provenance,
        "state": "planned",
    }


def _success_counts(state: Mapping[str, Any], *, original_successes: Mapping[str, int]) -> dict[str, int]:
    counts = {str(batch): int(count) for batch, count in original_successes.items()}
    for candidate in state["candidates"]:
        if candidate.get("state") == "successful":
            counts[candidate["batch_id"]] = counts.get(candidate["batch_id"], 0) + 1
    return counts


def uncovered_deficits(state: Mapping[str, Any], *, original_successes: Mapping[str, int]) -> dict[str, int]:
    """Return true remaining successful-case deficits without treating active work as success."""
    successes = _success_counts(state, original_successes=original_successes)
    return {target["batch_id"]: max(0, int(target["target_successes"]) - successes.get(target["batch_id"], 0)) for target in state["targets"]}


def schedulable_deficits(state: Mapping[str, Any], *, original_successes: Mapping[str, int]) -> dict[str, int]:
    """Return deficits not already reserved by planned, active, retry, or replay candidates."""
    result = uncovered_deficits(state, original_successes=original_successes)
    for candidate in state["candidates"]:
        if candidate.get("state") in _RESERVING_STATES:
            result[candidate["batch_id"]] = max(0, result[candidate["batch_id"]] - 1)
    return result


def _refresh_failure_circuit(state: dict[str, Any]) -> None:
    """Derive the persisted local circuit exclusively from terminal evidence."""
    counted = sum(
        int(candidate["terminal_evidence"]["failure_circuit_contribution"])
        for candidate in state["candidates"]
        if isinstance(candidate.get("terminal_evidence"), dict)
    )
    maximum = int(state["failure_policy"]["maximum_failed_cases"])
    state["failure_circuit"] = {"counted_failures": counted, "open": counted > maximum}


def reconcile_wave(
    state: dict[str, Any],
    *,
    candidate_states: Mapping[str, str],
    candidate_terminal_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    storage_root: Path | str | None = None,
) -> bool:
    """Apply scheduler/receipt states and exact terminal evidence transactionally."""
    working = copy.deepcopy(state)
    changed = False
    allowed = {"planned", "active", "retry", "replay", "successful", "failed"}
    terminal_evidence = {} if candidate_terminal_evidence is None else candidate_terminal_evidence
    by_id = {item["candidate_id"]: item for item in working["candidates"]}
    unknown = sorted(set(candidate_states).union(terminal_evidence).difference(by_id))
    if unknown:
        raise ValueError(f"Completion reconciliation references unknown candidates {unknown}.")
    for candidate_id, value in candidate_states.items():
        if value not in allowed:
            raise ValueError(f"Unsupported replacement candidate state {value!r}.")
        candidate = by_id[candidate_id]
        previous_state = str(candidate["state"])
        if previous_state in {"successful", "failed"} and value != previous_state:
            raise RuntimeError("Replacement terminal state is immutable.")
        evidence = terminal_evidence.get(candidate_id)
        if value in {"successful", "failed"}:
            if not isinstance(evidence, Mapping):
                raise ValueError("Terminal replacement reconciliation requires exact execution evidence.")
            admitted = copy.deepcopy(dict(evidence))
            previous_evidence = candidate.get("terminal_evidence")
            if previous_evidence is not None and previous_evidence != admitted:
                raise RuntimeError("Replacement terminal execution evidence is immutable.")
            if previous_evidence is None:
                candidate["terminal_evidence"] = admitted
                changed = True
        elif evidence is not None:
            raise ValueError("Non-terminal replacement state cannot carry terminal execution evidence.")
        if previous_state != value:
            candidate["state"] = value
            changed = True
    for wave in working["waves"]:
        states = {by_id[item]["state"] for item in wave["candidate_ids"]}
        next_state = "terminal" if states <= {"successful", "failed"} else "active"
        if wave["state"] != next_state:
            wave["state"] = next_state
            changed = True
    _refresh_failure_circuit(working)
    if changed:
        save_completion(working, storage_root=storage_root)
        state.clear()
        state.update(working)
    return changed


def allocate_next_wave(
    state: dict[str, Any],
    *,
    original_successes: Mapping[str, int],
    failed_case_ids: Mapping[str, Sequence[str]],
    sampling_seed: int | Mapping[str, int],
    unit_points: Mapping[tuple[str, int], Mapping[str, Sequence[float]]],
    storage_root: Path | str | None = None,
) -> ReplacementWave | None:
    """Allocate at most current uncovered deficits when no prior wave remains active."""
    if any(wave["state"] != "terminal" for wave in state["waves"]) or state["failure_circuit"]["open"]:
        return None
    existing_ids = [str(item["candidate_id"]) for item in state["candidates"]]
    if len(existing_ids) != len(set(existing_ids)):
        raise FileExistsError("Replacement candidate identity collision.")
    deficits = schedulable_deficits(state, original_successes=original_successes)
    remaining_pool = int(state["pool_high_water"]) - len(state["candidates"])
    if remaining_pool <= 0 or not any(deficits.values()):
        return None
    selected: list[dict[str, Any]] = []
    ordinal = len(state["candidates"])
    for target in state["targets"]:
        batch_id = target["batch_id"]
        failures = tuple(failed_case_ids.get(batch_id, ()))
        if not failures:
            continue
        for _local_index in range(min(deficits[batch_id], remaining_pool, 1)):
            ordinal += 1
            batch_ordinal = 1 + sum(item.get("batch_id") == batch_id for item in [*state["candidates"], *selected])
            candidate_seed = sampling_seed[batch_id] if isinstance(sampling_seed, Mapping) else sampling_seed
            selected.append(
                _candidate(
                    state=state,
                    ordinal=ordinal,
                    batch_ordinal=batch_ordinal,
                    batch_id=batch_id,
                    parent_failed_case_id=failures[(batch_ordinal - 1) % len(failures)],
                    sampling_seed=candidate_seed,
                    unit_point=unit_points[(batch_id, batch_ordinal)],
                )
            )
            remaining_pool -= 1
            if remaining_pool == 0:
                break
        if selected:
            break
    if not selected:
        return None
    if set(existing_ids).intersection(item["candidate_id"] for item in selected):
        raise FileExistsError("Replacement candidate identity collision.")
    state["candidates"].extend(selected)
    wave_ordinal = len(state["waves"]) + 1
    selected_ids = tuple(str(item["candidate_id"]) for item in selected)
    wave = {
        "wave_ordinal": wave_ordinal,
        "candidate_ids": list(selected_ids),
        "state": "active",
    }
    state["waves"].append(wave)
    save_completion(state, storage_root=storage_root)
    return ReplacementWave(
        wave_ordinal=wave_ordinal,
        candidate_ids=selected_ids,
        state="active",
    )


def _candidate_record(state: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    """Return one exact mutable candidate record from a completion state."""
    matches = [item for item in state["candidates"] if item.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Completion candidate identity is missing or ambiguous: {candidate_id!r}.")
    return matches[0]


def _parent_batch(
    campaign: config_service.CampaignConfig,
    candidate: Mapping[str, Any],
) -> config_service.GenerationConfig:
    """Resolve the immutable parent batch named by one candidate."""
    matches = [batch for batch in campaign.batches if batch.batch_id == candidate.get("batch_id")]
    if len(matches) != 1:
        raise ValueError("Replacement candidate parent batch is missing from the parent campaign.")
    return matches[0]


def _failed_case_index(batch: config_service.GenerationConfig, case_id: object) -> int:
    """Resolve one immutable failed original member by canonical case identity."""
    if not isinstance(case_id, str):
        raise TypeError("Replacement parent_failed_case_id must be text.")
    match = re.fullmatch(r"case_([0-9]{4,})", case_id)
    if match is None:
        raise ValueError("Replacement parent_failed_case_id is malformed.")
    case_index = int(match.group(1))
    if case_index not in batch.case_indices or batch.case_id(case_index) != case_id:
        raise ValueError("Replacement parent failure is not a member of its immutable parent batch.")
    return case_index


def _supplemental_design(
    *,
    state_identity: Mapping[str, Any],
    parent_campaign: config_service.CampaignConfig,
    parent_batch: config_service.GenerationConfig,
    candidate: Mapping[str, Any],
    physical_values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one complete scientific supplemental-design identity record."""
    provenance = candidate.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("Replacement candidate provenance is missing.")
    point = copy.deepcopy(provenance.get("unit_point"))
    physical = None if physical_values is None else copy.deepcopy(dict(physical_values))
    physical_digest = None if physical is None else _canonical({"physical_values": physical})
    return {
        "schema_kind": "generation_supplemental_design",
        "schema_version": 1,
        "completion_id": state_identity["completion_id"],
        "parent_run_id": state_identity["parent_run_id"],
        "parent_partial_sha256": state_identity["parent_partial_sha256"],
        "parent_campaign_digest": parent_campaign.campaign_digest,
        "parent_batch_id": parent_batch.batch_id,
        "parent_scientific_config_digest": parent_batch.scientific_config_digest,
        "parent_failed_case_id": candidate["parent_failed_case_id"],
        "candidate_id": candidate["candidate_id"],
        "science_id": candidate["science_id"],
        "candidate_ordinal": candidate["ordinal"],
        "batch_ordinal": candidate["batch_ordinal"],
        "sampling_algorithm": provenance["sampling_algorithm"],
        "sampling_seed": provenance["sampling_seed"],
        "block_seeds": copy.deepcopy(provenance["block_seeds"]),
        "unit_point": point,
        "unit_point_sha256": provenance["unit_point_sha256"],
        "physical_values": physical,
        "physical_values_sha256": physical_digest,
    }


def _supplemental_batch(
    parent: config_service.GenerationConfig,
    *,
    case_index: int,
    assignment: Mapping[str, Any],
    scientific: Mapping[str, Any],
) -> config_service.GenerationConfig:
    """Build one immutable one-case batch from resolved supplemental science."""
    values = copy.deepcopy(dict(scientific))
    scientific_digest = config_service.compute_scientific_config_digest(values)
    case_input_digest = config_service.compute_case_input_config_digest(values)
    return replace(
        parent,
        scientific_values=values,
        case_indices=(case_index,),
        assignments={case_index: copy.deepcopy(dict(assignment))},
        scientific_config_digest=scientific_digest,
        case_input_config_digest=case_input_digest,
        batch_identity=scientific_digest,
        batch_id=config_service.build_batch_id(parent.batch_name, scientific_digest),
        batch_storage_name=config_service.build_batch_storage_name(
            parent.profile.id,
            parent.material_family,
            parent.sampling_regime,
            parent.scientific_values["campaign_purpose"],
            scientific_digest,
        ),
    )


def materialize_candidate_campaign(
    *,
    state_identity: Mapping[str, Any],
    parent_campaign: config_service.CampaignConfig,
    candidate: Mapping[str, Any],
) -> config_service.CampaignConfig:
    """Materialize one persisted candidate as one ordinary one-case campaign."""
    required_identity = {"completion_id", "parent_run_id", "parent_partial_sha256"}
    if not required_identity <= set(state_identity):
        raise ValueError("Completion state identity is incomplete for candidate materialization.")
    parent_batch = _parent_batch(parent_campaign, candidate)
    failed_index = _failed_case_index(parent_batch, candidate.get("parent_failed_case_id"))
    _positive_int(candidate.get("ordinal"), label="candidate ordinal")
    batch_ordinal = _positive_int(candidate.get("batch_ordinal"), label="candidate batch ordinal")
    case_index = max(parent_batch.case_indices) + batch_ordinal
    assignment = parent_batch.case_assignment(failed_index)
    assignment["case_index"] = case_index
    assignment["regime_index"] = max(int(item["regime_index"]) for item in parent_batch.assignments.values()) + batch_ordinal
    scientific = copy.deepcopy(parent_batch.scientific_values)
    scientific["case_count"] = 1
    scientific["assignments"] = [copy.deepcopy(assignment)]
    scientific["supplemental_design"] = _supplemental_design(
        state_identity=state_identity,
        parent_campaign=parent_campaign,
        parent_batch=parent_batch,
        candidate=candidate,
        physical_values=None,
    )
    provisional = _supplemental_batch(
        parent_batch,
        case_index=case_index,
        assignment=assignment,
        scientific=scientific,
    )
    sample = sampling_service.sample_case(provisional, case_index)
    scientific["supplemental_design"] = _supplemental_design(
        state_identity=state_identity,
        parent_campaign=parent_campaign,
        parent_batch=parent_batch,
        candidate=candidate,
        physical_values=sample.values,
    )
    batch = _supplemental_batch(
        parent_batch,
        case_index=case_index,
        assignment=assignment,
        scientific=scientific,
    )
    verified = sampling_service.sample_case(batch, case_index)
    if verified.values != sample.values:
        raise RuntimeError("Supplemental physical values changed during final identity materialization.")
    campaign_name = f"replacement_{str(candidate['candidate_id']).removeprefix('replacement__')}"
    campaign_id = f"{campaign_name}_v1"
    batch_values = copy.deepcopy(batch.scientific_values)
    batch_values["campaign_id"] = campaign_id
    batch = _supplemental_batch(
        batch,
        case_index=case_index,
        assignment=assignment,
        scientific=batch_values,
    )
    campaign_digest = _canonical(
        {
            "schema_kind": "resolved_generation_supplemental_campaign",
            "schema_version": 1,
            "completion_id": state_identity["completion_id"],
            "parent_run_id": state_identity["parent_run_id"],
            "parent_partial_sha256": state_identity["parent_partial_sha256"],
            "parent_campaign_digest": parent_campaign.campaign_digest,
            "candidate_id": candidate["candidate_id"],
            "batch_id": batch.batch_id,
            "case_index": case_index,
        }
    )
    return replace(
        parent_campaign,
        campaign_name=campaign_name,
        campaign_digest=campaign_digest,
        package_request_digest=common.serialization.canonical_json_sha256([]),
        campaign_id=campaign_id,
        evaluation_regimes=(batch.evaluation_regime,),
        total_case_count=1,
        batches=(batch,),
        dataset_packages=(),
    )


def synthetic_manifest_extension(campaign: config_service.CampaignConfig) -> dict[str, Any] | None:
    """Return the typed manifest extension for one supplemental campaign."""
    if len(campaign.batches) != 1:
        return None
    batch = campaign.batches[0]
    design = batch.scientific_values.get("supplemental_design")
    if design is None:
        return None
    if not isinstance(design, dict):
        raise TypeError("Supplemental campaign design must be one mapping.")
    payload = {
        "schema_kind": "generation_supplemental_campaign_source",
        "schema_version": 1,
        "design": copy.deepcopy(design),
        "synthetic_campaign_id": campaign.campaign_id,
        "synthetic_campaign_digest": campaign.campaign_digest,
        "synthetic_batch_id": batch.batch_id,
        "synthetic_case_index": batch.case_indices[0],
    }
    return {**payload, "payload_sha256": _canonical(payload)}


def validate_synthetic_manifest_extension(value: object) -> dict[str, Any]:
    """Validate one exact portable supplemental campaign reconstruction record."""
    required = {
        "schema_kind",
        "schema_version",
        "design",
        "synthetic_campaign_id",
        "synthetic_campaign_digest",
        "synthetic_batch_id",
        "synthetic_case_index",
        "payload_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Supplemental campaign manifest extension has an unsupported shape.")
    payload = {key: copy.deepcopy(value[key]) for key in required - {"payload_sha256"}}
    if (
        value.get("schema_kind") != "generation_supplemental_campaign_source"
        or value.get("schema_version") != 1
        or _SHA256.fullmatch(str(value.get("synthetic_campaign_digest", ""))) is None
        or _SHA256.fullmatch(str(value.get("payload_sha256", ""))) is None
        or _canonical(payload) != value["payload_sha256"]
    ):
        raise ValueError("Supplemental campaign manifest extension identity is invalid.")
    return copy.deepcopy(value)


def campaign_from_synthetic_manifest(
    manifest: Mapping[str, Any],
    *,
    require_executable: bool = True,
) -> config_service.CampaignConfig:
    """Reconstruct and verify one supplemental campaign from portable evidence."""
    extension = validate_synthetic_manifest_extension(manifest.get("synthetic_completion"))
    design = extension["design"]
    if not isinstance(design, dict):
        raise TypeError("Supplemental campaign design is malformed.")
    relative = Path(str(manifest.get("campaign_config", "")))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:3] != ("configs", "generation", "campaigns"):
        raise ValueError("Supplemental parent campaign path is unsafe.")
    source = (common.paths.get_project_root().resolve() / relative).resolve()
    parent = config_service.load_campaign_config(source, require_executable=require_executable)
    if parent.campaign_digest != design.get("parent_campaign_digest"):
        raise RuntimeError("Supplemental parent campaign identity changed after launch.")
    candidate = {
        "batch_id": design["parent_batch_id"],
        "candidate_id": design["candidate_id"],
        "parent_failed_case_id": design["parent_failed_case_id"],
        "ordinal": design["candidate_ordinal"],
        "batch_ordinal": design["batch_ordinal"],
        "provenance": {
            "sampling_algorithm": design["sampling_algorithm"],
            "sampling_seed": design["sampling_seed"],
            "block_seeds": design["block_seeds"],
            "unit_point": design["unit_point"],
            "unit_point_sha256": design["unit_point_sha256"],
        },
        "science_id": design["science_id"],
    }
    state_identity = {
        "completion_id": design["completion_id"],
        "parent_run_id": design["parent_run_id"],
        "parent_partial_sha256": design["parent_partial_sha256"],
    }
    campaign = materialize_candidate_campaign(
        state_identity=state_identity,
        parent_campaign=parent,
        candidate=candidate,
    )
    batch = campaign.batches[0]
    expected = {
        "campaign_id": extension["synthetic_campaign_id"],
        "campaign_digest": extension["synthetic_campaign_digest"],
        "batch_id": extension["synthetic_batch_id"],
        "case_index": extension["synthetic_case_index"],
    }
    observed = {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "batch_id": batch.batch_id,
        "case_index": batch.case_indices[0],
    }
    if observed != expected:
        raise RuntimeError("Supplemental campaign reconstruction changed its persisted identity.")
    if (
        campaign.campaign_id != manifest.get("campaign_id")
        or campaign.campaign_digest != manifest.get("campaign_digest")
        or common.serialization.canonical_json_sha256(campaign.execution_values) != manifest.get("execution_config_digest")
        or [batch.batch_name] != manifest.get("selected_batch_names")
        or manifest.get("dataset_packages") != []
    ):
        raise RuntimeError("Supplemental campaign conflicts with its campaign-run manifest.")
    return campaign


def record_candidate_materialization(
    state: dict[str, Any],
    *,
    candidate_id: str,
    parent_campaign: config_service.CampaignConfig,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> config_service.CampaignConfig:
    """Persist a collision-checked one-case campaign identity before submission."""
    from . import generation_campaign as campaign_service

    candidate = _candidate_record(state, candidate_id)
    campaign = materialize_candidate_campaign(
        state_identity=state,
        parent_campaign=parent_campaign,
        candidate=candidate,
    )
    commit = source_service.validate_git_commit(git_commit)
    run_id = campaign_service.campaign_run_id(campaign, git_commit=commit)
    materialization = {
        "campaign_run_id": run_id,
        "extension": synthetic_manifest_extension(campaign),
        "batch_id": campaign.batches[0].batch_id,
        "case_id": campaign.batches[0].case_id(campaign.batches[0].case_indices[0]),
        "case_index": campaign.batches[0].case_indices[0],
        "simulation_science_id": candidate["science_id"],
        "git_commit": commit,
    }
    previous = candidate.get("materialization")
    if previous is not None and previous != materialization:
        raise FileExistsError("Persisted replacement materialization conflicts with deterministic reconstruction.")
    other = [item["materialization"] for item in state["candidates"] if item is not candidate and isinstance(item.get("materialization"), dict)]
    for field in ("campaign_run_id", "batch_id"):
        if any(item.get(field) == materialization[field] for item in other):
            raise FileExistsError(f"Replacement materialization collided on {field}.")
    if previous is None:
        candidate["materialization"] = materialization
        save_completion(state, storage_root=storage_root)
    return campaign


def parent_partial_counts(
    parent_campaign: config_service.CampaignConfig,
    evidence: Mapping[str, Any],
    *,
    parent_run_id: str,
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    """Validate one immutable parent partial snapshot and derive exact counts."""
    successful = evidence.get("successful_cases")
    failed = evidence.get("failed_cases")
    if (
        evidence.get("schema_kind") != "generation_campaign_partial"
        or evidence.get("schema_version") != 1
        or evidence.get("campaign_run_id") != parent_run_id
        or evidence.get("campaign_id") != parent_campaign.campaign_id
        or not isinstance(successful, list)
        or not isinstance(failed, list)
        or not failed
    ):
        raise ValueError("Parent partial evidence is not one compatible completed-with-failures snapshot.")
    expected = {
        (batch.batch_id, batch.batch_name, case_index, batch.case_id(case_index))
        for batch in parent_campaign.batches
        for case_index in batch.case_indices
    }
    records = [*successful, *failed]
    observed: set[tuple[str, str, int, str]] = set()
    success_counts = dict.fromkeys((batch.batch_id for batch in parent_campaign.batches), 0)
    failures: dict[str, list[str]] = {batch.batch_id: [] for batch in parent_campaign.batches}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Parent partial case evidence must contain objects.")
        case_index = record.get("case_index")
        if isinstance(case_index, bool) or not isinstance(case_index, int):
            raise TypeError("Parent partial case index must be an integer.")
        identity = (
            str(record.get("batch_id")),
            str(record.get("batch_name")),
            case_index,
            str(record.get("case_id")),
        )
        if identity not in expected or identity in observed:
            raise ValueError("Parent partial evidence contains unknown or duplicate campaign membership.")
        observed.add(identity)
        if record in successful:
            if record.get("state") != "successful" or record.get("classified_state") != "successful":
                raise ValueError("Parent partial successful-case classification is inconsistent.")
            success_counts[identity[0]] += 1
        else:
            if record.get("state") != "failed" or record.get("classified_state") == "successful":
                raise ValueError("Parent partial failed-case classification is inconsistent.")
            failures[identity[0]].append(identity[3])
    if observed != expected:
        raise ValueError("Parent partial evidence does not cover exact original campaign membership.")
    return success_counts, {batch_id: tuple(values) for batch_id, values in failures.items()}


def create_completion_from_partial(
    *,
    parent_campaign: config_service.CampaignConfig,
    parent_run_id: str,
    parent_partial_evidence: Mapping[str, Any],
    parent_partial_sha256: str,
    replacement_pool_size: int,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create or resume completion from a validated parent partial snapshot."""
    parent_partial_counts(
        parent_campaign,
        parent_partial_evidence,
        parent_run_id=parent_run_id,
    )
    return create_completion(
        parent_run_id=parent_run_id,
        parent_partial_evidence=parent_partial_evidence,
        parent_partial_sha256=parent_partial_sha256,
        targets=tuple(CompletionTarget(batch.batch_id, len(batch.case_indices)) for batch in parent_campaign.batches),
        replacement_pool_size=replacement_pool_size,
        maximum_failed_cases=int(parent_campaign.execution_values["runtime"]["maximum_failed_cases"]),
        storage_root=storage_root,
    )


def _campaign_compatibility_identity(
    campaign: config_service.CampaignConfig,
) -> dict[str, Any]:
    """Return the resolved structural identity used for parent discovery."""
    return {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "simulation_profile": campaign.profile.id,
        "template_sha256": campaign.template_sha256,
        "execution_config_digest": _canonical(campaign.execution_values),
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "batch_identity": batch.batch_identity,
                "scientific_config_digest": batch.scientific_config_digest,
                "case_input_config_digest": batch.case_input_config_digest,
                "case_indices": list(batch.case_indices),
                "material_family": batch.material_family,
                "material_role": batch.material_role,
                "sampling_regime": batch.sampling_regime,
                "evaluation_regime": batch.evaluation_regime,
            }
            for batch in campaign.batches
        ],
    }


def _compatible_campaign_run(
    run_id: str,
    *,
    parent_identity: Mapping[str, Any],
    storage_root: Path,
) -> tuple[dict[str, Any], config_service.CampaignConfig] | None:
    """Admit one structurally compatible non-synthetic campaign run, if present."""
    from .publication import generation_publication_campaign_evidence as campaign_evidence

    path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Campaign-run manifest is missing or unsafe: {path}.")
    raw = campaign_evidence.load_json_object(path, label="campaign-run manifest")
    if "synthetic_completion" in raw or raw.get("campaign_digest") != parent_identity.get("campaign_digest"):
        return None
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    if _campaign_compatibility_identity(campaign) != dict(parent_identity):
        return None
    return manifest, campaign


def _completion_owner_for_parent(
    parent_run_id: str,
    parent_partial_sha256: str,
    *,
    storage_root: Path,
) -> dict[str, Any] | None:
    """Return the unique compatible completion owner and reject parent conflicts."""
    root = common.paths.get_generation_meta_root(storage_root=storage_root) / "completions"
    if not root.exists():
        return None
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Completion owner root is missing or unsafe: {root}.")
    matches: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Completion owner entry is unsafe: {directory}.")
        path = directory / "completion.json"
        if not path.exists():
            continue
        state = load_completion(directory.name, storage_root=storage_root)
        if state["parent_run_id"] != parent_run_id:
            continue
        if state["parent_partial_sha256"] != parent_partial_sha256:
            conflicts.append(str(state["completion_id"]))
        else:
            matches.append(state)
    if conflicts:
        raise RuntimeError(f"Parent campaign has completion ownership bound to different immutable partial evidence: {sorted(conflicts)}.")
    if len(matches) > 1:
        raise RuntimeError("Parent campaign has ambiguous duplicate completion owners.")
    return None if not matches else matches[0]


def _completion_plan_report(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """Return immutable target, package, and transient-shard declarations for operators."""
    target_counts = {batch.batch_id: len(batch.case_indices) for batch in campaign.batches}
    package_declarations = [copy.deepcopy(dict(plan)) for plan in campaign.dataset_packages]
    shard_requirements = [
        {
            "dataset_name": plan["dataset_name"],
            "dataset_view": plan["dataset_view"],
            "evaluation_regime": plan["evaluation_regime"],
            "required": bool(plan["training_payload"]["required"]),
            "target_shard_bytes": int(plan["training_payload"]["target_shard_bytes"]),
        }
        for plan in campaign.dataset_packages
        if plan["dataset_view"] == "transient_drying" and plan.get("training_payload") is not None
    ]
    return {
        "target_counts": target_counts,
        "target_batches": [
            {
                "batch_id": batch.batch_id,
                "batch_name": batch.batch_name,
                "material_family": batch.material_family,
                "material_role": batch.material_role,
                "sampling_regime": batch.sampling_regime,
                "evaluation_regime": batch.evaluation_regime,
                "target_successes": len(batch.case_indices),
            }
            for batch in campaign.batches
        ],
        "package_declarations": package_declarations,
        "pt_shard_requirements": shard_requirements,
    }


def _active_parent_accounting(
    run_id: str,
    *,
    target_counts: Mapping[str, int],
    storage_root: Path,
) -> dict[str, Any]:
    """Return read-only successful, failed, active, and reserved parent counts."""
    from . import generation_campaign as campaign_service

    snapshot = campaign_service.campaign_status(
        run_id,
        storage_root=storage_root,
        query_scheduler=False,
    )
    current_successes = dict.fromkeys(target_counts, 0)
    original_failures = dict.fromkeys(target_counts, 0)
    for case in snapshot["cases"]:
        batch_id = str(case["batch_id"])
        if case["state"] == "successful":
            current_successes[batch_id] += 1
        elif case["state"] == "failed":
            original_failures[batch_id] += 1
    return {
        "original_successes": current_successes,
        "original_failures": original_failures,
        "current_successes": copy.deepcopy(current_successes),
        "success_deficits": {batch_id: int(target) - current_successes[batch_id] for batch_id, target in target_counts.items()},
        "parent_runtime": {
            "running": int(snapshot["running_jobs"]),
            "pending": int(snapshot["pending_jobs"]),
            "reserved": int(snapshot["admission"]["count"]),
        },
    }


def find_compatible_completion_parent(
    parent_campaign: config_service.CampaignConfig,
    *,
    parent_run_id: str | None = None,
    storage_root: Path | str | None = None,
    require_transferred: bool = True,
) -> dict[str, Any]:
    """Resolve zero or one structurally compatible historical parent campaign."""
    from . import generation_campaign as campaign_service
    from .publication import generation_publication_campaign_evidence as campaign_evidence

    if not isinstance(require_transferred, bool):
        raise TypeError("require_transferred must be boolean.")
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    expected = _campaign_compatibility_identity(parent_campaign)
    plan_report = _completion_plan_report(parent_campaign)
    candidates: list[tuple[str, dict[str, Any], config_service.CampaignConfig]] = []
    if parent_run_id is not None:
        requested = common.paths.validate_logical_name(parent_run_id, label="parent_run_id")
        admitted = _compatible_campaign_run(
            requested,
            parent_identity=expected,
            storage_root=storage,
        )
        if admitted is None:
            raise ValueError("Explicit parent run is not structurally compatible with the requested campaign.")
        candidates.append((requested, *admitted))
    elif root.exists():
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Campaign-run root is missing or unsafe: {root}.")
        for directory in sorted(root.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"Campaign-run entry is unsafe: {directory}.")
            path = directory / "campaign_run.json"
            if not path.is_file() or path.is_symlink():
                continue
            raw = campaign_evidence.load_json_object(path, label="campaign-run manifest")
            if "synthetic_completion" in raw or raw.get("campaign_digest") != expected["campaign_digest"]:
                continue
            admitted = _compatible_campaign_run(
                directory.name,
                parent_identity=expected,
                storage_root=storage,
            )
            if admitted is not None:
                candidates.append((directory.name, *admitted))
    if not candidates:
        zero_counts = dict.fromkeys(plan_report["target_counts"], 0)
        return {
            "schema_kind": "generation_completion_parent_resolution",
            "schema_version": 1,
            "status": "fresh",
            "compatible_parent_candidates": [],
            "selected_parent": None,
            "parent_run_id": None,
            "parent_state": None,
            "completion_id": None,
            "expected_completion_id": None,
            "original_successes": zero_counts,
            "original_failures": zero_counts,
            "current_successes": zero_counts,
            "success_deficits": copy.deepcopy(plan_report["target_counts"]),
            **plan_report,
        }
    if len(candidates) != 1:
        run_ids = sorted(item[0] for item in candidates)
        raise RuntimeError(f"Multiple structurally compatible parent campaigns require explicit --parent-run-id: {run_ids}.")
    selected_run_id, manifest, launched = candidates[0]
    state = str(manifest["state"])
    result: dict[str, Any] = {
        "schema_kind": "generation_completion_parent_resolution",
        "schema_version": 1,
        "status": "compatible_active",
        "compatible_parent_candidates": [selected_run_id],
        "selected_parent": {
            "campaign_run_id": selected_run_id,
            "state": state,
            "git_commit": manifest["git_commit"],
        },
        "parent_run_id": selected_run_id,
        "parent_state": state,
        "campaign_id": launched.campaign_id,
        "campaign_digest": launched.campaign_digest,
        "git_commit": manifest["git_commit"],
        "completion_id": None,
        "expected_completion_id": None,
        **plan_report,
    }
    if state == "complete":
        if require_transferred:
            campaign_service.validate_transferred_campaign(selected_run_id, storage_root=storage)
        result["status"] = "compatible_complete"
        result["original_successes"] = copy.deepcopy(plan_report["target_counts"])
        result["original_failures"] = dict.fromkeys(plan_report["target_counts"], 0)
        result["current_successes"] = copy.deepcopy(plan_report["target_counts"])
        result["success_deficits"] = dict.fromkeys(plan_report["target_counts"], 0)
        return result
    if state != "completed_with_failures":
        result.update(
            _active_parent_accounting(
                selected_run_id,
                target_counts=plan_report["target_counts"],
                storage_root=storage,
            )
        )
        return result
    if require_transferred:
        campaign_service.validate_partially_transferred_campaign(selected_run_id, storage_root=storage)
    evidence = campaign_service.read_partial_campaign_diagnostic_evidence(
        selected_run_id,
        storage_root=storage,
    )
    if evidence is None:
        raise FileNotFoundError("Compatible partial parent is missing campaign_partial.json.")
    partial_path = (
        campaign_evidence.campaign_run_directory(
            selected_run_id,
            storage_root=storage,
        )
        / "campaign_partial.json"
    )
    partial_sha256 = common.serialization.file_sha256(partial_path)
    original_successes, failures = parent_partial_counts(
        launched,
        evidence,
        parent_run_id=selected_run_id,
    )
    owner = _completion_owner_for_parent(
        selected_run_id,
        partial_sha256,
        storage_root=storage,
    )
    deficits = {batch.batch_id: len(batch.case_indices) - original_successes[batch.batch_id] for batch in launched.batches}
    result.update(
        {
            "status": "compatible_partial",
            "parent_partial_path": str(partial_path),
            "parent_partial_sha256": partial_sha256,
            "original_successes": original_successes,
            "original_failures": {batch_id: len(values) for batch_id, values in failures.items()},
            "current_successes": copy.deepcopy(original_successes),
            "success_deficits": deficits,
            "completion_id": completion_id(
                parent_run_id=selected_run_id,
                parent_partial_sha256=partial_sha256,
            ),
            "expected_completion_id": completion_id(
                parent_run_id=selected_run_id,
                parent_partial_sha256=partial_sha256,
            ),
            "completion_status": None,
            "replacement_pool_size": None,
        }
    )
    if owner is not None:
        status = completion_status(owner, original_successes=original_successes)
        result["completion_status"] = status["status"]
        result["replacement_pool_size"] = status["replacement_pool_size"]
        result["current_successes"] = {
            batch_id: int(plan_report["target_counts"][batch_id]) - int(status["success_deficits"][batch_id])
            for batch_id in plan_report["target_counts"]
        }
        result["completion"] = status
    return result


def _embedded_original_successes(state: Mapping[str, Any]) -> dict[str, int]:
    """Return exact original-success counts from production partial evidence."""
    targets = {str(item["batch_id"]) for item in state["targets"]}
    evidence = state.get("parent_partial_evidence")
    records = evidence.get("successful_cases") if isinstance(evidence, dict) else None
    if not isinstance(records, list):
        raise TypeError("Completion transfer requires production partial successful-case evidence.")
    counts = dict.fromkeys(targets, 0)
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("classified_state") != "successful":
            raise ValueError("Completion parent successful-case evidence is malformed.")
        batch_id = str(record.get("batch_id"))
        case_id = str(record.get("case_id"))
        if batch_id not in counts or (batch_id, case_id) in seen:
            raise ValueError("Completion parent successful-case membership is unknown or duplicated.")
        seen.add((batch_id, case_id))
        counts[batch_id] += 1
    return counts


def completion_transfer_plan(
    completion_id_value: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return exact completed state and successful replacement run transfer membership."""
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    state = load_completion(completion_id_value, storage_root=storage)
    original_successes = _embedded_original_successes(state)
    status = completion_status(state, original_successes=original_successes)
    if status["status"] != "complete":
        raise RuntimeError("Replacement transfer is forbidden until completion reaches exact successful targets.")
    replacements: list[dict[str, Any]] = []
    for candidate in state["candidates"]:
        if candidate["state"] != "successful":
            continue
        materialization = candidate.get("materialization")
        if not isinstance(materialization, dict):
            raise TypeError("Successful replacement lacks persisted campaign materialization.")
        terminal_evidence = candidate.get("terminal_evidence")
        if not isinstance(terminal_evidence, dict):
            raise TypeError("Successful replacement lacks terminal execution evidence.")
        replacements.append(
            {
                "candidate_id": candidate["candidate_id"],
                "target_batch_id": candidate["batch_id"],
                "campaign_run_id": materialization["campaign_run_id"],
                "git_commit": materialization["git_commit"],
                "campaign_run_manifest_sha256": terminal_evidence["campaign_run_manifest_sha256"],
                "terminal_batch_id": materialization["batch_id"],
            }
        )
    path = _state_path(completion_id_value, storage_root=storage)
    return {
        "schema_kind": "generation_completion_transfer_plan",
        "schema_version": 1,
        "completion_id": completion_id_value,
        "parent_run_id": state["parent_run_id"],
        "parent_partial_sha256": state["parent_partial_sha256"],
        "completion_state_path": str(path),
        "completion_state_sha256": common.serialization.file_sha256(path),
        "replacement_campaigns": replacements,
    }


def build_completion_composite(
    completion_id_value: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build or admit the exact parent-success plus replacement-success composite."""
    from . import generation_campaign as campaign_service
    from .publication import generation_publication_campaign_evidence as campaign_evidence
    from .publication import generation_publication_completion_composite as composite_service
    from .runtime import generation_runtime_batch as batch_runtime

    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    state = load_completion(completion_id_value, storage_root=storage)
    original_successes = _embedded_original_successes(state)
    status = completion_status(state, original_successes=original_successes)
    if status["status"] != "complete":
        raise RuntimeError("Completion composite requires exact successful target counts.")
    parent_run_id = str(state["parent_run_id"])
    campaign_service.validate_partially_transferred_campaign(parent_run_id, storage_root=storage)
    parent_directory = campaign_evidence.campaign_run_directory(parent_run_id, storage_root=storage)
    partial_path = parent_directory / "campaign_partial.json"
    if common.serialization.file_sha256(partial_path) != state["parent_partial_sha256"]:
        raise RuntimeError("Transferred parent partial evidence differs from completion ownership.")
    partial = campaign_evidence.load_json_object(partial_path, label="parent partial campaign evidence")
    if partial != state["parent_partial_evidence"]:
        raise RuntimeError("Transferred parent partial payload differs from completion snapshot.")
    parent_manifest_path = campaign_evidence.campaign_run_manifest_path(parent_run_id, storage_root=storage)
    parent_manifest = campaign_evidence.load_campaign_run(parent_run_id, storage_root=storage)
    parent_manifest_sha256 = common.serialization.file_sha256(parent_manifest_path)
    parent_campaign = campaign_evidence.campaign_from_manifest(parent_manifest)
    validated_successes, _failures = parent_partial_counts(
        parent_campaign,
        partial,
        parent_run_id=parent_run_id,
    )
    if validated_successes != original_successes:
        raise RuntimeError("Parent partial success accounting changed during composite admission.")
    parent_batches = {batch.batch_id: batch for batch in parent_campaign.batches}
    original_sources: list[composite_service.CompositeCaseSource] = []
    for record in partial["successful_cases"]:
        batch = parent_batches[str(record["batch_id"])]
        case = batch_runtime.admit_completed_case(
            batch,
            int(record["case_index"]),
            storage_root=storage,
            validation_depth="full",
            git_commit=str(parent_manifest["git_commit"]),
        )
        original_sources.append(
            composite_service.CompositeCaseSource(
                batch_id=batch.batch_id,
                batch_name=batch.batch_name,
                material_family=batch.material_family,
                material_role=batch.material_role,
                evaluation_regime=batch.evaluation_regime,
                sampling_regime=batch.sampling_regime,
                source_run_id=parent_run_id,
                source_git_commit=str(parent_manifest["git_commit"]),
                source_campaign_manifest_sha256=parent_manifest_sha256,
                terminal=None,
                case=case,
                source_kind="parent_partial",
            )
        )
    replacement_sources: list[composite_service.CompositeCaseSource] = []
    for candidate in state["candidates"]:
        if candidate["state"] != "successful":
            continue
        materialization = candidate.get("materialization")
        if not isinstance(materialization, dict):
            raise TypeError("Successful replacement lacks persisted campaign materialization.")
        terminal_evidence = candidate.get("terminal_evidence")
        if not isinstance(terminal_evidence, dict):
            raise TypeError("Successful replacement lacks terminal execution evidence.")
        run_id = str(materialization["campaign_run_id"])
        campaign_service.validate_transferred_campaign(run_id, storage_root=storage)
        manifest_path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage)
        manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
        if (
            manifest.get("git_commit") != materialization["git_commit"]
            or terminal_evidence["git_commit"] != materialization["git_commit"]
            or common.serialization.file_sha256(manifest_path) != terminal_evidence["campaign_run_manifest_sha256"]
        ):
            raise RuntimeError("Transferred replacement campaign manifest differs from persisted execution provenance.")
        campaign = campaign_evidence.campaign_from_manifest(manifest)
        extension = validate_synthetic_manifest_extension(manifest.get("synthetic_completion"))
        design = extension["design"]
        if (
            not isinstance(design, dict)
            or design.get("completion_id") != completion_id_value
            or design.get("parent_run_id") != parent_run_id
            or design.get("parent_partial_sha256") != state["parent_partial_sha256"]
            or design.get("candidate_id") != candidate["candidate_id"]
            or design.get("parent_batch_id") != candidate["batch_id"]
            or len(campaign.batches) != 1
        ):
            raise RuntimeError("Transferred replacement campaign conflicts with completion identity.")
        synthetic_batch = campaign.batches[0]
        if synthetic_batch.batch_id != materialization["batch_id"]:
            raise RuntimeError("Transferred replacement batch differs from persisted materialization.")
        terminal = batch_runtime.admit_terminal_batch(
            synthetic_batch.batch_storage_name,
            storage_root=storage,
            validation_depth="full",
        )
        if terminal.batch_id != synthetic_batch.batch_id or len(terminal.cases) != 1:
            raise RuntimeError("Transferred replacement terminal evidence has invalid membership.")
        parent_batch = parent_batches[str(candidate["batch_id"])]
        replacement_sources.append(
            composite_service.CompositeCaseSource(
                batch_id=parent_batch.batch_id,
                batch_name=parent_batch.batch_name,
                material_family=parent_batch.material_family,
                material_role=parent_batch.material_role,
                evaluation_regime=parent_batch.evaluation_regime,
                sampling_regime=parent_batch.sampling_regime,
                source_run_id=run_id,
                source_git_commit=str(materialization["git_commit"]),
                source_campaign_manifest_sha256=str(terminal_evidence["campaign_run_manifest_sha256"]),
                terminal=terminal,
                case=terminal.cases[0],
                source_kind="replacement",
            )
        )
    targets = {str(item["batch_id"]): int(item["target_successes"]) for item in state["targets"]}
    state_sha256 = _canonical(state)
    return composite_service.build_composite_receipt(
        completion_id=completion_id_value,
        completion_state=state,
        completion_state_sha256=state_sha256,
        parent_run_id=parent_run_id,
        parent_partial_sha256=str(state["parent_partial_sha256"]),
        targets=targets,
        original_cases=original_sources,
        replacement_cases=replacement_sources,
        storage_root=storage,
    )


def create_completion_from_partial_path(
    *,
    parent_campaign: config_service.CampaignConfig,
    parent_run_id: str,
    parent_partial_path: Path | str,
    parent_partial_sha256: str,
    replacement_pool_size: int,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Create or resume completion from one exact safe partial-evidence file."""
    path = Path(parent_partial_path).expanduser()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Parent partial evidence is missing or unsafe: {path}.")
    if _SHA256.fullmatch(parent_partial_sha256) is None:
        raise ValueError("Parent partial evidence digest must be lowercase SHA-256.")
    observed_sha256 = common.serialization.file_sha256(path)
    if observed_sha256 != parent_partial_sha256:
        raise RuntimeError("Parent partial evidence bytes differ from the requested immutable digest.")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Parent partial evidence is unreadable: {path}.") from error
    if not isinstance(evidence, dict):
        raise TypeError("Parent partial evidence must contain one JSON object.")
    return create_completion_from_partial(
        parent_campaign=parent_campaign,
        parent_run_id=parent_run_id,
        parent_partial_evidence=evidence,
        parent_partial_sha256=parent_partial_sha256,
        replacement_pool_size=replacement_pool_size,
        storage_root=storage_root,
    )


def completion_status_for_id(
    completion_id_value: str,
    *,
    storage_root: Path | str | None = None,
    if_present: bool = False,
) -> dict[str, Any]:
    """Report completion execution, publication, packages, shards, and readiness."""
    from .publication import generation_publication_completion_composite as composite_service

    if not isinstance(if_present, bool):
        raise TypeError("if_present must be boolean.")
    identifier = common.paths.validate_logical_name(completion_id_value, label="completion_id")
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    state_path = _state_path(identifier, storage_root=storage)
    if state_path.is_symlink() or state_path.parent.is_symlink():
        raise RuntimeError("Completion state path is unsafe.")
    if if_present and not state_path.exists():
        return {
            "schema_kind": "generation_campaign_completion_status",
            "schema_version": _SCHEMA_VERSION,
            "completion_id": identifier,
            "status": "absent",
        }
    state = load_completion(identifier, storage_root=storage)
    report = completion_status(
        state,
        original_successes=_embedded_original_successes(state),
    )
    evidence = state["parent_partial_evidence"]
    failed = evidence.get("failed_cases") if isinstance(evidence, dict) else None
    if not isinstance(failed, list):
        raise TypeError("Completion parent failure evidence is malformed.")
    original_failures = dict.fromkeys((str(item["batch_id"]) for item in state["targets"]), 0)
    for record in failed:
        if not isinstance(record, dict) or str(record.get("batch_id")) not in original_failures:
            raise ValueError("Completion parent failure membership is malformed.")
        original_failures[str(record["batch_id"])] += 1
    directory = completion_directory(completion_id_value, storage_root=storage)
    composite_path = directory / composite_service.RECEIPT_FILENAME
    lifecycle_path = directory / "completion_lifecycle.json"
    composite_state = "absent"
    package_state: dict[str, Any] = {"status": "absent", "count": 0}
    shard_state: dict[str, Any] = {"status": "absent", "count": 0}
    training_readiness = "absent"
    if composite_path.exists():
        if not composite_path.is_file() or composite_path.is_symlink():
            raise RuntimeError("Completion composite receipt is unsafe.")
        composite_service.load_composite_receipt(completion_id_value, storage_root=storage)
        composite_state = "complete"
    if lifecycle_path.exists():
        if not lifecycle_path.is_file() or lifecycle_path.is_symlink():
            raise RuntimeError("Completion lifecycle receipt is unsafe.")
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        if not isinstance(lifecycle, dict):
            raise TypeError("Completion lifecycle receipt must contain one object.")
        packages = lifecycle.get("packages")
        shards = lifecycle.get("shards")
        if (
            lifecycle.get("completion_id") != completion_id_value
            or lifecycle.get("parent_run_id") != state["parent_run_id"]
            or lifecycle.get("status") != "ready"
            or lifecycle.get("readiness") != "ready"
            or not isinstance(packages, list)
            or not isinstance(shards, list)
        ):
            raise RuntimeError("Completion lifecycle receipt is not ready or conflicts with its owner.")
        package_state = {"status": "ready", "count": len(packages)}
        shard_state = {"status": "ready", "count": len(shards)}
        training_readiness = "ready"
    report.update(
        {
            "parent_campaign": state["parent_run_id"],
            "parent_state": evidence.get("campaign_state") if isinstance(evidence, dict) else None,
            "original_failures": original_failures,
            "composite_source_state": composite_state,
            "package_state": package_state,
            "shard_state": shard_state,
            "training_readiness": training_readiness,
        }
    )
    return report


def _candidate_points(
    state: Mapping[str, Any],
    parent_campaign: config_service.CampaignConfig,
) -> tuple[dict[str, int], dict[tuple[str, int], dict[str, list[float]]]]:
    """Resolve the unmaterialized deterministic candidate prefix for every batch."""
    seeds: dict[str, int] = {}
    points: dict[tuple[str, int], dict[str, list[float]]] = {}
    remaining_pool = int(state["pool_high_water"]) - len(state["candidates"])
    for batch in parent_campaign.batches:
        if batch.seed_base is None:
            raise ValueError(f"Parent batch {batch.batch_id!r} has no executable sampling seed.")
        seed = seeding_service.derive_seed(
            batch.seed_base,
            "supplemental_replacement_pool",
            state["completion_id"],
            batch.batch_id,
        )
        seeds[batch.batch_id] = seed
        plans = batch.scientific_values["sampling"]["blocks"]
        blocks = tuple(plans)
        parameters = {block: tuple(plan["parameters"]) for block, plan in plans.items()}
        existing_batch_candidates = sum(candidate.get("batch_id") == batch.batch_id for candidate in state["candidates"])
        for batch_ordinal in range(existing_batch_candidates + 1, existing_batch_candidates + remaining_pool + 1):
            points[(batch.batch_id, batch_ordinal)] = supplemental_unit_point(
                registry=batch.scientific_values["material"]["parameter_registry"],
                blocks=blocks,
                block_parameters=parameters,
                seed=seed,
                ordinal=batch_ordinal,
            )
    return seeds, points


def _candidate_state_from_campaign(manifest: Mapping[str, Any]) -> str:
    """Project one ordinary synthetic campaign state into completion accounting."""
    state = manifest.get("state")
    if state == "complete":
        return "successful"
    if state in {"completed_with_failures", "failure_threshold_reached"}:
        return "failed"
    if state == "license_blocked":
        return "retry"
    if state in {
        "ready",
        "submitting",
        "active",
        "submission_failed",
        "submission_unknown",
        "scheduler_unknown",
        "cancel_requested",
        "force_cancel_requested",
    }:
        return "active"
    raise ValueError(f"Synthetic replacement campaign has an unsupported state: {state!r}.")


def _candidate_terminal_evidence(
    run_id: str,
    *,
    candidate_state: str,
    storage_root: Path | str | None,
) -> dict[str, Any]:
    """Bind one terminal synthetic result to its exact manifest and local failure class."""
    from . import generation_campaign as campaign_service
    from .publication import generation_publication_campaign_evidence as campaign_evidence

    if candidate_state not in {"successful", "failed"}:
        raise ValueError("Terminal execution evidence requires a successful or failed replacement state.")
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    commit = source_service.validate_git_commit(manifest.get("git_commit"))
    path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Replacement campaign-run manifest is missing or unsafe: {path}.")
    failure_counts = {"solver_failed": 0, "technical_runtime_timed_out": 0}
    if candidate_state == "failed":
        status = campaign_service.campaign_status(run_id, storage_root=storage_root, query_scheduler=True)
        observed = status.get("failure_counts")
        if status.get("campaign_run_id") != run_id or not isinstance(observed, Mapping):
            raise RuntimeError("Failed replacement lacks authoritative campaign failure accounting.")
        failure_counts = {
            "solver_failed": _nonnegative_int(observed.get("solver_failed"), label="replacement solver_failed"),
            "technical_runtime_timed_out": _nonnegative_int(
                observed.get("technical_runtime_timed_out"),
                label="replacement technical_runtime_timed_out",
            ),
        }
    contribution = sum(failure_counts.values())
    if contribution > 1:
        raise RuntimeError("One-case replacement produced impossible failure-circuit accounting.")
    return {
        "campaign_run_manifest_sha256": common.serialization.file_sha256(path),
        "git_commit": commit,
        "failure_counts": failure_counts,
        "failure_circuit_contribution": contribution,
    }


def completion_status(
    state: Mapping[str, Any],
    *,
    original_successes: Mapping[str, int],
) -> dict[str, Any]:
    """Return one bounded completion status without treating reservations as success."""
    deficits = uncovered_deficits(state, original_successes=original_successes)
    reserving = sum(item.get("state") in _RESERVING_STATES for item in state["candidates"])
    successes = sum(item.get("state") == "successful" for item in state["candidates"])
    failures = sum(item.get("state") == "failed" for item in state["candidates"])
    attempted = sum(item.get("state") != "planned" for item in state["candidates"])
    materialized = sum(isinstance(item.get("materialization"), dict) for item in state["candidates"])
    pool = int(state["pool_high_water"])
    circuit_open = bool(state["failure_circuit"]["open"])
    if circuit_open:
        status = "failure_circuit_open"
    elif not any(deficits.values()):
        status = "complete"
    elif len(state["candidates"]) >= pool and reserving == 0:
        status = "pool_exhausted"
    else:
        status = "active"
    return {
        "completion_id": state["completion_id"],
        "parent_run_id": state["parent_run_id"],
        "status": status,
        "replacement_pool_size": pool,
        "pool_high_water": pool,
        "allocated_candidates": len(state["candidates"]),
        "materialized_candidates": materialized,
        "attempted_candidates": attempted,
        "attempts": attempted,
        "running_candidates": sum(item.get("state") == "active" for item in state["candidates"]),
        "pending_candidates": sum(item.get("state") in {"planned", "retry", "replay"} for item in state["candidates"]),
        "unattempted_candidates": pool - attempted,
        "remaining_attempt_capacity": pool - attempted,
        "successful_replacements": successes,
        "failed_replacements": failures,
        "reserved_candidates": reserving,
        "failure_policy": copy.deepcopy(state["failure_policy"]),
        "failure_circuit": copy.deepcopy(state["failure_circuit"]),
        "original_successes": dict(original_successes),
        "current_successes": {
            batch_id: int(target["target_successes"]) - int(deficits[batch_id])
            for batch_id, target in ((item["batch_id"], item) for item in state["targets"])
        },
        "success_deficits": deficits,
        "next_command": (
            None
            if status in {"complete", "failure_circuit_open"}
            else (
                f"run <CONFIG> --replacement-pool-size {pool + 1}" if status == "pool_exhausted" else f"run <CONFIG> --replacement-pool-size {pool}"
            )
        ),
    }


def _advance_completion_campaigns_unlocked(
    state: dict[str, Any],
    *,
    parent_campaign: config_service.CampaignConfig,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Advance supplemental candidates exclusively through the normal campaign feeder."""
    from . import generation_campaign as campaign_service
    from .publication import generation_publication_campaign_evidence as campaign_evidence

    original_successes, failed_case_ids = parent_partial_counts(
        parent_campaign,
        state["parent_partial_evidence"],
        parent_run_id=state["parent_run_id"],
    )
    transitions: dict[str, str] = {}
    transition_evidence: dict[str, dict[str, Any]] = {}
    for candidate in state["candidates"]:
        if candidate.get("state") in {"successful", "failed"}:
            continue
        materialization = candidate.get("materialization")
        if not isinstance(materialization, dict):
            continue
        run_id = str(materialization["campaign_run_id"])
        path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
        if not path.exists():
            continue
        campaign = record_candidate_materialization(
            state,
            candidate_id=str(candidate["candidate_id"]),
            parent_campaign=parent_campaign,
            git_commit=git_commit,
            storage_root=storage_root,
        )
        manifest = campaign_service.feed_campaign(
            run_id,
            storage_root=storage_root,
            resolved_campaign=campaign,
        )
        candidate_id = str(candidate["candidate_id"])
        next_state = _candidate_state_from_campaign(manifest)
        transitions[candidate_id] = next_state
        if next_state in {"successful", "failed"}:
            transition_evidence[candidate_id] = _candidate_terminal_evidence(
                run_id,
                candidate_state=next_state,
                storage_root=storage_root,
            )
    if transitions:
        reconcile_wave(
            state,
            candidate_states=transitions,
            candidate_terminal_evidence=transition_evidence,
            storage_root=storage_root,
        )
    if not any(wave["state"] != "terminal" for wave in state["waves"]):
        seeds, points = _candidate_points(state, parent_campaign)
        allocate_next_wave(
            state,
            original_successes=original_successes,
            failed_case_ids=failed_case_ids,
            sampling_seed=seeds,
            unit_points=points,
            storage_root=storage_root,
        )
    active_candidate_ids = {candidate_id for wave in state["waves"] if wave["state"] != "terminal" for candidate_id in wave["candidate_ids"]}
    launch_transitions: dict[str, str] = {}
    launch_evidence: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(active_candidate_ids):
        candidate = _candidate_record(state, candidate_id)
        if candidate.get("state") in {"successful", "failed"}:
            continue
        campaign = record_candidate_materialization(
            state,
            candidate_id=candidate_id,
            parent_campaign=parent_campaign,
            git_commit=git_commit,
            storage_root=storage_root,
        )
        manifest = campaign_service.submit_campaign(
            campaign,
            git_commit=git_commit,
            storage_root=storage_root,
        )
        next_state = _candidate_state_from_campaign(manifest)
        launch_transitions[candidate_id] = next_state
        if next_state in {"successful", "failed"}:
            run_id = str(candidate["materialization"]["campaign_run_id"])
            launch_evidence[candidate_id] = _candidate_terminal_evidence(
                run_id,
                candidate_state=next_state,
                storage_root=storage_root,
            )
    if launch_transitions:
        reconcile_wave(
            state,
            candidate_states=launch_transitions,
            candidate_terminal_evidence=launch_evidence,
            storage_root=storage_root,
        )
    return completion_status(state, original_successes=original_successes)


def advance_completion_campaigns(
    state: dict[str, Any],
    *,
    parent_campaign: config_service.CampaignConfig,
    git_commit: str,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Serialize reload, reconciliation, and submission for one completion owner."""
    identifier = common.paths.validate_logical_name(state.get("completion_id"), label="completion_id")
    directory = completion_directory(identifier, storage_root=storage_root)
    with common.locking.exclusive_file_lock(directory / "completion.lock", blocking=True):
        durable = load_completion(identifier, storage_root=storage_root)
        result = _advance_completion_campaigns_unlocked(
            durable,
            parent_campaign=parent_campaign,
            git_commit=git_commit,
            storage_root=storage_root,
        )
        state.clear()
        state.update(copy.deepcopy(durable))
        return result
