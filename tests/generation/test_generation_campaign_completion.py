# ruff: noqa: S101, D103, SLF001
"""Focused contracts for bounded deterministic campaign completion."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import common, generation
from src.generation import completion
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence


def _state(tmp_path: Path, pool: int = 4, maximum_failed_cases: int = 1) -> dict[str, Any]:
    return completion.create_completion(
        parent_run_id="parent__abc",
        parent_partial_evidence={"successful": {"batch-a": ["case_0001"]}},
        parent_partial_sha256="a" * 64,
        targets=(completion.CompletionTarget("batch-a", 3),),
        replacement_pool_size=pool,
        maximum_failed_cases=maximum_failed_cases,
        storage_root=tmp_path,
    )


def _deficit_state(
    tmp_path: Path,
    deficits: dict[str, int],
    *,
    pool: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, tuple[str, ...]], dict[tuple[str, int], dict[str, list[float]]]]:
    targets = {batch_id: max(1, deficit) for batch_id, deficit in deficits.items()}
    state = completion.create_completion(
        parent_run_id="parent__deficits",
        parent_partial_evidence={"owner": "test"},
        parent_partial_sha256="7" * 64,
        targets=tuple(completion.CompletionTarget(batch_id, target) for batch_id, target in targets.items()),
        replacement_pool_size=pool,
        maximum_failed_cases=10,
        storage_root=tmp_path,
    )
    original = {batch_id: targets[batch_id] - deficit for batch_id, deficit in deficits.items()}
    failures = {batch_id: (f"case_{index:04d}",) for index, batch_id in enumerate(deficits, start=1)}
    points = {
        (batch_id, ordinal): {"thermal": [index / 10 + ordinal / 1000]}
        for index, batch_id in enumerate(deficits, start=1)
        for ordinal in range(1, pool + 1)
    }
    return state, original, failures, points


def _candidate_counts(state: dict[str, Any], candidate_ids: tuple[str, ...]) -> dict[str, int]:
    counts = dict.fromkeys((str(target["batch_id"]) for target in state["targets"]), 0)
    selected = set(candidate_ids)
    for candidate in state["candidates"]:
        if candidate["candidate_id"] in selected:
            counts[str(candidate["batch_id"])] += 1
    return counts


def _test_materialization(state: dict[str, Any], candidate_id: str) -> None:
    """Attach minimal identity-valid materialization for low-level state tests."""
    candidate = next(item for item in state["candidates"] if item["candidate_id"] == candidate_id)
    payload = {
        "schema_kind": "generation_supplemental_campaign_source",
        "schema_version": 1,
        "design": {
            "completion_id": state["completion_id"],
            "candidate_id": candidate_id,
        },
        "synthetic_campaign_id": f"campaign_{candidate['ordinal']}",
        "synthetic_campaign_digest": f"{candidate['ordinal']:064x}",
        "synthetic_batch_id": f"batch_{candidate['ordinal']}",
        "synthetic_case_index": int(candidate["ordinal"]),
    }
    extension = {**payload, "payload_sha256": common.serialization.canonical_json_sha256(payload)}
    candidate["materialization"] = {
        "campaign_run_id": f"replacement_run_{candidate['ordinal']}",
        "extension": extension,
        "batch_id": payload["synthetic_batch_id"],
        "case_id": f"case_{candidate['ordinal']:04d}",
        "case_index": int(candidate["ordinal"]),
        "simulation_science_id": candidate["science_id"],
        "git_commit": "d" * 40,
    }


def _terminal_evidence(*, solver_failed: int = 0, timed_out: int = 0) -> dict[str, Any]:
    return {
        "campaign_run_manifest_sha256": "f" * 64,
        "git_commit": "d" * 40,
        "failure_counts": {
            "solver_failed": solver_failed,
            "technical_runtime_timed_out": timed_out,
        },
        "failure_circuit_contribution": solver_failed + timed_out,
    }


def _reconcile_terminal(
    state: dict[str, Any],
    candidate_id: str,
    candidate_state: str,
    *,
    storage_root: Path,
    solver_failed: int = 0,
    timed_out: int = 0,
) -> None:
    if "materialization" not in next(item for item in state["candidates"] if item["candidate_id"] == candidate_id):
        _test_materialization(state, candidate_id)
    completion.reconcile_wave(
        state,
        candidate_states={candidate_id: candidate_state},
        candidate_terminal_evidence={
            candidate_id: _terminal_evidence(solver_failed=solver_failed, timed_out=timed_out),
        },
        storage_root=storage_root,
    )


def test_completion_status_if_present_is_read_only_and_rejects_unsafe_state(tmp_path: Path) -> None:
    identifier = "completion__" + "0" * 24
    report = completion.completion_status_for_id(
        identifier,
        storage_root=tmp_path,
        if_present=True,
    )
    assert report == {
        "schema_kind": "generation_campaign_completion_status",
        "schema_version": 1,
        "completion_id": identifier,
        "status": "absent",
    }

    state_path = completion.completion_directory(identifier, storage_root=tmp_path) / "completion.json"
    state_path.parent.mkdir(parents=True)
    state_path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="unsafe"):
        completion.completion_status_for_id(
            identifier,
            storage_root=tmp_path,
            if_present=True,
        )

    state_path.unlink()
    existing_target = tmp_path / "existing-completion.json"
    existing_target.write_text("{}\n", encoding="utf-8")
    state_path.symlink_to(existing_target)
    with pytest.raises(RuntimeError, match="unsafe"):
        completion.completion_status_for_id(
            identifier,
            storage_root=tmp_path,
            if_present=True,
        )


def test_completion_pool_is_monotonic_and_parent_bound(tmp_path: Path) -> None:
    initial_pool = 2
    state = _state(tmp_path, pool=initial_pool)
    wave = completion.allocate_next_wave(
        state,
        original_successes={"batch-a": 1},
        failed_case_ids={"batch-a": ("case_0002",)},
        sampling_seed=17,
        unit_points={
            ("batch-a", 1): {"thermal": [0.1]},
            ("batch-a", 2): {"thermal": [0.2]},
        },
        storage_root=tmp_path,
    )
    assert wave is not None
    persisted = json.loads(json.dumps(state))
    resumed = completion.create_completion(
        parent_run_id="parent__abc",
        parent_partial_evidence={"successful": {"batch-a": ["case_0001"]}},
        parent_partial_sha256="a" * 64,
        targets=(completion.CompletionTarget("batch-a", 3),),
        replacement_pool_size=None,
        maximum_failed_cases=1,
        storage_root=tmp_path,
    )
    assert resumed == persisted
    assert resumed["pool_high_water"] == initial_pool
    assert [candidate["candidate_id"] for candidate in resumed["candidates"]] == list(wave.candidate_ids)
    assert completion.extend_pool(resumed, replacement_pool_size=4)
    with pytest.raises(ValueError, match="high-water"):
        completion.ensure_pool_capacity(resumed, replacement_pool_size=1)
    with pytest.raises(ValueError, match="collides"):
        completion.create_completion(
            parent_run_id="parent__abc",
            parent_partial_evidence={"successful": {}},
            parent_partial_sha256="a" * 64,
            targets=(completion.CompletionTarget("batch-a", 3),),
            replacement_pool_size=4,
            storage_root=tmp_path,
        )

    with pytest.raises(RuntimeError, match="requires --replacement-pool-size"):
        completion.create_completion(
            parent_run_id="parent__missing",
            parent_partial_evidence={"successful": {}},
            parent_partial_sha256="b" * 64,
            targets=(completion.CompletionTarget("batch-a", 1),),
            replacement_pool_size=None,
            storage_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("deficits", "pool", "expected"),
    [
        pytest.param({"lentil": 1, "chickpea": 0, "kidney_bean": 0}, 20, {"lentil": 1, "chickpea": 0, "kidney_bean": 0}, id="one-material-slot"),
        pytest.param({"lentil": 3, "chickpea": 1, "kidney_bean": 4}, 20, {"lentil": 3, "chickpea": 1, "kidney_bean": 4}, id="three-one-four"),
        pytest.param({"lentil": 2, "chickpea": 0, "kidney_bean": 0}, 20, {"lentil": 2, "chickpea": 0, "kidney_bean": 0}, id="large-reserve-pool"),
    ],
)
def test_round_allocates_exact_per_material_deficits(
    tmp_path: Path,
    deficits: dict[str, int],
    pool: int,
    expected: dict[str, int],
) -> None:
    state, original, failures, points = _deficit_state(tmp_path, deficits, pool=pool)
    wave = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert wave is not None
    assert _candidate_counts(state, wave.candidate_ids) == expected
    assert len(wave.candidate_ids) == sum(expected.values())
    assert len(state["candidates"]) == sum(expected.values())


@pytest.mark.parametrize(("deficit", "active", "expected_new"), [(1, 1, 0), (3, 2, 1)])
def test_active_same_material_candidates_reserve_exact_slots(
    tmp_path: Path,
    deficit: int,
    active: int,
    expected_new: int,
) -> None:
    state, original, failures, points = _deficit_state(tmp_path, {"lentil": deficit}, pool=active)
    first = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert first is not None
    assert len(first.candidate_ids) == active
    completion.extend_pool(state, replacement_pool_size=20, storage_root=tmp_path)
    points.update({("lentil", ordinal): {"thermal": [ordinal / 100]} for ordinal in range(active + 1, 21)})
    second = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert (0 if second is None else len(second.candidate_ids)) == expected_new


@pytest.mark.parametrize(("failed_batch", "other_batch"), [("lentil", "chickpea"), ("chickpea", "lentil")])
def test_failed_replacement_reopens_only_its_material_slot(
    tmp_path: Path,
    failed_batch: str,
    other_batch: str,
) -> None:
    state, original, failures, points = _deficit_state(tmp_path, {"lentil": 1, "chickpea": 1}, pool=3)
    first = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert first is not None
    by_batch = {candidate["batch_id"]: candidate["candidate_id"] for candidate in state["candidates"]}
    _reconcile_terminal(state, by_batch[failed_batch], "failed", storage_root=tmp_path, solver_failed=1)
    _reconcile_terminal(state, by_batch[other_batch], "successful", storage_root=tmp_path)
    second = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert second is not None
    assert _candidate_counts(state, second.candidate_ids) == {failed_batch: 1, other_batch: 0}


def test_pool_smaller_than_deficit_preserves_canonical_material_order(tmp_path: Path) -> None:
    deficits = {"lentil": 3, "chickpea": 1, "kidney_bean": 4}
    pool = 5
    state, original, failures, points = _deficit_state(tmp_path, deficits, pool=pool)
    wave = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert wave is not None
    assert _candidate_counts(state, wave.candidate_ids) == {"lentil": 3, "chickpea": 1, "kidney_bean": 1}
    report = completion.completion_status(state, original_successes=original)
    assert report["pool_consumed"] == pool
    assert report["pool_remaining"] == 0
    assert report["pool_insufficient"] is True
    assert report["new_replacements_required"] == {"lentil": 0, "chickpea": 0, "kidney_bean": 3}
    assert report["next_command"] == "run <CONFIG> --replacement-pool-size 8"


def test_active_legacy_material_does_not_block_unrelated_deficit(tmp_path: Path) -> None:
    state, original, failures, points = _deficit_state(tmp_path, {"lentil": 1, "chickpea": 1}, pool=1)
    first = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert first is not None
    assert _candidate_counts(state, first.candidate_ids) == {"lentil": 1, "chickpea": 0}
    completion.extend_pool(state, replacement_pool_size=2, storage_root=tmp_path)
    second = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert second is not None
    assert _candidate_counts(state, second.candidate_ids) == {"lentil": 0, "chickpea": 1}


def test_completion_local_failure_circuit_stops_new_admission(tmp_path: Path) -> None:
    state = _state(tmp_path, pool=3, maximum_failed_cases=0)
    points = {("batch-a", ordinal): {"thermal": [ordinal / 10]} for ordinal in range(1, 4)}
    wave = completion.allocate_next_wave(
        state,
        original_successes={"batch-a": 1},
        failed_case_ids={"batch-a": ("case_0002",)},
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert wave is not None
    _reconcile_terminal(
        state,
        wave.candidate_ids[0],
        "failed",
        storage_root=tmp_path,
        timed_out=1,
    )
    assert completion.completion_status(state, original_successes={"batch-a": 1})["status"] == "failure_circuit_open"
    assert (
        completion.allocate_next_wave(
            state,
            original_successes={"batch-a": 1},
            failed_case_ids={"batch-a": ("case_0002",)},
            sampling_seed=17,
            unit_points=points,
            storage_root=tmp_path,
        )
        is None
    )


def test_timeout_counts_toward_circuit_while_replay_keeps_slot_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timed_out = {
        "state": "failed",
        "classified_state": "timed_out",
        "failure_stage": "solver",
        "temporary_license_retry": None,
        "replay_eligible": False,
    }
    replayable = {
        "state": "failed",
        "classified_state": "conversion_failed",
        "temporary_license_retry": None,
        "replay_eligible": True,
    }
    assert completion._candidate_state_from_case_status(timed_out) == "failed"
    assert completion._candidate_state_from_case_status(replayable) == "replay"

    run_id = "replacement_timeout__0123456789abcdef"
    manifest_path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=tmp_path)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: {"git_commit": "d" * 40})
    evidence = completion._candidate_terminal_evidence(
        run_id,
        candidate_state="failed",
        case_status=timed_out,
        storage_root=tmp_path,
    )
    assert evidence["failure_counts"] == {
        "solver_failed": 0,
        "technical_runtime_timed_out": 1,
    }
    assert evidence["failure_circuit_contribution"] == 1

    state, original, failures, points = _deficit_state(tmp_path / "replay", {"lentil": 1}, pool=1)
    wave = completion.allocate_next_wave(
        state,
        original_successes=original,
        failed_case_ids=failures,
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path / "replay",
    )
    assert wave is not None
    completion.reconcile_wave(
        state,
        candidate_states={wave.candidate_ids[0]: "replay"},
        storage_root=tmp_path / "replay",
    )
    completion.extend_pool(state, replacement_pool_size=2, storage_root=tmp_path / "replay")
    points[("lentil", 2)] = {"thermal": [0.2]}
    assert (
        completion.allocate_next_wave(
            state,
            original_successes=original,
            failed_case_ids=failures,
            sampling_seed=17,
            unit_points=points,
            storage_root=tmp_path / "replay",
        )
        is None
    )
    assert completion.completion_status(state, original_successes=original)["reserved_active_by_batch"] == {"lentil": 1}


def test_candidate_prefix_is_deterministic_and_collision_is_rejected(tmp_path: Path) -> None:
    state = _state(tmp_path, pool=2)
    points = {("batch-a", ordinal): {"thermal": [0.25]} for ordinal in (1, 2)}
    wave = completion.allocate_next_wave(
        state,
        original_successes={"batch-a": 1},
        failed_case_ids={"batch-a": ("case_0002",)},
        sampling_seed=3,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert wave is not None
    first = state["candidates"][0]
    assert first["provenance"]["candidate_ordinal"] == 1
    assert first["candidate_id"] != first["science_id"]
    _reconcile_terminal(state, wave.candidate_ids[0], "failed", storage_root=tmp_path)
    state["pool_high_water"] = 3
    state["candidates"].append(dict(first))
    with pytest.raises(FileExistsError, match="collision"):
        completion.allocate_next_wave(
            state,
            original_successes={"batch-a": 1},
            failed_case_ids={"batch-a": ("case_0002",)},
            sampling_seed=3,
            unit_points={("batch-a", 3): {"thermal": [0.5]}},
            storage_root=tmp_path,
        )


def test_supplemental_points_are_sobol_prefix_stable() -> None:
    registry = {
        "temperature": {"kind": "interval", "minimum": 0.0, "maximum": 1.0, "sampling_block": "thermal"},
    }
    first = completion.supplemental_unit_point(
        registry=registry, blocks=("thermal",), block_parameters={"thermal": ("temperature",)}, seed=11, ordinal=1
    )
    third = completion.supplemental_unit_point(
        registry=registry, blocks=("thermal",), block_parameters={"thermal": ("temperature",)}, seed=11, ordinal=3
    )
    assert len(first["thermal"]) == 1
    assert third["thermal"] == pytest.approx(
        completion.supplemental_unit_point(
            registry=registry, blocks=("thermal",), block_parameters={"thermal": ("temperature",)}, seed=11, ordinal=3
        )["thermal"]
    )
    assert first["thermal"] == pytest.approx(
        completion.supplemental_unit_point(
            registry=registry, blocks=("thermal",), block_parameters={"thermal": ("temperature",)}, seed=11, ordinal=1
        )["thermal"]
    )


def test_batch_local_candidate_prefix_survives_pool_extension(tmp_path: Path) -> None:
    def allocate_b(root: Path, *, initial_pool: int) -> dict[str, Any]:
        state = completion.create_completion(
            parent_run_id="parent__two_batches",
            parent_partial_evidence={"owner": "test"},
            parent_partial_sha256="b" * 64,
            targets=(
                completion.CompletionTarget("batch-a", 1),
                completion.CompletionTarget("batch-b", 1),
            ),
            replacement_pool_size=initial_pool,
            maximum_failed_cases=1,
            storage_root=root,
        )
        points = {
            ("batch-a", 1): {"thermal": [0.11]},
            ("batch-a", 2): {"thermal": [0.12]},
            ("batch-b", 1): {"thermal": [0.21]},
            ("batch-b", 2): {"thermal": [0.22]},
        }
        first = completion.allocate_next_wave(
            state,
            original_successes={"batch-a": 0, "batch-b": 0},
            failed_case_ids={"batch-a": ("case_a",), "batch-b": ("case_b",)},
            sampling_seed={"batch-a": 7, "batch-b": 9},
            unit_points=points,
            storage_root=root,
        )
        assert first is not None
        if initial_pool == 1:
            _reconcile_terminal(state, first.candidate_ids[0], "successful", storage_root=root)
            completion.extend_pool(state, replacement_pool_size=2, storage_root=root)
            second = completion.allocate_next_wave(
                state,
                original_successes={"batch-a": 0, "batch-b": 0},
                failed_case_ids={"batch-a": ("case_a",), "batch-b": ("case_b",)},
                sampling_seed={"batch-a": 7, "batch-b": 9},
                unit_points=points,
                storage_root=root,
            )
            assert second is not None
            candidate_id = second.candidate_ids[0]
        else:
            assert len(first.candidate_ids) == initial_pool
            candidate_id = first.candidate_ids[1]
        return next(item for item in state["candidates"] if item["candidate_id"] == candidate_id)

    direct = allocate_b(tmp_path / "direct", initial_pool=2)
    extended = allocate_b(tmp_path / "extended", initial_pool=1)
    assert direct["batch_id"] == extended["batch_id"] == "batch-b"
    assert direct["batch_ordinal"] == extended["batch_ordinal"] == 1
    assert direct["provenance"]["unit_point"] == extended["provenance"]["unit_point"] == {"thermal": [0.21]}
    assert direct["candidate_id"] == extended["candidate_id"]


def _partial_evidence(
    campaign: Any,
    *,
    failed_batch_id: str,
    failed_case_index: int,
    all_failed: bool = False,
) -> dict[str, Any]:
    """Return exact test-owned partial membership with one terminal failure."""
    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for batch in campaign.batches:
        for case_index in batch.case_indices:
            is_failed = all_failed or (batch.batch_id == failed_batch_id and case_index == failed_case_index)
            record = {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "case_id": batch.case_id(case_index),
                "case_index": case_index,
                "state": "failed" if is_failed else "successful",
                "classified_state": "solver_failed" if is_failed else "successful",
            }
            (failed if is_failed else successful).append(record)
    return {
        "schema_kind": "generation_campaign_partial",
        "schema_version": 1,
        "campaign_run_id": "parent__0123456789abcdef",
        "campaign_id": campaign.campaign_id,
        "successful_cases": successful,
        "failed_cases": failed,
    }


def test_all_failed_parent_can_be_completed_entirely_by_replacements(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=3)
    parent = generation.cases.config.load_campaign_config(config_path)
    first_batch = parent.batches[0]
    partial = _partial_evidence(
        parent,
        failed_batch_id=first_batch.batch_id,
        failed_case_index=first_batch.case_indices[0],
        all_failed=True,
    )
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="9" * 64,
        replacement_pool_size=parent.total_case_count,
        storage_root=tmp_path,
    )
    startup = completion.completion_status_for_id(
        state["completion_id"],
        storage_root=tmp_path,
        parent_campaign=parent,
    )
    assert startup["target_batches"] == [
        {
            **startup["batch_accounting"][0],
            "batch_name": first_batch.batch_name,
            "material_family": first_batch.material_family,
            "material_role": first_batch.material_role,
            "evaluation_regime": first_batch.evaluation_regime,
            "sampling_regime": first_batch.sampling_regime,
        }
    ]
    assert startup["new_replacements_required"] == {first_batch.batch_id: parent.total_case_count}
    assert startup["pool_high_water"] == parent.total_case_count
    assert startup["pool_consumed"] == 0
    assert startup["execution"] == completion.completion_execution_projection(parent)
    assert startup["package_state"] == {"status": "absent", "count": 0}
    assert startup["shard_state"] == {"status": "absent", "count": 0}
    assert startup["training_readiness"] == "absent"
    assert startup["recoverable_stale_artifacts"] == 0
    assert startup["conflicting_artifacts"] == 0
    assert startup["normal_run_requires_replacement_pool"] is False
    assert startup["next_command"] == "run <CONFIG>"
    original_successes, failures = completion.parent_partial_counts(
        parent,
        partial,
        parent_run_id=partial["campaign_run_id"],
    )
    assert original_successes == dict.fromkeys((batch.batch_id for batch in parent.batches), 0)
    seeds, points = completion._candidate_points(state, parent)
    wave = completion.allocate_next_wave(
        state,
        original_successes=original_successes,
        failed_case_ids=failures,
        sampling_seed=seeds,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert wave is not None
    assert len(wave.candidate_ids) == parent.total_case_count
    round_campaign = completion.record_replacement_round_materialization(
        state,
        candidate_ids=wave.candidate_ids,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    assert round_campaign.total_case_count == parent.total_case_count
    assert {candidate["materialization"]["campaign_run_id"] for candidate in state["candidates"]} == {
        generation.campaign.campaign_run_id(round_campaign, git_commit="d" * 40)
    }
    for candidate_id in wave.candidate_ids:
        _reconcile_terminal(state, candidate_id, "successful", storage_root=tmp_path)
    complete = completion.completion_status_for_id(state["completion_id"], storage_root=tmp_path)
    assert complete["status"] == "complete"
    assert complete["new_comsol_work_required"] == 0
    assert complete["normal_run_requires_replacement_pool"] is False
    assert complete["next_command"] == "run <CONFIG>"
    assert complete["normal_run_next_operation"] == ("collect successful replacement publications and build or recover the completion composite")

    composite_path = completion.completion_directory(state["completion_id"], storage_root=tmp_path) / "completion_composite.json"
    composite_path.write_text('{"completion_id":', encoding="utf-8")
    recoverable = completion.completion_status_for_id(state["completion_id"], storage_root=tmp_path)
    assert recoverable["composite_source_state"] == "recoverable"
    assert recoverable["recoverable_stale_artifacts"] == 1
    assert recoverable["conflicting_artifacts"] == 0

    common.serialization.atomic_write_json(
        composite_path,
        {
            "schema_kind": "generation_completion_composite",
            "schema_version": 1,
            "completion_id": "completion__different",
            "parent_run_id": state["parent_run_id"],
        },
    )
    conflicting = completion.completion_status_for_id(state["completion_id"], storage_root=tmp_path)
    assert conflicting["composite_source_state"] == "conflicting"
    assert conflicting["recoverable_stale_artifacts"] == 0
    assert conflicting["conflicting_artifacts"] == 1
    assert conflicting["normal_run_next_operation"] == "resolve conflicting finalization artifact identity"


def test_supplemental_materialization_has_new_science_and_portable_identity(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=3)
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(parent, failed_batch_id=batch.batch_id, failed_case_index=batch.case_indices[-1])
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="b" * 64,
        replacement_pool_size=1,
        storage_root=tmp_path,
    )
    seeds, points = completion._candidate_points(state, parent)
    original_successes, failures = completion.parent_partial_counts(
        parent,
        partial,
        parent_run_id=partial["campaign_run_id"],
    )
    wave = completion.allocate_next_wave(
        state,
        original_successes=original_successes,
        failed_case_ids=failures,
        sampling_seed=seeds,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert wave is not None
    synthetic = completion.materialize_candidate_campaign(
        state_identity=state,
        parent_campaign=parent,
        candidate=state["candidates"][0],
    )
    replacement = synthetic.batches[0]
    replacement_index = replacement.case_indices[0]
    assert replacement.batch_id != batch.batch_id
    assert replacement.scientific_config_digest != batch.scientific_config_digest
    assert replacement.case_id(replacement_index) not in {batch.case_id(index) for index in batch.case_indices}
    design = replacement.scientific_values["supplemental_design"]
    assert design["physical_values"] == generation.cases.sampling.sample_case(replacement, replacement_index).values
    assert design["physical_values_sha256"] == common.serialization.canonical_json_sha256({"physical_values": design["physical_values"]})

    extension = completion.synthetic_manifest_extension(synthetic)
    manifest = {
        "campaign_config": config_path.resolve().relative_to(common.paths.get_project_root()).as_posix(),
        "synthetic_completion": extension,
        "campaign_id": synthetic.campaign_id,
        "campaign_digest": synthetic.campaign_digest,
        "execution_config_digest": common.serialization.canonical_json_sha256(synthetic.execution_values),
        "selected_batch_names": [replacement.batch_name],
        "dataset_packages": [],
    }
    restored = completion.campaign_from_synthetic_manifest(manifest)
    assert restored == synthetic
    assert extension is not None
    extension["synthetic_case_index"] += 1
    with pytest.raises(ValueError, match="identity"):
        completion.campaign_from_synthetic_manifest(manifest)


def test_completion_controller_uses_one_normal_multi_case_round(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_count = 3
    max_admission_cases = 2
    max_running_cases = 3
    cores_per_case = 5
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=case_count,
        max_admission_cases=max_admission_cases,
        max_running_cases=max_running_cases,
        cores_per_case=cores_per_case,
    )
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(
        parent,
        failed_batch_id=batch.batch_id,
        failed_case_index=batch.case_indices[-1],
        all_failed=True,
    )
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="c" * 64,
        replacement_pool_size=20,
        storage_root=tmp_path,
    )
    calls: list[tuple[str, str]] = []
    submitted: dict[str, Any] = {}

    def status_for(campaign: Any, run_id: str, *, successful: bool) -> dict[str, Any]:
        case_state = "successful" if successful else "running"
        classified = "successful" if successful else "active"
        return {
            "campaign_run_id": run_id,
            "cases": [
                {
                    "batch_name": item.batch_name,
                    "batch_id": item.batch_id,
                    "case_id": item.case_id(item.case_indices[0]),
                    "case_index": item.case_indices[0],
                    "state": case_state,
                    "classified_state": classified,
                    "temporary_license_retry": None,
                    "replay_eligible": False,
                    "failure_stage": None,
                }
                for item in campaign.batches
            ],
        }

    def fake_submit(campaign: Any, *, git_commit: str, storage_root: Path) -> dict[str, Any]:
        run_id = generation.campaign.campaign_run_id(campaign, git_commit=git_commit)
        calls.append(("submit", run_id))
        submitted[run_id] = campaign
        manifest_path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {"state": "active"}

    def fake_status(run_id: str, *, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        return status_for(submitted[run_id], run_id, successful=False)

    def fake_resume(run_id: str, *, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        calls.append(("resume", run_id))
        return {
            "manifest": {"state": "complete"},
            "status": status_for(submitted[run_id], run_id, successful=True),
        }

    monkeypatch.setattr(generation.campaign, "submit_campaign", fake_submit)
    monkeypatch.setattr(generation.campaign, "campaign_status", fake_status)
    monkeypatch.setattr(generation.campaign, "resume_campaign_monitor_snapshot", fake_resume)
    monkeypatch.setattr(completion, "_candidate_terminal_evidence", lambda *_args, **_kwargs: _terminal_evidence())
    first = completion.advance_completion_campaigns(
        state,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    second = completion.advance_completion_campaigns(
        state,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    assert first["status"] == "active"
    assert second["status"] == "complete"
    assert [kind for kind, _run_id in calls] == ["submit", "resume"]
    assert len(submitted) == 1
    round_campaign = next(iter(submitted.values()))
    assert round_campaign.total_case_count == case_count
    assert len(round_campaign.batches) == case_count
    assert len({item.batch_name for item in round_campaign.batches}) == case_count
    assert round_campaign.execution_values["submission"] == {
        **parent.execution_values["submission"],
        "max_admission_cases": max_admission_cases,
        "max_running_cases": max_running_cases,
    }
    assert round_campaign.execution_values["cluster"]["cores_per_case"] == cores_per_case
    extension = completion.synthetic_manifest_extension(round_campaign)
    assert extension is not None
    assert extension["schema_kind"] == "generation_supplemental_round_source"
    assert len(extension["designs"]) == case_count
    restored = completion.campaign_from_synthetic_manifest(
        {
            "campaign_config": config_path.resolve().relative_to(common.paths.get_project_root()).as_posix(),
            "synthetic_completion": extension,
            "campaign_id": round_campaign.campaign_id,
            "campaign_digest": round_campaign.campaign_digest,
            "execution_config_digest": common.serialization.canonical_json_sha256(round_campaign.execution_values),
            "selected_batch_names": [item.batch_name for item in round_campaign.batches],
            "dataset_packages": [],
        }
    )
    assert restored == round_campaign
    assert len(state["candidates"]) == case_count
    assert len({candidate["materialization"]["campaign_run_id"] for candidate in state["candidates"]}) == 1


def test_persisted_unmaterialized_wave_recovers_as_normal_round(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
    )
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(
        parent,
        failed_batch_id=batch.batch_id,
        failed_case_index=batch.case_indices[0],
        all_failed=True,
    )
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="5" * 64,
        replacement_pool_size=1,
        storage_root=tmp_path,
    )
    original_successes, failures = completion.parent_partial_counts(
        parent,
        partial,
        parent_run_id=partial["campaign_run_id"],
    )
    seeds, points = completion._candidate_points(state, parent)
    persisted_wave = completion.allocate_next_wave(
        state,
        original_successes=original_successes,
        failed_case_ids=failures,
        sampling_seed=seeds,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert persisted_wave is not None
    candidate_id = persisted_wave.candidate_ids[0]
    assert "materialization" not in state["candidates"][0]
    submissions: list[Any] = []
    resumes: list[str] = []

    def fake_submit(campaign: Any, *, git_commit: str, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        assert git_commit == "d" * 40
        submissions.append(campaign)
        run_id = generation.campaign.campaign_run_id(campaign, git_commit=git_commit)
        manifest_path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {"state": "active"}

    def fake_status(run_id: str, *, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        campaign = submissions[0]
        assert run_id == generation.campaign.campaign_run_id(campaign, git_commit="d" * 40)
        replacement = campaign.batches[0]
        case_index = replacement.case_indices[0]
        return {
            "campaign_run_id": run_id,
            "cases": [
                {
                    "batch_id": replacement.batch_id,
                    "case_id": replacement.case_id(case_index),
                    "case_index": case_index,
                    "state": "running",
                    "classified_state": "active",
                    "temporary_license_retry": None,
                    "replay_eligible": False,
                }
            ],
        }

    def fake_resume(run_id: str, *, storage_root: Path) -> dict[str, Any]:
        resumes.append(run_id)
        return {
            "manifest": {"state": "active"},
            "status": fake_status(run_id, storage_root=storage_root),
        }

    monkeypatch.setattr(generation.campaign, "submit_campaign", fake_submit)
    monkeypatch.setattr(generation.campaign, "campaign_status", fake_status)
    monkeypatch.setattr(generation.campaign, "resume_campaign_monitor_snapshot", fake_resume)
    report = completion.advance_completion_campaigns(
        state,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    resumed_report = completion.advance_completion_campaigns(
        state,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )

    assert len(submissions) == 1
    run_id = generation.campaign.campaign_run_id(submissions[0], git_commit="d" * 40)
    assert resumes == [run_id]
    assert submissions[0].total_case_count == 1
    assert state["candidates"][0]["candidate_id"] == candidate_id
    assert state["candidates"][0]["state"] == "active"
    assert state["candidates"][0]["materialization"]["extension"]["schema_kind"] == "generation_supplemental_round_source"
    assert report["reserved_active_by_batch"] == {batch.batch_id: 1}
    assert resumed_report["reserved_active_by_batch"] == {batch.batch_id: 1}


def test_active_legacy_campaign_resumes_once_and_new_work_uses_round(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_count = 2
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=target_count,
        max_admission_cases=target_count,
    )
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(
        parent,
        failed_batch_id=batch.batch_id,
        failed_case_index=batch.case_indices[0],
        all_failed=True,
    )
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="8" * 64,
        replacement_pool_size=1,
        storage_root=tmp_path,
    )
    original_successes, failures = completion.parent_partial_counts(
        parent,
        partial,
        parent_run_id=partial["campaign_run_id"],
    )
    seeds, points = completion._candidate_points(state, parent)
    legacy_wave = completion.allocate_next_wave(
        state,
        original_successes=original_successes,
        failed_case_ids=failures,
        sampling_seed=seeds,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert legacy_wave is not None
    assert len(legacy_wave.candidate_ids) == 1
    legacy_campaign = completion.record_candidate_materialization(
        state,
        candidate_id=legacy_wave.candidate_ids[0],
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    legacy_run_id = generation.campaign.campaign_run_id(legacy_campaign, git_commit="d" * 40)
    legacy_manifest = campaign_evidence.campaign_run_manifest_path(legacy_run_id, storage_root=tmp_path)
    legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
    legacy_manifest.write_text("{}\n", encoding="utf-8")
    completion.extend_pool(state, replacement_pool_size=target_count, storage_root=tmp_path)
    legacy_id = legacy_wave.candidate_ids[0]
    legacy_materialization = dict(state["candidates"][0]["materialization"])
    calls: list[tuple[str, str]] = []
    submitted: dict[str, Any] = {}

    def fake_resume(run_id: str, *, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        assert run_id == legacy_run_id
        calls.append(("resume", run_id))
        return {
            "manifest": {"state": "license_blocked"},
            "status": {
                "campaign_run_id": run_id,
                "cases": [
                    {
                        "batch_id": legacy_materialization["batch_id"],
                        "case_id": legacy_materialization["case_id"],
                        "case_index": legacy_materialization["case_index"],
                        "state": "license_blocked",
                        "classified_state": "license_blocked",
                        "temporary_license_retry": {"next_retry_at": "later"},
                        "replay_eligible": False,
                    }
                ],
            },
        }

    def fake_submit(campaign: Any, *, git_commit: str, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        run_id = generation.campaign.campaign_run_id(campaign, git_commit=git_commit)
        assert run_id != legacy_run_id
        calls.append(("submit", run_id))
        submitted[run_id] = campaign
        return {"state": "active"}

    def fake_status(run_id: str, *, storage_root: Path) -> dict[str, Any]:
        assert storage_root == tmp_path
        campaign = submitted[run_id]
        replacement = campaign.batches[0]
        case_index = replacement.case_indices[0]
        return {
            "campaign_run_id": run_id,
            "cases": [
                {
                    "batch_id": replacement.batch_id,
                    "case_id": replacement.case_id(case_index),
                    "case_index": case_index,
                    "state": "running",
                    "classified_state": "active",
                    "temporary_license_retry": None,
                    "replay_eligible": False,
                }
            ],
        }

    monkeypatch.setattr(generation.campaign, "resume_campaign_monitor_snapshot", fake_resume)
    monkeypatch.setattr(generation.campaign, "submit_campaign", fake_submit)
    monkeypatch.setattr(generation.campaign, "campaign_status", fake_status)

    report = completion.advance_completion_campaigns(
        state,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )

    assert [kind for kind, _run_id in calls] == ["resume", "submit"]
    assert len(state["candidates"]) == target_count
    assert state["candidates"][0]["candidate_id"] == legacy_id
    assert state["candidates"][0]["state"] == "retry"
    assert state["candidates"][0]["materialization"] == legacy_materialization
    assert state["candidates"][1]["materialization"]["extension"]["schema_kind"] == "generation_supplemental_round_source"
    assert report["reserved_active_by_batch"] == {batch.batch_id: target_count}
    assert report["new_replacements_required"] == {batch.batch_id: 0}
    assert report["active_run_ids"] == [legacy_run_id, next(iter(submitted))]
    assert report["active_round_run_ids"] == [next(iter(submitted))]


def test_parent_resolution_is_structural_ambiguous_and_override_bound(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=3)
    parent = generation.cases.config.load_campaign_config(config_path)
    run_ids = ("parent__1111111111111111", "parent__2222222222222222")
    partials: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        directory = campaign_evidence.campaign_run_directory(run_id, storage_root=tmp_path)
        directory.mkdir(parents=True)
        (directory / "campaign_run.json").write_text(
            json.dumps({"campaign_digest": parent.campaign_digest}) + "\n",
            encoding="utf-8",
        )
        partial = _partial_evidence(
            parent,
            failed_batch_id=parent.batches[0].batch_id,
            failed_case_index=parent.batches[0].case_indices[-1],
        )
        partial["campaign_run_id"] = run_id
        partials[run_id] = partial
        common.serialization.atomic_write_json(directory / "campaign_partial.json", partial)

    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda run_id, **_kwargs: {
            "state": "completed_with_failures",
            "git_commit": "d" * 40,
            "campaign_run_id": run_id,
        },
    )
    monkeypatch.setattr(campaign_evidence, "campaign_from_manifest", lambda _manifest: parent)
    monkeypatch.setattr(
        generation.campaign,
        "read_partial_campaign_diagnostic_evidence",
        lambda run_id, **_kwargs: partials[run_id],
    )
    with pytest.raises(RuntimeError, match="Multiple structurally compatible"):
        completion.find_compatible_completion_parent(parent, storage_root=tmp_path, require_transferred=False)

    selected = completion.find_compatible_completion_parent(
        parent,
        parent_run_id=run_ids[0],
        storage_root=tmp_path,
        require_transferred=False,
    )
    assert selected["status"] == "compatible_partial"
    assert selected["parent_run_id"] == run_ids[0]
    assert sum(selected["success_deficits"].values()) == 1

    completion.create_completion(
        parent_run_id=run_ids[0],
        parent_partial_evidence=partials[run_ids[0]],
        parent_partial_sha256="e" * 64,
        targets=tuple(completion.CompletionTarget(batch.batch_id, len(batch.case_indices)) for batch in parent.batches),
        replacement_pool_size=1,
        storage_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="different immutable partial evidence"):
        completion.find_compatible_completion_parent(
            parent,
            parent_run_id=run_ids[0],
            storage_root=tmp_path,
            require_transferred=False,
        )


def test_parent_resolution_skips_stale_candidates_and_fails_closed_on_claimed_corruption(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore structurally stale runs but reject corruption after a digest match."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=3)
    parent = generation.cases.config.load_campaign_config(config_path)
    stale_run_id = "parent__1111111111111111"
    compatible_run_id = "parent__2222222222222222"
    for run_id, campaign_digest in (
        (stale_run_id, "0" * 64),
        (compatible_run_id, parent.campaign_digest),
    ):
        directory = campaign_evidence.campaign_run_directory(run_id, storage_root=tmp_path)
        directory.mkdir(parents=True)
        common.serialization.atomic_write_json(
            directory / "campaign_run.json",
            {"campaign_digest": campaign_digest},
        )

    admitted: list[str] = []

    def admit_compatible(run_id: str, **_kwargs: Any) -> tuple[dict[str, Any], Any]:
        admitted.append(run_id)
        return {"state": "complete", "git_commit": "d" * 40}, parent

    monkeypatch.setattr(completion, "_compatible_campaign_run", admit_compatible)
    selected = completion.find_compatible_completion_parent(
        parent,
        storage_root=tmp_path,
        require_transferred=False,
    )
    assert selected["status"] == "compatible_complete"
    assert selected["parent_run_id"] == compatible_run_id
    assert admitted == [compatible_run_id]

    def reject_corrupt(run_id: str, **_kwargs: Any) -> None:
        assert run_id == compatible_run_id
        message = "claimed-compatible campaign evidence is corrupt"
        raise RuntimeError(message)

    monkeypatch.setattr(completion, "_compatible_campaign_run", reject_corrupt)
    with pytest.raises(RuntimeError, match="claimed-compatible campaign evidence is corrupt"):
        completion.find_compatible_completion_parent(
            parent,
            storage_root=tmp_path,
            require_transferred=False,
        )


def _completed_replacement_state(
    generation_config_factory: Any,
    tmp_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=3)
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(
        parent,
        failed_batch_id=batch.batch_id,
        failed_case_index=batch.case_indices[-1],
    )
    parent_directory = campaign_evidence.campaign_run_directory(partial["campaign_run_id"], storage_root=tmp_path)
    parent_directory.mkdir(parents=True)
    partial_path = common.serialization.atomic_write_json(parent_directory / "campaign_partial.json", partial)
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256=common.serialization.file_sha256(partial_path),
        replacement_pool_size=2,
        storage_root=tmp_path,
    )
    original_successes, failures = completion.parent_partial_counts(parent, partial, parent_run_id=partial["campaign_run_id"])
    synthetics: dict[str, Any] = {}
    for candidate_state in ("failed", "successful"):
        seeds, points = completion._candidate_points(state, parent)
        wave = completion.allocate_next_wave(
            state,
            original_successes=original_successes,
            failed_case_ids=failures,
            sampling_seed=seeds,
            unit_points=points,
            storage_root=tmp_path,
        )
        assert wave is not None
        assert len(wave.candidate_ids) == 1
        candidate_id = wave.candidate_ids[0]
        synthetic = completion.record_candidate_materialization(
            state,
            candidate_id=candidate_id,
            parent_campaign=parent,
            git_commit="d" * 40,
            storage_root=tmp_path,
        )
        synthetics[candidate_id] = synthetic
        _reconcile_terminal(
            state,
            candidate_id,
            candidate_state,
            storage_root=tmp_path,
        )
    return parent, partial, state, synthetics


def test_transfer_plan_excludes_failed_replacements_and_binds_state(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    _parent, _partial, state, _synthetics = _completed_replacement_state(generation_config_factory, tmp_path)
    plan = completion.completion_transfer_plan(state["completion_id"], storage_root=tmp_path)
    assert [item["candidate_id"] for item in plan["replacement_campaigns"]] == [state["candidates"][1]["candidate_id"]]
    assert plan["replacement_campaigns"][0]["campaign_run_id"] == (state["candidates"][1]["materialization"]["campaign_run_id"])
    assert Path(plan["completion_state_path"]).is_file()
    assert plan["completion_state_sha256"] == common.serialization.file_sha256(Path(plan["completion_state_path"]))


def test_transfer_plan_groups_mixed_shared_round_and_excludes_failed_case(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    target_count = 2
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=target_count,
    )
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(
        parent,
        failed_batch_id=batch.batch_id,
        failed_case_index=batch.case_indices[0],
        all_failed=True,
    )
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="6" * 64,
        replacement_pool_size=target_count + 1,
        storage_root=tmp_path,
    )
    original_successes, failures = completion.parent_partial_counts(
        parent,
        partial,
        parent_run_id=partial["campaign_run_id"],
    )
    seeds, points = completion._candidate_points(state, parent)
    first = completion.allocate_next_wave(
        state,
        original_successes=original_successes,
        failed_case_ids=failures,
        sampling_seed=seeds,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert first is not None
    assert len(first.candidate_ids) == target_count
    first_campaign = completion.record_replacement_round_materialization(
        state,
        candidate_ids=first.candidate_ids,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    _reconcile_terminal(state, first.candidate_ids[0], "successful", storage_root=tmp_path)
    _reconcile_terminal(
        state,
        first.candidate_ids[1],
        "failed",
        storage_root=tmp_path,
        solver_failed=1,
    )
    seeds, points = completion._candidate_points(state, parent)
    second = completion.allocate_next_wave(
        state,
        original_successes=original_successes,
        failed_case_ids=failures,
        sampling_seed=seeds,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert second is not None
    assert len(second.candidate_ids) == 1
    second_campaign = completion.record_replacement_round_materialization(
        state,
        candidate_ids=second.candidate_ids,
        parent_campaign=parent,
        git_commit="d" * 40,
        storage_root=tmp_path,
    )
    _reconcile_terminal(state, second.candidate_ids[0], "successful", storage_root=tmp_path)

    plan = completion.completion_transfer_plan(state["completion_id"], storage_root=tmp_path)

    first_run = generation.campaign.campaign_run_id(first_campaign, git_commit="d" * 40)
    second_run = generation.campaign.campaign_run_id(second_campaign, git_commit="d" * 40)
    by_run = {item["campaign_run_id"]: item for item in plan["replacement_runs"]}
    assert set(by_run) == {first_run, second_run}
    assert by_run[first_run]["campaign_state"] == "completed_with_failures"
    assert by_run[first_run]["partial"] is True
    assert by_run[first_run]["terminal_batch_ids"] == [state["candidates"][0]["materialization"]["batch_id"]]
    assert by_run[second_run]["campaign_state"] == "complete"
    assert by_run[second_run]["partial"] is False
    assert [item["candidate_id"] for item in plan["replacement_campaigns"]] == [
        first.candidate_ids[0],
        second.candidate_ids[0],
    ]


def test_storage_composite_admits_parent_success_and_only_successful_replacement(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, partial, state, synthetics = _completed_replacement_state(generation_config_factory, tmp_path)
    successful = state["candidates"][1]
    replacement_run_id = successful["materialization"]["campaign_run_id"]
    synthetic = synthetics[successful["candidate_id"]]
    parent_manifest = {"kind": "parent", "git_commit": "d" * 40}
    replacement_manifest = {
        "kind": "replacement",
        "state": "complete",
        "git_commit": "d" * 40,
        "synthetic_completion": successful["materialization"]["extension"],
    }
    for run_id in (partial["campaign_run_id"], replacement_run_id):
        manifest_path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=tmp_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(f'{{"run_id": "{run_id}"}}\n', encoding="utf-8")
    successful["terminal_evidence"]["campaign_run_manifest_sha256"] = common.serialization.file_sha256(
        campaign_evidence.campaign_run_manifest_path(replacement_run_id, storage_root=tmp_path)
    )
    completion.save_completion(state, storage_root=tmp_path)
    monkeypatch.setattr(generation.campaign, "validate_partially_transferred_campaign", lambda *_args, **_kwargs: {})
    validated_replacements: list[str] = []
    monkeypatch.setattr(
        generation.campaign,
        "validate_transferred_campaign",
        lambda run_id, **_kwargs: validated_replacements.append(run_id) or {},
    )
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda run_id, **_kwargs: parent_manifest if run_id == partial["campaign_run_id"] else replacement_manifest,
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_from_manifest",
        lambda manifest: parent if manifest["kind"] == "parent" else synthetic,
    )
    parent_cases: list[int] = []
    monkeypatch.setattr(
        generation.runtime.batch,
        "admit_completed_case",
        lambda _batch, case_index, **_kwargs: parent_cases.append(case_index) or SimpleNamespace(case_id=f"parent-{case_index}"),
    )
    terminal_case = SimpleNamespace(case_id="replacement-success")
    monkeypatch.setattr(
        generation.runtime.batch,
        "admit_terminal_batch",
        lambda batch_storage_name, **_kwargs: SimpleNamespace(
            batch_id=synthetic.batches[0].batch_id,
            batch_storage_name=batch_storage_name,
            cases=(terminal_case,),
        ),
    )
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(
        generation.publication.completion_composite,
        "build_composite_receipt",
        fake_build,
    )
    receipt = completion.build_completion_composite(state["completion_id"], storage_root=tmp_path)
    assert receipt == {"status": "complete"}
    assert len(parent_cases) == len(partial["successful_cases"])
    assert validated_replacements == [replacement_run_id]
    assert len(captured["replacement_cases"]) == 1
    assert captured["replacement_cases"][0].case is terminal_case
    assert captured["targets"] == {batch.batch_id: len(batch.case_indices) for batch in parent.batches}
