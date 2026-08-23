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
    state = _state(tmp_path, pool=2)
    resumed = _state(tmp_path, pool=2)
    assert resumed == state
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


def test_waves_cover_only_current_deficit_and_resume_after_failure(tmp_path: Path) -> None:
    state = _state(tmp_path, pool=4)
    points = {("batch-a", ordinal): {"thermal": [ordinal / 10]} for ordinal in range(1, 5)}
    first = completion.allocate_next_wave(
        state,
        original_successes={"batch-a": 1},
        failed_case_ids={"batch-a": ("case_0002",)},
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert first is not None
    assert len(first.candidate_ids) == 1
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
    _reconcile_terminal(
        state,
        first.candidate_ids[0],
        "failed",
        storage_root=tmp_path,
        solver_failed=1,
    )
    assert state["failure_circuit"] == {"counted_failures": 1, "open": False}
    second = completion.allocate_next_wave(
        state,
        original_successes={"batch-a": 1},
        failed_case_ids={"batch-a": ("case_0002",)},
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert second is not None
    _reconcile_terminal(state, second.candidate_ids[0], "successful", storage_root=tmp_path)
    third = completion.allocate_next_wave(
        state,
        original_successes={"batch-a": 1},
        failed_case_ids={"batch-a": ("case_0002",)},
        sampling_seed=17,
        unit_points=points,
        storage_root=tmp_path,
    )
    assert third is not None
    assert completion.uncovered_deficits(state, original_successes={"batch-a": 1}) == {"batch-a": 1}


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
        _reconcile_terminal(state, first.candidate_ids[0], "successful", storage_root=root)
        if initial_pool == 1:
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
        return next(item for item in state["candidates"] if item["candidate_id"] == second.candidate_ids[0])

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
    original_successes, failures = completion.parent_partial_counts(
        parent,
        partial,
        parent_run_id=partial["campaign_run_id"],
    )
    assert original_successes == dict.fromkeys((batch.batch_id for batch in parent.batches), 0)
    for _ in range(parent.total_case_count):
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
        candidate_id = wave.candidate_ids[0]
        completion.record_candidate_materialization(
            state,
            candidate_id=candidate_id,
            parent_campaign=parent,
            git_commit="d" * 40,
            storage_root=tmp_path,
        )
        _reconcile_terminal(state, candidate_id, "successful", storage_root=tmp_path)
    assert completion.completion_status_for_id(state["completion_id"], storage_root=tmp_path)["status"] == "complete"


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


def test_completion_controller_reuses_submit_and_feed_without_overshoot(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=3)
    parent = generation.cases.config.load_campaign_config(config_path)
    batch = parent.batches[0]
    partial = _partial_evidence(parent, failed_batch_id=batch.batch_id, failed_case_index=batch.case_indices[-1])
    state = completion.create_completion_from_partial(
        parent_campaign=parent,
        parent_run_id=partial["campaign_run_id"],
        parent_partial_evidence=partial,
        parent_partial_sha256="c" * 64,
        replacement_pool_size=1,
        storage_root=tmp_path,
    )
    calls: list[tuple[str, str]] = []

    def fake_submit(campaign: Any, *, git_commit: str, storage_root: Path) -> dict[str, Any]:
        run_id = generation.campaign.campaign_run_id(campaign, git_commit=git_commit)
        calls.append(("submit", run_id))
        manifest_path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {"state": "active"}

    def fake_feed(run_id: str, *, storage_root: Path, resolved_campaign: Any) -> dict[str, Any]:
        assert storage_root == tmp_path
        assert generation.campaign.campaign_run_id(resolved_campaign, git_commit="d" * 40) == run_id
        calls.append(("feed", run_id))
        return {"state": "complete"}

    monkeypatch.setattr(generation.campaign, "submit_campaign", fake_submit)
    monkeypatch.setattr(generation.campaign, "feed_campaign", fake_feed)
    monkeypatch.setattr(
        completion,
        "_candidate_terminal_evidence",
        lambda *_args, **_kwargs: _terminal_evidence(),
    )
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
    assert [kind for kind, _run_id in calls] == ["submit", "feed"]
    assert len(state["candidates"]) == 1
    assert completion.uncovered_deficits(
        state,
        original_successes=completion.parent_partial_counts(
            parent,
            partial,
            parent_run_id=partial["campaign_run_id"],
        )[0],
    ) == dict.fromkeys((item.batch_id for item in parent.batches), 0)


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
