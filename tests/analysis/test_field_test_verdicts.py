"""Protect the field-test verdict contract from frozen tolerances to the published verdict files.

Responsibilities:
  - Freeze the tolerance table once and refuse a changed table, a table of another stage, or a linked analysis folder
  - Exclude and count physically invalid validation rollouts in the yardstick, record each run by its admitted identity
    and refuse a run named twice
  - Decide remove, keep, unresolved and merge from synthetic sealed members, including pairing validity, reproduced
    failures, the replicate envelope at each comparison's own hours and window end, coupled-stream base counts, per-base
    merge clearance, the joint removal test and duration censoring
  - Validate ``verdicts.json`` with its persisted measurements and read stage A's removals as stage B would

Design principles:
  - One synthetic field test per task carries one row per contract branch; variants are the base plus a known
    multiple of the fixture error pattern, so every ratio is predictable

This module does NOT:
  - Build tolerances from trained runs or run Generation
"""

from __future__ import annotations

import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from support import field_test_fixtures as fixtures

from src import common, generation
from src.analysis import field_tests

if TYPE_CHECKING:
    from pathlib import Path

STEADY = "steady_flow"
TRANSIENT = "transient_drying"


def _challenge(row: str, base: int, scale: float, task: str = STEADY, **extra: Any) -> dict[str, Any]:
    """Return one challenge of ``row`` at ``base`` that moves the base by ``scale`` error patterns."""
    completion = extra.pop("completion", {})
    return {"row": row, "level": "high", "base": base, "case": fixtures.case_kwargs(task, base, scale=scale, **completion), **extra}


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index verdict rows by name."""
    return {row["row"]: row for row in payload["rows"]}


def _build(storage: Path, task: str = STEADY) -> dict[str, Any]:
    """Build the verdicts of the fixture field test of ``task``."""
    return field_tests.verdicts.build_verdicts(task, fixtures.TEST_ID, storage_root=storage)


def test_tolerance_freeze_is_write_once_and_verdicts_refuse_a_changed_table(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    test_dir = fixtures.make_field_test(storage, task=STEADY, stage="A", bases=[("lentil", 1)], challenges=[_challenge("row.small", 0, 0.1)])
    with pytest.raises(FileNotFoundError):
        _build(storage)
    digest = fixtures.freeze(test_dir, STEADY)
    with pytest.raises(FileExistsError):
        fixtures.freeze(test_dir, STEADY)
    payload = _build(storage)
    assert payload["tolerances_sha256"] == digest

    table_path = test_dir / "analysis" / "tolerances.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["entries"]["steady/p/relative_l2"]["tolerance"] *= 10.0
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was frozen"):
        _build(storage)

    table_path.unlink()
    other = field_tests.tolerances.build_tolerance_table(STEADY, fixtures.validation_pairs(STEADY, run="other-run"))
    field_tests.tolerances.freeze_tolerances(table_path, other)
    with pytest.raises(ValueError, match="another tolerance table"):
        _build(storage)

    # A linked analysis folder would redirect the verdict files out of the field test.
    (test_dir / "analysis").rename(tmp_path / "elsewhere")
    (test_dir / "analysis").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        _build(storage)


def test_concurrent_freezes_publish_exactly_one_table(tmp_path: Path) -> None:
    table = field_tests.tolerances.build_tolerance_table(STEADY, fixtures.validation_pairs(STEADY))
    target = tmp_path / "analysis" / "tolerances.json"
    barrier = threading.Barrier(4)

    def freeze(run: int) -> str:
        barrier.wait()
        return field_tests.tolerances.freeze_tolerances(target, {**table, "model_runs": [f"run-{run}"]})

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(freeze, run) for run in range(4)]
    outcomes = [future.exception() for future in futures]

    assert sum(outcome is None for outcome in outcomes) == 1
    assert all(isinstance(outcome, FileExistsError) for outcome in outcomes if outcome is not None)
    assert [path.name for path in target.parent.iterdir()] == ["tolerances.json"]


def test_the_yardstick_excludes_and_counts_physically_invalid_rollouts() -> None:
    pairs = fixtures.validation_pairs(TRANSIENT)
    states = np.array(pairs[0].prediction.states, copy=True)
    states[2, 2, 32, 48] = -1.0  # finite negative surface moisture inside the bed after the initial state
    invalid = dataclasses.replace(pairs[0], case_id="case_0002", prediction=dataclasses.replace(pairs[0].prediction, states=states))

    table = field_tests.tolerances.build_tolerance_table(TRANSIENT, [*pairs, invalid])

    assert table["excluded_counts"] == {"physically_invalid_rollout": 1}
    assert table["cases"] == [{"run": "03_experiments/demo-run", "case_id": f"case_{seed:04d}"} for seed in (0, 1)]


def test_the_yardstick_records_each_run_by_its_admitted_identity_and_refuses_a_repeated_run(tmp_path: Path) -> None:
    storage = tmp_path / "storage"

    def identity(run_dir: Path, digest: str) -> dict[str, str]:
        completed = {"run_dir": run_dir, "effective_config_digest": digest * 64, "selected_checkpoint_sha256": "c" * 64}
        return field_tests.tolerances._run_identity(completed, storage_root=storage)

    # Two runs that share the directory name a0 stay distinguishable by their path below the storage root.
    runs = [identity(storage / "03_experiments" / group / "a0", digest) for group, digest in (("x", "a"), ("y", "b"))]
    pairs = [dataclasses.replace(pair, run=run) for run in runs for pair in fixtures.validation_pairs(STEADY)]

    table = field_tests.tolerances.build_tolerance_table(STEADY, pairs)

    assert table["model_runs"] == runs
    assert [case["run"] for case in table["cases"]] == ["03_experiments/x/a0"] * 2 + ["03_experiments/y/a0"] * 2
    with pytest.raises(ValueError, match="not below the storage root"):
        identity(tmp_path / "a0", "a")
    # A run named again through an alias resolves to the same admitted path and would count twice in every median.
    (storage / "03_experiments" / "x" / "a0").mkdir(parents=True)
    (storage / "alias").symlink_to(storage / "03_experiments" / "x", target_is_directory=True)
    alias = [dataclasses.replace(pair, run=identity(storage / "alias" / "a0", "a")) for pair in fixtures.validation_pairs(STEADY)]
    with pytest.raises(ValueError, match="named twice"):
        field_tests.tolerances.build_tolerance_table(STEADY, [*pairs, *alias])


def test_steady_rules_pairing_failures_coupled_rows_and_stage_b_reading(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    bases = [("lentil", 1), ("lentil", 2), ("chickpea", 3), ("chickpea", 4)]
    unreproduced = ["infrastructure_stop", "infrastructure_stop", "solver_failure"]
    unpaired = [("solver_failure", {"spatial_support_attempt": 1}), "solver_failure"]
    test_dir = fixtures.make_field_test(
        storage,
        task=STEADY,
        stage="A",
        bases=bases,
        challenges=[
            _challenge("row.small", 0, 0.1),
            _challenge("row.small", 1, 0.1),
            _challenge("row.large", 0, 0.1),
            _challenge("row.large", 1, 3.0),
            _challenge("row.attempt", 0, 0.1),
            _challenge("row.attempt", 1, 0.1, seal={"spatial_support_attempt": 1}),
            {"row": "row.fails", "level": "high", "base": 0, "case": None, "seal": {"attempts": 2}},
            {"row": "row.unreproduced", "level": "high", "base": 0, "case": None, "seal": {"attempts": 3}, "outcomes": unreproduced},
            {"row": "row.unpaired_failure", "level": "high", "base": 0, "case": None, "seal": {"attempts": 2}, "outcomes": unpaired},
            _challenge("bed.perturbations.amplitude", 0, 0.1),
            _challenge("bed.perturbations.amplitude", 1, 0.1),
            *(_challenge("bed.structure.coarse_len_rel", base, 0.1) for base in range(4)),
            _challenge("pressure_bc.profile_amp", 0, 0.1),
        ],
    )
    fixtures.freeze(test_dir, STEADY)

    payload = _build(storage)

    rows = _rows(payload)
    assert rows["row.small"]["verdict"] == "remove"
    assert 0.0 < rows["row.small"]["worst_ratio"] < 0.5
    assert rows["row.large"]["verdict"] == "keep"
    assert rows["row.large"]["worst_at"]["base"] == "lentil/sobol/2"
    assert rows["row.attempt"]["verdict"] == "unresolved"
    assert (rows["row.attempt"]["valid_comparisons"], rows["row.attempt"]["missing_or_broken"]) == (1, 1)
    assert rows["row.fails"]["verdict"] == "keep"
    # Two infrastructure stops and one solver failure reach the attempt limit but reproduce no failure.
    assert (rows["row.unreproduced"]["verdict"], rows["row.unreproduced"]["missing_or_broken"]) == ("unresolved", 1)
    # The first failure ran on another accepted support draw, so only one failure is validly paired.
    assert (rows["row.unpaired_failure"]["verdict"], rows["row.unpaired_failure"]["missing_or_broken"]) == ("unresolved", 1)
    assert rows["bed.perturbations.amplitude"]["verdict"] == "unresolved"
    assert rows["bed.structure.coarse_len_rel"]["verdict"] == "remove"
    assert rows["pressure_bc.profile_amp"]["verdict"] == "remove"
    assert any("provisional" in note for note in rows["pressure_bc.profile_amp"]["notes"])
    assert (test_dir / "analysis" / "verdicts.md").is_file()
    assert (test_dir / "analysis" / "verdicts.png").stat().st_size > 0

    published = field_tests.verdicts.load_verdicts(STEADY, fixtures.TEST_ID, storage_root=storage)
    assert set(published) >= {"schema_kind", "schema_version", "test_id", "roster_sha256", "tolerances_sha256", "rows"}
    assert all(
        set(row) == {"row", "verdict", "worst_ratio", "worst_at", "valid_comparisons", "missing_or_broken", "notes"} for row in published["rows"]
    )
    assert set(field_tests.verdicts.screening_removals(published)) == {"row.small", "bed.structure.coarse_len_rel", "pressure_bc.profile_amp"}
    # Every comparison persists its verdict and report-only values with their contributor counts, inputs in every region.
    small = next(item for item in published["comparisons"] if item["row"] == "row.small")
    assert small["measured"]["steady/p/patch_wall"][0] > 0.0 < small["measured"]["steady/p/patch_wall"][1]
    assert {f"input/Kxx/{name}" for name in ("patch_interior", "patch_wall", "patch_inlet", "wall_core_contrast")} <= set(small["report_only"])
    assert small["report_only"]["input/Kxx/patch_inlet"][0] == 0.0 < small["report_only"]["input/Kxx/patch_inlet"][1]
    assert [item["problem"] for item in published["replicates"]] == [None] * len(bases)


def test_a_replicate_difference_at_the_margin_blocks_removal_at_its_base(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    test_dir = fixtures.make_field_test(
        storage,
        task=STEADY,
        stage="A",
        bases=[("lentil", 1), ("lentil", 2)],
        challenges=[_challenge("row.small", 0, 0.1), _challenge("row.small", 1, 0.1), _challenge("row.other", 1, 0.1)],
        gate_cases={0: fixtures.case_kwargs(STEADY, 0, scale=0.6)},
    )
    fixtures.freeze(test_dir, STEADY)

    payload = _build(storage)

    rows = _rows(payload)
    assert (rows["row.small"]["verdict"], rows["row.other"]["verdict"]) == ("unresolved", "remove")
    assert any(note.startswith("replicate difference at lentil/sobol/1") for note in rows["row.small"]["notes"])
    replicates = {item["base"]: item["endpoints"] for item in payload["replicates"]}
    assert replicates["lentil/sobol/1"]["steady/p/patch_interior"][1] == pytest.approx(0.6, rel=1e-3)
    assert replicates["lentil/sobol/2"]["steady/p/patch_interior"][1] == 0.0


def test_a_transient_replicate_beyond_the_base_window_carries_the_largest_ratio_of_its_quantity(tmp_path: Path) -> None:
    storage = tmp_path / "storage"

    def short(seed: int) -> dict[str, Any]:
        base = fixtures.case_kwargs(TRANSIENT, seed)
        return {**base, "times": base["times"][:3], "states": base["states"][:3], "t_final": 2.0}

    # Both bases are published, and replicated by their gates, only up to hour 2; references and challenges store 0 to 4.
    # Base 1's gate moves hour 1 down and hour 2 up by 0.4 tolerance units: its states stay within 0.2 of their
    # tolerances, but its increment over the window ending at 2 reaches 0.8 of the increment tolerance.
    shift = 0.4 * np.array([0.1, 0.002, 0.5, 0.5])[:, None, None] * fixtures.error_pattern(fixtures.GRID)
    replica = short(1)
    replica["states"] = replica["states"] + np.stack([0.0 * shift, -shift, shift])
    test_dir = fixtures.make_field_test(
        storage,
        task=TRANSIENT,
        stage="A",
        bases=[("lentil", 1), ("lentil", 2)],
        challenges=[_challenge("row.small", 0, 0.1, TRANSIENT), _challenge("row.other", 1, 0.1, TRANSIENT)],
        published_cases={0: short(0), 1: short(1)},
        gate_cases={1: replica},
    )
    fixtures.freeze(test_dir, TRANSIENT)

    payload = _build(storage, TRANSIENT)

    rows = _rows(payload)
    assert (rows["row.small"]["verdict"], rows["row.other"]["verdict"]) == ("remove", "unresolved")
    assert "replicate envelope carried from the base window (max ratio 0): c0-0" in rows["row.small"]["notes"]
    assert "replicate envelope carried from the base window (max ratio 0.8): c1-1" in rows["row.other"]["notes"]
    # Hours 3 and 4 carry the base's state ratios (0.2 at most); only the window end 4 carries the blocking increment.
    blocked = next(note for note in rows["row.other"]["notes"] if note.startswith("replicate difference at lentil/sobol/2"))
    assert "8 endpoints (transient/T/increment_peak@h=4, transient/T/increment_rms@h=4" in blocked
    replicates = {item["base"]: item["endpoints"] for item in payload["replicates"]}
    assert {f"transient/T/increment_peak@h={hour}" for hour in (1, 2)} <= set(replicates["lentil/sobol/1"])


def test_cross_cpu_pairs_stay_valid_and_record_their_cpu_models(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    test_dir = fixtures.make_field_test(
        storage, task=STEADY, stage="A", bases=[("lentil", 1)], challenges=[_challenge("row.cross_cpu", 0, 0.1, seal={"cpu_model": "other"})]
    )
    fixtures.freeze(test_dir, STEADY)

    row = _rows(_build(storage))["row.cross_cpu"]

    assert (row["verdict"], row["missing_or_broken"]) == ("remove", 0)
    assert "cpu models (evidence only): fixture, other" in row["notes"]


@pytest.mark.parametrize(("layout", "expected"), [(((0, 1), (0, 2), (1, 1), (1, 2)), "merge"), (((0, 1), (0, 2), (0, 3), (1, 1)), "unresolved")])
def test_stage_b_merges_need_two_clearing_displacements_at_every_base(tmp_path: Path, layout: tuple[tuple[int, int], ...], expected: str) -> None:
    storage = tmp_path / "storage"
    # The producer's format: the reference's intervention with exactly the two merge parameters moved.
    displacements = [
        {**_challenge("merge:a+b", base, 0.1), "level": "merge", "intervention": {"fixture.nominal": 0.0, "a": shift, "b": -shift}}
        for base, shift in layout
    ]
    fixtures.make_field_test(storage, task=STEADY, stage="B", bases=[("kidney_bean", 7), ("kidney_bean", 8)], challenges=displacements)

    assert _rows(_build(storage))["merge:a+b"]["verdict"] == expected


@pytest.mark.parametrize(("joint_scale", "cleared", "verdict"), [(0.1, True, "remove"), (3.0, False, "unresolved")])
def test_the_joint_test_gates_every_stage_b_removal_against_stage_a_table(tmp_path: Path, joint_scale: float, cleared: bool, verdict: str) -> None:
    storage = tmp_path / "storage"
    joints = [
        {"role": "joint", "base": base, "case": fixtures.case_kwargs(STEADY, base, scale=scale), "intervention": {"row.confirmed": 0.0}}
        for base, scale in ((0, 0.1), (1, joint_scale))
    ]
    test_dir = fixtures.make_field_test(
        storage,
        task=STEADY,
        stage="B",
        bases=[("kidney_bean", 7), ("kidney_bean", 8)],
        challenges=[_challenge("row.confirmed", 0, 0.1), _challenge("row.confirmed", 1, 0.1), *joints],
    )

    payload = _build(storage)

    assert [row["row"] for row in payload["rows"]] == ["row.confirmed"]
    assert (payload["joint"]["cleared"], sorted(payload["joint"]["member_ids"])) == (cleared, ["c2-0", "c3-1"])
    row = _rows(payload)["row.confirmed"]
    assert (row["verdict"], field_tests.verdicts.JOINT_NOT_CLEARED in row["notes"]) == (verdict, not cleared)
    # Stage B is judged only against the stage-A table its roster binds and never freezes one.
    assert payload["tolerances_sha256"] == generation.field_tests.roster.load_roster(test_dir)["derived_from"]["tolerances_sha256"]
    with pytest.raises(ValueError, match="stage-B confirmation"):
        field_tests.workflow.freeze_field_test_tolerances(STEADY, fixtures.TEST_ID, run_dirs=["unused-run"], storage_root=storage)
    stage_a_table = common.paths.resolve_field_test_dir(STEADY, fixtures.STAGE_A_TEST_ID, storage_root=storage) / "analysis" / "tolerances.json"
    stage_a_table.unlink()
    other = field_tests.tolerances.build_tolerance_table(STEADY, fixtures.validation_pairs(STEADY, run="other-run"))
    field_tests.tolerances.freeze_tolerances(stage_a_table, other)
    with pytest.raises(ValueError, match="not the table this stage-B roster binds"):
        _build(storage)


def test_transient_duration_censoring_rules(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    censored = {"wet_final": 0.2, "t_final": 4.0}
    test_dir = fixtures.make_field_test(
        storage,
        task=TRANSIENT,
        stage="A",
        bases=[("lentil", 1), ("chickpea", 2)],
        base_completion={1: {"horizon": 4.0, "t_final": 3.5}},
        challenges=[
            _challenge("dur.both", 0, 0.1, TRANSIENT, completion={"t_final": 3.9}),
            _challenge("dur.censored_far", 0, 0.1, TRANSIENT, completion=censored),
            _challenge("dur.censored_near", 1, 0.1, TRANSIENT, completion={**censored, "horizon": 4.0}),
        ],
    )
    fixtures.freeze(test_dir, TRANSIENT)

    rows = _rows(_build(storage, TRANSIENT))

    assert rows["dur.both"]["verdict"] == "remove"
    assert rows["dur.censored_far"]["verdict"] == "keep"
    assert rows["dur.censored_far"]["worst_at"]["endpoint"] == field_tests.endpoints.DURATION
    assert rows["dur.censored_near"]["verdict"] == "unresolved"
    assert any("censored duration" in note for note in rows["dur.censored_near"]["notes"])
