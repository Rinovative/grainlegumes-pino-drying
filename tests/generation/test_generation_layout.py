"""Generation storage layout: campaign, execution, state, and case locators.

Responsibilities:
  - Check the central path resolvers that every Generation writer and reader shares
  - Check execution records keyed by material and regime and the uniqueness of those keys
  - Check the single stage state: lock and staging names carry task, campaign, material and regime
  - Check atomic per-case reservation, fail-closed allocation, and side-by-side attempt records

Design principles:
  - Locators are pure functions of the campaign tree, never of what already exists
  - Duplicate execution keys, case-ID gaps, and incomplete case directories fail closed

This module does NOT:
  - Execute COMSOL or submit Slurm jobs
  - Check record contents that other test modules own
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.publication import generation_publication_catalog as catalog
from src.generation.runtime import generation_runtime_batch as runtime_service

if TYPE_CHECKING:
    from pathlib import Path

_BATCH = "source~transient_drying~lentil-3+chickpea-3__ref-range__lhs-seed-9220~lentil~natural"


def test_execution_key_names_a_batch_by_material_and_regime() -> None:
    """Hash-free keys identify each batch, including replacement batches of one regime."""
    assert common.paths.generation_execution_key(_BATCH) == "lentil-natural"
    replacement = _BATCH.replace("~natural", "~natural_replacement_0001")
    assert common.paths.generation_execution_key(replacement) == "lentil-natural_replacement_0001"
    with pytest.raises(ValueError, match="Canonical source batch locator"):
        common.paths.generation_execution_key("transient_drying__lentil__natural")


def test_operational_records_live_in_the_execution_tree_and_locks_in_the_stage_state(tmp_path: Path) -> None:
    """Campaign records and logs resolve below _record/, operational ones below _record/execution/, cases below materials/."""
    campaign = common.paths.resolve_generation_campaign_directory(
        "transient_drying", "lentil-3+chickpea-3__ref-range__lhs-seed-9220", storage_root=tmp_path
    )
    execution = common.paths.resolve_generation_execution_directory(campaign)
    key = common.paths.generation_execution_key(_BATCH)
    assert common.paths.resolve_generation_campaign_plan_path(campaign) == campaign / "_record/campaign_plan.json"
    assert common.paths.resolve_generation_selected_pool_path(campaign) == campaign / "_record/selected_pool.json"
    assert common.paths.resolve_generation_campaign_logs_directory(campaign) == campaign / "_record/logs"
    material = campaign / "materials/lentil"
    assert common.paths.resolve_generation_material_directory("transient_drying", campaign.name, "lentil", storage_root=tmp_path) == material
    assert common.paths.resolve_generation_cases_directory(_BATCH, storage_root=tmp_path) == material
    assert common.paths.resolve_generation_batch_metadata_directory(_BATCH, storage_root=tmp_path) == material / "_record/natural"
    assert execution == campaign / "_record/execution"
    assert common.paths.resolve_generation_run_manifest_path(execution) == execution / "run.json"
    assert common.paths.resolve_generation_cancellations_path(execution) == execution / "cancellations.json"
    assert common.paths.resolve_generation_epochs_directory(execution) == execution / "epochs"
    assert common.paths.resolve_generation_slurm_directory(execution) == execution / "slurm"
    assert common.paths.resolve_generation_license_case_directory(execution, key, "case_0001") == execution / "license/lentil-natural/case_0001"
    assert (
        common.paths.resolve_generation_reconciliation_case_directory(execution, key, "case_0001")
        == execution / "reconciliation/lentil-natural/case_0001"
    )
    assert (
        common.paths.resolve_generation_reconciliation_failures_directory(execution, key, "case_0001")
        == execution / "reconciliation_failures/lentil-natural/case_0001"
    )


def test_locks_and_staging_live_in_one_stage_state_named_by_task_campaign_material_and_regime(tmp_path: Path) -> None:
    """Every lock is an anchor below 01_generation/.state/locks and all staging below .state/staging, never inside a campaign."""
    campaign = common.paths.resolve_generation_campaign_directory(
        "transient_drying", "lentil-3+chickpea-3__ref-range__lhs-seed-9220", storage_root=tmp_path
    )
    state = tmp_path / "01_generation/.state"
    assert common.paths.get_generation_state_root(storage_root=tmp_path) == state
    stem = f"transient_drying~{campaign.name}"
    assert common.paths.resolve_generation_controller_lock_path(campaign) == state / "locks" / f"{stem}~controller.lock"
    assert common.paths.resolve_generation_submission_lock_path(campaign) == state / "locks" / f"{stem}~submission.lock"
    assert common.paths.resolve_generation_input_lock_path(_BATCH, storage_root=tmp_path) == state / "locks" / f"{stem}~lentil~natural~inputs.lock"
    case_lock = common.paths.resolve_generation_case_lock_path(_BATCH, "case_0001", storage_root=tmp_path)
    assert case_lock == state / "locks" / f"{stem}~lentil~natural~case_0001.lock"
    staging = state / "staging" / stem
    assert common.paths.resolve_generation_reservation_staging_directory(campaign, "lentil") == staging / "lentil/reservation-staging"
    assert common.paths.resolve_generation_state_batch_directory(_BATCH, storage_root=tmp_path) == staging / "lentil/natural"
    assert common.paths.resolve_generation_case_publications_directory(_BATCH, storage_root=tmp_path) == staging / "lentil/natural/publications"
    transaction = common.paths.resolve_generation_input_transaction_directory(_BATCH, "input-abc", storage_root=tmp_path)
    assert transaction == staging / "lentil/natural/input-transactions/input-abc"
    metadata = transaction / "meta" / _BATCH / "input_generations" / "input-abc"
    assert common.paths.resolve_generation_input_metadata_storage_root(metadata, _BATCH, "input-abc") == tmp_path
    published = common.paths.resolve_generation_batch_metadata_directory(_BATCH, storage_root=tmp_path)
    assert common.paths.resolve_generation_input_metadata_storage_root(published, _BATCH, "input-abc") == tmp_path


def test_lock_names_are_injective_and_stay_within_one_path_component(tmp_path: Path) -> None:
    """A part holding the separator, a non-canonical campaign path, or an overlong lock name fails instead of aliasing another lock."""
    for unsafe in (tmp_path / "01_generation/transient_drying/a~b", tmp_path / "elsewhere/transient_drying/campaign", tmp_path / "campaign"):
        with pytest.raises(ValueError, match=r"canonical Generation campaign directory|must not contain"):
            common.paths.resolve_generation_controller_lock_path(unsafe)
    long_batch = "source~transient_drying~" + "c" * 211 + "~lentil~natural"  # a valid 250-byte batch name whose case lock needs 258 bytes
    with pytest.raises(ValueError, match="255 UTF-8 bytes"):
        common.paths.resolve_generation_case_lock_path(long_batch, "case_0001", storage_root=tmp_path)
    overlong = tmp_path / "01_generation/transient_drying" / ("c" * 250)  # a valid folder name whose joined staging name needs 267 bytes
    with pytest.raises(ValueError, match="255 UTF-8 bytes"):
        common.paths.resolve_generation_reservation_staging_directory(overlong, "lentil")
    with pytest.raises(ValueError, match=r"case_id must match case_NNNN"):  # "inputs" would name the batch's input lock
        common.paths.resolve_generation_case_lock_path(_BATCH, "inputs", storage_root=tmp_path)


def test_snapshots_and_source_trees_have_their_own_roots(tmp_path: Path) -> None:
    """Snapshots sit below storage and frozen source trees below runtime."""
    assert common.paths.get_source_snapshots_root(storage_root=tmp_path) == tmp_path / "source_snapshots"
    assert common.paths.get_source_trees_root(runtime_root=tmp_path) == tmp_path / "sources"


def test_frozen_batches_must_have_unique_material_regime_keys() -> None:
    """One campaign cannot freeze two batches that would share an execution directory."""
    lifecycle._require_unique_execution_keys([{"batch_storage_name": _BATCH}, {"batch_storage_name": _BATCH.replace("~lentil~", "~chickpea~")}])
    with pytest.raises(ValueError, match="unique material-regime execution keys"):
        lifecycle._require_unique_execution_keys([{"batch_storage_name": _BATCH}, {"batch_storage_name": _BATCH}])


def _reserve(campaign: Any, storage: Any, position: int, *, design_key: str | None = None) -> dict[str, Any]:
    """Reserve one primary candidate with a minimal frozen sample."""
    return lifecycle.reserve_candidate(
        campaign.profile.id,
        campaign.source_name,
        "lentil",
        origin="primary",
        design_key=design_key or f"natural:{position}",
        candidate_position=position,
        target_slot=position,
        sample={"position": position},
        derived_seeds={"case_seed": position},
        storage_root=storage,
    )


def _campaign(generation_config_factory: Any, tmp_path: Any, *, count: int = 4) -> tuple[Any, Any, Any]:
    """Freeze one small steady-flow campaign and return it with its storage and directory."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=count)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    return campaign, storage, root


def test_reservation_publishes_one_case_directory_with_its_only_reservation(generation_config_factory: Any, tmp_path: Any) -> None:
    """Allocation creates contiguous case directories and no second reservation ledger."""
    campaign, storage, root = _campaign(generation_config_factory, tmp_path)
    first = _reserve(campaign, storage, 1)
    second = _reserve(campaign, storage, 2)
    assert (first["case_id"], second["case_id"]) == ("case_0001", "case_0002")
    assert _reserve(campaign, storage, 1)["case_id"] == "case_0001"
    for case_id in ("case_0001", "case_0002"):
        assert json.loads((root / "materials/lentil" / case_id / "_record/reservation.json").read_text(encoding="utf-8"))["case_id"] == case_id
    assert not (root / "materials/lentil/reservations").exists()
    assert not any(common.paths.resolve_generation_reservation_staging_directory(root, "lentil").iterdir())
    assert [row["reservation"]["case_id"] for row in lifecycle.candidate_records("steady_flow", campaign.source_name, storage_root=storage)] == [
        "case_0001",
        "case_0002",
    ]


def test_interrupted_reservation_staging_is_discarded_before_the_next_allocation(generation_config_factory: Any, tmp_path: Any) -> None:
    """An unpublished staging leftover never becomes a case and never blocks allocation."""
    campaign, storage, root = _campaign(generation_config_factory, tmp_path)
    stale = common.paths.resolve_generation_reservation_staging_directory(root, "lentil") / "case_0001"
    stale.mkdir(parents=True)
    (stale / "reservation.json").write_text("{}", encoding="utf-8")
    reservation = _reserve(campaign, storage, 1)
    published = json.loads((root / "materials/lentil/case_0001/_record/reservation.json").read_text(encoding="utf-8"))
    assert published == reservation
    assert not stale.exists()


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("gap", "malformed or missing append-only candidate IDs"),
        ("incomplete", "no published reservation"),
        ("conflict", "conflicts with its case directory"),
        ("stray", "not a case directory"),
        ("material_record_link", "symbolic link"),  # the material-namespace resolver refuses it before the frozen-plan read
        ("case_record_link", "no published reservation"),
        ("duplicate", "duplicated durable reservations"),
    ],
)
def test_reservation_allocation_fails_closed_on_damaged_case_namespaces(
    generation_config_factory: Any, tmp_path: Any, damage: str, message: str
) -> None:
    """Gaps, incomplete directories, conflicting reservations, strays, linked _record/ folders, and duplicates are never repaired."""
    campaign, storage, root = _campaign(generation_config_factory, tmp_path)
    _reserve(campaign, storage, 1)
    _reserve(campaign, storage, 2)
    cases = root / "materials/lentil"
    if damage == "gap":
        shutil.rmtree(cases / "case_0001")
    elif damage == "incomplete":
        (cases / "case_0003").mkdir()
    elif damage == "conflict":
        path = cases / "case_0002/_record/reservation.json"
        path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "case_id": "case_0009"}), encoding="utf-8")
    elif damage == "stray":
        (cases / "notes.txt").write_text("stray", encoding="utf-8")
    elif damage == "material_record_link":
        (tmp_path / "elsewhere").mkdir()
        (cases / "_record").symlink_to(tmp_path / "elsewhere")
    elif damage == "case_record_link":
        (cases / "case_0002/_record").rename(tmp_path / "moved_record")
        (cases / "case_0002/_record").symlink_to(tmp_path / "moved_record")
    else:
        path = cases / "case_0002/_record/reservation.json"
        first = json.loads((cases / "case_0001/_record/reservation.json").read_text(encoding="utf-8"))
        path.write_text(json.dumps({**first, "case_id": "case_0002"}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _reserve(campaign, storage, 3)
    assert not (cases / "case_0003/_record/reservation.json").exists()


@pytest.mark.parametrize(
    ("writer", "linked"),
    [
        (writer, linked)
        for writer in ("amendment", "input_preparation", "transaction_recovery", "reservation")
        for linked in ("materials", "material", "record", "regime")
        if linked != "regime" or writer in {"input_preparation", "transaction_recovery"}  # only input generation writes below _record/<regime>
    ],
)
def test_no_writer_of_the_material_namespace_follows_a_symlink_out_of_the_campaign(
    generation_config_factory: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, writer: str, linked: str
) -> None:
    """A symlinked materials/, material, its _record/ or _record/<regime> is refused by every writer; nothing is written outside."""
    campaign, storage, root = _campaign(generation_config_factory, tmp_path)
    batch = campaign.batches[0]
    input_generation_id = generation.cases.input_generation.configured_input_generation_id(batch)
    transaction = common.paths.resolve_generation_input_transaction_directory(batch.batch_storage_name, input_generation_id, storage_root=storage)
    regime = common.paths.campaign_batch_parts(batch.batch_storage_name)[3]  # type: ignore[index]
    if writer == "transaction_recovery":
        # A real interrupted input generation: its journal is committed, its staged case is not yet published.
        def interrupt() -> None:
            message = "interrupted before publication"
            raise RuntimeError(message)

        with monkeypatch.context() as interrupted:
            interrupted.setattr(generation.cases.input_generation.source_service, "validate_admitted_source_before_publication", interrupt)
            with pytest.raises(RuntimeError, match="interrupted before publication"):
                generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage)
        assert (transaction / "transaction.json").is_file()
        shutil.move(root / "materials", tmp_path / "detached")  # the material tree is replaced by the symlink below
    outside = tmp_path / "outside"
    outside.mkdir()
    components = {
        "materials": root / "materials",
        "material": root / "materials/lentil",
        "record": root / "materials/lentil/_record",
        "regime": root / "materials/lentil/_record" / regime,
    }
    target = components[linked]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    operations = {
        "amendment": lambda: lifecycle.amend_minimum_success_rate(
            campaign.profile.id, campaign.source_name, "lentil", minimum_success_rate=0.9, reason="Synthetic tolerated failure", storage_root=storage
        ),
        "input_preparation": lambda: generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage),
        "transaction_recovery": lambda: generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage),
        "reservation": lambda: _reserve(campaign, storage, 1),
    }
    with pytest.raises(ValueError, match="symbolic link"):
        operations[writer]()
    assert not any(outside.iterdir())
    if writer == "transaction_recovery":  # the journaled transaction was neither recovered nor promoted through the symlink
        assert (transaction / "transaction.json").is_file()
        assert (transaction / "raw" / batch.batch_storage_name / "input_generations" / input_generation_id / "case_0001").is_dir()
    else:
        assert not transaction.exists()


def test_classified_attempts_and_failure_bundles_live_side_by_side(
    generation_config_factory: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both attempt families share attempts/ without sharing counters or names."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=1)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "storage"
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    generation.cases.input_generation.prepare_campaign_inputs(campaign, git_commit="a" * 40, storage_root=storage)
    batch = campaign.batches[0]
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", generation.campaign.campaign_run_id(campaign, git_commit="a" * 40))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("GENERATION_SUBMISSION_INDEX", "1")
    monkeypatch.setenv("GENERATION_DISPATCH_KEY", "4242-1")
    for reason in ("first synthetic rejection", "second synthetic rejection"):
        runtime_service.record_case_failure(
            batch,
            1,
            RuntimeError(reason),
            worker_slot=0,
            scheduler_kind="slurm",
            allocated_node="node01",
            work_directory=None,
            storage_root=storage,
            scratch_cleanup_status="not_created",
            failure_stage="input",
        )
    attempt: dict[str, Any] = {"attempt_index": 1, "stage": "solver", "failure_class": "solver", "retryable": False, "detail": "classified"}
    with pytest.raises(ValueError, match="typed solver reason"):  # a solver attempt names the runtime's reason, not an exception class
        lifecycle.record_candidate_attempt(
            campaign.profile.id,
            campaign.source_name,
            "lentil",
            "case_0001",
            **attempt,
            reason_code="CaseExecutionError",
            evidence={},
            storage_root=storage,
        )
    lifecycle.record_candidate_attempt(
        campaign.profile.id, campaign.source_name, "lentil", "case_0001", **attempt, reason_code="solver_error", evidence={}, storage_root=storage
    )
    attempts = common.paths.resolve_generation_attempts_directory(batch.batch_storage_name, "case_0001", storage_root=storage)
    assert attempts == common.paths.resolve_generation_material_directory("steady_flow", campaign.source_name, "lentil", storage_root=storage) / (
        "case_0001/_record/attempts"
    )
    assert sorted(entry.name for entry in attempts.iterdir()) == ["attempt_0001.json", "run_0001", "run_0002"]
    assert common.paths.locate_generation_attempt_case_directory(attempts / "run_0002") == attempts.parent.parent
    assert [directory.name for directory in attempt_service.attempt_run_directories(attempts)] == ["run_0001", "run_0002"]
    latest = attempt_service.latest_case_attempt(batch, 1, storage_root=storage)
    assert latest is not None
    assert latest.directory == attempts / "run_0002"
    assert latest.payload["previous_attempt"]["attempt_index"] == 1
    records = lifecycle.candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage)
    assert [len(row["attempts"]) for row in records] == [1]


def test_case_locators_invert_to_their_parts_and_reject_foreign_paths(tmp_path: Any) -> None:
    """Case directories and result locators round-trip through the central inversions."""
    case = common.paths.resolve_generation_campaign_case_directory(
        "transient_drying", "lentil-3__ref-range__lhs-seed-1", "lentil", "case_0002", storage_root=tmp_path
    )
    location = common.paths.locate_generation_case_directory(case)
    assert (location.storage_root, location.task, location.campaign_name, location.material, location.case_id) == (
        tmp_path,
        "transient_drying",
        "lentil-3__ref-range__lhs-seed-1",
        "lentil",
        "case_0002",
    )
    relative = (common.paths.resolve_generation_case_result_directory(case) / "case.h5").relative_to(tmp_path).as_posix()
    assert relative == "01_generation/transient_drying/lentil-3__ref-range__lhs-seed-1/materials/lentil/case_0002/result/case.h5"
    parsed = common.paths.locate_generation_case_result_file(relative, "case.h5")
    assert (parsed.task, parsed.material, parsed.case_id) == ("transient_drying", "lentil", "case_0002")
    for unsafe in (
        "../x/case.h5",
        "01_generation/transient_drying/c/lentil/case_0002/result/case.h5",
        "01_generation/transient_drying/c/lentil/cases/case_0002/result/case.h5",
        "01_generation/transient_drying/c/_record/case_0002/result/case.h5",
        "/abs/case.h5",
    ):
        with pytest.raises(ValueError, match="canonical Generation case result file"):
            common.paths.locate_generation_case_result_file(unsafe, "case.h5")
    with pytest.raises(ValueError, match="canonical Generation case directory"):
        common.paths.locate_generation_case_directory(tmp_path / "elsewhere/case_0001")
    with pytest.raises(ValueError, match="canonical Generation attempt bundle"):
        common.paths.locate_generation_attempt_case_directory(case / "attempts/run_0001")


def test_campaign_creation_and_execution_readers_refuse_a_record_directory_that_is_not_their_own(
    generation_config_factory: Any, tmp_path: Path
) -> None:
    """A symlinked or foreign _record/ and the superseded root-level plan are refused before any record is published or read."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=1)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "storage"
    root = common.paths.resolve_generation_campaign_directory("steady_flow", campaign.source_name, storage_root=storage)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    root.mkdir(parents=True)
    (root / "_record").symlink_to(foreign, target_is_directory=True)
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    for operation in (
        lambda: lifecycle.create_campaign(campaign, storage_root=storage),
        lambda: generation.publication.campaign_evidence.load_campaign_run(run_id, storage_root=storage),
    ):
        with pytest.raises(ValueError, match="not a real directory below its campaign"):
            operation()
    assert not any(foreign.iterdir())
    (root / "_record").unlink()
    (root / "campaign_plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="superseded storage layout"):
        lifecycle.create_campaign(campaign, storage_root=storage)
    assert sorted(entry.name for entry in root.iterdir()) == ["campaign_plan.json"]


def test_campaign_discovery_rejects_the_superseded_layout(tmp_path: Path) -> None:
    """A campaign whose plan sits outside _record/ is refused by name instead of being skipped or read."""
    old = common.paths.resolve_generation_campaign_directory("steady_flow", "lentil-1__ref-range__lhs-seed-1", storage_root=tmp_path)
    old.mkdir(parents=True)
    (old / "campaign_plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="superseded storage layout"):
        catalog.list_campaigns("steady_flow", storage_root=tmp_path)


def test_input_generation_locators_require_a_campaign_batch_locator(tmp_path: Path) -> None:
    """A batch name without a campaign namespace has no input-generation location."""
    with pytest.raises(ValueError, match="not a campaign locator"):
        common.paths.resolve_generation_input_metadata_storage_root(tmp_path, "flat_batch_name", "input-" + "a" * 24)
