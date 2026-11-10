"""Persisted source-candidate publication and exact reader admission.

Responsibilities:
  - Check exact candidate publication, reader admission, and pilot selection
  - Check the exact published tree of one successful and one failed case and the campaign root entries
  - Check staged continuation and replacement feeding as tickets of fake node jobs
  - Check that only a derived admission cap follows a lowered rate, an explicit cap that holds back a replacement blocks,
    and a rerun compares effective caps
  - Check that the feeder decides runtime-stopped solves with their typed reason, only after their step ended, and writes cases.csv

Design principles:
  - Failed candidates stay excluded while original design membership remains fixed

This module does NOT:
  - Perform licensed COMSOL solves or publish production Dataset packages
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import h5py
import pytest
import yaml
from support.fake_slurm import COMMIT, FakeSlurm, bind_worker, feed, submit
from support.generation_fixtures import successful_source_case

from src import common, generation
from src.generation import generation_campaign as campaign_runtime
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation import generation_campaign_status as status_formatting
from src.generation import generation_controller as controller_service
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.publication import generation_publication_catalog as catalog
from src.generation.runtime import generation_runtime_host as host_service
from src.generation.runtime import generation_runtime_license as license_service

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from pathlib import Path


def _tree(directory: Path) -> set[str]:
    """Return every path below ``directory`` relative to it."""
    return {path.relative_to(directory).as_posix() for path in directory.rglob("*")}


def test_source_publication_admits_exact_success_and_excludes_failed_candidate(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real plan, inputs, fake solver, case bytes, decision, reader, and the published storage tree."""
    base_path, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=2)
    config_path = base_path
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    assert [row["campaign_name"] for row in catalog.list_campaigns("steady_flow", storage_root=storage)] == [campaign.campaign_name]
    plan = json.loads((root / "_record/campaign_plan.json").read_text(encoding="utf-8"))
    assert plan["schema_kind"] == "generation_source_campaign_plan"
    assert plan["original_design_sizes"] == {"lentil": {"test": 2}}
    assert all(key not in plan for key in ("material_roles", "membership", "dataset_packages"))

    primary = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(primary, 2, storage_root=storage)
    assert inputs.metadata_directory == root / "materials" / "lentil" / "_record" / "test"
    assert inputs.raw_directory == root / "materials" / "lentil"
    assert not (root / "materials" / "lentil" / "batches").exists()
    assert not list(root.rglob("input_generations"))
    references = generation.cases.input_generation.admit_configured_input_references(primary, storage_root=storage)
    first = references[1].case_payload()
    run_id = campaign_runtime.campaign_run_id(campaign, git_commit="a" * 40)
    with monkeypatch.context() as step:
        # The first candidate runs as one failing case step of the campaign and publishes its failure bundle.
        for name, value in (
            ("FAKE_COMSOL_MODE", "failure"),
            ("GENERATION_CAMPAIGN_RUN_ID", run_id),
            ("SLURM_JOB_ID", "4242"),
            ("SLURM_STEP_ID", "0"),
            ("GENERATION_SUBMISSION_INDEX", "1"),
            ("GENERATION_DISPATCH_KEY", "4242-1"),
        ):
            step.setenv(name, value)
        with pytest.raises(generation.runtime.CaseExecutionError):
            generation.runtime.run_case(primary, 1, cores_per_case=1, storage_root=storage, work_root=tmp_path / "work", scheduler_kind="local")
    lifecycle.decide_candidate(
        "steady_flow",
        campaign.source_name,
        "lentil",
        "case_0001",
        outcome="failed",
        stage="validation",
        failure_class="validation",
        reason_code="synthetic_rejection",
        detail="Synthetic scientific rejection",
        case_input_id=first["case_input_id"],
        simulation_case_id=first["simulation_case_id"],
        case_hdf5_relative=None,
        case_hdf5_sha256=None,
        evidence={"input_manifest_relative": (inputs.metadata_directory / "input_generation_manifest.json").relative_to(storage).as_posix()},
        storage_root=storage,
    )
    successful_source_case(campaign.source_name, primary, 2, storage, tmp_path / "work")

    run_dir = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = campaign_runtime._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={
            primary.batch_name: {
                "input_generation_id": inputs.input_generation_id,
                "source_git_commit": "a" * 40,
                "case_indices": list(primary.case_indices),
            }
        },
    )
    common.serialization.atomic_write_json(campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), manifest)

    # Without a plan s is 1, so the first failure stops the batch; one failure in two decisions keeps the bound above 0.9.
    lifecycle.amend_minimum_success_rate(
        "steady_flow",
        campaign.source_name,
        primary.material_family,
        minimum_success_rate=0.9,
        reason="Synthetic tolerated failure",
        storage_root=storage,
    )
    replacement = lifecycle.materialize_replacement_batch(campaign, primary.batch_id, storage_root=storage)
    assert replacement.case_indices == (3,)
    # The replacement is the next unused point of its parent's design: point 2 after the original points 0 and 1.
    assert replacement.scientific_values["replacement_design"]["design_index"] == len(primary.case_indices)
    assert replacement.seed_base == primary.seed_base
    assert primary.seed_base is not None
    assert replacement.case_seed(3) == generation.cases.seeding.derive_seed(primary.seed_base, "case", str(len(primary.case_indices)))
    assert replacement.scientific_values["sampling"]["design_seed"] == primary.scientific_values["sampling"]["design_seed"]
    generation.cases.input_generation.generate_input_cases(replacement, 1, storage_root=storage)
    successful_source_case(campaign.source_name, replacement, 3, storage, tmp_path / "work")
    discovery = generation.cases.admission.discover_input_batches(storage, task="steady_flow", validation_depth="evidence")
    assert len(discovery.sources) == 2
    assert not discovery.issues

    selected = lifecycle.selected_successes("steady_flow", campaign.source_name, storage_root=storage)
    assert [record["reservation"]["case_id"] for record in selected] == ["case_0002", "case_0003"]
    assert [record["reservation"]["origin"] for record in selected] == ["primary", "replacement"]
    assert [record["decision"]["case_hdf5_relative"] for record in selected] == [  # the locator that Dataset manifests store
        f"01_generation/steady_flow/{campaign.source_name}/materials/lentil/{case_id}/result/case.h5" for case_id in ("case_0002", "case_0003")
    ]
    rows = catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)
    assert [(row["case_id"], row["outcome"], row["selected"]) for row in rows] == [
        ("case_0001", "failed", False),
        ("case_0002", "successful", True),
        ("case_0003", "successful", True),
    ]

    for row in rows[1:]:
        selector = {
            key: row[key]
            for key in (
                "campaign_name",
                "campaign_digest",
                "material_family",
                "case_id",
                "case_input_id",
                "simulation_case_id",
                "case_hdf5_sha256",
            )
        }
        admitted = catalog.resolve_raw_case_reference("steady_flow", selector, storage_root=storage)
        assert admitted.case.case_id == row["case_id"]
        assert admitted.source_identity["source_batch_id"] == (replacement.batch_id if row["origin"] == "replacement" else primary.batch_id)
        assert admitted.source_identity["source_design_parent_batch_id"] == primary.batch_id
        assert admitted.source_identity["domain_profile"] == "reference_range"
        assert admitted.source_identity["case_hdf5_sha256"] == row["case_hdf5_sha256"]
        assert admitted.design_provenance["original_design_size"] == 2
        assert admitted.design_provenance["candidate_position"] == row["candidate_position"]
        assert admitted.design_provenance["origin"] == row["origin"]
        assert admitted.case.metadata_payload()["schema_kind"] == "generation_source_case"
        assert admitted.case.metadata_payload()["domain_profile"] == "reference_range"
        timing = admitted.load_timing()
        assert timing.case_id == row["case_id"]
        assert timing.batch_id == admitted.source_identity["source_batch_id"]
        assert timing.simulation_profile == "steady_flow"
        with h5py.File(admitted.hdf5_path, "r") as handle:
            assert handle.attrs["schema_kind"] == "generation_source_case_hdf5"
            assert handle.attrs["domain_profile"] == "reference_range"
            assert "material_role" not in handle.attrs
            assert "evaluation_regime" not in handle.attrs

    failed_selector = {key: rows[0].get(key) for key in selector}
    with pytest.raises(ValueError, match="successful"):
        catalog.resolve_raw_case_reference("steady_flow", failed_selector, storage_root=storage)

    driver_log = tmp_path / "runtime/logs/generation/generation-run-4711.log"
    driver_log.parent.mkdir(parents=True)
    driver_log.write_text("[1/4] Run plan\n", encoding="utf-8")
    host_service.link_job_logs(root, [driver_log])
    assert sorted(entry.name for entry in root.iterdir()) == ["_record", "materials"]  # cases.csv follows when the campaign settles
    assert not any(path.name == ".state" for path in root.parent.rglob("*"))  # the stage's single .state sits beside the task folders
    locks = {path.name for path in (storage / "01_generation/.state/locks").iterdir()}
    prefix = f"steady_flow~{campaign.source_name}~"
    assert {f"{prefix}controller.lock", f"{prefix}lentil~test~inputs.lock", f"{prefix}lentil~test~case_0001.lock"} <= locks
    assert all(name.startswith(prefix) and name.endswith(".lock") for name in locks)
    assert sorted(entry.name for entry in (root / "materials").iterdir()) == ["lentil"]
    assert sorted(entry.name for entry in (root / "materials/lentil").iterdir()) == ["_record", "case_0001", "case_0002", "case_0003"]
    assert _tree(root / "_record") == {"campaign_plan.json", "execution", "execution/run.json", "logs", "logs/generation-run-4711.log"}
    assert _tree(root / "materials/lentil/_record") == {
        "amendments",
        "amendments/amendment_0001.json",
        "test",
        "test/input_generation_manifest.json",
        "test/resolved_generation_config.json",
        "test_replacement_0001",
        "test_replacement_0001/input_generation_manifest.json",
        "test_replacement_0001/resolved_generation_config.json",
    }
    inputs_tree = {"inputs", "inputs/case.json", "inputs/fields.csv", "inputs/input_reference.json"}
    records = {"_record", "_record/reservation.json", "_record/decision.json"}
    bundle = "_record/attempts/run_0001"
    failed_runtime = ("execution_provenance.json", "solver_steps.csv", "stderr.log", "stdout.log", "timing.json")
    assert _tree(root / "materials/lentil/case_0001") == {
        *inputs_tree,
        *records,
        "_record/attempts",
        bundle,
        f"{bundle}/attempt.json",
        f"{bundle}/cleanup.json",
        f"{bundle}/payload",
        f"{bundle}/payload/runtime",
        *(f"{bundle}/payload/runtime/{name}" for name in failed_runtime),
    }
    evidence = ("execution_provenance.json", "processing_provenance.json", "solver.log", "status.json", "timing.json")
    assert _tree(root / "materials/lentil/case_0002") == {
        *inputs_tree,
        *records,
        "result",
        "result/_SUCCESS",
        "result/case.h5",
        "result/provenance.json",
        "result/evidence",
        *(f"result/evidence/{name}" for name in evidence),
    }


@pytest.mark.parametrize(("time_mode", "expected_duration"), [("exact_stop", 1.5), ("short_rollout", 2.5)])
def test_transient_source_timing_admits_one_success_before_campaign_completion(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_mode: str,
    expected_duration: float,
) -> None:
    """Read true transient timing from a selected candidate in an incomplete source campaign."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying", executable=fake_comsol, natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    batch = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(batch, 2, storage_root=storage)
    monkeypatch.setenv("FAKE_COMSOL_TRANSIENT_TIME_MODE", time_mode)
    successful_source_case(campaign.source_name, batch, 1, storage, tmp_path / "work")
    run_id = campaign_runtime.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = campaign_runtime._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={
            batch.batch_name: {
                "input_generation_id": inputs.input_generation_id,
                "source_git_commit": "a" * 40,
                "case_indices": list(batch.case_indices),
            }
        },
    )
    common.serialization.atomic_write_json(campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), manifest)
    assert not (root / "_record/selected_pool.json").exists()
    rows = catalog.list_campaign_cases("transient_drying", campaign.source_name, storage_root=storage)
    assert [(row["case_id"], row["outcome"]) for row in rows] == [("case_0001", "successful"), ("case_0002", "pending")]
    selector = {
        key: rows[0][key]
        for key in (
            "campaign_name",
            "campaign_digest",
            "material_family",
            "case_id",
            "case_input_id",
            "simulation_case_id",
            "case_hdf5_sha256",
        )
    }
    admitted = catalog.resolve_raw_case_reference("transient_drying", selector, storage_root=storage)
    timing = admitted.load_timing()
    assert timing.case_id == "case_0001"
    assert timing.simulation_profile == "transient_drying"
    assert timing.physical_duration_hours is not None
    assert timing.physical_duration_hours == expected_duration
    assert admitted.design_provenance["original_design_size"] == 2
    assert admitted.source_identity["selected_pool_sha256"] is None
    corrupted = tmp_path / f"corrupted-scalar-{time_mode}.h5"
    shutil.copyfile(admitted.hdf5_path, corrupted)
    with h5py.File(corrupted, "r+") as handle:
        scalar = handle["provenance/scalar_handoff_json"]
        assert isinstance(scalar, h5py.Dataset)
        scalar_json = scalar.asstr()[()]
        assert isinstance(scalar_json, str)
        payload = json.loads(scalar_json)
        payload["entries"].append({"name": "unknown_scalar", "value": 1.0, "unit": "1", "owner": "case_dependent"})
        del handle["provenance/scalar_handoff_json"]
        provenance = handle["provenance"]
        assert isinstance(provenance, h5py.Group)
        provenance.create_dataset("scalar_handoff_json", data=json.dumps(payload), dtype=h5py.string_dtype())
    with pytest.raises(ValueError, match="missing, duplicate, unknown"):
        generation.publication.storage.validate_case_hdf5(corrupted, expected_profile="transient_drying")


def test_staged_admission_resumes_same_full_design_after_first_success(
    generation_config_factory: Any,
    fake_comsol: Path,
    world: FakeSlurm,
    tmp_path: Path,
) -> None:
    """One operational cap stages a full design, then a recorded cap increase admits its next original."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=3,
        max_admission_cases=1,
        node_jobs=1,
        cores_per_node=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    original_digest = campaign.campaign_digest
    original_batch = campaign.batches[0]
    storage = world.storage
    first = submit(campaign, world)
    assert world.leases == ["7001"]
    assert [(item["case"]["case_id"], item["mode"], item["job_id"]) for item in first["submissions"]] == [("case_0001", "initial", "7001")]
    assert lifecycle.load_campaign_plan("steady_flow", campaign.source_name, storage_root=storage)["original_design_sizes"] == {"lentil": {"test": 3}}

    successful_source_case(campaign.source_name, original_batch, 1, storage, tmp_path / "work")
    world.finish_dispatches(first)
    capped = feed(campaign, world)
    assert len(capped["submissions"]) == 1
    assert capped["state"] == "staged"
    assert capped["node_jobs"][0]["draining"] is True  # the staged campaign stops its idle node job
    world.set_state("7001", "COMPLETED")

    execution_path = config_path.parent / "execution.yaml"
    original_execution = execution_path.read_bytes()
    resumed_config = generation.cases.config.load_campaign_config(config_path)
    assert resumed_config.campaign_digest == original_digest
    assert resumed_config.batches[0].case_indices == original_batch.case_indices
    resumed = submit(resumed_config, world, max_admission_cases=2)
    assert [(item["case"]["case_id"], item["mode"], item["job_id"]) for item in resumed["submissions"]] == [
        ("case_0001", "initial", "7001"),
        ("case_0002", "initial", "7002"),
    ]
    assert [epoch["submission_config"]["max_admission_cases"] for epoch in lifecycle.execution_epochs(resumed)] == [1, 2]
    assert [epoch["first_submission_index"] for epoch in lifecycle.execution_epochs(resumed)] == [1, 2]
    assert [row["reservation"]["case_id"] for row in lifecycle.candidate_records("steady_flow", campaign.source_name, storage_root=storage)] == [
        "case_0001",
        "case_0002",
        "case_0003",
    ]

    second_input = generation.cases.input_generation.admit_configured_input_references(original_batch, storage_root=storage)[2].case_payload()
    lifecycle.decide_candidate(
        "steady_flow",
        campaign.source_name,
        "lentil",
        "case_0002",
        outcome="failed",
        stage="solver",
        failure_class="solver",
        reason_code="synthetic_solver_failure",
        detail="Test-owned failed scientific candidate",
        case_input_id=second_input["case_input_id"],
        simulation_case_id=second_input["simulation_case_id"],
        case_hdf5_relative=None,
        case_hdf5_sha256=None,
        evidence={},
        storage_root=storage,
    )
    world.finish_dispatches(resumed, "FAILED", "1:0")
    world.set_state("7002", "COMPLETED")
    after_failure = generation.cases.config.load_campaign_config(config_path)
    stopped = submit(after_failure, world, max_admission_cases=3)
    assert [item["case"]["case_id"] for item in stopped["submissions"]] == ["case_0001", "case_0002"]
    assert len(world.sbatch_arguments()) == 2
    assert execution_path.read_bytes() == original_execution


def test_four_case_smoke_capacity_continuation_keeps_original_design(
    generation_config_factory: Any,
    fake_comsol: Path,
    world: FakeSlurm,
    tmp_path: Path,
) -> None:
    """A cap-one first solve and cap-four continuation use the same four original cases."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        natural_count=4,
        max_admission_cases=1,
        node_jobs=1,
        cores_per_node=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    storage = world.storage
    original_config = config_path.read_bytes()
    original_execution = (config_path.parent / "execution.yaml").read_bytes()
    first = submit(campaign, world)
    assert [item["case"]["case_id"] for item in first["submissions"]] == ["case_0001"]
    successful_source_case(campaign.source_name, batch, 1, storage, tmp_path / "work")
    world.finish_dispatches(first)
    staged = feed(campaign, world)
    assert [item["case"]["case_id"] for item in staged["submissions"]] == ["case_0001"]
    world.set_state("7001", "COMPLETED")  # the staged campaign drained its node job

    continued = submit(campaign, world, max_admission_cases=4)
    assert [epoch["submission_config"]["max_admission_cases"] for epoch in lifecycle.execution_epochs(continued)] == [1, 4]
    assert [item["case"]["case_id"] for item in continued["submissions"]] == [batch.case_id(index) for index in batch.case_indices]
    assert {item["job_id"] for item in continued["submissions"][1:]} == {"7002"}  # one new node job takes the three remaining cases as tickets
    for index in range(2, 5):
        successful_source_case(campaign.source_name, batch, index, storage, tmp_path / "work")
    world.finish_dispatches(continued)
    continued = feed(campaign, world)
    assert continued["state"] == "complete"
    assert lifecycle.load_campaign_plan("transient_drying", campaign.source_name, storage_root=storage)["original_design_sizes"] == {
        "lentil": {"test": 4}
    }
    assert len(lifecycle.selected_successes("transient_drying", campaign.source_name, storage_root=storage)) == 4
    assert config_path.read_bytes() == original_config
    assert (config_path.parent / "execution.yaml").read_bytes() == original_execution


def _fail_candidate(run_id: str, material: str, case_id: str, *, storage: Path) -> Path:
    """Exhaust one candidate with a non-retryable solver attempt, then declare it terminal through the public operation."""
    lifecycle.record_candidate_attempt(
        "steady_flow",
        run_id.rsplit("~", 1)[-1],
        material,
        case_id,
        attempt_index=1,
        stage="solver",
        failure_class="solver",
        reason_code=lifecycle.UNCLASSIFIED_SOLVER_REASON,
        retryable=False,
        detail="Synthetic non-converging solve",
        evidence={},
        storage_root=storage,
    )
    return campaign_runtime.declare_candidate_failure(
        run_id, material, case_id, reason_code="synthetic_nonconvergence", detail="Synthetic non-converging solve", storage_root=storage
    )


def test_failed_replacement_is_terminalized_and_the_next_replacement_completes_selection(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """A failed replacement passes fail-case by its own execution batch, frees the slot again, and the next Sobol point fills it."""
    base_path, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=2)
    campaign = generation.cases.config.load_campaign_config(base_path)
    storage = tmp_path / "scientific-storage"
    lifecycle.create_campaign(campaign, storage_root=storage)
    primary = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(primary, 2, storage_root=storage)
    run_id = campaign_runtime.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = campaign_runtime._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={
            primary.batch_name: {
                "input_generation_id": inputs.input_generation_id,
                "source_git_commit": "a" * 40,
                "case_indices": list(primary.case_indices),
            }
        },
    )
    common.serialization.atomic_write_json(campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), manifest)
    successful_source_case(campaign.source_name, primary, 2, storage, tmp_path / "work")
    assert _fail_candidate(run_id, "lentil", "case_0001", storage=storage).is_file()
    # One success in three decisions keeps the bound (0.865) at or above 0.8, so the second replacement is still admitted.
    lifecycle.amend_minimum_success_rate(
        "steady_flow",
        campaign.source_name,
        primary.material_family,
        minimum_success_rate=0.8,
        reason="Synthetic tolerated failures",
        storage_root=storage,
    )

    first = lifecycle.materialize_replacement_batch(campaign, primary.batch_id, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(first, 1, storage_root=storage)
    assert first.case_indices == (3,)
    decision = _fail_candidate(run_id, "lentil", "case_0003", storage=storage)
    assert json.loads(decision.read_text(encoding="utf-8"))["outcome"] == "failed"
    assert (
        campaign_runtime.declare_candidate_failure(
            run_id, "lentil", "case_0003", reason_code="synthetic_nonconvergence", detail="Synthetic non-converging solve", storage_root=storage
        )
        == decision
    )
    with pytest.raises(FileExistsError, match="different immutable terminal decision"):
        campaign_runtime.declare_candidate_failure(
            run_id, "lentil", "case_0003", reason_code="other_reason", detail="Synthetic non-converging solve", storage_root=storage
        )

    second = lifecycle.materialize_replacement_batch(campaign, primary.batch_id, storage_root=storage)
    design = second.scientific_values["replacement_design"]
    assert (second.case_indices, design["candidate_ordinal"], design["target_slot"], design["parent_failed_case_id"]) == ((4,), 2, 1, "case_0001")
    generation.cases.input_generation.generate_input_cases(second, 1, storage_root=storage)
    successful_source_case(campaign.source_name, second, 4, storage, tmp_path / "work")
    selected = lifecycle.selected_successes("steady_flow", campaign.source_name, storage_root=storage)
    assert [(row["reservation"]["case_id"], row["reservation"]["origin"]) for row in selected] == [
        ("case_0002", "primary"),
        ("case_0004", "replacement"),
    ]
    assert [row["outcome"] for row in catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)] == [
        "failed",
        "successful",
        "failed",
        "successful",
    ]
    with pytest.raises(lifecycle.NoReplacementError, match="success deficit is closed"):
        lifecycle.materialize_replacement_batch(campaign, primary.batch_id, storage_root=storage)


def test_replacement_batch_passes_terminal_validation_under_its_canonical_identity(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """The worker finalizes a published replacement; the terminal validator binds its suffixed name, locator, and identifier."""
    base_path, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=2)
    campaign = generation.cases.config.load_campaign_config(base_path)
    storage = tmp_path / "scientific-storage"
    lifecycle.create_campaign(campaign, storage_root=storage)
    primary = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(primary, 2, storage_root=storage)
    first = generation.cases.input_generation.admit_configured_input_references(primary, storage_root=storage)[1].case_payload()
    lifecycle.decide_candidate(
        "steady_flow",
        campaign.source_name,
        "lentil",
        "case_0001",
        outcome="failed",
        stage="validation",
        failure_class="validation",
        reason_code="synthetic_rejection",
        detail="Synthetic scientific rejection",
        case_input_id=first["case_input_id"],
        simulation_case_id=first["simulation_case_id"],
        case_hdf5_relative=None,
        case_hdf5_sha256=None,
        evidence={"input_manifest_relative": (inputs.metadata_directory / "input_generation_manifest.json").relative_to(storage).as_posix()},
        storage_root=storage,
    )
    successful_source_case(campaign.source_name, primary, 2, storage, tmp_path / "work")
    lifecycle.amend_minimum_success_rate(
        "steady_flow",
        campaign.source_name,
        primary.material_family,
        minimum_success_rate=0.9,
        reason="Synthetic tolerated failure",
        storage_root=storage,
    )
    replacement = lifecycle.materialize_replacement_batch(campaign, primary.batch_id, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(replacement, 1, storage_root=storage)
    successful_source_case(campaign.source_name, replacement, 3, storage, tmp_path / "work")

    manifest_path = generation.runtime.finalize_batch(replacement, storage_root=storage)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["batch_name"] == f"{primary.batch_name}__replacement_0001"
    assert manifest["batch_storage_name"] == f"source~steady_flow~{campaign.source_name}~lentil~test_replacement_0001"
    assert generation.runtime.admit_terminal_batch(replacement.batch_storage_name, storage_root=storage).batch_name == replacement.batch_name

    for wrong_name in (primary.batch_name, f"{primary.batch_name}__replacement_0002"):
        manifest_path.write_text(json.dumps({**manifest, "batch_name": wrong_name}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Terminal batch name, purpose, storage locator, or immutable identifier is invalid"):
            generation.runtime.admit_terminal_batch(replacement.batch_storage_name, storage_root=storage)


@dataclass
class _FeederCampaign:
    """One two-original steady Source with an authored minimum success rate of 0.8, fed through a fake scheduler."""

    campaign: Any
    storage: Path
    work: Path
    run_id: str
    world: FakeSlurm

    def fail(self, case_id: str) -> dict[str, Any]:
        """Declare one exhausted candidate terminal through the public operation, then let the feeder plan its replacement."""
        _fail_candidate(self.run_id, "lentil", case_id, storage=self.storage)
        return campaign_runtime.feed_campaign(self.run_id, storage_root=self.storage)

    def end_step(self, case_id: str, state: str, exit_code: str) -> str:
        """Add the terminal step accounting of the latest dispatch of ``case_id`` and return its dispatch key."""
        manifest = campaign_evidence.load_campaign_run(self.run_id, storage_root=self.storage)
        record = [item for item in manifest["submissions"] if item["case"]["case_id"] == case_id][-1]
        self.world.add_step(record["job_id"], int(record["ticket_seq"]) - 1, record["step_name"], state, exit_code)
        return f"{record['job_id']}-{record['submission_index']}"

    def batch(self, case_index: int) -> Any:
        """Return the frozen execution batch of one case, originals and replacements alike."""
        manifest = campaign_evidence.load_campaign_run(self.run_id, storage_root=self.storage)
        return {batch.case_indices: batch for batch in lifecycle.campaign_from_frozen_run(manifest).batches}[(case_index,)]

    def run_step(self, monkeypatch: pytest.MonkeyPatch, case_id: str, *, error: Exception | None = None, comsol_mode: str = "success") -> None:
        """Run the latest dispatch of ``case_id`` through the case worker: ``error`` stands in for its solve, else fake COMSOL runs."""
        manifest = campaign_evidence.load_campaign_run(self.run_id, storage_root=self.storage)
        dispatches = [(position, item) for position, item in enumerate(manifest["submissions"], start=1) if item["case"]["case_id"] == case_id]
        position, record = dispatches[-1]
        assert record["submission_index"] == position
        cores = str(manifest["submission_config"]["cores_per_case"])

        def stopped(*_args: Any, **_kwargs: Any) -> NoReturn:
            assert error is not None
            raise error

        with monkeypatch.context() as step:
            bind_worker(step, manifest, index=position, step=str(int(record["ticket_seq"]) - 1), cpus=cores)
            step.setattr(source_service, "required_git_commit", lambda: COMMIT)
            step.setenv("FAKE_COMSOL_MODE", comsol_mode)
            if error is not None:
                step.setattr(campaign_runtime.cluster_service, "run_campaign_case", stopped)
            with pytest.raises((generation.runtime.CaseExecutionError, license_service.TemporaryLicenseCapacityError)):
                campaign_runtime.run_campaign_case_job(
                    self.run_id, record["case"]["batch_name"], int(record["case"]["case_index"]), storage_root=self.storage, work_root=self.work
                )

    def decision(self, case_id: str) -> dict[str, Any] | None:
        """Return the recorded terminal decision of one lentil candidate, if any."""
        records = lifecycle.candidate_records("steady_flow", self.campaign.source_name, storage_root=self.storage)
        return next(row["decision"] for row in records if row["reservation"]["case_id"] == case_id)


_STALLED = "stalled: 0.00 h of 1 h simulated after 900 s solving"


@pytest.fixture
def feeder_campaign(
    generation_config_factory: Any, fake_comsol: Path, world: FakeSlurm, tmp_path: Path, request: pytest.FixtureRequest
) -> _FeederCampaign:
    """Submit both originals of an s = 0.8 Source as tickets of one node job; the first failed, the second succeeded.

    With one success the bound stays at or above 0.8 after three decisions (0.865) and falls below it after four (0.751).
    Without an explicit cap (a test's ``param``) the cap is derived: the plan default 4 (two originals plus two tolerated failures).
    """
    explicit = getattr(request, "param", None)
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=2,
        node_jobs=1,
        cores_per_node=2,
        timeout=1.0,  # fake COMSOL's timeout mode outlasts it, so the runtime times the solve out
        license_retry_enabled=False,  # its silent startup would otherwise read as license capacity
    )
    authored = {**yaml.safe_load(config_path.read_text(encoding="utf-8")), "minimum_success_rate": 0.8}
    campaign = generation.cases.config.load_campaign_config(
        config_path, authored_value=authored, execution_overrides=None if explicit is None else {"max_admission_cases": explicit}
    )
    first = submit(campaign, world)
    assert [(item["job_id"], item["ticket_seq"]) for item in first["submissions"]] == [("7001", 1), ("7001", 2)]
    successful_source_case(campaign.source_name, campaign.batches[0], 2, world.storage, tmp_path / "work")
    fed = _FeederCampaign(campaign, world.storage, tmp_path / "work", first["campaign_run_id"], world)
    fed.end_step("case_0001", "FAILED", "1:0")
    fed.end_step("case_0002", "COMPLETED", "0:0")
    return fed


def test_feeder_replaces_failed_original_and_failed_replacement_while_the_bound_reaches_s(feeder_campaign: _FeederCampaign) -> None:
    """Replenishment survives a failed replacement: fail-case frees its slot, the feeder dispatches the next Sobol point, selection seals."""
    fed = feeder_campaign
    after_first = fed.fail("case_0001")
    assert [item["case"]["case_id"] for item in after_first["submissions"]] == ["case_0001", "case_0002", "case_0003"]
    assert after_first["submissions"][-1]["case"]["batch_id"] == fed.batch(3).batch_id
    fed.end_step("case_0003", "FAILED", "1:0")
    after_second = fed.fail("case_0003")
    assert [item["case"]["case_id"] for item in after_second["submissions"]][-1] == "case_0004"
    assert fed.batch(4).scientific_values["replacement_design"]["parent_failed_case_id"] == "case_0001"
    pending = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)
    assert {case["case_id"]: case["terminal_decision"] for case in pending["cases"]} == {
        "case_0001": "failed",
        "case_0002": "successful",
        "case_0003": "failed",
        "case_0004": None,
    }
    successful_source_case(fed.campaign.source_name, fed.batch(4), 4, fed.storage, fed.work)
    fed.end_step("case_0004", "COMPLETED", "0:0")
    done = campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert done["state"] == "complete"
    settled = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)
    assert settled["counts"]["replacement_deficit"] == 0
    assert [case["terminal_decision"] for case in settled["cases"] if case["case_id"] == "case_0004"] == ["successful"]
    pool = lifecycle.load_selected_pool("steady_flow", fed.campaign.source_name, storage_root=fed.storage)
    assert [row["case_id"] for row in pool["selected"]] == ["case_0002", "case_0004"]
    assert [(item["job_id"], item["ticket_seq"]) for item in done["submissions"]] == [("7001", 1), ("7001", 2), ("7001", 3), ("7001", 4)]
    assert len(fed.world.sbatch_arguments()) == 1  # every replacement is a ticket of the same node job


@pytest.mark.parametrize(("feeder_campaign", "derived"), [(None, True), (4, False)], indirect=["feeder_campaign"])
def test_a_lowered_rate_raises_only_a_derived_admission_cap(feeder_campaign: _FeederCampaign, derived: bool) -> None:
    """A derived cap (null) grows from 4 = 2 + 2 to 2 + 4 at s = 0.6 and admits case 5; an explicit cap equal to 4 holds it back."""
    fed = feeder_campaign
    recorded = campaign_evidence.load_campaign_run(fed.run_id, storage_root=fed.storage)["submission_config"]["max_admission_cases"]
    assert recorded == (None if derived else 4)
    fed.fail("case_0001")
    fed.end_step("case_0003", "FAILED", "1:0")
    fed.fail("case_0003")
    material = fed.campaign.batches[0].material_family
    lifecycle.amend_minimum_success_rate(
        "steady_flow", fed.campaign.source_name, material, minimum_success_rate=0.6, reason="Synthetic", storage_root=fed.storage
    )
    fed.end_step("case_0004", "FAILED", "1:0")
    after = fed.fail("case_0004")  # k/n 1/4 keeps the bound 0.751 at or above 0.6
    assert ([item["case"]["case_id"] for item in after["submissions"]][-1] == "case_0005") is derived
    assert after["state"] == ("active" if derived else "blocked")
    rendered = status_formatting.format_campaign_status_summary(campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage))
    assert ("max_admission_cases=6 (plan rates)" if derived else "max_admission_cases=4") in rendered
    assert ("Admission: 5/6" if derived else "Admission: 4/4") in rendered


def test_a_rerun_compares_effective_admission_caps(feeder_campaign: _FeederCampaign) -> None:
    """A rerun below the derived cap 4 is refused; switching to an explicit cap that keeps the effective cap is accepted."""
    fed = feeder_campaign
    fed.world.set_state("7001", "COMPLETED")
    with pytest.raises(ValueError, match="cannot decrease"):
        submit(fed.campaign, fed.world, max_admission_cases=3)
    continued = submit(fed.campaign, fed.world, max_admission_cases=4)
    assert [epoch["submission_config"]["max_admission_cases"] for epoch in lifecycle.execution_epochs(continued)] == [None, 4]


@pytest.mark.parametrize("feeder_campaign", [3], indirect=True)
def test_an_explicit_admission_cap_that_holds_back_a_needed_replacement_ends_blocked_and_names_the_cap(
    feeder_campaign: _FeederCampaign,
) -> None:
    """An explicit cap stays as given: once it holds back the replacement a slot needs, the campaign ends blocked with its way out."""
    fed = feeder_campaign
    fed.fail("case_0001")
    fed.end_step("case_0003", "FAILED", "1:0")
    blocked = fed.fail("case_0003")  # k/n 1/3 keeps the batch open, the cap 3 holds case_0004 back
    assert blocked["state"] == "blocked"
    assert [item["case"]["case_id"] for item in blocked["submissions"]] == ["case_0001", "case_0002", "case_0003"]
    status = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)
    assert status["success_rate_stop"] is None
    assert "max_admission_cases 3" in status["admission_cap_stop"]
    assert "--max-admission-cases" in status["admission_cap_stop"]


def test_operational_error_of_a_replacement_stops_feeding_by_its_own_task_view(feeder_campaign: _FeederCampaign) -> None:
    """A retryable controller error of a replacement is matched to the replacement's task, not to its parent batch."""
    fed = feeder_campaign
    fed.fail("case_0001")
    lifecycle.record_candidate_attempt(
        "steady_flow",
        fed.campaign.source_name,
        "lentil",
        "case_0003",
        attempt_index=1,
        stage="publication",
        failure_class="publication",
        reason_code="synthetic_publication_error",
        retryable=True,
        detail="Synthetic worker-side publication error",
        evidence={"dispatch_key": "7001-3"},
        storage_root=fed.storage,
    )
    assert fed.end_step("case_0003", "FAILED", "1:0") == "7001-3"
    with pytest.raises(RuntimeError, match="unexpected controller or publication error for case_0003 in Slurm step dispatch 7001-3"):
        campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)


def test_the_feeder_decides_stalled_and_timed_out_solves_itself_and_writes_the_cases_view(
    feeder_campaign: _FeederCampaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime-stopped solves get decisions with their typed reason once their step ended, are replaced, and the Source completes."""
    fed = feeder_campaign
    fed.run_step(monkeypatch, "case_0001", error=generation.runtime.CaseExecutionError(_STALLED, work_directory=fed.work, runtime_reason="stalled"))
    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    decided = fed.decision("case_0001")
    assert decided is not None
    assert (decided["outcome"], decided["class"], decided["reason_code"], decided["detail"]) == ("failed", "solver", "stalled", _STALLED)
    replaced = campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert replaced["submissions"][-1]["case"]["case_id"] == "case_0003"

    # The real runtime times the replacement out; while its step has no terminal accounting nothing is decided.
    fed.run_step(monkeypatch, "case_0003", comsol_mode="timeout")
    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert fed.decision("case_0003") is None
    fed.end_step("case_0003", "FAILED", "1:0")
    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    timed_out = fed.decision("case_0003")
    assert timed_out is not None
    assert (timed_out["reason_code"], timed_out["detail"].startswith("timed_out: ")) == ("timed_out", True)
    status = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)
    lineage = {case["case_id"]: (case["parent_failed_case_id"], case["failure_reason"]) for case in status["cases"]}
    assert lineage == {"case_0001": (None, "stalled"), "case_0002": (None, None), "case_0003": ("case_0001", "timed_out")}

    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    status = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)
    assert controller_service._unsettled_failure(fed.run_id, status) is None  # the driver needs no fail-case
    successful_source_case(fed.campaign.source_name, fed.batch(4), 4, fed.storage, fed.work)
    fed.end_step("case_0004", "COMPLETED", "0:0")
    assert campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)["state"] == "complete"
    view = common.paths.resolve_generation_campaign_directory("steady_flow", fed.campaign.source_name, storage_root=fed.storage) / "cases.csv"
    written = view.read_bytes()
    assert sorted(entry.name for entry in view.parent.iterdir()) == ["_record", "cases.csv", "materials"]  # a settled campaign has three entries
    assert not any(path.name == ".state" for path in view.parent.parent.rglob("*"))
    rows = list(csv.DictReader(written.decode("utf-8").splitlines()))
    assert [(row["case_id"], row["origin"], row["outcome"], row["reason_code"]) for row in rows] == [
        ("case_0001", "original", "failed", "stalled"),
        ("case_0002", "original", "successful", ""),
        ("case_0003", "replacement", "failed", "timed_out"),
        ("case_0004", "replacement", "successful", ""),
    ]
    sampled = json.loads((view.parent / "materials/lentil/case_0003/inputs/case.json").read_text(encoding="utf-8"))["sampled_values"]
    registry = lifecycle.load_campaign_plan("steady_flow", fed.campaign.source_name, storage_root=fed.storage)["batches"][0]["scientific_values"][
        "material"
    ]["parameter_registry"]
    columns = [name for name, entry in registry.items() if entry["classification"] == "sampled"]
    header = list(rows[2])
    assert header == ["material", "case_id", "origin", "outcome", "reason_code", "detail", "duration_s", *columns, "solver_log"]
    assert not {"T_flow_ref", "p_ref", "p_out", "simulated_end_h", "final_bulk_moisture_wb"} & set(header)
    assert {name: float(rows[2][name]) for name in columns} == pytest.approx({name: sampled[name] for name in columns})
    assert rows[2]["solver_log"] == "materials/lentil/case_0003/_record/attempts/run_0001/payload/runtime/stdout.log"
    assert (view.parent / rows[2]["solver_log"]).is_file()
    assert rows[0]["solver_log"] == ""  # the stand-in for the stalled solve left no failure bundle
    assert float(rows[3]["duration_s"]) >= 0
    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert view.read_bytes() == written
    assert len(fed.world.sbatch_arguments()) == 1  # the automatic decisions never drained the node job before the replacement


def test_a_license_capacity_failure_without_protection_is_decided_like_a_solver_error(
    feeder_campaign: _FeederCampaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the licence protection off (the fixture's), capacity is a solver error: decided, counted and replaced automatically."""
    fed = feeder_campaign
    fed.run_step(monkeypatch, "case_0001", comsol_mode="license_capacity")
    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    decided = fed.decision("case_0001")
    assert decided is not None
    assert (decided["outcome"], decided["class"], decided["reason_code"]) == ("failed", "solver", "solver_error")
    assert "license capacity" in decided["detail"]
    replaced = campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert replaced["submissions"][-1]["case"]["case_id"] == "case_0003"


def test_a_validation_failure_is_never_decided_by_the_feeder_and_stops_the_driver_with_fail_case(
    feeder_campaign: _FeederCampaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only runtime-stopped solves are decided automatically; an export failure keeps waiting for an explicit decision."""
    fed = feeder_campaign
    error = generation.runtime.CaseExecutionError("Missing airflow.csv export.", work_directory=fed.work, failure_stage="exports")
    fed.run_step(monkeypatch, "case_0001", error=error)
    campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert fed.decision("case_0001") is None
    status = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)
    message = controller_service._unsettled_failure(fed.run_id, status)
    assert message is not None
    assert "./scripts/generation fail-case" in message


def test_an_automatic_decision_that_stops_the_batch_ends_blocked_and_names_the_ways_out(
    feeder_campaign: _FeederCampaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fourth decision stops the batch; the campaign ends blocked, its idle node job is drained, and status names both ways out."""
    fed = feeder_campaign
    fed.fail("case_0001")
    fed.end_step("case_0003", "FAILED", "1:0")
    fed.fail("case_0003")
    fed.run_step(monkeypatch, "case_0004", error=generation.runtime.CaseExecutionError(_STALLED, work_directory=fed.work, runtime_reason="stalled"))
    fed.end_step("case_0004", "FAILED", "1:0")
    decided = campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert (fed.decision("case_0004") or {}).get("reason_code") == "stalled"
    assert [(job["draining"], job["stop_ticket_seq"]) for job in decided["node_jobs"]] == [(False, None)]
    blocked = campaign_runtime.feed_campaign(fed.run_id, storage_root=fed.storage)
    assert blocked["state"] == "blocked"
    assert [(job["draining"], job["stop_ticket_seq"]) for job in blocked["node_jobs"]] == [(True, 5)]
    assert len(blocked["submissions"]) == 4  # nothing was dispatched on the way to blocked
    stop = campaign_runtime.campaign_status(fed.run_id, storage_root=fed.storage)["success_rate_stop"]
    assert "lentil: k/n 1/4" in stop
    assert "--revision LABEL" in stop
    assert f"amend-replacements {fed.run_id} --material MATERIAL --minimum-success-rate S --reason TEXT" in stop
