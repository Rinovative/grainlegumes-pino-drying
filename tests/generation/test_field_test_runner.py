"""Run small steady field tests against a fake-COMSOL Source: guard, runner branches, resume, and roster builders.

Responsibilities:
  - Prove a roster run with a dead node job leaves every Source view byte-identical and seals each member once
  - Check every runner disposition branch: gate inputs and outputs, roster binding, licence, I/O while binding, solver,
    infeasible, and another CPU model, and that a linked attempt write path seals the member without a write through it
    while an uncopied runtime file is only noted
  - Check that resume leaves a live or ambiguous submission alone, settles a committed outcome after a crash, and that
    the stage-A and stage-B builders are deterministic
  - Check base admission per material, the material quota, and the replacement of a failed base by a new roster version
  - Check the node-job resources (current node values, the Source's case values) and one node job per seat over a
    near-equal share, whose never-started tickets resume redispatches

Design principles:
  - Steps run in-process through the real runner; only sbatch and the queue are a test-owned scheduler
  - Plan sizes are shrunk through the planning module's base counts, so a three-case Source suffices

This module does NOT:
  - Run licensed COMSOL, a Slurm node worker, or the analysis build
"""

from __future__ import annotations

import copy
import dataclasses
import errno
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
import yaml
from support import field_test_fixtures
from support.generation_fixtures import make_generation_config_factory, successful_source_case, write_fake_comsol

from src import common, generation
from src.analysis import field_tests as analysis_field_tests
from src.analysis.eda import eda_sources
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.field_tests import generation_field_tests_dispatch as dispatch
from src.generation.field_tests import generation_field_tests_planning as planning
from src.generation.field_tests import generation_field_tests_record as record
from src.generation.field_tests import generation_field_tests_roster as roster_service
from src.generation.field_tests import generation_field_tests_runner as runner
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_node as node_service

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.integration

TASK = "steady_flow"
_SELECTOR_KEYS = ("campaign_name", "campaign_digest", "material_family", "case_id", "case_input_id", "simulation_case_id", "case_hdf5_sha256")
_MERGE = "merge:permeability.anisotropy.max_ratio+permeability.anisotropy.strength"


@dataclass
class _World:
    """One fake-COMSOL steady Source with three Train cases and one validation and one test case outside the base region."""

    storage: Path
    work: Path
    source_name: str


@dataclass
class _Queue:
    """A test-owned scheduler: sbatch returns increasing job ids and the queue holds what the test puts there."""

    submitted: list[list[str]] = field(default_factory=list)
    active: dict[str, tuple[int, str, str]] = field(default_factory=dict)

    def scheduler(self) -> dispatch.Scheduler:
        return dispatch.Scheduler(
            submit=self._submit, active_jobs=lambda: dict(self.active), exclusions=lambda _partition, _cores: [], lease=lambda _job: True
        )

    def _submit(self, command: Sequence[str]) -> str:
        self.submitted.append(list(command))
        return str(1000 + len(self.submitted))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _World:
    """Build the Source with Train, validation, and test cases."""
    return _build_world(tmp_path, monkeypatch, {"train": 3, "validation": 1, "test": 1})


def _build_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, designs: dict[str, int]) -> _World:
    """Build the Source and its frozen run record, and shrink the plan to one base per stage."""
    project = tmp_path / "project"
    monkeypatch.setenv("PROJECT_ROOT", str(project))
    factory = make_generation_config_factory(project)
    config_path, _ = factory(simulation_profile=TASK, executable=write_fake_comsol(tmp_path), designs=designs)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "scientific-storage"
    generation.lifecycle.create_campaign(campaign, storage_root=storage)
    input_sources: dict[str, dict[str, Any]] = {}
    for batch in campaign.batches:
        inputs = generation.cases.input_generation.generate_input_cases(batch, len(batch.case_indices), storage_root=storage)
        for case_index in batch.case_indices:
            successful_source_case(campaign.source_name, batch, case_index, storage, tmp_path / "work")
        input_sources[batch.batch_name] = {
            "input_generation_id": inputs.input_generation_id,
            "source_git_commit": "a" * 40,
            "case_indices": list(batch.case_indices),
        }
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = generation.publication.campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources=input_sources,
    )
    common.serialization.atomic_write_json(
        generation.publication.campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), manifest
    )
    # The admitted-source bindings a frozen launcher tree would export; sbatch itself is the test-owned scheduler.
    monkeypatch.setenv("GENERATION_SOURCE_SHA256", "b" * 64)
    monkeypatch.setenv("GENERATION_NATIVE_VENV", str(common.paths.get_runtime_root().resolve() / "venvs" / "native"))
    monkeypatch.setattr(planning, "_BASES_PER_MATERIAL", {TASK: 1})
    monkeypatch.setattr(planning, "_FRESH_JOINT_BASES", 1)
    monkeypatch.setattr(planning, "_FRESH_CHALLENGE_BASES", 1)
    return _World(storage=storage, work=tmp_path / "work", source_name=campaign.source_name)


def _stage_a(world: _World, test_id: str = "screen") -> dict[str, Any]:
    roster, _evidence = planning.build_stage_a(TASK, test_id, world.source_name, storage_root=world.storage)
    return roster


def _small_roster(world: _World, test_id: str, *, extra: Sequence[dict[str, Any]] = (), full_row: bool = False) -> tuple[Path, dict[str, Any]]:
    """
    Write a roster of one base's gate, its R0 reference, the first challenge of an R0-only row, and any extra members.

    With ``full_row`` the roster holds every challenge of that row, the planned members a replacement base mirrors.
    """
    full = _stage_a(world)
    gate = next(item for item in full["members"] if item["role"] == "gate")
    reference = next(item for item in full["members"] if item["role"] == "reference" and item["reference_kind"] == "R0")
    challenges = [
        item
        for item in full["members"]
        if item["role"] == "challenge" and item["reference_member_id"] == reference["member_id"] and planning.reference_kinds(item["row"]) == ("R0",)
    ]
    row = [item for item in challenges if item["row"] == challenges[0]["row"]]
    roster = roster_service.build_roster(
        test_id=test_id,
        task=TASK,
        stage="A",
        base_source=full["base_source"],
        members=[gate, reference, *(row if full_row else row[:1]), *extra],
        max_attempts=planning.MAX_ATTEMPTS,
    )
    directory = common.paths.resolve_field_test_dir(TASK, test_id, storage_root=world.storage)
    roster_service.write_roster(directory, roster)
    return directory, roster


def _ids(roster: dict[str, Any]) -> dict[str, str]:
    return {item["role"]: item["member_id"] for item in roster["members"][:3]}


def _present(value: dict[str, Any] | None) -> dict[str, Any]:
    assert value is not None
    return value


def _run(world: _World, test_id: str, member_id: str, attempt: int) -> dict[str, Any] | None:
    return runner.run_member(TASK, test_id, member_id, attempt=attempt, cores=1, storage_root=world.storage, work_root=world.work)


def _submit_by_hand(directory: Path, member_id: str, attempt: int) -> None:
    roster, version = roster_service.load_roster_version(directory)
    binding = {
        "roster_version": version,
        "roster_sha256": roster["roster_sha256"],
        "member_sha256": roster_service.member_sha256(roster_service.member(roster, member_id)),
    }
    submission = {"node_token": "manual", "ticket_seq": attempt, "step_label": "manual", "submitted_at": record.utc_now(), **binding}
    record.write_submission(directory, member_id, attempt, submission)


def _run_jobs(world: _World, test_id: str, result: dict[str, Any]) -> None:
    for job in result["node_jobs"]:
        for member_id, attempt in job["members"]:
            _run(world, test_id, member_id, attempt)


def _snapshot(storage: Path) -> tuple[Any, ...]:
    root = storage / "01_generation"
    digests = {path.relative_to(root).as_posix(): common.serialization.file_sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}
    return (
        digests,
        repr(generation.publication.catalog.list_campaigns(TASK, storage_root=storage)),
        repr(generation.cases.admission.discover_input_batches(storage, task=TASK, validation_depth="evidence")),
        repr(eda_sources.discover_generated_output_eda_catalog(storage_root=storage)),
    )


def test_a_roster_run_with_a_dead_job_leaves_every_source_view_identical_and_seals_each_member_once(world: _World) -> None:
    before = _snapshot(world.storage)
    directory, roster = _small_roster(world, "guard")
    ids = _ids(roster)
    queue = _Queue()
    first = dispatch.dispatch(TASK, "guard", mode="submit", storage_root=world.storage, scheduler=queue.scheduler())
    assert [job["members"] for job in first["node_jobs"]] == [[[ids["gate"], 1]]]
    # The node job dies after the gate's step started and before it recorded an outcome.
    node_dir = record.node_directory(directory, first["node_jobs"][0]["token"])
    assert json.loads(node_service.ticket_path(node_dir, 2).read_text(encoding="utf-8"))["kind"] == node_service.STOP_KIND
    binding = _present(record.read_submission(directory, ids["gate"], 1))
    assert (binding["roster_version"], binding["member_sha256"]) == (1, roster_service.member_sha256(roster["members"][0]))
    common.serialization.atomic_write_json(
        node_service.receipt_path(node_dir, 1, "started"), {"schema": node_service.STARTED_RECEIPT_SCHEMA, "seq": 1}
    )
    resumed = dispatch.dispatch(TASK, "guard", mode="resume", storage_root=world.storage, scheduler=queue.scheduler())
    assert _present(record.read_outcome(directory, ids["gate"], 1))["classification"] == "infrastructure_stop"
    assert [job["members"] for job in resumed["node_jobs"]] == [[[ids["gate"], 2]]]
    _run_jobs(world, "guard", resumed)
    variants = dispatch.dispatch(TASK, "guard", mode="submit", storage_root=world.storage, scheduler=queue.scheduler())
    assert sorted(member for job in variants["node_jobs"] for member, _attempt in job["members"]) == sorted([ids["reference"], ids["challenge"]])
    _run_jobs(world, "guard", variants)

    # The second run finds every member sealed: nothing is dispatched and a stray step returns the existing seal.
    seals = {member_id: _present(record.read_seal(directory, member_id)) for member_id in ids.values()}
    assert {seal["disposition"] for seal in seals.values()} == {"succeeded"}
    for mode in ("submit", "resume"):
        again = dispatch.dispatch(TASK, "guard", mode=mode, storage_root=world.storage, scheduler=queue.scheduler())
        assert again["node_jobs"] == []
        assert again["states"] == {"sealed": 3}
    assert _run(world, "guard", ids["challenge"], 1) == seals[ids["challenge"]]
    assert [len(list((record.member_directory(directory, member_id)).glob("seal.json"))) for member_id in ids.values()] == [1, 1, 1]
    assert {member_id: record.attempt_numbers(directory, member_id) for member_id in ids.values()} == {
        ids["gate"]: (1, 2),
        ids["reference"]: (1,),
        ids["challenge"]: (1,),
    }
    gate = seals[ids["gate"]]
    assert gate["gate"]["inputs_equal"] is True

    assert _snapshot(world.storage) == before
    rows = generation.publication.catalog.list_campaign_cases(TASK, world.source_name, storage_root=world.storage)
    base = next(row for row in rows if row["case_id"] == roster["members"][0]["base"]["case_id"])
    selector = {key: base[key] for key in _SELECTOR_KEYS}
    challenge_seal = seals[ids["challenge"]]
    selector.update(simulation_case_id=challenge_seal["bundle_sha256"], case_hdf5_sha256=challenge_seal["case_h5_sha256"])
    with pytest.raises(ValueError, match=r"."):
        generation.publication.catalog.resolve_raw_case_reference(TASK, selector, storage_root=world.storage)


def test_a_variant_without_its_gate_is_gate_blocked(world: _World) -> None:
    directory, roster = _small_roster(world, "nogate")
    challenge = _ids(roster)["challenge"]
    _submit_by_hand(directory, challenge, 1)
    seal = _run(world, "nogate", challenge, 1)
    assert seal is not None
    assert (seal["disposition"], seal["failure"]["reason_code"]) == ("gate_blocked", "gate_missing")


def test_a_reproduced_solver_failure_seals_the_gate_and_dispatch_blocks_its_variants(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, roster = _small_roster(world, "failing")
    ids = _ids(roster)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    _submit_by_hand(directory, ids["gate"], 1)
    assert _run(world, "failing", ids["gate"], 1) is None
    assert _present(record.read_outcome(directory, ids["gate"], 1))["classification"] == "solver_failure"
    _submit_by_hand(directory, ids["gate"], 2)
    seal = _run(world, "failing", ids["gate"], 2)
    assert seal is not None
    assert (seal["disposition"], seal["attempts"], seal["case_h5_sha256"]) == ("solver_failure", 2, None)
    result = dispatch.dispatch(TASK, "failing", mode="submit", storage_root=world.storage, scheduler=_Queue().scheduler())
    assert sorted(result["gate_blocked"]) == sorted([ids["reference"], ids["challenge"]])
    assert result["node_jobs"] == []
    blocked = record.read_seal(directory, ids["challenge"])
    assert blocked is not None
    assert (blocked["disposition"], blocked["attempts"], blocked["failure"]["reason_code"]) == ("gate_blocked", 0, "gate_failed")


def _succeeded_gate(world: _World, test_id: str, *, extra: Sequence[dict[str, Any]] = ()) -> tuple[Path, dict[str, str], dict[str, Any]]:
    directory, roster = _small_roster(world, test_id, extra=extra)
    ids = _ids(roster)
    _submit_by_hand(directory, ids["gate"], 1)
    gate = _run(world, test_id, ids["gate"], 1)
    assert gate is not None
    assert gate["disposition"] == "succeeded"
    return directory, ids, roster


def _drift_regenerated_inputs(monkeypatch: pytest.MonkeyPatch, relative: float) -> None:
    """Shift every field map of every bundle a step regenerates, as another CPU type would in its last digits (or by more)."""
    original = runner.case_service.read_case_inputs

    def drifted(directory: Path, payload: Any, **kwargs: Any) -> Any:
        content = original(directory, payload, **kwargs)
        if "attempts" not in Path(directory).parts:
            return content
        fields = {name: values + relative * float(np.max(np.abs(values))) for name, values in content.fields.items()}
        return dataclasses.replace(content, fields=fields)

    monkeypatch.setattr(runner.case_service, "read_case_inputs", drifted)


@pytest.mark.parametrize(("relative", "expected"), [(1e-14, ("succeeded", None)), (1e-9, ("binding_refused", "gate_inputs_differ"))])
def test_a_gate_accepts_last_digit_input_differences_and_refuses_larger_ones(
    world: _World, monkeypatch: pytest.MonkeyPatch, relative: float, expected: tuple[str, str | None]
) -> None:
    directory, roster = _small_roster(world, "tolerance")
    gate = _ids(roster)["gate"]
    _drift_regenerated_inputs(monkeypatch, relative)
    _submit_by_hand(directory, gate, 1)
    seal = _present(_run(world, "tolerance", gate, 1))
    assert (seal["disposition"], (seal["failure"] or {}).get("reason_code")) == expected
    comparison = seal["input_comparison"]
    assert comparison["touched"] == []
    assert max(comparison["max_abs_diff"].values()) > 0.0
    assert record.gate_succeeded(seal) is (expected[0] == "succeeded")


def test_a_variant_refuses_untouched_inputs_that_drift_beyond_the_tolerance(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, ids, _roster = _succeeded_gate(world, "untouched")
    _drift_regenerated_inputs(monkeypatch, 1e-9)
    _submit_by_hand(directory, ids["challenge"], 1)
    seal = _present(_run(world, "untouched", ids["challenge"], 1))
    assert (seal["disposition"], seal["failure"]["reason_code"]) == ("binding_refused", "untouched_inputs_differ")
    assert not (record.attempt_directory(directory, ids["challenge"], 1) / "base_regenerated").exists()


def test_a_licence_shortage_is_an_infrastructure_stop_and_not_final(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, ids, _roster = _succeeded_gate(world, "licence")

    def capacity(*_args: Any, **_kwargs: Any) -> Any:
        error = license_service.TemporaryLicenseCapacityError.__new__(license_service.TemporaryLicenseCapacityError)
        RuntimeError.__init__(error, "Licensed number of users already reached.")
        raise error

    monkeypatch.setattr(runner.runtime_service, "execute_prepared_case", capacity)
    _submit_by_hand(directory, ids["challenge"], 1)
    assert _run(world, "licence", ids["challenge"], 1) is None
    outcome = record.read_outcome(directory, ids["challenge"], 1)
    assert outcome is not None
    assert (outcome["classification"], outcome["failure"]["reason_code"]) == ("infrastructure_stop", "license_capacity")


@pytest.mark.parametrize(
    ("service", "name", "code"), [("roster_service", "verified_base_payload", "base_io"), ("case_service", "read_case_inputs", "input_io")]
)
def test_an_io_error_while_binding_is_a_retryable_infrastructure_stop(
    world: _World, monkeypatch: pytest.MonkeyPatch, service: str, name: str, code: str
) -> None:
    directory, roster = _small_roster(world, f"io-{code}")
    gate = _ids(roster)["gate"]

    def unreadable(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(errno.EIO, "Input/output error")

    # The published base binding or the published base inputs, the first reads of their blocks, fail transiently.
    monkeypatch.setattr(getattr(runner, service), name, unreadable)
    _submit_by_hand(directory, gate, 1)
    assert _run(world, f"io-{code}", gate, 1) is None
    outcome = _present(record.read_outcome(directory, gate, 1))
    assert (outcome["classification"], outcome["failure"]["reason_code"]) == ("infrastructure_stop", code)


@pytest.mark.parametrize("name", ["bundle", "case.h5"])
def test_a_linked_attempt_write_path_ends_the_attempt_without_writing_through_it(world: _World, tmp_path: Path, name: str) -> None:
    directory, roster = _small_roster(world, "linked")
    gate = _ids(roster)["gate"]
    _submit_by_hand(directory, gate, 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    # The bundle is refused before generation, the case.h5 after the solver, when the step publishes its output.
    (record.attempt_directory(directory, gate, 1) / name).symlink_to(outside / name)
    seal = _present(_run(world, "linked", gate, 1))
    assert (seal["disposition"], seal["failure"]["reason_code"]) == ("binding_refused", "record_path")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("failure", ["linked", "failing"])
def test_missing_runtime_evidence_is_noted_without_changing_the_outcome_or_skipping_cleanup(
    world: _World, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    directory, roster = _small_roster(world, f"evidence-{failure}")
    gate = _ids(roster)["gate"]
    _submit_by_hand(directory, gate, 1)
    runtime = record.attempt_directory(directory, gate, 1) / "runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    copy, cleanup, cleanups = runner.shutil.copyfile, runner.workspace_service.cleanup_case_workspace, []

    def failing(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(target).parent.resolve() == runtime.resolve():
            raise OSError(errno.ENOSPC, "No space left on device")
        return copy(source, target, *args, **kwargs)

    def counted(work_directory: Path, **kwargs: Any) -> Any:
        cleanups.append(work_directory)
        return cleanup(work_directory, **kwargs)

    if failure == "linked":
        runtime.mkdir()
        (runtime / "stdout.log").symlink_to(outside / "stdout.log")
    else:
        monkeypatch.setattr(runner.shutil, "copyfile", failing)
    monkeypatch.setattr(runner.workspace_service, "cleanup_case_workspace", counted)
    seal = _present(_run(world, f"evidence-{failure}", gate, 1))
    notes = _present(record.read_outcome(directory, gate, 1))["notes"]
    # The solve's own success stands, the scratch is cleaned, and the uncopied file is noted, never written through a link.
    assert (seal["disposition"], len(cleanups)) == ("succeeded", 1)
    assert any(note.startswith("runtime evidence missing: stdout.log") for note in notes)
    assert list(outside.iterdir()) == []


def test_a_variant_on_another_cpu_model_than_its_gate_runs_and_records_its_model(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, ids, _roster = _succeeded_gate(world, "cpumodel")
    gate_model = _present(record.read_seal(directory, ids["gate"]))["cpu_model"]
    monkeypatch.setattr(runner.node_service, "read_cpu_model", lambda: "Another CPU model")
    _submit_by_hand(directory, ids["challenge"], 1)
    seal = _present(_run(world, "cpumodel", ids["challenge"], 1))
    assert (seal["disposition"], seal["cpu_model"]) == ("succeeded", "Another CPU model")
    assert gate_model != "Another CPU model"


def test_a_gate_whose_output_diverges_structurally_is_refused_without_a_member_case(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, roster = _small_roster(world, "divergent")
    gate = _ids(roster)["gate"]
    execute = runner._execute

    def nonfinite(*args: Any, **kwargs: Any) -> Path:
        produced = execute(*args, **kwargs)
        with h5py.File(produced, "r+") as handle:
            fields = handle["static/fields"]
            assert isinstance(fields, h5py.Dataset)
            fields[0, 0, 0] = np.nan
        return produced

    monkeypatch.setattr(runner, "_execute", nonfinite)
    _submit_by_hand(directory, gate, 1)
    seal = _present(_run(world, "divergent", gate, 1))
    assert (seal["disposition"], seal["failure"]["reason_code"]) == ("binding_refused", "gate_outputs_differ")
    assert seal["gate"]["output_divergences"] == ["nonfinite:static/fields"]
    assert not (record.member_directory(directory, gate) / "case.h5").exists()


def test_a_step_refuses_a_roster_entry_its_submission_does_not_bind(world: _World) -> None:
    directory, roster = _small_roster(world, "rebound")
    challenge = _ids(roster)["challenge"]
    _submit_by_hand(directory, challenge, 1)
    # After the ticket was written, the challenge's value is moved in place and the roster digest is recomputed.
    members = copy.deepcopy(roster["members"])
    members[2]["intervention"][members[2]["row"]] = members[1]["intervention"][members[2]["row"]]
    frame = {key: roster[key] for key in ("test_id", "task", "stage", "base_source", "max_attempts")}
    (directory / "roster.json").unlink()
    common.serialization.atomic_write_json(directory / "roster.json", roster_service.build_roster(members=members, **frame))
    seal = _present(_run(world, "rebound", challenge, 1))
    assert (seal["disposition"], seal["failure"]["reason_code"]) == ("binding_refused", "roster_binding")


def test_resume_reconciles_a_submission_without_a_job_id_by_its_token_and_receipts(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, roster = _small_roster(world, "ambiguous")
    gate = _ids(roster)["gate"]
    queue = _Queue()

    def unparsable(command: Sequence[str]) -> str:
        queue.submitted.append(list(command))
        message = "sbatch accepted the job, but its output could not be parsed."
        raise RuntimeError(message)

    failing = dataclasses.replace(queue.scheduler(), submit=unparsable)
    monkeypatch.setattr(dispatch, "SUBMISSION_GRACE_SECONDS", 0)
    with pytest.raises(RuntimeError, match="parsed"):
        dispatch.dispatch(TASK, "ambiguous", mode="submit", storage_root=world.storage, scheduler=failing)
    first = record.node_directory(directory, _present(record.read_submission(directory, gate, 1))["node_token"])
    assert json.loads((first / "job.json").read_text(encoding="utf-8"))["job_id"] is None
    # Neither a queue row nor a worker record exists, but the ticket started: the step ended without an outcome.
    common.serialization.atomic_write_json(node_service.receipt_path(first, 1, "started"), {"schema": node_service.STARTED_RECEIPT_SCHEMA, "seq": 1})
    with pytest.raises(RuntimeError, match="parsed"):
        dispatch.dispatch(TASK, "ambiguous", mode="resume", storage_root=world.storage, scheduler=failing)
    assert _present(record.read_outcome(directory, gate, 1))["classification"] == "infrastructure_stop"
    # The retry's job reached the queue under its token: resume records the job and waits instead of redispatching.
    token = _present(record.read_submission(directory, gate, 2))["node_token"]
    queue.active["4242"] = (1, "RUNNING", token)
    again = dispatch.dispatch(TASK, "ambiguous", mode="resume", storage_root=world.storage, scheduler=queue.scheduler())
    assert (again["node_jobs"], record.read_outcome(directory, gate, 2), len(queue.submitted)) == ([], None, 2)
    assert json.loads((record.node_directory(directory, token) / "job.recovered.json").read_text(encoding="utf-8"))["job_id"] == "4242"


def test_a_step_output_reaches_the_member_record_only_behind_its_committed_outcome(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, roster = _small_roster(world, "settle")
    gate = _ids(roster)["gate"]
    member_case = record.member_directory(directory, gate) / "case.h5"
    crashed: list[str] = []

    def crash_once(name: str) -> None:
        original = getattr(record, name)

        def crashing(*args: Any, **kwargs: Any) -> Any:
            if name not in crashed:
                crashed.append(name)
                message = f"The node died in {name}."
                raise RuntimeError(message)
            return original(*args, **kwargs)

        monkeypatch.setattr(record, name, crashing)

    # A crash before the outcome commit leaves the output in its attempt and no member case.h5.
    crash_once("record_outcome")
    _submit_by_hand(directory, gate, 1)
    with pytest.raises(RuntimeError, match="record_outcome"):
        _run(world, "settle", gate, 1)
    assert (record.attempt_directory(directory, gate, 1) / "case.h5").is_file()
    assert (record.read_outcome(directory, gate, 1), member_case.exists()) == (None, False)
    # A crash after the outcome commit is settled by resume, which links the committed output into the record and seals it.
    crash_once("settle")
    _submit_by_hand(directory, gate, 2)
    with pytest.raises(RuntimeError, match="settle"):
        _run(world, "settle", gate, 2)
    outcome = _present(record.read_outcome(directory, gate, 2))
    assert (outcome["classification"], member_case.exists()) == ("succeeded", False)
    dispatch.dispatch(TASK, "settle", mode="resume", storage_root=world.storage, scheduler=_Queue().scheduler())
    assert _present(record.read_seal(directory, gate))["case_h5_sha256"] == outcome["case_h5_sha256"] == common.serialization.file_sha256(member_case)
    assert member_case.samefile(record.attempt_directory(directory, gate, 2) / "case.h5")


def test_a_failed_base_is_replaced_by_the_next_base_in_a_new_roster_version(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, roster = _small_roster(world, "replace", full_row=True)
    ids = _ids(roster)
    with pytest.raises(ValueError, match="no failed base"):
        planning.replace_failed_bases(TASK, "replace", storage_root=world.storage)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    for attempt in (1, 2):
        _submit_by_hand(directory, ids["gate"], attempt)
        _run(world, "replace", ids["gate"], attempt)
    first_bytes = (directory / "roster.json").read_bytes()
    second, evidence = planning.replace_failed_bases(TASK, "replace", storage_root=world.storage)
    path = roster_service.write_roster(directory, second)
    assert path.name == "roster.v2.json"
    assert (directory / "roster.json").read_bytes() == first_bytes
    assert second["members"][: len(roster["members"])] == roster["members"]
    assert second["supersedes"] == {"version": 1, "roster_sha256": roster["roster_sha256"]}
    failed_case = roster["members"][0]["base"]["case_id"]
    [replacement] = second["replacements"]
    assert (replacement["failed_case_id"], replacement["failed_disposition"]) == (failed_case, "solver_failure")
    assert evidence["replacements"] == second["replacements"]
    added = second["members"][len(roster["members"]) :]
    assert {item["base"]["case_id"] for item in added} == {replacement["replacement_case_id"]} != {failed_case}
    assert added[0]["role"] == "gate"
    assert {item["row"] for item in added if item["role"] == "challenge"} == {roster["members"][2]["row"]}
    assert roster_service.load_roster(directory) == second
    with pytest.raises(ValueError, match="no failed base"):
        planning.replace_failed_bases(TASK, "replace", storage_root=world.storage)
    monkeypatch.delenv("FAKE_COMSOL_MODE")
    submitted = dispatch.dispatch(TASK, "replace", mode="submit", storage_root=world.storage, scheduler=_Queue().scheduler())
    assert [members for job in submitted["node_jobs"] for members in job["members"]] == [[added[0]["member_id"], 1]]


def test_verdicts_and_stage_b_use_the_replaced_roster_version(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, roster = _small_roster(world, "screen", full_row=True)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    for attempt in (1, 2):
        _submit_by_hand(directory, _ids(roster)["gate"], attempt)
        _run(world, "screen", _ids(roster)["gate"], attempt)
    monkeypatch.delenv("FAKE_COMSOL_MODE")
    second, _evidence = planning.replace_failed_bases(TASK, "screen", storage_root=world.storage)
    roster_service.write_roster(directory, second)
    for item in second["members"][len(roster["members"]) :]:
        _submit_by_hand(directory, item["member_id"], 1)
        _present(_run(world, "screen", item["member_id"], 1))
    field_test_fixtures.freeze(directory, TASK)
    # The version binding under test does not depend on endpoint values or base counts: the fake grid is smaller than one
    # production patch, every pair measures as identical, and one base suffices, so the row screens remove for stage B.
    endpoints = analysis_field_tests.endpoints
    monkeypatch.setattr(endpoints, "PATCH_CELLS", 4)
    measure = endpoints.endpoints
    monkeypatch.setattr(endpoints, "endpoints", lambda reference, _other: measure(reference, reference))
    monkeypatch.setattr(analysis_field_tests.verdicts, "COUPLED_MIN_BASES", 1)
    verdicts = analysis_field_tests.verdicts.build_verdicts(TASK, "screen", storage_root=world.storage)
    assert (verdicts["roster_version"], verdicts["roster_sha256"]) == (2, second["roster_sha256"])
    # The failed base's unsealed challenge leaves the verdict; only the replacement's comparison counts.
    challenged = {item["row"] for item in second["members"][len(roster["members"]) :] if item["role"] == "challenge"}
    assert [(row["row"], row["verdict"], row["missing_or_broken"]) for row in verdicts["rows"]] == [(row, "remove", 0) for row in sorted(challenged)]
    stage_b, evidence = planning.build_stage_b(TASK, "confirm", "screen", storage_root=world.storage)
    assert stage_b["derived_from"]["roster_sha256"] == second["roster_sha256"]
    assert evidence["removals"] or evidence["merges"]


def test_a_material_without_train_cases_takes_bases_of_any_design_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_only = _build_world(tmp_path, monkeypatch, {"test": 2})
    roster, evidence = planning.build_stage_a(TASK, "anykind", test_only.source_name, storage_root=test_only.storage)
    assert {item["base"]["design_kind"] for item in roster["members"]} == {"test"}
    assert evidence["excluded"] == []


def test_a_material_with_train_cases_takes_only_train_bases(world: _World) -> None:
    roster, evidence = planning.build_stage_a(TASK, "trainonly", world.source_name, storage_root=world.storage)
    assert {item["base"]["design_kind"] for item in roster["members"]} == {"train"}
    assert len(evidence["excluded"]) == 2


def test_a_design_material_without_admitted_bases_refuses_the_roster(world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
    def excluded(*_args: Any, **_kwargs: Any) -> Any:
        message = "excluded by the test"
        raise ValueError(message)

    monkeypatch.setattr(planning, "_admit", excluded)
    with pytest.raises(ValueError, match="'lentil' has 0 unused admitted bases"):
        planning.build_stage_a(TASK, "nobases", world.source_name, storage_root=world.storage)


def test_an_intervention_outside_its_support_is_sealed_infeasible(world: _World) -> None:
    full = _stage_a(world)
    challenge = next(
        item
        for item in full["members"]
        if item["role"] == "challenge" and item["level"] == "high" and type(item["intervention"][item["row"]]) is float
    )
    high = float(challenge["intervention"][challenge["row"]])
    outside = {**challenge, "member_id": f"{challenge['member_id']}-outside", "intervention": {challenge["row"]: 10.0 * abs(high) + 10.0}}
    directory, _ids_unused, _roster = _succeeded_gate(world, "infeasible", extra=[outside])
    _submit_by_hand(directory, outside["member_id"], 1)
    seal = _run(world, "infeasible", outside["member_id"], 1)
    assert seal is not None
    assert (seal["disposition"], seal["failure"]["reason_code"], seal["attempts"]) == ("infeasible", "intervention_infeasible", 1)


def test_resume_never_redispatches_a_running_submission(world: _World) -> None:
    directory, roster = _small_roster(world, "live")
    gate = _ids(roster)["gate"]
    queue = _Queue()
    submitted = dispatch.dispatch(TASK, "live", mode="submit", storage_root=world.storage, scheduler=queue.scheduler())
    job = submitted["node_jobs"][0]
    queue.active[job["job_id"]] = (1, "RUNNING", job["token"])
    for mode in ("resume", "submit"):
        again = dispatch.dispatch(TASK, "live", mode=mode, storage_root=world.storage, scheduler=queue.scheduler())
        assert again["node_jobs"] == []
    assert len(queue.submitted) == 1
    assert record.attempt_numbers(directory, gate) == (1,)
    assert record.read_outcome(directory, gate, 1) is None
    dry = dispatch.dispatch(TASK, "live", mode="submit", dry_run=True, storage_root=world.storage, scheduler=queue.scheduler())
    assert dry["node_jobs"] == []


def _current_execution(world: _World, roster: dict[str, Any], **groups: dict[str, Any]) -> None:
    """Change the execution file the base Source names, as a later edit of the maintained configuration would."""
    path = roster_service.load_base_campaign(TASK, roster["base_source"], storage_root=world.storage).execution_config_path
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    for group, values in groups.items():
        value[group].update(values)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _variant_world(world: _World, test_id: str, variants: int) -> tuple[Path, dict[str, Any]]:
    """Seal the gate and return a roster whose eligible members are its R0 reference and ``variants`` R0-only challenges."""
    reference = next(item for item in _stage_a(world)["members"] if item["role"] == "reference" and item["reference_kind"] == "R0")
    challenges = [
        item
        for item in _stage_a(world)["members"]
        if item["role"] == "challenge" and item["reference_member_id"] == reference["member_id"] and planning.reference_kinds(item["row"]) == ("R0",)
    ]
    assert len(challenges) >= variants
    directory, _, roster = _succeeded_gate(world, test_id, extra=challenges[1:variants])
    return directory, roster


def test_node_jobs_take_the_current_node_values_and_the_sources_case_values(world: _World) -> None:
    directory, roster = _small_roster(world, "resources")
    current_node = {"cores_per_node": 32, "partition": "current", "comsol_module": "Comsol/current", "comsol_executable": "comsol-current"}
    _current_execution(
        world,
        roster,
        run={"node_jobs": 3, "cores_per_case": 4, "timeout_seconds": 1000},
        node_job={"node_job_wall_time": "02:00:00", "scheduler_options": ["--account=current"]},
        site=current_node,
    )
    dry = dispatch.dispatch(TASK, "resources", mode="submit", dry_run=True, storage_root=world.storage, scheduler=_Queue().scheduler())
    assert not record.record_path(directory, "nodes").exists()
    command = dry["node_jobs"][0]["command"]
    # The Source ran one core per case: 32 concurrent cases of 1G each plus the 1G overhead.
    assert {"--cpus-per-task=32", "--mem=33G", "--time=02:00:00", "--partition=current", "--account=current", "--job-name=gen-seat-1"} <= set(command)
    assert any("Comsol/current" in argument for argument in command)
    # The solve reads its executable from the case's frozen configuration, so the node job checks that one.
    source = roster_service.load_base_campaign(TASK, roster["base_source"], storage_root=world.storage)
    assert any(f"GENERATION_COMSOL_EXECUTABLE={source.execution_values['runtime']['executable']}" in argument for argument in command)
    assert not any("comsol-current" in argument for argument in command)
    queue = _Queue()
    dispatch.dispatch(TASK, "resources", mode="submit", storage_root=world.storage, scheduler=queue.scheduler())
    token = _present(record.read_submission(directory, _ids(roster)["gate"], 1))["node_token"]
    ticket = json.loads(node_service.ticket_path(record.node_directory(directory, token), 1).read_text(encoding="utf-8"))
    # The case shape stays the Source's: 1 core, 1G and the step limits of its 5 s timeout.
    assert (ticket["cores"], ticket["memory"], ticket["time_limit"]) == (1, "1G", cluster.case_step_limits(5.0)[0])


def test_a_current_per_case_memory_other_than_the_sources_is_refused_before_anything_is_written(world: _World) -> None:
    directory, roster = _small_roster(world, "memory")
    _current_execution(world, roster, node_job={"memory_per_case": "2G"})
    with pytest.raises(ValueError, match="memory_per_case 2G"):
        dispatch.dispatch(TASK, "memory", mode="submit", storage_root=world.storage, scheduler=_Queue().scheduler())
    assert record.read_submission(directory, _ids(roster)["gate"], 1) is None


def test_each_seat_runs_one_node_job_over_a_contiguous_near_equal_share_of_the_members(world: _World) -> None:
    directory, roster = _variant_world(world, "seats", 9)
    _current_execution(world, roster, run={"node_jobs": 4})
    result = dispatch.dispatch(TASK, "seats", mode="submit", storage_root=world.storage, scheduler=_Queue().scheduler())
    jobs = result["node_jobs"]
    assert [(job["seat"], len(job["members"])) for job in jobs] == [(1, 3), (2, 3), (3, 2), (4, 2)]
    eligible = [item["member_id"] for item in roster["members"] if item["role"] != "gate"]
    assert [member for job in jobs for member, _attempt in job["members"]] == eligible
    # A job's tickets follow its chunk and end with the stop ticket.
    node_dir = record.node_directory(directory, jobs[0]["token"])
    kinds = [json.loads(node_service.ticket_path(node_dir, seq).read_text(encoding="utf-8"))["kind"] for seq in (1, 2, 3, 4)]
    assert kinds == [node_service.STEP_KIND] * 3 + [node_service.STOP_KIND]


def test_resume_redispatches_the_tickets_a_node_job_never_started(world: _World) -> None:
    directory, roster = _variant_world(world, "unstarted", 3)
    _current_execution(world, roster, run={"node_jobs": 2})
    queue = _Queue()
    first = dispatch.dispatch(TASK, "unstarted", mode="submit", storage_root=world.storage, scheduler=queue.scheduler())
    members = [member for job in first["node_jobs"] for member, _attempt in job["members"]]
    assert [len(job["members"]) for job in first["node_jobs"]] == [2, 2]
    # The first job ended after it started its first step; the worker never reached the rest of its tickets.
    node_dir = record.node_directory(directory, first["node_jobs"][0]["token"])
    started = {"schema": node_service.STARTED_RECEIPT_SCHEMA, "seq": 1}
    common.serialization.atomic_write_json(node_service.receipt_path(node_dir, 1, "started"), started)
    resumed = dispatch.dispatch(TASK, "unstarted", mode="resume", storage_root=world.storage, scheduler=queue.scheduler())
    classifications = [_present(record.read_outcome(directory, member, 1))["classification"] for member in members]
    assert classifications == ["infrastructure_stop", record.NOT_STARTED, record.NOT_STARTED, record.NOT_STARTED]
    assert [[member for member, _attempt in job["members"]] for job in resumed["node_jobs"]] == [members[:2], members[2:]]
    assert {attempt for job in resumed["node_jobs"] for _member, attempt in job["members"]} == {2}


def test_stage_a_is_deterministic_feasible_and_challenges_the_phase_at_three_levels(world: _World) -> None:
    first, second = _stage_a(world, "a1"), _stage_a(world, "a2")
    # The plan is a pure function of the Source; the roster digest also binds the test id, so only the members compare.
    assert first["members"] == second["members"]
    gate = next(item for item in first["members"] if item["role"] == "gate")
    batch, case_index = roster_service.base_batch(
        roster_service.load_base_campaign(TASK, first["base_source"], storage_root=world.storage), gate["base"]
    )
    rows = planning.tested_rows(TASK, batch)
    challenged = {item["row"] for item in first["members"] if item["role"] == "challenge"}
    assert challenged == set(rows)
    for item in first["members"]:
        if item["intervention"]:
            sampling.sample_case(batch, case_index, intervention=item["intervention"])
    # Perturbation shape rows pair with R0 and with the upper amplitude; orientation smoothing pairs with the upper jitter.
    for row, kinds in (("bed.perturbations.granularity", {"R0", "R_perturb"}), ("permeability.orientation.smooth_len_rel", {"R_jitter"})):
        assert row in rows
        paired = {roster_service.member(first, item["reference_member_id"])["reference_kind"] for item in first["members"] if item["row"] == row}
        assert paired == kinds
    phase = "pressure_bc.sin_phase"
    if phase in rows:
        assert {item["level"] for item in first["members"] if item["row"] == phase} == {"phase_minus", "phase_plus", "phase_zero"}
        assert any(item["reference_kind"] == "R_sin" for item in first["members"] if item["role"] == "reference")


def test_stage_b_reads_the_verdicts_and_confirms_removals_jointly_and_merges_at_a_constant_combination(world: _World) -> None:
    stage_a = _stage_a(world, "screen")
    directory_a = common.paths.resolve_field_test_dir(TASK, "screen", storage_root=world.storage)
    roster_service.write_roster(directory_a, stage_a)
    removed = next(item["row"] for item in stage_a["members"] if item["role"] == "challenge" and item["level"] == "low")
    verdicts = {
        "schema_kind": "field_test_verdicts",
        "schema_version": 1,
        "test_id": "screen",
        "stage": "A",
        "roster_sha256": stage_a["roster_sha256"],
        "roster_version": 1,
        "tolerances_sha256": "0" * 64,
        "rows": [{"row": removed, "verdict": "remove"}, {"row": "kappa_mean", "verdict": "keep"}],
    }
    (directory_a / "analysis").mkdir()
    (directory_a / "analysis" / "verdicts.json").write_text(json.dumps(verdicts), encoding="utf-8")
    roster, evidence = planning.build_stage_b(TASK, "confirm", "screen", storage_root=world.storage)
    assert evidence["removals"] == [removed]
    # The prespecified merge is confirmed because stage A tested both of its rows; stage A emits no merge verdict.
    assert evidence["merges"] == [_MERGE]
    assert roster == planning.build_stage_b(TASK, "confirm", "screen", storage_root=world.storage)[0]
    assert (roster["derived_from"]["roster_sha256"], roster["derived_from"]["tolerances_sha256"]) == (stage_a["roster_sha256"], "0" * 64)
    bases_a = {item["base"]["case_id"] for item in stage_a["members"]}
    assert not bases_a & {item["base"]["case_id"] for item in roster["members"]}
    challenges = [item for item in roster["members"] if item["role"] == "challenge" and item["row"] == removed]
    assert {item["level"] for item in challenges} == {"low", "high"}
    joint = [item for item in roster["members"] if item["role"] == "joint"]
    assert len(joint) == 1
    assert set(joint[0]["intervention"]) == {removed}
    merges = [item for item in roster["members"] if item["level"] == "merge"]
    assert len(merges) == 2
    reference = roster_service.member(roster, merges[0]["reference_member_id"])
    assert reference["reference_kind"] == "R0"
    strength, ratio = "permeability.anisotropy.strength", "permeability.anisotropy.max_ratio"
    combination = reference["intervention"][strength] * (reference["intervention"][ratio] - 1.0)
    for item in merges:
        assert item["row"] == _MERGE
        assert item["intervention"][strength] * (item["intervention"][ratio] - 1.0) == pytest.approx(combination, rel=1e-12)
        assert {name for name, value in item["intervention"].items() if value != reference["intervention"][name]} == {strength, ratio}
    merge_verdict = {**verdicts, "rows": [*verdicts["rows"], {"row": _MERGE, "verdict": "merge"}]}
    (directory_a / "analysis" / "verdicts.json").write_text(json.dumps(merge_verdict), encoding="utf-8")
    with pytest.raises(ValueError, match="carry a merge verdict"):
        planning.build_stage_b(TASK, "confirm2", "screen", storage_root=world.storage)
