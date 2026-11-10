"""Protect metadata-only campaign continuation and its explicit compute boundary.

Responsibilities:
  - Check prepared, absent, and invalid replacement input evidence
  - Preserve sole-feeder cancellation, scheduler reconciliation, and completion
  - Defer replacement generation and legacy heavy artifact admission; plan no replacement for a stopped batch
  - Check that the stop set and blocking read one candidate snapshot and a decision recorded mid-planning ends planning
    for its batch only
  - Check that a recorded integer cap stays and a recorded null cap follows the current plan rates, only under a plan rate
  - Check the one-line resume summary with its continuation reason and only its own reclassifications

Design principles:
  - Metadata-only continuation uses the same feeder and never launches input generation
  - Missing input is distinct from partial, corrupt, or conflicting evidence

This module does NOT:
  - Execute Slurm, scientific simulation, or production source publication
"""

from __future__ import annotations

import contextlib
import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from src import common
from src.generation import generation_campaign as service
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cases import generation_cases_input as inputs
from src.generation.cases import generation_cases_selection as selection
from src.generation.cli import cli_generation
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_catalog as catalog

if TYPE_CHECKING:
    from collections.abc import Iterator

_RUN = "source~steady_flow~lentil-1__ref-range__sobol-seed-9910"


def _view(state: str) -> dict[str, str]:
    """Return the task view of the fixture's only candidate in ``state``."""
    return {"batch_id": "primary", "case_id": "case_0001", "state": state}


@pytest.fixture
def feeder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Isolate persisted evidence owners around the actual feeder branching; a campaign without node jobs has no scheduler work."""
    scheduler_evidence = service._scheduler_evidence
    original = SimpleNamespace(batch_id="primary", batch_name="primary", material_family="lentil", case_indices=(1,), scientific_values={})
    replacement = SimpleNamespace(
        batch_id="replacement", batch_name="replacement", material_family="lentil", case_indices=(2,), scientific_values={"replacement_design": {}}
    )
    campaign = SimpleNamespace(
        batches=(original,),
        selected_targets={"lentil": 1},
        profile=SimpleNamespace(id="steady_flow"),
        campaign_name="fixture",
        source_name="fixture__" + "a" * 16,
        producer_identity={"source_implementation_sha256": source_service.GENERATION_SOURCE_PROJECTION.sha256()},
    )
    manifest = {
        "campaign_run_id": _RUN,
        "state": "active",
        "git_commit": "a" * 40,
        "slurm_job_ids": [],
        "node_jobs": [],
        "node_job_intent": None,
        "submissions": [],
        "submission_intent": None,
    }
    state = SimpleNamespace(
        campaign=campaign,
        manifest=manifest,
        replacement=replacement,
        records=[
            {
                "reservation": {"batch_id": "primary", "material_family": "lentil", "case_id": "case_0001", "origin": "primary"},
                "decision": None,
                "attempts": [],
            }
        ],
        views=[_view("active")],
        pending=0,
        running=1,
        admissions=[],
        generations=[],
        materializations=[],
        fills=[],
        seals=[],
        states=[],
        policy=[{"batch_id": "primary", "stopped": False, "amendments": []}],
    )
    run_directory = common.paths.resolve_generation_campaign_directory("steady_flow", _RUN.split("~")[2], storage_root=tmp_path)
    monkeypatch.setattr(service.campaign_evidence, "campaign_run_directory", lambda *_a, **_k: run_directory)
    monkeypatch.setattr(service.campaign_evidence, "load_campaign_run", lambda *_a, **_k: manifest)
    monkeypatch.setattr(service.campaign_evidence, "campaign_from_manifest", lambda _m: campaign)
    monkeypatch.setattr(lifecycle, "candidate_records", lambda *_a, **_k: state.records)
    monkeypatch.setattr(lifecycle, "replacement_policy", lambda *_a, **_k: state.policy)
    monkeypatch.setattr(lifecycle, "campaign_from_frozen_run", lambda _m: campaign)
    monkeypatch.setattr(service, "_execution_binding", lambda _m: {"submission_config": {"max_admission_cases": 4}})
    monkeypatch.setattr(service.cluster_service, "campaign_tasks", lambda _c: ())
    monkeypatch.setattr(service, "_submission_index", lambda _m: {})
    monkeypatch.setattr(service, "_scheduler_evidence", lambda _ids: scheduler_evidence([]))
    monkeypatch.setattr(service, "_reconciled", lambda *_a, **_k: (state.views, state.pending, state.running))
    monkeypatch.setattr(service, "_unexpected_operational_failure_views", lambda *_a: ())

    def fill(_m: Any, _c: Any, _v: Any, **kwargs: Any) -> dict[str, Any]:
        state.fills.append(kwargs)
        return manifest

    def set_state(_m: Any, value: str, **_kwargs: Any) -> dict[str, Any]:
        state.states.append(value)
        return {**manifest, "state": value}

    monkeypatch.setattr(service, "_fill_submission_capacity", fill)
    monkeypatch.setattr(service, "_set_campaign_state", set_state)
    monkeypatch.setattr(lifecycle, "seal_selected_pool", lambda *_a, **_k: state.seals.append(True))
    monkeypatch.setattr(catalog, "campaign_cases_csv", lambda *_a, **_k: "material,case_id\n")  # no campaign tree behind the stubs
    monkeypatch.setattr(inputs, "generate_input_cases", lambda *a, **_k: state.generations.append(a))
    monkeypatch.setattr(inputs, "admit_configured_input_references", lambda *a, **k: state.admissions.append((a, k)) or {2: object()})
    return state


def test_prepared_replacement_inputs_continue_without_full_validation(feeder: SimpleNamespace) -> None:
    """A prepared replacement source is evidence-admitted before ordinary feeding."""
    feeder.campaign.batches += (feeder.replacement,)
    result = service.resume_campaign(_RUN, storage_root=Path("unused"), metadata_only=True)
    assert result is feeder.manifest
    assert len(feeder.admissions) == 1
    assert feeder.admissions[0][1]["validation_depth"] == "evidence"
    assert feeder.admissions[0][1]["git_commit"] == feeder.manifest["git_commit"]
    assert not feeder.generations
    assert len(feeder.fills) == 1
    assert feeder.fills[0]["running_jobs"] == 1


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing membership"), FileExistsError("corrupt evidence"), RuntimeError("conflicting identity")]
)
def test_missing_replacement_inputs_request_compute_but_invalid_evidence_fails(
    feeder: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """Only missing prepared input can be escalated into spatial generation."""
    feeder.campaign.batches += (feeder.replacement,)

    def admit(*_a: Any, **_k: Any) -> None:
        raise error

    monkeypatch.setattr(inputs, "admit_configured_input_references", admit)
    if isinstance(error, FileNotFoundError):
        result = service.resume_campaign(_RUN, metadata_only=True)
        assert result["continuation_requires_compute"] is True
        assert result["continuation_reason"] == "replacement_inputs_missing"
        assert "continuation_requires_compute" not in feeder.manifest
    else:
        with pytest.raises(type(error), match=str(error)):
            service.resume_campaign(_RUN, metadata_only=True)
    assert not feeder.generations
    assert not feeder.fills


@pytest.mark.parametrize("stopped", [False, True])
def test_replacement_deficit_defers_before_candidate_materialization_unless_its_batch_is_stopped(
    feeder: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    stopped: bool,
) -> None:
    """A deficit requests compute without reserving a replacement; a stopped batch plans none and, all decided, ends blocked."""
    feeder.records[0]["decision"] = {"outcome": "failed"}
    feeder.policy = [{"batch_id": "primary", "stopped": stopped, "amendments": []}]
    monkeypatch.setattr(selection, "replacement_deficit", lambda *_a, **_k: 1)
    monkeypatch.setattr(lifecycle, "materialize_replacement_batch", lambda *a, **_k: feeder.materializations.append(a))
    result = service.resume_campaign(_RUN, metadata_only=True)
    assert result.get("continuation_reason") == (None if stopped else "replacement_input_generation")
    assert ("blocked" in feeder.states) is stopped
    assert not feeder.materializations
    assert not feeder.generations


def test_a_success_recorded_during_the_feed_lifts_its_batch_stop_instead_of_blocking(
    feeder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop set and the blocked decision read one snapshot: k/n 0/2 stops at s = 0.8, a success committed mid-feed (1/3) does not."""
    feeder.campaign.batches = (
        SimpleNamespace(batch_id="primary", batch_name="primary", material_family="lentil", case_indices=(1, 2, 3, 4), scientific_values={}),
    )
    feeder.campaign.selected_targets = {"lentil": 4}
    feeder.records = [
        {
            "reservation": {"batch_id": "primary", "material_family": "lentil", "case_id": f"case_000{index}", "origin": "primary"},
            "decision": None,
            "attempts": [],
        }
        for index in (1, 2, 3, 4)
    ]
    for row in feeder.records[:2]:
        row["decision"] = {"outcome": "failed"}

    def policy(*_a: Any, records: Any = None, **_k: Any) -> list[dict[str, Any]]:
        counts = selection.decided_counts(feeder.records if records is None else records)
        stopped = selection.success_rate_stopped(*counts, minimum_success_rate=0.8)
        return [{"batch_id": "primary", "stopped": stopped, "amendments": [], "minimum_success_rate": 0.8}]

    def success_commits(_manifest: Any) -> SimpleNamespace:
        feeder.records[2]["decision"] = {"outcome": "successful"}  # case_0003 succeeds after the first policy read
        return feeder.campaign

    monkeypatch.setattr(lifecycle, "replacement_policy", policy)
    monkeypatch.setattr(lifecycle, "campaign_from_frozen_run", success_commits)
    service.resume_campaign(_RUN)
    assert "blocked" not in feeder.states
    assert feeder.fills[0]["stopped"] == frozenset()


def test_a_success_recorded_while_a_replacement_is_planned_ends_planning_for_that_batch_only(
    feeder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A success that closes a batch's deficit after the check ends its planning without an error; the other batch gets its replacement."""
    second = SimpleNamespace(batch_id="second", batch_name="second", material_family="chickpea", case_indices=(1,), scientific_values={})
    feeder.replacement.scientific_values = {"replacement_design": {"parent_batch_id": "second", "candidate_id": "replacement__second"}}
    feeder.campaign.batches += (second,)
    feeder.campaign.selected_targets = {"lentil": 1, "chickpea": 1}
    materials = {"primary": "lentil", "second": "chickpea"}
    feeder.records = [
        {
            "reservation": {"batch_id": batch_id, "material_family": material, "case_id": "case_0001", "origin": "primary"},
            "decision": {"outcome": "failed"},
            "attempts": [],
        }
        for batch_id, material in materials.items()
    ]
    feeder.policy = [{"batch_id": batch_id, "stopped": False, "amendments": [], "minimum_success_rate": 0.5} for batch_id in ("primary", "second")]

    def materialize(_campaign: Any, batch_id: str, **_k: Any) -> SimpleNamespace:
        feeder.materializations.append(batch_id)
        if batch_id == "primary":  # a candidate of this batch is recorded successful after the deficit check
            reservation = {
                "batch_id": "primary",
                "material_family": "lentil",
                "case_id": "case_0002",
                "origin": "replacement",
                "design_key": "replacement__primary",
            }
            feeder.records.append({"reservation": reservation, "decision": {"outcome": "successful"}, "attempts": []})
            raise lifecycle.NoReplacementError(batch_id)
        reservation = {
            "batch_id": "second",
            "material_family": "chickpea",
            "case_id": "case_0002",
            "origin": "replacement",
            "design_key": "replacement__second",
        }
        feeder.records.append({"reservation": reservation, "decision": None, "attempts": []})
        feeder.campaign.batches += (feeder.replacement,)
        return feeder.replacement

    monkeypatch.setattr(lifecycle, "materialize_replacement_batch", materialize)
    service.resume_campaign(_RUN)
    assert feeder.materializations == ["primary", "second"]
    assert [call[0] for call in feeder.generations] == [feeder.replacement]
    assert len(feeder.fills) == 1


def test_a_derived_admission_cap_follows_current_rates_and_requires_a_plan_rate_for_every_original_batch() -> None:
    """A recorded integer is the cap; a recorded null is the plan default over the current rates, refused without a plan rate."""
    campaign: Any = SimpleNamespace(batches=(SimpleNamespace(batch_id="primary", case_indices=(1, 2)),))
    plan = {"registered_by": "plan", "minimum_success_rate": 0.8}
    policy = [{"batch_id": "primary", "minimum_success_rate": 0.6, "amendments": [plan, {"registered_by": "explicit", "minimum_success_rate": 0.6}]}]
    assert service._admission_cap({"max_admission_cases": 4}, campaign, policy) == 4
    assert service._admission_cap({"max_admission_cases": None}, campaign, policy) == 2 + selection.maximum_tolerated_failures(2, 0.6)
    with pytest.raises(RuntimeError, match="plan-registered"):
        service._admission_cap({"max_admission_cases": None}, campaign, [{**policy[0], "amendments": []}])


def test_completed_canonical_pool_seals_without_compute(feeder: SimpleNamespace) -> None:
    """Settled successful cases only require small immutable pool JSON publication and the derived cases view."""
    feeder.records[0]["decision"] = {"outcome": "successful"}
    result = service.resume_campaign(_RUN, metadata_only=True)
    assert result["state"] == "complete"
    assert feeder.seals == [True]
    campaign = common.paths.resolve_generation_campaign_directory("steady_flow", feeder.campaign.source_name)
    assert (campaign / "cases.csv").read_text(encoding="utf-8") == "material,case_id\n"
    assert not feeder.generations
    assert not feeder.fills


@pytest.mark.parametrize("state", ["cancel_requested", "force_cancel_requested"])
def test_cancelled_campaign_does_not_admit_inputs_or_feed(feeder: SimpleNamespace, state: str) -> None:
    """Cancellation remains ahead of prepared input and retry processing."""
    feeder.manifest["state"] = state
    feeder.campaign.batches += (feeder.replacement,)
    assert service.resume_campaign(_RUN, metadata_only=True) is feeder.manifest
    assert not feeder.admissions
    assert not feeder.fills


def test_changed_source_projection_refuses_continuation_before_replacement_inputs_or_reconciliation(
    feeder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen campaign recorded under another Source implementation is not continued with the running code."""
    feeder.campaign.batches += (feeder.replacement,)
    recorded = feeder.campaign.producer_identity["source_implementation_sha256"]
    monkeypatch.setattr(type(source_service.GENERATION_SOURCE_PROJECTION), "sha256", lambda _self, _root=None: "0" * 64)
    with pytest.raises(ValueError, match=r"generation_source implementation changed \(recorded " + recorded[:12]):
        service.resume_campaign(_RUN, metadata_only=True)
    assert [feeder.admissions, feeder.generations, feeder.fills, feeder.states, feeder.seals] == [[], [], [], [], []]


def test_scheduler_unknown_preserves_existing_transition(feeder: SimpleNamespace) -> None:
    """Missing scheduler evidence never becomes a compute or speculative retry request."""
    feeder.views = [_view("scheduler_unknown")]
    result = service.resume_campaign(_RUN, metadata_only=True)
    assert result["state"] == "scheduler_unknown"
    assert not feeder.fills
    assert not feeder.generations


@pytest.mark.parametrize("existing", ["absent", "partial", "corrupt"])
def test_configured_input_owner_distinguishes_absence_from_broken_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    """Wholly absent namespaces require compute; partial or corrupt sources fail closed."""
    metadata, raw = tmp_path / "meta", tmp_path / "raw"
    if existing != "absent":
        raw.mkdir()
    if existing == "corrupt":
        metadata.mkdir()
        (metadata / "input_generation_manifest.json").write_text("{")
    monkeypatch.setattr(inputs, "_configured_input_locations", lambda *_a, **_k: ({"input_generation_id": "input-" + "a" * 24}, {}, metadata, raw))
    expected = FileNotFoundError if existing == "absent" else FileExistsError
    config: Any = SimpleNamespace(case_indices=(1,))
    with pytest.raises(expected):
        inputs.admit_configured_input_references(config, validation_depth="evidence")


def test_cli_metadata_only_continuation_is_explicit() -> None:
    """Direct CLI resume remains full while the controller opts into the light contract."""
    parser = cli_generation._build_parser()
    assert parser.parse_args(["resume-campaign", _RUN]).metadata_only is False
    assert parser.parse_args(["resume-campaign", _RUN, "--metadata-only"]).metadata_only is True


def test_cli_resume_prints_one_summary_line_with_the_fields_the_controller_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resume prints the state, the compute request with its reason, the submission counts and the reclassifications of its feed."""
    case = {"batch_id": "primary", "batch_name": "primary", "case_id": "case_0002", "case_index": 2}
    manifest = {
        "campaign_run_id": _RUN,
        "state": "active",
        "continuation_requires_compute": True,
        "continuation_reason": "replacement_input_generation",
        "node_jobs": [{"job_id": "1"}] * 50,
        "submissions": [
            {"status": "not_started", "case": {**case, "case_id": "case_0001"}, "error": "reclassified by an earlier resume"},
            {"status": "not_started", "case": case, "error": "the node job ended before the ticket was published"},
        ],
    }

    def resume(*_a: Any, reclassified: list[Any], **_k: Any) -> dict[str, Any]:
        reclassified.append(manifest["submissions"][1])
        return manifest

    monkeypatch.setattr(cli_generation.campaign_runtime, "resume_campaign", resume)
    assert cli_generation.main(["resume-campaign", _RUN, "--storage-root", str(tmp_path)]) == 0
    (line,) = capsys.readouterr().out.splitlines()
    summary = json.loads(line)
    assert (summary["campaign_run_id"], summary["state"], summary["continuation_requires_compute"]) == (_RUN, "active", True)
    assert summary["continuation_reason"] == "replacement_input_generation"
    assert summary["submissions"] == {"not_started": 2}
    assert summary["reclassified"] == ["case_0002: the node job ended before the ticket was published"]
    assert datetime.fromisoformat(summary["at"]).utcoffset() is not None
    assert summary["record"].endswith("_record/execution/run.json")
    assert "node_jobs" not in summary


def test_cli_resume_claims_only_the_reclassifications_of_its_own_feed(
    feeder: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resume reports the dispatches its own feed reclassified, by either way, but not one another feed reclassified before."""
    case = {"batch_id": "primary", "batch_name": "primary", "case_index": 1}
    feeder.manifest["submissions"] = [
        {"status": "dispatched", "submission_index": index, "case": {**case, "case_id": f"case_000{index}"}, "error": None} for index in (1, 2, 3)
    ]
    persisted = copy.deepcopy(feeder.manifest)
    lock = common.locking.exclusive_file_lock

    @contextlib.contextmanager
    def after_another_feed(*args: Any, **kwargs: Any) -> Iterator[Path]:
        persisted["submissions"][0].update(status="not_started", error="reclassified by another feed")
        with lock(*args, **kwargs) as path:
            yield path

    def recover_interrupted_dispatch(manifest: Any, *_a: Any, **_k: Any) -> None:
        manifest["submissions"][2].update(status="not_started", error="the node job ended before the ticket was published")

    def evidence(manifest: Any, scheduler: Any, **_k: Any) -> dict[str, Any]:
        unstarted = [record for record in manifest["submissions"] if record["status"] == "dispatched" and record["case"]["case_id"] == "case_0002"]
        return {**scheduler, "active": {}, "accounted": {}, "not_started": [(record, "the node job ended") for record in unstarted]}

    monkeypatch.setattr(service.campaign_evidence, "load_campaign_run", lambda *_a, **_k: copy.deepcopy(persisted))
    monkeypatch.setattr(common.locking, "exclusive_file_lock", after_another_feed)
    monkeypatch.setattr(service, "_maintain_node_jobs", recover_interrupted_dispatch)
    monkeypatch.setattr(service, "_dispatch_evidence", evidence)
    monkeypatch.setattr(service, "_persist_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(service, "_fill_submission_capacity", lambda manifest, *_a, **_k: manifest)
    assert cli_generation.main(["resume-campaign", _RUN, "--storage-root", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["submissions"] == {"not_started": 3}
    assert summary["reclassified"] == ["case_0002: the node job ended", "case_0003: the node job ended before the ticket was published"]


def test_fresh_metadata_submission_requires_explicit_prepared_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public metadata-only contract never enters fresh spatial input preparation."""
    commit = "a" * 40
    config: Any = SimpleNamespace(total_case_count=1, selected_targets={"lentil": 1})
    monkeypatch.setattr(service, "_repository_commit", lambda: commit)
    monkeypatch.setattr(service, "campaign_run_id", lambda *_a, **_k: _RUN)
    run_directory = common.paths.resolve_generation_campaign_directory("steady_flow", _RUN.split("~")[2], storage_root=tmp_path)
    monkeypatch.setattr(service.campaign_evidence, "campaign_run_directory", lambda *_a, **_k: run_directory)
    monkeypatch.setattr(lifecycle, "create_campaign", lambda *_a, **_k: None)

    def unexpected_preparation(*_a: Any, **_k: Any) -> None:
        pytest.fail("Metadata-only submission entered scientific input preparation")

    monkeypatch.setattr(inputs, "prepare_campaign_inputs", unexpected_preparation)
    with pytest.raises(ValueError, match="prepare-campaign-inputs followed by --inputs-prepared"):
        service.submit_campaign(config, git_commit=commit, metadata_only=True)
    assert not common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(run_directory)).exists()
