"""Protect metadata-only campaign continuation and its explicit compute boundary.

Responsibilities:
  - Check prepared, absent, and invalid supplemental input evidence
  - Preserve sole-feeder cancellation, scheduler reconciliation, and completion
  - Defer replacement generation and legacy heavy artifact admission

Design principles:
  - Metadata-only continuation uses the same feeder and never launches input generation
  - Missing input is distinct from partial, corrupt, or conflicting evidence

This module does NOT:
  - Execute Slurm, scientific simulation, or production source publication
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import common
from src.generation import generation_campaign as service
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cases import generation_cases_input as inputs
from src.generation.cases import generation_cases_selection as selection
from src.generation.cli import cli_generation
from src.generation.publication import generation_publication_catalog as catalog

_RUN = "source~steady_flow~lentil-d1__reference-range__lhs-s9910"


def _view(state: str) -> dict[str, str]:
    """Return the task view of the fixture's only candidate in ``state``."""
    return {"batch_id": "primary", "case_id": "case_0001", "state": state}


@pytest.fixture
def feeder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Isolate persisted evidence owners around the actual feeder branching; a campaign without node jobs has no scheduler work."""
    scheduler_evidence = service._scheduler_evidence
    original = SimpleNamespace(batch_id="primary", batch_name="primary", case_indices=(1,), scientific_values={})
    replacement = SimpleNamespace(batch_id="replacement", batch_name="replacement", case_indices=(2,), scientific_values={"supplemental_design": {}})
    campaign = SimpleNamespace(
        schema_version=service.config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION,
        batches=(original,),
        profile=SimpleNamespace(id="steady_flow"),
        campaign_name="fixture",
        source_name="fixture__" + "a" * 16,
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
        records=[{"reservation": {"batch_id": "primary", "case_id": "case_0001", "origin": "primary"}, "decision": None, "attempts": []}],
        views=[_view("active")],
        pending=0,
        running=1,
        admissions=[],
        generations=[],
        materializations=[],
        fills=[],
        seals=[],
        states=[],
    )
    monkeypatch.setattr(service.campaign_evidence, "campaign_run_directory", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(service.campaign_evidence, "load_campaign_run", lambda *_a, **_k: manifest)
    monkeypatch.setattr(service.campaign_evidence, "campaign_from_manifest", lambda _m: campaign)
    monkeypatch.setattr(lifecycle, "candidate_records", lambda *_a, **_k: state.records)
    monkeypatch.setattr(lifecycle, "campaign_from_frozen_run", lambda _m: campaign)
    monkeypatch.setattr(service, "_execution_binding", lambda _m: {})
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


def test_prepared_supplemental_inputs_continue_without_full_validation(feeder: SimpleNamespace) -> None:
    """A prepared supplemental source is evidence-admitted before ordinary feeding."""
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
def test_supplemental_missing_requests_compute_but_invalid_evidence_fails(
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
        assert result["continuation_reason"] == "supplemental_inputs_missing"
        assert "continuation_requires_compute" not in feeder.manifest
    else:
        with pytest.raises(type(error), match=str(error)):
            service.resume_campaign(_RUN, metadata_only=True)
    assert not feeder.generations
    assert not feeder.fills


def test_replacement_deficit_defers_before_candidate_materialization(
    feeder: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling deficit requests compute without reserving or generating a replacement."""
    feeder.records[0]["decision"] = {"outcome": "failed"}
    monkeypatch.setattr(selection, "replacement_deficit", lambda *_a, **_k: 1)
    monkeypatch.setattr(lifecycle, "replacement_pool_high_water", lambda *_a, **_k: 2)
    monkeypatch.setattr(lifecycle, "materialize_replacement_batch", lambda *a, **_k: feeder.materializations.append(a))
    result = service.resume_campaign(_RUN, metadata_only=True)
    assert result["continuation_reason"] == "replacement_input_generation"
    assert not feeder.materializations
    assert not feeder.generations


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


def test_legacy_completion_requires_scheduled_artifact_admission(feeder: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy batch HDF5 admission is deferred before the heavyweight owner is called."""
    feeder.campaign.schema_version = service.config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION + 1
    feeder.views = [{"state": "successful"}]
    finalized: list[bool] = []
    monkeypatch.setattr(service, "_finalize_completed_batches", lambda *_a, **_k: finalized.append(True))
    result = service.feed_campaign(_RUN, metadata_only=True)
    assert result["continuation_reason"] == "batch_artifact_admission"
    assert not finalized


@pytest.mark.parametrize("state", ["cancel_requested", "force_cancel_requested"])
def test_cancelled_campaign_does_not_admit_inputs_or_feed(feeder: SimpleNamespace, state: str) -> None:
    """Cancellation remains ahead of prepared input and retry processing."""
    feeder.manifest["state"] = state
    feeder.campaign.batches += (feeder.replacement,)
    assert service.resume_campaign(_RUN, metadata_only=True) is feeder.manifest
    assert not feeder.admissions
    assert not feeder.fills


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


@pytest.mark.parametrize("canonical", [True, False])
def test_fresh_metadata_submission_requires_explicit_prepared_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, canonical: bool) -> None:
    """The public metadata-only contract never enters fresh spatial input preparation."""
    commit = "a" * 40
    config: Any = SimpleNamespace(
        schema_version=service.config_service.CANONICAL_CAMPAIGN_SCHEMA_VERSION + int(not canonical),
        total_case_count=1,
        selected_targets={"lentil": 1},
    )
    monkeypatch.setattr(service, "_repository_commit", lambda: commit)
    monkeypatch.setattr(service, "campaign_run_id", lambda *_a, **_k: _RUN)
    monkeypatch.setattr(service.campaign_evidence, "campaign_run_directory", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(lifecycle, "create_campaign", lambda *_a, **_k: None)

    def unexpected_preparation(*_a: Any, **_k: Any) -> None:
        pytest.fail("Metadata-only submission entered scientific input preparation")

    monkeypatch.setattr(inputs, "prepare_campaign_inputs", unexpected_preparation)
    with pytest.raises(ValueError, match="prepare-campaign-inputs followed by --inputs-prepared"):
        service.submit_campaign(config, git_commit=commit, metadata_only=True)
    assert not common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(tmp_path)).exists()
