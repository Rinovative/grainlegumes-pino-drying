# ruff: noqa: S101
"""No-COMSOL conversion and publication replay contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import h5py
import pytest

from src import common, generation
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_comsol as comsol_service

_REPLAY_FAILURE_ATTEMPT_INDEX = 2
_RETAINED_CONVERSION_DEFECT = "synthetic retained conversion defect"
_REPLAY_EVIDENCE_DEFECT = "synthetic replay evidence defect"

if TYPE_CHECKING:
    from pathlib import Path


def _production_case(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> tuple[Any, Path]:
    """Return one compact Production config with admitted canonical inputs."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        campaign_purpose="family_generalization",
        natural_count=3,
    )
    batch_name = generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=batch_name,
    )
    storage = tmp_path / "replay storage"
    generation.cases.input_generation.generate_input_cases(
        config,
        len(config.case_indices),
        storage_root=storage,
    )
    return config, storage


def _attempt(config: Any, storage: Path) -> Any:
    """Return one required failed attempt from the local campaign identity."""
    attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        config.batch_id,
        storage_root=storage,
    )
    assert attempt is not None
    return attempt


def _forbid_comsol(*_args: Any, **_kwargs: Any) -> None:
    """Fail if replay constructs any COMSOL command."""
    pytest.fail("Postprocessing replay constructed a COMSOL command.")


@pytest.mark.integration
def test_no_ramp_early_stop_conversion_replays_without_comsol(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the observed no-ramp early-stop shape after a converter failure."""
    config, storage = _production_case(
        generation_config_factory,
        fake_comsol,
        tmp_path,
    )
    original_convert = storage_service.convert_exports_to_hdf5

    def fail_conversion(*_args: Any, **_kwargs: Any) -> None:
        message = "synthetic conversion defect"
        raise RuntimeError(message)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            storage_service,
            "convert_exports_to_hdf5",
            fail_conversion,
        )
        with pytest.raises(
            generation.runtime.CaseExecutionError,
            match="synthetic conversion defect",
        ) as caught:
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "solver work",
            )
    assert caught.value.failure_stage == "conversion"
    attempt = _attempt(config, storage)
    assert attempt.payload["case_state"] == "conversion_failed"
    assert attempt.payload["solver_state"] == "succeeded"
    assert attempt.replay_available
    receipt_before = attempt.receipt_path.read_bytes()

    monkeypatch.setenv("GENERATION_GIT_COMMIT", "b" * 40)
    monkeypatch.setattr(comsol_service, "build_comsol_command", _forbid_comsol)
    monkeypatch.setattr(storage_service, "convert_exports_to_hdf5", original_convert)
    outcome = generation.runtime.replay_case_postprocessing(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / "replay work",
    )

    assert outcome.status == "replayed"
    generation.runtime.validate_completed_case(config, 1, storage_root=storage)
    status = json.loads((outcome.processed_directory / "status.json").read_text(encoding="utf-8"))
    assert status["case_state"] == "successful"
    assert status["schedule_valid"] is True
    assert status["has_exact_stop_state"] is True
    assert status["exact_stop_state_time"] == pytest.approx(1.5)
    assert status["stages"] == {
        "solver": "succeeded",
        "exports": "succeeded",
        "conversion": "succeeded",
        "diagnostics": status["stages"]["diagnostics"],
        "publication": "succeeded",
    }
    processing = json.loads((outcome.processed_directory / "processing_provenance.json").read_text(encoding="utf-8"))
    assert processing["mode"] == "replay_conversion"
    assert processing["solver_git_commit"] == "a" * 40
    assert processing["processing_git_commit"] == "b" * 40
    assert attempt.receipt_path.read_bytes() == receipt_before
    replay = json.loads((attempt.directory / "replay.json").read_text(encoding="utf-8"))
    assert replay["cleanup_state"] == "complete"
    assert all(not (attempt.directory / relative).exists() for relative in attempt.payload["temporary_recovery_payload"])


@pytest.mark.integration
def test_post_horizon_conversion_failure_replays_without_comsol(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay complete exports while excluding one target-crossing post-horizon state."""
    config, storage = _production_case(generation_config_factory, fake_comsol, tmp_path)
    expected_stop = 168.0
    expected_regular_states = int(expected_stop) + 1
    ignored_post_horizon = 172.2885558527809
    original_convert = storage_service.convert_exports_to_hdf5

    def reject_old_converter(*_args: Any, **_kwargs: Any) -> None:
        message = "legacy converter rejected post-horizon state"
        raise ValueError(message)

    monkeypatch.setenv("FAKE_COMSOL_TRANSIENT_TIME_MODE", "post_horizon")
    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", reject_old_converter)
        with pytest.raises(generation.runtime.CaseExecutionError, match="legacy converter"):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "post-horizon solver work",
            )
    attempt = _attempt(config, storage)
    original_receipt = attempt.receipt_path.read_bytes()
    assert attempt.payload["case_state"] == "conversion_failed"
    assert attempt.replay_available

    monkeypatch.setenv("GENERATION_GIT_COMMIT", "b" * 40)
    monkeypatch.setattr(comsol_service, "build_comsol_command", _forbid_comsol)
    monkeypatch.setattr(storage_service, "convert_exports_to_hdf5", original_convert)
    outcome = generation.runtime.replay_case_postprocessing(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / "post-horizon replay work",
    )

    assert outcome.status == "replayed"
    assert attempt.receipt_path.read_bytes() == original_receipt
    generation.runtime.validate_completed_case(config, 1, storage_root=storage)
    status = json.loads((outcome.processed_directory / "status.json").read_text(encoding="utf-8"))
    assert status["hit_t_max"] is True
    assert status["target_reached"] is False
    assert status["t_stop_exact"] == expected_stop
    assert status["has_exact_stop_state"] is False
    assert status["post_horizon_export_state_ignored"] is True
    assert status["ignored_post_horizon_state_time"] == ignored_post_horizon
    assert status["raw_export_state_count"] == expected_regular_states + 1
    assert status["canonical_state_count"] == expected_regular_states
    with h5py.File(outcome.processed_directory / "case.h5", "r") as handle:
        time_dataset = handle.get("time")
        assert isinstance(time_dataset, h5py.Dataset)
        canonical_time = time_dataset[...]
        assert canonical_time.size == expected_regular_states
        assert canonical_time[0] == 0.0
        assert canonical_time[-1] == expected_stop
        assert ignored_post_horizon not in canonical_time


@pytest.mark.integration
def test_compatible_historical_conversion_attempt_replays_in_place_across_commits(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse exact old replay evidence while preserving solver and processing commits."""
    old_run_id = "old-campaign__0123456789abcdef"
    new_run_id = "new-campaign__fedcba9876543210"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", old_run_id)
    config, storage = _production_case(generation_config_factory, fake_comsol, tmp_path)
    original_convert = storage_service.convert_exports_to_hdf5

    def fail_conversion(*_args: Any, **_kwargs: Any) -> None:
        message = "historical conversion defect"
        raise RuntimeError(message)

    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(generation.runtime.CaseExecutionError, match="historical conversion defect"):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "historical solver work",
            )
    old_attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        old_run_id,
        storage_root=storage,
    )
    assert old_attempt is not None
    old_receipt = old_attempt.receipt_path.read_bytes()
    task = cluster_service.CampaignTask(
        batch_name=config.batch_name,
        batch_id=config.batch_id,
        case_index=1,
        case_id=config.case_id(1),
    )
    historical = generation.campaign._admitted_case_attempt(  # noqa: SLF001 -- tests cross-run admission
        {"campaign_run_id": new_run_id, "git_commit": "b" * 40},
        config,
        task,
        storage_root=storage,
    )
    assert historical is not None
    assert historical.receipt_path == old_attempt.receipt_path

    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", new_run_id)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "b" * 40)
    monkeypatch.setattr(comsol_service, "build_comsol_command", _forbid_comsol)
    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(RuntimeError, match="historical conversion defect"):
            generation.runtime.replay_case_postprocessing(
                config,
                1,
                source_campaign_run_id=old_run_id,
                storage_root=storage,
                work_root=tmp_path / "failed cross-commit replay work",
            )
    retry_attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        old_run_id,
        storage_root=storage,
    )
    assert retry_attempt is not None
    assert retry_attempt.payload["attempt_index"] == _REPLAY_FAILURE_ATTEMPT_INDEX
    assert generation.publication.attempt.replay_failure_evidence(retry_attempt) is not None
    assert (
        generation.publication.attempt.latest_case_attempt(
            config,
            1,
            new_run_id,
            storage_root=storage,
        )
        is None
    )

    original_identities = runtime_service._current_replay_identities  # noqa: SLF001 -- models a converter change

    def changed_converter_identity(*args: Any, **kwargs: Any) -> dict[str, str]:
        identities = original_identities(*args, **kwargs)
        identities["converter_dependency_sha256"] = "f" * 64
        return identities

    monkeypatch.setattr(runtime_service, "_current_replay_identities", changed_converter_identity)
    monkeypatch.setattr(storage_service, "convert_exports_to_hdf5", original_convert)
    outcome = generation.runtime.replay_case_postprocessing(
        config,
        1,
        source_campaign_run_id=old_run_id,
        storage_root=storage,
        work_root=tmp_path / "changed cross-commit replay work",
    )

    assert outcome.status == "replayed"
    assert old_attempt.receipt_path.read_bytes() == old_receipt
    processing = json.loads((outcome.processed_directory / "processing_provenance.json").read_text(encoding="utf-8"))
    assert processing["solver_git_commit"] == "a" * 40
    assert processing["processing_git_commit"] == "b" * 40


@pytest.mark.integration
def test_new_campaign_submits_only_fresh_cases_beside_historical_replay_evidence(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an old conversion failure out of new-run Slurm submissions."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        campaign_purpose="family_generalization",
        scheduler_kind="slurm",
        natural_count=3,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    storage = tmp_path / "cross-commit campaign storage"
    old_run_id = "old-submit-campaign__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", old_run_id)
    generation.cases.input_generation.generate_input_cases(
        batch,
        len(batch.case_indices),
        storage_root=storage,
    )

    def fail_conversion(*_args: Any, **_kwargs: Any) -> None:
        message = "historical retained conversion failure"
        raise RuntimeError(message)

    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(generation.runtime.CaseExecutionError, match="historical retained conversion failure"):
            generation.runtime.run_case(
                batch,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "old campaign solver work",
            )

    new_commit = "b" * 40
    monkeypatch.setenv("GENERATION_GIT_COMMIT", new_commit)
    submitted: list[list[str]] = []

    def submit_case(command: list[str], **_kwargs: Any) -> str:
        submitted.append(command)
        return str(7_000 + len(submitted))

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: new_commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", submit_case)
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
            "active": {},
            "accounted": {},
        },
    )

    manifest = generation.campaign.submit_campaign(
        campaign,
        git_commit=new_commit,
        storage_root=storage,
    )

    assert submitted == [record["command"] for record in manifest["submissions"]]
    assert [record["case"]["case_index"] for record in manifest["submissions"]] == [2, 3]
    historical = generation.publication.attempt.latest_case_attempt(
        batch,
        1,
        old_run_id,
        storage_root=storage,
    )
    assert historical is not None
    assert historical.payload["case_state"] == "conversion_failed"


@pytest.mark.integration
def test_historical_solver_failure_is_not_misclassified_as_replayable(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave genuine old solver work eligible for a new campaign submission."""
    old_run_id = "old-solver-campaign__0123456789abcdef"
    new_run_id = "new-solver-campaign__fedcba9876543210"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", old_run_id)
    config, storage = _production_case(generation_config_factory, fake_comsol, tmp_path)
    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    with pytest.raises(generation.runtime.CaseExecutionError):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / "historical failed solver work",
        )
    old_attempt = generation.publication.attempt.latest_case_attempt(
        config,
        1,
        old_run_id,
        storage_root=storage,
    )
    assert old_attempt is not None
    assert old_attempt.payload["failure_stage"] == "solver"
    task = cluster_service.CampaignTask(
        batch_name=config.batch_name,
        batch_id=config.batch_id,
        case_index=1,
        case_id=config.case_id(1),
    )

    assert (
        generation.campaign._admitted_case_attempt(  # noqa: SLF001 -- tests cross-run admission
            {"campaign_run_id": new_run_id, "git_commit": "b" * 40},
            config,
            task,
            storage_root=storage,
        )
        is None
    )


@pytest.mark.integration
def test_publication_replay_uses_converted_payload_without_reconversion(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish a retained converted payload without COMSOL or reconversion."""
    config, storage = _production_case(
        generation_config_factory,
        fake_comsol,
        tmp_path,
    )

    def fail_publication(*_args: Any, **_kwargs: Any) -> None:
        message = "synthetic publication defect"
        raise RuntimeError(message)

    with monkeypatch.context() as scoped:
        scoped.setattr(runtime_service, "publish_completed_case", fail_publication)
        with pytest.raises(RuntimeError, match="synthetic publication defect"):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "solver work",
            )
    attempt = _attempt(config, storage)
    assert attempt.payload["case_state"] == "publication_failed"
    assert attempt.payload["conversion_state"] == "succeeded"
    assert attempt.replay_available

    monkeypatch.setenv("GENERATION_GIT_COMMIT", "b" * 40)
    monkeypatch.setattr(comsol_service, "build_comsol_command", _forbid_comsol)

    def forbid_conversion(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Publication replay invoked conversion.")

    monkeypatch.setattr(
        storage_service,
        "convert_exports_to_hdf5",
        forbid_conversion,
    )
    outcome = generation.runtime.replay_case_postprocessing(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / "publication replay work",
    )

    assert outcome.status == "replayed"
    generation.runtime.validate_completed_case(config, 1, storage_root=storage)
    processing = json.loads((outcome.processed_directory / "processing_provenance.json").read_text(encoding="utf-8"))
    assert processing["mode"] == "replay_publication"
    assert processing["solver_git_commit"] == "a" * 40
    assert processing["processing_git_commit"] == "b" * 40
    replay = json.loads((attempt.directory / "replay.json").read_text(encoding="utf-8"))
    assert replay["cleanup_state"] == "complete"


@pytest.mark.integration
def test_unchanged_replay_failure_is_blocked_without_a_workspace(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block identical retained conversion retries before scratch or COMSOL construction."""
    config, storage = _production_case(generation_config_factory, fake_comsol, tmp_path)
    original_convert = storage_service.convert_exports_to_hdf5

    def fail_conversion(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_RETAINED_CONVERSION_DEFECT)

    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(generation.runtime.CaseExecutionError):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "initial replay gate work",
            )
    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(RuntimeError, match=_RETAINED_CONVERSION_DEFECT):
            generation.runtime.replay_case_postprocessing(
                config,
                1,
                storage_root=storage,
                work_root=tmp_path / "failed replay gate work",
            )

    blocked_attempt = _attempt(config, storage)
    evidence_path = blocked_attempt.directory / "replay_failure.json"
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    assert evidence["schema_version"] == 1
    assert evidence["result"] == "failed"
    assert evidence["source_attempt"]["attempt_index"] == 1
    assert evidence["failed_attempt"]["attempt_index"] == _REPLAY_FAILURE_ATTEMPT_INDEX
    status = runtime_service.replay_case_postprocessing_status(config, blocked_attempt)
    assert status["blocked"] is True
    assert status["attempt_count"] == 1
    assert status["evidence_path"] == str(evidence_path)

    def forbid_workspace(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Blocked replay created a workspace.")

    original_prepare = runtime_service.prepare_case_work_directory
    monkeypatch.setattr(runtime_service, "prepare_case_work_directory", forbid_workspace)
    monkeypatch.setattr(comsol_service, "build_comsol_command", _forbid_comsol)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "c" * 40)
    outcome = generation.runtime.replay_case_postprocessing(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / "blocked replay work",
    )
    assert outcome.status == "replay_blocked"
    assert _attempt(config, storage).payload["attempt_index"] == _REPLAY_FAILURE_ATTEMPT_INDEX

    monkeypatch.setattr(runtime_service, "prepare_case_work_directory", original_prepare)
    monkeypatch.setattr(storage_service, "convert_exports_to_hdf5", original_convert)
    original_identities = runtime_service._current_replay_identities  # noqa: SLF001 -- validates narrow identity change

    for changed_key in ("converter_dependency_sha256", "output_contract_sha256"):

        def changed_identity(*args: Any, _key: str = changed_key, **kwargs: Any) -> dict[str, str]:
            identities = original_identities(*args, **kwargs)
            identities[_key] = "f" * 64
            return identities

        monkeypatch.setattr(runtime_service, "_current_replay_identities", changed_identity)
        changed_status = runtime_service.replay_case_postprocessing_status(config, blocked_attempt)
        assert changed_status["eligible"] is True
        assert changed_status["reason"] == "identity_changed"

    def changed_converter_identity(*args: Any, **kwargs: Any) -> dict[str, str]:
        identities = original_identities(*args, **kwargs)
        identities["converter_dependency_sha256"] = "f" * 64
        return identities

    monkeypatch.setattr(runtime_service, "_current_replay_identities", changed_converter_identity)
    outcome = generation.runtime.replay_case_postprocessing(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / "changed converter replay work",
    )
    assert outcome.status == "replayed"
    assert evidence_path.read_bytes() == evidence_bytes


def test_replay_failure_evidence_rejects_tampered_payload_identity(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when replay evidence no longer binds the retained source payload."""
    config, storage = _production_case(generation_config_factory, fake_comsol, tmp_path)

    def fail_conversion(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_REPLAY_EVIDENCE_DEFECT)

    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(generation.runtime.CaseExecutionError):
            generation.runtime.run_case(
                config,
                1,
                cores_per_case=1,
                storage_root=storage,
                work_root=tmp_path / "initial evidence work",
            )
    with monkeypatch.context() as scoped:
        scoped.setattr(storage_service, "convert_exports_to_hdf5", fail_conversion)
        with pytest.raises(RuntimeError, match=_REPLAY_EVIDENCE_DEFECT):
            generation.runtime.replay_case_postprocessing(
                config,
                1,
                storage_root=storage,
                work_root=tmp_path / "failed evidence work",
            )
    attempt = _attempt(config, storage)
    path = attempt.directory / "replay_failure.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["replay_payload"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="replay failure"):
        generation.publication.attempt.load_attempt(attempt.directory)


def test_completed_hdf5_repair_rejects_an_unowned_campaign_run(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject direct callers that bind repair evidence to another campaign."""
    config_path, _template = generation_config_factory(
        campaign_purpose="technical_runtime_smoke",
        natural_count=2,
        retain_raw_csv=True,
        retain_solved_model=True,
    )
    batch_name = generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=batch_name,
    )
    storage = tmp_path / "unowned repair storage"
    storage.mkdir()
    run_id = "different-smoke__0123456789abcdef"
    run_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / run_id
    run_directory.mkdir(parents=True)

    def reject_batch(_name: str) -> Any:
        message = "batch belongs to another campaign"
        raise ValueError(message)

    monkeypatch.setattr(
        runtime_service.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {
            "campaign_run_id": run_id,
            "state": "complete",
        },
    )
    monkeypatch.setattr(
        runtime_service.campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: SimpleNamespace(
            campaign_purpose="technical_runtime_smoke",
            batch=reject_batch,
        ),
    )

    with pytest.raises(RuntimeError, match="not owned"):
        generation.runtime.repair_completed_case_hdf5_from_retained_exports(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
            blocking_lock=False,
        )

    assert runtime_service.case_lock_path(
        config,
        1,
        storage_root=storage,
    ).is_file()


@pytest.mark.integration
def test_completed_smoke_hdf5_reconstructs_from_retained_exports_without_comsol(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair hash-bound completed HDF5 bytes from admitted Full-Retention CSV."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        campaign_purpose="technical_runtime_smoke",
        natural_count=2,
        retain_raw_csv=True,
        retain_solved_model=True,
    )
    batch_name = generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=batch_name,
    )
    storage = tmp_path / "completed Smoke repair storage"
    generation.cases.input_generation.generate_input_cases(
        config,
        len(config.case_indices),
        storage_root=storage,
    )
    run_id = "smoke-repair__0123456789abcdef"
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "initial Smoke work",
    )
    run_directory = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns" / run_id
    run_directory.mkdir(parents=True)
    manifest = {"campaign_run_id": run_id, "state": "complete"}
    monkeypatch.setattr(
        runtime_service.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        runtime_service.campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: campaign,
    )
    hdf5_path = outcome.processed_directory / "case.h5"
    original = hdf5_path.read_bytes()
    original_sha256 = common.serialization.file_sha256(hdf5_path)
    corrupted = bytearray(original)
    corrupted[len(corrupted) // 2] ^= 0x01
    hdf5_path.write_bytes(corrupted)
    with pytest.raises(RuntimeError, match="artifact integrity"):
        generation.runtime.validate_completed_case(
            config,
            1,
            storage_root=storage,
            validation_depth="deep",
        )

    monkeypatch.setattr(comsol_service, "build_comsol_command", _forbid_comsol)
    unsafe_root = run_directory / "hdf5_reconstructions"
    outside = tmp_path / "unsafe reconstruction target"
    outside.mkdir()
    unsafe_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="receipt root is unsafe"):
        generation.runtime.repair_completed_case_hdf5_from_retained_exports(
            config,
            1,
            campaign_run_id=run_id,
            storage_root=storage,
            work_root=tmp_path / "rejected HDF5 reconstruction work",
        )
    assert hdf5_path.read_bytes() == bytes(corrupted)
    unsafe_root.unlink()

    recovery = generation.runtime.repair_completed_case_hdf5_from_retained_exports(
        config,
        1,
        campaign_run_id=run_id,
        storage_root=storage,
        work_root=tmp_path / "HDF5 reconstruction work",
    )

    assert recovery["status"] == "complete"
    assert recovery["comsol_executed"] is False
    assert recovery["reconstructed_hdf5_sha256"] == original_sha256
    assert hdf5_path.read_bytes() == original
    generation.runtime.validate_completed_case(
        config,
        1,
        storage_root=storage,
        validation_depth="deep",
    )
    receipt = storage / recovery["receipt"]
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["comsol_executed"] is False
    assert persisted["source_exports"]
