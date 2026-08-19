# ruff: noqa: S101
"""No-COMSOL conversion and publication replay contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_comsol as comsol_service

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
