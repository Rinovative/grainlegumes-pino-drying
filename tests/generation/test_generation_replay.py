# ruff: noqa: S101
"""No-COMSOL conversion and publication replay contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from src import generation
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
