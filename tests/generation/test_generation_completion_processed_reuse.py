# ruff: noqa: S101
"""Completion reuse contracts for durable processed cases and retired raw inputs."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from src import common, generation
from src.generation.cli import cli_generation


def _natural_batch_name() -> str:
    """Return the synthetic transient natural-batch selector."""
    return generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )


def _publish_completed_case(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> tuple[Any, Path, Path, Path]:
    """Publish one complete case while retaining a second manifest member."""
    config_path, _template = generation_config_factory(
        executable=fake_comsol,
        natural_count=2,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name(),
    )
    storage = tmp_path / "source-storage"
    generation.cases.input_generation.generate_input_cases(
        config,
        2,
        storage_root=storage,
    )
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "work",
    )
    return config, config_path, storage, outcome.processed_directory


def _processed_directory(config: Any, storage: Path) -> Path:
    """Return the copied case-one processed directory."""
    return generation.runtime.processed_case_directory(
        config,
        1,
        storage_root=storage,
    )


def _persisted_raw_case(config: Any, storage: Path) -> Path:
    """Return the raw case named by processed provenance, independent of HEAD."""
    processed = _processed_directory(config, storage)
    provenance = json.loads((processed / "provenance.json").read_text(encoding="utf-8"))
    return common.paths.resolve_generation_input_generation_raw_directory(
        config.batch_storage_name,
        provenance["input_generation_id"],
        storage_root=storage,
    ) / config.case_id(1)


def _persisted_metadata_directory(config: Any, storage: Path) -> Path:
    """Return durable input-generation metadata named by processed provenance."""
    processed = _processed_directory(config, storage)
    provenance = json.loads((processed / "provenance.json").read_text(encoding="utf-8"))
    return (
        common.paths.get_generation_meta_root(storage_root=storage)
        / config.batch_storage_name
        / "input_generations"
        / provenance["input_generation_id"]
    )


def _copy_without_raw(config: Any, source: Path, destination: Path) -> Path:
    """Copy test-owned storage and retire only case one's raw directory."""
    shutil.copytree(source, destination)
    shutil.rmtree(_persisted_raw_case(config, destination))
    return destination


def _rebind_processed_artifact(processed: Path, relative_path: str) -> None:
    """Rebind one test-mutated artifact through publication and success digests."""
    artifact = processed / relative_path
    provenance_path = processed / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["artifacts"][relative_path] = {
        "sha256": common.serialization.file_sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    common.serialization.atomic_write_json(provenance_path, provenance)
    success_path = processed / "_SUCCESS"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["provenance_sha256"] = common.serialization.file_sha256(provenance_path)
    common.serialization.atomic_write_json(success_path, success)


@pytest.mark.integration
def test_completion_reuse_resolves_historical_raw_and_admits_processed_only(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reuse historic raw when present and exact processed evidence after retirement."""
    config, config_path, source, processed = _publish_completed_case(
        generation_config_factory,
        fake_comsol,
        tmp_path,
    )
    provenance = json.loads((processed / "provenance.json").read_text(encoding="utf-8"))
    source_commit = provenance["git_commit"]
    original_payload = json.loads((_persisted_raw_case(config, source) / "case.json").read_text(encoding="utf-8"))

    monkeypatch.setenv("GENERATION_GIT_COMMIT", "b" * 40)
    assert generation.runtime.completed_case_is_valid(config, 1, storage_root=source)
    raw_admission = generation.runtime.admit_completed_case(
        config,
        1,
        storage_root=source,
        validation_depth="full",
        git_commit=source_commit,
    )
    assert raw_admission.source_kind == "raw_and_processed"
    assert raw_admission.raw_directory == _persisted_raw_case(config, source)

    retired = _copy_without_raw(config, source, tmp_path / "retired-storage")
    processed_admission = generation.runtime.admit_completed_case(
        config,
        1,
        storage_root=retired,
        validation_depth="full",
        git_commit=source_commit,
    )
    assert processed_admission.source_kind == "processed_only"
    assert processed_admission.raw_directory is None
    assert processed_admission.raw_artifacts == ()
    assert processed_admission.metadata_payload() == original_payload
    validated = generation.runtime.validate_completed_case(
        config,
        1,
        storage_root=retired,
        validation_depth="full",
    )
    assert validated["simulation_case_id"] == processed_admission.simulation_case_id
    assert (
        cli_generation.main(
            [
                "validate-case",
                str(config_path),
                "--only-batch",
                _natural_batch_name(),
                "1",
                "--storage-root",
                str(retired),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "case_id": config.case_id(1),
        "simulation_case_id": processed_admission.simulation_case_id,
        "status": "valid",
    }

    historical = _copy_without_raw(config, source, tmp_path / "historical-timing-storage")
    historical_processed = _processed_directory(config, historical)
    timing_path = historical_processed / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    removed = {
        name: timing.pop(name)
        for name in (
            "comsol_stationary_airflow_seconds",
            "comsol_transient_drying_seconds",
            "comsol_scientific_solver_seconds",
            "comsol_solver_timing",
        )
        if name in timing
    }
    assert removed
    common.serialization.atomic_write_json(timing_path, timing)
    _rebind_processed_artifact(historical_processed, "timing.json")
    assert (
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=historical,
            validation_depth="full",
            git_commit=source_commit,
        ).source_kind
        == "processed_only"
    )


@pytest.mark.integration
def test_completion_reuse_fails_closed_on_missing_or_contradictory_evidence(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Reject insufficient metadata, contradictions, failed status, and collisions."""
    config, _config_path, source, _processed = _publish_completed_case(
        generation_config_factory,
        fake_comsol,
        tmp_path,
    )

    contradictory_raw = tmp_path / "contradictory-raw-storage"
    shutil.copytree(source, contradictory_raw)
    raw_case_path = _persisted_raw_case(config, contradictory_raw) / "case.json"
    raw_case = json.loads(raw_case_path.read_text(encoding="utf-8"))
    raw_case["simulation_case_id"] = "f" * 64
    common.serialization.atomic_write_json(raw_case_path, raw_case)
    with pytest.raises((RuntimeError, ValueError), match=r"raw|payload|digest|identity"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=contradictory_raw,
            validation_depth="full",
        )

    symlinked_raw_batch = tmp_path / "symlinked-raw-batch-storage"
    shutil.copytree(source, symlinked_raw_batch)
    raw_batch = _persisted_raw_case(config, symlinked_raw_batch).parent
    raw_batch_target = tmp_path / "aliased-raw-batch" / config.batch_storage_name / "input_generations" / raw_batch.name
    raw_batch_target.parent.mkdir(parents=True)
    shutil.copytree(raw_batch, raw_batch_target)
    shutil.rmtree(raw_batch)
    raw_batch.symlink_to(raw_batch_target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe contradictory raw"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=symlinked_raw_batch,
            validation_depth="full",
        )

    symlinked_raw = _copy_without_raw(
        config,
        source,
        tmp_path / "symlinked-raw-storage",
    )
    _persisted_raw_case(config, symlinked_raw).symlink_to(
        _persisted_raw_case(config, source),
        target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="unsafe contradictory raw"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=symlinked_raw,
            validation_depth="full",
        )

    missing_metadata = _copy_without_raw(
        config,
        source,
        tmp_path / "missing-metadata-storage",
    )
    (_persisted_metadata_directory(config, missing_metadata) / "input_generation_manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=missing_metadata,
            validation_depth="full",
        )

    symlinked_metadata = _copy_without_raw(
        config,
        source,
        tmp_path / "symlinked-metadata-storage",
    )
    metadata_directory = _persisted_metadata_directory(config, symlinked_metadata)
    metadata_target = tmp_path / "aliased-metadata" / config.batch_storage_name / "input_generations" / metadata_directory.name
    metadata_target.parent.mkdir(parents=True)
    shutil.copytree(metadata_directory, metadata_target)
    shutil.rmtree(metadata_directory)
    metadata_directory.symlink_to(metadata_target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe symbolic link"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=symlinked_metadata,
            validation_depth="full",
        )

    contradictory_metadata = _copy_without_raw(
        config,
        source,
        tmp_path / "contradictory-metadata-storage",
    )
    manifest_path = _persisted_metadata_directory(config, contradictory_metadata) / "input_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["simulation_case_id"] = "e" * 64
    common.serialization.atomic_write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="disagrees"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=contradictory_metadata,
            validation_depth="full",
        )

    duplicate_identity = _copy_without_raw(
        config,
        source,
        tmp_path / "duplicate-identity-storage",
    )
    manifest_path = _persisted_metadata_directory(config, duplicate_identity) / "input_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][1]["case_input_id"] = manifest["cases"][0]["case_input_id"]
    common.serialization.atomic_write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=duplicate_identity,
            validation_depth="full",
        )

    failed_status = _copy_without_raw(
        config,
        source,
        tmp_path / "failed-status-storage",
    )
    failed_processed = _processed_directory(config, failed_status)
    status_path = failed_processed / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["case_state"] = "failed"
    status["solver_success"] = False
    status["stages"]["solver"] = "failed"
    common.serialization.atomic_write_json(status_path, status)
    _rebind_processed_artifact(failed_processed, "status.json")
    with pytest.raises(RuntimeError, match="scientific or COMSOL success"):
        generation.runtime.admit_completed_case(
            config,
            1,
            storage_root=failed_status,
            validation_depth="full",
        )
