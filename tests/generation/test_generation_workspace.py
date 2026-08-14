# ruff: noqa: S101
"""Disposable workspace ownership, containment, and cleanup contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src import common, generation
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_preparation as preparation
from src.generation.runtime import generation_runtime_workspace as workspace


def _steady_natural_batch_name() -> str:
    """Return the canonical synthetic steady natural-batch selector."""
    return generation.cases.config.build_batch_name(
        "steady_flow",
        "lentil",
        "natural",
    )


def test_default_storage_is_the_canonical_repository_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the portable default outside the repository without requiring HOME."""
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(repository))
    monkeypatch.delenv("STORAGE_ROOT", raising=False)

    resolved = workspace.resolve_storage_root(None, create=True)

    assert resolved == (tmp_path / "storage").resolve()
    assert resolved.is_dir()
    assert not resolved.is_relative_to(repository.resolve())


def test_case_workspaces_are_unique_marked_and_support_spaces(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect collision-safe case roots and exact profile-local inputs."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        natural_count=2,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_steady_natural_batch_name(),
    )
    storage = tmp_path / "storage root"
    work = tmp_path / "scratch root with spaces"
    first = runtime_service.prepare_case_work_directory(
        config,
        1,
        storage_root=storage,
        work_root=work,
    )
    second = runtime_service.prepare_case_work_directory(
        config,
        2,
        storage_root=storage,
        work_root=work,
    )
    assert first.work_directory != second.work_directory
    assert first.work_directory.parent == work.resolve()
    assert second.work_directory.parent == work.resolve()
    for prepared in (first, second):
        assert prepared.workspace_marker.is_file()
        marker = json.loads(prepared.workspace_marker.read_text(encoding="utf-8"))
        assert marker["case_id"] == prepared.bundle.case_id
        assert marker["work_directory"] == str(prepared.work_directory)
        assert {path.name for path in prepared.bundle.input_paths} == {"fields.csv"}
        assert (prepared.work_directory / "model.mph").is_file()
        assert not (prepared.work_directory / "scalars.csv").exists()
        assert not (prepared.work_directory / "schedule.csv").exists()
        workspace.cleanup_case_workspace(
            prepared.work_directory,
            allowed_root=prepared.work_root,
            storage_root=storage.resolve(),
            expected_run_id=prepared.workspace_run_id,
            expected_case_id=prepared.bundle.case_id,
        )
        assert not prepared.work_directory.exists()


def test_cleanup_guard_rejects_broad_unowned_and_active_targets(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect every destructive cleanup boundary and active-job refusal."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_steady_natural_batch_name(),
    )
    storage = (tmp_path / "storage").resolve()
    work = (tmp_path / "work").resolve()
    prepared = runtime_service.prepare_case_work_directory(
        config,
        1,
        storage_root=storage,
        work_root=work,
    )
    protected = (
        "",
        "/",
        str(Path.home().resolve()),
        str(common.paths.get_project_root().resolve()),
        str(storage),
        str(common.paths.get_generation_processed_root(storage_root=storage)),
        str(common.paths.get_datasets_root(storage_root=storage)),
        str(work),
        str(tmp_path / "outside"),
    )
    for target in protected:
        with pytest.raises((ValueError, FileNotFoundError)):
            workspace.cleanup_case_workspace(
                target,
                allowed_root=work,
                storage_root=storage,
                expected_run_id=prepared.workspace_run_id,
                expected_case_id=prepared.bundle.case_id,
            )

    missing_marker = work / "missing-marker"
    missing_marker.mkdir()
    with pytest.raises(ValueError, match="marker"):
        workspace.cleanup_case_workspace(
            missing_marker,
            allowed_root=work,
            storage_root=storage,
            expected_run_id=prepared.workspace_run_id,
            expected_case_id=prepared.bundle.case_id,
        )

    marker = json.loads(prepared.workspace_marker.read_text(encoding="utf-8"))
    marker["run_id"] = "wrong-run"
    prepared.workspace_marker.write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marker identity"):
        workspace.cleanup_case_workspace(
            prepared.work_directory,
            allowed_root=work,
            storage_root=storage,
            expected_run_id=prepared.workspace_run_id,
            expected_case_id=prepared.bundle.case_id,
        )
    marker["run_id"] = prepared.workspace_run_id
    marker["slurm_job_id"] = "12345"
    prepared.workspace_marker.write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        workspace,
        "_slurm_job_is_active",
        lambda _job_id: True,
    )
    with pytest.raises(RuntimeError, match="remains active"):
        workspace.cleanup_case_workspace(
            prepared.work_directory,
            allowed_root=work,
            storage_root=storage,
            expected_run_id=prepared.workspace_run_id,
            expected_case_id=prepared.bundle.case_id,
        )
    workspace.cleanup_case_workspace(
        prepared.work_directory,
        allowed_root=work,
        storage_root=storage,
        expected_run_id=prepared.workspace_run_id,
        expected_case_id=prepared.bundle.case_id,
        allow_active_job_id="12345",
    )


def test_interrupted_case_persists_cancelled_evidence_and_remains_rerunnable(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect interruption evidence, scratch cleanup, and fresh retry identity."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_steady_natural_batch_name(),
    )
    storage = tmp_path / "storage"
    observed: dict[str, Path] = {}

    def interrupt(
        _config: Any,
        prepared: preparation.PreparedCase,
        **_kwargs: Any,
    ) -> None:
        observed["work"] = prepared.work_directory
        raise KeyboardInterrupt

    monkeypatch.setattr(
        runtime_service,
        "execute_prepared_case",
        interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / "work",
        )
    assert not observed["work"].exists()
    failure = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert failure["state"] == "cancelled"
    assert failure["scratch_cleanup"]["status"] == "complete"
    assert not generation.runtime.completed_case_is_valid(
        config,
        1,
        storage_root=storage,
    )
