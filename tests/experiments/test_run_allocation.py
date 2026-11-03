"""Protect collision-safe run allocation, writer leases, resume, and status transitions.

Responsibilities:
  - Check collision refusal, writer leases, atomic status, runtime sessions, and resume admission
  - Check admission of a start that names an existing run: continue, report complete, or reject
  - Check that a started run, but no Optuna trial, records its folder as a job log owner

Design principles:
  - Scientific config stays immutable while permitted duration extensions are explicit

This module does NOT:
  - Restore checkpoint tensors or train models
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import queue
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from support import configs, run_state

from src import common, experiments
from src.experiments.config import experiments_config_loader as config_loader
from src.experiments.config import experiments_config_naming as config_naming


def _minimal_config(output_root: Path) -> dict[str, Any]:
    """Return one validated synthetic config for allocation tests."""
    config = configs.resolve(configs.direct_config())
    config["paths"]["output_root"] = str(output_root)
    return config


def test_nearby_scientific_values_share_one_readable_name_and_are_rejected(
    tmp_path: Path,
) -> None:
    """Bind one readable name to one scientific content: a nearby value is a conflict, not a new leaf."""
    raw = configs.direct_config(model_kind="uno")
    first_raw = copy.deepcopy(raw)
    second_raw = copy.deepcopy(raw)
    first_raw["model"]["params"]["mode_ratio"] = 0.4951
    second_raw["model"]["params"]["mode_ratio"] = 0.4952
    first = configs.resolve(first_raw, output_root=tmp_path)
    second = configs.resolve(second_raw, output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(first)
    assert run_dir == common.paths.resolve_config_run_dir(second)  # Labels round the ratio to three decimals.
    run_state.write_lightweight_run_state(
        run_dir,
        first,
        status="interrupted",
        completed_epoch=2,
    )
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    config_path = configs.write_yaml(
        tmp_path / "requested.yaml",
        second_raw,
    )

    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        second,
        config_path=config_path,
    )

    assert report["resume_compatibility"] == "incompatible"
    assert "model.params.mode_ratio" in report["reason"]
    after = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    assert after == before


def test_fresh_run_persists_one_canonical_name(tmp_path: Path) -> None:
    """Persist the generated synthetic name across directory, config, and summary."""
    config = configs.resolve(configs.direct_config())
    config["paths"]["output_root"] = str(tmp_path)

    run_dir = experiments.run.prepare_fresh_run(config)
    saved = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(run_dir))
    summary = experiments.run.read_run_summary(run_dir)

    assert run_dir == common.paths.resolve_config_run_dir(config)
    assert saved["run"]["name"] == config["run"]["name"]
    assert summary["run_name"] == config["run"]["name"]


def test_fresh_collision_rejects_before_any_run_write(tmp_path: Path) -> None:
    """
    Prepare a fresh run at an existing leaf containing an authoritative marker.

    Collision must fail before config/summary writes and preserve every byte,
    leaving continuation to run admission rather than overwriting the leaf.
    """
    run_dir = tmp_path / "outputs" / "synthetic_task" / "runs" / "run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "marker.bin"
    marker.write_bytes(b"do-not-touch")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    with pytest.raises(FileExistsError, match="already exists"):
        experiments.run.prepare_fresh_run(_minimal_config(tmp_path / "outputs"), run_dir=run_dir)

    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    assert after == before
    assert not (run_dir / "config.yaml").exists()
    assert not (run_dir / "summary.json").exists()


def _synthetic_config_path(tmp_path: Path) -> Path:
    """Write the direct request used by read-only admission tests."""
    return configs.write_config(tmp_path, configs.direct_config(), canonical_root=tmp_path)


def _resolved_admission_config(output_root: Path) -> dict[str, Any]:
    """Return one complete artificial config redirected to a temporary root."""
    resolved = configs.resolve(configs.direct_config())
    resolved["paths"]["output_root"] = str(output_root)
    return resolved


def _admission_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    completed_epoch: int | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Create and inspect one lightweight existing synthetic run."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "STORAGE_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(
        run_dir,
        config,
        status=status,
        completed_epoch=(config["training"]["epochs"] if completed_epoch is None else completed_epoch),
    )
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=config_path,
    )
    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    assert after == before
    return report, run_dir, config


def test_changed_science_rejection_advises_revision_without_resume_or_gpu_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejecting a start whose science differs from the existing run advises a revision, never a resume command."""
    _report, run_dir, config = _admission_report(
        tmp_path,
        monkeypatch,
        status="interrupted",
        completed_epoch=1,
    )
    changed = copy.deepcopy(config)
    changed["optimizer"]["lr"] = float(config["optimizer"]["lr"]) * 2

    error = experiments.run.reject_existing_fresh_run(
        run_dir,
        changed,
        config_path=tmp_path / "synthetic-request.yaml",
    )

    reason = str(error.report["reason"])
    assert "optimizer.lr" in reason
    assert "--revision" in reason
    assert "--resume" not in reason
    assert "--mode gpu" not in reason


def test_completed_admission_is_distinct_from_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify a complete run without presenting it as resumable."""
    report, _run_dir, config = _admission_report(
        tmp_path,
        monkeypatch,
        status="completed",
        completed_epoch=None,
    )

    assert report["resume_compatibility"] == "completed"
    assert report["completed_epoch"] == config["training"]["epochs"]


def test_damaged_or_malformed_completed_run_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing selected state and invalid lifecycle JSON."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "STORAGE_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(
        run_dir,
        config,
        status="completed",
        completed_epoch=config["training"]["epochs"],
        include_best=False,
    )

    damaged = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=config_path,
    )
    assert damaged["resume_compatibility"] == "incompatible"

    common.paths.resolve_run_summary_path(run_dir).write_text(
        "{invalid",
        encoding="utf-8",
    )
    malformed = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=config_path,
    )
    assert malformed["resume_compatibility"] == "incompatible"


def test_active_writer_blocks_resume_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never admit simultaneous resume while another writer owns the run."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "STORAGE_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(
        run_dir,
        config,
        status="interrupted",
        completed_epoch=2,
    )

    with common.locking.exclusive_file_lock(
        common.paths.resolve_run_lock_path(run_dir),
        blocking=False,
    ):
        report = experiments.run.inspect_existing_run_admission(
            run_dir,
            config,
            config_path=config_path,
        )

    assert report["active_lock"] is True
    assert report["resume_compatibility"] == "blocked"

    lock_path = common.paths.resolve_run_lock_path(run_dir)
    lock_path.unlink()
    missing_target = tmp_path / "missing-writer-marker.lock"
    lock_path.symlink_to(missing_target)
    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=config_path,
    )
    assert report["active_lock"] is None
    assert report["resume_compatibility"] == "incompatible"
    assert "Active writer state could not be verified" in report["reason"]
    assert lock_path.readlink() == missing_target
    assert not missing_target.exists()


def test_existing_read_only_lock_marker_reports_exclusive_writer_without_mutation(tmp_path: Path) -> None:
    """A read-only probe detects an exclusive lease without changing its marker."""
    lock_path = tmp_path / "writer.lock"
    assert common.locking.file_lock_is_active(lock_path) is False
    assert not lock_path.exists()

    lock_path.write_bytes(b"persistent marker")
    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        lock_path.chmod(0o400)
        assert common.locking.file_lock_is_active(lock_path) is True

    assert common.locking.file_lock_is_active(lock_path) is False
    assert lock_path.read_bytes() == b"persistent marker"
    assert lock_path.stat().st_mode & 0o777 == 0o400

    link_path = tmp_path / "writer-link.lock"
    link_path.symlink_to(lock_path)
    with pytest.raises(OSError, match="Lock path is not a regular file"):
        common.locking.file_lock_is_active(link_path)
    broken_link_path = tmp_path / "broken-writer-link.lock"
    broken_link_path.symlink_to(tmp_path / "missing-marker.lock")
    with pytest.raises(OSError, match="Lock path is not a regular file"):
        common.locking.file_lock_is_active(broken_link_path)


def test_existing_target_that_cannot_continue_rejects_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect an existing leaf before device, model, or tracking setup; a failed run cannot continue."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "STORAGE_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(
        run_dir,
        config,
        status="failed",
        completed_epoch=2,
    )
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    def reject_device(*_args: Any, **_kwargs: Any) -> None:
        msg = "device resolution must not run"
        raise AssertionError(msg)

    monkeypatch.setattr(
        experiments.run.learning.device,
        "resolve_device",
        reject_device,
    )
    with pytest.raises(experiments.run.ExistingRunAdmissionError, match="cannot continue automatically"):
        experiments.run.run_experiment(
            config_path,
            output_root=output_root,
        )

    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    assert after == before


def test_rerunning_a_completed_run_returns_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run that reached the requested epochs is reported complete; no run state is written or trained, only its binding repaired."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="completed", completed_epoch=config["training"]["epochs"])
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {
            "schema_version": experiments.run.RUN_SUMMARY_SCHEMA_VERSION,
            "status": "completed",
            "completed_epoch": config["training"]["epochs"],
            "best_epoch": 1,
            "best_metric": 0.25,
        },
    )
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    def execute(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a complete run must not train again")

    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", execute)

    outcome = experiments.run.run_experiment(config_path, output_root=output_root)

    assert outcome["already_complete"] is True
    assert "already_complete" in capsys.readouterr().out
    assert outcome["run_dir"] == run_dir.resolve()
    assert outcome["result"] == {"status": "completed", "best_epoch": 1, "best_metric": 0.25}
    registry_state = output_root.relative_to(tmp_path) / ".state" / "identities"
    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    assert {path: payload for path, payload in after.items() if not path.is_relative_to(registry_state)} == before
    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert entry.target == run_dir.resolve()


def test_initialization_failure_never_looks_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject config-save failure immediately after run-directory allocation.

    The leaf must retain a failed status and remain non-loadable, preventing partial
    initialization from appearing complete merely because a directory exists.
    """
    run_dir = tmp_path / "run"

    def fail_save(*_args: Any, **_kwargs: Any) -> None:
        """Fail exactly where fresh-run initialization persists config."""
        msg = "injected config failure"
        raise OSError(msg)

    monkeypatch.setattr(experiments.run.config_loader, "save_yaml", fail_save)
    with pytest.raises(OSError, match="injected config failure"):
        experiments.run.prepare_fresh_run(_minimal_config(tmp_path), run_dir=run_dir)

    assert run_dir.is_dir()
    assert experiments.run.read_run_summary(run_dir)["status"] == "failed"


@pytest.mark.parametrize(("status", "evaluable"), [("completed", True), ("failed", False)])
def test_only_completed_runs_pass_the_shallow_evaluable_check(tmp_path: Path, status: str, evaluable: bool) -> None:
    """A failed run with every payload file present is still not evaluable."""
    run_dir = tmp_path / "run"
    for filename in common.paths.EVALUABLE_RUN_REQUIRED_FILES:
        (run_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / filename).touch()
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"schema_version": 1, "status": status},
    )

    assert common.paths.is_evaluable_run_dir(run_dir) is evaluable


@pytest.mark.parametrize("invalid_version", [True, 2])
def test_run_summary_requires_type_exact_schema_version(
    tmp_path: Path,
    invalid_version: object,
) -> None:
    """Reject every non-current integer and numeric lookalike before lifecycle use."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"schema_version": invalid_version, "status": "completed"},
    )

    with pytest.raises(experiments.run.RunLifecycleError, match="run summary schema"):
        experiments.run.read_run_summary(run_dir)


def test_optuna_trial_paths_are_study_and_trial_qualified(tmp_path: Path) -> None:
    """
    Resolve equal trial numbers across studies, datasets, epoch budgets and revisions and unequal trials within one study.

    All leaves must be distinct and exact reallocation must fail, protecting Optuna
    output ownership from study/trial collisions.
    """
    first = common.paths.resolve_study_dir("task", "data-a", "study-a", 5, output_root=tmp_path) / "trials" / "native003"
    other_study = common.paths.resolve_study_dir("task", "data-a", "study-b", 5, output_root=tmp_path) / "trials" / "native003"
    other_dataset = common.paths.resolve_study_dir("task", "data-b", "study-a", 5, output_root=tmp_path) / "trials" / "native003"
    other_budget = common.paths.resolve_study_dir("task", "data-a", "study-a", 6, output_root=tmp_path) / "trials" / "native003"
    other_revision = common.paths.resolve_study_dir("task", "data-a", "study-a", 5, "again", output_root=tmp_path) / "trials" / "native003"
    other_trial = common.paths.resolve_study_dir("task", "data-a", "study-a", 5, output_root=tmp_path) / "trials" / "native004"
    assert first.name == "native003"
    assert first.parent.parent.name == "study-a__ep-5"
    assert other_revision.parent.parent.name == "study-a__ep-5__rev-again"

    assert len({first, other_study, other_dataset, other_budget, other_revision, other_trial}) == 6
    experiments.run.allocate_run_directory(first)
    with pytest.raises(FileExistsError):
        experiments.run.allocate_run_directory(first)


def test_run_status_transitions_are_atomic_and_explicit(tmp_path: Path) -> None:
    """
    Drive one run through initialization, interruption, resume, and completion.

    Atomic history must retain every state in order and reject a terminal transition,
    preserving an auditable lifecycle rather than only the latest label.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / "run")
    experiments.run.transition_run_status(run_dir, "initializing")
    experiments.run.transition_run_status(run_dir, "running")
    experiments.run.transition_run_status(run_dir, "interrupted")
    experiments.run.transition_run_status(run_dir, "running")
    summary = experiments.run.transition_run_status(run_dir, "completed")

    assert summary["status"] == "completed"
    assert [entry["status"] for entry in summary["status_history"]] == [
        "initializing",
        "running",
        "interrupted",
        "running",
        "completed",
    ]
    with pytest.raises(experiments.run.RunLifecycleError, match="Invalid run status transition"):
        experiments.run.transition_run_status(run_dir, "failed")


def test_runtime_sessions_append_requested_and_resolved_facts_without_rewriting_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Start with ``auto`` resolved to CPU, interrupt, then resume under explicit CPU.

    The second session must append requested/resolved facts without rewriting the
    first, preserving runtime history across portable resume.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / "run-sessions")
    experiments.run.transition_run_status(run_dir, "initializing")
    monkeypatch.setattr(experiments.run.learning.device.torch.cuda, "is_available", lambda: False)
    first_resolution = experiments.run.learning.device.resolve_device("auto")
    first_started = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            first_resolution,
            started_at=first_started,
        ),
    )
    first_session = experiments.run.read_run_summary(run_dir)["runtime_sessions"][0].copy()

    experiments.run.transition_run_status(run_dir, "interrupted")
    second_resolution = experiments.run.learning.device.resolve_device("cpu")
    second_started = datetime(2025, 1, 2, 5, 6, tzinfo=UTC)
    summary = experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            second_resolution,
            started_at=second_started,
        ),
    )

    assert summary["runtime_sessions"][0] == first_session
    assert summary["runtime_sessions"][0]["requested_policy"] == "auto"
    assert summary["runtime_sessions"][0]["resolved_device"] == "cpu"
    assert summary["runtime_sessions"][1]["requested_policy"] == "cpu"
    assert summary["runtime_sessions"][1]["resolved_device"] == "cpu"
    assert summary["runtime_sessions"][1]["started_at"] == second_started.isoformat()
    assert summary["runtime_device"] == second_resolution.as_dict()


def test_a_failure_before_the_runtime_session_exists_keeps_its_own_error(tmp_path: Path) -> None:
    """Recording that tracking never started must not replace the run's own failure when no session exists yet."""
    run_dir = experiments.run.allocate_run_directory(tmp_path / "early-failure")
    experiments.run.transition_run_status(run_dir, "initializing")
    error = RuntimeError("dataset admission failed")

    experiments.run._record_tracking_never_started(run_dir, "session-0", error, "Local run admission failed before tracking initialization.")

    assert str(error) == "dataset admission failed"
    assert any("could not record this failure" in note for note in getattr(error, "__notes__", ()))


def test_runtime_session_records_validated_slurm_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach worker evidence to this session without altering run identity."""
    run_dir = experiments.run.allocate_run_directory(tmp_path / "slurm-session")
    experiments.run.transition_run_status(run_dir, "initializing")
    provenance = {
        "job_id": "12345",
        "partition": "gpu",
        "cpus_per_task": 4,
        "requested_memory": "32G",
        "requested_wall_time": "04:00:00",
        "allocated_mem_per_node": "32768",
        "mode": "gpu",
        "gres": "gpu:rtx6000ada:1",
        "source_commit": "a" * 40,
        "source_worktree_sha256": "b" * 64,
        "source_transition": "explicit",
        "config_sha256": "c" * 64,
        "sif_path": "/workspace/runtime/containers/grainlegumes-pino-drying.sif",
        "sif_sha256": "d" * 64,
    }
    monkeypatch.setenv("ML_SLURM_PROVENANCE", json.dumps(provenance))
    resolution = experiments.run.learning.device.resolve_device("cpu")

    summary = experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC)),
    )

    assert summary["runtime_sessions"][0]["slurm"] == provenance
    assert "slurm" not in summary["runtime_device"]
    provenance["source_worktree_sha256"] = "invalid"
    monkeypatch.setenv("ML_SLURM_PROVENANCE", json.dumps(provenance))
    with pytest.raises(experiments.run.RunLifecycleError, match="source_worktree_sha256"):
        experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC))


@pytest.mark.parametrize(
    ("transition", "admitted"),
    [("explicit", True), ("none", True), ("implicit", False), ("", False), (None, False), (["none"], False)],
)
def test_slurm_provenance_admits_exactly_the_two_source_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: object,
    admitted: bool,
) -> None:
    """A session records whether it deliberately continued a unit with newer code; any other value is refused."""
    run_dir = experiments.run.allocate_run_directory(tmp_path / "transition-session")
    experiments.run.transition_run_status(run_dir, "initializing")
    provenance = {
        "job_id": "12345",
        "partition": "standard",
        "cpus_per_task": 2,
        "requested_memory": "4G",
        "requested_wall_time": "00:10:00",
        "allocated_mem_per_node": "4096",
        "mode": "cpu",
        "gres": "none",
        "source_commit": "a" * 40,
        "source_worktree_sha256": "b" * 64,
        "source_transition": transition,
        "config_sha256": "c" * 64,
        "sif_path": "/workspace/runtime/containers/grainlegumes-pino-drying.sif",
        "sif_sha256": "d" * 64,
    }
    monkeypatch.setenv("ML_SLURM_PROVENANCE", json.dumps(provenance))
    resolution = experiments.run.learning.device.resolve_device("cpu")

    if admitted:
        updates = experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC))
        assert updates["runtime_sessions"][-1]["slurm"]["source_transition"] == transition
    else:
        with pytest.raises(experiments.run.RunLifecycleError, match="source_transition"):
            experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC))


def test_tracking_runtime_updates_are_atomic_and_session_scoped(
    tmp_path: Path,
) -> None:
    """
    Update tracking state inside one running session, then append a resume session.

    Atomic session mutation must preserve status history and earlier tracking facts,
    keeping observer state scoped to the runtime session that produced it.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / "tracking-sessions")
    experiments.run.transition_run_status(run_dir, "initializing")
    resolution = experiments.run.learning.device.resolve_device("cpu")
    first_started = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            resolution,
            started_at=first_started,
            session_id="first-session",
            tracking_state={
                "requested_mode": "online",
                "status": "active",
            },
        ),
    )
    updated = experiments.run.update_runtime_session(
        run_dir,
        "first-session",
        {
            "wandb_run_id": "opaque-run-id",
            "last_logged_epoch": 2,
            "status": "failed",
            "failed_operation": "history",
        },
    )
    first_session = copy.deepcopy(updated["runtime_sessions"][0])
    history = copy.deepcopy(updated["status_history"])

    appended = experiments.run.append_runtime_session(
        run_dir,
        resolution,
        started_at=datetime(2025, 1, 2, 5, 6, tzinfo=UTC),
        session_id="resume-session",
        tracking_state={
            "requested_mode": "online",
            "wandb_run_id": "opaque-run-id",
            "session_kind": "resume",
            "status": "active",
        },
    )

    assert appended["status"] == "running"
    assert appended["status_history"] == history
    assert appended["runtime_sessions"][0] == first_session
    assert first_session["tracking"]["requested_mode"] == "online"
    assert first_session["tracking"]["wandb_run_id"] == "opaque-run-id"
    assert first_session["tracking"]["status"] == "failed"
    assert first_session["tracking"]["failed_operation"] == "history"
    assert appended["runtime_sessions"][1]["tracking"]["session_kind"] == "resume"


def _resume_config(*, epochs: int = 4) -> dict[str, Any]:
    """Return fixed semantic sections used by resume identity tests."""
    raw = configs.direct_config()
    raw["training"]["epochs"] = epochs
    return configs.resolve(raw)


def test_resume_allows_only_duration_increase_and_runtime_location_changes() -> None:
    """
    Increase terminal epochs while changing only device and output-root location.

    Resume must accept retained/increased duration and operational relocation but
    reject shortening, protecting completed progress from rollback.
    """
    saved = _resume_config(epochs=4)
    requested = _resume_config(epochs=7)
    requested["run"]["device"] = "cuda"
    requested["paths"]["output_root"] = "/new/output"

    assert experiments.run.validate_resume_config(requested, saved) == 7
    assert experiments.run.validate_resume_config(saved, saved) == 4
    with pytest.raises(ValueError, match="only retain or increase"):
        experiments.run.validate_resume_config(_resume_config(epochs=3), saved)


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        ("data", {"train_dataset": "other", "additional_evaluation_datasets": ["ood"]}),
        ("model", {"kind": "tiny", "params": {"width": 8}}),
    ],
)
def test_resume_rejects_semantic_changes(section: str, replacement: Any) -> None:
    """
    Replace each scientific continuation section while increasing allowed duration.

    Every parametrized section must fail resume compatibility, proving operational
    allowances cannot conceal task, data, model, loss, or optimization drift.
    """
    saved = _resume_config()
    requested = _resume_config(epochs=6)
    if section == "data":
        requested["data"].update(replacement)
    else:
        requested[section] = replacement

    with pytest.raises(ValueError, match="incompatible with the saved run"):
        experiments.run.validate_resume_config(requested, saved)


def _hold_run_writer_lease(
    run_dir: Path,
    acquired: Any,
    release: Any,
    outcomes: Any,
) -> None:
    """
    Hold one run-writer lease while the parent probes resume admission.

    Events make lease acquisition and release deterministic. The outcome queue
    reports timeout, release, or an unexpected worker exception.
    """
    try:
        with experiments.run.run_writer_lease(run_dir):
            acquired.set()
            if not release.wait(timeout=10):
                outcomes.put("timed out")
                return
        outcomes.put("released")
    except Exception as error:  # noqa: BLE001 -- thread lease helper must report arbitrary acquisition failures to its test owner
        outcomes.put(f"{type(error).__name__}: {error}")


def test_file_lock_blocks_another_thread_in_the_same_process(tmp_path: Path) -> None:
    """
    Contend for one nonblocking file lock from a sibling thread in the same process.

    The second descriptor must be blocked, protecting process-local re-entrancy from
    being mistaken for ownership by arbitrary threads.
    """
    lock_path = tmp_path / "thread.lock"
    outcomes: queue.Queue[str] = queue.Queue()

    def contend() -> None:
        """Attempt sibling-thread acquisition and report its boundary result."""
        try:
            with common.locking.exclusive_file_lock(lock_path, blocking=False):
                outcomes.put("acquired")
        except common.locking.FileLockUnavailableError:
            outcomes.put("blocked")

    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        contender = threading.Thread(target=contend)
        contender.start()
        contender.join(timeout=5)
        assert not contender.is_alive()

    assert outcomes.get_nowait() == "blocked"


def test_shared_file_lock_admits_readers_and_refuses_an_active_writer(tmp_path: Path) -> None:
    """Share one lock file between sibling-thread readers; a fail-fast reader is blocked by a writer."""
    lock_path = tmp_path / "shared.lock"
    outcomes: queue.Queue[str] = queue.Queue()

    def read() -> None:
        """Attempt sibling-thread shared acquisition and report its boundary result."""
        try:
            with common.locking.shared_file_lock(lock_path, blocking=False):
                outcomes.put("read")
        except common.locking.FileLockUnavailableError:
            outcomes.put("blocked")

    for hold in (common.locking.shared_file_lock, common.locking.exclusive_file_lock):
        with hold(lock_path, blocking=False):
            reader = threading.Thread(target=read)
            reader.start()
            reader.join(timeout=5)

    assert [outcomes.get_nowait(), outcomes.get_nowait()] == ["read", "blocked"]


@pytest.mark.integration
def test_start_and_saved_resume_reject_a_second_process_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Hold a run-writer lease in one process while a second process attempts to continue the run.

    Both the rerun of the authored config and the saved-config resume must fail on
    the active writer before any run state is read for continuation or mutated.
    """
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("POSIX fork context is required for process lease coverage")
    context = mp.get_context("fork")
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    output_root = tmp_path / "outputs"
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    acquired = context.Event()
    release = context.Event()
    outcomes = context.Queue()
    holder = context.Process(
        target=_hold_run_writer_lease,
        args=(run_dir, acquired, release, outcomes),
    )
    holder.start()
    try:
        assert acquired.wait(timeout=5)
        with pytest.raises(experiments.run.ExistingRunAdmissionError, match="Wait until it finishes") as blocked:
            experiments.run.run_experiment(config_path, output_root=output_root)
        assert blocked.value.report["resume_compatibility"] == "blocked"
        with pytest.raises(experiments.run.RunLifecycleError, match="active writer lease"):
            experiments.run.run_experiment(common.paths.resolve_run_config_path(run_dir), resume=run_dir)
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)

    assert holder.exitcode == 0
    assert outcomes.get(timeout=5) == "released"
    lock_path = common.paths.resolve_run_lock_path(run_dir)
    assert lock_path.parent == storage_root / "03_experiments/.state/runs/locks"
    assert lock_path.is_file()
    assert lock_path.stat().st_size == 0
    assert not list(run_dir.rglob("*.lock"))
    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("renamed", [False, True])
def test_saved_effective_resume_service_preserves_exact_science_and_only_extends_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, renamed: bool
) -> None:
    """Saved config admission delegates the exact resolved request without a raw-schema fallback."""
    config = configs.resolve(configs.direct_config(), output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(config)
    source = configs.write_yaml(run_dir / "config.yaml", config)
    if renamed:
        moved = tmp_path / "renamed_direct_bundle"
        run_dir.rename(moved)
        run_dir = moved
        source = moved / "config.yaml"
    calls: dict[str, Any] = {}

    def execute(requested: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls["requested"] = requested
        calls.update(kwargs)
        return {"result": {"status": "completed"}, "run_dir": run_dir}

    monkeypatch.setattr(experiments.run, "_run_resolved_experiment", execute)
    original = source.read_bytes()
    cap = int(config["training"]["epochs"]) + 1
    experiments.run.run_experiment(source, resume=run_dir, max_epochs=cap)
    requested = calls["requested"]
    assert requested["training"]["epochs"] == cap
    assert requested["model"] == config["model"]
    assert requested["data"] == config["data"]
    assert requested["run"]["layout"] == config["run"]["layout"]
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="cannot override scientific"):
        experiments.run.run_experiment(source, resume=run_dir, train_dataset="different")
    with pytest.raises(ValueError, match="retain or increase"):
        experiments.run.run_experiment(source, resume=run_dir, max_epochs=1)


def test_a0_epoch_budget_cannot_be_moved_by_an_operational_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A0's epochs are its scientific budget: resume keeps them, and --max-epochs never extends or reselects them."""
    config = configs.resolve(configs.transient_stage_a_only_config(), output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(config)
    source = configs.write_yaml(run_dir / "config.yaml", config)
    calls: list[dict[str, Any]] = []

    def execute(requested: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append(requested)
        return {"result": {"status": "completed"}, "run_dir": run_dir}

    monkeypatch.setattr(experiments.run, "_run_resolved_experiment", execute)
    saved_epochs = int(config["training"]["epochs"])

    experiments.run.run_experiment(source, resume=run_dir, max_epochs=saved_epochs)
    with pytest.raises(ValueError, match="scientific budget"):
        experiments.run.run_experiment(source, resume=run_dir, max_epochs=saved_epochs + 2)
    with pytest.raises(ValueError, match="cannot override scientific"):
        experiments.run.run_experiment(source, resume=run_dir, lr_scale=0.5)
    plan_path = configs.write_config(tmp_path, configs.transient_matched_config())
    with pytest.raises(ValueError, match="scientific budget"):
        experiments.run.run_experiment(plan_path, output_root=tmp_path / "fresh", train_stages=("a0",), max_epochs=9)

    assert [requested["training"]["epochs"] for requested in calls] == [saved_epochs]
    assert calls[0]["training"]["stage_schedule"]["total_epochs"] == saved_epochs


def test_optuna_trial_directory_is_not_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reopening the study restarts interrupted trials as fresh trials, so --resume refuses a trial before any admission."""
    raw = configs.direct_config()
    config = config_loader.resolve_config(
        raw, source=config_naming.ConfigSource("steady_flow", common.paths.OPTUNA_FAMILY, "study"), output_root=tmp_path
    )
    run_dir = common.paths.resolve_config_run_dir(config)
    source = configs.write_yaml(run_dir / "config.yaml", config)

    def execute(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("a trial must be refused before any resume admission or execution")

    monkeypatch.setattr(experiments.run, "_run_resolved_experiment", execute)

    with pytest.raises(experiments.run.RunLifecycleError, match="trials are not resumed"):
        experiments.run.run_experiment(source, resume=run_dir)


def test_job_log_owner_is_recorded_only_inside_a_job_and_a_failed_write_only_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Without JOB_LOG_OWNERS_FILE nothing is written; with it every call appends one resolved line; a failed write warns."""
    owner = tmp_path / "run"
    owners_file = tmp_path / "owners"
    monkeypatch.delenv("JOB_LOG_OWNERS_FILE", raising=False)
    common.paths.record_job_log_owner(owner)
    assert not tuple(tmp_path.iterdir())

    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(owners_file))
    common.paths.record_job_log_owner(owner)
    common.paths.record_job_log_owner(owner / ".." / "run")
    assert owners_file.read_text(encoding="utf-8").splitlines() == [str(owner.resolve())] * 2
    assert capfd.readouterr().err == ""

    common.paths.record_job_log_owner(tmp_path / "undecodable-\udcff")
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(tmp_path / "missing" / "owners"))
    common.paths.record_job_log_owner(owner)
    assert capfd.readouterr().err.count("Warning:") == 2


@pytest.mark.parametrize(("start", "recorded"), [("resume", 2), ("trial", 0)])
def test_a_run_records_its_folder_as_job_log_owner_even_when_its_device_check_fails_but_a_trial_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str, recorded: int
) -> None:
    """Fresh and resumed executions each record their run folder before any check can fail them; a trial belongs to its study."""
    owners_file = tmp_path / "owners"
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(owners_file))
    config = _minimal_config(tmp_path / "outputs")
    if start == "trial":
        source = config_naming.ConfigSource("steady_flow", common.paths.OPTUNA_FAMILY, "study")
        config = config_loader.resolve_config(configs.direct_config(), source=source, output_root=tmp_path / "outputs")
        assert experiments.run.is_trial_config(config)
    run_dir = experiments.run.prepare_fresh_run(config)

    def interrupt(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(experiments.run, "validate_deterministic_model_device_policy", interrupt)
    resolution = experiments.run.learning.device.resolve_device("cpu")
    for resume_from in (None, run_dir / "last_checkpoint.pt")[: 2 if start == "resume" else 1]:
        with pytest.raises(KeyboardInterrupt):
            experiments.run.execute_prepared_run(config, run_dir=run_dir, resume_from=resume_from, device_resolution=resolution)

    lines = owners_file.read_text(encoding="utf-8").splitlines() if owners_file.exists() else []
    assert lines == [str(run_dir.resolve())] * recorded
