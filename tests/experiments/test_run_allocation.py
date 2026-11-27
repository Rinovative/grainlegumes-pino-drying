"""Protect collision-safe run allocation, writer leases, resume, and status transitions.

Responsibilities:
  - Check collision refusal, writer leases, atomic status, runtime sessions, and resume admission
  - Check that steady and transient inference refuse a run while a writer holds its lease
  - Check admission of a start that names an existing run: continue, report finished, or reject; a finished run, also an
    unsuccessful one, is reported through the finished-run admission and exits 2 without an eligible checkpoint
  - Check that a run recorded by other training code is neither reported complete, extended nor resumed
  - Check that a resume or rerun refused by its admission, history, device or another directory's identity binding
    creates no lock file or binding, and that an unsuccessful run is admitted against its summary's records and identity
  - Check that the train command reports a continuation refused by another directory's identity binding as a rejected admission,
    and another refused continuation, a damaged saved config or summary, an irregular writer lock marker or a completed run's
    malformed identity entry included, in one line without a traceback or a rerun
  - Check that an environment failure before a continuation's attempt, Slurm provenance read before the run's state included,
    prints its rerun in a matching allocation without a traceback, as does an I/O error inspecting an existing run or binding
    a completed run's identity, and that a completed run's rerun reports it complete only once nothing can end it
  - Check that an attempt the job's environment ends after its start, by a failed checkpoint write or by host memory
    exhausted while its best decodes, ends interrupted with its reason and continuing rerun, and no traceback
  - Check that a run that ended before its first checkpoint, staged checkpoint leftovers included, is replaced by its
    start command and trained to its end with its identity bound to the leaf, while an entry outside its layout, a
    symbolic link in its layout or config, other science or training code, a malformed status or another writer keeps
    its refusal and every file, and ``--resume`` refuses such a leaf naming its start command before its source binding
  - Check that an interrupted run is promised a restart only when its start replaces its leaf, and that a replacement
    the job's environment ends midway reports that the leaf may be partly replaced
  - Check that a fresh run, but no Optuna trial and no resume refused before its attempt, records its folder as a job
    log owner, and that a continuation ended before its attempt names its reason and a rerun only after its environment
  - Check that a continuation whose pinned Dataset content lacks a key, never claimed as a malformed manifest, or whose
    Dataset folder or manifest is a symbolic link, is refused naming it, never rerun, and that an absent manifest ends it
    with the rerun, also before a later Dataset that is a symbolic link

Design principles:
  - Scientific config stays immutable while permitted duration extensions are explicit

This module does NOT:
  - Restore checkpoint tensors or train models
"""

from __future__ import annotations

import contextlib
import copy
import errno
import functools
import json
import multiprocessing as mp
import queue
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import pytest
import torch
from support import configs, run_state
from torch.utils.data import DataLoader, TensorDataset

from src import common, datasets, experiments, learning
from src.experiments import experiments_run_source_binding as source_binding
from src.experiments.cli import cli_train
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
        completed_epoch=(config["training"]["budget_epochs"] if completed_epoch is None else completed_epoch),
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
    assert report["completed_epoch"] == config["training"]["budget_epochs"]


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
        completed_epoch=config["training"]["budget_epochs"],
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
    # A marker that is no regular file is the run's stored state, which no rerun repairs: the run cannot continue.
    report = experiments.run.inspect_existing_run_admission(run_dir, config, config_path=config_path)
    assert report["active_lock"] is None
    assert report["resume_compatibility"] == "incompatible"
    assert "no regular file" in report["reason"]
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


def test_training_code_drift_refuses_reuse_and_resume_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    A run recorded by older training code is neither reported complete nor resumed; nothing is written.

    The refused resume is an expected outcome, so the train command reports it in one line without a traceback.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    with monkeypatch.context() as drift:
        run_state.drift_training_code(drift, "steady_flow")
        stale = _resolved_admission_config(output_root)
    epochs = int(stale["training"]["budget_epochs"])
    run_dir = common.paths.resolve_config_run_dir(stale, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, stale, status="completed", completed_epoch=epochs)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("stale code must not train"))
    saved_config = common.paths.resolve_run_config_path(run_dir)
    changed = r"training_steady_flow implementation changed \(recorded [0-9a-f]{12}, current [0-9a-f]{12}\); finish it in its bound source tree"
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}

    with pytest.raises(experiments.run.ExistingRunAdmissionError, match=changed):
        experiments.run.run_experiment(config_path, output_root=output_root)
    run_state_summary = common.paths.resolve_run_summary_path(run_dir)
    common.serialization.atomic_write_json(
        run_state_summary, {**json.loads(run_state_summary.read_text(encoding="utf-8")), "status": "interrupted", "completed_epoch": 1}
    )
    before[run_state_summary.relative_to(run_dir)] = run_state_summary.read_bytes()
    with pytest.raises(experiments.run.ContinuationRefusedError, match=changed):
        experiments.run.run_experiment(saved_config, resume=run_dir)
    capsys.readouterr()
    assert cli_train.main(["--resume", str(run_dir)]) == 1
    stderr = capsys.readouterr().err
    assert "training_steady_flow implementation changed" in stderr
    assert "Traceback" not in stderr

    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("damage", ["config", "summary", "lock_marker"])
def test_a_continuation_refused_by_a_damaged_state_file_prints_its_refusal_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    """
    A saved config that is no YAML, a summary that is no text or a writer lock marker that is no regular file refuses it.

    Each is the run's persisted state, which a rerun in another allocation meets again: ``--resume`` prints the refusal's own
    sanitized line, and a start that names the run the existing-run rejection, both without a traceback or a rerun; the run
    stays byte-identical.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    if damage == "config":
        common.paths.resolve_run_config_path(run_dir).write_text("run: [unclosed\n", encoding="utf-8")
    elif damage == "summary":
        common.paths.resolve_run_summary_path(run_dir).write_bytes(b"\xff\xfe")
    else:
        lock_path = common.paths.resolve_run_lock_path(run_dir)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.symlink_to(tmp_path / "missing-writer-marker.lock")
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("a refused run must not train"))
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    capsys.readouterr()

    assert cli_train.main(["--resume", str(run_dir)]) == 1
    refused = capsys.readouterr().err
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    assert cli_train.main([str(config_path)]) == 1
    rejected = capsys.readouterr().err

    assert refused.startswith("Run continuation refused: The continuation was refused before its attempt started")
    assert rejected.startswith("Run admission rejected: The existing run at")
    assert "Traceback" not in refused + rejected
    assert "matching allocation" not in refused + rejected
    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("environment", ["active_writer", "slurm_provenance"])
def test_an_environment_failure_before_a_continuation_started_prints_the_rerun_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], environment: str
) -> None:
    """
    Another writer, or Slurm provenance that is no worker evidence, fails the job's environment before any attempt reported.

    The command names the failure and the rerun in a matching allocation in two lines without a traceback, and the run
    stays byte-identical. The provenance is read before any check of the run's state, so it is the reported end although
    that state, a failed run, refuses the continuation as well.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    status = "failed" if environment == "slurm_provenance" else "interrupted"
    run_state.write_lightweight_run_state(run_dir, config, status=status, completed_epoch=1)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("an unadmitted run must not train"))
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    capsys.readouterr()

    if environment == "slurm_provenance":
        monkeypatch.setenv("ML_SLURM_PROVENANCE", "not json")
        assert cli_train.main(["--resume", str(run_dir)]) == 1
    else:
        with experiments.run.run_writer_lease(run_dir):
            assert cli_train.main(["--resume", str(run_dir)]) == 1

    reason, step = capsys.readouterr().err.splitlines()
    assert reason.startswith("Run not started: RunLifecycleError: ")
    assert ("must contain valid JSON" if environment == "slurm_provenance" else "active writer lease") in reason
    assert step == "  next: rerun the same command in a matching allocation, the run is unchanged"
    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


class _ScaledMSELoss(torch.nn.MSELoss):
    """A training loss that accepts the run's Train scaling, as every training loss of the run does."""

    def set_scaling(self, _scaling: object) -> None:
        """Accept the scaling the attempt hands over."""


def _stub_attempt_runtime(monkeypatch: pytest.MonkeyPatch, *, train_loop: Callable[..., Any], evaluate_selected: Callable[..., Any]) -> None:
    """Replace a fresh single run's data, model, loss, metrics, training loop and selected evaluation; its lifecycle and report stay real."""
    scaling = SimpleNamespace(physical_state=SimpleNamespace(scale=torch.ones(1)), to=lambda _device: None)
    sampler = datasets.runtime.units.UnitSampler(unit_kind="case", cases=(0,), keys=None, units_per_update=1, memory_batch=1, seed=0)
    train = DataLoader(TensorDataset(torch.zeros(1)), batch_sampler=sampler)
    loaders = {"train": train, "eval": [0], "scaling": scaling, "scaling_artifact": {}, "split_indices": {}}
    run_learning = experiments.run.learning

    def create_dataloaders(*_args: Any, split_persist_callback: Callable[[dict[str, Any]], None] | None = None, **_kwargs: Any) -> dict[str, Any]:
        if split_persist_callback is not None:
            split_persist_callback({})  # a fresh run publishes its membership
        return loaders

    for target, name, stub in (
        (experiments.run, "_grid_spacing", lambda *_args: (0.003, 0.003)),
        (config_loader, "create_dataloaders_from_config", create_dataloaders),
        (run_learning.models.factory, "build_model", lambda *_args, **_kwargs: torch.nn.Linear(1, 1)),
        (run_learning.losses.factory, "build_training_loss", lambda *_args, **_kwargs: _ScaledMSELoss()),
        (run_learning.metrics.metrics, "build_evaluation_metrics", lambda *_args, **_kwargs: {}),
        (run_learning.training.optim, "build_scheduler", lambda *_args, **_kwargs: None),
        (run_learning.training.checkpoint, "build_checkpoint_identity", lambda *_args, **_kwargs: {"effective_config_digest": "a" * 64}),
        (experiments.run.tracking, "build_monitor_membership", lambda *_args: None),
        (run_learning.training.loop, "train_loop", train_loop),
        (run_learning.training.loop, "evaluate_selected_checkpoint", evaluate_selected),
    ):
        monkeypatch.setattr(target, name, stub)


@pytest.mark.parametrize("end", ["checkpoint_write", "best_decode_memory"])
def test_an_attempt_the_jobs_environment_interrupted_reports_its_rerun_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], end: str
) -> None:
    """
    A started attempt that the job's environment ends after its last checkpoint ends interrupted, reported by the run alone.

    The run's own checkpoint write failing, or host memory exhausted while the selected best decodes after the terminal
    last (a ``MemoryError``), ends it so. The run's report names the reason and the rerun that continues from the intact
    last checkpoint, and the command adds no traceback.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)

    def trained(**kwargs: Any) -> dict[str, Any]:
        (kwargs["save_dir"] / "last_checkpoint.pt").touch()  # the last checkpoint of the attempt's first event
        if end == "checkpoint_write":
            # The next publication fails in the training loop, which annotates its phase.
            with learning.training.loop._training_phase("last_checkpoint", 1):
                raise OSError(errno.ENOSPC, "No space left on device")
        objective = config_loader.get_resolved_objective(config)
        best = str(kwargs["save_dir"] / "best_checkpoint.pt")
        return {"status": "completed", "best_epoch": 1, "best_metric": 0.1, "objective": objective, "best_checkpoint_path": best}

    def exhausted(**_kwargs: Any) -> dict[str, Any]:
        message = "DefaultCPUAllocator: can't allocate memory"
        raise MemoryError(message)

    _stub_attempt_runtime(monkeypatch, train_loop=trained, evaluate_selected=exhausted)
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    capsys.readouterr()
    assert cli_train.main([str(config_path)]) == 1

    stderr = capsys.readouterr().err
    stop = next(line for line in stderr.splitlines() if line.startswith("interrupted "))
    step = next(line for line in stderr.splitlines() if line.startswith("  next: "))
    reason = "OSError: [Errno 28] No space left on device" if end == "checkpoint_write" else "MemoryError: DefaultCPUAllocator"
    assert stop.startswith(f"interrupted {config['run']['name']} ")
    assert reason in stop
    assert step.startswith("  next: rerun the same command to continue from the last checkpoint")
    assert "Traceback" not in stderr
    assert experiments.run.read_run_summary(run_dir)["status"] == "interrupted"


@pytest.mark.parametrize("stale", ["interrupted", "running"])
def test_a_run_that_ended_before_its_first_checkpoint_is_replaced_by_its_start_and_trains_to_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stale: str
) -> None:
    """
    An attempt interrupted after a history record but before its first checkpoint committed no work: ``train CONFIG`` replaces it.

    Its first checkpoint publication was cut off before its renames, which leaves a staged checkpoint and a writer's
    temporary of another, no committed work. Its report advises the rerun that restarts it. The rerun, also of a summary
    a killed job left ``running``, names the replacement and starts the leaf afresh: the old history record, the staged
    leftovers and the attempt are gone, and the identity stays bound to the leaf.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    history = learning.training.history.history_path(run_dir)
    leftovers = [run_dir / ".best_checkpoint.pt.staged", run_dir / "..last_checkpoint.pt.staged.k1ll3d.tmp"]

    def interrupted(**kwargs: Any) -> dict[str, Any]:
        kwargs["epoch_state_callback"](1, {"global_step": 1.0, "train/loss_total": 0.5}, None)
        for leftover in leftovers:
            leftover.touch()
        raise KeyboardInterrupt

    def trained(**kwargs: Any) -> dict[str, Any]:
        assert not history.exists()  # the earlier attempt's record is gone
        assert not any(leftover.exists() for leftover in leftovers)
        kwargs["epoch_state_callback"](1, {"global_step": 2.0, "train/loss_total": 0.25}, None)
        for name in ("best_checkpoint.pt", "last_checkpoint.pt"):
            (kwargs["save_dir"] / name).touch()
        budget = {"unit": "units", "units": 1, "total": 1, "train_seconds": 0.0, "skipped_points": []}
        outcome = {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None}
        return {
            "status": "completed", "best_epoch": 1, "best_metric": 0.1, "objective": config_loader.get_resolved_objective(config),
            "best_checkpoint_path": str(kwargs["save_dir"] / "best_checkpoint.pt"), "completed_epoch": 1, "global_step": 2,
            "training_work": {}, "terminal_epoch": 1, "training_outcome": outcome, "budget": budget,
        }  # fmt: skip

    selected = {"selected_epoch": 1, "selected_metrics": {}, "terminal_metrics": {}}
    _stub_attempt_runtime(monkeypatch, train_loop=interrupted, evaluate_selected=lambda **_kwargs: selected)
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    assert cli_train.main([str(config_path)]) == 130
    assert "  next: rerun the same command, which restarts the run: no checkpoint saved yet" in capsys.readouterr().err
    assert len(json.loads(history.read_text(encoding="utf-8"))["epochs"]) == 1
    summary = common.paths.resolve_run_summary_path(run_dir)
    common.serialization.atomic_write_json(summary, {**json.loads(summary.read_text(encoding="utf-8")), "status": stale})
    binding = common.paths.resolve_identity_registry_entry(output_root, config_loader.allocation_digest(config))
    bound = binding.read_bytes()
    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert entry.target == run_dir.resolve()

    monkeypatch.setattr(learning.training.loop, "train_loop", trained)
    assert cli_train.main([str(config_path)]) == 0

    assert f"run replaced: previous_status {stale}  run_dir " in capsys.readouterr().out
    replaced = experiments.run.read_run_summary(run_dir)
    assert [entry["status"] for entry in replaced["status_history"]] == ["initializing", "running", "completed"]
    assert len(replaced["runtime_sessions"]) == 1
    assert [record["global_step"] for record in json.loads(history.read_text(encoding="utf-8"))["epochs"]] == [2.0]
    assert binding.read_bytes() == bound


def _leaf_without_a_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any], Path]:
    """Return the request, its resolved config and its interrupted leaf that holds every run file but a checkpoint."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=0, include_best=False)
    common.paths.resolve_last_checkpoint_file(run_dir).unlink()
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("a refused run must not train"))
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    return config_path, config, run_dir


_REFUSED = "Run continuation refused: The continuation was refused before its attempt started"
_CONFLICT = "Run admission rejected: Scientific identity conflict"


@pytest.mark.parametrize(
    ("refusal", "expected", "reason"),
    [
        ("unknown_entry", _REFUSED, "'notes.txt', which is no file of its run"),
        ("handoff_manifest_of_a_steady_run", _REFUSED, "'teacher_handoff_manifest.json', which is no file of its run"),
        ("symlinked_layout", _REFUSED, "is no real folder"),
        ("symlinked_config", _REFUSED, "config.yaml is no regular file"),
        ("other_science", _CONFLICT, "--revision"),
        ("training_drift", _CONFLICT, "--revision"),
        ("malformed_status", "Run admission rejected: The existing run at", "Run summary has no valid lifecycle status"),
        ("active_writer", "Run admission rejected: An active writer lease already owns this run", "Wait until it finishes"),
    ],
)
def test_a_leaf_without_a_checkpoint_that_its_start_cannot_replace_keeps_its_refusal_and_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], refusal: str, expected: str, reason: str
) -> None:
    """
    A leaf its start cannot replace keeps its refusal, and nothing below the test root changes, so no lock file or binding exists.

    The leaf's state refuses an entry outside the run's layout, a teacher-handoff manifest in a steady run included, and
    a layout folder or config that is a symbolic link, before anything is read through it, each in its refusal's line.
    Other science, also another training digest, keeps the ``--revision`` advice, a status that is no string the
    existing-run rejection and an active writer the advice to wait; none is advised a rerun in a matching allocation.
    """
    config_path, _config, run_dir = _leaf_without_a_checkpoint(tmp_path, monkeypatch)
    if refusal == "unknown_entry":
        (run_dir / "notes.txt").write_text("kept by hand", encoding="utf-8")
    elif refusal == "handoff_manifest_of_a_steady_run":
        (run_dir / "teacher_handoff_manifest.json").write_text("{}", encoding="utf-8")  # only a transient handoff writes it
    elif refusal == "symlinked_layout":
        # The leaf's config folder is an alias of another folder below the output root.
        run_dir.parent.symlink_to(run_dir.parent.rename(tmp_path / "outputs" / "elsewhere"), target_is_directory=True)
    elif refusal == "symlinked_config":
        saved = common.paths.resolve_run_config_path(run_dir)
        saved.symlink_to(saved.rename(tmp_path / "saved-config.yaml"))
    elif refusal == "other_science":
        raw = configs.direct_config()
        raw["optimizer"]["lr"] = float(raw["optimizer"]["lr"]) * 1.0001  # the same three-digit name token
        config_path = configs.write_config(tmp_path, raw, canonical_root=tmp_path)
        assert common.paths.resolve_config_run_dir(configs.resolve(raw, output_root=tmp_path / "outputs")) == run_dir
    elif refusal == "training_drift":
        with monkeypatch.context() as drift:
            run_state.drift_training_code(drift, "steady_flow")
            stale = _resolved_admission_config(tmp_path / "outputs")
        config_loader.save_yaml(stale, common.paths.resolve_run_config_path(run_dir))
    elif refusal == "malformed_status":
        summary = common.paths.resolve_run_summary_path(run_dir)
        common.serialization.atomic_write_json(summary, {**json.loads(summary.read_text(encoding="utf-8")), "status": []})
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    capsys.readouterr()

    with experiments.run.run_writer_lease(run_dir) if refusal == "active_writer" else contextlib.nullcontext():
        assert cli_train.main([str(config_path)]) == 1
        after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    stderr = capsys.readouterr().err
    assert stderr.startswith(expected)
    assert reason in stderr
    assert "Traceback" not in stderr
    assert "matching allocation" not in stderr
    lock = common.paths.resolve_run_lock_path(run_dir).relative_to(tmp_path)
    assert {path: payload for path, payload in after.items() if path != lock} == before


@pytest.mark.parametrize("drifted", [False, True])
def test_a_resume_of_a_leaf_without_a_checkpoint_is_refused_naming_its_start_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], drifted: bool
) -> None:
    """
    ``--resume`` continues only from a last checkpoint: a leaf its start replaces is refused, naming that start, and stays unchanged.

    That reason precedes the source binding, so a leaf another source tree binds is not advised a continuation through
    Slurm, where ``--resume`` could not continue it either. After a training-code change no start replaces the leaf, so
    the refusal names the drift and its ``--revision`` remedy instead.
    """
    _config_path, _config, run_dir = _leaf_without_a_checkpoint(tmp_path, monkeypatch)
    if drifted:
        with monkeypatch.context() as drift:
            run_state.drift_training_code(drift, "steady_flow")
            stale = _resolved_admission_config(tmp_path / "outputs")
        config_loader.save_yaml(stale, common.paths.resolve_run_config_path(run_dir))

    def bound_to_another_tree(_leaves: Any) -> None:
        msg = f"Run {run_dir} is bound to another source tree; continue it through scripts/slurm_ml.sh."
        raise experiments.run.SourceBindingError(msg)

    monkeypatch.setattr(source_binding, "admit_continuation", bound_to_another_tree)
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    capsys.readouterr()

    assert cli_train.main(["--resume", str(run_dir)]) == 1

    stderr = capsys.readouterr().err
    assert stderr.startswith("Run continuation refused: The continuation was refused before its attempt started")
    assert ("ended before its first checkpoint" in stderr) is not drifted
    assert ("its start command (train CONFIG, or its plan for a matched stage) replaces it" in stderr) is not drifted
    assert ("--revision" in stderr) is drifted
    assert "matching allocation" not in stderr
    assert "slurm_ml.sh" not in stderr
    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before


def test_an_interrupted_run_without_a_checkpoint_is_promised_a_restart_only_when_its_start_replaces_its_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    The advice of an attempt interrupted before its first checkpoint follows the replacement admission of its published state.

    A file outside the run's layout keeps the start from replacing the leaf, so the report promises no restart, and the
    rerun it advises refuses the leaf naming that file.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)

    def interrupted(**kwargs: Any) -> dict[str, Any]:
        (kwargs["save_dir"] / "notes.txt").write_text("kept by hand", encoding="utf-8")
        raise KeyboardInterrupt

    _stub_attempt_runtime(monkeypatch, train_loop=interrupted, evaluate_selected=lambda **_kwargs: {})
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    assert cli_train.main([str(config_path)]) == 130
    step = next(line for line in capsys.readouterr().err.splitlines() if line.startswith("  next: "))
    assert "which restarts the run:" not in step
    assert "restarts the run only if its leaf admits that, otherwise names why" in step

    assert cli_train.main([str(config_path)]) == 1
    assert "'notes.txt', which is no file of its run" in capsys.readouterr().err


def test_a_replacement_the_jobs_environment_ends_midway_reports_a_leaf_that_may_be_partly_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    An I/O error after the replacement removed the earlier attempt's files, before its fresh summary, leaves the leaf partly replaced.

    The command says so, without a traceback, and advises the rerun, which replaces the leaf that is still replaceable.
    """
    config_path, config, run_dir = _leaf_without_a_checkpoint(tmp_path, monkeypatch)
    transition = experiments.run._transition_run_status_locked

    def unwritable_fresh_summary(*args: Any, anew: bool = False, **kwargs: Any) -> dict[str, Any]:
        if anew:
            raise OSError(errno.EIO, "Input/output error")
        return transition(*args, **kwargs)

    monkeypatch.setattr(experiments.run, "_transition_run_status_locked", unwritable_fresh_summary)
    capsys.readouterr()
    assert cli_train.main([str(config_path)]) == 1

    assert capsys.readouterr().err.splitlines() == [
        "Run not started: OSError: [Errno 5] Input/output error",
        "  next: rerun the same command in a matching allocation, which replaces the run; its leaf may be partly replaced already",
    ]
    assert not common.paths.resolve_split_indices_path(run_dir).exists()
    assert experiments.run.read_run_summary(run_dir)["status"] == "interrupted"
    assert experiments.run._replaceable_leaf(run_dir, config, matched_receipt=None) is not None


def test_an_io_error_inspecting_an_existing_run_is_the_jobs_environment_at_its_start_and_in_its_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    An I/O error while an existing run is inspected is the job's environment, never a reason the run cannot continue.

    A start that names the run prints the failure and the rerun in a matching allocation without a traceback, a matched
    stage's read-only recovery check offers that rerun, and the run stays byte-identical.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}

    def unreadable(*_args: Any, **_kwargs: Any) -> None:
        message = "Input/output error"
        raise OSError(errno.EIO, message)

    monkeypatch.setattr(experiments.run, "read_run_summary", unreadable)
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    capsys.readouterr()
    assert cli_train.main([str(config_path)]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "Run not started: OSError: [Errno 5] Input/output error",
        "  next: rerun the same command in a matching allocation, the run is unchanged",
    ]
    recovery, reason = experiments.run._matched_failed_stage_recovery(run_dir, config, prerequisites=lambda _finished: None)
    assert recovery == "environment"
    assert "OSError: [Errno 5] Input/output error" in reason
    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("end", ["completed", "diverged", "no_eligible_checkpoint"])
def test_rerunning_a_finished_run_returns_its_saved_outcome_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    end: str,
) -> None:
    """
    A finished run reports its saved end status, a divergence or no eligible checkpoint included; nothing is written or trained.

    Only its binding is repaired, and the command exits 2 for a run without an eligible checkpoint, as when it just ended so.
    The saved outcome is read through the finished-run admission of a matched leaf, stubbed here because the lightweight run
    holds no checkpoints or history (the steady lifecycle test admits a real one). An unsuccessful run that committed no
    optimizer group has no last checkpoint, which no finished run needs.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    status = "unsuccessful" if end == "no_eligible_checkpoint" else "completed"
    epochs = config["training"]["budget_epochs"]
    run_state.write_lightweight_run_state(run_dir, config, status=status, completed_epoch=epochs, include_best=status == "completed")
    if status == "unsuccessful":
        common.paths.resolve_last_checkpoint_file(run_dir).unlink()
    best = {"best_epoch": 1, "best_metric": 0.25} if status == "completed" else {"best_epoch": None, "best_metric": None}
    divergence = "loss is non-finite" if end == "diverged" else None
    eligible = int(status == "completed")
    training_outcome = {"status": end, "budget_progress": 0.5 if divergence else 1.0, "eligible_points": eligible, "divergence": divergence}
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"status": status, "completed_epoch": epochs, **best, "training_outcome": training_outcome},
    )
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    admitted: list[Path] = []

    def admit(path: Path) -> dict[str, Any]:
        admitted.append(path.resolve())
        return {"summary": experiments.run.read_run_summary(path)}

    def execute(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a complete run must not train again")

    monkeypatch.setattr(experiments.run, "validate_finished_run", admit)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", execute)

    outcome = experiments.run.run_experiment(config_path, output_root=output_root)

    assert admitted == [run_dir.resolve()]
    assert outcome["already_complete"] is True
    assert "already_complete" in capsys.readouterr().out
    assert outcome["run_dir"] == run_dir.resolve()
    assert outcome["result"] == {"status": end, "training_outcome": training_outcome, **best}
    registry_state = output_root.relative_to(tmp_path) / ".state" / "identities"
    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    assert {path: payload for path, payload in after.items() if not path.is_relative_to(registry_state)} == before
    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert entry.target == run_dir.resolve()
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    assert cli_train.main([str(config_path)]) == (2 if status == "unsuccessful" else 0)
    assert ("unsuccessful: no eligible checkpoint" in capsys.readouterr().out) is (status == "unsuccessful")


@pytest.mark.parametrize("status", ["completed", "unsuccessful", "interrupted"])
def test_a_rerun_refused_by_its_admission_changes_nothing_and_binds_no_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    """
    A rerun refused by its admission, the finished-run one or the resume one, is neither reported finished nor resumed.

    The lightweight run holds no valid checkpoint, split or history, so both refusals come before the identity binding is
    repaired and before the writer lease creates its lock file: nothing below the test root changes.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status=status, completed_epoch=1)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("a refused run must not train"))
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    with pytest.raises(experiments.run.RunLifecycleError):
        experiments.run.run_experiment(config_path, output_root=output_root)

    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before
    assert experiments.registry.lookup(config) is None


def test_a_completed_rerun_refused_by_its_device_leaves_a_stale_binding_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A completed run is admitted before any device setup, but its stale identity binding is repaired only after the device admission.

    The rerun reports the run complete only once nothing can end it any more, so a device it cannot use prints no such line.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="completed", completed_epoch=config["training"]["budget_epochs"])
    outcome = {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None}
    summary = {**experiments.run.read_run_summary(run_dir), "best_epoch": 1, "best_metric": 0.25, "training_outcome": outcome}
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    # The binding names a directory that no longer exists.
    gone = output_root / "gone"
    gone.mkdir()
    experiments.registry.ensure_bound(config, gone)
    gone.rmdir()
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda path: {"summary": experiments.run.read_run_summary(path)})

    def unusable(*_args: Any, **_kwargs: Any) -> None:
        msg = "CUDA is not usable"
        raise learning.device.DeviceResolutionError(msg)

    monkeypatch.setattr(experiments.run.learning.device, "resolve_device", unusable)
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    capsys.readouterr()

    with pytest.raises(learning.device.DeviceResolutionError):
        experiments.run.run_experiment(config_path, output_root=output_root)

    assert "already_complete" not in capsys.readouterr().out
    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("binding", ["malformed_entry", "io_error"])
def test_a_completed_rerun_whose_identity_binding_fails_ends_by_its_cause_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], binding: str
) -> None:
    """
    A completed run's rerun repairs its identity binding last: a malformed entry is stored state, an I/O error the job's environment.

    The malformed entry refuses the rerun in one line without a rerun, which would be refused again; an I/O error prints the
    rerun in a matching allocation. Neither prints a traceback or reports the run complete, and nothing below the test root
    changes but the registry's persistent lock anchor, lease infrastructure.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="completed", completed_epoch=config["training"]["budget_epochs"])
    outcome = {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None}
    summary = {**experiments.run.read_run_summary(run_dir), "best_epoch": 1, "best_metric": 0.25, "training_outcome": outcome}
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda path: {"summary": experiments.run.read_run_summary(path)})
    if binding == "malformed_entry":
        entry = common.paths.resolve_identity_registry_entry(output_root, config_loader.allocation_digest(config))
        entry.parent.mkdir(parents=True)
        entry.write_text("not json", encoding="utf-8")
    else:

        def unreadable(*_args: Any, **_kwargs: Any) -> None:
            message = "Input/output error"
            raise OSError(errno.EIO, message)

        monkeypatch.setattr(experiments.registry, "ensure_bound", unreadable)
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    locks = output_root / ".state" / "identities" / "locks"
    before = {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file() and not item.is_relative_to(locks)}
    capsys.readouterr()

    assert cli_train.main([str(config_path)]) == 1
    printed = capsys.readouterr()
    if binding == "malformed_entry":
        assert printed.err.startswith("Run continuation refused: The continuation was refused before its attempt started")
        assert "RuntimeError: Identity registry entry is not valid JSON" in printed.err
        assert "Traceback" not in printed.err
        assert "matching allocation" not in printed.err
    else:
        assert printed.err.splitlines() == [
            "Run not started: OSError: [Errno 5] Input/output error",
            "  next: rerun the same command in a matching allocation, the run is unchanged",
        ]
    assert "already_complete" not in printed.out
    assert {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file() and not item.is_relative_to(locks)} == before


def test_an_in_place_continuation_of_an_identity_another_run_owns_is_refused_before_the_writer_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A start that would continue an interrupted run whose identity another directory owns is refused read-only, without a lock file.

    It is a refused continuation caused by the identical experiment, which the train command reports as a rejected admission,
    as for a fresh start, never as a training traceback.
    """
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    owner = output_root / "owner"
    owner.mkdir()
    experiments.registry.ensure_bound(config, owner)
    # The lightweight run holds no restorable checkpoint, so its read-only resume admission admits it here.
    admitted = (config, config, {}, {}, common.paths.resolve_last_checkpoint_file(run_dir))
    monkeypatch.setattr(experiments.run, "_admit_resolved_resume", lambda *_args, **_kwargs: admitted)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("a refused run must not train"))
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    with pytest.raises(experiments.run.ContinuationRefusedError) as refused:
        experiments.run.run_experiment(config_path, output_root=output_root)
    assert isinstance(refused.value.__cause__, experiments.registry.IdenticalExperimentError)
    # The command, under the same output root.
    monkeypatch.setattr(experiments.run, "run_experiment", functools.partial(experiments.run.run_experiment, output_root=output_root))
    capsys.readouterr()
    assert cli_train.main([str(config_path)]) == 1
    stderr = capsys.readouterr().err
    assert "Run admission rejected: The identical experiment already exists at" in stderr
    assert "Traceback" not in stderr

    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before


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
        {"status": status},
    )

    assert common.paths.is_evaluable_run_dir(run_dir) is evaluable


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
            memory_batch=2,
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
            memory_batch=2,
        ),
    )

    assert summary["runtime_sessions"][0] == first_session
    assert summary["runtime_sessions"][0]["requested_policy"] == "auto"
    assert summary["runtime_sessions"][0]["resolved_device"] == "cpu"
    assert summary["runtime_sessions"][1]["requested_policy"] == "cpu"
    assert summary["runtime_sessions"][1]["resolved_device"] == "cpu"
    assert summary["runtime_sessions"][1]["started_at"] == second_started.isoformat()
    assert summary["runtime_device"] == second_resolution.as_dict()
    # A CPU attempt records no driver; its precision and its end entry stay null until its model exists and it ends.
    assert summary["runtime_sessions"][1]["cuda_driver_version"] is None
    assert summary["runtime_sessions"][1]["precision"] is None
    assert set(summary["runtime_sessions"][1]["attempt"].values()) == {None}
    assert (summary["elapsed_seconds"], summary["attempted_train_seconds"], summary["duration_coverage"]) == (None, None, "unknown_attempts")


@pytest.mark.parametrize(
    ("case", "refusal"),
    [
        ("stage_without_group", None),
        ("stage_without_group_with_a_record", "history"),
        ("stage_without_group_or_its_identity", "checkpoint identity"),
        ("stage_without_group_with_another_identity", "checkpoint identity"),
        ("stage_without_group_with_a_malformed_environment", "checkpoint identity is invalid"),
        ("stage_without_group_with_another_image", "runtime session"),
        ("stage_without_group_with_another_cuda_build", "runtime session"),
        ("stage_without_group_with_another_package_version", None),
        ("trial_with_groups", None),
        ("trial_history_beyond_its_step", "terminal global step"),
    ],
)
def test_an_unsuccessful_steady_run_without_a_checkpoint_holds_the_records_its_summary_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, refusal: str | None
) -> None:
    """
    An unsuccessful run without a last checkpoint is admitted against its summary's terminal epoch and step and its validated points.

    A stage that committed no optimizer group wrote no record, as a transient one: a record it holds is refused, and so is a
    summary without its producer identity or with another one, also without any record. An Optuna trial writes no last
    checkpoint, also after committed groups: its history holds their validation and epoch records, bound to the producer
    identity its summary records, whose numerical environment (a container on CUDA) is not the reader's; a final record
    beyond the summary's global step is refused. Without a record, only the runtime session of the attempt that ended the
    run binds that environment: a malformed value is refused, and so is an image or a CUDA build other than the session's;
    a package version is recorded once, in the identity alone, so another one is admitted.
    """
    trial = case.startswith("trial")
    if trial:
        source = config_naming.ConfigSource("steady_flow", common.paths.OPTUNA_FAMILY, "study")
        config = config_loader.resolve_config(configs.direct_config(), source=source, output_root=tmp_path)
    else:
        config = configs.resolve(configs.direct_config(), output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(config)
    run_state.write_lightweight_run_state(run_dir, config, status="unsuccessful", completed_epoch=0, include_best=False)
    common.paths.resolve_last_checkpoint_file(run_dir).unlink()
    rebuilt = run_state.producer_identity()
    producer = {**rebuilt["numerical_environment"], "sif_sha256": "f" * 64, "torch_cuda_build": "12.1"}
    recorded = {**rebuilt, "numerical_environment": producer}
    # The attempt that ended the run records its Slurm job's image and, on CUDA, PyTorch's CUDA build.
    session = {"device_type": "cuda", "cuda_build_version": "12.1", "slurm": {"sif_sha256": "f" * 64}}
    monkeypatch.setattr(experiments.run, "_saved_run_identity", lambda *_args, **_kwargs: ({}, None, rebuilt))
    total = learning.training.budget.passes_budget(int(config["training"]["budget_epochs"]), 4)
    budget = learning.training.budget.StageBudget(unit="units", total=total)
    if trial:
        budget.record_group(units=total // 2, supervised_transitions=total // 2, train_seconds=0.5)
        point = budget.record_validation()
    outcome = {"status": "diverged", "budget_progress": budget.progress, "eligible_points": 0, "divergence": "loss is non-finite"}
    summary = {
        **experiments.run.read_run_summary(run_dir),
        **experiments.run._terminal_budget_evidence(budget.state_dict()),
        "training_units": {"full_set_units": 4},
        "training_outcome": outcome,
        "completed_epoch": int(trial),
        "global_step": 2 * int(trial),
        "best_epoch": None,
        "last_checkpoint_sha256": None,
        "checkpoint_identity": recorded,
        "runtime_sessions": [session],
    }
    environment_change = {
        "stage_without_group_with_a_malformed_environment": {"numpy": ["2.0.0"]},
        "stage_without_group_with_another_image": {"sif_sha256": "0" * 64},
        "stage_without_group_with_another_cuda_build": {"torch_cuda_build": "11.8"},
        "stage_without_group_with_another_package_version": {"numpy": "0.0.1"},
    }.get(case)
    if case == "stage_without_group_or_its_identity":
        del summary["checkpoint_identity"]
    elif case == "stage_without_group_with_another_identity":
        summary["checkpoint_identity"] = {**recorded, "effective_config_digest": "0" * 64}
    elif environment_change is not None:
        summary["checkpoint_identity"] = {**recorded, "numerical_environment": {**producer, **environment_change}}
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)
    write = learning.training.history.make_epoch_state_callback(
        run_dir, task="steady_flow", run_name=str(config["run"]["name"]), checkpoint_identity=recorded
    )
    if trial:
        # The validation at the reached point inside epoch 1, then that epoch's end at the trial's terminal step (one later beyond it).
        write(1, None, {"global_step": 1.0, "validation/point": float(point), "validation/eligible": 0.0})
        write(1, {"global_step": 3.0 if case == "trial_history_beyond_its_step" else 2.0, "train/loss_total": 0.5}, None)
    elif case == "stage_without_group_with_a_record":
        write(1, {"global_step": 1.0, "train/loss_total": 0.5}, None)

    if refusal is None:
        assert experiments.run.validate_finished_run(run_dir)["terminal_budget"].spent == budget.spent
    else:
        with pytest.raises(experiments.run.RunLifecycleError, match=refusal):
            experiments.run.validate_finished_run(run_dir)


@pytest.mark.parametrize("earlier", ["no_history", "no_sessions_under_an_earlier_contract", "contract_without_history", "malformed_attempt"])
def test_resume_refuses_a_summary_that_records_its_attempts_in_an_earlier_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, earlier: str
) -> None:
    """A run whose summary was not created under the current duration contract is not continued, also without any attempt."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_config_run_dir(config, output_root=output_root)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    summary_path = common.paths.resolve_run_summary_path(run_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # The helper writes what a fresh run starts with: no sessions under the current contract.
    assert (summary["runtime_sessions"], summary["duration_contract"]) == ([], experiments.run.RUN_DURATION_CONTRACT)
    if earlier == "no_history":
        # A summary created before attempts were recorded carries neither the session list nor the contract.
        del summary["runtime_sessions"], summary["duration_contract"]
    elif earlier == "contract_without_history":
        # The current contract without its attempt history would let a resume publish totals of the new attempt alone.
        del summary["runtime_sessions"]
    elif earlier == "malformed_attempt":
        # An ended attempt without its committed work cannot enter complete totals.
        summary["runtime_sessions"] = [{"attempt": {"ended_at": "2025-01-02T03:04:00+00:00", "elapsed_seconds": 1.0, "train_seconds": 1.0}}]
    else:
        summary["duration_contract"] = {key: value for key, value in summary["duration_contract"].items() if key != "clock_source"}
    common.serialization.atomic_write_json(summary_path, summary)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("an earlier schema must not train"))
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}

    with pytest.raises(experiments.run.RunLifecycleError, match="earlier summary schema"):
        experiments.run.run_experiment(common.paths.resolve_run_config_path(run_dir), resume=run_dir)

    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


def test_resume_admission_refuses_a_contradictory_history_before_any_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The resume admission validates the history read-only against the last checkpoint it admits as continuation source.

    A history whose counted epoch ends at the last's step admits the attempt, after one admission before the writer lease
    and one under it. A history that ends after it is refused before the writer lease, so its lock file is not created,
    and the run, its summary and runtime sessions included, stays byte-identical. The last is loaded as continuation
    source, the best only to be read.
    """
    config = configs.resolve(configs.direct_config(), output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(config)
    run_state.write_lightweight_run_state(run_dir, config, status="interrupted", completed_epoch=1)
    identity = {"effective_config_digest": "a" * 64}
    loads: list[tuple[str, bool]] = []
    attempts: list[Path] = []

    def load(_path: Path, *, expected_role: str, resume: bool = False, **_kwargs: Any) -> dict[str, Any]:
        loads.append((expected_role, resume))
        # The last closed epoch 1 at its one committed update; the best agrees with it.
        return {"best_metric": 0.5, "best_epoch": 1, "objective_history": [], "global_step": 1}

    def attempt(_config: dict[str, Any], *, run_dir: Path, **_kwargs: Any) -> dict[str, Any]:
        attempts.append(run_dir)
        return {}

    monkeypatch.setattr(experiments.run, "_read_saved_data_artifacts", lambda *_args: (SimpleNamespace(as_payload=dict), {}))
    monkeypatch.setattr(experiments.run, "_validate_saved_data_contract", lambda *_args: None)
    monkeypatch.setattr(config_loader, "admit_pinned_datasets", lambda *_args: None)  # the synthetic config publishes no Dataset
    monkeypatch.setattr(learning.training.checkpoint, "build_checkpoint_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(learning.training.checkpoint, "load_checkpoint", load)
    monkeypatch.setattr(learning.training.checkpoint, "recorded_history_epochs", lambda _payload: 1)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", attempt)
    binding: dict[str, Any] = {"task": "steady_flow", "run_name": str(config["run"]["name"]), "checkpoint_identity": identity}

    def record_epoch_end(step: float) -> None:
        learning.training.history.make_epoch_state_callback(run_dir, **binding)(1, {"global_step": step, "train/loss_total": 0.5}, None)

    def resume() -> None:
        experiments.run._run_resolved_experiment(config, config_path=common.paths.resolve_run_config_path(run_dir), resume=run_dir)

    record_epoch_end(1.0)
    resume()
    assert (len(attempts), loads) == (1, [("last", True), ("best", False)] * 2)
    lock = experiments.run._run_writer_lock_path(run_dir)
    # The admitted resume created the lock file; the refused one must not create it again.
    lock.unlink()
    record_epoch_end(2.0)
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    with pytest.raises(experiments.run.ContinuationRefusedError, match="global step"):
        resume()
    assert len(attempts) == 1
    assert not lock.exists()
    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


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
        updates=experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC), memory_batch=2),
    )

    assert summary["runtime_sessions"][0]["slurm"] == provenance
    assert "slurm" not in summary["runtime_device"]
    provenance["source_worktree_sha256"] = "invalid"
    monkeypatch.setenv("ML_SLURM_PROVENANCE", json.dumps(provenance))
    with pytest.raises(experiments.run.RunLifecycleError, match="source_worktree_sha256"):
        experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC), memory_batch=2)


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
        updates = experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC), memory_batch=2)
        assert updates["runtime_sessions"][-1]["slurm"]["source_transition"] == transition
    else:
        with pytest.raises(experiments.run.RunLifecycleError, match="source_transition"):
            experiments.run.runtime_session_updates(run_dir, resolution, started_at=datetime.now(UTC), memory_batch=2)


def test_tracking_state_updates_merge_into_exactly_one_runtime_session(
    tmp_path: Path,
) -> None:
    """
    Merge observer facts into one running session under the writer lease.

    The update must keep status history and earlier tracking facts, change only the
    named session, and refuse a session id that is not recorded exactly once.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / "tracking-sessions")
    experiments.run.transition_run_status(run_dir, "initializing")
    resolution = experiments.run.learning.device.resolve_device("cpu")
    experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            resolution,
            started_at=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
            memory_batch=2,
            session_id="first-session",
            tracking_state={
                "requested_mode": "online",
                "status": "active",
            },
        ),
    )
    history = copy.deepcopy(experiments.run.read_run_summary(run_dir)["status_history"])

    with experiments.run.run_writer_lease(run_dir):
        updated = experiments.run._update_runtime_session_locked(
            run_dir,
            "first-session",
            {"wandb_run_id": "opaque-run-id", "last_logged_step": 2, "status": "failed", "failed_operation": "history"},
        )
        with pytest.raises(experiments.run.RunLifecycleError, match="Expected one runtime session"):
            experiments.run._update_runtime_session_locked(run_dir, "other-session", {"status": "failed"})

    assert updated["status"] == "running"
    assert updated["status_history"] == history
    assert experiments.run.read_run_summary(run_dir)["runtime_sessions"] == updated["runtime_sessions"]
    tracking_state = updated["runtime_sessions"][0]["tracking"]
    assert tracking_state["requested_mode"] == "online"
    assert tracking_state["wandb_run_id"] == "opaque-run-id"
    assert tracking_state["status"] == "failed"
    assert tracking_state["failed_operation"] == "history"


def _resume_config(*, epochs: int = 4) -> dict[str, Any]:
    """Return fixed semantic sections used by resume identity tests."""
    raw = configs.direct_config()
    raw["training"]["budget_epochs"] = epochs
    return configs.resolve(raw)


@pytest.mark.parametrize("epochs", [3, 7])
def test_resume_keeps_the_budget_and_admits_runtime_location_changes(epochs: int) -> None:
    """
    Resume admits device and output-root relocation but no other budget.

    The schedule and the validation points run over the budget, so neither a shorter
    nor a longer one continues the saved run.
    """
    saved = _resume_config(epochs=4)
    relocated = _resume_config(epochs=4)
    relocated["run"]["device"] = "cuda"
    relocated["paths"]["output_root"] = "/new/output"

    experiments.run.validate_resume_config(relocated, saved)
    with pytest.raises(ValueError, match="never extended or changed"):
        experiments.run.validate_resume_config(_resume_config(epochs=epochs), saved)


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        ("data", {"train_dataset": "other", "material_ood_datasets": ["ood"]}),
        ("model", {"kind": "tiny", "params": {"width": 8}}),
    ],
)
def test_resume_rejects_semantic_changes(section: str, replacement: Any) -> None:
    """
    Replace each scientific continuation section of an otherwise identical request.

    Every parametrized section must fail resume compatibility, proving operational
    allowances cannot conceal task, data, model, loss, or optimization drift.
    """
    saved = _resume_config()
    requested = _resume_config()
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


@pytest.mark.parametrize(
    "load_inference",
    [
        pytest.param(lambda run_dir: learning.inference.context.load_inference_context(run_dir=run_dir, device_policy="cpu"), id="steady"),
        pytest.param(lambda run_dir: learning.inference.transient.load_transient_inference_context(run_dir=run_dir), id="transient"),
    ],
)
def test_inference_of_both_tasks_refuses_a_run_while_a_writer_holds_it(load_inference: Callable[[Path], object], tmp_path: Path) -> None:
    """Steady and transient inference admit a run through the same reader lease, so neither reads a run being written."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    acquired, release = threading.Event(), threading.Event()
    outcomes: queue.Queue[str] = queue.Queue()
    holder = threading.Thread(target=_hold_run_writer_lease, args=(run_dir, acquired, release, outcomes))
    holder.start()
    try:
        assert acquired.wait(timeout=5)
        with pytest.raises(experiments.run.RunLifecycleError, match="active writer lease"):
            load_inference(run_dir)
    finally:
        release.set()
        holder.join(timeout=10)

    assert outcomes.get(timeout=5) == "released"


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
def test_saved_effective_resume_service_preserves_exact_science_and_refuses_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, renamed: bool
) -> None:
    """Saved config admission delegates the exact resolved request; it takes no scientific override, not even another budget."""
    config = configs.resolve(configs.direct_config(), output_root=tmp_path)
    run_dir = common.paths.resolve_config_run_dir(config)
    source = configs.write_yaml(run_dir / "config.yaml", config)
    # The saved status is admitted before the delegated admission, so the run must be one that continues.
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), {"status": "interrupted"})
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
    experiments.run.run_experiment(source, resume=run_dir)
    requested = calls["requested"]
    assert requested["training"] == config["training"]
    assert requested["model"] == config["model"]
    assert requested["data"] == config["data"]
    assert requested["run"]["layout"] == config["run"]["layout"]
    assert source.read_bytes() == original
    # An override contradicts the saved run, which refuses it like any other change of its saved science.
    refused = experiments.run.ContinuationRefusedError
    with pytest.raises(refused, match="cannot override scientific"):
        experiments.run.run_experiment(source, resume=run_dir, train_dataset="different")
    with pytest.raises(refused, match="cannot override scientific"):
        experiments.run.run_experiment(source, resume=run_dir, seed=19)
    with pytest.raises(refused, match="cannot override scientific"):
        experiments.run.run_experiment(source, resume=run_dir, epochs=int(config["training"]["budget_epochs"]) + 1)


def test_seed_override_selects_the_seed_leaf_and_seed_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replicate seed replaces run.seed before the layout and seed plan derive, so it needs no further config file."""
    request = configs.write_config(tmp_path, configs.direct_config())
    authored_seed = configs.direct_config()["run"]["seed"]
    calls: list[dict[str, Any]] = []

    def execute(requested: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append(requested)
        return {"result": {"status": "completed"}, "run_dir": tmp_path}

    monkeypatch.setattr(experiments.run, "_run_resolved_experiment", execute)

    experiments.run.run_experiment(request, device="cpu", output_root=tmp_path, seed=19)

    (requested,) = calls
    assert requested["run"]["seed"] == requested["run"]["layout"]["seed"] == 19 != authored_seed
    training = requested["training"]
    group = f"seed-19__ep-{training['budget_epochs']}__k-{training['units_per_update']}__sch-cosine"
    assert common.paths.resolve_config_run_dir(requested).name == group
    assert experiments.run.build_seed_plan(requested["run"]["seed"], stage=None) == experiments.run.build_seed_plan(19, stage=None)
    assert experiments.run.build_seed_plan(19, stage=None) != experiments.run.build_seed_plan(authored_seed, stage=None)


@pytest.mark.parametrize("stage", ["steady", "b"])
@pytest.mark.parametrize(("status", "refusal"), [("completed", "never extended"), ("unsuccessful", "not resumable")])
def test_a_finished_run_is_never_extended_and_another_budget_is_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, status: str, refusal: str
) -> None:
    """A finished run is never resumed: the refusal precedes any write, its lock file included, and a longer budget names another run."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "training"))
    if stage == "steady":
        raw, longer_raw = configs.direct_config(), configs.direct_config()
        longer_raw["training"]["budget_epochs"] += 1
        config, longer = (configs.resolve(item, output_root=tmp_path) for item in (raw, longer_raw))
    else:
        raw, longer_raw = configs.transient_matched_config(), configs.transient_matched_config()
        longer_raw["training"]["a0"]["budget_epochs"] *= 2  # the one K sizes B's K x N transitions
        config, longer = (dict(configs.resolve_plan(item, output_root=tmp_path).b) for item in (raw, longer_raw))
    run_dir = common.paths.resolve_config_run_dir(config)
    run_state.write_lightweight_run_state(run_dir, config, status=status, completed_epoch=1)
    monkeypatch.setattr(experiments.run, "_execute_prepared_run_locked", lambda *_args, **_kwargs: pytest.fail("a finished run must not train"))
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    with pytest.raises((ValueError, experiments.run.RunLifecycleError), match=refusal):
        experiments.run._run_resolved_experiment(config, config_path=common.paths.resolve_run_config_path(run_dir), resume=run_dir)
    assert {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before
    assert not common.paths.resolve_run_lock_path(run_dir).exists()
    assert common.paths.resolve_config_run_dir(longer) != run_dir
    assert learning.training.checkpoint.config_digest(longer) != learning.training.checkpoint.config_digest(config)


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

    with pytest.raises(experiments.run.ContinuationRefusedError, match="trials are not resumed"):
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

    class _ClosedStream:
        def write(self, _text: str) -> int:
            raise OSError(9, "Bad file descriptor")

        def flush(self) -> None:
            raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr("sys.stderr", _ClosedStream())
    common.paths.record_job_log_owner(owner)  # neither the failed write nor its failed warning may reach the job


@pytest.mark.parametrize(("start", "recorded"), [("fresh", 1), ("refused_resume", 0), ("trial", 0)])
def test_a_fresh_run_records_its_folder_as_job_log_owner_even_when_its_device_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str, recorded: int
) -> None:
    """
    A fresh execution records its run folder before any check can fail it; a trial's logs belong to its study.

    A resume records it only when its attempt starts, so one refused before, here by its device check, names no owner and
    no job log is linked into the run it left unchanged (the steady lifecycle test starts one).
    """
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

    def execute() -> None:
        if start != "refused_resume":
            experiments.run.execute_prepared_run(config, run_dir=run_dir, device_resolution=resolution)
            return
        with experiments.run.run_writer_lease(run_dir):
            experiments.run._execute_prepared_run_locked(
                config,
                run_dir=run_dir,
                persisted_config=config,
                saved_split_indices={},
                resume_from=common.paths.resolve_last_checkpoint_file(run_dir),
                device_resolution=resolution,
            )

    with pytest.raises(KeyboardInterrupt):
        execute()

    lines = owners_file.read_text(encoding="utf-8").splitlines() if owners_file.exists() else []
    assert lines == [str(run_dir.resolve())] * recorded
    # A started attempt ends interrupted; the refused resume leaves the run's status as it was.
    assert experiments.run.read_run_summary(run_dir)["status"] == ("initializing" if start == "refused_resume" else "interrupted")


@pytest.mark.parametrize("end", ["state_refusal", "environment_failure"])
def test_an_unstarted_continuation_names_its_reason_and_a_rerun_only_after_an_environment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], end: str
) -> None:
    """
    A continuation that ends before its attempt started leaves the run unchanged and names its reason.

    A refusal by the run's persisted state advises no rerun, which is refused again. The device check is the job's
    environment, so its failure, here a ``ValueError``, keeps its type and is advised the rerun in a matching allocation.
    """
    config = _minimal_config(tmp_path / "outputs")
    run_dir = experiments.run.prepare_fresh_run(config)
    reason = "the persisted state or the environment fails the continuation"
    error_type = experiments.run.ContinuationRefusedError if end == "state_refusal" else ValueError

    def fail(*_args: Any, **_kwargs: Any) -> None:
        # The device check is the attempt's first; a check of the run's state raises the typed refusal itself.
        raise error_type(reason)

    monkeypatch.setattr(experiments.run, "validate_deterministic_model_device_policy", fail)
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    with pytest.raises(error_type, match=reason), experiments.run.run_writer_lease(run_dir):
        experiments.run._execute_prepared_run_locked(
            config,
            run_dir=run_dir,
            persisted_config=config,
            saved_split_indices={},
            resume_from=common.paths.resolve_last_checkpoint_file(run_dir),
            device_resolution=experiments.run.learning.device.resolve_device("cpu"),
        )

    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before
    stop, *details = capsys.readouterr().err.splitlines()
    assert stop.startswith(f"{'refused' if end == 'state_refusal' else 'FAILED'} {config['run']['name']} ")
    assert reason in stop
    assert "unchanged" in details[-1]
    assert ("rerun" in details[-1]) is (end == "environment_failure")


@pytest.mark.parametrize("dataset", ["content_key", "linked_folder", "linked_manifest", "absent_manifest", "absent_before_a_linked_one"])
def test_a_continuation_is_refused_by_its_malformed_or_unsafe_pinned_dataset_and_ended_by_an_absent_one(
    transient_package: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], dataset: str
) -> None:
    """
    Malformed or unsafe content of a continuation's pinned Datasets refuses it as that input; an absent path is the job's environment.

    A key the loaders' Dataset code misses, as in a payload, is malformed Dataset content, named by its error and the
    Dataset folders, never claimed as a malformed manifest, which only the resume admission's manifest check reports. A
    Dataset folder or manifest that exists but is a symbolic link, which the manifest reader refuses, is refused naming
    it. Neither is a failed read, so no rerun heals it: the continuation is refused before its attempt started and advised
    no rerun. An absent manifest, as behind a mount the job cannot reach, keeps its error and is advised the rerun, also
    when a later pinned Dataset is a symbolic link: the readers stop at the first path they cannot use. Neither changes
    the run.
    """
    root, dataset_id = transient_package["root"], transient_package["train_dataset"]
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    dataset_dir = common.paths.resolve_dataset_dir("transient_drying", dataset_id, dataset_root=common.paths.get_datasets_root(storage_root=root))
    manifest_path = dataset_dir / common.paths.DATASET_MANIFEST_FILENAME
    named = {
        "content_key": f"Dataset content is malformed: KeyError: 'cases', in {dataset_dir}.",
        "linked_folder": str(dataset_dir),
        "linked_manifest": str(manifest_path),
    }.get(dataset)
    material_ood: list[str] = []
    if dataset.startswith("absent_"):
        manifest_path.unlink()
    if dataset == "absent_before_a_linked_one":
        linked = dataset_dir.with_name("linked_material_ood")
        linked.symlink_to(dataset_dir, target_is_directory=True)
        material_ood.append(linked.name)
    elif dataset.startswith("linked_"):
        unsafe = dataset_dir if dataset == "linked_folder" else manifest_path
        moved = unsafe.with_name(f"{unsafe.name}.moved")
        unsafe.rename(moved)
        unsafe.symlink_to(moved)
    raw = configs.transient_stage_a_only_config()
    raw["data"].update({"train_dataset": dataset_id, "material_ood_datasets": material_ood})
    config = configs.resolve(raw, output_root=tmp_path / "outputs")
    run_dir = experiments.run.prepare_fresh_run(config)

    def loaders(*_args: Any, **_kwargs: Any) -> None:
        # The training loaders read each manifest through its reader, then the Dataset's payload, which here lacks a key.
        experiments.run.datasets.packages.load_package_manifest("transient_drying", dataset_id, storage_root=root)
        key = "cases"
        raise KeyError(key)

    monkeypatch.setattr(config_loader, "create_dataloaders_from_config", loaders)
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    ending = experiments.run.ContinuationRefusedError if named is not None else FileNotFoundError
    with pytest.raises(ending) as ended, experiments.run.run_writer_lease(run_dir):
        experiments.run._execute_prepared_run_locked(
            config,
            run_dir=run_dir,
            persisted_config=config,
            saved_split_indices={},
            resume_from=common.paths.resolve_last_checkpoint_file(run_dir),
            device_resolution=experiments.run.learning.device.resolve_device("cpu"),
        )

    if named is not None:
        assert named in str(ended.value)
        assert (dataset != "content_key") or ("manifest" not in str(ended.value))
    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before
    assert ("rerun" in capsys.readouterr().err) is (named is None)
