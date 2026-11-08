"""Check which frozen source tree an experiment unit is bound to and how a job reads its own source evidence.

Responsibilities:
  - Check bound-tree discovery from the Slurm runtime sessions of steady, matched, resumed and Optuna units
  - Check that only leaves the job may write bind a tree and that a study binds independently of its trials
  - Check the continuation guard against implicit switches to another source tree
  - Check the preflight summary line the launcher consumes
  - Check that a Slurm ML job derives commit and dirty state from its source snapshot without Git
  - Check that artifacts name the executing tree inside a container job and in tree mode

Design principles:
  - Units are test-owned summaries at the exact layout paths the production resolvers name
  - Session order and start times, not file order, decide the newest session

This module does NOT:
  - Materialize or verify source trees, or submit jobs
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from support import configs, run_state

from src import analysis, common, experiments
from src.experiments import experiments_run_identity as run_identity
from src.experiments import experiments_run_source_binding as source_binding
from src.experiments.cli import cli_config_preflight
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_naming as config_naming
from src.experiments.config import experiments_config_preflight as preflight
from src.experiments.config import experiments_config_transient_plan as transient_plan
from src.experiments.tuning import experiments_tuning_optuna as optuna

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_START = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_COMMITS = {"a": "a" * 40, "b": "b" * 40, "c": "c" * 40}
_FINGERPRINTS = {"a": "1" * 64, "b": "2" * 64, "c": "3" * 64}
_TREES = {key: f"{_COMMITS[key][:12]}-{_FINGERPRINTS[key][:12]}" for key in _COMMITS}


def _session(hours: float, tree: str | None) -> dict[str, Any]:
    """Return one runtime session started ``hours`` after the reference time, with Slurm source provenance for ``tree``."""
    session: dict[str, Any] = {"session_id": f"session-{hours}", "started_at": (_START + timedelta(hours=hours)).isoformat()}
    if tree is not None:
        session["slurm"] = {"job_id": "12345", "source_commit": _COMMITS[tree], "source_worktree_sha256": _FINGERPRINTS[tree]}
    return session


def _write_leaf(run_dir: Path, *sessions: Mapping[str, Any], status: str = "completed") -> Path:
    """Publish a run leaf whose summary records the given sessions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"schema_version": common.paths.RUN_SUMMARY_SCHEMA_VERSION, "status": status, "runtime_sessions": list(sessions)},
    )
    return run_dir


def _authored(path: Path) -> dict[str, Any]:
    """Return the authored config exactly as the preflight hands it to the resolver."""
    return loader.apply_training_overrides(loader.load_authored_config(path))


def test_unit_without_slurm_sessions_binds_no_tree(tmp_path: Path) -> None:
    """Missing leaves, leaves without a summary, and interactive sessions bind nothing."""
    interactive = _write_leaf(tmp_path / "interactive", _session(1, None))
    (tmp_path / "bare").mkdir()

    assert source_binding.bound_source_tree_id([tmp_path / "absent", tmp_path / "bare", interactive]) is None
    assert source_binding.bound_source_tree_id([]) is None


def test_newest_slurm_session_across_leaves_binds_its_tree(tmp_path: Path) -> None:
    """The latest start time wins across leaves; interactive sessions are skipped even when newer."""
    first = _write_leaf(tmp_path / "first", _session(1, "a"), _session(4, "b"), _session(9, None))
    second = _write_leaf(tmp_path / "second", _session(2, "c"))

    assert source_binding.bound_source_tree_id([first, second]) == _TREES["b"]
    assert source_binding.bound_source_tree_id([second]) == _TREES["c"]


def test_session_times_are_compared_as_instants(tmp_path: Path) -> None:
    """A later instant wins whatever offset or missing zone its start time was written with."""
    earlier = _session(0, "a")
    earlier["started_at"] = "2026-01-02T05:04:00+02:00"  # 03:04 UTC
    naive = _session(0, "b")
    naive["started_at"] = "2026-01-02T03:05:00"  # naive times are UTC
    leaf = _write_leaf(tmp_path / "leaf", earlier, naive)

    assert source_binding.bound_source_tree_id([leaf]) == _TREES["b"]


@pytest.mark.parametrize(
    ("sessions", "error"),
    [
        ([{"started_at": _START.isoformat(), "slurm": {"source_commit": "abc", "source_worktree_sha256": "d" * 64}}], ValueError),
        ([{"slurm": {"source_commit": "a" * 40, "source_worktree_sha256": "d" * 64}}], experiments.run.RunLifecycleError),
        ("not-a-list", experiments.run.RunLifecycleError),
    ],
)
def test_malformed_recorded_sessions_fail_closed(tmp_path: Path, sessions: object, error: type[Exception]) -> None:
    """A unit whose history cannot be read is never silently treated as new."""
    run_dir = tmp_path / "leaf"
    run_dir.mkdir()
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"schema_version": common.paths.RUN_SUMMARY_SCHEMA_VERSION, "runtime_sessions": sessions},
    )

    with pytest.raises(error):
        source_binding.bound_source_tree_id([run_dir])


def test_steady_unit_reports_nothing_until_its_run_exists_then_its_newest_tree(tmp_path: Path) -> None:
    """A new steady unit has no bound tree; an existing run reports the tree of its newest Slurm session."""
    path = configs.write_config(tmp_path, configs.direct_config())

    new = preflight.validate_workflow(path, requested_workflow="train")

    assert new.unit_run_dirs == ()
    assert new.bound_source_tree is None
    run_dir = common.paths.resolve_config_run_dir(loader.resolve_config(_authored(path), source=config_naming.config_source_from_path(path)))
    _write_leaf(run_dir, _session(1, "a"), _session(2, "b"))

    existing = preflight.validate_workflow(path, requested_workflow="train")

    assert existing.unit_run_dirs == (run_dir,)
    assert existing.bound_source_tree == _TREES["b"]


def test_matched_unit_binds_only_the_unfinished_leaves_it_trains(tmp_path: Path) -> None:
    """Completed stages are read-only prerequisites and unselected stages are not written, so neither binds a tree."""
    path = configs.write_config(tmp_path, configs.transient_matched_config())
    plan = transient_plan.resolve_matched_a1_b_plan(_authored(path), source=config_naming.config_source_from_path(path))
    a0, b, a1 = (common.paths.resolve_config_run_dir(stage) for stage in plan.stages)
    _write_leaf(a0, _session(1, "a"))

    new_pair = preflight.validate_workflow(path, requested_workflow="train")

    assert new_pair.unit_run_dirs == ()
    assert new_pair.bound_source_tree is None
    _write_leaf(b, _session(2, "b"), status="interrupted")
    _write_leaf(a1, _session(3, "c"))

    pair = preflight.validate_workflow(path, requested_workflow="train")

    assert pair.unit_run_dirs == (b,)
    assert pair.bound_source_tree == _TREES["b"]
    only_a1 = preflight.validate_workflow(path, requested_workflow="train", train_stages=("a1",))
    assert (only_a1.unit_run_dirs, only_a1.bound_source_tree) == ((), None)
    selected = preflight.validate_workflow(path, requested_workflow="train", train_stages=("b", "a0"))
    assert (selected.unit_run_dirs, selected.bound_source_tree) == ((b,), _TREES["b"])
    _write_leaf(a0, _session(1, "a"), status="interrupted")
    resumed_a0 = preflight.validate_workflow(path, requested_workflow="train", train_stages=("a0",))
    assert (resumed_a0.unit_run_dirs, resumed_a0.bound_source_tree) == ((a0,), _TREES["a"])


def test_train_stages_are_admitted_only_for_matched_plans_and_as_distinct_stage_tokens(tmp_path: Path) -> None:
    """The stage selection is validated like the training CLI validates it."""
    matched = configs.write_config(tmp_path, configs.transient_matched_config())
    steady = configs.write_config(tmp_path / "steady", configs.direct_config())

    for invalid in (("a2",), ("b", "b"), ()):
        with pytest.raises(loader.ConfigError, match="distinct a0,b,a1 tokens"):
            preflight.validate_workflow(matched, requested_workflow="train", train_stages=invalid)
    with pytest.raises(loader.ConfigError, match="only to explicit matched_a1_b plans"):
        preflight.validate_workflow(steady, requested_workflow="train", train_stages=("a0",))


def test_saved_resume_binds_the_tree_of_that_stage_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resuming one saved stage continues exactly that leaf."""
    storage = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    resolved = configs.resolve(configs.direct_config(), storage_root=storage)
    run_dir = common.paths.resolve_config_run_dir(resolved)
    _write_leaf(run_dir, _session(1, "b"))
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")

    result = preflight.validate_workflow(config_path, requested_workflow="train", resume=run_dir)

    assert result.unit_run_dirs == (run_dir.resolve(),)
    assert result.bound_source_tree == _TREES["b"]


def test_optuna_study_binds_the_newest_session_of_its_existing_trials(tmp_path: Path) -> None:
    """The study directory's trial runs are the unit; files and a study without trials bind nothing."""
    path = configs.write_yaml(tmp_path / "study.yaml", configs.optuna_config())
    base_config = optuna.load_optuna_study_config(path).base_config
    layout = base_config["run"]["layout"]
    study = common.paths.resolve_study_dir(
        layout["task"],
        layout["dataset"],
        layout["config_name"],
        layout["budget_epochs"],
        layout["revision"],
        output_root=base_config["paths"]["output_root"],
    )
    study.mkdir(parents=True)

    empty = preflight.validate_workflow(path, requested_workflow="optuna")

    assert empty.unit_run_dirs == ()
    assert empty.bound_source_tree is None
    trials = study / common.paths.STUDY_TRIALS_DIRNAME
    first = _write_leaf(trials / "trial-0000", _session(1, "a"))
    second = _write_leaf(trials / "trial-0001", _session(5, "b"))
    (trials / "notes.txt").write_text("not a trial\n", encoding="utf-8")

    result = preflight.validate_workflow(path, requested_workflow="optuna")

    assert result.unit_run_dirs == (first, second)
    assert result.bound_source_tree == _TREES["b"]


def test_wrong_workflow_never_reads_unit_summaries(tmp_path: Path) -> None:
    """A workflow mismatch is rejected before any summary is opened, even when the unit's history is unreadable."""
    path = configs.write_config(tmp_path, configs.direct_config())
    run_dir = common.paths.resolve_config_run_dir(loader.resolve_config(_authored(path), source=config_naming.config_source_from_path(path)))
    run_dir.mkdir(parents=True)
    common.paths.resolve_run_summary_path(run_dir).write_text("{", encoding="utf-8")

    with pytest.raises(preflight.WorkflowMismatchError):
        preflight.validate_workflow(path, requested_workflow="optuna")


@pytest.mark.parametrize("existing", [False, True])
def test_preflight_cli_ends_its_summary_with_the_bound_tree_or_a_dash(tmp_path: Path, capsys: pytest.CaptureFixture[str], existing: bool) -> None:
    """The launcher reads the tree as the fifth tab field, ``-`` for a unit without history."""
    path = configs.write_config(tmp_path, configs.direct_config())
    if existing:
        run_dir = common.paths.resolve_config_run_dir(loader.resolve_config(_authored(path), source=config_naming.config_source_from_path(path)))
        _write_leaf(run_dir, _session(1, "c"))

    assert cli_config_preflight.main(["train", str(path)]) == 0

    fields = capsys.readouterr().out.rstrip("\n").split("\t")
    assert len(fields) == 5
    assert fields[0] == "experiment"
    assert fields[4] == (_TREES["c"] if existing else "-")


def test_preflight_cli_binds_only_the_selected_stage_leaves(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The launcher forwards ``--train-stages``; an unselected stage's tree never reaches the summary line."""
    path = configs.write_config(tmp_path, configs.transient_matched_config())
    plan = transient_plan.resolve_matched_a1_b_plan(_authored(path), source=config_naming.config_source_from_path(path))
    a0, b, _a1 = (common.paths.resolve_config_run_dir(stage) for stage in plan.stages)
    _write_leaf(a0, _session(1, "a"), status="interrupted")
    _write_leaf(b, _session(2, "b"), status="interrupted")

    assert cli_config_preflight.main(["train", str(path), "--train-stages", "a0"]) == 0
    assert capsys.readouterr().out.rstrip("\n").split("\t")[4] == _TREES["a"]
    assert cli_config_preflight.main(["train", str(path), "--train-stages", "all"]) == 0
    assert capsys.readouterr().out.rstrip("\n").split("\t")[4] == _TREES["b"]
    assert cli_config_preflight.main(["train", str(path), "--train-stages", "a1"]) == 0
    assert capsys.readouterr().out.rstrip("\n").split("\t")[4] == "-"
    assert cli_config_preflight.main(["train", str(path), "--train-stages", "a0,a0"]) == 2


def _provenance(tree: str) -> str:
    """Return the worker provenance payload of a job that runs in ``tree`` as far as source evidence is concerned."""
    return json.dumps({"source_commit": _COMMITS[tree], "source_worktree_sha256": _FINGERPRINTS[tree]})


def _snapshot(tree: str, **overrides: object) -> Path:
    """Publish a snapshot manifest below the isolated storage root."""
    manifest: dict[str, Any] = {
        "git_commit": _COMMITS[tree],
        "source_fingerprint_sha256": _FINGERPRINTS[tree],
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "untracked": [],
        **overrides,
    }
    path = common.paths.get_source_snapshots_root() / _TREES[tree] / "snapshot.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any attempt of the evidence code to find or run Git."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        message = "A frozen job must not call Git."
        raise AssertionError(message)

    monkeypatch.setattr(shutil, "which", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)


@pytest.mark.parametrize(
    ("overrides", "dirty"),
    [
        ({}, False),
        ({"untracked": [{"path": "new_module.py", "kind": "file"}]}, True),
        ({"tracked_diff_sha256": hashlib.sha256(b"diff --git").hexdigest()}, True),
    ],
)
@pytest.mark.usefixtures("no_git")
def test_slurm_job_reads_commit_and_dirty_state_from_its_snapshot_without_git(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], dirty: bool
) -> None:
    """Inside a Slurm ML job the submitted commit and the retained snapshot, not a Git checkout, describe the source."""
    _snapshot("a", **overrides)
    monkeypatch.setenv("ML_SLURM_PROVENANCE", _provenance("a"))

    assert run_identity.source_repository_evidence() == {"commit": _COMMITS["a"], "dirty": dirty}


@pytest.mark.usefixtures("no_git")
def test_slurm_job_without_its_snapshot_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing evidence is an error, never an unknown or clean state."""
    monkeypatch.setenv("ML_SLURM_PROVENANCE", _provenance("a"))

    with pytest.raises(FileNotFoundError, match="snapshot"):
        run_identity.source_repository_evidence()


@pytest.mark.usefixtures("no_git")
def test_slurm_job_rejects_a_snapshot_of_other_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot must be the one named by the submitted commit and fingerprint."""
    _snapshot("a", source_fingerprint_sha256=_FINGERPRINTS["b"])
    monkeypatch.setenv("ML_SLURM_PROVENANCE", _provenance("a"))

    with pytest.raises(ValueError, match="does not match"):
        run_identity.source_repository_evidence()


@pytest.mark.parametrize("raw", ["{", "[]", json.dumps({"source_commit": "abc", "source_worktree_sha256": "d" * 64})])
@pytest.mark.usefixtures("no_git")
def test_slurm_job_rejects_unreadable_provenance(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Provenance that cannot name a snapshot fails instead of falling back to Git."""
    monkeypatch.setenv("ML_SLURM_PROVENANCE", raw)

    with pytest.raises((ValueError, TypeError)):
        run_identity.source_repository_evidence()


def test_outside_a_slurm_job_the_checkout_still_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive runs keep the Git evidence; without Git the state is unknown."""
    monkeypatch.delenv("ML_SLURM_PROVENANCE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert run_identity.source_repository_evidence() == {"commit": None, "dirty": None}


def _job_environment(monkeypatch: pytest.MonkeyPatch, tree: str, *, transition: str = "none", config_sha256: str = "c" * 64) -> None:
    """Run the test as the Slurm ML job that the worker script would start in ``tree``."""
    provenance = {
        "job_id": "12345",
        "partition": "standard",
        "cpus_per_task": 2,
        "requested_memory": "4G",
        "requested_wall_time": "00:10:00",
        "allocated_mem_per_node": "4096",
        "mode": "cpu",
        "gres": "none",
        "source_commit": _COMMITS[tree],
        "source_worktree_sha256": _FINGERPRINTS[tree],
        "source_transition": transition,
        "config_sha256": config_sha256,
        "sif_path": "/workspace/runtime/containers/grainlegumes-pino-drying.sif",
        "sif_sha256": "d" * 64,
    }
    monkeypatch.setenv("ML_SLURM_PROVENANCE", json.dumps(provenance))


@pytest.mark.parametrize("config_sha256", ["c" * 64, "none"], ids=["train-job", "artifacts-job"])
def test_artifacts_record_the_tree_a_container_job_runs_in(monkeypatch: pytest.MonkeyPatch, config_sha256: str) -> None:
    """The container never sees ``SOURCE_TREE_ID``, so a job's artifacts name its tree from the provenance; elsewhere tree mode names it."""
    device = {"requested_policy": "cpu", "resolved_device": "cpu", "device_type": "cpu", "pytorch_version": "test"}
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    _job_environment(monkeypatch, "b", config_sha256=config_sha256)

    assert analysis.artifacts.contracts.runtime_provenance(device, inference_batch_size=1)["source_tree_id"] == _TREES["b"]
    monkeypatch.delenv("ML_SLURM_PROVENANCE")
    assert run_identity.executing_source_tree_id() is None
    monkeypatch.setenv("SOURCE_TREE_ID", _TREES["c"])
    assert run_identity.executing_source_tree_id() == _TREES["c"]


def test_continuing_a_leaf_in_its_bound_tree_or_a_fresh_leaf_is_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a bound leaf whose job runs elsewhere is refused; fresh, never-Slurm and same-tree leaves continue."""
    bound = _write_leaf(tmp_path / "bound", _session(1, "a"), _session(2, "b"), _session(3, None))
    interactive = _write_leaf(tmp_path / "interactive", _session(1, None))
    (tmp_path / "bare").mkdir()
    _job_environment(monkeypatch, "b")

    for leaf in (bound, interactive, tmp_path / "bare", tmp_path / "absent"):
        source_binding.admit_continuation((leaf,))
    monkeypatch.delenv("ML_SLURM_PROVENANCE")
    for leaf in (interactive, tmp_path / "bare", tmp_path / "absent"):
        source_binding.admit_continuation((leaf,))


@pytest.mark.parametrize("transition", ["none", "explicit"])
def test_job_in_another_tree_continues_a_bound_leaf_only_after_an_explicit_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    """The newest Slurm session names the tree; a job in a different tree needs ``source_transition`` explicit."""
    leaf = _write_leaf(tmp_path / "leaf", _session(1, "a"), _session(2, "b"))
    _job_environment(monkeypatch, "c", transition=transition)

    if transition == "explicit":
        source_binding.admit_continuation((leaf,))
        return
    with pytest.raises(experiments.run.SourceBindingError, match=rf"bound to source tree {_TREES['b']}.*runs in source tree {_TREES['c']}"):
        source_binding.admit_continuation((leaf,))


def test_job_without_slurm_provenance_cannot_continue_a_bound_leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interactive process would run the checkout's code, so it is told to go through the launcher."""
    leaf = _write_leaf(tmp_path / "leaf", _session(1, "a"))
    monkeypatch.delenv("ML_SLURM_PROVENANCE", raising=False)

    with pytest.raises(experiments.run.SourceBindingError, match=rf"bound to source tree {_TREES['a']}; continue it through scripts/slurm_ml.sh"):
        source_binding.admit_continuation((leaf,))


def test_the_continuation_path_refuses_an_implicit_tree_switch_before_any_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun of an authored config and a saved-config resume reach the same guard, and the leaf stays untouched."""
    storage = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    path = configs.write_config(tmp_path, configs.direct_config())
    resolved = loader.resolve_config(_authored(path), source=config_naming.config_source_from_path(path))
    run_dir = common.paths.resolve_config_run_dir(resolved)
    run_state.write_lightweight_run_state(run_dir, resolved, status="interrupted", completed_epoch=1)
    summary_path = common.paths.resolve_run_summary_path(run_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_sessions"] = [_session(1, "a")]
    common.serialization.atomic_write_json(summary_path, summary)
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    _job_environment(monkeypatch, "b")

    with pytest.raises(experiments.run.SourceBindingError, match="bound to source tree"):
        experiments.run.run_experiment(path)
    with pytest.raises(experiments.run.SourceBindingError, match="bound to source tree"):
        experiments.run.run_experiment(run_dir / "config.yaml", resume=run_dir)

    assert {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()} == before


def _study(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Return the study config path, its directory and its base config."""
    path = configs.write_yaml(tmp_path / "study.yaml", configs.optuna_config())
    base_config = optuna.load_optuna_study_config(path).base_config
    return path, source_binding.study_dir(base_config), base_config


def test_study_session_record_binds_a_study_before_its_first_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each recorded job session is kept; the newest one, trial sessions included, names the tree."""
    path, study, _base = _study(tmp_path)
    study.mkdir(parents=True)

    monkeypatch.delenv("ML_SLURM_PROVENANCE", raising=False)
    source_binding.record_study_session(study)
    assert not (study / source_binding.STUDY_SESSIONS_FILENAME).exists()
    assert preflight.validate_workflow(path, requested_workflow="optuna").bound_source_tree is None

    _job_environment(monkeypatch, "a")
    source_binding.record_study_session(study)
    first = preflight.validate_workflow(path, requested_workflow="optuna")
    assert first.unit_run_dirs == ()
    assert first.bound_source_tree == _TREES["a"]

    _job_environment(monkeypatch, "b", transition="explicit")
    source_binding.record_study_session(study)
    record = json.loads((study / source_binding.STUDY_SESSIONS_FILENAME).read_text(encoding="utf-8"))
    assert [session["slurm"]["source_commit"] for session in record["sessions"]] == [_COMMITS["a"], _COMMITS["b"]]
    assert record["sessions"][1]["slurm"]["source_transition"] == "explicit"
    assert preflight.validate_workflow(path, requested_workflow="optuna").bound_source_tree == _TREES["b"]

    future = _write_leaf(study / common.paths.STUDY_TRIALS_DIRNAME / "trial-0000", _session(10_000, "c"))
    assert source_binding.bound_source_tree_id([future], study=study) == _TREES["c"]


@pytest.mark.parametrize(
    "payload",
    ["{", "[]", json.dumps({"schema_version": 1, "sessions": "x"}), json.dumps({"schema_version": 2, "sessions": []})],
)
def test_malformed_study_session_record_fails_closed(tmp_path: Path, payload: str) -> None:
    """A study whose binding cannot be read is never treated as unbound."""
    _path, study, _base = _study(tmp_path)
    study.mkdir(parents=True)
    (study / source_binding.STUDY_SESSIONS_FILENAME).write_text(payload, encoding="utf-8")

    with pytest.raises(experiments.run.RunLifecycleError):
        source_binding.bound_source_tree_id([], study=study)
