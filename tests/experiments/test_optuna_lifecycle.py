"""Protect Optuna persistence, resume, pruning, and terminal-state behavior.

Responsibilities:
  - Check study persistence, resume, pruning at eligible points, signatures, and transient trial placement
  - Check that a reopened study continues its sampler reproducibly instead of repeating its first draws, and that a
    second invocation of a running study is refused before it opens the study
  - Check that a reopened study fails the trials a hard abort left running, refuses another device model or sampler, and
    runs only what is left of its allocation, its configured base first
  - Check that one trial driver publishes the same terminal record, Optuna attribute and trial line for both tasks, and
    advises an interrupted trial the reopen of its study, never a restart of the trial
  - Check that a diverged trial is pruned with the violated constraint TPE ranks last and an ineligible one without a value
  - Check that the study report counts every trial status, takes its best from completed trainings (a duplicate's stored
    value included), evaluates parameter importance on trained trials only and sums its invocations' seconds
  - Check that a failure while recording a trial's evidence or status, even an interrupt, never replaces its primary error
    and that a trial whose evidence was lost keeps its best checkpoint
  - Check that a failed run preparation, a lost status attribute and a tracking finish failure keep one agreeing end state
  - Check that a finished trial frees its checkpoint files while its summary keeps the best digest
  - Check that a trial repeating an existing experiment is reused or pruned without training, a completed owner's value
    only once the completed-run admission accepts that owner
  - Check that a study records its Slurm source binding independently of its trials and is continued only in that tree
  - Check that a study records its root as the owner of the job's log links

Design principles:
  - Scientific signatures and recoverable failures retain exact lifecycle meaning
  - Trial tests replace only data, model and training boundaries; the trial driver and run lifecycle stay real

This module does NOT:
  - Select production search spaces or contact a live study service
  - Validate completed trials against real artifacts; the integration lifecycles own that
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import optuna
import pytest
import torch
from optuna.samplers._tpe import sampler as tpe_sampler
from support import configs, optuna_trials
from support.synthetic_task import admitted_manifest
from torch.utils.data import DataLoader, TensorDataset

from src import common, datasets, domain, experiments, learning

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

optuna_runtime = experiments.tuning.optuna
_EXPECTED_GLOBAL_STEP = 7
_EXPECTED_RESUMED_TRIAL_COUNT = 4
_TASK = domain.tasks.registry.get_task("steady_flow")
_OBJECTIVE_ID = next(metric.id for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_TASKS = ("steady_flow", "transient_drying")
_SAMPLER_SEED = 1234  # the derived sampler seed an invocation hands its trials
_CPU_DEVICE = {"device": "cpu", "device_model": "cpu"}  # the device a CPU invocation admits for its trials


class _Trial:
    """Implement the small trial surface needed by reporter and failure tests."""

    def __init__(self, *, number: int = 0, prune: bool = False) -> None:
        self.number = number
        self.prune = prune
        self.attrs: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []
        self.constraints: dict[str, float] = {}

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> Any:
        del name
        return choices[0]

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        del name, high, log, step
        return low

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        del name, high, log, step
        return low

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def set_user_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def set_constraint(self, key: str, value: float) -> None:
        self.constraints[key] = value

    def should_prune(self) -> bool:
        return self.prune


_STUDY_ID = "steady_flow/synthetic_train/synthetic_lifecycle__ep-10"  # the lifecycle study trains 10 epochs


def _load(tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load one artificial study with external tracking disabled."""
    path = configs.write_yaml(
        tmp_path / "synthetic-lifecycle.yaml",
        _optuna_payload_with_dataset(),
    )
    loaded = optuna_runtime.load_optuna_study_config(path)
    base = copy.deepcopy(loaded.base_config)
    base["tracking"]["wandb"]["mode"] = "disabled"
    return replace(loaded, base_config=base)


def _optuna_payload_with_dataset() -> dict[str, Any]:
    """Set one exact synthetic Dataset ID in test-owned input."""
    payload = configs.optuna_config()
    payload["experiment"]["data"]["train_dataset"] = "synthetic_train"
    return payload


@pytest.fixture
def _steady_datasets_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a steady study bind its artificial Dataset IDs as admitted packages."""
    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest", admitted_manifest)


def test_signature_excludes_runtime_location_but_covers_science(
    tmp_path: Path,
) -> None:
    """Separate study continuation identity from invocation-only settings."""
    loaded = optuna_trials.with_configured_identities(_load(tmp_path))
    signature = optuna_runtime.build_study_signature(loaded)
    baseline = signature["digest"]

    operational_base = copy.deepcopy(loaded.base_config)
    operational_base["run"]["device"] = "cuda"
    operational_base["paths"]["output_root"] = str(tmp_path / "elsewhere")
    operational_study = copy.deepcopy(loaded.study)
    operational_study.update({"name": "renamed_display", "n_trials": 99})
    operational = replace(
        loaded,
        base_config=operational_base,
        study=operational_study,
    )
    assert optuna_runtime.build_study_signature(operational)["digest"] == baseline

    scientific_base = copy.deepcopy(loaded.base_config)
    scientific_base["optimizer"]["lr"] = 2.0e-3
    scientific_base = experiments.config.loader.finalize_run_layout(scientific_base)
    scientific = replace(loaded, base_config=scientific_base)
    assert optuna_runtime.build_study_signature(scientific)["digest"] != baseline

    changed_space = list(loaded.search_space)
    changed_space[0] = replace(
        changed_space[0],
        values=(*changed_space[0].values, 16),
    )
    changed = replace(loaded, search_space=tuple(changed_space))
    assert optuna_runtime.build_study_signature(changed)["digest"] != baseline


def _point(point: int, value: float | None, *, epoch: int = 1) -> dict[str, float]:
    """Return the validation metrics of one point: eligible with its objective, or not eligible without one."""
    metrics = {"validation/point": float(point), "validation/eligible": float(value is not None), "epoch": float(epoch)}
    return metrics if value is None else {**metrics, f"id/{_OBJECTIVE_ID}": value}


def test_reporter_reports_at_the_validation_point_index_and_prunes_immediately() -> None:
    """Report the objective keyed by the point index, reject malformed or non-increasing points, and stop on the pruning point."""
    malformed = optuna_runtime.OptunaPointReporter(trial=_Trial(), objective_id=_OBJECTIVE_ID, direction="minimize")
    with pytest.raises(RuntimeError, match="must carry a finite"):
        malformed(3, {**_point(1, None), "validation/eligible": 1.0})
    with pytest.raises(ValueError, match="validation/eligible"):
        malformed(3, {f"id/{_OBJECTIVE_ID}": 0.5, "validation/point": 2.0, "epoch": 1.0})
    with pytest.raises(ValueError, match="validation point between 1 and 20"):
        malformed(3, _point(21, 0.5))

    repeated = optuna_runtime.OptunaPointReporter(trial=_Trial(), objective_id=_OBJECTIVE_ID, direction="minimize")
    repeated(3, _point(2, 0.5))
    with pytest.raises(ValueError, match="must increase"):
        repeated(4, _point(2, 0.4))

    trial = _Trial(prune=True)
    reporter = optuna_runtime.OptunaPointReporter(trial=trial, objective_id=_OBJECTIVE_ID, direction="minimize")
    # Points 1 and 2 were skipped (one group crossed them) and point 3 was not eligible: no report and no pruning decision.
    reporter(2, _point(3, None))
    with pytest.raises(optuna.TrialPruned, match="validation point 4"):
        reporter(_EXPECTED_GLOBAL_STEP, _point(4, 0.5, epoch=2))
    # The pruner sees the point index, never the global step or the epoch.
    assert trial.reports == [(0.5, 4)]
    assert (trial.attrs["last_validation_point"], trial.attrs["last_reported_point"], trial.attrs["ineligible_points"]) == (4, 4, [3])
    assert trial.attrs["last_global_step"] == _EXPECTED_GLOBAL_STEP


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_existing_unbound_study_fails_before_trial_allocation(
    tmp_path: Path,
) -> None:
    """Reject a pre-existing SQLite study that holds a trial but no initialization record."""
    loaded = optuna_runtime.with_runtime_overrides(
        _load(tmp_path),
        device="cpu",
        output_root=tmp_path,
    )
    study_name = loaded.study["name"]
    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", study_name, loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    study_dir.mkdir(parents=True)
    storage = f"sqlite:///{study_dir / (study_name + '.db')}"
    optuna.create_study(
        study_name=f"steady_flow/synthetic_train/{study_dir.name}",
        direction="minimize",
        storage=storage,
    ).ask()

    with pytest.raises(
        ValueError,
        match="has no initialization record",
    ):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert not list(study_dir.glob("trial_*"))


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_a_system_exit_inside_a_trial_leaves_no_running_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Optuna finalizes only Exception and KeyboardInterrupt, so the study records a trial that SystemExit ended as failed."""
    loaded = optuna_runtime.with_runtime_overrides(_load(tmp_path), device="cpu", output_root=tmp_path)

    def exiting_factory(_config: optuna_runtime.OptunaStudyConfig, **_invocation: Any) -> Callable[[Any], float]:
        def objective(_trial: Any) -> float:
            raise SystemExit(3)

        return objective

    monkeypatch.setattr(optuna_runtime, "create_objective", exiting_factory)
    _slurm_job_in(monkeypatch, "a")
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(tmp_path / "owners"))
    with pytest.raises(SystemExit):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)

    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", loaded.study["name"], loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    stored = optuna.load_study(
        study_name=optuna_runtime._database_study_id(loaded), storage=f"sqlite:///{study_dir / (loaded.study['name'] + '.db')}"
    )
    assert [trial.state for trial in stored.trials] == [optuna.trial.TrialState.FAIL]


def test_epoch_budget_names_the_study_directory_and_database_identity(tmp_path: Path) -> None:
    """Another trial epoch budget is another study unit; the same budget reopens it and a revision label stays last."""
    path = configs.write_yaml(tmp_path / "synthetic-lifecycle.yaml", _optuna_payload_with_dataset())
    short = optuna_runtime.load_optuna_study_config(path, epochs=2)
    again = optuna_runtime.load_optuna_study_config(path, epochs=2)
    longer = optuna_runtime.load_optuna_study_config(path, epochs=5)
    revised = optuna_runtime.load_optuna_study_config(path, epochs=5, revision="fix")
    name = short.study["name"]

    assert optuna_runtime._study_dir(short).name == f"{name}__ep-2"
    assert optuna_runtime._study_dir(again) == optuna_runtime._study_dir(short)
    assert optuna_runtime._study_dir(longer).name == f"{name}__ep-5"
    assert optuna_runtime._study_dir(revised).name == f"{name}__ep-5__rev-fix"
    assert len({optuna_runtime._database_study_id(config) for config in (short, longer, revised)}) == 3
    assert optuna_runtime._database_study_id(short).endswith(f"/{name}__ep-2")


def _slurm_job_in(monkeypatch: pytest.MonkeyPatch, tree: str, *, transition: str = "none") -> None:
    """Run the test as the Slurm ML job that starts in the source tree whose commit is ``tree`` repeated."""
    provenance = {
        "job_id": "12345",
        "partition": "standard",
        "cpus_per_task": 2,
        "requested_memory": "4G",
        "requested_wall_time": "00:10:00",
        "allocated_mem_per_node": "4096",
        "mode": "cpu",
        "gres": "none",
        "source_commit": tree * 40,
        "source_worktree_sha256": "1" * 64,
        "source_transition": transition,
        "config_sha256": "c" * 64,
        "sif_path": "/workspace/runtime/containers/grainlegumes-pino-drying.sif",
        "sif_sha256": "d" * 64,
    }
    monkeypatch.setenv("ML_SLURM_PROVENANCE", json.dumps(provenance))


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_study_keeps_its_source_binding_from_creation_without_any_trial_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A study records its job's tree at creation and every reopen and is continued only in that tree unless switched explicitly.

    Its root, never a trial folder, is also named as the owner of the job's log links.
    """
    loaded = optuna_runtime.with_runtime_overrides(_load(tmp_path), device="cpu", output_root=tmp_path)
    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", loaded.study["name"], loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    sessions = study_dir / experiments.source_binding.STUDY_SESSIONS_FILENAME

    def crashing_factory(_config: optuna_runtime.OptunaStudyConfig, **_invocation: Any) -> Callable[[Any], float]:
        def objective(_trial: Any) -> float:
            message = "crash before the first trial session"
            raise RuntimeError(message)

        return objective

    monkeypatch.setattr(optuna_runtime, "create_objective", crashing_factory)
    _slurm_job_in(monkeypatch, "a")
    owners_file = tmp_path / "owners"
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(owners_file))
    with pytest.raises(RuntimeError, match="crash before the first trial session"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)

    assert owners_file.read_text(encoding="utf-8").splitlines() == [str(study_dir.resolve())]
    assert [entry["slurm"]["source_commit"] for entry in json.loads(sessions.read_text(encoding="utf-8"))["sessions"]] == ["a" * 40]
    assert experiments.source_binding.bound_source_tree_id(
        experiments.source_binding.existing_study_run_dirs(loaded.base_config), study=study_dir
    ) == ("a" * 12 + "-" + "1" * 12)
    _slurm_job_in(monkeypatch, "b")
    with pytest.raises(experiments.run.SourceBindingError, match="bound to source tree"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    monkeypatch.delenv("ML_SLURM_PROVENANCE")
    with pytest.raises(experiments.run.SourceBindingError, match=r"scripts/slurm_ml\.sh"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert len(json.loads(sessions.read_text(encoding="utf-8"))["sessions"]) == 1

    _slurm_job_in(monkeypatch, "b", transition="explicit")
    with pytest.raises(RuntimeError, match="crash before the first trial session"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    recorded = json.loads(sessions.read_text(encoding="utf-8"))["sessions"]
    assert [(entry["slurm"]["source_commit"], entry["slurm"]["source_transition"]) for entry in recorded] == [
        ("a" * 40, "none"),
        ("b" * 40, "explicit"),
    ]


def _stored_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[optuna_runtime.OptunaStudyConfig, dict[str, str | None], Path]:
    """
    Return a 10-epoch study stored in SQLite below ``tmp_path``, the numerical environment and the HPO implementation root.

    The study reads the returned environment and the HPO code below the root, so a test changes either in place.
    """
    environment: dict[str, str | None] = {
        "python": "3.12.1",
        "torch": "2.5.0",
        "neuraloperator_implementation_sha256": "a" * 64,
        "sif_sha256": "b" * 64,
        "device_model": "NVIDIA RTX 6000 Ada Generation",
    }
    monkeypatch.setattr(
        optuna_runtime.learning.training.checkpoint,
        "numerical_environment_identity",
        lambda: {key: value for key, value in environment.items() if key != "device_model"},
    )
    # The study runs on the CPU but admits the device model the environment names, as on a GPU node.
    monkeypatch.setattr(
        optuna_runtime, "_device_identity", lambda resolution: {"device": str(resolution.device), "device_model": environment["device_model"]}
    )
    implementation_root = tmp_path / "implementation"
    optuna_source = implementation_root / "experiments/tuning/experiments_tuning_optuna.py"
    search_source = implementation_root / "experiments/tuning/experiments_tuning_search_space.py"
    optuna_source.parent.mkdir(parents=True)
    optuna_source.write_text('def objective_policy():\n    return "minimize"\n', encoding="utf-8")
    search_source.write_text('def search_policy():\n    return "float"\n', encoding="utf-8")
    hpo_projection = common.code_identity.CodeProjection(
        name="test_hpo",
        files=("experiments/tuning/experiments_tuning_optuna.py", "experiments/tuning/experiments_tuning_search_space.py"),
    )
    monkeypatch.setattr(optuna_runtime, "_hpo_implementation_sha256", lambda _task_id: hpo_projection.sha256(implementation_root))
    loaded = _load(tmp_path)
    base = copy.deepcopy(loaded.base_config)
    base["training"]["budget_epochs"] = 10
    base = experiments.config.loader.finalize_run_layout(base)
    settings = copy.deepcopy(loaded.study)
    settings["name"] = "synthetic_lifecycle"
    settings["full_budget_epochs"] = 10
    settings["n_trials"] = 10
    settings["pruner"] = {
        "kind": "median",
        "n_startup_trials": 0,
        "n_min_trials": 1,
        "n_warmup_steps": 5,
        "interval_steps": 5,
    }
    loaded = optuna_runtime.with_runtime_overrides(
        replace(loaded, base_config=base, study=settings),
        device="cpu",
        output_root=tmp_path,
    )
    return loaded, environment, implementation_root


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_sqlite_study_persists_states_and_resumes_new_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve completed history while adding failed and pruned trial states."""
    loaded, environment, implementation_root = _stored_study(tmp_path, monkeypatch)
    optuna_source = implementation_root / "experiments/tuning/experiments_tuning_optuna.py"
    search_source = implementation_root / "experiments/tuning/experiments_tuning_search_space.py"

    def synthetic_factory(
        config: optuna_runtime.OptunaStudyConfig,
        **_invocation: Any,
    ) -> Callable[[Any], float]:
        def objective(trial: Any) -> float:
            experiments.tuning.search_space.suggest_trial_overrides(trial, config.search_space)
            if trial.number == 1:
                message = "recoverable synthetic failure"
                raise optuna_runtime.RecoverableTrialError(message)
            reporter = optuna_runtime.OptunaPointReporter(trial=trial, objective_id=_OBJECTIVE_ID, direction="minimize")
            values = (0.10, 0.09) if trial.number == 0 else (1.0, 0.9)
            for epoch, point, value in zip((5, 10), (10, 20), values, strict=True):
                reporter(epoch, _point(point, value, epoch=epoch))
            assert reporter.best_value is not None
            return reporter.best_value

        return objective

    monkeypatch.setattr(
        optuna_runtime,
        "create_objective",
        synthetic_factory,
    )
    first = optuna_runtime.run_optuna_study(loaded, n_trials=2)
    first_evidence = (
        first.trials[0].state,
        first.trials[0].value,
        dict(first.trials[0].intermediate_values),
    )
    resumed = optuna_runtime.run_optuna_study(loaded, n_trials=1)

    assert [trial.number for trial in resumed.trials] == [0, 1, 2]
    assert [trial.state for trial in resumed.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]
    assert first_evidence == (
        resumed.trials[0].state,
        resumed.trials[0].value,
        dict(resumed.trials[0].intermediate_values),
    )
    assert resumed.trials[0].intermediate_values == {10: 0.10, 20: 0.09}
    assert resumed.trials[2].intermediate_values == {10: 1.0}

    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", "synthetic_lifecycle", loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    summary = json.loads((study_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert [trial["number"] for trial in summary["trials"]] == [0, 1, 2]
    assert summary["training_seed"] == loaded.base_config["run"]["seed"]
    assert summary["sampler"]["seed"] == loaded.study["seed"]
    assert (study_dir / "best_config.yaml").is_file()
    # Trials ended outside the trial driver count by their Optuna state; the study's seconds sum both ended invocations.
    assert {status: count for status, count in summary["trial_counts"].items() if count} == {"completed": 1, "failed": 1, "pruned": 1}
    invocations = [value for key, value in resumed.user_attrs.items() if key.startswith("invocation/")]
    # Each invocation records the stored trial count its sampler seed derives from beside its seconds.
    assert sorted(record["stored_trials"] for record in invocations) == [0, 2]
    assert summary["unrecorded_invocations"] == 0
    assert summary["study_seconds"] == pytest.approx(sum(record["seconds"] for record in invocations))
    assert summary["study_seconds"] > summary["elapsed_seconds"]

    def unexpected_factory(
        _config: optuna_runtime.OptunaStudyConfig,
        **_invocation: Any,
    ) -> Callable[[Any], float]:
        def objective(_trial: Any) -> float:
            message = "unexpected synthetic bug"
            raise RuntimeError(message)

        return objective

    monkeypatch.setattr(
        optuna_runtime,
        "create_objective",
        unexpected_factory,
    )

    # A publication error of the failed study's summary must not replace the study's own error.
    def failing_publication(*_args: object) -> str:
        message = "publication error"
        raise OSError(message)

    with monkeypatch.context() as publication_patch:
        publication_patch.setattr(optuna_runtime, "_best_experiment_yaml", failing_publication)
        with pytest.raises(RuntimeError, match="unexpected synthetic bug"):
            optuna_runtime.run_optuna_study(loaded, n_trials=1)

    storage = f"sqlite:///{study_dir / 'synthetic_lifecycle.db'}"
    reopened = optuna.load_study(
        study_name=_STUDY_ID,
        storage=storage,
    )
    assert [trial.number for trial in reopened.trials] == [0, 1, 2, 3]
    assert reopened.trials[-1].state == optuna.trial.TrialState.FAIL

    drifted = copy.deepcopy(loaded.base_config)
    drifted["optimizer"]["lr"] = 2.0e-3
    drifted = experiments.config.loader.finalize_run_layout(drifted)
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(
            replace(loaded, base_config=drifted),
            n_trials=1,
        )
    assert (
        len(
            optuna.load_study(
                study_name=_STUDY_ID,
                storage=storage,
            ).trials
        )
        == _EXPECTED_RESUMED_TRIAL_COUNT
    )

    budget_drift = copy.deepcopy(loaded.base_config)
    budget_drift["training"]["budget_epochs"] = 11
    budget_drift = experiments.config.loader.finalize_run_layout(budget_drift)
    # Another budget names another study, so it can never reopen this one with a signature mismatch.
    assert optuna_runtime._database_study_id(replace(loaded, base_config=budget_drift)) != _STUDY_ID
    reopened = optuna.load_study(study_name=_STUDY_ID, storage=storage)
    assert len(reopened.trials) == _EXPECTED_RESUMED_TRIAL_COUNT

    persisted_environment = reopened.user_attrs["numerical_environment"]
    assert persisted_environment == {
        "python": "3.12.1",
        "torch": "2.5.0",
        "neuraloperator_implementation_sha256": "a" * 64,
        "sif_sha256": "b" * 64,
        "device_model": "NVIDIA RTX 6000 Ada Generation",
    }
    # One GPU type per study: reopening it on another GPU model is refused like another numerical environment.
    for key, changed_value in (
        ("torch", "2.5.1"),
        ("neuraloperator_implementation_sha256", "c" * 64),
        ("sif_sha256", "c" * 64),
        ("sif_sha256", None),
        ("device_model", "Tesla V100-SXM2-32GB"),
    ):
        environment[key] = changed_value
        with pytest.raises(ValueError, match="numerical_environment"):
            optuna_runtime.run_optuna_study(loaded, n_trials=1)
        assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT
        environment[key] = persisted_environment[key]

    other_sampler = {**loaded.study, "sampler": {"kind": "tpe", "multivariate": not loaded.study["sampler"]["multivariate"]}}
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(replace(loaded, study=other_sampler), n_trials=1)
    assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT

    optuna_source.write_text('def objective_policy():\n    return "maximize"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT
    optuna_source.write_text('def objective_policy():\n    return "minimize"\n', encoding="utf-8")
    search_source.write_text('def search_policy():\n    return "integer"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_a_reopened_study_continues_its_sampler_and_the_same_invocations_reproduce_its_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reopening during TPE's random startup repeats no earlier trial's parameters; the same invocations draw the same trials.

    Every invocation hands its trials the seed its sampler derives from the stored trial count.
    """
    trial_seeds: list[int] = []

    def suggesting_factory(config: optuna_runtime.OptunaStudyConfig, *, sampler_seed: int, **_invocation: Any) -> Callable[[Any], float]:
        trial_seeds.append(sampler_seed)

        def objective(trial: Any) -> float:
            experiments.tuning.search_space.suggest_trial_overrides(trial, config.search_space)
            return 1.0

        return objective

    monkeypatch.setattr(optuna_runtime, "create_objective", suggesting_factory)

    def two_invocations(root: Path) -> list[dict[str, Any]]:
        loaded, _environment, _implementation = _stored_study(root, monkeypatch)
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
        return [trial.params for trial in optuna_runtime.run_optuna_study(loaded, n_trials=1).trials]

    first = two_invocations(tmp_path / "first")
    assert len(first) == 2
    assert first[1] != first[0]
    seed = _load(tmp_path).study["seed"]
    assert trial_seeds == [common.random.derive_subseed(seed, f"optuna_sampler_after_{stored}_trials") % 2**32 for stored in (0, 1)]
    assert two_invocations(tmp_path / "second") == first


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_a_second_invocation_of_a_running_study_is_refused_before_it_opens_the_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One invocation runs a study at a time: while another holds the study's lock, a second one stops before the study exists."""
    loaded, _environment, _implementation = _stored_study(tmp_path, monkeypatch)
    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", "synthetic_lifecycle", loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    held, release = threading.Event(), threading.Event()

    def running_invocation() -> None:
        with common.locking.exclusive_file_lock(study_dir / optuna_runtime._STUDY_LOCK_FILENAME, blocking=False):
            held.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=running_invocation)
    holder.start()
    try:
        assert held.wait(timeout=10)
        with pytest.raises(RuntimeError, match="'synthetic_lifecycle' is already run by another invocation") as refused:
            optuna_runtime.run_optuna_study(loaded, n_trials=1)
    finally:
        release.set()
        holder.join()
    assert type(refused.value) is RuntimeError
    assert not (study_dir / "synthetic_lifecycle.db").exists()


def _sampling_factory(config: optuna_runtime.OptunaStudyConfig, **_invocation: Any) -> Callable[[Any], float]:
    """Return an objective that samples the search space and fails trial 1 recoverably, prunes trial 2 and completes the others."""

    def objective(trial: Any) -> float:
        experiments.tuning.search_space.suggest_trial_overrides(trial, config.search_space)
        if trial.number == 1:
            message = "recoverable synthetic failure"
            raise optuna_runtime.RecoverableTrialError(message)
        if trial.number == 2:
            raise optuna.TrialPruned
        return 1.0

    return objective


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_a_reopened_study_fails_the_trials_a_hard_abort_left_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial a killed invocation left RUNNING is failed with its reason and run evidence before the next trial is sampled."""
    loaded, _environment, _implementation = _stored_study(tmp_path, monkeypatch)
    monkeypatch.setattr(optuna_runtime, "create_objective", _sampling_factory)
    optuna_runtime.run_optuna_study(loaded, n_trials=1)
    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", "synthetic_lifecycle", loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    # A hard abort: the killed invocation's trial stays RUNNING and its run directory keeps the summary of a running run.
    aborted = optuna.load_study(study_name=_STUDY_ID, storage=f"sqlite:///{study_dir / 'synthetic_lifecycle.db'}").ask()
    run_dir = study_dir / common.paths.STUDY_TRIALS_DIRNAME / f"{aborted.number:04d}__aborted"
    run_dir.mkdir(parents=True)
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), {"status": "running"})

    reopened = optuna_runtime.run_optuna_study(loaded, n_trials=1)

    states = optuna.trial.TrialState
    assert [trial.state for trial in reopened.trials] == [states.COMPLETE, states.FAIL, states.PRUNED]
    record = reopened.user_attrs[f"aborted_trial/{aborted.number}"]
    assert record["run_summaries"] == {run_dir.name: "running"}
    assert "aborted" in record["reason"]


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_study_n_trials_is_its_total_allocation_and_its_configured_base_runs_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With 3 of 5 trials done a rerun runs exactly the 2 left and then none; trial 0 is the configured base.

    The report counts the allocation's finished, pruned and failed trials.
    """
    loaded, _environment, _implementation = _stored_study(tmp_path, monkeypatch)
    loaded = replace(loaded, study={**loaded.study, "n_trials": 5})
    monkeypatch.setattr(optuna_runtime, "create_objective", _sampling_factory)

    first = optuna_runtime.run_optuna_study(loaded, n_trials=3)

    assert len(first.trials) == 3
    assert first.trials[0].params == {
        "hidden_channels": loaded.base_config["model"]["params"]["hidden_channels"],
        "learning_rate": loaded.base_config["optimizer"]["lr"],
    }
    assert len(optuna_runtime.run_optuna_study(loaded).trials) == 5
    assert len(optuna_runtime.run_optuna_study(loaded).trials) == 5
    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", "synthetic_lifecycle", loaded.base_config["training"]["budget_epochs"], output_root=tmp_path
    )
    summary = json.loads((study_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["allocation"] == {"total": 5, "finished": 3, "pruned": 1, "failed": 1, "remaining": 0}


class _CreationCrashError(Exception):
    """Stand in for a kill while a new study is being initialized."""


def _crash_study_creation(monkeypatch: pytest.MonkeyPatch, loaded: optuna_runtime.OptunaStudyConfig, method: str, call: int) -> None:
    """Run one invocation that dies at the ``call``-th Optuna ``Study.<method>`` call, before that call writes."""
    original = getattr(optuna.study.Study, method)
    calls = [0]

    def crashing(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == call:
            raise _CreationCrashError
        return original(self, *args, **kwargs)

    with monkeypatch.context() as crash:
        crash.setattr(optuna.study.Study, method, crashing)
        with pytest.raises(_CreationCrashError):
            optuna_runtime.run_optuna_study(loaded, n_trials=1)


@pytest.mark.usefixtures("_steady_datasets_admitted")
@pytest.mark.parametrize(
    ("method", "call"),
    [("set_user_attr", 1), ("enqueue_trial", 1), ("set_user_attr", 2), ("set_user_attr", 4), ("set_user_attr", 7)],
    ids=["after-create", "after-start-record", "after-base-enqueued", "during-metadata", "before-complete-record"],
)
def test_an_interrupted_study_creation_completes_on_the_next_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, call: int
) -> None:
    """A creation killed at any boundary is completed under the lock by the retry, which runs the one queued base first."""
    loaded, _environment, _implementation = _stored_study(tmp_path, monkeypatch)
    monkeypatch.setattr(optuna_runtime, "create_objective", _sampling_factory)
    _crash_study_creation(monkeypatch, loaded, method, call)

    study = optuna_runtime.run_optuna_study(loaded, n_trials=1)

    base = {"hidden_channels": loaded.base_config["model"]["params"]["hidden_channels"], "learning_rate": loaded.base_config["optimizer"]["lr"]}
    assert [(trial.state, trial.params) for trial in study.trials] == [(optuna.trial.TrialState.COMPLETE, base)]
    assert study.user_attrs["initialization"] == {"state": "complete", "signature": study.user_attrs["semantic_signature"]}


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_an_interrupted_creation_of_another_signature_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry completes only the initialization it started: another signature names an inconsistent study and is refused."""
    loaded, _environment, _implementation = _stored_study(tmp_path, monkeypatch)
    _crash_study_creation(monkeypatch, loaded, "set_user_attr", 2)
    drifted = copy.deepcopy(loaded.base_config)
    drifted["optimizer"]["lr"] = 2.0e-3

    with pytest.raises(ValueError, match="interrupted initialization of another signature"):
        optuna_runtime.run_optuna_study(replace(loaded, base_config=experiments.config.loader.finalize_run_layout(drifted)), n_trials=1)


def test_a_trial_that_resolves_another_device_than_its_invocation_admitted_is_refused_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An auto policy that falls back to the CPU after the invocation admitted a GPU stops the trial before its run exists."""
    study = optuna_trials.with_configured_identities(optuna_runtime.with_runtime_overrides(_trial_study("steady_flow", tmp_path), device="auto"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def train(**_kwargs: Any) -> dict[str, Any]:
        pytest.fail("a trial on another device than the admitted one trained")

    _install_trial_harness("steady_flow", monkeypatch, train)
    gpu = {"device": "cuda:0", "device_model": "NVIDIA RTX 6000 Ada Generation"}
    with pytest.raises(RuntimeError, match="not the device its invocation admitted"):
        optuna_runtime.run_trial(study, _Trial(), sampler_seed=_SAMPLER_SEED, admitted_device=gpu)

    assert not _trial_run_dir(study).exists()


class _AnyScalingState(dict[str, Any]):
    """Answer every transient scaling key the run lifecycle copies into its completion evidence."""

    def __missing__(self, key: str) -> None:
        del key


class _SyntheticAdapter:
    """Stand in for the transient A0 training adapter whose controller a completed run records; teacher-forced A0 has no curriculum."""

    curriculum_state = None

    def controller_state(self) -> dict[str, Any]:
        return dict.fromkeys(experiments.run._TERMINAL_CONTROLLER_KEYS)


def _trial_study(task: str, tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load a compact test-owned study of either task on the CPU below a test-owned output root."""
    payload = _optuna_payload_with_dataset() if task == "steady_flow" else configs.optuna_transient_a0_config()
    loaded = optuna_runtime.load_optuna_study_config(configs.write_yaml(tmp_path / f"{task}-study.yaml", payload))
    return optuna_trials.with_configured_identities(optuna_runtime.with_runtime_overrides(loaded, device="cpu", output_root=tmp_path / "outputs"))


def _trial_run_dir(study: optuna_runtime.OptunaStudyConfig) -> Path:
    """Return the directory of trial 0, whose suggestions the fake trial fixes."""
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(), sampler_seed=_SAMPLER_SEED)
    return common.paths.resolve_config_run_dir(config)


def _install_trial_harness(task: str, monkeypatch: pytest.MonkeyPatch, train: Callable[..., dict[str, Any]]) -> None:
    """Replace data, model and training boundaries; the trial driver and the shared run lifecycle stay real."""
    channels = 3 if task == "steady_flow" else 4
    scaling = SimpleNamespace(
        state_dict=_AnyScalingState,
        physical_state=SimpleNamespace(scale=torch.ones(channels)),
        spatial_shape=(16, 16),
        digest="d" * 64,
        to=lambda _device: None,
    )
    # Both tasks form relative H1 metrics on the Train grid spacing.
    monkeypatch.setattr(experiments.run, "_grid_spacing", lambda *_args: (0.003, 0.003))
    if task != "steady_flow":
        monkeypatch.setattr(learning.models.factory, "validate_transient_model_spatial_shape", lambda *_args: None)
        monkeypatch.setattr(experiments.run.transient_adapter, "build_transient_training_adapter", lambda *_args, **_kwargs: _SyntheticAdapter())
    loaders = {
        "scaling": scaling,
        "scaling_artifact": {"synthetic": True},
        "split_indices": {},
        # A fresh unit sampler, as every run start gets; the replaced training never draws from it.
        "train": DataLoader(
            TensorDataset(torch.zeros(1)),
            batch_sampler=datasets.runtime.units.UnitSampler(unit_kind="case", cases=(0,), keys=None, units_per_update=1, memory_batch=1, seed=0),
        ),
        # The training loaders hold no held-out role: the run must not ask for one.
        "eval": None,
        "spatial_view": {},
    }

    def create_loaders(*_args: Any, split_persist_callback: Callable[[dict[str, Any]], None], **_kwargs: Any) -> dict[str, Any]:
        split_persist_callback({})
        return loaders

    monkeypatch.setattr(experiments.run, "configure_reproducibility", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiments.run, "seed_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiments.config.loader, "create_dataloaders_from_config", create_loaders)
    monkeypatch.setattr(learning.models.factory, "build_model", lambda *_args, **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(learning.losses.factory, "build_training_loss", lambda *_args, **_kwargs: SimpleNamespace(set_scaling=lambda _scaling: None))
    monkeypatch.setattr(learning.metrics.metrics, "build_evaluation_metrics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(learning.training.optim, "build_optimizer", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(learning.training.optim, "build_scheduler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(learning.training.checkpoint, "build_checkpoint_identity", lambda *_args, **_kwargs: {"effective_config_digest": "synthetic"})
    monkeypatch.setattr(experiments.tracking, "build_monitor_membership", lambda *_args: None)
    monkeypatch.setattr(learning.training.loop, "train_loop", train)
    monkeypatch.setattr(
        learning.training.loop,
        "evaluate_selected_checkpoint",
        lambda **_kwargs: {"selected_epoch": 1, "selected_metrics": {}, "terminal_epoch": 1, "terminal_metrics": {}},
    )


def _report(kwargs: Mapping[str, Any], point: int, value: float | None, *, epoch: int = 1, ends_epoch: bool = False) -> None:
    """Record one validation point in the run's history, then hand it to the event callback, as the loop does.

    A point without a value is not eligible and carries no objective. ``ends_epoch`` also records the
    epoch, as the event of an epoch end, a budget end or a divergence does. No transient one-step view
    accompanies the objective: it is a diagnostic, never a criterion of a trial.
    """
    objective_id = experiments.config.loader.get_resolved_objective(kwargs["config"])["id"]
    training = {"train/loss_total": 1.0, "epoch": float(epoch), "global_step": float(point)}
    metrics = {
        **training,
        "validation/point": float(point),
        "validation/eligible": float(value is not None),
        **({} if value is None else {f"id/{objective_id}": value}),
    }
    kwargs["epoch_state_callback"](epoch, training if ends_epoch else None, metrics)
    kwargs["event_callback"](point, metrics)


def _publish_best(digests: list[str], kwargs: Mapping[str, Any], epoch: int) -> None:
    """Rename the best of ``epoch`` into place, as the loop does before that epoch is reported, and keep its digest."""
    best = common.paths.resolve_best_checkpoint_file(kwargs["save_dir"])
    torch.save({"best_epoch": epoch, "best_metric": 0.5 / epoch, "completed_epoch": epoch, "global_step": epoch}, best)
    digests.append(hashlib.sha256(best.read_bytes()).hexdigest())


def _never_trains(_digests: list[str], **_kwargs: Any) -> dict[str, Any]:
    pytest.fail("a trial that failed before running must not train")


_PRUNED_WORK = {"gradient_predictions": 6, "optimizer_steps": 3, "units": 6, "update_distinct_cases": 6}


def _pruned(digests: list[str], **kwargs: Any) -> dict[str, Any]:
    _publish_best(digests, kwargs, 1)
    # A trial never resumes, so the loop makes every event's work durable before it reports the point that the study prunes.
    kwargs["attempt_work"].durable = {**_PRUNED_WORK, "train_seconds": 0.5}
    _report(kwargs, 1, 0.5, ends_epoch=True)
    pytest.fail("the pruned report must stop training")


def _nonfinite(_digests: list[str], **_kwargs: Any) -> dict[str, Any]:
    # A divergence ends training normally; a non-finite diagnostic (here the physics monitor) is a numerical failure.
    message = "Physics monitor 'momentum' is non-finite: nan."
    raise FloatingPointError(message)


def _allocator_oom(digests: list[str], **kwargs: Any) -> dict[str, Any]:
    _publish_best(digests, kwargs, 1)
    _report(kwargs, 1, 0.5, ends_epoch=True)
    message = "CUDA out of memory. Tried to allocate 2.00 GiB"
    raise RuntimeError(message)


def _allocator_oom_before_any_record(_digests: list[str], **_kwargs: Any) -> dict[str, Any]:
    message = "CUDA out of memory. Tried to allocate 2.00 GiB"
    raise RuntimeError(message)


def _system_exit(_digests: list[str], **_kwargs: Any) -> dict[str, Any]:
    raise SystemExit(3)


def _failure_after_an_unreported_best(digests: list[str], **kwargs: Any) -> dict[str, Any]:
    _publish_best(digests, kwargs, 1)
    _report(kwargs, 1, 0.5)
    _publish_best(digests, kwargs, 2)
    message = "synthetic failure after an unreported best"
    raise RuntimeError(message)


def _completes(digests: list[str], **kwargs: Any) -> dict[str, Any]:
    """Publish the epoch-1 best, overrun the run's budget by one unit, report its budget-end point and return the completed result."""
    epochs = int(kwargs["config"]["training"]["budget_epochs"])
    budget = kwargs["budget"]
    _publish_best(digests, kwargs, 1)
    # The last group overruns the budget; the overrun is still point 20, reported once. Each earlier epoch end has its record.
    budget.record_group(units=budget.total + 1, supervised_transitions=budget.total + 1, train_seconds=1.0)
    for epoch in range(1, epochs):
        kwargs["epoch_state_callback"](epoch, {"train/loss_total": 1.0, "epoch": float(epoch), "global_step": float(epoch)}, None)
    _report(kwargs, budget.record_validation(), 0.5, epoch=epochs, ends_epoch=True)
    return {
        "budget": budget.state_dict(),
        "status": "completed",
        "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
        "objective": experiments.config.loader.get_resolved_objective(kwargs["config"]),
        "best_metric": 0.5,
        "best_epoch": 1,
        "completed_epoch": epochs,
        "global_step": epochs,
        "training_work": {"optimizer_steps": epochs, "gradient_predictions": epochs},
        "best_checkpoint_path": common.paths.resolve_best_checkpoint_file(kwargs["save_dir"]),
    }


def _exhaust_host_memory(_seed: int, **_stage: Any) -> dict[str, int]:
    message = "synthetic host allocation failure before the run started"
    raise MemoryError(message)


def _trial_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Return the per-trial console lines of stdout and stderr."""
    return [line for line in "\n".join(capsys.readouterr()).splitlines() if line.startswith("trial ")]


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_error", "best_epoch", "reported_epoch", "recorded_epochs"),
    [
        pytest.param(None, "oom_pruned", optuna.TrialPruned, None, None, 0, id="host-oom-before-running"),
        pytest.param(_pruned, "pruned", optuna.TrialPruned, 1, 1, 1, id="pruned-after-an-epoch"),
        pytest.param(_nonfinite, "nonfinite_pruned", optuna.TrialPruned, None, None, 0, id="nonfinite"),
        pytest.param(_allocator_oom, "oom_pruned", optuna.TrialPruned, 1, 1, 1, id="oom-after-an-epoch"),
        pytest.param(_allocator_oom_before_any_record, "oom_pruned", optuna.TrialPruned, None, None, 0, id="oom-before-any-record"),
        pytest.param(_system_exit, "interrupted", SystemExit, None, None, 0, id="interrupted"),
        pytest.param(_failure_after_an_unreported_best, "failed", RuntimeError, 2, 1, 0, id="failed-after-an-unreported-best"),
    ],
)
def test_every_trial_outcome_publishes_one_terminal_record_for_both_tasks(
    task: str,
    scenario: Callable[..., dict[str, Any]] | None,
    expected_status: str,
    expected_error: type[BaseException],
    best_epoch: int | None,
    reported_epoch: int | None,
    recorded_epochs: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both tasks map an outcome to one status, bind the best evidence before the best goes, keep their history and print one trial line."""
    study = _trial_study(task, tmp_path)
    digests: list[str] = []
    _install_trial_harness(task, monkeypatch, functools.partial(scenario or _never_trains, digests))
    if scenario is None:
        monkeypatch.setattr(experiments.run, "build_seed_plan", _exhaust_host_memory)

    def cuda_query_forbidden() -> bool:
        pytest.fail("CPU cleanup must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_query_forbidden)
    trial = _Trial(prune=expected_status == "pruned")
    with pytest.raises(expected_error) as raised:
        optuna_runtime.run_trial(study, trial, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    run_dir = _trial_run_dir(study)
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == trial.attrs["terminal_status"] == expected_status
    assert summary["error_type"] == summary["failure_context"]["error_type"]
    assert {"started_at", "ended_at", "operational_timing", "error"} <= summary.keys()
    # An attempt that reached running publishes its end entry; a refusal before running records no attempt.
    assert summary.get("duration_coverage") == (None if scenario is None else "complete")
    # Every scenario reports at validation point 1 of epoch 1, so point and epoch coincide.
    assert summary["last_reported_point"] == summary["last_validation_epoch"] == reported_epoch
    assert summary["pruning_point"] == (reported_epoch if expected_status == "pruned" else None)
    assert (summary["best_epoch"], summary["completed_epoch"]) == (best_epoch, best_epoch)
    assert summary["best_checkpoint_sha256"] == (digests[-1] if digests else None)
    assert summary.get("training_work") == (_PRUNED_WORK if expected_status == "pruned" else None)
    assert not common.paths.resolve_best_checkpoint_file(run_dir).exists()
    assert "last_checkpoint_sha256" not in summary
    # The history keeps exactly the records written before the trial ended; one that ended before its first record has none.
    history = learning.training.history.history_path(run_dir)
    if reported_epoch is None:
        assert not history.exists()
    else:
        persisted = json.loads(history.read_text(encoding="utf-8"))
        assert [record["validation/point"] for record in persisted["validations"]] == [float(reported_epoch)]
        assert len(persisted["epochs"]) == recorded_epochs
    if expected_status == "nonfinite_pruned":
        assert "non-finite value" in str(raised.value)
    printed = "\n".join(capsys.readouterr())
    lines = [line for line in printed.splitlines() if line.startswith("trial ")]
    assert len(lines) == 1
    assert expected_status in lines[0]
    # A trial is never resumed or replaced: an interrupted one is advised what its study's rerun does.
    assert "restarts the run" not in printed
    assert ("reopens the study" in printed) is (expected_status == "interrupted")


@pytest.mark.parametrize("task", _TASKS)
def test_a_refused_sample_still_ends_with_a_status_and_one_trial_line(
    task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Trial setup runs inside the guarded attempt, so a refused sample is a failed trial with its one line."""
    study = _trial_study(task, tmp_path)

    def refuse(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        message = "synthetic refused sample"
        raise ValueError(message)

    monkeypatch.setattr(optuna_runtime, "_prepare_trial_config", refuse)
    trial = _Trial(number=4)
    with pytest.raises(ValueError, match="synthetic refused sample"):
        optuna_runtime.run_trial(study, trial, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert trial.attrs["terminal_status"] == "failed"
    lines = _trial_lines(capsys)
    assert len(lines) == 1
    assert lines[0].startswith("trial   4  failed")


@pytest.mark.parametrize("task", _TASKS)
def test_a_pruned_trial_prints_the_notes_of_its_lifecycle_record(
    task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pruned outcome prints no traceback, so a lifecycle violation noted while recording it still reaches stderr."""
    study = _trial_study(task, tmp_path)

    def train(**kwargs: Any) -> dict[str, Any]:
        _report(kwargs, 1, 0.5)
        message = "synthetic prune"
        raise optuna.TrialPruned(message)

    _install_trial_harness(task, monkeypatch, train)
    with pytest.raises(optuna.TrialPruned):
        optuna_runtime.run_trial(study, _Trial(), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert any(line.startswith("note: ") and "durable best checkpoint" in line for line in capsys.readouterr().err.splitlines())


@pytest.mark.parametrize("task", _TASKS)
def test_a_failing_evidence_record_never_replaces_the_primary_error(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported best without its best file is a lifecycle violation that is noted on the primary error, which still propagates."""
    study = _trial_study(task, tmp_path)
    primary = RuntimeError("synthetic primary failure")

    def train(**kwargs: Any) -> dict[str, Any]:
        _report(kwargs, 1, 0.5)
        raise primary

    _install_trial_harness(task, monkeypatch, train)
    with pytest.raises(RuntimeError) as raised:
        optuna_runtime.run_trial(study, _Trial(), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert raised.value is primary
    assert any("durable best checkpoint" in note for note in getattr(primary, "__notes__", ()))
    assert experiments.run.read_run_summary(_trial_run_dir(study))["status"] == "failed"


@pytest.mark.parametrize("task", _TASKS)
def test_completed_trial_summary_keeps_its_best_digest_while_the_file_goes_and_names_no_last(
    task: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A trial is never resumed and writes no last, so its summary keeps only the best digest while the best file is deleted."""
    study = _trial_study(task, tmp_path)
    digests: list[str] = []
    epochs = int(study.base_config["training"]["budget_epochs"])
    _install_trial_harness(task, monkeypatch, functools.partial(_completes, digests))
    trial = _Trial()

    assert optuna_runtime.run_trial(study, trial, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE) == pytest.approx(0.5)

    run_dir = _trial_run_dir(study)
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == trial.attrs["terminal_status"] == "completed"
    assert summary["best_checkpoint_sha256"] == digests[-1]
    assert (summary["last_reported_point"], summary["last_validation_epoch"], summary["failure_context"]) == (20, epochs, None)
    assert summary["terminal_budget"]["validated_point"] == 20
    assert summary["terminal_budget"]["units"] == summary["terminal_budget"]["total"] + 1
    # The work the study report shows per trial: the committed updates, the measured seconds and the trial's k.
    assert trial.attrs["work"] == {
        "optimizer_steps": epochs,
        "train_seconds": summary["attempted_train_seconds"],
        "units_per_update": study.base_config["training"]["units_per_update"],
    }
    assert set(summary["sampled_parameters"]) == {"hidden_channels", "learning_rate"}
    assert set(summary["overrides"]) == {"model.params.hidden_channels", "optimizer.lr"}
    assert trial.attrs["sampler_seed"] == _SAMPLER_SEED  # the invocation's derived seed, not the study seed
    assert "last_checkpoint" not in summary
    assert "last_checkpoint_sha256" not in summary
    assert not common.paths.resolve_best_checkpoint_file(run_dir).exists()
    assert common.paths.resolve_run_config_path(run_dir).is_file()
    lines = _trial_lines(capsys)
    assert len(lines) == 1
    assert "completed" in lines[0]


@pytest.mark.parametrize("task", _TASKS)
def test_a_trial_trains_and_selects_without_any_held_out_role(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training and the selected-checkpoint evaluation get only Train and validation; W&B and the summary name no held-out role."""
    study = _trial_study(task, tmp_path)
    received: dict[str, Mapping[str, Any]] = {}

    def train(**kwargs: Any) -> dict[str, Any]:
        received["train"] = kwargs
        return _completes([], **kwargs)

    _install_trial_harness(task, monkeypatch, train)
    select = learning.training.loop.evaluate_selected_checkpoint
    initialize = experiments.tracking.initialize_wandb

    def capture_selection(**kwargs: Any) -> dict[str, Any]:
        received["selected"] = kwargs
        return select(**kwargs)

    def capture_tracking(*args: Any, **kwargs: Any) -> Any:
        received["tracking"] = kwargs
        return initialize(*args, **kwargs)

    monkeypatch.setattr(learning.training.loop, "evaluate_selected_checkpoint", capture_selection)
    monkeypatch.setattr(experiments.tracking, "initialize_wandb", capture_tracking)

    assert optuna_runtime.run_trial(study, _Trial(), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE) == pytest.approx(0.5)

    assert "ood_loaders" not in received["train"]
    assert "ood_loaders" not in received["selected"]
    summary = common.paths.resolve_run_summary_path(_trial_run_dir(study)).read_text(encoding="utf-8")
    assert not any(f"{role}/" in summary for role in ("id_test", "material_ood", "parameter_ood"))


def _ends_early(outcome: str, value: float | None, digests: list[str], **kwargs: Any) -> dict[str, Any]:
    """Validate one point, eligible with ``value`` or not, and end training as ``outcome``: diverged at a third of the budget, else at its end."""
    budget = kwargs["budget"]
    diverged = outcome == "diverged"
    if value is not None:
        _publish_best(digests, kwargs, 1)
    spent = max(1, budget.total // 3) if diverged else budget.total
    budget.record_group(units=spent, supervised_transitions=spent, train_seconds=1.0)
    _report(kwargs, budget.record_validation(), value, ends_epoch=True)
    return {
        "budget": budget.state_dict(),
        "status": outcome,
        "training_outcome": {
            "status": outcome,
            "budget_progress": budget.progress,
            "eligible_points": int(value is not None),
            "divergence": {"kind": "synthetic"} if diverged else None,
        },
        "objective": experiments.config.loader.get_resolved_objective(kwargs["config"]),
        "best_metric": value,
        "best_epoch": None if value is None else 1,
        "completed_epoch": 1,
        "global_step": 1,
        "terminal_epoch": 1,
        "training_work": {"optimizer_steps": 1, "gradient_predictions": 1},
        "best_checkpoint_path": None if value is None else common.paths.resolve_best_checkpoint_file(kwargs["save_dir"]),
    }


# The suggestions of the fake trial, so an Optuna trial of the in-memory study trains in the directory ``_trial_run_dir`` names.
_FIRST_CHOICES = {"hidden_channels": 8, "learning_rate": 1.0e-4}


def _optimize_one(study_config: optuna_runtime.OptunaStudyConfig, study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Run one real Optuna trial of the first choices through the trial driver and return its stored record."""

    def objective(trial: optuna.Trial) -> float:
        # Optuna's trial number is a read-only property, which the driver's protocol states as an attribute.
        return optuna_runtime.run_trial(study_config, cast("Any", trial), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    study.enqueue_trial(_FIRST_CHOICES)
    study.optimize(objective, n_trials=1)
    return study.trials[-1]


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize(
    ("outcome", "value", "lifecycle"),
    [
        pytest.param("diverged", 0.4, "completed", id="diverged-after-an-eligible-point"),
        pytest.param("diverged", None, "unsuccessful", id="diverged-without-an-eligible-point"),
        pytest.param("no_eligible_checkpoint", None, "unsuccessful", id="no-eligible-point"),
    ],
)
def test_a_diverged_or_ineligible_trial_is_pruned_under_its_own_status(
    task: str,
    outcome: str,
    value: float | None,
    lifecycle: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A diverged trial is pruned with a violated constraint and an ineligible one without a value; both keep the run's own end state."""
    study_config = _trial_study(task, tmp_path)
    _install_trial_harness(task, monkeypatch, functools.partial(_ends_early, outcome, value, []))

    stored = _optimize_one(study_config, optuna.create_study(direction="minimize"))

    assert stored.state == optuna.trial.TrialState.PRUNED
    assert stored.user_attrs["terminal_status"] == outcome
    assert stored.constraints == ({"diverged": 1.0} if outcome == "diverged" else {})
    assert list(stored.intermediate_values.values()) == ([] if value is None else [value])
    assert stored.user_attrs["work"]["units_per_update"] == study_config.base_config["training"]["units_per_update"]
    summary = experiments.run.read_run_summary(_trial_run_dir(study_config))
    assert (summary["status"], summary["training_outcome"]["status"]) == (lifecycle, outcome)
    # The stage's end, a divergence included, recorded its one validation point and its epoch.
    persisted = json.loads(learning.training.history.history_path(_trial_run_dir(study_config)).read_text(encoding="utf-8"))
    assert [record["validation/point"] for record in persisted["validations"]] == [float(summary["terminal_budget"]["validated_point"])]
    assert [record["epoch"] for record in persisted["epochs"]] == [1.0]
    # Both are normal study outcomes: one line on stdout.
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("trial ")]
    assert len(lines) == 1
    assert outcome in lines[0]


def test_tpe_ranks_a_diverged_trial_behind_every_feasible_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TPE consumes the violated constraint: a diverged trial whose last point was the best still lands in the worse group."""
    study_config = _trial_study("steady_flow", tmp_path)
    _install_trial_harness("steady_flow", monkeypatch, functools.partial(_ends_early, "diverged", 0.01, []))
    study = optuna.create_study(direction="minimize")
    diverged = _optimize_one(study_config, study)
    for completed in (0.5, 0.6):
        study.add_trial(optuna.trial.create_trial(params=diverged.params, distributions=diverged.distributions, value=completed))
    study.add_trial(
        optuna.trial.create_trial(
            params=diverged.params, distributions=diverged.distributions, state=optuna.trial.TrialState.PRUNED, intermediate_values={1: 0.9}
        )
    )

    # Optuna's own split of the TPE sampler; without the constraint the diverged trial's later, lower point would rank it first among pruned trials.
    below, above = tpe_sampler._split_trials(study, study.get_trials(deepcopy=False), n_below=3)

    assert [trial.number for trial in above] == [diverged.number]
    assert len(below) == 3


def _study_report(study: optuna.Study, config: Any = None, *, include_importance: bool = False) -> dict[str, Any]:
    """Return the report of ``study`` under a study config, by default a minimal stand-in that names only what the report copies."""
    if config is None:
        study_settings = {"name": "s", "role": None, "sampler": {"kind": "tpe"}, "seed": 1, "pruner": {"kind": "none"}, "n_trials": 6}
        config = SimpleNamespace(
            study={**study_settings, "full_budget_epochs": 4, "budget_fraction": 0.5},
            base_config={"task": "steady_flow", "model": {"kind": "fno"}, "run": {"seed": 3}, "training": {"budget_epochs": 2}},
        )
    return optuna_runtime._study_lifecycle_summary(
        study,
        config=config,
        objective={"direction": study.direction.name.lower()},
        requested_trials=len(study.trials),
        reopened=False,
        status="completed",
        elapsed_seconds=3.0,
        include_importance=include_importance,
    )


def test_study_report_counts_every_status_and_takes_the_best_from_completed_trainings() -> None:
    """Duplicates, diverged and ineligible trials count apart; the best so far follows completed trainings; the seconds sum ended invocations."""
    study = optuna.create_study(direction="minimize")
    distribution: dict[str, optuna.distributions.BaseDistribution] = {"lr": optuna.distributions.FloatDistribution(1.0e-4, 1.0e-2)}
    work = {"optimizer_steps": 12, "train_seconds": 3.5, "units_per_update": 4}
    for status, value, state, constraints in (
        ("completed", 0.5, optuna.trial.TrialState.COMPLETE, None),
        ("duplicate", 0.5, optuna.trial.TrialState.COMPLETE, None),
        ("diverged", None, optuna.trial.TrialState.PRUNED, {"diverged": 1.0}),
        ("no_eligible_checkpoint", None, optuna.trial.TrialState.PRUNED, None),
        ("completed", 0.3, optuna.trial.TrialState.COMPLETE, None),
        ("duplicate_pruned", None, optuna.trial.TrialState.PRUNED, None),
    ):
        study.add_trial(
            optuna.trial.create_trial(
                state=state,
                value=value,
                params={"lr": 1.0e-3},
                distributions=distribution,
                user_attrs={"terminal_status": status, **({"work": work} if status != "duplicate" else {"duplicate_of": "trials/t0"})},
                constraints=constraints,
            )
        )
    study.set_user_attr("invocation/first", {"stored_trials": 0, "seconds": 2.0})
    study.set_user_attr("invocation/second", {"stored_trials": 2, "seconds": 3.0})
    study.set_user_attr("invocation/running", {"stored_trials": 4, "seconds": None})

    report = _study_report(study)

    assert {status: count for status, count in report["trial_counts"].items() if count} == {
        "completed": 2,
        "duplicate": 1,
        "diverged": 1,
        "no_eligible_checkpoint": 1,
        "duplicate_pruned": 1,
    }
    assert (report["best_trial_number"], report["best_objective"]) == (4, 0.3)
    assert [trial["best_objective_so_far"] for trial in report["trials"]] == [0.5, 0.5, 0.5, 0.5, 0.3, 0.3]
    assert [trial["work"] for trial in report["trials"]][:2] == [work, None]
    assert (report["study_seconds"], report["unrecorded_invocations"]) == (5.0, 1)


def test_a_duplicate_keeps_the_best_of_an_owner_whose_optuna_trial_did_not_complete(tmp_path: Path) -> None:
    """The owner's run was published but its Optuna trial crashed: the duplicate's stored value is the best and its config is written.

    A later trial of the same value keeps the earlier best.
    """
    study_config = optuna_runtime.with_runtime_overrides(_load(tmp_path), output_root=tmp_path)
    distributions: dict[str, optuna.distributions.BaseDistribution] = {
        "hidden_channels": optuna.distributions.CategoricalDistribution((_FIRST_CHOICES["hidden_channels"],)),
        "learning_rate": optuna.distributions.FloatDistribution(1.0e-4, 1.0e-2, log=True),
    }
    study = optuna.create_study(direction="minimize")
    for state, value, attrs in (
        (optuna.trial.TrialState.FAIL, None, {}),
        (optuna.trial.TrialState.COMPLETE, 0.4, {"terminal_status": "duplicate", "duplicate_of": "trials/t0"}),
        (optuna.trial.TrialState.COMPLETE, 0.4, {"terminal_status": "completed"}),
    ):
        study.add_trial(optuna.trial.create_trial(state=state, value=value, params=_FIRST_CHOICES, distributions=distributions, user_attrs=attrs))

    report = _study_report(study, study_config)

    assert (report["best_trial_number"], report["best_objective"], report["best_parameters"]) == (1, 0.4, _FIRST_CHOICES)
    assert [trial["best_objective_so_far"] for trial in report["trials"]] == [None, 0.4, 0.4]
    assert [report["trial_counts"][status] for status in ("failed", "duplicate", "completed")] == [1, 1, 1]
    paths = optuna_runtime._resolve_study_paths(study_config)
    paths.study_dir.mkdir(parents=True)
    optuna_runtime._publish_study_lifecycle_summary(paths, report, study_config)
    assert (paths.study_dir / "best_config.yaml").read_text(encoding="utf-8").startswith("# Best trial 1 of study ")


def test_parameter_importance_reads_only_the_trials_that_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplicate completes with a stored value but trained nothing, so it neither counts towards nor enters the importance input."""
    minimum = optuna_runtime._PARAMETER_IMPORTANCE_MIN_COMPLETED_TRIALS
    distribution: dict[str, optuna.distributions.BaseDistribution] = {"lr": optuna.distributions.FloatDistribution(1.0e-4, 1.0e-2)}
    study = optuna.create_study(direction="maximize")

    def add(value: float, **attrs: str) -> None:
        params = {"lr": 1.0e-4 * (1.0 + value)}
        user_attrs = {"terminal_status": "completed", **attrs}
        study.add_trial(optuna.trial.create_trial(value=value, params=params, distributions=distribution, user_attrs=user_attrs))

    for number in range(minimum - 1):
        add(float(number))
    add(0.5, terminal_status="duplicate", duplicate_of="trials/t0")
    inputs: list[optuna.Study] = []

    def importances(view: optuna.Study) -> dict[str, float]:
        inputs.append(view)
        return {"lr": 1.0}

    # Optuna exposes the module through a lazy proxy that its first use replaces, so patch the loaded module itself.
    monkeypatch.setattr(importlib.import_module("optuna.importance"), "get_param_importances", importances)

    assert _study_report(study, include_importance=True)["parameter_importance"]["status"] == "not_computed"
    add(float(minimum))
    assert _study_report(study, include_importance=True)["parameter_importance"]["values"] == {"lr": 1.0}
    (view,) = inputs
    assert view.direction == optuna.study.StudyDirection.MAXIMIZE
    trained = [(trial.value, trial.params) for trial in study.trials if "duplicate_of" not in trial.user_attrs]
    assert [(trial.value, trial.params) for trial in view.trials] == trained


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("secondary_type", [OSError, SystemExit])
def test_a_trial_whose_failure_evidence_is_lost_keeps_its_best_and_its_primary_error(
    task: str, secondary_type: type[BaseException], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence collection that fails, even by an interrupt, still publishes the status, keeps the unrecorded best and names why."""
    study = _trial_study(task, tmp_path)
    digests: list[str] = []
    primary = RuntimeError("synthetic primary failure")

    def train(**kwargs: Any) -> dict[str, Any]:
        _publish_best(digests, kwargs, 1)
        _report(kwargs, 1, 0.5)
        raise primary

    def lose_evidence(*_args: Any) -> dict[str, Any]:
        message = "synthetic unreadable best checkpoint"
        raise secondary_type(message)

    _install_trial_harness(task, monkeypatch, train)
    monkeypatch.setattr(optuna_runtime, "_checkpoint_progress", lose_evidence)
    trial = _Trial()
    with pytest.raises(RuntimeError) as raised:
        optuna_runtime.run_trial(study, trial, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    run_dir = _trial_run_dir(study)
    summary = experiments.run.read_run_summary(run_dir)
    assert raised.value is primary
    assert summary["status"] == trial.attrs["terminal_status"] == "failed"
    assert summary["terminal_evidence_error"].startswith(secondary_type.__name__)
    assert hashlib.sha256(common.paths.resolve_best_checkpoint_file(run_dir).read_bytes()).hexdigest() == digests[-1]


def test_an_interrupt_while_preserving_a_failure_becomes_a_note_on_the_primary_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A status resolver or a status publication that raises SystemExit leaves a note; the caller still raises the primary error."""
    primary = RuntimeError("synthetic primary failure")

    def interrupt(*_args: Any, **_kwargs: Any) -> str:
        raise SystemExit(3)

    assert experiments.run._failure_status(primary, interrupt) == "failed"
    monkeypatch.setattr(experiments.run, "transition_run_status", interrupt)
    experiments.run._mark_failure(tmp_path, primary, status="failed")

    notes = getattr(primary, "__notes__", ())
    assert len(notes) == 2
    assert all("SystemExit" in note for note in notes)


@pytest.mark.parametrize("task", _TASKS)
def test_a_host_oom_while_preparing_a_trial_run_publishes_the_trial_status(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh-run preparation classifies its failure with the trial's resolver, so the summary and Optuna agree."""
    study = _trial_study(task, tmp_path)

    def exhaust_host_memory(*_args: Any) -> None:
        message = "synthetic host allocation failure while saving the run config"
        raise MemoryError(message)

    monkeypatch.setattr(experiments.config.loader, "save_yaml", exhaust_host_memory)
    trial = _Trial()
    with pytest.raises(optuna.TrialPruned):
        optuna_runtime.run_trial(study, trial, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert experiments.run.read_run_summary(_trial_run_dir(study))["status"] == trial.attrs["terminal_status"] == "oom_pruned"


@pytest.mark.parametrize("task", _TASKS)
def test_a_tracking_finish_failure_after_a_completed_trial_only_warns(
    task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tracking never decides local validity: the completed run and trial keep their end state and the console warns."""
    study = _trial_study(task, tmp_path)
    _install_trial_harness(task, monkeypatch, functools.partial(_completes, []))

    def finish(**_kwargs: Any) -> None:
        message = "synthetic W&B finish failure"
        raise experiments.tracking.TrackingIOError(message)

    monkeypatch.setattr(experiments.tracking, "initialize_wandb", lambda *_args, **_kwargs: SimpleNamespace(enabled=False, finish=finish))
    trial = _Trial()

    assert optuna_runtime.run_trial(study, trial, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE) == pytest.approx(0.5)

    assert experiments.run.read_run_summary(_trial_run_dir(study))["status"] == trial.attrs["terminal_status"] == "completed"
    captured = capsys.readouterr()
    assert "synthetic W&B finish failure" in captured.err
    assert [line.split()[2] for line in (captured.out + captured.err).splitlines() if line.startswith("trial ")] == ["completed"]


class _AttributeLosingTrial(_Trial):
    """Fail to store the terminal status, as a lost study-storage write would."""

    def set_user_attr(self, key: str, value: Any) -> None:
        if key == "terminal_status":
            message = "synthetic study storage write failure"
            raise RuntimeError(message)
        super().set_user_attr(key, value)


@pytest.mark.parametrize(("scenario", "expected_error"), [(_system_exit, SystemExit), (_completes, RuntimeError)], ids=["interrupted", "completed"])
def test_a_lost_terminal_status_attribute_never_skips_the_trial_line_or_replaces_the_active_error(
    scenario: Callable[..., dict[str, Any]],
    expected_error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The trial line and cleanup always run; the attribute failure is a note on an active error and raised only without one."""
    study = _trial_study("steady_flow", tmp_path)
    _install_trial_harness("steady_flow", monkeypatch, functools.partial(scenario, []))
    with pytest.raises(expected_error) as raised:
        optuna_runtime.run_trial(study, _AttributeLosingTrial(), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert len(_trial_lines(capsys)) == 1
    if expected_error is SystemExit:
        assert any("synthetic study storage write failure" in note for note in getattr(raised.value, "__notes__", ()))
    else:
        assert "synthetic study storage write failure" in str(raised.value)


def test_a_completed_trial_raises_its_lost_attribute_even_inside_a_callers_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An error the caller is handling is not the trial's own: the lost attribute is raised, not noted on it."""
    study = _trial_study("steady_flow", tmp_path)
    _install_trial_harness("steady_flow", monkeypatch, functools.partial(_completes, []))
    outer = KeyError("synthetic error handled by the caller")

    def run_inside_handler() -> None:
        try:
            raise outer
        except KeyError:
            optuna_runtime.run_trial(study, _AttributeLosingTrial(), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    with pytest.raises(RuntimeError, match="synthetic study storage write failure"):
        run_inside_handler()
    assert not getattr(outer, "__notes__", ())
    assert len(_trial_lines(capsys)) == 1


def test_every_closing_step_runs_and_notes_its_failure_on_the_primary_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lost attribute and a broken trial line become notes on the primary SystemExit, and the cleanup still runs."""
    study = _trial_study("steady_flow", tmp_path)
    _install_trial_harness("steady_flow", monkeypatch, functools.partial(_system_exit, []))
    cleanups: list[int] = []

    def broken_line(*_args: Any, **_kwargs: Any) -> None:
        message = "synthetic closed console pipe"
        raise BrokenPipeError(message)

    def collect() -> int:
        cleanups.append(1)
        return 0

    monkeypatch.setattr(experiments.console, "optuna_trial_event", broken_line)
    monkeypatch.setattr(optuna_runtime.gc, "collect", collect)
    with pytest.raises(SystemExit) as raised:
        optuna_runtime.run_trial(study, _AttributeLosingTrial(), sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    notes = getattr(raised.value, "__notes__", ())
    assert any("synthetic study storage write failure" in note for note in notes)
    assert any("synthetic closed console pipe" in note for note in notes)
    assert cleanups


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (optuna_runtime.RecoverableTrialError("recoverable allocator out of memory"), "recoverable_failed"),
        (RuntimeError("CUDA out of memory"), "oom_pruned"),
        (SystemExit(3), "interrupted"),
    ],
)
def test_trial_outcomes_classify_recoverable_failures_before_allocator_text(error: BaseException, expected_status: str) -> None:
    """Explicit recoverable failures keep precedence over allocator-text RuntimeErrors; interrupts stay interrupts."""
    assert optuna_runtime._trial_terminal_status(error, optuna.TrialPruned) == expected_status


def test_public_run_statuses_distinguish_core_terminal_outcomes() -> None:
    """Keep completed, pruned, failed, and interrupted persistence distinct."""
    assert {"completed", "pruned", "failed", "interrupted"} <= (experiments.run.RUN_STATUSES)


def _run_with_heavy_files(tmp_path: Path, *, trial: bool) -> Path:
    """Allocate a running Optuna trial (best checkpoint only) or normal run (both checkpoints) with its split and normalizer."""
    config, _context = optuna_runtime._prepare_trial_config(_duplicate_study(tmp_path), _Trial(), sampler_seed=_SAMPLER_SEED)
    if trial:
        run_dir = experiments.run.prepare_fresh_run(config, run_dir=optuna_runtime._trial_run_dir(config))
    else:
        config["run"]["layout"]["trial"] = None
        run_dir = experiments.run.prepare_fresh_run(config, run_dir=tmp_path / "outputs" / "normal-run")
    experiments.run.transition_run_status(run_dir, "running")
    for path in (
        common.paths.resolve_split_indices_path(run_dir),
        common.paths.resolve_normalizer_path(run_dir),
        common.paths.resolve_best_checkpoint_file(run_dir),
    ):
        path.write_bytes(b"heavy")
    if not trial:
        common.paths.resolve_last_checkpoint_file(run_dir).write_bytes(b"heavy")
    return run_dir


@pytest.mark.parametrize(
    ("trial", "status", "files_remain"),
    [
        pytest.param(True, "completed", False, id="trial-completed"),
        pytest.param(True, "interrupted", False, id="trial-interrupted"),
        pytest.param(False, "completed", True, id="normal-run-completed"),
    ],
)
def test_run_lifecycle_frees_checkpoints_with_the_final_state_of_a_trial_only(
    trial: bool,
    status: str,
    files_remain: bool,
    tmp_path: Path,
) -> None:
    """The state transition that publishes a trial's final state deletes its checkpoints; nothing else does."""
    run_dir = _run_with_heavy_files(tmp_path, trial=trial)

    assert experiments.run.transition_run_status(run_dir, status)["status"] == status

    assert common.paths.resolve_best_checkpoint_file(run_dir).exists() is files_remain
    assert common.paths.resolve_last_checkpoint_file(run_dir).exists() is (not trial)
    assert common.paths.resolve_normalizer_path(run_dir).exists()


def _a0_study(tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load the maintained transient A0 study against an exact test-owned Dataset identity."""
    loaded = optuna_runtime.load_optuna_study_config(
        "configs/03_learning/transient_drying/optuna/fno_a0_search.yaml",
        train_dataset="synthetic_transient_train",
    )
    return optuna_trials.with_configured_identities(optuna_runtime.with_runtime_overrides(loaded, output_root=tmp_path))


def test_trial_layout_is_pure_and_nests_no_output_root(tmp_path: Path) -> None:
    """A trial directory follows its study, number and sampled values and is not allocated by preparing it."""
    study = _a0_study(tmp_path)
    study_dir = common.paths.resolve_study_dir(
        "transient_drying", "synthetic_transient_train", study.study["name"], study.base_config["training"]["budget_epochs"], output_root=tmp_path
    )

    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=7), sampler_seed=_SAMPLER_SEED)

    layout = config["run"]["layout"]
    assert Path(config["paths"]["output_root"]) == tmp_path
    assert (layout["family"], layout["config_name"], layout["stage"], layout["pair"]) == ("optuna", study.study["name"], "single", None)
    assert layout["trial"].startswith(f"0007__{layout['model_label']}__")
    assert config["run"]["name"] == f"{study.study['name']}__trial-0007"
    assert layout["budget_epochs"] == study.base_config["training"]["budget_epochs"]
    assert study_dir.name == f"{study.study['name']}__ep-{layout['budget_epochs']}"
    assert common.paths.resolve_config_run_dir(config) == study_dir / "trials" / layout["trial"]
    assert not study_dir.exists()  # Preparing a trial config allocates nothing.


def _publish_trial(study: optuna_runtime.OptunaStudyConfig, number: int, *, outcome: str | None) -> Path:
    """Allocate and register one trial directory whose run completed with a stored value and ``outcome``, or left it unfinished."""
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=number), sampler_seed=_SAMPLER_SEED)
    run_dir = experiments.run.prepare_fresh_run(config, run_dir=optuna_runtime._trial_run_dir(config))
    experiments.run.transition_run_status(run_dir, "running")
    if outcome is not None:
        experiments.run.transition_run_status(
            run_dir,
            "completed",
            updates={
                "best_metric": 0.5,
                "objective": dict(config["evaluation"]["objective"]),
                "completed_epoch": int(config["training"]["budget_epochs"]),
                "training_outcome": {
                    "status": outcome,
                    "budget_progress": 1.0 if outcome == "completed" else 0.5,
                    "eligible_points": 1,
                    "divergence": {"kind": "synthetic"} if outcome == "diverged" else None,
                },
            },
        )
    return run_dir


def _duplicate_study(tmp_path: Path, *, epochs: int | None = None) -> optuna_runtime.OptunaStudyConfig:
    """Load the artificial study under a test-owned output root, optionally with another epoch budget."""
    study = optuna_trials.with_configured_identities(optuna_runtime.with_runtime_overrides(_load(tmp_path), output_root=tmp_path / "outputs"))
    if epochs is None:
        return study
    base = copy.deepcopy(study.base_config)
    base["training"]["budget_epochs"] = epochs
    return replace(study, study={**study.study, "full_budget_epochs": epochs}, base_config=experiments.config.loader.finalize_run_layout(base))


def _admit_owners(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Admit each completed owner by its summary alone, since these owners hold no artifacts; return the admitted run paths."""
    admitted: list[Path] = []

    def admit(run_dir: Path) -> dict[str, Any]:
        admitted.append(Path(run_dir).resolve())
        return {"summary": experiments.run.read_run_summary(run_dir)}

    monkeypatch.setattr(experiments.run, "validate_completed_run", admit)
    return admitted


def test_duplicate_trial_of_a_completed_trial_reuses_its_value_without_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampler repeat of an admitted completed trial with the same epochs completes from the stored value, untrained."""
    study = _duplicate_study(tmp_path)
    original = _publish_trial(study, 0, outcome="completed")
    admitted = _admit_owners(monkeypatch)
    repeated = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(study, repeated, sampler_seed=_SAMPLER_SEED)
    owner = original.relative_to(tmp_path / "outputs").as_posix()

    assert optuna_runtime._duplicate_trial_outcome(config, repeated) == (pytest.approx(0.5), owner)
    assert admitted == [original.resolve()]
    assert repeated.attrs["duplicate_of"] == owner
    assert not optuna_runtime._trial_run_dir(config).exists()


def _refuse_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test as soon as a duplicate allocates a run, trains or resolves its device."""

    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a duplicate trial must not allocate, train or resolve its device")

    monkeypatch.setattr(optuna_runtime.experiments.run, "prepare_fresh_run", fail)
    monkeypatch.setattr(optuna_runtime.experiments.run, "execute_prepared_run", fail)
    monkeypatch.setattr(optuna_runtime.learning.device, "resolve_device", fail)


def test_run_trial_answers_a_repeated_trial_before_allocating_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The trial entry point returns a completed twin's value without allocating a run or contacting the device."""
    study = _duplicate_study(tmp_path)
    _publish_trial(study, 0, outcome="completed")
    _admit_owners(monkeypatch)
    repeated = _Trial(number=1)
    _refuse_training(monkeypatch)

    assert optuna_runtime.run_trial(study, repeated, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE) == pytest.approx(0.5)
    assert repeated.attrs["terminal_status"] == "duplicate"
    assert [line.split()[2] for line in _trial_lines(capsys)] == ["duplicate"]


def test_a_repeat_of_a_completed_trial_its_admission_refuses_fails_untrained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed owner that fails the completed-run admission (here: no split, normalizer or history) supplies no value."""
    study = _duplicate_study(tmp_path)
    _publish_trial(study, 0, outcome="completed")
    repeated = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=1), sampler_seed=_SAMPLER_SEED)
    _refuse_training(monkeypatch)

    with pytest.raises(experiments.run.RunLifecycleError, match="not loadable"):
        optuna_runtime.run_trial(study, repeated, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert repeated.attrs["terminal_status"] == "failed"
    assert not optuna_runtime._trial_run_dir(config).exists()


@pytest.mark.parametrize("owner_outcome", [None, "diverged"], ids=["unfinished-owner", "diverged-owner"])
def test_duplicate_trial_that_cannot_reuse_its_owner_is_pruned_without_training(
    owner_outcome: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat of an unfinished or diverged trial is pruned as a duplicate, the latter as infeasible as its owner; nothing is allocated."""
    study = _duplicate_study(tmp_path)
    original = _publish_trial(study, 0, outcome=owner_outcome)
    admitted = _admit_owners(monkeypatch)
    repeated = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=1), sampler_seed=_SAMPLER_SEED)

    with pytest.raises(optuna.TrialPruned, match="not trained again"):
        optuna_runtime.run_trial(study, repeated, sampler_seed=_SAMPLER_SEED, admitted_device=_CPU_DEVICE)

    assert repeated.attrs["terminal_status"] == "duplicate_pruned"
    assert repeated.attrs["duplicate_of"] == original.relative_to(tmp_path / "outputs").as_posix()
    assert repeated.constraints == ({} if owner_outcome is None else {"diverged": 1.0})
    # Only a completed owner (here the diverged one, which kept its best eligible point) passes the admission first.
    assert admitted == ([] if owner_outcome is None else [original.resolve()])
    assert "work" not in repeated.attrs
    assert not optuna_runtime._trial_run_dir(config).exists()


def test_trial_of_another_epoch_budget_is_not_a_duplicate(tmp_path: Path) -> None:
    """Identical values under another study epoch budget are another experiment: the trial trains itself."""
    _publish_trial(_duplicate_study(tmp_path), 0, outcome="completed")
    other = _duplicate_study(tmp_path, epochs=4)
    trial = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(other, trial, sampler_seed=_SAMPLER_SEED)
    original, _ = optuna_runtime._prepare_trial_config(_duplicate_study(tmp_path), _Trial(number=0), sampler_seed=_SAMPLER_SEED)

    assert experiments.config.loader.allocation_digest(config) != experiments.config.loader.allocation_digest(original)
    assert optuna_runtime._duplicate_trial_outcome(config, trial) is None


def test_new_trial_identity_is_not_a_duplicate_and_registers_as_a_trial(tmp_path: Path) -> None:
    """A trial whose science is unclaimed is trained; once allocated its identity is owned by the trial kind."""
    study = _duplicate_study(tmp_path)
    trial = _Trial(number=0)
    config, _context = optuna_runtime._prepare_trial_config(study, trial, sampler_seed=_SAMPLER_SEED)

    assert optuna_runtime._duplicate_trial_outcome(config, trial) is None
    _publish_trial(study, 0, outcome=None)

    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert entry.kind == "trial"
