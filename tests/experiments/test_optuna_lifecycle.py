"""Protect Optuna persistence, resume, pruning, and terminal-state behavior.

Responsibilities:
  - Check study persistence, resume, pruning, signatures, and transient trial placement
  - Check that one trial driver publishes the same terminal record, Optuna attribute and trial line for both tasks
  - Check that a failure while recording a trial's evidence or status, even an interrupt, never replaces its primary error
    and that a trial whose evidence was lost keeps its best checkpoint
  - Check that a failed run preparation, a lost status attribute and a tracking finish failure keep one agreeing end state
  - Check that a finished trial frees its checkpoint files while its summary keeps the best digest
  - Check that a trial repeating an existing experiment is reused or pruned without training
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
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
import torch
from support import configs, optuna_trials
from support.synthetic_task import admitted_manifest

from src import common, datasets, domain, experiments, learning

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

optuna_runtime = experiments.tuning.optuna
_EXPECTED_GLOBAL_STEP = 7
_EXPECTED_RESUMED_TRIAL_COUNT = 4
_TASK = domain.tasks.registry.get_task("steady_flow")
_OBJECTIVE_ID = next(metric.id for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_TASKS = ("steady_flow", "transient_drying")


class _Trial:
    """Implement the small trial surface needed by reporter and failure tests."""

    def __init__(self, *, number: int = 0, prune: bool = False) -> None:
        self.number = number
        self.prune = prune
        self.attrs: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []

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
    operational_study.update(
        {
            "name": "renamed_display",
            "n_trials": 99,
            "storage": "sqlite:///relocated.db",
        }
    )
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


def test_reporter_uses_completed_epochs_and_prunes_immediately() -> None:
    """Reject missing/discontinuous evidence and stop on the pruning epoch."""
    missing = optuna_runtime.OptunaEpochReporter(
        trial=_Trial(),
        objective_id=_OBJECTIVE_ID,
        direction="minimize",
    )
    with pytest.raises(KeyError, match="Held-out Optuna objective"):
        missing(1, {"train/loss_total": 0.1})

    discontinuous = optuna_runtime.OptunaEpochReporter(
        trial=_Trial(),
        objective_id=_OBJECTIVE_ID,
        direction="minimize",
    )
    with pytest.raises(ValueError, match="Expected 1, received 2"):
        discontinuous(2, {f"id/{_OBJECTIVE_ID}": 0.5})

    trial = _Trial(prune=True)
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=trial,
        objective_id=_OBJECTIVE_ID,
        direction="minimize",
    )
    with pytest.raises(optuna.TrialPruned, match="completed epoch 1"):
        reporter(
            1,
            {
                f"id/{_OBJECTIVE_ID}": 0.5,
                "global_step": float(_EXPECTED_GLOBAL_STEP),
            },
        )
    assert trial.reports == [(0.5, 1)]
    assert trial.attrs["last_reported_epoch"] == 1
    assert trial.attrs["last_global_step"] == _EXPECTED_GLOBAL_STEP


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_existing_unbound_study_fails_before_trial_allocation(
    tmp_path: Path,
) -> None:
    """Reject a pre-existing SQLite study without semantic metadata."""
    loaded = optuna_runtime.with_runtime_overrides(
        _load(tmp_path),
        device="cpu",
        output_root=tmp_path,
    )
    study_name = loaded.study["name"]
    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", study_name, loaded.base_config["training"]["epochs"], output_root=tmp_path
    )
    study_dir.mkdir(parents=True)
    storage = f"sqlite:///{study_dir / (study_name + '.db')}"
    optuna.create_study(
        study_name=f"steady_flow/synthetic_train/{study_dir.name}",
        direction="minimize",
        storage=storage,
    )

    with pytest.raises(
        ValueError,
        match="missing required semantic metadata",
    ):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert not list(study_dir.glob("trial_*"))


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_a_system_exit_inside_a_trial_leaves_no_running_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Optuna finalizes only Exception and KeyboardInterrupt, so the study records a trial that SystemExit ended as failed."""
    loaded = optuna_runtime.with_runtime_overrides(_load(tmp_path), device="cpu", output_root=tmp_path)

    def exiting_factory(_config: optuna_runtime.OptunaStudyConfig) -> Callable[[Any], float]:
        def objective(_trial: Any) -> float:
            raise SystemExit(3)

        return objective

    monkeypatch.setattr(optuna_runtime, "create_objective", exiting_factory)
    _slurm_job_in(monkeypatch, "a")
    monkeypatch.setenv("JOB_LOG_OWNERS_FILE", str(tmp_path / "owners"))
    with pytest.raises(SystemExit):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)

    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", loaded.study["name"], loaded.base_config["training"]["epochs"], output_root=tmp_path
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
        "steady_flow", "synthetic_train", loaded.study["name"], loaded.base_config["training"]["epochs"], output_root=tmp_path
    )
    sessions = study_dir / experiments.source_binding.STUDY_SESSIONS_FILENAME

    def crashing_factory(_config: optuna_runtime.OptunaStudyConfig) -> Callable[[Any], float]:
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


@pytest.mark.usefixtures("_steady_datasets_admitted")
def test_sqlite_study_persists_states_and_resumes_new_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve completed history while adding failed and pruned trial states."""
    environment: dict[str, str | None] = {
        "python": "3.12.1",
        "torch": "2.5.0",
        "neuraloperator_implementation_sha256": "a" * 64,
        "sif_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        optuna_runtime.learning.training.checkpoint,
        "numerical_environment_identity",
        lambda: dict(environment),
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
    base["training"]["epochs"] = 10
    base["training"]["evaluation_interval"] = 5
    base["training"]["ood_evaluation_interval"] = 5
    base["tracking"]["wandb"]["monitor"]["interval"] = 5
    base = experiments.config.loader.finalize_run_layout(base)
    settings = copy.deepcopy(loaded.study)
    settings["name"] = "synthetic_lifecycle"
    settings["pruner"] = {
        "kind": "median",
        "n_startup_trials": 0,
        "n_warmup_steps": 5,
        "interval_steps": 5,
    }
    loaded = optuna_runtime.with_runtime_overrides(
        replace(loaded, base_config=base, study=settings),
        device="cpu",
        output_root=tmp_path,
    )

    def synthetic_factory(
        config: optuna_runtime.OptunaStudyConfig,
    ) -> Callable[[Any], float]:
        def objective(trial: Any) -> float:
            experiments.tuning.search_space.suggest_trial_overrides(trial, config.search_space)
            if trial.number == 1:
                message = "recoverable synthetic failure"
                raise optuna_runtime.RecoverableTrialError(message)
            reporter = optuna_runtime.OptunaEpochReporter(
                trial=trial,
                objective_id=_OBJECTIVE_ID,
                direction="minimize",
                evaluation_interval=5,
                target_epoch=10,
            )
            values = (0.10, 0.09) if trial.number == 0 else (1.0, 0.9)
            for epoch, value in zip((5, 10), values, strict=True):
                reporter(
                    epoch,
                    {
                        f"id/{_OBJECTIVE_ID}": value,
                        "global_step": float(epoch),
                    },
                )
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
    assert resumed.trials[0].intermediate_values == {5: 0.10, 10: 0.09}
    assert resumed.trials[2].intermediate_values == {5: 1.0}

    study_dir = common.paths.resolve_study_dir(
        "steady_flow", "synthetic_train", "synthetic_lifecycle", loaded.base_config["training"]["epochs"], output_root=tmp_path
    )
    summary = json.loads((study_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert [trial["number"] for trial in summary["trials"]] == [0, 1, 2]
    assert summary["training_seed"] == loaded.base_config["run"]["seed"]
    assert summary["sampler"]["seed"] == loaded.study["seed"]
    assert (study_dir / "best_config.yaml").is_file()

    def unexpected_factory(
        _config: optuna_runtime.OptunaStudyConfig,
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
    budget_drift["training"]["epochs"] = 11
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
    }
    for key, changed_value in (
        ("torch", "2.5.1"),
        ("neuraloperator_implementation_sha256", "c" * 64),
        ("sif_sha256", "c" * 64),
        ("sif_sha256", None),
    ):
        environment[key] = changed_value
        with pytest.raises(ValueError, match="numerical_environment"):
            optuna_runtime.run_optuna_study(loaded, n_trials=1)
        assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT
        environment[key] = persisted_environment[key]

    optuna_source.write_text('def objective_policy():\n    return "maximize"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT
    optuna_source.write_text('def objective_policy():\n    return "minimize"\n', encoding="utf-8")
    search_source.write_text('def search_policy():\n    return "integer"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert len(optuna.load_study(study_name=_STUDY_ID, storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT


class _AnyScalingState(dict[str, Any]):
    """Answer every transient scaling key the run lifecycle copies into its completion evidence."""

    def __missing__(self, key: str) -> None:
        del key


class _SyntheticAdapter:
    """Stand in for the transient training adapter whose budget and curriculum a completed run records."""

    curriculum_state = SimpleNamespace(active_stage=0, max_horizon=1, draw_index=0)

    def budget_state(self) -> dict[str, Any]:
        return dict.fromkeys(experiments.run._TERMINAL_CONTROLLER_KEYS)


def _trial_study(task: str, tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load a compact test-owned study of either task on the CPU below a test-owned output root."""
    payload = _optuna_payload_with_dataset() if task == "steady_flow" else configs.optuna_transient_a0_config()
    loaded = optuna_runtime.load_optuna_study_config(configs.write_yaml(tmp_path / f"{task}-study.yaml", payload))
    return optuna_trials.with_configured_identities(optuna_runtime.with_runtime_overrides(loaded, device="cpu", output_root=tmp_path / "outputs"))


def _trial_run_dir(study: optuna_runtime.OptunaStudyConfig) -> Path:
    """Return the directory of trial 0, whose suggestions the fake trial fixes."""
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial())
    return common.paths.resolve_config_run_dir(config)


def _install_trial_harness(task: str, monkeypatch: pytest.MonkeyPatch, train: Callable[..., dict[str, Any]]) -> None:
    """Replace data, model and training boundaries; the trial driver and the shared run lifecycle stay real."""
    processor: Any
    if task == "steady_flow":
        processor = SimpleNamespace(in_normalizer=None, out_normalizer=SimpleNamespace(std=torch.ones(3)), to=lambda _device: None)
        monkeypatch.setattr(experiments.run, "_steady_grid_spacing", lambda *_args: (0.003, 0.003))
    else:
        processor = SimpleNamespace(state_dict=_AnyScalingState, state_std=torch.ones(4), spatial_shape=(16, 16), digest="d" * 64)
        monkeypatch.setattr(learning.models.factory, "validate_transient_model_spatial_shape", lambda *_args: None)
        monkeypatch.setattr(experiments.run.transient_adapter, "build_transient_training_adapter", lambda *_args, **_kwargs: _SyntheticAdapter())
    loaders = {
        "data_processor": processor,
        "scaling_artifact": {"synthetic": True},
        "split_indices": {},
        "train": [None],
        "eval": None,
        "material_ood": None,
        "parameter_ood": None,
        "spatial_view": {},
    }

    def create_loaders(*_args: Any, split_persist_callback: Callable[[dict[str, Any]], None], **_kwargs: Any) -> dict[str, Any]:
        split_persist_callback({})
        return loaders

    monkeypatch.setattr(experiments.run, "configure_reproducibility", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiments.run, "seed_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiments.config.loader, "create_dataloaders_from_config", create_loaders)
    monkeypatch.setattr(learning.models.factory, "build_model", lambda *_args, **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(learning.losses.factory, "build_training_loss", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(learning.metrics.metrics, "build_evaluation_metrics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(learning.training.optim, "build_optimizer", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(learning.training.optim, "build_scheduler", lambda *_args: None)
    monkeypatch.setattr(learning.training.checkpoint, "build_checkpoint_identity", lambda *_args, **_kwargs: {"effective_config_digest": "synthetic"})
    monkeypatch.setattr(experiments.tracking, "build_monitor_membership", lambda *_args: None)
    monkeypatch.setattr(learning.training.loop, "train_loop", train)
    monkeypatch.setattr(
        learning.training.loop,
        "evaluate_selected_checkpoint",
        lambda **_kwargs: {"selected_epoch": 1, "selected_metrics": {}, "terminal_epoch": 1, "terminal_metrics": {}},
    )


def _report(kwargs: Mapping[str, Any], epoch: int, value: float) -> None:
    """Hand one evaluated epoch to the run's epoch callback, with the transient one-step guardrail beside the objective."""
    objective_id = experiments.config.loader.get_resolved_objective(kwargs["config"])["id"]
    metrics = {
        "train/loss_total": 1.0,
        f"id/{objective_id}": value,
        f"id/guardrail/one_step/{objective_id}": value,
        "global_step": float(epoch),
    }
    kwargs["epoch_end_callback"](epoch, metrics)


def _publish_best(digests: list[str], kwargs: Mapping[str, Any], epoch: int) -> None:
    """Rename the best of ``epoch`` into place, as the loop does before that epoch is reported, and keep its digest."""
    best = common.paths.resolve_best_checkpoint_file(kwargs["save_dir"])
    torch.save({"best_epoch": epoch, "best_metric": 0.5 / epoch, "completed_epoch": epoch, "global_step": epoch}, best)
    digests.append(hashlib.sha256(best.read_bytes()).hexdigest())


def _never_trains(_digests: list[str], **_kwargs: Any) -> dict[str, Any]:
    pytest.fail("a trial that failed before running must not train")


_PRUNED_WORK = {"optimizer_steps": 3, "gradient_predictions": 6}


def _pruned(digests: list[str], **kwargs: Any) -> dict[str, Any]:
    _publish_best(digests, kwargs, 1)
    # The loop keeps the caller's counts current before it reports the epoch that the study prunes.
    kwargs["training_work"].update(_PRUNED_WORK)
    _report(kwargs, 1, 0.5)
    pytest.fail("the pruned report must stop training")


def _nonfinite(_digests: list[str], **kwargs: Any) -> dict[str, Any]:
    _report(kwargs, 1, float("nan"))
    pytest.fail("a non-finite report must stop training")


def _allocator_oom(digests: list[str], **kwargs: Any) -> dict[str, Any]:
    _publish_best(digests, kwargs, 1)
    _report(kwargs, 1, 0.5)
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
    """Publish the epoch-1 best, report every epoch and return the completed training result."""
    epochs = int(kwargs["config"]["training"]["epochs"])
    _publish_best(digests, kwargs, 1)
    for epoch in range(1, epochs + 1):
        _report(kwargs, epoch, 0.5 + epoch - 1)
    return {
        "status": "completed",
        "objective": experiments.config.loader.get_resolved_objective(kwargs["config"]),
        "best_metric": 0.5,
        "best_epoch": 1,
        "completed_epoch": epochs,
        "global_step": epochs,
        "training_work": {"optimizer_steps": epochs, "gradient_predictions": epochs},
        "best_checkpoint_path": common.paths.resolve_best_checkpoint_file(kwargs["save_dir"]),
    }


def _exhaust_host_memory(_seed: int) -> dict[str, int]:
    message = "synthetic host allocation failure before the run started"
    raise MemoryError(message)


def _trial_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Return the per-trial console lines of stdout and stderr."""
    return [line for line in "\n".join(capsys.readouterr()).splitlines() if line.startswith("trial ")]


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_error", "best_epoch", "reported_epoch"),
    [
        pytest.param(None, "oom_pruned", optuna.TrialPruned, None, None, id="host-oom-before-running"),
        pytest.param(_pruned, "pruned", optuna.TrialPruned, 1, 1, id="pruned"),
        pytest.param(_nonfinite, "nonfinite_pruned", optuna.TrialPruned, None, None, id="nonfinite"),
        pytest.param(_allocator_oom, "oom_pruned", optuna.TrialPruned, 1, 1, id="oom"),
        pytest.param(_system_exit, "interrupted", SystemExit, None, None, id="interrupted"),
        pytest.param(_failure_after_an_unreported_best, "failed", RuntimeError, 2, 1, id="failed-after-an-unreported-best"),
    ],
)
def test_every_trial_outcome_publishes_one_terminal_record_for_both_tasks(
    task: str,
    scenario: Callable[..., dict[str, Any]] | None,
    expected_status: str,
    expected_error: type[BaseException],
    best_epoch: int | None,
    reported_epoch: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both tasks map an outcome to one status, bind the best evidence before the best goes, and print one trial line."""
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
        optuna_runtime.run_trial(study, trial)

    run_dir = _trial_run_dir(study)
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == trial.attrs["terminal_status"] == expected_status
    assert summary["error_type"] == summary["failure_context"]["error_type"]
    assert {"started_at", "ended_at", "elapsed_seconds", "duration_coverage", "operational_timing", "error"} <= summary.keys()
    assert summary["last_reported_epoch"] == reported_epoch
    assert summary["pruning_epoch"] == (reported_epoch if expected_status == "pruned" else None)
    assert (summary["best_epoch"], summary["completed_epoch"]) == (best_epoch, best_epoch)
    assert summary["best_checkpoint_sha256"] == (digests[-1] if digests else None)
    assert summary.get("training_work") == (_PRUNED_WORK if expected_status == "pruned" else None)
    assert not common.paths.resolve_best_checkpoint_file(run_dir).exists()
    assert "last_checkpoint_sha256" not in summary
    if expected_status == "nonfinite_pruned":
        assert "non-finite value" in str(raised.value)
    lines = _trial_lines(capsys)
    assert len(lines) == 1
    assert expected_status in lines[0]


@pytest.mark.parametrize("task", _TASKS)
def test_a_refused_sample_still_ends_with_a_status_and_one_trial_line(
    task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Trial setup runs inside the guarded attempt, so a refused sample is a failed trial with its one line."""
    study = _trial_study(task, tmp_path)

    def refuse(*_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        message = "synthetic refused sample"
        raise ValueError(message)

    monkeypatch.setattr(optuna_runtime, "_prepare_trial_config", refuse)
    trial = _Trial(number=4)
    with pytest.raises(ValueError, match="synthetic refused sample"):
        optuna_runtime.run_trial(study, trial)

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
        optuna_runtime.run_trial(study, _Trial())

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
        optuna_runtime.run_trial(study, _Trial())

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
    epochs = int(study.base_config["training"]["epochs"])
    _install_trial_harness(task, monkeypatch, functools.partial(_completes, digests))
    trial = _Trial()

    assert optuna_runtime.run_trial(study, trial) == pytest.approx(0.5)

    run_dir = _trial_run_dir(study)
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == trial.attrs["terminal_status"] == "completed"
    assert summary["best_checkpoint_sha256"] == digests[-1]
    assert (summary["last_reported_epoch"], summary["failure_context"]) == (epochs, None)
    assert set(summary["sampled_parameters"]) == {"hidden_channels", "learning_rate"}
    assert set(summary["overrides"]) == {"model.params.hidden_channels", "optimizer.lr"}
    assert "last_checkpoint" not in summary
    assert "last_checkpoint_sha256" not in summary
    assert not common.paths.resolve_best_checkpoint_file(run_dir).exists()
    assert common.paths.resolve_run_config_path(run_dir).is_file()
    lines = _trial_lines(capsys)
    assert len(lines) == 1
    assert "completed" in lines[0]


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
        optuna_runtime.run_trial(study, trial)

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
        optuna_runtime.run_trial(study, trial)

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

    assert optuna_runtime.run_trial(study, trial) == pytest.approx(0.5)

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
        optuna_runtime.run_trial(study, _AttributeLosingTrial())

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
            optuna_runtime.run_trial(study, _AttributeLosingTrial())

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
        optuna_runtime.run_trial(study, _AttributeLosingTrial())

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
    config, _context = optuna_runtime._prepare_trial_config(_duplicate_study(tmp_path), _Trial())
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
        "transient_drying", "synthetic_transient_train", study.study["name"], study.base_config["training"]["epochs"], output_root=tmp_path
    )

    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=7))

    layout = config["run"]["layout"]
    assert Path(config["paths"]["output_root"]) == tmp_path
    assert (layout["family"], layout["config_name"], layout["stage"], layout["pair"]) == ("optuna", study.study["name"], "single", None)
    assert layout["trial"].startswith(f"0007__{layout['model_label']}__")
    assert config["run"]["name"] == f"{study.study['name']}__trial-0007"
    assert layout["budget_epochs"] == study.base_config["training"]["epochs"]
    assert study_dir.name == f"{study.study['name']}__ep-{layout['budget_epochs']}"
    assert common.paths.resolve_config_run_dir(config) == study_dir / "trials" / layout["trial"]
    assert not study_dir.exists()  # Preparing a trial config allocates nothing.


def _publish_trial(study: optuna_runtime.OptunaStudyConfig, number: int, *, completed: bool, extra_epochs: int = 0) -> Path:
    """Allocate and register one trial directory, completed with a stored objective value or left unfinished."""
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=number))
    run_dir = experiments.run.prepare_fresh_run(config, run_dir=optuna_runtime._trial_run_dir(config))
    experiments.run.transition_run_status(run_dir, "running")
    if completed:
        experiments.run.transition_run_status(
            run_dir,
            "completed",
            updates={
                "best_metric": 0.5,
                "objective": dict(config["evaluation"]["objective"]),
                "completed_epoch": int(config["training"]["epochs"]) + extra_epochs,
            },
        )
    return run_dir


def _duplicate_study(tmp_path: Path, *, epochs: int | None = None) -> optuna_runtime.OptunaStudyConfig:
    """Load the artificial study under a test-owned output root, optionally with another epoch budget."""
    study = optuna_trials.with_configured_identities(optuna_runtime.with_runtime_overrides(_load(tmp_path), output_root=tmp_path / "outputs"))
    if epochs is None:
        return study
    base = copy.deepcopy(study.base_config)
    base["training"]["epochs"] = epochs
    return replace(study, base_config=experiments.config.loader.finalize_run_layout(base))


def test_duplicate_trial_of_a_completed_trial_reuses_its_value_without_training(tmp_path: Path) -> None:
    """A sampler repeat of a completed trial with the same epochs completes from the stored value, untrained."""
    study = _duplicate_study(tmp_path)
    original = _publish_trial(study, 0, completed=True)
    repeated = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(study, repeated)
    owner = original.relative_to(tmp_path / "outputs").as_posix()

    assert optuna_runtime._duplicate_trial_outcome(config, repeated) == (pytest.approx(0.5), owner)
    assert repeated.attrs["duplicate_of"] == owner
    assert not optuna_runtime._trial_run_dir(config).exists()


def test_run_trial_answers_a_repeated_trial_before_allocating_or_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The trial entry point returns a completed twin's value without allocating a run or contacting the device."""
    study = _duplicate_study(tmp_path)
    _publish_trial(study, 0, completed=True)
    repeated = _Trial(number=1)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a duplicate trial must not allocate, train or resolve its device")

    monkeypatch.setattr(optuna_runtime.experiments.run, "prepare_fresh_run", fail)
    monkeypatch.setattr(optuna_runtime.experiments.run, "execute_prepared_run", fail)
    monkeypatch.setattr(optuna_runtime.learning.device, "resolve_device", fail)

    assert optuna_runtime.run_trial(study, repeated) == pytest.approx(0.5)
    assert repeated.attrs["terminal_status"] == "duplicate"
    assert [line.split()[2] for line in _trial_lines(capsys)] == ["duplicate"]


@pytest.mark.parametrize("scenario", ["unfinished", "extended_after_training"])
def test_duplicate_trial_that_cannot_reuse_its_owner_is_pruned_without_training(tmp_path: Path, scenario: str) -> None:
    """A repeat of an unfinished or later-extended trial is pruned as a duplicate; nothing is allocated."""
    study = _duplicate_study(tmp_path)
    original = _publish_trial(
        study,
        0,
        completed=scenario != "unfinished",
        extra_epochs=3 if scenario == "extended_after_training" else 0,
    )
    repeated = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=1))

    with pytest.raises(optuna.TrialPruned, match="not trained again"):
        optuna_runtime.run_trial(study, repeated)

    assert repeated.attrs["terminal_status"] == "duplicate_pruned"
    assert repeated.attrs["duplicate_of"] == original.relative_to(tmp_path / "outputs").as_posix()
    assert not optuna_runtime._trial_run_dir(config).exists()


def test_trial_of_another_epoch_budget_is_not_a_duplicate(tmp_path: Path) -> None:
    """Identical values under another study epoch budget are another experiment: the trial trains itself."""
    _publish_trial(_duplicate_study(tmp_path), 0, completed=True)
    other = _duplicate_study(tmp_path, epochs=4)
    trial = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(other, trial)
    original, _ = optuna_runtime._prepare_trial_config(_duplicate_study(tmp_path), _Trial(number=0))

    assert experiments.config.loader.allocation_digest(config) != experiments.config.loader.allocation_digest(original)
    assert optuna_runtime._duplicate_trial_outcome(config, trial) is None


def test_new_trial_identity_is_not_a_duplicate_and_registers_as_a_trial(tmp_path: Path) -> None:
    """A trial whose science is unclaimed is trained; once allocated its identity is owned by the trial kind."""
    study = _duplicate_study(tmp_path)
    trial = _Trial(number=0)
    config, _context = optuna_runtime._prepare_trial_config(study, trial)

    assert optuna_runtime._duplicate_trial_outcome(config, trial) is None
    _publish_trial(study, 0, completed=False)

    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert entry.kind == "trial"
