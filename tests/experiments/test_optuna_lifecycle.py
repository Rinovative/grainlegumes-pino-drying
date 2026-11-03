"""Protect Optuna persistence, resume, pruning, and terminal-state behavior.

Responsibilities:
  - Check study persistence, resume, pruning, terminal states, and transient trial placement
  - Check that a finished trial keeps its checkpoint digests in the summary, frees the checkpoint files, and validates without them
  - Check that a trial repeating an existing experiment is reused or pruned without training
  - Check that a study records its Slurm source binding independently of its trials and is continued only in that tree
  - Check that a study records its root as the owner of the job's log links

Design principles:
  - Scientific signatures and recoverable failures retain exact lifecycle meaning

This module does NOT:
  - Select production search spaces or contact a live study service
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
import torch
from support import configs

from src import common, domain, experiments

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

optuna_runtime = experiments.tuning.optuna
_EXPECTED_GLOBAL_STEP = 7
_EXPECTED_RESUMED_TRIAL_COUNT = 4
_TASK = domain.tasks.registry.get_task("steady_flow")
_OBJECTIVE_ID = next(metric.id for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")


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


def test_signature_excludes_runtime_location_but_covers_science(
    tmp_path: Path,
) -> None:
    """Separate study continuation identity from invocation-only settings."""
    loaded = _load(tmp_path)
    signature = optuna_runtime.build_study_signature(loaded)
    assert signature["schema_version"] == optuna_runtime.STUDY_SIGNATURE_SCHEMA_VERSION
    assert signature["payload"]["schema_version"] == optuna_runtime.STUDY_SIGNATURE_SCHEMA_VERSION
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
    base["training"]["additional_evaluation_interval"] = 5
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
        _config: optuna_runtime.OptunaStudyConfig,
    ) -> Callable[[Any], float]:
        def objective(trial: Any) -> float:
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


def _install_running_failure_harness(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Replace expensive boundaries while retaining real trial lifecycle state."""
    processor = SimpleNamespace(
        state_dict=dict,
        in_normalizer=None,
        out_normalizer=SimpleNamespace(std=torch.ones(3)),
        to=lambda _device: None,
    )
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    prepare = optuna_runtime._prepare_trial_config

    def capture_prepared(loaded: Any, trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        result = prepare(loaded, trial)
        prepared["trial"] = result
        return result

    monkeypatch.setattr(optuna_runtime, "_prepare_trial_config", capture_prepared)

    def configure_reproducibility(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> dict[str, int]:
        del device
        return {"model_init": 1}

    def seed_process(_seed: int, *, device: torch.device) -> None:
        del device

    monkeypatch.setattr(
        experiments.run,
        "configure_reproducibility",
        configure_reproducibility,
    )
    monkeypatch.setattr(experiments.run, "seed_process", seed_process)
    monkeypatch.setattr(
        experiments.config.loader,
        "create_dataloaders_from_config",
        lambda *_args, **_kwargs: {
            "data_processor": processor,
            "split_indices": {},
            "train": object(),
            "eval": object(),
            "id_test": object(),
            "additional_evaluation": object(),
        },
    )

    def build_model(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> torch.nn.Module:
        del device
        return torch.nn.Linear(1, 1)

    def build_training_loss(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> object:
        del device
        return SimpleNamespace()

    def build_evaluation_metrics(
        _config: dict[str, Any],
        *,
        device: torch.device,
        output_standard_deviations: torch.Tensor,
    ) -> dict[str, Any]:
        del device
        assert torch.equal(output_standard_deviations, torch.ones(3))
        return {}

    monkeypatch.setattr(
        optuna_runtime.learning.models.factory,
        "build_model",
        build_model,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.losses.factory,
        "build_training_loss",
        build_training_loss,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.metrics.metrics,
        "build_evaluation_metrics",
        build_evaluation_metrics,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.optim,
        "build_optimizer",
        lambda _model, _config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.optim,
        "build_scheduler",
        lambda _optimizer, _config: None,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.checkpoint,
        "build_checkpoint_identity",
        lambda *_args, **_kwargs: {"effective_config_digest": "synthetic"},
    )
    monkeypatch.setattr(
        optuna_runtime.experiments.tracking,
        "build_monitor_membership",
        lambda *_args: None,
    )

    def fail_training(**_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(
        optuna_runtime.learning.training.loop,
        "train_loop",
        fail_training,
    )
    return prepared


@pytest.mark.parametrize(
    ("error_factory", "expected_status", "expected_error"),
    [
        pytest.param(
            lambda: torch.cuda.OutOfMemoryError("synthetic CUDA OOM"),
            "oom_pruned",
            optuna.TrialPruned,
            id="oom",
        ),
        pytest.param(
            lambda: FloatingPointError("synthetic non-finite loss"),
            "nonfinite_pruned",
            optuna.TrialPruned,
            id="nonfinite",
        ),
        pytest.param(
            lambda: optuna_runtime.RecoverableTrialError("synthetic recoverable failure"),
            "recoverable_failed",
            optuna_runtime.RecoverableTrialError,
            id="recoverable",
        ),
        pytest.param(
            lambda: RuntimeError("synthetic programming failure"),
            "failed",
            RuntimeError,
            id="failed",
        ),
        pytest.param(
            lambda: KeyboardInterrupt("synthetic interrupt"),
            "interrupted",
            KeyboardInterrupt,
            id="interrupted",
        ),
    ],
)
def test_running_trial_persists_distinct_terminal_states(
    error_factory: Callable[[], BaseException],
    expected_status: str,
    expected_error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify representative terminal outcomes and avoid CUDA cleanup on CPU."""
    loaded = optuna_runtime.with_runtime_overrides(
        _load(tmp_path),
        device="cpu",
        output_root=tmp_path,
    )
    error = error_factory()
    prepared = _install_running_failure_harness(error, monkeypatch)

    def cuda_query_forbidden() -> bool:
        pytest.fail("CPU cleanup must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_query_forbidden)
    trial = _Trial()
    with pytest.raises(expected_error):
        optuna_runtime.run_trial(loaded, trial)

    config, _context = prepared["trial"]
    run_dir = common.paths.resolve_config_run_dir(config)
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == expected_status
    assert summary["failure_context"]["error_type"] == type(error).__name__
    assert summary["sampled_parameters"] == trial.attrs["overrides"]


def test_public_run_statuses_distinguish_core_terminal_outcomes() -> None:
    """Keep completed, pruned, failed, and interrupted persistence distinct."""
    assert {"completed", "pruned", "failed", "interrupted"} <= (experiments.run.RUN_STATUSES)


@pytest.mark.parametrize(
    ("error", "expected_exception", "expected_status"),
    [
        (
            optuna_runtime.RecoverableTrialError("recoverable allocator out of memory"),
            optuna_runtime.RecoverableTrialError,
            "recoverable_failed",
        ),
        (
            RuntimeError("CUDA out of memory"),
            optuna.TrialPruned,
            "oom_pruned",
        ),
    ],
)
def test_transient_trial_preserves_recoverable_precedence_over_oom_text(
    error: BaseException,
    expected_exception: type[BaseException],
    expected_status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Classify explicit recoverable failures before allocator-text RuntimeErrors."""
    config = {
        "task": "transient_drying",
        "training": {"stage": "a", "mixed_precision": False, "evaluation_interval": 1, "epochs": 1},
        "run": {"device": "cpu"},
    }
    study: Any = SimpleNamespace(base_config=config, study={"mode": "stage_a_only", "pruner": {}}, search_space=())
    trial = _Trial()
    objective = {"id": "normalized_drying_group_macro_rmse", "direction": "minimize"}

    monkeypatch.setattr(optuna_runtime, "_prepare_trial_config", lambda *_args: (config, {}))
    monkeypatch.setattr(optuna_runtime, "_validate_transient_study_policy", lambda *_args: None)
    monkeypatch.setattr(optuna_runtime, "_duplicate_trial_value", lambda *_args: None)
    monkeypatch.setattr(optuna_runtime.learning.device, "resolve_device", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(optuna_runtime.learning.device, "validate_mixed_precision_device", lambda *_args: None)
    monkeypatch.setattr(optuna_runtime.experiments.config.loader, "get_resolved_objective", lambda *_args: objective)
    monkeypatch.setattr(optuna_runtime, "_trial_run_dir", lambda *_args: tmp_path / "transient-trial")
    monkeypatch.setattr(optuna_runtime.experiments.run, "prepare_fresh_run", lambda *_args, **_kwargs: tmp_path / "transient-trial")
    monkeypatch.setattr(optuna_runtime.experiments.run, "execute_prepared_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(expected_exception) as caught:
        optuna_runtime._run_transient_trial(study, trial)

    if expected_exception is optuna_runtime.RecoverableTrialError:
        assert caught.value is error
    assert trial.attrs["terminal_status"] == expected_status


class _SilentConsole:
    """Accept the console reporter calls of a trial without printing."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def startup(self, **_kwargs: Any) -> None:
        pass

    def epoch(self, *_args: Any) -> None:
        pass

    def final(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def failure(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def test_completed_trial_summary_keeps_checkpoint_digests_while_its_files_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trial is never resumed, so its summary keeps the digests while the large checkpoint files are deleted."""
    loaded = optuna_runtime.with_runtime_overrides(_load(tmp_path), device="cpu", output_root=tmp_path)
    prepared = _install_running_failure_harness(RuntimeError("replaced below"), monkeypatch)
    monkeypatch.setattr(optuna_runtime.experiments.console, "ConsoleReporter", _SilentConsole)

    def train(**kwargs: Any) -> dict[str, Any]:
        config = kwargs["config"]
        best = common.paths.resolve_best_checkpoint_file(kwargs["save_dir"])
        last = common.paths.resolve_last_checkpoint_file(kwargs["save_dir"])
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        epoch = min(int(config["training"]["evaluation_interval"]), int(config["training"]["epochs"]))
        kwargs["epoch_end_callback"](epoch, {f"id/{_OBJECTIVE_ID}": 0.5, "global_step": 1.0})
        return {
            "objective": experiments.config.loader.get_resolved_objective(config),
            "best_metric": 0.5,
            "best_epoch": epoch,
            "completed_epoch": epoch,
            "global_step": 1,
            "best_checkpoint_path": best,
            "last_checkpoint_path": last,
        }

    monkeypatch.setattr(optuna_runtime.learning.training.loop, "train_loop", train)
    monkeypatch.setattr(
        optuna_runtime.learning.training.loop,
        "evaluate_selected_checkpoint",
        lambda **_kwargs: {"selected_epoch": 1, "selected_metrics": {}, "terminal_epoch": 1, "terminal_metrics": {}},
    )

    assert optuna_runtime.run_trial(loaded, _Trial()) == pytest.approx(0.5)

    config, _context = prepared["trial"]
    run_dir = common.paths.resolve_config_run_dir(config)
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == "completed"
    assert summary["best_checkpoint_sha256"] == hashlib.sha256(b"best").hexdigest()
    assert summary["last_checkpoint_sha256"] == hashlib.sha256(b"last").hexdigest()
    assert not common.paths.resolve_best_checkpoint_file(run_dir).exists()
    assert not common.paths.resolve_last_checkpoint_file(run_dir).exists()
    assert common.paths.resolve_run_config_path(run_dir).is_file()


def _run_with_heavy_files(tmp_path: Path, *, trial: bool) -> Path:
    """Allocate a running Optuna trial or normal run that holds its split, normalizer and both checkpoints."""
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
        common.paths.resolve_last_checkpoint_file(run_dir),
    ):
        path.write_bytes(b"heavy")
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
    assert common.paths.resolve_last_checkpoint_file(run_dir).exists() is files_remain
    assert common.paths.resolve_normalizer_path(run_dir).exists()


def _completed_run_without_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, trial: bool, transient: bool = False) -> Path:
    """Publish a completed run or Optuna trial whose summary is consistent but whose checkpoint files do not exist."""
    run_dir = tmp_path / "run"
    task = "transient_drying" if transient else "steady_flow"
    config = {
        "task": task,
        "task_contract": {"digest": "t" * 64},
        "run": {"name": "finished", "layout": {"trial": "0000__finished" if trial else None}},
        "evaluation": {"objective": {"id": "objective", "direction": "minimize"}},
    }
    config_path = configs.write_yaml(common.paths.resolve_run_config_path(run_dir), config)
    split_path = common.paths.resolve_split_indices_path(run_dir)
    normalizer_path = common.paths.resolve_normalizer_path(run_dir)
    common.serialization.atomic_torch_save({"runtime_provenance": {"backend": "hdf5"}} if transient else {}, split_path)
    common.serialization.atomic_torch_save({}, normalizer_path)
    transient_evidence = {
        "checkpoint_identity": {"effective_config_digest": "e" * 64},
        "transient_scaling": {"semantic_digest": "s" * 64, "task_contract_digest": "t" * 64},
        "runtime_backend_provenance": {"backend": "hdf5"},
        "terminal_controller": {},
        "terminal_curriculum": {},
    }
    for status in ("initializing", "running"):
        experiments.run.transition_run_status(run_dir, status)
    experiments.run.transition_run_status(
        run_dir,
        "completed",
        updates={
            "config_sha256": common.serialization.file_sha256(config_path),
            "split_indices_sha256": common.serialization.file_sha256(split_path),
            "normalizer_sha256": common.serialization.file_sha256(normalizer_path),
            "amp_enabled": False,
            "effective_config_digest": "e" * 64,
            "task": task,
            "run_name": "finished",
            "objective": config["evaluation"]["objective"],
            "best_checkpoint": "best_checkpoint.pt",
            "last_checkpoint": "last_checkpoint.pt",
            "best_checkpoint_sha256": "b" * 64,
            "last_checkpoint_sha256": "c" * 64,
            "completed_epoch": 3,
            "best_epoch": 2,
            "global_step": 3,
            "best_metric": 0.5,
            **(transient_evidence if transient else {}),
        },
    )
    monkeypatch.setattr(experiments.run.config_loader, "validate_resolved_config", lambda raw: raw)
    monkeypatch.setattr(experiments.run, "_validate_saved_data_contract", lambda *_args: SimpleNamespace(digest="s" * 64))
    monkeypatch.setattr(experiments.run, "_run_validation_identity", lambda **_kwargs: {"effective_config_digest": "e" * 64})
    return run_dir


def _rewrite_summary(run_dir: Path, **changes: Any) -> None:
    """Replace summary fields without going through the lifecycle, as a contradictory summary would."""
    summary = experiments.run.read_run_summary(run_dir)
    summary.update(changes)
    common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), summary)


@pytest.mark.parametrize("trial", [True, False])
def test_completed_validation_accepts_a_finished_trial_without_checkpoints_only(
    trial: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded trial layout makes the deleted checkpoints optional; a normal run still fails closed without them."""
    run_dir = _completed_run_without_checkpoints(tmp_path, monkeypatch, trial=trial)

    if not trial:
        with pytest.raises(experiments.run.RunLifecycleError, match=r"Missing: best_checkpoint\.pt, last_checkpoint\.pt"):
            experiments.run.validate_completed_run(run_dir)
        return
    completed = experiments.run.validate_completed_run(run_dir)
    assert completed["scientific_run_name"] == "finished"
    assert "best_checkpoint" not in completed
    _rewrite_summary(run_dir, effective_config_digest="f" * 64)
    with pytest.raises(experiments.run.RunLifecycleError, match="identity"):
        experiments.run.validate_completed_run(run_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("task", "transient_drying", id="task-contradicts-config"),
        pytest.param("last_checkpoint_sha256", None, id="digest-missing"),
        pytest.param("completed_epoch", 0, id="no-completed-epoch"),
        pytest.param("best_epoch", None, id="best-epoch-missing"),
        pytest.param("best_epoch", 4, id="best-epoch-beyond-completed"),
        pytest.param("best_metric", float("nan"), id="non-finite-metric"),
    ],
)
def test_completed_trial_without_checkpoints_rejects_contradictory_summary_evidence(
    field: str,
    value: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary is the only evidence left of a finished trial, so it must agree with the config and record usable results."""
    run_dir = _completed_run_without_checkpoints(tmp_path, monkeypatch, trial=True)
    _rewrite_summary(run_dir, **{field: value})

    with pytest.raises(experiments.run.RunLifecycleError, match="Completed trial summary must record"):
        experiments.run.validate_completed_run(run_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("checkpoint_identity", {"effective_config_digest": "f" * 64}, "checkpoint identity", id="identity"),
        pytest.param("transient_scaling", {"semantic_digest": "x" * 64, "task_contract_digest": "t" * 64}, "scaling identity", id="scaling"),
        pytest.param("runtime_backend_provenance", {"backend": "other"}, "provenance", id="provenance"),
    ],
)
def test_transient_trial_without_checkpoints_keeps_its_recorded_identity_scaling_and_provenance(
    field: str,
    value: object,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary-only transient checks of a checkpointed run still apply to a trial whose checkpoints are gone."""
    run_dir = _completed_run_without_checkpoints(tmp_path, monkeypatch, trial=True, transient=True)
    experiments.run.validate_completed_run(run_dir)
    _rewrite_summary(run_dir, **{field: value})

    with pytest.raises(experiments.run.RunLifecycleError, match=message):
        experiments.run.validate_completed_run(run_dir)


def _a0_study(tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load the maintained transient A0 study against an exact test-owned Dataset identity."""
    loaded = optuna_runtime.load_optuna_study_config(
        "configs/03_learning/transient_drying/optuna/fno_a0_search.yaml",
        train_dataset="synthetic_transient_train",
    )
    return optuna_runtime.with_runtime_overrides(loaded, output_root=tmp_path)


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
    study = optuna_runtime.with_runtime_overrides(_load(tmp_path), output_root=tmp_path / "outputs")
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

    value = optuna_runtime._duplicate_trial_value(config, repeated, optuna.TrialPruned)

    assert value == pytest.approx(0.5)
    assert repeated.attrs["terminal_status"] == "duplicate"
    assert repeated.attrs["duplicate_of"] == original.relative_to(tmp_path / "outputs").as_posix()
    assert not optuna_runtime._trial_run_dir(config).exists()


def test_run_trial_answers_a_repeated_trial_before_allocating_or_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trial entry point returns a completed twin's value without allocating a run or contacting the device."""
    study = _duplicate_study(tmp_path)
    _publish_trial(study, 0, completed=True)
    repeated = _Trial(number=1)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a duplicate trial must not allocate or train")

    monkeypatch.setattr(optuna_runtime.experiments.run, "prepare_fresh_run", fail)
    monkeypatch.setattr(optuna_runtime.experiments.run, "execute_prepared_run", fail)

    assert optuna_runtime.run_trial(study, repeated) == pytest.approx(0.5)
    assert repeated.attrs["terminal_status"] == "duplicate"


@pytest.mark.parametrize("scenario", ["unfinished", "extended_after_training"])
def test_duplicate_trial_that_cannot_reuse_its_owner_is_pruned_without_training(tmp_path: Path, scenario: str) -> None:
    """A repeat of an unfinished or later-extended trial is pruned as a duplicate; nothing is allocated."""
    original = _publish_trial(
        _duplicate_study(tmp_path),
        0,
        completed=scenario != "unfinished",
        extra_epochs=3 if scenario == "extended_after_training" else 0,
    )
    repeated = _Trial(number=1)
    config, _context = optuna_runtime._prepare_trial_config(_duplicate_study(tmp_path), repeated)

    with pytest.raises(optuna.TrialPruned, match="not trained again"):
        optuna_runtime._duplicate_trial_value(config, repeated, optuna.TrialPruned)

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
    assert optuna_runtime._duplicate_trial_value(config, trial, optuna.TrialPruned) is None


def test_new_trial_identity_is_not_a_duplicate_and_registers_as_a_trial(tmp_path: Path) -> None:
    """A trial whose science is unclaimed is trained; once allocated its identity is owned by the trial kind."""
    study = _duplicate_study(tmp_path)
    trial = _Trial(number=0)
    config, _context = optuna_runtime._prepare_trial_config(study, trial)

    assert optuna_runtime._duplicate_trial_value(config, trial, optuna.TrialPruned) is None
    _publish_trial(study, 0, completed=False)

    entry = experiments.registry.lookup(config)
    assert entry is not None
    assert entry.kind == "trial"
