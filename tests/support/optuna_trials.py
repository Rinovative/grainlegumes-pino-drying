"""Run one real Optuna trial of a test-owned study and admit its completed trial.

Responsibilities:
  - Run a one-trial CPU study of either task and compare its recorded signature with the dry run's
  - Admit the completed trial through completed-run validation without patches
  - Check that the summary, the only evidence left of a finished trial, refuses contradictions
  - Bind configured Dataset IDs as stand-in physical identities for storage-free study signatures

Design principles:
  - Both tasks pass the same acceptance; only the study request differs

This module does NOT:
  - Build Datasets, choose search spaces, or validate scientific accuracy
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
import pytest

from src import common, experiments
from support import configs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.experiments.tuning.experiments_tuning_optuna import OptunaStudyConfig


def with_configured_identities(config: OptunaStudyConfig) -> OptunaStudyConfig:
    """Bind each configured Dataset ID as its stand-in physical identity, so a signature needs no storage."""
    data = config.base_config["data"]
    identities = {
        "train": {"dataset_id": data["train_dataset"]},
        "material_ood": [{"dataset_id": dataset_id} for dataset_id in data["material_ood_datasets"]],
    }
    return replace(config, dataset_identities=identities)


def run_one_trial_study(request: Mapping[str, Any], *, directory: Path) -> dict[str, Any]:
    """Run trial 0 of ``request`` on the CPU below ``directory`` and return its admitted completed run."""
    optuna_runtime = experiments.tuning.optuna
    # The best config is built against the task's maintained base.yaml, which a synthetic project root lacks.
    task_base = experiments.tuning.optuna._task_base_path(request["experiment"]["task"])
    if not task_base.is_file():
        task_base.parent.mkdir(parents=True, exist_ok=True)
        task_base.write_text("{}\n", encoding="utf-8")
    path = configs.write_yaml(directory / "trial_lifecycle_study.yaml", request)
    study_config = optuna_runtime.with_runtime_overrides(
        optuna_runtime.load_optuna_study_config(path), device="cpu", output_root=directory / "outputs"
    )
    planned = optuna_runtime.describe_optuna_study_config(study_config)

    study = optuna_runtime.run_optuna_study(study_config, n_trials=1)

    assert study.user_attrs["semantic_signature"] == planned["semantic_signature"]["digest"]
    (trial,) = study.trials
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.user_attrs["terminal_status"] == "completed"
    (trial_dir,) = Path(planned["trial_root"]).iterdir()
    completed = experiments.run.validate_completed_run(trial_dir)
    assert completed["summary"]["best_metric"] == trial.value
    best = common.paths.resolve_best_checkpoint_file(trial_dir)
    assert not best.exists()

    summary_path = common.paths.resolve_run_summary_path(trial_dir)
    recorded = summary_path.read_bytes()
    contradiction = {**completed["summary"], "best_epoch": completed["summary"]["completed_epoch"] + 1}
    common.serialization.atomic_write_json(summary_path, contradiction)
    with pytest.raises(experiments.run.RunLifecycleError, match="Completed trial summary must record"):
        experiments.run.validate_completed_run(trial_dir)
    summary_path.write_bytes(recorded)
    # A kill between the completed summary and the best deletion leaves a best behind; only the recorded one is admitted.
    best.write_bytes(b"another best")
    with pytest.raises(experiments.run.RunLifecycleError, match="best checkpoint digest mismatch"):
        experiments.run.validate_completed_run(trial_dir)
    best.unlink()
    return completed
