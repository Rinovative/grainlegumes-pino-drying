"""Exercise CLI parsing, delegation, dry-run isolation, and failure status.

Responsibilities:
  - Check CLI parsing, artifact delegation, dry-run isolation, result lines, failure status, redaction, and the Optuna logger level
  - Check that training publishes no evaluation artifact and that a bounded artifact build needs an explicitly named role

Design principles:
  - Invalid requests fail before side effects; held-out roles are evaluated only by the explicit artifact command

This module does NOT:
  - Perform production training or contact external tracking services
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from support import configs
from support.synthetic_task import admitted_manifest

from src import analysis, common, experiments, learning
from src.experiments.cli import cli_build_artifacts, cli_config_preflight, cli_optuna, cli_train

_CLI_USAGE_ERROR = 2


def test_public_cli_rejects_one_unknown_option() -> None:
    """Reject an unknown option through the public command entry point."""
    with pytest.raises(SystemExit) as captured:
        cli_train.main(["experiment.yaml", "--unsupported-option"])

    assert captured.value.code == _CLI_USAGE_ERROR


def test_config_preflight_cli_forwards_matched_overrides(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the train-only rollout and continuation-scale overrides through the public preflight parser."""
    calls: dict[str, object] = {}

    def validate(path: str, **kwargs: object) -> SimpleNamespace:
        calls["path"] = path
        calls.update(kwargs)
        return SimpleNamespace(
            family="experiment",
            task="transient_drying",
            canonical_path="configs/03_learning/transient_drying/experiments/example.yaml",
            run_label="transient-example",
            bound_source_tree="0123456789ab-fedcba987654",
        )

    monkeypatch.setattr(experiments.config.preflight, "validate_workflow", validate)

    assert cli_config_preflight.main(["train", "example.yaml", "--rollout-length", "7", "--lr-scale", "0.5"]) == 0

    assert calls["rollout_length"] == 7
    assert calls["lr_scale"] == pytest.approx(0.5)
    assert capsys.readouterr().out.endswith("\ttransient-example\t0123456789ab-fedcba987654\n")


def _cuda_zero_resolution() -> learning.device.DeviceResolution:
    """Return a CUDA decision without querying test-host hardware."""
    device = torch.device("cuda", 0)
    metadata = learning.device.DeviceRuntimeMetadata(
        requested_policy="cuda",
        resolved_device=str(device),
        device_type="cuda",
        pytorch_version=str(torch.__version__),
        cuda_index=0,
        cuda_device_name="restricted fixture GPU",
    )
    return learning.device.DeviceResolution(
        requested_policy="cuda",
        device=device,
        device_type="cuda",
        cuda_index=0,
        metadata=metadata,
    )


def test_training_publishes_no_held_out_artifacts_and_the_artifact_cli_is_their_explicit_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training ends without any evaluation artifact and says so; only the separate artifact command evaluates held-out roles."""
    run_dir = tmp_path / "renamed completed bundle"
    training_call: dict[str, object] = {}
    artifact_cli_call: dict[str, object] = {}

    def capture_run(*_args: object, **kwargs: object) -> dict[str, object]:
        training_call.update(kwargs)
        return {
            "run_dir": run_dir,
            "result": {
                "status": "completed",
                "best_epoch": 1,
                "best_metric": 0.5,
                "phase_seconds": {"startup": 5.0, "training": 70.0, "validation": 12.0},
            },
            "device_resolution": _cuda_zero_resolution(),
        }

    def reject_artifacts(*_args: object, **_kwargs: object) -> object:
        pytest.fail("training reached held-out artifact evaluation")

    def capture_build(**kwargs: object) -> dict[str, object]:
        artifact_cli_call.update(kwargs)
        return {}

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    for name in ("cleanup_runtime", "load_prepared_run_artifacts", "build_artifacts"):
        monkeypatch.setattr(analysis.artifacts.service, name, reject_artifacts)
    with pytest.raises(SystemExit) as refused:
        cli_train.main(["experiment.yaml", "--artifacts", "selected"])
    assert refused.value.code == _CLI_USAGE_ERROR
    capsys.readouterr()

    assert (
        cli_train.main(
            [
                "experiment.yaml",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--train-dataset",
                "synthetic_train",
                "--seed",
                "19",
                "--rollout-length",
                "8",
                "--lr-scale",
                "0.5",
                "--revision",
                "fixed-loss",
            ]
        )
        == 0
    )
    assert training_call["revision"] == "fixed-loss"
    assert "output_root" not in training_call
    assert training_call["epochs"] == 1
    assert training_call["train_dataset"] == "synthetic_train"
    assert training_call["seed"] == 19
    assert training_call["rollout_length"] == 8
    assert training_call["lr_scale"] == pytest.approx(0.5)
    assert training_call["device"] == "cpu"
    final = capsys.readouterr().out.splitlines()[-1]
    assert final.startswith("Post-training artifacts: none, ID-test and OOD are evaluated only by cli_build_artifacts | total ")
    for part in ("startup 5 s", "training 1 min 10 s", "validation 12 s", "other "):
        assert part in final

    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", capture_build)
    assert (
        cli_build_artifacts.main(
            [
                "--runs-root",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path / "raw"),
                "--device",
                "cuda",
                "--split",
                "material_ood",
            ],
        )
        == 0
    )
    assert artifact_cli_call["device_policy"] == "cuda"
    assert artifact_cli_call["dataset_root"] == tmp_path / "raw"
    # The canonical build releases exactly the named roles.
    assert artifact_cli_call["roles"] == ("material_ood",)


def test_artifact_cli_task_selects_only_its_own_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve ``--task`` to that task's folder below the experiments root, never to the runs of every task."""
    roots: list[object] = []
    monkeypatch.setattr(common.paths, "get_experiments_root", lambda: tmp_path)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", lambda **kwargs: roots.append(kwargs["runs_root"]) or [])

    assert cli_build_artifacts.main(["--task", "transient_drying", "--dataset-root", str(tmp_path / "datasets"), "--split", "id"]) == 0
    assert roots == [tmp_path / "transient_drying"]


def test_artifact_cli_delegates_bounded_transient_scope_to_noncanonical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward exact split/cases/device while bypassing full canonical generation."""
    captured: dict[str, object] = {}

    def capture_scoped(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            root=kwargs["output_root"],
            case_ids=("case-a", "case-c"),
        )

    monkeypatch.setattr(
        analysis.artifacts.service,
        "build_scoped_transient_artifact",
        capture_scoped,
    )
    monkeypatch.setattr(
        analysis.artifacts.service,
        "build_artifacts",
        lambda **_kwargs: pytest.fail("bounded CLI reached full artifact generation"),
    )
    run_dir = tmp_path / "run"
    output_root = tmp_path / "scoped artifact"

    assert (
        cli_build_artifacts.main(
            [
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--case-id",
                "case-a",
                "--case-id",
                "case-c",
                "--split",
                "material_ood",
                "--output-root",
                str(output_root),
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert captured == {
        "run_dir": run_dir,
        "dataset_root": tmp_path / "datasets",
        "output_root": output_root,
        "split": "material_ood",
        "case_ids": ["case-a", "case-c"],
        "one_case": False,
        "device_policy": "cuda",
    }


@pytest.mark.parametrize(
    "scope",
    [[], ["--one-case", "--output-root", "scoped"], ["--one-case", "--output-root", "scoped", "--split", "id", "--split", "material_ood"]],
    ids=["canonical", "bounded", "bounded-two-roles"],
)
def test_artifact_cli_evaluates_no_held_out_role_unless_the_caller_names_it(
    scope: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No build has a default role: without ``--split`` (bounded: exactly one) it is refused before any evaluation."""
    for name in ("build_artifacts", "build_scoped_transient_artifact"):
        monkeypatch.setattr(analysis.artifacts.service, name, lambda **_kwargs: pytest.fail("an unnamed held-out role reached evaluation"))

    with pytest.raises(SystemExit) as refused:
        cli_build_artifacts.main(["--run-dir", str(tmp_path / "run"), *scope])

    assert refused.value.code == _CLI_USAGE_ERROR
    assert "--split" in capsys.readouterr().err


def test_matched_cli_forwards_its_plan_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward every plan override to the run service, which owns their resolution."""
    config_path = configs.write_yaml(tmp_path / "config folder" / "matched plan.yaml", {})
    run_dir = tmp_path / "leaf folder" / "B run"
    train_dataset = "lentil_four_hour__c3f0__selected dataset"
    calls: dict[str, object] = {}

    def capture_run(selected_config: object, **kwargs: object) -> dict[str, object]:
        calls["config_path"] = selected_config
        calls.update(kwargs)
        return {
            "run_dir": run_dir,
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "stage_runs": {"a0": tmp_path / "A0", "b": run_dir},
            "stage_outcomes": {"a0": {"result": {"status": "completed"}}, "b": {"result": {"status": "completed"}}},
            "device_resolution": SimpleNamespace(device=torch.device("cpu")),
        }

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)

    assert (
        cli_train.main(
            [
                str(config_path),
                "--train-dataset",
                train_dataset,
                "--epochs",
                "8",
                "--rollout-length",
                "4",
                "--lr-scale",
                "0.5",
            ]
        )
        == 0
    )

    assert calls["train_dataset"] == train_dataset
    assert calls["epochs"] == 8
    assert calls["rollout_length"] == 4
    assert calls["lr_scale"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "overrides",
    [["--rollout-length", "0"], ["--epochs", "0"]],
    ids=["rollout-length", "epochs"],
)
def test_train_cli_rejects_non_positive_budgets_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    overrides: list[str],
) -> None:
    """Reject budget values that can never describe a run before run-service dispatch."""
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: pytest.fail("invalid override reached training"),
    )

    assert cli_train.main(["unused.yaml", *overrides]) == 1


def test_train_cli_rejects_a_negative_seed_before_dispatch(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the seed contract at the CLI boundary instead of failing later inside the run service."""
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: pytest.fail("negative seed reached training"),
    )

    assert cli_train.main(["unused.yaml", "--seed", "-1"]) == 1
    assert "--seed must be a non-negative integer." in capsys.readouterr().err


@pytest.mark.parametrize("lr_scale", ["0", "-1", "nan"])
def test_train_cli_rejects_an_invalid_lr_scale_before_any_run_is_allocated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    lr_scale: str,
) -> None:
    """The public command surfaces the loader's rejection and leaves the output root untouched."""
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)

    assert cli_train.main([str(config_path), "--lr-scale", lr_scale]) == 1

    assert "lr_scale must be a positive finite" in capsys.readouterr().err
    assert not (tmp_path / "storage").exists()


@pytest.mark.parametrize(
    ("result", "exit_code", "line"),
    [
        (
            {"status": "no_eligible_checkpoint", "best_epoch": None, "training_outcome": {"eligible_points": 0, "budget_progress": 1.0}},
            2,
            "Training status: no_eligible_checkpoint | unsuccessful: no eligible checkpoint (0 eligible validation points)",
        ),
        (
            {"status": "diverged", "best_epoch": 2, "best_metric": 0.5, "training_outcome": {"eligible_points": 2, "budget_progress": 0.25}},
            0,
            "Training status: diverged | best epoch 2 | best metric 5.000e-01 | diverged at 25.0 % of the budget",
        ),
    ],
    ids=["no-eligible-checkpoint", "diverged"],
)
def test_the_status_line_names_the_end_status_of_the_run(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, result: dict[str, Any], exit_code: int, line: str
) -> None:
    """A run without an eligible checkpoint exits 2; a diverged run keeps its best and names how much budget it spent."""
    outcome = {"run_dir": Path("run"), "result": result, "device_resolution": _cuda_zero_resolution()}
    monkeypatch.setattr(experiments.run, "run_experiment", lambda *_args, **_kwargs: outcome)

    assert cli_train.main(["experiment.yaml"]) == exit_code

    assert line in capsys.readouterr().out


def test_run_admission_rejection_reports_the_saved_epochs(capsys: pytest.CaptureFixture[str]) -> None:
    """Name the saved run's completed epochs next to the rejection reason."""
    cli_train._print_existing_run_admission({"reason": "different settings", "completed_epoch": 5, "status": "completed"})

    error = capsys.readouterr().err
    assert "different settings" in error
    assert "saved 5 epochs" in error


def test_optuna_dry_run_resolves_artificial_plan_without_side_effects(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a test-owned study without SDK, allocator, tracking, or output writes."""
    config_path = configs.write_yaml(tmp_path / "artificial-study.yaml", configs.optuna_config())
    study_name = config_path.stem
    output_root = tmp_path / "storage" / "03_experiments"
    described: list[str] = []

    def describe(task: str, dataset_id: str, **_roots: object) -> dict[str, Any]:
        described.append(dataset_id)
        return admitted_manifest(task, dataset_id)

    def reject_side_effect(*_args: object, **_kwargs: object) -> object:
        pytest.fail("dry-run reached a side-effecting runtime boundary")

    runtime = experiments.tuning.optuna
    monkeypatch.setattr(runtime.datasets.packages.manifest, "load_package_manifest", describe)
    monkeypatch.setattr(runtime, "_optuna_module", reject_side_effect)
    monkeypatch.setattr(experiments.run, "prepare_fresh_run", reject_side_effect)
    monkeypatch.setattr(experiments.tracking, "initialize_wandb", reject_side_effect)

    assert (
        cli_optuna.main(
            [
                str(config_path),
                "--dry-run",
                "--device",
                "cpu",
                "--epochs",
                "3",
                "--train-dataset",
                "synthetic_train_override",
                "--revision",
                "short",
            ],
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    study_dir = output_root / "steady_flow" / "synthetic_train_override" / "optuna" / f"{study_name}__ep-3__rev-short"
    assert plan["device_policy"] == "cpu"
    assert plan["semantic_signature"]["payload"]["scientific_base_config"]["training"]["budget_epochs"] == 3
    assert plan["study_dir"] == str(study_dir)
    assert plan["trial_root"] == str(study_dir / "trials")
    assert plan["storage"] == f"sqlite:///{study_dir / f'{study_name}.db'}"
    assert plan["study"]["role"] is None
    assert plan["dataset_roles"]["train"]["dataset_id"] == "synthetic_train_override"
    assert plan["dataset_roles"]["material_ood"][0]["dataset_id"] == "synthetic_ood"
    assert described == ["synthetic_train_override", "synthetic_ood"]
    assert "semantic_signature" in plan
    assert not output_root.exists()


def test_optuna_run_keeps_the_library_logger_at_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the library's INFO lines by our own trial lines while keeping its warnings."""
    optuna_sdk = pytest.importorskip("optuna")
    previous = optuna_sdk.logging.get_verbosity()
    runtime = experiments.tuning.optuna
    monkeypatch.setattr(runtime, "load_optuna_study_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime, "with_runtime_overrides", lambda study_config, **_kwargs: study_config)
    monkeypatch.setattr(runtime, "run_optuna_study", lambda *_args, **_kwargs: None)
    optuna_sdk.logging.set_verbosity(optuna_sdk.logging.INFO)
    try:
        assert cli_optuna.main(["study.yaml"]) == 0
        assert optuna_sdk.logging.get_verbosity() == optuna_sdk.logging.WARNING
    finally:
        optuna_sdk.logging.set_verbosity(previous)


def test_train_cli_prints_the_result_with_storage_relative_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show status, best epoch, metric and the run folder relative to the storage root."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    run_dir = tmp_path / "03_experiments" / "run-a"
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": run_dir,
            "result": {"status": "completed", "best_epoch": 3, "best_metric": 0.0123456},
            "device_resolution": _cuda_zero_resolution(),
        },
    )

    assert cli_train.main(["experiment.yaml"]) == 0

    output = capsys.readouterr().out
    for expected in ("Training status: completed", "best epoch 3", "1.235e-02", "03_experiments/run-a"):
        assert expected in output
    assert str(tmp_path) not in output


def test_optuna_dry_run_dataset_failure_is_nonzero_without_output(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a missing artificial dataset before allocating study state."""
    config_path = configs.write_yaml(
        tmp_path / "artificial-study.yaml",
        configs.optuna_config(),
    )
    output_root = tmp_path / "storage" / "03_experiments"

    def reject_dataset(_task: str, dataset_id: str, **_kwargs: object) -> None:
        msg = f"configured dataset missing: {dataset_id}"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(experiments.tuning.optuna.datasets.packages.manifest, "load_package_manifest", reject_dataset)
    assert (
        cli_optuna.main(
            [
                str(config_path),
                "--dry-run",
                "--device",
                "cpu",
            ],
        )
        == 1
    )
    assert "synthetic_train" in capsys.readouterr().err
    assert not output_root.exists()


def test_cli_failures_are_nonzero_and_redact_credentials(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return failure status while keeping credential-shaped values out of errors."""

    def fail_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        msg = "api_key=never-disclose"
        raise RuntimeError(msg)

    def fail_load(_path: str) -> object:
        msg = "study invalid"
        raise ValueError(msg)

    def fail_build(**_kwargs: object) -> dict[str, object]:
        msg = "generation failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(experiments.run, "run_experiment", fail_run)
    monkeypatch.setattr(experiments.tuning.optuna, "load_optuna_study_config", fail_load)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", fail_build)

    assert cli_train.main(["unused.yaml"]) == 1
    training_error = capsys.readouterr().err
    assert "RuntimeError" in training_error
    assert "api_key=<redacted>" in training_error
    assert "never-disclose" not in training_error
    assert cli_optuna.main(["unused.yaml", "--dry-run"]) == 1
    assert cli_build_artifacts.main(["--runs-root", str(tmp_path), "--split", "id"]) == 1


def test_saved_effective_resume_cli_uses_run_config_without_repeating_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the immutable stage config and exact leaf to the existing service."""
    run_dir = tmp_path / "saved stage"
    calls: dict[str, object] = {}

    def run(path: object, **kwargs: object) -> dict[str, object]:
        calls["path"] = path
        calls.update(kwargs)
        return {
            "run_dir": run_dir,
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "device_resolution": SimpleNamespace(device=torch.device("cpu")),
        }

    monkeypatch.setattr(experiments.run, "run_experiment", run)
    assert cli_train.main(["--resume", str(run_dir)]) == 0
    assert Path(str(calls["path"])) == run_dir / "config.yaml"
    assert calls["resume"] == str(run_dir)
    assert all(calls[key] is None for key in ("epochs", "train_dataset", "rollout_length", "lr_scale"))
