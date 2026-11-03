"""Exercise CLI parsing, delegation, dry-run isolation, and failure status.

Responsibilities:
  - Check CLI parsing, artifact delegation, dry-run isolation, result lines, failure status, redaction, and the Optuna logger level

Design principles:
  - Invalid requests fail before side effects; artifact failure preserves completed training

This module does NOT:
  - Perform production training or contact external tracking services
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from support import configs

from src import analysis, common, experiments, learning
from src.experiments.cli import cli_build_artifacts, cli_config_preflight, cli_optuna, cli_train

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec

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


def test_direct_cli_builds_artifacts_after_strict_completion_on_training_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the exact completed path and CUDA decision after releasing training."""
    run_dir = tmp_path / "renamed completed bundle"
    resolution = _cuda_zero_resolution()
    events: list[str] = []
    training_call: dict[str, object] = {}
    post_training_call: dict[str, object] = {}
    artifact_cli_call: dict[str, object] = {}

    def capture_run(*_args: object, **kwargs: object) -> dict[str, object]:
        events.append("training returned")
        training_call.update(kwargs)
        return {
            "run_dir": run_dir,
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "device_resolution": resolution,
        }

    def validate_completed(selected: object) -> dict[str, object]:
        assert selected is run_dir
        events.append("strict completion validated")
        return {}

    def cleanup(device: torch.device) -> None:
        assert device is resolution.device
        events.append("training resources cleaned")

    def prepare(selected: object, **kwargs: object) -> SimpleNamespace:
        assert selected is run_dir
        post_training_call.update(kwargs)
        events.append("artifacts prepared")
        return SimpleNamespace(role_actions={"id": "reused", "additional_evaluation": "generated"})

    def capture_build(**kwargs: object) -> dict[str, object]:
        artifact_cli_call.update(kwargs)
        return {}

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    monkeypatch.setattr(experiments.run, "validate_completed_run", validate_completed)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", cleanup)
    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", prepare)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", capture_build)

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
                "--train-seconds",
                "25",
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
    assert training_call["train_seconds"] == 25
    assert training_call["rollout_length"] == 8
    assert training_call["lr_scale"] == pytest.approx(0.5)
    assert events == [
        "training returned",
        "strict completion validated",
        "training resources cleaned",
        "artifacts prepared",
    ]
    assert training_call["device"] == "cpu"
    assert post_training_call == {"rebuild_incompatible": True, "device_resolution": resolution}

    assert (
        cli_build_artifacts.main(
            [
                "--runs-root",
                str(tmp_path),
                "--dataset-root",
                str(tmp_path / "raw"),
                "--device",
                "cuda",
            ],
        )
        == 0
    )
    assert artifact_cli_call["device_policy"] == "cuda"
    assert artifact_cli_call["evaluation_spatial_stride"] == 1
    assert artifact_cli_call["dataset_root"] == tmp_path / "raw"


def test_artifact_cli_task_selects_only_its_own_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve ``--task`` to that task's folder below the experiments root, never to the runs of every task."""
    roots: list[object] = []
    monkeypatch.setattr(common.paths, "get_experiments_root", lambda: tmp_path)
    monkeypatch.setattr(analysis.artifacts.service, "build_artifacts", lambda **kwargs: roots.append(kwargs["runs_root"]) or [])

    assert cli_build_artifacts.main(["--task", "transient_drying", "--dataset-root", str(tmp_path / "datasets")]) == 0
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
                "additional_evaluation",
                "--output-root",
                str(output_root),
                "--device",
                "cuda",
                "--evaluation-spatial-stride",
                "2",
            ]
        )
        == 0
    )
    assert captured == {
        "run_dir": run_dir,
        "dataset_root": tmp_path / "datasets",
        "output_root": output_root,
        "split": "additional_evaluation",
        "case_ids": ["case-a", "case-c"],
        "one_case": False,
        "device_policy": "cuda",
        "evaluation_spatial_stride": 2,
    }


@pytest.mark.parametrize("value", ["0", "-1", "2.0", "02", "+2", "x"])
def test_artifact_cli_rejects_malformed_or_nonpositive_evaluation_stride(value: str) -> None:
    """Reject ambiguous or invalid Evaluation-grid stride syntax before delegation."""
    with pytest.raises(SystemExit):
        cli_build_artifacts.main(
            ["--runs-root", "unused", "--evaluation-spatial-stride", value],
        )


def test_direct_cli_opt_out_skips_only_post_training_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed training unchanged when the trailing invocation flag opts out."""
    resolution = _cuda_zero_resolution()
    calls: list[str] = []

    def capture_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("training")
        return {
            "run_dir": Path("run"),
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "device_resolution": resolution,
        }

    def reject_post_training(*_args: object, **_kwargs: object) -> object:
        pytest.fail("opt-out reached post-training artifact processing")

    monkeypatch.setattr(experiments.run, "run_experiment", capture_run)
    monkeypatch.setattr(experiments.run, "validate_completed_run", reject_post_training)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", reject_post_training)
    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", reject_post_training)

    assert cli_train.main(["experiment.yaml", "--artifacts", "none"]) == 0
    assert calls == ["training"]


def test_budget_unmet_continuation_uses_saved_effective_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the exact saved leaf so original plan overrides need no manual repetition."""
    config_path = configs.write_yaml(tmp_path / "config folder" / "matched plan.yaml", {})
    run_dir = tmp_path / "leaf folder" / "B run"
    run_dir.mkdir(parents=True)
    configs.write_yaml(common.paths.resolve_run_config_path(run_dir), {"training": {"epochs": 8}})
    train_dataset = "lentil_four_hour__c3f0__selected dataset"
    calls: dict[str, object] = {}

    def capture_run(selected_config: object, **kwargs: object) -> dict[str, object]:
        calls["config_path"] = selected_config
        calls.update(kwargs)
        return {
            "run_dir": run_dir,
            "result": {
                "status": "budget_unmet",
                "train_seconds": 0.5,
                "target_train_seconds": 1.0,
                "completed_epoch": 2,
                "global_step": 3,
            },
            "stage_runs": {"a0": tmp_path / "A0", "b": run_dir},
            "stage_outcomes": {
                "a0": {"result": {"status": "completed"}},
                "b": {"result": {"status": "budget_unmet"}},
            },
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
                "--train-seconds",
                "1",
                "--rollout-length",
                "4",
                "--lr-scale",
                "0.5",
                "--artifacts",
                "none",
            ]
        )
        == 2
    )

    assert calls["train_dataset"] == train_dataset
    assert calls["epochs"] == 8
    assert calls["train_seconds"] == 1
    assert calls["rollout_length"] == 4
    assert calls["lr_scale"] == pytest.approx(0.5)
    output = capsys.readouterr().out
    continuation_line = next(line for line in output.splitlines() if line.startswith("Continuation inside"))
    command = shlex.split(continuation_line.split(": ", maxsplit=1)[1])
    assert command[:3] == ["python", "-m", "src.experiments.cli.cli_train"]
    assert command[3] == "--resume"
    assert command[command.index("--resume") + 1] == str(run_dir)
    resumed_args = cli_train._build_parser().parse_args(command[3:])
    assert resumed_args.config_path is None
    assert resumed_args.train_dataset is None
    assert resumed_args.epochs is None
    assert resumed_args.train_seconds is None
    assert resumed_args.rollout_length is None
    assert resumed_args.lr_scale is None
    assert resumed_args.train_stages is None
    assert Path(resumed_args.resume) == run_dir
    assert resumed_args.max_epochs == 8


@pytest.mark.parametrize(
    "overrides",
    [["--rollout-length", "0"], ["--epochs", "0"], ["--max-epochs", "0"]],
    ids=["rollout-length", "epochs", "max-epochs"],
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


@pytest.mark.parametrize("lr_scale", ["0", "-1", "nan"])
def test_train_cli_rejects_an_invalid_lr_scale_before_any_run_is_allocated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    lr_scale: str,
) -> None:
    """The public command surfaces the loader's rejection and leaves the output root untouched."""
    config_path = configs.write_config(tmp_path, configs.transient_matched_config(), canonical_root=tmp_path)

    assert cli_train.main([str(config_path), "--lr-scale", lr_scale, "--artifacts", "none"]) == 1

    assert "lr_scale must be a positive finite" in capsys.readouterr().err
    assert not (tmp_path / "storage").exists()


def test_artifact_failure_is_nonzero_without_relabeling_completed_training(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report recoverable post-processing failure without a lifecycle transition."""
    resolution = _cuda_zero_resolution()

    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": Path("completed-run"),
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "device_resolution": resolution,
        },
    )
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: {})
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)
    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_or_build_run_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("artifact cache is incompatible")),
    )
    monkeypatch.setattr(
        experiments.run,
        "transition_run_status",
        lambda *_args, **_kwargs: pytest.fail("artifact failure changed training lifecycle"),
    )

    assert cli_train.main(["experiment.yaml"]) == 1
    captured = capsys.readouterr()
    assert "Training status: completed" in captured.out
    assert "Post-training artifacts: failed" in captured.err
    assert "python -m src.experiments.cli.cli_build_artifacts --run-dir completed-run" in captured.err
    assert "--rebuild" not in captured.err


def _capture_post_training_rebuild_flags(monkeypatch: pytest.MonkeyPatch, outcome: dict[str, object]) -> dict[Path, object]:
    """Run the train CLI on a fixed outcome and map each prepared run directory to its rebuild flag."""
    flags: dict[Path, object] = {}

    def prepare(selected: Path, **kwargs: object) -> SimpleNamespace:
        flags[selected] = kwargs["rebuild_incompatible"]
        # An authorized rebuild replaces the incompatible artifacts of an extended run, as the artifact service reports it.
        return SimpleNamespace(role_actions={"id": "rebuilt" if kwargs["rebuild_incompatible"] else "reused"})

    monkeypatch.setattr(experiments.run, "run_experiment", lambda *_args, **_kwargs: outcome)
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: {})
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)
    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", prepare)
    assert cli_train.main(["experiment.yaml"]) == 0
    return flags


@pytest.mark.parametrize(("extra", "rebuild"), [({}, True), ({"already_complete": True}, False)])
def test_single_run_rebuilds_incompatible_artifacts_only_when_trained_here(
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, object],
    rebuild: bool,
) -> None:
    """Authorize artifact replacement for a run trained now, never for a merely reused run."""
    outcome = {
        "run_dir": Path("run"),
        "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
        "device_resolution": _cuda_zero_resolution(),
        **extra,
    }

    assert _capture_post_training_rebuild_flags(monkeypatch, outcome) == {Path("run"): rebuild}


def test_matched_run_rebuilds_incompatible_artifacts_only_for_the_stage_trained_here(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge each matched stage by its own outcome so a reused stage is only validated."""
    a0, b = tmp_path / "a0", tmp_path / "b"
    completed = {"status": "completed", "best_epoch": 1, "best_metric": 0.5}
    outcome = {
        "run_dir": b,
        "result": completed,
        "stage_runs": {"a0": a0, "b": b},
        "stage_outcomes": {
            "a0": {"result": completed, "already_complete": True},
            "b": {"result": completed},
        },
        "device_resolution": _cuda_zero_resolution(),
    }

    assert _capture_post_training_rebuild_flags(monkeypatch, outcome) == {a0: False, b: True}


def test_incompatible_artifact_failure_recovery_command_requests_rebuild(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name --rebuild in the recovery command only for incompatible artifact targets."""
    monkeypatch.setattr(
        experiments.run,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "run_dir": Path("completed-run"),
            "result": {"status": "completed", "best_epoch": 1, "best_metric": 0.5},
            "device_resolution": _cuda_zero_resolution(),
        },
    )
    monkeypatch.setattr(experiments.run, "validate_completed_run", lambda _path: {})
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)
    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_or_build_run_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(analysis.evaluation.artifact_loader.IncompatibleEvaluationArtifactsError("stale")),
    )

    assert cli_train.main(["experiment.yaml"]) == 1
    recovery = [line for line in capsys.readouterr().err.splitlines() if line.startswith("Recovery:")]
    assert len(recovery) == 1
    assert recovery[0].endswith("--run-dir completed-run --rebuild")


def _describe_artificial_dataset(
    seen: list[str],
    dataset_id: str,
    *,
    task: TaskSpec,
    dataset_root: Path,
    metadata_root: Path,
) -> SimpleNamespace:
    """Return a valid bounded summary for a test-owned dataset identifier."""
    seen.append(dataset_id)
    return SimpleNamespace(
        dataset_id=dataset_id,
        dataset_path=common.paths.resolve_dataset_path(task.id, dataset_id, dataset_root=dataset_root),
        metadata_directory=common.paths.resolve_dataset_metadata_dir(task.id, dataset_id, metadata_root=metadata_root),
        dataset_exists=True,
        task_id=task.id,
        data_contract_digest=task.data_contract_digest,
        dataset_digest="a" * 64,
        sample_count=4,
    )


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

    def describe(dataset_id: str, **kwargs: object) -> SimpleNamespace:
        return _describe_artificial_dataset(described, dataset_id, **kwargs)  # type: ignore[arg-type]

    def reject_side_effect(*_args: object, **_kwargs: object) -> object:
        pytest.fail("dry-run reached a side-effecting runtime boundary")

    runtime = experiments.tuning.optuna
    monkeypatch.setattr(runtime.datasets.contracts.metadata, "load_dataset_metadata_summary", describe)
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
    assert plan["semantic_signature"]["payload"]["scientific_base_config"]["training"]["epochs"] == 3
    assert plan["study_dir"] == str(study_dir)
    assert plan["trial_root"] == str(study_dir / "trials")
    assert plan["storage"] == f"sqlite:///{study_dir / f'{study_name}.db'}"
    assert plan["study"]["role"] is None
    assert plan["dataset_roles"]["train"]["dataset_id"] == "synthetic_train_override"
    assert plan["dataset_roles"]["additional_evaluation"][0]["dataset_id"] == "synthetic_ood"
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

    assert cli_train.main(["experiment.yaml", "--artifacts", "none"]) == 0

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

    def reject_dataset(dataset_id: str, **_kwargs: object) -> None:
        msg = f"configured dataset missing: {dataset_id}"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(
        experiments.tuning.optuna.datasets.contracts.metadata,
        "load_dataset_metadata_summary",
        reject_dataset,
    )
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
    assert cli_build_artifacts.main(["--runs-root", str(tmp_path)]) == 1


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
    assert cli_train.main(["--resume", str(run_dir), "--max-epochs", "12", "--artifacts", "none"]) == 0
    assert Path(str(calls["path"])) == run_dir / "config.yaml"
    assert calls["resume"] == str(run_dir)
    assert calls["max_epochs"] == 12
    assert all(calls[key] is None for key in ("epochs", "train_dataset", "train_seconds", "rollout_length", "lr_scale"))
