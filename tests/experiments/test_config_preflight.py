"""Validate side-effect-free workflow classification on test-owned YAMLs.

Responsibilities:
  - Check side-effect-free classification of matching, wrong, and mixed workflow YAMLs
  - Check that saved-config resume is validated structurally, without the live scientific implementation

Design principles:
  - Schema content determines workflow rather than filenames

This module does NOT:
  - Execute a classified workflow or create run artifacts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml
from support import configs

from src import common
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_preflight as preflight

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("workflow", "expected_family"),
    [
        ("train", preflight.EXPERIMENT_FAMILY),
        ("optuna", preflight.OPTUNA_FAMILY),
    ],
)
def test_matching_workflow_is_admitted(
    tmp_path: Path,
    workflow: str,
    expected_family: str,
) -> None:
    """Classify one complete artificial request for each public workflow."""
    payload = configs.direct_config() if workflow == "train" else configs.optuna_config()
    path = configs.write_config(tmp_path, payload) if workflow == "train" else configs.write_yaml(tmp_path / "study.yaml", payload)

    result = preflight.validate_workflow(path, requested_workflow=workflow)

    assert result.family == expected_family
    assert result.task == "steady_flow"
    assert result.run_label


@pytest.mark.parametrize(
    ("requested_workflow", "payload_family"),
    [("train", "optuna"), ("optuna", "direct")],
)
def test_wrong_workflow_is_rejected(
    tmp_path: Path,
    requested_workflow: str,
    payload_family: str,
) -> None:
    """Reject cross-family dispatch without depending on correction wording."""
    payload = configs.optuna_config() if payload_family == "optuna" else configs.direct_config()
    path = configs.write_config(tmp_path, payload) if payload_family == "direct" else configs.write_yaml(tmp_path / "study.yaml", payload)

    with pytest.raises(preflight.WorkflowMismatchError) as captured:
        preflight.validate_workflow(
            path,
            requested_workflow=requested_workflow,
        )

    assert captured.value.result.family != requested_workflow
    assert "RESOURCE_OPTIONS" in str(captured.value)
    assert "./scripts/slurm_ml.sh --help" in str(captured.value)
    assert "--mode gpu" not in str(captured.value)


def test_mixed_root_is_never_classified_by_filename(tmp_path: Path) -> None:
    """Fail closed when one YAML root mixes direct and Optuna ownership."""
    payload = configs.direct_config()
    payload["study"] = {}
    path = configs.write_yaml(tmp_path / "looks_like_search.yaml", payload)

    with pytest.raises(ValueError, match="mixes normal experiment and Optuna wrapper"):
        preflight.inspect_config(path)


def test_matched_overrides_reach_loader_before_training_plan_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the train workflow using the requested B horizon, B target, and continuation scale."""
    path = configs.write_config(tmp_path, configs.transient_matched_config())
    original = preflight.loader.apply_training_overrides
    captured: dict[str, Any] = {}

    def capture(raw: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        resolved = original(raw, **kwargs)
        captured.update(kwargs)
        captured["training"] = resolved["training"]
        return resolved

    monkeypatch.setattr(preflight.loader, "apply_training_overrides", capture)
    result = preflight.validate_workflow(path, requested_workflow="train", rollout_length=2, train_seconds=3.0, lr_scale=0.5)

    assert result.task == "transient_drying"
    assert (captured["rollout_length"], captured["train_seconds"], captured["lr_scale"]) == (2, 3.0, 0.5)
    training = captured["training"]
    assert training["b"]["sampling"] == {
        "mode": "rollout_window",
        "rollout_length": 2,
        "window_stride": 2,
        "window_offset": 0,
    }
    assert training["b"]["curriculum"]["lengths"] == [2]
    assert training["b"]["curriculum"]["milestone_fractions"] == [0]
    assert training["b"]["train_seconds"] == pytest.approx(3.0)
    assert training["b"]["lr_scale"] == training["a1"]["lr_scale"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "override",
    [{"rollout_length": 6}, {"train_seconds": 1.0}, {"lr_scale": 0.5}],
    ids=["rollout-length", "train-seconds", "lr-scale"],
)
def test_matched_overrides_are_rejected_for_optuna_workflow(
    tmp_path: Path,
    override: dict[str, Any],
) -> None:
    """Keep the matched-plan overrides out of the Optuna interface."""
    path = configs.write_yaml(tmp_path / "study.yaml", configs.optuna_config())

    with pytest.raises(ValueError, match="apply only to matched transient training"):
        preflight.validate_workflow(path, requested_workflow="optuna", **override)


def _saved_run_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Write one strict resolved config at its test-owned storage run path."""
    storage = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    resolved = configs.resolve(configs.direct_config(), storage_root=storage)
    run_dir = common.paths.resolve_config_run_dir(resolved)
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    return config_path, run_dir


def test_saved_effective_resume_config_is_strictly_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognize only the exact run-owned config and validate it as resolved state."""
    config_path, run_dir = _saved_run_config(tmp_path, monkeypatch)

    result = preflight.validate_workflow(config_path, requested_workflow="train", resume=run_dir)

    assert result.family == preflight.EXPERIMENT_FAMILY
    assert result.task == "steady_flow"
    assert result.source_path == config_path.resolve()


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 4},
        {"train_dataset": "another_dataset"},
        {"train_seconds": 10.0},
        {"rollout_length": 4},
        {"lr_scale": 0.5},
        {"revision": "fixed"},
    ],
)
def test_saved_effective_resume_refuses_semantic_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
) -> None:
    """Do not reinterpret a saved effective config using authored-config overrides."""
    config_path, run_dir = _saved_run_config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="does not accept semantic overrides"):
        preflight.validate_workflow(config_path, requested_workflow="train", resume=run_dir, **override)


def test_resume_preflight_is_train_only(tmp_path: Path) -> None:
    """Keep the exact-resume option out of the Optuna workflow."""
    path = configs.write_yaml(tmp_path / "study.yaml", configs.optuna_config())

    with pytest.raises(ValueError, match="--resume applies only to the train workflow"):
        preflight.validate_workflow(path, requested_workflow="optuna", resume=tmp_path / "run")


def test_saved_resume_preflight_does_not_recompute_layout_identity_with_the_live_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The implementation that runs a saved config is chosen after the preflight, so only its structure is checked here."""
    config_path, run_dir = _saved_run_config(tmp_path, monkeypatch)
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    saved["run"]["layout"]["a0_id"] = "f" * 16  # what a scientific edit after the run does to the live allocation digest
    config_path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    before = config_path.read_bytes()

    result = preflight.validate_workflow(config_path, requested_workflow="train", resume=run_dir)

    assert result.unit_run_dirs == (run_dir.resolve(),)
    assert config_path.read_bytes() == before
    with pytest.raises(loader.ConfigError, match="Initial run layout identity"):
        loader.validate_resolved_config(saved)


def test_saved_resume_preflight_refuses_another_stage_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage selection other than the saved stage is refused as the run service refuses it."""
    config_path, run_dir = _saved_run_config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="only its exact saved stage"):
        preflight.validate_workflow(config_path, requested_workflow="train", resume=run_dir, train_stages=("b",))


def test_saved_single_resume_preflight_accepts_explicit_renamed_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact self-contained saved config remains usable at its renamed path."""
    config_path, run_dir = _saved_run_config(tmp_path, monkeypatch)
    before = config_path.read_bytes()
    renamed = tmp_path / "storage" / "renamed_resume_bundle"
    run_dir.rename(renamed)
    saved = renamed / "config.yaml"
    result = preflight.validate_workflow(saved, requested_workflow="train", resume=renamed)
    assert result.source_path == saved.resolve()
    assert saved.read_bytes() == before


def test_saved_branched_resume_preflight_rejects_lost_parent_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent-linked B recovery cannot acquire direct-bundle relocation semantics."""
    storage = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    plan = configs.resolve_plan(configs.transient_matched_config(), output_root=storage / "03_experiments")
    run_dir = common.paths.resolve_config_run_dir(plan.b)
    config_path = configs.write_yaml(run_dir / "config.yaml", dict(plan.b))
    before = config_path.read_bytes()
    renamed = storage / "renamed_b_bundle"
    run_dir.rename(renamed)
    saved = renamed / "config.yaml"
    with pytest.raises(ValueError, match="Physical result path contradicts"):
        preflight.validate_workflow(saved, requested_workflow="train", resume=renamed)
    assert saved.read_bytes() == before
