"""Protect Optuna semantics with artificial configs and controlled fake trials.

Responsibilities:
  - Check search admission, exact types, sampler/pruner projection, transient A0 study admission, and package identity

Design principles:
  - Invalid searches and physical-package drift fail before study allocation

This module does NOT:
  - Sample third-party outcome goldens or run production optimization
"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
from support import configs, optuna_trials
from support.synthetic_task import admitted_manifest

from src import common, datasets, experiments

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

optuna_runtime = experiments.tuning.optuna
search_space = experiments.tuning.search_space


def _load_a0_study(tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load a test-owned transient A0 study against an exact test-owned Dataset ID."""
    path = configs.write_yaml(tmp_path / "synthetic-a0-study.yaml", configs.optuna_transient_a0_config())
    return optuna_runtime.load_optuna_study_config(path, train_dataset="synthetic_transient_train")


def _write_study(
    tmp_path: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write one compact test-owned study request."""
    selected = copy.deepcopy(payload or configs.optuna_config())
    selected["experiment"]["data"]["train_dataset"] = "synthetic_train"
    return configs.write_yaml(tmp_path / "synthetic-study.yaml", selected)


def test_minimal_optuna_config_resolves_without_production_files(
    tmp_path: Path,
) -> None:
    """Resolve complete study, objective, and search policy from artificial YAML."""
    configured = configs.optuna_config(multivariate=False)
    study_path = _write_study(tmp_path, configured)
    study = optuna_runtime.load_optuna_study_config(study_path)

    assert study.study["name"] == study_path.stem
    assert (study.base_config["run"]["layout"]["family"], study.base_config["run"]["layout"]["config_name"]) == ("optuna", study_path.stem)
    assert study.study["seed"] == configured["study"]["seed"]
    assert study.study["sampler"]["multivariate"] is (configured["study"]["sampler"]["multivariate"])
    assert study.base_config["data"]["train_dataset"] == "synthetic_train"
    assert study.base_config["run"]["seed"] == (configured["experiment"]["run"]["seed"])
    objective = experiments.config.loader.get_resolved_objective(study.base_config)
    assert study.study["objective"] == objective["id"]
    assert study.study["direction"] == objective["direction"]


def test_epoch_override_is_applied_before_optuna_resolution_and_signature(
    tmp_path: Path,
) -> None:
    """Bind a study's total epoch override to its scientific signature."""
    payload = configs.optuna_config(multivariate=False)
    path = _write_study(tmp_path, payload)
    default = optuna_trials.with_configured_identities(optuna_runtime.load_optuna_study_config(path))
    shortened = optuna_trials.with_configured_identities(optuna_runtime.load_optuna_study_config(path, epochs=2, train_dataset="synthetic_ood"))
    dataset_changed = optuna_trials.with_configured_identities(optuna_runtime.load_optuna_study_config(path, train_dataset="synthetic_ood"))

    assert shortened.base_config["training"]["epochs"] == 2
    assert shortened.base_experiment["training"]["epochs"] == 2
    assert shortened.base_config["data"]["train_dataset"] == "synthetic_ood"
    assert shortened.base_experiment["data"]["train_dataset"] == "synthetic_ood"
    default_signature = optuna_runtime.build_study_signature(default)["digest"]
    assert optuna_runtime.build_study_signature(shortened)["digest"] != default_signature
    assert optuna_runtime.build_study_signature(dataset_changed)["digest"] != default_signature
    assert payload["experiment"]["training"]["epochs"] != 2


@pytest.mark.parametrize(
    ("spec", "error", "match"),
    [
        (
            {
                "name": "rate",
                "kind": "float",
                "low": True,
                "high": 1.0,
            },
            TypeError,
            "finite number",
        ),
        (
            {
                "name": "rate",
                "kind": "float",
                "low": 1.0,
                "high": 0.5,
            },
            ValueError,
            "low < high",
        ),
        (
            {
                "name": "rate",
                "kind": "unsupported",
                "low": 0.1,
                "high": 1.0,
            },
            ValueError,
            "must be one of",
        ),
    ],
)
def test_invalid_search_spec_fails_before_sampling(
    spec: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    """Reject representative type, range, and kind errors in the parser."""
    with pytest.raises(error, match=match):
        search_space.parse_search_space({"optimizer.lr": spec})


def test_unapproved_search_path_is_rejected_during_load(
    tmp_path: Path,
) -> None:
    """Prevent a study from sampling task-owned objective semantics."""
    configured = configs.optuna_config()
    configured["search_space"]["evaluation.objective.id"] = {
        "name": "objective",
        "kind": "categorical",
        "values": ["normalized_group_macro_rmse"],
    }

    with pytest.raises(ValueError, match="not approved"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["study"].update({"seed": 1.0}),
        lambda payload: payload["study"].update({"n_trials": True}),
        lambda payload: payload["study"]["sampler"].update({"multivariate": 1}),
    ],
)
def test_study_scalars_require_exact_types(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    """Reject scalar coercion before creating sampler or storage state."""
    configured = configs.optuna_config()
    mutation(configured)

    with pytest.raises(TypeError):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))


def test_sampler_pruner_and_metadata_mirror_configured_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass configured sampler and pruner settings through without snapshots."""
    runtime_environment = {
        "python": "3.12.1",
        "torch": "2.5.0",
        "neuraloperator_implementation_sha256": "a" * 64,
        "sif_sha256": None,
    }
    environment_reads: list[int] = []

    def read_environment() -> dict[str, Any]:
        environment_reads.append(1)
        return dict(runtime_environment)

    monkeypatch.setattr(
        experiments.tuning.optuna.learning.training.checkpoint,
        "numerical_environment_identity",
        read_environment,
    )
    configured = configs.optuna_config(multivariate=False)
    loaded = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))
    external_study = copy.deepcopy(loaded.study)
    external_study["storage"] = "external://synthetic-study"
    loaded = replace(loaded, study=external_study)
    captured: dict[str, Any] = {}

    def capture_sampler(**settings: Any) -> object:
        captured["sampler"] = settings
        return object()

    def capture_pruner(**settings: Any) -> object:
        captured["pruner"] = settings
        return object()

    user_attrs: dict[str, Any] = {}
    fake_study = SimpleNamespace(
        direction=SimpleNamespace(name="MINIMIZE"),
        user_attrs=user_attrs,
        trials=[],
        set_user_attr=user_attrs.__setitem__,
        optimize=lambda *_args, **_kwargs: None,
    )

    def create_study(**settings: Any) -> Any:
        assert environment_reads == [1]
        captured["create_study"] = settings
        return fake_study

    monkeypatch.setattr(optuna.samplers, "TPESampler", capture_sampler)
    monkeypatch.setattr(optuna.pruners, "MedianPruner", capture_pruner)
    monkeypatch.setattr(optuna, "create_study", create_study)
    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest", admitted_manifest)

    result = optuna_runtime.run_optuna_study(
        loaded,
        n_trials=1,
        output_root=tmp_path / "outputs",
    )

    sampler_config = loaded.study["sampler"]
    pruner_config = loaded.study["pruner"]
    assert result is fake_study
    assert captured["sampler"] == {
        "seed": loaded.study["seed"],
        "multivariate": sampler_config["multivariate"],
    }
    assert captured["pruner"] == {
        "n_startup_trials": pruner_config["n_startup_trials"],
        "n_warmup_steps": pruner_config["n_warmup_steps"],
        "interval_steps": pruner_config["interval_steps"],
    }
    assert user_attrs["sampler_metadata"] == {
        "kind": sampler_config["kind"],
        "multivariate": sampler_config["multivariate"],
        "seed": loaded.study["seed"],
    }
    assert user_attrs["numerical_environment"] == runtime_environment
    assert environment_reads == [1]
    assert captured["create_study"]["sampler"] is not None
    assert captured["create_study"]["pruner"] is not None


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("stage_b_only", r"study\.mode"),
        ("standard", "supports only the stage_a_only study mode"),
    ],
)
def test_transient_optuna_admits_only_stage_a_only_mode(tmp_path: Path, mode: str, match: str) -> None:
    """Reject unknown study modes and the ordinary mode for a transient A0 study."""
    loaded = _load_a0_study(tmp_path)
    invalid = copy.deepcopy(loaded.study)
    invalid["mode"] = mode

    with pytest.raises(ValueError, match=match):
        optuna_runtime._validate_study_contract(replace(loaded, study=invalid))


def test_steady_study_rejects_the_transient_stage_a_only_mode(tmp_path: Path) -> None:
    """stage_a_only names the transient A0 protocol and must not enter a steady study identity."""
    configured = configs.optuna_config()
    configured["study"]["mode"] = "stage_a_only"

    with pytest.raises(ValueError, match="transient_drying study mode"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))


def test_transient_optuna_rejects_matched_plan_experiment(tmp_path: Path) -> None:
    """Train matched A0/B/A1 plans instead of searching them, at load and on revalidation."""
    payload = configs.optuna_transient_a0_config()
    payload["experiment"]["training"] = {"mode": "matched_a1_b"}
    match = "tune A0 only through stage_a_only"

    with pytest.raises(ValueError, match=match):
        optuna_runtime.load_optuna_study_config(configs.write_yaml(tmp_path / "matched-study.yaml", payload))

    loaded = _load_a0_study(tmp_path)
    with pytest.raises(ValueError, match=match):
        optuna_runtime._validate_study_contract(replace(loaded, base_experiment=payload["experiment"]))


def _load_task_study(task: str, tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load the test-owned study of one task."""
    return optuna_runtime.load_optuna_study_config(_write_study(tmp_path)) if task == "steady_flow" else _load_a0_study(tmp_path)


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_study_identity_binds_manifest_evidence_through_the_one_reader(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the same manifest identity fields and payload digests for both tasks, read under the configured storage root."""
    config = _load_task_study(task, tmp_path).base_config
    captured: list[tuple[str, Path]] = []

    def load_manifest(task: str, dataset_id: str, *, storage_root: Path) -> dict[str, Any]:
        captured.append((task, storage_root))
        return admitted_manifest(task, dataset_id)

    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest", load_manifest)

    identities = optuna_runtime._configured_dataset_identities(config)

    additional = config["data"]["additional_evaluation_datasets"]
    root = common.paths.storage_root_for_datasets_root(config["paths"]["dataset_root"])
    assert captured == [(task, root)] * (1 + len(additional))
    assert [item["dataset_id"] for item in identities["additional_evaluation"]] == additional
    assert identities["train"]["payload_sha256"] == "a" * 64
    assert identities["train"]["derived_payload_sha256"] == ("c" * 64 if task == "transient_drying" else None)
    assert identities["train"]["source_case_identity_digest"] == common.serialization.canonical_json_sha256(
        [{"case_id": "case_a"}, {"case_id": "case_b"}]
    )


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_dry_run_and_real_study_share_one_signature(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sign the dry-run description and the real invocation of either task identically, Dataset evidence included."""
    loaded = _load_task_study(task, tmp_path)
    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest", admitted_manifest)
    signatures: list[dict[str, Any]] = []
    build = optuna_runtime.build_study_signature

    def recording_build(config: optuna_runtime.OptunaStudyConfig) -> dict[str, Any]:
        signatures.append(build(config))
        return signatures[-1]

    def stop_after_signature() -> None:
        message = "stop before study storage"
        raise RuntimeError(message)

    monkeypatch.setattr(optuna_runtime, "build_study_signature", recording_build)
    monkeypatch.setattr(optuna_runtime.learning.training.checkpoint, "numerical_environment_identity", stop_after_signature)
    with pytest.raises(RuntimeError, match="stop before study storage"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1, output_root=tmp_path / "outputs")

    described = optuna_runtime.describe_optuna_study_config(loaded)["semantic_signature"]

    assert signatures[0] == described
    assert described["payload"]["dataset_roles"]["roles"]["train"]["dataset_digest"] == "d" * 64


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_study_run_binds_its_physical_dataset_before_database(
    task: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail an invocation of either task before creating study state when its Dataset package is absent."""
    loaded = _load_task_study(task, tmp_path)
    output_root = tmp_path / "outputs"

    def missing_package(*_args: Any, **_kwargs: Any) -> Any:
        message = "synthetic Dataset package is absent"
        raise FileNotFoundError(message)

    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest", missing_package)

    with pytest.raises(FileNotFoundError, match="synthetic Dataset package is absent"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1, output_root=output_root)

    assert not output_root.exists()


def test_transient_study_signature_binds_physical_package_identity(tmp_path: Path) -> None:
    """Change study identity when validated physical Dataset evidence changes."""
    loaded = _load_a0_study(tmp_path)
    first = replace(loaded, dataset_identities={"train": {"dataset_digest": "a" * 64}, "additional_evaluation": []})
    second = replace(loaded, dataset_identities={"train": {"dataset_digest": "b" * 64}, "additional_evaluation": []})

    first_signature = optuna_runtime.build_study_signature(first)
    second_signature = optuna_runtime.build_study_signature(second)

    assert first_signature["digest"] != second_signature["digest"]
    assert first_signature["payload"]["dataset_roles"]["binding"] == "validated_physical_packages"


def test_transient_optuna_rejects_noncentral_declared_objective(tmp_path: Path) -> None:
    """Keep Optuna selection bound to the TaskSpec drying group-macro metric."""
    loaded = _load_a0_study(tmp_path)
    config = copy.deepcopy(loaded.base_config)
    alternate = next(metric for metric in config["evaluation"]["metrics"] if metric["id"] == "normalized_rmse_T")
    config["evaluation"]["objective"] = copy.deepcopy(alternate)
    config = experiments.config.loader.finalize_run_layout(config)

    with pytest.raises(ValueError, match="TaskSpec-owned normalized drying group-macro"):
        optuna_runtime._validate_transient_study_policy(config, loaded.search_space)
