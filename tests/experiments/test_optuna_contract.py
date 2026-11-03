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
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
from support import configs

from src import common, datasets, experiments

if TYPE_CHECKING:
    from collections.abc import Callable

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
    default = optuna_runtime.load_optuna_study_config(path)
    shortened = optuna_runtime.load_optuna_study_config(path, epochs=2, train_dataset="synthetic_ood")
    dataset_changed = optuna_runtime.load_optuna_study_config(path, train_dataset="synthetic_ood")

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


def test_transient_identity_uses_storage_derived_package_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind compact transient index evidence through the canonical storage owner."""
    loaded = _load_a0_study(tmp_path)
    config = copy.deepcopy(loaded.base_config)
    config["paths"]["storage_root"] = str(tmp_path / "storage")
    config["paths"].pop("dataset_packages_root", None)
    package_root = tmp_path / "canonical-packages"
    captured: dict[str, list[Path]] = {}

    def load_manifest(task: str, dataset_id: str, *, storage_root: Path) -> dict[str, object]:
        assert task == "transient_drying"
        captured.setdefault("storage_roots", []).append(storage_root)
        return {
            "dataset_id": dataset_id,
            "dataset_view": "transient_drying",
            "dataset_digest": "d" * 64,
            "schema_kind": "dataset_selection_manifest",
            "schema_version": 1,
            "payload_filename": "dataset.json",
            "payload_sha256": "a" * 64,
            "channel_contract_digest": "b" * 64,
            "materials": ["lentil", "chickpea"],
            "source_case_identities": [{"case_id": "case_a"}, {"case_id": "case_b"}],
        }

    def load_index(path: Path) -> dict[str, object]:
        captured.setdefault("index_paths", []).append(path)
        return {
            "dataset_id": path.parent.name,
            "contract_digest": "b" * 64,
            "index_digest": "c" * 64,
            "sample_count": 4,
            "source_case_count": 2,
        }

    monkeypatch.setattr(common.paths, "get_datasets_root", lambda *, storage_root: (storage_root, package_root)[1])
    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest_evidence", load_manifest)
    monkeypatch.setattr(datasets.packages.trajectory, "load_transient_index", load_index)

    identities = optuna_runtime._configured_dataset_identities(config)

    assert captured["storage_roots"] == [Path(config["paths"]["storage_root"])]
    assert captured["index_paths"] == [package_root / "transient_drying" / config["data"]["train_dataset"] / "dataset.json"]
    assert identities["train"]["manifest_payload_sha256"] == "a" * 64
    assert identities["train"]["index_digest"] == "c" * 64
    assert identities["train"]["materials"] == ["lentil", "chickpea"]
    assert identities["additional_evaluation"] == []
    assert identities["train"]["source_case_identity_digest"] == common.serialization.canonical_json_sha256(
        [{"case_id": "case_a"}, {"case_id": "case_b"}]
    )


def test_transient_run_requires_physical_package_before_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a transient invocation before creating study state when its package is absent."""
    loaded = _load_a0_study(tmp_path)
    base = copy.deepcopy(loaded.base_config)
    base["paths"]["storage_root"] = str(tmp_path / "missing-storage")
    output_root = tmp_path / "outputs"

    def missing_package(_task: str, _dataset_id: str, **_kwargs: Any) -> Any:
        message = "synthetic Dataset package is absent"
        raise FileNotFoundError(message)

    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest_evidence", missing_package)

    with pytest.raises(FileNotFoundError):
        optuna_runtime.run_optuna_study(
            replace(loaded, base_config=base),
            n_trials=1,
            output_root=output_root,
        )

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
