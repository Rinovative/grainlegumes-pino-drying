# ruff: noqa: S101, SLF001
"""Protect Optuna semantics with artificial configs and controlled fake trials."""

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
_TRANSIENT_ROLLOUT_HORIZON = 32
search_space = experiments.tuning.search_space


def _write_study(
    tmp_path: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write one compact test-owned study request."""
    return configs.write_yaml(
        tmp_path / "synthetic-study.yaml",
        payload or configs.optuna_config(),
    )


def test_minimal_optuna_config_resolves_without_production_files(
    tmp_path: Path,
) -> None:
    """Resolve complete study, objective, and search policy from artificial YAML."""
    configured = configs.optuna_config(multivariate=False)
    study = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))

    assert study.study["name"] == configured["study"]["name"]
    assert study.study["seed"] == configured["study"]["seed"]
    assert study.study["sampler"]["multivariate"] is (configured["study"]["sampler"]["multivariate"])
    assert study.base_config["data"]["train_dataset"] == "synthetic_train"
    assert study.base_config["run"]["seed"] == (configured["experiment"]["run"]["seed"])
    objective = experiments.config.loader.get_resolved_objective(study.base_config)
    assert study.study["objective"] == objective["id"]
    assert study.study["direction"] == objective["direction"]


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
    assert captured["create_study"]["sampler"] is not None
    assert captured["create_study"]["pruner"] is not None


@pytest.mark.parametrize(
    "filename",
    [
        "fno_stage_a.yaml",
        "fno_stage_b.yaml",
        "uno_stage_a.yaml",
        "uno_stage_b.yaml",
        "rno_stage_a.yaml",
        "rno_stage_b.yaml",
    ],
)
def test_transient_optuna_recipes_admit_strictly(filename: str) -> None:
    """Keep every maintained transient study recipe inside the loader contract."""
    path = Path("configs/learning/transient_drying/optuna") / filename
    loaded = optuna_runtime.load_optuna_study_config(path)
    assert loaded.base_config["task"] == "transient_drying"
    if loaded.base_config["training"]["stage"] == "b":
        assert loaded.base_config["training"]["fixed_evaluation_horizon"] == _TRANSIENT_ROLLOUT_HORIZON
        assert loaded.base_config["training"]["curriculum"]["lengths"][-1] == _TRANSIENT_ROLLOUT_HORIZON


def test_transient_stage_b_rejects_model_search_path() -> None:
    """Protect exact restored Stage-B model identity from trial overrides."""
    loaded = optuna_runtime.load_optuna_study_config(Path("configs/learning/transient_drying/optuna/fno_stage_b.yaml"))
    forbidden = search_space.SearchSpaceParameter(
        path="model.params.hidden_channels",
        name="hidden",
        kind="categorical",
        values=(64,),
    )
    with pytest.raises(ValueError, match="model, optimizer, and loss"):
        optuna_runtime._validate_transient_study_policy(
            loaded.base_config,
            (*loaded.search_space, forbidden),
        )


def test_transient_identity_uses_storage_derived_package_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind compact transient index evidence through the canonical storage owner."""
    loaded = optuna_runtime.load_optuna_study_config(Path("configs/learning/transient_drying/optuna/fno_stage_a.yaml"))
    config = copy.deepcopy(loaded.base_config)
    config["paths"]["storage_root"] = str(tmp_path / "storage")
    config["paths"].pop("dataset_packages_root", None)
    package_root = tmp_path / "canonical-packages"
    captured: dict[str, list[Path]] = {}

    def load_manifest(dataset_id: str, *, storage_root: Path) -> dict[str, object]:
        captured.setdefault("storage_roots", []).append(storage_root)
        return {
            "dataset_id": dataset_id,
            "dataset_view": "transient_drying",
            "schema_kind": "dataset_package",
            "schema_version": 1,
            "payload_filename": "compact-index.json",
            "payload_sha256": "a" * 64,
            "channel_contract_digest": "b" * 64,
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

    monkeypatch.setattr(common.paths, "get_dataset_packages_root", lambda *, storage_root: (storage_root, package_root)[1])
    monkeypatch.setattr(datasets.dataset_packages, "load_package_manifest_evidence", load_manifest)
    monkeypatch.setattr(datasets.packages.trajectory, "load_transient_index", load_index)

    identities = optuna_runtime._configured_dataset_identities(config)

    assert captured["storage_roots"] == [Path(config["paths"]["storage_root"])] * 2
    assert captured["index_paths"] == [
        package_root / config["data"]["train_dataset"] / "compact-index.json",
        package_root / config["data"]["ood_datasets"][0] / "compact-index.json",
    ]
    assert identities["id"]["manifest_payload_sha256"] == "a" * 64
    assert identities["id"]["index_digest"] == "c" * 64


def test_transient_optuna_rejects_noncentral_declared_objective() -> None:
    """Keep Optuna selection bound to the TaskSpec drying group-macro metric."""
    loaded = optuna_runtime.load_optuna_study_config(Path("configs/learning/transient_drying/optuna/fno_stage_a.yaml"))
    config = copy.deepcopy(loaded.base_config)
    alternate = next(metric for metric in config["evaluation"]["metrics"] if metric["id"] == "normalized_rmse_T")
    config["evaluation"]["objective"] = copy.deepcopy(alternate)

    with pytest.raises(ValueError, match="TaskSpec-owned normalized drying group-macro"):
        optuna_runtime._validate_transient_study_policy(config, loaded.search_space)
