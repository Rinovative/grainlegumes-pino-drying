"""Protect Optuna semantics with artificial configs and controlled fake trials.

Responsibilities:
  - Check search admission, exact types, sampler/pruner projection, the trial budget, transient A0 study admission, and package identity
  - Check that the median pruner starts after trained trials only, never counting a repeated trial

Design principles:
  - Invalid searches and physical-package drift fail before study allocation

This module does NOT:
  - Sample third-party outcome goldens or run production optimization
"""

from __future__ import annotations

import copy
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import optuna
import pytest
import yaml
from support import configs, optuna_trials
from support.synthetic_task import admitted_manifest

from src import common, datasets, experiments
from src.experiments.config import experiments_config_preflight as preflight
from src.experiments.config import experiments_config_transient_plan as transient_plan

if TYPE_CHECKING:
    from collections.abc import Callable

_REPOSITORY = Path(__file__).resolve().parents[2]

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

    # The override sets the budget K, which this study's trials train whole.
    assert (shortened.study["full_budget_epochs"], shortened.base_config["training"]["budget_epochs"]) == (2, 2)
    assert shortened.base_config["data"]["train_dataset"] == "synthetic_ood"
    default_signature = optuna_runtime.build_study_signature(default)["digest"]
    assert optuna_runtime.build_study_signature(shortened)["digest"] != default_signature
    assert optuna_runtime.build_study_signature(dataset_changed)["digest"] != default_signature
    assert payload["experiment"]["training"]["budget_epochs"] != 2


def test_a_trial_trains_the_study_fraction_of_the_budget_and_the_study_identity_binds_both(tmp_path: Path) -> None:
    """Half of K = 4 is a 2-epoch trial; the whole of K = 2 trains as long but is another study, and a fraction that splits an epoch is refused."""
    payload = configs.optuna_config()
    payload["study"]["budget_fraction"] = 0.5
    half = optuna_trials.with_configured_identities(optuna_runtime.load_optuna_study_config(_write_study(tmp_path, payload), epochs=4))
    whole = optuna_trials.with_configured_identities(optuna_runtime.load_optuna_study_config(_write_study(tmp_path), epochs=2))

    assert (half.study["full_budget_epochs"], half.study["budget_fraction"], half.base_config["training"]["budget_epochs"]) == (4, 0.5, 2)
    assert whole.base_config["training"]["budget_epochs"] == 2
    assert optuna_runtime.build_study_signature(half)["payload"]["trial_budget"] == {
        "full_budget_epochs": 4,
        "budget_fraction": 0.5,
        "budget_epochs": 2,
    }
    assert optuna_runtime.build_study_signature(half)["digest"] != optuna_runtime.build_study_signature(whole)["digest"]
    with pytest.raises(ValueError, match="choose K so that the trial budget is whole"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, payload), epochs=3)
    for fraction in (0.0, 1.5, None):
        payload["study"]["budget_fraction"] = fraction
        with pytest.raises(ValueError, match=r"budget_fraction must be a number in \(0, 1\]"):
            optuna_runtime.load_optuna_study_config(_write_study(tmp_path, payload))
    drifted = replace(half, study={**half.study, "budget_fraction": 1.0})
    with pytest.raises(ValueError, match="not the study's trial budget"):
        optuna_runtime.build_study_signature(drifted)


def test_a_trial_run_identity_names_the_trial_budget_its_k_and_the_schedule_its_study_binds(tmp_path: Path) -> None:
    """A trial's seed group and identity carry the study's trial budget, the sampled k and the bound schedule; another k is another run."""
    payload = configs.optuna_config()
    payload["study"]["budget_fraction"] = 0.5
    payload["search_space"]["training.units_per_update"] = _UNITS_PER_UPDATE
    study = optuna_trials.with_configured_identities(optuna_runtime.load_optuna_study_config(_write_study(tmp_path, payload), epochs=4))
    signature = optuna_runtime.build_study_signature(study)["payload"]

    def trial_layout(k: int) -> dict[str, Any]:
        trial = optuna.trial.FixedTrial({"hidden_channels": 8, "learning_rate": 1.0e-3, "units_per_update": k})
        # Optuna's trial number is a read-only property, which the driver's protocol states as an attribute.
        return optuna_runtime._prepare_trial_config(study, cast("Any", trial), sampler_seed=0)[0]["run"]["layout"]

    layout = trial_layout(4)
    schedule = common.names.text_token(signature["scientific_base_config"]["scheduler"]["kind"])
    assert (layout["budget_epochs"], layout["units_per_update"], layout["schedule"]) == (signature["trial_budget"]["budget_epochs"], 4, schedule)
    assert common.paths.seed_group_name(layout) == f"seed-{study.base_config['run']['seed']}__ep-2__k-4__sch-{schedule}"
    assert trial_layout(2)["a0_id"] != layout["a0_id"]


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


def test_an_authored_study_storage_is_refused(tmp_path: Path) -> None:
    """Every study keeps its SQLite database in its own directory, so the study lock there binds every invocation."""
    configured = configs.optuna_config()
    configured["study"]["storage"] = "sqlite:///elsewhere.db"

    with pytest.raises(ValueError, match=r"unknown key\(s\): \['storage'\]"):
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

    def capture_objective(_config: Any, *, sampler_seed: int) -> Callable[[Any], float]:
        captured["trial_sampler_seed"] = sampler_seed
        return lambda _trial: 0.0

    monkeypatch.setattr(optuna.samplers, "TPESampler", capture_sampler)
    monkeypatch.setattr(optuna.pruners, "MedianPruner", capture_pruner)
    monkeypatch.setattr(optuna, "create_study", create_study)
    monkeypatch.setattr(optuna_runtime, "create_objective", capture_objective)
    monkeypatch.setattr(datasets.packages.manifest, "load_package_manifest", admitted_manifest)

    result = optuna_runtime.run_optuna_study(
        loaded,
        n_trials=1,
        output_root=tmp_path / "outputs",
    )

    sampler_config = loaded.study["sampler"]
    pruner_config = loaded.study["pruner"]
    assert result is fake_study
    # A new study stores no trial yet; its sampler seed derives from the study seed and that count, which the invocation
    # records, and its trials record the same derived seed.
    assert captured["sampler"] == {
        "seed": common.random.derive_subseed(loaded.study["seed"], "optuna_sampler_after_0_trials") % 2**32,
        "multivariate": sampler_config["multivariate"],
    }
    assert captured["trial_sampler_seed"] == captured["sampler"]["seed"]
    assert [record["stored_trials"] for key, record in user_attrs.items() if key.startswith("invocation/")] == [0]
    assert captured["pruner"] == {
        "n_startup_trials": pruner_config["n_startup_trials"],
        "n_min_trials": pruner_config["n_min_trials"],
        "n_warmup_steps": pruner_config["n_warmup_steps"],
        "interval_steps": pruner_config["interval_steps"],
    }
    assert user_attrs["sampler_metadata"] == {
        "kind": sampler_config["kind"],
        "multivariate": sampler_config["multivariate"],
        "seed": loaded.study["seed"],
        "invocation_seed": "derived_from_seed_and_stored_trial_count",
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


def test_the_median_pruner_states_its_start_point_and_compares_five_trials_at_a_point(tmp_path: Path) -> None:
    """The start point is a config value below 20; pruning waits for 5 completed trials and 5 reports at the same point by default."""
    configured = configs.optuna_config()
    configured["study"]["pruner"] = {"kind": "median", "n_warmup_steps": 4}
    configured["study"]["n_trials"] = 6
    loaded = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))
    assert loaded.study["pruner"] == {"kind": "median", "n_startup_trials": 5, "n_min_trials": 5, "n_warmup_steps": 4, "interval_steps": 1}

    configured["study"]["pruner"] = {"kind": "median", "n_startup_trials": 1}
    with pytest.raises(KeyError, match="n_warmup_steps"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))
    invalid = copy.deepcopy(loaded.study)
    invalid["pruner"] = {"kind": "median", "n_startup_trials": 1, "n_warmup_steps": 20}
    with pytest.raises(ValueError, match=r"n_warmup_steps counts validation points and must be below 20"):
        optuna_runtime._validate_study_contract(replace(loaded, study=invalid))
    invalid["pruner"] = {"kind": "median", "n_warmup_steps": 4, "n_min_trials": 0}
    with pytest.raises(ValueError, match=r"n_min_trials must be at least 1"):
        optuna_runtime._validate_study_contract(replace(loaded, study=invalid))


def test_median_pruning_starts_after_five_trainings_and_never_counts_a_reused_duplicate(tmp_path: Path) -> None:
    """Four trained trials and two duplicates (complete with a reused value) do not start pruning; a fifth trained trial does."""
    configured = configs.optuna_config()
    # One comparison trial per point, so only the startup count decides.
    configured["study"]["pruner"] = {"kind": "median", "n_warmup_steps": 4, "n_min_trials": 1}
    configured["study"]["n_trials"] = 8
    loaded = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))
    study = optuna.create_study(direction="minimize", pruner=optuna_runtime._build_pruner(loaded.study))
    complete = optuna.trial.TrialState.COMPLETE

    def worse_trial_is_pruned() -> bool:
        trial = study.ask()
        trial.report(2.0, step=4)
        pruned = trial.should_prune()
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return pruned

    for _ in range(4):
        study.add_trial(optuna.trial.create_trial(state=complete, value=1.0, intermediate_values={4: 1.0}))
    for _ in range(2):
        study.add_trial(optuna.trial.create_trial(state=complete, value=1.0, user_attrs={"duplicate_of": "trials/0000"}))
    assert not worse_trial_is_pruned()

    study.add_trial(optuna.trial.create_trial(state=complete, value=1.0, intermediate_values={4: 1.0}))
    assert worse_trial_is_pruned()


def test_steady_study_rejects_the_transient_stage_a_only_mode(tmp_path: Path) -> None:
    """stage_a_only names the transient A0 protocol and must not enter a steady study identity."""
    configured = configs.optuna_config()
    configured["study"]["mode"] = "stage_a_only"

    with pytest.raises(ValueError, match="transient_drying study mode"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))


def test_transient_study_tunes_its_plans_a0_alone_at_the_study_fraction(tmp_path: Path) -> None:
    """The trial is the plan's stage-A-only A0 over half of A0's budget, without B; a single-stage experiment is refused."""
    payload = configs.optuna_transient_a0_config(model_kind="rno")
    payload["study"]["budget_fraction"] = 0.5
    plan = payload["experiment"]["training"]

    loaded = optuna_runtime.load_optuna_study_config(configs.write_yaml(tmp_path / "a0-study.yaml", payload))

    training = loaded.base_config["training"]
    assert (loaded.study["full_budget_epochs"], training["budget_epochs"]) == (plan["a0"]["budget_epochs"], plan["a0"]["budget_epochs"] // 2)
    assert (training["stage"], training["comparison_arm"], training["stage_schedule"]["mode"]) == ("a", "a0", "stage_a_only")
    assert training["units_per_update"] == plan["units_per_update"]
    assert loaded.base_config["temporal"]["sampling"]["mode"] == plan["a0"]["sampling"]["mode"]
    assert "curriculum" not in training  # teacher-forced A0 draws no rollout horizon
    payload["experiment"] = configs.transient_stage_a_only_config()
    with pytest.raises(ValueError, match="tunes the A0 of a matched plan"):
        optuna_runtime.load_optuna_study_config(configs.write_yaml(tmp_path / "single-stage-study.yaml", payload))


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

    additional = config["data"]["material_ood_datasets"]
    root = common.paths.storage_root_for_datasets_root(config["paths"]["dataset_root"])
    assert captured == [(task, root)] * (1 + len(additional))
    assert [item["dataset_id"] for item in identities["material_ood"]] == additional
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
    first = replace(loaded, dataset_identities={"train": {"dataset_digest": "a" * 64}, "material_ood": []})
    second = replace(loaded, dataset_identities={"train": {"dataset_digest": "b" * 64}, "material_ood": []})

    first_signature = optuna_runtime.build_study_signature(first)
    second_signature = optuna_runtime.build_study_signature(second)

    assert first_signature["digest"] != second_signature["digest"]
    assert first_signature["payload"]["dataset_roles"]["binding"] == "validated_physical_packages"


_MAINTAINED_STUDIES = {"steady_flow": "fno_search", "transient_drying": "fno_a0_search"}


def _best_values(study: optuna_runtime.OptunaStudyConfig) -> dict[str, Any]:
    """Return one value per sampled parameter that differs from the study's authored choice where the space allows."""
    return {
        parameter.name: parameter.values[-1] if parameter.kind == "categorical" else parameter.high
        for parameter in study.search_space
        if parameter.kind != "fixed"
    }


def _study_paths(study_dir: Path) -> optuna_runtime._OptunaStudyPaths:
    """Return the paths of a study directory below a test-owned root."""
    study_dir.mkdir(parents=True, exist_ok=True)
    return optuna_runtime._OptunaStudyPaths(study_dir, study_dir / "trials", "sqlite:///s.db", study_dir / "s.db")


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_best_config_trains_once_copied_into_the_experiments_tree(task: str, tmp_path: Path) -> None:
    """The best trial becomes an experiment config on the task base that the maintained loaders admit as a train config."""
    configs_root = _REPOSITORY / "configs/03_learning" / task
    study = optuna_runtime.load_optuna_study_config(configs_root / "optuna" / f"{_MAINTAINED_STUDIES[task]}.yaml", train_dataset="synthetic_dataset")
    best = _best_values(study)
    paths = _study_paths(tmp_path / "study__ep-300")
    summary = {"study_name": "s", "best_trial_number": 7, "best_objective": 0.25, "best_parameters": best}

    optuna_runtime._publish_study_lifecycle_summary(paths, summary, study)

    text = (paths.study_dir / "best_config.yaml").read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header.startswith("# Best trial 7 of study study__ep-300, objective 0.25;")
    if task == "steady_flow":
        label = re.search(r"/([^/<>]+)\.yaml", header)
        assert label is not None
        stem = label.group(1)
        assert "__lr-" in stem  # The copy target is the full config name: the best trial's learning rate is named.
    else:
        # A transient study names only its A0 part; B/A1 settings of the architecture experiment appear in its pair folder, not in the name.
        stem = "best"
        # The best trial's rate, no curriculum; its k names the run's seed group, not the config.
        pinned = "fno_m128x160_h96_l4_p0p1__lr-0p01__a0smp-one-step-transition"
        assert f"the name is {pinned}; B/A1 settings it sets appear in the pair folder" in header
    target = tmp_path / "configs/03_learning" / task
    (target / "experiments/group").mkdir(parents=True)
    shutil.copy(configs_root / "base.yaml", target / "base.yaml")
    config_path = target / "experiments/group" / f"{stem}.yaml"
    config_path.write_text(text, encoding="utf-8")
    shared = experiments.config.loader.load_yaml(configs_root / "base.yaml")

    authored = experiments.config.loader.load_authored_config(config_path)

    assert authored["tracking"]["wandb"]["workflow"] == "train"
    assert authored["run"] == shared["run"]
    if task == "steady_flow":
        result = preflight.inspect_config(config_path, train_dataset="synthetic_dataset")
        # The best trial's k is the only training value it changes.
        assert (result.family, authored["training"]) == ("experiment", {**shared["training"], "units_per_update": best["k"]})
        assert result.run_label.startswith(stem)
    else:
        # A transient matched experiment must also state B's rollout sampling, which no A0 study tunes; a default curriculum stays unstated.
        assert authored["training"]["a0"]["sampling"] == {"mode": "one_step_transition"}
        assert "curriculum" not in yaml.safe_load(text)["training"]["a0"]
        assert authored["training"]["b"] == shared["training"]["b"]
    for parameter in study.search_space:
        node: Any = authored
        for token in parameter.path.split("."):
            node = node[int(token)] if isinstance(node, list) else node[token]
        assert node == best[parameter.name]
    assert transient_plan.is_matched_a1_b_config(authored) == (task == "transient_drying")


_UNITS_PER_UPDATE = {"name": "units_per_update", "kind": "categorical", "values": [2, 4]}


def test_best_config_of_an_external_steady_study_exports_against_the_task_base(tmp_path: Path) -> None:
    """The task's own base.yaml is used wherever the study file lives, and the copied export resolves."""
    study = optuna_runtime.load_optuna_study_config(_write_study(tmp_path / "external"))
    paths = _study_paths(tmp_path / "study")
    best = {"hidden_channels": 12, "learning_rate": 0.002}

    optuna_runtime._publish_study_lifecycle_summary(
        paths, {"study_name": "s", "best_trial_number": 1, "best_objective": 0.5, "best_parameters": best}, study
    )

    text = (paths.study_dir / "best_config.yaml").read_text(encoding="utf-8")
    exported = yaml.safe_load(text)
    assert exported["optimizer"]["lr"] == 0.002
    target = tmp_path / "configs/03_learning/steady_flow"
    (target / "experiments/group").mkdir(parents=True)
    shutil.copy(_REPOSITORY / "configs/03_learning/steady_flow/base.yaml", target / "base.yaml")
    config_path = target / "experiments/group" / f"{re.search(r'/<group>/(.+)[.]yaml to train', text).group(1)}.yaml"  # type: ignore[union-attr]
    config_path.write_text(text, encoding="utf-8")
    assert experiments.config.loader.load_authored_config(config_path)["optimizer"]["lr"] == 0.002
    assert preflight.inspect_config(config_path, train_dataset="synthetic_train").family == "experiment"


def test_best_config_of_a_study_tuning_one_physics_weight_exports_the_whole_weight_groups(tmp_path: Path) -> None:
    """A name token that shows several leaves is exported with all of them, so the copy loads under the name its header states."""
    payload = configs.optuna_config(physics_enabled=True)
    for term in ("momentum_weight", "continuity_weight", "boundary_weight"):
        del payload["experiment"]["loss"]["physics"][term]  # omitted defaults: the study states none of them
    payload["search_space"]["loss.physics.momentum_weight.target"] = {"name": "momentum", "kind": "float", "low": 0.5, "high": 2.0}
    study = optuna_runtime.load_optuna_study_config(_write_study(tmp_path / "external", payload))
    paths = _study_paths(tmp_path / "study")
    best = {"hidden_channels": 12, "learning_rate": 0.002, "momentum": 1.5}

    optuna_runtime._publish_study_lifecycle_summary(
        paths, {"study_name": "s", "best_trial_number": 1, "best_objective": 0.5, "best_parameters": best}, study
    )

    text = (paths.study_dir / "best_config.yaml").read_text(encoding="utf-8")
    physics = yaml.safe_load(text)["loss"]["physics"]
    resolved = study.base_config["loss"]["physics"]
    assert physics["momentum_weight"]["target"] == 1.5
    for term in ("continuity_weight", "boundary_weight"):
        assert physics[term]["target"] == resolved[term]["target"]
    assert {term: physics[term]["warmup"]["epochs"] for term in resolved if term.endswith("_weight")} == {
        term: resolved[term]["warmup"]["epochs"] for term in resolved if term.endswith("_weight")
    }
    target = tmp_path / "configs/03_learning/steady_flow"
    (target / "experiments/group").mkdir(parents=True)
    shutil.copy(_REPOSITORY / "configs/03_learning/steady_flow/base.yaml", target / "base.yaml")
    config_path = target / "experiments/group" / f"{re.search(r'/<group>/(.+)[.]yaml to train', text).group(1)}.yaml"  # type: ignore[union-attr]
    config_path.write_text(text, encoding="utf-8")
    assert preflight.inspect_config(config_path, train_dataset="synthetic_train").family == "experiment"


def test_best_config_of_an_a0_study_puts_a_searched_k_at_the_plan_level(tmp_path: Path) -> None:
    """A searched units-per-update of a transient A0 study exports where the matched plan reads it: once for all stages."""
    payload = configs.optuna_transient_a0_config()
    payload["search_space"]["training.units_per_update"] = _UNITS_PER_UPDATE
    study = optuna_runtime.load_optuna_study_config(
        configs.write_yaml(tmp_path / "external" / "a0-study.yaml", payload), train_dataset="synthetic_transient_train"
    )
    paths = _study_paths(tmp_path / "study")
    best = {"hidden_channels": 12, "learning_rate": 0.002, "units_per_update": 4}

    optuna_runtime._publish_study_lifecycle_summary(
        paths, {"study_name": "s", "best_trial_number": 1, "best_objective": 0.5, "best_parameters": best}, study
    )

    training = yaml.safe_load((paths.study_dir / "best_config.yaml").read_text(encoding="utf-8"))["training"]
    assert training["units_per_update"] == 4
    assert "units_per_update" not in training["a0"]
    # The study's experiment is its plan, so A0's own settings stay under the A0 stage, without a curriculum.
    assert training["a0"]["sampling"] == payload["experiment"]["training"]["a0"]["sampling"]
    assert "curriculum" not in training["a0"]


def test_a_study_without_the_task_base_is_refused_when_it_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The task base the export needs is checked at the start of the study, not after its trials."""
    monkeypatch.setattr(optuna_runtime, "_task_base_path", lambda _task: tmp_path / "missing" / "base.yaml")

    with pytest.raises(ValueError, match="task base"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path))


def test_best_config_follows_the_summary_and_a_build_error_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The summary stays the commit record: a failed build keeps both files, and a missing best trial deletes a stale config."""
    study = optuna_runtime.load_optuna_study_config(
        _REPOSITORY / "configs/03_learning/steady_flow/optuna/fno_search.yaml", train_dataset="synthetic_dataset"
    )
    paths = _study_paths(tmp_path / "study")
    best_path, summary_path = paths.study_dir / "best_config.yaml", paths.study_dir / "study_summary.json"
    best = {"study_name": "s", "best_trial_number": 1, "best_objective": 0.5, "best_parameters": _best_values(study)}
    optuna_runtime._publish_study_lifecycle_summary(paths, best, study)
    before = (best_path.read_bytes(), summary_path.read_bytes())

    def failing_build(*_args: object) -> str:
        message = "synthetic build error"
        raise RuntimeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(optuna_runtime, "_best_experiment_yaml", failing_build)
        with pytest.raises(RuntimeError, match="synthetic build error"):
            optuna_runtime._publish_study_lifecycle_summary(paths, {**best, "best_trial_number": 2}, study)
    assert (best_path.read_bytes(), summary_path.read_bytes()) == before

    optuna_runtime._publish_study_lifecycle_summary(paths, {**best, "best_trial_number": None, "best_objective": None, "best_parameters": {}}, study)
    assert not best_path.exists()
    assert summary_path.is_file()


def test_transient_optuna_rejects_noncentral_declared_objective(tmp_path: Path) -> None:
    """Keep Optuna selection bound to the TaskSpec drying group-macro metric."""
    loaded = _load_a0_study(tmp_path)
    config = copy.deepcopy(loaded.base_config)
    alternate = next(metric for metric in config["evaluation"]["metrics"] if metric["id"] == "normalized_rmse_T")
    config["evaluation"]["objective"] = copy.deepcopy(alternate)
    config = experiments.config.loader.finalize_run_layout(config)

    with pytest.raises(ValueError, match="TaskSpec-owned normalized drying group-macro"):
        optuna_runtime._validate_transient_study_policy(config, loaded.search_space)
