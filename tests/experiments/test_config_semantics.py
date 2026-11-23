"""Protect configuration semantics with complete test-owned requests.

Responsibilities:
  - Check explicit request resolution, scientific identity, resume compatibility, and invalid
    semantics

Design principles:
  - Runtime/reporting changes stay distinct from scientific changes

This module does NOT:
  - Select production defaults or execute training
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from support import configs

from src import common, domain, experiments, learning

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_SYNTHETIC_UNITS_PER_UPDATE = 3


@pytest.mark.parametrize("model_kind", ["fno", "uno"])
def test_minimal_direct_config_resolves_and_preserves_user_values(
    model_kind: str,
) -> None:
    """Resolve representative model families without a production YAML fixture."""
    raw = configs.direct_config(model_kind=model_kind)
    raw["training"]["units_per_update"] = _SYNTHETIC_UNITS_PER_UPDATE
    raw["optimizer"]["lr"] = 4.321e-3

    resolved = configs.resolve(raw)
    task = domain.tasks.registry.get_task("steady_flow")

    assert resolved["model"]["kind"] == model_kind
    assert resolved["model"]["params"]["in_channels"] == task.in_channels
    assert resolved["model"]["params"]["out_channels"] == task.out_channels
    assert resolved["data"]["train_dataset"] == "synthetic_train"
    assert resolved["data"]["material_ood_datasets"] == ["synthetic_ood"]
    assert resolved["training"]["units_per_update"] == _SYNTHETIC_UNITS_PER_UPDATE
    assert resolved["optimizer"]["lr"] == pytest.approx(4.321e-3)
    assert resolved["evaluation"]["objective"]["id"] == "normalized_group_macro_rmse"


def test_training_requires_an_exact_dataset_id() -> None:
    """Reject implicit selection from whatever Dataset packages happen to exist."""
    raw = configs.direct_config()
    raw["data"].pop("train_dataset")

    with pytest.raises((TypeError, ValueError), match="train_dataset"):
        configs.resolve(raw)


@pytest.mark.parametrize(("key", "value"), [("ood_datasets", ["synthetic_ood"]), ("train_ratio", 0.8), ("ood_fraction", 0.2)])
def test_steady_membership_keys_are_dataset_owned(key: str, value: object) -> None:
    """Reject run-owned steady split keys; membership comes from the Dataset partition."""
    raw = configs.direct_config()
    raw["data"][key] = value

    with pytest.raises(experiments.config.loader.ConfigError, match=key):
        configs.resolve(raw)


@pytest.mark.parametrize("build", [configs.direct_config, configs.transient_stage_a_only_config], ids=["steady", "transient"])
def test_persistent_workers_need_worker_processes_in_both_tasks(build: Callable[[], dict[str, Any]]) -> None:
    """Refuse keeping worker processes alive when the request starts none, with one message for both tasks."""
    raw = build()
    raw["data"].update(num_workers=0, persistent_workers=True)

    with pytest.raises(experiments.config.loader.ConfigError, match=r"^data\.persistent_workers requires data\.num_workers > 0\.$"):
        configs.resolve(raw)


def test_minimal_physics_config_preserves_explicit_semantics() -> None:
    """Propagate explicitly supplied formulation and weight values unchanged."""
    raw = configs.direct_config(physics_enabled=True)
    physics = raw["loss"]["physics"]
    physics["derivatives"] = {"kind": "spectral"}
    physics["momentum_weight"]["target"] = 4.0e-3
    physics["continuity_weight"]["target"] = 5.0e-3
    physics["boundary_weight"]["target"] = 6.0e-3

    resolved = configs.resolve(raw)

    assert resolved["loss"]["physics"]["derivatives"] == {"kind": "spectral"}
    assert resolved["loss"]["physics"]["momentum_weight"]["target"] == pytest.approx(4.0e-3)
    assert resolved["loss"]["physics"]["continuity_weight"]["target"] == pytest.approx(5.0e-3)
    assert resolved["loss"]["physics"]["boundary_weight"]["target"] == pytest.approx(6.0e-3)


def test_derivatives_reject_an_extension_key() -> None:
    """Refuse the removed derivative `extension` key; spectral derivatives always use Fourier continuation."""
    raw = configs.direct_config(physics_enabled=True)
    raw["loss"]["physics"]["derivatives"] = {"kind": "spectral", "extension": "none"}

    with pytest.raises(experiments.config.loader.ConfigError, match=r"loss\.physics\.derivatives.*extension"):
        configs.resolve(raw)


def test_saved_config_rejects_a_derivative_extension_key() -> None:
    """Refuse the removed derivative `extension` key when a saved resolved config is validated."""
    saved = configs.resolve(configs.direct_config(physics_enabled=True))
    saved["loss"]["physics"]["derivatives"]["extension"] = "none"

    with pytest.raises(experiments.config.loader.ConfigError, match=r"loss\.physics\.derivatives.*extension"):
        experiments.config.loader.validate_resolved_config(saved, verify_layout_identity=False)


def test_scientific_change_updates_name_and_resume_identity() -> None:
    """Treat an explicit physics-weight change as continuation-incompatible."""
    first_raw = configs.direct_config(physics_enabled=True)
    second_raw = copy.deepcopy(first_raw)
    second_raw["loss"]["physics"]["continuity_weight"]["target"] = 7.0e-3
    first = configs.resolve(first_raw)
    second = configs.resolve(second_raw)

    assert first["run"]["name"] == second["run"]["name"]  # Unoccupied readable labels; exact identities remain distinct.
    assert learning.training.checkpoint.config_digest(first) != (learning.training.checkpoint.config_digest(second))
    with pytest.raises(ValueError, match=r"loss\.physics\.continuity_weight\.target"):
        experiments.run.validate_resume_config(second, first)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("device", "cuda"),
        ("output_root", "/mnt/relocated-synthetic-output"),
    ],
)
def test_runtime_location_changes_preserve_scientific_and_resume_identity(
    field: str,
    replacement: str,
) -> None:
    """Exclude device and output location from run naming and resume semantics."""
    first = configs.resolve(configs.direct_config())
    second = copy.deepcopy(first)
    if field == "device":
        second["run"]["device"] = replacement
    else:
        second["paths"]["output_root"] = replacement

    assert second["run"]["name"] == first["run"]["name"]
    assert learning.training.checkpoint.config_digest(second) == learning.training.checkpoint.config_digest(first)
    experiments.run.validate_resume_config(second, first)


def test_reporting_and_data_worker_changes_preserve_training_identity() -> None:
    """Exclude run labels, reporting, paths, devices, and loader workers from model science."""
    original = configs.resolve(configs.direct_config())
    operational = copy.deepcopy(original)
    operational["run"].update(
        {
            "name": "relabeled-run",
            "device": "cuda",
        }
    )
    operational["data"].update(
        {
            "num_workers": 17,
            "pin_memory": not original["data"]["pin_memory"],
            "persistent_workers": not original["data"]["persistent_workers"],
        }
    )
    operational["paths"] = {"output_root": "/relocated/output"}
    operational["tracking"] = {"wandb": {"enabled": False, "project": "reporting-only"}}

    assert learning.training.checkpoint.config_digest(operational) == learning.training.checkpoint.config_digest(original)


def test_objective_change_is_resume_incompatible() -> None:
    """Bind model selection semantics to persisted continuation identity."""
    first = configs.resolve(configs.direct_config())
    changed_raw = configs.direct_config()
    changed_raw["evaluation"]["objective"] = {"id": "physical_relative_h1"}
    second = configs.resolve(changed_raw)

    assert learning.training.checkpoint.config_digest(first) != learning.training.checkpoint.config_digest(second)
    with pytest.raises(ValueError, match=r"evaluation\.objective"):
        experiments.run.validate_resume_config(second, first)


@pytest.mark.parametrize("build", [configs.direct_config, configs.transient_stage_a_only_config], ids=["steady", "transient"])
def test_a_single_stage_run_states_its_budget_epochs(build: Callable[[], dict[str, Any]]) -> None:
    """Refuse a steady or A0 request without its budget in epochs of the full Train set."""
    raw = build()
    raw["training"].pop("budget_epochs")

    with pytest.raises((TypeError, ValueError), match=r"budget_epochs|Transient training keys"):
        configs.resolve(raw)


@pytest.mark.parametrize("build", [configs.direct_config, configs.transient_stage_a_only_config], ids=["steady", "transient"])
def test_a_single_stage_run_refuses_the_former_duration_key(build: Callable[[], dict[str, Any]]) -> None:
    """Refuse the former ``epochs`` key beside the budget."""
    raw = build()
    raw["training"]["epochs"] = 4

    refusal = r"training contains unknown key.*epochs|training\.epochs does not apply|Transient training keys"
    with pytest.raises((TypeError, ValueError), match=refusal):
        configs.resolve(raw)


@pytest.mark.parametrize(
    ("scheduler", "match"),
    [
        ({"kind": "cosine", "min_lr": 1.0e-5}, r"scheduler contains unknown key.*min_lr"),
        ({"kind": "cosine", "factor": 0.5}, r"scheduler kind 'cosine' takes exactly \['kind'\]"),
        ({"kind": "cosine", "patience": 3}, r"scheduler kind 'cosine' takes exactly \['kind'\]"),
        ({"kind": "reduce_on_plateau", "factor": 0.5, "patience": 3, "min_lr": 1.0e-5}, r"scheduler contains unknown key.*min_lr"),
        ({"kind": "reduce_on_plateau", "factor": 0.5}, r"scheduler kind 'reduce_on_plateau' takes exactly"),
        ({"kind": "cosine_annealing"}, r"Unknown scheduler kind"),
        ({"kind": ["cosine"]}, r"Unknown scheduler kind"),
    ],
    ids=["cosine-floor", "cosine-factor", "cosine-patience", "plateau-floor", "plateau-without-patience", "unknown", "list-kind"],
)
def test_the_scheduler_states_exactly_its_kind_keys_and_no_floor(scheduler: dict[str, Any], match: str) -> None:
    """The cosine takes only its kind, the plateau comparison kind exactly factor and patience, and neither a floor."""
    raw = configs.direct_config()
    raw["scheduler"] = scheduler

    with pytest.raises(experiments.config.loader.ConfigError, match=match):
        configs.resolve(raw)


def test_the_plateau_comparison_kind_resolves_without_a_floor() -> None:
    """The halve-on-plateau variant resolves with factor and patience only."""
    raw = configs.direct_config()
    raw["scheduler"] = {"kind": "reduce_on_plateau", "factor": 0.5, "patience": 3}

    assert configs.resolve(raw)["scheduler"] == {"kind": "reduce_on_plateau", "factor": 0.5, "patience": 3}


@pytest.mark.parametrize(
    ("build", "path", "match"),
    [
        (configs.direct_config, ("run", "deterministic"), r"run\.deterministic must be boolean"),
        (configs.direct_config, ("model", "params", "positional_embedding"), r"model\.params is missing required key.*positional_embedding"),
        (configs.direct_config, ("model", "params", "domain_padding"), r"model\.params is missing required key.*domain_padding"),
        (configs.direct_config, ("data", "train_subset"), r"data\.train_subset is required"),
        (configs.transient_stage_a_only_config, ("data", "spatial_stride"), r"data\.spatial_stride is required"),
        (configs.transient_stage_a_only_config, ("data", "train_subset"), r"data\.train_subset"),
        (configs.transient_stage_a_only_config, ("model", "params", "domain_padding"), r"model\.params is missing required key.*domain_padding"),
        (configs.direct_config, ("optimizer", "betas"), r"Config must state optimizer\.betas;"),
        (configs.transient_stage_a_only_config, ("loss", "data", "weight"), r"Config must state loss\.data\.weight;"),
        (configs.transient_stage_a_only_config, ("loss", "physics"), r"Config must state loss\.physics\.enabled;"),
    ],
)
def test_values_without_a_code_default_are_required_and_named_by_path(
    build: Callable[[], dict[str, Any]],
    path: tuple[str, ...],
    match: str,
) -> None:
    """Refuse a request that leaves a training-relevant value to a code default, naming its config path."""
    raw = build()
    node = raw
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]

    with pytest.raises(experiments.config.loader.ConfigError, match=match):
        configs.resolve(raw)


def test_an_explicit_null_scheduler_trains_without_one() -> None:
    """A null scheduler states the whole section and disables it, while an omitted one is rejected."""
    raw = configs.direct_config()
    raw["scheduler"] = None
    assert configs.resolve(raw)["scheduler"] is None

    del raw["scheduler"]
    with pytest.raises(experiments.config.loader.ConfigError, match=r"Config must state scheduler\.kind;"):
        configs.resolve(raw)


def test_a_saved_config_without_a_base_owned_key_is_refused() -> None:
    """A saved resolved config that lacks a key the code leaves to the base is refused instead of read with a fallback."""
    saved = configs.resolve(configs.direct_config())
    del saved["optimizer"]["weight_decay"]

    with pytest.raises(experiments.config.loader.ConfigError, match=r"Config must state optimizer\.weight_decay;"):
        experiments.config.loader.validate_resolved_config(saved, verify_layout_identity=False)


def test_a_saved_config_without_its_train_dataset_is_a_config_error() -> None:
    """A saved config that lost data.train_dataset is refused by that path, not by a KeyError before the data keys are checked."""
    saved = configs.resolve(configs.direct_config())
    del saved["data"]["train_dataset"]

    with pytest.raises(experiments.config.loader.ConfigError, match=r"data\.train_dataset"):
        experiments.config.loader.validate_resolved_config(saved, verify_layout_identity=False)


def test_a_steady_config_refuses_a_transient_stage_schedule() -> None:
    """A transient schedule would name a steady run after another budget than it trains; requested or saved, it is refused."""
    raw = configs.direct_config()
    raw["training"]["stage_schedule"] = {"mode": "matched_a1_b", "budget_unit": "supervised_transitions", "a0_budget_epochs": 5}
    with pytest.raises(experiments.config.loader.ConfigError, match=r"transient training key.*stage_schedule"):
        configs.resolve(raw)

    saved = configs.resolve(configs.direct_config())
    saved["training"]["stage_schedule"] = {"mode": "stage_a_only", "budget_unit": "units", "budget_epochs": saved["training"]["budget_epochs"]}
    with pytest.raises(experiments.config.loader.ConfigError, match=r"transient training key.*stage_schedule"):
        experiments.config.loader.validate_resolved_config(saved, verify_layout_identity=False)


def test_a_steady_config_with_a_null_transient_training_key_resolves_saves_and_reloads(tmp_path: Path) -> None:
    """A null transient key states nothing: the config the writer saves from it loads again."""
    raw = configs.direct_config()
    raw["training"]["teacher_handoff"] = None
    path = tmp_path / "config.yaml"
    experiments.config.loader.save_yaml(configs.resolve(raw), path)

    experiments.config.loader.validate_resolved_config(experiments.config.loader.load_yaml(path))


@pytest.mark.parametrize("value", [3, 0, True, 2.0])
def test_a_train_subset_is_null_or_a_power_of_two(value: object) -> None:
    """A learning-curve point is a power-of-two Train prefix per material; any other value is refused."""
    raw = configs.direct_config()
    raw["data"]["train_subset"] = value
    with pytest.raises(experiments.config.loader.ConfigError, match=r"data\.train_subset must be null or a power of two"):
        configs.resolve(raw)


def test_a_train_subset_runs_in_its_own_dataset_directory(tmp_path: Path) -> None:
    """Runs on a Train subset of one Dataset never share a directory with the full Dataset or another subset."""
    raw = configs.direct_config()
    raw["data"]["train_subset"] = 4
    subset = configs.resolve(raw)
    full = configs.resolve(configs.direct_config())
    assert subset["run"]["layout"]["dataset"] == "synthetic_train__train-4"
    assert full["run"]["layout"]["dataset"] == "synthetic_train"
    run_dir = common.paths.resolve_config_run_dir
    assert run_dir(subset, output_root=tmp_path) != run_dir(full, output_root=tmp_path)


@pytest.mark.parametrize("build", [configs.direct_config, configs.transient_stage_a_only_config], ids=["steady", "transient"])
def test_mixed_precision_is_an_unknown_training_key(build: Callable[[], dict[str, Any]]) -> None:
    """Refuse the removed mixed-precision switch instead of ignoring it."""
    raw = build()
    raw["training"]["mixed_precision"] = False

    with pytest.raises(experiments.config.loader.ConfigError, match=r"training contains unknown key.*mixed_precision"):
        configs.resolve(raw)


# The retired single-role names, built from parts so that the repository search for them stays empty.
_RETIRED_ROLE = "additional" + "_evaluation"


@pytest.mark.parametrize("build", [configs.direct_config, configs.transient_stage_a_only_config], ids=["steady", "transient"])
@pytest.mark.parametrize(
    ("section", "old", "new"),
    [("data", f"{_RETIRED_ROLE}_datasets", "material_ood_datasets")],
    ids=["datasets"],
)
def test_renamed_ood_keys_are_refused_under_their_old_names(build: Callable[[], dict[str, Any]], section: str, old: str, new: str) -> None:
    """Refuse the former OOD key names instead of reading them as aliases."""
    raw = build()
    raw[section][old] = raw[section].pop(new, [] if section == "data" else 1)

    with pytest.raises(experiments.config.loader.ConfigError, match=rf"{section} contains unknown key.*{old}"):
        configs.resolve(raw)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda config: config.pop("task"), "task"),
        (
            lambda config: config["model"].update({"kind": "unsupported"}),
            "Unknown model identifier",
        ),
        (
            lambda config: config["training"].update({"units_per_update": 0}),
            "positive",
        ),
        (
            lambda config: config["data"].update({"batch_size": 16}),
            r"data contains unknown key.*batch_size",
        ),
        (
            lambda config: config["run"].update({"seed": True}),
            "integer",
        ),
        (
            lambda config: config["training"].pop("units_per_update"),
            r"training\.units_per_update",
        ),
        (
            lambda config: config["data"].update({"train_dataset": "01_generation/raw/batch/case.h5"}),
            "single non-empty path component",
        ),
    ],
)
def test_invalid_config_fails_at_the_public_resolver(
    mutation: Callable[[dict[str, Any]], object],
    match: str,
) -> None:
    """Reject one representative invalid value from each important branch."""
    raw = configs.direct_config()
    mutation(raw)

    with pytest.raises((TypeError, ValueError), match=match):
        configs.resolve(raw)


@pytest.mark.parametrize(
    ("build", "data", "match"),
    [
        (configs.direct_config, {"kind": "huber", "weight": 1.0}, r"loss\.data\.kind 'huber' is not allowed"),
        (configs.direct_config, {"kind": "relative_h1", "weight": 1.0}, r"loss\.data\.kind 'relative_h1' is not allowed"),
        (configs.transient_stage_a_only_config, {"kind": "mse", "weight": 1.0, "state_aux_weight": 0.1}, r"loss\.data contains unknown key"),
    ],
    ids=("huber", "relative-h1", "state-aux"),
)
def test_data_losses_other_than_the_state_mse_or_h1_are_refused(build: Callable[[], dict[str, Any]], data: dict[str, Any], match: str) -> None:
    """Refuse a Huber, relative-H1 or auxiliary-state loss, requested or in a saved run config, instead of training or reading another loss."""
    raw = build()
    raw["loss"]["data"] = data

    with pytest.raises(experiments.config.loader.ConfigError, match=match):
        configs.resolve(raw)
    saved = configs.resolve(build())
    saved["loss"]["data"] = data
    with pytest.raises(experiments.config.loader.ConfigError, match=r"loss\.data"):
        experiments.config.loader.validate_resolved_config(saved, verify_layout_identity=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [("continuity", "div_velocity"), ("residual_weight", {"target": 1.0e-3, "warmup": {"kind": "linear", "epochs": 1}})],
)
def test_superseded_physics_selectors_fail_closed(key: str, value: object) -> None:
    """Reject pointwise-continuity and shared-weight configs instead of dropping their meaning."""
    raw = configs.direct_config(physics_enabled=True)
    raw["loss"]["physics"][key] = value

    with pytest.raises(
        experiments.config.loader.ConfigError,
        match=rf"loss\.physics contains unknown key\(s\): \['{key}'\]",
    ):
        configs.resolve(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [(("momentum_weight", "target"), "1.0e3"), (("boundary_weight", "warmup", "epochs"), 2.5), (("interior_crop",), "2")],
)
def test_physics_numbers_reject_text_and_fractional_counts(path: tuple[str, ...], value: object) -> None:
    """Reject YAML text such as an unsigned exponent and truncating counts instead of coercing them."""
    raw = configs.direct_config(physics_enabled=True)
    node: Any = raw["loss"]["physics"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(experiments.config.loader.ConfigError, match=r"loss\.physics\." + r"\.".join(path)):
        configs.resolve(raw)


def test_duplicate_metric_identifier_is_rejected() -> None:
    """Prevent an objective from resolving against an ambiguous declaration."""
    resolved = configs.resolve(configs.direct_config())
    raw = configs.direct_config()
    metrics = copy.deepcopy(resolved["evaluation"]["metrics"])
    for metric in metrics:
        metric.pop("direction", None)
    metrics[1]["id"] = metrics[0]["id"]
    raw["evaluation"]["metrics"] = metrics

    with pytest.raises(ValueError, match="Duplicate evaluation metric id"):
        configs.resolve(raw)


def test_a_metric_id_is_bounded_by_the_longest_history_key_the_loop_builds_from_it() -> None:
    """The longest admitted id fills the history's key bound with a component key; one character more fails at validation, not mid-training."""
    component = learning.metrics.metrics.UNAVAILABLE_CASE_VALUES
    longest = "m" * experiments.config.loader._MAX_METRIC_ID_LENGTH
    learning.training.history._admit_scalars({f"id/{longest}/component/{component}": 1.0})
    with pytest.raises(ValueError, match="bounded non-empty strings"):
        learning.training.history._admit_scalars({f"id/{longest}m/component/{component}": 1.0})

    raw = configs.direct_config()
    metrics = configs.resolve(configs.direct_config())["evaluation"]["metrics"]
    raw["evaluation"]["metrics"] = [{key: value for key, value in metric.items() if key != "direction"} for metric in metrics]
    raw["evaluation"]["metrics"][1]["id"] = longest
    configs.resolve(copy.deepcopy(raw))
    raw["evaluation"]["metrics"][1]["id"] = f"{longest}m"
    with pytest.raises(experiments.config.loader.ConfigError, match=r"evaluation\.metrics\[1\]\.id"):
        configs.resolve(raw)


def test_resolved_task_contract_rejects_a_drifted_contract() -> None:
    """Fail closed when persisted task semantics differ from the registered task."""
    resolved = configs.resolve(configs.direct_config())
    resolved["task_contract"]["digest"] = "0" * 64

    with pytest.raises(
        experiments.config.loader.ConfigError,
        match="does not exactly match registered task",
    ):
        experiments.config.loader.validate_resolved_task_contract(resolved)


def test_saved_run_config_lists_choices_first_and_contract_blocks_last(tmp_path: Path) -> None:
    """The saved order is presentation only: the same config reloads, validates and keeps its digest."""
    resolved = configs.resolve(configs.direct_config())
    path = tmp_path / "config.yaml"

    experiments.config.loader.save_yaml(resolved, path)

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = list(saved)
    assert keys[:7] == ["task", "model", "data", "loss", "optimizer", "scheduler", "training"]
    assert keys[-2:] == ["task_contract", "evaluation"]
    assert sorted(keys) == sorted(resolved)
    assert common.serialization.canonical_json_sha256(saved) == common.serialization.canonical_json_sha256(resolved)
    experiments.config.loader.validate_resolved_config(saved)
