"""Protect config sources, model labels, config-name validation, pair keys, trial leaves and run names.

Responsibilities:
  - Check the (task, family, name) source derived from canonical, Optuna and external config paths
  - Check model labels, the config-name rule, pair keys, trial leaves and run names, including the epoch-budget tokens of seed groups and studies
  - Check that the token tables of configs/README.md list exactly the code's name tokens
  - Check that revisions and the derived layout reach the resolved config as a pure function of its inputs
  - Check that the live layout identity is recomputed only when validation requests it

Design principles:
  - Names are pure functions of the resolved config, its source and its revision

This module does NOT:
  - Allocate runs, check occupancy or exercise the Optuna lifecycle
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
from support import configs

from src import common, learning
from src.experiments.config import experiments_config_defaults as config_defaults
from src.experiments.config import experiments_config_loader as loader
from src.experiments.config import experiments_config_naming as naming
from src.experiments.config import experiments_config_transient_plan as plan_service

_TREE = "configs/03_learning"
_CONFIGS_README = Path(__file__).resolve().parents[2] / "configs" / "README.md"


def _readme_rows(heading: str) -> list[list[str]]:
    """Return the cells of every table body row in the collapsible block of configs/README.md titled ``heading``."""
    section = _CONFIGS_README.read_text(encoding="utf-8").split(f"<summary><strong>{heading}</strong></summary>", 1)[1].split("</details>", 1)[0]
    return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in section.splitlines() if line.startswith("| `")]


def _fd_config() -> dict[str, Any]:
    """Return a resolved physics-enabled config using central finite differences."""
    return configs.resolve(configs.direct_config(physics_enabled=True))


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (
            f"{_TREE}/steady_flow/experiments/architecture_comparison/fno_m4x5_h8_l2.yaml",
            ("steady_flow", "architecture_comparison", "fno_m4x5_h8_l2"),
        ),
        (f"{_TREE}/transient_drying/optuna/uno_a0_search.yaml", ("transient_drying", "optuna", "uno_a0_search")),
    ],
)
def test_canonical_paths_name_task_family_and_stem(tmp_path: Path, relative: str, expected: tuple[str, str, str]) -> None:
    """The canonical tree alone decides task, family and name; the file need not exist."""
    assert naming.config_source_from_path(tmp_path / relative) == naming.ConfigSource(*expected)


def test_external_yaml_takes_its_task_from_the_yaml(tmp_path: Path) -> None:
    """A YAML outside the canonical tree is the explicit external family, labelled by its stem."""
    experiment = configs.write_yaml(tmp_path / "request.yaml", {"task": "steady_flow"})
    study = configs.write_yaml(tmp_path / "study.yaml", {"experiment": {"task": "transient_drying"}, "search_space": {}})
    taskless = configs.write_yaml(tmp_path / "taskless.yaml", {"model": {}})

    assert naming.config_source_from_path(experiment) == naming.ConfigSource("steady_flow", "external", "request")
    assert naming.config_source_from_path(study) == naming.ConfigSource("transient_drying", "external", "study")
    with pytest.raises(loader.ConfigError, match="must declare a task"):
        naming.config_source_from_path(taskless)
    with pytest.raises(FileNotFoundError):
        naming.config_source_from_path(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "relative",
    [
        f"{_TREE}/steady_flow/base.yaml",
        f"{_TREE}/steady_flow/experiments/plain.yaml",
        f"{_TREE}/steady_flow/experiments/family/nested/plain.yaml",
        f"{_TREE}/steady_flow/experiments/optuna/plain.yaml",
        f"{_TREE}/steady_flow/experiments/external/plain.yaml",
        f"{_TREE}/steady_flow/experiments/family/plain.yml",
        f"{_TREE}/steady_flow/optuna/nested/study.yaml",
        f"{_TREE}/steady_flow/optuna/study.json",
        f"{_TREE}/steady_flow/shared.yaml",
        f"{_TREE}/{_TREE}/steady_flow/experiments/family/plain.yaml",
    ],
)
def test_other_shapes_below_the_canonical_tree_are_rejected(tmp_path: Path, relative: str) -> None:
    """Base files, missing or nested families, reserved families and repeated trees are not executable configs."""
    with pytest.raises(loader.ConfigError, match=r"must be configs/03_learning|ambiguous"):
        naming.config_source_from_path(tmp_path / relative)


def test_only_a_study_file_may_live_in_the_optuna_directory(tmp_path: Path) -> None:
    """An experiment placed under optuna/ is refused instead of skipping the name check; a study keeps its exemption."""
    optuna_dir = tmp_path / _TREE / "steady_flow" / "optuna"
    study = configs.write_yaml(optuna_dir / "study.yaml", {"experiment": {"task": "steady_flow"}, "search_space": {}})
    experiment = configs.write_yaml(optuna_dir / "wrong_name.yaml", configs.direct_config())

    assert naming.config_source_from_path(study) == naming.ConfigSource("steady_flow", "optuna", "study")
    assert naming.config_source_from_path(study).authored is None
    with pytest.raises(loader.ConfigError, match=r"experiments/<family>/"):
        naming.config_source_from_path(experiment)
    unfiled_experiment = naming.ConfigSource("steady_flow", "optuna", "any", {"optimizer": {"lr": 0.001}})
    with pytest.raises(loader.ConfigError, match="must be 'fno_m4x5_h8_l2_p0p1__lr-0p001'"):
        naming.validate_config_name(unfiled_experiment, configs.resolve(configs.direct_config()))


def test_model_label_marks_physics_and_derivative_family() -> None:
    """Plain models use the architecture label; physics adds the pi- prefix and the fd or spec suffix."""
    plain = configs.resolve(configs.direct_config())
    assert naming.model_label(plain) == "fno_m4x5_h8_l2_p0p1"
    assert naming.model_family(plain) == "fno"

    finite_difference = _fd_config()
    assert naming.model_label(finite_difference) == "pi-fno_m4x5_h8_l2_p0p1__fd"
    assert naming.model_family(finite_difference) == "pi-fno"

    raw = configs.direct_config(model_kind="uno", physics_enabled=True)
    raw["loss"]["physics"]["derivatives"] = {"kind": "spectral"}
    spectral = configs.resolve(raw)
    # modes_x=4, modes_y=5: Y before X, as the FNO n_modes [4, 5] read in fno_m4x5.
    assert naming.model_label(spectral).startswith("pi-uno_m5x4_h8_l5_p0p1_s")
    assert naming.model_label(spectral).endswith("__spec")


def test_configs_readme_lists_exactly_the_name_tokens_in_name_order() -> None:
    """The lookup tables in configs/README.md name every token of the code, in name order, with the stages that have it."""
    stage_tokens = {token for token in naming._TOKENS if token.startswith("a0")}
    shared = [row[0].strip("`") for row in _readme_rows("Setting tokens")]
    assert shared == [token for token in naming._TOKENS if token not in stage_tokens]

    tokens = {**naming._TOKENS, **naming._PAIR_TOKENS}
    stage_rows = _readme_rows("Stage tokens of a transient matched plan")
    suffixes = [row[0].strip("`").removeprefix("<stage>") for row in stage_rows]
    assert suffixes == list(naming._STAGE_LEAVES)
    for suffix, row in zip(suffixes, stage_rows, strict=True):
        assert re.findall(r"`(a0|b|a1)`", row[3]) == [stage for stage in ("a0", "b", "a1") if f"{stage}{suffix}" in tokens], suffix

    assert _readme_rows("Pair folder")[0][0] == "`cev`"
    assert [token for token in naming._PAIR_TOKENS if not token.startswith(("b", "a1"))] == ["cev"]


def test_model_label_names_domain_padding_and_every_deviating_model_switch() -> None:
    """The label carries the padding (none is p0) and each non-default model parameter, in the display label too."""
    raw = configs.direct_config()
    raw["model"]["params"]["domain_padding"] = None
    raw["model"]["params"]["channel_mlp_skip"] = "linear"
    raw["model"]["params"]["lifting_channel_ratio"] = 4
    config = configs.resolve(raw)

    expected = "fno_m4x5_h8_l2_p0__channel-mlp-skip-linear__lifting-channel-ratio-4"
    assert naming.model_label(config) == expected
    assert learning.models.naming.architecture_label(config) == expected


def _name(raw: dict[str, Any], authored: dict[str, Any]) -> str:
    """Return the config name of a file that authors only ``authored``, for the complete request ``raw`` it resolves to."""
    return naming.config_name(loader.deep_merge(config_defaults.get_task_defaults(raw["task"]), raw), authored)


def test_steady_config_name_shows_every_authored_leaf_in_table_order() -> None:
    """Each leaf the file sets has its token; unnamed leaves add nothing; a multi-leaf token needs all of its leaves."""
    raw = configs.direct_config(physics_enabled=True)
    authored = {
        "run": {"seed": 3, "device": "cuda"},
        "tracking": {"wandb": {"mode": "online"}},
        "data": {"batch_size": 16, "num_workers": 4, "train_dataset": "x"},
        "scheduler": {"kind": "reduce_on_plateau", "factor": 0.5, "patience": 20, "min_lr": 0.001},
        "training": {"epochs": 300},
        "optimizer": {"lr": 0.000388361, "weight_decay": 0.0001, "betas": [0.9, 0.999]},
        "loss": {
            "data": {"kind": "relative_l2"},
            "physics": {
                "enabled": True,
                "derivatives": {"kind": "physical"},
                "interior_crop": 2,
                "momentum_weight": {"target": 1.0, "warmup": {"kind": "linear", "epochs": 100}},
                "continuity_weight": {"target": 1000.0, "warmup": {"epochs": 50}},
                "boundary_weight": {"target": 100.0, "warmup": {"epochs": 50}},
            },
        },
    }

    assert _name(raw, authored) == (
        "pi-fno_m4x5_h8_l2_p0p1__fd__crop-2__w-1-1000-100__wu-100-50-50__loss-relative-l2__bs-16"
        "__lr-0p000388__wd-0p0001__betas-0p9-0p999__schf-0p5__schp-20__schmin-0p001__ep-300"
    )
    assert _name(raw, {"scheduler": None}) == "pi-fno_m4x5_h8_l2_p0p1__fd__sch-none"
    all_equal = {"loss": {"physics": {term: {"target": 5.0} for term in ("momentum_weight", "continuity_weight", "boundary_weight")}}}
    assert _name(raw, all_equal) == "pi-fno_m4x5_h8_l2_p0p1__fd__w-5"
    assert _name(raw, {}) == "pi-fno_m4x5_h8_l2_p0p1__fd"


def test_name_values_round_but_a_scaling_keeps_its_decimal_marker() -> None:
    """A float shows three significant digits; a U-NO scaling of at least 1 keeps its decimal marker, so 1.5 and 15 differ."""
    raw = configs.direct_config()

    assert _name(raw, {"optimizer": {"lr": 0.0003881}}).endswith("__lr-0p000388")

    def label(scaling: float) -> str:
        params = {"modes_x": 4, "modes_y": 5, "hidden_channels": 8, "n_layers": 5, "domain_padding": 0.1, "mode_ratio": 0.4951}
        params["uno_scalings"] = [[1, 1], [scaling, scaling], [1, 1], [1, 1], [2, 2]]
        return learning.models.naming.architecture_label({"model": {"kind": "uno", "params": params}})

    assert "_s1-1p5-1-1-2_r0p495" in label(1.5)
    assert label(1.5) != label(15)
    assert "_s1-05-1-1-2_" in label(0.5)
    params = {"modes_x": 4, "modes_y": 5, "hidden_channels": 8, "n_layers": 5, "domain_padding": 0.1}
    params["uno_scalings"] = [[0.5, 1], [1, 1], [1, 1], [1, 1], [2, 2]]  # [y, x]: halve y only
    assert "_s05x1-1-1-1-2_" in learning.models.naming.architecture_label({"model": {"kind": "uno", "params": params}})


def test_a_partly_set_multi_leaf_token_is_refused_naming_the_missing_leaf() -> None:
    """A token that shows several leaves together never fills one from a default: the file sets all of them or none."""
    raw = configs.direct_config(physics_enabled=True)

    with pytest.raises(loader.ConfigError, match=r"part of name token 'w' but not 'loss\.physics\.continuity_weight\.target'"):
        _name(raw, {"loss": {"physics": {"momentum_weight": {"target": 1.0}}}})


def test_transient_plan_name_shows_shared_and_a0_leaves_but_leaves_b_and_a1_to_the_pair_key() -> None:
    """A matched plan's config name carries the shared and A0-prefixed leaves; its B/A1 leaves never enter it (they name the pair folder)."""
    raw = configs.transient_matched_config()
    b_a1 = {
        "continuation_evaluations": 20,
        "b": {"sampling": {"mode": "rollout_window", "window_stride": 16}, "train_seconds": 600.0, "lr_scale": 0.5},
        "a1": {"max_epochs": 150, "match_to": "b.train_seconds", "lr_scale": 0.5},
    }
    authored = {
        "loss": {"data": {"kind": "huber", "beta": 1.0}},
        "data": {"spatial_stride": 2},
        "optimizer": {"lr": 0.001},
        "training": {
            "mode": "matched_a1_b",
            "a0": {
                "max_epochs": 100,
                "sampling": {"mode": "trajectory", "rollout_length": 32},
                "curriculum": {"lengths": [2, 4, 8, 16, 32], "milestone_fractions": [0.0, 0.2, 0.4, 0.6, 0.8]},
            },
            **b_a1,
        },
    }
    expected = "fno_m4x5_h8_l2_p0p1__loss-huber__huber-1__stride-2__lr-0p001__a0ep-100__a0cur-2-4-8-16-32__a0smp-trajectory__a0rl-32"

    assert _name(raw, authored) == expected
    assert _name(raw, {key: value for key, value in authored.items() if key != "training"} | {"training": {"mode": "matched_a1_b", **b_a1}}) == (
        "fno_m4x5_h8_l2_p0p1__loss-huber__huber-1__stride-2__lr-0p001"
    )


def test_an_authored_leaf_no_token_shows_is_refused_by_its_key() -> None:
    """The name is complete: a leaf outside the token table and the exclusion list stops the config, including a stage marker key."""
    raw = configs.direct_config()

    with pytest.raises(loader.ConfigError, match=r"add a name token for optimizer\.mystery"):
        _name(raw, {"optimizer": {"lr": 0.001, "mystery": 1}})
    with pytest.raises(loader.ConfigError, match=r"add a name token for training\.matched_a1_b\.role"):
        _name(raw, {"training": {"matched_a1_b": {"role": "b"}}})
    with pytest.raises(loader.ConfigError, match=r"add a name token for evaluation\.metrics"):
        _name(raw, {"evaluation": {"metrics": []}})
    single_stage = configs.transient_stage_a_only_config()
    unsampled = {"temporal": {"temporal_conditioning": {"kind": "none"}}, "training": {"epochs": 4}}
    assert _name(single_stage, unsampled) == "fno_m4x5_h8_l2_p0p1__tcond-none__ep-4"
    with pytest.raises(loader.ConfigError, match=r"add a name token for temporal\.sampling\.mode"):
        _name(single_stage, {"temporal": {"sampling": {"mode": "trajectory"}}})


def _maintained_style_tree(root: Path, raw: dict[str, Any]) -> Path:
    """Write a test task base with all shared sections and return the path of the experiment file that sets only model and lr."""
    shared = {key: value for key, value in raw.items() if key not in {"task", "model"}}
    shared["optimizer"] = {key: value for key, value in raw["optimizer"].items() if key != "lr"}
    configs.write_yaml(root / _TREE / raw["task"] / "base.yaml", shared)
    authored = {"model": raw["model"], "optimizer": {"lr": raw["optimizer"]["lr"]}, "task": raw["task"], "base": "../../base.yaml"}
    name = _name(raw, {key: value for key, value in authored.items() if key not in {"task", "base"}})
    return configs.write_yaml(root / _TREE / raw["task"] / "experiments" / "group" / f"{name}.yaml", authored)


def test_a_maintained_config_file_must_carry_the_name_of_the_leaves_it_sets(tmp_path: Path) -> None:
    """The loader derives the expected name from the file alone: a changed base never renames it, a wrong stem is refused by its expected name."""
    raw = configs.direct_config()
    path = _maintained_style_tree(tmp_path, raw)
    assert path.stem == "fno_m4x5_h8_l2_p0p1__lr-0p001"

    assert loader.load_and_resolve_config(path)["run"]["layout"]["config_name"] == path.stem
    base_path = tmp_path / _TREE / "steady_flow" / "base.yaml"
    base = loader.load_yaml(base_path)
    base["data"]["batch_size"] += 1
    base["optimizer"]["weight_decay"] += 0.5
    base["training"]["epochs"] += 1
    configs.write_yaml(base_path, base)
    assert loader.load_and_resolve_config(path)["run"]["layout"]["config_name"] == path.stem

    for wrong in ("fno_m4x5_h8_l2_p0p1", "fno_m4x5_h8_l2_p0p1__lr-0p001__extra", "fno_m4x5_h8_l3_p0p1__lr-0p001", "request"):
        renamed = path.with_name(f"{wrong}.yaml")
        renamed.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(loader.ConfigError, match=r"must be 'fno_m4x5_h8_l2_p0p1__lr-0p001'"):
            loader.load_and_resolve_config(renamed)


def test_studies_and_unfiled_sources_are_not_name_checked() -> None:
    """Only a file's own leaves can be named: studies and in-memory sources keep any name; every other family is checked."""
    raw = configs.direct_config()
    config = configs.resolve(raw)
    authored = {"optimizer": {"lr": 0.001}}

    for source in (naming.ConfigSource("steady_flow", "optuna", "any"), naming.ConfigSource("steady_flow", "tests", "any")):
        naming.validate_config_name(source, config)
    with pytest.raises(loader.ConfigError, match=r"must be 'fno_m4x5_h8_l2_p0p1__lr-0p001'"):
        naming.validate_config_name(naming.ConfigSource("steady_flow", "tests", "any", authored), config)


def test_an_external_yaml_with_a_wrong_name_is_refused_even_by_a_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A YAML outside the canonical tree needs its computed name; a relative path to a maintained file is still recognised as one."""
    path = configs.write_yaml(tmp_path / "request.yaml", configs.direct_config())

    with pytest.raises(loader.ConfigError, match=r"must be 'fno_m4x5_h8_l2_p0p1__"):
        loader.load_and_resolve_config(path)
    maintained = configs.write_config(tmp_path, configs.direct_config(), canonical_root=tmp_path)
    monkeypatch.chdir(tmp_path)
    assert naming.config_source_from_path(maintained.relative_to(tmp_path)).family == configs.TEST_FAMILY


def test_worst_case_names_stay_within_one_path_component() -> None:
    """A long U-NO that sets the usual hyperparameters, or a plan that sets every stage leaf, keeps its names and run names within one component."""
    uno = configs.direct_config(model_kind="uno", physics_enabled=True)
    uno["model"]["params"].update({"modes_x": 128, "modes_y": 160, "hidden_channels": 128, "n_layers": 7, "mode_ratio": 0.4951318201313778})
    uno["model"]["params"]["uno_scalings"] = [[1.0, 1.0], [0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
    steady = _name(
        uno,
        {
            "loss": {
                "data": {"kind": "relative_l2"},
                "physics": {
                    "interior_crop": 12,
                    "momentum_weight": {"target": 0.00123, "warmup": {"epochs": 100}},
                    "continuity_weight": {"target": 4560.0, "warmup": {"epochs": 200}},
                    "boundary_weight": {"target": 0.789, "warmup": {"epochs": 300}},
                },
            },
            "data": {"batch_size": 16},
            "optimizer": {"lr": 0.000532, "weight_decay": 0.0001},
            "training": {"epochs": 300},
        },
    )
    assert steady.endswith("__crop-12__w-0p00123-4560-0p789__wu-100-200-300__loss-relative-l2__bs-16__lr-0p000532__wd-0p0001__ep-300")

    plan = configs.transient_matched_config(model_kind="uno")
    plan["model"] = uno["model"]
    transient = _name(plan, {"optimizer": {"lr": 0.000532}, "training": {"a0": {"sampling": {"mode": "one_step_transition"}}}})
    assert transient.endswith("__lr-0p000532__a0smp-one-step-transition")
    b, a1 = _plan_children(b_scale=0.5, a1_scale=0.25, seconds=3600.0)
    b_sampling = {"mode": "rollout_window", "window_stride": 32, "window_offset": 0}
    authored = {"training": {"continuation_evaluations": 20, "b": {"max_epochs": 100, "sampling": b_sampling}, "a1": {"max_epochs": 100}}}
    pair = naming.pair_key(b, a1, authored)
    assert pair.startswith("b-a1__budget-3600s__rollout-4__lr-b-0p5__lr-a1-0p25__cev-20__bep-100__bsmp-rollout-window__bws-32__bwo-0__a1ep-100")

    steady_suffix = "__seed-29__rev-wave-2026-11-09"
    transient_suffix = f"__seed-29__a0ep-1000__rev-x__{pair}__a1"
    for logical in (steady, steady + steady_suffix, transient, transient + transient_suffix):
        common.paths.validate_logical_name(logical, label="worst-case name")

    # The limit case: a B curriculum of dozens of unequally spaced entries outgrows one path component of the pair folder.
    lengths = list(range(1, 41))
    long_curriculum = {"lengths": lengths, "milestone_fractions": [index / 41 for index in range(40)]}
    with pytest.raises(loader.ConfigError, match=r"pair key must not exceed 255 UTF-8 bytes.*got \d+ bytes"):
        naming.pair_key(b, a1, {"training": {"b": {"curriculum": long_curriculum}}})


def test_a_name_beyond_the_component_limit_is_a_resolve_time_error_with_its_byte_count() -> None:
    """The limit case (a very long revision label on a plausible name) stays an error at load, naming the byte count."""
    raw = configs.direct_config()
    requested = loader.apply_training_overrides(raw, revision="r" * 250)

    with pytest.raises(loader.ConfigError, match=r"must not exceed 255 UTF-8 bytes.*got \d+ bytes"):
        loader.resolve_config(requested, source=configs.source_for(raw))


def test_a_final_b_run_name_beyond_the_component_limit_is_refused_with_its_byte_count() -> None:
    """The pair key and stage join B's name after its first resolution: a name that outgrows the component there stays a config error."""
    raw = configs.transient_matched_config()
    fits = loader.apply_training_overrides(raw, revision="r" * 160)
    assert max(len(stage["run"]["name"]) for stage in configs.resolve_plan(fits).stages) <= 255  # the component limit
    longer = loader.apply_training_overrides(raw, revision="r" * 170)  # A0 still fits; B and A1 outgrow the limit with pair key and stage

    with pytest.raises(loader.ConfigError, match=r"run\.name must not exceed 255 UTF-8 bytes.*got \d+ bytes"):
        configs.resolve_plan(longer)


def _plan_children(*, b_scale: float = 1.0, a1_scale: float = 1.0, seconds: float = 0.25) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the resolved B and A1 children of a matched plan with explicit pair settings."""
    raw = configs.transient_matched_config()
    raw["training"]["b"].update({"lr_scale": b_scale, "train_seconds": seconds})
    raw["training"]["a1"]["lr_scale"] = a1_scale
    plan = configs.resolve_plan(raw)
    return dict(plan.b), dict(plan.a1)


def test_pair_key_names_budget_rollout_and_one_shared_learning_rate_scale() -> None:
    """Equal B and A1 scales share one lr token; the rollout is B's window length."""
    b, a1 = _plan_children(b_scale=0.5, a1_scale=0.5, seconds=600.0)

    assert naming.pair_key(b, a1, None) == "b-a1__budget-600s__rollout-4__lr-0p5"
    assert b["run"]["layout"]["pair"] == a1["run"]["layout"]["pair"] == naming.pair_key(b, a1, None)


def test_pair_key_separates_unequal_learning_rate_scales() -> None:
    """Different B and A1 scales are named separately; a fractional budget keeps its decimal."""
    b, a1 = _plan_children(b_scale=1.0, a1_scale=0.25, seconds=0.25)

    assert naming.pair_key(b, a1, None) == "b-a1__budget-0p25s__rollout-4__lr-b-1__lr-a1-0p25"


def test_pair_key_requires_the_b_child_first() -> None:
    """The key reads B's budget and window, so swapped children are refused."""
    b, a1 = _plan_children()

    with pytest.raises(ValueError, match="B child first"):
        naming.pair_key(a1, b, None)


def test_pair_key_adds_one_token_per_authored_b_a1_leaf_the_fixed_part_does_not_show() -> None:
    """The fixed part shows budget, B's rollout and both scales; every other authored B/A1 leaf follows in table order, a partial one is refused."""
    b, a1 = _plan_children(b_scale=0.5, a1_scale=0.5, seconds=600.0)
    authored = {
        "optimizer": {"lr": 0.001},
        "training": {
            "continuation_evaluations": 20,
            "b": {
                "sampling": {"mode": "rollout_window", "rollout_length": 4, "window_stride": 16},
                "curriculum": {"lengths": [3, 6], "milestone_fractions": [0.0, 0.25]},
                "train_seconds": 600.0,
                "lr_scale": 0.5,
            },
            "a1": {"max_epochs": 150, "match_to": "b.train_seconds", "lr_scale": 0.5},
        },
    }

    assert naming.pair_key(b, a1, authored) == (
        "b-a1__budget-600s__rollout-4__lr-0p5__cev-20__bcur-3-6-at-0-0p25__bsmp-rollout-window__bws-16__a1ep-150"
    )
    with pytest.raises(loader.ConfigError, match=r"part of name token 'bcur' but not 'training\.b\.curriculum\.milestone_fractions'"):
        naming.pair_key(b, a1, {"training": {"b": {"curriculum": {"lengths": [3, 6]}}}})


def test_editing_a_b_leaf_of_a_plan_file_keeps_the_config_name_and_the_a0_folder(tmp_path: Path) -> None:
    """The config folder holds A0: a B setting of the plan file changes only the pair folder, so a later B edit still finds its A0."""
    raw = configs.transient_matched_config()
    name = configs.source_for(raw).name
    shared = {key: value for key, value in raw.items() if key not in {"task", "model"}}
    shared["optimizer"] = {key: value for key, value in raw["optimizer"].items() if key != "lr"}
    configs.write_yaml(tmp_path / _TREE / raw["task"] / "base.yaml", shared)
    path = tmp_path / _TREE / raw["task"] / "experiments" / "group" / f"{name}.yaml"

    def resolve(b_leaves: dict[str, Any]) -> Any:
        authored = {"task": raw["task"], "base": "../../base.yaml", "model": raw["model"], "optimizer": {"lr": raw["optimizer"]["lr"]}}
        configs.write_yaml(path, {**authored, "training": {"b": b_leaves}})
        return plan_service.resolve_matched_a1_b_plan(loader.load_authored_config(path), source=naming.config_source_from_path(path))

    first = resolve({"train_seconds": 0.25})
    second = resolve({"train_seconds": 0.5, "max_epochs": 5})

    assert first.a0["run"]["layout"]["config_name"] == second.a0["run"]["layout"]["config_name"] == name
    assert common.paths.resolve_config_run_dir(first.a0) == common.paths.resolve_config_run_dir(second.a0)
    assert first.b["run"]["layout"]["pair"] == "b-a1__budget-0p25s__rollout-4__lr-1"
    assert second.b["run"]["layout"]["pair"] == "b-a1__budget-0p5s__rollout-4__lr-1__bep-5"


def test_trial_leaf_formats_number_label_and_sorted_values() -> None:
    """Zero-padded number, model label, then one name-value token per sampled parameter sorted by name."""
    config = configs.resolve(configs.direct_config())
    values = {"lr": 0.00038836, "batch_size": 16, "warmup": True, "momentum_weight": 123456.7, "scheduler": "Plateau X", "budget": 3655.0}

    assert naming.trial_leaf(7, config, {}) == "0007__fno_m4x5_h8_l2_p0p1"
    assert naming.trial_leaf(7, config, values) == (
        "0007__fno_m4x5_h8_l2_p0p1__batch_size-16__budget-3655__lr-0p000388__momentum_weight-123000__scheduler-plateau-x__warmup-yes"
    )
    assert naming.trial_leaf(12345, config, {}).startswith("12345__")
    with pytest.raises(ValueError, match="non-negative integer"):
        naming.trial_leaf(-1, config, {})
    with pytest.raises(ValueError, match="finite"):
        naming.trial_leaf(1, config, {"lr": float("nan")})


def test_run_name_follows_steady_stage_pair_revision_and_trial() -> None:
    """Steady runs omit the stage; staged runs append pair and stage; trials use study and trial number."""
    base = {"config_name": "fno_m4x5_h8_l2", "seed": 9, "revision": None, "budget_epochs": None, "stage": "single", "pair": None, "trial": None}
    pair = "b-a1__budget-600s__rollout-4__lr-1"

    assert naming.run_name(base) == "fno_m4x5_h8_l2__seed-9"
    assert naming.run_name({**base, "revision": "fixed"}) == "fno_m4x5_h8_l2__seed-9__rev-fixed"
    assert naming.run_name({**base, "stage": "a0"}) == "fno_m4x5_h8_l2__seed-9__a0"
    assert naming.run_name({**base, "stage": "a1", "pair": pair, "revision": "fixed"}) == f"fno_m4x5_h8_l2__seed-9__rev-fixed__{pair}__a1"
    assert naming.run_name({**base, "config_name": "fno_search", "trial": "0003__fno_m4x5_h8_l2"}) == "fno_search__trial-0003"
    assert naming.parent_label({**base, "family": "tests", "revision": "fixed"}) == "tests__fno_m4x5_h8_l2__seed-9__rev-fixed"


def test_budget_epochs_name_the_transient_seed_group_and_the_study_before_the_revision() -> None:
    """A transient seed group carries its A0 epoch budget and a study its trial budget; a revision label stays last."""
    base = {"config_name": "fno_m4x5_h8_l2", "seed": 9, "revision": None, "budget_epochs": 40, "stage": "a0", "pair": None, "trial": None}
    pair = "b-a1__budget-600s__rollout-4__lr-1"
    study = {**base, "family": "optuna", "config_name": "fno_search", "stage": "single", "trial": "0003__fno_m4x5_h8_l2"}

    assert naming.run_name(base) == "fno_m4x5_h8_l2__seed-9__a0ep-40__a0"
    assert naming.run_name({**base, "stage": "a1", "pair": pair, "revision": "fixed"}) == f"fno_m4x5_h8_l2__seed-9__a0ep-40__rev-fixed__{pair}__a1"
    assert naming.parent_label({**base, "family": "tests", "revision": "fixed"}) == "tests__fno_m4x5_h8_l2__seed-9__a0ep-40__rev-fixed"
    assert naming.run_name(study) == "fno_search__trial-0003"
    assert naming.parent_label({**study, "revision": "fixed"}) == "optuna__fno_search__ep-40__rev-fixed__seed-9"
    assert common.paths.seed_group_name(9) == "seed-9"
    assert common.paths.study_group_name("fno_search", 40) == "fno_search__ep-40"
    for invalid in (0, -1, 2.0, True, None):
        with pytest.raises(ValueError, match="positive integer"):
            common.paths.study_group_name("fno_search", invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        common.paths.seed_group_name(9, a0_epochs=0)


def test_resolved_layout_is_a_pure_function_of_config_source_and_revision(tmp_path: Path) -> None:
    """The same inputs give the same layout; only the revision and the source change it."""
    raw = configs.direct_config()
    source = configs.source_for(raw)
    first = loader.resolve_config(copy.deepcopy(raw), source=source, output_root=tmp_path)
    second = loader.resolve_config(copy.deepcopy(raw), source=source, output_root=tmp_path)
    revised = loader.resolve_config(loader.apply_training_overrides(raw, revision="fixed-loss"), source=source, output_root=tmp_path)

    assert first["run"] == second["run"]
    assert first["run"]["layout"] == {
        "task": "steady_flow",
        "dataset": "synthetic_train",
        "family": "tests",
        "config_name": source.name,
        "model_label": "fno_m4x5_h8_l2_p0p1",
        "seed": 17,
        "revision": None,
        "budget_epochs": None,
        "stage": "single",
        "pair": None,
        "trial": None,
        "a0_id": first["run"]["layout"]["a0_id"],
        "branch_id": None,
    }
    assert first["run"]["name"] == f"{source.name}__seed-17"
    assert revised["run"]["revision"] == revised["run"]["layout"]["revision"] == "fixed-loss"
    assert revised["run"]["name"] == f"{source.name}__seed-17__rev-fixed-loss"
    assert revised["run"]["layout"]["a0_id"] == first["run"]["layout"]["a0_id"]
    assert common.paths.resolve_config_run_dir(revised) == common.paths.resolve_config_run_dir(first).with_name("seed-17__rev-fixed-loss")


def test_resolver_rejects_a_mismatching_source_task_and_config_name() -> None:
    """The source task must equal the config task and a source with its file's leaves must carry the name they derive."""
    raw = configs.direct_config()
    label = configs.source_for(raw).name
    authored = {"model": raw["model"], "optimizer": {"lr": raw["optimizer"]["lr"]}}

    with pytest.raises(loader.ConfigError, match="Task config path mismatch"):
        loader.resolve_config(copy.deepcopy(raw), source=naming.ConfigSource("transient_drying", "tests", label))
    loader.resolve_config(copy.deepcopy(raw), source=naming.ConfigSource("steady_flow", "tests", label, authored))
    with pytest.raises(loader.ConfigError, match=rf"must be {label!r}"):
        loader.resolve_config(copy.deepcopy(raw), source=naming.ConfigSource("steady_flow", "tests", "other", authored))


def test_saved_layout_must_match_its_config_for_execution() -> None:
    """Execution rejects a layout whose dataset, revision or model label contradicts the saved config."""
    config = configs.resolve(configs.direct_config())
    loader.validate_resolved_config(config)

    for key, value, message in (
        ("dataset", "other_dataset", "seed, revision and epoch budget"),
        ("revision", "x", "seed, revision and epoch budget"),
        ("budget_epochs", 5, "seed, revision and epoch budget"),
        ("model_label", "fno_m9x9_h8_l2", "Model label contradicts"),
    ):
        tampered = copy.deepcopy(config)
        tampered["run"]["layout"][key] = value
        with pytest.raises(loader.ConfigError, match=message):
            loader.validate_resolved_config(tampered)


def test_layout_identity_is_recomputed_only_when_verification_is_requested() -> None:
    """An initial-stage identity that the running implementation does not reproduce fails execution but not structural validation."""
    config = configs.resolve(configs.direct_config())
    drifted = copy.deepcopy(config)
    drifted["run"]["layout"]["a0_id"] = "f" * 16

    with pytest.raises(loader.ConfigError, match="Initial run layout identity"):
        loader.validate_resolved_config(drifted)
    structural = loader.validate_resolved_config(drifted, verify_layout_identity=False)
    assert structural["run"]["layout"]["a0_id"] == "f" * 16
    drifted["run"]["layout"]["dataset"] = "other_dataset"
    with pytest.raises(loader.ConfigError, match="seed, revision and epoch budget"):
        loader.validate_resolved_config(drifted, verify_layout_identity=False)
