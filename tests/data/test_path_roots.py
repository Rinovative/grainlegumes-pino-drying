"""Protect portable roots and the current Dataset and Experiment path contracts.

Responsibilities:
  - Check root precedence, task-qualified Dataset placement with fixed payload names, and layout-derived run and study placement
  - Check that source-tree mode demands explicit roots and never falls back to sibling defaults

Design principles:
  - Dataset identity is independent of its selected root
  - A frozen source tree never infers storage or runtime from its own location

This module does NOT:
  - Create scientific packages, runs, or runtime environments
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import common

_PAIR = "b-a1__budget-600s__rollout-8__lr-1"
_BUDGET = 40


def _config(
    output_root: Path,
    *,
    stage: str = "a0",
    branch_id: str | None = None,
    pair: str | None = None,
    **layout_update: Any,
) -> dict[str, Any]:
    layout = {
        "task": "transient_drying",
        "dataset": "lentil-train1",
        "family": "architecture_comparison",
        "config_name": "fno_m12x12_h8_l2",
        "model_label": "fno_m12x12_h8_l2",
        "seed": 9,
        "revision": None,
        "budget_epochs": _BUDGET,
        "stage": stage,
        "pair": pair,
        "trial": None,
        "a0_id": "a" * 16,
        "branch_id": branch_id,
    }
    return {
        "run": {"layout": {**layout, **layout_update}},
        "paths": {"output_root": str(output_root)},
    }


def test_runtime_root_uses_portable_default_and_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replaceable runtime defaults beside the selected project root."""
    project_root = tmp_path / "project" / "repo"
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("RUNTIME_ROOT", raising=False)

    assert common.paths.get_runtime_root() == tmp_path / "project" / "runtime"
    assert common.paths.get_source_trees_root() == tmp_path / "project/runtime/sources"

    configured_root = tmp_path / "configured runtime"
    monkeypatch.setenv("RUNTIME_ROOT", str(configured_root))
    assert common.paths.get_runtime_root() == configured_root
    explicit_root = tmp_path / "explicit runtime"
    assert common.paths.get_runtime_root(runtime_root=explicit_root) == explicit_root


_TREE_ID = "a" * 12 + "-" + "b" * 12


def _tree_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("SOURCE_TREE_ID", _TREE_ID)
    monkeypatch.setenv("PROJECT_ROOT", str(runtime / "sources" / _TREE_ID))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("RUNTIME_ROOT", str(runtime))
    return runtime


def test_source_tree_mode_resolves_only_explicit_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tree reads its project, storage and runtime roots from the environment and derives snapshot and tree locations from them."""
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    assert common.paths.source_tree_id() is None
    runtime = _tree_environment(monkeypatch, tmp_path)

    assert common.paths.source_tree_id() == _TREE_ID
    assert common.paths.get_project_root() == runtime / "sources" / _TREE_ID
    assert common.paths.get_storage_root() == tmp_path / "storage"
    assert common.paths.get_runtime_root() == runtime
    assert common.paths.get_source_snapshots_root() == tmp_path / "storage/source_snapshots"
    assert common.paths.get_source_trees_root() == runtime / "sources"
    assert common.paths.get_storage_root(storage_root=tmp_path / "explicit") == tmp_path / "explicit"
    assert common.paths.source_tree_id({"SOURCE_TREE_ID": "c" * 12 + "-" + "d" * 12}) == "c" * 12 + "-" + "d" * 12


@pytest.mark.parametrize("missing", ["PROJECT_ROOT", "STORAGE_ROOT", "RUNTIME_ROOT"])
def test_source_tree_mode_refuses_sibling_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Every root getter raises with the missing variable named instead of inferring it beside the tree."""
    _tree_environment(monkeypatch, tmp_path)
    monkeypatch.delenv(missing)

    for getter in (common.paths.get_project_root, common.paths.get_storage_root, common.paths.get_runtime_root):
        with pytest.raises(RuntimeError, match=f"{missing} must be set explicitly"):
            getter()


def test_source_tree_mode_requires_project_root_to_be_the_named_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PROJECT_ROOT must equal RUNTIME_ROOT/sources/<SOURCE_TREE_ID>, and the ID must be a tree name."""
    _tree_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path / "repo"))
    with pytest.raises(RuntimeError, match="PROJECT_ROOT must equal RUNTIME_ROOT/sources/"):
        common.paths.get_project_root()
    with pytest.raises(RuntimeError, match="PROJECT_ROOT must equal RUNTIME_ROOT/sources/"):
        common.paths.get_storage_root()
    monkeypatch.setenv("SOURCE_TREE_ID", "not-a-tree")
    with pytest.raises(ValueError, match="SOURCE_TREE_ID must be"):
        common.paths.get_project_root()


def test_dataset_directory_is_task_qualified_and_root_independent(tmp_path: Path) -> None:
    """A Dataset is one directory below its task, below the selected Dataset root."""
    first_root = common.paths.get_datasets_root(storage_root=tmp_path / "first")
    second_root = common.paths.get_datasets_root(storage_root=tmp_path / "second")
    dataset_id = "lentil-train1-val1-test1__ref-range__lhs-seed-9300"

    first_path = common.paths.resolve_dataset_dir("steady_flow", dataset_id, dataset_root=first_root)
    second_path = common.paths.resolve_dataset_dir("steady_flow", dataset_id, dataset_root=second_root)

    assert first_root == tmp_path / "first/02_datasets"
    assert second_root == tmp_path / "second/02_datasets"
    assert first_path == first_root / "steady_flow" / dataset_id
    assert second_path == second_root / "steady_flow" / dataset_id
    assert common.paths.resolve_dataset_dir("transient_drying", dataset_id, dataset_root=first_root) != first_path
    assert not first_path.exists()


def test_dataset_payload_and_manifest_names_are_fixed_per_task(tmp_path: Path) -> None:
    """The payload file is dataset.pt for steady flow and the dataset.json index for transient drying, whatever the Dataset name."""
    dataset_id = "lentil-train1-val1-test1__ref-range__lhs-seed-9300"
    steady = common.paths.resolve_dataset_dir("steady_flow", dataset_id, dataset_root=tmp_path)
    transient = common.paths.resolve_dataset_dir("transient_drying", dataset_id, dataset_root=tmp_path)

    assert common.paths.resolve_dataset_path("steady_flow", dataset_id, dataset_root=tmp_path) == steady / "dataset.pt"
    assert common.paths.resolve_dataset_path("transient_drying", dataset_id, dataset_root=tmp_path) == transient / "dataset.json"
    assert common.paths.resolve_dataset_manifest_path("steady_flow", dataset_id, dataset_root=tmp_path) == steady / "dataset_manifest.json"
    assert common.paths.resolve_dataset_metadata_dir("steady_flow", dataset_id, metadata_root=tmp_path) == steady
    with pytest.raises(ValueError, match="Datasets exist for the tasks"):
        common.paths.resolve_dataset_path("analysis", dataset_id, dataset_root=tmp_path)


def test_dataset_build_locks_are_qualified_by_task(tmp_path: Path) -> None:
    """The same Dataset name under two tasks never shares a build lock."""
    steady_lock = common.paths.resolve_dataset_build_lock_path("steady_flow", "name", storage_root=tmp_path)
    transient_lock = common.paths.resolve_dataset_build_lock_path("transient_drying", "name", storage_root=tmp_path)

    assert steady_lock.name == f"dataset-{common.paths.dataset_state_key('steady_flow', 'name')}.lock"
    assert transient_lock != steady_lock
    assert steady_lock.is_relative_to(common.paths.get_dataset_state_root(storage_root=tmp_path))


def test_storage_selection_controls_the_dataset_lifecycle_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit and environment storage roots precede the portable sibling default."""
    project_root = tmp_path / "project" / "repo"
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("STORAGE_ROOT", raising=False)
    assert common.paths.get_datasets_root() == tmp_path / "project/storage/02_datasets"

    configured_root = tmp_path / "configured storage"
    monkeypatch.setenv("STORAGE_ROOT", str(configured_root))
    assert common.paths.get_datasets_root() == configured_root / "02_datasets"
    explicit_root = tmp_path / "explicit storage"
    assert common.paths.get_datasets_root(storage_root=explicit_root) == explicit_root / "02_datasets"


def test_run_layout_shares_a0_group_and_separates_pairs(tmp_path: Path) -> None:
    """A0 is shared by the seed group, named by its epoch budget, while every B/A1 pair owns its directory."""
    output_root = tmp_path / "03_experiments"
    branch = "b" * 16
    group = output_root / f"transient_drying/lentil-train1/architecture_comparison/fno_m12x12_h8_l2/seed-9__a0ep-{_BUDGET}"
    other_pair = "b-a1__budget-300s__rollout-8__lr-1"

    def run_dir(stage: str, pair: str | None = None) -> Path:
        return common.paths.resolve_config_run_dir(_config(output_root, stage=stage, pair=pair, branch_id=None if pair is None else branch))

    assert common.paths.resolve_config_experiment_dir(_config(output_root)) == group
    assert run_dir("a0") == group / "a0"
    assert run_dir("b", _PAIR) == group / _PAIR / "b"
    assert run_dir("a1", _PAIR) == group / _PAIR / "a1"
    assert run_dir("b", other_pair) == group / other_pair / "b"
    assert common.paths.resolve_config_branch_dir(_config(output_root, stage="b", pair=_PAIR, branch_id=branch)) == group / _PAIR
    assert not group.exists()


def test_single_run_revision_and_explicit_output_root_override(tmp_path: Path) -> None:
    """A single-stage run is its seed directory, a revision suffixes it last, and an explicit root takes precedence."""
    configured_root = tmp_path / "configured"
    explicit_root = tmp_path / "explicit"
    config = _config(configured_root, stage="single")
    seed_dir = f"transient_drying/lentil-train1/architecture_comparison/fno_m12x12_h8_l2/seed-9__a0ep-{_BUDGET}"

    assert common.paths.resolve_config_run_dir(config) == configured_root / seed_dir
    assert common.paths.resolve_config_run_dir(config, output_root=explicit_root) == explicit_root / seed_dir
    revised = _config(configured_root, stage="single", revision="fixed-loss")
    assert common.paths.resolve_config_run_dir(revised) == configured_root / f"{seed_dir}__rev-fixed-loss"


def test_steady_run_directory_carries_no_epoch_budget(tmp_path: Path) -> None:
    """A run without an identity-bearing budget keeps the plain seed directory because its epochs extend it in place."""
    steady = _config(tmp_path, stage="single", budget_epochs=None)

    assert common.paths.resolve_config_run_dir(steady).name == "seed-9"
    assert common.paths.resolve_config_run_dir(_config(tmp_path, stage="single", budget_epochs=None, revision="x")).name == "seed-9__rev-x"


def test_trial_layout_and_study_directory_follow_the_dataset_and_revision(tmp_path: Path) -> None:
    """Trials live in their study directory, named by the study's epoch budget and followed by its revision suffix."""
    trial = "0007__fno_m12x12_h8_l2__lr-0p001"
    config = _config(tmp_path, stage="single", family="optuna", config_name="fno_search", trial=trial, revision="long")
    study = tmp_path / f"transient_drying/lentil-train1/optuna/fno_search__ep-{_BUDGET}__rev-long"

    assert common.paths.resolve_study_dir("transient_drying", "lentil-train1", "fno_search", _BUDGET, "long", output_root=tmp_path) == study
    unrevised = common.paths.resolve_study_dir("transient_drying", "lentil-train1", "fno_search", _BUDGET, output_root=tmp_path)
    assert unrevised == study.with_name(f"fno_search__ep-{_BUDGET}")
    assert common.paths.resolve_study_dir("transient_drying", "lentil-train1", "fno_search", _BUDGET + 1, output_root=tmp_path) != unrevised
    assert common.paths.resolve_config_run_dir(config) == study / "trials" / trial
    assert common.paths.resolve_config_output_root(config, study / "trials" / trial) == tmp_path


_INVALID_LOGICAL_NAMES = (
    "",
    ".",
    "..",
    "../escape",
    "nested/name",
    "nested\\name",
    "/outside/escape",
    " trailing",
)


def test_dataset_and_generation_resolvers_reject_unsafe_names(tmp_path: Path) -> None:
    """Logical identifiers cannot escape their Dataset or campaign roots."""
    for invalid_name in _INVALID_LOGICAL_NAMES:
        with pytest.raises(ValueError, match="single non-empty path component"):
            common.paths.resolve_dataset_dir("steady_flow", invalid_name, dataset_root=tmp_path)
        with pytest.raises(ValueError, match="single non-empty path component"):
            common.paths.resolve_dataset_dir(invalid_name, "name", dataset_root=tmp_path)
        with pytest.raises(ValueError, match="single non-empty path component"):
            common.paths.resolve_generation_campaign_directory(
                "steady_flow",
                invalid_name,
                storage_root=tmp_path,
            )


@pytest.mark.parametrize(
    ("layout_update", "message"),
    [
        ({"family": "../escape"}, "single non-empty path component"),
        ({"config_name": "nested/model"}, "single non-empty path component"),
        ({"dataset": "x" * 256}, "255 UTF-8 bytes"),
        ({"a0_id": "A" * 16}, "lowercase 16-character content identity"),
        ({"stage": "b", "pair": _PAIR}, "Only B and A1 carry a pair and a branch identity"),
        ({"stage": "b", "branch_id": "b" * 16}, "Only B and A1 carry a pair and a branch identity"),
        ({"pair": _PAIR}, "Only B and A1 carry a pair and a branch identity"),
        ({"branch_id": "b" * 16}, "Only B and A1 carry a pair and a branch identity"),
        ({"trial": "0001__fno"}, "Only single-stage runs of family 'optuna'"),
        ({"family": "optuna"}, "Only single-stage runs of family 'optuna'"),
        ({"family": "optuna", "trial": "0001__fno"}, "Only single-stage runs of family 'optuna'"),
        ({"family": "optuna", "trial": "0001__fno", "stage": "single", "budget_epochs": None}, "requires run.layout.budget_epochs"),
        ({"budget_epochs": 0}, "positive integer"),
        ({"budget_epochs": "40"}, "positive integer"),
    ],
)
def test_run_layout_rejects_invalid_identity_and_stage_combinations(
    tmp_path: Path,
    layout_update: dict[str, Any],
    message: str,
) -> None:
    """Run path derivation fails closed for unsafe or incoherent layout identity."""
    config = _config(tmp_path)
    config["run"]["layout"].update(layout_update)

    with pytest.raises(ValueError, match=message):
        common.paths.resolve_config_run_dir(config)


def test_run_layout_requires_the_exact_contract_keys(tmp_path: Path) -> None:
    """Extra or missing layout fields cannot silently alter saved-run identity."""
    config = _config(tmp_path)
    config["run"]["layout"].pop("branch_id")

    with pytest.raises(ValueError, match=r"run\.layout must contain"):
        common.paths.resolve_config_run_dir(config)

    config = _config(tmp_path)
    config["run"]["layout"]["dataset_id"] = "unused"
    with pytest.raises(ValueError, match=r"run\.layout must contain"):
        common.paths.resolve_config_run_dir(config)


def test_logical_names_are_bounded_in_utf8_bytes() -> None:
    """A path component may use 255 bytes but not one more, counted in UTF-8 bytes."""
    assert common.paths.validate_logical_name("a" * 255, label="name") == "a" * 255
    with pytest.raises(ValueError, match="255 UTF-8 bytes"):
        common.paths.validate_logical_name("a" * 256, label="name")
    with pytest.raises(ValueError, match="255 UTF-8 bytes"):
        common.paths.validate_logical_name("\u00e4" * 128, label="name")


def test_run_layout_requires_resolved_output_root() -> None:
    """A run cannot fall back to a hidden project or storage root."""
    config = {"run": _config(Path("unused"))["run"], "paths": {}}

    with pytest.raises(ValueError, match=r"config\.paths\.output_root is required"):
        common.paths.resolve_config_run_dir(config)


def test_physical_layout_root_locates_pair_records_and_rejects_links(tmp_path: Path) -> None:
    """Declared suffixes locate actual roots without assuming a fixed parent depth."""
    root = tmp_path / "experiments" / "elsewhere"
    config = _config(root, stage="b", branch_id="b" * 16, pair=_PAIR)
    run_dir = common.paths.resolve_config_run_dir(config)
    run_dir.mkdir(parents=True)
    assert common.paths.resolve_config_output_root(config, run_dir) == root
    marker = run_dir.parent / "experiment.json"
    marker.write_text("{}")
    assert common.paths.resolve_experiment_record_root(marker, config["run"]["layout"], experiments_root=tmp_path / "experiments") == root
    shared_a0 = marker.parent / ".." / "a0"
    a0_config = _config(root)
    shared_a0.resolve().mkdir(parents=True)
    assert common.paths.resolve_config_output_root(a0_config, shared_a0) == root
    with pytest.raises(ValueError, match=r"layout|suffix"):
        common.paths.resolve_config_output_root(config, root / "unrelated" / "b")
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        common.paths.resolve_config_output_root(config, link / run_dir.relative_to(root))


def test_dataset_state_files_stay_within_the_component_limit_for_long_names() -> None:
    """Locks use a digest key, so a Dataset name near 255 bytes still yields a valid file name."""
    name = "x" * 250
    lock = common.paths.resolve_dataset_build_lock_path("transient_drying", name, storage_root="/storage")

    assert len(lock.name.encode()) <= 255
    assert lock.name != common.paths.resolve_dataset_build_lock_path("steady_flow", name, storage_root="/storage").name
