"""Protect path-sensitive config semantics with temporary artificial files.

Responsibilities:
  - Check that artificial config category paths preserve resolved semantics
  - Check that a task base.yaml supplies shared values, is overridden, and is fail-closed

Design principles:
  - Equivalent authored values retain identity when their file location changes

This module does NOT:
  - Freeze production configuration defaults or allocate runs
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from support import configs

from src import experiments

if TYPE_CHECKING:
    from pathlib import Path


def _without_family(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy whose layout and run name no longer name the family folder and file stem that held the YAML."""
    copied = copy.deepcopy(config)
    copied["run"]["layout"].update({"family": None, "config_name": None})
    copied["run"]["name"] = None
    return copied


def test_family_folder_is_layout_only_and_does_not_change_resolved_semantics(tmp_path: Path) -> None:
    """Resolve identical test-owned requests from two arbitrary family folders."""
    raw = configs.direct_config()
    paths = [configs.write_config(tmp_path, raw, canonical_root=tmp_path, family=family) for family in ("category_a", "renamed_category")]

    resolved = [experiments.config.loader.load_and_resolve_config(path) for path in paths]

    assert [config["run"]["layout"]["family"] for config in resolved] == ["category_a", "renamed_category"]
    assert _without_family(resolved[0]) == _without_family(resolved[1])
    assert resolved[0]["run"]["name"] == resolved[1]["run"]["name"]


def _authored_path(root: Path, full: dict[str, Any], authored: dict[str, Any]) -> Path:
    """Return the path of an experiment file that sets ``authored``, named after its leaves as the loader requires."""
    effective = experiments.config.loader.deep_merge(experiments.config.defaults.get_task_defaults("steady_flow"), full)
    name = experiments.config.naming.config_name(effective, authored)
    return root / f"configs/03_learning/steady_flow/experiments/group/{name}.yaml"


def _split_request(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Write all but model and rate as the task base (with a batch size the file overrides); return the request, the file's mapping and path."""
    full = configs.direct_config()
    shared = {key: value for key, value in full.items() if key not in {"task", "model"}}
    shared["optimizer"] = {key: value for key, value in full["optimizer"].items() if key != "lr"}
    shared["data"] = {**full["data"], "batch_size": 99}
    configs.write_yaml(root / "configs/03_learning/steady_flow/base.yaml", shared)
    authored = {
        "task": "steady_flow",
        "model": full["model"],
        "optimizer": {"lr": full["optimizer"]["lr"]},
        "data": {"batch_size": full["data"]["batch_size"]},
    }
    return full, authored, _authored_path(root, full, {key: value for key, value in authored.items() if key != "task"})


def test_task_base_supplies_shared_values_and_authored_values_override(tmp_path: Path) -> None:
    """Resolve a request split into base and authored file exactly like the combined request."""
    full, authored, path = _split_request(tmp_path)
    configs.write_yaml(path, {**authored, "base": "../../base.yaml"})
    combined = configs.resolve(copy.deepcopy(full))  # in memory: a YAML holding the whole request would need the name of all its leaves

    assert _without_family(experiments.config.loader.load_and_resolve_config(path)) == _without_family(combined)

    learning_rate = float(full["optimizer"]["lr"]) / 2.0
    halved = {**authored, "optimizer": {"lr": learning_rate}}
    halved_path = _authored_path(tmp_path, full, {key: value for key, value in halved.items() if key != "task"})
    configs.write_yaml(halved_path, {**halved, "base": "../../base.yaml"})
    resolved = experiments.config.loader.load_and_resolve_config(halved_path)
    assert resolved["optimizer"]["lr"] == learning_rate
    assert resolved["scheduler"] == combined["scheduler"]


@pytest.mark.parametrize(
    "reference",
    [
        "/configs/03_learning/steady_flow/base.yaml",
        "../../shared.yaml",
        "../base.yaml",
        "../../../transient_drying/base.yaml",
        "../../missing/base.yaml",
        "../../../../../elsewhere/steady_flow/base.yaml",
    ],
)
def test_task_base_must_be_the_enclosing_task_base(tmp_path: Path, reference: str) -> None:
    """Reject absolute, renamed, non-enclosing, foreign-task and missing base references."""
    _full, authored, path = _split_request(tmp_path)
    configs.write_yaml(tmp_path / "configs/03_learning/steady_flow/experiments/base.yaml", {})
    configs.write_yaml(tmp_path / "configs/03_learning/transient_drying/base.yaml", {})
    configs.write_yaml(tmp_path / "configs/03_learning/steady_flow/shared.yaml", {})
    configs.write_yaml(tmp_path / "elsewhere/steady_flow/base.yaml", {})
    configs.write_yaml(path, {**authored, "base": reference})

    with pytest.raises(experiments.config.loader.ConfigError, match="base"):
        experiments.config.loader.load_and_resolve_config(path)


def test_task_base_cannot_chain_another_base(tmp_path: Path) -> None:
    """Keep one shared owner per value: a base naming a further base is rejected."""
    _full, authored, path = _split_request(tmp_path)
    configs.write_yaml(tmp_path / "configs/03_learning/steady_flow/base.yaml", {"base": "base.yaml"})
    configs.write_yaml(path, {**authored, "base": "../../base.yaml"})

    with pytest.raises(experiments.config.loader.ConfigError, match="another base"):
        experiments.config.loader.load_and_resolve_config(path)
