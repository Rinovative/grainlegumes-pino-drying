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
    """Return a copy whose layout no longer names the family folder that held the YAML."""
    copied = copy.deepcopy(config)
    copied["run"]["layout"]["family"] = None
    return copied


def test_family_folder_is_layout_only_and_does_not_change_resolved_semantics(tmp_path: Path) -> None:
    """Resolve identical test-owned requests from two arbitrary family folders."""
    raw = configs.direct_config()
    label = configs.source_for(raw).name
    paths = (
        tmp_path / f"configs/03_learning/steady_flow/experiments/category_a/{label}.yaml",
        tmp_path / f"configs/03_learning/steady_flow/experiments/renamed_category/{label}.yaml",
    )
    for path in paths:
        configs.write_yaml(path, raw)

    resolved = [experiments.config.loader.load_and_resolve_config(path) for path in paths]

    assert [config["run"]["layout"]["family"] for config in resolved] == ["category_a", "renamed_category"]
    assert _without_family(resolved[0]) == _without_family(resolved[1])
    assert resolved[0]["run"]["name"] == resolved[1]["run"]["name"]


_SHARED_SECTIONS = ("optimizer", "scheduler", "training")


def _split_request(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Write the shared sections of one test-owned request as its task base; return full and authored mappings."""
    full = configs.direct_config()
    shared = {key: full[key] for key in _SHARED_SECTIONS}
    authored = {key: value for key, value in full.items() if key not in _SHARED_SECTIONS}
    configs.write_yaml(root / "configs/03_learning/steady_flow/base.yaml", shared)
    return full, authored, root / f"configs/03_learning/steady_flow/experiments/group/{configs.source_for(full).name}.yaml"


def test_task_base_supplies_shared_values_and_authored_values_override(tmp_path: Path) -> None:
    """Resolve a request split into base and authored file exactly like the combined request."""
    full, authored, path = _split_request(tmp_path)
    configs.write_yaml(path, {**authored, "base": "../../base.yaml"})
    combined = tmp_path / f"combined/{configs.source_for(full).name}.yaml"
    configs.write_yaml(combined, full)

    assert _without_family(experiments.config.loader.load_and_resolve_config(path)) == _without_family(
        experiments.config.loader.load_and_resolve_config(combined)
    )

    learning_rate = float(full["optimizer"]["lr"]) / 2.0
    configs.write_yaml(path, {**authored, "base": "../../base.yaml", "optimizer": {"lr": learning_rate}})
    resolved = experiments.config.loader.load_and_resolve_config(path)
    assert resolved["optimizer"]["lr"] == learning_rate
    assert resolved["scheduler"] == experiments.config.loader.load_and_resolve_config(combined)["scheduler"]


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
