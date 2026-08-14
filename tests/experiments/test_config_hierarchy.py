# ruff: noqa: S101
"""Protect path-sensitive config semantics with temporary artificial files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from support import configs

from src import experiments

if TYPE_CHECKING:
    from pathlib import Path


def test_category_path_does_not_change_resolved_semantics(tmp_path: Path) -> None:
    """Resolve identical test-owned requests from two arbitrary category names."""
    raw = configs.direct_config()
    paths = (
        tmp_path / "configs/learning/steady_flow/experiments/category_a/request.yaml",
        tmp_path / "configs/learning/steady_flow/experiments/renamed_category/request.yaml",
    )
    for path in paths:
        configs.write_yaml(path, raw)

    resolved = [experiments.config.loader.load_and_resolve_config(path) for path in paths]

    assert resolved[0] == resolved[1]
    assert resolved[0]["run"]["name"] == resolved[1]["run"]["name"]
