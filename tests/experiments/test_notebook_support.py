# ruff: noqa: S101
"""Protect lightweight, read-only notebook support services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from support import configs

from src import experiments

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

support = experiments.notebook_support


def test_prepare_context_resolves_synthetic_dataset_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve public notebook context for absent synthetic Dataset packages."""
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    config_path = configs.write_yaml(
        tmp_path / "direct.yaml",
        configs.direct_config(),
    )

    context = support.prepare_notebook_context(config_path)

    assert isinstance(context, support.NotebookContext)
    assert context.config_path == config_path
    assert context.task.id == "steady_flow"
    assert tuple(preview.dataset_id for preview in context.dataset_previews) == (
        "synthetic_train",
        "synthetic_ood",
    )
    assert all(not preview.metadata_validated for preview in context.dataset_previews)
    assert all(preview.path.is_relative_to(storage_root) for preview in context.dataset_previews)
