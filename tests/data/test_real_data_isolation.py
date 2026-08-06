# ruff: noqa: S101
"""Protect the explicit opt-in boundary for mounted production packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from support import real_data

if TYPE_CHECKING:
    from pathlib import Path


def test_real_data_acceptance_requires_flag_and_storage_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ordinary sessions and enabled sessions without explicit storage."""
    monkeypatch.delenv(real_data.REAL_DATA_FLAG, raising=False)
    assert not real_data.real_data_tests_enabled()
    with pytest.raises(RuntimeError, match="RUN_REAL_DATA_TESTS=1"):
        real_data.require_real_storage_root()

    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    monkeypatch.delenv(real_data.STORAGE_ROOT, raising=False)
    with pytest.raises(RuntimeError, match="STORAGE_ROOT"):
        real_data.require_real_storage_root()


def test_enabled_real_data_acceptance_derives_numbered_areas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve numbered areas and fail when requested packages are absent."""
    storage_root = tmp_path / "mounted-storage"
    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    monkeypatch.setenv(real_data.STORAGE_ROOT, str(storage_root))

    assert real_data.require_real_storage_root() == storage_root
    assert real_data.require_real_data_root() == storage_root / "02_datasets"
    assert real_data.require_real_generation_root() == storage_root / "01_generation"
    with pytest.raises(FileNotFoundError, match="artificial_missing_package"):
        real_data.require_real_metadata_package("artificial_missing_package")
    with pytest.raises(FileNotFoundError, match="artificial_missing_package"):
        real_data.require_real_generated_batch("artificial_missing_package")
