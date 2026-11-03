"""Protect the explicit opt-in boundary for mounted production packages.

Responsibilities:
  - Check that real-data access requires both an opt-in flag and explicit storage root

Design principles:
  - Missing authorization or storage configuration fails without fallback

This module does NOT:
  - Read production packages as part of the default synthetic suite
"""

from __future__ import annotations

import pytest
from support import real_data


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
