"""Enforce explicit opt-in and storage ownership for real-package tests.

Responsibilities:
  - Resolve the explicit storage root that real-package acceptance requires

Design principles:
  - RUN_REAL_DATA_TESTS=1 and an explicit coherent STORAGE_ROOT are required without fallback

This module does NOT:
  - Enable real-data tests implicitly or create missing packages
"""

from __future__ import annotations

import os
from pathlib import Path

from src import common

REAL_DATA_FLAG = "RUN_REAL_DATA_TESTS"
STORAGE_ROOT = "STORAGE_ROOT"


def real_data_tests_enabled() -> bool:
    """Return whether strict mounted-package acceptance was explicitly enabled."""
    return os.environ.get(REAL_DATA_FLAG) == "1"


def require_real_storage_root() -> Path:
    """Return the explicit unified storage root without a fallback."""
    if not real_data_tests_enabled():
        message = f"{REAL_DATA_FLAG}=1 is required for real-data acceptance"
        raise RuntimeError(message)
    configured = os.environ.get(STORAGE_ROOT, "").strip()
    if not configured:
        message = f"{STORAGE_ROOT} is required for real-data acceptance"
        raise RuntimeError(message)
    root = Path(configured).expanduser()
    if common.paths.get_storage_root() != root:
        message = "Maintained storage path ownership disagrees with the explicit real-data root"
        raise RuntimeError(message)
    return root
