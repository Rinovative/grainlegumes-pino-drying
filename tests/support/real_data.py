"""Own the explicit flag and unified storage boundary for real-package tests."""

from __future__ import annotations

import os
from pathlib import Path

from src import common, datasets

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


def require_real_generation_root() -> Path:
    """Return the generated-simulation area below explicit real storage."""
    return common.paths.get_generation_root(storage_root=require_real_storage_root())


def require_real_generated_batch(batch_id: str) -> Path:
    """Return one mandatory completed generated batch or fail without fallback."""
    root = require_real_generation_root()
    required = (
        root / "meta" / f"{batch_id}.csv",
        root / "meta" / f"{batch_id}.json",
        root / "raw" / batch_id / "batch_manifest.json",
        root / "processed" / batch_id,
    )
    if not all(path.is_file() if path.suffix else path.is_dir() for path in required):
        message = f"Required real generated batch {batch_id!r} is missing"
        raise FileNotFoundError(message)
    return root


def require_real_data_root() -> Path:
    """Return the final-dataset area below explicit real storage."""
    return common.paths.get_datasets_root(storage_root=require_real_storage_root())


def require_real_metadata_package(dataset_id: str) -> Path:
    """Return one mandatory real metadata package or fail without fallback."""
    require_real_data_root()
    metadata_dir = common.paths.resolve_dataset_metadata_dir(dataset_id)
    required = (
        metadata_dir / datasets.metadata.METADATA_FILENAME,
        metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME,
    )
    if not all(path.is_file() for path in required):
        message = f"Required real production package {dataset_id!r} is missing"
        raise FileNotFoundError(message)
    return metadata_dir
