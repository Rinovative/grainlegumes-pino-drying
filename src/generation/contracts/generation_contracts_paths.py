"""
generation_contracts_paths.py

Validate persistent Generation storage roots and shared workspace path bounds.
Responsibilities:
  - Resolve the sole persistent Generation storage root
  - Require absolute, user-owned, writable filesystem boundaries
  - Provide containment checks shared by runtime workspace services
Design principles:
  - Persistent roots are rejected inside repositories and home/root boundaries
  - Symlink and ownership checks precede creation or use
This module does NOT:
  - Create disposable workspaces, staging directories, or publications
"""

from __future__ import annotations

import os
from pathlib import Path

from src import common

CASE_WORKSPACE_MARKER = ".generation-case-workspace.json"


def absolute_path(value: Path | str, *, label: str) -> Path:
    """Resolve one non-empty absolute path without creating it."""
    if isinstance(value, str) and not value:
        message = f"{label} must not be empty."
        raise ValueError(message)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        message = f"{label} must be absolute before use: {candidate}"
        raise ValueError(message)
    if candidate.exists() and candidate.is_symlink():
        message = f"{label} must not itself be a symbolic link: {candidate}"
        raise ValueError(message)
    return candidate.resolve(strict=False)


def _nearest_existing_parent(path: Path) -> Path:
    """Return the first existing parent used for ownership checks."""
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            message = f"No existing parent is available for path: {path}"
            raise FileNotFoundError(message)
        candidate = parent
    return candidate


def require_user_owned_writable(path: Path, *, label: str) -> None:
    """Require the existing path or nearest parent to be user-owned and writable."""
    existing = _nearest_existing_parent(path)
    if not existing.is_dir() or existing.is_symlink():
        message = f"{label} has no safe existing directory boundary: {existing}"
        raise ValueError(message)
    if existing.stat().st_uid != os.getuid():
        message = f"{label} must be user-owned before use: {existing}"
        raise PermissionError(message)
    if not os.access(existing, os.W_OK | os.X_OK):
        message = f"{label} is not writable and searchable: {existing}"
        raise PermissionError(message)


def is_relative_to(path: Path, root: Path) -> bool:
    """Return whether one resolved path descends from another."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_storage_root(
    storage_root: Path | str | None,
    *,
    create: bool,
) -> Path:
    """Resolve the sole persistent Generation storage root safely."""
    configured = common.paths.get_storage_root(storage_root=storage_root)
    resolved = absolute_path(configured, label="storage_root")
    home = Path.home().resolve()
    repository = common.paths.get_project_root().resolve()
    if resolved in {Path("/"), home, repository} or is_relative_to(resolved, repository):
        message = f"storage_root targets a forbidden persistent boundary: {resolved}"
        raise ValueError(message)
    require_user_owned_writable(resolved, label="storage_root")
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        if not resolved.is_dir() or resolved.is_symlink():
            message = f"storage_root must be one safe directory: {resolved}"
            raise ValueError(message)
        require_user_owned_writable(resolved, label="storage_root")
    return resolved
