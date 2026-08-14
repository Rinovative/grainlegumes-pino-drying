"""
===============================================================================
generation_contracts_templates.py
===============================================================================
Resolve configured COMSOL template paths and adjacent digest sidecars.
Responsibilities:
  - Validate one repository-relative configured COMSOL template path
  - Derive and parse the adjacent SHA-256 sidecar
  - Bind the expected digest to the exact template bytes
Design principles:
  - Profile YAML owns the concrete path while code owns validation
  - Path containment and byte identity fail closed
  - Sidecar updates are explicit user actions
This module does NOT:
  - Map profile identifiers to files or paths
  - Load profile YAML, update sidecars, or execute COMSOL
===============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src import common

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ResolvedTemplateIdentity:
    """Bind one configured template locator to its verified byte identity."""

    relative_path: str
    absolute_path: Path
    sidecar_path: Path
    sha256: str


def _safe_relative_path(value: Any, *, label: str, suffix: str | None = None) -> tuple[str, Path]:
    """Return one normalized repository-relative path with no traversal."""
    if not isinstance(value, str) or not value or value.strip() != value or any(character in value for character in ("\x00", "\n", "\r")):
        message = f"{label} must be non-empty text."
        raise ValueError(message)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path():
        message = f"{label} must be one repository-relative path without '..'."
        raise ValueError(message)
    normalized = path.as_posix()
    if normalized != value or (suffix is not None and path.suffix.lower() != suffix):
        message = f"{label} must be one normalized repository-relative {suffix or 'file'} path."
        raise ValueError(message)
    return normalized, path


def _contained_regular_file(path: Path, *, root: Path, label: str) -> Path:
    """Require one regular file whose resolved target remains in the repository."""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        message = f"{label} is missing: {path}"
        raise FileNotFoundError(message) from error
    if not resolved.is_relative_to(root):
        message = f"{label} escapes the repository through its resolved path: {path}"
        raise ValueError(message)
    if not path.is_file():
        message = f"{label} must be a regular file: {path}"
        raise ValueError(message)
    return resolved


def _sidecar_digest(sidecar_path: Path) -> str:
    """Read one digest-only adjacent template identity sidecar."""
    try:
        digest = sidecar_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        message = f"Could not read COMSOL template identity sidecar: {sidecar_path}"
        raise ValueError(message) from error
    if _SHA256_PATTERN.fullmatch(digest) is None:
        message = f"COMSOL template identity sidecar must contain one lowercase SHA-256 digest: {sidecar_path}"
        raise ValueError(message)
    return digest


def validate_template_relative_path(value: Any, *, label: str = "COMSOL template path") -> str:
    """
    Validate one persisted repository-relative COMSOL template locator.

    Parameters
    ----------
    value : Any
        Candidate repository-relative path.
    label : str, optional
        Context used by validation errors.

    Returns
    -------
    str
        Normalized repository-relative MPH path.

    Raises
    ------
    ValueError
        If the value is empty, absolute, traversing, non-normalized, or not an
        MPH path.

    """
    relative_path, _path = _safe_relative_path(
        value,
        label=label,
        suffix=".mph",
    )
    return relative_path


def resolve_template_identity(
    configured_path: Any,
    *,
    repository_root: Path | str | None = None,
) -> ResolvedTemplateIdentity:
    """
    Resolve and validate one configured COMSOL template identity.

    Parameters
    ----------
    configured_path : Any
        Repository-relative ``.mph`` path authored by one profile YAML.
    repository_root : Path | str | None, optional
        Repository root used to resolve the configured path.

    Returns
    -------
    ResolvedTemplateIdentity
        Safe path, mechanically adjacent sidecar, and verified byte digest.

    Raises
    ------
    FileNotFoundError
        If the template or adjacent sidecar is missing.
    ValueError
        If a path, sidecar, or byte digest violates the template contract.

    """
    relative_path = validate_template_relative_path(
        configured_path,
        label="Configured COMSOL template",
    )
    relative = Path(relative_path)
    root_value = common.paths.get_project_root() if repository_root is None else Path(repository_root).expanduser()
    root = root_value.resolve(strict=True)
    if not root.is_dir():
        message = f"COMSOL template repository root must be a directory: {root}"
        raise NotADirectoryError(message)
    absolute_path = root / relative
    _contained_regular_file(
        absolute_path,
        root=root,
        label="Configured COMSOL template",
    )
    sidecar_path = absolute_path.with_suffix(".sha256")
    _contained_regular_file(
        sidecar_path,
        root=root,
        label="Adjacent COMSOL template identity sidecar",
    )
    expected = _sidecar_digest(sidecar_path)
    actual = common.serialization.file_sha256(absolute_path)
    if actual != expected:
        message = f"COMSOL template SHA-256 mismatch for {relative_path!r}: expected {expected}, got {actual}."
        raise ValueError(message)
    return ResolvedTemplateIdentity(
        relative_path=relative_path,
        absolute_path=absolute_path,
        sidecar_path=sidecar_path,
        sha256=expected,
    )
