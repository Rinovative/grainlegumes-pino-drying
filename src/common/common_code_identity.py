"""Declare scientific-code projections and hash their syntax without discovery.

Responsibilities:
  - Define a code projection as declared directories, files, task filters, exemptions and composition
  - Resolve the sorted member list of a projection below one source root, with the package initializers it runs through, and fail on missing paths
  - Hash the syntax trees of a projection's members with a cache keyed on exact file content
  - Own the modules and directories that every projection treats as infrastructure outside scientific identity
  - Own the one guard that compares a recorded projection digest with the running code

Design principles:
  - Owners declare their projections; no import closure, sys.modules scan or other runtime discovery decides membership
  - A projection hash decides whether a unit's science changed; exact executed bytes are bound separately by source snapshots
  - A persisted result is evidence of the code that produced it: reading it compares only its own recorded evidence and
    its inputs, never the running code; only a request for a result already in place and a continuation of persisted
    computation compare a recorded digest with the running projection, through ``require_current_implementation``
    or a composite signature that embeds the recorded digest
  - Every exemption and infrastructure entry states its reason, so tests can require that each static import is classified
  - Package initializers decide which module an alias or lazy name dispatches to, so they are hashed with the modules they front

This module does NOT:
  - Compute import closures or prove that a projection is complete; the projection tests classify every static import
  - Capture source bytes, snapshots, interpreters or numerical environments
  - Choose which scientific units a projection identifies
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from . import common_serialization as serialization

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_EXECUTING_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_CACHE_SIZE = 64
_PACKAGE_INIT = "__init__.py"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_EXECUTION_REASON = "execution, scheduling and publication I/O; solver commands and templates are bound by their own recorded evidence"

INFRASTRUCTURE_MODULES: Mapping[str, str] = MappingProxyType(
    {
        "common/common_paths.py": "path resolution places artifacts and never changes their scientific content",
        "common/common_locking.py": "process coordination around artifacts, not scientific computation",
        "common/common_names.py": "name tokens label artifacts and are never scientific identity",
        "common/common_identity_registry.py": "one-owner registry of content digests; the digests it stores decide identity",
        "experiments/experiments_console.py": "console reporting of lifecycle payloads only",
        "experiments/experiments_tracking.py": "optional external tracking observer of local state",
        "experiments/experiments_run_registry.py": "run-directory bookkeeping; the resolved-config digest decides identity",
        "experiments/experiments_run_identity.py": "run identity evidence is recorded, not recomputed from code",
        "experiments/experiments_run_source_binding.py": "execution-tree selection from saved sessions, bound by the frozen source tree",
        "generation/contracts/generation_contracts_paths.py": "storage-root safety checks, not scientific content",
        "generation/contracts/generation_contracts_provenance.py": "source and parameter provenance records validated as evidence",
    }
)

INFRASTRUCTURE_DIRECTORIES: Mapping[str, str] = MappingProxyType(
    {
        "generation/runtime": _EXECUTION_REASON,
        "generation/publication": _EXECUTION_REASON,
    }
)


@lru_cache(maxsize=_CACHE_SIZE)
def _implementation_from_content(root: str, files: tuple[tuple[str, str], ...]) -> str:
    """Reuse AST parsing only for the exact observed file-content selection."""
    return serialization.python_implementation_sha256(Path(root), [name for name, _digest in files])


def _relative_python_path(name: str) -> str:
    """Return a validated contained relative Python path in POSIX form."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        message = f"Projection file must be a contained relative Python file: {name!r}"
        raise ValueError(message)
    return relative.as_posix()


def _relative_directory(name: str) -> str:
    """Return a validated contained relative directory in POSIX form."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        message = f"Projection directory must be a contained relative path: {name!r}"
        raise ValueError(message)
    return relative.as_posix()


def _package_initializers(source_root: Path, members: Iterable[str]) -> set[str]:
    """Return the existing ``__init__.py`` of every member's package and of each parent package up to the source root."""
    initializers: set[str] = set()
    for member in members:
        for package in (Path(member).parent, *Path(member).parent.parents):
            initializer = (package / _PACKAGE_INIT).as_posix()
            if (source_root / initializer).is_file():
                initializers.add(initializer)
    return initializers


@dataclass(frozen=True, slots=True)
class CodeProjection:
    """Declared set of source modules whose syntax decides one scientific identity.

    Attributes
    ----------
    name : str
        Stable label used in messages and tests.
    directories : tuple[str, ...]
        Source-relative directories; every ``*.py`` file below them, recursively,
        is a declared member unless its filename contains an excluded marker.
    files : tuple[str, ...]
        Source-relative Python files that are declared members unconditionally.
    excluded_markers : tuple[str, ...]
        Filename substrings that remove directory members owned by another task.
        Explicit ``files`` are never filtered.
    exemptions : Mapping[str, str]
        Source-relative modules that a member may import without being a member,
        each with a one-line reason. Exemptions never change the members or the hash.
    includes : tuple[CodeProjection, ...]
        Projections whose members are part of this one, for example a study
        projection that includes the training projection of its task.

    """

    name: str
    directories: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    excluded_markers: tuple[str, ...] = ()
    exemptions: Mapping[str, str] = field(default_factory=dict)
    includes: tuple[CodeProjection, ...] = ()

    def declared_members(self, root: Path | None = None) -> tuple[str, ...]:
        """Return the sorted source-relative modules the projection declares, with composition resolved.

        Parameters
        ----------
        root : Path | None, optional
            Source directory (the ``src`` package directory). Defaults to the
            executing source directory.

        Returns
        -------
        tuple[str, ...]
            Sorted, de-duplicated source-relative POSIX paths of the declared
            directories and files, without the implied package initializers.

        Raises
        ------
        ValueError
            A declared directory or file does not exist below ``root``, or a
            declared path is not a contained relative path.

        """
        source_root = _EXECUTING_SOURCE_ROOT if root is None else Path(root)
        selected: set[str] = set()
        for directory in self.directories:
            base = source_root / _relative_directory(directory)
            if not base.is_dir():
                message = f"Code projection {self.name!r} declares a missing directory: {directory!r} below {source_root}"
                raise ValueError(message)
            selected.update(
                path.relative_to(source_root).as_posix()
                for path in base.rglob("*.py")
                if not any(marker in path.name for marker in self.excluded_markers)
            )
        for name in self.files:
            relative = _relative_python_path(name)
            if not (source_root / relative).is_file():
                message = f"Code projection {self.name!r} declares a missing file: {name!r} below {source_root}"
                raise ValueError(message)
            selected.add(relative)
        for included in self.includes:
            selected.update(included.declared_members(source_root))
        return tuple(sorted(selected))

    def members(self, root: Path | None = None) -> tuple[str, ...]:
        """Return the sorted hashed members: the declared modules and the package initializers they run through.

        Importing a member executes the ``__init__.py`` of its package and of every parent package up to
        ``src/__init__.py``, and a package initializer decides which module an alias or lazy name dispatches
        to. Every existing initializer on those paths is therefore a member, so changing a dispatch changes
        the digest.

        Parameters
        ----------
        root : Path | None, optional
            Source directory (the ``src`` package directory). Defaults to the
            executing source directory.

        Returns
        -------
        tuple[str, ...]
            Sorted, de-duplicated source-relative POSIX paths.

        Raises
        ------
        ValueError
            A declared directory or file does not exist below ``root``, or a
            declared path is not a contained relative path.

        """
        source_root = _EXECUTING_SOURCE_ROOT if root is None else Path(root)
        declared = self.declared_members(source_root)
        return tuple(sorted({*declared, *_package_initializers(source_root, declared)}))

    def sha256(self, root: Path | None = None) -> str:
        """Return the syntax digest of the members, ignoring comments and docstrings.

        Parameters
        ----------
        root : Path | None, optional
            Source directory (the ``src`` package directory). Defaults to the
            executing source directory.

        Returns
        -------
        str
            Lowercase hexadecimal SHA-256 digest of the member syntax trees.

        Raises
        ------
        ValueError
            The projection is empty or declares a missing path.
        OSError, SyntaxError
            A member cannot be read or parsed.

        """
        source_root = (_EXECUTING_SOURCE_ROOT if root is None else Path(root)).resolve(strict=True)
        names = self.members(source_root)
        files = tuple((name, serialization.file_sha256(source_root / name)) for name in names)
        return _implementation_from_content(str(source_root), files)


def require_current_implementation(projection: CodeProjection, recorded_sha256: object, *, unit: str, remedy: str) -> None:
    """
    Refuse to continue or refresh a persisted result whose recorded implementation differs from the running code.

    Parameters
    ----------
    projection : CodeProjection
        Projection of the computation that would be continued.
    recorded_sha256 : object
        Digest the persisted result recorded for that projection.
    unit : str
        Human-readable name of the persisted result, used as the message prefix.
    remedy : str
        Action that resolves the refusal, appended to the message.

    Raises
    ------
    ValueError
        If the recorded digest is not a lowercase hexadecimal SHA-256 digest or differs from the running projection.

    """
    if not isinstance(recorded_sha256, str) or _SHA256_PATTERN.fullmatch(recorded_sha256) is None:
        msg = f"{unit}: recorded {projection.name} implementation digest is missing or malformed; {remedy}"
        raise ValueError(msg)
    current = projection.sha256()
    if recorded_sha256 != current:
        msg = f"{unit}: {projection.name} implementation changed (recorded {recorded_sha256[:12]}, current {current[:12]}); {remedy}"
        raise ValueError(msg)
