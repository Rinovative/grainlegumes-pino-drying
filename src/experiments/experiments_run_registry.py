"""Bind every experiment identity to exactly one run directory.

Responsibilities:
  - Derive the registry scope, digest and kind of a resolved experiment config
  - Reject a fresh start whose identity already owns another existing directory
  - Report that duplicate as an identical experiment with both directories

Design principles:
  - The shared identity registry owns entries, locking and stale replacement
  - The config loader's allocation digest (epoch-free except for A0 and Optuna trials) decides identity; names and paths never do
  - A ``--revision`` label is presentation, so it never makes an identical experiment new

This module does NOT:
  - Allocate, train, resume or validate runs. ``experiments.run`` owns the lifecycle
  - Move or delete run directories, or decide which differences may continue a run
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src import common

from .config import experiments_config_loader as config_loader

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager
    from pathlib import Path

_KINDS = frozenset({"run", "trial"})


class IdenticalExperimentError(FileExistsError):
    """Report a fresh start whose exact experiment already lives in another directory."""

    def __init__(self, *, existing_run_dir: Path, requested_run_dir: Path) -> None:
        """Keep both directories for callers that report or classify the duplicate."""
        self.existing_run_dir = existing_run_dir
        self.requested_run_dir = requested_run_dir
        super().__init__(
            f"The identical experiment already exists at {existing_run_dir}. Continue that run instead of starting a copy "
            f"at {requested_run_dir}; a --revision label does not make identical science a new experiment."
        )


def _registry(config: Mapping[str, Any]) -> common.identity_registry.IdentityRegistry:
    """Scope the registry to the config's resolved output root."""
    return common.identity_registry.IdentityRegistry(config["paths"]["output_root"], _KINDS)


def _kind(config: Mapping[str, Any]) -> str:
    """Classify a resolved config as an ordinary run or an Optuna trial."""
    return "trial" if config["run"]["layout"]["trial"] is not None else "run"


def _conflict(existing: Path, requested: Path) -> IdenticalExperimentError:
    """Build the duplicate-start error for the shared registry."""
    return IdenticalExperimentError(existing_run_dir=existing, requested_run_dir=requested)


def lookup(config: Mapping[str, Any]) -> common.identity_registry.RegistryEntry | None:
    """Return the current owner of a resolved config's identity, or ``None`` if unowned or stale."""
    return _registry(config).lookup(config_loader.allocation_digest(config))


def ensure_available(config: Mapping[str, Any], run_dir: Path | str) -> None:
    """Fail fast before any allocation when another existing directory owns this identity.

    Raises
    ------
    IdenticalExperimentError
        If the identity is bound to an existing directory other than ``run_dir``.

    """
    _registry(config).ensure_available(config_loader.allocation_digest(config), run_dir, conflict=_conflict)


def ensure_bound(config: Mapping[str, Any], run_dir: Path | str) -> None:
    """Bind an existing run that continues, if its binding is missing or stale.

    Raises
    ------
    IdenticalExperimentError
        If another existing directory already owns the identity.

    """
    _registry(config).ensure_bound(config_loader.allocation_digest(config), run_dir, kind=_kind(config), conflict=_conflict)


def reserve(config: Mapping[str, Any], run_dir: Path | str) -> AbstractContextManager[None]:
    """Bind ``run_dir`` under the identity lock while the caller allocates it.

    Raises
    ------
    IdenticalExperimentError
        If another existing directory already owns the identity.
    ValueError
        If ``run_dir`` lies outside the config's output root.

    """
    return _registry(config).reserve(config_loader.allocation_digest(config), run_dir, kind=_kind(config), conflict=_conflict)
