"""
generation_contracts_source.py

Validate exact source-repository provenance for generation execution.
Responsibilities:
  - Validate one full lowercase Git object identifier
  - Require the launch-provided source commit from the process environment
  - Resolve a clean repository HEAD for direct interactive generation
Design principles:
  - Source commit is execution provenance, not scientific case identity
  - Missing or abbreviated commit evidence fails closed before case generation
This module does NOT:
  - Mutate, fetch, or check out a Git repository
  - Add source revisions to human-readable batch or dataset names
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from src import common

GIT_COMMIT_ENVIRONMENT_VARIABLE = "GENERATION_GIT_COMMIT"
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_git_commit(value: Any) -> str:
    """Return one exact full lowercase Git object identifier."""
    if not isinstance(value, str) or _GIT_COMMIT_PATTERN.fullmatch(value) is None:
        message = "git_commit must be one exact 40-character lowercase Git object identifier."
        raise ValueError(message)
    return value


def required_git_commit() -> str:
    """Return the exact source commit provided by the generation launcher."""
    value = os.environ.get(GIT_COMMIT_ENVIRONMENT_VARIABLE)
    if value is None:
        message = f"{GIT_COMMIT_ENVIRONMENT_VARIABLE} is required for generation provenance."
        raise RuntimeError(message)
    return validate_git_commit(value)


def clean_repository_git_commit(
    repository_root: Path | str | None = None,
) -> str:
    """Return HEAD only when the complete repository worktree is clean."""
    root = common.paths.get_project_root() if repository_root is None else Path(repository_root).expanduser()
    repository = root.resolve()
    if not repository.is_dir():
        message = f"Generation source repository is not a directory: {repository}"
        raise NotADirectoryError(message)
    try:
        status = subprocess.run(  # noqa: S603 -- fixed Git inspection command
            ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            message = "Direct input generation requires a clean repository worktree for truthful source provenance."
            raise RuntimeError(message)
        commit = subprocess.run(  # noqa: S603 -- fixed Git inspection command
            ["git", "-C", str(repository), "rev-parse", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        message = f"Could not resolve exact Git source identity from {repository}."
        raise RuntimeError(message) from error
    return validate_git_commit(commit)
