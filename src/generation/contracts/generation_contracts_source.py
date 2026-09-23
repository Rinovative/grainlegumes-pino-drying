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
import sys
from pathlib import Path
from typing import Any

from src import common

GIT_COMMIT_ENVIRONMENT_VARIABLE = "GENERATION_GIT_COMMIT"
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SOURCE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SourceAdmissionError(RuntimeError):
    """Report changed or unavailable shared source before durable publication."""


def validate_admitted_source_before_publication() -> None:
    """Reject changed shared source before Slurm work publishes durable evidence."""
    expected = os.environ.get("GENERATION_SOURCE_SHA256")
    if expected is None:
        return
    if _SOURCE_SHA256_PATTERN.fullmatch(expected) is None:
        message = "Generation source admission fingerprint is malformed."
        raise SourceAdmissionError(message)
    try:
        expected_commit = required_git_commit()
    except (RuntimeError, ValueError) as error:
        message = "Generation source admission commit is missing or malformed."
        raise SourceAdmissionError(message) from error
    repository = common.paths.get_project_root().resolve()
    try:
        current_commit = subprocess.run(  # noqa: S603 -- fixed Git argument vector
            ["git", "-C", str(repository), "rev-parse", "HEAD"],  # noqa: S607 -- site PATH owns Git
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        message = "Shared Generation commit cannot be verified before publication."
        raise SourceAdmissionError(message) from error
    if current_commit != expected_commit:
        message = "Shared Generation commit changed after Slurm admission; publication refused."
        raise SourceAdmissionError(message)
    try:
        result = subprocess.run(  # noqa: S603 -- fixed source fingerprint invocation
            [sys.executable, str(repository / "scripts/source_fingerprint.py"), str(repository)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        message = "Shared Generation source cannot be fingerprinted before publication."
        raise SourceAdmissionError(message) from error
    if result.stdout.strip() != expected:
        message = "Shared Generation source changed after Slurm admission; publication refused."
        raise SourceAdmissionError(message)


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
