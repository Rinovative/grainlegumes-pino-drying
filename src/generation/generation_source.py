"""
===============================================================================
generation_source.py
===============================================================================
Validate exact source-repository provenance for generation execution.
Responsibilities:
  - Validate one full lowercase Git object identifier
  - Require the launch-provided source commit from the process environment
Design principles:
  - Source commit is execution provenance, not scientific case identity
  - Missing or abbreviated commit evidence fails closed before case generation
This module does NOT:
  - Inspect, mutate, fetch, or check out a Git repository
  - Add source revisions to human-readable batch or dataset names
===============================================================================
"""

from __future__ import annotations

import os
import re
from typing import Any

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
