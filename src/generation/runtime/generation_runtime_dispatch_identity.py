"""Define and resolve the dispatch identity of one campaign case step.

Responsibilities:
  - Define the dispatch-key syntax ``<allocation>-<submission_index>``
  - Build and parse dispatch keys with fail-closed validation
  - Resolve the dispatch identity of the running campaign case from its step environment

Design principles:
  - One authority for the key syntax shared by producers, validators, and readers
  - A campaign context with an incomplete or inconsistent identity fails closed
  - Outside a campaign context no identity is invented

This module does NOT:
  - Read or write campaign manifests or per-case evidence files
  - Submit, query, or cancel scheduler work

Notes:
  Identity:
    - Several case steps share one node-job allocation, so the allocation id alone never identifies a dispatch.
    - The submission index is the campaign-wide dispatch counter, unique within one campaign run.

"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

CAMPAIGN_RUN_ID_ENVIRONMENT: Final = "GENERATION_CAMPAIGN_RUN_ID"
SUBMISSION_INDEX_ENVIRONMENT: Final = "GENERATION_SUBMISSION_INDEX"
DISPATCH_KEY_ENVIRONMENT: Final = "GENERATION_DISPATCH_KEY"
DISPATCH_KEY_PATTERN: Final = re.compile(r"[0-9]+-[1-9][0-9]*")
_NUMERIC_PATTERN: Final = re.compile(r"[0-9]+")
_POSITIVE_INTEGER_PATTERN: Final = re.compile(r"[1-9][0-9]*")


def dispatch_key(allocation_id: str, submission_index: int) -> str:
    """Return the canonical ``<allocation>-<submission_index>`` dispatch key."""
    if not isinstance(allocation_id, str) or _NUMERIC_PATTERN.fullmatch(allocation_id) is None:
        message = f"Dispatch allocation id must be a numeric Slurm job id: {allocation_id!r}."
        raise ValueError(message)
    if isinstance(submission_index, bool) or not isinstance(submission_index, int):
        message = "Dispatch submission index must be an integer."
        raise TypeError(message)
    if submission_index < 1:
        message = f"Dispatch submission index must be positive: {submission_index}."
        raise ValueError(message)
    return f"{allocation_id}-{submission_index}"


def parse_dispatch_key(key: str) -> tuple[str, int]:
    """Return ``(allocation_id, submission_index)`` of one canonical dispatch key."""
    if not isinstance(key, str) or DISPATCH_KEY_PATTERN.fullmatch(key) is None:
        message = f"Dispatch key must match '<allocation>-<submission_index>': {key!r}."
        raise ValueError(message)
    allocation_id, _separator, index = key.partition("-")
    return allocation_id, int(index)


@dataclass(frozen=True, slots=True)
class CaseDispatch:
    """The validated dispatch identity of one running campaign case step."""

    allocation_id: str
    step_id: str
    submission_index: int
    dispatch_key: str

    def __post_init__(self) -> None:
        """Validate the step id and the agreement of the key with its parts."""
        if not isinstance(self.step_id, str) or _NUMERIC_PATTERN.fullmatch(self.step_id) is None:
            message = f"Dispatch step id must be a numeric Slurm step id: {self.step_id!r}."
            raise ValueError(message)
        if dispatch_key(self.allocation_id, self.submission_index) != self.dispatch_key:
            message = "Dispatch key does not match its allocation id and submission index."
            raise ValueError(message)


def current_case_dispatch(environ: Mapping[str, str] = os.environ) -> CaseDispatch | None:
    """
    Resolve the dispatch identity of the running campaign case.

    Parameters
    ----------
    environ : Mapping[str, str], optional
        Step environment; defaults to the process environment.

    Returns
    -------
    CaseDispatch | None
        ``None`` outside a campaign case, i.e. when ``GENERATION_CAMPAIGN_RUN_ID`` is absent.

    Raises
    ------
    RuntimeError
        If a campaign context lacks or contradicts any part of its dispatch identity.

    """
    if CAMPAIGN_RUN_ID_ENVIRONMENT not in environ:
        return None
    allocation_id = environ.get("SLURM_JOB_ID")
    step_id = environ.get("SLURM_STEP_ID")
    raw_index = environ.get(SUBMISSION_INDEX_ENVIRONMENT)
    declared_key = environ.get(DISPATCH_KEY_ENVIRONMENT)
    if allocation_id is None or _NUMERIC_PATTERN.fullmatch(allocation_id) is None:
        message = "Campaign case dispatch requires one numeric SLURM_JOB_ID."
        raise RuntimeError(message)
    if step_id is None or _NUMERIC_PATTERN.fullmatch(step_id) is None:
        message = "Campaign case dispatch requires one numeric SLURM_STEP_ID."
        raise RuntimeError(message)
    if raw_index is None or _POSITIVE_INTEGER_PATTERN.fullmatch(raw_index) is None:
        message = f"Campaign case dispatch requires a positive {SUBMISSION_INDEX_ENVIRONMENT}."
        raise RuntimeError(message)
    submission_index = int(raw_index)
    expected_key = dispatch_key(allocation_id, submission_index)
    if declared_key != expected_key:
        message = f"{DISPATCH_KEY_ENVIRONMENT} {declared_key!r} does not match the step identity {expected_key!r}."
        raise RuntimeError(message)
    return CaseDispatch(
        allocation_id=allocation_id,
        step_id=step_id,
        submission_index=submission_index,
        dispatch_key=expected_key,
    )
