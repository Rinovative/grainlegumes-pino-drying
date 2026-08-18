"""
generation_validation_policy.py

Define shared blocking-integrity and advisory-diagnostic severity contracts.
Responsibilities:
  - Classify stable validation codes as blocking or advisory
  - Construct complete immutable advisory diagnostic records
  - Keep quality flags distinct from structural publication failures
Design principles:
  - Structural, identity, and persistence integrity remain fail-closed
  - Finite scientific and numerical plausibility findings remain observational
  - Diagnostic records carry explicit evidence, thresholds, and timestamps
This module does NOT:
  - Validate concrete files, arrays, identities, or scientific formulas
  - Decide Dataset filtering or campaign outcomes
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ValidationDisposition = Literal["blocking", "advisory"]
AdvisorySeverity = Literal["info", "warning"]

BLOCKING_VALIDATION_CODES: Final = frozenset(
    {
        "array_shape_mismatch",
        "atomic_publication_failed",
        "column_inventory_mismatch",
        "conflicting_publication",
        "corrupt_hdf5",
        "duplicate_time_state",
        "hash_mismatch",
        "identity_mismatch",
        "missing_authoritative_provenance",
        "missing_required_export",
        "missing_required_input",
        "nonfinite_required_value",
        "path_escape",
        "schema_mismatch",
        "symlink_escape",
        "time_ordering_invalid",
        "unreadable_required_file",
    }
)
ADVISORY_VALIDATION_CODES: Final = frozenset(
    {
        "bulk_moisture_tolerance_exceeded",
        "diagnostic_tolerance_exceeded",
        "final_status_consistency",
        "initial_state_tolerance_exceeded",
        "mass_balance_tolerance_exceeded",
        "optional_diagnostic_unavailable",
        "physical_range_unexpected",
        "relative_humidity_excursion",
        "scientific_hypothesis_difference",
        "solver_nonlinear_failure_count_high",
        "solver_step_size_small",
        "solver_time_failure_count_high",
    }
)


class DiagnosticRecord(TypedDict):
    """Persisted post-solver advisory diagnostic evidence."""

    code: str
    severity: AdvisorySeverity
    stage: str
    message: str
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    source_artifacts: list[str]
    recorded_at: str
    quality_flag: bool


def validation_disposition(code: str) -> ValidationDisposition:
    """Return the shared disposition for one stable validation code."""
    if code in BLOCKING_VALIDATION_CODES:
        return "blocking"
    if code in ADVISORY_VALIDATION_CODES:
        return "advisory"
    message = f"Validation code has no shared severity policy: {code!r}."
    raise ValueError(message)


def diagnostic_record(
    code: str,
    *,
    severity: AdvisorySeverity,
    stage: str,
    message: str,
    metrics: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    source_artifacts: Sequence[str] = (),
    recorded_at: str | None = None,
    quality_flag: bool | None = None,
) -> DiagnosticRecord:
    """
    Construct one complete advisory diagnostic record.

    Parameters
    ----------
    code : str
        Stable advisory validation code.
    severity : AdvisorySeverity
        Human-facing advisory severity.
    stage : str
        Pipeline stage that observed the diagnostic.
    message : str
        Concise scientific or numerical observation.
    metrics : Mapping[str, Any] | None, optional
        Observed values supporting the diagnostic.
    thresholds : Mapping[str, Any] | None, optional
        Preferred tolerances or comparison thresholds.
    source_artifacts : Sequence[str], optional
        Storage-relative or case-relative source artifact names.
    recorded_at : str | None, optional
        Timezone-aware timestamp. The current UTC time is used when omitted.
    quality_flag : bool | None, optional
        Whether the record is a Dataset-visible quality flag. Warnings are
        quality flags by default; informational records are not.

    Returns
    -------
    DiagnosticRecord
        JSON-ready advisory-evidence representation.

    Raises
    ------
    ValueError
        If the code is blocking or any required text field is empty.

    """
    if validation_disposition(code) != "advisory":
        message_text = f"Blocking validation code cannot be encoded as a diagnostic: {code!r}."
        raise ValueError(message_text)
    if severity not in {"info", "warning"}:
        message_text = f"Unsupported advisory severity: {severity!r}."
        raise ValueError(message_text)
    if not stage or not message:
        message_text = "Diagnostic stage and message must be non-empty."
        raise ValueError(message_text)
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    return {
        "code": code,
        "severity": severity,
        "stage": stage,
        "message": message,
        "metrics": dict(metrics or {}),
        "thresholds": dict(thresholds or {}),
        "source_artifacts": [str(path) for path in source_artifacts],
        "recorded_at": timestamp,
        "quality_flag": severity == "warning" if quality_flag is None else quality_flag,
    }
