"""Own the prediction-validity vocabulary, count format and warnings both tasks share.

Responsibilities:
  - Define statuses, schema version and exact per-channel value counts
  - Classify one steady-flow field prediction as VALID or NONFINITE and admit its document
  - Summarize field documents as metric support counts and role-level status counts, also from a steady artifact table
  - Print the invalid-prediction warnings and the ``valid k of n`` note for both tasks
  - Format an unavailable metric with its support counts and drop non-finite cases from per-case values

Design principles:
  - Values are never clamped, filtered or imputed; documents keep counts and the first invalid coordinate
  - Steady flow defines no physical range, so it has no FINITE_BUT_PHYSICALLY_INVALID class
  - Metrics that need complete finite support report counts instead of a value

This module does NOT:
  - Define transient stages, rollout coordinates or the transient physical range
  - Compute error metrics, open NPZ payloads or publish artifacts
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from src import common

if TYPE_CHECKING:
    import pandas as pd

PREDICTION_VALIDITY_SCHEMA_VERSION: Final = 1
VALID: Final = "VALID"
FINITE_BUT_PHYSICALLY_INVALID: Final = "FINITE_BUT_PHYSICALLY_INVALID"
NONFINITE: Final = "NONFINITE"
PredictionValidityStatus = Literal["VALID", "FINITE_BUT_PHYSICALLY_INVALID", "NONFINITE"]
PREDICTION_VALIDITY_STATUSES: Final[tuple[PredictionValidityStatus, ...]] = (VALID, FINITE_BUT_PHYSICALLY_INVALID, NONFINITE)
SUPPORT_COUNT_FIELDS: Final = ("required_value_count", "computed_value_count", "finite_value_count", "nonfinite_value_count")
_COUNT_FIELDS: Final = frozenset(
    {"total_value_count", "finite_value_count", "nonfinite_value_count", "nan_count", "positive_infinity_count", "negative_infinity_count"}
)
_FIELD_DOCUMENT_FIELDS: Final = frozenset({"schema_version", "status", "count_scope", "channels", "first_invalid"})
_FIELD_COUNT_SCOPE: Final = "all_evaluation_grid_cells_in_physical_model_output"
_FIELD_PREDICTION_RANK: Final = 3


def require_schema_version(version: Any) -> None:
    """Reject superseded or unknown validity schema versions."""
    if not common.serialization.is_schema_version(version, PREDICTION_VALIDITY_SCHEMA_VERSION):
        msg = f"Prediction validity schema_version {version!r} is unsupported; expected {PREDICTION_VALIDITY_SCHEMA_VERSION}."
        raise ValueError(msg)


def channel_counts(values: np.ndarray, channels: Sequence[str], *, physical_invalid: np.ndarray | None = None) -> dict[str, dict[str, int]]:
    """Count exact values per channel along axis 1 of ``values`` without filtering."""
    counts: dict[str, dict[str, int]] = {}
    for channel_index, channel in enumerate(channels):
        selected = values[:, channel_index]
        finite = np.isfinite(selected)
        counts[channel] = {
            "total_value_count": int(selected.size),
            "finite_value_count": int(finite.sum()),
            "nonfinite_value_count": int((~finite).sum()),
            "nan_count": int(np.isnan(selected).sum()),
            "positive_infinity_count": int(np.isposinf(selected).sum()),
            "negative_infinity_count": int(np.isneginf(selected).sum()),
        }
        if physical_invalid is not None:
            counts[channel]["physically_invalid_finite_count"] = int(physical_invalid[:, channel_index].sum())
    return counts


def admit_channel_counts(raw: Any, channels: Sequence[str], *, physical_validity_defined: bool) -> dict[str, dict[str, int]]:
    """Admit persisted per-channel counts whose arithmetic is exact."""
    if not isinstance(raw, Mapping) or set(raw) != set(channels):
        msg = "Prediction validity channels do not match the declared channels."
        raise ValueError(msg)
    expected_fields = _COUNT_FIELDS | ({"physically_invalid_finite_count"} if physical_validity_defined else set())
    result: dict[str, dict[str, int]] = {}
    for channel in channels:
        counts = raw[channel]
        if (
            not isinstance(counts, Mapping)
            or set(counts) != expected_fields
            or any(isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0 for item in counts.values())
        ):
            msg = "Prediction validity channel counts are invalid."
            raise ValueError(msg)
        if (
            counts["total_value_count"] != counts["finite_value_count"] + counts["nonfinite_value_count"]
            or counts["nonfinite_value_count"] != counts["nan_count"] + counts["positive_infinity_count"] + counts["negative_infinity_count"]
            or (physical_validity_defined and counts["physically_invalid_finite_count"] > counts["finite_value_count"])
        ):
            msg = "Prediction validity channel count arithmetic is invalid."
            raise ValueError(msg)
        result[channel] = dict(counts)
    return result


def build_field_prediction_validity(prediction: Any, *, channels: Sequence[str]) -> dict[str, Any]:
    """Classify one physical ``[C,Y,X]`` field prediction as VALID or NONFINITE without changing it."""
    array = np.asarray(prediction)
    if array.dtype.kind != "f" or array.ndim != _FIELD_PREDICTION_RANK or array.shape[0] != len(channels) or array.size == 0:
        msg = "Field prediction validity requires one non-empty real [C,Y,X] array matching its channels."
        raise ValueError(msg)
    locations = np.argwhere(~np.isfinite(array))
    first_invalid = None
    if locations.size:
        channel_index, y_index, x_index = (int(item) for item in locations[0])
        first_invalid = {"kind": NONFINITE, "channel": channels[channel_index], "channel_index": channel_index, "spatial_index": [y_index, x_index]}
    return {
        "schema_version": PREDICTION_VALIDITY_SCHEMA_VERSION,
        "status": VALID if first_invalid is None else NONFINITE,
        "count_scope": _FIELD_COUNT_SCOPE,
        "channels": channel_counts(array[None], channels),
        "first_invalid": first_invalid,
    }


def admit_field_prediction_validity(text: Any, *, channels: Sequence[str]) -> dict[str, Any]:
    """Admit one persisted field document whose status and first invalid value agree with its counts."""
    document = json.loads(text) if isinstance(text, str) else None
    if not isinstance(document, dict) or set(document) != _FIELD_DOCUMENT_FIELDS:
        msg = "Field prediction validity must be one JSON document with the current fields."
        raise ValueError(msg)
    require_schema_version(document["schema_version"])
    counts = admit_channel_counts(document["channels"], channels, physical_validity_defined=False)
    nonfinite = sum(item["nonfinite_value_count"] for item in counts.values())
    first = document["first_invalid"]
    if (
        document["count_scope"] != _FIELD_COUNT_SCOPE
        or len({item["total_value_count"] for item in counts.values()}) != 1
        or document["status"] != (NONFINITE if nonfinite else VALID)
        or (first is None) != (not nonfinite)
    ):
        msg = "Field prediction validity status contradicts its counts."
        raise ValueError(msg)
    if first is not None and (
        not isinstance(first, Mapping)
        or set(first) != {"kind", "channel", "channel_index", "spatial_index"}
        or first["kind"] != NONFINITE
        or first["channel"] not in channels
        or first["channel_index"] != list(channels).index(first["channel"])
        or counts[first["channel"]]["nonfinite_value_count"] < 1
        or not isinstance(first["spatial_index"], list)
        or len(first["spatial_index"]) != _FIELD_PREDICTION_RANK - 1
        or any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in first["spatial_index"])
    ):
        msg = "Field prediction validity first-invalid evidence is invalid."
        raise ValueError(msg)
    return {**document, "channels": counts}


def support_evidence(*, required: int, computed: int, finite: int) -> dict[str, Any]:
    """Return the complete-finite-support evidence of one metric from its required, computed and finite value counts."""
    if computed < required:
        reason = "full_prediction_support_unavailable_after_nonfinite_output"
    elif finite < computed:
        reason = "full_prediction_support_contains_nonfinite_values"
    else:
        reason = None
    return {
        "classification": "REQUIRES_ALL_FINITE_VALUES",
        "available": reason is None,
        "unavailable_reason": reason,
        "required_value_count": required,
        "computed_value_count": computed,
        "finite_value_count": finite,
        "nonfinite_value_count": computed - finite,
    }


def unavailable_text(evidence: Mapping[str, Any]) -> str:
    """Return how both tasks show an unavailable metric: its support counts, never a number and never zero."""
    return f"unavailable ({evidence['finite_value_count']} finite of {evidence['required_value_count']} required values)"


def finite_case_mask(values: Any) -> np.ndarray:
    """Return which cases (first axis) per-case views keep: a case with any non-finite value is left out, never imputed."""
    array = np.asarray(values, dtype=float)
    return np.asarray(np.isfinite(array).reshape(array.shape[0], -1).all(axis=1), dtype=bool)


def finite_case_values(values: Any) -> np.ndarray:
    """Return per-case values without the non-finite cases, which the invalid-prediction warning names; nothing is imputed."""
    array = np.asarray(values, dtype=float)
    return array[finite_case_mask(array)]


def support_counts(documents: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the complete-finite-support evidence of a metric over every value of ``documents``."""
    counts = [item for document in documents for item in document["channels"].values()]
    required = sum(item["total_value_count"] for item in counts)
    return support_evidence(required=required, computed=required, finite=sum(item["finite_value_count"] for item in counts))


def role_prediction_validity(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return role-level status counts and per-case documents keyed by package case ID."""
    return {
        "schema_version": PREDICTION_VALIDITY_SCHEMA_VERSION,
        "case_count": len(cases),
        "status_counts": {status: sum(case["status"] == status for case in cases.values()) for status in PREDICTION_VALIDITY_STATUSES},
        "cases": {case_id: dict(cases[case_id]) for case_id in sorted(cases)},
    }


def table_role_prediction_validity(frame: pd.DataFrame) -> dict[str, Any]:
    """Return role-level prediction validity of an admitted steady table, keyed by the package case ID that names each row's NPZ."""
    return role_prediction_validity(
        {Path(str(path)).stem: json.loads(str(text)) for path, text in zip(frame["npz_path"], frame["prediction_validity"], strict=True)}
    )


def report_invalid_predictions(prediction_validity: Mapping[str, Any]) -> None:
    """Print the invalid-prediction warning and one line per affected case; print nothing when every case is valid."""
    counts = prediction_validity["status_counts"]
    if counts[VALID] == prediction_validity["case_count"]:
        return
    print(
        f"[WARNING] artifact completed with invalid predictions | valid={counts[VALID]}"
        f" | finite_but_physically_invalid={counts[FINITE_BUT_PHYSICALLY_INVALID]} | nonfinite={counts[NONFINITE]}",
        flush=True,
    )
    for case_id, case in sorted(prediction_validity["cases"].items()):
        if case["status"] == VALID:
            continue
        channels = case["channels"]
        affected = [name for name, item in channels.items() if item["nonfinite_value_count"] or item.get("physically_invalid_finite_count", 0)]
        first = case["first_invalid"]
        location = (
            f"first_step={first['rollout_step']} | first_time={first['physical_time']}"
            if "rollout_step" in first
            else f"first_channel={first['channel']} | first_index={first['spatial_index']}"
        )
        line = f"[WARNING] invalid prediction case | case={case_id} | status={case['status']} | channels={','.join(affected)} | {location}"
        print(line, flush=True)


def invalid_prediction_note(provenances: Iterable[Mapping[str, Any]]) -> str | None:
    """Return ``valid V of N`` summed over role provenances when any prediction is invalid, else ``None``."""
    evidence = [provenance["evaluation"]["prediction_validity"] for provenance in provenances]
    valid = sum(item["status_counts"][VALID] for item in evidence)
    total = sum(item["case_count"] for item in evidence)
    return f"valid {valid} of {total}" if valid < total else None
