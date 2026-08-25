"""
Classify transient artifact predictions without altering model output values.

This module owns analysis-only validity evidence for the raw scaled model
output, decoded physical increment, and reconstructed physical state. Source
Dataset, reference, scaling, and grid admission remain strict in their existing
owners.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Final, Literal, cast

import numpy as np

from src import common
from src.datasets.contracts import dataset_contracts_transient as transient_contract

# ruff: noqa: EM101, PLR2004, TRY003

PREDICTION_VALIDITY_SCHEMA_VERSION: Final = 1
VALID: Final = "VALID"
FINITE_BUT_PHYSICALLY_INVALID: Final = "FINITE_BUT_PHYSICALLY_INVALID"
NONFINITE: Final = "NONFINITE"
PredictionValidityStatus = Literal[
    "VALID",
    "FINITE_BUT_PHYSICALLY_INVALID",
    "NONFINITE",
]
PREDICTION_VALIDITY_STATUSES: Final[tuple[PredictionValidityStatus, ...]] = (
    VALID,
    FINITE_BUT_PHYSICALLY_INVALID,
    NONFINITE,
)
STATE_ORDER: Final = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
TEMPERATURE_RANGE_K: Final = (0.0, 2_000.0)
_STATUS_PRECEDENCE: Final = {
    VALID: 0,
    FINITE_BUT_PHYSICALLY_INVALID: 1,
    NONFINITE: 2,
}
_STAGE_NAMES: Final = (
    "raw_scaled_model_output",
    "decoded_physical_increment",
    "reconstructed_state",
)
_MINIMUM_PREDICTION_RANK: Final = 3


def _prediction_array(value: Any, *, label: str) -> np.ndarray:
    """Admit one real prediction-stage array while retaining IEEE values."""
    array = np.asarray(value)
    if (
        array.dtype.kind != "f"
        or array.ndim != _MINIMUM_PREDICTION_RANK + 1
        or array.shape[1] != len(STATE_ORDER)
        or min(array.shape[0], array.shape[-2], array.shape[-1]) < 1
    ):
        message = f"{label} must be one real [L,4,Y,X] array."
        raise ValueError(message)
    return np.ascontiguousarray(array)


def _availability(value: Any, *, length: int) -> np.ndarray:
    """Admit one non-empty computed-prefix mask."""
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != (length,) or not bool(array.any()):
        message = "prediction_available must be one non-empty boolean [L] mask."
        raise ValueError(message)
    computed = int(array.sum())
    expected = np.arange(length) < computed
    if not np.array_equal(array, expected):
        message = "prediction_available must identify one exact computed prefix."
        raise ValueError(message)
    return np.ascontiguousarray(array)


def _times(value: Any, *, length: int) -> np.ndarray:
    """Admit exact finite ascending state times."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length + 1,) or not np.isfinite(array).all() or not bool(np.all(np.diff(array) > 0.0)):
        message = "prediction validity requires finite ascending [L+1] physical times."
        raise ValueError(message)
    return np.ascontiguousarray(array)


def prediction_physical_invalid_mask(states: Any) -> np.ndarray:
    """Return finite reconstructed values outside maintained field domains."""
    array = _prediction_array(states, label="reconstructed_states")
    finite = np.isfinite(array)
    invalid = np.zeros(array.shape, dtype=bool)
    lower, upper = TEMPERATURE_RANGE_K
    invalid[:, 0] = finite[:, 0] & ((array[:, 0] < lower) | (array[:, 0] > upper))
    invalid[:, 1] = finite[:, 1] & ((array[:, 1] < 0.0) | (array[:, 1] > 1.0))
    invalid[:, 2:] = finite[:, 2:] & (array[:, 2:] < 0.0)
    return invalid


def _channel_counts(
    array: np.ndarray,
    available: np.ndarray,
    *,
    physical_invalid: np.ndarray | None,
) -> dict[str, dict[str, int]]:
    """Count exact computed values per channel without filtering."""
    counts: dict[str, dict[str, int]] = {}
    for channel_index, channel in enumerate(STATE_ORDER):
        values = array[available, channel_index]
        finite = np.isfinite(values)
        channel_counts = {
            "total_value_count": int(values.size),
            "finite_value_count": int(finite.sum()),
            "nonfinite_value_count": int((~finite).sum()),
            "nan_count": int(np.isnan(values).sum()),
            "positive_infinity_count": int(np.isposinf(values).sum()),
            "negative_infinity_count": int(np.isneginf(values).sum()),
        }
        if physical_invalid is not None:
            channel_counts["physically_invalid_finite_count"] = int(physical_invalid[available, channel_index].sum())
        counts[channel] = channel_counts
    return counts


def _first_invalid(
    *,
    stages: Sequence[tuple[str, np.ndarray]],
    physical_invalid: np.ndarray,
    available: np.ndarray,
    times: np.ndarray,
    mode: str,
    origin_index: int,
) -> dict[str, Any] | None:
    """Locate the first chronological raw or physical invalid value."""
    for step in range(len(available)):
        if not available[step]:
            break
        for stage_name, values in stages:
            locations = np.argwhere(~np.isfinite(values[step]))
            if locations.size:
                channel, y_index, x_index = (int(item) for item in locations[0])
                return {
                    "kind": "NONFINITE",
                    "stage": stage_name,
                    "mode": mode,
                    "origin_index": origin_index,
                    "rollout_step": step + 1,
                    "transition_index": origin_index + step,
                    "physical_time": float(times[step + 1]),
                    "channel": STATE_ORDER[channel],
                    "channel_index": channel,
                    "spatial_index": [y_index, x_index],
                }
        locations = np.argwhere(physical_invalid[step])
        if locations.size:
            channel, y_index, x_index = (int(item) for item in locations[0])
            return {
                "kind": "FINITE_BUT_PHYSICALLY_INVALID",
                "stage": "reconstructed_state",
                "mode": mode,
                "origin_index": origin_index,
                "rollout_step": step + 1,
                "transition_index": origin_index + step,
                "physical_time": float(times[step + 1]),
                "channel": STATE_ORDER[channel],
                "channel_index": channel,
                "spatial_index": [y_index, x_index],
            }
    return None


def build_prediction_validity(
    *,
    scaled_model_outputs: Any,
    decoded_physical_increments: Any,
    reconstructed_states: Any,
    prediction_available: Any,
    physical_times: Any,
    mode: str,
    origin_index: int,
) -> dict[str, Any]:
    """Build exact JSON-safe prediction diagnostics for one sequence record."""
    if not isinstance(mode, str) or not mode:
        raise TypeError("Prediction validity mode must be non-empty text.")
    if isinstance(origin_index, bool) or not isinstance(origin_index, Integral) or origin_index < 0:
        raise ValueError("Prediction validity origin_index must be non-negative.")
    scaled = _prediction_array(
        scaled_model_outputs,
        label="scaled_model_outputs",
    )
    decoded = _prediction_array(
        decoded_physical_increments,
        label="decoded_physical_increments",
    )
    states = _prediction_array(
        reconstructed_states,
        label="reconstructed_states",
    )
    if scaled.shape != decoded.shape or scaled.shape != states.shape:
        message = "Prediction validity stage arrays must have one exact shared shape."
        raise ValueError(message)
    available = _availability(prediction_available, length=scaled.shape[0])
    times = _times(physical_times, length=scaled.shape[0])
    physical_invalid = prediction_physical_invalid_mask(states)
    stages = (
        ("raw_scaled_model_output", scaled),
        ("decoded_physical_increment", decoded),
        ("reconstructed_state", states),
    )
    nonfinite_count = sum(int((~np.isfinite(values[available])).sum()) for _name, values in stages)
    physical_invalid_count = int(physical_invalid[available].sum())
    if nonfinite_count:
        status: PredictionValidityStatus = NONFINITE
    elif physical_invalid_count:
        status = FINITE_BUT_PHYSICALLY_INVALID
    else:
        status = VALID
    computed_steps = int(available.sum())
    uncomputed_steps = len(available) - computed_steps
    if uncomputed_steps and not nonfinite_count:
        message = "An uncomputed prediction tail requires preserved non-finite model evidence."
        raise ValueError(message)
    stage_counts = {
        name: {
            "physical_validity_defined": name == "reconstructed_state",
            "channels": _channel_counts(
                values,
                available,
                physical_invalid=(physical_invalid if name == "reconstructed_state" else None),
            ),
        }
        for name, values in stages
    }
    result = {
        "schema_version": PREDICTION_VALIDITY_SCHEMA_VERSION,
        "status": status,
        "count_scope": "all_evaluation_grid_cells_in_computed_model_outputs",
        "mode": mode,
        "origin_index": int(origin_index),
        "requested_step_count": len(available),
        "computed_step_count": computed_steps,
        "uncomputed_step_count": uncomputed_steps,
        "uncomputed_value_count": (uncomputed_steps * len(STATE_ORDER) * scaled.shape[-2] * scaled.shape[-1]),
        "channels": stage_counts["reconstructed_state"]["channels"],
        "stages": stage_counts,
        "first_invalid": _first_invalid(
            stages=stages,
            physical_invalid=physical_invalid,
            available=available,
            times=times,
            mode=mode,
            origin_index=int(origin_index),
        ),
        "physical_contract": {
            "T_kelvin_inclusive": list(TEMPERATURE_RANGE_K),
            "phi_inclusive": [0.0, 1.0],
            "w_surf_minimum_inclusive": 0.0,
            "w_int_minimum_inclusive": 0.0,
        },
    }
    common.serialization.canonical_json_sha256(result)
    return result


def validate_prediction_validity_document(  # noqa: C901, PLR0912
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit one JSON-safe record diagnostic without loading numerical payloads."""
    if not isinstance(value, Mapping):
        raise TypeError("prediction_validity must be one mapping.")
    document = dict(value)
    required = {
        "schema_version",
        "status",
        "count_scope",
        "mode",
        "origin_index",
        "requested_step_count",
        "computed_step_count",
        "uncomputed_step_count",
        "uncomputed_value_count",
        "channels",
        "stages",
        "first_invalid",
        "physical_contract",
    }
    if set(document) != required:
        raise ValueError("Prediction validity fields do not match the schema.")
    if (
        document["schema_version"] != PREDICTION_VALIDITY_SCHEMA_VERSION
        or document["status"] not in PREDICTION_VALIDITY_STATUSES
        or document["count_scope"] != "all_evaluation_grid_cells_in_computed_model_outputs"
        or not isinstance(document["mode"], str)
        or not document["mode"]
    ):
        raise ValueError("Prediction validity identity or status is invalid.")
    count_fields = (
        "origin_index",
        "requested_step_count",
        "computed_step_count",
        "uncomputed_step_count",
        "uncomputed_value_count",
    )
    if any(isinstance(document[name], bool) or not isinstance(document[name], Integral) or int(document[name]) < 0 for name in count_fields):
        raise ValueError("Prediction validity counts must be non-negative integers.")
    if document["requested_step_count"] != document["computed_step_count"] + document["uncomputed_step_count"] or document["computed_step_count"] < 1:
        raise ValueError("Prediction validity step counts are contradictory.")

    def admit_channels(
        raw: Any,
        *,
        physical_validity_defined: bool,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or set(raw) != set(STATE_ORDER):
            raise ValueError("Prediction validity channels do not match state order.")
        expected_fields = {
            "total_value_count",
            "finite_value_count",
            "nonfinite_value_count",
            "nan_count",
            "positive_infinity_count",
            "negative_infinity_count",
        }
        if physical_validity_defined:
            expected_fields.add("physically_invalid_finite_count")
        result: dict[str, Any] = {}
        for channel in STATE_ORDER:
            counts = raw[channel]
            if (
                not isinstance(counts, Mapping)
                or set(counts) != expected_fields
                or any(isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0 for item in counts.values())
            ):
                raise ValueError("Prediction validity channel counts are invalid.")
            if (
                counts["total_value_count"] != counts["finite_value_count"] + counts["nonfinite_value_count"]
                or counts["nonfinite_value_count"] != counts["nan_count"] + counts["positive_infinity_count"] + counts["negative_infinity_count"]
                or (physical_validity_defined and counts["physically_invalid_finite_count"] > counts["finite_value_count"])
            ):
                raise ValueError("Prediction validity channel count arithmetic is invalid.")
            result[channel] = dict(counts)
        return result

    stages = document["stages"]
    if not isinstance(stages, Mapping) or set(stages) != set(_STAGE_NAMES):
        raise ValueError("Prediction validity stages do not match the schema.")
    admitted_stages: dict[str, Any] = {}
    for stage_name in _STAGE_NAMES:
        stage = stages[stage_name]
        defined = stage_name == "reconstructed_state"
        if (
            not isinstance(stage, Mapping)
            or set(stage) != {"physical_validity_defined", "channels"}
            or stage["physical_validity_defined"] is not defined
        ):
            raise ValueError("Prediction validity stage evidence is invalid.")
        admitted_stages[stage_name] = {
            "physical_validity_defined": defined,
            "channels": admit_channels(
                stage["channels"],
                physical_validity_defined=defined,
            ),
        }
    channels = admit_channels(
        document["channels"],
        physical_validity_defined=True,
    )
    if channels != admitted_stages["reconstructed_state"]["channels"]:
        raise ValueError("Prediction validity reconstructed channel counts disagree.")
    total_value_counts = {int(counts["total_value_count"]) for stage in admitted_stages.values() for counts in stage["channels"].values()}
    computed_steps = int(document["computed_step_count"])
    if len(total_value_counts) != 1 or next(iter(total_value_counts)) < computed_steps or next(iter(total_value_counts)) % computed_steps:
        raise ValueError("Prediction validity stages do not share one exact computed grid support.")
    grid_value_count = next(iter(total_value_counts)) // computed_steps
    expected_uncomputed_values = int(document["uncomputed_step_count"]) * len(STATE_ORDER) * grid_value_count
    if document["uncomputed_value_count"] != expected_uncomputed_values:
        raise ValueError("Prediction validity uncomputed count contradicts its grid support.")
    total_nonfinite = sum(counts["nonfinite_value_count"] for stage in admitted_stages.values() for counts in stage["channels"].values())
    total_physical = sum(counts["physically_invalid_finite_count"] for counts in channels.values())
    if document["uncomputed_step_count"] and not total_nonfinite:
        raise ValueError("Prediction validity uncomputed tails require preserved non-finite evidence.")
    expected_status = NONFINITE if total_nonfinite else (FINITE_BUT_PHYSICALLY_INVALID if total_physical else VALID)
    if document["status"] != expected_status:
        raise ValueError("Prediction validity status contradicts its counts.")
    first = document["first_invalid"]
    if expected_status == VALID:
        if first is not None:
            raise ValueError("Valid prediction evidence cannot identify an invalid value.")
    elif (
        not isinstance(first, Mapping)
        or set(first)
        != {
            "kind",
            "stage",
            "mode",
            "origin_index",
            "rollout_step",
            "transition_index",
            "physical_time",
            "channel",
            "channel_index",
            "spatial_index",
        }
        or first["kind"] not in {NONFINITE, FINITE_BUT_PHYSICALLY_INVALID}
        or first["stage"] not in _STAGE_NAMES
        or first["mode"] != document["mode"]
        or first["origin_index"] != document["origin_index"]
        or first["channel"] not in STATE_ORDER
        or first["channel_index"] != STATE_ORDER.index(first["channel"])
        or not isinstance(first["spatial_index"], list)
        or len(first["spatial_index"]) != 2
        or any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in first["spatial_index"])
        or isinstance(first["physical_time"], bool)
        or not isinstance(first["physical_time"], Real)
        or not np.isfinite(float(first["physical_time"]))
    ):
        raise ValueError("Prediction validity first-invalid evidence is invalid.")
    if isinstance(first, Mapping):
        rollout_step = first["rollout_step"]
        transition_index = first["transition_index"]
        if (
            isinstance(rollout_step, bool)
            or not isinstance(rollout_step, Integral)
            or not 1 <= int(rollout_step) <= computed_steps
            or isinstance(transition_index, bool)
            or not isinstance(transition_index, Integral)
            or int(transition_index) != int(document["origin_index"]) + int(rollout_step) - 1
        ):
            raise ValueError("Prediction validity first-invalid sequence coordinates are invalid.")
        stage_counts = admitted_stages[str(first["stage"])]["channels"][str(first["channel"])]
        if first["kind"] == NONFINITE:
            if stage_counts["nonfinite_value_count"] < 1:
                raise ValueError("Prediction validity first non-finite value lacks counted evidence.")
        elif first["stage"] != "reconstructed_state" or channels[str(first["channel"])]["physically_invalid_finite_count"] < 1:
            raise ValueError("Prediction validity first physical invalidity lacks counted evidence.")
    contract = document["physical_contract"]
    expected_contract = {
        "T_kelvin_inclusive": list(TEMPERATURE_RANGE_K),
        "phi_inclusive": [0.0, 1.0],
        "w_surf_minimum_inclusive": 0.0,
        "w_int_minimum_inclusive": 0.0,
    }
    if contract != expected_contract:
        raise ValueError("Prediction validity physical contract is invalid.")
    common.serialization.canonical_json_sha256(document)
    document["channels"] = channels
    document["stages"] = admitted_stages
    document["first_invalid"] = None if first is None else dict(first)
    return document


def validate_prediction_validity(
    value: Mapping[str, Any],
    *,
    scaled_model_outputs: Any,
    decoded_physical_increments: Any,
    reconstructed_states: Any,
    prediction_available: Any,
    physical_times: Any,
    mode: str,
    origin_index: int,
) -> dict[str, Any]:
    """Recompute and admit prediction diagnostics against persisted arrays."""
    observed = validate_prediction_validity_document(value)
    expected = build_prediction_validity(
        scaled_model_outputs=scaled_model_outputs,
        decoded_physical_increments=decoded_physical_increments,
        reconstructed_states=reconstructed_states,
        prediction_available=prediction_available,
        physical_times=physical_times,
        mode=mode,
        origin_index=origin_index,
    )
    if observed != expected:
        message = "Prediction validity evidence contradicts persisted raw arrays."
        raise ValueError(message)
    return expected


def aggregate_case_prediction_validity(
    *,
    case_id: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate unique-chain record diagnostics into one exact case summary."""
    if not isinstance(case_id, str) or not case_id or not records:
        raise ValueError("Case prediction validity requires an ID and records.")
    admitted = [validate_prediction_validity_document(record) for record in records]
    status = cast(
        "PredictionValidityStatus",
        max(
            (str(value["status"]) for value in admitted),
            key=_STATUS_PRECEDENCE.__getitem__,
        ),
    )

    def sum_channels(
        source: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for channel in STATE_ORDER:
            channel_values = [cast("Mapping[str, Any]", item[channel]) for item in source]
            fields = set(channel_values[0])
            if any(set(item) != fields for item in channel_values):
                raise ValueError("Prediction validity channel count fields disagree.")
            result[channel] = {field: sum(int(item[field]) for item in channel_values) for field in sorted(fields)}
        return result

    stage_summary: dict[str, Any] = {}
    for stage in _STAGE_NAMES:
        values = [cast("Mapping[str, Any]", item["stages"])[stage] for item in admitted]
        defined = values[0]["physical_validity_defined"]
        if any(item["physical_validity_defined"] is not defined for item in values):
            raise ValueError("Prediction validity stage definitions disagree.")
        stage_summary[stage] = {
            "physical_validity_defined": defined,
            "channels": sum_channels([cast("Mapping[str, Any]", item["channels"]) for item in values]),
        }
    invalid = [cast("Mapping[str, Any]", value["first_invalid"]) for value in admitted if value["first_invalid"] is not None]
    first_invalid = (
        None
        if not invalid
        else dict(
            min(
                invalid,
                key=lambda item: (
                    float(item["physical_time"]),
                    int(item["transition_index"]),
                    int(item["channel_index"]),
                    tuple(int(index) for index in item["spatial_index"]),
                ),
            )
        )
    )
    result = {
        "schema_version": PREDICTION_VALIDITY_SCHEMA_VERSION,
        "case_id": case_id,
        "status": status,
        "unique_prediction_chain_count": len(admitted),
        "requested_step_count": sum(int(value["requested_step_count"]) for value in admitted),
        "computed_step_count": sum(int(value["computed_step_count"]) for value in admitted),
        "uncomputed_step_count": sum(int(value["uncomputed_step_count"]) for value in admitted),
        "uncomputed_value_count": sum(int(value["uncomputed_value_count"]) for value in admitted),
        "channels": sum_channels([cast("Mapping[str, Any]", value["channels"]) for value in admitted]),
        "stages": stage_summary,
        "first_invalid": first_invalid,
    }
    common.serialization.canonical_json_sha256(result)
    return result
