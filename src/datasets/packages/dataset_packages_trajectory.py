"""
===============================================================================
dataset_packages_trajectory.py
===============================================================================
Build and admit compact transient trajectory indexes from canonical HDF5 cases.
Responsibilities:
  - Validate source time, channel, status, and case identities
  - Publish portable case-level membership and deterministic transition indexes
  - Admit current index schema, contract, and digest evidence
Design principles:
  - Absolute HDF5 states remain canonical and indexes store only references
  - Every transition inherits one immutable case-level membership
  - Exact-stop diagnostic states never enter learned transitions
This module does NOT:
  - Own process-local HDF5 handles or Dataset item materialization
  - Fit normalization, train models, or perform rollouts
===============================================================================
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np

from src import common, generation
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views

if TYPE_CHECKING:
    from collections.abc import Sequence

TRANSIENT_INDEX_SCHEMA_KIND: Final = "vp2_transient_transition_index"
TRANSIENT_INDEX_SCHEMA_VERSION: Final = 1
_SOURCE_PROFILE: Final = generation.contracts.get_profile_contract(
    transient_contract.TRANSIENT_PROFILE_ID,
)
_SOURCE_TRANSIENT_FIELDS: Final = tuple(field.name for field in _SOURCE_PROFILE.transient_fields)
_SOURCE_SCHEDULE_FIELDS: Final = tuple(field.name for field in _SOURCE_PROFILE.schedule_fields)
_SOURCE_SCALAR_FIELDS: Final = tuple(field.name for field in _SOURCE_PROFILE.scalar_inputs)
_MINIMUM_TRANSIENT_STATE_COUNT: Final = 2


class TransientDataContractError(ValueError):
    """Report one actionable transient package or runtime contract violation."""


@dataclass(frozen=True, slots=True)
class TransientSourceCase:
    """Bind one admitted canonical source case to package-owned metadata."""

    path: Path
    package_case_id: str
    source_batch_id: str
    membership: str
    evaluation_regime: str
    expected_sha256: str
    expected_case_input_id: str
    expected_simulation_case_id: str
    material_family: str
    ood_group: str | None
    ood_parameters: tuple[str, ...]
    ood_evidence: dict[str, Any]


def decode_json_string_list(value: Any, *, label: str) -> list[str]:
    """Decode one HDF5 JSON string-list attribute."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        message = f"HDF5 attribute {label!r} must contain JSON text."
        raise TypeError(message)
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        message = f"HDF5 attribute {label!r} must contain a string list."
        raise TypeError(message)
    return decoded


def _text_attribute(value: Any, *, label: str) -> str:
    """Return one required non-empty HDF5 text attribute."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        message = f"HDF5 attribute {label!r} must be non-empty text."
        raise TypeError(message)
    return value


def require_hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required HDF5 dataset with an exact contextual error."""
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        message = f"Transient HDF5 entry {name!r} must be a dataset: {handle.filename}."
        raise TransientDataContractError(message)
    return value


def _scientific_config(
    handle: h5py.File,
    *,
    path: Path,
) -> Mapping[str, Any]:
    """Admit the embedded scientific config against its source identity."""
    dataset = require_hdf5_dataset(handle, "provenance/scientific_config_json")
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        message = f"Transient scientific config provenance must be JSON text: {path}."
        raise TransientDataContractError(message)
    try:
        scientific = json.loads(value)
    except json.JSONDecodeError as error:
        message = f"Transient scientific config provenance is invalid JSON: {path}."
        raise TransientDataContractError(message) from error
    digest = _text_attribute(handle.attrs.get("scientific_config_digest"), label="scientific_config_digest")
    if not isinstance(scientific, dict) or common.serialization.canonical_json_sha256(scientific) != digest:
        message = f"Transient scientific config provenance disagrees with its HDF5 identity: {path}."
        raise TransientDataContractError(message)
    return scientific


def _configured_time_number(value: Any, *, label: str, path: Path) -> float:
    """Return one finite non-boolean configured temporal number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"Transient configured time value {label!r} is invalid: {path}."
        raise TransientDataContractError(message)
    return float(value)


def configured_regular_horizon(
    handle: h5py.File,
    time: np.ndarray,
    *,
    path: Path,
    tolerance: float,
) -> float:
    """Return and validate the configured regular-time horizon from provenance."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        message = f"Transient time classification tolerance is invalid: {path}."
        raise TransientDataContractError(message)
    scientific = _scientific_config(handle, path=path)
    time_config = scientific.get("time")
    if not isinstance(time_config, Mapping):
        message = f"Transient scientific config lacks one time mapping: {path}."
        raise TransientDataContractError(message)
    start = _configured_time_number(time_config.get("start"), label="start", path=path)
    horizon = _configured_time_number(time_config.get("stop"), label="stop", path=path)
    interval = _configured_time_number(time_config.get("interval"), label="interval", path=path)
    regular_values = time_config.get("regular_times")
    if (
        not math.isclose(start, 0.0, rel_tol=0.0, abs_tol=tolerance)
        or horizon <= 0.0
        or interval <= 0.0
        or not isinstance(regular_values, list)
        or not regular_values
    ):
        message = f"Transient configured regular-time axis is invalid: {path}."
        raise TransientDataContractError(message)
    configured = np.asarray(
        [_configured_time_number(value, label="regular_times", path=path) for value in regular_values],
        dtype=np.float64,
    )
    expected = np.arange(configured.size, dtype=np.float64) * interval
    if (
        time.size > configured.size
        or not np.allclose(configured, expected, rtol=0.0, atol=tolerance)
        or not math.isclose(float(configured[-1]), horizon, rel_tol=0.0, abs_tol=tolerance)
        or not np.allclose(time, configured[: time.size], rtol=0.0, atol=tolerance)
    ):
        message = f"Transient HDF5 regular time is not a prefix of its configured horizon: {path}."
        raise TransientDataContractError(message)
    return horizon


def _regular_time_contract(
    time: np.ndarray,
    *,
    path: Path,
    tolerance: float,
) -> tuple[int, int]:
    """Return state count and source stride for the learned transition step."""
    if time.ndim != 1 or time.size < 1 or not np.isfinite(time).all():
        message = f"Transient regular time must be one non-empty finite sequence: {path}."
        raise TransientDataContractError(message)
    if not math.isclose(float(time[0]), 0.0, rel_tol=0.0, abs_tol=tolerance):
        message = f"Transient regular time must start at zero: {path}."
        raise TransientDataContractError(message)
    if time.size == 1:
        return 1, 1
    deltas = np.diff(time)
    source_step = float(deltas[0])
    expected = np.arange(time.size, dtype=np.float64) * source_step
    if (
        source_step <= 0.0
        or not np.allclose(deltas, source_step, rtol=0.0, atol=tolerance)
        or not np.allclose(
            time,
            expected,
            rtol=0.0,
            atol=tolerance,
        )
    ):
        message = f"Transient HDF5 time states must form one positive regular prefix: {path}."
        raise TransientDataContractError(message)
    learned_step = float(transient_contract.TRANSIENT_STEP_CONTRACT.time_step)
    stride = round(learned_step / source_step)
    if stride < 1 or not math.isclose(stride * source_step, learned_step, rel_tol=0.0, abs_tol=tolerance):
        message = f"Transient source step {source_step} h must evenly divide the learned {learned_step} h transition step: {path}."
        raise TransientDataContractError(message)
    return int(time.size), stride


def _status_evidence(
    path: Path,
    time: np.ndarray,
    exact_stop: float | None,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Validate an adjacent canonical status sidecar when it is available."""
    status_path = path.with_name("status.json")
    last_regular = float(time[-1])
    expected_stop = last_regular if exact_stop is None else exact_stop
    if not status_path.exists():
        return {
            "status_sha256": None,
            "t_stop_exact": expected_stop,
            "t_last_regular": last_regular,
        }
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Transient status sidecar is unreadable: {status_path}."
        raise TransientDataContractError(message) from error
    if (
        not isinstance(status, dict)
        or status.get("schema_kind") != "simulation_case_status"
        or status.get("schema_version") != generation.publication.storage.STATUS_SCHEMA_VERSION
    ):
        message = f"Transient status sidecar schema is invalid: {status_path}."
        raise TransientDataContractError(message)
    if (
        status.get("solver_success") is not True
        or status.get("contains_nan_or_inf") is not False
        or status.get("field_shape_valid") is not True
        or status.get("schedule_valid") is not True
        or status.get("n_regular_states") != time.size
        or status.get("has_exact_stop_state") is not (exact_stop is not None)
        or not math.isclose(float(status.get("t_last_regular", math.nan)), last_regular, rel_tol=0.0, abs_tol=tolerance)
        or not math.isclose(float(status.get("t_stop_exact", math.nan)), expected_stop, rel_tol=0.0, abs_tol=tolerance)
    ):
        message = f"Transient status sidecar disagrees with its regular/exact-stop HDF5 sequence: {status_path}."
        raise TransientDataContractError(message)
    observed_exact = status.get("exact_stop_state_time")
    if exact_stop is None:
        if observed_exact is not None:
            message = f"Transient status records an unexpected exact-stop state: {status_path}."
            raise TransientDataContractError(message)
    elif (
        isinstance(observed_exact, bool)
        or not isinstance(observed_exact, (int, float))
        or not math.isclose(float(observed_exact), exact_stop, rel_tol=0.0, abs_tol=tolerance)
    ):
        message = f"Transient exact-stop status disagrees with HDF5: {status_path}."
        raise TransientDataContractError(message)
    return {
        "status_sha256": common.serialization.file_sha256(status_path),
        "t_stop_exact": expected_stop,
        "t_last_regular": last_regular,
    }


def _safe_relative_source(path: Path, source_root: Path) -> str:
    """Return one portable source locator below the authoritative storage root."""
    resolved = path.expanduser().resolve()
    root = source_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        message = f"Transient source {resolved} is outside storage root {root}."
        raise TransientDataContractError(message) from error
    if resolved.is_symlink() or not resolved.is_file():
        message = f"Transient source is missing or unsafe: {resolved}."
        raise TransientDataContractError(message)
    return relative.as_posix()


def _validate_source_metadata(source: TransientSourceCase, handle: h5py.File) -> str:
    """Require source-bound identities and return its simulation profile."""
    profile = _text_attribute(handle.attrs.get("simulation_profile"), label="simulation_profile")
    observed = {
        "case_input_id": _text_attribute(handle.attrs.get("case_input_id"), label="case_input_id"),
        "simulation_case_id": _text_attribute(handle.attrs.get("simulation_case_id"), label="simulation_case_id"),
        "material_family": _text_attribute(handle.attrs.get("material_family"), label="material_family"),
    }
    expected = {
        "case_input_id": source.expected_case_input_id,
        "simulation_case_id": source.expected_simulation_case_id,
        "material_family": source.material_family,
    }
    if profile != transient_contract.TRANSIENT_PROFILE_ID or observed != expected:
        message = f"Transient source identity disagrees with package admission: {source.path}."
        raise TransientDataContractError(message)
    return profile


def _case_record(
    source: TransientSourceCase,
    *,
    source_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """Validate one source and return case, transition, and horizon records."""
    path = source.path.expanduser().resolve()
    if source.evaluation_regime not in views.PACKAGE_REGIMES:
        message = f"Unsupported transient evaluation regime: {source.evaluation_regime!r}."
        raise TransientDataContractError(message)
    valid_memberships = (*views.ID_MEMBERSHIPS, views.TECHNICAL_SMOKE_MEMBERSHIP) if source.evaluation_regime == "id" else (source.evaluation_regime,)
    if source.membership not in valid_memberships:
        message = f"Membership {source.membership!r} is invalid for {source.evaluation_regime!r}."
        raise TransientDataContractError(message)
    observed_sha256 = common.serialization.file_sha256(path)
    if observed_sha256 != source.expected_sha256:
        message = f"Transient source content differs from package admission: {path}."
        raise TransientDataContractError(message)
    with h5py.File(path, "r") as handle:
        profile = _validate_source_metadata(source, handle)
        time_dataset = require_hdf5_dataset(handle, "time")
        time = np.asarray(time_dataset, dtype=np.float64)
        tolerance = float(time_dataset.attrs.get("classification_atol", math.nan))
        time_unit = _text_attribute(time_dataset.attrs.get("unit"), label="time.unit")
        if not math.isfinite(tolerance) or tolerance <= 0.0 or time_unit != transient_contract.TRANSIENT_STEP_CONTRACT.time_unit:
            message = f"Transient time tolerance or unit is invalid: {path}."
            raise TransientDataContractError(message)
        configured_horizon = configured_regular_horizon(
            handle,
            time,
            path=path,
            tolerance=tolerance,
        )
        regular_count, transition_stride = _regular_time_contract(
            time,
            path=path,
            tolerance=tolerance,
        )
        exact_stop_dataset = handle.get("exact_stop/time")
        exact_stop = None
        if exact_stop_dataset is not None:
            if not isinstance(exact_stop_dataset, h5py.Dataset):
                message = f"Transient exact-stop time must be a dataset: {path}."
                raise TransientDataContractError(message)
            exact_values = np.asarray(exact_stop_dataset, dtype=np.float64)
            exact_unit = _text_attribute(exact_stop_dataset.attrs.get("unit"), label="exact_stop.time.unit")
            if exact_values.shape != (1,) or exact_unit != transient_contract.TRANSIENT_STEP_CONTRACT.time_unit:
                message = f"Transient exact-stop time must contain one value in the canonical unit: {path}."
                raise TransientDataContractError(message)
            exact_stop = float(exact_values[0])
            if not math.isfinite(exact_stop) or exact_stop <= float(time[-1]) + tolerance or exact_stop > configured_horizon + tolerance:
                message = f"Transient exact-stop time must follow the regular prefix within the configured horizon: {path}."
                raise TransientDataContractError(message)
        transient_dataset = require_hdf5_dataset(handle, "transient/fields")
        static_dataset = require_hdf5_dataset(handle, "static/fields")
        schedule_dataset = require_hdf5_dataset(handle, "schedule/values")
        scalar_dataset = require_hdf5_dataset(handle, "scalar/values")
        transient_names = decode_json_string_list(transient_dataset.attrs["field_names"], label="transient.field_names")
        static_names = decode_json_string_list(static_dataset.attrs["field_names"], label="static.field_names")
        schedule_names = decode_json_string_list(schedule_dataset.attrs["field_names"], label="schedule.field_names")
        scalar_names = decode_json_string_list(scalar_dataset.attrs["field_names"], label="scalar.field_names")
        if transient_names != list(_SOURCE_TRANSIENT_FIELDS):
            message = f"Transient fields are not canonical: {path}."
            raise TransientDataContractError(message)
        required_static = {field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning}.difference({"x", "y"})
        if not required_static.issubset(static_names):
            message = f"Transient static conditioning is incomplete: {path}."
            raise TransientDataContractError(message)
        if schedule_names != list(_SOURCE_SCHEDULE_FIELDS) or scalar_names != list(_SOURCE_SCALAR_FIELDS):
            message = f"Transient boundary or scalar conditioning is not canonical: {path}."
            raise TransientDataContractError(message)
        schedule_time = np.asarray(schedule_dataset[:, schedule_names.index("t")], dtype=np.float64)
        samples: list[dict[str, Any]] = []
        for time_index in range(0, max(regular_count - transition_stride, 0), transition_stride):
            t_n = float(time[time_index])
            t_np1 = float(time[time_index + transition_stride])
            current = np.flatnonzero(np.isclose(schedule_time, t_n, rtol=0.0, atol=tolerance))
            following = np.flatnonzero(np.isclose(schedule_time, t_np1, rtol=0.0, atol=tolerance))
            if current.size != 1 or following.size != 1:
                message = f"Schedule lacks exact endpoints for transition {t_n} -> {t_np1} h: {path}."
                raise TransientDataContractError(message)
            samples.append(
                {
                    "sample_id": f"{source.package_case_id}__step_{time_index:04d}",
                    "time_index_n": time_index,
                    "time_index_n_plus_1": time_index + transition_stride,
                    "schedule_index_n": int(current[0]),
                    "schedule_index_n_plus_1": int(following[0]),
                }
            )
    if not samples:
        message = f"Transient source has no complete one-hour transition: {path}."
        raise TransientDataContractError(message)
    status = _status_evidence(path, time, exact_stop, tolerance=tolerance)
    record = {
        "package_case_id": source.package_case_id,
        "source_relative_path": _safe_relative_source(path, source_root),
        "source_hdf5_sha256": observed_sha256,
        "source_status_sha256": status["status_sha256"],
        "case_input_id": source.expected_case_input_id,
        "simulation_case_id": source.expected_simulation_case_id,
        "source_batch_id": source.source_batch_id,
        "source_simulation_profile": profile,
        "material_family": source.material_family,
        "evaluation_regime": source.evaluation_regime,
        "membership": source.membership,
        "ood_group": source.ood_group,
        "ood_parameters": list(source.ood_parameters),
        "ood_evidence": deepcopy(source.ood_evidence),
        "sequence_length": regular_count,
        "stored_state_count": int(time.size + (exact_stop is not None)),
        "transition_count": len(samples),
        "t_stop_exact": status["t_stop_exact"],
        "irregular_stop_time": exact_stop,
    }
    return record, samples, configured_horizon


def _index_digest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return transition identity without operational source locator paths."""
    result = {key: deepcopy(value) for key, value in payload.items() if key != "index_digest"}
    cases = result.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                case.pop("source_relative_path", None)
    return result


def build_transient_index(
    sources: Sequence[TransientSourceCase],
    destination: Path | str | None,
    *,
    dataset_name: str,
    dataset_id: str,
    evaluation_regime: str,
    source_root: Path | str,
) -> dict[str, Any]:
    """Build one deterministic compact transition index from canonical cases."""
    if not sources:
        message = "Transient index construction requires at least one source case."
        raise TransientDataContractError(message)
    if evaluation_regime not in views.PACKAGE_REGIMES or not dataset_name or not dataset_id:
        message = "Transient dataset name, ID, or evaluation regime is invalid."
        raise TransientDataContractError(message)
    root = Path(source_root)
    cases: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    simulation_ids: set[str] = set()
    sample_ids: set[str] = set()
    configured_horizon: float | None = None
    for case_index, source in enumerate(sources):
        record, case_samples, case_horizon = _case_record(source, source_root=root)
        if configured_horizon is None:
            configured_horizon = case_horizon
        elif not math.isclose(case_horizon, configured_horizon, rel_tol=0.0, abs_tol=1.0e-12):
            message = "Transient package sources must share one configured regular-time horizon."
            raise TransientDataContractError(message)
        simulation_id = str(record["simulation_case_id"])
        if simulation_id in simulation_ids:
            message = f"Transient source simulation identity is duplicated: {simulation_id}."
            raise TransientDataContractError(message)
        simulation_ids.add(simulation_id)
        cases.append(record)
        for sample in case_samples:
            sample_id = str(sample["sample_id"])
            if sample_id in sample_ids:
                message = f"Transient sample identity is duplicated: {sample_id}."
                raise TransientDataContractError(message)
            sample_ids.add(sample_id)
            samples.append({"case_index": case_index, **sample})
    payload: dict[str, Any] = {
        "schema_kind": TRANSIENT_INDEX_SCHEMA_KIND,
        "schema_version": TRANSIENT_INDEX_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "dataset_view": "transient_drying",
        "evaluation_regime": evaluation_regime,
        "contract_digest": transient_contract.transient_contract_digest(),
        "contract": transient_contract.transient_contract_payload(),
        "configured_regular_horizon": {
            "value": configured_horizon,
            "unit": transient_contract.TRANSIENT_STEP_CONTRACT.time_unit,
            "source": transient_contract.TRANSIENT_STEP_CONTRACT.temporal.configured_horizon_source,
        },
        "source_locator_root": "storage_root",
        "cases": cases,
        "samples": samples,
        "source_case_count": len(cases),
        "sample_count": len(samples),
        "transition_count": len(samples),
    }
    payload["index_digest"] = common.serialization.canonical_json_sha256(_index_digest_payload(payload))
    if destination is None:
        return payload
    destination_path = Path(destination).expanduser().resolve()
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if destination_path.exists():
        if not destination_path.is_file() or destination_path.read_text(encoding="utf-8") != serialized:
            message = f"Existing transient index conflicts with requested identity: {destination_path}."
            raise FileExistsError(message)
        return payload
    common.serialization.atomic_write_text(destination_path, serialized)
    return payload


def load_transient_index(path: Path) -> dict[str, Any]:
    """Load and strictly validate one current transition-index payload."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not read transient dataset index: {path}."
        raise TransientDataContractError(message) from error
    required = {
        "schema_kind",
        "schema_version",
        "dataset_name",
        "dataset_id",
        "dataset_view",
        "evaluation_regime",
        "contract_digest",
        "contract",
        "configured_regular_horizon",
        "source_locator_root",
        "cases",
        "samples",
        "source_case_count",
        "sample_count",
        "transition_count",
        "index_digest",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        message = f"Transient dataset index keys do not match the current schema: {path}."
        raise TransientDataContractError(message)
    horizon = payload["configured_regular_horizon"]
    if (
        payload["schema_kind"] != TRANSIENT_INDEX_SCHEMA_KIND
        or payload["schema_version"] != TRANSIENT_INDEX_SCHEMA_VERSION
        or payload["dataset_view"] != "transient_drying"
        or payload["evaluation_regime"] not in views.PACKAGE_REGIMES
        or not isinstance(payload["dataset_name"], str)
        or not payload["dataset_name"]
        or not isinstance(payload["dataset_id"], str)
        or not payload["dataset_id"]
        or payload["contract_digest"] != transient_contract.transient_contract_digest()
        or payload["contract"] != transient_contract.transient_contract_payload()
        or not isinstance(horizon, dict)
        or set(horizon) != {"value", "unit", "source"}
        or isinstance(horizon.get("value"), bool)
        or not isinstance(horizon.get("value"), (int, float))
        or not math.isfinite(float(horizon["value"]))
        or float(horizon["value"]) <= 0.0
        or horizon.get("unit") != transient_contract.TRANSIENT_STEP_CONTRACT.time_unit
        or horizon.get("source") != transient_contract.TRANSIENT_STEP_CONTRACT.temporal.configured_horizon_source
        or payload["source_locator_root"] != "storage_root"
        or not isinstance(payload["cases"], list)
        or not isinstance(payload["samples"], list)
        or isinstance(payload["source_case_count"], bool)
        or payload["source_case_count"] != len(payload["cases"])
        or isinstance(payload["sample_count"], bool)
        or payload["sample_count"] != len(payload["samples"])
        or isinstance(payload["transition_count"], bool)
        or payload["transition_count"] != len(payload["samples"])
        or payload["index_digest"] != common.serialization.canonical_json_sha256(_index_digest_payload(payload))
    ):
        message = f"Transient dataset index contract or identity is invalid: {path}."
        raise TransientDataContractError(message)
    case_keys = {
        "package_case_id",
        "source_relative_path",
        "source_hdf5_sha256",
        "source_status_sha256",
        "case_input_id",
        "simulation_case_id",
        "source_batch_id",
        "source_simulation_profile",
        "material_family",
        "evaluation_regime",
        "membership",
        "ood_group",
        "ood_parameters",
        "ood_evidence",
        "sequence_length",
        "stored_state_count",
        "transition_count",
        "t_stop_exact",
        "irregular_stop_time",
    }
    sample_keys = {
        "case_index",
        "sample_id",
        "time_index_n",
        "time_index_n_plus_1",
        "schedule_index_n",
        "schedule_index_n_plus_1",
    }
    package_case_ids: set[str] = set()
    simulation_case_ids: set[str] = set()
    for case in payload["cases"]:
        if not isinstance(case, dict) or set(case) != case_keys:
            message = f"Transient case records do not match the current schema: {path}."
            raise TransientDataContractError(message)
        sequence_length = case["sequence_length"]
        stored_state_count = case["stored_state_count"]
        transition_count = case["transition_count"]
        irregular_stop = case["irregular_stop_time"]
        t_stop_exact = case["t_stop_exact"]
        package_case_id = case["package_case_id"]
        simulation_case_id = case["simulation_case_id"]
        numeric_stop = (
            not isinstance(t_stop_exact, bool)
            and isinstance(t_stop_exact, (int, float))
            and math.isfinite(float(t_stop_exact))
            and 0.0 <= float(t_stop_exact) <= float(horizon["value"])
        )
        irregular_valid = irregular_stop is None or (
            not isinstance(irregular_stop, bool)
            and isinstance(irregular_stop, (int, float))
            and math.isfinite(float(irregular_stop))
            and float(irregular_stop) == float(t_stop_exact)
        )
        if (
            not isinstance(package_case_id, str)
            or not package_case_id
            or package_case_id in package_case_ids
            or not isinstance(simulation_case_id, str)
            or not simulation_case_id
            or simulation_case_id in simulation_case_ids
            or case["source_simulation_profile"] != transient_contract.TRANSIENT_PROFILE_ID
            or case["evaluation_regime"] != payload["evaluation_regime"]
            or isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < _MINIMUM_TRANSIENT_STATE_COUNT
            or isinstance(stored_state_count, bool)
            or not isinstance(stored_state_count, int)
            or stored_state_count != sequence_length + int(irregular_stop is not None)
            or isinstance(transition_count, bool)
            or not isinstance(transition_count, int)
            or transition_count < 1
            or not numeric_stop
            or not irregular_valid
        ):
            message = f"Transient case temporal identity is invalid: {path}."
            raise TransientDataContractError(message)
        package_case_ids.add(package_case_id)
        simulation_case_ids.add(simulation_case_id)
    case_count = len(payload["cases"])
    samples_by_case: list[list[dict[str, Any]]] = [[] for _ in range(case_count)]
    sample_ids: set[str] = set()
    previous_case_index = -1
    for sample in payload["samples"]:
        if not isinstance(sample, dict) or set(sample) != sample_keys:
            message = f"Transient sample records do not match the current schema: {path}."
            raise TransientDataContractError(message)
        case_index = sample["case_index"]
        positions = tuple(sample[key] for key in ("time_index_n", "time_index_n_plus_1", "schedule_index_n", "schedule_index_n_plus_1"))
        if (
            isinstance(case_index, bool)
            or not isinstance(case_index, int)
            or not 0 <= case_index < case_count
            or case_index < previous_case_index
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in positions)
        ):
            message = f"Transient sample references invalid or unordered source indices: {path}."
            raise TransientDataContractError(message)
        sample_id = sample["sample_id"]
        expected_id = f"{payload['cases'][case_index]['package_case_id']}__step_{sample['time_index_n']:04d}"
        if not isinstance(sample_id, str) or sample_id != expected_id or sample_id in sample_ids:
            message = f"Transient sample identities are invalid or duplicated: {path}."
            raise TransientDataContractError(message)
        if not sample["time_index_n"] < sample["time_index_n_plus_1"] < payload["cases"][case_index]["sequence_length"]:
            message = f"Transient sample time indices leave the regular HDF5 sequence: {path}."
            raise TransientDataContractError(message)
        if not sample["schedule_index_n"] < sample["schedule_index_n_plus_1"]:
            message = f"Transient sample schedule endpoints are not forward ordered: {path}."
            raise TransientDataContractError(message)
        sample_ids.add(sample_id)
        samples_by_case[case_index].append(sample)
        previous_case_index = case_index
    for case_index, case_samples in enumerate(samples_by_case):
        case = payload["cases"][case_index]
        if len(case_samples) != case["transition_count"]:
            message = f"Transient case transition counts disagree with indexed samples: {path}."
            raise TransientDataContractError(message)
        time_stride = case_samples[0]["time_index_n_plus_1"] - case_samples[0]["time_index_n"]
        schedule_stride = case_samples[0]["schedule_index_n_plus_1"] - case_samples[0]["schedule_index_n"]
        for step_index, sample in enumerate(case_samples):
            if (
                sample["time_index_n"] != step_index * time_stride
                or sample["time_index_n_plus_1"] != (step_index + 1) * time_stride
                or sample["schedule_index_n"] != step_index * schedule_stride
                or sample["schedule_index_n_plus_1"] != (step_index + 1) * schedule_stride
            ):
                message = f"Transient samples are not one deterministic consecutive regular prefix: {path}."
                raise TransientDataContractError(message)
        unused_regular_steps = case["sequence_length"] - 1 - case_samples[-1]["time_index_n_plus_1"]
        if not 0 <= unused_regular_steps < time_stride:
            message = f"Transient samples omit a complete regular transition: {path}."
            raise TransientDataContractError(message)
    return payload
