"""
generation_cases_admission.py

Admit immutable canonical pre-execution input batches for inspection.
Responsibilities:
  - Validate input-generation manifests, case evidence, and adapter hashes
  - Reconstruct column-major spatial fields on the declared Cartesian grid
  - Compare independently generated case bundles under one exact identity contract
  - Expose immutable lazy discovery records to analysis consumers
Design principles:
  - Metadata is the sole discovery boundary; arbitrary raw directories are ignored
  - Persisted identities, lifecycle state, and adapter contracts fail closed
This module does NOT:
  - Run COMSOL, infer missing raw inputs, or publish generation results
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from src import common
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff

from . import generation_cases_case as case_service
from . import generation_cases_config as config_service
from . import generation_cases_schedule as schedule_service

_MINIMUM_ROWS = 2
_SCALAR_COLUMNS = 3
_SHA256_LENGTH = 64
INPUT_BATCH_SCHEMA_KIND = "generation_input_batch"
INPUT_BATCH_SCHEMA_VERSION = 1
_INPUT_EXECUTION_ARTIFACT_NAMES = frozenset(
    {
        "_SUCCESS",
        "case.h5",
        "execution_provenance.json",
        "model.mph",
        "solved.mph",
        "solver.log",
        "status.json",
        "timing.json",
    }
)
INPUT_GENERATION_IDENTITY_FIELDS = (
    "schema_kind",
    "schema_version",
    "campaign_id",
    "campaign_purpose",
    "batch_name",
    "batch_id",
    "batch_identity",
    "simulation_profile",
    "material_family",
    "sampling_regime",
    "case_input_config_digest",
    "scientific_config_digest",
    "git_commit",
    "generator_version",
    "case_schema_version",
    "case_contract_digest",
    "template_relative_path",
    "template_sha256",
    "resolved_config_sha256",
)
INPUT_GENERATION_COMPATIBILITY_FIELDS = tuple(field for field in INPUT_GENERATION_IDENTITY_FIELDS if field != "git_commit")
INPUT_MANIFEST_KEYS = frozenset(
    {
        *INPUT_GENERATION_IDENTITY_FIELDS,
        "batch_storage_name",
        "input_generation_id",
        "status",
        "case_indices",
        "cases",
    }
)
INPUT_CASE_RECORD_KEYS = frozenset(
    {
        "case_index",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "case_json_sha256",
        "seed_evidence_sha256",
        "input_files",
    }
)


def compute_input_generation_id(evidence: Mapping[str, Any]) -> str:
    """
    Derive the stable input-generation ID from source and batch evidence.

    Parameters
    ----------
    evidence : Mapping[str, Any]
        Manifest or pre-publication evidence containing every maintained
        input-generation identity field. Case membership is deliberately absent
        so bounded requests merge into one canonical input batch.

    Returns
    -------
    str
        Logical input-generation name with 24 hexadecimal identity digits.

    Raises
    ------
    ValueError
        If maintained identity evidence is incomplete.

    """
    try:
        identity = {field: evidence[field] for field in INPUT_GENERATION_IDENTITY_FIELDS}
    except KeyError as error:
        message = "Input-generation identity evidence is incomplete."
        raise ValueError(message) from error
    return "input-" + common.serialization.canonical_json_sha256(identity)[:24]


@dataclass(frozen=True, slots=True)
class AdmittedInputCase:
    """
    Hold a validated maintained-profile bundle for raw-input inspection.

    Attributes
    ----------
    source_id, source_kind : str
        Persisted source identity and lifecycle category.
    profile_id : str
        Maintained simulation-profile identifier.
    label : str
        Human-readable source and case label for selectors.
    case_id : str
        Persisted production case identity.
    case_index : int
        Configured member index within the generation batch.
    directory : pathlib.Path
        Directory containing the admitted ``case.json`` evidence.
    payload : Mapping[str, Any]
        Deeply immutable validated case payload.
    parameter_metadata : Mapping[str, Mapping[str, str]]
        Immutable canonical presentation metadata when available.
    fields : Mapping[str, numpy.ndarray]
        Read-only structured-grid fields in canonical profile order.
    scalars : Mapping[str, float]
        Exact transient scalar handoff, empty for steady-flow inputs.
    schedule : numpy.ndarray | None
        Exact final transient boundary schedule when profile-owned.

    """

    source_id: str
    source_kind: str
    profile_id: str
    batch_storage_name: str | None
    campaign_purpose: str | None
    label: str
    case_id: str
    case_index: int
    directory: Path
    payload: Mapping[str, Any]
    parameter_metadata: Mapping[str, Mapping[str, str]]
    fields: Mapping[str, np.ndarray]
    scalars: Mapping[str, float]
    schedule: np.ndarray | None


@dataclass(frozen=True, slots=True)
class InputCaseReference:
    """Bind one discoverable source case without loading adapter arrays."""

    source_id: str
    source_kind: str
    profile_id: str
    batch_id: str
    batch_storage_name: str
    batch_identity: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str
    case_id: str
    case_index: int
    case_input_id: str
    simulation_case_id: str
    case_json_sha256: str
    case_json_size_bytes: int
    case_directory: Path
    input_directory: Path
    parameter_metadata: Mapping[str, Any] | None
    _case_payload_json: str

    def case_payload(self) -> dict[str, Any]:
        """Return an independent copy of the validated case metadata."""
        value = json.loads(self._case_payload_json)
        if not isinstance(value, dict):
            message = "Validated input case metadata is no longer an object."
            raise TypeError(message)
        return value


@dataclass(frozen=True, slots=True)
class InputSource:
    """Describe one bounded metadata-validated input source and its case references."""

    source_id: str
    source_kind: str
    profile_id: str
    batch_id: str
    batch_storage_name: str
    batch_identity: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str
    directory: Path
    cases: tuple[InputCaseReference, ...]
    _manifest: Mapping[str, Any]
    _resolved_config: Mapping[str, Any]

    def manifest_payload(self) -> Mapping[str, Any]:
        """Return the immutable validated source manifest."""
        return self._manifest

    def resolved_config_evidence(self) -> Mapping[str, Any]:
        """Return the immutable validated resolved configuration."""
        return self._resolved_config

    def resolved_config_matches(self, value: Mapping[str, Any]) -> bool:
        """Compare JSON semantics without materializing or reparsing evidence."""
        return _json_evidence_matches(self._resolved_config, value)


@dataclass(frozen=True, slots=True)
class InputSourceDiscovery:
    """Hold metadata-first input-generated sources with compact issues."""

    sources: tuple[InputSource, ...]
    issues: tuple[InputDiscoveryIssue, ...]


@dataclass(frozen=True, slots=True)
class InputDiscoveryIssue:
    """
    Describe one raw-input source excluded from inspection.

    Attributes
    ----------
    source_id : str
        Logical input-generation or completed-batch identity.
    directory : pathlib.Path
        Candidate publication that could not be admitted.
    message : str
        Actionable validation or discovery failure.

    """

    source_id: str
    directory: Path
    message: str


def _json_evidence_matches(left: Any, right: Any) -> bool:
    """Compare frozen and ordinary JSON containers without allocating copies."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_json_evidence_matches(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _json_evidence_matches(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _freeze_json_evidence(value: Any) -> Any:
    """Recursively freeze validated JSON evidence without changing scalar leaves."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_evidence(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_evidence(item) for item in value)
    return value


def _admit_parameter_metadata(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Mapping[str, str]]:
    """Validate and freeze optional canonical parameter presentation metadata."""
    if value is None:
        return MappingProxyType({})
    result: dict[str, Mapping[str, str]] = {}
    for name, entry in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(entry, Mapping)
            or set(entry) != {"description", "report_symbol"}
            or any(not isinstance(item, str) or not item for item in entry.values())
        ):
            message = "Generation parameter metadata contains an invalid catalogue entry."
            raise ValueError(message)
        result[name] = MappingProxyType({key: str(entry[key]) for key in ("description", "report_symbol")})
    return MappingProxyType(result)


def read_input_adapter_table(path: Path, *, delimiter: str) -> tuple[list[str], np.ndarray]:
    """
    Read one finite numeric Generation adapter table without schema inference.

    Parameters
    ----------
    path : pathlib.Path
        Exact persisted adapter file.
    delimiter : str
        Canonical delimiter declared by the case input contract.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        Header names and a read-only two-dimensional float array.

    Raises
    ------
    ValueError
        If text, row-width, numeric, or finiteness validation fails.

    """
    try:
        rows = list(csv.reader([line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()], delimiter=delimiter))
    except (OSError, UnicodeDecodeError) as error:
        msg = f"Generation input adapter is not readable text: {path}"
        raise ValueError(msg) from error
    if len(rows) < _MINIMUM_ROWS:
        msg_0 = f"Generation input adapter must contain a header and data: {path}"
        raise ValueError(msg_0)
    header = [item.strip() for item in rows[0]]
    if not header or len(header) != len(set(header)) or any(len(row) != len(header) for row in rows[1:]):
        msg_1 = f"Generation input adapter has duplicate headers or inconsistent row widths: {path}"
        raise ValueError(msg_1)
    try:
        values = np.asarray([[float(item.strip()) for item in row] for row in rows[1:]], dtype=np.float64)
    except ValueError as error:
        msg_2 = f"Generation input adapter contains malformed values: {path}"
        raise ValueError(msg_2) from error
    if not np.isfinite(values).all():
        msg_3 = f"Generation input adapter contains non-finite values: {path}"
        raise ValueError(msg_3)
    values.setflags(write=False)
    return header, values


def _input_path(directory: Path, payload: Mapping[str, Any], kind: str) -> Path:
    spec = payload["input_contract"][kind]
    path = directory / spec["filename"]
    record = payload["input_files"].get(path.name)
    if not path.is_file() or not isinstance(record, Mapping):
        msg = f"Missing persisted {kind} input adapter: {path}"
        raise ValueError(msg)
    if path.stat().st_size != record.get("size_bytes") or common.serialization.file_sha256(path) != record.get("sha256"):
        msg_0 = f"Persisted {kind} input adapter hash or size disagrees with case.json: {path}"
        raise ValueError(msg_0)
    return path


def _admit_fields(directory: Path, payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Reconstruct the writer's ``ravel(order="F")`` Cartesian representation."""
    spec = payload["input_contract"]["spatial"]
    header, values = read_input_adapter_table(_input_path(directory, payload, "spatial"), delimiter=spec["delimiter"])
    if (
        header != list(profiles.spatial_input_fields(payload["simulation_profile"]))
        or header != list(spec["columns"])
        or "x" not in header
        or "y" not in header
    ):
        msg = "Spatial adapter header does not match the Cartesian case input contract."
        raise ValueError(msg)
    x_values, y_values = values[:, header.index("x")], values[:, header.index("y")]
    nx, ny = len(np.unique(x_values)), len(np.unique(y_values))
    if nx * ny != len(values):
        msg_0 = "Spatial adapter does not contain an exact Cartesian grid."
        raise ValueError(msg_0)
    # Preserve configured axis direction: physical coordinates need not ascend.
    x_axis, y_axis = x_values[::ny], y_values[:ny]
    expected_x = np.repeat(x_axis, ny)
    expected_y = np.tile(y_axis, nx)
    if not np.array_equal(x_values, expected_x) or not np.array_equal(y_values, expected_y):
        msg_1 = "Spatial adapter row order does not match canonical F-order serialization."
        raise ValueError(msg_1)
    result = {name: np.array(values[:, column].reshape((ny, nx), order="F"), copy=True) for column, name in enumerate(header)}
    for value in result.values():
        value.setflags(write=False)
    return result


def _stationary_value(payload: Mapping[str, Any], name: str) -> float:
    """Return one uniquely persisted finite stationary fixed value."""
    matches = [entry for entry in payload["stationary_fixed_values"] if entry.get("name") == name]
    if len(matches) != 1:
        msg = f"Case payload must contain exactly one stationary fixed value {name!r}."
        raise ValueError(msg)
    value = matches[0].get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        msg_0 = f"Stationary fixed value {name!r} must be finite numeric evidence."
        raise ValueError(msg_0)
    return float(value)


def _validate_schedule(payload: Mapping[str, Any], schedule: np.ndarray) -> None:
    """Revalidate the final COMSOL table through the canonical schedule owner."""
    metadata = payload["schedule_diagnostics"]
    handoff = metadata["boundary_handoff"]
    grid = handoff["canonical_regular_grid"]
    regular_times = float(grid["start_h"]) + float(grid["interval_h"]) * np.arange(int(grid["node_count"]), dtype=np.float64)
    if regular_times[-1] != float(grid["stop_h"]):
        msg = "Canonical regular-grid metadata is inconsistent."
        raise ValueError(msg)
    ramp = handoff["startup_ramp"]
    schedule_service.validate_comsol_boundary_schedule(
        schedule,
        regular_times=regular_times,
        startup_ramp={
            "enabled": ramp["enabled"],
            "duration_h": ramp["duration_h"],
            "initial_equilibrium_rh_dry_margin": ramp["initial_equilibrium_rh_dry_margin"],
            "max_relative_humidity": ramp["startup_relative_humidity_max"],
        },
        initial_temperature=float(payload["sampled_values"]["T_init"]),
        source_air_temperature=float(payload["sampled_values"]["T_amb"]),
        pressure=_stationary_value(payload, "p_ref"),
        metadata=metadata,
    )


def _resolve_adapter_directory(
    case_directory: Path,
    supplied: Path | str | None,
) -> Path:
    """Resolve an explicit adapter root or the canonical nested input directory."""
    if supplied is not None:
        return Path(supplied).expanduser().resolve()
    nested = case_directory / "inputs"
    return nested if nested.is_dir() else case_directory


def admit_input_case(
    directory: Path | str,
    *,
    source_id: str,
    source_kind: str = "input_generated",
    label: str,
    batch_storage_name: str | None = None,
    campaign_purpose: str | None = None,
    input_directory: Path | str | None = None,
    parameter_metadata: Mapping[str, Any] | None = None,
) -> AdmittedInputCase:
    """
    Validate and load one maintained-profile generated raw-input directory.

    Parameters
    ----------
    directory : pathlib.Path | str
        Directory containing the authoritative ``case.json``.
    source_id : str
        Logical input-generation or completed-batch identity.
    source_kind : str, optional
        Explicit lifecycle category exposed to EDA consumers.
    label : str
        Human-readable selector label.
    batch_storage_name : str | None, optional
        Manifest-validated flat semantic storage locator.
    campaign_purpose : str | None, optional
        Manifest-validated canonical campaign purpose.
    input_directory : pathlib.Path | str | None, optional
        Alternate directory containing retained input adapters.
    parameter_metadata : Mapping[str, Any] | None, optional
        Canonical parameter descriptions and report symbols.

    Returns
    -------
    AdmittedInputCase
        Immutable case evidence with exact reconstructed inputs.

    Raises
    ------
    ValueError
        If identity, hashes, schemas, field order, scalar handoff, or schedule
        evidence violates the maintained production contract.

    """
    directory = Path(directory).expanduser().resolve()
    if batch_storage_name is not None:
        common.paths.validate_logical_name(
            batch_storage_name,
            label="batch_storage_name",
        )
    if campaign_purpose is not None:
        common.paths.validate_logical_name(
            campaign_purpose,
            label="campaign_purpose",
        )
    adapter_directory = _resolve_adapter_directory(directory, input_directory)
    payload = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    case_service.validate_case_payload_schema(payload)
    profile_id = str(payload["simulation_profile"])
    profiles.resolve_profile(profile_id)
    if (
        case_service.compute_case_input_id(payload) != payload["case_input_id"]
        or case_service.compute_simulation_case_id(payload) != payload["simulation_case_id"]
    ):
        msg = "case.json input or simulation identity is invalid."
        raise ValueError(msg)
    fields = _admit_fields(adapter_directory, payload)
    scalars: Mapping[str, float] = MappingProxyType({})
    schedule: np.ndarray | None = None
    if profile_id == profiles.TRANSIENT_DRYING_PROFILE:
        scalar_spec = payload["input_contract"]["scalar"]
        scalar_path = _input_path(adapter_directory, payload, "scalar")
        scalar_rows = list(csv.reader(scalar_path.read_text(encoding="utf-8").splitlines(), delimiter=scalar_spec["delimiter"]))
        if (
            not scalar_rows
            or scalar_rows[0] != ["name", "value", "unit"]
            or len(scalar_rows) < _MINIMUM_ROWS
            or any(len(row) != _SCALAR_COLUMNS for row in scalar_rows[1:])
        ):
            msg_0 = "Scalar adapter header or row width is invalid."
            raise ValueError(msg_0)
        observed = tuple((str(row[0]), float(row[1]), str(row[2])) for row in scalar_rows[1:])
        entries = scalar_handoff.build_transient_scalar_entries(dict(payload["sampled_values"]), dict(payload["sampled_units"]))
        if observed != tuple((entry.name, entry.value, entry.unit) for entry in entries):
            msg_1 = "Scalar adapter order, values, or units do not match canonical scalar handoff."
            raise ValueError(msg_1)
        scalars = MappingProxyType({entry.name: entry.value for entry in entries})
        schedule_spec = payload["input_contract"]["schedule"]
        schedule_header, schedule = read_input_adapter_table(
            _input_path(adapter_directory, payload, "schedule"), delimiter=schedule_spec["delimiter"]
        )
        if tuple(schedule_header) != profiles.SCHEDULE_FIELDS:
            msg_2 = "Schedule adapter header is invalid."
            raise ValueError(msg_2)
        _validate_schedule(payload, schedule)
    return AdmittedInputCase(
        source_id=source_id,
        source_kind=source_kind,
        profile_id=profile_id,
        batch_storage_name=batch_storage_name,
        campaign_purpose=campaign_purpose,
        label=label,
        case_id=payload["case_id"],
        case_index=int(payload["case_index"]),
        directory=directory,
        payload=_freeze_json_evidence(payload),
        parameter_metadata=_admit_parameter_metadata(parameter_metadata),
        fields=MappingProxyType(fields),
        scalars=scalars,
        schedule=schedule,
    )


@dataclass(frozen=True, slots=True)
class _InputBatchMetadata:
    """Hold validated manifest-first input-batch metadata without adapter arrays."""

    directory: Path
    raw_directory: Path
    manifest: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    case_payloads: tuple[Mapping[str, Any], ...]
    parameter_metadata: Mapping[str, Any]


def _validate_input_manifest_shape(
    manifest: Mapping[str, Any],
    *,
    expected_input_generation_id: str | None,
) -> None:
    """Require the exact canonical input-manifest envelope and membership."""
    records = manifest.get("cases")
    indices = manifest.get("case_indices")
    generation_id = manifest.get("input_generation_id")
    if (
        set(manifest) != INPUT_MANIFEST_KEYS
        or manifest.get("schema_kind") != INPUT_BATCH_SCHEMA_KIND
        or manifest.get("schema_version") != INPUT_BATCH_SCHEMA_VERSION
        or manifest.get("status") != "ready"
        or not isinstance(generation_id, str)
        or (expected_input_generation_id is not None and generation_id != expected_input_generation_id)
        or not isinstance(indices, list)
        or not indices
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 1 for index in indices)
        or indices != sorted(set(indices))
        or not isinstance(records, list)
        or len(records) != len(indices)
    ):
        message = "Input-generation manifest schema, lifecycle, or membership is invalid."
        raise ValueError(message)


def _validated_batch_locators(
    manifest: Mapping[str, Any],
    directory: Path,
) -> tuple[str, str]:
    """Validate immutable identity and its distinct semantic storage locator."""
    batch_id = manifest.get("batch_id")
    batch_storage_name = manifest.get("batch_storage_name")
    if not isinstance(batch_id, str):
        message = "Input-generation manifest batch_id must be text."
        raise TypeError(message)
    if not isinstance(batch_storage_name, str):
        message = "Input-generation manifest batch_storage_name must be text."
        raise TypeError(message)
    common.paths.validate_logical_name(batch_id, label="batch_id")
    common.paths.validate_logical_name(
        batch_storage_name,
        label="batch_storage_name",
    )
    generation_id = manifest.get("input_generation_id")
    if not isinstance(generation_id, str):
        message = "Input-generation manifest input_generation_id must be text."
        raise TypeError(message)
    common.paths.validate_logical_name(generation_id, label="input_generation_id")
    if directory.name != generation_id or directory.parent.name != "input_generations" or directory.parent.parent.name != batch_storage_name:
        message = "Input-generation metadata directory must use its exact current generation identity."
        raise ValueError(message)
    return batch_id, batch_storage_name


@dataclass(frozen=True, slots=True)
class _InputBatchEnvelope:
    """Hold validated batch identity before any per-case reconstruction."""

    directory: Path
    raw_directory: Path
    manifest: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    parameter_metadata: Mapping[str, Any]


def _load_input_batch_envelope(
    directory: Path | str,
    *,
    raw_directory: Path | str | None,
    expected_input_generation_id: str | None,
) -> _InputBatchEnvelope:
    """Validate immutable batch identity before bounded case reconstruction."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        message = f"Input-generation metadata directory does not exist: {directory}"
        raise ValueError(message)
    manifest_path = directory / "input_generation_manifest.json"
    resolved_path = directory / "resolved_generation_config.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or not resolved_path.is_file() or resolved_path.is_symlink():
        message = "Input batch requires a safe manifest and resolved configuration."
        raise ValueError(message)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(
        resolved,
        Mapping,
    ):
        message = "Input-generation manifest and resolved configuration must be JSON objects."
        raise TypeError(message)
    _validate_input_manifest_shape(
        manifest,
        expected_input_generation_id=expected_input_generation_id,
    )
    generation_id = manifest["input_generation_id"]
    if compute_input_generation_id(manifest) != generation_id:
        message = "Input-generation manifest disagrees with its deterministic source identity."
        raise ValueError(message)
    batch_id, batch_storage_name = _validated_batch_locators(
        manifest,
        directory,
    )
    if common.serialization.canonical_json_sha256(resolved) != manifest["resolved_config_sha256"]:
        message = "Resolved generation configuration digest disagrees with the input manifest."
        raise ValueError(message)
    profile_id = manifest.get("simulation_profile")
    material = resolved.get("material")
    if (
        not isinstance(profile_id, str)
        or resolved.get("simulation_profile") != profile_id
        or resolved.get("campaign_id") != manifest["campaign_id"]
        or resolved.get("campaign_purpose") != manifest["campaign_purpose"]
        or resolved.get("sampling_regime") != manifest["sampling_regime"]
        or resolved.get("generator_version") != manifest["generator_version"]
        or manifest.get("case_schema_version") != case_service.CASE_SCHEMA_VERSION
        or manifest.get("case_contract_digest") != case_service.CASE_CONTRACT_DIGEST
        or not isinstance(resolved.get("reference_template"), Mapping)
        or resolved["reference_template"].get("sha256") != manifest["template_sha256"]
        or not isinstance(manifest.get("template_relative_path"), str)
        or not isinstance(material, Mapping)
        or material.get("material_family") != manifest["material_family"]
        or not isinstance(resolved.get("registry_metadata"), Mapping)
    ):
        message = "Resolved generation configuration does not bind the input manifest."
        raise ValueError(message)
    profiles.resolve_profile(profile_id)
    scientific_digest = config_service.compute_scientific_config_digest(resolved)
    case_input_digest = config_service.compute_case_input_config_digest(resolved)
    expected_batch_name = config_service.build_batch_name(
        profile_id,
        str(manifest["material_family"]),
        str(manifest["sampling_regime"]),
    )
    if resolved.get("campaign_purpose") == config_service.PILOT_CAMPAIGN_PURPOSE:
        expected_batch_name = config_service.build_batch_name(
            profile_id,
            str(manifest["material_family"]),
            config_service.PILOT_CAMPAIGN_PURPOSE,
        )
    if (
        scientific_digest != manifest["scientific_config_digest"]
        or scientific_digest != manifest["batch_identity"]
        or case_input_digest != manifest["case_input_config_digest"]
        or expected_batch_name != manifest["batch_name"]
        or config_service.build_batch_id(
            expected_batch_name,
            scientific_digest,
        )
        != batch_id
        or config_service.build_batch_storage_name(
            profile_id,
            str(manifest["material_family"]),
            str(manifest["sampling_regime"]),
            str(manifest["campaign_purpose"]),
            scientific_digest,
        )
        != batch_storage_name
    ):
        message = "Resolved generation identities do not bind the input manifest."
        raise ValueError(message)
    if raw_directory is None:
        raw_directory = common.paths.resolve_generation_input_generation_raw_directory(
            batch_storage_name,
            str(generation_id),
            storage_root=directory.parents[4],
        )
    raw = Path(raw_directory).expanduser().resolve()
    if not raw.is_dir() or raw.is_symlink():
        message = f"Canonical input raw batch is missing or unsafe: {raw}"
        raise ValueError(message)
    if raw.name != generation_id or raw.parent.name != "input_generations" or raw.parent.parent.name != batch_storage_name:
        message = "Input-generation raw directory must use its exact current generation identity."
        raise ValueError(message)
    return _InputBatchEnvelope(
        directory=directory,
        raw_directory=raw,
        manifest=manifest,
        resolved_config=resolved,
        parameter_metadata=resolved["registry_metadata"],
    )


def _raise_for_execution_artifact(*roots: Path) -> None:
    """Preserve exact artifact classification only after layout evidence conflicts."""
    for root in roots:
        for name in _INPUT_EXECUTION_ARTIFACT_NAMES:
            if next(root.rglob(name), None) is not None:
                message = f"Input-generated raw evidence contains an execution or success artifact under {root}: {name}."
                raise ValueError(message)


def _validate_input_case_record(
    envelope: _InputBatchEnvelope,
    record: Any,
    index: int,
    *,
    validation_depth: Literal["evidence", "full"],
) -> Mapping[str, Any]:
    """Validate one manifest-selected case without rebuilding its batch."""
    manifest = envelope.manifest
    if (
        not isinstance(record, Mapping)
        or set(record) != INPUT_CASE_RECORD_KEYS
        or record.get("case_index") != index
        or not isinstance(record.get("case_id"), str)
        or not isinstance(record.get("case_input_id"), str)
        or not isinstance(record.get("simulation_case_id"), str)
        or not isinstance(record.get("case_json_sha256"), str)
        or not isinstance(record.get("seed_evidence_sha256"), str)
        or not isinstance(record.get("input_files"), Mapping)
    ):
        message = "Input-generation case identity or hash inventory is invalid."
        raise ValueError(message)
    case_id = str(record["case_id"])
    case_directory = envelope.raw_directory / case_id
    input_directory = case_directory / "inputs"
    case_payload_path = case_directory / "case.json"
    try:
        case_payload_bytes = case_payload_path.read_bytes()
        case_payload = json.loads(case_payload_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Canonical raw case payload is unreadable: {case_payload_path}"
        raise ValueError(message) from error
    if not isinstance(case_payload, dict):
        message = f"Canonical raw case payload must be a JSON object: {case_payload_path}"
        raise TypeError(message)
    if validation_depth == "full" and hashlib.sha256(case_payload_bytes).hexdigest() != record["case_json_sha256"]:
        message = f"Canonical raw case payload digest disagrees with its manifest: {case_payload_path}"
        raise ValueError(message)
    case_service.validate_case_payload_schema(case_payload)
    case_bindings = {
        "simulation_profile": manifest["simulation_profile"],
        "batch_id": manifest["batch_id"],
        "batch_identity": manifest["batch_identity"],
        "scientific_config_digest": manifest["scientific_config_digest"],
        "case_input_config_digest": manifest["case_input_config_digest"],
        "material_family": manifest["material_family"],
        "sampling_regime": manifest["sampling_regime"],
        "git_commit": manifest["git_commit"],
        "generator_version": manifest["generator_version"],
    }
    template = case_payload.get("template")
    if (
        any(case_payload.get(key) != value for key, value in case_bindings.items())
        or not isinstance(template, Mapping)
        or template.get("relative_path") != manifest["template_relative_path"]
        or template.get("sha256") != manifest["template_sha256"]
    ):
        message = f"Canonical raw case source identity disagrees with its manifest: {case_payload_path}"
        raise ValueError(message)
    case_entries = tuple(case_directory.iterdir()) if case_directory.is_dir() and not case_directory.is_symlink() else ()
    input_entries = tuple(input_directory.iterdir()) if input_directory.is_dir() and not input_directory.is_symlink() else ()
    if (
        {entry.name for entry in case_entries} != {"case.json", "inputs"}
        or {entry.name for entry in input_entries} != set(record["input_files"])
        or any(not entry.is_file() or entry.is_symlink() for entry in input_entries)
    ):
        _raise_for_execution_artifact(case_directory)
        message = f"Canonical raw case layout is invalid: {case_directory}"
        raise ValueError(message)
    return case_payload


def _load_input_batch_metadata(
    directory: Path | str,
    *,
    raw_directory: Path | str | None = None,
    expected_input_generation_id: str | None = None,
    validation_depth: Literal["evidence", "full"] = "full",
) -> _InputBatchMetadata:
    """Validate one input batch through identity and exact raw membership."""
    envelope = _load_input_batch_envelope(
        directory,
        raw_directory=raw_directory,
        expected_input_generation_id=expected_input_generation_id,
    )
    records = envelope.manifest["cases"]
    indices = envelope.manifest["case_indices"]
    normalized_records: list[Mapping[str, Any]] = []
    case_payloads: list[Mapping[str, Any]] = []
    expected_names: set[str] = set()
    for record, index in zip(records, indices, strict=True):
        case_payload = _validate_input_case_record(
            envelope,
            record,
            index,
            validation_depth=validation_depth,
        )
        expected_names.add(str(record["case_id"]))
        normalized_records.append(record)
        case_payloads.append(case_payload)
    observed_names = {entry.name for entry in envelope.raw_directory.iterdir()}
    if (
        len(expected_names) != len(records)
        or observed_names != expected_names
        or any(not (envelope.raw_directory / case_id).is_dir() for case_id in expected_names)
    ):
        _raise_for_execution_artifact(envelope.raw_directory)
        message = "Input-generation raw case membership is not exact."
        raise ValueError(message)
    return _InputBatchMetadata(
        directory=envelope.directory,
        raw_directory=envelope.raw_directory,
        manifest=envelope.manifest,
        resolved_config=envelope.resolved_config,
        records=tuple(normalized_records),
        case_payloads=tuple(case_payloads),
        parameter_metadata=envelope.parameter_metadata,
    )


def _bounded_case_records(
    records: tuple[Mapping[str, Any], ...],
    maximum: int | None,
) -> tuple[Mapping[str, Any], ...]:
    """Return the first deterministic bounded case records."""
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1):
        message = "max_cases_per_source must be a positive integer when supplied."
        raise ValueError(message)
    return records if maximum is None else records[:maximum]


def _case_reference(
    *,
    source: InputSource,
    record: Mapping[str, Any],
    case_payload: Mapping[str, Any],
    case_directory: Path,
    input_directory: Path,
    parameter_metadata: Mapping[str, Any] | None,
    validation_depth: Literal["evidence", "full"],
) -> InputCaseReference:
    """Bind one case from admitted metadata with optional byte validation."""
    if validation_depth not in {"evidence", "full"}:
        message = f"Unsupported input admission depth: {validation_depth!r}."
        raise ValueError(message)
    payload_path = case_directory / "case.json"
    identity_fields = (
        "case_index",
        "case_id",
        "case_input_id",
        "simulation_case_id",
    )
    if any(case_payload.get(key) != record[key] for key in identity_fields):
        message = "Input source case.json disagrees with metadata identity evidence."
        raise ValueError(message)
    input_files = record["input_files"]
    if case_payload.get("input_files") != input_files or any(case_payload.get(key) != getattr(source, key) for key in ("batch_id", "batch_identity")):
        message = "Input source case.json disagrees with source metadata evidence."
        raise ValueError(message)
    if common.serialization.canonical_json_sha256(case_payload.get("seed_evidence")) != record["seed_evidence_sha256"]:
        message = "Input source case.json disagrees with metadata hash inventory."
        raise ValueError(message)
    for filename, identity in input_files.items():
        path = input_directory / str(filename)
        size_bytes = identity.get("size_bytes") if isinstance(identity, Mapping) else None
        digest = identity.get("sha256") if isinstance(identity, Mapping) else None
        if (
            not path.is_file()
            or path.is_symlink()
            or not isinstance(identity, Mapping)
            or set(identity) != {"sha256", "size_bytes"}
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
            or path.stat().st_size != size_bytes
            or (validation_depth == "full" and common.serialization.file_sha256(path) != digest)
        ):
            message = f"Input source adapter hash or size disagrees with metadata: {path}"
            raise ValueError(message)
    return InputCaseReference(
        source.source_id,
        source.source_kind,
        source.profile_id,
        source.batch_id,
        source.batch_storage_name,
        source.batch_identity,
        source.material_family,
        source.sampling_regime,
        source.campaign_purpose,
        str(record["case_id"]),
        int(record["case_index"]),
        str(record["case_input_id"]),
        str(record["simulation_case_id"]),
        str(record["case_json_sha256"]),
        payload_path.stat().st_size,
        case_directory,
        input_directory,
        parameter_metadata,
        json.dumps(case_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    )


def admit_input_case_evidence(
    directory: Path | str,
    case_index: int,
    *,
    raw_directory: Path | str | None = None,
    expected_input_generation_id: str | None = None,
    validation_depth: Literal["evidence", "full"] = "full",
) -> InputCaseReference:
    """Admit one selected immutable case without reconstructing its batch."""
    if isinstance(case_index, bool) or not isinstance(case_index, int) or case_index < 1:
        message = "Input case index must be a positive integer."
        raise ValueError(message)
    envelope = _load_input_batch_envelope(
        directory,
        raw_directory=raw_directory,
        expected_input_generation_id=expected_input_generation_id,
    )
    matches = [
        record
        for record, index in zip(
            envelope.manifest["cases"],
            envelope.manifest["case_indices"],
            strict=True,
        )
        if index == case_index
    ]
    if len(matches) != 1:
        message = f"Input manifest does not declare exactly one case index {case_index}."
        raise ValueError(message)
    record = matches[0]
    case_payload = _validate_input_case_record(
        envelope,
        record,
        case_index,
        validation_depth=validation_depth,
    )
    manifest = envelope.manifest
    provisional = InputSource(
        str(manifest["input_generation_id"]),
        "input_generated",
        str(manifest["simulation_profile"]),
        str(manifest["batch_id"]),
        str(manifest["batch_storage_name"]),
        str(manifest["batch_identity"]),
        str(manifest["material_family"]),
        str(manifest["sampling_regime"]),
        str(manifest["campaign_purpose"]),
        envelope.directory,
        (),
        _freeze_json_evidence(manifest),
        _freeze_json_evidence(envelope.resolved_config),
    )
    case_id = str(record["case_id"])
    return _case_reference(
        source=provisional,
        record=record,
        case_payload=case_payload,
        case_directory=envelope.raw_directory / case_id,
        input_directory=envelope.raw_directory / case_id / "inputs",
        parameter_metadata=envelope.parameter_metadata,
        validation_depth=validation_depth,
    )


def admit_input_batch(
    directory: Path | str,
    *,
    raw_directory: Path | str | None = None,
    expected_input_generation_id: str | None = None,
) -> tuple[AdmittedInputCase, ...]:
    """
    Validate one canonical non-executed input batch and all declared cases.

    Parameters
    ----------
    directory : Path | str
        Batch metadata directory containing the manifest and resolved config.
    raw_directory : Path | str | None, optional
        Explicit raw root for a staged publication. Final publications derive it
        from the canonical storage owner.
    expected_input_generation_id : str | None, optional
        Expected source identity while validating staged metadata.

    Returns
    -------
    tuple[AdmittedInputCase, ...]
        Manifest-declared cases in canonical index order.

    Raises
    ------
    TypeError
        If persisted JSON evidence has invalid shape.
    ValueError
        If lifecycle, source, identity, membership, or adapter evidence differs.

    """
    metadata = _load_input_batch_metadata(
        directory,
        raw_directory=raw_directory,
        expected_input_generation_id=expected_input_generation_id,
        validation_depth="full",
    )
    manifest = metadata.manifest
    provisional = InputSource(
        str(manifest["input_generation_id"]),
        "input_generated",
        str(manifest["simulation_profile"]),
        str(manifest["batch_id"]),
        str(manifest["batch_storage_name"]),
        str(manifest["batch_identity"]),
        str(manifest["material_family"]),
        str(manifest["sampling_regime"]),
        str(manifest["campaign_purpose"]),
        metadata.directory,
        (),
        _freeze_json_evidence(manifest),
        _freeze_json_evidence(metadata.resolved_config),
    )
    cases = []
    for record, case_payload in zip(
        metadata.records,
        metadata.case_payloads,
        strict=True,
    ):
        case_id = str(record["case_id"])
        reference = _case_reference(
            source=provisional,
            record=record,
            case_payload=case_payload,
            case_directory=metadata.raw_directory / case_id,
            input_directory=metadata.raw_directory / case_id / "inputs",
            parameter_metadata=metadata.parameter_metadata,
            validation_depth="full",
        )
        case = admit_input_case_reference(reference)
        if any(
            case.payload[key] != manifest[key]
            for key in (
                "batch_id",
                "batch_identity",
                "case_input_config_digest",
                "scientific_config_digest",
                "git_commit",
            )
        ):
            message = "Input-generated case provenance disagrees with its batch manifest."
            raise ValueError(message)
        cases.append(case)
    return tuple(cases)


def admit_input_batch_source(
    directory: Path | str,
    *,
    maximum_cases: int | None = None,
    raw_directory: Path | str | None = None,
    expected_input_generation_id: str | None = None,
    validation_depth: Literal["evidence", "full"] = "full",
) -> InputSource:
    """Admit one canonical source with metadata-only or full byte checks."""
    if validation_depth not in {"evidence", "full"}:
        message = f"Unsupported input admission depth: {validation_depth!r}."
        raise ValueError(message)
    metadata = _load_input_batch_metadata(
        directory,
        raw_directory=raw_directory,
        expected_input_generation_id=expected_input_generation_id,
        validation_depth=validation_depth,
    )
    manifest = metadata.manifest
    provisional = InputSource(
        str(manifest["input_generation_id"]),
        "input_generated",
        str(manifest["simulation_profile"]),
        str(manifest["batch_id"]),
        str(manifest["batch_storage_name"]),
        str(manifest["batch_identity"]),
        str(manifest["material_family"]),
        str(manifest["sampling_regime"]),
        str(manifest["campaign_purpose"]),
        metadata.directory,
        (),
        _freeze_json_evidence(manifest),
        _freeze_json_evidence(metadata.resolved_config),
    )
    bounded_records = _bounded_case_records(metadata.records, maximum_cases)
    references = tuple(
        _case_reference(
            source=provisional,
            record=record,
            case_payload=case_payload,
            case_directory=metadata.raw_directory / str(record["case_id"]),
            input_directory=metadata.raw_directory / str(record["case_id"]) / "inputs",
            parameter_metadata=metadata.parameter_metadata,
            validation_depth=validation_depth,
        )
        for record, case_payload in zip(
            bounded_records,
            metadata.case_payloads[: len(bounded_records)],
            strict=True,
        )
    )
    return InputSource(
        provisional.source_id,
        provisional.source_kind,
        provisional.profile_id,
        provisional.batch_id,
        provisional.batch_storage_name,
        provisional.batch_identity,
        provisional.material_family,
        provisional.sampling_regime,
        provisional.campaign_purpose,
        provisional.directory,
        references,
        provisional.manifest_payload(),
        provisional.resolved_config_evidence(),
    )


def discover_input_batches(
    storage_root: Path | str | None = None,
    *,
    max_sources: int | None = None,
    max_cases_per_source: int | None = None,
) -> InputSourceDiscovery:
    """
    Discover canonical input-generated batches through metadata only.

    Arbitrary raw directories are never discovery candidates. Adapter arrays
    remain unloaded until selection.
    """
    if max_sources is not None and (isinstance(max_sources, bool) or not isinstance(max_sources, int) or max_sources < 1):
        message = "max_sources must be a positive integer when supplied."
        raise ValueError(message)
    candidates = tuple(
        path
        for path in sorted(
            common.paths.get_generation_meta_root(
                storage_root=storage_root,
            ).glob("*/input_generations/*")
        )
        if path.is_dir() and (path / "input_generation_manifest.json").is_file()
    )
    sources: list[InputSource] = []
    issues: list[InputDiscoveryIssue] = []
    for directory in candidates:
        if max_sources is not None and len(sources) >= max_sources:
            break
        try:
            source = admit_input_batch_source(
                directory,
                maximum_cases=max_cases_per_source,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            issues.append(
                InputDiscoveryIssue(
                    directory.name,
                    directory,
                    f"input-generated batch is not inspectable: {error}",
                )
            )
        else:
            sources.append(source)
    return InputSourceDiscovery(tuple(sources), tuple(issues))


def admit_input_case_reference(
    reference: InputCaseReference,
) -> AdmittedInputCase:
    """Load and hash-validate one explicitly selected input-generated case."""
    if reference.source_kind != "input_generated":
        message = "Generation-input case references must come from input-generated batches."
        raise ValueError(message)
    label = f"{reference.profile_id.replace('_', ' ')} / {reference.case_id}"
    return admit_input_case(
        reference.case_directory,
        source_id=reference.source_id,
        source_kind=reference.source_kind,
        label=label,
        batch_storage_name=reference.batch_storage_name,
        campaign_purpose=reference.campaign_purpose,
        input_directory=reference.input_directory,
        parameter_metadata=reference.parameter_metadata,
    )


@dataclass(frozen=True, slots=True)
class CaseBundleEquivalence:
    """Summarize exact identity and byte equality for two generated bundles."""

    case_id: str
    case_index: int
    case_input_id: str
    simulation_case_id: str
    git_commit: str
    seed_evidence_sha256: str
    input_files: tuple[tuple[str, int, str], ...]


def assert_case_bundle_equivalent(
    first_directory: Path | str,
    second_directory: Path | str,
    *,
    first_input_directory: Path | str | None = None,
    second_input_directory: Path | str | None = None,
) -> CaseBundleEquivalence:
    """
    Require two independently generated case bundles to be exactly equivalent.

    Equality requires byte-identical ``case.json`` and declared adapter files,
    including matching Git, resolved science, batch, case, seed, template, input,
    and simulation identities. Extra runtime workspace files are ignored.

    Parameters
    ----------
    first_directory, second_directory : Path | str
        Directories containing the two authoritative ``case.json`` files.
    first_input_directory, second_input_directory : Path | str | None, optional
        Alternate adapter roots, such as retained completed-raw inputs.

    Returns
    -------
    CaseBundleEquivalence
        Shared exact identity and adapter hash evidence.

    Raises
    ------
    ValueError
        If either bundle is invalid or any source, payload, hash, or byte differs.

    """
    case_directories = tuple(Path(directory).expanduser().resolve() for directory in (first_directory, second_directory))
    input_directories = (
        _resolve_adapter_directory(case_directories[0], first_input_directory),
        _resolve_adapter_directory(case_directories[1], second_input_directory),
    )
    payloads = tuple(json.loads((directory / "case.json").read_text(encoding="utf-8")) for directory in case_directories)
    for payload, case_directory, input_directory in zip(
        payloads,
        case_directories,
        input_directories,
        strict=True,
    ):
        if not isinstance(payload, dict):
            message = "Case-equivalence payloads must be JSON objects."
            raise TypeError(message)
        case_service.validate_case_payload_schema(payload)
        admit_input_case(
            case_directory,
            source_id="equivalence-check",
            source_kind="input_generated",
            label="equivalence check",
            input_directory=input_directory,
        )
    if payloads[0] != payloads[1]:
        differing = sorted(key for key in set(payloads[0]) | set(payloads[1]) if payloads[0].get(key) != payloads[1].get(key))
        message = f"Generated case bundles are not equivalent; differing case.json fields: {differing}."
        raise ValueError(message)
    payload = payloads[0]
    filenames = ("case.json", *sorted(payload["input_files"]))
    for filename in filenames:
        first_path = case_directories[0] / filename if filename == "case.json" else input_directories[0] / filename
        second_path = case_directories[1] / filename if filename == "case.json" else input_directories[1] / filename
        if first_path.read_bytes() != second_path.read_bytes():
            message = f"Generated case bundles differ in persisted bytes: {filename}."
            raise ValueError(message)
    input_files = tuple(
        (
            filename,
            int(payload["input_files"][filename]["size_bytes"]),
            str(payload["input_files"][filename]["sha256"]),
        )
        for filename in sorted(payload["input_files"])
    )
    return CaseBundleEquivalence(
        case_id=str(payload["case_id"]),
        case_index=int(payload["case_index"]),
        case_input_id=str(payload["case_input_id"]),
        simulation_case_id=str(payload["simulation_case_id"]),
        git_commit=str(payload["git_commit"]),
        seed_evidence_sha256=common.serialization.canonical_json_sha256(payload["seed_evidence"]),
        input_files=input_files,
    )
