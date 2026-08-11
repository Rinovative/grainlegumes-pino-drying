"""
===============================================================================
generation_contracts_scalar_handoff.py
===============================================================================
Admit the exact transient scalar handoff and its immutable source identity.
Responsibilities:
  - Define the ordered 12-field runtime handoff and numeric serialization
  - Validate names, units, ownership, values, and exact source-byte identity
  - Expose one immutable admission for case, runtime, and HDF5 consumers
Design principles:
  - Only case-dependent values actually supplied at runtime enter the handoff
  - One canonical order is reused without fixed-value or derived aliases
  - Portable contract identity excludes machine-specific source paths
This module does NOT:
  - Sample scientific values, generate schedules, or execute COMSOL
  - Define learned Dataset scalar views or mutable production inventories
===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src import common

from . import generation_contracts_profiles as profiles

SCALAR_HANDOFF_CONTRACT_SCHEMA_KIND: Final = "transient_scalar_handoff_contract"
SCALAR_HANDOFF_CONTRACT_SCHEMA_VERSION: Final = 1
_ENTRY_KEYS = frozenset({"name", "value", "unit", "owner"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_UNSAFE_LIST_CHARACTERS = frozenset({",", "\n", "\r", "\x00"})
_SCALAR_COLUMN_COUNT = 3
_CANONICAL_OWNERSHIP: Final = tuple("case_dependent" for _name in profiles.TRANSIENT_SCALAR_INPUT_FIELDS)
_TRANSIENT_CONTRACT_PAYLOAD: Final = {
    "schema_kind": SCALAR_HANDOFF_CONTRACT_SCHEMA_KIND,
    "schema_version": SCALAR_HANDOFF_CONTRACT_SCHEMA_VERSION,
    "simulation_profile": profiles.TRANSIENT_DRYING_PROFILE,
    "field_names": list(profiles.TRANSIENT_SCALAR_INPUT_FIELDS),
    "units": list(profiles.TRANSIENT_SCALAR_INPUT_UNITS),
    "ownership": list(_CANONICAL_OWNERSHIP),
}
TRANSIENT_SCALAR_HANDOFF_CONTRACT_SHA256: Final = common.serialization.canonical_json_sha256(_TRANSIENT_CONTRACT_PAYLOAD)


@dataclass(frozen=True, slots=True)
class ScalarHandoffEntry:
    """One admitted scalar value in canonical handoff order."""

    name: str
    value: float
    unit: str
    owner: str

    def as_dict(self) -> dict[str, str | float]:
        """Return the portable persisted entry representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "owner": self.owner,
        }


@dataclass(frozen=True, slots=True)
class ScalarHandoffAdmission:
    """One immutable transient runtime handoff and observed source identity."""

    profile_id: str
    source_path: Path
    source_filename: str
    sha256: str
    size_bytes: int
    contract_sha256: str
    entries: tuple[ScalarHandoffEntry, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return exact admitted field order."""
        return tuple(entry.name for entry in self.entries)

    @property
    def values(self) -> tuple[float, ...]:
        """Return exact admitted values in contract order."""
        return tuple(entry.value for entry in self.entries)

    @property
    def units(self) -> tuple[str, ...]:
        """Return exact admitted units in contract order."""
        return tuple(entry.unit for entry in self.entries)

    @property
    def ownership(self) -> tuple[str, ...]:
        """Return exact persisted ownership labels in contract order."""
        return tuple(entry.owner for entry in self.entries)

    def entries_payload(self) -> list[dict[str, str | float]]:
        """Return independent mutable dictionaries for JSON persistence."""
        return [entry.as_dict() for entry in self.entries]

    def provenance_payload(self, *, include_source_path: bool = False) -> dict[str, Any]:
        """Return contract and source identity without path leakage by default."""
        source: dict[str, Any] = {
            "filename": self.source_filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if include_source_path:
            source["path"] = str(self.source_path)
        return {
            "simulation_profile": self.profile_id,
            "contract_sha256": self.contract_sha256,
            "source": source,
            "field_names": list(self.field_names),
            "units": list(self.units),
            "ownership": list(self.ownership),
            "entries": self.entries_payload(),
        }


def _finite_scalar(value: Any, *, name: str) -> float:
    """Coerce one non-Boolean finite scalar."""
    if isinstance(value, bool):
        message = f"Scalar handoff value {name!r} must be numeric, not Boolean."
        raise TypeError(message)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        message = f"Scalar handoff value {name!r} is not numeric."
        raise TypeError(message) from error
    if not math.isfinite(number):
        message = f"Scalar handoff value {name!r} must be finite."
        raise ValueError(message)
    return number


def format_scalar_number(value: Any) -> str:
    """Format one finite scalar with locale-independent round-trip precision."""
    return format(_finite_scalar(value, name="serialized_value"), ".17g")


def format_comsol_parameter(entry: ScalarHandoffEntry) -> str:
    """Format one admitted runtime parameter without ambiguous list syntax."""
    number = format_scalar_number(entry.value)
    for label, token in (("name", entry.name), ("unit", entry.unit), ("value", number)):
        if not token or any(character in token for character in _UNSAFE_LIST_CHARACTERS):
            message = f"COMSOL scalar {label} contains unsafe list syntax: {token!r}."
            raise ValueError(message)
    if any(character in entry.name for character in "[]") or any(character in entry.unit for character in "[]"):
        message = "COMSOL scalar names and units cannot contain square brackets."
        raise ValueError(message)
    return number if entry.unit == "1" else f"{number}[{entry.unit}]"


def _as_entry(value: ScalarHandoffEntry | Mapping[str, Any]) -> ScalarHandoffEntry:
    """Normalize one entry while rejecting undeclared persisted keys."""
    if isinstance(value, ScalarHandoffEntry):
        return ScalarHandoffEntry(
            value.name,
            _finite_scalar(value.value, name=value.name),
            value.unit,
            value.owner,
        )
    if not isinstance(value, Mapping) or set(value) != _ENTRY_KEYS:
        message = "Scalar handoff entries must contain exactly name, value, unit, and owner."
        raise ValueError(message)
    name = value["name"]
    unit = value["unit"]
    owner = value["owner"]
    if not isinstance(name, str) or not isinstance(unit, str) or not isinstance(owner, str):
        message = "Scalar handoff entry names, units, and owners must be strings."
        raise TypeError(message)
    return ScalarHandoffEntry(
        name,
        _finite_scalar(value["value"], name=name),
        unit,
        owner,
    )


def _validated_entries(
    entries: Sequence[ScalarHandoffEntry | Mapping[str, Any]],
) -> tuple[ScalarHandoffEntry, ...]:
    """Validate one exact ordered transient entry sequence."""
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        message = "Scalar handoff entries must be an ordered sequence."
        raise TypeError(message)
    normalized = tuple(_as_entry(entry) for entry in entries)
    names = tuple(entry.name for entry in normalized)
    if names != profiles.TRANSIENT_SCALAR_INPUT_FIELDS:
        message = "Scalar handoff contains missing, duplicate, unknown, or misordered names."
        raise ValueError(message)
    units = tuple(entry.unit for entry in normalized)
    if units != profiles.TRANSIENT_SCALAR_INPUT_UNITS:
        message = "Scalar handoff units do not match the exact canonical field order."
        raise ValueError(message)
    ownership = tuple(entry.owner for entry in normalized)
    if ownership != _CANONICAL_OWNERSHIP:
        message = "Scalar handoff ownership must be case_dependent for all twelve entries in canonical order."
        raise ValueError(message)
    return normalized


def build_transient_scalar_entries(
    values: Mapping[str, Any],
    units: Mapping[str, str],
) -> tuple[ScalarHandoffEntry, ...]:
    """Build the canonical transient runtime handoff from resolved case mappings."""
    entries: list[ScalarHandoffEntry] = []
    for name, owner in zip(
        profiles.TRANSIENT_SCALAR_INPUT_FIELDS,
        _CANONICAL_OWNERSHIP,
        strict=True,
    ):
        if name not in values:
            message = f"Required scalar handoff value {name!r} is unresolved."
            raise ValueError(message)
        entries.append(
            ScalarHandoffEntry(
                name=name,
                value=_finite_scalar(values[name], name=name),
                unit=units.get(name, ""),
                owner=owner,
            )
        )
    return _validated_entries(entries)


def admit_transient_scalar_handoff(
    entries: Sequence[ScalarHandoffEntry | Mapping[str, Any]],
    *,
    source_path: Path | str,
    source_filename: str,
    sha256: str,
    size_bytes: int,
) -> ScalarHandoffAdmission:
    """Admit exact entries and immutable observed source identity."""
    path = Path(source_path)
    if not source_filename or Path(source_filename).name != source_filename or path.name != source_filename:
        message = "Scalar handoff source filename must exactly match its source path."
        raise ValueError(message)
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        message = "Scalar handoff source SHA-256 is malformed."
        raise ValueError(message)
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        message = "Scalar handoff source size must be a positive integer."
        raise ValueError(message)
    return ScalarHandoffAdmission(
        profile_id=profiles.TRANSIENT_DRYING_PROFILE,
        source_path=path,
        source_filename=source_filename,
        sha256=sha256,
        size_bytes=size_bytes,
        contract_sha256=TRANSIENT_SCALAR_HANDOFF_CONTRACT_SHA256,
        entries=_validated_entries(entries),
    )


def validate_transient_scalar_source(admission: ScalarHandoffAdmission) -> None:
    """Require the admitted source path to retain its observed byte identity."""
    path = admission.source_path
    if path.is_symlink() or not path.is_file():
        message = f"Scalar handoff source is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    data = path.read_bytes()
    if len(data) != admission.size_bytes or hashlib.sha256(data).hexdigest() != admission.sha256:
        message = "Scalar adapter bytes changed after scalar-handoff admission."
        raise RuntimeError(message)


def admit_transient_scalar_file(
    source_path: Path | str,
    *,
    delimiter: str,
    expected_sha256: str,
    expected_size_bytes: int,
    recorded_entries: Sequence[ScalarHandoffEntry | Mapping[str, Any]],
) -> ScalarHandoffAdmission:
    """Read one identity-bound long-form CSV through canonical admission."""
    path = Path(source_path)
    if delimiter != ";":
        message = "Scalar handoff delimiter must be the canonical semicolon."
        raise ValueError(message)
    if path.is_symlink():
        message = f"Scalar handoff source cannot be a symbolic link: {path}"
        raise ValueError(message)
    try:
        before = path.stat()
    except OSError as error:
        message = f"Scalar handoff source is missing or unreadable: {path}"
        raise FileNotFoundError(message) from error
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or not os.access(path, os.R_OK):
        message = f"Scalar handoff source is not a readable non-empty regular file: {path}"
        raise ValueError(message)
    try:
        data = path.read_bytes()
        after = path.stat()
    except OSError as error:
        message = f"Scalar adapter is not readable deterministic CSV: {path}"
        raise ValueError(message) from error
    stable_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if stable_identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        message = "Scalar adapter bytes changed during scalar-handoff admission."
        raise RuntimeError(message)
    observed_sha256 = hashlib.sha256(data).hexdigest()
    if observed_sha256 != expected_sha256 or len(data) != expected_size_bytes:
        message = "Scalar adapter bytes changed after case-input identity was computed."
        raise RuntimeError(message)
    if b"\x00" in data or b"\r" in data or not data.endswith(b"\n"):
        message = "Scalar adapter must be UTF-8 with LF line endings and one final newline."
        raise ValueError(message)
    try:
        text = data.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        message = f"Scalar adapter is not readable deterministic CSV: {path}"
        raise ValueError(message) from error
    expected_count = len(profiles.TRANSIENT_SCALAR_INPUT_FIELDS)
    if not rows or rows[0] != ["name", "value", "unit"] or len(rows) != expected_count + 1:
        message = "Scalar adapter header or row count does not match its profile contract."
        raise ValueError(message)
    if any(len(row) != _SCALAR_COLUMN_COUNT for row in rows[1:]):
        message = "Scalar adapter rows must contain exactly name, value, and unit."
        raise ValueError(message)
    parsed_entries: list[dict[str, Any]] = []
    for row, owner in zip(rows[1:], _CANONICAL_OWNERSHIP, strict=True):
        value = _finite_scalar(row[1], name=row[0])
        if row[1] != format_scalar_number(value):
            message = f"Scalar adapter value {row[0]!r} is not in canonical round-trip format."
            raise ValueError(message)
        parsed_entries.append({"name": row[0], "value": value, "unit": row[2], "owner": owner})
    admission = admit_transient_scalar_handoff(
        parsed_entries,
        source_path=path,
        source_filename=path.name,
        sha256=observed_sha256,
        size_bytes=len(data),
    )
    if admission.entries != _validated_entries(recorded_entries):
        message = "Scalar adapter values disagree with recorded case provenance."
        raise ValueError(message)
    return admission


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return one required mapping with a precise contract error."""
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping."
        raise TypeError(message)
    return value


def admit_case_scalar_handoff(
    case_payload: Mapping[str, Any],
    work_directory: Path | str,
) -> ScalarHandoffAdmission:
    """Admit one case-local runtime-scalar file against its case provenance."""
    if case_payload.get("simulation_profile") != profiles.TRANSIENT_DRYING_PROFILE:
        message = "Scalar handoff admission requires a transient_drying case."
        raise ValueError(message)
    input_contract = _required_mapping(case_payload.get("input_contract"), label="case input_contract")
    input_files = _required_mapping(case_payload.get("input_files"), label="case input_files")
    scalar_handoff = _required_mapping(case_payload.get("scalar_handoff"), label="case scalar_handoff")
    sampled_values = _required_mapping(case_payload.get("sampled_values"), label="case sampled_values")
    sampled_units = _required_mapping(case_payload.get("sampled_units"), label="case sampled_units")
    recorded_entries = case_payload.get("scalars")
    if not isinstance(recorded_entries, Sequence) or isinstance(recorded_entries, (str, bytes)):
        message = "Case scalar entries must be an ordered sequence."
        raise TypeError(message)
    scalar_spec = _required_mapping(input_contract.get("scalar"), label="case scalar input contract")
    filename = scalar_spec.get("filename")
    delimiter = scalar_spec.get("delimiter")
    columns = scalar_spec.get("columns")
    if (
        not isinstance(filename, str)
        or filename != "scalars.csv"
        or Path(filename).name != filename
        or delimiter != ";"
        or columns != ["name", "value", "unit"]
    ):
        message = "Case scalar input contract does not describe canonical scalars.csv."
        raise ValueError(message)
    raw_workspace = Path(work_directory).expanduser()
    if raw_workspace.is_symlink() or not raw_workspace.is_dir():
        message = f"Prepared case workspace is missing or unsafe: {raw_workspace}"
        raise ValueError(message)
    workspace = raw_workspace.resolve(strict=True)
    source_path = workspace / filename
    resolved_source = source_path.resolve(strict=False)
    if resolved_source.parent != workspace or source_path.is_symlink():
        message = "Scalar handoff source escapes or aliases the prepared case workspace."
        raise ValueError(message)
    identity = _required_mapping(input_files.get(filename), label="case scalar input-file identity")
    if set(identity) != {"sha256", "size_bytes"}:
        message = "Case scalar input-file identity is incomplete or malformed."
        raise ValueError(message)
    expected_sha256 = identity.get("sha256")
    expected_size_bytes = identity.get("size_bytes")
    if not isinstance(expected_sha256, str) or isinstance(expected_size_bytes, bool) or not isinstance(expected_size_bytes, int):
        message = "Case scalar input-file hash or size has an invalid type."
        raise TypeError(message)
    if (
        set(scalar_handoff) != {"mechanism", "filename", "fresh_per_case", "runtime_validation", "entries"}
        or scalar_handoff.get("mechanism") != "case_local_long_form_csv"
        or scalar_handoff.get("filename") != filename
        or scalar_handoff.get("fresh_per_case") is not True
        or scalar_handoff.get("runtime_validation") != "required"
        or scalar_handoff.get("entries") != recorded_entries
    ):
        message = "Case scalar handoff envelope disagrees with scalar provenance."
        raise ValueError(message)
    admission = admit_transient_scalar_file(
        source_path,
        delimiter=delimiter,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        recorded_entries=recorded_entries,
    )
    for entry in admission.entries:
        if sampled_units.get(entry.name) != entry.unit:
            message = f"Case sampled unit for scalar {entry.name!r} disagrees with scalars.csv."
            raise ValueError(message)
        sampled_value = _finite_scalar(sampled_values.get(entry.name), name=entry.name)
        if sampled_value != entry.value:
            message = f"Case sampled value for scalar {entry.name!r} disagrees with scalars.csv."
            raise ValueError(message)
    return admission
