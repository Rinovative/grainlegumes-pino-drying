"""
===============================================================================
generation_contracts_comsol_spreadsheet.py
===============================================================================
Parse COMSOL Spreadsheet-format exports for observation and numeric admission.
Responsibilities:
  - Identify percent-prefixed COMSOL metadata and column-header records
  - Preserve raw headers while applying declared-unit-aware canonicalization
  - Admit finite rectangular numeric rows without consuming the first data row
  - Validate COMSOL Nodes and Expressions metadata when semantically applicable
Design principles:
  - Header identification follows table structure rather than fixed line numbers
  - Unit normalization removes only exact declared trailing unit decorations
  - Malformed or ambiguous tables fail closed without ordinary header inference
This module does NOT:
  - Infer source aliases, delimiters, scientific mappings, or export roles
  - Rewrite COMSOL output text or make metadata part of scientific identity
===============================================================================
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from itertools import chain, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

_SUPPORTED_DELIMITERS: Final = (",", ";")
_USEFUL_METADATA_KEYS: Final = frozenset(
    {
        "Model",
        "Version",
        "Date",
        "Dimension",
        "Nodes",
        "Expressions",
        "Description",
        "Length unit",
    }
)
_INTEGER_METADATA_KEYS: Final = frozenset({"Dimension", "Nodes", "Expressions"})
_METADATA_FIELD_COUNT: Final = 2
_TEMPORAL_SUFFIX_PATTERN: Final = re.compile(r"^(?P<field_with_unit>.+) @ t=(?P<time>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)$")

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ComsolSpreadsheetTable:
    """One parsed COMSOL Spreadsheet table with raw and canonical headers."""

    delimiter: str
    raw_header: tuple[str, ...]
    canonical_header: tuple[str, ...]
    values: np.ndarray | None
    metadata: dict[str, str | int]
    row_count: int
    column_count: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the numeric row and column count."""
        return self.row_count, self.column_count


@dataclass(frozen=True, slots=True)
class ComsolTemporalColumn:
    """One parsed native COMSOL time-dependent column descriptor."""

    raw_header: str
    source: str
    unit: str
    state_time: float
    state_time_text_atol: float
    column_index: int


@dataclass(frozen=True, slots=True)
class ComsolTemporalGroup:
    """One complete logical field group owned by a single state time."""

    state_time: float
    columns: tuple[ComsolTemporalColumn, ...]

    def column_index(self, source: str) -> int:
        """Return the unique raw column index for one logical source."""
        matches = [column.column_index for column in self.columns if column.source == source]
        if len(matches) != 1:
            message = f"Temporal state {self.state_time:g} does not own exactly one source {source!r}."
            raise ValueError(message)
        return matches[0]


def _parse_record(record: str, *, delimiter: str, label: str) -> tuple[str, ...]:
    """Parse and trim one delimited logical record."""
    try:
        parsed = next(csv.reader([record], delimiter=delimiter, strict=True))
    except csv.Error as error:
        message = f"{label} is not valid delimiter-separated text."
        raise ValueError(message) from error
    return tuple(value.strip() for value in parsed)


def _numeric_row(record: str, *, delimiter: str, width: int, label: str) -> list[float]:
    """Parse one finite fixed-width numeric data record."""
    fields = _parse_record(record, delimiter=delimiter, label=label)
    if len(fields) != width:
        message = f"{label} has {len(fields)} fields; expected {width}."
        raise ValueError(message)
    try:
        values = [float(value) for value in fields]
    except ValueError as error:
        message = f"{label} contains a malformed numeric value."
        raise ValueError(message) from error
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        message = f"{label} contains NaN or infinity."
        raise ValueError(message)
    return values


def _all_numeric(fields: Sequence[str]) -> bool:
    """Return whether every field is finite numeric text."""
    if not fields:
        return False
    try:
        values = np.asarray([float(value) for value in fields], dtype=np.float64)
    except ValueError:
        return False
    return bool(np.isfinite(values).all())


def canonicalize_header(
    raw_header: Sequence[str],
    *,
    expected_units: Mapping[str, str],
) -> tuple[str, ...]:
    """
    Canonicalize raw COMSOL headers against exact declared source units.

    Only the exact trailing declared unit decoration is removed from a header
    that otherwise matches one configured source expression. Parentheses inside
    the source expression remain untouched.
    """
    normalized: list[str] = []
    for raw_name in raw_header:
        canonical = raw_name
        for source, unit in expected_units.items():
            if raw_name == source:
                canonical = source
                break
            if raw_name == f"{source} ({unit})":
                canonical = source
                break
        normalized.append(canonical)
    if len(normalized) != len(set(normalized)):
        message = "COMSOL header canonicalization produced duplicate source fields."
        raise ValueError(message)
    return tuple(normalized)


def parse_temporal_column_descriptor(
    raw_header: str,
    *,
    expected_units: Mapping[str, str],
    column_index: int = 0,
) -> ComsolTemporalColumn:
    """
    Parse one native COMSOL time-dependent Spreadsheet column descriptor.

    The grammar owns only the final ``@ t=...`` suffix and an exact declared
    trailing unit. Parentheses and other syntax inside the source expression
    remain untouched.
    """
    match = _TEMPORAL_SUFFIX_PATTERN.fullmatch(raw_header)
    if match is None:
        message = f"Malformed COMSOL temporal column descriptor {raw_header!r}."
        raise ValueError(message)
    time_text = match.group("time")
    try:
        state_time = float(time_text)
    except ValueError as error:
        message = f"COMSOL temporal column has a malformed state time: {raw_header!r}."
        raise ValueError(message) from error
    if not math.isfinite(state_time):
        message = f"COMSOL temporal column has a non-finite state time: {raw_header!r}."
        raise ValueError(message)
    field_with_unit = match.group("field_with_unit")
    candidates = [(source, unit) for source, unit in expected_units.items() if field_with_unit == f"{source} ({unit})"]
    if len(candidates) != 1:
        message = f"COMSOL temporal column must end in one exact declared unit before its time suffix: {raw_header!r}."
        raise ValueError(message)
    source, unit = candidates[0]
    mantissa, _exponent_separator, exponent_text = time_text.lower().partition("e")
    exponent_text = exponent_text or "0"
    decimal_places = len(mantissa.partition(".")[2])
    exponent = int(exponent_text)
    text_atol = 0.0 if decimal_places == 0 else 0.5 * 10.0 ** (exponent - decimal_places)
    return ComsolTemporalColumn(
        raw_header=raw_header,
        source=source,
        unit=unit,
        state_time=state_time,
        state_time_text_atol=text_atol,
        column_index=column_index,
    )


def group_temporal_columns(
    raw_header: Sequence[str],
    *,
    expected_units: Mapping[str, str],
) -> tuple[ComsolTemporalGroup, ...]:
    """
    Group native wide COMSOL descriptors into complete increasing states.

    Every state owns exactly one column for every configured logical source.
    Group identity is derived from descriptors rather than raw column position.
    """
    if not expected_units:
        message = "COMSOL temporal grouping requires at least one declared source and unit."
        raise ValueError(message)
    expected_sources = tuple(expected_units)
    by_time: dict[float, dict[str, ComsolTemporalColumn]] = {}
    ordered_times: list[float] = []
    for column_index, raw_name in enumerate(raw_header):
        descriptor = parse_temporal_column_descriptor(
            raw_name,
            expected_units=expected_units,
            column_index=column_index,
        )
        if descriptor.state_time not in by_time:
            by_time[descriptor.state_time] = {}
            ordered_times.append(descriptor.state_time)
        state = by_time[descriptor.state_time]
        if descriptor.source in state:
            message = f"COMSOL temporal state {descriptor.state_time:g} repeats logical source {descriptor.source!r}."
            raise ValueError(message)
        state[descriptor.source] = descriptor
    if any(right <= left for left, right in pairwise(ordered_times)):
        message = "COMSOL temporal state times must be unique and strictly increasing."
        raise ValueError(message)
    groups: list[ComsolTemporalGroup] = []
    for state_time in ordered_times:
        state = by_time[state_time]
        missing = [source for source in expected_sources if source not in state]
        unknown = sorted(set(state).difference(expected_sources))
        if missing or unknown:
            message = f"COMSOL temporal state {state_time:g} is incomplete or ambiguous; missing={missing}, unknown={unknown}."
            raise ValueError(message)
        groups.append(
            ComsolTemporalGroup(
                state_time=state_time,
                columns=tuple(state[source] for source in expected_sources),
            )
        )
    if not groups:
        message = "COMSOL temporal export contains no state groups."
        raise ValueError(message)
    return tuple(groups)


def _metadata(
    records: Sequence[tuple[str, ...]],
    *,
    path: Path,
) -> dict[str, str | int]:
    """Return useful diagnostic metadata from records preceding the header."""
    metadata: dict[str, str | int] = {}
    for fields in records:
        if len(fields) < _METADATA_FIELD_COUNT or fields[0] not in _USEFUL_METADATA_KEYS:
            continue
        key = fields[0]
        if key in metadata:
            message = f"COMSOL Spreadsheet metadata repeats {key!r}: {path}"
            raise ValueError(message)
        if len(fields) != _METADATA_FIELD_COUNT:
            message = f"COMSOL Spreadsheet metadata {key!r} is malformed: {path}"
            raise ValueError(message)
        raw_value = fields[1]
        if key in _INTEGER_METADATA_KEYS:
            try:
                value = int(raw_value)
            except ValueError as error:
                message = f"COMSOL Spreadsheet metadata {key!r} must be an integer: {path}"
                raise ValueError(message) from error
            if value < 1:
                message = f"COMSOL Spreadsheet metadata {key!r} must be positive: {path}"
                raise ValueError(message)
            metadata[key] = value
        else:
            metadata[key] = raw_value
    return metadata


def _validate_metadata(
    metadata: Mapping[str, str | int],
    *,
    header: Sequence[str],
    row_count: int,
    path: Path,
) -> None:
    """Validate available COMSOL width and row-count evidence."""
    expressions = metadata.get("Expressions")
    if isinstance(expressions, int) and expressions != len(header):
        message = f"COMSOL Spreadsheet Expressions metadata disagrees with parsed width for {path}: metadata={expressions}, parsed={len(header)}."
        raise ValueError(message)
    nodes = metadata.get("Nodes")
    if not isinstance(nodes, int):
        return
    temporal = "t" in header or any(" @ t=" in name for name in header) or any(name.startswith("t (") and name.endswith(")") for name in header)
    matches = row_count % nodes == 0 if temporal else row_count == nodes
    if not matches:
        relation = "a positive multiple of" if temporal else "equal to"
        message = f"COMSOL Spreadsheet Nodes metadata disagrees with parsed rows for {path}: parsed rows must be {relation} {nodes}, got {row_count}."
        raise ValueError(message)


def read_comsol_spreadsheet(
    path: Path | str,
    *,
    delimiter: str,
    expected_units: Mapping[str, str] | None = None,
    include_values: bool = True,
) -> ComsolSpreadsheetTable:
    """
    Read one COMSOL Spreadsheet export or explicit-header delimited test table.

    Percent-prefixed files are interpreted structurally: the complete leading
    percent block is read, the first non-percent record determines data width,
    and the final compatible percent record immediately before data is the
    column header. Numeric records are streamed, so metadata-only admission does
    not retain the complete raw file or numeric matrix.
    """
    source = Path(path)
    if delimiter not in _SUPPORTED_DELIMITERS:
        message = f"Unsupported COMSOL Spreadsheet delimiter {delimiter!r}: {source}"
        raise ValueError(message)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            records = (line.rstrip("\r\n") for line in stream if line.strip())
            try:
                first_record = next(records)
            except StopIteration as error:
                message = f"COMSOL Spreadsheet export is empty: {source}"
                raise ValueError(message) from error

            leading_comments: list[str] = []
            while first_record.lstrip().startswith("%"):
                leading_comments.append(first_record.lstrip()[1:].strip())
                try:
                    first_record = next(records)
                except StopIteration as error:
                    message = f"COMSOL Spreadsheet export has metadata but no numeric data: {source}"
                    raise ValueError(message) from error

            metadata: dict[str, str | int]
            if leading_comments:
                first_data_fields = _parse_record(
                    first_record,
                    delimiter=delimiter,
                    label=f"First COMSOL data record in {source}",
                )
                data_width = len(first_data_fields)
                parsed_comments = [
                    _parse_record(record, delimiter=delimiter, label=f"COMSOL percent record in {source}") for record in leading_comments
                ]
                compatible = [position for position, fields in enumerate(parsed_comments) if len(fields) == data_width]
                candidate = parsed_comments[-1]
                if (
                    not compatible
                    or compatible[-1] != len(parsed_comments) - 1
                    or not candidate
                    or candidate[0] in _USEFUL_METADATA_KEYS
                    or _all_numeric(candidate)
                ):
                    message = f"Could not identify the final width-compatible percent-prefixed COMSOL header immediately before data: {source}"
                    raise ValueError(message)
                raw_header = candidate
                metadata = _metadata(parsed_comments[:-1], path=source)
                data_records = chain((first_record,), records)
            else:
                raw_header = _parse_record(first_record, delimiter=delimiter, label=f"Delimited header in {source}")
                if _all_numeric(raw_header):
                    message = f"Delimited table has no explicit nonnumeric header; refusing to consume its first numeric row: {source}"
                    raise ValueError(message)
                metadata = {}
                try:
                    first_data_record = next(records)
                except StopIteration as error:
                    message = f"Delimited table must contain an explicit header and numeric data: {source}"
                    raise ValueError(message) from error
                first_data_fields = _parse_record(
                    first_data_record,
                    delimiter=delimiter,
                    label=f"First numeric data record in {source}",
                )
                data_width = len(first_data_fields)
                if len(raw_header) != data_width:
                    message = f"Delimited header width disagrees with numeric data width: {source}"
                    raise ValueError(message)
                data_records = chain((first_data_record,), records)

            if not raw_header or len(raw_header) != len(set(raw_header)) or any(not name for name in raw_header):
                message = f"COMSOL Spreadsheet header is empty or contains duplicate fields: {source}"
                raise ValueError(message)
            values: list[list[float]] = []
            row_count = 0
            for row_number, record in enumerate(data_records, start=1):
                if record.lstrip().startswith(("%", "#")):
                    message = f"Unexpected comment record after COMSOL numeric data began at row {row_number}: {source}"
                    raise ValueError(message)
                numeric = _numeric_row(
                    record,
                    delimiter=delimiter,
                    width=len(raw_header),
                    label=f"COMSOL numeric record {row_number} in {source}",
                )
                row_count += 1
                if include_values:
                    values.append(numeric)
    except (OSError, UnicodeDecodeError) as error:
        message = f"Configured COMSOL export is not readable text: {source}"
        raise ValueError(message) from error

    array = np.asarray(values, dtype=np.float64) if include_values else None
    canonical_header = canonicalize_header(raw_header, expected_units={} if expected_units is None else expected_units)
    _validate_metadata(
        metadata,
        header=canonical_header,
        row_count=row_count,
        path=source,
    )
    return ComsolSpreadsheetTable(
        delimiter=delimiter,
        raw_header=raw_header,
        canonical_header=canonical_header,
        values=array,
        metadata=metadata,
        row_count=row_count,
        column_count=len(raw_header),
    )


def iter_comsol_spreadsheet_numeric_rows(
    path: Path | str,
    *,
    delimiter: str,
    width: int,
) -> Iterator[np.ndarray]:
    """Yield finite fixed-width numeric rows without retaining the raw table."""
    source = Path(path)
    if delimiter not in _SUPPORTED_DELIMITERS:
        message = f"Unsupported COMSOL Spreadsheet delimiter {delimiter!r}: {source}"
        raise ValueError(message)

    def _rows() -> Iterator[np.ndarray]:
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as stream:
                percent_prefixed: bool | None = None
                plain_header_consumed = False
                row_number = 0
                for record in stream:
                    if not record.strip():
                        continue
                    stripped = record.rstrip("\r\n")
                    if percent_prefixed is None:
                        percent_prefixed = stripped.lstrip().startswith("%")
                    if percent_prefixed and row_number == 0 and stripped.lstrip().startswith("%"):
                        continue
                    if not percent_prefixed and not plain_header_consumed:
                        plain_header_consumed = True
                        continue
                    if stripped.lstrip().startswith(("%", "#")):
                        message = f"Unexpected comment record after COMSOL numeric data began at row {row_number + 1}: {source}"
                        raise ValueError(message)
                    row_number += 1
                    yield np.asarray(
                        _numeric_row(
                            stripped,
                            delimiter=delimiter,
                            width=width,
                            label=f"COMSOL numeric record {row_number} in {source}",
                        ),
                        dtype=np.float64,
                    )
        except (OSError, UnicodeDecodeError) as error:
            message = f"Configured COMSOL export is not readable text: {source}"
            raise ValueError(message) from error

    return _rows()


def _delimiter_candidate(path: Path, delimiter: str) -> tuple[int | None, str | None]:
    """Return parsed width or one delimiter-specific failure."""
    try:
        table = read_comsol_spreadsheet(path, delimiter=delimiter, include_values=False)
    except ValueError as error:
        return None, str(error)
    return table.shape[1], None


def detect_comsol_spreadsheet_delimiter(path: Path | str) -> str:
    """Detect comma or semicolon from the uniquely parseable table structure."""
    source = Path(path)
    candidates: list[tuple[str, int]] = []
    errors: list[str] = []
    for delimiter in _SUPPORTED_DELIMITERS:
        width, error = _delimiter_candidate(source, delimiter)
        if width is None:
            errors.append(f"{delimiter!r}: {error}")
        else:
            candidates.append((delimiter, width))
    if not candidates:
        message = f"Could not parse COMSOL Spreadsheet export with a supported delimiter: {source}. {'; '.join(errors)}"
        raise ValueError(message)
    maximum_width = max(width for _, width in candidates)
    widest = [delimiter for delimiter, width in candidates if width == maximum_width]
    if len(widest) != 1:
        message = f"COMSOL Spreadsheet delimiter is ambiguous for {source}: {widest}."
        raise ValueError(message)
    return widest[0]
