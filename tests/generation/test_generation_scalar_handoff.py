# ruff: noqa: EM101, EM102, PLR2004, S101, TRY003
"""Exact transient scalar-handoff admission and single-owner contracts."""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from src import common, generation
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff

_EXPECTED_FIELDS = (
    "T_amb",
    "eps_bed_cal_ref",
    "rho_bu_dry_ref",
    "k_gr",
    "cp_gr_dry",
    "X_target_wb",
    "r_surf_0",
    "r_int_surf",
    "f_surf",
    "A_osw",
    "B_osw",
    "C_osw",
)
_EXPECTED_UNITS = (
    "K",
    "1",
    "kg/m^3",
    "W/(m*K)",
    "J/(kg*K)",
    "1",
    "1/s",
    "1",
    "1",
    "1",
    "1/K",
    "1",
)


def _scalar_case(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create one minimal identity-bound scalar case without production config data."""
    values = {name: float(index) + 0.125 for index, name in enumerate(_EXPECTED_FIELDS, start=1)}
    units: dict[str, str] = dict(zip(_EXPECTED_FIELDS, _EXPECTED_UNITS, strict=True))
    entries = scalar_handoff.build_transient_scalar_entries(values, units)
    source = tmp_path / "scalars.csv"
    rows = ["name;value;unit"]
    rows.extend(f"{entry.name};{scalar_handoff.format_scalar_number(entry.value)};{entry.unit}" for entry in entries)
    source.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    entry_payload = [entry.as_dict() for entry in entries]
    payload: dict[str, Any] = {
        "simulation_profile": profiles.TRANSIENT_DRYING_PROFILE,
        "input_contract": {
            "scalar": {
                "filename": source.name,
                "delimiter": ";",
                "columns": ["name", "value", "unit"],
            }
        },
        "input_files": {
            source.name: {
                "sha256": common.serialization.file_sha256(source),
                "size_bytes": source.stat().st_size,
            }
        },
        "scalar_handoff": {
            "mechanism": "case_local_long_form_csv",
            "filename": source.name,
            "fresh_per_case": True,
            "runtime_validation": "required",
            "entries": copy.deepcopy(entry_payload),
        },
        "scalars": entry_payload,
        "sampled_values": values,
        "sampled_units": units,
    }
    return source, payload


def _refresh_identity(payload: dict[str, Any], source: Path) -> None:
    """Update only the test-owned source evidence after an intentional rewrite."""
    payload["input_files"]["scalars.csv"] = {
        "sha256": common.serialization.file_sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _rewrite_rows(source: Path, payload: dict[str, Any], rows: list[str]) -> None:
    """Write deterministic mutated rows and bind their fresh test identity."""
    source.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    _refresh_identity(payload, source)


def _admit(
    payload: dict[str, Any],
    tmp_path: Path,
) -> scalar_handoff.ScalarHandoffAdmission:
    """Call the one case-level admission owner."""
    return scalar_handoff.admit_case_scalar_handoff(
        payload,
        tmp_path,
    )


def test_scalar_contract_case_and_schedule_boundaries_are_exact(tmp_path: Path) -> None:
    """Protect the exact 12-field runtime handoff and canonical CSV."""
    source, payload = _scalar_case(tmp_path)
    admission = _admit(payload, tmp_path)

    assert profiles.TRANSIENT_SCALAR_INPUT_FIELDS == _EXPECTED_FIELDS
    assert profiles.TRANSIENT_SCALAR_INPUT_UNITS == _EXPECTED_UNITS
    assert len(admission.entries) == 12
    assert admission.field_names == _EXPECTED_FIELDS
    assert admission.units == _EXPECTED_UNITS
    assert admission.ownership == ("case_dependent",) * 12
    assert admission.source_path == source.resolve()
    assert admission.source_filename == "scalars.csv"
    assert admission.sha256 == payload["input_files"]["scalars.csv"]["sha256"]
    assert admission.size_bytes == source.stat().st_size
    assert admission.contract_sha256 == scalar_handoff.TRANSIENT_SCALAR_HANDOFF_CONTRACT_SHA256
    assert "path" not in admission.provenance_payload()["source"]
    assert admission.provenance_payload(include_source_path=True)["source"]["path"] == str(source.resolve())
    assert source.read_text(encoding="utf-8").splitlines()[0] == "name;value;unit"
    assert len(source.read_text(encoding="utf-8").splitlines()) == 13
    assert payload["scalars"] == payload["scalar_handoff"]["entries"] == admission.entries_payload()
    assert tuple(field.name for field in fields(generation.cases.schedule.Schedule)) == ("values", "metadata")
    assert scalar_handoff.format_scalar_number(0.1) == "0.10000000000000001"


@pytest.mark.parametrize(
    ("name", "unit"),
    [
        ("bad,name", "K"),
        ("bad\nname", "K"),
        ("bad\rname", "K"),
        ("bad\x00name", "K"),
        ("valid_name", "kg,m^-3"),
        ("valid_name", "bad\nunit"),
        ("valid_name", "bad[unit]"),
    ],
)
def test_comsol_parameter_format_rejects_ambiguous_list_syntax(
    name: str,
    unit: str,
) -> None:
    """Reject tokens that could alter COMSOL list or unit parsing."""
    entry = scalar_handoff.ScalarHandoffEntry(
        name=name,
        value=1.25,
        unit=unit,
        owner="case_dependent",
    )
    with pytest.raises(ValueError, match=r"unsafe list syntax|square brackets"):
        scalar_handoff.format_comsol_parameter(entry)


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        ("missing", FileNotFoundError, "missing or unreadable"),
        ("symlink", ValueError, "escapes or aliases"),
        ("path_escape", ValueError, "canonical scalars.csv"),
        ("invalid_utf8", ValueError, "not readable deterministic CSV"),
        ("hash_mismatch", RuntimeError, "bytes changed"),
        ("size_mismatch", RuntimeError, "bytes changed"),
        ("malformed_header", ValueError, "header or row count"),
        ("missing_row", ValueError, "header or row count"),
        ("extra_row", ValueError, "header or row count"),
        ("duplicate_name", ValueError, "missing, duplicate, unknown"),
        ("wrong_order", ValueError, "missing, duplicate, unknown"),
        ("unknown_name", ValueError, "missing, duplicate, unknown"),
        ("wrong_unit", ValueError, "units do not match"),
        ("malformed_number", TypeError, "not numeric"),
        ("nan", ValueError, "finite"),
        ("positive_infinity", ValueError, "finite"),
        ("negative_infinity", ValueError, "finite"),
        ("noncanonical_number", ValueError, "canonical round-trip"),
    ],
)
def test_scalar_source_admission_fails_closed(
    tmp_path: Path,
    mutation: str,
    error: type[Exception],
    match: str,
) -> None:
    """Reject unsafe bytes and every schema/value drift before returning admission."""
    source, payload = _scalar_case(tmp_path)
    rows = source.read_text(encoding="utf-8").splitlines()
    if mutation == "missing":
        source.unlink()
    elif mutation == "symlink":
        target = tmp_path / "scalar-target.csv"
        source.replace(target)
        source.symlink_to(target)
    elif mutation == "path_escape":
        payload["input_contract"]["scalar"]["filename"] = "../scalars.csv"
    elif mutation == "invalid_utf8":
        source.write_bytes(b"name;value;unit\n\xff\n")
        _refresh_identity(payload, source)
    elif mutation == "hash_mismatch":
        source.write_bytes(source.read_bytes() + b"changed\n")
    elif mutation == "size_mismatch":
        payload["input_files"]["scalars.csv"]["size_bytes"] += 1
    elif mutation == "malformed_header":
        rows[0] = "name;value"
        _rewrite_rows(source, payload, rows)
    elif mutation == "missing_row":
        _rewrite_rows(source, payload, rows[:-1])
    elif mutation == "extra_row":
        _rewrite_rows(source, payload, [*rows, "extra_scalar;99;1"])
    elif mutation == "duplicate_name":
        columns = rows[2].split(";")
        columns[0] = rows[1].split(";")[0]
        rows[2] = ";".join(columns)
        _rewrite_rows(source, payload, rows)
    elif mutation == "wrong_order":
        rows[1], rows[2] = rows[2], rows[1]
        _rewrite_rows(source, payload, rows)
    elif mutation == "unknown_name":
        columns = rows[1].split(";")
        columns[0] = "unknown_scalar"
        rows[1] = ";".join(columns)
        _rewrite_rows(source, payload, rows)
    elif mutation == "wrong_unit":
        columns = rows[6].split(";")
        columns[2] = "K"
        rows[6] = ";".join(columns)
        _rewrite_rows(source, payload, rows)
    elif mutation in {"malformed_number", "nan", "positive_infinity", "negative_infinity", "noncanonical_number"}:
        replacements = {
            "malformed_number": "not-a-number",
            "nan": "nan",
            "positive_infinity": "+inf",
            "negative_infinity": "-inf",
            "noncanonical_number": "1.1250",
        }
        columns = rows[1].split(";")
        columns[1] = replacements[mutation]
        rows[1] = ";".join(columns)
        _rewrite_rows(source, payload, rows)
    else:
        raise AssertionError(f"Unhandled mutation {mutation!r}.")

    with pytest.raises(error, match=match):
        _admit(payload, tmp_path)


def test_scalar_source_unreadable_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate an observed source-read permission error before solver use."""
    source, payload = _scalar_case(tmp_path)
    original = Path.read_bytes

    def reject_source_read(path: Path) -> bytes:
        if path == source:
            raise PermissionError("synthetic unreadable source")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reject_source_read)
    with pytest.raises(ValueError, match="not readable deterministic CSV"):
        _admit(payload, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("sampled_value", "sampled value"),
        ("sampled_unit", "sampled unit"),
        ("recorded_value", "values disagree"),
        ("handoff_disagreement", "envelope disagrees"),
        ("ownership", "ownership"),
    ],
)
def test_scalar_case_provenance_must_agree_with_admitted_bytes(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    """Reject case JSON and runtime-scalar provenance mismatches."""
    _source, payload = _scalar_case(tmp_path)
    name = _EXPECTED_FIELDS[5]
    if mutation == "sampled_value":
        payload["sampled_values"][name] += 1.0
    elif mutation == "sampled_unit":
        payload["sampled_units"][name] = "K"
    elif mutation == "recorded_value":
        payload["scalars"][5]["value"] += 1.0
        payload["scalar_handoff"]["entries"] = copy.deepcopy(payload["scalars"])
    elif mutation == "handoff_disagreement":
        payload["scalar_handoff"]["entries"][5]["value"] += 1.0
    elif mutation == "ownership":
        payload["scalars"][3]["owner"] = "package_fixed"
        payload["scalar_handoff"]["entries"] = copy.deepcopy(payload["scalars"])
    else:
        raise AssertionError(f"Unhandled mutation {mutation!r}.")

    with pytest.raises(ValueError, match=match):
        _admit(payload, tmp_path)
