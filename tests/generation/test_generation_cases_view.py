"""Derived cases view: how a listed row reads the evidence of its case.

Responsibilities:
  - Check that a successful row reads the status and timing every published success carries strictly, for both tasks
  - Check that a transient success reads its canonical final bulk moisture strictly while a steady one reads no HDF5
  - Check that evidence the producer may omit (a stationary case's simulated end, a failed attempt's files) stays optional
  - Check the column layout per task: sampled parameters of the plan's registry only, transient result columns only for the transient task

Design principles:
  - A hand-built case directory states exactly which files exist; no solver or campaign is run
  - The view mirrors admission: what admission requires of a success, the view requires too

This module does NOT:
  - Render whole campaigns from a solver run or decide outcomes
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from src import common
from src.generation.publication import generation_publication_catalog as catalog
from src.generation.publication import generation_publication_storage as storage_service

if TYPE_CHECKING:
    from pathlib import Path

_CASE_ID = "case_0001"


def _measurements(tmp_path: Path, outcome: str, task: str) -> dict[str, object]:
    """Return the view's measurements of the one case below ``tmp_path``."""
    row = {"outcome": outcome, "material_family": "lentil", "case_id": _CASE_ID}
    return catalog._case_measurements(task, row, campaign_directory=tmp_path)  # pyright: ignore[reportPrivateUsage]


def _published_case(tmp_path: Path, *, status: object = None, timing: object = None) -> Path:
    """Write a published result whose status and timing are the given JSON objects, raw text, or absent for ``None``; return the result directory."""
    result = common.paths.resolve_generation_case_result_directory(tmp_path / "materials/lentil" / _CASE_ID)
    evidence = common.paths.resolve_generation_result_evidence_directory(result)
    evidence.mkdir(parents=True)
    for name, content in (("status.json", status), ("timing.json", timing)):
        if content is not None:
            (evidence / name).write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return result


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
@pytest.mark.parametrize("damaged", ["status.json", "timing.json"])
@pytest.mark.parametrize("damage", [None, "{broken"], ids=["absent", "malformed"])
def test_a_successful_row_raises_on_missing_or_malformed_required_evidence(tmp_path: Path, task: str, damaged: str, damage: str | None) -> None:
    """Admission requires both sidecars of either task, so the view names the case and file instead of showing a blank row."""
    sidecars: dict[str, Any] = {"status.json": {"t_stop_exact": 60.5}, "timing.json": {"runtime_s": 12.5}}
    sidecars[damaged] = damage
    result = _published_case(tmp_path, status=sidecars["status.json"], timing=sidecars["timing.json"])

    with pytest.raises(ValueError, match=rf"{_CASE_ID}: {re.escape(str(result / 'evidence' / damaged))}"):
        _measurements(tmp_path, "successful", task)


def test_a_transient_row_reads_its_final_bulk_moisture_strictly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Admission raises on unreadable canonical final bulk moisture, so the view reports a readable value and refuses a corrupted file."""
    result = _published_case(tmp_path, status={"t_stop_exact": 60.5}, timing={"runtime_s": 12.5})
    canonical = result / common.paths.GENERATION_RESULT_HDF5_FILENAME
    canonical.write_bytes(b"not an hdf5 file")

    with pytest.raises(ValueError, match=rf"{_CASE_ID}: {re.escape(str(canonical))}"):
        _measurements(tmp_path, "successful", "transient_drying")

    monkeypatch.setattr(storage_service, "read_transient_final_bulk_moisture", lambda path: 0.118 if path == canonical else -1.0)
    measured = _measurements(tmp_path, "successful", "transient_drying")
    assert (measured["duration_s"], measured["simulated_end_h"], measured["final_bulk_moisture_wb"]) == (12.5, 60.5, 0.118)


def test_a_stationary_row_keeps_the_omitted_simulated_end_optional_and_reads_no_hdf5(tmp_path: Path) -> None:
    """A stationary solve has no time axis, so its status legitimately has no simulated end; the view leaves it blank and needs no case.h5."""
    _published_case(tmp_path, status={"quality_flag_count": 0}, timing={"runtime_s": 12.5})

    assert _measurements(tmp_path, "successful", "steady_flow") == {"duration_s": 12.5, "simulated_end_h": None}


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_a_failed_row_keeps_the_files_an_ended_attempt_may_not_have_written_optional(tmp_path: Path, task: str) -> None:
    """An attempt can end before it writes its record or solver log: the failed row shows blanks instead of raising."""
    attempt = common.paths.resolve_generation_case_attempts_directory(tmp_path / "materials/lentil" / _CASE_ID) / "run_0001"
    attempt.mkdir(parents=True)

    assert _measurements(tmp_path, "failed", task) == {"duration_s": None, "simulated_end_h": None, "solver_log": None}


@pytest.mark.parametrize(
    ("task", "results"), [("steady_flow", []), ("transient_drying", ["simulated_end_h", "final_bulk_moisture_wb"])], ids=["steady", "transient"]
)
def test_the_cases_view_lists_the_tasks_result_columns_and_only_its_sampled_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, results: list[str]
) -> None:
    """Constants, derived values and the transient-only results stay out of a steady view; a sampled parameter is a column in registry order."""
    registry = {
        "kappa_mean": {"classification": "sampled"},
        "eps_min_global": {"classification": "fixed"},
        "r_surf": {"classification": "derived"},
        "k_gr": {"classification": "sampled"},
    }
    plan = {"batches": [{"material_family": "lentil", "scientific_values": {"material": {"parameter_registry": registry}}}]}
    row = {
        "outcome": "pending",
        "material_family": "lentil",
        "case_id": _CASE_ID,
        "origin": "primary",
        "failure": None,
        "parameters": {"kappa_mean": 1.5, "eps_min_global": 0.3, "r_surf": 2.0, "k_gr": 0.25, "T_flow_ref": 300.0},
    }
    monkeypatch.setattr(catalog.lifecycle, "load_campaign_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(catalog, "list_campaign_cases", lambda *_args, **_kwargs: (row,))

    header, line = catalog.campaign_cases_csv(task, "campaign", storage_root=tmp_path).splitlines()

    assert header.split(",") == [
        "material",
        "case_id",
        "origin",
        "outcome",
        "reason_code",
        "detail",
        "duration_s",
        *results,
        "kappa_mean",
        "k_gr",
        "solver_log",
    ]
    assert line.split(",")[-3:] == ["1.5", "0.25", ""]


def test_the_cases_view_unions_the_sampled_parameters_of_materials_and_blanks_a_fixed_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A parameter sampled in one material and fixed in the other is one column in registry order, empty in the rows of the material that fixes it."""
    lentil = {"kappa_mean": {"classification": "sampled"}, "k_gr": {"classification": "fixed"}}
    chickpea = {"kappa_mean": {"classification": "sampled"}, "k_gr": {"classification": "sampled"}}
    plan = {
        "batches": [
            {"material_family": material, "scientific_values": {"material": {"parameter_registry": registry}}}
            for material, registry in (("lentil", lentil), ("chickpea", chickpea))
        ]
    }
    parameters = {"kappa_mean": 1.5, "k_gr": 0.25}
    rows = tuple(
        {"outcome": "pending", "material_family": material, "case_id": _CASE_ID, "origin": "primary", "failure": None, "parameters": parameters}
        for material in ("lentil", "chickpea")
    )
    monkeypatch.setattr(catalog.lifecycle, "load_campaign_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(catalog, "list_campaign_cases", lambda *_args, **_kwargs: rows)

    header, *lines = catalog.campaign_cases_csv("steady_flow", "campaign", storage_root=tmp_path).splitlines()

    assert header.split(",")[-3:] == ["kappa_mean", "k_gr", "solver_log"]
    assert [line.split(",")[-3:] for line in lines] == [["1.5", "", ""], ["1.5", "0.25", ""]]
