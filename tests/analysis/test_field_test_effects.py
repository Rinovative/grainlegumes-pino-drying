"""Protect the exploratory field-test effect analysis from its quantity definitions to its write-once publication.

Responsibilities:
  - Check each transient and steady quantity of interest, the phase interpolation on stored hours (a censored case
    included), phase points outside the stored window, paired fields at different stop times, the duration censoring
    bounds on both informative sides, and the case contract the reader enforces with the case path
  - Check scales (censored Train cases counted and excluded), resolution floors, the per-comparison flags, the 20 x rule
    against a complete floor only, and the level, base and row summaries: supporting bases, tied winners, one-base medians
  - Check the inventory reconstruction against a persisted synthetic inventory and its refusal on any difference, on an
    exact field, a threshold side or a header identity, and that no call reaches ``build_verdicts`` or ``judge_field_test``
  - Check the Train binding of the yardstick runs (run identity, Dataset identity, membership, population) and the scale
    each base material is measured against
  - Check trigger evidence, the disposition order, both share sets with the coordinate's winning flag, the range claims
    and flags, stall attribution on a known identical CPU with crossings counted only for the frozen rule's minimum, the
    tri-state threshold proximity, the read-only sensitivity, the fail-closed stall check (900 s only without the frozen
    Source), the steady reachability order through its wrapper, and a distinct plot key per QoI in both roles
  - Check write-once publication (a committed directory is verified and never written), the admitted combination of both
    tasks with its containment and recomputed binding digest, the bound definitions, every module the analysis calls in
    its code projection, and that no initializer or endpoint projection names the module

Design principles:
  - Pure definitions are driven with small hand-made arrays whose expected values follow from the plan's formulas
  - End-to-end runs use the synthetic field tests of ``support.field_test_fixtures``, with fakes only at the run, split,
    Dataset manifest and Source admission boundaries, so the binding comparisons themselves run

This module does NOT:
  - Read production field tests, run Generation or train models
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import pytest
from support import field_test_fixtures as fixtures
from support import import_closure

from src import common
from src.analysis import field_tests
from src.analysis.field_tests import field_tests_effects as effects

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

STEADY = "steady_flow"
TRANSIENT = "transient_drying"
_COMPLETE = {"value": 0.1, "complete": True}
_TARGET = (0.12, 0.05, 10.0)


def _attempt(classification: str | None, cpu: str | None, crossing: float | None, *, reason: str | None = None) -> effects.Attempt:
    """Return one attempt record on the draws (1, 1)."""
    return effects.Attempt("m", 1, classification, reason, cpu, "host", (1, 1), 3600.0, crossing, "crossing clock")


def _run(path: str) -> dict[str, str]:
    """Return the admitted identity of one fixture run."""
    return {**fixtures.run_identity(), "path": path}


def _drying(size: tuple[int, int] = (3, 4)) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return five stored hours, drying-like states of ``size`` and their global diagnostics."""
    times = np.arange(5, dtype=np.float64)
    states = np.zeros((5, 4, *size))
    states[:, 0] = 300.0 + 2.0 * times[:, None, None]
    states[:, 0, 1, 2] += 5.0
    states[:, 2] = np.linspace(50.0, 150.0, size[0] * size[1]).reshape(size)[None] * (1.0 - 0.1 * times[:, None, None])
    states[:, 3] = 200.0
    diagnostics = {"f_wet_dm": 1.0 - 0.1 * times, "X_wb_bulk": 0.2 - 0.01 * times, "T_out_mean": 290.0 + times, "phi_out_mean": 0.5 + 0.0 * times}
    return times, states, diagnostics


def test_transient_quantities_use_the_case_phase_on_stored_hours_only() -> None:
    times, states, diagnostics = _drying()
    rho = np.full((3, 4), 2.0)

    case = effects.transient_qois(times, states, diagnostics, 0.5, rho, t_stop=6.0, reached=True, target=_TARGET, keep_fields=True)

    # tau 0.25 -> 1.5 h lies between the stored hours 1 and 2; tau 0.5 -> 3 h is a stored hour; tau 0.75 -> 4.5 h is past the last one.
    assert case.values["T1/duration"] == 6.0
    assert case.values["T2/f_wet_dm@tau=0.25"] == pytest.approx(0.85)
    assert case.values["T2/X_wb_bulk@tau=0.5"] == pytest.approx(0.17)
    assert case.values["T4/T_out_mean@tau=0.25"] == pytest.approx(291.5)
    assert case.values["T5/T_bed_max@tau=0.25"] == pytest.approx(308.0)
    surface = np.linspace(50.0, 150.0, 12).reshape(3, 4) * 0.85
    moisture = (0.5 * surface + 0.5 * 200.0) / 2.0
    assert case.values["T3/moisture_spread@tau=0.25"] == pytest.approx(np.percentile(moisture, 95) - np.percentile(moisture, 5))
    assert {name for name, reason in case.missing.items() if reason == "beyond_last_stored_hour"} == {
        f"{code}/{name}@tau=0.75" for code, name in effects._PHASE_QOIS
    }
    assert case.supplementary == pytest.approx({"bed_T_max_stored": 313.0, "T_out_mean_time_mean": 292.0, "phi_out_mean_time_mean": 0.5})
    assert set(case.fields) == {f"{name}@tau={tau}" for tau in ("0.25", "0.5") for name in ("T", "phi", "w_surf", "w_int")}

    # A censored case is measured on its stored window too, at its own stop.
    censored = effects.transient_qois(times, states, diagnostics, 0.5, rho, t_stop=4.0, reached=False, target=(0.12, 0.05, 4.0))
    assert (censored.values["T2/f_wet_dm@tau=0.75"], censored.missing) == (pytest.approx(0.7), {})
    # A phase point just after the last stored hour is missing: nothing extends the stored window.
    beyond = effects.transient_qois(times, states, diagnostics, 0.5, rho, t_stop=(4.0 + 1.0e-7) / 0.75, reached=True, target=_TARGET)
    assert beyond.missing["T5/T_bed_max@tau=0.75"] == "beyond_last_stored_hour"
    assert effects.bracket(times, -1.0) == "before_first_stored_hour"
    # Only an exact stored hour is read directly; a phase point just before one is interpolated.
    assert effects.bracket(times, 2.0) == (2, 2, 0.0)
    assert effects.bracket(times, 2.0 - 1.0e-7) == (1, 2, pytest.approx(1.0 - 1.0e-7, rel=0.0, abs=1.0e-12))


def test_paired_phase_fields_are_read_at_each_case_s_own_stop() -> None:
    times, states, diagnostics = _drying((16, 16))
    rho = np.full((16, 16), 2.0)
    reference = effects.transient_qois(times, states, diagnostics, 0.5, rho, t_stop=4.0, reached=True, target=_TARGET, keep_fields=True)
    variant = effects.transient_qois(times, states, diagnostics, 0.5, rho, t_stop=2.0, reached=True, target=_TARGET, keep_fields=True)

    # At tau 0.5 the reference is read at 2 h and the variant at 1 h: the bed is 2 K cooler everywhere.
    assert variant.fields["T@tau=0.5"] - reference.fields["T@tau=0.5"] == pytest.approx(np.full((16, 16), -2.0))
    assert effects.paired_delta("T5/T_bed_max@tau=0.5", reference, variant)[0] == pytest.approx(-2.0)
    assert effects.locality_index(reference.fields["T@tau=0.5"], variant.fields["T@tau=0.5"]) == pytest.approx(1.0)


def test_steady_quantities_and_unavailable_non_uniformity() -> None:
    x = np.linspace(0.0, 2.0, 40)
    fields = {"p": np.zeros((3, 40)), "u": np.ones((3, 40)), "v": np.zeros((3, 40))}
    fields["p"][0] = 10.0 + x
    fields["v"][-1] = 2.0 * x
    fields["speed"] = np.hypot(fields["u"], fields["v"])

    case = effects.steady_qois(fields, x)

    speed = fields["speed"]
    assert case.values["S1/pressure_drop"] == pytest.approx(11.0)
    assert case.values["S2/outlet_v"] == pytest.approx(2.0)
    assert case.values["S3/flow_nonuniformity"] == pytest.approx(speed.std() / speed.mean())
    assert case.values["S4/wall_core_speed"] == pytest.approx(field_tests.endpoints._wall_core_contrast(speed))
    still = effects.steady_qois({name: np.zeros((3, 40)) for name in ("p", "u", "v", "speed")}, x)
    assert still.missing == {"S3/flow_nonuniformity": "mean_speed_below_1e-12"}


def test_the_case_reader_refuses_a_broken_case_contract_with_the_case_path(tmp_path: Path) -> None:
    kwargs = fixtures.case_kwargs(TRANSIENT, 0)
    path = fixtures.write_case_h5(tmp_path / "case.h5", **kwargs)
    assert effects._read_case(path, common.serialization.file_sha256(path), TRANSIENT).values["T1/duration"] == 3.5
    with pytest.raises(ValueError, match="bound digest"):
        effects._read_case(path, "0" * 64, TRANSIENT)

    with h5py.File(path, "r+") as handle:
        diagnostics = cast("h5py.Dataset", handle["global/values"])
        diagnostics[1, json.loads(str(diagnostics.attrs["field_names"])).index("T_out_mean")] = np.nan
    with pytest.raises(ValueError, match=r"case\.h5 breaks .*finite"):
        effects._read_case(path, common.serialization.file_sha256(path), TRANSIENT)

    states = kwargs["states"].copy()
    states[3, 2, 0, 0] = -1.0
    negative = fixtures.write_case_h5(tmp_path / "negative.h5", **{**kwargs, "states": states})
    with pytest.raises(ValueError, match=r"negative\.h5 breaks .*non-negative water"):
        effects._read_case(negative, common.serialization.file_sha256(negative), TRANSIENT)

    unordered = fixtures.write_case_h5(tmp_path / "unordered.h5", **{**kwargs, "times": kwargs["times"][[0, 2, 1, 3, 4]]})
    with pytest.raises(ValueError, match=r"unordered\.h5 breaks .*strictly increasing"):
        effects._read_case(unordered, common.serialization.file_sha256(unordered), TRANSIENT)


@pytest.mark.parametrize(
    ("reached", "expected"),
    [
        ((True, True), (2.0, None, None)),
        ((True, False), (None, ("lower", 2.0), "censored_variant")),
        ((False, True), (None, ("upper", 2.0), "censored_reference")),
        ((False, False), (None, None, "both_censored")),
    ],
)
def test_duration_censoring_gives_bounds(reached: tuple[bool, bool], expected: tuple[Any, Any, Any]) -> None:
    reference = effects.CaseQoIs({"T1/duration": 5.0}, {}, reached=reached[0], target=_TARGET)
    variant = effects.CaseQoIs({"T1/duration": 7.0}, {}, reached=reached[1], target=_TARGET)
    assert effects.paired_delta("T1/duration", reference, variant) == expected
    other_target = effects.CaseQoIs({"T1/duration": 7.0}, {}, reached=True, target=(0.13, 0.05, 10.0))
    assert effects.paired_delta("T1/duration", reference, other_target)[2] == "target_or_horizon_differs"


def test_a_bound_reports_only_the_exceedance_it_establishes() -> None:
    scale = {"value": 2.0, "resolved": True}
    lower = effects.effect_entry(None, scale, _COMPLETE, censoring=("lower", 1.0), reason="censored_variant")
    assert (lower["d"], lower["bound"]["d_at_least"], lower["bound"]["band_at_least"]) == (None, 0.5, "substantial")
    # An upper bound above zero establishes no minimum |delta|; one below zero establishes |delta| > 1 on the informative side.
    upper = effects.effect_entry(None, scale, _COMPLETE, censoring=("upper", 1.0), reason="censored_reference")
    assert upper["bound"]["d_at_least"] is None
    informative = effects.effect_entry(None, scale, _COMPLETE, censoring=("upper", -1.0), reason="censored_reference")
    assert (informative["bound"]["d_at_least"], informative["bound"]["band_at_least"]) == (0.5, "substantial")


def test_non_finite_derived_values_leave_the_reductions_with_their_reason() -> None:
    case = effects._case_qois({"Q": float("inf"), "R": 1.0}, {})
    assert (case.values, case.missing) == ({"R": 1.0}, {"Q": "non-finite value"})
    overflow = effects.paired_delta("R", effects.CaseQoIs({"R": -1.0e308}, {}), effects.CaseQoIs({"R": 1.0e308}, {}))
    assert overflow == (None, None, "non-finite difference")
    one_side = effects.paired_delta("Q", case, effects.CaseQoIs({"Q": 1.0}, {}))
    assert one_side == (None, None, "reference: non-finite value")


def test_scales_floors_flags_and_the_twenty_times_rule() -> None:
    value, support = effects.robust_scale([1.0, 2.0, 3.0, 4.0, 100.0])
    assert (value, support) == (pytest.approx(1.4826), 5)
    assert effects.robust_scale([]) == (None, 0)
    floor = effects.resolution_floor([0.01, -0.02], planned=2)
    assert (floor["value"], floor["complete"]) == (0.02, True)
    incomplete = effects.resolution_floor([0.01, None], planned=2)
    assert not incomplete["complete"]

    assert effects.scale_status(1.0, floor) == (True, None)
    assert effects.scale_status(0.3, floor) == (False, "scale below 20 x floor")
    # The 20 x test needs a complete floor; without one the scale stays resolved and the test reads «resolution unknown».
    assert effects.scale_status(0.3, incomplete) == (True, "20 x floor test: resolution unknown (floor incomplete)")
    assert effects.scale_status(0.0, floor)[1] == "scale zero or not finite"

    resolved = effects.effect_entry(-0.01, {"value": 1.0, "resolved": True}, floor)
    assert (resolved["d"], resolved["within_resolution"]) == (0.01, True)
    unknown = effects.effect_entry(0.5, {"value": 1.0, "resolved": True}, incomplete)
    assert (unknown["d"], unknown["within_resolution"]) == (0.5, effects.RESOLUTION_UNKNOWN)
    undefined = effects.effect_entry(0.5, {"value": 0.0, "resolved": False}, floor)
    assert (undefined["d"], undefined["reason"]) == (None, "scale zero or not finite")
    assert effects.band(0.01, resolved=True) == "negligible"
    assert effects.band(0.01, resolved=False) == "below 0.05, scale not resolved"
    assert [effects.band(d, resolved=True) for d in (0.05, 0.2)] == ["small", "substantial"]


def test_censored_train_cases_are_counted_and_excluded_from_every_scale() -> None:
    population = [
        ("a", "lentil", effects.CaseQoIs({"Q": 1.0}, {}, reached=True)),
        ("b", "lentil", effects.CaseQoIs({"Q": 3.0}, {}, reached=True)),
        ("c", "lentil", effects.CaseQoIs({"Q": 100.0}, {}, reached=False)),
    ]

    scales, support = effects._scales(population, ["Q"], {"Q": {"value": 0.0, "complete": True}}, ["lentil"])

    assert (scales["lentil"]["Q"]["n"], scales[effects.POOLED]["Q"]["value"]) == (2, pytest.approx(1.4826))
    assert (support["planned"], support["censored_excluded"], support["observed"]["Q"]) == (3, 1, 2)


def _entry(row: str, level: str, base: str, material: str, d: float | None, *, valid: bool = True, reference: str = "R0") -> dict[str, Any]:
    """Return one comparison-table entry with one QoI."""
    entry: dict[str, Any] = {"member_id": f"{row}-{level}-{base}-{reference}", "row": row, "level": level, "reference": reference}
    entry.update(material=material, base=base, problem=None if valid else "unsealed")
    if valid:
        entry["effects"] = {"Q": {"delta": d, "d": d, "within_resolution": d is not None and d < 0.01, "scale_resolved": True, "reason": None}}
    return entry


def test_level_base_and_row_summaries_flag_incomplete_bases_and_publish_one_base_medians() -> None:
    table = [
        _entry("r", "low", "l/1", "lentil", 0.1),
        _entry("r", "high", "l/1", "lentil", 0.3),
        _entry("r", "low", "l/2", "lentil", 0.2),
        _entry("r", "high", "l/2", "lentil", 0.0, valid=False),
        _entry("r", "high", "k/9", "kidney_bean", 0.4),
    ]

    levels = effects._level_summaries(table, {"Q": {"value": 0.5, "complete": True}}, ["Q"])
    headlines = effects._headlines(table, ["Q"])
    summary = effects._row_summary(headlines["r"], ["lentil"], ["Q"])["Q"]

    low = next(item for item in levels if (item["level"], item["material"]) == ("low", "lentil"))["qois"]["Q"]
    assert (low["median_d"], low["support"], low["bases"], low["challenges"], low["within_resolution"]) == (
        pytest.approx(0.15),
        2,
        ["l/1", "l/2"],
        2,
        True,
    )
    assert headlines["r"]["l/1"]["qois"]["Q"]["winners"] == [["high", "R0"]]
    assert (headlines["r"]["l/2"]["incomplete"], headlines["r"]["l/1"]["incomplete"]) == (True, False)
    assert summary[effects.POOLED] == {
        "median": pytest.approx(0.25),
        "max": 0.3,
        "planned": 2,
        "observed": 2,
        "incomplete_bases": 1,
        "resolved": True,
        "band": "substantial",
    }
    # One kidney-bean base publishes its median with its support, without a band; an empty group is never resolved.
    assert (summary["kidney_bean"]["observed"], summary["kidney_bean"]["median"], summary["kidney_bean"]["band"]) == (1, 0.4, None)
    empty = effects._row_summary(effects._headlines([_entry("r", "low", "l/3", "lentil", None)], ["Q"])["r"], ["lentil"], ["Q"])["Q"]
    assert (empty[effects.POOLED]["median"], empty[effects.POOLED]["resolved"]) == (None, False)


def test_range_flags_follow_the_best_reference_s_supporting_bases_and_tied_winners() -> None:
    table = [
        _entry("r", "low", "l/1", "lentil", 0.5),
        _entry("r", "high", "l/1", "lentil", 0.5),
        _entry("r", "low", "l/2", "lentil", 0.1),
        _entry("r", "high", "l/2", "lentil", 0.6),
        _entry("r", "low", "l/3", "lentil", 0.3),
        _entry("r", "high", "l/3", "lentil", 0.05),
        _entry("r", "high", "l/3", "lentil", 0.2, reference="R_sin"),
        _entry("r", "low", "l/4", "lentil", 0.25),
        _entry("r", "high", "l/4", "lentil", None, valid=False),
        _entry("r", "high", "l/4", "lentil", 0.2, reference="R_sin"),
        _entry("r", "phase_minus", "l/1", "lentil", 0.4),
        _entry("r", "phase_minus", "l/2", "lentil", 0.4),
    ]
    levels = effects._level_summaries(table, {"Q": _COMPLETE}, ["Q"])
    headlines = effects._headlines(table, ["Q"])
    claim = {"row": "r", "material": "lentil", "level": "high"}

    flags = effects._range_flags("r", levels, headlines["r"], {("r", "lentil", "high"): claim}, ["Q"])

    assert headlines["r"]["l/1"]["qois"]["Q"]["winners"] == [["high", "R0"], ["low", "R0"]]
    # High's best reference R0 rests on l/1-l/3; high wins l/1 (tied) and l/2, so it supplies the maximum at 2 of 3 bases.
    assert flags["bereich_groesser_pruefen"] == [
        {"level": "high", "material": "lentil", "side": "above the ID maximum", "qois": ["Q"], "claim": claim}
    ]
    # Low and the phase level reach 0.2 without a claim and become questions for a new test.
    assert [flag["level"] for flag in flags["frage_fuer_neuen_test"]] == ["low", "phase_minus"]
    without = effects._range_flags("r", levels, headlines["r"], {}, ["Q"])
    assert [flag["level"] for flag in without["frage_fuer_neuen_test"]] == ["high", "low", "phase_minus"]
    assert without["bereich_groesser_pruefen"] == []


def test_range_claims_need_both_documents_a_known_key_a_citation_and_a_value_beyond_the_id_bound(tmp_path: Path) -> None:
    files = {name: tmp_path / name for name in effects.EVIDENCE_DOCUMENTS}
    for path in files.values():
        path.write_text("# evidence\n", encoding="utf-8")
    claim = effects.RangeClaim(value=0.3, unit="W/(m*K)", citation="PARAMETER_EVIDENCE.md, k_gr", id_bound=0.24)
    registry = {"k_gr": {"kind": "interval", "unit": "W/(m*K)", "lower": 0.18, "upper": 0.24}}
    campaign = SimpleNamespace(batches=[SimpleNamespace(material_family="lentil", scientific_values={"material": {"parameter_registry": registry}})])
    good = effects.RangeEvidence(files=files, claims={("k_gr", "lentil", "high"): claim})

    copies, claims = effects._admit_range_evidence(good, TRANSIENT, ["k_gr"], campaign)

    assert sorted(copies) == ["evidence-DECISION_TABLE.md", "evidence-PARAMETER_EVIDENCE.md"]
    assert claims == [{"row": "k_gr", "material": "lentil", "level": "high", **dataclasses.asdict(claim), "id_bound_checked": True}]
    assert effects._admit_range_evidence(good, TRANSIENT, ["k_gr"], None)[1][0]["id_bound_checked"] is False
    refused = [
        {"files": {"PARAMETER_EVIDENCE.md": files["PARAMETER_EVIDENCE.md"]}},
        {"files": {**files, "DECISION_TABLE.md": tmp_path / "absent.md"}},
        {"claims": {("k_gr", "lentil", "high"): dataclasses.replace(claim, citation=" ")}},
        {"claims": {("k_gr", "lentil", "low"): claim}},
        {"claims": {("k_gr", "kidney_bean", "high"): claim}},
        {"claims": {("k_gr ", "lentil", "high"): claim}},
        {"claims": {("k_gr", "lentil", "phase_minus"): claim}},
        {"claims": {("k_gr", "lentil", "high"): dataclasses.replace(claim, id_bound=0.25)}},
    ]
    for change in refused:
        with pytest.raises(ValueError, match="Range evidence"):
            effects._admit_range_evidence(dataclasses.replace(good, **change), TRANSIENT, ["k_gr"], campaign)


def test_trigger_evidence_keeps_endpoint_names_and_comparisons_and_separates_unmeasured_pairs() -> None:
    comparisons = [
        {
            "member_id": "a",
            "problem": None,
            "measured": {
                "transient/T/patch_wall@h=3": [2.0, 4],
                "transient/T/patch_wall@h=8": [3.0, 4],
                "transient/duration": [3.0, 1],
                "transient/T/rmse@h=3": [0.5, 9],
            },
        },
        {"member_id": "b", "problem": "unsealed", "measured": {"transient/phi/rmse@h=3": [9.0, 9]}},
    ]
    limits = dict.fromkeys(["transient/T/patch_wall@h=3", "transient/T/patch_wall@h=8", "transient/duration", "transient/T/rmse@h=3"], 1.0)

    evidence = effects.trigger_evidence(comparisons, limits, failure=True)

    assert evidence["patch"] == [["a", "transient/T/patch_wall@h=3"], ["a", "transient/T/patch_wall@h=8"]]
    assert (evidence["non_patch"], evidence["evidence"], evidence["missing"]) == ([["a", "transient/duration"]], "usable", False)
    unavailable = {"member_id": "c", "problem": None, "measured": {"transient/T/rmse@h=3": ["unavailable", 0]}}
    unmeasured = effects.trigger_evidence([unavailable], limits, failure=False)
    assert (unmeasured["evidence"], unmeasured["missing"]) == ("valid pair without measurements", True)
    assert effects.trigger_evidence([comparisons[1]], limits, failure=False)["evidence"] == "no valid comparison"


@pytest.mark.parametrize(
    ("verdict", "medians", "triggers", "expected"),
    [
        ("remove", {"Q": (0.0, True)}, ([], [], False, False), "not kept in this task"),
        ("keep", {"Q": (None, True)}, (["p"], [], False, False), "indeterminate"),
        ("keep", {"Q": (0.06, True)}, (["p"], [], False, True), "indeterminate"),
        ("keep", {"Q": (0.01, True)}, ([], [], False, False), "indeterminate"),
        ("keep", {"Q": (0.01, True)}, ([], [], True, False), "keep rests on failures only"),
        ("keep", {"Q": (0.06, False)}, (["p"], [], False, False), "acts globally"),
        ("keep", {"Q": (0.01, True)}, (["p"], [], True, False), "acts only locally"),
        ("keep", {"Q": (0.01, True)}, (["p"], ["n"], False, False), "small non-local effect"),
        ("keep", {"Q": (0.01, False)}, (["p"], [], False, False), "insufficient evidence"),
    ],
)
def test_disposition_order(
    verdict: str, medians: dict[str, tuple[float | None, bool]], triggers: tuple[list[str], list[str], bool, bool], expected: str
) -> None:
    patch, other, failure, missing = triggers
    evidence = {"patch": [["m", name] for name in patch], "non_patch": [["m", name] for name in other], "failure": failure, "missing": missing}
    evidence["evidence"] = "valid pair without measurements" if missing else "usable"
    assert effects.disposition(verdict, medians, evidence)[0] == expected


def _cell(d: float | None, within: bool = False) -> dict[str, Any]:
    """Return one headline cell of QoI ``Q``."""
    return {"d": d, "winners": [], "resolved": True, "within_resolution": within}


def test_both_share_sets_merge_the_anisotropy_pair_into_one_coordinate_with_the_winning_flag() -> None:
    rows = ["permeability.anisotropy.max_ratio", "permeability.anisotropy.strength", "permeability.orientation.jitter"]
    # strength is within the floor at l/2 but loses there to max_ratio, so the coordinate carries max_ratio's flag.
    cells = {rows[0]: (_cell(0.1), _cell(0.3)), rows[1]: (_cell(0.4), _cell(0.2, within=True)), rows[2]: (_cell(0.5), _cell(0.5))}
    headlines = {
        row: {base: {"material": "lentil", "incomplete": False, "qois": {"Q": cell}} for base, cell in zip(("l/1", "l/2"), pair, strict=True)}
        for row, pair in cells.items()
    }

    shares = effects._shares({rows[0]: 4, rows[1]: 3, rows[2]: 2}, headlines, ["lentil"], ["Q"])["Permeabilität"]

    raw = shares["raw"]["Q"]["members"]
    assert [raw[row]["median_d"] for row in rows] == pytest.approx([0.2, 0.3, 0.5])
    assert (raw[rows[2]]["share"], raw[rows[1]]["within_resolution"], raw[rows[1]]["challenges"]) == (pytest.approx(0.5), 1, 3)
    coordinate = shares["coordinates"]["Q"]["members"][effects.ANISOTROPY_COORDINATE]
    # The coordinate takes the larger d per base, 0.4 and 0.3 (median 0.35), and counts both rows' challenges.
    assert (coordinate["share"], coordinate["within_resolution"], coordinate["challenges"]) == (pytest.approx(0.35 / 0.85), 0, 7)
    # Every family publishes coordinates; a zero family sum leaves every share undefined.
    flat = {row: {"l/1": {"material": "lentil", "incomplete": False, "qois": {"Q": _cell(0.0)}}} for row in ("porosity.a", "porosity.b")}
    porosity = effects._shares({"porosity.a": 2, "porosity.b": 2}, flat, ["lentil"], ["Q"])["Porosität"]
    assert porosity["coordinates"] == porosity["raw"]
    assert porosity["raw"]["Q"]["members"]["porosity.a"]["share"] is None


def test_stall_attribution_needs_a_known_identical_cpu_and_matched_draws() -> None:
    failing = [_attempt("solver_failure", "cpu-a", None, reason="stalled"), _attempt("solver_failure", "cpu-b", None, reason="stalled")]
    clean = [_attempt("succeeded", "cpu-a", 100.0), _attempt("succeeded", "cpu-b", 200.0)]
    assert effects.attribute_failure(failing, clean, 900.0) == effects.PARAMETER_RELATED
    # A reference that failed on cpu-a and crossed late on cpu-b leaves the challenge contribution open.
    slow = [_attempt("solver_failure", "cpu-a", None), _attempt("succeeded", "cpu-b", 800.0)]
    assert effects.attribute_failure(failing, slow, 900.0) == effects.BASE_OR_HARDWARE
    # Conflicting same-CPU evidence (one failed and one clean matched reference attempt) does too.
    conflicting = [_attempt("solver_failure", "cpu-a", None), _attempt("succeeded", "cpu-a", 100.0)]
    assert effects.attribute_failure(failing[:1], conflicting, 900.0) == effects.BASE_OR_HARDWARE
    # A clean reference only on another CPU model, on other draws, or with an unknown CPU model matches nothing.
    assert effects.attribute_failure(failing, clean[:1], 900.0) == effects.MIXED
    other_draws = dataclasses.replace(clean[0], draws=(2, 1))
    assert effects.attribute_failure(failing[:1], [other_draws], 900.0) == effects.MIXED
    unknown = [_attempt("solver_failure", None, None, reason="stalled")]
    assert effects.attribute_failure(unknown, [_attempt("succeeded", None, 100.0)], 900.0) == effects.MIXED

    assert effects.near_threshold([_attempt("solver_failure", "cpu-a", None, reason="stalled")], 900.0) is True
    assert effects.near_threshold([_attempt("succeeded", "cpu-a", 750.0)], 900.0) is True
    assert effects.near_threshold([_attempt("succeeded", "cpu-a", 100.0)], 900.0) is False
    # Missing crossing evidence, a missing outcome or no reference attempt leave it unknown.
    assert effects.near_threshold([_attempt("succeeded", "cpu-a", None)], 900.0) is None
    assert effects.near_threshold([_attempt("succeeded", "cpu-a", 100.0), _attempt(None, None, None)], 900.0) is None
    assert effects.near_threshold([], 900.0) is None


def test_a_crossing_counts_only_for_the_minimum_of_the_frozen_stall_rule(tmp_path: Path) -> None:
    record = effects.generation.field_tests.record
    attempt = record.attempt_directory(tmp_path, "member-a", 1)
    (attempt / "runtime").mkdir(parents=True)
    outcome = record.new_outcome("member-a", 1, "succeeded", recorded_by="test", cpu_model="cpu-a")
    outcome.update(spatial_support_attempt=1, schedule_acceptance_attempt=1)
    (attempt / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
    progress = {"stall_progress": {"minimum_simulated_seconds": 1800.0, "reached_after_seconds": 100.0}}
    (attempt / "runtime" / "timing.json").write_text(json.dumps(progress), encoding="utf-8")
    bound = cast("Any", SimpleNamespace(directory=tmp_path))

    matched = effects._attempts(bound, "member-a", 1800.0)[0][0]
    other = effects._attempts(bound, "member-a", 3600.0)[0][0]
    unknown = effects._attempts(bound, "member-a", None)[0][0]

    assert (matched.evidence, matched.minimum_simulated, matched.crossing) == ("crossing clock", 1800.0, 100.0)
    # A crossing recorded for another minimum, or with no frozen minimum to compare, keeps both fields but is no evidence.
    for item in (other, unknown):
        assert item.evidence.startswith("unknown")
        assert (item.minimum_simulated, item.crossing, item.clock) == (1800.0, 100.0, None)
    failing = [_attempt("solver_failure", "cpu-a", None, reason="stalled")]
    assert effects.attribute_failure(failing, [matched], 900.0) == effects.PARAMETER_RELATED
    assert effects.attribute_failure(failing, [other], 900.0) == effects.MIXED
    assert (effects.near_threshold([matched], 900.0), effects.near_threshold([other], 900.0)) == (False, None)


def test_every_plotted_qoi_keeps_its_own_key_and_material_ood_only_hollows_it() -> None:
    qois = effects.QOIS[TRANSIENT]
    filled = [effects._style(index, ood=False) for index in range(len(qois))]
    hollow = [effects._style(index, ood=True) for index in range(len(qois))]

    assert len({(style["color"], style["marker"]) for style in filled}) == len(qois)
    for id_style, ood_style in zip(filled, hollow, strict=True):
        assert {**ood_style, "markerfacecolor": id_style["markerfacecolor"]} == id_style
        assert ood_style["markerfacecolor"] == "none" != id_style["markerfacecolor"]


def test_the_sensitivity_keeps_a_surviving_trigger_and_unresolves_a_failure_only_keep() -> None:
    comparison = field_tests.verdicts._Comparison
    base = {"material_family": "lentil", "design_kind": "sobol", "design_index": 1}
    label = field_tests.verdicts.base_label(base)
    failure = {"role": "challenge", "base": label, "material": "lentil", "problem": "challenge_solver_failure", "failure_effect": True}
    comparisons = [
        comparison(member_id="f1", row="mixed", **failure),
        comparison(member_id="v1", row="mixed", role="challenge", base=label, material="lentil", worst_ratio=2.0, worst_endpoint="e", exceeds=True),
        comparison(member_id="f2", row="only", **failure),
    ]
    members: dict[str, Any] = {"ref": {"role": "reference", "base": base}}
    members.update({name: {"role": "challenge", "base": base, "reference_member_id": "ref", "level": "high"} for name in ("f1", "v1", "f2")})
    seals = {"ref": {"spatial_support_attempt": 1, "schedule_acceptance_attempt": 1}}
    persisted = {"rows": [{"row": "mixed", "verdict": "keep"}, {"row": "only", "verdict": "keep"}]}
    bound = SimpleNamespace(comparisons=comparisons, members=members, seals=seals, persisted=persisted)
    stalled = [_attempt("solver_failure", "cpu-a", None, reason="stalled")]
    attempts = {"ref": stalled, "f1": stalled * 2, "f2": stalled * 2, "v1": []}

    stall = effects._stall_analysis(cast("Any", bound), attempts, {"seconds": 900.0, "status": "verified", "execution_config_digest": "e" * 64})

    assert {item["member_id"]: item["attribution"] for item in stall["reproduced_failures"]} == dict.fromkeys(("f1", "f2"), effects.BASE_OR_HARDWARE)
    assert [(item["row"], item["verdict_without_failures"]) for item in stall["sensitivity"]] == [("mixed", "keep"), ("only", "unresolved")]
    assert stall["near_threshold"] == {label: True}


def _raising(error: Exception) -> Any:
    """Return a stand-in that raises ``error`` on every call."""

    def call(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return call


def test_the_stall_check_recovers_the_frozen_rule_and_assumes_900_s_only_without_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    values: dict[str, Any] = {"runtime": {"stall_stop": {"check_after_seconds": 1200.0, "minimum_simulated_seconds": 3600.0}}}
    digest = common.serialization.canonical_json_sha256(values)
    record = {"execution_config_digest": digest}
    monkeypatch.setattr(effects.generation.publication.campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: record)
    bound = cast("Any", SimpleNamespace(task=TRANSIENT, roster={"base_source": {"campaign_name": "source"}}))
    campaign = SimpleNamespace(execution_values=values)

    verified = {"seconds": 1200.0, "minimum_simulated_seconds": 3600.0, "status": "verified", "execution_config_digest": digest}
    assert effects._stall_check(bound, campaign, storage_root=None) == verified
    # A frozen runtime without a stall rule has no check time; nothing is assumed.
    bare = {"runtime": {}}
    record["execution_config_digest"] = common.serialization.canonical_json_sha256(bare)
    no_rule = effects._stall_check(bound, SimpleNamespace(execution_values=bare), storage_root=None)
    assert (no_rule["seconds"], no_rule["minimum_simulated_seconds"], no_rule["status"]) == (None, None, "no stall rule in the frozen Source runtime")
    # A reconstruction that does not reproduce its run record's execution digest refuses.
    record["execution_config_digest"] = "0" * 64
    with pytest.raises(ValueError, match="execution digest"):
        effects._stall_check(bound, campaign, storage_root=None)
    # Only a Source whose frozen files are missing is unrecoverable and falls back to 900 s; a changed plan refuses.
    roster = effects.generation.field_tests.roster
    monkeypatch.setattr(roster, "load_base_campaign", _raising(FileNotFoundError("campaign_plan.json")))
    assert effects._frozen_campaign(bound, storage_root=None) is None
    assumed = effects._stall_check(bound, None, storage_root=None)
    assert (assumed["seconds"], assumed["minimum_simulated_seconds"]) == (900.0, None)
    monkeypatch.setattr(roster, "load_base_campaign", _raising(roster.BaseBindingError("the frozen plan changed")))
    with pytest.raises(roster.BaseBindingError):
        effects._frozen_campaign(bound, storage_root=None)


@pytest.mark.parametrize(
    ("delta", "endpoint", "expected"),
    [
        (1.0e-15, (0.3, 0.9, 0.1), effects.INVARIANT),
        (1.0, (0.3, 0.9, None), effects.RESOLUTION_UNKNOWN),
        (1.0, (None, None, 0.1), effects.UNKNOWN),
        (1.0, (0.1, 0.9, 0.2), "within resolution"),
        (1.0, (0.3, 0.4, 0.1), "resolved, max ratio < 0.5"),
        (1.0, (0.3, 0.7, 0.1), "resolved, 0.5 <= max ratio < 1"),
        (1.0, (0.3, 1.2, 0.1), "resolved, max ratio >= 1"),
    ],
)
def test_reachability_order(delta: float, endpoint: tuple[float | None, float | None, float | None], expected: str) -> None:
    inputs = {name: [delta, 1.0] for name in field_tests.endpoints.STEADY_INPUTS}
    comparison = {"problem": None, "inputs": inputs, "endpoints": {"steady/p/relative_l2": list(endpoint)}}
    assert effects.reachability_class([comparison]) == expected


def test_reachability_applies_the_input_rule_first_and_keeps_missing_comparisons_explicit() -> None:
    inputs = {name: [1.0e-15, 1.0] for name in field_tests.endpoints.STEADY_INPUTS}
    table = [
        {"member_id": "c1", "row": "r", "base": "b", "problem": None, "inputs": inputs, "effects": {}},
        {"member_id": "c2", "row": "r", "base": "b", "problem": "infeasible"},
    ]
    comparisons = [{"member_id": "c1", "measured": {"steady/p/relative_l2": ["unavailable", 0]}}, {"member_id": "c2", "measured": {}}]
    replicates = [{"base": "b", "endpoints": {"steady/p/relative_l2": [0.1, 0.2]}}]
    persisted = {"rows": [{"row": "r", "verdict": "remove"}], "replicates": replicates, "comparisons": comparisons}
    bound = cast("Any", SimpleNamespace(frozen=SimpleNamespace(tolerances={"steady/p/relative_l2": 1.0}), persisted=persisted))

    result = effects._reachability(bound, table)["rows"]["r"]

    # The input rule comes first although an endpoint value is unavailable; the infeasible contrast stays listed.
    assert (result["class"], [item["member_id"] for item in result["comparisons"]]) == (effects.INVARIANT, ["c1", "c2"])
    del inputs["p_in_bc"]
    assert effects._reachability(bound, table)["rows"]["r"]["class"] == effects.UNKNOWN


def test_locality_index_and_patch_coverage() -> None:
    reference = np.zeros((32, 48))
    other = reference.copy()
    other[:16, :16] = 1.0
    # One of six patches moves by 1: the patch mean is 1 and the field RMS is sqrt(1/6).
    assert effects.locality_index(reference, other) == pytest.approx(np.sqrt(6.0))
    assert effects.locality_index(reference, reference) is None
    assert effects.patch_coverage((32, 48)) == 1.0


def test_inventory_admission_is_exact_on_fields_threshold_sides_and_zero() -> None:
    limits = {"e": 1.0}
    record = {"member_id": "a", "worst_ratio": 0.3, "measured": {"e": [1.0 - 1.0e-12, 3]}}
    effects._require_same("comparisons", [record], [{**record, "worst_ratio": 0.3 * (1.0 + 1.0e-12)}], limits)
    # Each persisted variant meets its own reconstruction, so each refusal has exactly its stated cause.
    side_flips = [
        (record, {**record, "measured": {"e": [1.0 + 1.0e-12, 3]}}),
        ({**record, "worst_ratio": 0.5 * (1.0 - 1.0e-12)}, {**record, "worst_ratio": 0.5 * (1.0 + 1.0e-12)}),
    ]
    for reconstructed, persisted in side_flips:
        # Equal within the relative tolerance: only the keep-ratio or removal-margin side refuses.
        assert effects._same(reconstructed, persisted)
        with pytest.raises(ValueError, match="differs from the persisted"):
            effects._require_same("comparisons", [reconstructed], [persisted], limits)
    # A contributor count persisted as a float refuses although every threshold side agrees.
    retyped = {**record, "measured": {"e": [1.0 - 1.0e-12, 3.0]}}
    assert effects._sides(retyped, limits) == effects._sides(record, limits)
    with pytest.raises(ValueError, match="differs from the persisted"):
        effects._require_same("comparisons", [record], [retyped], limits)
    with pytest.raises(ValueError, match="differs from the persisted"):
        effects._require_same("rows", [{"worst_ratio": 0.0}], [{"worst_ratio": 1.0e-300}], limits)


def _challenge(row: str, base: int, scale: float, task: str, **extra: Any) -> dict[str, Any]:
    """Return one challenge of ``row`` at ``base``."""
    level = extra.pop("level", "high")
    return {"row": row, "level": level, "base": base, "case": fixtures.case_kwargs(task, base, scale=scale), **extra}


def _field_test(storage: Path, task: str) -> Path:
    """Write, freeze and judge one synthetic stage-A field test of ``task`` with one unsealed challenge."""
    bases = [("lentil", 1), ("lentil", 2), ("chickpea", 3), ("chickpea", 4)]
    challenges = [
        *(_challenge("row.small", base, 0.1, task) for base in range(4)),
        *(_challenge("row.large", base, 3.0, task, level=level) for base in range(4) for level in ("low", "high")),
        {"row": "row.fails", "level": "high", "base": 0, "case": None, "seal": {"attempts": 2}},
        {"row": "row.unsealed", "level": "high", "base": 1, "case": None},
    ]
    test_dir = fixtures.make_field_test(storage, task=task, stage="A", bases=bases, challenges=challenges)
    (test_dir / "members" / f"c{len(challenges) - 1}-1" / "seal.json").unlink()
    fixtures.freeze(test_dir, task)
    field_tests.verdicts.build_verdicts(task, fixtures.TEST_ID, storage_root=storage)
    return test_dir


def _patch_population(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    task: str,
    *,
    runs: Sequence[Sequence[str]] | None = None,
    manifest_changes: Mapping[str, Any] | None = None,
    identity_changes: Mapping[str, Any] | None = None,
) -> None:
    """Fake the run admission, split admission, Dataset manifest and Source admission over six written Train cases (3 + 3)."""
    paths: dict[str, Path] = {}
    identities = []
    for index in range(6):
        case_id, material = f"train-{index}", ("lentil", "chickpea")[index % 2]
        paths[case_id] = root / task / case_id / "case.h5"
        if not paths[case_id].is_file():
            kwargs = fixtures.case_kwargs(task, index, scale=0.05 * index, **({} if task == STEADY else {"t_final": 3.0 + 0.1 * index}))
            fixtures.write_case_h5(paths[case_id], **kwargs)
        source = {key: f"{key}-{index}" for key in effects._SOURCE_SELECTOR_KEYS}
        source.update(material_family=material, case_hdf5_sha256=common.serialization.file_sha256(paths[case_id]))
        identities.append({"package_case_id": case_id, "source": source, "design": {"design_kind": "train"}})
    roles = {"chickpea": "id", "kidney_bean": "ood", "lentil": "id"}
    partition = {"material_roles": roles, "train": list(paths), **{role: [] for role in ("validation", "id_test", "parameter_ood", "ood")}}
    dataset = {"dataset_id": "train-dataset", "dataset_digest": "d" * 64, "channel_contract_digest": "c" * 64}
    manifest = {**dataset, "source_task": task, "source_case_identities": identities, "case_partition": partition, "grid": {"axes_sha256": "a" * 64}}
    manifest.update(manifest_changes or {})
    memberships = list(runs) if runs is not None else [list(paths)]

    @contextlib.contextmanager
    def lease(run_dir: Path) -> Iterator[dict[str, Any]]:
        admitted = {"run_dir": Path(run_dir), "effective_config_digest": "0" * 64, "selected_checkpoint_sha256": "1" * 64}
        yield {**admitted, "split_indices": {"run": Path(run_dir).name}, **(identity_changes or {})}

    def admit_split(payload: Mapping[str, Any]) -> Any:
        members = tuple(memberships[min(len(memberships) - 1, int(str(payload["run"]).endswith("b")))])
        identity = {**dataset, "case_ids": list(paths), "index_digest": "i" * 64}
        part = SimpleNamespace(dataset=identity, dataset_id="train-dataset", case_ids=members, membership_digest="m" * 64)
        return SimpleNamespace(task=task, train_subset=None, case_partition=partition, role=lambda _name: part)

    def resolve(_task: str, selectors: list[dict[str, Any]], *, storage_root: Any) -> tuple[Any, ...]:
        del storage_root
        by_selector = {entry["source"]["case_id"]: paths[entry["package_case_id"]] for entry in identities}
        return tuple(SimpleNamespace(hdf5_path=by_selector[selector["case_id"]]) for selector in selectors)

    monkeypatch.setitem(effects.TRAIN_POPULATION, task, {"chickpea": 3, "lentil": 3})
    monkeypatch.setattr(effects.experiments.run, "saved_artifact_run_lease", lease)
    monkeypatch.setattr(effects.datasets.preprocessing.splits, "admit_split", admit_split)
    monkeypatch.setattr(effects.datasets.packages, "load_package_manifest_evidence", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(effects.generation.publication.catalog, "resolve_raw_case_references", resolve)


def _forbid_verdict_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every call of the verdict builder or the judge fail."""

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the effect analysis called the verdict builder")

    monkeypatch.setattr(field_tests.verdicts, "build_verdicts", forbidden)
    monkeypatch.setattr(field_tests.workflow, "judge_field_test", forbidden)


def test_the_inventory_reconstruction_matches_and_refuses_any_difference(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    test_dir = _field_test(storage, STEADY)
    bound = effects._bind_inventory(STEADY, fixtures.TEST_ID, storage_root=storage)
    assert len(bound.comparisons) == len(bound.persisted["comparisons"])

    path = test_dir / "analysis" / "verdicts.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    for factor, refused in ((1.0 + 1.0e-12, False), (1.0 + 1.0e-6, True)):
        payload = json.loads(json.dumps(original))
        measured = payload["comparisons"][0]["measured"]
        name = next(key for key, value in measured.items() if isinstance(value[0], float) and value[0] > 0.0)
        measured[name][0] *= factor
        path.write_text(json.dumps(payload), encoding="utf-8")
        if refused:
            with pytest.raises(ValueError, match="differs from the persisted verdict inventory"):
                effects._bind_inventory(STEADY, fixtures.TEST_ID, storage_root=storage)
        else:
            effects._bind_inventory(STEADY, fixtures.TEST_ID, storage_root=storage)
    for key, value in (("test_id", "another-test"), ("stage", "B")):
        path.write_text(json.dumps({**original, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match="does not bind field test"):
            effects._bind_inventory(STEADY, fixtures.TEST_ID, storage_root=storage)
    payload = json.loads(json.dumps(original))
    payload["rows"][0]["verdict"] = "unresolved" if payload["rows"][0]["verdict"] != "unresolved" else "keep"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="row verdicts"):
        effects._bind_inventory(STEADY, fixtures.TEST_ID, storage_root=storage)


def test_the_train_population_binds_run_identities_the_dataset_and_the_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bound = cast("Any", SimpleNamespace(task=TRANSIENT, frozen=SimpleNamespace(payload={"model_runs": [_run("runs/a"), _run("runs/b")]})))
    _patch_population(monkeypatch, tmp_path / "train", TRANSIENT)

    population, membership = effects._train_population(bound, storage_root=tmp_path)

    assert (len(population), membership["population"], membership["id_materials"]) == (6, {"chickpea": 3, "lentil": 3}, ["chickpea", "lentil"])
    assert [run["path"] for run in membership["runs"]] == ["runs/a", "runs/b"]
    refusals: list[tuple[dict[str, Any], str]] = [
        ({"runs": [[f"train-{index}" for index in range(6)], ["train-0"]]}, "disagree"),
        ({"manifest_changes": {"dataset_digest": "e" * 64}}, "Dataset identity"),
        ({"identity_changes": {"effective_config_digest": "9" * 64}}, "identity the tolerance table records"),
    ]
    for changes, message in refusals:
        _patch_population(monkeypatch, tmp_path / "train", TRANSIENT, **changes)
        with pytest.raises(ValueError, match=message):
            effects._train_population(bound, storage_root=tmp_path)
    _patch_population(monkeypatch, tmp_path / "train", TRANSIENT)
    monkeypatch.setitem(effects.TRAIN_POPULATION, TRANSIENT, {"chickpea": 32, "lentil": 32})
    with pytest.raises(ValueError, match="population"):
        effects._train_population(bound, storage_root=tmp_path)
    # Only a declared material-OOD material is measured against the pooled ID benchmark; any other material refuses.
    roles = {"kidney_bean": "ood", "lentil": "id"}
    assert [effects.scale_group(material, roles) for material in ("lentil", "kidney_bean")] == ["lentil", effects.POOLED]
    with pytest.raises(ValueError, match="neither"):
        effects.scale_group("faba_bean", roles)


def _evidence(tmp_path: Path) -> effects.RangeEvidence:
    """Return range evidence over both written documents and no claim."""
    files = {name: tmp_path / name for name in effects.EVIDENCE_DOCUMENTS}
    for name, path in files.items():
        path.write_text(f"# {name}\n", encoding="utf-8")
    return effects.RangeEvidence(files=files)


def _measure_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Measure and publish both synthetic field tests below ``tmp_path / storage``."""
    storage = tmp_path / "storage"
    results = {}
    for task in (TRANSIENT, STEADY):
        _field_test(storage, task)
        _patch_population(monkeypatch, tmp_path / "train", task)
        _forbid_verdict_builds(monkeypatch)
        results[task] = effects.measure_field_test_effects(task, fixtures.TEST_ID, range_evidence=_evidence(tmp_path), storage_root=storage)
        monkeypatch.undo()
    return results


def test_publication_is_write_once_and_a_committed_directory_is_never_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = tmp_path / "storage"
    results = _measure_both(tmp_path, monkeypatch)

    transient, steady = results[TRANSIENT], results[STEADY]
    directory = transient["result_directory"]
    assert directory.parent == storage / "04_field_tests" / TRANSIENT / fixtures.TEST_ID / "analysis"
    assert sorted(path.name for path in directory.iterdir()) == [
        "effects.json",
        "effects.md",
        "effects_families.png",
        "evidence-DECISION_TABLE.md",
        "evidence-PARAMETER_EVIDENCE.md",
        "manifest.json",
    ]
    assert (directory / "effects.md").read_bytes().startswith(effects.LABEL.encode())
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = transient["payload"]
    assert (manifest["label"], manifest["qualifications"]) == (effects.LABEL, list(effects.QUALIFICATIONS))
    assert manifest["input_sha256"] == payload["input_sha256"]
    assert (payload["label"], directory.name) == (effects.LABEL, f"effects-{payload['input_sha256'][:12]}")
    rows = payload["rows"]
    assert (rows["row.small"]["frozen_verdict"], rows["row.small"]["disposition"]) == ("remove", "not kept in this task")
    # The fixture's exhaust air is constant, so its Train spread is zero and its d undefined: rule 2 applies first.
    assert rows["row.large"]["disposition"] == "indeterminate"
    assert steady["payload"]["rows"]["row.large"]["disposition"] == "acts globally"
    # The unsealed challenge is bound with null digests and stays in the table with its reason.
    unsealed = next(entry for entry in payload["comparisons"] if entry["row"] == "row.unsealed")
    assert (unsealed["problem"], payload["binding"]["seals"][unsealed["member_id"]]) == ("unsealed", [None, None])
    assert payload["phase_support"]["T1/duration"] == {"planned": 14, "observed": 12, "missing": {"challenge_solver_failure": 1, "unsealed": 1}}
    assumed = {
        "seconds": 900.0,
        "status": "assumed 900 s: frozen Source campaign not recoverable; without its minimum no crossing counts",
        "execution_config_digest": None,
    }
    assert payload["binding"]["stall_check"] == {**assumed, "minimum_simulated_seconds": None}
    assert payload["stall"]["margin_seconds"] is None  # no crossing can be judged without the frozen minimum
    assert rows["row.fails"]["triggers"]["failure"] is True
    assert payload["stall"]["sensitivity"] == [{"row": "row.fails", "verdict_without_failures": "unresolved", "worst_ratio": None}]
    assert payload["scales"]["lentil"]["T1/duration"]["n"] == 3
    assert steady["payload"]["reachability"]["rows"]["row.small"]["class"] == "input invariant"

    # A rerun on the same evidence reproduces every byte and writes nothing.
    _patch_population(monkeypatch, tmp_path / "train", TRANSIENT)
    again = effects.measure_field_test_effects(TRANSIENT, fixtures.TEST_ID, range_evidence=_evidence(tmp_path), storage_root=storage)
    assert again["result_directory"] == directory
    # A changed or deleted file of a committed directory is refused and never rewritten.
    (directory / "effects.md").write_bytes(b"changed\n")
    with pytest.raises(FileExistsError, match="committed"):
        effects.measure_field_test_effects(TRANSIENT, fixtures.TEST_ID, range_evidence=_evidence(tmp_path), storage_root=storage)
    (directory / "effects.md").unlink()
    with pytest.raises(FileExistsError, match="committed"):
        effects.measure_field_test_effects(TRANSIENT, fixtures.TEST_ID, range_evidence=_evidence(tmp_path), storage_root=storage)
    assert not (directory / "effects.md").exists()
    with pytest.raises(FileExistsError, match="prefix collides"):
        effects._publish(directory, {"manifest.json": b"{}\n"})


def test_the_combination_admits_both_canonical_task_results_and_refuses_anything_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = tmp_path / "storage"
    results = _measure_both(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        effects.combine_field_test_effects([results[STEADY]], storage_root=storage)

    recommendations = effects.combine_field_test_effects(list(results.values()), storage_root=storage)

    assert (recommendations.name, recommendations.parent.parent) == ("recommendations.md", storage / "04_field_tests" / "effects")
    text = recommendations.read_text(encoding="utf-8")
    assert text.startswith(effects.LABEL)
    assert all(result["payload"]["input_sha256"] in text for result in results.values())
    assert "| row.small | remove | remove | weg possible (stage B to confirm) |" in text
    assert "- kappa_mean" in text
    assert effects.combine_field_test_effects(list(results.values()), storage_root=storage) == recommendations
    # A task result whose effects.json no longer holds its manifest's bytes is refused.
    effects_json = results[STEADY]["result_directory"] / "effects.json"
    original = effects_json.read_bytes()
    effects_json.write_bytes(original.replace(b"acts globally", b"acts locally!"))
    with pytest.raises(ValueError, match="manifest records"):
        effects.combine_field_test_effects(list(results.values()), storage_root=storage)
    effects_json.write_bytes(original)
    # The combined directory stays a real directory below the field-test area.
    shutil.rmtree(recommendations.parent.parent)
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage / "04_field_tests" / "effects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link or escapes"):
        effects.combine_field_test_effects(list(results.values()), storage_root=storage)
    # A manifest and payload that repeat the claimed digest over another binding are refused, though every file hash agrees.
    directory = results[STEADY]["result_directory"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads(original)
    for record in (manifest, payload):
        record["binding"]["roster_version"] += 1
    (directory / "effects.json").write_bytes(effects._json_bytes(payload))
    manifest["files"]["effects.json"] = common.serialization.file_sha256(directory / "effects.json")
    (directory / "manifest.json").write_bytes(effects._json_bytes(manifest))
    with pytest.raises(ValueError, match="binding"):
        effects.combine_field_test_effects(list(results.values()), storage_root=storage)


def test_a_row_either_task_keeps_untested_stays_in_the_combined_status() -> None:
    row = {"frozen_verdict": "remove", "disposition": "not kept in this task", "merge": [], "bereich_enger": []}
    row |= {"bereich_groesser_pruefen": [], "frage_fuer_neuen_test": []}
    payloads = {
        STEADY: {"test_id": "s", "input_sha256": "a" * 64, "rows": {"kept": row, "gone": row}, "untested_rows": []},
        TRANSIENT: {"test_id": "t", "input_sha256": "b" * 64, "rows": {"gone": row}, "untested_rows": ["kept"]},
    }

    text = effects._recommendations(payloads, {STEADY: "c" * 64, TRANSIENT: "d" * 64}).decode("utf-8")

    assert "| kept | remove | - | bleibt |" in text
    assert "| gone | remove | remove | weg possible (stage B to confirm) |" in text


def test_the_result_digest_binds_the_complete_definitions_and_every_module_the_analysis_calls() -> None:
    assert {"disposition_order", "range_flags", "stall", "reachability_order", "bands", "scale", "inventory"} <= set(effects.DEFINITIONS)
    assert [label for label, _rule in effects.DEFINITIONS["disposition_order"]][3:5] == ["acts globally", "acts only locally"]
    members = set(effects.EFFECTS_PROJECTION.members())
    source = Path(effects.__file__).parents[2]
    references = import_closure.module_references(import_closure.SourceTree(source), "analysis/field_tests/field_tests_effects.py")
    called = {item.target for item in references if Path(item.target).name != "__init__.py"}
    assert {"analysis/field_tests/field_tests_tolerances.py", "datasets/contracts/dataset_contracts_partition.py"} <= called
    assert called - members == set()


def test_the_effect_module_stays_outside_every_initializer_and_endpoint_projection() -> None:
    source = Path(field_tests.__file__).parents[2]
    initializers = [source / package / "__init__.py" for package in ("analysis", "analysis/field_tests", "generation", "generation/field_tests")]
    assert all("field_tests_effects" not in path.read_text(encoding="utf-8") for path in initializers)
    assert "effects" not in field_tests.__all__
    assert "analysis/field_tests/field_tests_effects.py" not in field_tests.endpoints.ENDPOINT_PROJECTION.members()
