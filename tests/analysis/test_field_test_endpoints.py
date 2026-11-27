"""Protect the field-test endpoint contract: local patches, input regions, same-hour alignment, windows and the horizon.

Responsibilities:
  - Show that a compact steady difference passes whole-bed relative L2 but fails its interior patch endpoint, and that
    input maps are reported in every region with their wall-minus-core contrast
  - Show that transient states and increments are compared only at stored hours and intervals both cases hold, with the
    initial hour reported only
  - Show that increment tolerances are frozen and looked up per stored window end
  - Show that the wet fraction is compared at the end of the shared stored window, whatever the configured horizon, and
    is unavailable without a stored hour both cases hold

Design principles:
  - Cases are written and read back in the Source ``case.h5`` format; tolerances come from fixture validation pairs

This module does NOT:
  - Exercise rosters, seals or verdict rules
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from support import field_test_fixtures as fixtures

from src.analysis import field_tests

if TYPE_CHECKING:
    from pathlib import Path

TRANSIENT = "transient_drying"


def test_local_patch_difference_fails_patch_endpoint_but_passes_bed_l2(tmp_path: Path) -> None:
    table = field_tests.tolerances.build_tolerance_table("steady_flow", fixtures.validation_pairs("steady_flow"))
    tolerance = {name: entry["tolerance"] for name, entry in table["entries"].items()}
    kwargs = fixtures.case_kwargs("steady_flow", 0)
    reference = field_tests.endpoints.read_case_fields(fixtures.write_case_h5(tmp_path / "reference.h5", **kwargs))
    kwargs["fields"]["p"] = kwargs["fields"]["p"].copy()
    kwargs["fields"]["p"][16:32, 32:48] += 3.0
    kwargs["fields"]["Kxx"] = kwargs["fields"]["Kxx"].copy()
    kwargs["fields"]["Kxx"][:, :4] *= 2.0
    variant = field_tests.endpoints.read_case_fields(fixtures.write_case_h5(tmp_path / "variant.h5", **kwargs))

    measured = field_tests.endpoints.endpoints(reference, variant)

    assert measured["steady/p/relative_l2"].value / tolerance["steady/p/relative_l2"] < 0.5
    assert measured["steady/p/patch_interior"].value / tolerance["steady/p/patch_interior"] > 1.0
    assert measured["steady/u/patch_interior"].value == 0.0
    assert (measured["input/Kxx/patch_wall"].value or 0.0) > 0.0 == measured["input/Kxx/patch_interior"].value
    assert (measured["input/Kxx/wall_core_contrast"].value or 0.0) > 0.0
    assert not any(endpoint.verdict for name, endpoint in measured.items() if name.startswith("input/"))


def test_transient_endpoints_align_on_common_stored_hours(tmp_path: Path) -> None:
    kwargs = fixtures.case_kwargs(TRANSIENT, 0)
    reference = field_tests.endpoints.read_case_fields(fixtures.write_case_h5(tmp_path / "reference.h5", **kwargs))
    shifted = kwargs["states"][:3].copy()
    shifted[2, 0] += 0.5
    variant_kwargs = {**kwargs, "times": kwargs["times"][:3].copy(), "states": shifted, "t_final": 2.0}
    variant = field_tests.endpoints.read_case_fields(fixtures.write_case_h5(tmp_path / "variant.h5", **variant_kwargs))

    measured = field_tests.endpoints.endpoints(reference, variant)

    assert "transient/T/rmse@h=2" in measured
    assert not any(name.endswith(("@h=3", "@h=4")) for name in measured)
    assert measured["transient/T/rmse@h=1"].value == 0.0
    assert measured["transient/T/rmse@h=2"].value == pytest.approx(0.5, abs=1e-4)
    assert (measured["transient/T/rmse@h=0"].value, measured["transient/T/rmse@h=0"].verdict) == (0.0, False)
    cells = fixtures.GRID[0] * fixtures.GRID[1]
    assert measured["transient/T/increment_rms@h=2"].contributors == 2 * cells
    assert measured["transient/T/increment_startup"].value == 0.0
    assert measured["transient/T/increment_peak@h=2"].value == pytest.approx(0.5, abs=1e-4)
    assert measured[field_tests.endpoints.DURATION].value == 1.5


def test_increment_tolerances_are_frozen_and_looked_up_per_window_end(tmp_path: Path) -> None:
    reference = field_tests.endpoints.read_case_fields(fixtures.write_case_h5(tmp_path / "reference.h5", **fixtures.case_kwargs(TRANSIENT, 0)))
    assert reference.states is not None
    # Increments of the prediction error grow with the hour, so a later window has a larger peak.
    growth = (fixtures.HOURS**2)[:, None, None, None] * np.array([0.1, 0.002, 0.5, 0.5])[None, :, None, None]
    prediction = dataclasses.replace(reference, states=(reference.states + growth).astype(np.float32))
    pair = field_tests.tolerances.ValidationPair(fixtures.run_identity(), "case_0000", reference, prediction)

    entries = field_tests.tolerances.build_tolerance_table(TRANSIENT, [pair])["entries"]
    assert prediction.times is not None
    assert prediction.states is not None
    shorter = dataclasses.replace(prediction, times=prediction.times[:3], states=prediction.states[:3])
    measured = field_tests.endpoints.endpoints(reference, shorter)

    peak = "transient/T/increment_peak"
    assert entries[f"{peak}@h=2"]["tolerance"] < entries[f"{peak}@h=4"]["tolerance"]
    # A pair sharing the hours up to 2 is measured over the window ending at 2, the window its tolerance was frozen for.
    assert measured[f"{peak}@h=2"].value == pytest.approx(entries[f"{peak}@h=2"]["tolerance"], rel=1e-9)
    assert not any(name.endswith("@h=4") for name in measured)


def test_the_horizon_wet_fraction_is_compared_at_the_end_of_the_shared_stored_window(tmp_path: Path) -> None:
    kwargs = fixtures.case_kwargs(TRANSIENT, 0)

    def read(name: str, **change: Any) -> field_tests.endpoints.CaseFields:
        case = {**kwargs, "horizon": 100.0, **change}
        return field_tests.endpoints.read_case_fields(fixtures.write_case_h5(tmp_path / f"{name}.h5", **case))

    reference = read("reference", t_final=4.0, wet_fractions=np.array([0.9, 0.7, 0.5, 0.3, 0.1]))
    # Both cases stop long before the configured horizon; the window they both store ends at hour 2.
    shorter = read("shorter", times=kwargs["times"][:3], states=kwargs["states"][:3], t_final=2.5, wet_fractions=np.array([0.9, 0.6, 0.4]))
    shifted = read("shifted", times=kwargs["times"] + 0.5)
    wet = field_tests.endpoints.WET_FRACTION_HORIZON

    measured = field_tests.endpoints.endpoints(reference, shorter)[wet]
    assert (measured.value, measured.verdict) == (pytest.approx(0.1), False)
    # Without a stored hour both cases hold there is no state to compare: unavailable.
    assert field_tests.endpoints.endpoints(reference, shifted)[wet].value is None
