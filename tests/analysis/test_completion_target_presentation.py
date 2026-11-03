"""Protect consolidated completion and target-attainment presentation semantics.

Responsibilities:
  - Check completion denominators, omissions, case details, and censoring classification

Design principles:
  - Eligible cases define each dataset's denominator without changing raw evidence

This module does NOT:
  - Determine solver stopping policy or generate completion evidence
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import domain
from src.analysis.eda import eda_transient as transient
from src.analysis.eda.plots import eda_plot_transient as plots
from src.datasets.contracts import dataset_contracts_transient as contract


def _case_row(*, reached: bool, position: int) -> dict[str, object]:
    """Return one minimal valid transient semantic row."""
    state_names = tuple(field.name for field in contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
    states = {name: np.full((2, 2, 2), position + index + 1.0, dtype=np.float32) for index, name in enumerate(state_names)}
    final_wet_fraction = 0.04 if reached else min(0.98, 0.70 + 0.02 * position)
    return {
        "state_trajectories": states,
        "static_fields": {
            "x": np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            "y": np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        },
        "boundary_intervals": {},
        "scalar_conditioning": {},
        "time": {
            "regular_state_hours": np.asarray((0.0, 168.0)),
            "valid_state_mask": np.asarray((True, True)),
            "trajectory_length": 2,
            "configured_horizon_hours": 168.0,
            "classification_tolerance_hours": 1.0e-9,
        },
        "exact_stop": None,
        "meta": {
            "material_family": "lentil",
            "dataset_role": "training",
            "case_family": "id",
        },
        "completion": {
            "target_reached": reached,
            "right_censored": not reached,
            "physical_duration_hours": 24.0 if reached else 168.0,
            "time_to_target_hours": 24.0 if reached else None,
            "final_wet_fraction": final_wet_fraction,
            "target_wet_fraction_limit": 0.05,
            "final_bulk_moisture_wb": 0.10 if reached else 0.28,
            "target_moisture_wb": 0.12,
        },
        "runtime": {
            "stationary_airflow_solver_seconds": 2.0,
            "transient_drying_solver_seconds": 7.0,
            "scientific_solver_seconds": 9.0,
            "comsol_process_seconds": 12.0,
            "queue_wait_seconds": 1.0,
            "licence_wait_seconds": 0.5,
            "generation_compute_end_to_end_seconds": 13.0,
            "complete_execution_seconds": 14.0,
        },
    }


def _frame(
    outcomes: tuple[bool, ...],
    *,
    total: int | None = None,
    failed: int = 0,
    incomplete: int = 0,
    exclusions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build one small validated frame with explicit discovery accounting."""
    rows = [_case_row(reached=reached, position=position) for position, reached in enumerate(outcomes)]
    columns = (
        "state_trajectories",
        "static_fields",
        "boundary_intervals",
        "scalar_conditioning",
        "time",
        "exact_stop",
        "meta",
        "completion",
        "runtime",
    )
    frame = pd.DataFrame(
        rows,
        columns=columns,
        index=pd.Index(
            [f"case_{position + 1:04d}" for position in range(len(rows))],
            name="sample_id",
        ),
    )
    task = domain.tasks.registry.get_task("transient_drying")
    frame.attrs.update(
        {
            "task_id": task.id,
            "task_contract_digest": task.contract_digest,
            "total_discovered_case_count": len(rows) if total is None else total,
            "failed_case_count": failed,
            "incomplete_case_count": incomplete,
            "exclusion_reasons": dict(exclusions or {}),
            "case_accounting_scope": "test_owned_discovery",
        }
    )
    return frame


def test_outcomes_use_eligible_denominator_and_report_omissions_separately() -> None:
    """Exclude non-evaluable cases without plotting an Excluded outcome."""
    frame = _frame(
        (True, True, True, False),
        total=6,
        failed=1,
        exclusions={"invalid_or_corrupt": 1},
    )
    analysis = transient.completion_target_analysis({"Lentil · Natural": frame})
    rows = analysis.outcomes
    assert tuple(rows["outcome"]) == ("Reached target", "Right-censored")
    assert "Excluded" not in set(rows["outcome"])
    assert rows["eligible_case_count"].eq(4).all()
    np.testing.assert_allclose(rows["percentage"], (75.0, 25.0))
    assert rows["percentage"].sum() == pytest.approx(100.0)
    assert tuple(rows["count"]) == (3, 1)
    omission = analysis.omissions.iloc[0]
    assert omission["omitted_case_count"] == 2
    assert omission["omission_reasons"] == {
        "invalid_or_corrupt": 1,
        "failed_simulation": 1,
    }


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("time_to_target_hours", True),
        ("time_to_target_hours", float("nan")),
        ("time_to_target_hours", -1.0),
        ("time_to_target_hours", 25.0),
        ("physical_duration_hours", -1.0),
        ("final_wet_fraction", 1.01),
        ("target_wet_fraction_limit", -0.01),
    ],
)
def test_invalid_completion_evidence_is_omitted_from_the_denominator(
    field: str,
    invalid_value: object,
) -> None:
    """Keep malformed timing and fraction evidence out of scientific outcomes."""
    frame = _frame((True,))
    completion = frame.loc["case_0001", "completion"]
    assert isinstance(completion, dict)
    completion[field] = invalid_value

    analysis = transient.completion_target_analysis({"Lentil · Natural": frame})
    assert analysis.outcomes["eligible_case_count"].eq(0).all()
    assert analysis.outcomes["percentage"].isna().all()
    omission = analysis.omissions.iloc[0]
    assert omission["omitted_case_count"] == 1
    assert omission["omission_reasons"] == {"invalid_completion_evidence": 1}


def test_machine_roundoff_at_the_fraction_boundary_remains_eligible() -> None:
    """Keep canonical one-ULP fraction roundoff without admitting real overflow."""
    frame = _frame((False,))
    completion = frame.loc["case_0001", "completion"]
    assert isinstance(completion, dict)
    completion["final_wet_fraction"] = 1.0 + math.ulp(1.0)

    analysis = transient.completion_target_analysis({"Lentil · Natural": frame})
    assert analysis.outcomes["eligible_case_count"].eq(1).all()
    assert analysis.omissions.iloc[0]["omitted_case_count"] == 0
    assert analysis.cases.iloc[0]["final_wet_fraction"] == (1.0 + math.ulp(1.0))


def test_multiple_datasets_have_independent_percentage_denominators() -> None:
    """Compute percentages independently for every selected dataset."""
    analysis = transient.completion_target_analysis(
        {
            "Lentil · Natural": _frame((True, False)),
            "Chickpea · Parameter Ood": _frame((True, False, False, False)),
        }
    )
    first = analysis.outcomes.loc[analysis.outcomes["dataset"] == "Lentil · Natural"]
    second = analysis.outcomes.loc[analysis.outcomes["dataset"] == "Chickpea · Parameter Ood"]
    assert first["eligible_case_count"].eq(2).all()
    assert second["eligible_case_count"].eq(4).all()
    np.testing.assert_allclose(first["percentage"], (50.0, 50.0))
    np.testing.assert_allclose(second["percentage"], (25.0, 75.0))
    assert first["percentage"].sum() == pytest.approx(100.0)
    assert second["percentage"].sum() == pytest.approx(100.0)


def test_zero_eligible_cases_retain_an_explicit_empty_denominator() -> None:
    """Keep an empty eligible denominator explicit and non-numeric."""
    frame = _frame((), total=2, failed=1, incomplete=1)
    analysis = transient.completion_target_analysis({"Lentil · Natural": frame})
    assert analysis.outcomes["eligible_case_count"].eq(0).all()
    assert analysis.outcomes["count"].eq(0).all()
    assert analysis.outcomes["percentage"].isna().all()


def test_cases_keep_authoritative_moisture_wet_fraction_gap_and_runtimes() -> None:
    """Retain exact direct quantities and the validated signed derived gap."""
    cases = transient.completion_target_analysis({"Lentil · Natural": _frame((True, False))}).cases
    wet = cases.iloc[1]
    assert wet["final_wet_fraction"] > 0.5
    assert wet["target_wet_fraction_limit"] == pytest.approx(0.05)
    assert wet["final_target_gap"] == pytest.approx(wet["final_wet_fraction"] - wet["target_wet_fraction_limit"])
    assert wet["final_bulk_moisture_wb"] == pytest.approx(0.28)
    assert wet["target_moisture_wb"] == pytest.approx(0.12)
    assert wet["comsol_process_seconds"] == pytest.approx(12.0)
    assert wet["complete_execution_seconds"] == pytest.approx(14.0)


def test_cases_follow_the_selected_dataset_cases() -> None:
    """Retain exact evidence for every selected eligible case."""
    cases = transient.completion_target_analysis(
        {
            "Lentil · Natural": _frame((True, False, True)),
            "Chickpea · Parameter OOD": _frame((False,)),
        }
    ).cases
    assert len(cases) == 4
    assert tuple(cases["dataset"].value_counts().sort_index()) == (1, 3)


def test_right_censored_case_is_classified_and_plotted() -> None:
    """Classify a horizon stop without target attainment and still build the figure."""
    frame = _frame((False,))
    time = frame.loc["case_0001", "time"]
    completion = frame.loc["case_0001", "completion"]
    assert isinstance(time, dict)
    assert isinstance(completion, dict)
    time["regular_state_hours"] = np.asarray((0.0, 85.0))
    completion["physical_duration_hours"] = 85.0

    analysis = transient.completion_target_analysis({"Drying · Lentil · ID": frame})
    figure = plots.plot_completion_target_analysis(datasets={"Drying · Lentil · ID": frame})
    plt.close(figure)

    assert analysis.cases.iloc[0]["terminal_time_kind"] == "right_censoring_time"
