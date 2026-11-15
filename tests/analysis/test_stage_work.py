"""Protect the work table that reads completed training stages from their persisted summaries.

Responsibilities:
  - Check one work row per stage for a steady run and for a matched transient A0, B and A1
  - Check the overrun of a units budget and of a seconds budget
  - Check that each stage reports its persisted end status, a diverged one included
  - Check that a value a summary does not record reads unavailable, never zero
  - Check that attempt evidence skips attempts without optimisation seconds and marks a computing attempt that lacks it

Design principles:
  - Synthetic summaries reuse the producers' own records: the budget state and the sampler's unit evidence

This module does NOT:
  - Train, write run directories or open evaluation artifacts
"""

from __future__ import annotations

from typing import Any

import pytest

from src import learning
from src.analysis.evaluation.evaluation_plot import evaluation_plot_run_summary as run_summary
from src.datasets.runtime import dataset_runtime_units as units

_PRECISION = {"parameter_dtypes": ["complex64", "float32"], "autocast_dtype": None, "tf32_matmul": False, "tf32_cudnn": True}


def _session(
    memory_batch: int, *, gpu: str | None = None, elapsed: float | None = 10.0, train_seconds: float = 1.0, driver: str | None = "550.54.15"
) -> dict[str, Any]:
    """Return one runtime session as a run records it: device facts, driver, precision and the attempt entry (none for ``elapsed=None``)."""
    cuda = {} if gpu is None else {"cuda_index": 0, "cuda_device_name": gpu, "cuda_build_version": "12.1"}
    ended = elapsed is not None
    return {
        "memory_batch": memory_batch,
        "device_type": "cpu" if gpu is None else "cuda",
        **cuda,
        "cuda_driver_version": None if gpu is None else driver,
        "precision": _PRECISION if ended else None,
        "attempt": {
            "ended_at": "2026-11-11T00:00:00+00:00" if ended else None,
            "elapsed_seconds": elapsed,
            "train_seconds": train_seconds if ended else None,
        },
    }


def _summary(task: str, budget: learning.training.budget.StageBudget, sessions: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Return a completed summary with the fields the work table reads."""
    sampler = units.UnitSampler(unit_kind="case", cases=(0, 0, 1, 2), keys=None, units_per_update=2, memory_batch=1, seed=0, full_set_units=8)
    sampler.begin_epoch(0)
    sampler.advance(2)
    known = all(session["attempt"]["elapsed_seconds"] is not None for session in sessions)
    return {
        "task": task,
        "run_name": f"{task}-run",
        "model_kind": "fno",
        "model_parameter_counts": {"total": 12, "trainable": 10},
        "training_work": {"optimizer_steps": 4, "gradient_predictions": 30, "units": budget.units, "update_distinct_cases": 6},
        "training_units": learning.training.loop.unit_evidence(sampler),
        "terminal_budget": budget.state_dict(),
        "runtime_sessions": sessions,
        "elapsed_seconds": 25.0 if known else None,
        "attempted_train_seconds": budget.train_seconds + 0.5 if known else None,
        "duration_coverage": "complete" if known else "unknown_attempts",
        **extra,
    }


def _budget(unit: str, total: float, spends: tuple[tuple[int, float], ...]) -> learning.training.budget.StageBudget:
    """Return a stage budget after committing groups of (units, optimisation seconds)."""
    budget = learning.training.budget.StageBudget(unit=unit, total=total)  # type: ignore[arg-type]
    for group_units, seconds in spends:
        budget.record_group(units=group_units, train_seconds=seconds)
    return budget


def test_work_rows_of_a_steady_run_and_of_the_matched_transient_stages() -> None:
    """Each stage reads its seconds, runtime, work, units and overrun in its own budget unit; steady forward transitions are its supervised ones."""
    controller = {"arm": "B", "forward_transitions": 90}
    stages = {
        "steady": (_summary("steady_flow", _budget("units", 6, ((4, 1.0), (4, 1.5))), [_session(4), _session(8)]), 0.01),
        "A0": (
            _summary("transient_drying", _budget("units", 4, ((4, 2.0),)), [_session(4, gpu="GPU")], terminal_controller={**controller, "arm": "A0"}),
            0.2,
        ),
        "B": (
            _summary("transient_drying", _budget("train_seconds", 10.0, ((2, 10.5),)), [_session(4, gpu="GPU")], terminal_controller=controller),
            0.2,
        ),
        "A1": (
            _summary(
                "transient_drying",
                _budget("train_seconds", 10.5, ((2, 11.0),)),
                [_session(4, gpu="GPU", elapsed=None), _session(4, gpu="GPU")],
                terminal_controller={**controller, "arm": "A1"},
            ),
            None,
        ),
    }

    table = run_summary.build_work_table(stages)

    steady, a0, b, a1 = (table.loc[label].to_dict() for label in ("steady", "A0", "B", "A1"))
    assert (steady["stage"], steady["optimisation_seconds"], steady["seen_units"], steady["budget"], steady["overrun"]) == ("steady", 2.5, 8, 6, 2)
    assert steady["forward_transitions"] == steady["supervised_transitions"] == 30
    assert (steady["device"], steady["driver"], steady["cuda_build"], steady["memory_batch"]) == ("cpu", "unavailable", "unavailable", "4; 8")
    assert steady["precision"] == "complex64+float32, autocast off, TF32 matmul/cuDNN off/on"
    assert (steady["unit_kind"], steady["train_units"], steady["full_set_units"], steady["train_cases"], steady["seen_cases"]) == ("case", 4, 8, 3, 1)
    assert (steady["units_per_update"], steady["inference_seconds_per_case"], steady["trainable_parameters"]) == (2, 0.01, 10)
    assert (a0["stage"], a0["overrun"], a0["budget_unit"], a0["forward_transitions"]) == ("A0", 0, "units", 90)
    assert (b["stage"], b["budget_unit"], b["overrun"], b["device"], b["driver"], b["cuda_build"]) == (
        "B",
        "train_seconds",
        pytest.approx(0.5),
        "GPU",
        "550.54.15",
        "12.1",
    )
    assert (b["total_seconds"], b["attempted_seconds"], b["time_coverage"]) == (25.0, 11.0, "complete")
    # An attempt that ended without its end entry leaves the totals unknown; no artifact timing leaves inference unknown.
    assert (a1["total_seconds"], a1["attempted_seconds"], a1["time_coverage"]) == ("unavailable", "unavailable", "unknown_attempts")
    assert (a1["inference_seconds_per_case"], a1["overrun"]) == ("unavailable", pytest.approx(0.5))
    assert a1["precision"] == "unavailable; complex64+float32, autocast off, TF32 matmul/cuDNN off/on"


def test_each_stage_reports_its_persisted_end_status() -> None:
    """A stage that diverged at 30 % of its seconds reads diverged from its training outcome, beside a completed one."""
    outcome = {"budget_progress": 0.3, "eligible_points": 1, "divergence": "non-finite loss"}
    stages = {
        label: (
            _summary(
                "transient_drying",
                _budget("train_seconds", 10.0, ((2, 3.0),)),
                [_session(4, gpu="GPU")],
                terminal_controller={"arm": "B"},
                training_outcome={**outcome, "status": status},
            ),
            None,
        )
        for label, status in (("diverged", "diverged"), ("completed", "completed"))
    }

    table = run_summary.build_work_table(stages)

    assert table["status"].to_dict() == {"diverged": "diverged", "completed": "completed"}


def test_attempt_evidence_marks_a_computing_attempt_without_it_and_skips_an_attempt_that_computed_nothing() -> None:
    """A GPU attempt without its driver leaves the driver partly unavailable; an attempt of zero optimisation seconds adds nothing."""
    sessions = [_session(4, gpu="V100", train_seconds=0.0, driver=None), _session(4, gpu="GPU", driver=None), _session(4, gpu="GPU")]
    summary = _summary("transient_drying", _budget("train_seconds", 10.0, ((2, 10.5),)), sessions, terminal_controller={"arm": "B"})

    row = run_summary.build_work_table({"B": (summary, None)}).loc["B"].to_dict()

    assert (row["device"], row["driver"], row["cuda_build"]) == ("GPU", "unavailable; 550.54.15", "12.1")


def test_a_summary_without_work_evidence_reads_unavailable_never_zero() -> None:
    """A summary that records none of the work quantities yields a row of unavailable values."""
    row = run_summary.build_work_table({"bare": ({"task": "steady_flow"}, None)}).loc["bare"].to_dict()

    assert row.pop("stage") == "steady"
    assert set(row.values()) == {"unavailable"}
