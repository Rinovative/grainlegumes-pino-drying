# ruff: noqa: S101, D103, FLY002, PLR2004
"""Verify structural COMSOL scientific solver timing ownership."""

from pathlib import Path

from src.generation.runtime import generation_runtime_comsol_timing as timing

_STATIONARY_TITLE = "Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution (sol1)"
_TRANSIENT_TITLE = "Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution (sol2)"


def _block(title: str, *body: str) -> str:
    return "\n".join((f"<---- {title} -", *body, f"----- {title} >"))


def _complete_transient_log() -> str:
    compile_stationary = _block(
        "Compile Equations: Stationary Airflow Solver in Stationary Airflow/Stationary Airflow Solution (sol1)",
        "Solution time: 0.25 s.",
    )
    nested_stationary = _block(
        "Dependent Variables 1 in Stationary Airflow/Stationary Airflow Solution (sol1)",
        "Solution time: 1 s.",
    )
    stationary = _block(
        _STATIONARY_TITLE,
        nested_stationary,
        "Solution time: 18 s.",
    )
    compile_transient = _block(
        "Compile Equations: Transient Drying Solver in Transient Drying/Transient Drying Solution (sol2)",
        "Solution time: 0.5 s.",
    )
    dependent_transient = _block(
        "Dependent Variables 1 in Transient Drying/Transient Drying Solution (sol2)",
        "Values of variables not solved for: Stationary Airflow Solution (sol1).",
        "Solution time: 1 s.",
    )
    transient = _block(
        _TRANSIENT_TITLE,
        "Time-dependent solver (BDF)",
        dependent_transient,
        "-----      Current Progress:  52 % - Evaluating",
        "Solution time: 523 s. (8 minutes, 43 seconds)",
    )
    return "\n".join((compile_stationary, stationary, compile_transient, transient))


def test_transient_parser_selects_only_structurally_owned_top_level_timings() -> None:
    result = timing.parse_comsol_batch_log_text(
        _complete_transient_log(),
        simulation_profile="transient_drying",
    )

    assert result.status == "complete"
    assert result.stationary_airflow.seconds == 18.0
    assert result.transient_drying.seconds == 523.0
    assert result.scientific_solver_seconds == 541.0
    assert result.solution_time_record_count == 6
    assert result.ignored_non_scientific_timing_count == 4
    assert result.stationary_airflow.candidates[0].block == _STATIONARY_TITLE
    assert result.transient_drying.candidates[0].block == _TRANSIENT_TITLE
    assert result.diagnostics == ()


def test_arbitrary_stationary_text_cannot_change_transient_block_ownership() -> None:
    result = timing.parse_comsol_batch_log_text(
        _complete_transient_log(),
        simulation_profile="transient_drying",
    )

    assert result.transient_drying.candidates == (
        timing.SolutionTimeRecord(
            seconds=523.0,
            block=_TRANSIENT_TITLE,
            line_number=result.transient_drying.candidates[0].line_number,
        ),
    )


def test_duplicate_top_level_phase_timings_are_ambiguous() -> None:
    stationary = _block(
        _STATIONARY_TITLE,
        "Solution time: 18 s.",
        "Solution time: 19 s.",
    )
    transient = _block(_TRANSIENT_TITLE, "Solution time: 523 s.")

    result = timing.parse_comsol_batch_log_text(
        f"{stationary}\n{transient}",
        simulation_profile="transient_drying",
    )

    assert result.status == "ambiguous"
    assert result.stationary_airflow.status == "ambiguous"
    assert result.stationary_airflow.occurrence_count == 2
    assert result.stationary_airflow.seconds is None
    assert result.scientific_solver_seconds is None


def test_successful_log_with_missing_transient_timing_keeps_sum_unavailable() -> None:
    result = timing.parse_comsol_batch_log_text(
        _block(_STATIONARY_TITLE, "Solution time: 18 s."),
        simulation_profile="transient_drying",
    )

    assert result.status == "missing"
    assert result.stationary_airflow.seconds == 18.0
    assert result.transient_drying.status == "missing"
    assert result.scientific_solver_seconds is None


def test_failed_transient_block_retains_only_confirmed_stationary_timing() -> None:
    log = "\n".join(
        (
            _block(_STATIONARY_TITLE, "Solution time: 18 s."),
            f"<---- {_TRANSIENT_TITLE} -",
            "Time-dependent solver (BDF)",
        )
    )

    result = timing.parse_comsol_batch_log_text(log, simulation_profile="transient_drying")

    assert result.status == "missing"
    assert result.stationary_airflow.seconds == 18.0
    assert result.transient_drying.seconds is None
    assert result.scientific_solver_seconds is None
    assert any("Unclosed COMSOL block" in diagnostic for diagnostic in result.diagnostics)


def test_steady_airflow_uses_confirmed_stationary_timing_without_zero_transient() -> None:
    result = timing.parse_comsol_batch_log_text(
        _block(_STATIONARY_TITLE, "Solution time: 18 s."),
        simulation_profile="steady_flow",
    )

    assert result.status == "complete"
    assert result.stationary_airflow.seconds == 18.0
    assert result.transient_drying.status == "not_applicable"
    assert result.transient_drying.seconds is None
    assert result.scientific_solver_seconds == 18.0


def test_authoritative_batch_log_is_not_double_counted_when_stdout_duplicates_it(tmp_path: Path) -> None:
    log = "\n".join(
        (
            _block(_STATIONARY_TITLE, "Solution time: 18 s."),
            _block(_TRANSIENT_TITLE, "Solution time: 523 s."),
        )
    )
    batch_log = tmp_path / "comsol_batch.log"
    stdout = tmp_path / "stdout.log"
    batch_log.write_text(log, encoding="utf-8")
    stdout.write_text(log, encoding="utf-8")

    result = timing.parse_comsol_batch_log(batch_log, simulation_profile="transient_drying")

    assert stdout.is_file()
    assert result.solution_time_record_count == 2
    assert result.scientific_solver_seconds == 541.0


def test_missing_batch_log_returns_nonfatal_missing_evidence(tmp_path: Path) -> None:
    result = timing.parse_comsol_batch_log(
        tmp_path / "absent.log",
        simulation_profile="transient_drying",
    )

    assert result.status == "missing"
    assert result.stationary_airflow.seconds is None
    assert result.transient_drying.seconds is None
    assert result.scientific_solver_seconds is None
    assert result.diagnostics
