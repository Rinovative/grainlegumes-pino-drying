"""Parse training CLI arguments and delegate to the reusable run service.

Responsibilities:
  - Parse config, resume, device, budget, dataset, seed, matched-plan, revision, and stage arguments
  - Report the training result and that training publishes no evaluation artifacts
  - Return a non-zero process result for training failures

Design principles:
  - Parser and dispatch code stays thin and import-light
  - Runtime overrides are forwarded without semantic reinterpretation
  - Training never evaluates a held-out role; ID test and OOD come only from the separate artifact command
  - Material lifecycle failures remain visible to shell and scheduler callers

This module does NOT:
  - Allocate, seed, persist, resume, or train runs. ``experiments.run`` owns lifecycle
  - Build or validate artifacts. ``cli_build_artifacts`` is the explicit held-out evaluation
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from . import cli_device


def _build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser."""
    parser = argparse.ArgumentParser(description="Train a neural operator model from config")
    parser.add_argument(
        "config_path",
        nargs="?",
        type=str,
        help="Experiment YAML. Rerunning it continues an interrupted run automatically; omit it with --resume",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Continue one saved run exactly from its own saved config and last_checkpoint.pt (no CONFIG)",
    )
    cli_device.add_device_argument(parser, default=None, help_prefix="Override run.device")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Set the single-stage epoch budget, or A0's in a matched plan; it names the run, so another budget is another run",
    )
    parser.add_argument(
        "--train-dataset",
        help="Override the exact training Dataset ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Replicate seed replacing run.seed; the run lands in its own seed-N leaf and derives its seed plan from N",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        help="Matched plan: Stage-B rollout window length and stride",
    )
    parser.add_argument(
        "--lr-scale",
        type=float,
        help="Matched plan: continuation learning-rate factor for both B and A1 (paired sweep)",
    )
    parser.add_argument(
        "--revision",
        help="Label of a deliberately separate output location for otherwise identical science (stored as run.revision)",
    )
    parser.add_argument("--train-stages", help="Matched mode only: all or a comma-separated subset of a0,b,a1")
    return parser


def _parse_stage_selection(value: str | None) -> tuple[str, ...] | None:
    """Reject duplicates and return canonical stage order independent of text order; ``all`` and no value select every stage."""
    if value is None or value == "all":
        return None
    tokens = value.split(",")
    if not tokens or any(token not in {"a0", "b", "a1"} for token in tokens) or len(tokens) != len(set(tokens)):
        msg = "--train-stages requires distinct a0,b,a1 tokens in a comma-separated list."
        raise ValueError(msg)
    return tuple(name for name in ("a0", "b", "a1") if name in tokens)


def _print_existing_run_admission(report: dict[str, object]) -> None:
    """Render one actionable, non-interactive fresh-run rejection in two lines."""
    from src.experiments import experiments_console  # noqa: PLC0415

    completed = report.get("completed_epoch")
    active = report.get("active_lock")
    state = report.get("state")
    state_label = ", ".join(f"{name} {'ok' if available else 'missing'}" for name, available in (state if isinstance(state, dict) else {}).items())
    print(f"Run admission rejected: {report.get('reason')}", file=sys.stderr)
    print(
        f"  existing run {experiments_console.display_path(report.get('run_dir'))}: status {report.get('status')},"
        f" saved {completed if completed is not None else 'unknown'} epochs,"
        f" last checkpoint {'ok' if report.get('last_checkpoint_available') else 'missing'},"
        f" best checkpoint {'ok' if report.get('best_checkpoint_available') else 'missing'}, {state_label or 'state unavailable'},"
        f" lock {'present' if active is True else 'absent' if active is False else 'unknown'}, resume {report.get('resume_compatibility')}",
        file=sys.stderr,
    )


def _append_cli_overrides(command: list[str], args: argparse.Namespace) -> None:
    """Append explicit invocation overrides to a reproducible continuation."""
    if args.device is not None:
        command.extend(("--device", shlex.quote(args.device)))
    for option, value in (
        ("--epochs", args.epochs),
        ("--train-dataset", args.train_dataset),
        ("--seed", args.seed),
        ("--rollout-length", args.rollout_length),
        ("--lr-scale", args.lr_scale),
        ("--revision", args.revision),
    ):
        if value is not None:
            command.extend((option, shlex.quote(str(value))))


def _print_matched_stage_failure(error: object, args: argparse.Namespace) -> None:
    """Give the exact failed leaf and a continuation only after saved-state admission."""
    from src.experiments import experiments_run  # noqa: PLC0415

    if not isinstance(error, (experiments_run.MatchedStageExecutionError, experiments_run.MatchedStageInterrupted)):
        msg = "Matched stage recovery requires its structured execution error."
        raise TypeError(msg)
    status = "interrupted" if isinstance(error, experiments_run.MatchedStageInterrupted) else "failed"
    print(f"Matched stage {error.stage} {status}. Run directory: {error.run_dir}", file=sys.stderr)
    command = [
        "python -m src.experiments.cli.cli_train",
        shlex.quote(str(Path(args.config_path).expanduser().resolve())),
        "--train-stages",
        error.stage,
    ]
    _append_cli_overrides(command, args)
    if error.resumable:
        print(f"Saved checkpoint can continue: {error.recovery_reason}", file=sys.stderr)
        print(f"Rerun the same command inside a matching Slurm allocation; it continues this stage: {' '.join(command)}", file=sys.stderr)
    elif not error.run_dir.exists():
        print(f"No resumable leaf exists: {error.recovery_reason}", file=sys.stderr)
        print(f"Retry this stage after fixing the cause, inside a matching Slurm allocation: {' '.join(command)}", file=sys.stderr)
    else:
        print(f"This leaf is not eligible for checkpoint resume: {error.recovery_reason}", file=sys.stderr)
        print(
            f"Inspect or repair the persisted state at {error.run_dir}, "
            "or choose a scientifically distinct configuration before retrying this stage.",
            file=sys.stderr,
        )


def _end_status(result: dict[str, Any]) -> str:
    """Return one stage's end status, naming the budget share at which a diverged stage ended."""
    if result["status"] != "diverged":
        return str(result["status"])
    budget_progress = float((result.get("training_outcome") or {})["budget_progress"])
    return f"diverged at {100.0 * budget_progress:.1f} % of the budget"


def _phase_seconds(outcome: dict[str, Any]) -> dict[str, float]:
    """Sum the training duration parts over the trained stages of one outcome."""
    stages = outcome.get("stage_outcomes")
    results = [stage["result"] for stage in stages.values()] if isinstance(stages, dict) else [outcome["result"]]
    totals: dict[str, float] = {}
    for result in results:
        for name, seconds in result.get("phase_seconds", {}).items():
            totals[name] = totals.get(name, 0.0) + seconds
    return totals


def main(argv: list[str] | None = None) -> int:
    """
    Start a run, or continue an interrupted one, and return its process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` after completed training, ``2`` without a result or with an unrun
        selected stage, ``1`` for a caught training failure, and ``130`` when interrupted.

    Notes
    -----
    The reusable run service owns allocation, persistence, device resolution,
    training, and resume. Training publishes no evaluation artifact: ID test and
    OOD are evaluated only by the separate ``cli_build_artifacts`` call once
    configurations and analysis rules are frozen. Parser usage errors raise
    ``SystemExit`` directly.

    """
    started = time.monotonic()
    args = _build_parser().parse_args(argv)
    if args.config_path is not None and args.resume is not None:
        _build_parser().error("--resume continues a saved run from its own config; rerun CONFIG without --resume to continue it")
    if args.config_path is None:
        if args.resume is None:
            _build_parser().error("config_path is required unless --resume identifies a saved run")
        args.config_path = str(Path(args.resume).expanduser() / "config.yaml")
    try:
        from src.experiments import experiments_run  # noqa: PLC0415

        train_stages = _parse_stage_selection(args.train_stages)
        if args.epochs is not None and args.epochs < 1:
            msg = "--epochs must be positive."
            raise ValueError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
        if args.seed is not None and args.seed < 0:
            msg = "--seed must be a non-negative integer."
            raise ValueError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
        if args.rollout_length is not None and args.rollout_length < 1:
            msg = "--rollout-length must be positive."
            raise ValueError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status

        outcome = experiments_run.run_experiment(
            args.config_path,
            resume=args.resume,
            device=args.device,
            train_stages=train_stages,
            epochs=args.epochs,
            train_dataset=args.train_dataset,
            seed=args.seed,
            rollout_length=args.rollout_length,
            lr_scale=args.lr_scale,
            revision=args.revision,
        )
    except KeyboardInterrupt as error:
        print("Training interrupted.", file=sys.stderr)
        from src.experiments import experiments_run as run_service  # noqa: PLC0415

        if isinstance(error, run_service.MatchedStageInterrupted):
            _print_matched_stage_failure(error, args)
        return 130
    except Exception as error:  # noqa: BLE001
        from src.experiments import experiments_console  # noqa: PLC0415
        from src.experiments import experiments_run as run_service  # noqa: PLC0415

        if isinstance(error, run_service.ExistingRunAdmissionError):
            _print_existing_run_admission(error.report)
            return 1
        if isinstance(error, run_service.SourceBindingError):
            print(f"Run admission rejected: {error}", file=sys.stderr)
            return 1
        from src.experiments import experiments_run_registry as run_registry  # noqa: PLC0415

        if isinstance(error, run_registry.IdenticalExperimentError):
            print(f"Run admission rejected: {error}", file=sys.stderr)
            return 1
        print("Training failed. Sanitized traceback follows.", file=sys.stderr, flush=True)
        experiments_console.print_sanitized_traceback(error)
        if isinstance(error, run_service.MatchedStageExecutionError):
            _print_matched_stage_failure(error, args)
        return 1

    from src.experiments import experiments_console as console  # noqa: PLC0415

    result = outcome["result"]
    run_dir = outcome["run_dir"]
    outcome_record = result.get("training_outcome") or {}
    # The terminal stage decides the exit code, matched or not: 2 without a result (no eligible checkpoint),
    # and 2 when a matched plan stopped before a selected stage (an A0 without an eligible checkpoint).
    exit_code = 2 if result.get("best_epoch") is None or outcome.get("unrun_stages") else 0
    if result.get("best_epoch") is None:
        best = f" | unsuccessful: no eligible checkpoint ({outcome_record.get('eligible_points', 0)} eligible validation points)"
    else:
        best = f" | best epoch {result['best_epoch']} | best metric {result['best_metric']:.3e}"
    if result["status"] == "diverged":
        best += f" | {_end_status(result)}"
    print(f"Training status: {result['status']}{best} | run {console.display_path(run_dir)}")
    stage_runs = outcome.get("stage_runs")
    stage_outcomes = outcome.get("stage_outcomes") or {}
    if isinstance(stage_runs, dict):
        for name, stage_dir in stage_runs.items():
            display_name = name.upper() if "stage_outcomes" not in outcome else name
            # The status line names only the last stage; each trained stage names its own end, so a diverged A0 stays visible.
            end = f" | {_end_status(stage_outcomes[name]['result'])}" if name in stage_outcomes else ""
            print(f"Stage {display_name} run directory: {console.display_path(stage_dir)}{end}")
    # Training never evaluates a held-out role; the stage-6 release of ID test and OOD is the explicit artifact command.
    durations = console.duration_summary(time.monotonic() - started, _phase_seconds(outcome))
    print(f"Post-training artifacts: none, ID-test and OOD are evaluated only by cli_build_artifacts | {durations}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
