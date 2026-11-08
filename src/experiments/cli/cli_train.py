"""Parse training CLI arguments and delegate to the reusable run service.

Responsibilities:
  - Parse config, resume, device, revision, seed, and artifact opt-out arguments
  - Orchestrate completed-run artifact preparation after training ownership ends
  - Return a non-zero process result for training or post-processing failures

Design principles:
  - Parser and dispatch code stays thin and import-light
  - Runtime overrides are forwarded without semantic reinterpretation
  - Training completion and artifact post-processing remain separate facts
  - Material lifecycle failures remain visible to shell and scheduler callers

This module does NOT:
  - Allocate, seed, persist, resume, or train runs. ``experiments.run`` owns lifecycle
  - Discover artifact roles, rebuild incompatible targets, or upload to W&B
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
        help="Experiment YAML. Rerunning it continues an existing run automatically; omit it with --resume",
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
        help="Set the single-stage epoch budget, or A0's epoch budget in a matched plan",
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
        "--train-seconds",
        type=float,
        help="Matched plan: measured B training-seconds target; A1 matches B's receipt",
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
    parser.add_argument(
        "--artifacts",
        default="selected",
        help="Post-training artifacts: selected (default) or none; matched plans also accept all or comma-separated a0,b,a1",
    )
    parser.add_argument(
        "--max-epochs", type=int, help="Operational epoch cap for one selected matched B or A1 stage; with --resume also a steady run"
    )
    return parser


def _parse_stage_selection(value: str | None, *, option: str, allow_policies: bool) -> tuple[str, ...] | str | None:
    """Reject duplicates and return canonical stage order independent of text order."""
    if value is None:
        return None
    if value == "all":
        return None if option == "--train-stages" else "all"
    if allow_policies and value in {"selected", "none"}:
        return value
    tokens = value.split(",")
    if not tokens or any(token not in {"a0", "b", "a1"} for token in tokens) or len(tokens) != len(set(tokens)):
        msg = f"{option} requires distinct a0,b,a1 tokens in a comma-separated list."
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
        f" saved {completed if completed is not None else 'unknown'} epochs, requested {report.get('target_epoch')},"
        f" last checkpoint {'ok' if report.get('last_checkpoint_available') else 'missing'},"
        f" best checkpoint {'ok' if report.get('best_checkpoint_available') else 'missing'}, {state_label or 'state unavailable'},"
        f" lock {'present' if active is True else 'absent' if active is False else 'unknown'}, resume {report.get('resume_compatibility')}",
        file=sys.stderr,
    )


def _append_cli_overrides(command: list[str], args: argparse.Namespace, *, include_limits: bool) -> None:
    """Append explicit invocation overrides to a reproducible continuation."""
    if args.device is not None:
        command.extend(("--device", shlex.quote(args.device)))
    for option, value in (
        ("--epochs", args.epochs),
        ("--train-dataset", args.train_dataset),
        ("--seed", args.seed),
        ("--train-seconds", args.train_seconds),
        ("--rollout-length", args.rollout_length),
        ("--lr-scale", args.lr_scale),
        ("--revision", args.revision),
    ):
        if value is not None:
            command.extend((option, shlex.quote(str(value))))
    if include_limits and args.max_epochs is not None:
        command.extend(("--max-epochs", str(args.max_epochs)))


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
    command.extend(("--artifacts", "none"))
    _append_cli_overrides(command, args, include_limits=True)
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


def _artifact_action_label(role_actions: dict[str, str]) -> str:
    """Summarize ID and OOD artifact actions without role-level verbosity."""
    actions = set(role_actions.values())
    automatic = {"generated", "reused", "rebuilt"}
    if len(actions) == 1 and actions <= automatic:
        return next(iter(actions))
    if actions and actions <= automatic:
        return "mixed"
    msg = f"Unexpected automatic artifact actions: {sorted(actions)}."
    raise RuntimeError(msg)


def _validity_label(label: str, note: str | None) -> str:
    """Replace an artifact status by the invalid-predictions status when the artifacts hold invalid predictions."""
    return label if note is None else f"completed with invalid predictions ({note})"


def _phase_seconds(outcome: dict[str, Any]) -> dict[str, float]:
    """Sum the training duration parts over the trained stages of one outcome."""
    stages = outcome.get("stage_outcomes")
    results = [stage["result"] for stage in stages.values()] if isinstance(stages, dict) else [outcome["result"]]
    totals: dict[str, float] = {}
    for result in results:
        for name, seconds in result.get("phase_seconds", {}).items():
            totals[name] = totals.get(name, 0.0) + seconds
    return totals


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    """
    Start, continue or extend a run and return its process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` after completed training and requested post-processing, ``1``
        for a caught training or artifact failure, and ``130`` when interrupted.

    Notes
    -----
    The reusable run service owns allocation, persistence, device resolution,
    training, and resume. The artifact service owns role discovery, validation,
    generation, and reuse. Parser usage errors raise ``SystemExit`` directly.

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
        from src.experiments.config import experiments_config_loader, experiments_config_transient_plan  # noqa: PLC0415

        train_stages = _parse_stage_selection(args.train_stages, option="--train-stages", allow_policies=False)
        artifact_stages = _parse_stage_selection(args.artifacts, option="--artifacts", allow_policies=True)
        if args.artifacts not in {"selected", "none"}:
            invocation_config = experiments_config_loader.load_authored_config(args.config_path)
            is_saved_resume = (
                args.resume is not None
                and Path(args.config_path).expanduser().resolve() == (Path(args.resume).expanduser() / "config.yaml").resolve()
            )
            is_matched = experiments_config_transient_plan.is_matched_a1_b_config(invocation_config) or (
                is_saved_resume and invocation_config.get("training", {}).get("comparison_arm") is not None
            )
            if not is_matched:
                msg = "--artifacts all or a stage list applies only to matched_a1_b plans."
                raise ValueError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
        if isinstance(train_stages, str):
            msg = "Training stage parser returned an artifact policy."
            raise TypeError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
        if args.max_epochs is not None and args.max_epochs < 1:
            msg = "--max-epochs must be positive."
            raise ValueError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
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
            max_epochs=args.max_epochs,
            epochs=args.epochs,
            train_dataset=args.train_dataset,
            seed=args.seed,
            train_seconds=args.train_seconds,
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
    device_resolution = outcome["device_resolution"]
    best = (
        f" after {result['completed_epoch']} epochs (operational cap)"
        if result["status"] == "budget_unmet"
        else f" | best epoch {result['best_epoch']} | best metric {result['best_metric']:.3e}"
    )
    print(f"Training status: {result['status']}{best} | run {console.display_path(run_dir)}")
    stage_runs = outcome.get("stage_runs")
    if isinstance(stage_runs, dict):
        for name, stage_dir in stage_runs.items():
            display_name = name.upper() if "stage_outcomes" not in outcome else name
            print(f"Stage {display_name} run directory: {console.display_path(stage_dir)}")
    if result["status"] == "budget_unmet":
        committed = result.get("train_seconds")
        target = result.get("target_train_seconds")
        if isinstance(committed, (int, float)) and isinstance(target, (int, float)):
            print(f"Committed train seconds: {committed:.6f} / {target:.6f}")
        else:
            print("Committed/target train seconds: unavailable in this result.")
        print("Increase the reached operational cap, or the selected stage max_epochs, then resume this exact run leaf.")
        from src import common  # noqa: PLC0415

        saved_config = experiments_config_loader.load_yaml(common.paths.resolve_run_config_path(run_dir))
        saved_epochs = int(saved_config["training"]["epochs"])
        continuation_epochs = max(saved_epochs, int(result["completed_epoch"]) + 1)
        continuation = [
            "python -m src.experiments.cli.cli_train",
            "--resume",
            shlex.quote(str(run_dir)),
            "--artifacts",
            "none",
            "--max-epochs",
            str(continuation_epochs),
        ]
        print(f"Continuation inside a matching Slurm allocation: {' '.join(continuation)}")
    if artifact_stages == "none":
        print(
            f"Post-training artifacts: skipped by --artifacts none | {console.duration_summary(time.monotonic() - started, _phase_seconds(outcome))}"
        )
        return 2 if result["status"] == "budget_unmet" else 0
    active_artifact_stage: str | None = None
    active_artifact_dir = run_dir
    artifacts_started = time.monotonic()
    try:
        from src import analysis  # noqa: PLC0415

        stage_outcomes = outcome.get("stage_outcomes")
        if isinstance(stage_outcomes, dict):
            if not isinstance(stage_runs, dict):
                msg = "Matched outcome lacks stage run directories."
                raise TypeError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
            if artifact_stages == "selected":
                requested = tuple(name for name, stage_outcome in stage_outcomes.items() if stage_outcome["result"]["status"] == "completed")
            elif artifact_stages == "all":
                requested = tuple(stage_runs)
            elif isinstance(artifact_stages, tuple):
                requested = artifact_stages
            else:
                msg = "Matched artifact selection is invalid."
                raise TypeError(msg)  # noqa: TRY301 -- thin CLI boundary translates argument-contract failures to process status
            available = []
            unavailable = []
            prepared_runs = []
            for name in requested:
                active_artifact_stage = name
                active_artifact_dir = stage_runs[name]
                try:
                    experiments_run.validate_completed_run(active_artifact_dir)
                except (FileNotFoundError, experiments_run.RunLifecycleError):
                    if result["status"] != "budget_unmet":
                        raise
                    if active_artifact_dir.exists() and experiments_run.read_run_summary(active_artifact_dir).get("status") == "completed":
                        raise
                    unavailable.append(name)
                else:
                    available.append(name)
            for name in available:
                active_artifact_stage = name
                active_artifact_dir = stage_runs[name]
                analysis.artifacts.service.cleanup_runtime(device_resolution.device)
                stage_outcome = stage_outcomes.get(name)
                # Only a stage trained here may have artifacts built for an older best checkpoint; a reused stage is validated as is.
                trained_here = stage_outcome is not None and not stage_outcome.get("already_complete", False)
                prepared = analysis.artifacts.service.load_or_build_run_artifacts(
                    stage_runs[name],
                    rebuild_incompatible=trained_here,
                    device_resolution=device_resolution,
                )
                prepared_runs.append(prepared)
                label = _artifact_action_label(prepared.role_actions)
                print(f"Stage {name} artifacts: {_validity_label(label, analysis.artifacts.service.invalid_prediction_note(prepared))}")
            for name in unavailable:
                print(f"Stage {name} artifacts: unavailable until training completes ({stage_runs[name]})")
            artifact_status = (
                "partial"
                if unavailable
                else _validity_label("completed" if requested else "skipped", analysis.artifacts.service.invalid_prediction_note(*prepared_runs))
            )
        else:
            active_artifact_dir = run_dir
            experiments_run.validate_completed_run(run_dir)
            analysis.artifacts.service.cleanup_runtime(device_resolution.device)
            prepared = analysis.artifacts.service.load_or_build_run_artifacts(
                run_dir,
                rebuild_incompatible=not outcome.get("already_complete", False),
                device_resolution=device_resolution,
            )
            artifact_status = _validity_label(
                _artifact_action_label(prepared.role_actions), analysis.artifacts.service.invalid_prediction_note(prepared)
            )
    except KeyboardInterrupt:
        print("Post-training artifacts: failed", file=sys.stderr)
        if active_artifact_stage is not None:
            print(f"Failed artifact stage: {active_artifact_stage}", file=sys.stderr)
        print(f"Failed artifact run directory: {console.display_path(active_artifact_dir)}", file=sys.stderr)
        print(f"Recovery: python -m src.experiments.cli.cli_build_artifacts --run-dir {shlex.quote(str(active_artifact_dir))}", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        from src import analysis  # noqa: PLC0415

        print("Post-training artifacts: failed", file=sys.stderr)
        if active_artifact_stage is not None:
            print(f"Failed artifact stage: {active_artifact_stage}", file=sys.stderr)
        print(f"Artifact error: {type(error).__name__}: {console.sanitized_exception_message(error)}", file=sys.stderr)
        print(f"Failed artifact run directory: {console.display_path(active_artifact_dir)}", file=sys.stderr)
        recovery = f"python -m src.experiments.cli.cli_build_artifacts --run-dir {shlex.quote(str(active_artifact_dir))}"
        if isinstance(error, analysis.evaluation.artifact_loader.IncompatibleEvaluationArtifactsError):
            recovery += " --rebuild"
        print(f"Recovery: {recovery}", file=sys.stderr)
        return 1

    durations = console.duration_summary(time.monotonic() - started, {**_phase_seconds(outcome), "artifacts": time.monotonic() - artifacts_started})
    print(f"Post-training artifacts: {artifact_status} ({device_resolution.device}) | {durations}")
    return 2 if result["status"] == "budget_unmet" else 0


if __name__ == "__main__":
    raise SystemExit(main())
