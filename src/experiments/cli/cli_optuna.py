"""Run or validate config-driven Optuna studies from the command line.

Responsibilities:
  - Parse Optuna YAML paths and runtime overrides
  - Print dry-run study summaries without starting training
  - Delegate study execution to experiments.tuning.optuna and keep the Optuna library logger at WARNING

Design principles:
  - CLI code stays thin and side-effect-light
  - Search spaces live in YAML files
  - Runtime overrides are explicit command-line options
  - The study lines come from the console reporter, not from the library logger

This module does NOT:
  - Create, reopen, or optimize studies. ``experiments.tuning.optuna`` owns lifecycle
  - Parse trial search schemas. ``experiments.tuning.search_space`` owns admission
"""

from __future__ import annotations

import argparse
import json
import sys

from . import cli_device


def _build_parser() -> argparse.ArgumentParser:
    """Build the Optuna CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=("Run additional fresh Optuna trials with held-out objective reports at every actual completed epoch")
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to Optuna YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Validate config, search policy, pruning cadence, device policy, and semantic signature without creating files or importing Optuna"),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of additional fresh trials to run (existing trial history is never resumed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the total epoch budget for every trial (names the study directory: another budget is another study)",
    )
    parser.add_argument("--train-dataset", help="Override the exact training Dataset ID (changes study identity)")
    cli_device.add_device_argument(parser, default=None, help_prefix="Override experiment run.device for all trials")
    parser.add_argument(
        "--revision",
        help="Label of a deliberately separate study location for otherwise identical science (stored as run.revision)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Validate or execute an Optuna study and return its process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` for a successful dry run or study invocation, ``1`` for a caught
        study failure, and ``130`` for an interrupted study.

    Notes
    -----
    Dry-run validation creates no study files and imports no Optuna SDK. Parser
    usage errors still raise ``SystemExit`` through ``argparse``.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        from src.experiments.tuning import optuna  # noqa: PLC0415

        study_config = optuna.load_optuna_study_config(
            args.config_path,
            epochs=args.epochs,
            train_dataset=args.train_dataset,
            revision=args.revision,
        )
        study_config = optuna.with_runtime_overrides(study_config, device=args.device)
        if args.dry_run:
            print(json.dumps(optuna.describe_optuna_study_config(study_config), indent=2))
            return 0
        import optuna as optuna_sdk  # noqa: PLC0415

        # Our own trial and study lines replace the library's INFO lines; its warnings and errors stay.
        optuna_sdk.logging.set_verbosity(optuna_sdk.logging.WARNING)
        optuna.run_optuna_study(study_config, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("Optuna study interrupted.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        from src.experiments import experiments_console  # noqa: PLC0415

        print(f"Optuna study failed: {type(error).__name__}: {experiments_console.sanitized_exception_message(error)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
