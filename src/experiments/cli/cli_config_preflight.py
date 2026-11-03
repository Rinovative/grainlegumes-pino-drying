"""Expose host-safe configuration workflow preflight on the command line.

Responsibilities:
  - Parse the requested train or Optuna workflow, configuration path, optional resume target, and matched stage selection
  - Delegate semantic validation and emit a tab-delimited configuration summary ending in the bound source tree or ``-``
  - Return exit status 2 for validation failures or unsafe summary fields

Design principles:
  - Configuration services are imported only after argument parsing
  - Output fields reject tabs and newlines before machine-readable publication

This module does NOT:
  - Resolve scientific configuration independently of the preflight service
  - Allocate runs, submit jobs, or start training or optimization
"""

from __future__ import annotations

import argparse
import sys

_NO_SOURCE_TREE = "-"


def _build_parser() -> argparse.ArgumentParser:
    """Build the config-family preflight parser."""
    parser = argparse.ArgumentParser(
        description="Classify and strictly validate a train or Optuna config without runtime allocation",
    )
    parser.add_argument("workflow", choices=("train", "optuna"))
    parser.add_argument("config_path")
    parser.add_argument("--epochs", type=int, help="Explicit single-stage, A0 or HPO epoch budget")
    parser.add_argument("--train-dataset")
    parser.add_argument("--train-seconds", type=float)
    parser.add_argument("--rollout-length", type=int)
    parser.add_argument("--lr-scale", type=float)
    parser.add_argument("--revision", help="Label of a deliberately separate output location")
    parser.add_argument("--resume", help="Exact run directory for explicit training resume")
    parser.add_argument("--train-stages", help="Matched mode only: all or a comma-separated subset of a0,b,a1")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate one requested workflow and emit a tab-delimited host summary."""
    args = _build_parser().parse_args(argv)
    try:
        from src.experiments.config import experiments_config_preflight as preflight  # noqa: PLC0415

        result = preflight.validate_workflow(
            args.config_path,
            requested_workflow=args.workflow,
            epochs=args.epochs,
            train_dataset=args.train_dataset,
            train_seconds=args.train_seconds,
            rollout_length=args.rollout_length,
            lr_scale=args.lr_scale,
            revision=args.revision,
            resume=args.resume,
            train_stages=None if args.train_stages in {None, "all"} else tuple(args.train_stages.split(",")),
        )
    except Exception as error:  # noqa: BLE001
        print(str(error), file=sys.stderr)
        return 2
    values = (result.family, result.task, result.canonical_path, result.run_label, result.bound_source_tree or _NO_SOURCE_TREE)
    if any("\t" in value or "\n" in value for value in values):
        print("Configuration preflight fields contain unsupported control characters.", file=sys.stderr)
        return 2
    print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
