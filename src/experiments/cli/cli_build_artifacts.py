"""Parse artifact-build arguments and delegate evaluable-run orchestration.

Responsibilities:
  - Validate task, run selection, paths, rebuild intent, and device policy
  - Report material discovery, inference, or publication failures as exit 1
  - Request atomic target replacement only through explicit ``--rebuild``

Design principles:
  - This explicit call is the only evaluation of the held-out roles (ID test and OOD), and it evaluates only the roles
    ``--split`` names; a bounded build names exactly one
  - Parser and dispatch code stays thin, import-light, and service-agnostic
  - Runtime device policy is forwarded unchanged for service-owned resolution
  - Material service failures remain observable through the process exit code

This module does NOT:
  - Admit, generate, cache, or publish artifacts. ``analysis`` owns those services
  - Inspect run contents or render scientific outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import cli_device

_HELD_OUT_ROLES = ("id", "material_ood", "parameter_ood")


def _build_parser() -> argparse.ArgumentParser:
    """Build the artifact-generation parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate or validate the held-out artifacts (ID test, material OOD, parameter OOD) of terminal evaluable runs; "
            "training never evaluates these roles, so this explicit call is their release."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--task",
        default=None,
        help="Registered task whose folder below the experiments root is the run discovery root.",
    )
    selection.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory containing runs, or one current run directory.",
    )
    selection.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        type=Path,
        default=None,
        help="Exact current training-run directory (an Optuna trial has no artifacts and is refused). May be repeated.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Current dataset root, independent from output/run paths.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=None,
        help="Selected run name under the resolved runs root. May be repeated.",
    )
    cli_device.add_device_argument(parser, default="auto")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Deliberately regenerate and atomically replace existing selected artifact targets; not required for initial generation.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--one-case",
        action="store_true",
        help="Generate the first saved case only for a disposable transient UI/debug artifact.",
    )
    scope.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        default=None,
        help="Exact saved transient case ID for a disposable scoped artifact. May be repeated.",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=_HELD_OUT_ROLES,
        default=None,
        help=(
            "Held-out role this call releases (id is the ID test); repeat it for several roles. Required: no role is "
            "evaluated unless named, and a bounded build (--one-case or --case-id) names exactly one."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="New noncanonical output directory required for bounded generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse artifact arguments and return the delegated process result.

    Parameters
    ----------
    argv : list[str] | None, optional
        Explicit argument vector. ``None`` uses the process arguments.

    Returns
    -------
    int
        ``0`` after all selected runs are validated, or ``1`` after a caught
        discovery, inference, generation, or publication failure.

    Notes
    -----
    ``argparse`` usage errors raise ``SystemExit`` before delegation. Runtime
    services are imported only after parsing succeeds.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    task = args.task
    runs_root = args.runs_root
    try:
        from src import analysis, common, domain  # noqa: PLC0415

        if task is not None:
            domain.tasks.registry.get_task(task)
            runs_root = common.paths.get_experiments_root() / task
        if runs_root is None and args.run_dirs is None:
            parser.error("exactly one of --task, --runs-root, or --run-dir is required")
        dataset_root = args.dataset_root if args.dataset_root is not None else common.paths.get_datasets_root()
        if args.run_dirs is not None and args.run_names is not None:
            parser.error("--run-name selects storage aliases below --task/--runs-root and cannot accompany --run-dir")
        if args.run_dirs is not None:
            selected_roots = args.run_dirs
        elif runs_root is not None:
            selected_roots = [runs_root]
        else:
            parser.error("artifact selection unexpectedly lacks a run root")
        if not args.splits:
            parser.error(f"an artifact build evaluates a held-out role only when --split names it; choose from {', '.join(_HELD_OUT_ROLES)}")
        if len(set(args.splits)) != len(args.splits):
            parser.error("--split names each held-out role at most once")
        bounded = bool(args.one_case or args.case_ids)
        if bounded:
            if args.run_dirs is None or len(args.run_dirs) != 1:
                parser.error("bounded generation requires exactly one --run-dir")
            if args.output_root is None:
                parser.error("bounded generation requires --output-root")
            if args.rebuild:
                parser.error("bounded generation never replaces an existing output; omit --rebuild")
            if len(args.splits) != 1:
                parser.error("bounded generation names exactly one held-out role with --split")
            scoped = analysis.artifacts.service.build_scoped_transient_artifact(
                run_dir=args.run_dirs[0],
                dataset_root=dataset_root,
                output_root=args.output_root,
                split=args.splits[0],
                case_ids=args.case_ids,
                one_case=args.one_case,
                device_policy=args.device,
            )
            print(f"artifacts done: scoped transient artifact for {len(scoped.case_ids)} case(s) in {scoped.root}")
            return 0
        if args.output_root is not None:
            parser.error("--output-root requires --one-case or --case-id")

        result_count = 0
        for selected_root in selected_roots:
            results = analysis.artifacts.service.build_artifacts(
                runs_root=selected_root,
                dataset_root=dataset_root,
                roles=tuple(args.splits),
                run_names=args.run_names,
                device_policy=args.device,
                rebuild=args.rebuild,
            )
            result_count += len(results)
    except Exception as error:  # noqa: BLE001
        print(f"Artifact generation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"artifacts done: {result_count} run(s) validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
