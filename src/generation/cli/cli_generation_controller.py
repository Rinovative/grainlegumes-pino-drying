"""Expose one public Generation workflow command backed by the Python controller.

Responsibilities:
  - Parse the public Generation command and admit the shared source, committed or not
  - Submit a run from the login node as its own driver job and run the controller inside that job
  - Dispatch lifecycle commands to the controller and native host boundary
  - Render run results with their process exit status

Design principles:
  - One public command covers plan, run, status, and cancellation
  - A run freezes the source on the login node and executes only from the frozen tree, like the Dataset plan run
  - Scientific and persisted state remain owned by Generation services
  - Host execution uses the canonical native interpreter and sibling roots

This module does NOT:
  - Implement scientific cases, campaign scheduling, or dataset persistence
  - Load compute-node modules or bypass Slurm for material work
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from src import common
from src.generation import controller as controller_service
from src.generation.contracts import generation_contracts_campaign_identity as campaign_identity
from src.generation.runtime import host as host_service

if TYPE_CHECKING:
    from collections.abc import Sequence

_INTERRUPTED_EXIT_CODE = 130


def _fail(message: str, *, exit_code: int = 1) -> NoReturn:
    """Raise one public command failure with a stable exit status."""
    raise controller_service.GenerationControllerError(message, exit_code=exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./scripts/generation", description="Run and inspect Generation workflows")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="start or resume a campaign as a Slurm driver job")
    run.add_argument("config")
    run.add_argument("--materials", nargs="+", help="select configured materials for this original design")
    run.add_argument("--duration-hours", type=float, help="set the physical transient horizon in hours; preserve the original boundary schedule")
    run.add_argument(
        "--revision",
        metavar="LABEL",
        help="keep changed science beside the existing Source of the same name as <name>__rev-LABEL (source campaigns only); seeds do not change",
    )
    run.add_argument(
        "--node-jobs",
        type=int,
        help=(
            "set concurrent node jobs across all Generation runs, 1 to 4 (execution.yaml default 2; a cluster-share cap, use 3 or 4 only "
            "when the standard nodes are quiet; the COMSOL batch licence counts once per user)"
        ),
    )
    run.add_argument("--cores-per-case", type=int, help="set CPUs for each case step of a node job")
    run.add_argument(
        "--case-timeout-seconds",
        type=float,
        help="set case runtime with a stop reserve capped at 20 percent; the node-job time must still exceed it by 600 s",
    )
    run.add_argument(
        "--no-retry",
        action="store_true",
        help="disable automatic license retries (the node wait and the retry within an allocation)",
    )
    run.add_argument("--dry-run", action="store_true", help="print the resolved plan and check execution readiness; start no work")
    run.add_argument(
        "--continue-with-current-code",
        action="store_true",
        help="continue an existing campaign with the current code instead of its bound source tree (recorded as a new execution epoch)",
    )
    run.add_argument("--max-admission-cases", type=int, help="set the recorded operational limit on fresh candidate admissions")
    run.add_argument("--driver-partition", default="standard", help="Slurm partition of the submitted driver job: standard or long")
    run.add_argument("--driver-time", default="06:00:00", help="time limit of the submitted driver job as [D-]HH:MM:SS")
    status = subcommands.add_parser("status", help="inspect a config or persisted run ID")
    status.add_argument("target")
    cancel = subcommands.add_parser("cancel", help="cancel a campaign")
    cancel.add_argument("run_id")
    cancel.add_argument("--force", action="store_true")
    fail_case = subcommands.add_parser("fail-case", help="declare one exhausted solver/validation candidate terminal")
    fail_case.add_argument("run_id")
    fail_case.add_argument("material")
    fail_case.add_argument("case_id")
    fail_case.add_argument("--reason-code", required=True)
    fail_case.add_argument("--detail", required=True)
    amend = subcommands.add_parser(
        "amend-replacements", help="explicitly lower the minimum success rate of one material, for all its original batches, of a campaign run"
    )
    amend.add_argument("run_id")
    amend.add_argument("--material", required=True)
    amend.add_argument(
        "--minimum-success-rate", type=float, required=True, help="new rate s with 0 < s <= 1; it can only be lowered (the current value is a no-op)"
    )
    amend.add_argument("--reason", required=True, help="why the rate is lowered after the original design was frozen")
    inputs = subcommands.add_parser("inputs", help="generate selected canonical campaign inputs")
    inputs.add_argument("config")
    inputs.add_argument("options", nargs=argparse.REMAINDER)
    return parser


def _inputs_options(options: list[str]) -> bool:
    """Reject options the command owns and return whether the selection is a dry run."""
    if any(item in {"--storage-root", "--git-commit"} or item.startswith(("--storage-root=", "--git-commit=")) for item in options):
        _fail("inputs owns the sibling storage root and the admitted source commit", exit_code=2)
    return "--dry-run" in options


def _source_line(admission: host_service.SourceAdmission) -> str:
    """Describe the admitted source, including whether its snapshot (or, read-only, its checkout) carries uncommitted changes."""
    carrier = "snapshot" if admission.frozen else "checkout"
    state = "carries uncommitted changes" if admission.uncommitted else "is clean"
    return f"Source: shared HEAD {admission.commit} ({carrier} {state}) fingerprint {admission.fingerprint} tree {admission.tree_id}"


def _run_inputs(repository: Path, config: str, options: list[str]) -> int:
    dry_run = _inputs_options(options)
    layout = host_service.HostLayout.resolve(repository)
    admission = host_service.admit_source(layout, freeze=not dry_run)
    print(_source_line(admission), file=sys.stderr)
    host = host_service.GenerationHost(layout, admission)
    path, _ = host.config_file(config)
    output = host.call(
        "generate-input-cases",
        str(path),
        *options,
        "--git-commit",
        admission.commit,
        "--storage-root",
        str(layout.storage),
        scheduled=True,
        partition="standard",
    )
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


def _run_record_command(port: host_service.GenerationHost, layout: host_service.HostLayout, args: argparse.Namespace) -> int:
    """Run one campaign record command through its service operation; it freezes no source and starts no job."""
    if common.paths.campaign_run_parts(args.run_id) is None:
        _fail(f"{args.command} requires a canonical campaign handle", exit_code=2)
    operands: tuple[str, ...]
    if args.command == "fail-case":
        operation = "declare-candidate-failure"
        operands = (args.material, args.case_id, "--reason-code", args.reason_code, "--detail", args.detail)
    else:
        operation = "amend-replacement-pool"
        operands = (
            "--material",
            args.material,
            "--minimum-success-rate",
            str(args.minimum_success_rate),
            "--reason",
            args.reason,
        )
    print(port.call(operation, args.run_id, *operands, "--storage-root", str(layout.storage), scheduled=False), end="")
    return 0


def _scientific_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    """Render the scientific and execution overrides of one run command as the service arguments that resolve its plan."""
    scientific_arguments: list[str] = []
    for key in (
        "materials",
        "duration_hours",
        "revision",
        "node_jobs",
        "max_admission_cases",
        "cores_per_case",
        "case_timeout_seconds",
        "no_retry",
    ):
        value = getattr(args, key)
        if value is not None and value is not False:
            scientific_arguments.append("--" + key.replace("_", "-"))
            if value is not True:
                if isinstance(value, list):
                    scientific_arguments.extend(str(item) for item in value)
                else:
                    scientific_arguments.append(str(value))
    return tuple(scientific_arguments)


def _run(
    port: host_service.GenerationHost,
    args: argparse.Namespace,
    arguments: Sequence[str],
    driver: host_service.LauncherJob | None,
) -> int:
    """Resolve one run plan, then print it with its readiness, submit its driver job from the login node, or run it inside that job."""
    controller = controller_service.GenerationController(port)
    plan = controller.plan(args.config, allow_incomplete=args.dry_run, scientific_arguments=_scientific_arguments(args))
    if args.dry_run:
        footprint = controller_service.scheduler_footprint(
            controller_service.plan_execution_values(plan.payload),
            driver_time=args.driver_time,
            driver_partition=args.driver_partition,
        )
        # First stdout line: the unchanged plan plus the Slurm footprint it would ask for (node jobs and driver).
        print(json.dumps({**plan.payload, "identity": plan.identity, "scheduler_footprint": footprint}, sort_keys=True))
        print(controller.preflight(plan), file=sys.stderr)
    elif driver is None:
        controller.run(plan, args.config, max_admission_cases=args.max_admission_cases)
    else:
        # A campaign runs from the frozen source snapshot (commit plus recorded changes) it is bound to.
        parts = common.paths.campaign_run_parts(plan.identity)
        if parts is None:
            _fail("Campaign plan is not a current source campaign", exit_code=2)
        task = parts[0]
        submitted = controller.submit_run_tree(
            plan,
            module="src.generation.cli.cli_generation_controller",
            arguments=arguments,
            continue_with_current_code=args.continue_with_current_code,
            job=dataclasses.replace(driver, name=f"generation-{task}"),
        )
        result = {"status": "SUBMITTED", "job_id": submitted.job_id, "tree_id": submitted.tree_id, "log_path": str(submitted.log_path)}
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one public command and return its process exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    repository = Path.cwd()
    try:
        if args.command == "inputs":
            return _run_inputs(repository, args.config, args.options)
        driver: host_service.LauncherJob | None = None
        if args.command == "run":
            if args.max_admission_cases is not None and args.max_admission_cases < 1:
                _fail("--max-admission-cases must be an integer >= 1", exit_code=2)
            if args.continue_with_current_code and args.dry_run:
                _fail("--continue-with-current-code requires an executable run", exit_code=2)
            if args.revision is not None:
                campaign_identity.revision_label(args.revision)  # a malformed label is refused before any source is frozen
            if not args.dry_run and common.paths.source_tree_id() is None:
                # Started in the live checkout on the login node: submit the controller as its own job from the frozen tree.
                # The final job name needs the plan's task; this validates the driver options before the source is frozen.
                driver = host_service.LauncherJob("generation", "generation", args.driver_partition, args.driver_time)
        layout = host_service.HostLayout.resolve(repository)
        # Only an executing run freezes the source; inspection, cancellation and dry runs stay read-only.
        admission = host_service.admit_source(layout, freeze=args.command == "run" and not args.dry_run)
        print(_source_line(admission), file=sys.stderr)
        port = host_service.GenerationHost(layout, admission)
        if args.command == "run":
            return _run(port, args, arguments, driver)
        controller = controller_service.GenerationController(port)
        if args.command == "status":
            print(controller.status(args.target))
            return 0
        if args.command == "cancel":
            print(controller.cancel(args.run_id, force=args.force))
            return 0
        if args.command in {"fail-case", "amend-replacements"}:
            return _run_record_command(port, layout, args)
        _fail(f"Unsupported subcommand: {args.command}", exit_code=2)
    except KeyboardInterrupt:
        print("Generation interrupted", file=sys.stderr)
        return 130
    except controller_service.GenerationControllerError as error:
        print(error.report(), file=sys.stderr)
        return error.exit_code
    except host_service.GenerationCommandError as error:
        print(error, file=sys.stderr)
        return _INTERRUPTED_EXIT_CODE if error.returncode == _INTERRUPTED_EXIT_CODE else 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    except (OSError, RuntimeError, KeyError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
