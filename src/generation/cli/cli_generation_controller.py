"""
cli_generation_controller.py

Expose one public Generation workflow command backed by the Python controller.

Responsibilities:
  - Parse the public Generation command and admit the shared source
  - Dispatch lifecycle commands to the controller and native host boundary
  - Render run and background-session results with their process exit status

Design principles:
  - One public command covers plan, run, status, cancellation, and smoke
  - Scientific and persisted state remain owned by Generation services
  - Host execution uses the canonical native interpreter and sibling roots

This module does NOT:
  - Implement scientific cases, campaign scheduling, or dataset persistence
  - Load compute-node modules or bypass Slurm for material work
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from src.generation import controller as controller_service
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
    run = subcommands.add_parser("run", help="start or resume a campaign, benchmark, or paired workflow")
    run.add_argument("config")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--preflight-only", action="store_true")
    run.add_argument("--replacement-pool-size", type=int)
    run.add_argument("--parent-run-id")
    run.add_argument("--background", action="store_true")
    run.add_argument("--git-commit")
    status = subcommands.add_parser("status", help="inspect a config or persisted run ID")
    status.add_argument("target")
    status.add_argument("--git-commit")
    cancel = subcommands.add_parser("cancel", help="cancel a campaign or benchmark")
    cancel.add_argument("run_id")
    cancel.add_argument("--force", action="store_true")
    cancel.add_argument("--git-commit")
    smoke = subcommands.add_parser("smoke", help="submit a small native COMSOL smoke job")
    smoke.add_argument("--partition", choices=("gpu", "standard"), action=_StoreOnce)
    smoke.add_argument("--git-commit")
    background_status = subcommands.add_parser("background-status", help="inspect a durable background session")
    background_status.add_argument("session_id")
    subcommands.add_parser("background-list", help="list durable background sessions")
    inputs = subcommands.add_parser("inputs", help="generate selected canonical campaign inputs")
    inputs.add_argument("config")
    inputs.add_argument("options", nargs=argparse.REMAINDER)
    return parser


class _StoreOnce(argparse.Action):
    """Reject a repeated valued option instead of silently taking the last."""

    def __call__(
        self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str | Sequence[str] | None, option_string: str | None = None
    ) -> None:
        """Store the first value and reject any later occurrence."""
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"Specify {option_string} at most once")
        setattr(namespace, self.dest, values)


def _background_status(port: host_service.GenerationHost, session_id: str) -> str:
    payload = port.inspect_background(session_id)
    fields = (
        ("workflow_session_id", "workflow_session_id"),
        ("source_commit", "source_commit"),
        ("subcommand", "subcommand"),
        ("tmux_session", "tmux_session_name"),
        ("tmux_active", "tmux_active"),
        ("workflow_state", "workflow_state"),
        ("exit_code", "exit_code"),
        ("started_at", "started_at"),
        ("ended_at", "ended_at"),
        ("campaign_run_ids", "campaign_run_ids"),
        ("benchmark_run_ids", "benchmark_run_ids"),
        ("current_or_final_stage", "final_stage"),
        ("log", "log_path"),
    )

    def render(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, list):
            return ",".join(map(str, value)) or "-"
        return str(value).replace("\n", " ").replace("\t", " ")

    lines = [f"{label}={render(payload[key])}" for label, key in fields]
    if payload["tmux_active"]:
        lines.append(f"Attach:\n  tmux attach-session -t {shlex.quote(payload['tmux_session_name'])}")
    else:
        lines.append(f"Follow log:\n  tail -n 100 -F {shlex.quote(payload['log_path'])}")
    return "\n".join(lines)


def _background_list(port: host_service.GenerationHost) -> str:
    sessions = port.list_background()
    if not sessions:
        return "No background workflow sessions."
    lines = ["workflow_session_id\tstate\tsubcommand\tstarted_at\ttmux_session"]
    lines.extend(
        "\t".join(str(item[key]) for key in ("workflow_session_id", "workflow_state", "subcommand", "started_at", "tmux_session_name"))
        for item in sessions
    )
    return "\n".join(lines)


def _launch_background(port: host_service.GenerationHost, arguments: Sequence[str]) -> int:
    """Render the host-owned background launch result."""
    session = port.launch_background(arguments)
    status = session["status"]
    session_id = session["workflow_session_id"]
    tmux_name = session["tmux_session_name"]
    log = session["log_path"]
    if status in {"completed", "failed"}:
        print(f"BACKGROUND {status.upper()}\nworkflow_session_id={session_id}\nlog={log}", file=sys.stderr if status == "failed" else sys.stdout)
    else:
        print(
            f"BACKGROUND {status.upper()}\nworkflow_session_id={session_id}\ntmux_session={tmux_name}\n"
            f"log={log}\nAttach: tmux attach-session -t {shlex.quote(tmux_name)}"
        )
        if status == "started":
            print(f"Status: ./scripts/generation background-status {session_id}")
    return int(session["exit_code"])


def _inputs_options(options: list[str]) -> tuple[list[str], str | None, bool]:
    requested_commit: str | None = None
    dry_run = False
    remaining: list[str] = []
    index = 0
    while index < len(options):
        item = options[index]
        if item == "--git-commit":
            if requested_commit is not None or index + 1 >= len(options):
                _fail("inputs accepts exactly one valued --git-commit", exit_code=2)
            requested_commit = options[index + 1]
            index += 2
            continue
        if item == "--storage-root" or item.startswith(("--storage-root=", "--git-commit=")):
            _fail("inputs owns the sibling storage root", exit_code=2)
        dry_run = dry_run or item == "--dry-run"
        remaining.append(item)
        index += 1
    return remaining, requested_commit, dry_run


def _run_inputs(repository: Path, config: str, options: list[str]) -> int:
    remaining, requested_commit, dry_run = _inputs_options(options)
    layout = host_service.HostLayout.resolve(repository)
    admission = host_service.admit_source(layout, requested_commit, require_clean=not dry_run)
    host = host_service.GenerationHost(layout, admission)
    path, _ = host.config_file(config)
    output = host.call(
        "generate-input-cases",
        str(path),
        *remaining,
        "--git-commit",
        admission.commit,
        "--storage-root",
        str(layout.storage),
        scheduled=True,
        partition="standard",
    )
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0911, PLR0912
    """Parse one public command and return its process exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    repository = Path.cwd()
    try:
        if args.command == "inputs":
            return _run_inputs(repository, args.config, args.options)
        if args.command == "run":
            if args.dry_run and args.preflight_only:
                _fail("--dry-run cannot be combined with --preflight-only", exit_code=2)
            if args.replacement_pool_size is not None and args.replacement_pool_size < 1:
                _fail("replacement_pool_size must be >= 1", exit_code=2)
            if args.parent_run_id and args.replacement_pool_size is None:
                _fail("--parent-run-id requires --replacement-pool-size", exit_code=2)
            if args.background and (args.dry_run or args.preflight_only):
                _fail("--background requires an executable run", exit_code=2)
        layout = host_service.HostLayout.resolve(repository)
        require_clean = args.command == "run" and not (args.dry_run or args.preflight_only)
        admission = host_service.admit_source(layout, getattr(args, "git_commit", None), require_clean=require_clean)
        print(f"Source: shared HEAD {admission.commit} fingerprint {admission.fingerprint}", file=sys.stderr)
        port = host_service.GenerationHost(layout, admission)
        if args.command == "run" and args.background:
            return _launch_background(port, arguments)
        controller = controller_service.GenerationController(port)
        if args.command == "run":
            plan = controller.plan(args.config, allow_incomplete=args.dry_run)
            if args.replacement_pool_size is not None and plan.kind != "campaign":
                _fail("Replacement completion requires a campaign plan", exit_code=2)
            if args.dry_run:
                payload = (
                    controller.completion_preview(plan, args.replacement_pool_size, args.parent_run_id)
                    if args.replacement_pool_size is not None
                    else plan.payload
                )
                print(json.dumps(payload, sort_keys=True))
            elif args.preflight_only:
                if args.replacement_pool_size is not None:
                    controller.completion_preview(plan, args.replacement_pool_size, args.parent_run_id)
                print(controller.preflight(plan))
            else:
                controller.run(plan, args.config, pool_size=args.replacement_pool_size, parent_run_id=args.parent_run_id)
            return 0
        if args.command == "status":
            print(controller.status(args.target))
            return 0
        if args.command == "cancel":
            print(controller.cancel(args.run_id, force=args.force))
            return 0
        if args.command == "smoke":
            partition = args.partition or "gpu"
            job_id = port.submit_smoke(partition=partition)
            print(f"GENERATION SMOKE SUBMITTED job={job_id} partition={partition} logs={layout.logs}")
            return 0
        if args.command == "background-status":
            print(_background_status(port, args.session_id))
            return 0
        if args.command == "background-list":
            print(_background_list(port))
            return 0
        _fail(f"Unsupported subcommand: {args.command}", exit_code=2)
    except KeyboardInterrupt:
        print("Generation interrupted", file=sys.stderr)
        return 130
    except controller_service.GenerationControllerError as error:
        print(error, file=sys.stderr)
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
