"""Expose the per-parameter Generation field test through one thin command line.

Responsibilities:
  - Parse plan, submit, resume, status, and cancel commands and the run-member command of a node ticket
  - Dispatch the field-test services and print one JSON result per command

Design principles:
  - Scientific choices live in the planning service and the frozen roster, never in command options
  - Submission prints the sbatch commands it would run under --dry-run and writes nothing then
  - A case step turns SIGTERM into the runtime's cooperative cancellation, so COMSOL stops in order

This module does NOT:
  - Build rosters, reconcile the scheduler, run cases, or decide dispositions itself
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from src import common
from src.generation import field_tests
from src.generation.runtime import generation_runtime_batch as runtime_service


def _build_parser() -> argparse.ArgumentParser:
    """Build the field-test command parser."""
    parser = argparse.ArgumentParser(description="Plan and run per-parameter Generation field tests outside every Source")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("task", choices=field_tests.roster.TASKS)
        sub.add_argument("test_id")
        sub.add_argument("--storage-root", type=Path)
        return sub

    plan = command("plan", "write the frozen roster of stage A (--source) or stage B (--from-stage-a)")
    origin = plan.add_mutually_exclusive_group(required=True)
    origin.add_argument("--source", help="published Source folder below 01_generation/TASK/")
    origin.add_argument("--from-stage-a", help="stage-A test id whose analysis/verdicts.json names the rows to confirm")
    origin.add_argument("--replace-failed-bases", action="store_true", help="write the next roster version replacing every base with a failed gate")
    command("submit", "dispatch gates, or the variants of bases whose gate succeeded").add_argument("--dry-run", action="store_true")
    command("resume", "reconcile and redispatch absent attempts").add_argument("--dry-run", action="store_true")
    command("status", "count member dispositions")
    command("cancel", "publish the cancel marker of every running field-test node job")
    member = command("run-member", "run one member attempt inside its node-ticket step")
    member.add_argument("member_id")
    member.add_argument("attempt", type=int)
    member.add_argument("--cores", type=int, required=True)
    member.add_argument("--work-root", type=Path)
    return parser


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build and write one roster and its planning evidence once."""
    if args.source is not None:
        roster, evidence = field_tests.planning.build_stage_a(args.task, args.test_id, args.source, storage_root=args.storage_root)
    elif args.from_stage_a is not None:
        roster, evidence = field_tests.planning.build_stage_b(args.task, args.test_id, args.from_stage_a, storage_root=args.storage_root)
    else:
        roster, evidence = field_tests.planning.replace_failed_bases(args.task, args.test_id, storage_root=args.storage_root)
    directory = common.paths.resolve_field_test_dir(args.task, args.test_id, storage_root=args.storage_root)
    path = field_tests.roster.write_roster(directory, roster)
    evidence_name = path.name.replace("roster", "planning", 1)
    field_tests.roster.publish_once(directory / evidence_name, {"roster_sha256": roster["roster_sha256"], **evidence})
    roles: dict[str, int] = {}
    for item in roster["members"]:
        roles[item["role"]] = roles.get(item["role"], 0) + 1
    return {"roster": str(path), "roster_sha256": roster["roster_sha256"], "members": roles, "excluded_bases": len(evidence["excluded"])}


def _dispatch(args: argparse.Namespace) -> Any:
    """Run one parsed command."""
    if args.command == "plan":
        return _plan(args)
    if args.command in {"submit", "resume"}:
        return field_tests.dispatch.dispatch(args.task, args.test_id, mode=args.command, dry_run=args.dry_run, storage_root=args.storage_root)
    if args.command == "status":
        return field_tests.dispatch.status(args.task, args.test_id, storage_root=args.storage_root)
    if args.command == "cancel":
        return {"cancelled_node_jobs": field_tests.dispatch.cancel(args.task, args.test_id, storage_root=args.storage_root)}
    seal = field_tests.runner.run_member(
        args.task,
        args.test_id,
        args.member_id,
        attempt=args.attempt,
        cores=args.cores,
        storage_root=args.storage_root,
        work_root=args.work_root,
    )
    return {"member_id": args.member_id, "attempt": args.attempt, "sealed": None if seal is None else seal["disposition"]}


def main(argv: list[str] | None = None) -> int:
    """Run one field-test command and translate failures to process status two."""
    args = _build_parser().parse_args(argv)
    if args.command == "run-member":
        signal.signal(signal.SIGTERM, lambda _signum, _frame: runtime_service.request_runtime_cancellation())
    try:
        print(json.dumps(_dispatch(args), sort_keys=True))
    except Exception as error:  # noqa: BLE001 -- CLI boundary reports actionable domain errors
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
