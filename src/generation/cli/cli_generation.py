"""Thin command-line interface for profile-qualified generation services."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from src.generation import generation_case as case_service
from src.generation import generation_cluster as cluster_service
from src.generation import generation_config as config_service
from src.generation import generation_runtime as runtime_service


def _add_storage_arguments(parser: argparse.ArgumentParser, *, include_work: bool = False) -> None:
    """Add shared storage and optional work-root boundaries."""
    parser.add_argument("--storage-root", type=Path)
    if include_work:
        parser.add_argument("--work-root", type=Path)


def _add_case_range(parser: argparse.ArgumentParser) -> None:
    """Add inclusive configured case-range selection arguments."""
    parser.add_argument("--case-start", type=int)
    parser.add_argument("--case-stop", type=int)


def _add_resources(parser: argparse.ArgumentParser) -> None:
    """Add the four mandatory production resource controls and one capacity override."""
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--cases-per-node", type=int, required=True)
    parser.add_argument("--cores-per-case", type=int, required=True)
    parser.add_argument("--max-parallel-cases", type=int, required=True)
    parser.add_argument("--cores-per-node", type=int)


def _build_parser() -> argparse.ArgumentParser:
    """Build the complete generation command parser."""
    parser = argparse.ArgumentParser(description="Generate and run isolated profile-qualified COMSOL cases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate one generation configuration")
    validate.add_argument("config", type=Path)

    generate = subparsers.add_parser("generate-case", help="generate one case input bundle")
    generate.add_argument("config", type=Path)
    generate.add_argument("case_index", type=int)
    generate.add_argument("destination", type=Path)

    prepare = subparsers.add_parser("prepare-case", help="prepare one isolated case work directory")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("case_index", type=int)
    _add_storage_arguments(prepare, include_work=True)

    command = subparsers.add_parser("print-command", help="print one COMSOL argument vector without execution")
    command.add_argument("config", type=Path)
    command.add_argument("case_index", type=int)
    command.add_argument("--cores-per-case", type=int, required=True)
    command.add_argument("--scheduler-kind", choices=("local", "slurm"), default="local")

    run_case = subparsers.add_parser("run-case", help="run, validate, and publish one case")
    run_case.add_argument("config", type=Path)
    run_case.add_argument("case_index", type=int)
    run_case.add_argument("--cores-per-case", type=int, required=True)
    run_case.add_argument("--scheduler-kind", choices=("local", "slurm"), default="local")
    run_case.add_argument("--cleanup-failed", action="store_true")
    _add_storage_arguments(run_case, include_work=True)

    validate_case = subparsers.add_parser("validate-case", help="validate one completed case")
    validate_case.add_argument("config", type=Path)
    validate_case.add_argument("case_index", type=int)
    _add_storage_arguments(validate_case)

    node = subparsers.add_parser("run-node", help="run one bounded node-level worker")
    node.add_argument("config", type=Path)
    _add_resources(node)
    _add_case_range(node)
    node.add_argument("--worker-index", type=int, required=True)
    node.add_argument("--worker-count", type=int, required=True)
    node.add_argument("--remaining-cases", type=int)
    node.add_argument("--assignment-mode", choices=("shared", "deterministic"), default="shared")
    node.add_argument("--scheduler-kind", choices=("local", "slurm"), required=True)
    _add_storage_arguments(node, include_work=True)

    batch = subparsers.add_parser("run-batch", help="run a local multi-worker development batch")
    batch.add_argument("config", type=Path)
    _add_resources(batch)
    _add_case_range(batch)
    batch.add_argument("--assignment-mode", choices=("shared", "deterministic"), default="shared")
    _add_storage_arguments(batch, include_work=True)

    finalize = subparsers.add_parser("finalize-batch", help="validate and terminally publish one batch")
    finalize.add_argument("config", type=Path)
    _add_storage_arguments(finalize)

    submit = subparsers.add_parser("print-submit", help="print one scheduler submission command without execution")
    submit.add_argument("config", type=Path)
    _add_resources(submit)
    _add_case_range(submit)
    _add_storage_arguments(submit)

    transfer = subparsers.add_parser("validate-transfer", help="validate one transferred terminal batch")
    transfer.add_argument("config", type=Path)
    _add_storage_arguments(transfer)
    return parser


def _load(args: argparse.Namespace) -> config_service.GenerationConfig:
    """Load the configuration referenced by one parsed command."""
    return config_service.load_generation_config(args.config)


def _selected(config: config_service.GenerationConfig, args: argparse.Namespace) -> tuple[int, ...]:
    """Resolve one parsed inclusive case range."""
    return cluster_service.select_case_indices(config, case_start=args.case_start, case_stop=args.case_stop)


def _remaining(config: config_service.GenerationConfig, selected: tuple[int, ...], *, storage_root: Path | None) -> int:
    """Count selection members that are not valid completed cases."""
    return sum(not runtime_service.completed_case_is_valid(config, case_index, storage_root=storage_root) for case_index in selected)


def _plan(
    config: config_service.GenerationConfig,
    args: argparse.Namespace,
    selected: tuple[int, ...],
) -> cluster_service.ResourcePlan:
    """Build one validated plan from mandatory parsed resource controls."""
    cores_per_node = config.values["cluster"]["cores_per_node"] if args.cores_per_node is None else args.cores_per_node
    return cluster_service.build_resource_plan(
        max_nodes=args.max_nodes,
        cases_per_node=args.cases_per_node,
        cores_per_case=args.cores_per_case,
        max_parallel_cases=args.max_parallel_cases,
        cores_per_node=cores_per_node,
        remaining_cases=(
            args.remaining_cases
            if getattr(args, "remaining_cases", None) is not None
            else len(selected)
            if args.command == "run-node"
            else _remaining(config, selected, storage_root=args.storage_root)
        ),
    )


def _summary(config: config_service.GenerationConfig) -> dict[str, Any]:
    """Return one concise configuration preflight summary."""
    return {
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_count": len(config.case_indices),
        "template_path": str(config.template_path),
        "template_sha256": config.template_sha256,
    }


def _dispatch(args: argparse.Namespace) -> int:  # noqa: PLR0911 -- thin CLI command dispatch
    """Dispatch one parsed command to its authoritative service."""
    config = _load(args)
    if args.command == "validate-config":
        print(json.dumps(_summary(config), sort_keys=True))
        return 0
    if args.command == "generate-case":
        bundle = case_service.generate_case_input_bundle(config, args.case_index, args.destination)
        print(bundle.directory)
        return 0
    if args.command == "prepare-case":
        prepared = case_service.prepare_case_work_directory(
            config,
            args.case_index,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(prepared.work_directory)
        return 0
    if args.command == "print-command":
        config.case_id(args.case_index)
        print(shlex.join(runtime_service.build_comsol_command(config, cores_per_case=args.cores_per_case, scheduler_kind=args.scheduler_kind)))
        return 0
    if args.command == "run-case":
        outcome = runtime_service.run_case(
            config,
            args.case_index,
            cores_per_case=args.cores_per_case,
            scheduler_kind=args.scheduler_kind,
            storage_root=args.storage_root,
            work_root=args.work_root,
            cleanup_failed=args.cleanup_failed,
        )
        print(json.dumps({"status": outcome.status, "case_id": outcome.case_id, "directory": str(outcome.processed_directory)}, sort_keys=True))
        return 0
    if args.command == "validate-case":
        provenance = runtime_service.validate_completed_case(config, args.case_index, storage_root=args.storage_root)
        print(json.dumps({"status": "valid", "case_id": provenance["case_id"], "case_identity": provenance["case_identity"]}, sort_keys=True))
        return 0
    if args.command in {"run-node", "run-batch", "print-submit"}:
        selected = _selected(config, args)
        plan = _plan(config, args, selected)
        if args.command == "run-node":
            result = cluster_service.run_node_worker(
                config,
                selected,
                plan=plan,
                worker_index=args.worker_index,
                worker_count=args.worker_count,
                assignment_mode=args.assignment_mode,
                scheduler_kind=args.scheduler_kind,
                storage_root=args.storage_root,
                work_root=args.work_root,
            )
            print(json.dumps({"worker_index": result.worker_index, "completed_or_skipped": len(result.outcomes)}, sort_keys=True))
            return 0
        if args.command == "run-batch":
            results = cluster_service.run_local_batch(
                config,
                selected,
                plan=plan,
                assignment_mode=args.assignment_mode,
                storage_root=args.storage_root,
                work_root=args.work_root,
            )
            print(json.dumps({"node_workers": len(results), "effective_parallel_cases": plan.effective_parallel_cases}, sort_keys=True))
            return 0
        print(shlex.join(cluster_service.build_slurm_submission_command(config, selected, plan=plan)))
        return 0
    if args.command == "finalize-batch":
        print(runtime_service.finalize_batch(config, storage_root=args.storage_root))
        return 0
    if args.command == "validate-transfer":
        manifest = runtime_service.validate_terminal_batch(config, storage_root=args.storage_root)
        print(json.dumps({"status": "valid", "batch_id": manifest["batch_id"], "case_count": len(manifest["cases"])}, sort_keys=True))
        return 0
    message = f"Unsupported generation command: {args.command!r}."
    raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    """Run one generation command and translate failures to process status two."""
    args = _build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except Exception as error:  # noqa: BLE001 -- CLI boundary reports actionable domain errors
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
