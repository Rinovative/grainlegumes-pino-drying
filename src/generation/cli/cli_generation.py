"""Thin command-line interface for profile-qualified generation services."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

from src.generation import generation_campaign_runtime as campaign_runtime
from src.generation import generation_case as case_service
from src.generation import generation_cluster as cluster_service
from src.generation import generation_config as config_service
from src.generation import generation_preflight as preflight_service
from src.generation import generation_runtime as runtime_service
from src.generation import generation_workflow as workflow_service
from src.generation import generation_workspace as workspace_service


def _add_storage_arguments(parser: argparse.ArgumentParser, *, include_work: bool = False) -> None:
    """Add shared storage and optional work-root boundaries."""
    parser.add_argument("--storage-root", type=Path)
    if include_work:
        parser.add_argument("--work-root", type=Path)


def _add_case_range(parser: argparse.ArgumentParser) -> None:
    """Add inclusive configured case-range selection arguments."""
    parser.add_argument("--case-start", type=int)
    parser.add_argument("--case-stop", type=int)


def _add_batch_selection(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add selection of one predeclared campaign batch."""
    parser.add_argument("--only-batch", required=required)


def _add_resources(parser: argparse.ArgumentParser) -> None:
    """Add the four mandatory production resource controls and one capacity override."""
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--cases-per-node", type=int, required=True)
    parser.add_argument("--cores-per-case", type=int, required=True)
    parser.add_argument("--max-parallel-cases", type=int, required=True)
    parser.add_argument("--cores-per-node", type=int)


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 -- one centralized thin CLI parser
    """Build the complete generation command parser."""
    parser = argparse.ArgumentParser(description="Generate and run isolated profile-qualified COMSOL cases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate one generation configuration")
    validate.add_argument("config", type=Path)
    _add_batch_selection(validate, required=False)

    preflight = subparsers.add_parser(
        "preflight",
        help="audit the native CPU environment without a production solve",
    )
    preflight.add_argument("config", type=Path)
    _add_batch_selection(preflight, required=False)
    preflight.add_argument("--storage-root", type=Path, required=True)
    preflight.add_argument("--work-root", type=Path, required=True)
    preflight.add_argument("--venv-path", type=Path, required=True)
    preflight.add_argument("--environment-only", action="store_true")
    _add_resources(preflight)

    worker_init = subparsers.add_parser(
        "initialize-worker-workspace",
        help="mark one mktemp-created Slurm worker root",
    )
    worker_init.add_argument("directory", type=Path)
    worker_init.add_argument("--campaign-run-id", required=True)
    worker_init.add_argument("--storage-root", type=Path, required=True)

    worker_cleanup = subparsers.add_parser(
        "cleanup-worker-workspace",
        help="guard and remove one current Slurm worker root",
    )
    worker_cleanup.add_argument("directory", type=Path)
    worker_cleanup.add_argument("--campaign-run-id", required=True)
    worker_cleanup.add_argument("--storage-root", type=Path, required=True)

    transfer_stage = subparsers.add_parser(
        "create-transfer-staging",
        help="create one marked local transfer staging root",
    )
    transfer_stage.add_argument("campaign_run_id")
    transfer_stage.add_argument("--storage-root", type=Path, required=True)

    cleanup_staging = subparsers.add_parser(
        "cleanup-transfer-staging",
        help="list marked transfer staging; delete only with --confirm",
    )
    cleanup_staging.add_argument("--campaign-run-id")
    cleanup_staging.add_argument("--directory", type=Path)
    cleanup_staging.add_argument("--storage-root", type=Path, required=True)
    cleanup_staging.add_argument("--confirm", action="store_true")

    generate = subparsers.add_parser("generate-case", help="generate one case input bundle")
    generate.add_argument("config", type=Path)
    _add_batch_selection(generate, required=True)
    generate.add_argument("case_index", type=int)
    generate.add_argument("destination", type=Path)

    prepare = subparsers.add_parser("prepare-case", help="prepare one isolated case work directory")
    prepare.add_argument("config", type=Path)
    _add_batch_selection(prepare, required=True)
    prepare.add_argument("case_index", type=int)
    _add_storage_arguments(prepare, include_work=True)

    command = subparsers.add_parser("print-command", help="print one COMSOL argument vector without execution")
    command.add_argument("config", type=Path)
    _add_batch_selection(command, required=True)
    command.add_argument("case_index", type=int)
    command.add_argument("--cores-per-case", type=int, required=True)
    command.add_argument("--scheduler-kind", choices=("local", "slurm"), default="local")

    run_case = subparsers.add_parser("run-case", help="run, validate, and publish one case")
    run_case.add_argument("config", type=Path)
    _add_batch_selection(run_case, required=True)
    run_case.add_argument("case_index", type=int)
    run_case.add_argument("--cores-per-case", type=int, required=True)
    run_case.add_argument("--scheduler-kind", choices=("local", "slurm"), default="local")
    _add_storage_arguments(run_case, include_work=True)

    validate_case = subparsers.add_parser("validate-case", help="validate one completed case")
    validate_case.add_argument("config", type=Path)
    _add_batch_selection(validate_case, required=True)
    validate_case.add_argument("case_index", type=int)
    _add_storage_arguments(validate_case)

    batch = subparsers.add_parser("run-batch", help="run a local multi-worker development batch")
    batch.add_argument("config", type=Path)
    _add_batch_selection(batch, required=True)
    _add_resources(batch)
    _add_case_range(batch)
    batch.add_argument("--assignment-mode", choices=("shared", "deterministic"), default="shared")
    _add_storage_arguments(batch, include_work=True)

    finalize = subparsers.add_parser("finalize-batch", help="validate and terminally publish one batch")
    finalize.add_argument("config", type=Path)
    _add_batch_selection(finalize, required=True)
    _add_storage_arguments(finalize)

    transfer = subparsers.add_parser("validate-transfer", help="validate one transferred terminal batch")
    transfer.add_argument("config", type=Path)
    _add_batch_selection(transfer, required=True)
    _add_storage_arguments(transfer)

    campaign_worker = subparsers.add_parser("run-campaign-worker", help="run one node from the shared campaign worker pool")
    campaign_worker.add_argument("config", type=Path)
    _add_batch_selection(campaign_worker, required=False)
    _add_resources(campaign_worker)
    campaign_worker.add_argument("--worker-index", type=int, required=True)
    campaign_worker.add_argument("--worker-count", type=int, required=True)
    campaign_worker.add_argument("--remaining-cases", type=int, required=True)
    campaign_worker.add_argument("--scheduler-kind", choices=("local", "slurm"), required=True)
    _add_storage_arguments(campaign_worker, include_work=True)

    campaign_plan = subparsers.add_parser(
        "plan-campaign",
        help="resolve paths and Slurm arguments without mutation",
    )
    campaign_plan.add_argument("config", type=Path)
    _add_batch_selection(campaign_plan, required=False)
    campaign_plan.add_argument("--wall-time")
    campaign_plan.add_argument("--git-commit", required=True)
    _add_resources(campaign_plan)
    _add_storage_arguments(campaign_plan)

    submit_campaign = subparsers.add_parser("submit-campaign", help="submit and persist one exact-commit campaign run")
    submit_campaign.add_argument("config", type=Path)
    _add_batch_selection(submit_campaign, required=False)
    submit_campaign.add_argument("--wall-time")
    submit_campaign.add_argument("--git-commit", required=True)
    _add_resources(submit_campaign)
    _add_storage_arguments(submit_campaign)

    campaign_status = subparsers.add_parser("campaign-status", help="reconstruct persistent campaign and scheduler status")
    campaign_status.add_argument("campaign_run_id")
    campaign_status.add_argument("--no-scheduler", action="store_true")
    campaign_status.add_argument("--format", choices=("json", "state"), default="json")
    _add_storage_arguments(campaign_status)

    accounting = subparsers.add_parser(
        "campaign-accounting",
        help="print exact squeue and sacct evidence",
    )
    accounting.add_argument("campaign_run_id")
    _add_storage_arguments(accounting)

    cancel = subparsers.add_parser(
        "cancel-campaign",
        help="cancel every persisted Slurm attempt",
    )
    cancel.add_argument("campaign_run_id")
    _add_storage_arguments(cancel)

    resume = subparsers.add_parser(
        "resume-campaign",
        help="submit a fresh pool for incomplete validated membership",
    )
    resume.add_argument("campaign_run_id")
    _add_storage_arguments(resume)

    interruption = subparsers.add_parser(
        "record-worker-interruption",
        help="persist best-effort Slurm worker interruption evidence",
    )
    interruption.add_argument("campaign_run_id")
    interruption.add_argument("--signal", required=True)
    interruption.add_argument("--exit-code", type=int, required=True)
    _add_storage_arguments(interruption)

    publish_transfer = subparsers.add_parser(
        "publish-transferred-campaign",
        help="validate staged bytes and atomically mark local transfer complete",
    )
    publish_transfer.add_argument("campaign_run_id")
    publish_transfer.add_argument("--staging-root", type=Path, required=True)
    publish_transfer.add_argument("--destination-root", type=Path, required=True)
    publish_transfer.add_argument("--source-host", required=True)
    publish_transfer.add_argument("--source-storage-root", required=True)

    campaign_terminal = subparsers.add_parser("validate-campaign-terminal", help="validate and publish terminal campaign evidence")
    campaign_terminal.add_argument("campaign_run_id")
    _add_storage_arguments(campaign_terminal)

    transfer_plan = subparsers.add_parser("campaign-transfer-plan", help="print terminally validated collection directories")
    transfer_plan.add_argument("campaign_run_id")
    transfer_plan.add_argument("--format", choices=("json", "tsv"), default="json")
    _add_storage_arguments(transfer_plan)

    validate_publication = subparsers.add_parser(
        "validate-published-campaign",
        help="validate an exact GPU generation publication and transfer receipt",
    )
    validate_publication.add_argument("campaign_run_id")
    _add_storage_arguments(validate_publication)

    build_datasets = subparsers.add_parser(
        "build-campaign-datasets",
        help="build or reuse, inspect, and loader-smoke every declared package",
    )
    build_datasets.add_argument("campaign_run_id")
    _add_storage_arguments(build_datasets)

    prepare_all = subparsers.add_parser(
        "prepare-all-workflow",
        help="persist every completed local gate before optional CPU cleanup",
    )
    prepare_all.add_argument("campaign_run_id")
    prepare_all.add_argument("--keep-cpu-source", action="store_true")
    _add_storage_arguments(prepare_all)

    validate_all = subparsers.add_parser(
        "validate-all-workflow",
        help="require one terminally successful all-workflow receipt",
    )
    validate_all.add_argument("campaign_run_id")
    _add_storage_arguments(validate_all)

    cleanup_authorization = subparsers.add_parser(
        "cpu-cleanup-authorization",
        help="issue compact CPU cleanup authorization after every local gate",
    )
    cleanup_authorization.add_argument("campaign_run_id")
    cleanup_authorization.add_argument("--format", choices=("json", "tsv"), default="json")
    _add_storage_arguments(cleanup_authorization)

    cleanup_source = subparsers.add_parser(
        "cleanup-campaign-source",
        help="dry-run or execute one GPU-authorized CPU campaign cleanup",
    )
    cleanup_source.add_argument("campaign_run_id")
    cleanup_source.add_argument("--source-host", required=True)
    cleanup_source.add_argument("--destination-storage-root", required=True)
    cleanup_source.add_argument("--transfer-receipt-sha256", required=True)
    cleanup_source.add_argument("--dataset-receipt-sha256", required=True)
    cleanup_source.add_argument("--workflow-gate-sha256", required=True)
    cleanup_source.add_argument("--source-inventory-sha256", required=True)
    cleanup_source.add_argument("--source-file-count", type=int, required=True)
    cleanup_source.add_argument("--source-bytes", type=int, required=True)
    cleanup_source.add_argument("--authorization-sha256", required=True)
    cleanup_source.add_argument("--confirm", action="store_true")
    cleanup_source.add_argument("--format", choices=("json", "tsv"), default="json")
    _add_storage_arguments(cleanup_source)

    record_cleanup = subparsers.add_parser(
        "record-cpu-cleanup",
        help="bind one authorized remote cleanup receipt to the local workflow",
    )
    record_cleanup.add_argument("campaign_run_id")
    record_cleanup.add_argument("--authorization-sha256", required=True)
    record_cleanup.add_argument("--cleanup-receipt-sha256", required=True)
    record_cleanup.add_argument("--reclaimed-bytes", type=int, required=True)
    _add_storage_arguments(record_cleanup)

    source_status = subparsers.add_parser(
        "campaign-source-status",
        help="report one host's campaign source and cleanup state",
    )
    source_status.add_argument("campaign_run_id")
    source_status.add_argument("--query-scheduler", action="store_true")
    source_status.add_argument("--format", choices=("json", "tsv"), default="json")
    _add_storage_arguments(source_status)

    storage_status = subparsers.add_parser(
        "storage-status",
        help="report generation, datasets, staging, packages, runs, and cleanup",
    )
    storage_status.add_argument("--role", choices=("gpu", "cpu"), required=True)
    storage_status.add_argument("--campaign-run-id")
    storage_status.add_argument("--query-scheduler", action="store_true")
    _add_storage_arguments(storage_status)

    workflow_failure = subparsers.add_parser(
        "record-workflow-failure",
        help="persist one compact resumable all-workflow failure record",
    )
    workflow_failure.add_argument("campaign_run_id")
    workflow_failure.add_argument("--stage", required=True)
    workflow_failure.add_argument("--resume-command", required=True)
    workflow_failure.add_argument("--cpu-bytes-retained", type=int, required=True)
    _add_storage_arguments(workflow_failure)
    return parser


def _load(args: argparse.Namespace) -> config_service.GenerationConfig:
    """Load one predeclared batch referenced by a parsed command."""
    return config_service.load_generation_config(args.config, only_batch=args.only_batch)


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
    cores_per_node = config.execution_values["cluster"]["cores_per_node"] if args.cores_per_node is None else args.cores_per_node
    return cluster_service.build_resource_plan(
        max_nodes=args.max_nodes,
        cases_per_node=args.cases_per_node,
        cores_per_case=args.cores_per_case,
        max_parallel_cases=args.max_parallel_cases,
        cores_per_node=cores_per_node,
        remaining_cases=(
            args.remaining_cases
            if getattr(args, "remaining_cases", None) is not None
            else _remaining(config, selected, storage_root=args.storage_root)
        ),
    )


def _summary(config: config_service.GenerationConfig) -> dict[str, Any]:
    """Return one concise configuration preflight summary."""
    return {
        "simulation_profile": config.profile.id,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "batch_name": config.batch_name,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_count": len(config.case_indices),
        "template_path": str(config.template_path),
        "template_sha256": config.template_sha256,
    }


def _dispatch(args: argparse.Namespace) -> int:  # noqa: C901, PLR0911, PLR0912, PLR0915 -- thin CLI command dispatch
    """Dispatch one parsed command to its authoritative service."""
    if args.command == "preflight":
        unresolved = config_service.load_campaign_config(
            args.config,
            require_executable=False,
        )
        cores_per_node = unresolved.execution_values["cluster"]["cores_per_node"] if args.cores_per_node is None else args.cores_per_node
        report = preflight_service.run_cpu_preflight(
            args.config,
            only_batch=args.only_batch,
            storage_root=args.storage_root,
            work_root=args.work_root,
            venv_path=args.venv_path,
            max_nodes=args.max_nodes,
            cases_per_node=args.cases_per_node,
            cores_per_case=args.cores_per_case,
            max_parallel_cases=args.max_parallel_cases,
            cores_per_node=cores_per_node,
        )
        print(json.dumps(report, sort_keys=True))
        if report["production_configuration_ready"] or args.environment_only:
            return 0
        return 2
    if args.command == "initialize-worker-workspace":
        marker = workspace_service.initialize_worker_workspace(
            args.directory,
            run_id=args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(marker)
        return 0
    if args.command == "cleanup-worker-workspace":
        reclaimed = workspace_service.cleanup_worker_workspace(
            args.directory,
            run_id=args.campaign_run_id,
            storage_root=args.storage_root,
            allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        print(json.dumps({"reclaimed_bytes": reclaimed}, sort_keys=True))
        return 0
    if args.command == "create-transfer-staging":
        directory = workspace_service.create_transfer_staging(
            storage_root=args.storage_root,
            run_id=args.campaign_run_id,
        )
        print(directory)
        return 0
    if args.command == "cleanup-transfer-staging":
        candidates = workspace_service.transfer_staging_candidates(
            storage_root=args.storage_root,
            run_id=args.campaign_run_id,
        )
        if args.directory is not None:
            expected = str(args.directory.expanduser().resolve())
            candidates = tuple(candidate for candidate in candidates if candidate["path"] == expected)
            if not candidates:
                message = f"Requested transfer staging is not a valid candidate: {expected}"
                raise ValueError(message)
        cleanup_results = [dict(candidate, removed=False) for candidate in candidates]
        if args.confirm:
            for result in cleanup_results:
                reclaimed = workspace_service.cleanup_transfer_staging(
                    result["path"],
                    storage_root=args.storage_root,
                    run_id=result["run_id"],
                )
                result["removed"] = True
                result["reclaimed_bytes"] = reclaimed
        print(
            json.dumps(
                {
                    "mode": "delete" if args.confirm else "dry-run",
                    "candidates": cleanup_results,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-config" and args.only_batch is None:
        campaign = config_service.load_campaign_config(args.config)
        print(
            json.dumps(
                {
                    "campaign_name": campaign.campaign_name,
                    "campaign_id": campaign.campaign_id,
                    "simulation_profile": campaign.profile.id,
                    "batches": [_summary(batch) for batch in campaign.batches],
                    "dataset_packages": list(campaign.dataset_packages),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "campaign-status":
        status = campaign_runtime.campaign_status(
            args.campaign_run_id,
            storage_root=args.storage_root,
            query_scheduler=not args.no_scheduler,
        )
        if args.format == "state":
            print(status["campaign_state"])
        else:
            print(json.dumps(status, sort_keys=True))
        return 0
    if args.command == "campaign-accounting":
        accounting = campaign_runtime.campaign_accounting(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(accounting, sort_keys=True))
        return 0
    if args.command == "cancel-campaign":
        receipt = campaign_runtime.cancel_campaign(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "resume-campaign":
        manifest = campaign_runtime.resume_campaign(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "record-worker-interruption":
        path = campaign_runtime.record_worker_interruption(
            args.campaign_run_id,
            storage_root=args.storage_root,
            signal_name=args.signal,
            exit_code=args.exit_code,
        )
        print(path)
        return 0
    if args.command == "publish-transferred-campaign":
        receipt = campaign_runtime.publish_transferred_campaign(
            args.campaign_run_id,
            staging_root=args.staging_root,
            destination_root=args.destination_root,
            source_host=args.source_host,
            source_storage_root=args.source_storage_root,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "validate-campaign-terminal":
        terminal = campaign_runtime.validate_terminal_campaign(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(terminal, sort_keys=True))
        return 0
    if args.command == "campaign-transfer-plan":
        transfer_plan = campaign_runtime.campaign_transfer_plan(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        if args.format == "json":
            print(json.dumps(transfer_plan, sort_keys=True))
        else:
            print(
                "\t".join(
                    (
                        "campaign",
                        transfer_plan["campaign_name"],
                        transfer_plan["git_commit"],
                        transfer_plan["campaign_directory"],
                        transfer_plan["campaign_config"],
                    )
                )
            )
            for batch in transfer_plan["batches"]:
                print(
                    "\t".join(
                        (
                            "batch",
                            batch["batch_name"],
                            batch["batch_id"],
                            str(batch["case_count"]),
                            batch["meta_directory"],
                            batch["raw_directory"],
                            batch["processed_directory"],
                        )
                    )
                )
        return 0
    if args.command == "validate-published-campaign":
        receipt = campaign_runtime.validate_transferred_campaign(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "build-campaign-datasets":
        receipt = workflow_service.build_campaign_datasets(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "prepare-all-workflow":
        receipt = workflow_service.prepare_all_workflow_receipt(
            args.campaign_run_id,
            storage_root=args.storage_root,
            cleanup_requested=not args.keep_cpu_source,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "validate-all-workflow":
        receipt = workflow_service.validate_completed_workflow(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "cpu-cleanup-authorization":
        authorization = workflow_service.cpu_cleanup_authorization(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        if args.format == "json":
            print(json.dumps(authorization, sort_keys=True))
        else:
            print(
                "	".join(
                    (
                        "authorization",
                        authorization["authorization_sha256"],
                        authorization["source_host"],
                        authorization["source_storage_root"],
                        authorization["destination_storage_root"],
                        authorization["transfer_receipt_sha256"],
                        authorization["dataset_receipt_sha256"],
                        authorization["workflow_gate_sha256"],
                        authorization["source_inventory_sha256"],
                        str(authorization["source_file_count"]),
                        str(authorization["source_bytes"]),
                    )
                )
            )
        return 0
    if args.command == "cleanup-campaign-source":
        result = workflow_service.cleanup_cpu_campaign_source(
            args.campaign_run_id,
            storage_root=args.storage_root,
            source_host=args.source_host,
            destination_storage_root=args.destination_storage_root,
            transfer_receipt_sha256=args.transfer_receipt_sha256,
            dataset_receipt_sha256=args.dataset_receipt_sha256,
            workflow_gate_sha256=args.workflow_gate_sha256,
            source_inventory_sha256=args.source_inventory_sha256,
            source_file_count=args.source_file_count,
            source_bytes=args.source_bytes,
            authorization_sha256=args.authorization_sha256,
            confirm=args.confirm,
        )
        if args.format == "json":
            print(json.dumps(result, sort_keys=True))
        else:
            print(
                "	".join(
                    (
                        "cleanup",
                        str(result["status"]),
                        str(result.get("mode", "complete")),
                        str(result["authorization_sha256"]),
                        str(result.get("reclaimable_bytes", result.get("source_bytes_reclaimed", 0))),
                        str(result.get("receipt_sha256", "-")),
                    )
                )
            )
        return 0
    if args.command == "record-cpu-cleanup":
        receipt = workflow_service.record_cpu_cleanup_complete(
            args.campaign_run_id,
            storage_root=args.storage_root,
            authorization_sha256=args.authorization_sha256,
            cleanup_receipt_sha256=args.cleanup_receipt_sha256,
            reclaimed_bytes=args.reclaimed_bytes,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "campaign-source-status":
        status = workflow_service.campaign_source_status(
            args.campaign_run_id,
            storage_root=args.storage_root,
            query_scheduler=args.query_scheduler,
        )
        if args.format == "json":
            print(json.dumps(status, sort_keys=True))
        else:
            print(
                "	".join(
                    (
                        "source-status",
                        str(status["campaign_run_id"]),
                        str(status["campaign_state"]),
                        str(status["reclaimable_bytes"]),
                        str(status["cleanup_eligibility"]),
                        str(status["active_slurm"]),
                    )
                )
            )
        return 0
    if args.command == "storage-status":
        status = workflow_service.storage_status(
            storage_root=args.storage_root,
            role=args.role,
            run_id=args.campaign_run_id,
            query_scheduler=args.query_scheduler,
        )
        print(json.dumps(status, sort_keys=True))
        return 0
    if args.command == "record-workflow-failure":
        path = workflow_service.record_workflow_failure(
            args.campaign_run_id,
            storage_root=args.storage_root,
            stage=args.stage,
            resume_command=args.resume_command,
            cpu_bytes_retained=args.cpu_bytes_retained,
        )
        print(path)
        return 0
    if args.command in {"run-campaign-worker", "plan-campaign", "submit-campaign"}:
        campaign = config_service.load_campaign_config(args.config).select_batches(None if args.only_batch is None else (args.only_batch,))
        if args.command != "run-campaign-worker":
            campaign = campaign.with_wall_time(args.wall_time)
        remaining = (
            args.remaining_cases
            if args.command == "run-campaign-worker"
            else sum(
                not runtime_service.completed_case_is_valid(batch, case_index, storage_root=args.storage_root)
                for batch in campaign.batches
                for case_index in batch.case_indices
            )
        )
        cores_per_node = campaign.execution_values["cluster"]["cores_per_node"] if args.cores_per_node is None else args.cores_per_node
        campaign_plan = cluster_service.build_resource_plan(
            max_nodes=args.max_nodes,
            cases_per_node=args.cases_per_node,
            cores_per_case=args.cores_per_case,
            max_parallel_cases=args.max_parallel_cases,
            cores_per_node=cores_per_node,
            remaining_cases=remaining,
        )
        if args.command == "plan-campaign":
            plan = campaign_runtime.plan_campaign(
                campaign,
                resource_plan=campaign_plan,
                git_commit=args.git_commit,
                storage_root=args.storage_root,
            )
            print(json.dumps(plan, sort_keys=True))
            return 0
        if args.command == "submit-campaign":
            manifest = campaign_runtime.submit_campaign(
                campaign,
                resource_plan=campaign_plan,
                git_commit=args.git_commit,
                storage_root=args.storage_root,
            )
            print(json.dumps(manifest, sort_keys=True))
            return 0
        campaign_result = cluster_service.run_campaign_worker(
            campaign,
            plan=campaign_plan,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            scheduler_kind=args.scheduler_kind,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(
            json.dumps(
                {
                    "worker_index": campaign_result.worker_index,
                    "completed_or_skipped": len(campaign_result.completed_tasks),
                    "claimed_elsewhere": len(campaign_result.claimed_elsewhere),
                },
                sort_keys=True,
            )
        )
        return 0
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
        )
        print(json.dumps({"status": outcome.status, "case_id": outcome.case_id, "directory": str(outcome.processed_directory)}, sort_keys=True))
        return 0
    if args.command == "validate-case":
        provenance = runtime_service.validate_completed_case(config, args.case_index, storage_root=args.storage_root)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "case_id": provenance["case_id"],
                    "simulation_case_id": provenance["simulation_case_id"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-batch":
        selected = _selected(config, args)
        node_plan = _plan(config, args, selected)
        results = cluster_service.run_local_batch(
            config,
            selected,
            plan=node_plan,
            assignment_mode=args.assignment_mode,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(
            json.dumps(
                {
                    "node_workers": len(results),
                    "effective_parallel_cases": (node_plan.effective_parallel_cases),
                },
                sort_keys=True,
            )
        )
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
    previous_term: Any = None
    if args.command == "run-campaign-worker":
        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(
            signal.SIGTERM,
            lambda _signum, _frame: runtime_service.request_runtime_cancellation(),
        )
    try:
        return _dispatch(args)
    except Exception as error:  # noqa: BLE001 -- CLI boundary reports actionable domain errors
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
