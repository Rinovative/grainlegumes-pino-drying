"""
===============================================================================
cli_generation.py
===============================================================================
Expose the profile-qualified generation services through one thin command line.
Responsibilities:
  - Parse explicit configuration, case, campaign, pilot, and publication commands
  - Dispatch reusable generation services without duplicating their domain logic
  - Emit machine-readable command results and propagate terminal failures
Design principles:
  - Scientific, execution, and storage choices remain explicit command inputs
  - Validation and lifecycle authority stay in the responsible source services
  - Destructive cleanup requires the service-owned identity and confirmation gates
This module does NOT:
  - Define scientific values, sampling behavior, COMSOL mappings, or data schemas
  - Implement simulation, scheduling, persistence, or cleanup domain logic
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

from src import common
from src.generation import generation_benchmark as benchmark_service
from src.generation import generation_campaign as campaign_runtime
from src.generation import generation_campaign_status as campaign_status_service
from src.generation import generation_readiness as readiness_service
from src.generation import generation_smoke as smoke_service
from src.generation import generation_workflow as workflow_service
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_descriptors as contracts_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.publication import generation_publication_inventory as inventory_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_comsol as comsol_service
from src.generation.runtime import generation_runtime_preflight as preflight_service
from src.generation.runtime import generation_runtime_workspace as workspace_service
from src.generation.validation import generation_validation_pilot as pilot_service
from src.generation.validation import generation_validation_sentinels as sentinel_service


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


def _add_local_resources(parser: argparse.ArgumentParser) -> None:
    """Add genuine local-development core and concurrency controls."""
    parser.add_argument("--cores-per-case", type=int, required=True)
    parser.add_argument("--max-parallel-cases", type=int, required=True)


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 -- one centralized thin CLI parser
    """Build the complete generation command parser."""
    parser = argparse.ArgumentParser(description="Generate and run isolated profile-qualified COMSOL cases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate one generation configuration")
    validate.add_argument("config", type=Path)
    _add_batch_selection(validate, required=False)
    validate.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="validate structure and report unresolved launch gates without executing",
    )
    validate.add_argument(
        "--inspect-parameter",
        action="append",
        default=[],
        help="include complete resolved evidence for one canonical parameter or atomic record",
    )

    campaign_catalog = subparsers.add_parser(
        "list-campaigns",
        help="discover Generation campaigns by schema kind without relying on filenames",
    )
    campaign_catalog.add_argument(
        "--workflow",
        action="store_true",
        help="require the unique primary and technical-smoke profile pairs used by the host workflow",
    )

    static_sentinels = subparsers.add_parser(
        "static-sentinels",
        help="run configured-family and all-OOD-group generator checks without COMSOL",
    )
    static_sentinels.add_argument("steady_campaign", type=Path)
    static_sentinels.add_argument("transient_campaign", type=Path)

    readiness = subparsers.add_parser(
        "readiness-report",
        help="report exact missing launch values, mappings, and runtime evidence",
    )
    readiness.add_argument("steady_primary", type=Path)
    readiness.add_argument("transient_primary", type=Path)
    readiness.add_argument("--run-static-sentinels", action="store_true")
    readiness.add_argument("--real-runtime-receipt", type=Path)
    readiness.add_argument("--storage-root", type=Path)
    readiness.add_argument("--comsol-version-output")

    smoke_evidence = subparsers.add_parser(
        "technical-smoke-evidence-status",
        help="query successful technical-smoke evidence for one selected profile",
    )
    smoke_evidence.add_argument("config", type=Path)
    smoke_evidence.add_argument("--storage-root", type=Path, required=True)
    smoke_evidence.add_argument("--comsol-version-output", required=True)

    finalize_profile_smoke = subparsers.add_parser(
        "finalize-technical-smoke-evidence",
        help="publish profile-scoped evidence after a complete technical smoke",
    )
    finalize_profile_smoke.add_argument("campaign_run_id")
    finalize_profile_smoke.add_argument("--comsol-version-output", required=True)
    finalize_profile_smoke.add_argument("--storage-root", type=Path, required=True)

    finalize_smoke = subparsers.add_parser(
        "finalize-real-smoke",
        help="write one immutable paired native runtime-smoke receipt",
    )
    finalize_smoke.add_argument("steady_campaign_run_id")
    finalize_smoke.add_argument("transient_campaign_run_id")
    finalize_smoke.add_argument("--comsol-version-output", required=True)
    finalize_smoke.add_argument("--storage-root", type=Path, required=True)

    validate_smoke = subparsers.add_parser(
        "validate-real-smoke",
        help="revalidate one immutable runtime-smoke receipt against current source",
    )
    validate_smoke.add_argument("receipt", type=Path, nargs="?")
    validate_smoke.add_argument("--storage-root", type=Path, required=True)

    inspect_benchmark = subparsers.add_parser(
        "inspect-core-benchmark",
        help="resolve the same-case core-scaling suite without materializing inputs",
    )
    inspect_benchmark.add_argument("suite", type=Path)
    inspect_benchmark.add_argument("--variant")
    inspect_benchmark.add_argument("--require-executable", action="store_true")

    plan_benchmark = subparsers.add_parser(
        "plan-core-benchmark",
        help="validate native evidence and print isolated Slurm submissions",
    )
    plan_benchmark.add_argument("suite", type=Path)
    plan_benchmark.add_argument("--git-commit", required=True)
    plan_benchmark.add_argument("--variant")
    plan_benchmark.add_argument("--storage-root", type=Path, required=True)

    submit_benchmark = subparsers.add_parser(
        "submit-core-benchmark",
        help="submit the shared-case four-variant core-scaling benchmark",
    )
    submit_benchmark.add_argument("suite", type=Path)
    submit_benchmark.add_argument("--git-commit", required=True)
    submit_benchmark.add_argument("--variant")
    submit_benchmark.add_argument("--storage-root", type=Path, required=True)

    prepare_benchmark = subparsers.add_parser(
        "prepare-core-benchmark-case",
        help="materialize and prove the canonical benchmark case on a CPU node",
    )
    prepare_benchmark.add_argument("benchmark_run_id")
    prepare_benchmark.add_argument("--storage-root", type=Path, required=True)
    prepare_benchmark.add_argument("--work-root", type=Path, required=True)

    run_benchmark = subparsers.add_parser(
        "run-core-benchmark-repetition",
        help="run one isolated, identity-checked benchmark repetition",
    )
    run_benchmark.add_argument("benchmark_run_id")
    run_benchmark.add_argument("variant_id")
    run_benchmark.add_argument("repetition", type=int)
    run_benchmark.add_argument("--storage-root", type=Path, required=True)
    run_benchmark.add_argument("--work-root", type=Path, required=True)

    benchmark_status = subparsers.add_parser(
        "core-benchmark-status",
        help="reconstruct benchmark repetition and scheduler state",
    )
    benchmark_status.add_argument("benchmark_run_id")
    benchmark_status.add_argument("--no-scheduler", action="store_true")
    benchmark_status.add_argument("--format", choices=("json", "state"), default="json")
    benchmark_status.add_argument("--storage-root", type=Path, required=True)

    resume_benchmark = subparsers.add_parser(
        "resume-core-benchmark",
        help="submit only benchmark repetitions without successful evidence",
    )
    resume_benchmark.add_argument("benchmark_run_id")
    resume_benchmark.add_argument("--variant")
    resume_benchmark.add_argument("--storage-root", type=Path, required=True)

    finalize_benchmark = subparsers.add_parser(
        "finalize-core-benchmark",
        help="publish per-run and aggregate core-scaling evidence",
    )
    finalize_benchmark.add_argument("benchmark_run_id")
    finalize_benchmark.add_argument("--storage-root", type=Path, required=True)

    benchmark_transfer = subparsers.add_parser(
        "core-benchmark-transfer-plan",
        help="print the terminal benchmark evidence directory",
    )
    benchmark_transfer.add_argument("benchmark_run_id")
    benchmark_transfer.add_argument("--format", choices=("json", "tsv"), default="json")
    benchmark_transfer.add_argument("--storage-root", type=Path, required=True)

    publish_benchmark = subparsers.add_parser(
        "publish-transferred-core-benchmark",
        help="validate and atomically publish staged benchmark evidence",
    )
    publish_benchmark.add_argument("benchmark_run_id")
    publish_benchmark.add_argument("--staging-root", type=Path, required=True)
    publish_benchmark.add_argument("--destination-root", type=Path, required=True)
    publish_benchmark.add_argument("--source-host", required=True)
    publish_benchmark.add_argument("--source-storage-root", required=True)
    publish_benchmark.add_argument("--expected-inventory-sha256", required=True)
    publish_benchmark.add_argument("--expected-file-count", type=int, required=True)
    publish_benchmark.add_argument("--expected-size-bytes", type=int, required=True)

    validate_benchmark = subparsers.add_parser(
        "validate-core-benchmark",
        help="validate terminal benchmark evidence and dataset isolation",
    )
    validate_benchmark.add_argument("benchmark_run_id")
    validate_benchmark.add_argument("--storage-root", type=Path, required=True)

    summarize_benchmark = subparsers.add_parser(
        "core-benchmark-summary",
        help="display the terminal benchmark summary",
    )
    summarize_benchmark.add_argument("benchmark_run_id")
    summarize_benchmark.add_argument("--format", choices=("json", "markdown"), default="json")
    summarize_benchmark.add_argument("--storage-root", type=Path, required=True)

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

    input_cases = subparsers.add_parser(
        "generate-input-cases",
        help="generate canonical batch-oriented input-only cases",
    )
    input_cases.add_argument("config", type=Path)
    input_batch_selection = input_cases.add_mutually_exclusive_group(required=True)
    input_batch_selection.add_argument("--only-batch")
    input_batch_selection.add_argument("--all-batches", action="store_true")
    input_cases.add_argument("--only-regime", choices=("natural",))
    input_case_selection = input_cases.add_mutually_exclusive_group(required=True)
    input_case_selection.add_argument("--case-count", type=int)
    input_case_selection.add_argument("--all-cases", action="store_true")
    input_cases.add_argument("--case-start", type=int)
    input_cases.add_argument("--dry-run", action="store_true")
    input_cases.add_argument("--git-commit", required=True)
    input_cases.add_argument("--storage-root", type=Path, required=True)

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
    _add_local_resources(batch)
    _add_case_range(batch)
    _add_storage_arguments(batch, include_work=True)

    finalize = subparsers.add_parser("finalize-batch", help="validate and terminally publish one batch")
    finalize.add_argument("config", type=Path)
    _add_batch_selection(finalize, required=True)
    _add_storage_arguments(finalize)

    transfer = subparsers.add_parser("validate-transfer", help="validate one transferred terminal batch")
    transfer.add_argument("config", type=Path)
    _add_batch_selection(transfer, required=True)
    _add_storage_arguments(transfer)

    campaign_case = subparsers.add_parser(
        "run-campaign-case",
        help="run one exact campaign case in its persisted Slurm job",
    )
    campaign_case.add_argument("campaign_run_id")
    campaign_case.add_argument("batch_name")
    campaign_case.add_argument("case_index", type=int)
    _add_storage_arguments(campaign_case, include_work=True)

    campaign_plan = subparsers.add_parser(
        "plan-campaign",
        help="resolve paths and Slurm arguments without mutation",
    )
    campaign_plan.add_argument("config", type=Path)
    _add_batch_selection(campaign_plan, required=False)
    campaign_plan.add_argument(
        "--skip-extreme-family-ood",
        action="store_true",
        help="skip the canonical extreme-family batches for this execution only",
    )
    campaign_plan.add_argument("--pilot-cases-per-material", type=int)
    campaign_plan.add_argument("--git-commit", required=True)
    _add_storage_arguments(campaign_plan)

    submit_campaign = subparsers.add_parser("submit-campaign", help="submit and persist one exact-commit campaign run")
    submit_campaign.add_argument("config", type=Path)
    _add_batch_selection(submit_campaign, required=False)
    submit_campaign.add_argument(
        "--skip-extreme-family-ood",
        action="store_true",
        help="skip the canonical extreme-family batches for this execution only",
    )
    submit_campaign.add_argument("--pilot-cases-per-material", type=int)
    submit_campaign.add_argument("--git-commit", required=True)
    _add_storage_arguments(submit_campaign)

    campaign_status = subparsers.add_parser("campaign-status", help="reconstruct persistent campaign and scheduler status")
    campaign_status.add_argument("campaign_run_id")
    campaign_status.add_argument("--no-scheduler", action="store_true")
    campaign_status.add_argument("--format", choices=("json", "state", "summary", "monitor"), default="json")
    campaign_status.add_argument("--max-active-cases", type=int)
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

    feed = subparsers.add_parser(
        "feed-campaign",
        help="reconcile exact jobs and restore at most one pending case job",
    )
    feed.add_argument("campaign_run_id")
    _add_storage_arguments(feed)

    resume = subparsers.add_parser(
        "resume-campaign",
        help="explicitly retry at most one failed campaign case",
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

    pilot_source_inventory = subparsers.add_parser(
        "record-pilot-source-inventory",
        help="record exact terminal pilot CPU bytes before transfer or cleanup",
    )
    pilot_source_inventory.add_argument("campaign_run_id")
    _add_storage_arguments(pilot_source_inventory)

    pilot_staging_inventory = subparsers.add_parser(
        "record-pilot-staging-inventory",
        help="record exact transfer-staging bytes before publication or cleanup",
    )
    pilot_staging_inventory.add_argument("campaign_run_id")
    pilot_staging_inventory.add_argument("--staging-root", type=Path, required=True)

    validate_pilot_staging = subparsers.add_parser(
        "validate-pilot-staging-inventory",
        help="validate and report the retained pilot transfer staging",
    )
    validate_pilot_staging.add_argument("campaign_run_id")
    validate_pilot_staging.add_argument("--require-present", action="store_true")
    validate_pilot_staging.add_argument("--format", choices=("json", "tsv"), default="json")
    _add_storage_arguments(validate_pilot_staging)

    cleanup_pilot_staging = subparsers.add_parser(
        "cleanup-pilot-staging",
        help="transactionally remove the exact authorized pilot transfer staging",
    )
    cleanup_pilot_staging.add_argument("campaign_run_id")
    cleanup_pilot_staging.add_argument("--confirm", action="store_true")
    cleanup_pilot_staging.add_argument("--format", choices=("json", "tsv"), default="json")
    _add_storage_arguments(cleanup_pilot_staging)

    prepare_pilot = subparsers.add_parser(
        "prepare-pilot-check",
        help="analyze pilot evidence and persist the immutable pre-cleanup receipt",
    )
    prepare_pilot.add_argument("campaign_run_id")
    prepare_pilot.add_argument("--production-campaign", type=Path, required=True)
    prepare_pilot.add_argument("--keep-cpu-source", action="store_true")
    _add_storage_arguments(prepare_pilot)

    record_pilot_cleanup = subparsers.add_parser(
        "record-pilot-cleanup",
        help="finalize verified CPU and transfer-staging cleanup in the pilot receipt",
    )
    record_pilot_cleanup.add_argument("campaign_run_id")
    record_pilot_cleanup.add_argument("--cpu-source-removed", action="store_true")
    record_pilot_cleanup.add_argument("--cpu-bytes-reclaimed", type=int, required=True)
    record_pilot_cleanup.add_argument("--cpu-cleanup-receipt-sha256")
    record_pilot_cleanup.add_argument("--transfer-staging-removed", action="store_true")
    record_pilot_cleanup.add_argument("--staging-bytes-reclaimed", type=int, required=True)
    record_pilot_cleanup.add_argument("--staging-cleanup-receipt-sha256")
    _add_storage_arguments(record_pilot_cleanup)

    validate_pilot = subparsers.add_parser(
        "validate-pilot-check",
        help="revalidate and display one canonical pilot receipt",
    )
    validate_pilot.add_argument("campaign_run_id")
    validate_pilot.add_argument("--require-cleanup-complete", action="store_true")
    validate_pilot.add_argument(
        "--if-present",
        action="store_true",
        help="return successfully without output when the canonical receipt is absent",
    )
    validate_pilot.add_argument("--format", choices=("json", "summary"), default="json")
    _add_storage_arguments(validate_pilot)

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
    return config_service.load_generation_config(
        args.config,
        require_executable=not getattr(args, "allow_incomplete", False),
        only_batch=args.only_batch,
    )


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
) -> cluster_service.LocalResourcePlan:
    """Build one local-development plan from explicit local controls."""
    return cluster_service.build_local_resource_plan(
        cores_per_case=args.cores_per_case,
        max_parallel_cases=args.max_parallel_cases,
        remaining_cases=_remaining(config, selected, storage_root=args.storage_root),
    )


def _summary(config: config_service.GenerationConfig) -> dict[str, Any]:
    """Return one concise configuration preflight summary."""
    return {
        "simulation_profile": config.profile.id,
        "material_family": config.material_family,
        "material_role": config.material_role,
        "evaluation_regime": config.evaluation_regime,
        "sampling_regime": config.sampling_regime,
        "batch_name": config.batch_name,
        "batch_id": config.batch_id,
        "batch_storage_name": config.batch_storage_name,
        "batch_identity": config.batch_identity,
        "case_count": config.scientific_values["case_count"],
        "seed_base": config.seed_base,
        "template_path": str(config.template_path),
        "template_sha256": config.template_sha256,
    }


def _campaign_case_counts(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """Return batch, regime, and material counts from resolved batches."""
    by_batch: dict[str, int] = {}
    by_sampling_regime: dict[str, dict[str, int]] = {}
    by_material: dict[str, dict[str, int]] = {}
    for batch in campaign.batches:
        count = int(batch.scientific_values["case_count"])
        by_batch[batch.batch_name] = count
        by_sampling_regime.setdefault(batch.sampling_regime, {})[batch.material_family] = count
        material_counts = by_material.setdefault(batch.material_family, {})
        material_counts[batch.sampling_regime] = count
        material_counts["total"] = int(material_counts.get("total", 0)) + count
    return {
        "by_batch": by_batch,
        "by_sampling_regime": by_sampling_regime,
        "by_material": by_material,
        "derived_total": campaign.total_case_count,
    }


def _campaign_seed_plan(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """Return every authored and derived campaign-level sampling seed."""
    first_batch = campaign.batches[0]
    batch_seeds: dict[str, Any] = {}
    for batch in campaign.batches:
        blocks = batch.scientific_values["sampling"]["blocks"]
        batch_seeds[batch.batch_name] = {
            "batch_seed": batch.seed_base,
            "block_seeds": {
                name: {
                    "seed_origin": plan["seed_origin"],
                    "design_seed": plan["design_seed"],
                    "permutation_seed": plan["permutation_seed"],
                }
                for name, plan in blocks.items()
            },
        }
    return {
        "campaign_seed": first_batch.scientific_values["campaign_seed"],
        "membership_seed": campaign.membership.get("seed"),
        "paired_equivalence_seed": campaign.paired_equivalence_seed,
        "batches": batch_seeds,
        "case_seed_derivation": "batch_seed_and_case_identity",
    }


def _dataset_package_requests(campaign: config_service.CampaignConfig) -> list[dict[str, str]]:
    """Return unique authored package intents represented by resolved packages."""
    requests: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for package in campaign.dataset_packages:
        key = (str(package["evaluation_regime"]), str(package["source_role"]))
        if key not in seen:
            seen.add(key)
            requests.append(
                {
                    "evaluation_regime": key[0],
                    "source_role": key[1],
                }
            )
    return requests


def _parameter_ood_summary(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """Return resolved eligible units and deterministic case allocations."""
    first_policy = campaign.batches[0].scientific_values["parameter_ood"]
    policy_keys = (
        "groups",
        "units_per_case",
        "allocation_strategy",
        "eligibility_source",
    )
    batches: dict[str, Any] = {}
    for batch in campaign.batches:
        if batch.sampling_regime != "parameter_ood":
            continue
        resolved = batch.scientific_values["parameter_ood"]
        batches[batch.batch_name] = {
            "material_family": batch.material_family,
            "case_count": len(batch.case_indices),
            "eligible_units": resolved["eligible_units"],
            "allocation_counts": resolved["allocation_counts"],
            "case_allocation": resolved["case_allocation"],
        }
    return {
        "policy": {name: first_policy[name] for name in policy_keys},
        "batches": batches,
    }


def _campaign_summary(
    campaign: config_service.CampaignConfig,
    *,
    unresolved_gates: dict[str, list[str]],
) -> dict[str, Any]:
    """Return the complete resolved campaign inspection view."""
    case_counts = _campaign_case_counts(campaign)
    package_inventory = list(campaign.dataset_packages)
    material_inventory = [material_family for role in config_service.MATERIAL_ROLES for material_family in campaign.material_roles[role]]
    summary: dict[str, Any] = {
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "simulation_profile": campaign.profile.id,
        "source_path": str(campaign.source_path),
        "material_inventory": material_inventory,
        "material_roles": {role: list(families) for role, families in campaign.material_roles.items()},
        "material_memberships": {membership: list(families) for membership, families in campaign.material_memberships.items()},
        "membership": campaign.membership,
        "evaluation_regimes": list(campaign.evaluation_regimes),
        "case_counts": case_counts,
        "total_case_count": campaign.total_case_count,
        "counts": case_counts["by_batch"],
        "seed_plan": _campaign_seed_plan(campaign),
        "seeds": {batch.batch_name: batch.seed_base for batch in campaign.batches},
        "sampling_method": campaign.batches[0].scientific_values["sampling"]["method"],
        "parameter_ood": _parameter_ood_summary(campaign),
        "selected_batches": [_summary(batch) for batch in campaign.batches],
        "dataset_package_requests": _dataset_package_requests(campaign),
        "dataset_package_inventory": package_inventory,
        "dataset_packages": package_inventory,
        "profile": {
            "id": campaign.profile.id,
            "available_learning_views": list(campaign.profile.available_learning_views),
            "airflow_source": campaign.profile.airflow_source,
            "template_path": str(campaign.template_path),
            "template_sha256": campaign.template_sha256,
        },
        "execution_resources": {
            "site": campaign.execution_values["site"],
            "runtime": campaign.execution_values["runtime"],
            "submission": campaign.execution_values["submission"],
            "cluster": campaign.execution_values["cluster"],
            "retention_profile": campaign.execution_values["retention_profile"],
            "retention": campaign.execution_values["retention"],
        },
        "unresolved_readiness_gates": unresolved_gates,
        "executable": not any(unresolved_gates.values()),
        "cases_per_material": None,
        "pilot_plan": None,
        "technical_smoke_plan": None,
        "static_sentinel_workload": None,
    }
    if campaign.campaign_purpose == "family_generalization":
        summary["static_sentinel_workload"] = sentinel_service.inspect_sentinel_workload(campaign)
    elif campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
        pilot = campaign.batches[0].scientific_values["pilot_check"]
        summary["cases_per_material"] = pilot["cases_per_material"]
        summary["pilot_plan"] = {
            "material_inventory": material_inventory,
            "cases_per_material": pilot["cases_per_material"],
            "total_case_count": campaign.total_case_count,
            "campaign_seed": campaign.batches[0].scientific_values["campaign_seed"],
            "seed_namespace": campaign.batches[0].scientific_values["campaign_seed"],
            "case_semantics": pilot["case_semantics"],
            "case_kinds": pilot["case_kinds"],
            "training_membership": "none",
            "dataset_package_count": len(campaign.dataset_packages),
            "evaluation_regime": config_service.NO_EVALUATION_REGIME,
        }
    else:
        summary["technical_smoke_plan"] = {
            "material_inventory": material_inventory,
            "case_counts": case_counts["by_material"],
            "total_case_count": campaign.total_case_count,
            "campaign_seed": campaign.batches[0].scientific_values["campaign_seed"],
            "paired_equivalence_seed": campaign.paired_equivalence_seed,
            "learning_membership": "none",
            "dataset_package_count": len(campaign.dataset_packages),
        }
    return summary


def _campaign_catalog(*, require_workflow: bool) -> dict[str, Any]:
    """Return discovered campaign metadata and an optional unique workflow view."""
    repository = common.paths.get_project_root().resolve()
    campaigns = config_service.discover_campaign_configs(
        repository,
        require_executable=False,
    )
    records: list[dict[str, Any]] = []
    for campaign in campaigns:
        profile_contract = contracts_service.get_profile_contract(campaign.profile.id)
        profile_kind = "transient" if profile_contract.transient_fields else "stationary"
        records.append(
            {
                "source_path": str(campaign.source_path),
                "repository_path": campaign.source_path.relative_to(repository).as_posix(),
                "campaign_name": campaign.campaign_name,
                "campaign_purpose": campaign.campaign_purpose,
                "simulation_profile": campaign.profile.id,
                "profile_kind": profile_kind,
                "execution_site": campaign.execution_values["site"],
            }
        )
    result: dict[str, Any] = {
        "schema_kind": "generation_campaign_catalog",
        "schema_version": 1,
        "campaigns": records,
    }
    if not require_workflow:
        return result
    workflow: dict[str, dict[str, dict[str, Any]]] = {}
    for purpose in ("family_generalization", "technical_runtime_smoke"):
        purpose_records = [record for record in records if record["campaign_purpose"] == purpose]
        selected: dict[str, dict[str, Any]] = {}
        for profile_kind in ("stationary", "transient"):
            matches = [record for record in purpose_records if record["profile_kind"] == profile_kind]
            if len(matches) != 1:
                message = f"Host workflow requires exactly one {purpose!r} {profile_kind} campaign; discovered {len(matches)}."
                raise ValueError(message)
            selected[profile_kind] = matches[0]
        workflow[purpose] = selected
    selected_sites = {
        common.serialization.canonical_json_sha256(record["execution_site"]): record["execution_site"]
        for purpose_records in workflow.values()
        for record in purpose_records.values()
    }
    if len(selected_sites) != 1:
        message = "Host workflow campaign pairs must resolve one shared execution site."
        raise ValueError(message)
    result["workflow"] = workflow
    result["shared_execution_site"] = next(iter(selected_sites.values()))
    return result


def _dispatch(args: argparse.Namespace) -> int:  # noqa: C901, PLR0911, PLR0912, PLR0915 -- thin CLI command dispatch
    """Dispatch one parsed command to its authoritative service."""
    if args.command == "list-campaigns":
        print(json.dumps(_campaign_catalog(require_workflow=args.workflow), sort_keys=True))
        return 0
    if args.command == "static-sentinels":
        report = sentinel_service.run_static_sentinels(
            args.steady_campaign,
            args.transient_campaign,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "pass" else 2
    if args.command == "readiness-report":
        report = readiness_service.build_readiness_report(
            args.steady_primary,
            args.transient_primary,
            run_static_sentinels=args.run_static_sentinels,
            real_runtime_receipt=args.real_runtime_receipt,
            storage_root=args.storage_root,
            comsol_version_output=args.comsol_version_output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["production_ready_for_user_launch"] else 2
    if args.command == "technical-smoke-evidence-status":
        report = smoke_service.technical_smoke_evidence_status(
            args.config,
            storage_root=args.storage_root,
            comsol_version_output=args.comsol_version_output,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "technical_smoke_evidence_valid" else 2
    if args.command == "finalize-technical-smoke-evidence":
        path = smoke_service.finalize_technical_smoke_evidence(
            args.campaign_run_id,
            storage_root=args.storage_root,
            comsol_version_output=args.comsol_version_output,
        )
        print(path)
        return 0
    if args.command == "finalize-real-smoke":
        path = smoke_service.finalize_real_smoke(
            args.steady_campaign_run_id,
            args.transient_campaign_run_id,
            comsol_version_output=args.comsol_version_output,
            storage_root=args.storage_root,
        )
        print(path)
        return 0
    if args.command == "validate-real-smoke":
        report = (
            smoke_service.validate_current_real_smoke_receipts(
                storage_root=args.storage_root,
            )
            if args.receipt is None
            else smoke_service.validate_real_smoke_receipt(
                args.receipt,
                storage_root=args.storage_root,
            )
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "inspect-core-benchmark":
        inspection = benchmark_service.inspect_core_benchmark(
            args.suite,
            variant_id=args.variant,
            require_executable=args.require_executable,
        )
        print(json.dumps(inspection, sort_keys=True))
        return 0
    if args.command == "plan-core-benchmark":
        plan = benchmark_service.plan_core_benchmark(
            args.suite,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            variant_id=args.variant,
        )
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.command == "submit-core-benchmark":
        manifest = benchmark_service.submit_core_benchmark(
            args.suite,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            variant_id=args.variant,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "prepare-core-benchmark-case":
        path = benchmark_service.prepare_core_benchmark_case(
            args.benchmark_run_id,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(path)
        return 0
    if args.command == "run-core-benchmark-repetition":
        result = benchmark_service.run_core_benchmark_repetition(
            args.benchmark_run_id,
            args.variant_id,
            args.repetition,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "core-benchmark-status":
        status = benchmark_service.core_benchmark_status(
            args.benchmark_run_id,
            storage_root=args.storage_root,
            query_scheduler=not args.no_scheduler,
        )
        if args.format == "state":
            print(status["state"])
        else:
            print(json.dumps(status, sort_keys=True))
        return 0
    if args.command == "resume-core-benchmark":
        manifest = benchmark_service.resume_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
            variant_id=args.variant,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "finalize-core-benchmark":
        summary = benchmark_service.finalize_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "core-benchmark-transfer-plan":
        plan = benchmark_service.core_benchmark_transfer_plan(
            args.benchmark_run_id,
            storage_root=args.storage_root,
        )
        if args.format == "json":
            print(json.dumps(plan, sort_keys=True))
        else:
            print(
                "\t".join(
                    (
                        "benchmark",
                        str(plan["benchmark_run_id"]),
                        str(plan["git_commit"]),
                        str(plan["relative_directory"]),
                        str(plan["inventory"]["inventory_sha256"]),
                        str(plan["inventory"]["file_count"]),
                        str(plan["inventory"]["size_bytes"]),
                    )
                )
            )
        return 0
    if args.command == "publish-transferred-core-benchmark":
        receipt = benchmark_service.publish_transferred_core_benchmark(
            args.benchmark_run_id,
            staging_root=args.staging_root,
            destination_root=args.destination_root,
            source_host=args.source_host,
            source_storage_root=args.source_storage_root,
            expected_inventory_sha256=args.expected_inventory_sha256,
            expected_file_count=args.expected_file_count,
            expected_size_bytes=args.expected_size_bytes,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "validate-core-benchmark":
        result = benchmark_service.validate_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "core-benchmark-summary":
        summary = benchmark_service.load_core_benchmark_summary(
            args.benchmark_run_id,
            storage_root=args.storage_root,
        )
        if args.format == "json":
            print(json.dumps(summary, sort_keys=True))
        else:
            print(benchmark_service.core_benchmark_markdown(summary), end="")
        return 0
    if args.command == "preflight":
        report = preflight_service.run_cpu_preflight(
            args.config,
            only_batch=args.only_batch,
            storage_root=args.storage_root,
            work_root=args.work_root,
            venv_path=args.venv_path,
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
        campaign = config_service.load_campaign_config(
            args.config,
            require_executable=not args.allow_incomplete,
        )
        gates = readiness_service.campaign_unresolved_gates(args.config)
        summary = _campaign_summary(campaign, unresolved_gates=gates)
        if args.inspect_parameter:
            summary["parameter_inspection"] = [inventory_service.inspect_campaign_parameter(campaign, name) for name in args.inspect_parameter]
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "campaign-status":
        status = campaign_runtime.campaign_status(
            args.campaign_run_id,
            storage_root=args.storage_root,
            query_scheduler=not args.no_scheduler,
        )
        if args.max_active_cases is not None and args.format not in {"summary", "monitor"}:
            message = "--max-active-cases requires --format summary or monitor."
            raise ValueError(message)
        if args.format == "state":
            print(status["campaign_state"])
        elif args.format == "summary":
            print(
                campaign_status_service.format_campaign_status_summary(
                    status,
                    max_active_cases=args.max_active_cases,
                )
            )
        elif args.format == "monitor":
            print(
                campaign_status_service.format_campaign_monitor(
                    status,
                    max_active_cases=(8 if args.max_active_cases is None else args.max_active_cases),
                )
            )
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
    if args.command == "feed-campaign":
        manifest = campaign_runtime.feed_campaign(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(manifest, sort_keys=True))
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
    if args.command == "record-pilot-source-inventory":
        inventory = pilot_service.record_cpu_source_inventory(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(inventory, sort_keys=True))
        return 0
    if args.command == "record-pilot-staging-inventory":
        inventory = pilot_service.record_transfer_staging_inventory(
            args.campaign_run_id,
            staging_root=args.staging_root,
        )
        print(json.dumps(inventory, sort_keys=True))
        return 0
    if args.command == "validate-pilot-staging-inventory":
        inventory = pilot_service.validate_transfer_staging_inventory(
            args.campaign_run_id,
            storage_root=args.storage_root,
            require_staging_present=args.require_present,
        )
        if args.format == "json":
            print(json.dumps(inventory, sort_keys=True))
        else:
            print(
                "\t".join(
                    (
                        "pilot-staging",
                        inventory["transfer_staging_path"],
                        str(inventory["transfer_staging_bytes_before_cleanup"]),
                        str(inventory["transfer_staging_file_count"]),
                    )
                )
            )
        return 0
    if args.command == "cleanup-pilot-staging":
        receipt = pilot_service.cleanup_recorded_transfer_staging(
            args.campaign_run_id,
            storage_root=args.storage_root,
            confirm=args.confirm,
        )
        if args.format == "json":
            print(json.dumps(receipt, sort_keys=True))
        else:
            print(
                "\t".join(
                    (
                        "pilot-staging-cleanup",
                        str(receipt["status"]),
                        str(receipt["removed"]),
                        str(receipt["reclaimed_bytes"]),
                        (
                            common.serialization.file_sha256(
                                pilot_service.pilot_check_directory(
                                    args.campaign_run_id,
                                    storage_root=args.storage_root,
                                )
                                / pilot_service.PILOT_STAGING_CLEANUP_FILENAME
                            )
                            if receipt["status"] == "complete"
                            else "-"
                        ),
                    )
                )
            )
        return 0
    if args.command == "prepare-pilot-check":
        receipt = pilot_service.prepare_pilot_receipt(
            args.campaign_run_id,
            production_campaign=args.production_campaign,
            storage_root=args.storage_root,
            cleanup_requested=not args.keep_cpu_source,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "record-pilot-cleanup":
        receipt = workflow_service.record_pilot_cleanup_result(
            args.campaign_run_id,
            storage_root=args.storage_root,
            cpu_source_removed=args.cpu_source_removed,
            cpu_bytes_reclaimed=args.cpu_bytes_reclaimed,
            cpu_cleanup_receipt_sha256=args.cpu_cleanup_receipt_sha256,
            transfer_staging_removed=args.transfer_staging_removed,
            staging_bytes_reclaimed=args.staging_bytes_reclaimed,
            staging_cleanup_receipt_sha256=args.staging_cleanup_receipt_sha256,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "validate-pilot-check":
        receipt_path = pilot_service.pilot_receipt_path(
            args.campaign_run_id,
            storage_root=args.storage_root,
        )
        if args.if_present and not receipt_path.is_file():
            return 0
        receipt = (
            workflow_service.validate_completed_pilot_receipt(
                args.campaign_run_id,
                storage_root=args.storage_root,
            )
            if args.require_cleanup_complete
            else pilot_service.validate_pilot_receipt(
                args.campaign_run_id,
                storage_root=args.storage_root,
            )
        )
        if args.format == "json":
            print(json.dumps(receipt, sort_keys=True))
        else:
            print(pilot_service.terminal_summary(receipt))
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
    if args.command == "run-campaign-case":
        outcome = campaign_runtime.run_campaign_case_job(
            args.campaign_run_id,
            args.batch_name,
            args.case_index,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(
            json.dumps(
                {
                    "status": outcome.status,
                    "case_id": outcome.case_id,
                    "directory": str(outcome.processed_directory),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"plan-campaign", "submit-campaign"}:
        if args.skip_extreme_family_ood and args.only_batch is not None:
            message = "--skip-extreme-family-ood cannot be combined with --only-batch."
            raise ValueError(message)
        if args.pilot_cases_per_material is not None and (args.skip_extreme_family_ood or args.only_batch is not None):
            message = "--pilot-cases-per-material cannot be combined with --skip-extreme-family-ood or --only-batch."
            raise ValueError(message)
        campaign = config_service.load_campaign_config(
            args.config,
            pilot_cases_per_material=args.pilot_cases_per_material,
        )
        if campaign.campaign_purpose == config_service.PILOT_CAMPAIGN_PURPOSE:
            if args.skip_extreme_family_ood or args.only_batch is not None:
                message = "Pilot-check campaigns do not support extreme-family skipping or batch selection."
                raise ValueError(message)
        else:
            if args.skip_extreme_family_ood:
                campaign = campaign.without_extreme_family_ood()
            campaign = campaign.select_batches(None if args.only_batch is None else (args.only_batch,))
        if args.command == "plan-campaign":
            plan = campaign_runtime.plan_campaign(
                campaign,
                git_commit=args.git_commit,
                storage_root=args.storage_root,
            )
            print(json.dumps(plan, sort_keys=True))
            return 0
        manifest = campaign_runtime.submit_campaign(
            campaign,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "generate-input-cases":
        action = "dry_run" if args.dry_run else "execute"
        response = input_service.run_campaign_input_generation(
            input_service.CampaignInputGenerationRequest(
                campaign_config=args.config,
                storage_root=args.storage_root,
                action=action,
                only_batch=args.only_batch,
                all_batches=args.all_batches,
                only_regime=args.only_regime,
                case_start=args.case_start,
                case_count=args.case_count,
                all_cases=args.all_cases,
                git_commit=args.git_commit,
            )
        )
        print(json.dumps(response, sort_keys=True))
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
        prepared = runtime_service.prepare_case_work_directory(
            config,
            args.case_index,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(prepared.work_directory)
        return 0
    if args.command == "print-command":
        if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
            with tempfile.TemporaryDirectory(prefix="generation-print-command-") as temporary_directory:
                bundle = case_service.generate_case_input_bundle(
                    config,
                    args.case_index,
                    Path(temporary_directory),
                )
                scalar_handoff = bundle.scalar_handoff
                if scalar_handoff is None:
                    message = "Transient print-command generation produced no scalar handoff."
                    raise RuntimeError(message)
                scalar_handoff_contract.validate_transient_scalar_source(scalar_handoff)
                command = comsol_service.build_comsol_command(
                    config,
                    cores_per_case=args.cores_per_case,
                    scalar_handoff=scalar_handoff,
                    scheduler_kind=args.scheduler_kind,
                )
        else:
            config.case_id(args.case_index)
            command = comsol_service.build_comsol_command(
                config,
                cores_per_case=args.cores_per_case,
                scheduler_kind=args.scheduler_kind,
            )
        print(shlex.join(command))
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
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(
            json.dumps(
                {
                    "case_outcomes": len(results),
                    "effective_parallel_cases": node_plan.effective_parallel_cases,
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
        print(
            json.dumps(
                {
                    "status": "valid",
                    "batch_id": manifest["batch_id"],
                    "batch_storage_name": manifest["batch_storage_name"],
                    "case_count": len(manifest["cases"]),
                },
                sort_keys=True,
            )
        )
        return 0
    message = f"Unsupported generation command: {args.command!r}."
    raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    """Run one generation command and translate failures to process status two."""
    args = _build_parser().parse_args(argv)
    previous_term: Any = None
    if args.command in {"run-campaign-case", "run-core-benchmark-repetition"}:
        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(
            signal.SIGTERM,
            lambda _signum, _frame: runtime_service.request_runtime_cancellation(),
        )
    try:
        return _dispatch(args)
    except Exception as error:  # noqa: BLE001 -- CLI boundary reports actionable domain errors
        if args.command == "validate-config":
            details = config_service.validation_error_details(args.config, error)
            print(json.dumps(details, sort_keys=True), file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return 2
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
