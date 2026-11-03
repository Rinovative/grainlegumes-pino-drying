"""Expose the profile-qualified generation services through one thin command line.

Responsibilities:
  - Parse explicit configuration, case, campaign, and publication commands
  - Dispatch reusable generation services without duplicating their domain logic
  - Emit machine-readable command results to their callers and one plain line per case and node-worker outcome to the node log
  - Link a campaign node job's own logs into its campaign when the node worker starts
  - Propagate terminal failures

Design principles:
  - Scientific, execution, and storage choices remain explicit command inputs
  - Validation and lifecycle authority stay in the responsible source services
  - Destructive cleanup requires the service-owned identity and confirmation gates

This module does NOT:
  - Define scientific values, sampling behavior, COMSOL mappings, or data schemas
  - Implement simulation, scheduling, persistence, or cleanup domain logic
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import common
from src.generation import generation_benchmark as benchmark_service
from src.generation import generation_benchmark_config as benchmark_config
from src.generation import generation_campaign as campaign_runtime
from src.generation import generation_campaign_status as campaign_status_service
from src.generation import generation_readiness as readiness_service
from src.generation import generation_run as run_service
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_descriptors as contracts_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_comsol as comsol_service
from src.generation.runtime import generation_runtime_host as host_service
from src.generation.runtime import generation_runtime_node as node_service
from src.generation.runtime import generation_runtime_preflight as preflight_service
from src.generation.runtime import generation_runtime_workspace as workspace_service
from src.generation.validation import generation_validation_sentinels as sentinel_service

if TYPE_CHECKING:
    from collections.abc import Mapping

_PROCESS_STARTED = time.monotonic()  # one command is one process, so this is the start of a case step


def _case_label(args: argparse.Namespace) -> str:
    """Return ``<material> <case id>`` of the campaign case a command runs, from its ``<profile>__<material>__<regime>[__...]`` batch name."""
    parts = args.batch_name.split("__")
    material = parts[1] if len(parts) > 2 else args.batch_name  # noqa: PLR2004 -- the material is the second of at least three name parts
    return f"{material} case_{args.case_index:04d}"


def _case_end_line(args: argparse.Namespace, status: str) -> str:
    """Return the one searchable line that ends a case step in the node log."""
    duration = campaign_status_service.format_duration(time.monotonic() - _PROCESS_STARTED)
    return f"CASE END {_case_label(args)} {status} {duration}"


def _add_storage_arguments(parser: argparse.ArgumentParser, *, include_work: bool = False) -> None:
    """Add shared storage and optional work-root boundaries."""
    parser.add_argument("--storage-root", type=Path)
    if include_work:
        parser.add_argument("--work-root", type=Path)


_WORKFLOW_PROGRESS_FIELDS = (
    "operation",
    "campaign_run_id",
    "dataset_id",
    "files_validated",
    "files_total",
    "bytes_validated",
    "bytes_total",
    "packages_completed",
    "packages_total",
    "batches_completed",
    "batches_total",
    "cases_completed",
    "cases_total",
    "generated_cases",
    "reused_cases",
    "cases_packed",
    "source_bytes_read",
    "shards_completed",
    "shards_total",
    "bytes_written",
    "training_payloads_completed",
    "training_payloads_total",
    "work_units_admitted",
    "work_units_total",
    "scheduler_submissions",
    "jobs_running",
    "jobs_pending",
    "jobs_terminal",
    "heartbeat",
    "source_host_dependency",
    "eta",
)


def _print_workflow_progress(event: Mapping[str, Any]) -> None:
    """Emit one bounded human-readable workflow progress line to stderr."""
    fields = []
    for name in _WORKFLOW_PROGRESS_FIELDS:
        value = event.get(name)
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, int | float | str):
            rendered = str(value)
        else:
            continue
        if any(character in rendered for character in "\r\n\t"):
            continue
        fields.append(f"{name}={shlex.quote(rendered)}")
    print(
        "progress " + " ".join(fields),
        file=sys.stderr,
        flush=True,
    )


def _add_batch_selection(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add selection of one predeclared campaign batch."""
    parser.add_argument("--only-batch", required=required)


def _add_scientific_overrides(parser: argparse.ArgumentParser) -> None:
    """Expose finite original-design and physical-duration overrides and the Source revision."""
    parser.add_argument("--materials", nargs="+")
    parser.add_argument("--cases-per-material", type=int)
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--revision")


def _add_execution_overrides(parser: argparse.ArgumentParser) -> None:
    """Expose explicit ordinary concurrency, CPU and failure-budget controls."""
    parser.add_argument("--node-jobs", type=int)
    parser.add_argument("--max-admission-cases", type=int)
    parser.add_argument("--cores-per-case", type=int)
    parser.add_argument("--case-timeout-seconds", type=float)
    parser.add_argument("--no-retry", action="store_true")


def _execution_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return requested resource values separately from scientific overrides."""
    result = {
        key: getattr(args, key)
        for key in ("node_jobs", "max_admission_cases", "cores_per_case", "case_timeout_seconds")
        if getattr(args, key, None) is not None
    }
    if getattr(args, "no_retry", False):
        result["no_retry"] = True
    return result


def _scientific_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return explicitly supplied scientific values without inventing defaults."""
    return {
        key: getattr(args, key) for key in ("materials", "cases_per_material", "duration_hours", "revision") if getattr(args, key, None) is not None
    }


def _add_campaign_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Add exact campaign selection and source-identity arguments."""
    parser.add_argument("config", type=Path)
    parser.add_argument("--authored-value-json", type=json.loads)
    _add_batch_selection(parser, required=False)
    parser.add_argument("--git-commit", required=True)
    _add_scientific_overrides(parser)
    _add_execution_overrides(parser)
    _add_storage_arguments(parser)


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 -- one centralized thin CLI parser
    """Build the complete generation command parser."""
    parser = argparse.ArgumentParser(description="Generate and run isolated profile-qualified COMSOL cases")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_run = subparsers.add_parser(
        "resolve-generation-run",
        help="resolve one campaign or benchmark into the common immutable run plan",
    )
    resolve_run.add_argument("config", type=Path)
    resolve_run.add_argument("--authored-value-json", type=json.loads)
    resolve_run.add_argument("--git-commit", required=True)
    resolve_run.add_argument("--allow-incomplete", action="store_true")
    _add_scientific_overrides(resolve_run)
    _add_execution_overrides(resolve_run)

    validate = subparsers.add_parser("validate-config", help="validate one generation configuration")
    validate.add_argument("config", type=Path)
    validate.add_argument("--authored-value-json", type=json.loads)
    _add_batch_selection(validate, required=False)
    _add_scientific_overrides(validate)
    _add_execution_overrides(validate)
    validate.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="validate structure and report unresolved launch gates without executing",
    )

    subparsers.add_parser(
        "list-campaigns",
        help="discover Generation campaigns by schema kind without relying on filenames",
    )

    inspect_benchmark = subparsers.add_parser(
        "inspect-core-benchmark",
        help="resolve the same-case core-scaling suite without materializing inputs",
    )
    inspect_benchmark.add_argument("suite", type=Path)
    inspect_benchmark.add_argument("--require-executable", action="store_true")

    resolve_benchmark = subparsers.add_parser(
        "resolve-core-benchmark-run",
        help="resolve the deterministic benchmark runtime identity without mutation",
    )
    resolve_benchmark.add_argument("suite", type=Path)
    resolve_benchmark.add_argument("--git-commit", required=True)
    resolve_benchmark.add_argument("--comsol-version-output", required=True)

    preflight_benchmark = subparsers.add_parser(
        "preflight-core-benchmark",
        help="persist benchmark-owned exact-source and runtime preflight evidence",
    )
    preflight_benchmark.add_argument("suite", type=Path)
    preflight_benchmark.add_argument("--git-commit", required=True)
    preflight_benchmark.add_argument("--storage-root", type=Path, required=True)
    preflight_benchmark.add_argument("--scratch-root", type=Path, required=True)
    preflight_benchmark.add_argument("--comsol-version-output", required=True)
    preflight_benchmark.add_argument(
        "--comsol-executable-path",
        type=Path,
        required=True,
    )

    plan_benchmark = subparsers.add_parser(
        "plan-core-benchmark",
        help="reuse standalone preflight and print isolated Slurm submissions",
    )
    plan_benchmark.add_argument("suite", type=Path)
    plan_benchmark.add_argument("--git-commit", required=True)
    plan_benchmark.add_argument("--storage-root", type=Path, required=True)
    plan_benchmark.add_argument("--scratch-root", type=Path, required=True)
    plan_benchmark.add_argument("--comsol-version-output", required=True)
    plan_benchmark.add_argument(
        "--comsol-executable-path",
        type=Path,
        required=True,
    )

    materialize_benchmark = subparsers.add_parser(
        "materialize-core-benchmark-inputs",
        help="materialize the canonical benchmark input without submitting work",
    )
    materialize_benchmark.add_argument("suite", type=Path)
    materialize_benchmark.add_argument("--git-commit", required=True)
    materialize_benchmark.add_argument("--storage-root", type=Path, required=True)
    materialize_benchmark.add_argument("--scratch-root", type=Path, required=True)
    materialize_benchmark.add_argument("--comsol-version-output", required=True)
    materialize_benchmark.add_argument(
        "--comsol-executable-path",
        type=Path,
        required=True,
    )

    submit_benchmark = subparsers.add_parser(
        "submit-core-benchmark",
        help="submit the standalone shared-case core-scaling benchmark",
    )
    submit_benchmark.add_argument("suite", type=Path)
    submit_benchmark.add_argument("--git-commit", required=True)
    submit_benchmark.add_argument("--storage-root", type=Path, required=True)
    submit_benchmark.add_argument("--scratch-root", type=Path, required=True)
    submit_benchmark.add_argument("--comsol-version-output", required=True)
    submit_benchmark.add_argument(
        "--comsol-executable-path",
        type=Path,
        required=True,
    )

    run_benchmark = subparsers.add_parser(
        "run-core-benchmark-case",
        help="run one isolated, identity-checked benchmark representative case",
    )
    run_benchmark.add_argument("benchmark_run_id")
    run_benchmark.add_argument("sweep_id")
    run_benchmark.add_argument("variant_id")
    run_benchmark.add_argument("variant_attempt", type=int)
    run_benchmark.add_argument("case_role")
    run_benchmark.add_argument("--storage-root", type=Path, required=True)
    run_benchmark.add_argument("--work-root", type=Path, required=True)

    benchmark_status = subparsers.add_parser(
        "core-benchmark-status",
        help="reconstruct benchmark work-unit and scheduler state",
    )
    benchmark_status.add_argument("benchmark_run_id")
    benchmark_status.add_argument("--no-scheduler", action="store_true")
    benchmark_status.add_argument("--format", choices=("json", "state", "summary", "monitor"), default="json")
    benchmark_status.add_argument("--storage-root", type=Path, required=True)

    resume_benchmark = subparsers.add_parser(
        "resume-core-benchmark",
        help="submit only benchmark cases without successful evidence",
    )
    resume_benchmark.add_argument("benchmark_run_id")
    resume_benchmark.add_argument("--storage-root", type=Path, required=True)
    resume_benchmark.add_argument("--scratch-root", type=Path)

    cancel_benchmark = subparsers.add_parser(
        "cancel-core-benchmark",
        help="cancel only Slurm jobs owned by one core benchmark",
    )
    cancel_benchmark.add_argument("benchmark_run_id")
    cancel_benchmark.add_argument("--force", action="store_true")
    cancel_benchmark.add_argument("--storage-root", type=Path, required=True)

    finalize_benchmark = subparsers.add_parser(
        "finalize-core-benchmark",
        help="publish per-run and aggregate core-scaling evidence",
    )
    finalize_benchmark.add_argument("benchmark_run_id")
    finalize_benchmark.add_argument("--storage-root", type=Path, required=True)

    benchmark_source_status = subparsers.add_parser(
        "core-benchmark-source-status",
        help="report benchmark CPU source and cleanup state",
    )
    benchmark_source_status.add_argument("benchmark_run_id")
    benchmark_source_status.add_argument("--no-scheduler", action="store_true")
    benchmark_source_status.add_argument("--format", choices=("json", "tsv"), default="json")
    benchmark_source_status.add_argument("--storage-root", type=Path, required=True)

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
    _add_campaign_execution_arguments(campaign_plan)

    prepare_campaign_inputs = subparsers.add_parser(
        "prepare-campaign-inputs",
        help="prepare or reuse exact canonical inputs before submission",
    )
    _add_campaign_execution_arguments(prepare_campaign_inputs)

    submit_campaign = subparsers.add_parser(
        "submit-campaign",
        help="submit and persist one exact-commit campaign run",
    )
    _add_campaign_execution_arguments(submit_campaign)
    submit_campaign.add_argument("--metadata-only", action="store_true", help="feed prepared metadata and report compute-required before heavy work")
    submit_campaign.add_argument(
        "--inputs-prepared",
        action="store_true",
        help="admit immutable inputs prepared by the immediately preceding workflow stage",
    )

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
    cancel.add_argument("--force", action="store_true")
    _add_storage_arguments(cancel)

    feed = subparsers.add_parser(
        "feed-campaign",
        help="reconcile node jobs and dispatches and dispatch pending cases into node jobs",
    )
    feed.add_argument("campaign_run_id")
    _add_storage_arguments(feed)

    declare_failure = subparsers.add_parser(
        "declare-candidate-failure",
        help="explicitly terminalize one exhausted scientific candidate after scheduler reconciliation",
    )
    declare_failure.add_argument("campaign_run_id")
    declare_failure.add_argument("material")
    declare_failure.add_argument("case_id")
    declare_failure.add_argument("--reason-code", required=True)
    declare_failure.add_argument("--detail", required=True)
    _add_storage_arguments(declare_failure)

    amend_replacements = subparsers.add_parser(
        "amend-replacement-pool",
        help="explicitly raise one original batch's replacement capacity, with its reason",
    )
    amend_replacements.add_argument("campaign_run_id")
    amend_replacements.add_argument("--material", required=True)
    amend_replacements.add_argument("--regime", required=True)
    amend_replacements.add_argument("--capacity", type=int, required=True, help="total replacement candidates the batch may use")
    amend_replacements.add_argument("--reason", required=True)
    _add_storage_arguments(amend_replacements)

    resume = subparsers.add_parser(
        "resume-campaign",
        help="advance one case through the campaign resume matrix",
    )
    resume.add_argument("campaign_run_id")
    resume.add_argument("--metadata-only", action="store_true", help="defer input generation or batch admission and report compute-required")
    _add_storage_arguments(resume)

    interruption = subparsers.add_parser(
        "record-worker-interruption",
        help="persist best-effort Slurm worker interruption evidence",
    )
    interruption.add_argument("campaign_run_id")
    interruption.add_argument("--signal", required=True)
    interruption.add_argument("--exit-code", type=int, required=True)
    _add_storage_arguments(interruption)

    node_worker = subparsers.add_parser(
        "run-node-worker",
        help="run the generic ticket worker inside one Slurm node job",
    )
    node_worker.add_argument("node_directory", type=Path)
    node_worker.add_argument("--cores", type=int, required=True)

    return parser


def _load(args: argparse.Namespace) -> config_service.GenerationConfig:
    """Load one predeclared batch referenced by a parsed command."""
    return config_service.load_generation_config(
        args.config,
        require_executable=not getattr(args, "allow_incomplete", False),
        only_batch=args.only_batch,
    )


def _summary(config: config_service.GenerationConfig) -> dict[str, Any]:
    """Return one concise configuration preflight summary."""
    return {
        "simulation_profile": config.profile.id,
        "material_family": config.material_family,
        "domain_profile": config.scientific_values["domain_profile"],
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
        "batches": batch_seeds,
        "case_seed_derivation": "batch_seed_and_case_identity",
    }


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
    material_inventory = list(campaign.material_inventory)
    summary: dict[str, Any] = {
        "campaign_name": campaign.campaign_name,
        "source_name": campaign.source_name,
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "simulation_profile": campaign.profile.id,
        "source_path": str(campaign.source_path),
        "material_inventory": material_inventory,
        "domain_profile": campaign.domain_profile,
        "original_design_sizes": dict(campaign.selected_targets),
        "case_counts": case_counts,
        "total_case_count": campaign.total_case_count,
        "counts": case_counts["by_batch"],
        "seed_plan": _campaign_seed_plan(campaign),
        "seeds": {batch.batch_name: batch.seed_base for batch in campaign.batches},
        "sampling_method": campaign.batches[0].scientific_values["sampling"]["method"],
        "parameter_ood": _parameter_ood_summary(campaign),
        "selected_batches": [_summary(batch) for batch in campaign.batches],
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
            "retention_policy": campaign.execution_values["retention_policy"],
        },
        "unresolved_readiness_gates": unresolved_gates,
        "executable": not any(unresolved_gates.values()),
        "static_sentinel_workload": None,
    }
    if campaign.campaign_purpose == "source_generation":
        summary["static_sentinel_workload"] = sentinel_service.inspect_sentinel_workload(campaign)
    return summary


def _campaign_catalog() -> dict[str, Any]:
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
                "source_name": campaign.source_name,
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
    return result


def _dispatch(args: argparse.Namespace) -> int:  # noqa: C901, PLR0911, PLR0912, PLR0915 -- thin CLI command dispatch
    """Dispatch one parsed command to its authoritative service."""
    if args.command == "resolve-generation-run":
        run_plan = run_service.resolve_generation_run(
            args.config,
            authored_value=args.authored_value_json,
            source_commit=args.git_commit,
            require_executable=not args.allow_incomplete,
            scientific_overrides=_scientific_overrides(args),
            execution_overrides=_execution_overrides(args),
        )
        print(json.dumps(run_plan.to_payload(), sort_keys=True))
        return 0
    if args.command == "list-campaigns":
        print(json.dumps(_campaign_catalog(), sort_keys=True))
        return 0
    if args.command == "inspect-core-benchmark":
        inspection = benchmark_config.inspect_core_benchmark(
            args.suite,
            require_executable=args.require_executable,
        )
        print(json.dumps(inspection, sort_keys=True))
        return 0
    if args.command == "resolve-core-benchmark-run":
        identity = benchmark_service.resolve_core_benchmark_runtime_identity(
            args.suite,
            git_commit=args.git_commit,
            comsol_version_output=args.comsol_version_output,
        )
        print(json.dumps(identity, sort_keys=True))
        return 0
    if args.command == "preflight-core-benchmark":
        preflight = benchmark_service.preflight_core_benchmark(
            args.suite,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            scratch_root=args.scratch_root,
            comsol_version_output=args.comsol_version_output,
            comsol_executable_path=args.comsol_executable_path,
        )
        print(json.dumps(preflight, sort_keys=True))
        return 0
    if args.command == "plan-core-benchmark":
        benchmark_plan = benchmark_service.plan_core_benchmark(
            args.suite,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            scratch_root=args.scratch_root,
            comsol_version_output=args.comsol_version_output,
            comsol_executable_path=args.comsol_executable_path,
        )
        print(json.dumps(benchmark_plan, sort_keys=True))
        return 0
    if args.command == "materialize-core-benchmark-inputs":
        manifest = benchmark_service.prepare_core_benchmark(
            args.suite,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            scratch_root=args.scratch_root,
            comsol_version_output=args.comsol_version_output,
            comsol_executable_path=args.comsol_executable_path,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "submit-core-benchmark":
        manifest = benchmark_service.submit_core_benchmark(
            args.suite,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            scratch_root=args.scratch_root,
            comsol_version_output=args.comsol_version_output,
            comsol_executable_path=args.comsol_executable_path,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "run-core-benchmark-case":
        result = benchmark_service.run_core_benchmark_case(
            args.benchmark_run_id,
            args.sweep_id,
            args.variant_id,
            args.variant_attempt,
            args.case_role,
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
        elif args.format == "summary":
            print(campaign_status_service.format_benchmark_status_summary(status))
        elif args.format == "monitor":
            print(campaign_status_service.format_benchmark_monitor(status, max_active_cases=8))
        else:
            print(json.dumps(status, sort_keys=True))
        return 0
    if args.command == "resume-core-benchmark":
        manifest = benchmark_service.resume_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
            scratch_root=args.scratch_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "cancel-core-benchmark":
        receipt = benchmark_service.cancel_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
            force=args.force,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "finalize-core-benchmark":
        summary = benchmark_service.finalize_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "core-benchmark-source-status":
        status = benchmark_service.core_benchmark_source_status(
            args.benchmark_run_id,
            storage_root=args.storage_root,
            query_scheduler=not args.no_scheduler,
        )
        if args.format == "json":
            print(json.dumps(status, sort_keys=True))
        else:
            print(
                "\t".join(
                    (
                        "source-status",
                        status["run_id"],
                        status["run_state"],
                        status["source_state"],
                        str(status["reclaimable_bytes"]),
                        status["cleanup_eligibility"],
                        str(status["active_slurm"]),
                    )
                )
            )
        return 0
    if args.command == "validate-core-benchmark":
        result = benchmark_service.validate_core_benchmark(
            args.benchmark_run_id,
            storage_root=args.storage_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "core-benchmark-summary":
        if args.format == "json":
            summary = benchmark_service.load_core_benchmark_summary(
                args.benchmark_run_id,
                storage_root=args.storage_root,
            )
            print(json.dumps(summary, sort_keys=True))
        else:
            markdown = benchmark_service.load_core_benchmark_markdown(
                args.benchmark_run_id,
                storage_root=args.storage_root,
            )
            print(markdown, end="")
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
        return 0 if report["production_configuration_ready"] else 2
    if args.command == "initialize-worker-workspace":
        marker = workspace_service.initialize_worker_workspace(
            args.directory,
            run_id=args.campaign_run_id,
            storage_root=args.storage_root,
        )
        print(marker)
        return 0
    if args.command == "cleanup-worker-workspace":
        workspace_service.cleanup_worker_workspace(
            args.directory,
            run_id=args.campaign_run_id,
            storage_root=args.storage_root,
            allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        return 0
    if args.command == "validate-config" and args.only_batch is None:
        campaign = config_service.load_campaign_config(
            args.config,
            authored_value=args.authored_value_json,
            require_executable=not args.allow_incomplete,
            scientific_overrides=_scientific_overrides(args),
            execution_overrides=_execution_overrides(args),
        )
        gates = readiness_service.campaign_unresolved_gates(args.config, authored_value=args.authored_value_json)
        print(json.dumps(_campaign_summary(campaign, unresolved_gates=gates), sort_keys=True))
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
            force=args.force,
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
    if args.command == "declare-candidate-failure":
        path = campaign_runtime.declare_candidate_failure(
            args.campaign_run_id,
            args.material,
            args.case_id,
            reason_code=args.reason_code,
            detail=args.detail,
            storage_root=args.storage_root,
        )
        print(path)
        return 0
    if args.command == "amend-replacement-pool":
        amendment = campaign_runtime.amend_replacement_capacity(
            args.campaign_run_id,
            args.material,
            args.regime,
            capacity=args.capacity,
            reason=args.reason,
            storage_root=args.storage_root,
        )
        print(json.dumps(amendment, sort_keys=True))
        return 0
    if args.command == "resume-campaign":
        manifest = campaign_runtime.resume_campaign(
            args.campaign_run_id,
            storage_root=args.storage_root,
            metadata_only=args.metadata_only,
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
    if args.command == "run-node-worker":
        campaign_directory = campaign_evidence.node_job_campaign_directory(args.node_directory)
        if campaign_directory is not None:
            templates = cluster_service.node_job_log_templates(common.paths.get_runtime_root().resolve())
            host_service.link_job_logs(campaign_directory, host_service.job_log_files(templates))
        node_result = node_service.run_node_worker(args.node_directory, cores=args.cores)
        print(f"Node worker ended: {node_result.detail} ({node_result.reason}); steps started {node_result.started}, finished {node_result.finished}")
        return node_result.exit_code
    if args.command == "run-campaign-case":
        outcome = campaign_runtime.run_campaign_case_job(
            args.campaign_run_id,
            args.batch_name,
            args.case_index,
            storage_root=args.storage_root,
            work_root=args.work_root,
        )
        print(_case_end_line(args, outcome.status))
        return 0
    if args.command in {"plan-campaign", "prepare-campaign-inputs", "submit-campaign"}:
        campaign = config_service.load_campaign_config(
            args.config,
            authored_value=args.authored_value_json,
            scientific_overrides=_scientific_overrides(args),
            execution_overrides=_execution_overrides(args),
        )
        campaign = campaign.select_batches(None if args.only_batch is None else (args.only_batch,))
        if args.command == "plan-campaign":
            campaign_plan = campaign_runtime.plan_campaign(
                campaign,
                git_commit=args.git_commit,
                storage_root=args.storage_root,
            )
            print(json.dumps(campaign_plan, sort_keys=True))
            return 0
        if args.command == "prepare-campaign-inputs":
            readiness = input_service.prepare_campaign_inputs(
                campaign,
                git_commit=args.git_commit,
                storage_root=args.storage_root,
                progress=_print_workflow_progress,
            )
            print(f"canonical-inputs\t{readiness['generated_case_count']}\t{readiness['reused_case_count']}")
            return 0
        manifest = campaign_runtime.submit_campaign(
            campaign,
            git_commit=args.git_commit,
            storage_root=args.storage_root,
            progress=_print_workflow_progress,
            inputs_prepared=args.inputs_prepared,
            max_admission_cases=args.max_admission_cases,
            metadata_only=args.metadata_only,
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
    message = f"Unsupported generation command: {args.command!r}."
    raise ValueError(message)


def _cli_error_payload(
    args: argparse.Namespace,
    error: Exception,
) -> dict[str, Any]:
    """Return one concise contextual error record for the CLI boundary."""
    context: dict[str, Any] = {}
    for name in (
        "campaign_run_id",
        "steady_campaign_run_id",
        "transient_campaign_run_id",
        "benchmark_run_id",
        "batch_name",
        "case_id",
        "case_index",
        "config",
        "suite",
        "storage_root",
    ):
        value = getattr(args, name, None)
        if value is not None:
            context[name] = str(value)
    for name in (
        "failure_stage",
        "work_directory",
        "cwd",
        "missing_or_invalid_artifacts",
    ):
        value = getattr(error, name, None)
        if value is not None:
            context[name] = [str(item) for item in value] if isinstance(value, (list, tuple)) else str(value)
    return {
        "status": "error",
        "command": args.command,
        "stage": str(getattr(error, "failure_stage", args.command)),
        "error_type": type(error).__name__,
        "message": str(error),
        "context": context,
    }


def main(argv: list[str] | None = None) -> int:
    """Run one generation command and translate failures to process status two."""
    args = _build_parser().parse_args(argv)
    previous_term: Any = None
    if args.command in {"run-campaign-case", "run-core-benchmark-case"}:
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
            classified = False
            if args.command == "run-campaign-case":
                # Nothing parses this step's output. The reason is stated in words, and a stop the runtime classified itself
                # (stall, timeout, solver error) is fully stated by that line and gets no JSON error event.
                reason = str(error).splitlines()[0] if str(error) else type(error).__name__
                print(f"{_case_label(args)} stopped: {reason}", file=sys.stderr)
                print(_case_end_line(args, "failed"))
                classified = getattr(error, "runtime_reason", None) is not None
            if not classified:
                print(
                    json.dumps(
                        _cli_error_payload(args, error),
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
        return 2
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    raise SystemExit(main())
