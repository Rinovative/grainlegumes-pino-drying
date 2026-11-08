"""Expose the canonical Dataset package builder identity and supported package CLI.

Responsibilities:
  - Preserve the persisted package-builder callable identity
  - Delegate package build, admission, inspection, and reconstruction operations
  - Parse the supported run (Dataset plan), inspect, and rebuild commands
  - Submit a Dataset plan run from the live checkout as its own driver job, and run it as the driver inside the source tree
  - Summarize a finished driver run in words and report an expected failure as one plain line instead of a traceback

Design principles:
  - The facade remains thin while its qualified public identity stays stable
  - Package construction and runtime validation each have one responsible owner
  - Inspection presents readable evidence; full deterministic JSON remains available
  - The driver summary reads the validated Dataset manifest, so it states the published evidence rather than the producer's counters

This module does NOT:
  - Implement package publication or Dataset loading
  - Duplicate manifest, identity, or runtime validation logic
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

_DATASETS_ANCESTOR = 2  # manifest.parents[2] is the 02_datasets area: <dataset>, <task>, 02_datasets
_LOGIN_INTERPRETER = "../runtime/venvs/native/bin/python"  # project layout: the command runs from the repo/ checkout
_SCHEMA_EXPORTS = frozenset(
    {
        "DATASET_PACKAGE_SCHEMA_KIND",
        "DATASET_PACKAGE_SCHEMA_VERSION",
    }
)


def __getattr__(name: str) -> object:
    """Resolve one manifest schema constant without introducing a facade cycle."""
    if name not in _SCHEMA_EXPORTS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    from .packages import dataset_packages_manifest as package_manifest  # noqa: PLC0415

    value = getattr(package_manifest, name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    DATASET_PACKAGE_SCHEMA_KIND: str
    DATASET_PACKAGE_SCHEMA_VERSION: int


def build_dataset_selection(
    request: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Publish one exact ordered Dataset selection from admitted source cases."""
    from .packages import dataset_packages_builder as builder  # noqa: PLC0415

    return builder.build_dataset_selection(request, storage_root=storage_root)


def inspect_dataset_package(
    task: str,
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect one package through its validated manifest and runtime object."""
    from .packages import dataset_packages_reproduction as reproduction  # noqa: PLC0415

    return reproduction.inspect_dataset_reproduction(task, dataset_id, storage_root=storage_root)


def _source_outcome_lines(outcomes: list[Mapping[str, Any]]) -> list[str]:
    """Show the original-candidate failure rate, replacement use and achieved success rate of every contributing Source."""
    result: list[str] = []
    for outcome in outcomes:
        result.append(f"Source outcomes of {outcome['source_name']} (full candidate table in --json):")
        for material, counts in outcome["materials"].items():
            policy = [item for item in outcome["replacement_policy"] if item["material_family"] == material]
            achieved = ", ".join(
                f"{item['successful']}/{item['decided']} decided successful against minimum {item['minimum_success_rate']:g}"
                f"{' (stopped)' if item['stopped'] else ''}"
                for item in policy
            )
            original, failed = counts["original_candidates"], counts["original_failed"]
            rate = f"{failed / original:.1%}" if original else "n/a"
            result.append(
                f"  {material}: original candidates failed {failed}/{original} ({rate}); "
                f"replacements used {counts['replacement_candidates']} (failed {counts['replacement_failed']}); "
                f"successful {counts['successful']}; {achieved}"
            )
    return result


def format_dataset_inspection(report: Mapping[str, Any]) -> str:
    """Present existing inspection evidence and exact reconstruction steps for a researcher."""

    def availability(value: object) -> str:
        return "available" if value is True else "unavailable" if value is False else "unknown"

    def evidence(value: object) -> str:
        return "unavailable in inspection evidence" if value is None else str(value)

    def channels(values: object) -> str:
        if not isinstance(values, list):
            return evidence(values)
        return ", ".join(f"{item['name']} ({item['unit']})" for item in values)

    def parameter_summary(values: object) -> str:
        if not isinstance(values, Mapping):
            return evidence(values)
        categories = ", ".join(f"{name}: {_category_count(value)}" for name, value in values.items())
        return f"{len(values)} parameter categories ({categories or 'none recorded'}); full values are in --json"

    def _category_count(value: object) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, (Mapping, list, tuple)):
            return str(len(value))
        return "recorded"

    def named_counts(values: object) -> str:
        if not isinstance(values, Mapping):
            return evidence(values)
        return ", ".join(f"{name}={count}" for name, count in values.items()) or "none recorded"

    producer = report["producer"]
    recovery = producer["source_recovery"]
    missing_inventory = recovery.get("missing_or_changed_untracked")
    missing_files = ", ".join(missing_inventory) or "none listed" if isinstance(missing_inventory, list) else "unknown (inventory not checked)"
    selection = report.get("selection") or {}
    selection_summary = f"{selection.get('kind', 'unknown')}; counts {named_counts(selection.get('target_counts'))}"
    manifest_path = Path(str(report["manifest"]))
    ancestors = manifest_path.parents  # <storage>/02_datasets/<task>/<dataset>/dataset_manifest.json
    in_datasets_area = (
        manifest_path.name == "dataset_manifest.json" and len(ancestors) > _DATASETS_ANCESTOR and ancestors[_DATASETS_ANCESTOR].name == "02_datasets"
    )
    storage_root = ancestors[_DATASETS_ANCESTOR].parent if in_datasets_area else None
    json_command = (
        f"python -m src.datasets.dataset_packages inspect {shlex.quote(str(report['dataset_view']))} {shlex.quote(str(report['dataset_id']))}"
    )
    if storage_root is not None:
        json_command += f" --storage-root {shlex.quote(str(storage_root))}"
    json_command += " --json"
    channel_contract = report["channel_contract"]
    if "input" in channel_contract or "inputs" in channel_contract:
        input_fields = channel_contract.get("inputs", channel_contract.get("input"))
        output_fields = channel_contract.get("outputs", channel_contract.get("target"))
        channel_lines = [f"Inputs: {channels(input_fields)}", f"Outputs: {channels(output_fields)}"]
        if "tensor_layout" in channel_contract:
            channel_lines.append(f"Tensor layout: {', '.join(channel_contract['tensor_layout'])}")
    else:
        channel_lines = [
            f"State: {channels(channel_contract['state'])}",
            f"Static: {channels(channel_contract['static'])}",
            f"Boundary: {channels(channel_contract['boundary'])}",
            f"Scalars: {channels(channel_contract['scalars'])}",
            f"Targets: {channels(channel_contract['target'])}",
            (
                f"Time: {channels(channel_contract['time']['fields'])}; "
                f"storage {channel_contract['storage']}; dtype {channel_contract['tensor_dtype']}"
            ),
        ]
    lines = [
        f"Dataset: {report['dataset_id']} ({report['dataset_view']})",
        (
            f"Contents: {report['source_case_count']} Source cases; "
            f"{report.get('sample_count', 'unknown')} samples; {report['transition_count']} transitions"
        ),
        f"Selection: {selection_summary}",
        f"Grid shape: {report['grid']['shape']}; axes SHA-256 {report['grid']['axes_sha256']}",
        *channel_lines,
        "",
        "Sources, in Dataset order:",
    ]
    fixed = report.get("case_partition")
    if fixed is None:
        lines.insert(lines.index("Sources, in Dataset order:"), "Fixed roles: unavailable; this low-level selection is not a training Dataset plan.")
    else:
        position = lines.index("Sources, in Dataset order:")
        lines[position:position] = [
            f"Fixed split: seed input {fixed['seed_input']}; effective seed {fixed['seed']}; "
            + ", ".join(f"{role}={len(fixed[role])}" for role in ("train", "validation", "id_test", "ood")),
            f"Material roles: {', '.join(f'{material}={role}' for material, role in fixed['material_roles'].items())}",
        ]
    position = lines.index("Sources, in Dataset order:")
    lines[position:position] = _source_outcome_lines(report["source_outcomes"])
    invocation = report.get("plan_invocation")
    if invocation:
        # The recorded vector is the driver's; the plan runs from the login node, which freezes the source and submits a new driver.
        command = shlex.join([_LOGIN_INTERPRETER, *invocation[1:]])
        lines.insert(
            lines.index("Sources, in Dataset order:"),
            f"Re-run Dataset plan (login node, from repo/; it submits its own driver job): {command}",
        )
    for number, case in enumerate(report["cases"], 1):
        source = case["source"]
        solver = case.get("solver_environment") or {}
        solver_fields = (
            ("python_version", "Python"),
            ("loaded_modules", "modules"),
            ("executable_path", "executable"),
            ("hostname", "host"),
            ("allocated_node", "allocated node"),
            ("scheduler_job_id", "scheduler job"),
            ("requested_cores", "requested cores"),
        )
        solver_summary = (
            "; ".join(f"{label} {solver[key]}" for key, label in solver_fields if solver.get(key) is not None) or "unavailable in inspection evidence"
        )
        lines.extend(
            [
                f"  {number}. {case['package_case_id']} | {source['material_family']} | {source['source_name']}",
                (
                    f"     Generation root/candidate seeds: {evidence(case['seeds'].get('root_seed'))} / "
                    f"{evidence(case['seeds'].get('candidate_seed'))}; "
                    f"position {evidence(case['seeds'].get('candidate_position'))}"
                ),
                f"     Parameters: {parameter_summary(case['effective_parameters'])}",
                (
                    f"     HDF5: {availability(case['hdf5'].get('available'))}; "
                    f"{case['hdf5'].get('path', 'path unavailable in inspection evidence')}; "
                    f"hash verified {evidence(case['hdf5'].get('hash_verified_in_this_inspection'))}"
                ),
                f"     Solver environment: {solver_summary}",
            ]
        )
    for name, campaign in report["campaigns"].items():
        model = campaign["template"]
        code_snapshot = campaign.get("generation_source_snapshot") or {}
        invocation = campaign.get("generation_invocation") or {}
        recorded_command = invocation.get("recorded_execution_command")
        lines.extend(
            [
                "",
                f"Source configuration: {name}; plan {availability(campaign.get('plan_available'))}",
                f"  Generation values: {parameter_summary(campaign.get('effective_generation_values'))}",
                f"  COMSOL model: {model.get('current_path', 'unavailable in inspection evidence')}",
                (
                    f"  Model SHA-256 {model.get('sha256', 'unavailable in inspection evidence')}; "
                    f"bytes {availability(model.get('current_bytes_available'))}; "
                    f"recorded-hash match {evidence(model.get('current_bytes_match_recorded_hash'))}"
                ),
                (
                    "  Generation code snapshot: "
                    f"{code_snapshot.get('path', 'unavailable in inspection evidence')}; "
                    f"SHA-256 {code_snapshot.get('sha256', 'unavailable in inspection evidence')}"
                ),
                (f"  Recorded Generation command: {recorded_command if recorded_command is not None else 'not recorded'}"),
                f"  Scientific Generation command: {invocation.get('scientific_command', 'unavailable in inspection evidence')}",
                f"  Generation execution boundary: {invocation.get('execution_boundary', 'unavailable in inspection evidence')}",
                (f"  Execution choices not recorded: {', '.join(invocation.get('execution_choices_not_recorded', [])) or 'none listed'}"),
            ]
        )
    packages = producer["environment"].get("packages", {})
    package_versions = ", ".join(f"{name} {version}" for name, version in packages.items()) or "unknown"
    fingerprint = report["relevant_compatibility"].get("recorded_implementation_sha256", "unavailable in inspection evidence")
    lines.extend(
        [
            "",
            (f"Producing code snapshot: {producer['source_snapshot']['relative_path']}; SHA-256 {producer['source_snapshot']['sha256']}"),
            (
                f"  Snapshot {availability(recovery.get('available'))}; "
                f"manifest verified {recovery.get('manifest_verified', 'not checked')}; "
                f"tracked diff verified {recovery.get('tracked_diff_verified', 'not checked')}"
            ),
            (
                f"  Base Git commit {availability(recovery.get('base_commit_available'))}; "
                f"full modes recoverable {recovery.get('full_modes_recoverable', 'unknown')}"
            ),
            f"  Missing or changed retained files: {missing_files}",
            (
                f"Producing environment: Python {producer['environment'].get('python', 'unknown')}; "
                f"platform {producer['environment'].get('platform', 'unknown')}; "
                f"packages {package_versions}"
            ),
            (f"Implementation fingerprint: {fingerprint}"),
            (
                "Current interpreter admission: "
                f"{report['relevant_compatibility'].get('current_interpreter_admission', 'unavailable in inspection evidence')}"
            ),
            "  Matching fingerprints are admission evidence; they do not prove scientific equivalence.",
            "",
            "Recorded recovery and rebuild commands (verbatim):",
            f"  Restore producing source: {recovery.get('restore_command', 'unavailable in inspection evidence')}",
            f"  {report['dependency_preparation']}",
            f"  {report['regeneration']['path_mapping']}",
            f"  Restore for rebuild: {report['regeneration']['restore_command']}",
            f"  Rebuild Dataset: {report['regeneration']['dataset_rebuild_command']}",
            "",
            "Limits: inspection does not rehash the Dataset payload or HDF5 Sources.",
            "  Fresh verification output depends on the original Source root; it is not a portable Source archive.",
            f"  {report['regeneration']['environment_boundary']}",
            f"  Source re-simulation: {report['regeneration']['source_regeneration']}",
            f"Complete evidence: {json_command}",
        ]
    )
    return "\n".join(lines)


def format_run_summary(result: Mapping[str, Any], *, seconds: float) -> str:
    """Present the outcome of one Dataset plan run in the driver: the published Dataset, its split, the failures and the time taken."""
    from src import common  # noqa: PLC0415

    from .packages import dataset_packages_manifest as manifests  # noqa: PLC0415

    duration = common.console.format_duration(seconds)
    if result["status"] != "OK":
        return "\n".join(
            (
                f"DATASET {result['status']}: Source {result['source_run']} is not complete, no Dataset published",
                f"Next: {result['next']}",
                f"Duration: {duration}",
            )
        )
    published = result["dataset"]
    storage = Path(published["dependency_storage_root"])
    manifest = manifests.load_package_manifest_evidence(published["dataset_view"], published["dataset_id"], storage_root=storage)
    fixed = manifest["case_partition"]
    split = (
        "unavailable"
        if fixed is None
        else "  ".join(f"{role} {len(fixed[role])}" for role in ("train", "validation", "id_test", "ood"))
        + f"  ({', '.join(f'{material}={role}' for material, role in fixed['material_roles'].items())})"
    )
    candidates = [candidate for outcome in manifest["source_outcomes"] for candidate in outcome["candidates"]]
    # Every replacement names its slot's failed primary; as on the live board, a failed case's successor is the first later
    # replacement of the same slot, so a slot whose first replacement failed too reads as a chain.
    slots: dict[tuple[str, str], list[str]] = {}
    for item in candidates:
        if item["origin"] == "replacement":
            slots.setdefault((item["material_family"], item["parent_failed_case_id"]), []).append(item["case_id"])

    def successor(item: Mapping[str, Any]) -> str | None:
        slot = slots.get((item["material_family"], item["parent_failed_case_id"] or item["case_id"]), [])
        return min((case_id for case_id in slot if case_id > item["case_id"]), default=None)

    failures = [
        f"  {item['material_family']} {item['case_id']}: {item['failure_stage']} failure ({item['reason_code']}), "
        + (f"replaced by {replacement}" if (replacement := successor(item)) else "no replacement")
        for item in candidates
        if item["outcome"] == "failed"
    ]
    lines = [
        f"DATASET COMPLETE: {published['dataset_id']} ({published['dataset_view']})",
        f"Published: {Path(published['manifest_path']).parent.relative_to(storage)}",
        f"Contents: {published['source_case_count']} Source cases; {published['sample_count']} samples; {published['transition_count']} transitions",
        f"Split: {split}",
        "Failures:" + ("" if failures else " none"),
        *failures,
        f"Duration: {duration}",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the Dataset package build, inspection, and reconstruction CLI."""
    parser = argparse.ArgumentParser(description="Build, inspect, or reconstruct immutable dataset packages")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run/reuse one normal Dataset plan through Generation, fixed split and PT")
    run.add_argument("plan", type=Path)
    run.add_argument("--preview", action="store_true", help="read-only effective plan; no Source admission or scientific writes")
    run.add_argument(
        "--revision",
        metavar="LABEL",
        help="label appended as __rev-LABEL to the Dataset name for a deliberately separate Dataset; the Source is reused",
    )
    run.add_argument(
        "--new-source",
        action="store_true",
        help="with --revision: also simulate the Source afresh as <default Source name>__rev-LABEL instead of reusing the default Source",
    )
    run.add_argument("--count", action="append", metavar="MATERIAL=COUNT")
    run.add_argument("--split-fractions", metavar="TRAIN,VALIDATION,TEST", help="explicit recorded temporary split; fractions such as 1/3 accepted")
    run.add_argument("--duration-hours", type=float)
    run.add_argument(
        "--continue-with-current-code",
        action="store_true",
        help="continue an existing campaign with the current code instead of its bound source tree",
    )
    for flag in ("node-jobs", "max-admission-cases", "cores-per-case", "case-timeout-seconds"):
        run.add_argument(f"--{flag}", type=int)
    run.add_argument("--driver-partition", default="standard", help="Slurm partition of the submitted driver job: standard or long")
    run.add_argument("--driver-time", default="06:00:00", help="time limit of the submitted driver job as [D-]HH:MM:SS")
    run.add_argument("--no-retry", action="store_true", default=None)

    inspect = commands.add_parser("inspect", help="inspect one immutable package")
    inspect.add_argument("task", help="Dataset task directory below 02_datasets (steady_flow or transient_drying)")
    inspect.add_argument("dataset_id")
    inspect.add_argument("--storage-root", type=Path)
    inspect.add_argument("--json", action="store_true", help="emit the complete inspection evidence as JSON")

    rebuild = commands.add_parser("rebuild", help="rebuild an exact selected Dataset into empty independent storage")
    rebuild.add_argument("manifest", type=Path)
    rebuild.add_argument("--storage-root", type=Path, required=True, help="retained original Source and snapshot dependencies")
    rebuild.add_argument("--output-directory", type=Path, required=True, help="new verification output outside source and dependency storage")
    return parser


def _run_plan(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    """Preview, submit, or run one Dataset plan; an expected failure ends as one plain report instead of a traceback."""
    from src.generation import controller as controller_service  # noqa: PLC0415

    from . import dataset_plan  # noqa: PLC0415

    if arguments.new_source and arguments.revision is None:
        parser.error("--new-source requires --revision LABEL: the revision names the new Source")
    try:
        plan = dataset_plan.load_dataset_plan(
            arguments.plan,
            counts=arguments.count,
            split_fractions=arguments.split_fractions,
            revision=arguments.revision,
            new_source=arguments.new_source,
            duration_hours=arguments.duration_hours,
        )
    except ValueError as error:
        print(f"FAILED: Dataset plan: {error}", file=sys.stderr)
        return 1
    execution = {
        key: getattr(arguments, key)
        for key in ("node_jobs", "max_admission_cases", "cores_per_case", "case_timeout_seconds", "no_retry")
        if getattr(arguments, key) is not None
    }
    try:
        if arguments.preview:
            # First stdout line: the Source run identity and the Slurm footprint of a run; the effective plan follows.
            footprint = dataset_plan.preview_scheduler_footprint(
                plan, execution_overrides=execution, driver_time=arguments.driver_time, driver_partition=arguments.driver_partition
            )
            print(json.dumps(footprint, sort_keys=True))
            result = dataset_plan.preview_dataset_plan(plan, execution_overrides=execution)
        else:
            invocation = [sys.executable, "-m", "src.datasets.dataset_packages", *sys.argv[1:]]
            from src import common  # noqa: PLC0415

            if common.paths.source_tree_id() is None:
                # Started in the live checkout on the login node: freeze the source and submit the driver to run from the campaign's tree.
                result = dataset_plan.submit_dataset_plan(
                    arguments.plan,
                    plan,
                    invocation=invocation,
                    driver_partition=arguments.driver_partition,
                    driver_time=arguments.driver_time,
                    execution_overrides=execution,
                    continue_with_current_code=arguments.continue_with_current_code,
                )
            else:
                started = time.monotonic()
                result = dataset_plan.run_dataset_plan(
                    arguments.plan,
                    plan,
                    invocation=invocation,
                    execution_overrides=execution,
                )
                print(format_run_summary(result, seconds=time.monotonic() - started))
                return 0
    except controller_service.GenerationControllerError as error:
        print(error.report(), file=sys.stderr)
        return 1
    print(json.dumps(result, default=str, sort_keys=True, indent=2))
    return 0


def main() -> int:
    """Execute one Dataset plan run, inspection, or reconstruction command."""
    parser = _build_parser()
    arguments = parser.parse_args()
    if arguments.command == "run":
        return _run_plan(parser, arguments)
    if arguments.command == "inspect":
        result = inspect_dataset_package(
            arguments.task,
            arguments.dataset_id,
            storage_root=arguments.storage_root,
        )
    elif arguments.command == "rebuild":
        from .packages import dataset_packages_reproduction as reproduction  # noqa: PLC0415

        result = reproduction.rebuild_dataset(arguments.manifest, storage_root=arguments.storage_root, output_directory=arguments.output_directory)
    else:
        message = f"Unsupported dataset package command: {arguments.command!r}."
        raise RuntimeError(message)
    if arguments.command == "inspect" and not arguments.json:
        print(format_dataset_inspection(result))
    else:
        print(json.dumps(result, default=str, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
