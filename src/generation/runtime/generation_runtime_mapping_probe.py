"""
===============================================================================
generation_runtime_mapping_probe.py
===============================================================================
Run one retained diagnostic COMSOL case before profile mappings are confirmed.
Responsibilities:
  - Prepare exactly one technical-smoke case in an isolated Slurm workspace
  - Retain actual output filenames, table headers, shapes, logs, and raw bytes
  - Compare observations with explicit profile mappings and list correction keys
Design principles:
  - Diagnostic inventory never becomes production auto-detection or an alias map
  - Profile YAML remains the only explicit mapping owner after manual review
  - Probe artifacts are immutable, commit-bound, and outside normal datasets
This module does NOT:
  - Guess missing mappings, edit profile YAML, publish HDF5, or launch production
  - Treat a diagnostic case as successful real-runtime smoke validation
===============================================================================
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_comsol_spreadsheet as spreadsheet_contract
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.contracts import generation_contracts_source as source_service

from . import generation_runtime_batch as runtime_service
from . import generation_runtime_comsol as comsol_service
from . import generation_runtime_workspace as workspace_service

MAPPING_PROBE_SCHEMA_KIND: Final = "generation_mapping_probe"
MAPPING_PROBE_SCHEMA_VERSION: Final = 3
_TECHNICAL_SMOKE_PURPOSE: Final = "technical_runtime_smoke"
_TEXT_SUFFIXES: Final = frozenset({".csv", ".dat", ".txt"})


def _profile_config_path(campaign_path: Path) -> Path:
    """Resolve the explicit profile configuration referenced by one campaign."""
    try:
        campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not read mapping-probe campaign: {campaign_path}"
        raise ValueError(message) from error
    if not isinstance(campaign, dict) or not isinstance(campaign.get("profile_config"), str):
        message = f"Mapping-probe campaign has no profile_config: {campaign_path}"
        raise TypeError(message)
    configured = Path(campaign["profile_config"])
    path = configured if configured.is_absolute() else common.paths.get_project_root() / configured
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        message = f"Mapping-probe profile configuration is missing or unsafe: {resolved}"
        raise FileNotFoundError(message)
    return resolved


def _profile_mapping(path: Path) -> dict[str, Any]:
    """Return one raw profile after validating every typed mapping state."""
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not parse profile mapping: {path}"
        raise ValueError(message) from error
    if not isinstance(profile, dict) or not isinstance(
        profile.get("exports"),
        list,
    ):
        message = f"Profile mapping must contain an exports list: {path}"
        raise TypeError(message)
    states = {"declared_unverified", "runtime_confirmed", "mapping_probe_required"}
    for index, export in enumerate(profile["exports"]):
        if not isinstance(export, dict) or not isinstance(export.get("source"), dict) or not isinstance(export.get("columns"), dict):
            message = f"Profile export {index} is malformed: {path}"
            raise TypeError(message)
        if export["source"].get("state") not in states:
            message = f"Profile export {index} source has no typed mapping state: {path}"
            raise TypeError(message)
        for logical, mapping in export["columns"].items():
            if not isinstance(mapping, dict) or mapping.get("state") not in states:
                message = f"Profile export {index} column {logical!r} has no typed mapping state: {path}"
                raise TypeError(message)
    return profile


def _snapshot(directory: Path) -> set[str]:
    """Return exact regular-file membership below a symlink-free workspace."""
    result: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            message = f"Mapping-probe workspace contains a symbolic link: {path}"
            raise ValueError(message)
        if path.is_file():
            result.add(path.relative_to(directory).as_posix())
    return result


def _table_observation(path: Path) -> dict[str, Any] | None:
    """Return structurally parsed COMSOL Spreadsheet evidence."""
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return None
    try:
        delimiter = spreadsheet_contract.detect_comsol_spreadsheet_delimiter(path)
        table = spreadsheet_contract.read_comsol_spreadsheet(path, delimiter=delimiter, include_values=False)
    except ValueError as error:
        return {"read_error": str(error)}
    header = list(table.canonical_header)
    return {
        "delimiter": delimiter,
        "raw_header": list(table.raw_header),
        "shape": list(table.shape),
        "rectangular": True,
        "comsol_metadata": dict(table.metadata),
        "time_header_candidates": [name for name in header if name.casefold() in {"t", "time", "time_h", "time_s"}],
    }


def _inventory(
    workspace: Path,
    relative_paths: set[str],
) -> list[dict[str, Any]]:
    """Return exact identity and table observations for produced files."""
    records: list[dict[str, Any]] = []
    for relative in sorted(relative_paths):
        path = workspace / relative
        record: dict[str, Any] = {
            "relative_path": relative,
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        table = _table_observation(path)
        if table is not None:
            record["table"] = table
        records.append(record)
    return records


def _mapping_comparison(
    profile: dict[str, Any],
    inventory: list[dict[str, Any]],
    *,
    profile_path: Path,
) -> dict[str, Any]:
    """Compare existing observed exports with the explicit typed contract."""
    prefix = profile_path.relative_to(common.paths.get_project_root().resolve()).as_posix()
    by_basename: dict[str, list[dict[str, Any]]] = {}
    for record in inventory:
        by_basename.setdefault(Path(record["relative_path"]).name, []).append(record)
    required_corrections: list[str] = []
    required_missing_exports: list[dict[str, str]] = []
    observations: list[dict[str, Any]] = []
    profile_spec = profiles.resolve_profile(str(profile["simulation_profile"]))
    for index, export in enumerate(profile["exports"]):
        role = str(export["role"])
        role_spec = profile_spec.export_role(role)
        units_by_logical = dict(zip(role_spec.logical_fields, role_spec.units, strict=True))
        corrections = required_corrections
        missing_exports = required_missing_exports
        source = export["source"]
        source_key = f"{prefix}:exports[{index}].source"
        pattern = source.get("pattern")
        matches = [] if not isinstance(pattern, str) else by_basename.get(pattern, [])
        if not isinstance(pattern, str):
            corrections.append(source_key)
        elif not matches:
            missing_exports.append({"role": role, "declared_pattern": pattern})
        elif len(matches) > 1 or source["state"] == "mapping_probe_required":
            corrections.append(source_key)

        table = matches[0].get("table") if len(matches) == 1 else None
        raw_header = table.get("raw_header", []) if isinstance(table, dict) else []
        expected_units = {
            str(mapping["source_header"]): units_by_logical[logical]
            for logical, mapping in export["columns"].items()
            if isinstance(mapping.get("source_header"), str)
        }
        temporal_groups: tuple[spreadsheet_contract.ComsolTemporalGroup, ...] = ()
        temporal_error: str | None = None
        if role == profiles.TRANSIENT_RAW_EXPORT_ROLE and raw_header:
            try:
                temporal_groups = spreadsheet_contract.group_temporal_columns(
                    raw_header,
                    expected_units=expected_units,
                )
            except ValueError as error:
                temporal_error = str(error)
                canonical_header = []
            else:
                canonical_header = [column.source for column in temporal_groups[0].columns]
        else:
            canonical_header = list(
                spreadsheet_contract.canonicalize_header(
                    raw_header,
                    expected_units=expected_units,
                )
            )
        observed_header = canonical_header
        delimiter_matches = bool(isinstance(table, dict) and table.get("delimiter") == export["delimiter"])
        if len(matches) == 1 and not delimiter_matches:
            corrections.append(f"{prefix}:exports[{index}].delimiter")
        column_results: dict[str, Any] = {}
        for logical, mapping in export["columns"].items():
            key = f"{prefix}:exports[{index}].columns.{logical}"
            header = mapping.get("source_header")
            matches_header = isinstance(header, str) and header in observed_header
            if len(matches) == 1 and (mapping["state"] == "mapping_probe_required" or not matches_header):
                corrections.append(key)
            column_results[logical] = {
                "state": mapping["state"],
                "declared_source_header": header,
                "observed_exact_header_match": matches_header,
            }
        observations.append(
            {
                "role": role,
                "source_state": source["state"],
                "declared_pattern": pattern,
                "matched_relative_paths": [record["relative_path"] for record in matches],
                "delimiter": table.get("delimiter") if isinstance(table, dict) else None,
                "raw_header": raw_header,
                "canonical_header": canonical_header,
                "observed_header": observed_header,
                "parsed_shape": table.get("shape") if isinstance(table, dict) else None,
                "comsol_metadata": table.get("comsol_metadata", {}) if isinstance(table, dict) else {},
                "temporal_state_times": [group.state_time for group in temporal_groups],
                "temporal_group_count": len(temporal_groups),
                "temporal_structure_error": temporal_error,
                "delimiter_matches": delimiter_matches,
                "columns": column_results,
            }
        )
    return {
        "required_corrections": sorted(set(required_corrections)),
        "required_missing_exports": required_missing_exports,
        "observations": observations,
        "aliases_used": False,
    }


def _mapping_probe_status(
    comparison: dict[str, Any],
    *,
    exit_code: int | None,
    timed_out: bool,
    start_error: str | None,
) -> str:
    """Classify process, export-execution, and mapping outcomes distinctly."""
    if start_error is not None or timed_out or exit_code != 0:
        return "comsol_execution_failed"
    if comparison["required_missing_exports"]:
        return "required_export_missing"
    if comparison["required_corrections"]:
        return "mapping_update_required"
    return "mapping_observation_complete"


def _copy_probe_artifacts(
    workspace: Path,
    relative_paths: set[str],
    *,
    staging: Path,
    input_paths: tuple[Path, ...],
) -> None:
    """Copy retained inputs and produced files into private publication staging."""
    inputs = staging / "inputs"
    inputs.mkdir(parents=True)
    for path in (*input_paths, workspace / "case.json"):
        shutil.copy2(path, inputs / path.name)
    outputs = staging / "produced_files"
    for relative in sorted(relative_paths):
        source = workspace / relative
        target = outputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _probe_id(profile_id: str, git_commit: str, slurm_job_id: str) -> str:
    """Return one immutable human-readable mapping-probe identifier."""
    return common.paths.validate_logical_name(
        f"{profile_id}__{git_commit[:12]}__slurm_{slurm_job_id}",
        label="mapping_probe_id",
    )


def run_mapping_probe(
    campaign_path: Path | str,
    *,
    storage_root: Path | str,
    work_root: Path | str,
    cores_per_case: int,
    only_batch: str | None = None,
) -> Path:
    """Run and retain one isolated diagnostic case without mapping inference."""
    source_path = Path(campaign_path).expanduser().resolve()
    campaign = config_service.load_campaign_config(source_path, require_executable=False)
    if campaign.campaign_purpose != _TECHNICAL_SMOKE_PURPOSE:
        message = "Mapping probes may run only against technical-runtime-smoke campaigns."
        raise ValueError(message)
    selected = campaign if only_batch is None else campaign.select_batches((only_batch,))
    if len(selected.batches) != 1 or len(selected.batches[0].case_indices) < 1:
        message = "Mapping probe requires exactly one selected non-empty technical batch."
        raise ValueError(message)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit():
        message = "Mapping probe must run inside one native Slurm allocation."
        raise RuntimeError(message)
    git_commit = source_service.required_git_commit()
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    profile_path = _profile_config_path(source_path)
    raw_profile = _profile_mapping(profile_path)
    config = selected.batches[0]
    case_index = config.case_indices[0]
    prepared = runtime_service.prepare_case_work_directory(
        config,
        case_index,
        storage_root=storage,
        work_root=work_root,
    )
    scalar_handoff = prepared.bundle.scalar_handoff
    if scalar_handoff is not None:
        scalar_handoff_contract.validate_transient_scalar_source(scalar_handoff)
    before = _snapshot(prepared.work_directory)
    command = comsol_service.build_comsol_command(
        config,
        cores_per_case=cores_per_case,
        scalar_handoff=scalar_handoff,
        scheduler_kind="slurm",
    )
    stdout_path = prepared.runtime_directory / "mapping_probe.stdout.log"
    stderr_path = prepared.runtime_directory / "mapping_probe.stderr.log"
    started = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    start_error: str | None = None
    with (
        stdout_path.open("w", encoding="utf-8", newline="\n") as stdout,
        stderr_path.open("w", encoding="utf-8", newline="\n") as stderr,
    ):
        try:
            result = subprocess.run(  # noqa: S603 -- validated fixed argument vector without a shell
                command,
                cwd=prepared.work_directory,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                text=True,
                timeout=float(config.execution_values["runtime"]["timeout_seconds"]),
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as error:
            start_error = str(error)
    runtime_s = time.monotonic() - started
    after = _snapshot(prepared.work_directory)
    produced = after.difference(before)
    inventory = _inventory(prepared.work_directory, produced)
    mapping_comparison = _mapping_comparison(
        raw_profile,
        inventory,
        profile_path=profile_path,
    )
    status = _mapping_probe_status(
        mapping_comparison,
        exit_code=exit_code,
        timed_out=timed_out,
        start_error=start_error,
    )
    probe_id = _probe_id(config.profile.id, git_commit, slurm_job_id)
    root = common.paths.get_generation_meta_root(storage_root=storage) / "mapping_probes"
    destination = root / probe_id
    if destination.exists():
        message = f"Mapping-probe identity already exists and is immutable: {destination}"
        raise FileExistsError(message)
    state = root / ".state"
    state.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{probe_id}.", dir=state)).resolve()
    try:
        _copy_probe_artifacts(
            prepared.work_directory,
            produced,
            staging=staging,
            input_paths=prepared.bundle.input_paths,
        )
        report = {
            "schema_kind": MAPPING_PROBE_SCHEMA_KIND,
            "schema_version": MAPPING_PROBE_SCHEMA_VERSION,
            "status": status,
            "probe_id": probe_id,
            "campaign_id": campaign.campaign_id,
            "campaign_config": source_path.relative_to(common.paths.get_project_root().resolve()).as_posix(),
            "simulation_profile": config.profile.id,
            "case_id": prepared.bundle.case_id,
            "case_input_id": prepared.bundle.case_input_id,
            "git_commit": git_commit,
            "slurm_job_id": slurm_job_id,
            "hostname": socket.gethostname(),
            "template": {
                "relative_path": config.profile.template_relative_path,
                "sha256": config.template_sha256,
            },
            "profile_config": {
                "relative_path": profile_path.relative_to(common.paths.get_project_root().resolve()).as_posix(),
                "sha256": common.serialization.file_sha256(profile_path),
            },
            "expected_mapping": {
                "profile_yaml": raw_profile,
                "resolved_output_contract": config.scientific_values["output_contract"],
            },
            "fields_requiring_correction": mapping_comparison["required_corrections"],
            "required_exports_missing": mapping_comparison["required_missing_exports"],
            "mapping_comparison": mapping_comparison,
            "actual_file_inventory": inventory,
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "start_error": start_error,
            "runtime_s": runtime_s,
            "raw_artifacts_relative_path": destination.relative_to(storage).as_posix(),
            "mapping_auto_detection_used": False,
            "production_solve_started": False,
            "technical_case_started": exit_code is not None or timed_out,
        }
        common.serialization.atomic_write_json(staging / "mapping_probe.json", report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        workspace_service.cleanup_case_workspace(
            prepared.work_directory,
            allowed_root=prepared.work_root,
            storage_root=storage,
            expected_run_id=prepared.workspace_run_id,
            expected_case_id=prepared.bundle.case_id,
            allow_active_job_id=slurm_job_id,
        )
    return destination / "mapping_probe.json"


def load_mapping_probe(path: Path | str) -> dict[str, Any]:
    """Load one immutable mapping-probe report and validate retained identities."""
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file() or report_path.is_symlink():
        message = f"Mapping-probe report is missing or unsafe: {report_path}"
        raise FileNotFoundError(message)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Mapping-probe report is unreadable: {report_path}"
        raise ValueError(message) from error
    if (
        not isinstance(report, dict)
        or report.get("schema_kind") != MAPPING_PROBE_SCHEMA_KIND
        or report.get("schema_version") != MAPPING_PROBE_SCHEMA_VERSION
        or report.get("mapping_auto_detection_used") is not False
        or report.get("production_solve_started") is not False
    ):
        message = f"Mapping-probe report schema is invalid: {report_path}"
        raise ValueError(message)
    artifact_root = report_path.parent / "produced_files"
    records = report.get("actual_file_inventory")
    if not isinstance(records, list):
        message = "Mapping-probe actual_file_inventory must be a list."
        raise TypeError(message)
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            message = "Mapping-probe file record is malformed."
            raise TypeError(message)
        path_value = artifact_root / record["relative_path"]
        if (
            not path_value.is_file()
            or path_value.is_symlink()
            or path_value.stat().st_size != record.get("size_bytes")
            or common.serialization.file_sha256(path_value) != record.get("sha256")
        ):
            message = f"Mapping-probe retained output identity failed: {path_value}"
            raise RuntimeError(message)
    return report
