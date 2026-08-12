"""
===============================================================================
generation_runtime_mapping_probe.py
===============================================================================
Run retained COMSOL mapping probes and validate their durable runtime evidence.
Responsibilities:
  - Prepare exactly one technical-smoke case in an isolated Slurm workspace
  - Retain actual output filenames, table headers, shapes, logs, and raw bytes
  - Compare observations with explicit profile mappings and list correction keys
  - Discover immutable reports matching the current semantic runtime identity
Design principles:
  - Diagnostic inventory never becomes production auto-detection or an alias map
  - Profile YAML owns expectations while immutable probe reports own verification
  - Evidence validity follows semantic mappings, template, COMSOL, and report version
This module does NOT:
  - Guess missing mappings, edit profile YAML, publish HDF5, or launch production
  - Treat a diagnostic case as successful real-runtime smoke validation
  - Use Git commit as an evidence-validity gate
===============================================================================
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.contracts import generation_contracts_comsol_spreadsheet as spreadsheet_contract
from src.generation.contracts import generation_contracts_mapping as mapping_contract
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.contracts import generation_contracts_source as source_service

from . import generation_runtime_batch as runtime_service
from . import generation_runtime_comsol as comsol_service
from . import generation_runtime_workspace as workspace_service

MAPPING_PROBE_SCHEMA_KIND: Final = "generation_mapping_probe"
MAPPING_PROBE_SCHEMA_VERSION: Final = 4
_TECHNICAL_SMOKE_PURPOSE: Final = "technical_runtime_smoke"
_TEXT_SUFFIXES: Final = frozenset({".csv", ".dat", ".txt"})
_COMSOL_EXACT_VERSION_PATTERN: Final = re.compile(r"(?<![0-9.])([0-9]+(?:[.][0-9]+){2,3})(?![0-9.])")


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
    """Return one raw state-free profile for complete or discovery-mode comparison."""
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        message = f"Could not parse profile mapping: {path}"
        raise ValueError(message) from error
    if (
        not isinstance(profile, dict)
        or profile.get("schema_kind") != "generation_profile"
        or profile.get("schema_version") != config_service.PROFILE_SCHEMA_VERSION
        or not isinstance(profile.get("simulation_profile"), str)
        or not isinstance(profile.get("exports"), list)
    ):
        message = f"Profile mapping must use generation_profile schema_version 2: {path}"
        raise TypeError(message)
    for index, export in enumerate(profile["exports"]):
        if not isinstance(export, dict) or not isinstance(export.get("columns"), dict):
            message = f"Profile export {index} is malformed: {path}"
            raise TypeError(message)
        source = export.get("source")
        if source is not None and not isinstance(source, str):
            message = f"Profile export {index} source must be scalar text or null: {path}"
            raise TypeError(message)
        for logical, source_header in export["columns"].items():
            if source_header is not None and not isinstance(source_header, str):
                message = f"Profile export {index} column {logical!r} must be scalar text or null: {path}"
                raise TypeError(message)
    return profile


def parse_comsol_exact_version(version_output: str) -> str:
    """Return the exact COMSOL runtime version from bounded command output."""
    if not isinstance(version_output, str) or not version_output.strip():
        message = "COMSOL version output must be non-empty text."
        raise ValueError(message)
    versions = tuple(dict.fromkeys(_COMSOL_EXACT_VERSION_PATTERN.findall(version_output)))
    if len(versions) != 1:
        message = f"COMSOL version output must contain exactly one full runtime version; found {list(versions)}."
        raise ValueError(message)
    return versions[0]


def _query_comsol_version(executable: str) -> tuple[str, str]:
    """Return bounded version output and its exact semantic version."""
    result = subprocess.run(  # noqa: S603 -- executable is validated configuration
        [executable, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    output = (result.stdout + result.stderr).strip()[:4096]
    if result.returncode != 0:
        message = f"COMSOL version query failed with exit code {result.returncode}: {output!r}."
        raise RuntimeError(message)
    return output, parse_comsol_exact_version(output)


def build_mapping_evidence_context(
    campaign_path: Path | str,
    *,
    comsol_version_output: str,
) -> dict[str, Any]:
    """
    Resolve the exact semantic and runtime identity required from probe evidence.

    Parameters
    ----------
    campaign_path : Path | str
        Generation campaign whose complete profile mapping is being checked.
    comsol_version_output : str
        Output of the active runtime executable's ``-version`` command.

    Returns
    -------
    dict[str, Any]
        Exact profile, mapping, template, COMSOL, and verifier identity.

    """
    campaign = config_service.load_campaign_config(campaign_path, require_executable=True)
    if not campaign.batches:
        message = "Mapping evidence requires at least one resolved campaign batch."
        raise ValueError(message)
    identities = {
        (
            batch.profile.id,
            mapping_contract.mapping_contract_sha256(
                batch.profile.id,
                batch.scientific_values["output_contract"],
            ),
            batch.template_sha256,
        )
        for batch in campaign.batches
    }
    if len(identities) != 1:
        message = "One campaign must resolve exactly one mapping contract and template identity."
        raise RuntimeError(message)
    simulation_profile, contract_sha256, template_sha256 = next(iter(identities))
    return {
        "simulation_profile": simulation_profile,
        "mapping_contract_sha256": contract_sha256,
        "template_sha256": template_sha256,
        "comsol_version": parse_comsol_exact_version(comsol_version_output),
        "verifier_schema_kind": MAPPING_PROBE_SCHEMA_KIND,
        "verifier_schema_version": MAPPING_PROBE_SCHEMA_VERSION,
    }


def evaluate_mapping_probe_report(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one report against the exact current evidence identity."""
    if report.get("schema_kind") != MAPPING_PROBE_SCHEMA_KIND:
        return {"valid": False, "classification": "malformed", "reasons": ["schema kind is invalid"]}
    stale_reasons: list[str] = []
    if report.get("schema_version") != expected.get("verifier_schema_version"):
        stale_reasons.append("verifier version differs")
    if report.get("simulation_profile") != expected.get("simulation_profile"):
        stale_reasons.append("simulation profile differs")
    if report.get("mapping_contract_sha256") != expected.get("mapping_contract_sha256"):
        stale_reasons.append("output mapping contract changed")
    template = report.get("template")
    if not isinstance(template, Mapping) or template.get("sha256") != expected.get("template_sha256"):
        stale_reasons.append("template SHA-256 changed")
    comsol = report.get("comsol")
    if not isinstance(comsol, Mapping) or comsol.get("exact_version") != expected.get("comsol_version"):
        stale_reasons.append("COMSOL version changed")
    if stale_reasons:
        return {"valid": False, "classification": "stale", "reasons": stale_reasons}
    corrections = report.get("fields_requiring_correction")
    missing_exports = report.get("required_exports_missing")
    comparison = report.get("mapping_comparison")
    inventory = report.get("actual_file_inventory")
    if (
        not isinstance(corrections, list)
        or not isinstance(missing_exports, list)
        or not isinstance(comparison, Mapping)
        or comparison.get("required_corrections") != corrections
        or comparison.get("required_missing_exports") != missing_exports
        or not isinstance(comparison.get("observations"), list)
        or not isinstance(inventory, list)
    ):
        return {
            "valid": False,
            "classification": "malformed",
            "reasons": ["mapping comparison, correction, missing-export, or inventory evidence is malformed"],
        }
    failure_reasons: list[str] = []
    if report.get("status") != "mapping_observation_complete":
        failure_reasons.append(f"probe status is {report.get('status')!r}")
    if report.get("exit_code") != 0 or report.get("timed_out") is not False or report.get("start_error") is not None:
        failure_reasons.append("COMSOL execution did not succeed")
    if corrections:
        failure_reasons.append("required mapping corrections are present")
    if missing_exports:
        failure_reasons.append("required exports are missing")
    observations = comparison["observations"]
    if report.get("status") == "mapping_observation_complete":
        if not observations or not inventory:
            failure_reasons.append("successful probe lacks observed export inventory")
        for observation in observations:
            columns = observation.get("columns") if isinstance(observation, Mapping) else None
            if (
                not isinstance(observation, Mapping)
                or not observation.get("matched_relative_paths")
                or observation.get("delimiter_matches") is not True
                or observation.get("temporal_structure_error") is not None
                or not isinstance(columns, Mapping)
                or any(not isinstance(column, Mapping) or column.get("observed_exact_header_match") is not True for column in columns.values())
            ):
                failure_reasons.append("successful probe observations do not prove exact source, delimiter, temporal, and header matches")
                break
    if failure_reasons:
        return {"valid": False, "classification": "failed", "reasons": failure_reasons}
    return {"valid": True, "classification": "valid", "reasons": []}


def discover_mapping_evidence(
    *,
    storage_root: Path | str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Find one valid immutable probe report using a bounded deterministic scan."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    root = common.paths.get_generation_meta_root(storage_root=storage) / "mapping_probes"
    if not root.is_dir() or root.is_symlink():
        return {
            "status": "mapping_evidence_missing",
            "reason": "no safe mapping-probe evidence directory exists",
            "valid_report": None,
            "inspected_reports": [],
        }
    candidates = tuple(
        sorted(
            (
                directory / "mapping_probe.json"
                for directory in root.iterdir()
                if directory.is_dir() and not directory.is_symlink() and (directory / "mapping_probe.json").is_file()
            ),
            reverse=True,
        )
    )
    inspected: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            report = load_mapping_probe(candidate)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            inspected.append({"path": str(candidate), "classification": "malformed", "reasons": [str(error)]})
            continue
        if report.get("simulation_profile") != expected.get("simulation_profile"):
            continue
        evaluation = evaluate_mapping_probe_report(report, expected)
        inspected.append({"path": str(candidate), **evaluation})
        if evaluation["valid"]:
            return {
                "status": "mapping_evidence_valid",
                "reason": None,
                "valid_report": str(candidate),
                "inspected_reports": inspected,
            }
    relevant = [record for record in inspected if record.get("classification") != "malformed"]
    if not relevant:
        status = "mapping_evidence_missing"
        reason = "no mapping-probe report exists for the current simulation profile"
    else:
        status = "mapping_evidence_invalid" if any(record["classification"] == "failed" for record in relevant) else "mapping_evidence_stale"
        reason = "; ".join(dict.fromkeys(reason for record in relevant for reason in record["reasons"]))
    return {
        "status": status,
        "reason": reason,
        "valid_report": None,
        "inspected_reports": inspected,
    }


def mapping_evidence_status(
    campaign_path: Path | str,
    *,
    storage_root: Path | str,
    comsol_version_output: str,
) -> dict[str, Any]:
    """Return current durable mapping-evidence status for one complete campaign."""
    expected = build_mapping_evidence_context(
        campaign_path,
        comsol_version_output=comsol_version_output,
    )
    return {
        "expected_identity": expected,
        **discover_mapping_evidence(storage_root=storage_root, expected=expected),
    }


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
    """Compare observed exports with explicit expected mapping declarations."""
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
        source_key = f"{prefix}:exports[{index}].source"
        pattern = export.get("source")
        matches = [] if not isinstance(pattern, str) else by_basename.get(pattern, [])
        if not isinstance(pattern, str):
            required_corrections.append(source_key)
        elif not matches:
            required_missing_exports.append({"role": role, "declared_pattern": pattern})
        elif len(matches) > 1:
            required_corrections.append(source_key)

        table = matches[0].get("table") if len(matches) == 1 else None
        raw_header = table.get("raw_header", []) if isinstance(table, dict) else []
        expected_units = {
            source_header: units_by_logical[logical] for logical, source_header in export["columns"].items() if isinstance(source_header, str)
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
            required_corrections.append(f"{prefix}:exports[{index}].delimiter")
        column_results: dict[str, Any] = {}
        for logical, source_header in export["columns"].items():
            key = f"{prefix}:exports[{index}].columns.{logical}"
            matches_header = isinstance(source_header, str) and source_header in observed_header
            if len(matches) == 1 and not matches_header:
                required_corrections.append(key)
            column_results[logical] = {
                "declared_source_header": source_header,
                "observed_exact_header_match": matches_header,
            }
        observations.append(
            {
                "role": role,
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
    comsol_version_output, comsol_exact_version = _query_comsol_version(str(config.execution_values["site"]["comsol_executable"]))
    try:
        mapping_contract_sha256 = mapping_contract.mapping_contract_sha256(
            config.profile.id,
            config.scientific_values["output_contract"],
        )
    except (TypeError, ValueError):
        mapping_contract_sha256 = None
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
            "mapping_contract_sha256": mapping_contract_sha256,
            "case_id": prepared.bundle.case_id,
            "case_input_id": prepared.bundle.case_input_id,
            "git_commit": git_commit,
            "slurm_job_id": slurm_job_id,
            "hostname": socket.gethostname(),
            "comsol": {
                "exact_version": comsol_exact_version,
                "version_command": [str(config.execution_values["site"]["comsol_executable"]), "-version"],
                "version_output": comsol_version_output,
            },
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
    allowed_statuses = {
        "mapping_observation_complete",
        "mapping_update_required",
        "required_export_missing",
        "comsol_execution_failed",
    }
    template = report.get("template")
    comsol = report.get("comsol")
    corrections = report.get("fields_requiring_correction")
    missing_exports = report.get("required_exports_missing")
    contract_sha256 = report.get("mapping_contract_sha256")
    if (
        not isinstance(report.get("simulation_profile"), str)
        or report.get("status") not in allowed_statuses
        or not isinstance(template, dict)
        or re.fullmatch(r"[0-9a-f]{64}", str(template.get("sha256"))) is None
        or not isinstance(comsol, dict)
        or not isinstance(comsol.get("exact_version"), str)
        or not isinstance(comsol.get("version_output"), str)
        or not isinstance(corrections, list)
        or not isinstance(missing_exports, list)
    ):
        message = f"Mapping-probe runtime identity or outcome is malformed: {report_path}"
        raise ValueError(message)
    if report["status"] == "mapping_observation_complete" and (
        re.fullmatch(r"[0-9a-f]{64}", str(contract_sha256)) is None
        or not report.get("actual_file_inventory")
        or corrections
        or missing_exports
        or report.get("exit_code") != 0
        or report.get("timed_out") is not False
        or report.get("start_error") is not None
    ):
        message = f"Successful mapping-probe evidence is internally inconsistent: {report_path}"
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
