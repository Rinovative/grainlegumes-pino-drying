"""
generation_runtime_preparation.py

Prepare isolated runtime workspaces from deterministic scientific case bundles.
Responsibilities:
  - Create one marked case workspace beneath approved scratch storage
  - Generate case-local adapters and copy a digest-verified COMSOL model
  - Preserve cleanup boundaries when preparation fails
Design principles:
  - Scientific bundle construction remains owned by the cases package
  - Workspace creation and template copying remain runtime concerns
This module does NOT:
  - Execute COMSOL, convert outputs, or publish canonical case results
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src import common
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_contract
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract

from . import generation_runtime_comsol as comsol_service
from . import generation_runtime_workspace as workspace_service

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """One isolated work directory containing adapters and a model copy."""

    bundle: case_service.CaseBundle
    work_directory: Path
    work_root: Path
    workspace_run_id: str
    workspace_marker: Path
    input_generation_id: str
    canonical_raw_directory: Path
    model_path: Path
    exports_directory: Path
    runtime_directory: Path


class CasePreparationError(RuntimeError):
    """Report preparation failure while preserving marked-workspace identity."""

    def __init__(
        self,
        message: str,
        *,
        work_directory: Path,
        work_root: Path,
        workspace_run_id: str,
    ) -> None:
        """Initialize one preparation error with its cleanup boundaries."""
        super().__init__(message)
        self.work_directory = work_directory
        self.work_root = work_root
        self.workspace_run_id = workspace_run_id


def _require_template_digest(path: Path, expected_sha256: str, *, message: str) -> None:
    """Require one template copy to retain its preflight digest."""
    if common.serialization.file_sha256(path) != expected_sha256:
        raise RuntimeError(message)


def _materialize_canonical_raw_bundle(
    config: config_contract.GenerationConfig,
    case_index: int,
    work_directory: Path,
    *,
    storage_root: Path,
    input_generation_id: str | None,
) -> tuple[case_service.CaseBundle, str, Path]:
    """Copy an exactly admitted persisted raw case into isolated scratch."""
    case_id = config.case_id(case_index)
    try:
        reference = (
            input_service.admit_configured_input_case(
                config,
                case_index,
                storage_root=storage_root,
            )
            if input_generation_id is None
            else input_service.admit_persisted_input_case(
                config,
                case_index,
                input_generation_id,
                storage_root=storage_root,
            )
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = (
            f"Canonical input readiness is required before worker execution for {case_id}; "
            "run submit-campaign or generate-input-cases on the CPU login node first."
        )
        raise RuntimeError(message) from error
    source_case_json = reference.case_directory / "case.json"
    payload = json.loads(source_case_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Canonical raw case payload is invalid: {reference.case_directory}"
        raise TypeError(message)
    case_service.validate_case_payload_schema(payload)
    target_case_json = work_directory / "case.json"
    shutil.copy2(source_case_json, target_case_json)
    if source_case_json.read_bytes() != target_case_json.read_bytes():
        message = f"Canonical raw case.json changed during scratch materialization: {source_case_json}"
        raise RuntimeError(message)
    input_paths: list[Path] = []
    for filename, identity in sorted(payload["input_files"].items()):
        source_path = reference.input_directory / filename
        target_path = work_directory / filename
        shutil.copy2(source_path, target_path)
        if (
            target_path.stat().st_size != identity["size_bytes"]
            or common.serialization.file_sha256(target_path) != identity["sha256"]
            or target_path.read_bytes() != source_path.read_bytes()
        ):
            message = f"Canonical raw input changed during scratch materialization: {source_path}"
            raise RuntimeError(message)
        input_paths.append(target_path)
    scalar_handoff = (
        None if config.profile.id != profiles.TRANSIENT_DRYING_PROFILE else scalar_handoff_contract.admit_case_scalar_handoff(payload, work_directory)
    )
    return (
        case_service.CaseBundle(
            directory=work_directory,
            case_id=case_id,
            case_input_id=str(payload["case_input_id"]),
            simulation_case_id=str(payload["simulation_case_id"]),
            case_payload=payload,
            input_paths=tuple(input_paths),
            scalar_handoff=scalar_handoff,
        ),
        reference.source_id,
        reference.case_directory.resolve(),
    )


def prepare_case_work_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
    input_generation_id: str | None = None,
) -> PreparedCase:
    """
    Prepare a fresh case directory beside a digest-verified disposable model.

    A supplied input-generation identity selects an already admitted historical
    source for postprocessing replay. Normal solver execution omits it and stays
    bound to the exact active source commit.
    """
    case_id = config.case_id(case_index)
    _require_template_digest(
        config.template_path,
        config.template_sha256,
        message=f"COMSOL template changed after configuration preflight: {config.template_path}",
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    work_directory, root, marker_path = workspace_service.create_case_workspace(
        config,
        case_id=case_id,
        storage_root=storage,
        work_root=work_root,
    )
    run_id = workspace_service.workspace_run_id(config)
    model_path = work_directory / comsol_service.WORK_MODEL_FILENAME
    try:
        (
            bundle,
            admitted_input_generation_id,
            canonical_raw_directory,
        ) = _materialize_canonical_raw_bundle(
            config,
            case_index,
            work_directory,
            storage_root=storage,
            input_generation_id=input_generation_id,
        )
        shutil.copyfile(config.template_path, model_path)
        _require_template_digest(
            model_path,
            config.template_sha256,
            message=f"Copied COMSOL template digest mismatch in {work_directory}.",
        )
        exports_directory = work_directory / config.scientific_values["output_contract"]["exports_root"]
        runtime_directory = work_directory / "runtime"
        exports_directory.mkdir()
        runtime_directory.mkdir()
        _require_template_digest(
            config.template_path,
            config.template_sha256,
            message=f"Source COMSOL template changed during case preparation: {config.template_path}",
        )
    except BaseException as error:
        message = f"Could not prepare isolated case workspace {work_directory}: {error}"
        raise CasePreparationError(
            message,
            work_directory=work_directory,
            work_root=root,
            workspace_run_id=run_id,
        ) from error
    return PreparedCase(
        bundle=bundle,
        work_directory=work_directory,
        work_root=root,
        workspace_run_id=run_id,
        workspace_marker=marker_path,
        input_generation_id=admitted_input_generation_id,
        canonical_raw_directory=canonical_raw_directory,
        model_path=model_path,
        exports_directory=exports_directory,
        runtime_directory=runtime_directory,
    )
