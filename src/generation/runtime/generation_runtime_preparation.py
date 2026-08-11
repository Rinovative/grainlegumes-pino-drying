"""
===============================================================================
generation_runtime_preparation.py
===============================================================================
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
===============================================================================
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src import common
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_contract

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


def prepare_case_work_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> PreparedCase:
    """Prepare a fresh case directory beside a digest-verified disposable model."""
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
    model_path = work_directory / "model.mph"
    try:
        bundle = case_service.generate_case_input_bundle(
            config,
            case_index,
            work_directory,
            _allow_workspace_marker=True,
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
        model_path=model_path,
        exports_directory=exports_directory,
        runtime_directory=runtime_directory,
    )
