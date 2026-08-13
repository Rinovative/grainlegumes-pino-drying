"""
Native solver, workspace, cluster, and single-batch execution services.

Provides:
- batch: single-case execution and terminal batch admission
- cluster: scheduler planning and submission
- license: bounded temporary floating-license retry evidence
- comsol: fixed COMSOL command and workspace-name conventions
- preflight: executable runtime preflight validation
- preparation: isolated model and case-workspace preparation
- workspace: bounded scratch and publication staging
- PUBLICATION_SCHEMA_VERSION: canonical case-publication schema version
- CASE_FAILURE_SCHEMA_KIND: canonical persisted case-failure schema kind
- CASE_FAILURE_SCHEMA_VERSION: canonical persisted case-failure schema version
- CaseCleanupError: post-publication scratch-cleanup failure
- CaseExecutionError: structured case-execution failure
- CaseInterruptedError: cooperative case-cancellation failure
- CollectedExport: validated raw export identity
- ExecutionResult: successful solver and conversion result
- CaseRunOutcome: completed or reused case outcome
- ArtifactEvidence: hash-validated terminal artifact evidence
- HDF5IdentityEvidence: admitted canonical HDF5 identity evidence
- TerminalCaseEvidence: admitted terminal case evidence
- TerminalBatchEvidence: admitted terminal batch evidence
- CasePreparationError: isolated case-preparation failure
- PreparedCase: prepared isolated case workspace
- reset_runtime_cancellation: clear worker cancellation state
- runtime_cancellation_requested: inspect worker cancellation state
- request_runtime_cancellation: cancel active worker solvers
- case_lock_path: persistent case-lock path resolution
- raw_case_directory: canonical raw case-directory resolution
- processed_case_directory: canonical processed case-directory resolution
- batch_meta_directory: canonical batch-metadata directory resolution
- initialize_batch_metadata: immutable batch-metadata publication
- resolve_comsol_executable: configured COMSOL executable resolution
- build_comsol_command: safe single-node COMSOL command construction
- collect_exports: explicit raw-export collection
- execute_prepared_case: isolated solver execution and conversion
- prepare_case_work_directory: isolated case-workspace preparation
- case_failure_path: persistent case-failure path resolution
- case_failure_artifacts_directory: retained failure-artifact path resolution
- record_case_failure: durable failed-case evidence publication
- clear_case_failure: validated failure-evidence cleanup
- case_failure_is_recorded: failed-case evidence admission
- validate_completed_case: completed-case publication validation
- completed_case_is_valid: completed-case validity inspection
- publish_completed_case: atomic completed-case publication
- run_case: one-case execution and terminal publication
- finalize_batch: terminal batch-manifest publication
- admit_terminal_batch: config-independent terminal batch admission
- validate_terminal_batch: config-bound terminal batch validation
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import generation_runtime_batch as batch
    from . import generation_runtime_cluster as cluster
    from . import generation_runtime_comsol as comsol
    from . import generation_runtime_license as license  # noqa: A004 -- public service name
    from . import generation_runtime_preflight as preflight
    from . import generation_runtime_preparation as preparation
    from . import generation_runtime_workspace as workspace
    from .generation_runtime_batch import (
        CASE_FAILURE_SCHEMA_KIND,
        CASE_FAILURE_SCHEMA_VERSION,
        PUBLICATION_SCHEMA_VERSION,
        ArtifactEvidence,
        CaseCleanupError,
        CaseExecutionError,
        CaseInterruptedError,
        CaseRunOutcome,
        CollectedExport,
        ExecutionResult,
        HDF5IdentityEvidence,
        TerminalBatchEvidence,
        TerminalCaseEvidence,
        admit_terminal_batch,
        batch_meta_directory,
        case_failure_artifacts_directory,
        case_failure_is_recorded,
        case_failure_path,
        case_lock_path,
        clear_case_failure,
        collect_exports,
        completed_case_is_valid,
        execute_prepared_case,
        finalize_batch,
        initialize_batch_metadata,
        processed_case_directory,
        publish_completed_case,
        raw_case_directory,
        record_case_failure,
        request_runtime_cancellation,
        reset_runtime_cancellation,
        run_case,
        runtime_cancellation_requested,
        validate_completed_case,
        validate_terminal_batch,
    )
    from .generation_runtime_comsol import build_comsol_command, resolve_comsol_executable
    from .generation_runtime_preparation import (
        CasePreparationError,
        PreparedCase,
        prepare_case_work_directory,
    )

_MODULES = {
    "batch": "generation_runtime_batch",
    "cluster": "generation_runtime_cluster",
    "comsol": "generation_runtime_comsol",
    "license": "generation_runtime_license",
    "preflight": "generation_runtime_preflight",
    "preparation": "generation_runtime_preparation",
    "workspace": "generation_runtime_workspace",
}
_BATCH_EXPORTS = frozenset(
    {
        "CASE_FAILURE_SCHEMA_KIND",
        "CASE_FAILURE_SCHEMA_VERSION",
        "PUBLICATION_SCHEMA_VERSION",
        "ArtifactEvidence",
        "CaseCleanupError",
        "CaseExecutionError",
        "CaseInterruptedError",
        "CaseRunOutcome",
        "CollectedExport",
        "ExecutionResult",
        "HDF5IdentityEvidence",
        "TerminalBatchEvidence",
        "TerminalCaseEvidence",
        "admit_terminal_batch",
        "batch_meta_directory",
        "case_failure_artifacts_directory",
        "case_failure_is_recorded",
        "case_failure_path",
        "case_lock_path",
        "clear_case_failure",
        "collect_exports",
        "completed_case_is_valid",
        "execute_prepared_case",
        "finalize_batch",
        "initialize_batch_metadata",
        "processed_case_directory",
        "publish_completed_case",
        "raw_case_directory",
        "record_case_failure",
        "request_runtime_cancellation",
        "reset_runtime_cancellation",
        "run_case",
        "runtime_cancellation_requested",
        "validate_completed_case",
        "validate_terminal_batch",
    }
)
_COMSOL_EXPORTS = frozenset({"build_comsol_command", "resolve_comsol_executable"})
_PREPARATION_EXPORTS = frozenset(
    {
        "CasePreparationError",
        "PreparedCase",
        "prepare_case_work_directory",
    }
)
__all__ = [
    "CASE_FAILURE_SCHEMA_KIND",
    "CASE_FAILURE_SCHEMA_VERSION",
    "PUBLICATION_SCHEMA_VERSION",
    "ArtifactEvidence",
    "CaseCleanupError",
    "CaseExecutionError",
    "CaseInterruptedError",
    "CasePreparationError",
    "CaseRunOutcome",
    "CollectedExport",
    "ExecutionResult",
    "HDF5IdentityEvidence",
    "PreparedCase",
    "TerminalBatchEvidence",
    "TerminalCaseEvidence",
    "admit_terminal_batch",
    "batch",
    "batch_meta_directory",
    "build_comsol_command",
    "case_failure_artifacts_directory",
    "case_failure_is_recorded",
    "case_failure_path",
    "case_lock_path",
    "clear_case_failure",
    "cluster",
    "collect_exports",
    "completed_case_is_valid",
    "comsol",
    "execute_prepared_case",
    "finalize_batch",
    "initialize_batch_metadata",
    "license",
    "preflight",
    "preparation",
    "prepare_case_work_directory",
    "processed_case_directory",
    "publish_completed_case",
    "raw_case_directory",
    "record_case_failure",
    "request_runtime_cancellation",
    "reset_runtime_cancellation",
    "resolve_comsol_executable",
    "run_case",
    "runtime_cancellation_requested",
    "validate_completed_case",
    "validate_terminal_batch",
    "workspace",
]


def __getattr__(name: str) -> object:
    """Resolve one declared runtime module or stable operation."""
    module_name = _MODULES.get(name)
    if module_name is not None:
        value = import_module(f"{__name__}.{module_name}")
    elif name in _BATCH_EXPORTS:
        value = getattr(import_module(f"{__name__}.generation_runtime_batch"), name)
    elif name in _COMSOL_EXPORTS:
        value = getattr(import_module(f"{__name__}.generation_runtime_comsol"), name)
    elif name in _PREPARATION_EXPORTS:
        value = getattr(import_module(f"{__name__}.generation_runtime_preparation"), name)
    else:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    globals()[name] = value
    return value
