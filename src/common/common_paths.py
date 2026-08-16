"""
===============================================================================
common_paths.py
===============================================================================
Resolve repository paths and the unified scientific storage lifecycle.

Responsibilities:
  - Resolve the sole public ``STORAGE_ROOT`` contract and its numbered areas
  - Derive generation, dataset, experiment, and coordination-state paths
  - Resolve final datasets, metadata snapshots, runs, studies, and artifacts
  - Identify completed and evaluable saved-run directories by required local files

Design principles:
  - Every scientific path descends from one explicitly resolved storage root
  - The portable default is the storage directory beside the repository
  - Generation, immutable datasets, and experiments remain separate lifecycle areas
  - Logical names are validated as single path components before composition
  - The current saved-run file contract is explicit and centralized

This module does NOT:
  - Create datasets, runs, checkpoints, summaries, or analysis artifacts
  - Decide dataset membership, experiment semantics, or resume eligibility
  - Validate run or artifact contents beyond shallow discovery predicates
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RUN_CONFIG_FILENAME = "config.yaml"
RUN_NORMALIZER_FILENAME = "normalizer.pt"
RUN_BEST_CHECKPOINT_FILENAME = "best_checkpoint.pt"
RUN_LAST_CHECKPOINT_FILENAME = "last_checkpoint.pt"
RUN_SPLIT_INDICES_FILENAME = "split_indices.pt"
RUN_SUMMARY_FILENAME = "summary.json"
GENERATION_AREA_NAME = "01_generation"
DATASETS_AREA_NAME = "02_datasets"
EXPERIMENTS_AREA_NAME = "03_experiments"
CURRENT_RUN_REQUIRED_FILES = (
    RUN_CONFIG_FILENAME,
    RUN_SPLIT_INDICES_FILENAME,
    RUN_NORMALIZER_FILENAME,
    RUN_BEST_CHECKPOINT_FILENAME,
    RUN_LAST_CHECKPOINT_FILENAME,
    RUN_SUMMARY_FILENAME,
)
EVALUABLE_RUN_REQUIRED_FILES = (
    RUN_CONFIG_FILENAME,
    RUN_SPLIT_INDICES_FILENAME,
    RUN_NORMALIZER_FILENAME,
    RUN_BEST_CHECKPOINT_FILENAME,
    RUN_SUMMARY_FILENAME,
)
RESUME_RUN_REQUIRED_FILES = (
    RUN_CONFIG_FILENAME,
    RUN_SPLIT_INDICES_FILENAME,
    RUN_NORMALIZER_FILENAME,
    RUN_LAST_CHECKPOINT_FILENAME,
    RUN_SUMMARY_FILENAME,
)


def get_project_root() -> Path:
    """
    Return the project root.

    Returns
    -------
    Path
        Root selected by ``PROJECT_ROOT`` or inferred from this module.

    """
    root = os.environ.get("PROJECT_ROOT")
    if root:
        return Path(root).expanduser()
    return Path(__file__).resolve().parents[2]


def get_storage_root(*, storage_root: Path | str | None = None) -> Path:
    """
    Return the authoritative scientific storage root.

    Parameters
    ----------
    storage_root : Path | str | None, optional
        Explicit internal boundary used by controlled callers and tests. When
        omitted, ``STORAGE_ROOT`` is the sole environment override and the
        portable default is ``<repository-parent>/storage``.

    Returns
    -------
    Path
        Expanded storage-root path without creating it.

    """
    if storage_root is not None:
        return Path(storage_root).expanduser()
    root = os.environ.get("STORAGE_ROOT")
    if root:
        return Path(root).expanduser()
    return get_project_root().parent / "storage"


def get_generation_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the generated-simulation lifecycle area."""
    return get_storage_root(storage_root=storage_root) / GENERATION_AREA_NAME


def get_datasets_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the immutable final-dataset lifecycle area."""
    return get_storage_root(storage_root=storage_root) / DATASETS_AREA_NAME


def get_experiments_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the run, study, log, and analysis lifecycle area."""
    return get_storage_root(storage_root=storage_root) / EXPERIMENTS_AREA_NAME


def get_generation_meta_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the authoritative generation metadata stage."""
    return get_generation_root(storage_root=storage_root) / "meta"


def get_generation_raw_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the authoritative generated raw-input stage."""
    return get_generation_root(storage_root=storage_root) / "raw"


def get_generation_processed_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the authoritative generated COMSOL-output stage."""
    return get_generation_root(storage_root=storage_root) / "processed"


def get_generation_state_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the transient generation-coordination state root."""
    return get_generation_root(storage_root=storage_root) / ".state"


def get_generation_input_locks_root(*, storage_root: Path | str | None = None) -> Path:
    """Return canonical raw-input batch lock anchors."""
    return get_generation_state_root(storage_root=storage_root) / "raw-inputs" / "locks"


def get_generation_input_transactions_root(*, storage_root: Path | str | None = None) -> Path:
    """Return canonical raw-input batch publication transactions."""
    return get_generation_state_root(storage_root=storage_root) / "raw-inputs" / "transactions"


def get_generation_performance_benchmark_root(
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the Generation metadata namespace for performance evidence."""
    return get_generation_meta_root(storage_root=storage_root) / "performance_benchmarks"


def get_dataset_metadata_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the validated dataset-metadata stage."""
    return get_datasets_root(storage_root=storage_root) / "meta"


def get_dataset_payload_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the immutable final-dataset payload stage."""
    return get_datasets_root(storage_root=storage_root) / "raw"


def get_dataset_state_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the hidden root for dataset-publication coordination state."""
    return get_datasets_root(storage_root=storage_root) / ".state"


def get_experiment_state_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the hidden root for experiment coordination state."""
    return get_experiments_root(storage_root=storage_root) / ".state"


def get_dataset_build_locks_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the persistent OS-lock-anchor root for dataset publication."""
    return get_dataset_state_root(storage_root=storage_root) / "dataset-builds" / "locks"


def get_dataset_build_transactions_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the recoverable dataset-publication transaction registry."""
    return get_dataset_state_root(storage_root=storage_root) / "dataset-builds" / "transactions"


def get_run_locks_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the persistent OS-lock-anchor root for saved-run writers."""
    return get_experiment_state_root(storage_root=storage_root) / "runs" / "locks"


def resolve_queue_log_dir(
    scope: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one validated experiment scope's host-visible queue-log directory."""
    scope = validate_logical_name(scope, label="queue log scope")
    return get_experiments_root(storage_root=storage_root) / scope / "logs" / "queue"


def resolve_generation_input_metadata_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one canonical input-batch metadata directory."""
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    return get_generation_meta_root(storage_root=storage_root) / name


def resolve_generation_input_raw_batch_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one canonical pre-execution raw input-batch directory."""
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    return get_generation_raw_root(storage_root=storage_root) / name


def resolve_generation_raw_case_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one immutable canonical raw case directory."""
    return resolve_generation_input_raw_batch_directory(
        batch_storage_name,
        storage_root=storage_root,
    ) / validate_logical_name(case_id, label="case_id")


def resolve_generation_raw_case_inputs_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the pre-COMSOL adapters for one canonical raw case."""
    return (
        resolve_generation_raw_case_directory(
            batch_storage_name,
            case_id,
            storage_root=storage_root,
        )
        / "inputs"
    )


def resolve_generation_processed_case_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one canonical post-COMSOL case directory."""
    return resolve_generated_batch_dir(
        batch_storage_name,
        stage="processed",
        storage_root=storage_root,
    ) / validate_logical_name(case_id, label="case_id")


def resolve_generation_comsol_exports_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve direct COMSOL export storage for one completed case."""
    return (
        resolve_generation_processed_case_directory(
            batch_storage_name,
            case_id,
            storage_root=storage_root,
        )
        / "comsol_exports"
    )


def resolve_generation_failure_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve batch-owned failed-execution evidence for one case."""
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    case_id = validate_logical_name(case_id, label="case_id")
    return get_generation_meta_root(storage_root=storage_root) / name / "failures" / case_id


def resolve_generation_state_batch_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve flat private runtime coordination state for one generation batch."""
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    return get_generation_state_root(storage_root=storage_root) / name


def resolve_generation_case_lock_path(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one persistent case-level execution lock anchor."""
    state = resolve_generation_state_batch_directory(
        batch_storage_name,
        storage_root=storage_root,
    )
    case_id = validate_logical_name(case_id, label="case_id")
    return state / "locks" / f"{case_id}.lock"


def resolve_generation_case_publications_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve atomic processed-case publication staging for one batch."""
    return (
        resolve_generation_state_batch_directory(
            batch_storage_name,
            storage_root=storage_root,
        )
        / "publications"
    )


def resolve_generation_input_lock_path(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the sole publisher lock for one canonical input batch."""
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    return get_generation_input_locks_root(storage_root=storage_root) / f"{name}.lock"


def resolve_generation_input_transaction_directory(
    input_generation_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one deterministic canonical input-generation transaction."""
    input_generation_id = validate_logical_name(
        input_generation_id,
        label="input_generation_id",
    )
    return get_generation_input_transactions_root(storage_root=storage_root) / input_generation_id


def resolve_dataset_build_lock_path(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one dataset builder's persistent advisory-lock anchor."""
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    return get_dataset_build_locks_root(storage_root=storage_root) / f"dataset-{dataset_id}.lock"


def resolve_dataset_build_transaction_path(
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one dataset builder's durable recovery marker."""
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    return get_dataset_build_transactions_root(storage_root=storage_root) / f"dataset-{dataset_id}.json"


def resolve_run_lock_path(
    run_dir: Path | str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one path-qualified saved-run writer lock below experiment state."""
    canonical_run = Path(run_dir).expanduser().resolve(strict=False)
    digest = hashlib.sha256(os.fsencode(canonical_run)).hexdigest()
    return get_run_locks_root(storage_root=storage_root) / f"run-{digest}.lock"


def resolve_artifact_lock_path(
    artifact_dir: Path | str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one path-qualified analysis-artifact lock below experiment state."""
    canonical_artifact = Path(artifact_dir).expanduser().resolve(strict=False)
    digest = hashlib.sha256(os.fsencode(canonical_artifact)).hexdigest()
    return get_run_locks_root(storage_root=storage_root) / f"artifact-{digest}.lock"


def validate_logical_name(value: object, *, label: str) -> str:
    """
    Validate one logical identifier for safe use as a path component.

    Parameters
    ----------
    value : object
        Candidate identifier. It must be a non-empty, already-trimmed string
        containing no separator, NUL, absolute path, ``.`` or ``..`` value.
    label : str
        Contract name included in validation errors.

    Returns
    -------
    str
        The unchanged validated component.

    Raises
    ------
    ValueError
        If ``value`` is not exactly one safe logical component.

    """
    if not isinstance(value, str) or not value or value.strip() != value:
        msg = f"{label} must be a single non-empty path component, got {value!r}."
        raise ValueError(msg)
    if value in {".", ".."} or Path(value).is_absolute() or "/" in value or "\\" in value or "\x00" in value:
        msg = f"{label} must be a single non-empty path component, got {value!r}."
        raise ValueError(msg)
    return value


def resolve_dataset_dir(dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one logical dataset directory.

    Parameters
    ----------
    dataset_id : str
        Non-empty logical dataset identifier.
    dataset_root : Path | str | None, optional
        Explicit dataset root. Output roots are never consulted.

    Returns
    -------
    Path
        ``<dataset_root>/<dataset_id>``.

    Raises
    ------
    ValueError
        If ``dataset_id`` is not one safe logical path component.

    """
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    root = Path(dataset_root).expanduser() if dataset_root is not None else get_dataset_payload_root()
    return root / dataset_id


def resolve_dataset_metadata_dir(
    dataset_id: str,
    *,
    metadata_root: Path | str | None = None,
) -> Path:
    """
    Resolve one dataset's validated metadata snapshot directory.

    The directory is distinct from the authoritative ``.pt`` payload and
    contains only small, builder-validated training/evaluation provenance.
    """
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    root = Path(metadata_root).expanduser() if metadata_root is not None else get_dataset_metadata_root()
    return root / dataset_id


def resolve_dataset_path(dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one logical final training-dataset file.

    Parameters
    ----------
    dataset_id : str
        Non-empty logical dataset identifier.
    dataset_root : Path | str | None, optional
        Explicit dataset root. Output roots are never consulted.

    Returns
    -------
    Path
        ``<dataset_root>/<dataset_id>/<dataset_id>.pt``.

    Raises
    ------
    ValueError
        If ``dataset_id`` is not one safe logical path component.

    """
    return resolve_dataset_dir(dataset_id, dataset_root=dataset_root) / f"{dataset_id}.pt"


def resolve_generated_batch_dir(
    batch_storage_name: str,
    *,
    stage: str,
    storage_root: Path | str | None = None,
) -> Path:
    """
    Resolve one generated-data batch directory.

    Parameters
    ----------
    batch_storage_name : str
        Flat semantic batch storage locator.
    stage : str
        Exact generated-data stage: raw or processed.
    storage_root : Path | str | None, optional
        Explicit unified storage root.

    Returns
    -------
    Path
        Batch directory below the selected Generation stage.

    Raises
    ------
    ValueError
        If batch_storage_name is unsafe or stage is unsupported.

    """
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    if stage not in {"raw", "processed"}:
        msg = f"stage must be 'raw' or 'processed', got {stage!r}."
        raise ValueError(msg)
    root = get_generation_root(storage_root=storage_root)
    return root / stage / name


def resolve_run_output_dir(
    task: str,
    run_name: str,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """
    Resolve a run output directory independently of dataset inputs.

    Parameters
    ----------
    task : str
        Registered task identifier.
    run_name : str
        Run name.
    output_root : Path | str | None, optional
        Explicit run/output root.

    Returns
    -------
    Path
        ``<output_root>/<task>/runs/<run_name>``.

    Raises
    ------
    ValueError
        If ``task`` or ``run_name`` is not one safe logical path component.

    """
    task = validate_logical_name(task, label="task")
    run_name = validate_logical_name(run_name, label="run_name")
    root = Path(output_root).expanduser() if output_root is not None else get_experiments_root()
    return root / task / "runs" / run_name


def resolve_optuna_trial_dir(
    task: str,
    study_name: str,
    run_name: str,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve a study-qualified trial whose leaf equals its canonical run name."""
    task = validate_logical_name(task, label="task")
    study_name = validate_logical_name(study_name, label="study_name")
    run_name = validate_logical_name(run_name, label="run_name")
    return resolve_study_dir(task, study_name, output_root=output_root) / "trials" / run_name


def resolve_study_dir(
    task: str,
    study_name: str,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve one Optuna study below the task-owned ``studies`` subtree."""
    task = validate_logical_name(task, label="task")
    study_name = validate_logical_name(study_name, label="study_name")
    root = Path(output_root).expanduser() if output_root is not None else get_experiments_root()
    return root / task / "studies" / study_name


def resolve_runs_root(task: str, *, output_root: Path | str | None = None) -> Path:
    """
    Resolve the directory containing saved runs for a task.

    Parameters
    ----------
    task : str
        Registered task identifier.
    output_root : Path | str | None, optional
        Explicit run/output root.

    Returns
    -------
    Path
        ``<output_root>/<task>/runs``.

    Raises
    ------
    ValueError
        If ``task`` is not one safe logical path component.

    """
    task = validate_logical_name(task, label="task")
    root = Path(output_root).expanduser() if output_root is not None else get_experiments_root()
    return root / task / "runs"


def resolve_run_config_path(run_dir: Path | str) -> Path:
    """
    Resolve the current run configuration path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to config.yaml file.

    """
    return Path(run_dir) / RUN_CONFIG_FILENAME


def resolve_best_checkpoint_file(run_dir: Path | str) -> Path:
    """
    Resolve the current best checkpoint path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to best_checkpoint.pt file.

    """
    return Path(run_dir) / RUN_BEST_CHECKPOINT_FILENAME


def resolve_last_checkpoint_file(run_dir: Path | str) -> Path:
    """
    Resolve the exact-resume checkpoint path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to last_checkpoint.pt.

    """
    return Path(run_dir) / RUN_LAST_CHECKPOINT_FILENAME


def resolve_split_indices_path(run_dir: Path | str) -> Path:
    """
    Resolve the split indices file path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to split_indices.pt file.

    """
    return Path(run_dir) / RUN_SPLIT_INDICES_FILENAME


def resolve_normalizer_path(run_dir: Path | str) -> Path:
    """
    Resolve the normalizer state file path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to normalizer.pt file.

    """
    return Path(run_dir) / RUN_NORMALIZER_FILENAME


def resolve_run_summary_path(run_dir: Path | str) -> Path:
    """
    Resolve the current run summary path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to summary.json file.

    """
    return Path(run_dir) / RUN_SUMMARY_FILENAME


def resolve_current_run_required_paths(run_dir: Path | str) -> tuple[Path, ...]:
    """
    Resolve required file paths for the current saved-run contract.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Required config, split, normalizer, best/last checkpoint, and summary paths.

    """
    run_dir = Path(run_dir)
    return tuple(run_dir / filename for filename in CURRENT_RUN_REQUIRED_FILES)


def missing_resume_run_files(run_dir: Path | str) -> tuple[Path, ...]:
    """
    Return files required before an explicit resume can inspect a run.

    Parameters
    ----------
    run_dir : Path | str
        Existing or prospective run output directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Missing config, split, normalizer, last-checkpoint, and summary paths in
        contract order. A best checkpoint is deliberately not required because
        an interrupted run may precede its first evaluation.

    """
    run_dir = Path(run_dir)
    return tuple(run_dir / filename for filename in RESUME_RUN_REQUIRED_FILES if not (run_dir / filename).is_file())


def missing_current_run_files(run_dir: Path | str) -> tuple[Path, ...]:
    """
    Return required current-contract run files missing from a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Missing required file paths.

    """
    return tuple(path for path in resolve_current_run_required_paths(run_dir) if not path.is_file())


def missing_evaluable_run_files(run_dir: Path | str) -> tuple[Path, ...]:
    """Return bundle-local files required by evidence-based evaluation."""
    path = Path(run_dir)
    return tuple(path / filename for filename in EVALUABLE_RUN_REQUIRED_FILES if not (path / filename).is_file())


def is_evaluable_run_dir(run_dir: Path | str) -> bool:
    """
    Return whether a directory has the shallow terminal evaluation contract.

    Full lifecycle, configuration, saved-data, checkpoint, digest, and writer
    checks remain owned by ``experiments.run.validate_evaluable_run``.
    """
    path = Path(run_dir)
    if not path.is_dir() or missing_evaluable_run_files(path):
        return False
    try:
        with resolve_run_summary_path(path).open(encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    schema_version = summary.get("schema_version")
    status = summary.get("status")
    return (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == 1
        and isinstance(status, str)
        and status not in {"initializing", "running"}
    )


def is_current_run_dir(run_dir: Path | str) -> bool:
    """
    Return whether a directory satisfies the current saved-run contract.

    This is a shallow discovery predicate. It requires every current run
    artifact plus a valid summary whose status is ``completed``.
    Consumers still call the full lifecycle validator before loading content.

    Parameters
    ----------
    run_dir : Path | str
        Candidate run output directory path.

    Returns
    -------
    bool
        ``True`` only when every required current-run file exists and the JSON
        summary has schema version 1 with status ``"completed"``.

    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir() or missing_current_run_files(run_dir):
        return False
    summary_path = resolve_run_summary_path(run_dir)
    try:
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    schema_version = summary.get("schema_version")
    return isinstance(schema_version, int) and not isinstance(schema_version, bool) and schema_version == 1 and summary.get("status") == "completed"


def resolve_analysis_root(run_dir: Path | str) -> Path:
    """
    Resolve the analysis artifact root for a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to the run's analysis artifact root.

    """
    return Path(run_dir) / "analysis"


def resolve_id_analysis_dir(run_dir: Path | str) -> Path:
    """
    Resolve the in-distribution artifact directory for a run.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to analysis/id.

    """
    return resolve_analysis_root(run_dir) / "id"


def resolve_ood_analysis_dir(run_dir: Path | str, dataset_name: str) -> Path:
    """
    Resolve the OOD artifact directory for a run and dataset.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.
    dataset_name : str
        Logical OOD dataset name.

    Returns
    -------
    Path
        Path to ``analysis/ood/<dataset_name>``.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not one safe logical path component.

    """
    dataset_name = validate_logical_name(dataset_name, label="dataset_name")
    return resolve_analysis_root(run_dir) / "ood" / dataset_name
