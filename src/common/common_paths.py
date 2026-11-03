"""Resolve repository, runtime, and scientific-storage paths.

Responsibilities:
  - Resolve repository, replaceable runtime, and scientific-storage root contracts
  - Detect source-tree mode and require its explicit repository, storage, and runtime roots
  - Derive generation, dataset, experiment, and coordination-state paths
  - Resolve Dataset bundles, config-named runs, Optuna studies and trials, and artifacts
  - Compose and invert run directories from the saved ``run.layout`` contract
  - Identify completed and evaluable saved-run directories by required local files
  - Name the logs/ folder of a campaign, Dataset, run or study and record which folders an ML job wrote for its log links

Design principles:
  - Durable scientific paths descend from one explicitly resolved storage root
  - Portable storage and runtime defaults are sibling directories beside the repository
  - A frozen source tree never infers storage or runtime from its own location
  - Generation, immutable datasets, and experiments remain separate lifecycle areas
  - Logical names are validated as single bounded path components before composition
  - Run and study directories are pure functions of the saved layout, never of what already exists
  - The current saved-run file contract is explicit and centralized

This module does NOT:
  - Create datasets, runs, checkpoints, summaries, or analysis artifacts
  - Decide dataset membership, experiment semantics, or resume eligibility
  - Validate run or artifact contents beyond shallow discovery predicates

Notes:
  Root and lifecycle navigation:
    Explicit storage/runtime roots take precedence over their environment overrides;
    otherwise the defaults are siblings of the resolved repository root. Scientific
    areas are 01_generation, 02_datasets, and 03_experiments beneath storage;
    source_snapshots/ (exact source evidence shared by all three areas) sits
    directly beneath storage, and runtime/sources/ holds the frozen read-only
    execution copies restored from those snapshots.
  Source-tree mode:
    A process is in tree mode exactly when SOURCE_TREE_ID is set. It then requires
    PROJECT_ROOT, STORAGE_ROOT, and RUNTIME_ROOT as explicit environment values and
    PROJECT_ROOT == RUNTIME_ROOT/sources/<SOURCE_TREE_ID>; no sibling default applies.
  Generation layout (relative to the generation root):
    Human files come first and machine records sit in reserved _record/ folders.
    <task>/<campaign>/ owns one <material>/ directory per material, logs/ with hard
    links of the campaign's job logs, cases.csv (a derived per-candidate view written
    when the campaign completes), .state/ for locks and staging, and
    _record/ with campaign_plan.json, selected_pool.json and execution/ (operational
    records keyed by <material>-<regime>). A material owns _record/<regime>/ (batch
    design and replacements/) beside its case_NNNN/ directories; a case owns inputs/
    (the immutable case definition), result/ (the published HDF5, provenance,
    success marker and evidence/ with the retained runtime evidence) and _record/
    with reservation.json (the only reservation copy), decision.json and attempts/
    (classified attempt records attempt_NNNN.json beside failure bundles run_NNNN/).
    _record, logs and cases.csv are the only reserved names; every enumeration
    recognizes them explicitly and rejects any other non-case entry. .state/ (raw-input
    transactions, workflow sessions) also sits beside the campaigns.
  Dataset layout (relative to the datasets root):
    <task>/<dataset>/ owns dataset_manifest.json and one fixed-name payload, namely
    dataset.pt for steady flow, or dataset.json plus transient_pt_shards/ for transient
    drying. Coordination state lives in .state/ (build locks, identity registry).
  Experiment layout (relative to the experiments root):
    <task>/<dataset>/<family>/<config_name>/seed-<n>[__a0ep-<epochs>][__rev-<revision>]/
    holds a steady run directly (without the A0 budget token); a transient seed
    directory carries its A0 epoch budget and holds the shared A0 in a0/ and each
    B/A1 pair in <pair>/{b,a1}/ beside the pair's experiment.json.
    <task>/<dataset>/optuna/<study>__ep-<epochs>[__rev-<revision>]/ holds the study
    database and summary and numbered trial runs in trials/<trial>/. The epoch
    budget names the unit because it is part of its scientific identity; a revision
    label always comes last.
  Saved-run discovery:
    Evaluable discovery requires the best checkpoint and a terminal summary, but
    not the last checkpoint; resume-file discovery requires the last checkpoint
    but not the best. These are shallow predicates, not lifecycle admission.
    Full saved-run admission is owned by experiments.run.validate_evaluable_run.

"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_CONFIG_FILENAME = "config.yaml"
RUN_NORMALIZER_FILENAME = "normalizer.pt"
RUN_BEST_CHECKPOINT_FILENAME = "best_checkpoint.pt"
RUN_LAST_CHECKPOINT_FILENAME = "last_checkpoint.pt"
RUN_SPLIT_INDICES_FILENAME = "split_indices.pt"
RUN_SUMMARY_FILENAME = "summary.json"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_PAYLOAD_FILENAME = "dataset.pt"
DATASET_INDEX_FILENAME = "dataset.json"
_DATASET_PAYLOAD_FILENAMES = {"steady_flow": DATASET_PAYLOAD_FILENAME, "transient_drying": DATASET_INDEX_FILENAME}
SOURCE_SNAPSHOTS_DIRNAME = "source_snapshots"
SOURCE_TREES_DIRNAME = "sources"
SOURCE_TREE_ID_ENVIRONMENT_VARIABLE = "SOURCE_TREE_ID"
_SOURCE_TREE_ID_PATTERN = re.compile(r"[0-9a-f]{12}-[0-9a-f]{12}")
GENERATION_AREA_NAME = "01_generation"
GENERATION_STATE_DIRNAME = ".state"
GENERATION_RECORD_DIRNAME = "_record"
JOB_LOGS_DIRNAME = "logs"
JOB_LOG_OWNERS_ENVIRONMENT_VARIABLE = "JOB_LOG_OWNERS_FILE"
GENERATION_EXECUTION_DIRNAME = "execution"
GENERATION_RUN_MANIFEST_FILENAME = "run.json"
GENERATION_CANCELLATIONS_FILENAME = "cancellations.json"
GENERATION_CAMPAIGN_PLAN_FILENAME = "campaign_plan.json"
GENERATION_SELECTED_POOL_FILENAME = "selected_pool.json"
GENERATION_CASES_VIEW_FILENAME = "cases.csv"
GENERATION_REPLACEMENTS_DIRNAME = "replacements"
GENERATION_RESERVATION_FILENAME = "reservation.json"
GENERATION_DECISION_FILENAME = "decision.json"
GENERATION_INPUTS_DIRNAME = "inputs"
GENERATION_RESULT_DIRNAME = "result"
GENERATION_EVIDENCE_DIRNAME = "evidence"
GENERATION_ATTEMPTS_DIRNAME = "attempts"
GENERATION_CASE_PAYLOAD_FILENAME = "case.json"
GENERATION_INPUT_REFERENCE_FILENAME = "input_reference.json"
GENERATION_RESULT_HDF5_FILENAME = "case.h5"
GENERATION_RESERVATION_STAGING_DIRNAME = "reservation-staging"
GENERATION_CONTROLLER_LOCK_FILENAME = "controller.lock"
GENERATION_SUBMISSION_LOCK_FILENAME = "submission.lock"
_CAMPAIGN_BATCH_PART_COUNT = 5
_CAMPAIGN_RUN_PART_COUNT = 3
_CASE_DIRECTORY_DEPTH = 5  # 01_generation/<task>/<campaign>/<material>/<case>
_GENERATION_CASE_ID = re.compile(r"case_[0-9]{4,}")
_GENERATION_ATTEMPT_RUN = re.compile(r"run_[0-9]{4,}")
# Campaign entries that are never materials; a material holds only its _record/ beside its cases.
_GENERATION_CAMPAIGN_RESERVED_NAMES = frozenset(
    {GENERATION_STATE_DIRNAME, GENERATION_RECORD_DIRNAME, JOB_LOGS_DIRNAME, GENERATION_CASES_VIEW_FILENAME}
)
DATASETS_AREA_NAME = "02_datasets"
EXPERIMENTS_AREA_NAME = "03_experiments"
OPTUNA_FAMILY = "optuna"
STUDY_TRIALS_DIRNAME = "trials"
EXPERIMENT_RECORD_FILENAME = "experiment.json"
_MAX_COMPONENT_BYTES = 255
_LAYOUT_KEYS = frozenset(
    {
        "task",
        "dataset",
        "family",
        "config_name",
        "model_label",
        "seed",
        "revision",
        "budget_epochs",
        "stage",
        "pair",
        "trial",
        "a0_id",
        "branch_id",
    }
)
_LAYOUT_STAGES = frozenset({"a0", "b", "a1", "single"})
_PAIRED_STAGES = frozenset({"b", "a1"})
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


def source_tree_id(environment: Mapping[str, str] | None = None) -> str | None:
    """
    Return the frozen source-tree ID of this process, or ``None`` outside tree mode.

    Parameters
    ----------
    environment : Mapping[str, str] | None, optional
        Environment to inspect instead of the process environment.

    Returns
    -------
    str | None
        The ``<commit12>-<fingerprint12>`` tree name when ``SOURCE_TREE_ID`` is set.

    Raises
    ------
    ValueError
        If ``SOURCE_TREE_ID`` is set but is not a tree name.

    """
    values = os.environ if environment is None else environment
    value = values.get(SOURCE_TREE_ID_ENVIRONMENT_VARIABLE)
    if value is None:
        return None
    if _SOURCE_TREE_ID_PATTERN.fullmatch(value) is None:
        message = f"{SOURCE_TREE_ID_ENVIRONMENT_VARIABLE} must be <commit12>-<fingerprint12> in lowercase hexadecimal, got {value!r}."
        raise ValueError(message)
    return value


def _require_tree_variable(name: str, tree_id: str) -> Path:
    """Return one explicit tree-mode root from the environment or fail with a remedy."""
    value = os.environ.get(name)
    if not value:
        message = (
            f"{name} must be set explicitly because {SOURCE_TREE_ID_ENVIRONMENT_VARIABLE}={tree_id} selects a frozen source tree; "
            "a source tree never infers its storage or runtime roots. Start the process through the Generation launcher, which sets them."
        )
        raise RuntimeError(message)
    return Path(value).expanduser()


def _tree_roots(tree_id: str) -> tuple[Path, Path, Path]:
    """Return the explicit project, storage, and runtime roots of tree mode after checking their relation."""
    project = _require_tree_variable("PROJECT_ROOT", tree_id)
    storage = _require_tree_variable("STORAGE_ROOT", tree_id)
    runtime = _require_tree_variable("RUNTIME_ROOT", tree_id)
    expected = runtime / SOURCE_TREES_DIRNAME / tree_id
    if project != expected:
        message = f"PROJECT_ROOT must equal RUNTIME_ROOT/{SOURCE_TREES_DIRNAME}/{tree_id} in source-tree mode: expected {expected}, got {project}."
        raise RuntimeError(message)
    return project, storage, runtime


def get_project_root() -> Path:
    """
    Return the project root.

    Returns
    -------
    Path
        Root selected by ``PROJECT_ROOT`` or inferred from this module. In
        source-tree mode it is the explicit tree under ``RUNTIME_ROOT/sources``.

    Raises
    ------
    RuntimeError
        In source-tree mode if a root is unset or ``PROJECT_ROOT`` is not the tree.

    """
    tree_id = source_tree_id()
    if tree_id is not None:
        return _tree_roots(tree_id)[0]
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
        portable default is ``<repository-parent>/storage``; source-tree mode has
        no default.

    Returns
    -------
    Path
        Expanded storage-root path without creating it.

    Raises
    ------
    RuntimeError
        In source-tree mode if ``STORAGE_ROOT`` and the other tree roots are not all explicit.

    """
    if storage_root is not None:
        return Path(storage_root).expanduser()
    tree_id = source_tree_id()
    if tree_id is not None:
        return _tree_roots(tree_id)[1]
    root = os.environ.get("STORAGE_ROOT")
    if root:
        return Path(root).expanduser()
    return get_project_root().parent / "storage"


def get_runtime_root(*, runtime_root: Path | str | None = None) -> Path:
    """
    Return the canonical replaceable runtime root.

    Parameters
    ----------
    runtime_root : Path | str | None, optional
        Explicit internal boundary used by controlled callers and tests. When
        omitted, ``RUNTIME_ROOT`` is the sole environment override and the
        portable default is ``<repository-parent>/runtime``; source-tree mode has
        no default.

    Returns
    -------
    Path
        Expanded runtime-root path without creating it.

    Raises
    ------
    RuntimeError
        In source-tree mode if ``RUNTIME_ROOT`` and the other tree roots are not all explicit.

    """
    if runtime_root is not None:
        return Path(runtime_root).expanduser()
    tree_id = source_tree_id()
    if tree_id is not None:
        return _tree_roots(tree_id)[2]
    root = os.environ.get("RUNTIME_ROOT")
    if root:
        return Path(root).expanduser()
    return get_project_root().parent / "runtime"


def get_source_snapshots_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the durable root of exact source snapshots shared by Generation, Datasets and ML."""
    return get_storage_root(storage_root=storage_root) / SOURCE_SNAPSHOTS_DIRNAME


def get_source_trees_root(*, runtime_root: Path | str | None = None) -> Path:
    """Return the runtime root of frozen read-only execution copies restored from source snapshots."""
    return get_runtime_root(runtime_root=runtime_root) / SOURCE_TREES_DIRNAME


def get_generation_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the generated-simulation lifecycle area."""
    return get_storage_root(storage_root=storage_root) / GENERATION_AREA_NAME


def _campaign_batch_parts(batch_storage_name: str) -> tuple[str, str, str, str] | None:
    """Decode only the versioned locator for one campaign-owned material batch."""
    name = validate_logical_name(batch_storage_name, label="batch_storage_name")
    if not name.startswith("source~"):
        return None
    parts = name.split("~")
    if len(parts) != _CAMPAIGN_BATCH_PART_COUNT or parts[0] != "source":
        message = "Malformed canonical campaign batch locator."
        raise ValueError(message)
    task, campaign, material, regime = parts[1:]
    if any(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", part) is None for part in (task, material, regime)):
        message = "Canonical campaign batch task, material, or regime is malformed."
        raise ValueError(message)
    if re.fullmatch(r"[a-z0-9]+(?:[-_+][a-z0-9]+)*(?:__[a-z0-9]+(?:[-_+][a-z0-9]+)*)*", campaign) is None:
        message = "Canonical campaign batch name is malformed."
        raise ValueError(message)
    return task, campaign, material, regime


def campaign_batch_parts(batch_storage_name: str) -> tuple[str, str, str, str] | None:
    """Expose one validated versioned batch locator to Generation admission."""
    return _campaign_batch_parts(batch_storage_name)


def campaign_run_parts(run_id: str) -> tuple[str, str] | None:
    """Decode a schema-v2 internal controller handle without changing its campaign name."""
    run_id = validate_logical_name(run_id, label="campaign_run_id")
    if not run_id.startswith("source~"):
        return None
    parts = run_id.split("~")
    if len(parts) != _CAMPAIGN_RUN_PART_COUNT or parts[0] != "source":
        message = "Malformed canonical campaign controller handle."
        raise ValueError(message)
    task, campaign = parts[1:]
    if (
        re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", task) is None
        or re.fullmatch(r"[a-z0-9]+(?:[-_+][a-z0-9]+)*(?:__[a-z0-9]+(?:[-_+][a-z0-9]+)*)*", campaign) is None
    ):
        message = "Canonical campaign controller task or name is malformed."
        raise ValueError(message)
    return task, campaign


def resolve_generation_campaign_directory(
    task: str,
    campaign_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Locate one Source by task and its explicit physical content name."""
    task = validate_logical_name(task, label="generation_task")
    campaign_name = validate_logical_name(campaign_name, label="campaign_name")
    return get_generation_root(storage_root=storage_root) / task / campaign_name


def generation_execution_key(batch_storage_name: str) -> str:
    """Return the ``<material>-<regime>`` key that names a batch inside ``execution/``.

    The key is injective in (material, regime): neither token may contain a hyphen, so one
    campaign can hold at most one batch per key.
    """
    parts = _campaign_batch_parts(batch_storage_name)
    if parts is None:
        message = "Canonical source batch locator is required."
        raise ValueError(message)
    _task, _campaign, material, regime = parts
    return f"{material}-{regime}"


def resolve_generation_state_directory(campaign_directory: Path | str) -> Path:
    """Return a campaign's private coordination-state directory."""
    return Path(campaign_directory) / GENERATION_STATE_DIRNAME


def resolve_generation_controller_lock_path(campaign_directory: Path | str) -> Path:
    """Return the lock anchor that serializes campaign-plan and reservation writers."""
    return resolve_generation_state_directory(campaign_directory) / GENERATION_CONTROLLER_LOCK_FILENAME


def resolve_generation_submission_lock_path(campaign_directory: Path | str) -> Path:
    """Return the lock anchor that serializes the campaign's scheduler submission and cancellation."""
    return resolve_generation_state_directory(campaign_directory) / GENERATION_SUBMISSION_LOCK_FILENAME


def resolve_generation_campaign_record_directory(campaign_directory: Path | str) -> Path:
    """Return the reserved machine-record directory of a campaign: plan, selected pool, and execution ledger."""
    return Path(campaign_directory) / GENERATION_RECORD_DIRNAME


def admit_generation_campaign_record_directory(campaign_directory: Path | str) -> Path:
    """
    Return a campaign's ``_record/`` once it is admitted as absent or as a real directory directly below the campaign.

    Raises
    ------
    ValueError
        If ``_record`` is a symlink, not a directory, or resolves elsewhere, or if the campaign keeps its plan
        beside ``_record/`` (the superseded layout), so nothing is read from or published into another tree.

    """
    campaign = Path(campaign_directory)
    superseded = campaign / GENERATION_CAMPAIGN_PLAN_FILENAME
    if superseded.exists() or superseded.is_symlink():
        message = f"Campaign uses the superseded storage layout (its plan is not in _record/) and is not readable: {campaign}"
        raise ValueError(message)
    record = resolve_generation_campaign_record_directory(campaign)
    if record.is_symlink() or (record.exists() and (not record.is_dir() or record.resolve() != campaign.resolve() / GENERATION_RECORD_DIRNAME)):
        message = f"Campaign record directory is not a real directory below its campaign: {record}"
        raise ValueError(message)
    return record


def resolve_generation_campaign_plan_path(campaign_directory: Path | str) -> Path:
    """Return the frozen resolved plan of a campaign."""
    return resolve_generation_campaign_record_directory(campaign_directory) / GENERATION_CAMPAIGN_PLAN_FILENAME


def resolve_generation_selected_pool_path(campaign_directory: Path | str) -> Path:
    """Return the sealed selected-success pool of a campaign."""
    return resolve_generation_campaign_record_directory(campaign_directory) / GENERATION_SELECTED_POOL_FILENAME


def resolve_job_logs_directory(owner_directory: Path | str) -> Path:
    """Return the ``logs/`` directory that holds hard links of the job logs of one campaign or published Dataset."""
    return Path(owner_directory) / JOB_LOGS_DIRNAME


def record_job_log_owner(owner: Path | str) -> None:
    """
    Name ``owner`` as a folder whose ``logs/`` receives this ML job's Slurm logs.

    The host wrapper that runs the container reads the file named by ``JOB_LOG_OWNERS_FILE``
    after the job and hard-links the logs there. Without that variable nothing happens, and a
    failed write is one warning on stderr, never an error.
    """
    record_path = os.environ.get(JOB_LOG_OWNERS_ENVIRONMENT_VARIABLE)
    if not record_path:
        return
    try:
        with Path(record_path).open("a", encoding="utf-8") as record:
            record.write(f"{Path(owner).resolve()}\n")
    except (OSError, ValueError) as error:  # ValueError: a path UTF-8 cannot encode
        print(f"Warning: could not record the job log owner {owner}: {error}", file=sys.stderr, flush=True)


def resolve_generation_execution_directory(campaign_directory: Path | str) -> Path:
    """Return the directory that holds every operational record of a campaign."""
    return resolve_generation_campaign_record_directory(campaign_directory) / GENERATION_EXECUTION_DIRNAME


def resolve_generation_run_manifest_path(execution_directory: Path | str) -> Path:
    """Return the scheduler and submission ledger of a campaign."""
    return Path(execution_directory) / GENERATION_RUN_MANIFEST_FILENAME


def resolve_generation_cancellations_path(execution_directory: Path | str) -> Path:
    """Return the cancellation receipt ledger of a campaign."""
    return Path(execution_directory) / GENERATION_CANCELLATIONS_FILENAME


def resolve_generation_epochs_directory(execution_directory: Path | str) -> Path:
    """Return the append-only execution-epoch directory."""
    return Path(execution_directory) / "epochs"


def resolve_generation_slurm_directory(execution_directory: Path | str) -> Path:
    """Return the per-job Slurm progress directory."""
    return Path(execution_directory) / "slurm"


def resolve_generation_interruptions_directory(execution_directory: Path | str) -> Path:
    """Return the per-job worker-interruption receipt directory."""
    return Path(execution_directory) / "interruptions"


def resolve_generation_license_case_directory(execution_directory: Path | str, execution_key: str, case_id: str) -> Path:
    """Return the license-wait, checkout-window and status-recovery directory of one case."""
    return (
        Path(execution_directory)
        / "license"
        / validate_logical_name(execution_key, label="execution_key")
        / validate_logical_name(case_id, label="case_id")
    )


def resolve_generation_reconciliation_case_directory(execution_directory: Path | str, execution_key: str, case_id: str) -> Path:
    """Return the Slurm-accounting reconciliation directory of one case."""
    return (
        Path(execution_directory)
        / "reconciliation"
        / validate_logical_name(execution_key, label="execution_key")
        / validate_logical_name(case_id, label="case_id")
    )


def resolve_generation_reconciliation_failures_directory(execution_directory: Path | str, execution_key: str, case_id: str) -> Path:
    """Return the case-local reconciliation-failure receipt directory of one case."""
    return (
        Path(execution_directory)
        / "reconciliation_failures"
        / validate_logical_name(execution_key, label="execution_key")
        / validate_logical_name(case_id, label="case_id")
    )


def resolve_generation_material_directory(
    task: str,
    campaign_name: str,
    material: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Locate the directory that owns one material's designs and cases inside a campaign."""
    material = validate_logical_name(material, label="material")
    if material in _GENERATION_CAMPAIGN_RESERVED_NAMES:
        message = f"Material name {material!r} is a reserved campaign entry."
        raise ValueError(message)
    return resolve_generation_campaign_directory(task, campaign_name, storage_root=storage_root) / material


def resolve_generation_campaign_case_directory(
    task: str,
    campaign_name: str,
    material: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Locate one append-only material-local candidate inside a campaign."""
    case_id = validate_logical_name(case_id, label="case_id")
    return resolve_generation_material_directory(task, campaign_name, material, storage_root=storage_root) / case_id


def generation_material_case_directories(material_directory: Path | str) -> tuple[Path, ...]:
    """
    List the case directories of one material in case-number order.

    The reserved ``_record/`` is the only other admitted entry; it must be a real
    directory. Any other entry, a symlink, or a file fails closed.

    Raises
    ------
    ValueError
        If the namespace holds an unsafe ``_record`` or an entry that is not a case directory.

    """
    directory = Path(material_directory)
    cases: list[tuple[int, Path]] = []
    for entry in directory.iterdir():
        if entry.name == GENERATION_RECORD_DIRNAME and not entry.is_symlink() and entry.is_dir():
            continue
        if _GENERATION_CASE_ID.fullmatch(entry.name) is None or entry.is_symlink() or not entry.is_dir():
            message = f"Material namespace contains an entry that is not a case directory or its real {GENERATION_RECORD_DIRNAME}/: {entry}"
            raise ValueError(message)
        cases.append((int(entry.name.removeprefix("case_")), entry))
    return tuple(entry for _number, entry in sorted(cases))


def _campaign_batch_material_directory(batch_storage_name: str, *, storage_root: Path | str | None) -> tuple[Path, str]:
    """Return the material directory and regime of one canonical batch locator."""
    parts = _campaign_batch_parts(batch_storage_name)
    if parts is None:
        message = "Canonical source batch locator is required."
        raise ValueError(message)
    task, campaign, material, regime = parts
    return resolve_generation_material_directory(task, campaign, material, storage_root=storage_root), regime


def resolve_generation_batch_metadata_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one canonical batch-owned design directory ``<material>/_record/<regime>``."""
    material_directory, regime = _campaign_batch_material_directory(batch_storage_name, storage_root=storage_root)
    return material_directory / GENERATION_RECORD_DIRNAME / regime


def resolve_generation_replacements_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the append-only replacement-capacity amendments of one original batch."""
    return resolve_generation_batch_metadata_directory(batch_storage_name, storage_root=storage_root) / GENERATION_REPLACEMENTS_DIRNAME


def resolve_generation_cases_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the material directory, the case namespace that holds one canonical batch's cases beside its ``_record/``."""
    material_directory, _regime = _campaign_batch_material_directory(batch_storage_name, storage_root=storage_root)
    return material_directory


def resolve_generation_input_generation_cases_directory(
    batch_storage_name: str,
    input_generation_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the case namespace that holds the inputs of one exact canonical input generation."""
    validate_logical_name(input_generation_id, label="input_generation_id")
    return resolve_generation_cases_directory(batch_storage_name, storage_root=storage_root)


def resolve_generation_case_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one canonical case directory of a batch."""
    return resolve_generation_cases_directory(batch_storage_name, storage_root=storage_root) / validate_logical_name(case_id, label="case_id")


def resolve_generation_case_record_directory(case_directory: Path | str) -> Path:
    """Return the reserved machine-record directory of a case: reservation, decision, and attempts."""
    return Path(case_directory) / GENERATION_RECORD_DIRNAME


def resolve_generation_case_reservation_path(case_directory: Path | str) -> Path:
    """Return the sole reservation record of a case."""
    return resolve_generation_case_record_directory(case_directory) / GENERATION_RESERVATION_FILENAME


def resolve_generation_case_decision_path(case_directory: Path | str) -> Path:
    """Return the terminal decision record of a case."""
    return resolve_generation_case_record_directory(case_directory) / GENERATION_DECISION_FILENAME


def resolve_generation_case_inputs_directory(case_directory: Path | str) -> Path:
    """Return the immutable input directory of a case: payload, adapters, and input reference."""
    return Path(case_directory) / GENERATION_INPUTS_DIRNAME


def resolve_generation_case_result_directory(case_directory: Path | str) -> Path:
    """Return the published result of a case: HDF5, provenance inventory, and success marker."""
    return Path(case_directory) / GENERATION_RESULT_DIRNAME


def resolve_generation_result_evidence_directory(result_directory: Path | str) -> Path:
    """Return the retained runtime evidence of a published result."""
    return Path(result_directory) / GENERATION_EVIDENCE_DIRNAME


def resolve_generation_case_attempts_directory(case_directory: Path | str) -> Path:
    """Return the directory of classified attempts and retained failure bundles of a case."""
    return resolve_generation_case_record_directory(case_directory) / GENERATION_ATTEMPTS_DIRNAME


def _attempt_number(attempt_index: int) -> str:
    """Return one zero-padded attempt number."""
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        message = f"attempt_index must be a positive integer, got {attempt_index!r}."
        raise ValueError(message)
    return f"{attempt_index:04d}"


def resolve_generation_case_attempt_record_path(case_directory: Path | str, attempt_index: int) -> Path:
    """Return one classified non-terminal attempt record of a case."""
    return resolve_generation_case_attempts_directory(case_directory) / f"attempt_{_attempt_number(attempt_index)}.json"


def resolve_generation_case_attempt_run_directory(case_directory: Path | str, attempt_index: int) -> Path:
    """Return the retained evidence bundle of one failed execution of a case."""
    return resolve_generation_case_attempts_directory(case_directory) / f"run_{_attempt_number(attempt_index)}"


def resolve_generation_inputs_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the input directory of one canonical case."""
    return resolve_generation_case_inputs_directory(resolve_generation_case_directory(batch_storage_name, case_id, storage_root=storage_root))


def resolve_generation_result_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the result directory of one canonical case."""
    return resolve_generation_case_result_directory(resolve_generation_case_directory(batch_storage_name, case_id, storage_root=storage_root))


def resolve_generation_attempts_directory(
    batch_storage_name: str,
    case_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve the attempts directory of one canonical case."""
    return resolve_generation_case_attempts_directory(resolve_generation_case_directory(batch_storage_name, case_id, storage_root=storage_root))


def resolve_generation_attempt_run_directory(
    batch_storage_name: str,
    case_id: str,
    attempt_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one failure bundle of one canonical case."""
    return resolve_generation_case_attempt_run_directory(
        resolve_generation_case_directory(batch_storage_name, case_id, storage_root=storage_root),
        attempt_index,
    )


def resolve_generation_reservation_staging_directory(campaign_directory: Path | str, material: str) -> Path:
    """Return the private area where a case directory is prepared before its atomic publication."""
    return (
        resolve_generation_state_directory(campaign_directory)
        / validate_logical_name(material, label="material")
        / GENERATION_RESERVATION_STAGING_DIRNAME
    )


@dataclass(frozen=True, slots=True)
class GenerationCaseLocation:
    """Task, campaign, material, and case of one canonical case directory."""

    storage_root: Path
    task: str
    campaign_name: str
    material: str
    case_id: str


def _case_location_parts(parts: tuple[str, ...]) -> tuple[str, str, str, str] | None:
    """Return task, campaign, material, and case of ``01_generation/<task>/<campaign>/<material>/<case>``, else ``None``."""
    if len(parts) != _CASE_DIRECTORY_DEPTH or parts[0] != GENERATION_AREA_NAME:
        return None
    _area, task, campaign_name, material, case_id = parts
    if material in _GENERATION_CAMPAIGN_RESERVED_NAMES or _GENERATION_CASE_ID.fullmatch(case_id) is None:
        return None
    for label, value in (("generation_task", task), ("campaign_name", campaign_name), ("material", material), ("case_id", case_id)):
        validate_logical_name(value, label=label)
    return task, campaign_name, material, case_id


def locate_generation_case_directory(case_directory: Path | str) -> GenerationCaseLocation:
    """Invert ``<storage>/01_generation/<task>/<campaign>/<material>/<case>`` to its parts."""
    path = Path(case_directory)
    parts = path.parts
    located = _case_location_parts(parts[-_CASE_DIRECTORY_DEPTH:])
    if located is None:
        message = f"Path is not a canonical Generation case directory: {path}"
        raise ValueError(message)
    return GenerationCaseLocation(Path(*parts[:-_CASE_DIRECTORY_DEPTH]) if len(parts) > _CASE_DIRECTORY_DEPTH else Path(), *located)


def locate_generation_case_result_file(relative_path: str, filename: str) -> GenerationCaseLocation:
    """Invert a storage-relative ``01_generation/<task>/<campaign>/<material>/<case>/result/<filename>`` locator to its case parts.

    The returned ``storage_root`` is empty because the locator is storage-relative.
    """
    path = Path(relative_path)
    located = None if path.is_absolute() or path.parts[-2:] != (GENERATION_RESULT_DIRNAME, filename) else _case_location_parts(path.parts[:-2])
    if located is None:
        message = f"Locator is not a canonical Generation case result file: {relative_path!r}"
        raise ValueError(message)
    return GenerationCaseLocation(Path(), *located)


def locate_generation_attempt_case_directory(attempt_run_directory: Path | str) -> Path:
    """Invert one retained failure bundle ``<case>/_record/attempts/run_NNNN`` to its case directory."""
    path = Path(attempt_run_directory)
    case_directory = path.parent.parent.parent
    if _GENERATION_ATTEMPT_RUN.fullmatch(path.name) is None or resolve_generation_case_attempts_directory(case_directory) != path.parent:
        message = f"Path is not a canonical Generation attempt bundle: {path}"
        raise ValueError(message)
    return case_directory


def resolve_generation_input_metadata_storage_root(directory: Path | str, batch_storage_name: str, input_generation_id: str) -> Path:
    """Invert the exact canonical or staged-transaction design location of an input generation to its storage root."""
    generation_id = validate_logical_name(input_generation_id, label="input_generation_id")
    batch = validate_logical_name(batch_storage_name, label="batch_storage_name")
    suffixes = [
        Path(GENERATION_AREA_NAME)
        / GENERATION_STATE_DIRNAME
        / "raw-inputs"
        / "transactions"
        / generation_id
        / "meta"
        / batch
        / "input_generations"
        / generation_id
    ]
    parts = _campaign_batch_parts(batch)
    if parts is not None:
        task, campaign, material, regime = parts
        suffixes.append(Path(GENERATION_AREA_NAME) / task / campaign / material / GENERATION_RECORD_DIRNAME / regime)
    for suffix in suffixes:
        try:
            return _physical_layout_root(directory, suffix)
        except ValueError:
            continue
    message = "Input-generation metadata directory must use its exact current generation identity."
    raise ValueError(message)


def generation_design_directories(task: str, *, storage_root: Path | str | None = None) -> tuple[Path, ...]:
    """List every ``<campaign>/<material>/_record/<regime>`` directory of a task in a deterministic order.

    These are discovery candidates only; the reserved campaign entries (``.state``,
    ``_record``, ``logs``, ``cases.csv``) are never materials, and admission validates each candidate.
    """
    task_root = get_generation_root(storage_root=storage_root) / validate_logical_name(task, label="generation_task")
    candidates = task_root.glob(f"*/*/{GENERATION_RECORD_DIRNAME}/*")
    return tuple(sorted(path for path in candidates if path.parent.parent.name not in _GENERATION_CAMPAIGN_RESERVED_NAMES))


def get_datasets_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the immutable final-dataset lifecycle area."""
    return get_storage_root(storage_root=storage_root) / DATASETS_AREA_NAME


def storage_root_for_datasets_root(datasets_root: Path | str) -> Path:
    """Return the storage root whose Dataset lifecycle area is ``datasets_root``."""
    resolved = Path(datasets_root).expanduser().resolve()
    if resolved.name != DATASETS_AREA_NAME:
        msg = f"Dataset root is not the {DATASETS_AREA_NAME!r} area of a storage root: {resolved}"
        raise ValueError(msg)
    return resolved.parent


def get_experiments_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the run, study, log, and analysis lifecycle area."""
    return get_storage_root(storage_root=storage_root) / EXPERIMENTS_AREA_NAME


def get_generation_state_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the transient generation-coordination state root."""
    return get_generation_root(storage_root=storage_root) / GENERATION_STATE_DIRNAME


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
    """Return the dedicated Generation performance evidence directory."""
    return get_generation_root(storage_root=storage_root) / "performance_benchmarks"


def get_dataset_state_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the hidden root for dataset-publication coordination state."""
    return get_datasets_root(storage_root=storage_root) / ".state"


def get_experiment_state_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the hidden root for experiment coordination state."""
    return get_experiments_root(storage_root=storage_root) / ".state"


def get_dataset_build_locks_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the persistent OS-lock-anchor root for dataset publication."""
    return get_dataset_state_root(storage_root=storage_root) / "dataset-builds" / "locks"


def get_run_locks_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the persistent OS-lock-anchor root for saved-run writers."""
    return get_experiment_state_root(storage_root=storage_root) / "runs" / "locks"


def get_transient_scaler_cache_root(*, storage_root: Path | str | None = None) -> Path:
    """Return the integrity-bound transient Train-scaler cache root."""
    return get_experiment_state_root(storage_root=storage_root) / "transient_scalers"


def resolve_identity_registry_entry(namespace_root: Path | str, digest: str) -> Path:
    """Resolve the registry entry that binds one content identity below a namespace root (runs or Datasets)."""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        message = "digest must be one lowercase SHA-256 digest."
        raise ValueError(message)
    return Path(namespace_root).expanduser() / ".state" / "identities" / f"{digest}.json"


def resolve_identity_registry_lock(namespace_root: Path | str, digest: str) -> Path:
    """Resolve the persistent lock anchor that serializes one registry entry."""
    entry = resolve_identity_registry_entry(namespace_root, digest)
    return entry.parent / "locks" / f"{digest}.lock"


def resolve_generation_input_generation_metadata_directory(
    batch_storage_name: str,
    input_generation_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve metadata for one exact canonical input generation."""
    batch = validate_logical_name(batch_storage_name, label="batch_storage_name")
    validate_logical_name(input_generation_id, label="input_generation_id")
    return resolve_generation_batch_metadata_directory(batch, storage_root=storage_root)


def resolve_generation_state_batch_directory(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve flat private runtime coordination state for one generation batch."""
    parts = _campaign_batch_parts(batch_storage_name)
    if parts is None:
        message = "Canonical source batch locator is required."
        raise ValueError(message)
    task, campaign, material, regime = parts
    return resolve_generation_state_directory(resolve_generation_campaign_directory(task, campaign, storage_root=storage_root)) / material / regime


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


def dataset_state_key(task: str, dataset_id: str) -> str:
    """Return a bounded file-name key for one Dataset's coordination state.

    Dataset names may use almost the whole 255-byte component limit, so locks,
    transactions and staging directories use this digest instead of the name.
    """
    task = validate_logical_name(task, label="task")
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    return hashlib.sha256(f"{task}/{dataset_id}".encode()).hexdigest()


def resolve_dataset_build_lock_path(
    task: str,
    dataset_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Resolve one dataset builder's persistent advisory-lock anchor, qualified by task."""
    task = validate_logical_name(task, label="task")
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    return get_dataset_build_locks_root(storage_root=storage_root) / f"dataset-{dataset_state_key(task, dataset_id)}.lock"


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
        Candidate identifier. It must be a non-empty, already-trimmed string of
        at most 255 UTF-8 bytes containing no separator, NUL, absolute path,
        ``.`` or ``..`` value.
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
    if len(value.encode("utf-8")) > _MAX_COMPONENT_BYTES:
        msg = f"{label} must not exceed {_MAX_COMPONENT_BYTES} UTF-8 bytes as a path component, got {len(value.encode('utf-8'))} bytes."
        raise ValueError(msg)
    return value


def resolve_dataset_dir(task: str, dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one logical dataset directory.

    Parameters
    ----------
    task : str
        Learning task that owns the Dataset (``steady_flow`` or ``transient_drying``).
    dataset_id : str
        Non-empty logical dataset identifier.
    dataset_root : Path | str | None, optional
        Explicit dataset root. Output roots are never consulted.

    Returns
    -------
    Path
        ``<dataset_root>/<task>/<dataset_id>``.

    Raises
    ------
    ValueError
        If ``task`` or ``dataset_id`` is not one safe logical path component.

    """
    task = validate_logical_name(task, label="task")
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    root = Path(dataset_root).expanduser() if dataset_root is not None else get_datasets_root()
    return root / task / dataset_id


def resolve_dataset_metadata_dir(
    task: str,
    dataset_id: str,
    *,
    metadata_root: Path | str | None = None,
) -> Path:
    """
    Resolve one dataset's validated metadata snapshot directory.

    The directory owns the manifest, ordered source selection and scientific payload
    for this exact Dataset identity.
    """
    return resolve_dataset_dir(task, dataset_id, dataset_root=metadata_root)


def resolve_dataset_manifest_path(task: str, dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """Resolve ``<dataset_root>/<task>/<dataset_id>/dataset_manifest.json``."""
    return resolve_dataset_dir(task, dataset_id, dataset_root=dataset_root) / DATASET_MANIFEST_FILENAME


def get_dataset_payload_filename(task: str) -> str:
    """Return the fixed payload file name of a task's Dataset: the tensor payload or the transient index."""
    try:
        return _DATASET_PAYLOAD_FILENAMES[task]
    except (KeyError, TypeError) as error:
        message = f"Datasets exist for the tasks {sorted(_DATASET_PAYLOAD_FILENAMES)}, got {task!r}."
        raise ValueError(message) from error


def resolve_dataset_path(task: str, dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one Dataset's payload file.

    Parameters
    ----------
    task : str
        Learning task that owns the Dataset.
    dataset_id : str
        Non-empty logical dataset identifier.
    dataset_root : Path | str | None, optional
        Explicit dataset root. Output roots are never consulted.

    Returns
    -------
    Path
        ``<dataset_root>/<task>/<dataset_id>/dataset.pt`` for steady flow and
        ``.../dataset.json`` (the transition index) for transient drying.

    Raises
    ------
    ValueError
        If ``task`` has no Dataset payload or ``dataset_id`` is not one safe logical path component.

    """
    filename = get_dataset_payload_filename(task)
    return resolve_dataset_dir(task, dataset_id, dataset_root=dataset_root) / filename


def _require_epochs(epochs: object, *, label: str) -> int:
    """Require the positive integer epoch budget that names a seed group or study directory."""
    if type(epochs) is not int or epochs < 1:
        message = f"{label} must be a positive integer, got {epochs!r}."
        raise ValueError(message)
    return epochs


def seed_group_name(seed: int, revision: str | None = None, *, a0_epochs: int | None = None) -> str:
    """Return the seed-group directory name, which is also the run-name seed segment.

    The result is ``seed-<n>[__a0ep-<epochs>][__rev-<revision>]``. A transient seed
    group passes its A0 epoch budget because that budget is part of the A0's
    identity; steady runs pass none, as their epochs extend a run in place.
    """
    name = f"seed-{seed}"
    if a0_epochs is not None:
        name = f"{name}__a0ep-{_require_epochs(a0_epochs, label='a0_epochs')}"
    return name if revision is None else f"{name}__rev-{revision}"


def study_group_name(study: str, epochs: int, revision: str | None = None) -> str:
    """Return the study directory name ``<study>__ep-<epochs>[__rev-<revision>]``."""
    name = f"{study}__ep-{_require_epochs(epochs, label='study epochs')}"
    return name if revision is None else f"{name}__rev-{revision}"


def _selected_output_root(config: Mapping[str, Any], output_root: Path | str | None) -> Path:
    """Select the explicit experiments root, else the resolved config's own root."""
    paths = config.get("paths")
    configured_root = paths.get("output_root") if isinstance(paths, Mapping) else None
    selected_root = output_root if output_root is not None else configured_root
    if selected_root is None:
        message = "Resolved config.paths.output_root is required for Experiment paths."
        raise ValueError(message)
    return Path(selected_root).expanduser()


def _config_layout(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require self-contained lexical layout metadata without model execution."""
    run = config.get("run")
    layout = run.get("layout") if isinstance(run, Mapping) else None
    if not isinstance(layout, Mapping) or set(layout) != _LAYOUT_KEYS:
        message = f"run.layout must contain exactly the layout keys {sorted(_LAYOUT_KEYS)}."
        raise ValueError(message)
    for key in ("task", "dataset", "family", "config_name", "model_label"):
        validate_logical_name(layout[key], label=f"run.layout.{key}")
    for key in ("revision", "pair", "trial"):
        if layout[key] is not None:
            validate_logical_name(layout[key], label=f"run.layout.{key}")
    if type(layout["seed"]) is not int or layout["seed"] < 0:
        message = "run.layout.seed must be a non-negative integer."
        raise ValueError(message)
    if layout["budget_epochs"] is not None:
        _require_epochs(layout["budget_epochs"], label="run.layout.budget_epochs")
    for key in ("a0_id", "branch_id"):
        value = layout[key]
        if key == "branch_id" and value is None:
            continue
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
            message = f"run.layout.{key} must be a lowercase 16-character content identity."
            raise ValueError(message)
    if layout["stage"] not in _LAYOUT_STAGES:
        message = "run.layout.stage must be a0, b, a1 or single."
        raise ValueError(message)
    paired = layout["stage"] in _PAIRED_STAGES
    if paired != (layout["branch_id"] is not None) or paired != (layout["pair"] is not None):
        message = "Only B and A1 carry a pair and a branch identity, and both are required for them."
        raise ValueError(message)
    is_trial = layout["family"] == OPTUNA_FAMILY
    if is_trial != (layout["trial"] is not None) or (is_trial and layout["stage"] != "single"):
        message = f"Only single-stage runs of family {OPTUNA_FAMILY!r} carry a trial, and they require one."
        raise ValueError(message)
    if is_trial and layout["budget_epochs"] is None:
        message = "An Optuna trial layout requires run.layout.budget_epochs, the epoch budget that names its study directory."
        raise ValueError(message)
    return layout


def resolve_config_family_dir(
    config: Mapping[str, Any],
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve the ``<task>/<dataset>/<family>`` directory shared by comparable configs."""
    layout = _config_layout(config)
    return _selected_output_root(config, output_root) / layout["task"] / layout["dataset"] / layout["family"]


def resolve_config_experiment_dir(
    config: Mapping[str, Any],
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve a config's seed directory, or a study trial's own directory, without creating it.

    The result is a pure function of the saved layout: it depends on neither the
    stage nor on which runs already exist. Steady runs live directly in it.
    """
    layout = _config_layout(config)
    root = _selected_output_root(config, output_root)
    if layout["trial"] is not None:
        study = resolve_study_dir(
            layout["task"], layout["dataset"], layout["config_name"], layout["budget_epochs"], layout["revision"], output_root=root
        )
        return study / STUDY_TRIALS_DIRNAME / layout["trial"]
    family = resolve_config_family_dir(config, output_root=root)
    return family / layout["config_name"] / seed_group_name(layout["seed"], layout["revision"], a0_epochs=layout["budget_epochs"])


def resolve_config_branch_dir(
    config: Mapping[str, Any],
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve the directory holding a B/A1 pair's record and stages, else the experiment directory.

    This pure metadata resolver never discovers anything, creates directories,
    loads neural networks or infers compatibility from a human label.
    """
    layout = _config_layout(config)
    experiment = resolve_config_experiment_dir(config, output_root=output_root)
    return experiment / layout["pair"] if layout["pair"] is not None else experiment


def resolve_config_run_dir(
    config: Mapping[str, Any],
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve the run directory of a single, a0, b or a1 stage from its saved layout."""
    layout = _config_layout(config)
    stage = layout["stage"]
    if stage == "single":
        return resolve_config_experiment_dir(config, output_root=output_root)
    if stage == "a0":
        return resolve_config_experiment_dir(config, output_root=output_root) / "a0"
    return resolve_config_branch_dir(config, output_root=output_root) / stage


def _physical_layout_root(actual_path: Path | str, suffix: Path) -> Path:
    """Invert an exact declared layout suffix without fixed-depth navigation."""
    original = Path(actual_path).expanduser().absolute()
    actual = original.resolve(strict=False)
    if len(actual.parts) <= len(suffix.parts) or actual.parts[-len(suffix.parts) :] != suffix.parts:
        msg = f"Physical result path contradicts its declared layout: {original}"
        raise ValueError(msg)
    root = Path(*actual.parts[: -len(suffix.parts)])
    current = original
    while current not in {current.parent, root}:
        if current.is_symlink():
            msg = f"Scientific layout contains a symlink: {current}"
            raise ValueError(msg)
        current = current.parent
    return root


def resolve_config_output_root(config: Mapping[str, Any], actual_run_dir: Path | str) -> Path:
    """Recover the actual output root from saved layout and physical run location.

    Stored environment paths remain provenance. The exact relative layout,
    rather than a historical mount path or fixed parent depth, binds location.
    This operation reads no model, checkpoint, tensor or Dataset payload.
    """
    suffix = resolve_config_run_dir(config, output_root=Path())
    return _physical_layout_root(actual_run_dir, suffix)


def resolve_experiment_record_root(
    marker: Path | str,
    layout: Mapping[str, Any],
    *,
    experiments_root: Path | str | None = None,
) -> Path:
    """Recover a parent marker's root and optionally bind it to a discovery root.

    The marker must lie inside the explicitly selected catalog root when one is
    given; equal-looking paths outside it fail.
    """
    config = {"run": {"layout": layout}}
    suffix = resolve_config_branch_dir(config, output_root=Path()) / EXPERIMENT_RECORD_FILENAME
    root = _physical_layout_root(marker, suffix)
    if experiments_root is not None and not root.is_relative_to(Path(experiments_root).expanduser().resolve(strict=False)):
        msg = "Experiment parent is outside the selected discovery root."
        raise ValueError(msg)
    return root


def resolve_study_dir(
    task: str,
    dataset: str,
    study: str,
    epochs: int,
    revision: str | None = None,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve one Optuna study directory ``<task>/<dataset>/optuna/<study>__ep-<epochs>[__rev-<revision>]``.

    ``epochs`` is the trial epoch budget: another budget is another study, so it names the directory.
    """
    task = validate_logical_name(task, label="task")
    dataset = validate_logical_name(dataset, label="dataset")
    study = validate_logical_name(study, label="study_name")
    if revision is not None:
        revision = validate_logical_name(revision, label="revision")
    root = Path(output_root).expanduser() if output_root is not None else get_experiments_root()
    return root / task / dataset / OPTUNA_FAMILY / validate_logical_name(study_group_name(study, epochs, revision), label="study directory")


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
    Return files required before a run can continue from its last checkpoint.

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
    Return whether a directory is a completed run with the shallow evaluation contract.

    Only ``completed`` runs are evaluable; failed, interrupted and running ones
    are not. Full lifecycle, configuration, saved-data, checkpoint, digest, and writer
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
    return isinstance(schema_version, int) and not isinstance(schema_version, bool) and schema_version == 1 and status == "completed"


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


def resolve_additional_evaluation_analysis_dir(run_dir: Path | str, dataset_name: str) -> Path:
    """
    Resolve the additional evaluation artifact directory for a run and dataset.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.
    dataset_name : str
        Logical (possibly combined) additional evaluation dataset name.

    Returns
    -------
    Path
        Path to ``analysis/additional_evaluation/<dataset_name>``.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not one safe logical path component.

    """
    dataset_name = validate_logical_name(dataset_name, label="dataset_name")
    return resolve_analysis_root(run_dir) / "additional_evaluation" / dataset_name
