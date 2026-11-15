"""Allocate, initialize, execute, resume, and validate saved experiment runs.

Responsibilities:
  - Allocate fresh run leaves exclusively and transition explicit statuses
  - Derive stable labeled seeds and configure deterministic execution early
  - Persist immutable run inputs and mutable lifecycle outputs atomically
  - Record every attempt in runtime_sessions: device, driver, precision, and at its end its seconds and committed work
  - Derive the run's total and attempted seconds from its attempts and refuse to continue an earlier attempt schema
  - Enforce allowed resume config changes and exact last-checkpoint continuation
  - Refuse continuing or reusing a run, and handing an A0 to B or A1, when its recorded training code is not current
  - Continue an interrupted run when a start names it, its full saved identity matches and its source binding admits the job
  - Validate the completed/loadable contract consumed by inference and artifacts
  - Free a finished Optuna trial's checkpoints with its terminal summary and validate such a trial without them
  - Merge caller-owned terminal status and evidence into the one terminal summary of every run and trial

Design principles:
  - A start that names an existing run continues it only when its saved science matches; anything else is refused
  - A finished run is never extended: budget, k and schedule belong to its identity and name
  - KeyboardInterrupt and SystemExit end as interrupted; a failure to record a failure never replaces the primary error
  - A run last executed by a Slurm job in one source tree is continued only in that tree unless the job switched explicitly
  - Config, split, and normalizer artifacts are immutable after fresh creation
  - Best and last checkpoints have distinct enforced lifecycle roles and config-derived payload kinds
  - Generic orchestration consumes task/config interfaces without field names

This module does NOT:
  - Define task physics, model architectures, loss formulas, or metric mathematics
  - Continue a run whose saved science differs, or treat observers as authoritative state
  - Render or publish post-training analysis artifacts
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from src import common, datasets, domain, learning
from src.common.common_random import derive_subseed
from src.learning.transient import learning_transient_adapter as transient_adapter
from src.learning.transient import learning_transient_curriculum as transient_curriculum
from src.learning.transient import learning_transient_handoff as transient_handoff
from src.learning.transient import learning_transient_history as transient_adapter_history
from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec

from . import experiments_console as console
from . import experiments_matched_budget as matched_budget
from . import experiments_run_identity as run_identity
from . import experiments_run_registry as run_registry
from . import experiments_tracking as tracking
from .config import experiments_config_loader as config_loader
from .config import experiments_config_naming as config_naming
from .config import experiments_config_transient_plan as transient_plan

RUN_DURATION_CONTRACT: dict[str, Any] = {
    "clock": "cumulative_wall_seconds",
    "cumulative_across_resume": True,
    "includes": [
        "prepared_run_local_admission",
        "dataset_loading",
        "normalizer_fit_or_restore",
        "model_and_optimizer_construction",
        "wandb_initialization",
        "training",
        "scheduled_validation_and_optional_diagnostic_evaluation",
        "scheduled_physics_monitor",
        "checkpoint_publication",
        "selected_checkpoint_evaluation",
    ],
    "excludes": ["imports", "caller_preflight_and_run_allocation", "scheduler_queue", "wandb_finalization", "terminal_summary_publication"],
    "coverage": "elapsed_seconds and attempted_train_seconds are null while an attempt in runtime_sessions has no end entry",
    "clock_source": "time.monotonic",
}
# The end entry of one attempt (runtime_sessions[].attempt); its start entry holds the same keys, all null.
_ATTEMPT_KEYS = frozenset({"ended_at", "elapsed_seconds", "train_seconds", "work_at_start", "work_at_end"})
RUN_STATUSES = frozenset(
    {
        "initializing",
        "running",
        "completed",
        "pruned",
        "nonfinite_pruned",
        "oom_pruned",
        "recoverable_failed",
        "failed",
        "interrupted",
        "unsuccessful",
    }
)
_TERMINAL_RUN_STATUSES = RUN_STATUSES.difference({"initializing", "running"})
# A failure may end a run before or after it starts running; only training reaches completed or unsuccessful.
_FAILURE_RUN_STATUSES = _TERMINAL_RUN_STATUSES.difference({"completed", "unsuccessful"})
_PRUNED_RUN_STATUSES = frozenset({"pruned", "nonfinite_pruned", "oom_pruned"})
_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"initializing"}),
    "initializing": frozenset({"running", *_FAILURE_RUN_STATUSES}),
    "running": frozenset({"running", *_TERMINAL_RUN_STATUSES}),
    "interrupted": frozenset({"running", "failed"}),
    # A finished run is never extended: another budget is another run.
    "completed": frozenset(),
    # Training ended without an eligible checkpoint; repeating it would repeat the same outcome.
    "unsuccessful": frozenset(),
    "failed": frozenset(),
    "pruned": frozenset(),
    "nonfinite_pruned": frozenset(),
    "oom_pruned": frozenset(),
    "recoverable_failed": frozenset(),
}
# Every run's component seeds; only Stage B draws rollout horizons, so only B's plan adds the curriculum seed.
_SEED_LABELS = ("process", "model_init", "loader", "worker")
_SELF_FED_STAGE = "b"
_CURRICULUM_SEED_LABEL = "curriculum_b"
# "explicit" marks a session that deliberately continues a unit with newer code than its bound source tree.
_SOURCE_TRANSITIONS = frozenset({"explicit", "none"})
_MISSING = object()
_MAX_CONFIG_DIFFERENCES = 12
_TRAINING_CONTINUATION_REMEDY = "finish it in its bound source tree or start a new run with --revision <label>"
# Caller facts merged into a terminal summary: called with the result before the completed transition (a raise
# fails the run) and with the error on the guarded failure path, always before a trial's best checkpoint goes.
TerminalEvidence = Callable[[str, Mapping[str, Any] | BaseException], Mapping[str, Any]]


class RunLifecycleError(RuntimeError):
    """
    Represent a saved-run lifecycle or identity contract violation.

    Raised at allocation, transition, resume, and completed-run admission
    boundaries. It distinguishes invalid persisted run state from configuration
    schema errors and ordinary missing-file failures.
    """


class SourceBindingError(RunLifecycleError):
    """Report an implicit switch of a run or study to another frozen source tree."""


class ExistingRunAdmissionError(FileExistsError):
    """Carry a read-only fail-closed report for a rejected fresh-run collision."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        """Initialize the rejection with an isolated diagnostic report."""
        super().__init__(str(report.get("reason", "Run admission rejected: the existing run cannot continue with this request.")))
        self.report = copy.deepcopy(dict(report))


class MatchedStageExecutionError(RuntimeError):
    """Keep the failed matched leaf and read-only recovery evidence with its cause."""

    def __init__(self, stage: str, run_dir: Path, *, resumable: bool, reason: str) -> None:
        """Capture the exact failed branch and verified recovery classification."""
        super().__init__(f"Matched stage {stage} failed at {run_dir}.")
        self.stage = stage
        self.run_dir = run_dir
        self.resumable = resumable
        self.recovery_reason = reason


class MatchedStageInterrupted(KeyboardInterrupt):
    """Preserve exit-130 interrupt semantics with an exact matched leaf."""

    def __init__(self, stage: str, run_dir: Path, *, resumable: bool, reason: str) -> None:
        """Capture the interrupted branch and verified recovery classification."""
        super().__init__(f"Matched stage {stage} interrupted at {run_dir}.")
        self.stage = stage
        self.run_dir = run_dir
        self.resumable = resumable
        self.recovery_reason = reason


def _run_writer_lock_path(run_dir: Path | str) -> Path:
    """Return the centralized persistent lock path for one canonical run leaf."""
    return common.paths.resolve_run_lock_path(run_dir)


@contextmanager
def run_writer_lease(
    run_dir: Path | str,
    *,
    blocking: bool = False,
) -> Iterator[Path]:
    """
    Hold the exclusive writer lease for one run lifecycle.

    The lock file lives below ``STORAGE_ROOT/03_experiments/.state/runs/locks`` so
    fresh allocation and resume prevalidation can use the same lease before touching run contents.
    Training fails fast by default. Coordinated artifact readers may wait for
    the current run writer by setting ``blocking=True``.

    Parameters
    ----------
    run_dir : Path | str
        Canonical run leaf, which need not exist yet for fresh allocation.
    blocking : bool, optional
        Wait for the current owner when true. Otherwise fail immediately.

    Yields
    ------
    pathlib.Path
        Expanded absolute run path protected by the lease.

    Raises
    ------
    RunLifecycleError
        If a non-blocking lease is already owned by another writer.

    Notes
    -----
    The persistent state-root anchor is not run content and is not removed when
    the lease closes. The underlying OS lock, rather than file existence, owns exclusion.

    """
    path = Path(run_dir).expanduser().resolve(strict=False)
    try:
        with common.locking.exclusive_file_lock(
            _run_writer_lock_path(path),
            blocking=blocking,
        ):
            yield path
    except common.locking.FileLockUnavailableError as error:
        msg = f"Run already has an active writer lease: {path}"
        raise RunLifecycleError(msg) from error


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def build_seed_plan(seed: int, *, stage: str | None) -> dict[str, int]:
    """
    Return the stable labeled seed plan of one run seed and its transient stage (``None`` for steady runs).

    Derivation is order-independent and covers process, model initialization,
    loader order and workers; Stage B, the only stage that draws rollout horizons,
    adds its curriculum seed. A cryptographic-prefix collision among the labels
    raises ``RuntimeError``.
    """
    labels = (*_SEED_LABELS, *((_CURRICULUM_SEED_LABEL,) if stage == _SELF_FED_STAGE else ()))
    plan = {label: derive_subseed(seed, label) for label in labels}
    if len(set(plan.values())) != len(plan):
        msg = "Stable labeled seed derivation produced a collision."
        raise RuntimeError(msg)
    return plan


def seed_process(seed: int, *, device: torch.device) -> None:
    """
    Seed process-global RNGs for one already resolved concrete device.

    Python, NumPy process-global, and Torch CPU state are always mutated. CUDA generators
    are seeded only for a concrete CUDA resolution, so CPU execution does not
    probe or initialize CUDA. Invalid device objects raise ``TypeError``.
    """
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Process seeding requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 -- exact process-global state is checkpointed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """
    Apply one exact deterministic policy to implemented Torch controls.

    The function mutates deterministic-algorithm and cuDNN process globals. When
    enabled it also sets the cuBLAS workspace environment contract. It performs
    no device selection and rejects non-boolean settings.
    """
    if not isinstance(enabled, bool):
        msg = f"run.deterministic must be boolean, got {enabled!r}."
        raise TypeError(msg)
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def configure_reproducibility(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, int]:
    """
    Apply reproducibility policy and seed the concrete process runtime.

    The resolved ``run`` section and the stage produce a stable labeled seed plan, then
    process-global deterministic controls and the ``process`` stream are applied.
    The full plan is returned for model, loader, worker, and curriculum owners.
    """
    run = config.get("run")
    if not isinstance(run, Mapping):
        msg = "Resolved config must contain a run mapping."
        raise TypeError(msg)
    seed_plan = build_seed_plan(int(run["seed"]), stage=_stage(config))
    configure_determinism(bool(run["deterministic"]))
    seed_process(seed_plan["process"], device=device)
    return seed_plan


def _strict_deterministic_cuda_conflicts(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return operation-level conflicts in one resolved training configuration."""
    conflicts: list[str] = []
    model = config.get("model")
    if isinstance(model, Mapping):
        uno_uses_bicubic = model.get("kind") == "uno" and learning.models.factory.UNO_RESAMPLING_MODE == "bicubic"
        if uno_uses_bicubic:
            conflicts.append(
                "the maintained UNO 2D resampling path uses bicubic interpolation, whose CUDA backward "
                "has no strict deterministic implementation in the supported environment"
            )
    return tuple(conflicts)


def validate_deterministic_model_device_policy(
    config: Mapping[str, Any],
    resolution: learning.device.DeviceResolution,
) -> None:
    """
    Reject resolved strict-deterministic CUDA requests with known conflicts.

    The check uses effective model, physics, derivative, and device properties.
    It runs before fresh-run or Optuna trial-run directory allocation. CPU requests,
    non-strict CUDA requests, and strict CUDA requests without a known conflict
    are admitted unchanged.

    Raises
    ------
    learning.device.DeviceResolutionError
        If strict CUDA would reach an operation without a deterministic backward
        implementation.

    Notes
    -----
    Rejection never changes model interpolation, derivative boundary semantics,
    seeds, or the resolved determinism policy. Authored configurations intended
    for these CUDA paths must set ``run.deterministic: false`` explicitly.

    """
    run = config.get("run")
    if not isinstance(run, Mapping) or resolution.device.type != "cuda" or not bool(run.get("deterministic")):
        return
    conflicts = _strict_deterministic_cuda_conflicts(config)
    if not conflicts:
        return
    details = "\n".join(f"- {conflict}." for conflict in conflicts)
    msg = (
        "Strict deterministic CUDA execution is incompatible with the resolved configuration:\n"
        f"{details}\n"
        "Set run.deterministic: false to retain explicit seeds with best-effort CUDA reproducibility, "
        "or select a device and scientific configuration whose operations support strict determinism. "
        "The runtime will not change interpolation, derivative boundary semantics, or determinism silently."
    )
    raise learning.device.DeviceResolutionError(msg)


def _validated_runtime_device(
    config: Mapping[str, Any],
    resolution: learning.device.DeviceResolution,
) -> torch.device:
    """
    Admit one service resolution against the requested config device policy.

    Requested policy equality is checked without re-resolving availability. The
    concrete decision is returned as a ``torch.device`` for downstream factories.
    """
    run = config.get("run")
    if not isinstance(run, Mapping):
        msg = "Resolved config must contain a run mapping."
        raise TypeError(msg)
    requested = learning.device_policy.validate_device_policy(run.get("device"), path="run.device")
    if resolution.requested_policy != requested:
        msg = f"Runtime device resolution does not match the requested config policy: {resolution.requested_policy!r} != {requested!r}."
        raise ValueError(msg)
    return resolution.device


def _grid_spacing(config: Mapping[str, Any], train_loader: Any) -> tuple[float, float]:
    """
    Return the physical grid spacing ``(spacing_y, spacing_x)`` in metres, rows then columns.

    Every Dataset stores physical coordinate channels (the steady inputs, the
    transient static conditioning) before the Train scaling standardizes them.
    Every case of a Dataset shares one coordinate grid, and the Dataset runtime
    refuses an Evaluation Dataset on another grid or view than Train
    (:func:`datasets.contracts.identity.require_train_grid`), so the first
    training sample defines the spacing on which loss and metrics differentiate
    for every case.
    """
    if config["task"] == "transient_drying":
        key = "static"
        names = tuple(field.name for field in datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning)
    else:
        key = "x"
        names = domain.tasks.registry.get_task(str(config["task"])).input_names
    sample = train_loader.dataset[0]
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(sample[key][names.index("x")], sample[key][names.index("y")])
    return float(spacing_y), float(spacing_x)


def runtime_session_updates(
    run_dir: Path,
    resolution: learning.device.DeviceResolution,
    *,
    started_at: datetime,
    memory_batch: int,
    session_id: str | None = None,
    tracking_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build summary updates that append one truthful runtime session, the start entry of one attempt.

    Prior sessions remain immutable. Each session records the memory batch its
    attempt split updates with, which stays outside run identity and may change on
    resume, its device facts with the driver version, its precision once the model
    exists, and an ``attempt`` entry whose values stay null until the attempt ends. The
    optional nested tracking state records disabled or pending observer facts without
    importing the W&B SDK.

    Parameters
    ----------
    run_dir : pathlib.Path
        Existing run whose summary supplies prior sessions.
    resolution : learning.device.DeviceResolution
        Truthful requested-versus-concrete runtime decision.
    started_at : datetime
        Session start instant persisted verbatim in ISO form.
    memory_batch : int
        ``training.memory_batch`` of this attempt.
    session_id : str | None, optional
        Caller identity, or a generated opaque ID when omitted.
    tracking_state : Mapping[str, Any] | None, optional
        Initial observer-only state copied into the new session.

    Returns
    -------
    dict[str, Any]
        Summary fields containing latest device facts and append-only sessions.

    """
    current = read_run_summary(run_dir)
    raw_sessions = current.get("runtime_sessions", [])
    if not isinstance(raw_sessions, list) or not all(isinstance(item, Mapping) for item in raw_sessions):
        msg = "Run summary runtime_sessions must be a list of mappings."
        raise RunLifecycleError(msg)
    resolved_session_id = session_id or uuid.uuid4().hex
    if not resolved_session_id:
        msg = "runtime session_id must be non-empty."
        raise ValueError(msg)
    device_metadata = resolution.as_dict()
    session: dict[str, Any] = {
        "session_id": resolved_session_id,
        "started_at": started_at.isoformat(),
        "memory_batch": memory_batch,
        **device_metadata,
        "cuda_driver_version": learning.device.cuda_driver_version(resolution),
        "precision": None,
        "attempt": dict.fromkeys(sorted(_ATTEMPT_KEYS)),
    }
    slurm_provenance = slurm_runtime_provenance()
    if slurm_provenance is not None:
        session["slurm"] = slurm_provenance
    if tracking_state is not None:
        session["tracking"] = copy.deepcopy(dict(tracking_state))
    sessions = [*copy.deepcopy(raw_sessions), session]
    return {
        "runtime_device": device_metadata,
        "runtime_sessions": sessions,
        **_attempt_totals(sessions),
    }


def _attempt_totals(sessions: list[Any]) -> dict[str, Any]:
    """
    Derive a run's total measured and attempted optimisation seconds from its attempts.

    Each total sums the attempts of ``runtime_sessions``; while one attempt has no end
    entry (in progress, or ended without publishing it), both are null and the coverage
    is ``unknown_attempts``, never a sum that counts that attempt as zero.
    """
    attempts = [session["attempt"] for session in sessions]
    complete = all(attempt["ended_at"] is not None for attempt in attempts)
    return {
        "elapsed_seconds": math.fsum(attempt["elapsed_seconds"] for attempt in attempts) if complete else None,
        "attempted_train_seconds": math.fsum(attempt["train_seconds"] for attempt in attempts) if complete else None,
        "duration_coverage": "complete" if complete else "unknown_attempts",
    }


def _require_attempt_history(summary: Mapping[str, Any], run_dir: Path) -> None:
    """
    Refuse to continue a run whose summary was not created under the current duration contract.

    The contract is written once when the summary is created, beside its empty attempt history,
    so a summary of an earlier attempt schema never carries it; a missing or malformed attempt
    history would let a resumed run publish totals that omit earlier attempts.
    """
    sessions = summary.get("runtime_sessions")
    history = isinstance(sessions, list) and all(
        isinstance(session, Mapping) and isinstance(session.get("attempt"), Mapping) and set(session["attempt"]) == _ATTEMPT_KEYS
        for session in sessions
    )
    if summary.get("duration_contract") != RUN_DURATION_CONTRACT or not history:
        msg = f"Run {run_dir} records its attempts in an earlier summary schema; start a new run with --revision <label>."
        raise RunLifecycleError(msg)


def slurm_runtime_provenance() -> dict[str, Any] | None:
    """Read the allocated worker's operational evidence, or ``None`` outside a Slurm ML job.

    Raises
    ------
    RunLifecycleError
        If ``ML_SLURM_PROVENANCE`` is not valid worker provenance.

    """
    raw = os.environ.get("ML_SLURM_PROVENANCE")
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        msg = "ML_SLURM_PROVENANCE must contain valid JSON from the Slurm worker."
        raise RunLifecycleError(msg) from error
    required = {
        "job_id",
        "partition",
        "cpus_per_task",
        "requested_memory",
        "requested_wall_time",
        "allocated_mem_per_node",
        "mode",
        "gres",
        "source_commit",
        "source_worktree_sha256",
        "source_transition",
        "config_sha256",
        "sif_path",
        "sif_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        msg = "ML_SLURM_PROVENANCE has an unsupported worker schema."
        raise RunLifecycleError(msg)
    if (
        not isinstance(payload["job_id"], str)
        or re.fullmatch(r"[1-9][0-9]*", payload["job_id"]) is None
        or not isinstance(payload["cpus_per_task"], int)
        or isinstance(payload["cpus_per_task"], bool)
        or payload["cpus_per_task"] < 1
        or not isinstance(payload["mode"], str)
        or payload["mode"] not in {"cpu", "gpu"}
        or not isinstance(payload["gres"], str)
        or (payload["mode"] == "cpu" and payload["gres"] != "none")
        or (payload["mode"] == "gpu" and re.fullmatch(r"gpu:(?:v100|rtxa6000|rtx6000ada):1", payload["gres"]) is None)
        or payload["sif_path"] != "/workspace/runtime/containers/grainlegumes-pino-drying.sif"
    ):
        msg = "ML_SLURM_PROVENANCE has invalid Slurm resource or SIF fields."
        raise RunLifecycleError(msg)
    for key in ("source_commit", "source_worktree_sha256", "config_sha256", "sif_sha256"):
        length = 40 if key == "source_commit" else 64
        if not isinstance(payload[key], str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", payload[key]) is None:
            msg = f"ML_SLURM_PROVENANCE has an invalid {key} digest."
            raise RunLifecycleError(msg)
    for key in ("partition", "requested_memory", "requested_wall_time", "allocated_mem_per_node"):
        if not isinstance(payload[key], str):
            msg = f"ML_SLURM_PROVENANCE has an invalid {key} field."
            raise RunLifecycleError(msg)
    if not isinstance(payload["source_transition"], str) or payload["source_transition"] not in _SOURCE_TRANSITIONS:
        msg = f"ML_SLURM_PROVENANCE has an invalid source_transition field; expected one of {sorted(_SOURCE_TRANSITIONS)}."
        raise RunLifecycleError(msg)
    return payload


def _merged_session_updates(
    summary: Mapping[str, Any],
    session_id: str,
    updates: Mapping[str, Any],
    *,
    field: str,
) -> list[Any]:
    """
    Return the summary's runtime sessions with ``updates`` merged into one session's ``field`` mapping.

    The session list is deep-copied. Missing/duplicate IDs or a malformed field
    raise ``RunLifecycleError``; a null field (an attempt's precision before the
    model exists) starts empty.
    """
    raw_sessions = summary.get("runtime_sessions")
    if not isinstance(raw_sessions, list):
        msg = "Run summary runtime_sessions must be a list."
        raise RunLifecycleError(msg)
    sessions = copy.deepcopy(raw_sessions)
    matches = [index for index, session in enumerate(sessions) if isinstance(session, Mapping) and session.get("session_id") == session_id]
    if len(matches) != 1:
        msg = f"Expected one runtime session {session_id!r}, found {len(matches)}."
        raise RunLifecycleError(msg)
    index = matches[0]
    session = dict(cast("Mapping[str, Any]", sessions[index]))
    current = session.get(field) or {}
    if not isinstance(current, Mapping):
        msg = f"Runtime session {session_id!r} {field} state must be a mapping."
        raise RunLifecycleError(msg)
    session[field] = {**copy.deepcopy(dict(current)), **copy.deepcopy(dict(updates))}
    sessions[index] = session
    return sessions


def _update_runtime_session_locked(
    run_dir: Path,
    session_id: str,
    updates: Mapping[str, Any],
    *,
    field: str = "tracking",
) -> dict[str, Any]:
    """
    Merge ``updates`` into one runtime session's ``field`` while the caller holds the writer lease.

    The complete summary is atomically replaced; a failed merge raises
    ``RunLifecycleError`` before publication.
    """
    summary = read_run_summary(run_dir)
    summary["runtime_sessions"] = _merged_session_updates(summary, session_id, updates, field=field)
    summary["updated_at"] = _utc_now()
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        summary,
    )
    return summary


def _ended_attempt_updates(
    run_dir: Path,
    session_id: str,
    *,
    started: float,
    work: learning.training.loop.AttemptWork,
    budget: learning.training.budget.StageBudget | None,
) -> dict[str, Any]:
    """
    Return the summary updates that publish one attempt's end entry and the totals of all attempts.

    The attempt computed every optimisation second its budget gained since its start,
    also for groups after its newest durable state that a resume trains again, plus the
    seconds of a group an exception or a divergence ended; before training started it computed none. Its
    committed work is that of its start and of its newest durable state (``AttemptWork``),
    ``None`` when training never started.
    """
    gained = 0.0 if work.at_start is None or budget is None else budget.train_seconds - float(work.at_start["train_seconds"])
    end = {
        "ended_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "train_seconds": gained + work.interrupted_group_seconds,
        "work_at_start": copy.deepcopy(work.at_start),
        "work_at_end": copy.deepcopy(work.durable),
    }
    sessions = _merged_session_updates(read_run_summary(run_dir), session_id, end, field="attempt")
    return {"runtime_sessions": sessions, **_attempt_totals(sessions)}


def _record_tracking_never_started(run_dir: Path, session_id: str, error: BaseException, message: str) -> None:
    """Record why tracking never started without masking the run's own failure.

    A run that fails before its runtime session exists cannot record this; the
    secondary failure is attached to ``error`` as a note instead of replacing it.
    """
    try:
        _update_runtime_session_locked(
            run_dir,
            session_id,
            {
                "status": "failed_before_start",
                "failed_operation": "local_admission",
                "error_class": type(error).__name__,
                "error_message": message,
            },
        )
    except (RunLifecycleError, FileNotFoundError) as secondary:
        error.add_note(f"Runtime session {session_id!r} could not record this failure: {secondary}")


def initial_tracking_state(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build safe initial observer facts before any SDK import or initialization.

    The result records configured project, local workflow, mode, tags, and status
    only. It performs no credential access, network operation, W&B import,
    or mutation of the resolved config.
    """
    settings = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    mode = str(settings["mode"])
    raw_tags = settings.get("tags", [])
    tags = list(cast("list[str]", raw_tags)) if isinstance(raw_tags, list) else []
    status = "disabled" if mode == "disabled" else ("offline" if mode == "offline" else "active")
    return {
        "requested_mode": mode,
        "workflow": settings.get("workflow"),
        "project": settings.get("project"),
        "entity": settings.get("entity"),
        "tags": tags,
        "status": status,
    }


def allocate_run_directory(run_dir: Path | str) -> Path:
    """
    Exclusively allocate one fresh run leaf.

    Existing leaves fail before any file inside them is read or written. Parent
    directories may be created as non-run containers.

    Parameters
    ----------
    run_dir : Path | str
        Exact fresh run leaf.

    Returns
    -------
    pathlib.Path
        Newly created absolute leaf.

    Raises
    ------
    FileExistsError
        If the leaf already exists; existing runs continue through run admission instead.

    """
    path = Path(run_dir).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        msg = f"Fresh run directory already exists; rerun the same request to continue it: {path}"
        raise FileExistsError(msg) from error
    return path.resolve()


def read_run_summary(run_dir: Path | str) -> dict[str, Any]:
    """
    Load one current run summary without mutation.

    Parameters
    ----------
    run_dir : Path | str
        Exact run leaf containing ``summary.json``.

    Returns
    -------
    dict[str, Any]
        Parsed summary mapping.

    Raises
    ------
    FileNotFoundError
        If the summary is absent.
    RunLifecycleError
        If the JSON is invalid or not an object.

    """
    path = common.paths.resolve_run_summary_path(run_dir)
    if not path.is_file():
        msg = f"Run summary not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as error:
        msg = f"Run summary is invalid JSON: {path}: {error}"
        raise RunLifecycleError(msg) from error
    if not isinstance(payload, dict):
        msg = f"Run summary must contain a JSON object: {path}"
        raise RunLifecycleError(msg)
    return payload


def is_trial_config(config: Mapping[str, Any]) -> bool:
    """Return whether a config's recorded run layout names an Optuna trial."""
    layout = (config.get("run") or {}).get("layout") or {}
    return layout.get("trial") is not None


def _is_trial_run(run_dir: Path | str) -> bool:
    """Return whether the run's saved config records an Optuna trial layout; a run without a saved config is none."""
    config_path = common.paths.resolve_run_config_path(run_dir)
    return config_path.is_file() and is_trial_config(config_loader.load_yaml(config_path))


def _transition_run_status_locked(
    run_dir: Path | str,
    status: str,
    *,
    updates: Mapping[str, Any] | None = None,
    keep_trial_best: bool = False,
) -> dict[str, Any]:
    """
    Atomically transition summary.json through the explicit run state machine.

    Parameters
    ----------
    run_dir : Path | str
        Allocated run directory.
    status : str
        Target lifecycle status.
    updates : Mapping[str, Any] | None, optional
        Additional JSON-safe summary fields.
    keep_trial_best : bool, optional
        Keep a trial's best checkpoint at its terminal status, because its evidence was not recorded.

    Returns
    -------
    dict[str, Any]
        Newly published summary payload.

    """
    summary_path = common.paths.resolve_run_summary_path(run_dir)
    current: dict[str, Any] = {}
    current_status: str | None = None
    if summary_path.exists():
        current = read_run_summary(run_dir)
        raw_status = current.get("status")
        if not isinstance(raw_status, str):
            msg = f"Run summary has no valid status: {summary_path}"
            raise RunLifecycleError(msg)
        current_status = raw_status
    allowed = _TRANSITIONS.get(current_status, frozenset())
    if status not in allowed:
        msg = f"Invalid run status transition {current_status!r} -> {status!r} for {run_dir}."
        raise RunLifecycleError(msg)

    now = _utc_now()
    history = list(current.get("status_history", []))
    history.append({"status": status, "time": now})
    payload = {
        **current,
        **dict(updates or {}),
        "status": status,
        "status_history": history,
        "updated_at": now,
    }
    payload.setdefault("created_at", now)
    common.serialization.atomic_write_json(summary_path, payload)
    if status in _TERMINAL_RUN_STATUSES and not keep_trial_best and _is_trial_run(run_dir):
        # Trials are never resumed and write no last; the summary keeps the best checkpoint's digest.
        common.paths.resolve_best_checkpoint_file(run_dir).unlink(missing_ok=True)
    return payload


def transition_run_status(
    run_dir: Path | str,
    status: str,
    *,
    updates: Mapping[str, Any] | None = None,
    keep_trial_best: bool = False,
) -> dict[str, Any]:
    """
    Publish one allowed summary-state transition under the writer lease.

    ``updates`` are merged with status history and timestamps through
    atomic JSON replacement. Unknown or forbidden transitions raise
    ``RunLifecycleError`` without changing the summary. A trial's terminal
    status deletes its best checkpoint unless ``keep_trial_best`` is set.
    """
    with run_writer_lease(run_dir):
        return _transition_run_status_locked(run_dir, status, updates=updates, keep_trial_best=keep_trial_best)


def _read_saved_data_artifacts(run_dir: Path | str) -> tuple[datasets.preprocessing.splits.SplitContract, dict[str, Any]]:
    """Read a run's saved split and scaling artifact through their one reader each; an unreadable or inadmissible one is a lifecycle error."""
    try:
        split = datasets.preprocessing.splits.read_saved_split(run_dir)
    except (TypeError, ValueError) as error:
        message = f"Saved split is invalid: {error}"
        raise RunLifecycleError(message) from error
    try:
        artifact = datasets.preprocessing.scaling.read_scaling_artifact(common.paths.resolve_normalizer_path(run_dir))
    except (TypeError, ValueError) as error:
        message = f"Saved scaling artifact is invalid: {error}"
        raise RunLifecycleError(message) from error
    return split, artifact


def _validate_saved_data_contract(
    config: Mapping[str, Any],
    split: datasets.preprocessing.splits.SplitContract,
    normalizer_artifact: Mapping[str, Any],
) -> datasets.preprocessing.normalization.TrainScaling:
    """
    Admit a saved split and scaling artifact against the resolved task config, the same way for both tasks.

    The admitted split must belong to the configured task, Datasets and item expansion: its Train Dataset is
    ``data.train_dataset``, its material-OOD parts are ``data.material_ood_datasets`` in order, followed by the
    training Dataset when that holds fixed OOD cases, and its parameter-OOD part is the training Dataset exactly when
    that holds fixed parameter-OOD cases; its Train subset is ``data.train_subset``.
    The scaling artifact must carry the scaling identity derived from that split, the configured grid stride and,
    for the transient task, the tensorizer. The admitted Train scaling is returned for resume, inference, or artifact
    consumers; a normalizer of an earlier format is refused.
    """
    task = config_loader.validate_resolved_task_contract(config)
    if split.task != task.id:
        msg = "Saved split task identity does not match the resolved config task contract."
        raise RunLifecycleError(msg)
    data = config["data"]
    own = {
        role: (split.train.dataset_id,) if datasets.contracts.partition.evaluation_cases(split.case_partition, role) else ()
        for role in datasets.preprocessing.splits.OOD_SPLIT_ROLES
    }
    expected_ood: dict[datasets.preprocessing.splits.OodRole, tuple[str, ...]] = {
        "material_ood": (*data["material_ood_datasets"], *own["material_ood"]),
        "parameter_ood": own["parameter_ood"],
    }
    if split.train.dataset_id != data["train_dataset"] or any(
        tuple(part.dataset_id for part in split.ood_parts(role)) != expected for role, expected in expected_ood.items()
    ):
        msg = "Saved split Datasets do not match the configured training and material-OOD Datasets."
        raise RunLifecycleError(msg)
    if split.train_subset != data["train_subset"]:
        msg = f"Saved split Train subset {split.train_subset!r} does not match data.train_subset {data['train_subset']!r}."
        raise RunLifecycleError(msg)
    try:
        if task.id == "transient_drying":
            tensorizer = TransientTensorizerSpec.from_mapping(
                {"input_profile": config["input_profile"], "temporal_conditioning": config["temporal"]["temporal_conditioning"]}
            )
            extension = cast("Mapping[str, Any]", split.extension)
            if (
                extension["tensorizer"] != tensorizer.selection_dict()
                or extension["sampling"] != datasets.contracts.transient.TransientSamplingSpec.from_mapping(config["temporal"]["sampling"]).as_dict()
                or extension["spatial_view"]["spatial_stride"] != config["data"]["spatial_stride"]
            ):
                msg = "Saved transient split item expansion conflicts with the resolved tensorizer, sampling or spatial stride."
                raise RunLifecycleError(msg)
            preprocessing = datasets.preprocessing.normalization.preprocessing_spec(
                task.id, spatial_view=extension["spatial_view"], tensorizer=tensorizer.as_dict()
            )
        else:
            shape = split.train.dataset["spatial_shape"]
            canonical = (int(shape[0]), int(shape[1]))
            view = datasets.contracts.grid.resolve_spatial_representation(canonical, config["data"]["spatial_stride"]).view_evidence()
            preprocessing = datasets.preprocessing.normalization.preprocessing_spec(task.id, spatial_view=view)
        identity = datasets.preprocessing.scaling.scaling_identity(split, preprocessing)
        state = datasets.preprocessing.scaling.admit_scaling_artifact(normalizer_artifact, identity=identity)
        return datasets.preprocessing.normalization.admit_scaling_state(state, identity=identity)
    except (KeyError, TypeError, ValueError) as error:
        message = f"Saved scaling artifact is invalid: {error}"
        raise RunLifecycleError(message) from error


def _different_fields(left: Any, right: Any, *, prefix: str = "") -> list[str]:
    """Return dotted leaf paths whose values differ."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left).union(right), key=str):
            field = f"{prefix}.{key}" if prefix else str(key)
            differences.extend(_different_fields(left.get(key, _MISSING), right.get(key, _MISSING), prefix=field))
        return differences
    return [prefix or "<root>"] if left is _MISSING or right is _MISSING or left != right else []


def _require_current_training(saved_config: Mapping[str, Any], *, unit: str) -> None:
    """Refuse to continue or reuse a run whose recorded training implementation differs from the running code."""
    common.code_identity.require_current_implementation(
        learning.training.checkpoint.TRAINING_PROJECTIONS[saved_config["task"]],
        saved_config["producer"]["training_sha256"],
        unit=unit,
        remedy=_TRAINING_CONTINUATION_REMEDY,
    )


def validate_resume_config(
    requested_config: Mapping[str, Any],
    saved_config: Mapping[str, Any],
) -> None:
    """
    Validate resume semantics: the request must hold the saved run's full identity.

    The identity (:func:`learning.training.checkpoint.effective_config_identity_payload`)
    includes the budget, k and the schedule, so a run is only continued after an
    interruption and never extended; another budget is another run. The requested
    derived ``run.name`` may differ from the saved name; the saved name remains
    authoritative for that run bundle. ``run.device``, resolved paths and
    ``training.memory_batch`` are execution settings handled separately.
    """
    identity = learning.training.checkpoint.effective_config_identity_payload
    differences = _different_fields(identity(requested_config), identity(saved_config))
    if differences:
        shown = ", ".join(differences[:_MAX_CONFIG_DIFFERENCES])
        suffix = " ..." if len(differences) > _MAX_CONFIG_DIFFERENCES else ""
        msg = (
            f"Requested config is incompatible with the saved run. Differing field(s): {shown}{suffix}. "
            "A run is never extended or changed: another budget, k or schedule is another run."
        )
        raise ValueError(msg)


def _canonical_config_source(config_path: Path | str) -> str:
    """Return a repository-relative source label without persisting or rewriting it."""
    source = Path(config_path).expanduser().resolve(strict=False)
    project_root = common.paths.get_project_root().expanduser().resolve(strict=False)
    try:
        return source.relative_to(project_root).as_posix()
    except ValueError:
        return str(source)


def inspect_existing_run_admission(
    run_dir: Path | str,
    requested_config: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> dict[str, Any]:
    """Build a lightweight read-only lifecycle and resume-compatibility report."""
    path = Path(run_dir).expanduser().resolve(strict=False)
    summary_path = common.paths.resolve_run_summary_path(path)
    config_file = common.paths.resolve_run_config_path(path)
    state = {
        "config": config_file.is_file(),
        "summary": summary_path.is_file(),
        "split_indices": common.paths.resolve_split_indices_path(path).is_file(),
        "normalizer": common.paths.resolve_normalizer_path(path).is_file(),
    }
    last_available = common.paths.resolve_last_checkpoint_file(path).is_file()
    best_available = common.paths.resolve_best_checkpoint_file(path).is_file()
    status = "unavailable"
    completed_epoch: int | None = None
    issues: list[str] = []
    requested_config_digest = run_identity.resolved_config_digest(requested_config)
    existing_config_digest: str | None = None

    try:
        summary = read_run_summary(path)
    except (OSError, RunLifecycleError) as error:
        issues.append(str(error))
    else:
        raw_status = summary.get("status")
        if isinstance(raw_status, str) and raw_status:
            status = raw_status
        else:
            issues.append("Run summary has no valid lifecycle status.")
        raw_completed = summary.get("completed_epoch")
        if type(raw_completed) is int and raw_completed >= 0:
            completed_epoch = raw_completed

    missing_resume = common.paths.missing_resume_run_files(path)
    if missing_resume:
        issues.append("Missing resume state: " + ", ".join(item.name for item in missing_resume) + ".")
    if status == "completed" and completed_epoch is None:
        issues.append("Completed run summary has no valid completed_epoch.")
    if status == "completed" and not best_available:
        issues.append("Completed run is missing best_checkpoint.pt.")

    if state["config"]:
        try:
            saved_config = config_loader.validate_resolved_config(
                config_loader.load_yaml(config_file),
            )
            existing_config_digest = run_identity.resolved_config_digest(saved_config)
            _require_current_training(saved_config, unit=f"Run {path}")
            validate_resume_config(requested_config, saved_config)
        except (OSError, KeyError, TypeError, ValueError) as error:
            issues.append(str(error))

    lock_state: bool | None
    try:
        lock_state = common.locking.file_lock_is_active(_run_writer_lock_path(path))
    except OSError as error:
        lock_state = None
        issues.append(f"Active writer state could not be verified: {error}")

    if lock_state is True:
        compatibility = "blocked"
        reason = "An active writer lease already owns this run."
    elif issues:
        compatibility = "incompatible"
        reason = issues[0]
    elif status == "completed":
        compatibility = "completed"
        reason = "The existing run already completed its budget."
    elif status not in {"running", "interrupted"}:
        compatibility = "incompatible"
        reason = f"Lifecycle status {status!r} is not eligible for explicit checkpoint resume."
    else:
        compatibility = "compatible"
        reason = "Saved semantics and required resume state are compatible."

    return {
        "requested_run_name": requested_config["run"]["name"],
        "config_path": _canonical_config_source(config_path),
        "run_dir": str(path),
        "status": status,
        "completed_epoch": completed_epoch,
        "last_checkpoint_available": last_available,
        "best_checkpoint_available": best_available,
        "state": state,
        "active_lock": lock_state,
        "resume_compatibility": compatibility,
        "reason": reason,
        "existing_config_digest": existing_config_digest,
        "requested_config_digest": requested_config_digest,
    }


def reject_existing_fresh_run(
    run_dir: Path | str,
    requested_config: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> ExistingRunAdmissionError:
    """Create the dedicated fresh-run rejection with exact identity diagnostics."""
    report = inspect_existing_run_admission(
        run_dir,
        requested_config,
        config_path=config_path,
    )
    existing_digest = report.get("existing_config_digest")
    requested_digest = report["requested_config_digest"]
    if report["resume_compatibility"] == "blocked":
        report["reason"] = f"{report['reason']} Wait until it finishes; running the same command again then continues the run."
    elif existing_digest is not None and existing_digest != requested_digest:
        report["reason"] = (
            f"Scientific identity conflict at {report['run_dir']}: the existing run holds different settings "
            f"({report['reason']}). Pass --revision <label> to start a deliberately separate run, or delete the old run."
        )
    else:
        report["reason"] = f"The existing run at {report['run_dir']} cannot continue automatically: {report['reason']}"
    return ExistingRunAdmissionError(report)


def _validate_resume_output_root(
    run_dir: Path,
    saved_config: Mapping[str, Any],
    output_root: Path | str | None,
) -> None:
    """Keep the explicit current resume directory authoritative."""
    del saved_config
    if output_root is not None:
        msg = f"--resume already identifies the authoritative current run directory {run_dir}. Do not combine it with an output-root override."
        raise ValueError(msg)


def _prepare_fresh_run_locked(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    terminal_status_resolver: Callable[[BaseException], str] | None = None,
) -> Path:
    """
    Allocate a fresh leaf and publish its initial summary/config under a lease.

    This internal variant requires its caller to own the destination writer
    lease. If initialization fails after allocation, it best-effort publishes a
    terminal failure while preserving the original exception.
    """
    task = str(config["task"])
    run_name = str(config["run"]["name"])
    destination = Path(run_dir) if run_dir is not None else common.paths.resolve_config_run_dir(config)
    with run_registry.reserve(config, destination):
        allocated = allocate_run_directory(destination)
    initial = {
        "task": task,
        "run_name": run_name,
        "objective": dict(config["evaluation"]["objective"]),
        # Every summary records its attempts from creation, so a resume admits only the current history schema.
        "runtime_sessions": [],
        "duration_contract": copy.deepcopy(RUN_DURATION_CONTRACT),
        **dict(summary_extra or {}),
    }
    try:
        transition_run_status(allocated, "initializing", updates=initial)
        config_loader.save_yaml(dict(config), common.paths.resolve_run_config_path(allocated))
    except BaseException as error:
        if common.paths.resolve_run_summary_path(allocated).is_file():
            _mark_failure(allocated, error, status=_failure_status(error, terminal_status_resolver))
        raise
    return allocated


def prepare_fresh_run(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    terminal_status_resolver: Callable[[BaseException], str] | None = None,
) -> Path:
    """
    Allocate and initialize a fresh run while holding its writer lease.

    Parameters
    ----------
    config : Mapping[str, Any]
        Fully resolved immutable experiment config.
    run_dir : Path | str | None, optional
        Exact fresh leaf override. Otherwise resolve from task, run name, and
        output root.
    summary_extra : Mapping[str, Any] | None, optional
        Additional caller-owned lifecycle facts for the initial summary.
    terminal_status_resolver : Callable[[BaseException], str] | None, optional
        Caller-owned failure classifier, as for :func:`execute_prepared_run`.

    Returns
    -------
    pathlib.Path
        Newly allocated run leaf containing an initializing summary and the
        authoritative immutable ``config.yaml``.

    Raises
    ------
    FileExistsError
        If the leaf already exists; only run admission may continue it.
    RunLifecycleError
        If another writer owns the destination or initialization cannot publish
        an allowed summary transition.
    OSError
        If initial summary or config publication fails.

    Notes
    -----
    The destination is allocated before its initial files are written. A
    post-allocation failure is re-raised after best-effort publication of a
    terminal failed summary. The incomplete leaf is retained for diagnosis and
    never treated as loadable.

    """
    destination = Path(run_dir) if run_dir is not None else common.paths.resolve_config_run_dir(config)
    with run_writer_lease(destination):
        return _prepare_fresh_run_locked(
            config,
            run_dir=destination,
            summary_extra=summary_extra,
            terminal_status_resolver=terminal_status_resolver,
        )


# Controller evidence that completed transient summaries mirror from the last checkpoint.
_TERMINAL_CONTROLLER_KEYS: tuple[str, ...] = (
    "arm",
    "stage",
    "post_handoff_optimizer_steps",
    "processed_target_transitions",
    "forward_transitions",
    "validation_seconds",
    "peak_cuda_memory_bytes",
    "best_within_budget_metric",
    "best_within_budget_epoch",
    "matched_receipt_sha256",
)
# Curriculum evidence that only a Stage B, the one stage that draws rollout horizons, records.
_TERMINAL_CURRICULUM_KEYS = ("active_stage", "max_horizon", "draw_index")
_TRAINING_OUTCOME_KEYS = frozenset({"status", "budget_progress", "eligible_points", "divergence"})
# A completed run has its best eligible checkpoint, at its budget end or up to its divergence; an unsuccessful one has none.
_LIFECYCLE_TRAINING_OUTCOMES = {
    "completed": frozenset({"completed", "diverged"}),
    "unsuccessful": frozenset({"diverged", "no_eligible_checkpoint"}),
}


def _terminal_adapter_evidence(adapter: transient_adapter.TransientTrainingAdapter) -> dict[str, Any]:
    """Return the terminal controller values and, for a Stage B, its terminal curriculum."""
    controller = adapter.controller_state()
    curriculum = adapter.curriculum_state
    return {
        "terminal_controller": {key: controller[key] for key in _TERMINAL_CONTROLLER_KEYS},
        **({} if curriculum is None else {"terminal_curriculum": {key: getattr(curriculum, key) for key in _TERMINAL_CURRICULUM_KEYS}}),
    }


def _stage(config: Mapping[str, Any]) -> str | None:
    """Return a transient config's stage, ``a`` or ``b``; a steady config has none."""
    return config["training"].get("stage")


def _is_self_fed(config: Mapping[str, Any]) -> bool:
    """Return whether a config is the Stage B, the one stage that draws rollout horizons and has a curriculum."""
    return _stage(config) == _SELF_FED_STAGE


def _stage_budget_target(
    config: Mapping[str, Any],
    *,
    full_set_units: object,
    matched_receipt: Mapping[str, Any] | None,
) -> tuple[learning.training.budget.BudgetUnit, float]:
    """
    Return one stage's budget unit and total, the one rule for training a stage and admitting its end.

    Steady runs and A0 train ``budget_epochs`` passes of the full Train set, counted in seen
    units, so a Train subset gets the full set's number. B trains its configured positive
    seconds of the optimisation clock and A1 exactly the seconds of B's receipt. Only that
    receipt can hold zero seconds (a B that diverged in its first group), so a zero total is
    admitted for its A1 and nowhere else.
    """
    protocol = config["training"].get("matched_a1_b")
    role = None if protocol is None else protocol["role"]
    if protocol is not None and role == "b":
        return "train_seconds", float(protocol["train_seconds"])
    if role == "a1":
        if matched_receipt is None:
            msg = "A matched A1 requires B's train-seconds receipt."
            raise RunLifecycleError(msg)
        return "train_seconds", float(matched_receipt["train_seconds"])
    if isinstance(full_set_units, bool) or not isinstance(full_set_units, int):
        msg = "A units budget requires the stage's full Train set unit count."
        raise RunLifecycleError(msg)
    return "units", learning.training.budget.units_budget(int(config["training"]["budget_epochs"]), full_set_units)


def _stage_budget(
    config: Mapping[str, Any],
    *,
    full_set_units: int,
    matched_receipt: Mapping[str, Any] | None,
) -> learning.training.budget.StageBudget:
    """Build one stage's budget from its resolved config (:func:`_stage_budget_target`)."""
    unit, total = _stage_budget_target(config, full_set_units=full_set_units, matched_receipt=matched_receipt)
    return learning.training.budget.StageBudget(unit=unit, total=total)


def _finished_stage_target(run_dir: Path, config: Mapping[str, Any], summary: Mapping[str, Any]) -> tuple[learning.training.budget.BudgetUnit, float]:
    """
    Return the budget unit and total a finished stage's terminal budget must carry.

    The same rule as at the stage's start (:func:`_stage_budget_target`): the full Train set
    units come from the summary's unit evidence, and an A1's receipt from its leaf, bound to
    the receipt digest its terminal controller records.
    """
    protocol = config["training"].get("matched_a1_b")
    receipt = None
    if protocol is not None and protocol["role"] == "a1":
        source = run_dir / matched_budget.RECEIPT_NAME
        if source.is_symlink() or not source.is_file():
            msg = f"Finished A1 lacks the B budget receipt it trained on: {source}"
            raise RunLifecycleError(msg)
        receipt = matched_budget.validate_b_receipt(json.loads(source.read_text(encoding="utf-8")))
        controller = summary.get("terminal_controller")
        if not isinstance(controller, Mapping) or controller.get("matched_receipt_sha256") != receipt["receipt_sha256"]:
            msg = "Finished A1 terminal controller is not bound to the B budget receipt in its leaf."
            raise RunLifecycleError(msg)
    units = summary.get("training_units")
    full_set_units = units.get("full_set_units") if isinstance(units, Mapping) else None
    return _stage_budget_target(config, full_set_units=full_set_units, matched_receipt=receipt)


def _terminal_budget_evidence(budget: Mapping[str, Any]) -> dict[str, Any]:
    """Return the terminal budget a summary records, with crossed points that had no own validation as plan deviation."""
    return {
        "terminal_budget": copy.deepcopy(dict(budget)),
        "validation_plan_deviation": {"skipped_points": list(budget["skipped_points"])},
    }


def _failure_status(error: BaseException, resolver: Callable[[BaseException], str] | None) -> str:
    """Return ``interrupted`` for an interrupt, otherwise the caller's admitted failure status or ``failed``."""
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    if resolver is None:
        return "failed"
    try:
        status = resolver(error)
    except BaseException as secondary:  # noqa: BLE001 -- a broken classifier, even an interrupt in it, must not replace the run's own failure
        error.add_note(f"Terminal status resolution failed: {type(secondary).__name__}: {secondary}")
        return "failed"
    if status not in _FAILURE_RUN_STATUSES:
        error.add_note(f"Terminal status resolver returned unsupported status {status!r}.")
        return "failed"
    return status


def _mark_failure(
    run_dir: Path,
    error: BaseException,
    *,
    status: str,
    updates: Mapping[str, Any] | None = None,
    terminal_evidence: TerminalEvidence | None = None,
) -> None:
    """
    Publish a terminal failure status on the one guarded path that never replaces the primary error.

    Caller evidence is merged before the transition, so it is bound before a
    trial's best checkpoint is deleted. When collecting it fails, the status is
    still published, the best checkpoint is kept and ``terminal_evidence_error``
    names the failure. A failure, including an interrupt, to collect the
    evidence or to publish the status is attached to ``error`` as a note.
    """
    payload = {"error_type": type(error).__name__, "error": str(error), "ended_at": _utc_now(), **dict(updates or {})}
    evidence_lost = False
    if terminal_evidence is not None:
        try:
            payload.update(terminal_evidence(status, error))
        except BaseException as secondary:  # noqa: BLE001 -- see docstring
            evidence_lost = True
            reason = payload["terminal_evidence_error"] = f"{type(secondary).__name__}: {console.sanitized_exception_message(secondary)}"
            error.add_note(f"Terminal evidence for status {status!r} was not recorded, so the best checkpoint is kept: {reason}")
    try:
        transition_run_status(run_dir, status, updates=payload, keep_trial_best=evidence_lost)
    except BaseException as secondary:  # noqa: BLE001 -- see docstring
        error.add_note(f"Terminal status {status!r} was not published: {type(secondary).__name__}: {secondary}")


def _validate_training_result_objective(
    result: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> None:
    """Require the training result to retain the resolved objective identity."""
    if result.get("objective") != objective:
        msg = "Training result objective does not match the resolved experiment objective."
        raise RunLifecycleError(msg)


def _execute_prepared_run_locked(  # noqa: C901, PLR0912, PLR0915
    config: dict[str, Any],
    *,
    run_dir: Path,
    persisted_config: Mapping[str, Any] | None = None,
    saved_split_indices: dict[str, Any] | None = None,
    restored_scaling_artifact: Mapping[str, Any] | None = None,
    resume_from: Path | None = None,
    event_callback: Callable[[int, dict[str, float]], None] | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    device_resolution: learning.device.DeviceResolution,
    terminal_status_resolver: Callable[[BaseException], str] | None = None,
    terminal_evidence: TerminalEvidence | None = None,
    matched_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build and execute one fresh or continued run in an allocated leaf.

    Authoritative data, split, normalizer, model, objective and device identity
    are admitted before optional W&B initialization. Local files and checkpoint
    state remain authoritative when a requested observer fails closed.

    Fresh execution atomically publishes fitted normalizer and split artifacts.
    Resume requires the identical admitted scaling and unchanged membership.
    The helper owns status transitions, factories, checkpoint execution, terminal
    digests, observer finalization, and the guarded failure publication while its
    caller retains the exclusive writer lease.
    """
    attempt_started = time.monotonic()
    start_time = datetime.now(UTC)
    operational_timing: dict[str, Any] = {"scope": "current_attempt"}
    # Filled by the training loop at its start and at every durable state, so an exceptional end still publishes the committed work.
    attempt_work = learning.training.loop.AttemptWork()
    budget: learning.training.budget.StageBudget | None = None
    attempt_started_entry = False
    runtime_session_id = uuid.uuid4().hex
    tracker: tracking.WandbSession | None = None
    tracking_status = "failed"
    tracking_result: Mapping[str, Any] | None = None
    tracking_error: BaseException | None = None
    tracking_initialization_attempted = False
    tracking_enabled = config["tracking"]["wandb"]["mode"] != "disabled"
    console_reporter = console.ConsoleReporter(
        config=config,
        run_dir=run_dir,
        resume=resume_from is not None,
        study_name=str(summary_extra["study_name"]) if summary_extra and "study_name" in summary_extra else None,
        trial_number=int(summary_extra["trial_number"]) if summary_extra and "trial_number" in summary_extra else None,
    )
    try:
        # Named before any check can fail the attempt, so a failed run keeps its logs; a trial's logs belong to its study.
        if not is_trial_config(config):
            common.paths.record_job_log_owner(run_dir)
        validate_deterministic_model_device_policy(config, device_resolution)
        device = _validated_runtime_device(config, device_resolution)
        seed_plan = build_seed_plan(int(config["run"]["seed"]), stage=_stage(config))
        transition_run_status(
            run_dir,
            "running",
            updates={
                "started_at": start_time.isoformat(),
                "operational_timing": copy.deepcopy(operational_timing),
                "seed_plan": seed_plan,
                "deterministic": bool(config["run"]["deterministic"]),
                **dict(summary_extra or {}),
                **runtime_session_updates(
                    run_dir,
                    device_resolution,
                    started_at=start_time,
                    memory_batch=int(config["training"]["memory_batch"]),
                    session_id=runtime_session_id,
                    tracking_state=initial_tracking_state(config),
                ),
            },
        )
        attempt_started_entry = True
        configure_reproducibility(config, device=device)
        is_transient = config["task"] == "transient_drying"
        console_reporter.startup(resolved_device=str(device))
        console_reporter.startup_phase("data_setup_start", {"spatial_stride": config["data"]["spatial_stride"]})
        handoff_manifest: dict[str, Any] | None = None
        handoff_directory: Path | None = None
        handoff_scaling = restored_scaling_artifact
        if is_transient and config["training"]["teacher_handoff"] is not None:
            if resume_from is None:
                source_name = str(config["training"]["teacher_handoff"]["source_run_name"])
                source_config = copy.deepcopy(dict(config))
                source_config["run"]["layout"].update({"stage": "a0", "branch_id": None, "pair": None})
                source_dir = common.paths.resolve_config_run_dir(source_config)
                source_completed = validate_completed_run(source_dir)
                handoff_directory = source_dir / transient_handoff.HANDOFF_DIRECTORY
                handoff_manifest = transient_handoff.validate_stage_a_handoff(
                    handoff_directory, target_config=config, device=device_resolution.as_dict(), expected_source_run_name=source_name
                )
                handoff_scaling = datasets.preprocessing.scaling.read_scaling_artifact(handoff_directory / "normalizer.pt")
                if handoff_manifest["checkpoint_identity"] != source_completed["checkpoint_identity"]:
                    message = "Teacher handoff checkpoint identity disagrees with its completed source run."
                    raise RunLifecycleError(message)  # noqa: TRY301
            else:
                local_normalizer = common.paths.resolve_normalizer_path(run_dir)
                handoff_manifest = transient_handoff.validate_local_teacher_handoff(
                    run_dir / "teacher_handoff_manifest.json",
                    local_normalizer_path=local_normalizer,
                    target_config=config,
                    device=device_resolution.as_dict(),
                )
                handoff_scaling = datasets.preprocessing.scaling.read_scaling_artifact(local_normalizer)

        def persist_split(resolved: Mapping[str, Any]) -> None:
            """Save exact Experiment membership before any scaler is fitted."""
            common.serialization.atomic_torch_save(dict(resolved), common.paths.resolve_split_indices_path(run_dir))

        dataloaders = config_loader.create_dataloaders_from_config(
            config,
            split_indices=saved_split_indices,
            scaling_artifact=handoff_scaling,
            seed_plan=seed_plan,
            startup_callback=console_reporter.startup_phase,
            split_persist_callback=persist_split if resume_from is None else None,
        )
        train_scaling = dataloaders["scaling"]
        split_indices = dataloaders["split_indices"]

        if resume_from is None and handoff_directory is not None:
            if handoff_manifest is None:
                message = "Teacher handoff directory requires one validated manifest."
                raise RunLifecycleError(message)  # noqa: TRY301
            normalizer_path = common.paths.resolve_normalizer_path(run_dir)
            temporary_normalizer = normalizer_path.with_name(f".{normalizer_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(handoff_directory / "normalizer.pt", temporary_normalizer)
                temporary_normalizer.replace(normalizer_path)
            finally:
                temporary_normalizer.unlink(missing_ok=True)
            common.serialization.atomic_write_json(run_dir / "teacher_handoff_manifest.json", handoff_manifest)
        elif resume_from is None:
            common.serialization.atomic_torch_save(dataloaders["scaling_artifact"], common.paths.resolve_normalizer_path(run_dir))

        seed_process(seed_plan["model_init"], device=device)
        model_started = time.perf_counter()
        console_reporter.startup_phase("model_build_start", {"model_kind": config["model"]["kind"]})
        if is_transient:
            learning.models.factory.validate_transient_model_spatial_shape(config, train_scaling.spatial_shape)
        model = learning.models.factory.build_model(config, device=device)
        console_reporter.startup_phase("model_cuda_ready", {"device": str(device), "seconds": time.perf_counter() - model_started})
        _update_runtime_session_locked(run_dir, runtime_session_id, learning.training.loop.attempt_precision(model), field="precision")
        # The relative H1 metrics of both tasks, the steady physics and the H1 data loss differentiate on the Train grid.
        grid_spacing = _grid_spacing(config, dataloaders["train"])
        train_loss = learning.losses.factory.build_training_loss(config, device=device, grid_spacing=grid_spacing)
        device_scaling = train_scaling.to(device)
        # The data loss weights vector terms by the Train scales, and the steady physics loss rebuilds physical views with exactly them.
        train_loss.set_scaling(device_scaling)
        adapter = None
        loop_scaling = None
        if is_transient:
            teacher_identity = None
            if handoff_manifest is not None:
                teacher_identity = transient_adapter.TeacherHandoffIdentity(
                    source_run_name=handoff_manifest["source_run_name"],
                    source_checkpoint_sha256=handoff_manifest["files"]["checkpoint"]["sha256"],
                    source_scaling_sha256=handoff_manifest["files"]["scaling"]["sha256"],
                    task_contract_sha256=handoff_manifest["task_contract_digest"],
                    tensorizer_sha256=handoff_manifest["tensorizer_digest"],
                    model_kind=handoff_manifest["model_kind"],
                    input_profile=handoff_manifest["input_profile"],
                )
            adapter = transient_adapter.build_transient_training_adapter(
                config,
                scaling=train_scaling,
                device=device,
                curriculum_seed=seed_plan.get(_CURRICULUM_SEED_LABEL),
                teacher_handoff=teacher_identity,
                matched_receipt_sha256=(str(matched_receipt["receipt_sha256"]) if matched_receipt is not None else None),
            )
            if matched_receipt is not None:
                matched_budget.publish_b_receipt(run_dir, matched_receipt)
        else:
            loop_scaling = device_scaling
        eval_metrics = learning.metrics.metrics.build_evaluation_metrics(
            config,
            device=device,
            output_scales=train_scaling.state_scale,
            grid_spacing=grid_spacing,
        )
        optimizer = learning.training.optim.build_optimizer(model, config)
        # B and A1 restart their own schedule at lr_scale times A0's start rate.
        continuation = config["training"].get("matched_a1_b") if is_transient else None
        lr_scale = 1.0 if continuation is None or continuation["role"] == "a0" else float(continuation["lr_scale"])
        scheduler_expected = config["scheduler"] is not None
        identity = learning.training.checkpoint.build_checkpoint_identity(
            config,
            split_indices,
            normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
            persisted_config=persisted_config,
        )

        transient_history_callback = None
        if is_transient:
            completed_epoch = 0
            if resume_from is not None:
                saved_last = learning.training.checkpoint.load_checkpoint(
                    resume_from,
                    expected_identity=identity,
                    expected_role="last",
                    payload_kind="full",
                    scheduler_expected=scheduler_expected,
                    require_best=False,
                    adapter_expected=True,
                )
                completed_epoch = learning.training.checkpoint.recorded_history_epochs(saved_last)
            initial_history = transient_adapter_history.reconcile_history(
                run_dir,
                task=str(config["task"]),
                run_name=str(config["run"]["name"]),
                checkpoint_identity=identity,
                completed_epoch=completed_epoch,
            )
            transient_history_callback = transient_adapter_history.make_epoch_state_callback(
                run_dir,
                task=str(config["task"]),
                run_name=str(config["run"]["name"]),
                checkpoint_identity=identity,
                initial_history=initial_history,
            )

        def state_updater(updates: Mapping[str, Any]) -> None:
            """Persist W&B observer facts while the run writer lease is already held."""
            _update_runtime_session_locked(
                run_dir,
                runtime_session_id,
                updates,
            )

        transient_scaling_payload: Mapping[str, Any] | None = None
        if is_transient:
            # The summary carries the scaling identity and evidence without its statistics; the digest binds those.
            transient_scaling_payload = {key: value for key, value in train_scaling.state_dict().items() if key != "groups"}
            transient_scaling_payload["semantic_digest"] = train_scaling.digest
        monitor_membership = tracking.build_monitor_membership(config, split_indices)
        if monitor_membership is not None:
            state_updater({"monitor": monitor_membership})

        persisted_run_id: str | None = None
        previous_last_logged_step: int | None = None
        previous_last_logged_point: int | None = None
        if tracking_enabled and resume_from is not None:
            persisted_run_id, previous_last_logged_step, previous_last_logged_point = tracking.persisted_wandb_identity(read_run_summary(run_dir))

        semantic_config: Mapping[str, Any] | None = None
        if tracking_enabled:
            semantic_config = tracking.build_semantic_config(
                config,
                split_indices=split_indices,
                split_indices_sha256=common.serialization.file_sha256(common.paths.resolve_split_indices_path(run_dir)),
                normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
                checkpoint_identity=identity,
                model=model,
                device_metadata=device_resolution.as_dict(),
                duration_contract=RUN_DURATION_CONTRACT,
                transient_scaling=transient_scaling_payload,
                transient_handoff=handoff_manifest,
                tuning_context=(dict(summary_extra) if summary_extra and "study_name" in summary_extra else None),
            )

        tracking_initialization_attempted = True
        tracker = tracking.initialize_wandb(
            config,
            run_dir=run_dir,
            semantic_config=semantic_config,
            resume=resume_from is not None,
            persisted_run_id=persisted_run_id,
            previous_last_logged_step=previous_last_logged_step,
            previous_last_logged_point=previous_last_logged_point,
            state_updater=state_updater,
            ood_roles=tuple(role for role in datasets.preprocessing.splits.OOD_SPLIT_ROLES if dataloaders[role] is not None),
        )

        if handoff_manifest is not None and handoff_directory is not None:
            # Only an is_teacher_handoff_source A0 publishes a handoff, so its linked best is full.
            teacher_payload = learning.training.checkpoint.load_checkpoint(
                handoff_directory / "best_checkpoint.pt",
                expected_identity=handoff_manifest["checkpoint_identity"],
                expected_role="best",
                payload_kind="full",
                scheduler_expected=False,
                require_best=True,
                adapter_expected=True,
            )
            learning.training.checkpoint.restore_handoff_checkpoint(
                teacher_payload,
                expected_source_identity=handoff_manifest["checkpoint_identity"],
                model=model,
                optimizer=optimizer,
                loss=train_loss,
                train_loader=dataloaders["train"],
            )
        # Built after the handoff, so a continuation keeps A0's moments and step counters but not its rate.
        scheduler = learning.training.optim.build_scheduler(optimizer, config, lr_scale=lr_scale)

        train_sampler = learning.training.loop.train_unit_sampler(dataloaders["train"])
        budget = _stage_budget(config, full_set_units=train_sampler.full_set_units, matched_receipt=matched_receipt)
        console_reporter.startup_phase("training_start", {"updates_per_epoch": train_sampler.updates_per_epoch})
        result = learning.training.loop.train_loop(
            config=config,
            device=device,
            model=model,
            optimizer=optimizer,
            train_loader=dataloaders["train"],
            eval_loader=dataloaders["eval"],
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            ood_loaders={role: dataloaders[role] for role in datasets.preprocessing.splits.OOD_SPLIT_ROLES},
            scaling=loop_scaling,
            scheduler=scheduler,
            save_dir=run_dir,
            resume_from=resume_from,
            event_callback=tracking.combine_event_callbacks(
                console_reporter.event,
                event_callback,
                tracking.event_callback(tracker),
            ),
            checkpoint_identity=identity,
            adapter=adapter,
            epoch_state_callback=transient_history_callback,
            operational_timing=operational_timing,
            attempt_work=attempt_work,
            budget=budget,
        )
        result["operational_timing"] = operational_timing
        if result["best_epoch"] is None:
            # A run without an eligible checkpoint (diverged or no eligible point) ends unsuccessful, never resumed.
            # The same terminal evidence as a completed transient run, so admission binds it to the last checkpoint.
            terminal: dict[str, Any] = {"terminal_controller": None} if adapter is None else _terminal_adapter_evidence(adapter)
            last_file = common.paths.resolve_last_checkpoint_file(run_dir)
            terminal_updates = {
                "completed_epoch": result["completed_epoch"],
                "global_step": result["global_step"],
                "training_work": dict(result["training_work"]),
                "training_units": learning.training.loop.unit_evidence(train_sampler),
                "best_epoch": result["best_epoch"],
                "best_metric": result["best_metric"],
                "terminal_epoch": result["terminal_epoch"],
                **terminal,
                "training_outcome": copy.deepcopy(result["training_outcome"]),
                **_terminal_budget_evidence(result["budget"]),
                "operational_timing": copy.deepcopy(operational_timing),
                "last_checkpoint_sha256": common.serialization.file_sha256(last_file) if last_file.is_file() else None,
                "ended_at": datetime.now(UTC).isoformat(),
                "teacher_handoff": handoff_manifest,
                **_ended_attempt_updates(run_dir, runtime_session_id, started=attempt_started, work=attempt_work, budget=budget),
            }
            transition_run_status(run_dir, "unsuccessful", updates=terminal_updates)
            tracking_status = "unsuccessful"
            tracking_result = result
            return result
        objective = config_loader.get_resolved_objective(config)
        _validate_training_result_objective(result, objective)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        selected_started = time.monotonic()
        try:
            selected = learning.training.loop.evaluate_selected_checkpoint(
                config=config,
                model=model,
                train_loss=train_loss,
                eval_loader=dataloaders["eval"],
                ood_loaders={role: dataloaders[role] for role in datasets.preprocessing.splits.OOD_SPLIT_ROLES},
                eval_metrics=eval_metrics,
                device=device,
                scaling=loop_scaling,
                checkpoint_identity=identity,
                best_checkpoint_path=result["best_checkpoint_path"],
                scheduler_expected=scheduler_expected,
                max_physics_cases=int(config["tracking"]["wandb"]["monitor"]["max_cases"]),
                adapter=adapter,
            )
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            operational_timing["selected_checkpoint_evaluation_seconds"] = time.monotonic() - selected_started
        result.update(selected)
        caller_evidence = dict(terminal_evidence("completed", result)) if terminal_evidence is not None else {}
        published_handoff: dict[str, Any] | None = None
        # Only a matched-plan A0 is ever continued by B/A1; stage-A-only trials never branch.
        if learning.training.checkpoint.is_teacher_handoff_source(config):
            tensorizer_digest = common.serialization.canonical_json_sha256(dataloaders["tensorizer"].as_dict())
            published_handoff = transient_handoff.publish_stage_a_handoff(
                run_dir,
                source_run_name=str(config["run"]["name"]),
                checkpoint_identity=identity,
                best_epoch=int(result["best_epoch"]),
                global_step=int(result["global_step"]),
                scaling_semantic_digest=train_scaling.digest,
                task_contract_digest=str(config["task_contract"]["digest"]),
                tensorizer_digest=tensorizer_digest,
                model_kind=str(config["model"]["kind"]),
                input_profile=str(config["input_profile"]),
                device=device_resolution.as_dict(),
                config=config,
            )
        end_time = datetime.now(UTC)
        result["phase_seconds"] = console_reporter.final(result, total_wall_seconds=time.monotonic() - attempt_started)
        transient_completion_evidence: dict[str, Any] = {}
        if is_transient:
            if adapter is None or transient_scaling_payload is None:
                message = "Transient completion requires adapter and scaling evidence."
                raise AssertionError(message)  # noqa: TRY301
            transient_completion_evidence = {
                "checkpoint_identity": copy.deepcopy(identity),
                "transient_scaling": copy.deepcopy(dict(transient_scaling_payload)),
                "spatial_view": copy.deepcopy(dict(dataloaders["spatial_view"])),
                **_terminal_adapter_evidence(adapter),
            }
        # A trial writes no last checkpoint, so its summary names none.
        last_checkpoint_evidence = (
            {}
            if is_trial_config(config)
            else {
                "last_checkpoint": common.paths.RUN_LAST_CHECKPOINT_FILENAME,
                "last_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_last_checkpoint_file(run_dir)),
            }
        )
        completed_updates = {
            "task": config["task"],
            "run_name": config["run"]["name"],
            "model_kind": config["model"]["kind"],
            "model_parameter_counts": tracking.model_parameter_counts(model),
            "objective": objective,
            "best_epoch": result["best_epoch"],
            "best_metric": result["best_metric"],
            "selected_epoch": result["selected_epoch"],
            "selected_metrics": copy.deepcopy(result["selected_metrics"]),
            "terminal_epoch": result["terminal_epoch"],
            "terminal_metrics": copy.deepcopy(result["terminal_metrics"]),
            "training_outcome": copy.deepcopy(result["training_outcome"]),
            "completed_epoch": result["completed_epoch"],
            "global_step": result["global_step"],
            "training_work": dict(result["training_work"]),
            "training_units": learning.training.loop.unit_evidence(train_sampler),
            "best_checkpoint": "best_checkpoint.pt",
            "config_sha256": common.serialization.file_sha256(common.paths.resolve_run_config_path(run_dir)),
            "split_indices_sha256": common.serialization.file_sha256(common.paths.resolve_split_indices_path(run_dir)),
            "normalizer_sha256": common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
            "best_checkpoint_sha256": common.serialization.file_sha256(common.paths.resolve_best_checkpoint_file(run_dir)),
            **last_checkpoint_evidence,
            "effective_config_digest": identity["effective_config_digest"],
            "operational_timing": copy.deepcopy(operational_timing),
            "ended_at": end_time.isoformat(),
            "error": None,
            "error_type": None,
            "teacher_handoff": handoff_manifest,
            "teacher_handoff_manifest_sha256": (
                common.serialization.file_sha256(run_dir / "teacher_handoff_manifest.json")
                if (run_dir / "teacher_handoff_manifest.json").is_file()
                else None
            ),
            "stage_a_handoff": published_handoff,
            **transient_completion_evidence,
            **dict(summary_extra or {}),
            **caller_evidence,
            **_terminal_budget_evidence(result["budget"]),
            **_ended_attempt_updates(run_dir, runtime_session_id, started=attempt_started, work=attempt_work, budget=budget),
        }
        transition_run_status(run_dir, "completed", updates=completed_updates)
        tracking_status = "completed"
        tracking_result = result
    except BaseException as error:
        tracking_status, tracking_error = _failure_status(error, terminal_status_resolver), error
        if tracking_enabled and tracker is None and not tracking_initialization_attempted:
            _record_tracking_never_started(
                run_dir,
                runtime_session_id,
                error,
                f"Local run {'was interrupted' if tracking_status == 'interrupted' else 'admission failed'} before tracking initialization.",
            )
        # A pruned trial is a normal outcome that its study reports in one line.
        if tracking_status not in _PRUNED_RUN_STATUSES:
            console_reporter.failure(error, status=tracking_status)
        ended: dict[str, Any] = {}
        if attempt_started_entry:
            try:
                ended = _ended_attempt_updates(run_dir, runtime_session_id, started=attempt_started, work=attempt_work, budget=budget)
            except BaseException as secondary:  # noqa: BLE001 -- the attempt's own failure must propagate; its start entry then stays unknown
                error.add_note(f"The attempt's end entry was not recorded: {type(secondary).__name__}: {secondary}")
        durable = attempt_work.durable
        _mark_failure(
            run_dir,
            error,
            status=tracking_status,
            terminal_evidence=terminal_evidence,
            updates={
                "started_at": start_time.isoformat(),
                **ended,
                "operational_timing": copy.deepcopy(operational_timing),
                # Only the work a continuation restores is committed; groups after it are in the attempt's train_seconds.
                **({"training_work": {key: durable[key] for key in sorted(learning.training.checkpoint.TRAINING_WORK_KEYS)}} if durable else {}),
            },
        )
        raise
    finally:
        if tracker is not None:
            local_summary: Mapping[str, Any] | None = None
            with suppress(Exception):
                local_summary = read_run_summary(run_dir)
            try:
                tracker.finish(
                    status=tracking_status,
                    result=tracking_result,
                    local_summary=local_summary,
                    error=tracking_error,
                )
            except tracking.TrackingError as finish_error:
                # Tracking never decides local validity and finish kept its failed state locally, so the published end
                # state stands; a failed run keeps its own error.
                if tracking_error is None:
                    print(
                        f"WARNING W&B finish failed after the local run ended {tracking_status}; that end state stands: "
                        f"{type(finish_error).__name__}: {console.sanitized_exception_message(finish_error)}",
                        file=sys.stderr,
                        flush=True,
                    )
    return result


def execute_prepared_run(
    config: dict[str, Any],
    *,
    run_dir: Path,
    persisted_config: Mapping[str, Any] | None = None,
    saved_split_indices: dict[str, Any] | None = None,
    restored_scaling_artifact: Mapping[str, Any] | None = None,
    resume_from: Path | None = None,
    event_callback: Callable[[int, dict[str, float]], None] | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    device_resolution: learning.device.DeviceResolution,
    terminal_status_resolver: Callable[[BaseException], str] | None = None,
    terminal_evidence: TerminalEvidence | None = None,
) -> dict[str, Any]:
    """
    Execute a prepared fresh or resume run under its exclusive writer lease.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved runtime config. On resume this may change only execution
        settings (device policy, memory batch) while preserving the saved identity.
    run_dir : pathlib.Path
        Already initialized run leaf whose lifecycle this call exclusively owns.
    persisted_config : Mapping[str, Any] | None, optional
        Immutable saved config used for checkpoint identity. Defaults to
        ``config`` for fresh execution.
    saved_split_indices : dict[str, Any] | None, optional
        Admitted saved membership reused while a run continues.
    restored_scaling_artifact : Mapping[str, Any] | None, optional
        Admitted saved scaling artifact (``normalizer.pt``) reused while a run continues.
    resume_from : pathlib.Path | None, optional
        Validated ``last_checkpoint.pt`` continuation source. ``None`` starts a
        fresh training state.
    event_callback : Callable[[int, dict[str, float]], None] | None, optional
        Local authoritative callback invoked with the global step after every
        training event (validation point, epoch end, budget end), before the
        optional tracking observer. Validation keys appear at validation points.
    summary_extra : Mapping[str, Any] | None, optional
        Caller-owned facts merged into lifecycle publications.
    device_resolution : learning.device.DeviceResolution
        Concrete service-resolved device and serializable runtime metadata.
    terminal_status_resolver : Callable[[BaseException], str] | None, optional
        Caller-owned classifier for failures that are not interrupts. A raise or
        a status outside the admitted failure statuses publishes ``failed``.
    terminal_evidence : TerminalEvidence | None, optional
        Caller-owned facts merged into the terminal summary. Called with
        ``("completed", result)`` before the completed transition, where a raise
        fails the run, and with ``(status, error)`` on the failure path, where a
        raise becomes a note on the error.

    Returns
    -------
    dict[str, Any]
        Completed training result with objective, history, and checkpoint facts.

    Raises
    ------
    RunLifecycleError
        If the writer lease, saved data, checkpoint identity, objective result,
        or lifecycle transition violates the admitted run contract.
    BaseException
        Runtime construction, training, interruption, and required tracking
        failures during the run are re-raised unchanged after the terminal
        publication; ``KeyboardInterrupt`` and ``SystemExit`` publish ``interrupted``.

    Notes
    -----
    Fresh execution atomically publishes split and normalizer state before model
    construction. Resume reuses those immutable artifacts and restores only the
    last checkpoint. The best checkpoint remains the selected inference source.
    Local summary/checkpoint publication is authoritative. A failure in an
    explicitly requested online or offline W&B session during the run follows
    the tracking contract and fails the run. A W&B finish failure after the
    local run ended without error is a console warning and leaves its published
    end state unchanged.

    This is a lower-level service for callers that already prepared lifecycle
    state. Normal CLI/notebook launches should prefer :func:`run_experiment`.

    """
    if resume_from is not None:
        # The service boundaries guard their own resumes; this public entry continues persisted computation directly.
        _require_current_training(config, unit=f"Run {run_dir}")
    with run_writer_lease(run_dir):
        return _execute_prepared_run_locked(
            config,
            run_dir=run_dir,
            persisted_config=persisted_config,
            saved_split_indices=saved_split_indices,
            restored_scaling_artifact=restored_scaling_artifact,
            resume_from=resume_from,
            event_callback=event_callback,
            summary_extra=summary_extra,
            device_resolution=device_resolution,
            terminal_status_resolver=terminal_status_resolver,
            terminal_evidence=terminal_evidence,
        )


def _evaluable_run_result(
    *,
    path: Path,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    normalizer_state: datasets.preprocessing.normalization.TrainScaling,
    checkpoint_identity: Mapping[str, Any],
    best_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common immutable evaluation-admission result."""
    scientific_run_name = str(config["run"]["name"])
    checkpoint_path = common.paths.resolve_best_checkpoint_file(path)
    normalizer_path = common.paths.resolve_normalizer_path(path)
    return {
        "run_dir": path,
        "summary": dict(summary),
        "config": dict(config),
        "split_indices": dict(split_indices),
        "normalizer_state": normalizer_state.state_dict(),
        "checkpoint_identity": dict(checkpoint_identity),
        "best_checkpoint": dict(best_checkpoint),
        "scientific_run_name": scientific_run_name,
        "storage_alias": path.name,
        "selected_checkpoint_role": "best",
        "selected_checkpoint_epoch": int(best_checkpoint["best_epoch"]),
        "selected_checkpoint_sha256": common.serialization.file_sha256(checkpoint_path),
        "normalizer_sha256": common.serialization.file_sha256(normalizer_path),
        "effective_config_digest": checkpoint_identity["effective_config_digest"],
    }


def _run_validation_identity(
    *,
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    normalizer_path: Path,
    checkpoint_path: Path,
    purpose: Literal["execution", "saved_artifacts"],
) -> dict[str, Any]:
    """Build the current checkpoint identity; saved-artifact reading takes the numerical environment from the checkpoint."""
    identity = learning.training.checkpoint.build_checkpoint_identity(
        config,
        split_indices,
        normalizer_sha256=common.serialization.file_sha256(normalizer_path),
        persisted_config=config,
    )
    if purpose == "saved_artifacts":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # checkpoints are read under their own module's policy
        saved = payload.get("identity") if isinstance(payload, Mapping) else None
        if not isinstance(saved, Mapping):
            msg = "Saved artifact checkpoint identity must be a mapping."
            raise RunLifecycleError(msg)
        identity["numerical_environment"] = saved.get("numerical_environment")
    return identity


def _require_completed_status(path: Path) -> None:
    """Refuse any other lifecycle status with one diagnostic, before payload files are required.

    A run without a summary is left to the callers' missing-file diagnostics.
    """
    if not common.paths.resolve_run_summary_path(path).is_file():
        return
    status = read_run_summary(path).get("status")
    if status != "completed":
        # An unsuccessful run ended at its budget without an eligible checkpoint; resuming it cannot give it a result.
        advice = "It ended without an eligible checkpoint and has no result." if status == "unsuccessful" else "Resume the run to completion first."
        msg = f"Run must be completed, got status {status!r}: {path}. {advice}"
        raise RunLifecycleError(msg)


def _validate_evaluable_run_unlocked(
    run_dir: Path | str,
    *,
    purpose: Literal["execution", "saved_artifacts"] = "execution",
) -> dict[str, Any]:
    """Validate completed evaluation evidence while the caller excludes writers.

    Any other lifecycle status is refused: evaluation never reads partial results.
    """
    path = Path(run_dir).expanduser().resolve()
    _require_completed_status(path)
    missing = common.paths.missing_evaluable_run_files(path)
    if missing:
        names = ", ".join(item.name for item in missing)
        if common.paths.resolve_best_checkpoint_file(path) in missing:
            msg = f"Evaluable run has no valid best checkpoint: {path}. Missing: {names}."
        else:
            msg = f"Run lacks required evaluation evidence: {path}. Missing: {names}."
        raise RunLifecycleError(msg)

    history = read_run_summary(path).get("status_history")
    if not isinstance(history, list) or not history or not isinstance(history[-1], Mapping) or history[-1].get("status") != "completed":
        msg = "Run summary status history does not end at its declared terminal status."
        raise RunLifecycleError(msg)
    return _validate_completed_run(path, purpose=purpose)


@contextmanager
def run_reader_lease(run_dir: Path | str) -> Iterator[Path]:
    """Hold a shared fail-fast reader lease that excludes lifecycle writers."""
    path = Path(run_dir).expanduser().resolve(strict=False)
    try:
        with common.locking.shared_file_lock(_run_writer_lock_path(path), blocking=False):
            yield path
    except common.locking.FileLockUnavailableError as error:
        msg = f"Run has an active writer lease and cannot be evaluated: {path}"
        raise RunLifecycleError(msg) from error


@contextmanager
def evaluable_run_lease(run_dir: Path | str) -> Iterator[dict[str, Any]]:
    """Hold a shared read lease and yield validated completed-run evaluation evidence."""
    with run_reader_lease(run_dir) as path:
        yield _validate_evaluable_run_unlocked(path)


@contextmanager
def saved_artifact_run_lease(run_dir: Path | str) -> Iterator[dict[str, Any]]:
    """Admit persisted run evidence only for reading already published artifacts.

    Only completed runs are admitted. This lease establishes no resume, inference,
    or artifact-generation authority.
    The run is validated like an execution run against its own recorded producer
    evidence, never the running code, except that the numerical environment is
    also the checkpoint's own, so a run copied to another machine stays readable;
    payload readers retain their own schema, provenance, membership, hash, and
    containment admission.
    """
    with run_reader_lease(run_dir) as path:
        required = common.paths.resolve_current_run_required_paths(path)
        if any(item.is_symlink() for item in required):
            msg = "Saved artifact run evidence must contain regular bundle-local files."
            raise RunLifecycleError(msg)
        yield _validate_evaluable_run_unlocked(path, purpose="saved_artifacts")


def validate_evaluable_run(run_dir: Path | str) -> dict[str, Any]:
    """
    Validate a completed run for best-checkpoint evaluation under a reader lease.

    Runs in any other lifecycle status (``unsuccessful``, ``interrupted``, ...)
    are refused: resume them to completion before evaluating.
    """
    with evaluable_run_lease(run_dir) as admitted:
        return admitted


def _validate_transient_summary_evidence(
    run_dir: Path,
    *,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    normalizer_state: Any,
    checkpoint_identity: Mapping[str, Any],
    completed_epoch: int,
) -> None:
    """Cross-check the summary's transient completion evidence and the epoch history against saved state."""
    if not {"checkpoint_identity", "transient_scaling"}.issubset(summary):
        message = "Completed transient run summary lacks required completion evidence."
        raise RunLifecycleError(message)
    _validate_transient_terminal_state(summary, None, config=config)
    if summary["checkpoint_identity"] != checkpoint_identity:
        message = "Completed transient summary checkpoint identity mismatch."
        raise RunLifecycleError(message)
    scaling = summary["transient_scaling"]
    if not isinstance(scaling, Mapping) or scaling.get("semantic_digest") != normalizer_state.digest:
        message = "Completed transient summary scaling identity mismatch."
        raise RunLifecycleError(message)
    if scaling.get("task_contract_digest") != config["task_contract"]["digest"]:
        message = "Completed transient summary task contract digest mismatch."
        raise RunLifecycleError(message)
    try:
        history = transient_adapter_history.validate_completed_history(
            run_dir,
            task=str(config["task"]),
            run_name=str(config["run"]["name"]),
            checkpoint_identity=checkpoint_identity,
            completed_epoch=completed_epoch,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        message = f"Completed transient history admission failed: {error}"
        raise RunLifecycleError(message) from error
    # A trial keeps no last checkpoint to compare, so the values are admitted by their owner against the final epoch record.
    try:
        records = history["epochs"]
        transient_curriculum.validate_terminal_evidence(
            summary["terminal_controller"], summary.get("terminal_curriculum", {}), final_record=records[-1] if records else {}
        )
    except (TypeError, ValueError) as error:
        message = f"Completed transient summary terminal evidence is invalid: {error}"
        raise RunLifecycleError(message) from error


def validate_training_outcome(
    summary: Mapping[str, Any], *, lifecycle: str, target: tuple[learning.training.budget.BudgetUnit, float]
) -> learning.training.budget.StageBudget:
    """
    Admit a finished run's training outcome against its terminal budget and return that budget.

    The one rule for every finished run: the terminal budget has the stage's unit and total
    (``target``, :func:`_stage_budget_target`), the outcome status is allowed for the run's
    lifecycle (``completed``: completed or diverged with its best eligible checkpoint;
    ``unsuccessful``: diverged or no eligible checkpoint), its budget progress is the terminal
    budget's, and the budget is complete unless the stage diverged. Example: a B that diverged
    at 30 % of its budget records progress 0.3 and an incomplete budget; the same budget without
    a divergence is refused, and so is a B budget of zero seconds.
    """
    try:
        budget = learning.training.budget.StageBudget(**learning.training.budget.validate_state(summary.get("terminal_budget")))
    except (TypeError, ValueError) as error:
        message = f"Finished run summary terminal budget is invalid: {error}"
        raise RunLifecycleError(message) from error
    if (budget.unit, budget.total) != target:
        message = f"Finished run terminal budget {budget.total} {budget.unit} differs from the stage's configured {target[1]} {target[0]}."
        raise RunLifecycleError(message)
    outcome = summary.get("training_outcome")
    if not isinstance(outcome, Mapping) or set(outcome) != _TRAINING_OUTCOME_KEYS:
        message = "Finished run summary lacks its training outcome."
        raise RunLifecycleError(message)
    status, eligible_points = outcome["status"], outcome["eligible_points"]
    if (
        status not in _LIFECYCLE_TRAINING_OUTCOMES[lifecycle]
        or outcome["budget_progress"] != budget.progress
        or (outcome["divergence"] is not None) != (status == "diverged")
        or isinstance(eligible_points, bool)
        or not isinstance(eligible_points, int)
        or (eligible_points > 0) != (lifecycle == "completed")
    ):
        message = f"A {lifecycle} run's training outcome is not allowed or contradicts its terminal budget progress."
        raise RunLifecycleError(message)
    if status != "diverged" and not budget.complete:
        message = "Finished run summary records an incomplete terminal budget without a divergence."
        raise RunLifecycleError(message)
    if summary.get("validation_plan_deviation") != {"skipped_points": budget.skipped_points}:
        message = "Finished run summary must record its skipped validation points."
        raise RunLifecycleError(message)
    return budget


def _validate_terminal_budget(
    summary: Mapping[str, Any],
    saved_budget: Mapping[str, Any] | None,
    *,
    lifecycle: str,
    target: tuple[learning.training.budget.BudgetUnit, float],
) -> learning.training.budget.StageBudget:
    """Admit a finished summary's training outcome and require its terminal budget to equal the last checkpoint's."""
    budget = validate_training_outcome(summary, lifecycle=lifecycle, target=target)
    if saved_budget is not None and dict(saved_budget) != budget.state_dict():
        message = f"A {lifecycle} run summary terminal budget disagrees with its last checkpoint."
        raise RunLifecycleError(message)
    return budget


def _saved_run_identity(
    path: Path, config: Mapping[str, Any], *, checkpoint_path: Path, purpose: Literal["execution", "saved_artifacts"]
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Return a run leaf's split indices, normalizer state and current checkpoint identity from its saved data artifacts."""
    split, normalizer_artifact = _read_saved_data_artifacts(path)
    split_indices = split.as_payload()
    normalizer_state = _validate_saved_data_contract(config, split, normalizer_artifact)
    identity = _run_validation_identity(
        config=config,
        split_indices=split_indices,
        normalizer_path=common.paths.resolve_normalizer_path(path),
        checkpoint_path=checkpoint_path,
        purpose=purpose,
    )
    return split_indices, normalizer_state, identity


def _load_last_checkpoint(path: Path, config: Mapping[str, Any], *, identity: Mapping[str, Any], require_best: bool) -> dict[str, Any]:
    """Load a finished run's last checkpoint read-only on CPU against its identity, role, payload, scheduler and adapter contract."""
    return learning.training.checkpoint.load_checkpoint(
        common.paths.resolve_last_checkpoint_file(path),
        expected_identity=identity,
        expected_role="last",
        payload_kind="full",
        scheduler_expected=config.get("scheduler") is not None,
        require_best=require_best,
        adapter_expected=config["task"] == "transient_drying",
    )


def _validate_transient_terminal_state(summary: Mapping[str, Any], last_checkpoint: Mapping[str, Any] | None, *, config: Mapping[str, Any]) -> None:
    """Require the summary's stage-specific terminal evidence shape and, with a last checkpoint, its equality to the adapter state.

    The stage comes from the config: B's summary and checkpoint both carry the curriculum, A0's and A1's carry the key in neither.
    A stage that ended before its first committed group has no last checkpoint, so only its summary's shape is checked.
    """
    self_fed = _is_self_fed(config)
    if not self_fed and "terminal_curriculum" in summary:
        message = "A Stage-A summary records no curriculum: only Stage B draws rollout horizons."
        raise RunLifecycleError(message)
    terminal = summary.get("terminal_controller")
    if not isinstance(terminal, Mapping) or set(terminal) != set(_TERMINAL_CONTROLLER_KEYS):
        message = "Transient summary terminal controller evidence is invalid."
        raise RunLifecycleError(message)
    terminal_curriculum = summary.get("terminal_curriculum")
    if self_fed and (not isinstance(terminal_curriculum, Mapping) or set(terminal_curriculum) != set(_TERMINAL_CURRICULUM_KEYS)):
        message = "A Stage-B summary lacks a valid terminal curriculum."
        raise RunLifecycleError(message)
    if last_checkpoint is None:
        return
    adapter_state = last_checkpoint.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping) or not isinstance(adapter_state.get("controller"), Mapping):
        message = "Finished transient last checkpoint lacks controller evidence."
        raise RunLifecycleError(message)
    controller = adapter_state["controller"]
    if any(terminal[key] != controller.get(key) for key in _TERMINAL_CONTROLLER_KEYS):
        message = "Finished transient summary terminal controller mismatch."
        raise RunLifecycleError(message)
    if not self_fed:
        if "curriculum" in adapter_state:
            message = "A finished Stage-A run records no curriculum in its last checkpoint: only Stage B draws rollout horizons."
            raise RunLifecycleError(message)
        return
    curriculum = adapter_state.get("curriculum")
    if not isinstance(curriculum, Mapping):
        message = "Finished Stage-B last checkpoint lacks curriculum evidence."
        raise RunLifecycleError(message)
    if terminal_curriculum != {key: curriculum.get(key) for key in _TERMINAL_CURRICULUM_KEYS}:
        message = "Finished transient summary terminal curriculum mismatch."
        raise RunLifecycleError(message)


def validate_completed_run(run_dir: Path | str) -> dict[str, Any]:
    """
    Validate and return the completed/loadable run contract.

    Both best and last checkpoints are schema- and identity-validated. This is
    the common gate used by inference and normal artifact generation.

    Parameters
    ----------
    run_dir : Path | str
        Candidate completed run leaf.

    Returns
    -------
    dict[str, Any]
        Validated summary, config, split/normalizer state, checkpoint identity,
        and distinct best/last checkpoint payloads.

    Raises
    ------
    FileNotFoundError
        If a required current run artifact is absent.
    RunLifecycleError
        If status, schema, digests, identity, progress, or checkpoint roles disagree.

    Notes
    -----
    Validation is read-only and loads checkpoints on CPU. It does not acquire a
    writer lease, initialize a runtime device, or repair incompatible artifacts.
    A finished Optuna trial (its saved config records ``run.layout.trial``) whose
    checkpoints were deleted with its final state is accepted without them and
    returns no checkpoint payloads or checkpoint identity.

    """
    return _validate_completed_run(run_dir, purpose="execution")


def validate_finished_run(run_dir: Path | str) -> dict[str, Any]:
    """
    Admit a run whose training ended: completed with its selected checkpoint, or unsuccessful without one.

    A completed run is validated in full (:func:`validate_completed_run`). An unsuccessful
    run has no selected checkpoint; its last checkpoint, bound by digest, is validated like a
    completed run's (saved data, identity, read-only CPU load) and its budget must equal the
    summary's terminal budget. Only a stage that committed no optimizer group (a divergence in
    its first group, or an A1 matched to such a B) has none; a transient stage's terminal evidence
    keeps its stage's shape either way. Both return the terminal budget
    their training outcome was admitted against (:func:`validate_training_outcome`). Matched B
    admission uses it, because B's receipt exists for every end of B.
    """
    path = Path(run_dir)
    summary = read_run_summary(path) if common.paths.resolve_run_summary_path(path).is_file() else {}
    if summary.get("status") != "unsuccessful":
        return validate_completed_run(path)
    config = config_loader.validate_resolved_config(config_loader.load_yaml(common.paths.resolve_run_config_path(path)))
    last = common.paths.resolve_last_checkpoint_file(path)
    digest = summary.get("last_checkpoint_sha256")
    saved_budget: Mapping[str, Any] | None = None
    last_payload: Mapping[str, Any] | None = None
    if digest is not None:
        # The recorded digest is compared before the checkpoint is deserialized.
        if not last.is_file() or digest != common.serialization.cached_file_sha256(last):
            message = f"Unsuccessful run last checkpoint digest mismatch: {path}"
            raise RunLifecycleError(message)
        _split_indices, _normalizer_state, identity = _saved_run_identity(path, config, checkpoint_path=last, purpose="execution")
        last_payload = _load_last_checkpoint(path, config, identity=identity, require_best=False)
        saved_budget = last_payload["budget_state_dict"]
        # The checkpoint records the stage's end too: its divergence and the eligible points of its objective history.
        outcome = summary.get("training_outcome")
        eligible_points = sum(entry["value"] is not None for entry in last_payload["objective_history"])
        if not isinstance(outcome, Mapping) or (outcome.get("divergence"), outcome.get("eligible_points")) != (
            last_payload["divergence"],
            eligible_points,
        ):
            message = f"Unsuccessful run training outcome disagrees with the divergence or eligible points of its last checkpoint: {path}"
            raise RunLifecycleError(message)
    # The terminal evidence binds a matched stage's receipt and handoff: its shape always, the last checkpoint's state when one exists.
    if config["task"] == "transient_drying":
        _validate_transient_terminal_state(summary, last_payload, config=config)
        _validate_unsuccessful_history(path, config, last_payload)
    budget = _validate_terminal_budget(summary, saved_budget, lifecycle="unsuccessful", target=_finished_stage_target(path, config, summary))
    if (saved_budget is None) != (budget.spent == 0) or (saved_budget is None and last.exists()) or summary.get("best_epoch") is not None:
        message = f"Unsuccessful run evidence is inconsistent: {path}"
        raise RunLifecycleError(message)
    return {"run_dir": path, "summary": dict(summary), "config": dict(config), "terminal_budget": budget}


def _validate_unsuccessful_history(path: Path, config: Mapping[str, Any], last_payload: Mapping[str, Any] | None) -> None:
    """
    Require an unsuccessful transient run's epoch history to hold exactly the records its producer wrote.

    With a last checkpoint these are the epochs it counts (:func:`recorded_history_epochs`). A stage
    that committed no group wrote none: its history is absent, or empty when a restart reset the
    record of an attempt that crashed before its first checkpoint.
    """
    if last_payload is not None:
        identity, epochs = last_payload["identity"], learning.training.checkpoint.recorded_history_epochs(last_payload)
    elif transient_adapter_history.history_path(path).exists():
        last = common.paths.resolve_last_checkpoint_file(path)
        identity, epochs = _saved_run_identity(path, config, checkpoint_path=last, purpose="execution")[2], 0
    else:
        return
    try:
        transient_adapter_history.validate_completed_history(
            path, task=str(config["task"]), run_name=str(config["run"]["name"]), checkpoint_identity=identity, completed_epoch=epochs
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        message = f"Unsuccessful transient history admission failed: {error}"
        raise RunLifecycleError(message) from error


def _validate_trial_summary_evidence(summary: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Check the summary evidence a finished trial keeps in place of its deleted checkpoints."""
    expected = {
        "task": config["task"],
        "run_name": config["run"]["name"],
        "objective": config["evaluation"]["objective"],
        "best_checkpoint": common.paths.RUN_BEST_CHECKPOINT_FILENAME,
    }
    digest = summary.get("best_checkpoint_sha256")
    completed_epoch, best_epoch, global_step = summary.get("completed_epoch"), summary.get("best_epoch"), summary.get("global_step")
    metric = summary.get("best_metric")
    if (
        any(summary.get(label) != value for label, value in expected.items())
        or not (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest))
        or type(completed_epoch) is not int
        or completed_epoch < 1
        or type(best_epoch) is not int
        or not 1 <= best_epoch <= completed_epoch
        or type(global_step) is not int
        or global_step < 0
        or not (isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric))
    ):
        msg = (
            "Completed trial summary must record its task, run name, objective, its best checkpoint digest, "
            "completed progress with a best epoch inside it, and a finite best metric."
        )
        raise RunLifecycleError(msg)


def _validate_completed_run(
    run_dir: Path | str,
    *,
    purpose: Literal["execution", "saved_artifacts"],
) -> dict[str, Any]:
    """Validate shared completed evidence for execution or saved artifact reading."""
    path = Path(run_dir)
    _require_completed_status(path)
    checkpoints = (common.paths.resolve_best_checkpoint_file(path), common.paths.resolve_last_checkpoint_file(path))
    # A finished Optuna trial writes no last and deletes its best; its summary keeps the best digest. Any other run, and a
    # trial bundle that holds a last checkpoint, requires both and is validated in full.
    trial = _is_trial_run(path) and not checkpoints[1].exists()
    missing = tuple(item for item in common.paths.missing_current_run_files(path) if not (trial and item in checkpoints))
    if missing:
        names = ", ".join(item.name for item in missing)
        msg = f"Run is incomplete and not loadable: {path}. Missing: {names}."
        raise RunLifecycleError(msg)
    summary = read_run_summary(path)
    # Checkpoints are deserialized with full pickle support, so their recorded digests are compared before any load.
    if trial:
        # A kill between the completed summary and the best deletion leaves the best behind; it must be the recorded one.
        if checkpoints[0].exists() and summary.get("best_checkpoint_sha256") != common.serialization.cached_file_sha256(checkpoints[0]):
            msg = "Completed trial best checkpoint digest mismatch."
            raise RunLifecycleError(msg)
    else:
        for role, checkpoint in zip(("best", "last"), checkpoints, strict=True):
            if summary.get(f"{role}_checkpoint_sha256") != common.serialization.cached_file_sha256(checkpoint):
                msg = f"Completed run {role} checkpoint digest mismatch."
                raise RunLifecycleError(msg)
    config_path = common.paths.resolve_run_config_path(path)
    split_path = common.paths.resolve_split_indices_path(path)
    normalizer_path = common.paths.resolve_normalizer_path(path)
    config = config_loader.validate_resolved_config(config_loader.load_yaml(config_path))
    split_indices, normalizer_state, identity = _saved_run_identity(
        path, config, checkpoint_path=common.paths.resolve_best_checkpoint_file(path), purpose=purpose
    )
    target = _finished_stage_target(path, config, summary)
    expected_file_digests = {
        "config_sha256": common.serialization.file_sha256(config_path),
        "split_indices_sha256": common.serialization.file_sha256(split_path),
        "normalizer_sha256": common.serialization.file_sha256(normalizer_path),
    }
    for label, expected_digest in expected_file_digests.items():
        if summary.get(label) != expected_digest:
            msg = f"Completed run {label} mismatch."
            raise RunLifecycleError(msg)
    if trial:
        _validate_trial_summary_evidence(summary, config)
        terminal_budget = _validate_terminal_budget(summary, None, lifecycle="completed", target=target)
        if config["task"] == "transient_drying":
            _validate_transient_summary_evidence(
                path,
                summary=summary,
                config=config,
                normalizer_state=normalizer_state,
                checkpoint_identity=identity,
                completed_epoch=int(summary["completed_epoch"]),
            )
        if summary.get("effective_config_digest") != identity["effective_config_digest"]:
            msg = "Completed trial summary/config identity mismatch."
            raise RunLifecycleError(msg)
        return {
            "run_dir": path,
            "summary": dict(summary),
            "config": dict(config),
            "split_indices": dict(split_indices),
            "scientific_run_name": str(config["run"]["name"]),
            "storage_alias": path.name,
            "effective_config_digest": identity["effective_config_digest"],
            "terminal_budget": terminal_budget,
        }
    scheduler_expected = config.get("scheduler") is not None
    best = learning.training.checkpoint.load_checkpoint(
        common.paths.resolve_best_checkpoint_file(path),
        expected_identity=identity,
        expected_role="best",
        payload_kind=learning.training.checkpoint.best_payload_kind(config),
        scheduler_expected=scheduler_expected,
        require_best=True,
        adapter_expected=config["task"] == "transient_drying",
    )
    last = _load_last_checkpoint(path, config, identity=identity, require_best=True)
    saved_identities = (best["identity"], last["identity"])
    if saved_identities[0] != saved_identities[1] or summary.get("effective_config_digest") != saved_identities[0]["effective_config_digest"]:
        message = "Completed run summary/config/checkpoint identity mismatch."
        raise RunLifecycleError(message)
    if summary.get("best_metric") != best["best_metric"] or summary.get("best_epoch") != best["best_epoch"]:
        msg = "Completed run summary disagrees with best checkpoint objective state."
        raise RunLifecycleError(msg)
    if last["best_metric"] != best["best_metric"] or last["best_epoch"] != best["best_epoch"]:
        msg = "Completed run best and last checkpoints disagree."
        raise RunLifecycleError(msg)
    if config["task"] == "transient_drying":
        _validate_transient_summary_evidence(
            path,
            summary=summary,
            config=config,
            normalizer_state=normalizer_state,
            checkpoint_identity=best["identity"],
            completed_epoch=int(last["completed_epoch"]),
        )
        _validate_transient_terminal_state(summary, last, config=config)
    terminal_budget = _validate_terminal_budget(summary, last["budget_state_dict"], lifecycle="completed", target=target)
    expected_summary_values = {
        "task": config.get("task"),
        "run_name": config.get("run", {}).get("name") if isinstance(config.get("run"), Mapping) else None,
        "objective": config.get("evaluation", {}).get("objective") if isinstance(config.get("evaluation"), Mapping) else None,
        "best_checkpoint": common.paths.RUN_BEST_CHECKPOINT_FILENAME,
        "last_checkpoint": common.paths.RUN_LAST_CHECKPOINT_FILENAME,
        "completed_epoch": last["completed_epoch"],
        "global_step": last["global_step"],
    }
    for label, expected_value in expected_summary_values.items():
        if summary.get(label) != expected_value:
            msg = f"Completed run summary {label!r} mismatch."
            raise RunLifecycleError(msg)
    return {
        **_evaluable_run_result(
            path=path,
            summary=summary,
            config=config,
            split_indices=split_indices,
            normalizer_state=normalizer_state,
            checkpoint_identity=best["identity"],
            best_checkpoint=best,
        ),
        "last_checkpoint": last,
        "terminal_budget": terminal_budget,
    }


def _resume_runtime_config(saved_config: Mapping[str, Any], requested: Mapping[str, Any], *, run_dir: Path) -> dict[str, Any]:
    """Return the saved config with the runtime-only values a resume may take from the request."""
    runtime_config = copy.deepcopy(dict(saved_config))
    if saved_config["run"]["layout"]["stage"] != "single":
        runtime_config["paths"]["output_root"] = str(common.paths.resolve_config_output_root(saved_config, run_dir))
    runtime_config["run"]["device"] = requested["run"]["device"]
    # Execution only (outside identity): the attempt splits its updates as requested; the summary records it.
    runtime_config["training"]["memory_batch"] = requested["training"]["memory_batch"]
    return runtime_config


def _run_resolved_experiment(
    requested: dict[str, Any],
    *,
    config_path: Path | str,
    resume: Path | str | None = None,
    output_root: Path | str | None = None,
    device_resolution: learning.device.DeviceResolution | None = None,
    matched_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve and execute a fresh experiment or continue an interrupted one.

    Config, requested device, and concrete device requirements
    are validated before fresh output allocation. Resume acquires the run writer
    lease before reading mutable lifecycle artifacts and restores only the
    identity-validated last checkpoint.

    Parameters
    ----------
    requested : dict[str, Any]
        Fully resolved ordinary single-run configuration.
    config_path : pathlib.Path | str
        Semantic request source used only for lifecycle diagnostics.
    device_resolution : learning.device.DeviceResolution | None, optional
        Pre-resolved concrete device shared by a higher-level sequencer when supplied.
    matched_receipt : Mapping[str, Any] | None, optional
        Exact immutable B receipt admitted before matched A1 execution.
    resume : pathlib.Path | str | None, optional
        Existing run directory explicitly continued from ``last_checkpoint.pt``.
        ``None`` requires exclusive allocation of a new run leaf.
    output_root : pathlib.Path | str | None, optional
        Fresh-run destination override. On resume it must resolve to the exact
        saved task/run leaf. Dataset roots remain unchanged.

    Returns
    -------
    dict[str, Any]
        Exact ``run_dir``, completed training-loop ``result``, and the immutable
        ``device_resolution`` used by training.

    Raises
    ------
    FileNotFoundError
        If the YAML, the saved-config resume leaf, or a required resume artifact is absent.
    FileExistsError
        ``ExistingRunAdmissionError`` if an existing destination cannot continue, or
        ``IdenticalExperimentError`` if another directory already owns the experiment.
    config_loader.ConfigError
        If YAML or an override violates the semantic experiment schema.
    learning.device.DeviceResolutionError
        If runtime policy is invalid or strict CUDA is unusable.
    RunLifecycleError
        If a writer lease is active or saved status, data, checkpoint identity,
        best/last roles, or lifecycle history is not resumable.
    ValueError
        If the requested identity differs from the saved config, the run already
        completed, or an output-root override identifies another leaf.

    Notes
    -----
    Fresh execution allocates first, publishes immutable config/split/normalizer
    inputs, then trains under one writer lease. Resume preserves those inputs and
    the full identity, permits only execution changes, and validates best
    versus last checkpoint roles before mutation. ``best_checkpoint.pt`` remains
    the inference/artifact source. ``last_checkpoint.pt`` is the sole continuation
    source.

    Training or admission failures publish failed/interrupted state best-effort
    before the original exception propagates. Local lifecycle files remain
    authoritative, while any failure in an explicitly requested online or offline
    W&B session fails the operation after durable local evidence is published.

    """
    fresh_destination: Path | None = None
    finished: dict[str, Any] | None = None
    if resume is None:
        if output_root is not None:
            requested["paths"]["output_root"] = str(Path(output_root).expanduser())
        destination = common.paths.resolve_config_run_dir(requested)
        if destination.exists() or destination.is_symlink():
            finished = _existing_run_continuation(destination, requested, config_path=config_path)
            if finished is None:
                resume = destination
                output_root = None  # Already applied to the requested layout, which located this leaf.
        else:
            run_registry.ensure_available(requested, destination)
            fresh_destination = destination

    if device_resolution is None:
        device_resolution = learning.device.resolve_device(
            requested["run"]["device"],
            path="run.device",
        )
    validate_deterministic_model_device_policy(requested, device_resolution)
    if finished is not None:
        return {**finished, "device_resolution": device_resolution}
    if resume is None:
        if fresh_destination is None:
            msg = "Fresh-run destination was not resolved before runtime admission."
            raise AssertionError(msg)
        summary_identity = {"run_identity": run_identity.run_identity_evidence(requested, config_path=config_path)}
        with run_writer_lease(fresh_destination):
            run_dir = _prepare_fresh_run_locked(
                requested,
                run_dir=fresh_destination,
                summary_extra=summary_identity,
            )
            result = _execute_prepared_run_locked(
                requested,
                run_dir=run_dir,
                persisted_config=requested,
                summary_extra=summary_identity,
                device_resolution=device_resolution,
                matched_receipt=matched_receipt,
            )
        return {"run_dir": run_dir, "result": result, "device_resolution": device_resolution, "task": str(requested["task"])}

    run_dir = Path(resume).expanduser().resolve()
    if not run_dir.is_dir():
        msg = f"Resume run directory not found: {run_dir}"
        raise FileNotFoundError(msg)
    if common.paths.resolve_run_summary_path(run_dir).is_file():
        # A finished run is refused by a read before the writer lease, which creates its lock file; it is checked again under the lease.
        _require_resumable_status(read_run_summary(run_dir), run_dir)
    with run_writer_lease(run_dir):
        # The one admission point of every continuation (single, matched stage, saved-config resume); the binding module imports this one.
        from . import experiments_run_source_binding as source_binding  # noqa: PLC0415

        source_binding.admit_continuation((run_dir,))
        missing = common.paths.missing_resume_run_files(run_dir)
        if missing:
            names = ", ".join(path.name for path in missing)
            msg = f"Resume run is incomplete: {run_dir}. Missing: {names}."
            raise RunLifecycleError(msg)
        summary = read_run_summary(run_dir)
        _require_resumable_status(summary, run_dir)
        _require_attempt_history(summary, run_dir)
        saved_config = config_loader.validate_resolved_config(config_loader.load_yaml(common.paths.resolve_run_config_path(run_dir)))
        _require_current_training(saved_config, unit=f"Run {run_dir}")
        validate_resume_config(requested, saved_config)
        _validate_resume_output_root(run_dir, saved_config, output_root)
        runtime_config = _resume_runtime_config(saved_config, requested, run_dir=run_dir)

        split, normalizer_artifact = _read_saved_data_artifacts(run_dir)
        split_indices = split.as_payload()
        _validate_saved_data_contract(saved_config, split, normalizer_artifact)
        identity = learning.training.checkpoint.build_checkpoint_identity(
            runtime_config,
            split_indices,
            normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
            persisted_config=saved_config,
        )
        last = learning.training.checkpoint.load_checkpoint(
            common.paths.resolve_last_checkpoint_file(run_dir),
            expected_identity=identity,
            expected_role="last",
            payload_kind="full",
            scheduler_expected=runtime_config.get("scheduler") is not None,
            require_best=False,
            adapter_expected=runtime_config["task"] == "transient_drying",
        )
        best_path = common.paths.resolve_best_checkpoint_file(run_dir)
        if last["best_metric"] is None:
            if best_path.exists():
                msg = "Resume run contains a best checkpoint that is inconsistent with last_checkpoint.pt."
                raise RunLifecycleError(msg)
        else:
            best = learning.training.checkpoint.load_checkpoint(
                best_path,
                expected_identity=identity,
                expected_role="best",
                payload_kind=learning.training.checkpoint.best_payload_kind(runtime_config),
                scheduler_expected=runtime_config.get("scheduler") is not None,
                require_best=True,
                adapter_expected=runtime_config["task"] == "transient_drying",
            )
            if best["best_metric"] != last["best_metric"] or best["best_epoch"] != last["best_epoch"]:
                msg = "Resume run best and last checkpoints disagree about selected objective state."
                raise RunLifecycleError(msg)

        result = _execute_prepared_run_locked(
            runtime_config,
            run_dir=run_dir,
            persisted_config=saved_config,
            saved_split_indices=split_indices,
            restored_scaling_artifact=normalizer_artifact,
            resume_from=common.paths.resolve_last_checkpoint_file(run_dir),
            device_resolution=device_resolution,
            matched_receipt=matched_receipt,
        )
        return {
            "run_dir": run_dir,
            "result": result,
            "device_resolution": device_resolution,
            "task": str(requested["task"]),
        }


def _require_resumable_status(summary: Mapping[str, Any], run_dir: Path) -> None:
    """Admit only an interrupted or running run to resume; a finished run is never extended."""
    status = summary.get("status")
    if status == "completed":
        msg = "A completed run has spent its budget and is never extended; another budget is another run."
        raise ValueError(msg)
    if status not in {"running", "interrupted"}:
        msg = f"Run status {status!r} is not resumable: {run_dir}"
        raise RunLifecycleError(msg)


def _existing_run_continuation(
    run_dir: Path,
    requested: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> dict[str, Any] | None:
    """Decide how a start that names an existing run proceeds.

    Returns the finished outcome when the run already completed its budget,
    ``None`` when it must continue in place after an interruption, and raises
    when it cannot continue.

    Raises
    ------
    ExistingRunAdmissionError
        If the existing run holds different science, is written by another
        process or is in a state that cannot continue.

    """
    report = inspect_existing_run_admission(run_dir, requested, config_path=config_path)
    compatibility = report["resume_compatibility"]
    if compatibility in {"completed", "compatible"}:
        run_registry.ensure_bound(requested, run_dir)
    if compatibility == "completed":
        summary = read_run_summary(run_dir)
        # The run's one result is its saved outcome: a divergence with a best checkpoint stays a divergence.
        training_outcome = summary.get("training_outcome")
        if not isinstance(training_outcome, Mapping) or training_outcome.get("status") not in _LIFECYCLE_TRAINING_OUTCOMES["completed"]:
            msg = f"Completed run {run_dir} has no valid training_outcome."
            raise RunLifecycleError(msg)
        console.run_admission_event("already_complete", completed_epoch=report["completed_epoch"], run_dir=str(run_dir))
        return {
            "run_dir": run_dir.resolve(),
            "result": {
                "status": training_outcome["status"],
                "training_outcome": dict(training_outcome),
                "best_epoch": summary.get("best_epoch"),
                "best_metric": summary.get("best_metric"),
            },
            "task": str(requested["task"]),
            "already_complete": True,
        }
    if compatibility == "compatible":
        return None
    raise reject_existing_fresh_run(run_dir, requested, config_path=config_path)


def _apply_device_override(raw: dict[str, Any], device: str | None) -> None:
    """Apply one CLI device policy before authored-plan or single-run resolution."""
    if device is None:
        return
    run = raw.get("run")
    if run is None:
        run = {}
        raw["run"] = run
    if not isinstance(run, dict):
        message = "run must be a mapping before applying --device."
        raise config_loader.ConfigError(message)
    run["device"] = device


def _with_output_root(config: Mapping[str, Any], output_root: Path | str | None) -> dict[str, Any]:
    """Return an isolated resolved config with an optional fresh-output root override."""
    resolved = copy.deepcopy(dict(config))
    if output_root is not None:
        resolved["paths"]["output_root"] = str(Path(output_root).expanduser())
    return resolved


def _stage_destination(config: Mapping[str, Any]) -> Path:
    """Return the deterministic leaf for one resolved transient stage."""
    return common.paths.resolve_config_run_dir(config)


def _validate_reusable_stage_a(
    run_dir: Path,
    *,
    requested_a: Mapping[str, Any],
    requested_b: Mapping[str, Any],
    device_resolution: learning.device.DeviceResolution,
) -> None:
    """
    Admit a completed A leaf only when its training code is current and config and immutable handoff match the plan.

    A diverged A0 with its best eligible checkpoint is completed and branches like one at its budget end;
    an A0 without an eligible checkpoint is ``unsuccessful`` and refused by :func:`validate_completed_run`.
    """
    completed = validate_completed_run(run_dir)
    saved = config_loader.validate_resolved_config(completed["config"])
    _require_current_training(saved, unit=f"A0 {run_dir}")
    validate_resume_config(requested_a, saved)
    transient_handoff.validate_stage_a_handoff(
        run_dir / transient_handoff.HANDOFF_DIRECTORY,
        target_config=requested_b,
        device=device_resolution.as_dict(),
        expected_source_run_name=str(requested_a["run"]["name"]),
    )


def _matched_failed_stage_recovery(
    run_dir: Path,
    requested: Mapping[str, Any],
    *,
    config_path: Path | str,
) -> tuple[bool, str]:
    """Check the persisted resume admission and checkpoint without mutating the leaf."""
    if not run_dir.is_dir() or run_dir.is_symlink():
        return False, "No safe persisted stage leaf is available for checkpoint resume."
    try:
        report = inspect_existing_run_admission(run_dir, requested, config_path=config_path)
        if report["resume_compatibility"] != "compatible":
            return False, str(report["reason"])
        saved_config = config_loader.validate_resolved_config(config_loader.load_yaml(common.paths.resolve_run_config_path(run_dir)))
        split, normalizer_artifact = _read_saved_data_artifacts(run_dir)
        _validate_saved_data_contract(saved_config, split, normalizer_artifact)
        identity = learning.training.checkpoint.build_checkpoint_identity(
            requested,
            split.as_payload(),
            normalizer_sha256=common.serialization.file_sha256(common.paths.resolve_normalizer_path(run_dir)),
            persisted_config=saved_config,
        )
        checkpoint = learning.training.checkpoint
        last = checkpoint.load_checkpoint(
            common.paths.resolve_last_checkpoint_file(run_dir),
            expected_identity=identity,
            expected_role="last",
            payload_kind="full",
            scheduler_expected=requested.get("scheduler") is not None,
            require_best=False,
            adapter_expected=True,
        )
        best_path = common.paths.resolve_best_checkpoint_file(run_dir)
        if last["best_metric"] is None:
            if best_path.exists():
                return False, "Best checkpoint conflicts with the last checkpoint."
        else:
            best = checkpoint.load_checkpoint(
                best_path,
                expected_identity=identity,
                expected_role="best",
                payload_kind=checkpoint.best_payload_kind(requested),
                scheduler_expected=requested.get("scheduler") is not None,
                require_best=True,
                adapter_expected=True,
            )
            if best["best_metric"] != last["best_metric"] or best["best_epoch"] != last["best_epoch"]:
                return False, "Best and last checkpoints disagree about the selected objective."
    except Exception:  # noqa: BLE001
        return False, "Persisted resume state or checkpoint validation failed."
    return True, "Persisted lifecycle, data state, and checkpoint passed read-only resume checks."


def _run_transient_matched_plan(  # noqa: C901, PLR0912
    raw: dict[str, Any],
    *,
    config_path: Path | str,
    output_root: Path | str | None,
    train_stages: tuple[str, ...],
) -> dict[str, Any]:
    """Run selected A0, B, A1 leaves in order with exact prerequisite admission."""
    plan = transient_plan.resolve_matched_a1_b_plan(
        raw,
        source=config_naming.config_source_from_path(config_path),
        output_root=output_root,
    )
    configs = {name: _with_output_root(plan.stage(name), output_root) for name in ("a0", "b", "a1")}
    dirs = {name: _stage_destination(config) for name, config in configs.items()}
    selected = set(train_stages)
    if not selected or selected.difference(dirs) or len(selected) != len(train_stages):
        msg = "--train-stages must select one or more distinct a0,b,a1 stages."
        raise ValueError(msg)
    device_resolution = learning.device.resolve_device(configs["a0"]["run"]["device"], path="run.device")
    for config in configs.values():
        validate_deterministic_model_device_policy(config, device_resolution)
    record = run_identity.build_transient_experiment_record(configs["a0"], configs["b"], stage_a1=configs["a1"], config_path=config_path)
    record_path = run_identity.experiment_record_path(record, output_root=configs["a0"]["paths"]["output_root"])

    # Reject incompatible existing leaves and absent unselected dependencies before any allocation.
    for name, run_dir in dirs.items():
        if run_dir.exists() or run_dir.is_symlink():
            saved_path = common.paths.resolve_run_config_path(run_dir)
            if run_dir.is_symlink() or not saved_path.is_file():
                msg = f"Matched stage {name} has an unsafe or incomplete config: {run_dir}"
                raise RunLifecycleError(msg)
            saved = config_loader.validate_resolved_config(config_loader.load_yaml(saved_path))
            _require_current_training(saved, unit=f"Matched stage {name} {run_dir}")
            if name == "a0" and saved["run"]["layout"]["a0_id"] != configs["a0"]["run"]["layout"]["a0_id"]:
                msg = (
                    f"The existing A0 at {run_dir} has another scientific identity than the requested plan. "
                    "Train the changed science with --revision <label>."
                )
                raise RunLifecycleError(msg)
            validate_resume_config(configs[name], saved)
    if record_path.exists():
        run_identity.validate_persisted_transient_experiment_record(record, output_root=configs["a0"]["paths"]["output_root"])
    elif any(dirs[name].exists() for name in ("b", "a1")):
        msg = "Matched children exist without their exact parent experiment record."
        raise RunLifecycleError(msg)
    outcomes: dict[str, dict[str, Any]] = {}
    a0_handoff_path = dirs["a0"] / transient_handoff.HANDOFF_DIRECTORY / "manifest.json"

    def admitted_a0() -> None:
        _validate_reusable_stage_a(dirs["a0"], requested_a=configs["a0"], requested_b=configs["b"], device_resolution=device_resolution)

    def admitted_receipt() -> dict[str, Any]:
        admitted_a0()
        finished_b = validate_finished_run(dirs["b"])
        validate_resume_config(configs["b"], finished_b["config"])
        receipt = matched_budget.build_b_receipt(
            parent_identity_sha256=record["parent_identity_sha256"],
            a0_handoff_sha256=common.serialization.file_sha256(a0_handoff_path),
            b_config=configs["b"],
            finished_b=finished_b,
        )
        matched_budget.publish_b_receipt(dirs["b"], receipt)
        return matched_budget.load_b_receipt(
            dirs["b"] / matched_budget.RECEIPT_NAME,
            parent_identity_sha256=record["parent_identity_sha256"],
            a0_handoff_sha256=common.serialization.file_sha256(a0_handoff_path),
            b_run_name=configs["b"]["run"]["name"],
        )

    if "b" in selected and "a0" not in selected:
        admitted_a0()
    if "a1" in selected and "b" not in selected:
        try:
            admitted_receipt()
        except KeyboardInterrupt as error:
            msg = "a1"
            raise MatchedStageInterrupted(
                msg,
                dirs["a1"],
                resumable=False,
                reason="A0/B prerequisite admission was interrupted before A1 execution.",
            ) from error
        except Exception as error:
            msg = "a1"
            raise MatchedStageExecutionError(
                msg,
                dirs["a1"],
                resumable=False,
                reason="A0/B prerequisite admission failed before A1 execution.",
            ) from error

    if not record_path.exists():
        record_path = run_identity.admit_fresh_transient_experiment_record(record, output_root=configs["a0"]["paths"]["output_root"])

    for name in ("a0", "b", "a1"):
        if name not in selected:
            continue
        run_dir = dirs[name]
        prerequisites_admitted = False
        try:
            receipt = None
            if name == "b":
                admitted_a0()
            if name == "a1":
                receipt = admitted_receipt()
            prerequisites_admitted = True
            if run_dir.exists():
                try:
                    completed: dict[str, Any] | None = validate_finished_run(run_dir)
                except (FileNotFoundError, RunLifecycleError):
                    completed = None
                if completed is None:
                    outcome = _run_resolved_experiment(
                        configs[name],
                        config_path=config_path,
                        resume=run_dir,
                        device_resolution=device_resolution,
                        matched_receipt=receipt,
                    )
                else:
                    outcome = {
                        "run_dir": run_dir,
                        "result": {
                            "status": completed["summary"]["training_outcome"]["status"],
                            "training_outcome": completed["summary"]["training_outcome"],
                            "best_epoch": completed["summary"]["best_epoch"],
                            "best_metric": completed["summary"]["best_metric"],
                        },
                        "device_resolution": device_resolution,
                        "task": "transient_drying",
                        "already_complete": True,
                    }
            else:
                outcome = _run_resolved_experiment(
                    configs[name],
                    config_path=config_path,
                    output_root=output_root,
                    device_resolution=device_resolution,
                    matched_receipt=receipt,
                )
        except (ExistingRunAdmissionError, SourceBindingError):
            raise
        except KeyboardInterrupt as error:
            if prerequisites_admitted:
                resumable, reason = _matched_failed_stage_recovery(run_dir, configs[name], config_path=config_path)
            else:
                resumable, reason = False, "Prerequisite admission was interrupted before this stage executed."
            raise MatchedStageInterrupted(name, run_dir, resumable=resumable, reason=reason) from error
        except Exception as error:
            if prerequisites_admitted:
                resumable, reason = _matched_failed_stage_recovery(run_dir, configs[name], config_path=config_path)
            else:
                resumable, reason = False, "Prerequisite admission failed before this stage executed."
            raise MatchedStageExecutionError(name, run_dir, resumable=resumable, reason=reason) from error
        outcomes[name] = outcome
        if name == "a0" and outcome["result"]["best_epoch"] is None:
            # Only an A0 without an eligible checkpoint is no branch point; a diverged A0 branches from its best eligible one.
            break
        if name == "a0":
            admitted_a0()
        if name == "b":
            admitted_receipt()

    last_name = next(reversed(outcomes))
    return {
        **outcomes[last_name],
        "stage_runs": dirs,
        "stage_outcomes": outcomes,
        "experiment_record": record_path,
        "training_status": outcomes[last_name]["result"]["status"],
        # Selected stages a stopped plan never started: the requested comparison was not trained.
        "unrun_stages": tuple(name for name in ("a0", "b", "a1") if name in selected and name not in outcomes),
    }


def _run_saved_effective_resume(
    config_path: Path | str,
    *,
    resume: Path | str,
    device: str | None,
    output_root: Path | str | None,
    train_stages: tuple[str, ...] | None,
) -> dict[str, Any]:
    """Continue one exact saved effective configuration through strict execution admission."""
    run_dir = Path(resume).expanduser().absolute()
    requested = config_loader.validate_resolved_config(config_loader.load_yaml(config_path))
    if requested["run"]["layout"]["trial"] is not None:
        msg = f"Optuna trials are not resumed: {run_dir}. Reopen the study instead; interrupted trials restart as fresh trials."
        raise RunLifecycleError(msg)
    # A direct bundle has no parent-relative dependencies: its explicit run_dir
    # remains authoritative after rename. Parent-linked stages retain layout
    # inversion so their shared A0 and immutable branch records stay contained.
    actual_root = Path(requested["paths"]["output_root"])
    if requested["run"]["layout"]["stage"] != "single":
        actual_root = common.paths.resolve_config_output_root(requested, run_dir)
        requested["paths"]["output_root"] = str(actual_root)
    _apply_device_override(requested, device)
    arm = requested["training"].get("comparison_arm")
    stage = str(requested["run"]["layout"]["stage"])
    if train_stages is not None and train_stages != (stage,):
        msg = "Saved-effective resume selects only its exact saved stage."
        raise ValueError(msg)
    device_resolution = learning.device.resolve_device(requested["run"]["device"], path="run.device")
    receipt = None
    stage_runs: dict[str, Path] = {stage: run_dir}
    marker = common.paths.resolve_config_branch_dir(requested, output_root=actual_root) / common.paths.EXPERIMENT_RECORD_FILENAME
    # Every pair shares the seed group's A0; its immutable record lives beside its own B/A1 leaves.
    if arm in {"b", "a1"}:
        admitted = run_identity.load_readonly_transient_experiment_record(marker, experiments_root=actual_root)
        record = admitted["record"]
        child = record["children"].get(f"stage_{arm}")
        if not isinstance(child, Mapping) or child["semantic_config_sha256"] != learning.training.checkpoint.config_digest(requested):
            msg = "Saved stage scientific identity disagrees with its immutable parent."
            raise RunLifecycleError(msg)
        stage_runs = {
            key.removeprefix("stage_"): run_identity.resolve_readonly_experiment_child(record, child_record, marker=marker, output_root=actual_root)
            for key, child_record in record["children"].items()
        }
        if stage_runs[arm].resolve(strict=False) != run_dir.resolve(strict=False):
            msg = "Saved stage does not identify its immutable parent child."
            raise RunLifecycleError(msg)
        saved_a0 = config_loader.validate_resolved_config(config_loader.load_yaml(common.paths.resolve_run_config_path(stage_runs["a0"])))
        prerequisite_b = (
            requested
            if arm == "b"
            else config_loader.validate_resolved_config(config_loader.load_yaml(common.paths.resolve_run_config_path(stage_runs["b"])))
        )
        _validate_reusable_stage_a(stage_runs["a0"], requested_a=saved_a0, requested_b=prerequisite_b, device_resolution=device_resolution)
        if arm == "a1":
            # Receipt admission preserves the current execution implementation,
            # finished B controller, common A0 checkpoint and exact handoff.
            finished_b = validate_finished_run(stage_runs["b"])
            expected = matched_budget.build_b_receipt(
                parent_identity_sha256=record["parent_identity_sha256"],
                a0_handoff_sha256=common.serialization.file_sha256(stage_runs["a0"] / transient_handoff.HANDOFF_DIRECTORY / "manifest.json"),
                b_config=finished_b["config"],
                finished_b=finished_b,
            )
            receipt = matched_budget.load_b_receipt(
                stage_runs["b"] / matched_budget.RECEIPT_NAME,
                parent_identity_sha256=record["parent_identity_sha256"],
                a0_handoff_sha256=expected["a0_handoff_sha256"],
                b_run_name=finished_b["config"]["run"]["name"],
            )
            if receipt != expected:
                msg = "Immutable B receipt disagrees with its completed controller evidence."
                raise RunLifecycleError(msg)
    outcome = _run_resolved_experiment(
        requested,
        config_path=config_path,
        resume=run_dir,
        output_root=output_root,
        device_resolution=device_resolution,
        matched_receipt=receipt,
    )
    if arm is not None:
        outcome["stage_runs"] = stage_runs
        outcome["stage_outcomes"] = {stage: dict(outcome)}
    return outcome


def run_experiment(
    config_path: Path | str,
    *,
    resume: Path | str | None = None,
    device: str | None = None,
    output_root: Path | str | None = None,
    train_stages: tuple[str, ...] | None = None,
    epochs: int | None = None,
    train_dataset: str | None = None,
    train_seconds: float | None = None,
    rollout_length: int | None = None,
    lr_scale: float | None = None,
    revision: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Resolve and execute one single-stage experiment or authored matched A0/B/A1 plan.

    Parameters
    ----------
    config_path : Path | str
        Authored single-stage experiment YAML or strict transient matched plan YAML.
    resume : Path | str | None, optional
        Explicit existing leaf to resume. Matched plans admit only a selected derived leaf.
    device : str | None, optional
        Device-policy override applied before configuration or plan resolution.
    output_root : Path | str | None, optional
        Fresh output root shared by all derived plan stages.
    train_stages : tuple[str, ...] | None, optional
        Selected matched stage leaves, or all stages by default.
    epochs : int | None, optional
        Single-stage epoch budget, or A0's epoch budget in a matched plan.
    train_dataset : str | None, optional
        Exact Dataset ID bound before scientific configuration resolution.
    train_seconds : float | None, optional
        Measured B target; A1 matches the resulting immutable receipt.
    rollout_length : int | None, optional
        Matched Stage-B rollout window length and stride override.
    lr_scale : float | None, optional
        Continuation learning-rate factor applied to both B and A1 for a paired sweep.
    revision : str | None, optional
        Label of a deliberately separate output location (``run.revision``) for otherwise identical science.
    seed : int | None, optional
        Replicate seed replacing ``run.seed`` before the layout and seed plan are derived.

    Returns
    -------
    dict[str, Any]
        Final single-run outcome or the last executed matched stage's outcome.

    """
    if resume is not None and Path(config_path).expanduser().resolve(strict=False) == common.paths.resolve_run_config_path(
        Path(resume).expanduser()
    ).resolve(strict=False):
        if any(value is not None for value in (epochs, train_dataset, train_seconds, rollout_length, lr_scale, revision, seed)):
            msg = "Saved-effective resume cannot override scientific settings; it continues its saved config exactly."
            raise ValueError(msg)
        return _run_saved_effective_resume(
            config_path,
            resume=resume,
            device=device,
            output_root=output_root,
            train_stages=train_stages,
        )
    if resume is not None:
        msg = (
            "--resume continues a saved run from its own saved config only. For an authored config, "
            "run the same command again without --resume; an existing run continues automatically."
        )
        raise ValueError(msg)
    raw_requested = config_loader.apply_training_overrides(
        config_loader.load_authored_config(config_path),
        epochs=epochs,
        train_dataset=train_dataset,
        train_seconds=train_seconds,
        rollout_length=rollout_length,
        lr_scale=lr_scale,
        revision=revision,
        seed=seed,
    )
    _apply_device_override(raw_requested, device)
    if transient_plan.is_matched_a1_b_config(raw_requested):
        return _run_transient_matched_plan(
            raw_requested,
            config_path=config_path,
            output_root=output_root,
            train_stages=train_stages or ("a0", "b", "a1"),
        )
    if train_stages is not None:
        msg = "--train-stages applies only to explicit matched_a1_b plans."
        raise ValueError(msg)
    requested = config_loader.resolve_config(raw_requested, source=config_naming.config_source_from_path(config_path))
    return _run_resolved_experiment(
        requested,
        config_path=config_path,
        output_root=output_root,
    )
