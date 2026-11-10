"""Find the frozen source tree in which an experiment unit last ran and guard continuing it.

Responsibilities:
  - List the existing run leaves a resolved train request or Optuna study would write into
  - Separate the leaves a job may still write from completed, read-only ones
  - Name the source tree of a runtime session from its recorded Slurm source commit and fingerprint
  - Select the tree of the newest Slurm-recorded session across a unit's leaves and study sessions, or none
  - Record the executing Slurm job's source binding in a study's own session file
  - Refuse continuing a leaf or study in another tree than the one it is bound to unless the job switched explicitly

Design principles:
  - The unit's own saved summaries and study session record are the only evidence; no scheduler, Git or source tree is consulted
  - Run leaves come from the saved layout resolvers, never from directory-name guesses
  - Sessions without Slurm source provenance (interactive runs) bind nothing
  - A study binds independently of its trials, so a study that never ran a trial keeps its tree

This module does NOT:
  - Materialize, verify or lease a source tree, or choose between the bound tree and the current code
  - Decide whether continuing with newer code is scientifically admissible
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from src import common

from . import experiments_run as run
from . import experiments_run_identity as run_identity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

STUDY_SESSIONS_FILENAME: Final = "study_sessions.json"
_RECORDED_SLURM_FIELDS: Final = ("job_id", "source_commit", "source_worktree_sha256", "source_transition")


def existing_train_run_dirs(configs: Iterable[Mapping[str, Any]]) -> tuple[Path, ...]:
    """Return the run leaves the resolved configs' layouts name that already exist.

    A steady run names its run directory; a matched transient plan names the shared
    A0 leaf and the pair's B and A1 leaves.
    """
    leaves = (common.paths.resolve_config_run_dir(config) for config in configs)
    return tuple(leaf for leaf in leaves if leaf.is_dir())


def study_dir(base_config: Mapping[str, Any]) -> Path:
    """Return the directory of the Optuna study a resolved base config names."""
    layout = base_config["run"]["layout"]
    return common.paths.resolve_study_dir(
        layout["task"],
        layout["dataset"],
        layout["config_name"],
        layout["budget_epochs"],
        layout["revision"],
        output_root=base_config["paths"]["output_root"],
    )


def existing_study_run_dirs(base_config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the existing trial run directories of the Optuna study a resolved base config names."""
    trials = study_dir(base_config) / common.paths.STUDY_TRIALS_DIRNAME
    if not trials.is_dir():
        return ()
    return tuple(sorted(path for path in trials.iterdir() if path.is_dir()))


def unfinished_run_dirs(run_dirs: Iterable[Path]) -> tuple[Path, ...]:
    """Return the leaves whose summary does not record the completed status.

    A completed leaf is read-only for the job: it is a prerequisite or an unchanged
    result, never written, and so never decides which source tree the job runs in.
    A leaf without a summary counts as unfinished.

    Raises
    ------
    RunLifecycleError
        If a leaf's summary is malformed.

    """
    unfinished: list[Path] = []
    for run_dir in run_dirs:
        try:
            completed = run.read_run_summary(run_dir).get("status") == "completed"
        except FileNotFoundError:
            completed = False
        if not completed:
            unfinished.append(run_dir)
    return tuple(unfinished)


def _slurm_sessions(raw_sessions: object, *, origin: Path) -> list[tuple[datetime, str]]:
    """Return start time and tree name of every Slurm-recorded session in recorded order."""
    if not isinstance(raw_sessions, list):
        msg = f"Recorded sessions must be a list: {origin}"
        raise run.RunLifecycleError(msg)
    sessions: list[tuple[datetime, str]] = []
    for session in raw_sessions:
        slurm = session.get("slurm") if isinstance(session, dict) else None
        if not isinstance(slurm, dict):
            continue
        started_at = session.get("started_at")
        if not isinstance(started_at, str):
            msg = f"Slurm session recorded in {origin} must record its started_at time."
            raise run.RunLifecycleError(msg)
        started = datetime.fromisoformat(started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        sessions.append((started, run_identity.source_tree_name(slurm.get("source_commit"), slurm.get("source_worktree_sha256"))))
    return sessions


def _leaf_sessions(run_dir: Path) -> list[tuple[datetime, str]]:
    """Return the Slurm sessions of one run leaf; a leaf without a summary has none."""
    try:
        summary = run.read_run_summary(run_dir)
    except FileNotFoundError:
        return []
    return _slurm_sessions(summary.get("runtime_sessions", []), origin=common.paths.resolve_run_summary_path(run_dir))


def _read_study_sessions(study: Path) -> dict[str, Any]:
    """Load the study's session record, or an empty one when no job recorded a binding yet."""
    path = study / STUDY_SESSIONS_FILENAME
    if not path.is_file():
        return {"sessions": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"Study session record is invalid JSON: {path}: {error}"
        raise run.RunLifecycleError(msg) from error
    if not (isinstance(payload, dict) and isinstance(payload.get("sessions"), list)):
        msg = f"Unsupported or malformed study session record: {path}"
        raise run.RunLifecycleError(msg)
    return payload


def bound_source_tree_id(run_dirs: Iterable[Path], *, study: Path | None = None) -> str | None:
    """
    Return the source tree the newest Slurm session of an experiment unit ran in.

    Parameters
    ----------
    run_dirs : Iterable[Path]
        The unit's existing run leaves; a leaf without a summary is skipped.
    study : Path | None, optional
        The unit's Optuna study directory, whose own session record counts like a
        leaf's sessions; a study without that record is skipped.

    Returns
    -------
    str | None
        ``<commit12>-<fingerprint12>`` of the session with the latest start time
        among all leaves and the study record (the later record wins a tie), or
        ``None`` when nothing recorded a Slurm session.

    Raises
    ------
    RunLifecycleError
        If a leaf's summary or the study session record is malformed.
    ValueError
        If a recorded session carries an invalid commit, fingerprint or start time.

    """
    recorded = [session for run_dir in run_dirs for session in _leaf_sessions(run_dir)]
    if study is not None:
        path = study / STUDY_SESSIONS_FILENAME
        recorded.extend(_slurm_sessions(_read_study_sessions(study)["sessions"], origin=path))
    newest: tuple[datetime, str] | None = None
    for started, tree_id in recorded:
        if newest is None or started >= newest[0]:
            newest = (started, tree_id)
    return None if newest is None else newest[1]


def record_study_session(study: Path) -> None:
    """Append the executing Slurm job's source binding to the study's session record.

    The record exists so that a study keeps its tree even when it never reached a
    trial session. Outside a Slurm ML job nothing is recorded. The file is replaced
    atomically; sessions already recorded are never changed.

    Raises
    ------
    RunLifecycleError
        If the job's provenance or an existing study session record is malformed.

    """
    provenance = run.slurm_runtime_provenance()
    if provenance is None:
        return
    payload = _read_study_sessions(study)
    session = {
        "session_id": uuid.uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "slurm": {key: provenance[key] for key in _RECORDED_SLURM_FIELDS},
    }
    payload["sessions"] = [*payload["sessions"], session]
    common.serialization.atomic_write_json(study / STUDY_SESSIONS_FILENAME, payload)


def admit_continuation(run_dirs: Iterable[Path], *, study: Path | None = None) -> None:
    """Refuse continuing a unit in another source tree than the one it is bound to.

    A unit without a Slurm-recorded session is unbound and always admitted. A bound
    unit is continued only by a Slurm ML job that runs in the bound tree, or in
    another tree after an explicit ``--continue-with-current-code`` switch (provenance
    ``source_transition`` ``explicit``). Call it before the first write, so the
    newest recorded session is still the previous job's.

    Raises
    ------
    SourceBindingError
        If the unit is bound to a tree this job does not run in and did not switch to explicitly.
    RunLifecycleError
        If the unit's recorded sessions are malformed.

    """
    bound = bound_source_tree_id(run_dirs, study=study)
    if bound is None:
        return
    kind = "run" if study is None else "study"
    provenance = run.slurm_runtime_provenance()
    if provenance is None:
        msg = (
            f"This {kind} is bound to source tree {bound}; continue it through scripts/slurm_ml.sh, which uses that tree, "
            "or through scripts/slurm_ml.sh with --continue-with-current-code to switch to the current code deliberately."
        )
        raise run.SourceBindingError(msg)
    current = run_identity.source_tree_name(provenance["source_commit"], provenance["source_worktree_sha256"])
    if current != bound and provenance["source_transition"] != "explicit":
        msg = (
            f"This {kind} is bound to source tree {bound}, but this job runs in source tree {current}; "
            "continue it through scripts/slurm_ml.sh, which uses the bound tree, "
            "or pass --continue-with-current-code to switch deliberately."
        )
        raise run.SourceBindingError(msg)
