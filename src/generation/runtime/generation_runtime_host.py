"""Own admitted host processes and scheduled commands for Generation.

Responsibilities:
  - Admit the shared source, committed or not: frozen as a read-only source tree for executing commands, fingerprinted only for read-only ones
  - Freeze the source of an existing snapshot again when a bound unit continues in its old code state
  - Resolve the storage, runtime, and native Python paths of a live checkout or of a frozen source tree
  - Invoke Generation service commands and submit native Slurm work from the admitted tree
  - Submit a launcher command to run from its chosen tree as its own Slurm job with an unbuffered live log and lease the tree for the returned job
  - Hard-link a job's own Slurm logs and its helper logs into the logs/ of their existing campaign

Design principles:
  - A log link never creates its campaign folder and never fails a job: every problem is one warning on stderr
  - A job finds its own logs from the sbatch templates that submitted it and keeps a link only to the file it checked
  - Keep source identity checks ahead of scheduled worker execution
  - A successful operation returning one JSON result prints a summary line to the driver log and leaves no file; other output stays raw
  - Every scheduled worker command, launcher, and child environment names the tree, never the live checkout
  - Uncommitted changes are always admitted: the snapshot captures them and the admission reports whether it carries any
  - Read-only commands never capture snapshots or trees, so inspection leaves no durable evidence behind
  - Freezing runs only on the login node: a Slurm allocation never captures or archives source through Git
  - A lease failure after a successful submission never loses the job: it warns, tells the caller, and the tree stays protected by its recency
  - Inside a tree no sibling root is inferred and Git is never called
  - Pass scheduler arguments as vectors without selecting physical hardware
  - Preserve worker-owned validation, simulation, and publication semantics

This module does NOT:
  - Interpret scientific configurations or persisted lifecycle state
  - Decide which tree an existing campaign continues in, read campaign manifests, or submit Generation node jobs
  - Run COMSOL cases directly on the login node
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from src import common
from src.generation.contracts import generation_contracts_source as source_contract

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


_DIGEST = re.compile(r"[0-9a-f]{64}")
_CLI_MODULE = "src.generation.cli.cli_generation"
_LAUNCHER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_LAUNCHER_MEMORY = re.compile(r"[1-9][0-9]*[KMGT]?")
_LAUNCHER_WALL_TIME = re.compile(r"(?:[0-9]+-)?[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]")
_LAUNCHER_PARTITIONS = ("standard", "long")
# The log directories below RUNTIME_ROOT/logs/ that launcher jobs write to: the Generation and the Dataset driver.
_LAUNCHER_LOG_AREAS = ("generation", "datasets")
_FREEZE_ON_LOGIN_NODE = "Freeze the source on the login node: start this command there; it submits its own Slurm work."
_EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class HostLayout:
    """Canonical paths and inherited environment for one live checkout or one source tree."""

    repository: Path
    storage: Path
    runtime: Path
    native_venv: Path
    environment: Mapping[str, str]

    @property
    def python(self) -> Path:
        """Return the native interpreter used for host Generation commands."""
        return self.native_venv / "bin/python"

    @property
    def logs(self) -> Path:
        """Return the replaceable Generation log directory."""
        return self.runtime / "logs/generation"

    @property
    def snapshots(self) -> Path:
        """Return the durable root of exact source snapshots."""
        return common.paths.get_source_snapshots_root(storage_root=self.storage)

    @property
    def trees(self) -> Path:
        """Return the runtime root of frozen read-only source trees."""
        return common.paths.get_source_trees_root(runtime_root=self.runtime)

    @classmethod
    def resolve(cls, repository: Path | str, *, environment: Mapping[str, str] | None = None) -> HostLayout:
        """
        Admit a physical repository and its storage and runtime roots.

        Parameters
        ----------
        repository : Path | str
            Explicit shared checkout, including a test-owned checkout when testing; in
            source-tree mode the tree named by ``SOURCE_TREE_ID`` under ``RUNTIME_ROOT``.
        environment : Mapping[str, str] | None, optional
            Host environment used for path overrides and child processes.

        Returns
        -------
        HostLayout
            Validated host paths and a snapshot of the inherited environment.

        """
        values = dict(os.environ if environment is None else environment)
        tree_id = common.paths.source_tree_id(values)
        candidate = Path(repository).absolute()
        if tree_id is not None and candidate.is_symlink():
            message = f"Generation source tree must be a real directory: {candidate}"
            raise RuntimeError(message)
        if tree_id is None and (candidate.is_symlink() or not (candidate / ".git").is_dir()):
            message = f"Generation requires the current shared Git repository: {candidate}"
            raise RuntimeError(message)
        root = candidate.resolve(strict=True)
        if root != candidate:
            message = f"Generation repository must be a physical path: {candidate}"
            raise RuntimeError(message)
        if tree_id is None:
            storage, runtime = _live_roots(root, values)
        else:
            storage, runtime = _tree_roots(root, values, tree_id)
        native_venv = runtime / "venvs/native"
        configured_venv = values.get("GENERATION_NATIVE_VENV") if tree_id is not None else values.get("GENERATION_NATIVE_VENV", str(native_venv))
        if configured_venv is None or Path(configured_venv) != native_venv or not os.access(native_venv / "bin/python", os.X_OK):
            message = f"Native Python 3.12 environment is missing: {native_venv}"
            raise RuntimeError(message)
        return cls(root, storage, runtime, native_venv, MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class SourceAdmission:
    """Exact source revision and fingerprint admitted for one invocation, with its frozen tree when executing."""

    commit: str
    fingerprint: str
    snapshot_path: str | None = None
    snapshot_sha256: str | None = None
    tree_id: str | None = None
    tree_path: Path | None = None
    uncommitted: bool = False  # whether the checkout (read-only) or its snapshot (frozen) holds changes against HEAD

    @property
    def frozen(self) -> bool:
        """Return whether the admission names a frozen source tree."""
        return self.tree_path is not None


@dataclass(frozen=True, slots=True)
class LauncherJob:
    """
    Scheduler identity and CPU resources of a launcher that runs as its own Slurm job.

    ``name`` is the job name and the log file prefix; the job writes
    ``RUNTIME_ROOT/logs/<log_area>/<name>-<job id>.log``. A driver only
    orchestrates, so its CPUs and memory default to a modest allocation.
    """

    name: str
    log_area: str
    partition: str
    wall_time: str
    cpus_per_task: int = 2
    memory: str = "8G"

    def __post_init__(self) -> None:
        """Reject names, partitions, and resources the launcher submission does not admit."""
        if _LAUNCHER_NAME.fullmatch(self.name) is None or self.log_area not in _LAUNCHER_LOG_AREAS:
            message = f"Launcher job name must be a simple name and its log area one of {_LAUNCHER_LOG_AREAS}, got {self.name!r}, {self.log_area!r}."
            raise ValueError(message)
        if self.partition not in _LAUNCHER_PARTITIONS:
            message = f"Launcher partition must be one of {_LAUNCHER_PARTITIONS}, got {self.partition!r}."
            raise ValueError(message)
        if isinstance(self.cpus_per_task, bool) or not isinstance(self.cpus_per_task, int) or self.cpus_per_task < 1:
            message = f"Launcher cpus per task must be an integer >= 1, got {self.cpus_per_task!r}."
            raise ValueError(message)
        if _LAUNCHER_MEMORY.fullmatch(self.memory) is None:
            message = f"Launcher memory must be a positive size such as 8G, got {self.memory!r}."
            raise ValueError(message)
        if _LAUNCHER_WALL_TIME.fullmatch(self.wall_time) is None or not any(character in "123456789" for character in self.wall_time):
            message = f"Launcher time must be a positive [D-]HH:MM:SS limit, got {self.wall_time!r}."
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SubmittedLauncher:
    """The Slurm job of a submitted launcher, the tree it runs from, and its log file."""

    job_id: str
    tree_id: str
    log_path: Path


@dataclass(frozen=True, slots=True)
class _LaunchPlan:
    """The chosen tree and the command and environment that run a launcher from it."""

    tree_id: str
    tree: Path
    command: list[str]
    environment: dict[str, str]


class GenerationCommandError(RuntimeError):
    """A host or scheduled Generation command failed with its original exit code."""

    def __init__(self, operation: str, returncode: int, stderr: str) -> None:
        """Preserve the failed operation, exit code, and diagnostic stderr."""
        self.operation = operation
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"Generation {operation} failed (exit {returncode}): {stderr.strip() or 'see runtime logs'}")


def _run_checked(command: Sequence[str], *, environment: Mapping[str, str] | None = None, cwd: Path | None = None) -> str:
    """Run one small host command while retaining its diagnostic stderr."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, env=environment, cwd=cwd)  # noqa: S603
    except OSError as error:
        message = f"Could not execute {command[0]}: {error}"
        raise RuntimeError(message) from error
    if result.returncode:
        raise GenerationCommandError(command[0], result.returncode, result.stderr)
    return result.stdout


def _physical_directory(value: str | None, *, label: str, variable: str) -> Path:
    """Return one explicit absolute physical directory without relative or symbolic components."""
    if not value:
        message = f"Generation {label} root must be set explicitly through {variable}."
        raise RuntimeError(message)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path != path.resolve(strict=False):
        message = f"Generation {label} root must be one physical absolute path."
        raise ValueError(message)
    if not path.is_dir():
        message = f"Generation {label} root is missing or unsafe: {path}"
        raise RuntimeError(message)
    return path


def _live_roots(root: Path, values: Mapping[str, str]) -> tuple[Path, Path]:
    """Return the storage and runtime roots of a live checkout, which default to its siblings."""
    shared_storage = root.parent / "storage"
    runtime = root.parent / "runtime"
    requested_storage = Path(values.get("STORAGE_ROOT", str(shared_storage)))
    if not requested_storage.is_absolute() or ".." in requested_storage.parts or requested_storage != requested_storage.resolve(strict=False):
        message = "Generation storage root must be one physical absolute path."
        raise ValueError(message)
    if requested_storage != shared_storage and not requested_storage.is_relative_to(shared_storage):
        message = "Generation storage root must be contained in sibling storage."
        raise ValueError(message)
    storage = requested_storage
    if Path(values.get("RUNTIME_ROOT", str(runtime))).resolve(strict=False) != runtime:
        message = "Generation requires the sibling runtime root."
        raise ValueError(message)
    if not storage.is_dir() or any(path.is_symlink() for path in (storage, *storage.parents) if path.is_relative_to(shared_storage)):
        message = f"Shared durable storage is missing or unsafe: {storage}"
        raise RuntimeError(message)
    if not runtime.is_dir() or runtime.is_symlink():
        message = f"Replaceable runtime root is missing or unsafe: {runtime}"
        raise RuntimeError(message)
    return storage, runtime


def _tree_roots(root: Path, values: Mapping[str, str], tree_id: str) -> tuple[Path, Path]:
    """Return the explicit storage and runtime roots of a source tree, which has no sibling defaults."""
    storage = _physical_directory(values.get("STORAGE_ROOT"), label="storage", variable="STORAGE_ROOT")
    runtime = _physical_directory(values.get("RUNTIME_ROOT"), label="runtime", variable="RUNTIME_ROOT")
    if root != runtime / common.paths.SOURCE_TREES_DIRNAME / tree_id or Path(values.get("PROJECT_ROOT", "")) != root:
        message = f"Generation source tree {root} must be RUNTIME_ROOT/{common.paths.SOURCE_TREES_DIRNAME}/{tree_id} and PROJECT_ROOT."
        raise RuntimeError(message)
    if storage == runtime or storage.is_relative_to(runtime) or runtime.is_relative_to(storage):
        message = "Generation storage and runtime roots must be separate directories."
        raise ValueError(message)
    return storage, runtime


def _source_script(layout: HostLayout) -> str:
    """Return the maintained fingerprint tool of the layout's source root."""
    return str(layout.repository / "scripts/source_fingerprint.py")


def _require_login_node_freeze(layout: HostLayout) -> None:
    """Refuse to capture or archive source inside a Slurm allocation, where only a frozen tree may execute."""
    if layout.environment.get("SLURM_JOB_ID"):
        raise RuntimeError(_FREEZE_ON_LOGIN_NODE)


def _manifest_uncommitted(manifest: Mapping[str, object]) -> bool:
    """Return whether a snapshot manifest records tracked changes or untracked files against its commit."""
    return bool(manifest["untracked"]) or manifest["tracked_diff_sha256"] != _EMPTY_DIFF_SHA256


def _admit_live_source(layout: HostLayout, *, freeze: bool) -> SourceAdmission:
    """Fingerprint the live checkout and, for executing commands, capture it as a snapshot and freeze it as a source tree."""
    if freeze:
        _require_login_node_freeze(layout)
    head = source_contract.validate_git_commit(
        _run_checked(("git", "-C", str(layout.repository), "rev-parse", "HEAD"), environment=layout.environment).strip()
    )
    if not freeze:
        status = _run_checked(
            ("git", "--no-optional-locks", "-C", str(layout.repository), "status", "--porcelain=v1", "--untracked-files=all"),
            environment=layout.environment,
        )
        fingerprint = _run_checked(("python3", _source_script(layout), str(layout.repository)), environment=layout.environment).strip()
        if _DIGEST.fullmatch(fingerprint) is None:
            message = "Malformed shared source SHA-256 fingerprint."
            raise ValueError(message)
        return SourceAdmission(head, fingerprint, uncommitted=bool(status))
    for root in (layout.snapshots, layout.trees):
        if root.resolve(strict=False) != root:
            message = f"Generation source snapshot or tree destination contains a symlink: {root}"
            raise ValueError(message)
    receipt = json.loads(
        _run_checked(
            ("python3", _source_script(layout), str(layout.repository), "--materialize", str(layout.snapshots), str(layout.trees)),
            environment=layout.environment,
        )
    )
    snapshot = Path(receipt["snapshot_path"])
    tree = Path(receipt["tree_path"])
    fingerprint = receipt["source_fingerprint_sha256"]
    if (
        receipt["git_commit"] != head
        or _DIGEST.fullmatch(str(fingerprint)) is None
        or tree != layout.trees / receipt["tree_id"]
        or snapshot != layout.snapshots / receipt["tree_id"] / "snapshot.json"
        or snapshot.is_symlink()
        or common.serialization.file_sha256(snapshot) != receipt["snapshot_sha256"]
    ):
        message = "Materialized Generation source tree differs from its admitted worktree."
        raise ValueError(message)
    uncommitted = _manifest_uncommitted(json.loads(snapshot.read_text(encoding="utf-8")))
    return SourceAdmission(head, fingerprint, str(snapshot), str(receipt["snapshot_sha256"]), str(receipt["tree_id"]), tree, uncommitted)


def admit_snapshot_tree(layout: HostLayout, snapshot_path: Path) -> SourceAdmission:
    """
    Freeze the code state of an existing durable snapshot as a source tree.

    A unit bound to that state (for example a campaign started with older code)
    continues there instead of in the live checkout; a pruned tree is restored
    from the snapshot and its base commit. Only a live checkout on the login node can do this.
    """
    if common.paths.source_tree_id(layout.environment) is not None:
        message = "A frozen source tree cannot materialize another tree; start the launcher from the live checkout."
        raise RuntimeError(message)
    _require_login_node_freeze(layout)
    snapshot_path = Path(snapshot_path)
    if snapshot_path.parent.parent != layout.snapshots:
        message = f"Bound source snapshot lies outside the snapshot root {layout.snapshots}: {snapshot_path}"
        raise ValueError(message)
    receipt = json.loads(
        _run_checked(
            ("python3", _source_script(layout), str(layout.repository), "--materialize-snapshot", str(snapshot_path), str(layout.trees)),
            environment=layout.environment,
        )
    )
    tree = Path(receipt["tree_path"])
    if (
        tree != layout.trees / receipt["tree_id"]
        or Path(receipt["snapshot_path"]) != snapshot_path
        or _DIGEST.fullmatch(str(receipt["source_fingerprint_sha256"])) is None
    ):
        message = "Materialized source tree differs from its bound snapshot."
        raise ValueError(message)
    commit = source_contract.validate_git_commit(receipt["git_commit"])
    uncommitted = _manifest_uncommitted(json.loads(snapshot_path.read_text(encoding="utf-8")))
    return SourceAdmission(
        commit, receipt["source_fingerprint_sha256"], str(snapshot_path), str(receipt["snapshot_sha256"]), str(receipt["tree_id"]), tree, uncommitted
    )


def _admit_tree_source(layout: HostLayout) -> SourceAdmission:
    """Verify the executing source tree against its snapshot without Git."""
    tree = layout.repository
    verified = json.loads(
        _run_checked(("python3", _source_script(layout), "--verify-tree", str(tree), str(layout.snapshots)), environment=layout.environment)
    )
    snapshot = layout.snapshots / tree.name / "snapshot.json"
    manifest = json.loads(snapshot.read_text(encoding="utf-8"))
    commit = source_contract.validate_git_commit(manifest["git_commit"])
    if (
        verified["tree_id"] != tree.name
        or verified["git_commit"] != commit
        or manifest["source_fingerprint_sha256"] != verified["source_fingerprint_sha256"]
        or _DIGEST.fullmatch(verified["source_fingerprint_sha256"]) is None
    ):
        message = "Generation source tree disagrees with its snapshot manifest."
        raise ValueError(message)
    return SourceAdmission(
        commit,
        verified["source_fingerprint_sha256"],
        str(snapshot),
        common.serialization.file_sha256(snapshot),
        tree.name,
        tree,
        _manifest_uncommitted(manifest),
    )


def admit_source(layout: HostLayout, *, freeze: bool) -> SourceAdmission:
    """
    Admit the executing source, committed or not.

    In a live checkout an executing command (``freeze``) is captured as a snapshot
    (also when clean, and including uncommitted changes) and materialized as a
    read-only tree; a read-only command is only fingerprinted and leaves no
    snapshot or tree. ``SourceAdmission.uncommitted`` reports whether the checkout
    or snapshot holds changes. A process that already runs from a source tree
    verifies the tree against its snapshot instead and never calls Git.
    """
    if common.paths.source_tree_id(layout.environment) is None:
        return _admit_live_source(layout, freeze=freeze)
    return _admit_tree_source(layout)


def _child_environment(layout: HostLayout, admission: SourceAdmission) -> dict[str, str]:
    """Bind child work to the admitted source tree and its explicit roots, or a read-only command to the live checkout."""
    if admission.tree_path is None or admission.tree_id is None or admission.snapshot_path is None or admission.snapshot_sha256 is None:
        return {
            **layout.environment,
            "GENERATION_GIT_COMMIT": admission.commit,
            "GENERATION_SOURCE_SHA256": admission.fingerprint,
            "GENERATION_NATIVE_VENV": str(layout.native_venv),
            "STORAGE_ROOT": str(layout.storage),
        }
    return {
        **layout.environment,
        # The tree's directories are read-only, so no child may try to write bytecode into it.
        "PYTHONDONTWRITEBYTECODE": "1",
        "GENERATION_GIT_COMMIT": admission.commit,
        "GENERATION_SOURCE_SHA256": admission.fingerprint,
        "GENERATION_SOURCE_SNAPSHOT": admission.snapshot_path,
        "GENERATION_SOURCE_SNAPSHOT_SHA256": admission.snapshot_sha256,
        "GENERATION_NATIVE_VENV": str(layout.native_venv),
        "PROJECT_ROOT": str(admission.tree_path),
        "SOURCE_TREE_ID": admission.tree_id,
        "STORAGE_ROOT": str(layout.storage),
        "RUNTIME_ROOT": str(layout.runtime),
    }


def lease_submitted_job(
    trees_root: Path, source_script: Path | str, tree_id: str, job_id: str, *, environment: Mapping[str, str] | None = None
) -> bool:
    """
    Lease a source tree for a job that was already submitted, so tree pruning keeps it while the job is queued or running.

    A lease failure never loses the job: it prints one warning and returns False, because a tree
    younger than 24 hours or with a newer lease stays protected by its recency
    and the caller still records the job ID. The caller owns what an unleased job means.

    Returns
    -------
    bool
        Whether the lease was recorded.

    """
    try:
        _run_checked(("python3", str(source_script), "--lease", str(trees_root), tree_id, job_id), environment=environment)
    except RuntimeError as error:
        reason = str(error).strip().splitlines()[-1] if str(error).strip() else "unknown error"
        print(
            f"Warning: could not lease source tree {tree_id} for job {job_id} ({reason}); the tree stays protected by its 24-hour recency.",
            file=sys.stderr,
        )
        return False
    return True


def _launch_plan(layout: HostLayout, admission: SourceAdmission, *, module: str, arguments: Sequence[str], explicit_transition: bool) -> _LaunchPlan:
    """
    Prune unused trees and plan the command and environment that run a launcher from its chosen tree.

    Absolute paths inside the live checkout are mapped to the tree. The
    environment names the tree and its explicit roots; ``explicit_transition``
    marks a deliberate continuation of a bound unit with the current code, which
    the campaign records as a new execution epoch.

    Raises
    ------
    RuntimeError
        If the process already runs from a tree or the admission is not frozen.

    """
    if common.paths.source_tree_id(layout.environment) is not None:
        message = "The launcher already runs from a frozen source tree."
        raise RuntimeError(message)
    if admission.tree_path is None or admission.tree_id is None:
        message = "Only a frozen admission names a source tree to execute from."
        raise RuntimeError(message)
    _run_checked(("python3", _source_script(layout), "--prune", str(layout.trees)), environment=layout.environment)
    mapped = []
    for argument in arguments:
        candidate = Path(argument)
        if candidate.is_absolute() and candidate.is_relative_to(layout.repository):
            mapped.append(str(admission.tree_path / candidate.relative_to(layout.repository)))
        else:
            mapped.append(argument)
    environment = _child_environment(layout, admission)
    # The job log is read while the driver runs, so its output must not wait in a block buffer.
    environment["PYTHONUNBUFFERED"] = "1"
    if explicit_transition:
        environment[source_contract.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE] = source_contract.EXPLICIT_SOURCE_TRANSITION
    else:
        environment.pop(source_contract.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE, None)
    return _LaunchPlan(admission.tree_id, admission.tree_path, [str(layout.python), "-m", module, *mapped], environment)


def submit_source_tree(
    layout: HostLayout,
    admission: SourceAdmission,
    *,
    module: str,
    arguments: Sequence[str],
    explicit_transition: bool,
    job: LauncherJob,
) -> SubmittedLauncher:
    """
    Submit one launcher command to run from its chosen frozen source tree as its own Slurm job.

    The job runs ``python -m module arguments`` from the tree with the tree
    environment, live checkout paths mapped into the tree, writes
    ``RUNTIME_ROOT/logs/<log_area>/<name>-<job id>.log``, and holds a lease on the
    tree under the returned job ID, so pruning keeps the tree while the job is
    queued or running. A failing lease only warns; the job ID is always returned.

    Raises
    ------
    RuntimeError
        If the process already runs from a tree, the admission is not frozen,
        the log directory is unsafe, or Slurm answers without a job ID.

    """
    plan = _launch_plan(layout, admission, module=module, arguments=arguments, explicit_transition=explicit_transition)
    logs = layout.runtime / "logs" / job.log_area
    logs.mkdir(parents=True, exist_ok=True)
    if logs.resolve(strict=True) != logs:
        message = f"Launcher log directory is unsafe: {logs}"
        raise RuntimeError(message)
    output = _run_checked(
        (
            "sbatch",
            "--parsable",
            f"--job-name={job.name}",
            f"--partition={job.partition}",
            f"--cpus-per-task={job.cpus_per_task}",
            f"--mem={job.memory}",
            f"--time={job.wall_time}",
            f"--output={launcher_log_template(layout.runtime, job.log_area, job.name)}",
            # A requeue must never truncate a log that is hard-linked into its campaign.
            "--open-mode=append",
            f"--chdir={plan.tree}",
            "--export=ALL",
            f"--wrap={shlex.join(plan.command)}",
        ),
        environment=plan.environment,
    )
    job_id = output.strip().split(";", maxsplit=1)[0]
    if re.fullmatch(r"[0-9]+", job_id) is None:
        message = f"Slurm returned an invalid parsable job identifier: {output!r}."
        raise RuntimeError(message)
    lease_submitted_job(layout.trees, _source_script(layout), plan.tree_id, job_id, environment=layout.environment)
    return SubmittedLauncher(job_id, plan.tree_id, Path(str(launcher_log_template(layout.runtime, job.log_area, job.name)).replace("%j", job_id)))


def launcher_log_template(runtime: Path, log_area: str, name: str) -> Path:
    """Return the sbatch ``--output`` template of a launcher job: ``RUNTIME_ROOT/logs/<log_area>/<name>-%j.log``."""
    return runtime / "logs" / log_area / f"{name}-%j.log"


def job_log_files(templates: Iterable[Path | str]) -> tuple[Path, ...]:
    """Return the existing regular log files of this Slurm job from the sbatch ``--output``/``--error`` templates that submitted it.

    Each template is expanded with ``SLURM_JOB_ID`` for ``%j``; outside a Slurm job nothing is returned.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[0-9]+", job_id) is None:
        return ()
    files: list[Path] = []
    for template in templates:
        path = Path(str(template).replace("%j", job_id))
        if path not in files and path.is_file() and not path.is_symlink():
            files.append(path)
    return tuple(files)


def current_job_log_files() -> tuple[Path, ...]:
    """Return this launcher job's own log from the launcher ``--output`` template, its Slurm job name, and its job ID."""
    name = os.environ.get("SLURM_JOB_NAME", "")
    if _LAUNCHER_NAME.fullmatch(name) is None:
        return ()
    runtime = common.paths.get_runtime_root()
    return job_log_files(launcher_log_template(runtime, area, name) for area in _LAUNCHER_LOG_AREAS)


def link_job_logs(owner_directory: Path, files: Sequence[Path]) -> None:
    """
    Hard-link job log files into ``<owner>/_record/logs/`` under their own names.

    An ``.err`` file that is empty when it is linked tells nothing and is not linked. The owner, a campaign folder,
    must already exist and is never created here. A link to the same file is kept, a different existing file is
    left alone with a warning, a new link whose inode differs from the file checked
    before linking is removed again, and any failure is one warning on stderr, never an error.
    """
    if not files:
        return
    logs = common.paths.resolve_generation_campaign_logs_directory(owner_directory)
    if owner_directory.is_symlink() or logs.parent.is_symlink() or logs.is_symlink():
        print(f"Warning: could not link job logs into {logs}: the owner, its _record/ or logs/ is a symbolic link", file=sys.stderr, flush=True)
        return
    try:
        logs.mkdir(exist_ok=True)
        for source in files:
            target = logs / source.name
            checked = source.lstat()
            if source.suffix == ".err" and checked.st_size == 0:
                continue
            try:
                target.hardlink_to(source)
            except FileExistsError:
                if not target.samefile(source):
                    print(f"Warning: {target} is a different file; {source} was not linked.", file=sys.stderr, flush=True)
                continue
            linked = target.lstat()
            if (linked.st_dev, linked.st_ino) != (checked.st_dev, checked.st_ino):
                target.unlink()
                print(f"Warning: {source} changed while it was linked; the link was removed.", file=sys.stderr, flush=True)
    except OSError as error:
        print(f"Warning: could not link job logs into {logs}: {error}", file=sys.stderr, flush=True)


def _prepare_logs(layout: HostLayout, *, probe_writable: bool = False) -> None:
    """Admit the canonical runtime log directory for scheduled work."""
    layout.logs.mkdir(parents=True, exist_ok=True)
    if layout.logs.resolve(strict=True) != layout.logs:
        message = f"Generation runtime log directory is unsafe: {layout.logs}"
        raise RuntimeError(message)
    if probe_writable:
        descriptor, probe = tempfile.mkstemp(prefix=".preflight.", dir=layout.logs)
        os.close(descriptor)
        Path(probe).unlink()


def _json_result_summary(operation: str, stdout: bytes) -> str | None:
    """Return the one-line log summary of a stdout that is a single JSON object, or None for any other output."""
    try:
        value = json.loads(stdout)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    state = value.get("state")
    suffix = f" state={state}" if isinstance(state, str) else ""
    return f"{operation}: JSON result returned to the driver ({len(stdout)} bytes, sha256 {hashlib.sha256(stdout).hexdigest()[:12]}){suffix}\n"


def _run_scheduled(
    command: Sequence[str],
    working_directory: Path,
    environment: Mapping[str, str],
    output_path: str,
    operation: str,
) -> tuple[int, str, str]:
    """
    Stream worker stdout into its runtime log and report long silent work.

    A successful run whose stdout is one JSON object prints only a summary line to the driver's own log and leaves no
    runtime log file, because the driver receives the object itself; any other output, and every failed run, keeps its
    raw stdout in the runtime log file.
    """
    interval = int(environment.get("GENERATION_CONSOLE_HEARTBEAT_SECONDS", "120"))
    if interval < 1:
        message = "Generation console heartbeat interval must be positive."
        raise ValueError(message)
    started = time.monotonic()
    last_heartbeat = started
    stdout = bytearray()
    stderr = bytearray()
    try:
        with (
            subprocess.Popen(  # noqa: S603
                command,
                cwd=working_directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as process,
            Path(output_path).open("wb") as log,
            selectors.DefaultSelector() as selector,
        ):
            if process.stdout is None or process.stderr is None:
                message = f"Could not capture Generation {operation} output."
                raise RuntimeError(message)
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                for key, _ in selector.select(timeout=interval):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == "stdout":
                        stdout.extend(chunk)
                        log.write(chunk)
                        log.flush()
                    else:
                        stderr.extend(chunk)
                now = time.monotonic()
                if now - last_heartbeat >= interval:
                    elapsed = int(now - started)
                    print(f"Generation {operation}: active for {elapsed}s; log={output_path}", file=sys.stderr, flush=True)
                    last_heartbeat = now
            status = process.wait()
            summary = _json_result_summary(operation, bytes(stdout)) if status == 0 else None
            if summary is not None:
                print(summary, end="", file=sys.stderr, flush=True)  # stdout belongs to the step's result; the driver's log takes both streams
                Path(output_path).unlink(missing_ok=True)
            return status, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    except OSError as error:
        message = f"Could not execute Generation {operation}: {error}"
        raise RuntimeError(message) from error


@dataclass(slots=True)
class GenerationHost:
    """
    Execute Generation commands bound to one admitted source tree.

    Parameters
    ----------
    layout : HostLayout
        Validated repository, storage, runtime, and inherited environment.
    admission : SourceAdmission
        Exact source commit, fingerprint, snapshot, and tree required by every worker.

    """

    layout: HostLayout
    admission: SourceAdmission
    # Helper logs written before the campaign is known, and the campaign their links go to once it is.
    _helper_logs: list[Path] = field(default_factory=list, init=False, repr=False)
    _log_campaign: Path | None = field(default=None, init=False, repr=False)

    def attach_logs(self, campaign_directory: Path) -> None:
        """Link this job's own logs and every helper log into the existing campaign; later helper logs follow as they finish."""
        self._log_campaign = campaign_directory
        link_job_logs(campaign_directory, (*current_job_log_files(), *self._helper_logs))
        self._helper_logs.clear()

    @property
    def repository(self) -> Path:
        """Return the admitted repository root, the live checkout or the executing tree."""
        return self.layout.repository

    @property
    def execution_root(self) -> Path:
        """Return the frozen source tree commands execute from, or the live checkout of a read-only admission."""
        return self.admission.tree_path if self.admission.tree_path is not None else self.layout.repository

    @property
    def node_launcher(self) -> Path:
        """Return the node-side launcher inside the admitted tree."""
        return self.execution_root / "scripts/generation_node.sh"

    @property
    def storage(self) -> Path:
        """Return the admitted durable storage root."""
        return self.layout.storage

    @property
    def commit(self) -> str:
        """Return the admitted source commit."""
        return self.admission.commit

    def choose_tree(self, *, bound_snapshot: Path | None, continue_with_current_code: bool) -> tuple[SourceAdmission, bool]:
        """
        Choose the frozen source tree a launcher executes in.

        ``bound_snapshot`` names the code state an existing unit executes in; the
        launcher continues there, restoring a pruned tree, unless
        ``continue_with_current_code`` selects this admission's current tree. The
        returned flag marks that explicit source transition.
        """
        admission = self.admission
        explicit = False
        if bound_snapshot is not None:
            if continue_with_current_code:
                explicit = admission.tree_id != bound_snapshot.parent.name
            elif admission.tree_id != bound_snapshot.parent.name:
                admission = admit_snapshot_tree(self.layout, bound_snapshot)
        return admission, explicit

    def submit_tree(
        self, admission: SourceAdmission, *, module: str, arguments: Sequence[str], explicit_transition: bool, job: LauncherJob
    ) -> SubmittedLauncher:
        """Submit a launcher to run from its chosen frozen source tree as its own Slurm job."""
        return submit_source_tree(self.layout, admission, module=module, arguments=arguments, explicit_transition=explicit_transition, job=job)

    def config_file(self, value: Path | str) -> tuple[Path, str]:
        """Return a regular repository file and its safe repository-relative path."""
        layout = self.layout
        raw = str(value)
        if any(character in raw for character in "\n\r\t"):
            message = "Generation config contains a control character."
            raise ValueError(message)
        if not raw.startswith("/") and any(component in ("", ".", "..") for component in raw.split("/")):
            message = "Generation config has an unsafe repository-relative path."
            raise ValueError(message)
        path = Path(os.path.normpath(raw)) if raw.startswith("/") else Path(raw)
        if path.is_absolute():
            try:
                relative = path.relative_to(layout.repository)
            except ValueError as error:
                message = "Generation config must remain inside the repository."
                raise ValueError(message) from error
        else:
            relative = path
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            message = "Generation config has an unsafe repository-relative path."
            raise ValueError(message)
        candidate = self.execution_root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            message = f"Generation config does not exist in the admitted source tree: {candidate}"
            raise FileNotFoundError(message) from error
        if resolved != candidate or not resolved.is_file() or resolved.is_symlink():
            message = f"Generation config must be a regular file without symbolic links: {candidate}"
            raise ValueError(message)
        return resolved, relative.as_posix()

    def call(
        self,
        operation: str,
        *arguments: str,
        scheduled: bool | None = None,
        partition: str = "standard",
        wall_time: str = "02:00:00",
    ) -> str:
        """Invoke one existing Generation CLI command on the host or through Slurm."""
        layout, admission = self.layout, self.admission
        environment = _child_environment(layout, admission)
        if scheduled:
            if not admission.frozen:
                message = "Scheduled Generation work runs only from a frozen source tree; this command was admitted read-only."
                raise RuntimeError(message)
            if partition not in ("standard", "long"):
                message = "Native Generation Python work requires the standard or long CPU partition."
                raise ValueError(message)
            worker = self.node_launcher
            if not worker.is_file() or worker.is_symlink() or not os.access(worker, os.X_OK):
                message = f"Native Generation Python worker is missing or unsafe: {worker}"
                raise RuntimeError(message)

            worker_command = (str(worker), str(self.execution_root), "cli", operation, *arguments)
            on_allocated_node = (
                re.fullmatch(r"[0-9]+", environment.get("SLURM_JOB_ID", "")) is not None
                and re.fullmatch(r"[1-9][0-9]*", environment.get("SLURM_CPUS_PER_TASK", "")) is not None
                and environment.get("SLURMD_NODENAME", "").split(".", 1)[0] == socket.gethostname().split(".", 1)[0]
            )
            _prepare_logs(layout)
            descriptor, output_path = tempfile.mkstemp(prefix=f"python-{operation}.", suffix=".out", dir=layout.logs)
            os.close(descriptor)
            if on_allocated_node:
                # Keep the node launcher's admission, environment and output guards.
                command = worker_command
                error_path = None
            else:
                error_path = str(Path(output_path).with_suffix(".err"))
                command = (
                    "srun",
                    f"--partition={partition}",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=4",
                    "--mem=16G",
                    f"--time={wall_time}",
                    f"--job-name=generation-python-{operation}",
                    f"--chdir={self.execution_root}",
                    "--export=ALL",
                    f"--error={error_path}",
                    *worker_command,
                )
        else:
            output_path = None
            error_path = None
            command = (str(layout.python), "-m", _CLI_MODULE, operation, *arguments)
        if output_path is not None:
            status, stdout, stderr = _run_scheduled(command, self.execution_root, environment, output_path, operation)
        else:
            try:
                result = subprocess.run(command, cwd=self.execution_root, env=environment, capture_output=True, text=True, check=False)  # noqa: S603
            except OSError as error:
                message = f"Could not execute Generation {operation}: {error}"
                raise RuntimeError(message) from error
            status, stdout, stderr = result.returncode, result.stdout, result.stderr
        if status and error_path is not None and Path(error_path).exists():
            stderr += "\n" + "\n".join(Path(error_path).read_text(encoding="utf-8").splitlines()[-80:])
        if error_path is not None and Path(error_path).is_file() and Path(error_path).stat().st_size == 0:
            Path(error_path).unlink()  # srun creates it for every step; an empty one tells nothing
        if output_path is not None:
            produced = [Path(path) for path in (output_path, error_path) if path is not None and Path(path).is_file()]
            if self._log_campaign is None:
                self._helper_logs.extend(produced)
            else:
                link_job_logs(self._log_campaign, produced)
        if status:
            raise GenerationCommandError(operation, status, stderr)
        return stdout

    def comsol_version(self, module: str, executable: str) -> str:
        """Query validated COMSOL version evidence in a login shell with module support."""
        layout = self.layout
        script = 'module load "$1" && source "$2" && generation_comsol_version "$3"'
        helper = str(self.execution_root / "scripts/generation_prerequisites.sh")
        output = _run_checked(("bash", "-lc", script, "generation-comsol-version", module, helper, executable), environment=layout.environment)
        return output.strip()

    def verify_execution(self) -> None:
        """Check the node launcher and native imports before reporting preflight success."""
        layout = self.layout
        worker = self.node_launcher
        if not worker.is_file() or worker.is_symlink() or not os.access(worker, os.X_OK):
            message = f"Native Generation Slurm worker is missing or unsafe: {worker}"
            raise RuntimeError(message)
        _prepare_logs(layout, probe_writable=True)
        _run_checked(
            (str(layout.python), "-c", "import h5py, numpy, scipy, yaml, torch; import src.generation.cli.cli_generation"),
            environment=_child_environment(layout, self.admission),
            cwd=self.execution_root,
        )
