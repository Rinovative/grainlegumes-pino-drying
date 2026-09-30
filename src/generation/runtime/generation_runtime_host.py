"""
generation_runtime_host.py

Own admitted host processes and scheduled commands for Generation.

Responsibilities:
  - Admit the shared source and repository-owned configuration files
  - Resolve the canonical sibling storage, runtime, and native Python paths
  - Invoke Generation service commands and submit disposable Slurm smoke work
  - Manage tmux processes using background-service session evidence

Design principles:
  - Keep source identity checks ahead of scheduled worker execution
  - Pass scheduler arguments as vectors without selecting physical hardware
  - Preserve worker-owned validation, simulation, and publication semantics

This module does NOT:
  - Interpret scientific configurations or persisted lifecycle state
  - Run COMSOL cases directly on the login node
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import selectors
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from src.generation.contracts import generation_contracts_source as source_contract

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_DIGEST = re.compile(r"[0-9a-f]{64}")
_SMOKE_JOB_ID = re.compile(r"[0-9]+(?:;[A-Za-z0-9._-]+)?")
_CLI_MODULE = "src.generation.cli.cli_generation"


@dataclass(frozen=True, slots=True)
class HostLayout:
    """Canonical paths and inherited environment for one shared checkout."""

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
    def node_launcher(self) -> Path:
        """Return the shared Slurm node-side launcher."""
        return self.repository / "scripts/generation_node.sh"

    @property
    def logs(self) -> Path:
        """Return the replaceable Generation log directory."""
        return self.runtime / "logs/generation"

    @classmethod
    def resolve(cls, repository: Path | str, *, environment: Mapping[str, str] | None = None) -> HostLayout:
        """
        Admit a physical repository and its canonical sibling directories.

        Parameters
        ----------
        repository : Path | str
            Explicit shared checkout, including a test-owned checkout when testing.
        environment : Mapping[str, str] | None, optional
            Host environment used for path overrides and child processes.

        Returns
        -------
        HostLayout
            Validated host paths and a snapshot of the inherited environment.

        """
        values = dict(os.environ if environment is None else environment)
        candidate = Path(repository).absolute()
        if candidate.is_symlink() or not (candidate / ".git").is_dir():
            message = f"Generation requires the current shared Git repository: {candidate}"
            raise RuntimeError(message)
        root = candidate.resolve(strict=True)
        if root != candidate:
            message = f"Generation repository must be a physical path: {candidate}"
            raise RuntimeError(message)
        storage = root.parent / "storage"
        runtime = root.parent / "runtime"
        if Path(values.get("STORAGE_ROOT", str(storage))).resolve(strict=False) != storage:
            message = "Generation requires the sibling storage root."
            raise ValueError(message)
        if Path(values.get("RUNTIME_ROOT", str(runtime))).resolve(strict=False) != runtime:
            message = "Generation requires the sibling runtime root."
            raise ValueError(message)
        if not storage.is_dir() or storage.is_symlink():
            message = f"Shared durable storage is missing or unsafe: {storage}"
            raise RuntimeError(message)
        if not runtime.is_dir() or runtime.is_symlink():
            message = f"Replaceable runtime root is missing or unsafe: {runtime}"
            raise RuntimeError(message)
        native_venv = runtime / "venvs/native"
        if Path(values.get("GENERATION_NATIVE_VENV", str(native_venv))) != native_venv or not os.access(native_venv / "bin/python", os.X_OK):
            message = f"Native Python 3.12 environment is missing: {native_venv}"
            raise RuntimeError(message)
        return cls(root, storage, runtime, native_venv, MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class SourceAdmission:
    """Exact source revision and worktree fingerprint admitted for one invocation."""

    commit: str
    fingerprint: str


class GenerationCommandError(RuntimeError):
    """A host or scheduled Generation command failed with its original exit code."""

    def __init__(self, operation: str, returncode: int, stderr: str) -> None:
        """Preserve the failed operation, exit code, and diagnostic stderr."""
        self.operation = operation
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"Generation {operation} failed (exit {returncode}): {stderr.strip() or 'see runtime logs'}")


def _run_checked(command: Sequence[str], *, environment: Mapping[str, str] | None = None) -> str:
    """Run one small host command while retaining its diagnostic stderr."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)  # noqa: S603
    except OSError as error:
        message = f"Could not execute {command[0]}: {error}"
        raise RuntimeError(message) from error
    if result.returncode:
        raise GenerationCommandError(command[0], result.returncode, result.stderr)
    return result.stdout


def admit_source(layout: HostLayout, requested_commit: str | None, *, require_clean: bool) -> SourceAdmission:
    """Admit the current HEAD and exact fingerprint without modifying the source."""
    head = source_contract.validate_git_commit(
        _run_checked(("git", "-C", str(layout.repository), "rev-parse", "HEAD"), environment=layout.environment).strip()
    )
    if requested_commit is not None and source_contract.validate_git_commit(requested_commit) != head:
        message = "Generation runs only the current shared repository HEAD; requested commit differs."
        raise ValueError(message)
    fingerprint = _run_checked(
        ("python3", str(layout.repository / "scripts/source_fingerprint.py"), str(layout.repository)),
        environment=layout.environment,
    ).strip()
    if _DIGEST.fullmatch(fingerprint) is None:
        message = "Malformed shared source SHA-256 fingerprint."
        raise ValueError(message)
    status = _run_checked(
        ("git", "--no-optional-locks", "-C", str(layout.repository), "status", "--porcelain=v1", "--untracked-files=all"),
        environment=layout.environment,
    )
    if require_clean and status:
        message = "Generation publication requires a clean committed shared source; current worktree has changes."
        raise ValueError(message)
    return SourceAdmission(head, fingerprint)


def _child_environment(layout: HostLayout, admission: SourceAdmission) -> dict[str, str]:
    """Bind all child work to the admitted source and sibling runtime paths."""
    return {
        **layout.environment,
        "GENERATION_GIT_COMMIT": admission.commit,
        "GENERATION_SOURCE_SHA256": admission.fingerprint,
        "GENERATION_NATIVE_VENV": str(layout.native_venv),
        "STORAGE_ROOT": str(layout.storage),
    }


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


def _run_scheduled(
    command: Sequence[str],
    layout: HostLayout,
    environment: Mapping[str, str],
    output_path: str,
    operation: str,
) -> tuple[int, str, str]:
    """Stream Slurm stdout into its runtime log and report long silent work."""
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
                cwd=layout.repository,
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
            return process.wait(), stdout.decode(errors="replace"), stderr.decode(errors="replace")
    except OSError as error:
        message = f"Could not execute Generation {operation}: {error}"
        raise RuntimeError(message) from error


@dataclass(slots=True)
class GenerationHost:
    """
    Execute Generation commands bound to one admitted shared checkout.

    Parameters
    ----------
    layout : HostLayout
        Validated repository, storage, runtime, and inherited environment.
    admission : SourceAdmission
        Exact source commit and fingerprint required by every worker.

    """

    layout: HostLayout
    admission: SourceAdmission

    @property
    def repository(self) -> Path:
        """Return the admitted repository root."""
        return self.layout.repository

    @property
    def storage(self) -> Path:
        """Return the admitted durable storage root."""
        return self.layout.storage

    @property
    def commit(self) -> str:
        """Return the admitted source commit."""
        return self.admission.commit

    def verify_source(self) -> None:
        """Require the admitted clean source again before paired publication."""
        current = admit_source(self.layout, self.admission.commit, require_clean=True)
        if current.fingerprint != self.admission.fingerprint:
            message = "Generation source fingerprint changed before paired publication."
            raise ValueError(message)

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
        candidate = layout.repository / relative
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            message = f"Generation config does not exist in the current shared repository: {candidate}"
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
        benchmark_preflight: bool = False,
        wall_time: str = "02:00:00",
    ) -> str:
        """Invoke one existing Generation CLI command on the host or through Slurm."""
        layout, admission = self.layout, self.admission
        environment = _child_environment(layout, admission)
        if scheduled:
            if partition not in ("standard", "long"):
                message = "Native Generation Python work requires the standard or long CPU partition."
                raise ValueError(message)
            worker = layout.node_launcher
            if not worker.is_file() or worker.is_symlink() or not os.access(worker, os.X_OK):
                message = f"Native Generation Python worker is missing or unsafe: {worker}"
                raise RuntimeError(message)
            _prepare_logs(layout)
            descriptor, output_path = tempfile.mkstemp(prefix=f"python-{operation}.", suffix=".out", dir=layout.logs)
            os.close(descriptor)
            error_path = str(Path(output_path).with_suffix(".err"))
            mode = "benchmark-preflight" if benchmark_preflight else "cli"
            command = (
                "srun",
                f"--partition={partition}",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=4",
                "--mem=16G",
                f"--time={wall_time}",
                f"--job-name=generation-python-{operation}",
                f"--chdir={layout.repository}",
                "--export=ALL",
                f"--error={error_path}",
                str(worker),
                str(layout.repository),
                mode,
                operation,
                *arguments,
            )
        else:
            output_path = None
            error_path = None
            command = (str(layout.python), "-m", _CLI_MODULE, operation, *arguments)
        if output_path is not None:
            status, stdout, stderr = _run_scheduled(command, layout, environment, output_path, operation)
        else:
            try:
                result = subprocess.run(command, cwd=layout.repository, env=environment, capture_output=True, text=True, check=False)  # noqa: S603
            except OSError as error:
                message = f"Could not execute Generation {operation}: {error}"
                raise RuntimeError(message) from error
            status, stdout, stderr = result.returncode, result.stdout, result.stderr
        if status and error_path is not None and Path(error_path).exists():
            stderr += "\n" + "\n".join(Path(error_path).read_text(encoding="utf-8").splitlines()[-80:])
        if status:
            raise GenerationCommandError(operation, status, stderr)
        return stdout

    def comsol_version(self, module: str, executable: str) -> str:
        """Query validated COMSOL version evidence in a login shell with module support."""
        layout = self.layout
        script = 'module load "$1" && source "$2" && generation_comsol_version "$3"'
        output = _run_checked(
            ("bash", "-lc", script, "generation-comsol-version", module, str(layout.repository / "scripts/generation_prerequisites.sh"), executable),
            environment=layout.environment,
        )
        return output.strip()

    def verify_execution(self) -> None:
        """Check the node launcher and native imports before reporting preflight success."""
        layout = self.layout
        worker = layout.node_launcher
        if not worker.is_file() or worker.is_symlink() or not os.access(worker, os.X_OK):
            message = f"Native Generation Slurm worker is missing or unsafe: {worker}"
            raise RuntimeError(message)
        _prepare_logs(layout, probe_writable=True)
        _run_checked(
            (str(layout.python), "-c", "import h5py, numpy, scipy, yaml, torch; import src.generation.cli.cli_generation"),
            environment=layout.environment,
        )

    def submit_smoke(self, *, partition: str = "gpu") -> str:
        """Submit the disposable, CPU-only native COMSOL smoke job and return its ID."""
        layout, admission = self.layout, self.admission
        if partition not in ("gpu", "standard"):
            message = "Smoke partition must be gpu or standard."
            raise ValueError(message)
        worker = layout.node_launcher
        if not worker.is_file() or worker.is_symlink() or not os.access(worker, os.X_OK):
            message = f"Native Generation smoke worker is missing or unsafe: {worker}"
            raise RuntimeError(message)
        _prepare_logs(layout)
        command = (
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=4G",
            "--time=00:10:00",
            f"--partition={partition}",
            f"--chdir={layout.repository}",
            "--job-name=generation-native-smoke",
            f"--output={layout.logs}/slurm-%j.out",
            f"--error={layout.logs}/slurm-%j.err",
            "--export=ALL",
            str(worker),
            str(layout.repository),
            "smoke",
        )
        job_id = _run_checked(command, environment=_child_environment(layout, admission)).strip()
        if _SMOKE_JOB_ID.fullmatch(job_id) is None:
            message = f"Native Generation smoke returned an invalid Slurm job ID: {job_id}"
            raise RuntimeError(message)
        return job_id

    def _background_arguments(self) -> list[str]:
        """Bind session inspection to the current tmux process inventory."""
        executable = shutil.which("tmux")
        if executable is None:
            return []
        result = subprocess.run((executable, "list-sessions", "-F", "#S"), capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode:
            return []
        return [item for session in result.stdout.splitlines() if session for item in ("--active-tmux-session", session)]

    def inspect_background(self, session_id: str) -> dict[str, Any]:
        """Inspect durable background evidence against current tmux ownership."""
        return json.loads(self.call("inspect-background-session", session_id, "--storage-root", str(self.storage), *self._background_arguments()))

    def list_background(self) -> list[dict[str, Any]]:
        """List durable background sessions with current process ownership."""
        return json.loads(self.call("list-background-sessions", "--storage-root", str(self.storage), *self._background_arguments()))["sessions"]

    def launch_background(self, arguments: Sequence[str]) -> dict[str, Any]:
        """
        Start or reuse one durable background controller through tmux.

        Parameters
        ----------
        arguments : Sequence[str]
            Public command arguments containing exactly one background flag.

        Returns
        -------
        dict[str, Any]
            Service-owned session metadata with launch status and exit code.

        """
        if os.environ.get("GENERATION_WORKFLOW_BACKGROUND_CHILD") == "1":
            message = "A background workflow child cannot create another session"
            raise ValueError(message)
        executable = shutil.which("tmux")
        if executable is None:
            message = "tmux is required for background Generation"
            raise ValueError(message)
        child_arguments = [item for item in arguments if item != "--background"]
        if sum(item == "--background" for item in arguments) != 1:
            message = "Specify --background exactly once"
            raise ValueError(message)
        if "--git-commit" not in child_arguments:
            child_arguments.extend(("--git-commit", self.commit))
        host_paths = json.dumps(
            {
                "stable_script": str(self.repository / "scripts/generation"),
                "python_executable": str(self.layout.python),
                "storage_root": str(self.storage),
                "host": socket.getfqdn(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        session = json.loads(
            self.call(
                "create-background-session",
                "--source-commit",
                self.commit,
                "--storage-root",
                str(self.storage),
                "--host-paths-json",
                host_paths,
                *self._background_arguments(),
                "--",
                *child_arguments,
            )
        )
        session_id = session["workflow_session_id"]
        tmux_name = session["tmux_session_name"]
        if session["status"] == "reused":
            return {**session, "exit_code": 3}
        command = Path(session["command_path"])
        if session["status"] != "created" or not command.is_file() or not os.access(command, os.X_OK):
            message = "Created background command is missing or unsafe"
            raise RuntimeError(message)
        launched = subprocess.run(  # noqa: S603
            (executable, "new-session", "-d", "-s", tmux_name, shlex.quote(str(command))), capture_output=True, text=True, check=False
        )
        if launched.returncode:
            with contextlib.suppress(GenerationCommandError):
                self.call("complete-background-session", session_id, "--exit-code", "1", "--storage-root", str(self.storage))
            message = f"tmux could not start the background workflow: {launched.stderr}"
            raise RuntimeError(message)
        active = subprocess.run((executable, "has-session", "-t", f"={tmux_name}"), capture_output=True, text=True, check=False)  # noqa: S603
        if active.returncode:
            state = self.inspect_background(session_id)
            if state["workflow_state"] in {"completed", "failed"}:
                return {**session, "status": state["workflow_state"], "exit_code": int(state["exit_code"])}
            message = "Background session disappeared without a terminal result"
            raise RuntimeError(message)
        return {**session, "status": "started", "exit_code": 0}
