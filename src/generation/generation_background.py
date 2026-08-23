"""
generation_background.py

Persist and inspect host-owned tmux workflow-session evidence.
Responsibilities:
  - Create owner-only session metadata and a safely quoted durable command
  - Prevent duplicate active controllers for the same semantic command digest
  - Atomically record the child workflow exit result and discover run identities
Design principles:
  - tmux remains the process/session owner; persisted files remain inspectable later
  - Session identity is operational evidence and never enters scientific identity
  - Exact argv and source commit are durable, while secrets are rejected
This module does NOT:
  - Start, attach, detach, kill, or query tmux
  - Execute campaign, benchmark, collection, or cleanup workflows
  - Claim that a tmux session survives a host reboot
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.runtime import generation_runtime_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKGROUND_SCHEMA_VERSION: Final = 1
_MAX_PROCESS_EXIT_CODE: Final = 255
_SESSION_SCHEMA_KIND: Final = "generation_workflow_session"
_RESULT_SCHEMA_KIND: Final = "generation_workflow_session_result"
_SESSION_ID_PATTERN: Final = re.compile(r"gw-[0-9]{8}T[0-9]{6}Z-[a-z0-9-]+-[0-9a-f]{8}(?:-[0-9]{2,})?")
_SAFE_SUBCOMMAND_PATTERN: Final = re.compile(r"[a-z][a-z0-9-]*")
_RUN_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9._-]+__[0-9a-f]{16}")
_BENCHMARK_RUN_ID_PATTERN: Final = re.compile(r"core_scaling_transient__[0-9a-f]{16}")
_SUPPORTED_SUBCOMMANDS: Final = frozenset({"run", "timing-probe"})
_SESSION_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "workflow_session_id",
        "tmux_session_name",
        "host",
        "user",
        "source_commit",
        "subcommand",
        "argv",
        "command_digest",
        "storage_root",
        "started_at",
        "state",
        "log_path",
        "command_path",
    }
)
_RESULT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "workflow_session_id",
        "exit_code",
        "terminal_state",
        "ended_at",
        "campaign_run_ids",
        "benchmark_run_ids",
        "final_stage",
    }
)


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _sessions_root(
    storage_root: Path | str,
    *,
    create: bool,
) -> Path:
    """Return the owner-only host workflow-session root without implicit writes."""
    storage = workspace_service.resolve_storage_root(storage_root, create=create)
    root = common.paths.get_generation_meta_root(storage_root=storage) / "workflow_sessions"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        message = f"Workflow-session root is unsafe: {root}"
        raise ValueError(message)
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
    elif not root.is_dir():
        message = f"Workflow-session root does not exist: {root}"
        raise FileNotFoundError(message)
    return root.resolve()


def _session_directory(session_id: str, *, storage_root: Path | str) -> Path:
    """Return one validated session directory below the selected storage root."""
    if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
        message = f"Malformed workflow session ID: {session_id!r}."
        raise ValueError(message)
    root = _sessions_root(storage_root, create=False)
    directory = (root / session_id).resolve()
    if not directory.is_relative_to(root):
        message = f"Workflow session escapes its owner root: {session_id!r}."
        raise ValueError(message)
    return directory


def _safe_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """Return exact safe argv without background recursion or control bytes."""
    values = tuple(arguments)
    if (
        not values
        or any(not isinstance(value, str) or not value or any(character in value for character in "\x00\r\n") for value in values)
        or "--background" in values
    ):
        message = "Background workflow argv is empty, unsafe, or still contains a launch-control flag."
        raise ValueError(message)
    return values


def _command_digest(
    *,
    source_commit: str,
    subcommand: str,
    arguments: Sequence[str],
    storage_root: Path,
) -> str:
    """Return the duplicate-controller identity for one semantic invocation."""
    return common.serialization.canonical_json_sha256(
        {
            "source_commit": source_commit,
            "subcommand": subcommand,
            "arguments": list(arguments),
            "storage_root": str(storage_root),
        }
    )


def _load_json(path: Path, *, expected_keys: frozenset[str], label: str) -> dict[str, Any]:
    """Load one exact owner-only workflow-session JSON object."""
    if not path.is_file() or path.is_symlink():
        message = f"Missing or unsafe {label}: {path}"
        raise FileNotFoundError(message)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not read {label}: {path}"
        raise ValueError(message) from error
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        message = f"Malformed {label}: {path}"
        raise ValueError(message)
    return payload


def _load_session(directory: Path) -> dict[str, Any]:
    """Load and validate one durable workflow session."""
    payload = _load_json(
        directory / "session.json",
        expected_keys=_SESSION_KEYS,
        label="workflow session metadata",
    )
    if (
        payload.get("schema_kind") != _SESSION_SCHEMA_KIND
        or payload.get("schema_version") != BACKGROUND_SCHEMA_VERSION
        or payload.get("workflow_session_id") != directory.name
        or payload.get("state") != "active"
        or payload.get("user") != getpass.getuser()
        or not isinstance(payload.get("storage_root"), str)
        or not Path(payload["storage_root"]).is_absolute()
        or payload.get("log_path")
        != str(Path(payload["storage_root"]) / "01_generation" / "meta" / "workflow_sessions" / directory.name / "workflow.log")
        or payload.get("command_path")
        != str(Path(payload["storage_root"]) / "01_generation" / "meta" / "workflow_sessions" / directory.name / "command.sh")
    ):
        message = f"Workflow session identity is invalid: {directory}"
        raise ValueError(message)
    source_service.validate_git_commit(payload.get("source_commit"))
    return payload


def _load_result(directory: Path) -> dict[str, Any] | None:
    """Load an optional exact terminal result."""
    path = directory / "result.json"
    if not path.exists():
        return None
    payload = _load_json(
        path,
        expected_keys=_RESULT_KEYS,
        label="workflow session result",
    )
    if (
        payload.get("schema_kind") != _RESULT_SCHEMA_KIND
        or payload.get("schema_version") != BACKGROUND_SCHEMA_VERSION
        or payload.get("workflow_session_id") != directory.name
        or payload.get("terminal_state") not in {"completed", "failed"}
        or isinstance(payload.get("exit_code"), bool)
        or not isinstance(payload.get("exit_code"), int)
    ):
        message = f"Workflow session result is invalid: {path}"
        raise ValueError(message)
    return payload


def _command_script(
    *,
    stable_script: Path,
    docker_python: Path,
    arguments: Sequence[str],
    session_id: str,
    storage_root: Path,
    log_path: Path,
) -> str:
    """Return an owner-only wrapper preserving output and the workflow exit code."""
    workflow = shlex.join((str(stable_script), *arguments))
    completion = shlex.join(
        (
            str(docker_python),
            "-m",
            "src.generation.cli.cli_generation",
            "complete-background-session",
            session_id,
            "--storage-root",
            str(storage_root),
        )
    )
    quoted_log = shlex.quote(str(log_path))
    return (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "umask 077\n"
        "export GENERATION_WORKFLOW_BACKGROUND_CHILD=1\n"
        "set +e\n"
        f"{workflow} > >(tee -a -- {quoted_log}) 2> >(tee -a -- {quoted_log} >&2)\n"
        "workflow_status=$?\n"
        "set -e\n"
        f'if ! {completion} --exit-code "${{workflow_status}}" >> {quoted_log} 2>&1; then\n'
        f"  printf '%s\n' 'Could not persist background workflow result.' >> {quoted_log}\n"
        "fi\n"
        'exit "${workflow_status}"\n'
    )


def create_background_session(
    subcommand: str,
    *,
    arguments: Sequence[str],
    source_commit: str,
    storage_root: Path | str,
    stable_script: Path | str,
    docker_python: Path | str,
    host_storage_root: Path | str,
    host_name: str,
    active_tmux_sessions: Sequence[str],
) -> dict[str, Any]:
    """Create one durable command or return the equivalent active session."""
    if subcommand not in _SUPPORTED_SUBCOMMANDS or _SAFE_SUBCOMMAND_PATTERN.fullmatch(subcommand) is None:
        message = f"--background is not supported for {subcommand!r}."
        raise ValueError(message)
    commit = source_service.validate_git_commit(source_commit)
    if not isinstance(host_name, str) or not host_name or any(character in host_name for character in "\x00\r\n\t"):
        message = "Background host name must be safe non-empty text."
        raise ValueError(message)
    argv = _safe_arguments(arguments)
    if argv[0] != subcommand:
        message = "Background subcommand disagrees with exact argv."
        raise ValueError(message)
    script = Path(stable_script).expanduser()
    runner = Path(docker_python).expanduser()
    host_storage = Path(host_storage_root).expanduser()
    paths = (script, runner, host_storage)
    if any(not value.is_absolute() or value == Path("/") or ".." in value.parts for value in paths):
        message = "Host workflow, Docker Python, and storage paths must be absolute, non-root, and traversal-free."
        raise ValueError(message)
    root = _sessions_root(storage_root, create=True)
    storage = root.parents[2]
    digest = _command_digest(
        source_commit=commit,
        subcommand=subcommand,
        arguments=argv,
        storage_root=host_storage,
    )
    active_names = frozenset(active_tmux_sessions)
    with common.locking.exclusive_file_lock(root / ".creation.lock", blocking=True):
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.is_symlink() or _SESSION_ID_PATTERN.fullmatch(directory.name) is None:
                continue
            existing = _load_session(directory)
            if existing["command_digest"] == digest and existing["tmux_session_name"] in active_names and _load_result(directory) is None:
                return {"status": "reused", **existing}
        now = _utc_now()
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        session_stem = f"gw-{timestamp}-{subcommand}-{digest[:8]}"
        collision_index = 0
        while True:
            collision_suffix = "" if collision_index == 0 else f"-{collision_index:02d}"
            session_id = f"{session_stem}{collision_suffix}"
            directory = _session_directory(session_id, storage_root=storage)
            if not directory.exists():
                break
            collision_index += 1
        directory.mkdir(mode=0o700)
        tmux_name = f"gw-{subcommand[:14]}-{timestamp[9:15]}-{digest[:8]}{collision_suffix}"
        local_log_path = directory / "workflow.log"
        local_command_path = directory / "command.sh"
        host_directory = host_storage / directory.relative_to(storage)
        log_path = host_directory / "workflow.log"
        command_path = host_directory / "command.sh"
        local_log_path.touch(mode=0o600)
        command = _command_script(
            stable_script=script,
            docker_python=runner,
            arguments=argv,
            session_id=session_id,
            storage_root=host_storage,
            log_path=log_path,
        )
        common.serialization.atomic_write_text(local_command_path, command)
        local_command_path.chmod(0o700)
        session = {
            "schema_kind": _SESSION_SCHEMA_KIND,
            "schema_version": BACKGROUND_SCHEMA_VERSION,
            "workflow_session_id": session_id,
            "tmux_session_name": tmux_name,
            "host": host_name,
            "user": getpass.getuser(),
            "source_commit": commit,
            "subcommand": subcommand,
            "argv": list(argv),
            "command_digest": digest,
            "storage_root": str(host_storage),
            "started_at": now.isoformat(),
            "state": "active",
            "log_path": str(log_path),
            "command_path": str(command_path),
        }
        common.serialization.atomic_write_json(directory / "session.json", session)
        (directory / "session.json").chmod(0o600)
    return {"status": "created", **session}


def _log_terminal_evidence(log_path: Path) -> tuple[list[str], list[str], str]:
    """Return run IDs and the last meaningful workflow stage from bounded log text."""
    if not log_path.is_file() or log_path.is_symlink():
        return [], [], "unavailable"
    size = log_path.stat().st_size
    maximum_bytes = 2 * 1024 * 1024
    with log_path.open("rb") as stream:
        if size > maximum_bytes:
            stream.seek(-maximum_bytes, os.SEEK_END)
        text = stream.read(maximum_bytes).decode("utf-8", errors="replace")
    campaign_ids = sorted(set(_RUN_ID_PATTERN.findall(text)).difference(_BENCHMARK_RUN_ID_PATTERN.findall(text)))
    benchmark_ids = sorted(set(_BENCHMARK_RUN_ID_PATTERN.findall(text)))
    final_stage = "unavailable"
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("DONE:", "DEFERRED:", "FAILED:", "[")):
            final_stage = stripped[:500]
            break
    return campaign_ids, benchmark_ids, final_stage


def complete_background_session(
    session_id: str,
    *,
    exit_code: int,
    storage_root: Path | str,
) -> dict[str, Any]:
    """Atomically record the exact child exit status and discovered run identities."""
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or not 0 <= exit_code <= _MAX_PROCESS_EXIT_CODE:
        message = f"Background workflow exit code must be in [0, 255], got {exit_code!r}."
        raise ValueError(message)
    directory = _session_directory(session_id, storage_root=storage_root)
    _load_session(directory)
    campaign_ids, benchmark_ids, final_stage = _log_terminal_evidence(directory / "workflow.log")
    result = {
        "schema_kind": _RESULT_SCHEMA_KIND,
        "schema_version": BACKGROUND_SCHEMA_VERSION,
        "workflow_session_id": session_id,
        "exit_code": exit_code,
        "terminal_state": "completed" if exit_code == 0 else "failed",
        "ended_at": _utc_now().isoformat(),
        "campaign_run_ids": campaign_ids,
        "benchmark_run_ids": benchmark_ids,
        "final_stage": final_stage,
    }
    path = directory / "result.json"
    if path.exists():
        existing = _load_result(directory)
        comparable = {key: value for key, value in result.items() if key != "ended_at"}
        if existing is None or any(existing.get(key) != value for key, value in comparable.items()):
            message = f"Existing workflow session result conflicts: {path}"
            raise FileExistsError(message)
        return existing
    common.serialization.atomic_write_json(path, result)
    path.chmod(0o600)
    return result


def inspect_background_session(
    session_id: str,
    *,
    storage_root: Path | str,
    active_tmux_sessions: Sequence[str],
) -> dict[str, Any]:
    """Return read-only durable and live status for one owned session."""
    directory = _session_directory(session_id, storage_root=storage_root)
    session = _load_session(directory)
    result = _load_result(directory)
    tmux_active = session["tmux_session_name"] in frozenset(active_tmux_sessions)
    if result is not None:
        workflow_state = result["terminal_state"]
    elif tmux_active:
        workflow_state = "running"
    else:
        workflow_state = "interrupted_or_host_rebooted"
    return {
        **session,
        "tmux_active": tmux_active,
        "workflow_state": workflow_state,
        "exit_code": None if result is None else result["exit_code"],
        "ended_at": None if result is None else result["ended_at"],
        "campaign_run_ids": [] if result is None else result["campaign_run_ids"],
        "benchmark_run_ids": [] if result is None else result["benchmark_run_ids"],
        "final_stage": "running" if result is None and tmux_active else "unavailable" if result is None else result["final_stage"],
    }


def list_background_sessions(
    *,
    storage_root: Path | str,
    active_tmux_sessions: Sequence[str],
) -> list[dict[str, Any]]:
    """Return read-only summaries for current-user sessions under one storage root."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    unresolved_root = common.paths.get_generation_meta_root(storage_root=storage) / "workflow_sessions"
    if not unresolved_root.exists() and not unresolved_root.is_symlink():
        return []
    root = _sessions_root(storage, create=False)
    sessions: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or _SESSION_ID_PATTERN.fullmatch(directory.name) is None:
            continue
        payload = _load_json(
            directory / "session.json",
            expected_keys=_SESSION_KEYS,
            label="workflow session metadata",
        )
        if payload.get("user") != getpass.getuser():
            continue
        sessions.append(
            inspect_background_session(
                directory.name,
                storage_root=storage_root,
                active_tmux_sessions=active_tmux_sessions,
            )
        )
    return sessions
