# ruff: noqa: S101, S603
"""Durable background workflow sessions preserve exact execution evidence."""

from __future__ import annotations

import json
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from src.generation import generation_background as background_service

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.integration

_COMMIT = "a" * 40
_CAMPAIGN_RUN_ID = "synthetic_campaign__0123456789abcdef"
_BENCHMARK_RUN_ID = "core_scaling_transient__fedcba9876543210"
_OWNER_DIRECTORY_MODE = 0o700
_OWNER_FILE_MODE = 0o600
_CHILD_EXIT_CODE = 7


def _executable(path: Path, lines: Sequence[str]) -> Path:
    """Write one test-owned executable with owner execution permission."""
    path.write_text("\n".join(("#!/usr/bin/env bash", "set -euo pipefail", *lines, "")), encoding="utf-8")
    path.chmod(_OWNER_DIRECTORY_MODE)
    return path


def test_background_session_executes_exact_argv_and_preserves_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect creation, duplicate reuse, exit status, logs, and terminal inspection."""
    storage = tmp_path / "storage root"
    storage.mkdir()
    observed_arguments = tmp_path / "observed arguments.txt"
    workflow = _executable(
        tmp_path / "workflow with spaces.sh",
        (
            f"printf '%s\\n' \"$@\" > {shlex.quote(str(observed_arguments))}",
            f"printf 'campaign_run_id=%s\\n' {shlex.quote(_CAMPAIGN_RUN_ID)}",
            f"printf 'benchmark_run_id=%s\\n' {shlex.quote(_BENCHMARK_RUN_ID)}",
            "printf 'FAILED: synthetic child result\\n'",
            f"exit {_CHILD_EXIT_CODE}",
        ),
    )
    docker_python = _executable(
        tmp_path / "docker python.sh",
        (f'exec {shlex.quote(sys.executable)} "$@"',),
    )
    fixed_now = datetime(2026, 8, 18, 15, 45, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(background_service, "_utc_now", lambda: fixed_now)
    arguments = (
        "all",
        "configs/generation/campaigns/transient_drying/family_generalization.yaml",
        "--git-commit",
        _COMMIT,
        "--keep-cpu-source",
    )

    created = background_service.create_background_session(
        "all",
        arguments=arguments,
        source_commit=_COMMIT,
        storage_root=storage,
        stable_script=workflow,
        docker_python=docker_python,
        host_storage_root=storage,
        host_name="synthetic-host",
        active_tmux_sessions=(),
    )
    assert created["status"] == "created"
    assert created["schema_version"] == 1
    directory = storage / "01_generation/meta/workflow_sessions" / created["workflow_session_id"]
    assert stat.S_IMODE(directory.stat().st_mode) == _OWNER_DIRECTORY_MODE
    assert stat.S_IMODE((directory / "session.json").stat().st_mode) == _OWNER_FILE_MODE
    assert stat.S_IMODE((directory / "workflow.log").stat().st_mode) == _OWNER_FILE_MODE
    assert stat.S_IMODE((directory / "command.sh").stat().st_mode) == _OWNER_DIRECTORY_MODE
    command_text = (directory / "command.sh").read_text(encoding="utf-8")
    assert "GENERATION_WORKFLOW_BACKGROUND_CHILD=1" in command_text
    assert "--background" not in command_text

    reused = background_service.create_background_session(
        "all",
        arguments=arguments,
        source_commit=_COMMIT,
        storage_root=storage,
        stable_script=workflow,
        docker_python=docker_python,
        host_storage_root=storage,
        host_name="synthetic-host",
        active_tmux_sessions=(created["tmux_session_name"],),
    )
    assert reused["status"] == "reused"
    assert reused["workflow_session_id"] == created["workflow_session_id"]

    completed = subprocess.run([str(directory / "command.sh")], check=False)
    assert completed.returncode == _CHILD_EXIT_CODE
    assert observed_arguments.read_text(encoding="utf-8").splitlines() == list(arguments)
    status = background_service.inspect_background_session(
        created["workflow_session_id"],
        storage_root=storage,
        active_tmux_sessions=(),
    )
    assert status["workflow_state"] == "failed"
    assert status["exit_code"] == _CHILD_EXIT_CODE
    assert status["tmux_active"] is False
    assert status["campaign_run_ids"] == [_CAMPAIGN_RUN_ID]
    assert status["benchmark_run_ids"] == [_BENCHMARK_RUN_ID]
    assert status["final_stage"] == "FAILED: synthetic child result"

    relaunched = background_service.create_background_session(
        "all",
        arguments=arguments,
        source_commit=_COMMIT,
        storage_root=storage,
        stable_script=workflow,
        docker_python=docker_python,
        host_storage_root=storage,
        host_name="synthetic-host",
        active_tmux_sessions=(created["tmux_session_name"],),
    )
    assert relaunched["status"] == "created"
    assert relaunched["workflow_session_id"].endswith("-01")
    assert relaunched["tmux_session_name"] != created["tmux_session_name"]
    listed = background_service.list_background_sessions(
        storage_root=storage,
        active_tmux_sessions=(relaunched["tmux_session_name"],),
    )
    assert [item["workflow_state"] for item in listed] == ["failed", "running"]

    completed_session_path = directory / "session.json"
    completed_session = json.loads(completed_session_path.read_text(encoding="utf-8"))
    completed_session["user"] = "another-user"
    completed_session_path.write_text(
        json.dumps(completed_session, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    owned = background_service.list_background_sessions(
        storage_root=storage,
        active_tmux_sessions=(relaunched["tmux_session_name"],),
    )
    assert [item["workflow_session_id"] for item in owned] == [relaunched["workflow_session_id"]]


def test_background_listing_is_read_only_when_no_session_root_exists(
    tmp_path: Path,
) -> None:
    """Listing an empty existing storage root must not create session state."""
    storage = tmp_path / "storage"
    storage.mkdir()
    sessions_root = storage / "01_generation/meta/workflow_sessions"

    assert (
        background_service.list_background_sessions(
            storage_root=storage,
            active_tmux_sessions=(),
        )
        == []
    )
    assert not sessions_root.exists()

    with pytest.raises(FileNotFoundError, match="Workflow-session root"):
        background_service.inspect_background_session(
            "gw-20260818T154501Z-all-01234567",
            storage_root=storage,
            active_tmux_sessions=(),
        )
    assert not sessions_root.exists()


def test_background_session_rejects_recursive_or_unsupported_argv(tmp_path: Path) -> None:
    """Keep launch-control flags and short unsupported commands out of children."""
    storage = tmp_path / "storage"
    storage.mkdir()
    common = {
        "source_commit": _COMMIT,
        "storage_root": storage,
        "stable_script": tmp_path / "workflow.sh",
        "docker_python": tmp_path / "docker-python.sh",
        "host_storage_root": storage,
        "host_name": "synthetic-host",
        "active_tmux_sessions": (),
    }
    with pytest.raises(ValueError, match="launch-control"):
        background_service.create_background_session(
            "all",
            arguments=("all", "--background"),
            **common,
        )
    with pytest.raises(ValueError, match="not supported"):
        background_service.create_background_session(
            "cancel",
            arguments=("cancel", _CAMPAIGN_RUN_ID),
            **common,
        )
