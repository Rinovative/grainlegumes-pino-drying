# ruff: noqa: PLR2004, S101, S603
"""Synthetic tests for the bounded Generation SSH transport."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _executable(path: Path, content: str) -> Path:
    """Write one test-owned executable."""
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _transport_environment(
    tmp_path: Path,
    outcomes: list[dict[str, Any]],
) -> tuple[dict[str, str], Path, Path, Path]:
    """Create one fake SSH transport with exact argument and stdin capture."""
    fake_bin = tmp_path / "bin"
    attempts = tmp_path / "attempts"
    temporary_parent = tmp_path / "temporary"
    fake_bin.mkdir()
    attempts.mkdir()
    temporary_parent.mkdir()
    fake_ssh = f"""#!{sys.executable}
import json
import os
import pathlib
import sys

attempts = pathlib.Path(os.environ["FAKE_SSH_ATTEMPTS"])
count_path = attempts / "count"
count = int(count_path.read_text(encoding="utf-8")) if count_path.exists() else 0
index = count + 1
count_path.write_text(str(index), encoding="utf-8")
(attempts / f"{{index}}.stdin").write_bytes(sys.stdin.buffer.read())
(attempts / f"{{index}}.args.json").write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
outcomes = json.loads(os.environ["FAKE_SSH_OUTCOMES"])
outcome = outcomes[min(count, len(outcomes) - 1)]
sys.stdout.buffer.write(outcome.get("stdout", "").encode())
sys.stderr.buffer.write(outcome.get("stderr", "").encode())
raise SystemExit(int(outcome["status"]))
"""
    _executable(fake_bin / "ssh", fake_ssh)
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "SSH_HELPER": str(repository / "scripts/generation_ssh_transport.sh"),
            "FAKE_SSH_ATTEMPTS": str(attempts),
            "FAKE_SSH_OUTCOMES": json.dumps(outcomes),
            "TEST_TEMP_PARENT": str(temporary_parent),
            "SLEEP_LOG": str(tmp_path / "sleep.log"),
        }
    )
    return environment, attempts, temporary_parent, tmp_path / "sleep.log"


def _run_transport(script: str, environment: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    """Run one isolated sourced transport helper through Bash."""
    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(
        [bash, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
    )


def _attempt_count(attempts: Path) -> int:
    """Return the fake SSH invocation count."""
    return int((attempts / "count").read_text(encoding="utf-8"))


def test_retry_replays_exact_stdin_and_arguments_and_emits_only_success_stdout(
    tmp_path: Path,
) -> None:
    """Recover once without losing heredoc bytes or contaminating stdout."""
    outcomes = [
        {
            "status": 255,
            "stdout": "discarded failed stdout\n",
            "stderr": "Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).\n",
        },
        {"status": 0, "stdout": "machine-output\n", "stderr": "remote-success-note\n"},
    ]
    environment, attempts, temporary_parent, sleep_log = _transport_environment(tmp_path, outcomes)
    remote_script = b"set -euo pipefail\nprintf 'tail-without-loss'\n# trailing content\n"
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { printf '%s\n' "$1" >> "${SLEEP_LOG}"; }
_remote_bash_retryable_with_delays "campaign status read" "0,0" \
  "cpu.example" "alpha" "two words" <<'REMOTE'
set -euo pipefail
printf 'tail-without-loss'
# trailing content
REMOTE
"""

    result = _run_transport(command, environment)

    assert result.returncode == 0
    assert result.stdout == b"machine-output\n"
    assert b"discarded failed stdout" not in result.stdout
    assert b"WARNING: transient SSH failure during campaign status read" in result.stderr
    assert b"Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)." in result.stderr
    assert b"SSH connection recovered:" in result.stderr
    assert b"remote-success-note" in result.stderr
    assert _attempt_count(attempts) == 2
    assert (attempts / "1.stdin").read_bytes() == remote_script
    assert (attempts / "2.stdin").read_bytes() == remote_script
    first_arguments = json.loads((attempts / "1.args.json").read_text(encoding="utf-8"))
    second_arguments = json.loads((attempts / "2.args.json").read_text(encoding="utf-8"))
    assert first_arguments == second_arguments
    assert first_arguments[:4] == ["-o", "BatchMode=yes", "--", "cpu.example"]
    assert first_arguments[4] == "bash -l -s -- alpha two\\ words"
    assert sleep_log.read_text(encoding="utf-8").splitlines() == ["0"]
    assert list(temporary_parent.iterdir()) == []


def test_successful_transport_uses_one_attempt_without_retry_diagnostics(
    tmp_path: Path,
) -> None:
    """Preserve the one-shot success path when SSH is healthy."""
    environment, attempts, temporary_parent, sleep_log = _transport_environment(
        tmp_path,
        [{"status": 0, "stdout": "healthy\n", "stderr": ""}],
    )
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { printf '%s\n' "$1" >> "${SLEEP_LOG}"; }
printf 'same bytes' | remote_bash_retryable "campaign status read" "cpu.example"
"""

    result = _run_transport(command, environment)

    assert result.returncode == 0
    assert result.stdout == b"healthy\n"
    assert result.stderr == b""
    assert _attempt_count(attempts) == 1
    assert not sleep_log.exists()
    assert list(temporary_parent.iterdir()) == []


@pytest.mark.parametrize(
    "reason",
    [
        "Connection closed by 192.0.2.1 port 22",
        "ssh: connect to host cpu.example port 22: No route to host",
        "ssh: Could not resolve hostname cpu.example: Temporary failure in name resolution",
        "ssh: Could not resolve hostname cpu.example: Name or service not known",
    ],
)
def test_allowlisted_transport_diagnostics_can_recover(
    tmp_path: Path,
    reason: str,
) -> None:
    """Recognize the maintained transport allowlist only with SSH status 255."""
    environment, attempts, temporary_parent, _sleep_log = _transport_environment(
        tmp_path,
        [
            {"status": 255, "stderr": f"{reason}\n"},
            {"status": 0, "stdout": "healthy\n"},
        ],
    )
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { :; }
printf 'same bytes' | _remote_bash_retryable_with_delays \
  "campaign status read" "0" "cpu.example"
"""

    result = _run_transport(command, environment)

    assert result.returncode == 0
    assert result.stdout == b"healthy\n"
    assert _attempt_count(attempts) == 2
    assert reason.encode() in result.stderr
    assert list(temporary_parent.iterdir()) == []


@pytest.mark.parametrize("delay_policy", ["", "-1", "1,,2", "3601"])
def test_invalid_internal_retry_policy_fails_before_ssh(
    tmp_path: Path,
    delay_policy: str,
) -> None:
    """Reject malformed or unbounded internal retry values before transport."""
    environment, attempts, temporary_parent, _sleep_log = _transport_environment(
        tmp_path,
        [{"status": 0, "stdout": "must not run\n"}],
    )
    environment["TEST_DELAY_POLICY"] = delay_policy
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
printf 'same bytes' | _remote_bash_retryable_with_delays \
  "campaign status read" "${TEST_DELAY_POLICY}" "cpu.example"
"""

    result = _run_transport(command, environment)

    assert result.returncode == 2
    assert not (attempts / "count").exists()
    assert list(temporary_parent.iterdir()) == []


def test_multiple_transport_failures_use_ordered_delays_and_stop_after_recovery(
    tmp_path: Path,
) -> None:
    """Use deterministic delays without another attempt after success."""
    outcomes = [
        {"status": 255, "stderr": "Connection timed out\n"},
        {"status": 255, "stderr": "Connection reset by peer\n"},
        {"status": 0, "stdout": "recovered\n"},
        {"status": 255, "stderr": "must not run\n"},
    ]
    environment, attempts, temporary_parent, sleep_log = _transport_environment(tmp_path, outcomes)
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { printf '%s\n' "$1" >> "${SLEEP_LOG}"; }
printf 'same bytes' | _remote_bash_retryable_with_delays \
  "remote setup verification" "1,2,3" "cpu.example" "argument"
"""

    result = _run_transport(command, environment)

    assert result.returncode == 0
    assert result.stdout == b"recovered\n"
    assert _attempt_count(attempts) == 3
    assert sleep_log.read_text(encoding="utf-8").splitlines() == ["1", "2"]
    assert list(temporary_parent.iterdir()) == []


def test_default_permission_denial_policy_exhausts_after_five_attempts(
    tmp_path: Path,
) -> None:
    """Bound repeated authentication failure to the maintained 110-second schedule."""
    permission = "Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)."
    outcomes = [{"status": 255, "stderr": f"{permission} attempt={index}\n"} for index in range(1, 6)]
    environment, attempts, temporary_parent, sleep_log = _transport_environment(tmp_path, outcomes)
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { printf '%s\n' "$1" >> "${SLEEP_LOG}"; }
printf 'same bytes' | remote_bash_retryable "remote HOME resolution" "cpu.example"
"""

    result = _run_transport(command, environment)

    assert result.returncode == 255
    assert result.stdout == b""
    assert _attempt_count(attempts) == 5
    assert sleep_log.read_text(encoding="utf-8").splitlines() == ["5", "15", "30", "60"]
    assert result.stderr.count(b"WARNING: transient SSH failure") == 4
    assert b"attempt=5" in result.stderr
    assert b"Persistent SSH transport/authentication failure after 5 attempts" in result.stderr
    assert b"No CPU-side campaign or Slurm cancellation was requested" in result.stderr
    assert list(temporary_parent.iterdir()) == []


@pytest.mark.parametrize(
    ("status", "stderr"),
    [
        (1, "Scientific validation failure.\n"),
        (2, "Malformed campaign evidence.\n"),
        (255, "REMOTE HOST IDENTIFICATION HAS CHANGED!\nHost key verification failed.\n"),
        (255, "Bad owner or permissions on /safe/test/config\n"),
        (255, "Bad configuration option: UnsafeOption\n"),
        (255, "Identity file /safe/test/key not accessible: No such file or directory\n"),
        (
            255,
            "Traceback (most recent call last):\nRuntimeError: Connection reset by peer\n",
        ),
    ],
)
def test_nonretryable_remote_and_ssh_failures_run_once(
    tmp_path: Path,
    status: int,
    stderr: str,
) -> None:
    """Do not replay ordinary remote errors or unsafe SSH configuration failures."""
    environment, attempts, temporary_parent, sleep_log = _transport_environment(
        tmp_path,
        [{"status": status, "stdout": "discarded\n", "stderr": stderr}],
    )
    command = r"""
set -Eeuo pipefail
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { printf '%s\n' "$1" >> "${SLEEP_LOG}"; }
printf 'same bytes' | _remote_bash_retryable_with_delays \
  "campaign status read" "0,0" "cpu.example"
"""

    result = _run_transport(command, environment)

    assert result.returncode == status
    assert result.stdout == b""
    assert result.stderr == stderr.encode()
    assert _attempt_count(attempts) == 1
    assert not sleep_log.exists()
    assert list(temporary_parent.iterdir()) == []


def test_interrupt_while_buffering_stdin_returns_130_and_cleans_partial_script(
    tmp_path: Path,
) -> None:
    """Install cleanup before a blocked heredoc buffer can be interrupted."""
    environment, attempts, temporary_parent, _sleep_log = _transport_environment(
        tmp_path,
        [{"status": 0, "stdout": "must not run\n"}],
    )
    ready = tmp_path / "buffer-ready"
    environment["BUFFER_READY"] = str(ready)
    command = r"""
set -Eeuo pipefail
trap 'exit 130' INT
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
produce_partial_script() {
  printf 'partial remote script'
  : > "${BUFFER_READY}"
  sleep 30
}
produce_partial_script | remote_bash_retryable "campaign status read" "cpu.example"
"""
    bash = shutil.which("bash")
    assert bash is not None
    process = subprocess.Popen(
        [bash, "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready.exists() and any(temporary_parent.iterdir()):
                break
            time.sleep(0.01)
        assert ready.exists()
        assert any(temporary_parent.iterdir())
        os.killpg(process.pid, signal.SIGINT)
        stdout, _stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert process.returncode == 130
    assert stdout == b""
    assert not (attempts / "count").exists()
    assert list(temporary_parent.iterdir()) == []


def test_interrupt_during_retry_wait_returns_130_and_cleans_buffers(
    tmp_path: Path,
) -> None:
    """Clean the replay buffer promptly when the workflow process is interrupted."""
    environment, attempts, temporary_parent, _sleep_log = _transport_environment(
        tmp_path,
        [{"status": 255, "stderr": "Connection refused\n"}],
    )
    ready = tmp_path / "sleep-ready"
    environment["SLEEP_READY"] = str(ready)
    command = r"""
set -Eeuo pipefail
trap 'exit 130' INT
source "${SSH_HELPER}"
_generation_ssh_retry_temp_parent() { printf '%s\n' "${TEST_TEMP_PARENT}"; }
_generation_ssh_retry_sleep() { : > "${SLEEP_READY}"; sleep 30; }
printf 'same bytes' | remote_bash_retryable "campaign status read" "cpu.example"
"""
    bash = shutil.which("bash")
    assert bash is not None
    process = subprocess.Popen(
        [bash, "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert process.returncode == 130
    assert stdout == b""
    assert b"WARNING: transient SSH failure" in stderr
    assert _attempt_count(attempts) == 1
    assert list(temporary_parent.iterdir()) == []
