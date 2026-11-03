"""Check fake-COMSOL tracker cleanup at deterministic termination boundaries.

Responsibilities:
  - Deliver SIGTERM during registration, finalizer entry and decrement.
  - Assert complete, balanced tracker records after process termination.

Design principles:
  - Exercise the emitted fixture in a separate process with test-owned paths.
  - Select lifecycle boundaries explicitly rather than relying on timing.

This module does NOT:
  - Execute physical COMSOL solves or change production process control.
  - Replace the Generation concurrency and cancellation contracts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_TERMINATION_PROBE = """
import linecache
import os
import runpy
import signal
import sys

script, boundary = sys.argv[1:]
sys.argv = [script, '-batchlog', 'batch.log', '-batchlogout']

def terminate_at_boundary(frame, event, argument):
    if frame.f_code.co_filename != script:
        return terminate_at_boundary
    delta = frame.f_locals.get('delta')
    registration = (
        boundary == 'registration' and frame.f_code.co_name == 'update_tracker'
        and event == 'return' and delta == 1
    )
    cleanup = (
        boundary == 'cleanup' and frame.f_code.co_name == 'update_tracker'
        and event == 'line' and delta == -1
        and linecache.getline(script, frame.f_lineno).strip() == 'state["active"] += delta'
    )
    finalization = boundary == 'finalization' and (
        (event == 'call' and frame.f_code.co_name == 'unregister_delayed_checkout')
        or (event == 'line' and frame.f_code.co_name == '<module>'
            and linecache.getline(script, frame.f_lineno).strip()
            == 'signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})')
    )
    if registration or finalization or cleanup:
        sys.settrace(None)
        os.kill(os.getpid(), signal.SIGTERM)
    return terminate_at_boundary

sys.settrace(terminate_at_boundary)
runpy.run_path(script, run_name='__main__')
"""


@pytest.mark.integration
@pytest.mark.parametrize("boundary", ["registration", "finalization", "cleanup"])
def test_fake_comsol_tracker_balances_termination(
    fake_comsol: Path,
    tmp_path: Path,
    boundary: str,
) -> None:
    """Keep the live counter and persisted JSON correct across SIGTERM."""
    tracker = tmp_path / "tracker.json"
    environment = {
        **os.environ,
        "FAKE_COMSOL_MODE": "license_capacity_delayed",
        "FAKE_COMSOL_TRACKER": str(tracker),
        "FAKE_COMSOL_EXPECT_STARTS": "1",
    }
    result = subprocess.run(  # noqa: S603 -- Execute only the test-owned fixture with the maintained interpreter.
        [sys.executable, "-c", _TERMINATION_PROBE, str(fake_comsol), boundary],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 143, result.stderr
    assert json.loads(tracker.read_text(encoding="utf-8")) == {
        "active": 0,
        "maximum": 1,
        "starts": 1,
    }
