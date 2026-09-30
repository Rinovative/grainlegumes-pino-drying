# ruff: noqa: S101, PLR0911
"""
test_generation_controller.py

Exercise Generation lifecycle decisions against isolated service evidence.

Responsibilities:
  - Verify fresh, repeated, partial, repair, and completion continuations
  - Assert exact publication and retention gates before terminal success
  - Keep scheduler and scientific services replaceable at one typed port

Design principles:
  - Tests assert service effects and ordering rather than controller structure
  - No scientific storage, solver, or live scheduler is touched

This module does NOT:
  - Reimplement campaign or benchmark persisted-data validation
  - Simulate COMSOL numerical behavior
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.generation.generation_controller import GenerationController, GenerationControllerError, LeafResult, RunPlan
from src.generation.runtime import generation_runtime_host as host_service
from src.generation.runtime.generation_runtime_host import GenerationCommandError

_RUN = "steady_flow_test__0123456789abcdef"
_SOURCE = "steady_flow_source__fedcba9876543210"
_REPLACEMENT = "steady_flow_replacement__1234567890abcdef"
_COMPLETION = "completion__" + "a" * 24
_DIGEST = "b" * 64
_CONFIG = "configs/generation/campaigns/steady_flow/test.yaml"


def _plan(kind: str = "campaign") -> RunPlan:
    payload: dict[str, Any] = {"run_kind": kind, "identity": _RUN, "config_path": _CONFIG, "children": []}
    return RunPlan(kind, _RUN, _CONFIG, "steady_flow_id_dataset", "steady_flow", (), payload)


def _resources() -> dict[str, Any]:
    return {
        "campaign_purpose": "steady_flow_id_dataset",
        "execution_resources": {
            "cluster": {"cores_per_case": 4, "cores_per_node": 32, "wall_time": "01:05:00"},
            "submission": {"max_admission_cases": 2, "max_running_cases": None, "poll_interval_seconds": 1},
            "site": {
                "partition": "standard",
                "scheduler": "slurm",
                "python_module": "Python/3.12",
                "comsol_module": "Comsol/v6.4",
                "python_executable": "python3",
                "comsol_executable": "comsol",
            },
        },
    }


def _monitor(state: str, run_id: str = _RUN) -> str:
    return f"campaign-monitor\t{state}\t{_DIGEST}\t{_DIGEST}\nsource-monitor\t{run_id}\t{state}\tretained\t128\teligible\tfalse\nsummary"


@dataclass
class FakePort:
    """Return controlled service evidence and retain observable operation order."""

    storage: Path
    responses: dict[str, list[str | GenerationCommandError]] = field(default_factory=dict)
    repository: Path = Path("test-repository")
    commit: str = "c" * 40
    calls: list[tuple[str, tuple[str, ...], bool | None]] = field(default_factory=list)
    published: bool = False
    terminal: bool = False

    def config_file(self, value: str) -> tuple[Path, str]:
        """Return the configured fixture path without touching a repository."""
        relative = value.removeprefix(str(self.repository) + "/")
        return self.repository / relative, relative

    def comsol_version(self, module: str, executable: str) -> str:
        """Return fixed native runtime identity for benchmark decisions."""
        assert module == "Comsol/v6.4"
        assert executable == "comsol"
        return "COMSOL Multiphysics 6.4.0.293"

    def verify_execution(self) -> None:
        """Treat the isolated worker boundary as ready."""

    def verify_source(self) -> None:
        """Treat the isolated source as unchanged."""

    def call(
        self,
        operation: str,
        *arguments: str,
        scheduled: bool | None = None,
        benchmark_preflight: bool = False,
        partition: str = "standard",
        wall_time: str = "02:00:00",
    ) -> str:
        """Return one queued response or the default persisted service state."""
        del benchmark_preflight, partition, wall_time
        self.calls.append((operation, arguments, scheduled))
        queued = self.responses.get(operation)
        if queued:
            value = queued.pop(0)
            if isinstance(value, GenerationCommandError):
                raise value
            return value
        if operation == "validate-config":
            return json.dumps(_resources())
        if operation == "find-completion-parent":
            return json.dumps({"status": "fresh"})
        if operation == "validate-all-workflow" and not self.terminal:
            raise GenerationCommandError(operation, 2, "not terminal")
        if operation == "validate-published-campaign" and not self.published:
            raise GenerationCommandError(operation, 2, "not published")
        if operation == "find-compatible-campaign-source":
            return json.dumps({"status": "missing", "campaign_run_id": None})
        if operation == "prepare-campaign-inputs":
            return "canonical-inputs\t1\t0"
        if operation == "submit-campaign":
            return json.dumps({"campaign_run_id": _RUN})
        if operation == "resume-campaign":
            return _monitor("successful")
        if operation == "campaign-transfer-authority":
            return '{"source":"exact"}'
        if operation == "repair-transferred-campaign":
            self.published = True
        if operation == "build-campaign-datasets":
            partial = "--partial" in arguments
            return json.dumps({"status": "incomplete" if partial else "complete", "declared_package_count": 1})
        if operation == "prepare-all-workflow":
            self.terminal = True
        if operation == "campaign-source-status":
            return f"source-status\t{arguments[0]}\tcomplete\tretained\t128\teligible\tfalse"
        return ""


@pytest.fixture
def port(tmp_path: Path) -> FakePort:
    """Give each test an isolated durable storage root."""
    storage = tmp_path / "storage"
    storage.mkdir()
    return FakePort(storage)


def _operations(port: FakePort) -> list[str]:
    return [name for name, _, _ in port.calls]


def test_fresh_campaign_reaches_publication_only_after_terminal_monitor(port: FakePort) -> None:
    """A fresh run submits once, repairs exact publication, and validates gates."""
    result = GenerationController(port).run(_plan(), _CONFIG)

    assert isinstance(result, LeafResult)
    assert result.run_id == _RUN
    assert result.result == "OK"
    operations = _operations(port)
    assert operations.index("submit-campaign") < operations.index("resume-campaign")
    assert operations.index("validate-campaign-terminal") < operations.index("campaign-transfer-authority")
    assert operations.index("repair-transferred-campaign") < operations.index("build-campaign-datasets")
    assert operations.index("build-campaign-datasets") < operations.index("prepare-all-workflow")
    assert operations[-1] == "validate-campaign-package-state"
    assert any(name == "prepare-campaign-inputs" and scheduled for name, _, scheduled in port.calls)


def test_repeated_completed_run_does_not_submit_or_republish(port: FakePort) -> None:
    """Persisted terminal evidence makes repeat invocation a read-only reuse."""
    port.terminal = True
    result = GenerationController(port).run(_plan(), _CONFIG)

    assert isinstance(result, LeafResult)
    assert result.result == "REUSED"
    assert "prepare-campaign-inputs" not in _operations(port)
    assert "submit-campaign" not in _operations(port)
    assert "build-campaign-datasets" not in _operations(port)


def test_partial_run_uses_partial_publication_and_retained_source(port: FakePort) -> None:
    """Failed cases preserve successful work and never pass a complete gate."""
    port.responses["resume-campaign"] = [_monitor("completed_with_failures")]
    port.responses["build-campaign-datasets"] = [json.dumps({"status": "incomplete", "declared_package_count": 1})]
    result = GenerationController(port).run(_plan(), _CONFIG)

    assert isinstance(result, LeafResult)
    assert result.partial
    operations = _operations(port)
    assert "repair-partial-campaign-publication" in operations
    assert "repair-transferred-campaign" not in operations
    assert "validate-campaign-terminal" not in operations
    assert any(name == "validate-all-workflow" and "--partial" in args for name, args, _ in port.calls)
    assert any(name == "prepare-all-workflow" and {"--partial", "--keep-cpu-source"}.issubset(args) for name, args, _ in port.calls)


def test_completed_compatible_source_extends_packages_without_solver_work(port: FakePort) -> None:
    """A compatible source with new packages continues from publication."""
    port.responses["find-compatible-campaign-source"] = [
        json.dumps(
            {
                "status": "compatible_complete",
                "campaign_run_id": _SOURCE,
                "artifact_set_sha256": _DIGEST,
                "package_state": "extension_required",
            }
        )
    ]
    port.responses["validate-all-workflow"] = [
        GenerationCommandError("validate-all-workflow", 2, "current plan incomplete"),
        "",
    ]
    controller = GenerationController(port)
    result = controller.run(_plan(), _CONFIG)

    assert isinstance(result, LeafResult)
    assert result.run_id == _SOURCE
    assert "build-campaign-datasets" in _operations(port)
    assert "prepare-campaign-inputs" not in _operations(port)
    assert "submit-campaign" not in _operations(port)


def test_failed_submission_stops_before_monitor_and_records_stage(port: FakePort) -> None:
    """A failed Slurm-facing submission cannot advance to publication."""
    port.responses["submit-campaign"] = [GenerationCommandError("submit-campaign", 7, "scheduler unavailable")]
    with pytest.raises(GenerationControllerError, match="Shared campaign submission failed"):
        GenerationController(port).run(_plan(), _CONFIG)
    assert "resume-campaign" not in _operations(port)
    assert "build-campaign-datasets" not in _operations(port)
    assert "record-workflow-failure" in _operations(port)
    receipt_args = next(args for name, args, _ in port.calls if name == "record-workflow-failure")
    assert receipt_args[receipt_args.index("--cpu-bytes-retained") + 1] == "128"


def test_failure_does_not_invent_zero_retained_bytes(port: FakePort) -> None:
    """Missing exact source-size evidence cannot become a false failure receipt."""
    port.responses["submit-campaign"] = [GenerationCommandError("submit-campaign", 7, "scheduler unavailable")]
    port.responses["campaign-source-status"] = [GenerationCommandError("campaign-source-status", 2, "source absent")]
    with pytest.raises(GenerationControllerError, match="Shared campaign submission failed"):
        GenerationController(port).run(_plan(), _CONFIG)
    assert "record-workflow-failure" not in _operations(port)


def test_paired_publication_rechecks_source_after_children(port: FakePort) -> None:
    """Changed source after child completion blocks the paired receipt."""
    port.terminal = True
    port.responses["find-compatible-technical-smoke-run"] = [
        json.dumps({"status": "missing"}),
        json.dumps({"status": "missing"}),
    ]
    second = RunPlan("campaign", _SOURCE, _CONFIG, "technical_runtime_smoke", "transient_drying", (), {})
    first = RunPlan("campaign", _RUN, _CONFIG, "technical_runtime_smoke", "steady_flow", (), {})
    workflow = RunPlan("workflow", "workflow__0123456789abcdef", _CONFIG, "-", "-", (first, second), {})

    def reject_source() -> None:
        message = "Generation source fingerprint changed"
        raise ValueError(message)

    port.verify_source = reject_source  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="source fingerprint changed"):
        GenerationController(port).run(workflow, _CONFIG)
    assert "finalize-technical-smoke-evidence" not in _operations(port)
    assert "finalize-real-smoke" not in _operations(port)


def test_completion_requires_parent_evidence_inside_storage(port: FakePort, tmp_path: Path) -> None:
    """A symlink cannot redirect completion admission outside durable storage."""
    outside = tmp_path / "unrelated-evidence.json"
    outside.write_text("{}", encoding="utf-8")
    partial = port.storage / "partial.json"
    partial.symlink_to(outside)
    port.responses["find-completion-parent"] = [
        json.dumps(
            {
                "status": "compatible_partial",
                "parent_run_id": _RUN,
                "completion_id": _COMPLETION,
                "parent_partial_path": str(partial),
                "parent_partial_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        )
    ]
    with pytest.raises(GenerationControllerError, match="escaped durable storage"):
        GenerationController(port).run(_plan(), _CONFIG, pool_size=4)
    assert "initialize-campaign-completion" not in _operations(port)


@pytest.mark.parametrize("supplied_pool", [None, 4])
def test_completion_rejects_unrelated_cleanup_membership(port: FakePort, supplied_pool: int | None) -> None:
    """Only successful replacement terminal batches may enter cleanup."""
    partial = port.storage / "partial.json"
    partial.write_text("exact", encoding="utf-8")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    state = port.storage / f"01_generation/meta/completions/{_COMPLETION}/completion.json"
    state.parent.mkdir(parents=True)
    state.write_text("state", encoding="utf-8")
    parent: dict[str, Any] = {
        "status": "compatible_partial",
        "parent_run_id": _RUN,
        "completion_id": _COMPLETION,
        "parent_partial_path": str(partial),
        "parent_partial_sha256": digest,
    }
    if supplied_pool is None:
        parent.update({"completion_status": "active", "replacement_pool_size": 4})
    port.responses.update(
        {
            "find-completion-parent": [json.dumps(parent)],
            "initialize-campaign-completion": [json.dumps({"status": "active", "completion_id": _COMPLETION})],
            "advance-campaign-completion": [json.dumps({"status": "complete", "completion_id": _COMPLETION})],
            "campaign-completion-transfer-plan": [
                json.dumps(
                    {
                        "completion_id": _COMPLETION,
                        "parent_run_id": _RUN,
                        "parent_partial_sha256": digest,
                        "completion_state_path": str(state),
                        "completion_state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                        "replacement_runs": [{"campaign_run_id": _REPLACEMENT, "campaign_state": "complete", "partial": False}],
                        "replacement_campaigns": [
                            {
                                "candidate_id": "replacement__" + "d" * 24,
                                "target_batch_id": "original_batch",
                                "campaign_run_id": _REPLACEMENT,
                                "terminal_batch_id": "replacement_good",
                            }
                        ],
                    }
                )
            ],
            "campaign-completion-cleanup-plan": [
                json.dumps(
                    {
                        "eligible": True,
                        "sources": [{"terminal_batch_id": "unrelated_source"}],
                    }
                )
            ],
        }
    )
    with pytest.raises(GenerationControllerError, match="cleanup plan differs"):
        GenerationController(port).run(_plan(), _CONFIG, pool_size=supplied_pool)
    assert "build-campaign-completion-lifecycle" in _operations(port)
    assert "validate-campaign-completion-lifecycle" in _operations(port)
    initialization = next(args for name, args, _ in port.calls if name == "initialize-campaign-completion")
    assert ("--replacement-pool-size" in initialization) == (supplied_pool is not None)
    assert initialization[initialization.index("--parent-partial") + 1] == str(partial)


@pytest.mark.parametrize("session_status", ["created", "reused", "launch_failure"])
def test_background_host_preserves_durable_process_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_status: str) -> None:
    """Launch only newly owned sessions and persist a failed process start."""
    command = tmp_path / "background command.sh"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o700)
    calls: list[str] = []
    processes: list[tuple[str, ...]] = []
    session = {
        "status": "reused" if session_status == "reused" else "created",
        "workflow_session_id": "test-session",
        "tmux_session_name": "test-tmux",
        "log_path": str(tmp_path / "workflow.log"),
        "command_path": str(command),
    }

    def call(_self: host_service.GenerationHost, operation: str, *arguments: str, **_options: Any) -> str:
        calls.append(operation)
        if operation == "complete-background-session":
            assert arguments[:3] == ("test-session", "--exit-code", "1")
        return json.dumps(session)

    def run(arguments: tuple[str, ...], **_options: Any) -> Any:
        processes.append(arguments)
        return host_service.subprocess.CompletedProcess(arguments, int(session_status == "launch_failure" and arguments[1] == "new-session"), "", "")

    monkeypatch.delenv("GENERATION_WORKFLOW_BACKGROUND_CHILD", raising=False)
    monkeypatch.setattr(host_service.GenerationHost, "call", call)
    monkeypatch.setattr(host_service.shutil, "which", lambda _name: "/test/tmux")
    monkeypatch.setattr(host_service.subprocess, "run", run)
    layout = host_service.HostLayout(tmp_path, tmp_path / "storage", tmp_path / "runtime", tmp_path / "native", {})
    host = host_service.GenerationHost(layout, host_service.SourceAdmission("c" * 40, _DIGEST))
    if session_status == "launch_failure":
        with pytest.raises(RuntimeError, match="could not start"):
            host.launch_background(("run", _CONFIG, "--background"))
        assert calls == ["create-background-session", "complete-background-session"]
    else:
        result = host.launch_background(("run", _CONFIG, "--background"))
        assert result["status"] == ("reused" if session_status == "reused" else "started")
        assert result["exit_code"] == (3 if session_status == "reused" else 0)
        assert calls == ["create-background-session"]
    launches = [arguments for arguments in processes if arguments[1] == "new-session"]
    assert bool(launches) == (session_status != "reused")
