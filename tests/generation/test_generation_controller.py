"""Exercise Generation controller decisions against isolated service evidence.

Responsibilities:
  - Check campaign continuations, source snapshots, and storage containment
  - Check that a public run freezes the source, including uncommitted changes, and submits its controller as a driver job from the login node
  - Check that the public minimum-success-rate amendment is a read-only admission delegating to the service operation
  - Check the printed stages, the live board cadence, and the one plain report of an expected failure
  - Check the failure context from plan resolution on, that a failing status suggests no dry run, and that logs follow the campaign
    as soon as it exists
  - Check that the runtime log of a scheduled operation summarizes a returned JSON object and keeps every other output raw

Design principles:
  - Service effects and ordering are asserted through isolated controller boundaries
  - Printing is shown to leave the service calls, sleeps and outcome of every driver path unchanged, even when the board fails

This module does NOT:
  - Simulate COMSOL numerics or replace persisted-data validation
"""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from src import common
from src.generation import generation_campaign_status as status_service
from src.generation import generation_controller as controller_module
from src.generation import generation_run as run_service
from src.generation.cases import generation_cases_config as config_service
from src.generation.cli import cli_generation_controller as public_cli
from src.generation.generation_controller import GenerationController, GenerationControllerError, LeafResult, RunPlan
from src.generation.runtime import generation_runtime_host as host_service
from src.generation.runtime.generation_runtime_host import GenerationCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence

_RUN = "steady_flow_test__0123456789abcdef"
_SOURCE = "steady_flow_source__fedcba9876543210"
_REPLACEMENT = "steady_flow_replacement__1234567890abcdef"
_COMPLETION = "completion__" + "a" * 24
_DIGEST = "b" * 64
_CONFIG = "configs/02_datasets/steady_flow/test.yaml"
_CANONICAL = "source~steady_flow~lentil-d1__reference-range__lhs-s9910"


def _admission(
    tree: Path | None = None, *, commit: str = "c" * 40, snapshot_path: str = "/test/snapshot.json", uncommitted: bool = False
) -> host_service.SourceAdmission:
    """Return a test-owned admission whose frozen tree is ``tree``."""
    return host_service.SourceAdmission(
        commit, _DIGEST, snapshot_path, _DIGEST, f"{commit[:12]}-{_DIGEST[:12]}", tree or Path("/test/tree"), uncommitted
    )


def _plan(kind: str = "campaign") -> RunPlan:
    payload: dict[str, Any] = {"run_kind": kind, "identity": _RUN, "config_path": _CONFIG, "children": []}
    return RunPlan(kind, _RUN, _CONFIG, "steady_flow_id_dataset", "steady_flow", (), payload)


def _canonical_plan() -> RunPlan:
    payload = {"run_kind": "campaign", "identity": _CANONICAL, "config_path": _CONFIG, "children": []}
    return RunPlan("campaign", _CANONICAL, _CONFIG, "steady_flow_id_dataset", "steady_flow", (), payload)


def _canonical_status(
    *,
    campaign_state: str,
    case_states: tuple[str, ...],
    terminal_decisions: tuple[str | None, ...] | None = None,
    replacement_deficit: int = 0,
    pending_jobs: int = 0,
    running_jobs: int = 0,
    submission_unknown: bool = False,
) -> str:
    """Represent the current schema-1 source status returned by campaign-status."""
    decisions = terminal_decisions or (None,) * len(case_states)
    return json.dumps(
        {
            "campaign_run_id": _CANONICAL,
            "campaign_state": campaign_state,
            "counts": {"successful": case_states.count("successful"), "replacement_deficit": replacement_deficit},
            "cases": [
                {
                    "material": "lentil",
                    "case_id": f"case_{index:04d}",
                    "state": "failed" if state in {"failed", "conversion_failed"} else state,
                    "classified_state": state,
                    "terminal_decision": decision,
                }
                for index, (state, decision) in enumerate(zip(case_states, decisions, strict=True), start=1)
            ],
            "pending_jobs": pending_jobs,
            "running_jobs": running_jobs,
            "submission_unknown": submission_unknown,
        }
    )


def test_canonical_campaign_prepares_submits_and_resumes_in_order(port: FakePort, monkeypatch: Any) -> None:
    """The public controller advances one stable source campaign through its lifecycle."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "active"})]
    port.responses["campaign-status"] = [_canonical_status(campaign_state="active", case_states=("running", "never_started"), running_jobs=1)]
    port.responses["resume-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "complete"})]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.run_id == _CANONICAL
    operations = _operations(port)
    assert operations.index("prepare-campaign-inputs") < operations.index("submit-campaign") < operations.index("resume-campaign")
    assert operations[-3:] == ["campaign-status", "resume-campaign", "campaign-status"]


@pytest.mark.parametrize("campaign_state", ["staged", "completed_with_failures"])
def test_canonical_campaign_returns_at_recorded_fresh_admission_cap(port: FakePort, capsys: Any, campaign_state: str) -> None:
    """The public run command exits once one staged candidate is terminal and the cap remains one."""
    port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": campaign_state})]
    port.responses["campaign-status"] = [
        _canonical_status(campaign_state=campaign_state, case_states=("successful", "never_started", "never_started", "never_started"))
    ]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "STAGED"
    printed = capsys.readouterr().out
    assert "GENERATION STAGED" in printed
    assert "[4/4] bounded source admission done in " in printed
    assert _operations(port)[-1] == "campaign-status"
    assert "resume-campaign" not in _operations(port)


_QUIESCENT = "completed_with_failures"


def _resumed(state: str = _QUIESCENT, **fields: Any) -> str:
    """Represent one resume-campaign response."""
    return json.dumps({"campaign_run_id": _CANONICAL, "state": state, **fields})


def _advance_clock_per_read(monkeypatch: Any, seconds: float = 61.0) -> None:
    """Let each settle-window clock read advance by ``seconds``, so the third read of a persisting failure is past the window."""
    ticks = iter(range(10_000))
    monkeypatch.setattr("src.generation.generation_controller.time.monotonic", lambda: next(ticks) * seconds)


@pytest.mark.parametrize("failed_state", ["failed", "conversion_failed"])
def test_canonical_campaign_fails_on_an_undecided_failure_that_persists_through_the_settle_window(
    port: FakePort, monkeypatch: Any, failed_state: str
) -> None:
    """A failed first source case of any stage is reconciled until the settle window passes, then reported with the fail-case route."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    _advance_clock_per_read(monkeypatch)
    undecided = _canonical_status(campaign_state=_QUIESCENT, case_states=(failed_state, "never_started", "never_started", "never_started"))
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [undecided, undecided, undecided]
    port.responses["resume-campaign"] = [_resumed(), _resumed()]
    with pytest.raises(
        GenerationControllerError, match=r"case_0001.*fail-case .* MATERIAL CASE_ID --reason-code CODE --detail TEXT.*rerun the same command"
    ):
        GenerationController(port).run(_canonical_plan(), _CONFIG, max_admission_cases=1)
    assert _operations(port)[-5:] == ["campaign-status", "resume-campaign", "campaign-status", "resume-campaign", "campaign-status"]


@pytest.mark.parametrize("requires_compute", [False, True])
def test_canonical_campaign_survives_a_failure_that_disappears_within_the_settle_window(
    port: FakePort, monkeypatch: Any, requires_compute: bool
) -> None:
    """A status read right after the jobs ended can predate their evidence; a failure read twice within the window does not stop the driver."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    port.responses["submit-campaign"] = [_resumed()]
    _advance_clock_per_read(monkeypatch, seconds=50.0)
    stale = _canonical_status(campaign_state=_QUIESCENT, case_states=("successful", "failed"))
    port.responses["campaign-status"] = [stale, stale, _canonical_status(campaign_state=_QUIESCENT, case_states=("successful", "successful"))]
    port.responses["resume-campaign"] = [
        _resumed(continuation_requires_compute=requires_compute),
        *([_resumed()] if requires_compute else []),
        _resumed(),
        _resumed("complete"),
    ]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "OK"
    assert _operations(port).count("campaign-status") == 4
    assert [scheduled for operation, _arguments, scheduled in port.calls if operation == "resume-campaign"] == (
        [False, True, False, False] if requires_compute else [False, False, False]
    )


def test_canonical_campaign_reconciles_again_when_a_failure_reappears_after_an_absent_one(port: FakePort, monkeypatch: Any) -> None:
    """The reconciliation guard resets whenever the failure is absent from a status."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    undecided = _canonical_status(campaign_state=_QUIESCENT, case_states=("successful", "failed"))
    settled = _canonical_status(campaign_state=_QUIESCENT, case_states=("successful", "successful"))
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [undecided, settled, undecided, settled]
    port.responses["resume-campaign"] = [_resumed(), _resumed(), _resumed(), _resumed("complete")]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert _operations(port).count("campaign-status") == 5


def test_canonical_campaign_continues_past_a_decided_failure_covered_by_a_replacement(port: FakePort, monkeypatch: Any) -> None:
    """A failure with a recorded decision never stops the driver; its replacement fills the slot and the campaign completes."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [
        _canonical_status(
            campaign_state=_QUIESCENT, case_states=("failed", "successful", "successful"), terminal_decisions=("failed", "successful", "successful")
        )
    ]
    port.responses["resume-campaign"] = [_resumed("complete")]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "OK"
    assert _operations(port)[-3:] == ["campaign-status", "resume-campaign", "campaign-status"]


def test_canonical_campaign_survives_a_deficit_that_the_reconciliation_plans_away(port: FakePort, monkeypatch: Any) -> None:
    """A recorded failure leaves a deficit only until the feeder plans its replacement, which then runs."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [
        _canonical_status(
            campaign_state=_QUIESCENT, case_states=("failed", "successful"), terminal_decisions=("failed", "successful"), replacement_deficit=1
        ),
        _canonical_status(
            campaign_state="active",
            case_states=("failed", "successful", "running"),
            terminal_decisions=("failed", "successful", None),
            running_jobs=1,
        ),
    ]
    port.responses["resume-campaign"] = [_resumed("active"), _resumed("complete")]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "OK"
    assert _operations(port).count("resume-campaign") == 2


def test_canonical_campaign_fails_on_an_unfillable_deficit_with_the_success_rate_stop(port: FakePort, monkeypatch: Any) -> None:
    """A decided failure whose slot no replacement can fill ends the driver once reconciliation plans nothing more, naming the stop."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    _advance_clock_per_read(monkeypatch)
    unfilled = _canonical_status(
        campaign_state=_QUIESCENT, case_states=("failed", "successful"), terminal_decisions=("failed", "successful"), replacement_deficit=1
    )
    unfilled = json.dumps({**json.loads(unfilled), "success_rate_stop": "Stopped: lentil (natural): k/n 1/2. Lower s with amend-replacements."})
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [unfilled, unfilled, unfilled]
    port.responses["resume-campaign"] = [_resumed(), _resumed()]
    with pytest.raises(
        GenerationControllerError, match=r"1 unfilled success slot\(s\) and no pending or running job\. Stopped: .*amend-replacements"
    ):
        GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert _operations(port)[-5:] == ["campaign-status", "resume-campaign", "campaign-status", "resume-campaign", "campaign-status"]


def test_canonical_campaign_rejects_a_status_without_decision_evidence(port: FakePort) -> None:
    """A status lacking the candidate decision fields is malformed, not an absent failure."""
    status = json.loads(_canonical_status(campaign_state=_QUIESCENT, case_states=("failed", "successful")))
    del status["cases"][0]["terminal_decision"]
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [json.dumps(status)]
    with pytest.raises(GenerationControllerError, match="lacks current candidate evidence"):
        GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert "resume-campaign" not in _operations(port)


def test_canonical_campaign_continues_unknown_scheduler_evidence(port: FakePort, monkeypatch: Any) -> None:
    """An unknown scheduler state remains with the service owner."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    state = "scheduler_unknown"
    port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": state})]
    port.responses["campaign-status"] = [
        _canonical_status(campaign_state=state, case_states=("scheduler_pending", "never_started"), submission_unknown=True)
    ]
    port.responses["resume-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "complete"})]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "OK"
    assert _operations(port)[-3:] == ["campaign-status", "resume-campaign", "campaign-status"]


def test_canonical_campaign_forwards_explicit_admission_limit_to_source_service(port: FakePort) -> None:
    """The public campaign path passes its operational cap to the persisted execution owner."""
    result = GenerationController(port).run(_canonical_plan(), _CONFIG, max_admission_cases=4)
    assert isinstance(result, LeafResult)
    submission = next(arguments for operation, arguments, _scheduled in port.calls if operation == "submit-campaign")
    assert submission[-2:] == ("--max-admission-cases", "4")
    requested_limit = 4
    assert public_cli._parser().parse_args(["run", _CONFIG, "--max-admission-cases", str(requested_limit)]).max_admission_cases == requested_limit


def test_canonical_status_and_cancel_accept_structured_handle(port: FakePort) -> None:
    """Status and owned cancellation admit the canonical v2 run handle."""
    port.responses["campaign-status"] = [json.dumps({"campaign_run_id": _CANONICAL, "campaign_state": "active"})]
    controller = GenerationController(port)
    assert _CANONICAL in controller.status(_CANONICAL)
    controller.cancel(_CANONICAL)
    assert _operations(port) == ["campaign-status", "cancel-campaign"]


def test_dirty_source_snapshot_retains_exact_diff_and_untracked_bytes(tmp_path: Path, monkeypatch: Any) -> None:
    """A bounded source snapshot records recoverable bytes and a stable fingerprint."""
    script = Path(__file__).resolve().parents[2] / "scripts/source_fingerprint.py"
    capture = runpy.run_path(str(script), run_name="generation_source_snapshot_test")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "added.py").write_bytes(b"new = 1\n")

    def fake_git(arguments: list[str], **_kwargs: Any) -> SimpleNamespace:
        if "diff" in arguments:
            return SimpleNamespace(stdout=b"diff --git a/code.py b/code.py\n+patch bytes\n")
        if "ls-files" in arguments:
            return SimpleNamespace(stdout=b"added.py\0")
        if "rev-parse" in arguments:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_git)
    result = capture["capture_source_snapshot"](repository, tmp_path / "snapshots")
    manifest = json.loads(Path(result["snapshot_path"]).read_text(encoding="utf-8"))
    assert manifest["git_commit"] == "a" * 40
    assert manifest["source_fingerprint_sha256"] == result["fingerprint"]
    assert (Path(result["snapshot_path"]).parent / "tracked.diff").read_bytes().endswith(b"patch bytes\n")
    assert (Path(result["snapshot_path"]).parent / "untracked/added.py").read_bytes() == b"new = 1\n"
    assert Path(result["snapshot_path"]).parent.name == f"{'a' * 12}-{result['fingerprint'][:12]}"


def test_source_snapshot_short_name_collision_with_other_bytes_fails_closed(tmp_path: Path, monkeypatch: Any) -> None:
    """A different snapshot under the same short name is rejected instead of reused."""
    script = Path(__file__).resolve().parents[2] / "scripts/source_fingerprint.py"
    capture = runpy.run_path(str(script), run_name="generation_source_snapshot_collision_test")
    repository = tmp_path / "repo"
    repository.mkdir()

    def fake_git(arguments: list[str], **_kwargs: Any) -> SimpleNamespace:
        if "diff" in arguments:
            return SimpleNamespace(stdout=b"diff --git a/code.py b/code.py\n+patch bytes\n")
        if "ls-files" in arguments:
            return SimpleNamespace(stdout=b"")
        if "rev-parse" in arguments:
            return SimpleNamespace(stdout="b" * 40 + "\n")
        raise AssertionError(arguments)

    monkeypatch.setattr(subprocess, "run", fake_git)
    fingerprint = capture["source_fingerprint"](repository)
    colliding = tmp_path / "snapshots" / f"{'b' * 12}-{fingerprint[:12]}"
    colliding.mkdir(parents=True)
    (colliding / "snapshot.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicts with current source bytes"):
        capture["capture_source_snapshot"](repository, tmp_path / "snapshots")


@pytest.mark.parametrize(
    ("options", "partition", "wall_time"),
    [
        ([], "standard", "06:00:00"),
        (["--driver-partition", "long", "--driver-time", "1-12:00:00"], "long", "1-12:00:00"),
    ],
)
def test_public_run_freezes_the_dirty_source_and_submits_its_controller_as_a_driver_job(
    monkeypatch: Any,
    generation_config_factory: Any,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    partition: str,
    wall_time: str,
) -> None:
    """A run on the login node admits uncommitted changes, resolves the plan, and submits the controller from the tree instead of running it."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying", natural_count=4)
    repository = config_path.parents[3]
    relative = config_path.relative_to(repository)
    campaign_name = config_service.load_campaign_config(config_path).source_name
    commit = "a" * 40
    admission = _admission(repository, commit=commit, snapshot_path="test-owned-snapshot", uncommitted=True)
    admissions: list[bool] = []
    submissions: list[dict[str, Any]] = []

    def admit(_layout: object, *, freeze: bool) -> host_service.SourceAdmission:
        admissions.append(freeze)
        return admission

    def resolve(_host: host_service.GenerationHost, operation: str, config: str, _commit_option: str, requested_commit: str, **_kwargs: Any) -> str:
        assert operation == "resolve-generation-run"
        assert requested_commit == commit
        return json.dumps(run_service.resolve_generation_run(config, source_commit=commit, repository_root=repository).to_payload())

    def submit(
        controller: GenerationController, plan: RunPlan, *, module: str, arguments: list[str], continue_with_current_code: bool, job: Any
    ) -> host_service.SubmittedLauncher:
        assert isinstance(controller.port, host_service.GenerationHost)
        submissions.append({"identity": plan.identity, "module": module, "arguments": arguments, "continue": continue_with_current_code, "job": job})
        return host_service.SubmittedLauncher("4711", "tree-id", Path("/logs/generation-4711.log"))

    monkeypatch.chdir(repository)
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    monkeypatch.setattr(public_cli.host_service.HostLayout, "resolve", lambda _repository: SimpleNamespace(repository=repository))
    monkeypatch.setattr(public_cli.host_service, "admit_source", admit)
    monkeypatch.setattr(host_service.GenerationHost, "call", resolve)
    monkeypatch.setattr(GenerationController, "submit_run_tree", submit)
    monkeypatch.setattr(GenerationController, "run", lambda *_args, **_kwargs: pytest.fail("the login node ran the controller"))

    assert public_cli.main(["run", relative.as_posix(), *options]) == 0
    assert admissions == [True]
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "SUBMITTED", "job_id": "4711", "tree_id": "tree-id", "log_path": "/logs/generation-4711.log"}
    assert f"Source: shared HEAD {commit} (snapshot carries uncommitted changes)" in captured.err
    (submission,) = submissions
    assert submission["identity"] == f"source~transient_drying~{campaign_name}"
    assert submission["module"] == "src.generation.cli.cli_generation_controller"
    assert submission["arguments"] == ["run", relative.as_posix(), *options]
    assert not submission["continue"]
    assert submission["job"] == host_service.LauncherJob("generation-transient_drying", "generation", partition, wall_time)


def test_public_run_validates_the_driver_options_before_freezing_the_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unsupported driver partition or time limit fails before the source is captured."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    monkeypatch.setattr(public_cli.host_service, "admit_source", lambda *_args, **_kwargs: pytest.fail("the source was admitted"))
    expected_exit_status = 2
    assert public_cli.main(["run", "campaign.yaml", "--driver-partition", "rsm"]) == expected_exit_status
    assert public_cli.main(["run", "campaign.yaml", "--driver-time", "00:00:00"]) == expected_exit_status


def test_public_run_inside_a_source_tree_runs_the_controller(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The driver job re-enters the same command from its tree, where the run executes instead of being submitted again."""
    runs: list[tuple[RunPlan, str, int | None]] = []

    class Controller:
        def __init__(self, _port: object) -> None:
            pass

        def plan(self, _config: str, *, allow_incomplete: bool, scientific_arguments: tuple[str, ...] = ()) -> RunPlan:
            assert not allow_incomplete
            assert scientific_arguments == ("--max-admission-cases", "2")
            return _canonical_plan()

        def run(self, plan: RunPlan, config: str, *, max_admission_cases: int | None = None) -> None:
            runs.append((plan, config, max_admission_cases))

        def submit_run_tree(self, *_args: Any, **_kwargs: Any) -> NoReturn:
            pytest.fail("a driver job submitted itself again")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_TREE_ID", "a" * 12 + "-" + "b" * 12)
    monkeypatch.setattr(public_cli.host_service.HostLayout, "resolve", lambda _repository: object())
    monkeypatch.setattr(public_cli.host_service, "admit_source", lambda *_args, **_kwargs: _admission(uncommitted=True))
    monkeypatch.setattr(public_cli.controller_service, "GenerationController", Controller)
    assert public_cli.main(["run", _CONFIG, "--max-admission-cases", "2", "--driver-time", "06:00:00"]) == 0
    assert [(config, limit) for _plan_, config, limit in runs] == [(_CONFIG, 2)]


def test_public_amend_replacements_delegates_without_freezing_the_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rate amendments admit the source read-only, like fail-case, and pass every field to the service operation."""
    calls: list[tuple[str, tuple[str, ...], bool | None]] = []
    admissions: list[bool] = []

    def admit(_layout: object, *, freeze: bool) -> host_service.SourceAdmission:
        admissions.append(freeze)
        return _admission(tmp_path)

    def call(_host: host_service.GenerationHost, operation: str, *arguments: str, scheduled: bool | None = None, **_kwargs: Any) -> str:
        calls.append((operation, arguments, scheduled))
        return json.dumps({"minimum_success_rate": 0.8}) + "\n"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        public_cli.host_service.HostLayout, "resolve", lambda _repository: SimpleNamespace(repository=tmp_path, storage=tmp_path / "storage")
    )
    monkeypatch.setattr(public_cli.host_service, "admit_source", admit)
    monkeypatch.setattr(host_service.GenerationHost, "call", call)

    def command(run_id: str) -> list[str]:
        return [
            "amend-replacements",
            run_id,
            "--material",
            "lentil",
            "--regime",
            "natural",
            "--minimum-success-rate",
            "0.8",
            "--reason",
            "after first failure",
        ]

    assert public_cli.main(command(_CANONICAL)) == 0
    assert json.loads(capsys.readouterr().out) == {"minimum_success_rate": 0.8}
    assert admissions == [False]
    assert calls == [("amend-replacement-pool", (*command(_CANONICAL)[1:], "--storage-root", str(tmp_path / "storage")), False)]
    assert public_cli.main(command(_RUN)) == 2
    assert len(calls) == 1


def test_host_admits_only_physical_task_storage_inside_sibling_root(tmp_path: Path) -> None:
    """An isolated storage namespace reaches children without aliases or overlap."""
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    task_storage = tmp_path / "storage" / "vp2-test-retained"
    task_storage.mkdir(parents=True)
    interpreter = tmp_path / "runtime" / "venvs" / "native" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    layout = host_service.HostLayout.resolve(repository, environment={"STORAGE_ROOT": str(task_storage)})
    assert layout.storage == task_storage
    assert host_service._child_environment(layout, _admission(commit="a" * 40))["STORAGE_ROOT"] == str(task_storage)
    alias = tmp_path / "storage" / "alias"
    alias.symlink_to(task_storage, target_is_directory=True)
    with pytest.raises(ValueError, match="physical absolute path"):
        host_service.HostLayout.resolve(repository, environment={"STORAGE_ROOT": str(alias)})
    with pytest.raises(ValueError, match="contained in sibling storage"):
        host_service.HostLayout.resolve(repository, environment={"STORAGE_ROOT": str(tmp_path / "runtime")})


def _resources() -> dict[str, Any]:
    return {
        "campaign_purpose": "steady_flow_id_dataset",
        "execution_resources": {
            "cluster": {"cores_per_case": 4, "cores_per_node": 32, "node_jobs": 2, "node_job_wall_time": "08:00:00"},
            "submission": {"max_admission_cases": 2, "poll_interval_seconds": 1},
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


@dataclass
class FakePort:
    """Return controlled service evidence and retain observable operation order."""

    storage: Path
    responses: dict[str, list[str | GenerationCommandError]] = field(default_factory=dict)
    repository: Path = Path("test-repository")
    commit: str = "c" * 40
    calls: list[tuple[str, tuple[str, ...], bool | None]] = field(default_factory=list)
    attached: list[Path] = field(default_factory=list)
    published: bool = False
    terminal: bool = False

    def attach_logs(self, campaign_directory: Path) -> None:
        """Record the campaign the driver's logs are linked into."""
        self.attached.append(campaign_directory)

    def choose_tree(self, *, bound_snapshot: Path | None, continue_with_current_code: bool) -> NoReturn:
        """Refuse tree selection: controller tests run as the process inside the tree."""
        message = f"unexpected tree choice ({bound_snapshot}, {continue_with_current_code})"
        raise AssertionError(message)

    def submit_tree(
        self,
        admission: host_service.SourceAdmission,
        *,
        module: str,
        arguments: Sequence[str],
        explicit_transition: bool,
        job: host_service.LauncherJob,
    ) -> NoReturn:
        """Refuse launcher submission: controller tests run as the process inside the tree."""
        message = f"unexpected launcher submission for {module} {list(arguments)} ({admission.tree_id}, {explicit_transition}, {job.name})"
        raise AssertionError(message)

    def config_file(self, value: str) -> tuple[Path, str]:
        """Return the configured fixture path without touching a repository."""
        relative = value.removeprefix(str(self.repository) + "/")
        return self.repository / relative, relative

    def comsol_version(self, module: str, executable: str) -> str:
        """Return fixed native runtime identity."""
        assert module == "Comsol/v6.4"
        assert executable == "comsol"
        return "COMSOL Multiphysics 6.4.0.293"

    def verify_execution(self) -> None:
        """Treat the isolated worker boundary as ready."""

    def call(
        self,
        operation: str,
        *arguments: str,
        scheduled: bool | None = None,
        partition: str = "standard",
        wall_time: str = "02:00:00",
    ) -> str:
        """Return one queued response or the default persisted service state."""
        del partition, wall_time
        self.calls.append((operation, arguments, scheduled))
        queued = self.responses.get(operation)
        if queued:
            value = queued.pop(0)
            if isinstance(value, GenerationCommandError):
                raise value
            return value
        if operation == "validate-config":
            return json.dumps(_resources())
        if operation == "prepare-campaign-inputs":
            # The real input preparation creates the campaign directory; the controller attaches its logs only once it exists.
            parts = common.paths.campaign_run_parts(_CANONICAL)
            assert parts is not None
            common.paths.resolve_generation_campaign_directory(*parts, storage_root=self.storage).mkdir(parents=True, exist_ok=True)
            return "canonical-inputs\t1\t0"
        if operation == "submit-campaign":
            return json.dumps({"campaign_run_id": _CANONICAL, "state": "complete"})
        if operation == "campaign-status":
            # Once the queued statuses are used up the campaign is complete and its node jobs have settled.
            return json.dumps({"campaign_run_id": _CANONICAL, "pending_jobs": 0, "running_jobs": 0, "open_node_jobs": 0})
        return ""


@pytest.fixture
def port(tmp_path: Path) -> FakePort:
    """Give each test an isolated durable storage root."""
    storage = tmp_path / "storage"
    storage.mkdir()
    return FakePort(storage)


def _operations(port: FakePort) -> list[str]:
    return [name for name, _, _ in port.calls]


def test_legacy_campaign_plan_is_rejected_before_submission(port: FakePort) -> None:
    """Only the current structured source handle reaches the controller."""
    with pytest.raises(GenerationControllerError, match="current source campaign"):
        GenerationController(port).run(_plan(), _CONFIG)
    assert "submit-campaign" not in _operations(port)


def test_preflight_checks_execution_readiness_without_starting_work(port: FakePort) -> None:
    """Readiness reads the resource contract, the node launcher and the COMSOL version, and submits nothing."""
    result = GenerationController(port).preflight(_canonical_plan())
    assert result == f"PREFLIGHT COMPLETE: plan={_CANONICAL} kind=campaign host=shared-filesystem COMSOL=COMSOL Multiphysics 6.4.0.293"
    assert _operations(port) == ["validate-config"]


@pytest.mark.parametrize(
    ("pending", "running", "needs_compute", "initial_compute"),
    [(1, 0, False, False), (0, 1, False, False), (0, 1, True, False), (0, 0, True, True)],
)
def test_executing_controller_schedules_only_explicit_compute_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending: int,
    running: int,
    needs_compute: bool,
    initial_compute: bool,
) -> None:
    """Exercise the real host command boundary rather than an execute-false port."""
    repository = tmp_path / "repository"
    config = repository / _CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("fixture: true\n")
    node = repository / "scripts/generation_node.sh"
    node.parent.mkdir()
    node.write_text("#!/bin/sh\n")
    node.chmod(0o755)
    storage = tmp_path / "storage"
    parts = common.paths.campaign_run_parts(_CANONICAL)
    assert parts is not None
    persisted = common.paths.resolve_generation_run_manifest_path(
        common.paths.resolve_generation_execution_directory(common.paths.resolve_generation_campaign_directory(*parts, storage_root=storage))
    )
    persisted.parent.mkdir(parents=True)
    persisted.write_text("{}")
    layout = host_service.HostLayout(repository, storage, tmp_path / "runtime", tmp_path / "venv", {})
    host = host_service.GenerationHost(layout, _admission(repository))
    scheduled: list[tuple[str, ...]] = []
    metadata_calls: list[str] = []
    count = 0
    monkeypatch.setattr(host_service.GenerationHost, "verify_execution", lambda _self: None)
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)

    def host_process(command: tuple[str, ...], **_kwargs: Any) -> SimpleNamespace:
        nonlocal count
        operation = command[3]
        if operation == "validate-config":
            value = json.dumps(_resources())
        elif operation == "submit-campaign":
            assert "--metadata-only" in command
            initial: dict[str, Any] = {"campaign_run_id": _CANONICAL, "state": "blocked" if initial_compute else "active"}
            if initial_compute:
                initial.update(continuation_requires_compute=True, continuation_reason="supplemental_inputs_missing")
            value = json.dumps(initial)
        elif operation == "campaign-status":
            assert not initial_compute or scheduled, "Initial compute request must precede state/status admission"
            complete = count >= 3 or bool(scheduled)
            # After completion the node jobs have settled, which the controller waits for before it returns.
            value = _canonical_status(
                campaign_state="complete" if complete else "active",
                case_states=("successful",) if complete else ("running",),
                pending_jobs=0 if complete else pending,
                running_jobs=0 if complete else running,
            )
        elif operation == "resume-campaign":
            assert "--metadata-only" in command
            metadata_calls.append(operation)
            count += 1
            result: dict[str, Any] = {"campaign_run_id": _CANONICAL, "state": "complete" if count == 3 else "active"}
            if needs_compute and count == 2:
                result["continuation_requires_compute"] = True
                result["continuation_reason"] = "replacement_input_generation"
            value = json.dumps(result)
        else:
            raise AssertionError(operation)
        return SimpleNamespace(returncode=0, stdout=value, stderr="")

    def scheduled_process(command: tuple[str, ...], *_args: Any) -> tuple[int, str, str]:
        assert command[0] == "srun"
        assert "--job-name=generation-python-resume-campaign" in command
        assert "--metadata-only" not in command
        scheduled.append(command)
        return 0, json.dumps({"campaign_run_id": _CANONICAL, "state": "complete"}), ""

    monkeypatch.setattr(host_service.subprocess, "run", host_process)
    monkeypatch.setattr(host_service, "_run_scheduled", scheduled_process)
    result = GenerationController(host, execute=True).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "OK"
    assert len(scheduled) == int(needs_compute)
    assert len(metadata_calls) == (0 if initial_compute else 2 if needs_compute else 3)


@pytest.mark.parametrize("field", ["success_rate_stop", "admission_cap_stop"])
def test_blocked_campaign_names_its_stop_and_its_restart_preserves_admission_and_scientific_flags(port: FakePort, field: str) -> None:
    """A blocked campaign reports its success-rate or cap stop from its status; following the continuation keeps the original constraints."""
    plan = _canonical_plan()
    plan.payload["scientific_overrides"] = {"materials": ["lentil"], "cases_per_material": 4, "duration_hours": 4, "revision": "producer-fix"}
    port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "blocked"})]
    port.responses["campaign-status"] = [json.dumps({"campaign_run_id": _CANONICAL, field: "Stopped: lentil (natural): k/n 1/4."})]
    with pytest.raises(GenerationControllerError, match=r"blocked\. Stopped: lentil \(natural\): k/n 1/4") as failure:
        GenerationController(port).run(plan, _CONFIG, max_admission_cases=2)
    continuation = failure.value.report()
    assert "--revision producer-fix" in continuation
    assert "--max-admission-cases 2" in continuation
    assert "--materials lentil" in continuation
    assert "--cases-per-material 4" in continuation
    assert "--duration-hours 4" in continuation


def test_normal_resource_flags_reach_inspection_submission_and_restart(port: FakePort) -> None:
    """One recorded request governs actual inspection, feeding and continuation."""
    plan = _canonical_plan()
    plan.payload["execution_overrides"] = {
        "node_jobs": 1,
        "cores_per_case": 8,
        "case_timeout_seconds": 1500,
        "no_retry": True,
        "max_admission_cases": 4,
    }
    port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "submission_unknown"})]
    with pytest.raises(GenerationControllerError) as failure:
        GenerationController(port).run(plan, _CONFIG, max_admission_cases=4)
    for operation in ("validate-config", "prepare-campaign-inputs", "submit-campaign"):
        arguments = next(args for op, args, _scheduled in port.calls if op == operation)
        assert "--no-retry" in arguments
        assert arguments[arguments.index("--cores-per-case") + 1] == "8"
        assert arguments[arguments.index("--case-timeout-seconds") + 1] == "1500"
        assert arguments[arguments.index("--node-jobs") + 1] == "1"
        assert arguments.count("--max-admission-cases") == 1
    continuation = failure.value.report()
    assert "--no-retry" in continuation
    assert "--cores-per-case 8" in continuation
    assert continuation.count("--max-admission-cases 4") == 1


def _route_public_plan_through_service_cli(monkeypatch: Any, config_path: Path) -> None:
    """Run the public command from the test repository and resolve its plan through the service parser and planner."""
    from src.generation.cli import cli_generation as service_cli  # noqa: PLC0415 -- exact public/service boundary under test

    repository = config_path.parents[3]
    monkeypatch.chdir(repository)
    monkeypatch.setattr(host_service.HostLayout, "resolve", lambda _repository: SimpleNamespace(repository=repository))
    monkeypatch.setattr(host_service, "admit_source", lambda *_args, **_kwargs: _admission(repository, commit="a" * 40))

    def resolve(_host: Any, operation: str, *arguments: str, **kwargs: Any) -> str:
        assert operation == "resolve-generation-run"
        assert kwargs["scheduled"] is False
        parsed = service_cli._build_parser().parse_args([operation, *arguments])
        plan = run_service.resolve_generation_run(
            parsed.config,
            source_commit=parsed.git_commit,
            repository_root=repository,
            require_executable=not parsed.allow_incomplete,
            scientific_overrides=service_cli._scientific_overrides(parsed),
            execution_overrides=service_cli._execution_overrides(parsed),
        )
        return json.dumps(plan.to_payload())

    monkeypatch.setattr(host_service.GenerationHost, "call", resolve)
    monkeypatch.setattr(GenerationController, "preflight", lambda _controller, plan: f"PREFLIGHT COMPLETE: plan={plan.identity}")


def test_public_normal_resource_dry_run_resolves_exact_budget(
    monkeypatch: Any,
    generation_config_factory: Any,
    capsys: Any,
) -> None:
    """The normal public command resolves scientific and resource flags together, then reports the readiness check on stderr."""
    path, _ = generation_config_factory(simulation_profile="transient_drying", natural_count=4, timeout=600, node_job_wall_time="02:00:00")
    _route_public_plan_through_service_cli(monkeypatch, path)
    status = public_cli.main(
        [
            "run",
            str(path),
            "--materials",
            "lentil",
            "--cases-per-material",
            "4",
            "--duration-hours",
            "4",
            "--node-jobs",
            "1",
            "--max-admission-cases",
            "4",
            "--cores-per-case",
            "8",
            "--case-timeout-seconds",
            "1500",
            "--no-retry",
            "--dry-run",
        ]
    )
    assert status == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert f"PREFLIGHT COMPLETE: plan={payload['identity']}" in captured.err
    assert payload["scientific_overrides"] == {"materials": ["lentil"], "cases_per_material": 4, "duration_hours": 4.0}
    assert payload["execution_overrides"] == {
        "node_jobs": 1,
        "max_admission_cases": 4,
        "cores_per_case": 8,
        "case_timeout_seconds": 1500.0,
        "no_retry": True,
    }
    assert payload["execution_values"]["cluster"]["node_job_wall_time"] == "02:00:00"
    assert payload["execution_values"]["cluster"]["node_jobs"] == 1
    assert payload["execution_values"]["runtime"]["license_wait"]["enabled"] is False
    # The first stdout line carries the Slurm footprint of this run: one 24-core node job with three 8-core case steps of 1G plus 1G overhead.
    assert len(captured.out.splitlines()) == 1
    assert payload["scheduler_footprint"] == {
        "node_jobs": 1,
        "cores_per_node_job": 24,
        "node_job_memory": "4G",
        "node_job_wall_time": "02:00:00",
        "driver_cpus": 2,
        "driver_time": "06:00:00",
    }


def test_scheduler_footprint_takes_the_largest_campaign_and_the_driver_time_as_given() -> None:
    """A run of several campaigns asks for the largest node job of any of them."""

    def values(node_jobs: int, cores: int, memory: str, wall_time: str) -> dict[str, Any]:
        return {"site": {"cores_per_node": cores}, "cluster": {"node_jobs": node_jobs, "node_job_memory": memory, "node_job_wall_time": wall_time}}

    plan = {"execution_values": values(1, 32, "88G", "08:00:00"), "children": [{"execution_values": values(2, 24, "1T", "02:00:00"), "children": []}]}
    footprint = controller_module.scheduler_footprint(
        controller_module.plan_execution_values(plan), driver_time="1-00:00:00", driver_partition="long"
    )

    assert footprint == {
        "node_jobs": 2,
        "cores_per_node_job": 32,
        "node_job_memory": "1T",
        "node_job_wall_time": "08:00:00",
        "driver_cpus": 2,
        "driver_time": "1-00:00:00",
    }
    assert controller_module.scheduler_footprint(controller_module.plan_execution_values({"children": []}), driver_time="06:00:00") is None
    with pytest.raises(ValueError, match="Launcher time"):
        controller_module.scheduler_footprint([], driver_time="soon")


def test_public_revision_reaches_the_resolved_campaign_and_its_continuation(
    monkeypatch: Any,
    generation_config_factory: Any,
    capsys: Any,
) -> None:
    """``run --revision`` is a scientific override: it names the resolved Source and its continuation, and changes no seed or digest."""
    path, _ = generation_config_factory(simulation_profile="transient_drying", natural_count=2)
    _route_public_plan_through_service_cli(monkeypatch, path)
    payloads = []
    for options in ([], ["--revision", "producer-fix"]):
        assert public_cli.main(["run", str(path), *options, "--dry-run"]) == 0
        payloads.append(json.loads(capsys.readouterr().out))
    plain, revised = payloads
    assert plain["scientific_overrides"] == {}
    assert revised["scientific_overrides"] == {"revision": "producer-fix"}
    assert revised["identity"] == f"{plain['identity']}__rev-producer-fix"
    assert revised["input_identity"] == plain["input_identity"]
    assert [unit["input_identity"] for unit in revised["units"]] == [unit["input_identity"] for unit in plain["units"]]


@pytest.mark.parametrize("label", ["Upper", "dot.v2", "-lead", "trail-", "a--b", "nested/label", ""])
def test_public_run_refuses_a_malformed_revision_before_any_source_is_frozen(label: str, monkeypatch: Any, capsys: Any, tmp_path: Path) -> None:
    """The label grammar is checked on the login node before admission, so a typo never freezes a snapshot."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(host_service, "admit_source", lambda *_args, **_kwargs: pytest.fail("source admitted for a malformed revision"))

    assert public_cli.main(["run", "campaign.yaml", f"--revision={label}", "--dry-run"]) == 2
    assert "Revision" in capsys.readouterr().err


@pytest.mark.parametrize("staged", [False, True])
def test_authored_campaign_uses_existing_controller_and_reports_admission_state(port: FakePort, staged: bool) -> None:
    """Carry the authored mapping through existing commands without a second YAML."""
    authored = {"schema_kind": "generation_source_campaign", "revision": "readable-source"}
    payload = {
        **_canonical_plan().payload,
        "authored_value": authored,
        "execution_overrides": {"node_jobs": 1},
        "scientific_overrides": {"duration_hours": 4.0},
    }
    port.responses["resolve-generation-run"] = [json.dumps(payload)]
    if staged:
        port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "staged"})]
        port.responses["campaign-status"] = [_canonical_status(campaign_state="staged", case_states=("successful", "never_started"))]
    result = GenerationController(port).run_authored_campaign(
        "configs/02_datasets/plan.yaml",
        authored,
        execution_overrides={"node_jobs": 1},
        scientific_overrides={"duration_hours": 4.0},
    )
    assert result.result == ("STAGED" if staged else "OK")
    run_parts = common.paths.campaign_run_parts(_CANONICAL)
    assert run_parts is not None
    assert result.source_name == run_parts[1]
    with pytest.raises(ValueError, match=f"Run {_RUN!r} is not a canonical campaign run"):
        _ = LeafResult(_RUN, result.kind, result.purpose).source_name
    assert _operations(port).count("resolve-generation-run") == 1
    for operation, arguments, _scheduled in port.calls:
        if operation in {"resolve-generation-run", "validate-config", "prepare-campaign-inputs", "submit-campaign"}:
            assert json.loads(arguments[arguments.index("--authored-value-json") + 1]) == authored
            assert arguments[arguments.index("--node-jobs") + 1] == "1"
            assert arguments[arguments.index("--duration-hours") + 1] == "4.0"


@pytest.mark.parametrize(
    ("environment", "allocated"),
    [
        ({"SLURM_JOB_ID": "865511", "SLURM_CPUS_PER_TASK": "2", "SLURMD_NODENAME": "hpc119"}, True),
        ({}, False),
        ({"SLURM_JOB_ID": "865511", "SLURM_CPUS_PER_TASK": "2", "SLURMD_NODENAME": "login"}, False),
        ({"SLURM_JOB_ID": "865511", "SLURM_CPUS_PER_TASK": "2"}, False),
        ({"SLURM_JOB_ID": "invalid", "SLURM_CPUS_PER_TASK": "2", "SLURMD_NODENAME": "hpc119"}, False),
        ({"SLURM_JOB_ID": "865511", "SLURM_CPUS_PER_TASK": "0", "SLURMD_NODENAME": "hpc119"}, False),
    ],
)
def test_scheduled_host_reuses_only_its_actual_allocated_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], allocated: bool
) -> None:
    """A two-CPU driver must not request a larger nested Slurm step."""
    repository = tmp_path / "repository"
    worker = repository / "scripts/generation_node.sh"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)
    layout = host_service.HostLayout(repository, tmp_path / "storage", tmp_path / "runtime", tmp_path / "venv", environment)
    host = host_service.GenerationHost(layout, _admission(repository))
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(host_service.socket, "gethostname", lambda: "hpc119.ice.example")

    def scheduled(
        command: tuple[str, ...], working_directory: Path, child_environment: Any, output_path: str, _operation: str
    ) -> tuple[int, str, str]:
        assert Path(output_path).is_relative_to(layout.logs)
        assert working_directory == repository
        assert child_environment["GENERATION_SOURCE_SHA256"] == _DIGEST
        assert child_environment["GENERATION_NATIVE_VENV"] == str(layout.native_venv)
        assert child_environment["PROJECT_ROOT"] == str(repository)
        if allocated:
            assert command == (str(worker), str(repository), "cli", "prepare-campaign-inputs", "plan.yaml")
            assert child_environment["SLURM_CPUS_PER_TASK"] == "2"
        else:
            assert command[0] == "srun"
            assert "--cpus-per-task=4" in command
            assert "--mem=16G" in command
            assert command[-5:] == (str(worker), str(repository), "cli", "prepare-campaign-inputs", "plan.yaml")
        seen.append(command)
        return 0, "prepared", ""

    monkeypatch.setattr(host_service, "_run_scheduled", scheduled)
    assert host.call("prepare-campaign-inputs", "plan.yaml", scheduled=True) == "prepared"
    assert len(seen) == 1


def test_allocated_host_preserves_worker_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct reuse still runs the guarded node worker and exposes its failure."""
    repository = tmp_path / "repository"
    worker = repository / "scripts/generation_node.sh"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "#!/bin/sh\n"
        'test "$2" = cli || exit 99\n'
        'test "$3" = prepare-campaign-inputs || exit 98\n'
        "printf 'partial output\\n'\n"
        "printf 'source guard rejected\\n' >&2\n"
        "exit 23\n"
    )
    worker.chmod(0o755)
    environment = {"SLURM_JOB_ID": "865511", "SLURM_CPUS_PER_TASK": "2", "SLURMD_NODENAME": "hpc119"}
    layout = host_service.HostLayout(repository, tmp_path / "storage", tmp_path / "runtime", tmp_path / "venv", environment)
    host = host_service.GenerationHost(layout, _admission(repository))
    monkeypatch.setattr(host_service.socket, "gethostname", lambda: "hpc119")

    with pytest.raises(host_service.GenerationCommandError, match="source guard rejected") as failure:
        host.call("prepare-campaign-inputs", scheduled=True)
    assert failure.value.returncode == 23
    logs = list(layout.logs.glob("python-prepare-campaign-inputs.*.out"))
    assert len(logs) == 1
    assert logs[0].read_text() == "partial output\n"


@pytest.mark.parametrize(
    ("output", "status", "state"),
    [
        ('{"campaign_run_id": "run", "state": "active", "submissions": [1, 2, 3]}\n', 0, " state=active"),
        ('{"campaign_run_id": "run", "submissions": []}\n', 0, ""),
        ("canonical-inputs\t6\t0\n", 0, None),
        ("[1, 2]\n", 0, None),
        ('{"state": "failed"}\n', 23, None),
    ],
    ids=["json-with-state", "json-without-state", "text", "json-array", "failed-run"],
)
def test_scheduled_log_summarizes_a_returned_json_object_and_keeps_all_other_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str, status: int, state: str | None
) -> None:
    """A successful JSON object is one summary line in the runtime log while the driver gets it byte for byte; anything else stays raw."""
    repository = tmp_path / "repository"
    worker = repository / "scripts/generation_node.sh"
    worker.parent.mkdir(parents=True)
    (tmp_path / "payload").write_text(output)
    worker.write_text(f"#!/bin/sh\ncat {tmp_path / 'payload'}\nexit {status}\n")
    worker.chmod(0o755)
    environment = {"SLURM_JOB_ID": "865511", "SLURM_CPUS_PER_TASK": "2", "SLURMD_NODENAME": "hpc119"}
    layout = host_service.HostLayout(repository, tmp_path / "storage", tmp_path / "runtime", tmp_path / "venv", environment)
    host = host_service.GenerationHost(layout, _admission(repository))
    monkeypatch.setattr(host_service.socket, "gethostname", lambda: "hpc119")

    if status:
        with pytest.raises(GenerationCommandError) as failure:
            host.call("resume-campaign", scheduled=True)
        assert failure.value.returncode == status
    else:
        assert host.call("resume-campaign", scheduled=True) == output
    (log,) = layout.logs.glob("python-resume-campaign.*.out")
    if state is None:
        assert log.read_text() == output
    else:
        digest = hashlib.sha256(output.encode()).hexdigest()[:12]
        assert log.read_text() == f"resume-campaign: JSON result returned to the driver ({len(output)} bytes, sha256 {digest}){state}\n"


def test_completed_campaign_waits_until_its_node_jobs_have_ended(port: FakePort, monkeypatch: Any) -> None:
    """A complete campaign whose node job still runs is reconciled again before the driver reports success."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    port.responses["submit-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "complete"})]
    port.responses["campaign-status"] = [json.dumps({"campaign_run_id": _CANONICAL, "pending_jobs": 0, "running_jobs": 0, "open_node_jobs": 1})]
    port.responses["resume-campaign"] = [json.dumps({"campaign_run_id": _CANONICAL, "state": "complete"})]
    result = GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert isinstance(result, LeafResult)
    assert result.result == "OK"
    assert _operations(port)[-3:] == ["campaign-status", "resume-campaign", "campaign-status"]


def _live_case(case_id: str, state: str = "running", *, simulated: float | None = None, **extra: Any) -> dict[str, Any]:
    """Represent one case of the canonical status with the solver evidence the board shows."""
    runtime: dict[str, Any] = {"availability": "unavailable", "reason": "no_job"}
    if simulated is not None:
        runtime = {
            "availability": "available",
            "reason": None,
            "phase": "transient_drying",
            "parser_state": "available",
            "simulated_time_seconds": simulated,
            "step_size_seconds": 30.0,
            "time_failures": 1,
            "nonlinear_failures": 0,
            "elapsed_seconds": 600.0,
            "age_seconds": 4.0,
            "stale": False,
        }
    return {
        "material": "lentil",
        "batch_name": "transient_drying__lentil__natural",
        "case_id": case_id,
        "state": state,
        "classified_state": state,
        "terminal_decision": None,
        "runtime_progress": runtime,
        "physical_start_hours": 0.0,
        "physical_horizon_hours": 1.0,
        **extra,
    }


def _live_status(*cases: dict[str, Any]) -> str:
    """Represent a running campaign's status; its campaign state never makes the failure rule look at it."""
    return json.dumps(
        {
            "campaign_run_id": _CANONICAL,
            "campaign_state": "active",
            "counts": {"successful": sum(case["state"] == "successful" for case in cases), "replacement_deficit": 0},
            "cases": list(cases),
            "pending_jobs": 0,
            "running_jobs": 1,
            "open_node_jobs": 1,
            "submission_unknown": False,
        }
    )


def _poll(port: FakePort, monkeypatch: Any, statuses: list[str], *, seconds_per_poll: float) -> None:
    """Run the canonical loop over ``statuses`` on a clock that only the sleeps between polls advance."""
    now = [0.0]
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _seconds: now.__setitem__(0, now[0] + seconds_per_poll))
    port.responses["submit-campaign"] = [_resumed("active")]
    port.responses["campaign-status"] = [*statuses]
    port.responses["resume-campaign"] = [*[_resumed("active")] * (len(statuses) - 1), _resumed("complete")]
    controller = GenerationController(port)
    controller.clock = lambda: now[0]
    controller.run(_canonical_plan(), _CONFIG)


def _boards(printed: str) -> list[list[str]]:
    """Split the driver output into the boards: the header line and the lines that follow it up to the next blank line."""
    return [block.splitlines() for block in printed.split("\n\n") if block.startswith("Board ")]


def test_each_stage_prints_its_line_and_its_duration(port: FakePort, capsys: pytest.CaptureFixture[str]) -> None:
    """A stage announces itself as [n/N] when it starts and repeats its name with the duration when it ends."""
    GenerationController(port).run(_canonical_plan(), _CONFIG)
    stages = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    names = ("common plan resolution", "canonical input readiness", "campaign submission and reconciliation", "selected source publication")
    expected = [
        pattern
        for number, name in enumerate(names, start=1)
        for pattern in (
            re.escape(f"[{number}/4] ")
            + (re.escape(f"Run plan: kind=campaign identity={_CANONICAL} config={_CONFIG}") if number == 1 else re.escape(name)),
            re.escape(f"[{number}/4] {name} done in ") + r"\d+ s",
        )
    ]
    assert len(stages) == len(expected)
    assert all(re.fullmatch(pattern, line) for pattern, line in zip(expected, stages, strict=True)), stages


def test_board_appears_on_a_state_change_and_after_the_interval_but_not_at_every_poll(
    port: FakePort, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Progress alone does not reprint the board; a changed case state does, and so does the repeat interval of 300 s."""

    def two(simulated: float) -> str:
        return _live_status(_live_case("case_0001", simulated=simulated), _live_case("case_0002", "never_started"))

    finished = _live_status(_live_case("case_0001", "successful", elapsed="00:10:00"), _live_case("case_0002", simulated=50.0))
    _poll(port, monkeypatch, [two(100.0), two(200.0), two(300.0), finished], seconds_per_poll=150.0)
    first, second, third = _boards(capsys.readouterr().out)
    assert first[0].startswith("Board  done 0  running 1  waiting 1  failed 0  of 2  elapsed 0 s  ETA: not yet estimable")
    assert any("3 %" in line and "0.02778 h / 1 h" in line for line in first)
    assert any("case_0002  waiting  not started yet" in line for line in first)
    assert second[0].startswith("Board  done 0  running 1  waiting 1  failed 0  of 2  elapsed 5 min")
    assert any("0.08333 h / 1 h" in line for line in second)
    assert third[0].startswith("Board  done 1  running 1  waiting 0  failed 0  of 2  elapsed 7 min")
    assert any("case_0001  done" in line and "took 10 min" in line for line in third)


def test_a_default_horizon_campaign_shows_progress_from_the_status_horizon(
    port: FakePort, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a --duration-hours override, each case's persisted time window gives its progress past the start."""
    case = _live_case("case_0001", simulated=25 * 3600.0, physical_start_hours=24.0, physical_horizon_hours=4.0)
    _poll(port, monkeypatch, [_live_status(case)], seconds_per_poll=1.0)
    (board,) = _boards(capsys.readouterr().out)
    assert any("case_0001" in line and "25 %" in line and "1 h / 4 h" in line for line in board)


@pytest.mark.parametrize("failing", ["prepare-campaign-inputs", "submit-campaign"])
def test_logs_are_attached_once_the_campaign_exists_also_when_its_submission_fails(port: FakePort, failing: str) -> None:
    """A failure after the campaign exists still attaches the logs; a failure before it neither attaches nor creates the campaign."""
    port.responses[failing] = [GenerationCommandError(failing, 3, "boom")]
    with pytest.raises(GenerationCommandError):
        GenerationController(port).run(_canonical_plan(), _CONFIG)
    parts = common.paths.campaign_run_parts(_CANONICAL)
    assert parts is not None
    campaign = common.paths.resolve_generation_campaign_directory(*parts, storage_root=port.storage)
    assert port.attached == ([] if failing == "prepare-campaign-inputs" else [campaign])
    assert campaign.is_dir() is (failing == "submit-campaign")


def test_a_running_case_without_new_solver_evidence_is_marked_after_ten_minutes(
    port: FakePort, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mark counts from when the board last saw the evidence change, not from the receipt's own heartbeat."""
    still = _live_status(_live_case("case_0001", simulated=100.0))
    _poll(port, monkeypatch, [still, still, still], seconds_per_poll=400.0)
    first, second, third = _boards(capsys.readouterr().out)
    assert not any("no progress" in line for line in (*first, *second))
    assert any(line.endswith("no progress for 13 min") for line in third)


def _undecided_failure_run(port: FakePort, monkeypatch: Any) -> tuple[list[str], GenerationControllerError]:
    """Run the settle-window scenario of an undecided failure on the stepping clock, as the failure rule is tested above."""
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", lambda _: None)
    _advance_clock_per_read(monkeypatch)
    undecided = _canonical_status(campaign_state=_QUIESCENT, case_states=("failed", "never_started", "never_started", "never_started"))
    port.responses["submit-campaign"] = [_resumed()]
    port.responses["campaign-status"] = [undecided, undecided, undecided]
    port.responses["resume-campaign"] = [_resumed(), _resumed()]
    with pytest.raises(GenerationControllerError) as failure:
        GenerationController(port).run(_canonical_plan(), _CONFIG, max_admission_cases=1)
    return _operations(port), failure.value


def _settling(open_node_jobs: int) -> str:
    """Represent a status with no pending or running dispatch and ``open_node_jobs`` node jobs still open."""
    return json.dumps({"campaign_run_id": _CANONICAL, "pending_jobs": 0, "running_jobs": 0, "open_node_jobs": open_node_jobs})


_Scenario = tuple[dict[str, list[str | GenerationCommandError]], float, bool, str]


def _cancelling(mode: str) -> _Scenario:
    """Return the evidence of a persisted cancellation whose node job ends after one poll."""
    evidence: dict[str, list[str | GenerationCommandError]] = {
        "submit-campaign": [_resumed(mode)],
        "campaign-status": [_settling(1)],
        "resume-campaign": [_resumed(mode)],
    }
    return evidence, 61.0, False, f"failed: .*explicit continuation from state '{mode}'"


# Each driver scenario: its queued service evidence, the seconds each settle-clock read advances, whether it shows the
# board, and its expected outcome.
_DRIVER_SCENARIOS: dict[str, _Scenario] = {
    "completion_with_final_settlement": (
        {
            "submit-campaign": [_resumed("active")],
            "campaign-status": [_live_status(_live_case("case_0001", simulated=100.0)), _settling(1)],
            "resume-campaign": [_resumed("complete"), _resumed("complete")],
        },
        61.0,
        True,
        "result: OK",
    ),
    "settlement_timeout": (
        {"submit-campaign": [_resumed("complete")], "campaign-status": [_settling(1)]},
        1800.0,
        False,
        "failed: .*did not settle within 1800 s",
    ),
    "staged": (
        {
            "submit-campaign": [_resumed("staged")],
            "campaign-status": [_canonical_status(campaign_state="staged", case_states=("successful", "never_started"))],
        },
        61.0,
        True,
        "result: STAGED",
    ),
    "cancel_requested": _cancelling("cancel_requested"),
    "force_cancel_requested": _cancelling("force_cancel_requested"),
    "persistent_failure": (
        {
            "submit-campaign": [_resumed()],
            "campaign-status": [_canonical_status(campaign_state=_QUIESCENT, case_states=("failed", "never_started"))] * 3,
            "resume-campaign": [_resumed(), _resumed()],
        },
        61.0,
        True,
        "failed: .*without a recorded decision: lentil case_0001.*fail-case",
    ),
}


def _driver_run(storage: Path, monkeypatch: Any, scenario: str) -> tuple[list[tuple[str, tuple[str, ...], bool | None]], list[float], str]:
    """Run one driver scenario on a stepping settle clock and return its complete call records, its sleeps and its outcome."""
    responses, seconds_per_read, _shows_board, _outcome = _DRIVER_SCENARIOS[scenario]
    port = FakePort(storage, responses={operation: list(values) for operation, values in responses.items()})
    sleeps: list[float] = []
    monkeypatch.setattr("src.generation.generation_controller.time.sleep", sleeps.append)
    _advance_clock_per_read(monkeypatch, seconds=seconds_per_read)
    try:
        result = GenerationController(port).run(_canonical_plan(), _CONFIG, max_admission_cases=1)
    except GenerationControllerError as error:
        outcome = f"failed: {error}"
    else:
        assert isinstance(result, LeafResult)
        outcome = f"result: {result.result}"
    return port.calls, sleeps, outcome


@pytest.mark.parametrize("scenario", list(_DRIVER_SCENARIOS))
def test_the_board_changes_neither_the_service_calls_nor_the_sleeps_nor_the_outcome_even_when_it_fails(
    port: FakePort, monkeypatch: Any, capsys: pytest.CaptureFixture[str], scenario: str
) -> None:
    """Each driver path records the same calls, sleeps and outcome with the board off, on, and raising; a raising board reports once."""

    def broken(*_args: Any, **_kwargs: Any) -> NoReturn:
        message = "render broke"
        raise ValueError(message)

    with monkeypatch.context() as patch:
        patch.setattr(status_service.CampaignBoard, "observe", lambda *_args, **_kwargs: None)
        reference = _driver_run(port.storage, patch, scenario)
    _evidence, _seconds, shows_board, outcome = _DRIVER_SCENARIOS[scenario]
    assert re.match(outcome, reference[2]), reference[2]
    capsys.readouterr()
    with monkeypatch.context() as patch:
        assert _driver_run(port.storage, patch, scenario) == reference
    assert "status board unavailable" not in capsys.readouterr().err
    with monkeypatch.context() as patch:
        patch.setattr(status_service.CampaignBoard, "observe", broken)
        assert _driver_run(port.storage, patch, scenario) == reference
    assert capsys.readouterr().err.splitlines() == ["status board unavailable: render broke"] * shows_board


def test_an_expected_failure_carries_its_stage_and_continuation_for_the_one_plain_report(
    port: FakePort, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error, not the controller, reports the failure: FAILED with the stage and message, then the command that continues."""
    _operations_seen, error = _undecided_failure_run(port, monkeypatch)
    assert error.stage == "campaign submission and reconciliation"
    assert (
        error.report() == f"FAILED: campaign submission and reconciliation: {error}\nNext: ./scripts/generation run {_CONFIG} --max-admission-cases 1"
    )
    assert "FAILED" not in capsys.readouterr().err
    assert GenerationControllerError("plain").report() == "plain"


@pytest.mark.parametrize("step", ["plan resolution", "preflight"])
def test_a_failure_before_a_run_carries_its_stage_and_the_dry_run_that_checks_the_config_again(
    port: FakePort, capsys: pytest.CaptureFixture[str], step: str
) -> None:
    """Plan resolution and preflight name the dry run that checks the config again."""
    controller = GenerationController(port)
    if step == "plan resolution":
        port.responses["resolve-generation-run"] = [GenerationCommandError("resolve-generation-run", 2, "invalid config")]
        with pytest.raises(GenerationCommandError):
            controller.plan(_CONFIG)
    else:
        port.responses["validate-config"] = [GenerationCommandError("validate-config", 2, "invalid resources")]
        with pytest.raises(GenerationCommandError):
            controller.preflight(_canonical_plan())
    assert capsys.readouterr().err.startswith(f"FAILED: {step}\nNext: ./scripts/generation run {_CONFIG} --dry-run")


def test_a_failing_status_suggests_no_dry_run(port: FakePort, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Status only reads, so a failed plan resolution there names no dry run of a run as the next command."""
    (tmp_path / _CONFIG).parent.mkdir(parents=True)
    (tmp_path / _CONFIG).touch()
    port.repository = tmp_path
    port.responses["resolve-generation-run"] = [GenerationCommandError("resolve-generation-run", 2, "invalid config")]
    with pytest.raises(GenerationCommandError):
        GenerationController(port).status(_CONFIG)
    assert "Next:" not in capsys.readouterr().err


def test_an_unexpected_failure_is_still_reported_by_the_controller_before_its_traceback(port: FakePort, capsys: pytest.CaptureFixture[str]) -> None:
    """Only expected controller errors move to the CLI report; anything else keeps the stage report and then raises."""
    port.responses["submit-campaign"] = [GenerationCommandError("submit-campaign", 3, "boom")]
    with pytest.raises(GenerationCommandError):
        GenerationController(port).run(_canonical_plan(), _CONFIG)
    assert capsys.readouterr().err.startswith(
        f"FAILED: campaign submission and reconciliation\ncampaign_run_id: {_CANONICAL}\nNext: ./scripts/generation run"
    )


def test_public_run_reports_an_expected_failure_as_one_plain_report_with_its_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints the stage report instead of the bare message and returns the error's own exit status."""
    failure = GenerationControllerError("campaign failed", exit_code=3)
    failure.stage, failure.next_command = "campaign submission and reconciliation", "./scripts/generation run campaign.yaml"

    class Controller:
        def __init__(self, _port: object) -> None:
            pass

        def plan(self, _config: str, *, allow_incomplete: bool, scientific_arguments: tuple[str, ...] = ()) -> RunPlan:
            del allow_incomplete, scientific_arguments
            return _canonical_plan()

        def run(self, *_args: Any, **_kwargs: Any) -> NoReturn:
            raise failure

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOURCE_TREE_ID", "a" * 12 + "-" + "b" * 12)
    monkeypatch.setattr(public_cli.host_service.HostLayout, "resolve", lambda _repository: object())
    monkeypatch.setattr(public_cli.host_service, "admit_source", lambda *_args, **_kwargs: _admission(uncommitted=True))
    monkeypatch.setattr(public_cli.controller_service, "GenerationController", Controller)
    assert public_cli.main(["run", _CONFIG]) == failure.exit_code
    assert capsys.readouterr().err.endswith(
        "FAILED: campaign submission and reconciliation: campaign failed\nNext: ./scripts/generation run campaign.yaml\n"
    )


def test_the_driver_job_is_submitted_with_an_unbuffered_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver's log is read while it runs, so its Python must not hold output in a block buffer."""
    layout = host_service.HostLayout(tmp_path / "repo", tmp_path / "storage", tmp_path / "runtime", tmp_path / "venv", {})
    monkeypatch.setattr(host_service, "_run_checked", lambda *_args, **_kwargs: "")
    plan = host_service._launch_plan(layout, _admission(tmp_path / "tree"), module="src.module", arguments=["run"], explicit_transition=False)
    assert plan.environment["PYTHONUNBUFFERED"] == "1"
