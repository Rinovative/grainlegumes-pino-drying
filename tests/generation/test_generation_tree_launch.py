"""Protect how Generation launchers choose, lease and enter the frozen source tree a run executes in.

Responsibilities:
  - Check that read-only admissions leave no snapshot or tree and cannot schedule work
  - Check that a launcher driver is submitted from the bound tree of an existing campaign, restoring a pruned one, or from the current tree on request
  - Check that a driver job runs from its tree with the tree environment, an appending log, and a job lease
  - Check that job logs are hard-linked only into an existing owner and never replace or fail
  - Check that node jobs and benchmark jobs are submitted only from a tree and lease it, even when the lease fails
  - Check that freezing is refused inside a Slurm allocation and that a bound campaign refuses other code without an explicit transition

Design principles:
  - The real snapshot, restore, lease and prune tool runs; Git and the scheduler are controlled

This module does NOT:
  - Run Slurm, COMSOL or a real Git repository
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from src.generation import generation_benchmark as benchmark_service
from src.generation import generation_campaign as campaign_service
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.generation_controller import GenerationController, RunPlan
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_host as host_service

if TYPE_CHECKING:
    from support.source_tree import FrozenProject

_SOURCE_VARIABLES = (
    "SOURCE_TREE_ID",
    "GENERATION_GIT_COMMIT",
    "GENERATION_SOURCE_SHA256",
    "GENERATION_SOURCE_SNAPSHOT",
    "GENERATION_SOURCE_SNAPSHOT_SHA256",
    "GENERATION_NATIVE_VENV",
    "GENERATION_SOURCE_TRANSITION",
    "PROJECT_ROOT",
    "RUNTIME_ROOT",
)


def _live_layout(project: FrozenProject, **extra: str) -> host_service.HostLayout:
    environment = {key: value for key, value in os.environ.items() if key not in _SOURCE_VARIABLES and not key.startswith("SLURM")}
    scheduler = project.root / "scheduler-bin"
    scheduler.mkdir(exist_ok=True)
    (scheduler / "squeue").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (scheduler / "squeue").chmod(0o755)
    (scheduler / "sbatch").write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > {project.root / "sbatch.argv"}\nenv > {project.root / "sbatch.env"}\necho 4711\n', encoding="utf-8"
    )
    (scheduler / "sbatch").chmod(0o755)
    path = f"{scheduler}{os.pathsep}{project.path_with_fake_git}"
    environment = {**environment, "PATH": path, "STORAGE_ROOT": str(project.storage), "RUNTIME_ROOT": str(project.runtime), **extra}
    return host_service.HostLayout.resolve(project.repository, environment=environment)


def test_read_only_admission_leaves_no_snapshot_or_tree_and_cannot_schedule(frozen_project: FrozenProject) -> None:
    """Inspection only fingerprints the checkout, keeps child work in it and refuses scheduled work."""
    layout = _live_layout(frozen_project)
    admission = host_service.admit_source(layout, freeze=False)

    assert not admission.frozen
    assert not layout.snapshots.exists()
    assert not layout.trees.exists()
    environment = host_service._child_environment(layout, admission)
    assert "SOURCE_TREE_ID" not in environment
    assert environment["GENERATION_SOURCE_SHA256"] == admission.fingerprint
    with pytest.raises(RuntimeError, match="only from a frozen source tree"):
        host_service.GenerationHost(layout, admission).call("resolve-generation-run", scheduled=True)


def test_submitting_a_driver_requires_a_live_launcher_and_a_frozen_admission(frozen_project: FrozenProject) -> None:
    """A read-only admission names no tree, and a process inside a tree cannot submit from or materialize another one."""
    layout = _live_layout(frozen_project)
    unfrozen = host_service.admit_source(layout, freeze=False)
    with pytest.raises(RuntimeError, match="Only a frozen admission"):
        host_service.submit_source_tree(layout, unfrozen, module="m", arguments=[], explicit_transition=False, job=_JOB)
    receipt = frozen_project.materialize()
    tree_layout = host_service.HostLayout.resolve(Path(receipt["tree_path"]), environment=frozen_project.tree_environment(receipt))
    with pytest.raises(RuntimeError, match="already runs from a frozen source tree"):
        host_service.submit_source_tree(tree_layout, unfrozen, module="m", arguments=[], explicit_transition=False, job=_JOB)
    with pytest.raises(RuntimeError, match="cannot materialize another tree"):
        host_service.admit_snapshot_tree(tree_layout, Path(receipt["snapshot_path"]))


def _enter_tree_mode(project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run this process as if launched from the project's current tree and return its receipt."""
    receipt = project.materialize()
    for key, value in project.tree_environment(receipt).items():
        monkeypatch.setenv(key, value)
    return receipt


def _fake_job_submission(monkeypatch: pytest.MonkeyPatch, *, output: str, lease_fails: bool = False) -> None:
    """Answer sbatch with ``output`` and optionally fail the lease command, running everything else for real."""
    real_run = subprocess.run

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "sbatch":
            return SimpleNamespace(stdout=output, returncode=0)
        if lease_fails and "--lease" in command:
            return SimpleNamespace(stdout="", stderr="FileNotFoundError: Cannot lease a missing source tree\n", returncode=1)
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def test_node_jobs_are_submitted_only_from_a_tree_and_leased_after_recording(frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live submission is refused; a tree submission returns the job ID, and the dispatcher's lease then protects the tree for it."""
    monkeypatch.delenv("SOURCE_TREE_ID", raising=False)
    with pytest.raises(RuntimeError, match="only from a frozen source tree"):
        cluster_service.submit_node_job(["sbatch", "--parsable", "node.sh"], git_commit="a" * 40, environment={})
    receipt = _enter_tree_mode(frozen_project, monkeypatch)
    _fake_job_submission(monkeypatch, output="31337\n")
    job_id = cluster_service.submit_node_job(["sbatch", "--parsable", "node.sh"], git_commit=receipt["git_commit"], environment={})

    lease = frozen_project.trees / ".leases" / receipt["tree_id"] / "31337"
    assert job_id == "31337"
    assert not lease.exists()  # the dispatcher records the job first and leases afterwards, with a visible result
    assert cluster_service.lease_executing_tree(job_id) is True
    assert lease.is_file()


def test_benchmark_jobs_are_submitted_only_from_a_tree_and_lease_it(frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> None:
    """A benchmark job is refused before submission outside a tree; inside one its returned job ID leases the tree."""
    with monkeypatch.context() as live:
        live.delenv("SOURCE_TREE_ID", raising=False)
        live.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("a benchmark job was submitted from live code"))
        with pytest.raises(RuntimeError, match="only from a frozen source tree"):
            benchmark_service._submit(
                ["sbatch", "--parsable", "benchmark.sh"], git_commit="a" * 40, run_id="core_scaling_transient__0123456789abcdef"
            )
    receipt = _enter_tree_mode(frozen_project, monkeypatch)
    _fake_job_submission(monkeypatch, output="31338;cluster\n")
    job_id = benchmark_service._submit(
        ["sbatch", "--parsable", "benchmark.sh"], git_commit=receipt["git_commit"], run_id="core_scaling_transient__0123456789abcdef"
    )

    assert job_id == "31338"
    assert (frozen_project.trees / ".leases" / receipt["tree_id"] / "31338").is_file()


def _submit_and_lease_node_job(command: list[str], *, git_commit: str, **_identity: str) -> str:
    """Submit a node job and lease its tree as the campaign dispatcher does; a failed lease is reported by its result."""
    job_id = cluster_service.submit_node_job(command, git_commit=git_commit, environment={})
    assert cluster_service.lease_executing_tree(job_id) is False
    return job_id


@pytest.mark.parametrize(
    ("submit", "output"),
    [(_submit_and_lease_node_job, "31337\n"), (benchmark_service._submit, "31337;cluster\n")],
    ids=["campaign", "benchmark"],
)
def test_a_failed_lease_after_submission_keeps_the_job_id_and_warns_once(
    frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], submit: Any, output: str
) -> None:
    """The job is already queued, so the lease failure only warns and the caller still receives and records the job ID."""
    receipt = _enter_tree_mode(frozen_project, monkeypatch)
    _fake_job_submission(monkeypatch, output=output, lease_fails=True)

    job_id = submit(["sbatch", "--parsable", "job.sh"], git_commit=receipt["git_commit"], run_id="core_scaling_transient__0123456789abcdef")

    assert job_id == "31337"
    warnings = capsys.readouterr().err.splitlines()
    assert len(warnings) == 1
    assert "31337" in warnings[0]
    assert receipt["tree_id"] in warnings[0]
    assert "stays protected by its 24-hour recency" in warnings[0]
    assert not (frozen_project.trees / ".leases").exists()


_JOB = host_service.LauncherJob("dataset-steady_flow", "datasets", "standard", "06:00:00")


def _sbatch_call(project: FrozenProject) -> tuple[list[str], dict[str, str]]:
    """Return the argument vector and environment the fake sbatch recorded."""
    argv = (project.root / "sbatch.argv").read_text(encoding="utf-8").splitlines()
    environment: dict[str, str] = {}
    for line in (project.root / "sbatch.env").read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator:
            environment[name] = value
    return argv, environment


@pytest.mark.parametrize("explicit_transition", [False, True])
def test_submitted_launcher_runs_from_the_tree_with_its_environment_and_holds_its_lease(
    frozen_project: FrozenProject, explicit_transition: bool
) -> None:
    """The batch job gets the tree as working directory, the launcher command with live paths mapped, the tree environment and a job lease."""
    layout = _live_layout(frozen_project)
    admission = host_service.admit_source(layout, freeze=True)
    tree = admission.tree_path
    assert tree is not None

    submitted = host_service.submit_source_tree(
        layout,
        admission,
        module="src.datasets.dataset_packages",
        arguments=["run", str(frozen_project.repository / "configs/plan.yaml"), "--count", "lentil=3"],
        explicit_transition=explicit_transition,
        job=_JOB,
    )

    logs = frozen_project.runtime / "logs/datasets"
    assert submitted == host_service.SubmittedLauncher("4711", str(admission.tree_id), logs / "dataset-steady_flow-4711.log")
    assert logs.is_dir()
    argv, environment = _sbatch_call(frozen_project)
    command = [str(layout.python), "-m", "src.datasets.dataset_packages", "run", str(tree / "configs/plan.yaml"), "--count", "lentil=3"]
    assert argv == [
        "--parsable",
        "--job-name=dataset-steady_flow",
        "--partition=standard",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=06:00:00",
        f"--output={logs}/dataset-steady_flow-%j.log",
        "--open-mode=append",
        f"--chdir={tree}",
        "--export=ALL",
        f"--wrap={shlex.join(command)}",
    ]
    assert {key: environment.get(key) for key in _SOURCE_VARIABLES} == {
        "SOURCE_TREE_ID": admission.tree_id,
        "GENERATION_GIT_COMMIT": admission.commit,
        "GENERATION_SOURCE_SHA256": admission.fingerprint,
        "GENERATION_SOURCE_SNAPSHOT": admission.snapshot_path,
        "GENERATION_SOURCE_SNAPSHOT_SHA256": admission.snapshot_sha256,
        "GENERATION_NATIVE_VENV": str(frozen_project.venv),
        "GENERATION_SOURCE_TRANSITION": source_service.EXPLICIT_SOURCE_TRANSITION if explicit_transition else None,
        "PROJECT_ROOT": str(tree),
        "RUNTIME_ROOT": str(frozen_project.runtime),
    }
    assert environment["STORAGE_ROOT"] == str(frozen_project.storage)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert (layout.trees / ".leases" / str(admission.tree_id) / "4711").is_file()


def test_submitted_launcher_keeps_its_job_id_when_the_lease_fails(
    frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lease failure after sbatch only warns; the submitted launcher is reported with its job ID and tree."""
    layout = _live_layout(frozen_project)
    admission = host_service.admit_source(layout, freeze=True)
    real = host_service._run_checked
    lease_error = host_service.GenerationCommandError("python3", 1, "FileNotFoundError: Cannot lease a missing source tree\n")

    def run_checked(command: tuple[str, ...], **kwargs: Any) -> str:
        if "--lease" in command:
            raise lease_error
        return real(command, **kwargs)

    monkeypatch.setattr(host_service, "_run_checked", run_checked)

    submitted = host_service.submit_source_tree(layout, admission, module="m", arguments=[], explicit_transition=False, job=_JOB)

    assert (submitted.job_id, submitted.tree_id) == ("4711", admission.tree_id)
    warnings = capsys.readouterr().err.splitlines()
    assert len(warnings) == 1
    assert "4711" in warnings[0]
    assert str(admission.tree_id) in warnings[0]
    assert "stays protected by its 24-hour recency" in warnings[0]


def test_job_logs_link_only_into_an_existing_owner_and_never_replace_another_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A link never creates its owner, a same-inode rerun is a no-op, and a different existing file stays with a warning."""
    log = tmp_path / "runtime/logs/datasets/dataset-steady_flow-4711.log"
    log.parent.mkdir(parents=True)
    log.write_text("[1/4] plan\n", encoding="utf-8")
    owner = tmp_path / "storage/02_datasets/steady_flow/dataset"
    host_service.link_job_logs(owner, [log])
    assert not owner.exists()
    assert "Warning:" in capsys.readouterr().err
    owner.mkdir(parents=True)
    host_service.link_job_logs(owner, [log])
    host_service.link_job_logs(owner, [log])
    assert (owner / "logs" / log.name).samefile(log)
    assert not capsys.readouterr().err
    stranger = tmp_path / "other" / log.name
    stranger.parent.mkdir()
    stranger.write_text("other\n", encoding="utf-8")
    host_service.link_job_logs(owner, [stranger])
    assert (owner / "logs" / log.name).samefile(log)
    assert "Warning:" in capsys.readouterr().err


def test_job_logs_come_from_the_submission_templates_and_a_file_swapped_while_linking_is_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Launcher and node jobs find their logs from their own sbatch templates; a log replaced between check and link is never kept."""
    runtime = tmp_path / "runtime"
    launcher_log = runtime / "logs/datasets/dataset-steady_flow-4711.log"
    node_logs = (runtime / "logs/generation/node-4711.out", runtime / "logs/generation/node-4711.err")
    for path in (launcher_log, *node_logs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("log\n", encoding="utf-8")
    monkeypatch.setenv("RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("SLURM_JOB_NAME", "dataset-steady_flow")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert host_service.current_job_log_files() == ()
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    assert host_service.current_job_log_files() == (launcher_log,)
    assert host_service.job_log_files(cluster_service.node_job_log_templates(runtime)) == node_logs
    owner = tmp_path / "campaign"
    owner.mkdir()
    link = Path.hardlink_to

    def swap_then_link(target: Path, source: Path) -> None:
        replacement = tmp_path / "replacement.log"
        replacement.write_text("other\n", encoding="utf-8")
        replacement.replace(source)
        link(target, source)

    monkeypatch.setattr(Path, "hardlink_to", swap_then_link)
    host_service.link_job_logs(owner, [launcher_log])
    assert not (owner / "logs" / launcher_log.name).exists()
    assert "changed while it was linked" in capsys.readouterr().err


@pytest.mark.parametrize(("continue_with_current_code", "bound_tree"), [(False, True), (True, False)])
def test_run_launcher_is_submitted_from_the_tree_the_controller_chooses(
    frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch, continue_with_current_code: bool, bound_tree: bool
) -> None:
    """The bound tree, restored after pruning, is chosen unless the current code is requested, which marks an explicit transition."""
    bound = frozen_project.materialize()
    frozen_project.unfreeze()
    host_service.subprocess.run(["rm", "-rf", bound["tree_path"]], check=True)  # pruned meanwhile
    (frozen_project.repository / "notes.txt").write_text("newer code\n", encoding="utf-8")
    layout = _live_layout(frozen_project)
    current = host_service.admit_source(layout, freeze=True)
    monkeypatch.setattr(campaign_service, "bound_source_snapshot", lambda _identity, **_kwargs: Path(bound["snapshot_path"]))
    plan = RunPlan("campaign", "source~steady_flow~x__0123456789abcdef", "plan.yaml", "purpose", "steady_flow", (), {})

    submitted = GenerationController(host_service.GenerationHost(layout, current)).submit_run_tree(
        plan, module="m", arguments=["run"], continue_with_current_code=continue_with_current_code, job=_JOB
    )

    expected = bound["tree_id"] if bound_tree else str(current.tree_id)
    assert submitted.tree_id == expected
    assert (frozen_project.trees / expected).is_dir()
    argv, environment = _sbatch_call(frozen_project)
    assert f"--chdir={frozen_project.trees / expected}" in argv
    assert environment["SOURCE_TREE_ID"] == expected
    assert (source_service.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE in environment) is continue_with_current_code
    assert (layout.trees / ".leases" / expected / "4711").is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("partition", "rsm", "partition must be one of"),
        ("cpus_per_task", 0, "cpus per task"),
        ("cpus_per_task", True, "cpus per task"),
        ("memory", "8 GB", "memory"),
        ("wall_time", "00:00:00", "positive"),
        ("wall_time", "6h", "positive"),
        ("name", "../escape", "simple name"),
        ("log_area", "a/b", "log area one of"),
    ],
)
def test_launcher_job_rejects_unsupported_scheduler_options(field: str, value: Any, message: str) -> None:
    """Only standard or long CPU jobs with a simple name, a known log area, positive CPUs, a memory size and a positive time limit are admitted."""
    longer = replace(_JOB, partition="long", wall_time="1-00:00:00")
    assert longer.partition == "long"
    with pytest.raises(ValueError, match=message):
        replace(_JOB, **{field: value})


def test_freezing_is_refused_inside_a_slurm_allocation_before_any_git_call(frozen_project: FrozenProject) -> None:
    """Compute nodes never capture or archive source: live freezing and snapshot restoration fail on the login-node message."""
    bound = frozen_project.materialize()
    failing_git, git_log = frozen_project.failing_git()
    in_job = _live_layout(frozen_project, SLURM_JOB_ID="4242", PATH=f"{failing_git}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(
        RuntimeError, match=re.escape("Freeze the source on the login node: start this command there; it submits its own Slurm work.")
    ):
        host_service.admit_source(in_job, freeze=True)
    with pytest.raises(RuntimeError, match="Freeze the source on the login node"):
        host_service.admit_snapshot_tree(in_job, Path(bound["snapshot_path"]))

    assert not git_log.exists()
    read_only = host_service.admit_source(_live_layout(frozen_project, SLURM_JOB_ID="4242"), freeze=False)
    assert not read_only.frozen


def test_bound_snapshot_follows_the_latest_epoch_or_the_origin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing campaign has no binding, the latest epoch wins over the origin, and a campaign without any snapshot fails closed."""
    manifest_path = tmp_path / "run.json"
    monkeypatch.setattr(campaign_service.campaign_evidence, "campaign_run_manifest_path", lambda *_args, **_kwargs: manifest_path)
    assert campaign_service.bound_source_snapshot("source~steady_flow~x") is None
    manifest_path.write_text("{}", encoding="utf-8")
    origin = {"source_snapshot": {"path": "/storage/source_snapshots/aaaaaaaaaaaa-bbbbbbbbbbbb/snapshot.json"}}
    monkeypatch.setattr(campaign_service.campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: origin)
    bindings = iter(
        [
            {"source_snapshot": None},
            {"source_snapshot": "/storage/source_snapshots/cccccccccccc-dddddddddddd/snapshot.json"},
            {"source_snapshot": None},
        ]
    )
    monkeypatch.setattr(campaign_service, "_execution_binding", lambda _manifest: next(bindings))
    assert campaign_service.bound_source_snapshot("source~steady_flow~x") == Path(origin["source_snapshot"]["path"])
    latest = campaign_service.bound_source_snapshot("source~steady_flow~x")
    assert latest is not None
    assert latest.parent.name == "cccccccccccc-dddddddddddd"
    origin.clear()
    with pytest.raises(RuntimeError, match="records no source snapshot"):
        campaign_service.bound_source_snapshot("source~steady_flow~x")


def test_bound_campaign_refuses_other_code_unless_the_transition_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source change on continuation needs the launcher's explicit transition marker; unchanged source needs none."""
    monkeypatch.delenv(source_service.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE, raising=False)
    campaign_service._require_explicit_source_transition("source~steady_flow~x", source_changed=False)
    with pytest.raises(RuntimeError, match="--continue-with-current-code"):
        campaign_service._require_explicit_source_transition("source~steady_flow~x", source_changed=True)
    monkeypatch.setenv(source_service.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE, source_service.EXPLICIT_SOURCE_TRANSITION)
    campaign_service._require_explicit_source_transition("source~steady_flow~x", source_changed=True)


def test_materialize_snapshot_receipt_matches_the_snapshot(frozen_project: FrozenProject) -> None:
    """Restoring an existing snapshot reports the snapshot's own commit and fingerprint, not the checkout's current state."""
    bound = frozen_project.materialize()
    (frozen_project.repository / "notes.txt").write_text("newer code\n", encoding="utf-8")
    result = subprocess.run(  # noqa: S603 -- test-owned interpreter and controlled argument vector
        [
            sys.executable,
            str(frozen_project.repository / "scripts/source_fingerprint.py"),
            str(frozen_project.repository),
            "--materialize-snapshot",
            bound["snapshot_path"],
            str(frozen_project.trees),
        ],
        env={**os.environ, "PATH": frozen_project.path_with_fake_git},
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt == bound
