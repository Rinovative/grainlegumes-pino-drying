"""Protect Generation source admission and publication checks that run from a Git-less source tree.

Responsibilities:
  - Check that live admission freezes a read-only tree, clean or with uncommitted changes, and tree-mode admission verifies it without Git
  - Check that host commands, configurations and child environments name the tree instead of the live checkout
  - Check that helper logs are hard-linked into their campaign once it is attached
  - Check that the publication guard, clean-commit and Dataset producer evidence accept an untouched tree and refuse any byte change

Design principles:
  - The real snapshot, restore and verification tool runs; only Git answers are controlled
  - Tree-mode processes get a failing Git stub whose call log must stay empty

This module does NOT:
  - Run Slurm, COMSOL or a real Git repository
  - Choose which tree an existing campaign continues in
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from support.fake_git import FAKE_GIT_HEAD
from support.source_tree import FrozenProject, build_project, snapshot_manifest, tampered

from src import common
from src.datasets.packages import dataset_packages_reproduction as reproduction
from src.generation import generation_campaign as campaign_service
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.runtime import generation_runtime_cluster as cluster_service
from src.generation.runtime import generation_runtime_host as host_service

if TYPE_CHECKING:
    from collections.abc import Mapping

_SOURCE_VARIABLES = (
    "SOURCE_TREE_ID",
    "GENERATION_GIT_COMMIT",
    "GENERATION_SOURCE_SHA256",
    "GENERATION_SOURCE_SNAPSHOT",
    "GENERATION_SOURCE_SNAPSHOT_SHA256",
    "GENERATION_NATIVE_VENV",
    "RUNTIME_ROOT",
)


def _live_environment(project: FrozenProject) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key not in _SOURCE_VARIABLES and not key.startswith("SLURM")}
    return {**environment, "PATH": project.path_with_fake_git, "STORAGE_ROOT": str(project.storage), "RUNTIME_ROOT": str(project.runtime)}


def _live_layout(project: FrozenProject) -> host_service.HostLayout:
    return host_service.HostLayout.resolve(project.repository, environment=_live_environment(project))


def _tree_layout(project: FrozenProject, receipt: Mapping[str, str]) -> tuple[host_service.HostLayout, Path]:
    failing_git, git_log = project.failing_git()
    environment = project.tree_environment(receipt, git_binary=failing_git)
    return host_service.HostLayout.resolve(Path(receipt["tree_path"]), environment=environment), git_log


def _use_tree(monkeypatch: pytest.MonkeyPatch, project: FrozenProject, receipt: Mapping[str, str]) -> Path:
    """Run this process as if launched from the receipt's tree with Git failing; return the Git call log."""
    failing_git, git_log = project.failing_git()
    for key, value in project.tree_environment(receipt, git_binary=failing_git).items():
        monkeypatch.setenv(key, value)
    return git_log


def test_live_admission_freezes_a_read_only_tree_even_for_a_clean_checkout(frozen_project: FrozenProject) -> None:
    """A clean checkout is captured and materialized exactly like a dirty one, and a repeat admission reuses the tree."""
    layout = _live_layout(frozen_project)
    admission = host_service.admit_source(layout, freeze=True)
    assert admission.tree_id is not None
    assert admission.snapshot_path is not None
    assert not admission.uncommitted

    tree = frozen_project.trees / admission.tree_id
    assert admission.tree_path == tree
    assert admission.commit == FAKE_GIT_HEAD
    assert admission.tree_id == f"{admission.commit[:12]}-{admission.fingerprint[:12]}"
    assert Path(admission.snapshot_path) == frozen_project.snapshots / admission.tree_id / "snapshot.json"
    assert admission.snapshot_sha256 == common.serialization.file_sha256(admission.snapshot_path)
    assert snapshot_manifest({"snapshot_path": admission.snapshot_path})["untracked"] == []
    assert (tree / "src/module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (tree / ".git").exists()
    assert not stat.S_IMODE(tree.stat().st_mode) & 0o222
    assert host_service.admit_source(layout, freeze=True) == admission

    environment = host_service._child_environment(layout, admission)
    assert environment["PROJECT_ROOT"] == str(tree)
    assert environment["SOURCE_TREE_ID"] == admission.tree_id
    assert environment["STORAGE_ROOT"] == str(frozen_project.storage)
    assert environment["RUNTIME_ROOT"] == str(frozen_project.runtime)
    assert environment["GENERATION_SOURCE_SNAPSHOT"] == admission.snapshot_path
    assert environment["GENERATION_SOURCE_SNAPSHOT_SHA256"] == admission.snapshot_sha256
    assert environment["GENERATION_GIT_COMMIT"] == admission.commit
    assert environment["GENERATION_SOURCE_SHA256"] == admission.fingerprint
    assert environment["GENERATION_NATIVE_VENV"] == str(frozen_project.venv)


def test_live_admission_freezes_a_dirty_checkout_and_reports_its_changes(frozen_project: FrozenProject) -> None:
    """Uncommitted changes are always admitted: the snapshot and tree carry the untracked bytes, a read-only admission only reports them."""
    layout = _live_layout(frozen_project)
    (frozen_project.repository / "notes.txt").write_text("local note\n", encoding="utf-8")

    read_only = host_service.admit_source(layout, freeze=False)
    assert read_only.uncommitted
    assert not read_only.frozen
    assert not frozen_project.snapshots.exists()

    admission = host_service.admit_source(layout, freeze=True)
    assert admission.uncommitted
    assert admission.tree_path is not None
    assert admission.snapshot_path is not None
    assert (admission.tree_path / "notes.txt").read_text(encoding="utf-8") == "local note\n"
    assert [entry["path"] for entry in snapshot_manifest({"snapshot_path": admission.snapshot_path})["untracked"]] == ["notes.txt"]


def test_tree_admission_verifies_the_tree_without_calling_git(frozen_project: FrozenProject) -> None:
    """A process running from a tree takes commit and fingerprint from the snapshot and refuses drift."""
    receipt = frozen_project.materialize()
    layout, git_log = _tree_layout(frozen_project, receipt)

    admission = host_service.admit_source(layout, freeze=True)
    assert admission.commit == receipt["git_commit"]
    assert admission.fingerprint == receipt["source_fingerprint_sha256"]
    assert admission.tree_path == Path(receipt["tree_path"])
    assert admission.snapshot_sha256 == receipt["snapshot_sha256"]
    assert not admission.uncommitted

    tampered(Path(receipt["tree_path"]), "src/module.py")
    with pytest.raises(host_service.GenerationCommandError, match="does not match"):
        host_service.admit_source(layout, freeze=True)
    assert not git_log.exists()


def test_tree_admission_reports_the_uncommitted_changes_its_snapshot_carries(frozen_project: FrozenProject) -> None:
    """A tree frozen from a dirty checkout runs like any other and reports, without Git, that its snapshot holds changes."""
    (frozen_project.repository / "notes.txt").write_text("local note\n", encoding="utf-8")
    receipt = frozen_project.materialize()
    layout, git_log = _tree_layout(frozen_project, receipt)

    admission = host_service.admit_source(layout, freeze=True)
    assert admission.tree_id == receipt["tree_id"]
    assert admission.uncommitted
    assert not git_log.exists()


@pytest.mark.parametrize(
    ("variable", "value", "error", "expected"),
    [
        ("STORAGE_ROOT", None, RuntimeError, "storage root must be set explicitly through STORAGE_ROOT"),
        ("RUNTIME_ROOT", None, RuntimeError, "runtime root must be set explicitly through RUNTIME_ROOT"),
        ("PROJECT_ROOT", "/elsewhere", RuntimeError, "must be RUNTIME_ROOT/sources/"),
        ("GENERATION_NATIVE_VENV", None, RuntimeError, "Native Python 3.12 environment is missing"),
        ("STORAGE_ROOT", "relative", ValueError, "physical absolute path"),
    ],
)
def test_tree_layout_needs_explicit_roots(
    frozen_project: FrozenProject, variable: str, value: str | None, error: type[Exception], expected: str
) -> None:
    """A tree has no sibling defaults: every root and the native venv come from the environment."""
    receipt = frozen_project.materialize()
    environment = frozen_project.tree_environment(receipt)
    if value is None:
        del environment[variable]
    else:
        environment[variable] = value
    with pytest.raises(error, match=expected):
        host_service.HostLayout.resolve(Path(receipt["tree_path"]), environment=environment)


def test_host_commands_run_from_the_tree_not_the_live_checkout(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker command, node launcher, srun chdir, working directory and configuration all name the admitted tree."""
    project = build_project(tmp_path / "project", files={"configs/example.yaml": "fixture: true\n"})
    request.addfinalizer(project.unfreeze)
    layout = _live_layout(project)
    admission = host_service.admit_source(layout, freeze=True)
    host = host_service.GenerationHost(layout, admission)
    launched: list[tuple[tuple[str, ...], Path, Mapping[str, str]]] = []

    def scheduled(
        command: tuple[str, ...], working_directory: Path, environment: Mapping[str, str], _output: str, _operation: str
    ) -> tuple[int, str, str]:
        launched.append((command, working_directory, environment))
        return 0, "scheduled", ""

    def run(command: tuple[str, ...], **kwargs: Any) -> Any:
        launched.append((command, kwargs["cwd"], kwargs["env"]))
        return host_service.subprocess.CompletedProcess(command, 0, "direct", "")

    monkeypatch.setattr(host_service, "_run_scheduled", scheduled)
    monkeypatch.setattr(host_service.subprocess, "run", run)
    assert host.call("prepare-campaign-inputs", "plan.yaml", scheduled=True) == "scheduled"
    assert host.call("campaign-status", "run", scheduled=False) == "direct"

    tree = admission.tree_path
    assert tree is not None
    srun, working_directory, environment = launched[0]
    assert srun[0] == "srun"
    assert f"--chdir={tree}" in srun
    assert srun[-5:] == (str(tree / "scripts/generation_node.sh"), str(tree), "cli", "prepare-campaign-inputs", "plan.yaml")
    assert working_directory == tree
    assert environment["PROJECT_ROOT"] == str(tree)
    direct, working_directory, environment = launched[1]
    assert direct[:3] == (str(layout.python), "-m", "src.generation.cli.cli_generation")
    assert working_directory == tree
    assert environment["SOURCE_TREE_ID"] == admission.tree_id
    assert str(project.repository) not in "".join(srun)

    # Helper logs written before the campaign exists follow it once attached; later ones are linked as they finish.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    campaign = tmp_path / "campaign"
    (campaign / "_record").mkdir(parents=True)
    host.attach_logs(campaign)
    host.call("prepare-campaign-inputs", "plan.yaml", scheduled=True)
    linked = sorted((campaign / "_record/logs").iterdir())
    assert [path.name.startswith("python-prepare-campaign-inputs.") and path.suffix == ".out" for path in linked] == [True, True]
    assert all(path.samefile(layout.logs / path.name) for path in linked)

    path, relative = host.config_file("configs/example.yaml")
    assert (path, relative) == (tree / "configs/example.yaml", "configs/example.yaml")
    assert host.config_file(project.repository / "configs/example.yaml")[0] == path
    with pytest.raises(FileNotFoundError, match="admitted source tree"):
        host.config_file("configs/missing.yaml")


def test_publication_guard_accepts_an_untouched_tree_and_refuses_every_drift(frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard verifies tree bytes against the admitted snapshot, never through Git, and demands every admission variable."""
    receipt = frozen_project.materialize()
    git_log = _use_tree(monkeypatch, frozen_project, receipt)

    source_service.validate_admitted_source_before_publication()
    admitted = source_service.verify_admitted_source_tree()
    assert (admitted.git_commit, admitted.source_fingerprint_sha256, admitted.clean) == (
        receipt["git_commit"],
        receipt["source_fingerprint_sha256"],
        True,
    )
    assert admitted.tree_path == Path(receipt["tree_path"])

    for variable, value, expected in (
        ("GENERATION_GIT_COMMIT", "c" * 40, "disagrees with the admitted commit"),
        ("GENERATION_SOURCE_SHA256", "d" * 64, "disagrees with the admitted commit or fingerprint"),
        ("GENERATION_SOURCE_SHA256", None, "fingerprint is missing or malformed"),
        ("GENERATION_GIT_COMMIT", None, "commit is missing or malformed"),
        ("GENERATION_SOURCE_SNAPSHOT", None, "snapshot evidence is missing or malformed"),
        ("GENERATION_SOURCE_SNAPSHOT_SHA256", "e" * 64, "missing, misplaced, or changed"),
        ("GENERATION_SOURCE_SNAPSHOT", str(frozen_project.snapshots / "other/snapshot.json"), "missing, misplaced, or changed"),
        ("STORAGE_ROOT", None, "STORAGE_ROOT must be set explicitly"),
    ):
        with monkeypatch.context() as scope:
            if value is None:
                scope.delenv(variable)
            else:
                scope.setenv(variable, value)
            with pytest.raises(source_service.SourceAdmissionError, match=expected):
                source_service.validate_admitted_source_before_publication()

    tampered(Path(receipt["tree_path"]), "src/module.py")
    with pytest.raises(source_service.SourceAdmissionError, match="failed verification against its snapshot: ValueError"):
        source_service.validate_admitted_source_before_publication()
    assert not git_log.exists()


def test_clean_commit_and_campaign_commit_come_from_the_verified_tree(frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean tree yields its snapshot commit for direct and campaign provenance; a dirty tree refuses only clean provenance."""
    receipt = frozen_project.materialize()
    git_log = _use_tree(monkeypatch, frozen_project, receipt)

    assert source_service.clean_repository_git_commit() == receipt["git_commit"]
    assert campaign_service._repository_commit() == receipt["git_commit"]
    with pytest.raises(RuntimeError, match="admits only its executing tree"):
        source_service.clean_repository_git_commit(frozen_project.repository)

    (frozen_project.repository / "notes.txt").write_text("local note\n", encoding="utf-8")
    dirty = frozen_project.materialize()
    _use_tree(monkeypatch, frozen_project, dirty)
    assert campaign_service._repository_commit() == dirty["git_commit"]
    with pytest.raises(RuntimeError, match="requires a clean repository worktree"):
        source_service.clean_repository_git_commit()
    assert not git_log.exists()


def test_dataset_producer_evidence_reuses_the_admitted_snapshot_inside_a_tree(frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside a tree the Dataset binds the admitted snapshot instead of capturing again, and verifies the read-only tree against it."""
    receipt = frozen_project.materialize()
    _use_tree(monkeypatch, frozen_project, receipt)
    before = sorted(path.name for path in frozen_project.snapshots.iterdir())

    evidence = reproduction.capture_producer_evidence()
    assert evidence["source_snapshot"] == {
        "relative_path": f"source_snapshots/{receipt['tree_id']}/snapshot.json",
        "sha256": receipt["snapshot_sha256"],
        "git_commit": receipt["git_commit"],
        "source_fingerprint_sha256": receipt["source_fingerprint_sha256"],
    }
    assert sorted(path.name for path in frozen_project.snapshots.iterdir()) == before
    reproduction.verify_current_producer(evidence)

    tampered(Path(receipt["tree_path"]), "src/module.py")
    with pytest.raises(source_service.SourceAdmissionError, match="failed verification"):
        reproduction.capture_producer_evidence()
    with pytest.raises(ValueError, match="does not match the snapshot"):
        reproduction.verify_current_producer(evidence)


def test_case_workers_receive_the_tree_bindings_and_never_a_partial_admission(frozen_project: FrozenProject, monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitted case jobs name tree, roots and snapshot explicitly; a tree without snapshot evidence cannot submit."""
    receipt = frozen_project.materialize()
    _use_tree(monkeypatch, frozen_project, receipt)

    assert cluster_service.source_worker_environment() == [
        f"PROJECT_ROOT={receipt['tree_path']}",
        f"SOURCE_TREE_ID={receipt['tree_id']}",
        f"RUNTIME_ROOT={frozen_project.runtime}",
        f"GENERATION_SOURCE_SNAPSHOT={receipt['snapshot_path']}",
        f"GENERATION_SOURCE_SNAPSHOT_SHA256={receipt['snapshot_sha256']}",
    ]
    monkeypatch.delenv("GENERATION_SOURCE_SNAPSHOT")
    monkeypatch.delenv("GENERATION_SOURCE_SNAPSHOT_SHA256")
    with pytest.raises(ValueError, match="Source-tree workers require"):
        cluster_service.source_worker_environment()
    monkeypatch.delenv("SOURCE_TREE_ID")
    assert cluster_service.source_worker_environment() == []
