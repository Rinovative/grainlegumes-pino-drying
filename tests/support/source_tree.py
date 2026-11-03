"""Freeze a fake-Git test checkout into a real read-only source tree.

Responsibilities:
  - Build a project folder with a live checkout, storage, runtime and an inert native interpreter
  - Materialize the checkout through the maintained source_fingerprint script under the fake Git
  - Describe the environment of a process that runs from the resulting tree, with Git failing or absent

Design principles:
  - The real snapshot, restore and verification code runs; only Git answers are controlled
  - A tree is read-only like the real one, and fixture cleanup unfreezes it first

This module does NOT:
  - Run Generation, Slurm or COMSOL, or initialize a real Git repository
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from support.fake_git import FakeGit, install_failing_git, install_fake_git

if TYPE_CHECKING:
    from collections.abc import Mapping

_SCRIPTS = ("source_fingerprint.py", "generation_node.sh", "generation_prerequisites.sh")


@dataclass
class FrozenProject:
    """A live fake-Git checkout beside storage and runtime, from which trees are materialized."""

    root: Path
    repository: Path
    storage: Path
    runtime: Path
    binary: Path
    fake_git: FakeGit
    receipts: list[dict[str, str]] = field(default_factory=list)

    @property
    def snapshots(self) -> Path:
        """Return the durable snapshot root."""
        return self.storage / "source_snapshots"

    @property
    def trees(self) -> Path:
        """Return the runtime tree root."""
        return self.runtime / "sources"

    @property
    def venv(self) -> Path:
        """Return the native environment directory."""
        return self.runtime / "venvs/native"

    @property
    def path_with_fake_git(self) -> str:
        """Return a PATH whose Git answers like the fake checkout."""
        return f"{self.binary}{os.pathsep}{os.environ['PATH']}"

    def materialize(self) -> dict[str, str]:
        """Freeze the checkout's current state with the real tool and return its receipt."""
        result = subprocess.run(  # noqa: S603 -- test-owned interpreter and controlled argument vector
            [
                sys.executable,
                str(self.repository / "scripts/source_fingerprint.py"),
                str(self.repository),
                "--materialize",
                str(self.snapshots),
                str(self.trees),
            ],
            env={**os.environ, "PATH": self.path_with_fake_git},
            capture_output=True,
            text=True,
            check=True,
        )
        receipt: dict[str, str] = json.loads(result.stdout)
        self.receipts.append(receipt)
        return receipt

    def tree_environment(self, receipt: Mapping[str, str], *, git_binary: Path | None = None) -> dict[str, str]:
        """Return the environment of a process running from the receipt's tree, with Git failing in ``git_binary``."""
        path = os.environ["PATH"] if git_binary is None else f"{git_binary}{os.pathsep}{os.environ['PATH']}"
        return {
            **{key: value for key, value in os.environ.items() if not key.startswith("BASH_FUNC_module")},
            "PATH": path,
            "SOURCE_TREE_ID": receipt["tree_id"],
            "PROJECT_ROOT": receipt["tree_path"],
            "STORAGE_ROOT": str(self.storage),
            "RUNTIME_ROOT": str(self.runtime),
            "GENERATION_NATIVE_VENV": str(self.venv),
            "GENERATION_GIT_COMMIT": receipt["git_commit"],
            "GENERATION_SOURCE_SHA256": receipt["source_fingerprint_sha256"],
            "GENERATION_SOURCE_SNAPSHOT": receipt["snapshot_path"],
            "GENERATION_SOURCE_SNAPSHOT_SHA256": receipt["snapshot_sha256"],
        }

    def failing_git(self) -> tuple[Path, Path]:
        """Install a Git stub that fails and logs; return its directory and call log."""
        directory = self.root / "failing-bin"
        return directory, install_failing_git(directory)

    def unfreeze(self) -> None:
        """Restore owner write access so the temporary folder can be removed."""
        for current, directories, _files in os.walk(self.trees):
            for name in directories:
                path = Path(current) / name
                if not path.is_symlink():
                    path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def build_project(root: Path, *, files: Mapping[str, str] | None = None) -> FrozenProject:
    """Create a project folder whose checkout holds the Generation scripts, some source and a fake Git."""
    scripts_source = Path(__file__).resolve().parents[2] / "scripts"
    repository = root / "repo"
    (repository / "scripts").mkdir(parents=True)
    for name in _SCRIPTS:
        shutil.copy2(scripts_source / name, repository / "scripts" / name)
    (repository / "src").mkdir()
    (repository / "src/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    for relative, content in (files or {}).items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    storage = root / "storage"
    runtime = root / "runtime"
    storage.mkdir()
    interpreter = runtime / "venvs/native/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    binary = root / "bin"
    fake_git = install_fake_git(repository, binary)
    return FrozenProject(root, repository, storage, runtime, binary, fake_git)


@pytest.fixture
def frozen_project(tmp_path: Path, request: pytest.FixtureRequest) -> FrozenProject:
    """Provide a fresh project and unfreeze every tree materialized from it at teardown."""
    project = build_project(tmp_path / "project")
    request.addfinalizer(project.unfreeze)  # noqa: PT021 -- fixture setup has no yield so cleanup also runs after a failed test body
    return project


def tampered(tree: Path, relative: str) -> Path:
    """Force-append one byte to a frozen tree file and return it; the tree no longer matches its snapshot."""
    target = tree / relative
    target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)  # tree files are read-only; a tamperer must unfreeze first
    with target.open("ab") as stream:
        stream.write(b"#")
    return target


def snapshot_manifest(receipt: Mapping[str, str]) -> dict[str, Any]:
    """Return a receipt's parsed snapshot manifest."""
    manifest: dict[str, Any] = json.loads(Path(receipt["snapshot_path"]).read_text(encoding="utf-8"))
    return manifest
