"""Validate exact source-repository provenance for generation execution.

Responsibilities:
  - Validate one full lowercase Git object identifier
  - Require the launch-provided source commit from the process environment
  - Resolve a clean repository HEAD for direct interactive generation
  - Recheck admitted source fingerprints and optional snapshot inventories before publication
  - Verify a frozen source tree against its admitted snapshot without Git and report its commit, fingerprint, and cleanliness
  - Declare the Generation input and Source code projections and bind the producer identities to them

Design principles:
  - Source commit is execution provenance, not scientific case identity
  - Missing or abbreviated commit evidence fails closed before case generation
  - Source-tree mode never calls Git and never accepts a missing admission variable
  - Producer identity hashes only the declared projections; the projection tests classify every import of a member

This module does NOT:
  - Mutate, fetch, or check out a Git repository
  - Materialize, lease, or select source trees
  - Add source revisions to human-readable batch or dataset names
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy

from src import common

GIT_COMMIT_ENVIRONMENT_VARIABLE = "GENERATION_GIT_COMMIT"
SOURCE_TRANSITION_ENVIRONMENT_VARIABLE = "GENERATION_SOURCE_TRANSITION"
EXPLICIT_SOURCE_TRANSITION = "explicit"
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SOURCE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

GENERATION_INPUT_PROJECTION = common.code_identity.CodeProjection(
    name="generation_input",
    files=(
        "common/common_code_identity.py",
        "common/common_serialization.py",
        "domain/domain_moisture.py",
        "domain/domain_permeability.py",
        "generation/cases/generation_cases_case.py",
        "generation/cases/generation_cases_config.py",
        "generation/cases/generation_cases_fields.py",
        "generation/cases/generation_cases_sampling.py",
        "generation/cases/generation_cases_schedule.py",
        "generation/cases/generation_cases_seeding.py",
        "generation/contracts/generation_contracts_materials.py",
        "generation/contracts/generation_contracts_porosity.py",
        "generation/contracts/generation_contracts_profiles.py",
        "generation/contracts/generation_contracts_registry.py",
        "generation/contracts/generation_contracts_scalar_handoff.py",
        "generation/contracts/generation_contracts_source.py",
        "generation/contracts/generation_contracts_vocabulary.py",
    ),
    exemptions={
        "generation/contracts/generation_contracts_campaign_identity.py": "campaign naming and digests; its output is the identity",
        "generation/contracts/generation_contracts_templates.py": "template bytes are bound by their own recorded hash",
    },
)
GENERATION_SOURCE_PROJECTION = common.code_identity.CodeProjection(
    name="generation_source",
    files=(
        "generation/cases/generation_cases_admission.py",
        "generation/cases/generation_cases_input.py",
        "generation/cases/generation_cases_selection.py",
        "generation/contracts/generation_contracts_comsol_spreadsheet.py",
        "generation/contracts/generation_contracts_material_registry.py",
        "generation/contracts/generation_contracts_templates.py",
        "generation/generation_campaign_lifecycle.py",
        "generation/publication/generation_publication_storage.py",
        "generation/runtime/generation_runtime_batch.py",
        "generation/runtime/generation_runtime_comsol.py",
        "generation/runtime/generation_runtime_preparation.py",
        "generation/validation/generation_validation_policy.py",
    ),
    includes=(GENERATION_INPUT_PROJECTION,),
)


class SourceAdmissionError(RuntimeError):
    """Report changed or unavailable shared source before durable publication."""


@dataclass(frozen=True, slots=True)
class AdmittedSourceTree:
    """A frozen source tree verified byte-for-byte against its admitted snapshot."""

    tree_id: str
    tree_path: Path
    snapshot_path: Path
    snapshot_sha256: str
    git_commit: str
    source_fingerprint_sha256: str
    clean: bool


_EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()


def _failure_line(error: subprocess.CalledProcessError) -> str:
    """Return the last diagnostic line of a failed verification command."""
    lines = [line for line in (error.stderr or "").strip().splitlines() if line]
    return lines[-1] if lines else f"exit {error.returncode}"


def verify_admitted_source_tree() -> AdmittedSourceTree:
    """
    Verify the executing source tree against its admitted snapshot without Git.

    Returns
    -------
    AdmittedSourceTree
        Tree identity and the snapshot's commit and fingerprint, both equal to the
        launch-provided ``GENERATION_GIT_COMMIT`` and ``GENERATION_SOURCE_SHA256``.

    Raises
    ------
    SourceAdmissionError
        Outside source-tree mode, if any admission variable is missing or malformed,
        or if the snapshot or the tree bytes differ from the admitted evidence.

    """
    tree_id = common.paths.source_tree_id()
    if tree_id is None:
        message = "Source-tree verification requires SOURCE_TREE_ID."
        raise SourceAdmissionError(message)
    expected = os.environ.get("GENERATION_SOURCE_SHA256")
    if expected is None or _SOURCE_SHA256_PATTERN.fullmatch(expected) is None:
        message = "Generation source admission fingerprint is missing or malformed."
        raise SourceAdmissionError(message)
    try:
        expected_commit = required_git_commit()
    except (RuntimeError, ValueError) as error:
        message = "Generation source admission commit is missing or malformed."
        raise SourceAdmissionError(message) from error
    snapshot_value = os.environ.get("GENERATION_SOURCE_SNAPSHOT")
    snapshot_sha = os.environ.get("GENERATION_SOURCE_SNAPSHOT_SHA256")
    if snapshot_value is None or snapshot_sha is None or _SOURCE_SHA256_PATTERN.fullmatch(snapshot_sha) is None:
        message = "Generation source snapshot evidence is missing or malformed; a source tree needs its admitted snapshot."
        raise SourceAdmissionError(message)
    try:
        tree = common.paths.get_project_root()
        snapshot_root = common.paths.get_source_snapshots_root(storage_root=common.paths.get_storage_root())
    except (RuntimeError, ValueError) as error:
        raise SourceAdmissionError(str(error)) from error
    snapshot = Path(snapshot_value)
    if (
        snapshot != snapshot_root / tree_id / "snapshot.json"
        or snapshot.is_symlink()
        or not snapshot.is_file()
        or common.serialization.file_sha256(snapshot) != snapshot_sha
    ):
        message = f"Generation source snapshot of tree {tree_id} is missing, misplaced, or changed."
        raise SourceAdmissionError(message)
    try:
        result = subprocess.run(  # noqa: S603 -- fixed tree verification invocation
            [sys.executable, str(tree / "scripts/source_fingerprint.py"), "--verify-tree", str(tree), str(snapshot_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        verified = json.loads(result.stdout)
        manifest = json.loads(snapshot.read_text(encoding="utf-8"))
    except subprocess.CalledProcessError as error:
        message = f"Generation source tree {tree_id} failed verification against its snapshot: {_failure_line(error)}"
        raise SourceAdmissionError(message) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Generation source tree {tree_id} cannot be verified against its snapshot."
        raise SourceAdmissionError(message) from error
    if (
        verified.get("tree_id") != tree_id
        or verified.get("git_commit") != expected_commit
        or verified.get("source_fingerprint_sha256") != expected
        or manifest.get("git_commit") != expected_commit
        or manifest.get("source_fingerprint_sha256") != expected
    ):
        message = "Generation source tree disagrees with the admitted commit or fingerprint; publication refused."
        raise SourceAdmissionError(message)
    return AdmittedSourceTree(
        tree_id=tree_id,
        tree_path=tree,
        snapshot_path=snapshot,
        snapshot_sha256=snapshot_sha,
        git_commit=expected_commit,
        source_fingerprint_sha256=expected,
        clean=manifest.get("tracked_diff_sha256") == _EMPTY_DIFF_SHA256 and manifest.get("untracked") == [],
    )


def validate_admitted_source_before_publication() -> None:
    """Reject changed shared source before Slurm work publishes durable evidence."""
    if common.paths.source_tree_id() is not None:
        verify_admitted_source_tree()
    else:
        _validate_live_source_before_publication()


def _validate_live_source_before_publication() -> None:  # noqa: C901, PLR0912 -- one exact source admission audit
    """Recheck the live checkout against the launch commit and fingerprint using Git."""
    expected = os.environ.get("GENERATION_SOURCE_SHA256")
    if expected is None:
        return
    if _SOURCE_SHA256_PATTERN.fullmatch(expected) is None:
        message = "Generation source admission fingerprint is malformed."
        raise SourceAdmissionError(message)
    try:
        expected_commit = required_git_commit()
    except (RuntimeError, ValueError) as error:
        message = "Generation source admission commit is missing or malformed."
        raise SourceAdmissionError(message) from error
    repository = common.paths.get_project_root().resolve()
    try:
        current_commit = subprocess.run(  # noqa: S603 -- fixed Git argument vector
            ["git", "-C", str(repository), "rev-parse", "HEAD"],  # noqa: S607 -- site PATH owns Git
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        message = "Shared Generation commit cannot be verified before publication."
        raise SourceAdmissionError(message) from error
    if current_commit != expected_commit:
        message = "Shared Generation commit changed after Slurm admission; publication refused."
        raise SourceAdmissionError(message)
    try:
        result = subprocess.run(  # noqa: S603 -- fixed source fingerprint invocation
            [sys.executable, str(repository / "scripts/source_fingerprint.py"), str(repository)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        message = "Shared Generation source cannot be fingerprinted before publication."
        raise SourceAdmissionError(message) from error
    if result.stdout.strip() != expected:
        message = "Shared Generation source changed after Slurm admission; publication refused."
        raise SourceAdmissionError(message)
    snapshot_value = os.environ.get("GENERATION_SOURCE_SNAPSHOT")
    snapshot_sha = os.environ.get("GENERATION_SOURCE_SNAPSHOT_SHA256")
    if snapshot_value is None and snapshot_sha is None:
        return
    if snapshot_value is None or snapshot_sha is None or _SOURCE_SHA256_PATTERN.fullmatch(snapshot_sha) is None:
        message = "Generation source snapshot evidence is incomplete or malformed."
        raise SourceAdmissionError(message)
    storage = common.paths.get_storage_root().resolve()
    snapshot_root = common.paths.get_source_snapshots_root(storage_root=storage)
    snapshot = Path(snapshot_value)
    if (
        not snapshot.is_absolute()
        or not snapshot.is_relative_to(snapshot_root)
        or not snapshot.resolve(strict=False).is_relative_to(snapshot_root)
        or snapshot.is_symlink()
        or snapshot.name != "snapshot.json"
        or not snapshot.is_file()
        or common.serialization.file_sha256(snapshot) != snapshot_sha
    ):
        message = "Generation source snapshot manifest is missing, unsafe, or changed."
        raise SourceAdmissionError(message)
    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = "Generation source snapshot manifest cannot be read."
        raise SourceAdmissionError(message) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_kind") != "generation_source_snapshot"
        or not common.serialization.is_schema_version(payload.get("schema_version"), SOURCE_SNAPSHOT_SCHEMA_VERSION)
        or payload.get("git_commit") != expected_commit
        or payload.get("source_fingerprint_sha256") != expected
    ):
        message = "Generation source snapshot disagrees with admitted HEAD or fingerprint."
        raise SourceAdmissionError(message)
    diff = snapshot.parent / "tracked.diff"
    if diff.is_symlink() or not diff.is_file() or common.serialization.file_sha256(diff) != payload.get("tracked_diff_sha256"):
        message = "Generation source snapshot tracked diff changed."
        raise SourceAdmissionError(message)
    records = payload.get("untracked")
    if not isinstance(records, list):
        message = "Generation source snapshot untracked inventory is malformed."
        raise SourceAdmissionError(message)
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            message = "Generation source snapshot untracked record is malformed."
            raise SourceAdmissionError(message)
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            message = "Generation source snapshot untracked locator is unsafe."
            raise SourceAdmissionError(message)
        path = snapshot.parent / "untracked" / relative
        if record.get("kind") == "file":
            if path.is_symlink() or not path.is_file() or common.serialization.file_sha256(path) != record.get("sha256"):
                message = "Generation source snapshot untracked file changed."
                raise SourceAdmissionError(message)
        elif record.get("kind") == "symlink":
            if not path.is_symlink() or str(path.readlink()) != record.get("target"):
                message = "Generation source snapshot untracked link changed."
                raise SourceAdmissionError(message)
        else:
            message = "Generation source snapshot untracked kind is unsupported."
            raise SourceAdmissionError(message)


def validate_git_commit(value: Any) -> str:
    """Return one exact full lowercase Git object identifier."""
    if not isinstance(value, str) or _GIT_COMMIT_PATTERN.fullmatch(value) is None:
        message = "git_commit must be one exact 40-character lowercase Git object identifier."
        raise ValueError(message)
    return value


def required_git_commit() -> str:
    """Return the exact source commit provided by the generation launcher."""
    value = os.environ.get(GIT_COMMIT_ENVIRONMENT_VARIABLE)
    if value is None:
        message = f"{GIT_COMMIT_ENVIRONMENT_VARIABLE} is required for generation provenance."
        raise RuntimeError(message)
    return validate_git_commit(value)


def clean_repository_git_commit(
    repository_root: Path | str | None = None,
) -> str:
    """Return HEAD only when the complete repository worktree is clean.

    In source-tree mode the verified snapshot is the evidence: its commit is returned
    only if the snapshot records neither tracked changes nor untracked files.
    """
    root = common.paths.get_project_root() if repository_root is None else Path(repository_root).expanduser()
    repository = root.resolve()
    if common.paths.source_tree_id() is not None:
        admitted = verify_admitted_source_tree()
        if repository != admitted.tree_path.resolve():
            message = f"Source-tree mode admits only its executing tree {admitted.tree_path}, not {repository}."
            raise RuntimeError(message)
        if not admitted.clean:
            message = "Direct input generation requires a clean repository worktree for truthful source provenance."
            raise RuntimeError(message)
        return admitted.git_commit
    if not repository.is_dir():
        message = f"Generation source repository is not a directory: {repository}"
        raise NotADirectoryError(message)
    try:
        status = subprocess.run(  # noqa: S603 -- fixed Git inspection command
            ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            message = "Direct input generation requires a clean repository worktree for truthful source provenance."
            raise RuntimeError(message)
        commit = subprocess.run(  # noqa: S603 -- fixed Git inspection command
            ["git", "-C", str(repository), "rev-parse", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        message = f"Could not resolve exact Git source identity from {repository}."
        raise RuntimeError(message) from error
    return validate_git_commit(commit)


def input_producer_identity(source_root: Path | None = None) -> dict[str, str]:
    """Bind input-producing code and numerical versions separately from seeds.

    The declared projection covers sampling, scalar handoff, fields, schedules and
    their material/physics contracts. Source bytes remain separately recoverable
    through the admitted snapshot; documentation does not invalidate input reuse.
    The executing source directory is independent of configured resource paths.
    """
    return {
        "implementation_sha256": GENERATION_INPUT_PROJECTION.sha256(source_root),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def source_producer_identity(source_root: Path | None = None) -> dict[str, str]:
    """Bind solver invocation and canonical export conversion outside input seeds.

    The source projection includes the input projection, so any input-producing
    change also changes the source implementation digest.
    """
    return {
        **input_producer_identity(source_root),
        "source_implementation_sha256": GENERATION_SOURCE_PROJECTION.sha256(source_root),
        "h5py_version": h5py.__version__,
    }
