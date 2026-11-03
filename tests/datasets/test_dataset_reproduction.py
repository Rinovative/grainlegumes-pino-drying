"""Protect recoverable source bytes and immutable Dataset publication.

Responsibilities:
  - Exercise source capture and isolated restoration without changing real Git state
  - Check package identity boundaries, producer evidence and atomic single-folder publication

Design principles:
  - Tiny controlled Git responses retain real filesystem, patch and hash checks
  - Corrupt recovery evidence and changed selected source bytes fail closed

This module does NOT:
  - Run scientific generation, consume production data or initialize Git repositories
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import runpy
import shlex
import shutil
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from support.synthetic_task import build_synthetic_selection_provenance

from src.datasets import dataset_packages as facade
from src.datasets import packages as package_facade
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_metadata as metadata
from src.datasets.contracts import dataset_contracts_partition as partition
from src.datasets.packages import dataset_packages_builder as builder
from src.datasets.packages import dataset_packages_manifest as manifests
from src.datasets.packages import dataset_packages_reproduction as reproduction
from src.datasets.packages import dataset_packages_selection as selection

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.domain.tasks.domain_task_spec import TaskSpec


@pytest.fixture
def snapshot_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    script = Path(__file__).resolve().parents[2] / "scripts/source_fingerprint.py"
    module = runpy.run_path(str(script), run_name="test_source_reconstruction")
    repository = tmp_path / "original"
    repository.mkdir(mode=0o750)
    (repository / "code.py").write_text("value = 2\n")
    (repository / "code.py").chmod(0o640)
    (repository / "notes").mkdir(mode=0o2750)
    (repository / "notes/readme.txt").write_text("retained input\n")
    (repository / "notes/readme.txt").chmod(0o700)
    (repository / "notes/link").symlink_to("readme.txt")
    head = {"value": "a" * 40}
    diff = b"diff --git a/code.py b/code.py\n--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"

    def git(_root: Path, *arguments: str, **kwargs: Any) -> SimpleNamespace:
        operation = arguments[0]
        if operation == "diff":
            return SimpleNamespace(stdout=diff)
        if operation == "rev-parse":
            return SimpleNamespace(stdout=head["value"] + "\n")
        if operation == "ls-files":
            if "--ignored" in arguments:
                return SimpleNamespace(stdout=b"")
            value = b"notes/link\0notes/readme.txt\0" if "--others" in arguments else b"code.py\0"
            return SimpleNamespace(stdout=value)
        if operation == "ls-tree":
            return SimpleNamespace(stdout=b"code.py\0")
        if operation == "archive":
            with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
                member = tarfile.TarInfo("code.py")
                content = b"value = 1\n"
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
            return SimpleNamespace(stdout=b"")
        message = f"Unexpected read-only Git operation: {arguments}"
        raise AssertionError(message)

    module["source_fingerprint"].__globals__["_git"] = git
    snapshots = tmp_path / "snapshots"
    return module, repository, snapshots, head


def test_dirty_snapshot_restores_bytes_full_modes_and_links(snapshot_fixture: tuple, tmp_path: Path) -> None:
    module, repository, snapshots, _head = snapshot_fixture
    receipt = module["capture_source_snapshot"](repository, snapshots)
    destination = tmp_path / "restored"
    verified = module["restore_source_snapshot"](repository, Path(receipt["snapshot_path"]), destination)
    assert verified["full_modes_recoverable"] is True
    assert (destination / "code.py").read_text() == "value = 2\n"
    assert (destination / "code.py").stat().st_mode & 0o7777 == 0o640
    assert (destination / "notes").stat().st_mode & 0o7777 == (repository / "notes").stat().st_mode & 0o7777
    assert (destination / "notes/link").readlink() == Path("readme.txt")
    assert not (destination / ".git").exists()
    (destination / "code.py").write_text("changed\n")
    with pytest.raises(ValueError, match=r"inventory|content"):
        module["verify_source_snapshot"](Path(receipt["snapshot_path"]), destination)


def test_snapshot_commit_identity_and_corruption_fail_closed(snapshot_fixture: tuple) -> None:
    module, repository, snapshots, head = snapshot_fixture
    first = module["capture_source_snapshot"](repository, snapshots)
    head["value"] = "b" * 40
    second = module["capture_source_snapshot"](repository, snapshots)
    assert first["fingerprint"] == second["fingerprint"]
    assert first["snapshot_path"] != second["snapshot_path"]
    saved = Path(second["snapshot_path"]).parent / "untracked/notes/readme.txt"
    saved.write_text("corrupted\n")
    with pytest.raises(ValueError, match=r"verification|fingerprint"):
        module["capture_source_snapshot"](repository, snapshots)


def test_restore_rejects_escaping_links_and_occupied_destination(snapshot_fixture: tuple, tmp_path: Path) -> None:
    module, repository, snapshots, _head = snapshot_fixture
    receipt = module["capture_source_snapshot"](repository, snapshots)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        module["restore_source_snapshot"](repository, Path(receipt["snapshot_path"]), occupied)
    with pytest.raises(ValueError, match="link"):
        module["source_fingerprint"].__globals__["_validate_link_target"]("notes/link", "../../outside")


def _provenance() -> dict[str, Any]:
    value = build_synthetic_selection_provenance(dataset_name="physical", sample_ids=("case_1",))
    value.pop("dataset_revision", None)
    value.pop("dataset_purpose", None)
    value["builder_identity"]["implementation_sha256"] = reproduction.conversion_implementation_sha256("steady_flow")
    value["producer_evidence"] = {
        "source_snapshot": {
            "relative_path": f"source_snapshots/{'a' * 40}/snapshot.json",
            "sha256": "b" * 64,
            "git_commit": "a" * 40,
            "source_fingerprint_sha256": "c" * 64,
        },
        "environment": {"python": "3.12", "platform": "test", "packages": {}},
    }
    return value


def test_dataset_identity_excludes_labels_and_selection_method_but_binds_conversion() -> None:
    original = _provenance()
    _identifier, digest = identity.package_identity_from_provenance(original)
    renamed = copy.deepcopy(original)
    renamed["dataset_name"] = "another_label"
    renamed["selection"] = {"kind": "seed_ranked", "root_seed": 3}
    renamed["producer_evidence"]["source_snapshot"]["git_commit"] = "d" * 40
    assert identity.package_identity_from_provenance(renamed)[1] == digest
    renamed["builder_identity"]["implementation_sha256"] = "e" * 64
    assert identity.package_identity_from_provenance(renamed)[1] != digest


def test_publication_is_one_atomic_immutable_folder(tmp_path: Path) -> None:
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(b"tiny tensor fixture")
        return {"sample_count": 1, "transition_count": 0}

    request = {"selection": {"kind": "exact"}}
    args = {
        "dataset_id": dataset_id,
        "task": "steady_flow",
        "prefix": prefix,
        "build_payload": payload,
        "selection_request": request,
        "storage_root": tmp_path,
    }
    payload_path, manifest_path, reused, _manifest = builder._publish(**args)
    assert payload_path.parent == manifest_path.parent == tmp_path / "02_datasets" / "steady_flow" / dataset_id
    assert (payload_path.name, manifest_path.name) == ("dataset.pt", "dataset_manifest.json")
    assert reused is False
    assert builder._publish(**args)[2] is True
    payload_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="payload bytes"):
        manifests.load_package_manifest("steady_flow", dataset_id, storage_root=tmp_path)


def _publish_fixture(storage: Path, provenance: dict[str, Any], builds: list[str] | None = None) -> tuple[str, Path, bool]:
    """Publish one tiny steady package under the name its provenance carries."""
    dataset_id, digest = identity.package_identity_from_provenance(provenance)

    def payload(path: Path) -> dict[str, int]:
        if builds is not None:
            builds.append(dataset_id)
        path.write_bytes(b"tiny tensor fixture")
        return {"sample_count": 1, "transition_count": 0}

    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}
    published, _manifest_path, reused, _manifest = builder._publish(
        task="steady_flow",
        dataset_id=dataset_id,
        prefix=prefix,
        build_payload=payload,
        selection_request={"selection": {"kind": "exact"}},
        storage_root=storage,
    )
    return dataset_id, published.parent, reused


def test_identical_content_cannot_be_published_under_a_second_name(tmp_path: Path) -> None:
    """The identity registry binds the content digest to its first Dataset directory, whatever the name or revision."""
    provenance = _provenance()
    builds: list[str] = []
    first_id, first_dir, _reused = _publish_fixture(tmp_path, provenance, builds)
    renamed = {**provenance, "dataset_name": f"{first_id}__rev-copy"}

    with pytest.raises(FileExistsError, match=rf"Identical Dataset content already exists as steady_flow/{first_id}; use that name"):
        _publish_fixture(tmp_path, renamed, builds)

    assert builds == [first_id]  # refused before the payload was built
    assert [path.name for path in first_dir.parent.iterdir()] == [first_id]
    assert not list((tmp_path / "02_datasets" / ".state").glob(".*.tmp"))

    shutil.rmtree(first_dir)  # The registry entry is stale now and is replaced by the next publication.
    second_id, second_dir, reused = _publish_fixture(tmp_path, renamed, builds)
    assert (second_id, second_dir.name, reused) == (f"{first_id}__rev-copy", f"{first_id}__rev-copy", False)
    with pytest.raises(FileExistsError, match=rf"already exists as steady_flow/{second_id}"):
        _publish_fixture(tmp_path, provenance, builds)


def test_occupied_name_with_different_content_points_to_a_revision(tmp_path: Path) -> None:
    """A name keeps its content; the same name with other content is refused, a revision name is a separate Dataset."""
    provenance = _provenance()
    first_id, first_dir, _reused = _publish_fixture(tmp_path, provenance)
    changed = copy.deepcopy(provenance)
    changed["grid"]["shape"] = [*changed["grid"]["shape"][:-1], changed["grid"]["shape"][-1] + 1]
    before = (first_dir / "dataset_manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match=r"already binds different immutable content; pass --revision <label>"):
        _publish_fixture(tmp_path, changed)

    assert (first_dir / "dataset_manifest.json").read_bytes() == before
    revised_id, revised_dir, reused = _publish_fixture(tmp_path, {**changed, "dataset_name": f"{first_id}__rev-b"})
    assert (revised_id, revised_dir.parent, reused) == (f"{first_id}__rev-b", first_dir.parent, False)
    assert _publish_fixture(tmp_path, provenance)[2] is True


def test_producer_evidence_rejects_escape() -> None:
    value = _provenance()["producer_evidence"]
    reproduction.validate_producer_evidence(value)
    value["source_snapshot"]["relative_path"] = "../elsewhere/snapshot.json"
    with pytest.raises(ValueError, match="unsafe"):
        reproduction.validate_producer_evidence(value)


def test_inspection_exposes_parameters_and_recovery_without_tensor_loading(
    snapshot_fixture: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, repository, _snapshots, _head = snapshot_fixture
    storage = tmp_path / "storage"
    receipt = module["capture_source_snapshot"](repository, storage / "source_snapshots")
    provenance = _provenance()
    snapshot_path = Path(receipt["snapshot_path"])
    provenance["producer_evidence"]["source_snapshot"] = {
        "relative_path": snapshot_path.relative_to(storage).as_posix(),
        "sha256": receipt["snapshot_sha256"],
        "git_commit": "a" * 40,
        "source_fingerprint_sha256": receipt["fingerprint"],
    }
    source = provenance["source_case_identities"][0]["source"]
    source["source_name"] = source["campaign_name"] + "__" + source["campaign_digest"][:16]
    relative = f"01_generation/steady_flow/{source['source_name']}/lentil/case_0001/result/case.h5"
    source["case_hdf5_relative"] = relative
    hdf5 = storage / relative
    hdf5.parent.mkdir(parents=True)
    hdf5.write_bytes(b"retained physical input")
    raw_root = hdf5.parent.parent / "inputs"
    raw_root.mkdir()
    (raw_root / "case.json").write_text(json.dumps({"sampled_values": {"temperature": 300}, "sampled_units": {"temperature": "K"}}))
    manifest = {
        **provenance,
        "selection": {
            "kind": "seed_ranked",
            "root_seed": 17,
            "source_names": [source["source_name"]],
            "target_counts": {"lentil": 1},
        },
        "source_case_count": 1,
        "transition_count": 0,
        "payload_filename": "dataset.pt",
        "sample_count": 1,
    }
    monkeypatch.setattr(manifests, "load_package_manifest_evidence", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(reproduction.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""))
    report = reproduction.inspect_dataset_reproduction("steady_flow", "fixture", storage_root=storage)
    assert report["cases"][0]["effective_parameters"]["sampled_values"]["temperature"] == 300
    assert report["campaigns"][source["source_name"]]["plan"].endswith(f"/{source['source_name']}/_record/campaign_plan.json")
    assert report["cases"][0]["hdf5"]["available"] is True
    assert report["cases"][0]["hdf5"]["hash_verified_in_this_inspection"] is False
    assert report["producer"]["source_recovery"]["manifest_verified"] is True
    readable = facade.format_dataset_inspection(report)
    assert "Dataset: fixture" in readable
    assert "Selection: seed_ranked; seed 17; counts lentil=1" in readable
    assert "Sources, in Dataset order:" in readable
    assert source["source_name"] in readable
    assert "Generation root/candidate seeds:" in readable
    assert "Inputs:" in readable
    assert "Outputs:" in readable
    assert "Grid shape:" in readable
    assert report["source_outcomes"] == manifest["source_outcomes"]
    assert "original candidates failed 0/1 (0.0%)" in readable
    assert "Current interpreter admission:" in readable
    assert "Base Git commit unavailable" in readable
    assert "scalars: unavailable" in readable
    assert "python -m src.datasets.dataset_packages inspect steady_flow fixture --storage-root" in readable
    assert "--json" in readable
    assert report["relevant_compatibility"]["current_interpreter_admission"] in readable
    assert report["regeneration"]["restore_command"] in readable
    assert report["regeneration"]["dataset_rebuild_command"] in readable
    assert "do not prove scientific equivalence" in readable

    transient_report = copy.deepcopy(report)
    transient_report["channel_contract"] = {
        "state": [{"name": "T", "unit": "K"}],
        "static": [{"name": "x", "unit": "m"}],
        "boundary": [{"name": "T_in", "unit": "K"}],
        "scalars": [{"name": "k", "unit": "W/(m*K)"}],
        "target": [{"name": "delta_T", "unit": "K"}],
        "time": {"fields": [{"name": "dt", "unit": "h"}]},
        "storage": "absolute_physical_states",
        "tensor_dtype": "float32",
    }
    transient_report["cases"][0]["solver_environment"] = None
    campaign = transient_report["campaigns"][source["source_name"]]
    campaign["generation_source_snapshot"] = None
    campaign["generation_invocation"] = {
        "recorded_execution_command": None,
        "scientific_command": "./scripts/generation run campaign.yaml",
        "execution_boundary": "Run through the recorded Slurm boundary.",
        "execution_choices_not_recorded": ["requested_cores"],
    }
    transient_readable = facade.format_dataset_inspection(transient_report)
    assert "State: T (K)" in transient_readable
    assert "Static: x (m)" in transient_readable
    assert "Boundary: T_in (K)" in transient_readable
    assert "Scalars: k (W/(m*K))" in transient_readable
    assert "Targets: delta_T (K)" in transient_readable
    assert "Solver environment: unavailable in inspection evidence" in transient_readable
    assert "Generation code snapshot: unavailable in inspection evidence" in transient_readable
    assert "Recorded Generation command: not recorded" in transient_readable
    assert "Scientific Generation command: ./scripts/generation run campaign.yaml" in transient_readable
    assert "Run through the recorded Slurm boundary." in transient_readable
    assert "Execution choices not recorded: requested_cores" in transient_readable
    assert "Missing or changed retained files: none listed" in transient_readable
    solver_report = copy.deepcopy(transient_report)
    solver_report["cases"][0]["solver_environment"] = {
        "python_version": "3.12.14",
        "loaded_modules": "COMSOL/6.3",
        "executable_path": "/usr/bin/python",
        "hostname": "node-1",
        "allocated_node": "node-1",
        "scheduler_job_id": "12345",
        "requested_cores": 8,
    }
    solver_readable = facade.format_dataset_inspection(solver_report)
    assert "Python 3.12.14" in solver_readable
    assert "modules COMSOL/6.3" in solver_readable
    assert "executable /usr/bin/python" in solver_readable
    assert "host node-1" in solver_readable
    assert "allocated node node-1" in solver_readable
    assert "scheduler job 12345" in solver_readable
    assert "requested cores 8" in solver_readable
    assert report["cases"][0]["effective_parameters"]["sampled_values"]["temperature"] == 300

    planned_report = copy.deepcopy(report)
    planned_report["plan_invocation"] = [
        "/recorded/runtime/venvs/native/bin/python",
        "-m",
        "src.datasets.dataset_packages",
        "run",
        "configs/02_datasets/plan.yaml",
        "--revision",
        "smoke",
    ]
    planned_readable = facade.format_dataset_inspection(planned_report)
    hint = next(line for line in planned_readable.splitlines() if line.startswith("Re-run Dataset plan"))
    # The plan runs from the login node: the hint says so and its command pastes from repo/ without the recorded absolute interpreter.
    assert "login node" in hint
    assert "Slurm allocation" not in hint
    assert hint.endswith(": ../runtime/venvs/native/bin/python -m src.datasets.dataset_packages run configs/02_datasets/plan.yaml --revision smoke")
    assert "/recorded/" not in hint
    assert "Re-run Dataset plan" not in readable

    monkeypatch.setattr(facade, "inspect_dataset_package", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(sys, "argv", ["dataset-packages", "inspect", "steady_flow", "fixture"])
    assert facade.main() == 0
    assert capsys.readouterr().out.strip() == readable
    monkeypatch.setattr(sys, "argv", ["dataset-packages", "inspect", "steady_flow", "fixture", "--json"])
    assert facade.main() == 0
    assert json.loads(capsys.readouterr().out) == report
    assert report["producer"]["source_recovery"]["base_commit_available"] is False
    report["producer"]["source_recovery"] = {"manifest": "absent/snapshot.json", "available": False, "verified": False}
    monkeypatch.setattr(sys, "argv", ["dataset-packages", "inspect", "steady_flow", "fixture"])
    assert facade.main() == 0
    missing = capsys.readouterr().out
    assert "manifest verified not checked" in missing
    assert "full modes recoverable unknown" in missing
    assert "Missing or changed retained files: unknown (inventory not checked)" in missing
    assert report["campaigns"][source["source_name"]]["plan_available"] is False


def test_publication_refuses_source_drift_before_exposing_the_package(tmp_path: Path) -> None:
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}
    calls = []

    def verify() -> None:
        calls.append("verified")
        if len(calls) == 2:
            message = "Producing source changed during conversion."
            raise ValueError(message)

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(b"tiny fixture")
        return {"sample_count": 1, "transition_count": 0}

    with pytest.raises(ValueError, match="Producing source changed"):
        builder._publish(
            dataset_id=dataset_id,
            task="steady_flow",
            prefix=prefix,
            build_payload=payload,
            selection_request={"selection": {"kind": "exact"}},
            storage_root=tmp_path,
            verify_producer=verify,
        )
    assert len(calls) == 2
    assert not (tmp_path / "02_datasets" / "steady_flow" / dataset_id).exists()


def test_snapshot_inventory_refuses_symlink_ancestor_before_reading(snapshot_fixture: tuple, tmp_path: Path) -> None:
    module, repository, _snapshots, _head = snapshot_fixture
    external = tmp_path / "external"
    external.mkdir()
    (external / "file.txt").write_text("test-owned outside bytes")
    (repository / "alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module["source_fingerprint"].__globals__["_inventory"](repository, [Path("alias/file.txt")])


def test_mismatching_rebuild_retains_evidence_without_publishing(tmp_path: Path) -> None:
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(b"unexpected reconstructed bytes")
        return {"sample_count": 1, "transition_count": 0}

    with pytest.raises(ValueError, match="unpublished evidence retained"):
        builder._publish(
            dataset_id=dataset_id,
            task="steady_flow",
            prefix=prefix,
            build_payload=payload,
            selection_request={"selection": {"kind": "exact"}},
            storage_root=tmp_path,
            expected_payload_sha256="0" * 64,
        )
    root = tmp_path / "02_datasets"
    assert not (root / "steady_flow" / dataset_id).exists()
    evidence = list((root / ".state").glob("*.rebuild-mismatch"))
    assert len(evidence) == 1
    assert (evidence[0] / "payload" / prefix["payload_filename"]).read_bytes() == b"unexpected reconstructed bytes"


def test_producer_verification_rejects_added_source_and_resurrected_deletion(snapshot_fixture: tuple) -> None:
    module, repository, snapshots, _head = snapshot_fixture
    receipt = module["capture_source_snapshot"](repository, snapshots)
    extra = repository / "new_interpreter.py"
    extra.write_text("value = 9\n")
    with pytest.raises(ValueError, match="membership"):
        module["verify_source_snapshot"](Path(receipt["snapshot_path"]), repository)
    extra.unlink()
    deleted = repository / "deleted_source.py"
    deleted.write_text("resurrected = True\n")
    with pytest.raises(ValueError, match="membership"):
        module["verify_source_snapshot"](Path(receipt["snapshot_path"]), repository)


def test_actual_interpreter_change_rejects_admission_but_documentation_is_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    payload = tmp_path / "dataset.pt"
    payload.write_bytes(b"test-owned tensor bytes")
    manifest = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": digest,
        "payload_filename": payload.name,
        "sample_count": 1,
        "source_case_count": 1,
        "transition_count": 0,
        "payload_sha256": reproduction.common.serialization.file_sha256(payload),
    }
    projection = reproduction.DATASET_CONVERSION_PROJECTIONS["steady_flow"]
    executing_source = Path(reproduction.__file__).resolve().parents[2]
    copied_source = tmp_path / "selected-source"
    for relative in projection.members(executing_source):
        destination = copied_source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(executing_source / relative, destination)
    monkeypatch.setattr(
        reproduction,
        "conversion_implementation_sha256",
        lambda view: reproduction.DATASET_CONVERSION_PROJECTIONS[view].sha256(copied_source),
    )
    assert reproduction.conversion_implementation_sha256("steady_flow") == provenance["builder_identity"]["implementation_sha256"]
    (copied_source / "README.md").write_text("Unrelated human documentation changes.\n")
    interpreter = copied_source / "datasets/runtime/dataset_runtime_steady.py"
    text = interpreter.read_text()
    interpreter.write_text(text.replace("Return one task-ordered pair and a defensive metadata copy.", "Explain the same task-ordered pair."))
    assert reproduction.conversion_implementation_sha256("steady_flow") == provenance["builder_identity"]["implementation_sha256"]
    manifests.validate_manifest_content(manifest, task="steady_flow", dataset_id=dataset_id, payload_path=payload)
    interpreter.write_text(text.replace('"x": self.inputs[index]', '"x": self.inputs[index] * 2'))
    assert reproduction.conversion_implementation_sha256("steady_flow") != provenance["builder_identity"]["implementation_sha256"]
    with pytest.raises(ValueError, match="implementation changed"):
        manifests.validate_manifest_content(manifest, task="steady_flow", dataset_id=dataset_id, payload_path=payload)


def test_fresh_reconstruction_output_reads_dependencies_without_mutating_them(tmp_path: Path) -> None:
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    sentinel = dependencies / "retained-source.txt"
    sentinel.write_bytes(b"original bytes")
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}
    content = b"exact reconstructed bytes"

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(content)
        return {"sample_count": 1, "transition_count": 0}

    output = tmp_path / "fresh-verification"
    args = {
        "dataset_id": dataset_id,
        "task": "steady_flow",
        "prefix": prefix,
        "build_payload": payload,
        "selection_request": {"selection": {"kind": "exact"}},
        "storage_root": dependencies,
        "expected_payload_sha256": hashlib.sha256(content).hexdigest(),
    }
    payload_path, manifest_path, reused, _manifest = builder._publish(**args, output_directory=output)
    assert payload_path.parent == manifest_path.parent == output
    assert reused is False
    assert sentinel.read_bytes() == b"original bytes"
    assert list(dependencies.iterdir()) == [sentinel]
    with pytest.raises(ValueError, match="new directory outside"):
        builder._publish(**args, output_directory=output)
    with pytest.raises(ValueError, match="new directory outside"):
        builder._publish(**args, output_directory=dependencies / "forbidden")


def test_steady_metadata_reads_manifest_and_payload_in_same_folder(tmp_path: Path) -> None:
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(b"small metadata fixture")
        return {"sample_count": 1, "transition_count": 0}

    payload_path, manifest_path, _reused, _manifest = builder._publish(
        dataset_id=dataset_id,
        task="steady_flow",
        prefix=prefix,
        build_payload=payload,
        selection_request={"selection": {"kind": "exact"}},
        storage_root=tmp_path,
    )
    manifest = metadata._load_manifest(manifest_path.parent, dataset_id=dataset_id, dataset_path=payload_path)
    assert manifest["dataset_digest"] == digest
    with pytest.raises(ValueError, match="same Dataset root"):
        metadata.load_dataset_metadata_summary(
            dataset_id,
            task=cast("TaskSpec", SimpleNamespace(id="steady_flow")),
            dataset_root=tmp_path / "02_datasets",
            metadata_root=tmp_path / "other",
        )


def test_external_reconstruction_refuses_targets_appearing_during_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = _provenance()
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}
    output = tmp_path / "raced-output"

    @contextmanager
    def raced_lock(*_args: Any, **_kwargs: Any) -> Iterator[None]:
        output.mkdir()
        yield

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(b"fixture")
        return {"sample_count": 1, "transition_count": 0}

    args = {
        "dataset_id": dataset_id,
        "task": "steady_flow",
        "prefix": prefix,
        "build_payload": payload,
        "selection_request": {"selection": {"kind": "exact"}},
        "storage_root": tmp_path / "dependencies",
        "output_directory": output,
    }
    with monkeypatch.context() as patch:
        patch.setattr(builder.common.locking, "exclusive_file_lock", raced_lock)
        with pytest.raises(FileExistsError, match="before acquiring"):
            builder._publish(**args)
    assert list(output.iterdir()) == []
    output.rmdir()

    def raced_payload(path: Path) -> dict[str, int]:
        output.symlink_to(tmp_path / "absent-target", target_is_directory=True)
        return payload(path)

    args["build_payload"] = raced_payload
    with pytest.raises(FileExistsError, match="appeared while holding"):
        builder._publish(**args)
    assert output.is_symlink()
    assert not (tmp_path / "absent-target").exists()


def test_reuse_validates_changed_ranked_claim_without_reranking_unchanged_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = _provenance()
    method = {"kind": "seed_ranked", "source_names": ["source_1"], "target_counts": {"lentil": 1}, "root_seed": 17}
    provenance["selection"] = method
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    prefix = {**provenance, "dataset_id": dataset_id, "dataset_digest": digest, "payload_filename": "dataset.pt", "source_case_count": 1}
    reference = {"case_id": "case_1"}
    request = {"selection": method, "source_task": "steady_flow", "source_cases": [reference]}

    def ranked(_task: str, _names: list[str], counts: dict[str, int], *, root_seed: int, **_kwargs: Any) -> tuple:
        return (reference,) if counts == {"lentil": 1} and root_seed == 17 else ()

    def payload(path: Path) -> dict[str, int]:
        path.write_bytes(b"immutable fixture")
        return {"sample_count": 1, "transition_count": 0}

    monkeypatch.setattr(selection, "select_ranked_cases", ranked)
    args = {
        "dataset_id": dataset_id,
        "task": "steady_flow",
        "prefix": prefix,
        "build_payload": payload,
        "selection_request": request,
        "storage_root": tmp_path,
    }
    _payload, manifest_path, reused, _manifest = builder._publish(**args)
    assert reused is False
    before = manifest_path.read_bytes()
    for changed in ({**method, "root_seed": 18}, {**method, "target_counts": {"lentil": 2}}):
        with pytest.raises(ValueError, match="Seed-ranked Dataset source cases disagree"):
            builder._publish(**{**args, "selection_request": {**request, "selection": changed}, "prefix": {**prefix, "selection": changed}})
        assert manifest_path.read_bytes() == before

    def forbidden_rerank(*_args: Any, **_kwargs: Any) -> tuple:
        pytest.fail("Unchanged immutable selection must not rerank a mutable Source pool")

    monkeypatch.setattr(selection, "select_ranked_cases", forbidden_rerank)
    assert builder._publish(**args)[2] is True
    assert manifest_path.read_bytes() == before


def test_generation_inspection_command_separates_recorded_resources_from_science() -> None:
    run = {"campaign_config": "configs/01_generation/research campaign.yaml"}
    science = {"materials": ["lentil"], "cases_per_material": 3, "duration_hours": 0.5, "revision": "fix2"}
    execution = {"node_jobs": 1, "max_admission_cases": 4, "cores_per_case": 8, "case_timeout_seconds": 1500, "no_retry": True}
    report = reproduction._generation_invocation(run, {"scientific_overrides": science, "execution_overrides": execution})
    scientific = shlex.split(report["scientific_command"])
    # A Source revision names a different Source, so the reproduction command must carry it.
    assert scientific[scientific.index("--revision") + 1] == "fix2"
    complete = shlex.split(report["recorded_execution_command"])
    assert complete[: len(scientific)] == scientific
    assert complete[len(scientific) :] == [
        "--node-jobs",
        "1",
        "--max-admission-cases",
        "4",
        "--cores-per-case",
        "8",
        "--case-timeout-seconds",
        "1500",
        "--no-retry",
    ]
    assert report["execution_choices_not_recorded"] == []
    missing = reproduction._generation_invocation(run, {"scientific_overrides": science})
    assert missing["scientific_command"] == report["scientific_command"]
    assert missing["recorded_execution_command"] is None
    assert missing["execution_overrides"] is None
    assert "no_retry" in missing["execution_choices_not_recorded"]
    partial = reproduction._generation_invocation(run, {"scientific_overrides": science, "execution_overrides": {"cores_per_case": 2}})
    assert shlex.split(partial["recorded_execution_command"])[len(scientific) :] == ["--cores-per-case", "2"]
    assert "max_admission_cases" in partial["execution_choices_not_recorded"]


def test_readable_dataset_name_binds_fixed_roles_membership_and_seed() -> None:
    provenance = build_synthetic_selection_provenance(dataset_name="readable-physical", sample_ids=("a", "b", "c"))
    recipe = {"kind": "exact", "train": ["a"], "validation": ["b"], "id_test": ["c"]}
    provenance["case_partition"] = partition.build_fixed_partition(
        ("a", "b", "c"),
        material_by_case=dict.fromkeys(("a", "b", "c"), "lentil"),
        material_roles={"lentil": "id"},
        recipe=recipe,
        seed_input=9,
        seed=1028622135437308793,
    )
    dataset_id, digest = identity.package_identity_from_provenance(provenance)
    assert dataset_id == "readable-physical"
    changed = copy.deepcopy(provenance)
    changed["case_partition"]["recipe"] = {"kind": "exact", "train": ["b"], "validation": ["a"], "id_test": ["c"]}
    changed["case_partition"]["train"] = ["b"]
    changed["case_partition"]["validation"] = ["a"]
    assert identity.package_identity_from_provenance(changed)[0] == dataset_id
    assert identity.package_identity_from_provenance(changed)[1] != digest
    corrupt = copy.deepcopy(provenance)
    corrupt["case_partition"]["train"] = ["b"]
    with pytest.raises(ValueError, match="cover source cases once"):
        identity.package_identity_from_provenance(corrupt)


def test_content_digest_ignores_split_seeds_but_binds_membership() -> None:
    """The same ordered membership from other split seeds is the same content; moving a case is not."""
    provenance = build_synthetic_selection_provenance(dataset_name="content", sample_ids=("a", "b", "c"))
    provenance["case_partition"] = partition.build_fixed_partition(
        ("a", "b", "c"),
        material_by_case=dict.fromkeys(("a", "b", "c"), "lentil"),
        material_roles={"lentil": "id"},
        recipe={"kind": "exact", "train": ["a"], "validation": ["b"], "id_test": ["c"]},
        seed_input=9,
        seed=1028622135437308793,
    )
    baseline = identity.package_content_digest(provenance)
    reseeded = copy.deepcopy(provenance)
    reseeded["case_partition"]["seed_input"] = 10
    reseeded["case_partition"]["seed"] = 77
    assert identity.package_content_digest(reseeded) == baseline
    assert identity.package_identity_from_provenance(reseeded)[1] != identity.package_identity_from_provenance(provenance)[1]
    moved = copy.deepcopy(provenance)
    moved["case_partition"]["recipe"] = {"kind": "exact", "train": ["b"], "validation": ["a"], "id_test": ["c"]}
    moved["case_partition"]["train"] = ["b"]
    moved["case_partition"]["validation"] = ["a"]
    assert identity.package_content_digest(moved) != baseline


def test_declared_evaluation_assignments_are_metadata_only_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:

    manifests_by_id: dict[str, dict[str, Any]] = {
        "train": {"case_partition": {"ood": ["primary-ood"]}},
        "external-b": {"case_partition": {"ood": ["b-ood"]}},
        "external-a": {"case_partition": {"ood": ["a-ood"]}},
        "primary-only": {"case_partition": {"ood": ["primary-ood"]}},
        "no-ood": {"case_partition": {"ood": []}},
        "no-partition": {"case_partition": None},
    }
    calls: list[tuple[str, Path | str | None]] = []

    def load_evidence(task: str, dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
        assert task == "steady_flow"
        calls.append((dataset_id, storage_root))
        return manifests_by_id[dataset_id]

    monkeypatch.setattr(manifests, "load_package_manifest_evidence", load_evidence)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("Declared assignments must not load tensors"))
    storage = Path("metadata-only-root")
    expected = [
        {
            "dataset_id": dataset_id,
            "role": "ood",
            "relation": "Dataset-declared material holdout",
            "assumptions": [],
        }
        for dataset_id in ("external-b", "external-a", "train")
    ]

    assert package_facade.declared_evaluation_assignments("steady_flow", "train", ("external-b", "external-a"), storage_root=storage) == expected
    assert calls == [("train", storage), ("external-b", storage), ("external-a", storage)]
    assert package_facade.declared_evaluation_assignments("steady_flow", "primary-only") == [{**expected[-1], "dataset_id": "primary-only"}]

    calls.clear()
    with pytest.raises(ValueError, match="must declare a fixed case partition"):
        manifests.declared_evaluation_assignments("steady_flow", "no-partition")
    with pytest.raises(ValueError, match="non-empty fixed OOD partition"):
        manifests.declared_evaluation_assignments("steady_flow", "train", ("no-ood",))
    with pytest.raises(ValueError, match="must be unique"):
        manifests.declared_evaluation_assignments("steady_flow", "train", ("external-b", "external-b"))
    with pytest.raises(ValueError, match="must be unique"):
        manifests.declared_evaluation_assignments("steady_flow", "train", ("train",))
