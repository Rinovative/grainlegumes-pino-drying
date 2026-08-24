# ruff: noqa: ARG001, D100, D103, PLR2004, PT011, S101, TC003

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src import common
from src.datasets import dataset_packages
from src.datasets.packages import dataset_packages_references as references


def _manifest(
    dataset_id: str,
    *,
    task: str = "transient_drying",
    dataset_view: str = "transient_drying",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "dataset_digest": "d" * 64,
        "payload_sha256": "a" * 64,
        "registered_task_id": None if dataset_view == "transient_drying" else task,
        "dataset_view": dataset_view,
        "evaluation_regime": "id",
        "materials": ["chickpea", "lentil"],
        "dataset_name": "small_family",
        "campaign_id": "campaign-1",
        "campaign_digest": "b" * 64,
        "source_case_count": 2,
        "source_batch_ids": ["batch-a"],
        "source_simulation_profiles": ["drying"],
        "source_git_commits": ["commit-a"],
        "channel_contract_digest": "c" * 64,
        "sample_count": 3,
        "transition_count": 5,
    }


def _prepare_manifest_file(storage_root: Path, dataset_id: str) -> None:
    path = common.paths.get_dataset_metadata_root(storage_root=storage_root) / dataset_id / "dataset_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")


def _patch_package_admission(monkeypatch: pytest.MonkeyPatch, manifests: dict[str, dict[str, object]]) -> None:
    def load(dataset_id: str, *, storage_root: Path | str | None = None) -> dict[str, object]:
        return dict(manifests[dataset_id])

    monkeypatch.setattr(references.package_manifest, "load_package_manifest", load)
    monkeypatch.setattr(references.package_manifest, "load_package_manifest_evidence", load)


def test_dataset_ref_validates_explicit_name_revision_and_display() -> None:
    assert references.DatasetRef.from_mapping({"name": "lentil+chickpea_id", "revision": 0}).display_name == "lentil+chickpea_id"
    assert references.DatasetRef("lentil+chickpea_id", 1).display_name == "lentil+chickpea_id_d1"
    with pytest.raises(ValueError):
        references.DatasetRef.from_mapping({"name": "../unsafe", "revision": 0})
    with pytest.raises(ValueError):
        references.DatasetRef.from_mapping({"name": "valid", "revision": True})


def test_publish_reuses_exact_binding_rejects_conflict_and_separates_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = {"dataset-a": _manifest("dataset-a"), "dataset-b": _manifest("dataset-b")}
    _prepare_manifest_file(tmp_path, "dataset-a")
    _prepare_manifest_file(tmp_path, "dataset-b")
    _patch_package_admission(monkeypatch, manifests)

    published = references.publish_dataset_reference("transient_drying", "family", 0, "dataset-a", storage_root=tmp_path)
    reused = references.publish_dataset_reference("transient_drying", "family", 0, "dataset-a", storage_root=tmp_path)
    assert published["status"] == "published"
    assert reused["status"] == "reused"
    with pytest.raises(
        FileExistsError,
        match=r"Dataset reference conflict: task='transient_drying', name='family', revision=0;.*Action:",
    ):
        references.publish_dataset_reference("transient_drying", "family", 0, "dataset-b", storage_root=tmp_path)

    steady = _manifest("dataset-b", task="steady_flow", dataset_view="steady_flow")
    manifests["dataset-b"] = steady
    references.publish_dataset_reference("steady_flow", "family", 0, "dataset-b", storage_root=tmp_path)
    assert len(references.list_dataset_references("transient_drying", storage_root=tmp_path)) == 1
    assert len(references.list_dataset_references("steady_flow", storage_root=tmp_path)) == 1


def test_resolve_rejects_missing_corrupt_and_traversal_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = {"dataset-a": _manifest("dataset-a")}
    _prepare_manifest_file(tmp_path, "dataset-a")
    _patch_package_admission(monkeypatch, manifests)
    references.publish_dataset_reference("transient_drying", "family", 1, "dataset-a", storage_root=tmp_path)
    resolved = references.resolve_dataset_reference("transient_drying", "family", 1, storage_root=tmp_path)
    assert resolved["dataset_id"] == "dataset-a"
    assert resolved["transition_count"] == 5
    with pytest.raises(FileNotFoundError, match=r"revision=2.*Available revisions: \[1\]"):
        references.resolve_dataset_reference("transient_drying", "family", 2, storage_root=tmp_path)

    path = common.paths.resolve_dataset_reference_path("transient_drying", "family", 1, storage_root=tmp_path)
    path.write_text(json.dumps({"task": "../escape"}), encoding="utf-8")
    with pytest.raises(ValueError):
        references.resolve_dataset_reference("transient_drying", "family", 1, storage_root=tmp_path)
    with pytest.raises(ValueError):
        common.paths.resolve_dataset_reference_path("transient_drying", "../escape", 0, storage_root=tmp_path)


def test_reference_cli_resolves_concise_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    manifests = {"dataset-a": _manifest("dataset-a")}
    _prepare_manifest_file(tmp_path, "dataset-a")
    _patch_package_admission(monkeypatch, manifests)
    references.publish_dataset_reference("transient_drying", "family", 0, "dataset-a", storage_root=tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset_packages", "resolve", "--task", "transient_drying", "--name", "family", "--revision", "0", "--storage-root", str(tmp_path)],
    )
    assert dataset_packages.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "dataset_digest": "d" * 64,
        "dataset_id": "dataset-a",
        "dataset_view": "transient_drying",
        "display_name": "family",
        "evaluation_regime": "id",
        "manifest_status": "valid",
        "name": "family",
        "revision": 0,
        "sample_count": 3,
        "source_case_count": 2,
        "status": "resolved",
        "task": "transient_drying",
        "transition_count": 5,
    }
