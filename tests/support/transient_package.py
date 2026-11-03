"""Build the compact transient Source-to-Dataset package once and hand out private copies.

Responsibilities:
  - Patch test-owned Dataset producer evidence for publication tests
  - Build five fake-COMSOL Sources and one partitioned, sharded training Dataset per session
  - Give each test a private storage copy so cleanup and removal tests stay isolated

Design principles:
  - The expensive solver, admission, and publication chain runs once; every consumer reads
    byte-identical evidence

This module does NOT:
  - Run licensed COMSOL, train models, or validate physical accuracy
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from src import common, datasets, generation
from src.datasets.packages import dataset_packages_builder as builder
from src.datasets.packages import dataset_packages_trajectory as trajectory
from support import generation_fixtures
from support.generation_fixtures import successful_source_case
from support.synthetic_task import build_synthetic_selection_provenance

_TARGET_SHARD_BYTES = 1_000_000
_SELECTOR_KEYS = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)


def patch_dataset_producer_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide test-owned producer metadata; full source recovery has dedicated tests."""
    producer = build_synthetic_selection_provenance(dataset_name="integration", sample_ids=("case_1",))["producer_evidence"]
    monkeypatch.setattr(datasets.packages.reproduction, "capture_producer_evidence", lambda **_kwargs: copy.deepcopy(producer))
    monkeypatch.setattr(datasets.packages.reproduction, "verify_current_producer", lambda *_args, **_kwargs: None)


@pytest.fixture
def _dataset_producer_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the test-owned producer evidence for one test."""
    patch_dataset_producer_evidence(monkeypatch)


def _publish_transient_selection(
    storage_root: Path,
    *,
    source_cases: list[dict[str, str]],
    dataset_name: str,
    partition_spec: dict[str, Any],
) -> tuple[str, Path]:
    """Publish and reopen one exact current Dataset with a Dataset-owned partition."""
    request = {
        "schema_kind": "dataset_selection_request",
        "schema_version": 2,
        "source_task": "transient_drying",
        "dataset_name": dataset_name,
        "dataset_view": "transient_drying",
        "selection": {"kind": "exact"},
        "source_cases": source_cases,
    }
    published = builder.build_dataset_selection(request, storage_root=storage_root, partition_spec=partition_spec)
    dataset_id = published["dataset_id"]
    manifest = datasets.packages.load_package_manifest("transient_drying", dataset_id, storage_root=storage_root)
    index_path = Path(published["payload_path"])
    index = trajectory.load_transient_index(index_path)
    assert manifest["dataset_id"] == index["dataset_id"] == dataset_id
    assert manifest["source_case_count"] == len(source_cases)
    return dataset_id, index_path


def _clone_item(value: Any) -> Any:
    """Return a deep tensor copy of one nested transient item."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_item(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_item(item) for item in value]
    return value


def _build_transient_package(root: Path) -> dict[str, Any]:
    """Build five real Source decisions and one partitioned immutable training selection."""
    project_root = root / "project"
    factory = generation_fixtures.make_generation_config_factory(project_root)
    config_path, _ = factory(
        simulation_profile="transient_drying",
        executable=generation_fixtures.write_fake_comsol(root),
        natural_count=5,
        grid_overrides={"nx": 16, "ny": 16, "Lx": 1.5, "Ly": 1.5},
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = root / "scientific-storage"
    source_root = generation.lifecycle.create_campaign(campaign, storage_root=storage)
    batch = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(batch, 5, storage_root=storage)
    for case_index in batch.case_indices:
        successful_source_case(campaign.source_name, batch, case_index, storage, root / "work")
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = generation.publication.campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=root / "logs",
        storage_root=storage,
        input_sources={
            batch.batch_name: {
                "input_generation_id": inputs.input_generation_id,
                "source_git_commit": "a" * 40,
                "case_indices": list(batch.case_indices),
            }
        },
    )
    common.serialization.atomic_write_json(
        generation.publication.campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), run_manifest
    )
    rows = generation.publication.catalog.list_campaign_cases("transient_drying", campaign.source_name, storage_root=storage)
    assert len(rows) == 5
    assert all(row["outcome"] == "successful" for row in rows)
    selectors = [{key: row[key] for key in _SELECTOR_KEYS} for row in rows]
    train_dataset_id, index_path = _publish_transient_selection(
        storage,
        source_cases=selectors[:4],
        dataset_name="transient_train_lifecycle",
        partition_spec={
            "seed_input": 31,
            "seed": 31,
            "recipe": {"kind": "seeded_material_fraction", "train_fraction": 0.5, "validation_fraction": 0.25, "id_test_fraction": 0.25},
            "material_roles": {"lentil": "id"},
        },
    )
    datasets.packages.transient_shards.build_transient_shards(train_dataset_id, storage_root=storage, target_shard_bytes=_TARGET_SHARD_BYTES)
    canonical = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        source_root=storage,
    )
    try:
        expected_item = _clone_item(canonical[0])
    finally:
        canonical.close()
    return {
        "project_root": project_root,
        "root": storage,
        "source_root": source_root,
        "source_campaign": campaign.source_name,
        "source_selectors": selectors,
        "train_dataset": train_dataset_id,
        "case_partition": datasets.packages.load_package_manifest("transient_drying", train_dataset_id, storage_root=storage)["case_partition"],
        "expected_item": expected_item,
    }


@pytest.fixture(scope="session")
def _transient_package_template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the package once under the environment the per-test fixtures provide."""
    root = tmp_path_factory.mktemp("transient-package")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
        monkeypatch.setenv("STORAGE_ROOT", str(root / "storage"))
        monkeypatch.setenv("PROJECT_ROOT", str(root / "project"))
        monkeypatch.setenv("FAKE_COMSOL_TRANSIENT_TIME_MODE", "short_rollout")
        patch_dataset_producer_evidence(monkeypatch)
        return _build_transient_package(root)


@pytest.fixture
def transient_package(
    _transient_package_template: dict[str, Any],
    _dataset_producer_evidence: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Return a private storage copy of the session-built package for one test."""
    template = _transient_package_template
    storage = tmp_path / "scientific-storage"
    # A copied package is admitted by its recorded hashes, as after a storage restore.
    shutil.copytree(template["root"], storage, symlinks=True)
    monkeypatch.setenv("PROJECT_ROOT", str(template["project_root"]))
    monkeypatch.setenv("FAKE_COMSOL_TRANSIENT_TIME_MODE", "short_rollout")
    return {
        **copy.deepcopy({key: value for key, value in template.items() if key != "expected_item"}),
        "root": storage,
        "source_root": storage / template["source_root"].relative_to(template["root"]),
        "expected_item": _clone_item(template["expected_item"]),
    }
