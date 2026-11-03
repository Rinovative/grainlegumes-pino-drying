"""Exercise immutable campaign evidence across an explicit read storage mount.

Responsibilities:
  - Admit relocated frozen snapshots and campaign metadata without rewriting their origin locators
  - Reject missing, changed, unsafe, traversing, and symlink-escaped snapshot evidence
  - Preserve strict executable-template checks and raw-case content and decision identity

Design principles:
  - An explicit read root relocates evidence lookup while immutable identities and path containment remain enforced
  - Read admission does not make frozen execution inputs executable or weaken template validation

This module does NOT:
  - Submit Slurm work or run licensed production COMSOL simulations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import common, generation
from src.generation import generation_campaign as runtime
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_campaign_evidence as evidence
from src.generation.publication import generation_publication_catalog as catalog


@pytest.fixture
def relocated_run(generation_config_factory: Any, tmp_path: Path) -> dict[str, Any]:
    """Keep frozen origin locators while publishing tiny evidence under a read root."""
    config, template = generation_config_factory(simulation_profile="steady_flow", scheduler_kind="slurm", natural_count=1)
    campaign = generation.cases.config.load_campaign_config(config)
    storage = tmp_path / "read-storage"
    origin = tmp_path / "unmounted-origin"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    run_id = runtime.campaign_run_id(campaign, git_commit="a" * 40)
    manifest = runtime._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=root,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={
            campaign.batches[0].batch_name: {"input_generation_id": "input-" + "a" * 24, "source_git_commit": "a" * 40, "case_indices": [1]}
        },
    )
    relative = Path("source_snapshots") / ("b" * 12 + "-" + "b" * 12) / "snapshot.json"
    snapshot = storage / relative
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('{"frozen":"source"}\n', encoding="utf-8")
    manifest["remote_storage_root"] = str(origin)
    manifest["source_snapshot"] = {
        "path": str(origin / relative),
        "sha256": common.serialization.file_sha256(snapshot),
        "source_fingerprint_sha256": "b" * 64,
    }
    path = common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(root))
    common.serialization.atomic_write_json(path, manifest)
    return {
        "manifest": manifest,
        "path": path,
        "run_id": run_id,
        "storage": storage,
        "snapshot": snapshot,
        "origin": origin,
        "template": template,
        "campaign": campaign,
    }


def test_relocated_snapshot_and_frozen_metadata_are_admitted_without_rewriting(relocated_run: dict[str, Any]) -> None:
    """Read the same snapshot bytes and plan without origin mount or current template."""
    row = relocated_run
    before = row["path"].read_bytes()
    loaded = evidence.load_campaign_run(row["run_id"], storage_root=row["storage"])
    row["template"].unlink()
    campaign = lifecycle.campaign_from_frozen_run(loaded, require_executable=False, storage_root=row["storage"])
    assert campaign.campaign_digest == row["campaign"].campaign_digest
    assert campaign.batches[0].batch_id == row["campaign"].batches[0].batch_id
    assert loaded["remote_storage_root"] == str(row["origin"])
    assert row["path"].read_bytes() == before
    with pytest.raises((FileNotFoundError, ValueError)):
        lifecycle.campaign_from_frozen_run(loaded)
    with pytest.raises((FileNotFoundError, ValueError)):
        lifecycle.campaign_from_frozen_run(loaded, storage_root=row["storage"])


@pytest.mark.parametrize(
    "mutation",
    ["outside", "traversal", "relative", "root_relative", "root_traversal", "root_filesystem", "changed", "missing", "symlink", "parent_escape"],
)
def test_relocated_snapshot_rejects_unsafe_or_changed_evidence(relocated_run: dict[str, Any], mutation: str, tmp_path: Path) -> None:
    """Mount relocation never admits arbitrary paths, corrupt bytes or symlink escape."""
    row = relocated_run
    manifest = row["manifest"]
    if mutation == "outside":
        manifest["source_snapshot"]["path"] = str(tmp_path / "outside/snapshot.json")
    elif mutation == "traversal":
        manifest["source_snapshot"]["path"] = str(row["origin"] / "source_snapshots/../outside.json")
    elif mutation == "relative":
        manifest["source_snapshot"]["path"] = "source_snapshots/snapshot.json"
    elif mutation == "root_relative":
        manifest["remote_storage_root"] = "relative-storage"
    elif mutation == "root_traversal":
        manifest["remote_storage_root"] = str(row["origin"] / "../other")
    elif mutation == "root_filesystem":
        manifest["remote_storage_root"] = "/"
    elif mutation == "changed":
        row["snapshot"].write_text("changed", encoding="utf-8")
    elif mutation == "missing":
        row["snapshot"].unlink()
    elif mutation == "symlink":
        target = tmp_path / "same-bytes.json"
        target.write_bytes(row["snapshot"].read_bytes())
        row["snapshot"].unlink()
        row["snapshot"].symlink_to(target)
    else:
        outside = tmp_path / "escape"
        outside.mkdir()
        original = row["snapshot"].parent
        content = row["snapshot"].read_bytes()
        row["snapshot"].unlink()
        original.rmdir()
        (outside / "snapshot.json").write_bytes(content)
        original.symlink_to(outside, target_is_directory=True)
    common.serialization.atomic_write_json(row["path"], manifest)
    with pytest.raises(ValueError, match="snapshot"):
        evidence.load_campaign_run(row["run_id"], storage_root=row["storage"])


def test_execution_template_check_remains_strict_with_explicit_root(relocated_run: dict[str, Any]) -> None:
    """Providing a storage root never disables the default executable template check."""
    row = relocated_run
    row["template"].write_bytes(b"changed executable template")
    with pytest.raises(ValueError, match=r"template|Template|sidecar"):
        lifecycle.campaign_from_frozen_run(row["manifest"], storage_root=row["storage"])


def test_raw_case_reader_admits_relocated_frozen_source_and_checks_exact_identity(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """The public reader preserves case content/decision identity after executable removal."""
    config, template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=1)
    campaign = generation.cases.config.load_campaign_config(config)
    storage = tmp_path / "storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    batch = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage)
    outcome = generation.runtime.run_case(batch, 1, cores_per_case=1, storage_root=storage, work_root=tmp_path / "work")
    assert outcome.status == "completed"
    validated = generation.runtime.validate_completed_case(batch, 1, storage_root=storage, validation_depth="routine")
    artifact = outcome.processed_directory / "case.h5"
    lifecycle.decide_candidate(
        "steady_flow",
        campaign.source_name,
        batch.material_family,
        outcome.case_id,
        outcome="successful",
        stage="publication",
        failure_class=None,
        reason_code=None,
        detail="Tiny fake-boundary immutable source",
        case_input_id=validated["case_input_id"],
        simulation_case_id=validated["simulation_case_id"],
        case_hdf5_relative=artifact.relative_to(storage).as_posix(),
        case_hdf5_sha256=common.serialization.file_sha256(artifact),
        evidence={},
        storage_root=storage,
    )
    run_id = runtime.campaign_run_id(campaign, git_commit="a" * 40)
    manifest = runtime._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=root,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={batch.batch_name: {"input_generation_id": inputs.input_generation_id, "source_git_commit": "a" * 40, "case_indices": [1]}},
    )
    manifest["remote_storage_root"] = str(tmp_path / "unmounted-origin")
    common.serialization.atomic_write_json(
        common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(root)), manifest
    )
    row = catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)[0]
    selector = {
        key: row[key]
        for key in ("campaign_name", "campaign_digest", "material_family", "case_id", "case_input_id", "simulation_case_id", "case_hdf5_sha256")
    }
    template.unlink()
    admitted = catalog.resolve_raw_case_reference("steady_flow", selector, storage_root=storage)
    assert admitted.case.case_id == outcome.case_id
    assert admitted.source_identity["case_hdf5_sha256"] == selector["case_hdf5_sha256"]
    with pytest.raises(ValueError, match="identity"):
        catalog.resolve_raw_case_reference("steady_flow", {**selector, "case_input_id": "f" * 64}, storage_root=storage)
    artifact.write_bytes(b"changed source payload")
    with pytest.raises(ValueError, match=r"hash|HDF5|case"):
        catalog.resolve_raw_case_reference("steady_flow", selector, storage_root=storage)
