"""Exercise immutable campaign evidence across an explicit read storage mount.

Responsibilities:
  - Admit relocated frozen snapshots and campaign metadata without rewriting their origin locators
  - Reject missing, changed, unsafe, traversing, and symlink-escaped snapshot evidence
  - Preserve strict executable-template checks and raw-case content and decision identity
  - Check that batch raw-case admission and exact Dataset selection read campaign-wide evidence once per Source while
    every case is still hashed and checked, and a repeated or changed case is refused

Design principles:
  - An explicit read root relocates evidence lookup while immutable identities and path containment remain enforced
  - Read admission does not make frozen execution inputs executable or weaken template validation
  - Counting wrappers delegate unchanged, so the counted admission is the real one

This module does NOT:
  - Submit Slurm work or run licensed production COMSOL simulations
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from support.generation_fixtures import successful_source_case

from src import common, generation
from src.datasets.packages import dataset_packages_planning as planning
from src.datasets.packages import dataset_packages_selection as selection
from src.generation import generation_campaign as runtime
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_campaign_evidence as evidence
from src.generation.publication import generation_publication_catalog as catalog
from src.generation.runtime import generation_runtime_batch as runtime_batch

# Campaign-wide reads of raw-case admission; their count must not grow with the number of admitted cases.
_CAMPAIGN_WIDE_READS = (
    (catalog, "list_campaigns"),
    (catalog, "list_campaign_cases"),
    (lifecycle, "load_campaign_plan"),
    (evidence, "load_campaign_run"),
    (lifecycle, "campaign_from_frozen_run"),
    (lifecycle, "candidate_records"),
)
_SELECTOR_KEYS = ("campaign_name", "campaign_digest", "material_family", "case_id", "case_input_id", "simulation_case_id", "case_hdf5_sha256")


@pytest.fixture
def relocated_run(generation_config_factory: Any, tmp_path: Path) -> dict[str, Any]:
    """Keep frozen origin locators while publishing tiny evidence under a read root."""
    config, template = generation_config_factory(simulation_profile="steady_flow", natural_count=1)
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


def test_evidence_reconstruction_ignores_a_changed_source_projection(relocated_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspection reads the recorded producer identity as evidence and never compares it with the running code."""
    row = relocated_run
    loaded = evidence.load_campaign_run(row["run_id"], storage_root=row["storage"])
    recorded = row["campaign"].producer_identity["source_implementation_sha256"]
    monkeypatch.setattr(type(runtime.source_service.GENERATION_SOURCE_PROJECTION), "sha256", lambda _self, _root=None: "0" * 64)

    campaign = lifecycle.campaign_from_frozen_run(loaded, require_executable=False, storage_root=row["storage"])

    assert (campaign.producer_identity or {}).get("source_implementation_sha256") == recorded


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
    outcome = generation.runtime.run_case(batch, 1, cores_per_case=1, storage_root=storage, work_root=tmp_path / "work", scheduler_kind="local")
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


@pytest.fixture
def three_case_source(generation_config_factory: Any, fake_comsol: Path, tmp_path: Path) -> dict[str, Any]:
    """Publish one frozen fake-solver Source of three successful cases with its campaign run manifest."""
    config, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=3)
    campaign = generation.cases.config.load_campaign_config(config)
    storage = tmp_path / "storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    batch = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(batch, len(batch.case_indices), storage_root=storage)
    for case_index in batch.case_indices:
        successful_source_case(campaign.source_name, batch, case_index, storage, tmp_path / "work")
    manifest = runtime._new_campaign_manifest(
        campaign,
        run_id=runtime.campaign_run_id(campaign, git_commit="a" * 40),
        requested_commit="a" * 40,
        run_directory=root,
        scheduler_log_directory=tmp_path / "logs",
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
        common.paths.resolve_generation_run_manifest_path(common.paths.resolve_generation_execution_directory(root)), manifest
    )
    rows = catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)
    assert len(rows) == 3
    return {"storage": storage, "source_name": campaign.source_name, "rows": rows}


def _counted(counts: Counter[str], name: str, original: Any) -> Any:
    """Return a wrapper that counts each call under ``name`` and delegates it unchanged."""

    def counted(*args: Any, **kwargs: Any) -> Any:
        counts[name] += 1
        return original(*args, **kwargs)

    return counted


def _count_calls(monkeypatch: pytest.MonkeyPatch, targets: tuple[tuple[Any, str], ...]) -> Counter[str]:
    """Wrap each named module function so that its calls are counted and then delegated unchanged."""
    counts: Counter[str] = Counter()
    for module, name in targets:
        monkeypatch.setattr(module, name, _counted(counts, name, getattr(module, name)))
    return counts


def test_raw_case_batch_reads_campaign_evidence_once_and_checks_each_case(three_case_source: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Three cases of one Source cost the campaign-wide reads of one case, while each case is still hashed and checked."""
    storage = three_case_source["storage"]
    selectors = [{key: row[key] for key in _SELECTOR_KEYS} for row in three_case_source["rows"]]
    counts = _count_calls(monkeypatch, _CAMPAIGN_WIDE_READS)
    hashed: list[Path] = []
    file_sha256 = common.serialization.file_sha256

    def recorded_sha256(path: Path) -> str:
        hashed.append(Path(path))
        return file_sha256(path)

    monkeypatch.setattr(common.serialization, "file_sha256", recorded_sha256)

    single = catalog.resolve_raw_case_reference("steady_flow", selectors[0], storage_root=storage)
    single_counts = dict(counts)
    counts.clear()
    admitted = catalog.resolve_raw_case_references("steady_flow", selectors, storage_root=storage)

    assert dict(counts) == single_counts
    assert all(count >= 1 for count in single_counts.values())
    assert len(single_counts) == len(_CAMPAIGN_WIDE_READS)
    assert [case.source_identity["case_id"] for case in admitted] == [selector["case_id"] for selector in selectors]
    assert admitted[0].source_identity == single.source_identity
    assert all(case.hdf5_path in hashed for case in admitted)
    with pytest.raises(ValueError, match="repeat"):
        catalog.resolve_raw_case_references("steady_flow", [selectors[0], selectors[1], selectors[0]], storage_root=storage)
    admitted[1].hdf5_path.write_bytes(b"changed source payload")
    with pytest.raises(ValueError, match="HDF5 hash disagrees"):
        catalog.resolve_raw_case_references("steady_flow", selectors, storage_root=storage)


@pytest.mark.parametrize("record_file", ["reservation", "decision"])
def test_raw_case_batch_refuses_a_later_record_file_turned_into_a_symlink(
    three_case_source: dict[str, Any], monkeypatch: pytest.MonkeyPatch, record_file: str
) -> None:
    """A later case's reservation or decision replaced by a symlink after the Source evidence was read is still refused when hashed."""
    storage = three_case_source["storage"]
    selectors = [{key: row[key] for key in _SELECTOR_KEYS} for row in three_case_source["rows"]]
    records = lifecycle.candidate_records("steady_flow", three_case_source["source_name"], storage_root=storage)
    second = next(record for record in records if record["reservation"]["case_id"] == selectors[1]["case_id"])
    resolvers = {"reservation": common.paths.resolve_generation_case_reservation_path, "decision": common.paths.resolve_generation_case_decision_path}
    target = resolvers[record_file](second["case_directory"])
    admit = runtime_batch.admit_completed_case

    def admit_then_swap(*args: Any, **kwargs: Any) -> Any:
        admitted = admit(*args, **kwargs)
        if not target.is_symlink():
            copy = target.with_name(f"{record_file}_copy.json")
            copy.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(copy)
        return admitted

    monkeypatch.setattr(runtime_batch, "admit_completed_case", admit_then_swap)
    with pytest.raises(ValueError, match="symlink"):
        catalog.resolve_raw_case_references("steady_flow", selectors, storage_root=storage)


def test_raw_case_selector_error_precedes_storage_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed selector is refused before the storage root is resolved, as for a single reference."""

    def unresolved(*_args: Any, **_kwargs: Any) -> Path:
        msg = "storage root unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(common.paths, "get_storage_root", unresolved)
    with pytest.raises(ValueError, match="seven source identity fields"):
        catalog.resolve_raw_case_references("steady_flow", [{}])


def test_exact_selection_lists_each_campaign_once(three_case_source: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Three exact picks from one Source list its cases once and keep pick order."""
    rows = three_case_source["rows"]
    picks = [{"source_name": three_case_source["source_name"], "material_family": row["material_family"], "case_id": row["case_id"]} for row in rows]
    counts = _count_calls(monkeypatch, ((catalog, "list_campaign_cases"),))

    selected = selection.select_exact_cases("steady_flow", picks, storage_root=three_case_source["storage"])

    assert counts == {"list_campaign_cases": 1}
    assert [reference["case_id"] for reference in selected] == [row["case_id"] for row in rows]


def test_dataset_planner_admits_an_exact_selection_in_one_batch(three_case_source: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The Dataset planner admits every case of a selection through one batch call, never case by case."""
    storage = three_case_source["storage"]
    picks = [
        {"source_name": three_case_source["source_name"], "material_family": row["material_family"], "case_id": row["case_id"]}
        for row in three_case_source["rows"]
    ]
    request = {
        "schema_kind": selection.SELECTION_SCHEMA_KIND,
        "source_task": "steady_flow",
        "dataset_name": "planner_batch_probe",
        "dataset_view": "steady_flow",
        "selection": {"kind": "exact"},
        "source_cases": [dict(reference) for reference in selection.select_exact_cases("steady_flow", picks, storage_root=storage)],
    }
    counts = _count_calls(monkeypatch, ((catalog, "resolve_raw_case_references"), (catalog, "resolve_raw_case_reference")))

    prepared = planning.prepare_dataset_selection(request, storage_root=storage)

    assert counts == {"resolve_raw_case_references": 1}
    assert len(prepared.cases) == len(picks)
