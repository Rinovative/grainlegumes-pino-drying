# ruff: noqa: S101, PLR2004, SLF001
"""Dataset-bound transient PT shard publication and runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pytest
import torch
import yaml

from src import common, datasets, generation
from tests.generation.test_generation_transient import (
    _small_scientific_contract,
    _source,
    _write_transient_case,
)


def _fixture_context(
    root: Path,
    *,
    dataset_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build two compact canonical GPU cases and one transient index."""
    scientific = _small_scientific_contract()
    scientific["time"].update(
        {
            "stop": 4.0,
            "interval": 1.0,
            "regular_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    processed = root / "01_generation/processed/synthetic_batch"
    first = processed / "case_0001/case.h5"
    second = processed / "case_0002/case.h5"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    _write_transient_case(
        first,
        exact_stop_time=3.5,
        scientific_contract=scientific,
        regular_state_count=4,
    )
    _write_transient_case(
        second,
        exact_stop_time=3.5,
        case_input_id="3" * 64,
        simulation_case_id="4" * 64,
        scientific_contract=scientific,
        regular_state_count=4,
    )
    package = root / "02_datasets/packages" / dataset_id
    package.mkdir(parents=True)
    index_path = package / f"{dataset_id}.json"
    index = datasets.packages.trajectory.build_transient_index(
        [
            _source(first),
            _source(
                second,
                package_case_id="synthetic_transient__case_0002",
                case_input_id="3" * 64,
                simulation_case_id="4" * 64,
            ),
        ],
        index_path,
        dataset_name="transient_drying__lentil__id",
        dataset_id=dataset_id,
        evaluation_regime="id",
        source_root=root,
    )
    payload_sha256 = common.serialization.file_sha256(index_path)
    manifest = {
        "dataset_id": dataset_id,
        "dataset_name": index["dataset_name"],
        "dataset_view": "transient_drying",
        "dataset_digest": "d" * 64,
        "payload_filename": index_path.name,
        "payload_sha256": payload_sha256,
        "sample_count": index["sample_count"],
        "source_case_count": index["source_case_count"],
        "channel_contract_digest": index["contract_digest"],
    }
    return manifest, index


def _publication_identity() -> dict[str, str]:
    """Return stable synthetic GPU publication evidence."""
    return {
        "campaign_run_id": "synthetic_run",
        "campaign_id": "synthetic_campaign",
        "git_commit": "1" * 40,
        "campaign_terminal_sha256": "2" * 64,
        "transfer_inventory_sha256": "3" * 64,
    }


def test_transient_shards_accept_distinct_composite_publication_identity() -> None:
    """Keep replacement-completion shard provenance separate from terminal campaigns."""
    identity = {
        "completion_id": "completion__synthetic",
        "parent_run_id": "synthetic_run",
        "parent_partial_sha256": "1" * 64,
        "completion_receipt_sha256": "2" * 64,
        "combined_inventory_sha256": "3" * 64,
    }
    admitted = datasets.packages.transient_shards._validate_publication_identity(identity)
    assert admitted == identity


def _assert_item_equal(left: Any, right: Any) -> None:
    """Require exact equality through one nested TransientItem tree."""
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_item_equal(left[key], right[key])
    else:
        assert left == right


def test_transient_shard_default_is_materialization_owned_and_identity_neutral(
    generation_config_factory: Any,
) -> None:
    """Resolve one 1.5 GiB default without changing package or campaign identity."""
    target_bytes = int(1.5 * 1024**3)
    assert datasets.packages.DEFAULT_TRANSIENT_PT_SHARD_BYTES == target_bytes == 1_610_612_736
    case_sizes = (target_bytes // 2, target_bytes // 2, target_bytes + 1)
    default_groups = datasets.packages.transient_shards._plan_case_groups(
        case_sizes,
        target_shard_bytes=datasets.packages.DEFAULT_TRANSIENT_PT_SHARD_BYTES,
    )
    explicit_groups = datasets.packages.transient_shards._plan_case_groups(
        case_sizes,
        target_shard_bytes=target_bytes,
    )
    assert default_groups == explicit_groups == ((0, 1), (2,))

    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        campaign_purpose="family_generalization",
        natural_count=3,
    )
    authored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    authored["dataset_packages"] = [
        {
            "dataset_view": "transient_drying",
            "evaluation_regime": "id",
            "source_role": "seen",
            "training_payload": {
                "backend": "pt_shards",
                "required": True,
                "target_shard_bytes": target_bytes,
            },
        }
    ]
    config_path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
    explicit = generation.cases.config.load_campaign_config(config_path)

    del authored["dataset_packages"][0]["training_payload"]["target_shard_bytes"]
    config_path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
    defaulted = generation.cases.config.load_campaign_config(config_path)

    assert defaulted.dataset_packages == explicit.dataset_packages
    assert defaulted.package_request_digest == explicit.package_request_digest
    assert defaulted.campaign_id == explicit.campaign_id
    assert defaulted.campaign_digest == explicit.campaign_digest
    assert [batch.batch_id for batch in defaulted.batches] == [batch.batch_id for batch in explicit.batches]
    assert defaulted.dataset_packages[0]["training_payload"]["target_shard_bytes"] == target_bytes


def test_transient_shards_pack_whole_cases_and_match_both_sampling_modes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Preserve exact items while packing multiple whole cases into one shard."""
    dataset_id = "transient_drying__lentil__id__shard_fixture"
    manifest, index = _fixture_context(tmp_path, dataset_id=dataset_id)
    shard_service = datasets.packages.transient_shards
    context_depths: list[str] = []

    def package_context(
        _dataset_id: str,
        *,
        storage_root: Path,
        validation_depth: str,
    ) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
        context_depths.append(validation_depth)
        return Path(storage_root), manifest, index, "4" * 64

    monkeypatch.setattr(
        shard_service,
        "_package_context",
        package_context,
    )
    built = shard_service.build_transient_shards(
        dataset_id,
        storage_root=tmp_path,
        publication_identity=_publication_identity(),
        target_shard_bytes=1_000_000,
    )
    receipt = built["receipt"]
    assert built["status"] == "complete"
    assert receipt["schema_version"] == shard_service.TRANSIENT_PT_RECEIPT_SCHEMA_VERSION == 1
    assert receipt["shard_count"] == 1
    assert receipt["shards"][0]["case_ids"] == [
        "synthetic_transient__case_0001",
        "synthetic_transient__case_0002",
    ]
    assert receipt["shards"][0]["oversized_single_case"] is False
    assert index["dataset_id"] == dataset_id
    assert context_depths == ["full"]

    index_path = tmp_path / "02_datasets/packages" / dataset_id / f"{dataset_id}.json"
    one_step = datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
    rollout = datasets.contracts.transient.TransientSamplingSpec(
        mode="rollout_window",
        rollout_length=2,
        window_stride=1,
        window_offset=0,
    )
    for sampling, positions in ((one_step, (3,)), (rollout, (0, 1))):
        canonical = datasets.runtime.transient.TransientPhysicalDataset(
            index_path,
            sampling=sampling,
            source_root=tmp_path,
            sample_indices=positions,
        )
        sharded = datasets.runtime.transient.TransientPTShardDataset(
            index_path,
            sampling=sampling,
            source_root=tmp_path,
            hdf5_cache_size=1,
            sample_indices=positions,
        )
        try:
            _assert_item_equal(canonical[0], sharded[0])
            assert sharded.storage_backend == "pt_shards"
            assert len(sharded._shard_cache) == 1
        finally:
            canonical.close()
            sharded.close()

    context_depths.clear()
    reused = shard_service.build_transient_shards(
        dataset_id,
        storage_root=tmp_path,
        publication_identity=_publication_identity(),
        target_shard_bytes=1_000_000,
    )
    assert reused["status"] == "reused"
    assert reused["receipt"] == receipt
    assert context_depths == ["evidence"]

    conflict = {**_publication_identity(), "campaign_terminal_sha256": "9" * 64}
    receipt_path = Path(reused["receipt_path"])
    before_conflict = {path.name: path.read_bytes() for path in receipt_path.parent.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError, match="publication identity conflicts"):
        shard_service.build_transient_shards(
            dataset_id,
            storage_root=tmp_path,
            publication_identity=conflict,
            target_shard_bytes=1_000_000,
            rebuild_invalid=True,
        )
    assert {path.name: path.read_bytes() for path in receipt_path.parent.iterdir() if path.is_file()} == before_conflict

    index_path.touch()
    context_depths.clear()
    state_root = common.paths.get_dataset_state_root(storage_root=tmp_path)
    orphan_staging = state_root / f".{dataset_id}.transient-pt.interrupted.tmp"
    orphan_staging.mkdir()
    orphan_backup = state_root / f".{dataset_id}.transient-pt.invalid-interrupted.backup"
    receipt_path.parent.replace(orphan_backup)
    rebound = shard_service.build_transient_shards(
        dataset_id,
        storage_root=tmp_path,
        publication_identity=_publication_identity(),
        target_shard_bytes=1_000_000,
        rebuild_invalid=True,
    )
    assert rebound["status"] == "complete"
    assert context_depths == ["full"]
    assert not orphan_staging.exists()
    assert not orphan_backup.exists()
    receipt = rebound["receipt"]

    shard_path = Path(rebound["receipt_path"]).parent / str(receipt["shards"][0]["filename"])
    changed = bytearray(shard_path.read_bytes())
    changed[len(changed) // 2] ^= 1
    shard_path.write_bytes(changed)
    assert shard_path.stat().st_size == receipt["shards"][0]["size_bytes"]
    rebuilt = shard_service.build_transient_shards(
        dataset_id,
        storage_root=tmp_path,
        publication_identity=_publication_identity(),
        target_shard_bytes=1_000_000,
        existing_validation_depth="evidence",
        rebuild_invalid=True,
    )
    assert rebuilt["status"] == "rebuilt"
    assert (
        shard_service.load_transient_shard_receipt(
            dataset_id,
            storage_root=tmp_path,
            publication_identity=_publication_identity(),
            validation_depth="full",
        )
        == rebuilt["receipt"]
    )

    def reject_hdf5(*_args: Any, **_kwargs: Any) -> Any:
        message = "Unchanged shard smoke reopened canonical HDF5."
        raise AssertionError(message)

    shard_hashes: list[Path] = []
    original_sha256 = common.serialization.file_sha256

    def count_shard_hashes(path: Path | str) -> str:
        candidate = Path(path)
        if candidate.suffix == ".pt":
            shard_hashes.append(candidate)
        return original_sha256(candidate)

    monkeypatch.setattr(h5py, "File", reject_hdf5)
    monkeypatch.setattr(common.serialization, "file_sha256", count_shard_hashes)
    smoke = generation.workflow._smoke_transient_shard_backend(
        {
            "dataset_id": dataset_id,
            "payload_relative_path": index_path.relative_to(tmp_path).as_posix(),
        },
        storage_root=tmp_path,
        compare_canonical=False,
    )
    assert set(smoke) == {"one_step_transition", "rollout_window"}
    assert all(result["status"] == "equivalent" for result in smoke.values())
    assert shard_hashes == []

    for case in index["cases"]:
        (tmp_path / str(case["source_relative_path"])).unlink()
    for sampling, positions in ((one_step, (3,)), (rollout, (0, 1))):
        source_free = datasets.runtime.transient.TransientPTShardDataset(
            index_path,
            sampling=sampling,
            source_root=tmp_path,
            hdf5_cache_size=1,
            sample_indices=positions,
        )
        try:
            assert source_free.storage_backend == "pt_shards"
            assert source_free[0]["metadata"]["dataset_id"] == dataset_id
        finally:
            source_free.close()

    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(raw_receipt, dict)
    raw_receipt["publication_identity"] = conflict
    common.serialization.atomic_write_json(receipt_path, raw_receipt)
    corrupt_conflict = receipt_path.read_bytes()
    with pytest.raises(FileExistsError, match="different immutable owner"):
        shard_service.build_transient_shards(
            dataset_id,
            storage_root=tmp_path,
            publication_identity=_publication_identity(),
            target_shard_bytes=1_000_000,
            rebuild_invalid=True,
        )
    assert receipt_path.read_bytes() == corrupt_conflict


def test_transient_shards_keep_oversized_cases_whole_and_evict_cache(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Allow one oversized shard per case and bound the worker-local LRU."""
    dataset_id = "transient_drying__lentil__id__oversized_fixture"
    manifest, index = _fixture_context(tmp_path, dataset_id=dataset_id)
    shard_service = datasets.packages.transient_shards
    monkeypatch.setattr(
        shard_service,
        "_package_context",
        lambda _dataset_id, *, storage_root, **_unused: (
            Path(storage_root),
            manifest,
            index,
            "5" * 64,
        ),
    )
    events: list[tuple[str, int]] = []
    original_case_payload = shard_service._case_payload
    original_torch_save = common.serialization.atomic_torch_save

    def tracked_case_payload(**kwargs: Any) -> dict[str, Any]:
        events.append(("case", int(kwargs["case_index"])))
        return original_case_payload(**kwargs)

    def tracked_torch_save(payload: Any, destination: Path | str) -> Path:
        events.append(("save", int(payload["shard_index"])))
        return original_torch_save(payload, destination)

    monkeypatch.setattr(shard_service, "_case_payload", tracked_case_payload)
    monkeypatch.setattr(common.serialization, "atomic_torch_save", tracked_torch_save)
    built = shard_service.build_transient_shards(
        dataset_id,
        storage_root=tmp_path,
        publication_identity=_publication_identity(),
        target_shard_bytes=1,
    )
    receipt = built["receipt"]
    assert receipt["shard_count"] == 2
    assert all(record["case_count"] == 1 for record in receipt["shards"])
    assert all(record["oversized_single_case"] is True for record in receipt["shards"])
    assert all(record["oversized_reason"] == "complete_case_exceeds_soft_target" for record in receipt["shards"])
    assert events == [
        ("case", 0),
        ("save", 0),
        ("case", 1),
        ("save", 1),
    ]

    index_path = tmp_path / "02_datasets/packages" / dataset_id / f"{dataset_id}.json"
    runtime = datasets.runtime.transient.TransientPTShardDataset(
        index_path,
        sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        source_root=tmp_path,
        hdf5_cache_size=1,
    )
    try:
        first_case_last_position = int(index["cases"][0]["transition_count"]) - 1
        runtime[0]
        assert tuple(runtime._shard_cache) == (0,)
        runtime[first_case_last_position + 1]
        assert tuple(runtime._shard_cache) == (1,)
    finally:
        runtime.close()
