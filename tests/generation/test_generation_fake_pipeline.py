# ruff: noqa: S101
"""Dual-profile fake-runtime publication and loader contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import h5py
import pytest
import yaml

from src import datasets, generation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("simulation_profile", "expected_views"),
    [
        ("steady_flow", ("steady_flow",)),
        ("transient_drying", ("steady_flow", "transient_drying")),
    ],
)
def test_technical_fake_runtime_reaches_packages_and_worker_modes(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    simulation_profile: str,
    expected_views: tuple[str, ...],
) -> None:
    """Exercise inputs through fake COMSOL, HDF5, packages, and both loader modes."""
    config_path, _template = generation_config_factory(
        simulation_profile=simulation_profile,
        executable=fake_comsol,
        natural_count=2,
        retain_raw_csv=True,
        retain_solved_model=True,
    )
    authored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    authored["campaign_purpose"] = "technical_runtime_smoke"
    authored["sampling"]["counts"] = {"natural": {"lentil": 2}}
    authored["dataset_packages"] = [{"evaluation_regime": "id", "source_role": "seen"}]
    config_path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")

    campaign = generation.config.load_campaign_config(config_path)
    batch = campaign.batch(f"{simulation_profile}__lentil__natural")
    storage = tmp_path / f"{simulation_profile} storage"
    outcomes = [
        generation.runtime.run_case(
            batch,
            case_index,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / f"{simulation_profile} work",
        )
        for case_index in batch.case_indices
    ]
    assert [outcome.status for outcome in outcomes] == ["completed", "completed"]
    for outcome in outcomes:
        with h5py.File(outcome.processed_directory / "case.h5", "r") as handle:
            assert handle.attrs["material_role"] == "seen"
            assert handle.attrs["evaluation_regime"] == "id"
            assert handle.attrs["natural_support_state"] == "natural"
            provenance_dataset = handle["provenance/case_scientific_provenance_json"]
            assert isinstance(provenance_dataset, h5py.Dataset)
            provenance_payload = provenance_dataset[()]
            if isinstance(provenance_payload, bytes):
                provenance_payload = provenance_payload.decode("utf-8")
            assert isinstance(provenance_payload, str)
            realized = json.loads(provenance_payload)
            assert realized["case_id"] == handle.attrs["case_id"]
            assert realized["case_index"] == handle.attrs["case_index"]
            assert realized["material_family"] == "lentil"
            assert realized["evaluation_regime"] == "id"
            assert set(batch.scientific_values["material"]["active_coordinate_names"]).issubset(realized["sampled_values"])
            assert set(realized["sampled_values"]) == set(realized["sampled_units"])
    generation.runtime.finalize_batch(batch, storage_root=storage)
    generation.runtime.validate_terminal_batch(batch, storage_root=storage)

    results = datasets.packages.build_campaign_packages(campaign, storage_root=storage)
    assert tuple(result["dataset_view"] for result in results) == expected_views
    for result in results:
        manifest = datasets.packages.load_package_manifest(result["dataset_id"], storage_root=storage)
        assert manifest["campaign_purpose"] == "technical_runtime_smoke"
        assert manifest["evaluation_regime"] == "id"
        assert manifest["training_eligible"] is False
        assert manifest["split_membership"] == {datasets.views.TECHNICAL_SMOKE_MEMBERSHIP: manifest["included_source_cases"]}
        with pytest.raises(ValueError, match="allow_technical_smoke"):
            datasets.factory.create_dataset(
                datasets.factory.DatasetRequest(
                    dataset_id=result["dataset_id"],
                    dataset_view=result["dataset_view"],
                    evaluation_regime="id",
                    storage_root=storage,
                )
            )
        inspection = datasets.packages.inspect_dataset_package(result["dataset_id"], storage_root=storage)
        assert inspection["available_selectors"] == [datasets.views.TECHNICAL_SMOKE_MEMBERSHIP]
        for workers in (0, 2):
            smoke = datasets.packages.smoke_dataset_package(
                result["dataset_id"],
                storage_root=storage,
                num_workers=workers,
            )
            assert smoke["status"] == "loaded"
            assert smoke["num_workers"] == workers
