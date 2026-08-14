# ruff: noqa: S101
"""Dual-profile fake-runtime publication and loader contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import h5py
import pytest
import yaml

from src import datasets, generation
from src.generation.contracts import generation_contracts_porosity as porosity_service

pytestmark = pytest.mark.integration

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

    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
        simulation_profile=simulation_profile,
    )
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
            porosity = realized["spatial_diagnostics"]["porosity"]
            natural_support = porosity["natural_porosity_support"]
            retry = realized["spatial_diagnostics"]["complete_case_support_retry"]
            assert "packing_scatter_z" not in realized["sampled_values"]
            assert porosity["packing_scatter_support_kind"] == "natural"
            assert porosity["packing_scatter_truncation_lower"] < porosity["packing_scatter_z"] < porosity["packing_scatter_truncation_upper"]
            assert porosity["packing_scatter_seed"] == realized["seed_evidence"]["subseeds"]["packing_scatter"]
            assert retry["accepted_attempt_seeds"]["packing_scatter"] == porosity["packing_scatter_seed"]
            assert porosity["packing_scatter_support_lower"] <= porosity["eps_reference"] <= porosity["packing_scatter_support_upper"]
            assert natural_support["lower"] <= porosity["eps_bed_mean"] <= natural_support["upper"]
            assert porosity["sampled_kappa_mean"] == realized["sampled_values"]["kappa_mean"]
            assert porosity["sampled_kappa_mean"] == pytest.approx(
                porosity["A_KC_reference"] * porosity_service.kozeny_carman_response(porosity["eps_kc_trend"])
            )
            assert porosity["texture_source"] == "z_background"
    generation.runtime.finalize_batch(batch, storage_root=storage)
    generation.runtime.validate_terminal_batch(batch, storage_root=storage)
    bounded = datasets.packages.generated_batch.load_generated_batch(
        batch.batch_id,
        storage_root=storage,
        max_cases=1,
    )
    assert bounded["sample_ids"] == ["case_0001"]
    assert bounded["available_case_count"] == len(batch.case_indices)
    assert len(bounded["rows"]) == 1

    results = datasets.packages.build_campaign_packages(campaign, storage_root=storage)
    assert tuple(result["dataset_view"] for result in results) == expected_views
    for result in results:
        manifest = datasets.packages.load_package_manifest(result["dataset_id"], storage_root=storage)
        assert manifest["campaign_purpose"] == "technical_runtime_smoke"
        assert manifest["evaluation_regime"] == "id"
        assert manifest["training_eligible"] is False
        assert manifest["split_membership"] == {datasets.contracts.views.TECHNICAL_SMOKE_MEMBERSHIP: manifest["included_source_cases"]}
        with pytest.raises(ValueError, match="allow_technical_smoke"):
            datasets.runtime.factory.create_dataset(
                datasets.runtime.factory.DatasetRequest(
                    dataset_id=result["dataset_id"],
                    dataset_view=result["dataset_view"],
                    evaluation_regime="id",
                    transient_sampling=(
                        datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")
                        if result["dataset_view"] == "transient_drying"
                        else None
                    ),
                    storage_root=storage,
                )
            )
        inspection = datasets.packages.inspect_dataset_package(result["dataset_id"], storage_root=storage)
        assert inspection["available_selectors"] == [datasets.contracts.views.TECHNICAL_SMOKE_MEMBERSHIP]
        smoke = datasets.packages.smoke_dataset_package(
            result["dataset_id"],
            storage_root=storage,
            num_workers=0,
        )
        assert smoke["status"] == "loaded"
        assert smoke["num_workers"] == 0
