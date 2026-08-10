# ruff: noqa: S101, PLR2004
"""Steady-flow fake-COMSOL, canonical HDF5, and package regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import pytest
import torch

from src import datasets, domain, generation


def _dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required test HDF5 dataset."""
    value = handle.get(name)
    assert isinstance(value, h5py.Dataset)
    return value


def test_steady_flow_publishes_hdf5_and_immutable_technical_package(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Protect steady HDF5 publication and immutable technical-package reuse."""
    config_path, template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=2,
        retain_raw_csv=True,
    )
    campaign = generation.config.load_campaign_config(config_path)
    batch = campaign.batch("steady_flow__lentil__natural")
    storage = tmp_path / "storage"
    template_bytes = template.read_bytes()
    outcomes = [
        generation.runtime.run_case(
            batch,
            case_index,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / "work",
        )
        for case_index in batch.case_indices
    ]
    assert [outcome.status for outcome in outcomes] == ["completed", "completed"]
    assert template.read_bytes() == template_bytes

    completed = outcomes[0].processed_directory
    assert sorted(path.name for path in completed.iterdir()) == [
        "_SUCCESS",
        "case.h5",
        "case.json",
        "execution_provenance.json",
        "provenance.json",
        "solver.log",
        "status.json",
        "timing.json",
    ]
    raw = generation.runtime.raw_case_directory(batch, 1, storage_root=storage)
    assert sorted(path.relative_to(raw).as_posix() for path in raw.rglob("*.csv")) == [
        "raw_csv/exports/airflow.csv",
        "raw_csv/inputs/fields.csv",
    ]
    identity = generation.storage.validate_case_hdf5(completed / "case.h5", expected_profile="steady_flow")
    assert identity["git_commit"] == "a" * 40
    with h5py.File(completed / "case.h5", "r") as handle:
        assert set(handle) == {
            "coords",
            "provenance",
            "stationary_fixed",
            "static",
        }
        assert _dataset(handle, "coords/x").shape == (401,)
        assert _dataset(handle, "coords/x").attrs["unit"] == "m"
        assert _dataset(handle, "coords/y").shape == (251,)
        assert _dataset(handle, "coords/y").attrs["unit"] == "m"
        static = _dataset(handle, "static/fields")
        assert static.shape == (8, 251, 401)
        assert static.dtype.name == "float32"
        assert static.compression == "gzip"
        assert static.compression_opts == 4
        assert static.shuffle
        fixed = _dataset(handle, "stationary_fixed/values")
        assert fixed.shape == (3,)
        assert fixed.attrs["runtime_source"] == "canonical_template"
    case = json.loads((completed / "case.json").read_text(encoding="utf-8"))
    assert identity["case_input_id"] == case["case_input_id"]
    assert identity["simulation_case_id"] == case["simulation_case_id"]
    assert (
        generation.runtime.run_case(
            batch,
            1,
            cores_per_case=1,
            storage_root=storage,
            work_root=tmp_path / "resume-work",
        ).status
        == "skipped"
    )

    generation.runtime.finalize_batch(batch, storage_root=storage)
    terminal = generation.runtime.validate_terminal_batch(batch, storage_root=storage)
    assert terminal["git_commit"] == "a" * 40
    assert [record["case_id"] for record in terminal["cases"]] == ["case_0001", "case_0002"]

    result = datasets.packages.build_dataset_package(campaign, "steady_flow", "id", storage_root=storage)
    assert result["dataset_name"] == "steady_flow__lentil__id"
    assert result["sample_count"] == 2
    payload = torch.load(result["payload_path"], map_location="cpu", weights_only=False)
    task = domain.tasks.registry.get_task("steady_flow")
    dataset_identity = datasets.identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    assert dataset_identity.sample_count == 2
    assert payload["inputs"].shape == (2, task.in_channels, 251, 401)
    assert payload["outputs"].shape == (2, task.out_channels, 251, 401)
    assert payload["fields"] == {"inputs": list(task.input_names), "outputs": list(task.output_names)}
    manifest = datasets.packages.load_package_manifest(
        result["dataset_id"],
        storage_root=storage,
    )
    assert manifest["training_eligible"] is False
    assert manifest["split_membership"] == {datasets.views.TECHNICAL_SMOKE_MEMBERSHIP: manifest["included_source_cases"]}
    assert manifest["source_git_commits"] == ["a" * 40]
    conditioning = manifest["steady_flow_conditioning"]
    assert conditioning["hidden_conditioning"] is False
    assert conditioning["T_flow_ref_owner"] == "package_fixed"
    assert conditioning["package_fixed_physics"] == [
        {"name": "T_flow_ref", "unit": "K", "value": 300.65},
        {"name": "p_ref", "unit": "Pa", "value": 101325.0},
        {"name": "p_out", "unit": "Pa", "value": 0.0},
    ]

    inspection = datasets.packages.inspect_dataset_package(result["dataset_id"], storage_root=storage)
    assert inspection["dataset_view"] == "steady_flow"
    assert inspection["available_selectors"] == [datasets.views.TECHNICAL_SMOKE_MEMBERSHIP]
    assert inspection["tensors"]["input"]["shape"] == [task.in_channels, 251, 401]
    assert inspection["tensors"]["target"]["shape"] == [task.out_channels, 251, 401]
    assert inspection["sample_identity"]["source_hdf5_sha256"] == manifest["source_case_identities"][0]["case_hdf5_sha256"]
    smoke = datasets.packages.smoke_dataset_package(
        result["dataset_id"],
        storage_root=storage,
        num_workers=0,
    )
    assert smoke["status"] == "loaded"
    assert smoke["batch_shapes"]["x"] == [1, task.in_channels, 251, 401]

    reused = datasets.packages.build_dataset_package(campaign, "steady_flow", "id", storage_root=storage)
    assert reused["status"] == "reused"
    assert reused["dataset_id"] == result["dataset_id"]

    manifest_path = Path(result["manifest_path"])
    conflicting_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conflicting_manifest["dataset_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(conflicting_manifest), encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicts"):
        datasets.packages.build_dataset_package(
            campaign,
            "steady_flow",
            "id",
            storage_root=storage,
        )
