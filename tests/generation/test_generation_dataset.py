# ruff: noqa: S101, PLR2004
"""Steady-flow fake-COMSOL, canonical HDF5, and package regression."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch
import yaml

from src import common, datasets, domain, generation
from src.datasets.packages import dataset_packages_builder as package_builder

pytestmark = pytest.mark.integration


def test_package_publication_recovers_owned_partial_and_corrupt_state_without_overwriting_conflicts(
    tmp_path: Path,
) -> None:
    """Recover an expected package atomically while preserving conflicting identity bytes."""
    publish_package = package_builder._publish  # noqa: SLF001 -- lowest publication contract owner
    schema_identity = package_builder._schema_identity("steady_flow")  # noqa: SLF001 -- test-owned provenance
    provenance = dict.fromkeys(package_builder.package_manifest.PACKAGE_PROVENANCE_KEYS)
    provenance.update(
        {
            "schema_kind": package_builder.DATASET_PACKAGE_SCHEMA_KIND,
            "schema_version": package_builder.DATASET_PACKAGE_SCHEMA_VERSION,
            "dataset_name": "synthetic-package",
            "dataset_view": "steady_flow",
            "builder_identity": package_builder.identity.dataset_conversion_contract_identity("steady_flow"),
            "schema_identity": schema_identity,
            "source_case_identities": [
                {
                    "package_case_id": "case-1",
                    "batch_id": "batch-1",
                    "source_case_id": "case-1",
                    "case_input_id": "input-1",
                    "simulation_case_id": "simulation-1",
                    "case_hdf5_sha256": "a" * 64,
                    "material_family": "lentil",
                    "material_role": "seen",
                    "evaluation_regime": "id",
                    "natural_support_state": "in_support",
                    "simulation_profile": "steady_flow",
                    "membership": "train",
                    "ood_group": None,
                    "ood_parameters": [],
                    "task_relevant_ood_parameters": [],
                }
            ],
        }
    )
    dataset_id, dataset_digest = package_builder.identity.package_identity_from_provenance(provenance)
    payload_filename = f"{dataset_id}.pt"
    prefix = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "payload_filename": payload_filename,
        "source_case_count": 1,
    }
    payload_bytes = b"synthetic immutable payload"
    build_count = 0

    def build_payload(path: Path) -> dict[str, int]:
        nonlocal build_count
        build_count += 1
        path.write_bytes(payload_bytes)
        return {"sample_count": 1, "transition_count": 0}

    first = publish_package(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest_prefix=prefix,
        build_payload=build_payload,
        storage_root=tmp_path,
    )
    assert first[2] is False
    assert build_count == 1
    reused = publish_package(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest_prefix=prefix,
        build_payload=build_payload,
        storage_root=tmp_path,
    )
    assert reused[2] is True
    assert build_count == 1

    metadata_dir = Path(first[1]).parent
    shutil.rmtree(metadata_dir)
    recovered_partial = publish_package(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest_prefix=prefix,
        build_payload=build_payload,
        storage_root=tmp_path,
    )
    assert recovered_partial[2] is False
    assert build_count == 2
    assert package_builder.package_manifest.load_package_manifest(dataset_id, storage_root=tmp_path)["dataset_id"] == dataset_id

    payload_path = Path(recovered_partial[0])
    payload_path.write_bytes(b"corrupt same-identity payload")
    recovered_corrupt = publish_package(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest_prefix=prefix,
        build_payload=build_payload,
        storage_root=tmp_path,
    )
    assert recovered_corrupt[2] is False
    assert build_count == 3
    assert payload_path.read_bytes() == payload_bytes

    state_root = common.paths.get_dataset_state_root(storage_root=tmp_path)
    orphan_staging = state_root / f".{dataset_id}.interrupted.tmp"
    orphan_staging.mkdir()
    payload_path.write_bytes(b"corrupt before interrupted replacement")
    interrupted_backup = state_root / f".{dataset_id}.interrupted.payload.backup"
    payload_path.parent.replace(interrupted_backup)
    recovered_interrupted = publish_package(
        dataset_id=dataset_id,
        payload_filename=payload_filename,
        manifest_prefix=prefix,
        build_payload=build_payload,
        storage_root=tmp_path,
    )
    assert recovered_interrupted[2] is False
    assert build_count == 4
    assert Path(recovered_interrupted[0]).read_bytes() == payload_bytes
    assert not interrupted_backup.exists()
    assert not orphan_staging.exists()
    assert not tuple(state_root.glob(f".{dataset_id}.*.backup"))
    assert not tuple(state_root.glob(f".{dataset_id}.*.tmp"))

    conflicting_prefix = {**prefix, "dataset_name": "different-valid-owner"}
    before_payload = payload_path.read_bytes()
    before_manifest = Path(recovered_corrupt[1]).read_bytes()
    with pytest.raises(FileExistsError, match="conflicts"):
        publish_package(
            dataset_id=dataset_id,
            payload_filename=payload_filename,
            manifest_prefix=conflicting_prefix,
            build_payload=build_payload,
            storage_root=tmp_path,
        )
    assert payload_path.read_bytes() == before_payload
    assert Path(recovered_corrupt[1]).read_bytes() == before_manifest
    assert build_count == 4


def _dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required test HDF5 dataset."""
    value = handle.get(name)
    assert isinstance(value, h5py.Dataset)
    return value


def _assert_provenance_and_relocation_preserve_hdf5(
    *,
    config_path: Path,
    template: Path,
    original_batch: Any,
    original_hdf5: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate the same science after locator, evidence, and Git changes."""
    project_root = template.parents[1]
    relocated_template = project_root / "relocated/steady_reference.mph"
    relocated_template.parent.mkdir(parents=True)
    relocated_template.write_bytes(template.read_bytes())
    relocated_template.with_suffix(".sha256").write_text(
        f"{common.serialization.file_sha256(relocated_template)}\n",
        encoding="utf-8",
    )
    profile_path = config_path.parent / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["template"] = relocated_template.relative_to(project_root).as_posix()
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    operations["parameter_values"]["pressure_bc.mean"]["provenance"]["note"] = "Changed evidence description only."
    operations_path.write_text(
        yaml.safe_dump(operations, sort_keys=False),
        encoding="utf-8",
    )

    relocated_campaign = generation.cases.config.load_campaign_config(config_path)
    relocated_batch = relocated_campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    assert relocated_batch.scientific_values != original_batch.scientific_values
    assert relocated_batch.batch_id == original_batch.batch_id
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "b" * 40)
    relocated_storage = tmp_path / "relocated-storage"
    try:
        generation.cases.input_generation.generate_input_cases(
            relocated_batch,
            1,
            storage_root=relocated_storage,
        )
        relocated_outcome = generation.runtime.run_case(
            relocated_batch,
            1,
            cores_per_case=1,
            storage_root=relocated_storage,
            work_root=tmp_path / "relocated-work",
        )
    finally:
        monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    relocated_hdf5 = relocated_outcome.processed_directory / "case.h5"
    original_identity = generation.publication.storage.validate_case_hdf5(
        original_hdf5,
        expected_profile="steady_flow",
    )
    relocated_identity = generation.publication.storage.validate_case_hdf5(
        relocated_hdf5,
        expected_profile="steady_flow",
    )
    assert relocated_identity["case_input_id"] == original_identity["case_input_id"]
    assert relocated_identity["simulation_case_id"] == original_identity["simulation_case_id"]
    assert relocated_identity["git_commit"] is None
    with h5py.File(original_hdf5, "r") as original, h5py.File(relocated_hdf5, "r") as relocated:
        for dataset_name in (
            "coords/x",
            "coords/y",
            "static/fields",
            "stationary_fixed/values",
        ):
            np.testing.assert_array_equal(original[dataset_name], relocated[dataset_name])


def _assert_terminal_purpose_corruption_rejected(*, batch: Any, storage: Path) -> None:
    """Reject a terminal manifest whose purpose contradicts persisted science."""
    metadata = common.paths.resolve_generation_batch_metadata_directory(
        batch.batch_storage_name,
        storage_root=storage,
    )
    manifest_path = metadata / "batch_manifest.json"
    success_path = metadata / "_SUCCESS"
    original_manifest = manifest_path.read_bytes()
    original_success = success_path.read_bytes()
    false_manifest = json.loads(original_manifest)
    false_manifest["campaign_purpose"] = "family_generalization"
    common.serialization.atomic_write_json(manifest_path, false_manifest)
    false_success = json.loads(original_success)
    false_success["manifest_sha256"] = common.serialization.file_sha256(manifest_path)
    common.serialization.atomic_write_json(success_path, false_success)
    try:
        with pytest.raises(RuntimeError, match="purpose, storage locator"):
            generation.runtime.admit_terminal_batch(
                batch.batch_storage_name,
                storage_root=storage,
            )
    finally:
        manifest_path.write_bytes(original_manifest)
        success_path.write_bytes(original_success)


def test_compact_case_builds_and_loads_package_without_direct_exports(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Build and load a normal package from only canonical compact HDF5."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=1,
        retain_raw_csv=False,
        retain_solved_model=False,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    storage = tmp_path / "compact-storage"
    generation.cases.input_generation.generate_input_cases(
        batch,
        1,
        storage_root=storage,
    )
    outcome = generation.runtime.run_case(
        batch,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "compact-work",
    )
    assert not (outcome.processed_directory / "comsol_exports").exists()
    assert not (outcome.processed_directory / "solved.mph").exists()
    assert not tuple(outcome.processed_directory.rglob("*.csv"))

    generation.runtime.finalize_batch(batch, storage_root=storage)
    result = datasets.packages.build_dataset_package(
        campaign,
        "steady_flow",
        "id",
        storage_root=storage,
    )
    loaded = datasets.runtime.factory.create_dataset(
        datasets.runtime.factory.DatasetRequest(
            dataset_id=result["dataset_id"],
            dataset_view="steady_flow",
            evaluation_regime="id",
            storage_root=storage,
            allow_technical_smoke=True,
        )
    )

    loaded_steady: Any = loaded
    assert len(loaded_steady) == 1
    sample = loaded_steady[0]
    task = domain.tasks.registry.get_task("steady_flow")
    assert set(sample) == {"meta", "x", "y"}
    assert sample["x"].shape[0] == task.in_channels
    assert sample["y"].shape[0] == task.out_channels


def test_transient_case_publishes_distinct_dual_view_dataset_ids(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind both advertised views to one canonical HDF5 without copying it."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        natural_count=1,
        retain_raw_csv=False,
        retain_solved_model=False,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(material_family="lentil", sampling_regime="natural")
    storage = tmp_path / "dual-view-storage"
    generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage)
    outcome = generation.runtime.run_case(
        batch,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / "dual-view-work",
    )
    generation.runtime.finalize_batch(batch, storage_root=storage)
    admitted = generation.runtime.admit_terminal_batch(batch.batch_storage_name, storage_root=storage)
    interpreted = datasets.packages.generated_batch.interpret_generated_transient_case(
        admitted,
        admitted.case("case_0001"),
        task=domain.tasks.registry.get_task("transient_drying"),
    )
    assert interpreted["runtime"]["stationary_airflow_solver_seconds"] == 3.5
    assert interpreted["runtime"]["transient_drying_solver_seconds"] == 7.25
    assert interpreted["runtime"]["scientific_solver_seconds"] == 10.75
    assert interpreted["runtime"]["comsol_solver_timing"]["status"] == "complete"

    transient = datasets.packages.build_dataset_package(
        campaign,
        "transient_drying",
        "id",
        storage_root=storage,
    )
    steady = datasets.packages.build_dataset_package(
        campaign,
        "steady_flow",
        "id",
        storage_root=storage,
    )

    assert transient["dataset_id"] != steady["dataset_id"]
    assert Path(transient["manifest_path"]).parent != Path(steady["manifest_path"]).parent
    transient_manifest = datasets.packages.load_package_manifest(transient["dataset_id"], storage_root=storage)
    steady_manifest = datasets.packages.load_package_manifest(steady["dataset_id"], storage_root=storage)
    transient_source = transient_manifest["source_case_identities"][0]
    steady_source = steady_manifest["source_case_identities"][0]
    composite_keys = {
        "composite_source_kind",
        "source_run_id",
        "source_git_commit",
        "source_campaign_manifest_sha256",
        "completion_receipt_sha256",
    }
    assert composite_keys.isdisjoint(transient_source)
    assert composite_keys.isdisjoint(steady_source)
    assert transient_source["source_relative_path"] == steady_source["source_relative_path"]
    assert transient_source["case_hdf5_sha256"] == steady_source["case_hdf5_sha256"]
    assert storage / transient_source["source_relative_path"] == outcome.processed_directory / "case.h5"

    def reject_transient_rebuild(*_args: Any, **_kwargs: Any) -> None:
        message = "Immutable transient package reuse rebuilt its HDF5 index."
        raise AssertionError(message)

    monkeypatch.setattr(
        package_builder.trajectory,
        "build_transient_index",
        reject_transient_rebuild,
    )
    reused = datasets.packages.build_dataset_package(
        campaign,
        "transient_drying",
        "id",
        storage_root=storage,
    )
    assert reused["status"] == "reused"
    assert reused["sample_count"] == transient["sample_count"]
    assert reused["transition_count"] == transient["transition_count"]

    package_root = common.paths.get_dataset_packages_root(storage_root=storage)
    assert not tuple(package_root.rglob("case.h5"))


def test_steady_flow_publishes_hdf5_and_immutable_technical_package(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect steady HDF5 publication and immutable technical-package reuse."""
    config_path, template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=1,
        retain_raw_csv=True,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    profile_contract = generation.contracts.get_profile_contract(batch.profile.id)
    grid = batch.scientific_values["grid"]
    fixed_values = batch.scientific_values["scientific_fixed_values"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    storage = tmp_path / "storage"
    template_bytes = template.read_bytes()
    generation.cases.input_generation.generate_input_cases(
        batch,
        len(batch.case_indices),
        storage_root=storage,
    )
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
    assert [outcome.status for outcome in outcomes] == ["completed"]
    assert template.read_bytes() == template_bytes

    completed = outcomes[0].processed_directory
    completed_files = {path.name for path in completed.iterdir()}
    assert {
        "_SUCCESS",
        "case.h5",
        "comsol_exports",
        "execution_provenance.json",
        "provenance.json",
        "status.json",
    }.issubset(completed_files)
    assert "case.json" not in completed_files
    raw = generation.runtime.raw_case_directory(batch, 1, storage_root=storage)
    assert {entry.name for entry in raw.iterdir()} == {"case.json", "inputs"}
    assert (raw / "inputs/fields.csv").is_file()
    assert (completed / "comsol_exports/airflow.csv").is_file()
    assert not (raw / "_SUCCESS").exists()
    identity = generation.publication.storage.validate_case_hdf5(completed / "case.h5", expected_profile="steady_flow")
    assert identity["git_commit"] is None
    with h5py.File(completed / "case.h5", "r") as handle:
        assert _dataset(handle, "coords/x").shape == (nx,)
        assert _dataset(handle, "coords/x").attrs["unit"] == "m"
        assert _dataset(handle, "coords/y").shape == (ny,)
        assert _dataset(handle, "coords/y").attrs["unit"] == "m"
        static = _dataset(handle, "static/fields")
        assert static.shape == (len(profile_contract.static_fields), ny, nx)
        assert static.dtype.name == "float32"
        fixed = _dataset(handle, "stationary_fixed/values")
        assert fixed.shape == (len(profile_contract.stationary_fixed_fields),)
        assert fixed.attrs["runtime_source"] == "canonical_template"
    case = json.loads((raw / "case.json").read_text(encoding="utf-8"))
    assert identity["case_input_id"] == case["case_input_id"]
    assert identity["simulation_case_id"] == case["simulation_case_id"]

    _assert_provenance_and_relocation_preserve_hdf5(
        config_path=config_path,
        template=template,
        original_batch=batch,
        original_hdf5=completed / "case.h5",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
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
    assert [record["case_id"] for record in terminal["cases"]] == ["case_0001"]
    missing_project = tmp_path / "project without templates"
    missing_project.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(missing_project))
    assert not (missing_project / batch.template_relative_path).exists()
    admitted = generation.runtime.admit_terminal_batch(batch.batch_storage_name, storage_root=storage)
    assert admitted.case("case_0001").hdf5_identity.git_commit is None
    unexpected = admitted.processed_directory / "unexpected_case"
    unexpected.mkdir()
    with pytest.raises(RuntimeError, match="membership mismatch"):
        generation.runtime.admit_terminal_batch(batch.batch_storage_name, storage_root=storage)
    unexpected.rmdir()

    _assert_terminal_purpose_corruption_rejected(batch=batch, storage=storage)

    result = datasets.packages.build_dataset_package(campaign, "steady_flow", "id", storage_root=storage)
    assert result["dataset_name"] == "steady_flow__lentil__id"
    assert result["sample_count"] == 1
    payload = torch.load(result["payload_path"], map_location="cpu", weights_only=False)
    task = domain.tasks.registry.get_task("steady_flow")
    dataset_identity = datasets.contracts.identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    assert dataset_identity.sample_count == 1
    assert payload["inputs"].shape == (1, task.in_channels, ny, nx)
    assert payload["outputs"].shape == (1, task.out_channels, ny, nx)
    assert payload["fields"] == {"inputs": list(task.input_names), "outputs": list(task.output_names)}
    manifest = datasets.packages.load_package_manifest(
        result["dataset_id"],
        storage_root=storage,
    )
    assert manifest["training_eligible"] is False
    assert manifest["builder_identity"] == datasets.contracts.identity.dataset_conversion_contract_identity("steady_flow")
    assert manifest["split_membership"] == {datasets.contracts.views.TECHNICAL_SMOKE_MEMBERSHIP: manifest["included_source_cases"]}
    assert manifest["source_git_commits"] == ["a" * 40]
    conditioning = manifest["steady_flow_conditioning"]
    assert conditioning["hidden_conditioning"] is False
    assert conditioning["T_flow_ref_owner"] == "package_fixed"
    assert conditioning["package_fixed_physics"] == [
        {
            "name": field.name,
            "unit": field.unit,
            "value": fixed_values[field.name],
        }
        for field in profile_contract.stationary_fixed_fields
    ]

    inspection = datasets.packages.inspect_dataset_package(result["dataset_id"], storage_root=storage)
    assert inspection["dataset_view"] == "steady_flow"
    assert inspection["available_selectors"] == [datasets.contracts.views.TECHNICAL_SMOKE_MEMBERSHIP]
    assert inspection["tensors"]["input"]["shape"] == [task.in_channels, ny, nx]
    assert inspection["tensors"]["target"]["shape"] == [task.out_channels, ny, nx]
    assert inspection["sample_identity"]["source_hdf5_sha256"] == manifest["source_case_identities"][0]["case_hdf5_sha256"]
    smoke = datasets.packages.smoke_dataset_package(
        result["dataset_id"],
        storage_root=storage,
        num_workers=0,
    )
    assert smoke["status"] == "loaded"
    assert smoke["batch_shapes"]["x"] == [1, task.in_channels, ny, nx]

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
