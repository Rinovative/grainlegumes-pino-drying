# ruff: noqa: S101, PLR2004, SLF001
"""Physical-unit transient one-hour index and lazy-loader contracts."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
import torch

from src import common, datasets, generation

if TYPE_CHECKING:
    from pathlib import Path


def _json(value: object) -> str:
    """Return compact HDF5 JSON metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required synthetic HDF5 dataset."""
    value = handle.get(name)
    assert isinstance(value, h5py.Dataset)
    return value


def _compact_steady_view(
    candidate: dict[str, Any],
    *,
    case_id: str,
    task: Any,
) -> tuple[Any, ...]:
    """Interpret one compact coupled HDF5 source through the steady transform."""
    with h5py.File(candidate["case_hdf5"], "r") as handle:
        x_axis = np.asarray(_hdf5_dataset(handle, "coords/x"), dtype=np.float64)
        y_axis = np.asarray(_hdf5_dataset(handle, "coords/y"), dtype=np.float64)
        static_dataset = _hdf5_dataset(handle, "static/fields")
        field_names = json.loads(str(static_dataset.attrs["field_names"]))
        static_values = np.asarray(static_dataset, dtype=np.float32)
    fields = {name: static_values[index] for index, name in enumerate(field_names)}
    case_inputs, case_outputs = datasets.generated_batch._steady_flow_fields(
        fields,
        x_axis=x_axis,
        y_axis=y_axis,
        task=task,
    )
    inputs = torch.stack([torch.from_numpy(case_inputs[name]) for name in task.input_names])
    outputs = torch.stack([torch.from_numpy(case_outputs[name]) for name in task.output_names])
    metadata = {
        "case_id": case_id,
        "case_input_id": candidate["case_input_id"],
        "simulation_case_id": candidate["simulation_case_id"],
        "material_family": candidate["material_family"],
        "simulation_profile": candidate["simulation_profile"],
    }
    source = {
        "case_id": case_id,
        "case_input_id": candidate["case_input_id"],
        "simulation_case_id": candidate["simulation_case_id"],
        "simulation_profile": candidate["simulation_profile"],
        "case_hdf5": {
            "sha256": candidate["case_hdf5_sha256"],
            "size_bytes": candidate["case_hdf5"].stat().st_size,
        },
    }
    fingerprint = datasets.identity.compute_case_fingerprint(
        task=task,
        case_id=case_id,
        source_identity=source,
        source_metadata=metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return (y_axis.size, x_axis.size), inputs, outputs, metadata, source, fingerprint


def _write_transient_case(
    path: Path,
    *,
    times: np.ndarray | None = None,
    case_input_id: str = "1" * 64,
    simulation_case_id: str = "2" * 64,
) -> np.ndarray:
    """Write one compact canonical case with an optional diagnostic stop state."""
    x_axis = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    y_axis = np.asarray([0.0, 1.0], dtype=np.float64)
    shape = (y_axis.size, x_axis.size)
    static_constants = (1.0e-10, 0.0, 2.0e-10, 0.4, 100.0, 0.2, 0.1, 0.05, 101325.0, 550.0)
    assert len(static_constants) == len(generation.profiles.STATIC_FIELD_NAMES)
    static = np.stack([np.full(shape, value, dtype=np.float32) for value in static_constants])
    state_times = np.asarray([0.0, 1.0, 2.0, 2.5], dtype=np.float64) if times is None else times
    base = np.asarray([295.0, 0.4, 10.0, 20.0], dtype=np.float32)
    increments = np.asarray([1.0, 0.01, 0.5, 0.25], dtype=np.float32)
    transient = np.stack(
        [
            np.stack([np.full(shape, base[channel] + step * increments[channel], dtype=np.float32) for channel in range(4)])
            for step in range(state_times.size)
        ]
    )
    schedule = np.zeros((169, len(generation.profiles.SCHEDULE_FIELDS)), dtype=np.float64)
    schedule[:, 0] = np.arange(169, dtype=np.float64)
    schedule[:, 1] = 295.0 + 0.1 * schedule[:, 0]
    schedule[:, 2] = 0.009
    schedule[:, 3] = 0.5 - 0.001 * schedule[:, 0]
    scalars = np.arange(1, len(generation.profiles.SCALAR_INPUT_FIELDS) + 1, dtype=np.float64)
    global_values = np.zeros((state_times.size, len(generation.profiles.GLOBAL_FIELD_NAMES)), dtype=np.float64)
    global_values[:, 0] = state_times

    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_kind": generation.storage.HDF5_SCHEMA_KIND,
                "schema_version": generation.storage.HDF5_SCHEMA_VERSION,
                "converter_version": "vp2_hdf5_v2",
                "simulation_profile": "transient_drying",
                "material_family": "lentil",
                "sampling_regime": "natural",
                "case_input_id": case_input_id,
                "simulation_case_id": simulation_case_id,
                "scientific_config_digest": "3" * 64,
                "export_contract_sha256": "4" * 64,
                "airflow_source": "comsol_coupled_reference",
                "git_commit": "a" * 40,
                "template_sha256": "5" * 64,
                "available_learning_views": _json(["steady_flow", "transient_drying"]),
                "source_export_hashes": _json({}),
            }
        )
        coordinates = handle.create_group("coords")
        x_dataset = coordinates.create_dataset("x", data=x_axis)
        y_dataset = coordinates.create_dataset("y", data=y_axis)
        x_dataset.attrs["unit"] = "m"
        y_dataset.attrs["unit"] = "m"
        static_dataset = handle.create_group("static").create_dataset(
            "fields",
            data=static,
            chunks=(1, *shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        static_dataset.attrs["field_names"] = _json(list(generation.profiles.STATIC_FIELD_NAMES))
        static_dataset.attrs["units"] = _json(list(generation.profiles.STATIC_FIELD_UNITS))
        scalar_dataset = handle.create_group("scalar").create_dataset("values", data=scalars)
        scalar_dataset.attrs["field_names"] = _json(list(generation.profiles.SCALAR_INPUT_FIELDS))
        scalar_dataset.attrs["units"] = _json(list(generation.profiles.SCALAR_INPUT_UNITS))
        time_dataset = handle.create_dataset("time", data=state_times)
        time_dataset.attrs["unit"] = "h"
        transient_dataset = handle.create_group("transient").create_dataset(
            "fields",
            data=transient,
            chunks=(1, 1, *shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        transient_dataset.attrs["field_names"] = _json(list(generation.profiles.TRANSIENT_FIELD_NAMES))
        transient_dataset.attrs["units"] = _json(list(generation.profiles.TRANSIENT_FIELD_UNITS))
        schedule_dataset = handle.create_group("schedule").create_dataset("values", data=schedule, compression="gzip")
        schedule_dataset.attrs["field_names"] = _json(list(generation.profiles.SCHEDULE_FIELDS))
        schedule_dataset.attrs["units"] = _json(list(generation.profiles.SCHEDULE_UNITS))
        global_dataset = handle.create_group("global").create_dataset("values", data=global_values, compression="gzip")
        global_dataset.attrs["field_names"] = _json(list(generation.profiles.GLOBAL_FIELD_NAMES))
        global_dataset.attrs["units"] = _json(list(generation.profiles.GLOBAL_FIELD_UNITS))
    return increments


def _source(path: Path, *, regime: str = "id", membership: str = "train") -> datasets.transient.TransientSourceCase:
    """Return one typed source bound to the synthetic HDF5 identities."""
    return datasets.transient.TransientSourceCase(
        path=path,
        package_case_id="synthetic_transient__case_0001",
        source_batch_id="synthetic_transient_batch",
        membership=membership,
        evaluation_regime=regime,
        expected_sha256=common.serialization.file_sha256(path),
        expected_case_input_id="1" * 64,
        expected_simulation_case_id="2" * 64,
        material_family="lentil",
        ood_group=None if regime == "id" else "bed",
        ood_parameters=() if regime == "id" else ("kappa_mean",),
        ood_evidence={},
    )


def _build_index(source: Path, index_path: Path, *, regime: str = "id", membership: str = "train") -> dict[str, Any]:
    """Build one compact synthetic transition index."""
    return datasets.transient.build_transient_index(
        [_source(source, regime=regime, membership=membership)],
        index_path,
        dataset_name=f"transient_drying__lentil__{regime}",
        dataset_id=f"transient_drying__lentil__{regime}__synthetic",
        evaluation_regime=regime,
        contract_digest=datasets.views.get_view("transient_drying").contract_digest,
        source_root=source.parent,
    )


def test_transient_index_excludes_irregular_stop_and_derives_increments(tmp_path: Path) -> None:
    """Protect exact regular transitions, typed physical tensors, and portable identity."""
    source = tmp_path / "case.h5"
    increments = _write_transient_case(source)
    generation.storage.validate_case_hdf5(source, expected_profile="transient_drying")
    index_path = tmp_path / "index.json"
    payload = _build_index(source, index_path)

    assert payload["sample_count"] == 2
    assert [(sample["t_n"], sample["t_np1"]) for sample in payload["samples"]] == [(0.0, 1.0), (1.0, 2.0)]
    assert payload["cases"][0]["sequence_length"] == 3
    assert payload["cases"][0]["stored_state_count"] == 4
    assert payload["cases"][0]["irregular_stop_time"] == 2.5
    assert payload["cases"][0]["transition_count"] == 2
    assert payload["contract"]["dt"] == {"value": 1.0, "unit": "h"}

    dataset = datasets.transient.TransientPhysicalDataset(index_path, source_root=tmp_path, hdf5_cache_size=1)
    assert len(dataset) == 2
    first = dataset[0]
    assert first["state"].shape == (4, 2, 3)
    assert first["static"].shape == (7, 2, 3)
    assert first["boundary"].shape == (5,)
    assert first["scalars"].shape == (8,)
    assert first["target"].shape == (4, 2, 3)
    assert first["dt"].shape == ()
    torch.testing.assert_close(
        first["target"],
        torch.from_numpy(np.broadcast_to(increments[:, None, None], (4, 2, 3)).copy()),
    )
    assert first["metadata"]["t_n"] == 0.0
    assert first["metadata"]["t_np1"] == 1.0
    assert first["metadata"]["split"] == "train"
    assert first["metadata"]["material_family"] == "lentil"
    assert len(dataset._handles) == 1

    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    relocated = relocated_root / "copy.h5"
    shutil.copyfile(source, relocated)
    relocated_payload = _build_index(relocated, relocated_root / "index.json")
    assert relocated_payload["dataset_id"] == payload["dataset_id"]
    assert relocated_payload["index_digest"] == payload["index_digest"]


def test_transient_loader_is_worker_safe_and_rejects_source_mutation(tmp_path: Path) -> None:
    """Protect zero/multi-worker collation, bounded handles, and source integrity."""
    source = tmp_path / "case.h5"
    _write_transient_case(source)
    index_path = tmp_path / "index.json"
    _build_index(source, index_path)

    zero_worker_dataset = datasets.transient.TransientPhysicalDataset(index_path, source_root=tmp_path)
    zero_loader = datasets.factory.make_data_loader(
        zero_worker_dataset,
        datasets.factory.LoaderSettings(batch_size=2),
    )
    zero_batch = next(iter(zero_loader))
    assert zero_batch["state"].shape == (2, 4, 2, 3)
    assert zero_batch["metadata"]["sample_id"] == [
        "synthetic_transient__case_0001__step_0000",
        "synthetic_transient__case_0001__step_0001",
    ]

    worker_dataset = datasets.transient.TransientPhysicalDataset(index_path, source_root=tmp_path, hdf5_cache_size=1)
    _ = worker_dataset[0]
    worker_loader = datasets.factory.make_data_loader(
        worker_dataset,
        datasets.factory.LoaderSettings(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=1,
            hdf5_cache_size=1,
        ),
    )
    worker_batch = next(iter(worker_loader))
    assert worker_batch["target"].shape == (1, 4, 2, 3)
    del worker_loader

    with source.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="changed after dataset admission"):
        _ = worker_dataset[0]


def test_transient_package_factory_selectors_inspection_and_worker_smoke(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise compact package publication, selectors, inspection, and both worker modes."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
    )
    campaign = generation.config.load_campaign_config(config_path)
    batch = campaign.batch("transient_drying__lentil__natural")
    storage_root = tmp_path / "storage"
    source_root = storage_root / "synthetic_sources"
    source_root.mkdir(parents=True)
    package_case_ids: list[str] = []
    candidates: list[dict[str, Any]] = []
    memberships = ("train", "validation", "id_test")
    for index, membership in enumerate(memberships, start=1):
        case_input_id = f"{index:x}" * 64
        simulation_case_id = f"{index + 8:x}" * 64
        source = source_root / f"case_{index:04d}.h5"
        _write_transient_case(
            source,
            case_input_id=case_input_id,
            simulation_case_id=simulation_case_id,
        )
        case_id = f"case_{index:04d}"
        package_case_id = f"{batch.batch_name}__{case_id}"
        package_case_ids.append(package_case_id)
        record = {
            "case_id": case_id,
            "case_input_id": case_input_id,
            "simulation_case_id": simulation_case_id,
            "case_hdf5_sha256": common.serialization.file_sha256(source),
            "success_sha256": f"{index + 3:x}" * 64,
            "provenance_sha256": f"{index + 6:x}" * 64,
        }
        candidates.append(
            {
                "batch_id": batch.batch_id,
                "case_id": case_id,
                "package_case_id": package_case_id,
                "case_hdf5": source,
                "case_hdf5_relative": source.relative_to(storage_root).as_posix(),
                "case_hdf5_sha256": common.serialization.file_sha256(source),
                "case_input_id": case_input_id,
                "simulation_case_id": simulation_case_id,
                "material_family": "lentil",
                "simulation_profile": "transient_drying",
                "dataset_membership": membership,
                "task_relevant_ood_parameters": [],
                "ood_evidence": {},
                "manifest": {},
                "record": record,
            }
        )
    batch_record = {
        "batch_name": batch.batch_name,
        "batch_id": batch.batch_id,
        "batch_identity": batch.batch_identity,
        "manifest_sha256": "6" * 64,
        "simulation_profile": batch.profile.id,
        "template": {
            "relative_path": batch.profile.template_relative_path,
            "sha256": batch.template_sha256,
        },
        "scientific_config_digest": batch.scientific_config_digest,
        "git_commit": "a" * 40,
        "material_config_digest": batch.scientific_values["material_config_digest"],
        "operation_config_digest": batch.scientific_values["operation_config_digest"],
        "airflow_source": batch.profile.airflow_source,
        "available_learning_views": list(batch.profile.available_learning_views),
        "export_contract_sha256": common.serialization.canonical_json_sha256(batch.scientific_values["output_contract"]),
        "steady_flow_conditioning": batch.scientific_values["steady_flow_conditioning"],
    }
    plan = next(
        dict(package)
        for package in campaign.dataset_packages
        if package["dataset_view"] == "transient_drying" and package["evaluation_regime"] == "id"
    )
    prepared = datasets.packages._PreparedPackage(
        plan=plan,
        batch_records=[batch_record],
        candidates=candidates,
        excluded=[],
        membership={membership: [package_case_id] for membership, package_case_id in zip(memberships, package_case_ids, strict=True)},
        source_decisions=[],
        steady_conditioning=None,
    )
    result = datasets.packages._publish_prepared(
        campaign,
        prepared,
        storage_root=storage_root,
    )

    manifest = datasets.packages.load_package_manifest(
        result["dataset_id"],
        storage_root=storage_root,
    )
    assert manifest["sample_count"] == 6
    assert manifest["source_case_count"] == 3
    assert manifest["transition_count"] == 6

    candidate_by_case = {str(candidate["case_id"]): candidate for candidate in candidates}

    def interpret_coupled_case(
        batch_id: str,
        case_id: str,
        *,
        task: Any,
        manifest: Any,
        record: Any,
        storage_root: Any,
    ) -> tuple[Any, ...]:
        """Interpret the compact canonical coupled case through the steady transform."""
        assert batch_id == batch.batch_id
        assert manifest == {}
        assert storage_root == storage_root_path
        candidate = candidate_by_case[case_id]
        assert record == candidate["record"]
        return _compact_steady_view(candidate, case_id=case_id, task=task)

    storage_root_path = storage_root
    monkeypatch.setattr(
        datasets.packages.generated,
        "interpret_generated_case",
        interpret_coupled_case,
    )
    steady_plan = next(
        dict(package) for package in campaign.dataset_packages if package["dataset_view"] == "steady_flow" and package["evaluation_regime"] == "id"
    )
    steady_prepared = datasets.packages._PreparedPackage(
        plan=steady_plan,
        batch_records=[batch_record],
        candidates=[dict(candidate) for candidate in candidates],
        excluded=[],
        membership={membership: [package_case_id] for membership, package_case_id in zip(memberships, package_case_ids, strict=True)},
        source_decisions=[],
        steady_conditioning=datasets.packages.audit_steady_flow_conditioning([batch_record]),
    )
    steady_result = datasets.packages._publish_prepared(
        campaign,
        steady_prepared,
        storage_root=storage_root,
    )
    steady_manifest = datasets.packages.load_package_manifest(
        steady_result["dataset_id"],
        storage_root=storage_root,
    )
    steady_payload = torch.load(steady_result["payload_path"], map_location="cpu", weights_only=False)
    assert steady_payload["inputs"].shape == (3, 7, 2, 3)
    assert steady_payload["outputs"].shape == (3, 3, 2, 3)
    assert steady_manifest["source_simulation_profiles"] == ["transient_drying"]
    assert steady_manifest["airflow_provenance"] == ["comsol_coupled_reference"]
    assert steady_manifest["case_membership"] == manifest["case_membership"]

    request = datasets.factory.DatasetRequest(
        dataset_id=result["dataset_id"],
        dataset_view="transient_drying",
        evaluation_regime="id",
        membership="validation",
        storage_root=storage_root,
    )
    validation = datasets.factory.create_dataset(request, hdf5_cache_size=1)
    assert isinstance(validation, datasets.transient.TransientPhysicalDataset)
    assert len(validation) == 2
    assert {validation[index]["metadata"]["split"] for index in range(len(validation))} == {"validation"}
    validation.close()

    inspection = datasets.packages.inspect_dataset_package(
        result["dataset_id"],
        storage_root=storage_root,
    )
    assert inspection["available_selectors"] == [
        "id/train",
        "id/validation",
        "id/id_test",
    ]
    assert inspection["tensors"]["state"]["shape"] == [4, 2, 3]
    assert inspection["tensors"]["static"]["shape"] == [7, 2, 3]
    assert inspection["sample_identity"]["sequence_length"] == 3

    zero_worker = datasets.packages.smoke_dataset_package(
        result["dataset_id"],
        storage_root=storage_root,
        membership="train",
        num_workers=0,
    )
    multi_worker = datasets.packages.smoke_dataset_package(
        result["dataset_id"],
        storage_root=storage_root,
        membership="train",
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=1,
    )
    assert zero_worker["status"] == multi_worker["status"] == "loaded"
    assert zero_worker["batch_shapes"]["target"] == [1, 4, 2, 3]
    assert multi_worker["batch_shapes"]["target"] == [1, 4, 2, 3]

    ood_source = source_root / "case_ood.h5"
    _write_transient_case(
        ood_source,
        case_input_id="d" * 64,
        simulation_case_id="e" * 64,
    )
    ood_case_id = f"{batch.batch_name}__case_ood"
    ood_evidence = {
        "group": "bed",
        "selected_units": ["kappa_mean"],
        "units_per_case": 1,
        "parameters": [{"name": "kappa_mean", "block": "airflow"}],
    }
    ood_plan = next(
        dict(package)
        for package in campaign.dataset_packages
        if package["dataset_view"] == "transient_drying" and package["evaluation_regime"] == "parameter_ood"
    )
    ood_prepared = datasets.packages._PreparedPackage(
        plan=ood_plan,
        batch_records=[batch_record],
        candidates=[
            {
                "batch_id": batch.batch_id,
                "case_id": "case_ood",
                "package_case_id": ood_case_id,
                "case_hdf5": ood_source,
                "case_hdf5_relative": ood_source.relative_to(storage_root).as_posix(),
                "case_hdf5_sha256": common.serialization.file_sha256(ood_source),
                "case_input_id": "d" * 64,
                "simulation_case_id": "e" * 64,
                "material_family": "lentil",
                "simulation_profile": "transient_drying",
                "dataset_membership": "parameter_ood",
                "task_relevant_ood_parameters": ["kappa_mean"],
                "ood_evidence": ood_evidence,
            }
        ],
        excluded=[],
        membership={"parameter_ood": [ood_case_id]},
        source_decisions=[],
        steady_conditioning=None,
    )
    ood_result = datasets.packages._publish_prepared(
        campaign,
        ood_prepared,
        storage_root=storage_root,
    )
    all_request = datasets.factory.DatasetRequest(
        dataset_id=ood_result["dataset_id"],
        dataset_view="transient_drying",
        evaluation_regime="parameter_ood",
        storage_root=storage_root,
    )
    bed_request = datasets.factory.DatasetRequest(
        dataset_id=ood_result["dataset_id"],
        dataset_view="transient_drying",
        evaluation_regime="parameter_ood",
        ood_group="bed",
        storage_root=storage_root,
    )
    all_dataset = datasets.factory.create_dataset(all_request)
    bed_dataset = datasets.factory.create_dataset(bed_request)
    assert isinstance(all_dataset, datasets.transient.TransientPhysicalDataset)
    assert isinstance(bed_dataset, datasets.transient.TransientPhysicalDataset)
    assert len(all_dataset) == 2
    assert len(bed_dataset) == 2
    all_dataset.close()
    bed_dataset.close()
    unavailable = datasets.factory.DatasetRequest(
        dataset_id=ood_result["dataset_id"],
        dataset_view="transient_drying",
        evaluation_regime="parameter_ood",
        ood_group="operation",
        storage_root=storage_root,
    )
    with pytest.raises(ValueError, match="unavailable"):
        datasets.factory.create_dataset(unavailable)

    invalid_membership: Any = "not_a_membership"
    with pytest.raises(ValueError, match="Unsupported ID membership selector"):
        datasets.factory.DatasetRequest(
            dataset_id=result["dataset_id"],
            dataset_view="transient_drying",
            evaluation_regime="id",
            membership=invalid_membership,
            storage_root=storage_root,
        )
    with pytest.raises(ValueError, match="require num_workers > 0"):
        datasets.factory.LoaderSettings(
            num_workers=0,
            persistent_workers=True,
        )


def test_transient_index_rejects_nonfinal_irregular_state_and_bad_commit(tmp_path: Path) -> None:
    """Protect regular-prefix time admission and exact commit validation."""
    irregular = tmp_path / "irregular.h5"
    _write_transient_case(irregular, times=np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
    with pytest.raises(datasets.transient.TransientDataContractError, match=r"regular 0\.\.N"):
        _build_index(irregular, tmp_path / "irregular.json")

    bad = tmp_path / "bad-commit.h5"
    _write_transient_case(bad)
    with h5py.File(bad, "r+") as handle:
        handle.attrs["git_commit"] = "not-a-commit"
    with pytest.raises(ValueError, match="40-character"):
        generation.storage.validate_case_hdf5(bad, expected_profile="transient_drying")
