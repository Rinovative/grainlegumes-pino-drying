# ruff: noqa: S101, PLR2004
"""Physical-unit transient one-hour index and loader contracts."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
import torch

from src import datasets, generation

if TYPE_CHECKING:
    from pathlib import Path


def _json(value: object) -> str:
    """Return compact HDF5 JSON metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_transient_case(path: Path) -> np.ndarray:
    """Write one compact canonical case with a diagnostic irregular stop state."""
    x_axis = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    y_axis = np.asarray([0.0, 1.0], dtype=np.float64)
    shape = (y_axis.size, x_axis.size)
    static = np.stack([np.full(shape, index + 1.0, dtype=np.float32) for index in range(len(generation.profiles.STATIC_FIELD_NAMES))])
    times = np.asarray([0.0, 1.0, 2.0, 2.5], dtype=np.float64)
    base = np.asarray([295.0, 0.4, 10.0, 20.0], dtype=np.float32)
    increments = np.asarray([1.0, 0.01, 0.5, 0.25], dtype=np.float32)
    transient = np.stack(
        [
            np.stack([np.full(shape, base[channel] + step * increments[channel], dtype=np.float32) for channel in range(4)])
            for step in range(times.size)
        ]
    )
    schedule = np.zeros((169, len(generation.profiles.SCHEDULE_FIELDS)), dtype=np.float64)
    schedule[:, 0] = np.arange(169, dtype=np.float64)
    schedule[:, 1] = 295.0 + 0.1 * schedule[:, 0]
    schedule[:, 2] = 0.009
    schedule[:, 3] = 0.5 - 0.001 * schedule[:, 0]
    scalars = np.arange(1, len(generation.profiles.SCALAR_INPUT_FIELDS) + 1, dtype=np.float64)
    global_values = np.zeros((times.size, len(generation.profiles.GLOBAL_FIELD_NAMES)), dtype=np.float64)
    global_values[:, 0] = times

    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_kind": generation.storage.HDF5_SCHEMA_KIND,
                "schema_version": generation.storage.HDF5_SCHEMA_VERSION,
                "converter_version": "vp2_hdf5_v2",
                "simulation_profile": "transient_drying",
                "material_family": "lentil",
                "sampling_regime": "natural",
                "case_input_id": "1" * 64,
                "simulation_case_id": "2" * 64,
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
        time_dataset = handle.create_dataset("time", data=times)
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


def test_transient_index_excludes_irregular_stop_and_derives_increments(tmp_path: Path) -> None:
    """Protect one-hour pair construction, physical tensors, units, and identity."""
    source = tmp_path / "case.h5"
    increments = _write_transient_case(source)
    generation.storage.validate_case_hdf5(source, expected_profile="transient_drying")
    index_path = tmp_path / "index.json"
    payload = datasets.transient.build_transient_index(
        [(source, "id")],
        index_path,
        dataset_name="transient_drying__lentil__id",
        source_provenance={"campaign": "synthetic"},
    )
    assert payload["sample_count"] == 2
    assert [sample["time"] for sample in payload["samples"]] == [0.0, 1.0]
    assert payload["cases"][0]["sequence_length"] == 4
    assert payload["cases"][0]["transition_count"] == 2
    assert payload["contract"]["time_step"] == 1.0
    assert payload["contract"]["time_unit"] == "h"

    dataset = datasets.transient.TransientPhysicalDataset(index_path)
    assert len(dataset) == 2
    first = dataset[0]
    assert first["dynamic_state"].shape == (4, 2, 3)
    assert first["static_spatial_conditioning"].shape == (7, 2, 3)
    assert first["step_boundary_conditioning"].shape == (5,)
    assert first["scalar_conditioning"].shape == (8,)
    assert first["target_increments"].shape == (4, 2, 3)
    torch.testing.assert_close(
        first["target_increments"],
        torch.from_numpy(np.broadcast_to(increments[:, None, None], (4, 2, 3)).copy()),
    )
    assert first["units"]["dynamic_state"] == ["K", "1", "kg/m^3", "kg/m^3"]
    assert first["units"]["target_increments"] == ["K", "1", "kg/m^3", "kg/m^3"]
    assert first["meta"]["dt"] == 1.0
    assert first["meta"]["time_unit"] == "h"
    assert first["meta"]["material_family"] == "lentil"

    relocated = tmp_path / "relocated.h5"
    shutil.copyfile(source, relocated)
    relocated_payload = datasets.transient.build_transient_index(
        [(relocated, "id")],
        tmp_path / "relocated-index.json",
        dataset_name="transient_drying__lentil__id",
        source_provenance={"campaign": "synthetic"},
    )
    assert relocated_payload["dataset_id"] == payload["dataset_id"]
    assert relocated_payload["dataset_digest"] == payload["dataset_digest"]


def test_transient_loader_rejects_changed_source_and_bad_commit(tmp_path: Path) -> None:
    """Protect source integrity and exact commit admission at the HDF5 boundary."""
    source = tmp_path / "case.h5"
    _write_transient_case(source)
    index_path = tmp_path / "index.json"
    datasets.transient.build_transient_index(
        [(source, "parameter_ood")],
        index_path,
        dataset_name="transient_drying__lentil__parameter_ood",
    )
    dataset = datasets.transient.TransientPhysicalDataset(index_path)
    with source.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="changed after indexing"):
        _ = dataset[0]

    bad = tmp_path / "bad-commit.h5"
    _write_transient_case(bad)
    with h5py.File(bad, "r+") as handle:
        handle.attrs["git_commit"] = "not-a-commit"
    with pytest.raises(ValueError, match="40-character"):
        generation.storage.validate_case_hdf5(bad, expected_profile="transient_drying")
